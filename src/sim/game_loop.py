"""
M6: Integration game loop.

Wires all layers together in the correct data-flow order:

  WorldState
    → project()          [Projection Layer]
    → adapter.infer()    [LM Adapter]
    → compile_action()   [Compiler / Validator]
    → apply_transitions() + apply_event_to_graph()
    → WorldState (updated)

Exposes:
  GameLoop.step()   — process one player turn
  GameLoop.replay() — replay from initial state + event log (core invariant test)
"""

from __future__ import annotations

import copy
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .npc_policy import NpcPolicy

from .compiler import apply_transitions, build_belief_transitions, compile_action
from .grounding import classify
from .lm_adapter import LMAdapter, MockLMAdapter, get_adapter
from .needs import VERB_SATISFY, satisfy_need
from .skills import maybe_gain_skill
from .narrator import NarratorActor
from .projection import assert_projection_invariants, project
from .relational import apply_event_to_graph
from .schemas import (
    ActionType,
    ActionZone,
    Coord,
    EntityId,
    EntityState,
    Event,
    GroundingResult,
    MalformedActionError,
    ProjectionFirewallError,
    RejectionReason,
    SemanticAction,
    SemanticProjection,
    Transition,
    TransitionProposal,
    ValidationResult,
    WorldState,
    new_event_id,
)

# Stubs that the grounding rescue path should NOT treat as synthesized verbs.
_TRIVIAL_WAIT_STRS = {"wait", "w", "pass", "skip", "do nothing", "", "interact"}

logger = logging.getLogger(__name__)


def _log_system_transitions(
    world: WorldState,
    transitions: list[Transition],
    *,
    raw_input: str,
    narrative_hint: str = "",
) -> None:
    """Apply and record deterministic system transitions in the event log."""
    if not transitions:
        return
    apply_transitions(world, transitions)
    world.event_log.append(
        Event(
            event_id=new_event_id(),
            tick=world.tick,
            action=SemanticAction(
                verb=ActionType.SYMBOLIC,
                actor=EntityId("system"),
                raw_input=raw_input,
            ),
            transitions=transitions,
            witnesses=[],
            narrative_hint=narrative_hint or raw_input,
        )
    )


def _run_propagation_passes(
    world: WorldState,
    tick_transitions: list[Transition],
    lm_adapter=None,
) -> list[Transition]:
    """
    Run all world-propagation passes for one tick and return the resulting
    secondary transitions.

    Passes run in this order:
      0. process_world_memory    — durable facts → NPC beliefs + observation memory
      1. causal_consequence_pass — second-order physical / social ripples
      2. synthesize_npc_goals    — emergent goal formation from events
      3. propagate_beliefs       — belief / rumour diffusion through social graph
      4. generate_dynamic_quests — periodic quest synthesis from world state
      5. schedule_delays_from_facts — queue delayed narration / NPC goals

    Each pass failure is logged at WARNING (not silently swallowed) so
    regressions surface in logs rather than disappearing.

    All secondary transitions are recorded as a first-class system Event in
    world.event_log so they appear in replays, projection windows, and the
    autonomous run log — making propagation auditable.
    """
    from .world_propagation import (
        causal_consequence_pass,
        enrich_goal_transitions,
        generate_dynamic_quests,
        propagate_beliefs,
        synthesize_goals_from_world_facts,
        synthesize_npc_goals,
    )
    secondary: list[Transition] = []
    try:
        from .world_memory import process_world_memory

        secondary.extend(process_world_memory(world))
    except Exception as exc:
        logger.warning("process_world_memory failed (tick=%d): %s", world.tick, exc)
    try:
        secondary.extend(causal_consequence_pass(world, tick_transitions))
    except Exception as exc:
        logger.warning("causal_consequence_pass failed (tick=%d): %s", world.tick, exc)
    try:
        goal_transitions = synthesize_npc_goals(world, tick_transitions)
        goal_transitions.extend(synthesize_goals_from_world_facts(world))
        enrich_goal_transitions(world, goal_transitions, lm_adapter)
        secondary.extend(goal_transitions)
    except Exception as exc:
        logger.warning("synthesize_npc_goals failed (tick=%d): %s", world.tick, exc)
    try:
        from .economy import record_trade_from_transitions

        record_trade_from_transitions(world, tick_transitions)
    except Exception as exc:
        logger.warning("record_trade_from_transitions failed (tick=%d): %s", world.tick, exc)
    belief_interval = max(1, int(world.config.semantic.belief_propagation_interval))
    if belief_interval <= 1 or world.tick % belief_interval == 0:
        try:
            secondary.extend(propagate_beliefs(world))
        except Exception as exc:
            logger.warning("propagate_beliefs failed (tick=%d): %s", world.tick, exc)
    # Dynamic quest generation every 7 ticks.
    if world.tick % 7 == 0:
        try:
            secondary.extend(generate_dynamic_quests(world))
        except Exception as exc:
            logger.warning("generate_dynamic_quests failed (tick=%d): %s", world.tick, exc)

    delayed_lines: list[str] = []
    try:
        from .delayed_consequences import schedule_delays_from_facts

        delayed_lines, delayed_transitions = schedule_delays_from_facts(world)
        secondary.extend(delayed_transitions)
    except Exception as exc:
        logger.warning("schedule_delays_from_facts failed (tick=%d): %s", world.tick, exc)
    if delayed_lines:
        world.meta.setdefault("delayed_log", []).extend(delayed_lines[-4:])

    # Record propagation as a first-class system Event so it is visible in
    # replays and NPC projection windows.  The caller still calls
    # apply_transitions(); this event is for auditability only.
    if secondary:
        prop_event = Event(
            event_id=new_event_id(),
            tick=world.tick,
            action=SemanticAction(
                verb=ActionType.SYMBOLIC,
                actor=EntityId("system"),
                raw_input="[propagation]",
            ),
            transitions=secondary,
            witnesses=[],
            narrative_hint="[system: propagation]",
        )
        world.event_log.append(prop_event)
        from .relational import apply_event_to_graph

        apply_event_to_graph(world.relational, prop_event, world.tick)

    return secondary


def _find_player_spawn(world: WorldState) -> Coord:
    """Pick a walkable, unoccupied tile so movement buttons are not instantly stuck."""
    grid = world.spatial
    occupied = {
        (e.position.x, e.position.y, e.position.z)
        for e in grid.entities.values()
        if e.alive
    }
    best: Optional[tuple[int, Coord]] = None
    z_range = range(1, grid.depth) if grid.depth > 1 else range(0, 1)
    for z in z_range:
        for x in range(grid.width):
            for y in range(grid.height):
                c = Coord(x=x, y=y, z=z)
                if grid.is_solid(c) or (x, y, z) in occupied:
                    continue
                open_n = 0
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    n = Coord(x=x + dx, y=y + dy, z=z)
                    if grid.is_in_bounds(n) and not grid.is_solid(n):
                        open_n += 1
                if open_n < 2:
                    continue
                score = open_n
                if grid.use_voxels and z > 0:
                    below = Coord(x=x, y=y, z=z - 1)
                    if grid.is_solid(below):
                        score += 2
                if best is None or score > best[0]:
                    best = (score, c)
    if best is not None:
        return best[1]
    return Coord(x=grid.width // 2, y=grid.height // 2, z=0)


def _spawn_interactive_player(world: WorldState) -> EntityId:
    """
    Packs like tavern have no player entity; place the player on a walkable tile.
    """
    from .schemas import EntityKind, EntityState, new_entity_id

    player_id = new_entity_id()
    spawn = _find_player_spawn(world)
    world.spatial.entities[player_id] = EntityState(
        entity_id=player_id,
        name="You",
        kind=EntityKind.PLAYER,
        position=spawn,
        sight_range=12,
    )
    return player_id


def _run_semantic_dynamics(
    world: WorldState,
    tick: int,
    *,
    ingest_all_new: bool = False,
    lm_adapter=None,
):
    """Delayed semantic interpretation pass (see semantic_pressure.py)."""
    from .semantic_pressure import on_tick_end

    return on_tick_end(
        world, tick, ingest_all_new=ingest_all_new, lm_adapter=lm_adapter
    )


# Intent pre-parsers live in intent_router.py (shared with play_client).
from .intent_router import (
    parse_player_intent,
    try_parse_player_fast_path,
    try_parse_compass_step as _try_parse_compass_step,
    try_parse_trivial_actions as _try_parse_trivial_actions,
)

_SOCIAL_VERBS: frozenset[str] = frozenset({
    ActionType.SPEAK, ActionType.ASK, ActionType.PERSUADE, ActionType.INTIMIDATE,
    ActionType.DECEIVE, "kiss", "hug", "embrace", "caress", "touch",
    "shake_hands", "bow", "greet", "flirt", "tease", "seduce",
    "compliment", "insult", "threaten", "beg", "plead",
})


def _is_social_action(action: SemanticAction) -> bool:
    """
    Return True if this action is social/physical-contact and directed at an
    entity, so D1 (NPC reply) and D5 (consequence proposals) should fire.
    """
    if action.target is None or not isinstance(action.target, str):
        return False
    verb = action.verb.lower().strip()
    return verb in _SOCIAL_VERBS or action.verb in _SOCIAL_VERBS


def _extract_utterance(action: SemanticAction) -> str:
    """
    Extract the most meaningful utterance string from a SemanticAction for the
    NPC reply prompt.  Prefers intent.manner (spoken words), then rationale.
    """
    if action.intent:
        manner = (action.intent.manner or "").strip()
        if manner and manner not in ("in character", "reactive", "conversational"):
            return manner
        rationale = (action.intent.rationale or "").strip()
        if rationale and len(rationale) > 4:
            return rationale
    verb_phrase = str(action.verb).replace("_", " ")
    return f"{verb_phrase} (from: {action.raw_input})" if action.raw_input else verb_phrase


_D5_SKIP_VERBS: frozenset = frozenset({
    ActionType.WAIT, ActionType.OBSERVE, ActionType.TURN,
    ActionType.LOOK, ActionType.FACE, ActionType.SYMBOLIC,
})


# ---------------------------------------------------------------------------
# Player policy abstraction
# ---------------------------------------------------------------------------


from abc import ABC, abstractmethod as _abstractmethod


class PlayerPolicy(ABC):
    """
    Abstract policy for deciding the player entity's action each turn.

    Separating the "decide what to do" step from the compile / apply / NPC
    pipeline makes the player entity swappable for research and headless use:

      HumanInputPolicy  — pre-parsers + LM-backed inference (default REPL mode)
      LMPlayerPolicy    — drives the player exactly like an NPC (autonomous /
                          benchmark mode — no raw-text pre-processing)

    Custom policies (e.g. MCTS planners, scripted agents) can also be plugged
    in by subclassing PlayerPolicy and passing an instance to GameLoop.
    """

    @_abstractmethod
    def decide(
        self,
        world: "WorldState",
        player_id: "EntityId",
        intent: str,
        proj: "SemanticProjection",
    ) -> "SemanticAction":
        """
        Produce the player's next SemanticAction.

        Parameters
        ----------
        world      : Current canonical world state (read-only; do not mutate).
        player_id  : EntityId of the player entity.
        intent     : Raw natural-language intent string from the REPL / caller.
                     Ignored by LMPlayerPolicy (which uses projection instead).
        proj       : Pre-computed SemanticProjection for the player.

        Returns
        -------
        SemanticAction — passed straight to the compiler.

        Raises
        ------
        MalformedActionError if the LM or adapter cannot produce a valid action.
        """


class HumanInputPolicy(PlayerPolicy):
    """
    Standard interactive-player policy.

    Processing order:
      1. Deterministic pre-parsers handle unambiguous structural commands
         (spell programs, item manipulation, orientation, verbatim speech,
         directional movement with explicit distance).  These bypass the LM
         entirely and return a SemanticAction directly.
      2. Any remaining intent goes to adapter.infer() — the LM layer.

    This is the default policy wired in by GameLoop when no explicit
    player_policy is provided.
    """

    def __init__(self, adapter: "LMAdapter") -> None:
        self.adapter = adapter

    def decide(
        self,
        world: "WorldState",
        player_id: "EntityId",
        intent: str,
        proj: "SemanticProjection",
    ) -> "SemanticAction":
        player_entity = world.spatial.entities.get(player_id)

        _pre = parse_player_intent(intent, player_id, world, player_entity)
        if _pre is not None:
            return _pre

        # LM inference
        return self.adapter.infer(proj, intent)


class LMPlayerPolicy(PlayerPolicy):
    """
    Drives the player entity via an NpcPolicy (LM-backed or reactive).

    Use for fully headless / benchmark runs where the player is also
    model-controlled.  The raw ``intent`` string is ignored; the policy
    decides based on canonical world state and the player's projection,
    exactly like any NPC.

    Example
    -------
    from src.sim.npc_lm_policy import LMNpcPolicy
    from src.sim.npc_policy import ReactivePolicy
    policy = LMPlayerPolicy(LMNpcPolicy(adapter, fallback=ReactivePolicy()))
    loop = GameLoop(world, adapter, player_policy=policy)
    for _ in range(100):
        loop.autonomous_tick()   # player acts autonomously like every other entity
    """

    def __init__(self, npc_policy: "NpcPolicy") -> None:
        self.npc_policy = npc_policy

    def decide(
        self,
        world: "WorldState",
        player_id: "EntityId",
        intent: str,
        proj: "SemanticProjection",
    ) -> "SemanticAction":
        entity = world.spatial.entities.get(player_id)
        if entity is None:
            return SemanticAction(verb=ActionType.WAIT, actor=player_id,
                                  raw_input="[lm_player: entity_missing]")
        return self.npc_policy.decide(entity, world)


@dataclass
class AutonomousTickResult:
    """Result of one autonomous (headless) tick."""
    tick: int
    ambient_events: list[Event] = field(default_factory=list)
    entity_results: list["StepResult"] = field(default_factory=list)
    completed_goals: list = field(default_factory=list)
    physics_lines: list[str] = field(default_factory=list)
    semantic_lines: list[str] = field(default_factory=list)
    pressure_lines: list[str] = field(default_factory=list)
    scenario_lines: list[str] = field(default_factory=list)
    campaign_lines: list[str] = field(default_factory=list)
    quest_journal: list = field(default_factory=list)


@dataclass
class StepResult:
    """Result of a single game loop step."""

    tick: int
    projection: SemanticProjection
    action: SemanticAction
    grounding: GroundingResult
    validation: ValidationResult
    event: Optional[Event]
    # Human-readable status for REPL / debug output
    status: str
    # Narrator output — Zone 3: primary response; Zone 1/2: optional flavor
    narration: Optional[str] = None
    # NPC reactions that ran after the player's turn (empty if none acted)
    npc_results: list["StepResult"] = field(default_factory=list)
    # Ambient events fired by the WorldClock before NPCs took their turns.
    # Surfaced so the REPL can narrate them in chronological order.
    ambient_events: list[Event] = field(default_factory=list)
    # Goals that completed this turn (populated by evaluate_goals()).
    completed_goals: list = field(default_factory=list)
    # Who noticed the player's action (speech, violence, …) before NPC turns.
    stimulus_feedback: list[str] = field(default_factory=list)
    pressure_lines: list[str] = field(default_factory=list)
    campaign_lines: list[str] = field(default_factory=list)
    quest_journal: list = field(default_factory=list)
    scenario_lines: list[str] = field(default_factory=list)
    tick_frame: Optional[Any] = None


class GameLoop:
    """
    Authoritative tick-based game loop.

    Parameters
    ----------
    world        : The canonical WorldState.  The loop mutates this in place.
    adapter      : LM adapter to use.  Defaults to MockLMAdapter.
    player_id    : EntityId of the player-controlled entity.
    validate_projection : If True, assert_projection_invariants() is called
                          after each projection, raising on violation.
    """

    def __init__(
        self,
        world: WorldState,
        adapter: Optional[LMAdapter] = None,
        player_id: Optional[EntityId] = None,
        *,
        validate_projection: bool = True,
        use_real_lm: bool = False,
        npc_policy: Optional["NpcPolicy"] = None,
        npcs_act_each_turn: bool = True,
        player_policy: Optional[PlayerPolicy] = None,
        debug_mode: bool = False,
        emit_presentation: bool = True,
    ):
        from .npc_lm_policy import LMNpcPolicy
        from .npc_policy import NpcPolicy as _NpcPolicy, ReactivePolicy
        self.world = world
        self.adapter = adapter or get_adapter(use_real_lm=use_real_lm)
        self.validate_projection = validate_projection
        self.narrator = NarratorActor(adapter=self.adapter)
        # Pick the default NPC policy. Three cases:
        #   1. Caller passed npc_policy → use it (test setups, custom rigs).
        #   2. Adapter is MockLMAdapter → ReactivePolicy (deterministic,
        #      preserves the historical test behavior).
        #   3. Adapter is a real LM → LMNpcPolicy wrapped around the same
        #      adapter, with ReactivePolicy as the fallback when the LM
        #      can't parse / raises.
        # This keeps the test suite hermetic while letting the REPL
        # automatically get LM-driven NPCs as soon as a real backend is
        # wired in.
        if npc_policy is None:
            cfg = world.config.npc_policy
            reactive = ReactivePolicy(
                threat_lookback_ticks=cfg.threat_lookback_ticks,
                flee_health_fraction=cfg.flee_health_fraction,
            )
            if isinstance(self.adapter, MockLMAdapter):
                npc_policy = reactive
            else:
                npc_policy = LMNpcPolicy(self.adapter, fallback=reactive)
        self.npc_policy: _NpcPolicy = npc_policy
        self.npcs_act_each_turn = npcs_act_each_turn

        # Real LM backends enable LM semantic pressure (MockLMAdapter only).
        if world.config.semantic.enabled and type(self.adapter) is not MockLMAdapter:
            world.config.semantic.use_mock = False

        # Player policy — defaults to HumanInputPolicy (pre-parsers + LM).
        # Pass LMPlayerPolicy for headless / research runs.
        self.player_policy: PlayerPolicy = (
            player_policy if player_policy is not None
            else HumanInputPolicy(self.adapter)
        )

        # Infer player_id as the first PLAYER-kind entity if not supplied
        if player_id is None:
            from .schemas import EntityKind
            for eid, entity in world.spatial.entities.items():
                if entity.kind == EntityKind.PLAYER:
                    player_id = eid
                    break
        if player_id is None:
            player_id = _spawn_interactive_player(world)
        self.player_id = player_id
        self.trace_recorder: Optional[Any] = None
        self.debug_mode = debug_mode
        if debug_mode:
            from .sim_debug import set_debug_mode
            set_debug_mode(world, True)

        from .lm_budget import wrap_adapter_metrics

        wrap_adapter_metrics(self.adapter, world)

        self.emit_presentation = emit_presentation
        self.presentation = None
        if emit_presentation:
            from .presentation_emitter import PresentationCollector

            self.presentation = PresentationCollector(world.name)

    def _emit_tick_frame(self, result: "StepResult") -> None:
        if not self.emit_presentation or self.presentation is None:
            return
        from .lm_budget import metrics_snapshot
        from .presentation_emitter import build_tick_frame

        frame = build_tick_frame(result, self.world, player_id=self.player_id)
        frame.lm_metrics = metrics_snapshot(self.world)
        self.presentation.push(frame)
        result.tick_frame = frame

    def _emit_autonomous_frame(self, result: "AutonomousTickResult") -> None:
        if not self.emit_presentation or self.presentation is None:
            return
        from .lm_budget import metrics_snapshot
        from .presentation_emitter import build_tick_frame_autonomous

        frame = build_tick_frame_autonomous(result, self.world)
        frame.lm_metrics = metrics_snapshot(self.world)
        self.presentation.push(frame)

    def step(self, player_intent: str) -> StepResult:
        """
        Process a single player turn.

        1. Project world state for the player entity.
        2. LM infers a SemanticAction from the projection.
        3. Compiler validates and compiles the action.
        4. Apply transitions to canonical state.
        5. Advance tick.

        Returns StepResult with full trace for testing and rendering.
        """
        if self.player_id is None:
            raise RuntimeError("No player entity found in world.")

        stimulus_feedback: list[str] = []

        # --- 0. Scenario director (same tick phase as autonomous mode) ---
        from .director import NarrativeDirector
        from .lm_budget import tick_lm_budget

        scenario_lines = NarrativeDirector(self.world).tick_start(self.world)
        tick_lm_budget(self.world)

        # --- 1. Projection ---
        # Pass the last rejection (if any) so the LM can learn from it.
        _player_entity = self.world.spatial.entities.get(self.player_id)
        _last_rejection = (
            _player_entity.meta.get("last_rejection")
            if _player_entity else None
        )
        proj = project(self.world, self.player_id, last_action_rejection=_last_rejection)
        if self.validate_projection:
            assert_projection_invariants(proj, self.world, self.player_id)

        # --- 2. Fast-path commands (spells, items) — skip LM, single pipeline entry ---
        _fast = try_parse_player_fast_path(
            player_intent, self.player_id, self.world, _player_entity,
        )
        if _fast is not None:
            action, spell_narrative = _fast.action, _fast.narrative_override
            from .schemas import GroundingResult
            grounding = GroundingResult(zone=ActionZone.GROUNDED)
            validation = compile_action(action, self.world)
            if validation.valid:
                apply_transitions(self.world, validation.concrete_transitions)
            _fast_event = Event(
                tick=self.world.tick,
                action=action,
                transitions=validation.concrete_transitions,
                witnesses=[self.player_id],
            )
            if validation.valid:
                self.world.event_log.append(_fast_event)
            self.world.tick += 1
            if spell_narrative:
                narrative = spell_narrative
            else:
                from .narrator import render_event as _render_event
                narrative = _render_event(_fast_event, self.world)
            return StepResult(
                tick=self.world.tick - 1,
                projection=proj,
                action=action,
                grounding=grounding,
                validation=validation,
                event=_fast_event if validation.valid else None,
                status="accepted" if validation.valid else "rejected",
                narration=narrative,
            )

        # --- 2. Player policy decide ---
        # Delegates to HumanInputPolicy (pre-parsers + LM) by default.
        # Swap in LMPlayerPolicy for headless / benchmark runs.
        try:
            action = self.player_policy.decide(
                self.world, self.player_id, player_intent, proj
            )
        except MalformedActionError as exc:
            logger.warning("Malformed action from player policy: %s", exc)
            from .schemas import GroundingResult
            fallback_grounding = GroundingResult(
                zone=ActionZone.ORPHANED,
                orphan_reason=f"Policy failed to parse intent: {exc}",
            )
            return StepResult(
                tick=self.world.tick,
                projection=proj,
                action=SemanticAction(
                    verb=ActionType.WAIT,
                    actor=self.player_id,
                    raw_input=player_intent,
                ),
                grounding=fallback_grounding,
                validation=ValidationResult(
                    valid=False,
                    rejection_reason=RejectionReason.MALFORMED_ACTION,
                    rejection_detail=str(exc),
                ),
                event=None,
                status=f"[MALFORMED] {exc}",
                narration=self.narrator.narrate_zone3(
                    player_intent,
                    f"Could not parse intent: {exc}",
                    proj,
                    self.world,
                ),
            )

        # --- 3. Grounding classification ---
        grounding = classify(action, self.world, player_intent)

        # --- 3b. Interaction rescue ---
        # When the LM returned WAIT but grounding rescued it (Zone 1 with a
        # synthesized verb stashed in orphan_reason), rewrite the action's verb
        # before handing it to the compiler.  This handles "shake hands with
        # guard" → verb="shake_hands" even on small models that default to WAIT.
        if (
            grounding.zone == ActionZone.GROUNDED
            and action.verb == ActionType.WAIT
            and grounding.orphan_reason
            and grounding.orphan_reason not in _TRIVIAL_WAIT_STRS
        ):
            action = action.model_copy(
                update={"verb": grounding.orphan_reason}
            )

        # --- 4. Compile ---
        try:
            result = compile_action(action, self.world, projection=proj)
        except ProjectionFirewallError as exc:
            logger.warning("Projection firewall triggered: %s", exc)
            result = ValidationResult(
                valid=False,
                rejection_reason=RejectionReason.ENTITY_NOT_VISIBLE,
                rejection_detail=str(exc),
            )

        # --- 4b. DM adjudication (Zone 3 / compile rejection) ---
        from .world_adjudicator import needs_adjudication, resolve_with_adjudication

        adjudication_ruling: Optional[str] = None
        adjudication_transitions: list[Transition] = []
        if needs_adjudication(grounding, result):
            (
                action,
                result,
                grounding,
                adjudication_transitions,
                adjudication_ruling,
                _adjudication_rescued,
            ) = resolve_with_adjudication(
                self.world,
                self.adapter,
                action,
                player_intent,
                proj,
                grounding,
                result,
            )

        # --- 5. Apply transitions ---
        narration: Optional[str] = None
        event: Optional[Event] = None
        stimulus_feedback: list[str] = []

        if result.valid:
            # --- Zone 1 / Zone 2 (or adjudication verb rescue) ---
            if grounding.zone == ActionZone.EXTENDING:
                belief_transitions = build_belief_transitions(
                    action, grounding, result, self.world
                )
                result.concrete_transitions.extend(belief_transitions)

            if adjudication_transitions:
                result.concrete_transitions.extend(adjudication_transitions)

            apply_transitions(self.world, result.concrete_transitions)

            # ── Skill gain on successful action ──────────────────────────
            actor_entity = self.world.spatial.entities.get(action.actor)
            if actor_entity:
                _verb_str = str(action.verb).lower().split(".")[-1]
                _skill_t = maybe_gain_skill(
                    actor_entity, _verb_str, success=True,
                    rng_seed=hash(f"{action.actor}:{self.world.tick}:{_verb_str}") & 0x7FFFFFFF,
                )
                if _skill_t:
                    apply_transitions(self.world, [_skill_t])
                    result.concrete_transitions.append(_skill_t)

            # ── Need satisfaction on successful action ────────────────────
            if actor_entity:
                _verb_key = str(action.verb).lower().split(".")[-1]
                for _need, _amount in VERB_SATISFY.get(_verb_key, {}).items():
                    _need_t = satisfy_need(str(action.actor), _need, _amount, cause=_verb_key)
                    apply_transitions(self.world, [_need_t])
                    result.concrete_transitions.append(_need_t)

            witnesses = _compute_witnesses(self.world, action.actor)
            event = Event(
                event_id=new_event_id(),
                tick=self.world.tick,
                action=action,
                transitions=result.concrete_transitions,
                witnesses=witnesses,
            )

            from .dialogue import update_threads
            update_threads(self.world, event)

            from .social_stimulus import broadcast_stimulus_from_event

            stimulus_feedback = broadcast_stimulus_from_event(self.world, event)
            from .contact_effects import apply_contact_social_effects

            apply_contact_social_effects(self.world, event)

            self._apply_post_action_lm_layer(action, event, proj)

            self.world.event_log.append(event)
            apply_event_to_graph(self.world.relational, event, self.world.tick)

            narration = self.narrator.narrate_outcome(
                event, grounding, proj, self.world
            )
            if adjudication_ruling:
                narration = (
                    f"{adjudication_ruling}\n{narration}"
                    if narration
                    else adjudication_ruling
                )
            if narration and event:
                event.narrative_hint = narration

        elif grounding.zone == ActionZone.ORPHANED or adjudication_transitions:
            transitions = list(adjudication_transitions)
            if not transitions:
                symbolic_result = compile_action(
                    SemanticAction(
                        verb=ActionType.SYMBOLIC,
                        actor=self.player_id,
                        raw_input=player_intent,
                    ),
                    self.world,
                )
                if symbolic_result.valid:
                    transitions = symbolic_result.concrete_transitions
                if not result.valid:
                    result = symbolic_result

            narration = adjudication_ruling or self.narrator.narrate_zone3(
                player_intent,
                grounding.orphan_reason or "No world affordance.",
                proj,
                self.world,
            )
            if transitions:
                apply_transitions(self.world, transitions)
                witnesses = _compute_witnesses(self.world, action.actor)
                event = Event(
                    event_id=new_event_id(),
                    tick=self.world.tick,
                    action=action,
                    transitions=transitions,
                    witnesses=witnesses,
                    narrative_hint=narration,
                )
                self._apply_post_action_lm_layer(
                    action,
                    event,
                    proj,
                    adjudicated=bool(adjudication_transitions),
                )
                self.world.event_log.append(event)
                apply_event_to_graph(self.world.relational, event, self.world.tick)

        # --- 7. Advance tick ---
        self.world.tick += 1

        if grounding.zone == ActionZone.ORPHANED and not result.valid:
            status = f"[ORPHANED] tick={self.world.tick-1} intent had no world anchor"
        elif adjudication_transitions and not result.valid:
            status = f"[ADJUDICATED] tick={self.world.tick-1} partial resolution"
        elif result.valid:
            zone_tag = "Z2" if grounding.zone == ActionZone.EXTENDING else "Z1"
            status = f"[OK:{zone_tag}] tick={self.world.tick-1} action={action.verb}"
        else:
            status = f"[REJECTED:{result.rejection_reason}] {result.rejection_detail or ''}"
        logger.info(status)

        # Store / clear rejection reason for the NEXT tick's projection.
        _pe = self.world.spatial.entities.get(self.player_id)
        if _pe is not None:
            if not result.valid and result.rejection_reason is not None:
                reason_str = result.rejection_reason.value if hasattr(result.rejection_reason, "value") else str(result.rejection_reason)
                _pe.meta["last_rejection"] = (
                    f"{reason_str}: {result.rejection_detail or ''}"
                ).rstrip(": ")
            else:
                _pe.meta.pop("last_rejection", None)

        # Tick down conditions for the player entity.
        _tick_down_conditions(self.world.spatial.entities.get(self.player_id))

        # --- 8. World clock — autonomic ambient events ---
        # Fires BEFORE the NPC turn so NPCs can react to anything the
        # clock added to the event log in the same tick (e.g., a guard
        # glances up when the bell tolls). Ambient events are not
        # actor-driven and never invoke the LM — they're scheduled
        # entirely from pack data.
        from .world_clock import world_tick as _world_tick
        ambient_events = _world_tick(self.world, lm_adapter=self.adapter) if self.npcs_act_each_turn else []

        # --- 9. NPC reactions ---
        npc_results: list[StepResult] = []
        if self.npcs_act_each_turn:
            npc_results = self.step_npcs()

        # --- 10. Episodic memory (no-op most ticks) ---
        from .episodic_memory import maybe_summarize
        maybe_summarize(self.world, self.adapter)

        # --- 10b. Physics tick ------------------------------------------
        # Run one physics/chemistry cycle over the active region after
        # every player turn.  Effects (fire spreading, temperature damage,
        # freezing, acid contact, etc.) are applied via apply_transitions()
        # using the same compiler path as all other state mutations.
        physics_lines: list[str] = []
        if self.world.config.physics_config is not None:
            from .physics import physics_tick, summarize_physics_transitions
            phys_transitions = physics_tick(self.world, self.world.config.physics_config)
            if phys_transitions:
                apply_transitions(self.world, phys_transitions)
                physics_lines = summarize_physics_transitions(
                    phys_transitions, self.world.spatial
                )

        if physics_lines:
            physics_narrative = "  ".join(physics_lines)
            narration = narration + "\n" + physics_narrative if narration else physics_narrative

        # --- 10c. Latent pressures (pack triggers) ---
        from .pressure_eval import evaluate_pressures

        pressure_lines, pressure_transitions = evaluate_pressures(self.world)
        _log_system_transitions(
            self.world,
            pressure_transitions,
            raw_input="[pressure]",
            narrative_hint="[system: pressure eval]",
        )

        # --- 11. Goal evaluation ---
        completed_goals = evaluate_goals(self.world)

        # --- 12. World propagation passes ---
        tick_transitions = (
            event.transitions
            if event is not None
            else (
                list(adjudication_transitions)
                if adjudication_transitions
                else (result.concrete_transitions if result.valid else [])
            )
        )
        prop_transitions = _run_propagation_passes(
            self.world, tick_transitions, lm_adapter=self.adapter,
        )
        if prop_transitions:
            apply_transitions(self.world, prop_transitions)

            # ── Same-tick ripple visibility ────────────────────────────────
            # Surface player-relevant second-order effects (nearby NPCs
            # alerting, mood shifts, new goals, fire spread, witness
            # edges) in the *same* narration beat as the player's action.
            # Without this, "I throw the lantern at the curtains" produces
            # prose that describes only the throw — the alarmed guards,
            # the spreading flames, and the new vengeance goals all show
            # up silently in NPC behaviour next tick, making the
            # simulation feel less reactive than it actually is.
            #
            # Mirrors the physics_lines append pattern above.
            from .world_propagation import summarize_player_visible_ripples

            ripple_lines = summarize_player_visible_ripples(
                self.world, self.player_id, prop_transitions,
            )
            if ripple_lines:
                ripple_text = "  ".join(ripple_lines)
                narration = (
                    f"{narration}\n{ripple_text}" if narration else ripple_text
                )

        from .campaign_director import run_campaign_director

        campaign_lines = run_campaign_director(self.world)

        from .quest_journal import build_quest_journal

        quest_journal = build_quest_journal(self.world, self.player_id)

        semantic_result = _run_semantic_dynamics(
            self.world, self.world.tick, ingest_all_new=True,
            lm_adapter=self.adapter,
        )

        step_result = StepResult(
            tick=self.world.tick - 1,
            projection=proj,
            action=action,
            grounding=grounding,
            validation=result,
            event=event,
            status=status,
            narration=narration,
            npc_results=npc_results,
            ambient_events=ambient_events,
            completed_goals=completed_goals,
            stimulus_feedback=stimulus_feedback if (result.valid or adjudication_transitions) else [],
            pressure_lines=pressure_lines,
            campaign_lines=campaign_lines,
            quest_journal=quest_journal,
            scenario_lines=scenario_lines,
        )
        self._emit_tick_frame(step_result)
        return step_result

    def _apply_post_action_lm_layer(
        self,
        action: SemanticAction,
        event: Event,
        proj: SemanticProjection,
        *,
        adjudicated: bool = False,
    ) -> None:
        """
        D1 (target reply hint) and D5 (consequence proposals) for any entity.

        Shared by the player step and autonomous ``_step_one_entity`` so NPC
        actions in headless runs get the same world-state mutations as the player.
        """
        from .npc_lm_policy import LMNpcPolicy
        from .npc_policy import build_npc_character_sheet

        infer_layer = (
            isinstance(self.npc_policy, LMNpcPolicy)
            and LMNpcPolicy.uses_infer_layer(self.world)
        )

        visible_ids = [str(e.entity_id) for e in proj.visible_entities]
        target_entity_for_d1 = None
        if _is_social_action(action):
            tid = action.target
            if isinstance(tid, str):
                from .region_utils import entity_or_none

                target_entity_for_d1 = entity_or_none(self.world, EntityId(tid))
                if target_entity_for_d1 is not None and not target_entity_for_d1.alive:
                    target_entity_for_d1 = None

        use_real_lm = type(self.adapter) is not MockLMAdapter
        should_propose_d5 = (
            use_real_lm
            and action.verb not in _D5_SKIP_VERBS
            and bool(event.transitions)
            and (
                infer_layer
                or adjudicated
                or str(action.actor) == str(self.player_id)
            )
        )

        consequence_proposals: list[TransitionProposal] = []
        if target_entity_for_d1 is not None or should_propose_d5:
            workers = (1 if target_entity_for_d1 is not None else 0) + (
                1 if should_propose_d5 else 0
            )
            with ThreadPoolExecutor(max_workers=max(workers, 1)) as ex:
                f_reply = None
                if target_entity_for_d1 is not None and infer_layer:
                    target_sheet = build_npc_character_sheet(
                        target_entity_for_d1, self.world,
                    )
                    utterance = _extract_utterance(action)
                    f_reply = ex.submit(
                        self.adapter.generate_npc_reply,
                        target_sheet,
                        utterance,
                        proj,
                    )
                f_cons = None
                if should_propose_d5:
                    f_cons = ex.submit(
                        self.adapter.propose_consequences,
                        action,
                        proj,
                        visible_ids,
                    )

                reply_text: Optional[str] = None
                if f_reply is not None:
                    try:
                        reply_text = f_reply.result()
                    except Exception as exc:
                        logger.warning("D1 generate_npc_reply raised: %s", exc)
                if f_cons is not None:
                    try:
                        consequence_proposals = f_cons.result()
                    except Exception as exc:
                        logger.warning("D5 propose_consequences raised: %s", exc)

            if reply_text and target_entity_for_d1 is not None:
                for t in event.transitions:
                    if t.kind.value == "dialogue_spoken":
                        t.payload["reply_text"] = reply_text
                        t.payload["reply_speaker"] = target_entity_for_d1.name
                        break
                else:
                    npc_line = f'{target_entity_for_d1.name}: "{reply_text}"'
                    if event.narrative_hint:
                        event.narrative_hint = f"{event.narrative_hint}\n{npc_line}"
                    else:
                        event.narrative_hint = npc_line

        if consequence_proposals or (
            action.verb not in _D5_SKIP_VERBS and event.transitions
        ):
            from .consequences import apply_post_action_consequences

            try:
                new_transitions = apply_post_action_consequences(
                    self.world,
                    action,
                    event.transitions,
                    proposals=consequence_proposals or None,
                    apply_ripples=(
                        action.verb not in _D5_SKIP_VERBS and bool(event.transitions)
                    ),
                )
                if new_transitions:
                    apply_transitions(self.world, new_transitions)
                    event.transitions.extend(new_transitions)
            except Exception as exc:
                logger.warning("post-action consequence pipeline failed: %s", exc)
                logger.debug("consequence traceback", exc_info=True)

    # ------------------------------------------------------------------
    # NPC tick — every NPC gets a SemanticAction from self.npc_policy,
    # which is then compiled through the same pipeline as the player.
    # ------------------------------------------------------------------

    def step_npcs(self) -> list[StepResult]:
        """
        Run one turn for each living NPC.

        LM inference (the decide phase) runs in parallel via a thread pool
        since it is pure I/O-bound network traffic that releases the GIL.
        The compile + apply phase remains sequential because it mutates
        WorldState — apply order matches entity iteration order for
        deterministic (if non-unique) outcomes.
        """
        from .schemas import EntityKind

        from .combat_tactics import sort_key_for_turn

        npc_entities = [
            (eid, entity)
            for eid, entity in self.world.all_entities().items()
            if entity.kind == EntityKind.NPC and entity.health > 0 and entity.alive
        ]
        if not npc_entities:
            return []

        npc_entities.sort(
            key=lambda pair: sort_key_for_turn(pair[0], pair[1], self.world)
        )

        decisions = self._parallel_decide(npc_entities)

        results: list[StepResult] = []
        for eid, entity in npc_entities:
            action = decisions.get(eid)
            if action is None:
                continue
            result = self._step_one_entity(eid, entity, action=action)
            if result is not None:
                results.append(result)
        return results

    def _parallel_decide(
        self,
        entities: list[tuple[EntityId, "EntityState"]],
    ) -> dict[EntityId, SemanticAction]:
        """
        Run npc_policy.decide() for all entities concurrently.

        Entities beyond the per-tick LM budget receive reactive fallback
        without an infer call.  Callers should pre-sort by stimulus_priority.
        """
        from .lm_budget import allocate_lm_slots
        from .npc_lm_policy import LMNpcPolicy
        from .npc_policy import ReactivePolicy

        uses_lm = (
            isinstance(self.npc_policy, LMNpcPolicy)
            and LMNpcPolicy.uses_lm_layer(self.world)
        )
        fallback = getattr(self.npc_policy, "fallback", None) or ReactivePolicy()
        lm_eids = allocate_lm_slots(self.world, entities) if uses_lm else set()

        def _reactive_fallback(entity: "EntityState", exc: Exception) -> SemanticAction:
            logger.warning(
                "Policy failed for %s: %s — using reactive fallback",
                entity.entity_id,
                exc,
            )
            action = fallback.decide(entity, self.world)
            entity.meta["last_policy_branch"] = "policy_exception_reactive"
            return action

        def _decide_one(eid: EntityId, entity: "EntityState") -> SemanticAction:
            if uses_lm and eid not in lm_eids:
                action = fallback.decide(entity, self.world)
                entity.meta["last_policy_branch"] = "budget_reactive"
                return action
            try:
                return self.npc_policy.decide(entity, self.world)
            except Exception as exc:
                return _reactive_fallback(entity, exc)

        if len(entities) <= 1:
            decisions: dict[EntityId, SemanticAction] = {}
            for eid, entity in entities:
                decisions[eid] = _decide_one(eid, entity)
            return decisions

        decisions = {}
        entity_by_id = dict(entities)
        with ThreadPoolExecutor(max_workers=len(entities)) as pool:
            future_to_eid = {
                pool.submit(_decide_one, eid, entity): eid
                for eid, entity in entities
            }
            for future in as_completed(future_to_eid):
                eid = future_to_eid[future]
                try:
                    decisions[eid] = future.result()
                except Exception as exc:
                    entity = entity_by_id[eid]
                    decisions[eid] = _reactive_fallback(entity, exc)
        return decisions

    # ------------------------------------------------------------------
    # Autonomous tick — headless mode, no player input
    # ------------------------------------------------------------------

    def autonomous_tick(self) -> "AutonomousTickResult":
        """
        Run one tick with no player input.

        Sequence:
          1. World clock fires ambient events.
          2. Every living entity (regardless of kind) gets one turn
             through self.npc_policy — player-kind entities are driven
             by the LM exactly like NPCs.
          3. Tick advances.

        Use this to run a fully self-contained simulation where the
        player is absent or is also an LM-controlled character.
        """
        from .world_clock import world_tick as _world_tick

        ambient_events = _world_tick(self.world, lm_adapter=self.adapter)

        from .director import NarrativeDirector
        from .lm_budget import lm_entity_priority

        scenario_lines = NarrativeDirector(self.world).tick_start(self.world)

        from .lm_budget import tick_lm_budget

        tick_lm_budget(self.world)

        living_entities = [
            (eid, entity)
            for eid, entity in self.world.all_entities().items()
            if entity.health > 0 and entity.alive
        ]
        living_entities.sort(
            key=lambda pair: (
                -lm_entity_priority(self.world, pair[1], entity_id=pair[0]),
                pair[1].name,
            )
        )

        decisions = self._parallel_decide(living_entities)

        entity_results: list[StepResult] = []
        for eid, entity in living_entities:
            action = decisions.get(eid)
            if action is None:
                action = self._fallback_reactive_action(entity)
            result = self._step_one_entity(eid, entity, action=action)
            if result is not None:
                entity_results.append(result)

        if living_entities and not entity_results:
            raise RuntimeError(
                "Autonomous tick produced no entity actions. "
                "Every living entity failed compile/apply even after reactive "
                "fallback; aborting instead of continuing an empty simulation."
            )

        tick_before = self.world.tick
        self.world.tick += 1

        # Episodic memory — summarize at episode boundary (no-op most ticks)
        from .episodic_memory import maybe_summarize
        maybe_summarize(self.world, self.adapter)

        # Physics tick (autonomous mode)
        physics_lines: list[str] = []
        if self.world.config.physics_config is not None:
            from .physics import physics_tick, summarize_physics_transitions
            phys_transitions = physics_tick(self.world, self.world.config.physics_config)
            if phys_transitions:
                apply_transitions(self.world, phys_transitions)
                physics_lines = summarize_physics_transitions(
                    phys_transitions, self.world.spatial
                )

        from .pressure_eval import evaluate_pressures

        pressure_lines, pressure_transitions = evaluate_pressures(self.world)
        _log_system_transitions(
            self.world,
            pressure_transitions,
            raw_input="[pressure]",
            narrative_hint="[system: pressure eval]",
        )

        # Goal evaluation
        completed_goals = evaluate_goals(self.world)

        # World propagation — belief spread + causal ripples across all NPC
        # transitions from this tick
        all_tick_transitions: list[Transition] = []
        for er in entity_results:
            if er.event is not None and er.event.transitions:
                all_tick_transitions.extend(er.event.transitions)
            elif er.validation and er.validation.concrete_transitions:
                all_tick_transitions.extend(er.validation.concrete_transitions)
        if all_tick_transitions or world.world_facts:
            prop_transitions = _run_propagation_passes(
                self.world, all_tick_transitions, lm_adapter=self.adapter,
            )
            if prop_transitions:
                apply_transitions(self.world, prop_transitions)

        from .campaign_director import run_campaign_director

        campaign_lines = run_campaign_director(self.world)

        from .quest_journal import build_quest_journal

        player_id = self.player_id or next(
            (eid for eid, e in self.world.spatial.entities.items()),
            None,
        )
        quest_journal = (
            build_quest_journal(self.world, player_id) if player_id else []
        )

        semantic_result = _run_semantic_dynamics(
            self.world, tick_before, lm_adapter=self.adapter,
        )

        auto_result = AutonomousTickResult(
            tick=tick_before,
            ambient_events=ambient_events,
            entity_results=entity_results,
            completed_goals=completed_goals,
            physics_lines=physics_lines,
            semantic_lines=semantic_result.log_lines,
            pressure_lines=pressure_lines,
            scenario_lines=scenario_lines,
            campaign_lines=campaign_lines,
            quest_journal=quest_journal,
        )
        self._emit_autonomous_frame(auto_result)
        return auto_result

    def _fallback_reactive_action(self, entity: EntityState) -> SemanticAction:
        """Last-resort reactive decide when parallel policy phase omits an entity."""
        from .npc_policy import ReactivePolicy

        fallback = getattr(self.npc_policy, "fallback", None)
        policy = fallback if fallback is not None else ReactivePolicy()
        try:
            return policy.decide(entity, self.world)
        except Exception as exc:
            logger.exception(
                "Reactive fallback failed for %s: %s", entity.entity_id, exc
            )
            return SemanticAction(
                verb=ActionType.OBSERVE,
                actor=entity.entity_id,
                raw_input="[fallback:observe]",
            )

    def _step_one_entity(
        self,
        eid: EntityId,
        entity,
        action: "Optional[SemanticAction]" = None,
    ) -> "Optional[StepResult]":
        """
        Compile and apply one entity's action, then run narrator enrichment.

        Parameters
        ----------
        action : Pre-decided SemanticAction from the parallel decide phase.
                 If None, the policy is called inline (sequential path).

        Returns StepResult, or None if the policy raised unrecoverably.
        """
        if action is None:
            from .npc_reflection import run_npc_reflection

            try:
                run_npc_reflection(entity, self.world, self.adapter)
                action = self.npc_policy.decide(entity, self.world)
            except Exception as exc:
                logger.warning(
                    "Policy failed for %s: %s — using reactive fallback",
                    eid,
                    exc,
                )
                action = self._fallback_reactive_action(entity)
                entity.meta["last_policy_branch"] = "policy_exception_reactive"

        validation = compile_action(action, self.world)
        event: Optional[Event] = None
        narration: Optional[str] = None

        grounding = classify(action, self.world, action.raw_input or str(action.verb))
        from .world_adjudicator import needs_adjudication, resolve_with_adjudication

        adjudication_ruling: Optional[str] = None
        adjudication_transitions: list[Transition] = []
        if needs_adjudication(grounding, validation):
            npc_proj = project(
                self.world,
                eid,
                last_action_rejection=entity.meta.get("last_rejection"),
            )
            if self.validate_projection:
                assert_projection_invariants(npc_proj, self.world, eid)
            (
                action,
                validation,
                grounding,
                adjudication_transitions,
                adjudication_ruling,
                _adjudication_rescued,
            ) = resolve_with_adjudication(
                self.world,
                self.adapter,
                action,
                action.raw_input or str(action.verb),
                npc_proj,
                grounding,
                validation,
            )

        if validation.valid:
            if grounding.zone == ActionZone.EXTENDING:
                belief_transitions = build_belief_transitions(
                    action, grounding, validation, self.world
                )
                validation.concrete_transitions.extend(belief_transitions)

            if adjudication_transitions:
                validation.concrete_transitions.extend(adjudication_transitions)

            apply_transitions(self.world, validation.concrete_transitions)

            # ── NPC skill gain on successful action ───────────────────────
            try:
                _npc_entity = self.world.grid_for_entity(eid).entities.get(eid)
            except KeyError:
                _npc_entity = self.world.spatial.entities.get(eid)
            if _npc_entity:
                _npc_verb = str(action.verb).lower().split(".")[-1]
                _npc_skill_t = maybe_gain_skill(
                    _npc_entity, _npc_verb, success=True,
                    rng_seed=hash(f"{eid}:{self.world.tick}:{_npc_verb}") & 0x7FFFFFFF,
                )
                if _npc_skill_t:
                    apply_transitions(self.world, [_npc_skill_t])
                    validation.concrete_transitions.append(_npc_skill_t)
                # ── NPC need satisfaction ─────────────────────────────────
                for _need, _amt in VERB_SATISFY.get(_npc_verb, {}).items():
                    _nt = satisfy_need(str(eid), _need, _amt, cause=_npc_verb)
                    apply_transitions(self.world, [_nt])
                    validation.concrete_transitions.append(_nt)

            witnesses = _compute_witnesses(self.world, eid)
            event = Event(
                event_id=new_event_id(),
                tick=self.world.tick,
                action=action,
                transitions=validation.concrete_transitions,
                witnesses=witnesses,
            )

            # Dialogue threading — track speaking verbs as conversation threads.
            from .dialogue import update_threads
            update_threads(self.world, event)

            from .social_stimulus import broadcast_stimulus_from_event

            broadcast_stimulus_from_event(self.world, event)
            from .contact_effects import apply_contact_social_effects

            apply_contact_social_effects(self.world, event)

            npc_proj = project(
                self.world,
                eid,
                last_action_rejection=entity.meta.get("last_rejection"),
            )
            if self.validate_projection:
                assert_projection_invariants(npc_proj, self.world, eid)

            if self.trace_recorder is not None:
                self.trace_recorder.record_entity_step(
                    tick=self.world.tick,
                    entity_id=str(eid),
                    entity_name=entity.name,
                    action=action,
                    validation=validation,
                    policy_branch=str(
                        entity.meta.get("last_policy_branch", "unknown")
                    ),
                    projection_tokens=npc_proj.estimated_tokens,
                )

            self._apply_post_action_lm_layer(action, event, npc_proj)

            self.world.event_log.append(event)
            apply_event_to_graph(self.world.relational, event, self.world.tick)

            # Advance any long-horizon plan this NPC is working on.
            # ``progress_plan`` is a no-op for entities without an
            # ``active_plan`` so this is safe and free for vanilla NPCs.
            try:
                from . import npc_planner

                npc_planner.progress_plan(
                    entity, self.world, validation.concrete_transitions,
                )
            except Exception as exc:
                logger.debug("npc_planner.progress_plan failed for %s: %s", eid, exc)

            # Narrator enrichment — same rules as for the player (mundane
            # verbs are skipped; mock adapter returns None).
            npc_grounding = GroundingResult(zone=ActionZone.GROUNDED)
            narration = self.narrator.narrate_outcome(
                event, npc_grounding, npc_proj, self.world
            )
            if narration:
                event.narrative_hint = narration
            if adjudication_ruling:
                event.narrative_hint = (
                    f"{adjudication_ruling}\n{event.narrative_hint}"
                    if event.narrative_hint
                    else adjudication_ruling
                )

            status = (
                f"[NPC:{entity.name}] tick={self.world.tick} "
                f"action={action.verb}"
            )
            # Clear any stored rejection on success.
            entity.meta.pop("last_rejection", None)
            # Track consecutive observe streak for escalation logic
            v = str(action.verb).lower()
            if v == "observe" or v == ActionType.OBSERVE:
                entity.meta["observe_streak"] = int(entity.meta.get("observe_streak", 0)) + 1
            else:
                entity.meta["observe_streak"] = 0
            entity.meta["last_action_verb"] = v
            from .npc_mind import record_expression

            record_expression(entity, self.world, action)
            _SPEECH_V = {
                "speak", "say", "tell", "ask", "shout", "whisper",
                "yell", "call", "announce", "mutter", "greet",
                "challenge", "warn", "threaten", "sing",
            }
            if v in _SPEECH_V:
                entity.meta["last_speak_tick"] = self.world.tick
                from .speech_utils import best_speech_line, sanitize_dialogue_text

                spoken = sanitize_dialogue_text(best_speech_line(action))
                if spoken:
                    from .speech_utils import normalize_speech_key

                    entity.meta["last_spoken_text"] = spoken
                    if isinstance(action.target, str):
                        entity.meta["last_speak_target"] = action.target
                        entity.meta["last_reply_to"] = action.target
                        entity.meta["last_reply_tick"] = self.world.tick
                    taboo = list(self.world.meta.get("room_dialogue_taboo") or [])
                    taboo.append(normalize_speech_key(spoken))
                    self.world.meta["room_dialogue_taboo"] = taboo[-24:]
                if entity.meta.get("focus_topic") == "need_food" or (
                    action.intent
                    and "obtain_food" in (action.intent.desired_outcome or [])
                ):
                    entity.meta["last_hunger_ask_tick"] = self.world.tick
        elif grounding.zone == ActionZone.ORPHANED or adjudication_transitions:
            transitions = list(adjudication_transitions)
            if transitions:
                apply_transitions(self.world, transitions)
                witnesses = _compute_witnesses(self.world, eid)
                event = Event(
                    event_id=new_event_id(),
                    tick=self.world.tick,
                    action=action,
                    transitions=transitions,
                    witnesses=witnesses,
                    narrative_hint=adjudication_ruling,
                )

                from .dialogue import update_threads
                update_threads(self.world, event)

                from .social_stimulus import broadcast_stimulus_from_event

                broadcast_stimulus_from_event(self.world, event)
                from .contact_effects import apply_contact_social_effects

                apply_contact_social_effects(self.world, event)

                npc_proj = project(
                    self.world,
                    eid,
                    last_action_rejection=entity.meta.get("last_rejection"),
                )
                self._apply_post_action_lm_layer(
                    action,
                    event,
                    npc_proj,
                    adjudicated=True,
                )
                self.world.event_log.append(event)
                apply_event_to_graph(self.world.relational, event, self.world.tick)
                npc_grounding = GroundingResult(zone=ActionZone.ORPHANED)
                narration = adjudication_ruling or self.narrator.narrate_zone3(
                    action.raw_input or str(action.verb),
                    grounding.orphan_reason or "No world affordance.",
                    npc_proj,
                    self.world,
                )
                if narration:
                    event.narrative_hint = narration

            status = (
                f"[NPC:{entity.name}:ADJUDICATED] tick={self.world.tick} "
                f"action={action.verb}"
            )
            entity.meta.pop("last_rejection", None)
        else:
            reason = validation.rejection_reason
            status = (
                f"[NPC:{entity.name}] action={action.verb} "
                f"rejected={reason}"
            )
            # Record rejection so LM knows next tick.
            if reason is not None:
                reason_str = reason.value if hasattr(reason, "value") else str(reason)
                entity.meta["last_rejection"] = (
                    f"{reason_str}: {validation.rejection_detail or ''}"
                ).rstrip(": ")

        # Tick down conditions for this NPC.
        _tick_down_conditions(entity)

        return StepResult(
            tick=self.world.tick,
            projection=SemanticProjection(focal_entity=eid, tick=self.world.tick),
            action=action,
            grounding=GroundingResult(zone=ActionZone.GROUNDED),
            validation=validation,
            event=event,
            status=status,
            narration=narration,
        )

    # ------------------------------------------------------------------
    # Replay (core invariant test)
    # ------------------------------------------------------------------

    def replay(self, initial_world: WorldState) -> WorldState:
        """
        Replay all events in the current world's event log starting from
        initial_world and return the reconstructed state.

        This verifies the core invariant: the event log is the sole source
        of truth.  The reconstructed state must match self.world exactly
        (for canonical fields).

        Usage in tests:
            initial = copy.deepcopy(world_before_any_steps)
            loop.replay(initial)
            # assert reconstructed == loop.world
        """
        reconstructed = copy.deepcopy(initial_world)
        if (
            self.player_id in self.world.spatial.entities
            and self.player_id not in reconstructed.spatial.entities
        ):
            # GameLoop may create an interactive player during construction,
            # after callers captured their pre-loop snapshot. Mirror that
            # deterministic constructor mutation before applying events.
            reconstructed.spatial.entities[self.player_id] = copy.deepcopy(
                self.world.spatial.entities[self.player_id]
            )
        for event in self.world.event_log:
            apply_transitions(reconstructed, event.transitions)
            apply_event_to_graph(reconstructed.relational, event, event.tick)
        # Multiple events share the same logical tick (NPC actions, propagation,
        # pressure, clock).  Canonical tick is advanced once per game step, not
        # once per event — mirror the live world's final counter.
        reconstructed.tick = self.world.tick
        return reconstructed


# ------------------------------------------------------------------
# Quest / goal evaluator
# ------------------------------------------------------------------


def evaluate_goals(world: WorldState) -> list:
    """
    Check every incomplete Goal; complete and return newly-completed ones.

    Called after each tick (from both ``GameLoop.step()`` and
    ``GameLoop.autonomous_tick()``).  All conditions in a goal are
    AND-combined.  When a goal completes its ``GoalReward`` side-effects
    (edge updates, lore unlock) are applied immediately.

    Returns the list of goals that *just* completed on this call.
    """
    newly_done = []
    for goal in world.config.goals:
        if goal.completed:
            continue
        if _all_conditions_met(goal.conditions, world):
            goal.completed = True
            goal.completed_at = world.tick
            _apply_goal_reward(goal, world)
            newly_done.append(goal)
    return newly_done


def _all_conditions_met(conditions, world: WorldState) -> bool:
    for cond in conditions:
        if not _condition_met(cond, world):
            return False
    return True


def _condition_met(cond, world: WorldState) -> bool:
    ctype = cond.type

    if ctype == "tick_reached":
        return cond.tick is not None and world.tick >= cond.tick

    if ctype == "edge_exists":
        edges_from = world.relational.edges.get(cond.source or "", {})
        edge_list = edges_from.get(cond.target or "", [])
        return any(
            (e.kind.value if hasattr(e.kind, "value") else e.kind) == cond.edge_kind
            for e in edge_list
        )

    if ctype == "entity_state":
        ent = world.spatial.entities.get(cond.entity or "")  # type: ignore
        if ent is None:
            return False
        return ent.emotional_state.value == (cond.state or "")

    if ctype == "event_occurred":
        for ev in world.event_log:
            verb_match = (
                not cond.verb
                or str(ev.action.verb).lower() == (cond.verb or "").lower()
            )
            actor_match = not cond.actor or ev.action.actor == cond.actor

            # target check: goal's `target` must match the action's target entity
            target_cond = getattr(cond, "target", None)
            if target_cond:
                action_target = str(ev.action.target or "").lower()
                # Resolve goal target to an actual entity id (search all regions)
                tgt_ent_id = None
                for ent in world.all_entities().values():
                    if (target_cond.lower() in ent.entity_id.lower()
                            or target_cond.lower() in ent.name.lower()):
                        tgt_ent_id = ent.entity_id
                        break
                target_match = (
                    target_cond.lower() in action_target
                    or (tgt_ent_id is not None and action_target == tgt_ent_id.lower())
                )
            else:
                target_match = True

            # same_region check: actor and target must be in the same region
            # at the time the event occurred (prevents off-screen interactions)
            same_region_match = True
            if getattr(cond, "same_region", False) and target_cond:
                actor_ent = world.all_entities().get(ev.action.actor or "")
                tgt_ent = None
                tgt_id_str = str(ev.action.target or "")
                if tgt_id_str:
                    tgt_ent = world.all_entities().get(tgt_id_str)
                if actor_ent is None or tgt_ent is None:
                    same_region_match = False
                else:
                    same_region_match = (
                        getattr(actor_ent, "region_id", None)
                        == getattr(tgt_ent, "region_id", None)
                    )

            if verb_match and actor_match and target_match and same_region_match:
                return True
        return False

    return False


def _apply_goal_reward(goal, world: WorldState) -> None:
    from .schemas import EdgeKind, RelationalEdge

    # Lore unlock — write into world meta
    if goal.reward.lore_unlock:
        unlocked = world.meta.setdefault("unlocked_lore", [])
        if goal.reward.lore_unlock not in unlocked:
            unlocked.append(goal.reward.lore_unlock)

    # Edge updates
    for upd in goal.reward.edge_updates:
        src_edges = world.relational.edges.setdefault(upd.source, {})
        tgt_list = src_edges.setdefault(upd.target, [])
        try:
            kind_val = EdgeKind(upd.edge_kind)
        except ValueError:
            kind_val = upd.edge_kind  # type: ignore
        # Replace existing edge of same kind or append
        for existing in tgt_list:
            ek = existing.kind.value if hasattr(existing.kind, "value") else existing.kind
            if ek == upd.edge_kind:
                existing.weight = upd.weight
                break
        else:
            tgt_list.append(
                RelationalEdge(
                    kind=kind_val,
                    weight=upd.weight,
                    since_tick=world.tick,
                )
            )


# ------------------------------------------------------------------
# Helper: compute witnesses for an action
# ------------------------------------------------------------------


def _tick_down_conditions(entity: "Optional[EntityState]") -> None:
    """Decrement all condition timers by 1 and remove those that expire."""
    if entity is None:
        return
    expired = [c for c, t in entity.conditions.items() if t <= 1]
    for c in expired:
        del entity.conditions[c]
    for c in list(entity.conditions):
        if c not in expired:
            entity.conditions[c] -= 1


def _compute_witnesses(world: WorldState, actor_id: EntityId) -> list[EntityId]:
    """Return entity ids in the actor's region that can see them (incl. actor)."""
    from .region_utils import witness_entity_ids

    return witness_entity_ids(world, actor_id)


# ------------------------------------------------------------------
# Convenience: build a minimal test world
# ------------------------------------------------------------------


_DEFAULT_WORLD_PACK = "worlds/default"


def make_test_world(
    pack: str = _DEFAULT_WORLD_PACK,
) -> WorldState:
    """
    Load the default world pack from disk and return the resulting
    WorldState. This is the entry point for tests and the REPL — the
    contents of the world are defined entirely by data files in
    ``worlds/<pack>/`` and can be edited without touching code.

    Pass a different ``pack`` path to load an alternate world. The path
    is resolved relative to the repository root.
    """
    from pathlib import Path
    from .world_loader import load_world_pack

    pack_path = Path(pack)
    if not pack_path.is_absolute():
        # Resolve relative to the repository root (parent of src/).
        repo_root = Path(__file__).resolve().parent.parent.parent
        pack_path = repo_root / pack
    world = load_world_pack(pack_path)
    world.config.consequence_policy.record_traces = False
    return world
