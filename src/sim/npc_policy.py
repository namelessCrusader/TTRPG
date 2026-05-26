"""
NPC behavior layer.

Architectural shape mirrors the LM adapter pattern:

    LMAdapter        : natural language intent → SemanticAction (for the player)
    NpcPolicy        : canonical world state    → SemanticAction (for NPCs)

Both produce SemanticActions that flow through the same Compiler. There
is no second action pipeline. The NPC policy is part of the simulation
(it reads canonical state directly) and does NOT receive a
SemanticProjection — the projection firewall exists to limit what the
LM sees, not what in-engine code sees.

Default policy is `ReactivePolicy`: a state-driven decision tree.
No hardcoded English, no per-NPC scripts. Decisions are derived from:

  - entity.alertness, entity.emotional_state, entity.health
  - the recent event log (who attacked me lately?)
  - the relational graph (who do I have hostile edges toward?)
  - line of sight + manhattan distance

Adding a new behavior tier is a new state predicate, not a new keyword.
"""

from __future__ import annotations

import logging
from typing import Optional, Protocol, TYPE_CHECKING

from .needs import critical_needs, need_urgency, NEED_LABELS, VERB_SATISFY
from .speech_utils import collect_recent_room_dialogue as _collect_recent_room_dialogue
from .schemas import (
    ActionType,
    AlertnessLevel,
    Coord,
    ConsentState,
    EdgeKind,
    EmotionalState,
    EntityId,
    EntityState,
    IntentBlock,
    MalformedActionError,
    NpcCharacterSheet,
    ObjectId,
    SemanticAction,
    SpatialGrid,
    StyleBlock,
    TransitionKind,
    WorldState,
)
from .observation_memory import (
    clear_actionable_stimulus,
    get_actionable_stimulus,
    has_fresh_stimulus,
    memory_tick_window,
    observations_in_window,
    format_observations_for_sheet,
)
from .social_stimulus import stimulus_priority
from .spatial import visible_from

if TYPE_CHECKING:
    from .lm_adapter import LMAdapter

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Policy protocol
# ─────────────────────────────────────────────────────────────────────────────


class NpcPolicy(Protocol):
    """A policy that decides what a single NPC does this tick."""

    def decide(self, entity: EntityState, world: WorldState) -> SemanticAction: ...


# ─────────────────────────────────────────────────────────────────────────────
# Default reactive policy
# ─────────────────────────────────────────────────────────────────────────────


# How far back in event history to look for recent threats.
_THREAT_LOOKBACK_TICKS = 5

# Health fraction below which the NPC prefers fleeing to fighting.
_FLEE_HEALTH_FRACTION = 0.25

# Knowledge prefixes that must not be spoken or used as dialogue seeds.
_META_KNOWLEDGE_PREFIXES = (
    "before tonight:",
    "before the scene",
    "before you arrived",
    "[history]",
    "[npc_auto:",
)


def _dialogue_safe_knowledge(knowledge: list[str]) -> list[str]:
    """Filter entity.knowledge to author-authored facts safe for spoken lines."""
    safe: list[str] = []
    for entry in knowledge:
        text = entry.strip()
        if len(text) < 20:
            continue
        lower = text.lower()
        if any(lower.startswith(p) for p in _META_KNOWLEDGE_PREFIXES):
            continue
        if " tick " in lower or lower.endswith(" speak.") or lower.endswith(" speak"):
            continue
        safe.append(text)
    return safe


class ReactivePolicy:
    """
    Deterministic state-driven NPC behavior.

    Decision priority (each gate checks canonical state only):

      1. RECENT_THREAT
         If an entity attacked us in the last _THREAT_LOOKBACK_TICKS ticks
         and is still alive:
            • health too low                → FLEE
            • attacker within melee range  → ATTACK
            • attacker visible elsewhere    → MOVE toward attacker (close)

      2. COMBAT_ALERT
         If alertness == COMBAT and any visible entity is currently
         classified as an enemy (hostile relational/emotional edge):
            • adjacent enemy                → ATTACK
            • visible enemy                 → MOVE toward
         
      3. FEAR
         If emotional_state == FEARFUL and any visible threat exists
         → FLEE.

      4. WATCHFUL
         If alertness >= MEDIUM and any entity is visible → OBSERVE.

      5. IDLE → WAIT.
    """

    def __init__(
        self,
        threat_lookback_ticks: int = _THREAT_LOOKBACK_TICKS,
        flee_health_fraction: float = _FLEE_HEALTH_FRACTION,
    ):
        self.threat_lookback_ticks = threat_lookback_ticks
        self.flee_health_fraction = flee_health_fraction

    # ── Public entry point ─────────────────────────────────────────────────
    def decide(self, entity: EntityState, world: WorldState) -> SemanticAction:
        """Generate feasible actions, score by mind context, pick the best fit."""
        from .npc_mind import interpret_mind, pick_best_action

        if world.meta.get("magic_duel"):
            return self.decide_magic_duel(entity, world)

        interpret_mind(entity, world)
        candidates = self.generate_candidates(entity, world)
        if _recently_answered_contact(entity.entity_id, world, self.threat_lookback_ticks):
            cooled = [
                c for c in candidates
                if str(c[0].verb).lower().split(".")[-1] in {
                    ActionType.SPEAK,
                    ActionType.OBSERVE,
                    ActionType.WAIT,
                    ActionType.MOVE,
                }
            ]
            if cooled:
                candidates = cooled
        valid_candidates = self._valid_candidates(candidates, world)
        if valid_candidates:
            candidates = valid_candidates
        action = pick_best_action(entity, world, candidates)
        entity.meta["last_policy_branch"] = "mind_scored"
        return action

    @staticmethod
    def _valid_candidates(
        candidates: list[tuple[SemanticAction, str, float]],
        world: WorldState,
    ) -> list[tuple[SemanticAction, str, float]]:
        """Keep reactive picks executable when used as an LM-failure fallback."""
        from .compiler import compile_action

        valid: list[tuple[SemanticAction, str, float]] = []
        for action, label, weight in candidates:
            try:
                if compile_action(action, world).valid:
                    valid.append((action, label, weight))
            except Exception:
                continue
        return valid

    @staticmethod
    def _filter_active_candidates(
        entity: EntityState,
        candidates: list[tuple[SemanticAction, str, float]],
    ) -> list[tuple[SemanticAction, str, float]]:
        """Drop passive loops so speak/move/cast beat endless watching."""
        if not candidates:
            return candidates
        passive = frozenset({"observe", "wait", "rest"})
        streak = int(entity.meta.get("observe_streak", 0))
        last_verb = str(entity.meta.get("last_action_verb", "")).lower()

        pool = list(candidates)
        if streak >= 1 or last_verb in passive:
            active = [c for c in pool if str(c[0].verb).lower() not in passive]
            if active:
                pool = active
        if last_verb == "observe":
            non_observe = [c for c in pool if str(c[0].verb).lower() != "observe"]
            if non_observe:
                pool = non_observe
        return pool

    @staticmethod
    def _pick_best_candidate(
        entity: EntityState,
        candidates: list[tuple[SemanticAction, str, float]],
        world: WorldState,
    ) -> SemanticAction:
        """Score candidates with npc_mind context (shared by reactive + LM fallback)."""
        from .npc_mind import interpret_mind, pick_best_action

        interpret_mind(entity, world)
        return pick_best_action(entity, world, candidates)

    def decide_magic_duel(
        self, entity: EntityState, world: WorldState
    ) -> SemanticAction:
        """Deterministic spell pick for 1v1 duel worlds (highest-weight CAST)."""
        candidates = self.generate_candidates(entity, world, max_candidates=12)
        valid_candidates = self._valid_candidates(candidates, world)
        if valid_candidates:
            candidates = valid_candidates
        if not candidates:
            return SemanticAction(
                verb=ActionType.WAIT, actor=entity.entity_id, raw_input="[npc_auto:duel_wait]"
            )
        cast_candidates = [
            c for c in candidates
            if str(c[0].verb).lower().split(".")[-1] == ActionType.CAST
        ]
        if cast_candidates:
            thermal = [
                c for c in cast_candidates
                if c[0].intent
                and any(
                    op in (c[0].intent.rationale or "").upper()
                    for op in ("HEAT", "CHILL")
                )
            ]
            return max(thermal or cast_candidates, key=lambda c: c[2])[0]
        return self._pick_best_candidate(entity, candidates, world)

    @staticmethod
    def _semantic_action_bias(
        entity: EntityState, world: WorldState, action: SemanticAction
    ) -> float:
        """Bias candidates from delayed semantic pressure field."""
        from .semantic_slice import ensure_semantic_state

        state = ensure_semantic_state(world)
        pressure = state.get("pressure")
        if not isinstance(pressure, dict):
            return 0.0
        bias = 0.0
        verb = str(action.verb).lower().split(".")[-1]
        local = pressure.get("local")
        if isinstance(local, dict):
            momentum = float(local.get("violence_vs_reconciliation", 0.0))
            if momentum > 0.2 and verb in (
                "accuse", "threaten", "shout", "intimidate", "attack",
            ):
                bias += 0.1
                if verb == "threaten" and entity.meta.get("last_action_verb") == "speak":
                    bias -= 0.25
            if momentum < -0.15 and verb in ("speak", "gossip", "apologize"):
                bias += 0.08
        return bias

    def generate_candidates(
        self,
        entity: EntityState,
        world: WorldState,
        max_candidates: int = 28,
    ) -> list[tuple[SemanticAction, str, float]]:
        """
        Return up to `max_candidates` (action, label, weight) tuples.

        The label is written in vivid, in-character terms so a language model
        can pick meaningfully.  Weights express plausibility; the LM is
        allowed to override them.

        Non-negotiable PRIORITY actions (return immediately, no LM choice):
          combat / flee / threat-response / low-stamina.

        Everything else is offered as a menu.
        """
        # Long-horizon planner bookkeeping runs unconditionally at the top so
        # that hazard interrupts (smoke / fire / combat) pause the active
        # plan even when a higher-priority reactive branch below short-
        # circuits with its own action (e.g. ``hazard_avoidance`` returns a
        # FLEE candidate).  Without this, the entity would lose track of
        # the project it was on and never resume it once the danger passed.
        from . import npc_planner as _planner

        _planner.check_interrupt(entity, world)
        _planner.try_resume(entity, world)
        if entity.active_plan is None:
            proposed = _planner.propose_plan(entity, world)
            if proposed is not None:
                entity.active_plan = proposed

        grid = world.spatial
        candidates: list[tuple[SemanticAction, str, float]] = []
        eid = entity.entity_id
        occ = (entity.occupation or "").lower()
        stamina = entity.stats.get("stamina", 100.0) if entity.stats else 100.0

        # Per-NPC speech cooldown — derived from social_openness (set in entities.yaml),
        # not hardcoded occupation strings.  More open characters speak more frequently;
        # guarded ones hold back.  A small hash-based stagger breaks lockstep synchrony.
        _openness_window_map = {
            "welcoming": 2, "open": 2,
            "guarded": 3, "reserved": 4, "closed": 5,
        }
        _base_window = _openness_window_map.get(entity.social_openness.lower(), 3)
        # 0-or-1 stagger so two NPCs of the same type don't speak simultaneously
        _stagger = int(abs(hash(eid)) % 2)
        _speak_window = _base_window + _stagger

        _last_speak_tick = int(entity.meta.get("last_speak_tick", -999))
        _ticks_since_speak = world.tick - _last_speak_tick
        _min_speak_gap = int(world.meta.get("npc_min_speak_gap", 0))
        if _ticks_since_speak >= _speak_window:
            _speak_weight_penalty = 0.0
        elif _ticks_since_speak <= 0:
            _speak_weight_penalty = -0.95        # same tick — no repeat line
        else:
            # Brief gap only — keep speak above use/observe/move
            _speak_weight_penalty = -0.12

        def _add(action: SemanticAction, label: str, weight: float) -> None:
            from .pressure_eval import pressure_weight_delta
            from .speech_utils import (
                best_speech_line,
                is_real_dialogue,
                pick_fallback_voice_line,
            )

            goals = list(getattr(entity, "goals", None) or [])
            if action.verb == ActionType.SPEAK and not is_real_dialogue(
                best_speech_line(action, goals=goals), goals=goals
            ):
                line = pick_fallback_voice_line(entity, world)
                if line:
                    intent = action.intent or IntentBlock()
                    action = action.model_copy(
                        update={
                            "intent": intent.model_copy(
                                update={
                                    "rationale": line,
                                    "manner": line,
                                }
                            )
                        }
                    )

            weight += pressure_weight_delta(
                eid,
                action.raw_input or "",
                str(action.verb),
                world,
            )
            weight += ReactivePolicy._semantic_action_bias(
                entity, world, action
            )
            if str(action.verb).lower() in (
                "speak", "say", "tell", "ask", "shout", "whisper",
                "yell", "call", "announce", "mutter", "greet",
                "challenge", "warn", "threaten",
            ):
                weight = max(0.05, weight + _speak_weight_penalty)
            candidates.append((action, label, weight))

        # ── PRIORITY: non-negotiable ──────────────────────────────────────

        contact = _most_recent_contact_against(eid, world, lookback=self.threat_lookback_ticks)
        if contact is not None:
            resp = self._respond_to_contact(entity, contact, world)
            _add(resp, f"respond to contact from {contact.actor}", 1.0)
            return candidates

        if stamina < 30.0 and entity.alertness in (AlertnessLevel.UNAWARE, AlertnessLevel.LOW):
            _add(SemanticAction(verb=ActionType.REST, actor=eid, raw_input="[npc_auto:low_stamina]"),
                 "collapse to rest — exhausted", 1.0)
            return candidates

        # Magic duel: spell menu only — no social/observe/wander branches.
        if world.meta.get("magic_duel"):
            opponent = _magic_duel_opponent(
                entity, world, self.threat_lookback_ticks
            )
            if opponent is not None:
                self._add_magic_duel_candidates(entity, world, opponent, _add)
                if candidates:
                    return candidates

        threat = _most_recent_attacker(eid, world, lookback=self.threat_lookback_ticks)
        if threat is not None:
            if world.meta.get("magic_duel"):
                self._add_magic_duel_candidates(entity, world, threat, _add)
                if candidates:
                    return candidates
            resp = self._respond_to_threat(entity, threat, world)
            _add(resp, "respond to attack", 1.0)
            return candidates

        if entity.alertness == AlertnessLevel.COMBAT:
            enemy = _magic_duel_opponent(entity, world, self.threat_lookback_ticks)
            if enemy is not None:
                if world.meta.get("magic_duel"):
                    self._add_magic_duel_candidates(entity, world, enemy, _add)
                    if candidates:
                        return candidates
                _add(self._engage(entity, enemy), "engage enemy in combat", 1.0)
                return candidates

        # Environmental hazards (fire, smoke) — before social menu.
        from .hazard_avoidance import assess_hazards

        haz = assess_hazards(entity, world)
        if haz.should_flee:
            label = "flee — " + (haz.reasons[0] if haz.reasons else "danger")
            _add(
                SemanticAction(
                    verb=ActionType.FLEE,
                    actor=eid,
                    raw_input="[npc_auto:hazard_flee]",
                    intent=IntentBlock(
                        rationale=label,
                        desired_outcome=["escape_hazard"],
                    ),
                ),
                label,
                1.0,
            )
            return candidates
        if haz.should_move_away and not haz.should_flee:
            if any("smoke" in r for r in haz.reasons):
                _add(
                    SemanticAction(
                        verb=ActionType.FLEE,
                        actor=eid,
                        raw_input="[npc_auto:hazard_move_away]",
                        intent=IntentBlock(
                            rationale="move away from smoke or fire",
                            desired_outcome=["avoid_hazard"],
                        ),
                    ),
                    "back away from the smoke and flames",
                    0.88,
                )

        # Flee only after real harm — not because a distrusted patron is visible
        if entity.emotional_state == EmotionalState.FEARFUL:
            if (
                entity.meta.get("focus_topic") == "threat"
                and _most_recent_attacker(eid, world, self.threat_lookback_ticks)
            ):
                _add(
                    SemanticAction(
                        verb=ActionType.FLEE,
                        actor=eid,
                        raw_input="[npc_auto:fearful]",
                    ),
                    "run away — attacked or burning",
                    1.0,
                )
                return candidates

        # ── Gather context ────────────────────────────────────────────────

        nearby = _first_visible_entity(entity, world)
        nearby_ent = grid.entities.get(nearby) if nearby else None
        nearby_name = nearby_ent.name if nearby_ent else None

        # How many ticks in a row has this entity been observing the same target?
        last_action_verb = entity.meta.get("last_action_verb", "")
        observe_streak = int(entity.meta.get("observe_streak", 0))

        # Track recently used rationale hints to avoid exact repeats in speech
        recent_rationales: set[str] = set()
        for _ev in world.event_log[-4:]:
            for _t in _ev.transitions:
                if (
                    _t.kind.value == "dialogue_spoken"
                    and _t.payload.get("actor") == eid
                ):
                    recent_rationales.add(_t.payload.get("intent_hint", ""))

        # Recent player speech / provocations
        recent_loud: list[dict] = []
        for ev in reversed(world.event_log[-5:]):
            for t in ev.transitions:
                if t.kind.value == "dialogue_spoken" and t.payload.get("loud"):
                    recent_loud.append(t.payload)
        has_provocative_speech = bool(recent_loud)
        recent_text = recent_loud[0].get("text", "") if recent_loud else ""

        # ── Most recent nearby speech (conversational thread) ────────────────
        # Conversational thread: observational memory (LOS, Markov window) first.
        _recent_nearby_speech: dict | None = None
        _max_reaction_lookback = memory_tick_window(world)
        from .speech_utils import is_real_dialogue, sanitize_dialogue_text

        for _obs in reversed(observations_in_window(entity, world)):
            if _obs.get("kind") != "speech" and not _obs.get("text"):
                continue
            _txt = sanitize_dialogue_text(_obs.get("text") or "")
            if not _txt:
                continue
            _recent_nearby_speech = {
                "speaker_name": _obs.get("actor_name", "someone"),
                "speaker_id": _obs.get("actor_id", ""),
                "text": _txt[:120],
                "addressed": bool(_obs.get("addressed")),
            }
            break
        if _recent_nearby_speech is None:
            for _ev in reversed(world.event_log[-10:]):
                _tick_age = world.tick - _ev.tick if hasattr(_ev, "tick") else 0
                if _tick_age > _max_reaction_lookback:
                    continue
                for _t in _ev.transitions:
                    if _t.kind.value != "dialogue_spoken":
                        continue
                    _speaker_id = _t.payload.get("actor", "")
                    if _speaker_id == eid:
                        continue
                    _txt = (_t.payload.get("text") or "").strip()
                    if not _txt or _txt == "...":
                        continue
                    _speaker_ent = grid.entities.get(_speaker_id)
                    if _speaker_ent is None:
                        continue
                    _dist = entity.position.manhattan(_speaker_ent.position)
                    if _dist > entity.sight_range:
                        continue
                    _recent_nearby_speech = {
                        "speaker_name": _speaker_ent.name,
                        "speaker_id": _speaker_id,
                        "text": _txt,
                    }
                    break
                if _recent_nearby_speech is not None:
                    break

        # ── PRIORITY 2: Needs override occupation behaviour ───────────────
        # When a need is in crisis it dominates the NPC's decision tree,
        # just like hunger overrides job duty in The Sims / RimWorld.

        urgent_needs = critical_needs(entity)
        if urgent_needs:
            most_urgent_need, need_val = urgent_needs[0]
            urgency = need_urgency(entity)

            if most_urgent_need == "hunger":
                # Look for food/drink items in inventory or nearby
                food_in_inv = next(
                    (oid for oid in entity.inventory
                     if any(t in ("food", "drink", "ale", "bread", "meat")
                            for t in (world.spatial.objects.get(oid, None) and
                                      world.spatial.objects[oid].tags or []))),
                    None
                )
                if food_in_inv:
                    _add(
                        SemanticAction(
                            verb=ActionType.EAT, actor=eid, target=food_in_inv,
                            intent=IntentBlock(
                                rationale="stomach is growling — must eat",
                                manner="urgent",
                                desired_outcome=["satisfy_hunger"],
                            ),
                            raw_input="[npc_auto:need_hunger]",
                        ),
                        f"eat something from inventory — hunger is critical ({need_val:.0f}/100)", 0.98,
                    )
                elif nearby:
                    last_ask = int(entity.meta.get("last_hunger_ask_tick", -999))
                    if world.tick - last_ask >= 4:
                        _add(
                            SemanticAction(
                                verb=ActionType.SPEAK, actor=eid, target=nearby,
                                intent=IntentBlock(
                                    rationale="",
                                    manner="conversational",
                                    desired_outcome=["obtain_food"],
                                ),
                                raw_input="[npc_auto:need_hunger_ask]",
                            ),
                            f"ask {nearby_name} about food or drink — starving", 0.88,
                        )

            elif most_urgent_need == "fatigue":
                _add(
                    SemanticAction(
                        verb=ActionType.REST, actor=eid,
                        intent=IntentBlock(
                            rationale="exhaustion is overwhelming — must rest",
                            manner="weary",
                            desired_outcome=["recover_energy"],
                        ),
                        raw_input="[npc_auto:need_fatigue]",
                    ),
                    f"slump into a chair — too exhausted to function ({need_val:.0f}/100)", 0.97,
                )

            elif most_urgent_need == "social_need":
                if nearby:
                    _add(
                        SemanticAction(
                            verb=ActionType.SPEAK, actor=eid, target=nearby,
                            intent=IntentBlock(
                                rationale="desperately need human connection — have been too isolated",
                                manner="open and eager",
                                desired_outcome=["connect", "be_heard"],
                            ),
                            raw_input="[npc_auto:need_social]",
                        ),
                        f"seek conversation with {nearby_name} — social isolation is unbearable", 0.93,
                    )

            elif most_urgent_need == "purpose":
                # Purpose-starved NPCs seek direction through talk, not idle watching.
                if entity.goals and nearby:
                    goal_snip = entity.goals[0][:50]
                    _add(
                        SemanticAction(
                            verb=ActionType.SPEAK, actor=eid, target=nearby,
                            intent=IntentBlock(
                                rationale=f"pursue goal: {goal_snip}",
                                manner="purposeful",
                                desired_outcome=["advance_goal"],
                            ),
                            raw_input="[npc_auto:need_purpose_speak]",
                        ),
                        f"talk to {nearby_name} about goal: '{goal_snip}'", 0.90,
                    )
                elif nearby:
                    _add(
                        SemanticAction(
                            verb=ActionType.SPEAK, actor=eid, target=nearby,
                            intent=IntentBlock(
                                rationale="purposeless and restless — looking for something, anything to do",
                                manner="restless",
                                desired_outcome=["find_direction"],
                            ),
                            raw_input="[npc_auto:need_purpose_seek]",
                        ),
                        f"approach {nearby_name} seeking some kind of direction or task", 0.85,
                    )
            # If urgency is high enough, return immediately (need overrides everything else)
            if urgency > 0.7 and candidates:
                return candidates

        # ── World fact investigation ──────────────────────────────────────
        # Durable environmental changes (graffiti, spills, marks) should draw
        # attention before idle wandering or generic social chatter.
        _world_fact_obs: dict | None = None
        for _o in reversed(observations_in_window(entity, world)):
            if (
                _o.get("kind") == "world_fact"
                and int(_o.get("priority", 0)) >= 40
            ):
                _world_fact_obs = _o
                break

        if _world_fact_obs is not None:
            _fact_summary = str(_world_fact_obs.get("summary", "something unusual"))
            _fact_dest = self._coord_for_world_fact(_world_fact_obs, world)
            if _fact_dest is not None:
                _fdist = entity.position.manhattan(_fact_dest)
                if _fdist > 1:
                    _add(
                        SemanticAction(
                            verb=ActionType.MOVE,
                            actor=eid,
                            target=_fact_dest,
                            raw_input="[npc_auto:investigate_fact]",
                        ),
                        f"investigate: {_fact_summary[:50]}",
                        0.86,
                    )
                else:
                    _add(
                        SemanticAction(
                            verb=ActionType.EXAMINE,
                            actor=eid,
                            target=_fact_dest,
                            raw_input="[npc_auto:examine_fact]",
                        ),
                        f"examine closely: {_fact_summary[:50]}",
                        0.88,
                    )
            if nearby:
                _add(
                    SemanticAction(
                        verb=ActionType.SPEAK,
                        actor=eid,
                        target=nearby,
                        intent=IntentBlock(
                            rationale=f"comment on noticed change: {_fact_summary}",
                            manner="concerned",
                            desired_outcome=["share_observation"],
                        ),
                        raw_input="[npc_auto:report_fact]",
                    ),
                    f"tell {nearby_name} about: {_fact_summary[:40]}",
                    0.84,
                )

        # ── Pack + embodied menu (before generic speak flood) ─────────────
        # Pack verbs were previously appended last and never fit in max_candidates.
        from .capability_menu import pack_capability_candidates

        for action, label, weight in pack_capability_candidates(entity, world):
            _add(action, label, weight)

        # ── Secret trading: share a secret when trust is high ────────────
        # NPCs with secrets and a high-trust nearby entity may offer info
        # as social currency or for personal gain.
        if entity.secrets and nearby_ent:
            # Check if we trust the nearby entity (intimate / ally edge)
            _outbound = world.relational.edges.get(str(eid), {})
            _nearby_edges = _outbound.get(str(nearby_ent.entity_id), [])
            _trust_kinds = {EdgeKind.ALLY_OF, EdgeKind.INTIMATE, EdgeKind.EMPLOYS}
            _is_trusted = any(e.kind in _trust_kinds for e in _nearby_edges)
            # Also reveal under social pressure: if social_need is low, NPCs become looser
            _social_need = entity.stats.get("social_need", 60.0)
            _loose_lipped = _social_need < 30.0

            if (_is_trusted or _loose_lipped) and _ticks_since_speak > _speak_window:
                secret = entity.secrets[0]
                secret_snip = secret[:60]
                _add(
                    SemanticAction(
                        verb=ActionType.SPEAK, actor=eid, target=nearby_ent.entity_id,
                        intent=IntentBlock(
                            rationale=f"confide secret to trusted ally: {secret_snip}",
                            manner="hushed, conspiratorial",
                            desired_outcome=["share_secret", "build_trust"],
                        ),
                        raw_input="[npc_auto:secret_trade]",
                    ),
                    f"lean in and whisper a secret to {nearby_name}: '{secret_snip[:30]}…'",
                    0.78 if _is_trusted else 0.55,
                )

        # ── Drive-based active behavior ──────────────────────────────────────
        # All candidate generation uses entity data (drive, personality, goals,
        # alertness) from entities.yaml — no hardcoded occupation strings.
        # The LM selector (_lm_select) then picks the most fitting candidate
        # using the NPC's full character context.

        if nearby:
            _nearby_dist = entity.position.manhattan(nearby_ent.position) if nearby_ent else 99

            # Build a label from drive text so the LM selector has rich context.
            _drive_first = entity.drive.split(";")[0].strip()[:70] if entity.drive else ""
            _speak_label = (
                f"speak to {nearby_name} — {_drive_first}"
                if _drive_first
                else f"say something to {nearby_name}"
            )

            # Below pack verbs so gossip/pour/nod can win unless conversation is urgent.
            _speak_weight = (
                0.62 if entity.alertness in (AlertnessLevel.MEDIUM, AlertnessLevel.HIGH)
                else 0.52
            )
            _add(
                SemanticAction(
                    verb=ActionType.SPEAK, actor=eid, target=nearby,
                    intent=IntentBlock(
                        rationale="engage socially",
                        manner="",
                        desired_outcome=["interact"],
                    ),
                    raw_input="[npc_auto:drive_speak]",
                ),
                _speak_label, _speak_weight,
            )

            # Observe — brief beat only; deprioritized so speak/move win after one glance.
            if observe_streak < 1:
                _add(
                    SemanticAction(
                        verb=ActionType.OBSERVE, actor=eid, target=nearby,
                        raw_input="[npc_auto:idle_observe]",
                    ),
                    f"glance at {nearby_name}", 0.35,
                )

            # Approach if too far for comfortable interaction
            if _nearby_dist > 4:
                _add(
                    SemanticAction(
                        verb=ActionType.MOVE, actor=eid, target=nearby_ent.position,
                        raw_input="[npc_auto:approach]",
                    ),
                    f"close the gap with {nearby_name} to engage", 0.52,
                )

            # React to drama in the room — higher priority than idle speech
            if has_provocative_speech:
                _add(
                    SemanticAction(
                        verb=ActionType.SPEAK, actor=eid, target=nearby,
                        intent=IntentBlock(
                            rationale="react to what was just said",
                            manner="reactive", desired_outcome=["respond"],
                        ),
                        raw_input="[npc_auto:react_drama]",
                    ),
                    f"react to the heated exchange near {nearby_name}", 0.78,
                )

        else:
            # No nearby entity — seek the closest other NPC to interact with.
            # Sort by distance so we always approach the nearest reachable person.
            _seek_candidates = sorted(
                [
                    (entity.position.manhattan(ent2.position), eid2)
                    for eid2, ent2 in grid.entities.items()
                    if eid2 != eid and ent2.alive
                    and entity.position.manhattan(ent2.position) <= entity.sight_range * 2
                ]
            )
            if _seek_candidates:
                _seek_ent = grid.entities[_seek_candidates[0][1]]
                _add(
                    SemanticAction(
                        verb=ActionType.MOVE, actor=eid, target=_seek_ent.position,
                        raw_input="[npc_auto:seek_social]",
                    ),
                    f"move toward {_seek_ent.name} — driven to interact", 0.72,
                )

        # ── Universal extras (any NPC, any situation) ─────────────────────

        # ── TIME-OF-DAY SCHEDULES ──────────────────────────────────────────
        # NPCs follow authored daily routines independent of player presence.
        # Schedule format in entity.meta["schedule"]:
        #   {"dawn": {"activity": "sleep", "target_tag": "bed"},
        #    "day":  {"activity": "work",  "target_tag": "workstation"},
        #    "dusk": {"activity": "socialize"},
        #    "night":{"activity": "patrol", "waypoints": [[x,y], ...]}}
        _schedule = entity.meta.get("schedule")
        if _schedule and isinstance(_schedule, dict):
            _tod = world.config.clock.time_of_day(world.tick).value
            _slot = _schedule.get(_tod)
            if _slot and isinstance(_slot, dict):
                _sched_activity = str(_slot.get("activity", ""))
                _sched_tag = str(_slot.get("target_tag", ""))

                if _sched_activity == "sleep" and entity.alertness in (
                    AlertnessLevel.UNAWARE, AlertnessLevel.LOW
                ):
                    _add(
                        SemanticAction(
                            verb=ActionType.REST, actor=eid,
                            intent=IntentBlock(rationale="time to sleep", desired_outcome=["rest"]),
                            raw_input="[npc_schedule:sleep]",
                        ),
                        "follow schedule — time to sleep",
                        0.85,
                    )

                elif _sched_activity in ("work", "tend", "serve"):
                    # Move toward nearest object with the scheduled tag
                    _target_obj = next(
                        (
                            obj for obj in grid.objects.values()
                            if obj.position is not None
                            and _sched_tag in obj.tags
                        ),
                        None,
                    )
                    if _target_obj is not None and _target_obj.position is not None:
                        dist_to_target = entity.position.manhattan(_target_obj.position)
                        if dist_to_target > 1:
                            _add(
                                SemanticAction(
                                    verb=ActionType.MOVE, actor=eid,
                                    target=_target_obj.position,
                                    intent=IntentBlock(rationale=f"heading to {_target_obj.name} for {_sched_activity}"),
                                    raw_input=f"[npc_schedule:{_sched_activity}]",
                                ),
                                f"follow schedule — {_sched_activity} near {_target_obj.name}",
                                0.75,
                            )

                elif _sched_activity == "socialize" and nearby:
                    _add(
                        SemanticAction(
                            verb=ActionType.SPEAK, actor=eid, target=nearby,
                            intent=IntentBlock(rationale="making rounds — social hour"),
                            raw_input="[npc_schedule:socialize]",
                        ),
                        "follow schedule — socialize",
                        0.68,
                    )

                elif _sched_activity == "patrol":
                    _sched_wps = _slot.get("waypoints")
                    if _sched_wps and isinstance(_sched_wps, list) and not entity.waypoints:
                        from .schemas import Coord
                        try:
                            _parsed_wps = [Coord(x=int(wp[0]), y=int(wp[1])) for wp in _sched_wps]
                            entity.waypoints = _parsed_wps
                            entity.waypoint_index = 0
                        except Exception:
                            pass

        # Spontaneous NPC-to-NPC interaction: when idle and a nearby NPC is also
        # idle, occasionally initiate social contact without player involvement.
        _social_cooldown = int(entity.meta.get("last_npc_npc_tick", -999))
        _npc_npc_gap = 8  # minimum ticks between spontaneous NPC-to-NPC interactions
        if (
            world.tick - _social_cooldown >= _npc_npc_gap
            and entity.alertness in (AlertnessLevel.UNAWARE, AlertnessLevel.LOW)
            and len(candidates) < 3  # only if not busy with other things
        ):
            for _other_eid, _other in grid.entities.items():
                if _other_eid == eid or not _other.alive:
                    continue
                if _other.alertness not in (AlertnessLevel.UNAWARE, AlertnessLevel.LOW):
                    continue
                dist = entity.position.manhattan(_other.position)
                if dist > 5:
                    continue
                # This NPC has a social aim toward the other, or they're connected
                _has_aim = any(
                    str(aim.target_entity_id) == str(_other_eid)
                    for aim in (entity.social_aims or [])
                )
                if _has_aim or dist <= 2:
                    _add(
                        SemanticAction(
                            verb=ActionType.SPEAK, actor=eid, target=_other_eid,
                            intent=IntentBlock(
                                rationale=f"strikes up conversation with {_other.name}",
                                desired_outcome=["social_exchange"],
                            ),
                            raw_input="[npc_auto:npc_npc_social]",
                        ),
                        f"initiate conversation with {_other.name}",
                        0.55,
                    )
                    entity.meta["last_npc_npc_tick"] = world.tick
                    break

        # Waypoint patrol: any NPC (guard, runner, courier, etc.) with a
        # defined route follows it regardless of occupation.  Duty (patrol)
        # outweighs casual social interaction — weight beats goal-driven speak.
        if entity.waypoints:
            wp = self._patrol_waypoint(entity)
            if wp:
                _add(wp, "continue patrol route", 0.92)

        # Direct conversational reply: if someone nearby spoke recently, offer a
        # high-weight "reply to them" candidate so the NPC responds directly.
        # This is the primary mechanism for conversational threading.
        if _recent_nearby_speech:
            _rns = _recent_nearby_speech
            _reply_target = _rns["speaker_id"]
            _reply_snip = _rns["text"][:60].rstrip(".,;")
            _last_reply_to = entity.meta.get("last_reply_to")
            _last_reply_tick = int(entity.meta.get("last_reply_tick", -999))
            _same_thread = (
                _last_reply_to == _reply_target
                and (world.tick - _last_reply_tick) < max(_speak_window, _min_speak_gap or 2)
            )
            if not _same_thread:
                _stim_boost = (
                    0.12
                    if has_fresh_stimulus(entity, world)
                    and get_actionable_stimulus(entity, world) is not None
                    else 0.0
                )
                if _recent_nearby_speech.get("addressed"):
                    _reply_weight = 0.78
                elif _ticks_since_speak >= _speak_window:
                    _reply_weight = 0.58 + _stim_boost
                elif _ticks_since_speak >= max(1, (_min_speak_gap or 1) - 1):
                    _reply_weight = 0.38 + _stim_boost
                else:
                    _reply_weight = 0.18 + _stim_boost
                _add(
                    SemanticAction(
                        verb=ActionType.SPEAK,
                        actor=eid,
                        target=_reply_target,
                        intent=IntentBlock(
                            rationale="",
                            manner=_reply_snip or "conversational",
                            desired_outcome=["respond"],
                        ),
                        raw_input="[npc_auto:direct_reply]",
                    ),
                    f'respond directly to {_rns["speaker_name"]}: "{_reply_snip}"',
                    _reply_weight,
                )

        # Comment on a notable world event (always a rich option if available)
        recent_event = self._recent_notable_event(world)
        if recent_event and nearby:
            _add(
                SemanticAction(
                    verb=ActionType.SPEAK, actor=eid, target=nearby,
                    intent=IntentBlock(rationale=f"react to: {recent_event}",
                                       manner="reactive", desired_outcome=[]),
                    raw_input="[npc_auto:react_event]",
                ),
                f"react to {recent_event} with {nearby_name}", 0.78,
            )

        # ── Long-horizon plan step (DF/Qud-style job stack) ───────────────
        # Plans sit *above* per-tick reactive decisions: if the entity is
        # working on a multi-tick project (patrol, tend, deliver, work),
        # the plan's next step is offered as a very high-weight candidate
        # so the candidate scorer almost always picks it over routine
        # reactive options.  Combat / flee / low-stamina branches above
        # already short-circuit before this code runs, so an interrupt
        # condition will pre-empt the plan automatically — and
        # ``check_interrupt`` will pause the plan itself so it survives
        # to be resumed once safe.
        # Planner state was already bookkept at the top of generate_candidates
        # (check_interrupt, try_resume, propose_plan).  Here we only emit
        # the plan's next-step candidate into the menu.
        from . import npc_planner

        plan_candidate = npc_planner.next_step_action(entity, world)
        if plan_candidate is not None:
            _action, _label, _weight = plan_candidate
            _add(_action, _label, _weight)

        # ── Goal-driven initiative ─────────────────────────────────────────
        from .npc_goal_actions import goal_driven_candidates

        for action, label, weight in goal_driven_candidates(entity, world):
            _add(action, label, weight)

        entity_goals: list[str] = list(getattr(entity, "goals", None) or [])
        if entity_goals and nearby:
            active_goal = entity_goals[0]
            _add(
                SemanticAction(
                    verb=ActionType.SPEAK, actor=eid, target=nearby,
                    intent=IntentBlock(
                        rationale=active_goal,
                        manner="purposeful",
                        desired_outcome=["advance_goal"],
                    ),
                    raw_input="[npc_auto:goal_driven]",
                ),
                f'pursue goal: "{active_goal[:60]}" by speaking to {nearby_name}', 0.82,
            )

        # Wandering: if the NPC hasn't moved recently, nudge them to a new spot
        # so the room feels alive (not everyone frozen in place)
        if last_action_verb not in ("move",) and not entity.waypoints:
            import random as _rng
            nudge_x = entity.position.x + _rng.randint(-3, 3)
            nudge_y = entity.position.y + _rng.randint(-3, 3)
            from .schemas import Coord as _Coord2
            nudge = _Coord2(
                x=max(0, min(nudge_x, grid.width - 1)),
                y=max(0, min(nudge_y, grid.height - 1)),
            )
            if grid.is_passable(nudge) and nudge != entity.position:
                _add(
                    SemanticAction(verb=ActionType.MOVE, actor=eid, target=nudge,
                                   raw_input="[npc_auto:wander]"),
                    "shift to a different spot in the room — restless energy", 0.38,
                )

        # Portable ground loot only — not fixtures (hearth, bar, etc.).
        interesting_oid = self._find_interesting_object(entity, grid)
        if interesting_oid is not None:
            obj = grid.objects.get(interesting_oid)
            if obj and obj.position is not None:
                blocked = {"fixture", "heat_source", "furniture", "structure"}
                dist_obj = entity.position.manhattan(obj.position)
                if (
                    dist_obj <= 1
                    and not blocked.intersection(obj.tags)
                    and interesting_oid not in entity.inventory
                ):
                    _add(
                        SemanticAction(
                            verb=ActionType.TAKE,
                            actor=eid,
                            target=interesting_oid,
                            raw_input="[npc_auto:take_object]",
                        ),
                        f"pick up {obj.name}", 0.52,
                    )

        # Investigate noise source
        curiosity_dest = self._curiosity_destination(entity, world)
        if curiosity_dest is not None and not candidates:
            _add(
                SemanticAction(verb=ActionType.MOVE, actor=eid, target=curiosity_dest,
                               raw_input="[npc_auto:investigate_noise]"),
                f"move toward the source of the noise at ({curiosity_dest.x},{curiosity_dest.y})", 0.55,
            )

        # Rest if tired
        if stamina < 60.0 and entity.alertness in (AlertnessLevel.UNAWARE, AlertnessLevel.LOW):
            _add(SemanticAction(verb=ActionType.REST, actor=eid, raw_input="[npc_auto:mild_fatigue]"),
                 "take a moment to catch your breath", 0.20)

        # Wait (always last)
        _add(SemanticAction(verb=ActionType.WAIT, actor=eid, raw_input="[npc_auto:idle]"),
             "stand around and take in the surroundings", 0.10)

        if not candidates:
            candidates.append((SemanticAction(verb=ActionType.WAIT, actor=eid,
                                              raw_input="[npc_auto:idle_fallback]"),
                                "wait", 1.0))

        speak_verbs_local = frozenset({
            "speak", "say", "tell", "ask", "shout", "whisper", "yell",
            "greet", "challenge", "warn",
        })
        speak_pool = [
            c for c in candidates
            if str(c[0].verb).lower().split(".")[-1] in speak_verbs_local
        ]
        candidates.sort(key=lambda c: c[2], reverse=True)
        trimmed = candidates[:max_candidates]
        if speak_pool and not any(
            str(c[0].verb).lower().split(".")[-1] in speak_verbs_local for c in trimmed
        ):
            best_speak = max(speak_pool, key=lambda c: c[2])
            if len(trimmed) >= max_candidates:
                trimmed = trimmed[: max_candidates - 1]
            trimmed.append(best_speak)
        return trimmed

    # ── Candidate helpers ─────────────────────────────────────────────────

    def _recent_notable_event(self, world: WorldState) -> Optional[str]:
        """Return a short description of the most recent notable world event."""
        for event in reversed(world.event_log[-10:]):
            v = event.action.verb
            if v in (ActionType.ATTACK, ActionType.STEAL, ActionType.CAST):
                actor = world.spatial.entities.get(event.action.actor)
                return f"{actor.name if actor else 'someone'} {v}ed"
            for t in event.transitions:
                if t.kind == TransitionKind.TILE_MARKED and t.payload.get("mark"):
                    return f"something was left here: {t.payload['mark'][:40]}"
                if t.kind.value == "dialogue_spoken" and t.payload.get("text"):
                    txt = t.payload["text"]
                    return f'someone said: "{txt[:40]}"'
        return None

    @staticmethod
    def _coord_for_world_fact(
        observation: dict,
        world: WorldState,
    ) -> Optional["Coord"]:
        """Resolve a tile coordinate for a world_fact observation."""
        fact_id = observation.get("event_id")
        if fact_id:
            for fact in world.world_facts:
                if fact.fact_id == fact_id and fact.subject_id:
                    parts = fact.subject_id.split(",")
                    if len(parts) >= 2:
                        try:
                            from .schemas import Coord

                            return Coord(
                                x=int(parts[0]),
                                y=int(parts[1]),
                                z=int(parts[2]) if len(parts) > 2 else 0,
                            )
                        except ValueError:
                            pass
        return None

    def _find_interesting_object(
        self, entity: EntityState, grid: "SpatialGrid"
    ) -> Optional["ObjectId"]:
        """Find the nearest unowned ground object within sight range."""
        best: Optional[tuple[int, "ObjectId"]] = None
        for oid, obj in grid.objects.items():
            if obj.position is None:
                continue
            # Skip objects in any entity's inventory
            owned = any(oid in e.inventory for e in grid.entities.values())
            if owned:
                continue
            dist = entity.position.manhattan(obj.position)
            if dist <= entity.sight_range:
                if best is None or dist < best[0]:
                    best = (dist, oid)
        return best[1] if best else None

    def _curiosity_destination(
        self, entity: EntityState, world: WorldState
    ) -> Optional["Coord"]:
        """Return a tile the NPC might wander toward out of curiosity."""
        grid = world.spatial
        # Move toward a durable mark the NPC can see
        for coord in visible_from(grid, entity.position, entity.sight_range):
            tile = grid.tile_at(coord)
            if tile.marks and entity.position.manhattan(coord) > 1:
                if grid.is_passable(coord):
                    return coord
        # Move toward the most recently heard loud-speech position
        for event in reversed(world.event_log[-5:]):
            for t in event.transitions:
                if t.kind == TransitionKind.TILE_MARKED and not t.payload.get("remove"):
                    from .schemas import Coord as _C

                    dest = _C(
                        x=int(t.payload["x"]),
                        y=int(t.payload["y"]),
                        z=int(t.payload.get("z", 0)),
                    )
                    if dest != entity.position and grid.is_passable(dest):
                        return dest
                if t.kind.value == "dialogue_spoken" and t.payload.get("loud"):
                    src = t.payload.get("source_pos", {})
                    if src:
                        from .schemas import Coord as _C
                        dest = _C(x=int(src["x"]), y=int(src["y"]))
                        if dest != entity.position and grid.is_passable(dest):
                            return dest
        return None

    # ── Contact response ───────────────────────────────────────────────────
    def _respond_to_contact(
        self,
        entity: EntityState,
        contact: "_ContactRecord",
        world: WorldState,
    ) -> SemanticAction:
        """
        Choose a response to a contact that just happened to this entity.

        The dispatch is **reactive vs considered** based on the entity's
        alertness AT THE TIME of the contact (carried in the
        `surprise` flag on the event). Surprised targets fall through to
        a small set of instinctive actions; un-surprised targets get the
        full action space.

        The engine never authors specific manners of response — it
        chooses an ActionType, and the narrator describes the prose.
        """
        actor = world.spatial.entities.get(contact.actor)

        # ── REACTIVE: caught off-guard, limited options ───────────────────
        if contact.surprise:
            if contact.consent == ConsentState.WELCOMED:
                # A pleasant surprise — just observe the actor.
                return SemanticAction(
                    verb=ActionType.OBSERVE,
                    actor=entity.entity_id,
                    target=contact.actor,
                    raw_input="[npc_auto:contact_pleasant_surprise]",
                )
            if contact.consent == ConsentState.HOSTILE:
                # Hostile contact while unaware → instinctive flee.
                return SemanticAction(
                    verb=ActionType.FLEE,
                    actor=entity.entity_id,
                    raw_input="[npc_auto:contact_panic]",
                )
            # UNWELCOMED / REFUSED: push away — modelled as low-aggression
            # CONTACT back, which the compiler will resolve based on actor's
            # state (likely UNWELCOMED back, no damage).
            return SemanticAction(
                verb=ActionType.CONTACT,
                actor=entity.entity_id,
                target=contact.actor,
                style=StyleBlock(aggression=40, emotional_tone="firm"),
                intent=IntentBlock(
                    manner="push away",
                    desired_outcome=["disengage"],
                ),
                raw_input="[npc_auto:contact_reflex]",
            )

        # ── CONSIDERED: target had time to think ──────────────────────────
        if contact.consent == ConsentState.WELCOMED:
            # Reciprocate (gentle CONTACT back) — strengthens the bond.
            return SemanticAction(
                verb=ActionType.CONTACT,
                actor=entity.entity_id,
                target=contact.actor,
                style=StyleBlock(aggression=0, emotional_tone="affectionate"),
                intent=IntentBlock(
                    manner="reciprocate",
                    desired_outcome=["closeness"],
                ),
                raw_input="[npc_auto:contact_reciprocate]",
            )

        if contact.consent == ConsentState.HOSTILE:
            # Treat as an assault: full threat-response path.
            return self._respond_to_threat(entity, contact.actor, world)

        # REFUSED — verbally rebuff. Higher-tier social action since the
        # target had time to compose themselves.
        if contact.consent == ConsentState.REFUSED:
            return SemanticAction(
                verb=ActionType.THREATEN,
                actor=entity.entity_id,
                target=contact.actor,
                intent=IntentBlock(
                    desired_outcome=["enforce_boundary"],
                    rationale="warn the actor away after refused contact",
                ),
                style=StyleBlock(aggression=60, emotional_tone="stern"),
                raw_input="[npc_auto:contact_rebuff]",
            )

        # UNWELCOMED — speak a verbal protest; not yet escalating.
        return SemanticAction(
            verb=ActionType.SPEAK,
            actor=entity.entity_id,
            target=contact.actor,
            intent=IntentBlock(
                rationale="protest unwelcome contact",
                desired_outcome=["disengage"],
            ),
            style=StyleBlock(aggression=30, emotional_tone="displeased"),
            raw_input="[npc_auto:contact_protest]",
        )

    # ── Threat response ────────────────────────────────────────────────────
    def _respond_to_threat(
        self, entity: EntityState, attacker_id: EntityId, world: WorldState
    ) -> SemanticAction:
        attacker = world.spatial.entities.get(attacker_id)
        if attacker is None:
            # Attacker gone — fall back to watchful
            return SemanticAction(
                verb=ActionType.WAIT,
                actor=entity.entity_id,
                raw_input="[npc_auto:threat_resolved]",
            )

        # Flee if health is low
        if entity.health < entity.max_health * self.flee_health_fraction:
            return SemanticAction(
                verb=ActionType.FLEE,
                actor=entity.entity_id,
                raw_input="[npc_auto:wounded]",
            )

        dist = entity.position.manhattan(attacker.position)
        if dist <= 1:
            # Adjacent → attack
            return SemanticAction(
                verb=ActionType.ATTACK,
                actor=entity.entity_id,
                target=attacker_id,
                raw_input="[npc_auto:retaliate]",
                intent=IntentBlock(desired_outcome=["neutralize_threat"]),
                style=StyleBlock(aggression=90, emotional_tone="furious"),
            )

        # Not adjacent → close distance (EntityId-targeted MOVE; compiler
        # resolves to an adjacent passable tile).
        return SemanticAction(
            verb=ActionType.MOVE,
            actor=entity.entity_id,
            target=attacker_id,
            raw_input="[npc_auto:close_distance]",
        )

    # ── Engage (alertness-driven, no recent attacker) ──────────────────────
    def _add_magic_duel_candidates(
        self,
        entity: EntityState,
        world: WorldState,
        opponent_id: EntityId,
        _add,
    ) -> None:
        """Offer inscribed spells as CAST actions for 1v1 magic duel worlds."""
        vocab = world.config.spell_vocab
        if vocab is None or not entity.inscribed_spells:
            return

        grid = world.spatial
        eid = entity.entity_id
        opponent = grid.entities.get(opponent_id)
        if opponent is None or not opponent.alive:
            return

        mana = float(entity.mana or entity.meta.get("mana", 0))
        health_frac = entity.health / max(1, entity.max_health)
        opp_on_fire = "on_fire" in opponent.tags

        for spell_name, program in entity.inscribed_spells.items():
            cost = _estimate_spell_mana(program, vocab)
            if cost > 0 and mana < cost:
                continue

            upper = program.upper().replace("\n", " ")
            if "SELF" in upper:
                cast_target = eid
                label_target = "SELF"
            else:
                cast_target = opponent_id
                label_target = "TARGET"

            weight = 0.80
            if "HEAT" in upper or "CHILL" in upper:
                weight = 0.90
            if "IGNITE" in upper:
                weight = 0.84 if "flammable" in opponent.tags else 0.45
            if "MEND" in upper and health_frac > 0.55:
                continue
            if "MEND" in upper:
                weight = 0.97
            if "QUENCH" in upper and not opp_on_fire:
                continue
            if "QUENCH" in upper:
                weight = 0.92

            program_line = program.strip().replace("\n", ", ")
            _add(
                SemanticAction(
                    verb=ActionType.CAST,
                    actor=eid,
                    target=cast_target,
                    intent=IntentBlock(
                        rationale=program,
                        manner=program_line,
                        desired_outcome=["defeat_opponent"],
                    ),
                    raw_input=f"[npc_auto:{program_line}]",
                ),
                f"{program_line} → {label_target}",
                weight,
            )

        if health_frac < 0.25:
            _add(
                SemanticAction(
                    verb=ActionType.FLEE,
                    actor=eid,
                    raw_input="[npc_auto:duel_retreat]",
                ),
                "retreat — badly wounded",
                0.55,
            )

        _add(
            SemanticAction(verb=ActionType.WAIT, actor=eid, raw_input="[npc_auto:duel_wait]"),
            "hold your ground and gather mana",
            0.12,
        )

    def _engage(self, entity: EntityState, enemy_id: EntityId) -> SemanticAction:
        # Treat the visible enemy the same way as a fresh threat:
        # the same gates (low health → flee, adjacent → attack, else close)
        # apply. The world is needed for the distance check, so delegate.
        return SemanticAction(
            verb=ActionType.MOVE,
            actor=entity.entity_id,
            target=enemy_id,
            raw_input="[npc_auto:engage]",
        )

    def _patrol_waypoint(
        self,
        entity: EntityState,
    ) -> "Optional[SemanticAction]":
        """Return a MOVE action toward the next patrol waypoint, or None."""
        if not entity.waypoints:
            return None
        idx = entity.waypoint_index % len(entity.waypoints)
        current_wp = entity.waypoints[idx]
        # If already at the waypoint (or adjacent), advance to the next.
        if entity.position == current_wp or entity.position.manhattan(current_wp) <= 1:
            entity.waypoint_index = (entity.waypoint_index + 1) % len(entity.waypoints)
            idx = entity.waypoint_index
            current_wp = entity.waypoints[idx]
        return SemanticAction(
            verb=ActionType.MOVE,
            actor=entity.entity_id,
            target=current_wp,
            raw_input="[npc_auto:patrol]",
        )


# ─────────────────────────────────────────────────────────────────────────────
# State predicates (generic helpers — no hardcoded names, no English)
# ─────────────────────────────────────────────────────────────────────────────


class _ContactRecord:
    """Lightweight record of a recent CONTACT_INITIATED event."""

    __slots__ = ("actor", "consent", "surprise", "manner", "tick")

    def __init__(self, actor: EntityId, consent: ConsentState,
                 surprise: bool, manner: str, tick: int):
        self.actor = actor
        self.consent = consent
        self.surprise = surprise
        self.manner = manner
        self.tick = tick


def _most_recent_contact_against(
    victim: EntityId, world: WorldState, lookback: int
) -> _ContactRecord | None:
    """
    Return the most recent CONTACT_INITIATED event targeting `victim`
    within the lookback window AND not yet responded to, or None.

    "Already responded" means `victim` has acted toward `actor` in any
    event whose tick is strictly later than the contact event. This
    keeps a single contact from triggering a response every tick.
    """
    if not world.event_log:
        return None
    cutoff_tick = max(0, world.tick - lookback)
    for i in range(len(world.event_log) - 1, -1, -1):
        event = world.event_log[i]
        if event.tick < cutoff_tick:
            break
        for t in event.transitions:
            if t.kind != TransitionKind.CONTACT_INITIATED:
                continue
            if t.payload.get("target") != victim:
                continue
            actor = t.payload.get("actor")
            if actor == victim or actor is None:
                continue
            # Skip if victim already addressed this actor after the contact.
            if _has_acted_toward(world, victim, EntityId(actor), after_index=i):
                continue
            consent_raw = t.payload.get("consent_state", "unwelcomed")
            try:
                consent = ConsentState(consent_raw)
            except ValueError:
                consent = ConsentState.UNWELCOMED
            return _ContactRecord(
                actor=EntityId(actor),
                consent=consent,
                surprise=bool(t.payload.get("surprise", False)),
                manner=str(t.payload.get("manner", "") or ""),
                tick=event.tick,
            )
    return None


def _has_acted_toward(
    world: WorldState,
    actor: EntityId,
    target: EntityId,
    *,
    after_index: int,
) -> bool:
    """True if `actor` has acted with `target` as target in any event
    after position `after_index` in the event log."""
    for j in range(after_index + 1, len(world.event_log)):
        ev = world.event_log[j]
        if ev.action.actor != actor:
            continue
        if ev.action.target == target:
            return True
    return False


def _recently_answered_contact(
    victim: EntityId, world: WorldState, lookback: int
) -> bool:
    """True when victim has already answered a recent contact from another actor."""
    if not world.event_log:
        return False
    cutoff_tick = max(0, world.tick - lookback)
    for i in range(len(world.event_log) - 1, -1, -1):
        event = world.event_log[i]
        if event.tick < cutoff_tick:
            break
        for t in event.transitions:
            if t.kind != TransitionKind.CONTACT_INITIATED:
                continue
            if t.payload.get("target") != victim:
                continue
            actor = t.payload.get("actor")
            if actor and actor != victim and _has_acted_toward(
                world, victim, EntityId(actor), after_index=i
            ):
                return True
    return False


def _most_recent_attacker(
    victim: EntityId, world: WorldState, lookback: int
) -> EntityId | None:
    """
    Return the entity_id of the most recent attacker of `victim` within the
    last `lookback` ticks, or None.

    "Attacker" = source actor of any ENTITY_HEALTH_CHANGED transition
    targeting `victim` with a negative delta.
    """
    if not world.event_log:
        return None
    cutoff_tick = max(0, world.tick - lookback)
    for event in reversed(world.event_log):
        if event.tick < cutoff_tick:
            break
        for t in event.transitions:
            if t.kind != TransitionKind.ENTITY_HEALTH_CHANGED:
                continue
            if t.payload.get("entity_id") != victim:
                continue
            if t.payload.get("delta", 0) >= 0:
                continue
            actor = t.payload.get("actor")
            if actor and actor != victim:
                return EntityId(actor)
    return None


def _estimate_spell_mana(source: str, vocab) -> float:
    """Rough mana cost from COST lines and operation mana_formula entries."""
    from .spell_runtime import SpellParser, _eval_formula, _lex
    from .spell_types import CostStatement, EffectStatement

    try:
        program = SpellParser(_lex(source), source=source).parse()
    except Exception:
        return 5.0

    total = 0.0
    for stmt in program.statements:
        if isinstance(stmt, CostStatement) and stmt.resource.lower() == "mana":
            total += stmt.amount
        elif isinstance(stmt, EffectStatement):
            op = vocab.operations.get(stmt.op_name)
            if op is None:
                continue
            bound: dict = {}
            for i, param in enumerate(op.args):
                if i < len(stmt.args):
                    bound[param] = stmt.args[i]
            if stmt.args:
                bound["n"] = stmt.args[0]
            if len(stmt.args) > 1:
                bound["m"] = stmt.args[1]
            total += _eval_formula(op.mana_formula, bound)
    return total


def _magic_duel_opponent(
    entity: EntityState,
    world: WorldState,
    threat_lookback_ticks: int,
) -> EntityId | None:
    """Visible rival for magic duel — hostile edge or nearest other NPC."""
    enemy = _first_visible_enemy(entity, world, threat_lookback_ticks)
    if enemy is not None:
        return enemy
    if not world.meta.get("magic_duel"):
        return None
    return _first_visible_entity(entity, world)


def _first_visible_enemy(
    entity: EntityState,
    world: WorldState,
    threat_lookback_ticks: int,
) -> EntityId | None:
    """
    Return the entity_id of the first visible entity that this NPC currently
    considers an enemy.

    Enemy = visible AND any of:
      - relational graph contains a hostile-class edge from us to them
      - they have recently attacked us (within threat_lookback_ticks)
    """
    grid = world.spatial
    visible = visible_from(grid, entity.position, entity.sight_range)
    if not visible:
        return None
    hostile_ids = _hostile_edge_targets(entity.entity_id, world)
    recent_attacker = _most_recent_attacker(
        entity.entity_id, world, threat_lookback_ticks
    )
    for other_id, other in grid.entities.items():
        if other_id == entity.entity_id:
            continue
        if other.position not in visible:
            continue
        if other_id in hostile_ids:
            return other_id
        if recent_attacker == other_id:
            return other_id
    return None


def _first_visible_entity(
    entity: EntityState, world: WorldState
) -> EntityId | None:
    """
    Return the nearest living entity within sight_range, direction-agnostic.

    For candidate generation (not the projection firewall), facing direction
    should not matter — an NPC can always turn to face someone nearby.
    Using omnidirectional distance prevents NPCs from being stuck in
    null-target observe loops just because someone is behind them.
    """
    grid = world.spatial
    best_id: EntityId | None = None
    best_dist = entity.sight_range + 1
    for other_id, other in grid.entities.items():
        if other_id == entity.entity_id or not other.alive:
            continue
        dist = entity.position.manhattan(other.position)
        if dist <= entity.sight_range and dist < best_dist:
            best_dist = dist
            best_id = other_id
    return best_id


def _hostile_edge_targets(actor: EntityId, world: WorldState) -> set[EntityId]:
    """
    Return entity_ids toward which `actor` holds a hostile-class relational
    edge (ENEMY_OF, FEARS, DISTRUSTS).
    """
    hostile_kinds = {EdgeKind.ENEMY_OF, EdgeKind.FEARS, EdgeKind.DISTRUSTS}
    out: set[EntityId] = set()
    edges_from_actor = world.relational.edges.get(actor, {})
    for target_id, edge_list in edges_from_actor.items():
        for edge in edge_list:
            if edge.kind in hostile_kinds:
                out.add(EntityId(target_id))
                break
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Character sheet — pure projection of canonical state for the LM-driven NPC
# policy. No LM call here; this is just summarization.
# ─────────────────────────────────────────────────────────────────────────────


# How many recent events to scan when building an NPC's perception summary.
_PERCEPTION_LOOKBACK_EVENTS = 12


def format_social_aims_for_sheet(
    entity: EntityState, world: WorldState
) -> list[str]:
    """Relational aims with canonical target ids and in-sight markers."""
    grid = world.spatial
    lines: list[str] = []
    for sa in entity.social_aims:
        tid = sa.target_entity_id
        target_ent = grid.entities.get(tid)
        if target_ent is None or not target_ent.alive:
            continue
        name = target_ent.name
        dist = entity.position.manhattan(target_ent.position)
        in_sight = dist <= entity.sight_range
        sight = "in sight" if in_sight else f"{dist} tiles away"
        lines.append(
            f"  - {name} (target id \"{tid}\", {sight}): {sa.aim}"
        )
    return lines


def build_npc_character_sheet(
    entity: EntityState, world: WorldState
) -> NpcCharacterSheet:
    """
    Build an `NpcCharacterSheet` for `entity` from canonical state.

    The sheet is what the LM-driven NPC policy hands to the adapter as
    the NPC's self-knowledge: role, personality, drive, current mood,
    outbound relationships, and a short list of recent perceptions
    (events targeting or witnessed by this NPC).

    The recent-perceptions extractor walks `world.event_log` and picks
    any event whose actor / target / witness list mentions `entity`.
    Crucially it inspects EVERY transition payload — not just kernel
    transitions — so open-verb interactions recorded by the freeform
    compiler (e.g. `kiss`, `tease`, `embrace` flowing through
    DIALOGUE_SPOKEN) are surfaced. This is the architectural fix for
    the kiss-blindness gap.
    """
    grid = world.spatial
    npc_id = entity.entity_id

    relations: list[str] = []
    for target_id, edges in world.relational.edges.get(npc_id, {}).items():
        target_name = _node_display_name(target_id, world)
        for e in edges:
            relations.append(
                f"{e.kind.value} {target_name} (weight={e.weight:.2f})"
            )

    # Short Markov window: what this NPC could observe (LOS) + direct involvement.
    window = memory_tick_window(world)
    cutoff = world.tick - window
    recent: list[str] = list(format_observations_for_sheet(entity, world))
    for ev in world.event_log[-_PERCEPTION_LOOKBACK_EVENTS:]:
        if ev.tick < cutoff:
            continue
        line = _perception_line(ev, npc_id, world)
        if line is not None and line not in recent:
            recent.append(line)
    recent = recent[-8:]

    health_fraction = (
        entity.health / entity.max_health if entity.max_health > 0 else 1.0
    )

    # Episode summaries from long-term memory (oldest first, max 5)
    episode_summaries = list(world.episode_memory.get(npc_id, []))

    from .world_adjudicator import facts_for_projection

    established_facts = facts_for_projection(world, EntityId(npc_id))

    from .memory_retrieval import memories_for_projection

    retrieved_memories, chronicle_memories = memories_for_projection(
        world, EntityId(npc_id), max_retrieved=4, max_chronicle=3,
    )

    # Active dialogue thread (multi-turn conversation context)
    active_dialogue: Optional[str] = None
    from .dialogue import format_thread_for_npc, get_active_thread_for
    thread = get_active_thread_for(world, npc_id)
    if thread:
        active_dialogue = format_thread_for_npc(thread, npc_id)

    # Determine what relational edges the NPC has toward any focal entity
    # for secrets disclosure: only share secrets with INTIMATE / ALLIED contacts.
    all_outbound_kinds = {
        e.kind
        for edges in world.relational.edges.get(npc_id, {}).values()
        for e in edges
    }
    from .schemas import EdgeKind
    disclose_secrets = bool(
        all_outbound_kinds & {EdgeKind.INTIMATE, EdgeKind.ALLY_OF}
    )

    stats = entity.stats or {}
    stamina = stats.get("stamina", 100.0)
    gold = stats.get("gold", 0.0)
    reputation = stats.get("reputation", 0.0)
    # Normalise gold to a 0–1 fraction relative to a "comfortable" 100 gold
    gold_fraction = min(1.0, gold / 100.0) if gold > 0 else 0.0

    from .npc_mind import load_mind

    mind = load_mind(entity)
    mind_bits: list[str] = []
    if mind.intent:
        mind_bits.append(f"intent={mind.intent}")
    if mind.attention_reason:
        mind_bits.append(f"attention={mind.attention_reason}")
    if mind.attention_target:
        mind_bits.append(f"focus_on={mind.attention_target}")
    mind_line = "; ".join(mind_bits)
    reflection = str(entity.meta.get("inner_thought", "") or "")
    if mind_line:
        reflection = f"{reflection} [{mind_line}]".strip() if reflection else f"[{mind_line}]"

    return NpcCharacterSheet(
        npc_id=npc_id,
        name=entity.name,
        role=entity.role,
        personality=entity.personality,
        drive=entity.drive,
        emotional_state=entity.emotional_state,
        alertness=entity.alertness,
        social_openness=entity.social_openness,
        health_fraction=health_fraction,
        facing=entity.facing.value,
        relations=relations,
        recent_perceptions=recent,
        episode_summaries=episode_summaries,
        established_facts=established_facts,
        retrieved_memories=retrieved_memories,
        chronicle_memories=chronicle_memories,
        active_dialogue=active_dialogue,
        knowledge=list(entity.knowledge),
        goals=list(entity.goals),
        social_aims=format_social_aims_for_sheet(entity, world),
        secrets=list(entity.secrets) if disclose_secrets else [],
        occupation=entity.occupation,
        faction=entity.faction,
        stamina_fraction=round(stamina / 100.0, 2),
        gold_fraction=round(gold_fraction, 2),
        reputation=reputation,
        voice_lines=list(entity.voice_lines),
        recent_room_dialogue=_collect_recent_room_dialogue(world),
        director_hint=str(entity.meta.get("director_hint", "")),
        taboo_phrases=list(entity.meta.get("director_taboo", []) or []),
        held_beliefs=_beliefs_for_entity(entity, world),
        inner_reflection=reflection,
        focus_topic=str(entity.meta.get("focus_topic", "")),
    )


def _beliefs_for_entity(entity: EntityState, world: WorldState) -> list[str]:
    """Summarise BELIEVES_CLAIM edges held by this entity."""
    from .schemas import EdgeKind

    out: list[str] = []
    eid = str(entity.entity_id)
    for tgt, edges in world.relational.edges.get(eid, {}).items():
        for edge in edges:
            if edge.kind == EdgeKind.BELIEVES_CLAIM:
                claim = (edge.meta or {}).get("claim", "")
                if claim:
                    name = _node_display_name(tgt, world)
                    out.append(f"Believes about {name}: {claim[:80]}")
    return out[:5]


def _node_display_name(node_id: str, world: WorldState) -> str:
    """Resolve an entity / faction id to a human-readable name, with
    fallback to the bare id if no nicer name is on file."""
    ent = world.spatial.entities.get(EntityId(node_id))
    if ent is not None:
        return ent.name
    node = world.relational.nodes.get(node_id)
    if node is not None:
        return node.name
    return node_id


def _perception_line(ev, npc_id: EntityId, world: WorldState) -> Optional[str]:
    """
    Turn one Event into a one-line perception, if it involves `npc_id`.

    The NPC perceives an event when ANY of the following hold:
      • the NPC is the actor              ("you did X")
      • the NPC is the action's target    ("X did Y to you")  ← kiss path
      • the NPC is in event.witnesses     ("you saw X do Y")
      • any transition payload references the NPC as entity_id / target
        / from_entity / to_entity (covers health, alertness, emotion,
        and CONTACT_INITIATED).

    Returns None if the NPC is not involved.
    """
    action = ev.action
    actor_id = action.actor
    target = action.target if isinstance(action.target, str) else None
    witnesses = set(ev.witnesses or [])

    involved_as_actor = actor_id == npc_id
    involved_as_target = target == npc_id
    involved_as_witness = npc_id in witnesses and not involved_as_actor

    if not (involved_as_actor or involved_as_target or involved_as_witness):
        # Fallback: scan transition payloads for any reference to npc_id.
        # Catches CONTACT_INITIATED + emotion/alertness ricochets even
        # when the SemanticAction's target field was a Coord or None.
        ref_in_transition = any(
            _transition_references(t, npc_id) for t in ev.transitions
        )
        if not ref_in_transition:
            return None
        # Inferred involvement counts as a passive perception.
        involved_as_target = True

    actor_name = _node_display_name(str(actor_id), world)
    verb = action.verb

    if involved_as_actor:
        # The NPC's own past action — keep terse so it doesn't dominate
        # the prompt.
        return f"tick {ev.tick}: you {verb}ed"

    suffix = ""
    from .speech_utils import best_speech_line, is_real_dialogue

    spoken = ""
    for t in ev.transitions:
        if t.kind.value == "dialogue_spoken":
            spoken = (t.payload.get("text") or "").strip()
            break
    if not is_real_dialogue(spoken):
        spoken = best_speech_line(action)
    if is_real_dialogue(spoken):
        snip = spoken if len(spoken) <= 80 else spoken[:77] + "…"
        suffix = f' — "{snip}"'

    # Surprise tag pulled off CONTACT_INITIATED transitions when present.
    surprise_tag = ""
    for t in ev.transitions:
        if t.kind == TransitionKind.CONTACT_INITIATED and t.payload.get("target") == npc_id:
            if t.payload.get("surprise"):
                surprise_tag = " (surprise)"
            break

    if involved_as_target:
        return f"tick {ev.tick}: {actor_name} {verb}ed you{suffix}{surprise_tag}"
    # witness
    tgt_name = (
        _node_display_name(target, world) if target is not None else "no one"
    )
    return f"tick {ev.tick}: you saw {actor_name} {verb} {tgt_name}{suffix}"


def _transition_references(transition, npc_id: EntityId) -> bool:
    """True if a transition's payload mentions npc_id anywhere meaningful."""
    p = transition.payload
    for key in ("entity_id", "target", "actor", "from_entity", "to_entity"):
        v = p.get(key)
        if isinstance(v, str) and v == npc_id:
            return True
    return False


