"""
World clock — the autonomic loop that fires pack-declared ambient events
between actor turns.

Design notes:
  • The clock is NOT an actor. It does not produce a SemanticAction
    that the LM or Compiler interprets. It runs ambient events from
    `world.config.ambient_events`, validates their effects through the
    same `_validate_proposal` pipeline used for LM-proposed effects,
    and writes a synthetic Event whose `action.actor == "system"` and
    `action.verb == "ambient"`. This keeps the event log
    architecturally homogeneous: every change to the world is an
    Event with traceable transitions, and replay walks the log
    indifferent to who or what produced each Event.

  • Probability rolls are deterministic, seeded from
    (world.rng_seed, event.id, world.tick). Replay does not re-roll —
    `apply_transitions` is invoked directly on the recorded
    transitions — so the world is reproducible from the log alone.

  • The world clock runs ONCE per game-loop step, BEFORE the NPC turn,
    so NPCs can react to ambient events in the same tick (e.g., a
    guard glances up when the bell tolls).
"""

from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .lm_adapter import LMAdapter

from .compiler import _seeded_float, _validate_proposals, apply_transitions
from .needs import needs_tick
from .relational import apply_event_to_graph, decay_edges
from .schemas import (
    ActionType,
    AmbientEvent,
    EdgeKind,
    EntityId,
    Event,
    ScheduledEffectKind,
    SemanticAction,
    TimeOfDay,
    Transition,
    TransitionKind,
    WorldState,
    new_event_id,
)

logger = logging.getLogger(__name__)


# Reserved actor id used for ambient events. Not a real entity in the
# spatial grid — the engine treats this id as the "world itself".
SYSTEM_ACTOR: EntityId = EntityId("system")

# Poison damage dealt per tick (before armor).
_POISON_DAMAGE_PER_TICK = 5


def _tick_entity_conditions(world: WorldState) -> None:
    """
    Decrement all active condition timers by 1 and apply side effects.

    Called once per world_tick() invocation, BEFORE ambient events fire,
    so that expired conditions don't affect the same tick they expire on.

    Side effects per condition:
      - "poisoned"  → deal _POISON_DAMAGE_PER_TICK HP each tick (uses
                       ENTITY_HEALTH_CHANGED so the death hook fires if needed)
      - "exhausted" → reduce speed to 1 for the duration (we just log it;
                       the attribute is checked in _compile_move)
      - "stunned"   → no movement/attack (enforced in compile_action's
                       condition gate)
    """
    grid = world.spatial
    for entity in list(grid.entities.values()):
        if not entity.alive:
            continue
        expired = []
        for condition, ticks_left in list(entity.conditions.items()):
            if ticks_left <= 1:
                expired.append(condition)
            else:
                entity.conditions[condition] = ticks_left - 1

            # Per-tick poison damage
            if condition == "poisoned":
                entity.health = max(0, entity.health - _POISON_DAMAGE_PER_TICK)
                if entity.health <= 0 and entity.alive:
                    from .compiler import apply_transitions
                    from .schemas import Transition, TransitionKind
                    apply_transitions(world, [
                        Transition(
                            kind=TransitionKind.ENTITY_HEALTH_CHANGED,
                            payload={
                                "entity_id": str(entity.entity_id),
                                "delta": 0,  # already applied above
                                "cause": "poison",
                            },
                        )
                    ])
            # Exhausted: halve speed (restore when it expires)
            if condition == "exhausted":
                entity.attributes["speed"] = max(1, entity.attributes.get("speed", 4) // 2)

        for cond in expired:
            entity.conditions.pop(cond, None)
            # Restore speed when exhaustion ends
            if cond == "exhausted":
                entity.attributes.pop("speed", None)  # let default (4) apply again


_DECAY_KINDS = {
    EdgeKind.REMEMBERS_EVENT,
    EdgeKind.WITNESSED,
    EdgeKind.BELIEVES_CLAIM,
    EdgeKind.OVERHEARD,
}
# Decay runs every N ticks to avoid O(edges) cost on every tick
_DECAY_EVERY_TICKS = 5
# Faction economy tick interval (ticks between resource changes)
_FACTION_ECONOMY_EVERY_TICKS = 10


def _process_scheduled_effects(
    world: WorldState,
) -> tuple[list[Transition], list[str], list[Event]]:
    """
    Fire adjudication-scheduled effects whose ``fire_tick`` matches now.

    Returns applied transitions, narration lines, and events for surfacing
    in the REPL / ambient feed.
    """
    if not world.scheduled_effects:
        return [], [], []

    remaining: list = []
    applied: list[Transition] = []
    narrations: list[str] = []
    fired_events: list[Event] = []
    tick = world.tick

    for effect in world.scheduled_effects:
        if effect.fire_tick != tick:
            remaining.append(effect)
            continue

        actor_id = effect.source_actor or str(SYSTEM_ACTOR)
        stub_action = SemanticAction(
            verb=ActionType.SYMBOLIC,
            actor=EntityId(actor_id),
            raw_input=effect.source_intent or "[scheduled_effect]",
        )

        if effect.kind == ScheduledEffectKind.NARRATION and effect.narration:
            narrations.append(effect.narration)
            evt = Event(
                event_id=new_event_id(),
                tick=tick,
                action=SemanticAction(
                    verb=ActionType.AMBIENT,
                    actor=SYSTEM_ACTOR,
                    raw_input=f"[scheduled:{effect.effect_id}]",
                ),
                transitions=[],
                witnesses=[],
                narrative_hint=effect.narration,
            )
            world.event_log.append(evt)
            fired_events.append(evt)
            continue

        if effect.transitions:
            validated = _validate_proposals(effect.transitions, world, stub_action)
            if validated:
                apply_transitions(world, validated)
                applied.extend(validated)
                evt = Event(
                    event_id=new_event_id(),
                    tick=tick,
                    action=SemanticAction(
                        verb=ActionType.AMBIENT,
                        actor=SYSTEM_ACTOR,
                        raw_input=f"[scheduled:{effect.effect_id}]",
                    ),
                    transitions=validated,
                    witnesses=[],
                    narrative_hint=effect.narration or effect.rationale,
                )
                world.event_log.append(evt)
                apply_event_to_graph(world.relational, evt, tick)
                fired_events.append(evt)
                if effect.narration:
                    narrations.append(effect.narration)
        elif effect.narration:
            narrations.append(effect.narration)
            evt = Event(
                event_id=new_event_id(),
                tick=tick,
                action=SemanticAction(
                    verb=ActionType.AMBIENT,
                    actor=SYSTEM_ACTOR,
                    raw_input=f"[scheduled:{effect.effect_id}]",
                ),
                transitions=[],
                witnesses=[],
                narrative_hint=effect.narration,
            )
            world.event_log.append(evt)
            fired_events.append(evt)

    world.scheduled_effects = remaining
    return applied, narrations, fired_events


def _faction_economy_tick(world: WorldState) -> None:
    """
    Tick faction resources: each faction's gold and food decrease by upkeep,
    and trade edges between factions transfer resources.  When a shortage
    occurs, a goal is synthesised for member NPCs via world.meta.
    """
    factions = world.meta.get("factions", {})
    if not factions:
        return
    for faction_id, faction_data in factions.items():
        resources = faction_data.setdefault("resources", {})
        upkeep = faction_data.get("upkeep", {"gold": 2.0, "food": 1.5})
        for resource, cost in upkeep.items():
            current = resources.get(resource, 100.0)
            resources[resource] = max(0.0, current - cost)
        # Flag shortage in meta so NPC goal synthesis can pick it up
        shortage_flags = faction_data.setdefault("shortages", [])
        shortage_flags.clear()
        for resource, amount in resources.items():
            if amount < 20.0:
                shortage_flags.append(resource)
        if shortage_flags:
            logger.debug(
                "Faction %s has shortages: %s", faction_id, shortage_flags
            )


def world_tick(world: WorldState, lm_adapter: Optional[LMAdapter] = None) -> list[Event]:
    """
    Evaluate the pack's ambient events for the current tick and fire
    those whose triggers pass.

    Returns the list of `Event`s that were actually fired and appended
    to the canonical event log (empty if the pack has no ambient
    events or none triggered this tick). The caller can use this list
    to surface narration in the REPL.

    Side effects: appends to `world.event_log`, applies all validated
    transitions to canonical state, and updates the relational graph.
    """
    # Tick down conditions for every entity and apply poison damage.
    _tick_entity_conditions(world)

    # ── Scheduled adjudication effects ─────────────────────────────────
    sched_transitions, sched_narrations, sched_events = _process_scheduled_effects(world)
    clock_transitions: list[Transition] = list(sched_transitions)

    # ── Needs decay ────────────────────────────────────────────────────
    need_transitions = needs_tick(world)
    if need_transitions:
        apply_transitions(world, need_transitions)
        clock_transitions.extend(need_transitions)

    from .bodily import bodily_tick

    bodily_transitions = bodily_tick(world)
    if bodily_transitions:
        apply_transitions(world, bodily_transitions)
        clock_transitions.extend(bodily_transitions)

    from .fields import field_tick

    contamination_transitions = field_tick(world)
    if contamination_transitions:
        apply_transitions(world, contamination_transitions)
        clock_transitions.extend(contamination_transitions)

    # ── Embodied drift: intoxication wears off, moods relax, candles burn ─
    drift_transitions = _world_drift_tick(world)
    if drift_transitions:
        apply_transitions(world, drift_transitions)
        clock_transitions.extend(drift_transitions)

    from .combat_state import combat_tick, stealth_tick

    combat_transitions = combat_tick(world)
    if combat_transitions:
        apply_transitions(world, combat_transitions)
        clock_transitions.extend(combat_transitions)

    stealth_transitions = stealth_tick(world)
    if stealth_transitions:
        apply_transitions(world, stealth_transitions)
        clock_transitions.extend(stealth_transitions)

    if clock_transitions:
        world.event_log.append(Event(
            event_id=new_event_id(),
            tick=world.tick,
            action=SemanticAction(
                verb=ActionType.AMBIENT,
                actor=SYSTEM_ACTOR,
                raw_input="[clock_tick]",
            ),
            transitions=clock_transitions,
            witnesses=[],
            narrative_hint="[system: clock tick]",
        ))

    # ── Relational memory decay ────────────────────────────────────────
    if world.tick % _DECAY_EVERY_TICKS == 0:
        removed = decay_edges(
            world.relational,
            world.tick,
            decay_rate=0.02,
            min_weight_to_remove=0.08,
            kinds_to_decay=_DECAY_KINDS,
        )
        if removed:
            logger.debug("Memory decay: removed %d stale edges at tick %d", removed, world.tick)

    # ── Faction economy ────────────────────────────────────────────────
    if world.tick % _FACTION_ECONOMY_EVERY_TICKS == 0:
        _faction_economy_tick(world)

    from .economy import economy_tick

    econ_transitions = economy_tick(world)
    if econ_transitions:
        apply_transitions(world, econ_transitions)
        clock_transitions.extend(econ_transitions)

    if sched_narrations:
        for line in sched_narrations:
            logger.info("Scheduled effect: %s", line)

    fired: list[Event] = list(sched_events)
    for ambient in world.config.ambient_events:
        event = _evaluate_and_fire(ambient, world, lm_adapter=lm_adapter)
        if event is not None:
            fired.append(event)
    return fired


def force_ambient_by_id(
    world: WorldState, event_id: str, lm_adapter: Optional[LMAdapter] = None
) -> Optional[Event]:
    """
    Fire a pack ambient event by id, bypassing probability rolls.

    Used by scenario beats and scripted pressure triggers.  Returns None if
    the id is unknown.
    """
    for ambient in world.config.ambient_events:
        if ambient.id == event_id:
            return _fire(ambient, world, lm_adapter=lm_adapter)
    return None


def _evaluate_and_fire(
    ambient: AmbientEvent, world: WorldState, lm_adapter: Optional[LMAdapter] = None
) -> Optional[Event]:
    """Evaluate one ambient event's trigger and, if it passes, fire it."""
    trigger = ambient.trigger

    if trigger.every_ticks is not None:
        if trigger.every_ticks <= 0 or world.tick % trigger.every_ticks != 0:
            return None

    if trigger.time_of_day is not None:
        current = world.config.clock.time_of_day(world.tick)
        if current != trigger.time_of_day:
            return None

    prob = trigger.probability
    from .pressure_eval import ambient_probability_multiplier

    prob = min(1.0, prob * ambient_probability_multiplier(world, ambient.id))
    if prob < 1.0:
        roll_seed = f"ambient:{world.rng_seed}:{ambient.id}:{world.tick}"
        if _seeded_float(roll_seed) >= prob:
            return None

    return _fire(ambient, world, lm_adapter=lm_adapter)


def _fire(
    ambient: AmbientEvent, world: WorldState, lm_adapter: Optional[LMAdapter] = None
) -> Event:
    """
    Build, log, and apply an ambient Event.

    The event's transitions are:
      1. A single AMBIENT_EVENT record carrying the event id, narrative,
         and time_of_day at the moment of firing.
      2. Whatever proposal-validated effects the pack declared (or enriched dynamically).

    The synthetic SemanticAction uses actor=SYSTEM_ACTOR and verb=ambient
    so existing event-log machinery (witnesses, replay, projection)
    can treat it uniformly with player/NPC events.
    """
    from .schemas import TransitionProposal, Coord

    scene_info = {
        "time_of_day": world.config.clock.time_of_day(world.tick).value,
        "ambient_temperature": 20.0,
        "active_characters": [],
        "recent_dialogue": [],
        "potential_coords": [],
    }

    if hasattr(world.config, "physics") and world.config.physics:
        scene_info["ambient_temperature"] = getattr(world.config.physics, "ambient_temp", 20.0)

    from .speech_utils import collect_recent_room_dialogue
    try:
        scene_info["recent_dialogue"] = collect_recent_room_dialogue(world)
    except Exception:
        scene_info["recent_dialogue"] = []

    visible_ids = []
    for eid, entity in world.spatial.entities.items():
        if not entity.alive:
            continue
        visible_ids.append(str(eid))
        scene_info["active_characters"].append({
            "id": str(eid),
            "name": entity.name,
            "pos": f"({entity.position.x},{entity.position.y},{entity.position.z})",
            "emotional_state": entity.emotional_state.value if hasattr(entity.emotional_state, "value") else str(entity.emotional_state)
        })
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                test_pos = Coord(x=entity.position.x + dx, y=entity.position.y + dy, z=entity.position.z)
                if world.spatial.is_in_bounds(test_pos):
                    coord_dict = {"x": test_pos.x, "y": test_pos.y, "z": test_pos.z}
                    if coord_dict not in scene_info["potential_coords"]:
                        scene_info["potential_coords"].append(coord_dict)

    if not scene_info["potential_coords"]:
        scene_info["potential_coords"].append({"x": 0, "y": 0, "z": 0})

    narrative = ambient.narrative
    effects_to_validate = list(ambient.effects)

    if lm_adapter is not None:
        try:
            enrichment = lm_adapter.enrich_ambient_event(
                ambient_id=ambient.id,
                base_narrative=ambient.narrative,
                current_scene_info=scene_info,
            )
            if enrichment:
                if "narrative" in enrichment and enrichment["narrative"]:
                    narrative = enrichment["narrative"]
                if "effects" in enrichment and isinstance(enrichment["effects"], list):
                    for eff in enrichment["effects"]:
                        if isinstance(eff, dict) and "kind" in eff and "payload" in eff:
                            effects_to_validate.append(
                                TransitionProposal(
                                    kind=eff["kind"],
                                    payload=eff["payload"]
                                )
                            )
        except Exception as exc:
            logger.debug("Failed to dynamically enrich ambient event %s: %s", ambient.id, exc)

    synthetic_action = SemanticAction(
        verb=ActionType.AMBIENT,
        actor=SYSTEM_ACTOR,
        raw_input=f"[ambient:{ambient.id}]",
    )

    record = Transition(
        kind=TransitionKind.AMBIENT_EVENT,
        payload={
            "id": ambient.id,
            "narrative": narrative,
            "time_of_day": world.config.clock.time_of_day(world.tick).value,
        },
    )

    # Effects go through the same validator that gates LM-proposed
    # effects, so the pack cannot blindly mutate canonical state.
    validated = _validate_proposals(effects_to_validate, world, synthetic_action)

    transitions: list[Transition] = [record] + validated

    # Witnesses: every entity that has line-of-sight to *something* is a
    # witness in spirit, but ambient events are world-wide narration —
    # we leave the list empty rather than synthesize a witness for the
    # whole grid. NPCs perceive ambient events via the projection's
    # recent_events window.
    event = Event(
        event_id=new_event_id(),
        tick=world.tick,
        action=synthetic_action,
        transitions=transitions,
        witnesses=[],
        narrative_hint=narrative or None,
    )

    apply_transitions(world, transitions)
    world.event_log.append(event)
    apply_event_to_graph(world.relational, event, world.tick)

    logger.info(
        "Ambient event fired | id=%s | tick=%d | narrative=%r",
        ambient.id, world.tick, narrative,
    )
    return event


# ──────────────────────────────────────────────────────────────────────
# World drift — slow, continuous changes that happen every tick without
# requiring an actor to perform a verb. These keep the world "alive":
#   • Intoxication wears off
#   • Emotional states slowly drift back toward neutral
#   • Lit candles / torches burn down
#   • Hot objects cool toward ambient
# Drift is intentionally cheap so it can run every tick.
# ──────────────────────────────────────────────────────────────────────

_DRIFT_EVERY_TICKS = 1
_INTOX_DECAY_PER_TICK = 1.0
_TEMP_AMBIENT = 20.0
_TEMP_COOL_RATE = 0.05   # fraction of (current - ambient) per tick

_MOOD_DRIFT_EVERY_TICKS = 6
_MOOD_RELAX_MAP = {
    "happy": "friendly",
    "friendly": "neutral",
    "humiliated": "suspicious",
    "suspicious": "neutral",
    "fearful": "suspicious",
    "angry": "suspicious",
    "hostile": "angry",
    "grieving": "neutral",
    "proud": "neutral",
}


def _world_drift_tick(world: WorldState) -> list[Transition]:
    transitions: list[Transition] = []
    grid = world.spatial

    # ── Intoxication wears off ────────────────────────────────────────
    for ent in grid.entities.values():
        if not ent.alive:
            continue
        intox = float(ent.stats.get("intoxication", 0))
        if intox > 0:
            transitions.append(Transition(
                kind=TransitionKind.ENTITY_STAT_CHANGED,
                payload={
                    "entity_id": str(ent.entity_id),
                    "stat": "intoxication",
                    "delta": -_INTOX_DECAY_PER_TICK,
                    "cause": "metabolism",
                },
            ))

    # ── Mood relaxation toward neutral every N ticks ───────────────────
    if world.tick % _MOOD_DRIFT_EVERY_TICKS == 0:
        for ent in grid.entities.values():
            if not ent.alive:
                continue
            cur = ent.emotional_state.value
            if cur == "neutral":
                continue
            nxt = _MOOD_RELAX_MAP.get(cur)
            if nxt and nxt != cur:
                transitions.append(Transition(
                    kind=TransitionKind.ENTITY_EMOTIONAL_STATE_CHANGED,
                    payload={
                        "entity_id": str(ent.entity_id),
                        "to": nxt,
                        "from": cur,
                        "cause": "mood_drift",
                    },
                ))

    # ── Candle burn-down ──────────────────────────────────────────────
    # Lit candles shorten their fuel every tick; when exhausted, their
    # fire goes out via a STRUCTURE_MODIFIED transition so replay stays
    # deterministic.
    for obj in grid.objects.values():
        if "on_fire" in obj.tags and "candle" in obj.tags and obj.position is not None:
            fuel = int(obj.attributes.get("fuel", 30))
            new_fuel = fuel - 1
            if new_fuel <= 0:
                transitions.append(Transition(
                    kind=TransitionKind.STRUCTURE_MODIFIED,
                    payload={
                        "object_id": str(obj.object_id),
                        "x": obj.position.x,
                        "y": obj.position.y,
                        "remove_tags": ["on_fire", "heat_source"],
                        "add_tags": ["spent"],
                        "attribute_set": {"fuel": 0},
                        "cause": "candle_burned_out",
                    },
                ))
            else:
                transitions.append(Transition(
                    kind=TransitionKind.STRUCTURE_MODIFIED,
                    payload={
                        "object_id": str(obj.object_id),
                        "x": obj.position.x,
                        "y": obj.position.y,
                        "remove_tags": [],
                        "add_tags": [],
                        "attribute_set": {"fuel": new_fuel},
                        "cause": "candle_burn",
                    },
                ))

    return transitions


__all__ = ["world_tick", "SYSTEM_ACTOR"]
