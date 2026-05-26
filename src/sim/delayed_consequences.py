"""
Delayed consequences — schedule future ripples from durable world facts.

When a significant ``WorldFact`` is established, this module queues
``ScheduledEffect`` entries (narration + optional NPC goals) to fire on
later ticks via ``world_clock._process_scheduled_effects``.

Deterministic, replay-safe, no LM calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .schemas import (
    ScheduledEffect,
    ScheduledEffectKind,
    Transition,
    TransitionKind,
    TransitionProposal,
    WorldFactScope,
)

if TYPE_CHECKING:
    from .schemas import WorldState, WorldFact

# tag → (tick_offset, narration, optional goal template for nearby NPCs)
_DELAY_RULES: dict[str, tuple[int, str, str | None]] = {
    "theft": (
        5,
        "Whispers spread — someone is searching for what was taken.",
        "Find out who stole {claim}",
    ),
    "violence": (
        3,
        "The shock wears off; people begin asking who started it.",
        "Investigate the violence: {claim}",
    ),
    "investigation": (
        4,
        "Questions circulate as witnesses compare notes.",
        "Follow up on: {claim}",
    ),
    "environment": (
        6,
        "Staff and regulars notice the damage and start assigning blame.",
        "Assess who caused: {claim}",
    ),
    "death": (
        2,
        "Word of death travels fast — the room falls unnaturally quiet.",
        "Understand what happened to {claim}",
    ),
}

_TAG_PRIORITY = ("death", "violence", "theft", "investigation", "environment")
_MAX_NPC_GOALS_PER_FACT = 2
_NPC_GOAL_RADIUS = 10


def _pick_tag(fact: "WorldFact") -> str | None:
    for tag in _TAG_PRIORITY:
        if tag in fact.tags:
            return tag
    return None


def _nearby_npc_ids(world: "WorldState", fact: "WorldFact") -> list[str]:
    from .region_utils import (
        entities_in_region,
        entity_or_none,
        iter_entities,
        parse_scoped_tile_key,
    )
    from .schemas import EntityId, EntityKind

    out: list[str] = []

    if fact.scope == WorldFactScope.ENTITY and fact.subject_id:
        subject = entity_or_none(world, EntityId(fact.subject_id))
        if subject:
            region_id = subject.region_id
            focal = subject.position
        else:
            return []
    elif fact.scope == WorldFactScope.TILE and fact.subject_id:
        region_id, focal = parse_scoped_tile_key(fact.subject_id)
        region_id = region_id or world.active_region_id
    elif fact.scope == WorldFactScope.REGION:
        region_id = fact.subject_id or world.active_region_id
        focal = None
    else:
        region_id = world.active_region_id
        focal = None

    if focal is not None:
        for eid, ent in entities_in_region(world, region_id).items():
            if not ent.alive or ent.kind == EntityKind.PLAYER:
                continue
            if ent.position.manhattan(focal) <= _NPC_GOAL_RADIUS:
                out.append(str(eid))
    else:
        for eid, ent, _ in iter_entities(world):
            if not ent.alive or ent.kind == EntityKind.PLAYER:
                continue
            if ent.region_id == region_id:
                out.append(str(eid))

    seen: set[str] = set()
    unique: list[str] = []
    for oid in out:
        if oid in seen:
            continue
        seen.add(oid)
        unique.append(oid)
    return unique[:_MAX_NPC_GOALS_PER_FACT]


def schedule_delays_from_facts(
    world: "WorldState",
    *,
    recent_tick_window: int | None = None,
) -> tuple[list[str], list[Transition]]:
    """
    Queue delayed effects for recent facts that have not yet been scheduled.

    Returns log lines and replay-safe transitions for the propagation pass.
    """
    from .world_adjudicator import scheduled_effect_transition

    if recent_tick_window is None:
        recent_tick_window = max(
            12,
            int(world.config.consequence_policy.world_memory_window),
        )

    tick = world.tick
    log_lines: list[str] = []
    transitions: list[Transition] = []

    for fact in world.world_facts:
        if fact.meta.get("delay_scheduled"):
            continue
        if tick - fact.established_tick > recent_tick_window:
            continue

        tag = _pick_tag(fact)
        if tag is None:
            continue

        offset, narration, goal_template = _DELAY_RULES[tag]
        fire_tick = tick + offset

        transitions.append(
            scheduled_effect_transition(
                ScheduledEffect(
                    fire_tick=fire_tick,
                    created_tick=tick,
                    source_actor=fact.established_by or "system",
                    source_intent=fact.claim[:120],
                    kind=ScheduledEffectKind.NARRATION,
                    narration=narration[:240],
                    rationale=f"delayed:{tag}",
                    meta={"fact_id": fact.fact_id, "tag": tag},
                ),
            ),
        )

        if goal_template:
            claim_snip = fact.claim[:80]
            goal_text = goal_template.format(
                claim=claim_snip,
                actor=fact.established_by or "someone",
            )
            proposals: list[TransitionProposal] = []
            for npc_id in _nearby_npc_ids(world, fact):
                proposals.append(
                    TransitionProposal(
                        kind=TransitionKind.ENTITY_GOAL_ADDED.value,
                        payload={
                            "entity_id": npc_id,
                            "goal": goal_text[:240],
                            "cause": f"delayed_{tag}",
                            "priority": 0.7,
                            "fact_id": fact.fact_id,
                        },
                    )
                )
            if proposals:
                transitions.append(
                    scheduled_effect_transition(
                        ScheduledEffect(
                            fire_tick=fire_tick,
                            created_tick=tick,
                            source_actor="system",
                            kind=ScheduledEffectKind.TRANSITIONS,
                            transitions=proposals,
                            rationale=f"delayed_goals:{tag}",
                            meta={"fact_id": fact.fact_id, "tag": tag},
                        ),
                    ),
                )

        fact.meta["delay_scheduled"] = True
        log_lines.append(
            f"[delayed] {tag} consequence scheduled @ tick+{offset} "
            f"(fact {fact.fact_id[:12]})"
        )

    return log_lines, transitions


__all__ = ["schedule_delays_from_facts"]
