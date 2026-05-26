"""
Deterministic post-action ripples — D5 fallback without LM calls.

When ``propose_consequences`` is skipped (mock adapter, budget cap, or
autonomous NPC tick) these rules still emit validated transitions so
open-ended actions leave second-order state the world can react to.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .schemas import (
    AlertnessLevel,
    EdgeKind,
    EmotionalState,
    EntityId,
    EntityKind,
    Transition,
    TransitionKind,
    WorldFact,
    WorldFactScope,
)

if TYPE_CHECKING:
    from .schemas import SemanticAction, WorldState

# Edge kinds worth recording as durable social facts.
_SIGNIFICANT_EDGES = frozenset({
    EdgeKind.FEARS,
    EdgeKind.OWES_DEBT,
    EdgeKind.WARY_OF,
    EdgeKind.DISTRUSTS,
    EdgeKind.INTIMATE,
})


def _witnesses_near(
    world: "WorldState",
    *,
    actor_id: str,
    radius: int = 8,
) -> list[str]:
    from .region_utils import entity_or_none

    actor = entity_or_none(world, EntityId(actor_id))
    if actor is None:
        return []
    out: list[str] = []
    for eid, ent in world.spatial.entities.items():
        eid_str = str(eid)
        if eid_str == actor_id or not ent.alive:
            continue
        if ent.kind not in (EntityKind.NPC, EntityKind.PLAYER):
            continue
        if actor.position.manhattan(ent.position) <= min(radius, ent.sight_range):
            out.append(eid_str)
    return out


def _fact_transition(world: "WorldState", fact: WorldFact) -> Transition | None:
    from .fact_registry import _duplicate_fact
    from .world_adjudicator import world_fact_transition

    if _duplicate_fact(world, fact):
        return None
    return world_fact_transition(fact)


def filter_duplicate_ripples(
    existing: list[Transition],
    proposed: list[Transition],
) -> list[Transition]:
    """Drop ripples that duplicate transitions already applied (incl. D5)."""

    def _keys(t: Transition) -> list[tuple]:
        if t.kind == TransitionKind.WORLD_FACT_REGISTERED:
            fact = t.payload.get("fact") or {}
            claim = " ".join(str(fact.get("claim", "")).lower().split())[:80]
            subject = fact.get("subject_id")
            keys = [(t.kind, claim, subject)]
            tags = fact.get("tags") or []
            if "theft" in tags and subject:
                keys.append((t.kind, "theft", subject))
            return keys
        if t.kind in (
            TransitionKind.ENTITY_EMOTIONAL_STATE_CHANGED,
            TransitionKind.ENTITY_ALERTNESS_CHANGED,
        ):
            return [(
                t.kind,
                t.payload.get("entity_id"),
                t.payload.get("to"),
                t.payload.get("cause"),
            )]
        if t.kind == TransitionKind.EDGE_UPDATED:
            return [(
                t.kind,
                t.payload.get("source"),
                t.payload.get("target"),
                t.payload.get("edge_kind"),
            )]
        return []

    seen: set[tuple] = set()
    for t in existing:
        seen.update(_keys(t))
    out: list[Transition] = []
    for t in proposed:
        keys = _keys(t)
        if keys and any(k in seen for k in keys):
            continue
        seen.update(keys)
        out.append(t)
    return out


def propose_rule_ripples(
    world: "WorldState",
    action: "SemanticAction",
    primary_transitions: list[Transition],
) -> list[Transition]:
    """
    Return additional transitions from rule-based second-order effects.

    Called from ``GameLoop._apply_post_action_lm_layer`` when LM D5
    produces nothing or is not invoked.
    """
    if not primary_transitions:
        return []

    actor_id = str(action.actor)
    tick = world.tick
    verb = str(action.verb).lower()
    ripples: list[Transition] = []
    seen_targets: set[str] = set()

    for t in primary_transitions:
        p = t.payload

        # Successful theft — victim may notice later; witnesses get a trace.
        if t.kind == TransitionKind.ITEM_TRANSFERRED:
            from_id = str(p.get("from_entity") or "")
            to_id = str(p.get("to_entity") or "")
            if from_id and to_id and to_id == actor_id and from_id != actor_id:
                obj_id = str(p.get("object_id") or "something")
                fact = WorldFact(
                    claim=f"{from_id} lost {obj_id} to theft"[:160],
                    scope=WorldFactScope.ENTITY,
                    subject_id=from_id,
                    established_tick=tick,
                    established_by=actor_id,
                    tags=["theft", "investigation"],
                )
                tr = _fact_transition(world, fact)
                if tr:
                    ripples.append(tr)
                for wid in _witnesses_near(world, actor_id=actor_id, radius=6):
                    if wid in seen_targets:
                        continue
                    seen_targets.add(wid)
                    ripples.append(Transition(
                        kind=TransitionKind.ENTITY_EMOTIONAL_STATE_CHANGED,
                        payload={
                            "entity_id": wid,
                            "to": EmotionalState.SUSPICIOUS.value,
                            "cause": "witnessed_theft",
                        },
                    ))

        # Caught stealing — bystanders alert authorities mentally.
        elif (
            t.kind == TransitionKind.ENTITY_STAT_CHANGED
            and p.get("cause") == "caught_stealing"
            and p.get("stat") == "reputation"
        ):
            for wid in _witnesses_near(world, actor_id=actor_id, radius=8):
                if wid in seen_targets:
                    continue
                seen_targets.add(wid)
                ripples.append(Transition(
                    kind=TransitionKind.ENTITY_ALERTNESS_CHANGED,
                    payload={
                        "entity_id": wid,
                        "to": AlertnessLevel.MEDIUM.value,
                        "cause": "witnessed_theft",
                    },
                ))

        # Significant social edge shift — promote to entity-scoped fact.
        elif t.kind == TransitionKind.EDGE_UPDATED:
            kind_raw = p.get("edge_kind") or p.get("kind") or ""
            try:
                edge_kind = EdgeKind(kind_raw)
            except ValueError:
                edge_kind = None
            if edge_kind in _SIGNIFICANT_EDGES:
                src = str(p.get("source") or "")
                tgt = str(p.get("target") or "")
                if src and tgt:
                    fact = WorldFact(
                        claim=f"{src} {edge_kind.value} {tgt}"[:160],
                        scope=WorldFactScope.ENTITY,
                        subject_id=src,
                        established_tick=tick,
                        established_by=actor_id,
                        tags=["social", edge_kind.value],
                    )
                    tr = _fact_transition(world, fact)
                    if tr:
                        ripples.append(tr)

        # Combat harm — nearby NPCs become wary of the attacker.
        elif (
            t.kind == TransitionKind.ENTITY_HEALTH_CHANGED
            and float(p.get("delta", 0)) < -10
            and p.get("cause") in ("combat", "attack", None)
        ):
            victim_id = str(p.get("entity_id") or "")
            for wid in _witnesses_near(world, actor_id=actor_id, radius=10):
                if wid in (actor_id, victim_id) or wid in seen_targets:
                    continue
                seen_targets.add(wid)
                ripples.append(Transition(
                    kind=TransitionKind.EDGE_UPDATED,
                    payload={
                        "source": wid,
                        "target": actor_id,
                        "edge_kind": EdgeKind.WARY_OF.value,
                        "delta": 0.12,
                        "meta": {"cause": "witnessed_violence", "verb": verb},
                    },
                ))

        # Sabotage / environmental damage — schedule investigation goal seed.
        elif t.kind == TransitionKind.STRUCTURE_MODIFIED:
            tags = p.get("add_tags") or []
            if "sabotaged" in tags or p.get("cause") == "sabotage":
                for wid in _witnesses_near(world, actor_id=actor_id, radius=8):
                    if wid in seen_targets:
                        continue
                    seen_targets.add(wid)
                    ripples.append(Transition(
                        kind=TransitionKind.ENTITY_ALERTNESS_CHANGED,
                        payload={
                            "entity_id": wid,
                            "to": AlertnessLevel.LOW.value,
                            "cause": "noticed_sabotage",
                        },
                    ))

        # Low-noise events still nudge nearby alertness (pickpocket, lockpick).
        elif t.kind == TransitionKind.NOISE_EVENT:
            noise = int(p.get("noise_level", 0))
            if noise < 5:
                continue
            src = str(p.get("source_id") or actor_id)
            src_ent = world.spatial.entities.get(EntityId(src))
            if src_ent is None:
                continue
            threshold = AlertnessLevel.LOW if noise < 25 else AlertnessLevel.MEDIUM
            for eid, ent in world.spatial.entities.items():
                eid_str = str(eid)
                if eid_str == src or not ent.alive or eid_str in seen_targets:
                    continue
                if src_ent.position.manhattan(ent.position) > ent.hearing_range:
                    continue
                if ent.alertness.value in (
                    AlertnessLevel.HIGH.value,
                    AlertnessLevel.COMBAT.value,
                ):
                    continue
                ripples.append(Transition(
                    kind=TransitionKind.ENTITY_ALERTNESS_CHANGED,
                    payload={
                        "entity_id": eid_str,
                        "to": threshold.value,
                        "cause": "heard_suspicious_noise",
                    },
                ))
                seen_targets.add(eid_str)

    return ripples


__all__ = ["filter_duplicate_ripples", "propose_rule_ripples"]
