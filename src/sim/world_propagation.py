"""
World Propagation Layer
=======================

This module implements the three open-world dynamics that transform the
engine from a *closed-world interpreter* into a *partially open world*:

  1. ``propagate_beliefs``     — beliefs and rumours spread through the
                                  social graph each tick.

  2. ``synthesize_npc_goals``  — significant events cause affected NPCs to
                                  form new goals that persist in their
                                  EntityState.goals list and drive future
                                  NPC policy decisions.

  3. ``causal_consequence_pass`` — after a tick's primary transitions are
                                  applied, a secondary rule-based pass
                                  identifies second-order effects on entities
                                  not directly named in any transition.

None of these functions call the LM — they are deterministic simulation
passes.  When an LM adapter is active it can augment the output (e.g.
craft more evocative goal text) but the engine never requires it.

All three return lists of ``Transition`` objects.  The caller (game_loop)
applies them via ``apply_transitions``.
"""

from __future__ import annotations

import logging
import math
import random
from typing import TYPE_CHECKING, Optional

from .schemas import (
    AlertnessLevel,
    EdgeKind,
    EmotionalState,
    EntityId,
    EntityState,
    ObjectId,
    RelationalEdge,
    Transition,
    TransitionKind,
    WorldState,
    Coord,
)

if TYPE_CHECKING:
    from .lm_adapter import LMAdapter

logger = logging.getLogger(__name__)


def _lookup_entity(world: WorldState, eid_raw: str) -> Optional[EntityState]:
    from .region_utils import entity_or_none

    if not eid_raw:
        return None
    return entity_or_none(world, EntityId(str(eid_raw)))


def _entities_near(
    world: WorldState,
    *,
    region_id: str,
    position: Coord,
    radius: int,
    exclude: Optional[set[str]] = None,
) -> list[tuple[EntityId, EntityState]]:
    from .region_utils import entities_in_region

    skip = exclude or set()
    out: list[tuple[EntityId, EntityState]] = []
    for eid, ent in entities_in_region(world, region_id).items():
        if not ent.alive:
            continue
        if str(eid) in skip:
            continue
        if ent.position.manhattan(position) <= radius:
            out.append((eid, ent))
    return out

# Ordinal rank for alertness comparisons (lexicographic string order is wrong).
_ALERTNESS_RANK: dict[AlertnessLevel, int] = {
    AlertnessLevel.UNAWARE: 0,
    AlertnessLevel.LOW: 1,
    AlertnessLevel.MEDIUM: 2,
    AlertnessLevel.HIGH: 3,
    AlertnessLevel.COMBAT: 4,
}

# ─────────────────────────────────────────────────────────────────────────────
# Tuning constants
# ─────────────────────────────────────────────────────────────────────────────

# Maximum number of belief hops per propagation tick.
_MAX_HOPS = 2
# Fidelity loss per hop (each relay distorts by this fraction).
_HOP_FIDELITY_DECAY = 0.25
# Minimum fidelity below which a belief is not propagated further.
_MIN_PROPAGATION_FIDELITY = 0.15
# Social edge types that carry beliefs.
_BELIEF_CARRIER_KINDS = {
    EdgeKind.ALLY_OF, EdgeKind.INTIMATE, EdgeKind.RESPECTS,
    EdgeKind.EMPLOYED_BY, EdgeKind.EMPLOYS, EdgeKind.KIN,
    EdgeKind.INTERACTED,
}
# Proximity-based propagation: entities within this range also share beliefs
# (even without explicit edges) to model overheard conversation.
_EAVESDROP_RANGE = 3

# Goals synthesised from event types — rule table for deterministic synthesis.
# Each entry: (TransitionKind, condition_fn, goal_template, priority, affects)
# affects: "actor" | "target" | "witness" | "all"
_GOAL_RULES: list[tuple] = [
    # Death events → avenge / mourn
    (
        TransitionKind.ENTITY_DIED,
        lambda p, w: True,
        "Mourn {victim} and understand what happened.",
        0.7,
        "witness",
    ),
    (
        TransitionKind.ENTITY_DIED,
        lambda p, w: True,
        "Find out who killed {victim} and bring them to justice.",
        0.8,
        "ally",
    ),
    # Theft caught → report / retaliate
    (
        TransitionKind.ENTITY_STAT_CHANGED,
        lambda p, w: p.get("cause") == "caught_stealing",
        "Report the thief to the authorities.",
        0.6,
        "target",
    ),
    # Major gold loss → seek redress
    (
        TransitionKind.ENTITY_STAT_CHANGED,
        lambda p, w: p.get("stat") == "gold" and float(p.get("delta", 0)) < -50,
        "Recover the lost gold by any means available.",
        0.75,
        "target",
    ),
    # Attack received → either defend or flee
    (
        TransitionKind.ENTITY_HEALTH_CHANGED,
        lambda p, w: float(p.get("delta", 0)) < -20 and p.get("cause") == "combat",
        "Protect myself from further harm — find allies or escape.",
        0.8,
        "target",
    ),
    # Trade completed → restock
    (
        TransitionKind.ENTITY_STAT_CHANGED,
        lambda p, w: p.get("cause") == "completed_trade" and p.get("stat") == "reputation",
        "Seek more trading opportunities with trustworthy partners.",
        0.4,
        "actor",
    ),
    # Reputation drop → repair standing
    (
        TransitionKind.ENTITY_STAT_CHANGED,
        lambda p, w: p.get("stat") == "reputation" and float(p.get("delta", 0)) < -5,
        "Repair my damaged reputation — prove my worth to others.",
        0.55,
        "actor",
    ),
    # Environmental marks from adjudication / open verbs
    (
        TransitionKind.TILE_MARKED,
        lambda p, w: bool(p.get("mark")),
        "Inspect the fresh mark here: {mark}",
        0.5,
        "witness",
    ),
    (
        TransitionKind.WORLD_MARK,
        lambda p, w: p.get("outcome") == "adjudicated",
        "Find out what really happened: {mark}",
        0.45,
        "witness",
    ),
    # Adjudicated open verbs → reactive goals
    (
        TransitionKind.ENTITY_ALERTNESS_CHANGED,
        lambda p, w: p.get("cause") == "adjudicated_intimidate",
        "Stay away from {actor} — they threatened me.",
        0.75,
        "target",
    ),
    (
        TransitionKind.ENTITY_ALERTNESS_CHANGED,
        lambda p, w: p.get("cause") == "adjudicated_intimidate",
        "Find out why {actor} was threatening people.",
        0.6,
        "witness",
    ),
    (
        TransitionKind.ENTITY_STAT_CHANGED,
        lambda p, w: p.get("cause") == "adjudicated_bribe" and p.get("stat") == "gold",
        "Investigate whether a bribe changed hands.",
        0.65,
        "witness",
    ),
    (
        TransitionKind.ENTITY_STAT_CHANGED,
        lambda p, w: p.get("cause") == "adjudicated_hire" and float(p.get("delta", 0)) < 0,
        "Learn who was hired and for what purpose.",
        0.5,
        "witness",
    ),
    (
        TransitionKind.NOISE_EVENT,
        lambda p, w: p.get("cause") == "adjudicated_destroy",
        "Investigate the commotion and assess the damage.",
        0.7,
        "witness",
    ),
    (
        TransitionKind.ENTITY_STAT_CHANGED,
        lambda p, w: p.get("cause") == "adjudicated_reputation"
        and float(p.get("delta", 0)) < -5,
        "Repair my damaged reputation — prove my worth to others.",
        0.55,
        "target",
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# 1. Belief propagation
# ─────────────────────────────────────────────────────────────────────────────


def propagate_beliefs(world: WorldState) -> list[Transition]:
    """
    Spread BELIEVES_CLAIM edges through the social graph one propagation
    step at a time.

    For each entity E that holds a belief (BELIEVES_CLAIM edge to some
    claim source S), look at every entity F that is socially connected to
    E (via _BELIEF_CARRIER_KINDS edges or physical proximity).  If F does
    not already hold the same belief:
      • Emit a BELIEF_PROPAGATED transition for E→F.
      • Fidelity starts at the E's own belief fidelity times
        (1 - _HOP_FIDELITY_DECAY), representing rumour distortion.

    This creates a genuine social-information-diffusion mechanic:
      - A player who tells a lie to one NPC will have it spread.
      - The further a belief travels, the less faithfully it is held.
      - Entities with no social edges are islands; information reaches
        them only via physical proximity (eavesdrop range).
    """
    grid = world.spatial
    graph = world.relational
    transitions: list[Transition] = []

    # Collect all active beliefs: entity_id → list of (claim, source, fidelity)
    active_beliefs: dict[str, list[tuple[str, str, float]]] = {}
    for holder_id, targets in graph.edges.items():
        for source_id, edges in targets.items():
            for edge in edges:
                if edge.kind == EdgeKind.BELIEVES_CLAIM:
                    claim = edge.meta.get("claim", "")
                    fidelity = edge.weight
                    if fidelity < _MIN_PROPAGATION_FIDELITY or not claim:
                        continue
                    active_beliefs.setdefault(holder_id, []).append(
                        (claim, source_id, fidelity)
                    )

    if not active_beliefs:
        return []

    from .region_utils import entities_in_region

    # Build social adjacency for all entities with active beliefs
    for holder_id, beliefs in active_beliefs.items():
        holder_ent = _lookup_entity(world, holder_id)
        if holder_ent is None or not holder_ent.alive:
            continue

        # Find social neighbours
        neighbours: set[str] = set()
        for target_id, edges in graph.edges.get(holder_id, {}).items():
            for edge in edges:
                if edge.kind in _BELIEF_CARRIER_KINDS and edge.weight > 0.1:
                    neighbours.add(target_id)

        # Physical proximity (eavesdropping) — same region only
        for eid, ent in entities_in_region(world, holder_ent.region_id).items():
            if not ent.alive:
                continue
            eid_str = str(eid)
            if eid_str != holder_id:
                dist = holder_ent.position.manhattan(ent.position)
                if dist <= _EAVESDROP_RANGE:
                    neighbours.add(eid_str)

        for claim, source_id, fidelity in beliefs:
            new_fidelity = fidelity * (1.0 - _HOP_FIDELITY_DECAY)
            if new_fidelity < _MIN_PROPAGATION_FIDELITY:
                continue

            for neighbour_id in neighbours:
                if neighbour_id == holder_id or neighbour_id == source_id:
                    continue
                # Check if neighbour already holds this belief
                already_holds = any(
                    e.kind == EdgeKind.BELIEVES_CLAIM
                    and e.meta.get("claim") == claim
                    for edges_to_src in graph.edges.get(neighbour_id, {}).values()
                    for e in edges_to_src
                )
                if already_holds:
                    continue

                transitions.append(Transition(
                    kind=TransitionKind.BELIEF_PROPAGATED,
                    payload={
                        "from_entity": holder_id,
                        "to_entity": neighbour_id,
                        "belief": claim,
                        "original_source": source_id,
                        "fidelity": round(new_fidelity, 3),
                    },
                ))

    return transitions


# ─────────────────────────────────────────────────────────────────────────────
# 2. Emergent goal synthesis
# ─────────────────────────────────────────────────────────────────────────────


def synthesize_npc_goals(
    world: WorldState,
    recent_transitions: list[Transition],
) -> list[Transition]:
    """
    Inspect the tick's transitions and emit ENTITY_GOAL_ADDED transitions
    for NPCs whose situation has meaningfully changed.

    Rules are defined in ``_GOAL_RULES``.  This is intentionally rule-based
    (not LM-driven) so goal formation is deterministic and auditable.

    The caller (game_loop) applies the resulting transitions so that goals
    are written into ``entity.goals`` before the next NPC policy turn.
    """
    grid = world.spatial
    graph = world.relational
    transitions: list[Transition] = []

    for t in recent_transitions:
        p = t.payload
        for (kind, condition_fn, template, priority, affects) in _GOAL_RULES:
            if t.kind != kind:
                continue
            try:
                if not condition_fn(p, world):
                    continue
            except Exception:
                continue

            # Determine which entities receive the new goal
            candidates: list[str] = []

            if affects == "actor":
                actor_id = str(p.get("actor") or p.get("entity_id") or "")
                if actor_id:
                    candidates.append(actor_id)

            elif affects == "target":
                target_id = str(p.get("entity_id") or "")
                if target_id:
                    candidates.append(target_id)

            elif affects == "witness":
                # Proximity witnesses — entity-scoped or tile-scoped events
                source_eid_raw = p.get("entity_id") or p.get("actor") or ""
                source_ent = _lookup_entity(world, str(source_eid_raw))
                if source_ent is not None:
                    for eid, ent in _entities_near(
                        world,
                        region_id=source_ent.region_id,
                        position=source_ent.position,
                        radius=8,
                        exclude={str(source_eid_raw)},
                    ):
                        candidates.append(str(eid))
                elif t.kind == TransitionKind.TILE_MARKED:
                    x, y = p.get("x"), p.get("y")
                    if x is not None and y is not None:
                        focal = Coord(
                            x=int(x),
                            y=int(y),
                            z=int(p.get("z", 0)),
                        )
                        region_id = world.active_region_id
                        for eid, _ent in _entities_near(
                            world,
                            region_id=region_id,
                            position=focal,
                            radius=8,
                        ):
                            candidates.append(str(eid))

            elif affects == "ally":
                # Entities with ALLY_OF / KIN / INTIMATE edges to the victim
                victim_id = str(p.get("entity_id") or "")
                if victim_id:
                    for holder_id, targets in graph.edges.items():
                        for tgt_id, edges in targets.items():
                            if tgt_id == victim_id:
                                for e in edges:
                                    if e.kind in (
                                        EdgeKind.ALLY_OF, EdgeKind.KIN, EdgeKind.INTIMATE
                                    ):
                                        candidates.append(holder_id)

            # Instantiate goal text with available context
            victim_name = ""
            victim_eid = p.get("entity_id") or ""
            if victim_eid:
                v_ent = _lookup_entity(world, str(victim_eid))
                victim_name = v_ent.name if v_ent else str(victim_eid)
            goal_text = template.format(
                victim=victim_name or "the fallen",
                actor=str(p.get("actor", "someone")),
                mark=str(p.get("mark") or p.get("summary") or "something suspicious")[:60],
            )

            for cand_id in candidates:
                cand_ent = _lookup_entity(world, cand_id)
                if cand_ent is None or not cand_ent.alive:
                    continue
                # Skip player (player makes their own goals)
                from .schemas import EntityKind
                if cand_ent.kind == EntityKind.PLAYER:
                    continue
                # Don't duplicate an existing identical goal
                if any(goal_text in g for g in cand_ent.goals):
                    continue
                transitions.append(Transition(
                    kind=TransitionKind.ENTITY_GOAL_ADDED,
                    payload={
                        "entity_id": cand_id,
                        "goal": goal_text,
                        "cause": f"response_to_{t.kind.value}",
                        "priority": priority,
                    },
                ))

    return transitions


# Tag-specific goal templates for world facts (tag → goal text, priority).
_FACT_TAG_GOALS: dict[str, tuple[str, float]] = {
    "intimidation": ("Confront or avoid whoever made threats nearby.", 0.65),
    "violence": ("Investigate the violence: {claim}", 0.75),
    "bribe": ("Report the bribe — someone paid for silence.", 0.7),
    "message": ("Find who left the message and why.", 0.55),
    "search": ("Follow up on what was discovered: {claim}", 0.5),
    "investigation": ("Look into recent suspicious activity.", 0.55),
    "hire": ("Find out who was hired and for what.", 0.45),
    "disguise": ("Identify the person in disguise.", 0.6),
    "theft": ("Find out who stole from {actor}.", 0.75),
    "environment": ("Inspect the damage and who caused it.", 0.6),
    "trade": ("Ask about the recent trade: {claim}", 0.35),
    "craft": ("Ask about what was crafted nearby.", 0.35),
    "travel": ("Learn who is travelling and where they went.", 0.4),
}


def synthesize_goals_from_world_facts(
    world: WorldState,
    *,
    recent_tick_window: Optional[int] = None,
) -> list[Transition]:
    """
    Turn newly established ``WorldFact`` records into NPC goals.

    Nearby NPCs react to tile/entity-scoped facts; world-scoped facts
    nudge a few present characters to investigate.
    """
    from .region_utils import entities_in_region, parse_scoped_tile_key
    from .schemas import EntityKind

    if recent_tick_window is None:
        recent_tick_window = max(
            8,
            int(world.config.consequence_policy.world_memory_window),
        )

    tick = world.tick
    transitions: list[Transition] = []

    for fact in world.world_facts:
        if fact.meta.get("goal_synthesized"):
            continue
        if tick - fact.established_tick > recent_tick_window:
            continue

        candidates: list[str] = []

        if fact.scope.value == "tile" and fact.subject_id:
            region_id, focal = parse_scoped_tile_key(fact.subject_id)
            region_id = region_id or world.active_region_id
            for eid, ent in _entities_near(
                world,
                region_id=region_id,
                position=focal,
                radius=8,
            ):
                if ent.kind != EntityKind.PLAYER:
                    candidates.append(str(eid))

        elif fact.scope.value == "entity" and fact.subject_id:
            candidates.append(fact.subject_id)
            subject = _lookup_entity(world, fact.subject_id)
            if subject:
                for eid, ent in _entities_near(
                    world,
                    region_id=subject.region_id,
                    position=subject.position,
                    radius=6,
                    exclude={fact.subject_id},
                ):
                    if ent.kind != EntityKind.PLAYER:
                        candidates.append(str(eid))

        elif fact.scope.value == "region" and fact.subject_id:
            for eid, ent in entities_in_region(world, fact.subject_id).items():
                if ent.alive and ent.kind != EntityKind.PLAYER:
                    candidates.append(str(eid))

        elif fact.scope.value == "world":
            from .region_utils import iter_entities

            for eid, ent, _grid in iter_entities(world):
                if ent.alive and ent.kind != EntityKind.PLAYER:
                    candidates.append(str(eid))

        elif fact.scope.value == "region":
            for eid, ent in entities_in_region(world, world.active_region_id).items():
                if ent.alive and ent.kind != EntityKind.PLAYER:
                    candidates.append(str(eid))

        goal_text = f"Respond to what the world now knows: {fact.claim[:100]}"
        priority = 0.5
        for tag, (template, tag_priority) in _FACT_TAG_GOALS.items():
            if tag in fact.tags:
                goal_text = template.format(
                    claim=fact.claim[:80],
                    actor=fact.established_by or "someone",
                )
                priority = tag_priority
                break

        seen: set[str] = set()
        for cand_id in candidates:
            if cand_id in seen:
                continue
            seen.add(cand_id)
            ent = _lookup_entity(world, cand_id)
            if ent is None or not ent.alive or ent.kind == EntityKind.PLAYER:
                continue
            if any(goal_text in g for g in ent.goals):
                continue
            transitions.append(
                Transition(
                    kind=TransitionKind.ENTITY_GOAL_ADDED,
                    payload={
                        "entity_id": cand_id,
                        "goal": goal_text,
                        "cause": "world_fact",
                        "priority": priority,
                        "fact_id": fact.fact_id,
                    },
                )
            )

        fact.meta["goal_synthesized"] = True

    return transitions


def enrich_goal_transitions(
    world: WorldState,
    transitions: list[Transition],
    adapter: Optional["LMAdapter"],
) -> None:
    """
    Polish ENTITY_GOAL_ADDED text via LM when a real adapter is wired.

    Mutates transition payloads in place; rule-based templates are kept
    when enrichment returns nothing.
    """
    if adapter is None:
        return
    from .lm_adapter import MockLMAdapter

    if type(adapter) is MockLMAdapter:
        return

    grid = world.spatial
    for t in transitions:
        if t.kind != TransitionKind.ENTITY_GOAL_ADDED:
            continue
        raw = str(t.payload.get("goal") or "").strip()
        if not raw:
            continue
        eid = str(t.payload.get("entity_id") or "")
        ent = grid.entities.get(EntityId(eid)) if eid else None
        name = ent.name if ent else "someone"
        enriched = adapter.enrich_goal(
            raw,
            name,
            context=f"cause={t.payload.get('cause', '')}",
        )
        if enriched:
            t.payload["goal"] = enriched[:240]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Causal consequence pass
# ─────────────────────────────────────────────────────────────────────────────


def causal_consequence_pass(
    world: WorldState,
    tick_transitions: list[Transition],
) -> list[Transition]:
    """
    Identify second-order effects on entities not directly named in any
    primary transition and return additional Transitions to apply.

    This is the "ripple effect" engine — the thing that makes actions
    feel consequential beyond their immediate target.

    Current rule set
    ─────────────────
    • FIRE SPREAD        — if an entity is on_fire, nearby flammable items
                           and entities have a chance of catching.
                           (Augments the physics engine's own spread logic with
                           entity-to-entity spread the physics tick doesn't cover.)

    • NOISE ALERT        — loud noise events raise alertness of entities within
                           hearing_range that weren't already in the transitions.
                           (The primary compiler handles adjacent entities;
                           this handles entities at the edge of hearing range.)

    • REPUTATION ECHO    — when a player's reputation changes significantly,
                           update the emotional_state of nearby NPCs who have
                           an edge toward the player.

    • ITEM WITNESSED     — when an ITEM_SYNTHESIZED transition fires, nearby
                           entities "witness" the creation and have a chance
                           to form a WITNESSED edge and respond.

    • FACTION AWARENESS  — when a FACTION_CREATED transition fires, all
                           existing entities with ALLY_OF edges to the founding
                           entity become FACTION_MEMBER of the new faction.
    """
    grid = world.spatial
    graph = world.relational
    transitions: list[Transition] = []

    # Index which entities are already targets of this tick's transitions
    already_targeted: set[str] = set()
    for t in tick_transitions:
        p = t.payload
        for key in ("entity_id", "actor", "target", "to_entity", "from_entity"):
            val = p.get(key)
            if val:
                already_targeted.add(str(val))

    for t in tick_transitions:
        p = t.payload

        # ── Noise alert ripple ──────────────────────────────────────────────
        if t.kind == TransitionKind.NOISE_EVENT:
            noise_level = int(p.get("noise_level", 0))
            if noise_level < 8:
                continue
            src_raw = p.get("source_id") or p.get("source_pos")
            if not src_raw:
                continue
            src_ent = grid.entities.get(EntityId(str(src_raw))) if isinstance(src_raw, str) else None
            src_pos = None
            if src_ent:
                src_pos = src_ent.position
            elif isinstance(src_raw, dict):
                from .schemas import Coord
                src_pos = Coord(x=int(src_raw.get("x", 0)), y=int(src_raw.get("y", 0)))
            if src_pos is None:
                continue

            for eid, ent in grid.entities.items():
                eid_str = str(eid)
                if eid_str in already_targeted or not ent.alive:
                    continue
                if ent.alertness in (AlertnessLevel.HIGH, AlertnessLevel.COMBAT):
                    continue
                dist = src_pos.manhattan(ent.position)
                if dist <= ent.hearing_range:
                    new_alertness = AlertnessLevel.MEDIUM
                    if noise_level >= 70:
                        new_alertness = AlertnessLevel.HIGH
                    if _ALERTNESS_RANK.get(ent.alertness, 0) < _ALERTNESS_RANK.get(
                        new_alertness, 0
                    ):
                        transitions.append(Transition(
                            kind=TransitionKind.ENTITY_ALERTNESS_CHANGED,
                            payload={"entity_id": eid_str,
                                     "from": ent.alertness.value,
                                     "to": new_alertness.value},
                        ))
                        already_targeted.add(eid_str)

        # ── Reputation echo ────────────────────────────────────────────────
        elif t.kind == TransitionKind.ENTITY_STAT_CHANGED:
            if p.get("stat") != "reputation":
                continue
            delta = float(p.get("delta", 0))
            if abs(delta) < 10:
                continue
            actor_id = str(p.get("entity_id") or "")
            actor_ent = grid.entities.get(EntityId(actor_id)) if actor_id else None
            if actor_ent is None:
                continue
            for eid, ent in grid.entities.items():
                eid_str = str(eid)
                if eid_str == actor_id or not ent.alive:
                    continue
                if eid_str in already_targeted:
                    continue
                dist = actor_ent.position.manhattan(ent.position)
                if dist > ent.sight_range:
                    continue
                # Check if they have an opinion of the actor
                has_edge = any(
                    tgt == actor_id
                    for tgt in graph.edges.get(eid_str, {}).keys()
                )
                if not has_edge and dist > 4:
                    continue
                # Positive rep → warmer; negative → cooler
                if delta > 0 and ent.emotional_state not in (
                    EmotionalState.FRIENDLY, EmotionalState.HAPPY
                ):
                    transitions.append(Transition(
                        kind=TransitionKind.ENTITY_EMOTIONAL_STATE_CHANGED,
                        payload={"entity_id": eid_str, "to": EmotionalState.NEUTRAL.value},
                    ))
                elif delta < 0 and ent.emotional_state not in (
                    EmotionalState.HOSTILE, EmotionalState.ANGRY
                ):
                    transitions.append(Transition(
                        kind=TransitionKind.ENTITY_EMOTIONAL_STATE_CHANGED,
                        payload={"entity_id": eid_str, "to": EmotionalState.SUSPICIOUS.value},
                    ))
                already_targeted.add(eid_str)

        # ── Item witnessed (synthesis draws attention) ─────────────────────
        elif t.kind == TransitionKind.ITEM_SYNTHESIZED:
            owner_id = str(p.get("owner_entity_id", ""))
            owner_ent = grid.entities.get(EntityId(owner_id)) if owner_id else None
            if owner_ent is None:
                continue
            item_name = str(p.get("name", "something"))
            for eid, ent in grid.entities.items():
                eid_str = str(eid)
                if eid_str == owner_id or not ent.alive:
                    continue
                dist = owner_ent.position.manhattan(ent.position)
                if dist <= ent.sight_range:
                    # Record a WITNESSED edge
                    transitions.append(Transition(
                        kind=TransitionKind.EDGE_UPDATED,
                        payload={
                            "source": eid_str,
                            "target": owner_id,
                            "kind": EdgeKind.WITNESSED.value,
                            "delta_weight": 0.1,
                            "meta": {"witnessed_event": f"crafted_{item_name}"},
                        },
                    ))

        # ── Faction awareness ──────────────────────────────────────────────
        elif t.kind == TransitionKind.FACTION_CREATED:
            founding_id = str(p.get("founding_entity_id", ""))
            faction_id = str(p.get("faction_id", ""))
            if not founding_id or not faction_id:
                continue
            # Allies of the founder auto-join the faction
            for tgt_id, edges in graph.edges.get(founding_id, {}).items():
                for edge in edges:
                    if edge.kind in (EdgeKind.ALLY_OF, EdgeKind.KIN, EdgeKind.INTIMATE):
                        transitions.append(Transition(
                            kind=TransitionKind.EDGE_UPDATED,
                            payload={
                                "source": tgt_id,
                                "target": faction_id,
                                "kind": EdgeKind.FACTION_MEMBER.value,
                                "delta_weight": 0.8,
                            },
                        ))

        # ── Intimidation ripple — allies of victim grow wary ───────────────
        elif t.kind == TransitionKind.ENTITY_ALERTNESS_CHANGED:
            if p.get("cause") != "adjudicated_intimidate":
                continue
            victim_id = str(p.get("entity_id") or "")
            actor_id = str(p.get("actor") or "")
            if not victim_id or not actor_id:
                continue
            for holder_id, targets in graph.edges.items():
                for edge in targets.get(victim_id, []):
                    if edge.kind not in (
                        EdgeKind.ALLY_OF, EdgeKind.KIN, EdgeKind.INTIMATE,
                    ):
                        continue
                    if holder_id in already_targeted or holder_id == actor_id:
                        continue
                    transitions.append(Transition(
                        kind=TransitionKind.EDGE_UPDATED,
                        payload={
                            "source": holder_id,
                            "target": actor_id,
                            "edge_kind": EdgeKind.WARY_OF.value,
                            "delta": 0.15,
                            "meta": {"cause": "intimidation_ripple"},
                        },
                    ))
                    already_targeted.add(holder_id)

        # ── Bribe ripple — nearby witnesses grow suspicious ────────────────
        elif (
            t.kind == TransitionKind.ENTITY_STAT_CHANGED
            and p.get("cause") == "adjudicated_bribe"
            and p.get("stat") == "gold"
            and float(p.get("delta", 0)) > 0
        ):
            recipient_id = str(p.get("entity_id") or "")
            actor_id = str(p.get("actor") or "")
            recipient = grid.entities.get(EntityId(recipient_id)) if recipient_id else None
            if recipient is None:
                continue
            for eid, ent in grid.entities.items():
                eid_str = str(eid)
                if eid_str in (recipient_id, actor_id) or not ent.alive:
                    continue
                if eid_str in already_targeted:
                    continue
                if recipient.position.manhattan(ent.position) > ent.sight_range:
                    continue
                transitions.append(Transition(
                    kind=TransitionKind.ENTITY_EMOTIONAL_STATE_CHANGED,
                    payload={
                        "entity_id": eid_str,
                        "to": EmotionalState.SUSPICIOUS.value,
                        "cause": "witnessed_bribe",
                    },
                ))
                already_targeted.add(eid_str)

        # ── Environmental mark ripple — witnesses notice durable changes ──
        elif t.kind == TransitionKind.TILE_MARKED:
            if p.get("remove"):
                continue
            mark = str(p.get("mark", "")).strip()
            if not mark:
                continue
            from .observation_memory import observation_transition
            from .schemas import Coord, EntityKind

            mark_pos = Coord(
                x=int(p["x"]),
                y=int(p["y"]),
                z=int(p.get("z", 0)),
            )
            actor_id = str(p.get("actor") or "")
            for eid, ent in grid.entities.items():
                eid_str = str(eid)
                if eid_str == actor_id or not ent.alive:
                    continue
                if eid_str in already_targeted:
                    continue
                if ent.kind not in (EntityKind.NPC, EntityKind.PLAYER):
                    continue
                dist = mark_pos.manhattan(ent.position)
                if dist > ent.sight_range:
                    continue
                obs_id = f"tile_mark_{world.tick}_{mark_pos.x}_{mark_pos.y}"
                transitions.append(
                    observation_transition(
                        eid_str,
                        {
                            "tick": world.tick,
                            "event_id": obs_id,
                            "kind": "world_fact",
                            "summary": f"Noticed mark nearby: {mark[:80]}",
                            "priority": 50,
                            "actor_id": actor_id or "unknown",
                        },
                    )
                )
                if _ALERTNESS_RANK.get(ent.alertness, 0) < _ALERTNESS_RANK.get(
                    AlertnessLevel.LOW, 1
                ):
                    transitions.append(Transition(
                        kind=TransitionKind.ENTITY_ALERTNESS_CHANGED,
                        payload={
                            "entity_id": eid_str,
                            "to": AlertnessLevel.LOW.value,
                            "cause": "noticed_environmental_change",
                        },
                    ))
                already_targeted.add(eid_str)

        # ── Theft ripple — victim may realize something is missing ─────────
        elif t.kind == TransitionKind.ITEM_TRANSFERRED:
            from_id = str(p.get("from_entity") or "")
            to_id = str(p.get("to_entity") or "")
            if not from_id or not to_id or from_id == to_id:
                continue
            victim = grid.entities.get(EntityId(from_id))
            if victim is None or not victim.alive:
                continue
            if from_id in already_targeted:
                continue
            transitions.append(Transition(
                kind=TransitionKind.ENTITY_GOAL_ADDED,
                payload={
                    "entity_id": from_id,
                    "goal": "Something is missing — find out who took it.",
                    "cause": "theft_aftermath",
                    "priority": 0.7,
                },
            ))
            already_targeted.add(from_id)

        # ── Sabotage witness ripple ────────────────────────────────────────
        elif t.kind == TransitionKind.STRUCTURE_MODIFIED:
            add_tags = p.get("add_tags") or []
            if "sabotaged" not in add_tags and p.get("cause") != "sabotage":
                continue
            actor_id = str(p.get("actor") or p.get("entity_id") or "")
            actor_ent = grid.entities.get(EntityId(actor_id)) if actor_id else None
            focal = actor_ent.position if actor_ent else None
            if focal is None:
                continue
            for eid, ent in grid.entities.items():
                eid_str = str(eid)
                if eid_str == actor_id or not ent.alive or eid_str in already_targeted:
                    continue
                if focal.manhattan(ent.position) > ent.sight_range:
                    continue
                transitions.append(Transition(
                    kind=TransitionKind.ENTITY_EMOTIONAL_STATE_CHANGED,
                    payload={
                        "entity_id": eid_str,
                        "to": EmotionalState.SUSPICIOUS.value,
                        "cause": "witnessed_sabotage",
                    },
                ))
                already_targeted.add(eid_str)

    return transitions


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic quest generation
# ─────────────────────────────────────────────────────────────────────────────


def generate_dynamic_quests(world: WorldState) -> list[Transition]:
    """
    Scan the current world state and generate new goals for NPCs based on
    observed conditions.  Runs every N ticks and produces ENTITY_GOAL_ADDED
    transitions.

    Quest sources:
      1. Faction resource shortages → members get procurement goals
      2. NPC critical needs → survival goals
      3. Unfulfilled existing secrets → share/leverage goals
      4. Witnessed events → reactive goals (investigate, intervene)
      5. Social isolation → connection goals
      6. Faction rivalry → competitive goals

    All rules are deterministic — no LM calls.
    """
    from .needs import critical_needs, NEED_LABELS

    transitions: list[Transition] = []
    grid = world.spatial
    graph = world.relational

    # Track which entities already have too many active goals (cap at 3)
    _MAX_GOALS = 3

    for entity in grid.entities.values():
        if not entity.alive or entity.kind.value not in ("npc", "player"):
            continue
        eid = str(entity.entity_id)

        # Don't add goals if already at cap
        if len(entity.goals) >= _MAX_GOALS:
            continue

        # ── 1. Faction resource shortage → procurement goal ────────────────
        faction_id = entity.faction
        if faction_id:
            faction_data = world.meta.get("factions", {}).get(faction_id, {})
            shortages = faction_data.get("shortages", [])
            for resource in shortages[:1]:  # at most 1 shortage quest per entity
                goal = f"Acquire {resource} for the {faction_id} — reserves are dangerously low."
                if goal not in entity.goals and not any("Acquire" in g and resource in g for g in entity.goals):
                    transitions.append(Transition(
                        kind=TransitionKind.ENTITY_GOAL_ADDED,
                        payload={
                            "entity_id": eid,
                            "goal": goal,
                            "cause": "faction_shortage",
                            "priority": 0.8,
                        },
                    ))

        # ── 2. Critical needs → survival goals ────────────────────────────
        urgent = critical_needs(entity)
        if urgent and len(entity.goals) < _MAX_GOALS - 1:
            need_name, need_val = urgent[0]
            label = NEED_LABELS.get(need_name, need_name)
            goal = f"Urgently: {label} (need level: {need_val:.0f}/100)."
            if not any(label[:15] in g for g in entity.goals):
                transitions.append(Transition(
                    kind=TransitionKind.ENTITY_GOAL_ADDED,
                    payload={
                        "entity_id": eid,
                        "goal": goal,
                        "cause": "critical_need",
                        "priority": 0.9,
                    },
                ))

        # ── 3. Unshared secrets → leverage goal ───────────────────────────
        if entity.secrets and len(entity.goals) < _MAX_GOALS:
            secret = entity.secrets[0]
            # Does this entity have anyone they somewhat trust?
            outbound = graph.edges.get(eid, {})
            has_trusted_contact = any(
                any(e.kind in (EdgeKind.ALLY_OF, EdgeKind.INTIMATE, EdgeKind.INTERACTED)
                    for e in edges)
                for edges in outbound.values()
            )
            if has_trusted_contact:
                goal = f"Decide whether to reveal: \"{secret[:50]}\"."
                if not any("reveal" in g.lower() for g in entity.goals):
                    transitions.append(Transition(
                        kind=TransitionKind.ENTITY_GOAL_ADDED,
                        payload={
                            "entity_id": eid,
                            "goal": goal,
                            "cause": "secret_leverage",
                            "priority": 0.6,
                        },
                    ))

        # ── 4. Recent violence witnessed → investigate / report ───────────
        if len(entity.goals) < _MAX_GOALS:
            for ev in reversed(world.event_log[-8:]):
                _is_violent = any(
                    t.kind in (TransitionKind.ENTITY_HEALTH_CHANGED, TransitionKind.ENTITY_DIED)
                    and float(t.payload.get("delta", 0)) < -15
                    for t in ev.transitions
                )
                if not _is_violent:
                    continue
                # Only if entity witnessed it
                _actor_id = str(ev.action.actor)
                _actor_ent = grid.entities.get(EntityId(_actor_id))
                if _actor_ent and entity.position.manhattan(_actor_ent.position) <= entity.sight_range:
                    _actor_name = _actor_ent.name
                    goal = f"Investigate the violence involving {_actor_name} — and decide whether to act."
                    if not any("Investigate" in g and _actor_name in g for g in entity.goals):
                        transitions.append(Transition(
                            kind=TransitionKind.ENTITY_GOAL_ADDED,
                            payload={
                                "entity_id": eid,
                                "goal": goal,
                                "cause": "witnessed_violence",
                                "priority": 0.7,
                            },
                        ))
                    break

        # ── 5. Social isolation → connection goal ─────────────────────────
        social_need = entity.stats.get("social_need", 60.0)
        if social_need < 30.0 and len(entity.goals) < _MAX_GOALS:
            # Find anyone they have a positive edge with
            outbound = graph.edges.get(eid, {})
            for target_id, edges in outbound.items():
                if any(e.kind in (EdgeKind.ALLY_OF, EdgeKind.INTIMATE) for e in edges):
                    target_ent = grid.entities.get(EntityId(target_id))
                    if target_ent and target_ent.alive:
                        goal = f"Reconnect with {target_ent.name} — it has been too long."
                        if not any("Reconnect" in g for g in entity.goals):
                            transitions.append(Transition(
                                kind=TransitionKind.ENTITY_GOAL_ADDED,
                                payload={
                                    "entity_id": eid,
                                    "goal": goal,
                                    "cause": "social_isolation",
                                    "priority": 0.65,
                                },
                            ))
                        break

        # ── 6. Durable world facts → long-horizon reactive quests ─────────
        if len(entity.goals) < _MAX_GOALS:
            for fact in reversed(world.world_facts[-30:]):
                if fact.meta.get("quest_generated"):
                    continue
                if not fact.tags:
                    continue
                priority_tags = {
                    "violence", "intimidation", "bribe", "investigation",
                    "environment", "disguise", "theft", "death",
                }
                if not priority_tags.intersection(fact.tags):
                    continue
                relevant = False
                if fact.scope.value == "entity" and fact.subject_id == eid:
                    relevant = True
                elif fact.scope.value == "tile" and fact.subject_id:
                    parts = fact.subject_id.split(",")
                    if len(parts) >= 2:
                        try:
                            from .schemas import Coord

                            focal = Coord(
                                x=int(parts[0]),
                                y=int(parts[1]),
                                z=int(parts[2]) if len(parts) > 2 else 0,
                            )
                            if entity.position.manhattan(focal) <= entity.sight_range:
                                relevant = True
                        except ValueError:
                            pass
                elif fact.scope.value in ("world", "region"):
                    relevant = True

                if not relevant:
                    continue

                tag = next(
                    (t for t in fact.tags if t in priority_tags),
                    fact.tags[0],
                )
                goal_templates = {
                    "violence": f"Investigate: {fact.claim[:80]}",
                    "intimidation": f"Respond to threats in the area: {fact.claim[:60]}",
                    "bribe": "Determine whether corruption is spreading.",
                    "investigation": f"Follow up: {fact.claim[:80]}",
                    "environment": f"Assess damage and responsibility: {fact.claim[:60]}",
                    "disguise": "Identify anyone travelling under false identity.",
                    "theft": f"Find who stole: {fact.claim[:80]}",
                    "death": f"Learn what happened: {fact.claim[:80]}",
                }
                goal = goal_templates.get(
                    tag,
                    f"Respond to: {fact.claim[:80]}",
                )
                if any(goal[:40] in g for g in entity.goals):
                    continue
                transitions.append(Transition(
                    kind=TransitionKind.ENTITY_GOAL_ADDED,
                    payload={
                        "entity_id": eid,
                        "goal": goal,
                        "cause": f"world_fact_{tag}",
                        "priority": 0.65,
                        "fact_id": fact.fact_id,
                    },
                ))
                fact.meta["quest_generated"] = True
                break

    return transitions


def summarize_player_visible_ripples(
    world: WorldState,
    player_id: EntityId,
    transitions: list[Transition],
    *,
    radius: int = 8,
    max_lines: int = 4,
) -> list[str]:
    """
    Produce short, player-facing prose lines summarising same-tick world
    ripples (alertness changes, mood shifts, new goals, fire spread, etc.)
    that occurred near the player.

    Used by the game loop to append a brief "(meanwhile…)" line to the
    player's narration *after* propagation runs, so causal ripples from
    the player's own action are visible in the same beat rather than
    only appearing in NPC behaviour on the next tick.

    Returns at most ``max_lines`` lines.  Returns ``[]`` when nothing
    interesting happened nearby.
    """
    if not transitions:
        return []

    grid = world.spatial
    player = grid.entities.get(player_id)
    if player is None or player.position is None:
        return []
    px, py = player.position.x, player.position.y

    def _near(eid_raw: object) -> Optional[EntityState]:
        if not eid_raw:
            return None
        ent = grid.entities.get(EntityId(str(eid_raw)))
        if ent is None or ent.position is None or not ent.alive:
            return None
        dx = ent.position.x - px
        dy = ent.position.y - py
        if (dx * dx + dy * dy) ** 0.5 > radius:
            return None
        return ent

    # Group by kind so we can produce one consolidated line per category
    # rather than spamming "X alerted. Y alerted. Z alerted."
    alerted: list[str] = []
    mood_shift: list[tuple[str, str]] = []
    new_goal: list[tuple[str, str]] = []
    caught_fire: list[str] = []
    new_witness: list[str] = []

    for t in transitions:
        p = t.payload or {}
        if t.kind == TransitionKind.ENTITY_ALERTNESS_CHANGED:
            ent = _near(p.get("entity_id"))
            if ent is None or str(ent.entity_id) == str(player_id):
                continue
            # Only mention escalations toward HIGH/COMBAT — the dramatic ones.
            to_val = str(p.get("to", "")).lower()
            if to_val in ("high", "combat", "medium"):
                alerted.append(ent.name)
        elif t.kind == TransitionKind.ENTITY_EMOTIONAL_STATE_CHANGED:
            ent = _near(p.get("entity_id"))
            if ent is None or str(ent.entity_id) == str(player_id):
                continue
            mood_shift.append((ent.name, str(p.get("to", "")).strip() or "shaken"))
        elif t.kind == TransitionKind.ENTITY_GOAL_ADDED:
            ent = _near(p.get("entity_id"))
            if ent is None or str(ent.entity_id) == str(player_id):
                continue
            goal_text = str(p.get("goal", "")).strip()
            if goal_text:
                new_goal.append((ent.name, goal_text[:60]))
        elif t.kind == TransitionKind.ENTITY_TAG_CHANGED:
            added = p.get("add") or p.get("added")
            if isinstance(added, str) and added.lower() == "on_fire":
                ent = _near(p.get("entity_id"))
                if ent is not None and str(ent.entity_id) != str(player_id):
                    caught_fire.append(ent.name)
        elif t.kind == TransitionKind.EDGE_CREATED:
            kind = str(p.get("kind", "")).lower()
            if kind != "witnessed":
                continue
            source = p.get("source")
            target = p.get("target")
            # Surface only when the player is the subject being witnessed.
            if str(target) == str(player_id):
                witness_ent = _near(source)
                if witness_ent is not None:
                    new_witness.append(witness_ent.name)

    lines: list[str] = []

    def _dedup(seq: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for x in seq:
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    if alerted:
        names = _dedup(alerted)
        if len(names) == 1:
            lines.append(f"{names[0]} snaps to attention.")
        else:
            head = ", ".join(names[:3])
            extra = "" if len(names) <= 3 else f" and {len(names) - 3} others"
            lines.append(f"{head}{extra} snap to attention.")

    if caught_fire:
        names = _dedup(caught_fire)
        if len(names) == 1:
            lines.append(f"Flames lick across {names[0]}.")
        else:
            lines.append(f"Flames spread to {', '.join(names[:3])}.")

    if mood_shift:
        first_name, first_mood = mood_shift[0]
        if len(mood_shift) == 1:
            lines.append(f"{first_name} shifts to {first_mood}.")
        else:
            extra_names = ", ".join({n for n, _ in mood_shift[1:3]})
            lines.append(
                f"{first_name} shifts to {first_mood}"
                + (f"; nearby moods also turn." if extra_names else ".")
            )

    if new_goal:
        first_name, first_goal = new_goal[0]
        lines.append(f"{first_name} now intends: {first_goal}.")
        if len(new_goal) > 1:
            others = ", ".join({n for n, _ in new_goal[1:3]})
            if others:
                lines.append(f"{others} forms a fresh purpose as well.")

    if new_witness:
        names = _dedup(new_witness)
        if len(names) == 1:
            lines.append(f"{names[0]} watches what you just did.")
        else:
            lines.append(f"{', '.join(names[:3])} witness what you did.")

    return lines[:max_lines]


__all__ = [
    "propagate_beliefs",
    "synthesize_npc_goals",
    "causal_consequence_pass",
    "generate_dynamic_quests",
    "summarize_player_visible_ripples",
]
