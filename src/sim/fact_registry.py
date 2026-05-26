"""
Bridge compiled actions ↔ durable WorldFacts.

Successful kernel/generic compiles often leave tile marks and environment
changes but no ``WorldFact`` — so NPC belief seeding and long-horizon
memory only saw adjudicated (rejected) actions.  This module:

  1. ``promote_compile_to_fact_transitions`` — emit WORLD_FACT_REGISTERED
     transitions from sticky compile output.
  2. ``survey_*`` helpers — consult marks, env state, and facts at compile time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from .schemas import (
    Coord,
    EntityId,
    Transition,
    TransitionKind,
    WorldFact,
    WorldFactScope,
    WorldState,
    coord_key,
)

if TYPE_CHECKING:
    from .schemas import EntityState, SemanticAction

_DEDUP_TICKS = 5


def _normalize_claim(text: str) -> str:
    return " ".join(text.lower().split())[:120]


def _duplicate_fact(world: WorldState, fact: WorldFact) -> bool:
    """Skip if an equivalent fact was registered recently at the same subject."""
    claim_norm = _normalize_claim(fact.claim)
    for existing in reversed(world.world_facts[-24:]):
        if existing.subject_id != fact.subject_id:
            continue
        if existing.scope != fact.scope:
            continue
        if world.tick - existing.established_tick > _DEDUP_TICKS:
            continue
        if _normalize_claim(existing.claim) == claim_norm:
            return True
    return False


def _fact_transition(world: WorldState, fact: WorldFact) -> Optional[Transition]:
    from .world_adjudicator import world_fact_transition

    if _duplicate_fact(world, fact):
        return None
    return world_fact_transition(fact)


def promote_compile_to_fact_transitions(
    world: WorldState,
    action: "SemanticAction",
    transitions: list[Transition],
) -> list[Transition]:
    """
    Turn sticky compile transitions into replay-safe WORLD_FACT_REGISTERED
    transitions.

    Called from ``consequences.finalize_compiled_result`` so every real
    action — not only adjudicated ones — leaves queryable world memory.
    """
    from .consequences import narrative_only_verb

    if narrative_only_verb(world, str(action.verb)):
        return []

    actor_id = str(action.actor)
    tick = world.tick
    raw = (action.raw_input or str(action.verb)).strip()
    verb = str(action.verb).lower()
    promoted: list[Transition] = []

    for t in transitions:
        if t.kind == TransitionKind.TILE_MARKED and not t.payload.get("remove"):
            mark = str(t.payload.get("mark", "")).strip()
            if not mark:
                continue
            z = int(t.payload.get("z", 0))
            from .region_utils import tile_key_for_entity

            subject = tile_key_for_entity(
                world,
                action.actor,
                Coord(x=int(t.payload["x"]), y=int(t.payload["y"]), z=z),
            )
            fact = WorldFact(
                claim=mark[:160],
                scope=WorldFactScope.TILE,
                subject_id=subject,
                established_tick=tick,
                established_by=actor_id,
                source_intent=raw[:200] or None,
                tags=["environment"],
            )
            tr = _fact_transition(world, fact)
            if tr:
                promoted.append(tr)

        elif t.kind == TransitionKind.ENVIRONMENT_STATE_CHANGED:
            key = str(t.payload.get("key") or t.payload.get("change") or "change")
            val = t.payload.get("value")
            tile_key = t.payload.get("tile_key")
            if tile_key is None and t.payload.get("x") is not None:
                z = int(t.payload.get("z", 0))
                tile_key = coord_key(
                    Coord(x=int(t.payload["x"]), y=int(t.payload["y"]), z=z),
                )
            claim = f"{key}={val}" if val is not None else key
            scope = WorldFactScope.TILE if tile_key else WorldFactScope.REGION
            fact = WorldFact(
                claim=claim[:160],
                scope=scope,
                subject_id=tile_key or world.active_region_id,
                established_tick=tick,
                established_by=actor_id,
                source_intent=raw[:200] or None,
                tags=["environment"],
            )
            tr = _fact_transition(world, fact)
            if tr:
                promoted.append(tr)

        elif t.kind == TransitionKind.CLAIM_MADE:
            claim = str(t.payload.get("text", "")).strip()
            if not claim:
                continue
            targets = list(t.payload.get("targets") or [])
            scope = WorldFactScope.ENTITY if targets else WorldFactScope.WORLD
            tags = ["message", str(t.payload.get("claim_type") or "claim")]
            fact = WorldFact(
                claim=claim[:160],
                scope=scope,
                subject_id=str(targets[0]) if targets else None,
                established_tick=tick,
                established_by=actor_id,
                source_intent=raw[:200] or None,
                tags=tags,
            )
            tr = _fact_transition(world, fact)
            if tr:
                promoted.append(tr)

        elif t.kind == TransitionKind.ITEM_SYNTHESIZED:
            name = str(t.payload.get("name") or "crafted item")
            fact = WorldFact(
                claim=f"{actor_id} created {name}"[:160],
                scope=WorldFactScope.ENTITY,
                subject_id=actor_id,
                established_tick=tick,
                established_by=actor_id,
                source_intent=raw[:200] or None,
                tags=["craft"],
            )
            tr = _fact_transition(world, fact)
            if tr:
                promoted.append(tr)

        elif t.kind == TransitionKind.ITEM_TRANSFERRED:
            from_id = str(t.payload.get("from_entity") or "")
            to_id = str(t.payload.get("to_entity") or "")
            obj_id = str(t.payload.get("object_id") or "item")
            if from_id and to_id:
                claim = f"{obj_id} moved from {from_id} to {to_id}"[:160]
                tags = ["trade"]
                if to_id == actor_id and from_id != actor_id:
                    tags = ["theft", "investigation"]
                    claim = f"{from_id} lost {obj_id}"[:160]
                fact = WorldFact(
                    claim=claim,
                    scope=WorldFactScope.ENTITY,
                    subject_id=from_id,
                    established_tick=tick,
                    established_by=actor_id,
                    source_intent=raw[:200] or None,
                    tags=tags,
                )
                tr = _fact_transition(world, fact)
                if tr:
                    promoted.append(tr)

        elif t.kind == TransitionKind.ENTITY_STAT_CHANGED:
            stat = str(t.payload.get("stat") or "")
            delta = float(t.payload.get("delta", 0))
            subject = str(t.payload.get("entity_id") or actor_id)
            cause = str(t.payload.get("cause") or "")
            if stat == "gold" and abs(delta) >= 5:
                direction = "gained" if delta > 0 else "lost"
                fact = WorldFact(
                    claim=f"{subject} {direction} {abs(delta):.0f} gold ({cause or verb})"[:160],
                    scope=WorldFactScope.ENTITY,
                    subject_id=subject,
                    established_tick=tick,
                    established_by=actor_id,
                    source_intent=raw[:200] or None,
                    tags=["bribe" if "bribe" in cause else "trade"],
                )
                tr = _fact_transition(world, fact)
                if tr:
                    promoted.append(tr)
            elif stat == "reputation" and delta <= -5:
                fact = WorldFact(
                    claim=f"{subject} reputation fell ({cause or verb})"[:160],
                    scope=WorldFactScope.ENTITY,
                    subject_id=subject,
                    established_tick=tick,
                    established_by=actor_id,
                    source_intent=raw[:200] or None,
                    tags=["investigation", "violence" if "steal" in cause else "social"],
                )
                tr = _fact_transition(world, fact)
                if tr:
                    promoted.append(tr)

        elif t.kind == TransitionKind.STRUCTURE_MODIFIED:
            add_tags = list(t.payload.get("add_tags") or [])
            obj_id = str(t.payload.get("object_id") or "structure")
            if "sabotaged" in add_tags or t.payload.get("cause") == "sabotage":
                fact = WorldFact(
                    claim=f"{obj_id} was sabotaged"[:160],
                    scope=WorldFactScope.REGION,
                    subject_id=world.active_region_id,
                    established_tick=tick,
                    established_by=actor_id,
                    source_intent=raw[:200] or None,
                    tags=["environment", "investigation"],
                )
                tr = _fact_transition(world, fact)
                if tr:
                    promoted.append(tr)

        elif t.kind == TransitionKind.ENTITY_HEALTH_CHANGED:
            delta = float(t.payload.get("delta", 0))
            subject = str(t.payload.get("entity_id") or actor_id)
            cause = str(t.payload.get("cause") or verb)
            if delta < -5 or cause in ("combat", "attack", "violence"):
                fact = WorldFact(
                    claim=f"{subject} was harmed ({cause or verb})"[:160],
                    scope=WorldFactScope.ENTITY,
                    subject_id=subject,
                    established_tick=tick,
                    established_by=actor_id,
                    source_intent=raw[:200] or None,
                    tags=["violence", "investigation"],
                )
                tr = _fact_transition(world, fact)
                if tr:
                    promoted.append(tr)

        elif t.kind == TransitionKind.ENTITY_DIED:
            victim = str(t.payload.get("entity_id") or "")
            cause = str(t.payload.get("cause") or verb)
            if victim:
                fact = WorldFact(
                    claim=f"{victim} died ({cause})"[:160],
                    scope=WorldFactScope.REGION,
                    subject_id=world.active_region_id,
                    established_tick=tick,
                    established_by=actor_id,
                    source_intent=raw[:200] or None,
                    tags=["violence", "death", "investigation"],
                )
                tr = _fact_transition(world, fact)
                if tr:
                    promoted.append(tr)

        elif t.kind == TransitionKind.NOISE_EVENT:
            noise = int(t.payload.get("noise_level", 0))
            if noise >= 15:
                src = str(t.payload.get("source_id") or actor_id)
                fact = WorldFact(
                    claim=f"loud noise ({noise}) near {src}"[:160],
                    scope=WorldFactScope.ENTITY,
                    subject_id=src,
                    established_tick=tick,
                    established_by=actor_id,
                    source_intent=raw[:200] or None,
                    tags=["investigation", "environment"],
                )
                tr = _fact_transition(world, fact)
                if tr:
                    promoted.append(tr)

        elif t.kind == TransitionKind.EDGE_UPDATED:
            from .schemas import EdgeKind

            kind_raw = str(t.payload.get("edge_kind") or t.payload.get("kind") or "")
            if kind_raw == EdgeKind.INTERACTED.value:
                continue
            src = str(t.payload.get("source") or "")
            tgt = str(t.payload.get("target") or "")
            if src and tgt and kind_raw:
                fact = WorldFact(
                    claim=f"{src} {kind_raw} {tgt}"[:160],
                    scope=WorldFactScope.ENTITY,
                    subject_id=src,
                    established_tick=tick,
                    established_by=actor_id,
                    source_intent=raw[:200] or None,
                    tags=["social"],
                )
                tr = _fact_transition(world, fact)
                if tr:
                    promoted.append(tr)

        elif t.kind == TransitionKind.SECRET_REVEALED:
            secret = str(t.payload.get("secret") or t.payload.get("text") or "a secret")
            from_id = str(t.payload.get("from_entity") or "")
            to_id = str(t.payload.get("to_entity") or "")
            subject = to_id or from_id or actor_id
            fact = WorldFact(
                claim=f"secret revealed: {secret}"[:160],
                scope=WorldFactScope.ENTITY,
                subject_id=subject,
                established_tick=tick,
                established_by=actor_id,
                source_intent=raw[:200] or None,
                tags=["message", "investigation"],
            )
            tr = _fact_transition(world, fact)
            if tr:
                promoted.append(tr)

        elif t.kind == TransitionKind.FACTION_RESOURCE_CHANGED:
            faction_id = str(t.payload.get("faction_id") or "faction")
            resource = str(t.payload.get("resource") or "resource")
            delta = float(t.payload.get("delta", 0))
            direction = "gained" if delta >= 0 else "lost"
            fact = WorldFact(
                claim=f"{faction_id} {direction} {abs(delta):.0f} {resource}"[:160],
                scope=WorldFactScope.REGION,
                subject_id=world.active_region_id,
                established_tick=tick,
                established_by=actor_id,
                source_intent=raw[:200] or None,
                tags=["faction", "trade"],
            )
            tr = _fact_transition(world, fact)
            if tr:
                promoted.append(tr)

    return promoted


def promote_compile_to_facts(
    world: WorldState,
    action: "SemanticAction",
    transitions: list[Transition],
) -> list[WorldFact]:
    """Legacy helper — prefer ``promote_compile_to_fact_transitions``."""
    from .world_adjudicator import _append_world_fact

    out: list[WorldFact] = []
    for tr in promote_compile_to_fact_transitions(world, action, transitions):
        from .schemas import WorldFact as WF

        fact = WF.model_validate(tr.payload.get("fact") or {})
        _append_world_fact(world, fact)
        out.append(fact)
    return out


def facts_at_coord(
    world: WorldState,
    coord: Coord,
    *,
    region_id: Optional[str] = None,
) -> list[WorldFact]:
    from .region_utils import scoped_tile_key

    keys = {coord_key(coord)}
    if region_id:
        keys.add(scoped_tile_key(region_id, coord))
    out: list[WorldFact] = []
    for fact in world.world_facts:
        if fact.scope == WorldFactScope.TILE and fact.subject_id in keys:
            out.append(fact)
        elif fact.scope == WorldFactScope.WORLD:
            out.append(fact)
    return out


def facts_for_entity(world: WorldState, entity_id: str) -> list[WorldFact]:
    out: list[WorldFact] = []
    for fact in world.world_facts:
        if fact.scope == WorldFactScope.ENTITY and fact.subject_id == entity_id:
            out.append(fact)
        elif fact.established_by == entity_id:
            out.append(fact)
    return out


def survey_tile(
    world: WorldState,
    actor: "EntityState",
    coord: Coord,
    *,
    perception: Optional[int] = None,
) -> dict[str, Any]:
    """Structured inspection data for a tile (marks, env, durable facts)."""
    from .region_utils import grid_for

    grid = grid_for(world, actor.entity_id)
    tile = grid.tile_at(coord)
    perc = perception if perception is not None else int(actor.attributes.get("perception", 50))
    detailed = perc >= 40

    marks = list(tile.marks[-3:])
    env = dict(tile.env or {})
    fact_claims = [
        f.claim
        for f in facts_at_coord(world, coord, region_id=actor.region_id)
    ][-4:]

    objects = [
        obj.name
        for obj in grid.objects.values()
        if obj.position == coord
    ][:4]

    result: dict[str, Any] = {
        "kind": "tile",
        "coord": coord_key(coord),
        "terrain": tile.terrain.value if hasattr(tile.terrain, "value") else str(tile.terrain),
        "distance": actor.position.manhattan(coord),
        "marks": marks,
        "environment": env if detailed else {k: env[k] for k in list(env)[:3]},
        "established_facts": fact_claims,
        "objects": objects,
        "tags": list(tile.tags)[:4],
    }
    if not detailed:
        result["note"] = "You only catch surface details from here."
    return result


def survey_area(
    world: WorldState,
    actor: "EntityState",
    *,
    radius: Optional[int] = None,
) -> dict[str, Any]:
    """Survey visible tiles around the actor."""
    from .spatial import visible_from
    from .region_utils import grid_for

    grid = grid_for(world, actor.entity_id)
    r = radius if radius is not None else actor.sight_range
    visible = visible_from(grid, actor.position, r)
    tile_surveys: list[dict[str, Any]] = []
    entity_notes: list[str] = []

    for coord in sorted(visible, key=lambda c: actor.position.manhattan(c)):
        tile = grid.tile_at(coord)
        if tile.marks or tile.env or facts_at_coord(world, coord, region_id=actor.region_id):
            tile_surveys.append(survey_tile(world, actor, coord))
        if len(tile_surveys) >= 6:
            break

    for ent in grid.entities.values():
        if ent.entity_id == actor.entity_id or not ent.alive:
            continue
        if ent.position not in visible:
            continue
        entity_notes.append(f"{ent.name} ({actor.position.manhattan(ent.position)} away)")

    return {
        "kind": "area",
        "location": str(world.meta.get("location_name", "here")),
        "tiles_of_interest": tile_surveys,
        "visible_entities": entity_notes[:8],
        "hazards": list(world.meta.get("hazards", []))[:4],
    }


def entity_fact_lines(world: WorldState, entity_id: str, *, max_items: int = 4) -> list[str]:
    """Durable claims about or by this entity."""
    from .memory_retrieval import fact_importance

    facts = facts_for_entity(world, entity_id)
    ranked = sorted(facts, key=lambda f: fact_importance(f, world), reverse=True)
    return [f.claim[:120] for f in ranked[:max_items]]


__all__ = [
    "entity_fact_lines",
    "facts_at_coord",
    "facts_for_entity",
    "promote_compile_to_fact_transitions",
    "promote_compile_to_facts",
    "survey_area",
    "survey_tile",
]
