"""Apply validated transitions."""

from __future__ import annotations

import hashlib
import logging
import random
from typing import Any, Optional

logger = logging.getLogger(__name__)

from ..relational import (
    get_entity_faction_ids,
    get_faction_tension,
)
from ..schemas import (
    ActionType,
    ActionZone,
    AlertnessLevel,
    ConsentState,
    Coord,
    EdgeKind,
    EmotionalState,
    EntityId,
    EntityState,
    EquipSlot,
    FacingDirection,
    FACING_VECTORS,
    GroundingResult,
    IntentBlock,
    MalformedActionError,
    ObjectId,
    ObjectState,
    ProjectionFirewallError,
    RejectionReason,
    SemanticAction,
    SemanticProjection,
    SpatialGrid,
    StyleBlock,
    TerrainType,
    Transition,
    TransitionKind,
    TransitionProposal,
    ValidationResult,
    VerbTemplate,
    WorldState,
)
from ..spatial import (
    entities_visible_from,
    find_path,
    is_reachable,
    visible_from,
)
from .common import *


def _payload_entity_id(
    payload: dict[str, Any],
    *,
    fallback: Optional[str] = None,
) -> Optional[EntityId]:
    """Resolve entity_id from a transition payload; None if absent."""
    raw = payload.get("entity_id") or payload.get("actor") or fallback
    if raw is None:
        return None
    return EntityId(str(raw))


def apply_transitions(
    world: WorldState,
    transitions: list[Transition],
) -> None:
    """
    Apply a list of validated Transitions to the canonical WorldState.

    This is the ONLY sanctioned mutation path for canonical state.
    """
    grid = world.spatial
    graph = world.relational

    # Use index loop so newly appended transitions (e.g. ENTITY_DIED) are
    # also processed in the same pass.
    i = 0
    while i < len(transitions):
        t = transitions[i]
        i += 1
        p = t.payload

        if t.kind == TransitionKind.ENTITY_MOVED:
            eid = _payload_entity_id(p)
            if eid is None:
                logger.debug("%s missing entity_id, skipping", t.kind.value)
                continue
            entity = grid.entities.get(eid)
            if entity:
                old_pos = entity.position
                entity.position = Coord(
                    x=p["to"]["x"],
                    y=p["to"]["y"],
                    z=p["to"].get("z", 0),
                )
                # Auto-face: entity faces the direction they moved.
                dx = entity.position.x - old_pos.x
                dy = entity.position.y - old_pos.y
                new_facing = FacingDirection.from_delta(dx, dy)
                if new_facing is not None:
                    entity.facing = new_facing
                entity.meta.pop("hidden", None)
                entity.meta.pop("hidden_ticks", None)

        elif t.kind == TransitionKind.ENTITY_TURNED:
            eid = EntityId(p["entity_id"])
            entity = grid.entities.get(eid)
            if entity:
                try:
                    entity.facing = FacingDirection(p["to"])
                except ValueError:
                    pass

        elif t.kind == TransitionKind.ENTITY_HEALTH_CHANGED:
            eid = EntityId(p["entity_id"])
            entity = grid.entities.get(eid)
            if entity:
                # Spell side-effects piggy-backed on this transition
                from ..spell_runtime import apply_spell_side_effects
                apply_spell_side_effects(world, t)

                entity.health = max(0, entity.health + p["delta"])
                # Death hook — fire ENTITY_DIED when health reaches 0.
                if entity.health <= 0 and entity.alive:
                    entity.alive = False
                    # Drop carried inventory onto the floor.
                    for oid in list(entity.inventory):
                        obj = grid.objects.get(oid)
                        if obj is not None:
                            obj.owner = None
                            obj.position = entity.position
                    entity.inventory.clear()
                    # Unequip dead entity
                    entity.equipped_weapon = None
                    entity.equipped_armor = None
                    # Append ENTITY_DIED so it appears in the event log.
                    transitions.append(
                        Transition(
                            kind=TransitionKind.ENTITY_DIED,
                            payload={
                                "entity_id": str(eid),
                                "cause": p.get("cause", "unknown"),
                            },
                        )
                    )

        elif t.kind == TransitionKind.ENTITY_STAT_CHANGED:
            # Generic stat update — covers stamina, gold, reputation, and any
            # pack-defined stat.  Also mirrors health/mana changes into their
            # dedicated scalar fields for backward compat.
            eid = EntityId(p["entity_id"])
            entity = grid.entities.get(eid)
            if entity:
                stat = str(p.get("stat", "stamina"))
                delta = float(p.get("delta", 0))
                set_val = p.get("set")
                if set_val is not None:
                    new_val = float(set_val)
                else:
                    new_val = entity.stats.get(stat, 0.0) + delta
                # Ensure stats dict exists (edge case: entity deserialized without it)
                if entity.stats is None:
                    entity.stats = {}
                entity.stats[stat] = new_val
                # Mirror into scalar fields for backward compat
                if stat == "health":
                    entity.health = max(0, int(new_val))
                    if entity.health <= 0 and entity.alive:
                        entity.alive = False
                elif stat == "mana":
                    entity.mana = max(0, int(new_val))
                elif stat == "intoxication":
                    entity.intoxication = max(
                        0, min(100, entity.intoxication + int(delta))
                    )

        elif t.kind == TransitionKind.ENTITY_EMOTIONAL_STATE_CHANGED:
            eid = EntityId(p["entity_id"])
            entity = grid.entities.get(eid)
            if entity:
                entity.emotional_state = EmotionalState(p["to"])

        elif t.kind == TransitionKind.ENTITY_ALERTNESS_CHANGED:
            eid = EntityId(p["entity_id"])
            entity = grid.entities.get(eid)
            if entity:
                entity.alertness = AlertnessLevel(p["to"])

        elif t.kind == TransitionKind.ITEM_TRANSFERRED:
            oid = ObjectId(p["object_id"])
            from_eid_raw = p.get("from_entity")
            to_eid_raw = p.get("to_entity")
            to_pos_raw = p.get("to_position")
            attr_patch = p.get("_attribute_patch")
            obj = grid.objects.get(oid)
            if obj is None:
                continue
            # Attribute patch (used by _compile_mix for surviving items)
            if attr_patch:
                for k, v in attr_patch.items():
                    obj.attributes[k] = v
                # Skip the transfer mechanics if this is a pure patch operation
                if from_eid_raw is None and to_eid_raw is not None:
                    # Ensure item is still in the entity's inventory
                    ent = grid.entities.get(EntityId(to_eid_raw))
                    if ent and oid not in ent.inventory:
                        ent.inventory.append(oid)
                    continue
            # Remove from source: either an entity's inventory or the ground.
            if from_eid_raw is not None:
                from_entity = grid.entities.get(EntityId(from_eid_raw))
                if from_entity and oid in from_entity.inventory:
                    from_entity.inventory.remove(oid)
                    # Keep equipped-slot sync
                    if from_entity.equipped_weapon == oid:
                        from_entity.equipped_weapon = None
                    if from_entity.equipped_armor == oid:
                        from_entity.equipped_armor = None
                    from_entity.equipped_slots.pop(obj.slot or "", None)
            else:
                # Picked up from the ground — clear the world position.
                obj.position = None
            # Add to destination: entity inventory, specific ground position, or limbo.
            if to_eid_raw is not None:
                to_entity = grid.entities.get(EntityId(to_eid_raw))
                if to_entity is not None:
                    obj.owner = EntityId(to_eid_raw)
                    obj.position = None
                    if oid not in to_entity.inventory:
                        to_entity.inventory.append(oid)
            elif to_pos_raw is not None:
                # Drop to specific ground position (thrown item lands here)
                obj.owner = None
                obj.position = Coord(
                    x=int(to_pos_raw["x"]),
                    y=int(to_pos_raw["y"]),
                    z=int(to_pos_raw.get("z", 0)),
                )

        elif t.kind == TransitionKind.ITEM_EQUIPPED:
            eid = EntityId(p["entity_id"])
            entity = grid.entities.get(eid)
            if entity:
                oid = ObjectId(p["object_id"])
                slot = p["slot"]
                entity.equipped_slots[slot] = oid
                # Keep legacy scalar fields in sync.
                if slot == EquipSlot.WEAPON_SLOT:
                    entity.equipped_weapon = oid
                elif slot == EquipSlot.ARMOR_SLOT:
                    entity.equipped_armor = oid
                # Ensure item is also in inventory (it should be, but guarantee it).
                if oid not in entity.inventory:
                    entity.inventory.append(oid)
                # Update object ownership
                obj = grid.objects.get(oid)
                if obj:
                    obj.owner = eid

        elif t.kind == TransitionKind.ITEM_UNEQUIPPED:
            eid = EntityId(p["entity_id"])
            entity = grid.entities.get(eid)
            if entity:
                oid = ObjectId(p["object_id"])
                slot = p["slot"]
                entity.equipped_slots.pop(slot, None)
                # Keep legacy scalar fields in sync.
                if slot == EquipSlot.WEAPON_SLOT and entity.equipped_weapon == oid:
                    entity.equipped_weapon = None
                elif slot == EquipSlot.ARMOR_SLOT and entity.equipped_armor == oid:
                    entity.equipped_armor = None
                # Item stays in inventory — just no longer in a slot.

        elif t.kind == TransitionKind.ITEM_STORED:
            eid = EntityId(p["entity_id"])
            entity = grid.entities.get(eid)
            item_oid = ObjectId(p["object_id"])
            container_oid = ObjectId(p["container_id"])
            item = grid.objects.get(item_oid)
            container = grid.objects.get(container_oid)
            if entity and item and container:
                # Move item from loose inventory into container.
                if item_oid in entity.inventory:
                    entity.inventory.remove(item_oid)
                if item_oid not in container.contents:
                    container.contents.append(item_oid)
                item.container_id = container_oid
                item.position = None

        elif t.kind == TransitionKind.ITEM_RETRIEVED:
            eid = EntityId(p["entity_id"])
            entity = grid.entities.get(eid)
            item_oid = ObjectId(p["object_id"])
            container_oid = ObjectId(p["container_id"])
            item = grid.objects.get(item_oid)
            container = grid.objects.get(container_oid)
            if entity and item and container:
                # Pull item out of container into loose inventory.
                if item_oid in container.contents:
                    container.contents.remove(item_oid)
                if item_oid not in entity.inventory:
                    entity.inventory.append(item_oid)
                item.container_id = None

        elif t.kind == TransitionKind.TILE_MARKED:
            # Write or erase a mark on a tile (chalk, blood, rune, arrow, etc.)
            x, y = int(p.get("x", 0)), int(p.get("y", 0))
            from ..schemas import Coord as _Coord
            coord = _Coord(x=x, y=y)
            tile = grid.tile_at(coord)
            mark = str(p.get("mark", ""))
            if p.get("remove"):
                tile.marks = [m for m in tile.marks if m != mark]
            elif mark and mark not in tile.marks:
                tile.marks.append(mark)
            grid.set_tile(coord, tile)

        elif t.kind == TransitionKind.STRUCTURE_CREATED:
            # Place a new physical structure at a tile position.
            # Creates an ObjectState anchored to a coordinate (not inventory).
            import uuid as _uuid
            new_oid = ObjectId(str(p.get("object_id", f"obj_{_uuid.uuid4().hex[:8]}")))
            x, y = int(p.get("x", 0)), int(p.get("y", 0))
            from ..schemas import Coord as _Coord
            coord = _Coord(x=x, y=y)
            if new_oid not in grid.objects:
                tags = list(p.get("tags") or [])
                if "structure" not in tags:
                    tags.append("structure")
                new_obj = ObjectState(
                    object_id=new_oid,
                    name=str(p.get("name", "structure")),
                    tags=tags,
                    passable=bool(p.get("passable", False)),
                    transparent=bool(p.get("transparent", True)),
                    weight=float(p.get("weight", 5.0)),
                    bulk=float(p.get("bulk", 5.0)),
                    attributes={k: int(v) for k, v in (p.get("attributes") or {}).items()
                                if isinstance(v, (int, float))},
                    meta=dict(p.get("meta") or {}),
                    position=coord,
                )
                grid.objects[new_oid] = new_obj
                # If the structure is impassable, mark the tile tag so
                # pathfinding and projection can surface it.
                if not new_obj.passable:
                    tile = grid.tile_at(coord)
                    if "blocked" not in tile.tags:
                        tile.tags.append("blocked")
                    grid.set_tile(coord, tile)

        elif t.kind == TransitionKind.STRUCTURE_MODIFIED:
            # Modify a structure/tile: toggle passable, add/remove tags,
            # set attributes (e.g. lock a door, light a torch).
            x, y = p.get("x"), p.get("y")
            from ..schemas import Coord as _Coord
            oid_raw = p.get("object_id")
            target_obj: Optional[ObjectState] = None
            if oid_raw:
                target_obj = grid.objects.get(ObjectId(str(oid_raw)))
            elif x is not None and y is not None:
                # Find the first structure object at this coordinate
                coord = _Coord(x=int(x), y=int(y))
                for obj in grid.objects.values():
                    if obj.position == coord and "structure" in obj.tags:
                        target_obj = obj
                        break

            set_passable = p.get("set_passable")
            set_transparent = p.get("set_transparent")
            add_tags = list(p.get("add_tags") or [])
            remove_tags = list(p.get("remove_tags") or [])
            attr_set = dict(p.get("attribute_set") or {})

            if target_obj is not None:
                if set_passable is not None:
                    target_obj.passable = bool(set_passable)
                if set_transparent is not None:
                    target_obj.transparent = bool(set_transparent)
                for tag in add_tags:
                    if tag not in target_obj.tags:
                        target_obj.tags.append(tag)
                target_obj.tags = [t for t in target_obj.tags if t not in remove_tags]
                for k, v in attr_set.items():
                    if isinstance(v, (int, float)):
                        target_obj.attributes[k] = int(v)
            elif x is not None and y is not None:
                # No object — modify the tile itself (e.g. unlock a door tile)
                z_val = int(p.get("z", 0))
                coord = _Coord(x=int(x), y=int(y), z=z_val)
                tile = grid.tile_at(coord)
                if set_passable is not None:
                    tile.passable = bool(set_passable)
                if set_transparent is not None:
                    tile.transparent = bool(set_transparent)
                for tag in add_tags:
                    if tag not in tile.tags:
                        tile.tags.append(tag)
                tile.tags = [t for t in tile.tags if t not in remove_tags]
                grid.set_tile(coord, tile)

        elif t.kind == TransitionKind.ENVIRONMENT_STATE_CHANGED:
            # Generic key/value write on a tile or region.
            key = str(p.get("key", ""))
            value = p.get("value")
            x, y = p.get("x"), p.get("y")
            if not key:
                pass
            elif x is not None and y is not None:
                from ..schemas import Coord as _Coord
                coord = _Coord(x=int(x), y=int(y))
                tile = grid.tile_at(coord)
                if value is None:
                    tile.env.pop(key, None)
                else:
                    tile.env[key] = value
                grid.set_tile(coord, tile)
            else:
                # Region-level
                if value is None:
                    grid.region_env.pop(key, None)
                else:
                    grid.region_env[key] = value

        elif t.kind == TransitionKind.ITEM_SYNTHESIZED:
            # Open-world synthesis: create a new LM/rule-proposed object and
            # add it to the actor's inventory.  Mirrors ITEM_CREATED but marks
            # the object as player-crafted via a "crafted" tag.
            new_oid = ObjectId(p["object_id"])
            if new_oid not in grid.objects:
                tags = list(p.get("tags") or [])
                if "crafted" not in tags:
                    tags.append("crafted")
                new_obj = ObjectState(
                    object_id=new_oid,
                    name=str(p.get("name", "crafted item")),
                    tags=tags,
                    passable=True,
                    weight=float(p.get("weight", 0.5)),
                    bulk=float(p.get("bulk", 0.5)),
                    attributes={k: int(v) for k, v in (p.get("attributes") or {}).items()
                                if isinstance(v, (int, float))},
                    meta=dict(p.get("meta") or {}),
                    position=None,
                )
                grid.objects[new_oid] = new_obj
            owner_eid_raw = p.get("owner_entity_id")
            if owner_eid_raw:
                owner_ent = grid.entities.get(EntityId(owner_eid_raw))
                if owner_ent:
                    new_obj = grid.objects[new_oid]
                    new_obj.owner = EntityId(owner_eid_raw)
                    if new_oid not in owner_ent.inventory:
                        owner_ent.inventory.append(new_oid)

        elif t.kind == TransitionKind.ENTITY_GOAL_ADDED:
            eid = _payload_entity_id(p)
            if eid is None:
                logger.debug("entity_goal_added missing entity_id, skipping")
                continue
            entity = grid.entities.get(eid)
            if entity:
                goal = str(p.get("goal", "")).strip()
                if goal and goal not in entity.goals:
                    entity.goals.append(goal)

        elif t.kind == TransitionKind.ENTITY_GOAL_REMOVED:
            eid = _payload_entity_id(p)
            if eid is None:
                logger.debug("entity_goal_removed missing entity_id, skipping")
                continue
            entity = grid.entities.get(eid)
            if entity:
                goal = str(p.get("goal", "")).strip()
                entity.goals = [g for g in entity.goals if g != goal]

        elif t.kind == TransitionKind.FACTION_CREATED:
            # Write a new faction node into the relational graph
            faction_id = str(p["faction_id"])
            label = str(p.get("name", faction_id))
            graph.nodes[faction_id] = type("RelationalNode", (), {
                "node_id": faction_id,
                "kind": "faction",
                "label": label,
                "meta": dict(p.get("meta") or {}),
            })()
            # Add initial members
            for member_id in (p.get("initial_members") or []):
                from ..schemas import EdgeKind, RelationalEdge
                edge = RelationalEdge(
                    source=member_id,
                    target=faction_id,
                    kind=EdgeKind.FACTION_MEMBER,
                    weight=1.0,
                    tick_created=world.tick,
                    tick_last_updated=world.tick,
                )
                if member_id not in graph.edges:
                    graph.edges[member_id] = {}
                if faction_id not in graph.edges[member_id]:
                    graph.edges[member_id][faction_id] = []
                graph.edges[member_id][faction_id].append(edge)

        elif t.kind == TransitionKind.RELATIONAL_NODE_CREATED:
            node_id = str(p["node_id"])
            if node_id not in graph.nodes:
                graph.nodes[node_id] = type("RelationalNode", (), {
                    "node_id": node_id,
                    "kind": str(p.get("node_kind", "concept")),
                    "label": str(p.get("label", node_id)),
                    "meta": dict(p.get("meta") or {}),
                })()

        elif t.kind == TransitionKind.CLAIM_MADE:
            # Record the claim as a BELIEVES_CLAIM edge from the actor to
            # each named target, so the social graph tracks what was said.
            from ..schemas import EdgeKind, RelationalEdge
            actor_id = str(p.get("actor", ""))
            claim_text = str(p.get("text", ""))
            claim_type = str(p.get("claim_type", "declaration"))
            severity = float(p.get("severity", 0.3))
            targets = list(p.get("targets") or [])
            # Store claim on actor's own meta so NPCs can reference it
            actor_ent2 = grid.entities.get(actor_id)
            if actor_ent2:
                claims = actor_ent2.meta.setdefault("recent_claims", [])
                claims.append({
                    "tick": world.tick,
                    "text": claim_text,
                    "type": claim_type,
                    "severity": severity,
                })
                # Keep last 10 claims
                actor_ent2.meta["recent_claims"] = claims[-10:]
            # Create MADE_CLAIM edges to targets
            for tgt in targets:
                if not tgt or tgt == actor_id:
                    continue
                if actor_id not in graph.edges:
                    graph.edges[actor_id] = {}
                if tgt not in graph.edges[actor_id]:
                    graph.edges[actor_id][tgt] = []
                edge = RelationalEdge(
                    source=actor_id,
                    target=tgt,
                    kind=EdgeKind.THREATENED,
                    weight=severity,
                    tick_created=world.tick,
                    tick_last_updated=world.tick,
                    meta={"claim_type": claim_type, "text": claim_text},
                )
                graph.edges[actor_id][tgt].append(edge)

        elif t.kind == TransitionKind.BELIEF_PROPAGATED:
            # Record belief propagation as a BELIEVES_CLAIM edge on the
            # receiving entity toward the original source.
            to_eid = str(p.get("to_entity", ""))
            original_source = str(p.get("original_source", ""))
            belief = str(p.get("belief", ""))
            fidelity = float(p.get("fidelity", 0.5))
            if to_eid and original_source and belief:
                from ..schemas import EdgeKind, RelationalEdge
                edge = RelationalEdge(
                    source=to_eid,
                    target=original_source,
                    kind=EdgeKind.BELIEVES_CLAIM,
                    weight=fidelity,
                    tick_created=world.tick,
                    tick_last_updated=world.tick,
                    meta={"claim": belief},
                )
                if to_eid not in graph.edges:
                    graph.edges[to_eid] = {}
                if original_source not in graph.edges[to_eid]:
                    graph.edges[to_eid][original_source] = []
                # Avoid duplicate belief edges for same claim
                existing_beliefs = [
                    e for e in graph.edges[to_eid][original_source]
                    if e.kind == EdgeKind.BELIEVES_CLAIM
                    and e.meta.get("claim") == belief
                ]
                if not existing_beliefs:
                    graph.edges[to_eid][original_source].append(edge)

        elif t.kind == TransitionKind.ITEM_CREATED:
            # Create a new object and optionally place it in an entity's inventory
            # or on the ground.  Used by _compile_mix to produce crafted items.
            new_oid = ObjectId(p["object_id"])
            if new_oid not in grid.objects:
                pos_raw = p.get("position")
                new_pos = (
                    Coord(x=int(pos_raw["x"]), y=int(pos_raw["y"]))
                    if pos_raw else None
                )
                new_obj = ObjectState(
                    object_id=new_oid,
                    name=str(p.get("name", "unknown item")),
                    tags=list(p.get("tags") or []),
                    passable=True,
                    weight=float(p.get("weight", 0.5)),
                    bulk=float(p.get("bulk", 0.5)),
                    attributes=dict(p.get("attributes") or {}),
                    meta=dict(p.get("meta") or {}),
                    position=new_pos,
                    durability=int(p.get("durability", 100)),
                    max_durability=int(p.get("max_durability", p.get("durability", 100))),
                )
                grid.objects[new_oid] = new_obj
            to_eid_raw = p.get("to_entity")
            if to_eid_raw is not None:
                to_entity = grid.entities.get(EntityId(to_eid_raw))
                if to_entity is not None:
                    new_obj = grid.objects[new_oid]
                    new_obj.owner = EntityId(to_eid_raw)
                    new_obj.position = None
                    if new_oid not in to_entity.inventory:
                        to_entity.inventory.append(new_oid)
            # Apply tag mutations to items from mix reactions
            item_tag_add = p.get("_item_tag_add")
            if item_tag_add:
                tgt_oid = ObjectId(str(item_tag_add["object_id"]))
                tag_val = str(item_tag_add["tag"])
                tgt_obj = grid.objects.get(tgt_oid)
                if tgt_obj and tag_val not in tgt_obj.tags:
                    tgt_obj.tags.append(tag_val)

        elif t.kind == TransitionKind.ITEM_DESTROYED:
            # Destroy an item — remove from owner's inventory and from grid.
            oid = ObjectId(p["object_id"])
            owner_raw = p.get("owner_entity")
            obj = grid.objects.get(oid)
            if obj is None:
                continue
            if owner_raw is not None:
                owner_ent = grid.entities.get(EntityId(owner_raw))
                if owner_ent:
                    owner_ent.inventory[:] = [x for x in owner_ent.inventory if x != oid]
                    if owner_ent.equipped_weapon == oid:
                        owner_ent.equipped_weapon = None
                    if owner_ent.equipped_armor == oid:
                        owner_ent.equipped_armor = None
                    for slot, soid in list(owner_ent.equipped_slots.items()):
                        if soid == oid:
                            del owner_ent.equipped_slots[slot]
            grid.objects.pop(oid, None)

        elif t.kind == TransitionKind.ITEM_DURABILITY_CHANGED:
            oid = ObjectId(p["object_id"])
            obj = grid.objects.get(oid)
            if obj is None:
                continue
            delta = int(p.get("delta", 0))
            obj.durability = max(0, min(obj.max_durability, obj.durability + delta))
            cause = str(p.get("cause", "wear"))
            if obj.durability <= 0 and p.get("destroy_at_zero", True):
                owner_raw = p.get("owner_entity")
                owner_id = None
                for eid, ent in grid.entities.items():
                    if oid in ent.inventory:
                        owner_id = str(eid)
                        break
                destroy = Transition(
                    kind=TransitionKind.ITEM_DESTROYED,
                    payload={
                        "object_id": str(oid),
                        "owner_entity": owner_id,
                        "cause": cause,
                    },
                )
                # Apply destruction inline (same tick).
                owner_ent = grid.entities.get(EntityId(owner_id)) if owner_id else None
                if owner_ent:
                    owner_ent.inventory[:] = [x for x in owner_ent.inventory if x != oid]
                    if owner_ent.equipped_weapon == oid:
                        owner_ent.equipped_weapon = None
                    if owner_ent.equipped_armor == oid:
                        owner_ent.equipped_armor = None
                grid.objects.pop(oid, None)
            elif obj.durability <= 25 and "broken" not in obj.tags:
                obj.tags.append("broken")

        elif t.kind == TransitionKind.ENTITY_CONDITION_CHANGED:
            # Intercept item tag mutations emitted by _compile_mix
            item_tag_add = p.get("_item_tag_add")
            item_tag_rem = p.get("_item_tag_remove")
            if item_tag_add:
                tgt_oid = ObjectId(str(item_tag_add["object_id"]))
                tag_val = str(item_tag_add["tag"])
                tgt_obj = grid.objects.get(tgt_oid)
                if tgt_obj and tag_val not in tgt_obj.tags:
                    tgt_obj.tags.append(tag_val)
            if item_tag_rem:
                tgt_oid = ObjectId(str(item_tag_rem["object_id"]))
                tag_val = str(item_tag_rem["tag"])
                tgt_obj = grid.objects.get(tgt_oid)
                if tgt_obj and tag_val in tgt_obj.tags:
                    tgt_obj.tags.remove(tag_val)
            eid = _payload_entity_id(p)
            condition = p.get("condition")
            if eid is not None and condition is not None:
                try:
                    entity_grid = world.grid_for_entity(eid)
                except KeyError:
                    entity_grid = grid
                entity = entity_grid.entities.get(eid)
                if entity:
                    ticks = int(p.get("ticks", 0))
                    tag_add = p.get("_tag_add")
                    if tag_add and tag_add not in entity.tags:
                        entity.tags.append(tag_add)
                    tag_remove = p.get("_tag_remove")
                    if tag_remove and tag_remove in entity.tags:
                        entity.tags.remove(tag_remove)
                    if ticks <= 0:
                        entity.conditions.pop(condition, None)
                    else:
                        entity.conditions[condition] = ticks
            elif eid is None and condition is not None:
                logger.debug(
                    "ENTITY_CONDITION_CHANGED missing entity_id, skipping condition"
                )
            continue   # skip the duplicate handler below

        elif t.kind == TransitionKind.CONTACT_INITIATED:
            # Pure record — alertness/emotional changes are emitted as
            # separate transitions by _compile_contact. Nothing to do
            # on the canonical state here.
            pass

        elif t.kind == TransitionKind.TILE_CHANGED:
            coord = Coord(x=p["coord"]["x"], y=p["coord"]["y"])
            tile = grid.tile_at(coord)
            tile.terrain = TerrainType(p["terrain"])
            # Update passability based on new terrain
            tile.passable = p["terrain"] not in (
                TerrainType.WALL.value,
                TerrainType.DOOR_CLOSED.value,
            )
            grid.set_tile(coord, tile)

        elif t.kind == TransitionKind.ENTITY_DIED:
            # State already mutated by the ENTITY_HEALTH_CHANGED handler.
            # This transition exists purely as a log record / narrative hook.
            pass

        elif t.kind == TransitionKind.FLUID_CHANGED:
            from ..fluids import apply_fluid_to_object, apply_wet_tile, wet_entity

            tk = str(p.get("target_kind", ""))
            if tk == "object":
                oid = ObjectId(p["object_id"])
                obj = grid.objects.get(oid)
                if obj is not None and "volume_ml" in p:
                    apply_fluid_to_object(
                        obj,
                        float(p["volume_ml"]),
                        str(p.get("material") or ""),
                    )
            elif tk == "tile":
                from ..field_coords import coord_from_field_payload, field_tile_coord
                coord = coord_from_field_payload(p)
                if coord is not None:
                    apply_wet_tile(
                        grid,
                        field_tile_coord(coord),
                        str(p.get("material") or "water"),
                        float(p.get("volume_ml", 0.0)),
                    )
            elif tk == "tile_dry":
                from ..field_coords import coord_from_field_payload, field_tile_coord
                coord = coord_from_field_payload(p)
                if coord is not None:
                    tc = field_tile_coord(coord)
                    tile = grid.tile_at(tc)
                    tile.env.pop("wet", None)
                    tile.env.pop("fluid_spill", None)
                    # Also clear the canonical ground field layer so
                    # downstream readers (npc_planner, pressure_eval,
                    # field_tick) don't see a stale spill substance.
                    fields_bucket = tile.env.get("fields")
                    if isinstance(fields_bucket, dict):
                        ground_layer = fields_bucket.get("ground")
                        if isinstance(ground_layer, dict):
                            wet_subs = [
                                sid for sid, entry in ground_layer.items()
                                if isinstance(entry, dict)
                                and float((entry.get("meta") or {}).get("volume_ml", 0.0)) > 0.0
                            ]
                            for sid in wet_subs:
                                ground_layer.pop(sid, None)
                            if ground_layer:
                                fields_bucket["ground"] = ground_layer
                            else:
                                fields_bucket.pop("ground", None)
                            if not fields_bucket:
                                tile.env.pop("fields", None)
                    if "wet" in tile.tags:
                        tile.tags = [t for t in tile.tags if t != "wet"]
                    grid.set_tile(tc, tile)
            elif tk == "entity":
                eid = EntityId(p["entity_id"])
                ent = grid.entities.get(eid)
                if ent and p.get("wet"):
                    wet_entity(ent, str(p.get("material") or "water"))
                    if ent.position:
                        apply_wet_tile(
                            grid, ent.position,
                            str(p.get("material") or "water"),
                            float(p.get("volume_ml", 100.0)),
                        )

        elif t.kind == TransitionKind.CONTAMINATION_CHANGED:
            from ..contamination import apply_contamination_transition
            apply_contamination_transition(world, t)

        elif t.kind == TransitionKind.FIELD_CHANGED:
            from ..fields import apply_field_transition
            apply_field_transition(world, t)

        elif t.kind == TransitionKind.ENTITY_PROPERTY_CHANGED:
            eid_raw = p.get("entity_id") or p.get("actor")
            if eid_raw:
                entity = grid.entities.get(EntityId(str(eid_raw)))
                if entity:
                    prop = str(p.get("prop", "")).strip()
                    if prop:
                        if "new_value" in p:
                            if p["new_value"] is None:
                                entity.meta.pop(prop, None)
                            else:
                                entity.meta[prop] = p["new_value"]
                        elif "delta" in p:
                            cur = entity.meta.get(prop, 0)
                            try:
                                entity.meta[prop] = float(cur) + float(p["delta"])
                            except (TypeError, ValueError):
                                entity.meta[prop] = p["delta"]

        elif t.kind == TransitionKind.OBSERVATION_RECORDED:
            from ..observation_memory import apply_observation_payload

            eid_raw = p.get("entity_id")
            if eid_raw:
                entity = grid.entities.get(EntityId(str(eid_raw)))
                if entity is not None:
                    apply_observation_payload(entity, world, p)

        elif t.kind == TransitionKind.NEED_CHANGED:
            # Needs system: update hunger/fatigue/social_need/purpose/thirst/bladder.
            eid = EntityId(p["entity_id"])
            entity = grid.entities.get(eid)
            if entity:
                need = str(p.get("need", "hunger"))
                delta = float(p.get("delta", 0.0))
                current = entity.stats.get(need, 75.0)
                entity.stats[need] = max(0.0, min(100.0, current + delta))

        elif t.kind == TransitionKind.WORLD_FACT_REGISTERED:
            from ..schemas import WorldFact
            from ..world_adjudicator import _append_world_fact

            fact_data = p.get("fact") or {}
            fact = WorldFact.model_validate(fact_data)
            _append_world_fact(world, fact)

        elif t.kind == TransitionKind.SCHEDULED_EFFECT_QUEUED:
            from ..schemas import ScheduledEffect
            from ..world_adjudicator import _queue_scheduled_effect

            effect_data = p.get("effect") or {}
            effect = ScheduledEffect.model_validate(effect_data)
            _queue_scheduled_effect(world, effect)

        elif t.kind == TransitionKind.MARKET_UPDATED:
            region_id = str(p.get("region_id", ""))
            tag = str(p.get("tag", ""))
            entry = p.get("entry")
            if region_id and tag and isinstance(entry, dict):
                from ..economy import init_market

                init_market(world)
                market = world.meta.setdefault("market", {})
                if not isinstance(market, dict):
                    market = {}
                    world.meta["market"] = market
                reg = market.setdefault(region_id, {})
                if isinstance(reg, dict):
                    reg[tag] = dict(entry)

        elif t.kind == TransitionKind.PRESSURE_STATE_UPDATED:
            from ..schemas import EntityId as _Eid

            active = p.get("active_pressures")
            if active:
                world.meta["active_pressures"] = active
            else:
                world.meta.pop("active_pressures", None)

            ambient = p.get("ambient_probability_mult")
            if ambient:
                world.meta["ambient_probability_mult"] = ambient
            else:
                world.meta.pop("ambient_probability_mult", None)

            boosts = p.get("entity_boosts") or {}
            if p.get("clear_entity_boosts"):
                for ent in grid.entities.values():
                    ent.meta.pop("pressure_boosts", None)
            for eid, boost_list in boosts.items():
                ent = grid.entities.get(_Eid(str(eid)))
                if ent is not None:
                    ent.meta["pressure_boosts"] = list(boost_list)

            for key, value in (p.get("meta_patches") or {}).items():
                world.meta[str(key)] = value

        elif t.kind == TransitionKind.WORLD_MARK:
            from ..consequences import append_trace_to_meta

            limit = world.config.consequence_policy.trace_limit
            trace = {
                "verb": p.get("verb"),
                "actor_id": p.get("actor_id"),
                "tick": p.get("tick", world.tick),
                "outcome": p.get("outcome"),
                "summary": p.get("summary"),
            }
            sk = p.get("subject_kind")
            sid = str(p.get("subject_id", ""))
            if sk == "entity":
                ent = grid.entities.get(EntityId(sid))
                if ent:
                    append_trace_to_meta(ent.meta, trace, limit=limit)
            elif sk == "object":
                obj = grid.objects.get(ObjectId(sid))
                if obj:
                    append_trace_to_meta(obj.meta, trace, limit=limit)
            elif sk == "tile":
                parts = sid.split(",")
                if len(parts) == 2:
                    from ..schemas import Coord as _Coord
                    coord = _Coord(x=int(parts[0]), y=int(parts[1]))
                    tile = grid.tile_at(coord)
                    append_trace_to_meta(tile.env, trace, limit=limit)
                    grid.set_tile(coord, tile)

        elif t.kind == TransitionKind.SKILL_INCREASED:
            # Skill system: update entity.skills dict.
            eid = EntityId(p["entity_id"])
            entity = grid.entities.get(eid)
            if entity:
                skill = str(p.get("skill", ""))
                new_val = float(p.get("new_value", 0.0))
                if skill:
                    entity.skills[skill] = min(100.0, new_val)

        elif t.kind == TransitionKind.SECRET_REVEALED:
            # Secret economy: transfer a secret from one entity to another.
            from_eid = EntityId(p["from_entity"])
            to_eid = EntityId(p["to_entity"])
            secret = str(p.get("secret", ""))
            voluntary = bool(p.get("voluntary", True))
            from_entity = grid.entities.get(from_eid)
            to_entity = grid.entities.get(to_eid)
            if to_entity and secret:
                if secret not in to_entity.knowledge:
                    to_entity.knowledge.append(secret)
            # If voluntary, remove the secret from the sharer
            if voluntary and from_entity and secret in from_entity.secrets:
                from_entity.secrets.remove(secret)

        elif t.kind == TransitionKind.FACTION_RESOURCE_CHANGED:
            # Faction economy: update faction resource in world.meta.
            faction_id = str(p.get("faction_id", ""))
            resource = str(p.get("resource", "gold"))
            delta = float(p.get("delta", 0.0))
            if faction_id:
                factions = world.meta.setdefault("factions", {})
                faction_data = factions.setdefault(faction_id, {})
                resources = faction_data.setdefault("resources", {})
                resources[resource] = max(0.0, resources.get(resource, 100.0) + delta)

        elif t.kind == TransitionKind.SOUND_PROPAGATED:
            # Sound propagation: raise alertness of entities within radius.
            radius = int(p.get("radius", 6))
            source_id = p.get("source_id", "")
            src_entity = grid.entities.get(EntityId(source_id)) if source_id else None
            if src_entity:
                src_pos = src_entity.position
                _levels = [
                    AlertnessLevel.UNAWARE, AlertnessLevel.LOW,
                    AlertnessLevel.MEDIUM, AlertnessLevel.HIGH,
                    AlertnessLevel.COMBAT,
                ]
                for eid2, ent2 in grid.entities.items():
                    if not ent2.alive or str(eid2) == source_id:
                        continue
                    dist = ent2.position.manhattan(src_pos)
                    if dist <= radius:
                        # Closer = more alertness escalation
                        new_level = AlertnessLevel.HIGH if dist <= radius // 2 else AlertnessLevel.MEDIUM
                        if _levels.index(new_level) > _levels.index(ent2.alertness):
                            ent2.alertness = new_level

        elif t.kind == TransitionKind.NOISE_EVENT:
            # Alert all entities within hearing_range of the noise source.
            noise_level = int(p.get("noise_level", 0))
            if noise_level > 0:
                src_raw = p.get("source_pos") or {}
                src_pos = Coord(
                    x=int(src_raw.get("x", 0)),
                    y=int(src_raw.get("y", 0)),
                )
                for eid, entity in grid.entities.items():
                    if not entity.alive:
                        continue
                    if entity.position.manhattan(src_pos) <= entity.hearing_range:
                        current = entity.alertness
                        if noise_level >= 70:
                            new_level = AlertnessLevel.HIGH
                        elif noise_level >= 40:
                            new_level = AlertnessLevel.MEDIUM
                        else:
                            continue
                        # Only escalate, never de-escalate via noise.
                        _levels = [
                            AlertnessLevel.UNAWARE,
                            AlertnessLevel.LOW,
                            AlertnessLevel.MEDIUM,
                            AlertnessLevel.HIGH,
                            AlertnessLevel.COMBAT,
                        ]
                        if _levels.index(new_level) > _levels.index(current):
                            entity.alertness = new_level

        elif t.kind == TransitionKind.ENTITY_TAG_CHANGED:
            # Generic tag add/remove for entities OR objects.
            #
            # Payload supports both:
            #   {entity_id|object_id: ..., add: "tag"}     -- single add
            #   {entity_id|object_id: ..., remove: "tag"}  -- single remove
            #   {entity_id|object_id: ..., add: ["a","b"]} -- bulk add
            #   {entity_id|object_id: ..., remove: ["c"]}  -- bulk remove
            #
            # This is the canonical handler invoked by interaction_rules /
            # property_interactions / spell runtime to mutate tags.  Without
            # it those subsystems can declare "douse removes on_fire" in YAML
            # but the engine quietly drops the request.
            raw_id = p.get("entity_id") or p.get("object_id")
            if raw_id is None:
                continue
            target_id_str = str(raw_id)

            def _coerce_tags(val) -> list[str]:
                if val is None:
                    return []
                if isinstance(val, str):
                    return [val]
                if isinstance(val, (list, tuple)):
                    return [str(v) for v in val if v]
                return []

            adds = _coerce_tags(p.get("add") or p.get("added"))
            removes = _coerce_tags(p.get("remove") or p.get("removed"))

            ent = grid.entities.get(EntityId(target_id_str))
            obj = None if ent is not None else grid.objects.get(ObjectId(target_id_str))
            tag_owner = ent or obj
            if tag_owner is None:
                continue

            tag_owner.tags = list(tag_owner.tags or [])
            for tag in adds:
                if tag not in tag_owner.tags:
                    tag_owner.tags.append(tag)
            for tag in removes:
                tag_owner.tags = [t for t in tag_owner.tags if t != tag]

        elif t.kind == TransitionKind.REGION_TRANSIT:
            eid = EntityId(p["entity_id"])
            from_rid = p["from_region"]
            to_rid = p["to_region"]
            arrival = Coord(x=p["arrival"]["x"], y=p["arrival"]["y"])
            from_region = world.regions.get(from_rid)
            to_region = world.regions.get(to_rid)
            if from_region is None or to_region is None:
                continue  # Stale transition; target region not loaded
            entity = from_region.grid.entities.get(eid)
            if entity is None:
                # Try any region (defensive — entity may have moved already)
                for reg in world.regions.values():
                    entity = reg.grid.entities.get(eid)
                    if entity is not None:
                        break
            if entity is None:
                continue
            # Remove from source region
            from_region.grid.entities.pop(eid, None)
            # Place in target region
            entity.region_id = to_rid
            entity.position = arrival
            to_region.grid.entities[eid] = entity
            # Carry all inventory objects to the new region
            for inv_oid in entity.inventory:
                obj = from_region.grid.objects.pop(inv_oid, None)
                if obj is not None:
                    obj.region_id = to_rid
                    to_region.grid.objects[inv_oid] = obj
            # Update active region when the player moves
            from ..schemas import EntityKind as _EK
            if entity.kind == _EK.PLAYER:
                world.active_region_id = to_rid

        # Relational transitions are handled by relational.apply_event_to_graph
        # after the event is committed to the log.


# ---------------------------------------------------------------------------
