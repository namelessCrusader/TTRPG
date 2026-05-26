"""
Tactical combat helpers — cover, range bands, and initiative ordering.

Used by the attack compiler and NPC turn scheduling during active combat.
All functions are deterministic and replay-safe.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Optional

from .schemas import AlertnessLevel, Coord, EntityId, Transition, TransitionKind

if TYPE_CHECKING:
    from .schemas import EntityState, SpatialGrid, WorldState

_INITIATIVE_META = "combat_initiative"
_MELEE_RANGE = 1
_NEAR_RANGE = 3
_MID_RANGE = 6


def range_zone(distance: int) -> str:
    if distance <= _MELEE_RANGE:
        return "melee"
    if distance <= _NEAR_RANGE:
        return "near"
    if distance <= _MID_RANGE:
        return "mid"
    return "far"


def range_damage_multiplier(distance: int, weapon_label: str) -> float:
    """Scale outgoing damage by range band and weapon type."""
    zone = range_zone(distance)
    if weapon_label == "unarmed melee":
        return 1.0 if zone == "melee" else 0.0
    if weapon_label == "thrown item":
        return {"melee": 0.5, "near": 1.0, "mid": 0.85, "far": 0.6}.get(zone, 0.4)
    # ranged weapon
    return {"melee": 0.7, "near": 0.95, "mid": 1.0, "far": 0.75}.get(zone, 0.5)


def _has_cover_at(grid: "SpatialGrid", coord: Coord) -> bool:
    tile = grid.tile_at(coord)
    if "cover" in tile.tags:
        return True
    for obj in grid.objects.values():
        if obj.position == coord and not obj.passable:
            return True
    return False


def cover_protection(
    grid: "SpatialGrid",
    attacker_pos: Coord,
    target_pos: Coord,
) -> int:
    """
    Extra damage reduction when the target is in or adjacent to cover,
    or when a cover tile lies on the line between attacker and target.
    """
    bonus = 0
    if _has_cover_at(grid, target_pos):
        bonus += 2
    for neighbor in target_pos.neighbors():
        if _has_cover_at(grid, neighbor):
            bonus += 1
            break

    # Partial cover along the attack line (simple midpoint check).
    mid = Coord(
        x=(attacker_pos.x + target_pos.x) // 2,
        y=(attacker_pos.y + target_pos.y) // 2,
        z=attacker_pos.z,
    )
    if mid != attacker_pos and mid != target_pos and _has_cover_at(grid, mid):
        bonus += 1

    return min(bonus, 4)


def _seeded_initiative(entity_id: str, world: "WorldState") -> float:
    raw = f"{world.rng_seed}:{world.tick}:{entity_id}:init"
    h = hashlib.sha256(raw.encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def refresh_combat_initiative(world: "WorldState") -> list[Transition]:
    """
    Recompute turn order for entities in COMBAT alertness.
    Stored in ``world.meta["combat_initiative"]`` as ordered entity ids.
    """
    combatants: list[tuple[float, str]] = []
    for eid, ent in world.all_entities().items():
        if not ent.alive or ent.health <= 0:
            continue
        if ent.alertness not in (AlertnessLevel.COMBAT, AlertnessLevel.HIGH):
            continue
        if ent.meta.get("combat_target") or ent.alertness == AlertnessLevel.COMBAT:
            agility = float(ent.attributes.get("agility", ent.attributes.get("dexterity", 50)))
            score = agility + _seeded_initiative(str(eid), world) * 20.0
            combatants.append((score, str(eid)))

    combatants.sort(key=lambda x: (-x[0], x[1]))
    order = [eid for _, eid in combatants]
    world.meta[_INITIATIVE_META] = order
    return []


def initiative_rank(entity_id: str, world: "WorldState") -> int:
    """Lower rank = acts earlier.  Non-combatants return a large default."""
    order = world.meta.get(_INITIATIVE_META) or []
    try:
        return order.index(str(entity_id))
    except ValueError:
        return 999


def sort_key_for_turn(entity_id: str, entity: "EntityState", world: "WorldState") -> tuple:
    """Sort key for NPC turn order — initiative first, then stimulus."""
    from .social_stimulus import stimulus_priority

    in_combat = entity.alertness in (AlertnessLevel.COMBAT, AlertnessLevel.HIGH)
    init = initiative_rank(str(entity_id), world) if in_combat else 999
    return (init, -stimulus_priority(entity), entity.name)


def wear_from_attack(
    attacker: "EntityState",
    target: "EntityState",
    grid: "SpatialGrid",
    damage: int,
) -> list[Transition]:
    """Emit durability loss for weapon and armor involved in an attack."""
    transitions: list[Transition] = []
    wear = max(1, damage // 3)

    if attacker.equipped_weapon:
        transitions.append(Transition(
            kind=TransitionKind.ITEM_DURABILITY_CHANGED,
            payload={
                "object_id": str(attacker.equipped_weapon),
                "delta": -wear,
                "cause": "combat_use",
            },
        ))
    elif attacker.inventory:
        for oid in attacker.inventory[:1]:
            obj = grid.objects.get(oid)
            if obj and obj.attributes.get("damage", 0) > 0:
                transitions.append(Transition(
                    kind=TransitionKind.ITEM_DURABILITY_CHANGED,
                    payload={
                        "object_id": str(oid),
                        "delta": -wear,
                        "cause": "combat_use",
                    },
                ))
                break

    if target.equipped_armor:
        transitions.append(Transition(
            kind=TransitionKind.ITEM_DURABILITY_CHANGED,
            payload={
                "object_id": str(target.equipped_armor),
                "delta": -max(1, damage // 2),
                "cause": "combat_hit",
            },
        ))

    return transitions


__all__ = [
    "cover_protection",
    "initiative_rank",
    "range_damage_multiplier",
    "range_zone",
    "refresh_combat_initiative",
    "sort_key_for_turn",
    "wear_from_attack",
]
