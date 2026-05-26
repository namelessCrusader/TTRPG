"""Shared compiler helpers."""

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
def _grid_for_action(world: "WorldState", action: "SemanticAction") -> "SpatialGrid":
    """Spatial grid containing the action's actor (multi-region safe)."""
    try:
        return world.grid_for_entity(action.actor)
    except KeyError:
        return world.spatial


def _entity_by_name(grid: "SpatialGrid", name: str) -> Optional["EntityState"]:
    """
    Case-insensitive name lookup as a fallback for when an LM outputs an
    entity's display name ("Guard", "player") instead of its canonical
    EntityId ("ent_abc123").

    Returns the first alive entity whose name matches, or None.
    """
    name_lower = name.lower()
    for entity in grid.entities.values():
        if entity.alive and entity.name.lower() == name_lower:
            return entity
    return None


def _resolve_entity_target(
    grid: "SpatialGrid",
    raw_target: str,
) -> Optional["EntityState"]:
    """
    Resolve a string target to an EntityState.

    Priority:
    1. Exact entity-id match (canonical).
    2. Case-insensitive name match (LM fallback — handles "Guard", "player").

    Returns None if neither succeeds.
    """
    entity = grid.entities.get(EntityId(raw_target))
    if entity is None:
        entity = _entity_by_name(grid, raw_target)
    return entity


# ---------------------------------------------------------------------------
# Deterministic noise source (seeded from action_id + tick for replay safety)
# ---------------------------------------------------------------------------


def _seeded_float(seed_str: str) -> float:
    """Return a float in [0, 1) deterministically derived from a string."""
    digest = hashlib.sha256(seed_str.encode()).digest()
    return int.from_bytes(digest[:4], "big") / (2**32)


def _contest_roll(
    actor_score: int,
    target_score: int,
    seed: str,
) -> bool:
    """
    Contested attribute roll.
    Returns True (actor wins) if:
      actor_score + noise_component > target_score
    noise component is ±20 drawn deterministically from seed.
    """
    noise = (_seeded_float(seed) * 40) - 20  # range [-20, 20]
    return (actor_score + noise) > target_score


# ---------------------------------------------------------------------------
# Projection firewall check
# ---------------------------------------------------------------------------


def _check_projection_firewall(
    action: SemanticAction,
    projection: Optional[SemanticProjection],
) -> None:
    """
    If a projection is provided, verify the action's target entity appears
    in it.  This prevents LM hallucinations from injecting unknown entities.

    Orientation verbs (TURN / LOOK / FACE) are exempt because their targets
    are direction strings ("north", "left", …), never entity IDs.
    """
    if projection is None:
        return
    if action.target is None:
        return
    if isinstance(action.target, Coord):
        return  # coord targets are validated against spatial layer
    # Direction strings are never entity IDs — exempt orientation verbs.
    if action.verb in (ActionType.TURN, ActionType.LOOK, ActionType.FACE):
        return

    target_id = str(action.target)
    visible_ids = {s.entity_id for s in projection.visible_entities}
    if target_id not in visible_ids:
        raise ProjectionFirewallError(
            f"Action targets entity '{target_id}' which is not present in "
            f"the focal entity's projection (visible: {sorted(visible_ids)})."
        )


# ---------------------------------------------------------------------------
# Inventory / carry-weight helpers
# ---------------------------------------------------------------------------


def current_carry_weight(entity: "EntityState", grid: "SpatialGrid") -> float:
    """
    Sum the weight of every object this entity currently holds or wears.

    Includes:
    • Items in entity.inventory (loose).
    • Items in entity.equipped_slots (worn/wielded).
    • For containers in inventory: the container's own weight + its contents'
      weight (one level deep — no recursive nesting).

    Equipped items that are also in inventory are counted once (equipped_slots
    is a subset of inventory for held items).
    """
    seen: set[ObjectId] = set()
    total = 0.0

    all_oids: list[ObjectId] = list(entity.inventory)
    for slot_oid in entity.equipped_slots.values():
        if slot_oid not in seen:
            all_oids.append(slot_oid)

    for oid in all_oids:
        if oid in seen:
            continue
        seen.add(oid)
        obj = grid.objects.get(oid)
        if obj is None:
            continue
        total += obj.weight
        # If it's a container, add the weight of everything inside it.
        if obj.is_container:
            for c_oid in obj.contents:
                c_obj = grid.objects.get(c_oid)
                if c_obj is not None and c_oid not in seen:
                    total += c_obj.weight
                    seen.add(c_oid)

    return total


def _encumbrance_ratio(entity: "EntityState", grid: "SpatialGrid") -> float:
    """
    Return carried_weight / carry_capacity, or 0 if capacity is zero.
    Values > 1.0 mean over-encumbered.
    """
    if entity.carry_capacity <= 0:
        return 0.0
    return current_carry_weight(entity, grid) / entity.carry_capacity


# ---------------------------------------------------------------------------
# Relative-movement keyword → facing vector resolution
# ---------------------------------------------------------------------------

# Maps relative-motion keywords to a function that produces the (dx, dy)
# offset relative to the entity's current facing vector.
_RELATIVE_MOVE: dict[str, tuple[int, int]] = {
    # Forward (same direction as facing)
    "forward":  (1,  0),   # placeholder — resolved at runtime using actual facing
    "advance":  (1,  0),
    "ahead":    (1,  0),
    # Backward
    "backward": (-1, 0),
    "back":     (-1, 0),
    "retreat":  (-1, 0),
    "reverse":  (-1, 0),
    # Strafe left (90° CCW from facing in grid coords)
    "left":     (0, -1),
    "strafe_left": (0, -1),
    "strafe left": (0, -1),
    # Strafe right (90° CW from facing)
    "right":    (0,  1),
    "strafe_right": (0, 1),
    "strafe right": (0, 1),
}


def _resolve_relative_move(keyword: str, actor: "EntityState") -> Optional[Coord]:
    """
    Translate a relative-direction keyword ("forward", "back", "left", "right")
    into an absolute grid Coord one step from the actor, based on their
    current facing.  Returns None if the keyword is not a relative keyword.
    """
    kw = keyword.lower().strip()
    if kw not in _RELATIVE_MOVE:
        return None
    rel = _RELATIVE_MOVE[kw]
    fx, fy = actor.facing.vector  # unit vector of facing direction
    # Build a local coordinate frame from facing:
    #   forward_axis = (fx, fy)
    #   left_axis    = (-fy, fx)   (90° CCW rotation)
    #   right_axis   = (fy, -fx)   (90° CW rotation)
    fwd_x, fwd_y = fx, fy
    lft_x, lft_y = -fy, fx
    # rel = (fwd_component, left_component)
    fwd_mul, lat_mul = rel[0], rel[1]
    dx = fwd_mul * fwd_x + lat_mul * lft_x
    dy = fwd_mul * fwd_y + lat_mul * lft_y
    return Coord(x=actor.position.x + dx, y=actor.position.y + dy)


_EDGE_KIND_ALIASES: dict[str, str] = {
    "friendly": EdgeKind.RESPECTS.value,
    "hostile": EdgeKind.ENEMY_OF.value,
    "allied": EdgeKind.ALLY_OF.value,
    "enemy": EdgeKind.ENEMY_OF.value,
    "ally": EdgeKind.ALLY_OF.value,
    "trusts": EdgeKind.RESPECTS.value,
    "trust": EdgeKind.RESPECTS.value,
    "fears": EdgeKind.FEARS.value,
    "fear": EdgeKind.FEARS.value,
    "distrust": EdgeKind.DISTRUSTS.value,
    "wary": EdgeKind.WARY_OF.value,
    "intimate": EdgeKind.INTIMATE.value,
    "interacted": EdgeKind.INTERACTED.value,
    "threatened": EdgeKind.THREATENED.value,
    "owes_debt": EdgeKind.OWES_DEBT.value,
    "respects": EdgeKind.RESPECTS.value,
    "respect": EdgeKind.RESPECTS.value,
    "enemy_of": EdgeKind.ENEMY_OF.value,
    "ally_of": EdgeKind.ALLY_OF.value,
    "distrusts": EdgeKind.DISTRUSTS.value,
    "wary_of": EdgeKind.WARY_OF.value,
}


def _normalize_edge_transition_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Ensure edge transitions use ``edge_kind`` with a valid EdgeKind value."""
    out = dict(payload)
    raw_kind = out.get("edge_kind") or out.get("kind")
    if raw_kind is None:
        normalized = EdgeKind.INTERACTED.value
    else:
        key = str(raw_kind).lower().strip()
        normalized = _EDGE_KIND_ALIASES.get(key, key)
        try:
            normalized = EdgeKind(normalized).value
        except ValueError:
            normalized = EdgeKind.INTERACTED.value
    out["edge_kind"] = normalized
    if "kind" in out and out["kind"] != normalized:
        del out["kind"]
    return out


__all__ = [
    "current_carry_weight",
    "_check_projection_firewall",
    "_contest_roll",
    "_encumbrance_ratio",
    "_entity_by_name",
    "_grid_for_action",
    "_normalize_edge_transition_payload",
    "_resolve_entity_target",
    "_resolve_relative_move",
    "_seeded_float",
    "_EDGE_KIND_ALIASES",
]

