"""Validate LM-proposed effects."""

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

def _validate_proposals(
    proposals: list["TransitionProposal"],
    world: "WorldState",
    action: "SemanticAction",
) -> list["Transition"]:
    out: list[Transition] = []
    for p in proposals:
        ok, transition_or_reason = _validate_proposal(p, world, action)
        if ok:
            out.append(transition_or_reason)  # type: ignore[arg-type]
    return out


def _substitute_proposal(
    prop: "TransitionProposal", subs: dict[str, Any]
) -> "TransitionProposal":
    """Replace $actor / $target placeholders in payload string values."""
    new_payload: dict[str, Any] = {}
    for k, v in prop.payload.items():
        if isinstance(v, str) and v in subs:
            sub = subs[v]
            if sub is not None:
                new_payload[k] = sub
            else:
                # Substitution requested but no value available — drop the
                # key, downstream validation will reject the proposal.
                new_payload[k] = None
        else:
            new_payload[k] = v
    return TransitionProposal(kind=prop.kind, payload=new_payload)


def _validate_proposal(
    prop: "TransitionProposal",
    world: "WorldState",
    action: "SemanticAction",
) -> tuple[bool, "Transition | str"]:
    """
    Validate a TransitionProposal against canonical state.

    Returns (True, Transition) if the proposal can be applied as-is
    (possibly with values clipped), or (False, reason) if it must be
    dropped.

    The engine NEVER trusts a proposal's references blindly:
      • entity / object ids must exist in canonical state;
      • numeric deltas are clipped to the field's valid range;
      • edge endpoints must exist;
      • health / attribute changes can't move values out of bounds.
    """
    payload = dict(prop.payload)

    try:
        kind = TransitionKind(prop.kind)
    except ValueError:
        return False, f"unknown transition kind '{prop.kind}'"

    try:
        grid = world.grid_for_entity(action.actor)
    except KeyError:
        grid = world.spatial

    from ..region_utils import entity_or_none

    def _entity_exists(maybe_id: Any) -> bool:
        if maybe_id is None:
            return False
        eid = EntityId(str(maybe_id))
        if eid in grid.entities:
            return True
        return entity_or_none(world, eid) is not None

    def _get_entity(maybe_id: Any) -> Optional[EntityState]:
        if maybe_id is None:
            return None
        eid = EntityId(str(maybe_id))
        ent = grid.entities.get(eid)
        if ent is not None:
            return ent
        return entity_or_none(world, eid)

    def _anchor_actor_tile(payload: dict[str, Any], actor_ent: Optional[EntityState]) -> None:
        """Pack templates use x:0,y:0 as sentinel for 'actor's tile'."""
        if actor_ent is None:
            return
        x, y = payload.get("x"), payload.get("y")
        if x is not None and y is not None and int(x) == 0 and int(y) == 0:
            payload["x"] = actor_ent.position.x
            payload["y"] = actor_ent.position.y
            payload.setdefault("z", actor_ent.position.z)

    # Generic entity reference validation (removed duplicate _entity_exists below)

    if kind == TransitionKind.ENTITY_HEALTH_CHANGED:
        eid = payload.get("entity_id")
        if not _entity_exists(eid):
            return False, f"unknown entity_id '{eid}' for health change"
        # Clip delta so the LM can't propose a one-shot kill via prose.
        delta = int(payload.get("delta", 0))
        max_delta = 10  # generic interactions can't crit
        delta = max(-max_delta, min(max_delta, delta))
        payload["delta"] = delta

    elif kind == TransitionKind.ENTITY_EMOTIONAL_STATE_CHANGED:
        eid = payload.get("entity_id")
        if not _entity_exists(eid):
            return False, f"unknown entity_id '{eid}' for emotion change"
        to_val = payload.get("to")
        try:
            EmotionalState(to_val)
        except ValueError:
            return False, f"invalid emotional state '{to_val}'"
        if "from" not in payload:
            ent = _get_entity(eid)
            if ent is not None:
                payload["from"] = ent.emotional_state.value

    elif kind == TransitionKind.ENTITY_ALERTNESS_CHANGED:
        eid = payload.get("entity_id")
        if not _entity_exists(eid):
            return False, f"unknown entity_id '{eid}' for alertness change"
        to_val = payload.get("to")
        try:
            AlertnessLevel(to_val)
        except ValueError:
            return False, f"invalid alertness level '{to_val}'"
        if "from" not in payload:
            ent = _get_entity(eid)
            if ent is not None:
                payload["from"] = ent.alertness.value

    elif kind == TransitionKind.ENTITY_CONDITION_CHANGED:
        item_only = (
            (payload.get("_item_tag_add") or payload.get("_item_tag_remove"))
            and not payload.get("condition")
        )
        if not item_only:
            if not payload.get("condition"):
                return False, "ENTITY_CONDITION_CHANGED requires condition"
            eid_raw = (
                payload.get("entity_id") or payload.get("actor") or str(action.actor)
            )
            if not _entity_exists(eid_raw):
                return False, f"unknown entity_id '{eid_raw}' for condition change"
            payload["entity_id"] = str(eid_raw)
            payload["ticks"] = max(0, min(50, int(payload.get("ticks", 3))))

    elif kind == TransitionKind.ENTITY_STAT_CHANGED:
        eid_raw = payload.get("entity_id") or payload.get("actor") or str(action.actor)
        if not _entity_exists(eid_raw):
            return False, f"unknown entity_id '{eid_raw}' for stat change"
        payload["entity_id"] = str(eid_raw)
        if "stat" not in payload:
            return False, "ENTITY_STAT_CHANGED requires stat"
        delta = float(payload.get("delta", 0))
        payload["delta"] = max(-50.0, min(50.0, delta))

    elif kind in (
        TransitionKind.EDGE_CREATED,
        TransitionKind.EDGE_UPDATED,
        TransitionKind.EDGE_REMOVED,
    ):
        payload = _normalize_edge_transition_payload(payload)
        src = payload.get("source")
        tgt = payload.get("target")
        # Edge endpoints may be entities or factions; accept if they
        # are known in either the spatial grid or the relational graph.
        known = set(grid.entities.keys()) | set(world.relational.nodes.keys())
        if src not in known or tgt not in known:
            return False, f"edge references unknown nodes '{src}'→'{tgt}'"
        if src == tgt:
            return False, f"edge cannot be self-loop '{src}'→'{tgt}'"
        # Clip weight to [-1, 1] if present.
        if "weight" in payload:
            payload["weight"] = max(-1.0, min(1.0, float(payload["weight"])))
        if kind == TransitionKind.EDGE_UPDATED and "delta" in payload:
            payload["delta"] = max(-0.5, min(0.5, float(payload["delta"])))

    elif kind == TransitionKind.DIALOGUE_SPOKEN:
        # Free-form, no validation beyond actor existence.
        if "actor" in payload and not _entity_exists(payload["actor"]):
            return False, f"unknown speaker '{payload.get('actor')}'"

    elif kind == TransitionKind.TILE_MARKED:
        actor_ent = _get_entity(action.actor)
        _anchor_actor_tile(payload, actor_ent)
        x, y = payload.get("x"), payload.get("y")
        if x is None or y is None:
            return False, "TILE_MARKED requires x and y"
        from ..schemas import Coord as _Coord
        coord = _Coord(x=int(x), y=int(y))
        if not grid.is_in_bounds(coord):
            return False, f"TILE_MARKED coordinate {x},{y} out of bounds"
        # Actor must be within 2 tiles
        actor_ent = grid.entities.get(action.actor)
        if actor_ent is None:
            actor_ent = _get_entity(action.actor)
        if actor_ent:
            dist = actor_ent.position.manhattan(coord)
            if dist > 2:
                return False, f"TILE_MARKED: too far ({dist} tiles)"
        mark = str(payload.get("mark", ""))
        if not mark:
            return False, "TILE_MARKED: mark string is empty"
        # Sanitise mark to prevent injection — max 80 chars, no newlines
        payload["mark"] = mark[:80].replace("\n", " ")
        payload["author_entity_id"] = str(action.actor)

    elif kind in (TransitionKind.STRUCTURE_CREATED, TransitionKind.STRUCTURE_MODIFIED):
        x, y = payload.get("x"), payload.get("y")
        if x is None or y is None:
            return False, f"{kind.value} requires x and y"
        from ..schemas import Coord as _Coord
        coord = _Coord(x=int(x), y=int(y))
        if not grid.is_in_bounds(coord):
            return False, f"{kind.value} coordinate {x},{y} out of bounds"
        actor_ent = grid.entities.get(action.actor)
        if actor_ent is None:
            actor_ent = _get_entity(action.actor)
        if actor_ent:
            dist = actor_ent.position.manhattan(coord)
            if dist > 2:
                return False, f"{kind.value}: too far ({dist} tiles)"
        if kind == TransitionKind.STRUCTURE_CREATED:
            # Must have a name
            if not payload.get("name"):
                return False, "STRUCTURE_CREATED: name is required"
            # Cap attributes
            for k in list((payload.get("attributes") or {}).keys()):
                v = payload["attributes"][k]
                if isinstance(v, (int, float)):
                    payload["attributes"][k] = min(int(v), 50)

    elif kind == TransitionKind.ENVIRONMENT_STATE_CHANGED:
        key = payload.get("key")
        if not key:
            return False, "ENVIRONMENT_STATE_CHANGED: key is required"
        actor_ent = _get_entity(action.actor)
        _anchor_actor_tile(payload, actor_ent)
        x, y = payload.get("x"), payload.get("y")
        if x is not None and y is not None:
            from ..schemas import Coord as _Coord
            coord = _Coord(x=int(x), y=int(y))
            if not grid.is_in_bounds(coord):
                return False, f"ENVIRONMENT_STATE_CHANGED coord {x},{y} out of bounds"
            actor_ent = _get_entity(action.actor)
            if actor_ent and actor_ent.position.manhattan(coord) > 3:
                return False, "ENVIRONMENT_STATE_CHANGED: tile too far"
        payload["author"] = str(action.actor)

    elif kind == TransitionKind.ENTITY_MOVED:
        # The LM is NOT allowed to move entities via proposed effects —
        # MOVE has its own kernel compiler with pathing.
        return False, "ENTITY_MOVED is kernel-only; use verb='move' instead"

    elif kind == TransitionKind.ITEM_TRANSFERRED:
        # Same reasoning as MOVE — inventory transfer is kernel-only.
        return False, "ITEM_TRANSFERRED is kernel-only; use give/take"

    elif kind in (
        TransitionKind.ENTITY_GOAL_ADDED,
        TransitionKind.ENTITY_GOAL_REMOVED,
    ):
        eid_raw = payload.get("entity_id") or payload.get("actor") or str(action.actor)
        if not _entity_exists(eid_raw):
            return False, f"unknown entity_id '{eid_raw}' for goal change"
        goal = str(payload.get("goal", "")).strip()
        if not goal:
            return False, f"{kind.value} requires non-empty goal text"
        payload["entity_id"] = str(eid_raw)
        payload["goal"] = goal[:240]

    # Other transition kinds pass through unchanged.
    return True, Transition(kind=kind, payload=payload)


# ---------------------------------------------------------------------------
# Main compile entry point
# ---------------------------------------------------------------------------


