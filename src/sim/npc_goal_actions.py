"""
Deterministic goal pursuit — translate entity.goals into concrete actions.

When an NPC holds a goal like "Inspect the fresh mark here", this module
resolves a destination (tile mark, world fact, combat target) and emits
MOVE / EXAMINE / SPEAK candidates with high weight.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Optional

from .schemas import ActionType, Coord, EntityId, IntentBlock, SemanticAction

if TYPE_CHECKING:
    from .schemas import EntityState, WorldState

_INVESTIGATE_RE = re.compile(
    r"\b(investigate|inspect|examine|look into|check|find out|search|follow up)\b",
    re.I,
)
_LOCATE_RE = re.compile(
    r"\b(find|locate|track|recover|get back|retrieve)\b",
    re.I,
)
_AVENGE_RE = re.compile(r"\b(avenge|justice|kill|bring down)\b", re.I)
_REPORT_RE = re.compile(r"\b(report|tell|warn|alert)\b", re.I)


def _parse_tile_key(subject_id: str) -> Optional[Coord]:
    parts = subject_id.split(",")
    if len(parts) < 2:
        return None
    try:
        return Coord(
            x=int(parts[0]),
            y=int(parts[1]),
            z=int(parts[2]) if len(parts) > 2 else 0,
        )
    except ValueError:
        return None


def _destination_for_goal(
    goal: str,
    entity: "EntityState",
    world: "WorldState",
) -> tuple[Optional[Coord], str]:
    """Return (coord, reason) for a goal, or (None, '')."""
    grid = world.spatial
    goal_lower = goal.lower()

    # Tile marks visible from current position
    from .spatial import visible_from

    for coord in visible_from(grid, entity.position, entity.sight_range):
        tile = grid.tile_at(coord)
        if tile.marks:
            for mark in tile.marks:
                if any(w in mark.lower() for w in goal_lower.split() if len(w) > 4):
                    return coord, f"mark: {mark[:40]}"
            if _INVESTIGATE_RE.search(goal):
                return coord, f"mark: {tile.marks[-1][:40]}"

    # Recent world facts
    for fact in reversed(world.world_facts[-20:]):
        if fact.scope.value == "tile" and fact.subject_id:
            coord = _parse_tile_key(fact.subject_id)
            if coord is not None:
                if any(w in fact.claim.lower() for w in goal_lower.split() if len(w) > 4):
                    return coord, fact.claim[:50]
                if _INVESTIGATE_RE.search(goal):
                    return coord, fact.claim[:50]

    # Combat / avenge goals → move toward combat_target
    if _AVENGE_RE.search(goal):
        ct = entity.meta.get("combat_target")
        if ct:
            tgt = grid.entities.get(EntityId(str(ct)))
            if tgt and tgt.alive:
                return tgt.position, f"confront {tgt.name}"

    # Entity-scoped facts
    for fact in reversed(world.world_facts[-15:]):
        if fact.scope.value == "entity" and fact.subject_id:
            subj = grid.entities.get(EntityId(fact.subject_id))
            if subj and subj.alive and _LOCATE_RE.search(goal):
                return subj.position, fact.claim[:50]

    return None, ""


def _step_toward(
    entity: "EntityState",
    world: "WorldState",
    dest: Coord,
) -> Optional[Coord]:
    from .spatial import find_path

    grid = world.spatial
    if entity.position.manhattan(dest) <= 1:
        return dest
    path = find_path(grid, entity.position, dest, max_steps=24)
    if path:
        return path[0]
    return None


def goal_driven_candidates(
    entity: "EntityState",
    world: "WorldState",
) -> list[tuple[SemanticAction, str, float]]:
    """
    Return high-weight candidates that advance the entity's top goal.
    """
    goals: list[str] = list(getattr(entity, "goals", None) or [])
    if not goals:
        return []

    eid = entity.entity_id
    active = goals[0]
    dest, reason = _destination_for_goal(active, entity, world)
    if dest is None:
        return []

    candidates: list[tuple[SemanticAction, str, float]] = []
    dist = entity.position.manhattan(dest)

    if dist <= 1:
        candidates.append((
            SemanticAction(
                verb=ActionType.EXAMINE,
                actor=eid,
                target=dest,
                intent=IntentBlock(rationale=active, manner="thorough"),
                raw_input="[npc_auto:goal_examine]",
            ),
            f"goal: examine — {reason}",
            0.91,
        ))
    else:
        step = _step_toward(entity, world, dest)
        if step is not None:
            candidates.append((
                SemanticAction(
                    verb=ActionType.MOVE,
                    actor=eid,
                    target=step,
                    intent=IntentBlock(rationale=active),
                    raw_input="[npc_auto:goal_move]",
                ),
                f"goal: move toward — {reason}",
                0.90,
            ))

    if _REPORT_RE.search(active):
        grid = world.spatial
        for other_id, other in grid.entities.items():
            if other_id == eid or not other.alive:
                continue
            if entity.position.manhattan(other.position) <= entity.sight_range:
                candidates.append((
                    SemanticAction(
                        verb=ActionType.SPEAK,
                        actor=eid,
                        target=other_id,
                        intent=IntentBlock(
                            rationale=active,
                            manner="urgent",
                            desired_outcome=["report"],
                        ),
                        raw_input="[npc_auto:goal_report]",
                    ),
                    f"goal: report to {other.name} — {active[:40]}",
                    0.88,
                ))
                break

    return candidates


__all__ = ["goal_driven_candidates"]
