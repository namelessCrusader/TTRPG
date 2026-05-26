"""Compile pour_drink and drink with presence + fluid physics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .fluids import (
    DEFAULT_POUR_ML,
    drink_from_container,
    find_pour_source,
    fluid_volume_ml,
    is_receptacle,
    resolve_receptacle_for_pour,
    transfer_pour,
)
from .schemas import (
    ObjectId,
    RejectionReason,
    SemanticAction,
    Transition,
    TransitionKind,
    ValidationResult,
)

if TYPE_CHECKING:
    from .schemas import WorldState


def _compile_pour_drink(action: SemanticAction, world: "WorldState") -> ValidationResult:
    grid = world.spatial
    actor = grid.entities.get(action.actor)
    if actor is None:
        return ValidationResult(valid=False, rejection_reason=RejectionReason.ACTOR_NOT_FOUND)

    source = find_pour_source(actor, grid, max_dist=3)
    if source is None:
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.TARGET_NOT_FOUND,
            rejection_detail="No tap, cask, or pour source within reach.",
        )

    target_raw = str(action.target) if action.target else None
    receptacle, detail = resolve_receptacle_for_pour(grid, actor, target_raw)
    if receptacle is None:
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.TARGET_NOT_FOUND,
            rejection_detail=detail or "Nothing to pour into.",
        )

    reach = 3
    if {"staff", "bartender"}.intersection(t.lower() for t in (actor.tags or [])):
        reach = 8
    if actor.position.manhattan(receptacle.position or actor.position) > reach:
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.PHYSICALLY_IMPOSSIBLE,
            rejection_detail="Too far from the cup or glass.",
        )

    fluid_transitions, outcome = transfer_pour(
        world, actor, source, receptacle, DEFAULT_POUR_ML,
    )
    if outcome == "empty_source":
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.PHYSICALLY_IMPOSSIBLE,
            rejection_detail=f"The {source.name} is empty.",
        )

    transitions: list[Transition] = list(fluid_transitions)

    from .compiler import _interaction_record
    transitions.append(_interaction_record(action, "success" if outcome != "overflow" else "failure"))

    return ValidationResult(
        valid=True,
        concrete_transitions=transitions,
        plausibility=0.9 if outcome == "success" else 0.7,
    )


def _compile_drink(action: SemanticAction, world: "WorldState") -> ValidationResult:
    grid = world.spatial
    actor = grid.entities.get(action.actor)
    if actor is None:
        return ValidationResult(valid=False, rejection_reason=RejectionReason.ACTOR_NOT_FOUND)

    container = None
    if action.target:
        oid = ObjectId(str(action.target))
        if oid in grid.objects:
            container = grid.objects[oid]
        elif oid in actor.inventory:
            container = grid.objects[oid]

    if container is None:
        for oid in actor.inventory:
            obj = grid.objects.get(oid)
            if obj and is_receptacle(obj) and fluid_volume_ml(obj) > 0:
                container = obj
                break

    if container is None:
        from .fluids import find_receptacle_near
        container = find_receptacle_near(grid, actor.position, max_dist=1)

    if container is None or not is_receptacle(container):
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.TARGET_NOT_FOUND,
            rejection_detail="Nothing drinkable in hand or within reach.",
        )

    if fluid_volume_ml(container) <= 0:
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.PHYSICALLY_IMPOSSIBLE,
            rejection_detail=f"The {container.name} is empty.",
        )

    if container.position is not None:
        if actor.position.manhattan(container.position) > 2:
            return ValidationResult(
                valid=False,
                rejection_reason=RejectionReason.PHYSICALLY_IMPOSSIBLE,
                rejection_detail="The drink is too far away.",
            )

    transitions, outcome = drink_from_container(world, actor, container)
    if outcome != "success":
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.PHYSICALLY_IMPOSSIBLE,
            rejection_detail="You cannot drink that.",
        )

    from .compiler import _interaction_record
    transitions.append(_interaction_record(action, "success"))
    return ValidationResult(valid=True, concrete_transitions=transitions, plausibility=0.95)
