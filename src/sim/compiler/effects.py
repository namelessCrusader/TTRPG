"""
Unified effect DSL — compile declarative TransitionProposals into Transitions.

Used by verb_templates.yaml, open_verbs.yaml, adjudication templates, and
LM consequence proposals (after validation).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from ..schemas import (
    ContestSpec,
    ObjectId,
    TransitionProposal,
    ValidationResult,
    VerbTemplate,
)
from .proposals import _substitute_proposal, _validate_proposals

if TYPE_CHECKING:
    from ..schemas import EntityState, SemanticAction, WorldState


def substitution_table(
    action: "SemanticAction",
    target_entity: "EntityState | None",
) -> dict[str, Any]:
    return {
        "$actor": action.actor,
        "$target": (
            target_entity.entity_id if target_entity
            else (str(action.target) if isinstance(action.target, str) else None)
        ),
    }


def compile_effects(
    action: "SemanticAction",
    world: "WorldState",
    *,
    contest: ContestSpec | None = None,
    effects_on_success: list[TransitionProposal] | None = None,
    effects_on_failure: list[TransitionProposal] | None = None,
    template: VerbTemplate | None = None,
    object_target_record_only: bool = True,
) -> ValidationResult:
    """
    Resolve contest, substitute $actor/$target, validate proposals, record interaction.

    Pass either ``template`` or explicit contest/effect lists (template wins).
    """
    from .verbs import (
        _grid_for_action,
        _interaction_record,
        _resolve_contest,
        _resolve_entity_target,
    )

    if template is not None:
        contest = template.contest
        effects_on_success = list(template.effects_on_success)
        effects_on_failure = list(template.effects_on_failure)

    effects_on_success = list(effects_on_success or [])
    effects_on_failure = list(effects_on_failure or [])

    grid = _grid_for_action(world, action)
    target_entity: EntityState | None = None
    target_object = None
    if isinstance(action.target, str):
        target_object = grid.objects.get(ObjectId(action.target))
        if target_object is None:
            target_entity = _resolve_entity_target(grid, action.target)

    if (
        object_target_record_only
        and target_object is not None
        and target_entity is None
    ):
        return ValidationResult(
            valid=True,
            concrete_transitions=[_interaction_record(action, "success")],
            plausibility=0.85,
        )

    if action.contest is not None or contest is not None:
        contest_outcome = _resolve_contest(action, world, target_entity)
    else:
        contest_outcome = "success"

    chosen = (
        effects_on_success if contest_outcome == "success" else effects_on_failure
    )
    subs = substitution_table(action, target_entity)
    substituted = [_substitute_proposal(p, subs) for p in chosen]
    substituted.extend(action.proposed_effects)

    transitions = _validate_proposals(substituted, world, action)
    transitions.append(_interaction_record(action, contest_outcome))

    return ValidationResult(
        valid=True,
        concrete_transitions=transitions,
        plausibility=0.9 if contest_outcome == "success" else 0.5,
    )


__all__ = ["compile_effects", "substitution_table"]
