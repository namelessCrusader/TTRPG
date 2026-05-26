"""compile_action entry point."""

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
from .common import _check_projection_firewall
from .verbs import (
    _BUILTIN_DEFAULTS,
    _KERNEL_COMPILERS,
    _compile_from_template,
    _compile_generic,
)

def compile_action(
    action: SemanticAction,
    world: WorldState,
    *,
    projection: Optional[SemanticProjection] = None,
) -> ValidationResult:
    """
    Validate and compile a SemanticAction into concrete Transitions.

    This function is DETERMINISTIC.  It contains no LM calls.
    If a projection is supplied, the firewall check is enforced.

    Parameters
    ----------
    action     : The action to compile.
    world      : Canonical world state (read-only from this function's POV).
    projection : Optional; if supplied, the projection firewall is enforced.

    Returns
    -------
    ValidationResult with valid=True and transitions, or valid=False with
    a rejection reason.
    """
    # Projection firewall
    try:
        _check_projection_firewall(action, projection)
    except ProjectionFirewallError as exc:
        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.ENTITY_NOT_VISIBLE,
            rejection_detail=str(exc),
        )

    # Condition gate — some conditions block certain verbs entirely.
    try:
        actor = world.grid_for_entity(action.actor).entities.get(action.actor)
    except KeyError:
        actor = world.spatial.entities.get(action.actor)
    if actor is not None:
        if "stunned" in actor.conditions:
            blocking_verbs = {
                ActionType.MOVE, ActionType.ATTACK, ActionType.THROW,
                "move", "attack", "throw", "flee", "run", "jump",
            }
            if action.verb in blocking_verbs:
                return ValidationResult(
                    valid=False,
                    rejection_reason=RejectionReason.PHYSICALLY_IMPOSSIBLE,
                    rejection_detail=f"Actor is stunned for {actor.conditions['stunned']} more tick(s).",
                )
        if not actor.alive:
            return ValidationResult(
                valid=False,
                rejection_reason=RejectionReason.ACTOR_NOT_FOUND,
                rejection_detail="Actor is dead and cannot take actions.",
            )

    # ── Kernel verbs: bespoke compilers with action-specific physics ──
    # Every other verb routes to _compile_generic, which honours pack
    # verb templates and LM-proposed effects without ever switching on
    # the verb string itself.
    template = world.config.verb_templates.get(action.verb)
    if (
        template is not None
        and action.verb not in ActionType.KERNEL
        and action.verb not in {"pour_drink", "drink"}
    ):
        result = _compile_from_template(action, world, template)
    else:
        compiler_fn = _KERNEL_COMPILERS.get(action.verb)
        if compiler_fn is not None:
            result = compiler_fn(action, world)
        else:
            result = _compile_generic(action, world)

    if result.valid:
        from ..speech_utils import (
            action_needs_spoken_line,
            best_speech_line,
            validate_spoken_line,
        )

        if (
            action_needs_spoken_line(action)
            and actor is not None
            and action.verb not in world.config.verb_templates
        ):
            goals = list(getattr(actor, "goals", None) or [])
            line = best_speech_line(action, goals=goals)
            emo = (
                actor.emotional_state.value
                if hasattr(actor.emotional_state, "value")
                else str(actor.emotional_state)
            )
            ok, detail = validate_spoken_line(
                line,
                goals=goals,
                personality=actor.personality or "",
                drive=actor.drive or "",
                role=actor.role or "",
                emotional_state=emo,
            )
            if not ok:
                return ValidationResult(
                    valid=False,
                    rejection_reason=RejectionReason.MALFORMED_ACTION,
                    rejection_detail=detail,
                )

        from ..consequences import finalize_compiled_result

        outcome = "success"
        for t in result.concrete_transitions:
            if t.kind == TransitionKind.DIALOGUE_SPOKEN:
                outcome = t.payload.get("contest_outcome", outcome)
                break
        result = finalize_compiled_result(
            action, world, result, contest_outcome=str(outcome),
        )
    return result


# ---------------------------------------------------------------------------
# Transition application (apply compiled transitions to canonical state)
# ---------------------------------------------------------------------------


