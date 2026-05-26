"""
Unified verb resolution pipeline for verbs without a kernel compiler.

Precedence (first match wins):
  1. Pack ``verb_templates.yaml``                  (author intent — most specific)
  2. Pack ``open_verbs.yaml``                      (author intent)
  3. Property-intersection rules                   (deterministic physics — tag × tag)
  4. Engine built-in defaults (``_BUILTIN_DEFAULTS``)  (generic per-verb fallback)
  5. Contact-category reroute (kiss, hug, shove, …)
  6. Pure freeform (contest + LM proposals + interaction record)

Why property-intersection beats built-ins
-----------------------------------------
Built-in defaults are generic per-verb compilers (``OPEN``, ``CLOSE``,
``FLEE``, ``INTIMIDATE`` …) that treat the verb in isolation.  Property
rules know *what the things involved actually are* (corrosive vs metal,
fire-source vs flammable, crowbar vs locked door) and produce a more
specific physical outcome.  When both could fire, the physically-specific
rule wins so creative material/object combinations feel mechanically real
instead of degrading to a generic verb behaviour.

Pack-authored content (templates + open_verbs) still wins above property
rules because explicit author intent is the highest signal — if a world
designer wrote "console NPC", that intent should not be hijacked by an
incidental tag overlap.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..schemas import (
    ActionType,
    IntentBlock,
    RejectionReason,
    StyleBlock,
    ValidationResult,
)

if TYPE_CHECKING:
    from ..schemas import SemanticAction, WorldState


def compile_unresolved_verb(
    action: "SemanticAction",
    world: "WorldState",
) -> ValidationResult:
    """Compile a verb that did not match a kernel compiler in dispatch."""
    from .verbs import (
        _BUILTIN_DEFAULTS,
        _compile_contact,
        _compile_freeform,
        _compile_from_template,
        _contact_verb_aggression,
        _grid_for_action,
        _interaction_record,
        _validate_proposals,
    )

    grid = _grid_for_action(world, action)
    actor = grid.entities.get(action.actor)
    if actor is None:
        return ValidationResult(
            valid=False, rejection_reason=RejectionReason.ACTOR_NOT_FOUND
        )

    # 1 — pack verb template
    template = world.config.verb_templates.get(action.verb)
    if template is not None:
        return _compile_from_template(action, world, template)

    # 2 — pack open verbs (handlers + declarative effects)
    from ..open_verbs_compiler import try_pack_open_verb

    pack_open = try_pack_open_verb(action, world)
    if pack_open is not None:
        if pack_open.valid:
            extras = _validate_proposals(action.proposed_effects, world, action)
            return ValidationResult(
                valid=True,
                concrete_transitions=list(pack_open.concrete_transitions) + extras,
                plausibility=pack_open.plausibility,
            )
        return pack_open

    # 3 — property-intersection rules (deterministic physics: tag × tag).
    # Promoted above built-in defaults so that material/object combinations
    # produce a specific physical outcome instead of a generic verb effect.
    # Social / observation / idle verbs are filtered inside the resolver
    # itself via intent-category rules, so this never hijacks speech.
    from ..property_interaction_resolver import try_property_interaction

    prop_result = try_property_interaction(action, world)
    if prop_result is not None:
        if prop_result.valid:
            extras = _validate_proposals(action.proposed_effects, world, action)
            return ValidationResult(
                valid=True,
                concrete_transitions=list(prop_result.concrete_transitions) + extras,
                plausibility=prop_result.plausibility,
            )
        return prop_result

    # 4 — engine built-in defaults (generic per-verb fallback)
    builtin = _BUILTIN_DEFAULTS.get(action.verb)
    if builtin is not None:
        result = builtin(action, world)
        if not result.valid:
            return result
        extras = _validate_proposals(action.proposed_effects, world, action)
        return ValidationResult(
            valid=True,
            concrete_transitions=list(result.concrete_transitions) + extras,
            plausibility=result.plausibility,
            rejection_reason=result.rejection_reason,
            rejection_detail=result.rejection_detail,
        )

    # 5 — contact-category reroute
    if isinstance(action.target, str) and action.target in grid.entities:
        contact_aggression = _contact_verb_aggression(action.verb)
        if contact_aggression is not None:
            contact_action = action.model_copy(
                update={
                    "verb": ActionType.CONTACT,
                    "style": StyleBlock(
                        aggression=contact_aggression,
                        emotional_tone=(
                            action.style.emotional_tone
                            if action.style else "neutral"
                        ),
                    ),
                    "intent": IntentBlock(
                        manner=action.verb,
                        desired_outcome=[action.verb],
                    ),
                }
            )
            contact_result = _compile_contact(contact_action, world)
            if contact_result.valid:
                contact_result.concrete_transitions.append(
                    _interaction_record(action, "success")
                )
                return contact_result

    # 6 — freeform fallback
    return _compile_freeform(action, world)


__all__ = ["compile_unresolved_verb"]
