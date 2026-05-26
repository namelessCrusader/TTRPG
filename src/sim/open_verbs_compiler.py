"""
Pack-declared open verb rules (``open_verbs.yaml``).

All open-ended verbs resolve through this module — handler delegation for
physics-heavy compilers, declarative effects for everything else.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

from .schemas import (
    RejectionReason,
    ValidationResult,
    VerbTemplate,
)

if TYPE_CHECKING:
    from .schemas import OpenVerbRule, SemanticAction, WorldState

HandlerFn = Callable[["SemanticAction", "WorldState"], Optional[ValidationResult]]


def _handler_registry() -> dict[str, HandlerFn]:
    from . import generic_verb_rules as gvr

    return {
        "environmental": gvr._compile_environmental,
        "bribe": gvr._compile_bribe,
        "sabotage": gvr._compile_sabotage,
        "repair": gvr._compile_repair,
        "forge": gvr._compile_forge,
        "blackmail": gvr._compile_blackmail,
        "seduce": lambda a, w: gvr._compile_social_influence(a, w, intimate=True),
        "manipulate": lambda a, w: gvr._compile_social_influence(a, w, intimate=False),
        "plant": gvr._compile_plant,
        "disguise": gvr._compile_disguise,
        "search": gvr._compile_search,
        "listen": gvr._compile_listen,
        "pick_lock": gvr._compile_pick_lock,
        "intimidate": gvr._compile_intimidate,
        "destroy": gvr._compile_destroy,
        "hire": gvr._compile_hire,
        "gift": gvr._compile_gift,
    }


def _rule_matches(rule: "OpenVerbRule", verb: str, text: str) -> bool:
    if verb in rule.verbs:
        return True
    for v in rule.verbs:
        if v and v in verb:
            return True
    for kw in rule.keywords:
        if kw.lower() in text:
            return True
    return False


def _check_requires(
    rule: "OpenVerbRule",
    action: "SemanticAction",
    world: "WorldState",
) -> Optional[ValidationResult]:
    req = (rule.requires or "").lower().strip()
    if not req:
        return None
    if req == "entity":
        if not isinstance(action.target, str):
            return ValidationResult(
                valid=False,
                rejection_reason=RejectionReason.MALFORMED_ACTION,
                rejection_detail="This action requires an entity target.",
            )
        from .generic_verb_rules import _resolve_target_entity

        if _resolve_target_entity(action, world) is None:
            return ValidationResult(
                valid=False,
                rejection_reason=RejectionReason.TARGET_NOT_FOUND,
            )
    elif req == "object":
        if not isinstance(action.target, str):
            return ValidationResult(
                valid=False,
                rejection_reason=RejectionReason.MALFORMED_ACTION,
                rejection_detail="This action requires an object target.",
            )
        from .generic_verb_rules import _resolve_target_object

        if _resolve_target_object(action, world) is None:
            return ValidationResult(
                valid=False,
                rejection_reason=RejectionReason.TARGET_NOT_FOUND,
            )
    return None


def _rule_to_template(rule: "OpenVerbRule", verb: str) -> VerbTemplate:
    return VerbTemplate(
        verb=verb,
        contest=rule.contest,
        effects_on_success=list(rule.on_success),
        effects_on_failure=list(rule.on_failure),
        narrative_success=rule.narrative_success,
        narrative_failure=rule.narrative_failure,
    )


def try_pack_open_verb(
    action: "SemanticAction",
    world: "WorldState",
) -> Optional[ValidationResult]:
    """Try pack ``open_verbs`` rules. Returns None if no rule matched."""
    rules = world.config.open_verbs
    if not rules:
        return None

    verb = str(action.verb).lower().split(".")[-1]
    text = (action.raw_input or "").lower()

    for rule in rules:
        if not _rule_matches(rule, verb, text):
            continue

        reject = _check_requires(rule, action, world)
        if reject is not None:
            return reject

        if rule.handler:
            handler = _handler_registry().get(rule.handler.lower())
            if handler is None:
                continue
            return handler(action, world)

        if rule.on_success or rule.contest:
            from .compiler.effects import compile_effects

            return compile_effects(
                action, world, template=_rule_to_template(rule, verb),
            )

    return None


__all__ = ["try_pack_open_verb"]
