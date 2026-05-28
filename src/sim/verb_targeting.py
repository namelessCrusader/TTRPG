"""
Infer whether pack verbs require an entity target from verb_templates.yaml.

Replaces maintaining a static frozenset that drifts from pack content.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Optional

from .schemas import SemanticAction, VerbTemplate, WorldState

if TYPE_CHECKING:
    pass

# Legacy pack verbs not always present in merged templates (kernel-adjacent).
_STATIC_ENTITY_TARGET: frozenset[str] = frozenset({
    "console", "tease", "flirt", "compliment", "gossip", "haggle", "barter",
    "threaten", "pour_drink", "perform", "eavesdrop", "accuse", "wink",
    "shrug", "slam_fist", "toast", "greet", "warn", "challenge", "teach",
    "apologize", "thank", "forgive", "bow_to", "wave", "salute",
    "encourage", "mock", "scoff", "joke", "promise", "vow", "lie",
    "agree", "disagree", "smile", "frown", "glare", "nod", "examine",
})

_STATIC_SELF_ONLY: frozenset[str] = frozenset({
    "disguise", "investigate", "rest", "wait", "observe",
})


def _normalize_verb(verb: object) -> str:
    return str(verb).lower().split(".")[-1]


def _payload_references_target(payload: dict) -> bool:
    blob = json.dumps(payload, default=str)
    return "$target" in blob or '"target"' in blob and "entity_id" in blob


def template_requires_entity_target(tmpl: VerbTemplate) -> bool:
    """True when template effects or contest need another entity."""
    if tmpl.contest is not None and tmpl.contest.target_attribute:
        return True
    for prop in tmpl.effects_on_success + tmpl.effects_on_failure:
        if _payload_references_target(prop.payload):
            return True
    return False


def template_is_self_only(tmpl: VerbTemplate) -> bool:
    """True when the verb only mutates the actor (no $target in effects, no target contest)."""
    if template_requires_entity_target(tmpl):
        return False
    if not tmpl.effects_on_success and not tmpl.effects_on_failure:
        return True
    return True


def entity_target_verbs_for_world(world: WorldState) -> frozenset[str]:
    out = set(_STATIC_ENTITY_TARGET)
    for verb, tmpl in (world.config.verb_templates or {}).items():
        if template_requires_entity_target(tmpl):
            out.add(_normalize_verb(verb))
    return frozenset(out)


def self_only_verbs_for_world(world: WorldState) -> frozenset[str]:
    out = set(_STATIC_SELF_ONLY)
    for verb, tmpl in (world.config.verb_templates or {}).items():
        v = _normalize_verb(verb)
        if template_is_self_only(tmpl) and not template_requires_entity_target(tmpl):
            out.add(v)
    return frozenset(out)


def verb_requires_entity_target(world: WorldState, verb: str) -> bool:
    v = _normalize_verb(verb)
    if v in self_only_verbs_for_world(world):
        return False
    if v in entity_target_verbs_for_world(world):
        return True
    tmpl = (world.config.verb_templates or {}).get(v)
    if tmpl is not None:
        return template_requires_entity_target(tmpl)
    return False


def action_missing_required_target(action: SemanticAction, world: WorldState) -> bool:
    """True when a social/template verb was emitted without a required entity target."""
    verb = _normalize_verb(action.verb)
    if not verb_requires_entity_target(world, verb):
        return False
    target = action.target
    if target is None:
        return True
    if isinstance(target, str):
        return not target.strip()
    return False
