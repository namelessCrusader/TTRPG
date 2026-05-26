"""
Keyword-driven compiler handlers for open-ended verbs.

Physics-heavy open verbs (gold transfer, object damage, etc.) live here as
named handlers referenced from ``open_verbs.yaml``.  Matching is pack-driven
only — no duplicate keyword tables in this module.

Deterministic — no LM calls.  Invoked from ``open_verbs_compiler``.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Optional

from .schemas import (
    AlertnessLevel,
    EdgeKind,
    EmotionalState,
    EntityId,
    ObjectId,
    Transition,
    TransitionKind,
    ValidationResult,
    coord_key,
)

if TYPE_CHECKING:
    from .schemas import SemanticAction, WorldState

_GOLD_RE = re.compile(r"(\d+)\s*(?:gold|coins?)", re.I)

_ENV_VERBS = frozenset({
    "vandalize", "deface", "graffiti", "carve", "scratch", "scribble",
    "write", "paint", "stain", "spill", "smear", "tag", "mark",
})
_BRIBE_VERBS = frozenset({"bribe", "payoff", "grease", "tip", "buyoff"})
_SABOTAGE_VERBS = frozenset({"sabotage", "tamper", "sabotage", "rig", "boobytrap"})
_REPAIR_VERBS = frozenset({"repair", "fix", "mend", "patch", "restore"})
_FORGE_VERBS = frozenset({"forge", "counterfeit", "fake", "knockoff"})
_BLACKMAIL_VERBS = frozenset({"blackmail", "extort", "threaten_with_secret"})
_SEDUCE_VERBS = frozenset({"seduce", "charm", "woo", "entice", "tempt"})
_MANIPULATE_VERBS = frozenset({"manipulate", "coerce", "gaslight", "pressure"})
_PLANT_VERBS = frozenset({"plant", "hide", "stash", "bury", "conceal", "deposit"})
_DISGUISE_VERBS = frozenset({"disguise", "impersonate", "masquerade", "blend_in"})
_SEARCH_VERBS = frozenset({"search", "rummage", "scour", "scan", "frisk"})
_LISTEN_VERBS = frozenset({"listen", "eavesdrop", "overhear", "hearken"})
_PICKLOCK_VERBS = frozenset({"pick_lock", "picklock", "jimmy", "unlock"})
_INTIMIDATE_VERBS = frozenset({"intimidate", "threaten", "menace", "bully", "frighten"})
_DESTROY_VERBS = frozenset({
    "destroy", "smash", "demolish", "break", "shatter", "obliterate", "burn",
})
_HIRE_VERBS = frozenset({"hire", "employ", "recruit", "commission"})
_GIFT_VERBS = frozenset({"gift", "donate", "bestow", "offer"})


def _parse_gold(text: str, default: float = 10.0) -> float:
    m = _GOLD_RE.search(text or "")
    return max(1.0, float(m.group(1))) if m else default


def _resolve_target_entity(action: "SemanticAction", world: "WorldState"):
    from .compiler.verbs import _resolve_entity_target

    if not isinstance(action.target, str):
        return None
    return _resolve_entity_target(world.spatial, action.target)


def _resolve_target_object(action: "SemanticAction", world: "WorldState"):
    if not isinstance(action.target, str):
        return None
    return world.spatial.objects.get(ObjectId(action.target))


def _contest_and_record(
    action: "SemanticAction",
    world: "WorldState",
    *,
    extra: list[Transition],
    difficulty: int = 45,
) -> ValidationResult:
    from .compiler.verbs import _interaction_record, _resolve_contest, _validate_proposals

    target_ent = _resolve_target_entity(action, world)
    outcome = _resolve_contest(action, world, target_ent)
    transitions = list(extra)
    transitions.extend(_validate_proposals(action.proposed_effects, world, action))
    transitions.append(_interaction_record(action, outcome))
    return ValidationResult(
        valid=True,
        concrete_transitions=transitions,
        plausibility=0.85 if outcome == "success" else 0.45,
    )


def _compile_environmental(
    action: "SemanticAction",
    world: "WorldState",
    verb: str | None = None,
) -> Optional[ValidationResult]:
    from .compiler.verbs import _interaction_record, _validate_proposals
    from .schemas import Coord, coord_key

    actor = world.spatial.entities.get(action.actor)
    if actor is None:
        return None

    mark_pos = actor.position
    if isinstance(action.target, Coord):
        mark_pos = action.target
    elif isinstance(action.target, str):
        tgt = _resolve_target_entity(action, world)
        if tgt is not None:
            mark_pos = tgt.position

    if verb is None:
        verb = str(action.verb).lower().split(".")[-1]
    raw = (action.raw_input or "").strip()
    mark_text = raw[:60] if raw else f"signs of {verb}"
    actor_id = str(action.actor)

    transitions: list[Transition] = [
        Transition(
            kind=TransitionKind.TILE_MARKED,
            payload={
                "x": mark_pos.x,
                "y": mark_pos.y,
                "z": mark_pos.z,
                "mark": mark_text,
                "actor": actor_id,
            },
        ),
        Transition(
            kind=TransitionKind.ENVIRONMENT_STATE_CHANGED,
            payload={
                "tile_key": coord_key(mark_pos),
                "change": verb,
                "actor": actor_id,
            },
        ),
    ]
    transitions.extend(_validate_proposals(action.proposed_effects, world, action))
    transitions.append(_interaction_record(action, "success"))
    return ValidationResult(valid=True, concrete_transitions=transitions, plausibility=0.8)


def _compile_bribe(action: "SemanticAction", world: "WorldState") -> Optional[ValidationResult]:
    target = _resolve_target_entity(action, world)
    if target is None:
        return None
    actor = world.spatial.entities.get(action.actor)
    if actor is None:
        return None

    gold = _parse_gold(action.raw_input or "", default=15.0)
    actor_gold = float(actor.stats.get("gold", 0))
    if gold > actor_gold:
        from .schemas import RejectionReason

        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.PHYSICALLY_IMPOSSIBLE,
            rejection_detail=f"Need {gold:.0f} gold; you have {actor_gold:.0f}.",
        )

    actor_id = str(action.actor)
    target_id = str(target.entity_id)
    transitions = [
        Transition(
            kind=TransitionKind.ENTITY_STAT_CHANGED,
            payload={
                "entity_id": actor_id,
                "stat": "gold",
                "delta": -gold,
                "cause": "adjudicated_bribe",
                "actor": actor_id,
            },
        ),
        Transition(
            kind=TransitionKind.ENTITY_STAT_CHANGED,
            payload={
                "entity_id": target_id,
                "stat": "gold",
                "delta": gold,
                "cause": "adjudicated_bribe",
                "actor": actor_id,
            },
        ),
        Transition(
            kind=TransitionKind.EDGE_UPDATED,
            payload={
                "source": target_id,
                "target": actor_id,
                "edge_kind": EdgeKind.OWES_DEBT.value,
                "delta": 0.2,
                "meta": {"cause": "bribe", "amount": gold},
            },
        ),
        Transition(
            kind=TransitionKind.ENTITY_EMOTIONAL_STATE_CHANGED,
            payload={"entity_id": target_id, "to": EmotionalState.FRIENDLY.value},
        ),
    ]
    return _contest_and_record(action, world, extra=transitions, difficulty=40)


def _compile_sabotage(action: "SemanticAction", world: "WorldState") -> Optional[ValidationResult]:
    actor = world.spatial.entities.get(action.actor)
    if actor is None:
        return None

    target_obj = _resolve_target_object(action, world)
    target_ent = _resolve_target_entity(action, world)
    actor_id = str(action.actor)
    transitions: list[Transition] = []

    if target_obj is not None:
        transitions.append(Transition(
            kind=TransitionKind.STRUCTURE_MODIFIED,
            payload={
                "object_id": str(target_obj.object_id),
                "add_tags": ["sabotaged", "unreliable"],
                "cause": "sabotage",
            },
        ))
        transitions.append(Transition(
            kind=TransitionKind.ITEM_DURABILITY_CHANGED,
            payload={
                "object_id": str(target_obj.object_id),
                "delta": -35,
                "cause": "sabotage",
            },
        ))
    elif target_ent is not None:
        transitions.append(Transition(
            kind=TransitionKind.ENTITY_ALERTNESS_CHANGED,
            payload={
                "entity_id": str(target_ent.entity_id),
                "to": AlertnessLevel.HIGH.value,
                "cause": "sabotage_nearby",
                "actor": actor_id,
            },
        ))

    transitions.append(Transition(
        kind=TransitionKind.TILE_MARKED,
        payload={
            "x": actor.position.x,
            "y": actor.position.y,
            "z": actor.position.z,
            "mark": "signs of tampering",
            "actor": actor_id,
        },
    ))
    return _contest_and_record(action, world, extra=transitions, difficulty=55)


def _compile_repair(action: "SemanticAction", world: "WorldState") -> Optional[ValidationResult]:
    target_obj = _resolve_target_object(action, world)
    if target_obj is None and isinstance(action.target, str):
        from .compiler.verbs import _find_item_by_name

        actor = world.spatial.entities.get(action.actor)
        if actor:
            target_obj = _find_item_by_name(action.target, actor, world.spatial)
    if target_obj is None:
        return None

    transitions = [
        Transition(
            kind=TransitionKind.ITEM_DURABILITY_CHANGED,
            payload={
                "object_id": str(target_obj.object_id),
                "delta": 25,
                "cause": "repair",
            },
        ),
        Transition(
            kind=TransitionKind.STRUCTURE_MODIFIED,
            payload={
                "object_id": str(target_obj.object_id),
                "remove_tags": ["broken", "sabotaged"],
                "cause": "repair",
            },
        ),
    ]
    return _contest_and_record(action, world, extra=transitions, difficulty=35)


def _compile_forge(action: "SemanticAction", world: "WorldState") -> Optional[ValidationResult]:
    import uuid

    actor = world.spatial.entities.get(action.actor)
    if actor is None:
        return None

    raw = (action.raw_input or action.intent.rationale if action.intent else "") or "item"
    name_match = re.search(
        r"(?:forge|fake|counterfeit|knockoff)\s+(?:a\s+|an\s+|the\s+)?(.+?)(?:\.|$)",
        raw,
        re.I,
    )
    item_name = (name_match.group(1).strip() if name_match else "forged item")[:40]
    oid = f"obj_{uuid.uuid4().hex[:8]}"
    actor_id = str(action.actor)

    transitions = [
        Transition(
            kind=TransitionKind.ITEM_CREATED,
            payload={
                "object_id": oid,
                "name": item_name,
                "tags": ["forged", "portable"],
                "to_entity": actor_id,
                "attributes": {"value": 5, "damage": 1},
                "durability": 40,
                "max_durability": 100,
            },
        ),
        Transition(
            kind=TransitionKind.ENTITY_STAT_CHANGED,
            payload={
                "entity_id": actor_id,
                "stat": "stamina",
                "delta": -8.0,
                "cause": "forging_effort",
            },
        ),
    ]
    return _contest_and_record(action, world, extra=transitions, difficulty=50)


def _compile_blackmail(action: "SemanticAction", world: "WorldState") -> Optional[ValidationResult]:
    target = _resolve_target_entity(action, world)
    if target is None:
        return None
    actor_id = str(action.actor)
    target_id = str(target.entity_id)
    claim = (action.raw_input or f"{actor_id} blackmailed {target.name}")[:120]

    transitions = [
        Transition(
            kind=TransitionKind.CLAIM_MADE,
            payload={
                "actor": actor_id,
                "text": claim,
                "claim_type": "blackmail",
                "severity": 0.7,
                "targets": [target_id],
            },
        ),
        Transition(
            kind=TransitionKind.ENTITY_EMOTIONAL_STATE_CHANGED,
            payload={"entity_id": target_id, "to": EmotionalState.FEARFUL.value},
        ),
        Transition(
            kind=TransitionKind.EDGE_UPDATED,
            payload={
                "source": target_id,
                "target": actor_id,
                "edge_kind": EdgeKind.FEARS.value,
                "delta": 0.25,
            },
        ),
    ]
    return _contest_and_record(action, world, extra=transitions, difficulty=48)


def _compile_social_influence(
    action: "SemanticAction",
    world: "WorldState",
    *,
    intimate: bool = False,
) -> Optional[ValidationResult]:
    target = _resolve_target_entity(action, world)
    if target is None:
        return None
    actor_id = str(action.actor)
    target_id = str(target.entity_id)

    transitions = [
        Transition(
            kind=TransitionKind.ENTITY_EMOTIONAL_STATE_CHANGED,
            payload={"entity_id": target_id, "to": EmotionalState.HAPPY.value},
        ),
    ]
    if intimate:
        transitions.append(Transition(
            kind=TransitionKind.EDGE_UPDATED,
            payload={
                "source": target_id,
                "target": actor_id,
                "edge_kind": EdgeKind.INTIMATE.value,
                "delta": 0.15,
            },
        ))
    else:
        transitions.append(Transition(
            kind=TransitionKind.EDGE_UPDATED,
            payload={
                "source": target_id,
                "target": actor_id,
                "edge_kind": EdgeKind.DISTRUSTS.value,
                "delta": 0.1,
            },
        ))
    return _contest_and_record(
        action, world, extra=transitions,
        difficulty=38 if intimate else 52,
    )


def _compile_plant(action: "SemanticAction", world: "WorldState") -> Optional[ValidationResult]:
    actor = world.spatial.entities.get(action.actor)
    if actor is None:
        return None

    raw = (action.raw_input or "").strip()
    mark = raw[:60] if raw else "something hidden here"
    actor_id = str(action.actor)
    pos = actor.position

    transitions = [
        Transition(
            kind=TransitionKind.TILE_MARKED,
            payload={
                "x": pos.x,
                "y": pos.y,
                "z": pos.z,
                "mark": f"concealed: {mark}"[:80],
                "actor": actor_id,
            },
        ),
        Transition(
            kind=TransitionKind.ENVIRONMENT_STATE_CHANGED,
            payload={
                "x": pos.x,
                "y": pos.y,
                "key": "hidden",
                "value": mark[:40],
            },
        ),
    ]
    return _contest_and_record(action, world, extra=transitions, difficulty=42)


def _compile_disguise(action: "SemanticAction", world: "WorldState") -> Optional[ValidationResult]:
    actor = world.spatial.entities.get(action.actor)
    if actor is None:
        return None

    raw = (action.raw_input or "").strip()
    identity_match = re.search(
        r"(?:as|like)\s+(?:a\s+|an\s+|the\s+)?(.+?)(?:\.|$)",
        raw,
        re.I,
    )
    identity = (identity_match.group(1).strip() if identity_match else "someone else")[:40]
    actor_id = str(action.actor)

    transitions = [
        Transition(
            kind=TransitionKind.ENTITY_PROPERTY_CHANGED,
            payload={
                "entity_id": actor_id,
                "prop": "disguise_as",
                "new_value": identity,
            },
        ),
        Transition(
            kind=TransitionKind.CLAIM_MADE,
            payload={
                "actor": actor_id,
                "text": f"{actor.name} appears as {identity}.",
                "claim_type": "disguise",
                "severity": 0.4,
                "targets": [],
            },
        ),
    ]
    return _contest_and_record(action, world, extra=transitions, difficulty=48)


def _compile_search(action: "SemanticAction", world: "WorldState") -> Optional[ValidationResult]:
    from .compiler.verbs import _interaction_record, _validate_proposals
    from .fact_registry import survey_area
    from .schemas import Coord

    actor = world.spatial.entities.get(action.actor)
    if actor is None:
        return None

    actor_id = str(action.actor)
    pos = actor.position
    if isinstance(action.target, Coord):
        pos = action.target

    survey = survey_area(world, actor)
    findings = survey.get("tiles_of_interest") or []
    summary = (
        f"searched and found {len(findings)} point(s) of interest"
        if findings
        else "searched but found nothing unusual"
    )

    transitions: list[Transition] = [
        Transition(
            kind=TransitionKind.TILE_MARKED,
            payload={
                "x": pos.x,
                "y": pos.y,
                "z": pos.z,
                "mark": summary[:80],
                "actor": actor_id,
            },
        ),
        Transition(
            kind=TransitionKind.CLAIM_MADE,
            payload={
                "actor": actor_id,
                "text": summary[:120],
                "claim_type": "search",
                "severity": 0.3,
                "targets": [],
            },
        ),
        Transition(
            kind=TransitionKind.ENTITY_STAT_CHANGED,
            payload={
                "entity_id": actor_id,
                "stat": "stamina",
                "delta": -2.0,
                "cause": "search_effort",
            },
        ),
    ]
    transitions.extend(_validate_proposals(action.proposed_effects, world, action))
    transitions.append(_interaction_record(action, "success"))
    return ValidationResult(valid=True, concrete_transitions=transitions, plausibility=0.75)


def _compile_listen(action: "SemanticAction", world: "WorldState") -> Optional[ValidationResult]:
    actor = world.spatial.entities.get(action.actor)
    if actor is None:
        return None

    actor_id = str(action.actor)
    transitions = [
        Transition(
            kind=TransitionKind.ENTITY_ALERTNESS_CHANGED,
            payload={
                "entity_id": actor_id,
                "to": AlertnessLevel.HIGH.value,
                "cause": "listening",
            },
        ),
        Transition(
            kind=TransitionKind.NOISE_EVENT,
            payload={
                "source_id": actor_id,
                "noise_level": 1,
                "description": "someone listening intently",
            },
        ),
    ]
    return _contest_and_record(action, world, extra=transitions, difficulty=30)


def _compile_pick_lock(action: "SemanticAction", world: "WorldState") -> Optional[ValidationResult]:
    target_obj = _resolve_target_object(action, world)
    actor = world.spatial.entities.get(action.actor)
    if actor is None:
        return None

    actor_id = str(action.actor)
    pos = actor.position
    if target_obj is not None:
        pos = target_obj.position

    transitions: list[Transition] = []
    if target_obj is not None:
        transitions.append(Transition(
            kind=TransitionKind.ITEM_DURABILITY_CHANGED,
            payload={
                "object_id": str(target_obj.object_id),
                "delta": -12,
                "cause": "lock_picking",
            },
        ))
    transitions.append(Transition(
        kind=TransitionKind.ENVIRONMENT_STATE_CHANGED,
        payload={
            "x": pos.x,
            "y": pos.y,
            "key": "lock_tampered",
            "value": actor_id,
        },
    ))
    transitions.append(Transition(
        kind=TransitionKind.NOISE_EVENT,
        payload={
            "source_id": actor_id,
            "noise_level": 3,
            "description": "metallic scraping at a lock",
        },
    ))
    return _contest_and_record(action, world, extra=transitions, difficulty=55)


def _compile_intimidate(action: "SemanticAction", world: "WorldState") -> Optional[ValidationResult]:
    target = _resolve_target_entity(action, world)
    if target is None:
        return None

    actor_id = str(action.actor)
    target_id = str(target.entity_id)
    claim = (action.raw_input or f"{actor_id} intimidated {target.name}")[:120]

    transitions = [
        Transition(
            kind=TransitionKind.CLAIM_MADE,
            payload={
                "actor": actor_id,
                "text": claim,
                "claim_type": "intimidation",
                "severity": 0.65,
                "targets": [target_id],
            },
        ),
        Transition(
            kind=TransitionKind.ENTITY_EMOTIONAL_STATE_CHANGED,
            payload={"entity_id": target_id, "to": EmotionalState.FEARFUL.value},
        ),
        Transition(
            kind=TransitionKind.ENTITY_ALERTNESS_CHANGED,
            payload={
                "entity_id": target_id,
                "to": AlertnessLevel.HIGH.value,
                "cause": "intimidated",
            },
        ),
        Transition(
            kind=TransitionKind.EDGE_UPDATED,
            payload={
                "source": target_id,
                "target": actor_id,
                "edge_kind": EdgeKind.FEARS.value,
                "delta": 0.2,
            },
        ),
    ]
    return _contest_and_record(action, world, extra=transitions, difficulty=44)


def _compile_destroy(action: "SemanticAction", world: "WorldState") -> Optional[ValidationResult]:
    target_obj = _resolve_target_object(action, world)
    actor = world.spatial.entities.get(action.actor)
    if actor is None:
        return None

    actor_id = str(action.actor)
    pos = actor.position
    transitions: list[Transition] = []

    if target_obj is not None:
        pos = target_obj.position
        transitions.append(Transition(
            kind=TransitionKind.ITEM_DESTROYED,
            payload={
                "object_id": str(target_obj.object_id),
                "cause": "destroyed",
                "actor": actor_id,
            },
        ))
        transitions.append(Transition(
            kind=TransitionKind.NOISE_EVENT,
            payload={
                "source_id": actor_id,
                "noise_level": 55,
                "description": f"destruction of {target_obj.name}",
            },
        ))
    else:
        raw = (action.raw_input or "").strip()
        mark = raw[:60] if raw else "signs of destruction"
        transitions.append(Transition(
            kind=TransitionKind.TILE_MARKED,
            payload={
                "x": pos.x,
                "y": pos.y,
                "z": pos.z,
                "mark": mark[:80],
                "actor": actor_id,
            },
        ))
        transitions.append(Transition(
            kind=TransitionKind.ENVIRONMENT_STATE_CHANGED,
            payload={
                "tile_key": coord_key(pos),
                "change": "destroyed",
                "actor": actor_id,
            },
        ))
        transitions.append(Transition(
            kind=TransitionKind.NOISE_EVENT,
            payload={
                "source_id": actor_id,
                "noise_level": 45,
                "description": "loud destruction nearby",
            },
        ))

    return _contest_and_record(action, world, extra=transitions, difficulty=50)


def _compile_hire(action: "SemanticAction", world: "WorldState") -> Optional[ValidationResult]:
    target = _resolve_target_entity(action, world)
    actor = world.spatial.entities.get(action.actor)
    if target is None or actor is None:
        return None

    gold = _parse_gold(action.raw_input or "", default=20.0)
    actor_gold = float(actor.stats.get("gold", 0))
    if gold > actor_gold:
        from .schemas import RejectionReason

        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.PHYSICALLY_IMPOSSIBLE,
            rejection_detail=f"Need {gold:.0f} gold to hire; you have {actor_gold:.0f}.",
        )

    actor_id = str(action.actor)
    target_id = str(target.entity_id)
    raw = (action.raw_input or f"hire {target.name}")[:120]

    transitions = [
        Transition(
            kind=TransitionKind.ENTITY_STAT_CHANGED,
            payload={
                "entity_id": actor_id,
                "stat": "gold",
                "delta": -gold,
                "cause": "adjudicated_hire",
                "actor": actor_id,
            },
        ),
        Transition(
            kind=TransitionKind.ENTITY_STAT_CHANGED,
            payload={
                "entity_id": target_id,
                "stat": "gold",
                "delta": gold,
                "cause": "adjudicated_hire",
                "actor": actor_id,
            },
        ),
        Transition(
            kind=TransitionKind.EDGE_UPDATED,
            payload={
                "source": target_id,
                "target": actor_id,
                "edge_kind": EdgeKind.OWES_DEBT.value,
                "delta": 0.25,
                "meta": {"cause": "hire", "amount": gold},
            },
        ),
        Transition(
            kind=TransitionKind.CLAIM_MADE,
            payload={
                "actor": actor_id,
                "text": raw,
                "claim_type": "hire",
                "severity": 0.5,
                "targets": [target_id],
            },
        ),
    ]
    return _contest_and_record(action, world, extra=transitions, difficulty=42)


def _compile_gift(action: "SemanticAction", world: "WorldState") -> Optional[ValidationResult]:
    target = _resolve_target_entity(action, world)
    actor = world.spatial.entities.get(action.actor)
    if target is None or actor is None:
        return None

    gold = _parse_gold(action.raw_input or "", default=5.0)
    actor_gold = float(actor.stats.get("gold", 0))
    if gold > actor_gold:
        from .schemas import RejectionReason

        return ValidationResult(
            valid=False,
            rejection_reason=RejectionReason.PHYSICALLY_IMPOSSIBLE,
            rejection_detail=f"Need {gold:.0f} gold to gift; you have {actor_gold:.0f}.",
        )

    actor_id = str(action.actor)
    target_id = str(target.entity_id)
    transitions = [
        Transition(
            kind=TransitionKind.ENTITY_STAT_CHANGED,
            payload={
                "entity_id": actor_id,
                "stat": "gold",
                "delta": -gold,
                "cause": "gift",
                "actor": actor_id,
            },
        ),
        Transition(
            kind=TransitionKind.ENTITY_STAT_CHANGED,
            payload={
                "entity_id": target_id,
                "stat": "gold",
                "delta": gold,
                "cause": "gift",
                "actor": actor_id,
            },
        ),
        Transition(
            kind=TransitionKind.ENTITY_EMOTIONAL_STATE_CHANGED,
            payload={"entity_id": target_id, "to": EmotionalState.FRIENDLY.value},
        ),
        Transition(
            kind=TransitionKind.EDGE_UPDATED,
            payload={
                "source": target_id,
                "target": actor_id,
                "edge_kind": EdgeKind.RESPECTS.value,
                "delta": 0.15,
                "meta": {"cause": "gift", "amount": gold},
            },
        ),
    ]
    return _contest_and_record(action, world, extra=transitions, difficulty=35)


__all__ = [
    "_compile_bribe",
    "_compile_environmental",
    "_resolve_target_entity",
    "_resolve_target_object",
]
