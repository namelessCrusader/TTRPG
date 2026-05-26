"""
World-sticky consequences — every real action leaves durable state.

Fluids (pour/drink) are one instance. This module defines what counts as
a *sticky* effect, attaches interaction traces, and ensures compiled
actions are not hollow unless explicitly marked narrative-only.

All consequence stages run through ``run_consequence_pipeline``.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

from .schemas import (
    Coord,
    EntityId,
    ObjectId,
    Transition,
    TransitionKind,
    TransitionProposal,
    ValidationResult,
    coord_key,
)

if TYPE_CHECKING:
    from .schemas import SemanticAction, WorldState

logger = logging.getLogger(__name__)

# Transitions that mutate canonical world (spatial, stats, graph, fluids).
# DIALOGUE_SPOKEN alone is narration / audit — not sufficient for physical verbs.
STICKY_TRANSITION_KINDS: frozenset[TransitionKind] = frozenset({
    TransitionKind.ENTITY_MOVED,
    TransitionKind.ENTITY_HEALTH_CHANGED,
    TransitionKind.ENTITY_EMOTIONAL_STATE_CHANGED,
    TransitionKind.ENTITY_ALERTNESS_CHANGED,
    TransitionKind.ENTITY_CONDITION_CHANGED,
    TransitionKind.ENTITY_PROPERTY_CHANGED,
    TransitionKind.ENTITY_STAT_CHANGED,
    TransitionKind.ENTITY_TAG_CHANGED,
    TransitionKind.ENTITY_MANA_CHANGED,
    TransitionKind.ITEM_TRANSFERRED,
    TransitionKind.ITEM_CREATED,
    TransitionKind.ITEM_DESTROYED,
    TransitionKind.ITEM_DURABILITY_CHANGED,
    TransitionKind.ITEM_EQUIPPED,
    TransitionKind.ITEM_UNEQUIPPED,
    TransitionKind.ITEM_STORED,
    TransitionKind.ITEM_RETRIEVED,
    TransitionKind.TILE_CHANGED,
    TransitionKind.TILE_MARKED,
    TransitionKind.STRUCTURE_CREATED,
    TransitionKind.STRUCTURE_MODIFIED,
    TransitionKind.ENVIRONMENT_STATE_CHANGED,
    TransitionKind.EDGE_CREATED,
    TransitionKind.EDGE_UPDATED,
    TransitionKind.EDGE_REMOVED,
    TransitionKind.CONTACT_INITIATED,
    TransitionKind.NOISE_EVENT,
    TransitionKind.CLAIM_MADE,
    TransitionKind.FLUID_CHANGED,
    TransitionKind.CONTAMINATION_CHANGED,
    TransitionKind.FIELD_CHANGED,
    TransitionKind.NEED_CHANGED,
    TransitionKind.REGION_TRANSIT,
    TransitionKind.ENTITY_TURNED,
    TransitionKind.SECRET_REVEALED,
    TransitionKind.FACTION_RESOURCE_CHANGED,
    TransitionKind.SOUND_PROPAGATED,
    TransitionKind.ITEM_SYNTHESIZED,
    TransitionKind.WORLD_FACT_REGISTERED,
})


class ConsequenceStage(str, Enum):
    """Where in the action lifecycle consequences are applied."""

    COMPILE = "compile"
    PROPOSALS = "proposals"
    RIPPLES = "ripples"


def is_sticky_transition(t: Transition) -> bool:
    return t.kind in STICKY_TRANSITION_KINDS


def has_world_stick(transitions: list[Transition]) -> bool:
    return any(is_sticky_transition(t) for t in transitions)


def _policy(world: "WorldState"):
    return world.config.consequence_policy


def narrative_only_verb(world: "WorldState", verb: str) -> bool:
    v = verb.lower().strip()
    policy = _policy(world)
    if v in policy.narrative_only_verbs:
        return True
    # Speech is socially sticky via graph + stimulus; not "hollow".
    if v in ("speak", "ask", "say", "tell", "reply", "whisper", "yell", "shout"):
        return True
    return False


def build_world_mark_transitions(
    action: "SemanticAction",
    world: "WorldState",
    *,
    outcome: str = "success",
) -> list[Transition]:
    """Audit trail on subjects so the world 'remembers' what happened here."""
    verb = str(action.verb).lower()
    actor_id = str(action.actor)
    tick = world.tick
    summary = f"{verb} ({outcome})"
    out: list[Transition] = []

    def _mark(subject_kind: str, subject_id: str) -> None:
        out.append(Transition(
            kind=TransitionKind.WORLD_MARK,
            payload={
                "subject_kind": subject_kind,
                "subject_id": subject_id,
                "verb": verb,
                "actor_id": actor_id,
                "tick": tick,
                "outcome": outcome,
                "summary": summary,
            },
        ))

    _mark("entity", actor_id)

    if action.target is not None:
        if isinstance(action.target, Coord):
            _mark("tile", coord_key(action.target))
        else:
            raw = str(action.target)
            grid = world.spatial
            if grid.entities.get(EntityId(raw)) is not None:
                _mark("entity", raw)
            elif grid.objects.get(ObjectId(raw)) is not None:
                _mark("object", raw)

    return out


def default_sticky_fallback(
    action: "SemanticAction",
    world: "WorldState",
) -> list[Transition]:
    """
    Minimal world stick when a verb would otherwise only leave dialogue.

    Entity targets → INTERACTED edge.  Tile/object/coord targets → tile mark
    + environment trace so propagation and fact seeding can react.
    """
    from .schemas import EdgeKind
    from .region_utils import entity_or_none, tile_key_for_entity

    policy = _policy(world)
    grid = world.spatial
    actor_id = str(action.actor)
    verb = str(action.verb).lower()
    actor = entity_or_none(world, action.actor)

    if action.target is None:
        return []

    if isinstance(action.target, Coord):
        pos = action.target
        mark = f"signs of {verb}"
        tile_key = (
            tile_key_for_entity(world, action.actor, pos)
            if actor is not None
            else coord_key(pos)
        )
        return [
            Transition(
                kind=TransitionKind.TILE_MARKED,
                payload={
                    "x": pos.x,
                    "y": pos.y,
                    "z": pos.z,
                    "mark": mark[:80],
                    "actor": actor_id,
                },
            ),
            Transition(
                kind=TransitionKind.ENVIRONMENT_STATE_CHANGED,
                payload={
                    "tile_key": tile_key,
                    "change": verb,
                    "actor": actor_id,
                },
            ),
        ]

    raw = str(action.target)
    target_ent = grid.entities.get(EntityId(raw))
    if target_ent is None:
        lower = raw.lower()
        for ent in grid.entities.values():
            if ent.name.lower() == lower:
                target_ent = ent
                break

    if target_ent is not None:
        return [
            Transition(
                kind=TransitionKind.EDGE_UPDATED,
                payload={
                    "source": actor_id,
                    "target": str(target_ent.entity_id),
                    "edge_kind": EdgeKind.INTERACTED.value,
                    "delta": policy.default_interacted_delta,
                    "meta": {"verb": verb, "cause": "consequence_fallback"},
                },
            ),
        ]

    target_obj = grid.objects.get(ObjectId(raw))
    if target_obj is not None:
        pos = target_obj.position
        return [
            Transition(
                kind=TransitionKind.TILE_MARKED,
                payload={
                    "x": pos.x,
                    "y": pos.y,
                    "z": pos.z,
                    "mark": f"{verb} on {target_obj.name}"[:80],
                    "actor": actor_id,
                },
            ),
        ]

    return []


def _stage_enforce_sticky(
    action: "SemanticAction",
    world: "WorldState",
    transitions: list[Transition],
) -> list[Transition]:
    verb = str(action.verb).lower()
    if _policy(world).require_sticky and not narrative_only_verb(world, verb):
        if not has_world_stick(transitions):
            transitions.extend(default_sticky_fallback(action, world))
    return transitions


def _stage_record_traces(
    action: "SemanticAction",
    world: "WorldState",
    transitions: list[Transition],
    *,
    contest_outcome: str,
) -> list[Transition]:
    if _policy(world).record_traces:
        transitions.extend(
            build_world_mark_transitions(action, world, outcome=contest_outcome)
        )
    return transitions


def _stage_promote_facts(
    action: "SemanticAction",
    world: "WorldState",
    transitions: list[Transition],
) -> list[Transition]:
    try:
        from .fact_registry import promote_compile_to_fact_transitions

        transitions.extend(
            promote_compile_to_fact_transitions(world, action, transitions)
        )
    except Exception as exc:
        logger.warning(
            "fact promotion failed for verb=%s tick=%d: %s",
            str(action.verb).lower(),
            world.tick,
            exc,
        )
    return transitions


def run_consequence_pipeline(
    action: "SemanticAction",
    world: "WorldState",
    transitions: list[Transition],
    *,
    stage: ConsequenceStage,
    contest_outcome: str = "success",
    proposals: list[TransitionProposal] | None = None,
) -> list[Transition]:
    """
    Single entry for all consequence stages.

    COMPILE   — sticky enforcement, world marks, fact promotion (post-compile)
    PROPOSALS — validate LM/adjudication TransitionProposals
    RIPPLES   — deterministic rule_ripples (deduped against existing transitions)
    """
    out = list(transitions)

    if stage == ConsequenceStage.COMPILE:
        out = _stage_enforce_sticky(action, world, out)
        out = _stage_record_traces(
            action, world, out, contest_outcome=contest_outcome,
        )
        out = _stage_promote_facts(action, world, out)
        return out

    if stage == ConsequenceStage.PROPOSALS and proposals:
        from .compiler import _validate_proposals

        out.extend(_validate_proposals(proposals, world, action))
        return out

    if stage == ConsequenceStage.RIPPLES and out:
        from .rule_ripples import filter_duplicate_ripples, propose_rule_ripples

        ripples = filter_duplicate_ripples(
            out,
            propose_rule_ripples(world, action, out),
        )
        out.extend(ripples)

    return out


def finalize_compiled_result(
    action: "SemanticAction",
    world: "WorldState",
    result: ValidationResult,
    *,
    contest_outcome: str = "success",
) -> ValidationResult:
    """Attach world marks and enforce sticky consequences on successful compiles."""
    if not result.valid:
        return result

    transitions = run_consequence_pipeline(
        action,
        world,
        list(result.concrete_transitions),
        stage=ConsequenceStage.COMPILE,
        contest_outcome=contest_outcome,
    )

    return ValidationResult(
        valid=True,
        concrete_transitions=transitions,
        plausibility=result.plausibility,
        rejection_reason=result.rejection_reason,
        rejection_detail=result.rejection_detail,
    )


def apply_post_action_consequences(
    world: "WorldState",
    action: "SemanticAction",
    event_transitions: list[Transition],
    *,
    proposals: list[TransitionProposal] | None = None,
    apply_ripples: bool = True,
) -> list[Transition]:
    """
    Apply LM consequence proposals and deterministic rule ripples.

    Returns new transitions to append to the event (caller applies to world).
    """
    new: list[Transition] = []

    if proposals:
        new.extend(
            run_consequence_pipeline(
                action,
                world,
                [],
                stage=ConsequenceStage.PROPOSALS,
                proposals=proposals,
            )
        )

    if apply_ripples and (event_transitions or new):
        combined = list(event_transitions) + new
        before = len(combined)
        with_ripples = run_consequence_pipeline(
            action,
            world,
            combined,
            stage=ConsequenceStage.RIPPLES,
        )
        new.extend(with_ripples[before:])

    return new


def append_trace_to_meta(
    meta: dict[str, Any],
    trace: dict[str, Any],
    *,
    limit: int,
) -> None:
    traces = list(meta.get("interaction_traces") or [])
    traces.append(trace)
    meta["interaction_traces"] = traces[-limit:]


def traces_for_projection(meta: dict[str, Any], *, max_items: int = 2) -> list[str]:
    lines: list[str] = []
    for t in (meta.get("interaction_traces") or [])[-max_items:]:
        if isinstance(t, dict):
            s = t.get("summary") or t.get("verb")
            if s:
                lines.append(str(s))
    return lines


__all__ = [
    "ConsequenceStage",
    "STICKY_TRANSITION_KINDS",
    "apply_post_action_consequences",
    "build_world_mark_transitions",
    "default_sticky_fallback",
    "finalize_compiled_result",
    "has_world_stick",
    "is_sticky_transition",
    "narrative_only_verb",
    "run_consequence_pipeline",
    "traces_for_projection",
]
