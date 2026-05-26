"""
World adjudication — DM layer for Zone 3 and rejected actions.

When the compiler cannot realize player intent, adjudication:
  1. Attempts verb synthesis and re-compile (interaction rescue).
  2. Registers durable ``WorldFact`` records.
  3. Validates ``TransitionProposal`` consequences.
  4. Queues ``ScheduledEffect`` entries for future ticks.

All mutations still pass through the deterministic compiler validator.
This module never writes canonical state directly except via registered
facts/effects lists on ``WorldState``.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Optional

from .grounding import _intent_to_verb
from .schemas import (
    ActionZone,
    AdjudicationResult,
    AlertnessLevel,
    EntityId,
    GroundingResult,
    ScheduledEffect,
    ScheduledEffectKind,
    SemanticAction,
    Transition,
    TransitionKind,
    TransitionProposal,
    ValidationResult,
    WorldFact,
    WorldFactScope,
    WorldState,
    coord_key,
)

if TYPE_CHECKING:
    from .lm_adapter import LMAdapter
    from .schemas import SemanticProjection

logger = logging.getLogger(__name__)

_MAX_WORLD_FACTS = 400  # default; overridden by ConsequencePolicy.max_world_facts
_MAX_SCHEDULED = 96

_ENVIRONMENT_KEYWORDS = frozenset({
    "carve", "scratch", "write", "mark", "spill", "break", "draw",
    "paint", "graffiti", "chip", "dent", "smash", "shatter", "stain",
})

_DELAY_KEYWORDS = frozenset({
    "later", "eventually", "soon", "tomorrow", "wait for", "in a while",
})


def _ensure_sticky_fallback(
    world: WorldState,
    action: SemanticAction,
    adj: AdjudicationResult,
    text: str,
    actor_id: str,
    tick: int,
) -> None:
    """
    Guarantee every adjudication leaves durable, sticky state.

    When templates and partial matchers did not emit transitions or delayed
    ripples, add a minimal tile trace, effort cost, and narration schedule.
    """
    from .region_utils import entity_or_none, tile_key_for_entity

    actor = entity_or_none(world, action.actor)

    if not adj.facts:
        claim = text[:160] if text else f"{action.verb} (unresolved)"
        fact = WorldFact(
            claim=claim,
            scope=WorldFactScope.WORLD,
            established_tick=tick,
            established_by=actor_id,
            source_intent=text[:200] or None,
            tags=["adjudicated"],
        )
        if actor is not None:
            fact.scope = WorldFactScope.TILE
            fact.subject_id = tile_key_for_entity(
                world, action.actor, actor.position,
            )
        adj.facts.append(fact)
    elif actor is not None and adj.facts[0].scope == WorldFactScope.WORLD:
        if not any(t in adj.facts[0].tags for t in ("travel", "faction")):
            adj.facts[0].scope = WorldFactScope.TILE
            adj.facts[0].subject_id = tile_key_for_entity(
                world, action.actor, actor.position,
            )

    if not adj.transition_proposals and actor is not None:
        mark = re.sub(r"\s+", " ", text)[:60] or "evidence of action"
        adj.transition_proposals.extend([
            TransitionProposal(
                kind=TransitionKind.ENTITY_STAT_CHANGED.value,
                payload={
                    "entity_id": actor_id,
                    "stat": "stamina",
                    "delta": -3.0,
                    "cause": "adjudicated_effort",
                    "actor": actor_id,
                },
            ),
            TransitionProposal(
                kind=TransitionKind.TILE_MARKED.value,
                payload={
                    "x": actor.position.x,
                    "y": actor.position.y,
                    "mark": mark,
                },
            ),
        ])

    if not adj.scheduled_effects:
        actor_name = actor.name if actor is not None else "Someone"
        snippet = (text[:80] + "…") if len(text) > 80 else text
        narration = (
            f"Word spreads about what {actor_name} did"
            + (f": {snippet}" if snippet else ".")
        )
        adj.scheduled_effects.append(
            ScheduledEffect(
                fire_tick=tick + 2,
                created_tick=tick,
                source_intent=text[:120] or None,
                source_actor=actor_id,
                kind=ScheduledEffectKind.NARRATION,
                narration=narration[:240],
                rationale="default adjudication ripple",
            )
        )

    if actor is not None and not any(
        e.kind == ScheduledEffectKind.TRANSITIONS for e in adj.scheduled_effects
    ):
        from .region_utils import witness_entity_ids

        witness_ids = [
            str(wid)
            for wid in witness_entity_ids(world, action.actor)
            if str(wid) != actor_id
        ][:4]
        if witness_ids:
            adj.scheduled_effects.append(
                ScheduledEffect(
                    fire_tick=tick + 1,
                    created_tick=tick,
                    source_intent=text[:120] or None,
                    source_actor=actor_id,
                    kind=ScheduledEffectKind.TRANSITIONS,
                    transitions=[
                        TransitionProposal(
                            kind=TransitionKind.ENTITY_ALERTNESS_CHANGED.value,
                            payload={
                                "entity_id": wid,
                                "to": AlertnessLevel.HIGH.value,
                                "cause": "witnessed_adjudicated_action",
                            },
                        )
                        for wid in witness_ids
                    ],
                    rationale="witnesses react to unusual action",
                )
            )


def needs_adjudication(
    grounding: GroundingResult,
    result: ValidationResult,
) -> bool:
    """Return True when the DM layer should attempt fail-forward resolution."""
    if grounding.zone == ActionZone.ORPHANED:
        return True
    if not result.valid:
        return True
    return False


def register_world_fact(world: WorldState, fact: WorldFact) -> None:
    """Direct registration (tests, bootstrap). Game path uses transitions."""
    _append_world_fact(world, fact)


def world_fact_transition(fact: WorldFact) -> Transition:
    """Build a replay-safe transition that registers a durable world fact."""
    return Transition(
        kind=TransitionKind.WORLD_FACT_REGISTERED,
        payload={"fact": fact.model_dump(mode="json")},
    )


def scheduled_effect_transition(effect: ScheduledEffect) -> Transition:
    """Build a replay-safe transition that queues a delayed effect."""
    return Transition(
        kind=TransitionKind.SCHEDULED_EFFECT_QUEUED,
        payload={"effect": effect.model_dump(mode="json")},
    )


def _max_world_facts(world: WorldState) -> int:
    return int(world.config.consequence_policy.max_world_facts)


def _append_world_fact(world: WorldState, fact: WorldFact) -> None:
    from .memory_retrieval import fact_importance

    cap = _max_world_facts(world)
    world.world_facts.append(fact)
    if len(world.world_facts) > cap:
        _trim_world_facts_by_importance(world, fact_importance, cap)


def _trim_world_facts_by_importance(
    world: WorldState,
    score_fn,
    cap: int | None = None,
) -> None:
    """Evict lowest-importance facts when over cap (not FIFO)."""
    limit = cap if cap is not None else _max_world_facts(world)
    if len(world.world_facts) <= limit:
        return
    ranked = sorted(
        enumerate(world.world_facts),
        key=lambda pair: (score_fn(pair[1], world), pair[1].established_tick),
    )
    drop_count = len(world.world_facts) - limit
    drop_indices = {idx for idx, _ in ranked[:drop_count]}
    world.world_facts = [
        f for i, f in enumerate(world.world_facts) if i not in drop_indices
    ]


def _queue_scheduled_effect(world: WorldState, effect: ScheduledEffect) -> None:
    world.scheduled_effects.append(effect)
    if len(world.scheduled_effects) > _MAX_SCHEDULED:
        world.scheduled_effects = world.scheduled_effects[-_MAX_SCHEDULED:]


def schedule_effect(world: WorldState, effect: ScheduledEffect) -> None:
    """Direct queue (bootstrap/tests). Game paths should emit transitions."""
    _queue_scheduled_effect(world, effect)


def facts_for_projection(
    world: WorldState,
    focal_entity_id: EntityId,
    *,
    max_items: int = 6,
) -> list[str]:
    """
    Return established fact claims relevant to the focal entity's context,
    ranked by importance rather than insertion order.
    """
    from .memory_retrieval import ranked_facts_for_projection

    return ranked_facts_for_projection(
        world, focal_entity_id, max_items=max_items,
    )


def rule_based_adjudicate(
    world: WorldState,
    action: SemanticAction,
    intent: str,
    grounding: GroundingResult,
    result: ValidationResult,
) -> AdjudicationResult:
    """
    Deterministic adjudication for tests and MockLMAdapter fallback.
    """
    from .grounding import classify
    from .compiler import compile_action

    text = (intent or action.raw_input or "").strip()
    lower = text.lower()
    actor_id = str(action.actor)
    tick = world.tick
    adj = AdjudicationResult()

    # --- Template matchers (trade, reputation, faction, craft) ---
    from .adjudication_templates import try_adjudication_templates

    templated = try_adjudication_templates(
        world, action, intent, grounding, result,
    )
    if templated is not None:
        return templated

    # --- Pack adjudication index (Overhaul D) ---
    from .adjudication_resolver import resolve_from_index

    indexed = resolve_from_index(world, action, intent, grounding, result)
    if indexed is not None:
        _ensure_sticky_fallback(world, action, indexed, text, actor_id, tick)
        return indexed

    # --- Mechanical heuristics (legacy fallback) ---
    from .adjudication_heuristics import try_adjudication_heuristics

    heuristic = try_adjudication_heuristics(
        world, action, intent, grounding, result,
    )
    if heuristic is not None:
        _ensure_sticky_fallback(world, action, heuristic, text, actor_id, tick)
        return heuristic

    # --- Verb rescue for orphaned WAIT-style actions ---
    if grounding.zone == ActionZone.ORPHANED and action.target is not None:
        if isinstance(action.target, str):
            tid = EntityId(str(action.target))
            if tid in world.spatial.entities:
                synthesized = _intent_to_verb(text)
                if synthesized and synthesized not in ("wait", "interact"):
                    trial = action.model_copy(update={"verb": synthesized})
                    trial_ground = classify(trial, world, text)
                    if trial_ground.zone != ActionZone.ORPHANED:
                        trial_result = compile_action(trial, world)
                        if trial_result.valid:
                            adj.synthesized_verb = synthesized
                            adj.ruling_text = (
                                f"You {synthesized.replace('_', ' ')}."
                            )
                            return adj

    # --- Durable fact ---
    claim = text[:160] if text else f"{action.verb} (unresolved)"
    fact = WorldFact(
        claim=claim,
        scope=WorldFactScope.WORLD,
        established_tick=tick,
        established_by=actor_id,
        source_intent=text[:200] or None,
        tags=["adjudicated"],
    )

    from .region_utils import entity_or_none, tile_key_for_entity

    actor = entity_or_none(world, action.actor)
    if actor is not None:
        if any(kw in lower for kw in _ENVIRONMENT_KEYWORDS):
            mark = re.sub(r"\s+", " ", text)[:60]
            fact.scope = WorldFactScope.TILE
            fact.subject_id = tile_key_for_entity(
                world, action.actor, actor.position,
            )
            fact.tags.append("environment")
            adj.transition_proposals.append(
                TransitionProposal(
                    kind=TransitionKind.TILE_MARKED.value,
                    payload={
                        "x": actor.position.x,
                        "y": actor.position.y,
                        "mark": mark or "altered",
                    },
                )
            )
            adj.ruling_text = f"The mark remains: {mark or 'something changed here'}."
            adj.scheduled_effects.append(
                ScheduledEffect(
                    fire_tick=tick + 2,
                    created_tick=tick,
                    source_intent=text[:120] or None,
                    source_actor=actor_id,
                    kind=ScheduledEffectKind.NARRATION,
                    narration=(
                        f"Someone notices the change nearby: {mark or 'something altered'}."
                    ),
                    rationale="delayed reaction to environmental adjudication",
                )
            )
        elif isinstance(action.target, str):
            tid = str(action.target)
            if EntityId(tid) in world.spatial.entities:
                fact.scope = WorldFactScope.ENTITY
                fact.subject_id = tid
                adj.transition_proposals.append(
                    TransitionProposal(
                        kind=TransitionKind.EDGE_UPDATED.value,
                        payload={
                            "source": actor_id,
                            "target": tid,
                            "edge_kind": "interacted",
                            "delta": 0.08,
                            "meta": {"verb": str(action.verb), "cause": "adjudication"},
                        },
                    )
                )
                target_name = world.spatial.entities[EntityId(tid)].name
                adj.ruling_text = (
                    f"The interaction with {target_name} leaves a trace "
                    "the world will remember."
                )

    if not adj.ruling_text:
        if grounding.zone == ActionZone.ORPHANED:
            # Build a grounded ruling from nearby world state instead of a generic fallback
            from .adjudication_heuristics import _nearby_scene_features
            features = (
                _nearby_scene_features(world, actor.position.x, actor.position.y)
                if actor is not None
                else []
            )
            _partial_templates = [
                "The world bends to your will — partway. Something gives, but not everything you hoped.",
                "Your action leaves a mark, if not the one you intended.",
                "It almost works. The world remembers the attempt.",
            ]
            import hashlib as _h2
            _idx = int.from_bytes(
                _h2.sha256(f"{actor_id}_{tick}_{text[:30]}".encode()).digest()[:2], "big"
            ) % len(_partial_templates)
            adj.ruling_text = _partial_templates[_idx]
            if features:
                adj.ruling_text += (
                    f" (You notice: {'; '.join(features[:2])} — "
                    f"these might help you push further.)"
                )
            elif action.target is not None:
                adj.ruling_text += (
                    " Perhaps a different approach — persuasion, a tool, or better timing — "
                    "would let you finish what you started."
                )
        else:
            detail = result.rejection_detail or "the action could not compile"
            adj.ruling_text = (
                f"The DM rules it anyway, with reduced mechanical effect: {detail}."
            )

    adj.facts.append(fact)

    # --- Delayed consequence hint ---
    if any(kw in lower for kw in _DELAY_KEYWORDS):
        adj.scheduled_effects.append(
            ScheduledEffect(
                fire_tick=tick + 3,
                created_tick=tick,
                source_intent=text[:120] or None,
                source_actor=actor_id,
                kind=ScheduledEffectKind.NARRATION,
                narration=adj.ruling_text,
                rationale="delayed follow-through from adjudicated action",
            )
        )

    _ensure_sticky_fallback(world, action, adj, text, actor_id, tick)
    return adj


def invoke_adjudication(
    adapter: "LMAdapter",
    world: WorldState,
    action: SemanticAction,
    intent: str,
    projection: "SemanticProjection",
    grounding: GroundingResult,
    result: ValidationResult,
) -> AdjudicationResult:
    """Call LM adjudication with rule-based fallback."""
    from .lm_adapter import MockLMAdapter

    if isinstance(adapter, MockLMAdapter):
        return rule_based_adjudicate(world, action, intent, grounding, result)

    from .adjudication_cache import cache_key, get_cached, put_cached

    key = cache_key(world, action, intent, grounding, result)
    cached = get_cached(world, key)
    if cached is not None:
        return cached

    from .adjudication_resolver import resolve_from_index

    indexed = resolve_from_index(world, action, intent, grounding, result)
    if indexed is not None:
        text = (intent or action.raw_input or "").strip()
        actor_id = str(action.actor)
        _ensure_sticky_fallback(
            world, action, indexed, text, actor_id, world.tick,
        )
        return indexed

    from .adjudication_heuristics import try_adjudication_heuristics

    heuristic = try_adjudication_heuristics(
        world, action, intent, grounding, result,
    )
    if heuristic is not None:
        text = (intent or action.raw_input or "").strip()
        actor_id = str(action.actor)
        _ensure_sticky_fallback(
            world, action, heuristic, text, actor_id, world.tick,
        )
        return heuristic

    try:
        adj = adapter.adjudicate(
            action, projection, intent, grounding, result,
        )
    except Exception as exc:
        logger.warning("adjudicate() raised: %s — rule fallback", exc)
        return rule_based_adjudicate(world, action, intent, grounding, result)

    if (
        not adj.ruling_text
        and not adj.facts
        and not adj.transition_proposals
        and not adj.scheduled_effects
        and not adj.synthesized_verb
    ):
        return rule_based_adjudicate(world, action, intent, grounding, result)

    # Capture this LM adjudication as a candidate pack rule for later
    # human review.  No-op unless the pack opts in via
    # ``world.config.extra["learn_from_adjudication"] = true``.
    try:
        from .rule_learner import record_adjudication

        record_adjudication(world, action, intent, adj)
    except Exception as exc:
        logger.debug("rule_learner: skipped (%s)", exc)

    put_cached(world, key, adj)
    return adj


def materialize_adjudication(
    world: WorldState,
    action: SemanticAction,
    adj: AdjudicationResult,
    projection: "SemanticProjection",
) -> list[Transition]:
    """
    Register facts/effects and validate transition proposals from adjudication.
    """
    from .compiler import _validate_proposals

    tick = world.tick
    actor_id = str(action.actor)

    transitions: list[Transition] = []
    for raw_fact in adj.facts:
        fact = raw_fact.model_copy(
            update={
                "established_tick": raw_fact.established_tick or tick,
                "established_by": raw_fact.established_by or actor_id,
            }
        )
        transitions.append(world_fact_transition(fact))

    for raw_effect in adj.scheduled_effects:
        effect = raw_effect.model_copy(
            update={
                "created_tick": raw_effect.created_tick or tick,
                "source_actor": raw_effect.source_actor or actor_id,
            }
        )
        transitions.append(scheduled_effect_transition(effect))

    visible_ids = [str(e.entity_id) for e in projection.visible_entities]
    proposals = list(adj.transition_proposals)
    transitions.extend(_validate_proposals(proposals, world, action))

    if transitions:
        transitions.append(
            Transition(
                kind=TransitionKind.WORLD_MARK,
                payload={
                    "subject_kind": "entity",
                    "subject_id": actor_id,
                    "verb": "adjudicate",
                    "actor_id": actor_id,
                    "tick": tick,
                    "outcome": "adjudicated",
                    "summary": adj.ruling_text[:120] if adj.ruling_text else "adjudicated",
                },
            )
        )

    return transitions


def resolve_with_adjudication(
    world: WorldState,
    adapter: "LMAdapter",
    action: SemanticAction,
    intent: str,
    projection: "SemanticProjection",
    grounding: GroundingResult,
    result: ValidationResult,
) -> tuple[
    SemanticAction,
    ValidationResult,
    GroundingResult,
    list[Transition],
    Optional[str],
    bool,
]:
    """
    Run adjudication and optionally rescue via synthesized verb.

    Returns
    -------
    action, result, grounding, extra_transitions, ruling_text, rescued
    """
    from .compiler import compile_action
    from .grounding import classify

    # ── Deterministic physical-possibility rescue ──────────────────────────
    # Before invoking the LM adjudicator, check whether the actor's
    # available tags (entity + inventory + equipped) combined with any
    # nearby target tags could fire a property-interaction rule.  When
    # they can, the action is *not* orphaned — the engine has a real
    # mechanical answer.  Skipping the LM here keeps the same situation
    # reproducible and prevents the "atmospheric prose with zero state
    # change" trapdoor for physically-plausible creative inputs.
    from .property_interaction_resolver import try_property_possibility_rescue

    rescue = try_property_possibility_rescue(action, world, intent)
    if rescue is not None:
        rescued_action, rescued_result = rescue
        rescued_grounding = GroundingResult(
            zone=ActionZone.GROUNDED,
            grounded_entities=[action.actor],
        )
        return (
            rescued_action,
            rescued_result,
            rescued_grounding,
            [],
            None,
            True,
        )

    adj = invoke_adjudication(
        adapter, world, action, intent, projection, grounding, result,
    )

    if adj.synthesized_verb:
        rescued_action = action.model_copy(update={"verb": adj.synthesized_verb})
        rescued_grounding = classify(rescued_action, world, intent)
        if rescued_grounding.zone != ActionZone.ORPHANED:
            rescued_result = compile_action(
                rescued_action, world, projection=projection,
            )
            if rescued_result.valid:
                return (
                    rescued_action,
                    rescued_result,
                    rescued_grounding,
                    [],
                    adj.ruling_text or None,
                    True,
                )

    extra = materialize_adjudication(world, action, adj, projection)
    return action, result, grounding, extra, adj.ruling_text or None, False
