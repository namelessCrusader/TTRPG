"""
Tests for the open-verb generic compiler path.

Architectural acceptance points:
  - Verbs are open: any string is a legal verb; the engine does not
    crash on unknown ones.
  - Kernel verbs (move, attack, take, give, throw) route to bespoke
    physics compilers. Everything else flows through the generic path.
  - The generic path consults, in order: pack-declared verb templates,
    then built-in defaults for conventional verbs, then a pure-freeform
    path.
  - LM-proposed effects are validated against canonical state — out-of-
    bounds values are clipped; references to non-existent entities are
    dropped; ENTITY_MOVED and ITEM_TRANSFERRED are always dropped from
    LM proposals because they are kernel-only.
  - Optional contests resolve deterministically against attributes;
    success vs failure determines which effect set the template applies.
  - An "interaction record" transition (DIALOGUE_SPOKEN with verb in
    payload) is always emitted so the narrator has substance.
"""

import pytest

from src.sim.compiler import (
    _validate_proposal,
    apply_transitions,
    compile_action,
)
from src.sim.game_loop import make_test_world
from src.sim.schemas import (
    ActionType,
    AlertnessLevel,
    ContestSpec,
    Coord,
    EdgeKind,
    EmotionalState,
    EntityId,
    EntityKind,
    IntentBlock,
    SemanticAction,
    StyleBlock,
    Transition,
    TransitionKind,
    TransitionProposal,
    VerbTemplate,
)


def _player_id(world) -> EntityId:
    for eid, e in world.spatial.entities.items():
        if e.kind == EntityKind.PLAYER:
            return eid
    raise RuntimeError("no player")


def _guard_id(world) -> EntityId:
    for eid, e in world.spatial.entities.items():
        if e.kind == EntityKind.NPC:
            return eid
    raise RuntimeError("no guard")


def _verb_record(transitions: list[Transition]) -> Transition | None:
    """Find the catch-all interaction-record transition (DIALOGUE_SPOKEN
    with a 'verb' key)."""
    for t in transitions:
        if t.kind == TransitionKind.DIALOGUE_SPOKEN and "verb" in t.payload:
            return t
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Open verbs: arbitrary strings compile and produce a record event
# ─────────────────────────────────────────────────────────────────────────────


def test_unknown_verb_compiles_and_records():
    """A verb the engine has never seen still compiles successfully."""
    world = make_test_world()
    pid = _player_id(world)
    action = SemanticAction(
        verb="ululate",
        actor=pid,
        target=None,
        intent=IntentBlock(manner="a mournful ululation"),
    )
    result = compile_action(action, world)
    assert result.valid
    record = _verb_record(result.concrete_transitions)
    assert record is not None
    assert record.payload["verb"] == "ululate"
    assert record.payload["manner"] == "a mournful ululation"


def test_unknown_verb_with_proposed_effect_applies_validated_effect():
    """LM proposes an effect; engine validates and applies it."""
    world = make_test_world()
    pid = _player_id(world)
    gid = _guard_id(world)
    action = SemanticAction(
        verb="serenade",
        actor=pid,
        target=gid,
        proposed_effects=[
            TransitionProposal(
                kind=TransitionKind.ENTITY_EMOTIONAL_STATE_CHANGED.value,
                payload={"entity_id": gid, "to": "happy"},
            )
        ],
    )
    result = compile_action(action, world)
    assert result.valid
    apply_transitions(world, result.concrete_transitions)
    assert world.spatial.entities[gid].emotional_state == EmotionalState.HAPPY


# ─────────────────────────────────────────────────────────────────────────────
# Proposal validation: reachability epsilon
# ─────────────────────────────────────────────────────────────────────────────


def test_proposal_referring_to_unknown_entity_is_dropped():
    """LM hallucinates an entity id; the proposal is dropped silently."""
    world = make_test_world()
    pid = _player_id(world)
    action = SemanticAction(
        verb="incite",
        actor=pid,
        target=None,
        proposed_effects=[
            TransitionProposal(
                kind=TransitionKind.ENTITY_EMOTIONAL_STATE_CHANGED.value,
                payload={"entity_id": "ent_does_not_exist", "to": "angry"},
            )
        ],
    )
    result = compile_action(action, world)
    assert result.valid
    # The verb record survives; the proposal does not.
    other = [
        t for t in result.concrete_transitions
        if t.kind != TransitionKind.DIALOGUE_SPOKEN
    ]
    assert other == []


def test_proposal_with_health_delta_is_clipped():
    """LM proposes an unreasonable health swing; engine clips it."""
    world = make_test_world()
    pid = _player_id(world)
    gid = _guard_id(world)
    action = SemanticAction(
        verb="curse",
        actor=pid,
        target=gid,
        proposed_effects=[
            TransitionProposal(
                kind=TransitionKind.ENTITY_HEALTH_CHANGED.value,
                payload={"entity_id": gid, "delta": -9999},
            )
        ],
    )
    result = compile_action(action, world)
    assert result.valid
    apply_transitions(world, result.concrete_transitions)
    # Clipped to engine's generic max (10).
    assert world.spatial.entities[gid].health >= 90  # 100 - 10


def test_lm_proposal_for_entity_moved_is_dropped():
    """ENTITY_MOVED is kernel-only — LM can't move via proposed effects."""
    world = make_test_world()
    pid = _player_id(world)
    action = SemanticAction(
        verb="teleport",
        actor=pid,
        target=None,
        proposed_effects=[
            TransitionProposal(
                kind=TransitionKind.ENTITY_MOVED.value,
                payload={
                    "entity_id": pid,
                    "from": {"x": 2, "y": 2},
                    "to": {"x": 15, "y": 15},
                },
            )
        ],
    )
    result = compile_action(action, world)
    assert result.valid
    moves = [
        t for t in result.concrete_transitions
        if t.kind == TransitionKind.ENTITY_MOVED
    ]
    assert moves == []


def test_lm_proposal_with_invalid_emotional_state_is_dropped():
    world = make_test_world()
    pid = _player_id(world)
    gid = _guard_id(world)
    action = SemanticAction(
        verb="bless",
        actor=pid,
        target=gid,
        proposed_effects=[
            TransitionProposal(
                kind=TransitionKind.ENTITY_EMOTIONAL_STATE_CHANGED.value,
                payload={"entity_id": gid, "to": "transcendent"},
            )
        ],
    )
    result = compile_action(action, world)
    assert result.valid
    emo_changes = [
        t for t in result.concrete_transitions
        if t.kind == TransitionKind.ENTITY_EMOTIONAL_STATE_CHANGED
    ]
    assert emo_changes == []


def test_lm_proposed_edge_with_unknown_nodes_is_dropped():
    world = make_test_world()
    pid = _player_id(world)
    action = SemanticAction(
        verb="pledge",
        actor=pid,
        target=None,
        proposed_effects=[
            TransitionProposal(
                kind=TransitionKind.EDGE_CREATED.value,
                payload={
                    "source": pid,
                    "target": "ent_does_not_exist",
                    "kind": "allies_of",
                    "weight": 0.5,
                },
            )
        ],
    )
    result = compile_action(action, world)
    assert result.valid
    edges = [
        t for t in result.concrete_transitions
        if t.kind == TransitionKind.EDGE_CREATED
    ]
    assert edges == []


def test_lm_proposed_edge_weight_is_clipped_to_unit_range():
    world = make_test_world()
    pid = _player_id(world)
    gid = _guard_id(world)
    action = SemanticAction(
        verb="impress",
        actor=pid,
        target=gid,
        proposed_effects=[
            TransitionProposal(
                kind=TransitionKind.EDGE_CREATED.value,
                payload={
                    "source": gid,
                    "target": pid,
                    "kind": "respects",
                    "weight": 9.0,  # absurd
                },
            )
        ],
    )
    result = compile_action(action, world)
    assert result.valid
    edges = [
        t for t in result.concrete_transitions
        if t.kind == TransitionKind.EDGE_CREATED
    ]
    assert len(edges) == 1
    assert -1.0 <= edges[0].payload["weight"] <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Contests
# ─────────────────────────────────────────────────────────────────────────────


def test_contest_outcome_recorded_on_interaction():
    """A contest is run and the outcome is on the record transition."""
    world = make_test_world()
    pid = _player_id(world)
    gid = _guard_id(world)
    action = SemanticAction(
        verb="impress",
        actor=pid,
        target=gid,
        contest=ContestSpec(
            actor_attribute="persuasion", target_attribute="perception"
        ),
    )
    result = compile_action(action, world)
    record = _verb_record(result.concrete_transitions)
    assert record is not None
    assert record.payload["contest_outcome"] in ("success", "failure")


def test_contest_deterministic_for_same_action_and_tick():
    """Same action_id + world.tick → same contest outcome (replay safety)."""
    from src.sim.schemas import new_event_id
    world = make_test_world()
    pid = _player_id(world)
    gid = _guard_id(world)

    action = SemanticAction(
        action_id="act_fixed_id",
        verb="impress",
        actor=pid,
        target=gid,
        contest=ContestSpec(
            actor_attribute="persuasion", target_attribute="perception"
        ),
    )
    r1 = compile_action(action, world)
    r2 = compile_action(action, world)
    o1 = _verb_record(r1.concrete_transitions).payload["contest_outcome"]
    o2 = _verb_record(r2.concrete_transitions).payload["contest_outcome"]
    assert o1 == o2


# ─────────────────────────────────────────────────────────────────────────────
# Verb templates from a pack
# ─────────────────────────────────────────────────────────────────────────────


def test_pack_template_applies_default_effects_on_success():
    """When the verb has a pack template, its on_success effects apply."""
    world = make_test_world()
    pid = _player_id(world)
    gid = _guard_id(world)
    # Inject a template into the world config at runtime to make the
    # test self-contained.
    world.config.verb_templates["nudge"] = VerbTemplate(
        verb="nudge",
        effects_on_success=[
            TransitionProposal(
                kind=TransitionKind.ENTITY_EMOTIONAL_STATE_CHANGED.value,
                payload={"entity_id": "$target", "to": "happy"},
            )
        ],
    )
    action = SemanticAction(verb="nudge", actor=pid, target=gid)
    result = compile_action(action, world)
    apply_transitions(world, result.concrete_transitions)
    assert world.spatial.entities[gid].emotional_state == EmotionalState.HAPPY


def test_pack_template_substitutes_actor_and_target_placeholders():
    """$actor and $target placeholders get substituted with action ids."""
    world = make_test_world()
    pid = _player_id(world)
    gid = _guard_id(world)
    world.config.verb_templates["bond"] = VerbTemplate(
        verb="bond",
        effects_on_success=[
            TransitionProposal(
                kind=TransitionKind.EDGE_CREATED.value,
                payload={
                    "source": "$actor",
                    "target": "$target",
                    "kind": "respects",
                    "weight": 0.3,
                },
            )
        ],
    )
    action = SemanticAction(verb="bond", actor=pid, target=gid)
    result = compile_action(action, world)
    edges = [t for t in result.concrete_transitions
             if t.kind == TransitionKind.EDGE_CREATED]
    assert len(edges) == 1
    assert edges[0].payload["source"] == pid
    assert edges[0].payload["target"] == gid


def test_pack_template_with_contest_branches_on_outcome():
    """A template with a contest chooses on_success vs on_failure."""
    world = make_test_world()
    pid = _player_id(world)
    gid = _guard_id(world)
    # Force-failure: target's perception astronomically high.
    world.spatial.entities[gid].attributes["perception"] = 100
    world.spatial.entities[pid].attributes["persuasion"] = 1

    world.config.verb_templates["deceive_v2"] = VerbTemplate(
        verb="deceive_v2",
        contest=ContestSpec(
            actor_attribute="persuasion", target_attribute="perception"
        ),
        effects_on_success=[
            TransitionProposal(
                kind=TransitionKind.ENTITY_EMOTIONAL_STATE_CHANGED.value,
                payload={"entity_id": "$target", "to": "friendly"},
            )
        ],
        effects_on_failure=[
            TransitionProposal(
                kind=TransitionKind.ENTITY_EMOTIONAL_STATE_CHANGED.value,
                payload={"entity_id": "$target", "to": "suspicious"},
            )
        ],
    )

    # Need a contest on the action to trigger the resolution path.
    action = SemanticAction(
        verb="deceive_v2",
        actor=pid,
        target=gid,
        contest=ContestSpec(
            actor_attribute="persuasion", target_attribute="perception"
        ),
    )
    result = compile_action(action, world)
    apply_transitions(world, result.concrete_transitions)
    # With actor=1 vs target=100, failure is overwhelmingly likely.
    assert world.spatial.entities[gid].emotional_state == EmotionalState.SUSPICIOUS


# ─────────────────────────────────────────────────────────────────────────────
# Default pack ships a few templates — make sure the loader read them
# ─────────────────────────────────────────────────────────────────────────────


def test_default_pack_loads_verb_templates():
    """The default world pack's verb_templates.yaml should be loaded."""
    world = make_test_world()
    # The default pack ships templates for at least these:
    assert "console" in world.config.verb_templates
    assert "tease" in world.config.verb_templates
    assert "haggle" in world.config.verb_templates


# ─────────────────────────────────────────────────────────────────────────────
# Kernel verbs are unaffected — they still route to bespoke compilers
# ─────────────────────────────────────────────────────────────────────────────


def test_kernel_verb_move_still_uses_pathing():
    """Kernel verb MOVE still runs through its bespoke compiler."""
    world = make_test_world()
    pid = _player_id(world)
    dest = Coord(x=3, y=2)
    action = SemanticAction(verb="move", actor=pid, target=dest)
    result = compile_action(action, world)
    assert result.valid
    moves = [
        t for t in result.concrete_transitions
        if t.kind == TransitionKind.ENTITY_MOVED
    ]
    assert len(moves) == 1
    assert moves[0].payload["to"] == {"x": 3, "y": 2}


def test_kernel_verb_attack_ignores_lm_health_proposal():
    """ATTACK uses its bespoke compiler — LM-proposed effects are
    layered on top but the cannot bypass health-change validation."""
    world = make_test_world()
    pid = _player_id(world)
    gid = _guard_id(world)
    world.spatial.entities[gid].position = Coord(x=3, y=2)
    action = SemanticAction(
        verb="attack",
        actor=pid,
        target=gid,
        proposed_effects=[
            TransitionProposal(
                kind=TransitionKind.ENTITY_HEALTH_CHANGED.value,
                payload={"entity_id": gid, "delta": -500},
            )
        ],
    )
    result = compile_action(action, world)
    assert result.valid
    # The kernel attack compiler emits exactly its own damage transition.
    # The LM-proposed -500 is NOT layered on top because attack is kernel,
    # not generic.
    health_changes = [
        t for t in result.concrete_transitions
        if t.kind == TransitionKind.ENTITY_HEALTH_CHANGED
    ]
    assert all(t.payload["delta"] > -100 for t in health_changes)
