"""Tests for adjudication templates, economy, and unified session bootstrap."""

from __future__ import annotations

from src.sim.adjudication_templates import try_adjudication_templates
from src.sim.compiler import apply_transitions
from src.sim.economy import (
    get_market_price,
    init_market,
    record_trade,
)
from src.sim.game_loop import GameLoop, make_test_world
from src.sim.grounding import classify
from src.sim.lm_adapter import MockLMAdapter, OllamaLMAdapter
from src.sim.schemas import (
    ActionType,
    EntityKind,
    SemanticAction,
    TransitionKind,
    ValidationResult,
    WorldState,
)
from src.sim.session_bootstrap import build_game_loop, resolve_npc_policy
from src.sim.world_adjudicator import rule_based_adjudicate
from src.sim.npc_lm_policy import LMNpcPolicy
from src.sim.npc_policy import ReactivePolicy


def _player_id(world):
    for eid, ent in world.spatial.entities.items():
        if ent.kind == EntityKind.PLAYER:
            return eid
    return None


def test_trade_adjudication_template():
    world = make_test_world()
    player_id = _player_id(world)
    assert player_id is not None
    player = world.spatial.entities[player_id]
    player.stats["gold"] = 100.0
    init_market(world)

    action = SemanticAction(
        verb=ActionType.SYMBOLIC,
        actor=player_id,
        raw_input="buy ale for 5 gold",
    )
    grounding = classify(action, world, action.raw_input)
    result = ValidationResult(valid=False)

    adj = rule_based_adjudicate(world, action, action.raw_input, grounding, result)
    assert "ale" in adj.ruling_text.lower() or "gold" in adj.ruling_text.lower()
    assert adj.transition_proposals
    assert any(
        p.kind == TransitionKind.ENTITY_STAT_CHANGED.value
        and p.payload.get("stat") == "gold"
        for p in adj.transition_proposals
    )


def test_craft_adjudication_creates_item_proposal():
    world = make_test_world()
    player_id = _player_id(world)
    action = SemanticAction(
        verb=ActionType.SYMBOLIC,
        actor=player_id,
        raw_input="craft a wooden whistle from scraps",
    )
    adj = try_adjudication_templates(
        world,
        action,
        action.raw_input,
        classify(action, world, action.raw_input),
        ValidationResult(valid=False),
    )
    assert adj is not None
    assert any(
        p.kind == TransitionKind.ITEM_SYNTHESIZED.value for p in adj.transition_proposals
    )


def test_economy_price_shifts_after_trade():
    world = make_test_world()
    init_market(world)
    world.meta["market"] = {
        "region_main": {
            "ale": {
                "base_price": 5.0,
                "price": 5.0,
                "supply": 50.0,
                "demand": 50.0,
            },
        },
    }
    before = get_market_price(world, "ale")
    record_trade(world, "ale", quantity=5)
    after = get_market_price(world, "ale")
    assert after != before


def test_resolve_npc_policy_mock_vs_real():
    world = make_test_world()
    mock = MockLMAdapter()
    assert isinstance(resolve_npc_policy(mock, world), ReactivePolicy)

    real = OllamaLMAdapter(model="test")
    assert isinstance(resolve_npc_policy(real, world), LMNpcPolicy)
    assert isinstance(
        resolve_npc_policy(real, world, force_reactive=True),
        ReactivePolicy,
    )


def test_build_game_loop_matches_game_loop_defaults():
    world = make_test_world()
    adapter = OllamaLMAdapter(model="test")
    loop = build_game_loop(world, adapter)
    default_loop = GameLoop(world, adapter)
    assert type(loop.npc_policy) is type(default_loop.npc_policy)


def test_intimidate_adjudication_targets_npc():
    world = make_test_world()
    player_id = _player_id(world)
    assert player_id is not None
    # Find an NPC to target
    npc_id = next(
        eid for eid, e in world.spatial.entities.items()
        if e.kind == EntityKind.NPC
    )
    action = SemanticAction(
        verb=ActionType.SYMBOLIC,
        actor=player_id,
        target=str(npc_id),
        raw_input="intimidate the guard",
    )
    adj = try_adjudication_templates(
        world,
        action,
        action.raw_input,
        classify(action, world, action.raw_input),
        ValidationResult(valid=False),
    )
    assert adj is not None
    assert any(
        p.kind == TransitionKind.ENTITY_ALERTNESS_CHANGED.value
        for p in adj.transition_proposals
    )
    assert any("intimidation" in t for f in adj.facts for t in f.tags)


def test_message_adjudication_leaves_tile_mark():
    world = make_test_world()
    player_id = _player_id(world)
    action = SemanticAction(
        verb=ActionType.SYMBOLIC,
        actor=player_id,
        raw_input="leave a note that says meet me at midnight",
    )
    adj = rule_based_adjudicate(
        world,
        action,
        action.raw_input,
        classify(action, world, action.raw_input),
        ValidationResult(valid=False),
    )
    assert any(
        p.kind == TransitionKind.TILE_MARKED.value for p in adj.transition_proposals
    )
    assert adj.scheduled_effects


def test_generic_adjudication_always_sticky():
    world = make_test_world()
    player_id = _player_id(world)
    action = SemanticAction(
        verb=ActionType.SYMBOLIC,
        actor=player_id,
        raw_input="do something utterly bizarre and unprecedented",
    )
    adj = rule_based_adjudicate(
        world,
        action,
        action.raw_input,
        classify(action, world, action.raw_input),
        ValidationResult(valid=False),
    )
    assert adj.facts
    assert adj.transition_proposals
    assert adj.scheduled_effects


def test_destroy_adjudication_creates_environment_damage():
    world = make_test_world()
    player_id = _player_id(world)
    action = SemanticAction(
        verb=ActionType.SYMBOLIC,
        actor=player_id,
        raw_input="smash the bar table",
    )
    adj = try_adjudication_templates(
        world,
        action,
        action.raw_input,
        classify(action, world, action.raw_input),
        ValidationResult(valid=False),
    )
    assert adj is not None
    kinds = {p.kind for p in adj.transition_proposals}
    assert TransitionKind.ENVIRONMENT_STATE_CHANGED.value in kinds
    assert TransitionKind.NOISE_EVENT.value in kinds


def test_world_fact_tag_drives_goal_synthesis():
    from src.sim.schemas import WorldFact, WorldFactScope
    from src.sim.world_propagation import synthesize_goals_from_world_facts

    world = make_test_world()
    world.tick = 5
    player_id = _player_id(world)
    register_fact = WorldFact(
        claim="Someone bribed the guard with gold.",
        scope=WorldFactScope.WORLD,
        established_tick=world.tick,
        established_by=str(player_id),
        tags=["bribe", "adjudicated"],
    )
    world.world_facts.append(register_fact)
    transitions = synthesize_goals_from_world_facts(world)
    assert any(
        t.kind == TransitionKind.ENTITY_GOAL_ADDED
        and "bribe" in t.payload.get("goal", "").lower()
        for t in transitions
    )
