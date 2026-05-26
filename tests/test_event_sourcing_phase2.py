"""Event-sourced economy, pressure, and unified player fast-path."""

from __future__ import annotations

import copy
from pathlib import Path

from src.sim.compiler import apply_transitions
from src.sim.economy import economy_tick, init_market
from src.sim.game_loop import GameLoop, make_test_world
from src.sim.intent_router import try_parse_player_fast_path
from src.sim.lm_adapter import MockLMAdapter
from src.sim.pressure_eval import evaluate_pressures
from src.sim.scenario_executor import apply_effect_bundle
from src.sim.schemas import (
    ActionType,
    EntityKind,
    Event,
    SemanticAction,
    TransitionKind,
    new_event_id,
)
from src.sim.world_loader import load_world_pack


def _player_id(world):
    return next(
        eid for eid, e in world.spatial.entities.items()
        if e.kind == EntityKind.PLAYER
    )


def test_economy_tick_emits_market_transitions():
    world = make_test_world()
    init_market(world)
    world.meta["market"] = {
        "main": {
            "ale": {
                "base_price": 5.0,
                "price": 5.0,
                "supply": 50.0,
                "demand": 80.0,
            },
        },
    }
    world.tick = 5
    transitions = economy_tick(world)
    assert any(t.kind == TransitionKind.MARKET_UPDATED for t in transitions)
    apply_transitions(world, transitions)
    assert world.meta["market"]["main"]["ale"]["supply"] > 50.0


def test_pressure_eval_emits_transition():
    world = load_world_pack(Path(__file__).resolve().parents[1] / "worlds" / "spicy")
    world.tick = 5
    lines, transitions = evaluate_pressures(world)
    assert any(t.kind == TransitionKind.PRESSURE_STATE_UPDATED for t in transitions)
    apply_transitions(world, transitions)
    assert lines
    assert world.meta.get("active_pressures")


def test_scenario_schedule_uses_transition():
    world = make_test_world()
    _, transitions = apply_effect_bundle(
        world,
        {
            "schedule": [
                {"tick_offset": 2, "narration": "A bell tolls in the distance."},
            ],
        },
        cause="test",
    )
    assert any(
        t.kind == TransitionKind.SCHEDULED_EFFECT_QUEUED for t in transitions
    )
    assert len(world.scheduled_effects) == 1


def test_try_parse_player_fast_path_spell_list():
    world = make_test_world()
    player_id = _player_id(world)
    result = try_parse_player_fast_path("spells", player_id, world)
    assert result is not None
    assert result.narrative_override is not None
    assert "Spell" in result.narrative_override


def test_market_replay_from_clock_event():
    world = make_test_world()
    init_market(world)
    world.meta["market"] = {
        "main": {
            "ale": {
                "base_price": 5.0,
                "price": 5.0,
                "supply": 40.0,
                "demand": 70.0,
            },
        },
    }
    initial = copy.deepcopy(world)
    world.tick = 5
    transitions = economy_tick(world)
    apply_transitions(world, transitions)

    world.event_log.append(
        Event(
            event_id=new_event_id(),
            tick=world.tick,
            action=SemanticAction(
                verb=ActionType.AMBIENT,
                actor="system",
                raw_input="[clock_tick]",
            ),
            transitions=transitions,
            witnesses=[],
        )
    )

    loop = GameLoop(world, adapter=MockLMAdapter())
    reconstructed = loop.replay(initial)
    assert world.meta["market"] == reconstructed.meta["market"]
