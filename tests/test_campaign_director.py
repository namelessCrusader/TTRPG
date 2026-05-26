"""Tests for condition-triggered campaign director (Phase 2)."""

from __future__ import annotations

from pathlib import Path

from src.sim.campaign_director import evaluate_reactive_triggers
from src.sim.game_loop import GameLoop
from src.sim.lm_adapter import MockLMAdapter
from src.sim.schemas import WorldFact, WorldFactScope
from src.sim.world_loader import load_world_pack
from src.sim.pressure_eval import conditions_met


def test_tavern_loads_reactive_triggers():
    world = load_world_pack(Path(__file__).resolve().parents[1] / "worlds" / "tavern")
    assert len(world.config.reactive_triggers) >= 3
    ids = {t["id"] for t in world.config.reactive_triggers}
    assert "violence_escalation" in ids
    assert "intimidation_tension" in ids


def test_violence_fact_fires_campaign_trigger():
    world = load_world_pack(Path(__file__).resolve().parents[1] / "worlds" / "tavern")
    world.tick = 5
    world.world_facts.append(
        WorldFact(
            claim="Someone smashed the bar table.",
            scope=WorldFactScope.TILE,
            subject_id="4,10,0",
            established_tick=world.tick,
            established_by="player",
            tags=["violence", "adjudicated"],
        )
    )

    lines = evaluate_reactive_triggers(world)
    assert any("violence_escalation" in ln for ln in lines)
    assert "violence_escalation" in (world.meta.get("campaign_director_fired") or [])

    mira = next(e for e in world.spatial.entities.values() if e.name == "Mira")
    assert any("violence" in g.lower() or "brawl" in g.lower() for g in mira.goals)


def test_trigger_fires_only_once():
    world = load_world_pack(Path(__file__).resolve().parents[1] / "worlds" / "tavern")
    world.tick = 3
    world.world_facts.append(
        WorldFact(
            claim="A threatening gesture was made.",
            scope=WorldFactScope.WORLD,
            established_tick=world.tick,
            tags=["intimidation", "adjudicated"],
        )
    )

    first = evaluate_reactive_triggers(world)
    second = evaluate_reactive_triggers(world)
    assert first
    assert not second


def test_game_loop_step_fires_campaign_after_adjudication():
    world = load_world_pack(Path(__file__).resolve().parents[1] / "worlds" / "tavern")
    world.config.consequence_policy.record_traces = False
    loop = GameLoop(world, MockLMAdapter())

    result = loop.step("smash the bar table into splinters")
    assert result.campaign_lines
    assert any("[campaign]" in ln for ln in result.campaign_lines)
    assert "environment_damage" in (world.meta.get("campaign_director_fired") or [])


def test_world_fact_tag_condition_respects_within_ticks():
    from src.sim.pressure_eval import conditions_met

    world = load_world_pack(Path(__file__).resolve().parents[1] / "worlds" / "tavern")
    world.tick = 50
    world.world_facts.append(
        WorldFact(
            claim="Old violence.",
            tags=["violence"],
            established_tick=10,
        )
    )
    assert not conditions_met(
        [{"type": "world_fact_tag", "tag": "violence", "within_ticks": 12}],
        world,
    )
    world.world_facts.append(
        WorldFact(
            claim="Recent violence.",
            tags=["violence"],
            established_tick=48,
        )
    )
    assert conditions_met(
        [{"type": "world_fact_tag", "tag": "violence", "within_ticks": 12}],
        world,
    )


def test_conditional_scenario_beat():
    from src.sim.scenario_executor import apply_conditional_scenario_beats

    world = load_world_pack(Path(__file__).resolve().parents[1] / "worlds" / "tavern")
    scenario = dict((world.config.extra or {}).get("scenario") or {})
    scenario["beats"] = list(scenario.get("beats") or []) + [
        {
            "id": "test_reactive_beat",
            "tick": 0,
            "when": [{"type": "world_fact_tag", "tag": "message"}],
            "effects": {
                "inject_goals": [
                    {
                        "entity": "mira",
                        "goal": "Find who left the mysterious note",
                        "priority": 0.8,
                    }
                ]
            },
        }
    ]
    world.config.extra["scenario"] = scenario
    world.world_facts.append(
        WorldFact(
            claim="A note was left.",
            tags=["message"],
            established_tick=world.tick,
        )
    )

    lines = apply_conditional_scenario_beats(world)
    assert lines
    mira = next(e for e in world.spatial.entities.values() if e.name == "Mira")
    assert any("note" in g.lower() for g in mira.goals)
