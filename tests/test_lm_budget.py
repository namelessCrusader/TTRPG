"""Per-tick LM call budgeting."""

import copy
from pathlib import Path

from src.sim.game_loop import GameLoop
from src.sim.lm_adapter import MockLMAdapter
from src.sim.lm_budget import (
    allocate_lm_slots,
    lm_budget_remaining,
    lm_entity_priority,
    tick_lm_budget,
)
from src.sim.npc_lm_policy import LMNpcPolicy
from src.sim.npc_policy import ReactivePolicy
from src.sim.schemas import AlertnessLevel, NpcCognitionConfig
from src.sim.world_loader import load_world_pack


def _tavern():
    return load_world_pack(Path(__file__).resolve().parents[1] / "worlds" / "tavern")


def test_tick_lm_budget_resets_counter():
    world = _tavern()
    world.config.npc_policy.cognition.max_lm_calls_per_tick = 2
    tick_lm_budget(world)
    assert lm_budget_remaining(world) == 2


def test_allocate_lm_slots_respects_cap_and_priority_order():
    world = _tavern()
    world.config.npc_policy.cognition.max_lm_calls_per_tick = 1
    tick_lm_budget(world)

    entities = list(world.all_entities().items())[:3]
    for _eid, ent in entities:
        ent.meta["director_infer_budget"] = 1

    chosen = allocate_lm_slots(world, entities)
    assert len(chosen) == 1
    assert lm_budget_remaining(world) == 0


def test_parallel_decide_uses_reactive_when_over_budget():
    world = _tavern()
    world.config.npc_policy.cognition.max_lm_calls_per_tick = 1
    world.rng_seed = 7

    adapter = MockLMAdapter()
    policy = LMNpcPolicy(adapter, fallback=ReactivePolicy())
    loop = GameLoop(world, adapter=adapter, npc_policy=policy)

    from src.sim.director import NarrativeDirector
    from src.sim.social_stimulus import stimulus_priority

    NarrativeDirector(world).tick_start(world)
    tick_lm_budget(world)

    living = [
        (eid, ent)
        for eid, ent in world.all_entities().items()
        if ent.alive and ent.health > 0
    ]
    living.sort(key=lambda pair: (-stimulus_priority(pair[1]), pair[1].name))
    for _eid, ent in living:
        ent.meta["director_infer_budget"] = 2

    decisions = loop._parallel_decide(living)
    assert len(decisions) == len(living)
    branches = [
        world.all_entities()[eid].meta.get("last_policy_branch")
        for eid in decisions
    ]
    assert branches.count("budget_reactive") >= len(living) - 1


def test_lm_entity_priority_prefers_combat_over_bystander():
    world = _tavern()
    entities = list(world.all_entities().items())[:2]
    assert len(entities) >= 2
    (_e1, ent1), (_e2, ent2) = entities[0], entities[1]
    ent1.alertness = AlertnessLevel.COMBAT
    ent2.alertness = AlertnessLevel.UNAWARE
    assert lm_entity_priority(world, ent1) > lm_entity_priority(world, ent2)


def test_allocate_lm_slots_picks_higher_priority_entity():
    world = _tavern()
    world.config.npc_policy.cognition.max_lm_calls_per_tick = 1
    tick_lm_budget(world)
    entities = list(world.all_entities().items())[:2]
    assert len(entities) >= 2
    (eid_low, ent_low), (eid_high, ent_high) = entities[0], entities[1]
    ent_low.alertness = AlertnessLevel.UNAWARE
    ent_high.alertness = AlertnessLevel.COMBAT
    for _eid, ent in entities:
        ent.meta["director_infer_budget"] = 1
    chosen = allocate_lm_slots(world, entities)
    assert eid_high in chosen
    assert eid_low not in chosen
