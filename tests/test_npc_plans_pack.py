"""Pack-driven npc_plans.yaml loading and matching."""

from __future__ import annotations

from src.sim import npc_planner
from src.sim.schemas import EntityKind, PlanStepKind
from src.sim.world_loader import load_world_pack


def test_tavern_mira_gets_bar_plan():
    world = load_world_pack("worlds/tavern")
    ent = next(
        e
        for e in world.spatial.entities.values()
        if world.meta["entity_pack_ids"].get(str(e.entity_id)) == "mira"
    )
    ent.drive = "keep the ale flowing; manage the mood of the room"
    ent.active_plan = None

    plan = npc_planner.propose_plan(ent, world)
    assert plan is not None
    assert plan.name == "tend the bar"
    assert any(s.kind == PlanStepKind.SPEND_TICKS for s in plan.steps)


def test_pack_plan_beats_builtin_patrol():
    world = load_world_pack("worlds/tavern")
    ent = next(
        e for e in world.spatial.entities.values()
        if e.kind == EntityKind.NPC
        and world.meta["entity_pack_ids"].get(str(e.entity_id)) == "aldric"
    )
    ent.drive = "guard the door and patrol when bored"
    ent.active_plan = None
    plan = npc_planner.propose_plan(ent, world)
    assert plan is not None
    assert plan.name == "door watch"
