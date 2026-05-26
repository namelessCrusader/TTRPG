"""voxel_tavern pack-level npc_plans.yaml — every NPC gets a baseline plan."""

from __future__ import annotations

import pytest

from src.sim import npc_planner
from src.sim.schemas import PlanStepKind
from src.sim.world_loader import load_world_pack


@pytest.fixture
def world():
    return load_world_pack("worlds/voxel_tavern")


def _entity(world, pack_id: str):
    for ent in world.spatial.entities.values():
        if world.meta["entity_pack_ids"].get(str(ent.entity_id)) == pack_id:
            return ent
    raise KeyError(pack_id)


@pytest.mark.parametrize("pack_id", ["mira", "aldric", "linna", "tomas"])
def test_every_voxel_tavern_npc_gets_a_plan(world, pack_id):
    ent = _entity(world, pack_id)
    ent.active_plan = None
    plan = npc_planner.propose_plan(ent, world)
    assert plan is not None, f"{pack_id} should receive a pack plan"
    assert plan.steps, f"{pack_id} plan should have steps"


def test_mira_serve_plan_targets_bar(world):
    mira = _entity(world, "mira")
    mira.drive = "keep the ale flowing"
    mira.active_plan = None
    plan = npc_planner.propose_plan(mira, world)
    assert plan is not None
    assert plan.name == "tend the bar"
    # Should start by moving to the bar fixtures.
    first = plan.steps[0]
    assert first.kind == PlanStepKind.MOVE_TO


def test_aldric_door_watch_targets_door(world):
    aldric = _entity(world, "aldric")
    aldric.drive = "keep an eye on Tomas"
    aldric.active_plan = None
    plan = npc_planner.propose_plan(aldric, world)
    assert plan is not None
    assert plan.name == "door watch"
    door_step = next(
        s for s in plan.steps
        if s.kind == PlanStepKind.MOVE_TO and "door" in (s.label or "").lower()
    )
    assert door_step.payload["tile"] == [12, 14, 1]


def test_linna_performs_then_moves(world):
    linna = _entity(world, "linna")
    linna.drive = "perform songs that extract reactions from the crowd"
    linna.active_plan = None
    plan = npc_planner.propose_plan(linna, world)
    assert plan is not None
    assert plan.name == "perform on stage"
    assert any(s.kind == PlanStepKind.SPEAK_TO for s in plan.steps)


def test_tomas_drives_to_back_room(world):
    tomas = _entity(world, "tomas")
    tomas.drive = "convince Mira to let him use the back room for a meeting"
    tomas.active_plan = None
    plan = npc_planner.propose_plan(tomas, world)
    assert plan is not None
    assert plan.name == "back-room business"
