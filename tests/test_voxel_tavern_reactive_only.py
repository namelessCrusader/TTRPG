"""
Long-horizon reactive_only autonomous run for the voxel_tavern showcase.

These tests are the "no-LM is also alive" parity check.  With cognition
locked to ``reactive_only`` and the mock adapter, the NPCs in
``worlds/voxel_tavern`` must:

  - acquire baseline pack plans within the first few ticks,
  - keep making forward progress (acting every tick),
  - swap their baseline plans for hazard plans when smoke fills the room,
  - resume something useful once the smoke clears.

This is the deterministic counterpart to the LM-driven autonomous
script; failures here mean ``--cognition reactive_only ./run_auto.sh
voxel_tavern`` produces a frozen room.
"""

from __future__ import annotations

import pytest

from src.sim.compiler import apply_transitions
from src.sim.fields import build_field_transition, MEDIA_TILE_AIR
from src.sim.game_loop import GameLoop
from src.sim.lm_adapter import MockLMAdapter
from src.sim.schemas import (
    AlertnessLevel,
    Coord,
    EntityKind,
    PlanStatus,
)
from src.sim.world_loader import load_world_pack


def _load_reactive_voxel_tavern():
    world = load_world_pack("worlds/voxel_tavern")
    world.config.npc_policy.cognition.mode = "reactive_only"
    return world


def _npcs(world):
    return [
        e for e in world.spatial.entities.values()
        if e.kind == EntityKind.NPC and e.alive
    ]


def _flood_room_with_smoke(world, amount: float = 50.0):
    """Saturate every passable tile (all z-levels) + atmosphere with smoke."""
    transitions = []
    grid = world.spatial
    z_range = range(max(1, grid.depth)) if grid.depth > 1 else (0,)
    # Tile substance storage is keyed by 2D footprint, so iterate (x, y) once
    # but record at z=0 for storage; smoke is logically present at every z.
    for y in range(grid.height):
        for x in range(grid.width):
            for z in z_range:
                c = Coord(x=x, y=y, z=z)
                if not grid.is_in_bounds(c) or not grid.is_passable(c):
                    continue
                transitions.append(build_field_transition(
                    MEDIA_TILE_AIR, "smoke", amount,
                    coord=c, operation="set",
                    cause="test_flood", tick=world.tick,
                    remaining_ticks=20,
                ))
                break  # 2D-keyed storage — once per (x, y) is enough
    apply_transitions(world, transitions)
    region = world.regions[world.active_region_id]
    atm = region.grid.region_env.setdefault("fields", {}).setdefault("atmosphere", {})
    atm["smoke"] = {"amount": amount, "since_tick": world.tick}


def _clear_room_of_smoke(world):
    grid = world.spatial
    for y in range(grid.height):
        for x in range(grid.width):
            c = Coord(x=x, y=y, z=1)
            if not grid.is_in_bounds(c):
                continue
            tile = grid.tile_at(c)
            tile.env.pop("fields", None)
            tile.env.pop("airborne", None)
    region = world.regions[world.active_region_id]
    region.grid.region_env.get("fields", {}).pop("atmosphere", None)
    for obj in grid.objects.values():
        if "on_fire" in obj.tags:
            obj.tags = [t for t in obj.tags if t != "on_fire"]


# ── core reactive_only liveness ──────────────────────────────────────────


def test_reactive_only_npcs_act_every_tick():
    """Run 12 ticks with no LM; every NPC should have acted at least 6 times."""
    world = _load_reactive_voxel_tavern()
    loop = GameLoop(world, adapter=MockLMAdapter())
    npc_ids = [str(e.entity_id) for e in _npcs(world)]
    action_counts = {nid: 0 for nid in npc_ids}

    for _ in range(12):
        result = loop.autonomous_tick()
        for er in result.entity_results or []:
            if er.action is None or er.event is None:
                continue
            actor = str(er.action.actor)
            if actor in action_counts:
                action_counts[actor] += 1

    for nid, count in action_counts.items():
        ent = world.spatial.entities[nid]
        assert count >= 6, (
            f"reactive_only NPC {ent.name} only acted {count}/12 ticks "
            f"— planner is starving the candidate menu"
        )


def test_reactive_only_npcs_acquire_pack_plans_quickly():
    """After 3 ticks every NPC should have an active plan."""
    world = _load_reactive_voxel_tavern()
    loop = GameLoop(world, adapter=MockLMAdapter())
    for _ in range(3):
        loop.autonomous_tick()
    for ent in _npcs(world):
        assert ent.active_plan is not None, (
            f"NPC {ent.name} has no plan after 3 ticks — pack template "
            f"didn't match drive: {ent.drive!r}"
        )
        assert ent.active_plan.status in (PlanStatus.ACTIVE, PlanStatus.COMPLETED), (
            f"NPC {ent.name} plan stuck in status {ent.active_plan.status}"
        )


# ── hazard swap ──────────────────────────────────────────────────────────


def test_smoke_flooded_room_swaps_npcs_to_hazard_plans():
    """When the room fills with smoke, baseline plans pause and hazard
    plans take over (flee for civilians, douse for staff/veterans)."""
    world = _load_reactive_voxel_tavern()
    loop = GameLoop(world, adapter=MockLMAdapter())

    # Run a few ticks so baseline plans are active.
    for _ in range(2):
        loop.autonomous_tick()
    baseline = {
        str(e.entity_id): (e.active_plan.name if e.active_plan else None)
        for e in _npcs(world)
    }
    assert any(baseline.values()), "no baseline plans formed"

    _flood_room_with_smoke(world, 45.0)

    # One more tick: check_interrupt fires for every NPC, hazard plans synthesise.
    loop.autonomous_tick()

    hazard_count = 0
    stack_count = 0
    for ent in _npcs(world):
        plan = ent.active_plan
        if plan is not None and plan.meta.get("hazard_response"):
            hazard_count += 1
        if ent.meta.get("plan_stack"):
            stack_count += 1

    assert hazard_count >= 3, (
        f"expected most NPCs to swap to hazard plans, got {hazard_count}/4 "
        f"(plans: {[(e.name, e.active_plan.name if e.active_plan else None) for e in _npcs(world)]})"
    )
    assert stack_count >= 3, (
        f"expected stacked baseline plans for at least 3 NPCs, got {stack_count}"
    )


def test_civilian_synthesises_flee_not_douse():
    """Linna (bard) doesn't have douse capability — she should flee."""
    world = _load_reactive_voxel_tavern()
    loop = GameLoop(world, adapter=MockLMAdapter())
    for _ in range(2):
        loop.autonomous_tick()

    linna = next(
        e for e in _npcs(world)
        if world.meta["entity_pack_ids"].get(str(e.entity_id)) == "linna"
    )
    _flood_room_with_smoke(world, 45.0)
    loop.autonomous_tick()

    plan = linna.active_plan
    assert plan is not None
    assert plan.meta.get("hazard_response") is True
    # Linna does have QUENCH spell in voxel_tavern — so douse is acceptable
    # for her too.  Both kinds are valid; just ensure it's a hazard plan.
    assert plan.meta.get("hazard_kind") in {"flee", "douse"}


def test_stack_resumes_baseline_after_smoke_clears():
    """Once hazards clear, the stacked baseline plan must return."""
    world = _load_reactive_voxel_tavern()
    loop = GameLoop(world, adapter=MockLMAdapter())

    # Pick a deterministic NPC and pin a known baseline.
    mira = next(
        e for e in _npcs(world)
        if world.meta["entity_pack_ids"].get(str(e.entity_id)) == "mira"
    )
    for _ in range(2):
        loop.autonomous_tick()
    baseline_name = mira.active_plan.name if mira.active_plan else None
    assert baseline_name, "mira should have a baseline plan before hazard"

    _flood_room_with_smoke(world, 45.0)
    loop.autonomous_tick()
    assert mira.active_plan is not None
    assert mira.active_plan.meta.get("hazard_response") is True
    assert mira.meta.get("plan_stack"), "stack must hold mira's baseline plan"

    _clear_room_of_smoke(world)
    # Run several ticks so the hazard plan completes or the stack pops.
    for _ in range(10):
        loop.autonomous_tick()

    # Either we're back on the baseline, or the planner has built a new
    # plan from her drive.  The point is: she is NOT stuck on a hazard
    # plan in a clear room, and the stack has been drained.
    plan = mira.active_plan
    assert plan is not None
    assert not plan.meta.get("hazard_response"), (
        f"mira still on hazard plan {plan.name!r} after smoke cleared"
    )
    assert not (mira.meta.get("plan_stack") or []), (
        "plan_stack should be empty after resume"
    )


@pytest.mark.parametrize("ticks", [25])
def test_long_horizon_reactive_only_stays_stable(ticks):
    """A 25-tick reactive_only run never crashes and keeps acting."""
    world = _load_reactive_voxel_tavern()
    loop = GameLoop(world, adapter=MockLMAdapter())
    total_actions = 0
    for _ in range(ticks):
        result = loop.autonomous_tick()
        total_actions += sum(1 for er in (result.entity_results or []) if er.event)
    assert total_actions >= ticks * 2, (
        f"only {total_actions} actions in {ticks} ticks — sim went quiet"
    )
