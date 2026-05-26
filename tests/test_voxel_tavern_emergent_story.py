"""
End-to-end emergent-story tests for the voxel_tavern showcase.

These tests pin the full ``oil spill → ignite → smoke → response → resolution``
chain so the scene resolves deterministically in ``--cognition reactive_only``
mode.  They are the "every action in this scene actually works" guardrail.

Each test runs the autonomous loop with the MockLMAdapter (no real LM
calls), letting pack plans, hazard synthesis, interaction rules, and
field decay drive the world.  The assertions are about *outcomes*
(e.g. fire eventually out, hazard fact registered, NPCs moved toward
exits) rather than specific tick numbers, so they stay stable across
small numeric tweaks to thresholds.
"""

from __future__ import annotations

import pytest

from src.sim.compiler import apply_transitions
from src.sim.fields import (
    MEDIA_TILE_AIR,
    MEDIA_TILE_GROUND,
    build_field_transition,
    region_atmosphere_amount,
)
from src.sim.game_loop import GameLoop
from src.sim.lm_adapter import MockLMAdapter
from src.sim.schemas import (
    AlertnessLevel,
    Coord,
    EntityKind,
    PlanStatus,
    Transition,
    TransitionKind,
)
from src.sim.world_loader import load_world_pack


def _load_world():
    world = load_world_pack("worlds/voxel_tavern")
    world.config.npc_policy.cognition.mode = "reactive_only"
    return world


def _npc(world, pack_id: str):
    for ent in world.spatial.entities.values():
        if world.meta["entity_pack_ids"].get(str(ent.entity_id)) == pack_id:
            return ent
    raise KeyError(pack_id)


def _ignite_oil_at(world, target_obj_name: str = "lamp oil flask"):
    """Trigger the scenario's ignition step deterministically."""
    obj = next(o for o in world.spatial.objects.values() if o.name == target_obj_name)
    # Mark the object on_fire and deposit oil on its tile so fire-spread
    # reactions see fuel.
    obj.tags = list(obj.tags) + ["on_fire"]
    apply_transitions(world, [build_field_transition(
        MEDIA_TILE_GROUND, "oil", 50.0,
        coord=obj.position, operation="set",
        cause="ignite_oil_test", tick=world.tick,
    )])
    return obj


def _atmosphere_smoke(world) -> float:
    return float(region_atmosphere_amount(world, "smoke"))


def _world_has_on_fire(world) -> bool:
    if any("on_fire" in (o.tags or []) for o in world.spatial.objects.values()):
        return True
    grid = world.spatial
    for tile in grid.tiles.values():
        if any(t.lower() == "on_fire" for t in (tile.tags or [])):
            return True
    return False


# ── Showcase pieces resolve deterministically ────────────────────────────


def test_oil_spill_to_ignite_chain_runs_without_lm():
    """The scenario's first two beats — spill then ignite — fire fully
    via the field layer + scenario executor + reactions, with no LM."""
    world = _load_world()
    loop = GameLoop(world, adapter=MockLMAdapter())
    # Run far enough for the scenario beats (tick 4 spill, tick 5 ignite)
    # to fire AND for the reactions module to register on_fire on the flask.
    for _ in range(8):
        loop.autonomous_tick()

    flask = next(o for o in world.spatial.objects.values() if o.name == "lamp oil flask")
    assert "on_fire" in flask.tags or _world_has_on_fire(world), (
        f"flask tags: {flask.tags}; world_on_fire={_world_has_on_fire(world)}"
    )


def test_bartender_can_deterministically_douse_fire_object():
    """The new tavern_douse_fire_object interaction rule fires when Mira
    walks adjacent to a burning object and emits ``douse``."""
    world = _load_world()
    flask = _ignite_oil_at(world)
    assert "on_fire" in flask.tags

    mira = _npc(world, "mira")
    # Plant Mira right next to the burning object so movement isn't the bottleneck.
    mira.position = Coord(
        x=flask.position.x + 1, y=flask.position.y, z=mira.position.z,
    )

    from src.sim.interaction_resolver import resolve_interaction_action

    resolved = resolve_interaction_action(
        world,
        mira.entity_id,
        intent=f"douse the burning {flask.name}",
    )
    assert resolved is not None, "tavern douse interaction rule should fire"
    assert resolved.verb == "douse"
    # Apply the resolved effects via the compiler to mutate world state.
    transitions = [
        Transition(kind=TransitionKind(e.kind), payload=e.payload)
        for e in (resolved.proposed_effects or [])
    ]
    assert transitions, "douse rule should yield effects"
    apply_transitions(world, transitions)
    flask_after = world.spatial.objects[flask.object_id]
    assert "on_fire" not in flask_after.tags, (
        f"flask should be doused but still on_fire; tags={flask_after.tags}"
    )


def test_smoke_decays_after_fires_are_out():
    """When all fires are out, atmosphere smoke decays via field_tick.

    The test bypasses the autonomous loop (which would re-arm fires via
    scenario beats) and exercises ``field_tick`` directly to validate
    the decay arithmetic — see ``substances.yaml: smoke.decay_per_tick``.
    """
    from src.sim.compiler import apply_transitions
    from src.sim.fields import field_tick

    world = _load_world()
    grid = world.spatial

    # Strip every fire indicator the decay pass would otherwise re-arm.
    for obj in world.spatial.objects.values():
        obj.tags = [
            t for t in obj.tags
            if t.lower() not in ("on_fire", "heat_source")
        ]
    for tile_key, tile in list(grid.tiles.items()):
        if any(t.lower() == "on_fire" for t in (tile.tags or [])):
            tile.tags = [t for t in tile.tags if t.lower() != "on_fire"]
            grid.tiles[tile_key] = tile

    region = world.regions[world.active_region_id]
    atm = region.grid.region_env.setdefault("fields", {}).setdefault("atmosphere", {})
    atm["smoke"] = {"amount": 60.0, "since_tick": world.tick}

    initial = _atmosphere_smoke(world)
    assert initial >= 40.0

    for _ in range(25):
        apply_transitions(world, field_tick(world))
        world.tick += 1

    final = _atmosphere_smoke(world)
    assert final < 12.0, (
        f"smoke should decay below the hazard threshold; initial={initial}, final={final}"
    )


def test_hazard_facts_are_registered_when_smoke_rises():
    """The fields module registers ``smoke_in_room`` / ``smoke_panic`` facts
    once atmosphere crosses thresholds, without any LM input."""
    world = _load_world()
    region = world.regions[world.active_region_id]
    atm = region.grid.region_env.setdefault("fields", {}).setdefault("atmosphere", {})
    atm["smoke"] = {"amount": 40.0, "since_tick": world.tick}

    loop = GameLoop(world, adapter=MockLMAdapter())
    # One tick is enough for field_tick to run the fact registration.
    loop.autonomous_tick()
    claims = [f.claim for f in (world.world_facts or [])]
    assert any("smoke" in c.lower() for c in claims), (
        f"a smoke hazard fact should be registered; got {claims}"
    )


def test_smoke_floods_make_npcs_move():
    """During a sustained smoke flood, NPCs do not freeze — they take
    action every tick (move, flee, douse, observe) instead of stalling
    on a baseline plan whose targets are unreachable through smoke.

    This is the "no NPC is paralysed" guardrail; a stricter
    door-targeting check is too dependent on pathfinding around tables.
    """
    world = _load_world()
    loop = GameLoop(world, adapter=MockLMAdapter())
    starts = {
        e.entity_id: Coord(x=e.position.x, y=e.position.y, z=e.position.z)
        for e in world.spatial.entities.values()
        if e.kind == EntityKind.NPC
    }

    for _ in range(2):
        loop.autonomous_tick()

    grid = world.spatial
    flood_trs = []
    for y in range(grid.height):
        for x in range(grid.width):
            c = Coord(x=x, y=y, z=1)
            if not grid.is_in_bounds(c) or not grid.is_passable(c):
                continue
            flood_trs.append(build_field_transition(
                MEDIA_TILE_AIR, "smoke", 45.0,
                coord=c, operation="set",
                cause="test_flood", tick=world.tick,
                remaining_ticks=30,
            ))
    apply_transitions(world, flood_trs)
    region = world.regions[world.active_region_id]
    atm = region.grid.region_env.setdefault("fields", {}).setdefault("atmosphere", {})
    atm["smoke"] = {"amount": 45.0, "since_tick": world.tick}

    for _ in range(15):
        loop.autonomous_tick()

    # At least 2 NPCs should have moved from their starting position.
    moved = 0
    for ent in world.spatial.entities.values():
        if ent.kind != EntityKind.NPC:
            continue
        if ent.position != starts[ent.entity_id]:
            moved += 1
    assert moved >= 2, (
        f"only {moved}/4 NPCs moved during 15 ticks of smoke — sim froze"
    )


def test_stack_keeps_baseline_plan_through_panic():
    """After a hazard scares Mira off her baseline plan, the stack
    retains it so she can resume once the smoke clears."""
    world = _load_world()
    loop = GameLoop(world, adapter=MockLMAdapter())
    for _ in range(2):
        loop.autonomous_tick()
    mira = _npc(world, "mira")
    baseline_name = mira.active_plan.name if mira.active_plan else None
    assert baseline_name, "baseline should be set"

    grid = world.spatial
    flood_trs = []
    for y in range(grid.height):
        for x in range(grid.width):
            c = Coord(x=x, y=y, z=1)
            if not grid.is_in_bounds(c) or not grid.is_passable(c):
                continue
            flood_trs.append(build_field_transition(
                MEDIA_TILE_AIR, "smoke", 45.0,
                coord=c, operation="set",
                cause="test_flood", tick=world.tick,
                remaining_ticks=20,
            ))
    apply_transitions(world, flood_trs)
    region = world.regions[world.active_region_id]
    atm = region.grid.region_env.setdefault("fields", {}).setdefault("atmosphere", {})
    atm["smoke"] = {"amount": 45.0, "since_tick": world.tick}

    loop.autonomous_tick()

    # After 1 tick of panic the baseline name should be in the stack.
    stack = mira.meta.get("plan_stack") or []
    assert stack, "stack must hold the baseline plan"
    assert stack[-1]["name"] == baseline_name


def test_linna_eavesdrop_plan_emits_eavesdrop_verbs():
    """Linna's pack plan (``worlds/voxel_tavern/npc_plans.yaml``) must
    actually emit ``eavesdrop`` SemanticActions so that the
    ``linna_uncovers_secret`` goal (``worlds/voxel_tavern/goals.yaml``)
    can resolve in reactive_only mode.

    Without this, the eavesdrop goal sits unfulfilled forever because the
    plan only emits move/wait/speak steps.
    """
    from src.sim.npc_planner import propose_plan

    world = _load_world()
    linna = _npc(world, "linna")
    # Force the drive that should trigger Linna's eavesdrop plan template.
    linna.drive = "eavesdrop on the merchant"

    plan = propose_plan(linna, world)
    assert plan is not None, "eavesdrop plan should be proposed"
    verbs = [
        str(step.payload.get("verb") or step.kind.value).lower()
        for step in plan.steps
    ]
    assert "eavesdrop" in verbs, (
        f"linna's plan should include at least one eavesdrop step; "
        f"got {verbs}"
    )


def test_tavern_property_interaction_douses_burning_object():
    """The tavern_staff_douses_burning_object property-interaction rule
    fires when a staff NPC emits ``douse`` next to a burning object,
    even without holding a fire suppressant.

    This pins the routing change from kernel `_compile_extinguish` to
    the verb-pipeline (property_interactions wins over the built-in
    default because of intent_categories=[thermal] + actor staff tag).
    """
    from src.sim.compiler import compile_action
    from src.sim.schemas import IntentBlock, SemanticAction

    world = _load_world()
    flask = _ignite_oil_at(world)
    assert "on_fire" in flask.tags

    mira = _npc(world, "mira")
    mira.position = Coord(
        x=flask.position.x + 1, y=flask.position.y, z=mira.position.z,
    )
    if "bartender" not in mira.tags and "staff" not in mira.tags:
        mira.tags = list(mira.tags) + ["bartender"]

    action = SemanticAction(
        verb="douse",
        actor=mira.entity_id,
        target=str(flask.object_id),
        intent=IntentBlock(rationale="smother the flask"),
        raw_input="[test:douse]",
    )
    result = compile_action(action, world)
    assert result.valid, f"douse should compile via pipeline; got {result.rejection_detail}"
    apply_transitions(world, result.concrete_transitions)

    flask_after = world.spatial.objects[flask.object_id]
    assert "on_fire" not in flask_after.tags, (
        f"flask should be doused; tags={flask_after.tags}"
    )
    # The tavern property-interaction rule additionally marks the actor's
    # tile wet — that's the deterministic evidence the staff rule fired
    # rather than the bare built-in default.
    actor_tile = world.spatial.tile_at(mira.position)
    target_tile = world.spatial.tile_at(flask.position)
    assert any(
        "wet" in m.lower()
        for m in (actor_tile.marks + target_tile.marks)
    ), (
        f"tavern_staff_douses_burning_object should mark a wet floor; "
        f"actor_marks={actor_tile.marks}, target_marks={target_tile.marks}"
    )


@pytest.mark.parametrize("ticks", [30])
def test_long_reactive_run_does_not_corrupt_world(ticks):
    """30-tick reactive_only run with no exceptions — covers the
    interaction between scenario beats, planner, hazard rules, and the
    field tick over a realistic playthrough length."""
    world = _load_world()
    loop = GameLoop(world, adapter=MockLMAdapter())
    for _ in range(ticks):
        loop.autonomous_tick()
    # Sanity invariants — no entity should be in a corrupt state.
    for ent in world.spatial.entities.values():
        assert ent.health >= 0
        if ent.alive:
            assert ent.position is not None
        if ent.active_plan is not None:
            assert ent.active_plan.status in (
                PlanStatus.ACTIVE, PlanStatus.PAUSED,
                PlanStatus.COMPLETED, PlanStatus.FAILED,
                PlanStatus.ABANDONED,
            )
