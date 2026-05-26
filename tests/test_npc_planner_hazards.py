"""
Hazard-priority plan synthesis and interrupts for the long-horizon planner.

These cover the new builders introduced for the voxel_tavern showcase
(flee_hazard, douse_fire, cook, serve, investigate) and the upgraded
``check_interrupt`` / ``try_resume`` that swap plans in and out of the
``entity.meta['plan_stack']`` when smoke / fire appear at an NPC's tile.

Reactive_only NPCs should now:
  - keep working their drive-derived plan while the room is safe,
  - drop everything when smoke fills the air,
  - synthesise a flee plan toward a safe tile (or a douse plan if the
    NPC is a bartender / cook / veteran / has QUENCH),
  - pop the original plan back onto active_plan once the hazard clears.
"""

from __future__ import annotations

from src.sim import npc_planner
from src.sim.compiler import apply_transitions
from src.sim.fields import build_field_transition, MEDIA_TILE_AIR, MEDIA_TILE_GROUND
from src.sim.npc_planner import (
    _build_cook_plan,
    _build_douse_fire_plan,
    _build_flee_hazard_plan,
    _build_investigate_plan,
    _build_serve_plan,
    _detect_hazard_reason,
    _synthesize_hazard_plan,
)
from src.sim.schemas import (
    AlertnessLevel,
    Coord,
    EntityKind,
    NpcPlan,
    PlanStatus,
    PlanStep,
    PlanStepKind,
    WorldFact,
    WorldFactScope,
)
from src.sim.world_loader import load_world_pack


def _entity(world, pack_id: str):
    for ent in world.spatial.entities.values():
        if world.meta["entity_pack_ids"].get(str(ent.entity_id)) == pack_id:
            return ent
    raise KeyError(pack_id)


def _add_smoke(world, coord: Coord, amount: float = 40.0):
    """Inject smoke into the air at *coord* and atmosphere both."""
    tr = build_field_transition(
        MEDIA_TILE_AIR, "smoke", amount,
        coord=coord, operation="set",
        cause="test", tick=world.tick, remaining_ticks=20,
    )
    apply_transitions(world, [tr])
    # Atmosphere mirror — drives region_atmosphere_amount() above threshold.
    region = world.regions[world.active_region_id]
    atm = region.grid.region_env.setdefault("fields", {}).setdefault("atmosphere", {})
    atm["smoke"] = {"amount": amount, "since_tick": world.tick}


# ── hazard detection ──────────────────────────────────────────────────────


def test_smoke_at_tile_detected_as_hazard_reason():
    world = load_world_pack("worlds/voxel_tavern")
    mira = _entity(world, "mira")
    _add_smoke(world, mira.position, 30.0)
    assert _detect_hazard_reason(mira, world) == "smoke"


def test_adjacent_fire_detected_as_hazard_reason():
    world = load_world_pack("worlds/voxel_tavern")
    mira = _entity(world, "mira")
    # Mark an adjacent object as on_fire.
    log = next(o for o in world.spatial.objects.values() if o.name == "burning log")
    # Move log next to Mira so we don't depend on the pack's hearth layout.
    log.position = Coord(x=mira.position.x + 1, y=mira.position.y, z=mira.position.z)
    log.tags = list(log.tags) + ["on_fire"]
    assert _detect_hazard_reason(mira, world) == "fire_near"


def test_clear_room_has_no_hazard_reason():
    world = load_world_pack("worlds/voxel_tavern")
    mira = _entity(world, "mira")
    # Make sure no log is currently on_fire.
    for obj in world.spatial.objects.values():
        if "on_fire" in obj.tags:
            obj.tags = [t for t in obj.tags if t != "on_fire"]
    assert _detect_hazard_reason(mira, world) == ""


# ── hazard plan synthesis ────────────────────────────────────────────────


def test_flee_plan_synthesised_for_civilian_in_smoke():
    world = load_world_pack("worlds/voxel_tavern")
    linna = _entity(world, "linna")  # bard — no douse capability
    _add_smoke(world, linna.position, 30.0)

    plan = _synthesize_hazard_plan(linna, world)
    assert plan is not None
    assert plan.meta["hazard_response"] is True
    assert plan.meta["hazard_kind"] in {"flee", "douse"}
    # Linna doesn't fight fires.
    assert plan.meta["hazard_kind"] == "flee"
    assert any(s.kind == PlanStepKind.MOVE_TO for s in plan.steps)


def test_douse_plan_synthesised_for_bartender_with_fire_nearby():
    world = load_world_pack("worlds/voxel_tavern")
    mira = _entity(world, "mira")
    log = next(o for o in world.spatial.objects.values() if o.name == "burning log")
    log.position = Coord(x=mira.position.x + 1, y=mira.position.y, z=mira.position.z)
    log.tags = list(log.tags) + ["on_fire"]

    plan = _synthesize_hazard_plan(mira, world)
    assert plan is not None
    assert plan.meta["hazard_kind"] == "douse"
    kinds = [s.kind for s in plan.steps]
    assert PlanStepKind.INTERACT in kinds
    assert PlanStepKind.MOVE_TO in kinds


def test_hazard_plan_returns_none_in_safe_room():
    world = load_world_pack("worlds/voxel_tavern")
    mira = _entity(world, "mira")
    for obj in world.spatial.objects.values():
        if "on_fire" in obj.tags:
            obj.tags = [t for t in obj.tags if t != "on_fire"]
    assert _synthesize_hazard_plan(mira, world) is None


# ── interrupt + stack ────────────────────────────────────────────────────


def test_smoke_interrupt_pushes_plan_onto_stack():
    world = load_world_pack("worlds/voxel_tavern")
    mira = _entity(world, "mira")
    mira.alertness = AlertnessLevel.LOW
    mira.active_plan = NpcPlan(
        owner_id=mira.entity_id,
        name="busy bar",
        steps=[PlanStep(kind=PlanStepKind.SPEND_TICKS, payload={"ticks": 5})],
        status=PlanStatus.ACTIVE,
    )
    _add_smoke(world, mira.position, 30.0)

    paused = npc_planner.check_interrupt(mira, world)
    assert paused is True
    assert mira.active_plan is None
    stack = mira.meta.get("plan_stack")
    assert isinstance(stack, list) and stack, "old plan should be stacked"
    assert stack[-1]["name"] == "busy bar"


def test_resume_pops_stacked_plan_once_room_clears():
    world = load_world_pack("worlds/voxel_tavern")
    mira = _entity(world, "mira")
    mira.alertness = AlertnessLevel.LOW
    mira.active_plan = NpcPlan(
        owner_id=mira.entity_id,
        name="busy bar",
        steps=[PlanStep(kind=PlanStepKind.SPEND_TICKS, payload={"ticks": 5})],
        status=PlanStatus.ACTIVE,
    )
    _add_smoke(world, mira.position, 30.0)
    npc_planner.check_interrupt(mira, world)
    assert mira.active_plan is None

    # Clear hazards: zero smoke + atmosphere + remove any fires.
    region = world.regions[world.active_region_id]
    region.grid.region_env.get("fields", {}).pop("atmosphere", None)
    tile = world.spatial.tile_at(mira.position)
    tile.env.pop("fields", None)
    tile.env.pop("airborne", None)
    for obj in world.spatial.objects.values():
        if "on_fire" in obj.tags:
            obj.tags = [t for t in obj.tags if t != "on_fire"]

    resumed = npc_planner.try_resume(mira, world)
    assert resumed is True
    assert mira.active_plan is not None
    assert mira.active_plan.name == "busy bar"
    assert mira.active_plan.status == PlanStatus.ACTIVE
    assert not (mira.meta.get("plan_stack") or [])


def test_hazard_response_plan_does_not_self_interrupt():
    """A flee plan in a smoky tile shouldn't immediately pause itself."""
    world = load_world_pack("worlds/voxel_tavern")
    linna = _entity(world, "linna")
    _add_smoke(world, linna.position, 30.0)

    plan = _synthesize_hazard_plan(linna, world)
    assert plan is not None
    linna.active_plan = plan
    paused = npc_planner.check_interrupt(linna, world)
    assert paused is False
    assert linna.active_plan is not None
    assert linna.active_plan.status == PlanStatus.ACTIVE


def test_hazard_plan_pre_empts_drive_plan_via_propose():
    world = load_world_pack("worlds/voxel_tavern")
    mira = _entity(world, "mira")
    mira.drive = "keep the ale flowing"
    mira.active_plan = None
    _add_smoke(world, mira.position, 30.0)

    plan = npc_planner.propose_plan(mira, world)
    assert plan is not None
    assert plan.meta.get("hazard_response") is True


# ── builder smoke tests ───────────────────────────────────────────────────


def test_cook_builder_targets_hearth():
    world = load_world_pack("worlds/voxel_tavern")
    mira = _entity(world, "mira")
    plan = _build_cook_plan(mira, world, drive="cook the stew")
    assert plan is not None
    assert plan.steps[0].kind == PlanStepKind.MOVE_TO
    assert any(s.kind == PlanStepKind.SPEND_TICKS for s in plan.steps)


def test_serve_builder_finds_bar():
    world = load_world_pack("worlds/voxel_tavern")
    mira = _entity(world, "mira")
    plan = _build_serve_plan(mira, world, drive="serve drinks")
    assert plan is not None
    assert plan.steps[0].kind == PlanStepKind.MOVE_TO


def test_flee_builder_picks_passable_tile():
    world = load_world_pack("worlds/voxel_tavern")
    linna = _entity(world, "linna")
    _add_smoke(world, linna.position, 30.0)
    plan = _build_flee_hazard_plan(linna, world, drive="get out alive")
    assert plan is not None
    move = plan.steps[0]
    assert move.kind == PlanStepKind.MOVE_TO
    x, y, z = move.payload["tile"]
    coord = Coord(x=x, y=y, z=z)
    assert world.spatial.is_passable(coord)


def test_douse_builder_aims_at_burning_object():
    world = load_world_pack("worlds/voxel_tavern")
    mira = _entity(world, "mira")
    log = next(o for o in world.spatial.objects.values() if o.name == "burning log")
    log.position = Coord(x=mira.position.x + 2, y=mira.position.y, z=mira.position.z)
    log.tags = list(log.tags) + ["on_fire"]
    plan = _build_douse_fire_plan(mira, world, drive="put out the fire")
    assert plan is not None
    kinds = [s.kind for s in plan.steps]
    assert PlanStepKind.MOVE_TO in kinds
    assert PlanStepKind.INTERACT in kinds


def test_investigate_builder_uses_recent_hazard_fact():
    world = load_world_pack("worlds/voxel_tavern")
    aldric = _entity(world, "aldric")
    fact_coord = Coord(x=aldric.position.x + 3, y=aldric.position.y, z=aldric.position.z)
    world.world_facts.append(
        WorldFact(
            claim="Strange noise from the back room.",
            scope=WorldFactScope.TILE,
            subject_id=f"{fact_coord.x},{fact_coord.y},{fact_coord.z}",
            established_tick=world.tick,
            established_by="system",
            tags=["mystery", "stranger"],
        )
    )
    plan = _build_investigate_plan(aldric, world, drive="investigate the noise")
    assert plan is not None
    move = plan.steps[0]
    assert move.kind == PlanStepKind.MOVE_TO
    x, y, _z = move.payload["tile"]
    assert (x, y) == (fact_coord.x, fact_coord.y)


# ── drive-pattern integration: drive text triggers right builder ─────────


def test_drive_pattern_routes_flee_keyword():
    world = load_world_pack("worlds/voxel_tavern")
    linna = _entity(world, "linna")
    linna.drive = "get out alive — forget the song"
    linna.active_plan = None
    plan = npc_planner.propose_plan(linna, world)
    assert plan is not None
    assert "flee" in plan.name.lower() or plan.meta.get("hazard_response")


def test_drive_pattern_routes_cook_keyword():
    world = load_world_pack("worlds/voxel_tavern")
    mira = _entity(world, "mira")
    mira.drive = "cook a fresh stew for the night"
    mira.active_plan = None
    # Clear pack templates to test the built-in pattern; pack templates
    # don't match "cook" alone in voxel_tavern's npc_plans.yaml.
    world.config.npc_plan_templates = []
    plan = npc_planner.propose_plan(mira, world)
    assert plan is not None
    assert plan.name == "cook"
