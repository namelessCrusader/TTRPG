"""Tests for the enriched player_inspect surfaces.

These tests pin the new "show, don't guess" reveals: NPC drive/plan,
tile fields, equipped gear with tags, recent world facts, and the
property-rule reaction preview.
"""

from __future__ import annotations

from src.sim.game_loop import make_test_world
from src.sim.player_inspect import format_inspect_report
from src.sim.schemas import (
    Coord,
    EntityKind,
    NpcPlan,
    ObjectState,
    PlanStatus,
    PlanStep,
    PlanStepKind,
    PropertyInteractionRule,
    TransitionProposal,
    WorldFact,
    WorldFactScope,
)


def _player_id(world):
    return next(
        eid for eid, e in world.spatial.entities.items()
        if e.kind == EntityKind.PLAYER
    )


def _npc(world):
    for eid, e in world.spatial.entities.items():
        if e.kind == EntityKind.NPC:
            return eid, e
    raise RuntimeError("no NPC")


# ── drive + plan surface ────────────────────────────────────────────────


def test_npc_drive_appears_in_target_inspect():
    world = make_test_world()
    pid = _player_id(world)
    _, npc = _npc(world)
    npc.drive = "patrol the eastern wall and report intruders"

    report = format_inspect_report(world, pid, npc.name)
    assert "drive" in report.lower()
    assert "patrol" in report.lower()


def test_npc_plan_progress_appears_in_target_inspect():
    world = make_test_world()
    pid = _player_id(world)
    _, npc = _npc(world)
    npc.active_plan = NpcPlan(
        owner_id=npc.entity_id,
        name="rebuild the workshop",
        drive_source="rebuild the workshop",
        steps=[
            PlanStep(kind=PlanStepKind.MOVE_TO, payload={"tile": [0, 0, 0]},
                     label="approach workshop"),
            PlanStep(kind=PlanStepKind.SPEND_TICKS,
                     payload={"ticks": 5, "label": "labour"},
                     label="labour"),
        ],
        current_step=1,
        status=PlanStatus.ACTIVE,
    )

    report = format_inspect_report(world, pid, npc.name)
    assert "plan:" in report.lower()
    assert "rebuild the workshop" in report.lower()
    assert "2/2" in report  # current_step is 1 → human-readable "2 of 2"


def test_room_view_shows_npc_drive_one_liner():
    world = make_test_world()
    pid = _player_id(world)
    _, npc = _npc(world)
    npc.drive = "tend the hearth"

    report = format_inspect_report(world, pid)
    assert "drive" in report.lower() or "plan" in report.lower()
    assert "tend the hearth" in report.lower()


# ── tile fields ──────────────────────────────────────────────────────────


def test_tile_field_substances_appear_in_room_view():
    world = make_test_world()
    pid = _player_id(world)
    player = world.spatial.entities[pid]
    tile = world.spatial.tile_at(player.position)
    # Inject a fluid spill the player should see when they inspect their tile.
    if tile is not None:
        tile.env["fluid_spill"] = {"material": "oil", "volume_ml": 200}

    report = format_inspect_report(world, pid)
    assert "Current tile" in report
    assert "oil" in report.lower()


# ── equipment + tag surfacing ────────────────────────────────────────────


def test_equipped_item_tags_appear_in_inspect():
    world = make_test_world()
    pid = _player_id(world)
    player = world.spatial.entities[pid]
    region_id = world.active_region_id or "region_main"
    oid = "obj_test_torch"
    world.spatial.objects[oid] = ObjectState(
        object_id=oid,
        name="oil torch",
        tags=["on_fire", "fire_source"],
        position=None,
        region_id=region_id,
    )
    player.inventory.append(oid)
    # Replace any existing main_hand so the new torch is the equipped item.
    player.equipped_weapon = oid
    player.equipped_slots["main_hand"] = oid

    report = format_inspect_report(world, pid, player.name)
    assert "equipped" in report.lower()
    assert "oil torch" in report.lower()
    assert "fire_source" in report.lower() or "on_fire" in report.lower()


# ── property-rule preview ────────────────────────────────────────────────


def test_property_rule_preview_lists_possible_reactions():
    world = make_test_world()
    pid = _player_id(world)
    nid, npc = _npc(world)
    player = world.spatial.entities[pid]

    # Place NPC adjacent so the resolver's distance gate is satisfied.
    world.spatial.entities[nid].position = Coord(
        x=player.position.x + 1, y=player.position.y, z=player.position.z,
    )
    # Player has a torch (fire_source), NPC is flammable.
    region_id = world.active_region_id or "region_main"
    oid = "obj_torch_review"
    world.spatial.objects[oid] = ObjectState(
        object_id=oid,
        name="torch",
        tags=["on_fire", "fire_source"],
        position=None,
        region_id=region_id,
    )
    player.inventory.append(oid)
    player.equipped_weapon = oid
    npc.tags = list(npc.tags or []) + ["flammable"]

    world.config.property_interactions = [PropertyInteractionRule(
        id="torch_ignites_flammable",
        name="torch ignites flammable",
        actor_or_tool_tags=["fire_source"],
        target_tags=["flammable"],
        intent_categories=["thermal", "any"],
        effects=[TransitionProposal(
            kind="entity_tag_changed",
            payload={"entity_id": "$target", "add": "on_fire"},
        )],
        priority=10,
    )]

    report = format_inspect_report(world, pid)
    assert "Possible reactions" in report
    # The matched rule's name plus the candidate target name should both appear.
    assert "torch ignites flammable" in report.lower() or "flammable" in report.lower()
    assert npc.name in report


# ── recent facts surface ─────────────────────────────────────────────────


def test_recent_facts_about_target_appear():
    world = make_test_world()
    pid = _player_id(world)
    _, npc = _npc(world)
    # Record a fact whose subject is this NPC.
    world.world_facts.append(WorldFact(
        claim=f"{npc.name} was seen tampering with the barrel",
        scope=WorldFactScope.ENTITY,
        subject_id=str(npc.entity_id),
        established_tick=world.tick,
        established_by=str(pid),
        source_intent="manual fact",
    ))
    report = format_inspect_report(world, pid, npc.name)
    assert "recent facts" in report.lower()
    assert "tampering" in report.lower()
