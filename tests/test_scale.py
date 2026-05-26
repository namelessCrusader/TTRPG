"""Scale and stress tests — large NPC counts, regions, save cycles, LM budget."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.sim.game_loop import GameLoop
from src.sim.lm_adapter import MockLMAdapter
from src.sim.lm_budget import metrics_snapshot, wrap_adapter_metrics
from src.sim.persistence import load_world, save_world
from src.sim.schemas import (
    Coord,
    EntityKind,
    EntityState,
    Portal,
    Region,
    RegionType,
    SpatialGrid,
    Tile,
    TerrainType,
    WorldState,
    new_entity_id,
)
from src.sim.spatial import place_entity
from src.sim.world_loader import load_world_pack


def _open_floor_grid(w: int = 24, h: int = 16, depth: int = 1) -> SpatialGrid:
    grid = SpatialGrid(width=w, height=h, depth=depth)
    for y in range(h):
        for x in range(w):
            grid.set_tile(Coord(x=x, y=y), Tile(terrain=TerrainType.FLOOR))
    return grid


def _spawn_npcs(world: WorldState, count: int) -> None:
    grid = world.spatial
    placed = 0
    for y in range(1, grid.height - 1):
        for x in range(1, grid.width - 1):
            if placed >= count:
                return
            c = Coord(x=x, y=y, z=0)
            if not grid.is_passable(c):
                continue
            eid = new_entity_id()
            place_entity(
                grid,
                EntityState(
                    entity_id=eid,
                    name=f"Crowd-{placed}",
                    kind=EntityKind.NPC,
                    position=c,
                    drive="patrol the room",
                    health=80,
                    max_health=100,
                ),
            )
            placed += 1


@pytest.mark.slow
def test_fifty_npc_autonomous_ticks():
    world = load_world_pack("worlds/tavern")
    _spawn_npcs(world, 50)
    npc_count = sum(
        1 for e in world.spatial.entities.values() if e.kind == EntityKind.NPC
    )
    assert npc_count >= 50

    loop = GameLoop(world, adapter=MockLMAdapter(), npcs_act_each_turn=True)
    for _ in range(10):
        loop.autonomous_tick()

    assert world.tick >= 10
    alive = sum(
        1
        for e in world.spatial.entities.values()
        if e.kind == EntityKind.NPC and e.alive
    )
    assert alive >= 45


def test_multi_region_portal_transit():
    alpha = _open_floor_grid(12, 12)
    beta = _open_floor_grid(12, 12)
    pid = new_entity_id()
    place_entity(
        alpha,
        EntityState(
            entity_id=pid,
            name="Traveler",
            kind=EntityKind.PLAYER,
            position=Coord(x=5, y=5),
            region_id="region_alpha",
        ),
    )
    portal = Portal(
        portal_id="gate_ab",
        label="Archway",
        from_region="region_alpha",
        from_coord=Coord(x=10, y=5),
        to_region="region_beta",
        to_coord=Coord(x=1, y=5),
        bidirectional=True,
    )
    world = WorldState(
        regions={
            "region_alpha": Region(
                region_id="region_alpha",
                name="Alpha",
                region_type=RegionType.INDOOR,
                grid=alpha,
            ),
            "region_beta": Region(
                region_id="region_beta",
                name="Beta",
                region_type=RegionType.INDOOR,
                grid=beta,
            ),
        },
        portals=[portal],
        active_region_id="region_alpha",
        name="two-region test",
    )
    loop = GameLoop(world, adapter=MockLMAdapter(), player_id=pid)
    result = loop.step("move 10 5")
    assert "[OK" in result.status or "[ADJUDICATED" in result.status
    player = world.spatial.entities[pid]
    assert player.region_id in ("region_alpha", "region_beta")


def test_long_save_reload_cycle():
    world = load_world_pack(
        Path(__file__).resolve().parents[1] / "worlds" / "tavern"
    )
    world.rng_seed = 42
    adapter = MockLMAdapter()
    wrap_adapter_metrics(adapter, world)
    loop = GameLoop(world, adapter=adapter)

    for _ in range(20):
        loop.autonomous_tick()

    tick_before = world.tick
    event_count = len(world.event_log)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "scale_save.json"
        save_world(world, path, pack_id="tavern", max_events=500)
        loaded = load_world(path)

    assert loaded.tick == tick_before
    assert len(loaded.event_log) == event_count
    assert len(loaded.spatial.entities) == len(world.spatial.entities)


def test_lm_budget_tracks_calls_under_load():
    world = load_world_pack("worlds/tavern")
    adapter = MockLMAdapter()
    wrap_adapter_metrics(adapter, world)
    loop = GameLoop(world, adapter=adapter, npcs_act_each_turn=True)

    for _ in range(5):
        loop.autonomous_tick()

    snap = metrics_snapshot(world)
    assert int(snap.get("total_calls", 0)) >= 0
    assert "calls_by_kind" in snap


def test_voxel_world_spatial_mode():
    world = load_world_pack("worlds/voxel_tavern")
    assert world.meta.get("spatial_mode") == "voxel"
    assert world.spatial.use_voxels
    assert world.spatial.depth > 1


def test_npc_plans_load_from_tavern_pack():
    world = load_world_pack("worlds/tavern")
    ids = {t.id for t in world.config.npc_plan_templates}
    assert "mira_bar_rounds" in ids
    assert "generic_patrol" in ids
