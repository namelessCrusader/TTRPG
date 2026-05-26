"""Tests for default pack merge and multi-region helpers."""

from __future__ import annotations

from src.sim.region_utils import scoped_tile_key, witness_entity_ids
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
    WorldFact,
    WorldFactScope,
    WorldState,
    new_entity_id,
)
from src.sim.spatial import place_entity
from src.sim.world_adjudicator import register_world_fact
from src.sim.world_loader import load_world_pack


def test_default_open_verbs_merge_into_tavern():
    world = load_world_pack("worlds/tavern")
    templates = world.config.verb_templates
    assert "console" in templates
    assert "disguise" in templates
    open_ids = {r.id for r in world.config.open_verbs}
    assert "default_plant" in open_ids
    plant = next(r for r in world.config.open_verbs if r.id == "default_plant")
    assert "stash" in (plant.verbs or [])


def test_default_reactive_triggers_merge():
    world = load_world_pack("worlds/spicy")
    ids = {t.get("id") for t in world.config.reactive_triggers}
    assert "generic_violence_escalation" in ids
    assert "chamber_intimacy" in ids


def test_castle_has_pack_reactive_triggers():
    world = load_world_pack("worlds/castle")
    ids = {t.get("id") for t in world.config.reactive_triggers}
    assert "courtyard_intruder" in ids


def test_witnesses_stay_in_actor_region():
    def _floor(w, h):
        g = SpatialGrid(width=w, height=h)
        for y in range(h):
            for x in range(w):
                if x == 0 or x == w - 1 or y == 0 or y == h - 1:
                    g.set_tile(Coord(x=x, y=y), Tile(terrain=TerrainType.WALL))
        return g

    alpha = _floor(10, 10)
    beta = _floor(10, 10)
    pid = new_entity_id()
    nid = new_entity_id()
    place_entity(alpha, EntityState(
        entity_id=pid, name="Player", kind=EntityKind.PLAYER,
        position=Coord(x=5, y=5), region_id="region_alpha",
    ))
    place_entity(beta, EntityState(
        entity_id=nid, name="NPC", kind=EntityKind.NPC,
        position=Coord(x=5, y=5), region_id="region_beta",
    ))
    world = WorldState(
        regions={
            "region_alpha": Region(region_id="region_alpha", name="A",
                                   region_type=RegionType.INDOOR, grid=alpha),
            "region_beta": Region(region_id="region_beta", name="B",
                                  region_type=RegionType.INDOOR, grid=beta),
        },
        portals=[],
        active_region_id="region_alpha",
    )
    witnesses = witness_entity_ids(world, pid)
    assert pid in witnesses
    assert nid not in witnesses


def test_scoped_tile_fact_reaches_projection():
    from src.sim.game_loop import make_test_world
    from src.sim.memory_retrieval import ranked_facts_for_projection

    world = make_test_world()
    player_id = next(
        eid for eid, e in world.spatial.entities.items()
        if e.kind == EntityKind.PLAYER
    )
    player = world.spatial.entities[player_id]
    key = scoped_tile_key(player.region_id, player.position)
    register_world_fact(
        world,
        WorldFact(
            claim="Hidden stash under the floorboard.",
            scope=WorldFactScope.TILE,
            subject_id=key,
            established_tick=world.tick,
            tags=["environment"],
        ),
    )
    facts = ranked_facts_for_projection(world, player_id)
    assert any("stash" in f.lower() for f in facts)
