"""Contamination, biofluids, airborne pathogens, and illness."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.sim.compiler import apply_transitions
from src.sim.contamination import (
    AgentCatalog,
    apply_contamination_transition,
    biofluid_deposit_from_relief,
    bleed_at_entity,
    contamination_tick,
    cough_aerosol_transitions,
    load_agent_catalog,
)
from src.sim.schemas import Coord, Transition, TransitionKind, WorldState
from src.sim.world_loader import load_world_pack


@pytest.fixture
def world_with_agents() -> WorldState:
    pack = Path(__file__).resolve().parents[1] / "worlds" / "tavern"
    world = load_world_pack(pack)
    assert world.config.agent_catalog is not None
    assert world.config.agent_catalog.get("blood") is not None
    return world


def test_agent_catalog_loads():
    catalog = load_agent_catalog("worlds/default")
    assert catalog.get("urine") is not None
    assert catalog.get("cough_droplets") is not None
    assert catalog.get("common_flora").pathogen is True


def test_bleed_stains_tile(world_with_agents: WorldState):
    world = world_with_agents
    ent = next(iter(world.spatial.entities.values()))
    pos = ent.position
    transitions = bleed_at_entity(world, ent, damage=10, cause="test")
    apply_transitions(world, transitions)
    tile = world.spatial.tile_at(pos)
    cont = tile.env.get("contamination") or {}
    assert "blood" in cont
    assert float(cont["blood"]["concentration"]) > 10
    spill = tile.env.get("fluid_spill") or {}
    assert spill.get("material") == "blood" or tile.env.get("wet")


def test_relieve_deposits_urine(world_with_agents: WorldState):
    world = world_with_agents
    ent = next(iter(world.spatial.entities.values()))
    transitions = biofluid_deposit_from_relief(world, ent, privy=False)
    apply_transitions(world, transitions)
    tile = world.spatial.tile_at(ent.position)
    cont = tile.env.get("contamination") or {}
    assert "urine" in cont
    assert "common_flora" in cont


def test_cough_airborne(world_with_agents: WorldState):
    world = world_with_agents
    ent = next(iter(world.spatial.entities.values()))
    transitions = cough_aerosol_transitions(world, ent)
    apply_transitions(world, transitions)
    tile = world.spatial.tile_at(ent.position)
    air = tile.env.get("airborne") or {}
    assert "cough_droplets" in air
    assert int(air["cough_droplets"]["remaining_ticks"]) > 0


def test_contamination_decay(world_with_agents: WorldState):
    world = world_with_agents
    coord = Coord(x=1, y=1)
    apply_transitions(world, [
        Transition(
            kind=TransitionKind.CONTAMINATION_CHANGED,
            payload={
                "target_kind": "tile",
                "x": coord.x,
                "y": coord.y,
                "agent": "blood",
                "operation": "deposit",
                "concentration": 50.0,
                "tick": world.tick,
            },
        ),
    ])
    world.tick = 1
    transitions = contamination_tick(world)
    apply_transitions(world, transitions)
    tile = world.spatial.tile_at(coord)
    cont = tile.env.get("contamination") or {}
    if "blood" in cont:
        assert float(cont["blood"]["concentration"]) < 50.0


def test_rain_wets_mud(world_with_agents: WorldState):
    world = world_with_agents
    world.meta["weather"] = "rain"
    coord = Coord(x=2, y=2)
    tile = world.spatial.tile_at(coord)
    tile.tags = list(tile.tags) + ["mud"]
    world.spatial.set_tile(coord, tile)
    world.tick = 2
    transitions = contamination_tick(world)
    apply_transitions(world, transitions)
    tile = world.spatial.tile_at(coord)
    cont = tile.env.get("contamination") or {}
    assert "rainwater" in cont or tile.env.get("wet")


def test_infection_from_exposure(world_with_agents: WorldState):
    world = world_with_agents
    ent = next(iter(world.spatial.entities.values()))
    coord = ent.position
    apply_transitions(world, [
        Transition(
            kind=TransitionKind.CONTAMINATION_CHANGED,
            payload={
                "target_kind": "air",
                "x": coord.x,
                "y": coord.y,
                "agent": "cough_droplets",
                "operation": "deposit",
                "concentration": 80.0,
                "remaining_ticks": 12,
                "tick": world.tick,
            },
        ),
    ])
    world.rng_seed = "infect_test_seed"
    for _ in range(30):
        world.tick += 1
        tr = contamination_tick(world)
        apply_transitions(world, tr)
        if ent.meta.get("infection"):
            break
    assert ent.meta.get("infection") or ent.meta.get("exposure")
