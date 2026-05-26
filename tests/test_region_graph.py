"""
Tests for the multi-region world graph (Direction D).

Covers:
  - WorldState.regions / .portals / .spatial property
  - Portal detection in _compile_move → REGION_TRANSIT
  - apply_transitions(REGION_TRANSIT) moves entity between region grids
  - active_region_id updates when player crosses portal
  - Bidirectional portal reverse lookup
  - Keyed portal blocks entry without required item
  - All existing worlds still load cleanly (regression)
"""

import pytest

from src.sim.compiler import apply_transitions, compile_action
from src.sim.schemas import (
    Coord,
    EntityKind,
    Portal,
    Region,
    RegionType,
    SemanticAction,
    SpatialGrid,
    Tile,
    TerrainType,
    TransitionKind,
    WorldState,
)
from src.sim.world_loader import load_world_pack
from src.sim.game_loop import make_test_world
from src.sim.spatial import place_entity
from src.sim.schemas import EntityState, EntityId, new_entity_id


# ---------------------------------------------------------------------------
# Helpers: minimal two-region world
# ---------------------------------------------------------------------------

def _floor_grid(w: int, h: int) -> SpatialGrid:
    g = SpatialGrid(width=w, height=h)
    for y in range(h):
        for x in range(w):
            if x == 0 or x == w - 1 or y == 0 or y == h - 1:
                g.set_tile(Coord(x=x, y=y), Tile(terrain=TerrainType.WALL))
    return g


def _make_two_region_world() -> tuple[WorldState, EntityId, EntityId]:
    """
    Two 10×10 regions (alpha, beta) connected by a portal at (5,8) → (5,1).
    Returns (world, player_id, npc_id).
    npc_id is placed in region_beta.
    """
    alpha = _floor_grid(10, 10)
    beta  = _floor_grid(10, 10)

    player_id = new_entity_id()
    player = EntityState(
        entity_id=player_id, name="Player",
        kind=EntityKind.PLAYER, position=Coord(x=5, y=5), region_id="region_alpha",
    )
    place_entity(alpha, player)

    npc_id = new_entity_id()
    npc = EntityState(
        entity_id=npc_id, name="NPC",
        kind=EntityKind.NPC, position=Coord(x=5, y=5), region_id="region_beta",
    )
    place_entity(beta, npc)

    world = WorldState(
        regions={
            "region_alpha": Region(region_id="region_alpha", name="Alpha",
                                   region_type=RegionType.INDOOR, grid=alpha),
            "region_beta":  Region(region_id="region_beta",  name="Beta",
                                   region_type=RegionType.INDOOR, grid=beta),
        },
        portals=[
            Portal(
                portal_id="p1",
                label="Gate",
                from_region="region_alpha",
                from_coord=Coord(x=5, y=8),
                to_region="region_beta",
                to_coord=Coord(x=5, y=1),
                bidirectional=True,
            )
        ],
        active_region_id="region_alpha",
    )
    return world, player_id, npc_id


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestWorldStateSpatialProperty:
    def test_spatial_returns_active_region_grid(self):
        world, pid, _ = _make_two_region_world()
        assert world.spatial is world.regions["region_alpha"].grid

    def test_spatial_updates_after_active_region_changes(self):
        world, pid, _ = _make_two_region_world()
        world.active_region_id = "region_beta"
        assert world.spatial is world.regions["region_beta"].grid


class TestPortalLookup:
    def test_forward_portal_found(self):
        world, _, _ = _make_two_region_world()
        p = world.portal_at("region_alpha", Coord(x=5, y=8))
        assert p is not None
        assert p.to_region == "region_beta"
        assert p.to_coord == Coord(x=5, y=1)

    def test_reverse_portal_found_for_bidirectional(self):
        world, _, _ = _make_two_region_world()
        p = world.portal_at("region_beta", Coord(x=5, y=1))
        assert p is not None
        assert p.to_region == "region_alpha"
        assert p.to_coord == Coord(x=5, y=8)

    def test_no_portal_on_random_tile(self):
        world, _, _ = _make_two_region_world()
        p = world.portal_at("region_alpha", Coord(x=3, y=3))
        assert p is None


class TestPortalTraversal:
    def test_move_onto_portal_tile_generates_region_transit(self):
        world, pid, _ = _make_two_region_world()
        player = world.spatial.entities[pid]
        player.position = Coord(x=5, y=7)  # one step before portal

        action = SemanticAction(verb="move", actor=pid, target=Coord(x=5, y=8))
        result = compile_action(action, world)

        assert result.valid
        transit = [
            t for t in result.concrete_transitions
            if t.kind == TransitionKind.REGION_TRANSIT
        ]
        assert len(transit) == 1
        t = transit[0]
        assert t.kind == TransitionKind.REGION_TRANSIT
        assert t.payload["from_region"] == "region_alpha"
        assert t.payload["to_region"] == "region_beta"
        assert t.payload["arrival"] == {"x": 5, "y": 1}

    def test_apply_region_transit_moves_entity(self):
        world, pid, _ = _make_two_region_world()
        player = world.spatial.entities[pid]
        player.position = Coord(x=5, y=7)

        action = SemanticAction(verb="move", actor=pid, target=Coord(x=5, y=8))
        result = compile_action(action, world)
        assert result.valid

        apply_transitions(world, result.concrete_transitions)

        alpha_grid = world.regions["region_alpha"].grid
        beta_grid  = world.regions["region_beta"].grid

        assert pid not in alpha_grid.entities, "Player should leave alpha"
        assert pid in beta_grid.entities, "Player should arrive in beta"

    def test_active_region_updates_after_player_transit(self):
        world, pid, _ = _make_two_region_world()
        player = world.spatial.entities[pid]
        player.position = Coord(x=5, y=7)

        action = SemanticAction(verb="move", actor=pid, target=Coord(x=5, y=8))
        result = compile_action(action, world)
        apply_transitions(world, result.concrete_transitions)

        assert world.active_region_id == "region_beta"

    def test_player_arrives_at_correct_coord(self):
        world, pid, _ = _make_two_region_world()
        player = world.spatial.entities[pid]
        player.position = Coord(x=5, y=7)

        action = SemanticAction(verb="move", actor=pid, target=Coord(x=5, y=8))
        result = compile_action(action, world)
        apply_transitions(world, result.concrete_transitions)

        arrived = world.regions["region_beta"].grid.entities[pid]
        assert arrived.position == Coord(x=5, y=1)

    def test_entity_region_id_updated(self):
        world, pid, _ = _make_two_region_world()
        player = world.spatial.entities[pid]
        player.position = Coord(x=5, y=7)

        action = SemanticAction(verb="move", actor=pid, target=Coord(x=5, y=8))
        result = compile_action(action, world)
        apply_transitions(world, result.concrete_transitions)

        arrived = world.regions["region_beta"].grid.entities[pid]
        assert arrived.region_id == "region_beta"

    def test_npc_transit_does_not_change_active_region(self):
        """Only player transits update active_region_id."""
        world, pid, npc_id = _make_two_region_world()
        # Move NPC from beta toward reverse portal
        npc = world.regions["region_beta"].grid.entities[npc_id]
        npc.position = Coord(x=5, y=2)

        action = SemanticAction(verb="move", actor=npc_id, target=Coord(x=5, y=1))
        result = compile_action(action, world)
        assert result.valid
        assert npc_id in world.regions["region_beta"].grid.entities
        assert world.active_region_id == "region_alpha"


class TestKeyedPortal:
    def test_keyed_portal_blocks_without_key(self):
        world, pid, _ = _make_two_region_world()
        # Add key requirement to the portal
        from src.sim.schemas import ObjectId
        world.portals[0].requires_key = ObjectId("fake_key_id")

        player = world.spatial.entities[pid]
        player.position = Coord(x=5, y=7)

        action = SemanticAction(verb="move", actor=pid, target=Coord(x=5, y=8))
        result = compile_action(action, world)

        assert not result.valid
        assert "locked" in (result.rejection_detail or "").lower()


class TestExistingWorldsStillLoad:
    @pytest.mark.parametrize("world_name", ["default", "spicy", "tavern", "castle"])
    def test_world_loads_with_regions(self, world_name):
        world = load_world_pack(f"worlds/{world_name}")
        assert len(world.regions) >= 1
        assert world.active_region_id in world.regions
        assert world.spatial is not None  # backward-compat property works

    def test_single_region_worlds_wrap_in_region_main(self):
        world = load_world_pack("worlds/default")
        assert "region_main" in world.regions
        assert world.spatial is world.regions["region_main"].grid

    def test_castle_has_three_regions(self):
        world = load_world_pack("worlds/castle")
        assert "region_main" in world.regions
        assert "region_great_hall" in world.regions
        assert "region_dungeon" in world.regions

    def test_castle_portals_resolve(self):
        world = load_world_pack("worlds/castle")
        p = world.portal_at("region_main", Coord(x=9, y=14))
        assert p is not None
        assert p.to_region == "region_great_hall"
