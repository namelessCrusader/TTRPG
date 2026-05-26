"""
Tests for M3: Projection layer.

Acceptance criteria from the spec:
  - Projection size is bounded by max_projection_tokens
  - Projection is deterministic: same inputs → same bytes
  - Projection contains no raw grid coordinates (only topological descriptors)
  - Projection for entity A never includes facts A has no causal path to know
  - Focal entity does not appear in its own visible_entities list
"""

import pytest

from src.sim.game_loop import make_test_world
from src.sim.projection import (
    MAX_PROJECTION_TOKENS,
    assert_projection_invariants,
    project,
)
from src.sim.schemas import (
    Coord,
    EdgeKind,
    EmotionalState,
    EntityId,
    EntityKind,
    EntityState,
    new_entity_id,
)
from src.sim.spatial import place_entity


def _player_id(world) -> EntityId:
    for eid, e in world.spatial.entities.items():
        if e.kind == EntityKind.PLAYER:
            return eid
    raise RuntimeError("No player in world")


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_projection_is_deterministic():
    world = make_test_world()
    pid = _player_id(world)
    p1 = project(world, pid)
    p2 = project(world, pid)
    assert p1.model_dump_json() == p2.model_dump_json()


# ---------------------------------------------------------------------------
# Token budget
# ---------------------------------------------------------------------------


def test_projection_within_token_budget():
    world = make_test_world()
    pid = _player_id(world)
    proj = project(world, pid, max_tokens=MAX_PROJECTION_TOKENS)
    assert proj.estimated_tokens <= MAX_PROJECTION_TOKENS


def test_projection_respects_custom_token_budget():
    world = make_test_world()
    pid = _player_id(world)
    # Budget of 215 tokens; the `facing` field added ~8 chars to every projection.
    proj = project(world, pid, max_tokens=215)
    assert proj.estimated_tokens <= 215


# ---------------------------------------------------------------------------
# No raw coordinates in entity summaries
# ---------------------------------------------------------------------------


def test_entity_summaries_have_no_position_field():
    world = make_test_world()
    pid = _player_id(world)
    proj = project(world, pid)
    for summary in proj.visible_entities:
        assert not hasattr(summary, "position"), (
            "EntitySummary must not expose raw grid position"
        )
        # relative_position should be a string descriptor
        assert isinstance(summary.relative_position, str)
        assert len(summary.relative_position) > 0


# ---------------------------------------------------------------------------
# Partial observability
# ---------------------------------------------------------------------------


def test_entity_behind_wall_not_in_projection():
    """An entity separated by a solid wall must not appear in the projection."""
    from src.sim.schemas import TerrainType, Tile
    world = make_test_world()
    pid = _player_id(world)

    # Build a solid wall down the middle of the room (x=10)
    for y in range(1, 19):
        world.spatial.set_tile(Coord(x=10, y=y), Tile(terrain=TerrainType.WALL))

    # Place a new entity on the far side
    hidden_id = new_entity_id()
    hidden = EntityState(
        entity_id=hidden_id,
        name="Hidden",
        kind=EntityKind.NPC,
        position=Coord(x=15, y=7),
        sight_range=8,
    )
    place_entity(world.spatial, hidden)

    proj = project(world, pid)
    visible_ids = {s.entity_id for s in proj.visible_entities}
    assert hidden_id not in visible_ids


def test_two_entities_receive_different_projections():
    """Partial observability: player and guard see different subsets."""
    world = make_test_world()
    pid = _player_id(world)
    # Find guard
    guard_id = next(
        eid for eid, e in world.spatial.entities.items()
        if e.kind == EntityKind.NPC
    )
    proj_player = project(world, pid)
    proj_guard = project(world, guard_id)
    # The focal entities must differ
    assert proj_player.focal_entity != proj_guard.focal_entity
    # Player's projection should not contain player; guard's should not contain guard
    player_visible_ids = {s.entity_id for s in proj_player.visible_entities}
    guard_visible_ids = {s.entity_id for s in proj_guard.visible_entities}
    assert pid not in player_visible_ids
    assert guard_id not in guard_visible_ids


# ---------------------------------------------------------------------------
# Invariant assertions
# ---------------------------------------------------------------------------


def test_assert_projection_invariants_passes():
    world = make_test_world()
    pid = _player_id(world)
    proj = project(world, pid)
    # Should not raise
    assert_projection_invariants(proj, world, pid)


def test_focal_entity_absent_from_visible_entities():
    world = make_test_world()
    pid = _player_id(world)
    proj = project(world, pid)
    ids = {s.entity_id for s in proj.visible_entities}
    assert pid not in ids
