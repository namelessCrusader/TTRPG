"""Player spawn placement for worlds without a player entity."""

from src.sim.game_loop import _find_player_spawn, _spawn_interactive_player
from src.sim.schemas import Coord, EntityKind
from src.sim.world_loader import load_world_pack


def test_voxel_tavern_spawn_is_walkable_and_not_occupied():
    world = load_world_pack("worlds/voxel_tavern")
    spawn = _find_player_spawn(world)
    grid = world.spatial
    assert not grid.is_solid(spawn)
    occupied = {
        (e.position.x, e.position.y, e.position.z)
        for e in grid.entities.values()
        if e.alive
    }
    assert (spawn.x, spawn.y, spawn.z) not in occupied
    open_n = sum(
        1
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
        if not grid.is_solid(Coord(x=spawn.x + dx, y=spawn.y + dy, z=spawn.z))
    )
    assert open_n >= 2


def test_spawn_interactive_player_can_move_north():
    world = load_world_pack("worlds/voxel_tavern")
    pid = _spawn_interactive_player(world)
    p0 = world.spatial.entities[pid].position
    from src.sim.game_loop import GameLoop
    from src.sim.lm_adapter import MockLMAdapter
    from src.sim.play_commands import run_player_turn

    loop = GameLoop(world, adapter=MockLMAdapter(), player_id=pid)
    run_player_turn("move north", loop, pid, lambda *_a, **_k: None)
    p1 = world.spatial.entities[pid].position
    assert p1 != p0
