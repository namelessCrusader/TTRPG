"""Magic duel: temperature clamping, damage, no pre-history stacking."""

from pathlib import Path

from src.sim.game_loop import GameLoop
from src.sim.lm_adapter import MockLMAdapter
from src.sim.npc_policy import ReactivePolicy
from src.sim.substance import apply_temperature_delta, clamp_temperature, get_temperature
from src.sim.world_loader import load_world_pack


def _duel():
    return load_world_pack(Path(__file__).resolve().parents[1] / "worlds" / "magic_duel")


def test_temperature_clamps():
    world = _duel()
    ent = next(iter(world.spatial.entities.values()))
    cfg = world.config.physics_config
    for _ in range(30):
        apply_temperature_delta(ent, -45.0, cfg)
    assert get_temperature(ent, cfg) >= -40.0


def test_cold_deals_damage_after_spell():
    world = _duel()
    pyra = next(e for e in world.spatial.entities.values() if "Pyra" in e.name)
    cryo = next(e for e in world.spatial.entities.values() if "Cryo" in e.name)
    cfg = world.config.physics_config
    apply_temperature_delta(pyra, -50.0, cfg)
    loop = GameLoop(world, adapter=MockLMAdapter(), npc_policy=ReactivePolicy(), player_id=None)
    loop.autonomous_tick()
    assert pyra.health < 100


def test_magic_duel_both_cast():
    world = _duel()
    loop = GameLoop(world, adapter=MockLMAdapter(), npc_policy=ReactivePolicy(), player_id=None)
    loop.autonomous_tick()
    casts = [e for e in world.event_log if str(e.action.verb).lower() == "cast"]
    assert len(casts) >= 2
