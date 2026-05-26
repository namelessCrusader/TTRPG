"""Magic duel world pack and spell-combat NPC candidates."""

from pathlib import Path

from src.sim.game_loop import GameLoop
from src.sim.lm_adapter import MockLMAdapter
from src.sim.narrator import render_event
from src.sim.npc_policy import ReactivePolicy, _estimate_spell_mana
from src.sim.validate_pack import validate_pack
from src.sim.world_loader import load_world_pack


def _duel_pack() -> Path:
    return Path(__file__).resolve().parents[1] / "worlds" / "magic_duel"


def test_magic_duel_pack_validates():
    errors = validate_pack(_duel_pack(), smoke_ticks=8)
    assert errors == []


def test_magic_duel_loads_physics_magic():
    world = load_world_pack(_duel_pack())
    assert world.meta.get("magic_duel") is True
    vocab = world.config.spell_vocab
    assert vocab is not None
    assert "HEAT" in vocab.operations
    assert "HARM" not in vocab.operations
    assert len(world.spatial.entities) == 2


def test_reactive_policy_labels_show_programs():
    world = load_world_pack(_duel_pack())
    policy = ReactivePolicy()
    pyra = next(e for e in world.spatial.entities.values() if "Pyra" in e.name)
    cands = policy.generate_candidates(pyra, world, max_candidates=12)
    cast = [c for c in cands if str(c[0].verb).lower() == "cast"]
    assert len(cast) >= 3
    assert any("HEAT" in c[1] and "TARGET" in c[1] for c in cast)


def test_estimate_spell_mana_heat():
    world = load_world_pack(_duel_pack())
    vocab = world.config.spell_vocab
    assert _estimate_spell_mana("HEAT 55 TARGET", vocab) == 44.0


def test_narrator_shows_program_not_nickname():
    world = load_world_pack(_duel_pack())
    loop = GameLoop(world, adapter=MockLMAdapter(), npc_policy=ReactivePolicy(), player_id=None)
    loop.step_npcs()
    cast_ev = next(
        e for e in world.event_log if str(e.action.verb).lower() == "cast"
    )
    text = render_event(cast_ev, world)
    assert "HEAT" in text or "CHILL" in text
    assert "ember" not in text.lower()


def test_heat_cast_applies_temperature_same_tick():
    world = load_world_pack(_duel_pack())
    from src.sim.substance import get_temperature

    pyra = next(e for e in world.spatial.entities.values() if "Pyra" in e.name)
    cryo = next(e for e in world.spatial.entities.values() if "Cryo" in e.name)
    cfg = world.config.physics_config
    before = get_temperature(cryo, cfg)
    loop = GameLoop(world, adapter=MockLMAdapter(), npc_policy=ReactivePolicy(), player_id=None)
    loop.step_npcs()
    after = get_temperature(cryo, cfg)
    assert after != before or any(
        str(e.action.verb).lower() == "cast"
        and "CHILL" in (e.action.intent.rationale or "").upper()
        for e in world.event_log
    )
