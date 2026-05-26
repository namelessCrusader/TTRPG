"""Scenario executor scripted physical effects."""

from pathlib import Path

from src.sim.compiler import apply_transitions
from src.sim.fields import get_layer, MEDIA_TILE_GROUND
from src.sim.scenario_executor import apply_effect_bundle
from src.sim.world_loader import load_world_pack


def test_effect_bundle_field_deposit_and_fact():
    world = load_world_pack(Path(__file__).resolve().parents[1] / "worlds" / "voxel_tavern")
    flask = next(o for o in world.spatial.objects.values() if "lamp oil" in o.name.lower())
    assert flask.position is not None

    lines, transitions = apply_effect_bundle(
        world,
        {
            "field_deposits": [
                {"object": "lamp_oil_flask", "substance": "oil", "amount": 40.0},
            ],
            "register_facts": [
                {"claim": "Oil pools on the stones.", "tags": ["oil", "spill"]},
            ],
        },
        cause="test",
    )
    apply_transitions(world, transitions)

    tile = world.spatial.tile_at(flask.position)
    layer = get_layer(tile.env, MEDIA_TILE_GROUND, legacy_tile=tile)
    assert float((layer.get("oil") or {}).get("amount", 0)) >= 30.0
    assert any("oil" in f.claim.lower() for f in world.world_facts)
    assert any("field deposit" in ln for ln in lines)
