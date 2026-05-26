"""Pack-authored relational aims surfaced to infer_npc."""

from pathlib import Path

from src.sim.npc_policy import build_npc_character_sheet, format_social_aims_for_sheet
from src.sim.world_loader import load_world_pack


def test_tavern_entities_load_social_aims():
    world = load_world_pack(Path(__file__).resolve().parents[1] / "worlds" / "tavern")
    mira = world.spatial.entities[
        next(eid for eid, e in world.spatial.entities.items() if e.name == "Mira")
    ]
    assert len(mira.social_aims) >= 2
    targets = {sa.target_pack_id for sa in mira.social_aims}
    assert "tomas" in targets


def test_social_aims_on_character_sheet_include_target_ids():
    world = load_world_pack(Path(__file__).resolve().parents[1] / "worlds" / "tavern")
    linna = next(e for e in world.spatial.entities.values() if e.name == "Linna")
    lines = format_social_aims_for_sheet(linna, world)
    assert any("Tomas" in ln for ln in lines)
    sheet = build_npc_character_sheet(linna, world)
    assert sheet.social_aims
    assert any("Aldric" in ln for ln in sheet.social_aims)
