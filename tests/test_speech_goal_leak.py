"""Goals and drive stubs must not appear as spoken dialogue."""

from pathlib import Path

from src.sim.speech_utils import is_planning_text, is_real_dialogue


def test_goal_text_is_planning_not_dialogue():
    goal = "get through tonight without getting involved in whatever Tomas is selling"
    assert is_planning_text(goal)
    assert not is_real_dialogue(goal)


def test_voice_line_is_dialogue():
    line = "Choose your words carefully — someone's always listening."
    assert not is_planning_text(line)
    assert is_real_dialogue(line)


def test_aldric_has_voice_lines_in_pack():
    from src.sim.world_loader import load_world_pack

    world = load_world_pack(Path(__file__).resolve().parents[1] / "worlds" / "tavern")
    aldric = next(e for e in world.spatial.entities.values() if "Aldric" in e.name)
    assert len(aldric.voice_lines) >= 2
