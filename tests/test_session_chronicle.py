"""Tests for interactive story presentation helpers."""

from pathlib import Path

from src.sim.game_loop import GameLoop
from src.sim.lm_adapter import MockLMAdapter
from src.sim.schemas import ActionZone, GroundingResult, RejectionReason, ValidationResult
from src.sim.session_chronicle import (
    apply_interactive_turn,
    format_rejection_hint,
    format_scene_opening,
)
from src.sim.world_loader import load_world_pack


def test_format_scene_opening_tavern():
    world = load_world_pack(Path(__file__).resolve().parents[1] / "worlds" / "tavern")
    lines = format_scene_opening(world)
    assert lines
    assert any("Opening night" in ln or "Flagon" in ln for ln in lines)
    assert any("Phase" in ln for ln in lines)


def test_player_step_surfaces_scenario_lines():
    from src.sim.game_loop import make_test_world

    world = make_test_world(str(Path(__file__).resolve().parents[1] / "worlds" / "tavern"))
    world.tick = 0
    world.meta.pop("scenario_last_beat_key", None)
    loop = GameLoop(world, MockLMAdapter())

    result = loop.step("wait")
    assert result.scenario_lines
    assert any("[scenario]" in ln for ln in result.scenario_lines)


def test_format_rejection_hint_orphaned():
    from src.sim.game_loop import GameLoop, StepResult, make_test_world
    from src.sim.projection import project
    from src.sim.schemas import SemanticAction, ActionType, GroundingResult, RejectionReason, ValidationResult, ActionZone

    world = make_test_world(str(Path(__file__).resolve().parents[1] / "worlds" / "tavern"))
    loop = GameLoop(world, MockLMAdapter())
    pid = loop.player_id
    proj = project(world, pid)
    result = StepResult(
        tick=0,
        projection=proj,
        action=SemanticAction(verb=ActionType.WAIT, actor=pid),
        grounding=GroundingResult(zone=ActionZone.ORPHANED, orphan_reason="no anchor"),
        validation=ValidationResult(valid=False, rejection_reason=RejectionReason.MALFORMED_ACTION),
        event=None,
        status="[ORPHANED]",
    )
    hint = format_rejection_hint(result)
    assert hint
    assert "say to" in hint.lower() or "observe" in hint.lower()


def test_apply_interactive_turn_story_mode_emits_chapter():
    from src.sim.game_loop import GameLoop, StepResult, make_test_world
    from src.sim.projection import project
    from src.sim.schemas import SemanticAction, ActionType, GroundingResult, ValidationResult, ActionZone

    world = make_test_world(str(Path(__file__).resolve().parents[1] / "worlds" / "tavern"))
    loop = GameLoop(world, MockLMAdapter())
    world.tick = 10
    world.meta.pop("scenario_last_beat_key", None)
    pid = loop.player_id
    proj = project(world, pid)
    scenario_lines = ["[scenario] beat entered: tick>=10 phase=1 — Rising curiosity"]
    result = StepResult(
        tick=10,
        projection=proj,
        action=SemanticAction(verb=ActionType.WAIT, actor=pid),
        grounding=GroundingResult(zone=ActionZone.GROUNDED),
        validation=ValidationResult(valid=True),
        event=None,
        status="[OK:Z1]",
        scenario_lines=scenario_lines,
    )
    emitted: list[tuple[str, str]] = []

    def emit(text: str, color: str = "normal") -> None:
        emitted.append((text, color))

    apply_interactive_turn(result, world, emit, story_mode=True)
    text = "\n".join(t for t, _ in emitted)
    assert "Chapter" in text or "phase" in text.lower()
