from src.sim.game_loop import GameLoop, make_test_world
from src.sim.lm_adapter import MockLMAdapter
from src.sim.narration_layers import render_event_layered
from src.sim.presentation_contract import PresentationStream, TickFrame
from src.sim.schemas import EntityId, SemanticProjection


def test_presentation_stream_serializes():
    proj = SemanticProjection(tick=1, focal_entity=EntityId("ent_player"))
    frame = TickFrame(tick=1, projection=proj)
    stream = PresentationStream(world_name="test", run_id="r1", frames=[frame])
    assert "test" in stream.to_json()


def test_narration_layered_falls_back_to_rules():
    world = make_test_world()
    loop = GameLoop(world, adapter=MockLMAdapter())
    loop.step("wait")
    assert world.event_log
    text = render_event_layered(world.event_log[-1], world, use_lm_color=False)
    assert isinstance(text, str) and text
