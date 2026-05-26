"""Tests for quick-win features: heuristics, inspect, tiered adapter, world diff."""

from src.sim.adjudication_heuristics import try_adjudication_heuristics
from src.sim.game_loop import make_test_world
from src.sim.grounding import classify
from src.sim.lm_adapter import TieredLMAdapter, get_adapter, MockLMAdapter
from src.sim.player_inspect import format_inspect_report
from src.sim.schemas import (
    ActionZone,
    EntityId,
    GroundingResult,
    IntentBlock,
    SemanticAction,
    ValidationResult,
)
from src.sim.world_adjudicator import rule_based_adjudicate
from src.sim.world_diff import capture_player_snapshot, format_world_diff


def _player_id(world):
    from src.sim.schemas import EntityKind
    for eid, ent in world.spatial.entities.items():
        if ent.kind == EntityKind.PLAYER:
            return eid
    return next(iter(world.spatial.entities))


def test_heuristic_ignite_produces_transitions():
    world = make_test_world("worlds/voxel_tavern")
    pid = _player_id(world)
    action = SemanticAction(
        verb="wait",
        actor=pid,
        target=None,
        intent=IntentBlock(rationale="", manner="", desired_outcome=[]),
        raw_input="set fire to the oil on the floor",
    )
    grounding = GroundingResult(zone=ActionZone.ORPHANED, orphan_reason="test")
    result = ValidationResult(valid=False)
    adj = try_adjudication_heuristics(
        world, action, action.raw_input, grounding, result,
    )
    assert adj is not None
    kinds = {p.kind for p in adj.transition_proposals}
    assert "tile_marked" in kinds
    assert "environment_state_changed" in kinds


def test_heuristic_wired_in_rule_based_adjudicate():
    world = make_test_world("worlds/voxel_tavern")
    pid = _player_id(world)
    intent = "wedge a chair against the door"
    action = SemanticAction(
        verb="wait",
        actor=pid,
        target=None,
        intent=IntentBlock(rationale="", manner="", desired_outcome=[]),
        raw_input=intent,
    )
    grounding = GroundingResult(zone=ActionZone.ORPHANED)
    result = ValidationResult(valid=False)
    adj = rule_based_adjudicate(world, action, intent, grounding, result)
    kinds = {p.kind for p in adj.transition_proposals}
    assert "structure_created" in kinds or "tile_marked" in kinds


def test_inspect_report_lists_actions():
    world = make_test_world("worlds/voxel_tavern")
    pid = _player_id(world)
    report = format_inspect_report(world, pid)
    assert "Inspect" in report
    assert "Actions you could try" in report or "kernel verbs" in report


def test_inspect_target_entity():
    world = make_test_world("worlds/voxel_tavern")
    pid = _player_id(world)
    # Pick any NPC name fragment
    npc = next(e for e in world.spatial.entities.values() if e.entity_id != pid)
    report = format_inspect_report(world, pid, npc.name.split()[0].lower())
    assert npc.name.split()[0] in report


def test_tiered_adapter_routes_infer():
    player = MockLMAdapter()
    npc = MockLMAdapter()
    tiered = TieredLMAdapter(player, npc)
    assert tiered.player_adapter is player
    assert tiered.npc_adapter is npc


def test_get_adapter_tiered_when_models_differ():
    adapter = get_adapter(
        use_ollama=True,
        model="gemma3:1b",
        player_model="qwen2.5:7b",
        npc_model="gemma3:1b",
    )
    assert isinstance(adapter, TieredLMAdapter)


def test_world_diff_after_health_change():
    world = make_test_world("worlds/tavern")
    pid = _player_id(world)
    before = capture_player_snapshot(world, pid)
    ent = world.spatial.entities[pid]
    ent.health -= 5
    after = capture_player_snapshot(world, pid)
    diff = format_world_diff(before, after, [], world)
    assert "hp" in diff.lower() or "World changed" in diff
