"""Tests for overhauls A, C, D, E."""

from pathlib import Path

from src.sim.adjudication_resolver import match_adjudication_index, resolve_from_index
from src.sim.grounding import classify
from src.sim.intent_router import parse_player_intent
from src.sim.interaction_resolver import resolve_interaction_action
from src.sim.schemas import (
    ActionZone,
    EntityKind,
    GroundingResult,
    SemanticAction,
    ValidationResult,
)
from src.sim.world_loader import load_world_pack

_REPO = Path(__file__).resolve().parent.parent


def _player_id(world):
    for eid, ent in world.spatial.entities.items():
        if ent.kind == EntityKind.PLAYER:
            return eid
    return next(iter(world.spatial.entities))


def test_pack_inheritance_loads_physics_and_rules():
    world = load_world_pack(_REPO / "worlds" / "tavern")
    assert world.config.physics_config is not None
    assert len(world.config.interaction_rules) >= 5
    assert len(world.config.adjudication_index) >= 5
    assert len(world.config.combination_rules) >= 3
    assert "pack_extends" in world.meta


def test_interaction_grammar_resolves_intimidate():
    world = load_world_pack(_REPO / "worlds" / "tavern")
    pid = _player_id(world)
    npc = next(e for e in world.spatial.entities.values() if e.entity_id != pid)
    action = resolve_interaction_action(
        world, pid, f"intimidate {npc.name.split()[0]}",
    )
    assert action is not None
    assert action.verb == "intimidate"
    assert action.proposed_effects


def test_interaction_wired_in_intent_router():
    world = load_world_pack(_REPO / "worlds" / "tavern")
    pid = _player_id(world)
    action = parse_player_intent("intimidate Tomas", pid, world)
    assert action is not None
    assert str(action.verb) == "intimidate"


def test_interaction_resolves_role_tag_bartender():
    world = load_world_pack(_REPO / "worlds" / "tavern")
    tomas = next(e for e in world.spatial.entities.values() if e.name == "Tomas")
    action = parse_player_intent("intimidate the bartender", tomas.entity_id, world)
    assert action is not None
    assert action.verb == "intimidate"
    assert action.target != tomas.entity_id


def test_adjudication_index_barricade():
    world = load_world_pack(_REPO / "worlds" / "tavern")
    entry = match_adjudication_index(world, "wedge a chair against the door")
    assert entry is not None
    assert entry.id == "barricade_door"


def test_adjudication_index_resolves_transitions():
    world = load_world_pack(_REPO / "worlds" / "tavern")
    pid = _player_id(world)
    action = SemanticAction(
        verb="wait",
        actor=pid,
        raw_input="wedge chair against door",
    )
    grounding = GroundingResult(zone=ActionZone.ORPHANED)
    result = ValidationResult(valid=False)
    adj = resolve_from_index(world, action, action.raw_input, grounding, result)
    assert adj is not None
    assert adj.transition_proposals
    assert adj.ruling_text


def test_projection_has_interaction_hints():
    from src.sim.projection import project

    world = load_world_pack(_REPO / "worlds" / "tavern")
    pid = _player_id(world)
    proj = project(world, pid)
    assert isinstance(proj.interaction_hints, list)


def test_social_inspect_format():
    from src.sim.social_inspect import format_social_inspect

    world = load_world_pack(_REPO / "worlds" / "tavern")
    pid = _player_id(world)
    report = format_social_inspect(world, pid)
    assert "Social" in report
