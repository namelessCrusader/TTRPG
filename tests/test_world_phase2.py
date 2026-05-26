"""
Tests for Phase 2 — world facts in cognition, chronicle, goals, D5 wiring.
"""

from __future__ import annotations

from src.sim.episodic_memory import (
    EPISODE_SIZE,
    _summarize_rule_based,
    chronicle_lines,
    maybe_summarize,
)
from src.sim.game_loop import GameLoop, make_test_world
from src.sim.grounding import classify
from src.sim.lm_adapter import MockLMAdapter
from src.sim.npc_policy import build_npc_character_sheet
from src.sim.projection import project
from src.sim.schemas import (
    ActionType,
    ActionZone,
    EntityId,
    EntityKind,
    Event,
    SemanticAction,
    Transition,
    TransitionKind,
    ValidationResult,
    WorldFact,
    WorldFactScope,
)
from src.sim.world_adjudicator import materialize_adjudication, rule_based_adjudicate
from src.sim.world_propagation import (
    enrich_goal_transitions,
    synthesize_goals_from_world_facts,
    synthesize_npc_goals,
)


def _player_id(world):
    for eid, ent in world.spatial.entities.items():
        if ent.kind == EntityKind.PLAYER:
            return eid
    return None


def test_world_facts_create_npc_goals():
    world = make_test_world()
    player_id = _player_id(world)
    assert player_id is not None

    world.world_facts.append(
        WorldFact(
            claim="Someone scribbled graffiti on the wall.",
            scope=WorldFactScope.WORLD,
            established_tick=world.tick,
            established_by=str(player_id),
        )
    )
    goals = synthesize_goals_from_world_facts(world)
    assert goals
    assert all(t.kind == TransitionKind.ENTITY_GOAL_ADDED for t in goals)


def test_tile_marked_synthesizes_witness_goal():
    world = make_test_world()
    transitions = [
        Transition(
            kind=TransitionKind.TILE_MARKED,
            payload={"x": 1, "y": 1, "mark": "fresh graffiti"},
        )
    ]
    goals = synthesize_npc_goals(world, transitions)
    assert any("graffiti" in t.payload.get("goal", "").lower() for t in goals)


def test_npc_sheet_includes_established_facts():
    world = make_test_world()
    player_id = _player_id(world)
    assert player_id is not None

    world.world_facts.append(
        WorldFact(
            claim="The counter has carvings.",
            scope=WorldFactScope.WORLD,
            established_tick=0,
        )
    )
    npc_id = next(
        eid for eid, e in world.spatial.entities.items()
        if e.kind != EntityKind.PLAYER
    )
    sheet = build_npc_character_sheet(world.spatial.entities[npc_id], world)
    assert any("carvings" in f.lower() for f in sheet.established_facts)


def test_rule_based_episode_summary_fallback():
    world = make_test_world()
    player_id = _player_id(world)
    ent = world.spatial.entities[player_id]
    ev = Event(
        tick=0,
        action=SemanticAction(verb="wait", actor=player_id),
        transitions=[],
        witnesses=[player_id],
    )
    summary = _summarize_rule_based(ent, [ev], world)
    assert ent.name in summary
    assert len(summary) > 10


def test_player_chronicle_on_episode_boundary():
    world = make_test_world()
    player_id = _player_id(world)
    assert player_id is not None

    world.event_log.append(
        Event(
            tick=1,
            action=SemanticAction(verb="wait", actor=player_id),
            transitions=[],
            witnesses=[player_id],
        )
    )
    world.tick = EPISODE_SIZE
    maybe_summarize(world, MockLMAdapter())
    assert world.player_chronicle
    assert chronicle_lines(world)


def test_enrich_goal_noop_on_mock():
    world = make_test_world()
    t = Transition(
        kind=TransitionKind.ENTITY_GOAL_ADDED,
        payload={"entity_id": "x", "goal": "Inspect the mark."},
    )
    before = t.payload["goal"]
    enrich_goal_transitions(world, [t], MockLMAdapter())
    assert t.payload["goal"] == before


class _EnrichAdapter(MockLMAdapter):
    def enrich_goal(self, goal_template, character_name, *, context=""):
        return f"{character_name} must: {goal_template[:40]}"


def test_enrich_goal_polishes_with_adapter():
    world = make_test_world()
    npc_id = next(
        str(eid)
        for eid, e in world.spatial.entities.items()
        if e.kind != EntityKind.PLAYER
    )
    t = Transition(
        kind=TransitionKind.ENTITY_GOAL_ADDED,
        payload={"entity_id": npc_id, "goal": "Inspect the fresh mark here."},
    )
    enrich_goal_transitions(world, [t], _EnrichAdapter())
    assert "must:" in t.payload["goal"]


def test_adjudicated_flag_wires_d5_for_real_adapter():
    """adjudicated=True enables D5 for non-mock adapters on player actions."""
    from src.sim.lm_adapter import OllamaLMAdapter

    world = make_test_world()
    player_id = _player_id(world)
    adapter = OllamaLMAdapter(model="test")
    loop = GameLoop(world, adapter=adapter, player_id=player_id)

    proposed: list[bool] = []
    original = adapter.propose_consequences

    def _track_propose(*args, **kwargs):
        proposed.append(True)
        return []

    adapter.propose_consequences = _track_propose  # type: ignore[method-assign]

    proj = project(world, player_id)
    action = SemanticAction(
        verb="levitate",
        actor=player_id,
        raw_input="levitate",
    )
    event = Event(
        tick=world.tick,
        action=action,
        transitions=[
            Transition(
                kind=TransitionKind.WORLD_MARK,
                payload={
                    "subject_kind": "entity",
                    "subject_id": str(player_id),
                    "verb": "adjudicate",
                    "actor_id": str(player_id),
                    "tick": world.tick,
                    "outcome": "adjudicated",
                    "summary": "ruled",
                },
            )
        ],
        witnesses=[player_id],
    )
    loop._apply_post_action_lm_layer(
        action, event, proj, adjudicated=True,
    )
    assert proposed, "D5 should run for adjudicated player events with real adapter"
