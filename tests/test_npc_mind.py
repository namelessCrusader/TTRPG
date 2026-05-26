"""Tests for context-driven NPC mind scoring."""

from pathlib import Path

import pytest

from src.sim.game_loop import GameLoop
from src.sim.lm_adapter import MockLMAdapter
from src.sim.npc_mind import (
    INTENT_ACKNOWLEDGE,
    INTENT_REPLY,
    interpret_mind,
    load_mind,
    pick_best_action,
    score_action,
)
from src.sim.npc_policy import ReactivePolicy
from src.sim.schemas import ActionType, EntityKind, IntentBlock, SemanticAction
from src.sim.world_loader import load_world_pack


@pytest.fixture
def tavern_world():
    return load_world_pack(Path(__file__).resolve().parents[1] / "worlds" / "tavern")


def test_reply_intent_prefers_speak_over_toast(tavern_world):
    world = tavern_world
    mira_id = next(
        eid for eid, e in world.spatial.entities.items() if e.name == "Mira"
    )
    tomas_id = next(
        eid for eid, e in world.spatial.entities.items() if e.name == "Tomas"
    )
    mira = world.spatial.entities[mira_id]
    mira.meta["focus_topic"] = "reply"
    mira.meta["should_reply_to_id"] = str(tomas_id)
    mira.meta["last_speak_tick"] = world.tick - 5
    interpret_mind(mira, world)
    mind = load_mind(mira)
    assert mind.intent == INTENT_REPLY
    assert mind.speak_ready

    speak = SemanticAction(
        verb=ActionType.SPEAK,
        actor=mira_id,
        target=tomas_id,
        intent=IntentBlock(manner="test"),
        raw_input="[test]",
    )
    toast = SemanticAction(
        verb="toast",
        actor=mira_id,
        target=tomas_id,
        raw_input="[test]",
    )
    s_speak = score_action(mira, world, speak, 0.5)
    s_toast = score_action(mira, world, toast, 0.68)
    assert s_speak > s_toast


def test_gesture_repeat_suppresses_second_toast(tavern_world):
    world = tavern_world
    linna_id = next(
        eid for eid, e in world.spatial.entities.items() if e.name == "Linna"
    )
    tomas_id = next(
        eid for eid, e in world.spatial.entities.items() if e.name == "Tomas"
    )
    linna = world.spatial.entities[linna_id]
    linna.meta["mind"] = {
        "intent": INTENT_ACKNOWLEDGE,
        "attention_target": str(tomas_id),
        "recent_gestures": {str(tomas_id): world.tick},
        "speak_ready": True,
        "ticks_since_speak": 5,
    }
    toast = SemanticAction(
        verb="toast",
        actor=linna_id,
        target=tomas_id,
        raw_input="[test]",
    )
    speak = SemanticAction(
        verb=ActionType.SPEAK,
        actor=linna_id,
        target=tomas_id,
        raw_input="[test]",
    )
    assert score_action(linna, world, toast, 0.65) < score_action(
        linna, world, speak, 0.55
    )


def test_autonomous_run_has_varied_verbs(tavern_world):
    world = tavern_world
    loop = GameLoop(
        world,
        adapter=MockLMAdapter(),
        npc_policy=ReactivePolicy(),
        player_id=None,
    )
    for _ in range(5):
        loop.autonomous_tick()
    verbs = [
        str(e.action.verb).split(".")[-1]
        for e in world.event_log
        if str(e.action.actor) != "system"
    ]
    assert "speak" in verbs or "gossip" in verbs or "accuse" in verbs
    assert verbs.count("toast") < len(verbs) * 0.85
