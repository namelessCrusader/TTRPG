from pathlib import Path

from src.sim.compiler import compile_action
from src.sim.compiler import apply_transitions
from src.sim.lm_adapter import MockLMAdapter
from src.sim.npc_reflection import run_npc_reflection, speech_intent_for_entity
from src.sim.game_loop import make_test_world
from src.sim.schemas import ActionType, EntityKind, Event, IntentBlock, SemanticAction
from src.sim.world_loader import load_world_pack


def test_reflection_sets_inner_thought_after_speech():
    world = load_world_pack(
        Path(__file__).resolve().parents[1] / "worlds" / "tavern"
    )
    npc_id = next(
        eid for eid, e in world.spatial.entities.items() if e.kind == EntityKind.NPC
    )
    npc = world.spatial.entities[npc_id]
    other_id = next(
        eid
        for eid, e in world.spatial.entities.items()
        if eid != npc_id and e.position.manhattan(npc.position) <= npc.sight_range
    )

    speak = SemanticAction(
        verb=ActionType.SPEAK,
        actor=other_id,
        target=npc_id,
        intent=IntentBlock(
            rationale="The kitchen is through the back door.",
            manner="The kitchen is through the back door.",
        ),
    )
    val = compile_action(speak, world)
    assert val.valid, val.rejection_detail
    apply_transitions(world, val.concrete_transitions)
    world.event_log.append(
        Event(
            tick=world.tick,
            action=speak,
            transitions=val.concrete_transitions,
            witnesses=[npc_id],
        )
    )

    run_npc_reflection(npc, world, MockLMAdapter())
    assert npc.meta.get("inner_thought")
    assert npc.meta.get("heard_recently")
    assert npc.meta.get("focus_topic") in (
        "reply",
        "need_food",
        "pursue_drive",
        "pursue_goal",
        "threat",
        "need_rest",
        "need_company",
    )


def test_speech_intent_for_hunger_focus():
    world = make_test_world()
    ent = next(iter(world.spatial.entities.values()))
    ent.meta["focus_topic"] = "need_food"
    hint = speech_intent_for_entity(ent)
    assert "hungry" in hint.lower() or "food" in hint.lower()
