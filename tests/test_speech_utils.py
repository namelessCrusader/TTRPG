"""Tests for dialogue quality helpers."""

from src.sim.speech_utils import (
    is_planning_text,
    is_real_dialogue,
    is_reply_rationale_stub,
    normalize_speech_key,
    sanitize_dialogue_text,
)
from src.sim.npc_lm_policy import LMNpcPolicy


def test_planning_text_detects_tone_words():
    assert is_planning_text("firmly")
    assert is_planning_text("hushed, alarmed")


def test_planning_text_detects_reactive_hunger_stub():
    assert is_planning_text("ask if they have food or know where to get some")
    assert is_planning_text("Who else knows about this — I need to know before I can act.")


def test_not_real_dialogue_for_prompt_fragments():
    assert not is_real_dialogue("a bawdy ballad")
    assert not is_real_dialogue("softly across the cheek")
    assert not is_real_dialogue("in peace")
    assert not is_real_dialogue("calm, observant, slightly weary")
    assert not is_real_dialogue("")
    assert not is_real_dialogue(
        "To gather insights about the eastern trade route and ensure my safety."
    )
    assert not is_real_dialogue("ask if they have food or know where to get some")


def test_real_dialogue_accepts_sentences():
    assert is_real_dialogue("Step back, stranger.")
    assert is_real_dialogue(
        "I've heard stranger things and lived to tell the tale."
    )


def test_normalize_speech_key():
    assert normalize_speech_key('  "Hello."  ') == normalize_speech_key("hello.")


def test_reply_rationale_stub():
    assert is_reply_rationale_stub('reply to Mira: "hello"')
    assert is_reply_rationale_stub(
        'reply to You: "reply to Mira: "reply to You: "nested"""'
    )
    assert not is_reply_rationale_stub("Step back, stranger.")


def test_sanitize_dialogue_rejects_stubs():
    assert sanitize_dialogue_text('reply to Tomas: "hi"') == ""
    assert sanitize_dialogue_text("I've heard stranger things.") == (
        "I've heard stranger things."
    )

