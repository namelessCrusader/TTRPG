"""Unit tests for LTL-style LM response parsers."""

from src.sim.lm_adapter.helpers import (
    _parse_adjudication_response,
    _parse_lm_response,
    _parse_mdp_selection_response,
    _parse_npc_speech_response,
    _parse_proposals,
)
from src.sim.schemas import AdjudicationResult, TransitionProposal


def test_parse_mdp_selection_ltl():
    raw = (
        "Thinking...\n"
        "NEXT(selected_option_index == 3)\n"
        'EVENTUALLY(rationale == "I want to greet Tomas.")\n'
        'ALWAYS(custom_speech_line == "Evening, Tomas.")\n'
    )
    out = _parse_mdp_selection_response(raw)
    assert out["selected_option_index"] == 3
    assert "Tomas" in out["rationale"]
    assert out["custom_speech_line"] == "Evening, Tomas."


def test_parse_lm_response_ltl():
    raw = (
        'NEXT(verb == "speak")\n'
        'NEXT(target == "ent_mira")\n'
        'EVENTUALLY(rationale == "Ask about the letter.")\n'
        'ALWAYS(manner == "Mira, have you heard anything?")\n'
        "STYLE(emotional_tone == \"curious\", aggression == 20, visibility == 70)\n"
    )
    action = _parse_lm_response(raw, "You", "ask mira about the letter", ["ent_mira"])
    assert str(action.verb).lower() == "speak"
    assert action.target == "ent_mira"
    assert action.intent.rationale
    assert "Mira" in (action.intent.manner or "")


def test_parse_proposals_consequence_lines():
    raw = (
        'CONSEQUENCE(kind == "entity_emotional_state_changed", '
        'entity_id == "ent_mira", to == "friendly")\n'
        'CONSEQUENCE(kind == "edge_created", source == "ent_mira", '
        'target == "You", edge_kind == "respects", weight == 0.1)\n'
    )
    props = _parse_proposals(raw, ["ent_mira", "You"])
    assert len(props) == 2
    assert props[0].kind == "entity_emotional_state_changed"
    assert props[0].payload["entity_id"] == "ent_mira"


def test_parse_adjudication_ltl():
    raw = (
        'RULING(ruling_text == "The merchant nods slowly.")\n'
        'SYNTHESIZED_VERB(verb == "nod")\n'
        'CONSEQUENCE(kind == "entity_emotional_state_changed", '
        'entity_id == "ent_aldric", to == "neutral")\n'
    )
    adj = _parse_adjudication_response(raw, ["ent_aldric"], current_tick=5)
    assert isinstance(adj, AdjudicationResult)
    assert "merchant" in adj.ruling_text.lower()
    assert adj.synthesized_verb == "nod"
    assert len(adj.transition_proposals) >= 1


def test_parse_npc_speech_ltl():
    raw = 'ALWAYS(spoken_line == "The hearth is warm tonight.")\n'
    assert _parse_npc_speech_response(raw) == "The hearth is warm tonight."
