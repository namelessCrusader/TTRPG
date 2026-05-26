"""
Tests for role-tiered LM routing on ``TieredLMAdapter``.

Verifies that ``adjudicator_adapter`` and ``narrator_adapter`` slots
correctly override the player/npc actor-routing for the adjudicate /
propose_consequences / narrate paths, and that defaults still match
the original 2-way contract when those slots are not supplied.
"""

from __future__ import annotations

from typing import Optional

from src.sim.lm_adapter import LMAdapter, MockLMAdapter, TieredLMAdapter
from src.sim.schemas import (
    AdjudicationResult,
    ActionZone,
    GroundingResult,
    IntentBlock,
    NpcCharacterSheet,
    SemanticAction,
    SemanticProjection,
    StyleBlock,
    TransitionProposal,
    ValidationResult,
)


class _RecordingAdapter(MockLMAdapter):
    """Mock adapter that records which methods were invoked on it."""

    def __init__(self, label: str):
        super().__init__()
        self.label = label
        self.calls: list[str] = []

    @property
    def model(self) -> str:  # for *_model_id properties
        return self.label

    def infer(self, projection, player_intent):
        self.calls.append("infer")
        return super().infer(projection, player_intent)

    def infer_npc(self, character_sheet, projection):
        self.calls.append("infer_npc")
        return super().infer_npc(character_sheet, projection)

    def adjudicate(self, action, projection, intent, grounding, result):
        self.calls.append("adjudicate")
        return AdjudicationResult(
            ruling_text=f"[{self.label}] adjudicated",
            transition_proposals=[],
        )

    def propose_consequences(self, action, projection, visible_ids):
        self.calls.append("propose_consequences")
        return []

    def narrate(self, prompt):
        self.calls.append("narrate")
        return f"[{self.label}] narration"


def _projection_stub() -> SemanticProjection:
    # Build a near-empty projection.  Most tests only care that *some*
    # projection is passed through; we don't exercise its contents.
    return SemanticProjection(
        focal_entity="ent_player",
        tick=0,
        visible_entities=[],
    )


def _action() -> SemanticAction:
    return SemanticAction(
        verb="rig_lantern_bomb",
        actor="ent_player",
        intent=IntentBlock(rationale="rig the lantern as a bomb"),
        style=StyleBlock(aggression=40, visibility=50),
        raw_input="rig the lantern",
    )


# ── backward-compatible 2-way behaviour ──────────────────────────────────


def test_default_routing_matches_two_way_contract():
    """When adjudicator/narrator are None, routing matches the original
    player/npc 2-way split: adjudicate→player, narrate→npc."""
    player = _RecordingAdapter("PLAYER")
    npc = _RecordingAdapter("NPC")
    adapter = TieredLMAdapter(player_adapter=player, npc_adapter=npc)

    adapter.adjudicate(
        _action(), _projection_stub(), "rig the lantern",
        GroundingResult(zone=ActionZone.GROUNDED),
        ValidationResult(valid=False),
    )
    adapter.narrate("describe the scene")

    assert player.calls == ["adjudicate"]
    assert npc.calls == ["narrate"]


def test_propose_consequences_falls_back_to_player_when_no_adjudicator():
    player = _RecordingAdapter("PLAYER")
    npc = _RecordingAdapter("NPC")
    adapter = TieredLMAdapter(player_adapter=player, npc_adapter=npc)

    adapter.propose_consequences(_action(), _projection_stub(), [])

    assert "propose_consequences" in player.calls
    assert "propose_consequences" not in npc.calls


# ── role-specific overrides ──────────────────────────────────────────────


def test_adjudicator_slot_overrides_player_for_adjudicate():
    player = _RecordingAdapter("PLAYER")
    npc = _RecordingAdapter("NPC")
    judge = _RecordingAdapter("JUDGE")
    adapter = TieredLMAdapter(
        player_adapter=player,
        npc_adapter=npc,
        adjudicator_adapter=judge,
    )

    adj = adapter.adjudicate(
        _action(), _projection_stub(), "rig the lantern",
        GroundingResult(zone=ActionZone.GROUNDED),
        ValidationResult(valid=False),
    )

    assert "adjudicate" in judge.calls
    assert "adjudicate" not in player.calls
    assert "[JUDGE]" in (adj.ruling_text or "")


def test_adjudicator_slot_also_routes_propose_consequences():
    """Schema-tight role: adjudicator slot owns *all* schema-constrained
    mutation paths, including propose_consequences."""
    player = _RecordingAdapter("PLAYER")
    npc = _RecordingAdapter("NPC")
    judge = _RecordingAdapter("JUDGE")
    adapter = TieredLMAdapter(
        player_adapter=player,
        npc_adapter=npc,
        adjudicator_adapter=judge,
    )

    adapter.propose_consequences(_action(), _projection_stub(), [])

    assert "propose_consequences" in judge.calls
    assert "propose_consequences" not in player.calls


def test_narrator_slot_overrides_npc_for_narrate():
    player = _RecordingAdapter("PLAYER")
    npc = _RecordingAdapter("NPC")
    bard = _RecordingAdapter("BARD")
    adapter = TieredLMAdapter(
        player_adapter=player,
        npc_adapter=npc,
        narrator_adapter=bard,
    )

    text = adapter.narrate("describe the scene")

    assert "narrate" in bard.calls
    assert "narrate" not in npc.calls
    assert text == "[BARD] narration"


def test_player_and_npc_paths_unchanged_when_role_slots_set():
    """Adding role slots must not regress the original infer / infer_npc
    routing — player intent still goes to player_adapter, NPC turns still
    go to npc_adapter."""
    player = _RecordingAdapter("PLAYER")
    npc = _RecordingAdapter("NPC")
    judge = _RecordingAdapter("JUDGE")
    bard = _RecordingAdapter("BARD")
    adapter = TieredLMAdapter(
        player_adapter=player,
        npc_adapter=npc,
        adjudicator_adapter=judge,
        narrator_adapter=bard,
    )

    sheet = NpcCharacterSheet(
        npc_id="ent_npc",
        name="Mira",
    )
    adapter.infer(_projection_stub(), "say hello")
    adapter.infer_npc(sheet, _projection_stub())

    assert "infer" in player.calls
    assert "infer_npc" in npc.calls
    assert "infer" not in judge.calls
    assert "infer" not in bard.calls


# ── model-id properties expose the right tiers ───────────────────────────


def test_model_id_properties_report_role_specific_slots():
    player = _RecordingAdapter("PLAYER")
    npc = _RecordingAdapter("NPC")
    judge = _RecordingAdapter("JUDGE")
    bard = _RecordingAdapter("BARD")
    adapter = TieredLMAdapter(
        player_adapter=player,
        npc_adapter=npc,
        adjudicator_adapter=judge,
        narrator_adapter=bard,
    )
    assert adapter.player_model_id == "PLAYER"
    assert adapter.npc_model_id == "NPC"
    assert adapter.adjudicator_model_id == "JUDGE"
    assert adapter.narrator_model_id == "BARD"


def test_model_id_properties_fall_back_to_actor_tier_when_slot_empty():
    player = _RecordingAdapter("PLAYER")
    npc = _RecordingAdapter("NPC")
    adapter = TieredLMAdapter(player_adapter=player, npc_adapter=npc)
    assert adapter.adjudicator_model_id == "PLAYER"
    assert adapter.narrator_model_id == "NPC"
