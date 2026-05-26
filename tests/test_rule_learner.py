"""Tests for the rule-learner (records LM adjudications as candidate rules)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.sim.game_loop import make_test_world
from src.sim.rule_learner import (
    is_enabled,
    record_adjudication,
    reset_dedupe,
)
from src.sim.schemas import (
    AdjudicationResult,
    EntityKind,
    IntentBlock,
    SemanticAction,
    StyleBlock,
    TransitionProposal,
    WorldFact,
    WorldFactScope,
)


# ── helpers ──────────────────────────────────────────────────────────────


def _player_id(world):
    return next(
        eid for eid, e in world.spatial.entities.items()
        if e.kind == EntityKind.PLAYER
    )


def _action(pid: str, verb: str = "improvise_torch_bomb",
            intent: str = "I rig the lantern as a thrown bomb") -> SemanticAction:
    return SemanticAction(
        verb=verb,
        actor=pid,
        target=None,
        intent=IntentBlock(rationale=intent),
        style=StyleBlock(aggression=40, visibility=70),
        raw_input=intent,
    )


def _adj(proposals: list[TransitionProposal] | None = None,
         facts: list[WorldFact] | None = None,
         ruling: str = "The lantern improvisation is plausible.",
         synth: str | None = None) -> AdjudicationResult:
    return AdjudicationResult(
        ruling_text=ruling,
        facts=facts or [],
        scheduled_effects=[],
        transition_proposals=proposals or [],
        synthesized_verb=synth,
    )


@pytest.fixture(autouse=True)
def _clear_dedupe():
    reset_dedupe()
    yield
    reset_dedupe()


# ── opt-in semantics ─────────────────────────────────────────────────────


def test_disabled_by_default(tmp_path):
    world = make_test_world()
    world.config.extra.pop("learn_from_adjudication", None)  # Ensure it is disabled
    world.meta["_pack_path"] = str(tmp_path)  # make pack path resolvable
    assert is_enabled(world) is False
    pid = _player_id(world)
    out = record_adjudication(
        world,
        _action(pid),
        intent="anything",
        adj=_adj(
            proposals=[TransitionProposal(
                kind="entity_health_changed",
                payload={"entity_id": pid, "delta": -3},
            )],
        ),
    )
    assert out is None
    assert not any((tmp_path / "learned").glob("*.yaml")) if (tmp_path / "learned").exists() else True


def test_enabled_via_extra_flag(tmp_path):
    world = make_test_world()
    world.meta["_pack_path"] = str(tmp_path)
    world.config.extra["learn_from_adjudication"] = True
    assert is_enabled(world) is True


# ── writing behaviour ────────────────────────────────────────────────────


def test_records_when_enabled_and_has_proposals(tmp_path):
    world = make_test_world()
    world.meta["_pack_path"] = str(tmp_path)
    world.config.extra["learn_from_adjudication"] = True

    pid = _player_id(world)
    proposals = [TransitionProposal(
        kind="entity_health_changed",
        payload={"entity_id": pid, "delta": -5, "cause": "improvised_bomb"},
    )]
    target = record_adjudication(
        world,
        _action(pid, verb="rig_lantern_bomb",
                intent="I rig the oil lantern into a thrown firebomb"),
        intent="I rig the oil lantern into a thrown firebomb",
        adj=_adj(proposals=proposals, synth="throw"),
    )
    assert target is not None
    assert target.exists()
    body = target.read_text(encoding="utf-8")
    assert "templates:" in body
    assert "rig_lantern_bomb" in body
    assert "entity_health_changed" in body
    assert "improvised_bomb" in body
    # The header comment should record the original intent verbatim.
    assert "I rig the oil lantern" in body
    # And the synthesized verb is recorded as DM metadata.
    assert "synthesized_verb (DM): throw" in body


def test_skips_empty_adjudication(tmp_path):
    world = make_test_world()
    world.meta["_pack_path"] = str(tmp_path)
    world.config.extra["learn_from_adjudication"] = True

    pid = _player_id(world)
    out = record_adjudication(
        world,
        _action(pid),
        intent="prose only",
        adj=_adj(ruling="just narration, no mechanical effect"),
    )
    assert out is None
    assert not (tmp_path / "learned").exists()


def test_dedupes_identical_rule_within_session(tmp_path):
    world = make_test_world()
    world.meta["_pack_path"] = str(tmp_path)
    world.config.extra["learn_from_adjudication"] = True

    pid = _player_id(world)
    proposals = [TransitionProposal(
        kind="entity_health_changed",
        payload={"entity_id": pid, "delta": -2},
    )]
    first = record_adjudication(
        world,
        _action(pid, verb="prod_with_stick"),
        intent="prod the guard with the broken stick",
        adj=_adj(proposals=proposals),
    )
    second = record_adjudication(
        world,
        _action(pid, verb="prod_with_stick"),
        intent="prod the guard with the broken stick",
        adj=_adj(proposals=proposals),
    )
    assert first is not None
    assert second is None, "second call with identical rule shape should dedupe"


def test_different_proposal_shapes_are_recorded_separately(tmp_path):
    world = make_test_world()
    world.meta["_pack_path"] = str(tmp_path)
    world.config.extra["learn_from_adjudication"] = True

    pid = _player_id(world)
    first = record_adjudication(
        world,
        _action(pid, verb="prod_with_stick"),
        intent="prod the guard",
        adj=_adj(proposals=[TransitionProposal(
            kind="entity_health_changed",
            payload={"entity_id": pid, "delta": -1},
        )]),
    )
    second = record_adjudication(
        world,
        _action(pid, verb="prod_with_stick"),
        intent="prod the guard harder",
        adj=_adj(proposals=[TransitionProposal(
            kind="entity_alertness_changed",
            payload={"entity_id": pid, "to": "high"},
        )]),
    )
    assert first is not None
    assert second is not None
    body = second.read_text(encoding="utf-8")
    assert body.count("- verb: prod_with_stick") == 2


def test_skips_when_pack_path_missing(tmp_path):
    world = make_test_world()
    # Remove pack path marker — simulating a synthetic test world.
    world.meta.pop("_pack_path", None)
    world.config.extra["learn_from_adjudication"] = True
    pid = _player_id(world)
    out = record_adjudication(
        world,
        _action(pid),
        intent="anything",
        adj=_adj(proposals=[TransitionProposal(
            kind="entity_health_changed",
            payload={"entity_id": pid, "delta": -1},
        )]),
    )
    assert out is None


def test_file_basename_override(tmp_path):
    world = make_test_world()
    world.meta["_pack_path"] = str(tmp_path)
    world.config.extra["learn_from_adjudication"] = True
    pid = _player_id(world)
    target = record_adjudication(
        world,
        _action(pid),
        intent="x",
        adj=_adj(proposals=[TransitionProposal(
            kind="entity_health_changed",
            payload={"entity_id": pid, "delta": -1},
        )]),
        file_basename="fixed-filename",
    )
    assert target is not None
    assert target.name == "fixed-filename.yaml"
