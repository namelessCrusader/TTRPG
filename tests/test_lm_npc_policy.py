"""
Tests for the LM-driven NPC policy.

Verifies that:
  • LMNpcPolicy calls the adapter's infer_npc with a character sheet
    summarized from canonical state.
  • The returned SemanticAction flows through the same Compiler as
    player actions (no new pipeline).
  • Failures (MalformedActionError, NotImplementedError) fall back to
    ReactivePolicy so an NPC never no-ops.
  • The actor field on the returned action is forced to match the
    decided-for entity even if the LM tries to impersonate someone else.
"""

from __future__ import annotations

from typing import Optional

import pytest

from src.sim.compiler import compile_action
from src.sim.game_loop import GameLoop, make_test_world
from src.sim.lm_adapter import LMAdapter
from src.sim.npc_lm_policy import LMNpcPolicy
from src.sim.npc_policy import ReactivePolicy
from src.sim.schemas import (
    ActionType,
    AlertnessLevel,
    EmotionalState,
    EntityKind,
    EntityId,
    IntentBlock,
    MalformedActionError,
    NpcCharacterSheet,
    SemanticAction,
    SemanticProjection,
)


def _player_id(world):
    for eid, e in world.spatial.entities.items():
        if e.kind == EntityKind.PLAYER:
            return eid
    return None


def _guard_id(world):
    for eid, e in world.spatial.entities.items():
        if e.kind == EntityKind.NPC:
            return eid
    return None


class _ScriptedAdapter(LMAdapter):
    """Adapter that drives MDP selection toward a scripted verb when possible."""

    def __init__(
        self,
        npc_action: Optional[SemanticAction] = None,
        npc_error: Optional[Exception] = None,
        *,
        selected_index: Optional[int] = None,
    ):
        self.npc_action = npc_action
        self.npc_error = npc_error
        self.selected_index = selected_index
        self.last_sheet: Optional[NpcCharacterSheet] = None
        self.last_projection: Optional[SemanticProjection] = None
        self.calls = 0

    def infer(self, projection, player_intent):
        return SemanticAction(verb=ActionType.WAIT, actor=projection.focal_entity)

    def infer_npc(self, character_sheet, projection):
        if self.npc_error is not None:
            raise self.npc_error
        assert self.npc_action is not None
        return self.npc_action

    def infer_npc_mdp(self, character_sheet, projection, options, option_indices):
        self.calls += 1
        self.last_sheet = character_sheet
        self.last_projection = projection
        if self.npc_error is not None:
            raise self.npc_error
        if self.selected_index is not None:
            idx = self.selected_index
        elif self.npc_action is not None:
            verb = str(self.npc_action.verb).lower()
            idx = 0
            for i, opt in enumerate(options, 1):
                if f"verb={verb}" in opt.lower() or f"verb={self.npc_action.verb}" in opt:
                    idx = i
                    break
            if idx == 0:
                idx = option_indices[0] if option_indices else 1
        else:
            idx = option_indices[0] if option_indices else 1
        return {
            "selected_option_index": idx,
            "rationale": "scripted test selection",
            "custom_speech_line": "",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Policy hands the adapter a character sheet and a projection
# ─────────────────────────────────────────────────────────────────────────────


def test_lm_npc_policy_passes_character_sheet_and_projection():
    world = make_test_world()
    gid = _guard_id(world)
    guard = world.spatial.entities[gid]
    guard.alertness = AlertnessLevel.MEDIUM

    # Adapter returns a freeform verb the engine doesn't know — proving
    # the policy passes any string through verbatim.
    scripted = SemanticAction(
        verb="rebuff",
        actor=gid,
        intent=IntentBlock(manner="firmly"),
    )
    adapter = _ScriptedAdapter(npc_action=scripted)
    policy = LMNpcPolicy(adapter, cognition_mode="lm")

    action = policy.decide(guard, world)

    assert adapter.calls == 1
    assert adapter.last_sheet is not None
    assert adapter.last_sheet.npc_id == gid
    assert adapter.last_sheet.name == guard.name
    assert adapter.last_projection is not None
    assert adapter.last_projection.focal_entity == gid

    assert action.actor == gid
    assert action.verb  # chosen from pre-validated forest


def test_lm_npc_policy_mdp_selects_from_forest():
    """MDP path picks a compiled option from the reactive candidate forest."""
    from src.sim.world_loader import load_world_pack

    world = load_world_pack("worlds/tavern")
    mira_id = next(eid for eid, ent in world.spatial.entities.items() if ent.name == "Mira")
    mira = world.spatial.entities[mira_id]
    mira.meta["last_speak_tick"] = world.tick - 10

    adapter = _ScriptedAdapter(selected_index=1)
    policy = LMNpcPolicy(adapter, fallback=ReactivePolicy(), cognition_mode="lm")

    action = policy.decide(mira, world)

    assert adapter.calls == 1
    assert action.actor == mira_id
    assert "infer_npc_mdp_option" in (mira.meta.get("last_policy_branch") or "")


@pytest.mark.parametrize("generic_verb", ["interact", "set_priority", "find", "wait"])
def test_lm_npc_policy_mdp_never_emits_raw_generic_verbs(generic_verb):
    """MDP forest only contains compiled verbs — never raw meta verbs."""
    from src.sim.world_loader import load_world_pack

    world = load_world_pack("worlds/tavern")
    mira_id = next(eid for eid, ent in world.spatial.entities.items() if ent.name == "Mira")
    mira = world.spatial.entities[mira_id]
    mira.meta["last_speak_tick"] = world.tick - 10

    adapter = _ScriptedAdapter(
        npc_action=SemanticAction(verb=generic_verb, actor=mira_id),
        selected_index=1,
    )
    policy = LMNpcPolicy(adapter, fallback=ReactivePolicy(), cognition_mode="lm")

    action = policy.decide(mira, world)

    assert str(action.verb).lower() != generic_verb
    assert "infer_npc_mdp_option" in (mira.meta.get("last_policy_branch") or "")


def test_lm_npc_action_flows_through_compiler():
    """A freeform verb returned by the LM is compilable by the generic path."""
    world = make_test_world()
    gid = _guard_id(world)
    guard = world.spatial.entities[gid]
    pid = _player_id(world)

    # Move the player adjacent to the guard so a contest/interaction is valid.
    player = world.spatial.entities[pid]
    player.position = guard.position.neighbors()[0]

    scripted = SemanticAction(
        verb="rebuff",
        actor=gid,
        target=pid,
        intent=IntentBlock(manner="sternly"),
    )
    policy = LMNpcPolicy(
        _ScriptedAdapter(npc_action=scripted),
        cognition_mode="lm",
    )
    action = policy.decide(guard, world)

    result = compile_action(action, world)
    assert result.valid, f"Expected valid compilation; got {result}"


# ─────────────────────────────────────────────────────────────────────────────
# LM failures surface instead of hiding behind a policy fallback
# ─────────────────────────────────────────────────────────────────────────────


def test_lm_npc_falls_back_to_reactive_on_malformed_infer():
    world = make_test_world()
    gid = _guard_id(world)
    guard = world.spatial.entities[gid]

    adapter = _ScriptedAdapter(
        npc_error=MalformedActionError("bad JSON")
    )
    policy = LMNpcPolicy(
        adapter,
        fallback=ReactivePolicy(),
        cognition_mode="lm",
    )
    action = policy.decide(guard, world)
    assert action is not None
    assert guard.meta.get("last_policy_branch") == "infer_failed_reactive"


def test_lm_npc_falls_back_when_adapter_doesnt_implement_infer_npc():
    """A custom adapter without infer_npc() should fall back to reactive."""
    world = make_test_world()
    gid = _guard_id(world)
    guard = world.spatial.entities[gid]

    class _NoNpcAdapter(LMAdapter):
        def infer(self, projection, player_intent):
            return SemanticAction(verb=ActionType.WAIT, actor=projection.focal_entity)
        # Inherits default infer_npc that raises NotImplementedError.

    policy = LMNpcPolicy(_NoNpcAdapter(), fallback=ReactivePolicy())
    action = policy.decide(guard, world)
    assert action is not None
    assert guard.meta.get("last_policy_branch") == "infer_failed_reactive"


def test_autonomous_tick_continues_when_lm_infer_fails():
    """Headless sims should survive LM failures via reactive fallback."""
    world = make_test_world()

    class _BrokenNpcAdapter(LMAdapter):
        def infer(self, projection, player_intent):
            return SemanticAction(verb=ActionType.WAIT, actor=projection.focal_entity)

        def infer_npc(self, character_sheet, projection):
            raise MalformedActionError("backend unavailable")

    loop = GameLoop(world, adapter=_BrokenNpcAdapter())
    result = loop.autonomous_tick()
    assert result.entity_results


def test_parallel_decide_survives_policy_exception():
    """A policy that raises must not abort the tick — reactive fallback applies."""
    world = make_test_world()

    class _ExplodingPolicy:
        def decide(self, entity, world):
            raise MalformedActionError("NPC LM inference failed: infer_npc failed")

    loop = GameLoop(
        world,
        adapter=_ScriptedAdapter(),
        npc_policy=_ExplodingPolicy(),
    )
    result = loop.autonomous_tick()
    assert result.entity_results
    assert any(
        e.meta.get("last_policy_branch") == "policy_exception_reactive"
        for e in world.all_entities().values()
    )


# ─────────────────────────────────────────────────────────────────────────────
# Identity safety: the LM cannot impersonate other entities
# ─────────────────────────────────────────────────────────────────────────────


def test_lm_npc_actor_is_forced_to_decided_entity():
    """If the LM returns an action whose actor field is wrong, the policy
    rewrites it so an NPC can't impersonate someone else."""
    world = make_test_world()
    gid = _guard_id(world)
    guard = world.spatial.entities[gid]
    pid = _player_id(world)

    # Adapter claims the guard's action is performed by the player.
    scripted = SemanticAction(
        verb="wave",
        actor=pid,           # WRONG — should be coerced to gid
        intent=IntentBlock(manner="cheerfully"),
    )
    policy = LMNpcPolicy(
        _ScriptedAdapter(npc_action=scripted),
        cognition_mode="lm",
    )
    action = policy.decide(guard, world)
    assert action.actor == gid, "Policy must clamp actor = decided-for entity"


# ─────────────────────────────────────────────────────────────────────────────
# Default GameLoop wiring
# ─────────────────────────────────────────────────────────────────────────────


def test_game_loop_uses_reactive_with_mock_adapter():
    """MockLMAdapter → ReactivePolicy default, so test suite stays deterministic."""
    from src.sim.lm_adapter import MockLMAdapter
    world = make_test_world()
    loop = GameLoop(world, adapter=MockLMAdapter())
    assert isinstance(loop.npc_policy, ReactivePolicy)


def test_game_loop_uses_lm_policy_with_real_adapter():
    """A non-Mock adapter → LMNpcPolicy default, with reactive fallback."""
    world = make_test_world()
    adapter = _ScriptedAdapter(
        npc_action=SemanticAction(verb=ActionType.WAIT, actor=_guard_id(world))
    )
    loop = GameLoop(world, adapter=adapter)
    assert isinstance(loop.npc_policy, LMNpcPolicy)
    assert isinstance(loop.npc_policy.fallback, ReactivePolicy)
