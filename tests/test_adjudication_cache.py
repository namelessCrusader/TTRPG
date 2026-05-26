"""Adjudication session cache — consistent LM outcomes within a campaign."""

from __future__ import annotations

from src.sim.adjudication_cache import cache_key, get_cached, put_cached
from src.sim.game_loop import make_test_world
from src.sim.schemas import (
    ActionZone,
    AdjudicationResult,
    EntityKind,
    GroundingResult,
    RejectionReason,
    SemanticAction,
    ValidationResult,
    WorldFact,
    WorldFactScope,
)


def _player_id(world):
    return next(
        eid for eid, e in world.spatial.entities.items()
        if e.kind == EntityKind.PLAYER
    )


def test_cache_round_trip():
    world = make_test_world()
    pid = _player_id(world)
    action = SemanticAction(
        verb="improvise",
        actor=pid,
        raw_input="jam the door with a chair",
    )
    grounding = GroundingResult(zone=ActionZone.ORPHANED)
    result = ValidationResult(
        valid=False,
        rejection_reason=RejectionReason.ACTION_TYPE_UNSUPPORTED,
    )
    key = cache_key(world, action, action.raw_input, grounding, result)
    adj = AdjudicationResult(
        ruling_text="The chair wedges the door.",
        facts=[
            WorldFact(
                claim="door wedged with furniture",
                scope=WorldFactScope.TILE,
                tags=["adjudicated", "environment"],
            ),
        ],
    )
    put_cached(world, key, adj)
    got = get_cached(world, key)
    assert got is not None
    assert got.ruling_text == adj.ruling_text
    assert got.facts[0].claim == adj.facts[0].claim


def test_cache_key_stable_for_same_intent_shape():
    world = make_test_world()
    pid = _player_id(world)
    g = GroundingResult(zone=ActionZone.ORPHANED)
    r = ValidationResult(
        valid=False,
        rejection_reason=RejectionReason.ACTION_TYPE_UNSUPPORTED,
    )
    a1 = SemanticAction(
        verb="stash",
        actor=pid,
        raw_input="stash the key under the floorboard",
    )
    a2 = SemanticAction(
        verb="stash",
        actor=pid,
        raw_input="stash the gem under the floorboard",
    )
    k1 = cache_key(world, a1, a1.raw_input, g, r)
    k2 = cache_key(world, a2, a2.raw_input, g, r)
    assert k1 == k2
