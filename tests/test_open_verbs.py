"""Tests for pack open_verbs.yaml loading and compilation."""

from __future__ import annotations

from pathlib import Path

from src.sim.compiler import compile_action
from src.sim.game_loop import make_test_world
from src.sim.schemas import EntityKind, SemanticAction
from src.sim.world_loader import load_world_pack


def _player_id(world):
    return next(
        eid for eid, e in world.spatial.entities.items()
        if e.kind == EntityKind.PLAYER
    )


def _npc_id(world):
    return next(
        eid for eid, e in world.spatial.entities.items()
        if e.kind == EntityKind.NPC
    )


def test_tavern_loads_open_verbs():
    pack = Path(__file__).resolve().parents[1] / "worlds" / "tavern"
    world = load_world_pack(pack)
    assert world.config.open_verbs
    ids = {r.id for r in world.config.open_verbs}
    assert "default_bribe" in ids
    assert "tavern_buy_round" in ids


def test_flatter_pack_rule_compiles():
    world = make_test_world()
    pid = _player_id(world)
    nid = _npc_id(world)
    world.config.open_verbs  # ensure field exists on test world (may be empty)

    from src.sim.schemas import ContestSpec, OpenVerbRule, TransitionProposal

    world.config.open_verbs = [
        OpenVerbRule(
            id="test_flatter",
            verbs=["flatter"],
            requires="entity",
            contest=ContestSpec(
                actor_attribute="persuasion",
                target_attribute="composure",
                difficulty=30,
            ),
            on_success=[
                TransitionProposal(
                    kind="entity_emotional_state_changed",
                    payload={"entity_id": "$target", "to": "friendly"},
                ),
            ],
        ),
    ]
    result = compile_action(
        SemanticAction(verb="flatter", actor=pid, target=str(nid)),
        world,
    )
    assert result.valid


def test_tavern_buy_round_costs_gold():
    pack = Path(__file__).resolve().parents[1] / "worlds" / "tavern"
    world = load_world_pack(pack)
    actors = list(world.spatial.entities.items())[:2]
    assert len(actors) >= 2
    (pid, player), (nid, _npc) = actors[0], actors[1]
    player.stats["gold"] = 50.0

    result = compile_action(
        SemanticAction(
            verb="buy_round",
            actor=pid,
            target=str(nid),
            raw_input="buy a round of drinks for everyone",
        ),
        world,
    )
    assert result.valid
    assert any(
        t.payload.get("stat") == "gold" and float(t.payload.get("delta", 0)) < 0
        for t in result.concrete_transitions
    )
