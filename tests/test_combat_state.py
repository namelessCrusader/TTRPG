"""Combat alertness decay and pursuit tracking."""

from src.sim.combat_state import combat_tick
from src.sim.schemas import (
    AlertnessLevel,
    EntityKind,
    Event,
    SemanticAction,
    TransitionKind,
    new_event_id,
)
from src.sim.game_loop import make_test_world


def _player_and_npc(world):
    player_id = next(
        eid for eid, e in world.all_entities().items() if e.kind == EntityKind.PLAYER
    )
    npc_id = next(
        eid
        for eid, e in world.all_entities().items()
        if e.kind == EntityKind.NPC and e.alive
    )
    return player_id, npc_id


def test_combat_decay_after_quiet_ticks():
    world = make_test_world()
    _player_id, npc_id = _player_and_npc(world)
    npc = world.spatial.entities[npc_id]
    npc.alertness = AlertnessLevel.COMBAT
    npc.meta["combat_ticks"] = 5

    transitions = combat_tick(world)
    assert npc.alertness == AlertnessLevel.HIGH
    assert "combat_target" not in npc.meta
    assert any(t.kind == TransitionKind.ENTITY_ALERTNESS_CHANGED for t in transitions)


def test_recent_attack_sets_combat_target():
    world = make_test_world()
    player_id, npc_id = _player_and_npc(world)
    npc = world.spatial.entities[npc_id]
    npc.alertness = AlertnessLevel.LOW

    world.event_log.append(
        Event(
            event_id=new_event_id(),
            tick=world.tick,
            action=SemanticAction(
                verb="attack",
                actor=player_id,
                target=npc_id,
            ),
            transitions=[],
            witnesses=[npc_id],
        )
    )

    combat_tick(world)
    assert npc.alertness == AlertnessLevel.COMBAT
    assert npc.meta.get("combat_target") == str(player_id)
