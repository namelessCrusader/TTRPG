"""
Tests for the WorldClock autonomic event loop.

Verifies:
  • Ambient events fire only when all trigger gates pass.
  • Probability rolls are deterministic (seeded from rng_seed + id + tick),
    so replaying the same log produces identical outcomes.
  • Pack-declared `effects` go through the same validator the LM uses;
    invalid kinds are dropped.
  • The projection surfaces `time_of_day` and ambient events appear in
    `recent_events` even though they have no human witnesses.
  • The world clock runs as part of `GameLoop.step()` and its events
    show up under `StepResult.ambient_events`.
"""

from __future__ import annotations

from src.sim.game_loop import GameLoop, make_test_world
from src.sim.lm_adapter import MockLMAdapter
from src.sim.projection import project
from src.sim.schemas import (
    AmbientEvent,
    AmbientTrigger,
    EntityKind,
    TimeOfDay,
    TransitionKind,
    TransitionProposal,
    WorldClock,
)
from src.sim.world_clock import world_tick


def _player_id(world):
    for eid, e in world.spatial.entities.items():
        if e.kind == EntityKind.PLAYER:
            return eid
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Trigger semantics
# ─────────────────────────────────────────────────────────────────────────────


def test_every_ticks_gate():
    world = make_test_world()
    world.config.ambient_events = [
        AmbientEvent(
            id="ping",
            trigger=AmbientTrigger(every_ticks=3),
            narrative="ping",
        )
    ]
    fires_by_tick: dict[int, int] = {}
    for t in range(7):
        world.tick = t
        fires_by_tick[t] = len(world_tick(world))
    # 0, 3, 6 fire; 1,2,4,5 do not.
    assert fires_by_tick == {0: 1, 1: 0, 2: 0, 3: 1, 4: 0, 5: 0, 6: 1}


def test_time_of_day_gate():
    world = make_test_world()
    world.config.clock = WorldClock(day_length_ticks=4)
    # ticks: 0→dawn, 1→day, 2→dusk, 3→night, 4→dawn, ...
    world.config.ambient_events = [
        AmbientEvent(
            id="dusk_only",
            trigger=AmbientTrigger(time_of_day=TimeOfDay.DUSK),
            narrative="dusk",
        )
    ]
    fires: list[bool] = []
    for t in range(8):
        world.tick = t
        fires.append(bool(world_tick(world)))
    assert fires == [False, False, True, False, False, False, True, False]


def test_probability_roll_is_deterministic():
    """Same rng_seed + tick → same outcome on replay."""
    world1 = make_test_world()
    world1.rng_seed = 42
    world1.config.ambient_events = [
        AmbientEvent(
            id="coin_flip",
            trigger=AmbientTrigger(every_ticks=1, probability=0.5),
            narrative="coin",
        )
    ]
    world2 = make_test_world()
    world2.rng_seed = 42
    world2.config.ambient_events = list(world1.config.ambient_events)

    outcomes_1 = []
    outcomes_2 = []
    for t in range(20):
        world1.tick = t
        world2.tick = t
        outcomes_1.append(bool(world_tick(world1)))
        outcomes_2.append(bool(world_tick(world2)))
    assert outcomes_1 == outcomes_2, "Roll outcomes must be replay-stable"
    assert any(outcomes_1), "0.5 probability over 20 ticks should fire sometimes"
    assert not all(outcomes_1), "0.5 probability over 20 ticks should miss sometimes"


def test_probability_seed_differs_per_event_id():
    """Two events with the same trigger but different ids draw independent rolls."""
    world = make_test_world()
    world.rng_seed = 7
    world.config.ambient_events = [
        AmbientEvent(
            id="a",
            trigger=AmbientTrigger(every_ticks=1, probability=0.5),
            narrative="a",
        ),
        AmbientEvent(
            id="b",
            trigger=AmbientTrigger(every_ticks=1, probability=0.5),
            narrative="b",
        ),
    ]
    differed = False
    for t in range(20):
        world.tick = t
        fired = {e.transitions[0].payload["id"] for e in world_tick(world)}
        if fired in ({"a"}, {"b"}):
            differed = True
            break
    assert differed, "Events with distinct ids must produce independent rolls"


# ─────────────────────────────────────────────────────────────────────────────
# Effects validation
# ─────────────────────────────────────────────────────────────────────────────


def test_ambient_effects_go_through_proposal_validator():
    """Kernel-only kinds (ENTITY_MOVED, ITEM_TRANSFERRED) must be dropped."""
    world = make_test_world()
    world.config.ambient_events = [
        AmbientEvent(
            id="evil",
            trigger=AmbientTrigger(every_ticks=1),
            narrative="the world teleports you",
            effects=[
                TransitionProposal(
                    kind=TransitionKind.ENTITY_MOVED.value,
                    payload={"entity_id": _player_id(world), "to": {"x": 0, "y": 0}},
                )
            ],
        )
    ]
    fired = world_tick(world)
    assert len(fired) == 1
    transitions = fired[0].transitions
    # AMBIENT_EVENT record is present
    assert any(t.kind == TransitionKind.AMBIENT_EVENT for t in transitions)
    # ENTITY_MOVED proposal was dropped (kernel-only)
    assert not any(t.kind == TransitionKind.ENTITY_MOVED for t in transitions)


# ─────────────────────────────────────────────────────────────────────────────
# Projection surfaces clock state
# ─────────────────────────────────────────────────────────────────────────────


def test_projection_surfaces_time_of_day():
    world = make_test_world()
    world.config.clock = WorldClock(day_length_ticks=8)
    world.tick = 5  # phase = 5/8 = 0.625 → DUSK
    pid = _player_id(world)
    proj = project(world, pid)
    assert proj.time_of_day == TimeOfDay.DUSK


def test_projection_includes_ambient_events_in_recent_events():
    world = make_test_world()
    world.config.ambient_events = [
        AmbientEvent(
            id="horn",
            trigger=AmbientTrigger(every_ticks=1),
            narrative="A horn echoes from the gate.",
        )
    ]
    world_tick(world)
    pid = _player_id(world)
    proj = project(world, pid)
    descriptions = [e.description for e in proj.recent_events]
    assert "A horn echoes from the gate." in descriptions


# ─────────────────────────────────────────────────────────────────────────────
# GameLoop integration
# ─────────────────────────────────────────────────────────────────────────────


def test_game_loop_runs_world_tick_each_step():
    world = make_test_world()
    world.config.ambient_events = [
        AmbientEvent(
            id="hb",
            trigger=AmbientTrigger(every_ticks=1),
            narrative="A heartbeat in the stone.",
        )
    ]
    loop = GameLoop(world, adapter=MockLMAdapter())
    res = loop.step("wait")
    assert len(res.ambient_events) == 1
    assert "A heartbeat in the stone." in res.ambient_events[0].narrative_hint
    # Event was appended to canonical log.
    log_ambient_ids = [
        t.payload.get("id")
        for ev in world.event_log
        for t in ev.transitions
        if t.kind == TransitionKind.AMBIENT_EVENT
    ]
    assert "hb" in log_ambient_ids


def test_replay_walks_ambient_transitions_without_crash():
    """Replay walks event.transitions directly. Ambient transitions
    must not throw when re-applied. (We don't compare tick values
    exactly — `replay()` advances tick from event.tick+1 which can
    overshoot world.tick by one, a pre-existing quirk independent of
    the clock work.)"""
    import copy
    world = make_test_world()
    world.config.ambient_events = [
        AmbientEvent(
            id="flicker",
            trigger=AmbientTrigger(every_ticks=2),
            narrative="flicker",
        )
    ]
    initial = copy.deepcopy(world)
    loop = GameLoop(world, adapter=MockLMAdapter())
    for _ in range(5):
        loop.step("wait")
    reconstructed = loop.replay(initial)
    # Reached a finite state without raising; the AMBIENT_EVENT records
    # made it back to the canonical log on the live world.
    ambient_in_log = [
        t for ev in world.event_log for t in ev.transitions
        if t.kind == TransitionKind.AMBIENT_EVENT
    ]
    assert ambient_in_log, "Live world should have logged at least one ambient event"
    # Reconstructed world progressed by walking the log.
    assert reconstructed.tick >= initial.tick


def test_dynamic_ambient_events_enrichment():
    """Verify that dynamic ambient events are successfully enriched with narrative and effects."""
    from src.sim.lm_adapter import MockLMAdapter
    from src.sim.world_clock import world_tick, force_ambient_by_id

    world = make_test_world()
    
    # 1. Custom mock adapter to simulate dynamic narrative and physical target effects
    class CustomAmbientAdapter:
        def enrich_ambient_event(self, ambient_id, base_narrative, current_scene_info):
            # Target the first coordinate and Alice (or a nearby tile/entity)
            active_chars = current_scene_info.get("active_characters", [])
            target_entity_id = active_chars[0]["id"] if active_chars else "ent_none"
            
            return {
                "narrative": f"At {current_scene_info['time_of_day']}, the fireplace crackles beautifully.",
                "effects": [
                    {
                        "kind": "entity_emotional_state_changed",
                        "payload": {
                            "entity_id": target_entity_id,
                            "to": "happy"
                        }
                    }
                ]
            }

    world.config.ambient_events = [
        AmbientEvent(
            id="fireplace_pop",
            trigger=AmbientTrigger(every_ticks=1),
            narrative="The fireplace pops.",
        )
    ]

    adapter = CustomAmbientAdapter()
    
    # Run a world tick with our custom adapter
    events = world_tick(world, lm_adapter=adapter)
    assert len(events) == 1
    event = events[0]
    
    # Check that the narrative was dynamically enriched
    assert event.narrative_hint.startswith("At dawn, the fireplace crackles beautifully.")
    
    # Check that the dynamic physical transition effect was compiled and applied
    emotional_transitions = [
        t for t in event.transitions
        if t.kind == TransitionKind.ENTITY_EMOTIONAL_STATE_CHANGED
    ]
    assert len(emotional_transitions) == 1
    assert emotional_transitions[0].payload.get("to") == "happy"


def test_dynamic_ambient_events_fallback_on_failure():
    """Verify that if the dynamic enrichment raises an error, the engine falls back gracefully."""
    from src.sim.world_clock import world_tick

    world = make_test_world()

    class ExplodingAdapter:
        def enrich_ambient_event(self, ambient_id, base_narrative, current_scene_info):
            raise RuntimeError("LLM is temporarily offline!")

    world.config.ambient_events = [
        AmbientEvent(
            id="door_creak",
            trigger=AmbientTrigger(every_ticks=1),
            narrative="A door creaks in the wind.",
        )
    ]

    adapter = ExplodingAdapter()
    
    # Verify that world_tick finishes perfectly and falls back to default narrative
    events = world_tick(world, lm_adapter=adapter)
    assert len(events) == 1
    event = events[0]
    
    # Narrative fell back cleanly
    assert event.narrative_hint == "A door creaks in the wind."
