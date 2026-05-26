"""Tests for the long-horizon NPC planner (DF-style job stacks)."""

from __future__ import annotations

from src.sim import npc_planner
from src.sim.game_loop import make_test_world
from src.sim.npc_policy import ReactivePolicy
from src.sim.schemas import (
    ActionType,
    AlertnessLevel,
    Coord,
    EntityKind,
    NpcPlan,
    PlanStatus,
    PlanStep,
    PlanStepKind,
)


def _npc(world):
    for eid, e in world.spatial.entities.items():
        if e.kind == EntityKind.NPC:
            return eid, e
    raise RuntimeError("no NPC in world")


# ── plan synthesis ────────────────────────────────────────────────────────


def test_no_plan_without_drive():
    world = make_test_world()
    _, npc = _npc(world)
    npc.drive = ""
    npc.active_plan = None
    assert npc_planner.propose_plan(npc, world) is None


def test_no_plan_when_one_already_active():
    world = make_test_world()
    _, npc = _npc(world)
    npc.drive = "patrol the eastern wall"
    npc.active_plan = NpcPlan(
        owner_id=npc.entity_id,
        name="existing",
        steps=[PlanStep(kind=PlanStepKind.SPEND_TICKS,
                        payload={"ticks": 1})],
        status=PlanStatus.ACTIVE,
    )
    assert npc_planner.propose_plan(npc, world) is None


def test_patrol_drive_synthesises_multi_step_plan():
    world = make_test_world()
    _, npc = _npc(world)
    npc.drive = "patrol the corridor"
    npc.active_plan = None
    npc.waypoints = []  # let the builder pick perimeter offsets

    plan = npc_planner.propose_plan(npc, world)
    assert plan is not None, "patrol drive should synthesise a plan"
    assert plan.status == PlanStatus.ACTIVE
    assert len(plan.steps) >= 4  # 2 waypoints minimum × (move + spend_ticks)
    assert any(s.kind == PlanStepKind.MOVE_TO for s in plan.steps)
    assert any(s.kind == PlanStepKind.SPEND_TICKS for s in plan.steps)


def test_work_drive_synthesises_short_plan():
    world = make_test_world()
    _, npc = _npc(world)
    npc.drive = "work at the forge"
    npc.active_plan = None

    plan = npc_planner.propose_plan(npc, world)
    assert plan is not None
    assert plan.steps
    # work plan should include a spend_ticks step (labour)
    assert any(s.kind == PlanStepKind.SPEND_TICKS for s in plan.steps)


def test_unknown_drive_yields_no_plan():
    world = make_test_world()
    _, npc = _npc(world)
    npc.drive = "contemplate the meaning of existence"
    npc.active_plan = None
    assert npc_planner.propose_plan(npc, world) is None


# ── interrupts ────────────────────────────────────────────────────────────


def test_combat_alertness_pauses_active_plan():
    world = make_test_world()
    _, npc = _npc(world)
    plan = NpcPlan(
        owner_id=npc.entity_id,
        name="x",
        steps=[PlanStep(kind=PlanStepKind.SPEND_TICKS, payload={"ticks": 5})],
        status=PlanStatus.ACTIVE,
    )
    npc.active_plan = plan
    npc.alertness = AlertnessLevel.COMBAT

    paused = npc_planner.check_interrupt(npc, world)
    assert paused is True
    assert plan.status == PlanStatus.PAUSED
    assert plan.meta.get("paused_reason") == "combat"


def test_low_health_pauses_active_plan():
    world = make_test_world()
    _, npc = _npc(world)
    npc.max_health = 100
    npc.health = 10
    plan = NpcPlan(
        owner_id=npc.entity_id,
        name="x",
        steps=[PlanStep(kind=PlanStepKind.SPEND_TICKS, payload={"ticks": 1})],
        status=PlanStatus.ACTIVE,
    )
    npc.active_plan = plan

    assert npc_planner.check_interrupt(npc, world) is True
    assert plan.status == PlanStatus.PAUSED
    assert plan.meta.get("paused_reason") == "low_health"


def test_paused_plan_resumes_when_safe():
    world = make_test_world()
    _, npc = _npc(world)
    npc.max_health = 100
    npc.health = 100
    npc.alertness = AlertnessLevel.LOW
    npc.conditions = {}

    plan = NpcPlan(
        owner_id=npc.entity_id,
        name="x",
        steps=[PlanStep(kind=PlanStepKind.SPEND_TICKS, payload={"ticks": 1})],
        status=PlanStatus.PAUSED,
        meta={"paused_reason": "combat"},
    )
    npc.active_plan = plan
    assert npc_planner.try_resume(npc, world) is True
    assert plan.status == PlanStatus.ACTIVE
    assert "paused_reason" not in plan.meta


# ── step emission ─────────────────────────────────────────────────────────


def test_next_step_action_emits_move_for_move_to_step():
    world = make_test_world()
    _, npc = _npc(world)
    # Pick a tile a few squares away so a path exists.
    dest = Coord(x=npc.position.x + 2, y=npc.position.y, z=npc.position.z)
    plan = NpcPlan(
        owner_id=npc.entity_id,
        name="goto",
        steps=[PlanStep(
            kind=PlanStepKind.MOVE_TO,
            payload={"tile": [dest.x, dest.y, dest.z]},
        )],
        status=PlanStatus.ACTIVE,
    )
    npc.active_plan = plan

    out = npc_planner.next_step_action(npc, world)
    assert out is not None
    action, label, weight = out
    assert action.verb == ActionType.MOVE
    assert weight >= 0.9


def test_next_step_action_emits_wait_for_spend_ticks():
    world = make_test_world()
    _, npc = _npc(world)
    plan = NpcPlan(
        owner_id=npc.entity_id,
        name="brew",
        steps=[PlanStep(
            kind=PlanStepKind.SPEND_TICKS,
            payload={"ticks": 4, "label": "brew tea"},
        )],
        status=PlanStatus.ACTIVE,
    )
    npc.active_plan = plan
    out = npc_planner.next_step_action(npc, world)
    assert out is not None
    action, _, _ = out
    assert action.verb == ActionType.WAIT


def test_next_step_returns_none_when_plan_completed():
    world = make_test_world()
    _, npc = _npc(world)
    plan = NpcPlan(
        owner_id=npc.entity_id,
        name="done",
        steps=[PlanStep(kind=PlanStepKind.SPEND_TICKS,
                        payload={"ticks": 1}, done=True)],
        status=PlanStatus.COMPLETED,
        current_step=1,
    )
    npc.active_plan = plan
    assert npc_planner.next_step_action(npc, world) is None


# ── progression ───────────────────────────────────────────────────────────


def test_spend_ticks_completes_after_payload_ticks_attempts():
    world = make_test_world()
    _, npc = _npc(world)
    plan = NpcPlan(
        owner_id=npc.entity_id,
        name="wait",
        steps=[
            PlanStep(
                kind=PlanStepKind.SPEND_TICKS,
                payload={"ticks": 3},
            ),
        ],
        status=PlanStatus.ACTIVE,
    )
    npc.active_plan = plan

    # First 2 calls should not advance; third should advance and complete.
    npc_planner.progress_plan(npc, world, [])
    assert plan.current_step == 0
    npc_planner.progress_plan(npc, world, [])
    assert plan.current_step == 0
    npc_planner.progress_plan(npc, world, [])
    assert plan.status == PlanStatus.COMPLETED


def test_step_fails_after_max_attempts():
    world = make_test_world()
    _, npc = _npc(world)
    plan = NpcPlan(
        owner_id=npc.entity_id,
        name="stuck",
        # Move-to with no resolvable target should never complete.
        steps=[PlanStep(
            kind=PlanStepKind.MOVE_TO,
            payload={"entity_id": "nonexistent_entity_id"},
        )],
        status=PlanStatus.ACTIVE,
        max_step_attempts=3,
    )
    npc.active_plan = plan

    # Step target resolution returning None makes MOVE_TO completer return
    # True (step is moot → done).  Use INTERACT against a missing target
    # instead, which never completes naturally.
    plan.steps[0] = PlanStep(
        kind=PlanStepKind.INTERACT,
        payload={"verb": "shove", "target": "nope"},
    )

    for _ in range(plan.max_step_attempts):
        npc_planner.progress_plan(npc, world, [])
    assert plan.status == PlanStatus.FAILED
    assert "exceeded" in plan.failure_reason.lower()


# ── integration: plan candidate ranks high in ReactivePolicy ──────────────


def test_reactive_policy_offers_plan_candidate():
    world = make_test_world()
    _, npc = _npc(world)
    # Make sure the NPC won't be in a combat / flee branch.
    npc.alertness = AlertnessLevel.LOW
    npc.max_health = 100
    npc.health = 100
    npc.drive = "patrol the hallway"
    npc.active_plan = None

    policy = ReactivePolicy()
    candidates = policy.generate_candidates(npc, world)
    # At least one candidate label should mention the plan.
    assert any(
        "plan:" in label.lower()
        for _action, label, _weight in candidates
    ), f"plan candidate should appear: {[c[1] for c in candidates]}"
