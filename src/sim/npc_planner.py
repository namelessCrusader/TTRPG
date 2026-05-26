"""
Long-horizon NPC planner — DF/Qud-style job stacks above ``ReactivePolicy``.

Motivation
----------

``ReactivePolicy`` is fundamentally per-tick: it scores ~28 candidate
actions and picks the best one *right now*.  That makes NPCs feel
*responsive* but not *intentional* — they never commit to a multi-tick
project that an outsider would call a "job".

In Dwarf Fortress and Caves of Qud, the world feels alive partly
because dwarves have a job stack (haul wood → place block → smooth
floor → mine ore → return to dorm) that survives across many decision
cycles, with reactive policy only kicking in when something acute
(invasion, hunger, injury) interrupts.

This module adds the same idea on top of the existing engine without
disturbing the reactive baseline.  Plans are *deterministic and
engine-owned*: the LM is not required to generate or execute them, so
``--cognition reactive_only`` runs gain real long-horizon agency.

Architecture
------------

A plan is a sequence of typed steps (``PlanStep``).  Each step has a
``kind`` that maps deterministically to a ``SemanticAction``:

  MOVE_TO     → ActionType.MOVE toward a tile or entity
  SPEND_TICKS → ActionType.WAIT for N consecutive ticks (e.g. crafting,
                resting, observing without moving)
  INTERACT    → arbitrary verb against a target
  SPEAK_TO    → ActionType.SPEAK to an entity with provided/derived line
  OBSERVE     → ActionType.OBSERVE the current tile / target

Each tick, the policy hot path calls (in order):

  1. ``check_interrupt(entity, world)``  — pause the plan if combat /
     low health / fleeing makes it unsafe to continue.
  2. ``next_step_action(entity, world)`` — translate the current
     active step into a high-weight candidate (~0.95) so the candidate
     scorer almost always picks it over routine reactive options.

After the NPC turn, ``progress_plan(entity, world, transitions)``
inspects the just-applied transitions and the world state to mark the
current step done when its completion criteria are met, then advances
to the next step (or completes the plan).

Plan synthesis from a drive is the engine's job, not the LM's, so
plans are reproducible across replays.  ``propose_plan(entity, world)``
matches the entity's ``drive`` string against a small library of
patterns and emits a corresponding plan.  Unknown drives produce no
plan — those NPCs continue to operate purely reactively, which is the
existing behaviour.

Why pattern matching instead of free LM planning
------------------------------------------------

For the same reason property-interaction rules beat LM adjudication
for the player layer: *deterministic mechanical depth compounds*, LM
improvisation does not.  When a pack author writes
``drive: "patrol the eastern wall"`` they should get the same plan
every replay; the LM may *narrate* it differently, but the structure
is engine-owned.

New drive patterns are added by extending ``_PLAN_PATTERNS`` — a YAML
pack-driven version is a natural future extension (see ``worlds/<pack>/
npc_plans.yaml`` once that loader exists).
"""

from __future__ import annotations

import logging
import random
import re
from typing import TYPE_CHECKING, Callable, Optional

from .schemas import (
    ActionType,
    AlertnessLevel,
    Coord,
    EntityId,
    EntityState,
    IntentBlock,
    NpcPlan,
    ObjectState,
    PlanStatus,
    PlanStep,
    PlanStepKind,
    SemanticAction,
    StyleBlock,
    Transition,
    TransitionKind,
    WorldState,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def propose_plan(entity: EntityState, world: WorldState) -> Optional[NpcPlan]:
    """
    Create a fresh plan from the entity's drive when there is no active plan.

    Hazard-priority pass runs first: when the entity stands on fire/smoke
    or has a flammable target nearby, a flee or douse plan is synthesised
    even if the entity's drive says nothing about hazards.  This is what
    keeps reactive_only NPCs from happily polishing the bar while the
    room burns down around them.

    Returns ``None`` when:
      - the entity already has an active or paused plan (and no hazard
        plan needs to pre-empt it),
      - the entity has no drive text to plan from,
      - no plan pattern matches the drive.

    The caller is responsible for assigning the returned plan to
    ``entity.active_plan``.  This function does not mutate state.
    """
    hazard_plan = _synthesize_hazard_plan(entity, world)
    if hazard_plan is not None and _hazard_plan_pre_empts(entity, hazard_plan):
        hazard_plan.started_tick = world.tick
        hazard_plan.last_progress_tick = world.tick
        return hazard_plan

    if entity.active_plan is not None and entity.active_plan.status in (
        PlanStatus.ACTIVE,
        PlanStatus.PAUSED,
    ):
        return None

    # Hazard interrupted and cleared the slot — do not mint a fresh baseline
    # while smoke/fire is still present even if flee synthesis failed.
    if _detect_hazard_reason(entity, world):
        return None

    drive = (entity.drive or "").strip()
    if not drive:
        return None

    pack_plan = _try_pack_plan_templates(entity, world, drive)
    if pack_plan is not None:
        pack_plan.started_tick = world.tick
        pack_plan.last_progress_tick = world.tick
        return pack_plan

    for pattern, builder in _PLAN_PATTERNS:
        if pattern.search(drive):
            try:
                plan = builder(entity, world, drive)
            except Exception as exc:  # defensive — never break the tick loop
                logger.warning(
                    "npc_planner: builder for %r failed (%s); skipped.",
                    drive[:40],
                    exc,
                )
                return None
            if plan is None or not plan.steps:
                continue
            plan.started_tick = world.tick
            plan.last_progress_tick = world.tick
            return plan
    return None


def check_interrupt(entity: EntityState, world: WorldState) -> bool:
    """
    Pause an active plan when it is unsafe to continue.

    Returns ``True`` when the plan was just paused (caller may want to
    log this).  Returns ``False`` when no interrupt applies, or when
    there is no plan.

    Interrupt rules (kept deliberately small; new ones add here):
      - alertness >= COMBAT     → combat takes priority
      - health < 25% of max     → self-preservation takes priority
      - "fleeing" condition     → already in panic; cannot pursue project
      - smoke at tile / room    → hazard pre-empts non-hazard plans
      - fire on/adjacent tile   → hazard pre-empts non-hazard plans

    Hazard interrupts push the prior plan onto ``entity.meta["plan_stack"]``
    so it can be resumed after the danger clears.  Hazard-response plans
    (flee_hazard / douse_fire) never self-interrupt.
    """
    plan = entity.active_plan
    if plan is None or plan.status != PlanStatus.ACTIVE:
        return False

    if entity.alertness == AlertnessLevel.COMBAT:
        plan.status = PlanStatus.PAUSED
        plan.meta["paused_reason"] = "combat"
        return True

    max_health = max(int(entity.max_health or 100), 1)
    if entity.health < 0.25 * max_health:
        plan.status = PlanStatus.PAUSED
        plan.meta["paused_reason"] = "low_health"
        return True

    if "fleeing" in (entity.conditions or {}):
        plan.status = PlanStatus.PAUSED
        plan.meta["paused_reason"] = "fleeing"
        return True

    # Hazards: smoke at tile / atmosphere, fire on or beside the tile.
    if not _is_hazard_response_plan(plan):
        reason = _detect_hazard_reason(entity, world)
        if reason:
            plan.status = PlanStatus.PAUSED
            plan.meta["paused_reason"] = reason
            _push_paused_to_stack(entity, plan)
            entity.active_plan = None
            return True

    return False


def try_resume(entity: EntityState, world: WorldState) -> bool:
    """
    Resume a paused plan when the interrupt condition has cleared.

    Also recycles finished hazard plans: a hazard-response plan in
    ``COMPLETED`` or ``FAILED`` state is dropped automatically and the
    stack is popped (or cleared) so the NPC returns to baseline once
    the danger passes — without this, a failed flee plan would pin the
    entity to a useless target forever.

    Returns ``True`` when the active plan slot was changed (resumed,
    swapped from stack, or cleared so the next ``propose_plan`` can run).
    """
    plan = entity.active_plan

    # Drop spent hazard plans so the entity is free to resume baseline.
    # Also drop hazard plans that are still active but no longer relevant
    # because the danger has already passed — otherwise the entity would
    # chase a flee waypoint long after the smoke cleared.
    if plan is not None and _is_hazard_response_plan(plan):
        if plan.status in (PlanStatus.COMPLETED, PlanStatus.FAILED):
            entity.active_plan = None
            plan = None
        elif plan.status == PlanStatus.ACTIVE and not _detect_hazard_reason(entity, world):
            plan.status = PlanStatus.ABANDONED
            entity.active_plan = None
            plan = None

    if plan is None:
        popped = _pop_stack_if_safe(entity, world)
        if popped is not None:
            entity.active_plan = popped
            return True
        return False

    if plan.status != PlanStatus.PAUSED:
        return False

    max_health = max(int(entity.max_health or 100), 1)
    reason = plan.meta.get("paused_reason", "")
    safe_general = (
        entity.alertness != AlertnessLevel.COMBAT
        and entity.health >= 0.5 * max_health
        and "fleeing" not in (entity.conditions or {})
    )
    hazard_clear = (
        reason not in _HAZARD_REASONS or not _detect_hazard_reason(entity, world)
    )
    if safe_general and hazard_clear:
        plan.status = PlanStatus.ACTIVE
        plan.meta.pop("paused_reason", None)
        return True
    return False


def next_step_action(
    entity: EntityState,
    world: WorldState,
) -> Optional[tuple[SemanticAction, str, float]]:
    """
    Return ``(action, label, weight)`` advancing the entity's active plan,
    or ``None`` when there is nothing actionable right now.

    Weight is intentionally high (0.95) so the candidate scorer in
    ``ReactivePolicy`` will pick this over routine reactive options
    unless something with weight > 0.95 fires (combat, flee).
    """
    plan = entity.active_plan
    if plan is None or plan.status != PlanStatus.ACTIVE:
        return None
    if plan.current_step >= len(plan.steps):
        return None

    step = plan.steps[plan.current_step]
    if step.done:
        return None

    eid = entity.entity_id
    emitter = _STEP_EMITTERS.get(step.kind)
    if emitter is None:
        return None

    try:
        action = emitter(entity, world, step)
    except Exception as exc:
        logger.warning(
            "npc_planner: step emitter for %s failed (%s); skipping.",
            step.kind,
            exc,
        )
        return None
    if action is None:
        return None

    label = (
        f"plan: {plan.name or plan.drive_source[:30]} "
        f"[step {plan.current_step + 1}/{len(plan.steps)}: {step.label or step.kind.value}]"
    )
    return action, label, 0.95


def progress_plan(
    entity: EntityState,
    world: WorldState,
    recent_transitions: list[Transition],
) -> bool:
    """
    Inspect just-applied transitions and the current world state and
    advance the plan's current step if completion criteria are met.

    Returns ``True`` when the plan made progress (a step advanced or the
    plan completed/failed).  Caller may want to record/log progress.
    """
    plan = entity.active_plan
    if plan is None or plan.status != PlanStatus.ACTIVE:
        return False
    if plan.current_step >= len(plan.steps):
        plan.status = PlanStatus.COMPLETED
        return True

    step = plan.steps[plan.current_step]
    step.attempts += 1

    # Per-kind completion check.
    completer = _STEP_COMPLETERS.get(step.kind)
    completed = bool(completer and completer(entity, world, step, recent_transitions))

    if completed:
        step.done = True
        plan.current_step += 1
        plan.last_progress_tick = world.tick
        if plan.current_step >= len(plan.steps):
            plan.status = PlanStatus.COMPLETED
        return True

    # Failure detection — too many attempts on one step.
    if step.attempts >= plan.max_step_attempts:
        plan.status = PlanStatus.FAILED
        plan.failure_reason = (
            f"step {plan.current_step + 1} ({step.kind.value}) "
            f"exceeded {plan.max_step_attempts} attempts"
        )
        return True

    return False


# ---------------------------------------------------------------------------
# Step emitters — turn a step into a SemanticAction
# ---------------------------------------------------------------------------


def _emit_move_to(
    entity: EntityState,
    world: WorldState,
    step: PlanStep,
) -> Optional[SemanticAction]:
    from .spatial import find_path
    from .hazard_avoidance import find_hazard_aware_path

    target_pos = _step_target_position(entity, world, step)
    if target_pos is None:
        return None

    grid = world.spatial
    if entity.position.manhattan(target_pos) <= 1:
        # Already there — the completer will mark this step done.
        return SemanticAction(
            verb=ActionType.WAIT,
            actor=entity.entity_id,
            intent=IntentBlock(rationale=step.label or "arrived at plan waypoint"),
            raw_input=f"[npc_plan:arrived:{step.kind.value}]",
        )

    hazard_flee = (
        entity.active_plan is not None
        and entity.active_plan.meta.get("hazard_response")
    )
    if hazard_flee:
        path = find_hazard_aware_path(
            grid, entity.position, target_pos, world, max_steps=512,
        )
    else:
        path = find_path(grid, entity.position, target_pos, max_steps=24)
    if not path:
        return None

    return SemanticAction(
        verb=ActionType.MOVE,
        actor=entity.entity_id,
        target=path[0],
        intent=IntentBlock(rationale=step.label or "advance plan"),
        raw_input=f"[npc_plan:move:{step.label or step.kind.value}]",
    )


def _emit_spend_ticks(
    entity: EntityState,
    world: WorldState,
    step: PlanStep,
) -> Optional[SemanticAction]:
    return SemanticAction(
        verb=ActionType.WAIT,
        actor=entity.entity_id,
        intent=IntentBlock(rationale=step.label or "work on plan"),
        raw_input=f"[npc_plan:wait:{step.label or 'tick'}]",
    )


def _emit_interact(
    entity: EntityState,
    world: WorldState,
    step: PlanStep,
) -> Optional[SemanticAction]:
    verb = str(step.payload.get("verb") or "use")
    target = step.payload.get("target")
    if target is None:
        return None
    # Targets may be:
    #   * Coord       — already structured (preferred for tile targets)
    #   * dict {x,y,z} / list[x,y,z]  — convert to Coord
    #   * str "x,y" / "x,y,z"         — convert to Coord
    #   * str entity / object id      — pass through
    resolved: Any
    if isinstance(target, Coord):
        resolved = target
    elif isinstance(target, dict) and {"x", "y"}.issubset(target.keys()):
        try:
            resolved = Coord(
                x=int(target["x"]),
                y=int(target["y"]),
                z=int(target.get("z", 0)),
            )
        except (TypeError, ValueError):
            resolved = str(target)
    elif isinstance(target, (list, tuple)) and len(target) >= 2:
        try:
            z = int(target[2]) if len(target) > 2 else 0
            resolved = Coord(x=int(target[0]), y=int(target[1]), z=z)
        except (TypeError, ValueError):
            resolved = str(target)
    elif isinstance(target, str) and "," in target:
        try:
            parts = target.split(",")
            z = int(parts[2]) if len(parts) >= 3 else (
                entity.position.z if entity.position else 0
            )
            resolved = Coord(x=int(parts[0]), y=int(parts[1]), z=z)
        except (TypeError, ValueError):
            resolved = str(target)
    else:
        resolved = str(target)
    return SemanticAction(
        verb=verb,
        actor=entity.entity_id,
        target=resolved,
        intent=IntentBlock(rationale=step.label or "advance plan"),
        raw_input=f"[npc_plan:interact:{verb}]",
    )


def _emit_speak_to(
    entity: EntityState,
    world: WorldState,
    step: PlanStep,
) -> Optional[SemanticAction]:
    target = step.payload.get("entity_id")
    if not target:
        return None
    line = step.payload.get("line") or step.label or "advance plan"
    return SemanticAction(
        verb=ActionType.SPEAK,
        actor=entity.entity_id,
        target=str(target),
        intent=IntentBlock(
            rationale=step.label or "plan dialogue",
            manner=str(line)[:80],
        ),
        raw_input=f"[npc_plan:speak:{str(line)[:30]}]",
    )


def _emit_observe(
    entity: EntityState,
    world: WorldState,
    step: PlanStep,
) -> Optional[SemanticAction]:
    target_pos = _step_target_position(entity, world, step)
    target: object
    if target_pos is not None:
        target = target_pos
    else:
        eid = step.payload.get("entity_id")
        if not eid:
            return None
        target = str(eid)
    return SemanticAction(
        verb=ActionType.OBSERVE,
        actor=entity.entity_id,
        target=target,
        intent=IntentBlock(rationale=step.label or "observe per plan"),
        raw_input=f"[npc_plan:observe]",
    )


_STEP_EMITTERS: dict[PlanStepKind, Callable[..., Optional[SemanticAction]]] = {
    PlanStepKind.MOVE_TO: _emit_move_to,
    PlanStepKind.SPEND_TICKS: _emit_spend_ticks,
    PlanStepKind.INTERACT: _emit_interact,
    PlanStepKind.SPEAK_TO: _emit_speak_to,
    PlanStepKind.OBSERVE: _emit_observe,
}


# ---------------------------------------------------------------------------
# Step completers — decide whether the current step is finished
# ---------------------------------------------------------------------------


def _complete_move_to(
    entity: EntityState,
    world: WorldState,
    step: PlanStep,
    transitions: list[Transition],
) -> bool:
    target_pos = _step_target_position(entity, world, step)
    if target_pos is None:
        return True  # target gone — step is moot, treat as done
    return entity.position.manhattan(target_pos) <= 1


def _complete_spend_ticks(
    entity: EntityState,
    world: WorldState,
    step: PlanStep,
    transitions: list[Transition],
) -> bool:
    # SPEND_TICKS always emits a WAIT action, so each tick the step is
    # the active step is one tick of "spending".  ``step.attempts`` is
    # incremented by ``progress_plan`` for the active step.
    needed = int(step.payload.get("ticks", 1) or 1)
    return step.attempts >= needed


def _complete_interact(
    entity: EntityState,
    world: WorldState,
    step: PlanStep,
    transitions: list[Transition],
) -> bool:
    # Treat any transition naming the step's target as completion.
    target = str(step.payload.get("target") or "")
    if not target:
        return True
    for t in transitions or []:
        p = t.payload or {}
        if any(str(p.get(k)) == target for k in ("entity_id", "target", "object_id")):
            return True
    return False


def _complete_speak_to(
    entity: EntityState,
    world: WorldState,
    step: PlanStep,
    transitions: list[Transition],
) -> bool:
    target = str(step.payload.get("entity_id") or "")
    if not target:
        return True
    for t in transitions or []:
        if t.kind != TransitionKind.DIALOGUE_SPOKEN:
            continue
        p = t.payload or {}
        if str(p.get("actor", "")) == str(entity.entity_id):
            return True
    return False


def _complete_observe(
    entity: EntityState,
    world: WorldState,
    step: PlanStep,
    transitions: list[Transition],
) -> bool:
    # Single-tick observation always completes on first attempt.
    return step.attempts >= 1


_STEP_COMPLETERS: dict[
    PlanStepKind, Callable[[EntityState, WorldState, PlanStep, list[Transition]], bool]
] = {
    PlanStepKind.MOVE_TO: _complete_move_to,
    PlanStepKind.SPEND_TICKS: _complete_spend_ticks,
    PlanStepKind.INTERACT: _complete_interact,
    PlanStepKind.SPEAK_TO: _complete_speak_to,
    PlanStepKind.OBSERVE: _complete_observe,
}


# ---------------------------------------------------------------------------
# Plan template library
# ---------------------------------------------------------------------------


_PATROL_RE = re.compile(r"\b(patrol|walk|pace)\b", re.I)
_TEND_RE = re.compile(r"\b(tend|watch|guard|mind|oversee|keep an eye on)\b", re.I)
_DELIVER_RE = re.compile(
    r"\b(deliver|bring|carry|take|fetch)\b.{0,40}\bto\b",
    re.I,
)
_WORK_RE = re.compile(
    r"\b(work|labor|craft|build|repair|fix|smith|brew|sweep|clean)\b",
    re.I,
)
_COOK_RE = re.compile(r"\b(cook|prepare food|stew|roast|bake|kitchen)\b", re.I)
_SERVE_RE = re.compile(
    r"\b(serve|pour|tend the bar|keep the ale|ale flowing|bar running|"
    r"work the bar|bartend)\b",
    re.I,
)
_FLEE_RE = re.compile(
    r"\b(flee|run|escape|get out|reach the (exit|door)|forget the song|stay alive)\b",
    re.I,
)
_DOUSE_RE = re.compile(
    r"\b(douse|put out|extinguish|quench|smother|smother the fire)\b",
    re.I,
)
_INVESTIGATE_RE = re.compile(
    r"\b(investigate|find out|check on|look into|inspect|see what)\b",
    re.I,
)


# ── Hazard detection (smoke, fire) ────────────────────────────────────────


_HAZARD_REASONS: frozenset[str] = frozenset({"smoke", "fire_near"})


def _entity_smoke_amount(entity: EntityState, world: WorldState) -> float:
    """Read smoke concentration at the entity's tile + atmosphere."""
    pos = entity.position
    if pos is None:
        return 0.0
    grid = world.spatial
    try:
        tile = grid.tile_at(pos)
    except Exception:
        return 0.0
    from .fields import (
        MEDIA_TILE_AIR,
        get_layer,
        region_atmosphere_amount,
    )
    air = get_layer(tile.env, MEDIA_TILE_AIR, legacy_tile=tile)
    local = float((air.get("smoke") or {}).get("amount", 0.0))
    atmospheric = float(region_atmosphere_amount(world, "smoke"))
    return max(local, atmospheric)


def _tile_on_fire(tile: Optional[Any]) -> bool:
    if tile is None:
        return False
    return any(t.lower() == "on_fire" for t in (tile.tags or []))


def _fire_adjacent(entity: EntityState, world: WorldState) -> bool:
    """True when the entity's tile or any 8-neighbour tile is on fire,
    or any flammable object on those tiles carries the on_fire tag."""
    pos = entity.position
    if pos is None:
        return False
    grid = world.spatial
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            c = Coord(x=pos.x + dx, y=pos.y + dy, z=pos.z)
            if not grid.is_in_bounds(c):
                continue
            try:
                tile = grid.tile_at(c)
            except Exception:
                continue
            if _tile_on_fire(tile):
                return True
    for obj in grid.objects.values():
        if obj.position is None or "on_fire" not in (obj.tags or []):
            continue
        if obj.position.chebyshev(pos) <= 1:
            return True
    return False


def _detect_hazard_reason(entity: EntityState, world: WorldState) -> str:
    """Return the strongest hazard reason for an interrupt, or ''."""
    if _entity_smoke_amount(entity, world) >= 12.0:
        return "smoke"
    if _fire_adjacent(entity, world):
        return "fire_near"
    return ""


def _is_hazard_response_plan(plan: NpcPlan) -> bool:
    return bool(plan.meta.get("hazard_response"))


def _hazard_plan_pre_empts(entity: EntityState, hazard_plan: NpcPlan) -> bool:
    """Decide whether a new hazard plan should pre-empt the current one.

    A hazard plan does NOT pre-empt:
      - another active hazard plan of the same kind,
      - a paused plan that already has a hazard reason on the stack.
    """
    cur = entity.active_plan
    if cur is None:
        return True
    if _is_hazard_response_plan(cur) and cur.meta.get("hazard_kind") == hazard_plan.meta.get("hazard_kind"):
        return False
    return True


def _push_paused_to_stack(entity: EntityState, plan: NpcPlan) -> None:
    stack = entity.meta.setdefault("plan_stack", [])
    if not isinstance(stack, list):
        stack = []
        entity.meta["plan_stack"] = stack
    if len(stack) >= 4:  # bound stack depth defensively
        stack.pop(0)
    stack.append(plan.model_dump(mode="json"))


def _pop_stack_if_safe(entity: EntityState, world: WorldState) -> Optional[NpcPlan]:
    stack = entity.meta.get("plan_stack")
    if not isinstance(stack, list) or not stack:
        return None
    if _detect_hazard_reason(entity, world):
        return None
    if entity.alertness == AlertnessLevel.COMBAT:
        return None
    raw = stack.pop()
    entity.meta["plan_stack"] = stack
    try:
        plan = NpcPlan.model_validate(raw)
    except Exception as exc:
        logger.warning("npc_planner: failed to restore stacked plan (%s)", exc)
        return None
    plan.status = PlanStatus.ACTIVE
    plan.meta.pop("paused_reason", None)
    return plan


def _synthesize_hazard_plan(entity: EntityState, world: WorldState) -> Optional[NpcPlan]:
    """Build a flee or douse plan when hazards are present.

    Hazard plans get a generous ``max_step_attempts`` because they often
    co-exist with the reactive flee branch in ``ReactivePolicy`` (which
    short-circuits with a higher-weight FLEE action while smoke is
    present).  The plan's own MOVE_TO step is not actually executed in
    that window, so the attempts counter would otherwise hit the
    default cap and the plan would FAIL while the entity is still in
    danger.  ``try_resume`` recycles spent hazard plans so they don't
    pin the NPC after the danger passes.
    """
    reason = _detect_hazard_reason(entity, world)
    if not reason:
        return None
    plan: Optional[NpcPlan] = None
    kind = "flee"
    # If the entity can douse and there's an actual fire (not just smoke),
    # prefer douse — heroic NPCs help instead of running.
    if _can_attempt_douse(entity, world):
        douse = _build_douse_fire_plan(entity, world, drive=entity.drive or "")
        if douse is not None:
            plan = douse
            kind = "douse"
    if plan is None:
        plan = _build_flee_hazard_plan(entity, world, drive=entity.drive or "")
        kind = "flee"
    if plan is None:
        return None
    plan.meta["hazard_response"] = True
    plan.meta["hazard_kind"] = kind
    plan.meta["hazard_reason"] = reason
    plan.max_step_attempts = 32
    return plan


def _can_attempt_douse(entity: EntityState, world: WorldState) -> bool:
    """True when the entity has the means / role to douse.

    Either a known douse spell op, or an occupation/role that implies
    fire-fighting reflex (bartender, cook, guard, watchman, veteran).
    """
    if "QUENCH" in (entity.known_spell_ops or []):
        return True
    if any("douse" in s.lower() for s in (entity.inscribed_spells or {}).values()):
        return True
    role = f"{entity.role or ''} {entity.occupation or ''}".lower()
    if any(k in role for k in ("bartender", "cook", "guard", "watch", "veteran", "soldier")):
        return True
    return False


def _step_target_position(
    entity: EntityState,
    world: WorldState,
    step: PlanStep,
) -> Optional[Coord]:
    """Resolve a step's payload to a concrete Coord, or None."""
    tile = step.payload.get("tile")
    if tile is not None:
        if isinstance(tile, Coord):
            return tile
        if isinstance(tile, dict):
            try:
                return Coord(
                    x=int(tile.get("x", 0)),
                    y=int(tile.get("y", 0)),
                    z=int(tile.get("z", 0)),
                )
            except (TypeError, ValueError):
                return None
        if isinstance(tile, (list, tuple)) and len(tile) >= 2:
            try:
                z = int(tile[2]) if len(tile) > 2 else 0
                return Coord(x=int(tile[0]), y=int(tile[1]), z=z)
            except (TypeError, ValueError):
                return None
    eid = step.payload.get("entity_id")
    if eid:
        ent = world.spatial.entities.get(EntityId(str(eid)))
        if ent is not None and ent.alive and ent.position is not None:
            return ent.position
    return None


def _build_patrol_plan(
    entity: EntityState,
    world: WorldState,
    drive: str,
) -> Optional[NpcPlan]:
    """A repeating walk-cycle.

    Uses entity.waypoints when present; otherwise generates a small
    triangle around the current position.  After visiting the last
    waypoint, the plan completes — synthesise_plan_if_needed builds
    a fresh one next turn, producing a perpetual patrol.
    """
    waypoints: list[Coord] = list(entity.waypoints or [])
    if not waypoints and entity.position is not None:
        px, py = entity.position.x, entity.position.y
        offsets = [(2, 0), (2, 2), (0, 2), (0, 0)]
        for dx, dy in offsets:
            cand = Coord(x=px + dx, y=py + dy, z=entity.position.z)
            if world.spatial.is_in_bounds(cand) and not world.spatial.is_solid(cand):
                waypoints.append(cand)

    if len(waypoints) < 2:
        return None

    steps: list[PlanStep] = []
    for i, wp in enumerate(waypoints):
        steps.append(PlanStep(
            kind=PlanStepKind.MOVE_TO,
            payload={"tile": [wp.x, wp.y, wp.z]},
            label=f"patrol leg {i + 1}",
        ))
        steps.append(PlanStep(
            kind=PlanStepKind.SPEND_TICKS,
            payload={"ticks": 2, "label": "watch surroundings"},
            label="pause at waypoint",
        ))

    return NpcPlan(
        owner_id=entity.entity_id,
        name=f"patrol ({len(waypoints)} waypoints)",
        drive_source=drive,
        steps=steps,
    )


def _build_tend_plan(
    entity: EntityState,
    world: WorldState,
    drive: str,
) -> Optional[NpcPlan]:
    """Stay near a target entity / tile and observe it periodically."""
    target_ent = _resolve_named_entity(world, drive)
    target_pos: Optional[Coord] = None
    if target_ent is not None and target_ent.position is not None:
        target_pos = target_ent.position
    elif entity.position is not None:
        target_pos = entity.position

    if target_pos is None:
        return None

    payload_tile = {"tile": [target_pos.x, target_pos.y, target_pos.z]}
    steps = [
        PlanStep(
            kind=PlanStepKind.MOVE_TO,
            payload=payload_tile,
            label=f"approach {target_ent.name if target_ent else 'post'}",
        ),
        PlanStep(
            kind=PlanStepKind.SPEND_TICKS,
            payload={"ticks": 4, "label": "tend"},
            label="tend",
        ),
        PlanStep(
            kind=PlanStepKind.OBSERVE,
            payload=payload_tile,
            label="check the area",
        ),
    ]
    return NpcPlan(
        owner_id=entity.entity_id,
        name=f"tend {target_ent.name if target_ent else 'post'}",
        drive_source=drive,
        steps=steps,
    )


def _build_work_plan(
    entity: EntityState,
    world: WorldState,
    drive: str,
) -> Optional[NpcPlan]:
    """Move to a workplace tile (or current tile) and spend ticks there."""
    if entity.position is None:
        return None
    p = entity.position
    steps = [
        PlanStep(
            kind=PlanStepKind.MOVE_TO,
            payload={"tile": [p.x, p.y, p.z]},
            label="at workplace",
        ),
        PlanStep(
            kind=PlanStepKind.SPEND_TICKS,
            payload={"ticks": 6, "label": "labour"},
            label="labour",
        ),
    ]
    return NpcPlan(
        owner_id=entity.entity_id,
        name="work",
        drive_source=drive,
        steps=steps,
    )


def _build_deliver_plan(
    entity: EntityState,
    world: WorldState,
    drive: str,
) -> Optional[NpcPlan]:
    """Try to recognise 'deliver X to Y' shape; fall back to None on parse failure."""
    target_ent = _resolve_named_entity(world, drive)
    if target_ent is None or target_ent.position is None:
        return None
    p = target_ent.position
    steps = [
        PlanStep(
            kind=PlanStepKind.MOVE_TO,
            payload={"tile": [p.x, p.y, p.z]},
            label=f"head to {target_ent.name}",
        ),
        PlanStep(
            kind=PlanStepKind.SPEAK_TO,
            payload={
                "entity_id": str(target_ent.entity_id),
                "line": "I have what you asked for.",
            },
            label=f"deliver to {target_ent.name}",
        ),
    ]
    return NpcPlan(
        owner_id=entity.entity_id,
        name=f"deliver to {target_ent.name}",
        drive_source=drive,
        steps=steps,
    )


def _find_nearest_object_with_any_tag(
    entity: EntityState,
    world: WorldState,
    tags: set[str],
    *,
    max_dist: int = 24,
) -> Optional[ObjectState]:
    if entity.position is None:
        return None
    grid = world.spatial
    best: Optional[tuple[int, ObjectState]] = None
    for obj in grid.objects.values():
        if obj.position is None:
            continue
        if not tags.intersection({t.lower() for t in (obj.tags or [])}):
            continue
        d = entity.position.manhattan(obj.position)
        if d > max_dist:
            continue
        if best is None or d < best[0]:
            best = (d, obj)
    return best[1] if best else None


def _find_safe_tile(
    entity: EntityState,
    world: WorldState,
    *,
    search_radius: int = 6,
) -> Optional[Coord]:
    """Pick a passable tile with low smoke and no fire, preferably nearer
    to an exit / door.

    This is *best-effort*: when the whole room is smoke-saturated and no
    truly clear tile exists, the function falls back to the candidate
    with the lowest smoke score — flee plans should always produce a
    move target so NPCs at least step away from on-fire objects and
    drift toward exits, not freeze in panic.

    Greedy scan; cheap enough for tavern-scale grids (≤ 200 tiles).
    """
    if entity.position is None:
        return None
    grid = world.spatial
    from .fields import (
        MEDIA_TILE_AIR,
        get_layer,
    )
    px, py, pz = entity.position.x, entity.position.y, entity.position.z

    # Search at the entity's z AND the standard walking floor (z=1 for
    # multi-level voxel worlds) so NPCs who slipped between z levels can
    # still find an escape route.  Floor z=0 is often solid stone.
    z_candidates: list[int] = [pz]
    if grid.depth > 1 and 1 not in z_candidates:
        z_candidates.append(1)

    # Collect exit/door coords once so we can bias flee paths toward them.
    exit_coords: list[Coord] = []
    from .schemas import TerrainType

    for z_search in z_candidates:
        for ty in range(grid.height):
            for tx in range(grid.width):
                c = Coord(x=tx, y=ty, z=z_search)
                if not grid.is_in_bounds(c):
                    continue
                try:
                    tile = grid.tile_at(c)
                except Exception:
                    continue
                if tile.terrain in (
                    TerrainType.DOOR_OPEN,
                    TerrainType.DOOR_CLOSED,
                    TerrainType.STAIRS_UP,
                    TerrainType.STAIRS_DOWN,
                ) or any(t.lower() in ("exit", "door") for t in (tile.tags or [])):
                    exit_coords.append(c)

    def _exit_distance(c: Coord) -> float:
        if not exit_coords:
            return 0.0
        return float(min(c.manhattan(e) for e in exit_coords))

    best_safe: Optional[tuple[float, Coord]] = None
    best_any: Optional[tuple[float, Coord]] = None

    for z_search in z_candidates:
        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                c = Coord(x=px + dx, y=py + dy, z=z_search)
                if c == entity.position:
                    continue
                if not grid.is_in_bounds(c) or not grid.is_passable(c):
                    continue
                try:
                    tile = grid.tile_at(c)
                except Exception:
                    continue
                if _tile_on_fire(tile):
                    continue
                air = get_layer(tile.env, MEDIA_TILE_AIR, legacy_tile=tile)
                smoke = float((air.get("smoke") or {}).get("amount", 0.0))
                exit_dist = _exit_distance(c)
                # Prefer farther from current spot, closer to exits, lower smoke.
                score = (
                    entity.position.manhattan(c) * 1.0
                    - exit_dist * 1.5
                    - smoke * 0.25
                )
                if smoke < 8.0:
                    if best_safe is None or score > best_safe[0]:
                        best_safe = (score, c)
                if best_any is None or score > best_any[0]:
                    best_any = (score, c)

    if best_safe is not None:
        return best_safe[1]
    if best_any is not None:
        return best_any[1]
    return None


def _build_flee_hazard_plan(
    entity: EntityState,
    world: WorldState,
    drive: str,
) -> Optional[NpcPlan]:
    """Move toward an exit through smoke if needed; observe to re-evaluate."""
    from .hazard_avoidance import flee_plan_destination

    safe = flee_plan_destination(entity, world)
    if safe is None:
        return None
    steps = [
        PlanStep(
            kind=PlanStepKind.MOVE_TO,
            payload={"tile": [safe.x, safe.y, safe.z]},
            label="flee to safer ground",
        ),
        PlanStep(
            kind=PlanStepKind.OBSERVE,
            payload={"tile": [safe.x, safe.y, safe.z]},
            label="check the room from a safer spot",
        ),
    ]
    return NpcPlan(
        owner_id=entity.entity_id,
        name="flee hazard",
        drive_source=drive,
        steps=steps,
    )


def _find_on_fire_target(
    entity: EntityState,
    world: WorldState,
) -> Optional[tuple[Coord, str, Optional[str]]]:
    """Return ``(position, label, object_id)`` of the nearest fire.

    Object id is non-None when the fire is attached to an object the
    planner can target directly; None when the fire is a tile-only
    blaze (e.g. spilled oil on the ground).  The douse INTERACT step
    uses the object id when available so the verb compiler resolves
    the target cleanly.
    """
    if entity.position is None:
        return None
    grid = world.spatial
    best: Optional[tuple[int, Coord, str, Optional[str]]] = None
    for obj in grid.objects.values():
        if obj.position is None or "on_fire" not in (obj.tags or []):
            continue
        d = entity.position.manhattan(obj.position)
        if best is None or d < best[0]:
            best = (d, obj.position, obj.name or "fire", str(obj.object_id))
    # Also consider tiles tagged on_fire (e.g. spilled oil ignited).
    px, py, pz = entity.position.x, entity.position.y, entity.position.z
    for dx in range(-6, 7):
        for dy in range(-6, 7):
            c = Coord(x=px + dx, y=py + dy, z=pz)
            if not grid.is_in_bounds(c):
                continue
            try:
                tile = grid.tile_at(c)
            except Exception:
                continue
            if _tile_on_fire(tile):
                d = entity.position.manhattan(c)
                if best is None or d < best[0]:
                    best = (d, c, "burning ground", None)
    if best is None:
        return None
    return best[1], best[2], best[3]


def _build_douse_fire_plan(
    entity: EntityState,
    world: WorldState,
    drive: str,
) -> Optional[NpcPlan]:
    """Move adjacent to the nearest fire, then INTERACT verb=douse.

    Resolution order inside the compiler:
      1. Kernel `_compile_extinguish` (registry.py) — handles the case
         where the actor is holding a fire_suppressant + liquid.
      2. Pack `property_interactions` rules — e.g.
         ``worlds/voxel_tavern/property_interactions.yaml`` lets staff
         and any liquid-holder douse without an explicit tool.
      3. Pack `verb_templates` / default open_verbs — narrative fallback.
      4. Adjudication only if all of the above reject the compile.

    The reactive_only showcase relies on (2) firing so that an empty-
    handed bartender can put out the kitchen oil flask without an LM.
    """
    target = _find_on_fire_target(entity, world)
    if target is None:
        return None
    pos, label, object_id = target
    interact_target = object_id if object_id else {"x": pos.x, "y": pos.y, "z": pos.z}
    steps = [
        PlanStep(
            kind=PlanStepKind.MOVE_TO,
            payload={"tile": [pos.x, pos.y, pos.z]},
            label=f"move toward {label}",
        ),
        PlanStep(
            kind=PlanStepKind.INTERACT,
            payload={"verb": "douse", "target": interact_target},
            label=f"douse {label}",
        ),
        PlanStep(
            kind=PlanStepKind.OBSERVE,
            payload={"tile": [pos.x, pos.y, pos.z]},
            label="check the fire is out",
        ),
    ]
    return NpcPlan(
        owner_id=entity.entity_id,
        name=f"douse {label}",
        drive_source=drive,
        steps=steps,
    )


def _build_cook_plan(
    entity: EntityState,
    world: WorldState,
    drive: str,
) -> Optional[NpcPlan]:
    """Go to a heat source / cookfire and spend ticks preparing food."""
    obj = _find_nearest_object_with_any_tag(
        entity, world, {"heat_source", "cooking", "hearth"},
    )
    if obj is None or obj.position is None:
        return None
    p = obj.position
    steps = [
        PlanStep(
            kind=PlanStepKind.MOVE_TO,
            payload={"tile": [p.x, p.y, p.z]},
            label=f"to {obj.name or 'cookfire'}",
        ),
        PlanStep(
            kind=PlanStepKind.SPEND_TICKS,
            payload={"ticks": 5, "label": "cook"},
            label="prepare food",
        ),
        PlanStep(
            kind=PlanStepKind.OBSERVE,
            payload={"tile": [p.x, p.y, p.z]},
            label="check the pot",
        ),
    ]
    return NpcPlan(
        owner_id=entity.entity_id,
        name="cook",
        drive_source=drive,
        steps=steps,
    )


def _build_serve_plan(
    entity: EntityState,
    world: WorldState,
    drive: str,
) -> Optional[NpcPlan]:
    """Bartender circuit: move to bar, pour, then move toward a nearby patron."""
    bar = _find_nearest_object_with_any_tag(
        entity, world, {"pour_source", "cask", "tap", "bar"},
    )
    if bar is None or bar.position is None:
        return None
    p = bar.position

    # Find a patron within sight to deliver to.
    patron: Optional[EntityState] = None
    if entity.position is not None:
        grid = world.spatial
        best: Optional[tuple[int, EntityState]] = None
        for other in grid.entities.values():
            if other.entity_id == entity.entity_id or not other.alive:
                continue
            if other.position is None:
                continue
            d = entity.position.manhattan(other.position)
            if d > 8:
                continue
            tags = {t.lower() for t in (other.tags or [])}
            if "staff" in tags:
                continue
            if best is None or d < best[0]:
                best = (d, other)
        patron = best[1] if best else None

    steps = [
        PlanStep(
            kind=PlanStepKind.MOVE_TO,
            payload={"tile": [p.x, p.y, p.z]},
            label=f"to {bar.name or 'the bar'}",
        ),
        PlanStep(
            kind=PlanStepKind.SPEND_TICKS,
            payload={"ticks": 2, "label": "pour"},
            label="pour and wipe",
        ),
    ]
    if patron is not None and patron.position is not None:
        steps.append(PlanStep(
            kind=PlanStepKind.MOVE_TO,
            payload={"tile": [patron.position.x, patron.position.y, patron.position.z]},
            label=f"carry round to {patron.name}",
        ))
        steps.append(PlanStep(
            kind=PlanStepKind.SPEAK_TO,
            payload={
                "entity_id": str(patron.entity_id),
                "line": "Another round?",
            },
            label=f"serve {patron.name}",
        ))
    return NpcPlan(
        owner_id=entity.entity_id,
        name="serve the room",
        drive_source=drive,
        steps=steps,
    )


def _build_investigate_plan(
    entity: EntityState,
    world: WorldState,
    drive: str,
) -> Optional[NpcPlan]:
    """Go to the location of the freshest interesting world fact and observe.

    Picks the newest WorldFact tagged ``hazard|secret|mystery|stranger``
    that has a tile or entity scope, then moves there.  Falls back to a
    named-entity tend pattern if no relevant fact exists.
    """
    candidates: list[Any] = []
    for fact in (world.world_facts or [])[-20:]:
        if not getattr(fact, "claim", None):
            continue
        tags = {t.lower() for t in (getattr(fact, "tags", None) or [])}
        if tags.intersection({"hazard", "secret", "mystery", "stranger", "fire", "smoke"}):
            candidates.append(fact)
    candidates.sort(key=lambda f: int(getattr(f, "established_tick", 0)), reverse=True)

    target_coord: Optional[Coord] = None
    label = "investigate"
    for fact in candidates:
        sid = str(getattr(fact, "subject_id", "") or "")
        if not sid:
            continue
        if "," in sid:
            try:
                parts = sid.split(",")
                target_coord = Coord(
                    x=int(parts[0]), y=int(parts[1]),
                    z=int(parts[2]) if len(parts) >= 3 else (entity.position.z if entity.position else 0),
                )
                label = (fact.claim or "investigate")[:40]
                break
            except (ValueError, IndexError):
                continue
        else:
            other = world.spatial.entities.get(EntityId(sid))
            if other is not None and other.alive and other.position is not None:
                target_coord = other.position
                label = (fact.claim or f"check on {other.name}")[:40]
                break
    if target_coord is None:
        # Fallback: investigate a named entity in the drive text.
        return _build_tend_plan(entity, world, drive)

    steps = [
        PlanStep(
            kind=PlanStepKind.MOVE_TO,
            payload={"tile": [target_coord.x, target_coord.y, target_coord.z]},
            label=f"approach: {label}",
        ),
        PlanStep(
            kind=PlanStepKind.OBSERVE,
            payload={"tile": [target_coord.x, target_coord.y, target_coord.z]},
            label="look around",
        ),
        PlanStep(
            kind=PlanStepKind.SPEND_TICKS,
            payload={"ticks": 2, "label": "consider"},
            label="weigh what you've seen",
        ),
    ]
    return NpcPlan(
        owner_id=entity.entity_id,
        name=f"investigate: {label}",
        drive_source=drive,
        steps=steps,
    )


def _resolve_named_entity(
    world: WorldState,
    drive: str,
) -> Optional[EntityState]:
    """
    Look for an entity name (case-insensitive) inside the drive string.

    Used by tend/deliver plans to identify who/what the drive refers to.
    Returns the first matching alive entity by length of match (longest
    name wins so "Old Mira" beats "Mira" when both exist).
    """
    if not drive:
        return None
    lower = drive.lower()
    best: Optional[tuple[int, EntityState]] = None
    for ent in world.spatial.entities.values():
        if not ent.alive or not ent.name:
            continue
        nl = ent.name.lower()
        if nl and nl in lower:
            score = len(nl)
            if best is None or score > best[0]:
                best = (score, ent)
    return best[1] if best else None


_PLAN_PATTERNS: list[tuple[re.Pattern[str], Callable[..., Optional[NpcPlan]]]] = [
    # Order matters — first match wins.  Hazard-response verbs come first
    # so an NPC whose drive already mentions fleeing/dousing gets the
    # correct hazard plan rather than a generic patrol.  Deliver beats
    # work because "deliver to" lexically overlaps "labor"/"fetch".
    (_DOUSE_RE, _build_douse_fire_plan),
    (_FLEE_RE, _build_flee_hazard_plan),
    (_INVESTIGATE_RE, _build_investigate_plan),
    (_DELIVER_RE, _build_deliver_plan),
    (_SERVE_RE, _build_serve_plan),
    (_COOK_RE, _build_cook_plan),
    (_TEND_RE, _build_tend_plan),
    (_PATROL_RE, _build_patrol_plan),
    (_WORK_RE, _build_work_plan),
]

_BUILDER_BY_NAME: dict[str, Callable[..., Optional[NpcPlan]]] = {
    "patrol": _build_patrol_plan,
    "tend": _build_tend_plan,
    "work": _build_work_plan,
    "deliver": _build_deliver_plan,
    "cook": _build_cook_plan,
    "serve": _build_serve_plan,
    "flee_hazard": _build_flee_hazard_plan,
    "douse_fire": _build_douse_fire_plan,
    "investigate": _build_investigate_plan,
}


def _entity_pack_id(entity: EntityState, world: WorldState) -> str:
    mapping = world.meta.get("entity_pack_ids") or {}
    return str(mapping.get(str(entity.entity_id), ""))


def _entity_matches_roles(entity: EntityState, roles: list[str]) -> bool:
    if not roles:
        return True
    hay = " ".join(
        filter(
            None,
            [
                (entity.role or "").lower(),
                (entity.occupation or "").lower(),
            ],
        )
    )
    return any(r.lower() in hay for r in roles)


def _try_pack_plan_templates(
    entity: EntityState,
    world: WorldState,
    drive: str,
) -> Optional[NpcPlan]:
    """Match ``world.config.npc_plan_templates`` before built-in patterns."""
    templates = world.config.npc_plan_templates
    if not templates:
        return None

    templates = sorted(
        templates,
        key=lambda t: (
            len(t.entity_ids) * 100,
            len(t.entity_roles) * 50,
            len(t.drive_pattern),
        ),
        reverse=True,
    )
    pack_id = _entity_pack_id(entity, world)
    for tmpl in templates:
        if tmpl.entity_ids and pack_id not in tmpl.entity_ids:
            continue
        if tmpl.entity_roles and not _entity_matches_roles(entity, tmpl.entity_roles):
            continue
        if not tmpl.drive_pattern:
            continue
        try:
            if not re.search(tmpl.drive_pattern, drive):
                continue
        except re.error as exc:
            logger.warning(
                "npc_plans: bad regex %r on template %r (%s)",
                tmpl.drive_pattern,
                tmpl.id,
                exc,
            )
            continue

        if tmpl.builder:
            builder = _BUILDER_BY_NAME.get(tmpl.builder.lower().strip())
            if builder is None:
                logger.warning(
                    "npc_plans: unknown builder %r on template %r",
                    tmpl.builder,
                    tmpl.id,
                )
                continue
            try:
                plan = builder(entity, world, drive)
            except Exception as exc:
                logger.warning(
                    "npc_plans: builder %r failed (%s)", tmpl.builder, exc,
                )
                continue
            if plan is None:
                continue
            if tmpl.name:
                plan.name = tmpl.name
            return plan

        if tmpl.steps:
            return NpcPlan(
                owner_id=entity.entity_id,
                name=tmpl.name or tmpl.id or "pack plan",
                drive_source=drive,
                steps=[s.model_copy(deep=True) for s in tmpl.steps],
            )
    return None


__all__ = [
    "propose_plan",
    "check_interrupt",
    "try_resume",
    "next_step_action",
    "progress_plan",
]
