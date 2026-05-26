"""
Per-tick LM call budgeting and usage metrics.

Budgeting caps parallel NPC infers per tick. Metrics track call counts and
latency for production observability.
"""

from __future__ import annotations

import functools
import time
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar

if TYPE_CHECKING:
    from .lm_adapter import LMAdapter
    from .schemas import EntityState, WorldState

F = TypeVar("F", bound=Callable[..., Any])

_BUDGET_META = "lm_call_budget"
_USED_META = "lm_calls_this_tick"
_METRICS_META = "lm_metrics"

# Default estimated USD per 1k tokens (rough; override via world.meta)
_DEFAULT_COST_PER_1K = 0.00015


def _metrics(world: "WorldState") -> dict[str, Any]:
    m = world.meta.setdefault(_METRICS_META, {})
    m.setdefault("total_calls", 0)
    m.setdefault("calls_by_kind", {})
    m.setdefault("total_latency_ms", 0.0)
    m.setdefault("tokens_estimated", 0)
    return m


def record_lm_call(
    world: "WorldState",
    kind: str,
    *,
    latency_ms: float = 0.0,
    model_id: str = "",
    tokens_estimated: int = 0,
) -> None:
    """Record one LM adapter invocation."""
    m = _metrics(world)
    m["total_calls"] = int(m["total_calls"]) + 1
    by_kind: dict[str, int] = m["calls_by_kind"]
    by_kind[kind] = int(by_kind.get(kind, 0)) + 1
    m["total_latency_ms"] = float(m["total_latency_ms"]) + float(latency_ms)
    if model_id:
        m["last_model_id"] = model_id
    if tokens_estimated:
        m["tokens_estimated"] = int(m.get("tokens_estimated", 0)) + tokens_estimated
    m["last_kind"] = kind


def metrics_snapshot(world: "WorldState") -> dict[str, Any]:
    """Return a copy of accumulated LM metrics."""
    m = dict(_metrics(world))
    m["calls_by_kind"] = dict(m.get("calls_by_kind") or {})
    cost_per_1k = float(world.meta.get("lm_cost_per_1k", _DEFAULT_COST_PER_1K))
    tokens = int(m.get("tokens_estimated", 0))
    m["estimated_cost_usd"] = round((tokens / 1000.0) * cost_per_1k, 6)
    return m


def format_metrics_summary(world: "WorldState") -> str:
    """One-line human summary for REPL / status command."""
    m = metrics_snapshot(world)
    kinds = m.get("calls_by_kind") or {}
    kind_str = ", ".join(f"{k}={v}" for k, v in sorted(kinds.items()))
    return (
        f"LM calls: {m.get('total_calls', 0)} "
        f"({kind_str or 'none'}) · "
        f"{m.get('total_latency_ms', 0):.0f}ms · "
        f"~{m.get('tokens_estimated', 0)} tok · "
        f"${m.get('estimated_cost_usd', 0):.4f} est."
    )


def wrap_adapter_metrics(adapter: "LMAdapter", world: "WorldState") -> "LMAdapter":
    """
    Instrument an LM adapter to record metrics on each public infer/narrate call.

    Safe to call multiple times — already-wrapped methods are skipped.
    """
    tracked = (
        ("infer", "infer"),
        ("infer_npc", "infer_npc"),
        ("narrate", "narrate"),
        ("generate_npc_reply", "npc_reply"),
        ("propose_consequences", "consequences"),
        ("adjudicate", "adjudicate"),
        ("enrich_goal", "enrich_goal"),
    )
    for method_name, kind in tracked:
        if getattr(getattr(adapter, method_name, None), "_metrics_wrapped", False):
            continue
        original = getattr(adapter, method_name, None)
        if original is None or not callable(original):
            continue

        @functools.wraps(original)
        def _wrapped(*args, _orig=original, _kind=kind, **kwargs):
            t0 = time.monotonic()
            model_id = getattr(adapter, "MODEL_ID", "") or getattr(adapter, "model", "")
            result = _orig(*args, **kwargs)
            latency = (time.monotonic() - t0) * 1000.0
            tokens = 0
            if _kind == "infer" and len(args) >= 1:
                proj = args[0]
                tokens = int(getattr(proj, "estimated_tokens", 0) or 0)
            record_lm_call(
                world,
                _kind,
                latency_ms=latency,
                model_id=str(model_id),
                tokens_estimated=tokens,
            )
            return result

        _wrapped._metrics_wrapped = True  # type: ignore[attr-defined]
        setattr(adapter, method_name, _wrapped)
    return adapter


def tick_lm_budget(world: "WorldState") -> Optional[int]:
    """Reset budget counter at tick start. Returns budget cap or None (unlimited)."""
    cap = world.config.npc_policy.cognition.max_lm_calls_per_tick
    if cap is None or cap <= 0:
        world.meta.pop(_BUDGET_META, None)
        world.meta.pop(_USED_META, None)
        return None
    world.meta[_BUDGET_META] = int(cap)
    world.meta[_USED_META] = 0
    return int(cap)


def lm_budget_remaining(world: "WorldState") -> Optional[int]:
    cap = world.meta.get(_BUDGET_META)
    if cap is None:
        return None
    used = int(world.meta.get(_USED_META, 0))
    return max(0, int(cap) - used)


def can_spend_lm_call(world: "WorldState") -> bool:
    remaining = lm_budget_remaining(world)
    if remaining is None:
        return True
    return remaining > 0


def spend_lm_call(world: "WorldState") -> bool:
    if not can_spend_lm_call(world):
        return False
    world.meta[_USED_META] = int(world.meta.get(_USED_META, 0)) + 1
    return True


def lm_entity_priority(
    world: "WorldState",
    entity: "EntityState",
    *,
    entity_id: object = "",
) -> int:
    """
    Rank entities for LM budget allocation (higher = more important).

    Tiers (approximate):
      combat / wounded / player / fresh stimulus → first
      goal-driven NPCs → next
      bystanders → reactive fallback when cap is hit
    """
    from .schemas import AlertnessLevel, EntityKind

    score = 0

    if entity.alertness == AlertnessLevel.COMBAT:
        score += 1000
    elif entity.alertness == AlertnessLevel.HIGH:
        score += 500
    elif entity.alertness == AlertnessLevel.MEDIUM:
        score += 120

    max_hp = max(1.0, float(getattr(entity, "max_health", entity.health) or 1))
    if entity.health / max_hp < 0.4:
        score += 250
    elif entity.health / max_hp < 0.65:
        score += 80

    if entity.kind == EntityKind.PLAYER:
        score += 350

    try:
        from .social_stimulus import stimulus_priority

        score += stimulus_priority(entity) * 3
    except Exception:
        pass

    if entity.goals:
        score += 100
        for g in entity.goals[:2]:
            if any(w in g.lower() for w in ("urgent", "protect", "escape", "kill", "find")):
                score += 60
                break

    if entity.meta.get("social_stimulus"):
        score += 180

    if int(entity.meta.get("director_infer_budget", 0)) > 0:
        score += 40

    if entity.meta.get("last_action_verb") in (
        "speak", "shout", "yell", "whisper", "threaten", "attack",
    ):
        score += 50

    return score


def allocate_lm_slots(
    world: "WorldState",
    entities: list[tuple[object, "EntityState"]],
) -> set[object]:
    """
    Pre-select which entities may call infer this tick (call before ThreadPool).

    Respects per-tick cap and director infer budget. Entities are ranked by
    ``lm_entity_priority`` so combatants and socially engaged NPCs win slots
    over passive bystanders.
    """
    cap = lm_budget_remaining(world)
    ranked = sorted(
        entities,
        key=lambda pair: (
            -lm_entity_priority(world, pair[1], entity_id=pair[0]),
            pair[1].name,
        ),
    )
    chosen: set[object] = set()
    for eid, entity in ranked:
        if cap is not None and len(chosen) >= cap:
            break
        if int(entity.meta.get("director_infer_budget", 0)) <= 0:
            continue
        chosen.add(eid)
    if cap is not None:
        world.meta[_USED_META] = len(chosen)
    return chosen


def should_use_lm_for_entity(
    world: "WorldState",
    entity: "EntityState",
    *,
    uses_lm_layer: bool,
) -> bool:
    """True if this entity should receive an LM infer call this tick."""
    if not uses_lm_layer:
        return False
    if not can_spend_lm_call(world):
        return False
    if int(entity.meta.get("director_infer_budget", 0)) <= 0:
        return False
    return True
