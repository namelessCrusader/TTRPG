"""Accumulate per-simulation counters on ``world.meta['run_metrics']``."""

from __future__ import annotations

from typing import Any

from .schemas import WorldState


def bump(world: WorldState, key: str, amount: int = 1) -> None:
    metrics: dict[str, Any] = world.meta.setdefault("run_metrics", {})
    metrics[key] = int(metrics.get(key, 0)) + amount


def record_mdp_choice(
    world: WorldState,
    *,
    entity_name: str,
    selected_index: int,
    menu_size: int,
    verb: str,
) -> None:
    bump(world, "mdp_selections")
    dist = world.meta.setdefault("mdp_index_histogram", {})
    dist[str(selected_index)] = int(dist.get(str(selected_index), 0)) + 1
    verbs = world.meta.setdefault("mdp_verbs_chosen", {})
    verbs[verb] = int(verbs.get(verb, 0)) + 1


def format_run_metrics(world: WorldState) -> list[str]:
    metrics = world.meta.get("run_metrics") or {}
    if not metrics:
        return []
    lines = ["  Run metrics:"]
    for key in sorted(metrics.keys()):
        lines.append(f"    {key}: {metrics[key]}")
    hist = world.meta.get("mdp_index_histogram") or {}
    if hist:
        lines.append(
            "    mdp_index_histogram: "
            + ", ".join(f"{k}={v}" for k, v in sorted(hist.items(), key=lambda x: int(x[0])))
        )
    return lines
