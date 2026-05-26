"""
Run benchmark suites headlessly and write traces + CSV reports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml

from ..game_loop import GameLoop
from ..lm_adapter import MockLMAdapter, get_adapter
from ..npc_lm_policy import LMNpcPolicy
from ..npc_policy import ReactivePolicy
from ..tracing import TraceRecorder, new_run_id
from ..world_loader import load_world_pack
from .metrics import RunMetrics, score_run_trace, write_csv_report


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_suite_manifest(suite_name: str) -> dict[str, Any]:
    path = _repo_root() / "benchmarks" / suite_name / "manifest.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Benchmark suite not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def run_experiment_suite(
    suite_name: str,
    *,
    out_dir: Path,
    seeds: Optional[list[int]] = None,
    adapter_label: str = "mock",
    policy_mode: str = "reactive_only",
    ticks: Optional[int] = None,
    world_pack: Optional[str] = None,
) -> list[RunMetrics]:
    """
    Execute a benchmark suite and write per-run traces + summary CSV.

    policy_mode: reactive_only | lm
    """
    manifest = load_suite_manifest(suite_name)
    pack_path = world_pack or manifest.get("world_pack", "worlds/tavern")
    pack = _repo_root() / pack_path if not str(pack_path).startswith("/") else Path(pack_path)
    n_ticks = ticks if ticks is not None else int(manifest.get("ticks", 20))
    seed_list = seeds if seeds is not None else list(range(int(manifest.get("seeds", 5))))

    out_dir.mkdir(parents=True, exist_ok=True)
    all_metrics: list[RunMetrics] = []

    for seed in seed_list:
        world = load_world_pack(pack)
        world.rng_seed = seed
        adapter = _make_adapter(adapter_label)
        loop, policy_label = _make_loop(world, adapter, policy_mode)

        trace_path = out_dir / f"{suite_name}_seed{seed}_{policy_label}.jsonl"
        recorder = TraceRecorder(
            run_id=new_run_id(),
            world_pack=str(pack_path),
            seed=seed,
            adapter_label=adapter_label,
            policy_label=policy_label,
            output_path=trace_path,
        )
        loop.trace_recorder = recorder

        for _ in range(n_ticks):
            loop.autonomous_tick()

        recorder.close(world)
        m = score_run_trace(trace_path)
        m.seed = seed
        all_metrics.append(m)

    csv_path = out_dir / f"{suite_name}_report.csv"
    write_csv_report([m.to_row() for m in all_metrics], csv_path)
    return all_metrics


def _make_adapter(label: str):
    if label == "mock" or label.startswith("mock"):
        return MockLMAdapter()
    if label.startswith("ollama:"):
        model = label.split(":", 1)[1]
        return get_adapter(use_ollama=True, model=model)
    if label.startswith("torch:"):
        spec = label.split(":", 1)[1]
        from pathlib import Path

        if Path(spec).expanduser().exists():
            return get_adapter(
                use_torch=True,
                torch_model_path=spec,
                torch_warmup=False,
            )
        return get_adapter(use_torch=True, model=spec, torch_warmup=False)
    return get_adapter()


def _make_loop(world, adapter, policy_mode: str):
    world.config.npc_policy.cognition.mode = policy_mode

    cfg = world.config.npc_policy
    reactive = ReactivePolicy(
        threat_lookback_ticks=cfg.threat_lookback_ticks,
        flee_health_fraction=cfg.flee_health_fraction,
    )
    if policy_mode == "reactive_only":
        return GameLoop(world, adapter=adapter, npc_policy=reactive), "reactive_only"

    if isinstance(adapter, MockLMAdapter):
        return GameLoop(world, adapter=adapter, npc_policy=reactive), "reactive_only"

    npc = LMNpcPolicy(adapter, fallback=reactive, cognition_mode=policy_mode)
    return GameLoop(world, adapter=adapter, npc_policy=npc), policy_mode
