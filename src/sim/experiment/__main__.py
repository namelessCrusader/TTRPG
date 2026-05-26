"""CLI: python -m src.sim.experiment --suite tavern_social_v1 --out runs/"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .runner import run_experiment_suite
from .metrics import score_run_trace, write_csv_report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Mark_1 experiment runner")
    p.add_argument("--suite", required=True, help="Benchmark suite name under benchmarks/")
    p.add_argument("--out", default="runs", help="Output directory")
    p.add_argument("--seeds", default="0,1,2,3,4", help="Comma-separated seeds")
    p.add_argument("--adapter", default="mock", help="mock | ollama:model")
    p.add_argument("--policy", default="reactive_only", help="Cognition policy mode")
    p.add_argument("--ticks", type=int, default=None)
    p.add_argument("command", nargs="?", default="run", choices=["run", "report"])
    p.add_argument("report_dir", nargs="?", default=None)
    args = p.parse_args(argv)

    if args.command == "report":
        rdir = Path(args.report_dir or args.out)
        rows = []
        for path in sorted(rdir.glob("*.jsonl")):
            m = score_run_trace(path)
            rows.append(m.to_row())
        out_csv = rdir / "combined_report.csv"
        write_csv_report(rows, out_csv)
        print(f"Wrote {out_csv} ({len(rows)} runs)")
        return 0

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    metrics = run_experiment_suite(
        args.suite,
        out_dir=Path(args.out),
        seeds=seeds,
        adapter_label=args.adapter,
        policy_mode=args.policy,
        ticks=args.ticks,
    )
    print(f"Completed {len(metrics)} runs → {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
