# Mark_1 Benchmarks

Frozen scenario suites for research runs. Scorers use trace JSONL only (no LM).

## Quick start

```bash
# Fast mock baseline (CI-friendly)
python -m src.sim.experiment --suite tavern_social_v1 --out runs/ --policy reactive_only

# Compare policies
python -m src.sim.experiment --suite tavern_social_v1 --out runs/ --policy lm
python -m src.sim.experiment report runs/
```

## Suites

| Suite | World | Ticks | Measures |
|-------|-------|-------|----------|
| `tavern_social_v1` | tavern | 20 | validity, speak rate, verb diversity, repetition |

## Trace format

Version 1 JSONL: `run_meta`, `entity_step`, `run_summary` records.
Produced by `autonomous.py --trace-dir` or experiment runner.
