import tempfile
from pathlib import Path

from src.sim.experiment.runner import run_experiment_suite
from src.sim.experiment.metrics import score_run_trace


def test_experiment_suite_mock():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        metrics = run_experiment_suite(
            "tavern_social_v1",
            out_dir=out,
            seeds=[0, 1],
            policy_mode="reactive_only",
            ticks=3,
        )
        assert len(metrics) == 2
        for m in metrics:
            assert m.ticks >= 3
            assert m.final_state_hash
        traces = list(out.glob("*.jsonl"))
        assert len(traces) == 2
        assert (out / "tavern_social_v1_report.csv").exists()
        assert score_run_trace(traces[0]).entity_steps > 0
