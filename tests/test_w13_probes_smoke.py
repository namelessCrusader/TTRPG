"""Smoke test for the six wave-13 measurement scripts in tools/ (A-M lane).

Each script is run on a tiny world (16x16x16 voxels) through its `--tiny` switch,
as a separate process (they patch World methods), writing into a temporary
results file. This checks that they run and write the shape of result that
phase B's B0 step reads; the real numbers come from running them without
`--tiny` under the shared lock (see tools/w13_*.py).

All six together must take under 20 s.
"""
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
SCRIPTS = ["w13_floor_probe", "w13_route_bench", "w13_scan_steps",
           "w13_wake_probe", "w13_walk_leak", "w13_f64_conduct"]


def _run_all(tmp_path):
    out = tmp_path / "results.json"
    env = dict(os.environ, MARK1_REPO=REPO, PYTHONDONTWRITEBYTECODE="1")
    t0 = time.perf_counter()
    for name in SCRIPTS:
        r = subprocess.run([sys.executable, os.path.join(REPO, "tools", name + ".py"),
                            "--tiny", "--out", str(out)],
                           env=env, capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, f"{name} failed:\n{r.stderr[-1500:]}"
    return json.loads(out.read_text()), time.perf_counter() - t0


def test_all_six_probes_run_on_a_16_cubed_world_in_under_20_s(tmp_path):
    res, secs = _run_all(tmp_path)
    assert secs < 20.0, f"the six probes took {secs:.1f} s"
    for key in ("floor_probe", "route_bench", "scan_steps", "wake_probe",
                "walk_leak", "f64_conduct"):
        row = res[key + "_tiny"]
        assert row["status"] in ("ok", "not run", "partial"), (key, row["status"])
        assert "load_note" in row, key


def test_the_probes_report_the_numbers_phase_b_reads(tmp_path):
    res, _ = _run_all(tmp_path)
    fp = res["floor_probe_tiny"]
    assert fp["probe_sees_a_clamp"] and fp["cases_below_0C"] == []
    rb = res["route_bench_tiny"]
    assert rb["value"] >= 0 and "tiny" in rb["table"]
    ss = res["scan_steps_tiny"]
    assert ss["value"] >= 1                     # the ledge it builds is found
    wk = res["wake_probe_tiny"]
    cell = wk["table"]["base"]["tiny"]["1"]
    assert "wake_ms_per_tick_per_person" in cell and wk["not_run"] == {}
    wl = res["walk_leak_tiny"]
    assert "controls" in wl and "fix_text" in wl and "found" in wl
    assert wl["lifted_never_put_J"] == 0.0
    fc = res["f64_conduct_tiny"]
    assert "conduct_pct_vs_f32" in fc["room"] and "conduct_pct_vs_f32" in fc["house"]
