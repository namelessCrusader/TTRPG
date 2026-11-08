"""THE SEAMS OF WAVE 13a (lane B1): sim.py, fields.py, heat.py and quiet.py.

Every seam here is a plug that a later lane fills (cold, body heat, the
scheduler, mixtures, injury) and that does today what the code it replaced did.
So each is tested twice: that it IS what it replaced, and that it IS a plug
(an override reaches the law, and nothing else does the old thing behind its back).
"""
import ast
import importlib.util
import os

import numpy as np
import pytest

from src.voxel import quiet as Q
from src.voxel.materials import AIR, AMBIENT, FACE_CLIP, SIGMA, STONE, TICK_S, WOOD, _EMISS_ARR
from src.voxel.sim import HAVE_NUMBA, World

HERE = os.path.dirname(os.path.abspath(__file__))
VOX = os.path.join(os.path.dirname(HERE), "src", "voxel")
TOOLS = os.path.join(os.path.dirname(HERE), "tools")
f32 = np.float32


def _room(nx=20, ny=16, nz=12):
    w = World(nx, ny, nz, voxel_cm=5)
    w.fill(0, nx, 0, ny, 0, 1, STONE)
    return w


# ── fields: what is not at rest ──────────────────────────────────────────────
def test_unsettled_is_the_window_of_E_not_zero_and_the_hot_window_is_its_box():
    rng = np.random.default_rng(4)
    for trial in range(12):
        w = _room()
        w.E[...] = 0.0
        for _ in range(int(rng.integers(0, 4))):
            x, y, z = (int(rng.integers(0, n)) for n in w.shape)
            w.E[x, y, z] = float(rng.normal() * 100.0) or 1.0
        for pad in (0, 1, 2):
            box, why = w._unsettled(pad)
            assert why == []
            want = w._window(w.E != 0, pad)
            assert box == want
            assert w._hot_window(pad) == want
        assert w._heat_busy() == bool(w.E.any())
        T, box = w._T_in()
        assert np.array_equal(T, AMBIENT + w.E / w.heat_capacity()), "the windowed T is the whole T"


def test_the_gate_asks_the_seam_and_skipping_off_means_always_busy():
    w = _room()
    assert w._heat_busy() is False
    w.E[3, 3, 3] = 5.0
    assert w._heat_busy() is True
    w.E[...] = 0.0
    w.skip_quiet = False
    assert w._heat_busy() is True
    assert w._unsettled(0)[0] == (slice(None),) * 3                 # the whole world
    assert w._rest_cell() == 0.0


def test_an_override_of_the_seams_reaches_the_laws_that_use_them():
    calls = []

    class Cold(World):
        def _heat_busy(self):
            calls.append("busy")
            return False

        def _unsettled(self, pad=0):
            calls.append(("unsettled", pad))
            return None, []

    w = Cold(12, 10, 8, voxel_cm=5)
    w.fill(0, 12, 0, 10, 0, 1, STONE)
    w.fill(4, 6, 4, 6, 1, 3, WOOD)
    w.E[4, 4, 1] = 9.0e5                                           # a fire waiting to start
    burned = []
    real = w._law_burn
    w._law_burn = lambda: (burned.append(1), real())[1]
    for _ in range(3):
        w.step()
    assert not burned, "the heat gate said not busy, and the burn law ran anyway"
    assert "busy" in calls and ("unsettled", 1) in calls         # the window of the heat law came from the seam
    assert w.E[4, 5, 1] == 0.0                                   # and, told nothing was unsettled, nothing conducted


def test_the_floor_seam_clamps_at_zero_celsius_and_books_the_heat_it_put_back():
    w = _room()
    C = float(w.heat_capacity()[5, 5, 0])
    w.E[5, 5, 0] = -(AMBIENT + 15.0) * C                           # -15 C
    shed0 = w.shed
    w._floor()
    assert w.E[5, 5, 0] == f32(-AMBIENT * C)                       # 0 C
    assert w.shed < shed0                                           # a source: it nets into shed
    assert w.shed - shed0 == pytest.approx(-15.0 * C, rel=1e-4)

    class NoFloor(World):
        def _floor(self):
            pass

    def cold_step(cls):
        n = cls(10, 10, 8, voxel_cm=5)
        n.fill(0, 10, 0, 10, 0, 1, STONE)
        n.E[5, 5, 0] = -(AMBIENT + 15.0) * C                        # -15 C: heat is "busy", the floor block runs
        n.step()
        return float(n.E[5, 5, 0])
    floor = float(f32(-AMBIENT * C))
    assert cold_step(World) >= floor                                # a default step calls the floor: not below 0 C
    assert cold_step(NoFloor) < floor                               # the floor is a plug: gone when overridden


# ── heat: the radiation and the sky ──────────────────────────────────────────
def _one_pass_loss(world):
    """Joules the radiation seam takes from each cell in one pass of the face law."""
    got = []
    real = type(world)._env_radiation

    class Spy(type(world)):
        def _env_radiation(self, u, eps, sl):
            out = real(self, u, eps, sl)
            got.append(out.copy())
            return out
    world.__class__ = Spy
    world._law_conduct()
    return got[0]


def test_the_radiation_seam_is_the_old_law_e_sigma_a_t4_less_t_ambient_4():
    w = World(7, 7, 7, voxel_cm=5)
    w.fill(3, 4, 3, 4, 3, 4, STONE)
    w.E[3, 3, 3] = 60.0 * float(w.heat_capacity()[3, 3, 3])
    loss = _one_pass_loss(w)
    A = 0.05 * 0.05
    T, T0 = 80.0 + 273.15, AMBIENT + 273.15
    want = float(_EMISS_ARR[STONE]) * SIGMA * A * 6 * TICK_S * (T ** 4 - T0 ** 4)
    assert float(loss[3, 3, 3]) == pytest.approx(want, rel=2e-4)
    assert not np.delete(loss.reshape(-1), np.ravel_multi_index((3, 3, 3), loss.shape)).any()
    assert loss.dtype == np.float32


def test_overriding_the_radiation_seam_changes_what_leaves_the_world():
    def run(cls):
        w = cls(7, 7, 7, voxel_cm=5)
        w.fill(3, 4, 3, 4, 3, 4, STONE)
        w.E[3, 3, 3] = 60.0 * float(w.heat_capacity()[3, 3, 3])
        e0 = float(w.E.sum(dtype=np.float64))
        w._law_conduct()
        return w, e0

    class NoRadiation(World):
        def _env_radiation(self, u, eps, sl):
            return np.zeros(u.Ts.shape, np.float32)
    a, e0 = run(World)
    b, e1 = run(NoRadiation)
    assert a.shed > 0.0 and b.shed == 0.0
    assert float(a.E.sum(dtype=np.float64)) == pytest.approx(e0 - a.shed, abs=1e-2)
    assert float(b.E.sum(dtype=np.float64)) == pytest.approx(e1, abs=1e-2)


def test_the_top_sink_seam_takes_the_plume_that_reaches_open_sky_and_books_it():
    def run(cls):
        w = cls(8, 8, 10, voxel_cm=5)
        w.open_sky = True
        w.E[3, 3, 9] = 4000.0                                      # hot air in the top layer
        w.E[4, 4, 9] = 2000.0
        w._law_rise()
        return w
    a = run(World)

    class NoSink(World):
        def _top_sink(self):
            pass
    b = run(NoSink)
    assert a.shed > 0.0 and b.shed == 0.0
    assert float(a.E[3, 3, 9]) < float(b.E[3, 3, 9])
    assert float(a.E.sum(dtype=np.float64)) + a.shed == pytest.approx(
        float(b.E.sum(dtype=np.float64)) + b.shed, abs=1e-2)


# ── sim: the scheduler hook, E_dtype, the lazy arrays ────────────────────────
def test_the_scheduler_hook_runs_once_a_tick_right_after_the_tick_starts():
    seen = []

    class Spy:
        def pre(self, w):
            seen.append((w.tick, w.o2 is None))

    w = _room(10, 10, 8)
    w.slow = Spy()
    for _ in range(4):
        w.step()
    assert seen == [(0, True), (1, False), (2, False), (3, False)]   # before the first-step air, ahead of every law


def test_E_dtype_is_float32_unless_the_world_is_built_for_float64():
    assert World.E_dtype is np.float32
    assert World(6, 6, 6, voxel_cm=5).E.dtype == np.float32
    w = World(6, 6, 6, voxel_cm=5, E_dtype=np.float64)
    assert w.E.dtype == np.float64 and w.E_dtype is np.float64
    assert "E_dtype" not in vars(World(6, 6, 6, voxel_cm=5))
    w.fill(0, 6, 0, 6, 0, 1, STONE)
    w.E[3, 3, 0] = 100.0
    e0 = w.total_energy()
    for _ in range(8):
        w.step()
    assert w.E.dtype == np.float64
    assert w.total_energy() == pytest.approx(e0, rel=1e-9)


def test_the_lazy_arrays_are_made_once_with_the_right_type_and_nothing_else_is():
    w = _room(9, 8, 7)
    for _ in range(5):
        w.step()
    assert w.lazy_bytes() == {} and w.mixes is None
    from src.voxel import outside_world
    assert w not in outside_world._OUT, "T8's cls cache (outside the world) exists only for a climate"
    n = w.E.size
    mixes = w.ensure_mixes()
    assert w.ensure_mixes() is mixes and w.mixes is mixes
    assert w.scomp.dtype == w.fcomp.dtype == np.uint32 and not w.scomp.any()
    assert w.mixes_live is False
    assert w.ensure_damage().dtype == np.float32
    assert w.ensure_bzone().dtype == np.uint16
    assert w.ensure_ice().dtype == np.float32
    assert w.lazy_bytes() == {"scomp": 4 * n, "fcomp": 4 * n, "dmg": 4 * n, "bzone": 2 * n,
                              "ice_g": 4 * n}
    assert w.ensure_bzone() is w.bzone
    assert w.ensure_records().n == 0 and w.ensure_records() is w.records
    assert w.ensure_layers().dx == pytest.approx(0.05)
    assert w.face_env is None                                         # nobody made the bundle for us
    for _ in range(3):
        w.step()                                                      # and a world that has them steps


def test_the_mix_table_collects_on_a_quiet_tick_and_renumbers_the_holders_it_is_given():
    w = _room(9, 8, 7)
    assert w.collect_mixes() is None and w.mix_holders() == {}      # no table: nothing to do
    mixes = w.ensure_mixes()
    assert set(w.mix_holders()) == {"scomp", "fcomp"}
    keep = [mixes.parse([("water", "dissolved", 0, 0.9), ("sodium_chloride", "dissolved", 0, 0.1)]),
            mixes.parse([("water", "dissolved", 0, 0.8), ("sodium_chloride", "dissolved", 0, 0.2)])]
    rng = np.random.default_rng(1)
    for i in range(5000):                                           # rows nobody uses
        f = 0.05 + 0.9 * rng.random()
        mixes.parse([("water", "dissolved", 0, f), ("sodium_chloride", "dissolved", 0, 1.0 - f)])
    w.scomp[2:4, 2:4, 1] = keep[0]
    w.fcomp[5, 5, 1] = keep[1]
    n0 = mixes.n_rows
    d0 = mixes.content_digest(np.array([keep[0], keep[1]]))
    w.bodies.append({"mats": None})                                  # matter off the lattice: no collection
    assert w.collect_mixes() is None and mixes.n_rows == n0
    w.bodies.clear()
    lut = w.collect_mixes()
    assert lut is not None and mixes.n_rows < n0 // 10
    assert np.array_equal(mixes.content_digest(np.array([w.scomp[2, 2, 1], w.fcomp[5, 5, 1]])), d0)
    assert (w.scomp[2:4, 2:4, 1] == w.scomp[2, 2, 1]).all()
    assert w.collect_mixes() is None                                 # nothing left to collect
    assert not mixes.check_table()


def test_law_timing_times_a_scene_and_a_world_given_by_name():
    LT = _law_timing()
    res = LT.run_scene("pond", 14, 4)                               # a scene of the repo's, as golden plays it
    assert res["ticks"] == 10 and "_law_acid" in res["laws"] and "_heat_busy" in res["laws"]
    res = LT.run_world("tests.test_w13_seams:_room", 9, 3)         # any module:function that returns a World
    assert res["ticks"] == 6 and res["step"]["mean_ms"] > 0.0


# ── quiet: the declared periods and the second look ──────────────────────────
def test_the_three_tick_gates_wave_13_adds_are_declared_and_the_period_is_still_600():
    got = {n: (v, w) for n, v, w in Q.period_sources()}
    assert got["BODY_CLOCK"][0] == 40                                 # the real constant, bodyheat.BODY_PERIOD
    assert got["CLIMATE_EVERY"][0] == 40
    assert got["CLIMATE_DAY"][0] == 0                                 # a float day coordinate: no period
    assert Q.PERIOD == Q.period() == 600
    from src.voxel import bodyheat, outside_world
    assert bodyheat.BODY_PERIOD == 40 and outside_world.OutsideMixin.CLIMATE_EVERY == 40


def test_every_declared_gate_is_really_in_the_tree_and_names_a_declared_period():
    """The scan of tests/test_quiet.py warns about a tick gate it cannot turn into
    an integer; B1 declares the three there were. A declaration must not rot: each
    row of quiet.DECLARED_GATES has to be a gate the scan still finds, and name a
    row of DECLARED_PERIODS. (A gate some OTHER lane adds later is a warning, as
    in test_quiet.py: it is that lane's to declare, not this test's to fail.)"""
    import warnings
    from tests import test_quiet as TQ
    found, undeclared = set(), []
    for f, line, text, right in TQ._tick_gate_expressions():
        if f[:-3] in TQ.BASE_FILES or f in TQ.NOT_GATES:
            continue
        with open(os.path.join(VOX, f)) as fh:
            mod = ast.parse(fh.read())
        consts = {n.targets[0].id: n.value.value for n in mod.body
                  if isinstance(n, ast.Assign) and len(n.targets) == 1
                  and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Constant)
                  and isinstance(n.value.value, int)}
        if TQ._resolve(right, consts) is None:
            key = (f, getattr(right, "id", ast.unparse(right)))
            (found.add(key) if key in Q.DECLARED_GATES else undeclared.append((key, line)))
    names = {n for n, _v, _w in Q.period_sources()}
    assert set(Q.DECLARED_GATES.values()) <= names
    mine = {("bodyheat.py", "period"), ("outside_world.py", "self.CLIMATE_EVERY"), ("climate.py", "1.0")}
    assert mine <= found, f"a declared gate is no longer in the tree: {sorted(mine - found)}"
    for key, line in undeclared:
        warnings.warn(f"{key[0]}:{line}: a tick gate with period `{key[1]}` that nobody declared in "
                      f"quiet.DECLARED_PERIODS")


def test_the_probe_looks_inside_the_period_too():
    n = Q.period()
    m = Q.inside_point(n)
    assert 0 < m < n and np.gcd(m, n) == 1 and abs(m - n / 2) < 5
    assert Q.inside_point(2) == 0 and Q.inside_point(1) == 0
    assert 0 < Q.inside_point(4200) < 4200 and np.gcd(Q.inside_point(4200), 4200) == 1


class _There(World):
    """A world whose state leaves and comes back inside one period: `wob` is 1
    for ticks 100..499 of every 600."""

    def step(self):
        super().step()
        self.wob = int(100 <= self.tick % 600 < 500)


def test_a_state_that_leaves_and_returns_within_a_period_is_not_quiet():
    w = _There(12, 10, 8, voxel_cm=5)
    w.fill(0, 12, 0, 10, 0, 1, STONE)
    w.step()                                       # the first step makes the air; quiet() waits for it
    while w.tick % 600 != 0:
        w.step()
    twin = _There(12, 10, 8, voxel_cm=5)
    twin.fill(0, 12, 0, 10, 0, 1, STONE)
    twin.restore(w.snapshot())
    # the two ends of the period agree, which is all the old probe looked at...
    twin.step()
    d0 = Q.digest(twin)
    for _ in range(Q.period()):
        twin.step()
    assert Q.diff(d0, Q.digest(twin)) == [], "the test world must look still from its two ends"
    # ...and the look inside does not
    q = Q.quiet(w, probe_=True)
    assert not q.ok and any(b.place == "wob" for b in q.blockers), q


def test_a_quiet_world_is_still_quiet_with_the_look_inside_and_the_tick_count_is_unchanged():
    w = _room(12, 10, 8)
    for _ in range(3):
        w.step()
    t0 = w.tick
    q = Q.quiet(w, probe_=True)
    assert q.ok and w.tick == t0 + 1 + Q.PERIOD


def test_the_heat_gate_reads_the_box_and_the_reasons_of_the_unsettled_seam():
    w = _room()
    for _ in range(3):
        w.step()
    assert not [b for b in Q.gate_heat(w, 0)]
    w.E[4, 4, 2] = 3.0
    assert any("E != 0" in b.reason for b in Q.gate_heat(w, 0))
    assert not Q.gate_heat(w, 1)                                    # the net owns the heat at tier 1

    class Driven(World):
        def _unsettled(self, pad=0):
            return (slice(None),) * 3, ["interface", "sky_drive"]
    d = Driven(10, 10, 8, voxel_cm=5)
    reasons = [b.reason for b in Q.gate_heat(d, 0)]
    assert reasons == ["interface", "sky_drive"]
    assert [b.reason for b in Q.gate_heat(d, 1)] == ["interface", "sky_drive"]


# ── tools/law_timing.py ──────────────────────────────────────────────────────
def _law_timing():
    spec = importlib.util.spec_from_file_location("law_timing_under_test",
                                                  os.path.join(TOOLS, "law_timing.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_law_timing_times_every_law_reports_mean_and_p99_and_takes_its_wrappers_off():
    LT = _law_timing()
    names = ("step", "_law_conduct", "_hot_window", "_heat_busy", "_law_burn")
    before = {n: getattr(World, n) for n in names}
    res = LT.main(["--tiny"])["tiny"]
    assert all(getattr(World, n) is before[n] for n in names)        # the class is as it was
    assert res["ticks"] == 8
    laws = res["laws"]
    for name in ("_law_conduct", "_law_burn", "_law_air", "_law_o2", "_hot_window", "_heat_busy"):
        assert name in laws, name
        row = laws[name]
        assert set(row) == {"calls_per_tick", "mean_ms", "p99_ms"}
        assert 0.0 <= row["mean_ms"] <= row["p99_ms"] + 1e-9
    assert laws["_law_conduct"]["calls_per_tick"] == pytest.approx(1.0)
    assert res["step"]["mean_ms"] >= laws["_law_conduct"]["mean_ms"]
    assert res["step"]["p99_ms"] >= res["step"]["mean_ms"]
    top = sum(laws[n]["mean_ms"] for n in laws if n.startswith("_law_") and n not in
              ("_law_skin_water", "_law_radiate"))
    assert top <= res["step"]["mean_ms"] * 1.5                       # the laws are the tick, give or take the nesting


def test_law_timing_takes_the_median_of_repeated_runs():
    LT = _law_timing()
    one = lambda m, p: {"ticks": 5, "step": {"mean_ms": m, "p99_ms": p},
                        "laws": {"_law_x": {"calls_per_tick": 1.0, "mean_ms": m / 2, "p99_ms": p / 2}}}
    res = LT.median_of([one(1.0, 2.0), one(9.0, 20.0), one(3.0, 4.0)])
    assert res["runs"] == 3 and res["step"] == {"mean_ms": 3.0, "p99_ms": 4.0}
    assert res["laws"]["_law_x"]["mean_ms"] == 1.5 and "median of 3 runs" in LT.table("s", res)
    assert LT.median_of([one(1.0, 2.0)]) == one(1.0, 2.0)


def test_law_timing_prints_a_table_with_the_biggest_law_first(capsys):
    LT = _law_timing()
    LT.main(["--tiny"])
    out = capsys.readouterr().out
    assert "tiny" in out and "mean ms" in out and "p99 ms" in out
    rows = [ln for ln in out.splitlines() if ln[2:].lstrip().startswith("_law_") or ln.startswith("* ")]
    means = [float(r.split()[-3]) for r in rows]
    assert means == sorted(means, reverse=True)
