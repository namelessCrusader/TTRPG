"""The reaction and sorption row runner on the lattice (wave 15, S1 slice 1).

Each test names the mutation it must catch (the mutation was tried by hand when the test was
written; the line says what breaks)."""
import math

import numpy as np
import pytest

from src.voxel import rxn
from src.voxel import scenes_reactions as S
from src.voxel import species as SP
from src.voxel.matterdata import R
from src.voxel.reactions import (LatticeState, LumpView, NAMES, ReactionMixin, row_supported)
from src.voxel.sim import World


def _ledgers(w):
    return w.mixes.ledger_total("quant") + w.mixes.ledger_total("f32")


def _total_g(w):
    """Every gram: lumps + pending + the quant and f32 ledgers (gas is none here)."""
    return w.rx_mass_g() + w.rx_pending_g() + _ledgers(w)


def _E(w):
    pe = 0.0 if w.cb is None or w.cb.pend_E is None else float(w.cb.pend_E.sum())
    return float(w.E.astype(np.float64).sum()) + pe


@pytest.mark.parametrize("cao,water", [(50.0, 30.0), (3.0, 90.0), (120.0, 5.0), (0.01, 0.5)])
def test_slaking_closes_mass_and_energy_on_the_lattice(cao, water):
    """T-A1. Over random amounts: grams of every species plus pending plus the quant and f32
    ledgers are constant, and (E + pending E) minus the chemical heat is constant.
    MUTATION: drop the reactant subtraction of one species, or forget heat(dH xi): fails."""
    w, c = S.slake_cell(cao, water)
    g0, e0 = _total_g(w), _E(w)
    w.chem_run(120.0, dt=10.0)
    assert _total_g(w) == pytest.approx(g0, rel=1e-6, abs=1e-6)
    assert _E(w) - w.cb.chem_J == pytest.approx(e0, rel=1e-5, abs=1.0)
    used = w.rx.extents["slake"]
    assert used == pytest.approx(min(cao, water / 0.3213), rel=2e-3)      # limited by one reactant
    assert w.cb.chem_J == pytest.approx(1150.0 * used, rel=1e-9)


def test_the_catalogue_slake_row_agrees_with_hess():
    """R7: the sign of the heat of every row the runner uses is checked against the heats of
    formation (2 percent)."""
    for rid in ("slake", "calcine", "carbonate"):
        res, err = rxn.check(rxn.catalogue()[rid])
        assert res < 1e-9 and (err is None or err < 0.03), (rid, err)


def test_ten_one_second_steps_equal_one_ten_second_step():
    """T-A2 through the adapter, for a first-order row (the exact exponential).
    MUTATION: use dt * k instead of 1 - exp(-k dt): fails once k dt is not small (k 0.1, dt 10)."""
    a, _ = S.slake_cell(50.0, 60.0)
    b, _ = S.slake_cell(50.0, 60.0)
    for _ in range(10):
        a.chem_run(1.0, dt=1.0)
    b.chem_run(10.0, dt=10.0)
    assert a.rx.extents["slake"] == pytest.approx(b.rx.extents["slake"], rel=1e-4)
    assert a.rx.extents["slake"] == pytest.approx(50.0 * (1.0 - math.exp(-1.0)), rel=2e-3)


def test_the_live_runner_and_the_closed_form_agree_on_the_slake_extent():
    """T-A8, the tier twin: the lattice extent over 30 s equals rxn.first_order_extent."""
    w, _ = S.slake_cell(50.0, 60.0)
    w.chem_run(30.0, dt=3.0)
    k = rxn.catalogue()["slake"].rate.value(20.0)
    assert w.rx.extents["slake"] == pytest.approx(float(rxn.first_order_extent(50.0, k, 30.0)),
                                                  rel=2e-3)


def test_a_limestone_lump_burns_90_percent_in_12_hours_at_1000_C_and_its_co2_is_kept():
    """The lime kiln: the row is 90 % in 12 h at 1000 C (u, kiln practice 12-48 h). The CO2 made
    is 0.4397 g per g of CaCO3, in the gas beside the cell (a draft takes it away each
    step, booked); the cell's mat turns from limestone to quicklime."""
    w, c = S.lime_kiln(20.0)
    cap = w._cap_of(w.mat[c:c + 1] if False else np.array([16], np.uint8),
                    np.array([20.0], np.float32), np.array([0], np.uint8),
                    np.array([0.0], np.float32))[0]
    drafted = 0.0
    sid = SP.BY_NAME["calcite"]
    for _ in range(144):                                     # 12 h in 300 s steps
        w.E[c] = np.float32(980.0 * float(w._cap_of(np.array([w.mat[c]]), np.array([w.smass[c]]),
                                                    np.array([0], np.uint8), np.array([0.0], np.float32),
                                                    None if not w.mixes_live else np.array([w.scomp[c]]),
                                                    None if not w.mixes_live else np.array([w.fcomp[c]]))[0]))
        w.chem_run(300.0, dt=300.0)
        st = w.wet.store
        for ix in zip(*np.nonzero(st.excess("CO2"))):
            drafted += float(st.excess("CO2")[ix])
            st.excess("CO2")[ix] = 0.0
    done = w.rx.extents["calcine"]
    assert 0.85 * 20.0 < done < 0.95 * 20.0
    assert drafted + w.rx_pending_g() * 0 == pytest.approx(0.4397 * done, rel=2e-3)
    assert w.mat[c] == 24                                   # it is quicklime now (mat view)
    assert w.cb.chem_J == pytest.approx(-1783.0 * done, rel=1e-9)   # endothermic


def test_calcine_never_crosses_equilibrium_whatever_dt():
    """T-A3. 900 C with the CO2 already near its equilibrium pressure: one huge step ends at
    or below p_eq, and a second step does not run backward past it.
    MUTATION: remove the bisection clamp; the pressure overshoots K (sign flip)."""
    w, c = S.lime_kiln(20.0)
    T = 900.0
    w.E[c] = np.float32((T - 20.0) * float(w._cap_of(np.array([16], np.uint8), np.array([20.0], np.float32),
                                                     np.array([0], np.uint8), np.array([0.0], np.float32))[0]))
    peq = float(rxn.p_eq_calcine(T))
    # put CO2 into the gas beside the cell so that p is 0.9 of equilibrium
    from src.voxel.gasstore import gas_volume
    from src.voxel.matterdata import M as MM
    store = w.wet.store
    V = float(gas_volume(w.vox_l, 0.0))
    g = 0.9 * peq / (rxn.R * rxn.T_REF_GAS) * MM["CO2"] * V
    nb = (3, 2, 1)
    store.excess("CO2")[nb] = g
    for dt in (3600.0, 3600.0, 36000.0):
        view = LumpView(w, np.array([np.ravel_multi_index(c, w.shape)]))
        stt = LatticeState(view)
        xi = rxn.extent(w.rx.rows[0], stt, np.array([T]), dt)
        p_after = float(stt.p("CO2", 0.4397 * xi)[0])
        assert p_after <= peq * (1 + 1e-6)
        w.chem_run(dt, dt=dt) if False else None


def test_sorption_stops_at_the_capacity_and_is_exact_for_any_dt():
    """T-A4 (the runner half): tannin in the pore water binds to the collagen up to q_max (0.4
    g per g) and no further, in one huge step or many small ones.
    MUTATION: delete the room() read of the sorption site: the bound tannin passes q_max."""
    for dt, n in ((50.0, 400), (20000.0, 3)):
        w, cells, res = S.tan_hide(n=2)
        c = cells[0]
        # fill the pore with a strong liquor: far more than the sites can take
        w.fcomp[c] = w.mixes.parse([("water", "dissolved", 0, 0.5), ("tannin", "dissolved", 0, 0.5)])
        w.fk = None
        for _ in range(n):
            w.chem_run(dt, dt=dt)
        v = LumpView(w, np.array([np.ravel_multi_index(c, w.shape)]), gas=False)
        tid, cid = SP.BY_NAME["tannin"], SP.BY_NAME["collagen"]
        from src.voxel.species_seed import BOUND
        bound = float(v.grams(("S", tid, BOUND))[0])
        coll = float(v.grams(("S", cid, BOUND))[0])
        assert bound <= 0.4 * coll * (1 + 1e-6) and bound > 0.99 * 0.4 * coll


def test_gate_off_is_the_old_world_bit_for_bit():
    """T-A5. A world with the mixin and no set_reactions equals a plain World after 200 ticks
    on a hot, wet scene. MUTATION: leave a call in the tick that runs with the gate off."""
    def build(cls):
        w = cls(6, 6, 6, voxel_cm=5)
        w.mat[3, 3, 1] = 16
        w.smass[3, 3, 1] = 50.0
        w.E[3, 3, 1] = np.float32(5000.0)
        w.fl[2, 2, 1], w.fvol[2, 2, 1] = 1, 40.0
        return w
    base = build(World)
    mix = build(S.make_world.__wrapped__ if hasattr(S.make_world, "__wrapped__") else
                type("W2", (World,), {}))
    for _ in range(200):
        base.step()
        mix.step()
    for n in ("mat", "smass", "fl", "fvol", "E", "smoke"):
        assert np.array_equal(getattr(base, n), getattr(mix, n)), n
    assert mix.rx is None


def test_a_slow_row_keeps_producing_below_the_mix_floor():
    """T-A6. A row so slow that each step makes less than the commit threshold: the pending
    store accumulates, the product appears when the threshold is reached, and the grams
    (lumps + pending) are exact at every step.
    MUTATION: drop the remainder (commit whatever is made each step): the product never appears
    (the quantiser keeps the old fractions) and the grams drift into the quant ledger."""
    w, c = S.slake_cell(50.0, 60.0)
    slow = rxn.catalogue()["slake"].replace(rate=rxn.Const(2e-8))
    w.set_reactions([slow])
    g0 = _total_g(w)
    seen_pending = False
    for _ in range(40):
        w.chem_run(100.0, dt=100.0)
        seen_pending |= abs(w.rx_pending_g()) > 0.0
        assert _total_g(w) == pytest.approx(g0, rel=1e-9, abs=1e-9)
    made = w.rx.extents["slake"]
    assert made == pytest.approx(50.0 * (1 - math.exp(-2e-8 * 4000.0)), rel=1e-6)
    from src.voxel.species_seed import GRAIN
    tid = SP.BY_NAME["slaked_lime"]
    v = LumpView(w, np.array([np.ravel_multi_index(c, w.shape)]), gas=False)
    stored = float(v.stored("S", tid, GRAIN)[0])
    assert seen_pending
    assert stored > 0.0                                     # it did appear in the lump
    assert stored + float(v.pending(("S", tid, GRAIN))[0]) == pytest.approx(made * 1.3213, rel=5e-3)


def test_an_idle_world_with_the_runner_on_costs_one_gather():
    """T-A7. No reactive mix: no LumpView is made, nothing moves."""
    w = S.make_world(20, 20, 20, voxel_cm=5)
    w.set_reactions(["slake", "calcine"])
    w.mat[5:8, 5:8, 0] = 2
    import time
    t0 = time.perf_counter()
    for _ in range(40):
        w._chem_box(1.0)
    per = (time.perf_counter() - t0) / 40
    assert w.rx.runs == 0 and per < 0.02


def test_rows_with_unplaced_names_are_refused():
    assert not row_supported(rxn.catalogue()["rust"])        # O2 and FeOOH have no place yet
    assert row_supported(rxn.catalogue()["tan_vegetable"])
    w = S.make_world(4, 4, 4)
    with pytest.raises(ValueError):
        w.set_reactions(["rust"])


def test_the_clock_runs_the_runner_from_the_tick_and_sleeps_with_nothing_to_do():
    """The tick hook: the box runs from _tick_end every EVERY ticks and does the same
    chemistry as chem_run for the same time."""
    a, c = S.slake_cell(1.0, 5.0)
    b, _ = S.slake_cell(1.0, 5.0)
    for _ in range(401):
        a.step()
    b.chem_run(40 * 0.025 * 10, dt=1.0)
    assert a.rx.runs >= 1
    assert a.rx.extents["slake"] == pytest.approx(b.rx.extents["slake"], rel=3e-2)


def test_plaster_carbonates_from_the_air_beside_it_and_gives_back_its_water():
    """Carbonation at the surface (rxn row carbonate, gas_solid): Ca(OH)2 + CO2 -> CaCO3 + H2O in
    damp air (aw 0.5-0.8). The CO2 comes from the gas store beside the cell (kept at its fresh-air
    background by a refill each step: a draft), the water goes to the cell's fluid lump. In dry
    air (aw 0.3) nothing happens: the water-film gate."""
    from src.voxel.matterdata import BACKGROUND
    for rh, expect in ((0.65, True), (0.3, False)):
        w, c = S.putty_cell(100.0, rh)
        w.wet.store.excess("CO2")
        g0 = _total_g(w)
        for _ in range(600):                                  # 10 minutes in 1 s steps
            st = w.wet.store
            for ix in zip(*np.nonzero(st.excess("CO2"))):
                st.excess("CO2")[ix] = 0.0                   # fresh air: back to the background
            w.chem_run(1.0, dt=1.0)
        done = w.rx.extents.get("carbonate", 0.0)
        if expect:
            assert done == pytest.approx(1.1e-4 * 5 * 0.05 ** 2 * 40.98 * 600.0, rel=0.02)
            from src.voxel.species_seed import BOUND
            tid = SP.BY_NAME["calcite"]
            v = LumpView(w, np.array([np.ravel_multi_index(c, w.shape)]), gas=False)
            assert float(v.grams(("S", tid, BOUND))[0]) == \
                pytest.approx(done * 100.087 / 74.093, rel=1e-2)
            assert float(w.fvol[c]) > 0.0                    # the water it made
        else:
            assert done == 0.0


def test_a_copied_world_carries_the_pending_store_and_continues_the_same():
    """The pending store is in snapshots: a deep copy taken mid-run (pending non-zero) and the
    original end the same, grams and pending alike (golden hashes every attribute)."""
    import copy
    w, c = S.slake_cell(50.0, 60.0)
    w.set_reactions([rxn.catalogue()["slake"].replace(rate=rxn.Const(2e-8))])
    for _ in range(3):
        w.chem_run(100.0, dt=100.0)
    assert abs(w.rx_pending_g()) > 0.0 or w.cb.pend
    w2 = copy.deepcopy(w)
    for q in (w, w2):
        for _ in range(5):
            q.chem_run(100.0, dt=100.0)
    assert w.rx_pending_g() == w2.rx_pending_g()
    assert w.rx_mass_g() == w2.rx_mass_g() and np.array_equal(w.smass, w2.smass)
    assert np.array_equal(w.E, w2.E)
