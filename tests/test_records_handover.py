"""T2 records: hand-over between a record and the lattice (spec 6.4, F4; test 12).

Tags (plan section 9): test_echo_* echo, test_codeonly_* code-only, test_u_* unverified data.
All tests here are exactness, conservation or arithmetic identities and may gate a scene."""
import numpy as np
import pytest

from src.voxel import handover as HO
from src.voxel import record as R
from src.voxel import record_data as rd

DX = 0.05
AX = np.eye(3)


# ── hysteresis ───────────────────────────────────────────────────────────────
def test_thresholds_are_half_and_one_cell():
    assert (HO.HYST_DOWN, HO.HYST_UP) == (0.5, 1.0)
    assert HO.decide(0.99 * DX, DX, True) and not HO.decide(1.0 * DX, DX, True)
    assert not HO.decide(0.51 * DX, DX, False) and HO.decide(0.5 * DX, DX, False)


@pytest.mark.parametrize("start", [True, False])
def test_no_thrash_while_d_star_wanders_between_0p6_and_0p9_cell(start):
    rng = np.random.default_rng(0)
    state = start
    flips = 0
    for d in rng.uniform(0.6, 0.9, 5000) * DX:
        new = bool(HO.decide(d, DX, state))
        flips += new != state
        state = new
    assert flips == 0 and state == start
    # and the real crossings happen once each, at the far end of the band
    seq = np.concatenate([np.linspace(0.3, 1.2, 100), np.linspace(1.2, 0.3, 100)]) * DX
    state, flips = True, []
    for i, d in enumerate(seq):
        new = bool(HO.decide(d, DX, state))
        if new != state:
            flips.append((i, round(d / DX, 2)))
        state = new
    assert len(flips) == 2 and flips[0][1] >= 1.0 and flips[1][1] <= 0.5


# ── footprint and restamp ────────────────────────────────────────────────────
def _blade(center=(0.2, 0.125, 0.125), axes=AX):
    return HO.footprint(center, axes, (0.40, 0.04, 0.002), DX)


def test_blade_footprint_fill_is_about_003_not_03():
    cells, vol = _blade()
    assert abs(vol.sum() - 0.40 * 0.04 * 0.002) < 1e-15        # 32 cm3
    f = HO.fills(vol, DX)
    assert len(cells) == 8 and f.max() < 0.04
    assert f.mean() == pytest.approx(0.032, rel=1e-6)         # 32 cm3 over eight 125 cm3 cells
    assert f.max() < 1.0


def test_footprint_of_a_rotated_blade_keeps_its_volume():
    th = np.deg2rad(30.0)
    ax = np.array([[np.cos(th), np.sin(th), 0], [-np.sin(th), np.cos(th), 0], [0, 0, 1.0]])
    cells, vol = _blade(axes=ax)
    assert abs(vol.sum() - 0.40 * 0.04 * 0.002) < 1e-15
    assert HO.fills(vol, DX).max() < 0.2 and len(cells) >= 8


def test_restamp_sum_equals_mass_fill_below_one_and_residue_is_tiny():
    cells, vol = _blade()
    smass = np.zeros((10, 10, 10), dtype=np.float32)
    flat = np.ravel_multi_index(tuple(cells.T), smass.shape)
    rho = 7800.0
    mass_g = rho * vol.sum() * 1000.0                          # 249.6 g
    cap = np.full(len(flat), rho * DX ** 3 * 1000.0)
    res, left = HO.restamp(smass, flat, mass_g, vol, cap)
    placed = smass.astype(np.float64).sum()
    assert left == 0.0 and abs(placed + res - mass_g) < 1e-12
    assert abs(res) < 1e-4 * mass_g * 2.0 ** -23 * 1e4 and abs(res) / mass_g < 2.0 ** -23
    fill = smass.reshape(-1)[flat] / cap
    assert fill.max() < 1.0 and fill.max() == pytest.approx(0.032, rel=1e-3)
    # cell density follows the volume to float32 resolution
    want = mass_g * vol / vol.sum()
    assert np.allclose(smass.reshape(-1)[flat], want, rtol=2.0 ** -22)


def test_restamp_never_fills_a_cell_above_one_even_when_the_footprint_shrinks():
    """A drying thing loses water: its volume shrinks but its mass left is dense. Weights
    that would overfill a cell spill to the neighbours; nothing is above capacity."""
    rho = 900.0
    cap = np.full(4, rho * DX ** 3 * 1000.0)                   # 101 g per cell
    w = np.array([10.0, 1.0, 1.0, 1.0])
    smass = np.zeros(4, dtype=np.float32)
    res, left = HO.restamp(smass, np.arange(4), 300.0, w, cap)
    assert left == 0.0 and (smass <= cap).all() and abs(smass.sum() + res - 300.0) < 1e-9
    assert smass[0] == pytest.approx(cap[0], rel=1e-6)
    # the water goes: less mass, the same cells, fills still at or below one
    res2, left2 = HO.restamp(smass, np.arange(4), 150.0, w, cap)
    assert (smass <= cap).all() and left2 == 0.0
    # more than the cells can hold: reported, not hidden
    res3, left3 = HO.restamp(smass, np.arange(4), 500.0, w, cap)
    assert left3 == pytest.approx(500.0 - cap.sum(), rel=1e-6) and (smass <= cap * (1 + 1e-6)).all()


def test_restamp_only_when_the_set_changes_or_fill_moves_by_an_eighth():
    idx = np.arange(5)
    f = np.full(5, 0.03)
    assert not HO.needs_restamp(idx, idx, f, f + 0.1)          # moved by 0.1 < 1/8
    assert HO.needs_restamp(idx, idx, f, f + np.array([0, 0, 0, 0, 0.13]))
    assert HO.needs_restamp(idx, np.arange(1, 6), f, f)        # the occupied set changed


# ── exact crossings ──────────────────────────────────────────────────────────
def _record_matter(recs, host=7):
    g = np.zeros(rd.NS)
    g[rd.SP["steel"]], g[rd.SP["water"]], g[rd.SP["hide"]] = 249.6000001234, 0.1234567891234, 33.3333333333
    return g, 12345.6789012345


def test_round_trip_conserves_mass_per_species_energy_and_count_to_1e12():
    recs = R.Records()
    g, E = _record_matter(recs)
    w = np.array([5.0, 3.0, 3.0, 1.0, 0.7])
    m, Ec = HO.to_lattice(recs, 7, g, E, w)
    counts = HO.largest_remainder(1000, w)
    assert counts.sum() == 1000 and m.dtype == np.float32
    # while it sits in the lattice the ledger still sees everything: lattice + records_owned
    o = R.records_owned(recs)
    assert np.allclose(m.astype(np.float64).sum(axis=0) + o.g, g, rtol=1e-13, atol=1e-13)
    assert abs(Ec.astype(np.float64).sum() + o.E - E) / E < 1e-13
    g2, E2 = HO.from_lattice(recs, 7, m, Ec)
    assert np.allclose(g2, g, rtol=1e-12, atol=0) and abs(E2 - E) / E < 1e-12
    assert counts.sum() == 1000 and not recs.rem               # the remainders were given back
    assert R.records_owned(recs).E == 0.0


def test_float32_lattice_sum_differs_by_at_most_an_ulp_per_cell():
    recs = R.Records()
    g, E = _record_matter(recs)
    m, _ = HO.to_lattice(recs, 7, g, E, np.ones(6))
    err = np.abs(m.astype(np.float64).sum(axis=0) - g)
    assert (err <= 6 * 2.0 ** -23 * np.maximum(g, 1e-300) + 1e-300).all()


def test_remainder_is_carried_over_1000_crossings():
    recs = R.Records()
    g0, E0 = _record_matter(recs)
    w = np.array([0.31, 0.27, 0.19, 0.23])
    g, E = g0.copy(), E0
    naive_g, naive_E = g0.copy(), E0
    for _ in range(1000):
        m, Ec = HO.to_lattice(recs, 7, g, E, w)
        g, E = HO.from_lattice(recs, 7, m, Ec)
        mn, En = (naive_g * w[:, None]).astype(np.float32), (naive_E * w).astype(np.float32)   # drops the residue
        naive_g, naive_E = mn.astype(np.float64).sum(axis=0), float(En.astype(np.float64).sum())
    nz = g0 > 0
    drift = np.max(np.abs(g - g0)[nz] / g0[nz])
    assert drift < 1e-12 and abs(E - E0) / E0 < 1e-12
    assert np.max(np.abs(naive_g - g0)[nz] / g0[nz]) > 1e-9           # the control: without `rem` it drifts
    assert not recs.rem


def test_remainder_shows_in_the_ledger_term_until_the_next_read():
    recs = R.Records()
    g, E = _record_matter(recs)
    m, Ec = HO.to_lattice(recs, 9, g, E, np.array([1.0, 2.0, 3.0]))
    assert recs.rem                                           # float32 cannot hold it all
    o = R.records_owned(recs)
    assert o.E == pytest.approx(E - float(Ec.astype(np.float64).sum()), abs=1e-12)
    assert any(o.g != 0.0)
    HO.from_lattice(recs, 9, m, Ec)
    assert R.records_owned(recs).E == 0.0 and not R.records_owned(recs).g.any()


def test_a_full_cell_keeps_the_rest_in_the_remainder_not_in_the_void():
    recs = R.Records()
    g = np.zeros(rd.NS)
    g[rd.SP["steel"]] = 500.0
    cap = np.full(2, 100.0)
    m, Ec = HO.to_lattice(recs, 3, g, 0.0, np.ones(2), cap_g=cap)
    assert m[:, rd.SP["steel"]].sum() == pytest.approx(200.0, rel=1e-6)
    assert R.records_owned(recs).g[rd.SP["steel"]] == pytest.approx(300.0, rel=1e-6)
    g2, _ = HO.from_lattice(recs, 3, m, Ec)
    assert g2[rd.SP["steel"]] == pytest.approx(500.0, rel=1e-12)


def test_largest_remainder_is_exact_and_deterministic():
    for n in (0, 1, 7, 1000, 123457):
        for w in ([1, 1, 1], [3, 1, 4, 1, 5], [0.5, 0.25, 0.25]):
            c = HO.largest_remainder(n, w)
            assert c.sum() == n and (c >= 0).all()
            assert np.array_equal(c, HO.largest_remainder(n, w))
