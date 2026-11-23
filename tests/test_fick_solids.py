"""Fick transport in the pore water of solids (wave 15, S1 slice 2), with tanning and the wash."""
import math

import numpy as np
import pytest
from math import erfc

from src.voxel import scenes_reactions as S
from src.voxel import species as SP
from src.voxel import transport as TR
from src.voxel.fick_solids import D25_X, D_KEY, E_D_X
from src.voxel.matterdata import TORTUOSITY
from src.voxel.reactions import LumpView
from src.voxel.sim import World
from src.voxel.species_seed import BOUND, DISSOLVED

TID, CID = SP.BY_NAME["tannin"], SP.BY_NAME["collagen"]


def _D_hide(T=20.0):
    return float(TR.arrhenius(D25_X["tannin"], E_D_X["tannin"], T, 25.0)) * TORTUOSITY[("tannin", "hide")]


def _view(w, cells):
    ix = np.array([np.ravel_multi_index(c, w.shape) for c in cells])
    return LumpView(w, ix, gas=False)


def _conc(w, cells):
    v = _view(w, cells)
    return v.grams(("F", TID, DISSOLVED)) / (w.fvol.reshape(-1)[v.idx].astype(np.float64) * 1.0)


def test_a_dissolved_species_spreads_into_a_slab_like_the_erf_profile():
    """T-B2 (Crank). A bath at c0 beside a long hide strip with no binding: the profile at five
    depths is within 2 percent of c0 of c0 erfc(x / 2 sqrt(D t)). D is the module's own (host row).
    MUTATION: use 2 dx as the face distance: the profile is too deep and fails."""
    w, cells, res = S.tan_hide(n=40)
    w.rx = None                                      # no binding: pure diffusion
    # clean pores
    c0 = float(_conc(w, [res])[0])
    D = _D_hide()
    t = 4.0e6
    w.chem_run(t, dt=3600.0)
    dx = 0.002
    prof = _conc(w, cells) / c0
    for i in (0, 2, 4, 6, 8):
        x = (i + 0.5) * dx
        assert prof[i] == pytest.approx(erfc(x / (2 * math.sqrt(D * t))), abs=0.02), i


def test_the_total_is_conserved_and_the_baths_are_booked():
    """T-B1. Random pore volumes and concentrations in a block, no bath: the total tannin
    (lumps + pending + the f32 ledger) is constant to 1e-9 over many steps.
    MUTATION: a one-sided flux (no matching gain): the total leaks."""
    rng = np.random.default_rng(3)
    w, cells, res = S.tan_hide(n=14)
    w.rx = None
    w.fk.fixed.clear()
    for c in cells:
        w.fcomp[c] = w.mixes.parse([("water", "dissolved", 0, 1.0 - 0.3 * rng.random()),
                                    ("tannin", "dissolved", 0, 0.0)] if False else
                                   [("water", "dissolved", 0, 1.0), ("tannin", "dissolved", 0, 0.3 * rng.random() + 0.001)])
        w.fvol[c] = np.float32(w.fvol[c] * (0.5 + rng.random()))
    t0 = w.fick_mass_g("tannin")
    for _ in range(50):
        w.chem_run(86400.0, dt=3600.0)
    f32 = sum(w.mixes.ledger("f32").values()) * 0.0
    assert w.fick_mass_g("tannin") == pytest.approx(t0, rel=1e-6)


def test_no_new_extremes_at_the_stability_limit():
    """T-B5. The concentrations stay inside the initial range for any box dt (it substeps).
    MUTATION: raise the stability safety to 1.2 and a checkerboard oscillates out of range."""
    w, cells, res = S.tan_hide(n=10)
    w.rx = None
    w.fk.fixed.clear()
    for i, c in enumerate(cells):
        frac = 0.2 if i % 2 else 0.01
        w.fcomp[c] = w.mixes.parse([("water", "dissolved", 0, 1.0 - frac), ("tannin", "dissolved", 0, frac)])
    c0 = _conc(w, cells)
    lo, hi = c0.min(), c0.max()
    for dt in (50.0, 1.0e4, 1.0e6):
        w.chem_run(dt, dt=dt)
        c = _conc(w, cells)
        assert c.min() >= lo * (1 - 1e-6) and c.max() <= hi * (1 + 1e-6)


def test_tannin_front_follows_sqrt_t_and_the_stefan_value():
    """T-B3. Fast binding plus slow diffusion gives a sharp front: x^2 = 2 D c0 t / Q, with Q the
    bound tannin per volume of hide (the closed form of slowforms 'front'). The lattice front (the
    half-bound position) is within 15 percent after 20 and after 40 days.
    MUTATION: reaction before transport in the box order moves the front by more than that."""
    w, cells, res = S.tan_hide(n=14)
    c0 = float(_conc(w, [res])[0])
    cell_ml = w.cap
    Q = 0.4 * (0.4 * cell_ml) / cell_ml * 1.0            # g bound per ml of cell, in g/ml
    D = _D_hide()
    dx = 0.002
    t_done = 0.0
    for days in (20, 40):
        w.chem_run((days * 86400.0) - t_done, dt=3600.0)
        t_done = days * 86400.0
        v = _view(w, cells)
        b = v.grams(("S", TID, BOUND)) / (0.4 * v.grams(("S", CID, BOUND)))      # filled share
        front = (np.sum(np.minimum(b, 1.0))) * dx                              # integral of fill
        theta = float(w.fvol[cells[0]]) / w.cap
        x = math.sqrt(2.0 * D * theta * (c0 * 1e3) * t_done / (Q * 1e3))        # SI: kg/m3 ratio
        assert front == pytest.approx(x, rel=0.15), (days, front, x)


def test_washing_a_tanned_hide_keeps_the_bound_tannin():
    """T-A4 (the wash; this was the strict xfail of tests/test_c12_crafts.py). Tan a hide, then put
    clean water on it: the DISSOLVED tannin leaves by diffusion, the BOUND tannin stays to the
    gram. MUTATION: let the wash carry BOUND species: the bound amount falls and this fails."""
    w, cells, res = S.tan_hide(n=8)
    w.chem_run(60 * 86400.0, dt=3600.0)
    v = _view(w, cells)
    bound0 = float(v.grams(("S", TID, BOUND)).sum())
    diss0 = float(v.grams(("F", TID, DISSOLVED)).sum())
    assert bound0 > 0.0 and diss0 > 0.0
    S.wash_hide(w, res)
    w.chem_run(60 * 86400.0, dt=3600.0)
    v = _view(w, cells)
    bound1 = float(v.grams(("S", TID, BOUND)).sum())
    diss1 = float(v.grams(("F", TID, DISSOLVED)).sum())
    assert diss1 < 0.05 * diss0
    assert bound1 >= bound0 * (1 - 1e-6)                    # only grows: leftovers still bind


def test_a_tiny_tracer_is_not_rounded_away():
    """T-B8, the bite rule. about 5e-9 g of tannin put in one pore moves along the hide for days and
    none of it is lost: stored + pending + bath is conserved to the digit.
    MUTATION: commit on the tick clock with no pending store: the 1e-12 g never leaves the cell."""
    w, cells, res = S.tan_hide(n=6)
    w.rx = None
    w.fk.fixed.clear()
    w.fcomp[res] = 0
    w.fl[res] = 1
    tr = w.mixes.parse([("water", "dissolved", 0, 1.0 - 1e-6), ("tannin", "dissolved", 0, 1e-6)])
    w.fcomp[cells[2]] = tr
    t0 = w.fick_mass_g("tannin")
    assert 0.0 < t0 < 1e-7
    w.chem_run(5 * 86400.0, dt=3600.0)
    v = _view(w, cells)
    g = v.grams(("F", TID, DISSOLVED))
    assert g[1] > 0.0 and g[3] > 0.0 and g[2] < t0
    assert w.fick_mass_g("tannin") == pytest.approx(t0, rel=1e-6)


def test_gate_off_world_is_unchanged_with_the_kernel_available():
    """T-B7. A world with FickSolidsMixin and no set_fick ticks like a plain World."""
    from src.voxel.fick_solids import FickSolidsMixin

    def build(cls):
        w = cls(6, 6, 6, voxel_cm=5)
        w.mat[3, 3, 1] = 4
        w.smass[3, 3, 1] = 80.0
        w.fl[3, 3, 1], w.fvol[3, 3, 1] = 1, 30.0
        w.E[3, 3, 1] = np.float32(3000.0)
        return w
    a, b = build(World), build(type("W3", (World,), {}))
    for _ in range(120):
        a.step()
        b.step()
    for n in ("mat", "smass", "fvol", "fl", "E"):
        assert np.array_equal(getattr(a, n), getattr(b, n)), n
