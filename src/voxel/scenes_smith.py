"""SMITHING STAGE 1 SCENES (wave 15, lane 1e). Each returns what it measured as a plain dict.

There is no new World here: `smith_world()` is the ordinary World with the heat-history gate on
(and the mixin composed in, until the merge step puts it among World's bases).

  anneal_vs_quench    the same bar to the same cold end by two paths: slow cool and quench
  temper_ladder       hardness of a quenched bar after one hour at 200, 300, 400, 500, 600 C
  edge_spine          a 2 mm blade quenched in water and in oil: edge against spine
  overheat            a bar held at 1010 C for 20 minutes: grain and hardenability
  forge_quench        the wave-14a forge bar (a 15 cm block) into a water pot, in the voxel world:
                      the skin cell and the core cell differ, the latent heat is booked
  bar_quench          a 2 cm bar, through its thickness (the slab sampler): skin, core, books
"""
import numpy as np

from . import heathist as H
from . import heathist_data as D
from . import heathist_colour as CO
from . import layer_laws as LL
from .heathist_slab import Slab
from .heathist_world import HeatHistoryMixin


def smith_world(*a, **k):
    from .sim import World
    cls = World if issubclass(World, HeatHistoryMixin) else type("SmithWorld", (HeatHistoryMixin, World), {})
    w = cls(*a, **k)
    w.heat_history = True
    return w


def _path(rates, c, mn=0.5, T_top=850.0, soak_s=300.0):
    s = H.new_state(c, mn)
    H.step(s, T_top, 60.0)
    H.step(s, T_top, soak_s)
    return s


def _cool(s, rate, T_end=20.0, dt=0.05):
    T = float(s["Tp"][0])
    while T > T_end:
        T = max(T_end, T - rate * dt)
        H.step(s, T, dt)
        if rate < 1.0:
            dt = 30.0
    return s


def anneal_vs_quench(c=0.2, mn=0.5):
    out = {}
    for name, rate in (("quench", 1500.0), ("air", 3.0), ("furnace", 0.03)):
        s = _cool(_path(None, c, mn), rate)
        out[name] = dict(HV=float(H.hardness(s)[0]), fM=float(s["fM"][0]), retained=float(s["fg"][0]))
    return out


def temper_ladder(c=0.4, mn=0.7):
    s0 = _cool(_path(None, c, mn), 1500.0)
    out = {"as_quenched": float(H.hardness(s0)[0])}
    for T in (200.0, 300.0, 400.0, 500.0, 600.0):
        s = {k: v.copy() for k, v in s0.items()}
        H.step(s, T, 300.0)
        H.step(s, T, 3600.0)
        _cool(s, 5.0)
        out[int(T)] = dict(HV=float(H.hardness(s)[0]), P=float(s["P"][0]),
                           tough=float(H.toughness_factor(s)[0]))
    return out


def overheat(c=0.4, mn=0.7):
    out = {}
    for T, mins in ((850.0, 5), (1010.0, 20)):
        s = H.new_state(c, mn)
        H.step(s, T, 30.0)
        H.step(s, T, mins * 60.0)
        d = float(s["d"][0])
        rate = float(H.tau_nose(c, mn, d))
        out[T] = dict(d_um=d, tau_nose_s=rate, tough=float(H.toughness_factor(H.new_state(c, mn))[0]) *
                      (D.GRAIN_REF_UM / d) ** D.GRAIN_HP)
    return out


def edge_spine(c=0.8, mn=0.5, edge_mm=0.4, spine_mm=8.0):
    """A blade is a wedge: thin at the edge, thick at the spine. Each station of the wedge is
    cooled as a slab of its own LOCAL thickness (a 1-D estimate, ledger heat_history.wedge_local,
    bound: it ignores the conduction along the blade, which only evens the two out)."""
    out = {}
    for med in ("water", "oil"):
        hv = []
        for mm in (edge_mm, spine_mm):
            sl = Slab(mm * 1e-3, c=c, mn=mn, n=11)
            sl.heat_to(850.0)
            sl.quench(med, 20.0, 0.004 if mm < 1.0 else 0.02)
            hv.append(float(sl.hv()[len(sl.hv()) // 2]))
        out[med] = dict(edge=hv[0], spine=hv[1], gap=hv[0] - hv[1])
    return out


def bar_quench(c=0.8, mn=0.5, thickness=0.02, medium="water", seconds=60.0, dt=0.01):
    sl = Slab(thickness, c=c, mn=mn, n=11)
    sl.heat_to(850.0)
    b0 = sl.books()
    sl.quench(medium, seconds, dt)
    hv = sl.hv()
    return dict(skin_HV=float(hv[0]), core_HV=float(hv[-1]), skin_fM=float(sl.s["fM"][0]),
                core_fM=float(sl.s["fM"][-1]), books_drift=(sl.books() - b0) / abs(b0),
                latent_booked_J_m2=float(np.sum(sl.m * H.enthalpy(sl.s) * 1000.0)))


def _film_cool(w, idx, medium, dt):
    """THE SCENE'S STAND-IN FOR A BOILING FILM. The lattice conducts across a 5 cm cell face, which
    is about 24 W/m2/K into water: a hundred times slower than a real quench (heat.py has no
    boiling film coefficient; reported). Here every exposed face of the bar's cells loses
    h(T_cell) A (T - 20) dt, with h from heathist_data.H_QUENCH, and it is BOOKED on `shed` (heat
    that left the lattice), so total_energy stays honest."""
    from .materials import AMBIENT
    tab = D.H_QUENCH[medium]
    nx, ny, nz = w.shape
    mat = w.mat
    A = (0.1 * w.scale) ** 2
    Ef = w.E.ravel()
    T = w.T().ravel()
    xs, ys, zs = np.unravel_index(idx, w.shape)
    n_exp = np.zeros(len(idx))
    for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
        x2, y2, z2 = xs + dx, ys + dy, zs + dz
        ok = (x2 >= 0) & (x2 < nx) & (y2 >= 0) & (y2 < ny) & (z2 >= 0) & (z2 < nz)
        nb = np.zeros(len(idx), bool)
        nb[ok] = mat[x2[ok], y2[ok], z2[ok]] != mat[xs[ok], ys[ok], zs[ok]]
        n_exp += nb
    # a cell's face is at half a cell from its centre: the cell mean is hotter than its face.
    # The face runs at the mean (estimate, bound: the face cools faster by about 2x for Bi near 1).
    h = np.interp(T[idx], *zip(*tab))
    q = np.minimum(h * A * n_exp * np.maximum(T[idx] - AMBIENT, 0.0) * dt, Ef[idx])
    Ef[idx] -= q.astype(Ef.dtype)
    w.shed += float(q.sum())


def forge_quench(c=0.8, mn=0.5, T0=1000.0, seconds=30.0, medium="water"):
    """A 15 cm block of steel at T0 (about the 14a forge's 1010 C) quenched in the voxel world
    with the gate on. Cells are 5 cm: the block is 3 x 3 x 3 cells, one centre cell and a skin of
    26. Returns the centre and the mean skin hardness, the latent heat booked (phase_J) and the
    energy drift of total_energy + phase_J (a quench scene must close)."""
    from .materials import AMBIENT, IRON, STONE, TICK_S
    w = smith_world(16, 16, 12, voxel_cm=5)
    w.fill(0, 16, 0, 16, 0, 1, STONE)
    w.fill(6, 9, 6, 9, 2, 5, IRON)
    eid = w.register("bar")
    w.stamp(eid, w.mat == IRON)
    w.hh_register(eid, c=c, mn=mn)
    idx = np.flatnonzero((w.mat == IRON).ravel())
    w.E.ravel()[idx] = (T0 - AMBIENT) * w.heat_capacity().ravel()[idx]
    s = w._hh[eid]["s"]
    s["Tp"][:] = T0
    dh = H.step(s, np.full(len(idx), T0), 120.0)     # it soaked in the fire: austenite, fine grain
    w.hh_book_soak(float(w.smass.ravel()[idx].astype(float) @ dh))   # what the fire put into the state
    e0 = w.total_energy() + w.phase_J
    phase0 = w.phase_J
    for _ in range(int(seconds / TICK_S)):
        _film_cool(w, idx, medium, TICK_S)
        w.step()
    hv = w.heat_hv(eid)
    cells = w._hh[eid]["idx"]
    ctr = np.flatnonzero(cells == np.ravel_multi_index((7, 7, 3), w.shape))
    skin = np.setdiff1d(np.arange(len(cells)), ctr)
    # the cell mean cannot see the gradient (a 5 cm cell cools as one lump: ledger row
    # heat_history.cell_mean), so the same block is also run through its thickness
    through = bar_quench(c, mn, thickness=0.15, medium=medium, seconds=1200.0, dt=0.25)
    return dict(through=through, core_HV=float(hv[ctr[0]]), skin_HV=float(hv[skin].mean()),
                T_core=float(w.T()[7, 7, 3]), phase_J=float(w.phase_J), phase_J_start=float(phase0),
                energy_drift_J=float(w.total_energy() + w.phase_J - e0), energy_J=float(e0),
                scale_nm_core=float(w.hh_scale_nm(eid)[ctr[0]]), scale_nm_skin=float(w.hh_scale_nm(eid)[skin].mean()))
