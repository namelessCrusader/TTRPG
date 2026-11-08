"""Run one measurement and write what it found.

    python3 run.py <experiment> [<out_dir>]

Each experiment is one function below, `b_*` for bonds and `c_*` for the
continuum. It writes <out_dir>/<experiment>.json (the numbers) and
<out_dir>/<experiment>.txt (side views as text: matplotlib is not installed).
One experiment per process, so each heavy run can sit under the shared lock
and memory cap on its own."""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bonds as B                       # noqa: E402
import scenes as S                      # noqa: E402
from common import Clock, ascii_side, long_axis, mpm_momentum   # noqa: E402

TICK = 0.025
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "out")


def _cost(clock, ticks, substeps, n, nb=None):
    return dict(wall_s=round(clock.s, 3), ticks=ticks, substeps=int(substeps),
                substeps_per_tick=round(substeps / max(ticks, 1), 1),
                ms_per_tick=round(1e3 * clock.s / max(ticks, 1), 3),
                us_per_substep=round(1e6 * clock.s / max(substeps, 1), 2),
                s_per_sim_s=round(clock.s / max(ticks * TICK, 1e-9), 3),
                pieces=int(n), bonds=None if nb is None else int(nb))


def _warm(b):
    """Compile (or load the compiled) solver outside the clock."""
    B._run(0, 1e-6, 0.0, b.x, b.v, b.q, b.ell, b.m, b.Ibinv, b.rc, b.cda, b.kin,
           b.x0, b.amp, b.freq, b.kwin, b.fext, b.kc, b.cc, b.bi, b.bj, b.ai, b.aj,
           b.Q0, b.kn, b.ks, b.kb, b.kt, b.cn, b.cs, b.cb, b.ct, b.A, b.rb, b.Ib,
           b.Jb, b.sig, b.tau, b.alive, b.tbreak, b.smax, b.adj_ptr, b.adj_b,
           b.wlo, b.whi, b.mu, b.zeta_c, b.g, B.AIR_DENS, b.pp, b.ledger.copy())


def _resid(b):
    """The conservation residual, against the scale of the motion."""
    P, L = b.momentum()
    dp, dl = b.residual()
    return dict(dP=dp, dL=dl, P_scale=float(b.M * 1.0),
                L_scale=float(max(np.linalg.norm(L), b.M * 0.1)))


# ══ the tree ═════════════════════════════════════════════════════════════════
# Both models: settle, then a trunk shaker (an orchard machine: a clamp at
# 0.55 m driven +-1.5 cm at 12 Hz, eased in and out) from 0.5 s to 2.0 s,
# then rest. Bonds then fell the tree: the shaker lets go, the notch makes
# the stump a hinge, and a 300 N push at 1.55 m starts it over.
SHAKE = dict(amp=(0.015, 0.0, 0.0), freq=12.0, on=0.5, off=2.0)
T_REST, T_FELL, T_END = 2.5, 2.5, 7.0


def _g_ramp(t, T=0.4):
    """Gravity eased on over the first 0.4 s: a structure built unloaded and
    loaded all at once rings at twice its static stress, which snapped the
    weakest petioles in the first millisecond."""
    u = min(max(t / T, 0.0), 1.0)
    return 9.81 * u * u * (3 - 2 * u)


def b_tree_autumn():
    return b_tree(B.PETIOLE_AUTUMN)


def b_tree_mat():
    """The green tree with src/mpm's wood strength (1e7 Pa) instead of the
    handbook's: how much of the tree breaks when it lands."""
    return b_tree(wood=(B.WOOD_L, B.WOOD_T))


def b_tree(petiole=B.PETIOLE, wood=(B.WOOD_L_HB, B.WOOD_T_HB)):
    b, ix = S.tree(petiole=petiole, wood=wood)
    b.finish(stiff=1.0, strength_scale=1.0)
    S.root(b, ix)
    b.retime()
    _warm(b)
    clamp = ix["clamp"]                        # gripping the trunk at 0.55 m
    sprays, pet = ix["sprays"], ix["petioles"]
    clock, sub, ticks = Clock(), 0, 0
    frames, res = [], {}
    first_touch = None
    land_t = np.full(len(sprays), np.nan)
    land_x = np.full((len(sprays), 3), np.nan)
    det_x = np.full((len(sprays), 3), np.nan)
    sway, tip_path = [], []
    b.pin(clamp, **SHAKE)
    push = ix["trunk"][7]
    k_fell, k_end = round(T_FELL / TICK), round(T_END / TICK)
    for k in range(k_end):
        if k == k_fell:
            res["shake_rest"] = _resid(b)
            b.alive[ix["grip"]] = False        # the shaker lets go
            b.rebase()
            S.fell(b, ix)
            b.fext[push, 0] = 300.0
        if k == k_fell + 20:
            b.fext[push, 0] = 0.0
        b.g = _g_ramp(k * TICK)
        with clock:
            sub += b.step(TICK)
        ticks += 1
        if k < round(T_REST / TICK):
            sway.append(b.x[ix["lead"][-1], 0] - 1.0)
        if k >= k_fell and (k - k_fell) % 20 == 0:
            top = b.x[ix["trunk"][-1]]
            tip_path.append([round(b.t, 3), _f(top[0]), _f(top[2])])
        gone = ~b.alive[pet]
        for q in np.flatnonzero(gone & np.isnan(det_x[:, 0])):
            det_x[q] = b.x[sprays[q]]
        for q in np.flatnonzero(gone & np.isnan(land_t) & (b.x[sprays, 2] < 0.1)):
            land_t[q], land_x[q] = b.t, b.x[sprays[q]]
        touched = False
        if first_touch is None and k >= k_fell:
            crown = np.isin(b.tag, (2, 3))
            if (b.x[crown, 2] < 0.15).any():
                first_touch, touched = b.t, True
        if k in (0, 49, k_fell + 40, k_end - 1) or touched:
            frames.append((round(b.t, 3), _tree_view(b, ix)))
    res["fell_end"] = _resid(b)
    tb = b.tbreak[pet]
    when = dict(settle=(0, SHAKE["on"]), shake=(SHAKE["on"], SHAKE["off"]),
                rest=(SHAKE["off"], T_FELL),
                fall=(T_FELL, first_touch or T_END),
                impact=(first_touch or T_END, (first_touch or T_END) + 0.3),
                after=((first_touch or T_END) + 0.3, T_END + 1))
    counts = {k: int(((tb > lo) & (tb <= hi)).sum()) for k, (lo, hi) in when.items()}
    det = ~np.isnan(tb)
    landed = ~np.isnan(land_t)
    fall_v = np.full(len(sprays), np.nan)
    fall_v[landed] = (det_x[landed, 2] - 0.08) / np.maximum(land_t[landed] - tb[landed], 1e-9)
    # the sprays the shaker took and the ones that came off once it was
    # felled, kept apart: they fall from different heights at different times
    groups = dict(all=landed, shaken_off=landed & (tb <= T_FELL),
                  after_felling=landed & (tb > T_FELL))

    def _land(sel):
        if not sel.any():
            return None
        return dict(n=int(sel.sum()), mean_descent_speed_ms=_stats(fall_v[sel]),
                    std_x=_f(np.std(land_x[sel, 0])), std_y=_f(np.std(land_x[sel, 1])),
                    x_range=[_f(land_x[sel, 0].min()), _f(land_x[sel, 0].max())])
    wood_b = np.flatnonzero(~b.alive & np.isin(b.btag, (1, 2, 3, 9)))
    names = {1: "trunk", 2: "leader", 3: "spoke", 9: "hinge"}
    trunk = b.x[ix["trunk"]]
    facts = dict(
        sprays=int(len(sprays)), spray_kg=S.SPRAY_M,
        detached=int(det.sum()), detached_by_phase=counts,
        first_detach_s=None if not det.any() else float(np.nanmin(tb)),
        crown_first_touch_s=first_touch,
        hinge_broke_s=None if b.alive[ix["root"]] else float(b.tbreak[ix["root"]]),
        landed=int(landed.sum()),
        detach_height_m=_stats(det_x[det, 2]),
        landing_by_group={k: _land(v) for k, v in groups.items()},
        # every piece of wood that broke, in order: which part, when, and
        # how high the bond sat when the tree was built
        wood_breaks=[dict(part=names[int(b.btag[k])], t=_f(b.tbreak[k]),
                          z0=_f(0.5 * (b.x0[b.bi[k], 2] + b.x0[b.bj[k], 2])))
                     for k in wood_b[np.argsort(b.tbreak[wood_b])]],
        trunk_tip_path=tip_path,
        leaf_groups_at_end=int(len(np.unique(b.fragments()[sprays]))),
        crown_sway_during_shake_m=dict(max=_f(np.max(np.abs(sway))), rms=_f(np.sqrt(np.mean(np.square(sway))))),
        trunk_angle_deg_at_end=long_axis(trunk),
        trunk_top_at_end=[_f(v) for v in b.x[ix["trunk"][-1]]],
        broken_by_part={name: int((~b.alive[b.btag == t]).sum())
                        for t, name in ((1, "trunk"), (2, "leader"), (3, "spokes"),
                                        (9, "hinge"))},
    )
    return dict(model="bonds", scene="tree", stiff=b.stiff, h=b.h,
                petiole_strength_Pa=petiole[2], wood_strength_Pa=[wood[0][2], wood[1][2]],
                cost=_cost(clock, ticks, sub, b.n, b.nb), facts=facts,
                conservation=res), frames


def _tree_view(b, ix):
    att = ix["sprays"][b.alive[ix["petioles"]]]
    off = ix["sprays"][~b.alive[ix["petioles"]]]
    wood = np.flatnonzero(np.isin(b.tag, (1, 2, 3)))
    return ascii_side([(b.x[att], "*"), (b.x[off], "o"), (b.x[wood], "#")],
                      [((-10, -10, -1), (10, 10, 0.05))], (-0.5, 4.5), (-0.1, 3.0),
                      cols=75, rows=22)


def _f(v):
    return None if v is None or not np.isfinite(v) else round(float(v), 4)


def _stats(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if not len(a):
        return None
    return dict(n=int(len(a)), mean=_f(a.mean()), min=_f(a.min()), max=_f(a.max()))


def c_tree_shake():
    """The promoted tree (one cluster) in src/mpm: base pinned, the same
    shaker on the trunk particles at 0.55-0.65 m."""
    import continuum as C
    import src.mpm.core as core
    w, m = C.tree_promoted()
    leaf = np.array([m.kinds[k] == "leaf" for k in m.kind])
    wood = ~leaf
    # the tree stands on the ground; the shaker's clamp is a pin (a promoted
    # tree has no roots: it is a free body on the floor)
    ring = wood & (m.x[:, 2] > 0.5) & (m.x[:, 2] < 0.65)
    pin = C.Pin(m, ring, **SHAKE)
    clock, sub, ticks, frames = Clock(), 0, 0, []
    top = wood & (m.x[:, 2] > m.x[wood, 2].max() - 0.06)
    loose, sway = [], []
    keep = core.GRAVITY.copy()
    while ticks < round(T_REST / TICK):
        core.GRAVITY[2] = -_g_ramp(ticks * TICK)
        with clock:
            sub += C.tick(m, after=pin)
        ticks += 1
        loose.append(int(((m.cid < 0) & leaf).sum()))
        sway.append(float(m.x[top, 0].mean() - 1.0))
        if ticks in (1, 50, 100):
            frames.append((round(m.t, 3), ascii_side(
                [(m.x[leaf], "*"), (m.x[wood], "#")], [((0, 0, 0), (2, 2, 0.05))],
                (-0.5, 4.5), (-0.1, 3.0), cols=75, rows=22)))
    core.GRAVITY[:] = keep
    facts = dict(particles=int(len(m.x)), leaf_particles=int(leaf.sum()),
                 leaf_loose_at_end=loose[-1], leaf_loose_max=max(loose),
                 cracked=int(m.dmg.sum()),
                 crown_sway_during_shake_m=dict(max=_f(np.max(np.abs(sway))),
                                                rms=_f(np.sqrt(np.mean(np.square(sway))))))
    return dict(model="continuum", scene="tree shake", stiff=m.stiff,
                cost=_cost(clock, ticks, sub, len(m.x)), facts=facts), frames


def c_tree_fell():
    """`tree_fell` as the voxel world runs it (lattice, `_law_torque`,
    promote, the particle solver with sleep and islands), in a world 4.5 m
    long so the tree has room to land (the scene's own world is 2 m across,
    smaller than the tree is tall)."""
    import continuum as C
    from src.voxel.sim import AIR, LEAF, WOOD
    nx, ny, nz, cx, cy = 90, 40, 70, 20, 20
    w = C.tree_world(nx, ny, nz, cx, cy)
    vm = 0.1 * w.scale                         # metres a voxel
    clock, ticks, frames = Clock(), 0, []
    rows = []
    leaf_kind = None
    leaf0 = int((w.mat == LEAF).sum())
    for t in range(180):
        if t in (15, 30, 45, 60):                  # tree_fell's four axe strokes
            k = (t // 15) - 1
            w.mat[cx + 1 - k, cy - 2:cy + 2, 6:9] = AIR
            w.smass[cx + 1 - k, cy - 2:cy + 2, 6:9] = 0.0
        with clock:
            w.step()
        ticks += 1
        m = w.matter.mpm
        lat = np.argwhere(w.mat == LEAF)
        if len(m.x):
            pleaf = np.array([m.kinds[k] == "leaf" for k in m.kind])
        else:
            pleaf = np.zeros(0, bool)
        px = m.x[pleaf] if len(m.x) else np.zeros((0, 3))
        allz = np.concatenate([(lat[:, 2] + 0.5) * vm, px[:, 2]]) if len(lat) + len(px) else np.zeros(1)
        rows.append(dict(t=round((t + 1) * TICK, 3), lattice_leaf=int(len(lat)),
                         particle_leaf=int(len(px)),
                         particle_leaf_loose=int(((m.cid < 0)[pleaf]).sum()) if len(m.x) else 0,
                         leaf_z_mean=_f(allz.mean()), leaf_z_max=_f(allz.max()),
                         clusters=int(len(np.unique(m.cid[pleaf]))) if len(px) else 0,
                         wood_cells=int((w.mat == WOOD).sum()),
                         particles=int(len(m.x)), asleep=int((m.nap >= 0).sum()) if len(m.x) else 0))
        if t in (0, 70, 100, 179):
            wl = np.argwhere(w.mat == WOOD)
            pw = m.x[~pleaf] if len(m.x) else np.zeros((0, 3))
            frames.append((round((t + 1) * TICK, 3), ascii_side(
                [((lat + 0.5) * vm, "*"), (px, "*"), ((wl + 0.5) * vm, "#"), (pw, "#")],
                [((0, 0, 0), (nx * vm, ny * vm, vm))], (-0.5, 4.5), (-0.1, 3.0),
                cols=75, rows=22)))
    lat = np.argwhere(w.mat == LEAF)
    comps = _components(lat)
    m = w.matter.mpm
    pleaf = np.array([m.kinds[k] == "leaf" for k in m.kind]) if len(m.x) else np.zeros(0, bool)
    groups = len(comps) + (len(np.unique(m.cid[pleaf & (m.cid >= 0)])) if len(m.x) else 0) \
        + (int(((m.cid < 0) & pleaf).sum()) if len(m.x) else 0)
    allx = np.concatenate([(lat + 0.5) * vm, m.x[pleaf] if len(m.x) else np.zeros((0, 3))])
    facts = dict(leaf_cells_at_start=leaf0, leaf_lattice_groups_at_end=len(comps),
                 leaf_lattice_group_sizes=sorted((len(c) for c in comps), reverse=True)[:8],
                 leaf_particles_at_end=int(pleaf.sum()),
                 leaf_particles_loose_at_end=int(((m.cid < 0) & pleaf).sum()) if len(m.x) else 0,
                 leaf_groups_at_end=int(groups),
                 leaf_spread_m=dict(std_x=_f(allx[:, 0].std()), std_y=_f(allx[:, 1].std())),
                 timeline=rows[::10] + [rows[-1]])
    return dict(model="continuum", scene="tree fell (voxel world)",
                cost=_cost(clock, ticks, 0, rows[-1]["particles"]), facts=facts), frames


def _components(cells):
    """26-connected groups of cells, as lists of indices."""
    if not len(cells):
        return []
    key = {tuple(c): i for i, c in enumerate(cells)}
    seen, out = np.zeros(len(cells), bool), []
    for s0 in range(len(cells)):
        if seen[s0]:
            continue
        stack, comp = [s0], []
        seen[s0] = True
        while stack:
            i = stack.pop()
            comp.append(i)
            c = cells[i]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        j = key.get((c[0] + dx, c[1] + dy, c[2] + dz))
                        if j is not None and not seen[j]:
                            seen[j] = True
                            stack.append(j)
        out.append(comp)
    return out


# ══ the rope ═════════════════════════════════════════════════════════════════

def _period(ts, xs, x0):
    xs = np.asarray(xs) - x0
    ts = np.asarray(ts)
    cr = [ts[k] - xs[k] * (ts[k + 1] - ts[k]) / (xs[k + 1] - xs[k])
          for k in range(len(xs) - 1) if xs[k] > 0 >= xs[k + 1]]
    return (float(np.diff(cr).mean()) if len(cr) > 1 else None), cr


def b_rope_swing():
    return _b_rope_swing(1.0)


def b_rope_swing_soft():
    """At the continuum world's softening, so the two are compared alike."""
    return _b_rope_swing(1e-4)


def b_rope_swing_stiff():
    return _b_rope_swing(1e-2)


def _b_rope_swing(stiff):
    b, ix = S.rope(angle=np.radians(20))
    b.finish(stiff=stiff, strength_scale=1.0 if stiff == 1.0 else None)
    _warm(b)
    clock, sub, ticks, ts, xs, ds, E = Clock(), 0, 0, [], [], [], []
    top = np.array((0.0, 0.0, 2.0))
    while b.t < 6.0 - 1e-9:
        with clock:
            sub += b.step(TICK)
        ticks += 1
        ts.append(b.t); xs.append(b.x[ix["bob"], 0])
        ds.append(float(np.linalg.norm(b.x[ix["bob"]] - top)))
        free = ~b.kin
        E.append(b.kinetic() + float((b.m[free] * B.GRAVITY * b.x[free, 2]).sum()))
    T, cr = _period(ts, xs, 0.0)
    frames = [(round(b.t, 3), ascii_side([(b.x[ix["links"]], "|"), (b.x[[ix["bob"]]], "O")],
                                         [], (-0.6, 0.6), (0.7, 2.1), cols=40, rows=20))]
    facts = dict(period_s=_f(T), period_theory_s=_f(S.rope_period()),
                 period_err=_f(T / S.rope_period() - 1) if T else None,
                 bob_distance_m=dict(min=_f(min(ds)), max=_f(max(ds))),
                 energy_J=dict(start=_f(E[0]), end=_f(E[-1]), min=_f(min(E)), max=_f(max(E))),
                 crossings=[_f(c) for c in cr], broken=int((~b.alive).sum()))
    return dict(model="bonds", scene="rope swing", stiff=b.stiff, h=b.h,
                cost=_cost(clock, ticks, sub, b.n, b.nb), facts=facts,
                conservation=dict(end=_resid(b))), frames


def _ramp(t, t0=0.5, F0=200.0, dbl=0.25):
    return 0.0 if t < t0 else F0 * 2 ** ((t - t0) / dbl)


def b_rope_break():
    return _b_rope_break(1.0)


def b_rope_break_stiff():
    """At c_rope_break_stiff's softening (1e-2), strength scaled by
    sqrt(stiff) as src/mpm scales it."""
    return _b_rope_break(1e-2)


def _b_rope_break(stiff):
    b, ix = S.rope(angle=0.0)
    b.finish(stiff=stiff, strength_scale=1.0 if stiff == 1.0 else None)
    _warm(b)
    clock, sub, ticks, frames = Clock(), 0, 0, []
    broke_t, F_at = None, None
    while b.t < 4.0 - 1e-9:
        for c in range(10):                       # the load moves every 2.5 ms
            F = _ramp(b.t)
            b.fext[ix["bob"], 2] = -F
            with clock:
                sub += b.step(TICK / 10)
            if broke_t is None and not b.alive.all():
                broke_t, F_at = b.t, F
        ticks += 1
        if broke_t is not None and b.t > broke_t + 0.3:
            break
    which = int(np.flatnonzero(~b.alive)[0]) if not b.alive.all() else None
    flaws = b.sig / b.sig.max()
    T_at = None if F_at is None else F_at + S.BOB * B.GRAVITY
    nb = len(ix["bonds"])
    mbs = B.ROPE[2] * np.pi * S.ROPE_R ** 2
    facts = dict(break_s=_f(broke_t), load_N_at_break=_f(F_at), tension_N_at_break=_f(T_at),
                 bond_broken=which, bonds_from_top=nb,
                 where=None if which is None else ("at the beam" if which == 0 else
                                                   "at the weight" if which == nb - 1 else
                                                   f"link joint {which} of {nb - 1} from the top"),
                 weakest_bond=int(np.argmin(b.sig)),
                 its_strength_N=_f(b.sig[which] * b.A[which]) if which is not None else None,
                 rope_breaking_load_nominal_N=_f(mbs),
                 rope_breaking_load_scaled_N=_f(mbs * b.strength_scale))
    frames.append((round(b.t, 3), ascii_side([(b.x[ix["links"]], "|"), (b.x[[ix["bob"]]], "O")],
                                             [], (-0.6, 0.6), (-0.1, 2.1), cols=40, rows=20)))
    return dict(model="bonds", scene="rope overload", stiff=b.stiff, h=b.h,
                cost=_cost(clock, ticks, sub, b.n, b.nb), facts=facts), frames


def _c_rope_swing(stiff):
    import continuum as C
    w, ix = C.rope(stiff=stiff, angle=np.radians(20))
    pin = C.Pin(w, ix["tied"])
    clock, sub, ticks, ts, xs, ds = Clock(), 0, 0, [], [], []
    bob = ix["bob"]
    while w.t < 6.0 - 1e-9:
        with clock:
            sub += C.tick(w, after=pin)
        ticks += 1
        c = (w.m[bob, None] * w.x[bob]).sum(0) / w.m[bob].sum()
        ts.append(w.t); xs.append(c[0]); ds.append(float(np.linalg.norm(c - ix["top"])))
    T, cr = _period(ts, xs, ix["top"][0])
    frames = [(round(w.t, 3), ascii_side([(w.x[ix["rope"]], "|"), (w.x[bob], "O")],
                                         [], (0.0, 1.2), (0.0, 2.1), cols=40, rows=20))]
    facts = dict(period_s=_f(T), period_theory_s=_f(S.rope_period()),
                 period_err=_f(T / S.rope_period() - 1) if T else None,
                 bob_distance_m=dict(min=_f(min(ds)), max=_f(max(ds)), rest_length=1.05),
                 cracked=int(w.dmg.sum()), crossings=[_f(c) for c in cr])
    return dict(model="continuum", scene="rope swing", stiff=stiff,
                cost=_cost(clock, ticks, sub, len(w.x)), facts=facts), frames


def c_rope_swing():
    return _c_rope_swing(1e-4)


def c_rope_swing_stiff():
    return _c_rope_swing(1e-2)


def _c_rope_break(stiff):
    import continuum as C
    w, ix = C.rope(stiff=stiff, angle=0.0)
    bob = ix["bob"]
    force = [0.0]
    push = C.load(w, bob, lambda: (0.0, 0.0, -force[0]))
    pin = C.Pin(w, ix["tied"])

    def after(h):
        pin(h)
        push(h)
    clock, sub, ticks = Clock(), 0, 0
    first_t = first_F = first_z = part_t = part_F = None
    z_hang = None
    while w.t < 4.0 - 1e-9:
        force[0] = _ramp(w.t)
        with clock:
            sub += C.tick(w, after=after)
        ticks += 1
        cz = (w.m[bob] * w.x[bob, 2]).sum() / w.m[bob].sum()
        vz = (w.m[bob] * w.v[bob, 2]).sum() / w.m[bob].sum()
        if ticks == 20:
            z_hang = cz
        d = w.dmg[ix["rope"]]
        if first_t is None and d.any():
            first_t, first_F = w.t, force[0]
            first_z = _f(w.x[ix["rope"]][d, 2].mean())
        if part_t is None and vz < -2.0:
            part_t, part_F = w.t, force[0]
        if part_t is not None and w.t > part_t + 0.3:
            break
    frames = [(round(w.t, 3), ascii_side([(w.x[ix["rope"]], "|"), (w.x[bob], "O")],
                                         [((0, 0, 0), (1.2, 0.6, 0.1))], (0.0, 1.2), (0.0, 2.1),
                                         cols=40, rows=20))]
    facts = dict(first_crack_s=_f(first_t), load_N_at_first_crack=_f(first_F),
                 first_crack_height_m=first_z, parted_s=_f(part_t), load_N_when_parted=_f(part_F),
                 bob_hang_z_m=_f(z_hang), bob_rest_z_m=0.95,
                 rope_cracked=int(w.dmg[ix["rope"]].sum()),
                 strength_scaled_Pa=_f(core_strength("rope") * np.sqrt(stiff)))
    return dict(model="continuum", scene="rope overload", stiff=stiff,
                cost=_cost(clock, ticks, sub, len(w.x)), facts=facts), frames


def core_strength(name):
    import src.mpm.core as core
    return core.MAT[name][5]


def c_rope_break():
    return _c_rope_break(1e-4)


def c_rope_break_stiff():
    return _c_rope_break(1e-2)


# ══ the glass ════════════════════════════════════════════════════════════════

def _mpm_field():
    import continuum as C
    return C.ms.glass_edge()._flaw_field(0)


def _b_glass(stiff, scale=None, ticks_=40):
    b, ix = S.glass(field=_mpm_field())
    b.finish(stiff=stiff, strength_scale=scale)
    _warm(b)
    b.v[:, 2] = -6.3
    b.rebase()
    clock, sub, ticks, frames = Clock(), 0, 0, []
    while ticks < ticks_:
        with clock:
            sub += b.step(TICK)
        ticks += 1
        if ticks in (2, 6, ticks_):
            frames.append((round(b.t, 3), _glass_view(b.x)))
    idx = ix["idx"]
    gone = ~b.alive
    nx, ny, nz = idx.shape
    # a strip (one piece wide, along y) is SEVERED where every bond across
    # some x-boundary in it, through the thickness, is broken
    pair = {(int(b.bi[k]), int(b.bj[k])): k for k in range(b.nb)}
    sev, where = 0, []
    for j in range(ny):
        cut = [k for k in range(nx - 1)
               if all(gone[pair[(int(idx[k, j, z]), int(idx[k + 1, j, z]))]] for z in range(nz))]
        if cut:
            sev += 1
            where.append([round(float((k + 1) * ix["s"] + ix["lo"][0]), 3) for k in cut])
    lab = b.fragments()
    sizes = np.sort(np.bincount(np.unique(lab, return_inverse=True)[1]))[::-1]
    mid = 0.5 * (b.x0[b.bi] + b.x0[b.bj])
    near = np.abs(mid[:, 0] - 1.5) <= 0.2
    tb = b.tbreak[gone]
    facts = dict(pieces=int(b.n), bonds=int(b.nb), bonds_broken=int(gone.sum()),
                 broken_near_edge_frac=_f(gone[near].sum() / max(gone.sum(), 1)),
                 fragments=int(len(sizes)), fragments_2plus=int((sizes >= 2).sum()),
                 single_shards=int((sizes == 1).sum()), largest=[int(s) for s in sizes[:4]],
                 strips_severed=f"{sev} of {ny}", sever_x=where,
                 first_break_s=_f(tb.min()) if len(tb) else None,
                 on_table=int((b.x[:, 2] > 1.0).sum()), on_floor=int((b.x[:, 2] < 0.5).sum()))
    return dict(model="bonds", scene="glass pane", stiff=stiff, h=b.h,
                cost=_cost(clock, ticks, sub, b.n, b.nb), facts=facts,
                conservation=dict(end=_resid(b))), frames


def _glass_view(x):
    return ascii_side([(x, "o")], [((0.0, 0.0, -1.0), (3.2, 1.4, 0.2)),
                                   ((0.3, 0.3, 0.2), (1.5, 1.1, 1.2))],
                      (0.2, 3.0), (0.0, 1.8), cols=70, rows=22)


def b_glass():
    return _b_glass(1e-3)


def b_glass_real():
    return _b_glass(1.0, ticks_=16)


def _c_glass(bound):
    import continuum as C
    w = C.glass(bound=bound)
    lo = np.array((0.8, 0.5, 3.2))
    x_init = w.x.copy()
    x_init[:, 2] += 3.2 - 1.25
    gi = np.floor((x_init - lo) / 0.05).astype(int)
    clock, sub, ticks, frames = Clock(), 0, 0, []
    first = None
    while ticks < 40:
        with clock:
            sub += C.tick(w)
        ticks += 1
        if first is None and w.dmg.any():
            first = w.t
        if ticks in (2, 6, 40):
            frames.append((round(w.t, 3), _glass_view(w.x)))
    d = w.dmg
    grid = -np.ones((20, 8, 3), int)
    grid[gi[:, 0], gi[:, 1], gi[:, 2]] = np.arange(len(w.x))
    sev, where = 0, []
    for j in range(8):
        cut = [k for k in range(20) if all(d[grid[k, j, z]] for z in range(3))]
        if cut:
            sev += 1
            where.append([round(0.8 + (k + 0.5) * 0.05, 3) for k in cut])
    # fragments: undamaged particles that were neighbours at rest and are
    # still within 1.5 spacings; a cracked particle is a shard of its own
    par = np.arange(len(w.x))

    def root(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a
    for a in range(20):
        for bb in range(8):
            for c in range(3):
                for dd in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    a2, b2, c2 = a + dd[0], bb + dd[1], c + dd[2]
                    if a2 >= 20 or b2 >= 8 or c2 >= 3:
                        continue
                    i, j = grid[a, bb, c], grid[a2, b2, c2]
                    if d[i] or d[j] or np.linalg.norm(w.x[i] - w.x[j]) > 0.075:
                        continue
                    ri, rj = root(i), root(j)
                    if ri != rj:
                        par[max(ri, rj)] = min(ri, rj)
    lab = np.array([root(a) for a in range(len(w.x))])
    sizes = np.sort(np.bincount(np.unique(lab, return_inverse=True)[1]))[::-1]
    near = np.abs(x_init[:, 0] - 1.5) <= 0.2
    facts = dict(particles=int(len(w.x)), cracked=int(d.sum()),
                 cracked_near_edge_frac=_f(d[near].sum() / max(d.sum(), 1)),
                 fragments=int(len(sizes)), fragments_2plus=int((sizes >= 2).sum()),
                 single_shards=int((sizes == 1).sum()), largest=[int(s) for s in sizes[:4]],
                 strips_severed=f"{sev} of 8", sever_x=where,
                 first_crack_s=_f(first),
                 on_table=int((w.x[:, 2] > 1.0).sum()), on_floor=int((w.x[:, 2] < 0.5).sum()))
    return dict(model="continuum", scene="glass pane" + (" (bound)" if bound else ""),
                stiff=w.stiff, cost=_cost(clock, ticks, sub, len(w.x)), facts=facts), frames


def c_glass():
    return _c_glass(False)


def c_glass_bound():
    return _c_glass(True)


# ══ the plank ════════════════════════════════════════════════════════════════

def _tip_track(xs, ms, t, out):
    """Track a tipping plank: when its centre drops below 1.5 m, when it
    first meets the floor."""
    com = (ms[:, None] * xs).sum(0) / ms.sum()
    if out["left_table_s"] is None and com[2] < 1.5:
        out["left_table_s"] = round(t, 3)
    if out["floor_s"] is None and xs[:, 2].min() < 0.26:
        out["floor_s"] = round(t, 3)
    return com


def _tip_view(x):
    return ascii_side([(x, "#")], [((0.0, 0.0, -1.0), (3.2, 1.4, 0.2)),
                                   ((0.3, 0.3, 0.2), (1.3, 1.1, 1.6))],
                      (0.2, 3.0), (0.0, 2.0), cols=70, rows=24)


def _b_tip(stiff, wood=(B.WOOD_L, B.WOOD_T)):
    b, ix = S.plank_tip(wood=wood)
    b.finish(stiff=stiff)
    _warm(b)
    clock, sub, ticks, frames = Clock(), 0, 0, []
    out = dict(left_table_s=None, floor_s=None)
    wmax = 0.0
    while ticks < 80:
        with clock:
            sub += b.step(TICK)
        ticks += 1
        _tip_track(b.x, b.m, b.t, out)
        P, L = b.momentum()
        if ticks in (1, 24, 40, 80):
            frames.append((round(b.t, 3), _tip_view(b.x)))
    com = (b.m[:, None] * b.x).sum(0) / b.M
    facts = dict(out, final_angle_deg=_f(long_axis(b.x)), final_com=[_f(v) for v in com],
                 bonds_broken=int((~b.alive).sum()),
                 still=_f(np.abs(b.v).max()))
    return dict(model="bonds", scene="plank tip", stiff=stiff, h=b.h,
                wood_strength_Pa=[wood[0][2], wood[1][2]],
                cost=_cost(clock, ticks, sub, b.n, b.nb), facts=facts,
                conservation=dict(end=_resid(b))), frames


def b_plank_tip():
    return _b_tip(1e-4)


def b_plank_tip_hb():
    return _b_tip(1e-4, (B.WOOD_L_HB, B.WOOD_T_HB))


def b_plank_tip_hb_stiff():
    return _b_tip(1e-2, (B.WOOD_L_HB, B.WOOD_T_HB))


def _c_tip(bound):
    import continuum as C
    w = C.plank_tip(bound=bound)
    clock, sub, ticks, frames = Clock(), 0, 0, []
    out = dict(left_table_s=None, floor_s=None)
    while ticks < 80:
        with clock:
            sub += C.tick(w)
        ticks += 1
        _tip_track(w.x, w.m, w.t, out)
        if ticks in (1, 24, 40, 80):
            frames.append((round(w.t, 3), _tip_view(w.x)))
    com = (w.m[:, None] * w.x).sum(0) / w.m.sum()
    facts = dict(out, final_angle_deg=_f(long_axis(w.x)), final_com=[_f(v) for v in com],
                 cracked=int(w.dmg.sum()), still=_f(np.abs(w.v).max()))
    return dict(model="continuum", scene="plank tip" + (" (bound)" if bound else ""),
                stiff=w.stiff, cost=_cost(clock, ticks, sub, len(w.x)), facts=facts), frames


def c_plank_tip():
    return _c_tip(False)


def c_plank_tip_bound():
    return _c_tip(True)


def _bend_ramp(t, t0=0.25, F0=100.0, dbl=0.1):
    return 0.0 if t < t0 else F0 * 2 ** ((t - t0) / dbl)


def _bend_view(x):
    return ascii_side([(x, "#")], [((0.0, 0.0, -1.0), (3.0, 0.6, 0.0)),
                                   ((0.5, 0.0, 0.0), (0.7, 0.6, 1.0)),
                                   ((2.3, 0.0, 0.0), (2.5, 0.6, 1.0))],
                      (0.3, 2.7), (0.0, 1.3), cols=72, rows=20)


def b_plank_bend():
    return _b_bend(1.0)


def b_plank_bend_soft():
    """At c_plank_bend's softening (1e-4), strength scaled by sqrt(stiff)."""
    return _b_bend(1e-4)


def b_plank_bend_stiff():
    """At c_plank_bend_stiff's softening (1e-2)."""
    return _b_bend(1e-2)


def _b_bend(stiff):
    b, ix = S.plank_bend()
    b.finish(stiff=stiff, strength_scale=1.0 if stiff == 1.0 else None)
    _warm(b)
    load = ix["load"]
    idx = ix["idx"]
    mid = idx[idx.shape[0] // 2 - 1:idx.shape[0] // 2 + 1, :, 0].ravel()
    ends = np.concatenate([idx[1:3, :, 0].ravel(), idx[-3:-1, :, 0].ravel()])
    clock, sub, ticks, frames = Clock(), 0, 0, []
    first_t = first_F = fail_t = fail_F = None
    sag0 = None
    curve = []
    while b.t < 2.0 - 1e-9:
        for c in range(10):
            F = _bend_ramp(b.t)
            b.fext[load, 2] = -F / len(load)
            with clock:
                sub += b.step(TICK / 10)
            if first_t is None and not b.alive.all():
                first_t, first_F = b.t, F
        ticks += 1
        sag = b.x[ends, 2].mean() - b.x[mid, 2].mean()
        if ticks == 10:
            sag0 = sag
        curve.append((round(b.t, 3), round(F, 1), sag))
        if fail_t is None and sag0 is not None and sag - sag0 > 0.05:
            fail_t, fail_F, fail_k = b.t, F, ticks
        if ticks == 10 or (fail_t is not None and ticks == fail_k + 4):
            frames.append((round(b.t, 3), _bend_view(b.x)))
        if fail_t is not None and ticks >= fail_k + 4:
            break
    Fth, per_N = S.bend_theory()
    k1 = next((k for k, c in enumerate(curve) if c[1] >= 1000.0), None)
    sag1k = None if k1 is None or sag0 is None else curve[k1][2] - sag0
    first = np.flatnonzero(~b.alive)
    fb = None
    if len(first):
        k = first[np.argmin(b.tbreak[first])]
        m_ = 0.5 * (b.x0[b.bi[k]] + b.x0[b.bj[k]])
        fb = dict(x=_f(m_[0]), z=_f(m_[2]), along_grain=bool(abs(b.x0[b.bj[k], 0] - b.x0[b.bi[k], 0]) > 1e-6),
                  flaw=_f(b.sig[k] / (B.WOOD_L[2] * b.strength_scale)))
    facts = dict(self_weight_sag_mm=_f(1e3 * sag0) if sag0 is not None else None,
                 self_weight_sag_mm_theory=_f(1e3 * 5 * 600 * 0.02 * 9.81 * S.BEND_L ** 4 /
                                              (384 * 1.1e10 * stiff * S.BEND_B * S.BEND_H ** 3 / 12)),
                 sag_mm_at_1kN=_f(1e3 * sag1k) if sag1k is not None else None,
                 sag_mm_at_1kN_theory=_f(1e3 * 1000.0 * per_N / stiff),
                 load_N_first_break=_f(first_F), first_break=fb,
                 load_N_failed=_f(fail_F), load_N_theory_nominal=_f(Fth),
                 load_N_theory_scaled=_f(Fth * b.strength_scale),
                 bonds_broken=int((~b.alive).sum()),
                 pieces_at_end=int(len(np.unique(b.fragments()))))
    frames.append((round(b.t, 3), _bend_view(b.x)))
    return dict(model="bonds", scene="plank bend", stiff=b.stiff, h=b.h,
                cost=_cost(clock, ticks, sub, b.n, b.nb), facts=facts,
                conservation=dict(end=_resid(b))), frames


def _c_bend(stiff, bound):
    import continuum as C
    w, ix = C.plank_bend(stiff=stiff, bound=bound)
    load = ix["load"]
    force = [0.0]
    push = C.load(w, load, lambda: (0.0, 0.0, -force[0]))
    x0 = w.x.copy()
    mid = np.flatnonzero((np.abs(x0[:, 0] - 1.5) < 0.05) & (x0[:, 2] < 1.05))
    ends = np.flatnonzero((np.abs(x0[:, 0] - 0.6) < 0.05) | (np.abs(x0[:, 0] - 2.4) < 0.05))
    clock, sub, ticks, frames = Clock(), 0, 0, []
    first_t = first_F = fail_t = fail_F = first_x = None
    sag0 = None
    while w.t < 2.0 - 1e-9:
        force[0] = _bend_ramp(w.t)
        with clock:
            sub += C.tick(w, after=push)
        ticks += 1
        sag = w.x[ends, 2].mean() - w.x[mid, 2].mean()
        if ticks == 10:
            sag0 = sag
        if first_t is None and w.dmg.any():
            first_t, first_F = w.t, force[0]
            first_x = _f(x0[w.dmg, 0].mean())
        if fail_t is None and sag0 is not None and sag - sag0 > 0.05:
            fail_t, fail_F, fail_k = w.t, force[0], ticks
        if ticks == 10 or (fail_t is not None and ticks == fail_k + 4):
            frames.append((round(w.t, 3), _bend_view(w.x)))
        if fail_t is not None and ticks >= fail_k + 4:
            break
    frames.append((round(w.t, 3), _bend_view(w.x)))
    Fth, per_N = S.bend_theory()
    facts = dict(self_weight_sag_mm=_f(1e3 * sag0) if sag0 is not None else None,
                 self_weight_sag_mm_theory=_f(1e3 * 5 * 600 * 0.02 * 9.81 * S.BEND_L ** 4 /
                                              (384 * 1.1e10 * stiff * S.BEND_B * S.BEND_H ** 3 / 12)),
                 load_N_first_crack=_f(first_F), first_crack_x=first_x,
                 load_N_failed=_f(fail_F), load_N_theory_nominal=_f(Fth),
                 load_N_theory_scaled=_f(Fth * np.sqrt(stiff)),
                 cracked=int(w.dmg.sum()), clusters_parted=int((w.cid < 0).sum()) if bound else None)
    return dict(model="continuum", scene="plank bend" + (" (bound)" if bound else ""),
                stiff=stiff, cost=_cost(clock, ticks, sub, len(w.x)), facts=facts), frames


def c_plank_bend():
    return _c_bend(1e-4, False)


def c_plank_bend_bound():
    return _c_bend(1e-4, True)


def c_plank_bend_stiff():
    return _c_bend(1e-2, False)


# ══ how much a block bounces ═════════════════════════════════════════════════

def b_bounce():
    """A 20 x 20 x 15 cm block of glass and of wood dropped 1 m onto a
    floor, for three contact dampings: the restitution that comes out
    (handbook, docs/realism-references.md M9: wood block 0.40-0.55, glass
    marble 0.66; src/mpm's wood block 0.26)."""
    out = {}
    for mat, stiff in (("glass", 1e-3), ("wood", 1e-4)):
        for zc in (0.2, 0.5, 0.8):
            b = B.Bonded(zeta_c=zc)
            b.wall((-2, -2, -1), (2, 2, 0.0))
            m_ = B.GLASS if mat == "glass" else (B.WOOD_L_HB, B.WOOD_T_HB)
            B.lattice(b, (0, 0, 1.0), (0.2, 0.2, 1.15), 0.05, m_, grain=(1, 0, 0))
            b.finish(stiff=stiff)
            zs = []
            for _ in range(160):
                b.step(TICK / 4)
                zs.append(b.x[:, 2].mean())
            zs = np.array(zs)
            land = int(np.argmin(zs[:100]))
            v_in = np.sqrt(2 * 9.81 * (1.075 - zs[land]))
            up = zs[land:].max() - zs[land]
            out[f"{mat} zeta_c={zc}"] = dict(e=_f(np.sqrt(2 * 9.81 * up) / v_in),
                                            broken=int((~b.alive).sum()))
    return dict(model="bonds", scene="bounce", facts=out), []


# ══ conservation in free space ═══════════════════════════════════════════════
# Each object with nothing outside it: no gravity, no walls, no pins, no air.
# Spun (0.05, 0.3, 0.1) rad/s about its centre plus 3 cm/s of noise per piece,
# so its springs ring (gently: the continuum's worlds are only a little
# bigger than its objects, and its edge clamp is an outside push).
# Momentum and angular momentum must stay what they were.

def _kick(x, m, seed=0):
    c = (m[:, None] * x).sum(0) / m.sum()
    rng = np.random.default_rng(seed)
    return np.cross((0.05, 0.3, 0.1), x - c) + rng.normal(0, 0.03, x.shape)


def b_free():
    out = {}
    for name, make, stiff, ss in (("tree", S.tree, 1.0, 1.0), ("rope", S.rope, 1.0, 1.0),
                                  ("glass", S.glass, 1e-3, None),
                                  ("plank", S.plank_tip, 1e-4, None)):
        b, ix = make()
        b.finish(stiff=stiff, strength_scale=ss)
        b.kin[:] = False
        b.g = 0.0
        b.cda[:] = 0.0                   # no air: it would push from outside
        b.wlo, b.whi = b.wlo[:0], b.whi[:0]
        b.v[:] = _kick(b.x, b.m)
        b.M = float(b.m.sum())
        b.rebase()
        P0, L0 = b.momentum()
        clock, sub = Clock(), 0
        _warm(b)
        for _ in range(20):
            with clock:
                sub += b.step(TICK)
        P, L = b.momentum()
        out[name] = dict(dP=float(np.linalg.norm(P - P0)), P=float(np.linalg.norm(P0)),
                         M=float(b.M), dL=float(np.linalg.norm(L - L0)),
                         L=float(np.linalg.norm(L0)), broken=int((~b.alive).sum()),
                         ms_per_tick=round(1e3 * clock.s / 20, 3))
    return dict(model="bonds", scene="free space", facts=out), []


def c_free():
    import continuum as C
    import src.mpm.core as core
    out = {}
    keep = core.GRAVITY.copy()
    core.GRAVITY[:] = 0.0
    try:
        def lifted_tree():
            m = C.tree_promoted()[1]
            m.x[:, 2] += 0.5          # off the world's floor edge, which clamps
            return m
        makers = (("tree", lifted_tree),
                  ("rope", lambda: C.rope()[0]),
                  ("glass", lambda: C.glass()),
                  ("glass (bound)", lambda: C.glass(bound=True)),
                  ("plank (bound)", lambda: C.plank_tip(bound=True)))
        for name, make in makers:
            w = make()
            w.wall[:] = False
            w._near = None
            w.v[:] = _kick(w.x, w.m)
            w.C[:] = 0.0
            P0, L0, S0 = mpm_momentum(w)
            clock, sub = Clock(), 0
            for _ in range(20):
                with clock:
                    sub += C.tick(w)
            P, L, Sp = mpm_momentum(w)
            out[name] = dict(dP=float(np.linalg.norm(P - P0)), P=float(np.linalg.norm(P0)),
                             M=float(w.m.sum()), dL=float(np.linalg.norm(L - L0)),
                             dL_with_spin=float(np.linalg.norm(Sp - S0)),
                             L=float(np.linalg.norm(L0)), cracked=int(w.dmg.sum()),
                             ms_per_tick=round(1e3 * clock.s / 20, 3))
            del w
    finally:
        core.GRAVITY[:] = keep
    return dict(model="continuum", scene="free space", facts=out), []


def main():
    name = sys.argv[1]
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    res, frames = globals()[name]()
    res["experiment"] = name
    res["total_wall_s"] = round(time.time() - t0, 2)
    with open(os.path.join(OUT, name + ".json"), "w") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    with open(os.path.join(OUT, name + ".txt"), "w") as f:
        for t, pic in frames:
            f.write(f"--- {name}  t = {t} s\n{pic}\n")
    print(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))[:6000])


if __name__ == "__main__":
    main()
