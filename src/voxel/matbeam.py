"""MATBEAM: load-aware support. Beams, columns, masonry sections, arches, and the
load path that feeds them (spec T9-material v2, sections 3.5, 4.4-4.6).

STAND-ALONE. Pure numpy on arrays; no World. It imports no other new module
except inside try/except (matrec for the record, matdamage for `combine`), and
every function that needs material numbers takes them as a plain dict of
arrays (`rec`) built by `rec_from_rows(rows)`.

SIGN AND UNITS: SI, newtons, metres, degrees C. h is the cell side (0.05 m at
5 cm). A load is a pair (sustained, transient): dead weight and carried things
are sustained; the walking extra and a landing are transient (the load that
lasts a fraction of a second gets more capacity, see `analyse(gain=...)`).

WHAT IT DOES
  1. THE MEMBER TEST (data, never a flag). A cell is a member only if its
     material is structural (E, compression and flexural strength above the
     numbers MEMBER_E and MEMBER_S) and it is not a person's claim, a grip
     carried cell, a moving (particle) cell or rubble. Leaf, ash, rope and
     flesh fail by their own numbers; they still have WEIGHT, which passes down.
  2. THE LOAD PATH, top to bottom, layer by layer. A cell with solid below it
     passes its whole load down. A line of member cells with air below is a
     HANGING RUN: pinned on two supports (effective span n h + h/2, reactions
     at the middle of the bearing) or a cantilever on one (critical section at
     the support face). Its end reactions load the support cells. Two runs
     stacked on the same two supports share by EI; layers of one REGISTERED
     thing are one deeper section; a joint-less masonry wall over an opening
     arches (only the 45 degree triangle loads the run).
  3. THE CHECKS. Bending (brittle: elastic S sf; ductile with a yield:
     plastic 1.5 S sy), shear 1.5 V/(A tau), column crushing and buckling
     (Euler with Johnson's parabola; end condition K and bracing read from
     the lattice), and the no-tension section of masonry.
  4. A FAILURE IS A PIECE, not a number: the caller sets `rubble` on the
     critical cell (and bumps `_rub_gen`); the existing support law drops it
     if air is below. Nothing here changes SPAN or touches `_relax_slack`.

Numbers marked (u) are unverified.
"""
import math
from dataclasses import dataclass, field, replace

import numpy as np

try:
    from . import matrec as _matrec
except ImportError:
    _matrec = None


def _combine_copy(D, dD):
    """Independent causes: 1 - (1 - D)(1 - dD), clipped to [0, 0.999]. A local
    copy of matdamage.combine, kept equal to it by a cross test; used when
    matdamage cannot be imported."""
    r = np.clip(1.0 - (1.0 - np.asarray(D, float)) * (1.0 - np.asarray(dD, float)), 0.0, 0.999)
    return float(r) if r.ndim == 0 else r


try:
    from .matdamage import combine
except ImportError:
    combine = _combine_copy

# ── constants (data) ────────────────────────────────────────────────────────
G = 9.81
MEMBER_E = 1.0e8            # Pa: below this a stuff is not a structural member
MEMBER_S = 1.0e6            # Pa: compression and flexural strength floor
BRITTLE = 0.05              # ductility below this is brittle
MASONRY_RATIO = 0.1         # st/sc below this and no joint: dry masonry (no tension across faces)
DAF_WALK = 1.2              # vertical ground force of walking, in body weights (u)
DAF_RUN = 2.5               # running (u)
D_BODY_PERSON = 0.15        # m: static compliance of a landing person's legs (u)
FOOT_M = 0.1                # m: width of a foot, the footprint of a concentrated load
GAIN_CAP = 1.3              # cap on the short-load strength gain (u)
K_HELD = 1.0                # Euler end factor, top held (pinned-pinned)
K_FREE = 2.0                # top free (fixed-free)
MAX_STACK = 16              # most layers of one registered thing read as one deeper section

M_NONE, M_BEND, M_SHEAR, M_CRUSH, M_BUCKLE, M_NOTEN = 0, 1, 2, 3, 4, 5
MODE_NAME = ("none", "bend", "shear", "crush", "buckle", "no-tension")

_T0, _DT, _NT = -100.0, 5.0, 321      # derate grid: -100 .. 1500 C every 5 K


# ── the record, as plain arrays ─────────────────────────────────────────────
def _get(row, key, default=None):
    v = row.get(key, default) if isinstance(row, dict) else getattr(row, key, default)
    return default if v is None else v


def rec_from_rows(rows):
    """rows: a list indexed by material id of dicts (or objects) with rho, E,
    nu, sy, st, sc, sf, tau, ductility and derate ((T_C, ks, kE), ...). None
    entries (gaps) are skipped. sy and sf may be None (nan here). Returns the
    dict of arrays every function below takes as `rec`."""
    n = len(rows)
    rec = {k: np.zeros(n) for k in ("rho", "E", "nu", "st", "sc", "tau", "ductility")}
    rec["sy"] = np.full(n, np.nan)
    rec["sf"] = np.full(n, np.nan)
    Tg = _T0 + _DT * np.arange(_NT)
    ks = np.ones((n, _NT))
    kE = np.ones((n, _NT))
    for i, r in enumerate(rows):
        if r is None:
            continue
        for k in ("rho", "E", "nu", "st", "sc", "tau", "ductility"):
            rec[k][i] = float(_get(r, k, 0.0))
        for k in ("sy", "sf"):
            v = _get(r, k, None)
            rec[k][i] = np.nan if v is None else float(v)
        d = _get(r, "derate", None)
        if d:
            d = sorted(d)
            ts = [float(a[0]) for a in d]
            ks[i] = np.interp(Tg, ts, [float(a[1]) for a in d])
            kE[i] = np.interp(Tg, ts, [float(a[2]) for a in d])
    rec["sf_eff"] = np.where(np.isnan(rec["sf"]), rec["st"], rec["sf"])
    rec["ks"], rec["kE"] = ks, kE
    rec["T0"], rec["dT"], rec["nT"] = _T0, _DT, _NT
    rec["structural"] = ((rec["E"] >= MEMBER_E) & (rec["sc"] >= MEMBER_S)
                         & (rec["sf_eff"] >= MEMBER_S))
    return rec


def rec_from_matrec():
    """The same dict from matrec.BY_ID, or RuntimeError when matrec is not there."""
    if _matrec is None:
        raise RuntimeError("matrec is not available: pass a rec from rec_from_rows(rows)")
    return rec_from_rows(list(_matrec.BY_ID))


def structural_mask(rec):
    """Per material id: passes the member test (data, 3.5)."""
    return rec["structural"]


def derate_at(rec, mat, T):
    """(ks, kE) at material ids `mat` and temperatures T (C), linear on the grid."""
    mat = np.asarray(mat)
    x = np.clip((np.asarray(T, float) - rec["T0"]) / rec["dT"], 0.0, rec["nT"] - 1.000001)
    i = x.astype(np.int64)
    f = x - i
    ks = rec["ks"][mat, i] * (1 - f) + rec["ks"][mat, i + 1] * f
    kE = rec["kE"][mat, i] * (1 - f) + rec["kE"][mat, i + 1] * f
    return ks, kE


def effective(rec, mat, T, D, temper=1.0):
    """Effective values of cells: strength x ks(T) x (1-D)^p, p = 1 for tension,
    bending and shear and 1/2 for compression; E x kE(T) x (1-D). sy keeps
    nan where the stuff has none."""
    ks, kE = derate_at(rec, mat, T)
    t = 1.0 - np.clip(np.asarray(D, float), 0.0, 0.999)
    mat = np.asarray(mat)
    return {"E": rec["E"][mat] * kE * t,
            "st": rec["st"][mat] * ks * t,
            "sc": rec["sc"][mat] * ks * np.sqrt(t),
            "sf": rec["sf_eff"][mat] * ks * t,
            "tau": rec["tau"][mat] * ks * t,
            "sy": rec["sy"][mat] * ks * t * temper,
            "duct": rec["ductility"][mat], "ks": ks}


# ── sections ────────────────────────────────────────────────────────────────
def stack_section(fills, h, plate):
    """Section of a lane of cells stacked on top of each other (one registered
    thing). fills: (n, K) fill fractions, 0 pads. plate: (n,) bool. Each cell is
    a BAR of side s = h sqrt(f) or a PLATE of width h and thickness f h; the
    stack's I is the parallel-axis sum about the area centroid (cell centres a
    cell side apart). Returns dict A, S, I, d (depth), b (width), r (gyration)."""
    f = np.clip(np.asarray(fills, float), 0.0, 1.0)
    if f.ndim == 1:
        f = f[None, :]
    n, K = f.shape
    pl = np.broadcast_to(np.asarray(plate, bool).reshape(-1, 1), (n, 1)) if np.ndim(plate) else \
        np.full((n, 1), bool(plate))
    A_j = f * h * h
    d_j = np.where(pl, f * h, h * np.sqrt(f))
    I_j = np.where(pl, h * d_j ** 3 / 12.0, d_j ** 4 / 12.0)
    y_j = np.arange(K) * h
    A = A_j.sum(1)
    ybar = (A_j * y_j).sum(1) / np.where(A > 0, A, 1.0)
    dy = y_j[None, :] - ybar[:, None]
    I = (I_j + A_j * dy ** 2).sum(1)
    present = f > 0
    c = np.where(present, np.abs(dy) + d_j / 2.0, 0.0).max(1)
    S = I / np.where(c > 0, c, 1.0)
    top = np.where(present, y_j[None, :] + d_j / 2.0, -np.inf).max(1)
    bot = np.where(present, y_j[None, :] - d_j / 2.0, np.inf).min(1)
    depth = np.where(A > 0, top - bot, 0.0)
    b = A / np.where(depth > 0, depth, 1.0)
    r = np.sqrt(I / np.where(A > 0, A, 1.0))
    return {"A": A, "S": S, "I": I, "d": depth, "b": b, "r": r}


def section(f, h, plate=False):
    """One cell of fill f: bar (single lane) or plate. A = f h2; bar S = h3 f^1.5/6,
    I = h4 f2/12; plate S = h3 f2/6."""
    f = np.atleast_1d(np.asarray(f, float))
    return stack_section(f[:, None], h, np.broadcast_to(np.asarray(plate, bool), f.shape))


# ── beam closed forms ───────────────────────────────────────────────────────
def pinned_reactions(W, x, L):
    """Left and right reaction of point loads W at positions x on a span L."""
    W = np.asarray(W, float)
    x = np.asarray(x, float)
    R0 = float((W * (L - x)).sum() / L)
    return R0, float(W.sum()) - R0


def pinned_moments(W, x, L):
    """Pinned-pinned beam, point loads W at x from the left reaction, span L
    (n h + h/2 for a run of n cells, x_i = (i - 1/4) h, i from 1). Returns
    (M, V) AT THE LOAD POINT of each cell (a cumulative sum, exact for point
    loads): M_j = R x_j - sum_{i<j} W_i (x_j - x_i); V is the larger of the
    shear just left and just right of the load. Between loads M is linear, so
    this is the cell's peak."""
    W = np.asarray(W, float)
    x = np.asarray(x, float)
    R0, _ = pinned_reactions(W, x, L)
    cum = np.cumsum(W) - W
    cumx = np.cumsum(W * x) - W * x
    M = R0 * x - (x * cum - cumx)
    Vl = R0 - cum
    V = np.maximum(np.abs(Vl), np.abs(Vl - W))
    return M, V


def pinned_demand(W, x, L, h):
    """Per cell: the largest moment over the cell's extent (centre and both
    faces, the diagram being linear between loads) and the largest shear."""
    M, V = pinned_moments(W, x, L)
    xk = np.concatenate(([0.0], x, [L]))
    Mk = np.concatenate(([0.0], M, [0.0]))
    ml = np.interp(x - h / 2.0, xk, Mk)
    mr = np.interp(x + h / 2.0, xk, Mk)
    return np.maximum(M, np.maximum(ml, mr)), V


def cantilever_moments(W, x, h=None):
    """Cantilever, root (support face) at 0, cell centres x_i = (i - 1/2) h.
    Returns (M, V) at the LEFT (root side) face of each cell, the first one
    being the root face: M_j = sum_{i>=j} W_i (x_i - (j-1) h), V_j = sum_{i>=j} W_i."""
    W = np.asarray(W, float)
    x = np.asarray(x, float)
    if h is None:
        h = float(x[1] - x[0]) if len(x) > 1 else 2.0 * float(x[0])
    xf = x - h / 2.0
    sW = np.cumsum(W[::-1])[::-1]
    sWx = np.cumsum((W * x)[::-1])[::-1]
    return sWx - xf * sW, sW


def bending_capacity(S, sf, sy, duct):
    """M_cap by data: brittle (ductility < 0.05) S sf; ductile with a yield
    1.5 S sy (plastic moment of a rectangle); ductile with no yield S sf."""
    plastic = (np.asarray(duct) >= BRITTLE) & np.isfinite(sy)
    return np.where(plastic, 1.5 * S * np.nan_to_num(sy), S * sf)


def _ratio(num, cap):
    """num/cap with cap <= 0 meaning no capacity: inf if there is a demand."""
    num = np.abs(np.asarray(num, float))
    cap = np.asarray(cap, float)
    ok = cap > 0
    out = np.where(num > 0, np.inf, 0.0)
    return np.where(ok, num / np.where(ok, cap, 1.0), out)


def section_util(M, V, N, sec, eff):
    """(u_bend, u_shear, u_axial): bending M/M_cap, shear 1.5 V/(A tau), axial
    N/(A sc). sec has A and S; eff has sf, sy, duct, tau, sc."""
    Mcap = bending_capacity(sec["S"], eff["sf"], eff["sy"], eff["duct"])
    ub = _ratio(M, Mcap)
    uv = _ratio(1.5 * np.abs(V), sec["A"] * eff["tau"])
    ua = _ratio(N, sec["A"] * eff["sc"])
    return ub, uv, ua


def deflection_pinned_uniform(w, L, EI):
    return 5.0 * w * L ** 4 / (384.0 * EI)


def deflection_pinned_point(P, L, EI):
    return P * L ** 3 / (48.0 * EI)


def deflection_cantilever_uniform(w, L, EI):
    return w * L ** 4 / (8.0 * EI)


def deflection_cantilever_tip(P, L, EI):
    return P * L ** 3 / (3.0 * EI)


def dead_load_capacity(M_cap, L, W_total, cantilever=False):
    """One-number screen: the largest extra point load a member takes at the
    worst place. Pinned midspan 4 M_cap/L - w L/2 (w L = W_total); cantilever
    tip M_cap/L - w L/2. Negative means it fails under its own load."""
    if cantilever:
        return M_cap / L - W_total / 2.0
    return 4.0 * M_cap / L - W_total / 2.0


def effective_width(b, a, L, ortho=1.0):
    """Width over which a concentrated load of footprint a spreads on a plate
    of width b and span L: min(b, a + L/3) (rule of thumb (u)). ortho in (0, 1]
    scales the L/3 term for a stuff much weaker across than along (wood: use
    (E_perp/E)**0.25, the orthotropic plate length scale)."""
    return min(float(b), float(a) + ortho * float(L) / 3.0)


def spread_lanes(p, h, L, a=FOOT_M, ortho=1.0):
    """Spread the loads p (one per lane across a plate, lane width h) at one
    position along the span over the effective width, each over a window of
    that width centred on its lane and shifted to stay on the plate. The
    total is kept exactly."""
    p = np.asarray(p, float)
    nl = len(p)
    m = float(np.clip(effective_width(nl * h, a, L, ortho) / h, 1.0, nl))
    out = np.zeros(nl)
    j = np.arange(nl)
    for k in np.nonzero(p)[0]:
        lo, hi = k + 0.5 - m / 2.0, k + 0.5 + m / 2.0
        if lo < 0:
            lo, hi = 0.0, hi - lo
        if hi > nl:
            lo, hi = max(lo - (hi - nl), 0.0), float(nl)
        ov = np.clip(np.minimum(hi, j + 1) - np.maximum(lo, j), 0.0, None)
        out += p[k] * ov / ov.sum()
    return out


# ── columns ─────────────────────────────────────────────────────────────────
def johnson_euler_stress(lam, E, sc):
    """Critical stress at slenderness lam = K H / r: Euler pi2 E/lam2 above
    lam_c = pi sqrt(2 E / sc), Johnson's parabola sc (1 - sc lam2/(4 pi2 E))
    below (continuous at lam_c, where both give sc/2)."""
    lam = np.asarray(lam, float)
    lam_c = math.pi * np.sqrt(2.0 * np.asarray(E, float) / np.maximum(sc, 1e-30))
    euler = math.pi ** 2 * E / np.maximum(lam, 1e-30) ** 2
    johnson = sc * (1.0 - sc * lam ** 2 / (4.0 * math.pi ** 2 * E))
    return np.where(lam >= lam_c, euler, johnson)


def column_util(N, A, I, H, K, E, sc):
    """(u_crush, u_buckle). A (m2); I, H, K arrays over the two axes (bending
    in x, bending in y); N the axial load. u_crush = N/(A sc); u_buckle per
    axis N/(A sigma_cr). Weakest axis governs: the caller takes the max."""
    A = np.asarray(A, float)
    I = np.asarray(I, float)
    r = np.sqrt(I / np.where(A > 0, A, 1.0))
    lam = np.asarray(K, float) * np.asarray(H, float) / np.where(r > 0, r, 1e-30)
    sig = johnson_euler_stress(lam, E, sc)
    return _ratio(N, A * sc), _ratio(N, A * sig)


def euler_critical_load(E, I, A, sc, H, K):
    """Load at which the column buckles (P_cr = A sigma_cr)."""
    r = math.sqrt(I / A)
    return float(A * johnson_euler_stress(K * H / r, E, sc))


def plan_section(fills, xy, h):
    """Section of a column made of cells side by side in plan. fills: (m,) fill
    of each cell, xy: (m, 2) integer cell coordinates. Returns A and I for
    bending in x and in y: the parallel-axis sum of each cell's own I (bar of
    side h sqrt(f)) about the area centroid."""
    f = np.clip(np.asarray(fills, float), 0.0, 1.0)
    xy = np.asarray(xy, float) * h
    A_j = f * h * h
    s = h * np.sqrt(f)
    I_own = s ** 4 / 12.0
    A = A_j.sum()
    I = np.zeros(2)
    for a in (0, 1):
        c = xy[:, a]
        cb = (A_j * c).sum() / A if A > 0 else 0.0
        I[a] = (I_own + A_j * (c - cb) ** 2).sum()
    return float(A), I


# ── masonry, arches ─────────────────────────────────────────────────────────
def no_tension_util(N, M, d, b, sc_eff):
    """Section b x d that takes no tension, axial compression N and moment M
    (e = M/N): e <= d/6 whole section in compression sigma = N/(b d) (1 + 6e/d)
    (2N/(b d) at d/6); d/6 < e < d/2 a stress block 2N/(3 b (d/2 - e));
    e >= d/2 the thrust line is outside and the joint opens (u = inf)."""
    N = np.asarray(N, float)
    M = np.abs(np.asarray(M, float))
    d = np.asarray(d, float)
    b = np.asarray(b, float)
    Ns = np.where(N > 0, N, 1.0)
    e = np.where(N > 0, M / Ns, np.where(M > 0, np.inf, 0.0))
    s1 = Ns / (b * d) * (1.0 + 6.0 * e / d)
    gap = np.maximum(d / 2.0 - e, 1e-300)
    s2 = 2.0 * Ns / (3.0 * b * gap)
    smax = np.where(e <= d / 6.0, s1, np.where(e < d / 2.0, s2, np.inf))
    smax = np.where((N <= 0) & (M <= 0), 0.0, smax)
    return smax / sc_eff


def arch_thrust(L, rise, w):
    """Three-hinge parabolic arch under uniform load w (N per metre of span):
    H = w L2/(8 f), crown force H, spring force sqrt(H2 + (w L/2)2)."""
    H = w * L ** 2 / (8.0 * rise) if rise > 0 else math.inf
    return H, H, math.hypot(H, w * L / 2.0)


def pier_check(H, V, base_width, height, mu_s):
    """Abutment under thrust H with vertical force V (pier plus arch half) and
    base width b, thrust acting `height` above the base. u_slide = H/(mu_s V);
    u_overturn = e/(b/2), e = H height/V the offset of the resultant from the
    base centre (1 = the resultant at the edge; the middle third, e <= b/6, is
    the no-tension limit)."""
    u_s = H / (mu_s * V) if V > 0 else math.inf
    u_o = (H * height / V) / (base_width / 2.0) if V > 0 else math.inf
    return u_s, u_o


def impact_factor(h_fall, d_beam, d_body=0.0):
    """Load multiplier of a body falling h_fall onto a member of static
    deflection d_beam, the body's own static compliance d_body in series
    (Roark; exact for two linear springs): 1 + sqrt(1 + 2 h/(d_beam + d_body)).
    2.0 at h = 0. d_body = 0 for a rigid object."""
    return 1.0 + math.sqrt(1.0 + 2.0 * h_fall / (d_beam + d_body))


def walker_extra(mass_kg, running=False):
    """Transient extra over body weight at the feet: (DAF - 1) m g."""
    return ((DAF_RUN if running else DAF_WALK) - 1.0) * mass_kg * G


def weights_N(smass_g, grip=None):
    """Weight of each cell (N) from its mass in grams. Grip-carried cells are
    zeroed: they hang from a hand and are on no column; the caller adds their
    weight at the holder's feet cells as a sustained extra."""
    W = np.asarray(smass_g, float) / 1000.0 * G
    if grip is not None:
        g = _mask(grip, W.shape)
        W = np.where(g, 0.0, W)
    return W


def profile_from_slice(mask2d, h=1.0):
    """Arch recogniser on an x-z slice (True = solid). Finds the widest opening
    on the bottom row that is walled on both sides, follows it upward row by
    row and returns (span, rise, thickness) in metres (cells x h): span = the
    opening width at the springing, rise = from the last full-width row to the
    top of the opening (0 for a flat lintel), thickness = solid above the
    crown. None when there is no opening. A dome is not an arch slice."""
    m = np.asarray(mask2d, bool)
    nx, nz = m.shape
    air = ~m[:, 0]
    best = None
    x = 0
    while x < nx:
        if air[x]:
            x0 = x
            while x < nx and air[x]:
                x += 1
            if x0 > 0 and x < nx and (best is None or x - x0 > best[1] - best[0]):
                best = (x0, x)
        else:
            x += 1
    if best is None:
        return None
    a, b = best
    w0 = b - a
    widths = [w0]
    z = 0
    cur = (a, b)
    while z + 1 < nz:
        row = ~m[:, z + 1]
        lo, hi = cur
        mid = (lo + hi) // 2
        seed = mid if row[mid] else next((i for i in range(lo, hi) if row[i]), None)
        if seed is None:
            break
        l = seed
        while l > 0 and row[l - 1]:
            l -= 1
        r = seed
        while r + 1 < nx and row[r + 1]:
            r += 1
        cur = (l, r + 1)
        z += 1
        widths.append(cur[1] - cur[0])
        if cur[1] - cur[0] >= nx:
            return None                      # open to the sky on both sides: not an opening
    z_top = z
    full = [i for i, w in enumerate(widths) if w >= w0]
    z_s = full[-1] if full else 0
    rise = (z_top - z_s) * h
    cx = (cur[0] + cur[1] - 1) // 2
    t = 0
    zz = z_top + 1
    while zz < nz and m[cx, zz]:
        t += 1
        zz += 1
    return w0 * h, rise, t * h


# ── members: the struct of runs and columns ─────────────────────────────────
@dataclass
class Run:
    """A line of member cells with air below. cells (n, 3) ordered root-first
    for a cantilever and low-to-high for a pinned run; xc is each centre's
    distance (m) from the left reaction (pinned) or the support face."""
    axis: int
    z: int
    cells: np.ndarray
    nsup: int
    sup_lo: object            # (x, y), "edge" or None; for a cantilever the root support
    sup_hi: object
    L: float
    xc: np.ndarray
    share: np.ndarray
    plate: np.ndarray
    fills: np.ndarray
    mat: np.ndarray
    eid: np.ndarray
    T: np.ndarray
    D: np.ndarray
    masonry: np.ndarray
    EI: np.ndarray
    stack: int = 1            # layers in the section (uniform for sharing)
    stacked_on: int = -1      # index of the run underneath when this one is a stacked beam
    two_way: bool = False

    @property
    def kind(self):
        return "pinned" if self.nsup == 2 else "cantilever"


@dataclass
class Column:
    """Member cells that are column cells in a vertical stack z0..z1 (all one
    material and eid); stacks of one REGISTERED eid side by side in plan are
    one column with one section, eid 0 stacks are each alone."""
    z0: int
    z1: int
    xy: np.ndarray            # (m, 2)
    mat: int
    eid: int
    fills: np.ndarray         # (m, nzl)
    T: np.ndarray             # (nzl,) hottest of the group, per layer
    D: np.ndarray
    braced: np.ndarray        # (nzl, 2) bool: braced in x, in y at that layer
    held_top: bool


@dataclass
class Members:
    shape: tuple
    h: float
    member: np.ndarray
    col: np.ndarray
    hang: np.ndarray
    masonry: np.ndarray
    beam: np.ndarray                    # cells that belong to a run or its section stack
    runs: list = field(default_factory=list)
    columns: list = field(default_factory=list)
    orphan: np.ndarray = None           # hanging solid cells in no run (old SPAN law keeps them)
    lane_groups: list = field(default_factory=list)


def _mask(v, shape):
    """A bool mask from None, a bool array or an iterable of (x, y, z)."""
    if v is None:
        return np.zeros(shape, bool)
    if isinstance(v, np.ndarray):
        return v.astype(bool)
    out = np.zeros(shape, bool)
    cells = list(v)
    if cells:
        c = np.asarray(cells, np.int64).reshape(-1, 3)
        ok = ((c >= 0) & (c < np.asarray(shape))).all(1)
        out[tuple(c[ok].T)] = True
    return out


def _lines(cells, axis, eid2d):
    """Maximal lines of consecutive cells along axis (0 = x, 1 = y), split
    where the eid changes. Returns [(fixed coordinate, lo, hi)]."""
    by = {}
    for (x, y) in cells:
        k, c = (y, x) if axis == 0 else (x, y)
        by.setdefault(k, []).append(c)
    out = []
    for k, cs in by.items():
        cs.sort()
        lo = prev = cs[0]
        for c in cs[1:]:
            xy = (c, k) if axis == 0 else (k, c)
            pxy = (prev, k) if axis == 0 else (k, prev)
            if c != prev + 1 or eid2d[xy] != eid2d[pxy]:
                out.append((k, lo, prev))
                lo = c
            prev = c
        out.append((k, lo, prev))
    return out


def _components(cells, eid2d):
    """4-connected groups of cells with the same eid."""
    left = set(cells)
    out = []
    while left:
        s = left.pop()
        comp = [s]
        stack = [s]
        while stack:
            x, y = stack.pop()
            for nb in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if nb in left and eid2d[nb] == eid2d[(x, y)]:
                    left.discard(nb)
                    comp.append(nb)
                    stack.append(nb)
        out.append(comp)
    return out


def _support(axis, k, c, col2d, edge_ok=False):
    nx, ny = col2d.shape
    n = nx if axis == 0 else ny
    if c < 0 or c >= n:
        return "edge" if edge_ok else None
    xy = (c, k) if axis == 0 else (k, c)
    return xy if col2d[xy] else None


class _Ctx:
    """Window arrays for the extraction, with the neighbour reads it needs."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _stack_up(c, x, y, z, nz):
    """z indices (including z) of the cells above (x, y, z) that are one built-up
    section with it: the same registered eid, the same material, members."""
    zs = [z]
    e = c.eid[x, y, z]
    if e <= 0 or not c.registered[x, y, z]:
        return zs
    m0 = c.mat[x, y, z]
    zz = z + 1
    while zz < nz and len(zs) < MAX_STACK and c.member[x, y, zz] \
            and c.eid[x, y, zz] == e and c.mat[x, y, zz] == m0:
        zs.append(zz)
        zz += 1
    return zs


def _plate_flag(c, x, y, z, axis):
    """A cell is a plate cell when a neighbour in the other horizontal axis is a
    member of the same material and eid (parallel lanes), else a bar."""
    for d in (-1, 1):
        xx, yy = (x, y + d) if axis == 0 else (x + d, y)
        if 0 <= xx < c.nx and 0 <= yy < c.ny and c.member[xx, yy, z] \
                and c.mat[xx, yy, z] == c.mat[x, y, z] and c.eid[xx, yy, z] == c.eid[x, y, z]:
            return True
    return False


def _build_run(c, axis, k, lo, hi, s_lo, s_hi, z, share=1.0, stacked_on=-1, two_way=False,
               uniform_stack=None):
    """Make a Run from a line. nsup 2 = pinned, 1 = cantilever (root first)."""
    h = c.h
    n = hi - lo + 1
    cs = list(range(lo, hi + 1))
    nsup = (s_lo is not None) + (s_hi is not None)
    if nsup == 1 and s_lo is None:
        cs = cs[::-1]
        s_lo, s_hi = s_hi, None
    elif nsup == 1:
        s_hi = None
    if axis == 0:
        cells = np.array([(a, k, z) for a in cs], np.int64)
    else:
        cells = np.array([(k, a, z) for a in cs], np.int64)
    i = np.arange(n)
    if nsup == 2:
        L = n * h + h / 2.0
        xc = (i + 0.75) * h
    else:
        L = n * h
        xc = (i + 0.5) * h
    K = 1
    stacks = []
    for (x, y, zz) in cells:
        zs = _stack_up(c, x, y, zz, c.nz)
        stacks.append(zs)
        K = max(K, len(zs))
    fills = np.zeros((n, K))
    T = np.zeros(n)
    D = np.zeros(n)
    for j, ((x, y, zz), zs) in enumerate(zip(cells, stacks)):
        for q, z2 in enumerate(zs):
            fills[j, q] = c.f[x, y, z2]
        T[j] = max(c.T[x, y, z2] for z2 in zs)
        D[j] = max(c.dmg[x, y, z2] for z2 in zs)
    mat = c.mat[tuple(cells.T)].astype(np.int64)
    eid = c.eid[tuple(cells.T)]
    plate = np.array([_plate_flag(c, x, y, zz, axis) for (x, y, zz) in cells])
    eff = effective(c.rec, mat, T, D)
    sec = stack_section(fills, h, plate)
    ks_all = {len(s) for s in stacks}
    r = Run(axis=axis, z=z, cells=cells, nsup=nsup, sup_lo=s_lo, sup_hi=s_hi, L=L, xc=xc,
            share=np.full(n, float(share)), plate=plate, fills=fills, mat=mat, eid=eid, T=T, D=D,
            masonry=c.masonry[tuple(cells.T)], EI=eff["E"] * sec["I"],
            stack=(ks_all.pop() if len(ks_all) == 1 else -K), stacked_on=stacked_on, two_way=two_way)
    return r


def extract_members(solid, fill, mat, eid, dmg, T, rubble, owned, grip, registered, joint,
                    moving, h, rec=None, edge_supports=False):
    """Runs and columns of a window of the lattice. All arrays have the window's
    shape: solid (bool), fill (0..1), mat (ids), eid, dmg (D_other), T (C), rubble,
    owned (a person's claim), grip (grip-carried or borne cells, a mask or an
    iterable of (x, y, z)), registered (a registered thing), joint (cells that
    hold a tension JOINT record, none until L10), moving (particle cells). Any
    of the optional ones may be None. rec is the dict from rec_from_rows (default:
    matrec). Cells in owned, grip, rubble or moving, and cells whose material
    fails the member test, are not members. edge_supports: a run that reaches the
    window edge counts the edge as a support (True for a sub-window of a bigger
    world, False for the whole world, where beyond the edge is nothing).
    Returns Members."""
    solid = np.asarray(solid, bool)
    shape = solid.shape
    nx, ny, nz = shape
    rec = rec if rec is not None else rec_from_matrec()
    mat = np.asarray(mat).astype(np.int64)
    eid = np.zeros(shape, np.int64) if eid is None else np.asarray(eid).astype(np.int64)
    f = np.clip(np.asarray(fill, float), 0.0, 1.0)
    dmg = np.zeros(shape) if dmg is None else np.asarray(dmg, float)
    T = np.full(shape, 20.0) if T is None else np.asarray(T, float)
    rubble, owned, moving, joint = (_mask(v, shape) for v in (rubble, owned, moving, joint))
    grip = _mask(grip, shape)
    registered = (eid > 0) if registered is None else np.asarray(registered, bool)
    member = (solid & rec["structural"][mat] & ~rubble & ~owned & ~grip & ~moving & (f > 0))
    below = np.ones(shape, bool)
    below[:, :, 1:] = solid[:, :, :-1]
    col = solid & below
    hang = solid & ~below
    masonry = (member & ~joint & ~registered
               & (rec["st"][mat] < MASONRY_RATIO * rec["sc"][mat]))
    c = _Ctx(nx=nx, ny=ny, nz=nz, h=h, f=f, mat=mat, eid=eid, dmg=dmg, T=T, member=member,
             registered=registered, masonry=masonry, rec=rec, solid_col=col,
             edge_ok=bool(edge_supports))
    m = Members(shape=shape, h=h, member=member, col=col, hang=hang, masonry=masonry,
                beam=np.zeros(shape, bool), orphan=np.zeros(shape, bool))
    held = set()

    def add(run):
        m.runs.append(run)
        for (x, y, z) in run.cells:
            m.beam[x, y, z:z + max(abs(run.stack), 1)] = True
        if run.nsup == 2:
            for s in (run.sup_lo, run.sup_hi):
                if isinstance(s, tuple):
                    held.add((s[0], s[1], run.z))
        return len(m.runs) - 1

    for z in range(1, nz):
        hm = hang[:, :, z] & member[:, :, z]
        if not hm.any():
            continue
        col2d = col[:, :, z]
        eid2d = eid[:, :, z]
        for comp in _components([(int(a), int(b)) for a, b in np.argwhere(hm)], eid2d):
            _runs_of_component(c, m, add, comp, z, col2d, eid2d)
        # a hanging member cell left in no run falls to the old law
    # stacked beams: a run resting on a run, on the same two supports
    for ri in range(len(m.runs)):
        r = m.runs[ri]
        if r.nsup != 2 or r.stacked_on >= 0 or r.share.min() < 1.0 or r.stack < 1:
            continue
        low = ri
        while True:
            lr = m.runs[low]
            if lr.stack < 1 or lr.nsup != 2:
                break
            zu = lr.z + lr.stack
            up = _stacked_beam(c, m, lr, zu, low)
            if up is None:
                break
            low = add(up)
    # orphans: hanging solid cells with no run
    inrun = np.zeros(shape, bool)
    for r in m.runs:
        inrun[tuple(r.cells.T)] = True
    m.orphan = hang & ~inrun
    # lane groups: parallel pinned runs of the same extent on adjacent lanes
    key = {}
    for ri, r in enumerate(m.runs):
        if r.nsup == 2 and r.stacked_on < 0 and not r.two_way and r.plate.all():
            lo = r.cells[0, r.axis]
            key.setdefault((r.axis, r.z, lo, len(r.cells)), []).append(ri)
    for g in key.values():
        g.sort(key=lambda i: m.runs[i].cells[0, 1 - m.runs[i].axis])
        grp, prev = [], None
        for i in g:
            kk = m.runs[i].cells[0, 1 - m.runs[i].axis]
            if prev is not None and kk != prev + 1:
                if len(grp) > 1:
                    m.lane_groups.append(grp)
                grp = []
            grp.append(i)
            prev = kk
        if len(grp) > 1:
            m.lane_groups.append(grp)
    # columns
    _extract_columns(c, m, held)
    return m


def _runs_of_component(c, m, add, comp, z, col2d, eid2d):
    """Pick the span direction of one hanging component and build its runs."""
    total = len(comp)
    lines = {}
    score = {}
    for a in (0, 1):
        rs = []
        n2 = n1 = 0
        for (k, lo, hi) in _lines(comp, a, eid2d):
            s_lo = _support(a, k, lo - 1, col2d, c.edge_ok)
            s_hi = _support(a, k, hi + 1, col2d, c.edge_ok)
            ns = (s_lo is not None) + (s_hi is not None)
            rs.append((k, lo, hi, s_lo, s_hi, ns))
            n2 += (hi - lo + 1) * (ns == 2)
            n1 += (hi - lo + 1) * (ns == 1)
        lines[a], score[a] = rs, (n2, n1)
    h = c.h
    if score[0][0] == total and score[1][0] == total:
        # supported on all sides: two-way. A strip in x takes Ly4/(Lx4+Ly4)
        span = {0: {}, 1: {}}
        for a in (0, 1):
            for (k, lo, hi, s_lo, s_hi, ns) in lines[a]:
                for q in range(lo, hi + 1):
                    xy = (q, k) if a == 0 else (k, q)
                    span[a][xy] = (hi - lo + 1) * h + h / 2.0
        for a in (0, 1):
            for (k, lo, hi, s_lo, s_hi, ns) in lines[a]:
                sh = []
                for q in range(lo, hi + 1):
                    xy = (q, k) if a == 0 else (k, q)
                    lx, ly = span[0][xy], span[1][xy]
                    rx = ly ** 4 / (lx ** 4 + ly ** 4)
                    sh.append(rx if a == 0 else 1.0 - rx)
                r = _build_run(c, a, k, lo, hi, s_lo, s_hi, z, two_way=True)
                r.share = np.array(sh)
                add(r)
        return
    a = 0 if score[0] >= score[1] else 1
    used = set()
    for (k, lo, hi, s_lo, s_hi, ns) in lines[a]:
        if ns == 0:
            continue
        add(_build_run(c, a, k, lo, hi, s_lo, s_hi, z))
        for q in range(lo, hi + 1):
            used.add((q, k) if a == 0 else (k, q))
    left = [p for p in comp if p not in used]
    if left:                                    # the other direction, for cells nothing else holds
        b = 1 - a
        for (k, lo, hi) in _lines(left, b, eid2d):
            s_lo = _support(b, k, lo - 1, col2d, c.edge_ok)
            s_hi = _support(b, k, hi + 1, col2d, c.edge_ok)
            if s_lo is not None or s_hi is not None:
                add(_build_run(c, b, k, lo, hi, s_lo, s_hi, z))


def _stacked_beam(c, m, low, zu, low_idx):
    """The run resting directly on pinned run `low` (above its stack) if it is a
    beam on the same two supports: members, not one thing with it, and the
    cells beyond its ends at that layer are column cells."""
    if zu >= c.nz:
        return None
    cells = low.cells.copy()
    cells[:, 2] = zu
    xs, ys = cells[:, 0], cells[:, 1]
    if not c.member[xs, ys, zu].all() or c.masonry[xs, ys, zu].any():
        return None                           # loose blocks do not act as a beam over the one below
    eids = c.eid[xs, ys, zu]
    if (eids != eids[0]).any() or (c.mat[xs, ys, zu] != c.mat[xs, ys, zu][0]).any():
        return None
    a = low.axis
    k = int(cells[0, 1 - a])
    lo, hi = int(cells[:, a].min()), int(cells[:, a].max())
    col2d = (c.solid_col[:, :, zu])
    s_lo = _support(a, k, lo - 1, col2d, c.edge_ok)
    s_hi = _support(a, k, hi + 1, col2d, c.edge_ok)
    if s_lo is None or s_hi is None:
        return None
    r = _build_run(c, a, k, lo, hi, s_lo, s_hi, zu, stacked_on=low_idx)
    return r


def _extract_columns(c, m, held):
    nx, ny, nz = m.shape
    colm = m.member & m.col & ~m.beam
    stacks = []
    for x, y in np.argwhere(colm.any(2)):
        zs = np.nonzero(colm[x, y])[0]
        z0 = prev = int(zs[0])
        for z in list(zs[1:]) + [None]:
            end = z is None or z != prev + 1 or c.mat[x, y, z] != c.mat[x, y, z0] \
                or c.eid[x, y, z] != c.eid[x, y, z0]
            if end:
                stacks.append((int(x), int(y), z0, prev))
                if z is not None:
                    z0 = int(z)
            if z is not None:
                prev = int(z)
    groups = {}
    for (x, y, z0, z1) in stacks:
        e = int(c.eid[x, y, z0])
        reg = e > 0 and bool(c.registered[x, y, z0])
        key = (int(c.mat[x, y, z0]), e, z0, z1) if reg else ("solo", x, y, z0, z1)
        groups.setdefault(key, []).append((x, y))
    for key, cells in groups.items():
        if key[0] == "solo":
            parts = [cells]
            z0, z1 = key[3], key[4]
        else:
            z0, z1 = key[2], key[3]
            eid2d = np.zeros((nx, ny), np.int64)
            parts = _components(cells, eid2d)
        for part in parts:
            x0, y0 = part[0]
            xy = np.array(part, np.int64)
            nzl = z1 - z0 + 1
            fills = np.stack([c.f[x, y, z0:z1 + 1] for x, y in part])
            T = np.max([c.T[x, y, z0:z1 + 1] for x, y in part], axis=0)
            D = np.max([c.dmg[x, y, z0:z1 + 1] for x, y in part], axis=0)
            inset = set(part)
            braced = np.zeros((nzl, 2), bool)
            for q, (x, y) in enumerate(part):
                for a in (0, 1):
                    for d in (-1, 1):
                        xx, yy = (x + d, y) if a == 0 else (x, y + d)
                        if (xx, yy) in inset or not (0 <= xx < nx and 0 <= yy < ny):
                            continue
                        braced[:, a] |= m.member[xx, yy, z0:z1 + 1]
            ht = any((x, y, z1 + 1) in held for x, y in part)
            m.columns.append(Column(z0=z0, z1=z1, xy=xy, mat=int(c.mat[x0, y0, z0]),
                                    eid=int(c.eid[x0, y0, z0]), fills=fills, T=T, D=D,
                                    braced=braced, held_top=ht))


# ── the load path ───────────────────────────────────────────────────────────
def _frac_inside(d):
    """Fraction of a cell inside the 45 degree line when its centre is d cells
    above (+) or below (-) it: exact for a square cut by a diagonal."""
    d = np.asarray(d, float)
    return np.where(d >= 1.0, 1.0, np.where(d >= 0.0, 1.0 - (1.0 - d) ** 2 / 2.0,
                                            np.where(d > -1.0, (1.0 + d) ** 2 / 2.0, 0.0)))


def _arch_plan(m):
    """Cells above an opening that a joint-less masonry wall shifts to the
    abutments: {z: [(x, y, phi, tx, ty)]}, phi the fraction that still goes down
    onto the run (the cells inside the 45 degree triangle on the clear span)."""
    nx, ny, nz = m.shape
    h = m.h
    plan = {}
    solid = m.col | m.hang
    for r in m.runs:
        if r.nsup != 2 or r.two_way or (r.share < 1).any():
            continue
        a = r.axis
        n = len(r.cells)
        k = int(r.cells[0, 1 - a])
        c0, c1 = int(r.cells[:, a].min()), int(r.cells[:, a].max())
        ztop = r.z + max(r.stack, 1) - 1
        xc = (np.arange(n) + 0.5) * h
        s_near = np.minimum(xc, n * h - xc)
        for kk in range(1, nz):
            zk = ztop + kk
            if zk >= nz:
                break
            idx = np.arange(c0 - 1, c1 + 2)
            if idx[0] < 0 or idx[-1] >= (nx if a == 0 else ny):
                break
            cx = idx if a == 0 else np.full(len(idx), k)
            cy = np.full(len(idx), k) if a == 0 else idx
            if not (solid[cx, cy, zk].all() and m.masonry[cx, cy, zk].all()):
                break
            phi = _frac_inside((s_near - (kk - 0.5) * h) / h)
            for j in range(n):
                if phi[j] >= 1.0:
                    continue
                q = c0 + j
                lo_side = xc[j] <= n * h / 2.0
                t = c0 - 1 if lo_side else c1 + 1
                src = (q, k) if a == 0 else (k, q)
                tgt = (t, k) if a == 0 else (k, t)
                plan.setdefault(zk, {})[src] = (float(phi[j]), tgt)
            if phi.max() <= 1e-9:
                break
    out = {}
    for zk, d in plan.items():
        rows = []
        for src, (phi, tgt) in d.items():
            hops = 0
            while tgt in d and hops < 4:
                tgt = d[tgt][1]
                hops += 1
            rows.append((src[0], src[1], phi, tgt[0], tgt[1]))
        out[zk] = rows
    return out


def _route_orphans(q, orph, col2d, lost):
    """Hanging solid cells held by nothing in this layer (a person's arm, a
    snapped piece) pass their load to the neighbours nearer a column cell, so a
    body's weight reaches its feet. What reaches no column goes to `lost`."""
    pts = [(int(a), int(b)) for a, b in np.argwhere(orph)]
    if not pts:
        return
    nx, ny = orph.shape
    st = set(pts)
    dist = {}
    front = []
    for p in pts:
        for nb in ((p[0] + 1, p[1]), (p[0] - 1, p[1]), (p[0], p[1] + 1), (p[0], p[1] - 1)):
            if 0 <= nb[0] < nx and 0 <= nb[1] < ny and col2d[nb]:
                dist[p] = 1
                front.append(p)
                break
    d = 1
    while front:
        nxt = []
        for p in front:
            for nb in ((p[0] + 1, p[1]), (p[0] - 1, p[1]), (p[0], p[1] + 1), (p[0], p[1] - 1)):
                if nb in st and nb not in dist:
                    dist[nb] = d + 1
                    nxt.append(nb)
        front, d = nxt, d + 1
    for p in sorted(pts, key=lambda p: -dist.get(p, 0)):
        if p not in dist:
            lost += q[:, p[0], p[1]]
            q[:, p[0], p[1]] = 0.0
            continue
        tg = []
        for nb in ((p[0] + 1, p[1]), (p[0] - 1, p[1]), (p[0], p[1] + 1), (p[0], p[1] - 1)):
            if not (0 <= nb[0] < nx and 0 <= nb[1] < ny):
                continue
            if dist[p] == 1 and col2d[nb]:
                tg.append(nb)
            elif dist[p] > 1 and dist.get(nb, 1 << 30) == dist[p] - 1:
                tg.append(nb)
        v = q[:, p[0], p[1]] / len(tg)
        for nb in tg:
            q[:, nb[0], nb[1]] += v
        q[:, p[0], p[1]] = 0.0


@dataclass
class Loads:
    N: np.ndarray                 # (2, nx, ny, nz): force each cell passes down (or into its beam)
    run_P: list                   # per run: (2, n) point loads on the beam (sustained, transient)
    run_R: list                   # per run: (2, 2) reactions [component, (lo, hi)]; cantilever root in lo
    ground: np.ndarray            # (2,) into the floor below the window
    edge: np.ndarray              # (2,) into supports past the window edge
    lost: np.ndarray              # (2,) from cells held by nothing (old law owns them)
    total: np.ndarray             # (2,) put in


def accumulate_loads(m, weight_N, extra_N=None, arch=True, spread=True, foot=FOOT_M, ortho=1.0):
    """Top-to-bottom load path (4.4 steps 1-2). weight_N: each cell's weight
    (sustained; use `weights_N`, grip cells zeroed). extra_N: transient loads
    at cells (feet of walkers, a landing; sustained carried things are added to
    weight_N at the holder's feet). Reactions sum to the total weight:
    ground + edge + lost == total to rounding. arch: a joint-less masonry wall
    over an opening loads the run with the 45 degree triangle only. spread:
    a concentrated transient load on a plate spreads over the effective width."""
    nx, ny, nz = m.shape
    h = m.h
    solid = m.col | m.hang
    Wt = np.zeros((2, nx, ny, nz))
    Wt[0] = np.where(solid, weight_N, 0.0)
    if extra_N is not None:
        Wt[1] = np.where(solid, extra_N, 0.0)
    inc = np.zeros_like(Wt)
    N = np.zeros_like(Wt)
    ground = np.zeros(2)
    edge = np.zeros(2)
    lost = np.zeros(2)
    run_P = [None] * len(m.runs)
    run_R = [None] * len(m.runs)
    plan = _arch_plan(m) if arch else {}
    by_z = {}
    for i, r in enumerate(m.runs):
        by_z.setdefault(r.z, []).append(i)
    in_group = {i: g for g in m.lane_groups for i in g}
    beam_layer = np.zeros(m.shape, bool)
    for r in m.runs:
        if r.stacked_on >= 0:
            beam_layer[tuple(r.cells.T)] = True
    for z in range(nz - 1, -1, -1):
        q = inc[:, :, :, z] + Wt[:, :, :, z]
        for (x, y, phi, tx, ty) in plan.get(z, ()):
            moved = (1.0 - phi) * q[:, x, y]
            q[:, tx, ty] += moved
            q[:, x, y] -= moved
        # beams of this layer: point loads first (a stacked beam keeps only the
        # part it does not hand down to the run under it), then the spread of
        # a concentrated transient load over a plate's lanes, then reactions
        P = {}
        for i in by_z.get(z, ()):
            r = m.runs[i]
            cx, cy = r.cells[:, 0], r.cells[:, 1]
            p = q[:, cx, cy] * r.share
            if r.stacked_on >= 0:
                lo = m.runs[r.stacked_on]
                qU = float(p.sum())
                qR = float(sum(Wt[0, x, y, lo.z:lo.z + max(lo.stack, 1)].sum()
                               for x, y, _ in lo.cells))
                EIU, EIR = float(r.EI.mean()), float(lo.EI.mean())
                pt = (qU * EIR - qR * EIU) / (EIU + EIR) if qU > 0 and EIU + EIR > 0 else 0.0
                fd = float(np.clip(pt / qU, 0.0, 1.0)) if qU > 0 else 0.0
                inc[:, cx, cy, z - 1] += p * fd
                p = p * (1.0 - fd)
            P[i] = p
        for i in P:
            r = m.runs[i]
            q[:, r.cells[:, 0], r.cells[:, 1]] = 0.0
        if spread:
            for g in {id(in_group[i]): in_group[i] for i in P if i in in_group}.values():
                if not all(j in P for j in g):
                    continue
                arr = np.stack([P[j][1] for j in g])          # (lanes, n)
                for k in range(arr.shape[1]):
                    arr[:, k] = spread_lanes(arr[:, k], h, m.runs[g[0]].L, foot, ortho)
                for k, j in enumerate(g):
                    P[j][1] = arr[k]
        for i, p in P.items():
            r = m.runs[i]
            cx, cy = r.cells[:, 0], r.cells[:, 1]
            if r.nsup == 2:
                Rlo = (p * ((r.L - r.xc) / r.L)).sum(1)
                R = np.stack([Rlo, p.sum(1) - Rlo], axis=1)
                tg = ((r.sup_lo, R[:, 0]), (r.sup_hi, R[:, 1]))
            else:
                R = np.stack([p.sum(1), np.zeros(2)], axis=1)
                tg = ((r.sup_lo, R[:, 0]),)
            for s, v in tg:
                if s == "edge":
                    edge += v
                elif beam_layer[s[0], s[1], z]:
                    lost += v                 # a support that is itself a stacked beam cell
                else:
                    q[:, s[0], s[1]] += v
            run_P[i] = p
            run_R[i] = R
            N[:, cx, cy, z] = p
        passing = m.col[:, :, z] & ~beam_layer[:, :, z]
        orph = m.orphan[:, :, z] & solid[:, :, z]
        if orph.any():
            _route_orphans(q, orph, passing, lost)
        N[:, :, :, z] = np.where(passing, q, N[:, :, :, z])
        if z > 0:
            inc[:, :, :, z - 1] += np.where(passing, q, 0.0)
        else:
            ground += np.where(passing, q, 0.0).reshape(2, -1).sum(1)
    total = Wt.reshape(2, -1).sum(1)
    return Loads(N=N, run_P=run_P, run_R=run_R, ground=ground, edge=edge, lost=lost, total=total)


# ── the checks ──────────────────────────────────────────────────────────────
@dataclass
class Analysis:
    u: np.ndarray                 # per cell: the largest utilisation of the member it belongs to at that cell
    mode: np.ndarray              # M_* code of that utilisation
    rows: list                    # one dict per member

    def over(self, thresh=0.5):
        return sorted((r for r in self.rows if r["u"] > thresh), key=lambda r: -r["u"])

    def failing(self):
        return [r for r in self.rows if r["u"] >= 1.0]


def with_extra_damage(m, dD):
    """A copy of the members with a new cause of damage dD (a scalar) combined
    into every run's and column's D (matdamage.combine: independent causes), to
    ask "what does this do to what is standing" without writing the lattice."""
    runs = [replace(r, D=combine(r.D, dD)) for r in m.runs]
    cols = [replace(c, D=combine(c.D, dD)) for c in m.columns]
    return replace(m, runs=runs, columns=cols)


def _combine_u(u_s, u_t, gain):
    return u_s + u_t / gain


def analyse(m, rec, loads, gain=1.0, no_tension=False, only_runs=None, only_cols=None):
    """Utilisation of every member. The transient part of the load is carried
    at `gain` times the capacity (the short-load duration gain, 1.0 = none,
    at most GAIN_CAP for wood by matdamage.duration_factor). no_tension=True
    treats dry-masonry cells (no joint record, not registered) as sections that
    take no tension: any bending moment without a compression to hold the
    thrust line fails them (u = inf), as a flat lintel of loose blocks does.
    only_runs / only_cols (sets of indices into m.runs / m.columns, or None =
    all): analyse just those members (the loads were worked out for all of
    them). A caller that knows where something changed asks for the members
    near it and not for the whole building."""
    h = m.h
    u = np.zeros(m.shape, np.float32)
    mode = np.zeros(m.shape, np.int8)
    rows = []
    for ri, r in enumerate(m.runs):
        if only_runs is not None and ri not in only_runs:
            continue
        P = loads.run_P[ri]
        eff = effective(rec, r.mat, r.T, r.D)
        sec = stack_section(r.fills, h, r.plate)
        parts = []
        for comp in (0, 1):
            if r.nsup == 2:
                M, V = pinned_demand(P[comp], r.xc, r.L, h)
            else:
                M, V = cantilever_moments(P[comp], r.xc, h)
            parts.append((M, V))
        ub_s, uv_s, _ = section_util(parts[0][0], parts[0][1], 0.0, sec, eff)
        ub_t, uv_t, _ = section_util(parts[1][0], parts[1][1], 0.0, sec, eff)
        ub = _combine_u(ub_s, ub_t, gain)
        uv = _combine_u(uv_s, uv_t, gain)
        uc = np.maximum(ub, uv)
        md = np.where(ub >= uv, M_BEND, M_SHEAR)
        us = np.maximum(ub_s, uv_s)
        if no_tension:
            mas = r.masonry & ((parts[0][0] + parts[1][0]) > 0)
            uc = np.where(mas, np.inf, uc)
            md = np.where(mas, M_NOTEN, md)
        j = int(np.argmax(uc))
        cell = tuple(int(v) for v in r.cells[j])
        for jj, cc in enumerate(r.cells):
            x, y, z = (int(v) for v in cc)
            if uc[jj] >= u[x, y, z]:
                u[x, y, z] = uc[jj]
                mode[x, y, z] = md[jj]
        rows.append({"kind": r.kind if r.stacked_on < 0 else "stacked-" + r.kind,
                     "cell": cell, "mat": int(r.mat[j]), "mode": int(md[j]), "u": float(uc[j]),
                     "u_sus": float(us.max()), "n": len(r.cells), "axis": r.axis,
                     "plate": bool(r.plate[j]), "nsup": r.nsup, "run": ri})
    for ci, col in enumerate(m.columns):
        if only_cols is not None and ci not in only_cols:
            continue
        rows.append(_column_row(m, col, rec, loads, gain, u, mode, ci))
    return Analysis(u=u, mode=mode, rows=rows)


def _column_row(m, col, rec, loads, gain, u, mode, ci):
    h = m.h
    nzl = col.z1 - col.z0 + 1
    xs, ys = col.xy[:, 0], col.xy[:, 1]
    Nl = np.stack([loads.N[c, xs, ys, col.z0:col.z1 + 1].sum(0) for c in (0, 1)])   # (2, nzl)
    matv = np.full(nzl, col.mat, np.int64)
    eff = effective(rec, matv, col.T, col.D)
    A = np.zeros(nzl)
    I = np.zeros((nzl, 2))
    for l in range(nzl):
        A[l], I[l] = plan_section(col.fills[:, l], col.xy, h)
    ucr = _combine_u(_ratio(Nl[0], A * eff["sc"]), _ratio(Nl[1], A * eff["sc"]), gain)
    ubk = np.zeros((nzl, 2))
    for a in (0, 1):
        free = ~col.braced[:, a]
        l = 0
        while l < nzl:
            if not free[l]:
                l += 1
                continue
            s = l
            while l < nzl and free[l]:
                l += 1
            e = l - 1
            H = (e - s + 1) * h + (h / 2.0 if s > 0 else 0.0) + (h / 2.0 if e < nzl - 1 else 0.0)
            K = K_FREE if (e == nzl - 1 and not col.held_top) else K_HELD
            sl = slice(s, e + 1)
            Es, scs = eff["E"][sl].min(), eff["sc"][sl].min()
            As, Is = A[sl].min(), I[sl, a].min()
            nss = Nl[0][sl].max()
            nst = Nl[1][sl].max()
            if As <= 0 or Es <= 0 or scs <= 0:
                ubk[sl, a] = np.inf
                continue
            _, k_s = column_util(nss, As, Is, H, K, Es, scs)
            _, k_t = column_util(nst, As, Is, H, K, Es, scs)
            ubk[sl, a] = _combine_u(k_s, k_t, gain)
    ucell = np.maximum(ucr, ubk.max(1))
    lcrit = int(np.argmax(ucell))
    crush_wins = ucr[lcrit] >= ubk[lcrit].max()
    if crush_wins:
        pick = int(np.argmax((loads.N[0, xs, ys, col.z0 + lcrit] + loads.N[1, xs, ys, col.z0 + lcrit])))
        md = M_CRUSH
    else:
        pick = 0
        md = M_BUCKLE
    x, y = int(xs[pick]), int(ys[pick])
    for l in range(nzl):
        for q in range(len(xs)):
            if ucell[l] >= u[xs[q], ys[q], col.z0 + l]:
                u[xs[q], ys[q], col.z0 + l] = ucell[l]
                mode[xs[q], ys[q], col.z0 + l] = M_CRUSH if ucr[l] >= ubk[l].max() else M_BUCKLE
    return {"kind": "column", "cell": (x, y, col.z0 + lcrit), "mat": col.mat, "mode": md,
            "u": float(ucell[lcrit]), "u_sus": float(ucell[lcrit]), "n": nzl * len(xs), "axis": -1,
            "plate": False, "nsup": 0, "run": -1, "col": ci,
            "u_crush": float(ucr.max()), "u_buckle": [float(ubk[:, 0].max()), float(ubk[:, 1].max())]}
