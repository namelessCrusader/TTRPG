"""WATER IN MIXTURES (wave 13, T4, law L14): vapour, ice, and the activity of water.

One master quantity, the activity of water aw, gives the boiling rise, the
freezing drop, the osmotic pressure and the humidity in equilibrium, so they
cannot disagree. Pure numpy, no World. Temperature in C, pressure in Pa, mass
in g; `comp` is a float array of grams with the species on the LAST axis in the
order of mixdata.ORDER (dissolved grams for the water-activity functions).

  psat_water, tsat_water   IAPWS-IF97 region 4 (NaN outside the triple point to
                           the critical point: the closed form would otherwise
                           give plausible wrong numbers)
  psat_ice, ln_ice_ratio   Murphy and Koop 2005; Lfus(T) = 333.55 + 2.09 T
  ln_aw, aw                1:1 electrolytes by Pitzer, sugars by Norrish, spirits
                           by Van Laar, the rest ideal; several solutes by the
                           ZSR rule (the same species split in halves cannot move aw)
  boiling_rise ...         offsets above pure water: pure water gives exactly 0.0
  freezing_drop, freeze_point, osmolality, tonicity, osmotic_pressure,
  density, heat_capacity, coverage, water_for_aw, water_for_boiling_rise,
  abv_to_grams

No function returns a flag ("safe", "cured"); thresholds are data elsewhere.
"""
import numpy as np

from . import mixdata as D

_N = D.row("if97", "n")
_PMIN = D.row("if97", "p_min_pa")
_PMAX = D.row("if97", "p_max_pa")
_TMAX = D.row("if97", "t_max_c")
_MK_ICE = D.row("mk05", "ice")
_MK_LIQ = D.row("mk05", "liquid")
MW = D.MW
_T0 = D.T0K

# the dilute gas/liquid cut for the molality: below this a cell has no water at all
_TINY = 1e-300


def prep(comp):
    """comp as float64 with the species axis exactly mixdata.n_species() wide (a
    narrower array is padded with zeros: a pack may add a species later)."""
    a = np.asarray(comp, dtype=float)
    n = D.n_species()
    if a.shape[-1] != n:
        if a.shape[-1] > n:
            raise ValueError(f"comp has {a.shape[-1]} species columns, the registry has {n}")
        pad = np.zeros(a.shape[:-1] + (n,))
        pad[..., :a.shape[-1]] = a
        a = pad
    return a


# -- vapour pressure of water (IF97 region 4) -----------------------------------
def _psat_k(Tk):
    """Saturation pressure in Pa from T in K, no domain guard (internal)."""
    n = _N
    th = Tk + n[8] / (Tk - n[9])
    th2 = th * th
    A = th2 + n[0] * th + n[1]
    B = n[2] * th2 + n[3] * th + n[4]
    C = n[5] * th2 + n[6] * th + n[7]
    with np.errstate(invalid="ignore"):
        return (2.0 * C / (-B + np.sqrt(B * B - 4.0 * A * C))) ** 4 * 1e6


def psat_water(T):
    """Saturation pressure of liquid water, Pa, T in C. NaN below 0 C (273.15 K)
    and above the critical temperature."""
    T = np.asarray(T, dtype=float)
    p = _psat_k(T + _T0)
    return np.where((T >= 0.0) & (T <= _TMAX), p, np.nan)


def tsat_water(P):
    """Boiling point of pure water, C, P in Pa (closed-form IF97 inverse). NaN
    below the triple-point pressure (611.657 Pa) and above 22.064 MPa."""
    P = np.asarray(P, dtype=float)
    n = _N
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.sqrt(np.sqrt(P * 1e-6))
        b2 = b * b
        E = b2 + (n[2] * b + n[5])
        F = n[0] * b2 + (n[3] * b + n[6])
        G = n[1] * b2 + (n[4] * b + n[7])
        Dd = 2.0 * G / (-F - np.sqrt(F * F - 4.0 * E * G))
        u = n[9] + Dd
        t = 0.5 * u - 0.5 * np.sqrt(u * u - 4.0 * (n[8] + n[9] * Dd)) - _T0
    return np.where((P >= _PMIN) & (P <= _PMAX), t, np.nan)


# -- ice and supercooled water (Murphy and Koop 2005) -----------------------------
def _p_ice_k(Tk):
    c = _MK_ICE
    return np.exp(c[0] + c[1] / Tk + c[2] * np.log(Tk) + c[3] * Tk)


def _p_liq_k(Tk):
    c = _MK_LIQ
    return np.exp(c[0] + c[1] / Tk + c[2] * np.log(Tk) + c[3] * Tk
                  + np.tanh(c[4] * (Tk - c[5])) * (c[6] + c[7] / Tk + c[8] * np.log(Tk) + c[9] * Tk))


def psat_ice(T):
    """Vapour pressure over ice, Pa, T in C. NaN above 0.01 C (no ice there)."""
    T = np.asarray(T, dtype=float)
    return np.where(T <= 0.01, _p_ice_k(T + _T0), np.nan)


_LIR0 = float(np.log(_p_ice_k(_T0) / _p_liq_k(_T0)))


def ln_ice_ratio(T):
    """ln(p_ice / p_liquid) at T in C, shifted to be exactly 0 at 0 C: the water
    activity a solution must have to sit in equilibrium with ice."""
    Tk = np.asarray(T, dtype=float) + _T0
    return np.log(_p_ice_k(Tk) / _p_liq_k(Tk)) - _LIR0


def Lfus(T):
    """Heat of fusion of ice, J/g, at T in C: 333.55 + 2.09 T (Kirchhoff, cp
    liquid 4.18, ice 2.09: 289.4 J/g at -21 C)."""
    return D.L_FUS0 + D.CP_ICE * np.asarray(T, dtype=float)


# -- the binary water-activity curves ---------------------------------------------
_APH_T = np.array(sorted(D.TABLES["aphi"]))
_APH_V = np.array([D.TABLES["aphi"][t][0] for t in _APH_T])
_APH_S0 = (_APH_V[1] - _APH_V[0]) / (_APH_T[1] - _APH_T[0])
_APH_S1 = (_APH_V[-1] - _APH_V[-2]) / (_APH_T[-1] - _APH_T[-2])
# two far knots on the end slopes make one np.interp a linear extension at both ends
_APH_XT = np.concatenate([[-400.0], _APH_T, [1000.0]])
_APH_XV = np.concatenate([[_APH_V[0] + _APH_S0 * (-400.0 - _APH_T[0])], _APH_V,
                          [_APH_V[-1] + _APH_S1 * (1000.0 - _APH_T[-1])]])


def aphi(T):
    """Debye-Hueckel osmotic slope, interpolated, extended linearly at both ends."""
    return np.interp(np.asarray(T, dtype=float), _APH_XT, _APH_XV)


_PITZ = {k: (D.row("pitzer", k + "_25"), D.row("pitzer", k + "_slope"), D.row("pitzer", k + "_kappa"))
         for k in ("nacl", "kcl")}


def phi_pitzer(m, T, key="nacl"):
    """Osmotic coefficient of a 1:1 electrolyte. Below 25 C the b0, b1, C follow
    their temperature slopes; above 25 C they stay at their 25 C values and the
    whole m-dependent part is scaled by s = 1 + kappa (T - 25) (kappa fitted so
    saturated NaCl boils at 108.70 C). Dilute limit phi -> 1 is left alone."""
    (b0, b1, c), (d0, d1, dc), kap = _PITZ[key]
    m = np.asarray(m, dtype=float)
    T = np.asarray(T, dtype=float)
    sI = np.sqrt(m)
    dT = T - 25.0
    e = np.exp(-2.0 * sI)
    if dT.min() >= 0.0:                      # the hot side: constants at their 25 C values
        poly = m * (b0 + b1 * e) + (m * m) * c
    else:
        cold = np.minimum(dT, 0.0)
        poly = m * ((b0 + d0 * cold) + (b1 + d1 * cold) * e) + (m * m) * (c + dc * cold)
    s = 1.0 + kap * np.maximum(dT, 0.0)
    return 1.0 - aphi(T) * sI / (1.0 + 1.2 * sI) + s * poly


def _lnaw_pitzer(m, T, key):
    m = np.asarray(m, dtype=float)
    return -2.0 * m * phi_pitzer(m, T, key) * (MW / 1000.0)


def _lnaw_norrish(m, K):
    r = np.asarray(m, dtype=float) * MW / 1000.0
    xs = r / (1.0 + r)
    return np.log1p(-xs) + K * xs * xs


def _lnaw_vanlaar(m, A12, A21):
    """Water activity in a volatile(1)-water(2) mixture: ln aw = ln x_w + ln g_w."""
    r = np.asarray(m, dtype=float) * MW / 1000.0
    x1 = r / (1.0 + r)
    x2 = 1.0 - x1
    d = A12 * x1 + A21 * x2
    return np.log(x2) + A21 * (A12 * x1 / d) ** 2


def _lnaw_ideal(m):
    return -np.log1p(np.asarray(m, dtype=float) * MW / 1000.0)


_MMAX = {"nacl": 30.0, "kcl": 25.0}          # polynomial range; beyond it the table clamps
_TROWS = np.array([-25.0, 0.0, 25.0, 50.0, 75.0, 100.0, 110.0, 130.0])


def _binary(key):
    """(kind, params) of one water-activity model key."""
    if key in _PITZ:
        return "pitzer", key
    if key in D.TABLES["norrish"]:
        return "norrish", D.row("norrish", key)
    if key in D.TABLES["vanlaar"]:
        return "vanlaar", D.row("vanlaar", key)
    return "ideal", None


def lnaw_binary(key, m, T=25.0):
    """ln aw of ONE solute alone at molality m (mol/kg water) and T (C): the
    exact forward function; no table."""
    kind, p = _binary(key)
    if kind == "pitzer":
        return _lnaw_pitzer(m, T, p)
    if kind == "norrish":
        return _lnaw_norrish(m, p)
    if kind == "vanlaar":
        return _lnaw_vanlaar(m, *p)
    return _lnaw_ideal(m)


def has_model(key):
    """True when the key has a real (non-ideal) water-activity model."""
    return _binary(key)[0] != "ideal"


# Tables for ZSR: ln aw against molality on a geometric grid, inverted by np.interp.
_TABCACHE = {}


def _grid(key):
    mmax = _MMAX.get(key, 300.0)
    return np.concatenate([[0.0], np.geomspace(1e-4, mmax, 2000)])


def _table(key):
    """(negla rows, m grid, T rows or None) for one model key; built once."""
    t = _TABCACHE.get(key)
    if t is None:
        m = _grid(key)
        if _binary(key)[0] == "pitzer":
            rows = np.array([-lnaw_binary(key, m, Tr) for Tr in _TROWS])
            t = (rows, m, _TROWS)
        else:
            t = (np.array([-lnaw_binary(key, m, 25.0)]), m, None)
        _TABCACHE[key] = t
    return t


def _m0(key, la, T):
    """Molality of solute `key` ALONE that has ln aw = la (the binary curve
    inverted); la and T are 1-D arrays of one length."""
    if _binary(key)[0] == "ideal":
        return np.expm1(-la) * 1000.0 / MW
    rows, m, tr = _table(key)
    if tr is None:
        return np.interp(-la, rows[0], m)
    i = np.clip(np.searchsorted(tr, T) - 1, 0, len(tr) - 2)
    w = np.clip((T - tr[i]) / (tr[i + 1] - tr[i]), 0.0, 1.0)
    out = np.empty_like(la)
    for iu in np.unique(i):
        sel = i == iu
        lo = np.interp(-la[sel], rows[iu], m)
        hi = np.interp(-la[sel], rows[iu + 1], m)
        out[sel] = (1.0 - w[sel]) * lo + w[sel] * hi
    return out


# -- the solute bookkeeping, cached per registry revision --------------------------
_SETUP = {}


def _setup():
    rev = D.revision()
    s = _SETUP.get("s")
    if s is not None and s["rev"] == rev:
        return s
    idx = [i for i, n in enumerate(D.ORDER) if D.SPECIES[n].aq]
    keys = []
    for i in idx:
        k = D.SPECIES[D.ORDER[i]].model
        if k not in keys:
            keys.append(k)
    G = np.zeros((len(idx), len(keys)))
    for j, i in enumerate(idx):
        sp = D.SPECIES[D.ORDER[i]]
        G[j, keys.index(sp.model)] = sp.nu if sp.model == "ideal" else 1.0
    s = {"rev": rev, "idx": np.array(idx, dtype=int), "keys": keys, "G": G,
         "invM": 1.0 / np.array([D.SPECIES[D.ORDER[i]].M for i in idx]) if idx else np.zeros(0),
         "iw": D.INDEX["h2o"]}
    _SETUP["s"] = s
    return s


def _lead_flat(comp, T):
    """comp (...,S) and T broadcast to flat (N,S), (N,) plus the lead shape."""
    lead = comp.shape[:-1]
    T = np.asarray(T, dtype=float)
    if T.shape != lead:
        lead = np.broadcast_shapes(lead, T.shape)
        T = np.broadcast_to(T, lead)
        comp = np.broadcast_to(comp, lead + (comp.shape[-1],))
    return comp.reshape(-1, comp.shape[-1]), np.ascontiguousarray(T).reshape(-1), lead


_ZSR_ITERS = 36


def _zsr(Gs, Ts, keys):
    """ZSR: find ln aw with sum_i m_i / m0_i(ln aw) = 1. Gs (n,g) molalities."""
    n = Gs.shape[0]
    lo = np.full(n, -8.0)
    hi = np.full(n, -1e-12)
    act = [j for j in range(Gs.shape[1]) if np.any(Gs[:, j] > 0.0)]
    for _ in range(_ZSR_ITERS):
        mid = 0.5 * (lo + hi)
        f = -1.0
        for j in act:
            f = f + Gs[:, j] / np.maximum(_m0(keys[j], mid, Ts), 1e-300)
        pos = f > 0.0
        hi = np.where(pos, mid, hi)
        lo = np.where(pos, lo, mid)
    return 0.5 * (lo + hi)


_SETUPC = {}


def _setup_for(cols):
    """The solute bookkeeping for a column subset `cols` (global species indices,
    ascending): positions of the solutes and of water inside the subset."""
    key = (D.revision(), cols.tobytes())
    t = _SETUPC.get(key)
    if t is None:
        s = _setup()
        pos = {int(g): j for j, g in enumerate(cols)}
        rows = [r for r, g in enumerate(s["idx"]) if int(g) in pos]
        t = dict(s)
        t["idx"] = np.array([pos[int(s["idx"][r])] for r in rows], dtype=int)
        t["G"] = s["G"][rows]
        t["invM"] = s["invM"][rows]
        t["iw"] = pos.get(s["iw"], -1)
        _SETUPC[key] = t
    return t


class Grp(tuple):
    """Grouped molalities of flat cells: (setup, Gm (N,g), has (N,g), dry (N,), W (N,),
    plan). plan is ('none',), ('single', j) or ('general',): decided once, because
    the boiling fixed point calls ln aw several times on the same grouping."""
    __slots__ = ()

    def scaled(self, k, cap=None):
        """The same cells with their water scaled by 1/k (molality goes as 1/water).
        A trial state in a step solve can be absurdly concentrated (water almost
        gone), so each group is held at the top of its model's range and, if `cap`
        (N,g) is given, at that molality (the saturation a crystal rule would leave);
        the state a step returns is always inside both."""
        if self[1].shape[1] == 0:
            return self
        top = self[0].get("mtop")
        if top is None:
            top = self[0]["mtop"] = np.array([_MMAX.get(key, 3000.0) for key in self[0]["keys"]])
        if cap is not None:
            top = np.minimum(top, cap)
        return Grp((self[0], np.minimum(self[1] * k[:, None], top), self[2], self[3], self[4], self[5]))

    def take(self, sel):
        return Grp((self[0], self[1][sel], self[2][sel], self[3][sel], self[4][sel], self[5]))


def _groups(c, cols=None):
    """Molality of each model group for flat cells c (N,K): the part of ln aw that
    does not depend on T. `cols` names the species columns c holds (None = all)."""
    s = _setup() if cols is None else _setup_for(cols)
    N = c.shape[0]
    W = c[:, s["iw"]] if s["iw"] >= 0 else np.zeros(N)
    if not len(s["idx"]):
        return Grp((s, np.zeros((N, 0)), np.zeros((N, 0), dtype=bool), np.zeros(N, dtype=bool), W, ("none",)))
    idx = s["idx"]
    a, b = int(idx[0]), int(idx[-1]) + 1
    sub = c[:, a:b] if b - a == len(idx) else c[:, idx]
    Gm = (sub * s["invM"]) @ s["G"]
    has = Gm > 0.0
    ng = has.any(axis=0)
    dry = has.any(axis=1) & (W <= 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        Gm = Gm * (1000.0 / np.where(W > 0.0, W, 1.0))[:, None]
    if not ng.any():
        plan = ("none",)
    elif ng.sum() == 1 and has.sum(axis=1).max() <= 1:
        plan = ("single", int(np.argmax(ng)))
    else:
        plan = ("general",)
    return Grp((s, Gm, has, dry, W, plan))


def _lnaw_g(grp, Tf):
    """ln aw from the grouped molalities at temperatures Tf (flat)."""
    s, Gm, has, dry, _W, plan = grp
    N = Gm.shape[0]
    if plan[0] == "none":
        out = np.zeros(N)
    elif plan[0] == "single":
        out = lnaw_binary(s["keys"][plan[1]], Gm[:, plan[1]], Tf)   # a zero molality gives exactly 0.0
    else:
        nact = has.sum(axis=1)
        out = np.zeros(N)
        one = (nact == 1) & ~dry
        for j, k in enumerate(s["keys"]):
            sel = one & has[:, j]
            if sel.any():
                out[sel] = lnaw_binary(k, Gm[sel, j], Tf[sel])
        many = (nact >= 2) & ~dry
        if many.any():
            out[many] = _zsr(Gm[many], Tf[many], s["keys"])
    if dry.any():
        out = np.where(dry, -np.inf, out)
    return out


def ln_aw(comp, T=25.0):
    """ln of the activity of water. comp: dissolved grams (species on the last
    axis); T in C. Crystal grams never enter. One solute: its exact binary curve.
    Several: ZSR. No clip at saturation: a supersaturated syrup keeps lowering aw."""
    comp = prep(comp)
    c, Tf, lead = _lead_flat(comp, T)
    return _lnaw_g(_groups(c), Tf).reshape(lead)


def aw(comp, T=25.0):
    """Activity of water (0..1)."""
    return np.exp(ln_aw(comp, T))


def has_solute(comp, T=25.0):
    return ln_aw(comp, T) != 0.0


# -- boiling ------------------------------------------------------------------------
_BOIL_ITERS = 3


def tb_from_groups(grp, Pf, t0, scale=None, cap=None):
    """The 3-step fixed point T = tsat(P / aw(T)) from grouped molalities (flat).
    `scale` multiplies the molalities (they go as 1/water, so removing water from a
    cell is a scale and not a new grouping). Returns (T, ln aw at the answer)."""
    if scale is not None:
        grp = grp.scaled(scale, cap)
    t = t0
    la = np.zeros(t0.shape)
    for _ in range(_BOIL_ITERS):
        la = _lnaw_g(grp, t)
        t = tsat_water(Pf / np.exp(la))
    return t, la


def _boil_core(comp, P):
    """(T boil, T0 pure water, ln aw at the answer, lead shape) for the 3-step fixed point."""
    comp = prep(comp)
    P = np.asarray(P, dtype=float)
    lead = np.broadcast_shapes(comp.shape[:-1], P.shape)
    c = np.broadcast_to(comp, lead + comp.shape[-1:]).reshape(-1, comp.shape[-1])
    Pf = np.broadcast_to(P, lead).reshape(-1)
    t0 = tsat_water(Pf)
    t, la = tb_from_groups(_groups(c), Pf, t0)
    return t, t0, la, lead


def boiling_point(comp, P=D.P_ATM, tb_water=None):
    """Boiling point, C: aw(T) psat(T) = P, by T = tsat(P / aw(T)) three times.
    NaN where P / aw passes the critical pressure. `tb_water` sets the pure-water
    boiling point of the host (the sim's BOIL = 100): the answer is then
    tb_water + the rise."""
    t, t0, la, lead = _boil_core(comp, P)
    if tb_water is not None:
        t = tb_water + np.where(la != 0.0, t - t0, 0.0)
    return t.reshape(lead)


def boiling_rise(comp, P=D.P_ATM):
    """Rise above pure water at the same pressure, K. Pure water gives exactly 0.0."""
    t, t0, la, lead = _boil_core(comp, P)
    return np.where(la != 0.0, t - t0, 0.0).reshape(lead)


# -- freezing -----------------------------------------------------------------------
def _ethanol_ice_table():
    pts = np.array(D.row("ethanol_ice", "points"))
    return pts[:, 0], pts[:, 1]


def ethanol_freeze_c(w_pct):
    """Freezing point C of ethanol-water at w_pct wt% ethanol from the digitised
    ice line; linear extrapolation to wmax_extrapolated; NaN beyond (no data)."""
    w, t = _ethanol_ice_table()
    wmax = D.row("ethanol_ice", "wmax_extrapolated")
    w_pct = np.asarray(w_pct, dtype=float)
    slope = (t[-1] - t[-2]) / (w[-1] - w[-2])
    v = np.interp(w_pct, w, t)
    v = np.where(w_pct > w[-1], t[-1] + slope * (w_pct - w[-1]), v)
    return np.where((w_pct >= 0.0) & (w_pct <= wmax), v, np.nan)


def ethanol_wt_at(T):
    """Inverse of the ice line: wt% ethanol of the liquid in equilibrium with ice
    at T (C); NaN colder than the table."""
    w, t = _ethanol_ice_table()
    T = np.asarray(T, dtype=float)
    return np.where(T >= t[-1], np.interp(T, t[::-1], w[::-1]), np.nan)


def _only_ethanol(comp):
    """Mask of cells whose dissolved set is ethanol alone (the ice-line rule)."""
    s = _setup()
    others = [i for i in s["idx"] if D.ORDER[i] != "ethanol"]
    e = comp[..., D.INDEX["ethanol"]] > 0.0
    return e & ~np.any(comp[..., others] > 0.0, axis=-1) if others else e


_FREEZE_ITERS = 30


def freezing_drop(comp):
    """Freezing-point drop, K (positive), from ln aw(T) = ln_ice_ratio(T) by 30
    bisections on [-70, 0] C. Pure water gives exactly 0.0. Ethanol alone uses the
    digitised ice line (the Van Laar fit made at 78 C does not give the freezing
    curve); NaN where the line has no data."""
    comp = prep(comp)
    lead = comp.shape[:-1]
    c = comp.reshape(-1, comp.shape[-1])
    n = c.shape[0]
    grp = _groups(c)
    lo = np.full(n, -70.0)
    hi = np.zeros(n)
    for _ in range(_FREEZE_ITERS):
        mid = 0.5 * (lo + hi)
        g = _lnaw_g(grp, mid) - ln_ice_ratio(mid)
        # g = ln aw - ln ice: < 0 while the solution stays liquid (warmer than the
        # freezing point), > 0 once ice is the stable phase (colder).
        pos = g > 0.0
        lo = np.where(pos, mid, lo)
        hi = np.where(pos, hi, mid)
    t = 0.5 * (lo + hi)
    cold = np.full(n, -70.0)
    edge = _lnaw_g(grp, cold) - ln_ice_ratio(cold) < 0.0
    drop = np.where(edge, np.nan, -t)
    drop = np.where(_lnaw_g(grp, np.zeros(n)) == 0.0, 0.0, drop)
    et = _only_ethanol(c)
    if et.any():
        wt = 100.0 * c[:, D.INDEX["ethanol"]] / (c[:, D.INDEX["ethanol"]] + c[:, D.INDEX["h2o"]])
        drop = np.where(et, -ethanol_freeze_c(wt), drop)
    return drop.reshape(lead)


def freeze_point(comp):
    """The freezing point itself, C (= -freezing_drop). The name T3 calls."""
    return -freezing_drop(comp)


# -- osmosis ------------------------------------------------------------------------
def osmolality(comp, T=25.0):
    """Total osmolality, osmol per kg water: -ln aw * 1000 / Mw."""
    return -ln_aw(comp, T) * 1000.0 / MW


def tonicity(comp, T=25.0):
    """Osmolality times the share of solute particles the membrane holds back
    (sum sigma nu n / sum nu n): what water movement across a cell wall follows.
    Ethanol (sigma 0) gives about 0, NaCl equals its osmolality."""
    comp = prep(comp)
    s = _setup()
    idx = s["idx"]
    nu = np.array([D.SPECIES[D.ORDER[i]].nu for i in idx])
    sg = np.array([D.SPECIES[D.ORDER[i]].sigma for i in idx])
    n = comp[..., idx] * s["invM"] * nu
    tot = n.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(tot > 0.0, (n * sg).sum(axis=-1) / np.where(tot > 0.0, tot, 1.0), 1.0)
    return osmolality(comp, T) * share


def osmotic_pressure(comp, T=25.0):
    """Pi = -(R T / Vw) ln aw, Pa."""
    return -(D.R_GAS * (np.asarray(T, dtype=float) + _T0) / D.VW) * ln_aw(comp, T)


# -- density and heat capacity --------------------------------------------------------
def water_density(T):
    """Density of pure water, g/ml (Tanaka et al. 2001 form), T in C."""
    T = np.asarray(T, dtype=float)
    return 0.99997495 * (1.0 - (T - 3.983035) ** 2 * (T + 301.797) / (522528.9 * (T + 69.34881)))


_ED_W = np.array(D.row("ethanol_density", "mass_pct"))
_ED_R = np.array(D.row("ethanol_density", "rho"))
_RHO_ET = float(_ED_R[-1])


def _ethanol_rho(wpct):
    return np.interp(wpct, _ED_W, _ED_R)


def density(comp, T=20.0):
    """Mixture density, g/ml. Ethanol-water is NOT volume additive (the CRC table
    is used); salts and sugars by Masson V = V0 + S sqrt(m); the rest additive on
    their pure density."""
    comp = prep(comp)
    T = np.asarray(T, dtype=float)
    mass = comp.sum(axis=-1)
    W = comp[..., D.INDEX["h2o"]]
    E = comp[..., D.INDEX["ethanol"]]
    rw = water_density(T)
    pair = W + E
    with np.errstate(invalid="ignore", divide="ignore"):
        wpct = np.where(pair > 0.0, 100.0 * E / np.where(pair > 0.0, pair, 1.0), 0.0)
        r20 = _ethanol_rho(wpct)
        # temperature: scale by the water density ratio (u: the CRC table is at 20 C)
        vpair = np.where(E > 0.0, pair / (r20 * rw / water_density(20.0)), W / rw)
        kg = np.where(W > 0.0, W / 1000.0, np.nan)
        V = vpair
        for name in D.ORDER:
            if name in ("h2o", "ethanol"):
                continue
            g = comp[..., D.INDEX[name]]
            if not np.any(g > 0.0):
                continue
            sp = D.SPECIES[name]
            ms = D.TABLES["masson"].get(name)
            if ms is not None:
                V0, S = ms[0]
                mol = g / sp.M
                m = mol / kg
                V = V + np.where(g > 0.0, mol * (V0 + S * np.sqrt(np.where(g > 0.0, m, 0.0))), 0.0)
            else:
                V = V + g / sp.rho
        return mass / V


def heat_capacity(comp, ice_g=0.0):
    """Heat capacity, J/K, mass weighted (water 4.18, the ice share 2.09, ethanol
    2.44, ...). `ice_g` is how much of the water is ice."""
    comp = prep(comp)
    cp = np.array([D.SPECIES[n].cp for n in D.ORDER])
    c = (comp * cp).sum(axis=-1)
    return c - np.asarray(ice_g, dtype=float) * (D.CP_WATER - D.CP_ICE)


# -- ethanol by volume ----------------------------------------------------------------
_W_GRID = np.linspace(0.0, 100.0, 2001)
_ABV_GRID = _W_GRID * _ethanol_rho(_W_GRID) / _RHO_ET


def abv_of(comp):
    """Alcohol by volume, percent, of an ethanol-water mixture (grams in)."""
    comp = prep(comp)
    E = comp[..., D.INDEX["ethanol"]]
    pair = E + comp[..., D.INDEX["h2o"]]
    with np.errstate(invalid="ignore", divide="ignore"):
        w = np.where(pair > 0.0, 100.0 * E / np.where(pair > 0.0, pair, 1.0), 0.0)
    return w * _ethanol_rho(w) / _RHO_ET


def abv_to_grams(abv_vol, total_g):
    """Grams of ethanol in `total_g` grams of mixture at `abv_vol` percent by volume."""
    w = np.interp(np.asarray(abv_vol, dtype=float), _ABV_GRID, _W_GRID)
    return w / 100.0 * np.asarray(total_g, dtype=float)


# -- closed forms for the scheduler (T5) -------------------------------------------------
def _with_water(comp, W):
    out = np.array(comp, dtype=float)
    out[..., D.INDEX["h2o"]] = W
    return out


def _water_bisect(comp, target, fn, increasing, iters=90):
    """Find the water grams W* where fn(comp with W*) = target (geometric bisection)."""
    comp = prep(comp)
    W0 = comp[..., D.INDEX["h2o"]]
    sol = comp.sum(axis=-1) - W0
    lo = np.maximum(sol, 1e-9) * 1e-6
    hi = np.maximum(W0, np.maximum(sol, 1e-9)) * 1e6
    for _ in range(iters):
        mid = np.sqrt(lo * hi)
        v = fn(_with_water(comp, mid))
        above = (v > target) if increasing else (v < target)
        hi = np.where(above, mid, hi)
        lo = np.where(above, lo, mid)
    return np.sqrt(lo * hi), W0


def water_for_aw(comp, aw_target, T=25.0):
    """Grams of water to REMOVE (negative: add) so that aw reaches aw_target at T."""
    ws, w0 = _water_bisect(comp, np.log(aw_target), lambda c: ln_aw(c, T), increasing=True)
    return w0 - ws


def water_for_boiling_rise(comp, rise, P=D.P_ATM):
    """Grams of water to REMOVE (negative: add) so that the boiling rise reaches
    `rise` K at pressure P: the closed form for T5's boil-down skip."""
    ws, w0 = _water_bisect(comp, rise, lambda c: boiling_rise(c, P), increasing=False)
    return w0 - ws


# -- what the model covers --------------------------------------------------------------
def coverage(comp, T=None):
    """Which models were used, and what is extrapolated, for the species present
    in `comp`: {'models': {species: model}, 'flags': sorted notes}."""
    comp = prep(comp)
    s = _setup()
    present = np.any(comp.reshape(-1, comp.shape[-1]) > 0.0, axis=0)
    models = {}
    flags = set()
    for i in s["idx"]:
        name = D.ORDER[i]
        if not present[i]:
            continue
        sp = D.SPECIES[name]
        kind = _binary(sp.model)[0]
        models[name] = kind
        if kind == "ideal":
            flags.add(f"{name}: ideal (no real model)")
        if kind == "vanlaar":
            flags.add(f"{name}: Van Laar valid 20-100 C only")
            if T is not None:
                Ta = np.asarray(T, dtype=float)
                if np.any((Ta < 20.0) | (Ta > 100.0)):
                    flags.add(f"{name}: extrapolated")
        if kind == "pitzer" and T is not None and np.any(np.asarray(T) > 130.0):
            flags.add(f"{name}: extrapolated above 130 C")
    nsol = sum(1 for n in models)
    if nsol >= 2:
        flags.add("no coupled step (several solutes): use freeze_split and sequential phase_step")
    if present[D.INDEX["ethanol"]]:
        flags.add("freezing: digitised ice line above 33.4 wt% extrapolated, none above 40")
    return {"models": models, "flags": sorted(flags)}
