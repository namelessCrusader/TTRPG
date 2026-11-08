"""PHASES, ALLOYS, SLAGS, EUTECTICS AND THE ONE COUPLED STEP (wave 13, T4, law L14).

A SYSTEM is a species pair (A, B) with w = the mass fraction of B and a rule:
  "pure"      zero width (a pure metal, ice): a step at Tm. Tm and the latent heat
              of tin, lead and iron are READ from materials.MELT, never retyped.
  "linear"    bulk solidus and liquidus digitised; f is linear between (flagged: no
              composition split)
  "eutectic"  two liquidus branches that meet at (w_E, T_E), solid-solubility lines
              w_alpha(T) and w_beta(T), the LEVER rule between. The aqueous systems
              (ice line against solubility line) are the same structure with pure
              solids (w_alpha = 0, w_beta = 1) and are built at first use from the
              water-activity model: the eutectic is COMPUTED, not typed.
  "ice_line"  an ice line only (ethanol-water, digitised): no right branch, no pin.
Pure numpy, no World. T in C, grams, J.

THE ONE COUPLED SOLVER, phase_step: from a state, a temperature and the heat
capacity C of the cell it solves the new phase state and the new temperature
together, for melting, freezing and dissolving, in both directions. The cell's
energy is  H = C T + Lambda,  Lambda = the latent heat held by the liquid:
  aqueous (pure solids)  Lambda = (ice melted) Lfus(T) + (salt dissolved) Hsol
  alloys                 Lambda = f m L  (L mass weighted from the pure latents)
and C is the heat capacity with every gram of water counted as ICE: with Lfus(T) =
333.55 + 2.09 T (Kirchhoff) that already carries the 4.18 against 2.09 of the liquid
water, so counting the liquid's cp as well would count it twice. phase_step takes
the cell's current C and removes that part itself.
f_eq(T) rises with T, so H(T) = C T + Lambda(T, f_eq(T)) is increasing and the root
is unique. Closed form for pure and linear systems; at most 60 bisections for lever
systems; if the root would be below T_E the cell sits AT T_E (a pin) and f follows
from the energy equation, in [0, f_E].
"""
import numpy as np

from . import mixdata as D
from . import mixwater as W
from . import mixsol as S
from .mixwater import prep

_IW = D.INDEX["h2o"]
_T_ITERS = 60


# -- the System record -----------------------------------------------------------------
class System:
    """Plain record; see the module docstring. w is a fraction here (0..1)."""

    def __init__(self, name, rule, A, B=None, **kw):
        self.name = name
        self.rule = rule
        self.A = A
        self.B = B
        self.pin = False
        self.pure_solids = False
        self.notes = ()
        self.__dict__.update(kw)

    # -- bulk quantities
    def Lbulk(self, w0):
        """Latent heat J/g of the bulk (mass weighted from the pure latents)."""
        return (1.0 - w0) * self.LAc + w0 * self.LBc

    def valid(self, w0):
        if self.rule == "pure":
            return np.ones(np.shape(w0), dtype=bool)
        if self.rule == "linear":
            return (w0 >= self.w[0]) & (w0 <= self.w[-1])
        top = self.wr[-1] if self.wr is not None else self.wE
        return (w0 >= self.wl[0]) & (w0 <= top)

    # -- liquidus and liquid composition (eutectic and ice_line)
    def T_liq(self, w0):
        if self.rule == "pure":
            return np.full(np.shape(w0), self.Tm)
        if self.rule == "linear":
            return np.where(self.valid(w0), np.interp(w0, self.w, self.Tl), np.nan)
        tl = np.interp(w0, self.wl, self.tl)
        if self.wr is None:
            out = np.where(w0 <= self.wE, tl, np.nan)
        else:
            out = np.where(w0 <= self.wE, tl, np.interp(w0, self.wr, self.tr))
        return np.where(self.valid(w0), out, np.nan)

    def T_sol(self, w0):
        if self.rule == "pure":
            return np.full(np.shape(w0), self.Tm)
        if self.rule == "linear":
            return np.where(self.valid(w0), np.interp(w0, self.w, self.Ts), np.nan)
        if self.rule == "ice_line":
            return np.full(np.shape(w0), np.nan)
        # first liquid: the eutectic if the alloy reaches it, else the solidus curve
        wa = self.alpha(np.full(np.shape(w0), self.TE))
        wb = self.beta(np.full(np.shape(w0), self.TE))
        reaches = np.where(w0 <= self.wE, w0 > wa, w0 < wb)
        return np.where(self.valid(w0), np.where(reaches, self.TE, self._solidus_T(w0)), np.nan)

    def _solidus_T(self, w0):
        """Where an alloy that does NOT reach the eutectic starts to melt: the
        temperature at which the solubility line equals the bulk composition."""
        out = np.full(np.shape(w0), np.nan)
        Ts = np.linspace(self.TE, max(self.tl[0], self.tr[-1] if self.wr is not None else 0.0), 400)
        wa = self.alpha(Ts)
        left = w0 <= self.wE
        # alpha falls with T: find T where wa == w0
        out = np.where(left, np.interp(w0, wa[::-1], Ts[::-1]), out)
        if self.wr is not None:
            wb = self.beta(Ts)
            out = np.where(~left, np.interp(w0, wb, Ts), out)
        return out

    def wL_of_T(self, T, left):
        """Liquid composition on the liquidus at T (clamped to the table ends)."""
        wl = np.interp(T, self.tl[::-1], self.wl[::-1])
        if self.wr is None:
            return wl
        return np.where(left, wl, np.interp(T, self.tr, self.wr))

    def f_eq(self, w0, T):
        """Equilibrium liquid mass fraction. NaN outside the table range (never clamped)."""
        w0 = np.asarray(w0, dtype=float)
        T = np.asarray(T, dtype=float)
        if self.rule == "pure":
            return np.where(T >= self.Tm, 1.0, 0.0) + np.zeros(np.shape(w0))
        if self.rule == "linear":
            ts, tl = self.T_sol(w0), self.T_liq(w0)
            with np.errstate(invalid="ignore", divide="ignore"):
                f = np.clip((T - ts) / np.where(tl > ts, tl - ts, 1.0), 0.0, 1.0)
            f = np.where(tl > ts, f, np.where(T >= tl, 1.0, 0.0))
            return np.where(self.valid(w0), f, np.nan)
        left = w0 <= self.wE
        tl = self.T_liq(w0)
        wlq = self.wL_of_T(T, left)
        wS = np.where(left, self.alpha(T), self.beta(T))
        with np.errstate(invalid="ignore", divide="ignore"):
            num = np.where(left, w0 - wS, wS - w0)
            den = np.where(left, wlq - wS, wS - wlq)
            f = np.clip(num / np.where(den != 0.0, den, 1.0), 0.0, 1.0)
        f = np.where(T >= tl, 1.0, f)
        if self.pin:
            f = np.where(T < self.TE, 0.0, f)
        else:
            f = np.where(T < self.Tlo, np.nan, f)
        return np.where(self.valid(w0) & np.isfinite(tl), f, np.nan)

    def f_pin_top(self, w0):
        """Liquid fraction just at T_E (the top of the pin)."""
        w0 = np.asarray(w0, dtype=float)
        left = w0 <= self.wE
        wa = self.alpha(np.full(w0.shape, self.TE))
        wb = self.beta(np.full(w0.shape, self.TE))
        with np.errstate(invalid="ignore", divide="ignore"):
            fl = (w0 - wa) / (self.wE - wa)
            fr = (wb - w0) / (wb - self.wE)
        return np.clip(np.where(left, fl, fr), 0.0, 1.0)

    def liq_w(self, w0, T, f):
        """Composition of the liquid phase: the bulk once everything is liquid,
        else the liquidus at T (w_E on the pin)."""
        left = w0 <= self.wE
        return np.where(f >= 1.0, w0, self.wL_of_T(T, left))

    # -- latent heat held by the liquid
    def LA_of(self, T):
        return self.LAfn(T)

    def solids(self, A, B, w0, T, f):
        """(A_s, B_s) the grams of pure solid A and B at T for liquid fraction f."""
        m = A + B
        wl = self.liq_w(w0, T, f)
        liq = f * m
        return np.clip(A - liq * (1.0 - wl), 0.0, A), np.clip(B - liq * wl, 0.0, B)

    def lam(self, A, B, w0, T, f, solids=None):
        """Latent heat content Lambda (J) at T for liquid fraction f."""
        if self.pure_solids:
            a_s, b_s = solids if solids is not None else self.solids(A, B, w0, T, f)
            return (A - a_s) * self.LA_of(T) + (B - b_s) * self.HB
        return f * (A + B) * self.Lbulk(w0)

    def L_pin(self, w0):
        """Latent heat J per gram of eutectic liquid at the pin."""
        if self.pure_solids:
            return (1.0 - self.wE) * self.LA_of(self.TE) + self.wE * self.HB
        return self.Lbulk(w0)

    def latent_at(self, w0, T):
        """L_eff, J per gram of liquid formed at T."""
        if not self.pure_solids:
            return self.Lbulk(w0)
        left = w0 <= self.wE
        wl = self.wL_of_T(T, left)
        wl = np.where(T >= self.T_liq(w0), w0, wl)
        return (1.0 - wl) * self.LA_of(T) + wl * self.HB


def _tab(pts, i0, i1):
    a = np.array([[p[i0], p[i1]] for p in pts], dtype=float)
    return a[:, 0], a[:, 1]


def _fn_from_table(T_pts, w_pts):
    t = np.asarray(T_pts, dtype=float)
    w = np.asarray(w_pts, dtype=float)
    order = np.argsort(t)
    t, w = t[order], w[order]
    return lambda T: np.interp(np.asarray(T, dtype=float), t, w)


def _const(v):
    return lambda T: np.zeros_like(np.asarray(T, dtype=float)) + v


# -- builders ---------------------------------------------------------------------------
_SYS = {}


def _pure_table():
    """{species: (Tm, L)}: tin, lead, iron from materials.MELT; the rest from mixdata."""
    out = dict(D.pure_metal_points())
    for k, (v, _s) in D.TABLES["pure_melts"].items():
        out.setdefault(k, v)
    return out


def _fill(pts, ends):
    """Replace a None temperature by the pure end value."""
    out = []
    for p in pts:
        if len(p) == 2 and p[1] is None:
            p = (p[0], ends[0] if p[0] == 0.0 else ends[1])
        out.append(p)
    return out


def _build_pure(name, sp, Tm, L):
    return System(name, "pure", sp, None, Tm=float(Tm), LAc=float(L), LBc=float(L), wE=1.0, TE=float(Tm))


def _build_pbsn(pure):
    d = D.TABLES
    TE, wE = d["sys_pbsn"]["TE_wE"][0]
    left = _fill(d["sys_pbsn"]["left"][0], (pure["pb"][0], pure["sn"][0]))
    right = _fill(d["sys_pbsn"]["right"][0], (pure["pb"][0], pure["sn"][0]))
    right = [(w, pure["sn"][0]) if w == 100.0 else (w, t) for w, t in right]
    wl, tl = _tab(left, 0, 1)
    wr, tr = _tab(right, 0, 1)
    at, aw_ = _tab(d["sys_pbsn"]["alpha"][0], 0, 1)
    bt, bw_ = _tab(d["sys_pbsn"]["beta"][0], 0, 1)
    return System("pb-sn", "eutectic", "pb", "sn", pin=True, TE=TE, wE=wE / 100.0,
                  wl=wl / 100.0, tl=tl, wr=wr / 100.0, tr=tr,
                  alpha=_fn_from_table(at, aw_ / 100.0), beta=_fn_from_table(bt, bw_ / 100.0),
                  LAc=pure["pb"][1], LBc=pure["sn"][1], Tlo=-np.inf)


def _lat(pure, sp, fallback):
    return pure[sp][1] if sp in pure else fallback


def _build_linear(name, A, B, pts, pure, LA=None, LB=None):
    pts = [tuple(p) for p in pts]
    w = np.array([p[0] for p in pts]) / 100.0
    Ts = np.array([p[1] for p in pts], dtype=float)
    Tl = np.array([p[-1] for p in pts], dtype=float)
    la = _lat(pure, A, 300.0) if LA is None else LA
    lb = _lat(pure, B, la) if LB is None else LB
    return System(name, "linear", A, B, w=w, Ts=Ts, Tl=Tl, LAc=la, LBc=lb,
                  notes=("no composition split: bulk-only (u)",))


def _lever_alpha(sol_pts, te):
    """Solid composition w_alpha(T) from a solidus curve (wt%, C): the inverse of the
    strictly sloped pieces; the flat pieces (peritectic and eutectic) are skipped."""
    pts = sorted(sol_pts)
    segs = [(a, b) for a, b in zip(pts[:-1], pts[1:]) if abs(a[1] - b[1]) > 1e-9]
    def fn(T):
        T = np.asarray(T, dtype=float)
        out = np.full(T.shape, np.nan)
        for a, b in segs:
            lo, hi = sorted((a[1], b[1]))
            sel = (T >= lo) & (T <= hi)
            val = a[0] + (b[0] - a[0]) * (T - a[1]) / (b[1] - a[1])
            out = np.where(sel & np.isnan(out), val, out)
        tmin = min(p[1] for p in sol_pts)
        tmax = max(p[1] for p in sol_pts)
        wmax = max(p[0] for p in sol_pts)
        out = np.where(np.isnan(out) & (T < tmin), wmax, out)
        out = np.where(np.isnan(out) & (T > tmax), 0.0, out)
        return np.where(np.isnan(out), 0.0, out) / 100.0
    return fn


def _build_fec(pure):
    d = D.TABLES["sys_fec"]
    fe_tm = pure["fe"][0]
    liq = _fill(d["liquidus"][0], (fe_tm, fe_tm))
    sol = _fill(d["solidus"][0], (fe_tm, fe_tm))
    TE, wE, wA = d["TE_wE_wA"][0]
    wl, tl = _tab(liq, 0, 1)
    alpha = _lever_alpha(sol, TE)
    return System("fe-c", "eutectic", "fe", "c", pin=True, TE=TE, wE=wE / 100.0, wl=wl / 100.0, tl=tl,
                  wr=None, tr=None, alpha=alpha, beta=_const(1.0), LAc=pure["fe"][1], LBc=pure["fe"][1],
                  Tlo=-np.inf, notes=("valid 0 to 4.3 wt% C; peritectic simplified (u)",))


def _ice_w(solute, T):
    """Liquid solute mass fraction on the ice line at T (C): ln aw(m, T) = ln_ice_ratio(T)."""
    sp = D.SPECIES[solute]
    nu = sp.nu if sp.model == "ideal" else 1.0
    T = np.asarray(T, dtype=float)
    target = W.ln_ice_ratio(T)
    lo = np.full(T.shape, 1e-9)
    hi = np.full(T.shape, 40.0)
    for _ in range(52):
        mid = 0.5 * (lo + hi)
        f = W.lnaw_binary(sp.model, mid * nu, T) - target
        lo = np.where(f > 0.0, mid, lo)
        hi = np.where(f > 0.0, hi, mid)
    m = 0.5 * (lo + hi)
    return m * sp.M / (1000.0 + m * sp.M)


def _sat_w(solute, T):
    s = S.solubility_water(solute, T)
    return s / (100.0 + s)


def _build_aqueous(solute, name=None):
    sp = D.SPECIES[solute]
    grid = np.linspace(-70.0, 0.0, 2801)               # 0.025 K
    g = _ice_w(solute, grid) - _sat_w(solute, grid)     # < 0 near 0 C, > 0 once the ice line is saltier
    k = int(np.nonzero(g[:-1] > 0.0)[0].max()) if np.any(g[:-1] > 0.0) else -1
    if k < 0 or k >= len(grid) - 1 or not (g[k] > 0.0 >= g[k + 1]):
        raise ValueError(f"no eutectic for {solute} between -70 and 0 C")
    TE = float(grid[k] + (grid[k + 1] - grid[k]) * g[k] / (g[k] - g[k + 1]))
    wE = float(_sat_w(solute, np.array([TE]))[0])
    Tg = np.linspace(0.0, TE, 90)
    wl = _ice_w(solute, Tg)
    wl[-1] = wE
    Tr = np.linspace(TE, 200.0, 160)
    wr = _sat_w(solute, Tr)
    wr[0] = wE
    hb = S.heat_of_solution(solute)
    return System(name or f"h2o-{solute}", "eutectic", "h2o", solute, pin=True, pure_solids=True,
                  TE=float(TE), wE=wE, wl=wl, tl=Tg, wr=wr, tr=Tr,
                  alpha=_const(0.0), beta=_const(1.0), LAfn=W.Lfus, HB=float(hb),
                  LAc=D.L_FUS0, LBc=float(hb), Tlo=-np.inf,
                  notes=("ice line from the water-activity model; solubility log-linear beyond 100 C",))


def _build_ethanol_ice():
    pts = np.array(D.row("ethanol_ice", "points"))
    wmax = D.row("ethanol_ice", "wmax_extrapolated")
    slope = (pts[-1, 1] - pts[-2, 1]) / (pts[-1, 0] - pts[-2, 0])
    wl = np.concatenate([pts[:, 0], [wmax]])
    tl = np.concatenate([pts[:, 1], [pts[-1, 1] + slope * (wmax - pts[-1, 0])]])
    # w as the mass fraction of ethanol in the liquid
    return System("h2o-ethanol", "ice_line", "h2o", "ethanol", pin=False, pure_solids=True,
                  TE=float(tl[-1]), wE=float(wl[-1]) / 100.0, wl=wl / 100.0, tl=tl, wr=None, tr=None,
                  alpha=_const(0.0), beta=_const(1.0), LAfn=W.Lfus, HB=0.0, LAc=D.L_FUS0, LBc=0.0,
                  Tlo=float(tl[-1]),
                  notes=("digitised ice line; extrapolated 33.4-40 wt%, none beyond",))


def _from_spec(name, spec, pure):
    r = spec["rule"]
    A, B = spec["A"], spec["B"]
    if r == "aqueous":
        return _build_aqueous(B, name)
    if r == "linear":
        return _build_linear(name, A, B, spec["points"], pure, spec.get("LA"), spec.get("LB"))
    if r == "eutectic":
        wl, tl = _tab(spec["left"], 0, 1)
        wr, tr = _tab(spec["right"], 0, 1)
        TE, wE = spec["TE"], spec["wE"]
        al = _fn_from_table(*zip(*[(t, w / 100.0) for t, w in spec["alpha"]])) if spec.get("alpha") else _const(0.0)
        be = _fn_from_table(*zip(*[(t, w / 100.0) for t, w in spec["beta"]])) if spec.get("beta") else _const(1.0)
        return System(name, "eutectic", A, B, pin=True, TE=TE, wE=wE / 100.0, wl=wl / 100.0, tl=tl,
                      wr=wr / 100.0, tr=tr, alpha=al, beta=be,
                      LAc=spec.get("LA", _lat(pure, A, 300.0)),
                      LBc=spec.get("LB", _lat(pure, B, _lat(pure, A, 300.0))), Tlo=-np.inf)
    # lever_pair
    wl, tl = _tab(spec["liquidus"], 0, 1)
    return System(name, "eutectic", A, B, pin=True, TE=spec["TE"], wE=spec["wE"] / 100.0, wl=wl / 100.0,
                  tl=tl, wr=None, tr=None, alpha=_lever_alpha(spec["solidus"], spec["TE"]), beta=_const(1.0),
                  LAc=spec.get("LA", _lat(pure, A, 300.0)), LBc=spec.get("LB", _lat(pure, A, 300.0)), Tlo=-np.inf)


def systems():
    """{name: System}, built at first use and again if the registry changes."""
    rev = D.revision()
    if _SYS.get("rev") == rev:
        return _SYS["all"]
    pure = _pure_table()
    allsys = {}
    for sp, (tm, L) in pure.items():
        if sp in D.SPECIES:
            allsys[sp] = _build_pure(sp, sp, tm, L)
    allsys["pb-sn"] = _build_pbsn(pure)
    allsys["cu-sn"] = _build_linear("cu-sn", "cu", "sn", D.row("sys_cusn", "points"), pure)
    allsys["cu-zn"] = _build_linear("cu-zn", "cu", "zn", D.row("sys_cuzn", "points"), pure)
    allsys["feo-sio2"] = _build_linear("feo-sio2", "feo", "sio2", D.row("sys_feosio2", "points"), pure)
    allsys["fe-c"] = _build_fec(pure)
    for solute in ("nacl", "kcl", "sucrose", "kno3"):
        allsys[f"h2o-{solute}"] = _build_aqueous(solute)
    allsys["h2o-ethanol"] = _build_ethanol_ice()
    for nm, spec in D.SYSTEM_SPECS.items():
        allsys[nm] = _from_spec(nm, spec, pure)
    _SYS["rev"] = rev
    _SYS["all"] = allsys
    return allsys


def get_system(name):
    return systems()[name]


def system_for(comp):
    """The system that governs `comp` (the species with grams above zero anywhere
    in the batch), or None: three or more active species, or a pair/single that
    has no row. Several solutes in water are handled by the host with sequential
    phase_step calls on (water, solute) pairs."""
    c = prep(comp).reshape(-1, D.n_species())
    act = frozenset(D.ORDER[i] for i in np.nonzero(c.any(axis=0))[0])
    S_ = systems()
    if len(act) == 1:
        (sp,) = tuple(act)
        return S_.get(sp)
    if len(act) == 2:
        for sysm in S_.values():
            if sysm.B is not None and frozenset((sysm.A, sysm.B)) == act:
                return sysm
    return None


def _pair(sysm, c):
    A = c[..., D.INDEX[sysm.A]]
    B = c[..., D.INDEX[sysm.B]] if sysm.B is not None else np.zeros_like(A)
    return A, B


def _w0(A, B):
    m = A + B
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(m > 0.0, B / np.where(m > 0.0, m, 1.0), 0.0)


def _need(comp, system):
    c = prep(comp)
    sysm = get_system(system) if isinstance(system, str) else (system or system_for(c))
    if sysm is None:
        raise ValueError("no phase system for this composition (three or more species, or no row)")
    return c, sysm


# -- state functions ----------------------------------------------------------------------
def liquid_fraction(comp, T, system=None):
    """Equilibrium liquid mass fraction at T (C); NaN outside a table's range."""
    c, sysm = _need(comp, system)
    A, B = _pair(sysm, c)
    return sysm.f_eq(_w0(A, B), T)


def solidus_liquidus(comp, system=None):
    """(T_solidus, T_liquidus) of the bulk, C (equal for a pure metal)."""
    c, sysm = _need(comp, system)
    A, B = _pair(sysm, c)
    w0 = _w0(A, B)
    return sysm.T_sol(w0), sysm.T_liq(w0)


def latent(comp, T, system=None):
    """L_eff, J per gram of liquid formed at T (mass weighted from the pure latents;
    aqueous: Lfus(T) on the water that melts plus the heat of solution)."""
    c, sysm = _need(comp, system)
    A, B = _pair(sysm, c)
    w0 = _w0(A, B)
    if sysm.rule == "pure":
        return np.full(w0.shape, sysm.LAc)
    if sysm.rule == "linear":
        return sysm.Lbulk(w0)
    return sysm.latent_at(w0, np.asarray(T, dtype=float))


def eutectic(name):
    """(T_E in C, w_E in wt percent B) of a system, by solute name ('nacl') or
    system name ('pb-sn'). The aqueous ones are computed from the ice line meeting
    the solubility line."""
    s_ = systems()
    sysm = s_.get(name) or s_.get(f"h2o-{name}")
    if sysm is None or sysm.rule not in ("eutectic",):
        raise KeyError(f"no eutectic for {name}")
    return float(sysm.TE), float(sysm.wE) * 100.0


def phase_assembly(comp, T, f=None, system=None):
    """(f_liquid, liquid_comp, solid_comp, crystal_comp) at T: the liquid and solid
    compositions, so a pour, decant or sweat can separate them (melting and
    rendering are separation laws). For aqueous systems solid_comp is the ice and
    crystal_comp the undissolved solute; for alloys solid_comp is the whole solid
    and crystal_comp is zero. `f` overrides the equilibrium fraction (the host's
    pinned state). The parts sum to the bulk."""
    c, sysm = _need(comp, system)
    A, B = _pair(sysm, c)
    w0 = _w0(A, B)
    T = np.asarray(T, dtype=float)
    fe = sysm.f_eq(w0, T) if f is None else np.asarray(f, dtype=float)
    if sysm.rule == "pure":
        wl = np.zeros_like(w0)
    elif sysm.rule == "linear":
        wl = w0
    else:
        wl = sysm.liq_w(w0, T, fe)
    m = A + B
    liq = fe * m
    liq_comp = np.zeros_like(c)
    ia = D.INDEX[sysm.A]
    liq_comp[..., ia] = liq * (1.0 - wl)
    if sysm.B is not None:
        liq_comp[..., D.INDEX[sysm.B]] = liq * wl
    crystal = np.zeros_like(c)
    if sysm.pure_solids and sysm.B is not None:
        liq_comp[..., ia] = np.minimum(liq_comp[..., ia], A)
        liq_comp[..., D.INDEX[sysm.B]] = np.minimum(liq_comp[..., D.INDEX[sysm.B]], B)
        crystal[..., D.INDEX[sysm.B]] = B - liq_comp[..., D.INDEX[sysm.B]]
        solid = np.zeros_like(c)
        solid[..., ia] = A - liq_comp[..., ia]
        return fe, liq_comp, solid, crystal
    solid = c - liq_comp
    return fe, liq_comp, solid, crystal


# -- the coupled step ------------------------------------------------------------------------
class PhaseOut(tuple):
    """(f_new, T_new, dE) with .solid_new (N,2) grams of pure solid A and B for
    aqueous systems, .pinned (True where the cell sits at T_E), .system."""


def _out(f, T, dE, solids, pinned, sysm):
    o = PhaseOut((f, T, dE))
    o.solid_new = solids
    o.pinned = pinned
    o.system = sysm
    return o


def _closed_form(Ts, Tl, m, L, C, T, lam_old):
    """pure (Ts == Tl) and linear systems: closed form. Returns (T_new, f_new)."""
    Ta = T + lam_old / C
    Tb = T + (lam_old - m * L) / C
    wid = Tl - Ts
    with np.errstate(invalid="ignore", divide="ignore"):
        wsafe = np.where(wid > 0.0, wid, 1.0)
        T_mush = (lam_old + C * T + m * L * Ts / wsafe) / (C + m * L / wsafe)
        f_mush = (T_mush - Ts) / wsafe
        f_pin = (lam_old - C * (Ts - T)) / (m * L)
    zero = ~(wid > 0.0)
    solid = Ta <= Ts
    liquid = (Tb >= Tl) & ~solid
    T_new = np.where(solid, Ta, np.where(liquid, Tb, np.where(zero, Ts, T_mush)))
    f_new = np.where(solid, 0.0, np.where(liquid, 1.0, np.where(zero, np.clip(f_pin, 0.0, 1.0), np.clip(f_mush, 0.0, 1.0))))
    return T_new, f_new


def _energy_C(C, A, solid_old, sysm):
    """Remove the liquid water's extra cp from the cell's C: the all-ice reference."""
    if not sysm.pure_solids:
        return C
    a_s = solid_old[..., 0]
    return C - (A - a_s) * (D.CP_WATER - D.CP_ICE)


def _solids_from_f(sysm, A, B, w0, f_old):
    """No solid_old given: the solid part is the PRIMARY solid of the bulk side
    (ice on the water-rich side, crystal on the salt-rich side)."""
    m = A + B
    solid = np.clip((1.0 - f_old) * m, 0.0, m)
    left = w0 <= sysm.wE
    a_s = np.where(left, np.minimum(solid, A), np.maximum(solid - B, 0.0))
    b_s = np.where(left, np.maximum(solid - A, 0.0), np.minimum(solid, B))
    return a_s, b_s


def phase_step(comp, f_old, T, C_JK, P=D.P_ATM, system=None, solid_old=None):
    """THE energy-coupled melt/freeze/dissolve solver. comp: grams of the system's
    species; f_old the liquid mass fraction of the old state; T (C); C_JK the cell's
    heat capacity (J/K). `solid_old` (aqueous systems) gives the grams of pure solid
    (A_s, B_s) of the old state, e.g. (100, 10) for 100 g ice plus 10 g salt crystals;
    without it the solid is taken as the primary solid of the bulk side.
    Returns PhaseOut (f_new, T_new, dE) with dE = C (T_new - T): the sensible
    energy change, equal to -(Lambda_new - Lambda_old), so
    C T + Lambda is the same before and after. P is accepted for symmetry with
    boil_step (no pressure term here). Runs in both directions; idempotent."""
    c, sysm = _need(comp, system)
    scalar = c.ndim == 1
    c = c.reshape(-1, c.shape[-1])
    A, B = _pair(sysm, c)
    N = A.shape[0]
    m = A + B
    w0 = _w0(A, B)
    Tf = np.broadcast_to(np.asarray(T, dtype=float), (N,)).astype(float)
    Cf = np.broadcast_to(np.asarray(C_JK, dtype=float), (N,)).astype(float)
    fo = np.broadcast_to(np.asarray(f_old, dtype=float), (N,)).astype(float)
    if sysm.pure_solids:
        if solid_old is None:
            a_s, b_s = _solids_from_f(sysm, A, B, w0, fo)
            sold = np.stack([a_s, b_s], axis=-1)
        else:
            sold = np.broadcast_to(np.asarray(solid_old, dtype=float), (N, 2)).astype(float)
        Cf = _energy_C(Cf, A, sold, sysm)
        lam_old = (A - sold[:, 0]) * sysm.LA_of(Tf) + (B - sold[:, 1]) * sysm.HB
    else:
        lam_old = fo * m * (sysm.Lbulk(w0) if sysm.rule != "pure" else sysm.LAc)

    if sysm.rule in ("pure", "linear"):
        if sysm.rule == "pure":
            Ts = Tl = np.full(N, sysm.Tm)
            L = np.full(N, sysm.LAc)
        else:
            Ts, Tl, L = sysm.T_sol(w0), sysm.T_liq(w0), sysm.Lbulk(w0)
        T_new, f_new = _closed_form(Ts, Tl, m, L, Cf, Tf, lam_old)
        pinned = (Ts == Tl) & (np.abs(T_new - Ts) < 1e-12) & (f_new > 0.0) & (f_new < 1.0)
        bad = ~sysm.valid(w0)
        T_new = np.where(bad, np.nan, T_new)
        f_new = np.where(bad, np.nan, f_new)
        res = _out(f_new, T_new, Cf * (T_new - Tf), None, pinned, sysm)
        return _squeeze(res, scalar)

    # eutectic and ice_line: pin, then bisection
    TE = sysm.TE
    ok = sysm.valid(w0)
    # if the old state is already the equilibrium one, change nothing
    f_here = sysm.f_eq(w0, Tf)
    lam_here = sysm.lam(A, B, w0, Tf, f_here)
    same = np.abs(lam_here - lam_old) <= 1e-12 * (np.abs(lam_old) + m + 1.0)
    pinned = np.zeros(N, dtype=bool)
    if sysm.pin:
        h0 = Cf * (TE - Tf) - lam_old
        fE = sysm.f_pin_top(w0)
        lamE = sysm.lam(A, B, w0, np.full(N, TE), fE)
        hE = Cf * (TE - Tf) + lamE - lam_old
        case1 = (h0 >= 0.0) & ~same
        case2 = ~case1 & (hE >= 0.0) & ~same & (h0 < 0.0)
    else:
        case1 = np.zeros(N, dtype=bool)
        case2 = np.zeros(N, dtype=bool)
    case3 = ~case1 & ~case2 & ~same
    T_new = Tf.copy()
    f_new = f_here.copy()
    if case1.any():
        T_new = np.where(case1, Tf + lam_old / Cf, T_new)
        f_new = np.where(case1, 0.0, f_new)
    if case2.any():
        Lp = sysm.L_pin(w0)
        with np.errstate(invalid="ignore", divide="ignore"):
            fp = (lam_old - Cf * (TE - Tf)) / (m * Lp)
        T_new = np.where(case2, TE, T_new)
        f_new = np.where(case2, np.clip(fp, 0.0, fE), f_new)
        pinned = case2
    if case3.any():
        lo = np.full(N, TE if sysm.pin else sysm.Tlo)
        hi = np.maximum(Tf, np.nan_to_num(sysm.T_liq(w0), nan=-np.inf))
        hi = np.maximum(hi, lo)
        for _ in range(_T_ITERS):
            mid = 0.5 * (lo + hi)
            fm = sysm.f_eq(w0, mid)
            h = Cf * (mid - Tf) + sysm.lam(A, B, w0, mid, fm) - lam_old
            up = h > 0.0
            hi = np.where(up, mid, hi)
            lo = np.where(up, lo, mid)
        t3 = 0.5 * (lo + hi)
        T_new = np.where(case3, t3, T_new)
        f_new = np.where(case3, sysm.f_eq(w0, t3), f_new)
        if not sysm.pin:
            # a root colder than the digitised table has no data
            below = case3 & (t3 <= sysm.Tlo + 1e-9)
            T_new = np.where(below, np.nan, T_new)
            f_new = np.where(below, np.nan, f_new)
    T_new = np.where(ok, T_new, np.nan)
    f_new = np.where(ok, f_new, np.nan)
    solids = None
    if sysm.pure_solids:
        a_s, b_s = sysm.solids(A, B, w0, T_new, f_new)
        solids = np.stack([a_s, b_s], axis=-1)
    res = _out(f_new, T_new, Cf * (T_new - Tf), solids, pinned, sysm)
    return _squeeze(res, scalar)


def _squeeze(res, scalar):
    if not scalar:
        return res
    o = PhaseOut((res[0][0], res[1][0], res[2][0]))
    o.solid_new = None if res.solid_new is None else res.solid_new[0]
    o.pinned = res.pinned[0] if np.ndim(res.pinned) else res.pinned
    o.system = res.system
    return o


def reference_heat_capacity(comp, C_JK, solid_old=None, system=None):
    """The all-ice reference heat capacity phase_step works with: the cell's C minus
    the extra cp (4.18 - 2.09) of the water that is liquid in the state C describes.
    solid_old = (A_s, B_s) grams of pure solid of that state (aqueous systems)."""
    c, sysm = _need(comp, system)
    A, _B = _pair(sysm, c)
    C = np.asarray(C_JK, dtype=float)
    if not sysm.pure_solids or solid_old is None:
        return C
    so = np.asarray(solid_old, dtype=float)
    return C - (A - so[..., 0]) * (D.CP_WATER - D.CP_ICE)


def energy_content(comp, f, T, C_ref, solids=None, system=None):
    """C_ref T + Lambda in J, the quantity phase_step keeps equal before and after.
    C_ref is the all-ice reference heat capacity (reference_heat_capacity); an
    aqueous cell also needs `solids` = (A_s, B_s) grams of pure solid of the state."""
    c, sysm = _need(comp, system)
    A, B = _pair(sysm, c)
    w0 = _w0(A, B)
    T = np.asarray(T, dtype=float)
    f = np.asarray(f, dtype=float)
    C = np.asarray(C_ref, dtype=float)
    if sysm.rule == "pure":
        return C * T + f * (A + B) * sysm.LAc
    if sysm.rule == "linear":
        return C * T + f * (A + B) * sysm.Lbulk(w0)
    if sysm.pure_solids:
        so = np.asarray(solids, dtype=float)
        return C * T + sysm.lam(A, B, w0, T, f, (so[..., 0], so[..., 1]))
    return C * T + sysm.lam(A, B, w0, T, f)


# thin names for the one function
melt_step = phase_step
freeze_step = phase_step
dissolve_step = phase_step


# -- freezing at a fixed temperature (several solutes) ------------------------------------
_FS_ITERS = 50


def freeze_split(comp, T):
    """(ice_g, liquid_g, liquid_comp) at a fixed T (C): the ice line by ZSR for any
    number of dissolved solutes, after the solubility split (what the water cannot
    hold stays crystal and is not in liquid_comp). Below the lowest eutectic of the
    solutes present nothing is liquid. Ethanol alone follows the digitised ice line
    (NaN beyond its data). Not energy coupled: with a heat balance use phase_step."""
    c = prep(comp)
    lead = c.shape[:-1]
    c = c.reshape(-1, c.shape[-1])
    N = c.shape[0]
    Tf = np.broadcast_to(np.asarray(T, dtype=float), (N,)).astype(float)
    dis, _cry = S.dissolved_split(c, Tf)
    Wt = c[:, _IW]
    grp_comp = dis.copy()
    s_ = systems()
    present = [n for n in D.ORDER if D.SPECIES[n].aq and n != "ethanol" and np.any(dis[:, D.INDEX[n]] > 0.0)]
    tes = [s_[f"h2o-{n}"].TE for n in present if f"h2o-{n}" in s_]
    TEmin = min(tes) if tes else -np.inf
    lo = np.zeros(N)
    hi = Wt.copy()
    target = W.ln_ice_ratio(Tf)

    def lnaw_with(Wl):
        g = grp_comp.copy()
        g[:, _IW] = Wl
        return W.ln_aw(g, Tf)

    for _ in range(_FS_ITERS):
        mid = 0.5 * (lo + hi)
        f = lnaw_with(np.maximum(mid, 1e-300)) - target
        # f > 0: the solution at `mid` liquid water is too dilute for ice -> less liquid water
        lo = np.where(f > 0.0, lo, mid)
        hi = np.where(f > 0.0, mid, hi)
    Wl = 0.5 * (lo + hi)
    Wl = np.where(lnaw_with(Wt) - target <= 0.0, Wt, Wl)     # all liquid when even the full water stays liquid
    Wl = np.where(Tf < TEmin, 0.0, Wl)
    nosolute = ~np.any(dis[:, [D.INDEX[n] for n in D.ORDER if D.SPECIES[n].aq]] > 0.0, axis=1)
    Wl = np.where(nosolute, np.where(Tf >= 0.0, Wt, 0.0), Wl)
    # ethanol alone
    et = W._only_ethanol(c)
    if et.any():
        E = c[:, D.INDEX["ethanol"]]
        wl = W.ethanol_wt_at(Tf) / 100.0
        with np.errstate(invalid="ignore", divide="ignore"):
            Wl_e = np.where(Tf >= W.ethanol_freeze_c(100.0 * E / (E + Wt)), Wt, E * (1.0 - wl) / wl)
        Wl = np.where(et, np.minimum(Wl_e, Wt), Wl)
    ice = Wt - Wl
    liq = grp_comp.copy()
    liq[:, _IW] = Wl
    below = Tf < TEmin
    liq[below] = 0.0
    shape = lead + (c.shape[-1],)
    liquid_g = liq.sum(axis=1)
    return ice.reshape(lead), liquid_g.reshape(lead), liq.reshape(shape)


# -- iron and carbon -----------------------------------------------------------------------
def fe_c_points(wt_c):
    """{'Ae1','Ae3','Acm','solidus','liquidus'} in C for wt% carbon in iron. Ae1 =
    727; Ae3 = 912 - 203 sqrt(C) below 0.77 (Andrews 1965; alloy terms u); Acm
    above 0.77 up to the eutectic 2.11 (linear to 1147, u). NaN where a line does
    not exist and above 4.3 wt% C (no clamping)."""
    C = np.asarray(wt_c, dtype=float)
    d = D.TABLES["sys_fec"]
    a1 = d["ae1"][0]
    c0, c1 = d["ae3"][0]
    (w0_, t0_), (w1_, t1_) = d["acm_pts"][0]
    ok = (C >= 0.0) & (C <= d["wmax"][0])
    ae3 = np.where(C < w0_, c0 - c1 * np.sqrt(np.maximum(C, 0.0)), np.nan)
    acm = np.where((C >= w0_) & (C <= w1_), t0_ + (t1_ - t0_) / (w1_ - w0_) * (C - w0_), np.nan)
    sysm = get_system("fe-c")
    w = C / 100.0
    liq = np.where(ok, sysm.T_liq(w), np.nan)
    sol = np.where(ok, sysm.T_sol(w), np.nan)
    return {"Ae1": np.where(ok, a1, np.nan), "Ae3": np.where(ok, ae3, np.nan),
            "Acm": np.where(ok, acm, np.nan), "solidus": sol, "liquidus": liq}
