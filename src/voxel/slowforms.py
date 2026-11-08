"""THE CLOSED-FORM LIBRARY (wave 13, T5, law L11).

A slow process (a rate that takes hours to years) is moved from now to later in
ONE step by a closed form, and a form never causes a consequence: it only moves
a state forward. This file is pure maths. It imports math and numpy and knows
nothing about a World.

TIME is the integer tick (TICK_S = 0.025 s, so 1 h = 144,000 ticks). A form
turns ticks into seconds itself. A DRIVE is an outside input given as data
(a constant, a sine, a table of steps, a sum) that can be integrated.

FORMS (kind -> what it is). A kind is a LAW; the substance is a row in params:
  decay        dX/dt = -k(T) X, k Arrhenius or constant            (EXACT/QUAD)
  relax        dx/dt = (x* - x)/tau                                 (EXACT/QUAD)
  drain        dX/dt = -r(t), with a floor                          (EXACT)
  fick         diffusion: half-space erf, slab modes, Wells-D table (EXACT/STANDIN)
  population   growth (Baranyi) and heat death (Bigelow) in one state (EXACT per piece)
  front        x^2/(2K) + R0 x = S(t): Stefan ice, carbonation       (EXACT/QUAD)
  powerlaw     D = r tau^b by the equivalent-time form               (EXACT/QUAD)
  drying       constant rate, then a falling-rate series             (STANDIN)
  rate_integral  degree-days, a clock                                (EXACT/QUAD)
  two_process  Borbely sleep pressure, a LIBRARY FORM ONLY           (QUAD)
  rc_net       a linear thermal network, dense eigen or CG           (EXACT/STEP)

FLUX SIGN. advance() returns (state, fluxes). A flux is the amount that LEFT the
host into the named sink (negative: it came out of the sink into the host).
Keys are (what, species, sink): ("mass", "co2", "room"), ("E", "heat", "air"),
("vol", "ice", "host"). Every form returns EVERY key of its flux_keys(params),
possibly zero, and tests/test_slowforms.py checks closure form by form.

R = 8.314462618 J/(mol K). T inside a rate is a temperature in Celsius from a
drive; the form adds 273.15 for any exponential.

Sources and (u) marks are in specs/T5-scheduler_v2.txt section 4. A number
marked (u) there is unverified and says so here.
"""
import math

import numpy as np

R_GAS = 8.314462618
R_CAL = 1.987                    # cal/(mol K), for the Wells-Mehl constants
K0 = 273.15
TICK_S = 0.025                   # the tick, repeated here so this file stands alone
SIGMA = 5.670374e-8

# 4-point Gauss-Legendre on [-1, 1]
_GLX = (-0.8611363115940526, -0.3399810435848563,
        0.3399810435848563, 0.8611363115940526)
_GLW = (0.3478548451374538, 0.6521451548625461,
        0.6521451548625461, 0.3478548451374538)


class Refused(Exception):
    """A fold or a form is refused; `reason` is data, not a string to parse."""

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


class OutOfRange(Exception):
    """A closed form was asked to move outside its assumptions (F11)."""

    def __init__(self, reason, until=None):
        super().__init__(reason)
        self.reason = reason
        self.until = until


def gl4(f, a, b):
    """Integral of f over [a, b], 4-point Gauss-Legendre (f takes a float)."""
    m, h = 0.5 * (a + b), 0.5 * (b - a)
    return h * sum(w * f(m + h * x) for x, w in zip(_GLX, _GLW))


# ─────────────────────────────────────────────────────────── DRIVES ──
class ConstDrive:
    __slots__ = ("v",)
    smooth = False

    def __init__(self, v):
        self.v = float(v)

    def value(self, t):
        return self.v

    def mean(self, t0, t1):
        return self.v

    def kinks(self, t0, t1):
        return []

    def max_slope(self):
        return 0.0

    def level_ticks(self, t0, t1, level):
        return []

    def lo_hi(self):
        return self.v, self.v


class SineDrive:
    """mean + amp sin(2 pi (t - phase_s)/period_s), t in seconds."""
    __slots__ = ("mean_", "amp", "period_s", "phase_s")
    smooth = True

    def __init__(self, mean, amp, period_s, phase_s=0.0):
        self.mean_, self.amp = float(mean), float(amp)
        self.period_s, self.phase_s = float(period_s), float(phase_s)

    def _theta(self, t):
        return 2.0 * math.pi * (t * TICK_S - self.phase_s) / self.period_s

    def value(self, t):
        return self.mean_ + self.amp * math.sin(self._theta(t))

    def mean(self, t0, t1):
        if t1 == t0:
            return self.value(t0)
        a, b = self._theta(t0), self._theta(t1)
        return self.mean_ + self.amp * (math.cos(a) - math.cos(b)) / (b - a)

    def kinks(self, t0, t1):
        return []

    def max_slope(self):
        """K per tick."""
        return abs(self.amp) * 2.0 * math.pi * TICK_S / self.period_s

    def level_ticks(self, t0, t1, level):
        """Integer ticks in (t0, t1) just after which the sine is on the other
        side of `level` (analytic: arcsin, then the periodic copies)."""
        if self.amp == 0.0:
            return []
        s = (level - self.mean_) / self.amp
        if abs(s) >= 1.0:
            return []
        a1 = math.asin(s)
        a2 = math.pi - a1
        per_t = self.period_s / TICK_S
        out = []
        for a in (a1, a2):
            base = (a / (2.0 * math.pi) * self.period_s + self.phase_s) / TICK_S
            k0 = math.floor((t0 - base) / per_t) - 1
            k1 = math.ceil((t1 - base) / per_t) + 1
            for k in range(int(k0), int(k1) + 1):
                tc = base + k * per_t
                tk = int(math.floor(tc)) + 1       # first tick on the far side
                if t0 < tk < t1:
                    out.append(tk)
        return sorted(set(out))

    def lo_hi(self):
        return self.mean_ - abs(self.amp), self.mean_ + abs(self.amp)


class TableDrive:
    """steps = [(t_end_tick, value), ...]: `value` holds until t_end; the last
    value holds for ever."""
    __slots__ = ("ends", "vals")
    smooth = False

    def __init__(self, steps):
        steps = sorted((int(a), float(b)) for a, b in steps)
        self.ends = [a for a, _ in steps]
        self.vals = [b for _, b in steps]

    def value(self, t):
        for e, v in zip(self.ends, self.vals):
            if t < e:
                return v
        return self.vals[-1]

    def mean(self, t0, t1):
        if t1 <= t0:
            return self.value(t0)
        pts = [t0] + [e for e in self.ends if t0 < e < t1] + [t1]
        return sum(self.value(0.5 * (a + b)) * (b - a)
                   for a, b in zip(pts[:-1], pts[1:])) / (t1 - t0)

    def kinks(self, t0, t1):
        return [e for e in self.ends if t0 < e < t1]

    def max_slope(self):
        return 0.0

    def level_ticks(self, t0, t1, level):
        return []

    def lo_hi(self):
        return min(self.vals), max(self.vals)


class SumDrive:
    __slots__ = ("parts",)

    def __init__(self, *parts):
        self.parts = tuple(parts)

    @property
    def smooth(self):
        return any(p.smooth for p in self.parts)

    def value(self, t):
        return sum(p.value(t) for p in self.parts)

    def mean(self, t0, t1):
        return sum(p.mean(t0, t1) for p in self.parts)

    def kinks(self, t0, t1):
        return sorted({k for p in self.parts for k in p.kinks(t0, t1)})

    def max_slope(self):
        return sum(p.max_slope() for p in self.parts)

    def level_ticks(self, t0, t1, level):
        """Integer ticks in (t0, t1) just after which the SUM is on the other
        side of `level` (value >= level against below). The sum is sampled every
        1/24 of its fastest period (every kink is a cut anyway), each sign change
        is bisected to the tick. Two crossings closer than a sample step (a graze
        of the level at a peak) are the one thing it can miss; a sine part alone
        is analytic (SineDrive)."""
        t0, t1 = int(t0), int(t1)
        if t1 - t0 < 2:
            return []
        periods = [p.period_s for p in self.parts if hasattr(p, "period_s")]
        if not periods and not any(p.kinks(t0, t1) for p in self.parts):
            return []
        step = max(1, int(min(periods) / TICK_S / 24.0)) if periods else max(1, (t1 - t0) // 64)
        ts = list(range(t0, t1, step)) + [t1]
        if len(ts) > 4_000_000:
            raise OutOfRange("a summed drive too long to scan for a level crossing", until=t0)
        edges = sorted(set(ts) | {k for k in self.kinks(t0, t1)} | {k - 1 for k in self.kinks(t0, t1)})
        above = [self.value(t) >= level for t in edges]
        out = []
        for i in range(len(edges) - 1):
            if above[i] == above[i + 1]:
                continue
            lo, hi = edges[i], edges[i + 1]
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if (self.value(mid) >= level) == above[i]:
                    lo = mid
                else:
                    hi = mid
            if t0 < hi < t1:
                out.append(hi)
        return sorted(set(out))

    def lo_hi(self):
        los, his = zip(*(p.lo_hi() for p in self.parts))
        return sum(los), sum(his)


DRIVES = {"const": ConstDrive, "sine": SineDrive, "table": TableDrive,
          "sum": SumDrive}


def make_drive(spec):
    """A drive from plain data: ("const", v) | ("sine", mean, amp, per, ph) |
    ("table", [(t_end, v), ...]) | ("sum", spec, spec...); or a Drive."""
    if not isinstance(spec, (tuple, list)):
        return spec if hasattr(spec, "value") else ConstDrive(spec)
    name, *rest = spec
    if name == "table":
        return TableDrive(rest[0])
    if name == "sum":
        return SumDrive(*(make_drive(s) for s in rest))
    return DRIVES[name](*rest)


def pieces(drives, t0, t1, g=0, cuts=()):
    """The ticks [t0, ..., t1] that cut (t0, t1) into pieces on which every
    drive is smooth. The cuts are the drives' kinks, `cuts` (extra ticks, e.g.
    a drive crossing a threshold row) and, when any drive is smooth and g > 0,
    every multiple of g on the ABSOLUTE grid. Splitting an interval anywhere
    reuses the same grid cells, so advance(a) then advance(b) sums the same
    pieces as advance(a + b)."""
    t0, t1 = int(t0), int(t1)
    if t1 <= t0:
        return [t0, t1]
    pts = {t0, t1}
    for d in drives.values():
        pts.update(d.kinks(t0, t1))
    pts.update(c for c in cuts if t0 < c < t1)
    if g and any(d.smooth for d in drives.values()):
        k = (t0 // g + 1) * g
        while k < t1:
            pts.add(k)
            k += g
    return sorted(pts)


def choose_grid(drives, dlnk_dT=0.112, max_dln=0.5, cap=144000 * 6):
    """Grid cell (ticks) so that ln k changes by at most `max_dln` per cell at
    the drives' fastest slope. 0 if no drive is smooth. dlnk_dT = Ea/(R T^2) is
    0.112 per K for Ea = 80 kJ/mol at 293 K."""
    sm = [d for d in drives.values() if d.smooth]
    if not sm:
        return 0
    slope = sum(d.max_slope() for d in sm) * dlnk_dT
    if slope <= 0.0:
        return int(cap)
    return int(max(1, min(cap, math.floor(max_dln / slope))))


def _grid(p, drives):
    g = p.get("grid")
    if g:
        return int(g)
    return choose_grid(drives, p.get("dlnk_dT", 0.112))


def _P(params):
    return params if isinstance(params, dict) else dict(params)


def _rate_int(rate, drive, a, b, smooth):
    """Integral of rate(drive value) dt over the piece [a, b], in SECONDS.
    A piece of a non-smooth drive has one value, so this is exact."""
    dt = (b - a) * TICK_S
    if not smooth:
        return rate(drive.value(0.5 * (a + b))) * dt
    return gl4(lambda t: rate(drive.value(t)), a, b) * TICK_S


def arrhenius(p):
    """k(T_celsius) in 1/s from params: k, half_s, or A/Ea (or kref/Tref/Ea)."""
    if "Ea" in p:
        Ea = p["Ea"]
        if "A" in p:
            A = p["A"]
            return lambda T: A * math.exp(-Ea / (R_GAS * (T + K0)))
        kref, Tref = p["kref"], p.get("Tref_C", 20.0) + K0
        return lambda T: kref * math.exp(Ea / R_GAS * (1.0 / Tref - 1.0 / (T + K0)))
    k = p["k"] if "k" in p else math.log(2.0) / p["half_s"]
    return lambda T: k


class Form:
    """A registered closed form. Code, not data: never stored in a snapshot."""
    __slots__ = ("name", "exactness", "needs_history", "advance", "observe",
                 "cross", "valid", "project", "inject", "flux_keys", "totals")

    def __init__(self, name, exactness, advance, observe, flux_keys,
                 cross=None, valid=None, project=None, inject=None,
                 needs_history=False, totals=None):
        self.name, self.exactness = name, exactness
        self.advance, self.observe, self.flux_keys = advance, observe, flux_keys
        self.cross = cross if cross is not None else _cross_generic(self)
        self.valid = valid if valid is not None else _valid_always
        self.project, self.inject = project, inject
        self.needs_history = needs_history
        self.totals = totals


FORMS = {}


def register(form):
    FORMS[form.name] = form
    return form


def _valid_always(state, params, drives, t0, t1=None):
    return True, None, ""


def _cross_generic(form):
    def cross(state, params, drives, t0, key, level, t_max):
        return cross_scan(form, state, params, drives, t0, key, level, t_max)
    return cross


def cross_scan(form, state, params, drives, t0, key, level, t_max, cuts=()):
    """Earliest integer tick in [t0, t_max] at which observe(key) reaches
    `level` (from the side it starts on), or None. Walks the form's pieces and
    bisects on the integer tick inside the piece where the sign changes, so the
    answer is exact for a monotone piece (bracket, then exact inside it)."""
    p = _P(params)
    v0 = form.observe(state, p, key)
    if v0 == level:
        return int(t0)
    up = level > v0
    g = _grid(p, drives)
    pts = pieces(drives, t0, t_max, g, cuts)
    s, ta = state, int(t0)
    for tb in pts[1:]:
        s1, _ = form.advance(s, p, drives, ta, tb)
        v1 = form.observe(s1, p, key)
        if (v1 >= level) if up else (v1 <= level):
            lo, hi = ta, tb                       # first True tick is in (lo, hi]
            while hi - lo > 1:
                mid = (lo + hi) // 2
                sm, _ = form.advance(s, p, drives, ta, mid)
                vm = form.observe(sm, p, key)
                if (vm >= level) if up else (vm <= level):
                    hi = mid
                else:
                    lo = mid
            return int(hi)
        s, ta = s1, tb
    return None


# ───────────────────────────────────────────── 4.1 DECAY and ARRHENIUS ──
def decay_advance(state, params, drives, t0, t1):
    p = _P(params)
    rate = arrhenius(p)
    dname = p.get("drive", "T")
    x0 = state[0]
    if dname in drives and "Ea" in p:
        d = drives[dname]
        sm = d.smooth
        pts = pieces({dname: d}, t0, t1, _grid(p, {dname: d}))
        kint = sum(_rate_int(rate, d, a, b, sm) for a, b in zip(pts[:-1], pts[1:]))
    else:
        kint = rate(p.get("T", 20.0)) * (t1 - t0) * TICK_S
    x1 = x0 * math.exp(-kint)
    lost = x0 - x1
    return (x1,), {("mass", p.get("product", "x"), "product"): lost,
                   ("E", "heat", "air"): lost * p.get("heat_J_per_unit", 0.0)}


def decay_observe(state, params, key):
    return state[0]


def decay_flux_keys(params):
    p = _P(params)
    return (("mass", p.get("product", "x"), "product"), ("E", "heat", "air"))


def decay_discrete(x0, k_per_tick, n):
    """The tick law's OWN recurrence x <- x (1 - k) in closed form. Used by the
    twins: it is exact to float rounding, where exp(-k n) is not."""
    return x0 * (1.0 - k_per_tick) ** n


register(Form("decay", "QUAD", decay_advance, decay_observe, decay_flux_keys))


# ─────────────────────────────────────────── 4.2 RELAXATION, 4.11 DRAIN ──
def relax_advance(state, params, drives, t0, t1):
    """x(t1) = x* + (x0 - x*) exp(-dt/tau), piecewise in x* (a drive 'target'
    or a number) and in tau (a number, or a drive 'tau' held per piece). A
    smooth target is integrated against the kernel: x e^{-dt/tau} +
    (1/tau) int x*(s) e^{-(t1-s)/tau} ds, on cells of at most tau/2."""
    p = _P(params)
    tau = p["tau_s"]
    tgt = drives.get("target")
    tdr = drives.get("tau")
    x = state[0]
    x0 = x
    both = {k: v for k, v in (("target", tgt), ("tau", tdr)) if v is not None}
    if both:
        g = _grid(p, both)
        if g:
            g = min(g, max(1, int(0.5 * tau / TICK_S)))
        pts = pieces(both, t0, t1, g)
    else:
        pts = [int(t0), int(t1)]
    for a, b in zip(pts[:-1], pts[1:]):
        dt = (b - a) * TICK_S
        if dt <= 0.0:
            continue
        tau_p = tdr.value(0.5 * (a + b)) if tdr is not None else tau
        if tgt is not None and tgt.smooth:
            def f(s, b=b, tau_p=tau_p):
                return tgt.value(s) * math.exp(-(b - s) * TICK_S / tau_p)
            x = x * math.exp(-dt / tau_p) + gl4(f, a, b) * TICK_S / tau_p
        else:
            xs = tgt.value(0.5 * (a + b)) if tgt is not None else p.get("target", 0.0)
            x = xs + (x - xs) * math.exp(-dt / tau_p)
    d = x - x0
    return (x,), {("store", "x", "sink"): -d,
                  ("E", "heat", "air"): abs(d) * p.get("heat_J_per_unit", 0.0)}


def relax_observe(state, params, key):
    return state[0]


def relax_flux_keys(params):
    return (("store", "x", "sink"), ("E", "heat", "air"))


register(Form("relax", "QUAD", relax_advance, relax_observe, relax_flux_keys))


def drain_advance(state, params, drives, t0, t1):
    """dX/dt = -r(t); x = max(x0 - integral r, floor). r is a number or the
    drive 'rate' (per second)."""
    p = _P(params)
    x0 = state[0]
    fl = p.get("floor", 0.0)
    d = drives.get("rate")
    if d is None:
        used = p.get("rate", 0.0) * (t1 - t0) * TICK_S
    else:
        pts = pieces({"rate": d}, t0, t1, _grid(p, {"rate": d}))
        used = 0.0
        for a, b in zip(pts[:-1], pts[1:]):
            used += (d.value(0.5 * (a + b)) * (b - a) * TICK_S if not d.smooth
                     else gl4(d.value, a, b) * TICK_S)
    x1 = max(x0 - used, fl)
    gone = x0 - x1
    return (x1,), {("store", "x", "drained"): gone,
                   ("E", "heat", "air"): gone * p.get("heat_J_per_unit", 0.0)}


def drain_flux_keys(params):
    return (("store", "x", "drained"), ("E", "heat", "air"))


register(Form("drain", "QUAD", drain_advance, relax_observe, drain_flux_keys))


# ───────────────────────────────────────────────── 4.3 FICK DIFFUSION ──
def D_of_T(p, T_c):
    """D(T) = D0 exp(-Q/(R T)) m2/s, or a constant D."""
    if "Q" in p:
        return p["D0"] * math.exp(-p["Q"] / (R_GAS * (T_c + K0)))
    return p["D"]


def deff_int(p, drives, t0, t1):
    """Deff = integral of D dt over [t0, t1], m2 (the time-change trick)."""
    d = drives.get("T")
    if "Q" not in p or d is None:
        D = p["D"] if "D" in p else D_of_T(p, p.get("T", 20.0))
        return D * (t1 - t0) * TICK_S
    rate = lambda T: D_of_T(p, T)                       # noqa: E731
    g = _grid({"grid": p.get("grid", 0), "dlnk_dT": p["Q"] / (R_GAS * 293.15 ** 2)},
              {"T": d})
    pts = pieces({"T": d}, t0, t1, g, _cuts_for(d, p, t0, t1))
    return sum(_rate_int(rate, d, a, b, d.smooth) for a, b in zip(pts[:-1], pts[1:]))


def _cuts_for(d, p, t0, t1):
    """Ticks where the T drive crosses a row named in params (A1 for carbon)."""
    cuts = []
    for k in ("A1_C", "Tmin_valid", "T_kill", "Tmin", "Tmax"):
        if k in p:
            cuts += d.level_ticks(t0, t1, p[k])
    return cuts


# --- half-space --------------------------------------------------------
def halfspace_c(x, deff, c0, cs):
    """C(x, t) = Cs + (C0 - Cs) erf(x / (2 sqrt(Deff)))."""
    if deff <= 0.0:
        return c0 if x > 0 else cs
    return cs + (c0 - cs) * math.erf(x / (2.0 * math.sqrt(deff)))


def halfspace_depth(c_target, deff, c0, cs):
    """Depth where C = c_target: erf(z) = (Cs - c)/(Cs - C0), depth 2 z sqrt(Deff)."""
    r = (cs - c_target) / (cs - c0)
    if not 0.0 < r < 1.0:
        return 0.0 if r <= 0.0 else float("inf")
    lo, hi = 0.0, 6.0
    for _ in range(80):
        m = 0.5 * (lo + hi)
        if math.erf(m) < r:
            lo = m
        else:
            hi = m
    return 2.0 * 0.5 * (lo + hi) * math.sqrt(deff)


def halfspace_uptake(deff, c0, cs):
    """Mass taken in per m2 by the half-space: 2 (Cs - C0) sqrt(Deff / pi)."""
    return 2.0 * (cs - c0) * math.sqrt(max(deff, 0.0) / math.pi)


# --- slab ----------------------------------------------------------------
def slab_k(L, bc, K):
    """Wave numbers of the modes. 'dd' both faces at Cs: sin(n pi x/L).
    'ds' x=0 at Cs, x=L sealed: sin((n-1/2) pi x/L). 'sd' the mirror: cos((n-1/2)
    pi x/L). 'ss' both sealed: cos(n pi x/L), n = 0 .. K-1."""
    n = np.arange(1, K + 1, dtype=float)
    if bc == "dd":
        return n * math.pi / L
    if bc in ("ds", "sd"):
        return (n - 0.5) * math.pi / L
    return (n - 1.0) * math.pi / L


def slab_basis(x, L, bc, K):
    """Mode values, shape (len(x), K)."""
    k = slab_k(L, bc, K)
    x = np.asarray(x, float)[:, None]
    return np.cos(k[None, :] * x) if bc in ("sd", "ss") else np.sin(k[None, :] * x)


def slab_mass_of_modes(L, bc, K):
    """Integral over [0, L] of each mode."""
    k = slab_k(L, bc, K)
    if bc == "dd" or bc == "ds":
        return (1.0 - np.cos(k * L)) / k
    out = np.empty(K)
    nz = k > 0
    out[nz] = np.sin(k[nz] * L) / k[nz]
    out[~nz] = L
    return out


def slab_profile(b, x, L, bc, cs):
    base = 0.0 if bc == "ss" else cs
    return base + slab_basis(x, L, bc, len(b)) @ np.asarray(b, float)


def slab_mass(b, L, bc, cs):
    """Mass per m2 of the slab implied by the modes."""
    base = 0.0 if bc == "ss" else cs * L
    return base + float(np.dot(np.asarray(b, float), slab_mass_of_modes(L, bc, len(b))))


def slab_uniform(c0, cs, L, K, bc="dd"):
    """Mode amplitudes of a uniform start c0 against faces at cs."""
    return _uniform_amp(c0, cs, L, slab_k(L, bc, K), bc)


def _uniform_amp(c0, cs, L, k, bc):
    # b_n = (2/L) int (c0 - cs) mode_n dx, norm int mode^2 = L/2 (L for n=0)
    if bc in ("dd", "ds"):
        m = (1.0 - np.cos(k * L)) / k
        return (c0 - cs) * (2.0 / L) * m
    if bc == "sd":
        m = np.sin(k * L) / k
        return (c0 - cs) * (2.0 / L) * m
    out = np.zeros(len(k))
    out[0] = c0
    return out


def slab_advance(state, params, drives, t0, t1):
    p = _P(params)
    L, bc, cs = p["L"], p.get("bc", "dd"), p.get("Cs", 0.0)
    b = np.asarray(state, float)
    deff = deff_int(p, drives, t0, t1)
    k = slab_k(L, bc, len(b))
    b1 = b * np.exp(-(k * k) * deff)
    f0, fL = _slab_face_fluxes(b, b1, k, L, bc)
    sp = p.get("species", "c")
    return tuple(float(v) for v in b1), {
        ("mass", sp, "face0"): f0, ("mass", sp, "faceL"): fL}


def _slab_face_fluxes(b, b1, k, L, bc):
    """Mass that left through x=0 and x=L over the step, per m2. For sin modes
    the gradient at 0 is k and at L is k cos(kL); the time integral of D times
    it is b (1 - e^{-k^2 Deff}) / k, i.e. (b - b1)/k."""
    d = (b - b1)
    kk = np.where(k > 0, k, 1.0)
    f0 = fL = 0.0
    if bc == "dd":
        f0 = float(np.sum(d / kk))
        fL = float(-np.sum(d * np.cos(k * L) / kk))
    elif bc == "ds":
        f0 = float(np.sum(d / kk))
    elif bc == "sd":                       # cos modes: gradient at L is -k sin(kL)
        fL = float(np.sum(d * np.sin(k * L) / kk))
    return f0, fL


def slab_observe(state, params, key):
    p = _P(params)
    L, bc, cs = p["L"], p.get("bc", "dd"), p.get("Cs", 0.0)
    if key == "mass":
        return slab_mass(state, L, bc, cs)
    if isinstance(key, tuple) and key[0] == "c":
        return float(slab_profile(state, [key[1]], L, bc, cs)[0])
    if key == "center":
        return float(slab_profile(state, [0.5 * L], L, bc, cs)[0])
    raise KeyError(key)


def slab_flux_keys(params):
    sp = _P(params).get("species", "c")
    return (("mass", sp, "face0"), ("mass", sp, "faceL"))


def slab_project(x, c, params, kmax=32, mass_tol=1e-9, shape_tol=1e-3,
                 mass_ref=None):
    """Fit the modes to a node profile, or raise Refused. `x`, `c`: node
    positions (vertex nodes, x[0] = 0, x[-1] = L) and values.

    Rules (spec 4.3): refuse when a Dirichlet face node differs from Cs by more
    than shape_tol of the range (an odd extension of a mismatched face rings);
    pick K from the spectral decay (K <= kmax); renormalise the mass-carrying
    modes by ONE common factor so the series reproduces mass_ref (trapezoid sum
    of the nodes by default) to mass_tol, and refuse if that factor differs
    from 1 by more than shape_tol; refuse if any reconstructed node leaves
    [min, max] of the profile by more than shape_tol of the range. The fit is a
    least-squares projection onto the first K modes, so fold -> inject -> fold
    is exactly idempotent."""
    p = _P(params)
    L, bc, cs = p["L"], p.get("bc", "dd"), p.get("Cs", 0.0)
    x = np.asarray(x, float)
    c = np.asarray(c, float)
    rng = float(c.max() - c.min()) or max(abs(float(c.max())), 1e-30)
    if bc in ("dd", "ds") and abs(c[0] - cs) > shape_tol * rng:
        raise Refused("face 0 differs from Cs")
    if bc == "dd" and abs(c[-1] - cs) > shape_tol * rng:
        raise Refused("face L differs from Cs")
    if bc == "sd" and abs(c[-1] - cs) > shape_tol * rng:
        raise Refused("face L differs from Cs")
    base = 0.0 if bc == "ss" else cs
    u = c - base
    w = 0.5 * (np.r_[0.0, np.diff(x)] + np.r_[np.diff(x), 0.0])
    kmax = int(min(kmax, len(x) - 2 if bc == "dd" else len(x) - 1))
    B = slab_basis(x, L, bc, kmax)
    a_all, *_ = np.linalg.lstsq(B, u, rcond=None)
    big = float(np.abs(a_all).max()) or 1.0
    K = kmax
    while K > 1 and abs(a_all[K - 1]) <= 1e-4 * big:
        K -= 1
    BK = B[:, :K]
    a, *_ = np.linalg.lstsq(BK, u, rcond=None)
    resid = float(np.linalg.norm(BK @ a - u) / max(np.linalg.norm(u), 1e-300))
    if resid > shape_tol:
        raise Refused(f"profile not representable by {K} modes (resid {resid:.2e})")
    m_modes = (BK * w[:, None]).sum(axis=0)         # mass of each mode on the nodes
    m_ref = float(np.dot(w, c)) if mass_ref is None else float(mass_ref)
    need = m_ref - base * float(w.sum())
    have = float(np.dot(a, m_modes))
    if abs(have) > 1e-300:
        f = need / have
        if abs(f - 1.0) > shape_tol:
            raise Refused(f"mass factor {f:.4f} out of tolerance")
        carry = np.abs(m_modes) > 1e-12 * float(np.abs(m_modes).max() or 1.0)
        a = a.copy()
        a[carry] *= f
    elif abs(need) > mass_tol * max(abs(m_ref), 1e-300):
        raise Refused("mass not carried by the modes")
    rec = base + BK @ a
    lo, hi = float(c.min()), float(c.max())
    if rec.min() < lo - shape_tol * rng or rec.max() > hi + shape_tol * rng:
        raise Refused("reconstruction leaves the profile's range (ringing)")
    fit = base * float(w.sum()) + float(np.dot(a, m_modes))
    if abs(fit - m_ref) > mass_tol * max(abs(m_ref), 1e-300):
        raise Refused("mass not reproduced to the tolerance")
    return tuple(float(v) for v in a), {"K": K, "resid": resid, "mass_ref": m_ref,
                                        "mass_fit": fit}


def slab_inject(state, x, params):
    p = _P(params)
    return slab_profile(state, x, p["L"], p.get("bc", "dd"), p.get("Cs", 0.0))


def slab_totals(state, params, x=None):
    """The mass the modes hold: the continuous integral, or, given the host's
    node positions, the trapezoid sum of the reconstruction on those nodes (the
    figure a fold renormalises to and an unfold hands back, so the two agree to
    rounding)."""
    p = _P(params)
    if x is None:
        return slab_mass(state, p["L"], p.get("bc", "dd"), p.get("Cs", 0.0))
    x = np.asarray(x, float)
    w = 0.5 * (np.r_[0.0, np.diff(x)] + np.r_[np.diff(x), 0.0])
    return float(np.dot(w, slab_inject(state, x, p)))


def slab_valid(state, params, drives, t0, t1=None):
    p = _P(params)
    if p.get("Dratio", 1.0) > 1.2:
        return False, int(t0), "D varies by more than 20% across the profile"
    return _range_valid(p, drives, t0, t1)


def _range_valid(p, drives, t0, t1):
    """Refuse below an A1-style row: the form's D applies only above it."""
    d = drives.get("T")
    lo = p.get("A1_C", p.get("Tmin_valid"))
    if d is None or lo is None:
        return True, None, ""
    horizon = int(t1 if t1 is not None else t0)
    if d.value(t0) < lo:
        return False, int(t0), "below the austenite row"
    pts = pieces({"T": d}, t0, horizon, 0, d.level_ticks(t0, horizon, lo))
    for a, b in zip(pts[:-1], pts[1:]):
        if d.value(0.5 * (a + b)) < lo:
            return False, int(a), "drive falls below the austenite row"
    return True, None, ""


# --- Wells-D similarity solution ------------------------------------------
_SIM = {}                           # (a, b, cs, c0) -> (xi, c) table, built once


def wells_D(c_pct, T_c):
    """Wells-Mehl (u, recalled): D = (0.07 + 0.06 C%) exp(-32000/(R' T)) cm2/s,
    returned in m2/s. 1.19e-11, 1.45e-11, 1.89e-11 at 0.2, 0.5, 1.0 %C, 925 C."""
    return (0.07 + 0.06 * c_pct) * math.exp(-32000.0 / (R_CAL * (T_c + K0))) * 1e-4


def similarity_table(a, b, cs, c0, n=400, steps_per=8):
    """Boltzmann-Matano similarity solution C = f(xi), xi = x/sqrt(t), for
    D(C) = a + b C (any unit; xi comes out in sqrt of that unit). Solves
    (D f')' + (xi/2) f' = 0 by shooting on the flux at the surface and RK4,
    with xi measured in units of sqrt(D_max) so the range is about 6. Returns
    (xi, c) arrays with xi in units of sqrt(D unit * s)."""
    key = (round(a, 12), round(b, 12), cs, c0)
    got = _SIM.get(key)
    if got is not None:
        return got
    Dmax = max(a + b * cs, a + b * c0)
    sq = math.sqrt(Dmax)
    xmax = 7.0                                     # in sqrt(Dmax) units

    def D(c):
        return (a + b * c) / Dmax

    def run(g0):
        f, g = cs, g0
        h = xmax / (n * steps_per)
        xs, fs = [0.0], [f]
        x = 0.0

        def rhs(x, f, g):
            d = D(f)
            return g / d, -0.5 * x * g / d
        for i in range(n * steps_per):
            k1 = rhs(x, f, g)
            k2 = rhs(x + h / 2, f + h / 2 * k1[0], g + h / 2 * k1[1])
            k3 = rhs(x + h / 2, f + h / 2 * k2[0], g + h / 2 * k2[1])
            k4 = rhs(x + h, f + h * k3[0], g + h * k3[1])
            f += h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            g += h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            x += h
            if (i + 1) % steps_per == 0:
                xs.append(x)
                fs.append(f)
        return np.array(xs), np.array(fs)

    lo, hi = -50.0, 0.0                            # g0 < 0: C falls away from Cs
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        xs, fs = run(mid)
        # too steep: C reaches c0 early and keeps falling below it
        if fs.min() < c0:
            lo = mid
        else:
            hi = mid
    xs, fs = run(0.5 * (lo + hi))
    fs = np.maximum(fs, c0)
    out = (xs * sq, fs)
    _SIM[key] = out
    return out


def similarity_depth(c_target, tau, a, b, cs, c0):
    """Depth (in sqrt(unit) * sqrt(tau)) where C = c_target, from the table."""
    xi, f = similarity_table(a, b, cs, c0)
    if not c0 < c_target < cs:
        raise ValueError("target outside (C0, Cs)")
    idx = np.searchsorted(-f, -c_target)           # f is decreasing
    x = np.interp(c_target, f[idx - 1:idx + 1][::-1], xi[idx - 1:idx + 1][::-1])
    return float(x) * math.sqrt(tau)


def wells_tau(drives, t0, t1, p):
    """The time-change integral tau = int exp(-Q'/(R' T)) dt (s): D(C, T) =
    (a + b C) 1e-4 exp(-Q'/(R' T)) is separable in C and T, so C = f(x/sqrt(tau))."""
    Q = p.get("Q_cal", 32000.0)
    d = drives["T"]
    rate = lambda T: math.exp(-Q / (R_CAL * (T + K0)))           # noqa: E731
    pts = pieces({"T": d}, t0, t1, _grid({"grid": p.get("grid", 0),
                                          "dlnk_dT": Q / (R_CAL * 1198.0 ** 2)},
                                         {"T": d}), _cuts_for(d, p, t0, t1))
    return sum(_rate_int(rate, d, a, b, d.smooth) for a, b in zip(pts[:-1], pts[1:]))


def wells_advance(state, params, drives, t0, t1):
    """state = (tau,). The surface sits at Cs; D(C) of Wells; no constant-D
    shortcut (it undershoots the 0.4 %C depth by 30-40 %)."""
    p = _P(params)
    ok, until, why = _range_valid(p, drives, t0, t1)
    if not ok:
        raise OutOfRange(why, until)
    tau1 = state[0] + wells_tau(drives, t0, t1, p)
    sp = p.get("species", "c")
    up0 = _wells_uptake(state[0], p)
    up1 = _wells_uptake(tau1, p)
    return (tau1,), {("mass", sp, "face0"): -(up1 - up0)}


def _wells_uptake(tau, p):
    """Mass in per m2 relative to the start: integral (C - C0) dx over the
    similarity profile, times sqrt(tau). (C in %C; x in m: table x is in
    sqrt(1e-4 m2/s unit)*sqrt(s), converted here.)"""
    if tau <= 0.0:
        return 0.0
    xi, f = similarity_table(p["wa"], p["wb"], p["Cs"], p["C0"])
    return float(np.trapezoid(f - p["C0"], xi)) * 1e-2 * math.sqrt(tau)


def wells_observe(state, params, key):
    """key ('depth', c_pct) -> depth in m where the carbon falls to c_pct."""
    p = _P(params)
    if isinstance(key, tuple) and key[0] == "depth":
        # the table is in sqrt(cm2/s * s) = cm * 100 / 100: D unit is 1e-4 m2/s,
        # so depth in m = table depth (sqrt(1e-4 m2) = 1e-2 m)
        return similarity_depth(key[1], state[0], p["wa"], p["wb"], p["Cs"], p["C0"]) * 1e-2
    if key == "tau":
        return state[0]
    raise KeyError(key)


def wells_flux_keys(params):
    return (("mass", _P(params).get("species", "c"), "face0"),)


def wells_valid(state, params, drives, t0, t1=None):
    return _range_valid(_P(params), drives, t0, t1)


def fick_advance(state, params, drives, t0, t1):
    p = _P(params)
    geo = p.get("geometry", "slab")
    if geo == "slab":
        return slab_advance(state, p, drives, t0, t1)
    if geo == "wells":
        return wells_advance(state, p, drives, t0, t1)
    # half-space, constant D (STANDIN-LOW when D varies; valid() says so)
    ok, until, why = _range_valid(p, drives, t0, t1)
    if not ok:
        raise OutOfRange(why, until)
    deff = state[0] + deff_int(p, drives, t0, t1)
    up = halfspace_uptake(deff, p["C0"], p["Cs"]) - halfspace_uptake(state[0], p["C0"], p["Cs"])
    return (deff,), {("mass", p.get("species", "c"), "face0"): -up}


def fick_observe(state, params, key):
    p = _P(params)
    geo = p.get("geometry", "slab")
    if geo == "slab":
        return slab_observe(state, p, key)
    if geo == "wells":
        return wells_observe(state, p, key)
    if isinstance(key, tuple) and key[0] == "c":
        return halfspace_c(key[1], state[0], p["C0"], p["Cs"])
    if isinstance(key, tuple) and key[0] == "depth":
        return halfspace_depth(key[1], state[0], p["C0"], p["Cs"])
    if key == "deff":
        return state[0]
    raise KeyError(key)


def fick_flux_keys(params):
    p = _P(params)
    if p.get("geometry", "slab") == "slab":
        return slab_flux_keys(p)
    return (("mass", p.get("species", "c"), "face0"),)


def fick_valid(state, params, drives, t0, t1=None):
    p = _P(params)
    geo = p.get("geometry", "slab")
    if geo == "wells":
        return wells_valid(state, p, drives, t0, t1)
    if geo == "slab":
        return slab_valid(state, p, drives, t0, t1)
    if p.get("Dratio", 1.0) > 1.2:
        return False, int(t0), "D varies by more than 20% across the profile"
    if p.get("L") and math.sqrt(state[0]) >= 0.1 * p["L"]:
        return False, int(t0), "sqrt(Deff) reached 0.1 L: use the slab"
    return _range_valid(p, drives, t0, t1)


register(Form("fick", "EXACT", fick_advance, fick_observe, fick_flux_keys,
              valid=fick_valid, project=slab_project, inject=slab_inject,
              totals=slab_totals))


# ───────────────────────────────────────────── 4.4 POPULATION (y, ln q) ──
def ratkowsky_mu(T, b, Tmin, Tmax, c):
    """Growth rate (1/h) from the FULL Ratkowsky form,
    sqrt(mu) = b (T - Tmin)(1 - exp(c (T - Tmax))), zero outside (Tmin, Tmax)."""
    if T <= Tmin or T >= Tmax:
        return 0.0
    return (b * (T - Tmin) * (1.0 - math.exp(c * (T - Tmax)))) ** 2


def _logq_grow(y0, lq0, mu, dt, ymax):
    """Baranyi growth over a piece of constant mu (1/s), dt seconds, in LOG
    FORM so that exp(mu t) never overflows for a year-long piece.
        A(t) = t + (1/mu) ln((exp(-mu t) + q0)/(1 + q0))
        y(t) = y0 + mu A - ln(1 + (exp(mu A) - 1)/exp(ymax - y0))."""
    if mu <= 0.0 or dt <= 0.0:
        return y0, lq0
    q0 = math.exp(min(lq0, 700.0))
    ln_num = float(np.logaddexp(-mu * dt, lq0))               # ln(e^{-mu t} + q0)
    A = dt + (ln_num - math.log1p(q0) if lq0 < 700 else ln_num - lq0) / mu
    muA = mu * A
    d = ymax - y0
    if d <= 0.0:
        return ymax, lq0 + mu * dt
    # ln(1 + (E-1)/K) = ln(K - 1 + E) - ln K, with ln(K-1) = d + log1p(-e^-d)
    ln_km1 = d + math.log1p(-math.exp(-d)) if d > 1e-12 else math.log(max(d, 1e-300))
    y1 = y0 + muA - (float(np.logaddexp(ln_km1, muA)) - d)
    return min(y1, ymax), lq0 + mu * dt


def population_advance(state, params, drives, t0, t1):
    p = _P(params)
    y, lq = state
    y0 = y
    d = drives["T"]
    ymax = p.get("ymax", math.log(1e9))
    T_kill = p.get("T_kill", 60.0)
    yfloor = p.get("y_floor", math.log(1e-6))
    gam = p.get("gamma", 1.0)
    mu_fn = lambda T: gam * ratkowsky_mu(T, p["b"], p["Tmin"], p["Tmax"], p.get("c", 0.3))   # noqa: E731
    cuts = []
    for lvl in (p["Tmin"], p["Tmax"], T_kill):
        cuts += d.level_ticks(t0, t1, lvl)
    pts = pieces({"T": d}, t0, t1, _grid({"grid": p.get("grid", 0), "dlnk_dT": 0.3}, {"T": d}),
                 cuts)
    for a, b in zip(pts[:-1], pts[1:]):
        dt = (b - a) * TICK_S
        if dt <= 0.0:
            continue
        Tm = d.value(0.5 * (a + b))
        if Tm >= T_kill:
            ln10_over_D = lambda T: (math.log(10.0) / p["D_ref_s"]                  # noqa: E731
                                     * 10.0 ** ((T - p["T_ref_C"]) / p.get("z", 10.0)))
            if d.smooth:                       # the same quadrature as growth: GL4 of the rate
                dy = gl4(lambda t: ln10_over_D(d.value(t)), a, b) * TICK_S
            else:
                dy = ln10_over_D(Tm) * dt
            y = max(y - dy, yfloor)
        else:
            if d.smooth:
                mu = gl4(lambda t: mu_fn(d.value(t)), a, b) / (b - a) / 3600.0
            else:
                mu = mu_fn(Tm) / 3600.0
            y, lq = _logq_grow(y, lq, mu, dt, ymax)
    # fluxes: biomass in grams; growth takes nutrient and makes heat, death
    # moves biomass to dead mass
    unit_g = p.get("host_g", 1.0) * p.get("cell_g", 1e-12)
    dB = (math.exp(min(y, 700.0)) - math.exp(min(y0, 700.0))) * unit_g
    return (y, lq), {("mass", "nutrient", "biomass"): max(dB, 0.0),
                     ("mass", "biomass", "dead"): max(-dB, 0.0),
                     ("E", "heat", "air"): max(dB, 0.0) * p.get("heat_J_per_g", 0.0)}


def population_observe(state, params, key):
    if key == "y":
        return state[0]
    if key == "count":
        return math.exp(min(state[0], 700.0))
    if key == "log10":
        return state[0] / math.log(10.0)
    if key == "q":
        return math.exp(min(state[1], 700.0))
    if key == "lq":
        return state[1]
    raise KeyError(key)


def population_flux_keys(params):
    return (("mass", "nutrient", "biomass"), ("mass", "biomass", "dead"),
            ("E", "heat", "air"))


def population_valid(state, params, drives, t0, t1=None):
    p = _P(params)
    if p.get("self_heat_J_per_kg_s", 0.0) > p.get("self_heat_max", 1e9):
        return False, int(t0), "growth heat per kg above the row (self-heating heap)"
    if p.get("host_temp_is_drive", True) is False:
        return False, int(t0), "host temperature is not a drive (V1)"
    return True, None, ""


register(Form("population", "EXACT", population_advance, population_observe,
              population_flux_keys, valid=population_valid))


# ──────────────────────────────────────── 4.5 MOVING FRONT (Stefan etc.) ──
def front_x_of_S(S, K, R0):
    """x^2/(2K) + R0 x = S  ->  x = -K R0 + sqrt(K^2 R0^2 + 2 K S)."""
    if S <= 0.0:
        return 0.0
    return -K * R0 + math.sqrt(K * K * R0 * R0 + 2.0 * K * S)


def front_S_of_x(x, K, R0):
    return x * x / (2.0 * K) + R0 * x


def ice_x_of_t(x0, K, R0, dT, rho_l, qw, dt):
    """Ice growth with a water-side flux over constant dT (K) for dt seconds.
    rho L dx/dt = dT/(x/K + R0) - q_w. With u = x/K + R0 the time to go from u0
    to u is t = rho L K [-u/q_w - (dT/q_w^2) ln(dT - q_w u)] between the two;
    inverted here by Newton. Returns (x1, melted_flag)."""
    u0 = x0 / K + R0
    if qw <= 0.0:
        S0 = front_S_of_x(x0, K, R0)
        return front_x_of_S(S0 + dT * dt / rho_l, K, R0), False
    if dT - qw * u0 <= 0.0:
        return x0, True                       # no growth (melting at this dT)
    umax = dT / qw

    def tfun(u):
        return rho_l * K * (-u / qw - (dT / qw ** 2) * math.log(dT - qw * u))
    target = tfun(u0) + dt
    lo, hi = u0, umax
    # t(u) rises to infinity at umax: bracket then Newton/bisect
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if hi - lo < 1e-15 * umax or mid >= umax:
            break
        if tfun(mid) < target:
            lo = mid
        else:
            hi = mid
    u = 0.5 * (lo + hi)
    for _ in range(4):                         # a few Newton polish steps
        fprime = rho_l * K * u / (dT - qw * u)
        if fprime <= 0.0:
            break
        un = u - (tfun(u) - target) / fprime
        if u0 <= un < umax:
            u = un
    return (u - R0) * K, False


def front_advance(state, params, drives, t0, t1):
    p = _P(params)
    mode = p.get("mode", "ice")
    x0 = state[0]
    K, R0 = p["K"], p.get("R0", 0.0)
    if mode == "ice":
        d = drives["T"]
        Tf = p.get("Tf", 0.0)
        rho_l = p["rho"] * p["L_f"]
        qw = p.get("q_w", 0.0)
        cuts = d.level_ticks(t0, t1, Tf)
        pts = pieces({"T": d}, t0, t1, _grid({"grid": p.get("grid", 0), "dlnk_dT": 0.05},
                                             {"T": d}), cuts)
        x = x0
        for a, b in zip(pts[:-1], pts[1:]):
            dt = (b - a) * TICK_S
            dT = (Tf - (d.mean(a, b) if d.smooth else d.value(0.5 * (a + b))))
            if dT <= 0.0:
                continue                       # above freezing: no growth (melt: valid())
            x, _ = ice_x_of_t(x, K, R0, dT, rho_l, qw, dt)
        h = p.get("area", 1.0)
        dx = x - x0
        lat = dx * rho_l * h                       # J released by freezing
        qwJ = qw * (t1 - t0) * TICK_S * h if dx > 0.0 else 0.0
        m = dx * p["rho"] * h                      # kg of water frozen
        # the latent heat and the water's own heat both leave UPWARD through
        # the ice to the air; the water gave qwJ, so it is booked negative
        return (x,), {("E", "latent", "air_top"): lat + qwJ,
                      ("E", "latent", "water"): -qwJ,
                      ("vol", "ice", "host"): (m / 1000.0) * (1000.0 / p["rho"] - 1.0)}
    # carbonation-like: S rate = c_rate * f(RH)
    dname = p.get("drive", "RH")
    d = drives.get(dname)
    S0 = front_S_of_x(x0, K, R0)
    fn = p.get("fn_table")                    # ((x0, y0), (x1, y1), ...) of f(RH)

    def rate(v):
        return p["c_rate"] * (_interp(fn, v) if fn else 1.0)
    if d is None:
        dS = rate(p.get("RH", 1.0)) * (t1 - t0) * TICK_S
    else:
        pts = pieces({dname: d}, t0, t1, _grid({"grid": p.get("grid", 0), "dlnk_dT": 0.05},
                                               {dname: d}))
        dS = sum(_rate_int(rate, d, a, b, d.smooth) for a, b in zip(pts[:-1], pts[1:]))
    x1 = front_x_of_S(S0 + dS, K, R0)
    flag = 0.0
    cap = p.get("cap_mol_m3", 0.0)
    avail = p.get("co2_avail")
    if cap and avail is not None and cap * (x1 - x0) > avail:
        x1 = x0 + avail / cap               # the room cannot supply it (V2)
        flag = 1.0
    h = p.get("area", 1.0)
    mol = cap * (x1 - x0) * h
    return (x1,), {("mass", "co2", "room"): -mol * 44.0,
                   ("mass", "h2o", "room"): mol * 18.0,
                   ("E", "reaction", "air"): mol * p.get("dH_J_mol", 110e3),
                   ("flag", "supply_limited", "-"): flag}


def _interp(table, v):
    xs = [a for a, _ in table]
    ys = [b for _, b in table]
    return float(np.interp(v, xs, ys))


def front_observe(state, params, key):
    p = _P(params)
    if key in ("x", "depth", "thickness"):
        return state[0]
    if key == "S":
        return front_S_of_x(state[0], p["K"], p.get("R0", 0.0))
    raise KeyError(key)


def front_flux_keys(params):
    if _P(params).get("mode", "ice") == "ice":
        return (("E", "latent", "air_top"), ("E", "latent", "water"), ("vol", "ice", "host"))
    return (("mass", "co2", "room"), ("mass", "h2o", "room"), ("E", "reaction", "air"),
            ("flag", "supply_limited", "-"))


def front_valid(state, params, drives, t0, t1=None):
    p = _P(params)
    if p.get("mode", "ice") == "ice":
        c = p.get("c_ice", 2100.0)
        d = drives.get("T")
        if d is not None:
            lo, _ = d.lo_hi()
            sten = c * max(p.get("Tf", 0.0) - lo, 0.0) / p["L_f"]
            if sten >= 0.1:
                return False, int(t0), "Stefan number above 0.1"
        if p.get("q_w", 0.0) > 0.0 and d is not None:
            u0 = state[0] / p["K"] + p.get("R0", 0.0)
            dT = p.get("Tf", 0.0) - d.value(t0)
            if dT - p["q_w"] * u0 < 0.0:
                return False, int(t0), "ice is melting (water-side flux exceeds the loss)"
    else:
        cap, avail = p.get("cap_mol_m3", 0.0), p.get("co2_avail")
        if cap and avail is None and p.get("supply_required", True):
            return False, int(t0), "no supply row for the room (V2)"
    return True, None, ""


register(Form("front", "EXACT", front_advance, front_observe, front_flux_keys,
              valid=front_valid))


# ─────────────────────────────────────────────────── 4.6 POWER LAW ──
YEAR_S = 31557600.0


def wetness(T, RH, rh_min=80.0, t_min=0.0):
    """ISO 9223 time of wetness: 1 only when RH > 80 % and T > 0 C."""
    return 1.0 if (RH > rh_min and T > t_min) else 0.0


def powerlaw_advance(state, params, drives, t0, t1):
    """D = r tau^b by the EQUIVALENT-TIME form: Z = D^(1/b) grows linearly,
    dZ/dt = r^(1/b) w(T, RH). Moving a thing to a dry cellar changes w and not
    D, so the process is a semigroup for any r."""
    p = _P(params)
    b, r = p["b"], p["r"]                      # r in um / year^b
    D0 = state[0]
    D = max(D0, p.get("floor", 1e-9))
    Z = D ** (1.0 / b)
    dT_, dR = drives.get("T"), drives.get("RH")
    rhm, tm = p.get("rh_min", 80.0), p.get("t_min", 0.0)
    used = {k: v for k, v in (("T", dT_), ("RH", dR)) if v is not None}
    cuts = []
    if dT_ is not None:
        cuts += dT_.level_ticks(t0, t1, tm)
    if dR is not None:
        cuts += dR.level_ticks(t0, t1, rhm)
    pts = pieces(used, t0, t1, _grid({"grid": p.get("grid", 0), "dlnk_dT": 0.05}, used)
                 if used else 0, cuts)
    wet = 0.0
    for a, c in zip(pts[:-1], pts[1:]):
        m = 0.5 * (a + c)
        T = dT_.value(m) if dT_ is not None else p.get("T", 20.0)
        RH = dR.value(m) if dR is not None else p.get("RH", 90.0)
        wet += wetness(T, RH, rhm, tm) * (c - a) * TICK_S
    Z1 = Z + r ** (1.0 / b) * wet / YEAR_S
    D1 = Z1 ** b
    dD = D1 - D
    h = p.get("area", 1.0)
    fe_g = dD * 1e-6 * p.get("rho_fe", 7870.0) * 1e3 * h        # g of iron turned to rust
    return (D1,), {("mass", "fe", "rust"): fe_g,
                   ("mass", "o2", "metal"): fe_g * p.get("o2_per_fe", 0.43),
                   ("mass", "h2o", "metal"): fe_g * p.get("h2o_per_fe", 0.0),
                   ("E", "reaction", "air"): fe_g * p.get("heat_J_per_g", 0.0),
                   ("vol", "rust", "host"): dD * 1e-6 * h * (p.get("expand", 3.0) - 1.0)}


def powerlaw_observe(state, params, key):
    return state[0]


def powerlaw_flux_keys(params):
    return (("mass", "fe", "rust"), ("mass", "o2", "metal"), ("mass", "h2o", "metal"),
            ("E", "reaction", "air"), ("vol", "rust", "host"))


register(Form("powerlaw", "QUAD", powerlaw_advance, powerlaw_observe, powerlaw_flux_keys))


# ───────────────────────────────────────────────────── 4.7 DRYING ──
def drying_series(deff_t, l, terms=60):
    """MR = sum over odd m of 8/(m^2 pi^2) exp(-m^2 pi^2 Deff t/(4 l^2)).
    For x = Deff t / l^2 below 0.05 the series converges too slowly (its tail is
    1/m^2) and is replaced by the early-time form 1 - 2 sqrt(x/pi), whose
    error is of order erfc(1/sqrt(x)) (1e-10 at x = 0.05)."""
    x = deff_t / (l * l)
    if x <= 0.0:
        return 1.0
    if x < 0.05:
        return 1.0 - 2.0 * math.sqrt(x / math.pi)
    return drying_modes(x, terms)


def drying_modes(x, terms=60):
    """The odd-mode series itself at x = Deff t / l^2."""
    m = np.arange(1, 2 * terms, 2, dtype=float)
    return float(np.sum(8.0 / (m * m * math.pi ** 2) * np.exp(-(m * m) * math.pi ** 2 * x / 4.0)))


def drying_state(X, p):
    """The (X, w2) state for a moisture X already in the falling-rate period:
    w2 is the Deff-time that gives this moisture ratio (bisection on the series)."""
    mr = (X - p["X_eq"]) / (p["X_c"] - p["X_eq"])
    if mr >= 1.0:
        return (X, 0.0)
    lo, hi = 0.0, 1.0
    while drying_series(hi, p["l"]) > mr:
        hi *= 2.0
        if hi > 1e30:
            break
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if drying_series(mid, p["l"]) > mr:
            lo = mid
        else:
            hi = mid
    return (X, 0.5 * (lo + hi))


def drying_advance(state, params, drives, t0, t1):
    """state = (X, w2): moisture content and the Deff-time spent in period 2.
    Period 1: dX/dt = -rate_c (linear, EXACT). Period 2: X = X_eq + (X_c - X_eq)
    MR(w2) from the two-sided slab series. D of bound water in wood varies by
    about a decade with moisture, so this is a STANDIN; valid() fails when the
    caller says D varies by more than 20 %."""
    p = _P(params)
    X, w2 = state
    X0 = X
    Xc, Xeq, rate_c = p["X_c"], p["X_eq"], p["rate_c"]
    dt = (t1 - t0) * TICK_S
    if X > Xc:
        t_to = (X - Xc) / rate_c if rate_c > 0 else float("inf")
        if dt < t_to:
            X1 = X - rate_c * dt
            return (X1, 0.0), _dry_flux(X0, X1, p)
        dt -= t_to
        X, w2 = Xc, 0.0
    t_enter = int(t1) - int(round(dt / TICK_S))
    w2 += deff_int(p, drives, t_enter, int(t1))
    X1 = Xeq + (Xc - Xeq) * drying_series(w2, p["l"])
    return (X1, w2), _dry_flux(X0, X1, p)


def _dry_flux(X0, X1, p):
    mass = p.get("dry_g", 1.0)
    return {("mass", "h2o", "vapor"): (X0 - X1) * mass,
            ("E", "latent", "air"): (X0 - X1) * mass * p.get("latent_J_g", 2450.0)}


def drying_observe(state, params, key):
    if key in ("X", "x", "moisture"):
        return state[0]
    p = _P(params)
    if key == "MR":
        return (state[0] - p["X_eq"]) / (p["X_c"] - p["X_eq"])
    raise KeyError(key)


def drying_flux_keys(params):
    return (("mass", "h2o", "vapor"), ("E", "latent", "air"))


def drying_valid(state, params, drives, t0, t1=None):
    p = _P(params)
    if p.get("Dratio", 1.0) > 1.2:
        return False, int(t0), "D varies by more than 20%: use the similarity table"
    return True, None, ""


register(Form("drying", "STANDIN", drying_advance, drying_observe, drying_flux_keys,
              valid=drying_valid))


# ────────────────────────────────────────── 4.8 DEGREE-DAYS (a clock) ──
def clipped_sine_integral(mean, amp, period_s, phase_s, lo, hi, ta_s, tb_s):
    """Integral over [ta, tb] (seconds) of clip(mean + amp sin(theta), lo, hi),
    in unit-seconds, in closed form (arcsin crossings, antiderivatives)."""
    w = 2.0 * math.pi / period_s
    return (_clip_theta_int(w * (tb_s - phase_s), mean, amp, lo, hi)
            - _clip_theta_int(w * (ta_s - phase_s), mean, amp, lo, hi)) / w


def _clip_theta_int(theta, mean, amp, lo, hi):
    """Integral over theta from 0 to `theta` of clip(mean + amp sin, lo, hi)."""
    two = 2.0 * math.pi
    n = math.floor(theta / two)
    rem = theta - n * two
    return n * _clip_period(mean, amp, lo, hi) + _clip_partial(rem, mean, amp, lo, hi)


def _clip_breaks(mean, amp, lo, hi):
    br = [0.0, 2.0 * math.pi]
    if amp != 0.0:
        for level in (lo, hi):
            s = (level - mean) / amp
            if -1.0 < s < 1.0:
                a = math.asin(s)
                br += [a % (2.0 * math.pi), (math.pi - a) % (2.0 * math.pi)]
    return sorted(set(br))


def _seg_int(a, b, mean, amp, lo, hi):
    m = 0.5 * (a + b)
    v = mean + amp * math.sin(m)
    if v <= lo:
        return lo * (b - a)
    if v >= hi:
        return hi * (b - a)
    return mean * (b - a) - amp * (math.cos(b) - math.cos(a))


def _clip_period(mean, amp, lo, hi):
    br = _clip_breaks(mean, amp, lo, hi)
    return sum(_seg_int(a, b, mean, amp, lo, hi) for a, b in zip(br[:-1], br[1:]))


def _clip_partial(rem, mean, amp, lo, hi):
    br = [b for b in _clip_breaks(mean, amp, lo, hi) if b < rem] + [rem]
    br = sorted(set([0.0] + br))
    return sum(_seg_int(a, b, mean, amp, lo, hi) for a, b in zip(br[:-1], br[1:]))


def gdd_advance(state, params, drives, t0, t1):
    """GDD = integral of max(0, min(T, T_cap) - T_base) dt, in Celsius-days."""
    p = _P(params)
    base, cap = p["T_base"], p.get("T_cap", 1e9)
    d = drives["T"]
    if isinstance(d, SineDrive):
        inte = clipped_sine_integral(d.mean_, d.amp, d.period_s, d.phase_s, base, cap,
                                     t0 * TICK_S, t1 * TICK_S) - base * (t1 - t0) * TICK_S
    else:
        cuts = d.level_ticks(t0, t1, base) + d.level_ticks(t0, t1, cap)
        pts = pieces({"T": d}, t0, t1, _grid({"grid": p.get("grid", 0), "dlnk_dT": 0.1},
                                             {"T": d}), cuts)
        rate = lambda T: max(min(T, cap) - base, 0.0)             # noqa: E731
        inte = sum(_rate_int(rate, d, a, b, d.smooth) for a, b in zip(pts[:-1], pts[1:]))
    return (state[0] + inte / 86400.0,), {}


def gdd_observe(state, params, key):
    return state[0]


register(Form("rate_integral", "QUAD", gdd_advance, gdd_observe, lambda p: ()))


# ──────────────────────────────────── 4.9 BORBELY TWO-PROCESS (a form) ──
class TwoProcessRows:
    """H, L, a, phi and the time constants. All (u); BLOCKING for any scene."""
    H, L, a, phi_h = 0.60, 0.17, 0.10, 0.0
    tau_i_h, tau_d_h = 18.2, 4.2


def _tp(p):
    return (p.get("H", 0.60), p.get("L", 0.17), p.get("a", 0.10), p.get("phi_h", 0.0),
            p.get("tau_i_h", 18.2), p.get("tau_d_h", 4.2))


def two_process_S(S0, awake, dt_h, tau_i, tau_d):
    return (1.0 - (1.0 - S0) * math.exp(-dt_h / tau_i)) if awake else S0 * math.exp(-dt_h / tau_d)


def two_process_C(t_tick, phi_h):
    return math.sin(2.0 * math.pi * (t_tick * TICK_S / 3600.0 - phi_h) / 24.0)


def _tp_switch(S0, awake, tA, p, t_max):
    """First tick in (tA, t_max] where S meets the C-modulated line, or None.
    Vectorised scan on a 1-minute grid, then bisection to one tick."""
    H, L, a, phi, ti, td = _tp(p)
    line = (lambda t: H + a * two_process_C(t, phi)) if awake else \
        (lambda t: L + a * two_process_C(t, phi))

    def gap(t):                      # positive once the switch has happened
        S = two_process_S(S0, awake, (t - tA) * TICK_S / 3600.0, ti, td)
        return (S - line(t)) if awake else (line(t) - S)
    if gap(tA) >= 0.0:
        return tA
    step = 2400                       # 1 minute
    t = tA
    while t < t_max:
        tn = min(t + step, t_max)
        if gap(tn) >= 0.0:
            lo, hi = t, tn
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if gap(mid) >= 0.0:
                    hi = mid
                else:
                    lo = mid
            return hi
        t = tn
    return None


def two_process_advance(state, params, drives, t0, t1):
    """state = (S, awake). Runs the sleep/wake switches that happen in (t0, t1]."""
    p = _P(params)
    H, L, a, phi, ti, td = _tp(p)
    S, awake = state[0], bool(state[1])
    t = int(t0)
    switches = 0
    while t < t1:
        ts = _tp_switch(S, awake, t, p, t1)
        if ts is None or ts > t1:
            S = two_process_S(S, awake, (t1 - t) * TICK_S / 3600.0, ti, td)
            t = t1
            break
        S = two_process_S(S, awake, (ts - t) * TICK_S / 3600.0, ti, td)
        awake = not awake
        t = ts
        switches += 1
        if switches > 10000:
            break
    return (S, 1.0 if awake else 0.0), {}


def two_process_observe(state, params, key):
    if key == "S":
        return state[0]
    if key == "awake":
        return state[1]
    raise KeyError(key)


def two_process_cross(state, params, drives, t0, key, level, t_max):
    """key 'switch': the next sleep-onset or wake tick (one-tick accuracy)."""
    if key != "switch":
        return cross_scan(FORMS["two_process"], state, params, drives, t0, key, level, t_max)
    return _tp_switch(state[0], bool(state[1]), int(t0), _P(params), t_max)


register(Form("two_process", "QUAD", two_process_advance, two_process_observe,
              lambda p: (), cross=two_process_cross))


# ────────────────────────────────────── 4.10 THE THERMAL NETWORK ──
class NetSpec:
    """An RC network, plain data. C (n,) J/K; links i<j with conductance
    g (W/K); Gb (n,) W/K to the boundary; rad (n,) W/K^4 for radiation to the
    boundary (re-linearised); q (n,) W of constant source. `dyn` (NetDyn or
    None): the links that depend on temperature, when g and Gb hold only the
    FIXED part and `at(T, tb)` returns the linear net read at T; the advance
    and the steady state follow those links through the step (_nl_piece,
    _steady_nl). Without dyn the net is linear and the closed form is exact."""
    __slots__ = ("C", "rows", "cols", "g", "Gb", "rad", "q", "dyn", "__weakref__")

    def __init__(self, C, rows, cols, g, Gb, q=None, rad=None, dyn=None):
        self.dyn = dyn
        self.C = np.asarray(C, float)
        self.rows = np.asarray(rows, np.int64)
        self.cols = np.asarray(cols, np.int64)
        self.g = np.asarray(g, float)
        n = len(self.C)
        self.Gb = np.asarray(Gb, float) if Gb is not None else np.zeros(n)
        self.q = np.zeros(n) if q is None else np.asarray(q, float)
        self.rad = np.zeros(n) if rad is None else np.asarray(rad, float)

    @property
    def n(self):
        return len(self.C)

    def laplacian_dense(self):
        n = self.n
        A = np.zeros((n, n))
        for i, j, g in zip(self.rows, self.cols, self.g):
            A[i, i] += g
            A[j, j] += g
            A[i, j] -= g
            A[j, i] -= g
        return A

    def matvec_lap(self, x):
        d = self.g * (x[self.rows] - x[self.cols])
        return (np.bincount(self.rows, d, self.n) - np.bincount(self.cols, d, self.n))

    def stored(self, T):
        return float(np.dot(self.C, T))

    def at(self, T, tb):
        """The same net with its TEMPERATURE-DEPENDENT links read at the node
        temperatures T and the boundary tb (a plain linear net, dyn = None).
        With no dyn this is the net itself."""
        if self.dyn is None:
            return self
        gl, gbx = dyn_eval(self.dyn, np.asarray(T, float), float(tb), len(self.g), self.n)
        return NetSpec(self.C, self.rows, self.cols, self.g + gl, self.Gb + gbx,
                       q=self.q, rad=self.rad)


class NetDyn:
    """The temperature-dependent faces of a net, PLAIN DATA (arrays and floats,
    no callables). `g` and `Gb` of the NetSpec hold the FIXED part; each face
    here adds to its link (`link` >= 0, between nodes na and nb) or, for a face
    onto the boundary (`nb` < 0), to the node's Gb. Three laws, the lattice's
    own, read at the node temperatures:
      kind 0  gas | gas       G = harmonic(ka, kb) dx, k = k_air + eddy (T - T0)+
      kind 1  surface | gas   G = (H_CONV + radiation of hot gas) c;  side = a is the gas
      kind 2  buoyant rise    G = min(v dx^-1, cap) c_air  while the lower node is the
                              hotter, v = sqrt(g dx dT / T); side = a is the lower node;
                              c = the air's heat capacity of the cell, J/K
    `c` is the face's constant (dx, A af af, or c_air)."""
    __slots__ = ("kind", "link", "na", "nb", "c", "side", "k")

    def __init__(self, kind, link, na, nb, c, side, k):
        self.kind = np.asarray(kind, np.int8)
        self.link = np.asarray(link, np.int64)
        self.na = np.asarray(na, np.int64)
        self.nb = np.asarray(nb, np.int64)
        self.c = np.asarray(c, float)
        self.side = np.asarray(side, bool)
        self.k = dict(k)            # constants: dx, k_air, eddy, t0, h_conv, eps_gas, sigma, grav, tick_s, rise_max


def dyn_face_g(d, T, tb):
    """G (W/K) of every dynamic face at node temperatures T and boundary tb."""
    kk = d.k
    Ta = T[d.na]
    Tc = np.where(d.nb >= 0, T[np.maximum(d.nb, 0)], tb)
    G = np.zeros(len(d.kind))
    m = d.kind == 0
    if m.any():
        ka = kk["k_air"] + kk["eddy"] * np.maximum(Ta[m] - kk["t0"], 0.0)
        kc = kk["k_air"] + kk["eddy"] * np.maximum(Tc[m] - kk["t0"], 0.0)
        G[m] = 2.0 * ka * kc / np.maximum(ka + kc, 1e-12) * d.c[m]
    m = d.kind == 1
    if m.any():
        ga = d.side[m]
        Tg = np.where(ga, Ta[m], Tc[m])
        Ts = np.where(ga, Tc[m], Ta[m])
        tg, ts = Tg + K0, Ts + K0
        rad = np.where(Tg > Ts, kk["eps_gas"] * kk["sigma"] * (tg * tg + ts * ts) * (tg + ts), 0.0)
        G[m] = (kk["h_conv"] + rad) * d.c[m]
    m = d.kind == 2
    if m.any():
        lowa = d.side[m]
        lo = np.where(lowa, Ta[m], Tc[m])
        up = np.where(lowa, Tc[m], Ta[m])
        dT = lo - up
        v = np.sqrt(kk["grav"] * kk["dx"] * np.maximum(dT, 0.0) / (up + K0))
        share = np.minimum(v * (kk["tick_s"] / kk["dx"]), kk["rise_max"])
        G[m] = np.where(dT > 0.0, share / kk["tick_s"] * d.c[m], 0.0)
    return G


def dyn_eval(d, T, tb, nlinks, n):
    """(g per link, Gb per node): the dynamic part of the net at node
    temperatures T and boundary tb."""
    G = dyn_face_g(d, T, tb)
    internal = d.nb >= 0
    gl = np.bincount(d.link[internal], G[internal], nlinks) if internal.any() else np.zeros(nlinks)
    gbx = np.bincount(d.na[~internal], G[~internal], n) if (~internal).any() else np.zeros(n)
    return gl, gbx


_EIG = {}                            # (id of spec, Gb bytes) -> (lam, V, c_half)


def _eig_of(spec, gb_extra=None):
    Gb = spec.Gb if gb_extra is None else spec.Gb + gb_extra
    key = (id(spec), Gb.tobytes())
    got = _EIG.get(key)
    if got is not None and got[3] is spec:
        return got[:3]
    ch = np.sqrt(spec.C)
    A = spec.laplacian_dense() + np.diag(Gb)
    K = A / ch[:, None] / ch[None, :]
    K = 0.5 * (K + K.T)
    lam, V = np.linalg.eigh(K)
    if len(_EIG) > 32:
        _EIG.clear()
    _EIG[key] = (lam, V, ch, spec)
    return lam, V, ch


def _expm_phi(lam, h):
    e = np.exp(-lam * h)
    with np.errstate(divide="ignore", invalid="ignore"):
        phi1 = np.where(lam > 1e-300, -np.expm1(-lam * h) / np.where(lam > 1e-300, lam, 1.0), h)
        # phi2 = (h - phi1)/lam: weight of the linear part of the forcing
        phi2 = np.where(lam > 1e-300, (h - phi1) / np.where(lam > 1e-300, lam, 1.0), 0.5 * h * h)
    return e, phi1, phi2


def rc_eigen_step(spec, T0, f0, f1, h, gb_extra=None):
    """Exact step of C dT/dt = -A T + f(t) over h seconds for forcing LINEAR in
    time from f0 to f1 (nodal vectors, W). Exact for constant forcing
    (f1 = f0) and for piecewise-linear forcing."""
    lam, V, ch = _eig_of(spec, gb_extra)
    th0 = ch * T0
    y0 = V.T @ th0
    g0 = V.T @ (f0 / ch)
    g1 = V.T @ (f1 / ch)
    e, phi1, phi2 = _expm_phi(lam, h)
    y1 = e * y0 + phi1 * g0 + phi2 * (g1 - g0) / h
    return (V @ y1) / ch


def rc_steady(spec, Tb, q=None, rad_iter=40, x0=None):
    """T with A T = Gb Tb + q (+ radiation), the large-dt limit. Dense solve up
    to EIGEN_MAX nodes, Jacobi-preconditioned CG above that."""
    if spec.dyn is not None:
        return _steady_nl(spec, Tb, q, rad_iter)
    n = spec.n
    q = spec.q if q is None else q
    big = n > EIGEN_MAX
    if big:
        diag0 = (np.bincount(spec.rows, spec.g, n) + np.bincount(spec.cols, spec.g, n)
                 + spec.Gb)
    else:
        A = spec.laplacian_dense() + np.diag(spec.Gb)

    def solve(geff, rhs, x0):
        if not big:
            return np.linalg.solve(A + np.diag(geff), rhs)
        d = diag0 + geff
        M = spec.matvec_lap
        x = x0.copy()
        r = rhs - (M(x) + (spec.Gb + geff) * x)
        z = r / d
        p_ = z.copy()
        rz = float(np.dot(r, z))
        bn = max(float(np.linalg.norm(rhs)), 1e-300)
        for _ in range(4000):
            if np.linalg.norm(r) <= 1e-10 * bn:
                break
            Ap = M(p_) + (spec.Gb + geff) * p_
            al = rz / float(np.dot(p_, Ap))
            x += al * p_
            r -= al * Ap
            z = r / d
            rz2 = float(np.dot(r, z))
            p_ = z + (rz2 / rz) * p_
            rz = rz2
        return x
    zero = np.zeros(n)
    start = np.full(n, float(Tb)) if x0 is None else np.asarray(x0, float).copy()
    if not np.any(spec.rad):
        return solve(zero, spec.Gb * Tb + q, start)
    T = start
    for _ in range(rad_iter):
        geff = _rad_secant(spec.rad, T, Tb)
        Tn = solve(geff, (spec.Gb + geff) * Tb + q, T)
        done = np.max(np.abs(Tn - T)) < 1e-8
        T = Tn
        if done:
            break
    return T


NL_STEADY_ITER = 40
NL_STEADY_TOL = 1e-5                 # K


def _steady_nl(spec, Tb, q, rad_iter):
    """The steady state of a net with temperature-dependent links: a damped
    fixed-point on the links (each pass solves the linear net read at the last
    temperatures), each pass started from the last. Stops when no node moves
    NL_STEADY_TOL."""
    T = np.full(spec.n, float(Tb))
    for it in range(NL_STEADY_ITER):
        sp = spec.at(T, Tb)
        Tn = rc_steady(sp, Tb, q, rad_iter, x0=T)
        done = float(np.max(np.abs(Tn - T))) < NL_STEADY_TOL
        T = Tn if it == 0 else 0.5 * (T + Tn)
        if done:
            break
    return T


def _rad_secant(a, T_c, Tb_c):
    """G_eff such that G_eff (T - Tb) = a (T_K^4 - Tb_K^4), at T."""
    Tk, Bk = np.asarray(T_c, float) + K0, Tb_c + K0
    return a * (Tk * Tk + Bk * Bk) * (Tk + Bk)


def rc_cg_step(spec, T0, f0, f1, h, gb_extra=None, tol=1e-13, maxit=2000):
    """Crank-Nicolson step solved by Jacobi-preconditioned CG on the sparse
    network (numpy index arrays only)."""
    Gb = spec.Gb if gb_extra is None else spec.Gb + gb_extra
    diag = np.bincount(spec.rows, spec.g, spec.n) + np.bincount(spec.cols, spec.g, spec.n) + Gb

    def Ax(x):
        return spec.matvec_lap(x) + Gb * x
    M = spec.C / h + 0.5 * diag
    rhs = (spec.C / h) * T0 - 0.5 * Ax(T0) + 0.5 * (f0 + f1)

    def Mx(x):
        return (spec.C / h) * x + 0.5 * Ax(x)
    x = T0.copy()
    r = rhs - Mx(x)
    z = r / M
    pdir = z.copy()
    rz = float(np.dot(r, z))
    bn = max(float(np.linalg.norm(rhs)), 1e-300)
    for _ in range(maxit):
        if np.linalg.norm(r) <= tol * bn:
            break
        Mp = Mx(pdir)
        al = rz / float(np.dot(pdir, Mp))
        x += al * pdir
        r -= al * Mp
        z = r / M
        rz2 = float(np.dot(r, z))
        pdir = z + (rz2 / rz) * pdir
        rz = rz2
    return x


EIGEN_MAX = 256                        # nodes up to which the dense eigen path is used
CN_MAX_DT = 300.0                      # seconds
RAD_STEP_REL = 0.005                   # a radiative step changes T by at most this


def _forcing(spec, drives, a, b, tbname="Tb", qname="q"):
    """Nodal forcing (W) at ticks a and b: Gb Tb(t) + q scale(t)."""
    dtb = drives.get(tbname)
    dq = drives.get(qname)
    tb_a = dtb.value(a) if dtb is not None else 20.0
    tb_b = dtb.value(b) if dtb is not None else 20.0
    qa = dq.value(a) if dq is not None else 1.0
    qb = dq.value(b) if dq is not None else 1.0
    return (spec.Gb * tb_a + spec.q * qa), (spec.Gb * tb_b + spec.q * qb), tb_a, tb_b


def rc_advance_arrays(spec, T0, drives, t0, t1, solver="auto", tol=1e-7, rad_forcing=True):
    """Advance a net from tick t0 to t1. Returns (T1, fluxes). Exact for
    constant coefficients and piecewise-linear drives (eigen path, n <= 256);
    Crank-Nicolson with step doubling above that (STEP class, tol relative to
    the temperature swing). Radiative links are re-linearised per step."""
    T = np.asarray(T0, float).copy()
    E0 = spec.stored(T)
    dtb = drives.get("Tb")
    use_eigen = solver == "eigen" or (solver == "auto" and spec.n <= EIGEN_MAX)
    hasrad = bool(np.any(spec.rad))
    g = _net_grid(drives)
    pts = pieces({k: v for k, v in drives.items() if k in ("Tb", "q")}, t0, t1, g)
    src_J = 0.0
    qd = drives.get("q")
    hold = None
    for a, b in zip(pts[:-1], pts[1:]):
        h_total = (b - a) * TICK_S
        if h_total <= 0:
            continue
        qmean = qd.mean(a, b) if qd is not None else 1.0
        src_J += float(spec.q.sum()) * qmean * h_total
        if spec.dyn is not None:
            T, hold = _nl_piece(spec, T, drives, a, b, use_eigen, hold)
        elif hasrad:
            T = _rad_cell(spec, T, drives, a, b, use_eigen)
        else:
            f0, f1, _, _ = _forcing(spec, drives, a, b)
            if use_eigen:
                T = rc_eigen_step(spec, T, f0, f1, h_total)
            else:
                T = _cn_adaptive(spec, T, f0, f1, h_total, tol)
    E1 = spec.stored(T)
    out = src_J - (E1 - E0)                  # what went to the boundary (energy balance)
    return T, {("E", "boundary", "sink"): out, ("E", "source", "twin"): -src_J}


def _net_grid(drives):
    sm = [d for k, d in drives.items() if k in ("Tb", "q") and d.smooth]
    if not sm:
        return 0
    return 24000                           # 10 minutes: linear forcing error ~ 2e-4 of the swing


def _rad_cell(spec, T, drives, a, b, use_eigen):
    """A cell with radiative links: sub-steps so no step changes T by more than
    RAD_STEP_REL, each linearised (secant) at a predictor-corrector mean."""
    dtb, qd = drives.get("Tb"), drives.get("q")

    def tbv(t):
        return dtb.value(t) if dtb is not None else 20.0

    def qv(t):
        return spec.q * (qd.value(t) if qd is not None else 1.0)
    ta = a
    while ta < b:
        Tb = tbv(ta)
        geff = _rad_secant(spec.rad, T, Tb)
        net = (spec.Gb + geff) * (Tb - T) + qv(ta) - spec.matvec_lap(T)
        rate = float(np.max(np.abs(net / spec.C)))
        step_s = RAD_STEP_REL * (float(np.min(T)) + K0) / rate if rate > 0 else (b - ta) * TICK_S
        step_t = int(max(1, min(b - ta, math.floor(step_s / TICK_S))))
        tb_ = ta + step_t
        h = step_t * TICK_S
        Tb1 = tbv(tb_)
        fa, fb = spec.Gb * Tb + qv(ta), spec.Gb * Tb1 + qv(tb_)
        Tmid = 0.5 * (Tb + Tb1)
        g1 = _rad_secant(spec.rad, T, Tmid)
        Tp = _step_with_extra(spec, T, fa + g1 * Tb, fb + g1 * Tb1, h, g1, use_eigen)
        g2 = _rad_secant(spec.rad, 0.5 * (T + Tp), Tmid)
        T = _step_with_extra(spec, T, fa + g2 * Tb, fb + g2 * Tb1, h, g2, use_eigen)
        ta = tb_
    return T


NL_H0_S = 1.0                          # first step of a nonlinear piece, seconds
NL_REL = 0.3                           # a step may change a node's total conductance by this much
NL_PICARD = 8                          # passes of the mid-step fixed point
NL_MIN_TICKS = 8


def _nodal_g(sp):
    return (np.bincount(sp.rows, sp.g, sp.n) + np.bincount(sp.cols, sp.g, sp.n) + sp.Gb)


def _nl_piece(spec, T, drives, a, b, use_eigen, hold=None):
    """A piece [a, b) of a net whose links depend on temperature (spec.dyn):
    implicit steps, each solved as a linear net with the links read at the
    MEAN of the step (a fixed point, NL_PICARD passes), radiation secant-
    linearised in the same pass. The step starts at NL_H0_S, is halved while the
    nodal conductance moves more than NL_REL over it and doubled when it moves
    less than a sixth of that, so the plume that forms in seconds is followed
    and the hours after it are taken in a few large steps. `hold` is the step
    the last piece ended on (a later piece starts there). Returns (T, hold)."""
    dtb, qd = drives.get("Tb"), drives.get("q")

    def tbv(t):
        return dtb.value(t) if dtb is not None else 20.0

    def qv(t):
        return spec.q * (qd.value(t) if qd is not None else 1.0)
    ta = a
    hold = max(NL_MIN_TICKS, int(round(NL_H0_S / TICK_S))) if hold is None else hold
    while ta < b:
        step_t = int(min(b - ta, hold))
        while True:
            tb_ = ta + step_t
            h = step_t * TICK_S
            Tb0, Tb1 = tbv(ta), tbv(tb_)
            Tmid = 0.5 * (Tb0 + Tb1)
            q0, q1 = qv(ta), qv(tb_)
            Tend = T
            for _ in range(NL_PICARD):
                Tm = 0.5 * (T + Tend)
                sp = spec.at(Tm, Tmid)
                gr = _rad_secant(spec.rad, Tm, Tmid)
                Tn = _step_with_extra(sp, T, sp.Gb * Tb0 + q0 + gr * Tb0,
                                      sp.Gb * Tb1 + q1 + gr * Tb1, h, gr, use_eigen)
                moved = float(np.max(np.abs(Tn - Tend)))
                Tend = Tn
                if moved < 1e-6:
                    break
            g0 = _nodal_g(spec.at(T, Tb0)) + _rad_secant(spec.rad, T, Tb0)
            g1 = _nodal_g(spec.at(Tend, Tb1)) + _rad_secant(spec.rad, Tend, Tb1)
            r = float(np.max(np.abs(g1 - g0) / np.maximum(0.5 * (g0 + g1), 1e-300)))
            if r > NL_REL and step_t > NL_MIN_TICKS:
                step_t = max(NL_MIN_TICKS, step_t // 2)
                continue
            break
        T = Tend
        cut_short = step_t < hold and tb_ == b and r <= NL_REL    # the piece ended it, not the control
        ta = tb_
        if cut_short:
            continue
        if r < NL_REL / 6.0:
            hold = hold * 2
        elif r > NL_REL:
            hold = max(NL_MIN_TICKS, step_t)
        else:
            hold = step_t
    return T, hold


def _step_with_extra(spec, T, f0, f1, h, gb_extra, use_eigen):
    if use_eigen:
        return rc_eigen_step(spec, T, f0, f1, h, gb_extra)
    return rc_cg_step(spec, T, f0, f1, h, gb_extra)


def _cn_adaptive(spec, T, f0, f1, h_total, tol):
    """Crank-Nicolson over h_total seconds with step doubling error control,
    dt up to CN_MAX_DT. Returns T at the end."""
    t = 0.0
    h = min(CN_MAX_DT, h_total)
    swing = max(float(np.ptp(T)), 1.0)
    while t < h_total - 1e-12:
        h = min(h, h_total - t, CN_MAX_DT)
        fa = f0 + (f1 - f0) * (t / h_total)
        fm = f0 + (f1 - f0) * ((t + 0.5 * h) / h_total)
        fb = f0 + (f1 - f0) * ((t + h) / h_total)
        T1 = rc_cg_step(spec, T, fa, fb, h)
        Th = rc_cg_step(spec, T, fa, fm, 0.5 * h)
        T2 = rc_cg_step(spec, Th, fm, fb, 0.5 * h)
        err = float(np.max(np.abs(T1 - T2))) / swing
        if err <= tol or h <= 1.0:
            T = T2 + (T2 - T1) / 3.0           # Richardson (CN is second order)
            t += h
            if err < 0.1 * tol:
                h = min(h * 2.0, CN_MAX_DT)
        else:
            h *= 0.5
    return T


def lumped_rad_cool(T0_c, Tinf_c, a_over_C, t_s):
    """Exact lumped radiative cooling dT/dt = -a (T^4 - Tinf^4), T in kelvin.
    The implicit solution t(T) = |G(T) - G(T0)| / a with
    G = (1/(4 Ti^3)) ln|(T - Ti)/(T + Ti)| - (1/(2 Ti^3)) atan(T/Ti)
    is inverted by bisection. a_over_C is a/C in 1/(s K^3)."""
    T0, Ti = T0_c + K0, Tinf_c + K0
    if T0 == Ti:
        return T0_c

    def G(T):
        return (0.25 * math.log(abs((T - Ti) / (T + Ti))) - 0.5 * math.atan(T / Ti)) / Ti ** 3

    g0 = G(T0)

    def tof(x):
        T = Ti + (T0 - Ti) * x
        if T == Ti:
            return float("inf")
        return abs(G(T) - g0) / a_over_C
    lo, hi = 0.0, 1.0                    # x = (T - Ti)/(T0 - Ti); tof falls as x rises
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if tof(mid) > t_s:
            lo = mid
        else:
            hi = mid
    return Ti + (T0 - Ti) * 0.5 * (lo + hi) - K0


def rc_advance(state, params, drives, t0, t1):
    p = _P(params)
    spec = p["net"]
    T, fl = rc_advance_arrays(spec, np.asarray(state, float), drives, t0, t1,
                              solver=p.get("solver", "auto"), tol=p.get("tol", 1e-7))
    return tuple(float(v) for v in T), fl


def rc_observe(state, params, key):
    p = _P(params)
    spec = p["net"]
    if key == "E":
        return spec.stored(np.asarray(state))
    if isinstance(key, tuple) and key[0] == "T":
        return state[key[1]]
    if key == "Tmax":
        return max(state)
    if key == "Tmean":
        return float(np.dot(spec.C, state) / spec.C.sum())
    raise KeyError(key)


def rc_flux_keys(params):
    return (("E", "boundary", "sink"), ("E", "source", "twin"))


def rc_valid(state, params, drives, t0, t1=None):
    p = _P(params)
    if p.get("needs_history"):
        return False, int(t0), "a hosted process needs its history"
    return True, None, ""


register(Form("rc_net", "STEP", rc_advance, rc_observe, rc_flux_keys, valid=rc_valid))
