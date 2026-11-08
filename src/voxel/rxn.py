"""REACTION ROWS (wave 13, T6): one row format, one engine, a catalogue.

A reaction is DATA: a Row with reactants and products in GRAMS PER GRAM OF KEY
(mass closes to 1e-9, checked at load), a heat, a rate, drivers, gates, an optional
equilibrium and an optional residue. No craft names in code. The row format also
holds the 8 legacy acid REACTIONS (legacy_rows) and the products of burning
(burn_row); burning keeps its own law and only shares the format for its products.

HEAT. Row.dH is JOULES RELEASED PER GRAM OF KEY (+ warms the cell). The T6 spec quotes
some catalogue heats as reaction enthalpies (calcine +1783, slake -1150); they are
stored here with the sign flipped. dH_hess(row) computes the same number from the
heats of formation, and check() demands 2 %.

RATES (Rate objects, value(T, x=None)): Const, Arrhenius k = A exp(-Ea/RT) (Ea 50 kJ/mol:
1.91 per 10 K near 300 K), Bigelow D(T) = D_ref 10^(-(T-T_ref)/z) (z 10 K at 60 C is
212 kJ/mol), Points (the legacy table np.interp(x, xs, ys); with Ea=None the numbers are
bit-for-bit today's).

DRIVERS multiply the rate: Linear, Power, MichaelisMenten, Step, PartialPressure (the
fixed background plus the store's excess, at the reference air density: 0.74 g/m3 of CO2
reads 41 Pa, 400 ppm), Area. GATES switch it: AwWindow, TAbove, TBelow.

KINDS: bulk (one cell), contact (a fluid beside a solid face: the acid law), dissolve
(capped by the solubility), gas_solid (a solid face beside a gas holding the partial
pressure: calcining, carbonating, rusting).

EQUILIBRIUM lnK = a - b/T on the gas quotient Q. Net rate = rate (1 - Q/K), then the
extent is CLAMPED to the amount that brings Q to K (one-variable bisection), so a small
voxel cannot cross equilibrium in one step and run backward in the next: no sign flip
across Q = K for any dt. First-order rows without equilibrium use the exact
m (1 - exp(-k dt)), so ten 1-s steps equal one 10-s step.

THE BOOKS. A row that makes or eats water or vapour books made['water'] (signed grams of
any water form) and made['latent'] (L0 times signed grams of VAPOUR). total_water minus
made['water'] and total_energy (E + L0 vapour - L_CONV20 ice - made['latent']) minus
the chemical heat released are then constant (ArrayState has both, tested through random
rows, freezes, boils and condensations).

ArrayState is the dict view tests (and the slow clock) use. The engine talks to any
state with: get, avail, add, heat, p, T, aw, area, made, room.
"""
import math
from typing import Optional

import numpy as np

from . import _standin, icephase, psychro
from .matterdata import (AMBIENT, ATOMIC, BACKGROUND, BURN_COMP, BURN_HEAT,
                         BURN_NAME_OF_MAT, CP, CW, DHF, H_SOL, L0, L_CONV20, M, PHASE,
                         R, T_REF_GAS)

WATER_FORMS = ("H2O", "H2O_g", "H2O_s")


# ---- rates ---------------------------------------------------------------------
class Const:
    def __init__(self, k):
        self.k = k

    def value(self, T, x=None):
        return self.k + 0.0 * np.asarray(T, np.float64)


class Arrhenius:
    """k = A exp(-Ea / (R T_K)); T in degC."""

    def __init__(self, A, Ea):
        self.A, self.Ea = A, Ea

    def value(self, T, x=None):
        return self.A * np.exp(-self.Ea / (R * (np.asarray(T, np.float64) + 273.15)))

    @classmethod
    def from_point(cls, k_ref, T_ref, Ea):
        """The row with k = k_ref at T_ref (degC) and activation energy Ea."""
        return cls(k_ref * math.exp(Ea / (R * (T_ref + 273.15))), Ea)


class Bigelow:
    """Thermal death: D(T) = D_ref 10^(-(T - T_ref)/z) seconds per decade; k = ln10 / D."""

    def __init__(self, D_ref, T_ref, z):
        self.D_ref, self.T_ref, self.z = D_ref, T_ref, z

    def D(self, T):
        return self.D_ref * 10.0 ** (-(np.asarray(T, np.float64) - self.T_ref) / self.z)

    def value(self, T, x=None):
        return math.log(10.0) / self.D(T)

    def Ea_equivalent(self, T_c):
        """The activation energy that gives the same local slope: R ln10 T^2 / z."""
        return R * math.log(10.0) * (T_c + 273.15) ** 2 / self.z


class Points:
    """A table read in straight lines: np.interp(x, xs, ys). With Ea None the numbers are
    exactly the table's; with Ea and T_ref the table is multiplied by the Arrhenius ratio."""

    def __init__(self, xs, ys, T_ref=None, Ea=None):
        self.xs, self.ys, self.T_ref, self.Ea = xs, ys, T_ref, Ea

    def value(self, T, x=None):
        base = np.interp(x, self.xs, self.ys)
        if self.Ea is None:
            return base
        Tk = np.asarray(T, np.float64) + 273.15
        return base * np.exp(-self.Ea / R * (1.0 / Tk - 1.0 / (self.T_ref + 273.15)))


# ---- drivers and gates -----------------------------------------------------------
class Linear:
    def __init__(self, sp, scale=1.0):
        self.sp, self.scale = sp, scale

    def value(self, st):
        return st.get(self.sp) * self.scale


class Power:
    def __init__(self, sp, n):
        self.sp, self.n = sp, n

    def value(self, st):
        return np.maximum(st.get(self.sp), 0.0) ** self.n


class MichaelisMenten:
    def __init__(self, sp, Km):
        self.sp, self.Km = sp, Km

    def value(self, st):
        m = np.maximum(st.get(self.sp), 0.0)
        return m / (self.Km + m)


class Step:
    def __init__(self, sp, thr):
        self.sp, self.thr = sp, thr

    def value(self, st):
        return (st.get(self.sp) >= self.thr).astype(np.float64)


class PartialPressure:
    """Pa of a gas: (B0 + excess / V) / M R T_ref, so the fixed background counts."""

    def __init__(self, sp):
        self.sp = sp

    def value(self, st):
        return st.p(self.sp)


class Area:
    def value(self, st):
        return st.area


class AwWindow:
    """Open only while the water activity (RH) is in [lo, hi]: carbonation 50-80 %."""

    def __init__(self, lo, hi):
        self.lo, self.hi = lo, hi

    def ok(self, st, T):
        a = np.asarray(st.aw, np.float64)
        return ((a >= self.lo) & (a <= self.hi)).astype(np.float64)


class TAbove:
    def __init__(self, T0):
        self.T0 = T0

    def ok(self, st, T):
        return (np.asarray(T, np.float64) > self.T0).astype(np.float64)


class TBelow:
    def __init__(self, T0):
        self.T0 = T0

    def ok(self, st, T):
        return (np.asarray(T, np.float64) < self.T0).astype(np.float64)


class Equilibrium:
    """lnK = a - b/T_K for the gas quotient Q = prod p_i^n_i (Pa; n > 0 for products)."""

    def __init__(self, a, b, powers):
        self.a, self.b, self.powers = a, b, dict(powers)

    def lnK(self, T):
        return self.a - self.b / (np.asarray(T, np.float64) + 273.15)


# ---- the row ---------------------------------------------------------------------
class Row:
    """A reaction row. Immutable by convention (nothing in the engine writes to it)."""

    def __init__(self, id, kind, key, reactants, products, dH, rate, drivers=(),
                 gates=(), equilibrium=None, residue=None, clock=1, first_order=False,
                 closed=True, spend=None, sat=None, legacy=None):
        self.id, self.kind, self.key = id, kind, key
        self.reactants, self.products = tuple(reactants), tuple(products)
        self.dH = dH
        self.rate = rate
        self.drivers, self.gates = tuple(drivers), tuple(gates)
        self.equilibrium, self.residue = equilibrium, residue
        self.clock, self.first_order, self.closed = clock, first_order, closed
        self.spend, self.sat, self.legacy = spend, sat, legacy

    def replace(self, **kw):
        """A copy with some fields changed (tests and scenes tune a catalogue row)."""
        f = dict(id=self.id, kind=self.kind, key=self.key, reactants=self.reactants,
                 products=self.products, dH=self.dH, rate=self.rate, drivers=self.drivers,
                 gates=self.gates, equilibrium=self.equilibrium, residue=self.residue,
                 clock=self.clock, first_order=self.first_order, closed=self.closed,
                 spend=self.spend, sat=self.sat, legacy=self.legacy)
        f.update(kw)
        return Row(**f)

    def nu(self):
        """Grams of each species gained per gram of key (negative = used)."""
        d = {}
        for sp, g in self.reactants:
            d[sp] = d.get(sp, 0.0) - g
        for sp, g in self.products:
            d[sp] = d.get(sp, 0.0) + g
        return d


def stoich(react, prod, key):
    """Grams per gram of key from mole counts: ({'CaCO3': 1}, {'CaO': 1, 'CO2': 1}, 'CaCO3').
    Mass closes to rounding because M is a sum of atomic masses."""
    mk = react[key] * M[key]
    return (tuple((sp, n * M[sp] / mk) for sp, n in react.items()),
            tuple((sp, n * M[sp] / mk) for sp, n in prod.items()))


def dH_hess(row):
    """J released per gram of key from the heats of formation (kJ/mol) and M; None if any
    species lacks one."""
    tot = 0.0
    for sp, g in row.reactants:
        if DHF.get(sp) is None:
            return None
        tot += g / M[sp] * DHF[sp]
    for sp, g in row.products:
        if DHF.get(sp) is None:
            return None
        tot -= g / M[sp] * DHF[sp]
    return 1000.0 * tot


def check(row, tol=1e-9):
    """(mass residual, relative Hess error or None). Raises ValueError for an open
    row (closed=True) whose mass does not close to tol."""
    res = abs(sum(g for _s, g in row.reactants) - sum(g for _s, g in row.products))
    if row.closed and res > tol:
        raise ValueError("row %s: mass residual %g" % (row.id, res))
    h = dH_hess(row)
    err = None if (h is None or not row.dH) else abs(row.dH - h) / abs(h)
    return res, err


# ---- state view for tests and the slow clock ----------------------------------------
class ArrayState:
    """A lump (a cell and the gas around it) as dict of arrays. Gas species hold EXCESS
    grams over the fixed background (matterdata.BACKGROUND), like GasStore.

    Water has three keys: H2O liquid, H2O_g vapour, H2O_s ice. E is the sensible energy
    from 20 C; T = 20 + E / C with C summed over the contents, so converting matter
    at constant E and adding the heat released is exactly energy-conserving."""

    def __init__(self, amounts, E=0.0, C_other=0.0, V_m3=1e-3, bg=None, aw=0.5,
                 area=1.0):
        self.a = {k: np.array(v, np.float64) for k, v in amounts.items()}
        shape = np.broadcast(*self.a.values()).shape if self.a else ()
        for k in self.a:
            self.a[k] = np.broadcast_to(self.a[k], shape).copy()
        self.shape = shape
        self.E = np.broadcast_to(np.asarray(E, np.float64), shape).copy()
        self.C_other = np.broadcast_to(np.asarray(C_other, np.float64), shape).copy()
        self.V = np.broadcast_to(np.asarray(V_m3, np.float64), shape).copy()
        self.bg = dict(BACKGROUND)
        if bg:
            self.bg.update(bg)
        self.aw = aw
        self.area = area
        self.x = None
        self.made = {"water": np.zeros(shape), "latent": np.zeros(shape)}
        self.chem = np.zeros(shape)

    # -- reading
    def get(self, sp):
        return self.a.get(sp, np.zeros(self.shape))

    def avail(self, sp):
        """What can be used up: the grams, plus the background of a gas."""
        v = self.get(sp)
        return v + self.bg.get(sp, 0.0) * self.V if PHASE[sp] == "g" else v

    def room(self, sp):
        return np.inf

    @property
    def C(self):
        c = self.C_other.copy()
        for sp, v in self.a.items():
            c = c + CP[sp] * v
        return c

    @property
    def T(self):
        return AMBIENT + self.E / self.C

    def conc(self, sp, dg=0.0):
        return self.bg.get(sp, 0.0) + (self.get(sp) + dg) / self.V

    def p(self, sp, dg=0.0):
        """Partial pressure, Pa, at the reference air density."""
        return self.conc(sp, dg) / M[sp] * R * T_REF_GAS

    # -- writing
    def add(self, sp, dg):
        if sp not in self.a:
            self.a[sp] = np.zeros(self.shape)
        self.a[sp] = self.a[sp] + dg

    def heat(self, J):
        self.E = self.E + J

    def book(self, water=0.0, latent=0.0):
        self.made["water"] = self.made["water"] + water
        self.made["latent"] = self.made["latent"] + latent

    # -- phase changes (the books of psychro and icephase, on a lump)
    def evaporate(self, g, Tv=None):
        """g > 0 evaporates liquid, g < 0 condenses vapour, both at Tv (default the
        lump's T). Clamped to what is there. Cell E -(L0 + cv (Tv-20)) g, gas E
        +cv (Tv-20) g: in a lump the net is -L0 g, and E + L0 vapour is unchanged."""
        g = np.asarray(g, np.float64)
        g = np.minimum(g, self.get("H2O"))
        g = np.maximum(g, -self.get("H2O_g"))
        Tv = self.T if Tv is None else Tv
        cell, gas, _st = psychro.evap_energy(Tv)
        self.add("H2O", -g)
        self.add("H2O_g", g)
        self.E = self.E - g * cell + g * gas
        return g

    def boil(self, T_b=100.0):
        """Boil what the heat above T_b can pay for: the steam leaves at T_b."""
        g = psychro.boil_g(self.C, self.T, self.get("H2O"), T_b)
        return self.evaporate(g, Tv=T_b)

    def freeze_melt(self, dT_nuc=0.0, dt=np.inf):
        """Pure water freezes or melts at constant enthalpy (icephase.settle)."""
        mw, mi = self.get("H2O"), self.get("H2O_s")
        Co = self.C - CW * mw - icephase.CI * mi
        r = icephase.settle(self.E, Co, mw, mi, 0.0, 0.0, dT_nuc, dt, icephase.PURE)
        self.a["H2O"], self.a["H2O_s"] = r.m_w, r.m_i
        self.E = r.E
        return r.frozen

    # -- ledgers
    def total_water(self):
        return sum(self.get(k) for k in WATER_FORMS)

    def total_energy(self):
        """E + L0 vapour - L_CONV20 ice - made['latent'] (T6 spec section 8)."""
        return (self.E + L0 * self.get("H2O_g") - L_CONV20 * self.get("H2O_s")
                - self.made["latent"])

    def water_identity(self):
        return self.total_water() - self.made["water"]

    def energy_identity(self):
        """total_energy minus the chemical heat released by rows: constant."""
        return self.total_energy() - self.chem


# ---- the engine -------------------------------------------------------------------
def _ln_q(row, st, xi):
    nu = row.nu()
    s = 0.0
    for sp, n in row.equilibrium.powers.items():
        p = st.p(sp, nu.get(sp, 0.0) * xi)
        s = s + n * np.log(np.maximum(p, 1e-300))
    return s


def _limits(row, st):
    """(largest forward extent, largest backward extent), grams of key, from the
    reactants present and the product room."""
    fwd = np.inf
    for sp, g in row.reactants:
        if g > 0.0:
            fwd = np.minimum(fwd, st.avail(sp) / g)
    bwd = np.inf
    for sp, g in row.products:
        if g > 0.0:
            bwd = np.minimum(bwd, st.avail(sp) / g)
            fwd = np.minimum(fwd, st.room(sp) / g)
    return np.maximum(fwd, 0.0), np.maximum(bwd, 0.0)


def extent(row, st, T=None, dt=1.0):
    """Grams of key reacted in dt (signed: negative = the reverse). Does not change
    the state."""
    T = st.T if T is None else T
    k = row.rate.value(T, st.x)
    gate = 1.0
    for g in row.gates:
        gate = gate * g.ok(st, T)
    drv = 1.0
    for d in row.drivers:
        drv = drv * d.value(st)
    fwd, bwd = _limits(row, st)
    if row.sat is not None:                      # dissolve: never pass the solubility
        bsat = _bsat(row.sat, T)
        cap = np.maximum(bsat * st.get("H2O") - st.get(row.products[0][0]), 0.0)
        fwd = np.minimum(fwd, cap / row.products[0][1])
    if row.first_order and row.equilibrium is None:
        ext = fwd * (-np.expm1(-k * drv * gate * dt))
        return np.minimum(ext, fwd)
    ext = k * drv * gate * dt
    if row.equilibrium is None:
        return np.minimum(ext, fwd)
    eq = row.equilibrium
    lnQ0 = _ln_q(row, st, 0.0)
    lnK = eq.lnK(T)
    ratio = np.exp(np.clip(lnQ0 - lnK, -50.0, 50.0))          # Q / K
    net = ext * np.clip(1.0 - ratio, -1.0, 1.0)
    forward = net >= 0.0
    xi_hi = np.where(forward, np.minimum(net, fwd), -np.minimum(-net, bwd))
    lnQ_hi = _ln_q(row, st, xi_hi)
    lo = np.zeros_like(xi_hi)
    hi = xi_hi.copy()
    # forward: the largest xi in [0, xi_hi] with lnQ <= lnK; backward: the most
    # negative xi in [xi_hi, 0] with lnQ >= lnK. lnQ rises with xi in both.
    crossed = np.where(forward, lnQ_hi > lnK, lnQ_hi < lnK)
    a, b = lo.copy(), hi.copy()                     # a: safe side, b: the far end
    for _ in range(40):
        mid = 0.5 * (a + b)
        lq = _ln_q(row, st, mid)
        safe = np.where(forward, lq <= lnK, lq >= lnK)
        a = np.where(safe, mid, a)
        b = np.where(safe, b, mid)
    return np.where(crossed, a, xi_hi)


def advance(rows, st, T=None, dt=1.0):
    """Run every row for dt seconds on a state view. Returns {row id: extent}.
    Reactants fall, products rise, the heat released goes to the cell, and water and
    vapour made or eaten are booked."""
    out = {}
    for row in rows:
        xi = extent(row, st, T, dt)
        nu = row.nu()
        for sp, g in nu.items():
            st.add(sp, g * xi)
        st.heat(row.dH * xi)
        st.chem = st.chem + row.dH * xi
        dw = sum(g for sp, g in nu.items() if sp in WATER_FORMS)
        dv = nu.get("H2O_g", 0.0)
        if dw or dv:
            st.book(water=dw * xi, latent=L0 * dv * xi)
        out[row.id] = xi
    return out


def _bsat(species, T):
    return _standin.solubility_load(species, T)


# ---- the catalogue ------------------------------------------------------------------
def _row(id, kind, react, prod, key, dH, rate, **kw):
    r, p = stoich(react, prod, key)
    return Row(id, kind, key, r, p, dH, rate, **kw)


def _calcine_rate():
    # 5 cm lump: 90 % calcined in about 12 h at 1000 C (u; kiln practice 12-48 h)
    return Arrhenius.from_point(math.log(10.0) / (12 * 3600.0), 1000.0, 190000.0)


def catalogue():
    """The catalogue rows by id. All numbers (u) unless the spec marks (c)."""
    rows = [
        # CaCO3 -> CaO + CO2. p_eq = 4.137e12 exp(-20474/T) Pa (Silcox 1989 via
        # Stanmore & Gilot 2005, u): 1 atm at 898 C, 40 Pa (open air) at 534 C.
        _row("calcine", "gas_solid", {"CaCO3": 1}, {"CaO": 1, "CO2": 1}, "CaCO3",
             -1783.0, _calcine_rate(), drivers=(Linear("CaCO3"),),
             equilibrium=Equilibrium(math.log(4.137e12), 20474.0, {"CO2": 1})),
        # CaO + H2O -> Ca(OH)2, +1150 J/g CaO: the heat boils the water.
        _row("slake", "bulk", {"CaO": 1, "H2O": 1}, {"Ca(OH)2": 1}, "CaO",
             1150.0, Const(0.1), first_order=True),
        # Ca(OH)2 + CO2 -> CaCO3 + H2O through a water film: best at 50-80 % RH.
        _row("carbonate", "gas_solid", {"Ca(OH)2": 1, "CO2": 1}, {"CaCO3": 1, "H2O": 1},
             "Ca(OH)2", 1539.0, Const(1.1e-4),
             drivers=(Area(), PartialPressure("CO2")), gates=(AwWindow(0.5, 0.8),)),
        # 2 Fe + 1.5 O2 + H2O -> 2 FeOOH; 25-50 micron in the first year at C3 (u)
        _row("rust", "gas_solid", {"Fe": 2, "O2": 1.5, "H2O": 1}, {"FeOOH": 2}, "Fe",
             7450.0, Const(9.2e-7), drivers=(Area(),),
             gates=(AwWindow(0.8, 1.0), TAbove(0.0))),
        _row("dissolve_nacl", "dissolve", {"NaCl": 1}, {"NaCl_aq": 1}, "NaCl",
             -H_SOL["NaCl"], Const(0.02), first_order=True, sat="NaCl"),
        _row("dissolve_sucrose", "dissolve", {"sucrose": 1}, {"sucrose_aq": 1}, "sucrose",
             -H_SOL["sucrose"], Const(0.02), first_order=True, sat="sucrose"),
        # glucose -> 2 ethanol + 2 CO2 (interface only; yeast is L9)
        _row("ferment", "bulk", {"glucose": 1}, {"ethanol": 2, "CO2": 2}, "glucose",
             384.0, Arrhenius.from_point(1e-5, 30.0, 50000.0), first_order=True),
        # C + CO2 -> 2 CO, 172 kJ/mol; K = 1 atm at about 700 C (u)
        _row("boudouard", "gas_solid", {"C": 1, "CO2": 1}, {"CO": 2}, "C",
             -14400.0, Arrhenius.from_point(1e-6, 900.0, 150000.0),
             drivers=(Linear("C"), PartialPressure("CO2")),
             equilibrium=Equilibrium(32.84, 20740.0, {"CO": 2, "CO2": -1})),
    ]
    return {r.id: r for r in rows}


def legacy_rows(REACTIONS=None, rate_pts=None):
    """Rows for the 8 legacy REACTIONS (acid eats solid), numbers unchanged: the rate is
    the legacy table itself (Points, Ea None: np.interp bit for bit), spend (ml of stock
    per g), heat (J/g) and residue are carried as they are. Not mass-closed on purpose
    (the eaten mass leaves as smoke today). Defaults read materials.py."""
    if REACTIONS is None:
        from . import materials
        REACTIONS, rate_pts = materials.REACTIONS, materials._RATE_PTS
    rows = []
    for (f, s), (_pts, spend, heat, residue) in REACTIONS.items():
        xs, ys = rate_pts[(f, s)]
        rows.append(Row("legacy:%d:%d" % (f, s), "contact", "solid%d" % s,
                        (("solid%d" % s, 1.0), ("fluid%d" % f, spend)), (), heat,
                        Points(xs, ys), residue=residue, closed=False, spend=spend,
                        legacy=(f, s)))
    return rows


def burn_row(mat, heat=None):
    """The products of burning one gram of fuel as a Row (kind bulk). `mat` is a name
    ('wood', 'flesh', 'leaf', 'char', 'rope') or a materials.py id. The elemental table
    gives CO2 (1 - char share of the carbon), soot as carbon, H2O from the hydrogen and the
    fuel's own water, ash; the O2 demand is whatever closes the mass (wood 1.31 against the
    repo's 1.33). `heat` J/g released is the repo's burn column, read as the LOWER heating
    value: the vapour carries stored latent L0, funded by made['latent'], so the cell
    still receives exactly the repo's J/g. _burn_cells decides how much burns and where."""
    name = BURN_NAME_OF_MAT.get(mat, mat) if not isinstance(mat, str) else mat
    c = BURN_COMP[name]
    soot = c["C"] * c["char"]
    co2 = (c["C"] - soot) * M["CO2"] / M["C"]
    h2o = c["H"] * M["H2O"] / (2.0 * ATOMIC["H"]) + c["water"]
    prods = [("CO2", co2), ("H2O_g", h2o), ("C", soot), ("ash", c["ash"])]
    prods = tuple((s, g) for s, g in prods if g > 0.0)
    o2 = sum(g for _s, g in prods) - 1.0
    if heat is None:
        heat = BURN_HEAT[name]
    return Row("burn:" + name, "bulk", name, ((name, 1.0), ("O2", o2)), prods, heat,
               Const(0.0), closed=True)


# ---- closed forms for the slow clock (T5) -------------------------------------------
def first_order_extent(m, k, t):
    """m (1 - exp(-k t)): a night skipped and a night simulated agree."""
    return np.asarray(m, np.float64) * (-np.expm1(-np.asarray(k, np.float64) * t))


def iso9223_depth_um(t_years, r_corr_um=37.0, b=0.55):
    """Corrosion depth, micron: D = r_corr t^b (ISO 9223; category C3, r_corr 25-50, b
    0.5-0.6, u)."""
    return r_corr_um * np.asarray(t_years, np.float64) ** b


def carbonation_depth_mm(t_s, k_mm_sqrt_day=1.0):
    """Carbonation front of lime mortar: about 1 mm per sqrt(day) (u)."""
    return k_mm_sqrt_day * np.sqrt(np.asarray(t_s, np.float64) / 86400.0)


def p_eq_calcine(T_c):
    """CO2 pressure over limestone, Pa: 4.137e12 exp(-20474/T)."""
    return 4.137e12 * np.exp(-20474.0 / (np.asarray(T_c, np.float64) + 273.15))
