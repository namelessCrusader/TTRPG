"""THE MATERIAL RECORD: one frozen row per material, in SI units (wave 13, T9).

Today the sim knows what a material IS in six places that do not agree
(materials.py, the particle table in src/mpm/core.py, the bond rows in
src/bonds/materials.py, the friction constant, the `edge` array, nothing for
damage). This module holds ONE row per material. The old tables become VIEWS
of it (`legacy_views`), and every place where two old tables disagree is
written down as a LEGACY OVERRIDE with its reason, so building the views moves
no number. Each override is later removed on purpose, one declared change at a
time.

Stand-alone: numpy and the standard library only. It does not import
materials.py (the molten fluid ids are passed in; materials.py imports THIS
file and builds its old tables from `legacy_views`) and it imports no other
wave 13 module at import: mixrules, species and icephase are found on first
use (`_sibling`), and read as None when they are absent.

WHAT IS HERE
  MatRec            the frozen row. Derived numbers are methods, never stored.
  REC, BY_ID, NMAT  the table of the 12 lattice materials (AIR is id 0 and has
                    a row), plus the bond-only rows (petiole, twig).
  MatTable          the same table as an immutable object. `ingest` (a plain
                    dict row, such as icephase.ICE_ROW) and `retire` (a retired
                    name keeps its id and names its successor, never renumbered)
                    return a NEW table. BASE is the shipped one.
  legacy_views      rebuilds SOLID, EMISS, SPAN, TOUGH, TSHOCK, MELT, FREEZE,
                    PYRO, O2_RATIO, SOOT, GLOW, the lookup arrays, the particle
                    table and the bond rows exactly as base18 has them.
  props(key, grain) the mechanical numbers as a plain dict (for T2 breaking and
                    T11 mechanisms).
  blend, porous     the mechanical numbers of a separate-phase composite and of
                    a porous solid.

SOURCES. Every number carries a tag in the comments: S2..S4 = the wood and
stone tables the repo already cites (docs/realism-references.md); B = the bond
rows (src/bonds/materials.py); M = the particle table (src/mpm/core.py); T = an
engineering handbook table (Engineering ToolBox); Y = checked against a source
in the T9 spec; (u) = independent but UNVERIFIED (from memory or an order of
magnitude). [I] = independent and checked against tests/data/sourced_numbers.json (lane A-S):
the wood and granite strengths and the carbon-steel derate rows carry it. Other
(u) numbers stay (u) until a source is read; a test that leans on one says so
in its name (test_u_*).
"""
import dataclasses
import importlib
import math
import numbers
from collections import namedtuple
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# THE THREE SIBLINGS ARE FOUND ON FIRST USE, NEVER AT IMPORT (wave 13a). materials.py
# imports this file to build its tables while it is itself still being built, and
# species and mixrules both read materials: finding them here, at import, would walk
# back into a half-built materials.py, and which module was imported first would
# decide whether it worked. This file still imports no other src module at import.
_FOUND = {}                            # sibling modules found so far


def _sibling(name):
    """A sibling wave 13 module, or None when it is not built or this file is read
    on its own: mixrules (T1: the one home of the thermal and mass rules), species
    (T1: species ids and component keys), icephase (T6: the ICE row is theirs, the
    record holds it). A module that cannot be imported is asked for again next time."""
    if name not in _FOUND:
        package = __name__.rpartition(".")[0]
        if not package:
            return None
        try:
            _FOUND[name] = importlib.import_module("." + name, package)
        except ImportError:
            return None
    return _FOUND[name]


def __getattr__(name):
    """`matrec._mixrules`, `matrec._species` and `matrec._icephase`, the names this
    file used to bind at import: the module, or None, looked up when asked."""
    if name in ("_mixrules", "_species", "_icephase"):
        return _sibling(name[1:])
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


# ── ids ─────────────────────────────────────────────────────────────────────
AIR, WOOD, STONE, IRON, FLESH, ASH, GLASS, LEAF, CHAR = 0, 1, 2, 3, 4, 5, 6, 7, 8
TIN, LEAD, ROPE = 9, 10, 11
ICE = 12                    # reserved: the first row ingested after the twelve (T6 owns the numbers)
LEGACY_NMAT = 12            # the ids the old tables know
NMAT = 12                   # rows of the shipped table (a MatTable has its own `nmat`)

# the three molten fluid ids are NOT imported (materials.py owns them): a
# caller passes them in, by these keys
MOLTEN_KEYS = ("mtin", "mlead", "miron")

MEMBER_E = 1.0e8            # Pa: the structural-member test (data, not a flag), spec 3.5
MEMBER_S = 1.0e6            # Pa

MU_CLASSES = ("rock", "wood", "steel", "softmetal", "glass", "skin", "fibre",
              "granular", "ice")          # matfric.py keeps the same list; a cross test checks

# ── legacy constants that belong to no material but that the old tables use ─
SKIN_T = (126.0, 1060.0)                 # s: materials.SKIN_T, the skin depth and path times (H-lane fit)
BOND_PETIOLE_R = 0.002                   # m: src/bonds/materials.py PETIOLE_R (a geometry, not a material)
BOND_LEAF_AREA = 0.04 / 30.0             # m2 per g of leaf: bonds LEAF_AREA (the BD prototype spray)

_SOLVER = ("solver tuning (damping, yield strain, rigid share, continuum strength): "
           "kept as the particle/bond solvers have it until measured (spec 5)")


# ── small records ───────────────────────────────────────────────────────────
@dataclass(frozen=True)
class BurnRec:
    """How a solid burns (the numbers FIRE owns; the record only holds them).
    ign_C None = does not light. o2_ratio None = not in the O2_RATIO table
    (ASH has 0.0, which is an entry). pyro = (L_g J/g, q_fl W/m2, O2 limit);
    glow = (O2 limit, share kept, kelvin to full rate) for charcoal."""
    ign_C: Optional[float] = None
    heat_Jg: float = 0.0
    rate_gm2s: float = 0.0
    o2_ratio: Optional[float] = None
    soot: Optional[float] = None
    pyro: Optional[tuple] = None
    glow: Optional[tuple] = None
    burn_area: Optional[float] = None


@dataclass(frozen=True)
class MeltRec:
    """Melting: the temperature, the key of the molten FLUID row, latent heat J/g."""
    T_C: float
    fluid_key: str
    latent_Jg: float


Eff = namedtuple("Eff", "E st sc sf tau sy H")


# ── the record ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class MatRec:
    """One material, in SI. Strengths are for CLEAN material at 20 C; the laws
    apply temperature (`derate`) and damage (`at`) on top. None means "this
    quantity does not exist for the material" (wood does not yield), 0.0 means
    "it exists and is nothing" (a rope carries no compression)."""
    key: str
    id: Optional[int]                 # None for a bond-only member (petiole, twig)
    name: str
    family: str
    # bulk and thermal
    rho: float = 0.0                  # kg/m3 (= g/L)
    cp: float = 0.0                   # J/(kg K)  (the old SOLID tuple holds J/(g K): /1000)
    k: float = 0.0                    # W/(m K)
    eps: float = 0.0                  # emissivity
    # elastic
    E: float = 0.0                    # Pa, along the strong axis
    E_perp: Optional[float] = None    # Pa across the grain (wood)
    nu: float = 0.0
    # strength (Pa)
    sy: Optional[float] = None        # yield; None = breaks before it yields
    st: float = 0.0                   # tension along the grain
    st_perp: Optional[float] = None
    sc: float = 0.0                   # compression
    sc_perp: Optional[float] = None
    sf: Optional[float] = None        # flexural rupture; None = use st
    tau: float = 0.0                  # shear
    ductility: float = 0.0            # strain at fracture; brittle = < 0.05
    dbtt_C: Optional[tuple] = None    # (T_C of the ductile-to-brittle step, width K)
    # thermal strain
    alpha: float = 0.0                # 1/K
    alpha_perp: Optional[float] = None
    # surface and fracture
    H: Optional[float] = None         # Pa, indentation hardness
    gimp: float = 0.0                 # J/m2: impact toughness (today's TOUGH x 1000)
    KIc: Optional[float] = None       # Pa sqrt(m)
    weibull_m: Optional[float] = None  # Weibull modulus of the strength scatter
    mu_class: str = "granular"        # key into the friction class table (matfric)
    transmit: float = 0.0             # light through ONE packed voxel of the stuff
    # time and temperature
    creep_n: Optional[float] = None
    t_R1_s: Optional[float] = None    # time to failure at utilisation 1; t0 = t_R1 (n + 1)
    rot_ratio: float = 1.0            # strength lost per mass lost to decay
    derate: tuple = ((20.0, 1.0, 1.0),)   # ((T_C, ks, kE), ...), T ascending, cold side included
    ring: Optional[float] = None      # SOUND ring fraction
    char_zero_layer_m: Optional[float] = None
    burn: Optional[BurnRec] = None
    melt: Optional[MeltRec] = None
    span_ref: int = 0                 # legacy SPAN in 10 cm voxels; removed with SPAN
    menu_name: Optional[str] = None   # what a hand calls it (materials.MATNAME); None = no entry
    successor: Optional[str] = None   # key of the row that replaces a retired one
    legacy: dict = field(default_factory=dict, compare=False, hash=False, repr=False)
    extra: dict = field(default_factory=dict, compare=False, hash=False, repr=False)
    src: str = ""                     # one line of provenance for the row as a whole

    # ---- derived (methods, never stored) ----
    def brittle(self):
        """Read from ductility, not set by a flag."""
        return self.ductility < 0.05

    def sf_eff(self):
        """Flexural strength; the tension strength when none is given."""
        return self.st if self.sf is None else self.sf

    def R_shock(self):
        """Thermal shock parameter R = sf (1 - nu) / (E alpha), kelvin.
        Infinite for a material that does not expand."""
        d = self.E * self.alpha
        if d <= 0.0:
            return math.inf
        return self.sf_eff() * (1.0 - self.nu) / d

    def tshock_face(self):
        """2 R, the face-to-body difference a one-sided quench carries (K).
        Reported and compared with TSHOCK; not a law limit until W7b passes."""
        return 2.0 * self.R_shock()

    def structural(self):
        """May this material be a beam or column member? Read from numbers."""
        return (self.E >= MEMBER_E and self.sc >= MEMBER_S
                and self.sf_eff() >= MEMBER_S)

    def derate_at(self, T_C):
        """(ks, kE) at temperature T_C (scalar or array); flat outside the table."""
        xs = np.array([p[0] for p in self.derate], float)
        ks = np.interp(T_C, xs, np.array([p[1] for p in self.derate], float))
        ke = np.interp(T_C, xs, np.array([p[2] for p in self.derate], float))
        return ks, ke

    def at(self, T=20.0, D=0.0, temper=1.0):
        """Effective numbers at temperature T (C), damage D and the thing's
        heat-treatment multiplier `temper` (default 1).

          strength s_eff = s ks(T) (1 - D)         tension, bending, shear, yield
          sc_eff         = sc ks(T) (1 - D)^0.5    a crack closes under compression
          E_eff          = E kE(T) (1 - D)
          sy_eff, H_eff  also times `temper` (heat treatment is a THING property)
          H_eff          also falls with ks(T) (a hot tool is soft), not with D

        `sf` is reported as sf_eff() (the tension value when none is given).
        A material without a yield or a hardness returns None for it.
        Works on scalars and on arrays of T and D."""
        Tn = np.asarray(T, float)
        Dn = np.clip(np.asarray(D, float), 0.0, 1.0)
        ks, ke = self.derate_at(Tn)
        t = 1.0 - Dn
        c = np.sqrt(t)
        scalar = Tn.ndim == 0 and Dn.ndim == 0

        def out(v):
            return float(v) if scalar else v
        sy = None if self.sy is None else out(self.sy * ks * t * temper)
        H = None if self.H is None else out(self.H * ks * temper)
        return Eff(E=out(self.E * ke * t), st=out(self.st * ks * t),
                   sc=out(self.sc * ks * c), sf=out(self.sf_eff() * ks * t),
                   tau=out(self.tau * ks * t), sy=sy, H=H)

    def brittle_at(self, T_C, tough=1.0):
        """(ductility, gimp) at T_C: iron falls off a ductile-to-brittle step
        centred on dbtt_C[0] with the stated width in kelvin (below it the
        ductility is 0.02 and gimp 0.1 of the warm value); everything else is
        flat.

        `tough` is the heat-history multiplier (heathist.toughness_factor: fresh hard
        martensite is brittle, tempering restores it, a coarse grain lowers it). It scales
        both numbers; at the default 1.0 the answer is the old one bit for bit."""
        if tough != 1.0:
            d, g = self.brittle_at(T_C)
            return d * tough, g * tough
        if self.dbtt_C is None:
            return self.ductility, self.gimp
        t0, w = self.dbtt_C
        x = np.clip((np.asarray(T_C, float) - (t0 - 0.5 * w)) / max(w, 1e-9), 0.0, 1.0)
        s = x * x * (3.0 - 2.0 * x)              # 0 below the step, 1 above it, smooth across
        duct = 0.02 + (self.ductility - 0.02) * s
        g = self.gimp * (0.1 + 0.9 * s)
        if np.ndim(T_C) == 0:
            return float(duct), float(g)
        return duct, g


def _ov(rec, group, name, default):
    """A legacy override of `rec` for `group` ("mpm", "bonds", "voxel"): the
    value, or `default` when the record has none. Overrides are stored as
    {name: (value, reason)}."""
    e = rec.legacy.get(group, {}).get(name)
    return default if e is None else e[0]


# ── the data rows (typed by hand, with sources) ─────────────────────────────
_PY = 15.0 / 21.0           # the O2 limit of flaming, as materials.py spells it

_WOOD_DERATE = ((-20.0, 1.05, 1.05), (20.0, 1.0, 1.0), (100.0, 0.75, 0.75),
                (200.0, 0.50, 0.50), (300.0, 0.20, 0.20), (350.0, 0.0, 0.0))   # (u) order of magnitude, Wood Handbook
_STONE_DERATE = ((-20.0, 1.05, 1.05), (20.0, 1.0, 1.0), (100.0, 1.0, 1.0),
                 (200.0, 0.95, 0.95), (300.0, 0.85, 0.85), (400.0, 0.75, 0.75),
                 (500.0, 0.60, 0.60), (600.0, 0.45, 0.45), (700.0, 0.30, 0.30),
                 (800.0, 0.15, 0.15), (900.0, 0.08, 0.08), (1000.0, 0.04, 0.04))   # (u) EN 1992-1-2 siliceous concrete as a stand-in
_IRON_DERATE = ((-20.0, 1.10, 1.0), (20.0, 1.0, 1.0), (100.0, 1.0, 1.0),
                (200.0, 1.0, 0.90), (300.0, 1.0, 0.80), (400.0, 1.0, 0.70),
                (500.0, 0.78, 0.60), (600.0, 0.47, 0.31), (700.0, 0.23, 0.13),
                (800.0, 0.11, 0.09), (900.0, 0.06, 0.0675), (1000.0, 0.04, 0.045),
                (1100.0, 0.02, 0.0225), (1200.0, 0.0, 0.0))   # [I] EN 1993-1-2 Table 3.1, k_y and k_E (A-S, steel.strength_retention): identical to every row from 20 C up; the cold gain (-20 C) is (u)
_GLASS_DERATE = ((-20.0, 1.0, 1.0), (20.0, 1.0, 1.0), (400.0, 1.0, 1.0),
                 (500.0, 0.6, 0.6), (550.0, 0.3, 0.3), (720.0, 0.02, 0.02),
                 (1000.0, 0.0, 0.0))   # (u) annealing 550 C, softening 720 C; above, the viscosity law (L6) takes over
_FLAT = ((-20.0, 1.0, 1.0), (20.0, 1.0, 1.0))


def _rows():
    """The twelve lattice rows, AIR first, then the bond-only rows."""
    R = []
    add = R.append
    add(MatRec(
        key="air", id=AIR, name="air", family="gas",
        rho=1.2, cp=1000.0, k=0.026, eps=0.0,
        gimp=1.0e12,                       # TOUGH[AIR] = 1e9 kJ/m2: air never shatters
        mu_class="granular",               # air never touches; any class will do
        transmit=1.0, derate=_FLAT, span_ref=0,
        src="materials.py SOLID/EMISS/TOUGH rows for AIR; a gas has no strength",
        legacy={"voxel": {"span": (0, "AIR carries nothing")}}))
    add(MatRec(
        key="wood", id=WOOD, name="wood", family="wood",
        rho=600.0, cp=1700.0, k=0.15, eps=0.9,                           # S (materials.py SOLID)
        E=1.1e10, E_perp=5.5e8, nu=0.35,                                 # B, S2: across the grain 1/20 as stiff
        sy=None, st=6.0e7, st_perp=3.0e6,                                # B, [I] st: inside the sourced clear-wood tension 58-112 MPa (A-S); st_perp [I] 1.7-5.5
        sc=4.0e7, sc_perp=6.0e6,                                         # [I] sc: sourced 33-58 MPa; sc_perp (u)
        sf=7.0e7, tau=8.0e6,                                             # [I] sf: sourced MOR 34-105; tau [I]: sourced shear 4.7-13.8
        ductility=0.01,                                                  # (u) brittle in tension
        alpha=4.0e-6, alpha_perp=4.0e-5,                                 # (u)
        H=3.0e7,                                                         # (u) Janka pine 2.2 kN / 100 mm2 = 22 MPa, oak 54
        gimp=8000.0,                                                     # today's TOUGH 8 kJ/m2
        KIc=4.0e5, weibull_m=10.0,                                       # (u)
        mu_class="wood",
        creep_n=24.7, t_R1_s=300.0,                                      # [E] chosen to give 0.57 of short-test strength for ten years
        rot_ratio=5.0,                                                   # (u) Wilcox 1978
        derate=_WOOD_DERATE, ring=1.0e-4, char_zero_layer_m=0.007,       # EN 1995-1-2
        burn=BurnRec(300.0, 16000.0, 20.0, 1.33, 0.6, (2000.0, 25000.0, _PY)),
        span_ref=12, menu_name="wood",
        src="spec T9 v2 4.1; strengths B/S2-S4 plus (u)",
        legacy={"mpm": {"yield_strain": (None, _SOLVER), "damp": (1.00, _SOLVER),
                        "strength": (1.0e7, "continuum strength 10 MPa: the softening was tuned around it; the bond store has 60 MPa"),
                        "rigid": (1.00, _SOLVER)},
                "bonds": {"damp": (0.10, _SOLVER)}}))
    add(MatRec(
        key="stone", id=STONE, name="stone", family="mineral",
        rho=2700.0, cp=800.0, k=2.0, eps=0.9,
        E=5.0e10, nu=0.25, st=1.0e7, sc=1.5e8,                           # B; [I] sc: sourced granite 114-159 MPa; [I] st: sourced Brazilian 4.7-18.5 (granite, limestone)
        sf=1.5e7, tau=2.0e7,                                             # (u)
        ductility=0.001, alpha=8.0e-6, H=6.0e9, gimp=3000.0,             # (u)
        KIc=1.5e6, weibull_m=10.0,                                       # (u)
        mu_class="rock", creep_n=40.0, t_R1_s=1.0,                       # (u) Atkinson and Meredith 1987: 30-50
        derate=_STONE_DERATE, ring=3.0e-4, span_ref=20, menu_name="stone",
        src="spec T9 v2 4.1",
        legacy={"mpm": {"yield_strain": (None, _SOLVER), "damp": (0.50, _SOLVER),
                        "strength": (8.0e6, "continuum 8 MPa is BELOW wood's 10 MPa there: the stone/wood inversion, removed first (spec 5)"),
                        "rigid": (1.00, _SOLVER)},
                "bonds": {"rho": (2600.0, "bond store used 2600, the voxel table 2700"),
                          "damp": (0.05, _SOLVER)}}))
    add(MatRec(
        key="iron", id=IRON, name="iron", family="metal",
        rho=7800.0, cp=450.0, k=55.0, eps=0.8,
        E=2.0e11, nu=0.29, sy=2.5e8, st=3.5e8, sc=2.5e8,                 # sy B (the bond store calls it the strength); st (u)
        sf=None, tau=1.45e8,                                             # 0.58 sy
        ductility=0.20, dbtt_C=(-30.0, 20.0),                            # mild steel Y; step (u)
        alpha=11.8e-6, H=1.2e9, gimp=150000.0,                           # alpha Y; H (u) HV 120
        KIc=5.0e7, weibull_m=40.0,                                       # (u)
        mu_class="steel", derate=_IRON_DERATE, ring=1.0e-3,
        melt=MeltRec(1538.0, "miron", 247.0), span_ref=30, menu_name="iron",
        src="spec T9 v2 4.1",
        legacy={"mpm": {"yield_strain": (2.0e-3, _SOLVER), "damp": (0.40, _SOLVER),
                        "strength": (None, _SOLVER), "rigid": (0.90, _SOLVER)},
                "bonds": {"rho": (7870.0, "bond store used wrought iron 7870, the voxel table 7800"),
                          "st": (2.5e8, "the bond store breaks iron at its YIELD, 250 MPa; the record keeps the tensile strength"),
                          "damp": (0.05, _SOLVER)}}))
    add(MatRec(
        key="flesh", id=FLESH, name="flesh", family="tissue",
        rho=1000.0, cp=3500.0, k=0.37, eps=0.95,
        E=1.0e6, nu=0.45, st=1.0e6, sc=1.0e6, sf=1.0e6, tau=3.0e5,       # (u) soft solid
        ductility=0.5, alpha=1.0e-4, H=1.0e4, gimp=50000.0,              # (u)
        weibull_m=10.0, mu_class="skin", derate=_FLAT, ring=1.0e-7,
        burn=BurnRec(280.0, 8000.0, 10.0, 0.6, 0.8, (3000.0, 20000.0, _PY)),
        span_ref=6, src="spec T9 v2 4.1; a soft solid",
        legacy={"mpm": {"yield_strain": (5.0e-2, _SOLVER), "damp": (4.00, _SOLVER),
                        "strength": (None, _SOLVER), "rigid": (0.05, _SOLVER)},
                "bonds": {"rho": (1050.0, "bond store used living tissue 1050, the voxel table 1000"),
                          "damp": (0.30, _SOLVER)}}))
    add(MatRec(
        key="ash", id=ASH, name="ash", family="residue",
        rho=150.0, cp=800.0, k=0.1, eps=0.9,
        E=1.0e8, nu=0.3, st=0.0, sc=1.0e4, sf=0.0, tau=0.0,              # M for E; sc is a HEAP's bearing figure from its friction angle, not a strength
        ductility=0.0, alpha=0.0, H=1.0e5, gimp=30000.0,                 # alpha: not applicable to a heap
        mu_class="granular", derate=_FLAT, ring=1.0e-7,
        burn=BurnRec(None, 0.0, 0.0, 0.0, None), span_ref=1, menu_name="ash",
        src="spec T9 v2 4.1; a heap, fails the member test by its own sc",
        legacy={"mpm": {"yield_strain": (5.0e-2, _SOLVER), "damp": (3.00, _SOLVER),
                        "strength": (None, _SOLVER), "rigid": (0.0, _SOLVER)}}))
    add(MatRec(
        key="glass", id=GLASS, name="glass", family="mineral",
        rho=2500.0, cp=800.0, k=0.9, eps=0.92,
        E=7.0e10, nu=0.22, st=5.0e7, sc=1.0e9, sf=5.0e7, tau=3.0e7,      # B bending 5e7 (30-90); sc (u)
        ductility=0.0, alpha=9.0e-6, H=5.5e9, gimp=200.0,                # alpha Y soda-lime; gimp is a calibrated game number
        KIc=0.75e6, weibull_m=7.0,                                       # KIc Y; m (u)
        mu_class="glass", transmit=0.96, creep_n=17.0, t_R1_s=1.0,       # (u) Wiederhorn 1967: 16-20
        derate=_GLASS_DERATE, ring=1.0e-3, span_ref=8, menu_name="glass",
        src="spec T9 v2 4.1",
        legacy={"mpm": {"yield_strain": (None, _SOLVER), "damp": (0.30, _SOLVER),
                        "strength": (5.0e7, "continuum strength = the bending strength"),
                        "rigid": (1.00, _SOLVER)},
                "bonds": {"damp": (0.02, _SOLVER)},
                "voxel": {"tshock": (120.0, "TSHOCK stays the case table until the vessels scene proves tshock_face() = 123.8 K (W7b)")}}))
    add(MatRec(
        key="leaf", id=LEAF, name="leaf", family="fibre",
        rho=200.0, cp=2400.0, k=0.12, eps=0.9,
        E=2.0e9, nu=0.3, st=2.0e7, sc=0.0, sf=0.0, tau=0.0,              # E: M (springy canopy; the bond petiole is 3e8); st B green petiole
        ductility=0.10, alpha=0.0, gimp=30000.0,
        mu_class="fibre", derate=_FLAT, ring=1.0e-7,
        burn=BurnRec(240.0, 15000.0, 20.0, 1.25, 0.8, (2000.0, 25000.0, _PY), None, 6.0),
        span_ref=4, menu_name="leaves",
        src="spec T9 v2 4.1; a leaf holds no compression",
        legacy={"mpm": {"yield_strain": (None, _SOLVER), "damp": (1.00, _SOLVER),
                        "strength": (None, _SOLVER), "rigid": (0.15, _SOLVER)},
                "bonds": {"row": ("petiole", "a leaf cell hangs on a stalk: the bond store reads the PETIOLE row, not a leaf row")}}))
    add(MatRec(
        key="char", id=CHAR, name="char", family="residue",
        rho=300.0, cp=1000.0, k=0.084, eps=0.9,
        E=1.0e9, nu=0.3, st=1.0e6, sc=2.0e6, sf=1.0e6, tau=1.0e6,        # (u) weak brittle wood
        ductility=0.005, alpha=5.0e-6, H=1.0e8, gimp=800.0,              # (u)
        KIc=1.0e5, weibull_m=5.0,                                        # (u)
        mu_class="granular", derate=_FLAT, ring=3.0e-5,
        burn=BurnRec(330.0, 30000.0, 3.0, 2.6, 0.05, None, (7.0 / 21.0, 0.75, 300.0)),
        span_ref=6, menu_name="char", src="spec T9 v2 4.1",
        legacy={"mpm": {"yield_strain": (2.0e-2, _SOLVER), "damp": (2.00, _SOLVER),
                        "strength": (None, _SOLVER), "rigid": (1.0, _SOLVER)},
                "bonds": {"damp": (0.10, _SOLVER)}}))
    add(MatRec(
        key="tin", id=TIN, name="tin", family="metal",
        rho=7300.0, cp=230.0, k=66.8, eps=0.3,
        E=5.0e10, nu=0.33, sy=1.1e7, st=2.0e7, sc=1.1e7, sf=None, tau=1.2e7,   # st B; the rest (u)
        ductility=0.40, alpha=22.0e-6, H=5.1e7, gimp=60000.0,                  # alpha, H Y (Brinell 5 kgf/mm2)
        weibull_m=30.0, mu_class="softmetal",                                  # (u)
        derate=((-20.0, 1.0, 1.0), (20.0, 1.0, 1.0), (232.0, 0.0, 0.0)),       # (u) homologous-temperature stand-in
        ring=1.0e-3, melt=MeltRec(232.0, "mtin", 59.0), span_ref=16, menu_name="tin",
        src="spec T9 v2 4.1",
        legacy={"mpm": {"yield_strain": (2.0e-3, _SOLVER), "damp": (0.40, _SOLVER),
                        "strength": (None, _SOLVER), "rigid": (0.90, _SOLVER)},
                "bonds": {"nu": (0.36, "bond store used 0.36, the particle table 0.33"),
                          "damp": (0.05, _SOLVER)}}))
    add(MatRec(
        key="lead", id=LEAD, name="lead", family="metal",
        rho=11340.0, cp=130.0, k=35.3, eps=0.3,
        E=1.6e10, nu=0.44, sy=5.5e6, st=1.2e7, sc=5.5e6, sf=None, tau=6.0e6,   # st B; the rest (u)
        ductility=0.50, alpha=29.0e-6, H=3.8e7, gimp=40000.0,                  # alpha, H Y
        weibull_m=30.0, mu_class="softmetal",
        derate=((-20.0, 1.0, 1.0), (20.0, 1.0, 1.0), (327.0, 0.0, 0.0)),       # (u)
        ring=1.0e-5, melt=MeltRec(327.0, "mlead", 23.0), span_ref=10, menu_name="lead",
        src="spec T9 v2 4.1",
        legacy={"mpm": {"yield_strain": (4.0e-4, _SOLVER), "damp": (3.00, _SOLVER),
                        "strength": (None, _SOLVER), "rigid": (0.30, _SOLVER)},
                "bonds": {"damp": (0.05, _SOLVER)}}))
    add(MatRec(
        key="rope", id=ROPE, name="rope", family="fibre",
        rho=885.0, cp=1400.0, k=0.10, eps=0.9,
        E=9.4e8, nu=0.4, sy=None, st=9.4e7, sc=0.0, sf=0.0, tau=0.0,     # B: 12 mm manila, 10.6 kN; a rope carries no compression
        ductility=0.15, alpha=0.0, gimp=30000.0,                         # manila 10-15 %
        weibull_m=20.0, mu_class="fibre", derate=_FLAT, ring=1.0e-7,     # (u)
        burn=BurnRec(260.0, 16500.0, 20.0, 1.26, 0.6, (2000.0, 25000.0, _PY)),
        span_ref=1, menu_name="rope", src="spec T9 v2 4.1",
        legacy={"mpm": {"yield_strain": (None, _SOLVER), "damp": (1.00, _SOLVER),
                        "strength": (None, _SOLVER), "rigid": (0.05, _SOLVER)},
                "bonds": {"damp": (0.10, _SOLVER)}}))
    # bond-only members: no lattice id; they exist only inside bonded things
    add(MatRec(key="petiole", id=None, name="petiole", family="bond",
               rho=900.0, E=3.0e8, nu=0.4, st=2.0e7, ductility=0.10,
               legacy={"bonds": {"damp": (0.10, _SOLVER)}},
               src="B: a green leaf stalk (u)"))
    add(MatRec(key="petiole_dry", id=None, name="petiole_dry", family="bond",
               rho=900.0, E=3.0e8, nu=0.4, st=2.0e6, ductility=0.05,
               legacy={"bonds": {"damp": (0.10, _SOLVER)}},
               src="B: the stalk of a dry leaf holds a tenth as much (assumption)"))
    add(MatRec(key="twig", id=None, name="twig", family="bond",
               rho=900.0, E=8.0e9, nu=0.35, st=4.0e7, ductility=0.02,
               legacy={"bonds": {"damp": (0.10, _SOLVER)}},
               src="B: a green twig, 0.7 as stiff as dry wood, 40 MPa in bending (u)"))
    return tuple(R)


# ── the table, as an immutable object ───────────────────────────────────────
class MatTable:
    """An ordered, immutable set of rows. `ingest` and `retire` return a NEW
    table; the ids already given never change (never renumbered)."""

    def __init__(self, rows):
        self.rows = tuple(rows)
        self._by_key = {}
        self._by_id = {}
        for r in self.rows:
            if r.key in self._by_key:
                raise ValueError("duplicate material key %r" % r.key)
            self._by_key[r.key] = r
            if r.id is not None:
                if r.id in self._by_id:
                    raise ValueError("duplicate material id %r" % r.id)
                self._by_id[r.id] = r
        ids = sorted(self._by_id)
        if ids != list(range(len(ids))):
            raise ValueError("lattice ids must be 0..n-1 with no gap, got %r" % ids)
        self.nmat = len(ids)
        self.by_id_list = tuple(self._by_id[i] for i in ids)

    # ---- lookup ----
    def row(self, key_or_id):
        """The row for a key or an id, as stored (a retired row is still here)."""
        if isinstance(key_or_id, str):
            return self._by_key[key_or_id]
        return self._by_id[int(key_or_id)]

    def resolve(self, key_or_id):
        """The LIVE row: follows `successor` links from a retired one."""
        r = self.row(key_or_id)
        seen = set()
        while r.successor is not None:
            if r.key in seen:
                raise ValueError("successor loop at %r" % r.key)
            seen.add(r.key)
            r = self._by_key[r.successor]
        return r

    def live_ids(self):
        return [r.id for r in self.by_id_list if r.successor is None]

    # ---- change (returns a new table) ----
    def ingest(self, row, id=None):
        """Add a row given as a plain dict (see `rec_from_dict`). The new row's
        id is the next free lattice id (12 for ICE) unless `id` says so."""
        if isinstance(row, MatRec):
            rec = row
        else:
            rec = rec_from_dict(row, self.nmat if id is None else id)
        check_handover(rec)
        return MatTable(self.rows + (rec,))

    def retire(self, key, successor):
        """Mark a row retired: it keeps its id and names `successor`."""
        if successor not in self._by_key:
            raise KeyError(successor)
        if key == successor:
            raise ValueError("a row cannot succeed itself")
        old = self._by_key[key]
        new = dataclasses.replace(old, successor=successor)
        return MatTable(tuple(new if r.key == key else r for r in self.rows))

    # ---- arrays ----
    def arrays(self):
        """Float32 arrays of length nmat for gathers by material id. A missing
        quantity reads 0.0 (strengths, hardness), inf for a yield that never
        comes, and the tension value where st_perp / sf are not given."""
        rs = self.by_id_list

        def col(f):
            return np.array([f(r) for r in rs], np.float32)
        return {
            "rho": col(lambda r: r.rho), "cp": col(lambda r: r.cp), "k": col(lambda r: r.k),
            "eps": col(lambda r: r.eps), "E": col(lambda r: r.E),
            "E_perp": col(lambda r: r.E if r.E_perp is None else r.E_perp),
            "nu": col(lambda r: r.nu),
            "sy": col(lambda r: math.inf if r.sy is None else r.sy),
            "st": col(lambda r: r.st),
            "st_perp": col(lambda r: r.st if r.st_perp is None else r.st_perp),
            "sc": col(lambda r: r.sc), "sf": col(lambda r: r.sf_eff()), "tau": col(lambda r: r.tau),
            "ductility": col(lambda r: r.ductility),
            "alpha": col(lambda r: r.alpha),
            "H": col(lambda r: 0.0 if r.H is None else r.H),
            "gimp": col(lambda r: r.gimp),
            "weibull_m": col(lambda r: 0.0 if r.weibull_m is None else r.weibull_m),
            "mu_class": np.array([MU_CLASSES.index(r.mu_class) for r in rs], np.int8),
            "structural": np.array([r.structural() for r in rs], bool),
        }

    def derate_grid(self, t0=-50.0, step=25.0, nt=61):
        """(KS, KE, T): float32 [nmat, nt] on a regular grid (-50..1450 C by 25),
        for gathers on the hot set; `grid_lookup` reads them."""
        T = t0 + step * np.arange(nt, dtype=np.float64)
        ks = np.zeros((self.nmat, nt), np.float32)
        ke = np.zeros((self.nmat, nt), np.float32)
        for r in self.by_id_list:
            a, b = r.derate_at(T)
            ks[r.id] = a
            ke[r.id] = b
        return ks, ke, T.astype(np.float32)

    def structural_keys(self):
        return [r.key for r in self.by_id_list if r.structural()]

    # ---- the plain-dict adapter ----
    def props(self, key, grain="along"):
        """The mechanical numbers of a material as a plain dict:
        E, E_perp, nu, sy, st, st_perp, sc, tau, ductility, KIc, gimp, weibull_m.
        `grain` is "along" (default) or "across": for "across" E, st and sc are
        the cross-grain values (wood: 0.55 GPa, 3 MPa), the same numbers the
        bond store holds. A retired name answers with its successor's numbers."""
        r = self.resolve(key)
        across = grain == "across"
        if grain not in ("along", "across", None):
            raise ValueError("grain must be 'along' or 'across'")
        return {
            "E": (r.E_perp if across and r.E_perp is not None else r.E),
            "E_perp": r.E_perp,
            "nu": r.nu, "sy": r.sy,
            "st": (r.st_perp if across and r.st_perp is not None else r.st),
            "st_perp": r.st_perp,
            "sc": (r.sc_perp if across and r.sc_perp is not None else r.sc),
            "tau": r.tau, "ductility": r.ductility, "KIc": r.KIc,
            "gimp": r.gimp, "weibull_m": r.weibull_m,
        }


def check_handover(rec):
    """Refuse a row that no law can divide by, at the door. Every later view
    divides by rho (and, for a lattice row, by rho cp: the thermal diffusivity),
    so a row with rho <= 0, a missing or non-finite number, or a lattice cp <= 0
    is a hand-over error, said clearly here and not as a ZeroDivisionError three
    calls later. Raises ValueError naming the row and the field."""
    def bad(field, why):
        raise ValueError("hand-over row %r: %s %s" % (rec.key, field, why))
    for f in ("rho", "cp", "k", "E", "eps", "nu"):
        v = getattr(rec, f)
        # any real scalar: int, float, fractions, and numpy's float32 / int64 ...
        if (not isinstance(v, (numbers.Real, np.number)) or isinstance(v, (bool, np.bool_))
                or not math.isfinite(v)):
            bad(f, "must be a finite number, got %r" % (v,))
    if rec.rho <= 0.0:
        bad("rho", "must be above zero (got %r; was it left out?)" % rec.rho)
    if rec.id is not None and rec.cp <= 0.0:
        bad("cp", "must be above zero for a lattice row (got %r)" % rec.cp)
    for f in ("cp", "k", "E"):
        if getattr(rec, f) < 0.0:
            bad(f, "must not be negative (got %r)" % getattr(rec, f))


# the aliases of a plain row dict (T6's ICE_ROW and the like): name -> (field, scale)
_ALIAS = {
    "rho": ("rho", 1.0), "density": ("rho", 1.0), "dens": ("rho", 1.0),
    "cp": ("cp", 1.0), "cp_Jg": ("cp", 1000.0), "c_Jg": ("cp", 1000.0), "c": ("cp", 1000.0),
    "k": ("k", 1.0), "conductivity": ("k", 1.0),
    "eps": ("eps", 1.0), "emissivity": ("eps", 1.0),
    "tough_kJm2": ("gimp", 1000.0), "TOUGH": ("gimp", 1000.0), "tough": ("gimp", 1000.0),
    "E_young": ("E", 1.0), "young": ("E", 1.0), "poisson": ("nu", 1.0),
    "span": ("span_ref", 1.0),
    "tensile": ("st", 1.0), "sigma_t": ("st", 1.0),
    "friction_class": ("mu_class", 1.0),
}
_FIELDS = {f.name for f in dataclasses.fields(MatRec)}


def rec_from_dict(d, mat_id):
    """A MatRec from a plain dict. Keys are MatRec field names, or one of a few
    aliases (density, cp_Jg in J/(g K), emissivity, tough_kJm2 ...). Keys that
    mean nothing to the record (isotherms, latent heats of another law) ride in
    `extra`, so their owner's numbers are held, not lost. Needs `key`."""
    d = dict(d)
    key = d.pop("key", None) or d.get("name")
    if not key:
        raise ValueError("a row needs a key")
    name = d.pop("name", key)
    bond_only = bool(d.pop("bond_only", False))
    given_id = d.pop("id", None)
    kw = {}
    extra = dict(d.pop("extra", {}) or {})
    legacy = dict(d.pop("legacy", {}) or {})
    for nm, v in d.items():
        if nm in _ALIAS:
            f, sc = _ALIAS[nm]
            kw[f] = v if (sc == 1.0 or not isinstance(v, (int, float))) else v * sc
        elif nm in _FIELDS:
            kw[nm] = v
        else:
            extra[nm] = v
    if bond_only:
        mat_id = None
    elif given_id is not None:
        mat_id = int(given_id)
    kw.setdefault("mu_class", "ice" if key == "ice" else "granular")
    if isinstance(kw.get("burn"), dict):
        kw["burn"] = BurnRec(**kw["burn"])
    if isinstance(kw.get("melt"), dict):
        kw["melt"] = MeltRec(**kw["melt"])
    if "derate" in kw:
        kw["derate"] = tuple(tuple(float(x) for x in p) for p in kw["derate"])
    kw.setdefault("derate", _FLAT)
    family = kw.pop("family", "ingested")
    return MatRec(key=key, id=mat_id, name=name, family=family,
                  legacy=legacy, extra=extra, **kw)


BASE = MatTable(_rows())
REC = {r.key: r for r in BASE.rows}                     # key -> row (all rows, bond-only too)
BY_ID = list(BASE.by_id_list)                           # id -> row (lattice rows only)
assert BASE.nmat == NMAT


def ingest_row(row, table=None, id=None):
    """A new table with the plain dict `row` added (default: BASE plus it).
    The hook T6 uses to hand over icephase.ICE_ROW."""
    return (BASE if table is None else table).ingest(row, id)


# ── the planned skeleton rows and the ground (wave 14a refinement, lane F) ───
# THE IDS. tests/data/mat_ids.json pins mat ids 13 to 26 to the species registry's planned
# materials (species_seed.MAT_SLOTS: copper, quartz rock, SAND at 15, calcite, clay ... humus
# at 26), and a lattice id table must run 0..n-1 with no gap. So the ground cannot take 13
# onward without renaming a pinned id. It is done in the pinned order instead: the planned rows
# 13 to 26 are built here (their thermal numbers are the registry's own, copied exactly; their
# mechanical numbers are the `MAT_LIKE` stand-in's, as the registry says until a row of their
# own is sourced), and the ground materials the world lacked, SOIL and GRASS, are appended at
# 27 and 28 (appending is what the pin allows). The planned sand (15) gets its own bearing data.
#
# THE GROUND. Until now nothing in the world was soft enough to give under a falling body: ash
# bears 10 kPa (a heap, cannot hold a person) and every other row bears over 1 MPa, so
# impact.ground_of never found a yielding surface. `sc` is the BEARING stress a landing is held
# to; R2's range for firm ground is 1e5 to 5e5 Pa. Allowable bearing pressures are 70 to 150 kPa
# for clay and sand in the presumptive tables of building codes (IBC 2018 table 1806.2: clay
# 1500 psf = 72 kPa, sand 2000 psf = 96 kPa; BS 8004 the same order) and an ultimate value is
# that times a safety factor of about 3: sand 2.9e5, soil (clay-loam) 2.2e5, turf over soil 1.5e5 (topsoil
# is the softest layer, u). Density: moist unit weight of clay soil 16 to 19 kN/m3, dry sand
# 14 to 17 (Das, Principles of Geotechnical Engineering); turf over topsoil 1300 (u). Thermal
# (soil, grass): moist clay-loam k 1.0 W/mK and cp 1.2 J/gK at 10 to 20 percent water (Abu-Hamdeh
# 2003, Soil Sci. Soc. Am. J. 67, 1213, u); turf 0.7 W/mK, 1.4 J/gK (u). Sand's thermal numbers
# are the registry's (dry quartz sand, 1590 g/L, 0.31 W/mK). Emissivity: soil 0.92, grass 0.97
# (Oke, Boundary Layer Climates, u). E: stiff clay 20 to 50 MPa, sand 10 to 80 MPa (Das, u).
# Friction: the loose-grain class ("granular", matfric: 0.50 static, u); a foot on soil or sand
# is 0.5 to 0.8, so the class is the low side. None of them burns or melts.
PLANNED_ROWS = (        # id, key, rho g/L, cp J/(g K), k W/(m K), emissivity, the row whose mechanics stand in
    (13, "copper", 8960.0, 0.385, 401.0, 0.3, "iron"),
    (14, "quartz_rock", 2650.0, 0.74, 7.7, 0.9, "stone"),
    (15, "sand", 1590.0, 0.74, 0.3122793330414563, 0.9, "ash"),
    (16, "calcite", 2710.0, 0.82, 3.6, 0.9, "stone"),
    (17, "clay", 2600.0, 0.95, 1.0, 0.9, "stone"),
    (18, "graphite", 2260.0, 0.709, 120.0, 0.8, "stone"),
    (19, "magnetite", 5170.0, 0.6, 5.1, 0.9, "iron"),
    (20, "iron_oxide", 5240.0, 0.65, 3.0, 0.9, "stone"),
    (21, "salt", 2165.0, 0.864, 6.5, 0.9, "stone"),
    (22, "tallow", 900.0, 2.0, 0.2, 0.9, "flesh"),
    (23, "sugar", 1587.0, 1.24, 0.5, 0.9, "stone"),
    (24, "quicklime", 3340.0, 0.75, 7.0, 0.9, "stone"),
    (25, "slaked_lime", 449.99999999999994, 1.18, 0.050339712828688946, 0.9, "ash"),
    (26, "humus", 700.0, 1.5, 0.2, 0.9, "ash"),
)
_GROUND_MECH = {        # own mechanics of the ground rows (see the comment above)
    "sand": dict(E=2.0e7, nu=0.30, sc=2.9e5, tau=0.0, H=1.0e5),
    "soil": dict(E=3.0e7, nu=0.35, sc=2.2e5, tau=3.0e4, H=1.0e5),
    "grass": dict(E=1.0e7, nu=0.40, sc=1.5e5, tau=1.0e4, H=5.0e4),
}
GROUND_ROWS = (        # appended after the pinned ids: id, key, rho, cp, k, eps
    (27, "soil", 1600.0, 1.2, 1.0, 0.92),
    (28, "grass", 1300.0, 1.4, 0.7, 0.97),
)


def _planned_rec(base, i, key, rho, cp, k, eps, like):
    r = base.row(like)
    mech = _GROUND_MECH.get(key)
    rec = dataclasses.replace(
        r, key=key, id=i, name=key, family="ground" if key in _GROUND_MECH else "planned",
        rho=rho, cp=cp * 1000.0, k=k, eps=eps, burn=None, melt=None, menu_name=key if mech else None,
        alpha=0.0,                      # no expansion figure yet: R_shock is infinite, so a stand-in row never thermal-shocks or spalls (a hot salt crust is not stone)
        legacy={}, extra={}, successor=None,
        src="wave 14a refinement F: thermal numbers from the species registry (planned row %d), "
            "mechanics of the %s row as a stand-in" % (i, like) if i < 27 else
            "wave 14a refinement F: appended ground row; see the comment above PLANNED_ROWS")
    if mech:
        d = dict(mech)
        rec = dataclasses.replace(rec, st=0.0, sf=0.0, ductility=0.0, alpha=0.0, gimp=30000.0,
                                  mu_class="granular", derate=_FLAT, ring=1.0e-7, span_ref=1, **d)
    return rec


def ingest_ground(table):
    """`table` (the old twelve and ICE) with the planned rows 13 to 26 and the ground rows 27, 28
    added, in id order, never renumbered."""
    base = BASE
    for i, key, rho, cp, k, eps, like in PLANNED_ROWS + tuple(
            (i, key, rho, cp, k, eps, "ash") for i, key, rho, cp, k, eps in GROUND_ROWS):
        rec = _planned_rec(base, i, key, rho, cp, k, eps, like)
        check_handover(rec)
        table = MatTable(table.rows + (rec,))
    return table


def ingest_ice(table=None):
    """Ingest T6's ICE row if icephase can be imported: its ICE_ROW dict, or
    its ice_row() function. Returns the new table, or None when icephase is
    not there yet (the caller keeps the old table)."""
    ip = _sibling("icephase")
    if ip is None:
        return None
    row = getattr(ip, "ICE_ROW", None)
    if row is None and callable(getattr(ip, "ice_row", None)):
        row = ip.ice_row()
    if row is None:
        return None
    return ingest_row(row, table)


def arrays():
    return BASE.arrays()


def derate_grid():
    return BASE.derate_grid()


def grid_lookup(grid, mat, T, t0=-50.0, step=25.0):
    """Gather from a derate grid at temperature T (C) by linear interpolation:
    grid[mat, i] with i from T. `mat` and `T` broadcast."""
    x = (np.asarray(T, float) - t0) / step
    x = np.clip(x, 0.0, grid.shape[1] - 1.0)
    i = np.minimum(np.floor(x).astype(np.int64), grid.shape[1] - 2)
    f = x - i
    m = np.asarray(mat)
    return grid[m, i] * (1.0 - f) + grid[m, i + 1] * f


def structural(key_or_id):
    return BASE.resolve(key_or_id).structural()


def props(key, grain="along"):
    """See MatTable.props."""
    return BASE.props(key, grain)


# ── blending (separate-phase composites only) ───────────────────────────────
# mixrules' own form numbers are indices into species.FORM_NAMES; a blend part's
# form word maps onto them. "matrix" and "bound" are both the BOUND form (they are
# the continuous phases); a word that is none of these is refused, never guessed.
_FORM_WORD = {"crystal": "crystal", "grain": "grain", "matrix": "bound", "bound": "bound"}


def _rule(rules, kind):
    """The mixrules function for `kind`, or None. `rules` is mixrules itself (or
    anything with the same two names): `RULES`, the dict kind -> fn(ent, slot) ->
    array[n], and `make_entries(n, cell, key, frac)`. The kinds are mixrules' own:
    "rho" (g/L), "c" (J/g/K), "k" (W/m/K), "mu" (mPa s)."""
    reg = getattr(rules, "RULES", None)
    if isinstance(reg, dict):
        return reg.get(kind)
    return None


def _species_entries(rules, recs, mass, forms):
    """One mixrules row for the parts, or None if some part cannot be one.

    A part can be one when its key names a species, its form word is one mixrules
    knows, and the record's rho, cp and k are the species' own (a porous or
    re-tempered record is not that species, so the species rules would answer for
    the wrong stuff). When any part cannot, the caller falls back to the stand-in."""
    species = _sibling("species")
    if species is None or getattr(rules, "make_entries", None) is None:
        return None
    keys = []
    for r, f in zip(recs, forms):
        sid = species.BY_NAME.get(r.key)
        if sid is None or f not in _FORM_WORD:
            return None
        sp = species.SPECIES[sid]
        same = (math.isclose(sp.get("rho_s", "engine", math.nan) or math.nan, r.rho, rel_tol=1e-9)
                and math.isclose((sp.get("c_s", "engine", math.nan) or math.nan) * 1000.0, r.cp, rel_tol=1e-9)
                and math.isclose(sp.get("k_s", "engine", math.nan) or math.nan, r.k, rel_tol=1e-9))
        if not same:
            return None
        keys.append(species.pack_key(sid, species.FORM_NAMES.index(_FORM_WORD[f])))
    order = np.argsort(np.asarray(keys, np.int64), kind="stable")
    return rules.make_entries(1, np.zeros(len(keys), np.int64),
                              np.asarray(keys, np.uint32)[order], np.asarray(mass, float)[order])


_CONTINUOUS = ("matrix", "bound")


def blend(parts, rules=None):
    """The mechanical numbers of a SEPARATE-PHASE composite. `parts` is a list
    of (rec, mass_fraction, form). Dissolved species are NOT blended here:
    alloys (carbon in iron, tin in copper) and eutectics (tin-lead at 183 C)
    are T4 mixture rows, and this raises for form "dissolved".

    Computed here, and nowhere else: E (Hill average of the Voigt and Reuss
    bounds), the strengths, hardness and toughness (the Voigt sum, capped by
    the weakest CONTINUOUS phase, (u)), ductility (the weakest continuous
    phase), Poisson ratio and thermal strain (volume-weighted). A phase is
    continuous when its form is "matrix" or "bound"; with none marked, the
    phase with the most volume is.

    Density, specific heat, conductivity and viscosity come from mixrules (T1)
    when every part is a species of its own table (its key names a species and
    its rho, cp and k are that species') and its form is crystal, grain, matrix
    or bound. This function holds no conductivity law. If mixrules is missing,
    or a part is not such a species (a porous record, a bond-only member),
    density and specific heat are the plain mass and energy sums, and k is the
    dominant phase's with extra["k_source"] saying so; a cross test fails if conductivity code is ever added here."""
    parts = list(parts)
    if not parts:
        raise ValueError("blend needs at least one part")
    for rec, w, form in parts:
        if str(form).lower() == "dissolved":
            raise ValueError("dissolved species are alloy/mixture rows (T4), not a blend")
    mass = np.array([w for _r, w, _f in parts], float)
    if not np.isfinite(mass).all() or (mass < 0).any() or mass.sum() <= 0:
        raise ValueError("mass fractions must be finite, non-negative and not all zero")
    mass = mass / mass.sum()
    # A part with no mass is not in the blend. Dropping it makes "a blend of one
    # part" mean one part of matter, and keeps the identity law below exact.
    keep = [i for i in range(len(parts)) if mass[i] > 0]
    parts = [parts[i] for i in keep]
    mass = mass[keep]
    mass = mass / mass.sum()
    vol = np.array([m / r.rho for (r, _w, _f), m in zip(parts, mass)])
    phi = vol / vol.sum()
    recs = [p[0] for p in parts]
    forms = [str(p[2]).lower() for p in parts]
    cont = [i for i, f in enumerate(forms) if f in _CONTINUOUS and phi[i] > 0]
    if not cont:
        cont = [int(np.argmax(phi))]
    dom = recs[int(np.argmax(phi))]

    def voigt(vals):
        v = [x for x in vals]
        if all(x is None for x in v):
            return None
        return float(sum(phi[i] * (0.0 if x is None else x) for i, x in enumerate(v)))

    def capped(vals):
        s = voigt(vals)
        if s is None:
            return None
        weakest = [vals[i] for i in cont if vals[i] is not None]
        return min(s, min(weakest)) if weakest else s

    Ev = float(sum(phi[i] * recs[i].E for i in range(len(recs))))
    if all(r.E > 0 for r in recs):
        Er = float(1.0 / sum(phi[i] / recs[i].E for i in range(len(recs))))
    else:
        Er = 0.0
    E = 0.5 * (Ev + Er)
    if len(recs) == 1:
        E = recs[0].E                               # exact, not 0.5 (E + E)

    rules = _sibling("mixrules") if rules is None else rules
    ent = _species_entries(rules, recs, mass, forms)
    extra = {}
    rho = float(1.0 / sum(m / r.rho for r, m in zip(recs, mass)))      # plain volume sum
    cp = float(sum(m * r.cp for r, m in zip(recs, mass)))              # plain energy sum
    k = dom.k
    if ent is None:
        extra["k_source"] = "standin: dominant phase (mixrules cannot take these parts)"
    else:
        rho = float(rules.RULES["rho"](ent, "engine")[0])
        cp = float(rules.RULES["c"](ent, "engine")[0]) * 1000.0         # J/g/K -> J/kg/K
        k = float(rules.RULES["k"](ent, "engine")[0])
        mu = float(rules.RULES["mu"](ent, "engine")[0]) if "mu" in rules.RULES else math.nan
        if math.isfinite(mu):
            extra["viscosity"] = mu
    if len(recs) == 1:
        # IDENTITY LAW: a blend of one part is that part. mixrules' electronic rule
        # would give a metal in matrix or bound form its Wiedemann-Franz k (1.34x
        # iron's record, 1.06x copper's), which is a different number for the SAME
        # stuff; the record's own rho, cp and k are the answer.
        rho, cp, k = dom.rho, dom.cp, dom.k              # dom IS the one part
        extra.pop("k_source", None)

    ductility = min(recs[i].ductility for i in cont)
    out = dataclasses.replace(
        dom, key="blend", id=None, name="blend", family="blend", successor=None,
        rho=rho, cp=cp, k=k, E=E,
        nu=voigt([r.nu for r in recs]), alpha=voigt([r.alpha for r in recs]),
        sy=capped([r.sy for r in recs]), st=capped([r.st for r in recs]),
        sc=capped([r.sc for r in recs]), tau=capped([r.tau for r in recs]),
        sf=capped([r.sf_eff() for r in recs]), H=capped([r.H for r in recs]),
        gimp=capped([r.gimp for r in recs]), ductility=ductility,
        weibull_m=min((recs[i].weibull_m for i in cont if recs[i].weibull_m is not None),
                      default=None),
        E_perp=None, st_perp=None, sc_perp=None, alpha_perp=None,
        burn=None, melt=None, legacy={}, extra=extra,
        src="blend of %s" % ", ".join(r.key for r in recs))
    return out


def porous(rec, rel_density):
    """A porous solid of relative density rho/rho_s (bread, pumice, wood char,
    a foam): Gibson and Ashby 1997 (recalled, (u)): E = E_s r^2, crushing
    strength 0.2 sf_s r^1.5 (brittle foam), flexural strength 0.65 sf_s r^1.5.
    Returns a MatRec-shaped row. Tension is taken as the flexural figure."""
    r = float(rel_density)
    if not 0.0 < r <= 1.0:
        raise ValueError("relative density must be in (0, 1]")
    sfs = rec.sf_eff()
    sf = 0.65 * sfs * r ** 1.5
    return dataclasses.replace(
        rec, key=rec.key + "_porous", id=None, name=rec.name + " (porous)",
        family="porous", successor=None, rho=rec.rho * r, E=rec.E * r * r,
        E_perp=None, st=sf, st_perp=None, sf=sf, sc=0.2 * sfs * r ** 1.5,
        sc_perp=None, tau=rec.tau * r ** 1.5, sy=None, ductility=min(rec.ductility, 0.01),
        burn=None, melt=None, legacy={}, extra={"relative_density": r},
        src="Gibson and Ashby 1997 porous scaling of " + rec.key)


# ── the old tables, rebuilt as views ────────────────────────────────────────
def _norm_fluid_ids(fluid_ids):
    """(mtin, mlead, miron) fluid ids, from a mapping (keys mtin/MTIN/tin ...)
    or a sequence in that order."""
    if isinstance(fluid_ids, dict):
        got = {}
        for k, v in fluid_ids.items():
            n = str(k).lower()
            n = n if n.startswith("m") else "m" + n
            got[n] = int(v)
        return {k: got[k] for k in MOLTEN_KEYS}
    seq = list(fluid_ids)
    if len(seq) != 3:
        raise ValueError("fluid_ids: the three molten ids (MTIN, MLEAD, MIRON)")
    return dict(zip(MOLTEN_KEYS, (int(x) for x in seq)))


def legacy_views(fluid_ids, table=None, upto=LEGACY_NMAT):
    """Every old table, rebuilt from the rows (F1: equal to the base18 oracle).

    `fluid_ids`: the molten fluid ids (MTIN, MLEAD, MIRON) - materials.py owns
    them and a test asserts they are 5, 6, 7. Only lattice ids below `upto`
    are rebuilt (the old tables know twelve).

    Returns a dict:
      SOLID {id: (rho g/L, cp J/(g K), k, ign C or None, heat J/g, rate g/(m2 s))}
      EMISS, SPAN, TOUGH (kJ/m2), TSHOCK, MELT {mat: (T, fluid id, latent)},
      FREEZE {fluid id: (T, mat, latent)}, PYRO, O2_RATIO, SOOT, GLOW,
      BURN_AREA, TRANSMIT, MATNAME, SOUND_RING, SKIN_T
      the float32 lookup arrays _DENS_ARR _CSOLID_ARR _KSOLID_ARR _SPAN_ARR
      _TOUGH_ARR _TSHOCK_ARR _EMISS_ARR _IGN_ARR _SKIN_DEPTH_ARR _SKIN_PATH_ARR
      _TRANSMIT_ARR and the float64 _ALPHA
      MPM {name: (rho, E, nu, yield strain, damp, strength, rigid)} and
      mpm_row(name)
      BOND_ROW {mat id: (E, nu, strength, damp, rho)}, NO_BOND, BOND_CONSTS
      (the named rows of src/bonds/materials.py) and bond_row(ma, mb, cos2)."""
    tb = BASE if table is None else table
    fid = _norm_fluid_ids(fluid_ids)
    rows = [r for r in tb.by_id_list if r.id < upto]
    V = {}
    solid, emiss, span, tough, tshock = {}, {}, {}, {}, {}
    melt, pyro, o2, soot, glow, barea, transmit, matname, ring = {}, {}, {}, {}, {}, {}, {}, {}, {}
    for r in rows:
        b = r.burn
        solid[r.id] = (r.rho, r.cp / 1000.0, r.k,
                       None if b is None else b.ign_C,
                       0.0 if b is None else b.heat_Jg,
                       0.0 if b is None else b.rate_gm2s)
        emiss[r.id] = r.eps
        span[r.id] = r.span_ref
        tough[r.id] = r.gimp / 1000.0
        t = _ov(r, "voxel", "tshock", None)
        if t is not None:
            tshock[r.id] = t
        if r.melt is not None:
            melt[r.id] = (r.melt.T_C, fid[r.melt.fluid_key], r.melt.latent_Jg)
        if b is not None:
            if b.pyro is not None:
                pyro[r.id] = tuple(b.pyro)
            if b.o2_ratio is not None:
                o2[r.id] = b.o2_ratio
            if b.soot is not None:
                soot[r.id] = b.soot
            if b.glow is not None:
                glow[r.id] = tuple(b.glow)
            if b.burn_area is not None:
                barea[r.id] = b.burn_area
        if r.id != AIR and r.transmit != 0.0:
            transmit[r.id] = r.transmit
        if r.menu_name is not None:
            matname[r.id] = r.menu_name
        if r.ring is not None:
            ring[r.id] = r.ring
    V.update(SOLID=solid, EMISS=emiss, SPAN=span, TOUGH=tough, TSHOCK=tshock,
             MELT=melt, FREEZE={f: (mt, m, lat) for m, (mt, f, lat) in melt.items()},
             PYRO=pyro, O2_RATIO=o2, SOOT=soot, GLOW=glow, BURN_AREA=barea,
             TRANSMIT=transmit, MATNAME=matname, SOUND_RING=ring, SKIN_T=SKIN_T)
    n = len(rows)
    f32 = np.float32
    V["_TOUGH_ARR"] = np.array([tough[m] for m in range(n)], f32)
    V["_TSHOCK_ARR"] = np.array([tshock.get(m, np.inf) for m in range(n)], f32)
    V["_DENS_ARR"] = np.array([solid[m][0] for m in range(n)], f32)
    V["_CSOLID_ARR"] = np.array([solid[m][1] for m in range(n)], f32)
    V["_KSOLID_ARR"] = np.array([solid[m][2] for m in range(n)], f32)
    V["_SPAN_ARR"] = np.array([span[m] for m in range(n)], f32)
    V["_EMISS_ARR"] = np.array([emiss[m] for m in range(n)], f32)
    V["_IGN_ARR"] = np.array([np.inf if solid[m][3] is None else solid[m][3]
                              for m in range(n)], f32)
    alpha = np.array([solid[m][2] / (solid[m][0] * solid[m][1] * 1000.0) for m in range(n)])
    V["_ALPHA"] = alpha
    V["_SKIN_DEPTH_ARR"] = np.sqrt(alpha * SKIN_T[0]).astype(f32)
    V["_SKIN_PATH_ARR"] = np.sqrt(alpha * SKIN_T[1]).astype(f32)
    V["_TRANSMIT_ARR"] = np.array([1.0 if m == AIR else transmit.get(m, 0.0)
                                   for m in range(n)], f32)

    # the particle table: rho, E, nu from the record; the solver numbers are legacy
    mpm = {}
    for r in rows:
        g = r.legacy.get("mpm")
        if g is None:
            continue
        mpm[r.key] = (r.rho, r.E, r.nu, g["yield_strain"][0], g["damp"][0],
                      g["strength"][0], g["rigid"][0])
    V["MPM"] = mpm
    V["mpm_row"] = lambda name: mpm[name]

    # the bond rows: E, nu, strength, damp, rho. The record gives the physics;
    # `legacy["bonds"]` holds each place the bond store differs, with its reason.
    def bond_tuple(r):
        return (_ov(r, "bonds", "E", r.E), _ov(r, "bonds", "nu", r.nu),
                _ov(r, "bonds", "st", r.st), _ov(r, "bonds", "damp", 0.1),
                _ov(r, "bonds", "rho", r.rho))
    wood = tb.row("wood")
    wood_l = bond_tuple(wood)
    wood_t = (wood.E_perp, wood.nu, wood.st_perp, _ov(wood, "bonds", "damp", 0.1),
              _ov(wood, "bonds", "rho", wood.rho))
    petiole = bond_tuple(tb.row("petiole"))
    named = {"WOOD_L": wood_l, "WOOD_T": wood_t, "PETIOLE": petiole,
             "PETIOLE_DRY": bond_tuple(tb.row("petiole_dry")),
             "TWIG": bond_tuple(tb.row("twig"))}
    bond_rows = {}
    for r in rows:
        if r.id in (AIR, WOOD, ASH):
            continue
        g = r.legacy.get("bonds", {}).get("row")
        row = bond_tuple(tb.row(g[0])) if g else bond_tuple(r)
        bond_rows[r.id] = row
        if not g:
            named[r.key.upper() + ("" if r.key == "rope" else "_B")] = row
    V["BOND_ROW"] = bond_rows
    V["NO_BOND"] = {ASH}
    named["PETIOLE_R"] = BOND_PETIOLE_R
    named["LEAF_AREA"] = BOND_LEAF_AREA
    V["BOND_CONSTS"] = named

    def hankinson(a, b, cos2):
        return a * b / (a * (1.0 - cos2) + b * cos2)

    def wood_row(cos2):
        return (hankinson(wood_l[0], wood_t[0], cos2), wood_l[1],
                hankinson(wood_l[2], wood_t[2], cos2), wood_l[3], wood_l[4])

    def bond_row(ma, mb, cos2=1.0):
        """The row of a bond between voxel materials ma and mb, or None."""
        ma, mb = int(ma), int(mb)
        if ma in V["NO_BOND"] or mb in V["NO_BOND"]:
            return None
        if ma == LEAF or mb == LEAF:
            return petiole
        ra = wood_row(cos2) if ma == WOOD else bond_rows.get(ma)
        rb = wood_row(cos2) if mb == WOOD else bond_rows.get(mb)
        if ra is None or rb is None:
            return None
        return ra if ra[2] <= rb[2] else rb
    V["bond_row"] = bond_row
    V["wood_row"] = wood_row
    return V
