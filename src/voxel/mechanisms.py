"""MECHANISMS (wave 13, T11): the MECHANISM record and the pure force laws.

A door, a latch, a bar, a spring, a lock: matter held by other matter. This
file has the RECORD (plain JSON-safe dicts: geometry and state, nothing that
a law computes) and the LAWS that read it (springs, a rigid plate on springs,
a door on its hinges, a two-body impact, damage, an exact quarter turn). It
knows no World: cells come in as integer arrays, material numbers through
`material_props`, and the one place a joule is booked is `deposit`.

CELL EDGE. d = 0.1 x scale = voxel_cm / 100 metres. Every length is derived
from it (`cell_edge_m`); nothing here says "5 cm".

THE ENERGY BOOKS (T11 4.0, the ONE-FLOW rule of the wave 13 plan). Elastic
energy, the kinetic energy of a swing and potential energy are off the
ledger; each act ends by turning them into heat inside one event, through
`deposit`. A joule an ACTOR delivers is debited from that actor and credited
to work_out; a joule with no actor (a spring let go, a slam from stored
swing, a burnt jamb) is booked in mech_in. One act books one or the other.

TAGS in the tests. test_echo_*: a constant was fitted to the number;
test_codeonly_*: the code against its own table; test_u_*: an independent
number whose source is unverified. Untagged tests are closed forms and
arithmetic identities.

Sources: Shigley (coil and cantilever springs, Wahl factor); Nadai,
Plasticity (torsion of an elastic-plastic bar); the Wood Handbook and the NDS
through fasteners.py; Hibbeler (impact, moments of inertia). (u) = unverified.
"""
import math

import numpy as np

from . import fasteners as fast

try:                                    # T9 lane: the one combine
    from . import matdamage as _matdamage
except ImportError:                     # stand-in; a cross test compares them
    _matdamage = None
try:                                    # T9 lane: the material record
    from . import matrec as _matrec
except ImportError:
    _matrec = None
try:                                    # the bond table: handbook numbers
    from ..bonds import materials as _bonds
    from . import materials as _vm
except ImportError:                     # pragma: no cover
    _bonds = None
    _vm = None

GRAVITY = 9.81
D_FAIL = 0.999                  # T9's D_MAX: at or above this a link holds nothing
SN_M = math.log(10.0) / 0.1     # (u) strength falls by 0.1 of itself per decade of blows
FLAW_AMP = 0.3                  # capacity scaled by 1 + 2 amp (flaw01 - 0.5): 0.7 to 1.3
K_DEFAULT = 1.0e6               # N/m for a link whose stiffness nobody gave (u)
SHEAR_OVER_YIELD_TORSION = 0.577  # tau_y = 0.577 sigma_y (von Mises)
MU_HINGE = 0.2                  # (u) hinge pin friction coefficient
R_PIN = 0.005                   # m (u) hinge pin radius


def cell_edge_m(voxel_cm):
    """Edge of one cell in metres: voxel_cm / 100 (= 0.1 x scale)."""
    return float(voxel_cm) / 100.0


# ── material numbers: the adapter ────────────────────────────────────────────
# NOT in the bond table: what a lock and a spring are made of. (u) handbook
# orders: brass E 100 GPa, yield 200 MPa, shear 200 MPa; steel is the bond
# table's iron row (200 GPa, 250 MPa).
EXTRA = {
    "brass": dict(E=1.0e11, nu=0.34, sy=2.0e8, st=3.0e8, tau=2.0e8, rho=8500.0,
                  ductility=0.3, damping=0.05),
}
# the wood shear strength along the grain is not in the bond table (u)
WOOD_TAU = 7.0e6


def yield_from_hardness(HV):
    """Tabor's rule: yield about HV x 9.81 / 3 MPa. A tempered spring steel
    (HV about 460) gives about 1.5 GPa, an annealed one (HV about 60) about
    0.2 GPa. This is how a spring's temper reaches its set: no table of
    springs."""
    return float(HV) * 9.81 / 3.0 * 1.0e6


def _names():
    if _vm is None:
        return {}
    return {"wood": _vm.WOOD, "stone": _vm.STONE, "iron": _vm.IRON,
            "steel": _vm.IRON, "glass": _vm.GLASS, "tin": _vm.TIN,
            "lead": _vm.LEAD, "flesh": _vm.FLESH, "char": _vm.CHAR,
            "leaf": _vm.LEAF, "rope": _vm.ROPE}


def _from_row(row, wood, across, mat_id):
    E, nu, st, damp, rho = row
    p = {"E": E, "E_perp": None, "nu": nu, "st": st, "st_perp": None,
         "sc": None, "damping": damp, "rho": rho, "ductility": 0.05}
    if wood:
        p["E_perp"] = _bonds.WOOD_T[0]
        p["st_perp"] = _bonds.WOOD_T[2]
        p["sy"] = None                         # wood breaks before it yields
        p["tau"] = WOOD_TAU
        p["ductility"] = 0.0
        if across:
            p["E"], p["st"] = _bonds.WOOD_T[0], _bonds.WOOD_T[2]
    else:
        p["sy"] = st                           # a metal's strength row is its yield
        p["tau"] = fast.SHEAR_OVER_YIELD * st
    return p


def material_props(mat, grain="along", HV=None, source="auto"):
    """The mechanical numbers of `mat` as a plain dict (E, E_perp, nu, sy, st,
    st_perp, sc, tau, ductility, rho, damping, source).

    `mat` is a voxel material id or a name ("wood", "iron", "steel", "brass",
    ...). `grain` is "along" or "across" (wood: E 0.55 GPa and 3 MPa across).
    `HV` replaces the yield by Tabor's rule (a tempered or annealed part).

    SOURCES. "bonds": src/bonds/materials.py, the handbook table this file
    stands in for, exactly (a test compares). "matrec": T9's MatRec through
    matrec.props, when the module exists. "auto": the bond table for what it
    holds, then brass, then matrec. At wiring the stopgap table is deleted
    and "auto" means matrec."""
    if grain not in ("along", "across"):
        raise ValueError("grain must be 'along' or 'across'")
    names = _names()
    key = mat.lower() if isinstance(mat, str) else None
    out = None
    if source in ("auto", "bonds") and _bonds is not None:
        mid = names.get(key) if key is not None else int(mat)
        if key in EXTRA:
            mid = None
        wood = mid is not None and int(mid) == int(_vm.WOOD)
        row = None
        if mid is not None:
            row = _bonds.wood_row(1.0) if wood else _bonds.ROW.get(int(mid))
        if row is not None:
            out = _from_row(row, wood, grain == "across", mid)
            out["source"] = "bonds"
    if out is None and key in EXTRA and source in ("auto", "bonds"):
        out = dict(EXTRA[key], E_perp=None, st_perp=None, sc=None, source="extra")
    if out is None and source in ("auto", "matrec") and _matrec is not None:
        k = key if key is not None else int(mat)
        out = dict(_matrec.props(k, grain))
        try:
            out["rho"] = float(_matrec.BASE.resolve(k).rho)
        except (AttributeError, KeyError):
            out["rho"] = None
        out["source"] = "matrec"
    if out is None:
        raise KeyError("no material numbers for %r (source %s)" % (mat, source))
    if HV is not None:
        out["sy"] = yield_from_hardness(HV)
    return out


# ── the record ───────────────────────────────────────────────────────────────
def new_joint(jid, type="hinge", a="leaf", b="frame", axis2=(0, 0), sign=1,
              limit_deg=(0.0, 100.0), mu_h=MU_HINGE, r_pin=R_PIN):
    """A joint. `axis2` is the pivot in DOUBLED cell units: an even pair is a
    cell centre, an odd pair a cell corner, so both are exact integers."""
    return {"id": jid, "type": type, "a": a, "b": b,
            "axis2": [int(axis2[0]), int(axis2[1])], "sign": int(sign),
            "limit_deg": [float(limit_deg[0]), float(limit_deg[1])],
            "fric_Nm_per_N": float(mu_h * r_pin),
            "theta": 0.0, "omega": 0.0, "links": []}


def new_link(lid, name, mode, mat, grain="along", geom=None, anchor=None):
    """A link in a load path. `geom` holds the numbers its mode needs (see
    `link_capacity`); `anchor` names the cells or eids that must still exist
    for it to hold. D is state; the capacity is a function."""
    return {"id": lid, "name": name, "mode": mode, "mat": mat, "grain": grain,
            "geom": dict(geom or {}), "anchor": list(anchor or []),
            "D": 0.0, "life": 0.0}


def new_catch(cid, kind="latchbolt", x_mm=0.0, throw_mm=15.0, engage_mm=10.0,
              spring=None, bevel_deg=45.0, at=None, side="open", path=None,
              held_by=None, blocked_by=None):
    return {"id": cid, "kind": kind, "x_mm": float(x_mm),
            "throw_mm": float(throw_mm), "engage_mm": float(engage_mm),
            "spring": spring, "bevel_deg": float(bevel_deg),
            "at": list(at) if at is not None else None, "side": side,
            "path": list(path or []), "held_by": held_by,
            "blocked_by": list(blocked_by) if blocked_by is not None else None}


def new_spring(sid, kind, mat, geom, x_m=0.0, set_m=0.0):
    return {"id": sid, "kind": kind, "mat": mat, "geom": dict(geom),
            "x_m": float(x_m), "set_m": float(set_m)}


def new_mechanism(mid, name, leaf=None, frame=None):
    """An empty MECHANISM record. There is NO mass in it: the mass, the centre
    of mass and the inertia of the leaf are summed from its cells whenever
    they are needed, so a burning or axed leaf stays right."""
    return {"mid": mid, "name": name, "leaf": leaf, "frame": frame,
            "joints": [], "catches": [], "locks": [], "springs": [],
            "links": [], "stop": {"links": []},
            "clear_mm": {"side": 3.0, "top": 3.0, "under": 12.0},
            "swing": None}


def engaged(catch):
    """A catch holds when its bolt is thrown past the engage point."""
    return catch["x_mm"] >= catch["engage_mm"]


# ── one joule, one book ──────────────────────────────────────────────────────
def _n_cells(cells):
    """How many cells a cell list names: an (n, 3) array is n; an index tuple of
    three arrays (the form numpy indexes with) is the length of one of them; a
    list of (x, y, z) is its length."""
    if isinstance(cells, np.ndarray):
        return int(cells.reshape(-1, 3).shape[0])
    if len(cells) == 3 and isinstance(cells[0], np.ndarray):
        return int(len(cells[0]))
    return len(cells)


def deposit(w, cells, joules, actor=None, heat=None, debit=None, work_out=None):
    """Turn `joules` into heat in `cells` and book them, ONCE.

    With an `actor` (a person's record) the joules were delivered by them:
    `debit(actor, J)` takes them from the actor's stores and `work_out(J)`
    credits the work_out book; `w.mech_in` is NOT touched. With no actor the
    joules came from nowhere that is booked (a spring let go, a door slamming
    from its stored swing, a burnt jamb): they are booked in `w.mech_in`.

    `heat(cells, J)` puts the heat in the lattice (phase B hands the world's
    own routine; None skips it, as the pure tests do). The default `debit`
    adds to actor["_work_J"], the stopgap until T7's energy store exists, and
    the default `work_out` adds to w.work_out when the world has that book.

    A deposit with a `heat` routine and joules to place but NO cell to put them
    in is refused (ValueError) before anything is booked: the joules would be
    booked and no heat would exist anywhere. The caller names the cells (the
    contact, the jamb, the actor's own hand) or deposits nothing."""
    J = float(joules)
    if J < 0.0:
        raise ValueError("a deposit is heat: joules must be >= 0")
    if heat is not None and J > 0.0:
        if _n_cells(cells) == 0:
            raise ValueError("a deposit of %g J names no cell to heat" % J)
        heat(cells, J)
    if actor is None:
        w.mech_in = float(getattr(w, "mech_in", 0.0)) + J
        return "mech_in"
    if debit is not None:
        debit(actor, J)
    else:
        actor["_work_J"] = float(actor.get("_work_J", 0.0)) + J
    if work_out is not None:
        work_out(J)
    elif hasattr(w, "work_out"):
        w.work_out = float(w.work_out) + J
    return "work_out"


# ── springs (F6) ─────────────────────────────────────────────────────────────
def coil_rate(d, D, n, G):
    """k = G d^4 / (8 D^3 n), N/m. d wire, D mean coil diameter, n active turns.
    0.30 mm wire on a 2.5 mm coil, 8 turns, G 79 GPa: 0.64 N/mm."""
    return G * d ** 4 / (8.0 * D ** 3 * n)


def wahl(C):
    """Wahl factor K_w = (4C - 1)/(4C - 4) + 0.615/C, C = D/d."""
    return (4.0 * C - 1.0) / (4.0 * C - 4.0) + 0.615 / C


def coil_peak_shear(F, d, D):
    """Peak shear in the wire: K_w 8 F D / (pi d^3)."""
    return wahl(D / d) * 8.0 * F * D / (math.pi * d ** 3)


def coil_yield_load(d, D, sigma_y):
    """The load at which the wire first yields: tau_y = 0.577 sigma_y at the
    Wahl-corrected peak."""
    tau_y = SHEAR_OVER_YIELD_TORSION * sigma_y
    return tau_y * math.pi * d ** 3 / (wahl(D / d) * 8.0 * D)


def leaf_rate(E, b, t, L):
    """Rectangular cantilever, end load: k = E b t^3 / (4 L^3)."""
    return E * b * t ** 3 / (4.0 * L ** 3)


def leaf_yield_load(sigma_y, b, t, L):
    """F_y = sigma_y b t^2 / (6 L)."""
    return sigma_y * b * t * t / (6.0 * L)


def torsion_rate(G, d, L):
    """A round bar in torsion: T/theta = pi G d^4 / (32 L), N m/rad."""
    return math.pi * G * d ** 4 / (32.0 * L)


def torsion_yield_torque(sigma_y, d):
    """T_y = tau_y pi d^3 / 16 with tau_y = 0.577 sigma_y."""
    return SHEAR_OVER_YIELD_TORSION * sigma_y * math.pi * d ** 3 / 16.0


def nadai_torque_ratio(r):
    """T/T_y of an elastic-perfectly-plastic round bar twisted to r = theta /
    theta_y: r below 1, then (4/3)(1 - 1/(4 r^3)) (Nadai, exact)."""
    r = float(r)
    return r if r <= 1.0 else (4.0 / 3.0) * (1.0 - 0.25 / r ** 3)


def nadai_residual(r):
    """The twist left after unloading elastically, in units of theta_y:
    r - T/T_y. 0 for r <= 1; 0.060, 0.265, 0.708, 1.679 at r = 1.2, 1.5, 2, 3."""
    r = float(r)
    return 0.0 if r <= 1.0 else r - nadai_torque_ratio(r)


def _leaf_curvature(mom):
    """kappa / kappa_y of a rectangular section at M / M_y = mom (elastic to
    1, then the plastic core: 1/sqrt(3 - 2 mom), up to the plastic hinge at
    1.5)."""
    if mom <= 1.0:
        return mom
    return 1.0 / math.sqrt(max(3.0 - 2.0 * mom, 1e-12))


def leaf_set_tip(m, arm="tip", n=200000):
    """The tip set of a leaf cantilever overloaded to m = F L / M_y (1 < m <=
    1.5), in units of kappa_y L^2: the numerical integral of the residual
    curvature (kappa - M/M_y)(kappa_y) over the yielded root zone, times the
    lever arm to the TIP.

    The moment falls linearly from the root, so only a short root zone
    yields. Midpoint rule, n points; the integrand has an integrable
    1/sqrt singularity only at m = 1.5. m 1.1 -> 5.1e-4, 1.2 -> 4.3e-3,
    1.4 -> 4.9e-2 (kappa_y L^2).

    arm="root": the spec's laws_v2.py figure (1.0e-5 at m 1.1, 1.8e-4 at 1.2),
    which weights the curvature by the distance from the ROOT, not from the
    tip. Wrong as a deflection (a curvature at the root moves the tip by
    kappa times the whole length), kept so the spec number is reproducible
    and the difference is tested. See test_echo_leaf_set_prototype_figure."""
    if m <= 1.0:
        return 0.0
    if m > 1.5:
        raise ValueError("m > 1.5 is a plastic hinge: use leaf_spring_set")
    s = (np.arange(n) + 0.5) / n                 # 0 at the root, 1 at the tip
    mom = m * (1.0 - s)
    kap = np.where(mom <= 1.0, mom,
                   1.0 / np.sqrt(np.maximum(3.0 - 2.0 * mom, 1e-12)))
    res = kap - mom
    lever = (1.0 - s) if arm == "tip" else s
    return float((res * lever).sum() / n)


def leaf_set_uniform(m):
    """The UPPER BOUND on the leaf set at m: the root's residual curvature
    over the whole length (a uniform moment m everywhere), times L^2/2, in
    kappa_y L^2 units. The real set is below it because the moment falls."""
    if m <= 1.0:
        return 0.0
    return (_leaf_curvature(m) - m) * 0.5


def leaf_spring_set(x, k, F_y, kappa_y_L2):
    """The set (m) of a leaf spring pushed to a tip deflection x (m): k its
    rate, F_y its yield load, kappa_y_L2 = kappa_y L^2 = 2 sigma_y L^2/(E t).

    Up to the yield deflection nothing. Between m = 1 and 1.5 the load rises
    and the deflection is m F_y/k plus the set of leaf_set_tip(m); the map is
    monotonic and inverted by bisection. Past the plastic hinge (m = 1.5) the
    load stays at 1.5 F_y and the set is the applied deflection minus the
    elastic recovery, x - 1.5 F_y/k."""
    x = abs(float(x))
    x_y = F_y / k
    if x <= x_y:
        return 0.0
    x15 = 1.5 * x_y + leaf_set_tip(1.5) * kappa_y_L2
    if x >= x15:
        return x - 1.5 * x_y
    lo, hi = 1.0, 1.5
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if mid * x_y + leaf_set_tip(mid, n=4000) * kappa_y_L2 < x:
            lo = mid
        else:
            hi = mid
    m = 0.5 * (lo + hi)
    return leaf_set_tip(m, n=20000) * kappa_y_L2


def spring_props(spring):
    """G and yield of a spring's wire or blade: from its material, with the
    temper (geom["HV"], Tabor) or an explicit geom["sy"] / geom["G"]."""
    g = spring["geom"]
    p = material_props(spring["mat"], HV=g.get("HV"))
    sy = g.get("sy", p["sy"])
    G = g.get("G", p["E"] / (2.0 * (1.0 + p["nu"])))
    return {"E": p["E"], "G": G, "sy": sy}


def spring_rate(spring):
    """N/m (coil, leaf) or N m/rad (torsion), from geometry and material."""
    g, p = spring["geom"], spring_props(spring)
    if spring["kind"] == "coil":
        return coil_rate(g["d"], g["D"], g["n"], p["G"])
    if spring["kind"] == "leaf":
        return leaf_rate(p["E"], g["b"], g["t"], g["L"])
    if spring["kind"] == "torsion":
        return torsion_rate(p["G"], g["d"], g["L"])
    raise ValueError("spring kind %r" % spring["kind"])


def spring_limit(spring):
    """The elastic limit as a deflection (m) or twist (rad): the load that
    first yields over the rate."""
    g, p = spring["geom"], spring_props(spring)
    k = spring_rate(spring)
    if spring["kind"] == "coil":
        return coil_yield_load(g["d"], g["D"], p["sy"]) / k
    if spring["kind"] == "leaf":
        return leaf_yield_load(p["sy"], g["b"], g["t"], g["L"]) / k
    return torsion_yield_torque(p["sy"], g["d"]) / k


def spring_set(spring, x):
    """The set a spring keeps after being pushed to x (m, or rad for a
    torsion bar) and let go, one law per kind.

    coil, torsion: the wire is a torsion bar at its Wahl-corrected peak, so
    Nadai: the set is nadai_residual(r) x the yield deflection, r = x / x_y.
    leaf: the root-zone integral (leaf_spring_set)."""
    x = abs(float(x))
    x_y = spring_limit(spring)
    if spring["kind"] in ("coil", "torsion"):
        return nadai_residual(x / x_y) * x_y
    g, p = spring["geom"], spring_props(spring)
    kappa_y_L2 = 2.0 * p["sy"] * g["L"] ** 2 / (p["E"] * g["t"])
    return leaf_spring_set(x, spring_rate(spring),
                           leaf_yield_load(p["sy"], g["b"], g["t"], g["L"]),
                           kappa_y_L2)


def spring_energy(k, x):
    """Stored elastic energy k x^2 / 2. Off the ledger until released."""
    return 0.5 * k * x * x


# ── a leaf on its hinge ──────────────────────────────────────────────────────
def _pivot_dist2(cells, pivot2, d):
    """Squared plan distance (m^2) from the pivot of integer cells (n, >=2)."""
    c = np.asarray(cells, np.int64)
    dx = (2 * c[:, 0] - int(pivot2[0])) * 0.5 * d
    dy = (2 * c[:, 1] - int(pivot2[1])) * 0.5 * d
    return dx * dx + dy * dy


def leaf_mass_cells(masses):
    return float(np.sum(masses))


def leaf_inertia(cells, masses, pivot2, d):
    """I = sum m_i r_i^2 + m_i d^2 / 6 about the vertical hinge axis, from the
    leaf's cells: r_i the plan distance of cell centre i from the pivot, and
    d^2/6 each cell's own inertia about its centre (a square of side d). A
    uniform slab of width W and mass m gives m W^2/3 (+ m d^2/12)."""
    m = np.asarray(masses, float)
    return float((m * (_pivot_dist2(cells, pivot2, d) + d * d / 6.0)).sum())


def leaf_com(cells, masses, d):
    """Centre of mass of the cells, in metres, (x, y, z) in cell-centre
    coordinates times d."""
    c = np.asarray(cells, float)[:, :3] * d
    m = np.asarray(masses, float)
    return (c * m[:, None]).sum(axis=0) / m.sum()


def hinge_friction(mu_h, r_pin, normal_N):
    """tau_f = mu_h r_pin N, the friction torque of the pins under the leaf's
    weight: 0.2 x 5 mm x 38 kg x g = 0.37 N m."""
    return mu_h * r_pin * normal_N


def net_open_torque(pushes, tau_f=0.0, hold=0.0):
    """The torque that turns the leaf. `pushes` is a list of (force N, arm m),
    positive toward opening. The sum of torques decides, not who pushes.
    Friction opposes either way; an engaged catch's `hold` opposes OPENING
    only. Returns the signed net torque (0 if the leaf does not move)."""
    tau = float(sum(f * a for f, a in pushes))
    if tau > 0.0:
        net = tau - tau_f - hold
        return net if net > 0.0 else 0.0
    if tau < 0.0:
        net = tau + tau_f
        return net if net < 0.0 else 0.0
    return 0.0


def swing_time_closed(I, tau_net, theta):
    """Closed form for a constant net torque from rest: t = sqrt(2 theta I /
    tau_net). 20 N at 0.8 m on the 43 kg leaf with 0.37 N m of friction opens
    a quarter turn in 1.52 s."""
    return math.sqrt(2.0 * theta * I / tau_net)


def integrate_swing(I, tau_push, tau_f, theta_end, dt=1e-4):
    """A numerical integration of the same swing, with Coulomb friction
    (opposes motion; holds the leaf still if the push is under it), semi-
    implicit Euler, the crossing time interpolated inside the last step.
    Returns (t, omega at theta_end). Independent of the closed form: it does
    not assume a constant net torque."""
    th, om, t = 0.0, 0.0, 0.0
    if abs(tau_push) <= tau_f:
        return math.inf, 0.0
    while True:
        tau = tau_push - (tau_f if om > 0.0 or tau_push > 0.0 else -tau_f)
        om_new = om + tau / I * dt
        th_new = th + om_new * dt
        if th_new >= theta_end:
            frac = (theta_end - th) / (th_new - th)
            return t + frac * dt, om + (om_new - om) * frac
        th, om, t = th_new, om_new, t + dt


def latch_closing_force(F_spring, bevel_deg=45.0):
    """The push that drives a bevelled bolt back as the door shuts:
    F_spring / tan(bevel). 15 N of spring at 45 degrees: 15 N."""
    return F_spring / math.tan(math.radians(bevel_deg))


# ── a rigid plate on springs ─────────────────────────────────────────────────
def plate_reactions(supports, loads):
    """STATICS OF A RIGID LEAF HELD AT SEVERAL POINTS.

    `supports`: [(x, y, k)], the support points (hinge knuckles on the axis;
    each engaged catch at its bolt line) and their stiffness k (N/m).
    `loads`: [(F, x, y)], forces normal to the leaf at their contact points.
    The plate moves rigidly, w = c0 + cx x + cy y, and each support carries
    R_i = k_i w(x_i, y_i). Three equations (sum of forces, two moments) give
    the three constants. Returns the array of R_i, positive where the support
    pushes back against the load; a NEGATIVE reaction is a support in
    tension (a hinge pulling out under a contact beyond the latch)."""
    A = np.zeros((3, 3))
    for x, y, k in supports:
        v = np.array([1.0, x, y])
        A += k * np.outer(v, v)
    rhs = np.zeros(3)
    for F, x, y in loads:
        rhs += F * np.array([1.0, x, y])
    try:
        c = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        raise ValueError("the supports cannot hold the plate (collinear or none)")
    return np.array([k * (c @ np.array([1.0, x, y])) for x, y, k in supports])


# ── the two-body impact ──────────────────────────────────────────────────────
def two_body_impact(m_s, v_s, I, a, k_c, k_path, e_c=0.2, F_cap=None,
                    k_link=None, ductility=0.0, leaf_held=True):
    """A striker (effective mass m_s, speed v_s) hits a leaf (inertia I about
    its axis) at distance a from the axis.

    The leaf's effective mass at the contact is m_t = I/a^2, the reduced mass
    mu = m_s m_t/(m_s + m_t), the relative speed v_rel = v_s (leaf at rest).
    The contact (k_c) and the path (k_path, already REFERRED to the contact)
    are in series: k_s. The peak force is F_peak = v_rel sqrt(mu k_s); only
    1/2 mu v_rel^2 of the striker's energy is ever in the spring.

    No failure (F_cap None or above F_peak): the striker rebounds with
    restitution e_c; the impulse is J = mu (1 + e_c) v_rel; the leaf takes J a
    of angular momentum, I omega = J a. Angular momentum about the axis
    m_s a v is conserved exactly (hinge forces act at the axis). If the leaf
    is held by a catch, its rebound KE is dissipated; otherwise it swings.

    Failure (F_peak >= F_cap): the failing link absorbs F_cap^2/(2 k_link)
    (1 + ductility), the rest of the relative energy goes on as relative
    speed v', and momentum gives omega = m_s a (v_s - v')/(I + m_s a^2).

    Returns a dict. `heat_J` is what is neither in the striker nor in the
    leaf afterwards; all energy bookkeeping closes: E0 = foot KE + leaf KE
    (only if free) + heat."""
    m_t = I / (a * a)
    mu = m_s * m_t / (m_s + m_t)
    k_s = fast.series_k(k_c, k_path)
    v_rel = float(v_s)
    F_peak = v_rel * math.sqrt(mu * k_s)
    E0 = 0.5 * m_s * v_s * v_s
    E_def = 0.5 * mu * v_rel * v_rel
    out = {"m_t": m_t, "mu": mu, "k_s": k_s, "F_peak": F_peak, "E0": E0,
           "E_def": E_def, "L_before": m_s * a * v_s, "failed": False}
    if F_cap is None or F_peak < F_cap:
        J = mu * (1.0 + e_c) * v_rel
        v_f = v_s - J / m_s
        om = J * a / I
        ke_f, ke_l = 0.5 * m_s * v_f * v_f, 0.5 * I * om * om
        ke_leaf_kept = 0.0 if leaf_held else ke_l
        out.update(J=J, v_foot=v_f, omega=om, KE_foot=ke_f, KE_leaf=ke_l,
                   KE_leaf_kept=ke_leaf_kept, L_after=m_s * a * v_f + I * om,
                   heat_J=E0 - ke_f - ke_leaf_kept)
        return out
    kl = k_link if k_link is not None else k_path
    E_abs = F_cap ** 2 / (2.0 * kl) * (1.0 + ductility)
    E_rel = max(0.0, E_def - E_abs)
    v_new = math.sqrt(2.0 * E_rel / mu)
    om = m_s * a * (v_s - v_new) / (I + m_s * a * a)
    v_f = a * om + v_new
    ke_f, ke_l = 0.5 * m_s * v_f * v_f, 0.5 * I * om * om
    out.update(failed=True, E_abs=E_abs, E_rel=E_rel, v_rel_after=v_new,
               omega=om, v_foot=v_f, KE_foot=ke_f, KE_leaf=ke_l,
               KE_leaf_kept=ke_l, L_after=m_s * a * v_f + I * om,
               heat_J=E0 - ke_f - ke_l)
    return out


# ── damage (T9's D, one combine) ─────────────────────────────────────────────
def combine(D, dD):
    """T9's combine, 1 - (1 - D)(1 - dD), clipped to [0, 0.999]. The stand-in
    has the same law; matdamage.combine is used when the module exists."""
    if _matdamage is not None:
        return float(_matdamage.combine(D, dD))
    r = 1.0 - (1.0 - D) * (1.0 - dD)
    return float(min(max(r, 0.0), D_FAIL))


def blow_life(r, m=SN_M):
    """The share of a link's life one blow of r = F_peak/cap0 uses: 1 for r >=
    1, else exp(-m (1 - r)) = 10^-((1 - r)/0.1) (strength falls 0.1 of itself
    per decade of blows). (u): the slope is a stand-in, tuned to a few to ten
    kicks on a latched door. r 0.97 -> 0.50, 0.9 -> 0.10, 0.8 -> 0.010,
    0.5 -> 1e-5."""
    return 1.0 if r >= 1.0 else math.exp(-m * (1.0 - r))


def blow_damage(D, life, F, cap0, m=SN_M):
    """One blow on one link. The S-N reading is about the PRISTINE strength
    (cap0), and a life share is additive (Miner): life += blow_life(r). The
    link fails when life reaches 1 or F reaches cap0. D only goes up (combine
    of the share this blow added); a failed link has D = 1.

    Returns (D_new, life_new, failed)."""
    life_new = life + blow_life(F / cap0, m)
    failed = life_new >= 1.0 or F >= cap0
    if failed:
        return 1.0, life_new, True
    dD = min(life_new, D_FAIL) - min(life, D_FAIL)
    return max(D, combine(D, dD)), life_new, False


def static_fails(F, cap0, D):
    """A held (not repeated) load against what is left: F >= cap0 (1 - D). A
    damaged latch gives way to a push it once held."""
    return F >= fast.scale_damage(cap0, D) or D >= D_FAIL


def link_failed(link):
    return link["D"] >= D_FAIL


def link_capacity(link, flaw01=0.5, flaw_amp=FLAW_AMP):
    """PRISTINE capacity (N) of a link from its mode, geometry and material,
    scaled by 1 + 2 amp (flaw01 - 0.5) (0.7 to 1.3 at the default, the
    _spend convention) so identical doors differ, deterministically.

    modes and geom: shear {d, n}; tension {d, n}; withdrawal {G, D, L, n}
    (m); lateral {G, D, l, n [, Fyb, Re]}; split {l_s, t_e [, sigma]};
    bearing {A [, sigma]}; bend {b, d, span}; panel {perimeter, t [, tau_u]}.
    geom["cap_N"] overrides any mode (a plate whose capacity someone else
    computed)."""
    g, mode = link["geom"], link["mode"]
    if "cap_N" in g:
        cap = float(g["cap_N"])
    elif mode == "shear":
        p = material_props(link["mat"])
        cap = fast.shear_round(p["sy"], g["d"], g.get("n", 1))
    elif mode == "tension":
        p = material_props(link["mat"], link["grain"])
        cap = fast.tension_round(p["sy"] or p["st"], g["d"], g.get("n", 1))
    elif mode == "withdrawal":
        cap = g.get("n", 1) * fast.screw_withdrawal(g["G"], g["D"], g["L"])
    elif mode == "lateral":
        cap = fast.dowel_lateral_ultimate(
            g["G"], g["D"], g["l"], g.get("n", 1),
            Fyb=g.get("Fyb", fast.FYB_SCREW), Re=g.get("Re", 0.0))
    elif mode == "split":
        sig = g.get("sigma", material_props(link["mat"], "across")["st"])
        cap = fast.split_across_grain(sig, g["l_s"], g["t_e"])
    elif mode == "bearing":
        sig = g.get("sigma", material_props(link["mat"], "across")["st"])
        cap = fast.bearing_crush(sig, g["A"])
    elif mode == "bend":
        p = material_props(link["mat"], link["grain"])
        cap = fast.bar_bending(g.get("sigma", p["sy"] or p["st"]),
                               g["b"], g["d"], g["span"])
    elif mode == "panel":
        cap = fast.punching(g.get("tau_u", fast.PUNCH_TAU), g["perimeter"], g["t"])
    else:
        raise ValueError("link mode %r" % mode)
    return cap * (1.0 + 2.0 * flaw_amp * (flaw01 - 0.5))


def link_capacity_now(link, **kw):
    """The capacity after damage: cap0 (1 - D)."""
    return fast.scale_damage(link_capacity(link, **kw), link["D"])


def link_stiffness(link):
    """N/m of one link, after damage: geom["k"] if given, else from the mode
    (tension: E A/L; bearing: E_perp A/t; bend: the simply supported beam
    48 E I/L^3), else K_DEFAULT (u). Times (1 - D)."""
    g, mode = link["geom"], link["mode"]
    if "k" in g:
        k = float(g["k"])
    elif mode == "tension" and "A" in g:
        k = fast.axial_k(material_props(link["mat"], link["grain"])["E"], g["A"], g["L"])
    elif mode == "bearing" and "t" in g:
        p = material_props(link["mat"], "across")
        k = fast.crush_k(p["E"], g["A"], g["t"])
    elif mode == "bend":
        p = material_props(link["mat"], link["grain"])
        k = 48.0 * p["E"] * fast.rect_I(g["b"], g["d"]) / g["span"] ** 3
    else:
        k = K_DEFAULT
    return fast.scale_damage(k, link["D"])


def path_stiffness(links):
    """Links in a row: 1/k = sum 1/k_i, damage included."""
    return fast.series_k(*[link_stiffness(lk) for lk in links])


def weakest_link(links, flaws=None, flaw_amp=FLAW_AMP):
    """(index, capacities): the link of least capacity now. `flaws` maps link
    id to its flaw01 (default 0.5, the mean)."""
    flaws = flaws or {}
    caps = [link_capacity_now(lk, flaw01=flaws.get(lk["id"], 0.5),
                              flaw_amp=flaw_amp) for lk in links]
    return int(np.argmin(caps)), caps


def link_absorb(F_cap, k, ductility=0.0):
    """Energy a failing link takes up: F_cap^2/(2 k) (1 + ductility); 0 for
    brittle wood, up to 1 for nail or steel (u)."""
    return F_cap ** 2 / (2.0 * k) * (1.0 + ductility)


def blow_path(links, F, flaws=None, flaw_amp=FLAW_AMP, m=SN_M):
    """One blow of peak force F down a path of links in series (each carries
    F). Updates D and life of every link, in place, and returns the ids of the
    links that failed this blow (the weakest first). Failed links stay failed.
    The caller picks the path: the catch's path for a push on the opening
    side, the stop's links for a push on the closing side."""
    flaws = flaws or {}
    out = []
    for lk in links:
        if link_failed(lk):
            continue
        cap0 = link_capacity(lk, flaw01=flaws.get(lk["id"], 0.5), flaw_amp=flaw_amp)
        lk["D"], lk["life"], failed = blow_damage(lk["D"], lk["life"], F, cap0, m)
        if failed:
            out.append((cap0, lk["id"]))
    return [i for _, i in sorted(out)]


# ── the exact quarter turn ───────────────────────────────────────────────────
def quarter_turn(cells, s, pivot2):
    """Rotate integer cells by s x 90 degrees (s = +1 anticlockwise, -1
    clockwise) about the pivot (px2, py2) in DOUBLED coordinates, with no
    rounding anywhere:
        x' = (px2 - s (2y - py2)) / 2     y' = (py2 + s (2x - px2)) / 2
    Both are whole numbers exactly when px2 and py2 have the same parity: a
    cell centre (both even) or a cell corner (both odd). With a corner pivot
    a leaf row (x0, y0 + j) hinged at (x0 - 1/2, y0 - 1/2) goes to
    (x0 - j - 1, y0). Cells are (n, 2) or (n, 3); z is untouched."""
    c = np.asarray(cells, np.int64)
    px2, py2 = int(pivot2[0]), int(pivot2[1])
    if (px2 - py2) % 2:
        raise ValueError("pivot parity: both doubled coordinates must be even "
                         "(a cell centre) or both odd (a cell corner)")
    if s not in (1, -1):
        raise ValueError("s is +1 or -1")
    out = c.copy()
    nx = px2 - s * (2 * c[:, 1] - py2)
    ny = py2 + s * (2 * c[:, 0] - px2)
    assert not ((nx % 2).any() or (ny % 2).any())
    out[:, 0] = nx // 2
    out[:, 1] = ny // 2
    return out


def quarter_turn_float(cells, s, pivot2):
    """The same turn through a float rotation matrix about the pivot, rounded
    last: the independent reference the exact form is tested against."""
    c = np.asarray(cells, float)
    cx, cy = pivot2[0] / 2.0, pivot2[1] / 2.0
    th = s * math.pi / 2.0
    ct, st = math.cos(th), math.sin(th)
    dx, dy = c[:, 0] - cx, c[:, 1] - cy
    out = c.copy()
    out[:, 0] = cx + ct * dx - st * dy
    out[:, 1] = cy + st * dx + ct * dy
    return np.rint(out).astype(np.int64)


def turn_exchange(leaf, s, pivot2):
    """The EXCHANGE a swing makes: the leaf cells go to their destinations and
    the non-leaf cells that were there (room air, with its heat and fluid)
    go into the cells the leaf vacated, so nothing is lost.

    Returns (dest, entered, vacated): `dest` the leaf cells' new positions in
    the leaf's own order; `entered` the destination cells that were not leaf
    cells and `vacated` the leaf cells that are not destinations, both sorted
    lexicographically so the pairing entered[i] -> vacated[i] is
    deterministic. Their counts are equal because the turn is a
    permutation."""
    leaf = np.asarray(leaf, np.int64)
    dest = quarter_turn(leaf, s, pivot2)
    src = {tuple(t) for t in leaf.tolist()}
    dst = {tuple(t) for t in dest.tolist()}
    entered = sorted(dst - src)
    vacated = sorted(src - dst)
    assert len(entered) == len(vacated)
    return dest, np.array(entered, np.int64).reshape(-1, leaf.shape[1]), \
        np.array(vacated, np.int64).reshape(-1, leaf.shape[1])


def sweep_span(cells, pivot2, s):
    """The bearings (a0, a1), radians, a0 < a1, swept by a leaf's cells turning
    s x 90 degrees about the pivot: from the cell centre of least bearing to
    the cell centre of greatest bearing after the turn. The leaf's own width
    (a hinge at a corner puts its first cells off the leaf's long axis) is
    inside the span."""
    c = np.asarray(cells, float)
    dx, dy = c[:, 0] - pivot2[0] / 2.0, c[:, 1] - pivot2[1] / 2.0
    b = np.arctan2(dy, dx)
    rel = (b - b[0] + math.pi) % (2.0 * math.pi) - math.pi
    lo, hi = b[0] + rel.min(), b[0] + rel.max()
    return (lo, hi + math.pi / 2.0) if s > 0 else (lo - math.pi / 2.0, hi)


def swept_sector(pivot2, r_leaf_cells, a0, a1, z_range, shape=None):
    """Cells swept by a leaf turning from bearing a0 to a1 (radians, a1 > a0,
    at most pi apart) about the pivot: the cells whose centre is within
    r_leaf_cells + 1/2 of the axis and between the two bearings, over
    z_range = (zlo, zhi) inclusive. A cell at the axis counts. Returns an
    (n, 3) int array; `shape` (nx, ny, nz) clips it to the world."""
    px, py = pivot2[0] / 2.0, pivot2[1] / 2.0
    R = r_leaf_cells + 0.5
    x0, x1 = int(math.floor(px - R)), int(math.ceil(px + R))
    y0, y1 = int(math.floor(py - R)), int(math.ceil(py + R))
    xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1),
                         indexing="ij")
    dx, dy = xs - px, ys - py
    r = np.hypot(dx, dy)
    ang = np.arctan2(dy, dx)
    rel = (ang - a0) % (2.0 * math.pi)
    inside = (r <= R + 1e-9) & ((rel <= (a1 - a0) + 1e-9) |
                                (rel >= 2.0 * math.pi - 1e-9)) | (r < 1e-9)
    pts = np.stack([xs[inside], ys[inside]], axis=1)
    zs = np.arange(z_range[0], z_range[1] + 1)
    out = np.concatenate([np.column_stack([pts, np.full(len(pts), z)]) for z in zs]
                         ) if len(pts) else np.zeros((0, 3), np.int64)
    if shape is not None:
        ok = ((out >= 0) & (out < np.asarray(shape))).all(axis=1)
        out = out[ok]
    return out.astype(np.int64)


# ── wedge and bar ────────────────────────────────────────────────────────────
def wedge_self_locks(alpha_deg, mu_a, mu_b):
    """A wedge of INCLUDED angle alpha between two faces of friction mu_a and
    mu_b holds (self-locks) when alpha <= atan(mu_a) + atan(mu_b): the sum of
    the two friction angles. 0.5 + 0.5 gives 53.13 degrees (not the 45 of the
    wrong form tan(alpha) <= mu_a + mu_b)."""
    return alpha_deg <= wedge_limit_deg(mu_a, mu_b)


def wedge_limit_deg(mu_a, mu_b):
    return math.degrees(math.atan(mu_a) + math.atan(mu_b))


def bar_hold(sigma_y, b, d, span, cleat_screws_N, leaf_bearing_N):
    """What a bar in two cleats holds: the smallest of its bending, its cleat
    screws in lateral shear and the leaf edge bearing on it. Returns (N,
    which)."""
    return fast.bar_holds(fast.bar_bending(sigma_y, b, d, span),
                          cleat_screws_N, leaf_bearing_N)
