"""JOINERY: joints, cure, composites and fusing (F6).

Stand-alone (wave 13, topic T2, spec section 9). Nothing here imports World.
Plain numpy and the standard library are the definition.

What it keeps:
  * A JOIN is made by an act (glue, nail, tie, sew, mortar, weld, clamp). The
    KIND is a row of data (JOIN_KINDS); nothing is "glued" or "strong" by flag.
    The capacity is computed per loading mode (tension Ft, shear Fs, moment Mc,
    compression Fc) from the row, the contact area and the cure state alpha,
    and each mode is capped by the MATCHING strength of the weaker substrate
    (shear by wood shear parallel to the grain, tension by wood tension
    PERPENDICULAR to the grain, not the shear figure).
  * Cure is a state that grows by law: dry (loss of water), set on cool (hide
    glue, reversible), hydration maturity (ASTM C1074 Arrhenius age, ACI
    209R-92 curve normalised by its 28-day value and clamped), carbonation
    (K sqrt(t) while the air is in the humidity band), and a weld, which is a
    RATE that is zero outside its window (below the diffusion-bond
    temperature, above the solidus, or under a thick oxide) -- no gate. An
    uncured joint (alpha 0) holds nothing.
  * A composite is a GROUP: its parts keep their own identity, and its joints
    are listed. It falls apart when the graph of joints that have not failed
    splits (union-find), at the weakest joint. Parallel joints share load by
    stiffness (a stand-in, (u)); the shared load never exceeds the sum of
    capacities.
  * FUSING makes one new record: masses add per species and per form, thermal
    energy adds, T comes from the summed enthalpy, counts add. The parts end
    with cause 'fused'.

Numbers marked (u) are unverified. Numbers marked (c) were computed from the
formula shown.
"""
import math
from collections import namedtuple

from .fasteners import PSI as _PSI      # Pa per psi: the NDS embedment is typed in psi, so it is converted here, not retyped in MPa
from .record_data import JOIN_KINDS, SUBSTRATE, dowel_Rd  # noqa: F401  (the ONE copies of the joint rows)

try:                                    # T11's fastener formulas (wave 13 phase B)
    from . import fasteners as _fasteners
except ImportError:
    _fasteners = None

Capacity = namedtuple("Capacity", "Ft Fs Mc Fc")
ZERO = Capacity(0.0, 0.0, 0.0, 0.0)

T_REF = 293.15              # K, the 20 C reference of the lattice
R_GAS = 8.314462618         # J/(mol K)


# The substrate strengths (SUBSTRATE) and the joint kinds (JOIN_KINDS) are DATA in record_data,
# imported above: one copy, and SUBSTRATE's tension figures are the STRENGTH rows' sigma_u.


def wood_density_G(rho_kg_m3):
    """Specific gravity (relative to water) from density."""
    return rho_kg_m3 / 1000.0


# ── fasteners ─────────────────────────────────────────────────────────────────
def nail_withdrawal(G, D_mm, L_mm):
    """Wood Handbook ultimate withdrawal, N: P = 54.12 G^2.5 D per mm of
    penetration. G 0.5, D 3.3 mm, 38 mm in: 1199.7 N (c), about 1.2 kN."""
    return 54.12 * G ** 2.5 * D_mm * L_mm


def nail_allowable(G, D_mm, L_mm):
    """The NDS allowable is about one fifth of the ultimate."""
    return nail_withdrawal(G, D_mm, L_mm) / 5.0


# STAND-IN COPY of T11 fasteners.py (screw withdrawal and lateral yield). At
# wiring the copy is replaced by an import in the same commit; lane X compares
# them (tests/test_w13_cross.py).
def screw_withdrawal_standin(G, D_mm, L_mm):
    """Wood Handbook ultimate: 108.25 G^2 D L (N; D and L in mm). #10 (4.83 mm)
    in pine G 0.45, 19 mm of thread: 2.0 kN (c)."""
    return 108.25 * G ** 2 * D_mm * L_mm


def dowel_lateral_yield_standin(G, D_mm, Fyb=600e6, Rd=None):
    """Mode IV yield of a dowel or screw bending in wood (NDS 12.3):
    Fe = 16600 G^1.84 psi = 114.453 MPa (Wood Handbook eq. 8-3 prints 114.5, sourced), Z = (D^2/Rd) sqrt(2 Fe Fyb/3), N.
    Rd comes from dowel_Rd(D) (ONE copy, fasteners.py: NDS-2018 Table 12.3.1B, 2.2 up to 0.17 in,
    10 D + 0.5 to 0.25 in, 3.2 above), unless the caller gives one. Fyb (u)."""
    Fe = 16600.0 * _PSI * G ** 1.84
    D = D_mm * 1e-3
    if Rd is None:
        Rd = float(dowel_Rd(D))
    return D * D / Rd * math.sqrt(2.0 * Fe * Fyb / 3.0)


def screw_withdrawal(G, D_mm, L_mm):
    """Ultimate screw withdrawal, N. D and L come in mm; fasteners.py takes
    metres, so the delegation converts."""
    f = getattr(_fasteners, "screw_withdrawal", None)
    if f:
        return f(G, D_mm * 1e-3, L_mm * 1e-3)
    return screw_withdrawal_standin(G, D_mm, L_mm)


def capstan_ratio(mu, phi):
    """T_load / T_hold = exp(mu phi); three turns (phi = 6 pi) at mu 0.3: 285."""
    return math.exp(mu * phi)


def lashing_capacity(break_load, knot_eff, turns, mu, tail_hold):
    """A lashing holds the smaller of what friction lets the tail hold
    (tail_hold x exp(mu phi)) and the rope after the knot (break x efficiency).
    Few turns slip; many turns break at about the knot's share of the rope."""
    slip = tail_hold * capstan_ratio(mu, turns * 2.0 * math.pi)
    brk = knot_eff * break_load
    return min(slip, brk), ("slip" if slip < brk else "break")


# ── capacity per loading mode ─────────────────────────────────────────────────
def _caps(substrate):
    names = (substrate,) if isinstance(substrate, str) else tuple(substrate)
    rows = [SUBSTRATE[n] for n in names]
    return (min(r["tau"] for r in rows), min(r["sig_t"] for r in rows),
            min(r["sig_c"] for r in rows))


def overlap_factor(length, lap0):
    """Glue-line strength falls with overlap length: sqrt(lap0/length) beyond
    the reference lap (u)."""
    if not length or length <= lap0:
        return 1.0
    return math.sqrt(lap0 / length)


def capacity(kind, alpha=1.0, area=0.0, substrate="wood", length=None,
             G=0.5, D_mm=3.3, L_mm=38.0, n=1, break_load=0.0, turns=0.0,
             mu=0.3, tail_hold=0.0, thread_break=0.0, cloth_tear=math.inf,
             normal=0.0, rows=None):
    """Capacity of one join by loading mode: Capacity(Ft N, Fs N, Mc N m, Fc N).
    Zero when alpha is 0, monotone in alpha, and each mode never above the weaker
    substrate's matching strength times the contact area."""
    row = (rows or JOIN_KINDS)[kind] if isinstance(kind, str) else kind
    cls = row["cls"]
    a = max(min(float(alpha), 1.0), 0.0)
    if a == 0.0 and cls != "clamp":
        return ZERO
    s_tau, s_t, s_c = _caps(substrate)
    if cls == "adhesive":
        f = overlap_factor(length, row["lap0"])
        Fs = min(row["tau_a"], s_tau) * area * a * f
        Ft = min(row["t_ratio"] * row["tau_a"], s_t) * area * a
        Mc = Ft * math.sqrt(area) / 6.0
        return Capacity(Ft, Fs, Mc, s_c * area)
    if cls == "nail":
        w = nail_withdrawal(G, D_mm, L_mm)
        lat = row["lateral_ult"] * dowel_lateral_yield_standin(G, D_mm)
        Fs = min(lat, s_c * D_mm * L_mm * 1e-6 * 4)
        return Capacity(n * w, n * Fs, n * w * L_mm * 1e-3 / 4.0, 0.0)
    if cls == "screw":
        w = screw_withdrawal(G, D_mm, L_mm)
        lat = row["lateral_ult"] * dowel_lateral_yield_standin(G, D_mm)
        return Capacity(n * w, n * lat, n * w * L_mm * 1e-3 / 4.0, 0.0)
    if cls == "lashing":
        cap, _ = lashing_capacity(break_load, row["knot_eff"], turns, mu, tail_hold)
        return Capacity(cap, cap, 0.0, 0.0)
    if cls == "sewn":
        F = min(n * row["eff"] * thread_break, cloth_tear)
        return Capacity(F, F, 0.0, 0.0)
    if cls == "mortar":
        Ft = min(row["sig_t"], s_t) * area * a
        Fs = min(row["tau"], s_tau) * area * a
        Fc = min(row["sig_c"], s_c) * area * a
        return Capacity(Ft, Fs, Ft * math.sqrt(area) / 6.0, Fc)
    if cls == "weld":
        Ft = min(s_t, 1.0e12) * area * a * row["eff"]
        Fs = s_tau * area * a * row["eff"]
        return Capacity(Ft, Fs, Ft * math.sqrt(area) / 6.0, s_c * area)
    if cls == "clamp":
        return Capacity(0.0, mu * normal, 0.0, normal)
    raise ValueError(cls)


def utilisation(cap, Fn=0.0, V=0.0, M=0.0):
    """Combined loading check (u form): (F_n+/Ft)^2 + (V/Fs)^2 + M/Mc. The
    joint holds while this is <= 1. A compression pushes on the substrate, not
    the joint, so only tension counts."""
    def term(x, c, p):
        if x <= 0:
            return 0.0
        return math.inf if c <= 0 else (x / c) ** p
    return term(Fn, cap.Ft, 2) + term(abs(V), cap.Fs, 2) + term(abs(M), cap.Mc, 1)


# ── cure laws ─────────────────────────────────────────────────────────────────
def aci_raw(t_days, a=4.0, beta=0.85):
    """ACI 209R-92 degree of strength t/(a + beta t), t in days. 0.7035 at 7 d,
    1.0072 at 28 d (c); it tends to 1/beta = 1.18."""
    return t_days / (a + beta * t_days)


def aci_alpha(t_days, a=4.0, beta=0.85):
    """The ACI curve divided by its own 28-day value and clamped to 1."""
    return min(max(aci_raw(t_days, a, beta) / aci_raw(28.0, a, beta), 0.0), 1.0)


def maturity_rate(T, Ea_R=4500.0, T_ref=T_REF):
    """Equivalent-age rate exp(-(Ea/R)(1/T - 1/T_ref)), ASTM C1074, Ea/R 4000-5000 K (u)."""
    return math.exp(-Ea_R * (1.0 / T - 1.0 / T_ref))


def dry_alpha(w, w0, w_eq):
    """Cure by drying: (w0 - w)/(w0 - w_eq), clamped."""
    if w0 <= w_eq:
        return 1.0
    return min(max((w0 - w) / (w0 - w_eq), 0.0), 1.0)


def set_on_cool(T, T_hi, T_lo):
    """Hide glue gels as it cools below T_hi and is fully set at T_lo (25-30 C,
    (u)). It is reversible: heat it and it lets go."""
    return min(max((T_hi - T) / (T_hi - T_lo), 0.0), 1.0)


def moisture_band(rh, lo, hi, taper):
    """Trapezoid humidity factor: 1 inside [lo, hi], falling to 0 over `taper`
    on each side. Lime carbonation: RH 40-80 percent."""
    if lo <= rh <= hi:
        return 1.0
    if rh < lo:
        return min(max((rh - (lo - taper)) / taper, 0.0), 1.0)
    return min(max(((hi + taper) - rh) / taper, 0.0), 1.0)


def moisture_ramp(rh, lo, hi):
    """Hydraulic cement needs moisture: zero below `lo`, 1 above `hi`."""
    return min(max((rh - lo) / (hi - lo), 0.0), 1.0)


def weld_rate(T, oxide, row):
    """Weld rate 1/tau, per second. tau is Arrhenius. The rate is ZERO (the
    weld never cures) below the diffusion-bond temperature, at or above the
    solidus of the composition, and where the oxide layer is thicker than its
    data threshold. Nothing is gated; the window is the data."""
    c = row["cure"] if "cure" in row else row
    if T < c["T_bond"] or T >= c["T_sol"] or oxide > c["oxide_max"]:
        return 0.0
    tau = c["tau_ref"] * math.exp(c["Q_R"] * (1.0 / T - 1.0 / c["T_tref"]))
    return 1.0 / tau


def new_joint(kind, area, mass_g=None, rid=0, a=None, b=None, alpha=None,
              rows=None):
    """A JOIN record as a plain dict. `mass_g` is the joining material (class
    b matter, counted by records_owned): {species key: grams}."""
    row = (rows or JOIN_KINDS)[kind]
    law = row["cure"]["law"]
    return {"rid": rid, "kind": kind, "a": a, "b": b, "area": float(area),
            "alpha": (1.0 if law == "none" else 0.0) if alpha is None else alpha,
            "te": 0.0, "moisture": 1.0, "x": 0.0, "mass_g": dict(mass_g or {}),
            "failed": None}


def cure_step(joint, dt, T=T_REF, rh=1.0, co2=1.0, w=None, w0=None, w_eq=None,
              oxide=0.0, rows=None):
    """Advance the cure of a joint by dt seconds (in place; returns alpha).
    T in K, rh in 0..1, co2 relative to the open-air value."""
    row = (rows or JOIN_KINDS)[joint["kind"]]["cure"]
    law = row["law"]
    if law == "none":
        joint["alpha"] = 1.0
    elif law == "hydration":
        f = moisture_ramp(rh, row["rh_lo"], row["rh_hi"])
        joint["te"] += dt * maturity_rate(T, row["Ea_R"]) * f
        joint["alpha"] = aci_alpha(joint["te"] / 86400.0)
    elif law == "maturity_exp":
        joint["te"] += dt * maturity_rate(T, row["Ea_R"])
        joint["alpha"] = 1.0 - math.exp(-joint["te"] / row["tau_ref"])
    elif law == "carbonation":
        f = moisture_band(rh, row["lo"], row["hi"], row["taper"]) * co2
        joint["te"] += dt * f
        half = 0.5 * math.sqrt(max(joint["area"], 1e-12)) * 0.05   # bed half-thickness (u)
        joint["alpha"] = min(row["K"] * math.sqrt(joint["te"]) / half, 1.0)
    elif law == "dry":
        joint["alpha"] = dry_alpha(w, w0, w_eq)
    elif law == "set_on_cool":
        joint["alpha"] = set_on_cool(T, row["T_hi"], row["T_lo"])
    elif law == "weld":
        r = weld_rate(T, oxide, row)
        joint["alpha"] = 1.0 - (1.0 - joint["alpha"]) * math.exp(-r * dt)
    else:
        raise ValueError(law)
    return joint["alpha"]


# ── sharing load between parallel joints (stand-in, (u)) ──────────────────────
def _force(x, k, cap, duc, before=False):
    """Total force of joints in parallel at the common displacement x. A joint
    carries k x until x reaches cap/k; then a brittle one drops to zero and a
    ductile one holds its capacity. `before` takes the value just before a
    joint at exactly x lets go."""
    tot = 0.0
    for i in range(len(k)):
        if k[i] <= 0:
            continue
        xi = cap[i] / k[i]
        live = (x <= xi) if before else (x < xi)
        tot += k[i] * x if live else (cap[i] if duc[i] else 0.0)
    return tot


def parallel_peak(k, cap, ductile=None):
    """Joints in parallel share a common displacement x, each carrying k_i x up
    to its capacity. A brittle joint then drops to zero, a ductile one holds
    its capacity. The force is piecewise linear in x, so its peak is found at
    the breakpoints x_i = cap_i/k_i. Returns (peak load, x at peak, order of
    failure). The peak never exceeds the sum of capacities."""
    n = len(k)
    duc = [False] * n if ductile is None else list(ductile)
    xs = sorted(set(cap[i] / k[i] for i in range(n) if k[i] > 0))
    best, bx = 0.0, 0.0
    for x in xs:
        f = _force(x, k, cap, duc, before=True)
        if f > best:
            best, bx = f, x
    order = sorted(range(n), key=lambda i: (cap[i] / k[i] if k[i] > 0 else math.inf, i))
    return best, bx, order


def load_shares(P, k, cap, ductile=None):
    """Load carried by each joint at the total P, on the first rising branch of
    the common-displacement curve. None if P is beyond the peak (the group
    lets go). The shares add to P and none exceeds its capacity."""
    n = len(k)
    duc = [False] * n if ductile is None else list(ductile)
    peak, _, _ = parallel_peak(k, cap, duc)
    if P > peak * (1 + 1e-12):
        return None
    xs = sorted(set(cap[i] / k[i] for i in range(n) if k[i] > 0))
    a, fa = 0.0, 0.0
    x_hit = xs[-1]
    for b in xs:
        fb = _force(b, k, cap, duc, before=True)
        if fa >= P:
            x_hit = a
            break
        if fb >= P:
            x_hit = a + (P - fa) * (b - a) / (fb - fa)
            break
        a, fa = b, _force(b, k, cap, duc)
    out = [(min(k[i] * x_hit, cap[i]) if x_hit <= cap[i] / k[i] * (1 + 1e-12)
            else (cap[i] if duc[i] else 0.0))
           if k[i] > 0 else 0.0 for i in range(n)]
    s = sum(out)
    return [o * P / s for o in out] if s > 0 else out


# ── composites (GROUP) ────────────────────────────────────────────────────────
def components(members, joints):
    """Connected components of the members through joints that have not
    failed. Union-find with path halving; deterministic order (by first member
    index). A member with no live joint is its own component."""
    idx = {m: i for i, m in enumerate(members)}
    parent = list(range(len(members)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for j in joints:
        if j.get("failed") is not None:
            continue
        ra, rb = find(idx[j["a"]]), find(idx[j["b"]])
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    groups = {}
    for m in members:
        groups.setdefault(find(idx[m]), []).append(m)
    return [groups[r] for r in sorted(groups)]


def new_group(gid, name, members, joints):
    """A composite: its own identity, KEEPING its parts and its joints."""
    return {"gid": gid, "name": name, "members": list(members),
            "joints": list(joints), "status": "whole", "log": []}


def joint_utilisations(group, loads, caps):
    """Utilisation of each joint of the group. `loads` and `caps` map rid to
    (Fn, V, M) and Capacity."""
    out = {}
    for j in group["joints"]:
        if j.get("failed") is not None:
            continue
        r = j["rid"]
        out[r] = utilisation(caps[r], *loads.get(r, (0.0, 0.0, 0.0)))
    return out


def weakest_joint(group, loads, caps):
    """The joint with the largest utilisation, (rid, u), ties by rid; None if
    the group has no live joint."""
    u = joint_utilisations(group, loads, caps)
    if not u:
        return None
    rid = min(u, key=lambda r: (-u[r], r))
    return rid, u[rid]


def fail_joint(group, rid, tick=0, mode="overload"):
    """Mark a joint failed. The parts and the joint stay in the group; the
    group is 'apart' when the live graph falls in more than one component.
    Returns the components."""
    for j in group["joints"]:
        if j["rid"] == rid:
            j["failed"] = (tick, mode)
    comps = components(group["members"], group["joints"])
    group["status"] = "apart" if len(comps) > 1 else "whole"
    group["log"].append(("comes_apart" if len(comps) > 1 else "joint_failed", rid, tick))
    return comps


def weakest_link(load, caps):
    """Which element of a chain gives way first under `load`: the one with the
    largest load/capacity. caps maps a name to a capacity in N. Returns the
    name, or 'holds' when none reaches 1."""
    name = min(caps, key=lambda n: (caps[n] / load if load > 0 else math.inf, n))
    return name if load >= caps[name] else "holds"


# ── fusing ────────────────────────────────────────────────────────────────────
def fuse(parts, new_id, T_of_H=None, name=None):
    """Fuse parts into ONE new record. Each part is a dict with
    'grams': {(species, form): g}, 'energy': J above the 20 C reference,
    'mass': kg, optional 'count' (integer lot items) and 'cap' (J/K).
    Masses add per species and per form (math.fsum, so the sum is the exact
    sum rounded once), energies add, counts add (exact integers). T comes from
    the summed enthalpy: T_of_H(H J/kg) if given, else 20 C + E/C with the
    summed capacities. The parts end with status 'fused', into = new_id; the
    new record lists them as lineage. Returns (new, parts)."""
    keys = []
    for p in parts:
        for k in p["grams"]:
            if k not in keys:
                keys.append(k)
    grams = {k: math.fsum(p["grams"].get(k, 0.0) for p in parts) for k in keys}
    mass = math.fsum(p["mass"] for p in parts)
    energy = math.fsum(p["energy"] for p in parts)
    count = sum(int(p.get("count", 0)) for p in parts)
    cap = math.fsum(p.get("cap", 0.0) for p in parts)
    if T_of_H is not None and mass > 0:
        T = T_of_H(energy / mass)
    elif cap > 0:
        T = 20.0 + energy / cap
    else:
        T = 20.0
    new = {"id": new_id, "name": name, "grams": grams, "mass": mass,
           "energy": energy, "count": count, "cap": cap, "T": T,
           "lineage": [p.get("id") for p in parts], "status": "whole"}
    for p in parts:
        p["status"] = "fused"
        p["cause"] = "fused"
        p["into"] = new_id
    return new, parts


def owned(joint):
    """What the joint's own material adds to the ledger (records_owned):
    grams per species key, and zero ml and J (the glue line's heat rides on its
    host)."""
    return dict(joint["mass_g"])
