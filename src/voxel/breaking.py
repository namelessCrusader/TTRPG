"""BREAKING: how a thin thing parts, by stress AND strain, at its weakest point.

Stand-alone (wave 13, topic T2, spec sections 6.1-6.3 and 9.4 swing load).
Nothing here imports World. Plain numpy is the definition. Every function
takes numbers or arrays and returns numbers or arrays, so each one can be
tested against its closed form.

What it keeps (the facts a scene is about):
  * A bend is a bend of a SECTION whose strength comes from its material row
    (E, yield, strain to fracture, K_Ic, a Weibull exponent), never from the
    word "blade" or "brittle". Both tests run on every segment:
        FLAW  (brittle)   surface stress >= K_Ic / (Y sqrt(pi a)) x (1 - dmg)
        TEAR  (ductile)   surface strain >= eps_f x (1 - dmg)
    A hard blade snaps at a strain of 0.7 percent; mild steel under the SAME
    bend only yields and takes a set; push it to eps_f and it tears. There is
    no "does not snap" branch anywhere.
  * Two load kinds: FORCE controlled (a moment diagram of a hanging weight or
    a blow) and DISPLACEMENT controlled (a curvature imposed by a post).
  * The weakest segment (largest utilisation) breaks first. Weibull scatter
    scales the flaw stress with the volume of the segment, so a longer piece
    is weaker per unit stress (the size effect) -- from a counter-based hash
    of the segment id, never from random state, so a seed replays exactly.
  * A cut splits one row at a fraction f. The second share is WHOLE MINUS
    FIRST for every extensive quantity (mass, length, energy, grams per
    species), and the first share is rounded to a coarse grid of the whole's
    own exponent, so the subtraction is exact in float64 (a + b == whole, bit
    for bit) and any sum of pieces taken with math.fsum equals the whole.
  * The cut costs G_c A_x of fracture energy, paid from the released elastic
    energy and the work of the blow; it goes into the two end nodes as heat
    (cause "fracture"), the rest is kinetic (momentum conserved, split by
    inverse mass) or more heat.
  * Sheets and vessel walls: thermal stress from the record's own node
    profile, hoop pressure, impact stress from the two impedances; cracks
    that do not part the wall LEAK by min(orifice, slit); a complete break is
    a LOOP of cuts that costs G_c t l / eta_f each, paid until the budget
    cannot pay, and the pieces are one LOT of shards.
  * A swing's load on a joint is the stopping-distance load F = m v^2/(2 s),
    not a wave transit time.

Numbers marked (u) are unverified. Numbers marked (c) were computed from the
formula shown.
"""
import heapq
import math
from collections import namedtuple

import numpy as np

from . import record_data as _rd

try:                                    # T9's material record (phase B import)
    from . import matrec as _matrec     # pragma: no cover
except ImportError:
    _matrec = None

G = 9.80665                 # m/s2
Y_FLAW = 1.12               # edge crack geometry factor (Anderson)
A0 = 50e-6                  # m, default initial flaw depth (u)
A_MAX_FRAC = 0.25           # worst flaw reaches a quarter of the thickness (u)
ETA_F = 0.01                # share of the energy that becomes new surface (u;
#                             comminution gives about 1 percent or less)
V_REF = 1e-6                # m3, Weibull reference volume, 1 cm3 (u)
CD_ORIFICE = 0.6            # sharp-edged orifice discharge coefficient
SPLIT_LO, SPLIT_HI = 0.25, 0.75     # where a shard cut falls (u)

# ── the row: what a material says about breaking ───────────────────────────────
# E Pa, nu, sigma_y Pa (None = breaks before it yields), sigma_u Pa, eps_f
# (strain to fracture, = T9 `ductility`), K_Ic Pa sqrt(m) (None = no flaw arm),
# m (Weibull exponent, None = no scatter), a0 m (the flaw this material
# carries), rho kg/m3.
Strength = namedtuple("Strength",
                      "E nu sigma_y sigma_u eps_f K_Ic m a0 rho")

# THE STRENGTH ROWS LIVE IN record_data.STRENGTH (one table; T9's matrec supplies the real
# ones at wiring). This module only reads them, so a number is never typed twice. The
# names differ in one place: breaking's "steel_mild" is record_data's "steel".
# The WOOD rows equal src/bonds/materials.py WOOD_L and WOOD_T (a test compares them).
_ROW_OF = {"steel_mild": "steel"}


def _strength(name, row):
    return Strength(row["E"], row["nu"], row["sigma_y"], row["sigma_u"], row["eps_f"],
                    row["K_Ic"], row["weibull_m"], row.get("a0", A0), row["rho"])


STRENGTH = {n: _strength(n, _rd.STRENGTH[_ROW_OF.get(n, n)])
            for n in ("wood", "wood_perp", "ash", "glass", "earthenware", "steel_hard",
                      "steel_mild", "rope")}

# shape factor Zp/Z (plastic modulus over elastic): rectangle 1.5, round 1.698
_SHAPES = ("rect", "round", "tube")


def strength_from_matrec(rec, a0=A0):
    """A Strength out of a T9 MatRec-shaped object (attributes sy, st, ductility,
    KIc, weibull_m, E, nu, rho). Missing optional attributes fall back."""
    st = getattr(rec, "st", None)
    sy = getattr(rec, "sy", None)
    return Strength(float(rec.E), float(getattr(rec, "nu", 0.3)),
                    None if sy is None else float(sy),
                    float(st if st is not None else sy),
                    float(getattr(rec, "ductility", getattr(rec, "eps_f", 0.0))),
                    getattr(rec, "KIc", None), getattr(rec, "weibull_m", None),
                    a0, float(getattr(rec, "rho", 1000.0)))


# The damage combine is OWNED by matdamage.combine (T9). This is a copy so the
# module stands alone; the cross test (lane X) compares them, and at wiring
# the copy is replaced by an import in the same commit.
def combine(D, dD):
    """Independent causes: each keeps its own survival fraction, so the order
    does not matter. Clipped to 0.999 (a cell at 0.999 is made a piece)."""
    return np.minimum(1.0 - (1.0 - np.asarray(D, dtype=float))
                      * (1.0 - np.asarray(dD, dtype=float)), 0.999)


# ── deterministic scatter (no random state) ────────────────────────────────────
_M64 = (1 << 64) - 1


def _mix64(z):
    z = (z + 0x9E3779B97F4A7C15) & _M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _M64
    return z ^ (z >> 31)


def unit(seed, i, stream=0):
    """A number in (0, 1) fixed by (seed, i, stream): splitmix64 of a counter."""
    h = _mix64((int(seed) & _M64) ^ _mix64((int(i) & _M64)
                                           ^ _mix64(int(stream) & _M64)))
    return ((h >> 11) + 0.5) / 9007199254740992.0


def units(seed, ids, stream=0):
    return np.array([unit(seed, int(i), stream) for i in np.asarray(ids).ravel()],
                    dtype=np.float64).reshape(np.shape(ids))


def weibull_factor(u, m, V, v_ref=V_REF):
    """Strength factor of a piece of volume V from a survival draw u in (0, 1):
    s/s0 = (V_ref/V)^(1/m) x (-ln u)^(1/m) / Gamma(1 + 1/m). Mean 1 at V_ref.
    m None or 0 means no scatter (factor 1). A larger V gives a smaller factor
    for the SAME draw, and the weakest of many pieces is weaker still."""
    if not m:
        return np.ones(np.shape(u), dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    V = np.maximum(np.asarray(V, dtype=np.float64), 1e-30)
    return (np.power(-np.log(u), 1.0 / m) / math.gamma(1.0 + 1.0 / m)
            * np.power(v_ref / V, 1.0 / m))


# ── exact shares ───────────────────────────────────────────────────────────────
def exact_split(x, f):
    """(a, b) with a + b == x EXACTLY in float64, a ~ f x. a is rounded to a
    multiple of one ulp of x, 2^(e-53) where x lies in [2^(e-1), 2^e), so x and
    a share one grid and x - a needs no rounding. Works on arrays. The error of
    a against f x is at most half an ulp of x (about 1e-16 of x): f = 1 gives
    (x, 0) and f = 0 gives (0, x) exactly; a share below an ulp of x rounds to
    zero, which is the finest split a float64 whole can hold."""
    x = np.asarray(x, dtype=np.float64)
    mant, e = np.frexp(x)
    q = np.ldexp(1.0, e - 53)
    a = np.round(x * f / q) * q
    a = np.where(x == 0, 0.0, a)
    b = x - a
    return (a if a.ndim else float(a)), (b if b.ndim else float(b))


EXTENSIVE = ("len", "mass", "energy", "grams")      # split by share, add on fuse


def split_row(row, f, extensive=EXTENSIVE):
    """Split a dict of per-segment values: extensive keys by exact_split, every
    other key copied to both. Returns (a, b)."""
    a, b = {}, {}
    for k, v in row.items():
        if k in extensive:
            a[k], b[k] = exact_split(v, f)
        else:
            a[k] = np.copy(v) if isinstance(v, np.ndarray) else v
            b[k] = np.copy(v) if isinstance(v, np.ndarray) else v
    return a, b


# ── sections: remaining geometry ───────────────────────────────────────────────
def eff_dims(shape, w, t, loss=0.0):
    """Real remaining dimensions after `loss` m is gone from every face (char,
    rust, rot, wear). rect: w = width b, t = depth h. round: w = diameter.
    tube: w = outer D, t = inner d (the bore grows by the loss too)."""
    w = np.asarray(w, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    loss = np.asarray(loss, dtype=np.float64)
    if shape == "rect":
        return np.maximum(w - 2 * loss, 0.0), np.maximum(t - 2 * loss, 0.0)
    if shape == "round":
        d = np.maximum(w - 2 * loss, 0.0)
        return d, d
    if shape == "tube":
        D = np.maximum(w - 2 * loss, 0.0)
        return D, np.minimum(t + 2 * loss, D)
    raise ValueError(shape)


def section_area(shape, w, t, loss=0.0):
    a, b = eff_dims(shape, w, t, loss)
    if shape == "rect":
        return a * b
    if shape == "round":
        return np.pi * a * a / 4
    return np.pi * (a * a - b * b) / 4


def section_modulus(shape, w, t, loss=0.0):
    """Elastic section modulus Z of the REMAINING section: rectangle b h^2/6,
    round pi d^3/32, tube pi (D^4 - d^4)/(32 D)."""
    a, b = eff_dims(shape, w, t, loss)
    if shape == "rect":
        return a * b * b / 6.0
    if shape == "round":
        return np.pi * a ** 3 / 32.0
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.pi * (a ** 4 - b ** 4) / (32.0 * a)
    return np.where(a > 0, z, 0.0)


def y_max(shape, w, t, loss=0.0):
    """Distance from the neutral axis to the outer fibre."""
    a, b = eff_dims(shape, w, t, loss)
    return b / 2.0 if shape == "rect" else a / 2.0


def shape_factor(shape, w, t):
    """Zp/Z: 1.5 rectangle, 16/(3 pi) = 1.698 round, from the plastic modulus
    (D^3 - d^3)/6 for a tube."""
    if shape == "rect":
        return 1.5
    if shape == "round":
        return 16.0 / (3.0 * math.pi)
    D, d = float(np.max(w)), float(np.max(t))
    zp = (D ** 3 - d ** 3) / 6.0
    return zp / float(section_modulus("tube", D, d))


def segment_volume(piece):
    return section_area(piece["shape"], piece["w"], piece["t"], piece["loss"]) * piece["len"]


# ── fracture numbers ───────────────────────────────────────────────────────────
def gc_from_kic(K_Ic, E, nu):
    """Fracture energy G_c = K_Ic^2 (1 - nu^2) / E, plane strain, J/m2."""
    return K_Ic * K_Ic * (1.0 - nu * nu) / E


def flaw_stress(K_Ic, a, dmg=0.0, factor=1.0, Y=Y_FLAW):
    """Surface stress at which a flaw of depth a runs: K_Ic / (Y sqrt(pi a)),
    scaled by (1 - dmg) and by the Weibull factor."""
    if K_Ic is None:
        return np.full(np.shape(a), np.inf)
    return (K_Ic / (Y * np.sqrt(np.pi * np.asarray(a, dtype=np.float64)))
            * (1.0 - np.asarray(dmg)) * factor)


def hardening(row):
    """Plastic slope H of the stress-strain line from yield to fracture, Pa.
    0 for a material that breaks before it yields."""
    if row.sigma_y is None:
        return 0.0
    ey = row.sigma_y / row.E
    if row.eps_f <= ey:
        return 0.0
    return float(min(max((row.sigma_u - row.sigma_y) / (row.eps_f - ey), 0.0),
                     0.5 * row.E))


def surface_stress(row, eps):
    """sigma = min(E eps, sigma_y + H (eps - eps_y)): elastic, then plastic."""
    eps = np.asarray(eps, dtype=np.float64)
    el = row.E * eps
    if row.sigma_y is None:
        return el
    ey = row.sigma_y / row.E
    return np.minimum(el, row.sigma_y + hardening(row) * (eps - ey))


def strain_from_moment(row, M, Z, shape_f=1.5):
    """Surface strain of a section under a bending moment M (force control).
    Elastic to M_y = sigma_y Z; then M/M_y = f_s - (f_s - 1)(eps_y/eps)^2
    (exact for a rectangle, an approximation for other shapes); at M >= f_s
    M_y the section is a plastic hinge and the strain is unbounded (inf)."""
    M = np.abs(np.asarray(M, dtype=np.float64))
    Z = np.asarray(Z, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        el = np.where(Z > 0, M / (Z * row.E), np.inf)
        if row.sigma_y is None:
            return el
        ey = row.sigma_y / row.E
        My = row.sigma_y * Z
        r = np.where(My > 0, M / My, np.inf)
        inner = (shape_f - r) / (shape_f - 1.0)
        pl = np.where(inner > 0, ey / np.sqrt(np.maximum(inner, 1e-300)), np.inf)
    return np.where(r <= 1.0, el, pl)


# ── pieces (a SECTION as a run of segments) ────────────────────────────────────
def make_section(shape, length, w, t, row, n=20, seed=1, species=("solid",),
                 T=20.0, x0=0.0):
    """A straight bar, blade or beam of n equal segments as a plain dict of
    float64 arrays. `sid` is the persistent id of each segment (its flaw draw
    follows the matter through cuts). `energy` is the thermal J above 20 C,
    `grams` the grams per species (column per name in `species`, all of it in
    the first column unless set by the caller)."""
    assert shape in _SHAPES
    seg = float(length) / n
    A = float(section_area(shape, w, t))
    mass = np.full(n, row.rho * A * seg)
    grams = np.zeros((n, len(species)))
    grams[:, 0] = mass * 1000.0
    return {"shape": shape, "seed": int(seed), "species": tuple(species),
            "len": np.full(n, seg), "w": np.full(n, float(w)),
            "t": np.full(n, float(t)), "loss": np.zeros(n),
            "dmg": np.zeros(n), "notch": np.zeros(n), "mass": mass,
            "energy": np.zeros(n), "grams": grams,
            "x": x0 + seg * (np.arange(n) + 0.5), "sid": np.arange(n, dtype=np.int64),
            "cuts": 0}


def total(piece, key):
    return math.fsum(np.asarray(piece[key]).ravel())


def cut_piece(piece, i, f=0.5):
    """Cut segment i at fraction f of its length. Returns (A, B): A holds
    segments 0..i-1 and the first f of i, B the rest. Extensive quantities are
    exactly shared; both halves of the cut segment keep their segment id (the
    same flaw draw, but a smaller volume, so each is the stronger)."""
    n = len(piece["len"])
    out = [{k: v for k, v in piece.items() if not isinstance(v, np.ndarray)}
           for _ in (0, 1)]
    ka, kb = {}, {}
    for k, v in piece.items():
        if not isinstance(v, np.ndarray):
            continue
        if k in EXTENSIVE:
            a_i, b_i = exact_split(v[i], f)
            ka[k] = np.concatenate([v[:i], np.atleast_1d(a_i).reshape((1,) + v.shape[1:])])
            kb[k] = np.concatenate([np.atleast_1d(b_i).reshape((1,) + v.shape[1:]), v[i + 1:]])
        else:
            ka[k] = np.concatenate([v[:i], v[i:i + 1]])
            kb[k] = np.concatenate([v[i:i + 1], v[i + 1:]])
    # positions: A ends at the cut, B starts at it
    seg_lo = piece["x"][i] - piece["len"][i] / 2.0
    cutx = seg_lo + f * piece["len"][i]
    ka["x"] = np.concatenate([piece["x"][:i], [seg_lo + 0.5 * f * piece["len"][i]]])
    kb["x"] = np.concatenate([[cutx + 0.5 * (1 - f) * piece["len"][i]], piece["x"][i + 1:]])
    out[0].update(ka)
    out[1].update(kb)
    for d in out:
        d["cuts"] = piece["cuts"] + 1
    assert len(out[0]["len"]) == i + 1 and len(out[1]["len"]) == n - i
    return out[0], out[1]


# ── limits and utilisation ─────────────────────────────────────────────────────
def segment_limits(row, piece, scatter=True):
    """Per-segment failure limits: the flaw stress (with the segment's own
    crack depth, damage and Weibull factor) and the tear strain."""
    sh = piece["shape"]
    a_dim, b_dim = eff_dims(sh, piece["w"], piece["t"], piece["loss"])
    thick = b_dim if sh == "rect" else a_dim
    a_i = row.a0 + piece["dmg"] * A_MAX_FRAC * thick + piece["notch"]
    if scatter and row.m:
        f = weibull_factor(units(piece["seed"], piece["sid"]), row.m,
                           segment_volume(piece))
    else:
        f = np.ones(len(piece["len"]))
    sig = flaw_stress(row.K_Ic, a_i, piece["dmg"], f)
    eps = (row.eps_f * (1.0 - piece["dmg"])) if row.eps_f > 0 else np.zeros(len(a_i))
    return {"sigma_flaw": sig, "eps_tear": eps, "a": a_i, "factor": f}


def utilisation(row, piece, eps, scatter=True):
    """Both tests on the surface strain eps of each segment. Returns a dict:
    sigma, u_flaw, u_tear, u (max), mode ('flaw', 'tear' or 'none' per segment),
    yielded (bool, past yield and not broken), eps."""
    lim = segment_limits(row, piece, scatter)
    eps = np.asarray(eps, dtype=np.float64)
    sig = surface_stress(row, eps)
    with np.errstate(divide="ignore", invalid="ignore"):
        u_fl = np.where(sig > 0, sig / lim["sigma_flaw"], 0.0)
        # brittle rows (eps_f 0) have no tear arm: the flaw arm is their limit
        u_te = np.where(lim["eps_tear"] > 0, eps / lim["eps_tear"], 0.0)
    u_fl = np.nan_to_num(u_fl, nan=np.inf)
    u_te = np.nan_to_num(u_te, nan=np.inf)
    u = np.maximum(u_fl, u_te)
    mode = np.where(u < 1.0, "none", np.where(u_fl >= u_te, "flaw", "tear"))
    yielded = np.zeros(len(u), dtype=bool)
    if row.sigma_y is not None:
        yielded = (sig >= row.sigma_y * (1 - 1e-12)) & (u < 1.0)
    return {"sigma": sig, "eps": eps, "u_flaw": u_fl, "u_tear": u_te, "u": u,
            "mode": mode, "yielded": yielded, "limits": lim}


def residual_curvature(row, eps, ymax):
    """Springback: what remains of the imposed curvature after release,
    (eps - sigma/E)/y_max, a readout (the permanent bend itself is wave 16)."""
    sig = surface_stress(row, eps)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ymax > 0, (np.asarray(eps) - sig / row.E) / ymax, 0.0)


def bend_displacement(row, piece, kappa, scatter=True):
    """DISPLACEMENT controlled: the curvature kappa (1/m, per segment or one
    number) is imposed. eps = kappa y_max."""
    ym = y_max(piece["shape"], piece["w"], piece["t"], piece["loss"])
    kappa = np.broadcast_to(np.asarray(kappa, dtype=np.float64), ym.shape)
    res = utilisation(row, piece, np.abs(kappa) * ym, scatter)
    res["residual_kappa"] = residual_curvature(row, res["eps"], ym)
    return res


def bend_force(row, piece, M, scatter=True):
    """FORCE controlled: the moment M (N m, per segment) acts."""
    sh = piece["shape"]
    Z = section_modulus(sh, piece["w"], piece["t"], piece["loss"])
    fs = shape_factor(sh, piece["w"], piece["t"])
    eps = strain_from_moment(row, M, Z, fs)
    res = utilisation(row, piece, eps, scatter)
    res["M"] = np.abs(M)
    res["Z"] = Z
    return res


def flaw_moment(row, piece, scatter=False):
    """The moment at which each segment's flaw runs: sigma_flaw x Z (brittle
    bend with no plasticity in the way). For the 40 x 2 mm hard blade this is
    1.42 GPa x 2.67e-8 m3 = 38 N m (c)."""
    Z = section_modulus(piece["shape"], piece["w"], piece["t"], piece["loss"])
    return segment_limits(row, piece, scatter)["sigma_flaw"] * Z


# moment diagrams for statically determinate cases -----------------------------
def moments(xc, loads, support="simple", xa=None, xb=None, masses=None):
    """Bending moment magnitude |M(x)| at positions xc.
      loads: list of (x, P), P downward in N.
      support 'simple': pinned at xa and xb;
              'cantilever': fixed at xa (the free end is at the other side);
              'free': no support: the loads are balanced by the beam's own
              inertia (masses per segment at xc, required) -- a blow, an impulse.
    """
    xc = np.asarray(xc, dtype=np.float64)
    M = np.zeros_like(xc)
    if support == "simple":
        Ptot = sum(P for _, P in loads)
        Rb = sum(P * (x - xa) for x, P in loads) / (xb - xa)
        Ra = Ptot - Rb
        M = Ra * np.maximum(xc - xa, 0) + Rb * np.maximum(xc - xb, 0)
        for x, P in loads:
            M = M - P * np.maximum(xc - x, 0)
        return np.abs(M)
    if support == "cantilever":
        for x, P in loads:
            M = M + P * np.maximum(x - xc, 0)
        return np.abs(M)
    if support == "free":
        m = np.asarray(masses, dtype=np.float64)
        Mt = m.sum()
        xcm = float((m * xc).sum() / Mt)
        Icm = float((m * (xc - xcm) ** 2).sum())
        F = sum(P for _, P in loads)
        torque = sum(P * (x - xcm) for x, P in loads)
        a = F / Mt
        alpha = torque / Icm if Icm > 0 else 0.0
        inertia = m * (a + alpha * (xc - xcm))          # downward force it takes
        for x, P in loads:
            M = M + P * np.maximum(xc - x, 0)
        # inertial pieces lie to the left of each section (own half ignored)
        mom = np.array([float((inertia[:i] * (xc[i] - xc[:i])).sum())
                        for i in range(len(xc))])
        return np.abs(M - mom)
    raise ValueError(support)


def elastic_energy(row, piece, eps):
    """Stored bending energy 0.5 E I kappa^2 len summed over the segments, with
    kappa = eps / y_max (elastic part only)."""
    sh = piece["shape"]
    ym = y_max(sh, piece["w"], piece["t"], piece["loss"])
    Z = section_modulus(sh, piece["w"], piece["t"], piece["loss"])
    I = Z * ym
    with np.errstate(divide="ignore", invalid="ignore"):
        kap = np.where(ym > 0, np.asarray(eps) / ym, 0.0)
    eps_el = np.minimum(np.asarray(eps), (row.sigma_y or np.inf) / row.E)
    kap = np.where(ym > 0, eps_el / np.where(ym > 0, ym, 1), 0.0)
    return float(np.sum(0.5 * row.E * I * kap ** 2 * piece["len"]))


# ── the cut: energy, shares, momentum ──────────────────────────────────────────
def cut_area(piece, i):
    return float(section_area(piece["shape"], piece["w"][i], piece["t"][i],
                              piece["loss"][i]))


def cut_energy(row, area, work, kin_frac=0.5):
    """Energy of one cut. surface = G_c A_x (pays for the new faces, goes into
    the two end nodes as heat). `work` is what the caller has: released
    elastic energy plus the work of the blow. If it cannot pay the surface the
    crack arrests (ok False, nothing spent). The rest is kinetic (kin_frac) or
    more end-node heat. surface + heat_extra + kinetic == work exactly."""
    Gc = gc_from_kic(row.K_Ic, row.E, row.nu) if row.K_Ic else 0.0
    surface = Gc * area
    if work < surface:
        return {"ok": False, "surface": surface, "heat_extra": 0.0,
                "kinetic": 0.0, "Gc": Gc, "area": area, "spent": 0.0}
    rest = work - surface
    heat_extra = (1.0 - kin_frac) * rest
    return {"ok": True, "surface": surface, "heat_extra": heat_extra,
            "kinetic": rest - heat_extra, "Gc": Gc, "area": area,
            "spent": work}


def impulses(m_a, m_b, kinetic, v0=0.0, sign=1.0):
    """Velocities of two pieces that part with `kinetic` J in the frame of the
    centre of mass: momentum is conserved (m_a v_a + m_b v_b = (m_a + m_b) v0)
    and the speeds go as the inverse masses. Returns (v_a, v_b)."""
    mu = m_a * m_b / (m_a + m_b)
    p = math.sqrt(max(2.0 * mu * kinetic, 0.0))
    return v0 - sign * p / m_a, v0 + sign * p / m_b


def apply_cut(row, piece, i, f, work, kin_frac=0.5, v0=0.0):
    """Cut piece at segment i, fraction f, paying from `work`. Returns
    (A, B, event) or None if the crack arrests. The thermal energy of the two
    pieces together rises by exactly surface + heat_extra, half in each end
    node (cause 'fracture')."""
    ev = cut_energy(row, cut_area(piece, i), work, kin_frac)
    if not ev["ok"]:
        return None
    A, B = cut_piece(piece, i, f)
    heat = ev["surface"] + ev["heat_extra"]
    h_a, h_b = exact_split(heat, 0.5)
    A["energy"][-1] += h_a
    B["energy"][0] += h_b
    ma, mb = total(A, "mass"), total(B, "mass")
    ev["v"] = impulses(ma, mb, ev["kinetic"], v0)
    ev["cause"] = "fracture"
    ev["heat"] = heat
    ev["masses"] = (ma, mb)
    return A, B, ev


def cut_loop(row, piece, solver, work=0.0, kin_frac=0.5, max_cuts=64,
             scatter=True, f=0.5):
    """The weakest-point loop. `solver(piece) -> utilisation dict` is the load
    case (see bend_displacement / bend_force closures). While the largest
    utilisation of any piece is >= 1 the weakest segment is cut, each free
    piece is re-solved and the loop repeats. `work` (J) is the blow's work and
    goes to the first cut; what a cut does not need as surface leaves as
    kinetic energy and heat of the pieces, so every later cut is paid by the
    elastic energy stored in its own piece.
    Returns {"pieces": [...], "events": [...], "arrested": bool, "work_left"}."""
    pieces = [piece]
    events = []
    arrested = False
    left = float(work)
    i_guard = 0
    while i_guard < max_cuts:
        i_guard += 1
        worst = None
        for pi, p in enumerate(pieces):
            if len(p["len"]) == 0:
                continue
            res = solver(p)
            j = int(np.argmax(res["u"]))
            if res["u"][j] >= 1.0 and (worst is None or res["u"][j] > worst[0]):
                worst = (float(res["u"][j]), pi, j, res)
        if worst is None:
            break
        _, pi, j, res = worst
        p = pieces[pi]
        eps = res["eps"]
        w_avail = left + elastic_energy(row, p, eps)
        out = apply_cut(row, p, j, f, w_avail, kin_frac)
        if out is None:
            arrested = True
            break
        A, B, ev = out
        ev["segment"] = j
        ev["mode"] = str(res["mode"][j])
        ev["sid"] = int(p["sid"][j])
        events.append(ev)
        left = max(w_avail - ev["spent"], 0.0)
        pieces[pi:pi + 1] = [A, B]
    return {"pieces": pieces, "events": events, "arrested": arrested,
            "work_left": left}


# load cases for cut_loop ------------------------------------------------------
def displacement_solver(row, kappa, scatter=True):
    """solver(piece) for cut_loop: an imposed curvature (number, array over the
    segments, or a function of the segment centres x). A piece that has been
    cut is free and unloaded."""
    def solve(p):
        n = len(p["len"])
        if p["cuts"] > 0:
            return bend_displacement(row, p, np.zeros(n), scatter)
        k = kappa(p["x"]) if callable(kappa) else kappa
        return bend_displacement(row, p, k, scatter)
    return solve


def force_solver(row, loads, support="simple", xa=None, xb=None, scatter=True):
    """solver(piece) for cut_loop: a moment diagram from point loads. After the
    first fracture the load case is over (the pieces are free)."""
    def solve(p):
        n = len(p["len"])
        if p["cuts"] > 0:
            return bend_force(row, p, np.zeros(n), scatter)
        M = moments(p["x"], loads, support, xa, xb, p["mass"])
        return bend_force(row, p, M, scatter)
    return solve


# ── impact, swing ──────────────────────────────────────────────────────────────
def impedance(rho, E):
    """Z = rho c with the bar wave speed c = sqrt(E/rho)."""
    return rho * math.sqrt(E / rho)


def impact_stress(v, z1, z2):
    """Contact stress of two bodies meeting at relative speed v:
    sigma = v Z1 Z2 / (Z1 + Z2) (not rho c dv, which is a rigid wall)."""
    return v * z1 * z2 / (z1 + z2)


def swing_load(m, v, s):
    """Peak load on a joint when a head of mass m at speed v stops in a
    distance s (the crush or penetration depth): F = m v^2 / (2 s). 1 kg at
    10 m/s with s = 1 cm is 5 kN (c). A stress-wave transit time would give 30
    times too much and is only the lower bound of the duration."""
    return m * v * v / (2.0 * max(s, 1e-9))


def swing_duration(v, s):
    """Time to stop: 2 s / v."""
    return 2.0 * s / max(v, 1e-12)


def wave_transit(L, E, rho):
    """2 L / c, the stress-wave round trip: only the lower bound of the
    duration of a load."""
    return 2.0 * L / math.sqrt(E / rho)


def drop_energy(m, h):
    """m g h: the energy a dropped thing carries into the floor. 0.5 kg from
    1.5 m: 7.4 J; 3 kg: 44 J (c)."""
    return m * G * h


# ── sheets and walls ───────────────────────────────────────────────────────────
def thermal_stress(E, alpha, nu, T_nodes, dz=None):
    """Stress at each node of a wall from its OWN temperature profile:
    sigma = E alpha (T_mean - T(x)) / (1 - nu) (cooled surface in tension). The
    mean is weighted by node thickness dz (equal if None)."""
    T = np.asarray(T_nodes, dtype=np.float64)
    wts = np.ones_like(T) if dz is None else np.asarray(dz, dtype=np.float64)
    Tm = float((T * wts).sum() / wts.sum())
    return E * alpha * (Tm - T) / (1.0 - nu)


def hoop_stress(p, r, t):
    """Thin-wall cylinder: p r / t."""
    return p * r / t


def wall_flaw_util(row, sigma, thickness, dmg=0.0, factor=1.0, notch=0.0):
    """Utilisation of a wall under the surface stress sigma (the flaw arm)."""
    a = row.a0 + dmg * A_MAX_FRAC * thickness + notch
    lim = flaw_stress(row.K_Ic, a, dmg, factor)
    return float(sigma / lim)


def crack_opening(eps, gauge):
    """Opening of a crack that does not part the wall: strain times the gauge
    length it relieved."""
    return max(float(eps), 0.0) * gauge


def orifice_leak(area, head, cd=CD_ORIFICE, g=G):
    """Sharp-edged orifice q = C_d A sqrt(2 g h), m3/s. Valid when the opening
    is large against the wall thickness."""
    return cd * area * math.sqrt(2.0 * g * max(head, 0.0))


def slit_leak(w, length, dp, mu, t_wall):
    """Poiseuille flow in a thin slit of opening w, length l through a wall of
    thickness t: q = w^3 l dp / (12 mu t), m3/s."""
    return w ** 3 * length * dp / (12.0 * mu * t_wall)


def leak_rate(w, length, t_wall, head, rho=1000.0, mu=1.0e-3, cd=CD_ORIFICE,
              g=G):
    """LEAK = min(orifice, slit), m3/s. A 20 um x 5 cm crack in a 6 mm wall
    under 10 cm of water: slit 0.0054 ml/s, the orifice law alone would give
    0.84 ml/s, about 150 times too much (c)."""
    q_or = orifice_leak(w * length, head, cd, g)
    q_sl = slit_leak(w, length, rho * g * max(head, 0.0), mu, t_wall)
    return min(q_or, q_sl)


# ── the fragmentation loop (a complete break) ──────────────────────────────────
def fragment(mass, area, thickness, row, budget, seed=1, eta_f=ETA_F,
             energy=0.0, grams=None, max_pieces=4096):
    """Successive random breakage as a LOOP. Available energy `budget` (blow
    work plus stored strain energy, J). Each step takes the piece with the
    highest utilisation (the largest, with its Weibull draw), cuts across it
    over l = sqrt(area), pays G_c t l / eta_f of the budget, splits at a
    fraction in [0.25, 0.75] drawn from the flaw field, and the budget
    shrinks. Stops when the biggest cut cannot be paid. The cumulative cost is
    the stop rule, never one floor.
    Returns a dict: area, mass, energy (J, same energy per kg), grams (rows),
    cost (list), spent, left, n. Masses are exact shares of `mass`: fsum of
    the shards equals `mass` bit for bit."""
    Gc = gc_from_kic(row.K_Ic, row.E, row.nu)
    m_w = row.m if row.m else 1e3

    def util(p):
        # bigger is weaker (volume), times the piece's own flaw draw
        u = unit(seed, p["id"], 1)
        return (p["area"] / area) ** (1.0 / m_w) * (-math.log(u)) ** (1.0 / m_w)

    first = {"area": float(area), "mass": float(mass), "energy": float(energy),
             "grams": None if grams is None else np.array(grams, dtype=float),
             "id": 0}
    heap = [(-util(first), 0, first)]       # the highest utilisation first, ties by id
    nid = 1
    costs = []
    left = float(budget)
    while heap and len(heap) < max_pieces:
        _, _, p = heap[0]
        cost = Gc * thickness * math.sqrt(p["area"]) / eta_f
        if cost > left:
            break                            # the weakest piece cannot be paid
        heapq.heappop(heap)
        frac = SPLIT_LO + (SPLIT_HI - SPLIT_LO) * unit(seed, p["id"], 2)
        a_ar, b_ar = exact_split(p["area"], frac)
        share = a_ar / p["area"]
        a_m, b_m = exact_split(p["mass"], share)
        a_e, b_e = exact_split(p["energy"], share)
        if p["grams"] is None:
            a_g = b_g = None
        else:
            a_g, b_g = exact_split(p["grams"], share)
        for ar, m_, e_, g_ in ((a_ar, a_m, a_e, a_g), (b_ar, b_m, b_e, b_g)):
            c = {"area": ar, "mass": m_, "energy": e_, "grams": g_, "id": nid}
            heapq.heappush(heap, (-util(c), nid, c))
            nid += 1
        left -= cost
        costs.append(cost)
    pieces = sorted([h[2] for h in heap], key=lambda q: q["id"])
    out = {"area": np.array([p["area"] for p in pieces]),
           "mass": np.array([p["mass"] for p in pieces]),
           "energy": np.array([p["energy"] for p in pieces]),
           "cost": costs, "spent": math.fsum(costs), "left": left,
           "n": len(pieces), "budget": float(budget)}
    if grams is not None:
        out["grams"] = np.array([p["grams"] for p in pieces])
    return out


# ── cutting a thing that lives in cells (records_cut, pure part) ───────────────
def records_cut(cell_mass, F, cap_mass=None):
    """Where a thing breaks when the cut falls INSIDE its cells. `cell_mass` are
    the masses of the thing's cells already ordered along the section axis
    (the caller orders them; ties by flat index). F is the cut as a fraction of
    the mass. The cut falls in the cell whose mass interval holds F; that
    straddling cell goes whole to the side with the larger overlap; the OTHER
    side is restamped so the two child masses sum to the parent exactly, with
    the residue in that side's largest cell, never above `cap_mass` (the fill
    limit of a cell) or below 0.
    Returns {"a": cells, "b": cells, "mass_a": per-cell masses, "mass_b": ...,
    "straddle": k, "separable": bool}. A one-cell thing (or a side with no
    cells) is not separable: it stays one thing with two sections and a failed
    internal joint."""
    m = np.asarray(cell_mass, dtype=np.float64)
    n = len(m)
    M = math.fsum(m)
    cum = np.cumsum(m)
    cut = F * M
    k = int(min(np.searchsorted(cum, cut, side="right"), n - 1))
    lo = 0.0 if k == 0 else float(cum[k - 1])
    over_a = cut - lo
    over_b = float(cum[k]) - cut
    if over_a >= over_b:
        a_idx, b_idx = list(range(0, k + 1)), list(range(k + 1, n))
    else:
        a_idx, b_idx = list(range(0, k)), list(range(k, n))
    sep = bool(a_idx) and bool(b_idx) and n > 1
    mass_a = m[a_idx].copy()
    mass_b = m[b_idx].copy()
    if sep:
        # the straddle went to one side: the other side takes up the slack
        if len(a_idx) and a_idx[-1] == k:            # A kept the straddle cell
            other, keep = mass_b, mass_a
            want = M - math.fsum(keep)
        else:                                        # B kept it
            other, keep = mass_a, mass_b
            want = M - math.fsum(keep)
        scale = want / math.fsum(other) if math.fsum(other) > 0 else 0.0
        new = other * scale
        j = int(np.argmax(new))
        new[j] += want - math.fsum(new)              # the residue to the largest
        if cap_mass is not None:
            new = np.minimum(new, cap_mass)
            new[j] += want - math.fsum(new)
        new = np.maximum(new, 0.0)
        if cap_mass is not None and np.any(new > cap_mass * (1 + 1e-12)):
            sep = False
        if other is mass_b:
            mass_b = new
        else:
            mass_a = new
    return {"a": a_idx, "b": b_idx, "mass_a": mass_a, "mass_b": mass_b,
            "straddle": k, "separable": sep, "parent_mass": M}
