"""LOTS: a thousand seeds, a heap of shards, a sack of grain, as ONE record.

Stand-alone (wave 13, topic T2, spec sections 3.7, 7.1-7.3). Nothing here
imports World. Plain numpy and the standard library are the definition.

A lot has two modes. Up to CAP = 4096 items it keeps PER-ITEM arrays (mass,
density, size, bond strength), which is exact for every scene here (1,000
seeds, 200 coins). Above that it keeps a 16 x 16 histogram (size by density)
of integer counts and float64 masses: the counts stay exact, the joint
distribution inside one bin is approximate, and each transform says so.

What it keeps:
  * Counts are integers and are split by LARGEST REMAINDER, so any number of
    splits adds back to the whole exactly. Mass is per item (or per bin) and
    is shared by an exact subtraction (breaking.exact_split), so fsum of the
    parts equals the whole.
  * FLOAT / SINK: items lighter than the liquid float (brine from the CRC
    table: 10 percent 1.071, 20 percent 1.148, 26 percent 1.199 g/ml).
  * WINNOW: a wind u and a fall height H spread a lot by LANDING DISTANCE, a
    smooth spread. With linear drag (relaxation time tau = v_t/g) the landing
    distance is exactly u H / v_t. The hard rule "carried if v_t < w" holds
    only for a VERTICAL updraft.
  * THRESH: items whose bond is weaker than the blow stress detach. SIEVE:
    by size against a mesh.

Numbers marked (u) are unverified. Numbers marked (c) were computed from the
formula shown.
"""
import math

import numpy as np

from .breaking import exact_split, unit

CAP = 4096                  # per-item up to here, histogram above
NB = 16                     # histogram bins per axis
RHO_AIR = 1.2               # kg/m3 at 20 C
MU_AIR = 1.8e-5             # Pa s
G = 9.80665

# CRC Handbook, NaCl in water at 20 C: mass percent -> g/ml. 0 percent is
# water at 20 C. Linear between the table's points.
BRINE_PCT = (0.0, 10.0, 20.0, 26.0)
BRINE_RHO = (0.9982, 1.071, 1.148, 1.199)


def brine_density(pct):
    """g/ml of NaCl brine at 20 C by mass percent (CRC), linear in between."""
    return float(np.interp(pct, BRINE_PCT, BRINE_RHO))


# ── making and reading a lot ──────────────────────────────────────────────────
def _edges(v, lo=None, hi=None):
    lo = float(np.min(v)) if lo is None else lo
    hi = float(np.max(v)) if hi is None else hi
    if hi <= lo:
        hi = lo + max(abs(lo) * 1e-6, 1e-9)
    e = np.linspace(lo, hi, NB + 1)
    e[-1] = np.nextafter(hi, np.inf)        # the maximum falls inside the last bin
    return e


def make_lot(mass, density, size, strength=None, species="grain", T=20.0,
             moisture=0.1, seed=1, hist=None):
    """A lot from per-item arrays. `hist` forces a mode; by default it is the
    histogram above CAP items. mass kg, density kg/m3, size m (a diameter),
    strength Pa (the bond that holds an item to its husk or stalk)."""
    mass = np.asarray(mass, dtype=np.float64)
    n = len(mass)
    lot = {"mode": "items", "n": n, "mass": mass.copy(),
           "density": np.asarray(density, dtype=np.float64).copy(),
           "size": np.asarray(size, dtype=np.float64).copy(),
           "strength": (np.zeros(n) if strength is None
                        else np.asarray(strength, dtype=np.float64).copy()),
           "id": np.arange(n, dtype=np.int64), "species": species, "T": float(T),
           "moisture": float(moisture), "seed": int(seed)}
    if hist is None:
        hist = n > CAP
    return to_hist(lot) if hist else lot


def to_hist(lot, size_edges=None, dens_edges=None):
    """The 16 x 16 histogram of a per-item lot (counts exact, mass float64,
    mean strength per bin). A histogram lot comes back unchanged."""
    if lot["mode"] == "hist":
        return lot
    se = _edges(lot["size"]) if size_edges is None else np.asarray(size_edges)
    de = _edges(lot["density"]) if dens_edges is None else np.asarray(dens_edges)
    i = np.clip(np.searchsorted(se, lot["size"], side="right") - 1, 0, NB - 1)
    j = np.clip(np.searchsorted(de, lot["density"], side="right") - 1, 0, NB - 1)
    flat = i * NB + j
    counts = np.bincount(flat, minlength=NB * NB).astype(np.int64)
    mass = np.zeros(NB * NB)
    stren = np.zeros(NB * NB)
    for b in np.unique(flat):
        sel = flat == b
        mass[b] = math.fsum(lot["mass"][sel])
        stren[b] = math.fsum(lot["strength"][sel]) / counts[b]
    return {"mode": "hist", "counts": counts.reshape(NB, NB),
            "mass": mass.reshape(NB, NB), "strength": stren.reshape(NB, NB),
            "size_edges": se, "dens_edges": de, "species": lot["species"],
            "T": lot["T"], "moisture": lot["moisture"], "seed": lot["seed"],
            "n": int(counts.sum())}


def count(lot):
    return int(lot["n"])


def total_mass(lot):
    return math.fsum(np.asarray(lot["mass"]).ravel())


def owned(lot):
    """What the lot adds to the ledger (records_owned): grams per species and
    the integer count."""
    return {"grams": {lot["species"]: total_mass(lot) * 1000.0},
            "count": count(lot)}


def centres(lot):
    """(size, density) at the centre of each histogram bin."""
    s = 0.5 * (lot["size_edges"][:-1] + lot["size_edges"][1:])
    d = 0.5 * (lot["dens_edges"][:-1] + lot["dens_edges"][1:])
    return np.meshgrid(s, d, indexing="ij")


# ── integers that add up ──────────────────────────────────────────────────────
def largest_remainder(total, weights):
    """Split an integer `total` in proportion to `weights` so the parts are
    integers and add to total EXACTLY. Floors first, then one more to each of
    the largest fractional remainders (ties to the lowest index)."""
    w = np.asarray(weights, dtype=np.float64)
    s = w.sum()
    if s <= 0:
        out = np.zeros(len(w), dtype=np.int64)
        return out
    raw = total * w / s
    base = np.floor(raw).astype(np.int64)
    short = int(total - base.sum())
    if short > 0:
        order = np.lexsort((np.arange(len(w)), -(raw - base)))
        base[order[:short]] += 1
    return base


def _round_grid(raw, target):
    """Round a float grid of expected counts to integers whose sum is `target`
    (largest remainder over the flat grid)."""
    flat = np.asarray(raw, dtype=np.float64).ravel()
    base = np.floor(flat + 1e-9).astype(np.int64)
    short = int(target - base.sum())
    if short > 0:
        order = np.lexsort((np.arange(len(flat)), -(flat - base)))
        base[order[:short]] += 1
    elif short < 0:
        order = np.lexsort((np.arange(len(flat)), (flat - base)))
        for k in order:
            if short == 0:
                break
            if base[k] > 0:
                base[k] -= 1
                short += 1
    return base.reshape(np.shape(raw))


# ── taking a part out of a lot ────────────────────────────────────────────────
def take_items(lot, mask):
    """(taken, rest) of a per-item lot by a boolean mask. Integer exact, mass
    per item (exact)."""
    keys = ("mass", "density", "size", "strength", "id")
    a = dict(lot)
    b = dict(lot)
    for k in keys:
        a[k] = lot[k][mask]
        b[k] = lot[k][~mask]
    a["n"], b["n"] = int(mask.sum()), int((~mask).sum())
    return a, b


def take_hist(lot, taken):
    """(taken, rest) of a histogram lot from an integer grid `taken` of counts
    to move (0 <= taken <= counts). Mass is shared per bin by exact_split."""
    c = lot["counts"]
    assert np.all(taken >= 0) and np.all(taken <= c)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(c > 0, taken / np.where(c > 0, c, 1), 0.0)
    ma, mb = exact_split(lot["mass"], frac)
    a = dict(lot)
    b = dict(lot)
    a["counts"], b["counts"] = taken.astype(np.int64), (c - taken).astype(np.int64)
    a["mass"], b["mass"] = ma, mb
    a["n"], b["n"] = int(a["counts"].sum()), int(b["counts"].sum())
    return a, b


def split_count(lot, k):
    """Take exactly k items out of a lot. Per-item: the k items with the
    smallest hash of (seed, id) (deterministic). Histogram: k shared over the
    bins by largest remainder. Returns (taken, rest)."""
    k = int(k)
    assert 0 <= k <= lot["n"]
    if lot["mode"] == "items":
        order = np.argsort([unit(lot["seed"], int(i), 3) for i in lot["id"]],
                           kind="stable")
        mask = np.zeros(lot["n"], dtype=bool)
        mask[order[:k]] = True
        return take_items(lot, mask)
    t = largest_remainder(k, lot["counts"].ravel()).reshape(NB, NB)
    return take_hist(lot, np.minimum(t, lot["counts"]))


def merge(a, b):
    """Add two lots of one mode and one histogram grid (or two per-item lots).
    Both must hold the same species: mixing two species is a different record
    (a mix), not a bigger lot, so it raises. Temperature and moisture of the
    result are mass-weighted means, which keeps thermal energy for equal heat
    capacity and keeps the water mass."""
    assert a["mode"] == b["mode"]
    if a["species"] != b["species"]:
        raise ValueError("merge: lots of different species (%r, %r) are a mix, not one lot"
                         % (a["species"], b["species"]))
    out = dict(a)
    ma, mb = total_mass(a), total_mass(b)
    mt = ma + mb
    wa = ma / mt if mt > 0 else 0.5
    out["T"] = float(wa * a["T"] + (1.0 - wa) * b["T"])
    out["moisture"] = float(wa * a["moisture"] + (1.0 - wa) * b["moisture"])
    if a["mode"] == "items":
        for k in ("mass", "density", "size", "strength"):
            out[k] = np.concatenate([a[k], b[k]])
        out["id"] = np.arange(len(out["mass"]), dtype=np.int64)
        out["n"] = a["n"] + b["n"]
    else:
        assert np.array_equal(a["size_edges"], b["size_edges"])
        assert np.array_equal(a["dens_edges"], b["dens_edges"])
        out["counts"] = a["counts"] + b["counts"]
        out["mass"] = a["mass"] + b["mass"]
        tot = out["counts"]
        with np.errstate(divide="ignore", invalid="ignore"):
            out["strength"] = np.where(
                tot > 0, (a["strength"] * a["counts"] + b["strength"] * b["counts"])
                / np.where(tot > 0, tot, 1), 0.0)
        out["n"] = int(tot.sum())
    return out


# ── float and sink ────────────────────────────────────────────────────────────
def float_split(lot, rho_liquid):
    """(floaters, sinkers) in a liquid of density rho_liquid (kg/m3). An item
    floats when it is lighter than the liquid. Per-item: exact. Histogram: a
    bin that straddles the liquid's density is shared linearly across the bin
    (the distribution inside one bin is approximate), counts rounded by
    largest remainder."""
    if lot["mode"] == "items":
        return take_items(lot, lot["density"] < rho_liquid)
    de = lot["dens_edges"]
    d0, d1 = de[:-1][None, :], de[1:][None, :]
    frac = np.clip((rho_liquid - d0) / (d1 - d0), 0.0, 1.0)
    raw = lot["counts"] * frac
    got = _round_grid(raw, int(round(float(raw.sum()))))
    got = np.minimum(got, lot["counts"])
    return take_hist(lot, got)


def float_share(lot, rho_liquid):
    """Fraction of the items that float."""
    f, _ = float_split(lot, rho_liquid)
    return f["n"] / lot["n"] if lot["n"] else 0.0


# ── winnow ────────────────────────────────────────────────────────────────────
def _cd(re):
    """Drag coefficient: Stokes below Re 1 (24/Re), Newton 0.44 above Re 1000,
    Schiller-Naumann between (24/Re)(1 + 0.15 Re^0.687), which meets both
    ends."""
    re = np.maximum(re, 1e-12)
    return np.where(re > 1000.0, 0.44, 24.0 / re * (1.0 + 0.15 * re ** 0.687))


def terminal_speed(rho_p, d, rho_f=RHO_AIR, mu=MU_AIR, g=G):
    """Terminal speed of a sphere of diameter d: solve
    (rho_p - rho_f) g pi d^3/6 = 0.5 C_d(Re) rho_f v^2 pi d^2/4 for v by
    bisection (monotone). Stokes v = 2 (rho_p - rho_f) g r^2/(9 mu) for Re < 1;
    Newton v = sqrt(4 g d (rho_p - rho_f)/(3 C_d rho_f)) at C_d 0.44. Wheat
    (d 3.5 mm, 1300 kg/m3): 10.6 m/s (c)."""
    rho_p = np.asarray(rho_p, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    shape = np.broadcast(rho_p, d).shape
    rho_p = np.broadcast_to(rho_p, shape)
    d = np.broadcast_to(d, shape)
    weight = (rho_p - rho_f) * g * np.pi * d ** 3 / 6.0
    lo = np.zeros(shape)
    hi = np.full(shape, 200.0)
    for _ in range(80):
        v = 0.5 * (lo + hi)
        re = rho_f * v * d / mu
        drag = 0.5 * _cd(re) * rho_f * v * v * np.pi * d * d / 4.0
        up = drag > weight
        hi = np.where(up, v, hi)
        lo = np.where(up, lo, v)
    return 0.5 * (lo + hi)


def landing_distance(v_t, u, H):
    """Where a particle dropped from height H into a horizontal wind u lands,
    with linear drag (relaxation time v_t/g): exactly u H / v_t. Lighter items
    (small v_t) land further."""
    return u * H / np.asarray(v_t, dtype=np.float64)


def landing_distance_integral(v_t, u, H, steps=4000):
    """The same distance by integrating the trajectory (RK4): dvz/dt =
    g - vz/tau, dvx/dt = (u - vx)/tau, tau = v_t/g; a test, not the law."""
    g = G
    tau = v_t / g

    def f(s):
        z, x, vz, vx = s
        return np.array([vz, vx, g - vz / tau, (u - vx) / tau])
    s = np.zeros(4)
    dt = (H / v_t + 5 * tau) / steps
    t = 0.0
    while s[0] < H and t < 1e3:
        k1 = f(s)
        k2 = f(s + 0.5 * dt * k1)
        k3 = f(s + 0.5 * dt * k2)
        k4 = f(s + dt * k3)
        new = s + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        if new[0] >= H:                     # interpolate the last step
            w = (H - s[0]) / (new[0] - s[0])
            return float(s[1] + w * (new[1] - s[1]))
        s = new
        t += dt
    return float(s[1])


def updraft_carried(v_t, w_up):
    """A VERTICAL updraft (a fan) carries away what is lighter than it: the
    hard rule v_t < w_up. A horizontal wind has no such threshold."""
    return np.asarray(v_t) < w_up


def winnow(lot, u, H, edges):
    """Spread a lot by landing distance into bins `edges` (m, increasing; items
    beyond the last edge go in the last bin). Returns a list of lots, one per
    bin. Per-item: each item lands where its own v_t puts it. Histogram: each
    (size, density) bin lands whole at its centre's distance (approximate)."""
    edges = np.asarray(edges, dtype=np.float64)
    nb = len(edges) - 1
    if lot["mode"] == "items":
        x = landing_distance(terminal_speed(lot["density"], lot["size"]), u, H)
        b = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, nb - 1)
        return [take_items(lot, b == k)[0] for k in range(nb)]
    s, d = centres(lot)
    x = landing_distance(terminal_speed(d, s), u, H)
    b = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, nb - 1)
    return [take_hist(lot, np.where(b == k, lot["counts"], 0))[0] for k in range(nb)]


# ── thresh and sieve ──────────────────────────────────────────────────────────
def thresh(lot, sigma_blow):
    """(detached, remaining): items whose bond strength is below the blow
    stress come off. A histogram lot uses each bin's mean strength
    (approximate)."""
    if lot["mode"] == "items":
        return take_items(lot, lot["strength"] < sigma_blow)
    return take_hist(lot, np.where(lot["strength"] < sigma_blow, lot["counts"], 0))


def sieve(lot, mesh):
    """(passed, retained) against a mesh of side `mesh` (m): items smaller than
    the mesh fall through. Histogram: a straddling bin is shared linearly."""
    if lot["mode"] == "items":
        return take_items(lot, lot["size"] < mesh)
    se = lot["size_edges"]
    frac = np.clip((mesh - se[:-1]) / (se[1:] - se[:-1]), 0.0, 1.0)[:, None]
    raw = lot["counts"] * frac
    got = np.minimum(_round_grid(raw, int(round(float(raw.sum())))), lot["counts"])
    return take_hist(lot, got)
