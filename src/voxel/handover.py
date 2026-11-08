"""HAND-OVER between a record and the lattice (T2, spec 6.4, F4).

A thin thing becomes lattice matter only when it spills, breaks, tips or grows past a cell,
and goes back when it shrinks. Three things must hold at every crossing:

  1. NO THRASH. The decisive dimension d* goes to the lattice at d* >= 1.0 cell and back to
     a record at d* <= 0.5 cell; between the two nothing changes (`decide`).
  2. EXACT BOOKS. Record-owned matter is float64, lattice arrays are float32. Across a
     crossing the difference between the float64 value and what the float32 array can hold
     goes into a REMAINDER (`rem`, per field per host) that belongs to the `records_owned`
     ledger term, and is given back on the next read. So the float64 ledger sum does not
     move by a crossing (1e-12 relative, in fact one float64 add), and the float32 lattice
     sum differs by at most one float32 ulp per touched cell.
  3. A SENSIBLE FOOTPRINT. A 40 x 4 x 0.2 cm blade is 32 cm3 and a 5 cm cell 125 cm3, so across
     eight cells the fill is about 0.03 (not 0.3). `restamp` writes mass against that volume
     with the rounding residue in the largest cell and no cell above its capacity.

Everything takes arrays as arguments; no World."""
import math

import numpy as np

from .record_data import NS

HYST_DOWN = 0.5            # cells: back to a record at or below this
HYST_UP = 1.0              # cells: to the lattice at or above this
RESTAMP_DFILL = 1.0 / 8.0  # restamp only when a cell's fill moves by more than this


def decide(d_star, dx, is_record):
    """Is the thing a record after this tick? True = record. `d_star` is the decisive
    dimension (thickness, bore, piece size, contents depth) in metres; `dx` the cell size.
    A record stays a record until d* reaches 1.0 cell; lattice matter stays lattice until
    d* falls to 0.5 cell. Between, nothing changes, so a d* wandering over 0.6-0.9 cells
    never flips it."""
    r = np.asarray(d_star) / dx
    if is_record:
        return r < HYST_UP
    return r <= HYST_DOWN


# ── footprint and restamp ────────────────────────────────────────────────────
def footprint(center, axes, dims, dx, origin=(0.0, 0.0, 0.0), max_points=40000):
    """Cells a box (length, width, thickness) of a record touches, and the volume in each.

    center (3,) m; axes (3,3) rows = unit vectors along length, width, thickness; dims
    (l, w, t) m. The box is sampled on a regular grid of midpoints (at least 3 along the
    thickness, 4 per cell elsewhere) and the samples are counted per cell: exact for a
    box aligned with the grid, close otherwise. Returns (cells (M,3) int sorted, vol (M,)
    m3). The volumes sum to l w t exactly (every sample carries l w t / N)."""
    dims = np.asarray(dims, dtype=np.float64)
    axes = np.asarray(axes, dtype=np.float64)
    n = np.maximum(np.ceil(dims / (dx / 4.0)), 3).astype(int)
    while n.prod() > max_points:
        n = np.maximum(np.ceil(n * 0.8), 3).astype(int)
    loc = [((np.arange(k) + 0.5) / k - 0.5) * d for k, d in zip(n, dims)]
    L, W, T = np.meshgrid(*loc, indexing="ij")
    pts = (np.asarray(center, dtype=np.float64)
           + L.reshape(-1, 1) * axes[0] + W.reshape(-1, 1) * axes[1] + T.reshape(-1, 1) * axes[2])
    idx = np.floor((pts - np.asarray(origin)) / dx).astype(np.int64)
    cells, counts = np.unique(idx, axis=0, return_counts=True)
    return cells, counts * (dims.prod() / pts.shape[0])


def fills(vol, dx):
    """Fill fraction of each footprint cell (volume / cell volume)."""
    return np.asarray(vol) / dx ** 3


def spread(mass, weights, cap=None):
    """Distribute `mass` over cells in proportion to `weights`, no cell above `cap`
    (water-filling: the excess of a capped cell goes to the others in proportion). Returns
    (placed (nc,) float64, unplaced). Sum of placed + unplaced = mass to float64 round-off."""
    w = np.asarray(weights, dtype=np.float64)
    t = mass * w / w.sum()
    if cap is None:
        return t, 0.0
    cap = np.asarray(cap, dtype=np.float64)
    for _ in range(t.size + 1):
        over = t > cap
        if not over.any():
            return t, 0.0
        excess = float((t - cap)[over].sum())
        t[over] = cap[over]
        room = (~over) & (t < cap)
        if not room.any():
            return t, excess
        t[room] += excess * w[room] / w[room].sum()
    return t, 0.0


def round_out(target, dtype=np.float32, room=None):
    """Write float64 targets into `dtype`, put the rounding residue in the largest cell
    (that has room, if `room` (nc,) float64 headroom is given), and return (values,
    residue) where residue = sum(target) - sum(float64(values)) is what no cell can hold
    (about one ulp of the largest cell). The caller books it into `rem`."""
    vals = target.astype(dtype)
    r = float(target.sum() - vals.astype(np.float64).sum())
    j = int(np.argmax(target if room is None else np.where(room > 0.0, target, -1.0)))
    vals[j] = dtype(float(vals[j]) + r)
    return vals, float(target.sum() - vals.astype(np.float64).sum())


def restamp(smass, flat_idx, mass_g, weights, cap_g=None, dtype=np.float32):
    """Write a record's mass into the lattice array `smass` at flat indices against its
    footprint weights (volume in each cell): smass = rho x volume_in_cell, rounding residue
    in the largest cell, no cell above `cap_g` (rho x cell volume: fill never above 1).
    Returns (residue, unplaced): residue to book into `rem`; unplaced > 0 only if the cells
    cannot hold the mass at all (add cells)."""
    placed, left = spread(float(mass_g), weights, cap_g)
    room = None if cap_g is None else np.asarray(cap_g, dtype=np.float64) - placed
    vals, res = round_out(placed, dtype, room)
    np.put(smass, np.asarray(flat_idx), vals)
    return res, left


def needs_restamp(old_idx, new_idx, old_fill, new_fill):
    """Restamp only when the occupied set changes or a cell's fill moves by more than 1/8.
    Moving a thing whole never restamps."""
    if not np.array_equal(old_idx, new_idx):
        return True
    return bool((np.abs(np.asarray(new_fill) - np.asarray(old_fill)) > RESTAMP_DFILL).any())


# ── crossings with exact remainders ──────────────────────────────────────────
def largest_remainder(n, weights):
    """Split the integer n over cells in proportion to weights so the integers add to n
    exactly (counts are exact in a crossing, never rounded)."""
    w = np.asarray(weights, dtype=np.float64)
    exact = n * w / w.sum()
    out = np.floor(exact).astype(np.int64)
    left = int(n - out.sum())
    order = np.argsort(-(exact - out), kind="stable")           # ties by index: deterministic
    out[order[:left]] += 1
    return out


def to_lattice(recs, host_id, g, E, weights, cap_g=None, dtype=np.float32):
    """A record (grams per species g (NS,), joules E, float64) becomes lattice matter in
    `len(weights)` cells. Returns (m (nc,NS) dtype grams, Ecell (nc,) dtype joules). What
    the dtype cannot hold goes into recs.rem[(host_id, ("m", s) / "E")] and is counted by
    `records_owned` until the next read. Total mass cap_g per cell as in `restamp`."""
    g = np.asarray(g, dtype=np.float64)
    tot = float(g.sum())
    nc = len(weights)
    if tot > 0.0:
        placed, left = spread(tot, weights, cap_g)
        w_eff = placed / tot
    else:
        w_eff, left = np.asarray(weights, dtype=np.float64) / np.sum(weights), 0.0
    m = np.zeros((nc, NS), dtype=dtype)
    for s in range(NS):
        if g[s] == 0.0:
            continue
        # unplaced matter stays in the remainder, so nothing is lost when the cells are full
        tgt = g[s] * w_eff
        vals, res = round_out(tgt, dtype)
        m[:, s] = vals
        recs.rem_add(host_id, ("m", s), (g[s] - float(tgt.sum())) + res)
    Ecell, res = round_out(E * w_eff, dtype)
    recs.rem_add(host_id, "E", (E - float((E * w_eff).sum())) + res)
    return m, Ecell


def from_lattice(recs, host_id, m, Ecell):
    """Lattice matter becomes a record: grams per species and joules in float64, plus the
    remainder the last crossing left (taken, so it is counted once)."""
    g = np.asarray(m, dtype=np.float64).sum(axis=0)
    for s in range(NS):
        g[s] += recs.rem_take(host_id, ("m", s))
    E = float(np.asarray(Ecell, dtype=np.float64).sum()) + recs.rem_take(host_id, "E")
    return g, E
