"""EDGEWEAR: the edge as state, wired into a World (wave 13c, lane C8b, T9 and T10).

matedge.py holds the closed forms (geometry, Archard wear, rolling, honing,
filings) and knows nothing of a World. This is the glue. Two switches, each its
own name in `world.mat_laws`:

  edge    A swung edge that meets something WEARS (Archard) and ROLLS if the
          blow is hard against a hard target. The tip radius r is the only
          thing that changes; the contact area follows (A = A0 + 2 L dr) and is
          written back into the `edge` array of the flying body, so every old
          reader (`props`, the mind menu, `_blow`) still reads one area. The
          array is rewritten only when r moved by 1 percent (matedge.REQUANT).
          The worn-off volume is owed as mass; whole grams leave the tool as
          rubble of its material (see `_deposit`).
  wound   The wound split. `_blow` used to open skin under ONE area (energy per
          edge area, 1e5 J/m2) and to drive the cut under the same area. Now
          skin opens under the TIP area L x 2r and the cut deepens under the
          wedge area A. A new edge opens skin at today's energy (the constant
          is rescaled so the two agree at r0), so only a worn or rolled edge
          changes: it bruises but does not cut.

State: `things[eid]["edge_st"]`, or, for an edge no thing owns, on the flying
body (`b["edge_st"]`). It holds L, theta, the exact r, the area last written,
and the grams owed. A cell with no state is pristine: its geometry is worked
out from its declared area (matedge.geom_from_area) every time it is asked.

Everything numeric is matedge's. The glue adds (all (u), in the C8b ledger):
  MU_FLANK   0.4, the friction of the flank against the target
  MAX_CHUNKS filings chunks placed per blow
"""
import functools

import numpy as np

from . import matedge as ME
from .materials import AIR, FLESH, MATREC

MU_FLANK = 0.4
MAX_CHUNKS = 4
R_MAX = 0.02             # m (u): a tip crushed past 2 cm is a stump; the radius stops there (the area is then far above a voxel face, so it reads as blunt)
SOFT_H = 1.0e5           # Pa: hardness of a row with none (leaf, rope, ice): as soft as ash

@functools.lru_cache(maxsize=1)
def _hard():
    """Indentation hardness by material id, Pa."""
    n = max(64, len(MATREC.by_id_list))
    h = np.full(n, SOFT_H)
    for i, r in enumerate(MATREC.by_id_list):
        if r.H:
            h[i] = float(r.H)
    return h


def on(w, law):
    return law in w.mat_laws


def _sharpest(b, vox_m):
    """(index, area) of the keenest edge cell of a body, or None. An area at or
    above a voxel face is no edge at all (the old rule in _spend)."""
    e = b.get("edges")
    if e is None or not len(e):
        return None
    pos = np.flatnonzero(e > 0)
    if not len(pos):
        return None
    i = int(pos[np.argmin(e[pos])])
    a = float(e[i])
    return None if a >= vox_m * vox_m else (i, a)


def _holder(w, b, i):
    """The dict the edge's state lives in: its thing, else the flying body."""
    eid = int(b["eids"][i]) if "eids" in b else 0
    t = w.things.get(eid) if eid else None
    return t if t is not None else b


def _state(w, b, i, a):
    """The edge's state now, or None if it has none (pristine). A state whose
    written area is not the area on the cell is stale (somebody redrew the
    edge) and is dropped."""
    st = _holder(w, b, i).get("edge_st")
    if st is not None and abs(st["A_w"] - a) > 1e-6 * max(a, 1e-12):
        return None
    return st


def geom_of(w, b, i, a):
    """The edge geometry now: from the state, else from the declared area."""
    st = _state(w, b, i, a)
    if st is not None:
        return ME.EdgeGeom(st["L"], st["theta"], st["r"])
    return ME.geom_from_area(a)


def plan(w, b, struck):
    """Look at the striking edge BEFORE the blow is spent (a target that
    shatters is air afterwards, and the wound is dealt by the edge as it was).
    Returns None when the body carries no edge or both switches are off, else
    (index, area, hardest target hardness, tip area or None)."""
    if not (on(w, "edge") or on(w, "wound")):
        return None
    s = _sharpest(b, 0.1 * w.scale)
    if s is None:
        return None
    i, a = s
    tip = ME.tip_area(geom_of(w, b, i, a)) if on(w, "wound") else None
    mats = np.asarray(w.mat[tuple(np.asarray(struck, np.int64).reshape(-1, 3).T)])
    mats = mats[mats != AIR]
    H_t = float(_hard()[mats].max()) if len(mats) else 0.0
    return i, a, H_t, tip


def after_blow(w, b, struck, joules, pl):
    """Wear and roll the striking edge for a blow of `joules` on `struck` cells.
    Called once at the end of `_spend`, so the wound it just made was dealt by
    the edge as it was."""
    if pl is None or not on(w, "edge") or joules <= 0.0:
        return
    i, a, H_t, _ = pl
    if H_t <= 0.0:
        return
    holder = _holder(w, b, i)
    g = geom_of(w, b, i, a)
    H = _hard()
    temper = float(holder.get("temper", 1.0)) if holder is not b else 1.0
    m_tool = int(b["mats"][i])
    H_e = float(H[m_tool]) * temper
    r1, V = ME.wear(g, joules, H_e, H_t, MU_FLANK)
    r2 = min(float(ME.roll(ME.EdgeGeom(g.L_m, g.theta_rad, r1), joules, H_e, H_t)), R_MAX)
    st = _state(w, b, i, a)
    if st is None:
        st = {"L": g.L_m, "theta": g.theta_rad, "r": g.r_m, "A_w": a, "r_w": g.r_m, "owed_g": 0.0}
    st["r"] = r2
    rho = float(MATREC.by_id_list[m_tool].rho) if m_tool < len(MATREC.by_id_list) else 0.0
    st["owed_g"], chunks = ME.filings_step(st["owed_g"], ME.volume_to_g(V, rho))
    if ME.needs_write(st["r_w"], r2):
        A2 = ME.area(ME.EdgeGeom(g.L_m, g.theta_rad, r2))
        e = b["edges"]
        e[e == np.float32(a)] = np.float32(A2)         # every cell that held this edge
        st["A_w"], st["r_w"] = float(np.float32(A2)), r2
    else:
        st["A_w"] = a
    holder["edge_st"] = st
    if chunks:
        placed = _deposit(w, b, i, struck, min(chunks, MAX_CHUNKS), holder)
        st["owed_g"] += (chunks - placed) * ME.FILING_CHUNK_G     # the chunk waits in the tool


def _deposit(w, b, i, struck, n, holder):
    """Put `n` one-gram chunks of the tool's material, as rubble, in an air cell
    next to the contact. The tool loses the gram and its share of the heat, so
    tool + filings is constant and the heat ledger closes. Returns how many
    were placed (0 when no air cell is free: the chunk then waits)."""
    c0 = tuple(int(v) for v in np.asarray(struck[0]))
    nx, ny, nz = w.shape
    spots = []
    for d in ((0, 0, 1), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, -1)):
        c = (c0[0] + d[0], c0[1] + d[1], c0[2] + d[2])
        if 0 <= c[0] < nx and 0 <= c[1] < ny and 0 <= c[2] < nz \
                and w.mat[c] == AIR and w.fvol[c] <= 0.0 and w.smass[c] <= 0.0:
            spots.append(c)
    done = 0
    eid = int(b["eids"][i]) if "eids" in b else 0
    for c in spots[:n]:
        m = float(b["masses"][i])
        if m <= 2.0 * ME.FILING_CHUNK_G:
            break
        dE = float(b["Es"][i]) * ME.FILING_CHUNK_G / m
        b["masses"][i] = m - ME.FILING_CHUNK_G
        b["Es"][i] -= dE
        w.mat[c] = int(b["mats"][i])
        w.smass[c] = ME.FILING_CHUNK_G
        w.E[c] += dE
        w.rubble[c] = True
        w.edge[c] = 0.0
        if eid:
            w.eid[c] = eid
        w._rub_gen += 1
        done += 1
    return done


def hone_thing(w, eid, strokes, force_N, stroke_m, H_abrasive, grit_m):
    """Hone a resting thing's edge on an abrasive (no action menu calls this
    yet: it is the law, ready for the verb). Returns the tip radius after, or
    None if the thing has no edge cell. The thing's `edge` cells are rewritten
    when r moved by 1 percent; the volume removed is owed as filings."""
    idx = w._cells_now(eid)
    if not len(idx):
        return None
    flat = w.edge.ravel()
    e = flat[idx]
    pos = np.flatnonzero(e > 0)
    if not len(pos):
        return None
    a = float(e[pos].min())
    t = w.things[eid]
    st = t.get("edge_st")
    if st is not None and abs(st["A_w"] - a) > 1e-6 * a:
        st = None
    g = ME.EdgeGeom(st["L"], st["theta"], st["r"]) if st else ME.geom_from_area(a)
    m_tool = int(w.mat.ravel()[idx[pos[0]]])
    H_e = float(_hard()[m_tool]) * float(t.get("temper", 1.0))
    r2, V = ME.hone(g, strokes, force_N, stroke_m, H_e, H_abrasive, grit_m)
    st = st or {"L": g.L_m, "theta": g.theta_rad, "r": g.r_m, "A_w": a, "r_w": g.r_m, "owed_g": 0.0}
    st["r"] = float(r2)
    rho = float(MATREC.by_id_list[m_tool].rho)
    st["owed_g"], _ = ME.filings_step(st["owed_g"], ME.volume_to_g(V, rho))
    if ME.needs_write(st["r_w"], r2):
        A2 = ME.area(ME.EdgeGeom(g.L_m, g.theta_rad, r2))
        flat[idx[e == np.float32(a)]] = np.float32(A2)
        st["A_w"], st["r_w"] = float(np.float32(A2)), float(r2)
    else:
        st["A_w"] = a
    t["edge_st"] = st
    return float(r2)
