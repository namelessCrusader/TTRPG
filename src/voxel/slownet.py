"""TIER 1: THE THERMAL NETWORK OF A REGION (wave 13, T5, spec 4.10).

With a living person in the room (a 80 W heat source) or an outside that is not
the 20 C reference, the lattice is never bit-still, so a night cannot be a
tier-0 skip. `thermal_net(w, region)` coarse-grains the lattice's own heat law
into an rc_net spec (slowforms.NetSpec), `slowforms.rc_advance_arrays` advances
it in closed form, and `write_back` puts the result into E with the energy
EXACT. Exact in HOW MUCH (relative to the lumped model), approximate in WHERE;
the bound against the tick law is a test, and the jump REFUSES where it cannot
be trusted.

NODES: (material kind, block of B cells per axis, connected component, and for
solids the layer index of the distance to the nearest different material,
capped at 4 so a 15 cm wall keeps its through-thickness gradient). C_A is the
sum of the cells' capacities; G_AB the parallel sum of the heat law's own FACE
conductances (solid|solid harmonic k plus radiation across gaps, solid|liquid
h; the AIR faces below), so no new tuned number. Every face that looks onto gas
radiates to the universe at the boundary temperature: a RADIATIVE link
`a (T^4 - Tb^4)`, re-linearised per step. The skin split of `_surfaces` is
ignored (it is a seconds-scale sub-cell effect, zeroed by the write-back).

THE AIR depends on temperature in the lattice itself, and so does its link here
(slowforms.NetDyn, read at the node temperatures of each step, the links of
the step taken at its mean by a fixed point): gas|gas still air plus the eddies
of hot air (k grows with T - 20, EDDY), surface|gas H_CONV plus the radiation of
hot gas (EPS_GAS), and the BUOYANT RISE of `_law_rise` (hot gas hands its
excess up at v = sqrt(g dx dT / T), never more than RISE_MAX a tick) as a
one-way link, live while the lower node is the hotter. An earlier version read
the eddies once at the start temperatures with a hand-set plume factor and was
60 % low on the room's air; the plume crosses a room in seconds, so the box
also grows to hold the whole closed air space it opens into (gas_reach) and a
jump that would still lose heat into the air of the ring is refused (escape).

WRITE-BACK keeps the sub-node shape and is exact in energy: every cell of node A
gets E_cell += c_cell dT_A, so the sum of dE_cell is C_A dT_A. After the write,
a cell within SETTLE (0.01 K) of its boundary value, in a node heading there,
is SNAPPED to the boundary value and the residue booked into `shed`, so tier 0
can resume (a proportional removal never reaches E == 0).

REFUSALS, checked before the jump:
  (a) a threshold row (ignition, boil, melt, freeze, oil flash) inside any
      cell's trajectory [start, steady] widened by the node's own spread;
  (b) the Biot number of a solid cluster with the RADIATIVE h at its current
      temperature, h L/k with L = V/A. Over BIOT_MAX it is refused unless the
      cluster resolves 3 or more layers (then the layer nodes carry the
      gradient). Only clusters that are thermally active (more than
      BIOT_ACTIVE_K away from the boundary) are asked;
  (c) any hosted process with needs_history whose box overlaps the region;
  (d) V2: the room's O2 against the twins' fluxes (v2_o2_ok);
  (e) ESCAPE: heat leaving the box into the air of the ring round it is more
      than ESCAPE_MAX of the air's heat budget (the ring is a sink at the
      boundary value, so a plume that reached it would be lost).
"""
import math

import numpy as np

from . import slowforms as F
from .heat import FILM, RISE_MAX
from .materials import (AIR, AIR_C, AMBIENT, BOIL, EDDY, EPS_GAS, FLUID, FREEZE, GRAVITY,
                        H_CONV, MELT, NOFLUID, OIL, SIGMA, SOLID, TICK_S, WATER,
                        _ALPHA, _EMISS_ARR, _FEMISS_ARR, _FK_ARR, _FLK_ARR,
                        _IGN_ARR, _KSOLID_ARR, NMAT)

K0 = 273.15
SETTLE_K = 0.01                 # the 'settle' row: a cell this close to its boundary is snapped
BIOT_MAX = 0.1                  # above this a lump is not a lump
BIOT_ACTIVE_K = 1.0             # a cluster this close to its steady state is not asked
V2_O2_MAX = 0.05                # a jump may use at most 5 % of a room's volume in O2 (1.2 % is fine, 12 % is not)
O2_PER_L_AIR = 0.2095           # litres of O2 per litre of fresh air
LAYERS = 4
WIND_DECAY_S = 60.0            # a jump at least this long forgets the warm-air wind


class NetResult:
    """The coarse-grained region. Plain data (numpy arrays), not stored in w."""
    __slots__ = ("box", "spec", "node_of", "T0", "Cn", "kind", "spread_lo", "spread_hi",
                 "ncell", "B", "refusals", "cell_C", "cell_T", "layer", "clusters",
                 "owned_cells", "tb", "cluster_id", "alpha", "dx", "skip", "skip_node")

    def __init__(self):
        self.refusals = []


def _neighbours_any(mask):
    """cells that have a True 6-neighbour (not themselves)."""
    n = np.zeros(mask.shape, bool)
    n[1:] |= mask[:-1]
    n[:-1] |= mask[1:]
    n[:, 1:] |= mask[:, :-1]
    n[:, :-1] |= mask[:, 1:]
    n[:, :, 1:] |= mask[:, :, :-1]
    n[:, :, :-1] |= mask[:, :, 1:]
    return n


def _components(kind, blk):
    """Connected components of equal `kind` (and equal `blk` when given), by
    min-label propagation, numpy only. Returns an int array of labels."""
    shape = kind.shape
    lab = np.arange(kind.size, dtype=np.int64).reshape(shape)
    pairs = []
    for axis in range(3):
        a = [slice(None)] * 3
        b = [slice(None)] * 3
        a[axis], b[axis] = slice(None, -1), slice(1, None)
        a, b = tuple(a), tuple(b)
        same = kind[a] == kind[b]
        if blk is not None:
            same &= blk[a] == blk[b]
        pairs.append((a, b, same))
    for _ in range(kind.size + 1):
        changed = False
        for a, b, same in pairs:
            la, lb = lab[a], lab[b]
            lo = np.minimum(la, lb)
            m = same & ((la != lo) | (lb != lo))
            if m.any():
                changed = True
                np.minimum(la, np.where(same, lo, la), out=la)
                np.minimum(lb, np.where(same, lo, lb), out=lb)
        if not changed:
            break
    return lab


def _layers(mat, solid):
    """Layer index of each solid cell: 0 where it touches a different material
    (or the edge of the box), k where it is k steps in, LAYERS-1 and deeper = core."""
    pad = np.pad(mat, 1, constant_values=255)
    own = mat
    diff = np.zeros(mat.shape, bool)
    for axis in range(3):
        for sgn in (-1, 1):
            sl = [slice(1, -1)] * 3
            sl[axis] = slice(0, -2) if sgn < 0 else slice(2, None)
            diff |= pad[tuple(sl)] != own
    layer = np.full(mat.shape, LAYERS - 1, np.int8)
    cur = solid & diff
    layer[cur] = 0
    done = cur.copy()
    for k in range(1, LAYERS - 1):
        nxt = solid & ~done & _neighbours_any(cur)
        layer[nxt] = k
        done |= nxt
        cur = nxt
    layer[~solid] = 0
    return layer


DEFAULT_HORIZON_S = 12 * 3600.0      # a night and a bit: how far heat can walk, for the pad
PAD_MAX = 8                          # cells


def diffusion_pad(w, horizon_s):
    """Cells of margin so heat cannot walk out of the box in `horizon_s`:
    1 + two diffusion lengths of the most diffusive solid, capped. What does
    leave is carried by the boundary links to the ring (at the boundary value)."""
    from .materials import SOLID
    alpha = max(SOLID[m][2] / (SOLID[m][0] * SOLID[m][1] * 1000.0)
                for m in SOLID if m != AIR and SOLID[m][0] > 100.0)
    dx = 0.1 * w.scale
    return int(min(PAD_MAX, 1 + math.ceil(2.0 * math.sqrt(alpha * horizon_s) / dx)))


GAS_CELLS_MAX = 250000               # the most air cells of an enclosure a net will take whole
ESCAPE_MAX = 0.05                    # heat leaving into the air of the ring may be this share of the air's heat budget


def gas_reach(w, box, cap=None):
    """The bounding box (three slices) of every air cell connected to the air
    inside `box` (flood fill through air), or None when that air is more than
    `cap` cells (the room is not closed in any useful sense). Heat in air does
    not stop at a diffusion length: a plume crosses a room in seconds."""
    cap = GAS_CELLS_MAX if cap is None else cap
    free = w.mat == AIR
    reach = np.zeros(w.shape, bool)
    reach[box] = free[box]
    count = int(reach[box].sum())
    if count == 0:
        return None
    lo = np.array([s.start or 0 for s in box])
    hi = np.array([s.stop if s.stop is not None else n for s, n in zip(box, w.shape)])
    while True:
        wl, wh = np.maximum(lo - 1, 0), np.minimum(hi + 1, w.shape)
        win = tuple(slice(int(a), int(b)) for a, b in zip(wl, wh))
        r, f = reach[win], free[win]
        new = _neighbours_any(r) & f & ~r
        if not new.any():
            break
        reach[win] = r | new
        count += int(new.sum())
        if count > cap:
            return None
        pos = np.argwhere(new) + wl
        lo = np.minimum(lo, pos.min(axis=0))
        hi = np.maximum(hi, pos.max(axis=0) + 1)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def grow_to_enclosure(w, box, pad):
    """`box` grown by `pad`, and then to hold the whole air space it opens into
    (when that is a closed room of at most GAS_CELLS_MAX cells), and the pad
    again round that."""
    first = _grown(box, w.shape, pad)
    reach = gas_reach(w, first)
    if reach is None:
        return first
    lo = [min(a.start, b.start) for a, b in zip(first, reach)]
    hi = [max(a.stop, b.stop) for a, b in zip(first, reach)]
    return _grown(tuple(slice(a, b) for a, b in zip(lo, hi)), w.shape, pad)


def default_region(w, pad=None, horizon_s=DEFAULT_HORIZON_S):
    """The hot window grown by `pad` cells (default: the diffusion pad for
    `horizon_s`) and to the closed air space it touches: every cell with E != 0
    and a margin the heat will not leave."""
    if not bool(w.E.any()):
        return None
    return grow_to_enclosure(w, w._window(w.E != 0, 0),
                             diffusion_pad(w, horizon_s) if pad is None else pad)


def _grown(sl, shape, k=1):
    return tuple(slice(max((s.start or 0) - k, 0), min((s.stop if s.stop is not None else n) + k, n))
                 for s, n in zip(sl, shape))


def face_arrays(w, sl, u=None, T=None, dyn=False):
    """The heat law's own face conductances (W/K) inside the box `sl`, linearised
    at the temperatures T (default: the world's), for the three axes. Returns
    [(ia, ib, G)] with flat indices into the box. Also the radiation-to-world
    coefficient a_i (W/K^4) of every cell and the surface bundle `u`.

    With dyn=True the faces that depend on temperature (gas | gas and surface |
    gas, and the buoyant rise of hot air) are NOT in the G of the list (it holds
    only the fixed part, zero there) and come back as a fifth value: a list of
    (ia, ib, kind, c, side) with the kinds of slowforms.NetDyn."""
    u = w._surfaces(sl) if u is None else u
    T = (AMBIENT + w.E[sl] / u.C).astype(np.float64) if T is None else T
    solid, wet, gas = u.solid, u.wet, u.gas
    mat, fl, af, void = u.mat, u.fl, u.af.astype(np.float64), u.void.astype(np.float64)
    dx = 0.1 * w.scale
    A = dx * dx
    k_sol = _KSOLID_ARR[mat].astype(np.float64)
    k_liq = _FLK_ARR[fl].astype(np.float64)
    eddy = 1000.0 * AIR_C / EDDY * np.maximum(T - AMBIENT, 0.0)
    k = np.where(solid, k_sol, k_liq)
    k = np.where(gas, float(_KSOLID_ARR[AIR]) + eddy, k)
    h_liq = _FK_ARR[fl].astype(np.float64)
    eps = _EMISS_ARR[mat].astype(np.float64)
    out, dynf = [], []
    shape = T.shape
    idx = np.arange(T.size, dtype=np.int64).reshape(shape)
    c_air = AIR_C * w.vox_l
    for axis in range(3):
        a = [slice(None)] * 3
        b = [slice(None)] * 3
        a[axis], b[axis] = slice(None, -1), slice(1, None)
        a, b = tuple(a), tuple(b)
        sa, sb, ga, gb, wa, wb = solid[a], solid[b], gas[a], gas[b], wet[a], wet[b]
        ka, kb = k[a], k[b]
        G = 2.0 * ka * kb / np.maximum(ka + kb, 1e-12) * dx
        ss = sa & sb
        Ta, Tb_ = T[a], T[b]
        if ss.any():
            tm = 0.5 * (Ta + Tb_) + K0
            rad = (np.maximum(void[a], void[b]) * np.minimum(eps[a], eps[b])
                   * 4.0 * SIGMA * tm ** 3 * A)
            G = np.where(ss, G + rad, G)
        sw = (sa & wb) | (wa & sb)
        if sw.any():
            G = np.where(sw, np.where(wa, h_liq[a], h_liq[b]) * A * af[a] * af[b], G)
        sg = ((sa | wa) & gb) | (ga & (sb | wb))
        if sg.any():
            tka, tkc = Ta + K0, Tb_ + K0
            gas_hot = np.where(ga, Ta > Tb_, Tb_ > Ta)
            rad = np.where(gas_hot, EPS_GAS * SIGMA * (tka * tka + tkc * tkc) * (tka + tkc), 0.0)
            G = np.where(sg, (H_CONV + rad) * A * af[a] * af[b], G)
        if dyn:
            gg = ga & gb & ~sg
            for mask, kind, cc, side in ((gg, 0, np.full(G.shape, dx), np.zeros(G.shape, bool)),
                                         (sg, 1, A * af[a] * af[b], ga)):
                if mask.any():
                    dynf.append((idx[a][mask], idx[b][mask], kind, cc[mask], side[mask]))
            G = np.where(gg | sg, 0.0, G)
            if axis == 2:
                both_air = (mat[a] == AIR) & (mat[b] == AIR)
                if both_air.any():
                    n_ = int(both_air.sum())
                    dynf.append((idx[a][both_air], idx[b][both_air], 2, np.full(n_, c_air),
                                 np.ones(n_, bool)))
        out.append((idx[a].ravel(), idx[b].ravel(), G.ravel()))
    e = np.where(solid, eps, _FEMISS_ARR[fl].astype(np.float64))
    rad_from = (solid | wet) & (u.n_gas > 0)
    a_i = np.where(rad_from, e * SIGMA * A * af * u.n_gas, 0.0)
    if dyn:
        return out, a_i, u, T, dynf
    return out, a_i, u, T


def _threshold_rows(mat_present, fl_present, has_fuel, has_water):
    """The temperatures (C) at which something other than temperature changes."""
    rows = []
    for m in mat_present:
        m = int(m)
        if m < NMAT and np.isfinite(_IGN_ARR[m]) and has_fuel.get(m, False):
            rows.append((float(_IGN_ARR[m]), f"ignition of mat {m}"))
        if m in MELT:
            rows.append((float(MELT[m][0]), f"melting of mat {m}"))
    for f in fl_present:
        f = int(f)
        if f == WATER and has_water:
            rows.append((float(BOIL), "water boils"))
        if f == OIL:
            rows.append((float(FLUID[OIL][3]), "oil flash"))
        if f in FREEZE:
            rows.append((float(FREEZE[f][0]), "freezing of molten metal"))
    return rows


def _DYN_CONSTS(dx):
    """The constants of the temperature-dependent faces (slowforms.NetDyn.k):
    the heat law's own numbers, none new."""
    return dict(dx=dx, k_air=float(_KSOLID_ARR[AIR]), eddy=1000.0 * AIR_C / EDDY, t0=AMBIENT,
                h_conv=H_CONV, eps_gas=EPS_GAS, sigma=SIGMA, grav=GRAVITY, tick_s=TICK_S,
                rise_max=RISE_MAX)


def thermal_net(w, region=None, B=None, max_nodes=2000, sources=(), tb=AMBIENT,
                horizon_s=DEFAULT_HORIZON_S, skip=None):
    """Coarse-grain the lattice heat law of the box `region` (default: the hot
    window grown by the diffusion pad of `horizon_s`) into a NetResult, or
    return one whose `refusals` says why not. `sources`: [(cell (x, y, z) or an
    (N, 3) array of cells, watts)] in world cells (a twin's heat). The faces
    between the box and the ring of cells around it become links to the
    boundary at `tb`; the ring must be at the boundary value (E == 0).

    `skip` (wave 13, C10): an (N, 3) array of world cells that the net must NOT
    touch, the flesh of the bodies a body-heat twin carries. Each skipped cell
    is its own node (kind 500) with no link, no radiation and no source: it
    holds its energy exactly, and the air next to it sees an insulated wall
    (the twin gives the room the body's heat as a source instead)."""
    res = NetResult()
    if region is None:
        region = default_region(w, horizon_s=horizon_s)
    if region is None:
        region = (slice(0, w.shape[0]), slice(0, w.shape[1]), slice(0, w.shape[2]))
    sl = region
    res.box, res.tb = sl, tb
    u = w._surfaces(sl)
    C = u.C.astype(np.float64)
    T = AMBIENT + w.E[sl].astype(np.float64) / C
    mat, fl = u.mat, u.fl
    solid, wet = u.solid, u.wet
    shape = mat.shape
    kind = np.where(solid, mat.astype(np.int64), np.where(wet, 100 + fl.astype(np.int64), 0))
    sk_reg = np.zeros(shape, bool)
    if skip is not None and len(skip):
        sc = np.asarray(skip, np.int64).reshape(-1, 3) - np.array([s.start or 0 for s in sl])
        ok = ((sc >= 0) & (sc < np.array(shape))).all(axis=1)
        sk_reg[tuple(sc[ok].T)] = True
        kind = np.where(sk_reg, 500, kind)
    res.skip = sk_reg
    ncell = mat.size
    if B is None:
        B = max(1, int(math.ceil((ncell / max_nodes) ** (1.0 / 3.0))))
    res.B = B
    # block of B cells per axis
    ix, iy, iz = np.indices(shape)
    nby, nbz = -(-shape[1] // B), -(-shape[2] // B)
    blk = ((ix // B) * nby + (iy // B)) * nbz + (iz // B)
    lab = _components(kind, blk) if B > 1 else np.arange(ncell, dtype=np.int64).reshape(shape)
    lay = _layers(mat, solid) if B > 1 else np.zeros(shape, np.int8)
    key = np.stack([kind.ravel(), blk.ravel(), lab.ravel(), lay.ravel().astype(np.int64)], axis=1)
    _, node_of = np.unique(key, axis=0, return_inverse=True)
    node_of = node_of.ravel().astype(np.int64)
    n = int(node_of.max()) + 1
    res.node_of = node_of.reshape(shape)
    Cn = np.bincount(node_of, C.ravel(), n)
    En = np.bincount(node_of, (C * (T - AMBIENT)).ravel(), n)
    T0 = AMBIENT + En / Cn
    res.Cn, res.T0, res.ncell = Cn, T0, np.bincount(node_of, minlength=n)
    res.skip_node = np.zeros(n, bool)
    res.skip_node[node_of[sk_reg.ravel()]] = True
    kn = np.zeros(n, np.int64)
    kn[node_of] = kind.ravel()
    res.kind = kn
    order = np.argsort(node_of, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(node_of[order])) + 1]
    Ts = T.ravel()[order]
    res.spread_lo = np.minimum.reduceat(Ts, starts)
    res.spread_hi = np.maximum.reduceat(Ts, starts)
    res.cell_C, res.cell_T, res.layer = C, T, lay
    res.dx = 0.1 * w.scale
    # The diffusivity of each material id, as long as the padded tables (a cell's id indexes it). The ids the OLD
    # dict knows (NMAT of them) keep the dict's own python-float formula, bit for bit; the dict has no key past
    # them (ICE is id 12 and lives in the arrays only), so the rest comes from the padded _ALPHA (a copy).
    a_sol = np.array(_ALPHA)
    a_sol[:NMAT] = [SOLID[m][2] / (SOLID[m][0] * SOLID[m][1] * 1000.0) if SOLID[m][0] > 0 else 0.0
                    for m in range(NMAT)]
    a_cell = np.where(solid, a_sol[mat], np.where(wet, 1.4e-7, 2.2e-5))   # gas: still air
    res.alpha = np.bincount(node_of, (a_cell * C).ravel(), n) / Cn
    big = _grown(sl, w.shape)
    ub = w._surfaces(big)
    Tb_ = (AMBIENT + w.E[big] / ub.C).astype(np.float64)
    inside = tuple(slice((s.start or 0) - (b.start or 0), (s.start or 0) - (b.start or 0) + (s.stop - (s.start or 0)))
                   for s, b in zip(sl, big))
    nb_all = np.full(Tb_.shape, -1, np.int64)
    nb_all[inside] = res.node_of
    ring = w.E[big].copy()
    ring[inside] = 0.0
    if bool(np.any(ring != 0.0)):
        res.refusals.append(("open", tuple(int(v) for v in np.argwhere(ring != 0.0)[0]),
                             "the ring round the region is not at the boundary value"))
    faces, a_big, _, _, dynf = face_arrays(w, big, ub, Tb_, dyn=True)
    a_i = np.where(sk_reg, 0.0, a_big[inside])
    nb_flat = nb_all.ravel()
    sk_big = np.zeros(Tb_.shape, bool)
    sk_big[inside] = sk_reg
    sk_flat = sk_big.ravel()
    gb = np.zeros(n)
    f_lo, f_hi, f_g = [], [], []                       # internal faces, the fixed part
    for ia, ib, G in faces:
        na, nb = nb_flat[ia], nb_flat[ib]
        live = ~(sk_flat[ia] | sk_flat[ib])
        m = (na != nb) & (na >= 0) & (nb >= 0) & live
        f_lo.append(np.minimum(na[m], nb[m]))
        f_hi.append(np.maximum(na[m], nb[m]))
        f_g.append(G[m])
        edge_a, edge_b = (na >= 0) & (nb < 0) & live, (nb >= 0) & (na < 0) & live
        gb += np.bincount(na[edge_a], G[edge_a], n) + np.bincount(nb[edge_b], G[edge_b], n)
    d_lo, d_hi, d_kind, d_c, d_side = [], [], [], [], []       # internal dynamic faces
    e_node, e_kind, e_c, e_side = [], [], [], []               # dynamic faces onto the boundary
    for ia, ib, kind, cc, side in dynf:
        na, nb = nb_flat[ia], nb_flat[ib]
        live = ~(sk_flat[ia] | sk_flat[ib])
        m = (na != nb) & (na >= 0) & (nb >= 0) & live
        swap = na[m] > nb[m]                           # `side` is about the first cell
        d_lo.append(np.minimum(na[m], nb[m]))
        d_hi.append(np.maximum(na[m], nb[m]))
        d_kind.append(np.full(int(m.sum()), kind, np.int8))
        d_c.append(cc[m])
        d_side.append(np.where(swap, ~side[m], side[m]))
        for sel, node, sd in (((na >= 0) & (nb < 0) & live, na, side),
                              ((nb >= 0) & (na < 0) & live, nb, ~side)):
            if sel.any():
                e_node.append(node[sel])
                e_kind.append(np.full(int(sel.sum()), kind, np.int8))
                e_c.append(cc[sel])
                e_side.append(sd[sel])
    cat = lambda xs, dt: np.concatenate(xs) if xs else np.zeros(0, dt)      # noqa: E731
    lo_all = cat(f_lo + d_lo, np.int64)
    hi_all = cat(f_hi + d_hi, np.int64)
    if len(lo_all):
        uk, inv = np.unique(lo_all * n + hi_all, return_inverse=True)
        inv = inv.ravel()
        nfix = sum(len(x) for x in f_lo)
        g = np.bincount(inv[:nfix], cat(f_g, float), len(uk))
        r, c = uk // n, uk % n
        d_link = inv[nfix:]
    else:
        r = c = np.zeros(0, np.int64)
        g = np.zeros(0)
        d_link = np.zeros(0, np.int64)
    dyn = F.NetDyn(
        np.concatenate([cat(d_kind, np.int8), cat(e_kind, np.int8)]),
        np.concatenate([d_link, np.full(len(cat(e_node, np.int64)), -1, np.int64)]),
        np.concatenate([cat(d_lo, np.int64), cat(e_node, np.int64)]),
        np.concatenate([cat(d_hi, np.int64), np.full(len(cat(e_node, np.int64)), -1, np.int64)]),
        np.concatenate([cat(d_c, float), cat(e_c, float)]),
        np.concatenate([cat(d_side, bool), cat(e_side, bool)]),
        _DYN_CONSTS(res.dx))
    rad = np.bincount(node_of, a_i.ravel(), n)
    q = np.zeros(n)
    for cell, watts in sources:
        cells = np.asarray(cell, np.int64).reshape(-1, 3)
        for cc in cells:
            c3 = tuple(int(cc[i]) - (sl[i].start or 0) for i in range(3))
            if not all(0 <= c3[i] < shape[i] for i in range(3)):
                res.refusals.append(("sources", tuple(int(v) for v in cc),
                                     "a source outside the region"))
                continue
            q[res.node_of[c3]] += watts / len(cells)
    res.spec = F.NetSpec(Cn, r, c, g, gb, q=q, rad=rad, dyn=dyn if len(dyn.kind) else None)
    res.clusters = _solid_clusters(w, res, u, T, a_i, sk_reg)
    return res


def _solid_clusters(w, res, u, T, a_i, skip=None):
    """[(cells, V, A, k, T mean, layers, place)] for each connected solid
    cluster, for the Biot check. Skipped cells (a twin's bodies) are no cluster."""
    solid = u.solid if skip is None else (u.solid & ~skip)
    if not solid.any():
        return []
    mat = u.mat
    kind = np.where(solid, mat.astype(np.int64), -1)
    lab = _components(kind, None)
    dx = 0.1 * w.scale
    lab_s = lab[solid]
    uniq, inv = np.unique(lab_s, return_inverse=True)
    inv = inv.ravel()
    nclu = len(uniq)
    cnt = np.bincount(inv, minlength=nclu)
    area = np.bincount(inv, (u.n_out[solid] * u.af[solid]).astype(np.float64), nclu) * dx * dx
    Tm = np.bincount(inv, T[solid], nclu) / cnt
    lay = _layers(mat, solid)
    lmax = np.zeros(nclu, np.int64)
    np.maximum.at(lmax, inv, lay[solid].astype(np.int64))
    kk = np.zeros(nclu)
    kk[inv] = _KSOLID_ARR[mat[solid]]
    eps = np.zeros(nclu)
    eps[inv] = _EMISS_ARR[mat[solid]]
    first = np.zeros((nclu, 3), np.int64)
    pos = np.argwhere(solid)
    first[inv[::-1]] = pos[::-1]
    cid = np.full(solid.shape, -1, np.int64)
    cid[solid] = inv
    res.cluster_id = cid
    return [(int(cnt[i]), cnt[i] * dx ** 3, area[i], kk[i], eps[i], Tm[i],
             int(lmax[i]) + 1, tuple(int(v) + (res.box[j].start or 0)
                                     for j, v in enumerate(first[i])))
            for i in range(nclu)]


def steady(res, tb=None):
    """The steady state of the net (large-dt limit), regularised so an isolated
    component with no sink is tolerated (its steady state is then the boundary
    value, which only makes the refusal check conservative)."""
    spec = res.spec
    tb = res.tb if tb is None else tb
    n = spec.n
    eps = 1e-6
    reg = F.NetSpec(spec.C, spec.rows, spec.cols, spec.g, np.full(n, eps), q=spec.q,
                    rad=spec.rad, dyn=spec.dyn)
    return F.rc_steady(reg, tb)


def gas_escape_power(res, T):
    """W leaving the box into the AIR of the ring round it (the faces onto the
    boundary that are gas | gas or buoyant rise) at node temperatures T. Heat
    into the ring's walls or ground is the boundary doing its job; heat into the
    ring's air is a plume leaving the box."""
    d = res.spec.dyn
    if d is None:
        return 0.0
    T = np.asarray(T, float)
    G = F.dyn_face_g(d, T, res.tb)
    sel = (d.nb < 0) & (d.kind != 1)
    return float(np.sum(G[sel] * (T[d.na[sel]] - res.tb)))


def refusals(w, res, sched=None, o2_used_g=None, T1=None, trajectory=True, t_s=None):
    """[(code, place, reason)] for every reason this region cannot be jumped.
    The trajectory checks (a thresholds, b Biot) read the node temperatures the
    jump will reach, `T1`, when given (a relaxation is monotone, so T0 .. T1
    bounds every cell); without it they read the steady state, which is
    conservative. `trajectory=False` leaves them out (the detector's pre-check,
    which has no duration to ask about). With `T1` and the jump's `t_s` seconds,
    (e) ESCAPE: the heat that left the box into the air of the ring round it,
    over that plus what the box's own air gained, may be at most ESCAPE_MAX. The ring is held at
    the boundary value, so a plume that reaches it would be lost; the box is
    grown to the whole closed room to make that rare."""
    out = list(res.refusals)
    spec = res.spec
    if spec.n > F.EIGEN_MAX * 100:
        out.append(("size", None, "too many nodes"))
    # (c) a hosted process that needs its history
    if sched is not None:
        for proc, form in sched.history_hosts():
            bx = sched.box_of(proc)
            if bx is not None and _overlap(bx, res.box):
                out.append(("history", bx, f"process {proc.id} ({proc.kind}) needs its history"))
    # (d) V2
    if o2_used_g is not None:
        ok, why = v2_o2_ok(w, res.box, o2_used_g)
        if not ok:
            out.append(("v2", None, why))
    if not trajectory:
        return out
    # (a) thresholds inside a trajectory
    if T1 is not None:
        Tss = np.asarray(T1, float)
    else:
        try:
            Tss = steady(res)
        except np.linalg.LinAlgError:
            Tss = np.full(spec.n, np.inf)
    if T1 is not None and t_s:
        p0, p1 = gas_escape_power(res, res.T0), gas_escape_power(res, Tss)
        e_esc = 0.5 * (abs(p0) + abs(p1)) * t_s
        gas = res.kind == 0
        budget = e_esc + float(np.dot(res.Cn[gas], np.abs(Tss - res.T0)[gas]))
        if budget > 1.0 and e_esc > ESCAPE_MAX * budget:
            out.append(("escape", None, f"{e_esc:.0f} J of the {budget:.0f} J the box's air gains leave it "
                                        f"into the air of the ring round it (limit {100 * ESCAPE_MAX:.0f} %)"))
    u_mat = w.mat[res.box]
    u_fl = w.fl[res.box]
    sm = w.smass[res.box]
    fv = w.fvol[res.box]
    node_of = res.node_of
    dT = Tss - res.T0
    lo = res.spread_lo + np.minimum(dT, 0.0)
    hi = res.spread_hi + np.maximum(dT, 0.0)
    for m in np.unique(u_mat):
        m = int(m)
        if m == AIR:
            continue
        fuel = {m: bool((sm[u_mat == m] > 0).any())}
        rows = _threshold_rows([m], [], fuel, False)
        for T_row, why in rows:
            nodes = np.unique(node_of[u_mat == m])
            nodes = nodes[~res.skip_node[nodes]]
            bad = nodes[(lo[nodes] < T_row) & (hi[nodes] > T_row)]
            if len(bad):
                cell = np.argwhere((node_of == bad[0]) & (u_mat == m))[0]
                out.append(("threshold", tuple(int(v) + (res.box[j].start or 0)
                                               for j, v in enumerate(cell)),
                            f"{why} at {T_row:g} C lies inside the trajectory"))
    wet_f = np.unique(u_fl[(fv > FILM * w.cap) & (u_mat == AIR)])
    for f in wet_f:
        rows = _threshold_rows([], [f], {}, True)
        for T_row, why in rows:
            sel = (u_fl == f) & (fv > FILM * w.cap)
            nodes = np.unique(node_of[sel])
            nodes = nodes[~res.skip_node[nodes]]
            bad = nodes[(lo[nodes] < T_row) & (hi[nodes] > T_row)]
            if len(bad):
                cell = np.argwhere(sel & (node_of == bad[0]))[0]
                out.append(("threshold", tuple(int(v) + (res.box[j].start or 0)
                                               for j, v in enumerate(cell)),
                            f"{why} at {T_row:g} C lies inside the trajectory"))
    # (b) Biot, with the radiative h at the cluster's CURRENT temperature
    nclu = len(res.clusters)
    shift = np.zeros(max(nclu, 1))
    if nclu:
        sel = res.cluster_id >= 0
        np.maximum.at(shift, res.cluster_id[sel],
                      np.abs(dT[node_of[sel]]) + np.abs(res.cell_T[sel] - res.tb))
    for ci, (cnt, V, A, k, eps, Tm, layers, place) in enumerate(res.clusters):
        if A <= 0.0:
            continue
        if shift[ci] <= BIOT_ACTIVE_K:
            continue                     # not thermally active: no lump error to fear
        h = H_CONV + 4.0 * eps * SIGMA * (Tm + K0) ** 3
        Bi = h * (V / A) / k
        if Bi > BIOT_MAX and layers < 3:
            out.append(("biot", place, f"Biot {Bi:.2f} with h = {h:.0f} W/m2K "
                                       f"and only {layers} layer(s) resolved"))
    return out


def _overlap(bx, sl):
    return all(bx[i] < (sl[i].stop if sl[i].stop is not None else 1 << 30)
               and bx[i + 3] > (sl[i].start or 0) for i in range(3))


def v2_o2_ok(w, box, o2_used_g):
    """V2 headroom: the grams of O2 the twins will use against the room's. 4
    sleepers use about 480 L in 8 h, 1.2 % of a 40 m3 room's volume: allowed.
    Ten times that is not."""
    vol_l = float(np.count_nonzero(w.mat[box] == AIR)) * w.vox_l
    used_l = o2_used_g / 1.429
    frac = used_l / max(vol_l, 1e-9)
    if frac > V2_O2_MAX:
        return False, f"O2 use is {100 * frac:.1f} % of the room's volume (limit {100 * V2_O2_MAX:.0f} %)"
    return True, ""


def o2_use_litres(persons_kg, hours, vo2_ml_kg_min=3.5):
    """Litres of O2 breathed by sleepers at 1 MET (3.5 ml/(kg min))."""
    return float(sum(persons_kg)) * vo2_ml_kg_min * 60.0 * hours / 1000.0


def collect_sources(w, plan, n=0):
    """([(cells, watts)], o2_grams) from the twins of the persons a jump will
    advance: each person's heat is spread over the flesh cells of its body."""
    sources, o2 = [], 0.0
    for p, twins in plan:
        for _name, tw, ctx in twins:
            pw = tw.power_w(w, p, ctx)
            if pw:
                flesh, sl = w._person_cells(p)
                if flesh is not None and flesh.any():
                    cells = np.argwhere(flesh) + np.array(
                        [sl[0].start or 0, sl[1].start or 0, sl[2].start or 0])
                    sources.append((cells, pw))
            o2 += tw.o2_use_g(w, p, ctx, n)
    return sources, o2


def region_for(w, sources=(), horizon_s=DEFAULT_HORIZON_S):
    """The box a jump owns: the hot window and the cells of every source, grown
    by the diffusion pad and to the closed air space they touch. None when there is nothing thermal at all."""
    pad = diffusion_pad(w, horizon_s)
    lo = np.full(3, 1 << 30)
    hi = np.full(3, -1)
    if bool(w.E.any()):
        win = w._window(w.E != 0, 0)
        lo = np.minimum(lo, [s.start for s in win])
        hi = np.maximum(hi, [s.stop - 1 for s in win])
    for cells, _watts in sources:
        c = np.asarray(cells, np.int64).reshape(-1, 3)
        lo = np.minimum(lo, c.min(axis=0))
        hi = np.maximum(hi, c.max(axis=0))
    if hi[0] < 0:
        return None
    box = tuple(slice(int(lo[i]), int(hi[i]) + 1) for i in range(3))
    return grow_to_enclosure(w, box, pad)


def check(w, sched=None, sources=(), horizon_s=DEFAULT_HORIZON_S, skip=None):
    """For quiet(tier=1): ((box,), [Blocker]). A net must be buildable and
    pass its refusals; the hot window it owns is then not a blocker."""
    from .quiet import Blocker
    region = region_for(w, sources, horizon_s)
    if region is None:
        return (), []
    res = thermal_net(w, region, sources=sources, horizon_s=horizon_s, skip=skip)
    ref = refusals(w, res, sched, trajectory=False)
    return (region,), [Blocker("thermal", place, why) for _c, place, why in ref]


def write_back(w, res, T1, sched=None, snap=True, tb=None, t_s=0.0):
    """Put the net's node temperatures into E, energy exact. The sub-node shape
    (a cell's offset from its node's mean) is kept for t_s = 0 and decays with
    the node's own internal time (B dx)^2/(pi^2 alpha) for a jump of t_s seconds:
    sum C (T_cell - T_node) is zero by construction, so the energy is exact for
    any decay, and a lump of a hot block and cold air does not stay lopsided. Returns a dict:
    expected (sum C dT), actual (sum of the change in E, float64), rounding,
    snapped (J removed by snapping, booked into shed), cells snapped."""
    sl = res.box
    tb = res.tb if tb is None else tb
    dT = np.asarray(T1, float) - res.T0
    dT[res.skip_node] = 0.0                      # a twin's bodies keep their energy to the bit
    node_of = res.node_of
    C = res.cell_C
    E = w.E[sl]
    e_old = E.astype(np.float64)
    if t_s > 0.0:
        tau = (res.B * res.dx) ** 2 / (np.pi ** 2 * np.maximum(res.alpha, 1e-12))
        keep = np.where(res.B > 1, np.exp(-t_s / tau), 1.0)
        shape = res.cell_T - res.T0[node_of]
        e_new = e_old + C * (dT[node_of] + (keep[node_of] - 1.0) * shape)
    else:
        e_new = e_old + C * dT[node_of]
    expected = float(np.dot(res.Cn, dT))
    snapped, nsnap = 0.0, 0
    if snap:
        Tc = AMBIENT + e_new / C
        eb = C * (tb - AMBIENT)
        try:
            Tss = steady(res, tb)
        except np.linalg.LinAlgError:
            Tss = np.full(len(dT), np.inf)
        heading = np.abs(Tss - tb)[node_of] < SETTLE_K
        cooling = ((Tc >= tb) & (dT[node_of] <= 0.0)) | ((Tc < tb) & (dT[node_of] >= 0.0))
        dust = np.abs(Tc - tb) < 0.1 * SETTLE_K           # rounding dust either side of the boundary
        m = ((np.abs(Tc - tb) < SETTLE_K) & cooling | dust) & heading & ~res.skip
        if m.any():
            snapped = float((e_new[m] - eb[m]).sum())
            e_new[m] = eb[m]
            nsnap = int(m.sum())
    before = float(e_old.sum())
    w.E[sl] = np.where(res.skip, e_old, e_new).astype(w.E.dtype)
    after = float(w.E[sl].astype(np.float64).sum())
    actual = after - before
    sk = getattr(w, "skin", None)
    if sk is not None and sk.shape == w.shape:
        sk[sl] = np.where(res.skip, sk[sl], 0.0)       # a twin's bodies keep their skin offsets
    rounding = actual - (expected - snapped)
    w.shed += snapped
    if sched is not None:
        sched.residual[("E",)] = sched.residual.get(("E",), 0.0) - rounding
    return {"expected": expected, "actual": actual, "rounding": rounding,
            "snapped": snapped, "cells_snapped": nsnap}


def advance_region(w, n, sched=None, sources=(), region=None, drives=None, o2_used_g=None,
                   max_nodes=2000, skip=None):
    """The whole tier-1 jump of the thermal region over n ticks from w.tick:
    build the net, check the refusals, advance it, write back, book the
    boundary flux into shed. Returns (ok, info). Does NOT move w.tick."""
    if region is None:
        region = region_for(w, sources, n * TICK_S)
    if region is None:
        return True, {"nodes": 0, "boundary_J": 0.0, "source_J": 0.0, "expected": 0.0,
                      "actual": 0.0, "rounding": 0.0, "snapped": 0.0, "cells_snapped": 0, "B": 0}
    res = thermal_net(w, region, sources=sources, max_nodes=max_nodes,
                      horizon_s=n * TICK_S, skip=skip)
    ref = refusals(w, res, sched, o2_used_g, trajectory=False)
    if ref:
        return False, {"refusals": ref}
    drv = dict(drives or {})
    drv.setdefault("Tb", F.ConstDrive(res.tb))
    T1, fl = F.rc_advance_arrays(res.spec, res.T0, drv, w.tick, w.tick + n)
    ref = refusals(w, res, sched, T1=T1, t_s=n * TICK_S)
    if ref:
        return False, {"refusals": ref}
    # The snap (a cell within SETTLE_K of the boundary joins it) needs the steady state of the whole net, a
    # nonlinear solve that was 0.6 s of a 1.8 s chunk. With a heat source in the net nothing is heading for the
    # boundary value, so there is nothing to snap: skip it (exact: no joule is moved; the books are unchanged).
    stats = write_back(w, res, T1, sched, t_s=n * TICK_S, snap=not bool(np.any(res.spec.q)))
    if n * TICK_S >= WIND_DECAY_S:
        # warm-air wind and pressure are functions of E and decay in seconds: after
        # a jump this long what they held is gone (THERMAL_DRIVEN, recomputed by
        # the first ticks). p_add is the one-tick injection.
        w.p_add[res.box] = 0.0
        if w.pcell is not None:
            w.pcell[...] = 0.0
            for v in w.vcell:
                v[...] = 0.0
    out = fl[("E", "boundary", "sink")]
    w.shed += out
    if sched is not None:
        sched.sinks[("E", "source", "twin")] = sched.sinks.get(("E", "source", "twin"), 0.0) \
            + fl[("E", "source", "twin")]
    stats.update({"boundary_J": out, "nodes": res.spec.n, "B": res.B,
                  "source_J": -fl[("E", "source", "twin")]})
    return True, stats
