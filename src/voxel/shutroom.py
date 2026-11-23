"""THE SHUT ROOM: one radiant-star law for a sealed room, read by the tick law and the night jump
(wave 15 batch 2, lane 2d, S4 item 1).

THE LAW (the room is one enclosure, the surfaces in it exchange through a radiant node of no capacity):

    T_env^4 = (1 - phi) * T_mr^4 + phi * T_inf^4          phi = share of a face's view that leaves the room
    T_mr^4  = sum_i(w_i T_i^4) / sum_i(w_i)                w_i = A * eps_i * af_i * n_in_i   (the pool)
    q_i     = sigma * dt * w_i * (T_i^4 - T_mr^4)          joules that LEAVE surface i; they sum to zero

phi is 0 for faces looking into a sealed room, 1 for a face looking at gas that is not in the room (the old
"radiate to the fixed 20 C world" law, which stays the law of the open world). A small convex body in a large
enclosure sees the enclosure, not itself (Incropera and DeWitt, Fundamentals of Heat and Mass Transfer, the
enclosure chapter: q = eps A sigma (T1^4 - T2^4)), so a body is NOT a member of the pool: it radiates against
T_mr and the joules it gives are shared out over the pool by w_i (the walls that receive them).

THE ROOM IS DETECTED, NEVER PASSED IN. `room_of(w)` takes the sink's box (a RegionSink), looks at the lattice and
calls the room sealed when no air cell of it has a face onto a gas cell outside it or onto the edge of the lattice.
A door or a window that is a gap in the wall is gas, so it makes the room not sealed and the old law stays. (A
window of glass is a solid in the lattice; its phi > 0 is data for later: see switches slow.shut_room_window.)

The tick law (bodyheat.BodyHeatMixin._env_radiation) and the night jump (slownet.thermal_net / advance_region,
bodyheat.bh_skip) both read this file: `star_loss`, `room_tmr`, `room_of`.

Marks: (u) unverified; (d) design choice.
"""
import numpy as np

from .materials import AIR, AMBIENT, FACE_CLIP, SIGMA, _EMISS_ARR, _FEMISS_ARR

K0 = 273.15
T_INF4 = (AMBIENT + K0) ** 4          # the open world's fixed environment, K^4


class Room:
    """What the lattice says about one sink box: plain arrays, never stored in a snapshot."""
    __slots__ = ("gen", "box", "air", "n_in", "sealed", "why", "lo", "hi")

    def __init__(self):
        self.gen = None
        self.box = None
        self.air = None                 # bool, the room's air cells (world shape)
        self.n_in = None                # int8, faces of each cell that look onto the room's air (world shape)
        self.sealed = False
        self.why = ""
        self.lo = self.hi = None        # bounding box of the room's air cells

    def contained_in(self, sl):
        """Are all the room's air cells inside the box `sl`?"""
        return bool(self.sealed and all((s.start or 0) <= self.lo[i] and self.hi[i] < s.stop
                                        for i, s in enumerate(sl)))


def build_room(w, box):
    """Detect the room of `box`: its air cells, the faces onto them, and whether it is sealed."""
    r = Room()
    r.box = box
    air = np.zeros(w.shape, bool)
    air[box] = (w.mat[box] == AIR)
    r.air = air
    if not air.any():
        r.why = "no air in the box"
        r.n_in = np.zeros(w.shape, np.int8)
        return r
    P = np.pad(w.mat == AIR, 1, constant_values=True)       # past the lattice edge is the open world: gas
    Pa = np.pad(air, 1, constant_values=False)
    n_in = np.zeros(w.shape, np.int8)
    leak = np.zeros(w.shape, bool)
    nx, ny, nz = w.shape
    for ax in range(3):
        for d in (-1, 1):
            sl = [slice(1, 1 + nx), slice(1, 1 + ny), slice(1, 1 + nz)]
            sl[ax] = slice(sl[ax].start + d, sl[ax].stop + d)
            sl = tuple(sl)
            n_in += Pa[sl]
            leak |= air & P[sl] & ~Pa[sl]
    r.n_in = n_in
    idx = np.argwhere(air)
    r.lo, r.hi = idx.min(axis=0), idx.max(axis=0)
    if leak.any():
        c = tuple(int(v) for v in np.argwhere(leak)[0])
        r.why = "open: gas leaves the room at %s" % (c,)
        return r
    r.sealed = True
    return r


def room_of(w):
    """The detected sealed room of the world's sink, or None. None when the world does not have the law
    (`shut_room_star` False), has no region sink, has a climate (the cold world has its own star), or the room
    is not sealed. Kept against the matter generation in `_bh_runtime` (never stored)."""
    if not getattr(w, "shut_room_star", False) or getattr(w, "climate", None) is not None:
        return None
    sink = getattr(w, "airsink", None)
    box = getattr(sink, "box", None)
    if box is None or not hasattr(sink, "air"):
        return None
    cache = w.__dict__.setdefault("_bh_runtime", {})
    gen = getattr(w, "_matter_gen", None)
    ent = cache.get("_room")
    if ent is None or ent.gen != gen or ent.box != box:
        ent = build_room(w, box)
        ent.gen = gen
        cache["_room"] = ent
    return ent if ent.sealed else None


def _weights(e, af, n_in, A):
    return A * np.asarray(e, np.float64) * np.asarray(af, np.float64) * n_in


def room_tmr(Ts, e, af, n_in, A, member=None):
    """T_mr in C and the pool weight: the T^4 mean of the surfaces that look into the room, `member` the cells
    that are in the pool (a body is not). Ts in C, all arrays one shape."""
    w = _weights(e, af, n_in, A)
    if member is not None:
        w = np.where(member, w, 0.0)
    W = float(w.sum())
    if W <= 0.0:
        return None, 0.0
    tk4 = (np.asarray(Ts, np.float64) + K0) ** 4
    return float((w * tk4).sum() / W) ** 0.25 - K0, W


def star_loss(Ts, e, af, n_in, n_gas, A, dt, Csur, body, dtype=np.float64, clip=FACE_CLIP):
    """Joules that LEAVE each surface cell in one tick under the star, and T_mr in C.

    `n_in` the faces looking into the room (never more than `n_gas`, the faces looking at gas); the rest of the
    gas faces keep the old law (radiate to the open world at 20 C). `body` marks cells that are not members of the
    pool: they radiate against T_mr and what they give is shared over the pool by its weights, so the sum over
    the whole array is zero for a room all of whose faces look inward (to float64 rounding; the clip is the
    only other thing that can move it, and it is returned in `resid`).
    -> (loss, tmr_C, resid): resid = sum(loss) in joules, which the caller books into shed with the clip's
    residue (zero when nothing is clipped)."""
    Ts = np.asarray(Ts, np.float64)
    n_in = np.minimum(np.asarray(n_in, np.int64), np.asarray(n_gas, np.int64))
    n_gh = np.asarray(n_gas, np.int64) - n_in
    e = np.asarray(e, np.float64)
    af = np.asarray(af, np.float64)
    w = _weights(e, af, n_in, A)
    pool = (w > 0.0) & ~body
    W = float(w[pool].sum())
    tk4 = (Ts + K0) ** 4
    loss = np.zeros(Ts.shape, np.float64)
    tmr = None
    if W > 0.0:
        t4p = tk4[pool]
        B4 = float((w[pool] * t4p).sum() / W)
        if t4p.max() == t4p.min():
            B4 = float(t4p[0])                 # a uniform room gives exactly 0, not rounding dust
        tmr = B4 ** 0.25 - K0
        q = SIGMA * dt * w * (tk4 - B4)
        lim = clip * np.asarray(Csur, np.float64) * np.abs(Ts - tmr)
        q = np.where(w > 0.0, np.clip(q, -lim, lim), 0.0)       # a cell never passes its room's temperature
        Qb = float(q[body].sum())
        loss += np.where(body, q, q - Qb * w / W)               # the pool receives what the bodies gave
    if (n_gh > 0).any():                        # faces onto gas that is not in the room: the open-world law
        g = e * SIGMA * A * af * n_gh * dt * (tk4 - T_INF4)
        lim = clip * np.asarray(Csur, np.float64) * np.abs(Ts - AMBIENT)
        loss += np.where(n_gh > 0, np.clip(g, -lim, lim), 0.0)
    return loss.astype(dtype), tmr, float(loss.sum())


def full_tmr(w, room, member_mask=None):
    """T_mr of the whole room now (a scan of its box and its surfaces), in C, or None. Used by the night jump
    (once per chunk) and by the body law before the tick law has run."""
    sl = tuple(slice(max(int(room.lo[i]) - 1, 0), min(int(room.hi[i]) + 2, w.shape[i])) for i in range(3))
    u = w._surfaces(sl)
    dx = 0.1 * w.scale
    eps = (w._emiss_table() if hasattr(w, "_emiss_table") else _EMISS_ARR)[u.mat]
    e = np.where(u.solid, eps, _FEMISS_ARR[u.fl])
    n_gas = u.n_gas
    n_in = np.minimum(room.n_in[sl], n_gas)
    rad_from = (u.solid | u.wet) & (n_gas > 0)
    member = rad_from if member_mask is None else (rad_from & ~member_mask[sl])
    t, _ = room_tmr(u.Ts, e, u.af, np.where(rad_from, n_in, 0), dx * dx, member)
    return t

