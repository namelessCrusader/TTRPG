"""MOMENTUM OF THE GAS: a flow that keeps going (wave 14a, lane D).

Until now a face of the gas flow kept its speed from tick to tick (bled off by
a drag) but the speed was never CARRIED by the flow itself. A plume that met a
ceiling lost its speed there and piled up, instead of turning the corner and
running along the ceiling the way Alpert's ceiling jet does. This is the missing
term of the momentum equation, du/dt = -(u . grad) u, done as SEMI-LAGRANGIAN
ADVECTION of the face speeds: the new speed on a face is the speed the gas had
where it came from, one tick ago (Stam 1999). Back-tracing never overshoots, so
it is stable for any speed.

Plain linear interpolation of that kind smears a jet a little each tick (the
numerical viscosity of the scheme is about u dx / 2). It is made second order
with the MacCormack error correction (Selle et al. 2008): trace forward, trace
back, add half the difference, then clamp to the range of the neighbours the
value came from so no new extreme appears. This is what keeps a jet running
along a ceiling for a metre instead of mixing out in a few cells.

Faces are the usual staggered (MAC) faces of gasflow.py: `u[a]` has one more
entry than there are cells along axis a, and sits at the middle of the cell
face. A face's other two velocity components are read by interpolating the
neighbouring faces of those axes. Everything is in cell widths (H) and seconds.

It is plain numpy (the grid is the coarse cell grid, 20 cm cells: tens of
thousands of numbers in a busy room, so the cost is small), only run over the
box where something moves, and never changes a world where `gas_momentum` is off.
"""
import numpy as np


def _grid(shape, comp):
    """Where each entry of a face array of axis `comp` sits, in cell widths from
    the first cell's low corner: faces on their own axis, cell middles on the other two."""
    out = []
    for a, n in enumerate(shape):
        v = np.arange(n, dtype=np.float32) + (0.0 if a == comp else 0.5)
        s = [1, 1, 1]
        s[a] = n
        out.append(v.reshape(s))
    return out


def _sample(arr, comp, q, want_range=False):
    """`arr`, a face array of axis `comp`, trilinearly sampled at positions `q`
    (three arrays, cell widths). Outside the array the nearest entry holds.
    With `want_range` also the smallest and largest of the 8 entries read."""
    shape = arr.shape
    idx0, fr = [], []
    for a in range(3):
        c = q[a] - (0.0 if a == comp else 0.5)
        c = np.clip(c, 0.0, shape[a] - 1.0)
        i0 = np.minimum(np.floor(c).astype(np.intp), max(shape[a] - 2, 0))
        idx0.append(i0)
        fr.append((c - i0).astype(np.float32))
    i, j, k = idx0
    i1 = np.minimum(i + 1, shape[0] - 1)
    j1 = np.minimum(j + 1, shape[1] - 1)
    k1 = np.minimum(k + 1, shape[2] - 1)
    c000, c100 = arr[i, j, k], arr[i1, j, k]
    c010, c110 = arr[i, j1, k], arr[i1, j1, k]
    c001, c101 = arr[i, j, k1], arr[i1, j, k1]
    c011, c111 = arr[i, j1, k1], arr[i1, j1, k1]
    fx, fy, fz = fr
    v = ((c000 * (1 - fx) + c100 * fx) * (1 - fy) + (c010 * (1 - fx) + c110 * fx) * fy) * (1 - fz) \
        + ((c001 * (1 - fx) + c101 * fx) * (1 - fy) + (c011 * (1 - fx) + c111 * fx) * fy) * fz
    if not want_range:
        return v
    lo = np.minimum.reduce([c000, c100, c010, c110, c001, c101, c011, c111])
    hi = np.maximum.reduce([c000, c100, c010, c110, c001, c101, c011, c111])
    return v, lo, hi


def moving_box(u, still, pad, ncells):
    """The cells (as slices) that hold every face faster than `still`, grown by
    `pad` cells, or None when nothing moves."""
    lo = [10 ** 9] * 3
    hi = [-1] * 3
    for c, a in enumerate(u):
        m = np.abs(a) > still
        if not m.any():
            continue
        for ax in range(3):
            line = np.flatnonzero(m.any(axis=tuple(b for b in range(3) if b != ax)))
            lo[ax] = min(lo[ax], int(line[0]))
            hi[ax] = max(hi[ax], int(line[-1]) + (0 if ax == c else 1))
    if hi[0] < 0:
        return None
    return tuple(slice(max(l - pad, 0), min(h + 1 + pad, n))
                 for l, h, n in zip(lo, hi, ncells))


def advect_faces(u, dt, H, still=1e-3, pad=2, correct=True):
    """The face speeds `u` (three float32 arrays, changed in place) carried by
    themselves for `dt` seconds over cells H metres wide. Returns the largest
    distance any face's gas came from, in cell widths (a Courant number), or
    0.0 when nothing moved."""
    ncells = tuple(a.shape[i] - 1 for i, a in enumerate(u))
    box = moving_box(u, still, pad, ncells)
    if box is None:
        return 0.0
    sub = []
    for c, a in enumerate(u):
        s = tuple(slice(b.start, b.stop + (1 if ax == c else 0)) for ax, b in enumerate(box))
        sub.append(np.ascontiguousarray(a[s]))
    k = np.float32(dt / H)
    out, cmax = [], 0.0
    for c in range(3):
        P = _grid(sub[c].shape, c)
        P = [np.broadcast_to(p, sub[c].shape) for p in P]
        V = [sub[c] if b == c else _sample(sub[b], b, P) for b in range(3)]
        Pd = [P[a] - k * V[a] for a in range(3)]
        cmax = max(cmax, float(max(np.abs(k * v).max() for v in V)))
        if correct:
            fwd, lo, hi = _sample(sub[c], c, Pd, want_range=True)
            Pb = [P[a] + k * V[a] for a in range(3)]
            back = _sample(fwd, c, Pb)
            new = fwd + np.float32(0.5) * (sub[c] - back)
            new = np.minimum(np.maximum(new, lo), hi)
        else:
            new = _sample(sub[c], c, Pd)
        out.append(new.astype(np.float32))
    for c, a in enumerate(u):
        s = tuple(slice(b.start, b.stop + (1 if ax == c else 0)) for ax, b in enumerate(box))
        a[s] = out[c]
    return cmax


def mix_faces(u, dt, H, alpha, still=1e-3, pad=2, kmax=0.1):
    """The mixing of momentum the cells cannot see (turbulent eddies the size of a cell
    and smaller): the face speeds spread to their neighbours with the same diffusivity the
    flow gives its heat and smoke, K = alpha * H * (local speed), alpha = GAS_ENTRAIN (the
    entrainment constant of a plume, Morton, Taylor and Turner 1956). Flux form, so the
    momentum is kept to rounding. It is what makes a jet or a plume draw in the still air
    at its sides and slow down as it widens. Changes `u` in place."""
    ncells = tuple(a.shape[i] - 1 for i, a in enumerate(u))
    box = moving_box(u, still, pad, ncells)
    if box is None or alpha <= 0.0:
        return
    sub = []
    for c, a in enumerate(u):
        s = tuple(slice(b.start, b.stop + (1 if ax == c else 0)) for ax, b in enumerate(box))
        sub.append(a[s].astype(np.float32))
    # the speed at the middle of every cell
    vc = [0.5 * (s_[tuple(slice(None, -1) if a == c else slice(None) for a in range(3))]
                 + s_[tuple(slice(1, None) if a == c else slice(None) for a in range(3))])
          for c, s_ in enumerate(sub)]
    spd = np.sqrt(vc[0] ** 2 + vc[1] ** 2 + vc[2] ** 2)
    coef = np.float32(dt / (H * H))
    for c in range(3):
        k = np.float32(alpha * H) * spd
        # a face's own diffusivity: the mean of the cells either side (the edge keeps its cell's)
        kf = np.empty(sub[c].shape, np.float32)
        sl_in = tuple(slice(1, -1) if a == c else slice(None) for a in range(3))
        kf[sl_in] = 0.5 * (k[tuple(slice(None, -1) if a == c else slice(None) for a in range(3))]
                           + k[tuple(slice(1, None) if a == c else slice(None) for a in range(3))])
        kf[tuple(slice(0, 1) if a == c else slice(None) for a in range(3))] = \
            k[tuple(slice(0, 1) if a == c else slice(None) for a in range(3))]
        kf[tuple(slice(-1, None) if a == c else slice(None) for a in range(3))] = \
            k[tuple(slice(-1, None) if a == c else slice(None) for a in range(3))]
        f = sub[c]
        for ax in range(3):
            lo = tuple(slice(None, -1) if a == ax else slice(None) for a in range(3))
            hi = tuple(slice(1, None) if a == ax else slice(None) for a in range(3))
            kk = np.minimum(0.5 * (kf[lo] + kf[hi]) * coef, np.float32(kmax))
            q = kk * (f[hi] - f[lo])
            f[lo] += q
            f[hi] -= q
    for c, a in enumerate(u):
        s = tuple(slice(b.start, b.stop + (1 if ax == c else 0)) for ax, b in enumerate(box))
        a[s] = sub[c]
