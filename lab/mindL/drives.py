"""H. Homeostasis (learned-mind plan, section 5). Written, never learned.

The world keeps each drive's need (lab/world2/world.py, World.drives); the
type's NATURE gives each drive a weight and a set point. Here:

    needs(w, p)     the seven needs, as a vector in DRIVES order
    error(w, p)     the weighted error, sum of w_d * max(0, need_d - set_d)
    reward(e0, e1)  the homeostatic reward: the DROP in that error, e0 - e1
                    (Keramati and Gutkin). No designer reward anywhere.
    grad(kind, need)  d error / d need: w_d where the need is above its set
                    point, else 0. A predicted change of needs, times this,
                    is a predicted change of error: this is how H biases the
                    act (through the world model's one-step lookahead).
    state(w, p)     what the critic reads: needs, weighted errors, the room,
                    company, danger, what is held, the hour.

With discounting, the sum of discounted drops is (up to a constant and a
scale) the discounted sum of the errors themselves, so a death costs the
drop to the dead state, which then lasts: no special death penalty.
"""
from __future__ import annotations

import numpy as np

from lab.world2.world import DRIVES, NATURE, ROOMS

ROOMLIST = tuple(ROOMS)
TURNS = 300
SDIM = 1 + 2 * len(DRIVES) + len(ROOMLIST) + 4


def needs(w, p):
    d = w.drives(p)
    return np.array([d[k] for k in DRIVES])


def weights(kind):
    return np.array([NATURE[kind][d][0] for d in DRIVES])


def setpoints(kind):
    return np.array([NATURE[kind][d][1] for d in DRIVES])


def errors(kind, need):
    """Each drive's weighted error."""
    return weights(kind) * np.maximum(0.0, need - setpoints(kind))


def error(w, p):
    return w.drive_error(p)


def reward(e0, e1):
    return e0 - e1


def grad(kind, need):
    return weights(kind) * (need > setpoints(kind))


def state(w, p, need=None):
    need = needs(w, p) if need is None else need
    s = np.zeros(SDIM)
    s[0] = 1.0
    nd = len(DRIVES)
    s[1:1 + nd] = need
    s[1 + nd:1 + 2 * nd] = errors(p.kind, need)
    if p.room in ROOMLIST:
        s[1 + 2 * nd + ROOMLIST.index(p.room)] = 1.0
    b = 1 + 2 * nd + len(ROOMLIST)
    s[b] = min(4, len(w.here(p))) / 4
    s[b + 1] = w.danger(p)
    s[b + 2] = min(1.0, w.wealth(p) / 10)
    s[b + 3] = min(1.0, w.t / TURNS)
    return s
