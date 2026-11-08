"""E. Episodic memory: what happened, where, with whom, and what it did to me
(learned-mind plan, section 4). ONE SHOT.

    store(act, z, value)   one act (reader.key: verb and whom/what), the
                           reader's numbers z at the choice, and how it went:
                           the TD error at the time (better or worse than
                           expected). Stored once; no averaging.
    lookup(act, Z)         for each row of Z, the mean value of the k nearest
                           stored episodes of the same act (Euclidean on z),
                           or 0 if none: a single betrayal by Fenn counts at
                           once the next time "give to Fenn" is on the menu.
    tonight                the night's full steps (menu, choice, rewards) for
                           REPLAY at the end of the night ("sleep") into A and K.

The store is a window: at most `cap` episodes per act, oldest dropped first.
"""
from __future__ import annotations

from collections import deque

import numpy as np


class Episodic:
    def __init__(self, k=5, cap=200):
        self.k, self.cap = k, cap
        self.mem: dict = {}            # act -> deque of (z, value)
        self._arr: dict = {}           # act -> (Z array, values) cache
        self.tonight: list = []
        self.size = 0

    def store(self, act, z, value):
        d = self.mem.get(act)
        if d is None:
            d = self.mem[act] = deque(maxlen=self.cap)
        if len(d) < self.cap:
            self.size += 1
        d.append((np.array(z), float(value)))
        self._arr.pop(act, None)

    def lookup(self, act, Z):
        """Z: rows of z for options that are all this act. Returns values."""
        d = self.mem.get(act)
        if not d:
            return np.zeros(len(Z))
        if act not in self._arr:
            self._arr[act] = (np.stack([z for z, _ in d]), np.array([v for _, v in d]))
        M, vals = self._arr[act]
        k = min(self.k, len(vals))
        dist = ((Z[:, None, :] - M[None, :, :]) ** 2).sum(-1)
        idx = np.argpartition(dist, k - 1, axis=1)[:, :k]
        return vals[idx].mean(1)
