"""W. World model: what an act does to my drives, one step ahead
(learned-mind plan, section 4). MEDIUM speed.

For each verb a linear map from the reader's numbers z to the change of the
seven needs over the act (from the choice to the next choice):
    predicted d_need = M[verb] @ z
learned by the normalised delta rule  M[v] += lr * outer(err, z) / |z|^2.
The mind's lookahead is the predicted DROP of drive error,
    -(d_need_pred . drives.grad(kind, need)),
which is how homeostasis biases the act. Each update's error is kept per verb
for curiosity (C).
"""
from __future__ import annotations

import numpy as np


class WorldModel:
    def __init__(self, nverbs, ndrives, zdim, lr=0.2):
        self.M = np.zeros((nverbs, ndrives, zdim))
        self.lr = lr

    def predict(self, menu):
        return np.einsum("ndz,nz->nd", self.M[menu.v], menu.Z)

    def lookahead(self, menu, g):
        """The predicted drop of drive error for each option; g = drives.grad."""
        return -(self.predict(menu) @ g)

    def update(self, v, z, d_need):
        """Learn one act's outcome. Returns the mean absolute error before."""
        err = d_need - self.M[v] @ z
        self.M[v] += self.lr * np.outer(err, z) / max(1e-9, z @ z)
        return float(np.abs(err).mean())


class WorldModelP(WorldModel):
    """drives2 version: the same delta rule, plus (mindL6)
      - a per-verb experience count `cnt` and prior weight `n0`: a prior (set_prior*) is the initial matrix with n0 pseudo-observations, and the
        step of verb v is  lr * (cnt + 1) / (cnt + 1 + n0)  (n0 = 0: exactly the old rule), so the prior's weight after n experiences is n0 / (n0 + n):
        experience overrides it. This is the channel through which being TOLD writes (own-act priors);
      - update(scale=, mask=): a weaker update (what is learned by WATCHING counts w_obs of an own experience) on the needs seen only (mask).
    """

    def __init__(self, nverbs, ndrives, zdim, lr=0.2):
        super().__init__(nverbs, ndrives, zdim, lr)
        self.cnt = np.zeros(nverbs)
        self.n0 = np.zeros(nverbs)

    def set_prior_matrix(self, v, P, n0):
        """Blend a prior matrix (drives x z) into verb v as n0 pseudo-observations."""
        w0 = self.cnt[v] + self.n0[v]
        self.M[v] = (w0 * self.M[v] + n0 * P) / (w0 + n0) if w0 > 0 else np.array(P, float)
        self.n0[v] += n0

    def set_prior(self, v, const, slope, n0, nk, nd_cols=None):
        """Prior for verb index v: expected change of drive d = const[d] + slope[d] * need[d]. const/slope: dicts {drive index: number};
        nk: the first need column of z is 1 + nk (z = [1, features, needs])."""
        P = np.zeros_like(self.M[v])
        for d, c in (const or {}).items():
            P[d, 0] = c
        for d, s in (slope or {}).items():
            P[d, 1 + nk + d] = s
        self.set_prior_matrix(v, P, n0)

    def update(self, v, z, d_need, scale=1.0, mask=None):
        err = d_need - self.M[v] @ z
        if mask is not None:
            err = err * mask
        step = self.lr * scale
        if self.n0[v] > 0:
            step *= (self.cnt[v] + 1.0) / (self.cnt[v] + 1.0 + self.n0[v])
        self.M[v] += step * np.outer(err, z) / max(1e-9, z @ z)
        self.cnt[v] += scale
        return float(np.abs(err).mean())
