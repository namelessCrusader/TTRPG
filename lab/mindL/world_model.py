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
