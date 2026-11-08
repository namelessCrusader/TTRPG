"""A. Action selection: actor-critic on the reward-prediction error, over the
menu features (learned-mind plan, section 4). MEDIUM speed.

    critic   V(s) = c . s, linear in the state (drives.state); TD(0) for a
             semi-Markov step: an act takes dt turns, so
             delta = r + gamma^dt * V(s') - V(s)   (V(s') = 0 at the end of
             the night or at death). Normalised LMS: c += lr_c delta s / |s|^2.
    actor    a score per option, linear in the reader's numbers crossed with
             the verb: W[verb] . z + U[verb, target]. The policy is a softmax
             over the menu of the WHOLE score (the mind adds its other terms);
             the actor's own part moves along grad log pi(a):
             W[v_a] += lr delta z_a, and every option b loses lr delta pi_b z_b.
"""
from __future__ import annotations

import numpy as np


class ActorCritic:
    def __init__(self, nverbs, zdim, ntargets, sdim, lr_actor=0.1, lr_critic=0.1, gamma=0.98):
        self.W = np.zeros((nverbs, zdim))
        self.U = np.zeros((nverbs, ntargets))
        self.c = np.zeros(sdim)
        self.lr_a, self.lr_c, self.gamma = lr_actor, lr_critic, gamma

    def scores(self, menu):
        return np.einsum("nz,nz->n", self.W[menu.v], menu.Z) + self.U[menu.v, menu.t]

    def value(self, s):
        return float(self.c @ s)

    def delta(self, r, s, s2, dt, terminal):
        nxt = 0.0 if terminal else self.gamma ** dt * self.value(s2)
        return r + nxt - self.value(s)

    def critic_update(self, s, delta, scale=1.0):
        self.c += scale * self.lr_c * delta * s / max(1e-9, s @ s)

    def actor_update(self, menu, a, pi, delta, scale=1.0):
        lr = scale * self.lr_a * delta
        np.add.at(self.W, menu.v, -lr * pi[:, None] * menu.Z)
        np.add.at(self.U, (menu.v, menu.t), -lr * pi)
        self.W[menu.v[a]] += lr * menu.Z[a]
        self.U[menu.v[a], menu.t[a]] += lr
