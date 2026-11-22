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

TRACES (option, off by default; 2026-11-20, library need 2): with `lam` > 0 the
critic is TRUE ONLINE TD(lambda) with dutch traces (van Seijen et al., arxiv:1512.04087),
and the actor's table U[verb, target] moves along an eligibility trace of grad log pi
instead of the step's own gradient only: e_U = (gamma lam)^dt e_U + grad log pi(a),
U += lr delta e_U. One global lambda, per turn (an act of dt turns decays the trace by
(gamma lam)^dt). W stays one-step. The trace and the critic's value carry across the
night's end; only a death resets them.
"""
from __future__ import annotations

import numpy as np


class ActorCritic:
    def __init__(self, nverbs, zdim, ntargets, sdim, lr_actor=0.1, lr_critic=0.1, gamma=0.98):
        self.W = np.zeros((nverbs, zdim))
        self.U = np.zeros((nverbs, ntargets))
        self.c = np.zeros(sdim)
        self.lr_a, self.lr_c, self.gamma = lr_actor, lr_critic, gamma
        self.lam = 0.0
        self.reset_traces()

    def reset_traces(self):
        self.e_c = np.zeros_like(self.c)          # the critic's dutch trace
        self.e_U = np.zeros_like(self.U)          # the actor's trace on U[verb, target]
        self.v_old = None                         # the value of the next state as seen before the last update
        self.s_next = None

    def scores(self, menu):
        return np.einsum("nz,nz->n", self.W[menu.v], menu.Z) + self.U[menu.v, menu.t]

    def value(self, s):
        return float(self.c @ s)

    def delta(self, r, s, s2, dt, terminal):
        nxt = 0.0 if terminal else self.gamma ** dt * self.value(s2)
        return r + nxt - self.value(s)

    def critic_update(self, s, delta, scale=1.0):
        self.c += scale * self.lr_c * delta * s / max(1e-9, s @ s)

    def actor_update(self, menu, a, pi, delta, scale=1.0, tables="WU"):
        lr = scale * self.lr_a * delta
        if "W" in tables:
            np.add.at(self.W, menu.v, -lr * pi[:, None] * menu.Z)
            self.W[menu.v[a]] += lr * menu.Z[a]
        if "U" in tables:
            np.add.at(self.U, (menu.v, menu.t), -lr * pi)
            self.U[menu.v[a], menu.t[a]] += lr

    # ---------------------------------------------------------------- traces
    def trace_step(self, menu, a, pi, s, s2, r, dt, terminal, lam):
        """One step with traces. Returns the TD error delta (with the weights as they were).
        Critic: true online TD(lambda), alpha = lr_c / |s|^2 (the same normalised step as TD(0)).
        Actor: U moves by lr_a * delta * e_U. A terminal step ends the trace."""
        g = self.gamma ** dt
        gl = g * lam ** dt
        V = self.value(s)
        V2 = 0.0 if terminal else self.value(s2)
        delta = r + g * V2 - V
        alpha = self.lr_c / max(1e-9, float(s @ s))
        # the old value of s: from the last step's look ahead, if that was this very state (not across a night)
        v_old = self.v_old if (self.s_next is not None and np.array_equal(self.s_next, s)) else V
        self.e_c = gl * self.e_c + (1.0 - alpha * gl * float(self.e_c @ s)) * s
        self.c += alpha * (delta + V - v_old) * self.e_c - alpha * (V - v_old) * s
        # actor
        grad = np.zeros_like(self.U)
        np.add.at(grad, (menu.v, menu.t), -pi)
        grad[menu.v[a], menu.t[a]] += 1.0
        self.e_U = gl * self.e_U + grad
        self.U += self.lr_a * delta * self.e_U
        if terminal:
            self.reset_traces()
        else:
            self.v_old, self.s_next = V2, s2
        return delta
