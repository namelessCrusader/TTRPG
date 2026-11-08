"""C. Curiosity: where my models still learn (learned-mind plan, section 4).
MEDIUM speed.

Learning progress (Oudeyer and Kaplan): for each context, two running means
of the model's absolute prediction error, a fast one and a slow one. While the
model is learning there, the fast mean is below the slow one, and
    progress = max(0, slow - fast).
Where the error is flat (already learned, or never learnable: noise) progress
is ~0, so pure noise holds no lasting interest.

Contexts: the world model's verb (W) and the partner (P). An act's bonus is
the progress of its verb's world model plus that of its target's partner
model, lowered by habituation: divided by 1 + n / hab, where n counts how
often this act was chosen.
"""
from __future__ import annotations


class Progress:
    def __init__(self, fast=0.2, slow=0.02):
        self.a_f, self.a_s = fast, slow
        self.f: dict = {}
        self.s: dict = {}

    def observe(self, ctx, err):
        if ctx not in self.f:
            self.f[ctx] = self.s[ctx] = err
            return
        self.f[ctx] += self.a_f * (err - self.f[ctx])
        self.s[ctx] += self.a_s * (err - self.s[ctx])

    def progress(self, ctx):
        return max(0.0, self.s.get(ctx, 0.0) - self.f.get(ctx, 0.0))


class Curiosity:
    def __init__(self, hab=30.0):
        self.w, self.p = Progress(), Progress()
        self.hab = hab
        self.n: dict = {}

    def visit(self, act):
        self.n[act] = self.n.get(act, 0) + 1

    def bonus(self, verb, partner, act):
        lp = self.w.progress(verb) + (self.p.progress(partner) if partner else 0.0)
        return lp / (1.0 + self.n.get(act, 0) / self.hab)
