"""V. Valence: how good or bad each person, place and thing is, from what
followed it (Pavlovian; learned-mind plan, section 4). FAST.

The rule is Rescorla-Wagner, the delta rule over the cues that were present:
after a stretch of time with reward r (the drop in drive error), the cues c
present with salience s_c share one prediction, sum_c s_c v_c, and each moves
by  lr * s_c * (r - prediction).  A "ctx" cue is always present, so the
steady drift of the needs (hunger rising) lands on it and not on a person.

Cues the mind uses (lab/mindL/mind.py): the people who did something to me
in that stretch (salience 1), whom or what my own act was about and where I
was (0.5), and two event-type cues, "got_kind" and "got_harm" (a kind or a
harsh act aimed at me). The last two give the partner models (P) the worth
of an answer.
"""
from __future__ import annotations


class Valence:
    def __init__(self, lr=0.2):
        self.lr = lr
        self.v: dict = {}
        self.n: dict = {}

    def value(self, c):
        return self.v.get(c, 0.0)

    def update(self, cues, r):
        """cues: {cue: salience}. Returns the prediction error."""
        pred = sum(s * self.v.get(c, 0.0) for c, s in cues.items())
        d = r - pred
        for c, s in cues.items():
            self.v[c] = self.v.get(c, 0.0) + self.lr * s * d
            self.n[c] = self.n.get(c, 0) + 1
        return d
