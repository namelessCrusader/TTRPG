"""K. Habit: what this character tends to do here (learned-mind plan,
section 4). SLOW, and value-free: it follows what was CHOSEN, not what paid.

A strength h in 0..1 for each (place, act), where the act is reader.key
(verb, whom or what). At each choice, every act on the menu moves toward
whether it was the one chosen:  h += lr * (1[chosen] - h).  So a strength
grows with each repeat and fades each time the act is passed over (disuse).
Overnight every strength fades by `fade` (the act not done at all).
"""
from __future__ import annotations


class Habit:
    def __init__(self, lr=0.02, fade=0.98):
        self.lr, self.fade = lr, fade
        self.h: dict = {}

    def strength(self, place, act):
        return self.h.get((place, act), 0.0)

    def update(self, place, acts, chosen, scale=1.0):
        lr = self.lr * scale
        for a in set(acts):
            k = (place, a)
            x = self.h.get(k, 0.0)
            self.h[k] = x + lr * ((a == chosen) - x)

    def night(self):
        for k in list(self.h):
            self.h[k] *= self.fade
            if self.h[k] < 1e-4:
                del self.h[k]
