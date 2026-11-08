"""P. Partner models, ONE PER PERSON: how THIS person answers my acts toward
them (learned-mind plan, section 4). FAST.

Online logistic regression per (partner, act kind), where the act kind is my
verb. Two outputs: does the partner answer with a KIND act aimed at me, and
with a HARSH one (the world's own KIND_ACTS and HARSH_ACTS, plus a theft from
me), within WINDOW turns of my act. The inputs x are the few option numbers
that say how things stand with that person: [1, like, dislike, anger, threat,
gain, need].

    probe(partner, verb, x, t)   I did `verb` to `partner`: open a window
    observe(events)              acts aimed at me that I perceived
    close_due(t) / close_all()   windows that ended: learn, w += lr (y - p) x
    predict(partner, verb, x)    (p_kind, p_harm)

Each partner and verb has its own weights, so what Fenn does teaches nothing
about Oren (the L2 test needs this). An answer counts for every window open
on that partner at that moment.

ADAPTING TO CHANGE (2026-10-29; both OFF by default, so the first round can
be redone). A model that has seen 100 payments is so sure that the first
betrayals barely move it (the first round: the harm chance moved by +0.02 after
a betrayal). Two cures, each a setting:

    forget   recency weighting: before each update the weights shrink by
             this share, so old evidence counts for less and the weights stay
             small enough to turn round.
    detect   change detection: the last DETECT_K residuals (the answer seen
             minus the answer predicted) of one (partner, verb) are kept; when
             their mean on one channel is DETECT_THR or more in size, the old
             weights are cut to KEEP of their size (the partner has changed:
             most of the old evidence is stale) and the partner's person-level
             rates start over from what was just seen.

    trust(partner, vk, vh)   a person-level reading, recency weighted: the
             partner's recent rate of kind and harsh answers (an EWMA over every
             window that ended, whatever the act) times the worth of such an
             answer. The mind uses it as the valence of the person.
"""
from __future__ import annotations

import math

import numpy as np

from lab.world2.world import HARSH_ACTS, KIND_ACTS

KIND = set(KIND_ACTS)
HARM = set(HARSH_ACTS) | {"steal"}
X_KEYS = ("like", "dislike", "anger", "threat", "gain", "need")
WINDOW = 30
DETECT_K, DETECT_THR, KEEP = 3, 0.6, 0.25
RATE_ALPHA = 0.25          # the person-level rates: the share the newest window takes


def sig(x):
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


def features(f):
    return np.array([1.0] + [float(f.get(k, 0.0)) for k in X_KEYS])


class PartnerModels:
    def __init__(self, lr=0.3, window=WINDOW, forget=0.0, detect=False):
        self.lr, self.window, self.forget, self.detect = lr, window, forget, detect
        self.w: dict = {}          # (partner, verb) -> 2 x len(x) weights
        self.open: list = []       # [partner, verb, x, t0, got_kind, got_harm]
        self.n: dict = {}
        self.res: dict = {}        # (partner, verb) -> the last residuals [(kind, harm)]
        self.rate: dict = {}       # partner -> [kind rate, harm rate] (recency weighted)
        self.changes: dict = {}    # partner -> how many changes were detected

    def _w(self, partner, verb):
        k = (partner, verb)
        if k not in self.w:
            self.w[k] = np.zeros((2, 1 + len(X_KEYS)))
        return self.w[k]

    def predict(self, partner, verb, x):
        if (partner, verb) not in self.w:
            return 0.5, 0.5
        w = self.w[(partner, verb)]
        return sig(w[0] @ x), sig(w[1] @ x)

    def probe(self, partner, verb, x, t):
        self.open.append([partner, verb, x, t, 0.0, 0.0])

    def observe(self, events):
        for e in events:
            for pr in self.open:
                if pr[0] == e["actor"] and e["t"] >= pr[3]:
                    if e["action"] in KIND:
                        pr[4] = 1.0
                    if e["action"] in HARM:
                        pr[5] = 1.0

    def trust(self, partner, vk, vh):
        """The recent worth of this person's treatment of me: its kind rate
        times the worth of a kind answer plus its harsh rate times the worth of a
        harsh one (0 until a window has ended on it)."""
        r = self.rate.get(partner)
        return 0.0 if r is None else r[0] * vk + r[1] * vh

    def _learn(self, pr):
        partner, verb, x, _, yk, yh = pr
        w = self._w(partner, verb)
        pk, ph = sig(w[0] @ x), sig(w[1] @ x)
        if self.detect:
            r = self.res.setdefault((partner, verb), [])
            r.append((yk - pk, yh - ph))
            del r[:-DETECT_K]
            if len(r) == DETECT_K and (abs(sum(a for a, _ in r)) / DETECT_K >= DETECT_THR
                                       or abs(sum(b for _, b in r)) / DETECT_K >= DETECT_THR):
                w *= KEEP                                # the partner has changed: the old evidence is stale
                self.res[(partner, verb)] = []
                self.changes[partner] = self.changes.get(partner, 0) + 1
                self.rate[partner] = [yk, yh]
                pk, ph = sig(w[0] @ x), sig(w[1] @ x)
        r = self.rate.get(partner)
        if r is None:
            self.rate[partner] = [yk, yh]
        else:
            r[0] += RATE_ALPHA * (yk - r[0])
            r[1] += RATE_ALPHA * (yh - r[1])
        if self.forget:
            w *= 1.0 - self.forget
        w[0] += self.lr * (yk - pk) * x
        w[1] += self.lr * (yh - ph) * x
        self.n[(partner, verb)] = self.n.get((partner, verb), 0) + 1
        return partner, (abs(yk - pk) + abs(yh - ph)) / 2

    def close_due(self, t):
        """Learn from the windows that have ended. Returns [(partner, error)]."""
        done = [pr for pr in self.open if t - pr[3] >= self.window]
        self.open = [pr for pr in self.open if t - pr[3] < self.window]
        return [self._learn(pr) for pr in done]

    def close_all(self):
        done, self.open = self.open, []
        return [self._learn(pr) for pr in done]
