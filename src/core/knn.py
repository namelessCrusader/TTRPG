"""kNN-Prompting — the frozen 0.6B is a feature extractor; the DECISION is a
non-parametric vote over past (score-distribution → correct-choice) anchors.

The one small-model technique with published evidence at 0.8B, and it needs no
training: the datastore IS our accumulated (situation → what-was-right) log,
keyed by the model's own scored distribution. It self-improves as play grows,
and it can never hurt a cold start (empty memory → defer to the model). ~1 KB
of numpy-free math (KL divergence over softmaxed option scores).
"""

from __future__ import annotations

import json
import math


def _softmax(xs: list) -> list:
    m = max(xs)
    es = [math.exp(x - m) for x in xs]
    s = sum(es) or 1.0
    return [e / s for e in es]


def _kl(p: list, q: list) -> float:
    return sum(pi * math.log((pi + 1e-9) / (qi + 1e-9)) for pi, qi in zip(p, q))


class Datastore:
    """Anchors bucketed by (intent, option-tuple) so votes compare like with like.
    An anchor is (softmax-distribution-over-options, chosen-option-id)."""

    def __init__(self):
        self.buckets: dict = {}

    def _key(self, intent, options):
        return intent + "|" + ",".join(options)

    def add(self, intent, options, scores, chosen):
        self.buckets.setdefault(self._key(intent, tuple(options)), []).append(
            [_softmax(list(scores)), chosen])

    def anchors(self, intent, options):
        return self.buckets.get(self._key(intent, tuple(options)), [])

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.buckets, f)

    @classmethod
    def load(cls, path):
        ds = cls()
        with open(path) as f:
            ds.buckets = json.load(f)
        return ds


def vote(ds: Datastore, intent, options, scores, k: int = 8) -> dict:
    """KL-distance-weighted vote of the k nearest anchors over the option ids."""
    q = _softmax(list(scores))
    scored = sorted(((_kl(q, a[0]), a[1]) for a in ds.anchors(intent, options)),
                    key=lambda t: t[0])[:k]
    tally = {o: 0.0 for o in options}
    for dist, chosen in scored:
        if chosen in tally:
            tally[chosen] += 1.0 / (dist + 0.1)         # nearer anchors weigh more
    return tally


def blend_pick(ds: Datastore, intent, options, scores, model_scores: dict,
               alpha: float = 0.4, k: int = 8) -> str:
    """Final choice = α·(model preference) + (1−α)·(memory vote). With no anchors
    the memory term is flat, so it cleanly reduces to the model's own pick."""
    v = vote(ds, intent, options, scores, k)
    total = sum(v.values())
    mprob = _softmax([model_scores.get(o, -1e9) for o in options])
    best, bestval = options[0], -1e18
    for i, o in enumerate(options):
        memp = (v[o] / total) if total else (1.0 / len(options))
        blended = alpha * mprob[i] + (1 - alpha) * memp
        if blended > bestval:
            best, bestval = o, blended
    return best
