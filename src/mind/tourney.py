"""Menus batched, then merged (wave 14b, lane F): a TOURNAMENT of small picks.

The user's rule: the size of a menu is whatever the decision models can do
meaningfully, then MERGE-SORT-LIKE. Pick inside small batches, then bring the
winners together and pick again, until one is left.

    merge(keys, size, pick)   the pure merge. Order does not matter: the
                              options are put in canonical order (their
                              keys) before they are cut into batches, so the
                              same options in any order give the same
                              batches, the same questions and the same pick.
    Tourney(chooser, size)    a chooser that wears any chooser of choosers.py
                              and answers a long request as a tournament.

A request of SIZE options or fewer goes straight to the chooser, untouched:
the same request, the same Decision, bit for bit.

The batch size is data (tools/menu_batch_measure.py, the lane F report):
  - a model reading a menu in one pass is best at 4 options (Laya: 70% at 4,
    57% at 16 pairwise, 7% at 128; docs/large-action-spaces-review.md, a
    measurement of the lab's own);
  - the rules and the voxel reflex table are exact at any size (a max merges
    exactly), so they are not what sets it.

PROBABILITIES. A Decision of a tournament gives one probability for each
option, like any other: P(i) = P(i wins its batch) x P(that batch's winner
wins the rest), the rest taken on the winners that were drawn. The numbers
add to 1. (u: it is the draw of the path that happened, as scope.joint does
for the two-level menu, not the exact chance over every possible path.)
The average character's probabilities are made the same way.

NPC COST. A tournament of a model-picked character over n options asks
about n/(size-1) batches. A rules chooser is no model call. The mind counts
every request that goes to the model (loop._Metered), so reactive mode stays
at 0.
"""
from __future__ import annotations

import dataclasses
from math import ceil

from .api import Decision


def chunks(order, size):
    """`order` cut into the fewest batches of `size` or fewer, as even as it
    goes, in order (so a batch of 9 and a size of 4 make 3, 3 and 3, never a
    lone last one)."""
    k = ceil(len(order) / size)
    q, r = divmod(len(order), k)
    out, at = [], 0
    for j in range(k):
        n = q + (1 if j < r else 0)
        out.append(order[at:at + n])
        at += n
    return out


def merge(keys, size, pick):
    """The tournament. `keys[i]` is the sort key of option i (its canonical
    place). `pick(members, rnd)` answers for ONE batch: `members` are option
    indices in canonical order, `rnd` the round (0 is the first), and it gives
    one of the members, or None for "none of these wins". Returns the winner
    (an option index) or None.

    The merge is one pass of pick over batches of `size` or fewer, then the
    same over the winners, until the winners fit one batch. A set of `size`
    or fewer is one pick, no more."""
    if size < 2:
        raise ValueError("a batch has room for two options at least")
    place = lambda i: (keys[i], i)
    cur = sorted(range(len(keys)), key=place)
    rnd = 0
    while len(cur) > size:
        cur = sorted((w for w in (pick(c, rnd) for c in chunks(cur, size)) if w is not None),
                     key=place)
        rnd += 1
    return pick(cur, rnd) if cur else None


def joint(n, records):
    """One probability for each of the n options, from the picks of a
    tournament. `records` is a list of (rnd, members, probs, winner), in the
    order they were asked."""
    last = max(r[0] for r in records)
    value = {}                                     # option -> its value in the round after
    for rnd in range(last, -1, -1):
        here = {}
        for r, members, probs, win in records:
            if r != rnd:
                continue
            for i, p in zip(members, probs):
                here[i] = p if rnd == last else p * value[win]
        value = here
    return tuple(value.get(i, 0.0) for i in range(n))


class Tourney:
    """A chooser that answers a long request by a tournament of the wrapped
    chooser. It has the interface of every chooser (name, remote, submit,
    poll, average). A request with `size` options or fewer is passed on as it
    is. The wrapped chooser's name stays, so a decision row names who picked."""

    def __init__(self, chooser, size):
        if size < 2:
            raise ValueError("a batch has room for two options at least")
        self.chooser, self.size = chooser, size
        self.name = chooser.name
        self.remote = getattr(chooser, "remote", False)
        self._done = {}
        self._avg = {}

    def submit(self, req):
        n = len(req.options)
        if n <= self.size:
            self.chooser.submit(req)
            self._done[req.id] = None              # the wrapped chooser holds it
            return
        key = [(req.groups[i], req.options[i], req.prior[i], req.prior_avg[i], req.habit[i])
               for i in range(n)]
        recs, avgs, fell = [], [], []
        count = [0]

        def pick(members, rnd):
            if len(members) == 1:                  # a lone one goes on, nobody is asked
                recs.append((rnd, members, (1.0,), members[0]))
                avgs.append((1.0,))
                return members[0]
            sub = dataclasses.replace(
                req, id=f"{req.id}:t{count[0]}",
                options=tuple(req.options[i] for i in members),
                groups=tuple(req.groups[i] for i in members),
                prior=tuple(req.prior[i] for i in members),
                prior_avg=tuple(req.prior_avg[i] for i in members),
                habit=tuple(req.habit[i] for i in members))
            count[0] += 1
            self.chooser.submit(sub)
            d = self.chooser.poll(sub.id, block=True)
            if d is None or d.request != sub.id or not 0 <= d.pick < len(members):
                raise RuntimeError(f"{self.name} gave no decision for request {sub.id}")
            m = len(members)
            probs = tuple((1.0 if j == d.pick else 0.0) for j in range(m)) \
                if d.probs is None else tuple(d.probs)
            recs.append((rnd, members, probs, members[d.pick]))
            avgs.append(self.chooser.average(sub))
            fell.append(d.fallback)
            return members[d.pick]

        win = merge(key, self.size, pick)
        self._done[req.id] = Decision(req.id, win, self.name, joint(n, recs), fallback=any(fell))
        if all(a is not None for a in avgs):
            self._avg[req.id] = joint(n, [(r[0], r[1], a, r[3]) for r, a in zip(recs, avgs)])

    def poll(self, req_id, block=True):
        if req_id not in self._done:
            return None
        d = self._done.pop(req_id)
        return self.chooser.poll(req_id, block) if d is None else d

    def average(self, req):
        if len(req.options) <= self.size:
            return self.chooser.average(req)
        return self._avg.pop(req.id, None)
