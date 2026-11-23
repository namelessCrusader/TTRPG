"""H2. Homeostasis, second version (mindL6 stage A, 2026-11-23; behind the mind option `drives2`, off by default: every older run is unchanged).

WHY (mindL5 stage B): 184 of 210 learned people died of thirst in the lethal family inn. The first drive vector folded thirst into one
"energy" need, energy = max(hunger, thirst), so a drink earned NOTHING whenever hunger was at least thirst; the error was linear (no urgency);
and the body's reserve, the thing that runs out, was invisible. This module is what EVOLUTION gives a mind (the owner's decision 2026-11-23):
separate physical needs that drive, relief worth more at greater deficit, and a body that keeps saying so as it fails. No rule about acts.

THE LETTERS of the option (drives2 = True means "SCFD"):
    S  separate drives: thirst is its own drive beside hunger (energy = hunger only). Always on with drives2.
    C  convex homeostatic drive (Keramati and Gutkin 2014, eLife, doi:10.7554/eLife.04811):
           D(H) = ( sum_i |h*_i - h_i|^n )^(1/m),  n > m > 1,   reward = D(t) - D(t+1)
       here the deficit of a drive is its weighted error e_i = w_i max(0, need_i - set_i) (the old weights and set points), so a weight
       is the drive's scale. n = 4, m = 3 (u: from memory of the paper's simulations, NOT confirmed against the text: the page would not
       load; the condition n > m > 1 holds). Two consequences, tested: relief is worth more at a greater deficit (ALLIESTHESIA: the same
       relief feels better the more one needs it, Cabanac 1971), and when two needs compete the greater deficit wins the gradient.
       Without C the error is the old linear weighted sum (the ablation).
    F  felt depletion (interoception): the body's reserve reaches the mind. With a need at its cap the reserve drains (world: lethal);
       dep = 1 - reserve / full reserve, 0..1. The felt need keeps growing past the cap:  felt = need + FELT_GAIN * dep  (so 0..2), in the state's
       needs (hence in the drive: the error and the reward read it) and as two extra state channels (dep_thirst, dep_hunger). It reads only
       the body (p.reserve), says nothing about what to do. FELT_GAIN = 1 (u) and the linear time course (u): until the library answers
       (dehydration symptoms by percent of body water lost; request 2026-11-23 D-b). Real anchors from memory (u): thirst by 1-2 percent of body water
       lost, performance and cognition drop by about 2-5, collapse near 10, death near 15-20 (Adolph 1947; Sawka et al. 2007).
    D  need by destination (mind.py): a walk option carries the relief the mind's OWN world model expects from what it knows is on offer at the
       destination, for its CURRENT needs (a one-step lookahead over known affordances; every need, no thirst rule).
    W  learning by watching (mind.py / world_model.py): see another person's act and the change in their signs.
    (priors, channel 1: the option wm_prior; being told: drives2.told.)

THE CONSTANTS. Thirst takes the SAME weight and set point as hunger for each kind (u: no source for a difference; the two needs are
alike in what regulation means, they differ in the body's reserve: 5 days against 45, in the world). Drive vector order: the old seven, then thirst.
"""
from __future__ import annotations

import numpy as np

from lab.world2.world import DRIVES, LETHAL_RESERVE, NATURE, ROOMS

DRIVES2 = tuple(DRIVES) + ("thirst",)
ND2 = len(DRIVES2)
IE, IT = DRIVES2.index("energy"), DRIVES2.index("thirst")
ROOMLIST = tuple(ROOMS)
TURNS = 300
N_EXP, M_EXP = 4.0, 3.0            # Keramati and Gutkin 2014: n > m > 1 (values u, from memory)
FELT_GAIN = 1.0                    # (u) the felt need grows by this much over the whole reserve
DEFAULT_LETTERS = "SCFD"


def _nat(kind):
    return NATURE.get(kind) or NATURE["peasant"]


def weights(kind):
    n = _nat(kind)
    return np.array([n[d][0] for d in DRIVES] + [n["energy"][0]])


def setpoints(kind):
    n = _nat(kind)
    return np.array([n[d][1] for d in DRIVES] + [n["energy"][1]])


class Drives2:
    """Same interface as the module lab/mindL/drives.py (needs, error, state, errors, SDIM, DRIVES) plus drop() and batch lookahead."""

    def __init__(self, letters=DEFAULT_LETTERS, n=N_EXP, m=M_EXP, felt_gain=FELT_GAIN):
        self.letters = set(letters)
        self.convex, self.felt = "C" in self.letters, "F" in self.letters
        self.n, self.m, self.felt_gain = n, m, felt_gain
        assert not self.convex or n > m > 1, "Keramati and Gutkin: n > m > 1"
        self.DRIVES, self.ND = DRIVES2, ND2
        self.SDIM = 1 + 2 * ND2 + len(ROOMLIST) + 4 + 2
        self.cap = np.ones(ND2)
        if self.felt:
            self.cap[[IE, IT]] = 1.0 + felt_gain

    # ---------------------------------------------------------------- the body
    @staticmethod
    def depletion(p):
        """(thirst, hunger) fraction of the reserve lost, 0..1: the body's own signal (interoception)."""
        r = getattr(p, "reserve", None) or {}
        return (min(1.0, max(0.0, 1.0 - r.get(k, LETHAL_RESERVE[k]) / LETHAL_RESERVE[k])) for k in ("thirst", "hunger"))

    def needs(self, w, p):
        d = w.drives(p)
        v = np.array([d[k] for k in DRIVES] + [p.thirst])
        v[IE] = p.hunger
        if self.felt:
            dt, dh = self.depletion(p)
            v[IT] += self.felt_gain * dt
            v[IE] += self.felt_gain * dh
        return v

    # ---------------------------------------------------------------- the drive
    def errors(self, kind, need):
        """Each drive's weighted deficit e_i = w_i max(0, need_i - set_i)."""
        return weights(kind) * np.maximum(0.0, np.asarray(need) - setpoints(kind))

    def D(self, kind, need):
        """The drive: convex (Keramati and Gutkin) or the linear weighted sum. `need` may be a vector or rows (axis -1)."""
        e = self.errors(kind, need)
        if not self.convex:
            return e.sum(axis=-1)
        return (e ** self.n).sum(axis=-1) ** (1.0 / self.m)

    def error(self, w, p):
        return float(self.D(p.kind, self.needs(w, p)))

    @staticmethod
    def reward(e0, e1):
        return e0 - e1

    def drop(self, kind, need, delta):
        """Drop of the drive if the needs change by `delta` (a vector, or rows): D(now) - D(after), needs kept inside 0..cap."""
        need = np.asarray(need)
        after = np.clip(need + np.asarray(delta), 0.0, self.cap)
        return self.D(kind, need) - self.D(kind, after)

    def grad(self, kind, need):
        """d D / d need (for inspection and the tests; the mind uses drop(), exact, since D is convex and a gradient overstates a big relief)."""
        e = self.errors(kind, need)
        w = weights(kind) * (np.asarray(need) > setpoints(kind))
        if not self.convex:
            return w
        S = float((e ** self.n).sum())
        if S <= 0:
            return np.zeros(ND2)
        return (self.n / self.m) * S ** (1.0 / self.m - 1.0) * e ** (self.n - 1.0) * w

    def state(self, w, p, need=None):
        need = self.needs(w, p) if need is None else need
        s = np.zeros(self.SDIM)
        s[0] = 1.0
        s[1:1 + ND2] = need
        s[1 + ND2:1 + 2 * ND2] = self.errors(p.kind, need)
        if p.room in ROOMLIST:
            s[1 + 2 * ND2 + ROOMLIST.index(p.room)] = 1.0
        b = 1 + 2 * ND2 + len(ROOMLIST)
        s[b] = min(4, len(w.here(p))) / 4
        s[b + 1] = w.danger(p)
        s[b + 2] = min(1.0, w.wealth(p) / 10)
        s[b + 3] = min(1.0, w.t / TURNS)
        if self.felt:
            s[b + 4], s[b + 5] = self.depletion(p)
        return s


# ------------------------------------------------------------------------------------------------ cultural channel 1: being told
T_CHANCE = 0.6      # (u) a told fact is taken up with this chance (one's own experience: 1)
N0_TOLD = 3.0       # (u) pseudo-count of a told fact (the weight of n0 experiences; one's own experience counts one each, with no cap)
T_NOISE = 0.25      # (u) relative sd of the error in what is passed on (the K design: never exactly lossless)
MIN_KNOWN = 3.0     # a teller passes on only what it has had at least this much experience of


def told(speaker_wm, listener_wm, verbs, rng, d_cols, chance=T_CHANCE, n0=N0_TOLD, noise=T_NOISE):
    """THE TELLING. A speaker tells a listener what an act does to the needs: for each verb index in `verbs` the speaker has experience of
    (count >= MIN_KNOWN), with probability `chance` the listener takes up the speaker's effect, as a prior of weight n0 (experience overrides it),
    with a multiplicative error (sd `noise`) on each number. The effect passed is what the model says at the bias and at the need columns
    (const and slope on the need): all that can be put into words. Returns the verbs taken up. Lossy by design: never exactly lossless."""
    out = []
    for v in verbs:
        if speaker_wm.cnt[v] < MIN_KNOWN or rng.random() >= chance:
            continue
        P = np.zeros_like(listener_wm.M[v])
        P[:, 0] = speaker_wm.M[v][:, 0]
        for d, col in enumerate(d_cols):
            P[d, col] = speaker_wm.M[v][d, col]
        P *= 1.0 + noise * rng.standard_normal(P.shape)
        listener_wm.set_prior_matrix(v, P, n0)
        out.append(v)
    return out
