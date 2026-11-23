"""E3b. Acquired concerns, second constants and the kin rule (mindL5 stage A, 2026-11-23; docs/family-design.md Fam1, Fam6).

Behind the mind option concerns="cn2" (concerns=True stays RWConcerns in concerns.py, bit for bit). The interface is the same
(Concerns). What changes is the TIME SCALE (set from sources, never tuned to pass) and one more input, the kin estimate.

THE LAW, per target T of a mind (a person; a thing keeps only the reward association a, as before):

    w_T = min(ceil, [ W_ATT * F_T * (1 + G_K * cap * khat_T)  +  gain * max(a_T, 0) ] * (1 + EJ_T)  +  e_T )

  F_T     familiarity from hours of ENGAGED contact (an act of mine at T, or T's act at me, in the step):  F = 1 - exp(-hours / H_F).
          Hall 2018: 94, 164, 219 hours of contact (adults) for casual friend, friend, close friend (more likely than not). H_F = 164 / ln 2
          puts "friend" at F = 0.5 (94 h -> 0.33, 219 h -> 0.60). At 300 turns a day (the library's convention) a turn is 4.8 minutes,
          so 164 h is 2,050 turns: not the 11 turns of the old per-turn rule. Mere presence is NOT contact (a placebo in the room stays 0).
          What the talk is about (Hall: goals add, small talk lowers) is not modelled.
  W_ATT   the weight a long, close friendship reaches (0.30). Anchor: Madsen et al. 2007 (effort for the recipient rises roughly linearly
          with relatedness; self > r=.5 > r=.25 = a best friend): self = 1, r=.5 kin = 0.5, a best friend 0.25.
  khat_T  the KINSHIP ESTIMATE from cues (never a flag):  khat = 1 - (1 - k_y)(1 - k_c).
            k_y = 1 - exp(-Y / TAU_K), Y the years I lived with T while I was under 18 (an earlier start counts more: each night is weighted by
                  max(EARLY_FLOOR, 1 - EARLY_SLOPE * my age at the start), no hard window); Lieberman, Tooby, Cosmides 2007 (r = .41 of care with
                  years co-resided, the lines cross at 14-18 years) -> TAU_K = 5 years (95% at 15; library's own heavy inference).
            k_c = the OBSERVED-CARE cue: seeing someone I care about (weight above CARE_LO, full at CARE_HI) care for T moves it up by RHO_OBS per
                  episode (Lieberman's maternal perinatal cue, in general form). RHO_OBS is unsourced.
  G_K     the gain of the estimate: (0.5 / W_ATT - 1) / 0.95, so a fully cued kin tie sits at r = .5 and a friend at .25. The per-person CAPACITY
          dial `cap` (lognormal, SD 0.15; learning-rate spread SD 0.1-0.2 within a task, Eckstein et al. 2022) scales it.
  a_T     the reward association, learned PER CONTACT BOUT (a run of steps with T in contact), not per turn: one bout is one spaced pairing
          (Balsam and Gallistel 2009). Rescorla-Wagner with the cues of the bout sharing one prediction (a placebo stays near 0). Rate LR_BOUT =
          0.07 (the deterministic-task mean of Eckstein 2022; slowest of the fitted human rates), times (1 + C_REC * H_T): the reciprocity
          coupling is scaled by the accumulated history (Ache, Gurven's slope .56 for close kin against .09: 6 times). EXTINCTION is slower than
          acquisition (EXT_RATIO) and does not erase: a never drops below EXT_FLOOR of its peak (Gershman 2013: freezing 24-30% after 24 trials).
  H_T     the accumulated history: H = 1 - exp(-N_w / (TAU_H * 365)), N_w the early-start-weighted nights I lived with T. Slows fade, strengthens
          the reciprocity coupling. TAU_H = 5 years (library, heavy inference).
  EJ_T    effort justification, a THRESHOLD not a dose (Aronson and Mills 1959: severe 195.3, mild 171.1, control 166.7): an act of mine at T that
          relieved T's need and COST me at least EJ_THETA (2) RMS step-rewards counts (the threshold is unsourced); EJ = min(1, EJ_STEP * count). The IKEA effect (Norton 2012: one
          completed build doubled the value, d = .57) caps it at doubling. Costs below the threshold teach nothing.
  e_T     only the swap control (set_table) puts weight here.

FADE (a night; a night is a day). fade = FADE0 / ((1 + age / TAU_F) * (1 + H_years / H0)) * dial * (1 - contact), applied to the hours, a and the
effort count; the kin estimate and the history do not fade (kin ties do not decay over 17 months, Roberts and Dunbar 2015).
  FADE0 = 0.0014 a night (Burt 2002 year-1 non-kin hazard; Roberts and Dunbar: 0.0014/day for friends); the old 0.02 was 14 times too fast.
  age: years since I first met T; TAU_F = 1 year (library). contact: min(1, engaged turns today / CONTACT_HOLD): a day of contact holds the tie
  (Roberts and Dunbar: the gap since the last contact predicted lost closeness). CONTACT_HOLD = 30 turns (2.4 h) is unsourced. H0 = 2 years is
  unsourced (at 15 years of history and age, fade is 1/136 of FADE0: "kin barely fade"). dial: 12% of minds fade at 0.4 of the rate (chronic
  grief 9.8-15.6%, Bonanno 2002 / Lundorff 2017; rate ratio 0.028/0.071).

HABITUATION per feature (what T and I do together, plus "presence"), with a NONZERO SET POINT and recovery by VARIETY: each bout multiplies the
excess of h_f over H_SET by (1 - HAB_F) (4.1-4.3% a play, Galak, Redden, Kruger 2009; the set point is unsourced, Diener et al. 2006 say adaptation is
incomplete). At night a feature not met today recovers REC_VAR of the gap if I met two or more different people today, else REC_TIME (23% in 21 days
alone, 79% with variety: Galak). A bout teaches in proportion to the mean h_f of its features.

THE COARSE STEP (coarse_advance). The same law over S nights at once, from the pair's exposure rates per night (engaged turns, bouts, mean bout
contrast, features, costly acts for T, care episodes seen). Closed forms of the same geometric recurrences; nothing new is learned there.
"""
from __future__ import annotations

import math
import random
from collections import namedtuple

from lab.mindL.concerns import JND, RWConcerns, need_of  # noqa: F401  (need_of re-exported for the mind)

CN2 = dict(
    hours_per_turn=24.0 / 300.0,
    h_f=164.0 / math.log(2.0),
    w_att=0.30,
    g_k=(0.5 / 0.30 - 1.0) / 0.95,
    tau_k=5.0, tau_h=5.0,
    early_slope=0.025, early_floor=0.25, child_age=18.0,
    care_lo=0.30, care_hi=0.50, rho_obs=0.02,
    lr_bout=0.07, c_rec=5.0, ext_ratio=0.30, ext_floor=0.27,
    ej_step=0.25, ej_theta=2.0,
    fade0=0.0014, tau_f=1.0, h0=2.0, contact_hold=30.0,
    hab_f=0.045, h_set=0.35, rec_time=1.0 - (1.0 - 0.23) ** (1.0 / 21), rec_var=1.0 - (1.0 - 0.79) ** (1.0 / 21),
    rel=False,                            # mindL6 (off: mindL5 bit for bit): effort justification and the observed-care cue graded RELATIVE to my other ties
    gain=20.0, ceil=1.0, lr_base=0.05, dis=0.05, helped_min=JND,
)
# RELATIVE GRADING (mindL6, option params={"rel": True}; mindL5's cause of the saturation of every weight in a full run). Two inputs were absolute
# and saturating: the effort count (EJ = min(1, 0.25 n): four costly acts for anyone double the weight) and the observed-care cue (kc moves up by
# RHO_OBS x w a hundred times a year: 0.9-0.99 for every pair, lodgers included). A cue to kinship and a sign of what I have paid for are worth
# something only against what I have seen and paid for OTHERS: a person's ties are ranked by them. So each is graded by its share of the
# strongest tie of mine: EJ_T = min(1, step n_T) * n_T / max_j n_j ; kc_T = kc_T * N_T / max_j N_j, N the unsaturated count of care episodes seen
# for T (care-gated, w_c each). The best-paid-for tie keeps its full effect (a best friend, a kin at r = .5 anchors as before: Madsen 2007), the
# others are in proportion to what they were. Nothing here has a new constant: the relative form is the only change.
CARE_VERBS = frozenset({"give", "help up", "bandage", "defend", "serve", "tend"})     # the world's HELPED set: what a witness reads as care

# a step as cn2 sees it: StepObs plus `got`, the people whose acts at me I perceived during the step
StepObs2 = namedtuple("StepObs2", "r before after contact act aimed helped dt got")


class _Row2:
    __slots__ = ("a", "a_max", "hrs", "hn", "ky", "kc", "ejn", "e", "hab", "first_age", "nights", "t_eng", "t_con", "t_feat", "kn")

    def __init__(self):
        self.a = self.a_max = self.hrs = self.hn = self.ky = self.kc = self.ejn = self.e = self.kn = 0.0
        self.hab: dict = {}
        self.first_age = None
        self.nights = 0.0
        self.t_eng = self.t_con = 0.0
        self.t_feat: set = set()


class _Bout:
    __slots__ = ("n", "sum", "turns", "conc", "feats")

    def __init__(self):
        self.n, self.sum, self.turns, self.conc, self.feats = 0, 0.0, 0, set(), set()


def dials(seed, name, c=CN2):
    """A mind's own dials, from its own generator (never the world's): capacity, learning-rate spread, fade."""
    r = random.Random(f"cn2/{seed}/{name}")
    cap = min(1.6, max(0.6, math.exp(r.gauss(0.0, 0.15))))
    lr = min(1.6, max(0.6, math.exp(r.gauss(0.0, 0.15))))
    fade = 0.4 if r.random() < 0.12 else 1.0
    return dict(cap=cap, lr=lr, fade=fade)


def start_weight(age, c=CN2):
    """The weight of a night of living with T that began when I was `age`: an earlier start counts more, no hard window."""
    return max(c["early_floor"], 1.0 - c["early_slope"] * max(0.0, age))


def fade_rate(age_years, hist_years, dial, protect, c=CN2):
    return c["fade0"] / ((1.0 + age_years / c["tau_f"]) * (1.0 + hist_years / c["h0"])) * dial * (1.0 - protect)


def refs(rows, c):
    """The strongest of my person ties by the unsaturated counts (kn, ejn), or None when grading is absolute."""
    if not c.get("rel"):
        return None
    pr = [r for T, r in rows.items() if not T.startswith("thing:")]
    return (max((r.kn for r in pr), default=0.0), max((r.ejn for r in pr), default=0.0))


def graded(r, c, ref):
    """(kc, EJ) of a row, absolute or relative to ref."""
    ej = min(1.0, c["ej_step"] * r.ejn)
    kc = r.kc
    if ref is not None:
        kc = kc * (r.kn / ref[0] if ref[0] > 0 else 0.0)
        ej = ej * (r.ejn / ref[1] if ref[1] > 0 else 0.0)
    return kc, ej


class KinConcerns(RWConcerns):
    def __init__(self, name="?", seed=0, age=30.0, params=None, dial=None, **kw):
        super().__init__(**{k: v for k, v in kw.items() if k in ("effects",)})
        self.c = {**CN2, **(params or {})}
        self.gain, self.ceil, self.lr_base, self.dis, self.helped_min = (self.c[k] for k in ("gain", "ceil", "lr_base", "dis", "helped_min"))
        self.name, self.dial = name, dial or dials(seed, name)
        self.age = float(age)
        self.bouts: dict = {}
        self.r2 = 1e-4                        # running mean square of a step's reward (the scale of a cost)
        self.nrs = 0                          # nights lived
        self.tele: dict = {}                  # target -> exposure counters (for the coarse step's rates)
        self.residents: tuple = ()

    # ------------------------------------------------------------ reading
    def _F(self, r):
        return 1.0 - math.exp(-r.hrs / self.c["h_f"])

    def _Hn(self, r):
        return 1.0 - math.exp(-r.hn / (self.c["tau_h"] * 365.0))

    def khat(self, r, ref=None):
        ky = 1.0 - math.exp(-r.ky / (self.c["tau_k"] * 365.0))
        return 1.0 - (1.0 - ky) * (1.0 - graded(r, self.c, ref)[0])

    def ej(self, r, ref=None):
        return graded(r, self.c, ref)[1]

    def _wr(self, r, T, ref=None):
        if T.startswith("thing:"):
            return min(self.ceil, self.gain * max(r.a, 0.0) + r.e)
        base = self.c["w_att"] * self._F(r) * (1.0 + self.c["g_k"] * self.dial["cap"] * self.khat(r, ref)) + self.gain * max(r.a, 0.0)
        return min(self.ceil, base * (1.0 + self.ej(r, ref)) + r.e)

    def pull(self, target):
        r = self.rows.get(target)
        return self._wr(r, target, refs(self.rows, self.c)) if r is not None else 0.0

    def table(self):
        ref = refs(self.rows, self.c)
        return {T: self._wr(r, T, ref) for T, r in self.rows.items()}

    def set_table(self, table):
        self.rows, self.bouts = {}, {}
        for T, w in table.items():
            r = self.rows[T] = _Row2()
            r.e = float(w)

    def state(self):
        """The inspectable state per target (the coarse check reads and loads this)."""
        ref = refs(self.rows, self.c)
        return {T: dict(w=self._wr(r, T, ref), a=r.a, a_max=r.a_max, hrs=r.hrs, F=self._F(r), hn=r.hn, H=self._Hn(r), ky=r.ky, kc=r.kc, kn=r.kn,
                        khat=self.khat(r, ref), ejn=r.ejn, EJ=self.ej(r, ref), nights=r.nights, first_age=r.first_age, hab=dict(r.hab))
                for T, r in self.rows.items() if not T.startswith("thing:")}

    def load_state(self, st):
        """Load rows from `state()` (the coarse past into a mind). Things are not loaded."""
        for T, s in st.items():
            r = self.rows.setdefault(T, _Row2())
            r.a, r.a_max, r.hrs, r.hn, r.ky, r.kc, r.ejn = (s[k] for k in ("a", "a_max", "hrs", "hn", "ky", "kc", "ejn"))
            r.nights, r.first_age, r.hab = s["nights"], s["first_age"], dict(s["hab"])
            r.kn = s.get("kn", 0.0)

    # ------------------------------------------------------------ what the mind sees
    def observe_care(self, carer, target, t=None):
        """I saw `carer` care for `target` (neither is me). If I care about the carer, my kin estimate for the target moves up."""
        if target.startswith("thing:") or target == carer:
            return
        c = self.c
        wc = min(1.0, max(0.0, (self.pull(carer) - c["care_lo"]) / (c["care_hi"] - c["care_lo"])))
        r = self.rows.setdefault(target, _Row2())
        r.kc += c["rho_obs"] * wc * (1.0 - r.kc)
        r.kn += wc
        tl = self._tele(target)
        tl["obs"] += 1
        tl["obs_by"][carer] = tl["obs_by"].get(carer, 0) + 1

    def _tele(self, T):
        t = self.tele.get(T)
        if t is None:
            t = self.tele[T] = dict(nights=0, con=0.0, eng=0.0, bouts=0, dsum=0.0, csum=0.0, feat={}, ej=0, care=0, obs=0, obs_by={}, res=0)
        return t

    # ------------------------------------------------------------ learning
    def learn(self, st):
        c = self.c
        r_c = 0.0
        for T, b in st.before.items():
            a = st.after.get(T)
            if a is not None:
                r_c += self.pull(T) * (b - a)
        cues = set(st.contact)
        if st.aimed is not None:
            cues.add(st.aimed)
        got = getattr(st, "got", ())
        d_step = st.r - self.base
        self.base += self.lr_base * (st.r - self.base)
        for T in cues:
            row = self.rows.setdefault(T, _Row2())
            b = self.bouts.get(T)
            if b is None:
                b = self.bouts[T] = _Bout()
            b.n += 1
            b.sum += d_step
            b.turns += st.dt
            b.conc |= cues
            b.feats.add("presence")
            if T == st.aimed and st.act:
                b.feats.add(st.act)
            row.t_con += st.dt
            if not T.startswith("thing:") and (T == st.aimed or T in got):
                row.hrs += st.dt * c["hours_per_turn"]
                row.t_eng += st.dt
        if st.aimed is not None and st.r < 0 and not st.aimed.startswith("thing:"):    # effort justification: it cost me, and it was for them
            b, a = st.before.get(st.aimed), st.after.get(st.aimed)
            seen = (b - a) if (b is not None and a is not None) else 0.0
            tl = self._tele(st.aimed)
            if max(seen, st.helped) >= self.helped_min:
                tl["care"] += 1
                if -st.r >= c["ej_theta"] * math.sqrt(self.r2):
                    self.rows.setdefault(st.aimed, _Row2()).ejn += 1.0
                    tl["ej"] += 1
        self.r2 += 0.01 * (st.r * st.r - self.r2)
        for T in [T for T in self.bouts if T not in cues]:
            self._close(T)
        return r_c

    def _close(self, T):
        c = self.c
        b = self.bouts.pop(T)
        row = self.rows[T]
        conc = b.conc
        pred = sum(self.rows[U].a for U in conc if U in self.rows)
        d = b.sum / b.n - pred
        hbar = sum(row.hab.get(f, 1.0) for f in b.feats) / len(b.feats)
        Hn = 0.0 if T.startswith("thing:") else self._Hn(row)
        lr = c["lr_bout"] * self.dial["lr"] * hbar * (1.0 + c["c_rec"] * Hn) / len(conc)
        if d >= 0:
            row.a += lr * d
        else:
            row.a += lr * c["ext_ratio"] * d
            if row.a_max > 0:
                row.a = max(row.a, c["ext_floor"] * row.a_max)
        row.a_max = max(row.a_max, row.a)
        if row.a < 0 and row.ejn > 0:
            row.ejn *= 1.0 - c["dis"]                               # disillusion
        for f in b.feats:
            h = row.hab.get(f, 1.0)
            row.hab[f] = c["h_set"] + (h - c["h_set"]) * (1.0 - c["hab_f"])
            row.t_feat.add(f)
        tl = self._tele(T)
        tl["bouts"] += 1
        tl["dsum"] += b.sum / b.n
        tl["csum"] += len(conc) - 1
        for f in b.feats:
            tl["feat"][f] = tl["feat"].get(f, 0) + 1

    def flush(self):
        for T in list(self.bouts):
            self._close(T)
        super().flush()

    def set_household(self, residents, age=None):
        """Who lives here tonight (names, not me), for the co-residence history; my age (years) at the first call."""
        self.residents = tuple(residents)

    def night(self):
        c = self.c
        res = set(self.residents)
        self.nrs += 1
        met = sum(1 for T, r in self.rows.items() if r.t_con > 0 and not T.startswith("thing:"))
        for T in res:
            self.rows.setdefault(T, _Row2())             # a co-resident is a history even before a first act
        for T in list(self.rows):
            r = self.rows[T]
            if T.startswith("thing:"):
                r.a *= 1.0 - c["fade0"] * self.dial["fade"]
                if abs(r.a) < 1e-5 and r.e < 1e-5:
                    del self.rows[T]
                continue
            tl = self._tele(T)
            if T in res:
                if r.first_age is None:
                    r.first_age = self.age
                ws = start_weight(r.first_age, c)
                r.hn += ws
                if self.age < c["child_age"]:
                    r.ky += ws
                tl["res"] += 1
            if r.first_age is None and (r.t_con > 0 or r.hrs > 0):
                r.first_age = self.age
            tl["nights"] += 1
            tl["con"] += r.t_con
            tl["eng"] += r.t_eng
            f = fade_rate(r.nights / 365.0, r.hn / 365.0, self.dial["fade"], min(1.0, r.t_eng / c["contact_hold"]), c)
            r.hrs *= 1.0 - f
            r.a *= 1.0 - f
            r.ejn *= 1.0 - f
            r.nights += 1
            rate = c["rec_var"] if met >= 2 else c["rec_time"]
            for ft in r.hab:
                if ft not in r.t_feat:
                    r.hab[ft] += (1.0 - r.hab[ft]) * rate
            r.t_con = r.t_eng = 0.0
            r.t_feat = set()
            if r.hrs < 1e-6 and abs(r.a) < 1e-6 and r.hn <= 0 and r.kc <= 0 and r.e < 1e-6:
                del self.rows[T]
        self.age += 1.0 / 365.0

    def telemetry(self):
        """Exposure rates per night per target, from the nights lived (for the coarse step)."""
        out = {}
        for T, t in self.tele.items():
            n = max(1, t["nights"])
            if T.startswith("thing:") or t["nights"] == 0:
                continue
            bo = max(1, t["bouts"])
            out[T] = dict(nights=t["nights"], res=t["res"] / n, con=t["con"] / n, eng=t["eng"] / n, bouts=t["bouts"] / n, dbar=t["dsum"] / bo,
                          cbar=t["csum"] / bo, feat={f: k / n for f, k in t["feat"].items()}, ej=t["ej"] / n, care=t["care"] / n, obs=t["obs"] / n,
                          obs_by={k: v / n for k, v in t["obs_by"].items()})
        return out

    def reset_telemetry(self):
        self.tele = {}


# ---------------------------------------------------------------------------------------------- the coarse step


def _geo(x, phi, q, f, S):
    """x after S nights of x <- (x + q)(1 - f), phi = (1 - f)^S: the closed form (f may be 0)."""
    if f < 1e-12:
        return x + q * S
    return x * phi + q * (1.0 - f) * (1.0 - phi) / f


def _poisson(rng, lam):
    if lam > 30:
        return max(0.0, rng.gauss(lam, math.sqrt(lam)))
    L, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= L:
            return float(k)
        k += 1


def _wr_plain(r, c, dial, ref=None):
    """A row's weight as KinConcerns computes it, without the mind (the coarse step has only rows)."""
    F = 1.0 - math.exp(-r.hrs / c["h_f"])
    ky = 1.0 - math.exp(-r.ky / (c["tau_k"] * 365.0))
    kc, ej = graded(r, c, ref)
    kh = 1.0 - (1.0 - ky) * (1.0 - kc)
    base = c["w_att"] * F * (1.0 + c["g_k"] * dial["cap"] * kh) + c["gain"] * max(r.a, 0.0)
    return min(c["ceil"], base * (1.0 + ej) + r.e)


def coarse_advance(rows, expo, S, age, dial, c=CN2, noise=None):   # c: the constants (CN2, or {**CN2, 'rel': True})
    """Advance one mind's rows (name -> _Row2; persons only) by S nights, given expo (name -> Expo-like dict of per-night rates), the mind's age
    (years, at the step's start) and dial. The same law as KinConcerns, in closed form. `noise`: a random.Random; the counts of the step are
    then Poisson draws around the rates (so a controlled past has sampling noise). Mutates rows; returns the new age."""
    S = float(S)

    def draw(rate):
        if noise is None or rate <= 0:
            return rate
        return _poisson(noise, rate * S) / S

    others_a = [r.a for r in rows.values()]
    mean_a = sum(others_a) / len(others_a) if others_a else 0.0
    met = sum(1 for T, e in expo.items() if e["con"] > 0 and e.get("res", 0) > 0.5)
    for T, e in expo.items():
        r = rows.get(T)
        resident = e.get("res", 0.0)
        if r is None:
            r = rows[T] = _Row2()
        if resident > 0 and r.first_age is None:
            r.first_age = age
        eng, con = draw(e["eng"]), e["con"]
        bouts, dbar, cbar = draw(e["bouts"]), e["dbar"], e["cbar"]
        ej = draw(e["ej"])
        # ---- history and kin estimate (no fade)
        if resident > 0:
            ws = start_weight(r.first_age if r.first_age is not None else age, c)
            r.hn += ws * resident * S
            if age < c["child_age"]:
                r.ky += ws * resident * S
        for carer, rate in e["obs_by"].items():          # the care I saw for T, by whom: my weight for the carer now gates it (the same wc as observe_care)
            cr = rows.get(carer)
            if cr is None or rate <= 0:
                continue
            wc = min(1.0, max(0.0, (_wr_plain(cr, c, dial, refs(rows, c)) - c["care_lo"]) / (c["care_hi"] - c["care_lo"])))
            k_eps = draw(rate) * S
            r.kc = 1.0 - (1.0 - r.kc) * (1.0 - c["rho_obs"] * wc) ** k_eps
            r.kn += wc * k_eps
        # ---- fade at the step's middle
        mid_age = (r.nights + S / 2) / 365.0
        f = fade_rate(mid_age, r.hn / 365.0, dial["fade"], min(1.0, eng / c["contact_hold"]), c)
        phi = (1.0 - f) ** S
        # ---- hours, effort count: x <- (x + q)(1 - f)
        r.hrs = _geo(r.hrs, phi, eng * c["hours_per_turn"], f, S)
        r.ejn = _geo(r.ejn, phi, ej, f, S)
        # ---- habituation per feature
        feats = e["feat"]
        nf = max(1e-9, sum(feats.values()))
        hsum = 0.0
        for ft, n_b in feats.items():
            h0 = r.hab.get(ft, 1.0)
            dec = (1.0 - c["hab_f"]) ** n_b
            rec = c["rec_var"] if met >= 2 else c["rec_time"]
            # per night: bouts pull h toward the set point, then (if the feature was met today) no recovery; mean-field: recover by the share of nights it was not met
            p_not = math.exp(-n_b)                                        # chance of a night without it (Poisson)
            alpha = dec * (1.0 - p_not * rec) if n_b > 0 else (1.0 - rec)
            beta = c["h_set"] * (1.0 - dec) * (1.0 - p_not * rec) + p_not * rec
            h_inf = beta / (1.0 - alpha) if alpha < 1.0 else h0
            r.hab[ft] = h_inf + (h0 - h_inf) * alpha ** S
            hsum += r.hab[ft] * n_b
        hbar = hsum / nf if feats else 1.0
        # ---- the reward association, per bout
        if bouts > 0:
            Hn = 1.0 - math.exp(-r.hn / (c["tau_h"] * 365.0))
            share = 1.0 + cbar
            lam = c["lr_bout"] * dial["lr"] * hbar * (1.0 + c["c_rec"] * Hn) / share
            a_star = dbar - cbar * mean_a
            if a_star < r.a:
                lam *= c["ext_ratio"]
            lam_n = 1.0 - (1.0 - min(1.0, lam)) ** bouts
            alpha = (1.0 - lam_n) * (1.0 - f)
            beta = lam_n * a_star * (1.0 - f)
            a_inf = beta / (1.0 - alpha) if alpha < 1.0 else r.a
            a_new = a_inf + (r.a - a_inf) * alpha ** S
            if a_star < r.a and r.a_max > 0:
                a_new = max(a_new, c["ext_floor"] * r.a_max)
            r.a = a_new
            r.a_max = max(r.a_max, r.a)
        else:
            r.a *= phi
        r.nights += S
    # a target not in expo (no exposure this step) fades
    for T, r in rows.items():
        if T in expo:
            continue
        f = fade_rate((r.nights + S / 2) / 365.0, r.hn / 365.0, dial["fade"], 0.0, c)
        phi = (1.0 - f) ** S
        r.hrs *= phi
        r.a *= phi
        r.ejn *= phi
        r.nights += S
    return age + S / 365.0
