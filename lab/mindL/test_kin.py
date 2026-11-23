"""Controls for the second constants and the kin rule (mindL5 stage A; lab/mindL/concerns2.py). Plain asserts.

    lab/.venv/bin/python -m lab.mindL.test_kin [out.json]

Synthetic streams through KinConcerns (the module the mind uses) with a known payer. The sourced constants are checked against their sources
(Hall hours, Galak habituation, Gershman extinction floor, the 14-18 year crossing), the old option is checked to be bit for bit, and the coarse
step is checked against the same law run night by night.
"""
from __future__ import annotations

import json
import math
import random
import sys

from lab.mindL.concerns import RWConcerns
from lab.mindL.concerns2 import CN2, KinConcerns, StepObs2, _Row2, coarse_advance, dials, start_weight

TABLE: dict = {}
NEUTRAL = dict(cap=1.0, lr=1.0, fade=1.0)


def mk(**kw):
    return KinConcerns(name="x", seed=0, dial=NEUTRAL, **kw)


def step(cn, targets, aimed=None, r=0.0, dt=3, got=(), helped=0.0, before=None, after=None, act="give"):
    cn.learn(StepObs2(r, before or {}, after or {}, tuple(targets), act, aimed, helped, dt, tuple(got)))


def night(cn, residents=()):
    cn.flush()
    cn.set_household(residents)
    cn.night()


def test_hall_hours():
    """Engaged hours set F: 94 / 164 / 219 h -> 0.33 / 0.50 / 0.60; a friend needs over 1,000 contact turns."""
    cn = mk()
    out = {}
    turns = 0
    marks = {94: None, 164: None, 219: None}
    while turns < 4000:
        step(cn, ["A"], aimed="A", dt=3)
        turns += 3
        h = cn.rows["A"].hrs
        for m in marks:
            if marks[m] is None and h >= m:
                marks[m] = (turns, cn._F(cn.rows["A"]))
    for m, (t, F) in marks.items():
        out[m] = dict(turns=t, F=round(F, 3))
    assert abs(marks[164][1] - 0.5) < 0.02, marks
    assert abs(marks[94][1] - 0.33) < 0.02 and abs(marks[219][1] - 0.60) < 0.02, marks
    assert marks[94][0] > 1000, marks
    TABLE["hall_hours"] = out


def test_placebo_and_presence():
    """A target that is only present (never an act at it or from it) stays at 0 however long it is there; a payer present with it gets the weight."""
    rng = random.Random(1)
    cn = mk()
    for n in range(120):
        for k in range(25):
            on = rng.random() < 0.5
            r = (0.03 if on else 0.0) + rng.gauss(0, 0.01)
            cues = ["helper", "placebo"] if on else ["placebo"]
            step(cn, cues, aimed="helper" if on else None, r=r, got=("helper",) if on else ())
        night(cn, ("helper", "placebo"))
    w = cn.table()
    TABLE["placebo"] = dict(helper=round(w["helper"], 3), placebo=round(w["placebo"], 4), a_placebo=round(cn.rows["placebo"].a, 4))
    assert w["placebo"] < 0.02 and w["helper"] > 5 * max(w["placebo"], 0.01), w


def test_extinction_slower_not_erasing():
    cn = mk()
    rng = random.Random(2)
    up = []
    for n in range(60):                      # acquisition: T present with payoff
        for k in range(30):
            on = k % 15 == 0
            step(cn, ["T"] if on else [], aimed="T" if on else None, r=(0.05 if on else 0.0), got=("T",) if on else ())
        night(cn, ("T",))
        up.append(cn.rows["T"].a)
    peak = cn.rows["T"].a_max
    half_up = next(i for i, a in enumerate(up) if a >= 0.5 * peak)
    down = []
    for n in range(300):                     # extinction: contact now costs
        for k in range(30):
            on = k % 15 == 0
            step(cn, ["T"] if on else [], aimed="T" if on else None, r=(-0.05 if on else 0.0), got=("T",) if on else ())
        night(cn, ("T",))
        down.append(cn.rows["T"].a)
    half_down = next((i for i, a in enumerate(down) if a <= 0.5 * peak), None)
    end = down[-1]
    TABLE["extinction"] = dict(peak=round(peak, 4), nights_to_half_up=half_up, nights_to_half_down=half_down, end_over_peak=round(end / peak, 3))
    assert half_down is None or half_down > half_up, (half_up, half_down)
    assert end >= 0.9 * CN2["ext_floor"] * peak, (end, peak)          # does not erase (the floor, less a little nightly fade)


def test_fade_and_history():
    """No contact for 17 months: a friend of 2 years fades; the same tie with 15 years of history barely does. Old 0.02 a night is 14 times faster."""
    def hold(hn_nights, age_nights, months=17):
        cn = mk()
        r = cn.rows.setdefault("T", _Row2())
        r.hrs, r.hn, r.nights, r.first_age = 400.0, hn_nights, age_nights, 30.0
        w0 = cn.pull("T")
        for n in range(int(months * 30.4)):
            night(cn, ())
        return cn.pull("T") / w0
    friend = hold(0, 2 * 365)             # a friend I do not live with: no co-residence history
    kin = hold(15 * 365, 15 * 365)
    old = (1 - 0.02) ** int(17 * 30.4)
    TABLE["fade_17_months"] = dict(friend_kept=round(friend, 3), long_history_kept=round(kin, 3), old_rule_kept=round(old, 6))
    assert kin > friend and kin > 0.97 and 0.85 < friend < 0.97 and old < 1e-4, TABLE['fade_17_months']


def test_habituation_set_point_and_variety():
    cn = mk()
    for n in range(300):
        for k in range(6):
            step(cn, ["T"], aimed="T", dt=3, act="give")
            step(cn, [], dt=3)               # a bout is a run of steps with T in contact
        night(cn, ("T",))
    h = cn.rows["T"].hab["give"]
    assert abs(h - CN2["h_set"]) < 0.05, h             # nonzero set point, not 0
    # recovery by variety (met two or more people today) against time alone
    def recover(variety):
        c2 = mk()
        r = c2.rows.setdefault("T", _Row2())
        r.hab = {"give": 0.35}
        r.hrs = 50.0
        if variety:
            c2.rows.setdefault("U", _Row2())
        for n in range(21):
            if variety:
                step(c2, ["U"], aimed="U")
                step(c2, ["V"], aimed="V")
            night(c2, ())
        return c2.rows["T"].hab["give"]
    t_only, with_var = recover(False), recover(True)
    gap = lambda x: (x - 0.35) / 0.65
    TABLE["habituation"] = dict(set_point_reached=round(h, 3), recovered_21d_time=round(gap(t_only), 3), recovered_21d_variety=round(gap(with_var), 3))
    assert abs(gap(t_only) - 0.23) < 0.03 and abs(gap(with_var) - 0.79) < 0.04


def test_effort_threshold_not_dose():
    cn = mk()
    rng = random.Random(3)
    for k in range(400):                                   # a scale of rewards
        step(cn, [], r=rng.gauss(0, 0.02))
    rms = math.sqrt(cn.r2)
    for k in range(30):                                    # mild cost, below the threshold: nothing
        step(cn, ["T"], aimed="T", r=-1.0 * rms, helped=0.5, before={"T": 0.6}, after={"T": 0.2})
    mild = cn.rows["T"].ejn
    for k in range(3):                                     # severe: counts
        step(cn, ["T"], aimed="T", r=-3.0 * rms, helped=0.5, before={"T": 0.6}, after={"T": 0.2})
    sev = cn.rows["T"].ejn
    for k in range(40):
        step(cn, ["T"], aimed="T", r=-3.0 * rms, helped=0.5, before={"T": 0.6}, after={"T": 0.2})
    EJ = cn.ej(cn.rows["T"])
    TABLE["effort"] = dict(mild_count=mild, severe_count_after_3=sev, EJ_after_43=EJ)
    assert mild == 0 and sev == 3 and EJ == 1.0               # a threshold; capped at doubling


def test_observed_care_cue():
    """Seeing someone I care about care for T raises my kin estimate for T; seeing someone I do not care about does not."""
    a, b = mk(), mk()
    a.rows["carer"] = _Row2()
    a.rows["carer"].e = 0.6                                # I care for the carer (swap-style weight)
    b.rows["carer"] = _Row2()
    b.rows["carer"].e = 0.1
    for k in range(150):
        a.observe_care("carer", "kid")
        b.observe_care("carer", "kid")
    ka, kb = a.khat(a.rows["kid"]), (b.khat(b.rows["kid"]) if "kid" in b.rows else 0.0)
    TABLE["observed_care"] = dict(cared_carer=round(ka, 3), indifferent_carer=round(kb, 3))
    assert ka > 0.9 and kb < 0.01


def test_kin_estimate_from_years():
    """Years co-resided as a child give the estimate: 95% at 15 years, none from adult co-residence, and an earlier start counts more."""
    def years(age0, n_years, my_age_adult=False):
        cn = mk()
        cn.age = float(age0)
        for n in range(int(n_years * 365)):
            night(cn, ("S",))
        r = cn.rows["S"]
        return cn.khat(r), cn.pull("S"), r
    k15, w15, _ = years(0, 15)
    kad, wad, _ = years(25, 15)
    k_early, _, _ = years(2, 8)
    k_late, _, _ = years(10, 8)
    TABLE["kin_estimate"] = dict(khat_15y_from_birth=round(k15, 3), khat_15y_adult=round(kad, 3), khat_8y_from_2=round(k_early, 3), khat_8y_from_10=round(k_late, 3),
                                 start_weight_age0=start_weight(0), start_weight_age20=start_weight(20), start_weight_age40=start_weight(40))
    assert 0.9 < k15 < 0.99 and kad == 0.0 and k_early > k_late > 0.0


def test_swap_flips_and_null():
    """A swap of the weight tables flips the ordering of the choice term; a mind with no history has no pull (the null)."""
    cn = mk()
    for T, hrs in (("A", 400.0), ("B", 20.0)):
        cn.rows[T] = _Row2()
        cn.rows[T].hrs = hrs
    cn.fx.set_prior("serve", rho=0.8, eta=0.0, n0=50)
    before = (cn.expected("serve", "A", 0.6), cn.expected("serve", "B", 0.6))
    t = cn.table()
    cn.set_table({"A": t["B"], "B": t["A"]})
    after = (cn.expected("serve", "A", 0.6), cn.expected("serve", "B", 0.6))
    null = mk()
    TABLE["swap"] = dict(before=[round(x, 3) for x in before], after=[round(x, 3) for x in after], null=null.expected("serve", "A", 0.6))
    assert before[0] > before[1] and after[0] < after[1] and null.expected("serve", "A", 0.6) == 0.0


def test_old_option_unchanged():
    """concerns=True is still RWConcerns with its old constants; cn2 is a different class behind the option."""
    from lab.mindL.mind import Mind
    a = Mind("Tom", "peasant", 0, concerns=True)
    b = Mind("Tom", "peasant", 0, concerns="cn2")
    assert type(a.cn) is RWConcerns and a.cn.fade == 0.02 and a.cn.lr == 0.2
    assert type(b.cn) is KinConcerns
    assert Mind("Tom", "peasant", 0).cn is None
    TABLE["options"] = dict(old=type(a.cn).__name__, cn2=type(b.cn).__name__, old_fade=a.cn.fade, new_fade0=CN2["fade0"])


def _script(cn, nights, B=2, K=6, dt=3, pay=0.03, seed=1, resid=("T",)):
    """A fixed life: each night B bouts of K steps with T (engaged), pay at T's steps, nothing otherwise."""
    for n in range(nights):
        for b in range(B):
            for k in range(K):
                step(cn, ["T"], aimed="T", r=pay, dt=dt, got=("T",), act="give", helped=0.0)
            for k in range(8):
                step(cn, [], r=0.0, dt=dt)
        night(cn, resid)


def test_coarse_matches_fine():
    """The coarse step (same law, S nights at once) against the law run night by night, on a scripted life; S = 1, 7, 30, 91."""
    nights = 365
    fine = mk()
    _script(fine, 30)
    tel = fine.telemetry()["T"]
    rows_ref = None
    ex = dict(res=1.0, con=tel["con"], eng=tel["eng"], bouts=tel["bouts"], dbar=tel["dbar"], cbar=tel["cbar"], feat=tel["feat"], ej=tel["ej"], obs_by={})
    import copy
    base_rows = copy.deepcopy(fine.rows)
    base_age = fine.age
    _script(fine, nights - 30)
    ref = fine.state()["T"]
    res = {}
    for S in (1, 7, 30, 91):
        rows = copy.deepcopy(base_rows)
        left = nights - 30
        age = base_age
        while left > 0:
            s = min(S, left)
            age = coarse_advance(rows, {"T": ex}, s, age, NEUTRAL)
            left -= s
        c = KinConcerns(name="y", seed=0, dial=NEUTRAL)
        c.rows = rows
        st = c.state()["T"]
        res[S] = {k: (round(st[k], 3), round(ref[k], 3)) for k in ("w", "F", "H", "a")}
        assert abs(st["F"] - ref["F"]) < 0.05 and abs(st["H"] - ref["H"]) < 0.01 and abs(st["w"] - ref["w"]) < 0.06, (S, st, ref)
    TABLE["coarse_vs_fine"] = res


TESTS = [test_hall_hours, test_placebo_and_presence, test_extinction_slower_not_erasing, test_fade_and_history, test_habituation_set_point_and_variety,
         test_effort_threshold_not_dose, test_observed_care_cue, test_kin_estimate_from_years, test_swap_flips_and_null, test_old_option_unchanged,
         test_coarse_matches_fine]


def main():
    bad = 0
    for t in TESTS:
        try:
            t()
            print("ok  ", t.__name__)
        except AssertionError as e:
            bad += 1
            print("FAIL", t.__name__, str(e)[:300])
    print(json.dumps(TABLE, indent=1)[:4000])
    if len(sys.argv) > 1:
        json.dump(TABLE, open(sys.argv[1], "w"), indent=1)
    print("failures:", bad)
    return bad


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
