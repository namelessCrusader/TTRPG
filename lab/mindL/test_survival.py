"""The survival statistic (lab/mindL/lethal2_report.py): the stratified Cox fit recovers a known hazard ratio, and the two estimators agree."""
import numpy as np

from lab.mindL import lethal2_report as L


def _sim(hr, n, seed, base=0.002, end=730):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n):
        h = base * np.exp(rng.normal(0, 0.8))            # each seed (stratum) has its own baseline hazard
        for x in (0, 1):
            t = rng.exponential(1 / (h * hr ** x))
            rows.append((s, int(min(t, end)) + 1, int(t < end), x))
    return rows


def test_cox_recovers_a_known_hazard_ratio_and_covers_it():
    c = L.cox(_sim(0.5, 400, 1))
    assert 0.4 < c["hr"] < 0.62, c["hr"]
    assert c["hr_profile"][0] < 0.5 < c["hr_profile"][1] and c["hr_profile"][1] < 1.0
    m = L.mh(_sim(0.5, 400, 1))
    assert 0.4 < m["rr"] < 0.65, m


def test_cox_null_is_not_called_and_a_missing_arm_is_not_finite():
    c = L.cox(_sim(1.0, 300, 2))
    assert c["hr_profile"][0] < 1.0 < c["hr_profile"][1]
    rows = [(0, 100, 1, 0), (0, 730, 0, 1)]
    assert L.cox(rows)["hr"] is None                       # no death in the oracle arm: no finite ratio


def test_cox_is_paired_by_seed_a_seed_effect_cancels():
    a = _sim(0.6, 200, 3)
    b = [(s, t, e, x) for (s, t, e, x) in a]                # the same lives, strata relabelled: the same fit
    assert abs(L.cox(a)["beta"] - L.cox([(s + 7, t, e, x) for s, t, e, x in b])["beta"]) < 1e-9


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f()
    print("3 passed")
