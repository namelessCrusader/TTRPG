"""The food check's pieces (2026-11-22): the field flag sets the mean catch from the demand, leaves the variance alone, halves the raid, and is OFF by default."""
import numpy as np

from lab.mindL import food_report as F
from lab.mindL import tournament as T
from lab.world2 import world as W


def _worlds(world, n=500, **k):
    return [T.World(seed=s, bandit=False, turns=T.TURNS, flags=tuple(sorted(T.WORLDS[world])), **k) for s in range(1, n + 1)]


def test_demand_is_the_hunger_drift_over_the_portion():
    assert abs(W.NEED_RATIONS - 3.6) < 1e-9 and abs(W.FIELD_KCAL_DAY / W.NEED_RATIONS - 666.67) < 0.01


def test_field_mean_catch_meets_the_demand_and_keeps_the_variance():
    m = (5 * W.NEED_RATIONS - 5) / (4 * (1 - W.HARD_PREVALENCE))
    assert abs(W.field_catch_mean() - m) < 0.01                      # the whole-ration rounding changes the mean by well under 1 percent
    ws = _worlds("all6hl2f")
    well = [c for w in ws for n, c in w.catch.items() if n not in w.laid]
    a = np.array(well, float)
    assert abs(a.mean() - m) < 0.15, a.mean()                        # a well producer's mean catch
    assert abs(np.mean(a == 0) - 0.48) < 0.04                        # the empty chance is the field's 0.48
    assert 1.2 < a.std() / a.mean() < 1.6                            # person-day CV in the field range (1.1-1.6)
    supply = 5 + 4 * 0.9 * m                                          # counter refill + the well share of the producers
    assert abs(supply - 5 * W.NEED_RATIONS) < 1e-9


def test_field_is_off_by_default_the_old_catch_stays():
    old = [c for w in _worlds("all6hl2", 300) for n, c in w.catch.items() if n not in w.laid]
    assert abs(np.mean(old) - 2.5) < 0.2
    assert "field" not in T.WORLDS["all6hl2"] and "field" in T.WORLDS["all6hl2f"]


def test_field_raid_takes_its_share_of_the_violence_budget():
    seen = []
    orig = W.World._raid_plan
    W.World._raid_plan = lambda self, seed, p: (seen.append(p), orig(self, seed, p))[1]
    try:
        for world, share in (("all6hl2", 1.0), ("all6hl2f", W.RAID_SHARE)):
            seen.clear()
            T.World(seed=1, bandit=False, turns=T.TURNS, flags=tuple(sorted(T.WORLDS[world])))
            assert abs(seen[0] - W.HARD_RAID * share) < 1e-15, (world, seen)
    finally:
        W.World._raid_plan = orig
    assert abs(W.HARD_RAID * 365 * 0.011 - 0.01) < 1e-12              # the raid alone used the whole 1 percent before; now half of it


def test_bandit_chance_comes_from_the_budget_and_the_measured_deaths_per_arrival():
    old = W.FIELD_D_BANDIT
    try:
        W.FIELD_D_BANDIT = 1e-3
        assert abs(W.field_bandit_p() - W.R_VIOL * 0.5 * 5 / (365 * 1e-3)) < 1e-12
        assert 0.06 < W.field_bandit_p() < 0.08
    finally:
        W.FIELD_D_BANDIT = old


def _feed_world(world):
    w = T.World(seed=3, bandit=False, turns=T.TURNS, flags=tuple(sorted(T.WORLDS[world])))
    item = next(n for n in w.portions if n in W.COOK_MAKES)
    w.portions[item] = 20
    return w, item


def test_field_conserves_energy_a_repeat_ration_relieves_as_much_as_the_first():
    for world, second in (("all6hl2f", 0.25), ("all6hl2", 0.25 * W.LABOUR_REPEAT)):
        w, item = _feed_world(world)
        p = next(iter(w.people.values()))
        p.hunger, p.last_kind = 0.9, None
        w._feed(p, 1, item)
        h1 = p.hunger
        w._feed(p, 1, item)
        assert abs((h1 - p.hunger) - second) < 1e-9, (world, h1 - p.hunger)


def test_field_a_person_eats_only_when_a_whole_ration_fits_and_the_portion_stays():
    w, item = _feed_world("all6hl2f")
    p = next(iter(w.people.values()))
    p.hunger = 0.2
    assert w._feed(p, 1, item) == 0 and w.portions[item] == 20 and p.hunger == 0.2
    p.hunger = 0.3
    assert w._feed(p, 1, item) == 1 and w.portions[item] == 19
    w2, item2 = _feed_world("all6hl2")               # the old worlds feed a person at 0.2 (and waste part of the ration)
    q = next(iter(w2.people.values()))
    q.hunger = 0.2
    assert w2._feed(q, 1, item2) == 1


def _run(focal_res, dead_from=None, name="Tom"):
    res = []
    for n in range(100):
        x = {"end": 1.0, "min": focal_res(n), "burn": 0.0, "cap": 0, "hunger": 0.3, "laid": 0, "dead": int(dead_from is not None and n >= dead_from)}
        res.append({p: dict(x) for p in F.NAMES})
    return {"res": res, "deaths": [], "focal": name, "catch": [{}] * 100}


def test_depletion_counts_a_dead_night_as_full_depletion():

    a = F.life_risk(_run(lambda n: 1.0), "Tom")
    b = F.life_risk(_run(lambda n: 0.5), "Tom")
    c = F.life_risk(_run(lambda n: 1.0, dead_from=50), "Tom")
    assert a["depletion"] == 0 and abs(b["depletion"] - 0.5) < 1e-9 and abs(c["depletion"] - 0.5) < 1e-9
    assert c["near_miss_per_1000"] == 500.0 and b["near_miss_per_1000"] == 0.0


def test_paired_ci_excludes_zero_for_a_clear_shift_and_not_for_noise():
    rng = np.random.default_rng(0)
    a = F.ci_paired(rng.normal(-0.3, 0.2, 30))
    assert a["t_ci"][1] < 0 and a["boot_ci"][1] < 0
    b = F.ci_paired(rng.normal(0.0, 0.2, 30))
    assert b["t_ci"][0] < 0 < b["t_ci"][1]


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f()
    print("passed")
