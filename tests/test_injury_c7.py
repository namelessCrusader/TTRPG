"""INJURY LAWS (wave 13c, lane C7): the real causes of a faint, the vasovagal
reflex with its per-person dial, falls through the landing chain, the pain
withdrawal and the startle, both on the D2 sleep-threshold law.

Each test is on one law with a person or a plain dict; no scene code."""
import math
import re
from pathlib import Path

import numpy as np

from src.voxel import injury as IJ
from src.voxel import reflex as RF
from src.voxel.materials import BODY, IRON, STONE, TICK_S
from src.voxel.scenes import _person
from src.voxel.sim import World

ROOT = Path(__file__).resolve().parents[1]


def _world():
    w = World(40, 40, 30, voxel_cm=5)
    w.fill(0, 40, 0, 40, 0, 1, STONE)
    w.exits = []
    p = _person(w, 20, 20)
    w.step()
    w.step()
    return w, p


def _verbs(w, since):
    return [(e.verb, e.why) for e in w.events.rows if e.id >= since]


# ── the shim is gone ────────────────────────────────────────────────────────

def test_the_incap_SHIM_is_RETIRED():
    for f in (ROOT / "src" / "voxel").glob("*.py"):
        assert not re.search(r"""p(\.get\(|\[)["']incap""", f.read_text()), f.name


# ── what knocks a person out ────────────────────────────────────────────────

def test_a_BRUISE_does_not_faint_the_average_person():
    w, p = _world()
    p["vagal_dial"] = 5.0                  # an average, hard-to-faint person
    p["hurt"] = 0.9
    for _ in range(80):
        w.step()
    assert p["awake"] and p["alive"], "hurt alone is not a cause"


def test_BURN_SHOCK_faints_and_kills_at_the_old_lines_and_names_the_burns():
    w, p = _world()
    p["burn"] = BODY["faint_burn"] + 0.02
    w._law_life()
    f = w.events.rows[-1]
    assert not p["awake"] and f.verb == "faints" and f.why == "the burns"
    p["burn"] = BODY["death_burn"] + 0.02
    w._law_life()
    assert not p["alive"] and w.events.rows[-1].verb == "dies"
    # and a burn short of the line does neither
    w2, q = _world()
    q["burn"] = BODY["faint_burn"] - 0.03
    q["vagal_dial"] = 5.0
    w2._law_life()
    assert q["awake"]


def test_a_HEAD_BLOW_keeps_a_person_out_until_ko_until_and_then_he_wakes():
    w, p = _world()
    p["ko_until"] = w.tick + 40
    w._law_life()
    assert not p["awake"] and w.events.rows[-1].why == "a blow"
    for _ in range(30):
        w.tick += 1
        w._law_life()
    assert not p["awake"], "still out"
    for _ in range(20):
        w.tick += 1
        w._law_life()
    assert p["awake"], "round, once the blow's time is over and the oxygen is good"


def test_a_head_BLOW_of_the_head_bone_sets_ko_until_by_the_head_result():
    p = {"pid": 1, "name": "x", "awake": True, "alive": True}
    soft = IJ.head_blow(p, 5.0, 0)
    assert not soft["ko"] and "ko_until" not in p
    hard = IJ.head_blow(p, 150.0, 0)
    assert hard["ko"] and p["ko_until"] > 0


# ── the vasovagal reflex ────────────────────────────────────────────────────

def test_the_vagal_DIAL_is_a_distribution_most_need_a_lot_a_few_faint_easily():
    th = sorted(IJ.vagal_threshold({"pid": i, "name": f"n{i}"}) for i in range(3000))
    med = th[1500]
    assert 0.5 < med < 0.72
    assert th[150] < 0.25 < med          # the lowest 5 percent
    assert th[-1] < 1.6 and th[0] > 0.1
    a = IJ.vagal_threshold({"pid": 7, "name": "a"})
    assert a == IJ.vagal_threshold({"pid": 7, "name": "a"}), "deterministic"
    assert a != IJ.vagal_threshold({"pid": 8, "name": "a"})


def _hit_hard(w, p, hurt_joules):
    w._wound(p, hurt_joules)


def test_the_SAME_pain_faints_a_touchy_person_and_not_a_steady_one():
    out = {}
    for tag, dial in (("touchy", 0.15), ("steady", 1.3)):
        w, p = _world()
        p["vagal_dial"] = dial
        w._wound(p, 0.45 * BODY["hurt_J"])         # a hard knock: hurt 0.45
        for _ in range(int(20.0 / TICK_S)):
            w.step()
            if not p["awake"]:
                break
        out[tag] = p
    assert not out["touchy"]["awake"], "pain -> vagal -> flow falls -> faint"
    assert out["steady"]["awake"]


def test_a_vasovagal_faint_names_the_pain_and_he_comes_round_lying_flat():
    w, p = _world()
    p["vagal_dial"] = 0.15
    w._wound(p, 0.45 * BODY["hurt_J"])
    since = w.events.next_id
    woke = None
    for t in range(int(120.0 / TICK_S)):
        w.step()
        if woke is None and any(v == "wakes" for v, _ in _verbs(w, since)):
            woke = t
            break
    vs = _verbs(w, since)
    assert ("faints", "the pain") in vs
    assert woke is not None and woke * TICK_S < 100.0, "lying flat takes the drive away"
    assert p["alive"]
    assert p["vagal"] < 0.5


def test_the_vagal_flow_never_kills_alone():
    assert IJ.flow_factor({"vagal": 1.0}) > BODY["death_o2"]


# ── falls ────────────────────────────────────────────────────────────────────

def _stand(pid=1):
    return {"pid": pid, "name": "f", "awake": True, "alive": True}


def _pose_upright():
    z = np.arange(0, 30)
    return np.stack([np.zeros(30), np.zeros(30), z], axis=1)


def test_a_low_FEET_FIRST_drop_hurts_nobody_and_a_high_one_breaks_and_pools_blood():
    sheet_m = IJ.sheet_of(_stand())["mass_kg"]
    pose = _pose_upright()
    p = _stand()
    p["_eye"] = (0, 0, 29)
    assert IJ.landing_effects(p, sheet_m * 9.81 * 2.0, pose, None, 1, 0) is not None
    assert "ko_until" not in p and "organ_dead" not in p
    q = _stand(2)
    q["_eye"] = (0, 0, 29)
    got = IJ.landing_effects(q, sheet_m * 9.81 * 10.0, pose, None, 1, 0)
    assert got["mode"] == "feet"
    assert q.get("fractures") and q.get("pool_left_ml", 0) > 0


def test_a_HEAD_FIRST_fall_from_height_kills_through_the_brain():
    sheet_m = IJ.sheet_of(_stand())["mass_kg"]
    p = _stand(3)
    p["_eye"] = (0, 0, 0)                     # the head is at the low end
    got = IJ.landing_effects(p, sheet_m * 9.81 * 8.0, _pose_upright(), None, 1, 0)
    assert got["mode"] == "head"
    assert p.get("organ_dead") == "brain"


def test_pooled_blood_leaves_the_circulation_but_not_the_body():
    w, p = _world()
    b0 = w._blood0(p)
    p["pool_left_ml"] = 0.45 * b0
    before = p["blood_ml"]
    for _ in range(int(6000 / TICK_S)):        # the pooling clock is 30 minutes
        IJ.pool_step(p)
    assert p["blood_ml"] == before, "the books: it is still in the body"
    assert p["pooled_ml"] > 0.4 * b0
    assert w._perfusion(p) < 0.6, "but it no longer perfuses"


# ── the pain withdrawal and the startle (D2) ─────────────────────────────────

def _held_hot(w, p, T_C):
    cell = (20, 20, 3)
    w.mat[cell] = IRON
    p["held"] = {"cell": cell, "mat": int(IRON), "label": "bar"}
    T = np.full(w.shape, 20.0, np.float32)
    T[cell] = T_C
    return T


def test_a_HOT_hold_is_dropped_and_a_cool_one_is_kept():
    w, p = _world()
    T = _held_hot(w, p, 30.0)
    for _ in range(80):
        w.tick += 1
        w._withdraw(p, T)
    assert p["held"] is not None
    T = _held_hot(w, p, 200.0)
    since = w.events.next_id
    for _ in range(80):
        w.tick += 1
        if p["held"] is None:
            break
        w._withdraw(p, T)
    assert p["held"] is None
    ev = [e for e in w.events.rows if e.id >= since][-1]
    assert ev.verb == "releases" and ev.result == "withdrew" and ev.why == "reflex:withdraw"


def test_a_chosen_hold_or_a_catch_silences_the_withdrawal():
    w, p = _world()
    T = _held_hot(w, p, 200.0)
    w._chose(p, "hold_hot", ticks=500)
    for _ in range(100):
        w.tick += 1
        w._withdraw(p, T)
    assert p["held"] is not None


def test_sleep_RAISES_the_withdraw_threshold_it_does_not_switch_it_off():
    w, p = _world()
    awake = RF.threshold_now(p, "withdraw")
    p["asleep"] = True
    asleep = RF.threshold_now(p, "withdraw")
    assert asleep == awake * (1 + RF.DEPTH_GAIN * RF.SLEEP_DEPTH)
    p["awake"], p["asleep"] = False, False
    assert RF.threshold_now(p, "withdraw") > asleep, "deeper still, and not infinite"
    assert RF.fire_chance(p, "withdraw", 1.0) < RF.fire_chance({**p, "awake": True}, "withdraw", 1.0)
    p["alive"] = False
    assert RF.threshold_now(p, "withdraw") is None, "the dead are off"


class _Ears(RF.ReflexMixin):
    tick = 0


def test_a_NOISE_wakes_a_sleeper_by_chance_louder_is_likelier_deeper_is_harder():
    ears = _Ears()

    def wake_rate(db, depth):
        n = 0
        for i in range(400):
            p = {"pid": i, "name": f"s{i}", "alive": True, "awake": True,
                 "asleep": True, "sleep_depth": depth}
            ears.tick = 0
            for t in range(int(2.0 / TICK_S)):
                ears.tick = t
                if ears.noise_wakes(p, db):
                    n += 1
                    break
        return n / 400.0
    assert wake_rate(35.0, 0.6) < 0.02
    assert wake_rate(90.0, 0.6) > 0.98
    mid = wake_rate(60.0, 0.6)
    assert 0.3 < mid < 0.95, "around the old fixed line it is a chance"
    assert wake_rate(60.0, 0.9) < mid, "deeper sleep needs more"
    assert wake_rate(60.0, 0.2) > mid, "light sleep needs less"


def test_a_MINDS_sheet_without_body_numbers_does_not_break_the_injury_sheet():
    p = {"pid": 1, "name": "m", "sheet": {"harm": {"wounds": 1.0}, "traits": {}}}
    assert IJ.sheet_of(p)["mass_kg"] > 0
    IJ.head_blow(p, 150.0, 0)
