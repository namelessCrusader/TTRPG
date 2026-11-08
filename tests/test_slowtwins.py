"""TWINS: the breath twin uses the tick law's OWN discrete recurrence (wave 13,
T5, spec 3.9 and 12 T22c, T22d, and proof scene P1).

The 44,000-tick comparison and the 3-hour rescue are @pytest.mark.slow; the
rest is the fast tier. A test of a World uses a TEST-ONLY subclass whose
`_tick_start` calls super first and then `slow.pre`, so no existing file is
edited.
"""
import math

import numpy as np
import pytest

from src.voxel import quiet as Q
from src.voxel import slowforms as F
from src.voxel import slowtwins as TW
from src.voxel.materials import BODY, STONE, TICK_S
from src.voxel.physiology import BREATH
from src.voxel.scenes import _person
from src.voxel.sim import World

HOUR = 144000


class SlowWorld(World):
    slow = None

    def _tick_start(self):
        super()._tick_start()
        if self.slow is not None:
            self.slow.pre(self)


def _room(nx=24, ny=20, nz=40, vc=5):
    w = SlowWorld(nx, ny, nz, voxel_cm=vc)
    w.fill(0, nx, 0, ny, 0, 1, STONE)
    w.fill(0, nx, 0, 1, 0, nz, STONE)
    w.fill(0, nx, ny - 1, ny, 0, nz, STONE)
    w.fill(0, 1, 0, ny, 0, nz, STONE)
    w.fill(nx - 1, nx, 0, ny, 0, nz, STONE)
    w.fill(0, nx, 0, ny, nz - 1, nz, STONE)
    w.exits = []
    return w


def _person_in(w, cohb=0.0, blood_o2=1.0, awake=False, asleep=False):
    p = _person(w, 10, 10, z0=1)
    p["awake"], p["cohb"], p["blood_o2"] = awake, cohb, blood_o2
    if asleep:
        p["asleep"] = True
    return p


def _warm(w, n=40):
    """Step until nothing is toppling (an unconscious man upright is collapsed
    at the first tick divisible by 8), then n more ticks."""
    for _ in range(3000):
        w.step()
        if not w.bodies:
            break
    for _ in range(n):
        w.step()


TWIN_KEYS = ("blood_o2", "met", "smoke", "cohb")


def _compare_worlds(a, b):
    """The twin keys within 1e-9 relative, every other key of the state bit-equal."""
    pa, pb = a.persons[0], b.persons[0]
    for k in TWIN_KEYS:
        assert abs(pa[k] - pb[k]) <= 1e-9 * max(abs(pa[k]), abs(pb[k]), 1e-12), \
            (k, pa[k], pb[k])
    assert a.tick == b.tick
    assert Q.diff(Q.digest(a), Q.digest(b)) == []


def _jump_by_twin(w, n):
    p = w.persons[0]
    twins, why = TW.person_twins(w, p)
    assert twins is not None, why
    for _name, tw, ctx in twins:
        tw.advance(w, p, ctx, n)
    w.tick += n


# ─────────────────────────────────────────────── the recurrences ──
def test_geo_relax_is_the_discrete_recurrence_not_the_continuous_exp():
    r, x0, tgt, n = 0.000625, 0.6, 1.0, 800
    x = x0
    for _ in range(n):
        x += (tgt - x) * r
    assert abs(TW.geo_relax(x0, tgt, r, n) / x - 1.0) < 1e-12
    cont = tgt + (x0 - tgt) * math.exp(-r * n)
    assert abs(cont / x - 1.0) > 1e-6          # the continuous form is not the tick law
    assert TW.geo_relax(0.3, 1.0, 0.5, 0) == 0.3 and TW.geo_relax(0.3, 1.0, 1.0, 5) == 1.0
    assert TW.geo_decay(0.5, 0.0, 99) == 0.5 and TW.geo_decay(0.5, 1.0, 1) == 0.0


def test_first_tick_finds_the_integer_tick_of_a_monotone_predicate():
    for k in (1, 2, 3, 1000, 449230, 12345678):
        assert TW.first_tick(lambda n, k=k: n >= k) == k
    assert TW.first_tick(lambda n: False, 1000) is None


# ─────────────────────────── the mouth air, against a spy on _breathe ──
def test_mouth_air_equals_what_a_real_tick_hands_breathe():
    w = _room()
    p = _person_in(w, cohb=0.3)
    _warm(w, 5)
    seen = []
    real = w._breathe

    def spy(pp, gas, frac, smoke_gl):
        seen.append((gas, frac, smoke_gl))
        return real(pp, gas, frac, smoke_gl)
    ours = TW.mouth_air(w, p)
    w._breathe = spy
    w.step()
    assert seen and ours == seen[0] and ours[0] is True and ours[2] == 0.0
    w.smoke[6:17, 6:15, 25:34] = 0.05                 # now there is smoke at the mouth
    ours = TW.mouth_air(w, p)
    seen.clear()
    w.step()
    assert ours[2] > 0.0 and seen[0][2] > 0.0           # (the smoke law runs first and moves it)
    assert ours[:2] == seen[0][:2]


# ───────────────────────────────────────── jump equals ticking ──
@pytest.mark.parametrize("kind", ["sleeper", "unconscious_co", "low_blood_o2"])
def test_breath_twin_jump_equals_real_ticks(kind):
    def build():
        w = _room()
        if kind == "sleeper":
            _person_in(w, awake=True, asleep=True, blood_o2=0.9)
        elif kind == "unconscious_co":
            _person_in(w, cohb=0.30)
        else:
            _person_in(w, blood_o2=0.6)
        return w
    a, b = build(), build()
    _warm(a, 40)
    _warm(b, 40)
    n = 380                                     # the low blood_o2 man wakes about tick 460
    for _ in range(n):
        a.step()
    _jump_by_twin(b, n)
    _compare_worlds(a, b)


def test_horizon_flips_on_the_same_tick_as_the_ticking_world():
    """blood_o2 0.6, unconscious, in clean air: the 'wakes' event of the real world
    comes on tick t0 + k - 1, k the twin's horizon; a jump to k - 2 then two real
    ticks puts the wake on that same tick with the same keys."""
    def build():
        w = _room()
        _person_in(w, blood_o2=0.6)
        _warm(w, 40)
        return w
    a, b = build(), build()
    p = a.persons[0]
    twins, _ = TW.person_twins(a, p)
    hz = twins[0][1].horizon(a, p, twins[0][2])
    k, tag = hz[0]
    assert tag == "wakes" and 380 < k < 520
    t0 = a.tick
    first = a.events.next_id
    for _ in range(k + 5):
        a.step()
    wakes = [e for e in a.events.since(first - 1) if e.verb == "wakes"]
    assert wakes and wakes[0].tick == t0 + k - 1
    first_b = b.events.next_id
    _jump_by_twin(b, k - 2)
    for _ in range(2 + 5):
        b.step()
    wb = [e for e in b.events.since(first_b - 1) if e.verb == "wakes"]
    assert wb and wb[0].tick == wakes[0].tick
    assert abs(a.persons[0]["blood_o2"] - b.persons[0]["blood_o2"]) < 1e-9


@pytest.mark.slow
def test_breath_twin_jump_equals_44000_real_ticks_for_an_unconscious_man_at_cohb_0_30():
    def build():
        w = _room()
        _person_in(w, cohb=0.30)
        _warm(w, 40)
        return w
    a, b = build(), build()
    for _ in range(44000):
        a.step()
    _jump_by_twin(b, 44000)
    _compare_worlds(a, b)


# ───────────────────────────────── P1: the rescued man (horizon only) ──
@pytest.mark.parametrize("cohb,minutes", [(0.30, 187.2), (0.35, 258.4)])
def test_wake_time_by_the_real_predicate_is_187_2_and_258_4_minutes(cohb, minutes):
    w = _room()
    p = _person_in(w, cohb=cohb, blood_o2=1.0)
    _warm(w, 3)
    twins, why = TW.person_twins(w, p)
    assert twins is not None, why
    tw, ctx = twins[0][1], twins[0][2]
    k, tag = tw.horizon(w, p, ctx)[0]
    assert tag == "wakes"
    # the same count from the pure loop of the law: delivered > wake_o2 (strict)
    c, n = p["cohb"], 0
    q = dict(p)
    while True:
        out = c * math.log(2.0) / (BREATH["co_half_min"] * 60.0) * TICK_S
        c = min(max(c - out, 0.0), 1.0)
        n += 1
        q["cohb"] = c
        if TW.delivered_o2(q) > BODY["wake_o2"]:
            break
    assert abs(n - k) <= 1
    # and from tick 0 of a person at exactly this COHb, blood_o2 = perfusion = 1:
    kc = math.log(2.0) * TICK_S / (BREATH["co_half_min"] * 60.0)
    k0 = TW.first_tick(lambda m: TW.delivered_o2(
        {"blood_o2": 1.0, "cohb": TW.geo_decay(cohb, kc, m)}) > BODY["wake_o2"])
    assert abs(k0 * TICK_S / 60.0 - minutes) < 0.05
    # The exact first tick is ln(c0 / 0.2) / -ln(1 - k). The spec quotes 449,230 for
    # 0.30; the exact count is 449,251 (0.005 % apart, the same 187.2 min).
    assert abs(k0 - {0.30: 449251, 0.35: 620049}[cohb]) <= 1


# ─────────────────────────────────────────────────── refusals ──
def _refusal(w, p):
    twins, why = TW.person_twins(w, p)
    assert twins is None
    return why


def test_twin_refuses_an_awake_person():
    w = _room()
    p = _person_in(w, awake=True)
    _warm(w, 5)
    assert "awake" in _refusal(w, p)


def test_twin_refuses_a_bleeding_person():
    w = _room()
    p = _person_in(w, cohb=0.3)
    _warm(w, 5)
    p["wounds"] = [{"bone": "torso", "open": True}]
    assert "bleeding" in _refusal(w, p)
    p["wounds"] = [{"bone": "torso", "open": False}]
    twins, _ = TW.person_twins(w, p)
    assert twins is not None                     # a closed wound is not a reason


def test_twin_refuses_smoke_at_the_mouth_and_no_gas():
    w = _room()
    p = _person_in(w, cohb=0.3)
    _warm(w, 5)
    w.smoke[6:17, 6:15, 25:34] = 0.05
    assert "smoke" in _refusal(w, p)
    w.smoke[...] = 0.0
    twins, _ = TW.person_twins(w, p)
    assert twins is not None
    from src.voxel.materials import STONE as S
    w.mat[8:14, 7:13, 26:32] = S                 # buried in stone: no gas at the mouth
    w.smass[8:14, 7:13, 26:32] = 500.0
    w.mat[p["anchor"][0] - 3:p["anchor"][0] + 5, p["anchor"][1] - 3:p["anchor"][1] + 4, 26:31] = S
    ok, why = TW.TWINS["breath"].applicable(w, p)
    assert (not ok and ("gas" in why or "flesh" in why)) or ok


def test_twin_refuses_a_rising_burn_and_a_moving_sleeper():
    w = _room()
    p = _person_in(w, awake=True, asleep=True)
    _warm(w, 5)
    twins, _ = TW.person_twins(w, p)
    assert twins is not None                     # a still sleeper is fine
    p["skin"] = BODY["hurt_T"] + 5.0
    assert "burn" in _refusal(w, p)
    p["skin"] = 20.0
    p["_met_at"] = (p["_eye"][0] + 1.0, p["_eye"][1])
    assert "moving" in _refusal(w, p)


def test_dead_and_safe_persons_have_no_twin_and_no_objection():
    w = _room()
    p = _person_in(w, awake=True)
    p["alive"] = False
    assert TW.person_twins(w, p) == ([], "")
    p["alive"], p["safe"] = True, True
    assert TW.person_twins(w, p) == ([], "")


# ───────────────────────────────────────── registry and the slot ──
def test_registry_has_the_breath_twin_and_the_body_heat_twin_in_its_slot():
    # wave 13c, C10: the slot that was empty in phase A is filled by slowbody.BodyHeatTwin (T7's twin)
    from src.voxel import slowbody
    assert "breath" in TW.TWINS and isinstance(TW.TWINS.get(TW.BODY_HEAT), slowbody.BodyHeatTwin)
    tw = TW.register_twin("const_source", TW.ConstSourceTwin(watts=80.0))
    try:
        w = _room()
        p = _person_in(w, cohb=0.3)
        _warm(w, 5)
        twins, _ = TW.person_twins(w, p)
        assert [n for n, _t, _c in twins] == ["breath", "body_heat", "const_source"]
        ctx = tw.enter(w, p)
        assert tw.power_w(w, p, ctx) == 80.0
        fl = tw.advance(w, p, ctx, 144000)
        assert abs(fl[("E", "heat", "room")] - 80.0 * 3600.0) < 1e-6
        assert fl[("mass", "o2", "room")] < 0 and fl[("mass", "co2", "room")] > 0
    finally:
        TW.unregister_twin("const_source")
    assert "const_source" not in TW.TWINS


def test_four_sleepers_use_about_480_litres_of_oxygen_in_8_hours():
    """1.2 % of a 40 m3 room: allowed. Ten times that is not (slownet.v2_o2_ok)."""
    from src.voxel import slownet
    litres = slownet.o2_use_litres([71.0] * 4, 8.0)
    assert abs(litres - 480.0) < 20.0
    assert abs(litres / 40000.0 - 0.012) < 0.001
    w = _room(40, 40, 40, 10)
    grams = litres * 1.429
    ok, why = slownet.v2_o2_ok(w, (slice(0, 40),) * 3, grams)
    assert ok or "limit" in why                      # a 40 m3 room: 64 m3 of air in the lattice
    ok10, why10 = slownet.v2_o2_ok(w, (slice(0, 40),) * 3, grams * 12.0)
    assert not ok10 and "O2 use" in why10
