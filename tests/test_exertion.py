"""Wave 14b lane G: exertion from the body (src/voxel/exertion.py, the gate
World.exertion_from_body in mechlaw.py). Numbers marked (u) in the module are
unverified; the calibration asserts the ordinary 115 J (decision 6) and the
one-in-a-hundred 215 J of Ramakrishnan et al. 2018."""
import math

import numpy as np
import pytest

from src.voxel import bodysheet as B
from src.voxel import exertion as X
from src.voxel import digestion as DG
from src.voxel import bodyheat as BH
from src.voxel import scenes
from src.voxel.mechlaw import MechanismsMixin
from src.voxel.sim import World

import tests.test_mechlaw as TM


def _ref(**kw):
    p = {"pid": 1, "name": "ref", "strength_N": 400.0, "skill": {k: 0.6 for k in ("kick", "punch", "shove", "throw")},
         "exert_z": {"kick": 1.0, "punch": 1.0, "shove": 1.0, "throw": 1.0}}
    p.update(kw)
    return p


def stock_arm(p):
    """Scene helper (wave 14bF): pin THIS person to the stock adult's arm and legs, the one
    the older scenes were calibrated on (middle technique, no spread of their own), keeping
    whatever `strength_N` the scene gave them as a LOAD knob (what they can lift and hold).
    A scene that wants a person with a muscle of their own does not call this."""
    n = float(p.get("strength_N", 400.0))
    p["skill"] = {k: 0.6 for k in ("kick", "punch", "shove", "throw")}
    p["exert_z"] = {"kick": 400.0 / n, "punch": 400.0 / n, "shove": 400.0 / n,
                    "throw": 400.0 / n, "lift": 1.0}
    return p


def test_the_ordinary_adult_kick_is_115_J():
    k = X.strike(_ref())
    assert k["E_J"] == pytest.approx(115.0, rel=1e-6)
    assert k["m_kg"] == pytest.approx(5.6)                    # 8 percent of 70 kg
    assert 0.5 * k["m_kg"] * k["v_ms"] ** 2 == pytest.approx(k["E_J"])


def test_the_population_median_spread_and_tail():
    v = X.population_kick(4000)
    n = len(v)
    assert v[n // 2] == pytest.approx(115.0, rel=0.04)
    assert 0.007 <= sum(x > 215.0 for x in v) / n <= 0.015     # about one in a hundred
    assert sum(x > 100.0 for x in v) / n > 0.6                 # most adults pass 100 J
    assert v[-1] > 215.0 and v[0] < 70.0                       # a real spread


def test_same_person_same_answer_and_people_differ():
    a = {"pid": 3, "name": "ann"}
    assert X.strike(a)["E_J"] == X.strike(dict(a))["E_J"]
    es = {X.strike({"pid": i, "name": "x"})["E_J"] for i in range(20)}
    assert len(es) == 20


def test_strength_training_and_skill_raise_it_with_diminishing_returns():
    e = lambda **kw: X.strike(_ref(**kw))["E_J"]
    assert e(strength_N=600.0) == pytest.approx(1.5 * e())
    gains = [e(training=t) for t in (0.0, 0.25, 0.5, 0.75, 1.0)]
    d = np.diff(gains)
    assert all(d > 0) and all(d[:-1] > d[1:])                  # up, ever less
    assert e(training=1.0) == pytest.approx(1.35 * e())
    sk = [e(skill={"kick": s}) for s in (0.2, 0.4, 0.6, 0.8, 1.0)]
    d = np.diff(sk)
    assert all(d > 0) and all(d[:-1] > d[1:])


def test_fatigue_injury_and_situation_lower_it():
    base = X.strike(_ref())["E_J"]
    tired = _ref(body={"fat": {"legs": {"gam": 0.6, "wp": 1.0}}})
    assert X.strike(tired)["E_J"] == pytest.approx(0.4 * base)
    empty = _ref(body={"fat": {"legs": {"gam": 0.0, "wp": 0.0}}})
    assert X.strike(empty)["E_J"] == pytest.approx(0.6 * base)       # critical power level
    assert X.strike(_ref(), situation=0.5)["E_J"] == pytest.approx(0.5 * base)
    from src.voxel import bonestate as BS
    sh = B.get_sheet("adult male")
    shin = BS.new_bone(sh, "right shin")
    shin["integrity"] = 0.0
    hurt = _ref(sheet=sh, body={"bones": {"right shin": shin}})
    healthy = _ref(sheet=sh)
    assert X.strike(hurt)["E_J"] < 0.2 * X.strike(healthy)["E_J"]    # a broken shin barely kicks


def test_a_heavier_body_swings_more_mass_slower_for_the_same_energy():
    light = X.strike(_ref(sheet=dict(B.ADULT_FEMALE), strength_N=400.0))
    heavy = X.strike(_ref(sheet=dict(B.ADULT_MALE), strength_N=400.0))
    assert heavy["m_kg"] > light["m_kg"]
    assert heavy["E_J"] == pytest.approx(light["E_J"])
    assert heavy["v_ms"] < light["v_ms"]


def test_every_act_comes_from_the_same_law():
    for kind in X.KINDS:
        k = X.strike(_ref(), kind)
        assert k["E_J"] == pytest.approx(400.0 * X.KINDS[kind]["stroke_m"])
    assert X.throw_speed_ms(_ref(), 0.15) <= X.THROW_V_MAX_MS
    assert X.throw_speed_ms(_ref(), 5.0) < X.throw_speed_ms(_ref(), 0.5)
    assert X.throw_speed_ms(_ref(training=1.0), 5.0) > X.throw_speed_ms(_ref(), 5.0)


# ---- the gate ---------------------------------------------------------------
def test_gate_off_is_the_old_100_J_bit_for_bit():
    assert MechanismsMixin.exertion_from_body is True       # on for every world since 14bF
    w, mid, p = TM._room(flaw=False)
    w.exertion_from_body = False
    m, v = w.kick_energy(p)
    assert (m, v) == (5.6, 6.0)


def test_gate_on_the_kick_is_the_persons_own():
    w, mid, p = TM._room(flaw=False)
    w.exertion_from_body = True
    m, v = w.kick_energy(p)
    assert 0.5 * m * v * v == pytest.approx(X.strike(p)["E_J"])
    r = w.kick(p, mid)
    assert r["side"] == "stop"
    assert p["_work_J"] == pytest.approx(X.strike(p)["E_J"], rel=1e-9)


class _BHDoor(BH.BodyHeatMixin, TM.DoorWorld):
    def step(self):
        TM.DoorWorld.step(self)
        self._law_body_heat()


def test_a_kick_is_paid_from_the_body_at_muscle_efficiency():
    from src.voxel import scenes_bodyheat as SB
    w, mid, p = TM._room(cls=_BHDoor, flaw=False)
    w.body_heat_on(p, segs=p["segs"], sheet=SB.stock_sheet())
    w.exertion_from_body = True
    w.step()
    E = X.strike(p)["E_J"]
    s0 = DG.stores_J(p["heat"]["stores"])
    wo0, mi0 = w.work_out, w.mech_in
    r = w.kick(p, mid)
    paid = s0 - DG.stores_J(p["heat"]["stores"])
    assert paid >= E / 0.22 - 1.0                       # the stores gave E / eta
    assert w.work_out - wo0 == pytest.approx(E, rel=1e-6)   # booked once, not twice
    assert w.mech_in == mi0                             # the actor paid, not nowhere
