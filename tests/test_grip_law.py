"""GRIP LAW (wave 15, lane 2c): grip.py and its wiring in hands.py behind `grip_law`."""
import math

import pytest

from src.voxel import fatigue as FT
from src.voxel import grip as G
from src.voxel import matfric as MF
from src.voxel.scenes import _person, _Wants
from src.voxel.sim import LEAD, STONE, World

DRY_MU = 0.5
ARM = "right arm"


def _p(**kw):
    p = {"pid": 1, "name": "holder", "strength_N": 400.0, "exert_z": {"lift": 1.0}}
    p.update(kw)
    return p


# T1: a wet or oiled pair needs a harder squeeze; the oiled handle goes first (pure law)
def test_oiled_pair_slips_where_dry_holds():
    p = _p()
    mu_dry = DRY_MU
    mu_oil = MF.OILED_PAIR[0]
    assert G.holds(p, ARM, 300.0, mu_dry)
    assert not G.holds(p, ARM, 300.0, mu_oil)
    assert G.n_required(300.0, mu_oil) > G.n_required(300.0, mu_dry) * 4.0


def test_dry_skin_holds_exactly_the_old_strength_number():
    p = _p()
    assert G.holds(p, ARM, 399.9, G.MU_REF)
    assert not G.holds(p, ARM, 400.1, G.MU_REF)


# T2: a constant hold ends at ET(f) within 5 percent (the rested maximum sets the rate)
def test_endurance_of_a_constant_hold():
    p = _p()
    mu, load = 0.5, 150.0                     # N_req 150, N_use 195, f 0.4875
    f = G.SM * load / (mu * G.N_EFF) / 400.0
    want = FT.et_s(f)
    t, dt = 0.0, 0.05
    while t < 3 * want:
        r = G.step(p, ARM, load, mu, dt)
        t += dt
        if not r["holds"]:
            break
    assert t == pytest.approx(want, rel=0.05)


def test_wet_hands_tire_sooner():
    def endure(mu):
        p, t = _p(), 0.0
        while t < 2000.0:
            if not G.step(p, ARM, 150.0, mu, 0.1)["holds"]:
                return t
            t += 0.1
        return t
    assert endure(0.3) < endure(0.5)


# T3: slide heat is mu_k N s
def test_slide_heat_and_decel():
    assert G.slide_heat_J(0.3, 200.0, 1.0) == pytest.approx(60.0)
    # the work the hand's friction takes equals the heat (energy closes)
    m, F_t, N, muk = 5.0, 80.0, 100.0, 0.3
    a = G.slide_accel(F_t, muk, N, m)
    s = 0.5 * a * 1.0 ** 2
    assert F_t * s - m * (a * 1.0) ** 2 / 2 == pytest.approx(G.slide_heat_J(muk, N * G.N_EFF, s))


# T4: a cut tendon cannot hold; a half cut holds half
def test_tendon_cut():
    whole, half, cut = _p(), _p(body={"tendon_cut": {ARM: 0.5}}), _p(body={"tendon_cut": {ARM: 1.0}})
    assert G.holds(whole, ARM, 300.0, DRY_MU)
    assert G.capacity(cut, ARM, 0.1) == 0.0
    assert not G.holds(cut, ARM, 1.0, DRY_MU)
    assert G.capacity(half, ARM, 0.1) == pytest.approx(0.5 * G.capacity(whole, ARM, 0.1))
    assert not G.holds(half, ARM, 300.0, DRY_MU)


# T5: a hang fails when its time passes ET of the body's weight over G_hand
def test_a_hanging_body_ends_at_its_endurance():
    p = _p(strength_N=900.0, exert_z={"lift": 1.0})
    t, dt, load = 0.0, 0.05, 686.0
    while G.step(p, ARM, load, DRY_MU, dt)["holds"] and t < 600:
        t += dt
    f = G.SM * load / (DRY_MU * G.N_EFF) / 900.0
    assert t == pytest.approx(FT.et_s(min(f, 1.0)), rel=0.1)
    assert t < 60.0


# T6: capstan
def test_capstan_wraps():
    one = G.capstan_hold(1000.0, 0.4, 2 * math.pi)
    two = G.capstan_hold(1000.0, 0.4, 4 * math.pi)
    assert one == pytest.approx(1000.0 * math.exp(-0.4 * 2 * math.pi))
    assert two < one < 1000.0
    p = _p()
    assert not G.holds(p, ARM, 3000.0, 0.4)
    assert G.holds(p, ARM, 3000.0, 0.4, wraps_rad=4 * math.pi)


# T8: off hand 0.8; the person's own muscle sets the number
def test_off_hand_and_dials():
    p = _p()
    assert G.hand_rested_N(p, "left arm") == pytest.approx(0.8 * G.hand_rested_N(p, ARM))
    strong = _p(strength_N=800.0)
    assert G.hand_rested_N(strong, ARM) == pytest.approx(2.0 * G.hand_rested_N(p, ARM))
    trained = _p(training=1.0)
    assert G.hand_rested_N(trained, ARM) > G.hand_rested_N(p, ARM)
    assert G.holds(strong, ARM, 600.0, DRY_MU) and not G.holds(p, ARM, 600.0, DRY_MU)


def test_a_tired_hand_holds_less_and_rest_restores_it():
    p = _p()
    for _ in range(400):
        G.step(p, ARM, 150.0, 0.5, 0.1)
    gam = p["body"]["fat"][ARM]["gam"]
    assert 0.0 < gam < 1.0
    for _ in range(1500):
        G.step(p, ARM, 0.0, 0.5, 1.0)
    assert p["body"]["fat"][ARM]["gam"] < gam * 0.01


# ---- the world: the gate and its two outcomes ----------------------------------------------
def _carry(gate, oil=False, ticks=120, awake=True):
    w = World(52, 12, 90, voxel_cm=5)
    w.grip_law = gate
    w.fill(0, 52, 0, 12, 0, 1, STONE)
    w.fill(16, 52, 0, 12, 1, 41, STONE)
    w.exits = []
    a = _person(w, 20, 6, z0=41)
    a["name"], a["facing"], a["strength_N"] = "holder", (1.0, 0.0), 400.0
    w.fill(26, 29, 5, 8, 41, 43, LEAD)
    w.policy = _Wants(each={"holder": {"hands": "take hold of the lead"}})
    for t in range(ticks):
        if t == 40:
            if oil:
                cl = w._held_cells(a)
                w.fvol[tuple(cl.T)] = 0.5 * w.cap
                w.fl[tuple(cl.T)] = 2
            if not awake:
                a["awake"] = False
        w.step()
    return w, a


def test_gate_off_is_the_old_law_and_writes_no_fatigue():
    w, a = _carry(False)
    assert a.get("held") is not None
    assert not (a.get("body") or {}).get("fat", {}).get("right arm")


def test_gate_on_dry_holds_and_the_hand_tires():
    w, a = _carry(True)
    assert a.get("held") is not None
    assert a["body"]["fat"]["right arm"]["gam"] > 0.0


def test_gate_on_oiled_thing_slips_out_of_the_hand():
    w, a = _carry(True, oil=True)
    assert any("slips from the hand" in e or "grip fails" in e for e in a["events"])


def test_gate_on_an_unconscious_holder_drops():
    w, a = _carry(True, awake=False)
    assert any("grip fails" in e for e in a["events"])
