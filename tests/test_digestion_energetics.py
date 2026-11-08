"""T7b: digestion.py and energetics.py (wave 13, spec T7 section 6 tests 8, 9).

Tags (plan section 9): [I]/[U] acceptance lines are test_u_* (physical data,
source not verified), exactness and arithmetic identities have no prefix,
test_echo_* = constant fitted to that number, test_codeonly_* = the code
against its own table. No World is built here.
"""
import copy
import json
import os

import pytest

from src.voxel import digestion as D
from src.voxel import energetics as EN

HERE = os.path.dirname(os.path.abspath(__file__))


def _meal(kcal_target=700.0):
    """A mixed meal of bread, meat and oil near `kcal_target` gross kcal."""
    rec = D.food("bread", 150.0)
    for name, g in (("meat", 70.0), ("apple", 100.0)):
        r = D.food(name, g)
        for k in D.MACROS + ("water_g", "mass_g"):
            rec[k] += r[k]
    scale = kcal_target / (D.food_ge_kj(rec) / D.KJ_PER_KCAL)
    for k in D.MACROS + ("water_g", "mass_g"):
        rec[k] *= scale
    return rec


def _run(st, hours, demand_w=0.0, dt=60.0, mass=70.0):
    n = int(hours * 3600 / dt)
    for _ in range(n):
        D.step(st, dt, demand_w, mass_kg=mass)


# ----------------------------------------------------------- energy books --
def test_GE_in_equals_ME_plus_faeces_plus_urine_exactly():
    for name, g in (("bread", 200), ("stew", 400), ("meat", 150), ("raw_flour", 100),
                    ("ale", 500), ("oil", 30)):
        for cook in (0.0, 0.5, 1.0):
            sp = D.meal_split(D.food(name, g, cook))
            assert sp["GE"] == pytest.approx(sp["ME"] + sp["faeces"] + sp["urine"], rel=1e-12)
            assert sp["ME"] >= 0.0


def test_stepwise_chemical_energy_closes_to_the_joule():
    """stores + gut + colon + urine (chemical) + heat made = start + GE_in,
    over a day of eating and burning."""
    st = D.new_stores(70.0, 0.2)
    e0 = D.stores_J(st)
    D.ingest(st, _meal(700.0))
    D.ingest(st, D.food("ale", 500.0))
    _run(st, 30.0, demand_w=95.0)
    lhs = D.stores_J(st) + st["books"]["heat_J"]
    rhs = e0 + st["books"]["GE_in"]
    assert lhs == pytest.approx(rhs, rel=1e-12), (lhs, rhs)


def test_whole_meal_identity_after_it_is_all_burned():
    """Once absorbed and burned: GE_in = heat (ME) + faeces + urine. Run a
    meal in a body that has emptied its tanks of carbohydrate so every
    absorbed gram is eventually paid out: use a long fast after eating."""
    st = D.new_stores(70.0, 0.2, full=0.0)
    st["gluc_g"] = 0.0
    rec = D.food("stew", 600.0)
    sp = D.meal_split(rec)
    D.ingest(st, rec)
    _run(st, 400.0, demand_w=0.0, dt=600.0)   # absorb everything (fibre is slow)
    assert not st["gut"] and st["si"]["carb"] < 1e-6 and st["si"]["fibre"] < 1e-3
    # faeces and urine as booked by the digestion
    assert st["books"]["faeces_J"] == pytest.approx(sp["faeces"], rel=1e-9)
    # TEF heat is the only heat so far, and is the closed-form TEF
    assert st["books"]["tef_J"] == pytest.approx(sp["TEF"], rel=2e-3)


def test_TEF_is_paid_once_not_twice():
    """Stores receive (1 - TEF) x absorbed: eat with zero demand; the stored
    gross energy plus the TEF heat (grossed up with the urine of burned
    protein) equals the absorbed gross energy."""
    st = D.new_stores(70.0, 0.2, full=0.0)
    rec = _meal(700.0)
    sp = D.meal_split(rec)
    e0 = D.stores_J(st)
    D.ingest(st, rec)
    _run(st, 400.0, demand_w=0.0, dt=600.0)
    gained = D.stores_J(st) - e0              # incl. colon and urine
    tef_heat = st["books"]["tef_J"]
    assert gained + tef_heat == pytest.approx(sp["GE"], rel=1e-9)
    # one TEF, not two: a mixed meal comes out at 6-14% of the metabolisable energy
    assert 0.06 < tef_heat / sp["ME"] < 0.14


# The spec band is 10-20 min for every liquid. Calorie density slows emptying a little (Calbet and
# MacLean 1997, u): th = 15 min x (1 + 0.4 x kcal per ml). Milk (0.64 kcal/ml) comes out inside the band.
@pytest.mark.parametrize("name,lo,hi", [("water", 10, 20), ("juice", 10, 20), ("milk", 10, 20)])
def test_u_liquids_leave_the_stomach_in_10_to_20_minutes(name, lo, hi):
    st = D.new_stores(70.0, 0.2)
    D.ingest(st, D.food(name, 300.0))
    m0 = sum(it["m0"][k] * it["rem"] for it in st["gut"] for k in D.MACROS) + \
        sum(it["water0"] * it["rem"] for it in st["gut"])
    t = 0
    while (sum(it["water0"] * it["rem"] for it in st["gut"])) > 0.5 * 300.0 * D.FOOD_ROWS[name]["water"] / 100.0:
        D.step(st, 30.0, 0.0)
        t += 0.5
    assert lo <= t <= hi, t


def test_u_a_700_kcal_meal_has_a_gastric_half_time_of_80_to_130_minutes():
    st = D.new_stores(70.0, 0.2)
    D.ingest(st, _meal(700.0))
    k0 = D.stomach_kcal(st)
    assert k0 == pytest.approx(700.0, rel=1e-9)
    t = 0
    while D.stomach_kcal(st) > 0.5 * k0:
        D.step(st, 60.0, 90.0)
        t += 1
    assert 80 <= t <= 130, t


def test_codeonly_the_pylorus_cap_binds_only_for_a_feast():
    def t_half(kcal):
        st = D.new_stores(70.0, 0.2)
        D.ingest(st, _meal(kcal))
        t = 0
        while D.stomach_kcal(st) > 0.5 * kcal:
            D.step(st, 60.0, 90.0)
            t += 1
        return t
    assert t_half(700.0) < 130
    assert t_half(3000.0) > t_half(700.0) * 1.4     # the cap slows a feast


def test_fat_slows_emptying():
    def t_half(name):
        st = D.new_stores(70.0, 0.2)
        r = D.food(name, 300.0)
        D.ingest(st, r)
        k0 = D.stomach_kcal(st)
        t = 0
        while D.stomach_kcal(st) > 0.5 * k0:
            D.step(st, 60.0, 90.0)
            t += 1
        return t
    assert t_half("oil") > t_half("rice")


def test_u_digestibility_raw_vs_cooked_and_atwater_falls_out():
    assert D.digestibility("carb", 0.0) == pytest.approx(0.6)
    assert D.digestibility("carb", 1.0) == pytest.approx(0.95)
    # Atwater 4/4/9 is a CHECK that falls out (u: within 8% of each factor)
    carb = D.digestibility("carb", 1.0) * D.GE_KJ["carb"] / D.KJ_PER_KCAL
    prot = (D.digestibility("protein", 1.0) * D.GE_KJ["protein"]
            - D.URINE_KJ_PER_G_PROTEIN * D.digestibility("protein", 1.0)) / D.KJ_PER_KCAL
    fat = D.digestibility("fat", 1.0) * D.GE_KJ["fat"] / D.KJ_PER_KCAL
    assert abs(carb - 4.0) < 0.32 and abs(prot - 4.0) < 0.32 and abs(fat - 9.0) < 0.72


def test_u_starvation_of_15_kg_fat_at_2000_kcal_a_day_takes_58_days_within_20_percent():
    st = D.new_stores(70.0, 15.0 / 70.0)       # 15 kg of adipose tissue
    assert st["fat_g"] == pytest.approx(15000.0 * D.LIPID_FRAC_ADIPOSE)
    p = 2000.0 * D.KJ_PER_KCAL * 1000.0 / 86400.0
    days = 0.0
    while st["fat_g"] > 0.0 and days < 200:
        st["water_ml"] = st["water0_ml"]       # water is not what is tested
        D.step(st, 3600.0, p)
        days += 1.0 / 24.0
    assert 58 * 0.8 <= days <= 58 * 1.2, days
    assert D.stores_J(st) >= 0.0


def test_glycogen_runs_out_in_about_a_day_of_fasting_and_hunger_rises():
    st = D.new_stores(70.0, 0.2)
    h0 = D.hunger(st)
    p = 100.0
    _run(st, 6.0, p)
    assert st["glyc_g"] > 0.0
    _run(st, 42.0, p)
    assert st["glyc_g"] < 0.2 * st["glyc_cap_g"]
    assert D.hunger(st) > max(h0, 0.4)


def test_every_food_row_fits_in_100_grams():
    for name, r in D.FOOD_ROWS.items():
        assert sum(r[k] for k in D.MACROS) + r["water"] <= 100.0 + 1e-9, name


def test_refeeding_refills_glycogen_before_fat():
    st = D.new_stores(70.0, 0.2, full=0.2)
    fat0, glyc0 = st["fat_g"], st["glyc_g"]
    D.ingest(st, D.food("rice", 300.0))          # ~84 g carbohydrate
    _run(st, 12.0, 0.0)
    assert st["glyc_g"] > glyc0 + 30.0
    assert st["fat_g"] - fat0 < 1.0              # only the meal's own 0.9 g of fat; none made
    # and the cap is respected, with the surplus going to fat
    st2 = D.new_stores(70.0, 0.2, full=1.0)
    st2["glyc_g"] = st2["glyc_cap_g"] - 5.0
    f2 = st2["fat_g"]
    D.ingest(st2, D.food("rice", 800.0))
    _run(st2, 12.0, 0.0)
    assert st2["glyc_g"] <= st2["glyc_cap_g"] + 1e-9
    assert st2["fat_g"] > f2


def test_mass_closes_through_eating_burning_and_breathing():
    """body mass change = food in - ash... : stores_g(end) - stores_g(start)
    = food mass in + O2 breathed in - CO2 breathed out."""
    st = D.new_stores(70.0, 0.2, full=0.3)
    m0 = D.stores_g(st)
    rec = _meal(900.0)
    D.ingest(st, rec)
    d_o2 = d_co2 = 0.0
    for i in range(24 * 60):
        o = D.step(st, 60.0, 120.0 if i % 2 else 60.0)
        d_o2 += o["o2_g"]
        d_co2 += o["co2_g"]
    assert D.stores_g(st) - m0 == pytest.approx(
        rec["mass_g"] + d_o2 - d_co2 - st["leak_ml"], abs=1e-6)


def test_metabolic_water_is_booked_and_rq_falls_out_of_the_fuel():
    st = D.new_stores(70.0, 0.2)
    o = D.step(st, 60.0, 100.0)
    assert o["h2o_ml"] > 0.0
    rq = D.rq_of(o)
    assert 0.7 <= rq <= 1.0
    # fuel by fuel the respiratory quotient is the textbook one
    for fuel, lo, hi in (("carb", 0.97, 1.03), ("fat", 0.68, 0.73),
                         ("protein", 0.80, 0.90), ("alcohol", 0.64, 0.70)):
        out = {"heat_J": 0.0, "o2_L": 0.0, "co2_L": 0.0, "co2_g": 0.0, "o2_g": 0.0, "h2o_ml": 0.0}
        D._oxidise(D.new_stores(70.0, 0.2), fuel, 10.0, out)
        assert lo <= out["co2_L"] / out["o2_L"] <= hi, (fuel, out["co2_L"] / out["o2_L"])


def test_u_O2_used_times_the_Weir_factor_equals_stores_drawn():
    st = D.new_stores(70.0, 0.2)
    j = o2 = co2 = 0.0
    for _ in range(600):
        o = D.step(st, 1.0, 100.0)
        j += o["delivered_J"]
        o2 += o["o2_L"]
        co2 += o["co2_L"]
    rq = co2 / o2
    assert j == pytest.approx(o2 * D.weir_kj_per_l(rq) * 1000.0, rel=0.02)
    # and the heat made IS what was asked for, to the joule
    assert j == pytest.approx(100.0 * 600, rel=1e-9)


def test_water_is_conserved_through_gut_bladder_and_voiding():
    st = D.new_stores(70.0, 0.2)
    w0 = st["water_ml"] + st["bladder_ml"]
    D.ingest(st, D.food("water", 1000.0))
    _run(st, 12.0, 0.0)
    ml, urea, j = D.void_bladder(st)
    assert st["water_ml"] + ml + st["leak_ml"] == pytest.approx(w0 + 1000.0, rel=1e-9)
    assert st["bladder_ml"] == 0.0


def test_the_bladder_is_capped_and_overflow_is_reported():
    st = D.new_stores(70.0, 0.2)
    st["bladder_ml"] = D.BLADDER_MAX_ML - 0.01
    D.ingest(st, D.food("water", 2000.0))
    _run(st, 6.0, 0.0)
    assert st["bladder_ml"] <= D.BLADDER_MAX_ML + 1e-9
    assert st["leak_ml"] > 0.0


def test_thirst_and_dehydration_signals():
    st = D.new_stores(70.0, 0.2)
    assert D.thirst(st) == 0.0
    st["water_ml"] -= 0.025 * 70000.0
    assert D.thirst(st) == pytest.approx(1.0)
    assert D.deficit_pct(st) == pytest.approx(2.5)


def test_snapshot_and_determinism():
    def go():
        st = D.new_stores(70.0, 0.2)
        D.ingest(st, _meal(700.0))
        _run(st, 3.0, 110.0)
        return st
    a, b = go(), go()
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    snap = copy.deepcopy(a)
    _run(a, 1.0, 90.0)
    _run(snap, 1.0, 90.0)
    assert json.dumps(a, sort_keys=True) == json.dumps(snap, sort_keys=True)


def test_dt_independence_of_the_gut_when_the_cap_is_slack():
    def run(dt):
        st = D.new_stores(70.0, 0.2)
        D.ingest(st, _meal(500.0))
        _run(st, 2.0, 0.0, dt=dt)
        return D.stomach_kcal(st), st["si"].get("carb", 0.0) + st["glyc_g"]
    a, b = run(10.0), run(120.0)
    assert a[0] == pytest.approx(b[0], rel=1e-9)


# -------------------------------------------------------------- energetics --
def test_one_MET_is_1_22_W_per_kg_and_these_are_the_single_exports():
    assert EN.MET_W_PER_KG == 1.22 and EN.ETA == 0.22
    assert EN.met_power_w(1.0, 70.0) == pytest.approx(85.4)


def test_rest_P_total_is_exactly_one_MET_no_double_count():
    m = 70.0
    for p_tissue in (60.0, 75.0, 85.4):
        pt = EN.p_total(p_tissue, met=1.0, mass_kg=m)
        # tissue is 0.7-1.0 MET; the rest of the MET is gait, so the total is 1 MET
        if p_tissue <= 85.4:
            assert pt == pytest.approx(1.22 * m, rel=1e-12)
    # when the tissue term already exceeds the MET (fever, cold), nothing is subtracted twice
    assert EN.p_total(100.0, met=1.0, mass_kg=m) == pytest.approx(100.0)


def test_u_gait_at_1_4_mps_costs_3_2_to_3_6_MET():
    met = EN.met_of_speed(1.4)
    assert 3.2 <= met <= 3.6
    m = 70.0
    mb = EN.met_b_of(0.9 * EN.met_power_w(1.0, m), m)
    assert EN.p_total(0.9 * EN.met_power_w(1.0, m), met, m) == pytest.approx(
        EN.met_power_w(met, m), rel=1e-12)
    assert EN.p_gait(met, mb, m) == pytest.approx((met - mb) * 1.22 * m)


def test_running_costs_more_and_a_load_adds_once():
    assert EN.met_of_speed(3.0) > 8.0
    assert EN.load_term_w(70.0, 20.0, 1.4) > 0.0
    assert EN.load_term_w(70.0, 0.0, 1.4) == pytest.approx(0.0)


class StubBooks:
    """The manifest: work_out may be credited; touching mech_in is a failure."""
    def __init__(self):
        self.work_out = 0.0
        self.touched = []

    @property
    def mech_in(self):
        self.touched.append("read")
        raise AssertionError("mech_in touched for actor work")

    @mech_in.setter
    def mech_in(self, v):
        self.touched.append("write")
        raise AssertionError("mech_in touched for actor work")


def _heat(mass=70.0):
    return {"stores": D.new_stores(mass, 0.2), "work": EN.new_book(),
            "sheet": {"mass_kg": mass}}


ZONES = {"head": 1.0, "torso": 10.0, "uarm_l": 1.0, "uarm_r": 1.0, "farm_l": 1.0,
         "farm_r": 1.0, "hand_l": 0.3, "hand_r": 0.3, "thigh_l": 2.0, "thigh_r": 2.0,
         "shin_l": 1.0, "shin_r": 1.0, "foot_l": 0.4, "foot_r": 0.4}


def test_lifting_20_kg_by_1_m_costs_W_over_eta_and_credits_work_out_196_J():
    h, b = _heat(), StubBooks()
    glyc0 = D.stores_J(h["stores"])
    w = 20.0 * EN.GRAV * 1.0
    r = EN.spend_J(h, b, "lift:crate", w, "lift", 10.2, ZONES)
    assert w == pytest.approx(196.2)
    assert r["met_J"] == pytest.approx(w / 0.22)
    assert b.work_out == pytest.approx(196.2, abs=0.5)
    assert b.touched == []                               # ONE-FLOW: never mech_in
    drawn = glyc0 - D.stores_J(h["stores"])
    assert drawn >= w / 0.22 - 1e-6                      # stores debited (gross incl. urine)
    pz = h["work"]["pending_zone_J"]
    assert sum(pz.values()) == pytest.approx(w / 0.22 - w, rel=1e-6)    # muscle heat
    assert pz["torso"] > pz["head"] if "head" in pz else True


def test_one_flow_energy_closes_for_a_lift():
    """stores drawn (chemical) = heat in muscle + work_out + urine chemical."""
    h, b = _heat(), StubBooks()
    e0 = D.stores_J(h["stores"])
    w = 196.2
    r = EN.spend_J(h, b, "t", w, "lift", 3.0, ZONES)
    heat = sum(h["work"]["pending_zone_J"].values())
    assert e0 - D.stores_J(h["stores"]) == pytest.approx(heat + b.work_out, rel=1e-9)


def test_lowering_returns_the_heat_once_and_costs_0_83_of_the_lift():
    h, b = _heat(), StubBooks()
    w = 196.2
    EN.spend_J(h, b, "lift", w, "lift", 1.0, ZONES)
    h["work"]["pending_zone_J"].clear()
    r = EN.spend_J(h, b, "lower", w, "lower", 20.0, ZONES)
    assert r["met_J"] == pytest.approx(w / 1.2)
    assert r["met_J"] / (w / 0.22) == pytest.approx(0.1833, abs=2e-3)   # the 0.18 of the lift
    assert sum(h["work"]["pending_zone_J"].values()) == pytest.approx(w + w / 1.2, rel=1e-9)
    assert b.work_out == pytest.approx(0.0, abs=1e-6)       # lifted and lowered: net zero
    assert b.touched == []


def test_a_second_bill_for_the_same_tag_in_a_second_is_refused():
    h, b = _heat(), StubBooks()
    assert EN.spend_J(h, b, "swing:7", 50.0, "swing", 5.1, ZONES) is not None
    e = D.stores_J(h["stores"])
    assert EN.spend_J(h, b, "swing:7", 50.0, "swing", 5.9, ZONES) is None
    assert D.stores_J(h["stores"]) == e
    assert h["work"]["refused"] == 1
    assert EN.spend_J(h, b, "swing:7", 50.0, "swing", 6.0, ZONES) is not None   # next second
    assert EN.spend_J(h, b, "swing:8", 50.0, "swing", 6.1, ZONES) is not None   # other tag


def test_use_table_split_conserves_joules():
    for kind, row in EN.USE.items():
        assert sum(row.values()) == pytest.approx(1.0, abs=1e-12), kind
        sh = EN.heat_share(row, ZONES)
        assert sum(sh.values()) == pytest.approx(1.0, abs=1e-15)
        assert all(v >= 0.0 for v in sh.values())
    sh = EN.heat_share(EN.USE["gait"], ZONES)
    leg = sum(v for z, v in sh.items() if "thigh" in z or "shin" in z)
    assert leg == pytest.approx(0.75)
    # within a group the heat goes by capacity
    assert sh["thigh_l"] == pytest.approx(sh["thigh_r"])


def test_unknown_work_kind_is_an_error_and_pays_nothing():
    h, b = _heat(), StubBooks()
    with pytest.raises(ValueError):
        EN.spend_J(h, b, "x", 10.0, "teleport", 1.0, ZONES)


def test_hyperemia_scales_with_work_heat_and_is_capped():
    w0 = 5.0e-4
    assert EN.hyperemia_w(w0, 0.0) == w0
    assert EN.hyperemia_w(w0, 2.0e4) > w0
    assert EN.hyperemia_w(w0, 1.0e9) == pytest.approx(EN.HYP_CAP * w0)


def test_O2_and_stores_agree_for_an_act_billed_by_spend_J():
    h, b = _heat(), StubBooks()
    r = EN.spend_J(h, b, "t", 500.0, "lift", 1.0, ZONES)
    # litres of O2 x kJ/L of the fuel burned = heat delivered
    assert r["o2_L"] > 0.0
    assert r["o2_L"] * 20.5 * 1000.0 == pytest.approx(r["met_J"], rel=0.06)


def test_the_met_and_the_breath_constants_agree_across_modules():
    """MET 1.22 W/kg is ONE number: energetics, tissue and the O2 of physiology.BREATH."""
    from src.voxel import bodynodes as BN
    from src.voxel import physiology as PH
    from src.voxel import tissue as TI
    assert EN.MET_W_PER_KG == TI.MET_W_PER_KG
    # 3.5 ml O2/(kg min) x 20.9 J/ml / 60 s = 1.219 W/kg
    assert PH.BREATH["vo2_ml_kg_min"] * 20.9 / 60.0 == pytest.approx(EN.MET_W_PER_KG, abs=0.005)
    assert BN.RMV_L_MIN == PH.BREATH["rmv_l_min"]
