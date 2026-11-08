"""Anatomy: regions, segment masses, the sheet, the draws, the member tables (src/voxel/anatomy.py, wave 13 T10).

Tags: test_echo_* checks a number the constants were FITTED to; test_codeonly_*
checks the code against its own table; test_u_* leans on an unverified number
(and says which status tests/data/sourced_numbers.json gave it); untagged
tests are exact (arithmetic identities, closed forms, conservation).
"""
import math
import zlib

import pytest

from src.voxel import anatomy as A

TISSUE = None
try:
    from src.voxel import tissue as TISSUE      # lane A-T7a; cross test skips when absent
except Exception:                               # noqa: BLE001 - any failure to import skips
    TISSUE = None


def _reflex_draw(p, key, lo, hi):
    """reflex._draw from the live repo, copied: the hash the draws must equal."""
    v = p.get(key)
    if v is not None:
        return float(v)
    u = (zlib.crc32(f"{key}|{p.get('pid')}|{p.get('name')}".encode()) & 0xFFFFFFFF) / 2.0 ** 32
    return float(lo + (hi - lo) * u)


def _adult(**kw):
    return A.make_sheet(mass_kg=70.0, height_m=1.75, draws=False, **kw)


# ── regions and masses ──────────────────────────────────────────────────────

def test_eleven_regions_and_winter_fractions_sum_to_one():
    assert len(A.REGIONS) == 11 and len(set(A.REGIONS)) == 11
    assert math.fsum(A.REGION_FRAC.values()) == pytest.approx(1.0, abs=1e-9)
    # Winter 1990 by segment, both sides: head + trunk + 2 x (arm + forearm + hand + thigh + shank + foot)
    W = A.WINTER
    tot = W["head"] + W["trunk"] + 2 * (W["upper arm"] + W["forearm"] + W["hand"]
                                         + W["thigh"] + W["shank"] + W["foot"])
    assert tot == pytest.approx(1.0, abs=1e-9)


def test_the_torso_is_split_into_chest_and_abdomen_by_55_percent():
    assert A.REGION_FRAC["chest"] / (A.REGION_FRAC["chest"] + A.REGION_FRAC["abdomen"]) \
        == pytest.approx(0.55, abs=1e-12)
    s = _adult()
    assert A.segment_mass(s, "torso") == pytest.approx(70.0 * 0.497, rel=1e-9)
    assert A.segment_mass(s, "left thigh") == pytest.approx(7.0, rel=1e-9)
    assert A.segment_mass(s, "thigh") == A.segment_mass(s, "right thigh")


def test_every_zone_maps_to_a_region_and_the_hand_and_foot_ride_in_their_bones():
    assert len(A.ZONES) == 14
    assert set(A.REGION_OF_ZONE) == set(A.ZONES)
    assert set(A.REGION_OF_ZONE.values()) <= set(A.REGIONS)
    assert A.REGION_OF_ZONE["left hand"] == "left forearm"
    assert A.REGION_OF_ZONE["right foot"] == "right shin"
    assert A.REGION_OF_ZONE["head"] == "head" and A.REGION_OF_ZONE["torso"] == "chest"
    assert A.region_of_zone("torso", 0.8) == "chest" and A.region_of_zone("torso", 0.2) == "abdomen"
    assert A.region_of_zone("left thigh", 0.9) == "left thigh"


@pytest.mark.skipif(TISSUE is None, reason="tissue.py (lane A-T7a) not importable")
def test_codeonly_zones_equal_the_body_heat_topic_zones():
    assert tuple(TISSUE.ZONES) == tuple(A.ZONES)
    for z in TISSUE.ZONES:
        assert A.REGION_OF_ZONE[z] in A.REGIONS


def test_du_bois_70kg_175cm_is_1_85():
    assert A.bsa_dubois(70.0, 175.0) == pytest.approx(1.85, abs=0.02)
    assert _adult()["bsa_m2"] == pytest.approx(A.bsa_dubois(70.0, 175.0), rel=1e-12)


# ── draws and the sheet ─────────────────────────────────────────────────────

def test_draws_are_deterministic_in_range_and_differ_between_names():
    a = A.draw(3, "Bran", "pain_scale", 0.8, 1.2)
    assert a == A.draw(3, "Bran", "pain_scale", 0.8, 1.2)
    assert 0.8 <= a <= 1.2
    vals = {A.draw(i, f"p{i}", "ko_scale", 0.85, 1.15) for i in range(40)}
    assert len(vals) == 40 and all(0.85 <= v <= 1.15 for v in vals)
    assert A.draw(3, "Bran", "pain_scale", 0.8, 1.2) != A.draw(3, "Mara", "pain_scale", 0.8, 1.2)


def test_the_draw_equals_reflex_draw_for_a_person_with_no_value_of_the_key():
    for pid, name in ((0, ""), (7, "Bran"), (31, "Mara of the Reach")):
        for key, lo, hi in (("bone_scale", 0.92, 1.08), ("ko_scale", 0.85, 1.15)):
            p = {"pid": pid, "name": name}
            assert A.draw(pid, name, key, lo, hi) == _reflex_draw(p, key, lo, hi)


def test_make_sheet_draw_ranges_and_neutral_person():
    s = A.make_sheet(mass_kg=70.0, height_m=1.75, pid=5, name="Ede")
    ratio = s["bone_scale"] / (70.0 / 70.0) ** (2.0 / 3.0)
    assert 0.92 <= ratio <= 1.08 and 0.8 <= s["pain_scale"] <= 1.2 and 0.85 <= s["ko_scale"] <= 1.15
    n = _adult()
    assert n["bone_scale"] == 1.0 and n["pain_scale"] == 1.0 and n["ko_scale"] == 1.0
    assert n["muscle_scale"] == 1.0


def test_the_sex_factor_is_applied_once_on_top_of_the_size_rule():
    s = A.make_sheet(mass_kg=60.0, height_m=1.62, sex="f", draws=False)
    size = (60.0 / 70.0) ** (2.0 / 3.0)
    assert s["sex"] == "female"
    assert s["bone_scale"] == pytest.approx(size * 0.90, rel=1e-12)         # exactly once
    assert abs(s["bone_scale"] / (size * 0.90) - 1.0) < 0.05                # the spec's 5 % bound
    # with the draw in, the sheet divided by its draw is still exactly size x 0.90
    d = A.make_sheet(mass_kg=60.0, height_m=1.62, sex="f", pid=9, name="Ede")
    assert d["bone_scale"] / A.draw(9, "Ede", "bone_scale", 0.92, 1.08) == pytest.approx(size * 0.90, rel=1e-12)


def test_the_stock_figure_is_the_default_and_has_bone_scale_0_73():
    s = A.make_sheet(draws=False)
    assert s["mass_kg"] == pytest.approx(43.9) and s["height_m"] == pytest.approx(1.5)
    assert s["bone_scale"] == pytest.approx(0.7327, abs=2e-4)


def test_make_sheet_reads_the_body_heat_sheet_and_age_lowers_bone():
    t7 = {"sex": "female", "mass_kg": 60.0, "height_m": 1.62, "fat_frac": 0.27,
          "age_y": 70.0, "bmr_scale": 1.0}
    s = A.make_sheet(t7, draws=False)
    assert s["mass_kg"] == 60.0 and s["fat_frac"] == 0.27
    size = (60.0 / 70.0) ** (2.0 / 3.0)
    assert s["bone_scale"] == pytest.approx(size * 0.90 * (1 - 0.05 * 2), rel=1e-12)
    assert t7 == {"sex": "female", "mass_kg": 60.0, "height_m": 1.62, "fat_frac": 0.27,
                  "age_y": 70.0, "bmr_scale": 1.0}, "the input is not mutated"


def test_only_mass_or_only_height_gives_the_other_by_cube_root():
    s = A.make_sheet(mass_kg=70.0, draws=False)
    assert s["height_m"] == pytest.approx(1.75)
    s2 = A.make_sheet(height_m=1.75, draws=False)
    assert s2["mass_kg"] == pytest.approx(70.0)


# ── member tables ───────────────────────────────────────────────────────────

def test_u_femur_M_fail_is_inside_the_241_to_408_N_m_of_the_composite_events_document():
    Mf = A.m_fail_adult("femur")
    assert 241.0 <= Mf <= 408.0
    assert Mf == pytest.approx(364.0, abs=1.0)
    # the other shafts
    assert A.m_fail_adult("tibia") == pytest.approx(290.0, rel=0.03)
    assert A.m_fail_adult("humerus") == pytest.approx(170.0, rel=0.03)
    assert A.m_fail_adult("forearm") == pytest.approx(70.0, rel=0.03)


def test_section_modulus_is_pi_over_32_Do_of_the_fourth_powers():
    Z = A.section_modulus_m3("femur")
    assert Z == pytest.approx(math.pi * (0.028 ** 4 - 0.014 ** 4) / (32 * 0.028), rel=1e-12)
    assert A.m_fail_adult("femur") == pytest.approx(A.SIGMA_B_PA * Z, rel=1e-12)
    assert A.section_modulus_m3("calcaneus") is None


def test_codeonly_every_member_names_a_real_region_and_a_bone_has_a_member():
    for name, row in A.MEMBERS.items():
        assert row["F_FAIL_N"] > 0 and row["crush_m"] > 0 and row["tau_wk"] > 0
        assert A.m_fail_adult(name) > 0
    for bone, m in A.MEMBER_OF_BONE.items():
        assert m in A.MEMBERS
    assert A.TAU_WEEKS["shin"] == 20.0 and A.TAU_WEEKS["thigh"] == 18.0
    assert A.TAU_WEEKS["upper arm"] == 12.0 and A.TAU_WEEKS["forearm"] == 10.0
    assert A.TAU_WEEKS["head"] == 16.0


def test_m_above_is_the_winter_table():
    assert (A.m_above("calcaneus"), A.m_above("tibia"), A.m_above("femur"),
            A.m_above("pelvis"), A.m_above("lumbar")) == (0.99, 0.96, 0.90, 0.68, 0.62)
    assert A.m_above("left shin") == 0.96 and A.m_above("left thigh") == 0.90


def test_codeonly_vessel_table_regions_and_skin_layers():
    regs = set()
    for v in A.VESSELS:
        regs.add(v["region"])
        assert v["kind"] in ("a", "v") and v["d_mm"] > 0 and v["Q_ml_s"] > 0
        assert v["region"] in A.D_ADULT_MM
    assert {"thigh", "shin", "upper arm", "forearm", "head", "chest"} <= regs
    assert A.VESSEL_OF["femoral artery"]["Q_ml_s"] == 15.0      # 0.9 L/min
    depths = [row[1] for row in A.SKIN_LAYERS]
    assert len(depths) == 8 and depths == sorted(depths) and depths[-1] == 10.0
    assert A.SKIN_LAYERS[1][0] == "basal" and A.SKIN_LAYERS[5][0] == "dermis_base"
    assert A.SKIN_LAYERS[5][1] == 2.0


def test_cavity_table_is_the_atls_midpoints():
    assert A.CAVITY_ML["left thigh"] == 1500.0 and A.CAVITY_ML["abdomen"] == 2500.0
    assert A.CAVITY_ML["right shin"] == 750.0 and A.CAVITY_ML["left upper arm"] == 600.0
    assert A.CAVITY_ML["left forearm"] == 400.0 and A.CAVITY_ML["chest"] == 125.0


# ── sourced numbers (lane A-S) ──────────────────────────────────────────────

def test_u_bone_loads_against_the_sourced_numbers_file():
    got = A.sourced_bone_loads()
    if not got:
        pytest.skip("tests/data/sourced_numbers.json absent: the bone loads stay (u)")
    seen = {m: (v["status"], v["kind"]) for m, v in got.items()}
    print("sourced bone loads, status seen:", seen)
    if "femur" in got:
        # a lower bound: the table value must not be BELOW what a source says is exceeded
        assert got["femur"]["kind"] == "bound"
        assert A.MEMBERS["femur"]["F_FAIL_N"] >= got["femur"]["value_N"] - 1e-9
    if "lumbar" in got:
        # the Jager regression at age 30: the table value is within 10 %
        assert A.MEMBERS["lumbar"]["F_FAIL_N"] == pytest.approx(got["lumbar"]["value_N"], rel=0.10)
    for missing in ("tibia", "calcaneus"):
        # not_found upstream: nothing may claim them as sourced
        if missing not in got:
            assert A.MEMBERS[missing]["F_FAIL_N"] > 0


def test_sourced_loader_never_raises_on_a_bad_path():
    assert A.sourced_bone_loads("/nonexistent/none.json") == {}
