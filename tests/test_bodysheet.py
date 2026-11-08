"""Body sheet (wave 13, lane G10): the one source of body numbers.

Tags (wave 13 rules): plain names are conservation / exactness / closed-form /
arithmetic identities and may gate a scene. test_u_* = independent but source
unverified. test_echo_* = the constant was fitted to that number.
test_codeonly_* = the code against its own table; never gates a scene.
Counts (test functions): identity/closed-form 18, u 5, echo 3, codeonly 3.
"""
import ast
import json
import math
import os
import zlib

import numpy as np
import pytest

from src.voxel import bodysheet as B

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src", "voxel", "bodysheet.py")


# ── sheets and Winter fractions ──────────────────────────────────────────────
def test_winter_fractions_sum_to_one():
    total = sum(B.WINTER[k] for k in B.CENTRAL) + 2 * sum(B.WINTER[k] for k in B.SIDED)
    assert abs(total - 1.0) < 1e-12


def test_segment_masses_sum_to_sheet_mass():
    for sh in (B.ADULT_MALE, B.ADULT_FEMALE, B.STOCK, B.get_sheet("stock", mass_kg=101.3)):
        masses = B.segment_masses(sh)
        assert len(masses) == 2 + 2 * 6
        assert abs(sum(masses.values()) - sh["mass_kg"]) < 1e-9


def test_adult_sheets_store_exact_values():
    assert (B.ADULT_MALE["mass_kg"], B.ADULT_MALE["height_m"]) == (73.0, 1.75)
    assert (B.ADULT_FEMALE["mass_kg"], B.ADULT_FEMALE["height_m"]) == (60.0, 1.62)
    assert B.get_sheet("adult male")["mass_kg"] == 73.0
    assert B.get_sheet("female")["height_m"] == 1.62
    for sh in (B.ADULT_MALE, B.ADULT_FEMALE, B.STOCK):
        assert {"sex", "mass_kg", "height_m", "fat_frac", "age_y", "bmr_scale"} <= set(sh)
        assert 0.05 < sh["fat_frac"] < 0.5 and sh["age_y"] > 0


def test_get_sheet_returns_a_copy():
    s = B.get_sheet("stock")
    s["mass_kg"] = 1.0
    assert B.STOCK["mass_kg"] == 43.9
    assert B.sheet_of({})["mass_kg"] == 43.9 and B.sheet_of(None)["mass_kg"] == 43.9
    assert B.sheet_of({"sheet": B.ADULT_FEMALE})["mass_kg"] == 60.0


def test_u_winter_table_values():
    """The published table, as the specs quote it (head and neck .081, trunk
    .497, thigh .100, shank+foot .061, forearm+hand .022)."""
    assert B.WINTER_PCT == {"head": 8.1, "trunk": 49.7, "upper arm": 2.8,
                            "forearm": 1.6, "hand": 0.6, "thigh": 10.0,
                            "shank": 4.65, "foot": 1.45}
    sh = B.ADULT_MALE
    assert B.bone_mass(sh, "forearm") == pytest.approx(0.022 * 73.0)
    assert B.bone_mass(sh, "left shin") == pytest.approx(0.061 * 73.0)
    assert B.segment_mass(sh, "torso") == pytest.approx(0.497 * 73.0)


def test_segment_names_and_sides():
    sh = B.ADULT_MALE
    assert B.segment_mass(sh, "left thigh") == B.segment_mass(sh, "right_thigh") \
        == B.segment_mass(sh, "thigh")
    assert B.segment_mass(sh, "shin") == B.segment_mass(sh, "shank")
    assert B.segment_mass(sh, "torso") == B.segment_mass(sh, "trunk")
    with pytest.raises(KeyError):
        B.segment_mass(sh, "left trunk")
    with pytest.raises(KeyError):
        B.segment_mass(sh, "tail")
    # the six scene bones, both sides plus head and torso, add up to the whole
    whole = B.bone_mass(sh, "head") + B.bone_mass(sh, "torso") + 2 * sum(
        B.bone_mass(sh, b) for b in ("upper arm", "forearm", "thigh", "shin"))
    assert whole == pytest.approx(sh["mass_kg"], abs=1e-9)


# ── the stock figure against what the scene builder makes ────────────────────
def test_stock_sheet_matches_scenes_person():
    """Count the flesh of the real `scenes._person` at 5 cm."""
    from src.voxel.sim import World, FLESH, STONE
    from src.voxel.scenes import _person
    w = World(48, 20, 40, voxel_cm=5)
    w.fill(0, 48, 0, 20, 0, 1, STONE)
    _person(w, 8, 6)
    flesh = w.mat == FLESH
    kg = float(w.smass[flesh].sum()) / 1000.0
    assert int(flesh.sum()) == 396                      # the docstring's count
    assert abs(kg - B.STOCK["mass_kg"]) / kg < 0.02
    # its height: 30 cells of 5 cm
    zs = np.nonzero(flesh.any(axis=(0, 1)))[0]
    assert (zs.max() - zs.min() + 1) * 0.05 == pytest.approx(B.STOCK["height_m"])


def test_echo_stock_keeps_todays_flat_strength():
    from src.voxel.materials import BODY
    assert B.strength_N(B.STOCK) == BODY["strength_N"] == 400.0
    assert B.OFF_HAND == BODY["off_hand"]
    assert B.G == pytest.importorskip("src.voxel.materials").GRAVITY


# ── bone scale ───────────────────────────────────────────────────────────────
def test_bone_scale_size_rule_and_sex_factor_once():
    size = lambda m: (m / 70.0) ** (2.0 / 3.0)
    male = B.get_sheet("adult male", mass_kg=60.0)
    female = B.get_sheet("adult female")                 # 60 kg
    assert B.bone_scale(male) == pytest.approx(size(60.0))
    assert B.bone_scale(female) == pytest.approx(0.90 * size(60.0))
    assert abs(B.bone_scale(female) / size(60.0) - 0.90) < 1e-12   # not 0.90^2
    assert B.bone_scale(B.get_sheet("adult male", mass_kg=70.0)) == pytest.approx(1.0)


def test_bone_scale_age_and_draw():
    old = B.get_sheet("adult male", mass_kg=70.0, age_y=70.0)
    assert B.bone_scale(old) == pytest.approx(1.0 - 0.05 * 2)        # 2 decades over 50
    assert B.bone_scale(B.get_sheet("adult male", mass_kg=70.0, age_y=45.0)) == pytest.approx(1.0)
    assert B.bone_scale(B.get_sheet("adult male", mass_kg=70.0), draw=1.08) == pytest.approx(1.08)


def test_echo_stock_bone_scale_is_the_quoted_073():
    assert B.bone_scale(B.STOCK) == pytest.approx(0.73, abs=0.005)


def test_du_bois_area():
    s = B.get_sheet("adult male", mass_kg=70.0, height_m=1.75)
    assert abs(B.bsa_m2(s) - 1.85) < 0.02


# ── strength, grip, kick ─────────────────────────────────────────────────────
def test_echo_strength_is_400_at_68_kg():
    s = B.get_sheet("adult male", mass_kg=68.0)
    assert B.strength_N(s) == pytest.approx(400.0, abs=1e-9)
    assert B.MU_FOOT * 68.0 * 9.81 == pytest.approx(400.0)
    assert 0.5 <= B.MU_FOOT <= 0.9


def test_strength_scales_with_mass_and_can_be_overridden():
    a = B.strength_N(B.ADULT_MALE)
    assert a == pytest.approx(B.MU_FOOT * 73.0 * B.G)
    assert B.strength_N(B.ADULT_FEMALE) / a == pytest.approx(60.0 / 73.0)
    assert B.strength_N(B.get_sheet("adult male", strength_N=900.0)) == 900.0
    assert B.strength_N(B.get_sheet("adult male", muscle_scale=0.5)) == pytest.approx(a * 0.5)


def test_grip_off_hand_is_eight_tenths():
    s = B.ADULT_MALE
    assert B.grip_N(s) == B.strength_N(s)
    assert B.grip_N(s, dominant=False) == pytest.approx(0.8 * B.strength_N(s))


def test_u_kick_reference_is_about_100_joules_at_400_n(tmp_path):
    nowhere = str(tmp_path / "none.json")                # no file: the stored (u) values
    assert B.kick_numbers(nowhere) == (5.6, 6.0, "u")
    ref = B.get_sheet("adult male", strength_N=400.0)
    assert B.kick_mass_kg(ref, nowhere) == pytest.approx(5.6)
    assert B.kick_speed_ms(ref, nowhere) == pytest.approx(6.0)
    assert B.kick_energy_J(ref, nowhere) == pytest.approx(100.8)      # the specs say 'about 100 J'
    assert B.kick_energy_J(B.ADULT_MALE, nowhere) > B.kick_energy_J(B.STOCK, nowhere)


def test_u_kick_scales_with_strength(tmp_path):
    nowhere = str(tmp_path / "none.json")
    weak = B.get_sheet("adult male", strength_N=200.0)
    assert B.kick_mass_kg(weak, nowhere) == pytest.approx(5.6 * 0.5)
    assert B.kick_speed_ms(weak, nowhere) == pytest.approx(6.0 * 0.5)


def test_u_kick_takes_sourced_values_when_the_file_has_them(tmp_path):
    f = tmp_path / "sourced.json"
    f.write_text(json.dumps({"numbers": {"kick_mass_kg": {"value": 4.0, "status": "sourced"},
                                         "kick_speed_ms": {"value": 7.5, "status": "sourced"}}}))
    assert B.kick_numbers(str(f)) == (4.0, 7.5, "sourced")
    ref = B.get_sheet("adult male", strength_N=400.0)
    assert B.kick_energy_J(ref, str(f)) == pytest.approx(0.5 * 4.0 * 7.5 ** 2)
    f.write_text("not json")                              # a broken file never raises
    assert B.kick_numbers(str(f)) == (5.6, 6.0, "u")


def test_u_kick_status_seen_in_the_real_file_is_reported():
    mass, speed, status = B.kick_numbers()               # whatever A-S has by now
    print("kick numbers status:", status)
    if status == "u":
        assert (mass, speed) == (5.6, 6.0)
    assert mass > 0 and speed > 0


# ── skill ────────────────────────────────────────────────────────────────────
def test_skill_default_is_deterministic_and_in_range():
    p = {"pid": 7, "name": "wren"}
    for act, (lo, hi) in B.SKILL_RANGE.items():
        v = B.skill_default(p, act)
        assert lo <= v <= hi
        assert v == B.skill_default(dict(p), act)
    u = B.skill_default(p, "mystery")
    assert B.SKILL_OTHER[0] <= u <= B.SKILL_OTHER[1]


def test_skill_differs_between_people_and_acts():
    a, b = {"pid": 1, "name": "wren"}, {"pid": 2, "name": "bram"}
    assert B.skill_default(a, "kick") != B.skill_default(b, "kick")
    assert B.skill_default(a, "kick") != B.skill_default(a, "carry")


def test_skill_independent_of_dict_and_call_order():
    p = {"name": "wren", "pid": 3}
    q = {"pid": 3, "name": "wren", "junk": 1}               # other key order, extra keys
    acts = sorted(B.SKILL_RANGE)
    fwd = {a: B.skill_default(p, a) for a in acts}
    back = {a: B.skill_default(q, a) for a in reversed(acts)}
    assert fwd == back
    assert list(B.ensure_skill(p)) == acts                  # built in sorted order
    assert B.ensure_skill(p) == fwd


def test_skill_stored_value_wins_and_ensure_keeps_it():
    p = {"pid": 1, "name": "a", "skill": {"pick": 0.9}}
    assert B.skill_of(p, "pick") == 0.9
    B.ensure_skill(p)
    assert p["skill"]["pick"] == 0.9 and set(p["skill"]) == set(B.SKILL_RANGE)
    r = {"pid": 2, "name": "b"}
    assert B.skill_of(r, "kick") == B.skill_default(r, "kick")
    assert "skill" not in r                                  # reading never writes
    B.ensure_skill(r, "kick")
    assert list(r["skill"]) == ["kick"]


def test_codeonly_skill_uses_the_reflex_draw_recipe():
    p = {"pid": 5, "name": "wren"}
    u = (zlib.crc32("skill|climb|5|wren".encode()) & 0xFFFFFFFF) / 2.0 ** 32
    lo, hi = B.SKILL_RANGE["climb"]
    assert B.skill_default(p, "climb") == pytest.approx(lo + (hi - lo) * u)


# ── the layout ───────────────────────────────────────────────────────────────
def _voxelise(lay):
    """Draw the cell boxes into an array the way a builder would: a cell's
    flesh is its box fraction. Returns (mass kg at 1 g/cm3, max writes to one
    cell, number of cells drawn)."""
    n = 120
    g = np.zeros((n, n, n))
    cnt = np.zeros((n, n, n), int)
    for s in lay["segments"].values():
        x0, x1, y0, y1, z0, z1 = s["cells"]
        assert 0 < s["frac"] <= 1.0 + 1e-12
        assert z0 >= 0 and z1 > z0 and x1 > x0 and y1 > y0
        g[x0 + 60:x1 + 60, y0 + 60:y1 + 60, z0:z1] += s["frac"]
        cnt[x0 + 60:x1 + 60, y0 + 60:y1 + 60, z0:z1] += 1
    v = lay["voxel_cm"] / 100.0
    return g.sum() * v ** 3 * B.FLESH_KG_M3, int(cnt.max()), int((cnt > 0).sum())


@pytest.mark.parametrize("sheet", ["adult male", "adult female"])
def test_layout_voxelised_mass_matches_the_sheet(sheet):
    sh = B.get_sheet(sheet)
    for cm, tol in ((5, 0.05), (10, 0.12)):
        lay = B.adult_layout(cm, sh)
        kg, writes, cells = _voxelise(lay)
        assert writes == 1                                   # no two boxes share a cell
        assert abs(kg - sh["mass_kg"]) / sh["mass_kg"] < tol
        assert abs(lay["height_drawn_m"] - sh["height_m"]) <= 0.1 + 1e-9


def test_layout_true_shapes_weigh_winter_masses():
    sh = B.ADULT_MALE
    lay = B.adult_layout(5, sh)
    total = 0.0
    for name, s in lay["segments"].items():
        vol = math.pi * s["rx"] * s["ry"] * s["length"]
        assert vol * B.FLESH_KG_M3 == pytest.approx(s["mass_kg"], rel=1e-9)
        assert s["mass_kg"] == pytest.approx(B.segment_mass(sh, name))
        total += s["mass_kg"]
    assert total == pytest.approx(sh["mass_kg"], abs=1e-9)


def test_codeonly_layout_is_anatomical():
    lay = B.adult_layout(5, B.ADULT_MALE)
    j, sg = lay["joints"], lay["segments"]
    assert j["left hip"][2] > j["left knee"][2] > j["left ankle"][2] > 0
    assert j["left shoulder"][2] > j["left elbow"][2] > j["left wrist"][2]
    assert j["left shoulder"][0] < 0 < j["right shoulder"][0]
    assert sg["trunk"]["rx"] > sg["trunk"]["ry"]
    b = lay["bones"]
    assert 0.3 < b["thigh"] < 0.5 and 0.3 < b["upper arm"] < 0.4 and b["forearm"] > b["upper arm"] * 0.8
    # a 5 cm layout is a person of about 1.75 m, whose arms do not touch the legs
    assert lay["height_drawn_m"] == pytest.approx(1.75, abs=0.05)
    assert lay["segments"]["left hand"]["cells"][4] > 0


def test_codeonly_layout_is_pure_and_repeatable():
    assert B.adult_layout(5, B.ADULT_MALE) == B.adult_layout(5, B.ADULT_MALE)
    assert B.adult_layout(5)["mass_kg"] == 73.0                     # default sheet
    assert B.adult_layout(5, B.STOCK)["mass_kg"] == 43.9


def test_module_imports_nothing_from_the_engine():
    tree = ast.parse(open(SRC).read())
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            mods.add((n.module or "").split(".")[0] or "relative")
    assert mods <= {"json", "math", "os", "zlib"}
    assert not any(isinstance(n, ast.Global) for n in ast.walk(tree))
