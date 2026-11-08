"""T10a: signs.py (wave 13, spec T10 sections 3.7 and 7 "signs").

Tags (plan section 9): range gating, determinism and the no-leak rule are
exact and gate. test_codeonly_* = the code against its own table. No World.
"""
import copy

from src.voxel import signs as G

# One state per sign: the smallest state that shows it, and its range.
SHOW = {
    "limp": {"leg_capacity": 0.6},
    "bleeding": {"bleed_ml_s": 1.0},
    "deformed": {"bone_angle": 0.5},
    "pale": {"f": 0.6},
    "shivering": {"shiver": 0.5},
    "drunk": {"drunk": 0.4},
    "burned": {"burn_second_exposed": True},
    "still": {"awake": False, "lying": True},
    "cradling": {"arm_pain": 0.7},
    "tired": {"fat_gam": 0.8},
}
HIDDEN = {"vessels": [{"d": 0.004}], "cuts": [{"f_open": 1.0}], "omega": [9.0],
          "blood_ml": 123.0, "wounds": [1], "mind": "afraid", "cause": "axe"}


def _moving(name):
    return name in ("limp", "drunk")


def test_every_range_is_at_most_12_m():
    assert G.SEE_M <= 12.0
    for r in G.SIGNS:
        assert 0.0 < r["range_m"] <= 12.0
        assert 0.0 < r["range_strong_m"] <= 12.0
        assert r["range_m"] <= G.SEE_M and r["range_strong_m"] <= G.SEE_M


def test_a_healthy_person_shows_nothing():
    assert G.signs_of({}) == []
    assert G.signs_of({"f": 1.0, "leg_capacity": 1.0, "awake": True,
                       "blood_o2": 1.0, "drunk": 0.0}) == []
    assert G.words([]) == ""


def test_each_sign_appears_for_its_state_and_only_that_one():
    for name, st in SHOW.items():
        got = [n for n, s in G.signs_of(st)]
        assert got == [name], (name, got)
        assert all(0.0 < s <= 1.0 for _, s in G.signs_of(st))


def test_each_sign_is_visible_inside_its_range_and_not_outside():
    for name, st in SHOW.items():
        (nm, s), = G.signs_of(st)
        rng = G.range_of((nm, s))
        mv = _moving(name)
        assert G.visible((nm, s), rng - 0.01, moving=mv)
        assert G.visible((nm, s), rng, moving=mv)
        assert not G.visible((nm, s), rng + 0.01, moving=mv)
        assert not G.visible(nm, 100.0, moving=True)
        assert G.visible((nm, s), 0.0, moving=mv)


def test_light_scales_the_range_and_never_past_the_clamp_at_full_light():
    assert G.visible("bleeding", 6.0, light=0.5)
    assert not G.visible("bleeding", 6.5, light=0.5)
    assert not G.visible("bleeding", 12.5, light=1.0)
    assert not G.visible("bleeding", 1.0, light=0.0)


def test_limp_and_swaying_need_the_target_moving():
    assert not G.visible("limp", 3.0, moving=False)
    assert G.visible("limp", 3.0, moving=True)
    assert not G.visible("drunk", 3.0)
    assert G.visible("drunk", 3.0, moving=True)
    assert G.visible("bleeding", 3.0, moving=False)       # needs no motion


def test_pale_needs_the_face_and_is_short_range():
    assert G.visible("pale", 3.0, facing_ok=True)
    assert not G.visible("pale", 3.0, facing_ok=False)
    assert not G.visible("pale", 5.0)
    assert G.range_of("pale") == 4.0


def test_a_third_degree_burn_is_seen_from_further_than_blisters():
    s2 = dict(G.signs_of({"burn_second_exposed": True}))["burned"]
    s3 = dict(G.signs_of({"burn_third": True}))["burned"]
    assert s3 > s2
    assert G.visible(("burned", s2), 5.0) and not G.visible(("burned", s2), 6.0)
    assert G.visible(("burned", s3), 11.0) and not G.visible(("burned", s3), 12.5)
    assert G.range_of(("burned", s3)) == 12.0 and G.range_of("burned") == 5.0


def test_the_signs_the_scene_expects_at_10_m_and_5_m():
    # scene 8 (ranges inside see_m = 12): a limper who is also bleeding is
    # seen as both from 10 m, and a blister is seen at 5 m but not at 6 m.
    st = {"leg_capacity": 0.7, "bleed_ml_s": 0.5}
    names = G.visible_signs(st, 10.0, moving=True)
    assert names == ("limp", "bleeding")
    assert G.words(names) == "limping and bleeding"
    # a burn blister at 5 m but not 6 m
    st = {"burn_second_exposed": True}
    assert G.visible_signs(st, 5.0) == ("burned",)
    assert G.visible_signs(st, 6.0) == ()


def test_thresholds_are_the_spec_numbers():
    assert G.signs_of({"bleed_ml_s": 0.049}) == []
    assert [n for n, _ in G.signs_of({"bleed_ml_s": 0.05})] == ["bleeding"]
    assert [n for n, _ in G.signs_of({"blood_on_clothes": True})] == ["bleeding"]
    assert G.signs_of({"bone_angle": 0.19}) == []
    assert [n for n, _ in G.signs_of({"bone_angle": 0.2})] == ["deformed"]
    assert [n for n, _ in G.signs_of({"open_fracture": True})] == ["deformed"]
    assert G.signs_of({"f": 0.8}) == []
    assert [n for n, _ in G.signs_of({"f": 0.79})] == ["pale"]
    assert [n for n, _ in G.signs_of({"skin_cold": True})] == ["pale"]
    assert G.signs_of({"shiver": 0.09}) == []
    assert [n for n, _ in G.signs_of({"shiver": 0.1})] == ["shivering"]
    assert G.signs_of({"drunk": 0.09}) == []
    assert [n for n, _ in G.signs_of({"drunk": 0.1})] == ["drunk"]
    assert G.signs_of({"leg_pain": 0.29}) == []
    assert [n for n, _ in G.signs_of({"leg_pain": 0.3})] == ["limp"]
    assert G.signs_of({"arm_pain": 0.49}) == []
    assert [n for n, _ in G.signs_of({"arm_pain": 0.5})] == ["cradling"]
    assert G.signs_of({"fat_gam": 0.59}) == []
    assert [n for n, _ in G.signs_of({"fat_gam": 0.6})] == ["tired"]
    # still: a faint line or not awake, and lying
    assert [n for n, _ in G.signs_of({"blood_o2": 0.5, "lying": True})] == ["still"]
    assert G.signs_of({"blood_o2": 0.5, "lying": False}) == []
    assert G.signs_of({"awake": False, "lying": False}) == []
    assert G.signs_of({"blood_o2": 0.55, "lying": True}) == []


def test_strengths_grow_with_the_state_and_stay_in_0_1():
    a = dict(G.signs_of({"bleed_ml_s": 0.2}))["bleeding"]
    b = dict(G.signs_of({"bleed_ml_s": 1.0}))["bleeding"]
    c = dict(G.signs_of({"bleed_ml_s": 50.0}))["bleeding"]
    assert 0.0 < a < b < c == 1.0
    p1 = dict(G.signs_of({"f": 0.75}))["pale"]
    p2 = dict(G.signs_of({"f": 0.55}))["pale"]
    assert 0.0 < p1 < p2 <= 1.0
    assert dict(G.signs_of({"f": 0.1}))["pale"] == 1.0


def test_no_hidden_field_leaks_into_the_result():
    for name, st in SHOW.items():
        plain = G.signs_of(st)
        mixed = dict(st)
        mixed.update(HIDDEN)
        assert G.signs_of(mixed) == plain
    # and with a state made of hidden fields only, nothing at all
    assert G.signs_of(dict(HIDDEN)) == []


def test_signs_read_only_the_listed_inputs():
    seen = set()

    class Spy(dict):
        def get(self, k, d=None):
            seen.add(k)
            return super().get(k, d)

        def __getitem__(self, k):
            seen.add(k)
            return super().__getitem__(k)

    st = Spy(HIDDEN)
    G.signs_of(st)
    assert seen <= set(G.INPUTS)
    assert seen & set(HIDDEN) == set()


def test_signs_are_deterministic_and_do_not_change_the_state():
    st = {"leg_capacity": 0.5, "bleed_ml_s": 0.7, "f": 0.7, "drunk": 0.3}
    before = copy.deepcopy(st)
    a = G.signs_of(st)
    b = G.signs_of(copy.deepcopy(st))
    assert a == b and st == before
    assert [n for n, _ in a] == ["limp", "bleeding", "pale", "drunk"]   # table order
    # the order of the keys in the dict does not matter
    rev = dict(reversed(list(st.items())))
    assert G.signs_of(rev) == a


def test_words_are_stable():
    assert G.words(["limp"]) == "limping"
    assert G.words(["bleeding"]) == "bleeding"
    assert G.words(["bleeding", "limp"]) == "limping and bleeding"
    assert G.words({"limp", "bleeding"}) == "limping and bleeding"
    assert G.words(["tired", "pale", "limp"]) == "limping, pale and tired"
    assert G.words(("limp", "limp")) == "limping"
    got = {n: G.words([n]) for n in G.BY_NAME}
    assert got == {"limp": "limping", "bleeding": "bleeding",
                   "deformed": "deformed", "pale": "pale",
                   "shivering": "shivering", "drunk": "swaying",
                   "burned": "burned", "still": "still",
                   "cradling": "cradling an arm", "tired": "tired"}
    try:
        G.words(["haunted"])
    except KeyError:
        pass
    else:
        raise AssertionError("an unknown sign must be refused")


def test_codeonly_table_has_the_ten_spec_signs_in_order():
    names = [r["name"] for r in G.SIGNS]
    assert sorted(names) == sorted(
        ["bleeding", "limp", "deformed", "pale", "shivering", "drunk",
         "burned", "still", "cradling", "tired"])
    assert len(set(names)) == 10
    ranges = {r["name"]: r["range_m"] for r in G.SIGNS}
    assert ranges["pale"] == 4.0 and ranges["burned"] == 5.0
    assert ranges["tired"] == 6.0 and ranges["still"] == 8.0
    assert ranges["deformed"] == 10.0 and ranges["cradling"] == 10.0
    assert ranges["shivering"] == 10.0
    assert ranges["bleeding"] == ranges["limp"] == ranges["drunk"] == 12.0
