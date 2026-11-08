"""Wounds: what a blow does to a body, layer by layer (roadmap 95, step 1).

FAST TIER. These call the wound law directly on a standing body and never
step the world, so the whole file runs in seconds. The scene-level question —
does a SWUNG blade actually deliver one of these — is asked once at the end.

One quantity decides everything here: energy per unit of contact area, met
by armour, then skin, then tissue. There is no "slashing" or "bludgeoning"
anywhere in the code, and these tests are what hold it to that: the same
energy is a cut on an edge and a bruise on a face.
"""
import numpy as np
import pytest

from src.voxel.sim import IRON, STONE, World
from src.voxel.scenes import _person

AXE = 2e-4          # a declared axe bit, m^2 (the tree-felling test's axe)
POINT = 1e-5        # a bodkin arrowhead
FACE = 0.05 ** 2    # a club or a fist: a whole 5 cm voxel face


def _body():
    w = World(30, 20, 40, voxel_cm=5)
    w.fill(0, 30, 0, 20, 0, 1, STONE)
    p = _person(w, 12, 10)
    return w, p


def _cell(w, p, bone):
    """A cell on the outside of a named bone."""
    c = w._bone_parts(p, [bone])[0]
    return tuple(int(v) for v in c[np.argmax(c[:, 0])])


def test_the_SAME_ENERGY_cuts_on_an_EDGE_and_bruises_on_a_FACE():
    w, p = _body()
    cut = w._blow(p, _cell(w, p, "right upper arm"), 60.0, AXE)
    w2, p2 = _body()
    bruise = w2._blow(p2, _cell(w2, p2, "right upper arm"), 60.0, FACE)
    assert cut["open"] and cut["depth_m"] > 0.0, cut
    assert not bruise["open"], "a club does not open the body"
    assert p2["hurt"] > 0.0, "but it hurts"
    assert cut["bone"] == bruise["bone"] == "right upper arm"


def test_a_HARDER_blow_cuts_DEEPER():
    depths = []
    for e in (30.0, 45.0, 60.0):
        w, p = _body()
        depths.append(w._blow(p, _cell(w, p, "torso"), e, AXE)["depth_m"])
    assert depths[0] < depths[1] < depths[2], depths


def test_a_blow_too_light_to_part_the_skin_opens_NOTHING():
    w, p = _body()
    wd = w._blow(p, _cell(w, p, "torso"), 10.0, AXE)   # skin needs 20 J here
    assert not wd["open"]


def test_a_cut_cannot_go_deeper_than_the_LIMB_it_is_in():
    w, p = _body()
    wd = w._blow(p, _cell(w, p, "right forearm"), 5000.0, AXE)
    assert wd["depth_m"] == pytest.approx(w._thickness(p, "right forearm"))
    assert wd["through"], "an axe through a forearm goes through it"


def test_PLATE_turns_an_edge_that_opens_bare_flesh():
    w, p = _body()
    p["worn"] = {"torso": {"mat": IRON, "thick_m": 1.5e-3, "kind": "plate",
                           "area_m2": 0.06}}
    armoured = w._blow(p, _cell(w, p, "torso"), 60.0, AXE)
    w2, p2 = _body()
    bare = w2._blow(p2, _cell(w2, p2, "torso"), 60.0, AXE)
    assert bare["open"], "without the plate it cuts"
    assert not armoured["open"] and armoured["armour"] == "turned"
    assert p["hurt"] < p2["hurt"], \
        "and the plate spreads what it stops, so it hurts less"


def test_a_POINT_goes_through_plate_that_an_EDGE_does_not():
    """The same plate, the same energy. What differs is the area the blow is
    concentrated into — which is the whole argument about arrows and plate,
    arrived at from yield stress rather than looked up."""
    plate = {"torso": {"mat": IRON, "thick_m": 1.5e-3, "kind": "plate",
                       "area_m2": 0.06}}
    w, p = _body()
    p["worn"] = dict(plate)
    point = w._blow(p, _cell(w, p, "torso"), 60.0, POINT)
    w2, p2 = _body()
    p2["worn"] = dict(plate)
    edge = w2._blow(p2, _cell(w2, p2, "torso"), 60.0, AXE)
    assert point["armour"] == "pierced" and point["open"]
    assert edge["armour"] == "turned" and not edge["open"]


def test_MAIL_stops_the_edge_but_passes_the_BLOW():
    """Rings stop a cut the way a plate does but are not stiff, so a blow
    arrives nearly as narrow as it landed. Why mail was worn over padding."""
    def hit(kind):
        w, p = _body()
        p["worn"] = {"torso": {"mat": IRON, "thick_m": 1.5e-3, "kind": kind,
                               "area_m2": 0.06}}
        wd = w._blow(p, _cell(w, p, "torso"), 60.0, AXE)
        return wd, p["hurt"]
    mail, hurt_mail = hit("mail")
    plate, hurt_plate = hit("plate")
    assert not mail["open"] and not plate["open"], "both stop the edge"
    assert hurt_mail > hurt_plate, "only the plate spreads the blow"


def test_a_CUT_ARM_is_a_WEAKER_arm_and_only_that_arm():
    """No penalty table. A cut is a plane through the limb and the muscle
    past it pulls on nothing, so the same section-times-stress every limb's
    strength is already made of comes out smaller."""
    w, p = _body()
    right0 = w._torque_of(p, "right arm", 0.0)
    left0 = w._torque_of(p, "left arm", 0.0)
    w._blow(p, _cell(w, p, "right upper arm"), 60.0, AXE)
    assert w._torque_of(p, "right arm", 0.0) < right0
    assert w._torque_of(p, "left arm", 0.0) == pytest.approx(left0)


def test_a_CUT_LEG_pushes_less():
    w, p = _body()
    legs0 = w._legs_of(p)
    w._blow(p, _cell(w, p, "right thigh"), 60.0, AXE)
    assert w._legs_of(p) < legs0


# ---------------------------------------------------------------- the scene

class _Duel:
    """One person picks up what is at their feet and swings it; the other
    stands. Nobody is told to hit anybody — the swing goes where the arm
    goes, and whether it lands on a person is the world's business."""
    name = "duel"

    def __init__(self, swinger, then="swing"):
        self.swinger, self.then = swinger, then

    def pick(self, sit, menu):
        want = "stay"
        if sit["who"] == self.swinger and sit["limb"] == "hands":
            want = self.then if sit["holding"] else "take"
        elif sit["limb"] == "hands":
            want = "keep"
        for i, o in enumerate(menu):
            if o["tag"] == want:
                return i
        return 0


def _duel(edge, worn=None, ticks=150, stand=31):
    """`stand` is where the defender is. It matters, and not as a fudge: the
    swing builds speed through its arc, so the same axe lands 6.7 J on a man
    45 cm away and 38.6 J on one at 70 cm, the edge of reach — and 20 J is
    what this edge needs to part skin. Too close to cut is a real thing
    swordsmen say, and it came out of the arithmetic here unasked.

    Both men stand as built (neither walks, so neither turns — see
    `_square_up`). Squared up to each other the same curve sits 25 cm closer:
    11 J at 20 cm rising to 38 J at 45, nothing past 50."""
    w = World(46, 20, 40, voxel_cm=5)
    w.fill(0, 46, 0, 20, 0, 1, STONE)
    w.exits = [(44, 10)]
    a = _person(w, 17, 10)
    a["name"] = "attacker"
    b = _person(w, stand, 10)
    b["name"] = "defender"
    if worn:
        b["worn"] = worn
    if edge:
        w.fill(22, 24, 10, 11, 1, 2, IRON, edge=edge)     # a blade at his feet
    else:
        w.fill(22, 24, 10, 11, 1, 2, IRON)                # a bar with no edge
    w.policy = _Duel("attacker")
    for _ in range(ticks):
        w.step()
    return b.get("wounds") or [], b


def test_a_SWUNG_BLADE_opens_the_man_it_meets_and_a_SWUNG_BAR_bruises_him():
    cuts, b1 = _duel(AXE)
    bruises, b2 = _duel(None)
    assert any(wd["open"] for wd in cuts), \
        f"the blade found him and opened him: {cuts}"
    assert b2["hurt"] > 0.0, "the bar found him too"
    assert not any(wd["open"] for wd in bruises), \
        f"and did not open him: {bruises}"


def test_TOO_CLOSE_TO_CUT():
    """The same swing, the same axe, a man standing inside it. The blade
    meets him before the arm has any speed, and a blow that has not built
    up does not part the skin."""
    close, _ = _duel(AXE, stand=26)
    far, _ = _duel(AXE, stand=31)
    assert close and not any(wd["open"] for wd in close)
    assert any(wd["open"] for wd in far)


# ------------------------------------------------------------ bleeding (step 2)

from src.voxel.sim import BLEED, BLOOD, BODY  # noqa: E402


def _cut(bone="right forearm", joules=60.0, area=AXE):
    w, p = _body()
    wd = w._blow(p, _cell(w, p, bone), joules, area)
    assert wd["open"]
    return w, p, wd


def _bleed(w, seconds):
    for _ in range(int(seconds / 0.025)):
        w._law_bleed()


def test_an_open_wound_BLEEDS_onto_the_world_and_NOT_ONE_ML_is_lost():
    w, p, wd = _cut()
    total0 = w.total_fluid(BLOOD)
    inside0 = p["blood_ml"]
    _bleed(w, 20.0)
    on_ground = float(w.fvol[w.fl == BLOOD].sum())
    assert on_ground > 5.0, f"it came out: {on_ground:.1f} ml"
    assert p["blood_ml"] < inside0
    assert abs(w.total_fluid(BLOOD) - total0) < 1e-6 * total0, \
        "every millilitre is either in him or on the floor"


def test_a_SCRATCH_CLOTS_by_itself_and_a_DEEP_CUT_does_not():
    ws, ps, scratch = _cut(joules=21.0)       # just through the skin
    wd_, pd, deep = _cut(joules=60.0)
    # a 10 cm cut 3 mm deep is ~3 cm^2 of cut plane: about three minutes to
    # seal at a square centimetre a minute, which is what a long shallow cut
    # does. The deep one is a hundred square centimetres and would take an
    # hour and a half — it will not stop without help.
    _bleed(ws, 300.0)
    _bleed(wd_, 300.0)
    assert not scratch["open"], "five minutes and a scratch has stopped"
    assert deep["open"], "a deep cut has not"


def test_PRESSURE_on_a_wound_slows_it():
    w1, p1, a = _cut()
    w2, p2, b = _cut()
    b["pressed"] = True
    _bleed(w1, 20.0)
    _bleed(w2, 20.0)
    assert b.get("shed_ml", 0.0) < 0.2 * a.get("shed_ml", 0.0)


def test_BLOOD_LOSS_is_a_shortage_of_OXYGEN_not_a_hit_point():
    """No new threshold. The brain gets the blood's oxygen times how much
    blood is still going round, and faints and dies on the same lines as a
    man in smoke. ATLS: collapse at a third lost, lethal past a half."""
    w, p, _ = _cut()
    b0 = p["blood0_ml"]
    for lost, awake, alive in ((0.15, True, True), (0.40, False, True),
                               (0.55, False, False)):
        p["blood_ml"] = b0 * (1.0 - lost)
        o2 = w._delivered_o2(p)
        assert (o2 >= BODY["faint_o2"]) == awake, (lost, o2)
        assert (o2 >= BODY["death_o2"]) == alive, (lost, o2)


def test_a_BIGGER_body_carries_MORE_blood():
    w, p = _body()
    small = w._blood0(p)
    assert small == pytest.approx(
        BLEED["ml_per_kg"] * float(w.smass[w.mat == 4].sum()) / 1000.0, rel=0.05)


def test_a_man_CUT_in_a_fight_leaves_BLOOD_ON_THE_FLOOR():
    """The whole chain, nothing called by hand: an arm swings an axe, the axe
    meets a man, the wound law opens him, the bleed law pours what leaves,
    and the fluid law carries it down to the floor he is standing on."""
    w = World(46, 20, 40, voxel_cm=5)
    w.fill(0, 46, 0, 20, 0, 1, STONE)
    w.exits = [(44, 10)]
    a = _person(w, 17, 10)
    a["name"] = "attacker"
    b = _person(w, 31, 10)
    b["name"] = "defender"
    w.fill(22, 24, 10, 11, 1, 2, IRON, edge=AXE)
    w.policy = _Duel("attacker", then="keep")
    swung = False
    for t in range(400):
        if t == 20:
            w.policy = _Duel("attacker")          # one blow, then stop
        if b.get("wounds") and not swung:
            swung = True
            w.policy = _Duel("attacker", then="keep")
        w.step()
    assert any(wd["open"] for wd in b.get("wounds") or []), "he was cut"
    blood = (w.fl == BLOOD) & (w.fvol > 0)
    assert blood.any(), "and he bled"
    assert int(np.argwhere(blood)[:, 2].min()) <= 1, "and it reached the floor"
