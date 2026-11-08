"""Projectiles and cover (roadmap 95, step 3).

A projectile is one cell of matter in flight, let go at a speed. Nothing
here knows what an arrow is: the flight law moves it and the air slows it,
the sweep stops it at the first thing in the way, and the wound law decides
what that thing's contact area and energy do to a body.
"""
import numpy as np

from src.voxel.sim import IRON, STONE, WOOD, World
from src.voxel.scenes import _person

POINT = 1e-5            # a bodkin head, m^2
SHAFT = 8e-5            # an arrow's real frontal area with its fletching


def _range(stand=30, ticks=40):
    w = World(46, 20, 40, voxel_cm=5)
    w.fill(0, 46, 0, 20, 0, 1, STONE)
    w.exits = [(44, 10)]
    p = _person(w, stand, 10)
    p["name"] = "target"
    return w, p


def _shoot(w, speed=50.0, z=22, grams=60.0, edge=POINT, area=SHAFT):
    """Level, straight down the x axis, at chest height."""
    return w.loose((2, 10, z), (speed, 0.0, 0.0), WOOD, grams,
                   edge=edge, area_m2=area)


def test_an_ARROW_does_not_pass_THROUGH_a_man():
    """Two metres a tick is forty cells. Checked only where it lands, an
    arrow went clean through a man without touching him."""
    w, p = _range()
    _shoot(w)
    for _ in range(40):
        w.step()
    assert p.get("wounds"), "it hit him"
    xs = np.argwhere(w.mat == WOOD)[:, 0]
    assert not len(xs) or xs.max() <= 34, "and did not come out the far side"


def test_an_ARROW_opens_a_bare_man_and_a_THROWN_STONE_bruises_him():
    w, p = _range()
    _shoot(w)
    for _ in range(40):
        w.step()
    w2, p2 = _range()
    w2.loose((2, 10, 22), (15.0, 0.0, 0.0), STONE, 300.0)   # a hard throw
    for _ in range(40):
        w2.step()
    assert any(wd["open"] for wd in p["wounds"])
    assert p2["hurt"] > 0.0 and not any(
        wd["open"] for wd in p2.get("wounds") or [])


def test_COVER_stops_what_would_have_hit():
    w, p = _range()
    w.fill(20, 21, 5, 16, 1, 30, WOOD)          # a board wall between them
    _shoot(w)
    for _ in range(40):
        w.step()
    assert not p.get("wounds"), "the board took it"


def test_a_BODKIN_goes_through_PLATE_and_a_BROADHEAD_does_not():
    plate = {"torso": {"mat": IRON, "thick_m": 1.5e-3, "kind": "plate",
                       "area_m2": 0.06}}
    # FROM THE FRONT. He faces along y with his arms at his sides in x, so a
    # shot along x meets a hanging forearm before it meets his chest.
    def front(w, edge):
        return w.loose((31, 1, 20), (0.0, 50.0, 0.0), WOOD, 60.0,
                       edge=edge, area_m2=SHAFT)
    w, p = _range()
    p["worn"] = dict(plate)
    front(w, POINT)
    for _ in range(40):
        w.step()
    w2, p2 = _range()
    p2["worn"] = dict(plate)
    front(w2, 2e-4)                              # a broad cutting head
    for _ in range(40):
        w2.step()
    torso = [wd for wd in p.get("wounds") or [] if wd["bone"] == "torso"]
    torso2 = [wd for wd in p2.get("wounds") or [] if wd["bone"] == "torso"]
    assert torso and torso[0]["armour"] == "pierced", torso
    assert torso2 and torso2[0]["armour"] == "turned", torso2


def test_a_NARROW_thing_keeps_its_speed_and_a_CUBE_does_not():
    """The same mass, the same speed. An arrow declares its 8 mm face; a
    5 cm lump of the same weight meets the air with thirty times it. Over a
    50 m shot the arrow should lose a few m/s, not most of its speed."""
    w = World(10, 10, 10, voxel_cm=5)
    arrow = w.loose((2, 5, 5), (50.0, 0.0, 0.0), WOOD, 60.0, area_m2=SHAFT)
    lump = w.loose((2, 5, 5), (50.0, 0.0, 0.0), WOOD, 60.0)
    a_arrow = float(np.linalg.norm(w._drag_a(arrow)))
    a_lump = float(np.linalg.norm(w._drag_a(lump)))
    assert a_arrow * 1.0 < 10.0, f"a second in the air costs it {a_arrow:.1f} m/s"
    assert a_lump > 20.0 * a_arrow
