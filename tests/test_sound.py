"""Sound and stealth (roadmap 95, step 4).

Anything that strikes anything makes a noise, as loud as the fraction of the
blow the two materials ring with rather than damp. It travels by the same
law shouts always did — spreading, and walls by their mass per area — and a
sleeper wakes, or a watchman looks round, when it arrives loud enough.
"""
import numpy as np

from src.voxel.sim import FLESH, GLASS, SOUND, STONE, WOOD, TablePolicy, World
from src.voxel.scenes import _person


def test_a_PANE_on_stone_is_a_CRASH_and_a_FOOTSTEP_is_not():
    """The two anchors the constants were pinned to, and the gap between
    them, which is what stealth is made of."""
    w = World(10, 10, 10, voxel_cm=5)
    crash = w._noise((5, 5, 1), 10.0, [GLASS, STONE])
    step = w._noise((5, 5, 1), 45.0 * 9.81 * 0.01, [FLESH, STONE])
    assert 90.0 <= crash["db"] <= 110.0, crash["db"]
    assert 50.0 <= step["db"] <= 70.0, step["db"]


def test_a_SOFT_thing_swallows_a_blow_a_HARD_one_rings_with():
    w = World(10, 10, 10, voxel_cm=5)
    on_stone = w._noise((5, 5, 1), 10.0, [GLASS, STONE])["db"]
    on_flesh = w._noise((5, 5, 1), 10.0, [GLASS, FLESH])["db"]
    assert on_stone > on_flesh + 30.0


def test_STONE_muffles_more_than_WOOD_of_the_same_thickness():
    """No table of walls. It is mass per area, read off the cells."""
    def through(mat):
        w = World(40, 10, 10, voxel_cm=5)
        w.fill(20, 21, 0, 10, 0, 10, mat)
        return w._level((10, 5, 5), (30, 5, 5), 90.0)
    assert through(STONE) < through(WOOD) - 5.0


def _rooms():
    """Two rooms, one stone wall between them."""
    w = World(50, 20, 30, voxel_cm=5)
    w.fill(0, 50, 0, 20, 0, 1, STONE)
    w.fill(24, 26, 0, 20, 1, 30, STONE)       # the wall
    w.exits = [(48, 10)]
    return w


def _crash_and_sleeper(wall):
    w = World(50, 20, 30, voxel_cm=5)
    w.fill(0, 50, 0, 20, 0, 1, STONE)
    if wall:
        w.fill(24, 26, 0, 20, 1, 30, STONE)
    w.exits = [(48, 10)]
    sleeper = _person(w, 38, 10)
    sleeper["name"] = "sleeper"
    sleeper["asleep"] = True
    w.fill(10, 13, 8, 12, 20, 21, GLASS)       # a pane, falling from 1 m
    for _ in range(60):
        w.step()
    return sleeper


def test_a_CRASH_wakes_a_sleeper_in_the_room_but_NOT_through_a_STONE_WALL():
    """The pane lands at about 105 dB. Across an open room that is a crash;
    through 10 cm of masonry — ~55 dB of mass law — it arrives near 48, under
    the ~60 dB that reliably wakes someone. Walls are what stealth is for."""
    open_room = _crash_and_sleeper(wall=False)
    walled = _crash_and_sleeper(wall=True)
    assert not open_room["asleep"], f"it woke him: {open_room['events']}"
    assert walled["asleep"], "the wall kept it from him"


def test_a_SLEEPER_sleeps_through_someone_WALKING_next_door():
    w = _rooms()
    sleeper = _person(w, 38, 10)
    sleeper["name"] = "sleeper"
    sleeper["asleep"] = True
    walker = _person(w, 6, 10)
    walker["name"] = "walker"
    walker["goal"] = (20, 10)
    furthest = 0
    for _ in range(200):
        w.step()
        furthest = max(furthest, walker["anchor"][0])
    # HOW FAR HE GOT, not where he ended: he roams, and a roam that turns back
    # west ends where it started having walked the whole way there and back.
    assert furthest > 8, "he did walk"
    assert sleeper["asleep"], f"and it did not wake anyone: {sleeper['events']}"


def test_a_WATCHMAN_turns_toward_a_noise_behind_him():
    """Nobody tells him where it was. He heard it, and his eyes go to it."""
    w = World(50, 20, 30, voxel_cm=5)
    w.fill(0, 50, 0, 20, 0, 1, STONE)
    w.exits = [(48, 10)]
    guard = _person(w, 30, 10)
    guard["name"] = "guard"
    guard["facing"] = (1.0, 0.0)               # looking away, down +x
    for _ in range(3):
        w.step()
    w._noise((10, 10, 1), 10.0, [GLASS, STONE])  # behind him, 1 m of glass
    for _ in range(15):                        # a reaction time, 0.15-0.3 s
        w.step()
    look = guard.get("look")
    assert look is not None and look[0] < -0.5, \
        f"he looked back toward it: {look}"


def _stroll(sneak):
    """A man walks along a corridor toward a guard twelve metres off."""
    w = World(260, 16, 30, voxel_cm=5)
    w.fill(0, 260, 0, 16, 0, 1, STONE)
    w.exits = [(258, 8)]
    guard = _person(w, 250, 8)
    guard["name"] = "guard"
    walker = _person(w, 6, 8)
    walker["name"] = "walker"
    walker["sneak"] = sneak
    walker["goal"] = (14, 8)

    class _Post(TablePolicy):
        """The guard keeps his post; everyone else does as the table says."""
        def pick(self, sit, menu):
            if sit["who"] == "guard" and sit["limb"] == "legs":
                return 0
            return super().pick(sit, menu)
    w.policy = _Post()
    steps = []
    for _ in range(120):
        w.step()
        steps += [snd["db"] for snd in w.sounds if snd.get("who") == "walker"
                  and snd["tick"] == w.tick - 1]
    return guard, steps


def test_CREEPING_is_quieter_than_WALKING_and_a_guard_hears_the_difference():
    """Half the pace, a quarter the energy per footfall, 6 dB — which halves
    the distance at which a footstep can be noticed. A guard twelve metres
    down the corridor hears one and not the other."""
    g_walk, walk = _stroll(sneak=False)
    g_creep, creep = _stroll(sneak=True)
    assert walk and creep, "both of them took steps"
    assert max(creep) < max(walk) - 5.0, (max(walk), max(creep))
    assert g_walk.get("noise") is not None, "the guard heard the walker"
    assert g_creep.get("noise") is None, "and not the one creeping"
