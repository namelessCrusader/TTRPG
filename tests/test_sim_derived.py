"""Category-3 audit: heuristics that FAKED a fact the sim already holds are made
to READ the sim instead.

Two conversions live here:
  1. The `1000.0` "danger" sentinel that stood in for the boolean `on_fire`. The
     grid already stores the tag; danger should be the REAL thermal load, and
     "is this on fire?" should be asked directly — not smuggled through a float.
  2. Noise loudness was a step function: any hp delta past a threshold became one
     canned crash. The log carries the ACTUAL delta; a bigger break must carry
     farther and read louder than a small one, because the sim knows the size.
"""

from src.core import mind
from src.core.seed import vault
from src.core.state import Cell


# DESIGN: "on fire" is a fact the grid stores, not a magnitude to threshold. A
# blazing cell reads as fire no matter its heat number; a merely-scorching cell
# (hot, but no flame) is dangerous WITHOUT being fire. The old sentinel welded
# these two together at heat>=1000 — an impossible number that meant "tagged".
def test_on_fire_is_the_tag_not_a_heat_threshold():
    world, _ = vault(seed=0)
    world.cells[(5, 5)] = Cell(material="stone", tags={"on_fire"}, heat=140.0)
    world.cells[(6, 5)] = Cell(material="stone", heat=500.0)      # searing, not alight
    assert mind._on_fire(world, (5, 5)), "the tag IS the fact"
    assert not mind._on_fire(world, (6, 5)), "hot != on fire — no sentinel conflation"
    # danger now reports the real thermal load, never a magic 1000
    assert mind._danger(world, (6, 5)) > mind._danger(world, (5, 5)) - 1000, \
        "danger is the actual heat term, not an on_fire sentinel bolted on"
    assert mind._danger(world, (6, 5)) == 480.0, "heat 500 - ambient 20 = the real number"


# DESIGN: fire-avoidance must key on the FACT. An NPC standing on flame is in
# fire; standing on a merely-hot cell is not (heat can be fled by other means).
def test_fire_near_reads_the_tag_not_the_sentinel():
    world, player = vault(seed=0)
    guard = world.entities["guard"]
    guard.pos = (5, 5)
    world.cells[(5, 5)] = Cell(material="stone", heat=9999.0)     # absurd heat, NO tag
    assert not mind._fire_near(world, guard), "no flame tag anywhere → not 'in fire'"
    world.cells[(5, 5)].tags.add("on_fire")
    assert mind._fire_near(world, guard), "now it's actually alight"


# DESIGN: loudness scales with the real hp delta the log carries. A splintering
# oak table (big delta) is heard farther and described bigger than a cracked
# stool (small delta). The threshold stays as an AUDIBILITY FLOOR, not the whole
# story — below it, no sound; above it, the size is read from the delta.
def test_noise_loudness_scales_with_the_real_delta():
    world, _ = vault(seed=0)
    world.entities["barrel_oil"].tags.discard("person")          # an object, so it's a crash
    from src.core.state import LogEntry
    small = LogEntry(tick=1, kind="adjust_prop",
                     data={"t": "barrel_oil", "prop": "hp", "delta": -4.0}, cause="")
    big = LogEntry(tick=2, kind="adjust_prop",
                   data={"t": "barrel_oil", "prop": "hp", "delta": -20.0}, cause="")
    ns, nb = mind._classify_noise(world, small), mind._classify_noise(world, big)
    assert ns is not None and nb is not None, "both are above the audibility floor"
    assert nb[1] > ns[1], f"the bigger break carries farther: {ns[1]} vs {nb[1]}"
    assert nb[0] != ns[0], f"and it reads louder, not the same canned string: {ns[0]!r} {nb[0]!r}"
    # a nick below the floor makes no reportable sound at all
    tiny = LogEntry(tick=3, kind="adjust_prop",
                    data={"t": "barrel_oil", "prop": "hp", "delta": -1.0}, cause="")
    assert mind._classify_noise(world, tiny) is None, "a scratch is not a crash"
