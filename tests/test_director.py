"""The Director (TTRPG_MECHANICS #3/#5): chaos factor, lull detection, and the
GM-move menu — including PbtA's soft/hard move trick: a warning with a fuse
that COMMITS if ignored and cancels if heeded.
"""

from src.core import director, engine, mind, reactions
from src.core.seed import vault


# DESIGN (Mythic): chaos is a bounded pacing dial — mayhem raises it, quiet
# lowers it, and it can never run away.
def test_chaos_rises_with_mayhem_and_stays_bounded():
    world, _ = vault(seed=0)
    calm0 = world.chaos
    for _ in range(30):
        director.update(world)
    assert world.chaos <= calm0, "a quiet room should cool the dial"
    world.cells[(3, 1)].tags.add("on_fire")
    for e in world.entities.values():
        if "person" in e.tags:
            e.mind["salience"] = 90.0
    for _ in range(60):
        director.update(world)
    assert world.chaos > calm0, "mayhem heats the dial"
    assert 1.0 <= world.chaos <= 9.0, "bounded, always"


# DESIGN (PbtA): when nothing happens for too long, the DM MAKES something
# happen — a move fires, marked on the log, and it counts as a story beat.
def test_lull_triggers_a_gm_move():
    world, player = vault(seed=0)
    del world.entities["vial"]              # nothing to scheme over — pure dead air
    for _ in range(40):
        reactions.tick(world)
        mind.take_turns(world)
        director.update(world)
        if any(e.kind == "gm_move" for e in world.log):
            return
    raise AssertionError("40 dead-air ticks and the DM never stirred the pot")


# DESIGN (PbtA soft→hard): the world WARNS first. Ignore the groaning timber and
# it comes down on you; step away and the moment passes.
def test_soft_move_commits_when_ignored_and_cancels_when_heeded():
    world, player = vault(seed=0)
    player.pos = (3, 2)                     # beside the timber support at (3,3)
    director.fire_soft(world, pos=(3, 3, 0), fuse=3,
                       warn="the timber above groans ominously...",
                       commit_narr="the beam cracks — debris crashes down!",
                       damage=2.0, victim="player")
    assert any("groans" in e.cause for e in world.log if e.cause), "the warning is told"
    hp0 = player.props["hp"]
    for _ in range(4):                      # stand there like a fool
        director.update(world)
    assert player.props["hp"] < hp0, "an ignored warning becomes the hard move"

    world2, player2 = vault(seed=0)
    player2.pos = (3, 2)
    director.fire_soft(world2, pos=(3, 3, 0), fuse=3,
                       warn="the timber above groans ominously...",
                       commit_narr="the beam cracks — debris crashes down!",
                       damage=2.0, victim="player")
    player2.pos = (0, 0)                    # heed it — walk away
    hp0 = player2.props["hp"]
    for _ in range(4):
        director.update(world2)
    assert player2.props["hp"] == hp0, "a heeded warning costs nothing"
