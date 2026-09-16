"""Sound: the second witness channel (CDDA's lesson — perception is sight, sound,
scent). Heard-not-seen events make memories WITHOUT actor attribution: you know
a barrel burst behind the door; you don't know who did it. Brave people go look.
"""

from src.core import engine, mind
from src.core.seed import vault
from src.core.state import Cell


def _wall_off_guard(world):
    """Stone column between the guard's post and the barrels."""
    for y in range(4):
        world.cells[(2, y)] = Cell(material="stone")


# DESIGN: a splintering crash carries through walls — but sound names no names.
# The guard learns THAT something broke and WHERE, never WHO.
def test_smash_is_heard_through_the_wall_without_attribution():
    world, player = vault(seed=0)
    _wall_off_guard(world)
    guard = world.entities["guard"]
    guard.pos, player.pos = (1, 2), (3, 1)              # wall between them; ~3 cells apart
    opt = next(o for o in engine.affordance_menu(world, player) if "smash" in o.label)
    engine.step(world, opt)
    heard = [m for m in guard.mind.get("memory", []) if m["deed"] == "noise"]
    assert heard, f"the crash carries: {guard.mind.get('memory')}"
    assert heard[0]["actor"] is None, "sound names no names"
    assert not [m for m in guard.mind.get("memory", []) if m["deed"] != "noise"], \
        "through a wall he HEARD it — he didn't SEE the vandal"


# DESIGN: noise demands an answer. The brave walk toward it; the timid flinch.
def test_brave_guard_investigates_the_noise_but_pip_flinches():
    world, player = vault(seed=0)
    for y in range(2):                                  # partial wall — a way around exists
        world.cells[(2, y)] = Cell(material="stone")
    del world.entities["thief"]                        # this test is about a NOISE, not a theft to chase
    guard, pip = world.entities["guard"], world.entities["apprentice"]
    guard.pos, pip.pos, player.pos = (1, 2), (0, 0), (3, 1)
    opt = next(o for o in engine.affordance_menu(world, player) if "smash" in o.label)
    engine.step(world, opt)
    d0 = dmin = mind._l1(guard.pos, (3, 1, 0))
    for _ in range(6):
        mind.take_turns(world)
        dmin = min(dmin, mind._l1(guard.pos, (3, 1, 0)))   # closest approach — he may return to post after
    assert dmin < d0, f"Bran goes to look (closest approach {dmin}, started {d0})"
    assert "afraid" in pip.tags or not [m for m in pip.mind.get("memory", [])
                                        if m["deed"] == "noise"], \
        "Pip either flinched or was too far to hear — he does NOT play constable"
