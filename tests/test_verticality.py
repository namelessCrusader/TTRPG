"""Verticality: support, gravity, stairs, falling fluids, floors that burn.

The voxel rules in one sentence: you stand on a floor slab or solid ground;
nothing holds you — you fall; stairs are the only way up; fire eats timber
floors and then gravity writes the drama.
"""

from src.core import engine, reactions, spatial
from src.core.state import Cell, Entity, World


def _tower(z_levels=3):
    """A 3×3×N shaft of open air over bedrock, with a wood floor slab option."""
    cells = {(x, y, z): Cell(material=None) for x in range(3) for y in range(3) for z in range(z_levels)}
    player = Entity("player", "you", (1, 1, 0), material="flesh",
                    tags={"player", "alive"}, props={"hp": 10.0})
    return World(dims=(3, 3, z_levels), cells=cells, entities={"player": player}, seed=0), player


# DESIGN: gravity is not optional. Nothing under your feet → you fall, and the
# floor you land on doesn't care who you are — damage scales with the drop.
def test_unsupported_entities_fall_and_landing_hurts():
    world, player = _tower()
    player.pos = (1, 1, 2)                  # two levels up, open air below
    for _ in range(4):
        reactions.tick(world)
    assert player.pos == (1, 1, 0), f"gravity: {player.pos}"
    assert player.props["hp"] < 10.0, "a two-storey drop leaves a mark"


def test_floor_slab_holds_you_up():
    world, player = _tower()
    world.cells[(1, 1, 1)].floor = "wood"
    player.pos = (1, 1, 1)
    reactions.tick(world)
    assert player.pos == (1, 1, 1), "the slab holds"


# DESIGN: stairs are the only staircase up; walking off an edge is a choice
# gravity gets to grade.
def test_stairs_connect_levels():
    world, player = _tower()
    world.cells[(0, 0, 0)].tags.add("stairs")
    world.cells[(0, 0, 1)].floor = None     # the stairwell opening
    world.cells[(0, 1, 1)].floor = "wood"   # a landing to step onto
    player.pos = (0, 0, 0)
    ups = [q for q in spatial.walk_neighbors(world, player.pos) if q[2] == 1]
    assert ups == [(0, 0, 1)], f"stairs lead up: {ups}"
    opts = [o.label for o in engine.affordance_menu(world, player)]
    assert any("climb" in l for l in opts), f"the menu offers the climb: {opts}"


# DESIGN: water finds the way down before it spreads sideways — waterfalls, not
# hovering puddles.
def test_fluids_fall_through_open_air():
    world, player = _tower()
    world.cells[(1, 1, 1)].fluids["water"] = 500.0      # no floor here
    reactions.tick(world)
    assert world.cells[(1, 1, 1)].fluids.get("water", 0) < 100
    assert world.cells[(1, 1, 0)].fluids.get("water", 0) > 400, "it rains down"


# DESIGN: the set-piece. Fire below heats the timber floor above until it burns
# away — and whoever stood on it drops into the flames. Nobody scripts this.
def test_fire_below_burns_the_floor_above_and_drops_the_watcher():
    world, player = _tower()
    world.cells[(1, 1, 1)].floor = "wood"
    player.pos = (1, 1, 1)
    world.cells[(1, 1, 0)].fluids["oil"] = 3000.0       # a burst barrel, not a puddle
    world.cells[(1, 1, 0)].tags.add("on_fire")
    for _ in range(20):
        reactions.tick(world)
        if player.pos[2] == 0:
            break
    assert world.cells[(1, 1, 1)].floor is None, "the slab burns away"
    assert player.pos[2] == 0, "and the watcher comes down with it"


# DESIGN: a spark is an ATTEMPT, not a heat pump. Clicking ignite ten times must
# not cook a log to 1200° — the spark either catches or it fizzles, idempotently.
def test_spamming_ignite_does_not_stack_heat():
    from src.core.seed import vault
    world, player = vault(seed=0)
    player.pos = (3, 2)                      # next to the timber support — wood, ignition 250°
    tgt = (3, 3, 0)
    opt = next(o for o in engine.affordance_menu(world, player)
               if "ignite" in o.label and o.target == tgt)
    engine.step(world, opt)
    once = world.cell(tgt).heat
    for _ in range(5):                       # spam the button
        opts = [o for o in engine.affordance_menu(world, player)
                if "ignite" in o.label and o.target == tgt]
        if not opts:
            break                            # already burning — menu withdrew the option
        engine.step(world, opts[0])
    assert world.cell(tgt).heat <= once + 60, \
        f"spark-spam must not stack heat: first={once:.0f}° now={world.cell(tgt).heat:.0f}°"


# DESIGN: fires burn AT their combustion temperature — they don't climb forever.
def test_fire_temperature_plateaus():
    from src.core.state import Cell, Entity, World
    cells = {(x, y, 0): Cell(material=None) for x in range(3) for y in range(3)}
    w = World(dims=(3, 3, 1), cells=cells,
              entities={"player": Entity("player", "you", (0, 0, 0),
                                         tags={"player", "alive"}, props={"hp": 10.0})}, seed=0)
    w.cells[(1, 1, 0)].fluids["oil"] = 5000.0
    w.cells[(1, 1, 0)].tags.add("on_fire")
    peak = 0.0
    for _ in range(25):
        reactions.tick(w)
        peak = max(peak, w.cells[(1, 1, 0)].heat)
    # ceiling 550 + one tick of transient neighbor-radiation before headroom closes
    assert peak <= 700, f"an oil fire is not a star: peaked at {peak:.0f}°"


# DESIGN (DF conduction): one formula — heat moves by Δtemp ÷ specific heat — makes
# metal answer fire fast and stone slow, with no per-material special cases.
def test_metal_conducts_heat_faster_than_stone():
    from src.core.state import Cell, Entity, World

    def rig(mat):
        cells = {(x, 0, 0): Cell(material=None) for x in range(3)}
        cells[(1, 0, 0)] = Cell(material=mat)
        w = World(dims=(3, 1, 1), cells=cells,
                  entities={"player": Entity("player", "you", (0, 0, 0),
                                             tags={"player", "alive"}, props={"hp": 10.0})}, seed=0)
        w.cells[(0, 0, 0)].fluids["oil"] = 3000.0
        w.cells[(0, 0, 0)].tags.add("on_fire")
        for _ in range(10):
            reactions.tick(w)
        return w.cells[(1, 0, 0)].heat

    metal, stone = rig("metal"), rig("stone")
    assert metal > stone + 30, f"metal {metal:.0f}° must outpace stone {stone:.0f}°"


# DESIGN (L4D Director): the scene has a PULSE — quiet rooms read near zero, a
# witnessed theft amid a fire reads high. One scalar the pacing systems consult.
def test_drama_pulse_tracks_scene_intensity():
    from src.core.seed import vault
    from src.core import mind
    calm_world, _ = vault(seed=0)
    calm = engine.drama(calm_world)

    hot_world, _ = vault(seed=0)
    hot_world.entities["guard"].pos = (4, 3)            # witness incoming theft
    for _ in range(6):
        mind.take_turns(hot_world)                      # thief steals in view
    hot_world.cells[(3, 1)].tags.add("on_fire")
    hot = engine.drama(hot_world)
    assert hot > calm + 20, f"drama must feel the scene: calm={calm:.0f} hot={hot:.0f}"
