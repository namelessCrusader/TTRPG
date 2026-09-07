"""Fire counterplay (Haiku playtest ask #1): being on fire is a CRISIS with
outs, not a death sentence. Stop-drop-roll, dousing, fleeing to safety.
"""

from src.core import engine, mind, reactions
from src.core.seed import guildhall, vault
from src.core.state import Cell


def _ignite_player(world, player):
    player.tags.add("on_fire")


# DESIGN: a burning player has a menu OPTION to fight it — drop and roll — that
# actually has a chance to put yourself out (skill-checked, not guaranteed).
def test_burning_player_can_stop_drop_and_roll():
    put_out = still = 0
    for s in range(40):
        world, player = vault(seed=s)
        _ignite_player(world, player)
        opt = next((o for o in engine.affordance_menu(world, player)
                    if "roll" in o.label.lower()), None)
        assert opt is not None, "a burning player must SEE the stop-drop-roll option"
        engine.step(world, opt)
        if "on_fire" in player.tags:
            still += 1
        else:
            put_out += 1
    assert put_out > 0 and still > 0, f"a check, not a guarantee: out={put_out} still={still}/40"


# DESIGN: the option is only offered WHEN you're on fire (no clutter otherwise).
def test_roll_option_only_when_burning():
    world, player = vault(seed=0)
    assert not any("roll" in o.label.lower() for o in engine.affordance_menu(world, player)), \
        "not on fire: no drop-and-roll"


# DESIGN: dousing yourself with carried water puts you out reliably (a resource
# solution beats a dice solution — spend the water, kill the fire).
def test_water_on_self_extinguishes():
    world, player = vault(seed=0)
    _ignite_player(world, player)
    from src.core.composer import MockPicker
    engine.free_text(world, player, "pour water on myself", MockPicker())
    assert "on_fire" not in player.tags, "water beats fire, on yourself too"


# DESIGN: rolling on a WATER tile always works (environment as counterplay);
# rolling in an inferno cannot (nothing to smother against).
def test_rolling_in_water_always_works_in_fire_never():
    world, player = vault(seed=0)
    world.cells[player.pos].fluids["water"] = 300.0
    _ignite_player(world, player)
    opt = next(o for o in engine.affordance_menu(world, player) if "roll" in o.label.lower())
    engine.step(world, opt)
    assert "on_fire" not in player.tags, "rolling in water is a sure thing"

    world2, player2 = vault(seed=0)
    world2.cells[player2.pos].tags.add("on_fire")
    _ignite_player(world2, player2)
    opt2 = next(o for o in engine.affordance_menu(world2, player2) if "roll" in o.label.lower())
    engine.step(world2, opt2)
    assert "on_fire" in player2.tags, "you can't roll out a fire you're standing in"


# DESIGN (playtest bug): water must actually PUT fire out and keep it out — not
# clear the flame tag while heat + fuel remain, letting it relight next tick. The
# playtester saw "water douses the fire" narrated while temperature ROSE.
def test_water_douses_fire_and_it_stays_out():
    world, player = guildhall(seed=0)
    pos = (player.pos[0] + 1, player.pos[1], player.pos[2])
    c = world.cell(pos)
    c.fluids["oil"] = 200.0
    c.tags.add("on_fire")
    c.heat = 140.0
    reactions.tick(world)                      # let it burn once
    from src.core import effects as fx
    fx.apply(world, fx.add_fluid(pos, "water", 400.0, "splash"))
    relit = False
    for t in range(6):
        reactions.tick(world)
        if t > 0 and "on_fire" in world.cell(pos).tags:
            relit = True
    assert "on_fire" not in world.cell(pos).tags, "the fire is out and stays out"
    assert not relit, "a doused fire must not oscillate back alight"
    assert world.cell(pos).heat < 60.0, "water cooled the cell below the ignition point"


# DESIGN (playtest bug): a burning player who splashes water on their own cell is
# put out AND stays out — no same-tick re-ignition from the still-burning floor,
# and the narration reads "you catch fire" not "you catches fire".
def test_water_on_self_extinguishes_and_stays_out():
    from src.core.composer import MockPicker
    world, player = guildhall(seed=1)
    c = world.cell(player.pos)
    c.fluids["oil"] = 100.0
    c.tags.add("on_fire")
    c.heat = 65.0
    player.tags.add("on_fire")
    res = engine.free_text(world, player, "splash water on here", MockPicker())
    assert "on_fire" not in player.tags, "water on your own cell puts you out"
    assert "on_fire" not in c.tags, "and douses the floor fire too"
    assert not any("catches fire" in ln for ln in res["narrative"]), \
        f"grammar: second person is 'you catch', not 'you catches': {res['narrative']}"


# DESIGN (arson playtest): a wet fuel pool must NOT re-ignite. Splashing water on a
# spread oil fire douses it AND keeps it out — the fuel is wet. Before, fluid_ignites
# ignored water in the cell, so hot oil relit next tick despite the water sitting in it.
def test_watered_fuel_does_not_reignite():
    from src.core import reactions, effects as fx
    world, _ = guildhall(seed=13)
    cells = [(2, 1, 0), (2, 2, 0), (2, 3, 0), (3, 2, 0)]
    for c in cells:
        world.cell(c).fluids["oil"] = 150.0
        world.cell(c).heat = 130.0
    world.cell((2, 2, 0)).tags.add("on_fire")
    reactions.tick(world)                                  # fire spreads across the pool
    fx.apply(world, fx.add_fluid((2, 2, 0), "water", 400.0, "splash"))
    relit = False
    for t in range(6):
        reactions.tick(world)
        if t > 0 and any("on_fire" in world.cell(c).tags for c in cells):
            relit = True
    assert not relit, "wet fuel re-ignited — water in the cell must smother it"
