"""The DCC pack: the System is a voice over borrowed mechanics.

XP is qi-shaped (a prop + threshold reactions); a level is a realm; a loot box
is a sealed container whose contents drop with the same op the death rule uses;
the floor countdown is a Clock ticked once per turn off the unique player locus.
What makes it DCC is that every narrative line mocks you.
"""

from src.core import engine, reactions
from src.core.seed import dungeon, vault


def _crawl(seed=3):
    world, player = dungeon(seed=seed)
    world.packs.append(reactions.load_pack("dcc"))
    return world, player


# DESIGN: "I open the loot box" — adjacent sealed box unseals, contents drop
# lootable, the System comments. A second open earns the greed line.
def test_loot_box_opens_once_and_spills():
    world, player = _crawl()
    box = world.entities["box0"]
    player.pos = box.pos
    res = engine.free_text(world, player, "I open the loot box")
    assert any("motes" in l or "gold sparks" in l for l in res["narrative"]), res["narrative"]
    inside = world.entities["crowbar"]
    assert "taken" not in inside.tags and inside.pos == box.pos, "the loot is on the floor now"
    res2 = engine.free_text(world, player, "I open the loot box")
    assert any("Greed" in l or "already" in l or "memory" in l for l in res2["narrative"]), res2["narrative"]


# DESIGN: no box in reach → the System mocks the groping, costing nothing.
def test_opening_air_earns_system_mockery():
    world, player = _crawl()
    for e in world.entities.values():
        if "loot_box" in e.tags:
            e.pos = (8, 0, 0)                        # all boxes far away
    player.pos = (0, 5, 0)
    res = engine.free_text(world, player, "I open a loot box")
    assert any("groping the air" in l for l in res["narrative"]), res["narrative"]


# DESIGN: achievements are one-shot reactions in the book's format (mock first,
# reward second) that pay XP; enough XP crosses a level threshold — hp up.
def test_achievements_pay_xp_and_levels_follow():
    world, player = _crawl()
    player.props["xp"] = 6.0
    player.tags.add("on_fire")
    reactions.tick(world)
    achs = [e for e in world.log if e.cause and "Literally On Fire" in e.cause]
    assert len(achs) == 1 and player.props["xp"] == 11.0, "mocked once, paid once"
    player.tags.discard("on_fire")                   # put himself out; the ach is banked
    hp0 = player.props["hp"]
    reactions.tick(world)
    assert "level_2" in player.tags and player.props["hp"] > hp0, "11 xp crosses level 2"
    assert any("statistically" in e.cause for e in world.log if e.cause), "the System celebrates, rudely"
    reactions.tick(world)
    assert len([e for e in world.log if e.cause and "Literally On Fire" in e.cause]) == 1, \
        "achievements never re-fire"


# DESIGN: the collapse countdown ticks once per turn (unique player locus) and
# the bell rings ON the log when it fills — a deadline, not a script.
def test_collapse_clock_ticks_once_per_turn_and_rings():
    world, player = _crawl()
    reactions.tick(world)
    assert world.clocks["collapse"].fill == 1, "one tick per turn, not per locus"
    for _ in range(70):
        reactions.tick(world)
    assert world.clocks["collapse"].fired
    assert any("collapse imminent" in e.cause.lower() for e in world.log if e.cause), \
        "the System announces the deadline"


# DESIGN: the gate still holds both ways — dungeon vocabulary is alien magic in
# a mundane world.
def test_loot_gates_in_the_vault():
    world, player = vault(seed=0)
    res = engine.free_text(world, player, "I open a loot box")
    assert any("holds no such thing" in l for l in res["narrative"]), res["narrative"]


# PLAYTEST (Carl, dungeon run): "!! INCOHERENCE: scuttler is inside a solid
# stone block" — random pillars could land on the fixed den. Never again.
def test_nobody_spawns_inside_stone_across_seeds():
    for s in range(30):
        world, _ = dungeon(seed=s)
        assert engine.invariants(world) == [], f"seed {s}: {engine.invariants(world)}"


# PLAYTEST (Carl): loot still sealed inside a box showed up in look/render, so
# he tried to grab a coin the world (rightly) refused — the DISPLAY lied.
def test_sealed_loot_is_invisible_until_the_box_opens():
    world, player = _crawl()
    box = world.entities["box0"]
    player.pos = box.pos
    assert world.entities["crowbar"].name not in engine.cell_info(world, box.pos)
    engine.free_text(world, player, "I open the loot box")
    assert world.entities["crowbar"].name in engine.cell_info(world, box.pos), \
        "opened: now it's really on the floor"


# PLAYTEST (Carl): explored the whole 9x6 and never found the stairs — the
# terminal map had no glyph for them. Now they render on the PLAYER'S storey,
# descending on them physically drops you to floor two, and arriving pays out;
# anywhere else, the System corrects your geography.
def test_stairs_render_and_descending_reaches_floor_two():
    world, player = _crawl()
    assert ">" in engine.render(world, player), "the way down is on the player's map"
    res = engine.free_text(world, player, "I climb down the stairs")
    assert any("Crawler" in l or "stairs" in l.lower() or "wall" in l.lower()
               for l in res["narrative"]), res["narrative"]
    assert player.pos[2] == 1, "no stairs here — still on floor one"
    player.pos = (8, 5, 1)
    xp0 = player.props["xp"]
    res = engine.free_text(world, player, "I climb down the stairs")
    assert player.pos[2] == 0, "through the floor"
    assert any("Floor One cleared" in l for l in res["narrative"]), res["narrative"]
    assert "escaped" in player.tags and player.props["xp"] >= xp0 + 10, "arrival pays"


# DESIGN: harm remembers its author (hit_by: provenance) — the show pays for
# the kill, once, and credits the KILLER even though the rule ran on the corpse.
def test_the_kill_ledger_pays_the_killer_once():
    from src.core import effects as fx
    world, player = _crawl()
    eff = fx.adjust_prop("scuttler", "hp", -30.0, "you run it through")
    eff.actor = "player"
    fx.apply(world, eff)
    for _ in range(3):
        reactions.tick(world)
    assert "dead" in world.entities["scuttler"].tags
    assert player.props["xp"] == 8.0, "the kill paid the player, not the corpse"
    assert len([e for e in world.log if e.cause and "Kill confirmed" in e.cause]) == 1
    for _ in range(3):
        reactions.tick(world)
    assert player.props["xp"] == 8.0, "one payout per corpse, ever"


# PLAYTEST (Carl): 46 turns, no countdown pressure felt. The deadline is LOUD
# now — omen at the half, warning at the brink, doom at the end, in order.
def test_countdown_omens_land_in_order():
    world, player = _crawl()
    beats = []
    for _ in range(65):
        reactions.tick(world)
        for e in world.log[len(world.log) - 4:]:
            if e.cause and "System" in e.cause and e.cause not in beats:
                beats.append(e.cause)
    said = " || ".join(beats)
    ih, ib, ic = said.find("Halfway"), said.find("Final stretch"), said.find("collapse imminent")
    assert -1 < ih < ib < ic, f"omens in order: {beats}"
