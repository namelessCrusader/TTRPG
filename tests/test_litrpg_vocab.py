"""World-gated vocabulary: what you can SAY depends on what the world IS.

The spec (user, verbatim in spirit): "I put Qi in the ground" must WORK when the
world has qi — a continuous, conserved transfer that does something sensible —
and in a world with no qi it must earn an in-fiction refusal, not a mangled
parse. "I open my inventory" works wherever carrying exists. The pack's verbs
are DATA over the same effect schema as reactions; the engine grew one seam.
"""

from src.core import engine, reactions
from src.core.seed import vault
from src.core.state import Cell, Entity, World


def _farm(seed=0, qi=30.0, hp=40.0):
    """A 3x3 plot with a player-cultivator on it, cultivation pack active."""
    cells = {(x, y, 0): Cell() for x in range(3) for y in range(3)}
    player = Entity("player", "you", (1, 1, 0), material="flesh",
                    tags={"player", "alive"}, props={"hp": hp, "qi": qi})
    world = World(dims=(3, 3, 1), cells=cells, entities={"player": player}, seed=seed)
    world.packs.append(reactions.load_pack("cultivation"))
    return world, player


# DESIGN: "I put Qi in the ground" is a CONTINUOUS, CONSERVED transfer — qi
# leaves the player and enters the cell, clamped by what the player holds.
def test_put_qi_in_the_ground_transfers_continuously():
    world, player = _farm(qi=30.0)
    res = engine.free_text(world, player, "I put qi in the ground")
    assert any("drinks it in" in l for l in res["narrative"]), res["narrative"]
    assert player.props["qi"] < 30.0, "qi left the player"
    assert world.cell((1, 1, 0)).props.get("qi", 0.0) > 0.0, "…and entered the soil"


# DESIGN: fed enough, the land AWAKENS (spirit vein) — and a beast that lives on
# awakened land drinks its qi until one day it looks at you with knowing eyes.
# Nobody scripts the chicken: it emerges from location coupling alone.
def test_feed_the_land_and_the_rooster_awakens():
    world, player = _farm(qi=30.0)
    world.entities["rooster"] = Entity("rooster", "the rooster", (1, 1, 0),
                                       tags={"animal", "alive"}, props={"hp": 5.0, "qi": 0.0})
    engine.free_text(world, player, "I pour my qi into the soil")
    engine.free_text(world, player, "I pour my qi into the soil")
    assert "spirit" in world.cell((1, 1, 0)).tags, "16 qi in the soil wakes it"
    for _ in range(10):
        reactions.tick(world)
    rooster = world.entities["rooster"]
    assert "awakened" in rooster.tags, f"the land fed what lives on it: qi={rooster.props['qi']}"
    assert any(("knowing" in e.cause or "looks back" in e.cause or "too aware" in e.cause)
               for e in world.log if e.cause), "the Bi De moment is narrated (any variant)"


# DESIGN: "too much Qi could cause things to explode" — the overload threshold
# makes the continuous value DANGEROUS, not just a fuel gauge.
def test_overfeeding_the_earth_detonates():
    world, player = _farm(qi=80.0)
    for _ in range(7):
        engine.free_text(world, player, "I push qi into the ground")
        if any(("detonat" in e.cause or "erupts" in e.cause or "bursts" in e.cause)
               for e in world.log if e.cause):
            break
    assert any(("detonat" in e.cause or "erupts" in e.cause or "bursts" in e.cause)
               for e in world.log if e.cause), "the overfed earth blows"
    # checked AT the blast, before conduction bleeds it off — the heat is real and
    # then COOLS like any heat, because it entered the same thermal substrate
    assert world.cell((1, 1, 0)).heat > 60.0, "a qi detonation is HOT — real physics follows"


# DESIGN: you cannot give what you don't have — an empty cultivator gets the
# authored fail line and the world does not move.
def test_channeling_with_no_qi_fails_in_fiction():
    world, player = _farm(qi=0.0)
    t0 = world.tick
    res = engine.free_text(world, player, "I put qi in the ground")
    assert any("you are empty" in l for l in res["narrative"]), res["narrative"]
    assert world.tick == t0, "a failed channel costs no time"


# DESIGN: the concept gate. The SAME sentence in a world with no qi is not an
# action — it's a person straining at nothing, and the world says so in fiction.
def test_qi_in_a_mundane_world_earns_the_idiot_line():
    world, player = vault(seed=0)
    t0 = world.tick
    res = engine.free_text(world, player, "I put qi in the ground")
    assert any("holds no such thing" in l for l in res["narrative"]), res["narrative"]
    assert world.tick == t0 and not world.cell(player.pos).props.get("qi"), \
        "nothing happened — no time passed, no qi appeared"


# DESIGN: "I open my inventory" — carrying exists in every world, so this always
# answers (a pack like DCC's System can later dress it as a shimmering menu).
def test_open_my_inventory_lists_what_you_carry():
    world, player = vault(seed=0)
    res = engine.free_text(world, player, "I open my inventory")
    line = res["narrative"][0]
    assert "torch" in line and "oil" in line, line
