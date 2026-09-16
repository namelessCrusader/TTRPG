"""Farming is a DATA pack too — the playtest's loudest finding made mechanism.

Jin's verdict: "A cultivator-farmer needs to *farm*; this engine lets him water
puddles instead." The farming pack answers with till → sow → water → grow →
ripen → harvest over the SAME cells, and it COMPOSES with cultivation: awakened
(spirit) soil grows crops without watering — which is literally the book.
"""

from src.core import engine, reactions
from src.core.state import Cell, Entity, World


def _plot(packs=("farming",), qi=0.0):
    cells = {(x, y, 0): Cell() for x in range(3) for y in range(3)}
    player = Entity("player", "you", (1, 1, 0), tags={"player", "alive"},
                    props={"hp": 10.0, "qi": qi})
    world = World(dims=(3, 3, 1), cells=cells, entities={"player": player}, seed=5)
    for p in packs:
        world.packs.append(reactions.load_pack(p))
    return world, player


# DESIGN: the full loop, in the player's own words, using the BASE pour verb for
# water — farming needed zero new plumbing, just vocabulary + growth rules.
def test_till_sow_water_harvest_loop():
    world, player = _plot()
    engine.free_text(world, player, "I till the field")
    here = world.cell(player.pos)
    assert "tilled" in here.tags, "the till narrative may vary; the STATE may not"
    engine.free_text(world, player, "I plant rice seeds")
    assert "sown" in here.tags
    here.fluids["water"] = 400.0                      # a good soaking
    for _ in range(8):
        reactions.tick(world)
    assert "ripe" in here.tags, f"watered rice grows: crop={here.props.get('crop')}"
    res = engine.free_text(world, player, "I harvest the rice")
    assert any("sheaves" in l for l in res["narrative"]), res["narrative"]
    assert "sown" not in here.tags and "ripe" not in here.tags, "the field is cleared"
    assert player.props.get("sheaves") == 1.0, "the yield is real and carried"


# DESIGN: order matters and failure teaches — sowing unbroken ground tells you
# what the land needs, instead of a generic refusal.
def test_sowing_untilled_ground_teaches_the_loop():
    world, player = _plot()
    res = engine.free_text(world, player, "I plant rice seeds")
    # any VARIANT of the lesson counts — they all point at the hoe
    assert any("till" in l or "hoe" in l for l in res["narrative"]), res["narrative"]


# DESIGN: PACK COMPOSITION — with cultivation AND farming active, qi poured into
# the soil awakens it, and awakened soil grows rice with no watering at all.
# Two yaml files that never name each other, cooperating through the substrate.
def test_qi_fed_soil_grows_rice_without_water():
    world, player = _plot(packs=("cultivation", "farming"), qi=30.0)
    engine.free_text(world, player, "I till the field")
    engine.free_text(world, player, "I plant rice seeds")
    engine.free_text(world, player, "I pour qi into the soil")
    engine.free_text(world, player, "I pour qi into the soil")
    here = world.cell(player.pos)
    assert "spirit" in here.tags, "16 qi wakes the plot"
    for _ in range(6):
        reactions.tick(world)
    assert "ripe" in here.tags, \
        f"spirit soil ripens rice dry: crop={here.props.get('crop')}"


# DESIGN: the concept gate covers farming too — in a world without the pack,
# farming words are alien magic, refused in fiction.
def test_farming_words_gate_in_a_mundane_world():
    from src.core.seed import vault
    world, player = vault(seed=0)
    res = engine.free_text(world, player, "I till the field")
    assert any("holds no such thing" in l for l in res["narrative"]), res["narrative"]
