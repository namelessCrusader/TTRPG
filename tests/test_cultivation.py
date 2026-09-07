"""A LitRPG magic system is a DATA PACK, not a subsystem.

This proves the [[no-special-case-subsystems]] lesson at its limit: a whole xianxia
cultivation system — qi, realms, breakthroughs, heavenly tribulation — is authored
as reactions in data/packs/cultivation.yaml and runs on the SAME interpreter that
runs fire chemistry. The engine gains ZERO cultivation-specific code. And because
the sim stays authoritative, the world never forgets WHERE the cultivator stands —
a spirit vein feeds qi faster (location-coupled), the thing pure-narration LM games
(AI Dungeon) lose. Swap the pack → a different genre; keep the sim → same guarantees.
"""

from src.core import reactions
from src.core.state import Cell, Entity, World


def _dojo(seed=0):
    """A tiny cell with a spirit vein at its centre and one mortal on it."""
    cells = {(x, y, 0): Cell() for x in range(3) for y in range(3)}
    cells[(1, 1, 0)].tags.add("spirit")          # a spirit vein: qi pools here
    hero = Entity("hero", "Jin", (1, 1, 0), material="flesh",
                  tags={"person", "alive", "cultivating", "mortal"},
                  props={"hp": 30.0, "qi": 0.0, "power": 0.0})
    world = World(dims=(3, 3, 1), cells=cells, entities={"hero": hero}, seed=seed)
    world.packs = [reactions.load_pack("cultivation")]
    return world, hero


# DESIGN: realms advance from data alone. Qi accrues (faster on the vein), and each
# threshold spends it, swaps the stage tag, and TEMPERS the body — no engine code.
def test_cultivation_realms_advance_from_data_alone():
    world, hero = _dojo(seed=1)
    stages, hp0 = [], hero.props["hp"]
    for _ in range(60):
        reactions.tick(world)
        stage = next((t for t in ("mortal", "qi_condensation", "foundation") if t in hero.tags), None)
        if not stages or stages[-1] != stage:
            stages.append(stage)
    # the realms were climbed IN ORDER, purely by threshold reactions over `qi`
    assert stages[:3] == ["mortal", "qi_condensation", "foundation"], stages
    # ascension is not free flavour — the flesh is stronger for it
    assert hero.props["hp"] > hp0, "breakthroughs temper the body"
    assert hero.props["power"] >= 3.0, "each realm grants power"


# DESIGN: the spirit vein MATTERS — cultivating on it outpaces cultivating off it.
# Location-coupled effect = the spatial awareness a narration-only engine can't hold.
def test_spirit_vein_feeds_qi_faster_than_bare_ground():
    on_vein, hero_on = _dojo(seed=0)
    off_vein, hero_off = _dojo(seed=0)
    hero_off.pos = (0, 0, 0)                       # same person, off the vein
    for _ in range(4):                             # before any breakthrough spends qi
        reactions.tick(on_vein)
        reactions.tick(off_vein)
    assert hero_on.props["qi"] > hero_off.props["qi"] * 2, \
        f"the vein should roughly quintuple the draw: {hero_on.props['qi']} vs {hero_off.props['qi']}"


# DESIGN: heavenly tribulation is real RISK, seeded (replayable) not scripted — a
# Foundation summons lightning that either kills or tempers. Here Jin endures.
def test_tribulation_resolves_and_is_survivable():
    world, hero = _dojo(seed=1)
    saw_tribulation = False
    for _ in range(80):
        reactions.tick(world)
        saw_tribulation = saw_tribulation or "tribulation" in hero.tags
        if "foundation" in hero.tags and "tribulation" not in hero.tags:
            break
    assert saw_tribulation, "laying a Foundation must provoke the heavens"
    assert "alive" in hero.tags and "tribulation" not in hero.tags, \
        "with this seed Jin weathers the storm — the tag clears and he lives"
