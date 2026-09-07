"""Parse honesty: a wrong action is worse than a refusal.

Every case here is a VERBATIM misparse from the two-persona playtest (Jin the
cultivator, Carl the crawler). The composer's typo tolerance kept catching real
English ("fertile LAND" → hand → give a knife; "COOK" → look), "feed" meant
pour ("I feed the chickens" poured ACID), and looking for an absent thing
inspected an arbitrary cell ("I look for loot" → "(2,4) you").
"""

import re

from src.core import engine
from src.core.seed import guildhall, vault


def _do(world, player, text):
    return engine.free_text(world, player, text)["narrative"]


# PLAYTEST (Jin): "I feed the chickens" → "you pour acid at (2, 3, 0)".
def test_feeding_chickens_is_not_arson():
    world, player = vault(seed=0)
    out = _do(world, player, "I feed the chickens")
    assert not any("acid" in l or "pour" in l for l in out), out
    # ...but feeding the FIRE is still pouring — the context gate, not a ban
    world2, player2 = vault(seed=0)
    out2 = _do(world2, player2, "I feed the fire with oil")
    assert any("oil" in l for l in out2), out2


# PLAYTEST (Jin): "I wash my hands" → the give-path ("hands" stemmed to "hand").
def test_washing_hands_is_not_giving():
    world, player = vault(seed=0)
    out = _do(world, player, "I wash my hands")
    assert not any("hand the" in l or "You carry nothing you could give" in l for l in out), out


# PLAYTEST (Jin): "I tell Pip I'm looking for fertile land to farm" → handed
# Pip the belt knife ("land" → "hand" → give). Telling someone something TALKS.
def test_telling_someone_something_talks_instead_of_gifting():
    world, player = guildhall(seed=0)
    from src.core import social
    social.use_brain("mock")
    pip = world.entities["apprentice"]
    player.pos = (pip.pos[0] + 1, pip.pos[1], pip.pos[2])
    out = _do(world, player, "I tell Pip I am looking for fertile land")
    assert not any("knife" in l for l in out), out
    assert any("Pip" in l and '"' in l for l in out), f"Pip should ANSWER: {out}"


# PLAYTEST (Carl ×3, Jin ×2): "I look for loot/exit/my car" → "(2,4) you".
def test_looking_for_absent_things_misses_gracefully():
    world, player = vault(seed=0)
    # ("loot" itself now gates as a DCC concept — an even better answer; these
    #  probe plain mundane nouns the world simply doesn't hold)
    for ask in ("I look for treasure", "I look for an exit", "I look for my car"):
        out = _do(world, player, ask)
        assert any("no sign of" in l for l in out), (ask, out)
        assert not any(re.match(r"\(\d", l) for l in out), f"no coordinate dumps: {out}"
    # bare looking still works — the scene, not a refusal
    out = _do(world, player, "I look around")
    assert out and "no sign of" not in out[0], out


# PLAYTEST (Carl): "I check my stats" → "The world offers no way to do that."
# The readout is REAL state (hp, qi if the world has it, load), never invented.
def test_stats_readout_answers_from_real_props():
    world, player = vault(seed=0)
    out = _do(world, player, "I check my stats")
    assert any("hp 10" in l and "item" in l for l in out), out


# RERUN (Carl): "I listen at the wall" → moved ("wall" fuzzy-matched "walk").
# Fuzzy rescue is verb-position-only now: a trailing noun stays a noun.
def test_listening_at_a_wall_is_not_walking():
    world, player = vault(seed=0)
    p0 = player.pos
    out = _do(world, player, "I listen at the wall")
    assert player.pos == p0, f"he stood still: {out}"


# RERUN (Jin): "I rest" → "you move to (1, 5, 0)" ("rest" noun-fuzzed to "west").
def test_resting_is_not_going_west():
    world, player = vault(seed=0)
    p0 = player.pos
    out = _do(world, player, "I rest for a moment")
    assert player.pos == p0, f"resting must not walk: {out}"


# RERUN (Carl): "I drink the water" → poured the waterskin on the floor (the
# "named liquid implies pouring" fallback). Drinking is recognized and declined.
def test_drinking_water_is_not_spilling_it():
    world, player = vault(seed=0)
    out = _do(world, player, "I drink the water")
    assert any("swallow" in l for l in out), out
    assert not any("pour" in l for l in out), out


# RERUN (Jin, the one still-broken): "I look for water" FOUND the water, then
# narrated like a debugger: "(1,4) 60ml water; stairs; you". Look is PROSE now;
# cell_info stays the debug panel for the UI.
def test_looking_at_the_world_speaks_prose_not_debug():
    world, player = vault(seed=0)
    world.cell((1, 0)).fluids["water"] = 200.0
    out = _do(world, player, "I look for the water")
    line = " ".join(out)
    assert "puddle of water" in line, out
    assert "ml" not in line and not line.startswith("("), f"no units, no coordinates: {out}"


# RERUN (Jin): "I examine Pip" dumped the cell's acid inventory and temperature.
# Examining a PERSON answers about the person, in words.
def test_examining_a_person_describes_the_person():
    from src.core.seed import guildhall
    world, player = guildhall(seed=0)
    pip = world.entities["apprentice"]
    player.pos = (pip.pos[0] + 1, pip.pos[1], pip.pos[2])
    out = _do(world, player, "I examine Pip")
    line = " ".join(out)
    assert "Pip" in line and "unhurt" in line, out
    assert "°C" not in line and "ml" not in line, f"no debug units: {out}"
