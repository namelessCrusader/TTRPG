"""Scene-gated persona cards (harness-sweep #4: World Info without keywords).
Each NPC carries its OWN persona; the social prompt is built from whoever is
actually being addressed — no more one hard-coded doorman for everyone.
"""

from src.core import social
from src.core.seed import guildhall


# DESIGN: the persona fed to the brain must be the ADDRESSED npc's, drawn from
# their card (flaw/bond/traits) — not a global "jittery doorman" for all.
def test_prompt_uses_the_addressed_npcs_own_persona():
    world, _ = guildhall(seed=0)
    voss = world.entities["financier"]        # cold, greedy, "everything has a price"
    card = social.persona_card(world, voss)
    assert "Voss" in card
    assert "price" in card.lower() or "ruined" in card.lower(), f"Voss's own hooks: {card}"
    assert "doorman" not in card.lower(), "no borrowed doorman identity"

    pip = world.entities["apprentice"]         # timid, warm, "flinches at everything"
    pcard = social.persona_card(world, pip)
    assert "Pip" in pcard and pcard != card, "distinct people, distinct cards"
    assert "flinch" in pcard.lower() or "timid" in pcard.lower() or "warm" in pcard.lower()


# DESIGN: a card is BUDGETED — a few short lines, not a biography (small-model
# context is precious; every token must earn its place).
def test_persona_card_is_compact():
    world, _ = guildhall(seed=0)
    for e in world.entities.values():
        if "person" in e.tags:
            card = social.persona_card(world, e)
            assert card.count("\n") <= 4 and len(card) <= 300, f"card too fat for {e.id}: {card!r}"


# DESIGN: the card reflects LIVE relationship — an NPC who has soured on the
# player says so, so the brain can read it (mechanically we already gate on the
# edge; the CARD makes it legible to the LM voice).
def test_card_reflects_live_disposition():
    world, _ = guildhall(seed=0)
    guard = world.entities["guard"]
    world.edges[("guard", "player")] = {"disposition": -0.6}
    card = social.persona_card(world, guard)
    assert "distrust" in card.lower() or "wary" in card.lower() or "cold" in card.lower(), \
        f"a soured guard's card should show it: {card}"


# DESIGN (playtest #2): NPCs must not all say the SAME line for a stance. A brave
# guard and a timid apprentice voice the same reaction in their own register.
def test_stance_lines_vary_by_trait():
    from src.core import social
    from src.core.social import REACTIONS, _voiced
    from src.core.seed import guildhall
    world, _ = guildhall(seed=1)
    wary = next(r for r in REACTIONS["talk"] if r.id == "wary")
    guard = world.entities["guard"]        # brave
    pip = world.entities["apprentice"]     # timid
    gline, pline = _voiced(wary, guard), _voiced(wary, pip)
    assert gline != pline, f"a brave guard and a timid apprentice differ: {gline!r} vs {pline!r}"
    # an even-keeled NPC with no matching variant falls back to the neutral line
    class _Bland:
        mind = {"traits": {}}
    assert _voiced(wary, _Bland()) == wary.line, "no strong trait → the default line"


# DESIGN (playtest: 3/4 agents flagged it): pestering an NPC turn after turn wears
# their patience — repeated civil greetings escalate to a brush-off, and a break
# lets them cool off. Repetition must COST something, or dialogue feels like a wall.
def test_conversation_fatigue_wears_patience():
    from src.core import engine, social
    from src.core.seed import guildhall
    social.use_brain("mock")
    world, player = guildhall(seed=12)
    pip = world.entities["apprentice"]
    player.pos = (pip.pos[0] + 1, pip.pos[1], pip.pos[2])
    lines = []
    for _ in range(6):
        res = engine.say(world, "apprentice", "hello there friend")
        lines.append(next(l for l in res["narrative"] if "Pip" in l and '"' in l))
    assert any("busy" in l for l in lines[3:]), f"pestering should wear patience: {lines}"
    assert "busy" not in lines[0], "the FIRST greeting is still civil"
    # a break resets the fatigue — coming back later is fresh
    for _ in range(6):
        engine._advance(world, 1)
    res = engine.say(world, "apprentice", "hello there friend")
    fresh = next(l for l in res["narrative"] if "Pip" in l and '"' in l)
    assert "busy" not in fresh, f"after a break the NPC is civil again: {fresh}"


# DESIGN (brawler playtest): a witness's accusation must name the RIGHT wrong —
# theft took an item, assault hurt a person. Before, both used "take the {item}",
# so an assault read "I saw you take the Maren the alchemist!".
def test_confront_line_matches_the_deed():
    from src.core import mind
    from src.core.seed import guildhall
    world, player = guildhall(seed=14)
    guard, maren = world.entities["guard"], world.entities["alchemist"]
    mem = {"deed": "assault", "actor": "player", "item": maren.id,
           "pos": maren.pos, "tick": 0, "secondhand": False}
    player.pos = (guard.pos[0] + 1, guard.pos[1], guard.pos[2])   # guard is on the culprit
    out, _res = mind._aggress(world, guard, player, mem, "crime", adj=True)
    line = next((e.cause for e in out if e.cause and "Stop" in e.cause), "")
    assert "attack" in line and "take the" not in line, f"assault named as theft: {line!r}"
