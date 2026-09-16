"""Round-3 regressions: the 2x2 playtest (Haiku+Sonnet x dungeon+farm).

Every case is a verbatim finding. The theme of the round: the sim never broke
(0 incoherence in ~150 ticks) — every defect was in UNDERSTANDING (parse) or
PRESENTATION (debug leaks, canned misses, dropped clauses).
"""

from src.core import engine, reactions
from src.core.seed import dungeon, farm, vault


def _crawl(seed=3):
    world, player = dungeon(seed=seed)
    world.packs.append(reactions.load_pack("dcc"))
    return world, player


def _farm(seed=0):
    world, player = farm(seed=seed)
    world.packs += [reactions.load_pack(p) for p in ("cultivation", "farming")]
    return world, player


# SONNET-CARL: "I grab the longsword" → "you haul the pale broodmother" (twice,
# mid-fight). The grab→grapple remap fires ONLY when a person is named.
def test_grabbing_an_absent_item_never_hauls_a_monster():
    world, player = _crawl()
    player.pos = (7, 2, 0)                    # beside the broodmother, sword still boxed
    out = engine.free_text(world, player, "I grab the longsword")["narrative"]
    assert not any("haul" in l or "broodmother" in l for l in out), out


# HAIKU-CARL: "I take a deep breath" → "you take the iron crowbar". Idioms are
# not acquisitions; breath-taking now rests instead of looting.
def test_taking_a_breath_takes_nothing():
    world, player = _crawl()
    box = world.entities["box0"]
    player.pos = box.pos
    engine.free_text(world, player, "I open the loot box")     # crowbar now on the floor
    out = engine.free_text(world, player, "I take a deep breath")["narrative"]
    assert not any("crowbar" in l for l in out), out
    assert "inv:crowbar" not in world.entities["player"].tags


# HAIKU-JIN: "I rest a while" → flat refusal. Resting is waiting, with prose.
def test_resting_waits_in_prose():
    world, player = vault(seed=0)
    t0 = world.tick
    out = engine.free_text(world, player, "I rest a while")["narrative"]
    assert any("moment" in l for l in out), out
    assert world.tick > t0, "resting spends time"


# SONNET-CARL: "I look for the stairs" gave a canned miss beside visible stairs.
# HAIKU-CARL: "where is the gold box" answered with the nearest item. The map
# KNOWS — both now answer with a bearing, never coordinates.
def test_locator_answers_from_the_whole_map():
    world, player = _crawl()
    out = " ".join(engine.free_text(world, player, "I look for the stairs")["narrative"])
    assert "stairs" in out and ("to the" in out or "beside" in out), out
    assert "(" not in out, f"bearings, not coordinates: {out}"
    out2 = " ".join(engine.free_text(world, player, "where is the gold box")["narrative"])
    assert "Gold" in out2 and "below" in out2, out2


# SONNET-JIN: compound "till + plant" silently dropped the second clause.
def test_compound_clauses_both_run():
    world, player = _farm()
    engine.free_text(world, player, "I till the field and plant my rice seeds")
    here = world.cell(player.pos)
    assert "tilled" in here.tags and "sown" in here.tags, here.tags


# SONNET-JIN: "cultivate again" refused as a no-op, breaking the meditate loop.
# Continuing a state now spends the time and gathers.
def test_repeat_cultivation_continues_instead_of_refusing():
    world, player = _farm()
    engine.free_text(world, player, "I cultivate")
    q0 = player.props.get("qi", 0.0)
    out = engine.free_text(world, player, "I cultivate")["narrative"]
    assert not any("change nothing" in l for l in out), out
    assert player.props["qi"] > q0, "meditating on still gathers qi"


# SONNET-JIN: 3 near-empty "pours" read as triumphs and did nothing. A trickle
# is not a transfer — the authored fail line fires instead.
def test_near_empty_pour_fails_honestly():
    world, player = _farm()
    player.props["qi"] = 0.3
    out = engine.free_text(world, player, "I pour qi into the soil")["narrative"]
    assert any("empty" in l or "none left" in l or "nothing flows" in l for l in out), out


# SONNET-JIN top defect: "the actual scenario is a heist map". Jin has a farm
# now — pond, hens, a neighbour, and NO vault cast; feeding the chickens WORKS.
def test_the_farm_is_a_farm():
    world, player = _farm()
    assert engine.invariants(world) == []
    assert "thief" not in world.entities and "financier" not in world.entities
    assert any("animal" in e.tags for e in world.entities.values()), "hens live here"
    hen = world.entities["hen0"]
    player.pos = hen.pos
    out = engine.free_text(world, player, "I feed the chickens")["narrative"]
    assert any("peck" in l or "feed" in l or "flap" in l for l in out), out
    assert hen.props.get("fed", 0.0) > 0.0, "feeding is a real act, not flavor"


# SONNET-CARL: hp hit -1 and combat stayed on the menu. Death must land on the
# very next tick after lethal damage — no undead window.
def test_lethal_damage_kills_within_one_tick():
    from src.core import effects as fx
    world, player = _crawl()
    eff = fx.adjust_prop("player", "hp", -11.0, "the broodmother opens you up")
    eff.actor = "broodmother"
    fx.apply(world, eff)
    reactions.tick(world)
    assert "dead" in player.tags and engine.is_over(world), "no undead window"


# SONNET-CARL: XP is paid but "check stats" never shows it. Stats now lists
# every numeric prop the player carries — hp first, then the rest.
def test_stats_show_all_numeric_props():
    world, player = _crawl()
    player.props["xp"] = 18.0
    out = engine.free_text(world, player, "I check my stats")["narrative"][0]
    assert "hp 10" in out and "xp 18" in out, out


# SONNET-JIN: variety degraded — the 3rd repeat was a verbatim copy (linear
# hash cycled through half the variants on 2-tick turns). The mixer must reach
# at least 3 of 4 variants across consecutive even ticks.
def test_voice_mixer_does_not_cycle():
    from src.core.state import World
    w = World(dims=(1, 1), cells={}, entities={}, seed=5)
    picks = set()
    for t in range(0, 16, 2):
        w.tick = t
        picks.add(reactions._voice(w, "channel_qi_into_ground", ["a", "b", "c", "d"]))
    assert len(picks) >= 3, f"even-tick repeats must still vary: {picks}"


# DM BRAIN (user-directed): a big model plays DM by SELECTING a stance. The brain
# surfaces the choice (raises) with no answer, returns the forced index when
# resumed, and harvests the (ctx->pick) pair — grounded data for the local model.
def test_agent_brain_defers_then_selects_and_harvests():
    from src.core import social
    from src.core.social import AgentBrain, DMChoice, ALL_REACTIONS
    ctx = {"intent": "say", "utterance": "well met, friend", "npc_name": "Pip",
           "npc_id": "apprentice", "composure": 40.0, "tags": set(), "disposition": 0.1}
    unanswered = AgentBrain()
    try:
        unanswered.choose(ctx, ALL_REACTIONS)
        assert False, "an unanswered DM brain must defer"
    except DMChoice as dc:
        assert dc.options is ALL_REACTIONS and dc.ctx["npc_name"] == "Pip"
    warm = next(i for i, o in enumerate(ALL_REACTIONS) if o.id == "warm")
    answered = AgentBrain(answers=[warm])
    assert answered.choose(ctx, ALL_REACTIONS) == warm
    assert answered.log[-1]["chose"] == "warm", "the DM's pick is harvested with its context"
    assert social.use_brain("agent").__class__.__name__ == "AgentBrain"


# ── A/B ROUND (Haiku+Sonnet x {DM brain, torch brain}) findings ──────────────

# THE CEILING FINDING: the DM picks the right stance but every stance led to one
# static guildhall line ("...in this den!"). Packs now voice stances in their own
# register — Meiling on her farm sounds like a neighbour, not a guild doorman.
def test_pack_overrides_stance_lines_in_its_own_register():
    from src.core import social
    from src.core.seed import farm
    social.use_brain("mock")
    world, player = farm(seed=0)
    world.packs += [reactions.load_pack(p) for p in ("cultivation", "farming")]
    mei = world.entities["neighbour"]
    player.pos = (mei.pos[0] + 1, mei.pos[1], mei.pos[2])
    out = " ".join(engine.free_text(world, player, "I greet Meiling warmly")["narrative"])
    assert "den" not in out and "guild" not in out, f"no guildhall register on a farm: {out}"
    assert "Meiling" in out and '"' in out, out
    # the same greeting in the vault still uses the default bank (no farm pack)
    from src.core.seed import guildhall
    w2, p2 = guildhall(seed=0)
    pip = w2.entities["apprentice"]
    p2.pos = (pip.pos[0] + 1, pip.pos[1], pip.pos[2])
    assert engine.free_text(w2, p2, "I greet Pip warmly")["narrative"], "default bank still works"


# SONNET/HAIKU-CARL: "I take a deep breath"/fall damage printed "you plunges
# down!". Player fall/land narration is now second-person and clear (the
# "invisible attacker" that killed Carl was just unnarrated fall damage).
def test_player_fall_narration_is_clear_second_person():
    from src.core.state import Cell, Entity, World
    cells = {(0, 0, 0): Cell(), (0, 0, 1): Cell(floor=None)}   # a gap above open air
    player = Entity("player", "you", (0, 0, 1), tags={"player", "alive"}, props={"hp": 10.0})
    world = World(dims=(1, 1, 2), cells=cells, entities={"player": player}, seed=0)
    reactions.tick(world)                                     # falls
    reactions.tick(world)                                     # lands
    causes = " ".join(e.cause for e in world.log if e.cause)
    assert "you plunges" not in causes and "you slams" not in causes, causes
    assert player.pos[2] == 0, "fell to the floor below"


# SONNET-CARL INCOHERENCE: "player is dead but hp=8.0" — a same-tick level-up
# healed a corpse. Death wins the tick.
def test_no_corpse_carries_positive_hp():
    world, player = _crawl()
    player.props["xp"] = 9.5                          # one hit from a level-up
    from src.core import effects as fx
    eff = fx.adjust_prop("player", "hp", -20.0, "the broodmother guts you")
    eff.actor = "broodmother"
    fx.apply(world, eff)
    reactions.tick(world)                            # death AND level_2 both want this tick
    assert "dead" in player.tags and player.props["hp"] <= 0.0, \
        f"a corpse must not carry hp: dead={('dead' in player.tags)} hp={player.props['hp']}"
    assert engine.invariants(world) == [], engine.invariants(world)


# SONNET-JIN: hens/rooster (tagged `animal`, not `person`) were invisible on the
# map — Jin reported "no rooster". Living creatures render.
def test_animals_render_on_the_map():
    world, player = _farm()
    grid = engine.render(world, player)
    rooster = world.entities["rooster"]
    # its glyph 'e' appears at its row; simplest: it is not skipped by _glyph
    assert engine._glyph(world, rooster.pos) == "e", "the rooster is visible"


# SONNET/HAIKU-CARL: "I taunt the scuttler" / "reason with it" fell through to
# attack. Talking verbs route to dialogue even for a hostile creature.
def test_talking_to_a_monster_is_dialogue_not_a_swing():
    from src.core import social
    social.use_brain("mock")
    world, player = _crawl()
    scut = world.entities["scuttler"]
    player.pos = (scut.pos[0] + 1, scut.pos[1], scut.pos[2])
    out = " ".join(engine.free_text(world, player, "I taunt the scuttler")["narrative"])
    assert "strike" not in out and "attack" not in out, f"a taunt is not a swing: {out}"


# SONNET-JIN: sharing a worry with a not-adjacent Meiling silently became a water
# pour. Addressing someone out of earshot says so, never acts physically.
def test_addressing_a_far_npc_refuses_instead_of_acting():
    from src.core import social
    social.use_brain("mock")
    world, player = _farm()
    mei = world.entities["neighbour"]
    player.pos = (0, 5, 0)                            # far from Meiling
    player.tags.add("inv:water")
    out = " ".join(engine.free_text(world, player,
                                    "I tell Meiling I am worried about the harvest")["narrative"])
    assert "too far" in out and "pour" not in out, out


# SONNET-JIN: "look for good soil to farm" sought "farm" (the purpose word).
# The object is the noun before the purpose clause.
def test_look_stops_at_the_purpose_clause():
    world, player = _farm()
    out = " ".join(engine.free_text(world, player, "I look for good soil to farm")["narrative"])
    assert "farm" not in out.split("soil")[-1] or "soil" in out, out
    assert "for farm" not in out, out
