"""The interaction verb set (CoQ-style contextual menu, DF-style topical talk)
and log legibility. Each test notes the game-design reason it exists.
"""

from src.core import engine, mind, social
from src.core.seed import guildhall, vault


# DESIGN: four identical "heads for the vial" lines per turn drown the drama.
# Narrate a goal ONCE when adopted; silent pathing after (the map shows motion).
def test_goal_movement_is_narrated_once_not_every_tick():
    world, _ = guildhall(seed=0)
    for _ in range(6):
        mind.take_turns(world)
    sly_lines = [e.cause for e in world.log
                 if e.kind == "move" and e.cause and "Sly" in e.cause and "starfire" in e.cause]
    assert len(sly_lines) <= 1, f"spam: {sly_lines}"


# DESIGN: conversation is not a contact sport. You can address anyone you can
# SEE within earshot (5 cells, walls block); diagonal neighbours are within reach.
def test_talk_works_diagonally_and_at_shouting_distance():
    world, player = vault(seed=0)
    pip = world.entities["apprentice"]
    player.pos, pip.pos = (1, 1), (2, 2)                 # diagonal neighbour
    assert "apprentice" in dict(engine.npcs_in_reach(world, player)), "diagonal talk missing"

    pip.pos = (5, 1)                                     # 4 cells off, clear line
    assert "apprentice" in dict(engine.npcs_in_reach(world, player)), "shouting distance talk missing"

    from src.core.state import Cell
    for y in range(4):                                   # wall him off — no talking through stone
        world.cells[(3, y)] = Cell(material="stone")
    assert "apprentice" not in dict(engine.npcs_in_reach(world, player)), "can't chat through a wall"


# DESIGN (caught by the playtest harness): social/greeting words must route to
# TALK, never fall through to the physical-verb scorer where a small model can
# pick "throw" and knife an NPC you meant to greet. This was the #1 playtest bug.
def test_social_words_never_route_to_violence():
    from src.core.composer import MockPicker, compose
    world, player = vault(seed=0)
    pip = world.entities["apprentice"]
    pip.pos, player.pos = (2, 0, 0), (2, 1, 0)                  # Pip adjacent, both on-grid
    for utt in ["befriend pip", "hello pip", "greet the apprentice", "thank pip"]:
        steps = compose(world, player, utt, MockPicker())
        assert steps and steps[0][0] in ("social", "refuse"), \
            f"{utt!r} routed to {steps} — greeting must be talk, not a physical verb"
        # and specifically never an attack/throw at the person
        assert not any(s[0] in ("attack", "throw", "smash") for s in steps), \
            f"{utt!r} became violence: {steps}"


# DESIGN (caught by the playtest harness): "talk to the guard" must resolve to the
# guard by NAME/ROLE, not let the small model guess and pick the wrong person.
def test_talk_resolves_the_named_person():
    from src.core.composer import MockPicker, compose
    world, player = vault(seed=0)
    guard, pip = world.entities["guard"], world.entities["apprentice"]
    player.pos, guard.pos, pip.pos = (2, 2, 0), (3, 2, 0), (2, 1, 0)   # both in reach, on-grid
    for utt, want in [("talk to the guard", "guard"), ("hello bran", "guard"),
                      ("greet the apprentice", "apprentice"), ("say hi to pip", "apprentice")]:
        steps = compose(world, player, utt, MockPicker())
        assert steps[0] == ("social", want), f"{utt!r} → {steps[0]}, wanted social {want}"


# DESIGN (caught by the playtest harness): a talk-intent with NOBODY in reach must
# refuse coherently, not crash on people[0]. Surfaced live under the torch picker.
def test_talk_to_nobody_refuses_instead_of_crashing():
    from src.core.composer import MockPicker
    world, player = vault(seed=0)
    for e in list(world.entities.values()):             # clear the room of people
        if "person" in e.tags and "player" not in e.tags:
            e.pos = (0, 0)                               # far corner, out of reach
    player.pos = (5, 3)
    res = engine.free_text(world, player, "talk to the guard", MockPicker())
    assert res["narrative"], "a refusal is narrated, not an exception"
    assert not res["violations"], "and the world stays coherent"


# DESIGN: HOW you say a thing is the player's choice, not the classifier's guess.
# The same words in a threatening tone vs friendly tone land differently.
def test_declared_tone_changes_the_reaction():
    social.use_brain("mock")
    world, player = vault(seed=0)
    pip = world.entities["apprentice"]
    player.pos, pip.pos = (1, 1), (1, 2)
    threat = engine.say(world, "apprentice", "open the vault for me", tone="threatening")["narrative"]
    world2, _ = vault(seed=0)
    world2.entities["apprentice"].pos = (1, 2)
    world2.entities["player"].pos = (1, 1)
    friendly = engine.say(world2, "apprentice", "open the vault for me", tone="friendly")["narrative"]
    t_line = next(ln for ln in threat if "Pip" in ln)
    f_line = next(ln for ln in friendly if "Pip" in ln)
    assert t_line != f_line, f"tone must matter: {t_line!r} vs {f_line!r}"


# DESIGN: actions must be CHECKS, not certainties — and skill must matter.
# A clumsy attacker misses sometimes; a deft one lands far more often. Same
# seeded-dice principle the NPCs already live by (flee rolls), applied to the player.
def test_attacks_are_skill_checks_not_certainties():
    from src.core.composer import MockPicker

    def attempt(seed, might):
        world, player = vault(seed=seed)
        sly = world.entities["thief"]
        player.pos, sly.pos = (2, 2), (2, 1)
        sly.tags.add("hostile")                 # squared up, so this isolates SKILL, not the
        sly.tags.discard("calm")                # ambush bonus a still-unaware target would grant
        player.props["might"] = might
        engine.free_text(world, player, "attack Sly", MockPicker())
        return sly.props["hp"] < 10.0

    clumsy = sum(attempt(s, 0.05) for s in range(40))
    deft = sum(attempt(s, 0.95) for s in range(40))
    assert 0 < clumsy < 40, f"low skill: some hits, some misses ({clumsy}/40)"
    assert deft > clumsy + 8, f"skill must matter: deft {deft}/40 vs clumsy {clumsy}/40"


# DESIGN: a miss is information, not silence — the log says the blow went wide.
def test_missed_attack_is_narrated():
    from src.core.composer import MockPicker
    for s in range(40):                                  # find a seed where a clumsy blow misses
        world, player = vault(seed=s)
        sly = world.entities["thief"]
        player.pos, sly.pos = (2, 2), (2, 1)
        player.props["might"] = 0.05
        res = engine.free_text(world, player, "attack Sly", MockPicker())
        if sly.props["hp"] == 10.0:
            assert any("miss" in ln.lower() or "wide" in ln.lower() or "dodge" in ln.lower()
                       or "duck" in ln.lower() for ln in res["narrative"]), \
                f"the miss must be told: {res['narrative']}"
            return
    raise AssertionError("clumsy attacker never missed in 40 tries?!")


# DESIGN (DF conversations): rumors must be QUERYABLE. Ask a witness and he tells
# you what he SAW; ask someone who only heard, and he says so; ask the ignorant
# and learn nothing. Now "who knows what" is investigable — detective gameplay.
def test_ask_about_reveals_what_people_know():
    world, player = vault(seed=0)
    guard, pip, mind_mod = world.entities["guard"], world.entities["apprentice"], None
    guard.pos = (4, 3)                                  # witnesses the theft
    pip.pos = (0, 0)
    from src.core import mind as mind_mod
    for _ in range(8):
        mind_mod.take_turns(world)
    pip.pos = tuple(guard.pos[:1] + (guard.pos[1] - 1,))  # hears the rumor
    mind_mod.take_turns(world)

    saw = engine.ask_about(world, "guard", "thief")["narrative"]
    assert any("saw" in ln for ln in saw), f"the witness testifies: {saw}"
    heard = engine.ask_about(world, "apprentice", "thief")["narrative"]
    assert any("heard" in ln or "word is" in ln.lower() for ln in heard), f"hearsay sounds like hearsay: {heard}"
    nothing = engine.ask_about(world, "guard", "apprentice")["narrative"]
    assert any("nothing" in ln.lower() or "couldn't" in ln.lower() for ln in nothing), \
        f"no knowledge = says so: {nothing}"


# DESIGN (CoQ give-items): generosity is a social lever — a gift warms disposition.
def test_giving_a_gift_warms_disposition():
    world, player = vault(seed=0)
    from src.core.composer import MockPicker
    pip = world.entities["apprentice"]
    player.pos, pip.pos = (5, 2), (0, 0)
    world.entities["thief"].pos = (0, 3)               # keep the greedy rival away — this is a GIFT
    engine.free_text(world, player, "grab the vial", MockPicker())  # test, not a heist-rivalry test
    pip.pos = (5, 1)                                    # bring him the prize
    before = world.edges.get(("apprentice", "player"), {}).get("disposition", 0)
    opt = next(o for o in engine.affordance_menu(world, player) if "give" in o.label and "Pip" in o.label)
    engine.step(world, opt)
    assert "inv:vial" not in player.tags and "inv:vial" in pip.tags
    assert world.edges[("apprentice", "player")]["disposition"] > before


# DESIGN: violence must exist, be witnessed, and be ANSWERED. Attacking Sly in
# front of the guard: Sly is hurt and reacts; the guard remembers the assault.
def test_attack_hurts_is_witnessed_and_answered():
    world, player = vault(seed=0)
    from src.core import mind as mind_mod
    sly, guard = world.entities["thief"], world.entities["guard"]
    player.pos, sly.pos, guard.pos = (2, 2), (2, 1), (3, 2)
    player.props["might"] = 2.0                         # this test is about consequences, not the roll
    opt = next(o for o in engine.affordance_menu(world, player) if "attack" in o.label and "Sly" in o.label)
    engine.step(world, opt)
    assert sly.props["hp"] < 10.0, "the blow lands"
    assert [m for m in guard.mind.get("memory", []) if m["deed"] == "assault"], \
        f"the guard watched you do it: {guard.mind.get('memory')}"
    assert world.edges.get(("thief", "player"), {}).get("hostility", 0) > 0 \
        or sly.pos != (2, 1), f"Sly must answer violence — fight or flight, not indifference"


def _arm(world, player, damage=5.0):
    """Give the player a belt knife (tests inject it; eval fixtures stay knife-free)."""
    from src.core.state import Entity
    knife = Entity("knife", "belt knife", player.pos, material="metal",
                   tags={"item", "weapon", "taken"}, props={"damage": damage})
    world.entities["knife"] = knife
    player.tags.add("inv:knife")
    return knife


# DESIGN: a refusal must be a CLUE, not a wall. "throw knife" with no knife should
# say you lack one — the verb was understood; the means were missing.
def test_refusal_explains_whats_missing():
    world, player = vault(seed=0)
    from src.core.composer import MockPicker
    res = engine.free_text(world, player, "throw a knife at Sly", MockPicker())
    line = " ".join(res["narrative"]).lower()
    assert "nothing" in line and "throw" in line, \
        f"the DM should say WHY it failed (no knife to throw): {res['narrative']}"


# DESIGN: a thrown knife is a ranged attack, not a delivery service. It wounds
# whoever it hits — and the wound is an assault like any other (witnessed, answered).
def test_thrown_knife_wounds_its_target():
    world, player = vault(seed=0)
    from src.core.composer import MockPicker
    sly = world.entities["thief"]
    player.pos, sly.pos = (2, 2), (2, 0)                # two cells — beyond arm's reach
    player.props["finesse"] = 2.0                       # about the wound pipeline, not the roll
    _arm(world, player)
    engine.free_text(world, player, "throw the knife at Sly", MockPicker())
    assert sly.props["hp"] < 10.0, "the knife should wound him"
    # the knife lands where he STOOD — he may well have staggered off since
    assert "inv:knife" not in player.tags and world.entities["knife"].pos == (2, 0, 0)


# DESIGN: what goes down must be retrievable. Throw your knife, walk over, pick
# it back up — otherwise every thrown item is consumed, and nobody throws anything.
def test_pick_up_and_drop_close_the_item_loop():
    world, player = vault(seed=0)
    knife = _arm(world, player)
    knife.tags.discard("taken"); player.tags.discard("inv:knife")
    knife.pos = (1, 0)                                  # lying on the ground nearby
    player.pos = (1, 1)
    pick = next(o for o in engine.affordance_menu(world, player) if "pick up" in o.label)
    engine.step(world, pick)
    assert "inv:knife" in player.tags, "it's in hand again"
    drop = next(o for o in engine.affordance_menu(world, player) if "drop" in o.label)
    engine.step(world, drop)
    assert "inv:knife" not in player.tags and knife.pos == player.pos and "taken" not in knife.tags


# DESIGN: heavy things can't be carried but CAN be shoved around — pushing the
# powder keg somewhere interesting is half the fun of having a powder keg.
def test_heavy_objects_can_be_pushed():
    world, player = vault(seed=0)
    barrel = world.entities["barrel_oil"]               # heavy container at (3,1)
    player.pos = (2, 1)                                 # west of it: push sends it east
    push = next(o for o in engine.affordance_menu(world, player) if "push" in o.label and "barrel" in o.label)
    engine.step(world, push)
    assert barrel.pos == (4, 1, 0), f"pushed one cell away from you: {barrel.pos}"


# DESIGN: contact leaves marks. Wade through oil and you are VISIBLY oil-soaked —
# a walking clue minutes after an arson. Water washes it off. Contamination = evidence.
def test_wading_through_oil_soaks_you_and_water_washes_it():
    world, player = vault(seed=0)
    world.cells[(1, 1)].fluids.clear()
    world.cells[(1, 1)].fluids["oil"] = 300.0
    player.pos = (1, 1)
    from src.core import reactions
    reactions.tick(world)
    assert "soaked_oil" in player.tags, "wading through oil marks you"
    assert "soaked in oil" in engine.cell_info(world, player.pos), "and everyone can see it"
    world.cells[(0, 1)].fluids["water"] = 400.0
    player.pos = (0, 1)
    reactions.tick(world)
    assert "soaked_oil" not in player.tags, "water washes the evidence away"


# DESIGN: naming a target the action can't reach must REFUSE WITH THE REASON —
# never silently substitute a different victim ("at Maren" must not knife Pip).
def test_named_target_out_of_range_refuses_and_harms_nobody():
    from src.core.seed import guildhall
    from src.core.composer import MockPicker
    world, player = guildhall(seed=0)                   # Maren at (4,2), player (1,4): range 5 > 3
    res = engine.free_text(world, player, "throw the knife at Maren", MockPicker())
    line = " ".join(res["narrative"]).lower()
    assert "maren" in line and ("far" in line or "range" in line or "reach" in line), \
        f"say WHY it can't be done: {res['narrative']}"
    assert all(e.props.get("hp", 10) == 10 for e in world.entities.values() if "person" in e.tags), \
        "and absolutely nobody gets stabbed by accident"


# DESIGN: violence with no named target is a question, not a guess. "throw the
# knife" amid a crowd → the DM asks "at what?" — bystanders are not defaults.
def test_unnamed_violent_action_asks_back():
    from src.core.composer import MockPicker
    world, player = vault(seed=0)
    _arm(world, player)
    world.entities["thief"].pos = (2, 1)                # two possible victims in range
    world.entities["apprentice"].pos = (3, 2)
    player.pos = (2, 2)
    res = engine.free_text(world, player, "throw the knife", MockPicker())
    line = " ".join(res["narrative"]).lower()
    assert "?" in line and ("at what" in line or "at whom" in line), f"ask, don't guess: {res['narrative']}"
    assert all(e.props.get("hp", 10) == 10 for e in world.entities.values() if "person" in e.tags)


# DESIGN (playtest bug): fluid options must only appear where they'd DO something,
# not as 24 empty-air splash lines across 8 directions. The menu was overwhelming.
def test_fluid_options_only_where_they_act():
    from src.core.state import Cell
    world, player = vault(seed=0)
    # clear people out, and wipe the cells around a chosen empty spot so nothing
    # flammable/container is adjacent → no fluid lines belong here
    for e in list(world.entities.values()):
        if "player" not in e.tags:
            e.pos = (0, 0, 0)
    player.pos = (4, 3, 0)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            world.cells[(4 + dx, 3 + dy, 0)] = Cell(material=None)
    opts = engine.affordance_menu(world, player)
    fluid = [o for o in opts if any(v in o.label for v in ("pour oil", "throw acid", "splash water"))]
    assert not fluid, f"no worthwhile fluid target, yet menu offered {[o.label for o in fluid]}"
    # now put a flammable wooden support to the N — fluid actions reappear, targeted
    world.cells[(4, 2, 0)] = Cell(material="wood", tags={"flammable"}, props={"hp": 6.0})
    opts = engine.affordance_menu(world, player)
    assert any("N" in o.label and "oil" in o.label for o in opts), "oil-on-the-wood should be offered"


# DESIGN (playtest bug): killing the holder of the prize must DROP it — a corpse's
# inventory falls to the ground, lootable. Combat→loot→win was a dead-end before.
def test_the_dead_drop_their_inventory():
    from src.core import reactions, effects as fx
    world, player = guildhall(seed=1)
    guard, vial = world.entities["guard"], world.entities["vial"]
    vial.pos = guard.pos
    fx.apply(world, fx.set_tag(vial.id, "taken"))
    fx.apply(world, fx.set_tag(guard.id, f"inv:{vial.id}"))
    fx.apply(world, fx.adjust_prop(guard.id, "hp", -20.0))
    reactions.tick(world)
    assert "dead" in guard.tags, "the guard is slain"
    assert f"inv:{vial.id}" not in guard.tags, "the corpse no longer holds the vial"
    assert "taken" not in vial.tags, "and the vial is lootable again"
    player.pos = guard.pos
    opts = engine.affordance_menu(world, player)
    assert any("pick up" in o.label and "starfire" in o.label for o in opts), \
        "the dropped prize can be picked up off the corpse"
