"""Social consequence: do bold things → the world NOTICES → it finds you later.

Each test is a player-facing spec, annotated with the game-design reason it
exists. All deterministic (seeded), all through public interfaces: run the
world, read minds/edges/dialogue.
"""

from src.core import engine, mind, social, spatial
from src.core.composer import MockPicker
from src.core.seed import vault


def _run_until(world, pred, max_turns=12):
    for _ in range(max_turns):
        mind.take_turns(world)
        if pred():
            return True
    return pred()


# DESIGN: crime needs witnesses or theft is free — and boring. A guard who SEES
# the heist must remember who did it.
def test_guard_who_sees_theft_remembers_it():
    world, _ = vault(seed=0)
    guard = world.entities["guard"]
    guard.pos = (4, 3)                      # posted next to the vial — clear view
    stolen = _run_until(world, lambda: "taken" in world.entities["vial"].tags)
    assert stolen, "thief should steal the vial within a few turns"
    memory = guard.mind.get("memory", [])
    thefts = [m for m in memory if m["deed"] == "theft" and m["actor"] == "thief"]
    assert thefts, f"guard saw the theft but remembers nothing: {memory}"
    assert not thefts[0]["secondhand"]      # he SAW it — firsthand


# DESIGN: the negative case IS the stealth mechanic. A theft nobody saw must
# leave no memory — otherwise "did anyone see me?" has no answer worth caring about.
def test_unseen_theft_leaves_no_witness():
    world, _ = vault(seed=0)
    guard = world.entities["guard"]         # default post (0,2) — far from the vial
    stolen = _run_until(world, lambda: "taken" in world.entities["vial"].tags)
    assert stolen
    assert not [m for m in guard.mind.get("memory", []) if m["deed"] == "theft"], \
        "guard was across the room — he cannot have seen it"


# DESIGN: witnessing without reaction is a diary, not drama. The guard (sworn to
# protect, brave) must DROP his routine, chase the thief down, and confront him.
# Jittery Pip witnessing the same theft should NOT play policeman — personality gates it.
def test_guard_confronts_witnessed_thief_but_pip_does_not():
    world, _ = vault(seed=0)
    guard, pip, thief = (world.entities[k] for k in ("guard", "apprentice", "thief"))
    guard.pos, pip.pos = (4, 3), (4, 2)     # both watch the heist
    _run_until(world, lambda: "taken" in world.entities["vial"].tags)

    def confronted():
        return (mind._l1(guard.pos, thief.pos) <= 1
                and world.edges.get(("guard", "thief"), {}).get("hostility", 0) > 0)
    assert _run_until(world, confronted, max_turns=20), \
        f"guard should corner the thief: guard@{guard.pos} thief@{thief.pos} edges={world.edges.get(('guard','thief'))}"
    assert world.edges.get(("apprentice", "thief"), {}).get("hostility", 0) == 0, \
        "Pip is a coward, not a constable — he shouldn't confront"


# DESIGN: rumor is how one witness becomes a reputation. Pip never saw the theft —
# but standing next to someone who DID, he hears of it (marked secondhand, so
# later systems can treat hearsay as weaker than sight) and thinks less of Sly.
def test_rumor_spreads_to_bystander_who_saw_nothing():
    world, _ = vault(seed=0)
    guard, pip = world.entities["guard"], world.entities["apprentice"]
    guard.pos = (4, 3)                      # witness
    pip.pos = (0, 0)                        # far away — sees nothing
    _run_until(world, lambda: "taken" in world.entities["vial"].tags)
    assert not pip.mind.get("memory"), "sanity: Pip saw nothing himself"

    pip.pos = (guard.pos[0], guard.pos[1] - 1) if guard.pos[1] else guard.pos  # sidle up
    pip.pos = tuple(pip.pos)
    mind.take_turns(world)
    rumors = [m for m in pip.mind.get("memory", []) if m["deed"] == "theft"]
    assert rumors and rumors[0]["secondhand"], f"Pip should have HEARD of the theft: {pip.mind.get('memory')}"
    assert world.edges.get(("apprentice", "thief"), {}).get("disposition", 0) < 0, \
        "hearing of the theft should sour Pip on Sly"


# DESIGN: consequences must reach the PLAYER'S FACE. If you steal the vial and the
# story reaches Pip, his greeting goes cold — your reputation precedes you.
def test_your_reputation_reaches_dialogue():
    social.use_brain("mock")
    world, player = vault(seed=0)
    guard, pip = world.entities["guard"], world.entities["apprentice"]
    player.pos, guard.pos, pip.pos = (5, 2), (4, 2), (0, 0)

    # civility now reads as a warm/relent STANCE, whatever the trait-voiced surface
    # line — Pip is warm, so match his civil register, not one fixed string.
    def _pip_line(res):
        return next(ln for ln in res["narrative"] if "Pip" in ln)
    CIVIL = ("listening", "friendly face", "Tell me what's wrong", "Good to see")
    COLD = ("busy", "Save your breath", "What do you want")
    warm_before = _pip_line(engine.say(vault(seed=0)[0], "apprentice", "hello there friend"))
    assert any(c in warm_before for c in CIVIL), \
        f"sanity: an innocent stranger gets civility: {warm_before!r}"

    res = engine.free_text(world, player, "grab the vial", MockPicker())   # the crime, in view
    # NOTE: Bran stands one step away — he seizes it straight back. That's the
    # teeth working. The rumor of the ATTEMPT still spreads; that's what we test.
    assert "taken" in world.entities["vial"].tags, f"the theft attempt happened: {res}"
    pip.pos = (guard.pos[0], guard.pos[1] + 1)                             # rumor jumps to Pip
    mind.take_turns(world)
    assert world.edges.get(("apprentice", "player"), {}).get("disposition", 0) < 0

    after = _pip_line(engine.say(world, "apprentice", "hello there friend"))
    assert not any(c in after for c in CIVIL), \
        f"Pip heard what you did — no warm welcome: {after!r}"


# DESIGN: stealth must respect walls. A guard on the far side of stone cannot
# witness a theft, however close — otherwise rooms are decoration, not cover.
def test_walls_block_line_of_sight_and_witnessing():
    world, _ = vault(seed=0)
    from src.core.state import Cell
    for y in (0, 1, 2, 3):                  # a stone wall at x=4 splits the room
        world.cells[(4, y)] = Cell(material="stone")
    guard = world.entities["guard"]
    guard.pos = (3, 2)                      # 3 cells from the theft — but behind the wall
    assert not spatial.line_of_sight(world, (3, 2), (5, 2))
    _run_until(world, lambda: "taken" in world.entities["vial"].tags)
    assert not [m for m in guard.mind.get("memory", []) if m["deed"] == "theft"], \
        "the wall was between them — he cannot have seen it"


# DESIGN: the playtester's #1 ask — confrontation must have TEETH. A cornered
# thief loses the goods: the guard seizes the vial back. Crime, witnessed, undone.
def test_confrontation_seizes_the_stolen_goods():
    world, _ = vault(seed=0)
    guard, thief = world.entities["guard"], world.entities["thief"]
    guard.pos = (4, 3)
    _run_until(world, lambda: "taken" in world.entities["vial"].tags)

    def seized():
        return "inv:vial" in guard.tags and "has_prize" not in thief.tags
    assert _run_until(world, seized, max_turns=25), \
        f"guard should take the vial off Sly: guard tags={guard.tags} thief tags={thief.tags}"


# DESIGN: starting fires in front of people should make you INFAMOUS, not just warm.
# Arson is a witnessed deed like theft — same mechanism, new classification.
def test_arson_is_witnessed_and_blamed():
    world, player = vault(seed=0)
    guard = world.entities["guard"]
    player.pos, guard.pos = (3, 0), (3, 2)              # guard watches you torch the barrel
    opt = next(o for o in engine.affordance_menu(world, player) if "ignite" in o.label)
    engine.step(world, opt)
    arsons = [m for m in guard.mind.get("memory", []) if m["deed"] == "arson"]
    assert arsons and arsons[0]["actor"] == "player", \
        f"guard watched you start a fire: {guard.mind.get('memory')}"


# DESIGN: "moves toward their goal" is filler. The log must state FACTS the player
# can act on: who is heading where, after what.
def test_npc_movement_narration_names_the_goal():
    world, _ = vault(seed=0)
    for _ in range(3):
        mind.take_turns(world)
    moves = [e.cause for e in world.log if "Sly" in e.cause and e.kind == "move"]
    assert moves and any("starfire vial" in c for c in moves), \
        f"the thief's movement should say what he's after: {moves}"


# DESIGN: the playtester sat through 5 turns of 'crackles and chars'. Waiting
# should fast-forward dead air and hand control back the moment something HAPPENS.
def test_wait_fast_forwards_until_something_happens():
    world, player = vault(seed=0)
    del world.entities["vial"]              # nothing for anyone to do — pure dead air
    for e in world.entities.values():
        e.mind and e.mind.update(goal="work")
    wait = next(o for o in engine.affordance_menu(world, player) if o.label.startswith("wait"))
    engine.step(world, wait)
    assert world.tick >= 6, f"quiet world: waiting should skip ahead (tick={world.tick})"

    world2, player2 = vault(seed=0)         # a fire is spreading — events incoming
    world2.cells[(3, 1)].tags.add("on_fire")
    wait2 = next(o for o in engine.affordance_menu(world2, player2) if o.label.startswith("wait"))
    engine.step(world2, wait2)
    assert world2.tick < 6, f"eventful world: waiting should return control early (tick={world2.tick})"


# DESIGN (talk must have BEHAVIOURAL consequences, not just set a tag): a threat
# that frightens an NPC makes them BACK AWAY from the player — fear finally drives
# motion, so a conversation changes what an NPC DOES, not only what it says.
def test_a_frightened_npc_backs_away():
    social.use_brain("mock")
    world, player = vault(seed=0)
    pip = world.entities["apprentice"]
    player.pos, pip.pos = (2, 2), (2, 1)                 # adjacent
    engine.say(world, "apprentice", "give it up or I gut you", tone="threatening")
    assert "afraid" in pip.tags, "the threat landed as fear"
    d0 = mind._l1(pip.pos, player.pos)
    for _ in range(3):
        mind.take_turns(world)
    assert mind._l1(pip.pos, player.pos) > d0, \
        f"a frightened NPC retreats: dist {d0} -> {mind._l1(pip.pos, player.pos)}"


# DESIGN (the headline consequence): "give me X or else" that COWS an NPC into
# compliance makes them actually HAND X OVER. A threat with teeth — the sim yields
# the demanded item, so the whole intent finally pays off.
def test_a_cowed_npc_hands_over_what_it_holds():
    world, player = vault(seed=0)
    thief, vial = world.entities["thief"], world.entities["vial"]
    player.pos, thief.pos = (2, 2), (2, 1)              # adjacent
    vial.pos = thief.pos                                 # the thief is holding the prize
    thief.tags.add("inv:vial")
    vial.tags.add("taken")
    # the threaten→comply reaction cowed them (set the resulting state directly; the
    # reply routing is exercised elsewhere — here we test the CONSEQUENCE of it).
    thief.tags.discard("calm")
    thief.tags.add("compliant")
    yielded = _run_until(world, lambda: "inv:vial" in player.tags, max_turns=5)
    assert yielded, "the cowed NPC hands the prize over (appease outscores holding it)"
    assert "inv:vial" not in thief.tags, "and no longer holds it"
