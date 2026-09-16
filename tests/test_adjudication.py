"""The adjudication organism (TTRPG_MECHANICS #1/#2/#4): graded outcome bands,
a closed consequence menu, the fact oracle, and the universal Clock.

Design sources: Blades in the Dark (bands + consequences + clocks), Ironsworn
(oracle likelihood ladder, facts become canon).
"""

from src.core import checks, engine, reactions
from src.core.seed import vault


# DESIGN (Blades): four bands from one roll. Skill moves the whole distribution;
# "partial" means the action WORKS but costs — the band that makes stories.
def test_bands_are_graded_and_skill_matters():
    from collections import Counter

    def dist(p, n=300):
        c = Counter()
        world, _ = vault(seed=7)
        for _ in range(n):
            c[checks.roll_band(world, p)] += 1
        return c

    lo, hi = dist(0.3), dist(0.85)
    for band in ("crit", "clean", "partial", "miss"):
        assert lo[band] > 0, f"band {band} must be reachable at p=0.3"
    assert hi["crit"] > lo["crit"] and hi["clean"] > lo["clean"]
    assert hi["miss"] < lo["miss"], "skill must buy you out of failure"


# DESIGN: "you fail, nothing happens" is banned. A partial hit LANDS but draws a
# consequence; a miss draws one too. Consequences are typed deltas on the log.
def test_partial_and_miss_always_carry_a_consequence():
    world, player = vault(seed=0)
    sly = world.entities["thief"]
    player.pos, sly.pos = (2, 2), (2, 1)
    seen_partial = seen_miss = False
    for s in range(60):
        w, p = vault(seed=s)
        t = w.entities["thief"]
        p.pos, t.pos = (2, 2), (2, 1)
        p.props["might"] = 0.5
        start = len(w.log)
        from src.core import composer
        effects, _ = composer.to_effects(w, p, [("attack", "thief")])
        from src.core import effects as fx
        fx.apply_all(w, effects)
        new = w.log[start:]
        hit = any(e.kind == "adjust_prop" and e.data.get("t") == "thief" for e in new)
        conseq = any(e.kind == "consequence" for e in new)
        if hit and conseq:
            seen_partial = True
        if not hit:
            seen_miss = True
            assert conseq, f"a miss must still MOVE the story: {[(e.kind, e.cause) for e in new]}"
        if seen_partial and seen_miss:
            return
    assert seen_partial and seen_miss, "60 seeds should show both partial hits and misses"


# DESIGN (Ironsworn): unknowable facts get an oracle roll on a likelihood ladder —
# and the answer becomes CANON: ask twice, same answer, forever. No drift.
def test_oracle_answers_become_permanent_canon():
    world, _ = vault(seed=3)
    first = checks.oracle(world, "is there rope in the storeroom", "likely")
    again = checks.oracle(world, "is there rope in the storeroom", "likely")
    assert first == again, "canon does not flicker"
    assert any(e.kind == "fact" for e in world.log), "facts are logged deltas"
    yes = sum(checks.oracle(vault(seed=s)[0], "q", "likely") for s in range(100))
    no_ = sum(checks.oracle(vault(seed=s)[0], "q", "unlikely") for s in range(100))
    assert yes > 60 > no_ > 5, f"the ladder must mean something: likely={yes}/100 unlikely={no_}/100"


# DESIGN (Blades clocks): one counter type for alarms, quests, doom. Fed only by
# bus deltas; fires its event exactly once when it fills.
def test_clock_fills_via_bus_and_fires_once():
    from src.core import effects as fx
    world, _ = vault(seed=0)
    checks.add_clock(world, "alarm", size=4, on_full="the guild bell rings the alarm!")
    for _ in range(3):
        fx.apply(world, fx.clock_tick("alarm", 1))
    assert world.clocks["alarm"].fill == 3
    assert not any("bell rings" in e.cause for e in world.log if e.cause)
    fx.apply(world, fx.clock_tick("alarm", 1))
    rings = [e for e in world.log if e.cause and "bell rings" in e.cause]
    assert len(rings) == 1, "fires exactly once"
    fx.apply(world, fx.clock_tick("alarm", 1))
    rings = [e for e in world.log if e.cause and "bell rings" in e.cause]
    assert len(rings) == 1, "a full clock stays fired, never re-fires"


# DESIGN: off_balance is a REAL cost — it eats your next check and is consumed.
def test_off_balance_penalizes_and_is_consumed():
    from src.core import composer
    from src.core import effects as fx

    def hits(with_tag, n=60):
        wins = 0
        for s in range(n):
            w, p = vault(seed=s)
            t = w.entities["thief"]
            p.pos, t.pos = (2, 2), (2, 1)
            p.props["might"] = 0.5
            if with_tag:
                p.tags.add("off_balance")
            effs, _ = composer.to_effects(w, p, [("attack", "thief")])
            fx.apply_all(w, effs)
            if t.props["hp"] < 10.0:
                wins += 1
        return wins

    assert hits(True) < hits(False) - 5, "stumbling must cost real odds"
    w, p = vault(seed=1)
    t = w.entities["thief"]
    p.pos, t.pos, p.tags = (2, 2), (2, 1), p.tags | {"off_balance"}
    from src.core import composer as c
    effs, _ = c.to_effects(w, p, [("attack", "thief")])
    from src.core import effects as fx2
    fx2.apply_all(w, effs)
    assert "off_balance" not in p.tags, "the stumble is spent by the next action"


# DESIGN (Ironsworn wiring): ask the world something it doesn't model and the
# oracle rules ONCE — the answer is narrated as canon and never flickers.
def test_unknowable_questions_route_to_the_oracle():
    from src.core.composer import MockPicker
    world, player = vault(seed=5)
    r1 = engine.free_text(world, player, "is there a rat tunnel under the vault?", MockPicker())
    line = " ".join(r1["narrative"]).lower()
    assert "canon" in line and ("yes" in line or "no" in line), f"the oracle rules: {r1['narrative']}"
    r2 = engine.free_text(world, player, "is there a rat tunnel under the vault?", MockPicker())
    assert (" yes" in " ".join(r1["narrative"]).lower()) == (" yes" in " ".join(r2["narrative"]).lower()), \
        "canon does not flicker on re-asking"
