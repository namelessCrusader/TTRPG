"""Harness ports #6/#3/#8: outcome-graded narration + forced voice prefixes,
the slot/tier prompt assembler, and free-text stall detection.
"""

from src.core import engine, llm, social
from src.core.composer import MockPicker
from src.core.seed import vault


# DESIGN (#6, Clover d20): the ROLL decides the words. A crit reads masterful,
# a partial reads scrappy — graded narration from the band, zero LM.
def test_attack_narration_is_graded_by_band():
    from src.core import composer
    from src.core import effects as fx
    seen = set()
    for s in range(80):
        w, p = vault(seed=s)
        t = w.entities["thief"]
        p.pos, t.pos = (2, 2), (2, 1)
        p.props["might"] = 0.8
        effs, _ = composer.to_effects(w, p, [("attack", "thief")])
        fx.apply_all(w, effs)
        text = " ".join(e.cause for e in w.log if e.cause)
        if "expertly" in text or "clean through" in text:
            seen.add("crit")
        if "glancing" in text or "barely" in text or "scrappy" in text:
            seen.add("partial")
        if len(seen) == 2:
            return
    raise AssertionError(f"80 seeds should show graded narration; saw {seen}")


# DESIGN (#6): the voice model is STEERED BY COMMITTED TOKENS — each stance has
# a forced opener the 0.5B must continue from. Test the pure mapping.
def test_every_stance_has_a_forced_opener():
    for rid in ("cower", "defy", "warm", "wary", "relent", "dismiss", "comply", "refuse"):
        pre = social.stance_prefix(rid)
        assert pre and len(pre) <= 25 and pre[-1] in " —,", \
            f"{rid}: opener must exist and end mid-flow so the model continues: {pre!r}"


# DESIGN (#3): prompts are ASSEMBLED — deterministic slot order, tier-based
# dropping under budget, never truncated mid-line.
def test_prompt_assembler_is_stable_and_budgeted():
    slots = [("persona", "You are Bran.", 0), ("scene", "The hall is burning.", 1),
             ("lore", "The guild was founded long ago.", 2), ("note", "Be terse.", 1)]
    full = llm.assemble(slots, budget=1000)
    assert full.index("You are Bran") < full.index("hall is burning") < full.index("Be terse")
    assert full == llm.assemble(slots, budget=1000), "byte-stable"
    tight = llm.assemble(slots, budget=len(full) - 5)
    assert "founded long ago" not in tight, "lowest tier dropped first"
    assert "You are Bran." in tight and "Be terse." in tight, "high tiers survive whole"
    for line in tight.split("\n"):
        assert not line.endswith(("fou", "lon")), "never mid-line truncation"


# DESIGN (#8, NetPlay stall rule): repeating a failing request must not loop —
# the third identical refusal becomes a nudge to try something else, logged.
def test_repeated_failing_freetext_is_stalled():
    world, player = vault(seed=0)
    outs = []
    for _ in range(3):
        outs.append(" ".join(engine.free_text(world, player, "summon a dragon",
                                              MockPicker())["narrative"]))
    assert outs[0] == outs[1], "same request, same refusal — twice is fine"
    assert outs[2] != outs[1] and ("tried" in outs[2].lower() or "else" in outs[2].lower()), \
        f"the third try gets a nudge, not an echo: {outs[2]}"
    assert any(e.kind == "fallback" for e in world.log), "stall is a typed, harvestable event"


# DESIGN (#7, Kobold banned_strings + Clover sanitizer): the voice lane is
# mechanically hygienic — no RP slop, no meta-leakage, no stat-speak, no
# stutter-loops. Reject means REJECT (None → caller retries or goes canned).
def test_voice_sanitizer_bans_slop_and_stat_speak():
    s = social.sanitize_line
    assert s("*smiles warmly* Good evening.") == "Good evening."
    assert s("As an AI language model, I cannot help.") is None, "meta = reject"
    assert s("I'll do 3 damage to your HP!") is None, "stat-speak = reject"
    assert s("Roll a d20 for me.") is None, "dice-speak = reject"
    assert s("please please please please stop") == "please stop", "stutter collapsed"
    assert s('He said "never') == "He said never", "unbalanced quote stripped"
    assert s("Fine. I'll help you. And another thing about the weather and") == \
        "Fine. I'll help you.", "trailing fragment dropped at sentence boundary"
    assert s("Hand it over, friend.") == "Hand it over, friend.", "clean lines pass untouched"


# DESIGN: a voice that produces slop twice falls back to the canned line —
# the player never sees a broken register.
def test_voice_falls_back_to_canned_when_generation_is_slop():
    calls = {"n": 0}

    class SloppyVoice(social.TorchVoice):
        def __init__(self):
            pass
        def _gen(self, ctx, chosen):
            calls["n"] += 1
            return "As an AI, I cannot roleplay this."

    v = SloppyVoice()
    line = v.line({"tick": 0, "intent": "talk", "npc_name": "Pip"},
                  social.REACTIONS["talk"][0])
    assert line == social.REACTIONS["talk"][0].line, "canned beats slop"
    assert calls["n"] == 2, "one reseeded retry, then give up"