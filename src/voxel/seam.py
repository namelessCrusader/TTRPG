"""The words a decision is written in.

ONE TEXT, READ BY BOTH SIDES. A teacher labelling a decision and a student
learning from that label must see the SAME bytes, or the label is attached to
something the student never gets shown. That is the whole job of this file, and
it is why it is not in `sim.py`: the simulation does not need a decision to have
a spelling, and the moment a model does, the spelling becomes an interface.

THREE RULES, and each of them was chosen for a reason that costs something if
it is broken.

1. DETERMINISTIC TO THE BYTE. Keys sorted, floats to a fixed number of places,
   no dictionary iteration order anywhere. Prompt caching is a prefix match — a
   single byte that moves invalidates everything after it — and a labelling run
   that silently stops hitting cache costs about ten times what it should. A
   test compares the same situation serialised twice.

2. THE REFLEX TABLE IS NOT IN IT. `situation["reflexes"]` is the policy's own
   answer key: hand it to a teacher and the teacher is reading the answer off
   the page, hand it to a student and the student learns to look it up rather
   than to decide. The ablation called this the god-channel and removing it is
   what made the numbers mean anything. It is dropped here rather than at the
   call site so that nobody can forget.

3. SPLIT INTO STABLE AND VARYING. `rules()` is the same for every call in a
   run and is what gets cached; `situation()` is the part that differs. Mixing
   them is the single easiest way to destroy the cache hit rate, so they are
   different functions returning different strings and never concatenated here.

The version string goes in the stable half. A dataset labelled under one
spelling cannot be mixed with a dataset labelled under another, and the only
thing worse than re-labelling is not knowing you have to.
"""
from __future__ import annotations

SPELLING = "mark1-menu-v1"

# WHAT A BODY IS NEVER TOLD ABOUT ITSELF. `reflexes` is the answer key (rule 2).
# `tick` and `who` are facts about the RUN rather than about the situation: a
# body does not decide differently because it is tick 400 or because it is
# called Berel, and putting them in would hand the model a way to memorise runs
# instead of learning situations — as well as busting the cache on every call.
HIDDEN = ("reflexes", "tick", "who")

# The order fields are written in. Fixed here rather than sorted at use, so that
# adding a field is a deliberate edit to this list and bumps SPELLING.
ORDER = ("percept", "hurt", "burn", "smoke", "blood_o2", "holding",
         "knows_a_way_out", "others_in_earshot", "seen_of_the_world",
         "ground_it_trusts")


def _scalar(v):
    """One value, spelled the same way every time.

    Floats get three places whatever they are — `0.0` and `0` and `0.000` are
    the same number and must not be three different prefixes. Booleans are
    words because a text encoder has seen "yes" and "no" rather more often than
    it has seen "True".
    """
    if v is None:
        return "nothing"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def rules(limbs):
    """The half that never changes — cache this.

    Everything a reader needs to interpret a situation and pick from a menu,
    and nothing that varies between decisions. Keep it boring: every edit here
    is a cache miss on the whole run and a reason to bump SPELLING.
    """
    return (
        f"[{SPELLING}]\n"
        "You are deciding what one body does next, in a physical simulation.\n"
        "A body decides with several limbs at once; you are asked about one\n"
        "limb at a time, and told what the other limbs have already chosen.\n"
        "Every option offered is one the world has already checked is possible\n"
        "— you are choosing between real acts, not proposing them.\n"
        "Answer with the NUMBER of the option and nothing else.\n"
        f"Limbs, in the order they are asked: {', '.join(limbs)}.\n"
        "Fields: percept is what the body just noticed, or 'nothing'. hurt,\n"
        "burn, smoke and blood_o2 are that body's own state, 0 to 1. holding\n"
        "is what is in its hands. knows_a_way_out is whether it has seen a\n"
        "door. seen_of_the_world and ground_it_trusts are how much of the\n"
        "place it has looked at and how much of the floor it believes is\n"
        "walkable.\n"
    )


def situation(sit, limb, chosen, menu):
    """The half that differs — do not cache this.

    `sit` is the dict `_law_will` builds, `limb` the one being asked about,
    `chosen` what the other limbs have settled on so far, and `menu` the
    options as the world offered them.
    """
    lines = []
    for k in ORDER:
        if k in HIDDEN or k not in sit:
            continue
        lines.append(f"{k}: {_scalar(sit[k])}")
    # ANYTHING ELSE THE SITUATION CARRIES, sorted, so a field added upstream
    # shows up in the text instead of being silently dropped — and shows up in
    # the same place every time.
    for k in sorted(sit):
        if k in HIDDEN or k in ORDER or k in ("limb", "chosen"):
            continue
        lines.append(f"{k}: {_scalar(sit[k])}")
    done = ", ".join(f"{l}={chosen[l]}" for l in sorted(chosen)) or "nothing yet"
    lines.append(f"already chosen: {done}")
    lines.append(f"limb: {limb}")
    lines.append("options:")
    for i, o in enumerate(menu):
        key = o["key"] if isinstance(o, dict) else str(o)
        lines.append(f"  {i}. {key}")
    return "\n".join(lines) + "\n"


def row(sit, limb, chosen, menu, pick, outcome=None):
    """One labelled decision, ready to be written to disk or sent to a teacher.

    The text is what a model sees; the rest is what a harvest needs to sort,
    weight and de-duplicate rows later. `key` is what "distinct decision" means
    — the census counted 242 rows and 51 of these, and that gap is the whole
    reason this file exists.
    """
    text = situation(sit, limb, chosen, menu)
    keys = [o["key"] if isinstance(o, dict) else str(o) for o in menu]
    return {"spelling": SPELLING,
            "text": text,
            "limb": limb,
            "options": keys,
            "pick": int(pick),
            "outcome": outcome,
            "key": text}
