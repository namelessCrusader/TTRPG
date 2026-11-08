"""The scene text: the words a chooser reads about one decision
(docs/sim-mind-design.md 3.5, step M4). core.scene() gives the facts; this
file gives them words. SPELLING names the words: a dataset made with one
spelling is never mixed with another.

THE RULES OF A CLEAN INPUT. The lab measured what the deciding model reads
well (design 1.1), and each rule comes from that:
- It reads stated facts well and infers badly. So the text states the facts
  that the engine knows: who is here and what they hold, the character's
  own place, what it holds and feels, and what it lacks.
- What the character holds is said only when it holds something. "Tom
  holds nothing." cost the model on the event cases (LAB10: people 36.7%
  right against the lab text's 41.1%, personality2 67.1% against 79.3%;
  without that sentence, a tie), so v2 drops it (the user, 2026-10-16).
- A sentence that names a useless thing makes it pick that thing (100%). So
  the text never lists the things that lie about ("Things here: ..."). It
  names a thing only when someone holds it, or when a percept or a belief
  that bears on the menu names it.
- Said as what is MISSING, a need helps it most ("Fenn needs something to
  eat"), more than a list of what is there.
- The last few percepts (at most 5) and at most 2 beliefs.
- No numbers: a chooser never sees a tick, an id or a raw number (design
  2.4). `describe` refuses a text that has a digit, so the backend that wrote
  it is found at once.
The same scene gives the same bytes.

An example (the inn):

    It is night at the village inn. Fenn is a traveller, in the main room.
    Here: Oren (holding the purse of silver and the lute); Sir Ada (holding
    the sword). Fenn holds the dagger. Fenn feels hungry. Fenn needs
    something to eat. Recently: Oren walked to the main room. Fenn heard
    sounds of a fight in the kitchen. Fenn believes the bandit took the axe.
"""
from __future__ import annotations

import re

SPELLING = "mark1-mind-v2"      # v2: no "X holds nothing." (v1 said it)
QUESTION = "What does {name} do next?"
DIGIT = re.compile(r"\d")


def describe(sc, setting=""):
    """The scene text of `sc` (a core.Scene). `setting` is the backend's
    first sentence ("It is night at the village inn"), or ""."""
    s = sc.state
    name = s.name
    out = [setting] if setting else []
    if s.public and s.place:
        out.append(f"{name} is {s.public}, in {s.place}")
    elif s.public or s.place:
        out.append(f"{name} is {s.public}" if s.public else f"{name} is in {s.place}")
    if sc.people:
        out.append("Here: " + "; ".join(_person(p) for p in sc.people))
    if s.holds:
        out.append(f"{name} holds {_and(s.holds)}")
    if s.feels:
        out.append(f"{name} feels {_and(s.feels)}")
    for n in sc.needs:
        if n.blocked is not None:
            out.append(f"{name} needs {_and(n.says)} to {n.blocked.text}")
        else:
            out.append(f"{name} needs {_or(n.says)}")
    if sc.percepts:
        out.append("Recently: " + " ".join(_sentence(f"{name} heard {x.text}" if x.channel == "heard"
                                                     else x.text) for x in sc.percepts))
    for b in sc.beliefs:
        out.append(f"{name} was told that {b.text}" if b.how == "told" else f"{name} believes {b.text}")
    text = " ".join(_sentence(x) for x in out)
    if DIGIT.search(text):
        raise ValueError(f"a scene text has no numbers, and this one has: {text!r}")
    return text


def question(state):
    """The question a chooser answers: "What does Fenn do next?"."""
    return QUESTION.format(name=state.name)


def _person(p):
    """ "Oren (holding the lute)": a person the scene names."""
    more = list(p.state) + ([f"holding {_and(p.holds)}"] if p.holds else [])
    return f"{p.who.name} ({', '.join(more)})" if more else p.who.name


def _sentence(x):
    """A sentence: a capital first letter and a full stop."""
    x = x.strip()
    if not x:
        return x
    x = x[0].upper() + x[1:]
    return x if x[-1] in ".!?" else x + "."


def _join(words, last):
    words = list(words)
    if len(words) < 2:
        return "".join(words)
    return ", ".join(words[:-1]) + f" {last} " + words[-1]


def _and(words):
    return _join(words, "and")


def _or(words):
    return _join(words, "or")
