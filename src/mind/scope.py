"""The two-level menu: first SCOPE, then ACT (step M12).

A small model picks well from a few options and badly from many (the lab:
Laya right 19 of 20 times from 4 options, 11 from 16, 3 from 64), and a cut
made by the rules before it reads drops its own pick (M9: in 28% of the
requests the cut changed). An engine gate that hides an option is the engine
choosing for the character. So a character is asked small questions:

    level 1  SCOPE   what to turn to: "deal with Oren", "do something with
                     the axe", "go somewhere or leave", "wait", ... and
                     "something else", which holds every other act
    level 2  ACT     the concrete options inside that scope, in canonical
                     order: "strike Oren with the axe", "shove Oren", ...
    level 3  MOTOR   the backend: paths, reach, limbs (as before)

NOTHING IS CUT. The scopes PARTITION the full menu: every legal option is in
exactly one scope, and every option can be picked in every decision. The
backend filters only by fact (what is possible from here). Which scope
matters is the character's choice.

THE SCOPES of one decision. Each legal option goes to the first of these
that fits it:
    keep    the option of the character's last job, when that job was
            stopped or failed and the option is still legal: "keep at it
            (take the iron)"
    wait    the null act
    deal    an act on a person (the `target` role is a person), not social:
            "deal with Oren"
    talk    a social act on a person: "talk with Oren"
    thing   an act on a thing (a role of kind thing): "do something with
            the axe"
    go      an act toward a place: "go somewhere or leave"
    call    the rest, an act with no one and nothing to aim at: "call out"
A scope with more than ACT_MAX options is SPLIT into parts by its verbs, in
canonical order: whole verbs go together while they fit ("deal with Oren:
strike or shove", "deal with Oren: give, tell or threaten"), and a verb with
more options than a part holds is split into even pieces ("talk with Sir
Ada: tell (1 of 2)"). So every entry that leads to level 2 holds ACT_MAX
options or fewer, and level 2 shows all of them.

LEVEL 1 has at most SCOPE_MAX entries. When the scopes are more, coarser
scopes are tried (COARSER): one scope for each person (talk goes into "deal
with Oren"), then also one for every thing ("get or use something"). When
there are still too many, the most SALIENT scopes stay alone and the rest are
PACKED into bins of ACT_MAX options or fewer (the biggest first, each with
its own kind when it can), as many alone as can be. A bin is one entry,
named by what is in it: "do something with the dice or the stool; go
somewhere or leave; or deal with Oren or Tom: threaten". Salience is read off the situation, never
off the rule scores (that would be the rules' cut again): first the job it
was doing (keep), then the scopes about someone or something in the recent
percepts (what is happening), then the scopes that serve a drive the
character has now, then the rest; ties in canonical order. The entries are
shown in canonical order (the place of their first option). Never by score:
a list sorted by score tells a model the answer. The layout taken is the one
that asks the fewest questions at most, then the one with no "something
else", then the one with the fewest bins, then the finest.

SHORT AND UNAMBIGUOUS NAMES (wave 11; M12's verifier read the old ones as a
small model would):
- The parts of a bin's name are cut by ";", and the last by "; or": "deal
  with Sir Ada: steal; or call out". A comma only joins names or verbs
  inside one part. (The old "deal with Sir Ada: steal, keep at it (tell
  Tom that ...), or call out" read as if "keep at it" were one of Sir
  Ada's verbs.)
- "wait" and "keep at it (...)" are entries of their own, and a bin's name
  is at most NAME_MAX characters (the old ones reached 192: "do something
  with the copper coin, the dagger, the lantern, the purse of silver or the
  stool, deal with Sir Ada: ...").
- These give way, one rung of LADDER at a time, only when a page cannot be
  laid out without "something else" otherwise: first "wait" may join a bin
  ("...; or wait"), then a name may be long, then "keep at it" may join a
  bin. A third question costs more than a long name.
lab/methods/scope_eval.py measures what the names cost the model.

THE RULES ON A SCOPE. The rule score of a scope is the soft maximum of the
rule scores of its options, t * log(sum(exp(score / t))), with t the rules'
temperature, for this character (prior) and for an average one (prior_avg);
its habit is 0 (each option's habit is inside). So the rules' probability of
a scope is the sum of the probabilities they give its options, and a draw of
a scope, then of an act in it, is exactly the flat draw over the full menu:
the hierarchy changes what a chooser reads, never what the rules want. No
scope is at zero. A scope is worth what the best things in it are worth to
this character: the backend scores the options from the character's drives
and traits (the voxel sheet; the inn's rules).

Pure: the same input gives the same output, in every process. Standard
library only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .api import Menu, Ref
from .core import T_RULES

SCOPE_MAX = 6       # the most entries of level 1, "wait" and "something else" in them
ACT_MAX = 8         # the most options of level 2
# the longest name of a bin, in characters, or None for no cap. The user
# chose no cap (2026-10-16): on GLiNER the 80 cap cost 2.3 points of scope
# recall, and the names stay unambiguous without it (keep and wait alone,
# parts cut by ";"). Re-test a cap when a trained or different model comes in.
NAME_MAX = None
# THE LADDER: how far the names give way before a third question. Each rung
# is (the kinds of scope that may go into a bin, the longest name of a bin);
# the first rung that lays the page out in one question is taken, and when
# none does, "something else" opens a page (with the first rung's rules).
LADDER = (((), NAME_MAX),                   # keep and wait alone; short names
          (("wait",), NAME_MAX),            # wait in a bin: "...; or wait"
          (("wait",), None),                # names of any length
          (("wait", "keep"), None))         # keep in a bin too (wave 10's layout, with ";")
# the kinds of scope, in the order an option is put in them
KINDS = ("keep", "wait", "deal", "talk", "thing", "things", "go", "call")   # and BIN, ELSE
# COARSER SCOPES: the fine ones; one scope for each person (talk goes into
# deal); also every thing in one scope. Each is laid out, and the cheapest
# layout is taken (the top of this file).
COARSER = ((), ("person",), ("person", "things"))
ELSE = "else"       # the kind of "something else"
BIN = "mixed"       # the kind of an entry that holds scopes of more than one kind
SOCIAL = "social"   # the catalog group whose acts on a person are talk
WORDS = {"keep": "keep at it ({text})", "wait": "wait", "deal": "deal with {name}",
         "talk": "talk with {name}", "thing": "do something with {name}",
         "things": "get or use something",
         "go": "go somewhere or leave", "call": "call out", ELSE: "something else"}
PART = "{text}: {verbs}"        # a part of a split scope
PIECE = "{verb} ({k} of {n})"   # a piece of a verb split in even pieces
# what the next question says of the entry picked: "Fenn means to deal with Oren."
MEANS = "{name} means to {what}."
ELSE_MEANS = "do something else"


@dataclass(frozen=True)
class Scope:
    """One entry of a page. `members` are the indices of its options in
    the full menu (`Level.full.options`), in canonical order; `ref` is the
    person or thing it is about, or None."""
    kind: str
    text: str
    members: tuple[int, ...]
    ref: Ref | None = None


@dataclass(frozen=True)
class Level:
    """A page of scopes: level 1 of one decision, or the page that "something
    else" opens.

    `full` is the menu with nothing cut (core.menu with player=True). `scopes`
    are the entries, in the order a chooser reads them (a scope, a part of
    one, or a bin of them); `prior`, `prior_avg` and `habit` are their rule
    numbers. `acts[j]` is what entry j leads to: the level-2 Menu of all its
    options in canonical order (never more than ACT_MAX, and nothing in its
    `cut`), or, for "something else", the Level of the scopes it folded.
    `folded` are the scopes that "something else" holds (empty when there is
    no "something else")."""
    full: Menu
    scopes: tuple[Scope, ...]
    prior: tuple[float, ...]
    prior_avg: tuple[float, ...]
    habit: tuple[float, ...]
    acts: tuple  # of Menu | Level
    folded: tuple[Scope, ...] = ()
    relaxed: int = 0        # the rung of LADDER the page was laid out on

    @property
    def texts(self):
        return tuple(s.text for s in self.scopes)

    @property
    def groups(self):
        """The `group` of each entry: "scope"."""
        return ("scope",) * len(self.scopes)

    @property
    def depth(self):
        """The most questions a decision on this page can take."""
        return 1 + max(a.depth if isinstance(a, Level) else 1 for a in self.acts)

    def of(self, option):
        """The index of the entry that holds `option` (an Option of the full
        menu)."""
        i = self.full.options.index(option)
        return next(j for j, s in enumerate(self.scopes) if i in s.members)


def soft_max(scores, t=T_RULES):
    """t * log(sum(exp(x / t))): the rule score of a set of options whose
    probability is the sum of theirs."""
    m = max(scores)
    return m + t * math.log(sum(math.exp((x - m) / t) for x in scores))


def _kind(o, null, last, merged=()):
    """(the kind of scope of option `o`, the Ref it is about). `merged`
    names the coarser scopes in use: "person" (talk goes into deal) and
    "things" (every thing in one scope)."""
    if o.verb == null:                  # (an interrupted wait is still "wait")
        return "wait", None
    if last is not None and o.key == last:
        return "keep", None
    target = o.role("target")
    if target is not None and target.kind == "person":
        return ("talk" if o.group == SOCIAL and "person" not in merged else "deal"), target
    thing = next((r for _, r in o.roles if r.kind == "thing"), None)
    if thing is not None:
        return ("things", None) if "things" in merged else ("thing", thing)
    if any(r.kind == "place" for _, r in o.roles):
        return "go", None
    return "call", None


def _text(kind, ref, members, full):
    if kind == "keep":
        return WORDS[kind].format(text=full.options[members[0]].text)
    return WORDS[kind].format(name=ref.name if ref is not None else "")


def partition(full, cat, last=None, merged=()):
    """The scopes of the full menu `full`, every one of them, in canonical
    order. `last` is the key of the option of a job that was stopped or
    failed (keep), or None; `merged` the coarser scopes in use (`_kind`)."""
    null = cat.null_act.id
    got = {}
    for i, o in enumerate(full.options):
        kind, ref = _kind(o, null, last, merged)
        # by the words: two things with one name are one scope to a reader
        name = ref.name if ref is not None else ""
        got.setdefault((kind, name), (ref, []))[1].append(i)
    out = [Scope(k, _text(k, ref, tuple(ix), full), tuple(ix), ref) for (k, _), (ref, ix) in got.items()]
    # the options are in canonical order already: a scope's place is its first option's
    return sorted(out, key=lambda s: s.members[0])


def _said(words):
    """"a", "a or b", "a, b or c"."""
    return words[0] if len(words) == 1 else f"{', '.join(words[:-1])} or {words[-1]}"


def split(s, full, cap=ACT_MAX):
    """Scope `s` as parts of `cap` options or fewer (the top of this file):
    [s] when it is no bigger. Every member is in exactly one part."""
    if len(s.members) <= cap:
        return [s]
    by = {}                             # the verbs, in canonical order
    for i in s.members:
        by.setdefault(full.options[i].verb, []).append(i)
    pieces = []                         # (words, members): a whole verb, or an even piece of one
    for v, ix in by.items():
        verb = v.replace("_", " ")
        n = -(-len(ix) // cap)
        size = -(-len(ix) // n)
        pieces += [([verb if n == 1 else PIECE.format(verb=verb, k=k + 1, n=n)], ix[a:a + size])
                   for k, a in enumerate(range(0, len(ix), size))]
    parts = [pieces[0]]
    for words, ix in pieces[1:]:        # together, in order, while they fit
        if len(parts[-1][1]) + len(ix) <= cap:
            parts[-1] = (parts[-1][0] + words, parts[-1][1] + ix)
        else:
            parts.append((words, ix))
    return [Scope(s.kind, PART.format(text=s.text, verbs=_said(words)), tuple(ix), s.ref)
            for words, ix in parts]


def salience(s, full, drives, percepts, cat):
    """The rank of scope `s` by the situation, lower first: 0 the job it was
    doing, 1 about someone or something in the recent percepts, 2 serving a
    drive the character has now, 3 the rest."""
    if s.kind == "keep":
        return 0
    if s.ref is not None and any(r is not None and r.name == s.ref.name
                                 for x in percepts for r in (x.actor, x.target)):
        return 1
    if any(set(cat.verbs[full.options[i].verb].serves) & set(drives) for i in s.members):
        return 2
    return 3


def _menu(full, members):
    """The level-2 Menu of an entry: all its options, nothing cut."""
    o = full.options
    return Menu(tuple(o[i] for i in members), full.blocked, tuple(o[i] for i in members), (),
                tuple(full.prior[i] for i in members), tuple(full.prior_avg[i] for i in members),
                tuple(full.habit[i] for i in members))


def _about(s):
    """(kind, verbs) of scope `s` when it is about one person or thing: a
    whole scope (verbs "") or a part of a split one; else None."""
    if s.ref is None or s.kind not in ("deal", "talk", "thing"):
        return None
    whole = WORDS[s.kind].format(name=s.ref.name)
    if s.text == whole:
        return s.kind, ""
    head = PART.format(text=whole, verbs="")
    return (s.kind, s.text[len(head):]) if s.text.startswith(head) else None


def _bin(scopes):
    """One entry that holds the scopes `scopes` (canonical order), named by
    what is in it: "deal with Fenn or Marta, talk with Oren, or call out";
    the parts with the same verbs together: "deal with Oren or Tom: threaten"."""
    if len(scopes) == 1:
        return scopes[0]
    phrases, names = [], {}
    for s in scopes:
        key = _about(s)
        if key is None:
            phrases.append(s.text)
        else:
            if key not in names:
                names[key] = []
                phrases.append(key)
            names[key].append(s.ref.name)
    words = [x if isinstance(x, str) else
             (PART.format(text=WORDS[x[0]].format(name=_said(names[x])), verbs=x[1]) if x[1]
              else WORDS[x[0]].format(name=_said(names[x]))) for x in phrases]
    text = words[0] if len(words) == 1 else f"{'; '.join(words[:-1])}; or {words[-1]}"
    kinds = {s.kind for s in scopes}
    refs = {s.ref for s in scopes}
    return Scope(kinds.pop() if len(kinds) == 1 else BIN, text,
                 tuple(sorted(i for s in scopes for i in s.members)), refs.pop() if len(refs) == 1 else None)


def _ffd(units, cap, name_max=NAME_MAX):
    """Units (lists of scopes) in bins of `cap` options or fewer and names
    of `name_max` characters or fewer (None: any): the biggest first, each
    into the first bin with room, else a new bin."""
    n = lambda u: sum(len(x.members) for x in u)
    room = lambda b, u: (n(b) + n(u) <= cap and (
        name_max is None or len(_bin(sorted(b + u, key=lambda s: s.members[0])).text) <= name_max))
    bins = []
    for u in sorted(units, key=lambda u: (-n(u), u[0].members[0])):
        fit = next((b for b in bins if room(b, u)), None)
        if fit is None:
            bins.append(list(u))
        else:
            fit += u
    return [sorted(b, key=lambda s: s.members[0]) for b in bins]


def _pack(scopes, cap, name_max=NAME_MAX):
    """The scopes in bins of `cap` options or fewer (and names of `name_max`
    characters or fewer), each bin a list in canonical order: first the
    scopes of each kind together ("do something with the dice or the
    stool"), then those bins together; or all at once, when that makes
    fewer bins."""
    kinds = {}
    for s in scopes:
        kinds.setdefault(s.kind, []).append([s])
    by_kind = _ffd([b for u in kinds.values() for b in _ffd(u, cap, name_max)], cap, name_max)
    at_once = _ffd([[s] for s in scopes], cap, name_max)
    return by_kind if len(by_kind) <= len(at_once) else at_once


def _page(entries, full, cat, drives, percepts, scope_max, act_max, t, ladder=None):
    """The Level of `entries` (scopes of at most `act_max` options, in
    canonical order; the top of this file): all of them when they fit; else
    the most salient alone and the rest packed in bins, as many alone as can
    be, on the first rung of `ladder` (LADDER) that fits; else the bins,
    the most salient first, and "something else", which opens a page of the
    bins that do not fit."""
    ladder = LADDER if ladder is None else ladder
    keep = [s for s in entries if s.kind == "keep"]
    wait = [s for s in entries if s.kind == "wait"]
    rest = [s for s in entries if s.kind not in ("keep", "wait")]
    folded, shown, rung = (), None, 0
    if len(rest) <= scope_max - len(keep) - len(wait):
        shown = list(entries)
    else:
        rank = {s: (salience(s, full, drives, percepts, cat), s.members[0]) for s in entries}
        ranked = sorted(rest, key=rank.__getitem__)
        for rung, (binned, name_max) in enumerate(ladder):
            # k scopes alone and the rest in bins, as many alone as can be;
            # a keep that may be binned is still the first to stay alone
            alone = [s for s in keep + wait if s.kind not in binned]
            others = ([s for s in keep if s.kind in binned] + ranked
                      + [s for s in wait if s.kind in binned])
            r = scope_max - len(alone)
            k = next((k for k in range(r - 1, -1, -1)
                      if k + len(_pack(others[k:], act_max, name_max)) <= r), None)
            if k is not None:
                shown = sorted(others[:k] + [_bin(b) for b in _pack(others[k:], act_max, name_max)] + alone,
                               key=lambda s: s.members[0])
                break
        if shown is None:                            # too many for two questions
            rung, name_max = 0, ladder[0][1]
            alone = keep + wait
            room = scope_max - len(alone)
            bins = sorted(_pack(rest, act_max, name_max), key=lambda b: min(rank[s] for s in b))
            named = [_bin(b) for b in bins[:room - 1]]
            folded = tuple(_bin(b) for b in bins[room - 1:])
            other = Scope(ELSE, WORDS[ELSE], tuple(sorted(i for s in folded for i in s.members)))
            shown = sorted(named + alone, key=lambda s: s.members[0]) + [other]
    acts = tuple(_page(folded, full, cat, drives, percepts, scope_max, act_max, t, ladder)
                 if s.kind == ELSE else _menu(full, s.members) for s in shown)
    prior = tuple(soft_max([full.prior[i] + full.habit[i] for i in s.members], t) for s in shown)
    avg = tuple(soft_max([full.prior_avg[i] + full.habit[i] for i in s.members], t) for s in shown)
    return Level(full, tuple(shown), prior, avg, (0.0,) * len(shown), acts, folded, rung)


def levels(full, cat, drives=(), percepts=(), last=None, scope_max=SCOPE_MAX, act_max=ACT_MAX,
           t=T_RULES, ladder=None):
    """The pages of one decision: its level 1 (a `Level`). `full` is the
    uncut menu, `drives` the character's drives now (SelfState.drives),
    `percepts` the ones the scene shows, `last` the key of the option of its
    last job when that job was stopped or failed; `ladder` the rungs the
    names may give way on (LADDER; LADDER[:1] never gives way)."""
    if scope_max < 3:
        raise ValueError("level 1 has room for a scope, wait and something else")
    if act_max < 1:
        raise ValueError("level 2 has room for an option")
    if last is not None and not any(o.key == last for o in full.options):
        last = None
    best = None
    for merged in COARSER:
        entries = sorted((p for s in partition(full, cat, last, merged) for p in split(s, full, act_max)),
                         key=lambda s: s.members[0])
        lv = _page(entries, full, cat, drives, percepts, scope_max, act_max, t, ladder)
        # the fewest questions, then nothing folded, then the strictest names,
        # then the fewest bins
        cost = (lv.depth, bool(lv.folded), lv.relaxed, sum(s not in entries for s in lv.scopes))
        if best is None or cost < best[0]:
            best = cost, lv
        if cost == (2, False, 0, 0):
            break
    return best[1]


def means(name, scope):
    """The sentence the next question adds to the scene: "Fenn means to deal
    with Oren"."""
    what = ELSE_MEANS if scope.kind == ELSE else scope.text
    return MEANS.format(name=name, what=what)


def joint(lv, path, act, rules):
    """The probability of each option of the full menu as one draw: the
    product of the draws on the way to it. `path` lists the pages asked, top
    first, as (page, the entry picked, the draw over its entries); `act` is
    the draw over the options of the last entry picked; `rules(members)` is
    the draw over those options of the full menu for an entry not picked."""
    out = [0.0] * len(lv.full.options)
    w = 1.0
    for page, j, p in path:
        for k, s in enumerate(page.scopes):
            if k != j:
                for i, q in zip(s.members, rules(s.members)):
                    out[i] = w * p[k] * q
        w *= p[j]
    page, j, _ = path[-1]
    for i, q in zip(page.scopes[j].members, act):
        out[i] = w * q
    return tuple(out)


__all__ = ["ACT_MAX", "BIN", "ELSE", "KINDS", "LADDER", "Level", "NAME_MAX", "SCOPE_MAX", "Scope", "joint", "levels", "means",
           "partition", "salience", "soft_max", "split"]
