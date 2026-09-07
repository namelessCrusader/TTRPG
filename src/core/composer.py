"""Free-text action composer — the player's "do something…" box.

No generation, no JSON, no grammar library: the utterance is compiled to typed
deltas by DEPENDENCY-ORDERED SLOT SELECTION. Route first (physical / social /
inspect / refuse), then per step: verb → its slots, each slot's candidates
queried LIVE from world state (so the model can only ever choose things that
exist and are in reach), each pick a single PMI selection the 0.6B does
reliably. Earlier steps are threaded into later steps' context ("already done:
poured oil at (4,0)"), which is how "pour oil east THEN LIGHT IT" grounds.

Output is canonical step tuples; `to_effects` compiles them to bus deltas. The
composer interprets — the reaction sim adjudicates what actually happens.
"""

from __future__ import annotations

import re

from . import combat
from .state import FLUIDS, World, is_solid, mat_tags

MAX_STEPS = 3
THROW_RANGE = 3
DIRNAME = {(0, -1): "north", (0, 1): "south", (1, 0): "east", (-1, 0): "west",
           (1, -1): "northeast", (-1, -1): "northwest", (1, 1): "southeast", (-1, 1): "southwest"}

SYS = ("You are the rules engine of a fantasy game. Read what the player says "
       "and choose the option that best matches their intent. Answer with the option word.")


# ── world queries (live; no knowledge graph — the state IS the knowledge) ──
def _l1(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _cell_bits(world, p):
    c = world.cell(p)
    bits = []
    if "on_fire" in c.tags:
        bits.append("burning fire")
    bits += [f"{m} ({int(ml)}ml)" for m, ml in c.fluids.items()]
    if c.material:
        bits.append(f"{c.material} block")
    for e in world.entities_at(p):
        if "player" not in e.tags and "taken" not in e.tags:
            bits.append(e.name + (" [dead]" if "dead" in e.tags else ""))
    return ", ".join(bits) if bits else "bare floor"


def _reach(world, player):
    """[(pos, direction-word, description)] for here + the four neighbours."""
    out = [(player.pos, "here", _cell_bits(world, player.pos))]
    for d, name in DIRNAME.items():
        p = (player.pos[0] + d[0], player.pos[1] + d[1], player.pos[2])
        if world.in_bounds(p):
            out.append((p, name, _cell_bits(world, p)))
    return out


def _flammable(world, p):
    c = world.cell(p)
    if "on_fire" in c.tags:
        return False
    return (any(FLUIDS.get(m, {}).get("flammable") and ml > 0 for m, ml in c.fluids.items())
            or "flammable" in mat_tags(c.material)
            or any("flammable" in mat_tags(e.material) and "on_fire" not in e.tags and "broken" not in e.tags
                   for e in world.entities_at(p) if "item" in e.tags))


def _carried_fluids(player):
    return [t[4:] for t in player.tags if t.startswith("inv:") and t[4:] in FLUIDS]


def _people_in_reach(world, player):
    return [e for p, _, _ in _reach(world, player) for e in world.entities_at(p)
            if "person" in e.tags and "dead" not in e.tags]


def _items_in_reach(world, player):
    return [e for p, _, _ in _reach(world, player) for e in world.entities_at(p)
            if "item" in e.tags and "taken" not in e.tags and "heavy" not in e.tags]


def _breakables_in_reach(world, player):
    return [e for p, _, _ in _reach(world, player) for e in world.entities_at(p)
            if "container" in e.tags and "broken" not in e.tags]


def _carried_items(world, player):
    return [world.entities[t[4:]] for t in player.tags
            if t.startswith("inv:") and t[4:] in world.entities]


def _notable_in_range(world, player, rng):
    out = []
    for p, c in world.cells.items():
        if _l1(p, player.pos) <= rng and (c.tags or c.fluids or c.material):
            out.append(p)
        elif _l1(p, player.pos) <= rng and any(
                "person" in e.tags and "dead" not in e.tags for e in world.entities_at(p)):
            out.append(p)
    return out


def _away_dest(world, eid, ref):
    """Deterministic 'shove/pull X away from ref' destination (shared with eval)."""
    e = world.entities[eid]
    opts = [p for p in world.neighbors(e.pos) if not is_solid(world.cell(p).material)]
    if not opts:
        return e.pos
    return max(sorted(opts), key=lambda p: _l1(p, ref))


def _bfs_adjacent(world, start, goal):
    """Shortest passable path from start to a cell adjacent to goal (auto-goto)."""
    from collections import deque
    q, seen = deque([(start, [])]), {start}
    while q:
        pos, path = q.popleft()
        if _l1(pos, goal) == 1:
            return path
        for n in world.neighbors(pos):
            if n not in seen and not is_solid(world.cell(n).material):
                seen.add(n)
                q.append((n, path + [n]))
    return None


# ── the pickers: options are (id, desc) pairs. The DESC goes in the prompt
# listing; only the short, semantically-loaded ID is scored (the social-layer
# lesson: PMI over long sentence labels drowns; loaded single words work). ──
class TorchPicker:
    def __init__(self, model_id="Qwen/Qwen3-0.6B"):
        from .llm import get_lm
        self._lm = get_lm(model_id)

    def pick_scored(self, scene, utterance, question, options):
        ids = [o[0] for o in options]
        listing = "\n".join(f"- {i}: {d}" if d else f"- {i}" for i, d in options)
        tail = f"\n{question}\nAnswer with exactly one of: {', '.join(ids)}."
        cond = f"{scene}\nThe player says: \"{utterance}\"\n{listing}{tail}"
        neut = f"{scene}\nThe player says something.\n{listing}{tail}"
        return self._lm.pick_scored(SYS, cond, neut, ids)

    def pick(self, scene, utterance, question, options):
        return self.pick_scored(scene, utterance, question, options)[0]


class MockPicker:
    """Keyword-overlap picker — deterministic harness bring-up, no model."""

    STOP = {"the", "and", "with", "onto", "into", "them", "they", "want", "this",
            "that", "something", "someone", "one", "step", "back"}

    def pick_scored(self, scene, utterance, question, options):
        u = set(utterance.lower().replace(",", " ").split())
        scores = [float(sum(1 for w in f"{i} {d or ''}".lower().replace("(", " ").replace(")", " ").split()
                            if len(w) > 2 and w not in self.STOP and w in u)) for i, d in options]
        return max(range(len(options)), key=lambda i: scores[i]), scores

    def pick(self, scene, utterance, question, options):
        return self.pick_scored(scene, utterance, question, options)[0]


class DebugPicker:
    """Wraps a picker; prints every selection with its scores (eval --debug)."""

    def __init__(self, inner):
        self.inner = inner

    def pick_scored(self, scene, utterance, question, options):
        i, scores = self.inner.pick_scored(scene, utterance, question, options)
        print(f"    Q: {question}")
        for j, (oid, _) in enumerate(options):
            mark = " <==" if j == i else ""
            print(f"       {scores[j]:+7.3f}  {oid}{mark}")
        return i, scores

    def pick(self, scene, utterance, question, options):
        return self.pick_scored(scene, utterance, question, options)[0]


# ── the composer ──
# verb → (scored id, prompt description). Ids are the words players actually say.
VERB_OPTS = {"pour": ("pour", "pour, splash or dump a carried liquid onto something"),
             "ignite": ("light", "set something on fire with your torch"),
             "smother": ("extinguish", "put out a fire — douse, smother or beat it out"),
             "shove": ("shove", "shove, push, pull or drag a person"),
             "take": ("grab", "pick up, take or snatch an object"),
             "throw": ("throw", "throw, toss or hurl a carried object somewhere"),
             "walk": ("walk", "walk, go or step in a direction"),
             "smash": ("smash", "smash, break or kick open an object"),
             "attack": ("attack", "strike, hit or stab a person"),
             "give": ("give", "hand a carried object to a person"),
             "stop": ("stop", "nothing more — the request is complete")}
ROUTE_OPTS = {"talk": ("talk", "this is speech — asking, telling or persuading a person"),
              "look": ("look", "this is a question — they want to see or know, not act"),
              "refuse": ("impossible", "cannot be done here — magic, summoning, time travel")}


def _main(world, p):
    """The one word that names what's at a cell — the semantic hook for scoring."""
    c = world.cell(p)
    if "on_fire" in c.tags:
        return "fire"
    for e in world.entities_at(p):
        if ("person" in e.tags or "item" in e.tags) and "taken" not in e.tags and "dead" not in e.tags:
            return e.name.split()[0]
    if c.fluids:
        return max(c.fluids, key=c.fluids.get)
    if "support" in c.tags:
        return "beam"          # world-data naming: the seed calls it the timber support
    if c.material:
        return c.material
    return "floor"


def _scene(world, player, done):
    lines = [f"Scene: you stand at {player.pos}."]
    inv = sorted(t[4:] for t in player.tags if t.startswith("inv:"))
    lines.append("You carry: " + (", ".join(inv) if inv else "nothing") + ".")
    for p, name, desc in _reach(world, player):
        lines.append(f"  {name} {p}: {desc}")
    for s in done:
        lines.append(f"Already done this turn: {s}")
    return "\n".join(lines)


def _verbs_available(world, player, steps, virtual=()):
    v = []
    if _carried_fluids(player):
        v.append("pour")
    if "inv:torch" in player.tags and (any(_flammable(world, p) for p, _, _ in _reach(world, player))
                                       or any(_flammable(world, p) for p in _notable_in_range(world, player, 99))):
        v.append("ignite")
    if virtual or any("on_fire" in world.cell(p).tags for p, _, _ in _reach(world, player)):
        v.append("smother")
    if _people_in_reach(world, player):
        v.append("shove")
    if _items_in_reach(world, player):
        v.append("take")
    if _carried_items(world, player):
        v.append("throw")
    if _breakables_in_reach(world, player):
        v.append("smash")
    if _people_in_reach(world, player):
        v.append("attack")
        if _carried_items(world, player):
            v.append("give")
    v.append("walk")
    if steps:
        v.append("stop")
    return v


# ── lexical grounding: closed-class words the player says settle slots
# deterministically; the LM only judges what words can't (deixis, vagueness).
SYN = {"flames": "fire", "blaze": "fire", "flame": "fire", "support": "beam",
       "timber": "beam", "wood": "beam", "wooden": "beam", "ground": "floor",
       "puddle": "water", "slick": "oil", "pool": "oil",
       "soak": "water", "feet": "here", "underfoot": "here", "myself": "here",
       "thief": "cutpurse"}
_DEIXIS = {"it", "that", "there", "them"}

# explicit verb words resolve deterministically (closed class); the LM only
# judges verbless/vague clauses ("weaken it", "make it stop", "get rid of...")
VERB_LEX = {"pour": "pour", "splash": "pour", "dump": "pour", "spill": "pour",
            "soak": "pour", "drench": "pour", "feed": "pour",
            "light": "ignite", "ignite": "ignite", "burn": "ignite", "torch": "ignite",
            "set": "ignite", "lite": "ignite",
            "smother": "smother", "extinguish": "smother", "beat": "smother",
            "shove": "shove", "push": "shove", "pull": "shove", "drag": "shove", "haul": "shove",
            "grab": "take", "take": "take", "snatch": "take", "pick": "take",
            "throw": "throw", "toss": "throw", "hurl": "throw", "lob": "throw",
            "walk": "walk", "go": "walk", "run": "walk", "step": "walk",
            "smash": "smash", "break": "smash", "shatter": "smash", "crack": "smash",
            "kick": "smash", "bust": "smash",
            "attack": "attack", "hit": "attack", "strike": "attack", "stab": "attack",
            "punch": "attack", "fight": "attack",
            "give": "give", "hand": "give", "offer": "give",
            "tell": "talk", "ask": "talk", "say": "talk", "talk": "talk", "speak": "talk",
            "greet": "talk", "hello": "talk", "hi": "talk", "hey": "talk", "chat": "talk",
            "converse": "talk", "befriend": "talk", "persuade": "talk", "thank": "talk",
            "warn": "talk", "threaten": "talk", "reassure": "talk", "address": "talk",
            "call": "talk", "shout": "talk", "whisper": "talk", "beg": "talk", "plead": "talk",
            "look": "look", "see": "look", "examine": "look", "inspect": "look",
            "drink": "drink", "sip": "drink", "quaff": "drink"}


def _ed1(a, b):
    """edit distance ≤ 1 (typo tolerance for the closed vocabularies)."""
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1 or len(a) < 3:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    s, l = (a, b) if len(a) < len(b) else (b, a)
    for i in range(len(l)):
        if l[:i] + l[i + 1:] == s:
            return True
    return False


def _lex_verbs(clause, available, player, world=None):
    """Deterministic verb resolution from explicit words. Returns candidate verbs
    (∩ available); empty = let the LM judge."""
    toks = clause.lower().replace(",", " ").replace(".", " ").replace("!", " ").split()
    raw = set(toks)
    exact = any(w in VERB_LEX for w in raw)
    hits = []
    for w in raw:
        v = VERB_LEX.get(w)
        # typo tolerance ("poor watr" → pour) is a LAST resort: only when NO word
        # matched exactly, only in VERB POSITION (imperatives lead with the verb —
        # a trailing noun is a noun: "listen at the WALL" is not walk), same first
        # letter, and never a plural-for-verb ("wash my handS" is not "hand me") —
        # real English kept falling into the typo net (playtests: "fertile LAND" →
        # hand → give; "COOK a meal" → look).
        if v is None and len(w) >= 4 and not exact and w in toks[:2]:
            v = next((VERB_LEX[k] for k in VERB_LEX
                      if _ed1(w, k) and w[0] == k[0] and w != k + "s"), None)
        if v == "ignite" and w == "set" and not ({"fire", "alight", "light", "ablaze"} & raw):
            continue                                   # "set" alone isn't ignition
        if v == "pour" and w == "feed" and not ({"fire", "flames", "blaze"} & raw
                                                or raw & set(FLUIDS)):
            continue                                   # "feed the chickens" is not arson
        if v:
            hits.append(v)
    if "ignite" in hits and "torch" in raw:
        hits = ["ignite"]                              # the torch is the instrument: "throw my torch" = ignite
    elif "throw" in hits and _words(clause) & set(_carried_fluids(player)):
        hits = ["pour" if h == "throw" else h for h in hits]   # "throw water" = pour it
    if "douse" in raw:                                 # douse = pour-liquid if one is named, else smother
        hits.append("pour" if _words(clause) & set(FLUIDS) else "smother")
    if "put" in raw and "out" in raw:
        hits.append("smother")
    if ("get" in raw and raw & {"clear", "away", "out", "back"}) or ("back" in raw and raw & {"away", "off"}):
        hits.append("walk")
    if "look" in hits and "take" in hits:
        hits = [h for h in hits if h != "take"]        # "take a look" is looking
    if "take" in hits and raw & {"breath", "breather", "stock", "nap", "moment", "rest",
                                 "swig", "sip", "break"}:
        hits = [h for h in hits if h != "take"]        # "take a deep breath" grabs nothing
    if "take" in hits and "take" not in available and "shove" in available and \
            world is not None and _lex_match(clause, _people_in_reach(world, player),
                                             lambda e: e.name):
        hits = ["shove" if h == "take" else h for h in hits]   # "grab HIM" = grapple —
        # but ONLY when a person is NAMED ("I grab the longsword" must never haul
        # the broodmother; playtest hit this twice, mid-fight)
    out, recognized = [], []
    for h in hits:
        if h not in recognized:
            recognized.append(h)
        if h in (available + ["talk", "look"]) and h not in out:
            out.append(h)
    return out, recognized


def _named_target_check(world, player, utterance, valid_cells, rng, what, verb_phrase=None):
    """Violence discipline: if the utterance NAMES a person/thing that exists but
    isn't reachable, refuse with the reason. If it names nothing and several
    victims are possible, ASK — bystanders are not defaults. Returns None if ok."""
    pool = [e for e in world.entities.values()
            if ("person" in e.tags or "item" in e.tags) and "dead" not in e.tags
            and "taken" not in e.tags and "player" not in e.tags]
    named = _lex_match(utterance, pool, lambda e: e.name)
    if named and not any(e.pos in valid_cells for e in named):
        e = named[0]
        return ("refuse", f"{e.name.capitalize()} is too far to {verb_phrase or what} from here.")
    if not named:
        people_in = [e for e in pool if "person" in e.tags and e.pos in valid_cells]
        if len(people_in) > 1 or (people_in and len(valid_cells) > 1):
            names = ", ".join(sorted(e.name for e in people_in)) or "several things"
            return ("clarify", f"At what? Within {what} range: {names}.")
    return None


# why a recognized verb can't be done — the refusal is a clue, not a wall
CANT = {"throw": "You carry nothing you could throw.",
        "give": "You carry nothing you could give.",
        "pour": "You carry no liquids to pour.",
        "ignite": "There is nothing here you could set alight.",
        "smother": "There is no fire within reach to put out.",
        "smash": "Nothing within reach would break.",
        "take": "There is nothing here to pick up.",
        "attack": "Nobody is within reach to strike.",
        "shove": "Nobody is within reach to shove.",
        # recognized-and-declined: without this, "I DRINK the water" fell through
        # to "a named liquid implies pouring" and dumped the waterskin (playtest)
        "drink": "You take a swallow from your waterskin. It changes nothing the world tracks."}


_NOUNS = {"oil", "water", "acid", "fire", "beam", "here", "north", "south", "east",
          "west", "vial", "torch", "floor"}


def _words(clause):
    out = set()
    for w in clause.lower().replace(",", " ").replace(".", " ").replace("!", " ").replace("?", " ").split():
        w = SYN.get(w, w)
        # typo tolerance ("oyl"→oil) — but never remap real words (there≠here),
        # and only within the same first letter ("REST" is not west — playtest)
        if w not in _NOUNS and w not in _DEIXIS and w not in VERB_LEX and len(w) >= 3:
            w = next((n for n in _NOUNS if _ed1(w, n) and w[0] == n[0]), w)
        out.add(w)
    return out


_MATCH_STOP = {"the", "a", "an", "of", "at", "to", "my", "his", "her", "their", "and", "on", "in"}


def _lex_match(clause, cands, namer):
    """cands whose name-words appear verbatim in the clause. One hit = settled.
    Stopwords never match — 'Sly THE cutpurse' must not hit on 'the'."""
    w = _words(clause) - _MATCH_STOP
    return [c for c in cands if (_words(namer(c)) - _MATCH_STOP) & w]


def _cell_names(world, c):
    names = [c[1], _main(world, c[0])]
    names += [e.name for e in world.entities_at(c[0])
              if ("person" in e.tags or "item" in e.tags) and "dead" not in e.tags]
    return " ".join(names)


def _pick_cell(world, picker, scene, utterance, question, cands, virtual=()):
    w = _words(utterance)
    dir_hits = [c for c in cands if c[1] in w]     # "here"/"south" beat content words
    if len(dir_hits) == 1:
        return dir_hits[0][0]
    hits = dir_hits or _lex_match(utterance, cands, lambda c: _cell_names(world, c))
    if len(hits) == 1:
        return hits[0][0]
    if not hits and _words(utterance) & _DEIXIS and virtual:
        for p in reversed(list(virtual)):    # "light IT" → most recent cell acted on
            if any(c[0] == p for c in cands):
                return p
    pool = hits or cands
    if len(pool) == 1:
        return pool[0][0]
    opts = [(f"{name} {_main(world, p)}", f"{p}: {desc}") for p, name, desc in pool]
    return pool[picker.pick(scene, utterance, question, opts)][0]


# deterministic clause structure: " then "/" and "/commas split a request into
# ordered steps IF the next fragment starts like an action — sequence is parsed,
# semantics stay with the model (dependency order, cheap where cheap works)
_ACTIONISH = {"pour", "splash", "dump", "spill", "light", "lite", "set", "ignite", "burn",
              "torch", "douse", "smother", "extinguish", "beat", "put", "throw", "toss",
              "hurl", "grab", "take", "snatch", "pick", "walk", "go", "run", "step", "move",
              "get", "back", "push", "pull", "shove", "drag", "haul", "soak", "tell", "ask",
              "cool", "more", "acid", "water", "oil"}


def _clauses(u: str) -> list:
    import re
    parts, cur = [], ""
    for frag in re.split(r"\s+then\s+|\s+and\s+|,", u, flags=re.I):
        frag = frag.strip()
        if not frag:
            continue
        first = frag.split()[0].lower().strip("!.")
        if not cur or first in _ACTIONISH:
            if cur:
                parts.append(cur)
            cur = frag
        else:                       # "me and Pip" — not a new action; rejoin
            cur += " and " + frag
    if cur:
        parts.append(cur)
    return parts


REFUSE_PMI = 0.15   # if no option is even weakly indicated, the request is out of scope


def _compose_step(world, player, picker, utterance, done, virtual, first=False):
    """One dependency-ordered step. `virtual` = cells already altered this turn
    (so 'then light it' can target the oil poured a step ago). On the FIRST step
    the routing exits (talk/look) sit alongside the verbs — and if NO option is
    even weakly indicated (max PMI below threshold), the request is refused."""
    scene = _scene(world, player, done)
    verbs = _verbs_available(world, player, done, virtual)
    lex, recognized = _lex_verbs(utterance, verbs, player, world)
    if not lex and recognized:      # verb understood, means missing — say WHY
        return ("refuse", CANT.get(recognized[0], "That can't be done right now.")), None
    if len(lex) == 1:
        verb = lex[0]                                   # explicit verb word — settled
    else:
        pool = lex or verbs[:]
        if not lex:
            w = _words(utterance)
            if w & {"it", "that"} and not _lex_match(utterance, _people_in_reach(world, player),
                                                     lambda e: e.name):
                pool = [v for v in pool if v != "shove"]   # "it" is a thing, not a person
            elif _people_in_reach(world, player):
                pool.append("talk")
            if first:
                pool.append("look")
        opts = [VERB_OPTS.get(v) or ROUTE_OPTS[v] for v in pool]
        q = "What is the player's first action?" if first else "What should be done next?"
        vi, scores = picker.pick_scored(scene, utterance, q, opts)
        verb = pool[vi]
        from . import trace
        trace.capture("verb", pool, scores, verb)      # harvest the verb pick if the turn is blessed
        if first and not lex and max(scores) < REFUSE_PMI:
            w = _words(utterance)
            if w & set(FLUIDS) and "pour" in verbs:     # a named liquid implies pouring
                verb = "pour"
            elif w & set(DIRNAME.values()):
                verb = "walk"
            else:
                return ("refuse",), None

    if verb == "stop":
        return None, None
    if verb == "look":
        return ("route_inspect",), None
    if verb == "talk":
        people = _people_in_reach(world, player)
        if not people:                                   # nobody to talk to — a coherent refusal, not a crash
            return ("refuse", "There's nobody close enough to talk to."), None
        named = _lex_match(utterance, people, lambda e: f"{e.name} {e.id}")   # "the guard"/"Bran"/"Sly"
        if len(named) == 1:
            who = named[0]                               # an explicit name/role settles it — don't ask the LM
        elif len(people) == 1:
            who = people[0]
        else:
            pool = named or people
            who = pool[picker.pick(scene, utterance, "Speaking to whom?",
                                   [(e.name.split()[0], e.name) for e in pool])]
        return ("social", who.id), None
    if verb == "pour":
        mats = _carried_fluids(player)
        named = [m for m in mats if m in _words(utterance)]
        if len(named) == 1:
            mat = named[0]
        elif len(mats) > 1:
            mat = mats[picker.pick(scene, utterance, "Pour which liquid?", [(m, None) for m in mats])]
        else:
            mat = mats[0]
        # strip the material word so "pour OIL at my feet" doesn't target the oil pool
        tgt_utt = " ".join(w for w in utterance.split() if SYN.get(w.lower(), w.lower()) != mat)
        tgt = _pick_cell(world, picker, scene, tgt_utt or utterance, f"Pour the {mat} where?",
                         _reach(world, player), virtual)
        return ("pour", tgt, mat), f"poured {mat} at {tgt}"
    if verb == "ignite":
        cands = [(p, n, d) for p, n, d in _reach(world, player) if _flammable(world, p) or p in virtual]
        if not cands:   # auto-goto: nearest flammable anywhere, walk adjacent first
            flams = sorted(p for p in _notable_in_range(world, player, 99) if _flammable(world, p))
            if not flams:
                return ("refuse",), None
            goal = min(flams, key=lambda p: _l1(p, player.pos))
            path = _bfs_adjacent(world, player.pos, goal)
            if path is None:
                return ("refuse",), None
            return ("goto_path", tuple(path), goal), f"walked to {path[-1] if path else player.pos} and lit {goal}"
        tgt = _pick_cell(world, picker, scene, utterance, "Set fire to what?", cands, virtual)
        return ("ignite", tgt), f"lit the {tgt} cell"
    if verb == "smother":
        cands = [(p, n, d) for p, n, d in _reach(world, player)
                 if "on_fire" in world.cell(p).tags or p in virtual]
        if not cands:
            return ("refuse",), None
        tgt = _pick_cell(world, picker, scene, utterance, "Smother which fire?", cands, virtual)
        return ("smother", tgt), f"smothered the fire at {tgt}"
    if verb == "shove":
        people = _people_in_reach(world, player)
        named = _lex_match(utterance, people, lambda e: e.name)
        who = named[0] if len(named) == 1 else (
            people[picker.pick(scene, utterance, "Shove or pull whom?",
                               [(e.name.split()[0], e.name) for e in people])] if len(people) > 1 else people[0])
        notable = sorted({p for p in world.neighbors(who.pos) + [who.pos]
                          if world.cell(p).fluids or "on_fire" in world.cell(p).tags})
        w = _words(utterance)
        away_words, into_words = w & {"away", "clear", "off", "out", "from"}, w & {"into", "onto", "in"}
        opts, dests = [], []
        for p in notable:
            hit = _words(_main(world, p)) & w
            if p != who.pos and not is_solid(world.cell(p).material) and not away_words:
                if hit and into_words:
                    return ("shove", who.id, p), f"moved {who.name} to {p}"
                opts.append((f"into the {_main(world, p)}", f"push them into {p}")); dests.append(p)
            if not into_words:
                d = _away_dest(world, who.id, p)
                if hit and away_words:
                    return ("shove", who.id, d), f"moved {who.name} to {d}"
                opts.append((f"away from the {_main(world, p)}", f"pull them clear of {p}")); dests.append(d)
        if not opts:
            opts, dests = [("back", "push them one step back")], [_away_dest(world, who.id, player.pos)]
        di = picker.pick(scene, utterance, f"Move {who.name} where?", opts) if len(opts) > 1 else 0
        return ("shove", who.id, dests[di]), f"moved {who.name} to {dests[di]}"
    if verb == "take":
        items = _items_in_reach(world, player)
        named = _lex_match(utterance, items, lambda e: e.name)
        it = named[0] if len(named) == 1 else (
            items[picker.pick(scene, utterance, "Pick up what?",
                              [(e.name.split()[-1], e.name) for e in items])] if len(items) > 1 else items[0])
        return ("take", it.id), f"took the {it.name}"
    if verb == "throw":
        items = _carried_items(world, player)
        named = _lex_match(utterance, items, lambda e: e.name)
        it = named[0] if len(named) == 1 else (
            items[picker.pick(scene, utterance, "Throw what?",
                              [(e.name.split()[-1], e.name) for e in items])] if len(items) > 1 else items[0])
        cells = sorted(set(_notable_in_range(world, player, THROW_RANGE)) - {player.pos})
        # the thrown item's own name mustn't count as "naming a target"
        tgt_utt = " ".join(w for w in utterance.split() if w.lower() not in _words(it.name))
        guard = _named_target_check(world, player, tgt_utt, set(cells), world.rng,
                                    "throwing", verb_phrase="hit with a throw")
        if guard:
            return guard, None
        cands = [(p, "the", _cell_bits(world, p)) for p in cells]
        tgt = _pick_cell(world, picker, scene, tgt_utt, f"Throw the {it.name} at what?", cands, virtual)
        return ("throw", it.id, tgt), f"threw the {it.name} to {tgt}"
    if verb == "attack":
        people = _people_in_reach(world, player)
        guard = _named_target_check(world, player, utterance,
                                    {e.pos for e in people}, world.rng, "strike")
        if guard:
            return guard, None
        named = _lex_match(utterance, people, lambda e: e.name)
        who = named[0] if len(named) == 1 else (
            people[picker.pick(scene, utterance, "Attack whom?",
                               [(e.name.split()[0], e.name) for e in people])] if len(people) > 1 else people[0])
        return ("attack", who.id), f"attacked {who.name}"
    if verb == "give":
        items = _carried_items(world, player)
        named_it = _lex_match(utterance, items, lambda e: e.name)
        it = named_it[0] if len(named_it) == 1 else (
            items[picker.pick(scene, utterance, "Give what?",
                              [(e.name.split()[-1], e.name) for e in items])] if len(items) > 1 else items[0])
        people = _people_in_reach(world, player)
        named_p = _lex_match(utterance, people, lambda e: e.name)
        who = named_p[0] if len(named_p) == 1 else (
            people[picker.pick(scene, utterance, "Give it to whom?",
                               [(e.name.split()[0], e.name) for e in people])] if len(people) > 1 else people[0])
        return ("give", it.id, who.id), f"gave the {it.name} to {who.name}"
    if verb == "smash":
        things = _breakables_in_reach(world, player)
        named = _lex_match(utterance, things, lambda e: e.name)
        it = named[0] if len(named) == 1 else (
            things[picker.pick(scene, utterance, "Smash what?",
                               [(e.name.split()[0], e.name) for e in things])] if len(things) > 1 else things[0])
        return ("smash", it.id), f"smashed the {it.name}"
    if verb == "walk":
        cands = [(p, n, d) for p, n, d in _reach(world, player)
                 if p != player.pos and not is_solid(world.cell(p).material)]
        tgt = _pick_cell(world, picker, scene, utterance, "Walk which way?", cands, virtual)
        return ("move", tgt), f"walked to {tgt}"
    return ("refuse",), None


def _spot(player, pos):
    """A cell as fiction, never a tuple (playtest: 'you pour water at (2, 1, 0)')."""
    if tuple(pos) == tuple(player.pos):
        return "at your feet"
    s = lambda a, b: (a > b) - (a < b)
    n = DIRNAME.get((s(pos[0], player.pos[0]), s(pos[1], player.pos[1])))
    return f"to the {n}" if n else "nearby"


def _locate(world, player, sought):
    """Answer 'where is / look for X' from the WHOLE world, not just arm's reach
    (playtest: a canned miss three tiles from clearly-visible stairs). Names the
    nearest match and a rough bearing — never exact coordinates."""
    px, py, pz = player.pos
    best = None                 # (-specificity, distance, label, pos): "gold box"
    for w in sought:            # must find the GOLD box, not the nearer Bronze one
        for p, c in world.cells.items():
            if w in c.tags or w == c.material or c.fluids.get(w, 0.0) > 0:
                d = abs(p[0] - px) + abs(p[1] - py) + 4 * abs(p[2] - pz)
                best = min(best or (-1, d, w, p), (-1, d, w, p))
    for e in world.entities.values():
        if "player" in e.tags or "taken" in e.tags:
            continue
        score = sum(1 for w in sought if w in e.name.lower().split())
        if score:
            d = abs(e.pos[0] - px) + abs(e.pos[1] - py) + 4 * abs(e.pos[2] - pz)
            best = min(best or (-score, d, e.name, e.pos), (-score, d, e.name, e.pos))
    if best is None:
        return None
    _, d, label, (tx, ty, tz) = best
    if tz < pz:
        where = "somewhere below — you'd need a way down"
    elif tz > pz:
        where = "somewhere overhead"
    elif (tx, ty) == (px, py):
        where = "right where you stand"
    elif d <= 1:
        where = "right beside you"
    else:
        ns = "north" if ty < py else "south" if ty > py else ""
        ew = "west" if tx < px else "east" if tx > px else ""
        where = "off to the " + (ns + ew if ns and ew else ns or ew)
    return f"You scan the room: the {label}, {where}."


def _route_inspect(world, player, picker, scene, u):
    things = [(p, f"{n} {_main(world, p)}", d) for p, n, d in _reach(world, player)]
    things += [(e.pos, e.name.split()[-1], e.name) for e in
               _items_in_reach(world, player) + _carried_items(world, player)]
    hits = _lex_match(u, things, lambda t: f"{t[1]} {t[2]}")
    named = _lex_match(u, [e for e in world.entities.values()
                           if "player" not in e.tags and "taken" not in e.tags],
                       lambda e: e.name)
    is_where = u.lower().split()[:1] == ["where"]      # "where is X" wants a BEARING
    if not hits and named and not is_where:  # "examine Pip" answers about PIP, not the floor
        return [("inspect", named[0].id)]
    # a question about nothing the world models → the ORACLE rules, once, as canon
    if not hits and u.split() and \
            u.lower().split()[0] in ("is", "are", "does", "do", "was", "were", "can", "could"):
        return [("oracle", u)]
    STOP_LOOK = {"look", "looking", "see", "examine", "inspect", "search", "find", "check",
                 "where", "what", "whats", "is", "are", "around", "about", "here", "room",
                 "area", "scene", "surroundings", "nearby", "for", "at", "in", "on", "the",
                 "a", "an", "my", "me", "i", "of", "to", "any", "some", "there", "out",
                 "over", "up", "good", "best", "this", "that"}
    _pick = lambda s: [w for w in re.findall(r"[a-z']+", s) if w not in STOP_LOOK]  # stop at purpose clause
    sought = _pick(re.split(r"\b(?:to|that|which|so|because)\b", u.lower())[0]) or _pick(u.lower())
    if (not hits or is_where) and sought:
        loc = _locate(world, player, sought)
        if loc:                 # the MAP knows where the stairs are — say so
            return [("answer", loc)]
        return [("refuse", f"You cast about for {sought[-1]}, but see no sign of it here.")]
    pool = hits if len(hits) == 1 else (hits or things)
    if len(pool) > 1:
        i = picker.pick(scene, u, "Look at what?", [(n, d) for _, n, d in pool])
    else:
        i = 0
    p, n, _ = pool[i]
    ent = next((e for e in world.entities_at(p) if "item" in e.tags and n in e.name), None)
    return [("inspect", ent.id if ent else p)]


def compose(world: World, player, utterance: str, picker=None) -> list:
    """utterance → canonical step tuples. Deterministic guards first, then routing,
    then up to MAX_STEPS dependency-ordered slot-filled steps."""
    picker = picker or MockPicker()
    u = utterance.strip()

    # conditionals are a real mechanic we don't support yet — refuse loudly, never half-execute
    if u.lower().split()[:1] in (["if"], ["when"], ["unless"], ["once"]):
        return [("refuse",)]

    # deterministic sequence structure: one composed step per clause, each clause
    # seeing what earlier clauses did (the "pour oil east THEN LIGHT IT" thread)
    steps, done, virtual = [], [], []      # virtual: ordered cells acted on (deixis recency)
    scene = _scene(world, player, [])
    clauses = _clauses(u)[:MAX_STEPS]

    # interrogatives are questions — route to inspect deterministically
    first_word = u.lower().split()[0] if u.split() else ""
    if first_word in ("how", "what", "where", "who", "why", "when", "is", "are", "does", "do"):
        return _route_inspect(world, player, picker, scene, u)

    # feasibility gate — ONLY for requests with no grounding at all: no verb word,
    # no named liquid/direction, no deixis (deixis points at something real).
    # Refusal must be an explicit binary judgment, not an emergent low score:
    # "summon a dragon" scores light+1.5 by dragon→fire association.
    grounded = any(_lex_verbs(c, _verbs_available(world, player, [], ["x"]), player)[1] or
                   (_words(c) & (set(FLUIDS) | set(DIRNAME.values()) | _DEIXIS))
                   for c in clauses)
    if clauses and not grounded:
        gate = [("possible", "an ordinary act — pouring, lighting, moving, throwing, talking — can do this here"),
                ("impossible", "this needs magic, superpowers or breaking reality — nobody here can")]
        if picker.pick(scene, u, "Can this request actually be carried out here?", gate) == 1:
            return [("refuse",)]

    for ci, clause in enumerate(clauses):
        step, label = _compose_step(world, player, picker, clause, done, virtual, first=(ci == 0))
        if step is None:
            break
        if step[0] == "social":
            return [step] if not steps else steps + [step]
        if step[0] == "route_inspect":
            return _route_inspect(world, player, picker, scene, u)
        if step[0] in ("refuse", "clarify"):
            if not steps:
                return [step]           # carries the reason/question — a clue, not a wall
            break
        if steps and step[0] == "shove" and steps[-1][0] == "shove" and step[1] == steps[-1][1]:
            steps[-1] = step; done[-1] = label          # "grab Pip and pull him away" = one drag
            continue
        steps.append(step)
        done.append(label)
        if step[0] in ("pour", "ignite", "smother", "move"):
            virtual.append(step[1])
        elif step[0] == "smash":                       # "break the barrel THEN LIGHT IT"
            virtual.append(world.entities[step[1]].pos)
    return steps or [("refuse",)]


# ── compile canonical steps → bus deltas ──
def _skill(e, name):
    return e.props.get(name, 0.5)


def to_effects(world: World, player, steps: list):
    """Steps → deltas, WITH the dice: physical contests roll here, at execution
    time, on the seeded world rng — skill sets the odds, the roll decides."""
    from . import effects as fx
    out, cost = [], 0
    for s in steps:
        k = s[0]
        if k == "stop_drop_roll":
            from . import checks
            c = world.cell(player.pos)
            if c.fluids.get("water", 0) > 0:            # water underfoot: a sure thing
                band = "clean"
            elif "on_fire" in c.tags:                   # standing in fire: nothing to smother
                band = "miss"
            else:
                band = checks.roll_band(world, 0.45 + 0.3 * _skill(player, "might"))
            if band in ("crit", "clean"):               # only a solid roll smothers it
                out.append(fx.clear_tag(player.id, "on_fire", "you drop and roll — the flames are smothered!"))
            else:
                out.append(fx.adjust_prop(player.id, "hp", 0.0, "you thrash on the ground but the flames cling!"))
            cost += 1
        elif k == "pour":
            out.append(fx.add_fluid(s[1], s[2], 400.0, f"you pour {s[2]} {_spot(player, s[1])}")); cost += 1
            if s[2] == "water" and s[1] == player.pos and "on_fire" in player.tags:
                out.append(fx.clear_tag(player.id, "on_fire", "the water douses the flames on you!"))
        elif k == "ignite":
            out.append(fx.spark(s[1], f"you set a spark {_spot(player, s[1])}")); cost += 1
        elif k == "smother":
            out.append(fx.clear_tag(s[1], "on_fire", f"you smother the fire {_spot(player, s[1])}")); cost += 1
        elif k == "shove":
            out.append(fx.move(s[1], s[2], f"you haul {world.entities[s[1]].name} to {s[2]}")); cost += 1
        elif k == "smash":
            e = world.entities[s[1]]
            dmg = combat.attack_damage(world, player, e.material, "clean", dtype="crush")   # a smash is a crush blow
            out.append(fx.adjust_prop(s[1], "hp", -dmg, f"you smash the {e.name}")); cost += 1
        elif k == "attack":
            from . import checks
            e = world.entities[s[1]]
            base = 0.35 + 0.55 * _skill(player, "might")
            p_hit = combat.to_hit(world, player, e, base)     # reads BOTH sides' real condition
            if "off_balance" in player.tags:            # a stumble of your OWN comes due — and is spent
                p_hit = max(0.05, p_hit - 0.2)
                out.append(fx.clear_tag(player.id, "off_balance"))
            band = checks.roll_band(world, p_hit)
            dmg = combat.attack_damage(world, player, e.material, band)   # reads weapon type × target material
            if band == "crit":
                out.append(fx.adjust_prop(s[1], "hp", -dmg,
                                          f"your blow lands expertly — clean through {e.name}'s guard!"))
            elif band == "clean":
                out.append(fx.adjust_prop(s[1], "hp", -dmg, f"you strike {e.name}!"))
            elif band == "partial":
                out.append(fx.adjust_prop(s[1], "hp", -dmg,
                                          f"a scrappy, glancing hit — {e.name} is struck, but you're exposed"))
                out += checks.consequence(world, player, "attack", e.pos, victim_id=s[1])
            else:
                out.append(fx.set_edge(s[1], player.id, "hostility", 1.0,
                                       f"{e.name} ducks — your blow goes wide!"))
                out += checks.consequence(world, player, "attack", e.pos, victim_id=s[1])
            cost += 1
        elif k == "give":
            it, who = world.entities[s[1]], world.entities[s[2]]
            cur = world.edges.get((s[2], "player"), {}).get("disposition", 0.0)
            out += [fx.clear_tag(player.id, f"inv:{s[1]}"), fx.set_tag(s[2], f"inv:{s[1]}"),
                    fx.move(s[1], who.pos, f"you hand the {it.name} to {who.name}"),
                    fx.set_edge(s[2], "player", "disposition", cur + 0.4)]; cost += 1
        elif k == "take":
            e = world.entities[s[1]]
            out += [fx.set_tag(s[1], "taken", f"you take the {e.name}"),
                    fx.set_tag(player.id, f"inv:{s[1]}")]; cost += 1
        elif k == "throw":
            from . import checks
            e = world.entities[s[1]]
            aimed = next((v for v in world.entities_at(s[2])          # who you threw AT
                          if "person" in v.tags and "dead" not in v.tags), None)
            p_hit = combat.throw_hit(world, player, aimed, _l1(player.pos, s[2]))
            band = checks.roll_band(world, p_hit)
            hit, land = band != "miss", s[2]
            if not hit:                 # a wild throw scatters beside the mark
                near = [n for n in world.neighbors(s[2]) if not is_solid(world.cell(n).material)]
                land = world.rng.choice(near) if near else s[2]
            out += [fx.move(s[1], land, f"you hurl the {e.name} at {s[2]}" +
                            ("" if hit else " — it goes wide!")),
                    fx.clear_tag(s[1], "taken"), fx.clear_tag(player.id, f"inv:{s[1]}")]; cost += 1
            victim = next((v for v in world.entities_at(land)
                           if "person" in v.tags and "dead" not in v.tags), None)
            if hit and victim:          # a thrown thing is a ranged attack, not a delivery
                dmg = combat.thrown_damage(e, victim.material, band == "crit")   # reads item type × target material
                out.append(fx.adjust_prop(victim.id, "hp", -dmg,
                                          f"the {e.name} strikes {victim.name}!"))
            if band in ("partial", "miss"):
                out += checks.consequence(world, player, "throw", land,
                                          victim_id=victim.id if victim else None)
        elif k == "move":
            out.append(fx.move(player.id, s[1], f"you move to {s[1]}")); cost += 1
        elif k == "goto_path":
            for p in s[1]:
                out.append(fx.move(player.id, p, f"you move to {p}")); cost += 1
            out.append(fx.spark(s[2], f"you set a spark to {s[2]}")); cost += 1
    for ef in out:
        ef.actor = player.id
    return out, max(1, cost)
