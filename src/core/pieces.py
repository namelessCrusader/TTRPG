"""pieces.py — the decision-2 surface: "composed tree from engine-listed pieces."

The engine's menu is a list of finished Options. This module turns that list into the
PIECE TABLE the author reads: one row per verb, each part slot with its legal values.
The author then composes one bracket line/tree from listed pieces only. `index()` is the
reverse map the parser uses: (verb + parts) → the exact Option, so a composed answer
resolves to a real, already-legal action — or to nothing, and the refusal can name the
exact bad piece. Values are ids of SEEN things, directions, item names, ordinals — the
menu already guarantees that; this module adds no legality of its own (D8: one predicate)."""
import re


def _groups(opts):
    g = {}
    for o in opts:
        if not o.verb:
            continue
        slots = g.setdefault(o.verb, {})
        for k, v in o.args.items():
            vals = slots.setdefault(k, [])
            if v not in vals:
                vals.append(v)
    return g


def table(opts, slice_=None) -> str:
    """The text the author reads. One row per verb; each slot lists its legal values.
    With a slice_, also lists the legal CONDITION pieces — ids, their tags, your carried
    items — so an (if …) test is composed from listed pieces too, never guessed."""
    lines = ["Build ONE action. Pick a verb, then one value for each of its slots:"]
    for verb, slots in _groups(opts).items():
        segs = "  ".join(f"{k}:[{'|'.join(str(v) for v in vals)}]" for k, vals in slots.items())
        note = "   (you write the words yourself)" if verb == "speak" else ""
        lines.append(f"  {verb:14s}{segs}{note}")
    if slice_ is not None:
        refs, tags, inv = _cond_pieces(slice_)
        lines += ["Condition pieces (for (if …) tests — these exact names only):",
                  f"  near ID        ID:[{'|'.join(refs)}]",
                  "  has ID TAG     " + "  ".join(f"{r}:«{','.join(tags[r]) or '·'}»" for r in refs),
                  f"  carrying ITEM  ITEM:[{'|'.join(inv) or '·'}]"]
    return "\n".join(lines)


def _cond_pieces(slice_):
    """The legal condition values, straight from what the author can SEE: entity ids,
    each id's current tags, and the items the author carries."""
    refs = [n["ref"] for n in slice_["nodes"] if not n["self"]]
    tags = {n["ref"]: n["tags"] for n in slice_["nodes"]}
    inv = next((n["inv"] for n in slice_["nodes"] if n["self"]), [])
    return refs, tags, inv


def index(opts):
    """(verb, sorted parts) → Option. The parser's reverse map: a composed line either
    names a real Option exactly, or it fails on a nameable piece."""
    return {(o.verb, tuple(sorted(o.args.items()))): o for o in opts if o.verb}


# ── the parser ───────────────────────────────────────────────────────────────
# Answers the author may write, all from listed pieces:
#   pour water at:SE                      one flat action (keys optional when clear)
#   (then (take target:brazier) (move dir:E))          a sequence
#   (if (near guard) (move dir:W) (smash target:cask_water))   condition, then, else
# The parse resolves every leaf against the CURRENT menu via index() — so a parsed plan
# is legal by construction. A bad piece fails with its NAME. Conditions are checked at
# run time against the live world (same double-buffer rule as everything else).

class PieceError(ValueError):
    """A composed answer used a piece that is not on the table. .piece names it."""
    def __init__(self, piece, why):
        self.piece, self.why = piece, why
        super().__init__(f"not a listed piece: {piece!r} — {why}")


CONDS = ("near", "has", "carrying")            # the condition heads the grammar offers


def _tokens(text):
    # quoted words are ONE token — so spoken text can live inside a tree node:
    #   (then (speak to:guard "Sly is going for the vial!") (move dir:S))
    return re.findall(r'"[^"]*"|\(|\)|[^\s()]+', text)


def _read(toks, i):
    """One s-expression (or atom) starting at i → (node, next_i). A node is a list or a str."""
    if toks[i] == "(":
        out, i = [], i + 1
        while i < len(toks) and toks[i] != ")":
            node, i = _read(toks, i)
            out.append(node)
        return out, i + 1
    return toks[i], i + 1


def _flat(words, opts):
    """[verb, token…] → the exact Option. Tokens are key:value, or bare values matched
    to the verb's slots when that is unambiguous."""
    verb, rest = words[0], words[1:]
    slots = _groups(opts).get(verb)
    if slots is None:
        raise PieceError(verb, "no such verb on the table")
    args, said = {}, None
    for tok in rest:
        if tok.startswith('"'):
            said = tok.strip('"')                       # the words, kept inside this node
        elif ":" in tok:
            k, v = tok.split(":", 1)
            if k not in slots:
                raise PieceError(k, f"'{verb}' has no slot named {k}")
            if v not in [str(x) for x in slots[k]]:
                raise PieceError(v, f"not a listed value for {verb} {k}")
            args[k] = v
        else:
            homes = [k for k, vals in slots.items() if tok in [str(x) for x in vals]]
            if not homes:
                raise PieceError(tok, f"no slot of '{verb}' lists this value")
            if len(homes) > 1:
                raise PieceError(tok, f"ambiguous — write one of: " +
                                 ", ".join(f"{k}:{tok}" for k in homes))
            args[homes[0]] = tok
    key = (verb, tuple(sorted(args.items())))
    opt = index(opts).get(key)
    if opt is None:
        need = [k for k in slots if k not in args]
        raise PieceError(verb, f"incomplete — missing slot(s): {', '.join(need)}" if need
                         else "no action matches these parts together")
    return opt, said


def _check_cond_shape(c, slice_):
    """A condition's VALUES are listed pieces too (found the hard way: a guessed tag reads
    quietly false and the wrong branch runs). With a slice_, refuse unlisted values by name."""
    if slice_ is None:
        return
    refs, tags, inv = _cond_pieces(slice_)
    if c[0] in ("near", "has") and c[1] not in tags:
        raise PieceError(c[1], "not an id you can see")
    if c[0] == "has":
        if len(c) < 3:
            raise PieceError("has", "write: (has ID TAG)")
        if c[2] not in tags[c[1]]:
            raise PieceError(c[2], f"{c[1]} does not show this tag — it shows: "
                             + (",".join(tags[c[1]]) or "none"))
    if c[0] == "carrying" and c[1] not in inv:
        raise PieceError(c[1], "you are not carrying this — you carry: " + (",".join(inv) or "nothing"))


def parse(text, opts, slice_=None):
    """A composed answer → a PLAN: a list of (condition, Option, words) steps, in order.
    condition is None (always run) or a tuple like ('near', 'guard') checked at run time;
    words is the spoken text for a speak step (or None). With a slice_, condition values
    are checked against the listed pieces too."""
    toks = _tokens(text.strip())
    if not toks:
        raise PieceError("", "empty answer")
    node, _ = _read(toks, 0) if toks[0] == "(" else (toks, len(toks))

    def walk(n, cond=None):
        if isinstance(n, str):
            raise PieceError(n, "a bare word is not an action")
        head = n[0]
        if head == "then":
            steps = []
            for sub in n[1:]:
                steps += walk(sub, cond)
            return steps
        if head == "if":
            if len(n) not in (3, 4) or not isinstance(n[1], list) or n[1][0] not in CONDS:
                raise PieceError("if", "write: (if (near ID) (action) (optional else-action))")
            c = tuple(n[1])
            _check_cond_shape(c, slice_)
            steps = walk(n[2], c)
            if len(n) == 4:
                steps += walk(n[3], ("not",) + c)
            return steps
        if head == "contest":
            raise PieceError("contest", "not supported yet — needs the dsl bridge; use if/then")
        opt, said = _flat([w if isinstance(w, str) else "(" for w in n], opts)
        return [(cond, opt, said)]

    return walk(node)


def check(world, actor, cond):
    """Is a run-time condition true NOW? Same live-world re-check as every action."""
    if cond is None:
        return True
    if cond[0] == "not":
        return not check(world, actor, tuple(cond[1:]))
    head, a = cond[0], cond[1]
    e = world.entities.get(a)
    if head == "near":
        return e is not None and max(abs(e.pos[0] - actor.pos[0]),
                                     abs(e.pos[1] - actor.pos[1])) <= 1
    if head == "has":
        return e is not None and len(cond) > 2 and cond[2] in e.tags
    if head == "carrying":
        return f"inv:{a}" in actor.tags
    return False
