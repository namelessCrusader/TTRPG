"""The DSL: a TOTAL, loop-free, recursive language a model emits, over the effect
VM. A program is a JSON AST; the sim is the authoritative interpreter.

  program := { "target": <targetexpr>, "guard": <guard>|null,
               "roll": {"contest":[<arg>,<arg>]}|null,
               "do": [<effect>...],                          # roll==null (deterministic)
               "clean": [<effect>...], "fail": [<effect>...] } # roll present (stochastic)
  targetexpr := {"entity":"<id|name substr>"} | {"cell":"here|N|S|E|W"} | {"self":true}
  guard := {"has_tag":[<ref>,"tag"]} | {"cmp":[<arg>,"<|<=|>|>=|==|!=",<arg>]}
         | {"and":[g,...]} | {"or":[g,...]} | {"not": g} | null
  arg   := number | {"prop":[<ref>,"name"]} | {"edge":[<ref>,<ref>,"kind"]} | {"add":[a,b]}
         | {"sub":[a,b]} | {"mul":[a,b]} | {"min":[a,b]} | {"max":[a,b]} | {"if":[<guard>,a,b]}
  ref   := "target" | "self" | "<id|name substr>" | {"cell_of": <ref>} | {"cell":"here|N|S|E|W"}   # a cell is a ref too
  effect ops: set_tag clear_tag adjust_prop set_prop adjust_edge set_edge add_heat spark
              add_fluid remove_fluid transfer_prop move spawn despawn speech

NO loops: a program is a finite tree, always terminates, statically analyzable —
the sim reads it and knows the exact deterministic-or-(seeded)-stochastic edit.
Iteration is the CA's job (reactions.tick), not the program's. The model emits a
LOCAL edit; the sim propagates globally. Verbs/reactions/stances are all special
cases of a program — this is the general (state,intent,target)→edit operation.
"""
from __future__ import annotations

import copy

from . import effects as fx
from .state import World, is_solid

DIRS = {"here": (0, 0, 0), "N": (0, -1, 0), "S": (0, 1, 0), "E": (1, 0, 0), "W": (-1, 0, 0),
        "U": (0, 0, 1), "D": (0, 0, -1)}
CMP = {"<": lambda x, y: x < y, "<=": lambda x, y: x <= y, ">": lambda x, y: x > y,
       ">=": lambda x, y: x >= y, "==": lambda x, y: x == y, "!=": lambda x, y: x != y}


class DSLError(Exception):
    """A malformed or unresolvable program — the sim refuses it (like Incoherence)."""


def _entity(world, actor, key):
    """id-exact first, else name-substring, preferring same storey & nearest."""
    if key in world.entities:
        return key
    k = str(key).lower()
    cands = [e for e in world.entities.values() if k in e.name.lower()]
    if not cands:
        raise DSLError(f"no entity matching {key!r}")
    ax, ay, az = actor.pos
    cands.sort(key=lambda e: (e.pos[2] != az, abs(e.pos[0] - ax) + abs(e.pos[1] - ay)))
    return cands[0].id


def _cellpos(world, actor, spec):
    d = DIRS.get(spec)
    if d is None:
        raise DSLError(f"bad cell {spec!r}")
    p = (actor.pos[0] + d[0], actor.pos[1] + d[1], actor.pos[2] + d[2])
    if not world.in_bounds(p):
        raise DSLError(f"cell {spec} out of bounds")
    return p


def _bind(world, actor, tspec):
    if not isinstance(tspec, dict):
        raise DSLError(f"target not an object: {tspec!r}")
    if tspec.get("self"):
        return actor.id
    if "entity" in tspec:
        return _entity(world, actor, tspec["entity"])
    if "cell" in tspec:
        return _cellpos(world, actor, tspec["cell"])
    raise DSLError(f"unknown target {tspec!r}")


def _ref(world, actor, binds, ref):
    if isinstance(ref, dict):                       # a cell is a first-class ref, not only a target
        if "cell_of" in ref:                        # {"cell_of": <ref>} → the cell that entity occupies
            loc = _ref(world, actor, binds, ref["cell_of"])
            return loc if isinstance(loc, tuple) else world.entities[loc].pos
        if "cell" in ref:                           # {"cell": "here|N|S|E|W"} → a cell by direction from the actor
            return _cellpos(world, actor, ref["cell"])
        raise DSLError(f"bad ref {ref!r}")
    if ref in binds:
        return binds[ref]
    if ref in ("self", actor.id):
        return actor.id
    return _entity(world, actor, ref)


def _props_of(world, loc):
    return world.cell(loc).props if isinstance(loc, tuple) else world.entities[loc].props


def _tags_of(world, loc):
    return world.cell(loc).tags if isinstance(loc, tuple) else world.entities[loc].tags


def _arg(world, actor, binds, a):
    if isinstance(a, bool):
        raise DSLError("bool is not an arg")
    if isinstance(a, (int, float)):
        return float(a)
    if not isinstance(a, dict):
        raise DSLError(f"bad arg {a!r}")
    if "prop" in a:
        loc = _ref(world, actor, binds, a["prop"][0])
        name = a["prop"][1]
        if isinstance(loc, tuple):                  # a cell exposes its spatial fields as readable props
            cell = world.cell(loc)
            if name == "temperature":
                return cell.heat
            if name in cell.fluids:                 # pooled oil/water is readable — "is there oil here to light?"
                return cell.fluids[name]
        return _props_of(world, loc).get(name, 0.0)
    if "edge" in a:                                 # read a RELATIONSHIP — "does she like me?"
        ea, eb, kind = a["edge"]
        pair = (_ref(world, actor, binds, ea), _ref(world, actor, binds, eb))
        return world.edges.get(pair, {}).get(kind, 0.0)
    for k, f in (("add", lambda x, y: x + y), ("sub", lambda x, y: x - y), ("mul", lambda x, y: x * y),
                 ("min", min), ("max", max)):
        if k in a:
            return f(_arg(world, actor, binds, a[k][0]), _arg(world, actor, binds, a[k][1]))
    if "if" in a:
        g, x, y = a["if"]
        return _arg(world, actor, binds, x if _guard(world, actor, binds, g) else y)
    raise DSLError(f"unknown arg {a!r}")


def _guard(world, actor, binds, g):
    if g is None:
        return True
    if "has_tag" in g:
        return g["has_tag"][1] in _tags_of(world, _ref(world, actor, binds, g["has_tag"][0]))
    if "cmp" in g:
        a, op, b = g["cmp"]
        if op not in CMP:
            raise DSLError(f"bad cmp op {op!r}")
        return CMP[op](_arg(world, actor, binds, a), _arg(world, actor, binds, b))
    if "and" in g:
        return all(_guard(world, actor, binds, x) for x in g["and"])
    if "or" in g:
        return any(_guard(world, actor, binds, x) for x in g["or"])
    if "not" in g:
        return not _guard(world, actor, binds, g["not"])
    raise DSLError(f"unknown guard {g!r}")


def _effect(world, actor, binds, e):
    if not isinstance(e, dict) or "op" not in e:
        raise DSLError(f"bad effect {e!r}")
    op = e["op"]
    A = lambda k: _arg(world, actor, binds, e[k])
    R = lambda k="t": _ref(world, actor, binds, e[k])
    c = e.get("cause", "")
    if op == "set_tag":      return [fx.set_tag(R(), e["tag"], c)]
    if op == "clear_tag":    return [fx.clear_tag(R(), e["tag"], c)]
    if op == "adjust_prop":  return [fx.adjust_prop(R(), e["prop"], A("by"), c)]
    if op == "set_prop":     return [fx.set_prop(R(), e["prop"], A("to"), c)]
    if op == "add_heat":     return [fx.add_heat(R(), A("by"), c)]
    if op == "spark":        return [fx.spark(R(), c)]
    if op == "add_fluid":    return [fx.add_fluid(R(), e["mat"], A("ml"), c)]
    if op == "remove_fluid": return [fx.remove_fluid(R(), e["mat"], A("ml"), c)]
    if op == "move":
        loc, d = R(), DIRS[e["dir"]]
        p = world.entities[loc].pos
        q = (p[0] + d[0], p[1] + d[1], p[2] + d[2])
        if not world.in_bounds(q) or is_solid(world.cell(q).material):
            raise DSLError(f"cannot move into {q} — out of bounds or a solid block")
        return [fx.move(loc, q, c)]
    if op in ("set_edge", "adjust_edge"):
        a, b = _ref(world, actor, binds, e["a"]), _ref(world, actor, binds, e["b"])
        cur = world.edges.get((a, b), {}).get(e["kind"], 0.0)
        return [fx.set_edge(a, b, e["kind"], A("to") if op == "set_edge" else cur + A("by"), c)]
    if op == "transfer_prop":                       # conserved, clamped; entity OR cell either side
        src, dst = _ref(world, actor, binds, e["from"]), _ref(world, actor, binds, e["to"])
        amt = min(_props_of(world, src).get(e["prop"], 0.0), A("amount"))
        return [fx.adjust_prop(src, e["prop"], -amt, c), fx.adjust_prop(dst, e["prop"], amt)] if amt > 1e-9 else []
    if op == "spawn":
        binds[e["id"]] = e["id"]                    # the new id enters scope: later effects in THIS program can target it
        return [fx.spawn(e["id"], e.get("name", e["id"]), _cellpos(world, actor, e.get("at", "here")),
                         e.get("tags", []), e.get("props", {}), e.get("material", "flesh"), c)]
    if op == "despawn":      return [fx.despawn(R(), c)]
    if op == "speech":                              # an entity SAYS something — the reply/voice
        who = R("who")
        name = world.entities[who].name if who in world.entities else who
        return [fx.speech(who, e["text"], c or f'{name}: "{e["text"]}"')]
    raise DSLError(f"unknown op {op!r}")


def _band(world, actor, binds, roll):
    a, b = _arg(world, actor, binds, roll["contest"][0]), _arg(world, actor, binds, roll["contest"][1])
    margin = (a - b) + (world.rng.random() - 0.5) * 20.0     # seeded → replayable stochasticity
    return "clean" if margin > 0 else "fail"


def run(world: World, actor, prog: dict) -> dict:
    """Execute a program atomically: build the whole effect list (raising before
    ANY apply if malformed), then apply through the one bus. {ok, changed, band,
    n_effects, error}."""
    from . import engine
    before = engine._fingerprint(world)
    try:
        binds = {"target": _bind(world, actor, prog["target"])}
        if not _guard(world, actor, binds, prog.get("guard")):
            return {"ok": True, "changed": False, "band": "guard-false", "n_effects": 0, "error": None}
        band = _band(world, actor, binds, prog["roll"]) if prog.get("roll") else "do"
        effs = []
        for e in prog.get(band, []):
            effs.extend(_effect(world, actor, binds, e))
        fx.apply_all(world, effs)
        return {"ok": True, "changed": engine._fingerprint(world) != before,
                "band": band, "n_effects": len(effs), "error": None}
    except (DSLError, KeyError, TypeError, IndexError, ValueError, fx.Incoherence) as ex:
        return {"ok": False, "changed": False, "band": None, "n_effects": 0,
                "error": f"{type(ex).__name__}: {ex}"}


def validate(world: World, actor, prog: dict) -> dict:
    """Dry-run on a fork — the data engine's filter. The real world is untouched."""
    return run(copy.deepcopy(world), copy.deepcopy(actor), prog)


# ── canonicalize: absorb a free-form author's dialect into the schema ────────
# The free-vs-scaffold spike showed content is equal but ENVELOPES drift; the
# data engine harvests free-form (richest) and normalizes here before the sim
# filters by execution. Handles top-level key synonyms + the {resolve:{...}} wrap.
_SYN = {"condition": "guard", "when": "guard", "if_": "guard",
        "check": "roll", "test": "roll", "contest_roll": "roll",
        "effects": "do", "actions": "do", "then": "do",
        "on_success": "clean", "success": "clean", "on_win": "clean", "win": "clean",
        "on_fail": "fail", "failure": "fail", "on_lose": "fail", "lose": "fail"}


def canonicalize(prog: dict) -> dict:
    if not isinstance(prog, dict):
        return prog
    out = {_SYN.get(k, k): v for k, v in prog.items()}
    res = out.pop("resolve", None) or out.pop("resolution", None)
    if isinstance(res, dict):                       # a nested resolve-wrapper → hoist it
        if res.get("type") in ("contest", "roll", "opposed"):
            out.setdefault("roll", {"contest": res.get("roll") or res.get("contest")})
            out["clean"] = res.get("success") or res.get("clean") or res.get("on_success", [])
            out["fail"] = res.get("failure") or res.get("fail") or res.get("on_fail", [])
        else:
            out["roll"] = None
            out["do"] = res.get("effects") or res.get("do") or res.get("actions", [])
    r = out.get("roll")
    if isinstance(r, list) and len(r) == 2:         # roll:[a,b] → roll:{contest:[a,b]}
        out["roll"] = {"contest": r}
    return out
