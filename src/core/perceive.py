"""perceive() — an entity's CURRENT belief-slice, the model's input and the basis of the
perception-filtered affordance menu (design doc D8).

`observe()` folds what an entity can SEE now into a belief store (`e.mind["belief"]`),
refreshing the last-known snapshot of every visible thing; unseen beliefs persist and go
STALE (a remembered node carries its last-seen snapshot + an age, and may be wrong).
`perceive()` renders that belief store as an ID-ADDRESSED PROXIMITY graph — nodes once in
distance order, relations as typed id-triples, notable cells with spatial fields — so a
model's output pointers become a bounded index. This is the single source of "what an entity
knows"; the affordance menu is built from the slice alone, so it structurally cannot offer an
action on something unperceived (partial observability by construction).

Ported into the repo from the data-engine prototype; it already targeted repo World/Cell/
Entity objects, so only the import lines changed.
"""
from . import spatial
from .mind import SIGHT

_L1 = spatial.l1


def _num(v):
    return int(v) if isinstance(v, float) and v.is_integer() else round(v, 1) if isinstance(v, float) else v


def _props(p):
    return " ".join(f"{k}={_num(v)}" for k, v in sorted(p.items()) if not k.startswith("holds:")) or "·"


def _tags(t):
    return ",".join(sorted(x for x in t if not x.startswith("inv:"))) or "·"


def _inv(t):
    return sorted(x[4:] for x in t if x.startswith("inv:"))


def _snapshot(x, tick):
    lv = x.parts.get("liquid_volume") or {}            # the holds surface reads the PART now
    return {"ref": x.id, "name": x.name, "pos": x.pos, "tick": tick,
            "props": dict(x.props),
            "holds": {lv["mat"]: lv["ml"]} if lv.get("ml", 0) > 0 else {},
            "tags": [t for t in x.tags if not t.startswith("inv:")], "inv": _inv(x.tags)}


def observe(world, e, tick, radius=SIGHT):
    """Fold what e can see NOW into its belief store — refreshing the last-known snapshot of
    every currently-visible thing. Unseen beliefs are left untouched (they persist, possibly
    going stale). This IS the fold of witnessed events."""
    b = e.mind.setdefault("belief", {})
    for x in world.entities.values():
        if x.id == e.id or spatial.visible(world, e.pos, x.pos, radius):
            b[x.id] = _snapshot(x, tick)
    b[e.id] = _snapshot(e, tick)
    return b


def perceive(world, e, now=0, radius=SIGHT):
    """The entity's BELIEF-STATE → an id-addressed node/edge/cell graph. Currently-visible
    nodes are fresh; remembered-but-unseen nodes carry their last-known snapshot + an age
    (they may be WRONG). Falls back to a live observe() if the entity has no belief store yet."""
    b = e.mind.get("belief")
    if not b:
        b = observe(world, e, now, radius)
    snaps = sorted(b.values(), key=lambda s: (s["ref"] != e.id, _L1(e.pos, s["pos"]), s["ref"]))
    idx = {s["ref"]: i for i, s in enumerate(snaps)}                    # stable local ids over the belief-state
    nodes = [{"id": i, "ref": s["ref"], "name": s["name"], "pos": s["pos"], "d": _L1(e.pos, s["pos"]),
              "self": s["ref"] == e.id, "age": now - s["tick"], "fresh": (now - s["tick"]) == 0,
              "props": s["props"], "tags": s["tags"], "inv": s["inv"], "holds": s["holds"]}
             for i, s in enumerate(snaps)]
    fresh = {n["ref"] for n in nodes if n["fresh"]}
    edges = [(idx[a], kind, idx[b], round(val, 2))                      # relations between co-visible fresh nodes
             for (a, b), kinds in world.edges.items() if a in fresh and b in fresh
             for kind, val in kinds.items()]
    cells = []                                                         # notable cells currently in sight (fresh)
    ex, ey, ez = e.pos
    for x in range(max(0, ex - radius), min(world.dims[0], ex + radius + 1)):
        for y in range(max(0, ey - radius), min(world.dims[1], ey + radius + 1)):
            p = (x, y, ez)
            if _L1(e.pos, p) > radius or not spatial.line_of_sight(world, e.pos, p):
                continue
            c = world.cell(p)
            if c.heat > 40 or c.fluids or (c.tags - {"floor"}):
                cells.append({"pos": p, "d": _L1(e.pos, p), "heat": round(c.heat, 1),
                              "fluids": dict(c.fluids), "tags": sorted(c.tags), "material": c.material})
    cells.sort(key=lambda c: (c["d"], c["pos"]))
    return {"viewer": e.id, "radius": radius, "nodes": nodes, "edges": edges, "cells": cells}


def render(slice_) -> str:
    """A human-readable panel + the compact id-addressed token form the model consumes."""
    v = slice_
    who = next(n["name"] for n in v["nodes"] if n["self"])
    L = [f"### PERCEIVED by {who}  (sight r={v['radius']})",
         f"_{len(v['nodes'])} entities, {len(v['edges'])} relations, {len(v['cells'])} cells visible_", ""]
    for n in v["nodes"]:
        tag = " ·SELF·" if n["self"] else f" fresh d{n['d']}" if n["fresh"] else f" STALE d{n['d']} ~{n['age']}t"
        inv = f"  carrying[{','.join(n['inv'])}]" if n["inv"] else ""
        holds = f"  holds[{' '.join(f'{k}={_num(x)}' for k, x in n['holds'].items())}]" if n["holds"] else ""
        L.append(f"  #{n['id']}{tag}  {n['name']} @{n['pos']}   {_props(n['props'])}   «{_tags(n['tags'])}»{inv}{holds}")
    for a, kind, b, val in v["edges"]:
        L.append(f"  edge #{a} —{kind}→ #{b} = {val}")
    for c in v["cells"]:
        f = " ".join(f"{k}={_num(x)}" for k, x in c["fluids"].items())
        bits = " ".join(x for x in [f"heat={_num(c['heat'])}" if c["heat"] > 40 else "", f, c["material"],
                                    ",".join(t for t in c["tags"] if t != "floor")] if x)
        L.append(f"  cell @{c['pos']} d{c['d']}   {bits}")
    L += ["", "```", "# token stream (id-addressed):"]
    for n in v["nodes"]:
        s = "self" if n["self"] else (f"seen d{n['d']}" if n["fresh"] else f"stale{n['age']} d{n['d']}")
        extra = (" inv:" + ",".join(n["inv"]) if n["inv"] else "") + \
                ("".join(f" holds:{k}={_num(x)}" for k, x in n["holds"].items()))
        L.append(f"(ent {n['id']} {s} {n['ref']} @{n['pos'][0]},{n['pos'][1]} | {_props(n['props'])} | {_tags(n['tags'])}{extra})")
    for a, kind, b, val in v["edges"]:
        L.append(f"(edge {a} {kind} {b} = {val})")
    for c in v["cells"]:
        fl = "".join(f" {k}={_num(x)}" for k, x in c["fluids"].items())
        L.append(f"(cell @{c['pos'][0]},{c['pos'][1]} d{c['d']} heat={_num(c['heat'])}{fl} | {','.join(t for t in c['tags'] if t != 'floor') or '·'})")
    L.append("```")
    return "\n".join(L)
