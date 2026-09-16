"""The one effect algebra: fine universal deltas + one apply() + one log.

A `target` is either a cell position (a tuple) or an entity id (a str). The same
deltas — set_tag, adjust_prop, set_material — work on either, so "any property
applies to anything". Spatial deltas (heat, fluids) address a cell position.

THE INVARIANT: every mutation goes through apply(). apply() validates the
reference and appends to world.log, so the log is a complete replayable tape.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .state import Cell, Entity, World


class Incoherence(Exception):
    """apply() refused a delta referencing something that doesn't exist."""


@dataclass
class Effect:
    kind: str
    data: dict = field(default_factory=dict)
    cause: str = ""
    actor: str | None = None    # provenance: which entity caused this delta (None = the world)


# --- delta constructors (a "verb" is just a list of these) ------------------
def set_tag(t, tag, cause=""):        return Effect("set_tag", {"t": t, "tag": tag}, cause)
def clear_tag(t, tag, cause=""):      return Effect("clear_tag", {"t": t, "tag": tag}, cause)
def adjust_prop(t, prop, d, cause=""):return Effect("adjust_prop", {"t": t, "prop": prop, "delta": d}, cause)
def set_prop(t, prop, v, cause=""):   return Effect("set_prop", {"t": t, "prop": prop, "value": v}, cause)
def set_material(t, mat, cause=""):   return Effect("set_material", {"t": t, "mat": mat}, cause)
def move(eid, pos, cause=""):         return Effect("move", {"id": eid, "pos": pos}, cause)
def add_heat(pos, d, cause=""):       return Effect("add_heat", {"pos": pos, "delta": d}, cause)
def spark(pos, cause=""):             return Effect("spark", {"pos": pos}, cause)
def set_part(t, kind, key, to, cause=""):  # write PART state through the bus — a consuming
    return Effect("set_part", {"t": t, "kind": kind, "key": key, "to": to}, cause)  # affordance never mutates parts directly
def add_fluid(pos, mat, ml, cause=""):return Effect("add_fluid", {"pos": pos, "mat": mat, "ml": ml}, cause)
def remove_fluid(pos, mat, ml, cause=""):return Effect("remove_fluid", {"pos": pos, "mat": mat, "ml": ml}, cause)
def set_edge(a, b, kind, w, cause=""):return Effect("set_edge", {"a": a, "b": b, "kind": kind, "w": w}, cause)
def set_floor(pos, mat, cause=""):    return Effect("set_floor", {"pos": pos, "mat": mat}, cause)
def speech(who, text, cause=""):      return Effect("speech", {"who": who, "text": text}, cause)
def clock_tick(cid, n=1, cause=""):   return Effect("clock_tick", {"clock": cid, "n": n}, cause)
def spawn(eid, name, pos, tags=(), props=None, material="flesh", cause=""):
    return Effect("spawn", {"id": eid, "name": name, "pos": pos, "tags": list(tags),
                            "props": dict(props or {}), "material": material}, cause)
def despawn(eid, cause=""):           return Effect("despawn", {"id": eid}, cause)


def resolve(world: World, target):
    """target: a pos tuple -> Cell, or an entity id -> Entity."""
    if isinstance(target, tuple):
        if not world.in_bounds(target):
            raise Incoherence(f"effect references out-of-bounds cell {target}")
        return world.cell(target)
    e = world.entities.get(target)
    if e is None:
        raise Incoherence(f"effect references missing entity {target!r}")
    return e


def _cell(world, pos) -> Cell:
    if not world.in_bounds(pos):
        raise Incoherence(f"effect references out-of-bounds cell {pos}")
    return world.cell(pos)


def apply(world: World, eff: Effect) -> None:
    k, d = eff.kind, eff.data
    if k == "set_tag":
        resolve(world, d["t"]).tags.add(d["tag"])
    elif k == "clear_tag":
        resolve(world, d["t"]).tags.discard(d["tag"])
    elif k == "adjust_prop":
        o = resolve(world, d["t"])
        o.props[d["prop"]] = o.props.get(d["prop"], 0.0) + d["delta"]
        # provenance: harm remembers its author — the victim carries hit_by:<actor>
        # so rules can attribute the kill (bounties, revenge, a game-show's xp)
        if d["prop"] == "hp" and d["delta"] < 0 and eff.actor and isinstance(o, Entity):
            o.tags = {t for t in o.tags if not t.startswith("hit_by:")} | {f"hit_by:{eff.actor}"}
    elif k == "set_prop":
        resolve(world, d["t"]).props[d["prop"]] = d["value"]
    elif k == "set_material":
        resolve(world, d["t"]).material = d["mat"]
    elif k == "move":
        e = world.entities.get(d["id"])
        if e is None:
            raise Incoherence(f"move of missing entity {d['id']!r}")
        if not world.in_bounds(d["pos"]):
            raise Incoherence(f"move {e.id} out-of-bounds {d['pos']}")
        e.pos = d["pos"]
    elif k == "add_heat":
        _cell(world, d["pos"]).heat += d["delta"]
    elif k == "spark":
        # holding a flame to a spot brings it TO flame temperature — it never stacks
        c = _cell(world, d["pos"])
        c.heat = max(c.heat, 140.0)
    elif k == "add_fluid":
        c = _cell(world, d["pos"])
        c.fluids[d["mat"]] = c.fluids.get(d["mat"], 0.0) + d["ml"]
    elif k == "remove_fluid":
        c = _cell(world, d["pos"])
        left = c.fluids.get(d["mat"], 0.0) - d["ml"]
        if left > 1e-6:
            c.fluids[d["mat"]] = left
        else:
            c.fluids.pop(d["mat"], None)
    elif k == "set_edge":
        w = max(-1.0, min(1.0, d["w"])) if d["kind"] == "disposition" else d["w"]
        world.edges.setdefault((d["a"], d["b"]), {})[d["kind"]] = w
    elif k == "set_floor":
        _cell(world, d["pos"]).floor = d["mat"]
    elif k == "speech":
        # dialogue mutates no state, but flows through the one log like everything else
        if d["who"] not in world.entities:
            raise Incoherence(f"speech from missing entity {d['who']!r}")
    elif k == "spawn":
        # entity lifecycle — the one primitive a closed-state VM still needs
        # (summon a beast, mint a loot drop, birth a chicken). Through the bus, logged.
        if d["id"] in world.entities:
            raise Incoherence(f"spawn of existing entity {d['id']!r}")
        if not world.in_bounds(d["pos"]):
            raise Incoherence(f"spawn out-of-bounds {d['pos']}")
        world.entities[d["id"]] = Entity(d["id"], d.get("name", d["id"]), d["pos"],
                                         material=d.get("material", "flesh"),
                                         tags=set(d.get("tags", [])), props=dict(d.get("props", {})))
    elif k == "despawn":
        world.entities.pop(d["id"], None)
    elif k == "set_part":
        e = world.entities.get(d["t"])
        if e is None or d["kind"] not in e.parts:
            raise Incoherence(f"set_part on missing part {d['t']!r}.{d['kind']}")
        e.parts[d["kind"]][d["key"]] = d["to"]
    elif k == "clock_tick":
        c = world.clocks.get(d["clock"])
        if c is None:
            raise Incoherence(f"tick of unknown clock {d['clock']!r}")
        if not c.fired:
            c.fill = min(c.size, c.fill + d["n"])
            if c.fill >= c.size:
                c.fired = True
                if c.on_full:
                    eff.cause = eff.cause or c.on_full   # the bell rings on the log
    elif k == "fact":
        world.facts[d["q"]] = d["truth"]               # oracle canon — permanent
    elif k == "consequence":
        pass                                            # marker: a cost was paid (log-only)
    elif k == "gm_move":
        pass                                            # marker: the Director acted (log-only)
    elif k == "fallback":
        pass                                            # marker: a stalled request (eval-case candidate)
    else:
        raise Incoherence(f"unknown effect kind {k!r}")

    from .state import LogEntry
    world.log.append(LogEntry(world.tick, eff.kind, dict(eff.data), eff.cause, eff.actor))


def apply_all(world: World, effects) -> None:
    for e in effects:
        apply(world, e)
