"""Data-driven reactions: rules are DATA (data/reactions.yaml), read by a small
interpreter, evaluated every tick over every LOCUS (cell or entity) through one
uniform interface. This is the schema from the old worlds/default/reactions.yaml
(has_tag / prop_above / adjacent_tag / damage / conduct_heat / change_material…)
re-targeted at the unified cell substrate.

A rule reads state and PROPOSES deltas; it never mutates. The engine gathers all
proposals for the tick, then applies them through the one bus.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import yaml

from . import effects as fx
from . import parts
from .state import AMBIENT, FLUIDS, World, is_solid, mat_tags

# irregular third-person verbs whose base isn't just "drop the s/es" — narration
# is authored in third person ("{name} catches fire") and re-agreed when {name}=you.
_IRREGULAR_2P = {"is": "are", "dies": "die", "does": "do", "has": "have", "goes": "go"}


def _second_person(line: str) -> str:
    """Re-conjugate an authored third-person narration line to second person when
    the subject is 'you'. Handles the irregulars explicitly and the regular
    -es/-s classes generically, so a new verb never needs a bespoke replace()."""
    line = line.replace("you's", "your")

    def deconj(m):
        v = m.group(1)
        if v in _IRREGULAR_2P:
            return "you " + _IRREGULAR_2P[v]
        if v.endswith("ies"):        return "you " + v[:-3] + "y"      # flies → fly
        if v.endswith(("ches", "shes", "sses", "xes", "zes")):
            return "you " + v[:-2]                                     # catches → catch
        if v.endswith("s"):          return "you " + v[:-1]           # screams → scream
        return m.group(0)

    return re.sub(r"\byou ([a-z]+)\b", deconj, line)

FLOWABLE = {"oil", "water", "acid"}   # liquids spread; gases (oxygen/steam) don't, here

_DATA = os.path.join(os.path.dirname(__file__), "data", "reactions.yaml")


@dataclass
class Locus:
    target: object          # pos tuple (cell) or entity id (str)
    pos: tuple
    tags: set
    material: object
    props: dict
    temperature: float
    fluids: dict
    is_entity: bool
    name: str


def _loci(world: World):
    for pos, c in world.cells.items():
        yield Locus(pos, pos, c.tags | mat_tags(c.material), c.material, c.props,
                    c.heat, c.fluids, False, f"the {c.material or 'space'} at {pos}")
    for e in world.entities.values():
        cell = world.cell(e.pos)
        yield Locus(e.id, e.pos, e.tags | mat_tags(e.material), e.material, e.props,
                    cell.heat, cell.fluids, True, e.name)


def _flammable_fluid(fluids):
    return any(FLUIDS.get(m, {}).get("flammable") and ml > 0 for m, ml in fluids.items())


def _has_fuel(lo: Locus):
    return _flammable_fluid(lo.fluids) or "flammable" in lo.tags


def _field(lo: Locus, prop):
    return lo.temperature if prop == "temperature" else lo.props.get(prop, 0.0)


# --- conditions -------------------------------------------------------------
def _cond(lo: Locus, world: World, c: dict) -> bool:
    (k, v), = c.items()
    if k == "has_tag":            return v in lo.tags
    if k == "missing_tag":        return v not in lo.tags
    if k == "has_material_tag":   return v in mat_tags(lo.material)
    if k == "is_entity":          return lo.is_entity == v
    if k == "is_cell":            return lo.is_entity != v
    if k == "has_flammable_fluid":return _flammable_fluid(lo.fluids) == v
    if k == "has_fuel":           return _has_fuel(lo) == v
    if k == "has_fluid":          return lo.fluids.get(v, 0.0) > 0
    if k == "cell_on_fire":       return ("on_fire" in world.cell(lo.pos).tags) == v
    if k == "cell_has_tag":       return v in world.cell(lo.pos).tags   # location-coupled: spirit vein, cursed ground
    if k == "at_level":           return lo.pos[2] == v                 # which storey (floor-clear conditions)
    if k == "cell_has_fluid":     # {mat: water, present: false} → true when the cell has NO water
        return (world.cell(lo.pos).fluids.get(v["mat"], 0.0) > 0) == v.get("present", True)
    if k == "prop_above":         return _field(lo, v["prop"]) > v["value"]
    if k == "prop_below":         return _field(lo, v["prop"]) < v["value"]
    if k == "adjacent_tag":
        for np in world.neighbors(lo.pos):
            if v in (world.cell(np).tags | mat_tags(world.cell(np).material)):
                return True
            if any(v in e.tags for e in world.entities_at(np)):
                return True
        return False
    if k == "adjacent_fluid":
        return any(world.cell(np).fluids.get(v, 0.0) > 0 for np in world.neighbors(lo.pos))
    if k == "fluid_above":
        return lo.fluids.get(v["mat"], 0.0) > v["ml"]
    if k == "floor_is":
        return (not lo.is_entity) and world.cell(lo.pos).floor == v
    if k == "below_hotter_than":            # a floor chars from the flames beneath it
        below = (lo.pos[0], lo.pos[1], lo.pos[2] - 1)
        return world.in_bounds(below) and world.cell(below).heat > v
    raise ValueError(f"unknown condition {k!r}")


# --- effects (compile to core deltas) ---------------------------------------
def _effect(lo: Locus, world: World, e: dict) -> list:
    (k, v), = e.items()
    t, pos = lo.target, lo.pos
    if k == "add_tag":     return [fx.set_tag(t, v)]
    if k == "remove_tag":  return [fx.clear_tag(t, v)]
    if k == "damage":      return [fx.adjust_prop(t, "hp", -v)]
    if k == "heal":        return [fx.adjust_prop(t, "hp", v)]
    if k == "adjust_prop": return [fx.adjust_prop(t, v["prop"], v["delta"])]
    if k == "change_material": return [fx.set_material(t, None if v == "air" else v)]
    if k == "damage_by_excess":
        return [fx.adjust_prop(t, "hp", -max(0.0, _field(lo, v["prop"]) - v["threshold"]) * v["rate"])]
    if k == "damage_by_deficit":
        return [fx.adjust_prop(t, "hp", -max(0.0, v["threshold"] - _field(lo, v["prop"])) * v["rate"])]
    if k == "relax_heat":  return [fx.add_heat(pos, (AMBIENT - lo.temperature) * v)]
    if k == "radiate":
        # fires burn AT combustion temperature — heating (self AND received) only
        # fills the headroom below `max`, so packed fires saturate instead of stacking.
        # Received heat is ENERGY: temperature change scales by the receiver's
        # thermal mass (metal answers fast, stone slowly — the DF formula).
        from .state import specific_heat
        ceil = v.get("max", 550.0)
        out = [fx.add_heat(pos, min(v.get("self", 0.0), max(0.0, ceil - lo.temperature)))]
        for np in world.neighbors(pos):
            ncell = world.cell(np)
            room = max(0.0, ceil - ncell.heat)
            out.append(fx.add_heat(np, min(v.get("neighbors", 0.0) / specific_heat(ncell.material), room)))
        return out
    if k == "conduct_heat":
        nb = world.neighbors(pos); out = []
        for np in nb:
            transfer = (lo.temperature - world.cell(np).heat) * v / max(1, len(nb))
            out += [fx.add_heat(np, transfer), fx.add_heat(pos, -transfer)]
        return out
    if k == "consume_flammable_fluid":
        for m, ml in lo.fluids.items():
            if FLUIDS.get(m, {}).get("flammable") and ml > 0:
                return [fx.remove_fluid(pos, m, v["ml"])]
        return []
    if k == "consume_fluid":
        return [fx.remove_fluid(pos, v["mat"], v["ml"])]
    if k == "destroy_floor":
        return [fx.set_floor(pos, None)]
    if k == "spill_contents":
        ent = world.entities.get(lo.target) if lo.is_entity else None
        lv = (ent.parts.get("liquid_volume") if ent is not None else None) or {}
        if lv.get("ml", 0) > 0:
            return [fx.add_fluid(pos, lv.get("mat", "water"), lv["ml"]),
                    fx.set_part(lo.target, "liquid_volume", "ml", 0.0)]
        return []
    if k == "clock_tick":       # a scene without this clock simply has no countdown
        return [fx.clock_tick(v["clock"], v.get("n", 1))] if v["clock"] in world.clocks else []
    if k == "adjust_entity":    # credit a NAMED entity (the kill's xp goes to the killer)
        return [fx.adjust_prop(v["id"], v["prop"], v["delta"])] if v["id"] in world.entities else []
    if k == "descend":          # step down through the floor here (stairs, hatches)
        below = (pos[0], pos[1], pos[2] - 1)
        return [fx.move(t, below)] if lo.is_entity and world.in_bounds(below) else []
    if k == "transfer_prop":
        # continuous, CONSERVED movement of a prop between an entity and the cell
        # it stands on — clamped by what the source actually holds (you cannot give
        # qi you don't have). The substrate for qi-into-the-land, siphons, drains.
        if not lo.is_entity:
            return []
        if v.get("to") == "cell":
            amt = min(lo.props.get(v["prop"], 0.0), v["amount"])
            pair = [fx.adjust_prop(t, v["prop"], -amt), fx.adjust_prop(pos, v["prop"], amt)]
        else:                                   # to: actor — draw up out of the ground
            amt = min(world.cell(pos).props.get(v["prop"], 0.0), v["amount"])
            pair = [fx.adjust_prop(pos, v["prop"], -amt), fx.adjust_prop(t, v["prop"], amt)]
        return pair if amt > 0.5 else []    # a trickle is not a transfer — say so
                                            # (playtest: 3 "pours" from a near-empty
                                            # cultivator read as triumphs, did nothing)
    if k == "drop_inventory":
        # the dead drop what they carried — items fall to the corpse's cell and
        # become lootable again (taken cleared). Same shape as spill_contents, over
        # inv:<id> tags. This is why killing the prize-holder finally yields the prize.
        out = []
        for tag in list(lo.tags):
            if tag.startswith("inv:") and tag[4:] in world.entities:
                iid = tag[4:]
                out += [fx.move(iid, pos, f"the {world.entities[iid].name} falls from {lo.name}"),
                        fx.clear_tag(iid, "taken"), fx.clear_tag(t, tag)]
        if any(tag.startswith("inv:") for tag in lo.tags):
            out.append(fx.clear_tag(t, "has_prize"))
        return out
    if k == "add_heat":
        return [fx.add_heat(pos, v)]
    if k == "flow_liquids":
        # cellular-automaton liquid: DOWN first (waterfalls), then excess above
        # `cling` flows to lateral neighbours at a per-fluid rate
        cell = world.cell(pos)
        out = []
        x, y, z = pos
        below = (x, y, z - 1)
        if (z > 0 and not cell.floor and world.in_bounds(below)
                and not is_solid(world.cell(below).material)):
            for mat, amt in list(cell.fluids.items()):
                if mat in FLOWABLE and amt > 0.5:
                    out += [fx.remove_fluid(pos, mat, amt * 0.9),
                            fx.add_fluid(below, mat, amt * 0.9)]
            return out
        for mat, amt in list(cell.fluids.items()):
            if mat not in FLOWABLE:
                continue
            fdef = FLUIDS.get(mat, {})
            keep = fdef.get("cling", v.get("keep", 150.0))
            rate = fdef.get("flow_rate", v.get("rate", 0.25))
            if amt <= keep:
                continue
            lowers = []
            for npos in world.neighbors(pos):
                if npos[2] != pos[2] or is_solid(world.cell(npos).material):
                    continue            # lateral spread only; vertical is the waterfall above
                namt = world.cell(npos).fluids.get(mat, 0.0)
                if amt - namt > 1.0:
                    lowers.append((npos, namt))
            if not lowers:
                continue
            share = (amt - keep) / len(lowers)      # cap total outflow to the excess
            for npos, namt in lowers:
                flow = min((amt - namt) * rate, share)
                if flow > 0.5:
                    out += [fx.remove_fluid(pos, mat, flow), fx.add_fluid(npos, mat, flow)]
        return out
    raise ValueError(f"unknown effect {k!r}")


_RULES: list[dict] | None = None
_KNOWN: set | None = None


def _voice(world, key: str, nar):
    """narrative: may be one line or a VARIANT LIST — same beat, different words
    each telling. Seeded hash (world seed+tick+rule id), NOT world.rng: narration
    must never perturb the dice stream. Variants are authored data — the place
    big models pour the books' voice in, offline."""
    if not isinstance(nar, list):
        return nar
    # a real bit-mixer, not a linear step: tick*131 with 2-tick turns cycled
    # through only half the variants (playtest caught the 3rd repeat verbatim)
    h = (world.tick * 2654435761 ^ world.seed * 2246822519
         ^ sum((i + 7) * ord(c) for i, c in enumerate(key))) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 2654435761) & 0xFFFFFFFF
    return nar[(h ^ (h >> 11)) % len(nar)]


def rules() -> list[dict]:
    global _RULES
    if _RULES is None:
        with open(_DATA) as f:
            _RULES = yaml.safe_load(f)["reactions"]
    return _RULES


def load_pack(name: str) -> dict:
    """A LitRPG "system" is a DATA pack: reactions the same interpreter runs, plus
    CONCEPTS (the words this world gives meaning to) and VERBS (free-text macros
    over those concepts). Append to world.packs to layer it on the physics —
    that's the whole mechanism. No new engine code per system."""
    path = os.path.join(os.path.dirname(__file__), "data", "packs", f"{name}.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def world_concepts(world) -> set:
    """Every word the ACTIVE packs give meaning to, in this world."""
    return {c for p in world.packs for c in p.get("concepts", [])}


def known_concepts() -> set:
    """Every concept ANY pack on disk defines — words that are SOMEONE'S magic.
    A world whose packs don't grant one refuses it in fiction ("you strain at
    nothing") instead of letting the parser mangle it into something mundane."""
    global _KNOWN
    if _KNOWN is None:
        _KNOWN = set()
        d = os.path.join(os.path.dirname(__file__), "data", "packs")
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".yaml"):
                with open(os.path.join(d, fn)) as f:
                    _KNOWN |= set(yaml.safe_load(f).get("concepts", []))
    return _KNOWN


def _locus_of(world, e):
    c = world.cell(e.pos)
    return Locus(e.id, e.pos, e.tags | mat_tags(e.material), e.material, e.props,
                 c.heat, c.fluids, True, e.name)


def pack_verb(world, actor, text):
    """Resolve free text against the active packs' verb vocabulary. A verb fires
    when one of its concept-words AND one of its action-words both appear — so
    "put qi in the ground" is an ACTION in a cultivation world and nonsense
    elsewhere. Returns ("do", effects, cost) | ("fail", line) | None. The verb's
    `do` uses the SAME effect schema as reactions, compiled over the actor."""
    words = set(re.findall(r"[a-z']+", text.lower()))
    for p in world.packs:
        for v in p.get("verbs", []):
            cs = v["concept"] if isinstance(v["concept"], list) else [v["concept"]]
            if not (set(cs) & words) or not (set(v["match"]) & words):
                continue
            if v.get("at") == "cell":               # act ON the ground underfoot
                c = world.cell(actor.pos)
                lo = Locus(tuple(actor.pos), tuple(actor.pos), c.tags | mat_tags(c.material),
                           c.material, c.props, c.heat, c.fluids, False,
                           f"the ground at {actor.pos}")
            elif v.get("at") == "entity":           # act on an adjacent tagged entity
                tgt = next((e for e in world.entities.values() if v["tag"] in e.tags
                            and e.pos[2] == actor.pos[2]
                            and max(abs(e.pos[0] - actor.pos[0]),
                                    abs(e.pos[1] - actor.pos[1])) <= 1), None)
                if tgt is None:
                    return ("fail", _voice(world, v["id"],
                                           v.get("missing", "There is no such thing within reach.")))
                lo = _locus_of(world, tgt)
            else:
                lo = _locus_of(world, actor)
            if not all(_cond(lo, world, cnd) for cnd in v.get("when", [])):
                return ("fail", _voice(world, v["id"], v.get("fail", "Nothing answers.")))
            effs = []
            for spec in v["do"]:
                effs.extend(_effect(lo, world, spec))
            if not effs:                        # e.g. a transfer with an empty source
                return ("fail", _voice(world, v["id"], v.get("fail", "Nothing answers.")))
            effs[0].cause = _voice(world, v["id"], v.get("narrative", ""))
            return ("do", effs, v.get("cost", 1),
                    _voice(world, v["id"], v.get("again", "You continue, unhurried.")))
    return None


def pack_affords(world, actor, seen):
    """The MENU surface of the packs' verbs (found in run 2: 'open'/'descend' lived only on
    the free-text path, so the piece table could not win the dungeon). Same gates as
    pack_verb — entity verbs bind only to SEEN ids (the `seen` list) at arm's reach; cell
    and self verbs read the actor's own ground. Yields the parts.affords 5-tuple shape."""
    out = []
    for p in world.packs:
        for v in p.get("verbs", []):
            if v.get("at") == "cell":
                c = world.cell(actor.pos)
                cands = [(Locus(tuple(actor.pos), tuple(actor.pos),
                                c.tags | mat_tags(c.material), c.material, c.props,
                                c.heat, c.fluids, False, f"the ground at {actor.pos}"),
                          None, {})]
            elif v.get("at") == "entity":
                cands = [(_locus_of(world, e), e, {"target": e.id})
                         for e in (world.entities.get(i) for i in seen)
                         if e is not None and v["tag"] in e.tags and e.pos[2] == actor.pos[2]
                         and max(abs(e.pos[0] - actor.pos[0]),
                                 abs(e.pos[1] - actor.pos[1])) <= 1]
            else:
                cands = [(_locus_of(world, actor), None, {})]
            for lo, tgt, args in cands:
                if not all(_cond(lo, world, cnd) for cnd in v.get("when", [])):
                    continue
                effs = []
                for spec in v["do"]:
                    effs.extend(_effect(lo, world, spec))
                if not effs:
                    continue
                effs[0].cause = _voice(world, v["id"], v.get("narrative", ""))
                name = v["match"][0]
                out.append((f"{name} the {tgt.name}" if tgt else name, effs,
                            tuple(tgt.pos) if tgt else tuple(actor.pos), name, args))
    return out


def _conduction(world: World):
    """DF's one-formula heat model: each adjacent pair exchanges Δtemp·k, scaled by
    each side's specific heat — metal answers fire fast, stone slow — plus a small
    ambient leak. Replaces flat relax/conduct special-case rules. Direct mutation
    (not bus deltas): this is background physics, hundreds of tiny flows per tick."""
    from .state import specific_heat
    K, LEAK = 0.12, 0.06
    cells = world.cells
    flows = []
    for p, c in cells.items():
        x, y, z = p
        for q in ((x + 1, y, z), (x, y + 1, z), (x, y, z + 1)):   # each pair once
            n = cells.get(q)
            if n is None:
                continue
            if q[2] != z and n.floor:       # a floor slab insulates the storey above
                continue
            flows.append((c, n, (n.heat - c.heat) * K))
    for c, n, f in flows:
        c.heat += f / specific_heat(c.material)
        n.heat -= f / specific_heat(n.material)
    for p, c in cells.items():
        if "on_fire" not in c.tags:
            c.heat += (AMBIENT - c.heat) * LEAK / specific_heat(c.material)


def _gravity(world: World):
    """Unsupported things fall one level per tick; landing hurts by the drop.
    Runs through the bus like everything else — a fall someone CAUSED is on the log."""
    from .spatial import supported
    out = []
    for e in world.entities.values():
        if "taken" in e.tags or not (e.tags & {"person", "item", "player"}):
            continue
        x, y, z = e.pos
        seen = e.tags & {"person", "player"}
        if z > 0 and not supported(world, e.pos):
            e.mind["fall"] = e.mind.get("fall", 0) + 1
            msg = "the floor gives way — you plunge into the dark below!" if "player" in e.tags \
                else f"{e.name} plunges down through a gap in the floor!" if seen else ""
            out.append(fx.move(e.id, (x, y, z - 1), msg))
        elif e.mind.get("fall"):
            drop = e.mind.pop("fall")
            if "hp" in e.props:
                dmg = 3.0 * drop * max(1.0, e.props.get("size", 1.0))
                land = ("you slam into the ground below" if "player" in e.tags
                        else f"{e.name} slams into the ground" if seen else "")
                out.append(fx.adjust_prop(e.id, "hp", -dmg, land))
    fx.apply_all(world, out)


def tick(world: World) -> None:
    world.tick += 1
    _conduction(world)
    _gravity(world)
    proposed = parts.step(world)            # D13 behavior layer: parts act BEFORE rules read heat
    loci = list(_loci(world))
    packed = [r for p in world.packs for r in p.get("reactions", [])]
    for rule in rules() + packed:           # base physics + whatever LitRPG systems are active
        prob = rule.get("probability", 1.0)
        for lo in loci:
            if not all(_cond(lo, world, c) for c in rule["when"]):
                continue
            if prob < 1.0 and world.rng.random() > prob:
                continue
            effs = []
            for spec in rule["do"]:
                effs.extend(_effect(lo, world, spec))
            if effs:
                nar = _voice(world, rule["id"], rule.get("narrative"))
                if nar and not rule.get("silent"):
                    line = nar.replace("{name}", lo.name).replace("{pos}", str(lo.pos))
                    if lo.name == "you":                 # second-person agreement
                        line = _second_person(line)
                    effs[0].cause = line
                proposed.extend(effs)
    fx.apply_all(world, proposed)
    # death wins the tick: a same-tick heal/level-up must not leave a corpse with
    # positive hp (playtest INCOHERENCE: "player is dead but hp=8.0" — the level_2
    # reaction fired on the same tick fall damage killed Carl).
    for e in world.entities.values():
        if "dead" in e.tags and e.props.get("hp", 0.0) > 0.0:
            e.props["hp"] = 0.0
