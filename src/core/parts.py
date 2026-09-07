"""D13 parts — the BEHAVIOR layer of the hybrid schema (design doc §2a).

The boundary rule: a *material property* lives in a MATERIALS template; a *static boolean
the menu matches* is a tag; anything that CHANGES OVER TICKS or HOLDS A RESOURCE is a part.
Most entities carry no parts at all.

A part is `entity.parts[kind] = {…state…}`. Its stepper may mutate its own private counters
directly (sim-internal physics, like conduction) — but every WORLD-VISIBLE consequence goes
out as a typed effect on the bus, so parts can never bypass the log/replay contract.
`step(world)` is called once per reactions.tick; parts also advertise affordances to the
derived menu (that half lands with engine item 2)."""
from . import effects as fx


def _fuel_burn(world, e, p):
    """A self-consuming flame (torch, brazier, campfire): while fueled it keeps its OWN cell
    hot each tick; when the fuel runs out it gutters — one 'spent' tag, then silence. The
    behavior `holds:` props could never express: state that acts over time."""
    if p.get("ticks", 0) > 0:
        p["ticks"] -= 1
        return [fx.add_heat(e.pos, p.get("heat", 40.0))]
    if "spent" not in e.tags:
        return [fx.set_tag(e.id, "spent", f"the {e.name} gutters out, its fuel spent")]
    return []


def _fuel_burn_affords(world, e, p, actor):
    """A live flame invites hands: snuff it. The option exists BECAUSE the part does —
    no verb table anywhere (D2 in miniature); consuming it writes part state via the bus."""
    if p.get("ticks", 0) > 0 and "spent" not in e.tags:
        return [(f"snuff the {e.name}",
                 [fx.set_part(e.id, "fuel_burn", "ticks", 0, f"you snuff the {e.name}"),
                  fx.set_tag(e.id, "spent")], e.pos, "snuff", {"target": e.id})]
    return []


DRAUGHT = 200.0                                        # one drawn skinful — the PART's constant


def _liquid_volume_affords(world, e, p, actor):
    """A held liquid is two invitations: TIP it out where it stands, or DRAW a skinful.
    The amounts are the part's own constants (D6): the author picks the verb, the sim owns
    every number. No stepper — a resource part acts only when acted upon."""
    ml, mat = p.get("ml", 0.0), p.get("mat", "water")
    out = []
    if ml > 0:
        out.append((f"tip the {e.name} over",
                    [fx.add_fluid(e.pos, mat, ml, f"you tip the {e.name}, {mat} flooding out"),
                     fx.set_part(e.id, "liquid_volume", "ml", 0.0)], e.pos,
                    "tip", {"target": e.id}))
    if ml >= DRAUGHT and f"inv:{mat}" not in actor.tags:
        out.append((f"draw {mat} from the {e.name}",
                    [fx.set_tag(actor.id, f"inv:{mat}", f"you draw {mat} from the {e.name}"),
                     fx.set_part(e.id, "liquid_volume", "ml", ml - DRAUGHT)], e.pos,
                    "draw", {"what": mat, "from": e.id}))
    return out


REGISTRY = {"fuel_burn": _fuel_burn}
AFFORDS = {"fuel_burn": _fuel_burn_affords, "liquid_volume": _liquid_volume_affords}


def step(world):
    """One beat of part behavior across the world → the effects to propose this tick."""
    effs = []
    for e in list(world.entities.values()):
        for kind, p in getattr(e, "parts", {}).items():
            fn = REGISTRY.get(kind)
            if fn:
                effs.extend(fn(world, e, p) or [])
    return effs


def affords(world, e, actor):
    """What an entity's parts OFFER an adjacent actor: [(label, effects, target_pos, verb, args)].
    Plain tuples — the engine wraps them into menu Options (no import cycle)."""
    out = []
    for kind, p in getattr(e, "parts", {}).items():
        fn = AFFORDS.get(kind)
        if fn:
            out.extend(fn(world, e, p, actor) or [])
    return out
