"""Seed scene — the vault, now populated. A cast with distinct traits and agendas,
so the stage is full of people with somewhere to be (not statues).

Emergent setups: the thief walks to the vial to steal it; the guard holds the
vault; the apprentice works — and if you ignite the oil, each reacts by *their
own* nerve (the brave guard holds, the jittery apprentice bolts), decided by a
dice roll, not a blanket flee.
"""

from __future__ import annotations

import random as _random

from .state import Cell, Entity, World, is_solid


def _person(eid, name, pos, traits, flaw, bond, composure=50.0, job=None):
    m = {"traits": traits, "flaw": flaw, "bond": bond, "salience": 0.0}
    if job:
        m["job"] = job
    return Entity(eid, name, pos, material="flesh",
                  tags={"alive", "person", "calm"},
                  props={"hp": 10.0, "composure": composure}, mind=m)


def vault(seed: int = 0) -> tuple[World, Entity]:
    dims = (6, 4)
    cells = {(x, y): Cell(material=None) for x in range(6) for y in range(4)}
    cells[(3, 3)] = Cell(material="wood", tags={"support"}, props={"hp": 6.0})   # corrodible via material
    cells[(0, 3)].tags.add("vault")          # the guard's post
    cells[(1, 0)].tags.add("workbench")      # the apprentice's station
    # fluids live in CONTAINERS; they spill only when the container breaks
    # (smashed, corroded through, or burst by fire). parts["liquid_volume"] = {mat, ml}.
    barrels = [
        Entity("barrel_oil", "barrel of oil", (3, 1), material="wood",
               tags={"item", "container", "heavy"}, props={"hp": 4.0}, parts={"liquid_volume": {"mat": "oil", "ml": 800.0}}),
        Entity("keg_oil", "keg of oil", (4, 1), material="wood",
               tags={"item", "container", "heavy"}, props={"hp": 3.0}, parts={"liquid_volume": {"mat": "oil", "ml": 400.0}}),
        Entity("cask_water", "cask of water", (1, 1), material="wood",
               tags={"item", "container", "heavy"}, props={"hp": 4.0}, parts={"liquid_volume": {"mat": "water", "ml": 600.0}}),
    ]

    player = Entity("player", "you", (0, 0), material="flesh",
                    tags={"player", "alive", "inv:torch", "inv:oil", "inv:acid", "inv:water"},
                    props={"hp": 10.0})
    vial = Entity("vial", "starfire vial", (5, 3), material="glass", tags={"prize", "item"})

    thief = _person("thief", "Sly the cutpurse", (5, 1),
                    {"bravery": 0.5, "warmth": 0.2, "self_interest": 0.85, "aggression": 0.45},
                    "greed will be his undoing", "owes the financier a blood-debt", composure=55)
    guard = _person("guard", "Bran the doorman", (0, 2),
                    {"bravery": 0.85, "warmth": 0.5, "self_interest": 0.2, "aggression": 0.5},
                    "too proud to retreat", "sworn to protect the guild", composure=70)
    pip = _person("apprentice", "Pip the apprentice", (2, 0),
                  {"bravery": 0.2, "warmth": 0.85, "self_interest": 0.4, "aggression": 0.1},
                  "flinches at everything", "idolizes the alchemist", composure=35)

    world = World(dims=dims, cells=cells,
                  entities={e.id: e for e in [player, vial, thief, guard, pip] + barrels}, seed=seed)
    world.edges[("thief", "player")] = {"disposition": 0.0}
    world.edges[("guard", "player")] = {"disposition": 0.0}
    world.edges[("apprentice", "player")] = {"disposition": 0.0}
    return world, player


def guildhall(seed: int = 0) -> tuple[World, Entity]:
    """The Gilded Cinder guildhall on auction night — the full stage.

    Three spaces: the auction HALL (west), the VAULT ROOM (north-east, open
    archway) holding the starfire vial, and a locked STOREROOM (south-east,
    wooden door — burn or smash your way in) full of oil, blackpowder and acid.
    Two rivals (Sly, Voss) both want the vial; Bran guards it; Maren and Pip
    work the hall. Pour oil on the door and light it: fire chars wood away.
    """
    W, H, Z = 14, 9, 2
    cells = {(x, y, z): Cell(material=None) for x in range(W) for y in range(H) for z in range(Z)}

    for y in range(H):                                   # hall/east dividing wall (both storeys)
        cells[(8, y, 0)] = Cell(material="stone")
        cells[(8, y, 1)] = Cell(material="stone")
    cells[(8, 2, 0)] = Cell(material=None)               # open archway to the vault room
    cells[(8, 6, 0)] = Cell(material="wood", tags={"door"}, props={"hp": 5.0})   # storeroom door
    for x in range(9, W):                                # vault room / storeroom divider
        cells[(x, 4, 0)] = Cell(material="stone")

    cells[(4, 4, 0)] = Cell(material="wood", tags={"support"}, props={"hp": 6.0})
    cells[(2, 1, 0)].tags.add("workbench")
    cells[(11, 1, 0)].tags.add("vault")

    # the TIMBER GALLERY: a wooden balcony over the north hall, reached by stairs.
    # Its floor is slabs — fire below can char through and drop whoever's up there.
    for x in range(1, 7):
        for y in range(1, 4):
            cells[(x, y, 1)].floor = "wood"
    cells[(1, 4, 0)].tags.add("stairs")
    cells[(1, 4, 1)].floor = None                        # the stairwell opening

    player = Entity("player", "you", (1, 4), material="flesh",
                    tags={"player", "alive", "inv:torch", "inv:oil", "inv:acid", "inv:water", "inv:knife"},
                    props={"hp": 10.0})
    vial = Entity("vial", "starfire vial", (12, 1), material="glass", tags={"prize", "item"})

    stock = [
        Entity("barrel_oil", "barrel of oil", (10, 6), material="wood",
               tags={"item", "container", "heavy"}, props={"hp": 4.0}, parts={"liquid_volume": {"mat": "oil", "ml": 900.0}}),
        Entity("keg_oil", "keg of oil", (11, 7), material="wood",
               tags={"item", "container", "heavy"}, props={"hp": 3.0}, parts={"liquid_volume": {"mat": "oil", "ml": 500.0}}),
        Entity("keg_powder", "keg of blackpowder", (12, 6), material="wood",
               tags={"item", "container", "heavy"}, props={"hp": 3.0}, parts={"liquid_volume": {"mat": "blackpowder", "ml": 500.0}}),
        Entity("jar_acid", "glass jar of vitriol", (10, 8), material="glass",
               tags={"item", "container"}, props={"hp": 1.0}, parts={"liquid_volume": {"mat": "acid", "ml": 300.0}}),
        Entity("cask_water", "cask of water", (1, 7), material="wood",
               tags={"item", "container", "heavy"}, props={"hp": 4.0}, parts={"liquid_volume": {"mat": "water", "ml": 900.0}}),
        # the reason the storeroom is locked — loot worth the burned door
        Entity("strongbox", "guild strongbox", (13, 8), material="metal",
               tags={"item", "prize", "heavy"}, props={"hp": 8.0}),
        Entity("purse", "purse of auction gold", (13, 5), material="flesh",
               tags={"item"}, props={}),
        # blades — it's that kind of auction. (taken = carried by their owner)
        Entity("knife", "belt knife", (1, 4), material="metal",
               tags={"item", "weapon", "taken"}, props={"damage": 5.0}),
        Entity("knife_sly", "wicked little knife", (6, 5), material="metal",
               tags={"item", "weapon", "taken"}, props={"damage": 5.0}),
    ]
    cast = [
        _person("thief", "Sly the cutpurse", (6, 5),
                {"bravery": 0.5, "warmth": 0.2, "self_interest": 0.85, "aggression": 0.45},
                "greed will be his undoing", "owes Voss a blood-debt", composure=55),
        _person("guard", "Bran the doorman", (10, 2),
                {"bravery": 0.85, "warmth": 0.5, "self_interest": 0.2, "aggression": 0.5},
                "too proud to retreat", "sworn to protect the guild", composure=70,
                job={"kind": "patrol", "posts": [(10, 2, 0), (8, 2, 0), (12, 2, 0)], "i": 0}),
        _person("apprentice", "Pip the apprentice", (2, 2),
                {"bravery": 0.2, "warmth": 0.85, "self_interest": 0.4, "aggression": 0.1},
                "flinches at everything", "idolizes Maren", composure=35),
        _person("alchemist", "Maren the alchemist", (4, 2),
                {"bravery": 0.55, "warmth": 0.6, "self_interest": 0.3, "aggression": 0.3},
                "cannot resist an experiment", "the vial is her life's work", composure=60),
        _person("financier", "Voss the financier", (4, 2, 1),
                {"bravery": 0.6, "warmth": 0.15, "self_interest": 0.9, "aggression": 0.6},
                "believes everything has a price", "ruined three rivals for less", composure=75),
    ]
    world = World(dims=(W, H, Z), cells=cells,
                  entities={e.id: e for e in [player, vial] + stock + cast}, seed=seed)
    for e in cast:
        world.edges[(e.id, "player")] = {"disposition": 0.0}
    from . import checks
    checks.add_clock(world, "alarm", size=6,
                     on_full="the guild bell rings — the whole household is roused!")
    return world, player


# ── procedural distribution: many small varied rooms for the generalization gate ──
# One seed → one deterministic room. The DISTRIBUTION (not any single room) is the
# point: train on some seeds, hold out others, measure whether learning TRANSFERS.
# Same World contract as the hand-authored scenes; the sim adjudicates all of it.
_FIRST = ["Corin", "Mabel", "Doran", "Isolde", "Fen", "Rys", "Talia", "Ord", "Neve", "Gwyn", "Bram", "Sena"]
_ROLE = ["the sellsword", "the clerk", "the smuggler", "the healer", "the tough", "the scholar", "the fence", "the novice"]
_FLAWS = ["quick to anger", "never forgets a slight", "greedy to a fault", "flinches at everything", "too proud to retreat", "trusts no one"]
_BONDS = ["sworn to the guild", "protects a younger sibling", "owes a dangerous debt", "loyal to the house", "wants only to go home", "serves the highest bidder"]
_STORE = [("oil", 600.0), ("water", 700.0), ("acid", 300.0)]


def generate(seed: int = 0) -> tuple[World, Entity]:
    """A random small single-storey room, deterministic in `seed`. Varies layout,
    cast (randomized traits/flaw/bond), hazards, items and the prize's spot."""
    r = _random.Random(seed)
    W, H = r.randint(6, 10), r.randint(5, 8)
    cells = {(x, y): Cell(material=None) for x in range(W) for y in range(H)}
    for _ in range(r.randint(0, (W * H) // 9)):          # scattered stone obstacles
        cells[(r.randrange(W), r.randrange(H))] = Cell(material="stone")
    for _ in range(r.randint(0, 2)):                     # flammable wooden supports
        p = (r.randrange(W), r.randrange(H))
        cells[p] = Cell(material="wood", tags={"support"}, props={"hp": 6.0})

    free = [p for p, c in cells.items() if not is_solid(c.material)]
    r.shuffle(free)                                      # distinct empty cells for occupants
    take = free.pop

    player = Entity("player", "you", take(), material="flesh",
                    tags={"player", "alive", "inv:torch", "inv:oil", "inv:acid", "inv:water"}, props={"hp": 10.0})
    vial = Entity("vial", "the prize", take(), material="glass", tags={"prize", "item"})
    stock = []
    for i in range(r.randint(1, 3)):
        mat, ml = r.choice(_STORE)
        stock.append(Entity(f"cont{i}", f"cask of {mat}", take(), material="wood",
                            tags={"item", "container", "heavy"},
                            props={"hp": float(r.randint(2, 5))}, parts={"liquid_volume": {"mat": mat, "ml": ml}}))
    cast = []
    for i in range(r.randint(1, 3)):
        traits = {k: round(r.random(), 2) for k in ("bravery", "warmth", "self_interest", "aggression")}
        cast.append(_person(f"npc{i}", f"{r.choice(_FIRST)} {r.choice(_ROLE)}", take(),
                            traits, r.choice(_FLAWS), r.choice(_BONDS), composure=float(r.randint(35, 75))))

    world = World(dims=(W, H), cells=cells,
                  entities={e.id: e for e in [player, vial] + stock + cast}, seed=seed)
    for e in cast:
        world.edges[(e.id, "player")] = {"disposition": 0.0}
    return world, player


def dungeon(seed: int = 0) -> tuple[World, Entity]:
    """DCC floors one AND two: an upper stone gallery (z=1, tiered loot boxes, a
    scuttler, stairs down) over a darker floor below (z=0: the Gold box, a worse
    resident). The dcc pack's descend verb drops you through the stairs; its
    floor-clear rule pays out on arrival. This is just the stage — mechanics
    live in the pack; the collapse Clocks tick from there too."""
    from .checks import add_clock
    r = _random.Random(seed)
    W, H = 9, 6
    cells = {}
    for x in range(W):
        for y in range(H):
            cells[(x, y, 0)] = Cell(material=None)                    # floor two: the dark
            cells[(x, y, 1)] = Cell(material=None, floor="stone")     # floor one: the gallery
    for _ in range(4):                                   # rubbled pillars, one per storey level
        p = (r.randrange(1, W - 1), r.randrange(1, H - 1), r.randrange(2))
        if p[:2] != (W - 2, 1):                          # never atop a scuttler's den
            cells[p] = Cell(material="stone", floor="stone" if p[2] else None)
    cells[(W - 1, H - 1, 1)] = Cell(material=None, tags={"stairs"})   # the way DOWN

    player = Entity("player", "you", (0, 0, 1), material="flesh",
                    tags={"player", "alive", "crawler", "inv:torch", "inv:water"},
                    props={"hp": 10.0, "xp": 0.0})
    loot, boxes = [], []
    TIERS = [("Bronze", "crowbar", "iron crowbar", 1), ("Silver", "coin", "fat silver coin", 1),
             ("Gold", "longsword", "humming blue longsword", 0)]
    for i, (tier, iid, nm, z) in enumerate(TIERS):
        pos = (r.randrange(1, W - 1), r.randrange(1, H - 1), z)
        while is_solid(cells[pos].material):
            pos = (r.randrange(1, W - 1), r.randrange(1, H - 1), z)
        loot.append(Entity(iid, nm, pos, material="metal", tags={"item", "taken"}))
        boxes.append(Entity(f"box{i}", f"{tier} Adventurer Box", pos, material="wood",
                            tags={"item", "loot_box", "sealed", "heavy", f"inv:{iid}"},
                            props={"hp": 4.0}))
    scut = _person("scuttler", "a chittering scuttler", (W - 2, 1, 1),
                   {"bravery": 0.8, "warmth": 0.0, "self_interest": 0.6, "aggression": 0.9},
                   "hungers, always", "belongs to the dungeon", composure=60)
    brood = _person("broodmother", "the pale broodmother", (W - 2, 1, 0),
                    {"bravery": 0.95, "warmth": 0.0, "self_interest": 0.4, "aggression": 1.0},
                    "defends the clutch", "IS the dungeon, in a small way", composure=80)
    for s in (scut, brood):
        s.tags.add("hostile")

    world = World(dims=(W, H, 2), cells=cells,
                  entities={e.id: e for e in [player, scut, brood] + boxes + loot}, seed=seed)
    world.edges[("scuttler", "player")] = {"disposition": -0.5}
    world.edges[("broodmother", "player")] = {"disposition": -0.8}
    # the countdown is ANNOUNCED, book-style — omens at the half and the brink,
    # doom at the end. Each is a clock the dcc pack's countdown rules tick.
    add_clock(world, "omen_half", 30,
              "The System, bored: 'Halfway to floor collapse, Crawlers. The viewers do hope you have a plan.'")
    add_clock(world, "omen_brink", 48,
              "Dust sifts from the ceiling. The System: 'Final stretch. Find the stairs or become part of the show's highlight reel.'")
    add_clock(world, "collapse", 60,
              "The System purrs: 'Floor collapse imminent, Crawlers. Do try to be somewhere else.'")
    return world, player


def farm(seed: int = 0) -> tuple[World, Entity]:
    """The Fa Ram: Jin's OWN stage (playtest: 'farming bolts onto a heist world').
    Field rows, a pond, hens and a young rooster, one neighbour. No vault, no
    prize — the drama here is soil, weather, and what wakes up. Pair with
    --pack cultivation,farming."""
    r = _random.Random(seed)
    W, H = 8, 6
    cells = {(x, y): Cell(material=None) for x in range(W) for y in range(H)}
    for y in (2, 3):
        cells[(0, y)].fluids["water"] = 900.0            # the pond, fed off the hill
    cells[(W - 1, 0)] = Cell(material="wood", tags={"support"}, props={"hp": 8.0})   # barn corner
    player = Entity("player", "you", (2, 1), material="flesh",
                    tags={"player", "alive", "inv:water"}, props={"hp": 10.0, "qi": 24.0})
    hens = [Entity(f"hen{i}", nm, (3 + i, 4), material="flesh",
                   tags={"animal", "alive"}, props={"hp": 4.0, "qi": 0.0})
            for i, nm in enumerate(["black-feathered hen", "fussing brown hen"])]
    rooster = Entity("rooster", "proud young rooster", (5, 4), material="flesh",
                     tags={"animal", "alive"}, props={"hp": 5.0, "qi": 0.0})
    meiling = _person("neighbour", "Meiling the herbalist", (6, 1),
                      {"bravery": 0.6, "warmth": 0.85, "self_interest": 0.3, "aggression": 0.2},
                      "sharp-tongued when fussed over", "keeps the village healthy", composure=65)
    world = World(dims=(W, H), cells=cells,
                  entities={e.id: e for e in [player, meiling, rooster] + hens}, seed=seed)
    world.edges[("neighbour", "player")] = {"disposition": 0.3}
    return world, player
