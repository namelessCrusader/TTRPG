"""THINGS (docs/sim-mind-design.md, 3.3 and step E0): the defects an entity
layer would inherit, fixed before there is one.

An identity is only as good as the fields it is built on. A held blade whose
edge stays behind when the arm moves, a cell of air that still says "broken
piece", a hand that can pick up what the particle solver is carrying: each is
wrong today in a small way, and each would become a wrong FACT about a thing
the moment things have names. So each one is pinned here, by the field it
corrupted:

  1. `_repose` and `_crouch` carry `edge` (a reach used to blunt the tool and
     could sharpen the arm);
  2. a wedged topple puts `edge` back with everything else;
  3. melt and acid unmake a cell whole (`_unmake`), and the identity pass
     clears `edge` and `rubble` in any cell that became air;
  4. a shard lands blunt, whatever the air it lands in once held;
  5. drawn particle matter (`moving`) cannot be reached for, taken, shoved or
     walked through: it is the solver's, and moving the drawing duplicated it;
  6. two fliers in the air at once can both be caught.

Then the entity steps build on them. E1: the eid field rides with the matter
through every law that moves it. E2: things are registered and named; the
identity pass splits a thing that came apart into its pieces (the grip going
with the piece in the hand), and buries one that is nowhere, with its cause;
the menus say the names; and each of the fourteen sites that carry the eid
keeps its own books. E3: a hand holds a named thing by its eid, from the take
to the catch; reach joins cells on the key (a named axe of iron and wood is
one thing, and only itself); `thing()` makes a thing out of whatever its
block put in the world, and audits what its author says it is for; and a
body can carry its own eid. E4: what a thing is now (`props`), reach that
finds every thing in it and does not wrap round its box, a hand that makes
what it takes a thing (adopt-on-take), and the tip check on the key.
"""
import contextlib
import copy

import numpy as np
import pytest

from src.voxel import support
from src.voxel.bodies import AS_BODY, CARRIED
from src.voxel.label import flood, label
from src.voxel.scenes import _person, _Wants, thing
from src.voxel.sim import (ACID, AIR, CHAR, FLESH, GLASS, IRON, MTIN, STONE,
                           TIN, WATER, WOOD, World)

BLADE = 2e-4                                  # an axe bit, m² (test_voxel's)


def _armed():
    """A man who has picked up a blade lying at his feet (the review probe
    `tools/protos/review/edge_probe3.py`)."""
    w = World(64, 20, 40, voxel_cm=5)
    w.fill(0, 64, 0, 20, 0, 1, STONE)
    a = _person(w, 17, 10)
    a["name"] = "attacker"
    w.fill(22, 24, 10, 11, 1, 2, IRON, edge=BLADE)
    w.policy = _Wants(each={"attacker": {"hands": "take", "legs": "stay"}})
    for _ in range(12):
        w.step()
    assert a.get("held"), "he has it in his hand, or nothing below means anything"
    return w, a


def test_a_HELD_EDGE_stays_on_the_tool_through_a_reach_and_back():
    """The edge is a fact about the iron, so it goes where the iron goes.
    Before E0 the first reach left both edged cells behind in the air: the
    tool was blunt for as long as the arm was out, and a swing from there
    could not cut."""
    w, a = _armed()
    arm = a["held"]["arm"]
    moved = 0
    for th in (0.3, 0.6, 0.9, 1.5, 1.0, 0.5, 0.0):
        moved += w._repose(a, arm, th) == "moved"
        iron = w.mat == IRON
        assert np.all(w.edge[iron] == np.float32(BLADE)), \
            f"at {th} rad every cell of the blade still has its edge"
        assert not (w.edge[~iron] > 0).any(), \
            f"at {th} rad and nothing else has one: " \
            f"{[(tuple(c), int(w.mat[tuple(c)])) for c in np.argwhere((w.edge > 0) & ~iron)]}"
    assert moved >= 5, "the arm really went out and came back"


def test_the_FOREARM_inherits_no_edge():
    """Flesh that moves into a cell takes ITS OWN edge there, which is none.
    The cells are seeded with an edge first, as a mover that forgets the
    field leaves them; before E0 the arm and the legs picked it up."""
    w, a = _armed()
    snap = w.snapshot()
    air = w.mat == AIR
    w.edge[air] = 1e-3                        # every empty cell: "sharp"
    assert w._repose(a, a["held"]["arm"], 0.6) == "moved"
    assert not (w.edge[w.mat == FLESH] > 0).any(), \
        "a reaching arm is not a blade"
    assert np.all(w.edge[w.mat == IRON] == np.float32(BLADE)), \
        "and the blade in its hand still is"

    w.restore(snap)
    w.edge[w.mat == AIR] = 1e-3
    assert w._crouch(w.persons[0], 1.0) == "moved"
    assert not (w.edge[w.mat == FLESH] > 0).any(), "nor is a bent knee"


def test_a_WEDGED_TOPPLE_puts_the_EDGE_back():
    """A body that cannot even begin to lean is set back where it stood, with
    everything it had. The edge was lifted with the rest and never restored,
    so a blade propped in a slot came back blunt."""
    w = World(12, 6, 16)
    w.fill(0, 12, 0, 6, 0, 1, STONE)
    w.fill(6, 7, 0, 6, 1, 14, STONE)          # the wall it would lean into
    w.fill(5, 6, 3, 4, 1, 13, IRON, edge=BLADE)
    cells = np.argwhere(w.mat == IRON)
    before = (w.mat.copy(), w.smass.copy(), w.edge.copy())
    w._topple(cells, 0, 1, 5.5, 0.5, owner="nobody")
    assert not w.bodies, "it is wedged: nothing left the lattice"
    assert np.array_equal(w.mat, before[0]) and np.array_equal(w.smass, before[1])
    assert np.array_equal(w.edge, before[2]), "and the edge is where it was"


def test_MELT_and_ACID_leave_no_EDGE_or_RUBBLE_in_the_air():
    """A cell that stops being solid stops being everything a solid was. Melt
    and acid set `mat` and `smass` and left the rest, so the next thing into
    that cell was born sharp, or born a broken piece."""
    w = World(6, 6, 6)
    w.fill(0, 6, 0, 6, 0, 1, STONE)
    w.fill(2, 3, 2, 3, 1, 2, TIN, edge=BLADE)
    w.rubble[2, 2, 1] = True
    w.E[2, 2, 1] = 1e9                        # far past what melts it all
    w._law_melt()
    assert w.mat[2, 2, 1] == AIR and w.fl[2, 2, 1] == MTIN, "it melted"
    assert w.edge[2, 2, 1] == 0.0 and not w.rubble[2, 2, 1]

    w = World(6, 6, 6)
    w.fill(0, 6, 0, 6, 0, 1, STONE)
    w.fill(2, 3, 2, 3, 1, 2, WOOD, edge=BLADE)
    w.smass[2, 2, 1] = 0.005                  # under the 0.01 g floor the acid
                                              # law sweeps away: real vitriol
                                              # chars wood far too slowly to
                                              # empty a cell in one call (R1)
    w.rubble[2, 2, 1] = True
    w.pour(3, 2, 1, ACID, 900.0)
    w._law_acid()
    assert w.mat[2, 2, 1] == AIR, "it dissolved"
    assert w.edge[2, 2, 1] == 0.0 and not w.rubble[2, 2, 1]


def test_the_IDENTITY_PASS_clears_what_a_cell_of_AIR_cannot_be():
    """Whatever emptied a cell, and however it forgot to, the identity pass
    clears `edge` and `rubble` there before anything can arrive. It leaves
    every cell that is still solid alone."""
    w = World(8, 8, 8)
    w.fill(0, 8, 0, 8, 0, 1, STONE)
    w.fill(2, 4, 2, 3, 1, 2, IRON, edge=BLADE)
    w.rubble[2:4, 2, 1] = True
    w._law_identity()                         # nothing has emptied yet
    assert w.edge[2, 2, 1] == np.float32(BLADE) and w.rubble[2, 2, 1]
    w.mat[2, 2, 1] = AIR                      # emptied by hand, fields left
    w.smass[2, 2, 1] = 0.0
    w._law_identity()
    assert w.edge[2, 2, 1] == 0.0 and not w.rubble[2, 2, 1], "the empty cell"
    assert w.edge[3, 2, 1] == np.float32(BLADE) and w.rubble[3, 2, 1], \
        "and its neighbour, still iron, keeps both"


def test_a_SHARD_lands_BLUNT():
    """Rubble is blunt. A shard that lands in a cell of air takes that cell's
    edge as it finds it, and the air beside a broken blade can still hold
    one; it must not become a blade by landing there."""
    w = World(8, 8, 6)
    w.fill(0, 8, 0, 8, 0, 1, STONE)
    w.fill(4, 5, 4, 5, 1, 2, GLASS)
    w.edge[:, :, 1][w.mat[:, :, 1] == AIR] = 1e-3
    w._shatter(4, 4, 1)
    shards = w.rubble & (w.mat == GLASS)
    assert shards.any(), "it broke"
    assert not (w.edge[shards] > 0).any(), "and no shard of it is sharp"


def _drawn_block():
    """A small block of wood sent through the seam and left to come to rest:
    drawn onto the lattice every tick as `moving` cells (drawn_shove.py)."""
    w = World(20, 12, 16, voxel_cm=5)
    w.fill(0, 20, 0, 12, 0, 1, STONE)
    w.fill(8, 10, 5, 7, 1, 3, WOOD)
    w.matter.promote(np.argwhere(w.mat == WOOD))
    for _ in range(15):
        w.step()
    cells = np.argwhere((w.mat == WOOD) & w.moving)
    assert len(cells), "the solver's wood is drawn on the lattice"
    return w, cells


def test_DRAWN_matter_cannot_be_SHOVED_and_no_wood_is_made():
    """The particles are the thing; the drawn cells are where they were drawn
    this tick. A shove used to move the drawing, which then stayed on the
    lattice as wood while the solver drew the same particles again: 600 g
    of wood became 1200."""
    w, cells = _drawn_block()
    m0 = w.total_mass(WOOD)
    assert len(w._object_at(*cells[0])) == 0, "a hand finds no thing there"
    assert not w._shove(cells.astype(np.int32), 3, 0, 0), "and cannot shove it"
    for _ in range(3):
        w.step()
    assert w.total_mass(WOOD) == pytest.approx(m0, abs=1e-3), \
        "every gram of wood is still the solver's, and only once"
    assert not ((w.mat == WOOD) & ~w.moving).any(), "none of it on the lattice"


def test_DRAWN_matter_is_not_within_REACH():
    """Taking starts with what a hand could close on. The same block, at the
    same man's feet, is a thing to take while it is lattice and nothing to
    take once the solver has it."""
    def setup(promote):
        w = World(40, 24, 44, voxel_cm=5)
        w.fill(0, 40, 0, 24, 0, 1, STONE)
        p = _person(w, 12, 12)
        p["name"] = "taker"
        w.fill(16, 18, 11, 13, 1, 3, WOOD)
        w.policy = _Wants(each={"taker": {"hands": "keep", "legs": "stay"}})
        if promote:
            w.matter.promote(np.argwhere(w.mat == WOOD))
        for _ in range(4):
            w.step()
        comp, sl = w._person_cells(p)
        own = np.argwhere(comp)
        own[:, 0] += sl[0].start
        own[:, 1] += sl[1].start
        return w, w._objects_within_reach(p, own)

    w, seen = setup(False)
    assert any(o["mat"] == WOOD for o in seen), \
        f"on the lattice it is in reach ({seen})"
    w, seen = setup(True)
    assert w.moving.any(), "the solver has it"
    assert not any(o["mat"] == WOOD for o in seen), \
        f"in the solver it is not ({seen})"


def test_TWO_FLIERS_can_both_be_CAUGHT_in_one_tick():
    """`b in self.bodies` compared body dicts by equality the moment `b` was
    not first in the list, and a body dict holds arrays, so catching the
    second of two fliers raised instead of catching it."""
    w = World(40, 30, 44, voxel_cm=5)
    w.fill(0, 40, 0, 30, 0, 1, STONE)
    p = _person(w, 12, 10)
    p["name"] = "first"
    q = _person(w, 12, 20)
    q["name"] = "second"
    w.policy = _Wants()
    w.step()
    a = w.loose((20, 10, 20), (0.0, 0.0, -1.0), IRON, 100.0)
    b = w.loose((20, 20, 20), (0.0, 0.0, -1.0), IRON, 100.0)
    for f in (a, b):
        f["born"] = w.tick - 1000             # long in the air, and both men
        f["_seen"] = {p["pid"]: (w.tick - 1000, True),   # saw it long ago:
                      q["pid"]: (w.tick - 1000, True)}   # the catch gate
                                              # counts from the sighting (R3)
    assert w._catch(p, b, "right arm"), "the second one in the air is caught"
    assert len(w.bodies) == 1 and w.bodies[0] is a, "the other is still flying"
    assert w._catch(q, a, "right arm"), "and so is the first, the same tick"
    assert not w.bodies, "nothing is left in the air"
    assert p.get("held") and q.get("held"), "each has the one he caught"


# ── E1: the eid field, carried wherever matter moves ────────────────────────
BAR, PANE, POST, MAN = 1, 2, 3, 4


def _stamped():
    """A man, an iron bar at his feet, a glass pane high up with nothing under
    it and a wooden post with an arm that overhangs its foot. Each of the
    three is a registered thing, stamped through `fill` (`_stamping`). The
    man's flesh is stamped with an eid registered as a PERSON, which is not a
    thing (`_thing_lut` says so), as every body is (E7): it rides every move
    a body makes, and only a thing's move bumps `_thing_gen`. The man is
    registered after the three things, so their eids are 1, 2 and 3."""
    w = World(40, 14, 50, voxel_cm=5)
    w.stamp_bodies = False
    w.fill(0, 40, 0, 14, 0, 1, STONE)
    p = _person(w, 12, 7)
    p["name"] = "him"
    assert [w.register(n) for n in ("bar", "pane", "post")] == [BAR, PANE, POST]
    assert w._eid_of(p) == MAN and w.things[MAN]["kind"] == "person"
    w.stamp(MAN, w.mat == FLESH)
    for e, box, m in ((BAR, (17, 19, 7, 8, 1, 2), IRON),
                      (PANE, (31, 33, 3, 5, 36, 38), GLASS),
                      (POST, (27, 28, 10, 11, 1, 14), WOOD),
                      (POST, (28, 33, 10, 11, 13, 14), WOOD)):
        w._stamping = e
        w.fill(*box, m)
    w._stamping = 0
    return w, p


def _family(w, e):
    """Thing `e` and every thing the identity pass has made of it since: its
    pieces, their pieces, its debris."""
    fam, grew = {e}, True
    while grew:
        more = {k for k, t in w.things.items() if t["parent"] in fam} - fam
        fam |= more
        grew = bool(more)
    return sorted(fam)


def _grams(w, e, landed=None):
    """Every gram of thing `e` and its family: its cells on the lattice, in
    bodies in the air, and in the solver's particles (drawn cells are the
    particles'). A body that has just `landed` is on the lattice now, and
    no longer counts as in the air."""
    fam = _family(w, e) if e in w.things else [e]
    ents = w.matter.entities()
    return (float(w.smass[np.isin(w.eid, fam) & ~w.moving].sum())
            + sum(float(b["masses"][np.isin(b["eids"], fam)].sum())
                  for b in w.bodies if b is not landed)
            + sum(sum(ents[k]["mix"].values()) for k in fam if k in ents))


def test_the_EID_rides_with_the_matter_for_200_ticks_and_the_index_keeps_up():
    """`id_check` asks the entity layer for its own books every tick: no cell
    of air says which thing it is, the kept index of where things are
    (`_thing_cells`, rebuilt only when `_thing_gen` moves) is the one the eid
    field gives now, and no dead or unknown eid is on the lattice. This asks
    the same again at the end of each tick, after the laws that run later,
    and keeps a ledger by thing: every gram of each thing is still that
    thing's, or its pieces' (the identity pass names what breaks off), on the
    lattice, in the air or in the solver.

    The man takes the bar, swings it, reaches out, walks, throws it, crouches,
    turns and jumps; the pane falls and breaks on the floor; the post tips
    over into the solver. Between them they move cells through eleven of the
    fourteen carry sites (not `loose`, `_left_world` or `_leave`: each of
    the fourteen has its own test below)."""
    w, p = _stamped()
    w.id_check = True
    mass0 = {e: _grams(w, e) for e in (BAR, PANE, POST, MAN)}
    plan = {0: {"hands": "take", "legs": "stay"},
            25: {"hands": "swing", "legs": "stay"},
            60: {"hands": "reach", "legs": "stay"},
            80: {"hands": "pull_in", "legs": "go"},
            110: {"hands": "throw", "legs": "stay"},
            140: {"hands": "keep", "legs": "jump"}}
    seen = {"held": False, "thrown": False, "in the solver": False}
    for t in range(200):
        if t in plan:
            w.policy = _Wants(each={"him": plan[t]})
            p["goal"] = None                  # decide again: a walk keeps going
        if t == 125:
            assert w._crouch(p, 0.8) == "moved", "he bends his knees"
        if t == 132:
            assert w._turn(p, 1), "and turns a quarter on his feet"
        w.step()
        w._check_ids()
        seen["held"] |= bool(p.get("held"))
        seen["thrown"] |= any(b.get("fly") and (b["eids"] == BAR).any()
                              for b in w.bodies)
        seen["in the solver"] |= bool(np.isin(w.matter.mpm.eid,
                                              _family(w, POST)).any())
        for e, g in mass0.items():
            assert _grams(w, e) == pytest.approx(g, rel=1e-5), \
                f"t{t}: thing {e} is {_grams(w, e):.1f} g, it was {g:.1f} g"
    assert all(seen.values()), f"each journey was made: {seen}"
    assert (w.rubble & np.isin(w.eid, _family(w, PANE))).any(), \
        "the pane broke, and its pieces are its"
    assert all(t["dead"] is None for t in w.things.values()), \
        "and nothing left the world"


# ── E2: the registry, the identity pass, names in the menus ─────────────────
def _floor(nx, ny, nz, cm=10):
    w = World(nx, ny, nz, voxel_cm=cm)
    w.fill(0, nx, 0, ny, 0, 1, STONE)
    return w


def _make(w, name, box, m, **kw):
    """Register a thing and fill its box with it."""
    e = w.register(name, **kw)
    w._stamping = e
    w.fill(*box, m)
    w._stamping = 0
    return e


def _live(w):
    """Every live thing's name, said to nobody: {eid: name}."""
    return {k: w.name(k, None) for k, t in w.things.items()
            if t["dead"] is None and t["kind"] != "person"}


def test_a_world_that_names_nothing_reads_as_it_always_did():
    """The fallback, unchanged, for every eid that is not a registered thing:
    no thing at all, a person's flesh, an eid nobody registered. That is what
    keeps every menu and every line of text byte for byte what it was."""
    w = _floor(8, 8, 8)
    him = w.register("him", kind="person")
    for e in (0, him, 7, 10 ** 6):
        assert w.name(e, "the iron") == "the iron"
    bar = _make(w, "bar", (2, 4, 2, 3, 1, 2), IRON)
    assert w.name(bar, "the iron") == "the bar"
    assert w.entity_at(2, 2, 1) == bar and w.entity_at(5, 5, 5) == 0
    assert w.cells_of(bar).tolist() == [[2, 2, 1], [3, 2, 1]]
    assert w.cells_of(him).shape == (0, 3)


def test_a_BURNED_THROUGH_plank_is_the_plank_and_a_piece_of_the_plank():
    """Ash holds nothing together. A plank on two trestles, burning in the
    middle, is one thing until a whole cross-section of it is ash; from that
    tick it is two: the heavier half keeps the plank's eid and name, the
    other is a new thing called after it. Every gram is still one or the
    other, and the identity pass says so as one typed event."""
    w = _floor(30, 8, 12, cm=5)
    w.fill(4, 6, 3, 5, 1, 4, STONE)
    w.fill(20, 22, 3, 5, 1, 4, STONE)
    plank = _make(w, "plank", (3, 23, 3, 5, 4, 5), WOOD)
    w.thermostats = [(12, y, 4, 900.0) for y in (3, 4)]
    w.id_check = True
    g0 = _grams(w, plank)
    # a fire some minutes old (G4: a 5 cm plank takes ten minutes to burn
    # through): the middle is down to its last embers (ash below 1.9 g of
    # char), its neighbours alight
    w.smass[12, 3:5, 4] = 1.8
    for x in (11, 13):
        w.E[x, 3:5, 4] = 430.0 * w.heat_capacity()[x, 3:5, 4]
    for t in range(400):
        w.step()
        if len(_live(w)) > 1:
            break
    assert len(_live(w)) == 2, f"it burned through in {t} ticks: {_live(w)}"
    piece = max(w.things)
    assert w.things[plank]["called"] == "the plank"
    assert w.things[piece]["called"] == "a piece of the plank"
    assert w.things[piece]["parent"] == plank
    assert len(w.cells_of(plank)) > len(w.cells_of(piece)), "the heavier keeps it"
    # the state word goes in front, and here both halves are still alight
    assert w.name(plank, "?") == "the burning plank"
    assert w.name(piece, "?") == "a burning piece of the plank"
    assert _grams(w, plank) < g0, "it burned"
    assert float(w.smass[w.eid == plank].sum() + w.smass[w.eid == piece].sum()) \
        == pytest.approx(_grams(w, plank)), "and what is left is the two of them"
    ev = [e for e in w.events.rows if e.target == ("thing", plank)]
    assert [(e.verb, e.data) for e in ev] == [("splits", (("into", (piece,)),))]


def _held_axe():
    """A man holding an axe by its haft: an iron head on a wooden haft that
    runs out from his right fist. The grip's cell is out on the haft, past
    where it will break, which is as good a cell as any while the axe is one
    thing (it is where a hand first closed on it, wherever that was)."""
    w = World(40, 24, 44, voxel_cm=5)
    w.fill(0, 40, 0, 24, 0, 1, STONE)
    p = _person(w, 12, 12)
    p["name"] = "him"
    w.policy = _Wants(each={"him": {"hands": "keep", "legs": "stay"}})
    w.step()
    comp, sl = w._person_cells(p)
    own = np.argwhere(comp)
    own[:, 0] += sl[0].start
    own[:, 1] += sl[1].start
    fx, fy, fz = (int(v) for v in w._fist_of(p, "right arm", own))
    axe = w.register("axe", kind="tool", parts={"head": IRON, "haft": WOOD})
    w._stamping = axe
    w.fill(fx + 1, fx + 6, fy, fy + 1, fz, fz + 1, WOOD)   # the haft
    w.fill(fx + 6, fx + 7, fy, fy + 1, fz, fz + 1, IRON)   # the head
    w._stamping = 0
    p["held"] = {"cell": (fx + 5, fy, fz), "mat": int(WOOD), "label": "wood",
                 "arm": "right arm"}
    w._law_identity()
    return w, p, own, axe, (fx, fy, fz)


def test_a_HAFT_that_breaks_while_held_takes_the_grip_with_it():
    """The grip follows the piece in the hand. The haft snaps between the
    fist and the head: the head end is heavier and stays the axe ("the broken
    axe"), the end in the fist is the haft, by the part it was declared as.
    The hand was closed on a cell of the far end, and now holds the near one,
    and the menu says so."""
    w, p, own, axe, (fx, fy, fz) = _held_axe()
    keep = w._menu(p, own, None, "hands", {})
    assert keep[0]["key"] == "keep hold of the axe"
    assert {"swing the axe", "throw the axe"} <= {o["key"] for o in keep}
    snap = (fx + 3, fy, fz)
    m = np.zeros(w.shape, bool)
    m[snap] = True
    w._unmake(m, "dissolves")                 # the acid eats through it
    w._law_identity()
    haft = w._held_eid(p)
    assert haft != axe and w.things[haft]["parent"] == axe
    assert w.name(haft, "?", p) == "the haft of the axe"
    assert w.name(axe, "?") == "the broken axe"
    assert p["held"]["cell"] in {(fx + 1, fy, fz), (fx + 2, fy, fz)}, \
        f"the grip moved to the end in the fist: {p['held']}"
    rows = [o["key"] for o in w._menu(p, own, None, "hands", {})]
    assert rows[0] == "keep hold of the haft of the axe"
    assert "swing the haft of the axe" in rows and "throw the haft of the axe" in rows
    assert sorted(map(tuple, w._held_cells(p).tolist())) == \
        [(fx + 1, fy, fz), (fx + 2, fy, fz)], "and the hand holds just that"


def test_a_SHATTERED_bottle_leaves_one_heap_of_pieces():
    """Shards hold nothing together either. A bottle dropped from high up is
    all shards on the floor: nothing of it is whole, so it does not come
    apart into a hundred things, it is one thing, "the pieces of the bottle".
    A pane dropped flat breaks only where it lands: what is still whole is
    "the broken pane", and the shards that fell clear of it go to its ONE
    debris child, whichever tick they fell."""
    w = _floor(24, 24, 50, cm=5)
    _make(w, "bottle", (11, 13, 11, 13, 30, 33), GLASS)
    w.id_check = True
    for _ in range(60):
        w.step()
    assert list(_live(w).values()) == ["the pieces of the bottle"]

    w = _floor(24, 24, 30, cm=5)
    pane = _make(w, "pane", (9, 15, 9, 15, 4, 5), GLASS)
    w.id_check = True
    for _ in range(60):
        w.step()
    names = sorted(_live(w).values())
    assert names == ["the broken pane", "the pieces of the pane"], names
    heap = w.things[pane]["debris"]
    assert w.things[heap]["kind"] == "debris" and w.rubble[w.eid == heap].all()


def test_melted_TIN_dies_MELTED():
    """The census: a thing that is nowhere is dead, of what took its last
    matter. `_unmake` writes the cause down as it empties the cells."""
    w = _floor(6, 6, 6)
    tin = _make(w, "spoon", (2, 3, 2, 3, 1, 2), TIN)
    w.E[2, 2, 1] = 1e9                        # far past what melts it all
    w._law_melt()
    assert w.mat[2, 2, 1] == AIR and w.fl[2, 2, 1] == MTIN, "it melted"
    w._law_identity()
    assert w.things[tin]["dead"] == "melted" and w.things[tin]["died"] == w.tick
    ev = [e for e in w.events.rows if e.target == ("thing", tin)]
    assert [(e.verb, e.result) for e in ev] == [("melts", "melted")]
    w._law_identity()
    assert len([e for e in w.events.rows if e.target == ("thing", tin)]) == 1, \
        "and a thing dies once"


def _edge_bar():
    """A short iron bar standing on end at the very edge of the world, tipped
    outward: it goes over the edge as it lands, and nothing of it ever comes
    back down onto the lattice."""
    w = _floor(12, 6, 16)
    bar = _make(w, "bar", (11, 12, 3, 4, 1, 3), IRON)
    w.id_check = True
    w.step()
    w._topple(np.argwhere(w.mat == IRON), 0, 1, 11.5, 0.5, owner="nobody")
    return w, bar


def test_a_bar_that_goes_OFF_THE_EDGE_dies_though_the_lattice_did_not_change():
    """Nothing on the lattice shows that a thing left the world: its cells
    were in a body in the air, and the body landed past the edge. The loss
    `_left_world` writes down is all the evidence there is, and the census
    reads it whether or not a cell changed that tick. (A thing in FLIGHT
    meets the edge as a wall and comes down inside the world; it is a body
    turning as it lands, a topple or a swing, that can put cells past it.)"""
    w, bar = _edge_bar()
    for _ in range(40):
        was = w.mat.copy()
        w.step()
        if w.things[bar]["dead"]:
            break
    assert w.things[bar]["dead"] == "gone"
    assert np.array_equal(was, w.mat), "the lattice did not change that tick"
    assert w.gone.get(IRON, 0.0) > 0.0 and not w.bodies
    ev = [e for e in w.events.rows if e.target == ("thing", bar)]
    assert [(e.verb, e.result) for e in ev] == [("gone", "gone")]


def test_a_HALF_PROMOTED_thing_is_two_pieces():
    """A piece in the solver is a bound cluster of particles; a piece on the
    lattice is joined cells. A plank of which one end went into the solver is
    in two pieces, and two things: the four cells on the floor keep the
    plank, and the two cells' particles are relabelled as the new piece,
    which is named but cannot be taken (nothing reaches into the solver)."""
    w = _floor(20, 12, 16, cm=5)
    plank = _make(w, "plank", (6, 12, 5, 6, 1, 2), WOOD)
    w.id_check = True
    g0 = _grams(w, plank)
    w.matter.promote(np.argwhere(w.mat == WOOD)[:2])
    w.step()
    kids = [k for k, t in w.things.items() if t["parent"] == plank]
    assert len(kids) == 1, w.things
    piece = kids[0]
    assert w.things[piece]["called"] == "a piece of the plank"
    assert set(np.unique(w.matter.mpm.eid).tolist()) == {piece}
    assert len(w.cells_of(plank)) == 4 and len(w.cells_of(piece)) == 0
    assert _grams(w, plank) == pytest.approx(g0, rel=1e-6)
    w.step()
    assert w.name(piece, "?") == "a piece of the plank", "and drawn, it is named"
    assert (w.eid[w.moving] == piece).all()


def test_the_PER_CELL_T_and_BURNING_are_the_whole_world_ones():
    """`T_at` and `burning_at` read a handful of cells without building the
    whole world's heat capacity, and must agree with `T()` and `burning()`
    to the bit, whatever is in the cells: stone, burning wood, char, ash,
    water, air."""
    w = _floor(30, 8, 12, cm=5)
    _make(w, "plank", (3, 23, 3, 5, 1, 2), WOOD)
    w.pour(3, 3, 2, WATER, 40.0)
    w.thermostats = [(12, y, 1, 900.0) for y in (3, 4)]
    w.smass[12, 3, 1] = 13.6             # a fire some minutes old: this cell is
                                         # down to just above the char line (13.5 g)
    for _ in range(300):
        w.step()
        if (w.mat == CHAR).any() and w.burning().any():
            break
    assert w.burning().any() and (w.mat == CHAR).any(), "the scene has a fire"
    T, fire = w.T(), w.burning()
    every = np.arange(w.mat.size)
    assert np.array_equal(w.T_at(every), T.ravel())
    assert np.array_equal(w.burning_at(every), fire.ravel())
    some = np.nonzero(w.mat != STONE)
    assert np.array_equal(w.T_at(some), T[some])
    assert np.array_equal(w.burning_at(some), fire[some])


def test_a_SNAPSHOT_brings_the_registry_back_and_the_same_split_happens_again():
    """The registry is plain dicts and the field is an array, so a snapshot
    holds them and a restore puts them back: the same break after the
    restore makes the same pieces under the same eids. Eids are never
    reused within a run, and a rewind is not a different run."""
    w = _floor(20, 6, 8)
    plank = _make(w, "plank", (4, 14, 2, 3, 1, 2), WOOD)
    w.step()
    snap = w.snapshot()
    before = (copy.deepcopy(w.things), w.eid.copy(), w._next_eid)

    def snap_it():
        m = np.zeros(w.shape, bool)
        m[8, 2, 1] = True
        w._unmake(m, "dissolves")
        w.step()
        return copy.deepcopy(w.things), w.eid.copy(), w._next_eid

    after = snap_it()
    assert len(after[0]) == 2 and after[0][plank]["broken"]
    w.restore(snap)
    assert w.things == before[0] and np.array_equal(w.eid, before[1])
    assert w._next_eid == before[2]
    again = snap_it()
    assert again[0] == after[0] and np.array_equal(again[1], after[1])
    assert again[2] == after[2]


def test_two_LOOKALIKES_are_told_apart_from_where_the_body_stands():
    """Two things called the same get words relative to the body ("on your
    left"), never metres; still the same, they are counted off nearest
    first. The menu's take rows stay one per thing, and unique."""
    w = World(40, 24, 44, voxel_cm=5)
    w.fill(0, 40, 0, 24, 0, 1, STONE)
    p = _person(w, 12, 12)
    p["name"] = "him"
    p["facing"] = (1.0, 0.0)
    left = _make(w, "bottle", (15, 16, 18, 19, 1, 3), GLASS)
    right = _make(w, "bottle", (15, 16, 6, 7, 1, 3), GLASS)
    far = _make(w, "bottle", (16, 17, 22, 23, 1, 3), GLASS)
    w.policy = _Wants(each={"him": {"hands": "keep", "legs": "stay"}})
    w.step()
    assert w.name(left, "?") == "the bottle", "said to no one, a bottle"
    assert w.name(right, "?", p) == "the bottle on your right"
    assert w.name(left, "?", p) == "the first bottle on your left"
    assert w.name(far, "?", p) == "the second bottle on your left"
    comp, sl = w._person_cells(p)
    own = np.argwhere(comp)
    own[:, 0] += sl[0].start
    own[:, 1] += sl[1].start
    keys = [o["key"] for o in w._menu(p, own, None, "hands", {})
            if o["tag"] == "take"]
    assert keys and len(keys) == len(set(keys)), keys
    assert all(k.startswith("take hold of the ") and "bottle" in k for k in keys)


# ── E2: an identity ledger at each of the fourteen carry sites ───────────────
# Each site that moves matter carries `eid` with it. Each is checked here the
# moment it returns, before the identity pass could tidy anything up: no cell
# of air says which thing it is, and every gram of the thing is still the
# thing's (on the lattice, in the air, in the solver). The sites only a body
# goes through (turn, crouch, walk, leave) are checked on a man whose flesh
# carries his own eid. A site that drops the eid fails its own test: step E2
# of docs/sim-mind-design.md checks that by running each test against a copy
# of the tree with the eid taken out of that one site.
def _no_stray(w):
    stray = (w.mat == AIR) & (w.eid != 0)
    assert not stray.any(), f"air that says it is a thing at {np.argwhere(stray)[:3].tolist()}"


def _books(w, e, g0, landed=None):
    _no_stray(w)
    g = _grams(w, e, landed)
    assert g == pytest.approx(g0, rel=1e-5), f"thing {e} is {g:.1f} g, it was {g0:.1f} g"


def _spy(w, name, check):
    """Run `check(result, *args)` every time the world's method `name`
    returns, and count the calls."""
    real, calls = getattr(w, name), []

    def spied(*a, **k):
        out = real(*a, **k)
        calls.append(out)
        check(out, *a)
        return out
    setattr(w, name, spied)
    return calls


def _body(w, x, y):
    """A man whose flesh is stamped with his own eid (a person's: not a
    thing), so a body-only site shows whether his flesh kept it."""
    p = _person(w, x, y)
    p["name"] = "him"
    man = p["eid"]             # registered as a person, stamped by the scene
    assert w.things[man]["kind"] == "person"
    assert (w.eid[w.mat == FLESH] == man).all()
    n = int((w.eid == man).sum())

    def flesh_books(*_):
        _no_stray(w)
        assert (w.eid[w.mat == FLESH] == man).all(), "all of him is still his"
        assert int((w.eid == man).sum()) + sum(
            int((b["eids"] == man).sum()) for b in w.bodies) == n
    return p, man, flesh_books


def _site_settle():
    w = _floor(10, 10, 12)
    bar = _make(w, "bar", (4, 6, 4, 5, 8, 9), IRON)
    g0 = _grams(w, bar)
    calls = _spy(w, "_settle", lambda *_: _books(w, bar, g0))
    for _ in range(40):
        w.step()
        if w.cells_of(bar)[:, 2].tolist() == [1, 1]:
            break
    assert calls and w.cells_of(bar)[:, 2].tolist() == [1, 1], "it came down"


def _site_shatter():
    w = _floor(8, 8, 6)
    pane = _make(w, "pane", (4, 5, 4, 5, 1, 2), GLASS)
    g0 = _grams(w, pane)
    w._shatter(4, 4, 1)
    assert (w.rubble & (w.mat == GLASS)).any(), "it broke"
    _books(w, pane, g0)


def _site_topple():
    w = _floor(16, 6, 16)
    post = _make(w, "post", (5, 6, 3, 4, 1, 9), IRON)
    g0 = _grams(w, post)
    w._topple(np.argwhere(w.mat == IRON), 0, 1, 5.5, 0.5, owner="nobody")
    assert len(w.bodies) == 1, "it is going over as a body"
    _books(w, post, g0)
    for _ in range(40):
        w.step()
    assert not w.bodies and len(w.cells_of(post)) == 8, "and it came down whole"
    _books(w, post, g0)


def _site_launch():
    w = _floor(20, 8, 16)
    bar = _make(w, "bar", (5, 8, 4, 5, 1, 2), IRON)
    g0 = _grams(w, bar)
    b = w._launch(np.argwhere(w.mat == IRON), (2.0, 0.0, 3.0))
    assert (b["eids"] == bar).all()
    _books(w, bar, g0)


def _site_land():
    w = _floor(20, 8, 16)
    bar = _make(w, "bar", (5, 8, 4, 5, 1, 2), IRON)
    g0 = _grams(w, bar)
    w._launch(np.argwhere(w.mat == IRON), (2.0, 0.0, 3.0))
    calls = _spy(w, "_land_body", lambda _o, b, *_: _books(w, bar, g0, landed=b))
    for _ in range(30):
        w.step()
    assert calls and not w.bodies and len(w.cells_of(bar)) == 3


def _site_left_world():
    w, bar = _edge_bar()

    def check(*_):
        assert (bar, "gone") in w._id_losses, "the loss of a thing is written down"
    calls = _spy(w, "_left_world", check)
    for _ in range(40):
        w.step()
    assert calls and w.things[bar]["dead"] == "gone"


def _site_loose():
    w = _floor(20, 8, 16)
    b = w.loose((4, 4, 6), (3.0, 0.0, 1.0), IRON, 100.0)
    assert (b["eids"] == 0).all(), "a bolt off a string is no thing yet"
    w.id_check = True
    for _ in range(30):
        w.step()
    assert not w.bodies, "it came down"
    _no_stray(w)
    assert not w.eid.any() and not w.things


def _armed_stamped():
    """`_armed`, with the blade a registered thing and the man's flesh his."""
    w = World(64, 20, 40, voxel_cm=5)
    w.fill(0, 64, 0, 20, 0, 1, STONE)
    p, man, flesh_books = _body(w, 17, 10)
    blade = _make(w, "blade", (22, 24, 10, 11, 1, 2), IRON)
    w.policy = _Wants(each={"him": {"hands": "take", "legs": "stay"}})
    for _ in range(12):
        w.step()
    assert p.get("held") and w._held_eid(p) == blade
    return w, p, blade, flesh_books


def _site_repose():
    w, p, blade, flesh_books = _armed_stamped()
    g0 = _grams(w, blade)
    assert w._repose(p, p["held"]["arm"], 0.9) == "moved"
    _books(w, blade, g0)
    flesh_books()


def _site_release():
    w, p, blade, flesh_books = _armed_stamped()
    g0 = _grams(w, blade)

    def check(fl, *_):
        assert (fl["eids"][fl["mats"] != FLESH] == blade).all(), "it flies as itself"
        assert all(len(fl[k]) == len(fl["cells"]) for k in AS_BODY), \
            "with every row the hand had (E8)"
        _books(w, blade, g0)
    calls = _spy(w, "_release", check)
    w.policy = _Wants(each={"him": {"hands": "throw", "legs": "stay"}})
    for _ in range(40):
        w.step()
        if calls:
            break
    assert calls, "he let it fly"


def _site_turn():
    w = _floor(40, 24, 44, cm=5)
    p, man, flesh_books = _body(w, 12, 12)
    w.policy = _Wants()
    w.step()
    assert w._turn(p, 1), "he turns a quarter on his feet"
    flesh_books()


def _site_crouch():
    w = _floor(40, 24, 44, cm=5)
    p, man, flesh_books = _body(w, 12, 12)
    w.policy = _Wants()
    w.step()
    assert w._crouch(p, 0.8) == "moved", "he bends his knees"
    flesh_books()


def _site_walk():
    w = _floor(40, 24, 44, cm=5)
    w.step_up_m = w.step_down_m = 0.0      # the flat legs: `_walk` (lane C9 made the range 30 cm by default)
    p, man, flesh_books = _body(w, 12, 12)
    w.policy = _Wants(each={"him": {"legs": "go"}})
    calls = _spy(w, "_walk", flesh_books)
    for _ in range(20):
        w.step()
    assert any(calls), "he walked"


def _site_stride():
    # lane C9: with the default step range every step goes through the stepper
    # `_step_to_k` (pathlaw), which moves flesh with the same _lift and _put.
    w = _floor(40, 24, 44, cm=5)
    p, man, flesh_books = _body(w, 12, 12)
    w.policy = _Wants(each={"him": {"legs": "go"}})
    calls = _spy(w, "_step_to_k", flesh_books)
    for _ in range(20):
        w.step()
    assert any(c is not None for c in calls), "he walked"


def _site_leave():
    w = _floor(40, 24, 44, cm=5)
    p, man, flesh_books = _body(w, 12, 12)
    w.policy = _Wants()
    w.step()
    comp, sl = w._person_cells(p)
    own = np.argwhere(comp)
    own[:, 0] += sl[0].start
    own[:, 1] += sl[1].start
    w._leave(p, own)
    _no_stray(w)
    assert not (w.eid == man).any() and (man, "gone") in w._id_losses


def _site_shove():
    w = _floor(12, 8, 8)
    crate = _make(w, "crate", (4, 6, 3, 5, 1, 3), WOOD)
    g0 = _grams(w, crate)
    assert w._shove(w.cells_of(crate).astype(np.int32), 1, 0, 0)
    _books(w, crate, g0)
    assert w.cells_of(crate)[:, 0].min() == 5, "and it moved"


SITES = {"settle": _site_settle, "shatter": _site_shatter,
         "topple": _site_topple, "launch": _site_launch,
         "turn": _site_turn, "repose": _site_repose, "crouch": _site_crouch,
         "release": _site_release, "loose": _site_loose, "land": _site_land,
         "left_world": _site_left_world, "leave": _site_leave,
         "shove": _site_shove, "walk": _site_walk,
         "stride": _site_stride}


# the sites that move matter on the lattice: the method, and whether it lifts
# matter off and puts matter down (E8). The other three move it only between
# free bodies, or off the edge of the world.
_ON_LATTICE = {"settle": ("_settle", True, True),
               "shatter": ("_shatter", True, False),
               "topple": ("_topple", True, False),
               "launch": ("_launch", True, False),
               "turn": ("_turn", True, True), "repose": ("_repose", True, True),
               "crouch": ("_crouch", True, True),
               "land": ("_land_body", False, True),
               "leave": ("_leave", True, False), "shove": ("_shove", True, True),
               "walk": ("_walk", True, True),
               "stride": ("_step_to_k", True, True)}


@pytest.mark.parametrize("site", sorted(SITES))
def test_each_CARRY_SITE_keeps_the_books(site, monkeypatch):
    """And since E8, each site that moves matter on the lattice does it
    through `_lift` and `_put`, the one list of what a cell carries: counted
    only while the site's own method is running."""
    inside, went = [0], set()
    if site in _ON_LATTICE:
        meth, lifts, puts = _ON_LATTICE[site]
        real_site = getattr(World, meth)

        def in_site(self, *a, **k):
            inside[0] += 1
            try:
                return real_site(self, *a, **k)
            finally:
                inside[0] -= 1
        monkeypatch.setattr(World, meth, in_site)
        for name in ("_lift", "_put"):
            real = getattr(World, name)

            def spied(self, *a, real=real, name=name):
                if inside[0]:
                    went.add(name)
                return real(self, *a)
            monkeypatch.setattr(World, name, spied)
    SITES[site]()
    if site in _ON_LATTICE:
        assert ("_lift" in went) == lifts, f"{meth} lifts: {went}"
        assert ("_put" in went) == puts, f"{meth} puts: {went}"


# ── E8: one CARRIED tuple ───────────────────────────────────────────────────
def test_CARRIED_is_ONE_list_and_a_free_body_keeps_all_of_it():
    """The design's list, in its order; a free body keeps a row of every
    field, and a thrown thing takes every row it had in the hand."""
    assert CARRIED == ("mat", "smass", "E", "fl", "fvol", "fpot", "edge",
                       "fallh", "rubble", "eid")
    w = _floor(20, 8, 16)
    w.fill(5, 8, 4, 5, 1, 2, IRON)
    w.rubble[6, 4, 1] = True
    w.fallh[7, 4, 1] = 0.25
    b = w._launch(np.argwhere(w.mat == IRON), (2.0, 0.0, 3.0))
    assert all(len(b[k]) == 3 for k in AS_BODY)
    assert b["rubbles"].tolist() == [False, True, False]
    assert b["fallhs"].tolist() == [0.0, 0.0, 0.25]
    assert not w.rubble.any() and not w.fallh.any(), "and the cells are empty"


def test_a_SHOVED_SHARD_is_still_a_SHARD():
    """`rubble` was not in `_shove`'s list: a broken piece pushed one cell
    arrived as whole glass, which spans and holds up what rests on it."""
    w = _floor(12, 8, 8)
    w.fill(4, 5, 4, 5, 1, 2, GLASS)
    w.rubble[4, 4, 1] = True
    assert w._shove(np.array([[4, 4, 1]], np.int32), 1, 0, 0)
    assert w.mat[5, 4, 1] == GLASS and w.rubble[5, 4, 1], "it is still a piece"
    assert not w.rubble[4, 4, 1], "and the air it left is not"


def test_a_STEP_takes_ALL_that_its_cells_carry():
    """`_walk` left `fallh` and `rubble` where they were: a shard a man
    walked through was swapped behind him as whole glass, and the drop
    height his feet carried stayed in the air he stepped out of."""
    w = _floor(40, 24, 44, cm=5)
    p = _person(w, 12, 12)
    w.policy = _Wants()
    w.step()
    comp, sl = w._person_cells(p)
    cells = w._claim_cells(p, comp, sl)
    w.fallh[tuple(cells.T)] = 0.25
    own = {tuple(c) for c in cells.tolist()}
    front = next(c for c in sorted(own, key=lambda c: (-c[0], c[2]))
                 if (c[0] + 1, c[1], c[2]) not in own and c[2] == 1)
    shard = (front[0] + 1, front[1], 1)
    w.fill(shard[0], shard[0] + 1, shard[1], shard[1] + 1, 1, 2, GLASS, frac=0.1)
    w.rubble[shard] = True
    g = float(w.smass[shard])
    assert w._walk(cells, 1, 0), "he steps through it"
    glass = np.argwhere(w.mat == GLASS)
    assert len(glass) == 1 and tuple(glass[0]) != shard, "it went behind him"
    assert w.rubble[tuple(glass[0])] and int(w.rubble.sum()) == 1
    assert float(w.smass[tuple(glass[0])]) == g
    flesh = w.mat == FLESH
    assert (w.fallh[flesh] == np.float32(0.25)).all(), "his drop height came"
    assert not w.fallh[~flesh].any(), "and none stayed behind"


def test_a_WEDGED_TOPPLE_puts_the_DROP_HEIGHT_and_the_PIECES_back():
    """The wedged restore put back what it had lifted, except `fallh`,
    which it had zeroed: a thing half-way down a fall, propped for a tick,
    forgot how far it had fallen."""
    w = World(12, 6, 16)
    w.fill(0, 12, 0, 6, 0, 1, STONE)
    w.fill(6, 7, 0, 6, 1, 14, STONE)          # the wall it would lean into
    w.fill(5, 6, 3, 4, 1, 13, IRON)
    cells = np.argwhere(w.mat == IRON)
    w.fallh[tuple(cells.T)] = 0.25
    w.rubble[5, 3, 4] = True
    before = (w.fallh.copy(), w.rubble.copy())
    w._topple(cells, 0, 1, 5.5, 0.5, owner="nobody")
    assert not w.bodies, "it is wedged: nothing left the lattice"
    assert np.array_equal(w.fallh, before[0]), "it remembers how far it fell"
    assert np.array_equal(w.rubble, before[1])


# ── E3: holding by eid, and authoring ───────────────────────────────────────
class _Saw(_Wants):
    """`_Wants`, keeping every hands row each body was offered, by tick."""

    def __init__(self, w, each):
        super().__init__(each=each)
        self.w, self.rows = w, {}

    def pick(self, situation, menu):
        if situation["limb"] == "hands":
            self.rows.setdefault(situation["who"], []).append(
                (self.w.tick, [o["key"] for o in menu]))
        return super().pick(situation, menu)


def _cells_of_body(w, p):
    comp, sl = w._person_cells(p)
    own = np.argwhere(comp)
    own[:, 0] += sl[0].start
    own[:, 1] += sl[1].start
    return own


def test_a_NAMED_AXE_is_held_by_WHICH_THING_it_is_from_the_take_to_the_catch():
    """An iron head on a wooden haft, made with `thing()`, is one thing in
    reach (one take row, by its name), and a hand that takes it holds it by
    its eid: through thirty ticks of a reach that carries it out and back, a
    swing that takes it into the air with the arm and puts it down again, and
    a throw; and the hand that catches it holds the same eid. The grip's cell
    and material come from the same voxel all the way, and every gram of the
    axe is still the axe at the end.

    The axe is two voxels thick on purpose. A haft one voxel thin, thrown at
    an angle, is drawn back onto the lattice as a staircase whose cells touch
    only at their corners, and the identity pass then (rightly, by what the
    lattice holds) finds it in two pieces."""
    w = World(90, 14, 50, voxel_cm=5)
    w.open_sky = False
    w.fill(0, 90, 0, 14, 0, 50, STONE)
    w.mat[1:89, 1:13, 1:46] = AIR
    w.smass[1:89, 1:13, 1:46] = 0.0
    w.exits = []
    a = _person(w, 10, 7)
    a["name"], a["facing"], a["strength_N"] = "thrower", (1.0, 0.0), 3000.0
    b = _person(w, 44, 7)
    b["name"], b["facing"], b["strength_N"] = "catcher", (-1.0, 0.0), 3000.0
    b["ready"] = True                     # he stands waiting for it (R3)
    with thing(w, "axe", kind="tool", parts={"head": IRON, "haft": WOOD},
               says={"affords": ["cut", "chop"]}) as axe:
        w.fill(14, 17, 7, 8, 1, 3, WOOD)                    # the haft
        w.fill(17, 18, 7, 8, 1, 3, IRON, edge=BLADE)        # the head
    g0 = _grams(w, axe)
    assert g0 == pytest.approx(w.things[axe]["mass0"])
    w.id_check = True
    want = {"thrower": {"hands": "take", "legs": "stay", "waist": "stand"},
            "catcher": {"hands": "catch", "legs": "stay", "waist": "stand"}}
    w.policy = saw = _Saw(w, want)
    said, held, caught, where, grip_at = [], {}, {}, {}, {}
    for t in range(160):
        if t == 2:
            want["thrower"]["hands"] = "keep"
        if t == 30:                           # the holding arm goes out ...
            a.setdefault("reach", {})[a["held"]["arm"]] = float(np.pi / 2)
        if t == 60:                           # ... and comes back in
            a["reach"].pop(a["held"]["arm"])
        if t == 79:
            # BY THE TOP OF THE HEAD NOW. A grip can be on any cell of a
            # named thing, and the swing has to bring it back to the same
            # one. Put on whichever cell of the axe came down first (all a
            # grip on an unnamed thing can ask for), it would point at the
            # wood along the bottom and say iron.
            at = w.cells_of(axe)
            at = at[w.mat[tuple(at.T)] == IRON]
            head = tuple(int(v) for v in at[at[:, 2].argmax()])
            a["held"].update(cell=head, mat=int(IRON), label="iron")
        if t == 80:
            want["thrower"]["hands"] = "swing"
        if t == 82:
            want["thrower"]["hands"] = "keep"
        if t == 120:
            want["thrower"]["hands"] = "throw"
        w.step()
        for p in (a, b):
            said += [e.split(": ", 1)[1] for e in p["events"]]
            p["events"].clear()
        for p, got in ((a, held), (b, caught)):
            h = p.get("held")
            if h:
                got[t] = h["eid"]
                if p is a:
                    grip_at[t] = (tuple(int(v) for v in h["cell"]), h["mat"])
                c = tuple(h["cell"])
                if w.eid[c] == axe:           # on the lattice, not in the air
                    assert int(w.mat[c]) == h["mat"], \
                        f"t{t}: the grip's cell and material are one voxel"
                    assert h["name"] == "the axe"
        at = w.cells_of(axe)
        where[t] = at.mean(axis=0) if len(at) else None
    first, then = saw.rows["thrower"][0], next(
        keys for t, keys in saw.rows["thrower"] if t > 0)
    takes = [k for k in first[1] if k.startswith("take hold of")]
    assert first[0] == 0 and takes == ["take hold of the axe"], \
        f"one thing, one row: {first}"
    assert {"swing the axe", "throw the axe"} <= set(then), then
    assert set(held.values()) == {axe} and set(caught.values()) == {axe}
    assert all(t in held for t in range(0, 120)), \
        f"held every tick from the take to the throw: {sorted(set(range(120)) - set(held))}"
    assert np.abs(where[45] - where[29]).max() >= 4, \
        f"the reach carried it out: {where[29]} -> {where[45]}"
    assert np.abs(where[75] - where[29]).max() <= 1, "and back"
    assert any(where[t] is None for t in range(80, 92)), "the swing took it up"
    assert where[100] is not None and 100 in held, "and it came down in the hand"
    assert grip_at[100] == (head, int(IRON)), "held by the head still"
    assert "thrower throws the axe" in said and "thrower lets fly" in said
    assert a.get("held") is None, "the thrower's hand is empty"
    assert "catcher catches the axe" in said, said
    assert any("catch the axe" in keys for _t, keys in saw.rows["catcher"])
    assert [k for k, t in w.things.items() if t["kind"] != "person"] == [axe], \
        "and it is still one thing"
    assert _grams(w, axe) == pytest.approx(g0, rel=1e-5), "every gram of it"


def test_REACH_joins_cells_on_the_KEY():
    """A named thing is one thing whatever it is made of, and only itself:
    an unnamed iron bar laid against the axe's iron head is not the axe, and
    is still a thing of its own on the menu."""
    w = World(40, 24, 44, voxel_cm=5)
    w.fill(0, 40, 0, 24, 0, 1, STONE)
    p = _person(w, 12, 12)
    p["name"] = "him"
    with thing(w, "axe", kind="tool", parts={"head": IRON, "haft": WOOD}) as axe:
        w.fill(17, 20, 12, 13, 1, 2, WOOD)
        w.fill(20, 21, 12, 13, 1, 2, IRON)
    w.fill(21, 23, 12, 13, 1, 2, IRON)                  # nobody named this
    w.policy = _Wants(each={"him": {"hands": "keep", "legs": "stay"}})
    w.step()
    seen = w._objects_within_reach(p, _cells_of_body(w, p))
    assert sorted(w.entity_at(*o["cell"]) for o in seen) == [0, axe]
    for o in seen:
        assert o["mat"] == int(w.mat[o["cell"]]), "cell and material, one voxel"
    whole = {tuple(c) for c in w._object_at(17, 12, 1).tolist()}
    assert whole == {tuple(c) for c in w.cells_of(axe).tolist()}, \
        "from the haft, the whole axe, head and all, and nothing else"
    assert {tuple(c) for c in w._object_at(22, 12, 1).tolist()} == \
        {(21, 12, 1), (22, 12, 1)}, "from the bar, the bar"
    keys = [o["key"] for o in w._menu(p, _cells_of_body(w, p), None, "hands", {})
            if o["tag"] == "take"]
    assert sorted(keys) == ["take hold of the axe", "take hold of the iron"]


def test_two_NAMED_BOTTLES_are_said_from_where_the_body_stands():
    """Two bottles made with `thing()` are called the same, so each is told
    apart by where it is from the body that is asked, on the menu as in the
    names."""
    w = World(40, 24, 44, voxel_cm=5)
    w.fill(0, 40, 0, 24, 0, 1, STONE)
    p = _person(w, 12, 12)
    p["name"], p["facing"] = "him", (1.0, 0.0)
    with thing(w, "bottle") as left:
        w.fill(15, 16, 18, 19, 1, 3, GLASS)
    with thing(w, "bottle") as right:
        w.fill(15, 16, 6, 7, 1, 3, GLASS)
    w.policy = _Wants(each={"him": {"hands": "keep", "legs": "stay"}})
    w.step()
    assert w.name(left, "?") == w.name(right, "?") == "the bottle"
    assert w.name(left, "?", p) == "the bottle on your left"
    assert w.name(right, "?", p) == "the bottle on your right"
    keys = [o["key"] for o in w._menu(p, _cells_of_body(w, p), None, "hands", {})
            if o["tag"] == "take"]
    assert sorted(keys) == ["take hold of the bottle on your left",
                            "take hold of the bottle on your right"]


def test_THING_stamps_what_its_block_made_and_AUDITS_what_it_says():
    """`thing()` stamps every cell whose matter the block changed, whatever
    wrote it (`fill` or a direct write), leaves a thing made inside it its
    own cells, notes what it weighed, and checks each word of `says` against
    the thing's own cells: an author's "cut" on a club with no edge goes to
    `id_audit`, and is not a capability."""
    w = _floor(20, 8, 8)
    with thing(w, "cart", says={"affords": ["burn", "carry"]}) as cart:
        w.fill(2, 6, 2, 3, 1, 2, WOOD)
        w.mat[6, 2, 1] = WOOD                         # a direct write
        w.smass[6, 2, 1] = w.smass[2, 2, 1]
        with thing(w, "wheel") as wheel:
            w.fill(2, 3, 3, 4, 1, 2, WOOD)
    with thing(w, "club", says={"affords": ["cut", "chop"]}) as club:
        w.fill(10, 13, 2, 3, 1, 2, WOOD)
    with thing(w, "knife", says={"affords": ["cut"]}) as knife:
        w.fill(15, 16, 2, 3, 1, 2, IRON, edge=BLADE)
    assert len(w.cells_of(cart)) == 5 and (w.eid[6, 2, 1] == cart)
    assert w.cells_of(wheel).tolist() == [[2, 3, 1]], "the wheel is its own"
    assert w.things[cart]["mass0"] == pytest.approx(
        float(w.smass[w.eid == cart].sum()))
    assert w.things[cart]["affords"] == ["burn"]
    assert w.things[club]["affords"] == [] and w.things[knife]["affords"] == ["cut"]
    assert [(r["eid"], r["says"], r["but"]) for r in w.id_audit] == [
        (cart, "carry", "unchecked"), (club, "cut", "not sharp"),
        (club, "chop", "not sharp")]
    assert w.things[club]["says"] == {"affords": ["cut", "chop"]}, \
        "what the author said is kept, as a claim"


def test_a_BODY_carries_its_EID():
    """`stamp_bodies` (on by default since E7, where a body IS its eid): a
    person is registered, not as a thing, and the flesh the scene builds is
    stamped with that eid. Switched off, nothing is stamped when the body is
    built, and the first claim registers the person and adopts all of it.
    Flesh filled in by hand before `add_person` is adopted by the first
    claim on it, and flesh someone else already stamped is not."""
    w = _floor(40, 24, 44, cm=5)
    w.stamp_bodies = False
    p = _person(w, 12, 12)
    assert "eid" not in p and not w.eid.any(), "off: nothing is stamped"
    w.step()
    assert p["eid"] and (w.eid[w.mat == FLESH] == p["eid"]).all(), \
        "until the first claim adopts him"

    w = _floor(40, 24, 44, cm=5)
    assert w.stamp_bodies, "on by default"
    p = _person(w, 12, 12)
    man = p["eid"]
    assert w.things[man]["kind"] == "person" and not w._is_thing([man])[0]
    assert (w.eid[w.mat == FLESH] == man).all(), "all of him, from the start"
    assert w.name(man, "him") == "him" and w.entity_at(*np.argwhere(
        w.mat == FLESH)[0]) == 0, "a body is not a thing"

    w = _floor(24, 24, 20)
    w.fill(4, 5, 4, 5, 1, 15, FLESH, frac=0.9)          # by hand, as the
    w.fill(5, 6, 4, 5, 1, 15, FLESH, frac=0.9)          # older tests build one
    other = w.register("someone", kind="person")
    w.stamp(other, np.array([[5, 4, 14]]))              # already somebody's
    q = w.add_person(4, 4)
    assert not (w.eid == q["eid"]).any(), "nothing of him is stamped yet"
    w.step()
    flesh = w.mat == FLESH
    assert int((w.eid[flesh] == q["eid"]).sum()) == int(flesh.sum()) - 1
    assert int((w.eid[flesh] == other).sum()) == 1, \
        "the claim adopts, it does not overwrite"


# ── E4: properties, reach, adopt-on-take, torque ────────────────────────────
def test_PROPS_say_what_a_thing_IS_now():
    """`props(eid)` is read off the thing's own matter. An axe whose head has
    an edge of 2e-4 is sharp (that edge), weighs what its cells weigh, and a
    man can lift it; a club an author called a cutter has no edge, and the
    audit says so. Kept per (tick, `_thing_gen`): asked twice in a tick it
    is worked out once, and a move works it out again."""
    w = _floor(30, 12, 20, cm=5)
    p = _person(w, 6, 6)
    with thing(w, "axe", kind="tool", parts={"head": IRON, "haft": WOOD},
               says={"affords": ["cut"]}) as axe:
        w.fill(12, 16, 6, 7, 1, 2, WOOD)
        w.fill(16, 17, 6, 7, 1, 2, IRON, edge=BLADE)
    with thing(w, "club", says={"affords": ["cut"]}) as club:
        w.fill(12, 16, 9, 10, 1, 2, WOOD)
    got = w.props(axe)
    assert got["sharp"] == pytest.approx(BLADE)
    assert got["mass_g"] == pytest.approx(float(w.smass[w.eid == axe].sum()))
    assert got["integrity"] == pytest.approx(1.0)
    assert not got["burning"] and not got["broken"] and not got["in_solver"]
    assert got["hot_c"] == pytest.approx(20.0, abs=1e-3)
    assert got["contains"] == {} and got["can_hold_ml"] == 0.0
    assert got["held_by"] == ()
    assert w.liftable_by(axe, p) and not w.liftable_by(0, p)
    assert w.props(club)["sharp"] == 0.0
    assert w.things[club]["affords"] == [] and w.things[axe]["affords"] == ["cut"]
    assert [(r["eid"], r["says"], r["but"]) for r in w.id_audit] == [
        (club, "cut", "not sharp")], "a cut on a blunt thing is flagged"
    assert w.props(0) is None and w.props(10 ** 6) is None
    # CACHED: the second ask is the first answer; a move asks again
    w._props[1][axe]["mass_g"] = -1.0
    assert w.props(axe)["mass_g"] == -1.0, "kept within the tick"
    w._shove(w.cells_of(axe), 0, 1)
    assert w.props(axe)["mass_g"] > 0, "and worked out again once it moved"


def test_a_FILLED_VAT_contains_what_is_in_it_and_says_what_it_holds():
    """A stone vat with water standing in it: `contains` names the water and
    how much, and `can_hold_ml` is the whole hollow (the particle seam's
    `_cavity`, asked for every walled-in cell), so a full vat still says it
    can hold something."""
    w = _floor(16, 16, 12, cm=5)
    with thing(w, "vat") as vat:
        w.fill(4, 9, 4, 9, 1, 2, STONE)                  # the bottom
        w.fill(4, 9, 4, 5, 2, 5, STONE)                  # four walls
        w.fill(4, 9, 8, 9, 2, 5, STONE)
        w.fill(4, 5, 5, 8, 2, 5, STONE)
        w.fill(8, 9, 5, 8, 2, 5, STONE)
    for x in range(5, 8):
        for y in range(5, 8):
            w.pour(x, y, 2, WATER, w.cap)
    got = w.props(vat)
    assert got["contains"] == {"water": pytest.approx(9 * w.cap)}
    assert got["can_hold_ml"] == pytest.approx(27 * w.cap), "3 x 3 x 3 inside"
    assert got["wet_ml"] == 0.0, "the stone itself is dry"


def test_a_BURNING_PLANK_is_the_burning_plank():
    """Heat a plank past where wood catches: its name and its props both say
    it is burning, and how hot."""
    w = _floor(20, 8, 10, cm=5)
    with thing(w, "plank") as plank:
        w.fill(4, 14, 3, 5, 1, 2, WOOD)
    at = tuple(w.cells_of(plank).T)
    w.E[at] = (400.0 - 20.0) * w.heat_capacity()[at]
    assert w.name(plank, "the wood") == "the burning plank"
    got = w.props(plank)
    assert got["burning"] and got["hot_c"] == pytest.approx(400.0, rel=1e-4)


def _reach_row(w, p):
    return w._objects_within_reach(p, _cells_of_body(w, p))


def test_TWO_LIKE_THINGS_on_opposite_faces_of_the_reach_box_are_two_things():
    """The reach box used to wrap round its faces, so an iron bar at the
    furthest a hand goes on one side and another at the furthest on the
    other side were one lump to it, and only one was ever on the menu.
    Both are there now, each with how far it is."""
    w = _floor(60, 24, 44, cm=5)
    p = _person(w, 30, 12)
    p["name"] = "him"
    w.policy = _Wants(each={"him": {"hands": "keep", "legs": "stay"}})
    w.step()
    own = _cells_of_body(w, p)
    R = max(int(round(w._reach_of(p) / (0.1 * w.scale))), 1)
    # level with the body's furthest cell each way (a hand), at the very
    # edge of reach, in the same row and at the same height, held up by
    # nothing: reach is asked before anything falls
    a, b = own[own[:, 0].argmin()], own[own[:, 0].argmax()]
    y, z = int(a[1]), int(a[2])
    xa, xb = int(a[0]) - R, int(b[0]) + R
    w.fill(xa, xa + 1, y, y + 1, z, z + 1, IRON)
    w.fill(xb, xb + 1, y, y + 1, z, z + 1, IRON)
    seen = _reach_row(w, p)
    assert sorted(o["cell"] for o in seen) == [(xa, y, z), (xb, y, z)]
    assert [o["away"] for o in seen] == [R, R]
    # and they are on the box's two faces: wrapped, the two were one
    sl = (slice(xa, xb + 1), slice(y, y + 1), slice(z, z + 1))
    lab = label(w.mat[sl] == IRON, w.mat[sl], canon="min", wrap=True)
    assert lab[0, 0, 0] == lab[-1, 0, 0] == 0


def test_REACH_finds_EVERY_thing_unless_told_to_stop():
    """Three bars within reach are three rows, each with how far it is; the
    old stop at two is `reach_things_cap`."""
    w = _floor(60, 24, 44, cm=5)
    p = _person(w, 30, 12)
    p["name"] = "him"
    w.policy = _Wants(each={"him": {"hands": "keep", "legs": "stay"}})
    w.step()
    own = _cells_of_body(w, p)
    x1 = int(own[:, 0].max())
    for dx, y in ((3, 4), (1, 12), (5, 18)):
        w.fill(x1 + dx, x1 + dx + 1, y, y + 1, 1, 2, IRON)
    w.reach_things_cap = None
    seen = _reach_row(w, p)
    assert len(seen) == 3
    for o in seen:                    # the nearest cell of the body, in cells
        assert o["away"] == int(np.abs(own - np.array(o["cell"])).max(axis=1).min())
    w.reach_things_cap = 2
    assert len(_reach_row(w, p)) == 2


def test_a_TAKEN_thing_is_a_THING_and_only_itself():
    """ADOPT-ON-TAKE. An unnamed iron bar is taken: from that act it is a
    thing ("the iron", so every line says what it said), the grip holds it
    by its eid, and iron that comes to touch it later is not it."""
    w = _floor(40, 20, 40, cm=5)
    p = _person(w, 17, 10)
    p["name"] = "him"
    w.fill(22, 24, 10, 11, 1, 2, IRON)
    w.policy = _Wants(each={"him": {"hands": "take", "legs": "stay"}})
    said = []
    for _ in range(3):
        w.step()
        said += [e.split(": ", 1)[1] for e in p["events"]]
        p["events"].clear()
    assert p.get("held"), "he took it"
    e = p["held"].get("eid")
    assert e and w.things[e]["called"] == "the iron"
    assert "him takes up the iron" in said
    bar = {tuple(c) for c in w.cells_of(e).tolist()}
    assert len(bar) == 2 and w.things[e]["mass0"] == pytest.approx(
        float(w.smass[w.eid == e].sum()))
    # more iron, touching it: not the bar
    x, y, z = max(bar)
    w.fill(x, x + 1, y + 1, y + 2, z, z + 1, IRON)
    assert {tuple(c) for c in w._held_cells(p).tolist()} == bar
    assert {tuple(c) for c in w._object_at(x, y + 1, z).tolist()} == \
        {(x, y + 1, z)}


def test_OBJECT_AT_floods_a_box_and_finds_what_the_world_flood_finds():
    """`_object_at` in a world with things in it floods a box round the cell
    (growing it until the flood is inside it): the same cells, in the same
    order, as a flood of the whole world's key, from a stick to the floor."""
    w = _floor(80, 60, 20, cm=5)
    with thing(w, "axe", parts={"head": IRON, "haft": WOOD}):
        w.fill(10, 30, 10, 11, 1, 2, WOOD)
        w.fill(30, 31, 10, 11, 1, 2, IRON)
    w.fill(31, 34, 10, 11, 1, 2, IRON)
    w.fill(40, 41, 40, 41, 1, 15, WOOD)
    key = w._key()
    for c in ((10, 10, 1), (30, 10, 1), (33, 10, 1), (40, 40, 7), (5, 5, 0),
              (79, 59, 0)):
        want = flood(key, *c, 4000, block=w.moving)
        got = w._object_at(*c)
        assert np.array_equal(got, want), c


def test_a_NAMED_AXE_tips_as_ONE_thing_and_an_unnamed_world_labels_as_before():
    """The tip check joins cells on the key. An axe standing on its wooden
    haft with its iron head out to one side tips over as one lump, head and
    all; the same matter unnamed is a wooden post and an iron lump, and the
    post stands. In a world that names nothing the key handed to `label` is
    `mat` itself, so every label is what it was."""
    def build(named):
        w = _floor(30, 12, 20, cm=5)
        ctx = thing(w, "axe", parts={"head": IRON, "haft": WOOD}) if named \
            else contextlib.nullcontext()
        with ctx:
            w.fill(10, 11, 6, 7, 1, 9, WOOD)
            w.fill(11, 14, 6, 7, 7, 9, IRON)
        calls = []
        real = w._topple
        w._topple = lambda cells, *a, **k: (
            calls.append({tuple(c) for c in np.asarray(cells).tolist()}),
            real(cells, *a, **k))[1]
        return w, calls

    w, calls = build(True)
    whole = {tuple(c) for c in np.argwhere(w.mat != AIR).tolist() if c[2] > 0}
    w.step()
    assert calls == [whole], "the whole axe, haft and head, in one topple"
    w, calls = build(False)
    keys = []
    real = w._key_at
    w._key_at = lambda idx: (keys.append((idx, real(idx))), keys[-1][1])[1]
    for _ in range(6):
        w.step()
    assert calls == [], "unnamed, the post stands and the iron hangs on"
    assert keys and all(np.array_equal(k, w.mat.reshape(-1)[i])
                        for i, k in keys), "the key is the material"


# ── wave 6b: what a hand may make a thing of ────────────────────────────────
def test_ADOPT_makes_a_thing_only_of_what_could_be_LIFTED():
    """A grip may close on what it cannot move, and a small world's floor,
    or a laid table, is well under the flood cap: adopted, the floor was
    "the stone" from then on. Only a lump that is not the ground, carries
    nothing, and could be lifted becomes a thing."""
    w = _floor(20, 20, 10)
    n0 = len(w.things)
    assert w._adopt(5, 5, 0, "stone") == 0, "the floor is the world"
    w.fill(2, 6, 2, 4, 3, 4, WOOD)            # a table top ...
    w.fill(3, 4, 2, 3, 4, 5, GLASS)           # ... with a cup on it
    assert w._adopt(2, 2, 3, "wood") == 0, "a laid table carries the cup"
    w.fill(12, 18, 12, 18, 1, 5, IRON)        # over a tonne of iron
    assert w._adopt(12, 12, 1, "iron") == 0, "nobody could lift it"
    held = {"cell": (5, 5, 0), "mat": STONE, "label": "stone"}
    assert w._grip_learns(held) == 0 and "eid" not in held, \
        "a grip on the floor stays a plain grip"
    assert len(w.things) == n0 and not w.eid.any(), "nothing was adopted"
    w.fill(3, 4, 2, 3, 4, 5, AIR)             # the cup taken off
    e = w._adopt(2, 2, 3, "wood")
    assert e and len(w.cells_of(e)) == 8, "the bare table top can be lifted"
    w.fill(12, 14, 2, 3, 1, 2, IRON)          # a bar a man can lift
    e = w._adopt(12, 2, 1, "iron")
    assert e and len(w.cells_of(e)) == 2
    # THE GROUND even when it is light: a floor of sixteen small cells
    tiny = _floor(4, 4, 4, cm=5)
    assert tiny._effort(tiny._object_at(0, 0, 0))[0] < 400.0
    assert tiny._adopt(0, 0, 0, "stone") == 0


def test_a_CATCH_closes_on_what_LANDED_not_on_the_table_it_landed_on():
    """The catch of an unnamed thing took the first cell of the flight pose
    that was not air once it landed. A stone caught over a table piles up
    onto the table, and that first cell was the table: the hand held the
    furniture, and looking at the grip made the table a thing."""
    w = World(40, 30, 44, voxel_cm=5)
    w.fill(0, 40, 0, 30, 0, 1, STONE)
    p = _person(w, 12, 10)
    p["name"] = "him"
    w.policy = _Wants()
    w.step()
    w.fill(18, 23, 8, 13, 6, 7, WOOD)         # a table top
    b = w.loose((20, 10, 6), (0.0, 0.0, -1.0), IRON, 100.0)  # in it
    b["born"] = w.tick - 1000
    b["_seen"] = {p["pid"]: (w.tick - 1000, True)}   # seen long ago (R3)
    assert w._catch(p, b, "right arm")
    held = p["held"]
    assert held["mat"] == IRON and held["cell"] == (20, 10, 7), held
    w._held_cells(p)                           # the grip learns WHICH thing
    e = p["held"].get("eid")
    assert e and w.cells_of(e).tolist() == [[20, 10, 7]]
    assert not w.eid[18:23, 8:13, 6].any(), "the table is no thing"


def test_a_WOODEN_thing_caught_onto_a_WOODEN_table_is_the_thing_alone():
    """The grip closed on what landed, but the first look adopted the lump
    it was in, and unnamed matter is a lump with all of its own material it
    touches: wood caught onto a wooden table became one thing with the
    table. A catch knows what came out of the air."""
    w = World(40, 30, 44, voxel_cm=5)
    w.fill(0, 40, 0, 30, 0, 1, STONE)
    p = _person(w, 12, 10)
    p["name"] = "him"
    w.policy = _Wants()
    w.step()
    w.fill(18, 23, 8, 13, 6, 7, WOOD)         # a wooden table top
    b = w.loose((20, 10, 6), (0.0, 0.0, -1.0), WOOD, 100.0)
    b["born"] = w.tick - 1000
    b["_seen"] = {p["pid"]: (w.tick - 1000, True)}   # seen long ago (R3)
    assert w._catch(p, b, "right arm")
    assert p["held"]["cell"] == (20, 10, 7), p["held"]
    assert w._held_cells(p) is not None
    e = p["held"].get("eid")
    assert e and w.cells_of(e).tolist() == [[20, 10, 7]]
    assert not w.eid[18:23, 8:13, 6].any(), "the table is no thing"


def test_a_thing_that_LEAVES_THE_REACH_BOX_and_comes_back_is_ONE_row():
    """A U of iron whose two prongs reach into the box a hand reaches into,
    joined outside it: two pieces in the box, each flooded to the same
    thing. It was two rows, so one bar could be offered twice."""
    w = _floor(60, 24, 44, cm=5)
    p = _person(w, 30, 12)
    p["name"] = "him"
    w.policy = _Wants(each={"him": {"hands": "keep", "legs": "stay"}})
    w.step()
    own = _cells_of_body(w, p)
    R = max(int(round(w._reach_of(p) / (0.1 * w.scale))), 1)
    a = own[own[:, 0].argmin()]
    y, z = int(a[1]), int(a[2])
    xa = int(a[0]) - R                        # the box's face
    w.fill(xa - 2, xa + 1, y, y + 1, z, z + 1, IRON)
    w.fill(xa - 2, xa + 1, y + 2, y + 3, z, z + 1, IRON)
    w.fill(xa - 2, xa - 1, y, y + 3, z, z + 1, IRON)   # joined outside it
    seen = _reach_row(w, p)
    assert len(seen) == 1, seen
    rows = [o for o in w._menu(p, own, None, "hands", {}, cap=10 ** 6)
            if o["tag"] == "take"]
    assert len(rows) == 1, [o["key"] for o in rows]
