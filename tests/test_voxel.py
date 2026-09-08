"""The voxel core's promises — outcomes that must FALL OUT of the four laws,
with zero case-code anywhere (Ruling 1). If one of these breaks, a law is wrong,
not a flag missing."""
import numpy as np
import pytest

from src.voxel.demo import build, dump_water, torch
from src.voxel.sim import (ACID, AIR, ASH, BODY, CHAR, FLESH, GLASS, IRON, LEAF,
                           MIRON, MTIN, O2_PER_L, OIL, STONE, TIN, WATER,
                           WEAK_ACID, WOOD, World)


def _lit_stick(ticks):
    w = build()
    for t in range(ticks):
        if t < 25:
            torch(w, 8, 3, 1)
        w.step()
    return w


def test_fluid_mass_is_conserved_while_flowing():
    w = World(12, 5, 8)
    w.fill(0, 12, 0, 5, 0, 1, STONE)
    for x in range(4, 8):
        w.pour(x, 2, 5, WATER, 900.0)         # a column of water dropped from mid-air
    total0 = w.total_fluid(WATER)
    for _ in range(60):
        w.step()
    assert abs(w.total_fluid(WATER) - total0) < 1.0, "flow must neither create nor destroy"
    zs = np.argwhere(w.fvol > 1)[:, 2]
    assert zs.max() <= 1, "water must have fallen and pooled at the floor"


def test_energy_is_conserved_by_pure_conduction():
    w = World(8, 8, 4)
    w.E[4, 4, 1] = 5000.0
    e0 = float(w.E.sum())
    w._law_conduct()                          # one law in isolation: transport only moves E
    assert abs(float(w.E.sum()) - e0) < 1.0


def test_boiling_pins_water_near_100():
    w = World(4, 4, 4)
    w.fill(0, 4, 0, 4, 0, 1, STONE)
    w.pour(1, 1, 1, WATER, 800.0)
    w.E[1, 1, 1] = 1.5e6                      # absurd energy dumped into the puddle
    for _ in range(3):
        w.step()
    assert w.T()[1, 1, 1] < 130, "excess energy must BOIL water away, not superheat it"


def test_torch_lights_dry_stick_and_fire_spreads():
    w = _lit_stick(300)
    assert int(w.burning().sum()) >= 3, "the stick must catch and creep"
    assert w.total_wood() < 2000, "burning must consume real mass"
    burned_state = np.isin(w.mat[8:10, 3, 1], (CHAR, ASH))
    assert burned_state.any(), \
        "burned wood must become CHAR (embers, black forever) and at last ASH"
    assert w.smoke.sum() > 50, "the burned mass must LEAVE as smoke — conservation"


def test_wet_wood_does_not_light():
    w = build()
    for x in range(8, 20):
        w.pour(x, 3, 2, WATER, 900.0)         # soak the stick's whole length first
    for t in range(80):
        if t < 25:
            torch(w, 8, 3, 1)
        w.step()
    assert int(w.burning().sum()) == 0, "water pins the wood near 100 °C — under ignition"


def test_water_quenches_wood_fire():
    w = _lit_stick(200)
    assert int(w.burning().sum()) >= 3, "precondition: the stick is burning"
    dump_water(w, 8, 20)
    for _ in range(140):
        w.step()
    wood_burning = w.burning() & (w.mat == WOOD)
    assert int(wood_burning.sum()) == 0, "wood needs 300 °C — a real dousing gets below it"
    assert w.total_wood() > 900, "the quench must actually SAVE most of the wood"


def test_water_fails_against_oil_fire():
    w = _lit_stick(400)
    oil_burning = w.burning() & (w.fl == OIL)
    assert int(oil_burning.sum()) >= 3, "precondition: the pool is alight"
    dump_water(w, 20, 27)                     # the same dousing that kills a wood fire
    for _ in range(60):
        w.step()
    assert int((w.burning() & (w.fl == OIL)).sum()) >= 3, \
        "oil needs only 250 °C — the water boils off before it can cool that far"
    assert w.total_fluid(WATER) < 21000, "some of the water must have boiled away trying"


def test_unsupported_solids_fall_and_supported_stay():
    w = World(10, 10, 12)
    w.fill(0, 10, 0, 10, 0, 1, STONE)
    w.fill(2, 6, 2, 6, 6, 7, WOOD)              # a slab floating in mid-air
    w.fill(7, 8, 7, 8, 1, 5, WOOD)              # a pillar standing on the floor
    w.fill(6, 9, 6, 9, 5, 6, WOOD)              # a shelf resting ON the pillar
    for _ in range(40):                         # falls ACCELERATE from rest now, so
        w.step()                                # a drop that used to take one tick a
    for _ in range(0):                          # voxel takes a few to get going
        w.step()
    assert (w.mat[2:6, 2:6, 1] == WOOD).all(), "the floating slab must land on the floor"
    assert (w.mat[2:6, 2:6, 6] == AIR).all(), "and no longer hang in the air"
    assert (w.mat[6:9, 6:9, 5] == WOOD).all(), "the shelf on the pillar must NOT fall"


def test_burning_away_the_legs_drops_the_shelf():
    w = World(8, 8, 10)
    w.fill(0, 8, 0, 8, 0, 1, STONE)
    w.fill(3, 4, 3, 4, 1, 4, WOOD)              # one leg
    w.fill(2, 6, 2, 6, 4, 5, WOOD)              # tabletop on it
    w.mat[3, 3, 1:4] = AIR                      # the leg burns away (as combustion does)
    w.smass[3, 3, 1:4] = 0.0
    for _ in range(40):                         # see above: falling has a SPEED now
        w.step()
    assert (w.mat[2:6, 2:6, 1] == WOOD).all(), "the orphaned top must fall to the floor"


def test_full_water_seal_smothers_burning_wood():
    def scene(sealed):
        w = World(5, 5, 5)
        w.fill(0, 5, 0, 5, 0, 3, STONE)         # a stone block...
        w.mat[2, 2, 1] = WOOD                   # ...with the hot wood set inside it
        w.smass[2, 2, 1] = 600.0
        w.mat[2, 2, 2] = AIR                    # one breathing hole above
        w.smass[2, 2, 2] = 0.0
        w.E[2, 2, 1] = 8.0e5                    # well above ignition
        if sealed:
            w.pour(2, 2, 2, WATER, 1000.0)      # the hole drowned — walls hold the water
        m0 = float(w.smass[2, 2, 1])
        for _ in range(3):
            w.step()
        return m0 - float(w.smass[2, 2, 1])
    assert scene(sealed=False) > 1.0, "in open air the hot wood must burn"
    assert scene(sealed=True) < 0.01, "sealed under water: no gas contact, NO burning"


def _flame_on_pool(ml, power):
    w = World(8, 8, 8)
    w.fill(0, 8, 0, 8, 0, 1, STONE)
    w.pour(3, 3, 1, OIL, ml)
    lit = 0
    for t in range(80):
        if t < 40:
            w.E[3, 3, 1] += power
        w.step()
        lit = max(lit, int((w.burning() & (w.fl == OIL)).sum()))
    return lit


def test_a_pool_is_not_a_plume():
    """Found by the oil-lamp scenario: the rise law treated liquid-bearing voxels
    as gas and blew the pool's heat skyward — NO surface flame, however strong,
    could ever light oil under open air. A torch must be able to."""
    assert _flame_on_pool(30.0, 6000.0) >= 1, \
        "a torch held to a shallow pour must light it in open air"


def test_a_small_taper_cannot_light_a_deep_cold_pool():
    """Flash-point behavior falls out of bulk heat capacity: a match-sized flame
    drowns in 400 ml of cold oil and the pool never reaches 250 °C."""
    assert _flame_on_pool(400.0, 800.0) == 0


def test_water_sinks_beneath_oil():
    w = World(3, 3, 6)
    w.fill(0, 3, 0, 3, 0, 1, STONE)
    w.pour(1, 1, 1, OIL, 900.0)
    w.pour(1, 1, 3, WATER, 900.0)               # dumped on top of the oil
    for _ in range(10):
        w.step()
    assert w.fl[1, 1, 1] == WATER, "the heavier water must end up on the bottom"
    assert w.fl[1, 1, 2] == OIL, "and the oil must float up above it"


def test_sealed_room_fire_suffocates_with_fuel_left():
    w = World(8, 8, 8)
    w.open_sky = False                          # a SEALED stone room, no fresh air
    w.fill(0, 8, 0, 8, 0, 1, STONE)
    w.fill(2, 6, 2, 6, 1, 2, WOOD)              # plenty of fuel
    w.E[3, 3, 1] = 9.0e5
    w.E[4, 4, 1] = 9.0e5
    for _ in range(200):
        w.step()
    assert w.total_wood() > 1000, "the fire must die from lack of AIR, not lack of wood"
    burned = 4 * 4 * 600.0 - w.total_wood()
    assert burned > 20, "but it must have burned SOMETHING before the air ran out"
    assert float(w.o2.sum()) < 0.28 * 8 * 8 * 8 * 0.2, "the room's oxygen must be spent"


def test_a_plank_resting_on_a_shelf_cannot_act_as_a_girder():
    """The ratchet bug: full-span-restore-on-carry let any two-layer slab support
    itself forever (a stick lying on the round-4 table let ONE leg hold 89% of the
    tabletop). Slack must only DECAY along a path. The pillar is CENTERED so the
    COM stays over it — this test isolates the span break from the tipping law."""
    w = World(36, 8, 8)
    w.fill(0, 36, 0, 8, 0, 1, STONE)
    w.fill(16, 20, 3, 5, 1, 4, WOOD)            # a central pillar
    w.fill(1, 35, 3, 5, 4, 5, WOOD)             # a LONG shelf balanced on it
    w.fill(1, 35, 3, 5, 5, 6, WOOD)             # and a plank resting along the shelf
    for _ in range(10):
        w.step()
    assert (w.mat[2, 3:5, 4:6] == AIR).all() and (w.mat[33, 3:5, 4:6] == AIR).all(), \
        "both far ends are 13+ hops out — 12-span wood must BREAK, plank or no plank"
    assert (w.mat[8, 3:5, 4:6] == WOOD).all() and (w.mat[27, 3:5, 4:6] == WOOD).all(), \
        "the reachable middle holds"


def _two_chambers(doorway):
    """A sealed stone shell split into two chambers; hot wood in chamber A.
    doorway=True carves a hole in the dividing wall."""
    w = World(22, 8, 8)
    w.open_sky = False
    w.fill(0, 22, 0, 8, 0, 8, STONE)            # solid block...
    w.mat[1:10, 1:7, 1:7] = AIR                 # ...with chamber A carved out
    w.smass[1:10, 1:7, 1:7] = 0.0
    w.mat[12:21, 1:7, 1:7] = AIR                # ...and chamber B
    w.smass[12:21, 1:7, 1:7] = 0.0
    if doorway:
        w.mat[10:12, 3:5, 1:5] = AIR
        w.smass[10:12, 3:5, 1:5] = 0.0
    w.fill(3, 5, 3, 5, 1, 2, WOOD)              # fuel in A
    w.E[3, 3, 1] = 9.0e5
    w.E[4, 4, 1] = 9.0e5
    for _ in range(150):
        w.step()
    o2_b = float(w.o2[12:21, 1:7, 1:7].sum())
    fresh_b = 0.28 * 9 * 6 * 6
    return w, o2_b / fresh_b


def test_fire_cannot_breathe_through_a_sealed_wall():
    w, b_frac = _two_chambers(doorway=False)
    assert w.total_wood() > 1000, "chamber A's fire must suffocate with fuel left"
    assert b_frac > 0.8, "chamber B's air is BEHIND A WALL — the fire must not drink it"


def test_a_doorway_makes_two_rooms_one_lung():
    w_sealed, _ = _two_chambers(doorway=False)
    w_open, b_frac = _two_chambers(doorway=True)
    assert b_frac < 0.6, "with a doorway, the fire drinks chamber B's air too"
    assert w_open.total_wood() < w_sealed.total_wood(), \
        "twice the air to breathe must burn MORE wood"


def test_acids_differ_by_table_row_not_by_code():
    """'What kind of acid is it?' — a REACTIONS row. Vitriol devours what the
    weak stock only stings; nothing but data separates them."""
    def soak(fluid, ticks=60):
        w = World(6, 6, 6)
        w.fill(0, 6, 0, 6, 0, 1, STONE)
        w.fill(2, 4, 2, 4, 1, 2, WOOD)
        for x in range(2, 4):
            for y in range(2, 4):
                w.pour(x, y, 2, fluid, 900.0)
        m0 = float(w.smass[2:4, 2:4, 1].sum())
        for _ in range(ticks):
            w.step()
        return m0 - float(w.smass[2:4, 2:4, 1].sum())
    strong, weak = soak(ACID), soak(WEAK_ACID)
    assert strong > 5 * weak > 0, "same law, different rows: tenfold-ish gap"


def _bath(material):
    """A stone basin of acid with a block of `material` submerged in it."""
    w = World(8, 8, 8)
    w.fill(0, 8, 0, 8, 0, 1, STONE)
    w.fill(1, 7, 1, 7, 1, 4, STONE)             # basin walls
    w.mat[2:6, 2:6, 1:4] = AIR                  # basin hollow
    w.smass[2:6, 2:6, 1:4] = 0.0
    w.fill(3, 5, 3, 5, 1, 2, material)          # the block on the basin floor
    for x in range(2, 6):
        for y in range(2, 6):
            for z in range(1, 4):
                w.pour(x, y, z, ACID, 900.0)    # drown it
    m0 = float(w.smass[3:5, 3:5, 1].sum())
    a0 = w.total_fluid(ACID)
    for _ in range(120):
        w.step()
    return w, m0 - float(w.smass[3:5, 3:5, 1].sum()), a0


def test_acid_dissolves_wood_and_is_spent_doing_it():
    w, eaten, a0 = _bath(WOOD)
    assert eaten > 500, "the wood must dissolve away"
    am = w.fl == ACID
    spent = a0 - float((w.fpot[am] * w.fvol[am]).sum())
    assert spent > 100, "dissolving must CONSUME the acid's potency"
    assert abs(w.total_fluid(ACID) - a0) < 0.05 * a0, \
        "while the liquid itself stays behind as weakening sludge"
    assert w.smoke.sum() > 5, "fumes still venting at the end — the mass LEFT as gas"


def test_acid_barely_marks_stone():
    w, eaten, _ = _bath(STONE)
    assert eaten < 150, "stone shrugs off what devours wood — rate, not a flag"


def test_a_hinged_tree_topples_toward_the_notch():
    """The torque law: COM outside the support footprint tips the body about the
    footprint edge. A pole standing on a 1-voxel hinge at its edge cannot stand."""
    w = World(24, 8, 16)
    w.fill(0, 24, 0, 8, 0, 1, STONE)
    w.fill(4, 7, 3, 5, 1, 12, WOOD)             # a 3x2 pole, 11 tall
    w.mat[5:7, 3:5, 1] = AIR                    # notch: only the x=4 base column left
    w.smass[5:7, 3:5, 1] = 0.0
    for _ in range(80):                         # the topple is an ARC now — the body
        w.step()                                # leans, accelerates, lands, settles
    assert not w.bodies, "the body must have LANDED by now"
    zs = np.argwhere(w.mat == WOOD)[:, 2]
    assert int(zs.max()) <= 5, "the pole must TOPPLE, not stand on its hinge"
    xs = np.argwhere(w.mat == WOOD)[:, 0]
    assert int(xs.max()) >= 9, "and it must lie TOWARD the overhang, not sink in place"


def test_fresh_leaves_resist_fire_dry_leaves_flash():
    """The leaf idea, no flags: FRESH means the voxel HOLDS water. The moisture
    pins it near 100 °C until it boils off; a dry leaf catches almost at once."""
    def clump(fresh):
        w = World(8, 8, 8)
        w.fill(0, 8, 0, 8, 0, 1, STONE)
        w.fill(3, 5, 3, 5, 1, 3, LEAF, frac=0.4)
        if fresh:
            w.fl[3:5, 3:5, 1:3] = WATER
            w.fvol[3:5, 3:5, 1:3] = 0.08 * w.cap    # ~ the leaf's own weight in sap
        lit = None
        for t in range(120):
            if t < 40:
                w.E[3, 3, 1] += 4000.0              # the same torch held to both
            w.step()
            if lit is None and int((w.burning() & (w.mat == LEAF)).sum()) > 0:
                lit = t
        return lit
    dry, fresh = clump(False), clump(True)
    assert dry is not None and dry < 25, "dry leaves are TINDER"
    assert fresh is None or fresh > 3 * dry, "fresh leaves buy real time — the water pins them"


def test_the_canopy_rides_the_falling_trunk():
    """A LEAF canopy is a different material than its WOOD trunk — but its whole
    support stands on the trunk, so when the trunk topples the canopy RIDES the
    body instead of hovering where the tree used to be."""
    w = World(30, 10, 20)
    w.fill(0, 30, 0, 10, 0, 1, STONE)
    w.fill(4, 7, 4, 6, 1, 12, WOOD)                 # trunk
    w.fill(3, 8, 3, 7, 12, 15, LEAF, frac=0.4)      # canopy block on top
    w.mat[5:7, 4:6, 1] = AIR                        # notch: hinge at x=4
    w.smass[5:7, 4:6, 1] = 0.0
    leaf0 = float(w.smass[w.mat == LEAF].sum())
    for _ in range(400):                            # fall + rubble re-settling
        w.step()
    assert not w.bodies
    leaves = np.argwhere(w.mat == LEAF)
    assert float(w.smass[w.mat == LEAF].sum()) > 0.9 * leaf0, "no leaf mass lost"
    assert float(leaves[:, 0].mean()) > 9, "the canopy came DOWN AND OVER with the trunk"
    # the canopy was built at z=12..14, so anything below that came DOWN. The
    # heap it makes is a voxel taller than it used to be now that leaves fall
    # at a leaf's speed rather than a stone's, and settle on each other on the
    # way — which is what a heap of leaves does.
    assert int(leaves[:, 2].max()) <= 10, "and lies low — not hovering at tree height"


def test_a_four_legged_table_does_not_tip():
    w = World(16, 16, 10)
    w.fill(0, 16, 0, 16, 0, 1, STONE)
    for (lx, ly) in ((3, 3), (3, 11), (11, 3), (11, 11)):
        w.fill(lx, lx + 2, ly, ly + 2, 1, 5, WOOD)
    w.fill(2, 14, 2, 14, 5, 6, WOOD)            # top: COM sits inside the leg box
    before = w.mat.copy()
    for _ in range(8):
        w.step()
    assert (w.mat == before).all(), "a stable table must not move a single voxel"


def test_a_sealed_fire_builds_pressure_an_open_one_vents():
    """The pneumatic field: burning injects pressure (hot gas expands); a sealed
    room holds it, the open sky bleeds it away."""
    def room(sealed):
        w = World(12, 12, 12)
        w.open_sky = not sealed
        w.fill(0, 12, 0, 12, 0, 1, STONE)
        if sealed:
            w.fill(0, 12, 0, 12, 11, 12, STONE)
            for (x0, x1, y0, y1) in ((0, 1, 0, 12), (11, 12, 0, 12),
                                     (0, 12, 0, 1), (0, 12, 11, 12)):
                w.fill(x0, x1, y0, y1, 0, 12, STONE)
        w.fill(4, 8, 4, 8, 1, 2, WOOD)
        w.E[5, 5, 1] = 9.0e5
        for _ in range(60):
            w.step()
        return float(np.maximum(w.pcell, 0).sum())
    assert room(sealed=True) > 3 * room(sealed=False), \
        "pressure must ACCUMULATE only where it cannot escape"


def test_communicating_vessels_equalize():
    """Hydrostatic head: one connected body of water seeks ONE surface level,
    even through a pipe at the bottom — p = rho*g*h, not a special case."""
    w = World(20, 6, 14)
    w.fill(0, 20, 0, 6, 0, 1, STONE)
    w.fill(1, 7, 1, 5, 1, 13, STONE)            # tank A shell
    w.mat[2:6, 2:4, 1:13] = AIR
    w.smass[2:6, 2:4, 1:13] = 0.0
    w.fill(13, 19, 1, 5, 1, 13, STONE)          # tank B shell
    w.mat[14:18, 2:4, 1:13] = AIR
    w.smass[14:18, 2:4, 1:13] = 0.0
    w.fill(7, 13, 1, 5, 1, 4, STONE)            # the pipe...
    w.mat[5:15, 2:4, 1:3] = AIR                 # ...bored through both walls
    w.smass[5:15, 2:4, 1:3] = 0.0
    for x in range(2, 6):
        for y in range(2, 4):
            for z in range(1, 11):
                w.pour(x, y, z, WATER, w.cap)   # tank A filled to z10
    for _ in range(150):
        w.step()
    la = np.argwhere(w.fvol[2:6, 2:4, :] > 0.3 * w.cap)[:, 2].max()
    lb = np.argwhere(w.fvol[14:18, 2:4, :] > 0.3 * w.cap)[:, 2].max()
    assert abs(int(la) - int(lb)) <= 1, f"levels must MEET: A z={la}, B z={lb}"
    assert lb >= 3, "tank B must have genuinely RISEN, not just wet its floor"


def test_a_splash_throws_droplets_and_loses_nothing():
    """Displaced water becomes flying parcels — and every milliliter is still
    accounted for while airborne (conservation includes the drops in flight)."""
    w = World(14, 14, 14)
    w.fill(0, 14, 0, 14, 0, 1, STONE)
    w.fill(1, 13, 1, 13, 1, 5, STONE)
    w.mat[2:12, 2:12, 1:5] = AIR
    w.smass[2:12, 2:12, 1:5] = 0.0
    for x in range(2, 12):
        for y in range(2, 12):
            for z in range(1, 4):
                w.pour(x, y, z, WATER, w.cap)
    total0 = w.total_fluid(WATER)
    w.fill(6, 8, 6, 8, 9, 11, STONE)            # a rock dropped from above
    flew = 0
    for _ in range(30):
        w.step()
        flew = max(flew, len(w.drops))
    assert flew >= 2, "the impact must throw real droplets into the air"
    assert abs(w.total_fluid(WATER) - total0) < 2.0, "splashing must not destroy water"


def test_stone_sinks_in_a_pond_but_wood_floats():
    """Density decides entry: a solid denser than the pool below sinks into it
    (displacing the water upward); a lighter one rests afloat. No float flag."""
    w = World(10, 10, 12)
    w.fill(0, 10, 0, 10, 0, 1, STONE)
    w.fill(1, 9, 1, 9, 1, 5, STONE)             # basin
    w.mat[2:8, 2:8, 1:5] = AIR
    w.smass[2:8, 2:8, 1:5] = 0.0
    for x in range(2, 8):
        for y in range(2, 8):
            for z in range(1, 5):
                w.pour(x, y, z, WATER, w.cap)   # 40 cm of water
    w.fill(3, 4, 3, 4, 8, 9, WOOD)              # a log dropped in
    w.fill(6, 7, 6, 7, 8, 9, STONE)             # a rock dropped in
    for _ in range(20):
        w.step()
    assert w.mat[6, 6, 1] == STONE, "the rock must sink to the basin floor"
    assert w.fvol[6, 6, 2] > 0.3 * w.cap, "with the displaced water above it"
    zs = np.argwhere(w.mat[3, 3, :] == WOOD)
    assert int(zs[:, 0].min()) >= 4, "wood is lighter than water — it stays at the surface"


def test_ash_cannot_carry_what_wood_could():
    w = World(12, 12, 8)
    w.fill(0, 12, 0, 12, 0, 1, STONE)
    w.fill(5, 6, 5, 6, 1, 4, WOOD)              # one central leg
    w.fill(2, 9, 2, 9, 4, 5, WOOD)              # a wide wooden top: wood span holds it
    for _ in range(4):
        w.step()
    assert (w.mat[2, 2, 4] == WOOD), "wood carries a 3-voxel overhang easily"
    top = w.mat[:, :, 4] == WOOD                # now the same shape, but made of ASH
    w.mat[:, :, 4][top] = ASH
    for _ in range(40):                         # see above: falling has a SPEED now
        w.step()
    assert w.mat[2, 2, 4] == AIR and w.mat[2, 2, 1] == ASH, \
        "ash far from the leg must CRUMBLE and fall"
    assert w.mat[5, 5, 4] == ASH, "ash directly on the leg still sits there"


def test_acid_spends_its_potency_not_its_volume():
    """Vitriol working on wood self-quenches: the pool stays put and its
    STRENGTH drains — and a half-potent pour eats at roughly half speed,
    because the rate IS the potency (one law, no dilution flag)."""
    def bath(pot):
        w = World(10, 10, 8)
        w.fill(0, 10, 0, 10, 0, 1, STONE)
        w.fill(2, 8, 2, 8, 1, 4, STONE)         # a basin so nothing runs off
        w.mat[3:7, 3:7, 1:4] = AIR
        w.smass[3:7, 3:7, 1:4] = 0.0
        w.fill(3, 7, 3, 7, 1, 2, WOOD)          # wooden basin floor
        for x in range(3, 7):
            for y in range(3, 7):
                w.pour(x, y, 2, ACID, 400.0, pot=pot)
        a0, wood0 = w.total_fluid(ACID), w.total_wood()
        for _ in range(150):
            w.step()
        return w, a0, wood0 - w.total_wood()
    w, a0, eaten_full = bath(1.0)
    _, _, eaten_half = bath(0.5)
    assert eaten_full > 500.0, "the acid must actually eat"
    assert 0.3 * eaten_full < eaten_half < 0.7 * eaten_full, \
        "half the potency must mean roughly half the appetite"
    assert abs(w.total_fluid(ACID) - a0) < 0.05 * a0, \
        "the liquid itself remains behind as weakening sludge"
    touching = (w.fl == ACID) & (w.fvol > 1.0)
    touching[:, :, 3:] = False                  # the layer working the wood face
    assert touching.any() and float(w.fpot[touching].min()) < 0.85, \
        "the working layer must have SPENT real potency doing it"


def test_water_dilutes_acid_and_the_blend_pales():
    """Water into vitriol: one liquid, acid species, volume-averaged potency —
    the color change the eye sees is just this number falling."""
    w = World(8, 8, 8)
    w.fill(0, 8, 0, 8, 0, 1, STONE)
    w.pour(4, 4, 1, ACID, 300.0)
    w.pour(4, 4, 2, WATER, 600.0)               # thrice the water lands on top
    for _ in range(6):
        w.step()
    am = (w.fl == ACID) & (w.fvol > 1.0)
    assert am.any(), "the blend keeps the acid's species"
    pot = float((w.fpot[am] * w.fvol[am]).sum() / w.fvol[am].sum())
    assert 0.15 < pot < 0.55, f"potency must dilute toward 1/3, got {pot:.2f}"
    assert not ((w.fl == WATER) & (w.fvol > 1.0)).any(), \
        "no separate water puddle survives in the same cells"


def test_person_faints_then_dies_in_a_sealed_airless_room():
    """Physiology, no fire needed: thin air at head height drains blood O2 —
    the person slumps, later stops breathing. Thresholds on state, no script."""
    w = World(24, 24, 20)
    w.open_sky = False
    w.fill(0, 24, 0, 24, 0, 1, STONE)
    w.fill(4, 5, 4, 5, 1, 15, FLESH, frac=0.9)  # a standing simplified body
    w.fill(5, 6, 4, 5, 1, 15, FLESH, frac=0.9)
    p = w.add_person(4, 4)
    w.step()
    w.o2[:] *= 0.10                             # the air is suddenly ruined
    fainted = died = None
    for t in range(400):
        w.step()
        if fainted is None and not p["awake"]:
            fainted = t
        if died is None and not p["alive"]:
            died = t
            break
    assert fainted is not None, "they must lose consciousness"
    assert died is not None and died > fainted, "and only LATER stop breathing"
    assert any("slumps" in e for e in p["events"])


def test_unconscious_body_keels_over():
    """A slack body is an object: on fainting it topples by the tree machinery
    and ends up LYING — its height collapses, its voxels survive."""
    w = World(30, 30, 24)
    w.open_sky = False
    w.fill(0, 30, 0, 30, 0, 1, STONE)
    w.fill(14, 16, 14, 16, 1, 16, FLESH, frac=0.9)   # a 75 cm standing column
    p = w.add_person(15, 15)
    n0 = int((w.mat == FLESH).sum())
    w.step()
    w.o2[:] *= 0.05
    for _ in range(400):
        w.step()
        if not w.bodies and not p["awake"]:
            zs = np.argwhere(w.mat == FLESH)
            if len(zs) and zs[:, 2].max() <= 8:
                break
    assert not p["awake"]
    assert not w.bodies, "the fall must END"
    zs = np.argwhere(w.mat == FLESH)
    assert len(zs) >= n0 - 2, "the body's voxels survive the fall"
    assert zs[:, 2].max() <= 8, "and it lies low, no longer standing"


def test_glass_shatters_on_landing_where_wood_thuds():
    """Impact toughness is a material COLUMN, not a case: the same fall smashes
    glass into many smaller fragments and leaves wood as a tidy intact stack."""
    def drop(material):
        w = World(12, 12, 14)
        w.fill(0, 12, 0, 12, 0, 1, STONE)
        w.fill(5, 6, 5, 6, 9, 12, material)     # a 1x1x3 column, high in the air
        for _ in range(60):                     # falling has a SPEED now, and starts
                                                # from a standstill (sim._fall_speed)
            w.step()
        return w
    wg = drop(GLASS)
    full = 2500.0 * wg.vox_l
    shards = np.argwhere(wg.mat == GLASS)
    assert len(shards) > 6, "glass must SMASH into many smaller pieces"
    assert float(wg.smass[wg.mat == GLASS].max()) < 0.9 * full, \
        "no fragment is a whole voxel anymore"
    assert abs(float(wg.smass[wg.mat == GLASS].sum()) - 3 * full) < 1.0, \
        "shattering conserves every gram"
    assert len(set((int(x), int(y)) for x, y, _ in shards)) > 1, \
        "and the shards SPREAD, not stack"
    ww = drop(WOOD)
    stack = np.argwhere(ww.mat == WOOD)
    assert len(stack) == 3, "wood survives the same fall as the same three voxels"
    assert stack[:, 2].max() == 3, "restacked, intact, on the floor"


def test_a_pond_cushions_the_fall():
    """Drag through liquid eats the drop: glass that would burst on stone
    lands whole on a basin floor under deep water."""
    w = World(10, 10, 14)
    w.fill(0, 10, 0, 10, 0, 1, STONE)
    w.fill(2, 8, 2, 8, 1, 6, STONE)             # basin walls
    w.mat[3:7, 3:7, 1:6] = AIR
    w.smass[3:7, 3:7, 1:6] = 0.0
    for x in range(3, 7):
        for y in range(3, 7):
            for z in range(1, 6):
                w.pour(x, y, z, WATER, w.cap)
    w.fill(4, 5, 4, 5, 10, 11, GLASS)           # one glass block over the pond
    for _ in range(20):
        w.step()
    glass = np.argwhere(w.mat == GLASS)
    assert len(glass) == 1 and int(glass[0][2]) == 1, \
        "it must sink whole to the basin floor — water drag, no shatter"


def test_flame_licked_glass_cracks_but_warmed_glass_survives():
    """Thermal shock from data: a steep DT across a face bursts glass; gentle
    warming never does."""
    def pane(hot_E):
        w = World(8, 8, 6)
        w.fill(0, 8, 0, 8, 0, 1, STONE)
        w.fill(3, 6, 3, 4, 1, 4, GLASS)         # a small standing pane
        for t in range(12):
            if t < 8:
                w.E[4, 4, 1] += hot_E           # air cell right beside it
            w.step()
        full = 2500.0 * w.vox_l
        return int(((w.mat == GLASS) & (w.smass > 0.9 * full)).sum())
    assert pane(6000.0) < 9, "flame-hot air beside cold glass must burst cells"
    assert pane(150.0) == 9, "a warm draft leaves the pane whole"


def test_tin_melts_over_embers_and_iron_only_glows():
    """Transformation from three numbers: same fire, same distance — tin (232°)
    runs, iron (1538°) shrugs. The difference is a MELT row, never a flag."""
    def bar(metal, ticks=520):
        w = World(12, 8, 10)
        w.fill(0, 12, 0, 8, 0, 1, STONE)
        w.fill(3, 9, 2, 6, 1, 3, CHAR, frac=0.8)     # an ESTABLISHED forge bed —
        bed = w.mat == CHAR                          # glowing coals, already lit
        w.E[bed] = 600.0 * w.heat_capacity()[bed]    # (charcoal cold-starts hard;
        w.fill(4, 8, 5, 7, 4, 5, STONE)              # lighting it is its own scene)
        w.fill(4, 8, 3, 5, 4, 5, metal)              # the bar, over the coals
        for _ in range(ticks):
            w.step()
        molten = float(w.fvol[np.isin(w.fl, (MTIN, MIRON))].sum())
        return molten, float(w.smass[w.mat == metal].sum())
    tin_ml, tin_left = bar(TIN)
    iron_ml, iron_left = bar(IRON)
    assert tin_ml > 50.0, "tin over glowing coals must RUN"
    assert iron_ml == 0.0 and iron_left > 0, "iron over the same coals only glows"


def test_molten_tin_freezes_back_where_it_pools():
    """The cycle closes: melt poured on cold stone sets solid again, grams
    conserved through both phase changes."""
    w = World(10, 10, 8)
    w.fill(0, 10, 0, 10, 0, 1, STONE)
    w.fl[4, 4, 4] = MTIN                             # a ladle of melt, mid-air
    w.fvol[4, 4, 4] = 100.0
    w.E[4, 4, 4] += 500.0                            # hot enough to stay liquid a bit
    g0 = 100.0 * 6.98
    for _ in range(120):
        w.step()
    tin = w.mat == TIN
    melt = np.isin(w.fl, (MTIN,)) & (w.fvol > 0)
    total = float(w.smass[tin].sum()) + float((w.fvol[melt] * 6.98).sum())
    assert tin.any(), "cooling melt must SET as solid tin"
    assert np.argwhere(tin)[:, 2].max() <= 1, "as a splat at floor level"
    assert abs(total - g0) < 1.0, "no gram lost between the states"


def test_reflexes_flee_and_alarm_spreads_by_shout_or_sight():
    """The WILL layer with voices: the one who SEES the fire shouts; across
    open floor the shout outruns everything and the other flees on HEARING
    alone. Behind enough masonry the shout dies — then only the sight of the
    runner passing can raise them."""
    def scene(wall):
        w = World(140, 26, 20, voxel_cm=5)   # roomy in y: bodies have DEPTH
        w.fill(0, 140, 0, 26, 0, 1, STONE)   # now, and two of them queueing at
        w.exits = [(6, 12), (6, 7)]          # one door cannot pass each other
        if wall:                                     # a thick baffle between them
            w.fill(60, 66, 0, 22, 1, 18, STONE)      # (leaves a south gap to walk)
        w.fill(128, 130, 6, 8, 1, 4, WOOD)
        w.fill(120, 121, 7, 8, 1, 15, FLESH, frac=0.9)
        pa = w.add_person(120, 7, "A")
        w.fill(18, 19, 7, 8, 1, 15, FLESH, frac=0.9)   # far enough that B cannot
        pb = w.add_person(18, 7, "B")                  # SEE the fire even after
                                                       # wandering a little
        w.E[128, 6, 1] = 9.0e5
        b_event = None
        for t in range(900):
            w.step()
            for e in pb["events"]:
                if "startles" in e and b_event is None:
                    b_event = e
            pb["events"].clear()
            pa["events"].clear()
            if pa["safe"] and pb["safe"]:
                break
        return w, pa, pb, b_event
    w, pa, pb, be = scene(wall=False)
    assert any("Fire" in s[2] for s in w.speech), "the witness must SHOUT"
    assert be is not None and "hears" in be, \
        "across open floor, B flees on HEARING the shout, seeing nothing"
    assert pa["safe"] and pb["safe"], "both walk out"
    w2, pa2, pb2, be2 = scene(wall=True)
    assert be2 is not None and "hears" not in be2, \
        "six voxels of stone eat the shout — B startles some other way"


def test_thermostat_block_heats_without_any_fire():
    """The dev heater: a block pinned at temperature radiates and conducts
    like anything hot, with NO combustion — tin beside it melts while the
    room's oxygen never moves. Heat and fire are separate things."""
    w = World(12, 6, 8)
    w.open_sky = False
    w.fill(0, 12, 0, 6, 0, 1, STONE)
    w.fill(4, 5, 2, 3, 1, 2, IRON)              # the element
    w.thermostats.append((4, 2, 1, 800.0))
    w.fill(5, 8, 2, 3, 1, 2, TIN)               # work-piece touching it
    w.step()
    o2_0 = float(w.o2.sum())
    for _ in range(300):
        w.step()
    assert float(w.fvol[w.fl == MTIN].sum()) > 50.0, \
        "the pinned block must melt the tin beside it"
    assert abs(float(w.o2.sum()) - o2_0) < 0.5, \
        "and burn NOTHING doing it — no fire, no oxygen spent"


def _alarm_room(policy=None):
    """One room, one door, one body, one fire in the corner."""
    w = World(60, 16, 20, voxel_cm=5)
    if policy is not None:
        w.policy = policy
    w.fill(0, 60, 0, 16, 0, 1, STONE)
    w.exits = [(6, 7)]
    w.fill(48, 50, 6, 8, 1, 4, WOOD)
    w.fill(40, 41, 7, 8, 1, 15, FLESH, frac=0.9)
    p = w.add_person(40, 7, "A")
    w.E[48, 6, 1] = 9.0e5
    return w, p


def test_every_decision_logs_the_whole_menu_beside_the_pick():
    """The harvest's first requirement: you cannot learn from a choice
    without knowing what ELSE was on offer. So a trace row carries the full
    menu, the situation that produced it, the pick, and who picked."""
    w, p = _alarm_room()
    for _ in range(600):
        w.step()
        if p["safe"]:
            break
    assert w.traces, "startling is a DECISION and must leave a trace"
    row = next(r for r in w.traces if r["percept"] == "sees_fire")
    for limb, menu in row["menus"].items():
        assert row["pick"][limb] in menu, f"a {limb} pick must come FROM its menu"
    assert len(row["menus"]["legs"]) > 1, "a menu of one is not a choice"
    assert row["by"] == "table", "the reflex table is today's policy"
    assert "stay" in row["menus"]["legs"], \
        "standing pat is always an option, and must be OFFERED as one"
    assert any(k.startswith("go(the door") for k in row["menus"]["legs"]), \
        "and so is the door it can see and reach"
    assert "say(coming)" not in row["menus"]["mouth"], \
        "answering presupposes something to answer — this body heard nothing"
    assert row["situation"]["percept"] == row["percept"]
    outs = {r["outcome"] for r in w.trace_outcomes()}
    assert outs == {"safe"}, "and the row learns how that body ended"


def test_swapping_the_policy_changes_what_the_body_does():
    """The seam is real, not decoration: the sim asks something else, and
    obeys a different answer. A body told to stand pat stands pat — and
    still gets the same menu, so the choice was available and declined."""
    class Coward:
        name = "always_stay"

        def pick(self, situation, menu):
            return [o["key"] for o in menu].index("stay") \
                if situation["limb"] == "legs" else 0

    w, p = _alarm_room(policy=Coward())
    for _ in range(600):
        w.step()
        if p["safe"]:
            break
    assert not p["safe"], "this policy never leaves, so nobody reaches the door"
    assert w.traces and all(r["pick"]["legs"] == "stay" for r in w.traces)
    fire = next(r for r in w.traces if r["percept"] == "sees_fire")
    assert any(k.startswith("go(the door") for k in fire["menus"]["legs"]), \
        "fleeing was OFFERED and passed over"
    assert not w.speech, "and a body that stays put says none of the shout lines"
    w2, p2 = _alarm_room()
    for _ in range(600):
        w2.step()
        if p2["safe"]:
            break
    assert p2["safe"], "the same world with the table policy walks out"


def test_the_menu_never_offers_a_door_that_is_not_there():
    """Legality is CHECKED. Wall the body in and the flee rows vanish — the
    mind is never shown an option the sim cannot carry out, which is what
    makes an illegal action impossible rather than merely discouraged."""
    w, p = _alarm_room()
    w.fill(36, 38, 0, 16, 1, 19, STONE)          # seal the room, floor to roof
    for _ in range(400):
        w.step()
        if w.traces:
            break
    assert w.traces, "a walled-in body still NOTICES the fire"
    row = w.traces[0]
    assert not any(k.startswith("go(the door") for k in row["menus"]["legs"]), \
        "with no route out, the door is not on the menu at all"
    assert row["pick"]["legs"] == "stay", \
        "and the table's answer being unavailable falls back to a legal one"


def _two_rooms(barrier, frac=1.0):
    """Two sealed rooms; a fire in one. `barrier` is what stands between them."""
    w = World(40, 20, 16, voxel_cm=5)
    w.open_sky = False
    w.fill(0, 40, 0, 20, 0, 16, STONE)
    for x0, x1 in ((1, 19), (21, 39)):
        w.mat[x0:x1, 1:19, 1:15] = AIR
        w.smass[x0:x1, 1:19, 1:15] = 0.0
    if barrier == "open":
        w.mat[19:21, 7:13, 1:9] = AIR
        w.smass[19:21, 7:13, 1:9] = 0.0
    elif barrier == "door":
        w.fill(19, 21, 7, 13, 1, 9, WOOD, frac=frac)
    w.fill(5, 8, 8, 11, 1, 3, WOOD)
    w.E[5, 8, 1] = 6.0e5
    w.step()
    for _ in range(400):
        w.step()
    return float(w.smoke[21:39, 1:19, 1:15].sum())


def test_a_barrier_leaks_by_how_it_FITS_not_by_what_it_is_made_of():
    """Ruling 2: porosity is DERIVED from the space left in a voxel, never
    declared per material. Every door here is the same wood — only the fit
    differs, and only the fit changes what gets through."""
    opening = _two_rooms("open")
    tight = _two_rooms("door", frac=1.0)
    gap = _two_rooms("door", frac=0.97)
    badly_hung = _two_rooms("door", frac=0.90)
    assert opening > 1.0, "an open doorway passes smoke freely"
    assert tight == 0.0, \
        "a door that fills its voxels completely leaves nothing to pass through"
    assert 0.0 < gap < badly_hung < opening, \
        "the wider the gap, the more gets through — and a gap is never a doorway"
    assert badly_hung > 5 * gap, \
        "and the leak scales with the gap, not with the fact that it is wood"


def test_burning_opens_a_solid_block_with_nobody_writing_that():
    """The tell that a derived quantity is the right shape: behaviour nobody
    wrote. Fire eats mass, mass is what fills the voxel, so a burning block
    grows porous at its charred face and starts breathing on its own."""
    w = World(20, 12, 14, voxel_cm=5)
    w.open_sky = False
    w.fill(0, 20, 0, 12, 0, 14, STONE)
    w.mat[1:19, 1:11, 1:13] = AIR
    w.smass[1:19, 1:11, 1:13] = 0.0
    w.fill(6, 12, 4, 8, 1, 5, WOOD)              # packed solid: no void at all
    blk = (slice(6, 12), slice(4, 8), slice(1, 5))
    assert float(w.porosity()[blk].max()) == 0.0, "a packed block starts sealed"
    w.E[6, 4, 1] = 8.0e5
    m0 = float(w.smass[blk].sum())
    for _ in range(150):
        w.step()
    assert float(w.smass[blk].sum()) < m0, "the fire ate some of the block"
    assert float(w.porosity()[blk].max()) > 0.1, \
        "and the mass it ate is now void the gases can move through"


def _walled_pair(window):
    """Two rooms, a fire in one, a person in the other, and a stone divider
    that either carries a window or does not. ONE line differs."""
    w = World(64, 24, 22, voxel_cm=5)
    w.open_sky = False
    w.fill(0, 64, 0, 24, 0, 22, STONE)
    for x0, x1 in ((1, 30), (34, 63)):
        w.mat[x0:x1, 1:23, 1:21] = AIR
        w.smass[x0:x1, 1:23, 1:21] = 0.0
    w.fill(30, 34, 1, 23, 1, 21, STONE)                  # the divider
    if window:
        w.fill(30, 34, 9, 15, 8, 16, GLASS)              # <-- the only difference
    w.exits = [(60, 12)]
    w.fill(6, 14, 9, 15, 3, 5, WOOD, frac=0.7)           # a bench
    for (lx, ly) in ((6, 9), (6, 14), (13, 9), (13, 14)):
        w.fill(lx, lx + 1, ly, ly + 1, 1, 3, WOOD)       # ...on legs
    w.fill(40, 41, 11, 12, 1, 15, FLESH, frac=0.9)
    p = w.add_person(40, 11, "Witness")
    w.E[8, 10, 3] = 9.0e5
    seen, peak = None, 0
    for t in range(260):
        w.step()
        lit = int(w.burning().sum())
        peak = max(peak, lit)
        if seen is None and lit and w._sees((41.0, 11.0, 15.0), (8.0, 10.0, 4.0)):
            seen = t
        p["events"].clear()
    return w, p, seen, peak


def test_a_body_learns_of_the_fire_ONLY_by_perceiving_it():
    """No information may reach a mind except through its senses. Brick up the
    window and the same fire burns just as hard, but the witness never learns
    of it: no sight, no percept, no decision, no word, no step. The sim knows
    where the fire is; the person is not the sim."""
    w1, p1, saw1, peak1 = _walled_pair(True)
    w0, p0, saw0, peak0 = _walled_pair(False)
    assert peak1 > 5 and peak0 > 5, "both worlds must actually burn"
    assert abs(peak1 - peak0) <= 3, \
        "the FIRE must be the same fire — only perception may differ"
    assert saw1 is not None, "through a window, the fire is visible"
    assert p1["fleeing"] and w1.traces, "and the witness acts on having seen it"
    assert saw0 is None, "through masonry, it is not"
    assert any(r["percept"] == "sees_fire" for r in w1.traces), \
        "the witness who could see it decided ABOUT it"
    assert all(r["percept"] is None for r in w0.traces), \
        "the walled-off one may potter about, but NO percept may reach it — " \
        "no decision may ever be taken about a fire never perceived"
    assert not w0.speech, \
        "and it never cries out — nothing leaks from the lattice into a mind"


def _two_room_house(knows_world):
    """Two rooms joined by one doorway; a body at the far end from the exit."""
    w = World(70, 26, 22, voxel_cm=5)
    w.open_sky = False
    w.fill(0, 70, 0, 26, 0, 22, STONE)
    for x0, x1 in ((1, 32), (36, 69)):
        w.mat[x0:x1, 1:25, 1:21] = AIR
        w.smass[x0:x1, 1:25, 1:21] = 0.0
    w.fill(32, 36, 1, 25, 1, 21, STONE)
    w.mat[32:36, 11:15, 1:16] = AIR          # the one doorway between them
    w.smass[32:36, 11:15, 1:16] = 0.0
    w.fill(50, 52, 1, 20, 1, 21, STONE)      # a baffle: the far door is NOT on
    w.exits = [(66, 4)]                      # the sightline through the doorway
    w.fill(8, 9, 12, 13, 1, 15, FLESH, frac=0.9)
    p = w.add_person(8, 12, "X", knows_world=knows_world)
    return w, p


def _place_tags(w, p):
    """Which KINDS of place this body could name, right now."""
    cells = np.argwhere(w.mat == FLESH)
    fit, _start = w._fit_grid(cells, p.get("known"))
    return {pl["tag"] for pl in w._places(p, cells, fit)}


def test_a_stranger_must_FIND_the_door_a_resident_already_knows_it():
    """A body may only aim at what it has seen. The resident is given the
    building and can head for a door across two rooms; the stranger starts
    blind, is offered no exit at all, and has to go and look — which is what
    turns wandering from filler into the thing that makes knowing honest."""
    w_r, resident = _two_room_house(True)
    w_s, stranger = _two_room_house(False)
    w_s.step()
    assert stranger["known"].mean() < 0.6, \
        "a stranger has seen only what one look affords"
    assert resident["known"].all(), "the resident is given the place"
    assert "go:exit" not in _place_tags(w_s, stranger), \
        "a door never laid eyes on is not a destination — not even an option"
    for _ in range(600):
        w_s.step()
        stranger["events"].clear()
    assert stranger["known"].mean() > 0.9, \
        "left alone, it goes and looks, and comes to know the building"
    assert any(r["tags"].get("legs") == "go:frontier" for r in w_s.traces), \
        "and it chose to — going to look is a PICK from the menu, not a script"
    assert "go:exit" in _place_tags(w_s, stranger), \
        "having found the door, it can now aim at it"


def test_a_body_with_nothing_happening_still_does_something():
    """Impetus: no fire, no shout, nothing to react to — and the body still
    weighs options, picks one, and moves. Idling is a choice made through the
    same menu and written to the same trace, not a gap between choices."""
    w, p = _two_room_house(True)
    start = p["anchor"]
    seen = set()
    for _ in range(400):
        w.step()
        seen.add(p["anchor"])
        p["events"].clear()
    assert w.traces, "an undisturbed body still DECIDES"
    assert all(r["percept"] is None for r in w.traces), "and nothing prompted it"
    assert len(seen) > 8, f"it should get about; it visited only {len(seen)} spots"
    assert p["anchor"] != start, "it did not simply stand where it was put"
    assert {r["tags"]["legs"] for r in w.traces} <= \
        {"go:frontier", "go:roam", "go:step", "stay"}


def _cantilever(frac):
    """A beam jutting out of a stone pillar with nothing under its far end."""
    w = World(40, 10, 24, voxel_cm=5)
    w.fill(0, 40, 0, 10, 0, 1, STONE)
    w.fill(2, 5, 3, 7, 1, 20, STONE)                  # the pillar
    w.fill(5, 34, 3, 7, 16, 18, WOOD, frac=frac)      # the beam
    for _ in range(150):
        w.step()
    still = np.argwhere((w.mat[5:34, 3:7, 16:18] == WOOD)
                        & (w.smass[5:34, 3:7, 16:18] > 0))
    return 0 if not len(still) else int(still[:, 0].max()) + 5


def test_a_beam_carries_by_the_MASS_it_still_has():
    """Strength is a property of the beam that is actually there, not of the
    word 'wood'. A voxel eaten to a fifth of itself used to carry like sound
    timber right up to the tick it turned to ash — so a burning tree held its
    own canopy over the fire consuming it. Reach now fades with what is left,
    with the knee at half mass so a scene's `frac` fills keep their strength."""
    full, half = _cantilever(1.0), _cantilever(0.5)
    assert full == half, \
        "down to half mass a beam is still a beam — scene fills must not sag"
    thin, thinner, gone = _cantilever(0.4), _cantilever(0.3), _cantilever(0.2)
    assert thin < full, "past the knee, a thinner beam cannot hold as far out"
    assert gone < thinner < thin, "and it keeps shortening as the mass goes"


def test_a_walker_reaches_a_door_set_in_an_outer_wall():
    """A body is not a point. The route is walked by its centre, but the body
    is several voxels across, so a door in an outer wall is a column the
    centre can never occupy — its shoulder would be in the masonry. The
    planner used to route there anyway, the legs refused (rightly), and the
    walker jammed against the wall shuffling sideways for the rest of the run.
    Plan on columns the whole footprint fits, and arrive by the BODY."""
    w = World(40, 40, 40, voxel_cm=5)
    w.fill(0, 40, 0, 40, 0, 40, STONE)
    w.mat[1:39, 1:39, 1:39] = AIR
    w.smass[1:39, 1:39, 1:39] = 0.0
    w.exits = [(38, 10)]                      # hard against the east wall
    from src.voxel.scenes import _person
    p = _person(w, 30, 30)
    w.step()
    p["emergency"], p["fleeing"], p["goal"] = True, True, (38, 10)
    p["_path"] = None
    for _ in range(260):
        w.step()
        p["events"].clear()
        if p["safe"]:
            break
    assert p["safe"], f"it must get to the door; it stopped at {p['anchor']}"


def test_a_body_that_went_down_from_bad_air_comes_round_in_good_air():
    """The fainting threshold used to trip one way only, so a body carried out
    of the smoke stayed unconscious for ever and rescuing anyone was pointless.
    Air can undo what air did — with hysteresis, since recovery genuinely lags,
    and never for burns, whose integral only ever climbs."""
    w = World(30, 14, 22, voxel_cm=5)
    w.open_sky = False
    w.fill(0, 30, 0, 14, 0, 22, STONE)
    w.mat[1:29, 1:13, 1:21] = AIR
    w.smass[1:29, 1:13, 1:21] = 0.0
    w.fill(3, 9, 4, 9, 1, 3, WOOD, frac=0.8)
    from src.voxel.scenes import _person
    p = _person(w, 20, 6)
    p["name"] = "S"
    w.E[3, 4, 1] = 9.0e5
    went_down = came_round = None
    for t in range(600):
        w.step()
        if went_down is None and not p["awake"]:
            went_down = t
        if went_down is not None and came_round is None and p["awake"]:
            came_round = t
        if not p["awake"] and p["alive"]:          # the rescue: clean air
            w.o2[:] = O2_PER_L * w.vox_l
            w.smoke[:] = 0.0
        p["events"].clear()
        if came_round:
            break
    assert went_down is not None, "the smoke must put this body down"
    assert p["alive"], "and the rescue must reach it before it dies"
    assert came_round is not None, "clean air must bring it back"
    assert came_round > went_down, "and it must have been out for a while"
    assert p["blood_o2"] > BODY["faint_o2"], \
        "it wakes clear of the line it fell at, not balanced on it"


def _anvil_onto(plate, from_z):
    """An iron block dropped onto a plate spanning two piers."""
    w = World(16, 16, 60, voxel_cm=5)
    w.fill(0, 16, 0, 16, 0, 1, STONE)
    w.fill(4, 12, 4, 12, 6, 7, plate)
    w.fill(4, 5, 4, 12, 1, 6, STONE)
    w.fill(11, 12, 4, 12, 1, 6, STONE)
    w.fill(7, 9, 7, 9, from_z, from_z + 2, IRON)
    for _ in range(300):
        w.step()
    blk = np.argwhere(w.mat == IRON)
    return int(blk[:, 2].min()) if len(blk) else -1


def test_a_falling_body_damages_WHAT_IT_LANDS_ON():
    """Newton's third law: the impulse is shared, so the struck cell is tested
    against its own toughness with the same energy. A falling anvil used to be
    able to hurt only itself — it went through a glass table without marking
    it. Glass gives; timber and masonry take the same blow and hold."""
    assert _anvil_onto(GLASS, 40) < _anvil_onto(WOOD, 40), \
        "the glass plate must give under the anvil where the plank does not"
    assert _anvil_onto(WOOD, 40) == _anvil_onto(WOOD, 10), \
        "and a plank holds whatever height it is dropped from"


def test_lifting_and_shoving_fall_out_of_ONE_force():
    """Hands are not five verbs. A body puts a bounded force on a thing; that
    force meets gravity when you lift and friction when you shove, so the same
    strength gives two different limits and nobody writes them down separately.
    An anvil you cannot pick up is an anvil you can still slide."""
    w = World(20, 20, 20, voxel_cm=5)
    w.fill(0, 20, 0, 20, 0, 1, STONE)
    w.fill(5, 9, 5, 9, 1, 5, IRON)                    # the anvil
    anvil = w._object_at(6, 6, 2)
    lift, shove = w._effort(anvil)
    assert lift > BODY["strength_N"], "an anvil is not picked up"
    assert shove < BODY["strength_N"], "...but it does slide"
    w2 = World(20, 20, 20, voxel_cm=5)
    w2.fill(0, 20, 0, 20, 0, 1, STONE)
    w2.fill(5, 9, 5, 9, 1, 5, WOOD, frac=0.2)         # a wicker basket
    light = w2._object_at(6, 6, 2)
    assert w2._effort(light)[0] < BODY["strength_N"], \
        "and something light goes straight up — same law, different mass"
    assert w._shove(anvil, 1, 0), "a thing shoved into clear space moves"
    assert int(w.mat[9, 6, 2]) == IRON and int(w.mat[5, 6, 2]) == AIR, \
        "...taking its whole self with it"


def test_two_people_who_touch_stay_two_people():
    """Identity used to be adjacency, so the moment two bodies touched they
    became one two-headed person: both resolved to the same cells and both
    anchors converged. Anything that brings people into contact — dragging,
    carrying, a crowd at a door — broke on it."""
    from src.voxel.scenes import _person
    w = World(50, 26, 40, voxel_cm=5)
    w.fill(0, 50, 0, 26, 0, 40, STONE)
    w.mat[1:49, 1:25, 1:39] = AIR
    w.smass[1:49, 1:25, 1:39] = 0.0
    a = _person(w, 16, 12); a["name"] = "A"
    b = _person(w, 22, 12); b["name"] = "B"
    w.step()
    apart = int(w._person_cells(a)[0].sum())
    assert apart == int(w._person_cells(b)[0].sum()) > 100
    for _ in range(4):                                # walk B into A
        cb, sl = w._person_cells(b)
        c = np.argwhere(cb); c[:, 0] += sl[0].start; c[:, 1] += sl[1].start
        w._shove(c, -1, 0)
        b["anchor"] = (b["anchor"][0] - 1, b["anchor"][1])
        w.tick += 1
    total = int((w.mat == FLESH).sum())
    for who in (a, b):
        n = int(w._person_cells(who)[0].sum())
        assert n < total * 0.75, \
            f"{who['name']} claimed {n} of {total} — that is both bodies"


def _burning_room_with_a_body_on_the_floor(rescuer):
    """A fire, a person already unconscious on the floor, and one who is not."""
    from src.voxel.scenes import _person
    w = World(50, 26, 40, voxel_cm=5)
    w.open_sky = False
    w.fill(0, 50, 0, 26, 0, 40, STONE)
    w.mat[1:49, 1:25, 1:39] = AIR
    w.smass[1:49, 1:25, 1:39] = 0.0
    w.exits = [(46, 12)]
    w.fill(4, 10, 8, 16, 1, 3, WOOD, frac=0.8)
    w.E[4, 8, 1] = 9.0e5
    down = _person(w, 16, 12); down["name"] = "Fallen"
    hero = _person(w, 22, 12); hero["name"] = "Hero"
    if rescuer:                                   # the character sheet, edited:
        hero["reflexes"] = {k: {"legs": "go:exit", "hands": "hold"}
                            for k in ("sees_fire", "chokes",   # legs leave, and
                                      "hears_alarm", "sees_runner")}  # hands do
                                                  # not leave empty. One row, two
                                                  # parts of a body, no compound
                                                  # response written anywhere
    for _ in range(1500):
        w.step()
        down["awake"] = False                     # hold them under for the test
        down["blood_o2"] = min(down["blood_o2"], 0.30)
        hero["events"].clear(); down["events"].clear()
        if hero["safe"]:
            break
    return hero, down, w


def test_someone_can_be_DRAGGED_out_and_it_is_a_choice():
    """The whole point of hands. Nobody could be helped before: a fainted body
    was scenery. Now a rescuer takes hold — legal only because the force
    arithmetic says this body can shift that one — and hauls them to the door
    with the same _shove that slides a crate. It is a CHOICE, not a rule: the
    same world with an ordinary character sheet leaves them where they lie."""
    hero_r, down_r, w_r = _burning_room_with_a_body_on_the_floor(True)
    hero_d, down_d, w_d = _burning_room_with_a_body_on_the_floor(False)
    assert hero_d["safe"] and not down_d["safe"], \
        "an ordinary person saves themselves and leaves the body"
    assert hero_r["safe"] and down_r["safe"], \
        "a rescuer brings them out too"
    assert any(r["tags"].get("hands") == "hold" for r in w_r.traces), \
        "and it went through the menu like any other decision"
    assert all(r["tags"].get("hands") != "hold" for r in w_d.traces), \
        "while the other never even considered it"
    grab = next(r for r in w_r.traces if r["tags"].get("hands") == "hold")
    assert " & " in grab["program"], \
        "hauling someone out is TWO parts of a body used at once, composed — " \
        "there is no drag_them_out response anywhere in the sim"
    # AND A WHOLE BODY CAME OUT. This test once passed while the hauler towed
    # a THREE-voxel fragment of a 342-voxel person to the door and the flag
    # said "safe" — a rescue that read correct and moved nobody. Count the
    # flesh: a person who was carried out is a person who is no longer here.
    assert not (w_r.mat == FLESH).any(), \
        "both bodies left the lattice; nothing of either was left behind"
    assert int((w_d.mat == FLESH).sum()) > 300, \
        "while in the other world a whole person is still lying on the floor"


# ---------------------------------------------------------------------------
# MOMENTUM. Four things a body should be able to do, none of which it can:
# jump, fall off a cliff and roll, hang under a canopy, swing an axe. They are
# one missing quantity, not four missing features — matter here has position
# and mass and no VELOCITY, so every fall is one voxel a tick whatever the
# thing weighs or however long it has been falling, and a blow carries the
# force of a push. These tests are the specification for that quantity. They
# are marked xfail(strict) so they shout the day they pass.
# ---------------------------------------------------------------------------

class _Wants:
    """A policy that picks one named act per limb and the null act otherwise.
    These tests are about BODIES, not about taste, so taste is pinned."""

    name = "wants"

    def __init__(self, **want):
        self.want = want

    def pick(self, situation, menu):
        tag = self.want.get(situation["limb"])
        for i, opt in enumerate(menu):
            if opt["tag"] == tag:
                return i
        return 0


def _lowest_flesh_z(w):
    """How low the person's matter is, ON the lattice or IN THE AIR. A body in
    flight has left the grid, so counting only lattice cells says a jumper
    vanished rather than rose."""
    zs = []
    on = np.argwhere(w.mat == FLESH)
    if len(on):
        zs.append(int(on[:, 2].min()))
    flying = w.bodies_array()
    if len(flying):
        sel = flying[flying[:, 3] == FLESH]
        if len(sel):
            zs.append(int(sel[:, 2].min()))
    return min(zs) if zs else None


def _flesh_shape(w):
    """The person's SHAPE, free of where they happen to be — counting matter in
    flight as well as matter on the lattice, since a jumper spends most of a
    jump off the grid entirely."""
    parts = [np.argwhere(w.mat == FLESH)]
    flying = w.bodies_array()
    if len(flying):
        sel = flying[flying[:, 3] == FLESH]
        if len(sel):
            parts.append(np.round(sel[:, :3]).astype(np.int64))
    cells = np.concatenate([q for q in parts if len(q)]) if any(
        len(q) for q in parts) else None
    if cells is None:
        return None
    return set(map(tuple, cells - cells.min(axis=0)))


def _flesh_grams(w):
    """Every gram of person there is, wherever it happens to be."""
    on = float(w.smass[w.mat == FLESH].sum())
    return on + sum(float(b["masses"][b["mats"] == FLESH].sum()) for b in w.bodies)


def _standing_room(nx=30, ny=20, nz=80):
    w = World(nx, ny, nz, voxel_cm=5)
    w.fill(0, nx, 0, ny, 0, 1, STONE)
    w.exits = [(nx - 2, ny // 2)]
    return w


def test_a_person_can_JUMP_and_it_is_a_CHOICE():
    """A jump is legs pushing the whole body off the ground, and the height
    must FALL OUT of the push and the mass — not be a number anybody typed.
    ~400 N against 28.5 kg gets a person clear of the floor; a heavier person
    gets less clear, for free, because it is the same arithmetic.

    It is also a CHOICE: offered only when there is ground underfoot to push
    against, and taken through the same menu as everything else. The body that
    does not choose it never leaves the floor, in the same room."""
    from src.voxel.scenes import _person
    w = _standing_room()
    _person(w, 10, 10)
    w.policy = _Wants(legs="jump")
    whole = _flesh_grams(w)
    floor = _lowest_flesh_z(w)
    shape = _flesh_shape(w)
    high, back, lightest, landed = floor, False, whole, None
    for _ in range(200):
        w.step()
        z = _lowest_flesh_z(w) or floor
        high = max(high, z)
        back = back or (high > floor and z == floor)   # up, and down again
        lightest = min(lightest, _flesh_grams(w))
        landed = _flesh_shape(w) or landed
    # ~1 m, and that is a CONSEQUENCE, not a setting: 1400 N of leg against
    # 28.5 kg, over a 25 cm crouch. The body is known to be light for its size
    # (see scenes._person), so it jumps high for a person — give it a real
    # adult's mass and the same legs and the same arithmetic gives ~0.4 m.
    assert (high - floor) * 0.05 >= 0.10, \
        f"the body never left the ground (best {high - floor} voxels up)"
    assert back, "and it comes back DOWN — what goes up is not a policy choice"
    assert abs(lightest - whole) < 1.0, \
        f"whole all the way through — a jump is not an injury ({lightest} of {whole} g)"
    # AND THE SAME SHAPE. Mass alone does not catch this: a landing that drove
    # the body into the floor shoved the buried cells upward as debris, which
    # comes out through whatever is at the top of a person, and every gram was
    # still present while the face was being rearranged.
    assert landed == shape, \
        "and lands in the shape it left in — a jump is not a disfigurement"

    w2 = _standing_room()
    _person(w2, 10, 10)
    base = _lowest_flesh_z(w2)
    for _ in range(200):
        w2.step()
        assert _lowest_flesh_z(w2) == base, \
            "a body that did not choose to jump never leaves the floor"


@pytest.mark.xfail(strict=True, reason="no momentum: a fall cannot be spread over time")
def test_ROLLING_on_landing_spreads_the_blow_that_a_rigid_landing_takes_whole():
    """The same fall, the same body, the same floor — and one of them survives
    it. A landing is a momentum change: the force is the change divided by the
    TIME taken to make it, so a body that keeps moving and comes to rest over
    many ticks is struck far less hard than one that stops dead. That is the
    whole of rolling, and it is arithmetic rather than a rule about rolls.

    Measured in flesh: a rigid landing from this height breaks the body,
    the same landing rolled does not."""
    from src.voxel.scenes import _person

    def drop(roll):
        w = World(40, 20, 60, voxel_cm=5)
        w.fill(0, 40, 0, 20, 0, 1, STONE)             # the ground
        w.fill(0, 14, 0, 20, 1, 26, STONE)            # a ledge to stand off
        w.exits = [(38, 10)]
        p = _person(w, 9, 10)                         # standing on the ledge
        for c in ("torso",):                          # lift the body onto it
            pass
        w.policy = _Wants(legs="roll" if roll else "stay")
        before = _flesh_grams(w)
        for _ in range(400):
            w.step()
            if _lowest_flesh_z(w) is not None and _lowest_flesh_z(w) <= 2:
                break
        for _ in range(40):
            w.step()
        return before, _flesh_grams(w), p

    b0, hard, p0 = drop(False)
    b1, soft, p1 = drop(True)
    assert hard < b0 * 0.98, "a rigid landing from 1.2 m breaks a body"
    assert soft > hard, "and rolling through it costs less"
    assert p1["alive"], "the one who rolled lives"


def test_a_WIDE_thing_falls_SLOWER_than_a_compact_one_of_the_same_mass():
    """The parachute, with nothing in it that is about parachutes. Two objects
    of exactly the same material and exactly the same mass, one spread flat and
    one balled up, dropped the same distance. Air resists what it must go
    around, so the flat one loses. Everything a canopy does is this, and the sim
    already knows the shape — a cross-section is countable.

    Today they land on the same tick, because everything unsupported falls one
    voxel a tick whatever it is."""

    def fall(x0, x1, y0, y1, deep):
        w = World(30, 30, 140, voxel_cm=5)
        w.fill(0, 30, 0, 30, 0, 1, STONE)
        # a CANOPY is a lot of area with very little behind it, so the stuff is
        # thin: at 1% fill this is fabric, not planking
        w.fill(x0, x1, y0, y1, 110, 110 + deep, WOOD, frac=0.01)
        mass = float(w.smass[w.mat == WOOD].sum())
        for t in range(3000):
            w.step()
            solid = np.argwhere(w.mat == WOOD)
            if len(solid) and int(solid[:, 2].min()) <= 1:
                return t, mass
        return 3000, mass

    t_flat, m_flat = fall(3, 28, 3, 28, 1)          # spread: 25 x 25 x 1
    t_ball, m_ball = fall(13, 18, 13, 18, 25)       # balled: 5 x 5 x 25
    assert abs(m_flat - m_ball) < 1.0, \
        f"the test is only fair if the masses match ({m_flat} vs {m_ball})"
    assert t_flat > t_ball * 1.3, \
        f"the spread sheet must lose to the wad ({t_flat} vs {t_ball} ticks)"


@pytest.mark.xfail(strict=True, reason="no momentum: a swing carries no more than a push")
def test_an_axe_SWUNG_bites_where_the_same_axe_PRESSED_does_not():
    """The same body, the same axe, the same tree. Leaning on it does nothing;
    swinging it takes a bite. Nothing here is about axes — a swing puts the
    body's force behind a MOVING mass, and what arrives is kinetic energy
    delivered over the short distance the edge takes to stop, which is a far
    greater force than the same body could ever push with. The blade is sharp
    in the only way the sim can mean it: the contact is a few voxels, so the
    same energy lands as a much larger stress."""
    from src.voxel.scenes import _person

    def chop(swing):
        w = World(40, 20, 40, voxel_cm=5)
        w.fill(0, 40, 0, 20, 0, 1, STONE)
        w.fill(24, 27, 9, 12, 1, 30, WOOD)             # the trunk
        w.exits = [(38, 10)]
        p = _person(w, 18, 10)
        w.fill(21, 23, 10, 11, 20, 21, IRON)           # an axe head, in reach
        w.policy = _Wants(hands="swing" if swing else "press", legs="stay")
        before = float(w.smass[w.mat == WOOD].sum())
        for _ in range(300):
            w.step()
        return before, float(w.smass[w.mat == WOOD].sum()), p

    b0, pressed, _ = chop(False)
    b1, hewn, _ = chop(True)
    assert pressed > b0 * 0.999, "leaning on an axe does not fell anything"
    assert hewn < b1 * 0.99, "swinging it takes wood out of the trunk"


def _drop_sheet(x0, x1, y0, y1, frac, top=100, nz=110):
    """Drop one flat sheet from `top` and return (ticks to land, grams)."""
    w = World(30, 30, nz, voxel_cm=5)
    w.fill(0, 30, 0, 30, 0, 1, STONE)
    w.fill(x0, x1, y0, y1, top, top + 1, WOOD, frac=frac)
    grams = float(w.smass[w.mat == WOOD].sum())
    for t in range(3000):
        w.step()
        here = np.argwhere(w.mat == WOOD)
        if len(here) and int(here[:, 2].min()) <= 1:
            return t, grams
    return 3000, grams


def test_a_WIDER_sheet_of_the_same_stuff_falls_at_the_SAME_speed():
    """The invariant that says we modelled the right thing. Terminal speed
    balances drag against weight, and BOTH grow with area — so a big flat sheet
    and a small flat sheet of the same material and the same thickness come
    down together, however different their sizes. Mass per unit area is what
    matters, not size.

    This is the test that would catch drag being read off a footprint: an
    implementation that made bigger things slower would fail it, and would look
    perfectly convincing on the wide-versus-balled test alone."""
    small, gs = _drop_sheet(11, 19, 11, 19, 0.02)        # 8 x 8
    big, gb = _drop_sheet(5, 25, 5, 25, 0.02)            # 20 x 20, 6x the mass
    assert gb > gs * 5, "the wider sheet really is much heavier in total"
    assert abs(small - big) <= 2, \
        f"and yet they land together ({small} vs {big} ticks) — same mass per area"


def test_a_WIDER_canopy_under_the_SAME_mass_falls_SLOWER():
    """And the invariant is exactly why a parachute works. Hold the LOAD fixed
    and spread it over more canopy: mass per unit area drops, and so does the
    speed it settles at. This is the design rule for every canopy ever sewn,
    and the sim was told none of it."""
    tight, gt = _drop_sheet(11, 19, 11, 19, 0.05)        # 8 x 8, thick
    wide, gw = _drop_sheet(5, 25, 5, 25, 0.008)          # 20 x 20, thin
    assert abs(gt - gw) < gt * 0.05, \
        f"the same load hangs under both ({gt:.1f} g vs {gw:.1f} g)"
    assert wide > tight * 1.5, \
        f"the wider canopy must come down slower ({wide} vs {tight} ticks)"


def _chasm(policy_tag):
    """Two ledges with a drop between them, and a person on the near one."""
    from src.voxel.scenes import _person
    w = World(70, 20, 80, voxel_cm=5)
    w.fill(0, 70, 0, 20, 0, 1, STONE)               # the bottom, far below
    w.fill(2, 24, 0, 20, 1, 12, STONE)              # the near ledge
    w.fill(34, 68, 0, 20, 1, 12, STONE)             # the far one, 50 cm away
    w.exits = [(66, 10)]
    p = _person(w, 9, 10, z0=12)                    # standing on the near ledge
    w.policy = _Wants(legs=policy_tag)
    far, low = 0.0, 99
    for _ in range(120):
        w.step()
        on = np.argwhere(w.mat == FLESH)
        if len(on):                                 # standing somewhere, not
            far = max(far, float(on[:, 0].mean()))  # mid-flight
            if far > 34:
                low = min(low, int(on[:, 2].min()))
    return p, far, low


def test_a_person_can_LEAP_a_gap_they_cannot_WALK_across():
    """A jump that goes nowhere is barely a jump. Aimed, the body leaves the
    ground at the angle that carries furthest and at exactly the speed the
    distance needs — so it crosses a gap the legs cannot, and lands on the far
    side rather than hurling itself as hard as it can.

    The walker is the control, and it is not a strawman: the same body, the
    same ledges, told to walk. It stays on the near side, because a column of
    open air is not somewhere to stand."""
    walker, wfar, _wlow = _chasm("go:step")
    leaper, lfar, llow = _chasm("leap")
    assert wfar < 24, \
        f"the walker never crosses — open air is not somewhere to stand ({wfar:.1f})"
    assert lfar > 34, f"and the leaper lands on the FAR ledge ({lfar:.1f})"
    assert llow >= 12, f"on TOP of it, not fallen into the gap (z={llow})"
    assert leaper["alive"], "and it survived the landing"


def _lumps(mask):
    """How many separate connected pieces that matter is in."""
    lab = np.where(mask, -1, -2)
    n = 0
    while (lab == -1).any():
        lab[tuple(np.argwhere(lab == -1)[0])] = n
        while True:
            cur = lab == n
            g = cur.copy()
            g[1:] |= cur[:-1]; g[:-1] |= cur[1:]
            g[:, 1:] |= cur[:, :-1]; g[:, :-1] |= cur[:, 1:]
            g[:, :, 1:] |= cur[:, :, :-1]; g[:, :, :-1] |= cur[:, :, 1:]
            g &= (lab == -1) | cur
            if (g == cur).all():
                break
            lab[g & (lab == -1)] = n
        n += 1
    return n


def test_a_SWUNG_arm_breaks_GLASS_and_bounces_off_a_POST():
    """The test that settles animation versus physics — and it discriminates on
    MATERIAL, not just on motion.

    A pose is animation if it only changes the picture. There is no picture
    here: every law reads the lattice, so while the arm comes round it really is
    somewhere else, and what stops it is paid the rotational energy it had.
    Glass gives way at 0.2 kJ/m2 and a wooden post does not at 8.0, so the same
    fist at the same speed shatters one and marks neither the other nor itself.

    Nothing about the blow is authored. The muscle makes a torque that fades
    with speed (Hill), the limb's own inertia decides how fast it comes round,
    and 1/2 I omega-squared meets each target's own toughness. Nobody typed how
    long a swing takes, how fast it goes, or what it is strong enough to break."""
    from src.voxel.scenes import _person

    def strike(tag, target):
        w = World(30, 20, 44, voxel_cm=5)
        w.fill(0, 30, 0, 20, 0, 1, STONE)
        w.exits = [(28, 10)]
        p = _person(w, 9, 10)
        w.fill(17, 18, 10, 11, 1, 30, target)      # a post, at arm's height
        full = float(w.smass[w.mat == target].max())
        w.policy = _Wants(hands=tag, legs="stay")
        pieces = 1
        for _ in range(40):
            w.step()
            pieces = max(pieces, _lumps(w.mat == FLESH))
        hit = w.mat == target
        return (int((hit & (w.smass < 0.9 * full)).sum()),
                float(w.smass[hit].sum()), w, pieces)

    still, m_still, w0, _ = strike("keep", GLASS)
    swung, m_swung, w1, pieces = strike("swing", GLASS)
    wood, m_wood, _w2, _ = strike("swing", WOOD)
    assert still == 0, "a hand held at rest breaks nothing, not even glass"
    assert swung > 0, f"a swung one shatters glass ({swung} voxels)"
    assert wood == 0, \
        f"and the same swing does NOT break a wooden post ({wood} voxels) — " \
        f"a bare fist is not an axe, and the difference is the material"
    assert abs(m_still - m_swung) < 1.0, \
        f"breaking is not losing: every gram is still there " \
        f"({m_still:.0f} vs {m_swung:.0f} g)"
    assert any(r["tags"].get("hands") == "swing" for r in w1.traces), \
        "and swinging went through the menu like any other act"
    assert all(r["tags"].get("hands") != "swing" for r in w0.traces), \
        "while the other never chose it"
    # AND THE ARM STAYS ON. A limb is held at its joint; rasterising it wherever
    # the swing stopped left the far end nowhere near the body, and a lump of
    # flesh touching nothing is not an arm — measured, one person became three
    # pieces and left 18 voxels of hand on the floor.
    assert pieces == 1, \
        f"a person who swings is still ONE person afterwards ({pieces} pieces)"

def _on_a_ledge(pull_N, brace_N, ticks=70):
    """One man on a ledge 1.5 m up, one below with hold of his ankle, walking
    away. Everything about the two of them is identical except two numbers."""
    from src.voxel.scenes import _person, _Wants

    w = World(32, 12, 64, voxel_cm=5)
    w.fill(0, 32, 0, 12, 0, 1, STONE)                 # the ground
    w.fill(18, 32, 0, 12, 1, 31, STONE)               # the ledge, face at x=18
    w.exits = []
    a = _person(w, 14, 6, z0=1)                       # below
    a["name"], a["strength_N"], a["facing"] = "the puller", pull_N, (-1.0, 0.0)
    b = _person(w, 21, 6, z0=31)                      # above, at the lip
    b["name"], b["strength_N"] = "the mark", brace_N
    w.policy = _Wants(each={"the puller": {"hands": "take hold of the mark",
                                           "legs": "straight on"}})
    grams = _flesh_grams(w)
    for _ in range(ticks):
        w.step()
    comp, sl = w._person_cells(b)
    cells = np.argwhere(comp)
    return {"x": float(cells[:, 0].mean()) + sl[0].start,
            "z": int(cells[:, 2].min()),
            "grams": grams, "after": _flesh_grams(w), "w": w}


def test_a_BRACED_man_cannot_be_PULLED_off_a_LEDGE_and_a_WEAKER_one_can():
    """Can a man pull another off a ledge? It depends, and what it depends on is
    arithmetic — which is the whole claim.

    A body that is awake, alive and has something under its feet BRACES: it puts
    its own strength into the floor against whoever is pulling. That is one line,
    and three different stories come out of it without any of them being written
    down — an ordinary man cannot shift an equal, a stronger man can shift the
    same equal, and an ordinary man can shift someone weaker. The unconscious
    case every rescue in this suite depends on is the SAME line with the brace at
    zero, which is why it did not need its own rule either.

    Before this, taking hold was legal only on someone already unconscious. That
    was a case wearing a flag: it made rescue the only reason two people ever
    touched, and it made this question unaskable rather than merely hard. The
    force test also used to be answered ONCE, at the moment of grabbing, which
    said a body knows before touching you whether you can be moved; it is asked
    every tick now, because the answer changes when a man's heels leave the floor.

    Nothing carries him over the lip but the support law. He is walked off his
    own footing, hangs by his span, and goes."""
    braced = _on_a_ledge(pull_N=400.0, brace_N=400.0)
    hard = _on_a_ledge(pull_N=700.0, brace_N=400.0)
    weak = _on_a_ledge(pull_N=400.0, brace_N=120.0)

    assert braced["z"] == 31, \
        f"an ordinary man cannot drag an equal anywhere: the mark should still " \
        f"be stood on the ledge, and is at z={braced['z']}"
    assert abs(braced["x"] - 21.0) < 1.0, \
        f"...nor even shift him along it (x {braced['x']:.1f}, was 21)"
    assert hard["z"] <= 2, \
        f"a STRONGER man drags the same braced man off it (z={hard['z']})"
    assert weak["z"] <= 2, \
        f"and an ordinary man drags a WEAKER one off it (z={weak['z']}) — same " \
        f"line of arithmetic, other side of it"
    for got in (braced, hard, weak):
        assert abs(got["after"] - got["grams"]) < 1.0, \
            f"and nobody loses a gram falling ({got['grams']:.0f} -> " \
            f"{got['after']:.0f} g)"
    w = hard["w"]
    assert _lumps(w.mat == FLESH) == 2, \
        "two people went over the lip's worth of trouble and are still two whole " \
        "people, not a scatter of flesh"
    assert any(r["tags"].get("hands") == "hold" for r in w.traces), \
        "and taking hold went through the menu like any other act"
