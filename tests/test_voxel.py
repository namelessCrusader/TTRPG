"""The voxel core's promises — outcomes that must FALL OUT of the four laws,
with zero case-code anywhere (Ruling 1). If one of these breaks, a law is wrong,
not a flag missing."""
import numpy as np
import pytest

from src.voxel.demo import build, dump_water, torch
from src.voxel.sim import (ACID, AIR, ASH, BODY, CHAR, FLESH, GLASS, IRON, LEAD,
                           LEAF, MIRON, MTIN, O2_PER_L, OIL, STONE, TIN, WATER,
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
    fit, _start = w._fit_grid(cells, p)
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


def _dropped_onto_stone(h_m, legs="stay", ticks=200):
    """One person, h_m of air under their feet, a stone floor. What the fall
    does to them is the whole result."""
    from src.voxel.scenes import _person

    zn = 1 + int(round(h_m / 0.05))
    w = World(24, 12, zn + 40, voxel_cm=5)
    w.fill(0, 24, 0, 12, 0, 1, STONE)
    w.exits = []
    p = _person(w, 11, 6, z0=zn)
    w.policy = _Wants(legs=legs)
    grams = _flesh_grams(w)
    for _ in range(ticks):
        w.step()
    return p, grams, _flesh_grams(w)


def test_a_FALL_HURTS_and_the_height_decides_how_much():
    """A man dropped 1.5 m onto stone used to land whole and entirely
    unbothered: _land_body paid 1/2 m v² against what he STRUCK and nothing
    against him, so "pulled off a cliff" was a change of address. Tissue takes
    damage well below the toughness that tears it apart; landing energy beyond
    that threshold is now the body's WOUND — one integral, the same shape as
    burns, read against the same faint and death thresholds.

    And flesh DEFORMS rather than fragments. Before this, a big enough fall
    shattered a man's feet into debris and the rest of him walked away
    unhurt — the shatter branch ate exactly the energy the wound should have
    carried, so the harder the landing the less it hurt."""
    low, g0, g1 = _dropped_onto_stone(1.5)
    mid, m0, m1 = _dropped_onto_stone(3.0)
    high, h0, h1 = _dropped_onto_stone(6.0)
    assert low["hurt"] > 0, "1.5 m onto stone is not nothing"
    assert low["awake"] and low["alive"], "...but a man takes it bruised"
    assert not mid["awake"], "3 m knocks him out"
    assert mid["alive"], "...and no more than that"
    assert not high["alive"], "6 m onto stone, landed rigid, kills"
    assert low["hurt"] < mid["hurt"] < high["hurt"], \
        "the height decides, monotonically"
    for a, b in ((g0, g1), (m0, m1), (h0, h1)):
        assert abs(a - b) < 1.0, \
            f"flesh deforms, it does not fragment: every gram stays ({a:.0f} " \
            f"-> {b:.0f} g)"


def test_ROLLING_on_landing_spreads_the_blow_that_a_rigid_landing_takes_whole():
    """The same fall, the same body, the same floor — and one of them gets up.
    A landing is a momentum change: the force is the change divided by the
    TIME taken to make it, so a body that keeps moving and comes to rest over
    many ticks is struck far less hard than one that stops dead. That is the
    whole of rolling, and it is arithmetic rather than a rule about rolls.

    Measured in wounds: a rigid landing from 3 m knocks a body out; the same
    fall rolled through leaves it conscious and costs it less."""
    rigid, g0, g1 = _dropped_onto_stone(3.0)
    rolled, h0, h1 = _dropped_onto_stone(3.0, legs="roll")
    assert not rigid["awake"], "a rigid landing from 3 m knocks a body out"
    assert rolled["awake"], "the same fall rolled through leaves it conscious"
    assert rolled["hurt"] < rigid["hurt"], "rolling spreads the blow"
    for a, b in ((g0, g1), (h0, h1)):
        assert abs(a - b) < 1.0, \
            "and either way every gram of them is still there: spreading a " \
            "blow is not shedding it"

    # IT IS A CHOICE, made from MID-AIR — which used to be the one place in
    # this sim where nothing could be decided at all, because a flying body has
    # no cells on the lattice for the will layer to read, so it was skipped.
    assert any("tucks to roll" in e for e in rolled["events"]), \
        "the body chose it while falling, rather than it happening to them"
    assert not any("tucks to roll" in e for e in rigid["events"]), \
        "and the one that did not choose it did not get it"
    # AND IT IS NOT A GET-OUT-OF-JAIL CARD. Dividing a blow only helps while
    # the parts land under the threshold; past that the arithmetic runs out,
    # which is why the ceiling is measured rather than asserted to be absent.
    high, _, _ = _dropped_onto_stone(11.0, legs="roll", ticks=320)
    assert not high["alive"], \
        f"eleven metres kills you however well you land (hurt {high['hurt']:.2f})"
    mid, _, _ = _dropped_onto_stone(8.0, legs="roll", ticks=320)
    assert mid["alive"] and not mid["awake"], \
        "at eight it knocks you out and you live, where landing rigid kills"


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


def test_an_axe_SWUNG_bites_where_the_same_axe_PRESSED_does_not():
    """The same body, the same axe, the same tree. Leaning on it does nothing;
    swinging it takes a bite. A swing puts the body's force behind a MOVING
    mass, and what arrives is kinetic energy — the axe is taken up through the
    menu like any other act, carried to the fist (where the grip, not the
    floor, holds it), and swung WITH the arm, its mass slowing the swing by
    Hill's relation and its edge being what lands.

    The blade is sharp in the only way a 5 cm lattice can mean it: the OBJECT
    declares the area its edge concentrates a blow into (`fill(edge=...)`) —
    a fact about a manufactured thing, the same class as its density
    (Ruling 2 q1), because an edge is sub-voxel shape the lattice cannot draw.
    An earlier note here claimed sharpness came from the contact being a few
    voxels; a 5 cm contact is not sharp by any measure."""
    from src.voxel.scenes import _person

    class _Chopper:
        """Take something up first; after that, do `then` with the hands."""
        name = "chopper"

        def __init__(self, then):
            self.then = then

        def pick(self, sit, menu):
            want = None
            if sit["limb"] == "hands":
                want = self.then if sit["holding"] else "take"
            elif sit["limb"] == "legs":
                want = "stay"
            for i, o in enumerate(menu):
                if o["tag"] == want:
                    return i
            return 0

    def chop(then):
        w = World(40, 20, 40, voxel_cm=5)
        w.fill(0, 40, 0, 20, 0, 1, STONE)
        w.fill(24, 27, 9, 12, 1, 30, WOOD)             # the trunk
        w.exits = [(38, 10)]
        _person(w, 18, 10)
        w.fill(22, 24, 10, 11, 1, 2, IRON, edge=2e-4)  # an axe head on the
        w.policy = _Chopper(then)                      # ground, edge declared
        full = float(w.smass[w.mat == WOOD].max())
        grams = float(w.smass[w.mat == WOOD].sum())
        for _ in range(300):
            w.step()
        hit = w.mat == WOOD
        return (int((hit & (w.smass < 0.9 * full)).sum()),
                grams, float(w.smass[hit].sum()), w)

    pressed, g0, g1, _w0 = chop("keep")     # holds it, and leans
    hewn, h0, h1, w1 = chop("swing")
    assert pressed == 0, "leaning on an axe fells nothing"
    assert hewn > 0, f"swinging it takes a bite ({hewn} voxels chewed)"
    assert abs(g0 - g1) < 1.0 and abs(h0 - h1) < 1.0, \
        "chopping is breaking, not losing"
    assert any(r["tags"].get("hands") == "take" for r in w1.traces), \
        "taking the axe up went through the menu"
    assert any(r["tags"].get("hands") == "swing" for r in w1.traces), \
        "and so did every swing"


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


def test_a_GRIP_CARRIES_LOAD_a_man_can_be_HELD_over_the_drop():
    """A body pulled over an edge always fell, because support relaxes from
    the ground up THROUGH material and a hand is not material — nothing on the
    lattice could hold a hanging man. The grip is now an edge in the support
    graph: what a footed body holds, within its strength, hangs from it. So
    "left holding him over the drop" is a state of the world, not a foregone
    fall — and the same man, unheld, is a wound on the ground.

    The grip does not bear a load wholly above the holder's own crown (you
    hold things UP), which is why the braced-ledge test above still ends with
    its man on the ground: his puller stands BELOW him."""
    from src.voxel.scenes import _person, _Wants

    def over_the_drop(grab):
        w = World(40, 12, 90, voxel_cm=5)
        w.fill(0, 40, 0, 12, 0, 1, STONE)             # the ground
        w.fill(16, 40, 0, 12, 1, 41, STONE)           # a ledge, 2 m up
        w.exits = []
        a = _person(w, 20, 6, z0=41)
        a["name"] = "the holder"
        b = _person(w, 26, 6, z0=41)
        b["name"] = "the mark"
        if grab:
            w.policy = _Wants(each={"the holder":
                                    {"hands": "take hold of the mark"}})
        grams = _flesh_grams(w)
        for _ in range(40):
            w.step()
        w.fill(25, 32, 0, 12, 1, 41, AIR)             # the floor under the
        for _ in range(120):                          # mark's feet goes
            w.step()
        comp, sl = w._person_cells(b)
        cells = np.argwhere(comp)
        return {"z": int(cells[:, 2].min()) if len(cells) else -1,
                "b": b, "grams": grams, "after": _flesh_grams(w)}

    held = over_the_drop(True)
    dropped = over_the_drop(False)
    assert held["z"] == 41, \
        f"held, he HANGS at the lip instead of falling (z={held['z']})"
    assert held["b"]["hurt"] == 0.0 and held["b"]["awake"], \
        "hanging from a grip costs him nothing"
    assert dropped["z"] <= 2, \
        f"unheld, the same man is on the ground (z={dropped['z']})"
    assert dropped["b"]["hurt"] > 0.0, "and the fall was not free"
    for got in (held, dropped):
        assert abs(got["after"] - got["grams"]) < 1.0, \
            "either way, every gram of person is accounted for"


def test_a_HELD_THING_HANGS_from_the_fist_and_COMES_ALONG():
    """Grab an OBJECT, not just a person. A grip is a constraint, not a verb
    list: the held thing is carried to the hand, the grip (not the floor)
    holds it there — item 20's support edge, pointed at a stick — and it
    walks with the body as one kinematic unit, because what is held moves
    with the hand."""
    from src.voxel.scenes import _person, _Wants as _SceneWants

    w = World(60, 20, 40, voxel_cm=5)
    w.fill(0, 60, 0, 20, 0, 1, STONE)
    w.exits = []
    p = _person(w, 10, 10)
    p["facing"] = (1.0, 0.0)
    w.fill(14, 16, 10, 11, 1, 2, WOOD)            # a stick on the ground
    w.policy = _SceneWants(hands="hold", legs="straight on")
    grams = float(w.smass[w.mat == WOOD].sum())
    for _ in range(240):
        w.step()
    stick = np.argwhere(w.mat == WOOD)
    assert p.get("held"), "the grip survives the walk"
    assert len(stick), "and the stick is somewhere"
    assert float(stick[:, 0].mean()) > 22.0, \
        f"the stick came along (x {float(stick[:, 0].mean()):.1f}, was 14.5)"
    assert int(stick[:, 2].min()) > 10, \
        f"and it HANGS from the fist, not dragged along the floor " \
        f"(z={int(stick[:, 2].min())})"
    assert abs(float(w.smass[w.mat == WOOD].sum()) - grams) < 1.0


def test_a_flier_SCRAPES_PAST_a_wall_instead_of_stopping_on_it():
    """`hit > 0` was an arrival, which is right for the ground and wrong for
    the rock a falling body is grazing. What a wall takes is the sideways
    speed; what stops a fall is something underneath. A block thrown at a
    tall wall slides down its face and lands at the bottom — it does not
    re-rasterise mid-air where it happened to touch. (The sideways energy is
    absorbed by the wall unpaid: a stated softness that starts to matter once
    throwing is built.)"""
    w = World(40, 12, 40, voxel_cm=5)
    w.fill(0, 40, 0, 12, 0, 1, STONE)
    w.fill(20, 23, 0, 12, 1, 32, STONE)           # a tall wall
    w.fill(10, 12, 5, 7, 26, 28, WOOD)            # a block in mid-air
    cells = np.argwhere(w.mat == WOOD)
    grams = float(w.smass[w.mat == WOOD].sum())
    w._launch(cells, (6.0, 0.0, 0.0))             # thrown at the wall face
    first = None
    for _ in range(200):
        w.step()
        on = np.argwhere(w.mat == WOOD)
        if len(on) and first is None:
            first = int(on[:, 2].min())
    on = np.argwhere(w.mat == WOOD)
    assert first is not None and first <= 4, \
        f"a flier re-rasterises where it LANDS, not where it grazed (z={first})"
    assert int(on[:, 2].min()) <= 2, "it lies at the wall's base"
    assert abs(float(w.smass[w.mat == WOOD].sum()) - grams) < 1.0, \
        "and every gram arrived with it"


def test_WHAT_YOU_HOLD_UP_STANDS_ON_YOUR_FEET_TOO():
    """The other half of a grip being an edge in the support graph: the load
    hangs from the holder, so the holder answers for it. The weight that has to
    sit over his feet is his own PLUS whatever hangs from his fists.

    Here the arm never moves — both loads are held the same way, in the same
    hand, by the same man. Only the MASS differs, and that is enough. (Its twin,
    `test_the_SAME_WEIGHT_at_ARMS_LENGTH_takes_a_man_off_his_feet`, holds the
    mass still and moves the arm instead; between them they are the two halves
    of one moment.)

    How far back he can shift his own weight is read off his build — his own
    half-width — and not typed in, so a small man is taken over by a load a big
    one shrugs at. Without that term the sum says nobody may hold anything at
    arm's length, which is plainly false.

    THE WEIGHT COMES IN AT THE FIST. A thing hanging still pulls straight DOWN
    along the arm holding it; where its own mass sits is the WRIST's question,
    which `_grip_holds` asks separately. Summed at the load's own centre, a body
    trailing on the floor behind a hauler read as hanging out past his toes.

    AND WHAT YOU CARRY IS NOT WHAT CARRIES YOU. A man holding a block has his
    arm directly above it, so the block came back as part of his own FOOTPRINT —
    which made his base as wide as his reach and said he could never be
    overbalanced by anything he was strong enough to hold."""
    from src.voxel.scenes import _person, _Wants

    def hold_it(voxels):
        w = World(48, 12, 90, voxel_cm=5)
        w.fill(0, 48, 0, 12, 0, 1, STONE)             # the ground
        w.fill(16, 48, 0, 12, 1, 41, STONE)           # a broad shelf, 2 m up
        w.exits = []
        a = _person(w, 20, 6, z0=41)
        a["name"] = "the holder"
        a["strength_N"] = 4000.0      # a winch of a man on purpose: this is a
                                      # question about BALANCE, and lifting has
                                      # its own answer elsewhere
        w.fill(24, 24 + voxels, 5, 8, 41, 44, LEAD)   # a block of lead beside him
        w.policy = _Wants(each={"the holder":
                                {"hands": "take hold of the lead"}})
        grams = sum(float(w.smass[w.mat == m].sum()) for m in (FLESH, LEAD))
        kg = float(w.smass[w.mat == LEAD].sum()) / 1000.0
        tipped = 0
        for _ in range(45):
            w.step()
            tipped += len([e for e in a["events"] if "pulled off" in e])
            a["events"].clear()
        comp, _sl = w._person_cells(a)
        after = sum(float(w.smass[w.mat == m].sum()) for m in (FLESH, LEAD)) \
            + sum(float(b["masses"].sum()) for b in w.bodies)
        return {"kg": kg, "tipped": tipped, "grams": grams, "after": after,
                "up": comp is not None and comp.any()}

    light = hold_it(5)
    heavy = hold_it(10)

    assert 55.0 < light["kg"] < 75.0 and 115.0 < heavy["kg"] < 140.0, \
        f"two loads either side of what a 38 kg man can balance " \
        f"({light['kg']:.0f} kg and {heavy['kg']:.0f} kg)"
    assert light["tipped"] == 0, \
        f"he carries the lighter one and keeps his feet ({light['kg']:.0f} kg)"
    assert heavy["tipped"] >= 1, \
        f"the heavier one takes him off them ({heavy['kg']:.0f} kg) — same man, " \
        f"same hand, same ground; the only difference is the mass"
    for got in (light, heavy):
        assert abs(got["after"] - got["grams"]) < 1.0, \
            f"and either way every gram of man and metal is accounted for " \
            f"({got['grams']:.0f} -> {got['after']:.0f} g)"


def _compare_worlds(build, ticks, poke=None):
    """Two worlds started from ONE state — so how they were built cannot be the
    difference — stepped side by side, one computing every law over the whole
    world every tick and one skipping what provably cannot have changed.

    Compared EVERY tick, not just at the end, so a failure names the tick and
    the field it started on. That is the difference between knowing a
    divergence happened and being able to find it."""
    slow, fast = build(), build()
    fast.restore(slow.snapshot())            # identical starting state
    slow.skip_quiet, fast.skip_quiet = False, True
    for t in range(ticks):
        for w in (slow, fast):
            if poke is not None:
                poke(w, t)
            w.step()
        a, b = _state_of(slow), _state_of(fast)
        assert a.keys() == b.keys(), f"t{t}: different things exist"
        for k in a:
            diff = float(np.abs(a[k].astype(np.float64)
                                - b[k].astype(np.float64)).max())
            scale = max(float(np.abs(a[k]).max()), 1.0)
            if diff > 1e-4 * scale:
                return t, k, diff, scale
    return None


def _state_of(w):
    """Everything a law is allowed to touch, as plain arrays."""
    st = {"mat": w.mat.astype(np.int32), "smass": w.smass.copy(),
          "E": w.E.copy(), "fl": w.fl.astype(np.int32), "fvol": w.fvol.copy(),
          "fpot": w.fpot.copy(), "smoke": w.smoke.copy()}
    if w.o2 is not None:
        st["o2"] = w.o2.copy()
    for i, p in enumerate(w.persons):
        for k in ("blood_o2", "smoke", "burn", "hurt", "awake", "alive", "safe"):
            st[f"person{i}.{k}"] = np.array([float(p.get(k, 0.0))])
    return st


def test_ACTIVE_REGIONS_change_NOTHING():
    """THE GUARD, written before the optimisation it guards.

    Skipping work is only allowed to make the sim FASTER, never different. So
    every scene here is run twice — once with every law sweeping the whole world
    every tick, once with the skips on — and the two worlds have to agree, cell
    for cell, on everything a law is allowed to touch.

    The scenes are chosen for the ways a skip can be wrong rather than for
    variety. A fire is not local: it heats what it does not touch, so a room
    that looks quiet three metres away is not. Gas is not local either — it
    mixes room-wide, so an airspace is the smallest thing that can be called
    still. And a world that starts quiet and is disturbed LATER is the case
    where a skip becomes a bug, because something has to notice."""
    from src.voxel.scenes import _person

    def room():
        w = World(40, 30, 30, voxel_cm=5)
        w.open_sky = False
        w.fill(0, 40, 0, 30, 0, 1, STONE)
        w.fill(0, 40, 0, 30, 29, 30, STONE)
        for (x0, x1, y0, y1) in ((0, 1, 0, 30), (39, 40, 0, 30),
                                 (0, 40, 0, 1), (0, 40, 29, 30)):
            w.fill(x0, x1, y0, y1, 0, 30, STONE)
        w.fill(19, 21, 0, 30, 0, 30, STONE)          # a dividing wall...
        w.mat[19:21, 12:18, 1:12] = AIR              # ...with a doorway
        w.smass[19:21, 12:18, 1:12] = 0.0
        w.fill(5, 12, 12, 18, 1, 3, WOOD, frac=0.6)  # a crib of sticks
        w.exits = [(38, 15)]
        return w

    def quiet():
        return room()

    def lit(w, t):
        if t < 25:
            w.E[8, 15, 2] += 2500.0                  # a taper held to the crib

    def late(w, t):
        if 30 <= t < 55:
            w.E[8, 15, 2] += 2500.0                  # ...but only after a while

    def under_the_sky():
        """No roof, so smoke LEAVES at the top of the world — the one place a
        windowed gas law must know the difference between the edge of its box
        and the edge of the world."""
        w = World(30, 24, 34, voxel_cm=5)
        w.fill(0, 30, 0, 24, 0, 1, STONE)
        w.fill(6, 14, 9, 15, 1, 3, WOOD, frac=0.6)
        w.exits = [(28, 12)]
        return w

    def two_sealed_rooms():
        """Two rooms with no way between them, and a fire in one. The far room
        is an airspace that nothing has disturbed — the case an island rule is
        allowed to sleep through, and must not get wrong."""
        w = World(44, 22, 26, voxel_cm=5)
        w.open_sky = False
        w.fill(0, 44, 0, 22, 0, 26, STONE)
        for x0 in (2, 24):
            w.mat[x0:x0 + 18, 2:20, 1:18] = AIR
            w.smass[x0:x0 + 18, 2:20, 1:18] = 0.0
        w.fill(5, 13, 8, 14, 1, 3, WOOD, frac=0.6)        # fuel in the near one
        w.fill(28, 36, 8, 14, 1, 3, WOOD, frac=0.6)       # and in the far one
        w.exits = []
        return w

    def wet():
        w = room()
        for x in range(6, 11):
            w.pour(x, 15, 4, WATER, 300.0)           # a puddle over the sticks
        return w

    def peopled():
        w = room()
        p = _person(w, 30, 15, z0=1)
        p["name"] = "the witness"
        return w

    for tag, build, poke, ticks in (("a quiet room", quiet, None, 40),
                                    ("a fire from the first tick", room, lit, 60),
                                    ("a room disturbed LATE", room, late, 70),
                                    ("water over the fuel", wet, lit, 50),
                                    ("a sealed room next door", two_sealed_rooms,
                                     lambda w, t: w.E.__setitem__((8, 11, 2),
                                         w.E[8, 11, 2] + 2500.0) if t < 35 else None,
                                     70),
                                    ("a fire under the open sky", under_the_sky,
                                     lambda w, t: w.E.__setitem__((9, 12, 2),
                                         w.E[9, 12, 2] + 2500.0) if t < 30 else None,
                                     70),
                                    ("someone in the room", peopled, lit, 60)):
        split = _compare_worlds(build, ticks, poke)
        assert split is None, \
            f"{tag}: skipping work changed {split[1]} by {split[2]:.6g} " \
            f"(scale {split[3]:.6g}) at TICK {split[0]} — a skip may only be " \
            f"faster, never different"


def test_a_WORLD_can_be_PUT_BACK_exactly_as_it_was():
    """A world is a value. Copy it, play on, put the copy back, and the next
    tick is the tick that would have followed — not one like it.

    This is a tool rather than a physical claim, and it earns its place three
    times over. A guard that can start two runs from the SAME state removes how
    the world was built as a variable, which is what makes it safe to rewrite
    how the laws are applied at all. A table wants to rewind and take the other
    branch. And a harvest of decisions is only honest if the run behind it can
    be played again.

    The mind is deliberately not part of it: a policy may one day be a model
    whose weights dwarf the lattice, and it is not what the world IS."""
    from src.voxel.scenes import _person

    def burning_room_with_someone_in_it():
        w = World(30, 20, 40, voxel_cm=5)
        w.fill(0, 30, 0, 20, 0, 1, STONE)
        w.fill(8, 16, 8, 12, 1, 3, WOOD, frac=0.6)
        w.exits = [(28, 10)]
        _person(w, 22, 10, z0=1)["name"] = "the witness"
        return w

    w = burning_room_with_someone_in_it()
    for _ in range(25):
        w.E[10, 10, 2] += 2500.0
        w.step()

    keep = w.snapshot()
    marked = float(w.E.sum())

    def play(n):
        for _ in range(n):
            w.step()
        return _state_of(w)

    first = play(20)
    w.restore(keep)
    assert abs(float(w.E.sum()) - marked) < 1e-6, \
        "put back means put back: the world is where it was"
    second = play(20)
    for k in first:
        assert np.array_equal(first[k], second[k]), \
            f"the same twenty ticks from the same state gave a different {k} — " \
            f"either the copy was shallow or the sim is not deterministic"

    # AND THE COPY IS NOT A VIEW. A snapshot that shares its arrays with the
    # world is not a snapshot; it is a second name for the present.
    w.restore(keep)
    before = keep["E"].copy()
    for _ in range(10):
        w.E[10, 10, 2] += 2500.0
        w.step()
    assert np.array_equal(keep["E"], before), \
        "playing on did not disturb the copy"
    assert not np.array_equal(w.E, before), "...and playing on did something"
    w.restore(keep)
    assert np.array_equal(w.E, before), "and it can be put back more than once"


def test_the_FAST_PATH_and_the_PLAIN_ONE_agree():
    """The engine now has two ways to apply some laws — plain numpy, and the
    same arithmetic compiled — and only one of them is the DEFINITION.

    numpy is the definition. The compiled kernels exist to be quicker and are
    written to match it step for step: the same order of multiplications, the
    same two clip comparisons in the same direction, and the loss swept over
    every cell before the gain is, because a float32 sum reordered is a float32
    sum changed. So the two must agree EXACTLY, not nearly — a tolerance here
    would quietly license a second physics.

    Run on a world with everything in it at once: fire, fuel burning away, smoke
    filling a room, oxygen going, a person breathing it. Compared every tick,
    because agreeing at the end is not the same as agreeing.

    Skipped when numba is not installed, which is a supported way to run: the
    engine's only hard dependency is PyYAML, and without the accelerator nothing
    changes but the speed."""
    from src.voxel.sim import HAVE_NUMBA
    from src.voxel.scenes import _person
    if not HAVE_NUMBA:
        pytest.skip("numba is optional; the plain path is the whole engine")

    def smoky_room():
        w = World(34, 26, 32, voxel_cm=5)
        w.open_sky = False
        w.fill(0, 34, 0, 26, 0, 1, STONE)
        w.fill(0, 34, 0, 26, 31, 32, STONE)
        for (x0, x1, y0, y1) in ((0, 1, 0, 26), (33, 34, 0, 26),
                                 (0, 34, 0, 1), (0, 34, 25, 26)):
            w.fill(x0, x1, y0, y1, 0, 32, STONE)
        w.fill(5, 13, 10, 16, 1, 3, WOOD, frac=0.6)
        for x in range(14, 17):
            w.pour(x, 13, 4, WATER, 200.0)
        w.exits = [(32, 13)]
        _person(w, 26, 13, z0=1)["name"] = "the witness"
        return w

    plain, fast = smoky_room(), smoky_room()
    fast.restore(plain.snapshot())
    plain.fused, fast.fused = False, True
    for t in range(70):
        for w in (plain, fast):
            if t < 40:
                w.E[8, 13, 2] += 2500.0
            w.step()
        a, b = _state_of(plain), _state_of(fast)
        for k in a:
            assert np.array_equal(a[k], b[k]), (
                f"the compiled laws and the plain ones parted company on {k} "
                f"at TICK {t} — by "
                f"{float(np.abs(a[k].astype(np.float64) - b[k].astype(np.float64)).max()):.6g}. "
                f"numpy is the definition; the kernel has to match it, not "
                f"merely resemble it")


def test_an_arm_REACHES_OUT_and_STAYS_OUT_and_a_WALL_stops_it():
    """A body that can hold a pose. Until now a limb could only be flung — the
    swing promoted it off the lattice, turned it, and put it back at rest — so
    an arm could never simply BE somewhere. Three things were waiting on this:
    the arm that stays where a swing left it, a crouch, and a man leaning out
    to take hold of something beyond his toes.

    It is not animation, and the test is the same one that settles a swing:
    there is no picture here to change. The arm's voxels leave the cells they
    are in and arrive in others, so while it is out it really is out — its mass
    is there, its hand is there, and A WALL STOPS IT. A pose that could pass
    through stone would be a drawing.

    What it holds turns with it, because that is the same rotation about the
    same joint and most of what holding a thing is for.

    And the lattice gets a say. A limb one voxel wide can only be DRAWN at a
    handful of angles — at the rest, two of its voxels round into one cell and
    moving would destroy a voxel of flesh. So the angle runs on smoothly and the
    flesh catches up at the next angle that can be drawn, which is what a thin
    thing turning on a coarse grid is, not a workaround for it."""
    from src.voxel.scenes import _person, _Wants

    def reacher(wall_at=None):
        w = World(34, 16, 44, voxel_cm=5)
        w.fill(0, 34, 0, 16, 0, 1, STONE)
        if wall_at is not None:
            w.fill(wall_at, wall_at + 2, 0, 16, 1, 40, STONE)
        w.exits = []
        p = _person(w, 12, 8, z0=1)
        p["name"], p["facing"] = "the reacher", (1.0, 0.0)
        w.policy = _Wants(each={"the reacher": {"hands": "reach out",
                                                "legs": "stay"}})
        return w, p

    w, p = reacher()
    grams = _flesh_grams(w)
    rest = w._limb_cells(p, "right arm")
    rest_span = int(rest[:, 0].max())
    for _ in range(30):
        w.step()
    out = w._limb_cells(p, "right arm")
    assert out is not None and len(out) == len(rest), \
        f"the arm still has all of itself ({len(out)} voxels, was {len(rest)})"
    assert int(out[:, 0].max()) > rest_span + 3, \
        f"an arm put out REACHES: its far end went from x{rest_span} to " \
        f"x{int(out[:, 0].max())}"
    assert abs(_flesh_grams(w) - grams) < 1.0, \
        "and moving it cost the body nothing — every gram is still there"
    assert _lumps(w.mat == FLESH) == 1, "the arm is still attached to him"
    held = int(out[:, 0].max())
    for _ in range(40):                       # ...and it STAYS there
        w.step()
    assert int(w._limb_cells(p, "right arm")[:, 0].max()) == held, \
        "an arm held out stays out — that is what holding a pose means"
    assert any(r["tags"].get("hands") == "reach" for r in w.traces), \
        "and reaching went through the menu like any other act"

    # A WALL STOPS IT. Same person, same reach, one difference: there is stone
    # where the arm wants to be.
    w2, p2 = reacher(wall_at=19)
    stone_before = int((w2.mat == STONE).sum())
    for _ in range(30):
        w2.step()
    arm = w2._limb_cells(p2, "right arm")
    assert int(arm[:, 0].max()) < 19, \
        f"the arm stopped at the wall rather than reaching through it " \
        f"(its far end is x{int(arm[:, 0].max())}, the wall starts at x19)"
    assert int(arm[:, 0].max()) > int(rest[:, 0].max()) - 1, \
        "...having got as far as it could before the stone"
    assert int((w2.mat == STONE).sum()) == stone_before, \
        "and it did not take a bite out of the wall to get there"


def test_the_SAME_WEIGHT_at_ARMS_LENGTH_takes_a_man_off_his_feet():
    """What a lever is, with a person on one end of it.

    Everything here is identical twice over — the same man, the same block, the
    same grip, the same ground — except where he holds it. At his side the
    weight hangs almost over his own toes and he is fine. Put out on the end of
    an arm it hangs a good deal further, and the moment arithmetic that has
    always decided whether a leaning thing tips now decides it about HIM.

    Nobody wrote either outcome. It is the load's weight times how far out it
    hangs, against his weight times how far back he can put it — and how far
    back is read off his own build, not typed in.

    This is what a body that can hold a pose was FOR. Before it a limb could
    only be flung, so an arm could never simply be somewhere, and a lever a
    person makes with their own arm could not exist. The three ingredients had
    all been here for days and could not be put together.

    And the arm comes down when it happens: you cannot keep a thing at arm's
    length while it is taking you over. So he goes over once, rather than being
    thrown off his feet again every tick he spends back on them."""
    from src.voxel.scenes import _person, _Wants

    def carry(at_arms_length):
        w = World(52, 12, 90, voxel_cm=5)
        w.fill(0, 52, 0, 12, 0, 1, STONE)
        w.fill(16, 52, 0, 12, 1, 41, STONE)           # a broad shelf, 2 m up
        w.exits = []
        a = _person(w, 20, 6, z0=41)
        a["name"], a["facing"] = "the holder", (1.0, 0.0)
        a["strength_N"] = 4000.0      # a winch of a man on purpose: this is a
                                      # question about BALANCE, not about lifting
        w.fill(26, 29, 5, 8, 41, 44, LEAD)            # a block of lead beside him
        w.policy = _Wants(each={"the holder":
                                {"hands": "take hold of the lead"}})
        grams = sum(float(w.smass[w.mat == m].sum()) for m in (FLESH, LEAD))
        for _ in range(30):                           # he takes it up...
            w.step()
        load = w._held_cells(a)
        assert load is not None and int(load[:, 2].min()) > 45, \
            "he picked it up rather than leaving it on the floor to drag"
        kg = float(w.smass[tuple(load.T)].sum()) / 1000.0
        if at_arms_length:
            a.setdefault("reach", {})["right arm"] = float(np.pi / 2)
        tipped = 0
        for _ in range(70):
            w.step()
            tipped += len([e for e in a["events"] if "pulled off" in e])
            a["events"].clear()
        after = sum(float(w.smass[w.mat == m].sum()) for m in (FLESH, LEAD)) \
            + sum(float(b["masses"].sum()) for b in w.bodies)
        return {"kg": kg, "tipped": tipped, "grams": grams, "after": after}

    side = carry(False)
    out = carry(True)

    assert 30.0 < side["kg"] < 50.0, \
        f"a load a man can plainly lift and plainly not ignore ({side['kg']:.0f} kg)"
    assert side["tipped"] == 0, \
        "held at his side it hangs over his own feet and he keeps them"
    assert out["tipped"] == 1, \
        f"held out on an arm the same weight takes him off them, and does it " \
        f"ONCE ({out['tipped']} times) — the arm comes down with him"
    for got in (side, out):
        assert abs(got["after"] - got["grams"]) < 1.0, \
            f"and either way every gram of man and metal is accounted for " \
            f"({got['grams']:.0f} -> {got['after']:.0f} g)"


def test_a_body_REMEMBERS_WHERE_THE_FIRE_WAS_and_FORGETS_IT():
    """Belief used to record GEOMETRY ONLY — that a column had been laid eyes
    on, never what was in it. So a body could walk through a burning room and
    remember the shape of it and nothing whatever about the fire, and when it
    came to choose somewhere to go, every place it knew looked equally good.

    That is why an idle body would stroll toward a blaze. Not bravery: nothing
    it remembered said otherwise.

    And it FADES, which is a model and not a leak. A body that never forgets
    treats an hour-old fire as a fire; one that forgets at once has no memory
    to speak of. What is remembered is what was SEEN — by column, because light
    from a fire at your feet still reaches your eyes, and whether you could see
    that far is the occlusion the ray already settled."""
    from src.voxel.scenes import _person

    w = World(50, 26, 40, voxel_cm=5)
    w.open_sky = False
    w.fill(0, 50, 0, 26, 0, 40, STONE)
    w.mat[1:49, 1:25, 1:39] = AIR
    w.smass[1:49, 1:25, 1:39] = 0.0
    w.exits = [(46, 12)]
    w.fill(4, 10, 8, 16, 1, 3, WOOD, frac=0.8)
    w.E[4, 8, 1] = 9.0e5
    p = _person(w, 22, 12)
    p["name"] = "the witness"
    for _ in range(60):
        w.step()

    seen = p.get("danger")
    assert seen is not None and float(seen.max()) > 0.5, \
        "it remembers having seen a fire at all"
    lit = np.argwhere(seen > 0.5)
    assert 2 <= int(lit[:, 0].max()) <= 12 and 6 <= int(lit[:, 1].max()) <= 18, \
        f"and remembers WHERE — the fire is at x4-10 y8-16 and the memory sits " \
        f"at x{lit[:, 0].min()}-{lit[:, 0].max()} y{lit[:, 1].min()}-{lit[:, 1].max()}"
    assert float(seen[40, 20]) == 0.0, \
        "while the far corner, which was never alight, is remembered as fine"

    # AND IT FADES. Put the fire out and let the body look at the world again.
    w.mat[4:11, 8:17, 1:4] = AIR
    w.smass[4:11, 8:17, 1:4] = 0.0
    w.E[:] = 0.0
    was = float(seen.max())
    for _ in range(400):
        w.step()
    assert float(p["danger"].max()) < 0.5 * was, \
        f"a fire that is out stops being remembered as one " \
        f"({was:.2f} -> {float(p['danger'].max()):.2f})"


def test_the_TABLE_POLICY_PREFERS_the_way_that_is_not_past_a_fire():
    """The plainest taste there is, and the seam it lives in.

    The reflex table says WHAT to do — make for a door. Which door, and by
    which way, it has never had anything to say about, so the pick was the
    FIRST matching row and the answer came down to the order `_places` happened
    to build its list in. A policy that prefers nothing is not neutral; it is
    arbitrary, and arbitrary is not a thing a body does.

    Now every `go` row carries how bad the way there looks — the worst step on
    the route, out of what this body remembers seeing. The SIM states the fact;
    minding it is the policy's business, and a policy with no taste gets the
    same menu."""
    from src.voxel.sim import REFLEXES, TablePolicy
    pol = TablePolicy()
    sit = {"limb": "legs", "percept": "sees_fire", "reflexes": REFLEXES}
    menu = [{"key": "stay", "tag": "stay", "verb": "stay"},
            {"key": "go(the door west)", "tag": "go:exit", "verb": "go",
             "away": 4.0, "danger": 1.0},          # nearer, and past the fire
            {"key": "go(the door east)", "tag": "go:exit", "verb": "go",
             "away": 9.0, "danger": 0.0}]          # further, and clear
    assert pol.pick(sit, menu) == 2, \
        "it takes the longer way round rather than the way past the fire"

    menu[2]["danger"] = 1.0                        # both ways look as bad
    assert pol.pick(sit, menu) == 1, \
        "...and with nothing to choose between them on danger, the nearer one"

    idle = {"limb": "legs", "percept": None, "reflexes": REFLEXES}
    roam = [{"key": "stay", "tag": "stay", "verb": "stay"},
            {"key": "go(the floor west)", "tag": "go:roam", "verb": "go",
             "away": 3.0, "danger": 0.9},
            {"key": "go(the floor east)", "tag": "go:roam", "verb": "go",
             "away": 7.0, "danger": 0.0}]
    assert pol.pick(idle, roam) == 2, \
        "and an IDLE body does not wander toward a fire it remembers, which is " \
        "the whole of what was wrong: nothing it knew said otherwise"


def test_the_ATTENTION_CAP_never_hides_the_answer():
    """A capped menu can drop the right option INVISIBLY, and for a long time
    nobody knew whether it did. This measures it.

    `MENU_CAP` is a model of attention, not a budget for whatever is choosing:
    a person in a burning room weighs the door, the window and the child — not
    the forty places a full legality sweep would list. Modelling the limit is
    more true than pretending it is absent. But a limit that quietly removes the
    thing a body would have done is not a model of attention, it is a bug with
    a comment on it.

    RECALL is the measure: build the menu the body would have had with
    attention free, ask the SAME policy, and see whether it would have done
    something else. Measured at 81% before this — one decision in five, the cap
    was hiding the answer — because it kept the nearest rows and that cut every
    roam spot at once, so a body could not choose to wander however much it
    wanted to.

    What fixed it is a better model rather than a bigger cap: ONE OF EACH KIND
    first, then the nearest of what is left. A body notices categories before
    instances — that there is a door, that there is somewhere it has not seen,
    that there is a place it could go. And it keeps nothing because the policy
    would prefer it: salience is not preference, and a fire is worth noticing
    whether you mean to run at it or away."""
    from src.voxel.scenes import _person
    from src.voxel.sim import MENU_CAP

    def crowded_hall():
        w = World(90, 60, 40, voxel_cm=5)
        w.open_sky = False
        w.fill(0, 90, 0, 60, 0, 40, STONE)
        w.mat[1:89, 1:59, 1:39] = AIR
        w.smass[1:89, 1:59, 1:39] = 0.0
        w.exits = [(88, 12), (88, 30), (88, 48), (1, 12), (1, 48), (45, 58)]
        w.fill(10, 18, 20, 30, 1, 3, WOOD, frac=0.8)      # and a fire in it
        w.E[10, 20, 1] = 9.0e5
        for n, (x, y) in enumerate(((30, 10), (60, 40), (20, 50), (70, 12))):
            _person(w, x, y)["name"] = f"P{n}"
        w.recall_check = True
        return w

    w = crowded_hall()
    for _ in range(200):
        w.step()

    offered, capped, kept, checked = 0, 0, 0, 0
    for r in w.traces:
        for limb, n in (r.get("offered") or {}).items():
            offered = max(offered, n)
            if n > len(r["menus"].get(limb, [])):
                capped += 1
        for limb, key in (r.get("would") or {}).items():
            checked += 1
            kept += key == r["pick"][limb]

    assert offered > MENU_CAP, \
        f"the hall really does offer more than a body can weigh at once " \
        f"({offered} against a cap of {MENU_CAP}) — otherwise this measures nothing"
    assert capped > 0 and checked > 0, \
        "and the cap really does bite, so recall is a question about something"
    assert kept == checked, \
        f"attention narrows what a body weighs; it must never remove what the " \
        f"body would have DONE — recall {kept}/{checked} " \
        f"({100 * kept / max(checked, 1):.0f}%)"


def test_a_body_PLANS_OVER_THE_FLOOR_IT_SAW_and_finds_out_by_going():
    """Belief said WHERE a body had looked; the floor itself was read live.

    So the route planner was still the sim wearing a person's face, one layer
    below the door register and much better hidden. Wall off a corridor two
    rooms away, behind a baffle, with nobody within sight of it, and every body
    in the building re-planned around it on the very next tick — without one of
    them turning its head.

    Now a body plans over the floor AS IT LAST SAW IT. It keeps walking toward
    a door it cannot reach, all the way across two rooms, and finds out when it
    gets there. That is not a body being stupid; that is the only way a body
    could possibly know."""
    w, p = _two_room_house(True)                 # a resident: knows the place
    for _ in range(8):
        w.step()
        p["events"].clear()

    def exit_rows():
        cells = np.argwhere(w.mat == FLESH)
        fit, _ = w._fit_grid(cells, p)
        routes = w._routes(cells, [pl["xy"] for pl in w._places(p, cells, fit)],
                           p, fit=fit)
        return [pl["label"] for pl in w._places(p, cells, fit)
                if pl["tag"] == "go:exit" and routes.get(pl["xy"])]

    assert exit_rows(), "the resident starts with a way out it can name"
    # SEAL THE FAR ROOM, behind the baffle, two rooms off and out of any
    # sightline. Nothing about this reaches the body's eyes.
    w.fill(50, 52, 20, 25, 1, 21, STONE)
    cells = np.argwhere(w.mat == FLESH)
    assert not w._walkable(cells)[51, 22], "the world really is shut"
    assert p["free"][50, 22], \
        "and the body has no way of knowing it — nobody looked"
    assert exit_rows(), \
        "so it still aims at the door, because the floor it remembers still " \
        "runs there; a body that re-planned here would be reading the lattice"

    committed, learned = 0, None
    for t in range(400):
        w.step()
        p["events"].clear()
        if exit_rows():
            committed += 1
        elif learned is None:
            learned = t
            break
    assert learned is not None, \
        "and it does find out — by walking there and looking at the wall"
    assert committed > 40, \
        f"finding out takes crossing two rooms, not a tick ({committed})"
    assert not p["free"][50, 22], "what it found out, it now believes"
    assert p["anchor"][0] > 40, \
        "and it found out THERE: it had to get within sight of the wall"


def test_WHAT_YOU_WALK_INTO_you_learn():
    """A body that remembers a way as clear will plan it again the moment it
    replans, and grind against the thing in it forever. Walking into something
    is a percept of its own: the shin is a sense organ. Belief is corrected
    where the step failed, so the next plan goes round."""
    w = World(40, 20, 14, voxel_cm=5)
    w.open_sky = False
    w.fill(0, 40, 0, 20, 0, 14, STONE)
    w.mat[1:39, 1:19, 1:13] = AIR
    w.smass[1:39, 1:19, 1:13] = 0.0
    w.exits = [(37, 10)]
    w.fill(4, 5, 9, 10, 1, 10, FLESH, frac=0.9)
    p = w.add_person(4, 9, "X", knows_world=True)
    for _ in range(6):
        w.step()
        p["events"].clear()
    # a pillar goes up right in front of the face, in the dark: belief keeps
    # saying "clear" until something says otherwise
    w.fill(7, 9, 7, 12, 1, 13, STONE)
    p["free"][7:9, 7:12] = True                 # it has not seen this
    before = int(p["free"][7:9, 7:12].sum())
    cells = np.argwhere(w.mat == FLESH)
    p["known"][:] = True
    p["_path"] = w._plan_path(cells, (37, 10), p)
    assert p["_path"], "it plans straight through, because that is what it knows"
    for _ in range(40):
        w.step()
        p["events"].clear()
    after = int(p["free"][7:9, 7:12].sum())
    assert after < before, \
        f"it walked into the pillar and now knows it is there ({before} -> {after})"


def test_a_body_DOES_NOT_WALL_ITSELF_IN_behind_its_own_back():
    """The first thing a remembered floor gets wrong is the looker.

    Flesh is denser than anything a body can shove through, so every look wrote
    the ground under its own feet down as blocked — and then it walked on, and
    nothing ever looked back to correct it. Measured on an empty hall: 109
    phantom walls laid along its own path, every one of them believed. The
    planner already knows a body is not an obstacle to itself; belief has to
    know it at the moment of looking, or the map is poisoned by the one thing
    guaranteed to be in front of the eyes."""
    from src.voxel.scenes import _person
    w = World(80, 40, 40, voxel_cm=5)
    w.open_sky = False
    w.fill(0, 80, 0, 40, 0, 40, STONE)
    w.mat[1:79, 1:39, 1:39] = AIR
    w.smass[1:79, 1:39, 1:39] = 0.0
    w.exits = [(77, 20)]
    p = _person(w, 8, 20)
    start = p["anchor"]
    for _ in range(160):
        w.step()
        p["events"].clear()
    assert max(abs(p["anchor"][0] - start[0]),
               abs(p["anchor"][1] - start[1])) > 10, "it went somewhere"
    cells = np.argwhere(w.mat == FLESH)
    truth = w._walkable(cells)
    phantom = int((truth & ~p["free"] & p["known"]).sum())
    assert phantom == 0, \
        f"it believes in {phantom} walls that are open floor, and it made " \
        f"every one of them by standing there"


def _reach_set(w, p, snap, elbow, grid=0.1):
    """Every cell the HAND can be put in, over the whole range of the joints."""
    out = set()
    span = np.arange(-1.8, 1.8, grid)
    for sh in span:
        for el in (span if elbow else [0.0]):
            w.restore(snap)
            q = w.persons[0]
            if w._repose(q, "right arm", [float(sh), float(el)]) != "moved":
                continue
            fore = w._limb_parts(q, "right arm", None)[-1]
            if fore is None or not len(fore):
                continue
            d = np.abs(fore[:, 0] - 13) + np.abs(fore[:, 2] - 25)
            out.add(tuple(int(v) for v in fore[np.argmax(d)]))
    return out


def _reacher(wall_x=None):
    from src.voxel.scenes import _person, _Wants
    w = World(40, 16, 44, voxel_cm=5)
    w.fill(0, 40, 0, 16, 0, 1, STONE)
    w.exits = []
    p = _person(w, 10, 8, z0=1)
    p["name"], p["facing"] = "the reacher", (1.0, 0.0)
    if wall_x is not None:
        w.fill(wall_x, wall_x + 2, 0, 16, 1, 34, STONE)
    w.policy = _Wants(each={"the reacher": {"hands": "reach out",
                                            "legs": "stay"}})
    return w, p


def test_an_ELBOW_puts_the_hand_where_ONE_BONE_never_could():
    """What a second joint BUYS, counted rather than asserted.

    One bone has exactly one path to a place: the hand rides a circle about the
    shoulder, and on a 5 cm lattice almost every point of that circle rounds to
    something that is not a joined arm. So a one-boned man has TWO poses —
    hanging down, and straight out — and nothing in between is a pose at all.
    An elbow multiplies that severalfold, and the honest way to say so is to
    count the cells the hand can be put in."""
    w, p = _reacher()
    w.step()
    snap = w.snapshot()
    one = _reach_set(w, p, snap, elbow=False)
    two = _reach_set(w, p, snap, elbow=True)
    assert one <= two, "an elbow held straight is still an arm — it loses nothing"
    assert len(two) >= 3 * len(one), \
        f"a second joint should open a REGION, not a point or two " \
        f"({len(one)} places with one bone, {len(two)} with an elbow)"


def test_an_ARM_BENDS_round_what_it_cannot_reach_THROUGH():
    """Item 36, and the reason it was on the list: with one degree of freedom
    an arm that meets anything simply stops, because there is only ever one way
    to where it was going.

    A wall stands just past this man's elbow. His upper arm can get out; his
    forearm cannot go on into stone. So he puts the arm out and BENDS — hand
    as far forward as there is room for, every voxel of arm still on him, the
    arm still in one piece, and the wall untouched.

    Which way the bones get there is motor competence, not a decision: the
    CHOICE was "reach out", the same way choosing a door is a choice and
    knowing the way round the table is not."""
    for wall_x in (18, 19, 20):
        w, p = _reacher(wall_x)
        stone = int((w.mat == STONE).sum())
        rest = w._limb_cells(p, "right arm")
        n, x_rest = len(rest), int(rest[:, 0].max())
        for _ in range(60):
            w.step()
            p["events"].clear()
        arm = w._limb_cells(p, "right arm")
        pose = list(np.atleast_1d((p.get("pose") or {})["right arm"]))
        assert len(arm) == n, \
            f"every voxel of the arm is still on him ({len(arm)}, was {n})"
        assert _lumps(w.mat == FLESH) == 1, \
            "and it is still all ONE man — a bent arm that comes out in two " \
            "pieces has the right mass and is not an arm"
        assert int((w.mat == STONE).sum()) == stone, \
            "and it did not take a bite out of the wall to get there"
        assert abs(pose[1]) > 0.1, \
            f"the arm BENT rather than stopping dead (elbow {pose[1]:.2f} rad)"
        assert int(arm[:, 0].max()) > x_rest + 2, \
            f"and the hand got out past where it hung ({x_rest} -> " \
            f"{int(arm[:, 0].max())})"
        assert int(arm[:, 0].max()) < wall_x, "without reaching into the stone"

    # AND IT IS THE WORLD SAYING NO, not a liking for bent arms: held straight,
    # that same shoulder angle is refused.
    w, p = _reacher(18)
    w.step()
    assert w._repose(p, "right arm", [float(np.pi / 2), 0.0]) != "moved", \
        "a straight arm really cannot be put there — that is what it went round"


def test_a_POSE_that_rounds_a_limb_APART_is_not_a_POSE():
    """The twin of "two voxels must not round into one".

    A line of voxels turned to anything but a right angle rounds to a
    STAIRCASE, and a staircase touches only at its corners — which in a sim
    that decides what a THING is by 6-connectivity is not one limb, it is
    several. Measured before the check went in: an arm bent 1.2 rad came out
    in two pieces, every gram present, no longer an arm.

    So the lattice gets the same answer it always gave — the angle is not
    drawable, the flesh holds its last good pose, the angle runs on and the
    arm catches up when the two agree."""
    w, p = _reacher()
    w.step()
    snap = w.snapshot()
    tried = broke = 0
    for sh in np.arange(-1.6, 1.65, 0.1):
        for el in np.arange(-1.6, 1.65, 0.1):
            w.restore(snap)
            q = w.persons[0]
            if w._repose(q, "right arm", [float(sh), float(el)]) != "moved":
                continue
            tried += 1
            if _lumps(w.mat == FLESH) != 1:
                broke += 1
    assert tried > 0, "some poses are drawable, or this test proves nothing"
    assert broke == 0, \
        f"{broke} of {tried} accepted poses left the man in pieces"


def _waller(top=None):
    """A reacher with a waist-high wall in front of him — one his arm CLEARS
    once it is horizontal, and cannot get to without sweeping through."""
    from src.voxel.scenes import _person, _Wants
    w = World(40, 16, 44, voxel_cm=5)
    w.fill(0, 40, 0, 16, 0, 1, STONE)
    w.exits = []
    p = _person(w, 10, 8, z0=1)
    p["name"], p["facing"] = "the reacher", (1.0, 0.0)
    if top is not None:
        w.fill(15, 17, 0, 16, 1, top, STONE)      # standing on the floor
    w.policy = _Wants(each={"the reacher": {"hands": "reach out",
                                            "legs": "stay"}})
    return w, p


def test_an_ARM_CANNOT_SWEEP_THROUGH_what_it_would_CLEAR_at_the_end():
    """Only the pose at the END of a tick was ever checked.

    That is honest while a limb turns a little at a time, and stops being
    honest the moment it does not. The flesh waits at every angle the lattice
    cannot draw and then catches up several angles at once — so a wall sitting
    in the undrawable part of the sweep was never touched by anything. The arm
    was on one side of it, and then it was on the other, and the wall was
    unmarked because nothing had happened to it.

    A waist-high wall is exactly that shape: the arm CLEARS it once horizontal,
    so the final pose is legal, and the only way there is through the stone."""
    import src.voxel.sim as S
    w, p = _waller()                          # no wall: the reach works
    for _ in range(60):
        w.step()
        p["events"].clear()
    assert int(w._limb_cells(p, "right arm")[:, 0].max()) > 16, \
        "with nothing in the way the arm gets well past x16"

    for top in (20, 24, 26):
        w, p = _waller(top)
        stone = int((w.mat == STONE).sum())
        for _ in range(60):
            w.step()
            p["events"].clear()
        arm = w._limb_cells(p, "right arm")
        assert int(arm[:, 0].max()) < 15, \
            f"the arm is on ITS side of the wall (reached x" \
            f"{int(arm[:, 0].max())}, wall at x15)"
        assert int((w.mat == STONE).sum()) == stone, \
            "and the wall is whole, because nothing went through it"
        assert _lumps(w.mat == FLESH) == 1, "and he is still one man"

    # AND IT IS THE SWEEP DOING IT. Turn the sweep off and the same arm walks
    # straight through the same wall, which is what this was written for.
    was = S._SWEEP_RAD
    try:
        S._SWEEP_RAD = 99.0                   # no gap is ever bigger: never sweeps
        w, p = _waller(24)
        for _ in range(60):
            w.step()
            p["events"].clear()
        assert int(w._limb_cells(p, "right arm")[:, 0].max()) > 16, \
            "without the sweep the arm really did pass through the stone — " \
            "if this stops being true the test above proves nothing"
    finally:
        S._SWEEP_RAD = was


def test_a_REFUSED_REACH_is_not_asked_again_until_something_CHANGES():
    """A picked option must be one that can happen. That is the whole value of
    the menu seam, and a man pinned against a wall was breaking it: he decided
    to reach out, the world refused, the intention was dropped, and a few ticks
    later the same option was on the same menu again — for ever. Every one of
    those went into the trace as a decision that changed nothing, which is
    exactly the kind of row a harvest must never learn from.

    So a body remembers that it could not, against the spot it was standing on
    and the way it was facing — and forgets the moment either changes, because
    then it is a different question about a different wall."""
    from collections import Counter
    counts = {}
    for forget in (True, False):
        w, p = _waller(24)
        for _ in range(400):
            w.step()
            p["events"].clear()
            if forget:
                p.pop("_no_reach", None)      # no memory: the OLD behaviour
        picks = [r["tags"].get("hands") for r in w.traces
                 if r["who"] == "the reacher"]
        counts[forget] = (sum(1 for k in picks if k == "reach"), len(picks))
    old, new = counts[True], counts[False]
    assert old[0] == old[1] and old[1] > 5, \
        f"without the memory every single decision is the same refused reach " \
        f"({old[0]} of {old[1]}) — if that stops being true this proves nothing"
    assert new[0] <= 2, \
        f"it asks once, finds out, and stops asking ({new[0]} of {new[1]})"


def test_an_arm_LEFT_OUT_is_offered_the_way_BACK_IN():
    """The menu read the INTENTION — "is a reach wanted" — and called that
    "is the arm out". They part company the moment a reach is abandoned: the
    world said no, the wanting stopped, and the flesh is still out there. A man
    with his arm stuck half out was then offered the chance to reach out, and
    never once the chance to bring it down."""
    w, p = _reacher()
    for _ in range(30):
        w.step()
        p["events"].clear()
    arm = w._limb_cells(p, "right arm")
    assert int(arm[:, 0].max()) > 16, "the arm really is out"
    p["reach"] = {}                           # the wanting stops; the arm does not
    cells = np.argwhere(w.mat == FLESH)
    keys = [o["key"] for o in w._menu(p, cells, None, "hands", {})]
    assert "pull the arm back in" in keys, \
        f"an arm that is OUT can be brought in, whatever the body meant: {keys}"
    assert "reach out" not in keys, "and it is not asked to do what it has done"


def _leaner(ledge=False):
    from src.voxel.scenes import _person, _Wants
    w = World(44, 16, 50, voxel_cm=5)
    w.fill(0, 44, 0, 16, 0, 1, STONE)
    if ledge:
        w.fill(0, 18, 0, 16, 1, 30, STONE)        # a shelf; the drop is at x18
    w.exits = []
    p = _person(w, 12, 8, z0=30 if ledge else 1)
    p["name"], p["facing"] = "the leaner", (1.0, 0.0)
    w.policy = _Wants(each={"the leaner": {"waist": "lean out", "legs": "stay"}})
    return w, p


def test_a_SPINE_BENDS_rather_than_SWINGING():
    """A torso is a solid slab, and a rigid rotation of a solid slab is never
    injective on a grid: at EVERY angle some pair of its voxels rounds into one
    cell. So while a lean was a rotation, every lean was refused as undrawable
    and a body could not bend at all — the elbow's own machinery, working
    perfectly, saying no to everything.

    A lean is not one bone swinging. A trunk is a stack of vertebrae and
    bending it is each slice sliding forward over the one below — a SHEAR.
    Which is what the thing actually is, and which has the property the lattice
    needs for nothing: every row moves by one constant, so within a row it is a
    translation of integers, and rows never meet because their height does not
    change."""
    w, p = _leaner()
    w.step()
    snap = w.snapshot()
    grams = _flesh_grams(w)
    ok = 0
    for th in (0.1, 0.2, 0.3, 0.4, 0.6, 0.8):
        w.restore(snap)
        q = w.persons[0]
        assert w._repose(q, "lean", float(th)) == "moved", \
            f"a spine can be bent {th} rad — a rotation could not be bent at all"
        ok += 1
        assert _lumps(w.mat == FLESH) == 1, f"and he is one man at {th} rad"
        assert abs(_flesh_grams(w) - grams) < 1.0, f"and all there at {th} rad"
        head = w._limb_cells(q, "head")
        legs = w._limb_cells(q, "left leg")
        assert int(head[:, 0].mean()) > int(legs[:, 0].mean()), \
            "the head goes out over the toes and the legs stay standing"
    assert ok == 6


def test_a_body_LEANS_AS_FAR_AS_IT_CAN_STAND_and_no_further():
    """Balance was only ever asked about what a body was CARRYING. Nobody
    noticed while a body was one rigid block, because a block standing up
    straight has its weight over its feet by construction — give it a waist and
    the hole opens, and a man bent 46 degrees with his head fourteen voxels
    past his toes stood there indefinitely.

    Now a pose you cannot KEEP is not a pose you adopt, which is the gate the
    muscle already had with the other reason a body stops short. The angle is
    not written anywhere: it falls out of where this body's mass sits over this
    body's feet, so a heavier head or a longer foot would give a different one."""
    w, p = _leaner()
    for _ in range(150):
        w.step()
        p["events"].clear()
    # DRAWN, not posed: the angle is a motor command and runs on through every
    # angle the lattice cannot draw. `drawn` is where the flesh IS, and that is
    # the only number a test about a body's shape may believe.
    bent = float(np.atleast_1d((p.get("drawn") or {})["lean"])[0])
    assert 0.05 < bent < 0.3, \
        f"it leaned, and it stopped well short of folding in half ({bent:.2f} rad)"
    assert p["alive"] and p["awake"], "and it is still standing"
    own = np.argwhere(w.mat == FLESH)
    assert w._overbalanced(p, own) is None, \
        "it is holding a pose it can actually hold"

    # AND IT REALLY IS BALANCE DOING IT: put him past that angle by hand and
    # the floor stops being enough.
    w2, p2 = _leaner()
    w2.step()
    w2._repose(p2, "lean", 0.5)
    went = []
    for _ in range(40):
        w2.step()
        went += [e for e in p2["events"] if "overbalance" in e]
        p2["events"].clear()
    assert went, "bent past what he can stand, he goes over"


def test_LEANING_is_a_CHOICE_like_any_other():
    """A waist is a part of the body, so it gets a menu like every other part.
    The null act KEEPS what the body is doing — a trunk takes many ticks to
    bend and the body has to be able to go on bending — and straightening up is
    an act of its own, the same shape as hands keeping or letting go."""
    w, p = _leaner()
    cells = np.argwhere(w.mat == FLESH)     # BEFORE it has decided anything
    keys = [o["key"] for o in w._menu(p, cells, None, "waist", {})]
    assert "lean out" in keys and "stand as it is" in keys, keys
    for _ in range(40):
        w.step()
        p["events"].clear()
    assert any(r["tags"].get("waist") == "lean" for r in w.traces), \
        "leaning went through the menu like any other act"
    keys = [o["key"] for o in w._menu(p, np.argwhere(w.mat == FLESH), None,
                                      "waist", {})]
    assert "straighten up" in keys, \
        f"and a body that is bent can choose to stop being bent: {keys}"


def test_a_HAND_ON_THE_RAIL_lets_a_man_lean_out_over_the_LIP():
    """What a lean is FOR, and it is the same sum `_pulled_over` does read from
    the other end.

    A held thing that is not resting on anything hangs from you, and drags you
    over. A held thing that IS resting on something is holding itself up — so
    it can hold you too, and your base reaches your hand. Nothing new is
    declared for it: a grip was already an edge in the support graph, and
    balance already read the foot contact; they simply had never met.

    Same man, same back, same spine. The world is the only difference."""
    from src.voxel.scenes import _person, _Wants

    def ledge(rail):
        w = World(46, 16, 60, voxel_cm=5)
        w.fill(0, 46, 0, 16, 0, 1, STONE)
        w.fill(0, 20, 0, 16, 1, 30, STONE)        # the shelf; the drop is at x20
        w.exits = []
        p = _person(w, 13, 8, z0=30)
        p["name"], p["facing"] = "the leaner", (1.0, 0.0)
        want = {"waist": "lean out", "legs": "stay"}
        if rail:
            w.fill(18, 19, 6, 10, 30, 49, IRON)   # a post at the lip
            want["hands"] = "take hold of the iron"
        w.policy = _Wants(each={"the leaner": want})
        for _ in range(200):
            w.step()
            p["events"].clear()
        return w, p, float(np.atleast_1d(
            (p.get("drawn") or {}).get("lean", 0.0))[0])   # the FLESH, not the wish

    w_free, p_free, free = ledge(False)
    w_held, p_held, held = ledge(True)
    assert p_held.get("held"), "he took hold of the post"
    assert free < 0.3, \
        f"with nothing to hold he leans as far as his own toes allow ({free:.2f})"
    assert held > 1.5 * free, \
        f"with a hand on the post he leans far further out ({held:.2f} vs " \
        f"{free:.2f} rad)"
    # AND THE TWO ARE STOPPED BY DIFFERENT THINGS, which is worth knowing: the
    # free man is stopped by BALANCE and the holding man by the LATTICE — past
    # about 0.34 rad the shear tears a one-voxel-wide arm off its own shoulder
    # and `_still_joined` refuses it. So the second figure is a limit of the
    # voxel size (item 41), not of the man.
    own = np.argwhere(w_held.mat == FLESH)
    assert w_held._overbalanced(p_held, own) is None, \
        "the man holding on is not straining his balance at all — he has run " \
        "out of lattice, not out of grip"
    for w, p in ((w_free, p_free), (w_held, p_held)):
        assert p["alive"] and p["awake"], "and neither of them fell"
        assert _lumps(w.mat == FLESH) == 1, "and neither came apart"
    assert int(w_held._limb_cells(p_held, "head")[:, 0].mean()) > \
        int(w_free._limb_cells(p_free, "head")[:, 0].mean()), \
        "the one holding on has his head further out over the drop"


def test_a_body_USES_ITS_DOMINANT_HAND_and_the_other_when_that_is_the_one():
    """A body has two arms and a preference.

    The dominant one is stronger and better practised, so it is what a body
    reaches with — but it is a PREFERENCE, not a rule, and the other hand wins
    when the other hand is plainly the one for the job. That is a REASON rather
    than a die: a thing on your left is nearer your left hand, and past half a
    shoulder width that beats being right-handed. Which matters twice over,
    because the sim carries no die, and because "sometimes the other hand" is
    not a coin toss in real bodies either — it is where the thing is."""
    from src.voxel.scenes import _person
    for handed, other in (("right", "left"), ("left", "right")):
        w = World(40, 30, 50, voxel_cm=5)
        w.open_sky = False
        w.fill(0, 40, 0, 30, 0, 1, STONE)
        p = _person(w, 12, 15, handed=handed)
        own = np.argwhere(w.mat == FLESH)
        assert w._hand(p, own) == f"{handed} arm", \
            "with nothing to reach for, a body uses its good hand"
        near = w._fist_of(p, f"{other} arm", own)
        assert w._hand(p, own, toward=near) == f"{other} arm", \
            "a thing right by the other hand is taken with the other hand"
        far = w._fist_of(p, f"{handed} arm", own)
        assert w._hand(p, own, toward=far) == f"{handed} arm"
        assert w._hand_strength(p, f"{other} arm") < \
            w._hand_strength(p, f"{handed} arm"), \
            "and the other hand is weaker, which is what dominance MEANS"


def test_ONE_HAND_HOLDS_THE_POST_while_the_OTHER_REACHES():
    """The one thing a lean was built for, and it needed two working hands.

    Every act used to name `"right arm"` — seven places — so the left arm was
    flesh, mass, a lever and a fist that nothing could ever be done with, and a
    man could not hold on and reach at the same time. The arm is chosen now,
    and a hand that is already full is not a candidate: an indisposed hand is
    not a choice, it is an absence."""
    from src.voxel.scenes import _person, _Wants

    def rescue(handed):
        w = World(46, 16, 60, voxel_cm=5)
        w.fill(0, 46, 0, 16, 0, 1, STONE)
        w.fill(0, 20, 0, 16, 1, 30, STONE)        # the lip is at x20
        w.exits = []
        p = _person(w, 13, 8, z0=30, handed=handed)
        p["name"], p["facing"] = "the rescuer", (1.0, 0.0)
        w.fill(18, 19, 6, 10, 30, 49, IRON)       # a post at the lip
        w.policy = _Wants(each={"the rescuer": {"hands": "take hold of the iron",
                                                "waist": "lean out",
                                                "legs": "stay"}})
        for _ in range(60):
            w.step()
            p["events"].clear()
        w.policy = _Wants(each={"the rescuer": {"hands": "reach out",
                                                "waist": "lean out",
                                                "legs": "stay"}})
        for _ in range(80):
            w.step()
            p["events"].clear()
        return w, p

    for handed, other in (("right", "left"), ("left", "right")):
        w, p = rescue(handed)
        held = p.get("held") or {}
        assert held.get("label") == "iron", "he has hold of the post"
        assert held["arm"] == f"{handed} arm", \
            f"and took it with his good hand ({held['arm']})"
        posed = [k for k in (p.get("pose") or {}) if "arm" in k]
        assert posed == [f"{other} arm"], \
            f"while the OTHER arm is the one doing the reaching ({posed})"
        assert _lumps(w.mat == FLESH) == 1 and p["alive"] and p["awake"]


def test_a_BODY_STOPS_ASKING_for_a_pose_it_can_never_be_IN():
    """The angle outrunning the flesh is deliberate and right: it is how a limb
    crosses the angles the lattice cannot draw and catches up at the next one
    that can. What was missing was the end of that story.

    A lean has a long undrawable tail, so the command ran to its stop and the
    flesh stayed far behind it — a body whose angle read 45.8 degrees was bent
    19.3, for ever, and anything that believed the angle was wrong about the
    body. When the command has been given in full and the body is still not
    there, THAT is how far this joint goes here, and it settles to where it
    actually is."""
    w, p = _leaner()
    for _ in range(200):
        w.step()
        p["events"].clear()
    pose = float(np.atleast_1d((p.get("pose") or {})["lean"])[0])
    drawn = float(np.atleast_1d((p.get("drawn") or {})["lean"])[0])
    assert abs(pose - drawn) < 1e-9, \
        f"what the body is asking for and what it IS have to agree once it " \
        f"has stopped moving (asking {pose:.3f}, at {drawn:.3f})"
    assert drawn > 0.05, "and it did actually bend"


def test_what_LEAVES_THE_WORLD_is_counted_not_lost():
    """A lattice has edges, and a thing thrown past one is outside it.

    Those cells used to be CLAMPED — a voxel swung past the wall was set down
    ON the wall instead, several of them into the same column, and each write
    overwrote the last. So mass went missing, quietly, in the one field every
    test in the suite leans on. Clamping was never right anyway: it teleports
    matter to the edge and calls that a landing.

    A thing that leaves is gone, and saying so is what keeps "mass is
    conserved" a statement you can CHECK — it is on the lattice, or in a body
    in flight, or on the tally, and the three add up."""
    w = World(30, 12, 40, voxel_cm=5)
    w.fill(0, 30, 0, 12, 0, 1, STONE)
    w.fill(1, 3, 5, 8, 1, 34, WOOD, frac=0.8)     # tall, and right at the edge
    before = float(w.smass[w.mat == WOOD].sum())
    b = w._launch(np.argwhere(w.mat == WOOD), (0.0, 0.0, 0.0))
    b["fly"] = False                              # topple it OUT of the world
    b["axis"], b["s"] = 0, 1
    b["pivot"], b["zb"] = 2.0, 1.0
    b["theta"], b["omega"] = 0.0, 0.35
    b["Nm"], b["I"] = 0.0, 1.0
    for _ in range(40):
        w.step()
        if not w.bodies:
            break
    on = float(w.smass[w.mat == WOOD].sum())
    assert w.total_lost() > 0.0, \
        "some of it really did go over the edge — or this proves nothing"
    assert abs(on + w.total_lost() - before) < 1.0, \
        f"every gram is on the lattice or on the tally " \
        f"({on:.0f} + {w.total_lost():.0f} vs {before:.0f})"
    assert abs(w.total_wood() - before) < 1.0, \
        "and the world's own total says so without being asked twice"
    assert float(w.smass[0, :, :][w.mat[0, :, :] == WOOD].sum()) == 0.0, \
        "and nothing was teleported into the wall to make the sum work"


def test_MATTER_IN_THE_AIR_still_exists():
    """Matter in this sim lives in three places: cells, bodies that have left
    the grid to topple or fly, and the tally of what went past the edge of the
    world. A total that reads only the first is not a total — it says a swung
    axe has ceased to exist for as long as it is in the air.

    This had already gone wrong in the small. `total_wood` counted the tally
    and none of the others did, which is worse than none of them counting it:
    a conservation check that is right for one material and wrong for the rest
    fails at whichever moment is least convenient."""
    w = World(30, 12, 40, voxel_cm=5)
    w.fill(0, 30, 0, 12, 0, 1, STONE)
    w.fill(1, 3, 5, 8, 1, 34, WOOD, frac=0.8)
    before = w.total_wood()
    assert before > 0
    b = w._launch(np.argwhere(w.mat == WOOD), (0.0, 0.0, 0.0))
    assert float(w.smass[w.mat == WOOD].sum()) == 0.0, \
        "the lattice really is empty of it while it flies"
    assert abs(w.total_wood() - before) < 1.0, \
        "and it still weighs what it weighed, because it still exists"
    b["fly"] = False                              # topple it over the edge
    b["axis"], b["s"] = 0, 1
    b["pivot"], b["zb"] = 2.0, 1.0
    b["theta"], b["omega"] = 0.0, 0.35
    b["Nm"], b["I"] = 0.0, 1.0
    for _ in range(40):
        w.step()
        if not w.bodies:
            break
    assert w.total_lost() > 0.0, "some of it went over the edge"
    assert abs(w.total_wood() - before) < 1.0, \
        "and the one total covers all three places it can be"


def test_WHAT_YOU_DRAG_YOU_PAY_FOR():
    """The force arithmetic said a haul was legal and then charged nothing for
    it, so a man towing an unconscious body walked at exactly the pace of a man
    carrying nothing. Rescuing someone was free, and dragging a crate across a
    room was the same act as strolling across it.

    A body has only so much to put out. What the load takes, the legs do not
    get. The number is not written down anywhere — it is this load's friction
    against this body's strength, so a lighter load or a stronger man gives a
    different one, and the same load on ice would give another."""
    from src.voxel.scenes import _person, _Wants

    def race(load):
        w = World(70, 16, 44, voxel_cm=5)
        w.open_sky = False
        w.fill(0, 70, 0, 16, 0, 44, STONE)
        w.mat[1:69, 1:15, 1:40] = AIR
        w.smass[1:69, 1:15, 1:40] = 0.0
        w.exits = [(67, 8)]
        p = _person(w, 8, 8)
        p["name"], p["facing"] = "walker", (1.0, 0.0)
        p["strength_N"] = 1200.0
        want = {"legs": "go(the door", "waist": "stand"}
        if load:
            w.fill(12, 15, 6, 10, 1, 4, LEAD)
            want["hands"] = "take hold of the lead"
        w.policy = _Wants(each={"walker": want})
        x0 = None
        for t in range(900):
            w.step()
            p["events"].clear()
            own = np.argwhere(w.mat == FLESH)
            if not len(own):
                return None
            if x0 is None:
                x0 = float(own[:, 0].mean())
            if float(own[:, 0].mean()) - x0 > 30:
                return t
        return None

    free, laden = race(False), race(True)
    assert free is not None and laden is not None, \
        f"both of them cross the room ({free}, {laden})"
    assert laden > 1.4 * free, \
        f"hauling a load across a room takes markedly longer than walking it " \
        f"({laden} ticks against {free})"


def test_the_WORLD_SAYS_HOW_MUCH_HEAT_IT_SHEDS():
    """Every voxel radiates into a colder universe, and that is right — it is
    what stops a flame climbing for ever. But it was the one place in the sim
    where a conserved quantity changed and nothing wrote it down, so "energy is
    conserved" was not a statement anyone could CHECK.

    Found by accident, chasing friction heat that seemed to go missing: 1000 J
    left completely alone in a closed room is 779 J sixty ticks later. Nothing
    was wrong. Nothing could say so either."""
    w = World(20, 12, 20, voxel_cm=5)
    w.open_sky = False
    w.fill(0, 20, 0, 12, 0, 20, STONE)
    w.mat[1:19, 1:11, 1:16] = AIR
    w.smass[1:19, 1:11, 1:16] = 0.0
    w.fill(5, 8, 5, 8, 1, 3, IRON)
    w.E[5, 5, 1] += 1000.0
    before = w.total_energy()
    for _ in range(60):
        w.step()
    assert float(w.E.sum()) < before - 100.0, \
        "it really does shed heat — or this test is about nothing"
    assert w.shed > 0.0, "and it says how much"
    assert abs(w.total_energy() - before) < 1.0, \
        f"what is in the world plus what it has shed is what it started with " \
        f"({w.total_energy():.1f} against {before:.1f})"


def test_DRAGGING_A_THING_HEATS_IT_and_the_floor():
    """Work is force times distance and friction is a force, so a thing hauled
    over the ground has turned muscle into heat — there is nowhere else for it
    to have gone. Split between the thing and what it is dragged over, because
    a rubbing pair is two surfaces.

    And this needed NO NEW RULE: `E` is the field combustion already reads, so
    a thing dragged far enough over a rough floor gets hot, and a thing hot
    enough catches. Nobody wrote "dragging can start a fire"."""
    from src.voxel.scenes import _person, _Wants

    def haul(rubbing):
        w = World(90, 20, 44, voxel_cm=5)
        w.open_sky = False
        w.fill(0, 90, 0, 20, 0, 44, STONE)
        w.mat[1:89, 1:19, 1:40] = AIR
        w.smass[1:89, 1:19, 1:40] = 0.0
        w.exits = [(87, 10)]
        p = _person(w, 20, 10)
        p["name"], p["facing"] = "hauler", (1.0, 0.0)
        p["strength_N"] = 1500.0
        w.fill(14, 20, 7, 13, 1, 7, IRON)         # behind him, too heavy to lift
        w.policy = _Wants(each={"hauler": {"legs": "go(the door",
                                           "hands": "take hold of the iron",
                                           "waist": "stand"}})
        if not rubbing:
            w._rub = lambda *a, **k: None         # the same haul, no friction heat
        start = w.total_energy()
        for _ in range(400):
            w.step()
            p["events"].clear()
            if w._held_cells(p) is None:
                break
        return w.total_energy() - start

    with_rub, without = haul(True), haul(False)
    assert abs(without) < 1.0, "with no rubbing the world gains nothing"
    assert with_rub > 100.0, \
        f"dragging iron across a stone floor puts real joules into the world " \
        f"({with_rub:.0f} J)"


def test_a_LONG_FALL_arrives_at_the_speed_gravity_gives_it():
    """`FALL_SUBSTEPS` caps how many voxels a column may drop in one tick,
    because the thing it might land on has to be re-asked each time. The note
    against it said that capped falls at 8 m/s. It did not.

    It capped the DESCENT and left the speed running. A body that cannot fall
    as fast as gravity is pulling it spends LONGER falling, and gravity goes on
    adding to it the whole time — so a long drop arrived too FAST. Energy goes
    as v squared, so a 20 m fall landed with twice the blow it should have,
    which is the sort of error that makes every cliff in the sim a liar."""
    for h, tol in ((120, 0.08), (240, 0.08), (400, 0.10)):
        w = World(10, 10, h + 10, voxel_cm=5)
        w.open_sky = True
        w.fill(0, 10, 0, 10, 0, 1, STONE)
        w.fill(4, 6, 4, 6, h, h + 2, IRON)
        peak, landed = 0.0, False
        for _ in range(2000):
            w.step()
            peak = max(peak, float(w.vfall.max()))
            c = np.argwhere(w.mat == IRON)
            if len(c) and int(c[:, 2].min()) <= 1:
                landed = True
                break
        assert landed, "it reached the floor"
        drop_m = (h - 1) * 0.05
        ideal = np.sqrt(2 * 9.81 * drop_m)
        assert abs(peak - ideal) / ideal < tol, \
            f"a {drop_m:.0f} m drop arrives at {peak:.1f} m/s where gravity " \
            f"gives {ideal:.1f}"
