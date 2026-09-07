"""The voxel core's promises — outcomes that must FALL OUT of the four laws,
with zero case-code anywhere (Ruling 1). If one of these breaks, a law is wrong,
not a flag missing."""
import numpy as np

from src.voxel.demo import build, dump_water, torch
from src.voxel.sim import (ACID, AIR, ASH, CHAR, FLESH, GLASS, IRON, LEAF, MIRON, MTIN, OIL, STONE, TIN, WATER, WEAK_ACID,
                           WOOD, World)


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
    for _ in range(8):
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
    for _ in range(6):
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
    for _ in range(200):                            # fall + rubble re-settling
        w.step()
    assert not w.bodies
    leaves = np.argwhere(w.mat == LEAF)
    assert float(w.smass[w.mat == LEAF].sum()) > 0.9 * leaf0, "no leaf mass lost"
    assert float(leaves[:, 0].mean()) > 9, "the canopy came DOWN AND OVER with the trunk"
    assert int(leaves[:, 2].max()) <= 8, "and lies low — not hovering at tree height"


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
    for _ in range(8):
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
        for _ in range(16):
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
        w = World(140, 16, 20, voxel_cm=5)
        w.fill(0, 140, 0, 16, 0, 1, STONE)
        w.exits = [(6, 7)]
        if wall:                                     # a thick baffle between them
            w.fill(60, 66, 0, 14, 1, 18, STONE)      # (leaves a south gap to walk)
        w.fill(128, 130, 6, 8, 1, 4, WOOD)
        w.fill(120, 121, 7, 8, 1, 15, FLESH, frac=0.9)
        pa = w.add_person(120, 7, "A")
        w.fill(40, 41, 7, 8, 1, 15, FLESH, frac=0.9)
        pb = w.add_person(40, 7, "B")
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
    row = w.traces[0]
    assert row["pick"] in row["menu"], "a pick must come FROM the menu"
    assert len(row["menu"]) > 1, "a menu of one is not a choice"
    assert row["by"] == "table", "the reflex table is today's policy"
    assert "flee" in row["menu"] and "stay" in row["menu"], \
        "standing pat is always an option, and must be OFFERED as one"
    assert "flee_answering" not in row["menu"], \
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
            return [o["key"] for o in menu].index("stay")

    w, p = _alarm_room(policy=Coward())
    for _ in range(600):
        w.step()
        if p["safe"]:
            break
    assert not p["safe"], "this policy never leaves, so nobody reaches the door"
    assert w.traces and all(r["pick"] == "stay" for r in w.traces)
    assert "flee" in w.traces[0]["menu"], "fleeing was OFFERED and passed over"
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
    assert row["menu"] == ["stay"], \
        "with no route out, fleeing is not on the menu at all"
    assert row["pick"] == "stay", \
        "and the table's answer being unavailable falls back to a legal one"
