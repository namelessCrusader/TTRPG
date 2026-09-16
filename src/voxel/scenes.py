"""Stress scenarios — each aims at a suspected weak spot in the laws.

  two_rooms     two sealed rooms, one doorway, a wood-crib fire in room A.
                Does the fire breathe ONLY its own airspace? Does smoke walk
                through the doorway? (built to expose the old one-lung O2 bug)
  tree_fell     a tree chopped through at axe height. Exposes missing torque:
                the severed trunk falls STRAIGHT DOWN and lands standing.
  lamp_shelf    oil lamps under wooden shelves at 5/10/20 cm. Can a small flame
                ignite wood it does not touch — and where is the threshold?

Run:  python -m src.voxel.scenes <name> <frames_dir>
"""
import os
import sys

import numpy as np

from .sim import (ACID, AIR, BODY, CHAR, FLESH, GLASS, IRON, LEAD, LEAF, MIRON,
                  MLEAD, MTIN, O2_PER_L, OIL, STONE, TIN, WATER, WOOD, World)


def _save(w, frames_dir, t):
    drops = np.array([[d[0], d[1], d[2], d[7]] for d in w.drops], np.float32) \
        if w.drops else np.zeros((0, 4), np.float32)
    np.savez_compressed(
        os.path.join(frames_dir, f"f{t:04d}.npz"),
        mat=w.mat, fl=w.fl, fvol=w.fvol.astype(np.float16),
        fpot=w.fpot.astype(np.float16),
        T=w.T().astype(np.float16), burn=w.burning(),
        smoke=w.smoke.astype(np.float16), drops=drops, bodies=w.bodies_array())


# ── scenario: two rooms, one doorway ─────────────────────────────────────────
def two_rooms(frames_dir, ticks=800, every=10):
    w = World(120, 60, 44, voxel_cm=5)                   # 6 m x 3 m x 2.2 m, SEALED
    w.open_sky = False
    w.fill(0, 120, 0, 60, 0, 1, STONE)                   # floor
    w.fill(0, 120, 0, 60, 43, 44, STONE)                 # ceiling (hidden by renderer)
    for (x0, x1, y0, y1) in ((0, 1, 0, 60), (119, 120, 0, 60),
                             (0, 120, 0, 1), (0, 120, 59, 60)):
        w.fill(x0, x1, y0, y1, 0, 44, STONE)             # outer walls
    w.fill(59, 61, 0, 60, 0, 44, STONE)                  # dividing wall...
    w.mat[59:61, 26:39, 1:38] = AIR                      # ...with an open doorway
    w.smass[59:61, 26:39, 1:38] = 0.0
    for y0 in (18, 24, 30, 36, 42):                      # a wood crib in room A:
        w.fill(16, 40, y0, y0 + 2, 1, 3, WOOD, frac=0.6)     # sticks along x
    for x0 in (18, 24, 30, 36):
        w.fill(x0, x0 + 2, 16, 44, 3, 5, WOOD, frac=0.6)     # crossed layer along y

    def o2_frac(xs):
        m = (w.mat[xs, 1:59, 1:43] == AIR)
        return float(w.o2[xs, 1:59, 1:43][m].mean()) / (O2_PER_L * w.vox_l)

    for t in range(ticks):
        if t < 60:
            w.E[22, 24, 2] += 2500.0                     # a torch held to one stick
        w.step()
        if t % every == 0 or t == ticks - 1:
            _save(w, frames_dir, t)
        if t % 50 == 0:
            print(f"t{t}: burning {int(w.burning().sum())}, wood {w.total_wood():.0f} g, "
                  f"O2 roomA {100 * o2_frac(slice(1, 59)):.0f}% "
                  f"roomB {100 * o2_frac(slice(61, 119)):.0f}%", flush=True)


def _grow_tree(w, cx, cy, trunk_top=41, ball_z=46, r=9):
    """A tree with a SKELETON: trunk, wood branch spokes through the canopy (real
    leaves hang on twigs — without branches the canopy's underside is beyond
    leaf-span of the trunk and sheds immediately), and fresh sap-wet leaves."""
    w.fill(cx - 2, cx + 2, cy - 2, cy + 2, 1, trunk_top, WOOD)
    X, Y, Z = np.ogrid[:w.shape[0], :w.shape[1], :w.shape[2]]
    ball = (X - cx) ** 2 + (Y - cy) ** 2 + (Z - ball_z) ** 2 <= r * r
    ball &= w.mat == AIR
    w.mat[ball] = LEAF                                   # fresh canopy...
    w.smass[ball] = 200.0 * 0.4 * w.vox_l
    w.fl[ball] = WATER                                   # ...holding its own weight
    w.fvol[ball] = 0.06 * w.cap                          # of sap
    for (bx, by) in ((1, 0), (-1, 0), (0, 1), (0, -1),
                     (1, 1), (-1, -1), (1, -1), (-1, 1)):
        for t in range(1, r):                            # branch spokes
            px, py = cx + int(round(bx * t * 0.8)), cy + int(round(by * t * 0.8))
            pz = ball_z - 3 + t // 3
            if 0 <= px < w.shape[0] and 0 <= py < w.shape[1]:
                w.fill(px, px + 1, py, py + 1, pz, pz + 1, WOOD, frac=0.4)


# ── scenario: felling a tree ─────────────────────────────────────────────────
def tree_fell(frames_dir, ticks=160, every=3):
    w = World(40, 40, 70, voxel_cm=5)                    # 2 m x 2 m x 3.5 m, open sky
    w.fill(0, 40, 0, 40, 0, 1, STONE)                    # ground
    _grow_tree(w, 20, 20)

    for t in range(ticks):
        if t in (15, 30, 45, 60):                        # four axe strokes at knee height,
            k = (t // 15) - 1                            # each biting one slice deeper
            w.mat[21 - k, 18:22, 6:9] = AIR              # (from the camera side, so the
            w.smass[21 - k, 18:22, 6:9] = 0.0            # notch is visible in the clip)
            print(f"t{t}: axe stroke {k + 1}/4", flush=True)
        w.step()
        if t % every == 0 or t == ticks - 1:
            _save(w, frames_dir, t)
    top = int(np.argwhere(w.mat == WOOD)[:, 2].max())
    print(f"final: tree top at z={top} (was 54) — did it TOPPLE or just sink?", flush=True)


# ── scenario: oil lamps under shelves ────────────────────────────────────────
# (Probed first with a plain candle — 300-2200 J/tick of injected heat 15 cm below:
#  the shelf never passed 63 °C. A candle CAN'T torch a shelf here, which is fair.
#  So the flame is modeled as its cause: a lamp — a stone cup of actually-burning
#  oil. Three lamps, three gaps: the threshold is the story.)
def lamp_shelf(frames_dir, ticks=520, every=8):
    w = World(42, 40, 40, voxel_cm=5)                    # open sky
    w.fill(0, 42, 0, 40, 0, 1, STONE)
    lamps = []
    for x0, gap in ((2, 1), (15, 2), (28, 4)):           # flame 5 / 10 / 20 cm below
        w.fill(x0, x0 + 3, 16, 25, 1, 16, STONE)         # two piers
        w.fill(x0 + 9, x0 + 12, 16, 25, 1, 16, STONE)
        w.fill(x0, x0 + 12, 15, 26, 16, 18, WOOD, frac=0.8)   # the shelf segment
        cx, zc = x0 + 5, 16 - gap - 2
        w.fill(cx - 1, cx + 3, 18, 22, 1, zc + 2, STONE)      # stand + cup with rim
        w.mat[cx:cx + 2, 19:21, zc + 1] = AIR
        w.smass[cx:cx + 2, 19:21, zc + 1] = 0.0
        for (x, y) in ((cx, 19), (cx, 20), (cx + 1, 19), (cx + 1, 20)):
            w.pour(x, y, zc + 1, OIL, 60.0)              # 240 ml of lamp oil
        lamps.append((cx, zc + 1))
    for t in range(ticks):
        if t < 40:
            for cx, zf in lamps:
                w.E[cx, 19, zf] += 2500.0                # a taper lights all three
        w.step()
        if t % every == 0 or t == ticks - 1:
            _save(w, frames_dir, t)
        if t % 100 == 0:
            peaks = [f"{float(w.T()[x0:x0 + 12, 15:26, 16:18].max()):.0f}"
                     for x0 in (2, 15, 28)]
            print(f"t{t}: burning {int(w.burning().sum())}, oil {w.total_fluid(OIL):.0f} ml, "
                  f"shelf peaks °C (5/10/20 cm): {'/'.join(peaks)}", flush=True)


# ── scenario: the tree with a HINGE left ─────────────────────────────────────
def tree_hinge(frames_dir, ticks=120, every=3):
    """Same tree, but the axe leaves the far side: a 1-voxel hinge. The torque
    law's waist check finds the notch, and the tree FALLS TOWARD IT — the world
    is wide enough eastward for the whole trunk to land."""
    w = World(90, 40, 70, voxel_cm=5)
    w.fill(0, 90, 0, 40, 0, 1, STONE)
    _grow_tree(w, 20, 20)
    for t in range(ticks):
        if t in (15, 30, 45):                            # three strokes, camera side —
            k = (t // 15) - 1                            # the x=18 slice is LEFT standing
            w.mat[21 - k, 18:22, 6:9] = AIR
            w.smass[21 - k, 18:22, 6:9] = 0.0
            print(f"t{t}: axe stroke {k + 1}/3 (hinge left)", flush=True)
        w.step()
        if w.bodies or t % every == 0 or t == ticks - 1:     # every tick of the ARC
            _save(w, frames_dir, t)
    top = int(np.argwhere(w.mat == WOOD)[:, 2].max())
    hinge = int((w.mat[18, 18:22, 6:9] == WOOD).sum())
    print(f"final: tree top z={top}, hinge voxels intact {hinge}/12", flush=True)


# ── scenario: acid poured over a wood block ──────────────────────────────────
def acid_bath(frames_dir, ticks=500, every=8):
    w = World(30, 30, 30, voxel_cm=5)
    w.fill(0, 30, 0, 30, 0, 1, STONE)
    w.fill(10, 20, 10, 20, 1, 9, WOOD)                   # a 50 cm wood cube
    m0 = w.total_wood()
    for t in range(ticks):
        if t < 80:
            for (px, py) in ((13, 14), (15, 14), (14, 13), (14, 15)):
                w.pour(px, py, 20, ACID, 120.0)          # a CARBOY upended over it
        w.step()
        if t % every == 0 or t == ticks - 1:
            _save(w, frames_dir, t)
        if t % 100 == 0:
            print(f"t{t}: wood {w.total_wood():.0f}/{m0:.0f} g, "
                  f"acid {w.total_fluid(ACID):.0f} ml, fumes {w.smoke.sum():.0f} g", flush=True)


# ── scenario: a real torch vs a bare stick ───────────────────────────────────
def torch_pillar(frames_dir, ticks=600, every=10):
    """The user's question: why does a torch burn only at its top? Answer to test:
    because the top is the FUEL (oil-soaked), heat rises away from the handle, and
    wood's ignition is higher than oil's. Left: stick with an oil-soaked head.
    Right: bare stick with the taper held to its top wood directly."""
    w = World(40, 40, 36, voxel_cm=5)
    w.fill(0, 40, 0, 40, 0, 1, STONE)
    w.fill(9, 11, 19, 21, 1, 16, WOOD, frac=0.5)         # torch handle
    w.fl[9:11, 19:21, 14:16] = OIL                       # the head is SOAKED — oil
    w.fvol[9:11, 19:21, 14:16] = 50.0                    # held in the wood's fibers
                                                         # (like sap in a leaf), so it
                                                         # burns in place instead of
                                                         # dribbling off the top
    w.fill(29, 31, 19, 21, 1, 16, WOOD, frac=0.5)        # bare stick
    for t in range(ticks):
        if t < 30:
            w.E[9, 19, 15] += 2000.0                     # taper to the soaked head
            w.E[29, 19, 15] += 2000.0                    # taper to the bare stick's top
        w.step()
        if t % every == 0 or t == ticks - 1:
            _save(w, frames_dir, t)
        if t % 100 == 0:
            burn = w.burning()
            handle = float(w.smass[9:11, 19:21, 1:15].sum())
            stick = float(w.smass[29:31, 19:21, 1:15].sum())
            air = w.mat == AIR
            o2pc = 100 * float(w.o2[air].mean()) / (O2_PER_L * w.vox_l) if w.o2 is not None and air.any() else 100.0
            print(f"t{t}: oil {w.total_fluid(OIL):.0f} ml, torch handle {handle:.0f} g, "
                  f"bare stick {stick:.0f} g, burning wood {int((burn & (w.mat == WOOD)).sum())}, "
                  f"room O2 {o2pc:.0f}%", flush=True)


# ── scenario: communicating vessels ──────────────────────────────────────────
def vessels(frames_dir, ticks=220, every=5):
    w = World(30, 10, 20, voxel_cm=10)
    w.fill(0, 30, 0, 10, 0, 1, STONE)
    w.fill(1, 9, 1, 9, 1, 17, GLASS)                     # tank A, GLASS...
    w.mat[2:8, 2:8, 1:17] = AIR                          # ...hollowed
    w.smass[2:8, 2:8, 1:17] = 0.0
    w.fill(21, 29, 1, 9, 1, 17, GLASS)                   # tank B likewise
    w.mat[22:28, 2:8, 1:17] = AIR
    w.smass[22:28, 2:8, 1:17] = 0.0
    w.fill(8, 22, 3, 7, 1, 4, GLASS)                     # the connecting pipe...
    w.mat[7:23, 4:6, 1:3] = AIR                          # ...bored through both walls
    w.smass[7:23, 4:6, 1:3] = 0.0
    for x in range(2, 8):                                # tank A filled high
        for y in range(2, 8):
            for z in range(1, 13):
                w.pour(x, y, z, WATER, w.cap)
    def level(xs):
        zs = np.argwhere(w.fvol[xs, 2:8, :] > 0.3 * w.cap)
        return int(zs[:, 2].max()) if len(zs) else 0
    for t in range(ticks):
        w.step()
        if t % every == 0 or t == ticks - 1:
            _save(w, frames_dir, t)
        if t % 40 == 0:
            print(f"t{t}: level A z={level(slice(2, 8))}, level B z={level(slice(22, 28))}",
                  flush=True)


# ── scenario: blocks thrown in a pond ────────────────────────────────────────
def pond(frames_dir, ticks=40, every=2):
    w = World(24, 14, 16, voxel_cm=10)
    w.fill(0, 24, 0, 14, 0, 1, STONE)
    w.fill(1, 23, 1, 13, 1, 5, STONE)                    # basin...
    w.mat[2:22, 2:12, 1:5] = AIR                         # ...hollow
    w.smass[2:22, 2:12, 1:5] = 0.0
    for x in range(2, 22):
        for y in range(2, 12):
            for z in range(1, 4):
                w.pour(x, y, z, WATER, w.cap)            # 30 cm of water
    w.fill(5, 8, 5, 8, 10, 13, WOOD)                     # wood block, dropped
    w.fill(15, 18, 5, 8, 10, 13, STONE)                  # stone block, dropped
    for t in range(ticks):
        w.step()
        if t < 16 or t % every == 0 or t == ticks - 1:   # every tick through the
            _save(w, frames_dir, t)                      # splash — it is FAST
    for name, xs in (("wood", slice(5, 8)), ("stone", slice(15, 18))):
        mat = WOOD if name == "wood" else STONE
        zs = np.argwhere(w.mat[xs, 5:8, :] == mat)
        print(f"{name} block rests at z={int(zs[:, 2].min())}..{int(zs[:, 2].max())} "
              f"(water surface z=3)", flush=True)


# ── scenario: burning a living tree ──────────────────────────────────────────
def burning_tree(frames_dir, ticks=900, every=8):
    """A bonfire at the base of a FRESH tree. The trunk catches; fire climbs;
    the green canopy refuses to burn until its sap cooks off (leaves turning
    brown in the render as they dry); then it crowns. If the trunk burns
    through, the torque law fells the burning tree."""
    w = World(60, 60, 78, voxel_cm=5)
    w.fill(0, 60, 0, 60, 0, 1, STONE)
    _grow_tree(w, 30, 30)
    for (px, py) in ((27, 30), (33, 30), (30, 27), (30, 33)):
        w.pour(px, py, 1, OIL, 120.0)                    # kindling oil at the base
    leaf0 = None
    for t in range(ticks):
        if t < 40:
            w.E[27, 30, 1] += 2500.0                     # the arsonist's torch
        w.step()
        if leaf0 is None:
            leaf0 = float(w.smass[w.mat == LEAF].sum())
        if w.bodies or t % every == 0 or t == ticks - 1:
            _save(w, frames_dir, t)
        if t % 100 == 0:
            lm = w.mat == LEAF
            sap = float(w.fvol[lm].sum())
            print(f"t{t}: burning {int(w.burning().sum())}, "
                  f"leaves {float(w.smass[lm].sum()):.0f}/{leaf0:.0f} g, "
                  f"sap {sap:.0f} ml, wood {w.total_wood():.0f} g", flush=True)


# ── scenario: the alchemist's accident ───────────────────────────────────────
def _vat(w, x0, y0, zb):
    """A glass vat, open-topped, holding a liter of vitriol."""
    w.fill(x0, x0 + 4, y0, y0 + 4, zb, zb + 1, GLASS)        # base
    w.fill(x0, x0 + 4, y0, y0 + 4, zb + 1, zb + 4, GLASS)    # walls...
    w.mat[x0 + 1:x0 + 3, y0 + 1:y0 + 3, zb + 1:zb + 4] = AIR
    w.smass[x0 + 1:x0 + 3, y0 + 1:y0 + 3, zb + 1:zb + 4] = 0.0
    for xx in range(x0 + 1, x0 + 3):                         # ...cavity, part-filled
        for yy in range(y0 + 1, y0 + 3):
            for zz in (zb + 1, zb + 2):
                w.pour(xx, yy, zz, ACID, w.cap)


def drop_test(frames_dir, ticks=80, every=1):
    """HOW GLASS BREAKS, side by side: a water-filled glass vat and a solid
    wood block are let go above the same wooden shelf. Both fall by the support
    law, both cash in m·g·h on landing — the vat's cells exceed glass's
    toughness and it BURSTS (shards scatter, the water spills and pours off
    the shelf); the wood block's cells don't come close, and it just thuds."""
    w = World(44, 24, 32, voxel_cm=5)
    w.fill(0, 44, 0, 24, 0, 1, STONE)
    w.fill(6, 38, 8, 16, 10, 11, WOOD, frac=0.8)             # the shelf board
    for (lx, ly) in ((7, 9), (7, 13), (35, 9), (35, 13)):
        w.fill(lx, lx + 2, ly, ly + 2, 1, 10, WOOD)          # its legs
    w.fill(12, 16, 10, 14, 24, 25, GLASS)                    # vat base, high in the air
    w.fill(12, 16, 10, 14, 25, 28, GLASS)                    # vat walls...
    w.mat[13:15, 11:13, 25:28] = AIR
    w.smass[13:15, 11:13, 25:28] = 0.0
    for xx in range(13, 15):                                 # ...holding water
        for yy in range(11, 13):
            for zz in (25, 26):
                w.pour(xx, yy, zz, WATER, w.cap)
    w.fill(26, 30, 10, 14, 24, 28, WOOD, frac=0.9)           # the wood block twin
    for t in range(ticks):
        w.step()
        if t % every == 0 or t == ticks - 1:
            _save(w, frames_dir, t)
        if t % 20 == 0:
            g = w.mat == GLASS
            full = 2500.0 * w.vox_l
            print(f"t{t}: glass cells {int(g.sum())} (intact "
                  f"{int((w.smass[g] > 0.9 * full).sum())}), wood cells "
                  f"{int((w.mat == WOOD).sum())}, loose water "
                  f"{w.total_fluid(WATER):.0f} ml", flush=True)


def forge(frames_dir, ticks=700, every=8):
    """TRANSFORMATION: an established bed of glowing coals under three bars —
    tin, lead, iron. Three MELT rows do everything: tin (232°) runs first and
    drips off its rail, lead (327°) follows sluggishly, iron (1538°) only
    glows. What runs, pools on the cold stone floor and FREEZES into splats —
    the same table read backwards."""
    w = World(48, 30, 26, voxel_cm=5)
    w.fill(0, 48, 0, 30, 0, 1, STONE)
    w.fill(6, 42, 6, 24, 1, 2, STONE)                        # hearth slab
    w.fill(8, 40, 8, 22, 2, 4, CHAR, frac=0.8)               # the coal bed, and a
    w.fill(9, 41, 11, 19, 4, 7, CHAR, frac=0.8)              # heap over the work —
    w.fill(11, 15, 13, 17, 4, 6, TIN)                        # the bars sit BURIED
    w.fill(23, 27, 13, 17, 4, 6, LEAD)                       # in the coals, the way
    w.fill(35, 39, 13, 17, 4, 6, IRON)                       # a real smelt holds
    bed = w.mat == CHAR                                      # its heat
    w.E[bed] = 680.0 * w.heat_capacity()[bed]
    for t in range(ticks):
        w.step()
        if w.bodies or t % every == 0 or t == ticks - 1:
            _save(w, frames_dir, t)
        if t % 100 == 0:
            T = w.T()
            line = []
            for name, m, fl_id in (("tin", TIN, MTIN), ("lead", LEAD, MLEAD),
                                   ("iron", IRON, MIRON)):
                solid = float(w.smass[w.mat == m].sum())
                molten = float(w.fvol[w.fl == fl_id].sum())
                bar = w.mat == m
                tt = float(T[bar].max()) if bar.any() else 0.0
                line.append(f"{name}: {solid / 1000:.1f} kg solid, "
                            f"{molten:.0f} ml molten, {tt:.0f} °C")
            print(f"t{t}: " + " | ".join(line), flush=True)


def alchemist(frames_dir, ticks=900, every=8):
    """Glass vats of vitriol on wooden shelving over a workbench. The shelf edge
    under one vat cracks; the vat tips; everything after that is the laws'
    business: spilled acid eats the shelf, weakened wood drops the next vat,
    pools eat the bench and the uprights. A consequence CASCADE, zero scripting
    past the first crack."""
    w = World(50, 44, 40, voxel_cm=5)
    w.fill(0, 50, 0, 44, 0, 1, STONE)
    for x0 in (8, 40):                                       # shelving uprights
        w.fill(x0, x0 + 2, 34, 36, 1, 25, WOOD)
    w.fill(6, 44, 32, 38, 11, 12, WOOD, frac=0.8)            # lower shelf board
    w.fill(6, 44, 32, 38, 22, 23, WOOD, frac=0.8)            # upper shelf board
    _vat(w, 12, 33, 23)                                      # vat A (upper, will tip)
    _vat(w, 24, 33, 23)                                      # vat B (upper)
    _vat(w, 32, 33, 12)                                      # vat C (lower)
    w.fill(10, 34, 16, 37, 8, 9, WOOD, frac=0.8)             # the workbench top runs
    for (lx, ly) in ((11, 17), (11, 27), (31, 17), (31, 27)):    # back UNDER the shelves
        w.fill(lx, lx + 2, ly, ly + 2, 1, 8, WOOD)           # bench legs
    a0 = w.total_fluid(ACID)
    for t in range(ticks):
        if t == 40:                                          # the shelf SPLITS along
            w.mat[11:17, 32:36, 22] = AIR                    # its front edge — vat A
            w.smass[11:17, 32:36, 22] = 0.0                  # keeps only its back row
            print("t40: the shelf cracks away under vat A's front", flush=True)
        w.step()
        if w.bodies or t % every == 0 or t == ticks - 1:
            _save(w, frames_dir, t)
        if t % 100 == 0:
            am = w.fl == ACID
            active = float((w.fpot[am] * w.fvol[am]).sum())
            print(f"t{t}: acid {w.total_fluid(ACID):.0f}/{a0:.0f} ml "
                  f"(still potent: {active:.0f} ml), "
                  f"wood {w.total_wood():.0f} g, glass standing "
                  f"{int((w.mat == GLASS).sum())}, fumes {w.smoke.sum():.0f} g", flush=True)


# ── the blocky humanoid ──────────────────────────────────────────────────────
def _person(w, px, py, z0=1, handed="right"):
    """A person, finally with a body: two legs, torso, two arms, a head.
    ~1.5 m of FLESH at 5 cm voxels. A physical object first — it stands by the
    support law, tips by the torque law, chars by the combustion law.

    It has DEPTH because it has to weigh something. Built one voxel thin it came
    to 10 kg, and a 10 kg person can be picked up like a cat — which quietly
    made nonsense of every question about lifting, dragging and carrying. Given
    a real chest, and arms two voxels through instead of one, it measures
    43.9 kg against the ~47 kg a real person of this height would be. The rest
    of the gap is that a blocky figure at 5 cm still fills less of its own
    outline than a person does.

    (This said 28.5 kg before the chest had depth and 38.1 kg before the arms
    did. A number in a docstring goes stale the same way a number in a roadmap
    does — 396 cells, 43.9 kg, 7 voxels across, 30 tall, measured 2026-09-13.)
    The chest's thickness went into DEPTH (y) and not width (x) on purpose —
    the walking footprint, and so every doorway in every scene, was unchanged.

    THE ARMS ARE TWO VOXELS ACROSS, and the number that settled it: a bone
    cannot be turned in the plane it is thin in, and a shoulder swings in the
    plane the body FACES. One voxel across, a man facing along y could put his
    arm through 30 of 30 angles and a man facing along x through ONE — not a
    limit but a paralysis, in half the directions he can stand. Two voxels
    takes the bad direction to 15 of 30, and a 5 cm arm drew as a blade rather
    than a limb besides.

    It cost him two voxels of width: 35 cm across the shoulders where he was
    25, which is still narrow for a man, and fourteen scenes measured against
    the narrower one had to be re-measured (roadmap 66, 70, 74)."""
    z = z0 - 1                                   # stands ON whatever is at z0
    segs = {}

    def part(name, x0, x1, y0, y1, z1, z2, frac):
        """Build one piece and REMEMBER WHICH VOXELS IT IS.

        A body made of named pieces is not decoration: it is the difference
        between a lump that can only be moved whole and one that can move an arm.
        The sim cannot work this out for itself — every voxel is just FLESH, and
        nothing in the lattice says where a shoulder is — but the scene that
        built the body knows exactly, and how a thing was built is the object's
        own business to declare (Ruling 2, question 1)."""
        was = w.mat == FLESH
        w.fill(x0, x1, y0, y1, z1, z2, FLESH, frac=frac)
        segs[name] = np.argwhere((w.mat == FLESH) & ~was)

    # A LEG IS TWO BONES TOO. Same voxels, same outline, same mass — split at
    # the knee, so the body has somewhere to bend on the way down. Without a
    # knee there is no crouch, no kneel and no step up: a leg that is one bone
    # can only be somewhere, never on its way anywhere.
    #
    # Shin first in the chain and THIGH second, which is the other way round
    # from an arm and is the whole difference between the two. An arm hangs
    # from the body and its far end is the hand; a leg STANDS ON THE GROUND and
    # its far end is the hip. So a leg's chain runs upward from the ankle, and
    # bending the knee carries the hip — which is what a crouch is.
    for side, lx0 in (("left", px), ("right", px + 2)):
        part(f"{side} shin", lx0, lx0 + 1, py - 1, py + 2, z + 1, z + 8, 0.9)
        part(f"{side} thigh", lx0, lx0 + 1, py - 1, py + 2, z + 8, z + 15, 0.9)
    part("torso", px, px + 3, py - 2, py + 3, z + 15, z + 27, 0.9)
    # AN ARM IS TWO BONES. Same voxels, same outline, same 28.5 kg — split at
    # the elbow so the arm has somewhere to BEND. One bone can only reach a
    # place along one path, and a man standing beside the person he is holding
    # found that path already occupied by the person.
    # TWO VOXELS THICK, NOT ONE, AND THAT IS WHAT LETS AN ARM MOVE. A bone one
    # voxel across cannot be turned in the plane it is thin in: it rounds into
    # a staircase whose steps touch at corners only, which is not a limb.
    # Measured, 31 angles apiece: one voxel thick held 4 of them, two voxels
    # held 27. It showed up as a left-handed man who could not reach out while
    # facing along x — his arm had no drawable path at all — and it was the
    # flesh, not the reach.
    #
    # It is also the truer body. A 5 cm arm is a matchstick; a real upper arm
    # is about 10 cm through, which is exactly these two voxels, and the man
    # goes from 38 kg to nearer the 47 kg his height asks for.
    for side, ax0 in (("left", px - 2), ("right", px + 3)):
        part(f"{side} upper arm", ax0, ax0 + 2, py - 1, py + 2, z + 23, z + 27, 0.85)
        part(f"{side} forearm", ax0, ax0 + 2, py - 1, py + 2, z + 18, z + 23, 0.85)
    part("head", px, px + 2, py - 1, py + 2, z + 27, z + 31, 0.9)
    person = w.add_person(px + 1, py)                            # and now: alive
    # segments and joints are kept as OFFSETS from the body's own corner, so
    # they survive the body walking, being shoved, or being carried out
    origin = np.concatenate(list(segs.values())).min(axis=0)
    # A BODY BUILT INTO A SPACE TOO SHORT FOR IT HAS NO HEAD. `fill` clamps at
    # the lattice edge, so a 1.5 m person in a 1.1 m room is genuinely missing
    # his top half — and a part with no voxels is not a part. Said here rather
    # than defended everywhere downstream: a scene that wants a whole man gives
    # him room. (Before arms had elbows this hid, because the one arm segment
    # still had its lower half inside the room.)
    segs = {k: v for k, v in segs.items() if len(v)}
    person["segs"] = {k: v - origin for k, v in segs.items()}
    # A LIMB IS A CHAIN OF BONES, nearest the body first. Everything outside
    # the scene asks for "right arm" and never has to know how many bones that
    # is — which is the point of declaring it here (Ruling 2, question 1).
    person["chain"] = {f"{s} arm": [b for b in (f"{s} upper arm", f"{s} forearm")
                                    if b in segs]
                       for s in ("left", "right")}
    person["chain"].update(
        {f"{s} leg": [b for b in (f"{s} shin", f"{s} thigh") if b in segs]
         for s in ("left", "right")})
    person["chain"] = {k: v for k, v in person["chain"].items() if v}
    person["joints"] = {}
    # A SOCKET IS INSIDE THE BODY. The shoulder joint was put at the top of the
    # arm, which is the surface where the arm meets the man — and an arm turned
    # about its own surface swings that surface AWAY. Measured over 1089
    # shoulder poses, 25 of them left the man in two pieces, every one of them
    # an arm raised past 1.2 rad toward the horizontal, which is the ordinary
    # act of pointing at something. Moved one cell in, toward the middle of the
    # body: nought of 1089.
    #
    # It is where the joint actually is, too. A shoulder is a ball in a socket
    # under the deltoid, not a hinge on the skin.
    mid_x = float(px) + 1.0
    for bone in ("left upper arm", "left forearm",
                 "right upper arm", "right forearm"):
        cells = segs.get(bone)
        if cells is None:
            continue
        cx = int(round(cells[:, 0].mean()))
        person["joints"][bone] = np.array(                 # the joint: the TOP
            [cx + (1 if cx < mid_x else -1),               # of the bone, where
             int(round(cells[:, 1].mean())),               # it hangs from what
             int(cells[:, 2].max())]) - origin             # is above it
    # AND A LEG'S JOINTS ARE AT ITS BOTTOM, for the same reason its chain runs
    # the other way: an ankle is under a shin and a knee is under a thigh,
    # because a leg is held up by the ground and not by the body.
    for bone in ("left shin", "left thigh", "right shin", "right thigh"):
        cells = segs.get(bone)
        if cells is None:
            continue
        person["joints"][bone] = np.array(
            [int(round(cells[:, 0].mean())),
             int(round(cells[:, 1].mean())),
             int(cells[:, 2].min())]) - origin
    # AND THE BODY HANGS FROM THE HIPS. What the arms and the head hang FROM is
    # not something any chain says, because they are not in one another's
    # chains — so it is said here. The legs are deliberately NOT children of the
    # torso: leaning bends a man at the waist and leaves his feet where they are
    # standing, which is the whole point of a lean.
    if "torso" in segs:
        person["parent"] = {b: "torso" for b in
                            ("head", "left upper arm", "right upper arm")
                            if b in segs}
        person["chain"]["lean"] = ["torso"]
        # the hips: the BOTTOM of the torso, because unlike every other bone
        # the torso hangs from something below it rather than above
        t = segs["torso"]
        person["joints"]["torso"] = np.array(
            [int(round(t[:, 0].mean())), int(round(t[:, 1].mean())),
             int(t[:, 2].min())]) - origin
        # NO `torque_Nm` DECLARED. What this waist can hold is read off the
        # torso it has — section times stress times insertion — which comes to
        # about 400 N.m where the typed `back_Nm` said 200. A real trunk
        # extensor is 200 to 400, so the derivation lands at the top of the
        # range where the guess sat at the bottom, and a scene that wants a
        # weaker back can still say so by putting the key back.
        person["bend"] = ["torso"]    # a spine BENDS, it does not swing
        person["wmax"] = {"lean": BODY["waist_wmax"]}
    # MOST PEOPLE ARE RIGHT-HANDED. A fact about this body, like its mass —
    # a scene that wants a left-handed one says so.
    person["handed"] = handed
    person.update({"px": px, "py": py, "head_z": z + 28})
    return person


def _exposure(w, p):
    """What the world is DOING to the body: peak skin heat, and the air at head
    height vs knee height (O2 %% and smoke) — the damage model's raw senses."""
    flesh = w.mat == FLESH
    skin = float(w.T()[flesh].max()) if flesh.any() else 0.0
    def air_at(z):
        sl = (slice(max(p["px"] - 4, 0), p["px"] + 6),
              slice(max(p["py"] - 4, 0), p["py"] + 6), slice(z, z + 3))
        m = w.mat[sl] == AIR
        if not m.any() or w.o2 is None:
            return 100.0, 0.0
        o2 = 100 * float(w.o2[sl][m].mean()) / (O2_PER_L * w.vox_l)
        return o2, float(w.smoke[sl][m].mean())
    ho2, hsm = air_at(p["head_z"])
    ko2, ksm = air_at(4)
    return (f"skin {skin:.0f} °C | head air: O2 {ho2:.0f}%, smoke {hsm:.2f} g | "
            f"knee air: O2 {ko2:.0f}%, smoke {ksm:.2f} g | {w.person_status(p)}")


def fire_alarm(frames_dir, ticks=700, every=4):
    """SOCIAL SITUATION, layer one — no minds yet, just reflexes (the WILL
    data): the same burning house with TWO people and a door. Aldan stands
    near the bed and SEES the flames catch; he bolts. Berel stands behind an
    alcove wall and genuinely CANNOT see the fire — sight is a ray now, and
    masonry blocks it. What Berel eventually sees is Aldan sprinting for the
    door. Alarm spreads by sight, not by a message channel; both walk out of
    the same door the smoke is starting to curl through."""
    w = World(100, 80, 50, voxel_cm=5)
    w.fill(0, 100, 0, 80, 0, 1, STONE)
    w.fill(10, 90, 10, 70, 0, 45, STONE)
    w.mat[11:89, 11:69, 1:44] = AIR
    w.smass[11:89, 11:69, 1:44] = 0.0
    w.mat[10:11, 35:45, 1:38] = AIR
    w.smass[10:11, 35:45, 1:38] = 0.0
    w.fill(66, 86, 14, 28, 5, 7, WOOD, frac=0.7)             # the bed
    for (lx, ly) in ((67, 15), (67, 25), (83, 15), (83, 25)):
        w.fill(lx, lx + 2, ly, ly + 2, 1, 5, WOOD)
    w.fill(36, 60, 50, 52, 1, 44, STONE)                     # an alcove wall hiding
    w.exits = [(13, 40)]                                     # the bed; its OPENING
    a = _person(w, 60, 30)                                   # faces the door, so the
    a["name"] = "Aldan"                                      # one thing Berel can
    a["lines"] = {"fire": "Fire! The bed's caught — get out, get out!"}
    b = _person(w, 32, 58)                                   # see is a man running
    b["name"] = "Berel"                                      # for it — but a SHOUT
    b["lines"] = {"coming": "I hear you! I'm coming!"}   # gets there first
    for (ox, oy) in ((64, 16), (65, 17), (64, 18)):
        w.pour(ox, oy, 1, OIL, 100.0)
    for t in range(ticks):
        if t < 30:
            w.E[64, 16, 1] += 2500.0
        w.step()
        for p in (a, b):
            for e in p["events"]:
                print(e, flush=True)
            p["events"].clear()
        if w.bodies or t % every == 0 or t == ticks - 1:
            _save(w, frames_dir, t)
        if t % 100 == 0:
            print(f"t{t}: burning {int(w.burning().sum())} | "
                  f"Aldan at {a['anchor']}{' SAFE' if a['safe'] else ''} | "
                  f"Berel at {b['anchor']}{' SAFE' if b['safe'] else ''}", flush=True)
        if a["safe"] and b["safe"] and "out_at" not in a:
            a["out_at"] = t
            print(f"t{t}: both outside — the house can have the rest", flush=True)
        if "out_at" in a and t > a["out_at"] + 80:           # a beat of aftermath
            break
    import json
    with open(os.path.join(frames_dir, "speech.json"), "w") as f:
        json.dump({"speech": [[int(tk), nm, tx] for tk, nm, tx in w.speech],
                   "every": every}, f)
    # the harvest: every decision, the whole menu it was picked from, and how
    # that body ended. This is the file a model is later trained on.
    rows = [{k: v for k, v in r.items() if k != "situation"}
            | {"situation": {k: v for k, v in r["situation"].items()
                             if k != "reflexes"}}
            for r in w.trace_outcomes()]
    with open(os.path.join(frames_dir, "traces.json"), "w") as f:
        json.dump(rows, f, indent=1)
    for r in rows:
        # `menus` and `pick` are PER LIMB — a body decides with its legs and
        # its hands at once, and has since the menu seam grew limbs. This read
        # `r['menu']`, which stopped existing at that point, and the scenario
        # has been dying on its last line ever since with all its frames
        # already written. Nothing caught it: scenarios are not in the suite.
        menus = " | ".join(f"{limb}:{opts}" for limb, opts
                           in sorted((r.get("menus") or {}).items()))
        print(f"t{r['tick']}: {r['who']} [{r['percept']}] "
              f"menu={menus} -> {r.get('pick')} ({r.get('by')}) "
              f"=> {r['outcome']}", flush=True)


# ── scenario: the glasshouse — perception reads continuous fields ────────────
def glasshouse(frames_dir, ticks=760, every=4):
    """Everything the Ruling-2 pass bought, in one room and one run.

    A workshop split by a stone wall that carries two openings: a GLASS WINDOW
    and a shut WOODEN DOOR hung with a gap (frac 0.97). A fire starts in the
    west room, where nobody is.

      SIGHT IS OPTICAL DEPTH — Wren sees the fire THROUGH THE WINDOW. Under the
        old material whitelist glass was as blind as masonry and she would have
        stood there until the smoke took her.
      SOUND IS THE MASS LAW — her shout crosses the pine door to Bram at ~37 dB
        of loss. Under the old flat cost every solid voxel charged the same, so
        pine and masonry were the same wall.
      GAPS ARE GEOMETRY — smoke seeps under the shut door because the scene says
        it fills 97% of its cells, not because wood was declared leaky.
      FILL DECIDES PASSAGE — a hedge stands across the only route out. She
        shoves through it. One leaf voxel used to be as impassable as stone.
      AND SMOKE BLINDS — as the room fills, the window she saw the fire through
        goes out. That was simply missing.
    """
    w = World(130, 70, 44, voxel_cm=5)                    # 6.5 m x 3.5 m x 2.2 m
    w.fill(0, 130, 0, 70, 0, 44, STONE)
    w.mat[1:129, 1:69, 1:43] = AIR
    w.smass[1:129, 1:69, 1:43] = 0.0
    w.fill(60, 62, 1, 69, 1, 43, STONE)                   # the dividing wall
    w.fill(60, 62, 20, 34, 16, 30, GLASS)                 # ... with a window
    w.fill(60, 62, 45, 58, 1, 30, WOOD, frac=0.97)        # ... and a shut door
    w.fill(100, 104, 1, 69, 1, 32, LEAF, frac=0.30)       # a hedge, wall to wall
    w.mat[128:129, 32:40, 1:30] = AIR                     # the way out, east
    w.smass[128:129, 32:40, 1:30] = 0.0
    w.exits = [(128, 36)]
    w.fill(20, 40, 22, 32, 5, 7, WOOD, frac=0.7)          # the workbench, west
    for (lx, ly) in ((21, 23), (21, 30), (37, 23), (37, 30)):
        w.fill(lx, lx + 2, ly, ly + 2, 1, 5, WOOD)
    for (ox, oy) in ((24, 25), (25, 26), (24, 27)):
        w.pour(ox, oy, 5, OIL, 100.0)                     # oil on the bench
    wren = _person(w, 70, 27)                             # facing the window
    wren["name"] = "Wren"
    wren["lines"] = {"fire": "The bench is alight! Bram — the hedge, go!"}
    bram = _person(w, 84, 52)                             # the far side, by the door
    bram["name"] = "Bram"
    bram["lines"] = {"coming": "Right behind you!"}
    eye_w = (71.0, 27.0, 30.0)
    fire_at = (25.0, 26.0, 6.0)
    seen_through_glass = None
    blinded_at = None
    for t in range(ticks):
        if t < 30:
            w.E[24, 25, 5] += 2500.0
        w.step()
        if seen_through_glass is None and w._sees(eye_w, fire_at) and w.burning().any():
            seen_through_glass = t
        if seen_through_glass is not None and blinded_at is None \
                and not w._sees(eye_w, fire_at):
            blinded_at = t
            print(f"t{t}: the smoke has closed the window — Wren can no longer "
                  f"see the fire she is running from", flush=True)
        for p in (wren, bram):
            for e in p["events"]:
                print(e, flush=True)
            p["events"].clear()
        if w.bodies or t % every == 0 or t == ticks - 1:
            _save(w, frames_dir, t)
        if t % 100 == 0:
            east = float(w.smoke[62:129, 1:69, 1:43].sum())
            print(f"t{t}: burning {int(w.burning().sum())} | smoke past the shut "
                  f"door {east:7.2f} g | Wren {wren['anchor']}"
                  f"{' SAFE' if wren['safe'] else ''} | Bram {bram['anchor']}"
                  f"{' SAFE' if bram['safe'] else ''}", flush=True)
        if wren["safe"] and bram["safe"] and "out_at" not in wren:
            wren["out_at"] = t
            print(f"t{t}: both through the hedge and out", flush=True)
        if "out_at" in wren and t > wren["out_at"] + 60:
            break
    print(f"\n-- what each fix did --", flush=True)
    print(f"sight through GLASS: fire first seen at t{seen_through_glass}"
          f"  (was impossible: glass counted as masonry)", flush=True)
    print(f"smoke BLINDED that same line of sight at t{blinded_at}", flush=True)
    door = w._hears((71.0, 50.0, 20.0), (50.0, 50.0, 20.0))   # through the door
    wall = w._hears((71.0, 10.0, 20.0), (50.0, 10.0, 20.0))   # through masonry
    print(f"shout across the pine door: {'heard' if door else 'lost'} | "
          f"across the stone wall: {'heard' if wall else 'lost'}"
          f"  (was identical: 4 m per voxel, whatever it was made of)", flush=True)
    print(f"smoke through a door nobody opened: "
          f"{float(w.smoke[62:129, 1:69, 1:43].sum()):.2f} g", flush=True)
    import json
    with open(os.path.join(frames_dir, "speech.json"), "w") as f:
        json.dump({"speech": [[int(tk), nm, tx] for tk, nm, tx in w.speech],
                   "every": every}, f)
    rows = [{k: v for k, v in r.items() if k != "situation"}
            | {"situation": {k: v for k, v in r["situation"].items()
                             if k != "reflexes"}}
            for r in w.trace_outcomes()]
    with open(os.path.join(frames_dir, "traces.json"), "w") as f:
        json.dump(rows, f, indent=1)


# ── scenario: house fire, someone inside ─────────────────────────────────────
def house_fire(frames_dir, ticks=900, every=10):
    """A furnished room: bed, table, wardrobe — and a PERSON standing in it.
    A spilled lamp lights the bed. The scene is the world acting on a body:
    smoke banks down from the ceiling, the air at head height goes foul while
    the knee-level air stays breathable (crawl low), the skin readout climbs."""
    w = World(100, 80, 50, voxel_cm=5)
    w.fill(0, 100, 0, 80, 0, 1, STONE)
    w.fill(10, 90, 10, 70, 0, 45, STONE)                     # the house block...
    w.mat[11:89, 11:69, 1:44] = AIR                          # ...hollowed to a room
    w.smass[11:89, 11:69, 1:44] = 0.0
    w.mat[10:11, 35:45, 1:38] = AIR                          # a doorway to outside
    w.smass[10:11, 35:45, 1:38] = 0.0
    w.fill(66, 86, 14, 28, 5, 7, WOOD, frac=0.7)             # bed platform
    for (lx, ly) in ((67, 15), (67, 25), (83, 15), (83, 25)):
        w.fill(lx, lx + 2, ly, ly + 2, 1, 5, WOOD)
    w.fill(30, 44, 50, 62, 10, 11, WOOD, frac=0.8)           # table
    for (lx, ly) in ((31, 51), (31, 59), (41, 51), (41, 59)):
        w.fill(lx, lx + 2, ly, ly + 2, 1, 10, WOOD)
    w.fill(84, 88, 55, 67, 1, 30, WOOD, frac=0.5)            # wardrobe
    p = _person(w, 46, 34)                                   # someone mid-room
    for (ox, oy) in ((64, 16), (65, 17), (64, 18)):
        w.pour(ox, oy, 1, OIL, 100.0)                        # the dropped lamp
    for t in range(ticks):
        if t < 30:
            w.E[64, 16, 1] += 2500.0                         # its burning wick
        w.step()
        for e in p["events"]:                                # physiology narrates itself
            print(e, flush=True)
        p["events"].clear()
        if w.bodies or t % every == 0 or t == ticks - 1:
            _save(w, frames_dir, t)
        if t % 50 == 0:
            print(f"t{t}: burning {int(w.burning().sum())}, {_exposure(w, p)}", flush=True)



# ── scenario: a body leaves the ground ───────────────────────────────────────
class _Wants:
    """A policy that picks one named act per limb, the null act otherwise.

    A render is a demonstration, not a drama: pinning the taste keeps the clip
    about whether the PHYSICS works. The pick still goes through the same menu,
    is still checked for legality, and is still written to the trace — a body
    with nothing underfoot is never offered a jump, however much this wants one.

    A want is matched against a row's TAG first and then against its NAME, so a
    scene can ask for a kind of act ("swing") or for one particular row ("go(the
    door east)") without the sim growing a second way to be told.

    And it can be given PER PERSON, because the moment a scene has two people in
    it they stop wanting the same thing: one pulls, one digs their heels in."""

    name = "wants"

    def __init__(self, each=None, **want):
        self.want, self.each = want, each or {}

    def pick(self, situation, menu):
        want = self.each.get(situation["who"], self.want)
        tag = want.get(situation["limb"])
        if not tag:
            return 0
        for i, opt in enumerate(menu):
            if opt["tag"] == tag:
                return i
        for i, opt in enumerate(menu):
            if tag in opt["key"]:
                return i
        return 0


def jumper(frames_dir, ticks=260, every=2):
    """A person jumps, and a crate falls beside them.

    Two things to watch, both impossible a day ago. The body LEAVES THE GROUND —
    legs put 1400 N into the floor over a crouch, what is left after holding its
    own weight up becomes speed, and the height follows. And the crate does not
    descend at a stately one voxel a tick: it starts from rest and picks up
    speed, because falling finally has one."""
    w = World(34, 20, 64, voxel_cm=5)
    w.fill(0, 34, 0, 20, 0, 1, STONE)                    # the floor
    w.exits = [(32, 10)]
    p = _person(w, 8, 10)
    w.policy = _Wants(legs="jump")
    w.fill(22, 26, 8, 12, 44, 48, WOOD, frac=0.6)        # a crate, high up
    for t in range(ticks):
        w.step()
        for e in p["events"]:
            print(e, flush=True)
        p["events"].clear()
        if w.bodies or t % every == 0 or t == ticks - 1:
            _save(w, frames_dir, t)
        if t % 40 == 0:
            flying = "airborne" if w.bodies else "on the floor"
            print(f"t{t}: {flying}", flush=True)


# ── scenario: a gap the legs cannot cross ────────────────────────────────────
def leap(frames_dir, ticks=90, every=1):
    """A person jumps a chasm they cannot walk across.

    The jump used to go straight up and land where it started, which is barely
    a jump — the whole point of having a velocity is that it has a DIRECTION.
    Aimed, the body leaves at the angle that carries furthest and at exactly
    the speed the distance needs, so it lands where it meant to rather than
    hurling itself as hard as it can. And the gap is a real obstacle: a column
    of open air is not somewhere to stand, so the route planner will not walk
    anyone over it any more.

    Nothing decides to leap because the gap is there. The option is OFFERED
    because a landing spot exists that no route reaches; a leap to somewhere it
    could simply walk to is not on the menu at all."""
    w = World(70, 20, 80, voxel_cm=5)
    w.fill(0, 70, 0, 20, 0, 1, STONE)               # the bottom, far below
    w.fill(2, 24, 0, 20, 1, 12, STONE)              # the near ledge
    w.fill(34, 68, 0, 20, 1, 12, STONE)             # the far one, 50 cm away
    w.exits = [(66, 10)]
    p = _person(w, 9, 10, z0=12)
    w.policy = _Wants(legs="leap")
    for t in range(ticks):
        w.step()
        for e in p["events"]:
            print(e, flush=True)
        p["events"].clear()
        _save(w, frames_dir, t)
        if not w.bodies and t % 10 == 0:
            on = np.argwhere(w.mat == FLESH)
            if len(on):
                print(f"t{t}: standing at x={on[:, 0].mean():.0f}", flush=True)


# ── scenario: a limb that is really there ────────────────────────────────────
def swing(frames_dir, ticks=120, every=1):
    """Two people swing a bare fist. One shatters a pane; one bruises a post.

    The body was one rigid lump until now: it could travel, but nothing on it
    could move by itself. An arm is the same rotation a toppling tree does —
    cells turning about a pivot — with the pivot moved INSIDE the body and the
    acceleration coming from a muscle instead of from gravity.

    It is not animation. There is no picture here to change: every law reads the
    lattice, so while the arm comes round it really is somewhere else, and what
    stops it is paid 1/2 I omega-squared against its OWN toughness. Glass gives
    at 0.2 kJ/m2, a wooden post does not at 8.0 — same fist, same speed, two
    answers, and nobody wrote either of them down.

    The speed is not typed in either. A muscle's torque fades as it speeds up
    (Hill), so the swing settles at a shoulder's pace instead of accelerating
    for as long as the sweep lasts. Before that, a bare fist splintered the
    post, which is what having no force-velocity relation buys you."""
    w = World(34, 22, 44, voxel_cm=5)
    w.fill(0, 34, 0, 22, 0, 1, STONE)
    w.exits = [(32, 11)]
    a = _person(w, 9, 6)
    a["name"] = "at the pane"
    w.fill(17, 18, 5, 8, 1, 30, GLASS)                   # a pane
    b = _person(w, 9, 16)
    b["name"] = "at the post"
    w.fill(17, 18, 15, 18, 1, 30, WOOD)                  # a post
    w.policy = _Wants(hands="swing", legs="stay")
    fullg = float(w.smass[w.mat == GLASS].max())
    fullw = float(w.smass[w.mat == WOOD].max())
    for t in range(ticks):
        w.step()
        for q in (a, b):
            for e in q["events"]:
                print(e, flush=True)
            q["events"].clear()
        _save(w, frames_dir, t)
        if t % 20 == 0:
            g, o = w.mat == GLASS, w.mat == WOOD
            print(f"t{t}: pane {int((g & (w.smass < 0.9 * fullg)).sum())} broken "
                  f"of {int(g.sum())} | post "
                  f"{int((o & (w.smass < 0.9 * fullw)).sum())} broken "
                  f"of {int(o.sum())}", flush=True)


# ── scenario: pulling a man off a ledge ──────────────────────────────────────
def ledge(frames_dir, ticks=150, every=1):
    """Two men stand on the same ledge. Two men on the ground below take hold of
    an ankle each and walk away. One goes over. One does not budge.

    The only difference is how hard the one pulling can pull: 700 N at the front,
    400 N at the back, against the same braced man both times. Nothing here is a
    rule about cliffs, or about fighting. A body that is awake and has something
    under its feet PUSHES BACK with its own strength, and that single line is the
    whole of it — it is why an unconscious man can be dragged out of a fire and a
    standing one cannot be dragged anywhere, why a stronger man manages it, and
    why the resistance is gone the moment his heels are over air. Before this a
    grip was legal only on someone already unconscious, which made this scene not
    hard but UNASKABLE: the row was never on the menu.

    What carries him over the lip is not the pull. Support is relaxed from the
    ground up through material, so a man walked off his own footing hangs by his
    span and then goes — he teeters, which nobody wrote either.

    A KNOWN SOFTNESS: reach is measured across the floor and ignores height, so
    the man below can take an ankle 1.5 m above him. At this height that is about
    right; at five metres it would be nonsense."""
    w = World(40, 22, 64, voxel_cm=5)
    w.fill(0, 40, 0, 22, 0, 1, STONE)                    # the ground
    w.fill(20, 40, 0, 22, 1, 31, STONE)                  # the ledge, 1.5 m of it
    w.exits = []
    pairs = []
    for py, puller, mark, strength in ((5, "the strong one", "the first", 700.0),
                                       (16, "the ordinary one", "the second", 400.0)):
        a = _person(w, 16, py, z0=1)                     # below, on the ground
        a["name"], a["strength_N"], a["facing"] = puller, strength, (-1.0, 0.0)
        b = _person(w, 23, py, z0=31)                    # above, at the lip
        b["name"] = mark
        pairs.append((a, b))
        print(f"{puller}: {strength:.0f} N against {mark}, braced", flush=True)
    w.policy = _Wants(each={
        a["name"]: {"hands": f"take hold of {b['name']}", "legs": "straight on"}
        for a, b in pairs})
    for t in range(ticks):
        w.step()
        for a, b in pairs:
            for q in (a, b):
                for e in q["events"]:
                    print(e, flush=True)
                q["events"].clear()
        if t % every == 0 or t == ticks - 1:
            _save(w, frames_dir, t)
        if t % 20 == 0:
            say = []
            for _a, b in pairs:
                c, sl = w._person_cells(b)
                if c is None or not c.any():
                    continue
                cells = np.argwhere(c)
                say.append(f"{b['name']} x{cells[:, 0].mean() + sl[0].start:.0f} "
                           f"z{cells[:, 2].min()}")
            print(f"t{t}: " + " | ".join(say), flush=True)


# ── scenario: the same mass, two shapes ──────────────────────────────────────
def parachute(frames_dir, ticks=420, every=3):
    """A sheet and a wad of EXACTLY the same stuff, dropped together.

    Nothing here is about parachutes. Air resists what it has to go around, so
    the one spread flat presents many faces to its own fall and the one balled
    up presents few — and the wad reaches the floor while the sheet is still
    coming down. Every canopy that ever worked is this. The sim was never told
    an area: it counts the faces, because it has always known the shape."""
    w = World(44, 26, 100, voxel_cm=5)
    w.fill(0, 44, 0, 26, 0, 1, STONE)
    w.fill(3, 19, 5, 21, 90, 91, WOOD, frac=0.01)        # spread: 16 x 16 x 1
    w.fill(28, 32, 11, 15, 90, 106, WOOD, frac=0.01)     # balled: 4 x 4 x 16
    wood = w.mat == WOOD
    sheet = float(w.smass[:22][wood[:22]].sum())
    wad = float(w.smass[22:][wood[22:]].sum())
    print(f"sheet {sheet:.1f} g vs wad {wad:.1f} g — the same matter, two shapes",
          flush=True)
    down = {}
    for t in range(ticks):
        w.step()
        for name, xs in (("sheet", slice(0, 22)), ("wad", slice(22, 44))):
            if name in down:
                continue
            here = np.argwhere(w.mat[xs] == WOOD)
            if len(here) and int(here[:, 2].min()) <= 1:
                down[name] = t
                print(f"t{t}: the {name} is down", flush=True)
        if t % every == 0 or t == ticks - 1:
            _save(w, frames_dir, t)
    print(f"landed: {down}", flush=True)


# ── scenario: somebody goes over the edge, and somebody catches them ─────────
def rescue(frames_dir, ticks=300, every=4):
    """This session's first question and its last, in one scene.

    A man below takes an ankle and walks away with it — that is "can a man pull
    another off a cliff", and the answer is yes if he can out-pull a braced
    man's own strength. A second man on the ledge has one hand on a post and
    the other free, and catches him as he goes.

    Nobody wrote a rescue. What meets here was built separately and for other
    reasons: a percept that tells a falling PERSON from a thrown stone, a flier
    with a real trajectory, a hand whose strength decides what it can stop, a
    grip that is an edge in the support graph, and two hands that can do two
    things at once."""
    # HEADROOM. The camera frames the whole world, so a body standing on a
    # ledge at the top of it is a body with its head off the top of the
    # picture — which is what the first render of this scene was.
    w = World(52, 24, 86, voxel_cm=5)
    w.fill(0, 52, 0, 24, 0, 1, STONE)                    # the ground, far below
    w.fill(22, 52, 0, 24, 1, 31, STONE)                  # the ledge, 1.5 m up
    w.fill(30, 31, 9, 15, 31, 45, IRON)                  # a post to hold
    w.exits = []
    puller = _person(w, 18, 12, z0=1)                    # below, on the ground
    puller["name"], puller["strength_N"] = "the puller", 900.0
    puller["facing"] = (-1.0, 0.0)
    falls = _person(w, 25, 12, z0=31)                    # above, at the lip
    falls["name"], falls["facing"] = "the one pulled", (-1.0, 0.0)
    # x34, not 33. He is 7 voxels across since the arms went to two, so 33
    # built him THROUGH the post he is supposed to be holding.
    saves = _person(w, 34, 12, z0=31)                    # further back, by the post
    saves["name"], saves["facing"] = "the rescuer", (-1.0, 0.0)
    saves["strength_N"] = 4000.0
    w.policy = _Wants(each={
        "the puller": {"hands": "take hold of the one pulled",
                       "legs": "go(straight on", "waist": "stand"},
        "the one pulled": {"hands": "hands free", "legs": "stay",
                           "waist": "stand"},
        "the rescuer": {"hands": "take hold of the iron", "legs": "stay",
                        "waist": "stand"}})
    for t in range(ticks):
        if t == 12:                          # post in hand; now the other hand
            w.policy = _Wants(each={
                "the puller": {"hands": "take hold of the one pulled",
                               "legs": "go(straight on", "waist": "stand"},
                "the one pulled": {"hands": "hands free", "legs": "stay",
                                   "waist": "stand"},
                "the rescuer": {"hands": "catch", "legs": "stay",
                                "waist": "lean out"}})
        w.step()
        for q in (puller, falls, saves):
            for e in q["events"]:
                print(e, flush=True)
            q["events"].clear()
        if t % every == 0 or t == ticks - 1:
            _save(w, frames_dir, t)


# ── scenario: a man goes down under a low beam ───────────────────────────────
def crouch(frames_dir, ticks=120, every=1):
    """A man sinks, and a beam says how far he had to.

    Watch the LEGS, not the height. A crouch here is not a fold — at 5 cm a
    knee crease is one cell wide and there is nowhere to draw it — so the legs
    go SHORTER AND THICKER, which is what a crouching man's legs actually do to
    look at. The feet never leave the floor and not one gram goes anywhere: the
    surplus flesh bulges out where a folded leg bulges.

    How deep is read off his own legs by `_crouch_of` — 0.252 m for this man,
    five voxels — so a taller one would go further and nobody wrote that down.
    Past the knee angle that uses it up, bending harder buys nothing, and the
    frames show him stop."""
    w = World(26, 20, 34, voxel_cm=5)
    w.fill(0, 26, 0, 20, 0, 1, STONE)                    # the floor
    w.exits = [(24, 10)]
    p = _person(w, 10, 10)
    p["name"], p["facing"] = "the man", (0.0, 1.0)
    w.policy = _Wants(legs="stay")                       # he is not going out
    # A POST BESIDE HIM to read his height against, and nothing else — the
    # first version of this scene put a beam overhead with nothing holding it
    # up, and a slab with nothing under it falls. It came down on his head
    # while he was crouching under it, and the crouch took the blame.
    w.fill(18, 20, 9, 11, 1, 31, STONE)                  # as tall as he stands
    w.step()
    _save(w, frames_dir, 0)
    deep = w._crouch_of(p) / (0.1 * w.scale)
    print(f"t0: his legs allow {deep:.2f} voxels of crouch", flush=True)
    stood = int(np.argwhere(w.mat == FLESH)[:, 2].max())
    t = 1
    for th in list(np.arange(0.1, 1.45, 0.05)) + list(np.arange(1.4, 0.05, -0.1)):
        v = w._crouch(p, float(th))
        w.step()
        top = int(np.argwhere(w.mat == FLESH)[:, 2].max())
        legs = np.concatenate([np.asarray(p["segs"][b])
                               for b in ("left shin", "left thigh")])
        print(f"t{t}: knee {th:.2f} rad — {v}, down {stood - top} voxels, "
              f"leg {int(legs[:, 2].max() - legs[:, 2].min()) + 1} tall, "
              f"{w.total_mass(FLESH) / 1000.0:.3f} kg", flush=True)
        if t % every == 0:
            _save(w, frames_dir, t)
        t += 1


# ── scenario: three men, three shoulders ─────────────────────────────────────
def arms(frames_dir, ticks=170, every=2):
    """The same arm, three times, and every difference between them is physical.

    All three are told to put an arm out and hold it there. Same build, same
    shoulder, same command.

    The FIRST has an empty hand and gets there fastest.

    The SECOND is holding a block of iron. What is in the hand is part of the
    arm, so it is in the inertia, so it is in the TIME — his arm climbs visibly
    behind the other man's, and nobody wrote down that it should.

    The THIRD is knocked out partway through. Muscle tone is something a living
    waking body spends energy on every tick; when it stops there is no longer
    anything holding the joint, gravity has a moment about it, and his arm
    falls, swings past the bottom, comes back and settles. How long that takes
    falls out of the limb's own mass and inertia — 1.117 s measured against the
    1.083 s a pendulum of those numbers asks for.

    THEY FACE ALONG Y ON PURPOSE, and toward the camera. A shoulder swings in
    the plane the body faces, and a bone cannot be turned in the plane it is
    thin in: these arms are one voxel across, so facing along y they have 30 of
    30 angles and facing along x they have ONE. That is the body's largest
    remaining fault, and it is not hidden here — it is why the scene is built
    this way round."""
    w = World(28, 20, 36, voxel_cm=5)
    w.fill(0, 28, 0, 20, 0, 1, STONE)
    w.exits = []
    men = []
    for name, px in (("the empty hand", 4), ("the laden man", 12),
                     ("the one out cold", 20)):
        p = _person(w, px, 13)
        p["name"], p["facing"] = name, (0.0, -1.0)
        p["strength_N"] = 4000.0      # this scene is about TIME, not lifting
        men.append(p)
    # A BLOCK ON A PEDESTAL at the middle man's own hand height.
    w.fill(14, 17, 7, 10, 1, 21, STONE)
    w.fill(14, 17, 7, 10, 21, 24, IRON)
    w.policy = _Wants(each={"the laden man":
                            {"hands": "take hold of the iron"}})
    for _ in range(40):                           # he takes it up
        w.step()
        for p in men:
            p["events"].clear()
    held = w._held_cells(men[1])
    kg = 0.0 if held is None else \
        float(w.smass[tuple(np.asarray(held).T)].sum()) / 1000.0
    print(f"t0: the laden man has {kg:.1f} kg in his hand", flush=True)
    for p in men:
        got = w._swing_of(p, (p.get("held") or {}).get("arm", "right arm"))
        if got:
            print(f"t0: {p['name']:18s} arm inertia {got[0]:.3f} kg m2  "
                  f"({got[1]:.1f} kg hanging off the joint)", flush=True)
    w.policy = _Wants()
    _save(w, frames_dir, 0)
    ang = lambda p: float(np.atleast_1d((p.get("pose") or {}).get(
        (p.get("held") or {}).get("arm", "right arm"), 0.0))[0])
    for t in range(1, ticks):
        for p in men:
            if p["awake"]:
                arm = (p.get("held") or {}).get("arm", "right arm")
                p["reach"] = {arm: [1.2, 0.0, 0.0, 0.0]}
        # AND ONE OF THEM STOPS PAYING FOR IT. Held under by the scene rather
        # than hurt by it: an uninjured man comes round on his own, which is
        # `_law_life` being right and had to be worked around rather than
        # broken. What the scene is showing is a joint with no muscle on it.
        if t >= 70:
            men[2]["awake"] = False
            men[2]["reach"] = {}
            if t == 70:
                print("t70: the third man goes out", flush=True)
        w.step()
        for p in men:
            p["events"].clear()
        if t % every == 0 or t == ticks - 1:
            _save(w, frames_dir, t)
        if t % (2 if t < 40 else 10) == 0:
            print("t%-4d " % t + "   ".join(
                f"{p['name'].split()[-1]:5s} {ang(p):+.2f}" for p in men),
                flush=True)


SCENARIOS = {"rescue": rescue, "two_rooms": two_rooms, "tree_fell": tree_fell, "lamp_shelf": lamp_shelf,
             "drop_test": drop_test, "forge": forge, "fire_alarm": fire_alarm,
             "glasshouse": glasshouse,
             "alchemist": alchemist, "house_fire": house_fire,
             "burning_tree": burning_tree,
             "jumper": jumper, "parachute": parachute, "leap": leap,
             "swing": swing, "ledge": ledge, "crouch": crouch, "arms": arms,
             "tree_hinge": tree_hinge, "acid_bath": acid_bath, "torch_pillar": torch_pillar,
             "vessels": vessels, "pond": pond}

if __name__ == "__main__":
    name, frames_dir = sys.argv[1], sys.argv[2]
    os.makedirs(frames_dir, exist_ok=True)
    SCENARIOS[name](frames_dir)
    print("frames written to", frames_dir, flush=True)
