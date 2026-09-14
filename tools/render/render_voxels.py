"""Blender headless renderer for voxel frames.

Usage: blender --background --python render_voxels.py -- <frames_dir> <out_dir>

Reads each f####.npz, builds one mesh per material category (surface voxels only),
renders a PNG with the fast Workbench engine. Near walls are hidden so the camera
sees into the room. Fire = burning voxels (bright orange) + hot air (small amber
cubes). After all frames, a second pass in encode_video.py turns PNGs into an mp4.
"""
import os
import sys

import bpy
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:]
FRAMES, OUT = argv[0], argv[1]
os.makedirs(OUT, exist_ok=True)

AIR, WOOD, STONE, IRON, FLESH = 0, 1, 2, 3, 4
ASH, GLASS, LEAF, CHAR, TIN, LEAD = 5, 6, 7, 8, 9, 10
NOFLUID, WATER, OIL, ACID, WEAK_ACID = 0, 1, 2, 3, 4
MTIN, MLEAD, MIRON = 5, 6, 7

CATS = {   # name: (color rgba, cube scale)
    "stone": ((0.66, 0.60, 0.50, 1.0), 1.0),   # warm plaster — grey smoke must
                                               # READ against the walls
    "wood":  ((0.42, 0.26, 0.12, 1.0), 1.0),
    "flesh": ((0.80, 0.60, 0.45, 1.0), 1.0),
    "oil":   ((0.10, 0.08, 0.03, 1.0), 0.95),
    "water": ((0.15, 0.35, 0.85, 1.0), 0.95),
    "acid":  ((0.25, 0.90, 0.25, 1.0), 0.95),   # potent vitriol: vivid green
    "acidweak":  ((0.58, 0.72, 0.28, 1.0), 0.95),   # half-spent: murky yellow-green
    "acidspent": ((0.46, 0.43, 0.32, 1.0), 0.95),   # exhausted sludge: dull brown
    "vinegar":   ((0.68, 0.85, 0.55, 1.0), 0.95),   # aqua debilis: pale watery green
    "fire":  ((1.00, 0.35, 0.05, 1.0), 1.05),
    "flame": ((1.00, 0.55, 0.10, 1.0), 0.55),
    "char":  ((0.12, 0.10, 0.08, 1.0), 1.0),
    "glass": ((0.62, 0.80, 0.85, 1.0), 0.90),   # pale blue-green, drawn SMALL so
                                                # the fire reads through the pane
    "leaf":  ((0.24, 0.45, 0.16, 1.0), 0.80),   # hedge green
    "ash":   ((0.55, 0.53, 0.50, 1.0), 1.0),
    "iron":  ((0.36, 0.38, 0.42, 1.0), 1.0),
    "smoke": ((0.20, 0.19, 0.18, 1.0), 0.45),   # dark, small, and CHECKERED
}
SMOKE_SHOW = 0.006      # grams per voxel worth drawing — below this the air is
                        # hazy, not sooty, and drawing it hides the whole room.
                        # Even above it, smoke is drawn on a checkerboard at
                        # under half cube size: solid cubes at one per voxel make
                        # an opaque black slab you cannot see the fire through,
                        # and a smoke-logged room should obscure, not delete.

CUBE_V = np.array([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                   (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)], np.float32)
CUBE_F = [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1),
          (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]


def surface(mask, solid):
    """Voxels of mask with at least one non-solid neighbor (or on the boundary)."""
    keep = np.zeros_like(mask)
    pad = np.pad(solid, 1, constant_values=False)
    for ax in range(3):
        for d in (1, -1):
            sh = np.roll(pad, d, axis=ax)[1:-1, 1:-1, 1:-1]
            keep |= mask & ~sh
    return keep


def mesh_from(coords, name, color, scale):
    n = len(coords)
    if n == 0:
        return
    off = (1 - scale) / 2.0
    verts = (coords[:, None, :] + off + CUBE_V[None] * scale).reshape(-1, 3)
    faces = (np.array(CUBE_F, np.int64)[None] + (np.arange(n) * 8)[:, None, None]).reshape(-1, 4)
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts.tolist(), [], faces.tolist())
    ob = bpy.data.objects.new(name, me)
    m = bpy.data.materials.new(name)
    m.diffuse_color = color
    me.materials.append(m)
    bpy.context.scene.collection.objects.link(ob)


def clear():
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    for me in list(bpy.data.meshes):
        bpy.data.meshes.remove(me)
    for m in list(bpy.data.materials):
        bpy.data.materials.remove(m)


def setup_scene(shape, filled=None):
    """Frame WHAT IS IN the world, not the box it came in.

    The camera used to be pinned at one hand-tuned position, which only ever
    suited the world it was tuned on; then it was derived from the world's
    SHAPE, which is better and still wrong the moment a world has headroom. A
    scene with people standing on a ledge in a tall room aimed at 30% of the
    room's height and cut their heads off — measured, twice, because making
    the room taller to fix it moved the camera too and changed nothing.

    `filled` is the bounding box of everything that is not air. Aim at the
    middle of THAT and stand back by ITS size, and a world can be as empty
    above as it likes."""
    from mathutils import Vector
    nx, ny, nz = shape
    if filled is not None and len(filled):
        lo = filled.min(axis=0).astype(float)
        hi = filled.max(axis=0).astype(float) + 1.0
    else:
        lo, hi = np.zeros(3), np.array([nx, ny, nz], float)
    cx, cy, cz = (lo + hi) * 0.5
    ex, ey, ez = hi - lo
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_WORKBENCH"
    sc.display.shading.color_type = "MATERIAL"
    sc.display.shading.light = "STUDIO"
    sc.render.resolution_x, sc.render.resolution_y = 640, 360
    cam = bpy.data.cameras.new("cam")
    cam.lens = 40.0
    co = bpy.data.objects.new("cam", cam)
    # HOW FAR BACK THE LENS ACTUALLY NEEDS TO BE. Standing off by a multiple
    # of the content's size is a guess that holds for square rooms and fails
    # for tall ones — measured twice on the same scene, once by making the room
    # taller (which moved the camera and changed nothing) and once by framing
    # the content (which helped and still clipped). A 40 mm lens on a 36 mm
    # sensor at 640x360 sees about 28 degrees vertically; the distance that
    # fits a sphere of radius r into that is r / tan(fov/2), and everything
    # else is the direction to stand in.
    look = Vector((cx, cy, cz))
    r = 0.5 * float(np.linalg.norm([ex, ey, ez]))
    sensor_h = 36.0 * sc.render.resolution_y / sc.render.resolution_x
    half_fov = np.arctan((sensor_h * 0.5) / cam.lens)
    dist = r / max(np.tan(half_fov), 1e-6) * 1.05        # a little air round it
    # sit on the -x / -y side, above: those are the two walls the frame hides,
    # so the camera always looks INTO the room rather than at its back
    away = Vector((-0.62, -0.62, 0.48))
    away.normalize()
    loc = look + away * dist
    co.location = loc
    co.rotation_euler = (look - loc).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.collection.objects.link(co)
    sc.camera = co


def render_frame(path, out_png):
    d = np.load(path)
    mat, fl, T, burn = d["mat"], d["fl"], d["T"].astype(np.float32), d["burn"]
    solid = mat != AIR
    hide = np.zeros_like(solid)      # the two near walls and the roof: cut away
    hide[0:1, :, :] = True           # so the camera sees into the room
    hide[:, 0:1, :] = True
    hide[:, :, -1:] = True
    vis = solid & ~hide
    charred = (mat == WOOD) & ~burn & (T > 200)
    smoke = d["smoke"].astype(np.float32) if "smoke" in d \
        else np.zeros_like(T)
    gx, gy, gz = np.indices(mat.shape)
    CHECKER = ((gx + gy + gz) % 2 == 0)
    cats = {
        "fire":  burn & vis | (burn & (fl == OIL)),
        "char":  surface(charred & vis, solid),
        "stone": surface((mat == STONE) & vis & ~burn, solid),
        "wood":  surface((mat == WOOD) & vis & ~burn & ~charred, solid),
        "flesh": surface((mat == FLESH) & vis & ~burn, solid),
        "glass": surface((mat == GLASS) & vis & ~burn, solid),
        "leaf":  surface((mat == LEAF) & vis & ~burn, solid),
        "ash":   surface((mat == ASH) & vis, solid),
        "iron":  surface((mat == IRON) & vis, solid),
        # FLUID IN THE AIR IS A POOL; FLUID IN A SOLID IS WETNESS. These drew
        # a full-size cube wherever the fluid field was set, solid or not — so
        # a fresh canopy, which holds its own weight of sap on purpose and is
        # why it resists burning, came out as 2966 cubes of WATER and the tree
        # rendered blue. Poured fluid only ever enters AIR (`pour` checks), so
        # a fluid inside a solid is something the solid is carrying, and the
        # solid is what you should see.
        "oil":   (mat == AIR) & (fl == OIL) & ~burn,
        "water": (mat == AIR) & (fl == WATER),
        "acid":  (mat == AIR) & (fl == ACID),
        "flame": (mat == AIR) & (T > 400) & ~burn,
        "smoke": (mat == AIR) & ~hide & (smoke > SMOKE_SHOW) & CHECKER,
    }
    clear()
    setup_scene(mat.shape, np.argwhere(mat != 0))
    for name, mask in cats.items():
        color, scale = CATS[name]
        mesh_from(np.argwhere(mask).astype(np.float32), name, color, scale)
    if "drops" in d and len(d["drops"]):                 # parcels in flight
        mesh_from(d["drops"][:, :3].astype(np.float32), "drop",
                  CATS["water"][0], 0.35)
    if "bodies" in d and len(d["bodies"]):               # rigid bodies mid-topple
        bmat = {1: "wood", 2: "stone", 4: "flesh", 5: "ash", 6: "glass"}
        for mid, name in bmat.items():
            sel = d["bodies"][d["bodies"][:, 3] == mid]
            if len(sel):
                mesh_from(sel[:, :3].astype(np.float32), f"body_{name}",
                          CATS[name][0], 1.0)
    bpy.context.scene.render.filepath = out_png
    bpy.ops.render.render(write_still=True)


frames = sorted(f for f in os.listdir(FRAMES) if f.endswith(".npz"))
for i, f in enumerate(frames):
    render_frame(os.path.join(FRAMES, f), os.path.join(OUT, f"r{i:04d}.png"))
    print(f"rendered {i + 1}/{len(frames)}", flush=True)
print("done")
