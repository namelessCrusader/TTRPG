"""Blender headless renderer for MPM particle frames.

Usage: blender --background --python render_mpm.py -- <frames_dir> <out_dir>

Reads each f####.npz holding `x` (particle positions in METRES), `kind` (which
material each one is), `names`, plus the static `wall` grid and its spacing.
Draws every particle as a small cube at its real position — which is the whole
point of the exercise, since a particle is not on the lattice and a renderer
that snapped it to one would hide exactly the thing being tested.

Shares the voxel renderer's camera reasoning: frame what is IN the world, not
the box it came in.
"""
import os
import sys

import bpy
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:]
FRAMES, OUT = argv[0], argv[1]
os.makedirs(OUT, exist_ok=True)

CATS = {
    "wood":  ((0.42, 0.26, 0.12, 1.0), 0.55),
    "stone": ((0.66, 0.60, 0.50, 1.0), 0.55),
    "iron":  ((0.36, 0.38, 0.42, 1.0), 0.55),
    "glass": ((0.62, 0.80, 0.85, 1.0), 0.50),
    "flesh": ((0.80, 0.60, 0.45, 1.0), 0.55),
    "lead":  ((0.30, 0.30, 0.34, 1.0), 0.55),
    "wall":  ((0.50, 0.47, 0.43, 1.0), 1.00),
}

CUBE_V = np.array([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                   (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)], np.float32)
CUBE_F = [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1),
          (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]


def mesh_from(coords, name, color, size):
    n = len(coords)
    if n == 0:
        return
    verts = (coords[:, None, :] + (CUBE_V[None] - 0.5) * size).reshape(-1, 3)
    faces = (np.array(CUBE_F, np.int64)[None]
             + (np.arange(n) * 8)[:, None, None]).reshape(-1, 4)
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


def setup_scene(lo, hi):
    from mathutils import Vector
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
    look = Vector((float(cx), float(cy), float(cz)))
    r = 0.5 * float(np.linalg.norm([ex, ey, ez]))
    sensor_h = 36.0 * sc.render.resolution_y / sc.render.resolution_x
    half_fov = np.arctan((sensor_h * 0.5) / cam.lens)
    dist = r / max(np.tan(half_fov), 1e-6) * 1.05
    # straight down the +y axis and a little above: these scenes are drawn in
    # PROFILE (a plank tipping, a bar sagging), and the tilt is the whole point
    away = Vector((-0.12, -0.97, 0.22))
    away.normalize()
    loc = look + away * dist
    co.location = loc
    co.rotation_euler = (look - loc).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.collection.objects.link(co)
    sc.camera = co


def render_frame(path, out_png):
    d = np.load(path, allow_pickle=True)
    x, kind = d["x"], d["kind"]
    names = [str(s) for s in d["names"]]
    dx = float(d["dx"])
    wall = d["wall"]
    clear()
    wc = np.argwhere(wall).astype(np.float64) * dx
    allpts = np.concatenate([x, wc]) if len(wc) else x
    setup_scene(allpts.min(axis=0), allpts.max(axis=0))
    if len(wc):
        mesh_from(wc, "wall", CATS["wall"][0], dx)
    for i, nm in enumerate(names):
        sel = x[kind == i]
        col, sz = CATS.get(nm, CATS["stone"])
        mesh_from(sel, f"p_{nm}", col, dx * sz * 0.5)
    bpy.context.scene.render.filepath = out_png
    bpy.ops.render.render(write_still=True)


frames = sorted(f for f in os.listdir(FRAMES) if f.endswith(".npz"))
for i, f in enumerate(frames):
    render_frame(os.path.join(FRAMES, f), os.path.join(OUT, f"r{i:04d}.png"))
    print(f"rendered {i + 1}/{len(frames)}", flush=True)
print("done")
