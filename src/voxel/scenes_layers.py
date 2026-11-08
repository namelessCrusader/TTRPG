"""Proof-scene builders for the surface layers (T3, wave 13c lane C4). No scene code:
the room, the wood and the lit match are all the scene is; the soot comes from the
smoke law, the deposition law and the emissivity read-out.

    sooty_room   P2: a closed stone room with a door crack and a wood crib burning in it
"""
from .materials import AIR, STONE, WOOD
from .sim import World


def sooty_room(nx=50, ny=50, nz=25, voxel_cm=10, crib=3, layers=True, soot=True,
               match_w=20000.0, match_s=200.0):
    """A stone room `nx` x `ny` x `nz` voxels inside (5 x 5 x 2.5 m at 10 cm), one
    voxel of door crack low in a wall so the fire does not stifle at once, a wood crib
    in the middle lit at its base by a `match_w` watt flame held for `match_s` seconds. `layers` False builds the world without calling enable_layers (the class default is ON; a caller who wants the old world sets `w.surface_laws = False`)."""
    w = World(nx + 2, ny + 2, nz + 2, voxel_cm=voxel_cm)
    w.open_sky = False
    w.fill(0, nx + 2, 0, ny + 2, 0, nz + 2, STONE)
    w.fill(1, nx + 1, 1, ny + 1, 1, nz + 1, AIR)
    w.smass[1:nx + 1, 1:ny + 1, 1:nz + 1] = 0.0
    w.mat[0:1, ny // 2:ny // 2 + 1, 1:2] = AIR                     # the door crack
    w.smass[0:1, ny // 2:ny // 2 + 1, 1:2] = 0.0
    cx, cy = nx // 2, ny // 2
    for k in range(crib):                                           # a crib of sticks
        z = 1 + 2 * k
        w.fill(cx - 3, cx + 3, cy - 1, cy, z, z + 1, WOOD)
        w.fill(cx - 3, cx + 3, cy + 1, cy + 2, z, z + 1, WOOD)
        w.fill(cx - 1, cx, cy - 3, cy + 3, z + 1, z + 2, WOOD)
        w.fill(cx + 1, cx + 2, cy - 3, cy + 3, z + 1, z + 2, WOOD)
    w.ignite(cx - 2, cy - 1, 1, match_w, match_s)
    if layers:
        w.enable_layers(soot=soot)
    return w
