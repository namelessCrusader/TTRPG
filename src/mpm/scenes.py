"""The three things a voxel cannot do, each as small as it can be made.

Drawn by the user, in this order: a plank tips off a table and must land
TURNED; a glass slab breaks on a table edge and its pieces must go two ways;
a bar melts over an overhang and must SAG rather than step. The first is here
because it has no chemistry attached — it tests the solve and nothing else.
"""

import numpy as np

from .core import MPM


def plank(stiff=1e-4):
    """A plank lying on a table with more of itself off the edge than on it.

    In the voxel sim this is `_law_torque` promoting a cluster to a free body,
    swinging it about a pivot line, and re-rasterising it onto the grid with
    `np.round` — which drops the rotation on the floor, so the plank that lands
    is indistinguishable from one that never turned. Here the turn is carried
    in the deformation gradient and the position is continuous, so where it
    lands and which way it is facing are both answers rather than roundings.
    """
    # The table is TALL and the plank is short, because the first cut of this
    # scene had a 1.6 m plank on a 0.8 m table and the plank's far end reached
    # the floor before it had turned far: it came to rest as a ramp at 40
    # degrees, which is correct physics and not the thing being tested.
    w = MPM((32, 14, 26), dx=0.1, stiff=stiff)
    w.add_wall((0.0, 0.0, 0.0), (3.2, 1.4, 0.2))         # floor
    w.add_wall((0.3, 0.3, 0.2), (1.3, 1.1, 1.6))         # table
    w.add_box((0.95, 0.5, 1.6), (1.95, 0.9, 1.8), "wood")  # .35 on, .65 off
    return w


def save(w, path):
    """One frame of particles, as the renderer wants them."""
    np.savez_compressed(path, x=w.x.astype(np.float32),
                        kind=w.kind, names=np.array(w.kinds),
                        dx=w.dx, wall=w.wall)


def long_axis(w):
    """Angle of the plank's long axis above horizontal, in degrees — the
    measurement that says whether it TURNED, which is the whole question."""
    c = w.x - w.x.mean(axis=0)
    vec = np.linalg.eigh(c.T @ c)[1][:, -1]
    return float(np.degrees(np.arcsin(abs(vec[2]))))


def glass_edge(stiff=1e-3):
    """A glass slab dropped so it straddles a table edge.

    The user's drawing: half of it lands on the table and half over the drop,
    it breaks at the edge, and the pieces go TWO WAYS — some staying up, some
    falling on. The voxel sim can nearly do this already (`_shatter` scatters
    into neighbours, `rubble` takes support only from directly beneath, so
    pieces past the edge do fall). What it cannot do is make a shard SMALLER
    THAN THE CELL THE PANE WAS DRAWN IN, or let one arrive anywhere but a cell
    centre. Here a fragment is whatever the cracks enclosed.
    """
    # stiff=1e-3 is a MEASURED FLOOR, not a taste. Bending fracture is the one
    # thing softening cannot be scaled out of: a compliant slab DRAPES over an
    # edge and relieves its own bending moment, so it never reaches the stress
    # that would break it. Measured on this scene, 1e-3 and 1e-2 agree (66 and
    # 77 pieces broken, largest chunk 414 and 403) and 1e-4 does not break at
    # all. Where two stiffnesses a decade apart give the same answer, the
    # cheaper one is enough.
    # DROPPED FROM TWO METRES, not from eighty centimetres. The short drop
    # arrives at 4 m/s and glass parts at about that speed, so it cracked a
    # corner off and slid: 85 pieces of 480, first break twenty ticks after
    # the edge, which is to say it broke on the FLOOR rather than on the edge
    # and the scene was not testing what it claimed. At 6.3 m/s it breaks on
    # contact and the pieces go two ways, which is the drawing.
    w = MPM((32, 14, 40), dx=0.1, stiff=stiff)
    w.add_wall((0.0, 0.0, 0.0), (3.2, 1.4, 0.2))         # floor
    w.add_wall((0.3, 0.3, 0.2), (1.5, 1.1, 1.2))         # table, edge at x=1.5
    # 0.7 m supported, 0.3 m over. Not half and half: with the slab centred
    # exactly on the edge its centre of mass sits ON the pivot, so any
    # rotation at all takes the whole thing over and NOTHING can stay up.
    # Measured across four offsets, the outcome is bimodal — at 0.70 m of
    # support 295 pieces stay on the table, at 0.65 m every one of the 480
    # ends on the floor.
    w.add_box((0.8, 0.5, 3.2), (1.8, 0.9, 3.35), "glass")
    return w


def resting_slab(stiff=1e-3):
    """The same slab set down gently, fully supported. It must NOT break —
    that is the other half of choosing a failure threshold, and the half a
    shatter test on its own would never catch."""
    w = MPM((32, 14, 26), dx=0.1, stiff=stiff)
    w.add_wall((0.0, 0.0, 0.0), (3.2, 1.4, 0.2))
    w.add_wall((0.3, 0.3, 0.2), (2.2, 1.1, 1.2))
    w.add_box((0.5, 0.5, 1.2), (1.5, 0.9, 1.35), "glass")
    return w
