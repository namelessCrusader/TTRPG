"""S2 item 4a: the long-flight trace and the planner footprint.

tools/stairs_trace.py (the harness) named the cause of the long-flight failures
(see the lane report, w15b1/1c): the person's own will and a second driver both
write the one `p["_path"]`, which walks the body back to the lip; the reflex
pose then puts the arms out and the WHOLE-SHADOW footprint (12 cells long, three
treads) fits nowhere, so the plan is empty (699 of 736 plans); the same
positions route with the STANCE footprint. With the will off, 4, 6, 8 and 12
step flights are all climbed and no free body appears.

These tests guard what is built: the harness runs; the solo climb of a
12 x 20 cm flight (the rigid body does not tip when one writer drives); the
stance footprint's definition; and the planner gate (default OFF, bit for bit
the old footprint; ON routes where the whole shadow cannot).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))

from src.voxel import routing                                  # noqa: E402


def test_default_planner_footprint_is_the_whole_shadow():
    from src.voxel.pathlaw import PathingMixin
    from src.voxel.sim import World
    assert PathingMixin.stance_footprint is False
    assert World.stance_footprint is False
    from src.voxel.scenes_stairs import kerb_yard
    w, p = kerb_yard()
    comp, sl = w._person_cells(p)
    c = w._claim_cells(p, comp, sl)
    a, b = w._plan_footprint(c), routing.footprint(c)
    assert a[0].tolist() == b[0].tolist() and a[1].tolist() == b[1].tolist()


def test_stance_footprint_is_the_soles_and_the_trunk_not_the_raised_arms():
    # a body 5 wide: soles at z 0 in columns x 0..1, trunk column x 0, one arm
    # held out forward at z 5 over x 2..8
    cells = [(x, 0, 0) for x in (0, 1)] + [(0, 0, z) for z in range(1, 6)] \
        + [(x, 0, 5) for x in range(2, 9)]
    c = np.array(cells, np.int64)
    whole = routing.footprint(c)
    stance = routing.stance_footprint(c)
    cx = int(round(float(c[:, 0].mean())))
    assert whole[0].max() - whole[0].min() >= 8
    offs = sorted(zip(stance[0].tolist(), stance[1].tolist()))
    # the sole columns (offsets from the centre column), the centre left out
    assert offs == sorted((x - cx, 0) for x in (0, 1) if x - cx != 0)
    assert len(stance[0]) < len(whole[0])


def _flight_world():
    from src.voxel.scenes_stairs import stair_flight
    return stair_flight(steps=12, riser=4, tread=6)


def test_trace_harness_solo_climbs_a_12_x_20_cm_flight_and_never_tips():
    """T1 (the part the trace proved): one driver, the person's will off, the
    stock rigid body climbs 12 x 20 cm and never becomes a free body."""
    import stairs_trace as st
    st.COUNT.clear()
    w, p, rows, info = st.run(steps=12, riser=4, tread=6, ticks=600, solo=True)
    assert info["zmax"] >= info["top"], info
    assert info["first_free"] is None and not w.bodies
    assert info["first_down"] is None


def test_gate_on_routes_from_a_pose_the_whole_shadow_cannot():
    """The position the trace found (with-will 12 x 20 cm, tick 197): the
    whole-shadow plan is empty, the stance plan is not. MUTATION: gate off
    again -> the second assertion fails."""
    import stairs_trace as st
    st.COUNT.clear()
    w, p, rows, info = st.run(steps=12, riser=4, tread=6, ticks=199, solo=False)
    cells = st._cells(w, p)
    goal = (w.shape[0] - 6, 7)
    ex = [goal]
    w.stance_footprint = False
    full = w._routes(cells, ex, p).get(goal)
    w.stance_footprint = True
    stance = w._routes(cells, ex, p).get(goal)
    w.stance_footprint = False
    assert stance, "the stance footprint finds the way up from where the body stands"
    assert not full, "(the finding) the whole shadow finds none"


def test_the_harness_leaves_the_world_class_as_it_found_it():
    import stairs_trace as st
    from src.voxel import sim
    before = {n: sim.World._step_to_k for n in ("a",)}
    keep = (sim.World._step_to_k, sim.World._plan_path, sim.World._law_will,
            sim.World._balance_reflex)
    st.run(steps=4, riser=3, tread=6, ticks=30, solo=True)
    assert keep == (sim.World._step_to_k, sim.World._plan_path, sim.World._law_will,
                    sim.World._balance_reflex)
