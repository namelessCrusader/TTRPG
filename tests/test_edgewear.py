"""EDGEWEAR (wave 13c, C8b): the edge as state, wear, roll, hone, filings, and
the tip/wedge wound split. The closed forms are tests/test_matedge.py's; these
test the glue in a World."""
import numpy as np

from src.voxel import edgewear as EW
from src.voxel import matedge as ME
from src.voxel.sim import IRON, STONE, WOOD, World
from src.voxel.scenes import _person

AXE = 2e-4


def _world(target):
    w = World(20, 10, 20, voxel_cm=5)
    w.fill(0, 20, 0, 10, 0, 1, STONE)
    w.fill(10, 12, 4, 6, 1, 3, target)
    w.fill(5, 7, 4, 5, 1, 2, IRON, edge=AXE)
    b = w._launch(np.array([[5, 4, 1], [6, 4, 1]]), (0, 0, 0))
    return w, b


def _hit(w, b, J, n=1):
    s = np.array([[10, 4, 1]])
    for _ in range(n):
        pl = EW.plan(w, b, s)
        EW.after_blow(w, b, s, J, pl)


def test_with_the_gates_OFF_nothing_is_kept_and_the_edge_never_moves():
    w, b = _world(STONE)
    w.mat_laws = frozenset()
    _hit(w, b, 50.0, 3)
    assert "edge_st" not in b and float(b["edges"][0]) == np.float32(AXE)
    assert EW.plan(w, b, np.array([[10, 4, 1]])) is None


def test_chopping_WOOD_wears_the_tip_and_does_not_touch_the_array():
    w, b = _world(WOOD)
    _hit(w, b, 30.0, 300)
    st = b["edge_st"]
    assert st["r"] > 1.0e-4                      # it wore ...
    assert st["r"] < 1.01e-4                     # ... by less than one percent
    assert float(b["edges"][0]) == np.float32(AXE)   # so the array was not rewritten


def test_ONE_blow_on_STONE_rolls_the_edge_and_the_area_follows():
    w, b = _world(STONE)
    _hit(w, b, 50.0)
    st = b["edge_st"]
    assert 3.5e-4 < st["r"] < 5.5e-4             # matedge: 50 J of iron on stone, about 0.47 mm
    A = ME.area(ME.EdgeGeom(st["L"], st["theta"], st["r"]))
    assert abs(float(b["edges"][0]) - A) < 1e-6 * A
    assert float(b["edges"][0]) > 1.3 * AXE
    assert (b["edges"] == b["edges"][0]).all()   # both cells of the bit moved together


def test_a_FLESH_target_never_rolls_an_edge():
    w, b = _world(STONE)
    w.mat[10:12, 4:6, 1:3] = 4                   # FLESH
    _hit(w, b, 60.0, 5)
    assert b["edge_st"]["r"] < 1.01e-4


def test_filings_leave_the_tool_as_rubble_and_the_mass_adds_up():
    w, b = _world(STONE)
    m0 = float(w.smass.sum()) + float(b["masses"].sum())
    e0 = float(w.E.sum()) + float(b["Es"].sum())
    tool0 = float(b["masses"].sum())
    _hit(w, b, 50.0, 400)
    m1 = float(w.smass.sum()) + float(b["masses"].sum())
    e1 = float(w.E.sum()) + float(b["Es"].sum())
    assert abs(m1 - m0) < 1e-6 * m0
    assert abs(e1 - e0) < 1e-6 * max(abs(e0), 1.0)
    assert float(b["masses"].sum()) <= tool0 - 1.0   # at least a gram left the tool
    assert w.rubble[9, 4, 1] or w.rubble[10, 4, 2] or w.rubble.any()


def _person_world():
    w = World(30, 20, 40, voxel_cm=5)
    w.fill(0, 30, 0, 20, 0, 1, STONE)
    p = _person(w, 12, 10)
    c = w._bone_parts(p, ["right upper arm"])[0]
    return w, p, tuple(int(v) for v in c[np.argmax(c[:, 0])])


def test_the_SPLIT_gives_a_new_edge_the_old_skin_energy():
    tip = ME.tip_area(ME.geom_from_area(AXE))
    for J, opens in ((19.0, False), (22.0, True)):
        w, p, cell = _person_world()
        old = w._blow(p, cell, J, AXE)
        w2, p2, cell2 = _person_world()
        new = w2._blow(p2, cell2, J, AXE, tip=tip)
        assert old["open"] is opens and new["open"] is opens


def test_a_ROLLED_edge_bruises_where_the_new_one_cuts():
    g = ME.geom_from_area(AXE)
    rolled = ME.EdgeGeom(g.L_m, g.theta_rad, 4.66e-4)
    w, p, cell = _person_world()
    sharp = w._blow(p, cell, 40.0, AXE, tip=ME.tip_area(g))
    w2, p2, cell2 = _person_world()
    dull = w2._blow(p2, cell2, 40.0, ME.area(rolled), tip=ME.tip_area(rolled))
    assert sharp["open"] and not dull["open"]      # 97 J to open it, against 21
    assert p2["hurt"] > 0                          # but it still hurt


def test_HONING_brings_a_rolled_edge_back_and_a_soft_stone_does_nothing():
    from src.voxel.scenes import thing
    w = World(20, 10, 20, voxel_cm=5)
    w.fill(0, 20, 0, 10, 0, 1, STONE)
    with thing(w, "axe", kind="tool", parts={"head": IRON}):
        w.fill(5, 7, 4, 5, 1, 2, IRON, edge=AXE)
    eid = next(k for k, t in w.things.items() if t["name"] == "axe")
    g = ME.geom_from_area(AXE)
    # roll it by hand first: a state at 466 um with the array following
    st = {"L": g.L_m, "theta": g.theta_rad, "r": 4.66e-4, "A_w": 0.0, "r_w": 4.66e-4, "owed_g": 0.0}
    A = ME.area(ME.EdgeGeom(g.L_m, g.theta_rad, 4.66e-4))
    w.edge[5:7, 4, 1] = np.float32(A)
    st["A_w"] = float(np.float32(A))
    w.things[eid]["edge_st"] = st
    soft = EW.hone_thing(w, eid, 500, 20.0, 0.1, 0.5e9, 70e-6)      # a stone softer than iron (1.2 GPa)
    assert soft > 4.6e-4                                            # K_soft: nothing in an afternoon
    r = EW.hone_thing(w, eid, 2000, 20.0, 0.1, 6e9, 70e-6)          # a river stone (granite, 6 GPa)
    assert r < 1.2e-4
    assert float(w.edge[5, 4, 1]) < 0.7 * A
    assert w.things[eid]["edge_st"]["owed_g"] > 0.0
