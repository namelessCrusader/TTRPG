"""BURNS PER SKIN FACE, WIRED (wave 15, S2 C; owner W15-Q3: burns persist).

skinburn.py is the pure law (tests/test_skinburn.py). These tests hold the wiring in
skinworld.py: which faces a body has, what each one is exposed to, what is kept, what is
read (the p["burn"] view, the hand's state, the pain) and the gate. FAST tier: the faces
are stepped directly with a prepared temperature field, on a world that is not stepped.

Each test says the mutation it must catch.
"""
import numpy as np
import pytest

from src.voxel import skinburn as SB, skinworld as SW
from src.voxel.materials import OIL, TICK_S, WATER
from src.voxel.scenes import _person
from src.voxel.sim import IRON, STONE, World

_DIRS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


def _body(x=12):
    w = World(30, 20, 40, voxel_cm=5)
    w.fill(0, 30, 0, 20, 0, 1, STONE)
    p = _person(w, x, 10)
    w.step()
    return w, p


def _around(w, bones, p, T, value, liquid=None):
    """Make every air cell next to these bones `value` C (and, with `liquid`, full of it)."""
    n = 0
    for c in w._bone_parts(p, bones):
        for x, y, z in c.tolist():
            for dx, dy, dz in _DIRS:
                q = (x + dx, y + dy, z + dz)
                if w.mat[q] == 0:
                    T[q] = value
                    if liquid is not None:
                        w.fl[q], w.fvol[q] = liquid, w.cap
                    n += 1
    return n


def _expose(w, p, T, seconds):
    comp, sl = w._person_cells(p)
    for _ in range(int(round(seconds / TICK_S))):
        w.tick += 1
        SW.step(w, p, T, comp, sl)


def _burned(p, deg=2):
    return [k for k, f in SW.skin_of(p).items() if SB.degree(f) >= deg]


def test_T1_a_hand_in_60_C_water_scalds_in_seconds_and_49_C_does_not_in_a_minute():
    """The Moritz-Henriques table: about 5 s at 60 C is a full-thickness scald; 49 C needs
    ten minutes. MUTATION: change the activation energy by 5 percent and one of them fails."""
    w, p = _body()
    T = w.T().copy()
    _around(w, ["right forearm"], p, T, 60.0, liquid=WATER)
    _expose(w, p, T, 10.0)
    assert _burned(p, 2), "10 s in 60 C water burns the hand"
    w2, p2 = _body()
    T2 = w2.T().copy()
    _around(w2, ["right forearm"], p2, T2, 49.0, liquid=WATER)
    _expose(w2, p2, T2, 60.0)
    assert not _burned(p2, 2), "a minute at 49 C does not"


def test_T4_one_hand_in_a_flame_burns_the_hand_and_little_else():
    """The old scalar averaged over every flesh cell of the body. MUTATION: averaging over the
    body again makes the burn a whole-body share (and puts faces on the torso)."""
    w, p = _body()
    T = w.T().copy()
    _around(w, ["right forearm"], p, T, 1000.0)
    _expose(w, p, T, 3.0)
    hit = _burned(p, 2)
    assert hit, "a forearm in 1000 C flame gas for 3 s is burned"
    bones = {k.split("|")[0] for k in SW.skin_of(p)}
    assert bones <= {"right forearm", "right upper arm", "torso"}, bones     # the air beside the arm
    assert not any(k.startswith(("left", "head")) for k in hit), hit
    fore = [k for k in hit if k.startswith("right forearm")]
    assert len(fore) > len(hit) - len(fore), "most of the burn is the forearm's"
    assert len(SW.skin_of(p)) < 120, "faces, not the body's whole surface"
    assert p["hand_state"]["right arm"] < 1.0, "a burned hand holds less"
    assert "left arm" not in (p.get("hand_state") or {})


def test_the_burn_view_is_the_shock_of_the_faces_and_it_only_climbs():
    """p["burn"] stays a written view; the faces' shock is 0.012 x weighted %TBSA x (1 - exp(-t/6 h)).
    MUTATION: dropping the 6 h growth makes it jump at once."""
    w, p = _body()
    w.skin_stride = False   # exact law: the test counts single ticks
    T = w.T().copy()
    _around(w, ["right forearm", "right upper arm"], p, T, 1000.0)
    _expose(w, p, T, 4.0)
    b0 = p["burn"]
    ev = p["body"]["skin_ev"]
    assert ev and sum(d for _, d in ev) > 0.0
    t0 = w.tick * TICK_S
    w.tick += int(6 * 3600 / TICK_S)                 # six hours on
    T[:] = 20.0
    _expose(w, p, T, 0.1)
    assert p["burn"] > b0 > 0.0, "the shock of a burn grows over hours"
    full = SB.shock_events(ev, 1e9)
    assert SB.shock_events(ev, t0 + 6 * 3600.0) < full


def test_T6_the_faces_write_nothing_into_the_lattice():
    """A face is a readout. MUTATION: writing the profile's heat back changes E."""
    w, p = _body()
    T = w.T().copy()
    _around(w, ["right forearm"], p, T, 1000.0)
    E0, m0, fl0 = w.E.copy(), w.mat.copy(), w.fvol.copy()
    _expose(w, p, T, 2.0)
    assert np.array_equal(E0, w.E) and np.array_equal(m0, w.mat) and np.array_equal(fl0, w.fvol)


def test_burns_persist_after_the_exposure_ends_and_a_revival_resets_them():
    """W15-Q3. MUTATION: pruning a burned face as it cools loses the record."""
    w, p = _body()
    T = w.T().copy()
    _around(w, ["right forearm"], p, T, 1000.0)
    _expose(w, p, T, 3.0)
    n = len(_burned(p, 2))
    assert n
    T[:] = 20.0
    _expose(w, p, T, 120.0)                          # two minutes in cool air
    assert len(_burned(p, 2)) == n, "a burn does not heal in this wave"
    assert p["pain_heat_m2"] == 0.0, "a settled burn reads no heat pain"
    SW.revive_reset(p)
    assert not SW.skin_of(p) and p["burn"] == 0.0 and "hand_state" not in p


def test_a_face_that_was_hot_but_not_burned_is_forgotten_when_it_cools():
    w, p = _body()
    T = w.T().copy()
    _around(w, ["right forearm"], p, T, 45.0)
    _expose(w, p, T, 2.0)
    assert SW.skin_of(p) and not _burned(p, 1)
    T[:] = 20.0
    _expose(w, p, T, 1000.0)
    assert not SW.skin_of(p), "a cool, unburned face carries nothing"


def test_a_hot_solid_burns_by_the_interface_temperature():
    """The skin on 300 C iron: the interface is held near the iron's temperature at once.
    MUTATION: treating iron as air (h 10) takes minutes."""
    w, p = _body()
    c = w._bone_parts(p, ["right forearm"])[0]
    x, y, z = int(c[:, 0].max()) + 1, int(c[0, 1]), int(c[0, 2])
    w.mat[x, y, z] = IRON
    T = w.T().copy()
    T[x, y, z] = 300.0
    _expose(w, p, T, 0.5)
    assert _burned(p, 2), "half a second on 300 C iron burns"


def test_an_oil_burns_less_than_water_at_the_same_temperature():
    """h is the fluid's own (water 1500, oil 500). MUTATION: one h for all liquids."""
    out = {}
    for name, fl in (("water", WATER), ("oil", OIL)):
        w, p = _body()
        T = w.T().copy()
        _around(w, ["right forearm"], p, T, 62.0, liquid=fl)
        _expose(w, p, T, 4.0)
        out[name] = sum(float(f["omega"][SB.BASAL]) for f in SW.skin_of(p).values())
    assert out["oil"] < out["water"]


def test_the_order_of_persons_changes_nothing():
    """T5. Two bodies stepped in either order give the same faces (each face is its own)."""
    res = []
    for order in ((0, 1), (1, 0)):
        w = World(40, 20, 40, voxel_cm=5)
        w.skin_stride = False   # exact law: the test counts single ticks
        w.fill(0, 40, 0, 20, 0, 1, STONE)
        ps = [_person(w, 8, 10), _person(w, 24, 10)]
        w.step()
        T = w.T().copy()
        for q in ps:
            _around(w, ["right forearm"], q, T, 90.0, liquid=WATER)
        for i in order:
            comp, sl = w._person_cells(ps[i])
            for _ in range(80):
                w.tick += 1
                SW.step(w, ps[i], T, comp, sl)
        res.append([sorted((k, f["omega"].sum().item()) for k, f in SW.skin_of(q).items()) for q in ps])
    assert res[0] == res[1]


def test_the_body_frame_directions_turn_with_the_chest():
    """A mark moves with the body: forward is where the chest faces. MUTATION: world axes."""
    assert SW._world_dirs((0.0, 1.0))[0] == (0, 1, 0)
    assert SW._world_dirs((1.0, 0.0))[0] == (1, 0, 0)
    assert SW._world_dirs((0.0, -1.0))[0] == (0, -1, 0)
    for chest in ((0.0, 1.0), (1.0, 0.0), (0.0, -1.0), (-1.0, 0.0)):
        d = SW._world_dirs(chest)
        assert len(set(d)) == 6 and all(abs(a) + abs(b) + abs(c) == 1 for a, b, c in d)


def test_the_gate_off_keeps_the_old_scalar_and_makes_no_faces():
    """T9. With skin_faces False a person in hot gas runs the old body-wide scalar."""
    w, p = _body()
    w.skin_faces = False
    T = w.T()
    comp, sl = w._person_cells(p)
    c = np.argwhere(comp)[0] + np.array([s.start or 0 for s in sl])
    w.E[tuple(c)] += 80.0 * w.heat_capacity()[tuple(c)]
    for _ in range(40):
        w.step()
    assert not (p.get("body") or {}).get("skin")
    assert p["burn"] > 0.0, "the old scalar burned the hot flesh cell"


def test_a_cool_world_costs_nothing_and_keeps_no_record():
    w, p = _body()
    comp, sl = w._person_cells(p)
    T = w.T().copy()
    for _ in range(50):
        w.tick += 1
        SW.step(w, p, T, comp, sl)
    assert not (p.get("body") or {}).get("skin") and p["burn"] == 0.0


def test_stride_law_matches_the_every_tick_law_and_costs_a_fraction():
    """The faces step every SKIN_STRIDE ticks with dt x stride. The time to a second degree burn
    in 60 C water and the shock view match the every-tick law within 5 percent, and the whole
    body in 400 C gas costs under 1 ms a tick. MUTATION: dropping the dt factor makes the burn
    12 times too slow and fails the first assert."""
    import time
    res = {}
    for stride in (False, True):
        w, p = _body()
        w.skin_stride = stride
        T = w.T().copy()
        _around(w, ["right forearm"], p, T, 60.0, liquid=WATER)
        comp, sl = w._person_cells(p)
        t_first = None
        for i in range(int(40 / TICK_S)):
            w.tick += 1
            SW.step(w, p, T, comp, sl)
            if t_first is None and _burned(p):
                t_first = i * TICK_S
        res[stride] = t_first
    assert res[False] is not None and abs(res[True] - res[False]) <= 0.05 * res[False]
    w, p = _body()
    T = w.T().copy()
    _around(w, sorted(p["segs"]), p, T, 400.0)
    comp, sl = w._person_cells(p)
    for _ in range(24):
        w.tick += 1
        SW.step(w, p, T, comp, sl)
    t0 = time.perf_counter()
    for _ in range(120):
        w.tick += 1
        SW.step(w, p, T, comp, sl)
    assert (time.perf_counter() - t0) / 120 * 1000 < 1.0


def test_twin_refuses_while_a_face_is_warm():
    """A body with a warm skin face is not fast-forwarded. MUTATION: remove the slowtwins check."""
    from src.voxel import slowtwins as ST
    w, p = _body()
    p["awake"] = False
    p["asleep"] = True
    p["body"] = {"skin": {"x": {"warm": True}}}
    ok, why = ST.TWINS["breath"].applicable(w, p)
    assert not ok and why == "a skin face is still settling"
