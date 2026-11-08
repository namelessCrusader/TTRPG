"""Wave 13b, lane C6: body heat ON (T7). The guards of the switch.

What is switched on, and behind what:
  * BodyWorld (src/voxel/bodyworld.py) is the capability gate of body heat (N5):
    a person gets a heat record only in a BodyWorld with body_heat True. A plain
    World has none (golden says so on 35 cases; the first test here says it for
    a BodyWorld with the capability off).
  * gas-only buoyancy (air.py): a hot solid is not a buoyant gas. Global.
  * N2 (a) the ISO rest, the two-sided thermoregulation fix, the D2 sleep law of
    the reflex rows, and the two walking-leak cures.

The expensive ones are marked slow.
"""
import numpy as np
import pytest

from src.voxel import energetics as EN
from src.voxel import reflex as RF
from src.voxel import thermoreg as TR
from src.voxel import tissue as TI
from src.voxel.bodyworld import BodyWorld
from src.voxel.coldworld import ColdWorld
from src.voxel.scenes import _person, _Wants
from src.voxel.sim import AIR, AMBIENT, FLESH, STONE, World


def _room(cls=BodyWorld, n=32, nz=40, **kw):
    w = cls(n, n, nz, voxel_cm=5, **kw)
    w.fill(0, n, 0, n, 0, 1, STONE)
    p = _person(w, 12, 12)
    w.policy = _Wants()
    return w, p


# ================================================================ the gate ====
def test_the_capability_off_is_the_old_world_bit_for_bit():
    """BodyWorld with body_heat False and the old float32 E steps exactly as World does: no record, no overlay."""
    a, pa = _room(World)
    b, pb = _room(BodyWorld, E_dtype=np.float32)
    b.body_heat = False
    for _ in range(80):
        a.step()
        b.step()
    for name in ("E", "mat", "smass", "fvol", "o2", "smoke"):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name
    assert pb.get("heat") is None and b.face_env is None and b.bzone is None
    assert a.shed == b.shed


def test_a_plain_world_and_a_cold_world_have_no_body_heat():
    for cls in (World, ColdWorld):
        w, p = _room(cls)
        for _ in range(3):
            w.step()
        assert p.get("heat") is None and w.bzone is None


def test_a_bodyworld_gives_a_person_a_record_and_the_books_close():
    w, p = _room()
    e0 = w.total_energy()
    for _ in range(81):
        w.step()
    h = p["heat"]
    assert h["alive"] and 36.0 < h["T_pool"] < 38.0
    inj = w.bh_injected()
    assert inj["body_warmup"] > 1e5 and inj["stores_init"] > 1e7        # named, not leaked
    assert abs(w.total_energy() - e0) < 1.0                              # float64 lattice: books to the joule


def test_the_flesh_is_warm_and_the_lattice_energy_is_float64():
    w, p = _room()
    w.step()
    assert w.E.dtype == np.float64
    f = (w.mat == FLESH)
    T = AMBIENT + w.E[f] / w.heat_capacity()[f]
    assert T.max() > 33.0 and T.min() > 25.0


def test_the_overlay_rows_are_per_person_and_per_zone():
    w, p = _room()
    q = _person(w, 22, 22)
    q["name"] = "second"
    for _ in range(81):
        w.step()
    assert p["heat"]["slot"] == 0 and q["heat"]["slot"] == 1
    ov = w.face_env.ov
    assert w.face_env.overlay and len(ov) >= 2 * 16 + 1
    ids0 = {int(v) for v in np.unique(w.bzone[w.bzone > 0]) if v <= 16}
    ids1 = {int(v) for v in np.unique(w.bzone[w.bzone > 0]) if v > 16}
    assert ids0 and ids1 and not ids0 & ids1
    from src.voxel import faceenv as FE
    assert (ov[1:15, FE.OV_HNAT] > 0).all() and (ov[17:31, FE.OV_HNAT] > 0).all()
    assert ov[0, FE.OV_HNAT] == 0.0                                      # the default row is untouched


def test_skin_emits_at_098_in_a_bodyworld_and_095_in_a_world():
    from src.voxel.materials import EMISS, _EMISS_ARR
    w, _ = _room()
    assert w._emiss_table()[FLESH] == pytest.approx(0.98)
    assert _EMISS_ARR[FLESH] == pytest.approx(EMISS[FLESH]) and EMISS[FLESH] == pytest.approx(0.95)


def test_the_overlay_ids_follow_a_body_that_is_moved():
    w, p = _room()
    for _ in range(81):
        w.step()
    h = p["heat"]
    before = h["cells"].copy()
    comp, sl = w._person_cells(p)
    own = w._claim_cells(p, comp, sl)
    shift = own + np.array([3, 0, 0])
    w._put(tuple(shift.T), w._lift(tuple(own.T)))
    w._claim_bodies()
    w._bh_follow(p, h)
    assert len(h["cells"]) == len(before)
    assert np.array_equal(np.sort(h["cells"][:, 0]), np.sort(before[:, 0] + 3))


# ======================================================= the walking leaks ====
def test_14b_a_warm_walking_body_conserves_energy():
    """T7 test 14b: the world's energy drifts under 1 J per 1000 ticks with a warm body that walks and turns."""
    n = 160
    w = BodyWorld(n, 20, 40, voxel_cm=5)
    w.fill(0, n, 0, 20, 0, 1, STONE)
    w.exits = [(n - 12, 10)]
    p = _person(w, 12, 10)
    p["name"] = "walker"
    p["known"][:] = True
    w.policy = _Wants(each={"walker": {"legs": "go(the door"}})
    w.step()
    e0 = w.total_energy()
    x0 = np.argwhere(w.mat == FLESH)[:, 0].mean()
    for _ in range(140):                      # 4 m of floor; the walker is gone through the door by tick 170
        w.step()
        p["events"].clear()
    moved = (np.argwhere(w.mat == FLESH)[:, 0].mean() - x0) * 0.05
    assert moved > 3.0, "the walker did not walk (%.2f m)" % moved
    # (float32 E leaked 0.075 J a tick here: 10 J in 140 ticks; the bound is 1 J per 1000 ticks)
    assert abs(w.total_energy() - e0) < 0.14, w.total_energy() - e0


def test_a_turn_on_the_spot_keeps_the_air_it_turns_into():
    """The _turn leak (-1.21 J a turn for a 34 C body): the air a body turns into was written over."""
    w, p = _room()
    for _ in range(81):
        w.step()
    # warm the air around the body so that a lost cell would show
    comp, sl = w._person_cells(p)
    own = w._claim_cells(p, comp, sl)
    box = (slice(own[:, 0].min() - 2, own[:, 0].max() + 3), slice(own[:, 1].min() - 2, own[:, 1].max() + 3),
           slice(1, own[:, 2].max() + 2))
    air = w.mat[box] == AIR
    C = w.heat_capacity()[box]
    w.E[box][air] = 3.0 * C[air]
    w.shed -= float((3.0 * C[air]).sum())          # (the injection is booked)
    e0 = float(w.E.sum()) + w.shed
    assert w._turn(p, 1)
    assert abs(float(w.E.sum()) + w.shed - e0) < 1e-6
    for _ in range(3):
        assert w._turn(p, 1)
    assert abs(float(w.E.sum()) + w.shed - e0) < 1e-6


# ============================================================ gas-only air ====
def test_only_gas_pushes_a_hot_stone_is_no_wind():
    w = World(24, 24, 24, voxel_cm=5)
    w.fill(0, 24, 0, 24, 0, 1, STONE)
    w.fill(8, 12, 8, 12, 1, 5, STONE)
    C = w.heat_capacity()
    s = (w.mat == STONE)
    w.E[s] = 200.0 * C[s]                                    # a 220 C block, no hot gas
    assert not w._warm_cells().any()
    assert not w._gas_busy()
    w.E[14, 14, 5] = 30.0 * C[14, 14, 5]                     # one warm gas voxel
    assert w._gas_busy() and w._warm_cells().any()


def test_the_cheap_gate_is_exact_the_air_law_changes_nothing_when_it_says_no():
    """_pushed() false  =>  running _law_air anyway changes nothing (hot solids, no hot gas, fields at rest)."""
    w = World(24, 24, 24, voxel_cm=5)
    w.fill(0, 24, 0, 24, 0, 1, STONE)
    w.fill(8, 12, 8, 12, 1, 5, STONE)
    C = w.heat_capacity()
    s = (w.mat == STONE)
    w.E[s] = 200.0 * C[s]
    w.o2 = np.full(w.shape, 0.2, np.float32)
    w.smoke[10, 10, 8] = 0.5                                 # some smoke to carry, if the law moved anything
    assert not w._pushed()                                   # (no step: conduction would warm the gas next to the stone)
    before = [w.smoke.copy(), w.o2.copy()]
    w._law_air()
    assert not w.pcell.any() and not any(v.any() for v in w.vcell)
    for a, b in zip(before, [w.smoke, w.o2]):
        assert np.array_equal(a, b)


# ===================================================================== N2 (a) =
def test_n2a_an_awake_body_rests_at_the_iso_level_asleep_at_basal():
    m, hgt = 73.0, 1.75
    bsa = TI.bsa_dubois(m, hgt)
    p_tissue = 82.0
    f = EN.rest_floor_w(p_tissue, m, hgt, awake=True)
    assert p_tissue + f == pytest.approx(58.15 * bsa, rel=1e-9)
    assert 105.0 < p_tissue + f < 112.0
    assert EN.rest_floor_w(p_tissue, m, hgt, awake=False) == 0.0
    mb = EN.met_b_of(p_tissue, m)
    # 1.22 W/kg stays the MET unit: at 3 MET the old law is larger and wins
    assert EN.p_gait(3.0, mb, m, f) == pytest.approx(EN.p_gait(3.0, mb, m))
    assert EN.p_gait(1.0, mb, m, f) == pytest.approx(f)
    assert EN.p_gait(1.0, mb, m) < f                        # (the old law gave 9 W here)


# =============================================== thermoreg is two-sided now ===
def _sig(T_core, skin_shift):
    T_sk = {z: TR.SET_SKIN[z] + skin_shift for z in TR.ZONES}
    return TR.signals(T_core + TR.DT_BRAIN, T_sk, TR.SET_CORE)


def test_a_cold_core_takes_sweat_away_from_a_warm_skin():
    prm = TR.person_params(1, "a")
    warm_skin_cold_core = TR.targets(_sig(36.6, +1.0), prm, 73.0, 36.6)
    warm_skin_set_core = TR.targets(_sig(TR.SET_CORE - TR.DT_BRAIN, +1.0), prm, 73.0, 36.8)
    assert warm_skin_cold_core.sweat_total() == 0.0
    assert warm_skin_set_core.sweat_total() > 0.0
    hot = TR.targets(_sig(37.6, +1.0), prm, 73.0, 37.6)
    assert hot.sweat_total() > warm_skin_set_core.sweat_total()          # still monotone in the core error


def test_a_cold_core_shuts_the_skin_vessels_with_warm_skin():
    prm = TR.person_params(2, "b")
    neutral = TR.targets(_sig(TR.SET_CORE - TR.DT_BRAIN, +0.5), prm, 73.0, 36.8)
    cold_core = TR.targets(_sig(36.5, +0.5), prm, 73.0, 36.5)
    for z in TR.ZONES:
        assert cold_core.vessel[z] < neutral.vessel[z] + 1e-12
    assert sum(cold_core.vessel.values()) < 0.75 * sum(neutral.vessel.values())


def test_the_neutral_zone_core_does_not_dip_at_31_C():
    """The JOS-3 finding (marker case zone_31C): the compact core sat 0.36 K under JOS-3 at 31 C; two-sided it is within 0.1 K."""
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    import test_bodyheat_scenes as T
    ours = T._compact_on_jos3_case("zone_31C")
    jos = [c for c in T.JOS3_CASES if c["id"] == "zone_31C"][0]["result"]
    assert abs(ours["core"] - jos["core"]) < 0.1, (ours["core"], jos["core"])


# ============================================================= D2, the sleep law
def _pp(**kw):
    p = {"pid": 3, "name": "sleeper", "alive": True, "awake": True}
    p.update(kw)
    return p


@pytest.fixture
def row():
    RF.REFLEX["_test_start"] = {**RF.REFLEX_DEFAULTS, "band": (0.1, 0.2), "yields_to": (), "threshold": 10.0}
    yield "_test_start"
    del RF.REFLEX["_test_start"]


def test_sleep_raises_the_threshold_and_never_switches_the_reflex_off(row):
    w = World(4, 4, 4)
    awake, asleep, out = _pp(), _pp(asleep=True), _pp(awake=False)
    t = [RF.threshold_now(p, row) for p in (awake, asleep, out)]
    assert t[0] < t[1] < t[2] < float("inf")                 # deep unconsciousness: high, not infinite
    assert t[2] / t[0] == pytest.approx((1 + RF.DEPTH_GAIN * RF.OUT_DEPTH) / (1 + 0.0), rel=1e-9)
    for p in (awake, asleep, out):
        assert w._reflex_off(p, row) is None                 # not off, any of them


def test_the_dead_are_off_in_every_row(row):
    w = World(4, 4, 4)
    dead = _pp(alive=False)
    assert RF.threshold_now(dead, row) is None and RF.fire_chance(dead, row, 1e9) == 0.0
    for kind in (row, "shiver", "sweat", "vessel", "balance"):
        assert w._reflex_off(dead, kind) == "out"


def test_the_chance_rises_with_the_stimulus_and_falls_with_depth(row):
    awake, asleep, out = _pp(), _pp(asleep=True), _pp(awake=False)
    for p in (awake, asleep, out):
        c = [RF.fire_chance(p, row, s) for s in (0.0, 2.0, 5.0, 10.0, 20.0, 60.0, 1e4)]
        assert c[0] == 0.0 and all(x <= y + 1e-12 for x, y in zip(c, c[1:])) and c[-1] > 0.999
    s = 20.0
    assert RF.fire_chance(awake, row, s) > RF.fire_chance(asleep, row, s) > RF.fire_chance(out, row, s) > 0.0


def test_the_per_person_dial_is_the_persons_own_and_changes_the_threshold(row):
    a, b = _pp(pid=1, name="a"), _pp(pid=2, name="b")
    assert RF.threshold_now(a, row) != RF.threshold_now(b, row)
    assert RF.threshold_now(a, row) == RF.threshold_now(dict(a), row)    # the same every run
    c = _pp(**{row + "_dial": 0.5})
    assert RF.threshold_now(c, row) == pytest.approx(5.0)                 # sheet value wins, awake depth 0


def test_the_draw_is_a_hash_and_is_the_same_every_run(row):
    w = World(4, 4, 4)
    p = _pp(asleep=True)
    runs = []
    for _ in range(2):
        w.tick = 0
        hits = []
        for t in range(2000):
            w.tick = t
            hits.append(w.reflex_fires(p, row, 14.0))
        runs.append(hits)
    assert runs[0] == runs[1]
    n = sum(runs[0])
    c = RF.fire_chance(p, row, 14.0)
    assert 0 < n < 2000
    # 2000 ticks = 50 s; expected firings (per tick p) = 2000 * (1 - (1 - c)^(0.025)), within a loose band
    exp = 2000 * (1 - (1 - c) ** 0.025)
    assert abs(n - exp) < 6 * np.sqrt(max(exp, 1.0))


def test_the_thermoregulation_rows_work_asleep_and_unconscious_and_the_dead_do_not_shiver():
    w = World(4, 4, 4)
    for kind in ("shiver", "sweat", "vessel"):
        assert RF.REFLEX[kind]["kind"] == "continuous" and RF.REFLEX[kind]["works_when"] == "alive"
        assert w._reflex_off(_pp(asleep=True), kind) is None
        assert w._reflex_off(_pp(awake=False), kind) is None
        assert w._reflex_off(_pp(alive=False), kind) == "out"
    assert w._reflex_off(_pp(asleep=True), "balance") == "out"           # no stimulus scale: the wave 12 rule


def test_fatigue_is_read_per_limb():
    p = {"fatigue": 0.2, "limb_fatigue": {"left arm": 0.8}}
    assert RF.fatigue_of(p, "left arm") == 0.8
    assert RF.fatigue_of(p, "right arm") == 0.2 and RF.fatigue_of(p) == 0.2


def test_the_reflex_events_of_a_shivering_body_are_typed_events():
    w, p = _room()
    w.step()
    h = p["heat"]
    h["flags"]["shiver"] = False
    w.bh_emit(p, "shivers", "on")
    assert any("shiver" in str(e) for e in p["events"])


# ===================================================== lockstep: the windows ====
@pytest.mark.slow
def test_a_world_with_a_record_is_the_same_with_the_quiet_gates_on_and_off_for_120_ticks():
    a, pa = _room()
    b, pb = _room()
    b.skip_quiet = False
    for _ in range(120):
        a.step()
        b.step()
        for name in ("mat", "smass", "fvol"):
            assert np.array_equal(getattr(a, name), getattr(b, name)), (a.tick, name)
        assert np.allclose(a.E, b.E, rtol=0, atol=1e-6), (a.tick, "E", float(np.abs(a.E - b.E).max()))


# ================================================== the body's life and work ====
def _fit(p, cold=1.0, hot=1.0):
    p["heat"]["fit"] = {"cold": cold, "hot": hot}


def test_a_cold_core_faints_a_body_and_a_warm_one_wakes_it_the_cold_is_named():
    w, p = _room()
    for _ in range(81):
        w.step()
    assert p["awake"]
    for _ in range(3):
        _fit(p, cold=0.3)                                   # a core near 30 C (the fit is rewritten at each body second)
        w.step()
    assert not p["awake"] and p["alive"]
    assert any("the cold" in str(e) for e in p["events"]), p["events"]
    for _ in range(3):
        _fit(p, cold=1.0)
        w.step()
    assert p["awake"]                                       # round again once the share is over 0.70


def test_a_core_at_the_end_of_the_band_is_death_and_it_names_cold_or_heat():
    for side, kw in (("cold", {"cold": 0.0}), ("heat", {"hot": 0.0})):
        w, p = _room()
        for _ in range(81):
            w.step()
        for _ in range(2):
            _fit(p, **kw)
            w.step()
        assert not p["alive"], side


def test_a_body_with_no_record_ignores_the_core_bands():
    w, p = _room(World)
    for _ in range(5):
        w.step()
    assert p["alive"] and p["awake"]


def test_a_dragging_body_pays_for_the_work_from_its_stores_and_the_books_close():
    w, p = _room()
    q = _person(w, 12, 20)
    q["name"] = "dragged"
    for _ in range(81):
        w.step()
    p["dragging"] = q["name"]
    comp, sl = w._person_cells(p)
    own = w._claim_cells(p, comp, sl)
    from src.voxel import digestion as DG
    st = p["heat"]["stores"]

    def book():
        t = w.bh_terms()
        return (float(w.E.sum()) + w.shed + t["pool_J"] + t["stores_J"] + t["pending_J"] + t["work_out"])

    s0, b0 = DG.stores_J(st), book()
    w._rub(p, own)
    s1 = DG.stores_J(st)
    assert s0 - s1 > 0.0                                       # the stores paid
    assert abs(book() - b0) < 1e-6                             # and nothing was made or lost
