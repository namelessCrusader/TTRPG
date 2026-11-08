"""T7b: the compact body (bodynodes.py) and its hand-over with the lattice
(spec T7 section 6 tests 11, 18, 19; plan critic 4).

The voxel side is the SLAB FIXTURE (src/voxel/scenes_bodyheat.py: a small
float64-parametrised lattice that implements the hook arguments BH1-BH3
itself), never the unmodified World face law. Tags: test_u_* = physical data
not yet sourced; test_codeonly_* = the code against its own table; untagged =
conservation, exactness or arithmetic identity.
"""
import copy

import numpy as np
import pytest

from src.voxel import bodynodes as BN
from src.voxel import digestion as DG
from src.voxel import scenes_bodyheat as SB
from src.voxel import tissue as TI
from src.voxel.airsink import SinkLog

DTYPES = (np.float64, np.float32)


def _std():
    sh = TI.sheet("male", mass_kg=73.0, height_m=1.75)
    g = BN.geom_from_sheet(sh)
    h = BN.new_heat_record(sh, 1, "std")
    SB.mid_params(h)
    return sh, g, h, BN.nodes_from_T(g, h, BN.comfort_T(g, h))


# ------------------------------------------------------------ the layout ---
def test_there_are_43_nodes_three_per_zone_and_a_pool():
    assert BN.N_NODES == 43 == 3 * 14 + 1
    assert len(BN.NODE_NAMES) == 43 and BN.NODE_NAMES[-1] == "pool"
    assert BN.nidx(TI.ZONE_INDEX["torso"], 2) == 3 * 1 + 2


def test_a_zone_too_thin_for_a_core_has_a_massless_core_at_its_shell_temperature():
    sh, g, h, nodes = _std()
    z = TI.ZONE_INDEX["left hand"]
    assert nodes["C"][BN.nidx(z, 2)] == 0.0
    BN.advance(h, nodes, g, {"T_air": 25.0, "rh": 0.5, "v": 0.0, "T_rad": 25.0}, 300.0)
    T = BN.temps(nodes)
    assert T[BN.nidx(z, 2)] == T[BN.nidx(z, 1)]
    assert nodes["E"][BN.nidx(z, 2)] == 0.0
    assert np.isfinite(T).all()


# ---------------------------------------------------- books of the skip -----
@pytest.mark.parametrize("T_air,rh,met", [(28.0, 0.5, 1.0), (38.0, 0.4, 3.0), (12.0, 0.6, 1.0)])
def test_the_skip_closes_energy_to_the_joule(T_air, rh, met):
    """node energy change + heat given to the room = metabolic heat + TEF."""
    sh, g, h, nodes = _std()
    DG.ingest(h["stores"], DG.food("stew", 300.0))
    e0 = BN.node_energy(nodes)
    out = BN.advance(h, nodes, g, {"T_air": T_air, "rh": rh, "v": 0.1, "T_rad": T_air}, 3600.0, met=met)
    d = BN.node_energy(nodes) - e0
    made = out["metabolic_J"] + out["tef_J"]
    assert d + out["heat_J"] - made == pytest.approx(0.0, abs=1e-3)
    # and the stores paid exactly the heat made, urine energy staying in the books
    assert out["stores_J"] >= out["metabolic_J"] - 1e-6


def test_the_skip_closes_water_and_gas_mass():
    sh, g, h, nodes = _std()
    m0 = DG.stores_g(h["stores"]) + nodes["film_g"].sum()
    out = BN.advance(h, nodes, g, {"T_air": 40.0, "rh": 0.3, "v": 0.1, "T_rad": 40.0}, 3600.0, met=2.0)
    m1 = DG.stores_g(h["stores"]) + nodes["film_g"].sum()
    o2_g, co2_g = out["o2_l"] * DG.O2_G_L, out["co2_l"] * DG.CO2_G_L
    assert out["sweat_g"] > 0.0
    assert m1 - m0 + out["vapour_g"] + out["drip_g"] == pytest.approx(o2_g - co2_g, abs=1e-6)


def test_the_skip_is_the_same_for_one_big_call_and_many_small_ones():
    sh, g, h, nodes = _std()
    sh2, g2, h2, nodes2 = _std()
    env = {"T_air": 25.0, "rh": 0.5, "v": 0.1, "T_rad": 25.0}
    BN.advance(h, nodes, g, env, 1800.0)
    for _ in range(6):
        BN.advance(h2, nodes2, g2, env, 300.0)
    assert np.allclose(BN.temps(nodes), BN.temps(nodes2), atol=1e-9)


def test_codeonly_stores_and_breath_follow_one_source_of_power():
    """O2 drawn from the air = O2 of the fuel burned (Weir within 2 percent) and
    the CO2 made has the respiratory quotient of the mix."""
    sh, g, h, nodes = _std()
    out = BN.advance(h, nodes, g, {"T_air": 29.0, "rh": 0.5, "v": 0.1, "T_rad": 29.0}, 1800.0)
    rq = out["co2_l"] / out["o2_l"]
    assert 0.7 < rq < 1.0
    assert out["metabolic_J"] == pytest.approx(out["o2_l"] * DG.weir_kj_per_l(rq) * 1000.0, rel=0.03)


def test_the_dead_do_not_shiver_sweat_or_breathe_and_energy_is_kept():
    sh, g, h, nodes = _std()
    e0 = BN.node_energy(nodes)
    BN.kill_nodes(h, nodes, g)
    assert BN.node_energy(nodes) == pytest.approx(e0, abs=1e-6)
    out = BN.advance(h, nodes, g, {"T_air": 40.0, "rh": 0.3, "v": 0.0, "T_rad": 40.0}, 1800.0)
    assert out["o2_l"] == 0.0 and out["sweat_g"] == 0.0 and out["breath_water_g"] == 0.0
    assert BN.temps(nodes)[BN.POOL] == pytest.approx(20.0)


# ------------------------------------------------------------ hand-over ----
@pytest.mark.parametrize("dtype", DTYPES)
def test_hand_over_node_energies_are_exact_and_the_round_trip_is_identical(dtype):
    w, p = SB.person_world(E_dtype=dtype, T_air=22.0)
    SB.run(w, 120)
    h = p["heat"]
    g = w._bh_geom(p, h)
    idx = tuple(h["cells"].T)
    E = np.asarray(w.E[idx], dtype=np.float64)
    nodes = w.bh_to_nodes(p)
    body = np.delete(nodes["E"], BN.POOL)
    assert body.sum() == pytest.approx(E.sum(), rel=1e-13, abs=1e-9)         # float64 exact
    E2, s2 = BN.from_nodes(g, h, nodes)
    assert E2.sum() == pytest.approx(body.sum(), rel=1e-13, abs=1e-9)
    n2 = BN.to_nodes(g, h, E2, s2)
    assert np.allclose(n2["E"], nodes["E"], rtol=0, atol=1e-6)
    # into the lattice dtype: the voxel sum is exact to one spacing of that dtype
    r = BN.exact_round(E2, dtype)
    sp = float(np.spacing(np.abs(E2).max().astype(dtype)))
    assert abs(float(r.astype(np.float64).sum()) - E2.sum()) <= sp
    if dtype == np.float64:
        assert np.array_equal(r, E2)


def test_exact_round_beats_plain_rounding():
    rng = np.random.default_rng(2)
    v = rng.uniform(4000.0, 8000.0, 400)
    plain = abs(v.astype(np.float32).astype(np.float64).sum() - v.sum())
    exact = abs(BN.exact_round(v, np.float32).astype(np.float64).sum() - v.sum())
    assert exact <= np.spacing(np.float32(8000.0))
    assert exact <= plain


@pytest.mark.parametrize("dtype", DTYPES)
def test_hand_over_state_drift_over_60_s_is_within_the_tolerance(dtype):
    """voxel -> nodes -> voxel -> 60 s of ticks agrees with 60 s of ticks
    without the hand-over: core 0.05 K, mean skin 0.2 K."""
    def make():
        w, p = SB.person_world(E_dtype=dtype, T_air=24.0)
        SB.run(w, 240)
        return w, p
    wa, pa = make()
    wb, pb = make()
    nodes = wb.bh_to_nodes(pb)
    wb.bh_from_nodes(pb, nodes)
    SB.run(wa, 60)
    SB.run(wb, 60)
    ta, tb = wa.body_temps(pa), wb.body_temps(pb)
    assert abs(ta["core"] - tb["core"]) < 0.05
    assert abs(ta["skin_mean"] - tb["skin_mean"]) < 0.2


# ---------------------------------------------------------- sink parity ----
def _sleeper(T_air, rh=0.5):
    w, p = SB.person_world(T_air=T_air, rh=rh)
    room = SinkLog(t_c=T_air)
    w.airsink = room
    w.face_sink = room
    p["heat"]["asleep"] = True
    p["heat"]["met"] = 0.95
    SB.run(w, 300)
    return w, p, room


def test_sink_parity_a_sleeper_ticked_30_minutes_equals_skipped_30_minutes():
    """A sealed small room test double (a SinkLog that every call of the body
    goes to, the face flux of the lattice included): O2, CO2, vapour and the
    heat given to the air agree within 3 percent."""
    wa, pa, ra = _sleeper(30.0)
    wb, pb, rb = _sleeper(30.0)
    a0, b0 = copy.deepcopy(ra.totals()), copy.deepcopy(rb.totals())
    SB.run(wa, 1800)
    wb.bh_skip(pb, 1800)
    ta, tb = ra.totals(), rb.totals()
    for k in ("vapour_g", "heat_J", "o2_l", "co2_l"):
        x, y = ta[k] - a0[k], tb[k] - b0[k]
        assert abs(x - y) <= 0.03 * abs(x), (k, x, y)


def test_sink_parity_the_water_is_conserved_to_1e_6_in_a_sealed_room():
    """Every gram a sleeper loses is in the room, on the skin, or dripped."""
    for mode in ("ticks", "skip"):
        w, p, room = _sleeper(32.0, 0.5)
        h = p["heat"]
        m0 = DG.stores_g(h["stores"]) + float(np.sum(h["film_g"])) + h["stores"]["leak_ml"]
        v0 = room.gone.get("vapour", 0.0)
        d0 = h.get("drip_g", 0.0)                                   # what ran off before this window
        o0, c0 = room.gone.get("o2_l", 0.0), room.gone.get("co2_l", 0.0)
        if mode == "ticks":
            SB.run(w, 1800)
        else:
            w.bh_skip(p, 1800)
        m1 = DG.stores_g(h["stores"]) + float(np.sum(h["film_g"])) + h["stores"]["leak_ml"]    # urine that overflowed is still ours
        dv = room.gone.get("vapour", 0.0) - v0
        do2 = (room.gone.get("o2_l", 0.0) - o0) * DG.O2_G_L          # negative: taken from the air
        dco2 = (room.gone.get("co2_l", 0.0) - c0) * DG.CO2_G_L
        assert (m1 - m0) + dv + (h.get("drip_g", 0.0) - d0) + do2 + dco2 == pytest.approx(0.0, abs=1e-6), mode


def test_the_skip_takes_the_room_heat_the_ticks_would():
    """The same sleeper at 22 C in a thin blanket: the two paths still agree."""
    from src.voxel import clothing as CL
    cov = {z: 1.0 for z in TI.ZONES if not z.endswith(("hand", "foot")) and z != "head"}
    g = CL.garment("blanket", "wool", CL.thickness_for_clo(2.0), cov, {z: 1.0 for z in cov})
    outs = []
    for mode in ("ticks", "skip"):
        w, p = SB.person_world(T_air=22.0, clothes=[g])
        room = SinkLog(t_c=22.0)
        w.airsink = w.face_sink = room
        p["heat"]["asleep"] = True
        SB.run(w, 300)
        h0 = room.totals()["heat_J"]
        if mode == "ticks":
            SB.run(w, 1200)
        else:
            w.bh_skip(p, 1200)
        outs.append(room.totals()["heat_J"] - h0)
    assert abs(outs[0] - outs[1]) <= 0.05 * abs(outs[0])


# ------------------------------------------------------------ the lattice --
def test_a_z_is_reported_per_zone_inside_0_4_3_or_the_zone_is_marked_under_resolved():
    w, p = SB.person_world()
    h = p["heat"]
    assert len(h["a_z"]) == 14 and len(h["under"]) == 14
    for z, (a, under) in enumerate(zip(h["a_z"], h["under"])):
        assert under == (not (0.4 <= a <= 3.0)), TI.ZONES[z]
        assert a > 0.0
    # the stock figure at 5 cm: every zone resolves
    assert not any(h["under"])


def test_fused_equals_plain_bits_for_stub_hook_arguments():
    """The face law with the hook tables set to the OLD constants gives the
    bits of the face law with no tables."""
    def make(stub):
        w = SB.SlabWorld()
        cells, fill, zone, _ = SB.stock_figure()
        w.add_body(cells, fill, zone)
        b = w._body
        idx = b["idx"]
        rng = np.random.default_rng(5)
        w.E[idx] = 7000.0 + rng.uniform(-3000.0, 3000.0, b["n"])
        if stub:
            n = 14
            w.bh_set_hooks(0, cells, zone, {"az": np.ones(n), "emiss": np.full(n, SB.EMISS_OLD),
                                            "conv_nat": np.zeros(n), "R_cl": np.zeros(n),
                                            "f_cl": np.ones(n),
                                            "gsb": np.full(n, SB.K_FLESH / SB.SKIN_PATH)})
        return w
    a, b = make(False), make(True)
    for _ in range(100):
        a.step()
        b.step()
    assert np.array_equal(a.E, b.E) and np.array_equal(a.skin, b.skin) and a.shed == b.shed


def test_plain_face_law_matches_a_hand_calculation():
    # 1 m2 of skin at 34 C in 20 C air, old constants: 10 W/m2K + 0.95 sigma (Ts4 - 293.15^4)
    q = SB.SlabWorld.face_loss(np.array([34.0]), np.array([1.0]), np.array([1.0]), np.array([0.95]),
                               np.array([0.0]), np.array([0.0]), np.array([1.0]), 20.0)
    sig = 5.670374e-8
    want = 10.0 * 14.0 + 0.95 * sig * (307.15 ** 4 - 293.15 ** 4)
    assert q[0] == pytest.approx(want, rel=1e-12)


def test_the_series_clothing_hook_conducts_less_than_bare_skin():
    kw = dict(Ts=np.array([33.0]), A=np.array([1.0]), az=np.array([1.0]), emiss=np.array([0.98]),
              conv_nat=np.array([2.38]), f_cl=np.array([1.15]), T_gas=20.0)
    bare = SB.SlabWorld.face_loss(R_cl=np.array([0.0]), **kw)[0]
    dressed = SB.SlabWorld.face_loss(R_cl=np.array([0.155]), **kw)[0]
    assert 0.0 < dressed < 0.5 * bare


def test_snapshot_restore_reproduces_the_next_second():
    w, p = SB.person_world(T_air=26.0)
    SB.run(w, 90)
    snap = w.snapshot()
    SB.run(w, 1)
    after = (w.E.copy(), w.skin.copy(), w.persons[0]["heat"]["T_pool"], w.shed)
    w.restore(snap)
    SB.run(w, 1)
    again = (w.E, w.skin, w.persons[0]["heat"]["T_pool"], w.shed)
    assert np.array_equal(after[0], again[0]) and np.array_equal(after[1], again[1])
    assert after[2] == again[2] and after[3] == again[3]


def test_determinism_same_seed_same_numbers():
    def go():
        w, p = SB.person_world(T_air=26.0)
        SB.run(w, 120)
        return w.E.copy(), p["heat"]["T_pool"]
    a, b = go(), go()
    assert np.array_equal(a[0], b[0]) and a[1] == b[1]


def test_codeonly_the_view_factors_come_from_the_rays():
    w, p = SB.person_world()
    h = p["heat"]
    g = w._bh_geom(p, h)
    w._bh_fself(h, g, w._bh_cache()[0])
    f = g["f_self"]
    assert (f >= 0).all() and (f <= 1).all()
    # the inner thigh sees the other thigh; the head sees almost nothing of the body
    assert f[TI.ZONE_INDEX["head"]] < f[TI.ZONE_INDEX["torso"]] + 0.4
    assert f.max() > 0.05


# =========================================== the T7b fix: what the fit may not touch ===
def test_the_physical_transfer_coefficients_are_the_iso_ones_not_fitted():
    """An earlier fit let HC_FLESH and RAD_VIEW fall to 0.41 and 0.40 to make the human cases fit; that hid
    a wrong heat balance (a nude body at -30 C stayed warm). They are 1.0 and are not fit constants."""
    assert BN.HC_FLESH == 1.0 and BN.RAD_VIEW == 1.0


def test_u_a_nude_body_in_the_deep_cold_loses_heat_at_the_iso_rate_and_its_skin_falls():
    """Independent of any fit: at -30 C nude, 20 min, the skin falls well below 20 C and the loss at the start is
    about (h_c + h_r) A dT = 8-11 W/m2K x 1.7 m2 x 60 K = 0.8-1.2 kW."""
    sh, g, h, nodes = _std()
    env = {"T_air": -30.0, "rh": 0.5, "v": 0.1, "T_rad": -30.0, "zones": {}}
    first = BN.advance(h, nodes, g, env, 60.0, sub=10.0)
    assert 700.0 <= first["sensible_J"] / 60.0 <= 1400.0, first["sensible_J"] / 60.0
    for _ in range(19):
        BN.advance(h, nodes, g, env, 60.0, sub=10.0)
    assert BN.mean_skin(nodes, g) < 20.0, BN.mean_skin(nodes, g)


def test_the_exposed_share_of_a_zone_scales_what_its_skin_gives_the_air():
    Ts = 32.0
    e = {"T_air": 20.0, "rh": 0.5, "v": 0.1, "T_rad": 20.0, "R_cl": 0.0, "f_cl": 1.0, "T_mrt": None, "h_fixed": None}
    full = BN.skin_loss(Ts, e, 0.5, 0.1, Ts, 0.0, 0.0, 10.0)[0]
    half = BN.skin_loss(Ts, e, 0.5 * 0.5, 0.1, Ts, 0.0, 0.0, 10.0)[0]
    assert half == pytest.approx(0.5 * full, rel=1e-12)
    assert BN.zone_env({"T_air": 20.0}, 0, 33.0)["exposed"] == 1.0
    assert BN.zone_env({"zones": {"torso": {"exposed": 0.55}}}, TI.ZONE_INDEX["torso"], 33.0)["exposed"] == 0.55


def test_shell_flow_follows_the_vessel_reflex_and_is_one_at_rest():
    f = BN.shell_flow_factor
    assert f(1.0) == pytest.approx(1.0, abs=1e-12)
    assert f(0.2) < 0.5 and f(0.1) < f(0.2)                       # cold closes it
    assert f(2.0) > 1.0 and f(60.0) == pytest.approx(f(BN.SHELL_VASO_CAP))   # heat opens it, capped
    v = np.array([0.1, 1.0, 5.0])
    assert np.array_equal(f(v), np.array([f(0.1), f(1.0), f(5.0)]))


@pytest.mark.parametrize("sweat,T_air,rh", [(0.5, 40.0, 0.3), (0.05, 30.0, 0.5), (3.0, 40.0, 0.9)])
def test_sweat_run_off_closes_the_water_book_and_never_exceeds_half(sweat, T_air, rh):
    """Gagge's 1 - w^2/2: the part of the sweat that runs off goes to drip_g; every gram is placed."""
    e = {"T_air": T_air, "rh": rh, "v": 0.1, "T_rad": T_air, "R_cl": 0.0, "f_cl": 1.0, "T_mrt": None, "h_fixed": None}
    film0, area, dt = 2.0, 0.5, 30.0
    ev = BN.evap_zone(35.0, e, area, 3.0, sweat, film0, dt)
    placed = ev["evap_g"] + ev["film_g"] + ev["drip_g"]
    assert placed == pytest.approx(sweat + film0, rel=1e-9, abs=1e-9)
    low = BN.evap_zone(35.0, e, area, 3.0, sweat * 1e-3, 0.0, dt)
    assert low["drip_g"] <= 1e-6 * sweat + 1e-12                    # a trickle all evaporates
    assert ev["drip_g"] <= 0.5 * sweat + 1e-12              # the film (2 g) is far below its cap (15 g)


def test_u_the_sweating_efficiency_falls_as_the_skin_gets_wetter():
    e = {"T_air": 40.0, "rh": 0.3, "v": 0.1, "T_rad": 40.0, "R_cl": 0.0, "f_cl": 1.0, "T_mrt": None, "h_fixed": None}
    area, dt, hc = 0.5, 30.0, 3.0
    emax_w = float(BN.BA.evap_max(36.0, 40.0, 0.3, hc, I_cl=0.0, area_m2=area))
    L = float(BN.BA.latent_heat(36.0))
    effs = []
    for w in (0.1, 0.5, 0.95):
        sweat = w * emax_w * dt / (L / 1000.0)
        ev = BN.evap_zone(36.0, e, area, hc, sweat, 0.0, dt)
        effs.append(1.0 - ev["drip_g"] / sweat)
    assert effs[0] > effs[1] > effs[2] > 0.5
    assert effs[0] == pytest.approx(1.0 - 0.1 ** 2 / 2.0, abs=1e-6)


def test_work_heat_is_shared_between_shell_and_core_by_capacity_like_the_voxel_law():
    sh, g, h, nodes = _std()
    for zn, watts in (("torso", 37.5), ("left thigh", 12.25), ("left hand", 3.0)):
        z = TI.ZONE_INDEX[zn]
        Q = np.zeros(BN.N_NODES)
        BN._deposit(Q, g, z, watts)
        assert Q.sum() == pytest.approx(watts, rel=1e-14)
        cs, cc = g["C_shell"][z], g["C_core"][z]
        assert Q[BN.nidx(z, 1)] == pytest.approx(watts * cs / (cs + cc), rel=1e-12)
        assert Q[BN.nidx(z, 0)] == 0.0
    hand = TI.ZONE_INDEX["left hand"]
    Q = np.zeros(BN.N_NODES)
    BN._deposit(Q, g, hand, 3.0)
    assert Q[BN.nidx(hand, 2)] == 0.0 and Q[BN.nidx(hand, 1)] == 3.0      # no core: all in the shell
