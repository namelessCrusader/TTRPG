"""T7 tissue: the table, the sheet, the zones and Pennes in a voxel (no World).

Tags (wave 13 rule): test_u_* = physical data checked against an independent
number whose source is unverified (A-S upgrades it to [I] when it marks the
number sourced); test_echo_* = the constant was fitted to that number;
test_codeonly_* = the code against its own table. Unmarked tests are
exactness, conservation, closed-form and arithmetic identities.
"""
import json
import math
import os

import numpy as np
import pytest

from src.voxel import tissue as T

SOURCED = os.path.join(os.path.dirname(__file__), "data", "sourced_numbers.json")


def _sourced_status(key):
    """'sourced'/'unsourced'/'absent': what tests/data/sourced_numbers.json says about `key`."""
    try:
        d = json.load(open(SOURCED))
    except (OSError, ValueError):
        return "absent"
    row = d.get(key) if isinstance(d, dict) else None
    if row is None:
        return "unsourced"
    return "sourced" if (row.get("sourced") if isinstance(row, dict) else row) else "unsourced"


MEN = [(65, 1.72), (73, 1.75), (85, 1.80)]
WOMEN = [(55, 1.60), (62, 1.65), (75, 1.70)]


# ── 1. the table ─────────────────────────────────────────────────────────────
def test_echo_reference_man_73kg_is_82_W():
    """The organ q_m was fitted to this number."""
    sh = T.sheet("male", 73.0, 1.75)
    P = T.basal_power_w(sh)
    assert abs(P / T.bmr_mifflin("male", 73.0, 1.75) - 1.0) < 0.10
    assert 78.0 < P < 86.0


@pytest.mark.parametrize("sex,sizes", [("male", MEN), ("female", WOMEN)])
def test_u_reference_people_within_10pct_of_mifflin(sex, sizes):
    """Reference man and woman at three sizes: sum of q_m V within 10 percent of
    Mifflin-St Jeor (Mifflin is independent of the table, but the organ row was
    fitted across these sizes, so the spread is partly an echo). Reports which
    status the sourced-numbers file gives Mifflin."""
    seen = _sourced_status("mifflin_st_jeor")
    assert seen in ("absent", "sourced", "unsourced")
    for m, h in sizes:
        sh = T.sheet(sex, m, h)
        P = T.basal_power_w(sh)
        M = T.bmr_mifflin(sex, m, h)
        assert abs(P / M - 1.0) < 0.10, (sex, m, P, M, seen)


def test_u_cardiac_sum_4p5_to_6p5_lmin():
    """Cardiac sum for adults of normal size: 4.5-6.5 L/min (flow per kg falls as mass^-0.25)."""
    for sex, sizes in (("male", MEN), ("female", WOMEN)):
        for m, h in sizes:
            co = T.cardiac_output_l_min(T.sheet(sex, m, h))
            assert 4.5 <= co <= 6.5, (sex, m, co)


def test_composition_fractions_sum_to_one():
    for sex, m, h, ff in (("male", 73, 1.75, None), ("female", 62, 1.65, None),
                          ("male", 120, 1.80, 0.40), ("female", 45, 1.55, 0.08)):
        sh = T.sheet(sex, m, h, ff)
        comp = T.composition(sh)
        tot = math.fsum(math.fsum(r.values()) for r in comp.values())
        assert abs(tot / sh.mass_kg - 1.0) < 1e-12
        assert abs(math.fsum(T.WINTER_SHARE.values()) - 1.0) < 1e-12
        assert abs(math.fsum(T.AREA_SHARE.values()) - 1.0) < 1e-12
        for z, row in comp.items():
            assert abs(math.fsum(row.values()) / (T.WINTER_SHARE[z] * sh.mass_kg) - 1.0) < 1e-12
            assert all(v >= 0.0 for v in row.values())


def test_fat_frac_moves_fat_thickness_monotonically():
    prev = None
    for ff in (0.08, 0.14, 0.20, 0.28, 0.36, 0.45):
        zp = T.zone_params(T.sheet("male", 73.0, 1.75, ff))
        row = (zp["torso"].fat_m, zp["left thigh"].fat_m, zp["torso"].R_sc, zp["torso"].fat_kg)
        if prev is not None:
            assert all(a > b for a, b in zip(row, prev)), (ff, row, prev)
        prev = row


def test_zone_tables_complete():
    assert len(T.ZONES) == 14 and len(set(T.ZONES)) == 14
    assert set(T.WINTER_SHARE) == set(T.ZONES) == set(T.AREA_SHARE)
    zp = T.zone_params(T.sheet("male"))
    assert list(zp) == list(T.ZONES)
    for z in T.ZONES:
        v = zp[z]
        assert v.mass_kg > 0 and v.volume_m3 > 0 and v.R_sc > 0 and 0 <= v.eps_cc < 1
        assert 0 < v.dermal_frac < 0.3


def test_u_winter_shares_match_the_published_segments():
    """Winter 1990 / ICRP: head+neck 8.1, trunk 49.7, thigh 10, shank 4.65, foot 1.45 percent."""
    W = T.WINTER_SHARE
    assert W["head"] == pytest.approx(0.081, abs=1e-6)
    assert W["torso"] == pytest.approx(0.497, abs=1e-6)
    assert W["left thigh"] == pytest.approx(0.100, abs=1e-6)
    assert W["right foot"] == pytest.approx(0.0145, abs=1e-6)


def test_zone_of_bone_is_the_one_source():
    assert T.ZONE_OF_BONE["torso"] == "torso" and T.ZONE_OF_BONE["head"] == "head"
    assert T.zone_of("left forearm") == "left forearm"
    assert T.zone_of("left forearm", 0.5) == "left forearm"
    assert T.zone_of("left forearm", 0.8) == "left hand"
    assert T.zone_of("right shin", 0.9) == "right foot"
    assert T.zone_of("right shin", 0.5) == "right shin"
    assert T.zone_of("right thigh", 1.0) == "right thigh"
    assert T.zone_of("tail") is None
    # every zone is the main zone of a bone or the distal zone of one
    covered = set(T.ZONE_OF_BONE.values()) | {d[0] for d in T.DISTAL_OF_BONE.values()}
    assert covered == set(T.ZONES)


def test_sheet_roundtrip_and_defaults():
    sh = T.sheet("female")
    assert sh.mass_kg in (60.0, 62.0) and 0.2 < sh.fat_frac < 0.4      # bodysheet's adult woman, or our stand-in
    d = sh.as_dict()
    assert set(d) == {"sex", "mass_kg", "height_m", "fat_frac", "age_y", "bmr_scale"}
    assert T.Sheet.from_dict(d).as_dict() == d
    assert T.pool_capacity(T.sheet("male", 73.0)) == pytest.approx(7.7e3, rel=0.05)
    # Deurenberg: a 73 kg 175 cm 30 y man is about 19 percent fat
    assert T.fat_frac_deurenberg("male", 73.0, 1.75) == pytest.approx(0.19, abs=0.02)


def test_codeonly_bodysheet_cross():
    """Winter shares and the default adults come from lane A-G10's module when it exists."""
    bs = pytest.importorskip("src.voxel.bodysheet", reason="lane A-G10 not wired yet")
    alias = {"torso": "trunk", "shin": "shank"}
    tot = math.fsum(bs.WINTER[alias.get(T._kind(z), T._kind(z))] for z in T.ZONES)
    for z in T.ZONES:
        want = bs.WINTER[alias.get(T._kind(z), T._kind(z))] / tot
        assert T.WINTER_SHARE[z] == pytest.approx(want, rel=1e-12), z
    m = T.sheet("male")
    assert (m.mass_kg, m.height_m, m.fat_frac) == (bs.ADULT_MALE["mass_kg"], bs.ADULT_MALE["height_m"], bs.ADULT_MALE["fat_frac"])
    assert T.bodysheet_sheet("adult female").mass_kg == bs.ADULT_FEMALE["mass_kg"]
    assert T.bsa_dubois(70.0, 1.75) == pytest.approx(bs.bsa_m2({"mass_kg": 70.0, "height_m": 1.75}), rel=1e-12)
    # the stock figure (43.9 kg, 1.5 m) goes through the same table
    st = T.bodysheet_sheet("stock")
    assert st.mass_kg == 43.9
    assert abs(T.basal_power_w(st) / T.bmr_mifflin("male", st.mass_kg, st.height_m) - 1.0) < 0.10
    sm = bs.segment_masses(bs.get_sheet("adult male"))
    comp = T.composition(m)
    for z in T.ZONES:
        key = z.replace("torso", "trunk").replace("shin", "shank")
        assert math.fsum(comp[z].values()) == pytest.approx(sm[key], rel=1e-12), z


# ── 2. Pennes in a voxel ─────────────────────────────────────────────────────
def _body(n=400, seed=1, dead=50):
    rng = np.random.default_rng(seed)
    C = rng.uniform(300, 500, n)
    g = rng.uniform(0.0, 0.4, n)
    g[:dead] = 0.0
    T0 = rng.uniform(33, 37, n)
    Q = rng.uniform(0, 0.4, n)
    return T0, C, g, Q


def _fine(T0, C, g, Q, dt, N, Tp, Cp, Qo, eps=None):
    ge = g if eps is None else g * (1.0 - eps)
    h = dt / N
    Tv, tp = T0.copy(), Tp
    for _ in range(N):
        flow = ge * (tp - Tv)
        Tv = Tv + h * (flow + Q) / C
        tp = tp + h * (-flow.sum() - Qo) / Cp
    return Tv, tp


def test_pennes_voxel_alone_is_exact_vs_fine_loop():
    """A voxel on a fixed pool: the phi form is the exact exponential; a fine
    explicit loop (dt/1000) agrees to 1e-6 when x = g dt/C is small."""
    T0, C, g, Q = _body()
    g = g * 1e-3                               # x ~ 1e-3
    dt = 10.0
    Tn, Tp1, B = T.pennes_step(T0, C, g, Q, dt, 36.8, 1e18, 0.0)
    Tv, _ = _fine(T0, C, g, Q, dt, 1000, 36.8, 1e18, 0.0)
    assert np.abs(Tn - Tv).max() < 1e-6
    assert Tp1 == pytest.approx(36.8, abs=1e-9)


def test_pennes_pool_coupled_vs_fine_loop_small_dt():
    T0, C, g, Q = _body()
    dt = 0.05
    Tn, Tp1, B = T.pennes_step(T0, C, g, Q, dt, 36.8, 7700.0, 10.0)
    Tv, tp = _fine(T0, C, g, Q, dt, 1000, 36.8, 7700.0, 10.0)
    assert np.abs(Tn - Tv).max() < 1e-6 and abs(Tp1 - tp) < 1e-6


@pytest.mark.parametrize("eps_val", [0.0, 0.3, 0.6])
def test_pennes_energy_closes_also_with_eps_cc(eps_val):
    T0, C, g, Q = _body()
    eps = np.full(len(T0), eps_val)
    for dt in (1.0, 60.0, 600.0):
        Tn, Tp1, B = T.pennes_step(T0, C, g, Q, dt, 36.8, 7700.0, 10.0, eps=eps)
        gain = float(np.sum(C * (Tn - T0)))
        loss = 7700.0 * (Tp1 - 36.8)
        assert abs(gain + loss - (Q.sum() * dt - 10.0 * dt)) < 1e-9
        # the voxels' gain from the blood plus the pool's loss is zero
        assert abs(float(B.sum()) + loss + 10.0 * dt) < 1e-9


def test_pennes_g_zero_dead_hot_and_frozen():
    n = 6
    C = np.full(n, 437.5)
    # dead: g = 0 -> only Q moves the voxel, nothing divides by zero
    Tn, Tp1, B = T.pennes_step(np.full(n, 20.0), C, np.zeros(n), np.full(n, 0.2), 100.0, 36.8, 7700.0)
    assert np.allclose(Tn, 20.0 + 0.2 * 100.0 / 437.5, atol=1e-12) and Tp1 == 36.8
    assert np.abs(B).max() < 1e-9
    # hot (60 C, Omega = 1 coagulated): the perfusion factor kills g; the voxel is not pulled back to the pool
    f = T.perfusion_factor(np.array([60.0]), omega=1.0)
    assert f[0] == 0.0
    g = T.perfusion_g(5e-4, 5e-5, f)
    Tn, Tp1, B = T.pennes_step(np.array([60.0]), np.array([437.5]), np.atleast_1d(g), np.array([0.0]), 60.0, 36.8, 7700.0)
    assert Tn[0] == 60.0
    # frozen (-5 C): flow stops by the smoothstep, metabolism too; a warm pool does not thaw it
    f = T.perfusion_factor(np.array([-5.0]))
    assert f[0] == 0.0
    assert T.metab_power(684.0, 1e-4, np.array([-5.0]))[0] == 0.0
    # in between, smooth and bounded
    ft = T.perfusion_factor(np.linspace(-5, 40, 91))
    assert ft.min() == 0.0 and ft.max() == 1.0 and np.all(np.diff(ft) >= 0)


def test_perfusion_factor_states():
    assert T.perfusion_factor(37.0) == 1.0
    assert T.perfusion_factor(37.0, circ=0.0) == 0.0            # heart stopped: dead body, no flag
    assert T.perfusion_factor(37.0, circ=0.5, vaso=2.0) == pytest.approx(1.0)
    assert T.perfusion_factor(37.0, omega=0.25) == pytest.approx(0.75)


def test_metab_q10():
    a = float(T.metab_power(684.0, 1e-3, 37.0))
    b = float(T.metab_power(684.0, 1e-3, 47.0))
    c = float(T.metab_power(684.0, 1e-3, 27.0))
    assert b / a == pytest.approx(2.0) and a / c == pytest.approx(2.0)


def _realistic_pool_case():
    rng = np.random.default_rng(2)
    n = 400
    C = np.full(n, 437.5)
    g = rng.lognormal(np.log(0.8), 0.8, n)
    g *= 350.0 / g.sum()                      # 350 W/K = 5.5 L/min x 3.8 J/(ml K)
    Q = np.full(n, 82.0 / 400.0) * rng.uniform(0.5, 1.5, n)
    Cp, Tp, Qo = 7760.0, 36.8, 10.0
    Qs0 = np.zeros(n)
    Qs0[:100] = -(Q.sum() - Qo) / 100.0        # steady: skin loses what the body makes
    T0 = Tp + (Q + Qs0) / g
    return C, g, Q, Qs0, T0, Tp, Cp, Qo


def _ref600(C, g, Q, Qs, T0, Tp, Cp, Qo, tot=600.0, N=60000):
    h = tot / N
    Tv, tp = T0.copy(), Tp
    for _ in range(N):
        flow = g * (tp - Tv)
        Tv = Tv + h * (flow + Q + Qs) / C
        tp = tp + h * (-flow.sum() - Qo) / Cp
    return Tv, tp


def test_pool_and_400_voxels_stable_at_all_dt_and_core_error_at_600s():
    """-100 W skin cooling for 600 s on pool + 400 voxels. Every dt is stable
    and monotone (the core only falls); the core error at 600 s against the fine
    loop is under 0.02 K with `pennes_advance` (steps of at most 20 s). A single
    600 s backward-Euler step is first order and is NOT claimed to meet it."""
    C, g, Q, Qs0, T0, Tp, Cp, Qo = _realistic_pool_case()
    Qs = Qs0.copy()
    Qs[:100] -= 1.0
    Tv, tp = _ref600(C, g, Q, Qs, T0, Tp, Cp, Qo)
    assert tp < Tp - 0.2                       # the case really cools the core
    errs = {}
    for dt in (1, 10, 60, 300, 600):
        Tn, tpn = T0.copy(), Tp
        trace = [tpn]
        for _ in range(600 // dt):
            Tn, tpn, _B = T.pennes_step(Tn, C, g, Q + Qs, float(dt), tpn, Cp, Qo)
            trace.append(tpn)
        assert np.all(np.diff(trace) <= 1e-12), ("core rose while cooling", dt)    # no oscillation
        assert np.all(np.isfinite(Tn))
        errs[dt] = abs(tpn - tp)
    assert errs[1] < 1e-3 and errs[10] < 5e-3 and errs[60] < 0.02
    assert errs[600] > errs[60]                # single big steps are less accurate
    Tn, tpn, _B = T.pennes_advance(T0, C, g, Q + Qs, 600.0, Tp, Cp, Qo)
    assert abs(tpn - tp) < 0.02, abs(tpn - tp)
    assert np.abs(Tn - Tv).max() < 0.02


def test_advance_energy_closes():
    C, g, Q, Qs0, T0, Tp, Cp, Qo = _realistic_pool_case()
    Tn, tpn, B = T.pennes_advance(T0, C, g, Q + Qs0, 600.0, Tp, Cp, Qo)
    gain = float(np.sum(C * (Tn - T0))) + Cp * (tpn - Tp)
    assert abs(gain - ((Q + Qs0).sum() - Qo) * 600.0) < 1e-6


def test_pennes_slab_matches_cosh_within_2pct():
    """1-D steady Pennes slab: k T'' = rho_b c_b w (T - T_a) - q with the ends held.
    Solved by the lattice-style split (explicit conduction, pennes_step for the
    sources on a stiff pool) and compared with the closed form sinh solution."""
    mus = T.TISSUE["muscle"]
    L, n = 0.05, 50
    dx = L / n
    k, rho, c = mus.k, mus.rho, mus.c
    w = mus.w_b
    q = mus.q_m
    Ta = 36.8
    T1, T2 = 36.8, 33.0                      # core side, skin side (held)
    m = math.sqrt(T.RHOC_B * w / k)
    th_part = q / (T.RHOC_B * w)             # steady offset above arterial blood
    th1, th2 = T1 - Ta - th_part, T2 - Ta - th_part
    xs = (np.arange(n) + 0.5) * dx
    exact = Ta + th_part + (th1 * np.sinh(m * (L - xs)) + th2 * np.sinh(m * xs)) / math.sinh(m * L)
    V = dx ** 3
    C = np.full(n, rho * c * V)
    g = np.full(n, T.perfusion_g(w, V))
    Qv = np.full(n, q * V)
    Tn = np.full(n, 35.0)
    G = k * dx                               # W/K between neighbours (area dx^2, length dx)
    dt = 1.0
    for _ in range(40000):
        flux = G * (Tn[:-1] - Tn[1:]) * dt
        d = np.zeros(n)
        d[:-1] -= flux
        d[1:] += flux
        d[0] += G * 2 * (T1 - Tn[0]) * dt        # held end, half a cell away
        d[-1] += G * 2 * (T2 - Tn[-1]) * dt
        Tn = Tn + d / C
        Tn, _tp, _B = T.pennes_step(Tn, C, g, Qv, dt, Ta, 1e18, 0.0)
    rel = np.abs(Tn - exact) / np.abs(exact - Ta + 1e-9).clip(1.0)
    assert np.abs(Tn - exact).max() / (T1 - T2) < 0.02, np.abs(Tn - exact).max()
    assert rel.max() < 0.02


# ── 4. BSA and the two MET conventions ───────────────────────────────────────
def test_bsa_dubois_70kg_175cm():
    assert T.bsa_dubois(70.0, 1.75) == pytest.approx(1.85, abs=0.01)


def test_met_conventions_stay_separate():
    assert T.met_to_w(1.0, 70.0) == pytest.approx(85.4)
    bsa = 1.85
    assert T.iso_met_to_w(1.0, bsa) == pytest.approx(107.58, abs=0.1)
    assert abs(T.iso_met_to_w(1.0, bsa) - T.met_to_w(1.0, 70.0)) > 15.0     # they differ by design
    assert T.MET_W_PER_KG == 1.22 and T.ISO_MET_W_PER_M2 == 58.15
