"""T2 JOINERY (src/voxel/joinery.py): joint kinds as data, capacity per mode
with substrate caps, cure laws, weld as a rate, composites, load sharing and
fusing. Stand-alone: no World.

Tags: test_echo_* (fitted to the number), test_codeonly_* (code against its
own table or formula), test_u_* (independent, source unverified).
"""
import math

import numpy as np
import pytest

from src.voxel import breaking as B
from src.voxel import joinery as J


# ── fasteners ─────────────────────────────────────────────────────────────────
def test_codeonly_nail_withdrawal_is_1_2_kN():
    """54.12 G^2.5 D per mm: G 0.5, D 3.3 mm, 38 mm in: 1199.7 N (c)."""
    assert J.nail_withdrawal(0.5, 3.3, 38.0) == pytest.approx(1199.7, abs=0.1)
    assert J.nail_allowable(0.5, 3.3, 38.0) == pytest.approx(213.0, rel=0.15)
    c = J.capacity("nail", G=0.5, D_mm=3.3, L_mm=38.0)
    assert c.Ft == pytest.approx(1199.7, abs=0.1) and c.Fs > 0
    # two nails carry twice
    assert J.capacity("nail", n=2).Ft == pytest.approx(2 * c.Ft)
    # deeper or denser wood holds more, as the law says
    assert J.capacity("nail", L_mm=50.0).Ft > c.Ft
    assert J.capacity("nail", G=0.6).Ft > c.Ft


def test_codeonly_screw_stand_in_copy_marked_and_numbers():
    """#10 (4.83 mm) screw in pine G 0.45, 19 mm of thread: 2.0 kN (c). The
    copy is replaced by an import of fasteners.py at wiring; lane X compares."""
    assert J.screw_withdrawal_standin(0.45, 4.83, 19.0) == pytest.approx(2.0e3, rel=0.02)
    # lateral: Mode IV yield 0.75-1.1 kN (u)
    z = J.dowel_lateral_yield_standin(0.45, 4.83)
    assert 700.0 < z < 1100.0
    assert J.capacity("screw", G=0.45, D_mm=4.83, L_mm=19.0).Ft == pytest.approx(
        J.screw_withdrawal(0.45, 4.83, 19.0))


def test_codeonly_screw_stand_in_equals_fasteners_when_it_exists():
    f = pytest.importorskip("src.voxel.fasteners")
    if not hasattr(f, "screw_withdrawal"):
        pytest.skip("fasteners.py has no screw_withdrawal yet")
    for G, D, L in ((0.45, 4.83, 19.0), (0.6, 3.5, 25.0)):
        # fasteners.py takes metres, the stand-in takes mm
        assert f.screw_withdrawal(G, D * 1e-3, L * 1e-3) == pytest.approx(
            J.screw_withdrawal_standin(G, D, L), rel=1e-12)
        # and the wrapper joinery calls gives the same number from mm
        assert J.screw_withdrawal(G, D, L) == pytest.approx(
            J.screw_withdrawal_standin(G, D, L), rel=1e-12)


def test_codeonly_capstan_three_turns_at_mu_0_3_is_285():
    assert J.capstan_ratio(0.3, 6 * math.pi) == pytest.approx(285.0, rel=0.005)
    assert J.capstan_ratio(0.0, 10.0) == 1.0
    assert J.capstan_ratio(0.3, 2 * math.pi) < J.capstan_ratio(0.3, 4 * math.pi)


def test_lashing_slips_with_few_turns_and_breaks_with_many():
    brk, tail = 10.6e3, 0.02 * 10.6e3            # 12 mm manila (bonds table), a weak tail
    kind = J.JOIN_KINDS["lash_square"]
    caps = []
    for turns in (0.5, 1, 2, 3, 5):
        cap, mode = J.lashing_capacity(brk, kind["knot_eff"], turns, 0.3, tail)
        caps.append((cap, mode))
    assert caps[0][1] == "slip" and caps[-1][1] == "break"
    assert [c for c, _ in caps] == sorted(c for c, _ in caps)       # monotone in turns
    assert caps[-1][0] == pytest.approx(0.5 * brk)                  # about half its load
    for k in ("lash_overhand", "lash_square", "lash_bowline"):
        assert 0.45 <= J.JOIN_KINDS[k]["knot_eff"] <= 0.75
    bow = J.lashing_capacity(brk, J.JOIN_KINDS["lash_bowline"]["knot_eff"], 10, 0.3, tail)[0]
    ove = J.lashing_capacity(brk, J.JOIN_KINDS["lash_overhand"]["knot_eff"], 10, 0.3, tail)[0]
    assert bow > ove                                                # per-kind efficiency


def test_sewn_seam_and_clamp():
    c = J.capacity("stitch", n=10, thread_break=40.0, cloth_tear=1e9)
    assert c.Fs == pytest.approx(10 * 0.8 * 40.0)
    c = J.capacity("stitch", n=10, thread_break=40.0, cloth_tear=200.0)
    assert c.Fs == 200.0                                       # capped by the cloth
    cl = J.capacity("clamp", normal=500.0, mu=0.4)
    assert cl.Fs == pytest.approx(200.0)                       # F = mu N
    assert cl.Ft == 0.0


# ── capacity: the substrate caps and cure ─────────────────────────────────────
def test_capacity_zero_at_alpha_zero_and_monotone():
    for kind in ("hide_glue", "pva", "epoxy", "mortar_lime", "mortar_portland",
                 "weld_steel_1c"):
        zero = J.capacity(kind, alpha=0.0, area=8e-4)
        assert zero == J.ZERO
        prev = zero
        for a in (0.1, 0.4, 0.7, 1.0):
            c = J.capacity(kind, alpha=a, area=8e-4)
            assert all(x >= y for x, y in zip(c, prev)) and c.Fs > 0
            prev = c


def test_adhesive_modes_are_capped_by_the_matching_substrate_strength():
    A = 8e-4
    c = J.capacity("hide_glue", alpha=1.0, area=A, substrate="wood")
    assert c.Fs == pytest.approx(J.SUBSTRATE["wood"]["tau"] * A)          # shear: wood shear
    assert c.Ft == pytest.approx(J.SUBSTRATE["wood"]["sig_t"] * A)        # tension: PERPENDICULAR
    assert c.Ft < 0.6 * J.JOIN_KINDS["hide_glue"]["tau_a"] * A            # not the shear figure
    assert c.Ft < c.Fs
    # never above the weaker of two substrates, per mode
    wc = J.capacity("epoxy", 1.0, A, substrate=("wood", "steel"))
    assert wc.Fs <= J.capacity("epoxy", 1.0, A, substrate="steel").Fs
    assert wc.Fs <= J.SUBSTRATE["wood"]["tau"] * A and wc.Ft <= J.SUBSTRATE["wood"]["sig_t"] * A
    # epoxy on steel is limited by the glue, not the metal
    ss = J.capacity("epoxy", 1.0, A, substrate="steel")
    assert ss.Fs == pytest.approx(17.5e6 * A)
    # a longer lap carries less per unit length of lap
    short = J.capacity("pva", 1.0, A, length=0.05).Fs
    long_ = J.capacity("pva", 1.0, A, length=0.20).Fs
    assert long_ == pytest.approx(0.5 * short)


def test_wood_substrate_tension_equals_the_bonds_table():
    from src.bonds import materials as BM
    assert J.SUBSTRATE["wood"]["sig_t"] == BM.WOOD_T[2]


def test_combined_loading_check_boundary():
    c = J.Capacity(1000.0, 2000.0, 10.0, 0.0)
    assert J.utilisation(c, Fn=1000.0) == pytest.approx(1.0)
    assert J.utilisation(c, V=2000.0) == pytest.approx(1.0)
    assert J.utilisation(c, M=10.0) == pytest.approx(1.0)
    assert J.utilisation(c, Fn=500.0, V=1000.0, M=5.0) == pytest.approx(0.25 + 0.25 + 0.5)
    assert J.utilisation(c, Fn=-5000.0) == 0.0               # compression is not tension
    assert J.utilisation(J.ZERO, V=1.0) == math.inf


# ── cure ──────────────────────────────────────────────────────────────────────
def test_cure_aci_raw_at_7_days_and_normalised_at_28():
    assert J.aci_raw(7.0) == pytest.approx(0.7035, abs=5e-5)
    assert J.aci_raw(28.0) == pytest.approx(1.0072, abs=5e-4)
    assert J.aci_raw(1e6) == pytest.approx(1 / 0.85, rel=1e-3)            # tends to 1.18
    assert J.aci_alpha(28.0) == pytest.approx(1.0, abs=0.01)
    assert J.aci_alpha(1000.0) == 1.0                                      # clamped
    assert 0.0 < J.aci_alpha(7.0) < 0.71
    ts = [J.aci_alpha(t) for t in (1, 3, 7, 14, 28)]
    assert ts == sorted(ts)


def test_cure_hydration_run_through_the_joint_record():
    j = J.new_joint("mortar_portland", 1e-3, {("cement", "BOUND"): 40.0})
    assert j["alpha"] == 0.0 and J.capacity(J.JOIN_KINDS["mortar_portland"], j["alpha"],
                                            j["area"]) == J.ZERO
    day = 86400.0
    for _ in range(28):
        J.cure_step(j, day, T=J.T_REF, rh=1.0)
    assert j["alpha"] == pytest.approx(1.0, abs=0.01)
    # warmer cures faster (Arrhenius maturity)
    hot = J.new_joint("mortar_portland", 1e-3)
    cold = J.new_joint("mortar_portland", 1e-3)
    for _ in range(7):
        J.cure_step(hot, day, T=303.15, rh=1.0)
        J.cure_step(cold, day, T=278.15, rh=1.0)
    assert hot["alpha"] > cold["alpha"]


def test_cure_stops_when_the_mortar_dries():
    j = J.new_joint("mortar_portland", 1e-3)
    day = 86400.0
    for _ in range(3):
        J.cure_step(j, day, rh=1.0)
    a3, te3 = j["alpha"], j["te"]
    assert 0 < a3 < 0.6
    for _ in range(30):                                  # dries in the sun
        J.cure_step(j, day, rh=0.5)
    assert j["alpha"] == a3 and j["te"] == te3           # nothing happened
    J.cure_step(j, day, rh=1.0)                          # wetted again: it resumes
    assert j["alpha"] > a3
    # lime carbonation needs the humidity band and CO2
    lime_dry = J.new_joint("mortar_lime", 1e-3)
    lime_ok = J.new_joint("mortar_lime", 1e-3)
    lime_no_co2 = J.new_joint("mortar_lime", 1e-3)
    for _ in range(200):
        J.cure_step(lime_dry, day, rh=0.1)
        J.cure_step(lime_ok, day, rh=0.6)
        J.cure_step(lime_no_co2, day, rh=0.6, co2=0.0)
    assert lime_dry["alpha"] == 0.0 and lime_no_co2["alpha"] == 0.0
    assert lime_ok["alpha"] > 0.0
    assert J.moisture_band(0.6, 0.4, 0.8, 0.1) == 1.0
    assert J.moisture_band(0.95, 0.4, 0.8, 0.1) == 0.0
    assert 0 < J.moisture_band(0.35, 0.4, 0.8, 0.1) < 1


def test_cure_drying_and_set_on_cool():
    assert J.dry_alpha(0.5, 0.5, 0.1) == 0.0
    assert J.dry_alpha(0.3, 0.5, 0.1) == pytest.approx(0.5)
    assert J.dry_alpha(0.05, 0.5, 0.1) == 1.0
    p = J.new_joint("pva", 1e-3)
    assert J.cure_step(p, 1.0, w=0.3, w0=0.5, w_eq=0.1) == pytest.approx(0.5)
    h = J.new_joint("hide_glue", 1e-3)
    assert J.cure_step(h, 1.0, T=323.15) == 0.0                 # hot glue: liquid
    assert J.cure_step(h, 1.0, T=300.65) == pytest.approx(0.5)  # between 25 and 30 C
    assert J.cure_step(h, 1.0, T=293.15) == 1.0                 # cool: set
    assert J.cure_step(h, 1.0, T=323.15) == 0.0                 # reheated: lets go
    # a nail needs no cure
    n = J.new_joint("nail", 1e-5)
    assert n["alpha"] == 1.0


def test_weld_rate_is_zero_outside_its_window():
    row = J.JOIN_KINDS["weld_steel_1c"]
    c = row["cure"]
    assert J.weld_rate(c["T_bond"] - 1.0, 0.0, row) == 0.0              # too cold
    assert J.weld_rate(c["T_sol"] + 1.0, 0.0, row) == 0.0               # past the solidus
    assert J.weld_rate(1380.0, c["oxide_max"] * 2, row) == 0.0          # scale in the way
    inside = J.weld_rate(1380.0, 0.0, row)
    assert inside > 0.0
    assert J.weld_rate(1400.0, 0.0, row) > inside                       # Arrhenius
    # the weld simply never cures outside the window; inside it does, as a rate
    j = J.new_joint("weld_steel_1c", 1e-3)
    for _ in range(100):
        J.cure_step(j, 1.0, T=1200.0, oxide=0.0)
    assert j["alpha"] == 0.0
    for _ in range(100):
        J.cure_step(j, 1.0, T=1380.0, oxide=0.0)
    assert j["alpha"] == pytest.approx(1.0 - math.exp(-100 * inside), rel=1e-9)
    # wrought iron welds higher (and has no 1147 C ceiling)
    wi = J.JOIN_KINDS["weld_wrought"]
    assert J.weld_rate(1500.0, 0.0, wi) > 0.0 and J.weld_rate(1500.0, 0.0, row) == 0.0


# ── load sharing ──────────────────────────────────────────────────────────────
def test_stiffness_share_never_exceeds_the_sum_of_capacities():
    rng = np.random.default_rng(9)
    for _ in range(200):
        n = int(rng.integers(2, 6))
        k = list(rng.random(n) * 1e6 + 1e4)
        cap = list(rng.random(n) * 5e3 + 100)
        duc = list(rng.random(n) < 0.5)
        peak, _, order = J.parallel_peak(k, cap, duc)
        assert 0 < peak <= sum(cap) * (1 + 1e-12)
        assert sorted(order) == list(range(n))
        for frac in (0.3, 0.7, 1.0):
            sh = J.load_shares(frac * peak, k, cap, duc)
            assert sh is not None
            assert sum(sh) == pytest.approx(frac * peak, rel=1e-9)
            assert all(s <= c * (1 + 1e-9) for s, c in zip(sh, cap))
        assert J.load_shares(1.001 * peak, k, cap, duc) is None


def test_stiff_glue_takes_its_share_first_then_the_ductile_nail():
    k = [1.0e7, 1.0e5]              # glue line, nail
    cap = [5000.0, 1200.0]
    sh = J.load_shares(1000.0, k, cap, [False, True])
    assert sh[0] == pytest.approx(1000.0 * 1e7 / (1e7 + 1e5))      # by stiffness
    # brittle glue lets go at its capacity; the ductile nail holds on at its own
    peak, x, order = J.parallel_peak(k, cap, [False, True])
    assert order[0] == 0 and peak >= cap[0] * (1 - 1e-9)
    assert peak < sum(cap)
    p2, _, _ = J.parallel_peak([1.0e5, 1.0e7], [1200.0, 5000.0], [True, False])
    assert p2 == pytest.approx(peak)


# ── composites ────────────────────────────────────────────────────────────────
def _axe():
    j1 = J.new_joint("hide_glue", 8e-4, rid=1, a="head", b="haft", alpha=1.0)
    j2 = J.new_joint("nail", 1e-5, rid=2, a="head", b="haft", alpha=1.0)
    j3 = J.new_joint("lash_square", 1e-5, rid=3, a="haft", b="cap", alpha=1.0)
    return J.new_group(9, "hand axe", ["head", "haft", "cap"], [j1, j2, j3])


def test_composite_keeps_its_parts_and_comes_apart_at_the_weakest_joint():
    g = _axe()
    assert J.components(g["members"], g["joints"]) == [["head", "haft", "cap"]]
    caps = {1: J.capacity("hide_glue", 1.0, 8e-4), 2: J.capacity("nail"),
            3: J.Capacity(2000.0, 2000.0, 0.0, 0.0)}
    loads = {1: (0.0, 500.0, 0.0), 2: (0.0, 300.0, 0.0), 3: (0.0, 1900.0, 0.0)}
    rid, u = J.weakest_joint(g, loads, caps)
    assert rid == 3 and u == pytest.approx((1900.0 / 2000.0) ** 2)
    # failing one of the redundant pair (glue OR nail between head and haft) does not
    # split the composite
    comps = J.fail_joint(g, 1, tick=5, mode="overload")
    assert len(comps) == 1 and g["status"] == "whole"
    comps = J.fail_joint(g, 3, tick=6)
    assert sorted(map(sorted, comps)) == [["cap"], ["haft", "head"]]
    assert g["status"] == "apart" and g["log"][-1][0] == "comes_apart"
    # the parts and the joints are all still there (it KEEPS them)
    assert g["members"] == ["head", "haft", "cap"] and len(g["joints"]) == 3
    comps = J.fail_joint(g, 2, tick=7)
    assert len(comps) == 3
    # a failed joint is never picked again
    assert J.weakest_joint(g, loads, caps) is None


def test_components_are_deterministic_and_handle_cycles():
    members = list("abcdef")
    joints = [{"a": "a", "b": "b"}, {"a": "b", "b": "c"}, {"a": "c", "b": "a"},
              {"a": "d", "b": "e"}]
    comps = J.components(members, joints)
    assert comps == [["a", "b", "c"], ["d", "e"], ["f"]]
    joints[0]["failed"] = (1, "x")
    assert J.components(members, joints) == comps       # the cycle still holds a-b via c


def test_codeonly_swing_three_outcomes_uncured_thin_haft_thick_haft():
    """1 kg head at 10 m/s stopping in 1 cm loads the joint with 5 kN. Uncured
    (alpha 0.1): the head flies off. Cured with a thin haft: the haft breaks.
    Cured with a thick haft: it holds. Only the cure and the haft differ."""
    F = B.swing_load(1.0, 10.0, 0.01)
    assert F == pytest.approx(5000.0)
    A_glue = 8e-4
    wood = B.STRENGTH["ash"]
    haft = lambda area_m2: wood.sigma_u * area_m2          # tension along the grain, N
    thin, thick = haft(0.4e-4), haft(6e-4)
    assert thin < F < thick
    uncured = J.capacity("hide_glue", 0.1, A_glue).Fs
    cured = J.capacity("hide_glue", 1.0, A_glue).Fs
    assert uncured < 1000.0 < F < cured
    assert J.weakest_link(F, {"joint": uncured, "haft": thick}) == "joint"
    assert J.weakest_link(F, {"joint": cured, "haft": thin}) == "haft"
    assert J.weakest_link(F, {"joint": cured, "haft": thick}) == "holds"
    # the first two differ only in elapsed time (cure), the third in the haft section
    assert J.weakest_link(F, {"joint": uncured, "haft": thin}) in ("joint", "haft")


# ── fusing ────────────────────────────────────────────────────────────────────
def test_fuse_totals_add_exactly_and_parts_end_fused():
    iron = ("iron", "BOUND")
    slag = ("slag", "GRAIN")
    a = {"id": 1, "grams": {iron: 123.456789, slag: 0.1}, "energy": 1.1e5, "mass": 0.123556789,
         "count": 0, "cap": 56.0}
    b = {"id": 2, "grams": {iron: 76.54321}, "energy": 0.4e5, "mass": 0.07654321,
         "count": 0, "cap": 34.0}
    c = {"id": 3, "grams": {iron: 0.3, slag: 0.7}, "energy": 10.0, "mass": 0.001, "count": 5,
         "cap": 0.5}
    new, parts = J.fuse([a, b, c], new_id=77, name="blade")
    assert new["grams"][iron] == math.fsum([123.456789, 76.54321, 0.3])
    assert new["grams"][slag] == math.fsum([0.1, 0.7])
    assert set(new["grams"]) == {iron, slag}                      # forms kept
    assert new["mass"] == math.fsum([a["mass"], b["mass"], c["mass"]])
    assert new["energy"] == math.fsum([1.1e5, 0.4e5, 10.0])
    assert new["count"] == 5
    assert new["T"] == pytest.approx(20.0 + new["energy"] / new["cap"])
    assert new["lineage"] == [1, 2, 3]
    for p in parts:
        assert p["status"] == "fused" and p["cause"] == "fused" and p["into"] == 77
    # T from a summed-enthalpy table when one is given
    new2, _ = J.fuse([dict(a), dict(b)], 78, T_of_H=lambda H: 20.0 + H / 450.0)
    assert new2["T"] == pytest.approx(20.0 + (1.5e5 / (0.123556789 + 0.07654321)) / 450.0)


def test_fuse_conserves_to_the_exact_sum_of_many_parts():
    rng = np.random.default_rng(1)
    parts = [{"id": i, "grams": {("iron", "BOUND"): float(g)}, "energy": float(e),
              "mass": float(g) / 1000.0}
             for i, (g, e) in enumerate(zip(rng.random(500) * 10, rng.random(500) * 1e3))]
    new, _ = J.fuse(parts, 999)
    assert new["grams"][("iron", "BOUND")] == math.fsum(p["grams"][("iron", "BOUND")]
                                                          for p in parts)
    assert new["energy"] == math.fsum(p["energy"] for p in parts)


def test_owned_matter_of_a_joint_is_its_glue_line():
    j = J.new_joint("hide_glue", 8e-4, {("hide_glue", "BOUND"): 0.8})
    assert J.owned(j) == {("hide_glue", "BOUND"): 0.8}


# ── one source for the strength and joint rows (record_data) ──────────────────
def test_joinery_and_breaking_read_the_one_table_in_record_data():
    """Joinery's joint kinds and substrate strengths ARE record_data's objects, not copies, and
    the tension figure of a substrate is the sigma_u of its STRENGTH row (the same row breaking
    reads). Retyping a table in either module fails the identity and the source scan."""
    import inspect
    from src.voxel import record_data as rd
    assert J.JOIN_KINDS is rd.JOIN_KINDS and J.SUBSTRATE is rd.SUBSTRATE
    for name, spec in rd._SUBSTRATE_ROWS.items():
        if spec["row"]:
            assert J.SUBSTRATE[name]["sig_t"] == rd.STRENGTH[spec["row"]]["sigma_u"], name
    assert J.SUBSTRATE["glass"]["sig_t"] == B.STRENGTH["glass"].sigma_u
    assert J.SUBSTRATE["wood"]["sig_t"] == B.STRENGTH["wood_perp"].sigma_u
    assert J.SUBSTRATE["steel"]["sig_t"] == B.STRENGTH["steel_mild"].sigma_u
    assert J.SUBSTRATE["leather"]["sig_t"] == rd.STRENGTH["hide"]["sigma_u"]
    jsrc = "\n".join(ln.split("#")[0] for ln in inspect.getsource(J).splitlines())
    bsrc = "\n".join(ln.split("#")[0] for ln in inspect.getsource(B).splitlines())
    assert "SUBSTRATE = {" not in jsrc and "JOIN_KINDS = {" not in jsrc and "sig_t" not in jsrc.replace('["sig_t"]', "")
    assert "sigma_u=" not in bsrc and "K_Ic=" not in bsrc
    # every capacity cap is read through those objects: a joint on glass is capped by glass's row
    A = 1e-4
    c = J.capacity("epoxy", 1.0, A, substrate="glass")
    assert c.Ft <= rd.STRENGTH["glass"]["sigma_u"] * A * (1 + 1e-12)


def test_every_joint_kind_has_a_cure_law_and_a_class_and_the_names_list_matches():
    from src.voxel import record_data as rd
    assert rd.JOIN_KIND_NAMES == tuple(rd.JOIN_KINDS)
    for name, row in rd.JOIN_KINDS.items():
        assert "cls" in row and "law" in row["cure"], name
