"""The material record (src/voxel/matrec.py, wave 13 topic T9).

Tags (plan section 9): a test_echo_* checks a number the constant was FITTED to;
a test_codeonly_* checks the code against its own table; a test_u_* leans on a
number that is independent but unverified, and reads
tests/data/sourced_numbers.json when it is there (and says which status it
saw). Untagged tests are exactness, closed-form or arithmetic identities and
may gate a scene.

The oracle tests (F1) read the frozen numbers in
tests/data/matrec_oracle_base19.json (lane A-O; B0 froze it from base19, no table row differs from base18). If the oracle or its tool is
missing they skip and say so.
"""
import dataclasses
import importlib.util
import inspect
import json
import os
import re
import sys

import numpy as np
import pytest

from src.voxel import matrec as M

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ORACLE = os.path.join(HERE, "data", "matrec_oracle_base19.json")
TOOL = os.path.join(ROOT, "tools", "matrec_extract.py")
SOURCED = os.path.join(HERE, "data", "sourced_numbers.json")
WORK = os.environ.get("W13_WORK", "/home/nisargparikh/.cache/mark1-build/w13/A-T9a")


def _tool():
    if not (os.path.exists(ORACLE) and os.path.exists(TOOL)):
        pytest.skip("oracle or tools/matrec_extract.py absent (lane A-O)")
    spec = importlib.util.spec_from_file_location("matrec_extract_t9a", TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["matrec_extract_t9a"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def X():
    return _tool()


@pytest.fixture(scope="module")
def doc(X):
    return X.load_oracle(ORACLE)


@pytest.fixture(scope="module")
def mats():
    from src.voxel import materials
    return materials


@pytest.fixture(scope="module")
def V(doc):
    ids = doc["ids"]["fluid"]
    by = {v: int(k) for k, v in ids.items()}
    return M.legacy_views({"mtin": by["MTIN"], "mlead": by["MLEAD"], "miron": by["MIRON"]})


# ── F1: the views equal the frozen oracle ───────────────────────────────────

TABLES = ("SOLID", "EMISS", "SPAN", "TOUGH", "TSHOCK", "MELT", "FREEZE", "PYRO",
          "O2_RATIO", "SOOT", "GLOW", "BURN_AREA", "TRANSMIT", "MATNAME", "SKIN_T")
ARRAYS = ("_DENS_ARR", "_CSOLID_ARR", "_KSOLID_ARR", "_SPAN_ARR", "_TOUGH_ARR",
          "_TSHOCK_ARR", "_EMISS_ARR", "_IGN_ARR", "_ALPHA", "_SKIN_DEPTH_ARR",
          "_SKIN_PATH_ARR", "_TRANSMIT_ARR")


@pytest.mark.parametrize("name", TABLES)
def test_view_table_equals_the_frozen_oracle(X, doc, V, name):
    assert X.enc(V[name]) == doc["tables"][name], name


@pytest.mark.parametrize("name", ARRAYS)
def test_view_array_equals_the_frozen_oracle_bit_for_bit(X, doc, V, name):
    want = X.decode_array(doc["arrays"][name])
    got = np.asarray(V[name])
    assert got.dtype == want.dtype and got.shape == want.shape, name
    assert got.tobytes() == want.tobytes(), name


def test_particle_table_equals_the_frozen_oracle(X, doc, V):
    assert X.enc(V["MPM"]) == doc["mpm_mat"]


def test_bond_rows_and_named_rows_equal_the_frozen_oracle(X, doc, V):
    b = doc["bond"]
    assert X.enc(V["BOND_ROW"]) == b["ROW"]
    assert X.enc(sorted(V["NO_BOND"])) == b["NO_BOND"]
    assert X.enc(V["BOND_CONSTS"]) == b["consts"]


def test_every_bond_pair_row_equals_the_frozen_oracle(X, doc, V):
    n = 0
    for key, want in doc["bond"]["pairs"].items():
        ma, mb, c2 = key.split(",")
        got = V["bond_row"](int(ma), int(mb), float(c2))
        assert (None if got is None else X.enc(tuple(got))) == want, key
        n += 1
    assert n == 12 * 13 // 2 * 5


def test_sound_ring_equals_the_frozen_oracle(X, doc, V):
    assert X.enc(V["SOUND_RING"]) == doc["other"]["SOUND"]["ring"]


def test_ids_and_counts_equal_the_oracle(doc):
    assert M.NMAT == doc["nmat"] == 12
    assert {str(r.id): r.key.upper() for r in M.BY_ID} == \
        {k: v for k, v in doc["ids"]["mat"].items()}


def test_views_equal_the_LIVE_materials_module(mats, V):
    for name in ("SOLID", "EMISS", "SPAN", "TOUGH", "TSHOCK", "MELT", "FREEZE",
                 "PYRO", "O2_RATIO", "SOOT", "GLOW", "BURN_AREA", "TRANSMIT",
                 "MATNAME", "SKIN_T"):
        assert V[name] == getattr(mats, name), name
    for name in ARRAYS:
        # the live arrays are 64 long since wave 13 (B3); the view has the twelve old ids
        live = getattr(mats, name)
        assert live.shape == (mats.NMAT_PAD,) and len(V[name]) == 12, name
        assert np.array_equal(V[name], live[:len(V[name])], equal_nan=True), name
        assert V[name].dtype == live.dtype, name
    assert V["SOUND_RING"] == mats.SOUND["ring"]


def test_views_equal_the_LIVE_bond_store(mats, V):
    from src.bonds import materials as B
    assert V["BOND_ROW"] == B.ROW
    assert V["NO_BOND"] == B.NO_BOND
    for name, val in V["BOND_CONSTS"].items():
        assert getattr(B, name) == val, name
    for ma in range(12):
        for mb in range(ma, 12):
            for c2 in (0.0, 0.25, 0.5, 0.75, 1.0):
                assert V["bond_row"](ma, mb, c2) == B.pair_row(ma, mb, c2)


def test_molten_fluid_ids_equal_materials_and_are_5_6_7(mats, doc):
    assert (mats.MTIN, mats.MLEAD, mats.MIRON) == (5, 6, 7)
    by = {v: int(k) for k, v in doc["ids"]["fluid"].items()}
    assert (by["MTIN"], by["MLEAD"], by["MIRON"]) == (5, 6, 7)
    v1 = M.legacy_views((mats.MTIN, mats.MLEAD, mats.MIRON))
    v2 = M.legacy_views({"MTIN": 5, "MLEAD": 6, "MIRON": 7})
    assert v1["MELT"] == v2["MELT"] == mats.MELT
    assert {m: t[1] for m, t in v1["MELT"].items()} == {M.TIN: 5, M.LEAD: 6, M.IRON: 7}
    assert set(M.MOLTEN_KEYS) == {r.melt.fluid_key for r in M.BY_ID if r.melt}


def test_the_AIR_row_is_present(V, mats):
    air = M.BY_ID[M.AIR]
    assert air.id == 0 and air.key == "air"
    assert V["SOLID"][0] == mats.SOLID[0] and V["TOUGH"][0] == 1.0e9
    assert V["_TRANSMIT_ARR"][0] == 1.0 and V["EMISS"][0] == 0.0


def test_the_module_does_not_import_materials():
    src = inspect.getsource(M)
    assert not re.search(r"^\s*(from|import)\s+.*materials", src, re.M)


def test_every_legacy_override_has_a_reason():
    n = 0
    for r in M.REC.values():
        for group, d in r.legacy.items():
            for field_, e in d.items():
                assert isinstance(e, tuple) and len(e) == 2, (r.key, group, field_)
                assert isinstance(e[1], str) and len(e[1]) > 15, (r.key, group, field_)
                n += 1
    assert n > 30


def test_the_real_disagreements_are_written_down_as_overrides():
    got = {(r.key, group, f) for r in M.REC.values() for group, d in r.legacy.items() for f in d}
    for must in (("stone", "bonds", "rho"), ("iron", "bonds", "rho"), ("flesh", "bonds", "rho"),
                 ("tin", "bonds", "nu"), ("iron", "bonds", "st"), ("wood", "mpm", "strength"),
                 ("stone", "mpm", "strength"), ("glass", "voxel", "tshock")):
        assert must in got, must


# ── row sanity, by data rules ───────────────────────────────────────────────

SOLIDS = [r for r in M.BY_ID if r.id != M.AIR]


def test_codeonly_row_sanity_by_data_rules():
    for r in SOLIDS:
        assert 0.0 < r.nu < 0.5, r.key
        assert r.ductility >= 0.0, r.key
        if r.E_perp is None and r.ductility < 0.05:
            assert r.st <= r.sc, r.key            # wood is excluded: its grain allows st > sc
        if r.ductility >= 0.05 and r.sy is not None:
            assert r.sy <= r.st, r.key
        assert r.mu_class in M.MU_CLASSES, r.key
        assert r.rho > 0 and r.cp > 0 and r.k > 0 and 0 <= r.eps <= 1, r.key
    for r in M.BASE.by_id_list:
        ks, ke = zip(*[(p[1], p[2]) for p in r.derate])
        T = [p[0] for p in r.derate]
        assert T == sorted(T) and len(set(T)) == len(T), r.key
        assert all(a >= b for a, b in zip(ks, ks[1:])), r.key        # non-increasing
        assert all(a >= b for a, b in zip(ke, ke[1:])), r.key
        assert min(ke) >= 0 and min(ks) >= 0, r.key
        assert r.derate_at(20.0) == (1.0, 1.0), r.key                # 1 at 20 C
    air = M.BY_ID[0]
    assert air.E == 0.0 and air.st == 0.0                            # a gas has no strength


def test_codeonly_the_cold_side_exists():
    for key in ("wood", "stone", "iron"):
        ks, ke = M.REC[key].derate_at(-20.0)
        assert ks > 1.0, key
    assert M.REC["glass"].derate_at(-20.0) == (1.0, 1.0)
    assert M.REC["iron"].dbtt_C == (-30.0, 20.0)
    d_cold, g_cold = M.REC["iron"].brittle_at(-60.0)
    d_warm, g_warm = M.REC["iron"].brittle_at(20.0)
    assert d_cold < 0.05 < d_warm
    assert g_cold < 0.2 * g_warm
    assert M.REC["stone"].brittle_at(-60.0) == (M.REC["stone"].ductility, M.REC["stone"].gimp)


def test_codeonly_structural_lists_exactly_the_seven():
    got = {r.key for r in M.BY_ID if r.structural()}
    assert got == {"wood", "stone", "iron", "glass", "tin", "lead", "char"}
    assert set(M.BASE.structural_keys()) == got
    assert M.structural("iron") and not M.structural("rope")
    assert not M.REC["petiole"].structural()


def test_codeonly_every_row_has_ductility_and_brittle_is_read_from_it():
    assert M.REC["glass"].brittle() and M.REC["stone"].brittle() and M.REC["wood"].brittle()
    assert not M.REC["iron"].brittle() and not M.REC["rope"].brittle()
    assert M.REC["glass"].sf_eff() == 5.0e7 and M.REC["iron"].sf_eff() == M.REC["iron"].st


def test_codeonly_R_shock_glass_and_stone():
    g, s = M.REC["glass"], M.REC["stone"]
    assert abs(g.R_shock() - 61.9) < 0.1
    assert abs(g.tshock_face() - 123.8) < 0.1
    assert abs(s.R_shock() - 28.1) < 0.1
    assert M.REC["ash"].R_shock() == float("inf")


# ── at(T, D, temper) ────────────────────────────────────────────────────────

@pytest.mark.parametrize("key", [r.key for r in SOLIDS])
def test_at_returns_the_stored_values_at_D0_T20(key):
    r = M.REC[key]
    e = r.at(20.0, 0.0, 1.0)
    assert (e.E, e.st, e.sc, e.sf, e.tau) == (r.E, r.st, r.sc, r.sf_eff(), r.tau)
    assert e.sy == r.sy and e.H == r.H


def test_at_limits_as_D_goes_to_one():
    r = M.REC["iron"]
    e, e0 = r.at(20.0, 1.0 - 1e-12), r.at(20.0, 0.0)
    for v, v0 in zip((e.E, e.st, e.sc, e.sf, e.tau, e.sy), (e0.E, e0.st, e0.sc, e0.sf, e0.tau, e0.sy)):
        assert v < 2e-6 * v0                            # compression falls as the square root: 1e-6
    assert r.at(20.0, 1.0).E == 0.0 and r.at(20.0, 1.0).sc == 0.0
    assert r.at(20.0, 5.0).st == 0.0                       # D is clipped, never negative strength


def test_at_compression_falls_as_sqrt_and_the_rest_linearly():
    r = M.REC["stone"]
    e0, e = r.at(20.0, 0.0), r.at(20.0, 0.75)
    assert e.sc == pytest.approx(e0.sc * 0.5, rel=1e-12)
    assert e.st == pytest.approx(e0.st * 0.25, rel=1e-12)
    assert e.E == pytest.approx(e0.E * 0.25, rel=1e-12)
    assert e.tau == pytest.approx(e0.tau * 0.25, rel=1e-12)


def test_at_temper_multiplies_sy_and_H_only():
    r = M.REC["iron"]
    a, b = r.at(20.0, 0.2, 1.0), r.at(20.0, 0.2, 5.0)
    assert b.sy == pytest.approx(5.0 * a.sy) and b.H == pytest.approx(5.0 * a.H)
    assert (b.E, b.st, b.sc, b.sf, b.tau) == (a.E, a.st, a.sc, a.sf, a.tau)
    assert M.REC["wood"].at(20.0, 0.0, 3.0).sy is None


def test_codeonly_at_applies_temperature_from_the_derate_table():
    r = M.REC["iron"]
    assert r.at(600.0).st == pytest.approx(r.st * 0.47)
    assert r.at(600.0).E == pytest.approx(r.E * 0.31)
    assert r.at(550.0).st == pytest.approx(r.st * (0.78 + 0.47) / 2)       # linear between rows
    assert M.REC["wood"].at(400.0).st == 0.0                                # past pyrolysis
    assert M.REC["tin"].at(232.0).st == 0.0 and M.REC["tin"].at(126.0).st == pytest.approx(M.REC["tin"].st * 0.5)


def test_at_works_on_arrays():
    r = M.REC["glass"]
    T = np.array([20.0, 550.0, 720.0])
    D = np.array([0.0, 0.5, 0.0])
    e = r.at(T, D)
    assert e.st.shape == (3,) and e.st[0] == r.st
    assert e.st[1] == pytest.approx(r.st * 0.3 * 0.5)


# ── blend and porous ────────────────────────────────────────────────────────

def test_blend_of_one_material_returns_that_material():
    for key in ("wood", "stone", "iron", "glass", "tin"):
        r = M.REC[key]
        b = M.blend([(r, 1.0, "grain")])
        for f in ("rho", "cp", "k", "E", "nu", "sy", "st", "sc", "tau", "H", "gimp", "ductility", "alpha"):
            assert getattr(b, f) == getattr(r, f), (key, f)
        assert b.sf_eff() == r.sf_eff()


def test_blend_E_is_the_Hill_average_between_Reuss_and_Voigt():
    s, i = M.REC["stone"], M.REC["iron"]
    m = [0.5, 0.5]
    b = M.blend([(s, m[0], "grain"), (i, m[1], "grain")])
    v = np.array([m[0] / s.rho, m[1] / i.rho]); phi = v / v.sum()
    voigt = phi[0] * s.E + phi[1] * i.E
    reuss = 1.0 / (phi[0] / s.E + phi[1] / i.E)
    assert reuss < b.E < voigt
    assert b.E == pytest.approx(0.5 * (voigt + reuss), rel=1e-12)
    assert b.rho == pytest.approx(1.0 / (m[0] / s.rho + m[1] / i.rho), rel=1e-12)
    assert b.cp == pytest.approx(m[0] * s.cp + m[1] * i.cp, rel=1e-12)


def test_codeonly_blend_strength_is_capped_by_the_weakest_continuous_phase():
    wood, iron = M.REC["wood"], M.REC["iron"]
    b = M.blend([(wood, 0.5, "matrix"), (iron, 0.5, "grain")])
    assert b.st <= wood.st and b.sc <= wood.sc
    free = M.blend([(wood, 0.5, "grain"), (iron, 0.5, "grain")])      # dominant-volume phase continuous
    assert free.st <= max(wood.st, iron.st)


def test_blend_refuses_dissolved_and_empty():
    with pytest.raises(ValueError):
        M.blend([(M.REC["iron"], 0.99, "bound"), (M.REC["stone"], 0.01, "dissolved")])
    with pytest.raises(ValueError):
        M.blend([])
    assert "alloy" in M.blend.__doc__ and "T4" in M.blend.__doc__


def test_blend_calls_mixrules_for_density_c_and_k():
    """The real interface: RULES[kind](entries, slot) -> array[n], entries from
    make_entries(n, cell, key, frac). c comes back in J/g/K, blend returns J/kg/K."""
    seen = {}

    class Fake:
        @staticmethod
        def make_entries(n, cell, key, frac):
            seen["entries"] = (n, list(cell), list(key), list(frac))
            return "ENT"
        RULES = {k: (lambda kind, v: (lambda ent, slot="engine": (seen.setdefault(kind, ent), np.array([v]))[1]))(k, v)
                 for k, v in {"rho": 111.0, "c": 2.5, "k": 333.0, "mu": 4.0}.items()}
    b = M.blend([(M.REC["wood"], 0.5, "grain"), (M.REC["stone"], 0.5, "grain")], rules=Fake)
    assert (b.rho, b.cp, b.k) == (111.0, 2500.0, 333.0)
    assert b.extra["viscosity"] == 4.0 and "k_source" not in b.extra
    assert [seen[k] for k in ("rho", "c", "k", "mu")] == ["ENT"] * 4
    n, cell, key, frac = seen["entries"]
    assert n == 1 and cell == [0, 0] and key == sorted(key) and sum(frac) == pytest.approx(1.0)


def test_blend_without_mixrules_says_the_k_is_a_standin():
    b = M.blend([(M.REC["wood"], 0.5, "grain"), (M.REC["stone"], 0.5, "grain")], rules=object())
    assert b.k == M.REC["stone"].k or b.k == M.REC["wood"].k
    assert "standin" in b.extra["k_source"]


def test_cross_blend_holds_no_thermal_conductivity_code():
    """Critic 8: there is no second thermal rule in the material record."""
    src = inspect.getsource(M.blend)
    code = "\n".join(l.split("#")[0] for l in src.splitlines())
    # drop the docstring: it may SAY there is no law
    code = re.sub(r'""".*?"""', "", code, count=1, flags=re.S)
    for banned in ("prod", "geometric", "bruggeman", "maxwell", "eucken", "wiener",
                   "np.exp", "np.log", "math.log", "math.exp", "** phi", "**phi", "sqrt"):
        assert banned not in code.lower(), banned
    assert re.search(r"\bk\s*=\s*dom\.k\b", code)           # the only k is a stand-in or the rule's
    # and the whole module has no conductivity mixing law anywhere else
    whole = inspect.getsource(M)
    assert "bruggeman" not in whole.lower() and "maxwell" not in whole.lower()


@pytest.mark.skipif(M._mixrules is None, reason="mixrules (T1) not built yet; the cross test skips")
def test_cross_blend_uses_the_real_mixrules():
    b = M.blend([(M.REC["iron"], 0.5, "grain"), (M.REC["tin"], 0.5, "grain")])
    assert b.k > 0 and "k_source" not in b.extra


def test_porous_follows_Gibson_and_Ashby():
    wood = M.REC["wood"]
    for rel in (0.1, 0.5, 1.0):
        p = M.porous(wood, rel)
        assert p.rho == pytest.approx(wood.rho * rel)
        assert p.E / wood.E == pytest.approx(rel ** 2)
        assert p.sc / wood.sf_eff() == pytest.approx(0.2 * rel ** 1.5)
        assert p.sf / wood.sf_eff() == pytest.approx(0.65 * rel ** 1.5)
    with pytest.raises(ValueError):
        M.porous(wood, 0.0)
    assert not M.porous(wood, 0.05).structural()           # a very light foam is no member


# ── props, retire, ingest ───────────────────────────────────────────────────

BONDS_KEYS = {"E", "E_perp", "nu", "sy", "st", "st_perp", "sc", "tau", "ductility",
              "KIc", "gimp", "weibull_m"}


def test_props_wood_equals_the_bonds_table(V):
    a, t = M.props("wood", "along"), M.props("wood", "across")
    assert set(a) == set(t) == BONDS_KEYS
    wl, wt = V["BOND_CONSTS"]["WOOD_L"], V["BOND_CONSTS"]["WOOD_T"]
    assert (a["E"], a["st"]) == (wl[0], wl[2]) == (1.1e10, 6.0e7)
    assert (t["E"], t["st"]) == (wt[0], wt[2]) == (5.5e8, 3.0e6)
    assert a["nu"] == wl[1] and a["E_perp"] == 5.5e8 and a["st_perp"] == 3.0e6
    assert t["sc"] == M.REC["wood"].sc_perp and a["sc"] == M.REC["wood"].sc


def test_props_iron_is_200_GPa_and_250_MPa(V):
    p = M.props("iron")
    assert p["E"] == 2.0e11
    assert p["sy"] == 2.5e8 == V["BOND_CONSTS"]["IRON_B"][2]     # the bond store's strength IS the yield
    assert p["st"] == 3.5e8 and p["ductility"] == 0.20
    assert M.props(M.IRON) == p and M.props("iron", None) == p


def test_props_values_come_from_the_record_and_include_the_new_fields():
    p = M.props("glass")
    assert p["KIc"] == 0.75e6 and p["gimp"] == 200.0 and p["weibull_m"] > 0 and p["ductility"] == 0.0
    assert M.props("tin")["sy"] == 1.1e7
    with pytest.raises(ValueError):
        M.props("iron", "sideways")


def test_a_retired_material_keeps_its_id_and_names_its_successor():
    t = M.BASE.retire("char", "ash")
    assert t.row("char").id == M.CHAR and t.row("char").successor == "ash"
    assert t.row(M.CHAR).key == "char"                      # an old id still resolves
    assert t.resolve("char").key == "ash" and t.resolve(M.CHAR).id == M.ASH
    assert M.CHAR not in t.live_ids() and M.ASH in t.live_ids()
    assert [r.id for r in t.by_id_list] == list(range(12))  # never renumbered
    assert t.props("char") == M.BASE.props("ash")           # the successor answers
    assert M.REC["char"].successor is None                  # the shipped table is untouched
    with pytest.raises(ValueError):
        t.retire("ash", "ash")


def test_a_retired_chain_resolves_and_a_loop_is_refused():
    t = M.BASE.retire("char", "ash").retire("ash", "stone")
    assert t.resolve("char").key == "stone"
    loop = M.BASE.retire("char", "ash").retire("ash", "char")
    with pytest.raises(ValueError):
        loop.resolve("char")


ICE_STANDIN = {"key": "ice", "density": 917.0, "cp_Jg": 2.09, "k": 2.2, "emissivity": 0.97,
               "E": 9.3e9, "nu": 0.32, "st": 1.0e6, "span": 20, "tough_kJm2": 0.3,
               "latent_Jg": 333.55, "isotherm": [1, 2, 3]}


def test_an_ICE_row_ingested_from_a_dict_gets_mat_id_12():
    t = M.ingest_row(ICE_STANDIN)
    ice = t.row("ice")
    assert ice.id == 12 == M.ICE and t.nmat == 13 and t.row(12) is ice
    assert (ice.rho, ice.cp, ice.k, ice.eps) == (917.0, 2090.0, 2.2, 0.97)
    assert ice.gimp == 300.0 and ice.span_ref == 20 and ice.mu_class == "ice"
    assert ice.extra["latent_Jg"] == 333.55 and ice.extra["isotherm"] == [1, 2, 3]
    assert M.BASE.nmat == 12 and M.NMAT == 12               # the shipped table did not move
    assert t.arrays()["rho"].shape == (13,) and t.derate_grid()[0].shape[0] == 13


def test_ingest_leaves_the_twelve_old_rows_and_views_alone(V):
    t = M.ingest_row(ICE_STANDIN)
    V2 = M.legacy_views((5, 6, 7), table=t)
    assert V2["SOLID"] == V["SOLID"] and V2["BOND_ROW"] == V["BOND_ROW"]
    V3 = M.legacy_views((5, 6, 7), table=t, upto=13)
    assert 12 in V3["SOLID"] and V3["SOLID"][12][0] == 917.0


def test_ingest_refuses_a_duplicate_key_and_a_gap():
    t = M.ingest_row(ICE_STANDIN)
    with pytest.raises(ValueError):
        t.ingest(dict(ICE_STANDIN))
    with pytest.raises(ValueError):
        M.BASE.ingest(dict(ICE_STANDIN), id=20)
    with pytest.raises(ValueError):
        M.BASE.ingest({"density": 1.0})


def test_ingest_ice_reads_icephase_when_it_exists():
    got = M.ingest_ice()
    if M._icephase is None:
        assert got is None
        pytest.skip("icephase (T6) not built yet; the cross test skips")
    ice = got.row("ice")
    row = M._icephase.ICE_ROW
    assert ice.id == 12 and got.nmat == 13
    assert (ice.rho, ice.cp / 1000.0, ice.k, ice.eps) == (row["rho"], row["c"], row["k"], row["emissivity"])
    assert (ice.E, ice.nu, ice.st, ice.span_ref) == (row["E_young"], row["poisson"], row["tensile"], row["span"])
    assert ice.gimp == pytest.approx(row["tough"] * 1000.0) and ice.transmit == row["transmit"]
    assert ice.extra["Lf"] == row["Lf"] and ice.mu_class == "ice"


def test_ingest_a_bond_only_row_takes_no_lattice_id():
    t = M.ingest_row({"key": "sinew", "bond_only": True, "E": 1e9, "nu": 0.4, "st": 5e7, "rho": 1100.0})
    assert t.row("sinew").id is None and t.nmat == 12


# ── arrays and the derate grid ──────────────────────────────────────────────

def test_arrays_are_float32_by_id_and_agree_with_the_rows():
    a = M.arrays()
    for name in ("rho", "E", "st", "sc", "sf", "H", "gimp", "alpha"):
        assert a[name].dtype == np.float32 and a[name].shape == (12,), name
    assert a["rho"][M.IRON] == np.float32(7800.0)
    assert a["sy"][M.WOOD] == np.inf and a["E_perp"][M.WOOD] == np.float32(5.5e8)
    assert a["sf"][M.IRON] == np.float32(M.REC["iron"].st)
    assert list(np.flatnonzero(a["structural"])) == [M.WOOD, M.STONE, M.IRON, M.GLASS, M.CHAR, M.TIN, M.LEAD]
    assert a["mu_class"][M.STONE] == M.MU_CLASSES.index("rock")


def test_derate_grid_matches_the_rows_and_the_lookup_interpolates():
    KS, KE, T = M.derate_grid()
    assert KS.shape == KE.shape == (12, 61) and KS.dtype == np.float32
    assert T[0] == -50.0 and T[-1] == 1450.0
    i = int(np.argmin(np.abs(T - 600.0)))
    assert KS[M.IRON, i] == np.float32(0.47) and KE[M.IRON, i] == np.float32(0.31)
    at20 = M.grid_lookup(KS, np.arange(12), 20.0)
    # the grid is 25 C coarse: a row whose table has a kink between grid points reads
    # a little off 1.0 at 20 C (tin and lead fall linearly from 20 C; the cold step is smeared)
    assert np.all(np.abs(at20 - 1.0) < 0.03)
    got = M.grid_lookup(KS, M.IRON, 612.5)
    assert 0.23 < float(got) < 0.47
    assert float(M.grid_lookup(KS, M.IRON, 600.0)) == pytest.approx(0.47, abs=1e-6)
    assert M.grid_lookup(KS, np.array([M.WOOD, M.STONE]), 20.0).shape == (2,)


def test_the_record_is_frozen_and_hashable_where_it_should_be():
    r = M.REC["wood"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.rho = 1.0
    assert isinstance(hash(r.derate), int)


# ── the (u) numbers and the sourced file ────────────────────────────────────
# tests/data/sourced_numbers.json (lane A-S) holds numbers read from sources. A
# record number that falls inside its sourced range is promoted from [U] to [I]
# (independent and verified); the table below says which, and the test writes
# the comparison to WORK. When the file is absent every number stays (u).

def _nums(v):
    """All numbers in a nested value (dicts, lists), unit as given."""
    if isinstance(v, bool) or v is None:
        return []
    if isinstance(v, (int, float)):
        return [float(v)]
    if isinstance(v, dict):
        return [x for e in v.values() for x in _nums(e)]
    if isinstance(v, list):
        return [x for e in v for x in _nums(e)]
    return []


def _range_pa(entry):
    """(lo, hi, status) of an entry in Pa: the value and every alternative."""
    unit = str(entry.get("unit", "")).split()[0].lower()
    scale = {"mpa": 1e6, "gpa": 1e9, "pa": 1.0}[unit]
    nums = _nums(entry["value"])
    for alt in entry.get("alt", []):
        nums += _nums(alt.get("value"))
    return min(nums) * scale, max(nums) * scale, entry["status"]


def _sourced():
    if not os.path.exists(SOURCED):
        return None
    with open(SOURCED) as fh:
        return json.load(fh)["entries"]


# record number -> (sourced entry key, how to read the record)
SOURCED_CHECKS = {
    "wood st (tension along)": ("wood.tension_parallel", lambda: M.REC["wood"].st),
    "wood sf (bending, MOR)": ("wood.bending_mor", lambda: M.REC["wood"].sf),
    "wood tau (shear)": ("wood.shear_parallel", lambda: M.REC["wood"].tau),
    "wood st_perp": ("wood.tension_perpendicular", lambda: M.REC["wood"].st_perp),
    "wood sc (compression)": ("wood.compression_parallel", lambda: M.REC["wood"].sc),
    "stone sc (granite)": ("rock.granite.compressive", lambda: M.REC["stone"].sc),
    "stone st (Brazilian; granite and limestone)": (("rock.granite.tensile", "rock.limestone.tensile"),
                                                    lambda: M.REC["stone"].st),
}


def test_u_wood_and_stone_strengths_against_the_sourced_file():
    """[U] -> [I] when the sourced file is there and the record number lies in
    its range. Writes the comparison to WORK/sourced_diff.txt and says which
    status each number had."""
    src = _sourced()
    if src is None:
        pytest.skip("status seen: tests/data/sourced_numbers.json ABSENT -> every wood and stone strength stays (u)")
    lines = []
    for name, (key, getter) in SOURCED_CHECKS.items():
        keys = (key,) if isinstance(key, str) else key
        es = [src[k] for k in keys]
        if any(e["status"] != "sourced" for e in es):
            lines.append("%s: sourced entry %s is not sourced -> stays (u)" % (name, keys))
            continue
        rngs = [_range_pa(e) for e in es]
        lo, hi = min(r[0] for r in rngs), max(r[1] for r in rngs)
        v = getter()
        mid = 0.5 * (lo + hi)
        ok = lo <= v <= hi
        lines.append("%s: record %.3g Pa, sourced range %.3g..%.3g (%s): %s, record/mid = %.2f"
                     % (name, v, lo, hi, key, "[I] inside" if ok else "[U] OUTSIDE", v / mid))
        assert ok, lines[-1]
    os.makedirs(WORK, exist_ok=True)
    with open(os.path.join(WORK, "sourced_diff.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def test_u_wood_duration_of_load_against_the_sourced_file():
    """The ten-year strength fraction the wood clock was chosen to give (0.57)
    against the sourced 0.60 ('or perhaps less'): inside 0.05."""
    src = _sourced()
    if src is None:
        pytest.skip("status seen: tests/data/sourced_numbers.json ABSENT -> the 0.57 stays (u)")
    want = src["wood.duration_of_load_10yr"]["value"]
    w = M.REC["wood"]
    ten_years = 10.0 * 365.25 * 86400.0
    frac = (w.t_R1_s / ten_years) ** (1.0 / w.creep_n)            # the utilisation that lasts ten years
    assert abs(frac - want) < 0.05
    assert frac < want                                              # the record is the more cautious side


def test_u_iron_derate_equals_en_1993_1_2_table_3_1_of_the_sourced_file():
    src = _sourced()
    if src is None:
        pytest.skip("status seen: tests/data/sourced_numbers.json ABSENT -> the iron derate stays (u)")
    e = src["steel.strength_retention_EN1993-1-2"]
    assert e["status"] == "sourced"
    v = e["value"]
    iron = M.REC["iron"]
    for T, ky, kE in zip(v["T_C"], v["k_y_effective_yield"], v["k_E_modulus"]):
        ks, ke = iron.derate_at(float(T))
        assert ks == pytest.approx(ky, abs=1e-12) and ke == pytest.approx(kE, abs=1e-12), T


def test_u_the_kingery_prefactor_is_not_sourced():
    src = _sourced()
    if src is None:
        pytest.skip("status seen: tests/data/sourced_numbers.json ABSENT")
    assert src["hasselman_kingery_quench"]["status"] == "not_found"        # the 3.25 stays (u): matstress, lane T9b


# ── blend against the real mixrules, and the hand-over door ───────────────────

_SPECIES_KEYS = [k for k, r in M.REC.items() if k in __import__("src.voxel.species", fromlist=["x"]).BY_NAME]


@pytest.mark.skipif(M._mixrules is None, reason="mixrules (T1) not built yet")
def test_blend_random_pairs_agree_with_mixrules_run_by_hand():
    """Random pairs, masses and form words: blend equals the mixrules rows run
    directly on the same entries, and a mix lies between its parts."""
    from src.voxel import species as SP, mixrules as X
    rng = np.random.default_rng(13)
    words = ("crystal", "grain", "matrix", "bound")
    for _ in range(60):
        ka, kb = rng.choice(_SPECIES_KEYS, 2, replace=False)
        w = rng.uniform(0.05, 0.95)
        fa, fb = rng.choice(words, 2)
        a, b = M.REC[ka], M.REC[kb]
        out = M.blend([(a, w, fa), (b, 1 - w, fb)])
        assert "k_source" not in out.extra
        keys = [SP.pack_key(SP.BY_NAME[ka], SP.FORM_NAMES.index(M._FORM_WORD[fa])),
                SP.pack_key(SP.BY_NAME[kb], SP.FORM_NAMES.index(M._FORM_WORD[fb]))]
        fr = [w, 1 - w]
        o = np.argsort(keys)
        ent = X.make_entries(1, [0, 0], np.array(keys, np.uint32)[o], np.array(fr)[o])
        assert out.rho == pytest.approx(float(X.RULES["rho"](ent)[0]), rel=1e-12)
        assert out.cp == pytest.approx(1000.0 * float(X.RULES["c"](ent)[0]), rel=1e-12)
        assert out.k == pytest.approx(float(X.RULES["k"](ent)[0]), rel=1e-12)
        lo, hi = sorted((a.rho, b.rho))
        assert lo - 1e-9 <= out.rho <= hi + 1e-9
        lo, hi = sorted((a.cp, b.cp))
        assert lo - 1e-9 <= out.cp <= hi + 1e-9


def test_blend_of_a_porous_record_does_not_pretend_it_is_the_species():
    wood = M.REC["wood"]
    p = M.porous(wood, 0.4)                      # same key, different rho and E
    b = M.blend([(p, 0.5, "grain"), (M.REC["stone"], 0.5, "grain")])
    assert "standin" in b.extra["k_source"]
    assert b.rho == pytest.approx(1.0 / (0.5 / p.rho + 0.5 / M.REC["stone"].rho), rel=1e-12)


def test_blend_refuses_a_form_word_nobody_knows_by_falling_back_not_guessing():
    b = M.blend([(M.REC["wood"], 0.5, "powder"), (M.REC["stone"], 0.5, "grain")])
    assert "standin" in b.extra["k_source"]


def test_handover_refuses_a_row_with_no_density_at_the_door():
    """A dict with only a key (rho = 0) used to be ingested and later blew up
    in legacy_views with ZeroDivisionError. It is refused at ingest, by name."""
    with pytest.raises(ValueError, match="rho"):
        M.ingest_row({"key": "x"})
    with pytest.raises(ValueError, match="rho"):
        M.BASE.ingest(M.MatRec(key="y", id=None, name="y", family="t", rho=0.0))


def test_handover_random_bad_numbers_are_all_refused_and_good_ones_pass():
    rng = np.random.default_rng(7)
    good = dict(ICE_STANDIN)
    for i in range(80):
        row = dict(good, key="r%d" % i)
        field = str(rng.choice(["density", "cp_Jg", "k", "E"]))
        bad = rng.choice([0.0, -1.0 - rng.random(), float("nan"), float("inf")])
        row[field] = float(bad)
        if field == "E" and bad == 0.0:
            continue                                   # E = 0 is a legal "no stiffness" row
        if field == "k" and bad == 0.0:
            continue                                   # k = 0 is legal (an insulator limit)
        assert field in good                           # the sweep really changes the row
        with pytest.raises(ValueError):
            M.ingest_row(row)
    t = M.ingest_row(good)                             # and the good row still goes in
    assert t.nmat == M.BASE.nmat + 1
    t.rows and M.legacy_views((5, 6, 7), table=t, upto=t.nmat)   # no ZeroDivisionError


# ── polish round 2: the species guard, numpy scalars, NaN masses, the identity law ──

import dataclasses as _dc


@pytest.mark.skipif(M._mixrules is None, reason="mixrules (T1) not built yet")
@pytest.mark.parametrize("field", ["rho", "cp", "k"])
def test_a_record_that_keeps_its_species_key_but_carries_other_numbers_falls_back(field):
    """The guard 'numbers equal the species' in _species_entries: the KEY test
    alone passes here (the key is still 'stone'), so only the numbers can refuse.
    Removing the guard makes the species rules answer for the wrong stuff."""
    rng = np.random.default_rng(31)
    for _ in range(20):
        scale = float(rng.choice([0.5, 0.8, 1.05, 1.3, 2.0]))
        base = M.REC["stone"]
        other = _dc.replace(base, **{field: getattr(base, field) * scale})
        assert other.key == "stone"
        b = M.blend([(other, 0.5, "grain"), (M.REC["wood"], 0.5, "grain")])
        assert "standin" in b.extra["k_source"], (field, scale)
    # and the untouched record still goes through mixrules
    b = M.blend([(M.REC["stone"], 0.5, "grain"), (M.REC["wood"], 0.5, "grain")])
    assert "k_source" not in b.extra


@pytest.mark.parametrize("make", [np.float32, np.float64, np.int64, np.int32, int, float])
def test_check_handover_accepts_any_finite_real_including_numpy_scalars(make):
    r = _dc.replace(M.REC["iron"], rho=make(7800), cp=make(450), k=make(55), E=make(200), nu=make(0), eps=make(1))
    M.check_handover(r)


@pytest.mark.parametrize("bad", [np.float32("nan"), np.float64("inf"), np.float32("-inf"), True, np.bool_(True), "7", None, 1j])
def test_check_handover_still_refuses_non_finite_and_non_real_numbers(bad):
    for f in ("rho", "cp", "k", "E", "eps", "nu"):
        r = _dc.replace(M.REC["iron"], **{f: bad})
        with pytest.raises(ValueError, match=f):
            M.check_handover(r)


def test_check_handover_numpy_negative_density_is_refused_for_the_right_reason():
    r = _dc.replace(M.REC["iron"], rho=np.float32(-1.0))
    with pytest.raises(ValueError, match="above zero"):
        M.check_handover(r)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), np.float64("nan")])
def test_blend_refuses_a_non_finite_mass_fraction(bad):
    for pos in (0, 1):
        w = [0.5, 0.5]
        w[pos] = bad
        with pytest.raises(ValueError, match="finite"):
            M.blend([(M.REC["iron"], w[0], "grain"), (M.REC["stone"], w[1], "grain")])


def test_a_blend_of_one_part_is_that_part_for_every_row_and_form():
    """IDENTITY LAW, exact: every record, every form word (also a zero-mass
    partner, which is not in the blend), rho cp k E nu and the strengths."""
    fields = ("rho", "cp", "k", "E", "nu", "sy", "st", "sc", "tau", "H", "gimp", "ductility", "alpha")
    rng = np.random.default_rng(5)
    for key, r in M.REC.items():
        if r.rho <= 0:
            continue
        for form in ("crystal", "grain", "matrix", "bound", "powder"):
            for scale in (1.0, float(rng.uniform(0.1, 9.0))):
                b = M.blend([(r, scale, form)])
                for f in fields:
                    assert getattr(b, f) == getattr(r, f), (key, form, f)
                assert b.sf_eff() == r.sf_eff()
                assert "k_source" not in b.extra
                z = M.blend([(r, scale, form), (M.REC["stone"], 0.0, "grain")])
                for f in ("rho", "cp", "k", "E"):
                    assert getattr(z, f) == getattr(r, f), (key, form, f, "zero-mass partner")


def test_a_blend_of_one_metal_in_a_bound_form_is_not_given_the_electronic_k():
    for key in ("iron", "tin", "lead"):
        r = M.REC[key]
        for form in ("matrix", "bound"):
            assert M.blend([(r, 1.0, form)]).k == r.k
