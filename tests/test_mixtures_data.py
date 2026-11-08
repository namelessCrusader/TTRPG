"""T4 mixtures, the data (wave 13, law L14): sources, species, registration, hygiene.

Tags in the test names (plan section 9): test_exact_* conservation / exactness /
arithmetic identity (gate scenes); test_strict_* an independent number from a row
that is NOT marked (u) (refuses (u) rows); test_u_* a physical number whose source
is still unverified (status read from tests/data/sourced_numbers.json when it
exists and printed); test_echo_* a fitted constant reproducing its own fit;
test_codeonly_* the code against its own table. [E] and [C] never gate a scene.
"""
import ast
import json
import os

import numpy as np
import pytest

from src.voxel import mixdata as D
from src.voxel import mixphase as P

HERE = os.path.dirname(os.path.abspath(__file__))
VOX = os.path.join(os.path.dirname(HERE), "src", "voxel")
SOURCED = os.path.join(HERE, "data", "sourced_numbers.json")
MIX_FILES = ["mixdata.py", "mixwater.py", "mixvle.py", "mixphase.py", "mixsol.py", "mixsolid.py", "mixtures.py"]


def sourced(key):
    """(value, status) from the sourced-numbers file; status says what this test saw:
    'sourced', 'not_found', 'no-key' or 'no-file'."""
    try:
        with open(SOURCED) as f:
            e = json.load(f)["entries"]
    except OSError:
        return None, "no-file"
    if key not in e:
        return None, "no-key"
    return e[key]["value"], e[key]["status"]


# -- exactness -------------------------------------------------------------------
def test_exact_every_table_row_has_a_source_mark():
    bad = [(t, k, s) for t, rows in D.TABLES.items() for k, (_, s) in rows.items() if not D.src_ok(s)]
    assert not bad, bad
    for sp in D.SPECIES.values():
        assert D.src_ok(sp.src), sp.name
    for k in D.SOURCES:
        assert isinstance(D.SOURCES[k], str) and D.SOURCES[k]
    fits = {s for rows in D.TABLES.values() for _, s in rows.values() if s.startswith("fit:")}
    assert fits and fits <= set(D.SOURCES), fits - set(D.SOURCES)


def test_exact_the_u_report_prints_rows_u_count_and_sources(capsys):
    text = D.report()
    n_u = len(D.unverified())
    assert n_u > 0
    assert f"(u) rows {n_u}" in text
    assert "sources used" in text and "fitted (echo tests only)" in text
    for t in D.TABLES:
        assert t in text


def test_exact_strict_tests_refuse_u_rows():
    with pytest.raises(AssertionError):
        D.refuse_u("solubility", "nacl")                 # a (u) row
    assert D.refuse_u("if97", "p_min_pa") == 611.657     # a sourced row passes


def test_exact_every_species_has_positive_mass_and_columns():
    assert D.ORDER[0] == "h2o"
    for n in D.ORDER:
        sp = D.SPECIES[n]
        assert sp.M > 0 and sp.name == n
        assert 0.0 <= sp.sigma <= 1.0 and sp.nucleation_ratio >= 1.0 and sp.Lv >= 0.0
    assert D.SPECIES["ethanol"].Lv == 837.0 and D.SPECIES["methanol"].Lv == 1099.0
    assert D.SPECIES["sucrose"].nucleation_ratio == 1.3 and D.SPECIES["nacl"].nucleation_ratio == 1.02
    assert D.SPECIES["ethanol"].sigma == 0.0 and D.SPECIES["nacl"].sigma == 1.0
    assert D.n_species() >= 36


def test_exact_sim_constants_and_the_pure_metals_are_READ_from_materials():
    from src.voxel import materials as m
    assert D.BOIL_SIM == m.BOIL and D.LATENT_SIM == m.LATENT
    pts = D.pure_metal_points()
    assert pts == {"sn": (m.MELT[m.TIN][0], m.MELT[m.TIN][2]),
                   "pb": (m.MELT[m.LEAD][0], m.MELT[m.LEAD][2]),
                   "fe": (m.MELT[m.IRON][0], m.MELT[m.IRON][2])}
    assert pts["sn"] == (232.0, 59.0) and pts["pb"] == (327.0, 23.0) and pts["fe"] == (1538.0, 247.0)
    # and the systems built from them carry exactly those numbers
    s = P.systems()
    assert (s["sn"].Tm, s["sn"].LAc) == pts["sn"] and (s["pb"].Tm, s["pb"].LAc) == pts["pb"]
    assert (s["fe"].Tm, s["fe"].LAc) == pts["fe"]
    assert s["pb-sn"].tl[0] == pts["pb"][0] and s["pb-sn"].tr[-1] == pts["sn"][0]
    assert s["fe-c"].tl[0] == pts["fe"][0]


def test_exact_no_craft_words_in_names():
    names = list(D.ORDER) + list(D.TABLES) + list(P.systems())
    for f in MIX_FILES:
        tree = ast.parse(open(os.path.join(VOX, f)).read())
        names += [n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    bad = [n for n in names for w in D.CRAFT_WORDS if w in n.lower()]
    assert not bad, bad


def test_exact_register_species_needs_a_src_and_accepts_a_pack_mark():
    n0 = D.n_species()
    with D.scratch_registry():
        with pytest.raises(ValueError):
            D.register_species("zz1", 100.0, "metal", aq=False)
        with pytest.raises(ValueError):
            D.register_species("zz1", 100.0, "metal", aq=False, src="made up")
        with pytest.raises(ValueError):
            D.register_species("zz1", -1.0, "metal", aq=False, src="pack:test")
        with pytest.raises(ValueError):
            D.register_species("nacl", 58.0, "salt", src="pack:test")
        sp = D.register_species("zz_metal", 100.0, "metal", aq=False, src="pack:test", rho=9.0)
        assert sp.src == "pack:test" and D.n_species() == n0 + 1 and D.INDEX["zz_metal"] == n0
    assert D.n_species() == n0 and "zz_metal" not in D.SPECIES      # the scratch registry restores


def test_exact_register_system_needs_a_src_and_registered_species():
    with D.scratch_registry():
        D.register_species("zz_metal", 100.0, "metal", aq=False, src="pack:test")
        spec = {"rule": "linear", "A": "cu", "B": "zz_metal", "points": [(0.0, 1084.6), (50.0, 900.0, 950.0)]}
        with pytest.raises(ValueError):
            D.register_system("cu-zz", spec)
        with pytest.raises(ValueError):
            D.register_system("cu-zz", dict(spec, B="nothing"), src="pack:test")
        with pytest.raises(ValueError):
            D.register_system("cu-zz", dict(spec, rule="magic"), src="pack:test")
        D.register_system("cu-zz", spec, src="pack:test")
        assert "cu-zz" in P.systems()                           # a pack's alloy is data
    assert "cu-zz" not in P.systems()


def test_exact_a_packs_species_widens_comp_arrays_without_breaking_old_ones():
    from src.voxel import mixwater as W
    old = np.zeros(D.n_species())
    old[0], old[D.INDEX["nacl"]] = 100.0, 10.0
    a0 = W.aw(old)
    with D.scratch_registry():
        D.register_species("zz_metal", 100.0, "metal", aq=False, src="pack:test")
        assert W.aw(old) == a0                                  # a narrower array is padded with zeros


def test_exact_imports_numpy_and_stdlib_only():
    std = {"collections", "contextlib", "json", "os", "math", "functools", "itertools", "dataclasses"}
    for f in MIX_FILES:
        tree = ast.parse(open(os.path.join(VOX, f)).read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods = [a.name.split(".")[0] for a in node.names]
                assert all(m == "numpy" or m in std for m in mods), (f, mods)
            if isinstance(node, ast.ImportFrom):
                if node.level == 0:
                    assert node.module.split(".")[0] in std | {"numpy"}, (f, node.module)
                else:
                    # the mix modules, and the one lazy read of materials inside mixdata
                    mod = node.module or ""
                    names = [a.name for a in node.names]
                    ok = mod.startswith("mix") or (mod == "" and all(n.startswith("mix") for n in names)) \
                        or (f == "mixdata.py" and names == ["materials"])
                    assert ok, (f, mod, names)
        # materials is read lazily (inside a function), never at the top level of any file
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.level and "materials" in [a.name for a in node.names]:
                pytest.fail(f"{f} imports materials at import time")


def test_exact_every_salt_has_an_aw_model_or_is_excluded():
    from src.voxel import mixwater as W
    for n, sp in D.SPECIES.items():
        if sp.kind == "salt":
            assert sp.aq and W.has_model(sp.model), n
    assert not [n for n in D.AW_EXCLUDED if n in D.SPECIES]    # CaCl2 and MgCl2 are not comp inputs yet


def test_exact_every_table_with_a_range_returns_nan_outside_it():
    # Fe-C above 4.3 wt% C and Cu-Sn above 25.5 wt% Sn: NaN, never a clamped number
    fec = P.fe_c_points(np.array([5.0, 4.3, 0.4]))
    assert np.isnan(fec["liquidus"][0]) and np.isnan(fec["solidus"][0]) and np.isfinite(fec["liquidus"][1])
    comp = np.zeros(D.n_species())
    comp[D.INDEX["cu"]], comp[D.INDEX["sn"]] = 70.0, 30.0
    assert np.isnan(P.solidus_liquidus(comp)[1]) and np.isnan(P.liquid_fraction(comp, 900.0))
    comp[D.INDEX["cu"]], comp[D.INDEX["sn"]] = 88.0, 12.0
    assert np.isfinite(P.solidus_liquidus(comp)[1])


def test_exact_rows_of_the_alloy_tables_are_ordered_and_in_range():
    for name, s in P.systems().items():
        if s.rule == "eutectic":
            assert np.all(np.diff(s.wl) > 0) and np.all(np.diff(s.tl) < 0), name
            if s.wr is not None:
                assert np.all(np.diff(s.wr) > 0) and np.all(np.diff(s.tr) > 0), name
            assert abs(s.wl[-1] - s.wE) < 1e-9 and abs(s.tl[-1] - s.TE) < 1e-6, name
        if s.rule == "linear":
            assert np.all(np.diff(s.w) > 0) and np.all(s.Tl >= s.Ts), name


# -- cross checks against the other lanes (skip when absent) ----------------------------------
T1_NAMES = {"h2o": "water", "nacl": "sodium_chloride", "ethanol": "ethanol", "sucrose": "sucrose",
            "cu": "copper", "sn": "tin", "pb": "lead", "fe": "iron", "c": "graphite", "caco3": "calcite",
            "sio2": "quartz", "fe3o4": "magnetite", "cao": "quicklime", "caoh2": "slaked_lime"}


def test_codeonly_cross_t1_species_agrees_on_the_shared_pure_numbers():
    sp = pytest.importorskip("src.voxel.species", reason="T1 species.py not built yet")
    seen = 0
    for mine, theirs in T1_NAMES.items():
        try:
            t1 = sp.get(theirs)
        except KeyError:
            continue
        M = t1.get("M", "ref", None)
        if M:
            assert abs(M - D.SPECIES[mine].M) / M < 0.01, (mine, M, D.SPECIES[mine].M)
            seen += 1
    assert seen >= 5
    et = sp.get("ethanol")
    Lv = et.get("Lv", "ref", None)
    if Lv:
        assert abs(Lv - D.SPECIES["ethanol"].Lv) / Lv < 0.02
    wat = sp.get("water")
    ant = wat.get("antoine", "ref", None)
    if ant is not None:
        A, B, C = D.row("antoine", "water_t1")
        assert abs(ant[0] - A) < 1e-3 and abs(ant[1] - B) < 1e-2


def test_u_ethanol_latent_heat_matches_the_sourced_nist_value(record_property):
    v, status = sourced("ethanol.latent_heat")
    record_property("sourced_status", status)
    if v is None:
        pytest.skip(f"sourced_numbers: {status}")
    assert abs(D.SPECIES["ethanol"].Lv - v["J_per_g"]) < 2.0


def test_u_nacl_solubility_table_matches_the_sourced_series(record_property):
    from src.voxel import mixsol as S
    v, status = sourced("nacl.solubility_g_per_100g_water")
    record_property("sourced_status", status)
    if v is None:
        pytest.skip(f"sourced_numbers: {status}")
    T = np.array(v["T_C"], dtype=float)
    ref = np.array(v["g_per_100g"])
    got = S.solubility_water("nacl", T)
    assert np.all(np.abs(got / ref - 1.0) < 0.006), np.c_[T, got, ref]
