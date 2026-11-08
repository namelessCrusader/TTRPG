"""Wave 13, T1: the species registry (src/voxel/species.py, species_seed.py).

Tags: [E] echo, [C] code-only (test_codeonly_*), [U] independent but the number's
source is unverified (test_u_*), [I] independent exact/closed-form/arithmetic.
Only [I] lines gate a scene. A (u) number reads tests/data/sourced_numbers.json when
it is present and the test says which status it saw.
"""
import ast
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

from src.voxel import species as SP
from src.voxel import species_seed as SEED

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src", "voxel")
SOURCED = os.path.join(HERE, "data", "sourced_numbers.json")
D, C, G, B = SP.DISSOLVED, SP.CRYSTAL, SP.GRAIN, SP.BOUND


def _sourced_status():
    """What the A-S file says, or why it says nothing. Printed by the u-tests."""
    if not os.path.exists(SOURCED):
        return "sourced_numbers.json absent: every (u) number stays (u)"
    try:
        n = len(json.load(open(SOURCED)))
    except Exception as e:                                   # pragma: no cover
        return f"sourced_numbers.json unreadable ({e})"
    return f"sourced_numbers.json present ({n} top-level entries)"


# ---- 1. the pinned id prefix -------------------------------------------------
def _pinned():
    return json.load(open(os.path.join(HERE, "data", "species_ids.json")))


def test_species_ids_json_pins_the_prefix_0_to_16():
    pinned = _pinned()
    assert [p[0] for p in pinned] == list(range(17))
    assert SP.prefix_problem(pinned) is None
    assert SP.id_prefix(17) == pinned


def test_a_reorder_fails_the_pin():
    pinned = _pinned()
    swapped = [list(p) for p in pinned]
    swapped[3], swapped[4] = [3, "flesh"], [4, "iron"]
    assert SP.prefix_problem(pinned, swapped) is not None
    renamed = [list(p) for p in pinned]
    renamed[12] = [12, "ice"]
    assert "id 12" in SP.prefix_problem(pinned, renamed)
    assert "removed" in SP.prefix_problem(pinned, pinned[:10])


def test_an_append_passes_the_pin():
    pinned = _pinned()
    assert SP.prefix_problem(pinned, pinned + [[17, "copper"], [18, "anything"]]) is None
    assert SP.prefix_problem(pinned[:5]) is None          # a shorter pin also passes


def test_species_ids_are_dense_unique_and_named():
    live = [s for s in SP.SPECIES if s is not None]
    assert [s.id for s in live] == list(range(len(live)))
    assert len({s.name for s in live}) == len(live)
    assert len(SP.SPECIES) == 64 and all(s is None for s in SP.SPECIES[len(live):])
    assert SEED.GAS_LO == 64 and SEED.GAS_HI == 96 and SEED.PACK_LO == 96
    with pytest.raises(KeyError):
        SP.get(60)
    assert SP.get("copper").id == SP.BY_NAME["copper"] == 17


# ---- 2. keys ------------------------------------------------------------------
def test_key_layout_is_species_shl_10_form_shl_8_size():
    assert SP.pack_key(1, 0, 0) == 1 << 10
    assert SP.pack_key(0, 3, 0) == 3 << 8
    assert SP.pack_key(0, 0, 255) == 255
    assert SP.pack_key(17, B, 64) == (17 << 10) | (3 << 8) | 64
    assert SP.pack_key(0xFFFF, 3, 255) == 0xFFFFFFFF >> 6          # 26 bits, fits uint32
    assert SP.unpack_key(SP.pack_key(17, B, 64)) == (17, B, 64)


def test_key_round_trips_for_every_registered_species_form_and_size():
    ids = np.array([s.id for s in SP.SPECIES if s is not None])
    sp, fo, sz = np.meshgrid(ids, np.arange(4), np.arange(256), indexing="ij")
    keys = SP.pack_key(sp.ravel(), fo.ravel(), sz.ravel())
    assert keys.dtype == np.uint32
    a, b, c = SP.unpack_key(keys)
    assert (a == sp.ravel()).all() and (b == fo.ravel()).all() and (c == sz.ravel()).all()
    assert len(np.unique(keys)) == keys.size                       # no two parts share a key


def test_key_round_trips_for_random_uint16_species_and_the_edges():
    rng = np.random.default_rng(1)
    sp = rng.integers(0, 1 << 16, 100000)
    fo, sz = rng.integers(0, 4, 100000), rng.integers(0, 256, 100000)
    a, b, c = SP.unpack_key(SP.pack_key(sp, fo, sz))
    assert (a == sp).all() and (b == fo).all() and (c == sz).all()
    for s in (0, 1, 0xFFFF):
        for f in range(4):
            for z in (0, 1, 255):
                assert SP.unpack_key(SP.pack_key(s, f, z)) == (s, f, z)
                assert int(SP.pack_key(np.array([s]), np.array([f]), np.array([z]))[0]) == \
                    SP.pack_key(s, f, z)


def test_key_parts_out_of_range_raise():
    for bad in ((1 << 16, 0, 0), (0, 4, 0), (0, 0, 256), (-1, 0, 0), (0, -1, 0)):
        with pytest.raises(ValueError):
            SP.pack_key(*bad)
    with pytest.raises(ValueError):
        SP.pack_key(np.array([1, 70000]), 0, 0)


def test_size_class_examples():
    """d = 1e-6 m * 2 ** (sz / 8): 8 = 2 um, 48 = 64 um, 64 = 0.25 mm, 100 = 5.8 mm,
    133 = 0.1 m. The formula is exact (2, 64 um); the other names are round
    numbers: 64 is 256 um (2.4 percent over 0.25 mm), 133 is 0.1011 m (1.07 percent
    over 0.1 m). Classes are 9 percent apart, so these are the nearest ones."""
    near = lambda a, b, tol: abs(a / b - 1.0) < tol
    assert near(SP.size_to_m(8), 2e-6, 1e-12)
    assert near(SP.size_to_m(48), 64e-6, 1e-12)
    assert SP.size_to_m(64) == pytest.approx(256e-6, rel=1e-12)
    assert near(SP.size_to_m(64), 0.25e-3, 0.025)
    assert near(SP.size_to_m(100), 5.8e-3, 0.01)
    assert near(SP.size_to_m(133), 0.1, 0.011)
    assert not near(SP.size_to_m(133), 0.1, 0.01)             # (a deviation from "1 percent")
    assert SP.size_to_m(0) == 0.0
    assert near(SP.size_to_m(16) / SP.size_to_m(8), 2.0, 1e-12)   # eight classes = 2x


def test_size_class_inverse_and_monotone():
    sz = np.arange(1, 256)
    d = SP.size_to_m(sz)
    assert (np.diff(d) > 0).all()
    assert (SP.m_to_size(d) == sz).all()
    assert SP.m_to_size(1e-9) == 1 and SP.m_to_size(1e9) == 255     # clipped
    assert SP.m_to_size(2.05e-6) == 8
    assert np.allclose(d[1:] / d[:-1], 2 ** (1 / 8))


# ---- 3. forms, states, slots --------------------------------------------------
def test_forms_are_the_four_and_gas_is_not_one():
    assert (D, C, G, B) == (0, 1, 2, 3) and SP.FORM_NAMES == ("dissolved", "crystal",
                                                            "grain", "bound")
    assert SP.NFORMS == 4


def test_seed_mats_and_registry_states_agree_both_ways():
    for m, slot in enumerate(SP.MATS):
        assert slot.id == m
        if slot.species is None or m == 0:
            continue
        sp = SP.get(slot.species)
        assert sp.solid_states.get(slot.form) == m, (m, sp.name)
        if slot.form == G:
            assert sp.grain_sz == slot.sz
    for sp in (s for s in SP.SPECIES if s is not None):
        for form, m in sp.solid_states.items():
            assert SP.MATS[m].species == sp.id and SP.MATS[m].form == form
        if sp.liquid is not None:
            assert SP.FLUIDS[sp.liquid].species == sp.id
    assert len(SP.MATS) == 64 and len(SP.FLUIDS) == 64


def test_ice_is_water_bound_with_the_reserved_id_and_the_empty_skeleton():
    w = SP.get("water")
    assert SP.MAT_ICE == 12 and w.solid_states == {B: 12} and w.liquid == 1
    assert SP.MATS[12].species == w.id and SP.MATS[12].form == B
    assert w.skeleton == frozenset()               # bound water never wins the mat view
    assert B in SP.get("wood").skeleton and G in SP.get("quartz").skeleton


def test_quartz_rock_and_sand_are_different_mats_and_nacl_is_not_a_fluid_of_its_own():
    q = SP.get("quartz")
    assert q.solid_states[B] != q.solid_states[G]
    assert SP.get("sodium_chloride").liquid is None and SP.get("sodium_chloride").fluid_like == 1
    assert SP.get("ethanol").liquid == 9 and SP.get("copper").liquid == 10
    assert SP.get("iron").liquid == 7 and SP.get("tin").liquid == 5 and SP.get("lead").liquid == 6


def test_every_dissolvable_species_has_a_fluid_view_and_it_is_a_legacy_fluid():
    for sp in (s for s in SP.SPECIES if s is not None):
        dissolves = any(sp.has(k, "ref") for k in ("phiV0", "k_app", "mu_rel", "c_app"))
        if dissolves:
            assert sp.liquid is not None or sp.fluid_like is not None, sp.name
        if sp.fluid_like is not None:
            assert 0 < sp.fluid_like < 9, sp.name         # a LEGACY fluid row stands in
        assert sp.like in range(12), sp.name


def test_composites_are_the_placeholders_and_the_pure_ones_are_not():
    comp = {s.name for s in SP.SPECIES if s is not None and s.composite}
    for n in ("wood", "stone", "flesh", "ash", "leaf", "char", "rope", "blood", "lamp_oil"):
        assert n in comp, n
    for n in ("iron", "tin", "lead", "water", "copper", "sodium_chloride", "ethanol"):
        assert n not in comp, n


# ---- 4. legacy species read materials.py (code-only) -------------------------
def test_codeonly_legacy_engine_numbers_are_read_from_materials():
    from src.voxel import materials as M
    for m in range(1, 12):
        sp = SP.get(m)
        assert sp.engine["rho_s"] == M.SOLID[m][0] and sp.engine["c_s"] == M.SOLID[m][1]
        assert sp.engine["k_s"] == M.SOLID[m][2] and sp.engine["emiss_s"] == M.EMISS[m]
        assert sp.engine.get("ign") == M.SOLID[m][3] or (M.SOLID[m][3] is None
                                                         and "ign" not in sp.engine)
    for sid, fl in ((12, M.WATER), (13, M.OIL), (14, M.ACID), (15, M.WEAK_ACID), (16, M.BLOOD),
                    (3, M.MIRON), (9, M.MTIN), (10, M.MLEAD)):
        e = SP.get(sid).engine
        assert e["rho_l"] == M.FLUID[fl][0] * 1000.0 and e["c_l"] == M.FLUID[fl][1]
        assert e["h_wet"] == M.FLUID[fl][6] and e["k_l"] == M.FLUID_K[fl]
    for m, (tm, fl, lat) in M.MELT.items():
        assert (SP.get(m).engine["Tm"], SP.get(m).engine["Lf"]) == (tm, lat)
        assert SP.get(m).liquid == fl
    assert SP.get("water").engine["Tb"] == M.BOIL and SP.get("water").engine["Lv"] == M.LATENT
    assert SP.get("sulfuric_acid").engine["stock"] == M.STOCK[M.ACID]
    assert SP.get("acetic_acid").engine["stock"] == M.STOCK[M.WEAK_ACID]


def test_the_engine_slot_of_legacy_species_is_tagged_with_the_repo_file():
    for sid in range(1, 17):
        sp = SP.get(sid)
        for k in sp.engine:
            assert sp.src[k] == "repo:materials.py", (sp.name, k)


# ---- 5. rows ------------------------------------------------------------------
def test_no_row_number_is_used_twice_across_mats_fluids_and_mixes():
    rows = SP.pure_rows()
    assert sorted(rows) == list(range(128))
    mat_rows = {m: m for m in range(64)}
    fluid_rows = {f: 64 + f for f in range(64)}
    assert not set(mat_rows.values()) & set(fluid_rows.values())
    assert max(fluid_rows.values()) < SP.MIX_BASE == 128
    assert SEED.SOLID_ROWS == (0, 64) and SEED.FLUID_ROWS == (64, 128)
    with pytest.raises(ValueError):
        SP.pure_row(128)
    with pytest.raises(ValueError):
        SP.pure_row(-1)


def test_every_pure_row_is_a_valid_composition():
    for r, comp in SP.pure_rows().items():
        if not comp:                                   # air, no fluid, or a pad of them
            assert r in (0, 64) or (SP.MATS[r].species is None if r < 64
                                    else SP.FLUIDS[r - 64].species is None)
            continue
        assert abs(sum(e[3] for e in comp) - 1.0) < 1e-12, r
        assert len({(e[0], e[1], e[2]) for e in comp}) == len(comp)
        for sid, form, sz, frac in comp:
            assert SP.SPECIES[sid] is not None and 0 <= form < 4 and 0 <= sz < 256
            assert frac > 0
            assert (sz > 0) == (form == G), r          # only grains carry a size here


def test_ACID_and_WEAK_ACID_pure_rows_are_the_STOCK_composition():
    acid = SP.pure_row(64 + 3)
    weak = SP.pure_row(64 + 4)
    assert [(e[0], e[1]) for e in acid] == [(SP.BY_NAME["sulfuric_acid"], D),
                                           (SP.BY_NAME["water"], D)]
    assert acid[0][3] == 0.983 and abs(acid[1][3] - 0.017) < 1e-15
    assert [(e[0], e[1]) for e in weak] == [(SP.BY_NAME["acetic_acid"], D),
                                           (SP.BY_NAME["water"], D)]
    assert weak[0][3] == 0.05 and abs(weak[1][3] - 0.95) < 1e-15
    # the other legacy fluids are single-species rows
    assert SP.pure_row(64 + 1) == ((12, D, 0, 1.0),)
    assert SP.pure_row(64 + 7) == ((3, D, 0, 1.0),)         # molten iron is iron DISSOLVED


def test_pure_solid_rows_are_species_bound_or_grain_and_pads_copy_their_like():
    assert SP.pure_row(0) == () and SP.pure_row(3) == ((3, B, 0, 1.0),)
    assert SP.pure_row(12) == ((12, B, 0, 1.0),)             # ICE = water BOUND
    assert SP.pure_row(15) == ((21, G, 64, 1.0),)            # sand
    assert SP.pure_row(14) == ((21, B, 0, 1.0),)             # quartz rock
    assert SP.pure_row(50) == SP.pure_row(0)                 # an unlisted pad copies AIR
    assert SP.pure_row(64 + 50) == SP.pure_row(64)


# ---- 6. seed data: every number has a tag or is (u) --------------------------
def _numeric_leaves(v):
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return True
    if isinstance(v, (tuple, list)) and v:
        return all(_numeric_leaves(x) for x in v)
    return False


def test_every_numeric_seed_field_has_a_source_tag_or_is_in_the_unverified_set():
    n_src = n_u = 0
    for sp in (s for s in SP.SPECIES if s is not None):
        for slot in (sp.ref, sp.engine):
            for k, v in slot.items():
                assert _numeric_leaves(v), (sp.name, k, v)          # no stray strings
                tagged, unv = k in sp.src, k in sp.unverified
                assert tagged != unv, (sp.name, k, "tagged and unverified, or neither")
                if tagged:
                    assert sp.src[k] in SEED.SOURCES, (sp.name, k, sp.src[k])
                    n_src += 1
                else:
                    n_u += 1
        for k in sp.src:
            assert k in sp.ref or k in sp.engine, (sp.name, k, "tag on a missing field")
        assert set(sp.unverified) <= set(sp.ref) | set(sp.engine)
    assert n_u > 100 and n_src >= 4
    print(f"tags: {n_src} sourced or computed numeric fields, {n_u} unverified (u); "
          + _sourced_status())


def test_the_seed_source_tags_name_documents_the_repo_holds():
    docs = os.path.join(ROOT, "docs", "realism-references-2.md")
    if os.path.exists(docs):
        assert "R12" in open(docs).read()
    assert set(SEED.SOURCES) >= {"repo:realism-references-2:R12", "repo:materials.py",
                                 "computed"}
    nacl = SP.get("sodium_chloride")
    assert nacl.src["dH_sol"] == nacl.src["solubility"] == "repo:realism-references-2:R12"
    assert nacl.ref["dH_sol"] == 66.0                       # +3.87 kJ/mol = +66 J/g (R12)
    assert nacl.ref["solubility"] == ((25.0, 36.0), (50.0, 36.8), (80.0, 38.1))


def test_computed_stoichiometry_is_arithmetic_on_the_row():
    """[I] o2_ratio = (mol O2 per mol fuel) x 31.998 / M, from the formula."""
    O2 = 31.998
    assert abs(SP.get("ethanol").ref["o2_ratio"] - 3 * O2 / 46.07) < 5e-4
    assert abs(SP.get("graphite").ref["o2_ratio"] - O2 / 12.011) < 5e-4
    assert abs(SP.get("tallow").ref["o2_ratio"] - 80 * O2 / 885.4) < 5e-4


def test_the_water_set_is_one_set():
    """[I] one Tm and one Lf for water in the whole registry; ice is water BOUND."""
    w = SP.get("water")
    assert (w.ref["Tm"], w.ref["Lf"]) == (0.0, 333.55)
    assert not [s.name for s in SP.SPECIES if s is not None and s.name != "water"
                and s.ref.get("Lf") == 333.55]
    assert SP.get("sodium_chloride").ref["phiV0"] == 16.62 and \
        SP.get("sodium_chloride").ref["Sv"] == 1.87         # Masson, the one set


# ---- 7. sanity ranges (u) -----------------------------------------------------
def _in(x, lo, hi, what):
    assert lo <= x <= hi, f"{what}: {x} not in [{lo}, {hi}]"


def test_u_sanity_ranges_for_the_metals():
    """[U] ranges from handbook memory; the sourced file (A-S) upgrades them."""
    print(_sourced_status())
    r = lambda n: SP.get(n).ref
    _in(r("copper")["rho_s"], 8900, 9000, "Cu rho"); _in(r("copper")["c_s"], 0.38, 0.39, "Cu c")
    _in(r("copper")["k_s"], 380, 410, "Cu k"); _in(r("copper")["Tm"], 1084, 1086, "Cu Tm")
    _in(r("copper")["rho_e"], 1.6e-8, 1.75e-8, "Cu rho_e")
    _in(r("tin")["rho_s"], 7280, 7320, "Sn rho"); _in(r("tin")["Tm"], 231, 233, "Sn Tm")
    _in(r("tin")["k_s"], 60, 70, "Sn k"); _in(r("tin")["Lf"], 55, 62, "Sn Lf")
    _in(r("lead")["rho_s"], 11300, 11400, "Pb rho"); _in(r("lead")["Tm"], 327, 328, "Pb Tm")
    _in(r("lead")["c_s"], 0.125, 0.135, "Pb c"); _in(r("lead")["k_s"], 33, 37, "Pb k")
    _in(r("iron")["rho_s"], 7800, 7900, "Fe rho"); _in(r("iron")["c_s"], 0.44, 0.46, "Fe c")
    _in(r("iron")["Tm"], 1530, 1545, "Fe Tm"); _in(r("iron")["Tc"], 765, 775, "Fe Tc")
    _in(r("iron")["sigma_s"], 215, 220, "Fe sigma_s")
    # engine copies of the same metals stay near the real ones
    _in(SP.get("tin").engine["rho_s"], 7000, 7500, "Sn engine rho")
    # the metals melt in the real order tin < lead < copper < iron
    assert r("tin")["Tm"] < r("lead")["Tm"] < r("copper")["Tm"] < r("iron")["Tm"]


def test_u_sanity_ranges_for_water_salt_and_ethanol():
    w, s, e = (SP.get(n).ref for n in ("water", "sodium_chloride", "ethanol"))
    assert w["rho_l"] == 998.0 and w["c_l"] == 4.18 and w["Tb"] == 100.0 and w["Lv"] == 2257.0
    _in(w["M"], 18.0, 18.03, "water M"); _in(w["k_l"], 0.55, 0.65, "water k")
    assert s["M"] == 58.44 and s["Tm"] == 801.0
    _in(s["rho_s"], 2100, 2200, "NaCl rho"); _in(s["c_s"], 0.8, 0.9, "NaCl c")
    _in(e["M"], 46.0, 46.1, "ethanol M"); _in(e["rho_l"], 780, 800, "ethanol rho")
    _in(e["c_l"], 2.3, 2.5, "ethanol c"); _in(e["Tb"], 78, 78.6, "ethanol Tb")
    _in(e["Tm"], -115, -113, "ethanol Tm"); _in(e["Lv"], 830, 850, "ethanol Lv")
    _in(e["hcomb"], 29000, 30000, "ethanol hcomb")
    # the engine water is the legacy one, a few percent from the real one
    ew = SP.get("water").engine
    _in(ew["rho_l"] / w["rho_l"], 0.99, 1.01, "engine/ref water rho")
    _in(ew["c_l"] / w["c_l"], 0.99, 1.02, "engine/ref water c")
    assert ew["Lv"] == 2260.0 and abs(ew["Lv"] / w["Lv"] - 1) < 0.005


def test_u_other_seed_rows_are_in_plausible_ranges():
    for n, (lo, hi) in {"graphite": (2100, 2300), "quartz": (2600, 2700),
                        "calcite": (2650, 2750), "kaolinite": (2500, 2700),
                        "iron_oxide": (5000, 5300), "magnetite": (5100, 5250),
                        "quicklime": (3300, 3400), "slaked_lime": (2200, 2250),
                        "sucrose": (1550, 1620), "tallow": (850, 950)}.items():
        _in(SP.get(n).ref["rho_s"], lo, hi, n + " rho")
    _in(SP.get("magnetite").ref["sigma_s"], 85, 95, "magnetite sigma_s")
    _in(SP.get("magnetite").ref["Tc"], 580, 590, "magnetite Tc")
    _in(SP.get("tallow").ref["Tm"], 35, 50, "tallow Tm")
    # relative viscosity tables rise with concentration and start at 1
    for n in ("sodium_chloride", "sucrose"):
        t = SP.get(n).ref["mu_rel"]
        assert t[0] == (0.0, 1.0) and all(a[0] < b[0] and a[1] < b[1] for a, b in zip(t, t[1:]))
    # (the NaCl table is now SOURCED and is gated by test_nacl_relative_viscosity_*)


def test_antoine_rows_reproduce_the_normal_boiling_points():
    """[I] arithmetic on the row: water 760 mmHg at 100 C, ethanol 760 at 78.37 C."""
    for n, T, want in (("water", 100.0, 760.0), ("ethanol", 78.37, 760.0)):
        A, Bc, Cc, tmin, tmax = SP.get(n).ref["antoine"]
        p = 10 ** (A - Bc / (Cc + T))
        assert abs(p / want - 1) < 0.005, (n, p)
        assert tmin < T <= tmax + 0.5          # the boiling point is at the edge of the fit


def test_andrade_rows_reproduce_the_two_viscosity_points():
    """[I] water 1.002 mPa s at 20 C and 0.547 at 50 C fit B = 1911 K (computed)."""
    w = SP.get("water").ref
    mu = lambda T: w["mu_A"] * math.exp(w["mu_B"] / (T + 273.15))
    assert abs(mu(20) / 1.002 - 1) < 0.003 and abs(mu(50) / 0.547 - 1) < 0.003
    assert abs(w["mu_B"] - 1911) < 1.0
    e = SP.get("ethanol").ref
    mue = lambda T: e["mu_A"] * math.exp(e["mu_B"] / (T + 273.15))
    assert abs(mue(20) / 1.2 - 1) < 0.01 and mue(40) < mue(20)


# ---- 8. hygiene ---------------------------------------------------------------
def _imports(path):
    tree = ast.parse(open(path).read())
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out += [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            out.append(("." * n.level) + (n.module or ""))
    return out


def test_species_seed_has_no_imports_at_all():
    path = os.path.join(SRC, "species_seed.py")
    assert _imports(path) == []
    src = open(path).read()
    assert "__import__" not in src and "importlib" not in src


def test_species_and_legacy_tables_import_no_world_and_no_mix_table():
    for f in ("species.py", "legacy_tables.py", "species_seed.py"):
        for name in _imports(os.path.join(SRC, f)):
            low = name.lower()
            for banned in ("sim", "world", "mixtable", "mixrules", "partition", "fields"):
                assert banned not in low, (f, name)
    assert sorted(_imports(os.path.join(SRC, "species.py"))) == sorted(
        ["dataclasses", "numpy", ".", ".", ".legacy_tables", ".species_seed"])


def test_importing_species_does_not_import_mixtable_or_the_world():
    code = ("import sys; sys.path.insert(0, %r); import src.voxel.species; "
            "bad = [m for m in sys.modules if m.startswith('src.voxel.') and m.split('.')[-1] "
            "in ('mixtable', 'mixrules', 'partition', 'sim', 'fields', 'heat', 'fluids')]; "
            "print('BAD', bad) if bad else print('CLEAN')" % ROOT)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=ROOT, timeout=60)
    assert out.stdout.strip().endswith("CLEAN"), out.stdout + out.stderr


def test_species_data_is_deterministic_and_frozen():
    a, b = SP._build(), SP._build()
    assert [(s.id, s.name, s.ref, s.engine) if s else None for s in a] == \
        [(s.id, s.name, s.ref, s.engine) if s else None for s in b]
    with pytest.raises(Exception):
        SP.get("copper").id = 99                        # frozen dataclass


# ---- 8. polish P-T1: three seed numbers, each against its source ------------
def _sourced(key):
    if not os.path.exists(SOURCED):
        pytest.skip("tests/data/sourced_numbers.json absent")
    return json.load(open(SOURCED))["entries"][key]["value"]


def test_nacl_relative_viscosity_table_is_the_kestin_1981_table():
    """[I] every node of the seed table is the SOURCED relative viscosity at 20 C
    (Kestin, Khalifa, Correia 1981), not the old spec (u) table (1.99 at 20 wt percent)."""
    kestin = _sourced("nacl.rel_viscosity_20C")
    got = dict(SP.get("sodium_chloride").ref["mu_rel"])
    assert got[0.0] == 1.0 and len(got) == len(kestin) + 1
    for pct, row in kestin.items():
        assert got[int(pct) / 100.0] == pytest.approx(row["relative"], rel=1e-9), pct
    assert got[0.20] != 1.99 and got[0.20] == pytest.approx(1.562, rel=1e-9)
    assert SP.get("sodium_chloride").src["mu_rel"] == "kestin1981"
    assert "mu_rel" not in SP.get("sodium_chloride").unverified
    assert "kestin1981" in SEED.SOURCES


def test_brine_viscosity_through_the_mix_rule_matches_kestin_at_every_node_and_between():
    """[I] the fix is in the seed, so it shows in the mix rule: at every sourced node
    the brine is mu_water x Kestin to 1e-6, and between nodes (random seeds) it stays
    inside its two neighbours and rises with salt."""
    from src.voxel import mixrules as MR
    from src.voxel.species import pack_key
    W, N = SP.BY_NAME["water"], SP.BY_NAME["sodium_chloride"]

    def rel(w):
        keys = np.array(sorted([pack_key(W, D), pack_key(N, D)]), np.uint32)
        fr = np.array([1.0 - w if k == pack_key(W, D) else w for k in keys])
        e = MR.make_entries(1, np.zeros(2, np.int64), keys, fr)
        return MR.mu_mix(e, "ref", T_C=20.0)[0] / MR.andrade(W, 20.0, "ref")
    kestin = _sourced("nacl.rel_viscosity_20C")
    nodes = [(0.0, 1.0)] + [(int(p) / 100.0, r["relative"]) for p, r in kestin.items()]
    for w, r in nodes:
        assert rel(w) == pytest.approx(r, rel=1e-6), w
    rng = np.random.default_rng(1308)
    ws = np.sort(rng.uniform(0.0, 0.25, 40))
    vals = [rel(w) for w in ws]
    assert all(a < b for a, b in zip(vals, vals[1:]))
    for w, v in zip(ws, vals):
        lo = max(n for n in nodes if n[0] <= w)
        hi = min(n for n in nodes if n[0] >= w)
        assert lo[1] - 1e-9 <= v <= hi[1] + 1e-9, (w, v)


def _cho_counts(formula):
    """C, H, O atom counts of a plain formula like C6H10O5 or (C6H10O5)n (the unit)."""
    import re
    f = formula.replace("(", "").replace(")n", "")
    if not f or re.search(r"[^CHO0-9]", f):
        return None
    n = {"C": 0, "H": 0, "O": 0}
    for el, k in re.findall(r"([CHO])(\d*)", f):
        n[el] += int(k or 1)
    return n


def test_o2_ratio_of_every_chon_species_is_its_stoichiometry():
    """[I] sweep over every seed species with a plain C/H/O formula and an o2_ratio:
    o2_ratio = (C + H/4 - O/2) x 31.998 / M, M from atomic weights, to 0.5 percent.
    Cellulose is 6 x 31.998 / 162.14 = 1.184 (the old 1.26 was the rope row's heat ratio)."""
    AW = {"C": 12.011, "H": 1.008, "O": 15.999}
    seen = []
    for s in SP.SPECIES:
        if s is None or "o2_ratio" not in s.ref:
            continue
        n = _cho_counts(s.formula)
        if n is None:
            continue
        M = sum(AW[e] * k for e, k in n.items())
        want = (n["C"] + n["H"] / 4.0 - n["O"] / 2.0) * 31.998 / M
        assert s.ref["o2_ratio"] == pytest.approx(want, rel=5e-3), s.name
        assert s.ref["M"] == pytest.approx(M, rel=1e-3), s.name
        seen.append(s.name)
    assert {"ethanol", "graphite", "cellulose"} <= set(seen)
    assert SP.get("cellulose").ref["o2_ratio"] == pytest.approx(1.184, abs=1e-3)
    assert SP.get("cellulose").src["o2_ratio"] == "computed"


def test_sand_is_a_packed_bed_particle_density_and_bulk_density_are_both_stored():
    """[I] quartz keeps particle density 2650 (rock, species, settling); the GRAIN form is
    a packed bed whose porosity is inside the sourced 26 to 50 percent sand range, so a
    cell of sand is 2650 x (1 - por). BOUND quartz (rock) has no pores."""
    q = SP.get("quartz")
    assert q.ref["rho_s"] == 2650.0
    por = q.ref["por_grain"]
    assert 0.30 <= por <= 0.45                      # coarse-sand range, GW-Project Table 1
    assert q.src["por_grain"] == "gwproject:porosity-range" and "por_grain" not in q.unverified
    assert SP.bulk_density("quartz", G) == pytest.approx(2650.0 * (1.0 - por))
    assert 1430.0 <= SP.bulk_density("quartz", G) <= 1800.0   # loose (1.43) to dense (1.8) sand
    for form in (B, C, D):
        assert SP.bulk_density("quartz", form) == 2650.0 and SP.form_porosity("quartz", form) == 0.0


def test_bulk_density_sweep_never_exceeds_particle_density_and_only_packs_grain():
    """[I] over every species and every form (and both slots): bulk <= particle density,
    equal unless the form is GRAIN of a species with por_grain; porosity in [0, 1)."""
    n_packed = 0
    for s in SP.SPECIES:
        if s is None:
            continue
        for slot in ("engine", "ref"):
            if s.get("rho_s", slot) is None:
                assert SP.bulk_density(s.id, G, slot) is None
                continue
            for form in (D, C, G, B):
                b, p = SP.bulk_density(s.id, form, slot), SP.form_porosity(s.id, form, slot)
                assert 0.0 <= p < 1.0 and b <= s.get("rho_s", slot) + 1e-9
                if form != G or "por_grain" not in s.ref:
                    assert b == s.get("rho_s", slot) and p == 0.0
                else:
                    n_packed += 1
                    assert b == pytest.approx(s.get("rho_s", slot) * (1.0 - s.ref["por_grain"]))
    assert n_packed >= 2                            # quartz, both slots


# ---- the packed-bed conductivity and the hydrated-lime bulk density -----------------------------
def _maxwell(k_matrix, k_incl, f_incl):
    return k_matrix * (2 * k_matrix + k_incl - 2 * f_incl * (k_matrix - k_incl)) / \
        (2 * k_matrix + k_incl + f_incl * (k_matrix - k_incl))


def test_i_a_dry_sand_bed_conducts_like_a_bed_not_like_solid_quartz():
    """[I] 0.25 to 0.35 W/m/K for a dry quartz sand bed of porosity 0.40 (Woodside and
    Messmer 1961 give about 0.3 for dry sand; (u) as a recalled range), not the 7.7 of
    solid quartz. The rock (BOUND) keeps 7.7. Packed_k is Zehner-Schlunder; its value
    here is 0.312 (hand-computed from the formula: B = 1.25 x 1.5^(10/9) = 1.9632)."""
    k = SP.bulk_conductivity("quartz", G)
    assert 0.25 <= k <= 0.35
    assert k == pytest.approx(0.3123, rel=1e-3)
    for form in (B, C, D):
        assert SP.bulk_conductivity("quartz", form) == 7.7
    assert SP.get("quartz").get("k_s") == 7.7                 # the solid itself is untouched
    rows = {r["mat"]: r for r in SP.seed_mat_rows()}
    assert rows[15]["k"] == pytest.approx(k, rel=1e-12) and rows[14]["k"] == 7.7
    # at the old number a bed would have a thermal diffusivity 25 times too large
    assert 7.7 / k > 20


def test_i_packed_k_limits_and_the_lower_bound_of_a_bed():
    """[I] e = 0 is the solid, e = 1 is the gas, a solid equal to the gas gives the gas
    (also through the removable point B kappa = 1, where the formula is 0/0), and the
    bed is above the Maxwell gas-matrix (Hashin-Shtrikman lower) bound at e = 0.40."""
    assert SP.packed_k(7.7, 0.0263, 0.0) == 7.7 and SP.packed_k(7.7, 0.0263, 1.0) == 0.0263
    for kg in (0.0263, 1.0, 40.0):
        assert SP.packed_k(kg, kg, 0.4) == pytest.approx(kg, rel=1e-6)
    e = 0.40
    B_ = 1.25 * ((1 - e) / e) ** (10.0 / 9.0)
    kg = 0.0263
    k_at = SP.packed_k(B_ * kg, kg, e)                           # B kappa = 1 exactly
    # the limit of the formula, from 60-digit arithmetic at d = 1e-15 (mpmath)
    assert k_at == pytest.approx(0.03935707376090505, rel=1e-7)
    # and the neighbours on both sides agree with it to first order in the step
    up, dn = (SP.packed_k(B_ * kg * (1 + d), kg, e) for d in (2e-3, -2e-3))
    assert dn < k_at < up and (up + dn) / 2 == pytest.approx(k_at, rel=1e-5)
    assert SP.packed_k(7.7, kg, e) > _maxwell(kg, 7.7, 1 - e)
    with pytest.raises(ValueError):
        SP.packed_k(0.0, kg, e)
    with pytest.raises(ValueError):
        SP.packed_k(7.7, kg, 1.5)


@pytest.mark.parametrize("seed", range(8))
def test_i_packed_k_sweep_stays_between_the_hashin_shtrikman_bounds_and_is_monotone(seed):
    """[I] 50000 random beds per seed (solid 10 to 1000 times the gas, e 0.25 to 0.6): no
    NaN; between the Maxwell gas-matrix (lower) and solid-matrix (upper) bounds; falling
    with porosity and rising with the solid's k. A row that left the bounds would make a
    bed conduct like a rock or like still air."""
    rng = np.random.default_rng(7000 + seed)
    n = 50000
    kg = 10 ** rng.uniform(-2, -0.5, n)
    ks = kg * 10 ** rng.uniform(1.0, 3.0, n)
    e = rng.uniform(0.25, 0.6, n)
    k = SP.packed_k(ks, kg, e)
    assert np.isfinite(k).all()
    assert (k >= _maxwell(kg, ks, 1 - e) * (1 - 1e-9)).all()
    assert (k <= _maxwell(ks, kg, e) * (1 + 1e-9)).all()
    assert (SP.packed_k(ks, kg, e + 0.01) < k).all()
    assert (SP.packed_k(ks * 1.1, kg, e) > k).all()


def test_i_hydrated_lime_powder_has_a_sourced_bulk_density():
    """[I] slaked-lime powder (mat 25, GRAIN) is a heap of bulk density 350 to 560 kg/m3
    (loose to packed, supplier data sheets cited in species_seed), not the 2211 of the
    particle; the porosity is a sourced-tagged field, the cell table reads the bulk
    density, and the particle density is kept for the laws about one grain."""
    sl = SP.get("slaked_lime")
    assert sl.ref["rho_s"] == 2211.0
    assert "por_grain" in sl.src and "por_grain" not in sl.unverified
    assert "hydrated lime" in SP.seed.SOURCES[sl.src["por_grain"]]
    bulk = SP.bulk_density("slaked_lime", G)
    assert 350.0 <= bulk <= 560.0 and bulk == pytest.approx(450.0)
    assert SP.bulk_density("slaked_lime", B) == 2211.0
    rows = {r["mat"]: r for r in SP.seed_mat_rows()}
    assert rows[25]["rho"] == pytest.approx(bulk)
    # a powder of 80 percent air conducts far less than the solid hydroxide
    assert 0.02 < rows[25]["k"] < 0.2 and rows[25]["k"] < 0.1 * sl.get("k_s")
