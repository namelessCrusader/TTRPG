"""Source check of the numbers that decide results (wave 13, lane A-S).

tests/data/sourced_numbers.json holds, for each number the plan named, a value
with its citation, or the status "not_found". tools/w13_sourced.py is the
loader. These tests check three things:

  * form: every listed key is there; a sourced entry has a value and a URL; a
    not_found entry has no value; every alternative source carries a URL;
  * the loader: get() returns exactly what is stored, an unknown key raises
    KeyError, the module imports no network library and opens no socket;
  * arithmetic identities inside the data (acceptance tag [I] for closed-form
    lines): conversions and formulas the entries state about themselves.

Tags (plan section 9): test_codeonly_* test the code or the file against
itself; test_arith_* are closed-form identities. No test here asserts a
physical number against a constant of this repo: the repo's own (u) numbers are
compared with these in the entries' `vs_spec` text, for the owners to retune.
"""
import ast
import importlib.util
import json
import math
import os
import re
import socket

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "w13_sourced.py")
DATA = os.path.join(ROOT, "tests", "data", "sourced_numbers.json")

_spec = importlib.util.spec_from_file_location("w13_sourced_tool", TOOL)
W = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(W)

with open(DATA) as _f:
    RAW = json.load(_f)
ENT = RAW["entries"]

# One group per line of the task list: the key that answers it.
REQUIRED = [
    "wood.tension_parallel", "wood.bending_mor", "wood.shear_parallel", "wood.tension_perpendicular",
    "rock.granite.compressive", "rock.granite.tensile", "rock.limestone.compressive", "rock.limestone.tensile",
    "cusn.resistivity_vs_tin", "cusn.nordheim_coefficient",
    "nacl.k_app", "nacl.rel_viscosity_20C", "water.viscosity_andrade",
    "nacl.solubility_g_per_100g_water", "nacl.freezing_curve_C", "nacl.eutectic",
    "janaf.boudouard", "janaf.wuestite", "janaf.calcite",
    "ethanol.freezing_curve_C", "ethanol.latent_heat",
    "richardson_zaki", "hasselman_kingery_quench", "steel.strength_retention_EN1993-1-2",
    "wood.duration_of_load_10yr", "wood.screw_withdrawal", "wood.dowel_embedment",
    "body.kick", "body.stair_power",
    "bone.femur_fracture_load", "bone.tibia_fracture_load", "bone.vertebra_fracture_load", "bone.calcaneus_fracture_load",
    "burn.moritz_henriques", "bleeding.femoral_flow", "jos3.package",
]


def test_codeonly_every_listed_key_is_present():
    assert [k for k in REQUIRED if k not in ENT] == []


def test_codeonly_status_is_sourced_or_not_found_and_value_matches():
    for k, e in ENT.items():
        assert e["status"] in ("sourced", "not_found"), k
        if e["status"] == "sourced":
            assert e["value"] is not None, k
        else:
            assert e["value"] is None, k


def test_codeonly_no_entry_lacks_a_url_when_sourced():
    for k, e in ENT.items():
        assert isinstance(e["citation"], str) and e["citation"].strip(), k
        if e["status"] == "sourced":
            assert re.search(r"https?://", e["citation"]), k


def test_codeonly_not_found_entries_say_where_they_looked_and_carry_a_url_or_search_note():
    for k, e in ENT.items():
        if e["status"] == "not_found":
            assert len(e["citation"]) > 60, k
            assert re.search(r"https?://", e["citation"]) or "searched" in e["citation"].lower(), k


def test_codeonly_every_alternative_source_has_a_url():
    n = 0
    for k, e in ENT.items():
        for a in e.get("alt", []):
            n += 1
            assert re.search(r"https?://", a["citation"]), (k, a)
    assert n >= 8


def test_codeonly_retrieval_date_is_iso_on_every_entry():
    for k, e in ENT.items():
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", e["retrieved"]), k
    assert RAW["_meta"]["count_sourced"] + RAW["_meta"]["count_not_found"] == len(ENT)
    assert RAW["_meta"]["count_sourced"] == sum(e["status"] == "sourced" for e in ENT.values())


def test_codeonly_get_returns_the_stored_tuple():
    for k, e in ENT.items():
        v, st, c = W.get(k)
        assert (v, st, c) == (e["value"], e["status"], e["citation"]), k
    assert isinstance(W.get("wood.bending_mor"), tuple)
    assert len(W.get("wood.bending_mor")) == 3


def test_codeonly_get_hands_out_copies_not_the_stored_object():
    v, _, _ = W.get("wood.bending_mor")
    v["12pct"]["white_oak"] = -1
    assert W.get("wood.bending_mor")[0]["12pct"]["white_oak"] == 105
    e = W.entry("wood.bending_mor")
    e["status"] = "x"
    assert W.entry("wood.bending_mor")["status"] == "sourced"


def test_codeonly_unknown_key_raises_keyerror():
    for bad in ("no.such.key", "", "Wood.Bending_MOR", "wood"):
        with pytest.raises(KeyError):
            W.get(bad)
    with pytest.raises(KeyError):
        W.entry("no.such.key")


def test_codeonly_keys_lists_every_entry_in_file_order():
    assert W.keys() == list(ENT)


NETWORK = {"urllib", "urllib2", "urllib3", "http", "httplib", "requests", "httpx", "socket", "ssl", "ftplib", "smtplib",
           "aiohttp", "websocket", "websockets", "telnetlib", "xmlrpc", "asyncio", "selectors", "socketserver", "webbrowser"}


def test_codeonly_module_imports_no_network_library():
    tree = ast.parse(open(TOOL).read())
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            mods.add(n.module.split(".")[0])
    assert mods & NETWORK == set(), mods
    assert mods <= {"copy", "json", "os"}, mods


def test_codeonly_import_and_get_open_no_socket(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network used")
    monkeypatch.setattr(socket, "socket", boom)
    monkeypatch.setattr(socket, "create_connection", boom)
    monkeypatch.setattr(socket, "getaddrinfo", boom)
    spec = importlib.util.spec_from_file_location("w13_sourced_nonet", TOOL)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m.get("steel.strength_retention_EN1993-1-2")[1] == "sourced"


def test_codeonly_import_reads_no_file_until_the_first_call():
    src = open(TOOL).read()
    tree = ast.parse(src)
    top_calls = [n for n in tree.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)]
    assert top_calls == []
    spec = importlib.util.spec_from_file_location("w13_sourced_lazy", TOOL)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m._CACHE == {}
    m.get("jos3.package")
    assert "entries" in m._CACHE


def test_codeonly_a_missing_file_is_an_oserror_for_callers_to_catch(monkeypatch):
    spec = importlib.util.spec_from_file_location("w13_sourced_missing", TOOL)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    monkeypatch.setattr(m, "PATH", os.path.join(ROOT, "tests", "data", "does_not_exist.json"))
    with pytest.raises(OSError):
        m.get("jos3.package")


# ---------------------------------------------------------------- arithmetic identities [I]

def test_arith_dowel_embedment_formula_gives_26_MPa_for_G_045():
    v, st, _ = W.get("wood.dowel_embedment")
    assert st == "sourced" and v["formula"] == "Fe = 114.5 * G^1.84"
    assert abs(114.5 * 0.45 ** 1.84 - 26.3) < 0.1
    # the inch-pound form is the same line: 16,600 psi * 0.00689476 MPa/psi = 114.45 MPa
    assert abs(16600 * 0.00689476 - 114.5) < 0.1


def test_arith_screw_and_nail_formulas_are_the_handbook_ones():
    assert W.get("wood.screw_withdrawal")[0]["formula"] == "p = 108.25 * G^2 * D * L"
    assert W.get("wood.nail_withdrawal")[0]["formula"] == "p = 54.12 * G^2.5 * D * L"
    # #10 screw (4.83 mm) in pine G 0.45 with 19 mm thread: about 2.0 kN (T11 spec check)
    assert abs(108.25 * 0.45 ** 2 * 4.83 * 19 - 2012) < 5


def test_arith_ethanol_latent_heat_J_per_g_from_kJ_per_mol():
    v, _, _ = W.get("ethanol.latent_heat")
    assert abs(v["dH_vap_kJ_per_mol"] * 1000 / v["M_g_mol"] - v["J_per_g"]) < 0.2
    assert 830 < v["J_per_g"] < 845


def test_arith_texas_brine_fahrenheit_curve_converts_to_celsius():
    v, _, _ = W.get("nacl.freezing_curve_C")
    c = v["wt_pct_to_freezing_C"]
    assert abs(c["5"] - (26.7 - 32) * 5 / 9) < 0.01
    ks = sorted(c, key=int)
    assert all(c[a] > c[b] for a, b in zip(ks, ks[1:]))     # falls with salt up to the eutectic
    assert abs(c["23"] - (-21.1)) < 0.6                     # next to the eutectic temperature
    e = W.get("nacl.eutectic")[0]
    assert abs(e["T_C"] + 21.12) < 1e-9 and abs(e["wt_pct"] - 23.31) < 1e-9


def test_arith_nacl_solubility_series_is_monotone_and_has_eleven_points():
    v, _, _ = W.get("nacl.solubility_g_per_100g_water")
    assert len(v["T_C"]) == len(v["g_per_100g"]) == 11
    g = v["g_per_100g"]
    assert all(b > a for a, b in zip(g, g[1:]))
    # 26.4 wt % saturated at 20 C is 35.9 g per 100 g water
    assert abs(100 * v["saturated_wt_pct_20C"] / (100 - v["saturated_wt_pct_20C"]) - 35.87) < 0.05


def test_arith_relative_viscosity_is_mu_over_water_and_rises_with_salt():
    v, _, _ = W.get("nacl.rel_viscosity_20C")
    prev = 1.0
    for w in ("5", "10", "15", "20", "25"):
        r = v[w]
        assert abs(r["mu_uPas"] / 1002.0 - r["relative"]) < 2e-4
        m = 1000 * (int(w) / 100) / (58.443 * (1 - int(w) / 100))
        assert abs(m - r["molality"]) < 2e-3
        assert r["relative"] > prev
        prev = r["relative"]


def test_arith_andrade_two_point_fit_returns_its_two_inputs():
    v, _, _ = W.get("water.viscosity_andrade")
    for T, eta in ((293.15, v["eta_20C_mPas"]), (323.15, v["eta_50C_mPas"])):
        assert abs(v["andrade_A_mPas"] * math.exp(v["andrade_B_K_two_point_fit"] / T) - eta) / eta < 2e-3


def test_arith_steel_table_is_linear_interpolable_and_ends_at_zero():
    v, _, _ = W.get("steel.strength_retention_EN1993-1-2")
    n = len(v["T_C"])
    assert n == len(v["k_y_effective_yield"]) == len(v["k_p_proportional_limit"]) == len(v["k_E_modulus"]) == 13
    for col in ("k_y_effective_yield", "k_p_proportional_limit", "k_E_modulus"):
        c = v[col]
        assert c[0] == 1.0 and c[-1] == 0.0
        assert all(b <= a for a, b in zip(c, c[1:]))
    # the worked interpolations printed in New Steel Construction (547 C) follow from the 500 and 600 C rows
    ty, ke = v["k_y_effective_yield"], v["k_E_modulus"]
    i5, i6 = v["T_C"].index(500), v["T_C"].index(600)
    assert abs(ty[i5] + (ty[i6] - ty[i5]) * 0.47 - 0.634) < 5e-4
    assert abs(ke[i5] + (ke[i6] - ke[i5]) * 0.47 - 0.464) < 5e-4


def test_arith_janaf_reaction_constants_are_self_consistent():
    for key in ("janaf.boudouard", "janaf.wuestite", "janaf.calcite"):
        v, st, _ = W.get(key)
        assert st == "sourced"
        for c, r in v["by_T_C"].items():
            assert abs(r["T_K"] - (int(c) + 273.15)) < 1e-9
            assert abs(r["lnK"] - r["log10K"] * math.log(10)) < 1e-3
            assert abs(math.log10(r["K"]) - r["log10K"]) < 2e-3
    # Boudouard: equilibrium constant is one at about 700 C and rises steeply with T
    b = W.get("janaf.boudouard")[0]["by_T_C"]
    assert abs(b["700"]["log10K"]) < 0.05
    assert b["600"]["K"] < 1 < b["800"]["K"]
    # calcite: p_CO2 = 1 bar at 888 C by the tables used, within 12 K of the 898 C of the literature
    assert abs(W.get("janaf.calcite")[0]["T_p1bar_C"] - 898) < 12
    # van 't Hoff: Boudouard dH from the table slope is the 170.7 kJ the note quotes
    t1, t2 = b["800"], b["1000"]
    dH = -8.314462618 * (t2["lnK"] - t1["lnK"]) / (1 / t2["T_K"] - 1 / t1["T_K"])
    assert abs(dH / 1000 - 170.7) < 6


def test_arith_cusn_derived_resistivities_follow_from_iacs():
    v, _, _ = W.get("cusn.resistivity_vs_tin")
    for r in v["room_temp_20C"]:
        assert abs(1.7241 / (r["iacs_pct"] / 100) - r["resistivity_uOhm_cm_derived"]) < 0.01
        # the sheet quotes the conductivity to 2 digits (0.088 MS/cm), so allow 2 %
        assert abs(1 / (r["resistivity_uOhm_cm_derived"] * 1e-8) - r["S_per_m"]) / r["S_per_m"] < 0.02


def test_arith_cusn_nordheim_estimate_is_the_stated_formula():
    e = W.entry("cusn.nordheim_coefficient")
    assert e["status"] == "not_found"
    d = e["derived"]
    x = d["x_atom_fraction"][2]
    assert abs(13.0 / (x * (1 - x)) - d["C_from_10.7wtpct_4K2_residual_uOhm_cm"]) < 1.0


def test_arith_kick_energy_numbers_in_the_note_hold():
    v, _, _ = W.get("body.kick")
    m = v["effective_mass"]["front_kick_fraction_of_body_mass"] * v["effective_mass"]["athlete_body_mass_kg"]
    assert abs(m - 26.4) < 0.01
    assert abs(0.5 * m * 36 - 475) < 1
    assert abs(math.sqrt(200 / m) - 2.75) < 0.01


def test_arith_stair_power_for_70_kg_matches_the_per_kg_values():
    v, _, _ = W.get("body.stair_power")
    w = v["mechanical_power_W_per_kg"]
    assert abs(w["walking_1_flight"] * 70 - v["for_70_kg_W"]["walking"][0]) < 1
    assert abs(w["walking_2_flights"] * 70 - v["for_70_kg_W"]["walking"][1]) < 1
    assert abs(w["running_2"] * 70 - v["for_70_kg_W"]["running"][0]) < 1
    assert abs(w["running_every_other_step_1"] * 70 - v["for_70_kg_W"]["running"][1]) < 1


def test_arith_vertebra_age_regression_gives_the_young_male_value():
    v, _, _ = W.get("bone.vertebra_fracture_load")
    assert abs((10.43 - 0.923 * 2.5) - 8.1) < 0.05
    d = v["lumbar_ultimate_compressive_strength_kN"]
    assert d["min"] < d["female_adult_mean"] < d["all_mean"] < d["male_adult_mean"] < d["max"]


def test_arith_burn_table_celsius_matches_fahrenheit():
    v, _, _ = W.get("burn.moritz_henriques")
    for r in v["second_degree_exposure_time_by_surface_temp"]:
        assert abs((r["T_F"] - 32) * 5 / 9 - r["T_C"]) < 0.1
    # the damage integral at a constant basal temperature: Omega = 1 in 1/(P exp(-dE/RT)) seconds
    di = v["damage_integral"]

    def t_one(c):
        return 1.0 / (di["P_per_s"] * math.exp(-di["dE_J_per_mol"] / (8.314462618 * (c + 273.15))))
    assert t_one(65) < t_one(60) < t_one(55) < t_one(50)
    assert 0.5 < t_one(60) < 2 and 20 < t_one(55) < 40 and 600 < t_one(50) < 1800


def test_arith_ethanol_freezing_table_falls_with_ethanol():
    v, _, _ = W.get("ethanol.freezing_curve_C")
    w, t = v["wt_pct_ethanol"], v["freezing_C"]
    assert len(w) == len(t) and w == sorted(w)
    assert all(b < a for a, b in zip(t, t[1:]))
    assert t[0] == 0 and t[-1] == -114.3


def test_arith_wood_tables_carry_the_same_species_in_every_strength_column():
    cols = [W.get(k)[0] for k in ("wood.bending_mor", "wood.shear_parallel", "wood.tension_perpendicular", "wood.compression_parallel")]
    for c in cols[:3]:
        assert set(c["12pct"]) == set(c["green"]) == set(cols[0]["12pct"])
    # 12 % MC wood is stronger than green wood in every column and species
    for c in cols[:3]:
        for s, x in c["12pct"].items():
            assert x >= c["green"][s]
    assert W.get("wood.duration_of_load_10yr")[0] == 0.60
