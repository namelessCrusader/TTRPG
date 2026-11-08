"""CROSS-TOPIC CONSISTENCY OF THE WAVE 13 PHASE A MODULES (lane A-X).

Eleven topics were built in parallel, each with its own stand-in copy of the
numbers it borrows. This file compares the copies. It builds no world and
imports no World: pure functions and tables only (the audit tool,
tools/w13_audit.py, checks the import side).

How it reads
  * A case SKIPS only when a module (or a name in it) is absent, and the skip
    reason names it: "module mixtable absent" or "mechanisms.deposit missing".
    The audit prints every skip with its reason.
  * A numeric disagreement FAILS with both values, unless the pair is listed in
    KNOWN_DIFFS below. A known difference is a real difference that the owner
    lane or phase B must remove; the case is then reported as xfail with both
    values (pytest -rx shows them) and passes silently once the two agree.
  * Tags (plan section 9): the names say it. Everything here that checks
    arithmetic identity, conservation or a closed form is [I]. The cases that
    compare physical data against a handbook value are test_u_*.
"""
import ast
import importlib
import importlib.util
import math
import os
import re
import subprocess
import sys
import types

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
VOX = os.path.join(ROOT, "src", "voxel")
MARK = os.environ.get("MARK1_BUILD", "/home/nisargparikh/.cache/mark1-build")


def _load_manifest():
    spec = importlib.util.spec_from_file_location(
        "w13_manifest", os.path.join(ROOT, "tools", "w13_manifest.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


MAN = _load_manifest()

# Real differences between two copies, found by this file. key -> why, who fixes.
# A key listed here turns a failure into an xfail that prints both values.
KNOWN_DIFFS = {
    "masson_nacl_mixdata": "T4 mixdata.TABLES['masson']['nacl'] holds its own (16.6, 1.8); T1's "
                           "species record (16.62, 1.87) is the one source (plan 3.1): T4 reads it at wiring",
    "combine_breaking_repair": "T2 breaking.combine has no lower clip at 0, matdamage.combine does: a "
                               "repair larger than the damage gives a negative D there",
    "l_ice_climate": "T8 climate.L_ICE is 334e3 J/kg, the shared water Lf is 333.55e3 (0.13 percent)",
    "record_data_wood": "T2 record_data.STRENGTH['wood'] is a stand-in (nu 0.37, K_Ic 0.5e6) against "
                        "the bond table and T9 (nu 0.35, KIc 0.4e6)",
}

# A known difference is allowed only up to a BOUND (rel, abs): the xfail does not hide a larger drift
# under the same key. |a - b| <= abs + rel * max(|a|, |b|) for every element, else the case FAILS.
KNOWN_BOUND = {
    "masson_nacl_mixdata": (0.05, 0.0),         # 16.6/16.62 and 1.8/1.87 (3.7 percent)
    "combine_breaking_repair": (0.0, 1.0),      # the missing clip at 0: D is in [0, 1], so at most 1
    "l_ice_climate": (0.002, 0.0),              # 334000 / 333550 (0.13 percent)
    "record_data_wood": (0.30, 0.0),            # nu 0.37 / 0.35, K_Ic 5e5 / 4e5 (20 percent)
}
assert set(KNOWN_BOUND) == set(KNOWN_DIFFS), "every known difference has a bound, and no bound without one"


def _within_bound(key, a, b):
    rel, absol = KNOWN_BOUND[key]
    with np.errstate(invalid="ignore"):
        return bool(np.all((a == b) | (np.abs(a - b) <= absol + rel * np.maximum(np.abs(a), np.abs(b)))))


# ---- helpers -------------------------------------------------------------------------
def mod(name):
    """A new src/voxel module, or skip naming it. Any other import error is real."""
    full = "src.voxel." + name
    try:
        return importlib.import_module(full)
    except ModuleNotFoundError as e:
        if e.name == full:
            pytest.skip(f"module {name} absent")
        raise


def attr(m, *names):
    """The first of `names` that `m` has, or skip naming the module and the name."""
    for n in names:
        if hasattr(m, n):
            return getattr(m, n)
    pytest.skip(f"{m.__name__.split('.')[-1]}.{names[0]} missing (tried {', '.join(names)})")


def same(key, a, b, rel=1e-12, absol=0.0, what=""):
    """a and b agree to rel (and absol), or the case fails with both values (xfail if the
    pair is a KNOWN_DIFF)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    with np.errstate(invalid="ignore"):
        ok = bool(np.all((a == b) | (np.abs(a - b) <= absol + rel * np.maximum(np.abs(a), np.abs(b)))))
    if ok:
        return
    msg = f"{what or key}: {a.tolist()} vs {b.tolist()}"
    if key in KNOWN_DIFFS:
        if not _within_bound(key, a, b):
            raise AssertionError(f"{msg} -- DRIFT: larger than the known bound {KNOWN_BOUND[key]} of {key}")
        pytest.xfail(f"KNOWN {key}: {msg} -- {KNOWN_DIFFS[key]}")
    raise AssertionError(msg)


def agree_all(items):
    """items: [(key, a, b, rel, what)]. Every pair is checked; unknown mismatches fail together,
    and if all mismatches are KNOWN_DIFFS the case xfails listing every one."""
    unknown, known = [], []
    for key, a, b, rel, what in items:
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        with np.errstate(invalid="ignore"):
            ok = bool(np.all((a == b) | (np.abs(a - b) <= rel * np.maximum(np.abs(a), np.abs(b)))))
        if not ok:
            if key in KNOWN_DIFFS and _within_bound(key, a, b):
                known.append(f"{what}: {a.tolist()} vs {b.tolist()} [{key}]")
            else:
                unknown.append(f"{what}: {a.tolist()} vs {b.tolist()} [{key}]"
                               + (" DRIFT past the known bound" if key in KNOWN_DIFFS else ""))
    if unknown:
        raise AssertionError("; ".join(unknown + known))
    if known:
        keys = sorted({k for k, *_ in items if k in KNOWN_DIFFS})
        pytest.xfail("KNOWN " + "; ".join(known) + " -- " + " / ".join(KNOWN_DIFFS[k] for k in keys))


def src_of(name):
    return open(os.path.join(VOX, name + ".py")).read()


def present(name):
    return os.path.exists(os.path.join(VOX, name + ".py"))


# =====================================================================================
# (1) THE WATER SET
# =====================================================================================
def _lf_sources():
    def species():
        return float(mod("species").get("water").get("Lf", "ref"))
    return {
        "species.ref": species,
        "matterdata.LF0": lambda: float(mod("matterdata").LF0),
        "mixdata.L_FUS0": lambda: float(mod("mixdata").L_FUS0),
        "mixwater.Lfus(0)": lambda: float(mod("mixwater").Lfus(0.0)),
        "icephase.ICE_ROW": lambda: float(mod("icephase").ICE_ROW["Lf"]),
        "_standin.lfus(0)": lambda: float(mod("_standin").lfus(0.0)),
    }


@pytest.mark.parametrize("who", sorted(_lf_sources()))
def test_water_heat_of_fusion_is_one_number(who):
    ref = _lf_sources()["species.ref"]()
    assert ref == 333.55, f"species water Lf {ref}"
    same("lf_" + who, _lf_sources()[who](), ref, rel=1e-12, what=f"Lf at 0 C: {who} vs species")


def test_water_Lfus_slope_is_the_same_everywhere():
    """Kirchhoff: Lfus(T) = 333.55 + 2.09 T in T4, the stand-in and T6's constants."""
    mw, st, md = mod("mixwater"), mod("_standin"), mod("matterdata")
    for T in (-20.0, -5.0, 0.0, 10.0):
        same("lfus_T", mw.Lfus(T), st.lfus(T), what=f"Lfus({T}) mixwater vs _standin")
        same("lfus_T", mw.Lfus(T), md.LF0 + md.LF_SLOPE * T, what=f"Lfus({T}) mixwater vs matterdata")


def test_icephase_L_CONV20_is_derived_from_the_shared_Lf():
    """L_conv20 = Lf + 20 (c_water - c_ice) with the ENGINE c of water (4.2): 375.75, the 375.8
    of the plan, and the same number everywhere it is used."""
    sp, ip = mod("species"), mod("icephase")
    w = sp.get("water")
    Lf = float(w.get("Lf", "ref"))
    cw = float(w.get("c_l", "engine"))
    ci = float(ip.ICE_ROW["c"])
    derived = Lf + 20.0 * (cw - ci)
    same("l_conv20", ip.L_CONV20, derived, rel=1e-12, what="icephase.L_CONV20 vs Lf + 20 (cw - ci)")
    assert abs(ip.L_CONV20 - 375.8) < 0.06, ip.L_CONV20
    for name in ("psychro", "bodyheat", "rxn"):
        if present(name):
            m = mod(name)
            v = getattr(m, "L_CONV20", None)
            if v is not None:
                same("l_conv20", v, ip.L_CONV20, what=f"{name}.L_CONV20 vs icephase.L_CONV20")


def _psat_pa_sources():
    """name -> (function T[C] -> Pa, relative tolerance against T4's IF97 over 0..100 C)."""
    def kpa(modname, fname, scale):
        return lambda T: scale * np.asarray(attr(mod(modname), fname)(T), dtype=float)
    return {
        "_standin.psat_water": (kpa("_standin", "psat_water", 1.0), 1.5e-3),
        "psychro.p_sat_water": (kpa("psychro", "p_sat_water", 1.0), 1.5e-3),
        "layer_laws.psat_water": (lambda T: np.array([attr(mod("layer_laws"), "psat_water")(float(t))
                                                      for t in np.atleast_1d(T)]), 1.5e-3),
        "layer_laws.psat_sonntag": (lambda T: np.array([attr(mod("layer_laws"), "psat_sonntag")(float(t))
                                                        for t in np.atleast_1d(T)]), 5e-3),
        "bodyair.p_sat (Magnus, kPa)": (kpa("bodyair", "p_sat", 1000.0), 5e-3),
        "climate.psat_magnus (hPa)": (kpa("climate", "psat_magnus", 100.0), 5e-3),
    }


@pytest.mark.parametrize("who", sorted(_psat_pa_sources()))
def test_psat_of_water_agrees_with_mixwater_in_its_band(who):
    """Saturation pressure at 0..100 C against T4's IF97 (2339.2 Pa at 20 C): the stand-ins are
    inside 0.15 percent, the Magnus forms inside 0.5 percent."""
    mw = mod("mixwater")
    fn, tol = _psat_pa_sources()[who]
    T = np.array([0.5, 10.0, 20.0, 30.0, 37.0, 60.0, 90.0]) if "Magnus" in who or "magnus" in who \
        else np.array([0.5, 10.0, 20.0, 37.0, 60.0, 100.0])
    if "Magnus" in who or "magnus" in who or "sonntag" in who:
        T = np.array([0.5, 10.0, 20.0, 30.0, 37.0])
    ref = np.asarray(mw.psat_water(T), dtype=float)
    got = np.asarray(fn(T), dtype=float)
    rel = np.abs(got / ref - 1.0)
    assert rel.max() <= tol, f"{who}: worst relative error {rel.max():.2e} > {tol} at T={T[rel.argmax()]}: " \
                             f"{got[rel.argmax()]} vs IF97 {ref[rel.argmax()]}"


def test_u_psat_of_water_at_20C_is_2339_pa_in_the_if97_source():
    """[U] the one source value: IF97 at 20 C is 2339.2 Pa (handbook 2.339 kPa)."""
    v = float(mod("mixwater").psat_water(20.0))
    assert abs(v - 2339.2) < 1.0, v


def test_climate_dew_point_is_the_inverse_of_its_own_psat_and_near_if97():
    cl, mw = mod("climate"), mod("mixwater")
    for T, rh in ((20.0, 0.5), (30.0, 0.8), (5.0, 0.9), (25.0, 0.3)):
        td = float(cl.dew_point(T, rh))
        same("dew_roundtrip", cl.psat_magnus(td), rh * cl.psat_magnus(T), rel=1e-9,
             what=f"psat(dew({T},{rh})) vs rh psat({T})")
        # the dew point T4's IF97 would give: psat(Td) = rh psat(T)
        lo, hi = 0.5, 60.0
        target = rh * float(mw.psat_water(T))
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if float(mw.psat_water(mid)) < target:
                lo = mid
            else:
                hi = mid
        assert abs(td - 0.5 * (lo + hi)) < 0.12, f"dew point at {T} C, rh {rh}: climate {td}, IF97 {0.5 * (lo + hi)}"


def test_dew_point_copies_agree_between_climate_and_layer_laws():
    cl, ll = mod("climate"), mod("layer_laws")
    for T, rh in ((20.0, 0.5), (30.0, 0.8), (5.0, 0.9)):
        same("dew_copies", cl.dew_point(T, rh), ll.dew_point(T, rh), rel=1e-12,
             what=f"dew point ({T} C, rh {rh}) climate vs layer_laws")


def test_tsat_stand_in_and_mixwater_agree():
    mw, ll = mod("mixwater"), mod("layer_laws")
    for P in (50e3, 101325.0, 200e3):
        a, b = float(mw.tsat_water(P)), float(ll.tsat_water(P))
        assert abs(a - b) < 0.05, f"boiling point at {P} Pa: mixwater {a}, layer_laws {b}"
    assert abs(float(mw.tsat_water(101325.0)) - 100.0) < 0.05


# =====================================================================================
# (2) MASSON AND THE DISSOLVED NaCl NUMBERS
# =====================================================================================
def test_masson_for_nacl_is_the_species_record_everywhere():
    sp, md = mod("species"), mod("mixdata")
    s = sp.get("sodium_chloride")
    assert (s.get("phiV0", "ref"), s.get("Sv", "ref")) == (16.62, 1.87)
    mv = md.TABLES["masson"]["nacl"][0]
    same("masson_nacl_mixdata", mv, (s.get("phiV0", "ref"), s.get("Sv", "ref")), rel=1e-12,
         what="Masson (V0, S) for NaCl: mixdata vs species")


def test_mixrules_reads_the_species_record_for_dissolved_nacl_and_water_fusion():
    """T1's own rule module (mixrules.sp_arrays) must hand back the species record's numbers."""
    sp, mx = mod("species"), mod("mixrules")
    arr = attr(mx, "sp_arrays")()
    i = sp.BY_NAME["sodium_chloride"]
    s = sp.get(i)
    agree_all([("mixrules_" + f, arr[f][i], s.get(f, "engine"), 1e-12, f"NaCl {f}: mixrules.sp_arrays vs species")
               for f in ("phiV0", "Sv", "c_app", "k_app")])
    w = sp.BY_NAME["water"]
    same("lf", arr["Lf"][w], sp.get("water").get("Lf", "engine"), what="water Lf: mixrules.sp_arrays vs species")


def test_u_masson_solution_density_of_20pct_brine_is_the_same_to_half_a_percent():
    """The T6 solution rule (linear, v = 0.355 ml/g) and T4's Masson density must give the same
    20 wt percent brine; the handbook is 1.148 g/ml (u)."""
    sp, mw, md = mod("species"), mod("mixwater"), mod("matterdata")
    s = sp.get("sodium_chloride")
    # species Masson: V = n (phiV0 + Sv sqrt(m)) + water volume
    M = s.M
    water_g, salt_g = 80.0, 20.0
    mol = salt_g / M
    m = mol / (water_g / 1000.0)
    V = water_g / 0.9982 + mol * (s.get("phiV0", "ref") + s.get("Sv", "ref") * math.sqrt(m))
    rho_species = 100.0 / V
    rho_linear = 100.0 / (water_g + md.V_APP["NaCl"] * salt_g)
    assert abs(rho_species / rho_linear - 1.0) < 5e-3, (rho_species, rho_linear)
    assert abs(rho_species - 1.148) < 0.006, rho_species
    comp = np.zeros(len(mod("mixdata").ORDER))
    comp[mod("mixdata").INDEX["h2o"]] = water_g
    comp[mod("mixdata").INDEX["nacl"]] = salt_g
    rho4 = float(mw.density(comp, 20.0))
    assert abs(rho4 / rho_species - 1.0) < 5e-3, f"mixwater.density {rho4} vs species Masson {rho_species}"


def test_u_nacl_dissolved_heat_capacity_has_one_owner():
    """c_app of dissolved NaCl (0.28 J/g/K) lives in the species record; no other new module may
    carry a second copy under that name. A brine of 20 wt percent then has c = 0.8 c_w + 0.2 c_app."""
    sp = mod("species")
    s = sp.get("sodium_chloride")
    assert s.get("c_app", "ref") == 0.28
    c_brine = 0.8 * 4.18 + 0.2 * 0.28
    assert abs(c_brine - 3.4) < 0.05          # [U] handbook near 3.4
    copies = []
    for name in MAN.modules():
        if name in ("species", "species_seed") or not present(name):
            continue
        for m in re.finditer(r"""(["'])(?:c_app|cp_app)\1\s*[:\]]|\b(?:c_app|cp_app)\s*=\s*[0-9.]""", src_of(name)):
            copies.append((name, m.group(0)))
    assert not copies, f"a second dissolved-heat-capacity constant: {copies}"


# =====================================================================================
# (3) WOOD STRENGTHS
# =====================================================================================
def _wood_source(name):
    from src.bonds import materials as B
    if name == "bonds":
        return dict(E=B.WOOD_L[0], nu=B.WOOD_L[1], st=B.WOOD_L[2], E_perp=B.WOOD_T[0], st_perp=B.WOOD_T[2])
    if name == "matrec":
        p = mod("matrec").props("wood")
        return dict(E=p["E"], nu=p["nu"], st=p["st"], E_perp=p["E_perp"], st_perp=p["st_perp"],
                    tau=p["tau"], ductility=p["ductility"], weibull_m=p["weibull_m"], KIc=p["KIc"])
    if name == "record_data":
        d = mod("record_data").STRENGTH["wood"]
        return dict(E=d["E"], nu=d["nu"], st=d["sigma_u"], ductility=d["eps_f"],
                    weibull_m=d["weibull_m"], KIc=d["K_Ic"])
    if name == "joinery":
        d = mod("joinery").SUBSTRATE["wood"]
        return dict(st_perp=d["sig_t"], tau=d["tau"])
    if name == "mechanisms":
        f = attr(mod("mechanisms"), "material_props")
        a, c = f("wood", "along"), f("wood", "across")
        return dict(E=a["E"], nu=a["nu"], st=a["st"], E_perp=a["E_perp"], st_perp=a["st_perp"],
                    tau=a.get("tau"), ductility=a.get("ductility"))
    if name == "fasteners":
        # fasteners takes the wood strengths as ARGUMENTS (sigma_t_perp ...): it must type none of its own.
        mod("fasteners")
        code = "\n".join(ln.split("#")[0] for ln in src_of("fasteners").splitlines())
        typed = re.findall(r"\b(?:3(?:\.0)?e6|0?\.55e9|11e9|1\.1e10|6(?:\.0)?e7|60e6|5\.5e8)\b", code)
        assert not typed, f"fasteners.py types a wood strength: {typed}"
        return {}
    raise KeyError(name)


@pytest.mark.parametrize("who", ["matrec", "record_data", "joinery", "mechanisms", "fasteners"])
def test_wood_strengths_equal_the_bond_table(who):
    bond = _wood_source("bonds")
    got = _wood_source(who)
    key = "record_data_wood" if who == "record_data" else f"wood_{who}"
    agree_all([(key, v, bond[k], 1e-12, f"wood {k}: {who} vs bond table")
               for k, v in got.items() if v is not None and k in bond])


def test_wood_shear_and_ductility_between_matrec_joinery_and_record_data():
    mr = _wood_source("matrec")
    jo = _wood_source("joinery")
    rd = _wood_source("record_data")
    agree_all([
        ("wood_tau", jo["tau"], mr["tau"], 0.05, "wood shear: joinery (Douglas-fir 7.8 MPa) vs matrec 8 MPa (5 percent band)"),
        ("wood_weibull", rd["weibull_m"], mr["weibull_m"], 1e-12, "wood Weibull m: record_data vs matrec"),
        ("wood_ductility", rd["ductility"], mr["ductility"], 1e-12, "wood ductility: record_data eps_f vs matrec"),
        ("record_data_wood", rd["KIc"], mr["KIc"], 1e-12, "wood K_Ic: record_data vs matrec")])


def test_wood_row_hankinson_matches_the_bond_store():
    """matrec's wood_row(cos2) is the bond store's (wave 12 BONDS): the same mix of along and across."""
    from src.bonds import materials as B
    views = mod("matrec")
    if not hasattr(views, "legacy_views"):
        pytest.skip("matrec.legacy_views missing")
    V = views.legacy_views((5, 6, 7))
    wr = V["wood_row"]
    for c2 in (0.0, 0.25, 0.5, 1.0):
        a, b = wr(c2), B.wood_row(c2)
        same("wood_row", a, b, rel=1e-12, what=f"wood_row({c2}) matrec vs bonds")


# =====================================================================================
# (4) STEFAN ICE
# =====================================================================================
GRID_STEFAN = [(dT, t) for dT in (2.0, 10.0, 25.0) for t in (3600.0, 86400.0, 7 * 86400.0)]


@pytest.mark.parametrize("dT,t", GRID_STEFAN)
def test_stefan_ice_front_icephase_and_climate_agree(dT, t):
    """One equation, three integrators: slowforms front (K = k dT/(rho Lf), S = t), icephase.stefan
    and climate.ice_thickness, with each module's own constants, agree to 1e-3."""
    sf, ip, cl = mod("slowforms"), mod("icephase"), mod("climate")
    k, rho, Lf = 2.2, 917.0, 333.55e3
    K = k * dT / (rho * Lf)
    x_front = sf.front_x_of_S(t, K, 0.0)
    x_ice = float(ip.stefan(dT, t))
    x_cl = float(cl.ice_thickness(dT * t))
    same("stefan", x_front, x_ice, rel=1e-3, what="slowforms.front vs icephase.stefan")
    same("stefan", x_cl, x_ice, rel=1e-3, what="climate.ice_thickness vs icephase.stefan")
    same("stefan", x_cl, x_front, rel=1e-3, what="climate.ice_thickness vs slowforms.front")


@pytest.mark.parametrize("dT,t", GRID_STEFAN)
def test_stefan_ice_with_an_air_film_agrees(dT, t):
    """The film form: the formulas are the same when the constants are the same (checked with climate's
    latent heat set to the shared Lf); with climate's own L_ICE (334e3) the film form is 0.13 percent
    apart, a KNOWN_DIFF (the no-film form hides it behind the square root)."""
    sf, ip, cl = mod("slowforms"), mod("icephase"), mod("climate")
    h = 20.0
    k, rho, Lf = 2.2, 917.0, 333.55e3
    K = k * dT / (rho * Lf)
    R0 = rho * Lf / (h * dT)                  # x^2/(2K) + R0 x = t
    x_front = sf.front_x_of_S(t, K, R0)
    x_ice = float(ip.stefan_film(dT, t, h))
    same("stefan", x_front, x_ice, rel=1e-9, what="film: slowforms.front vs icephase.stefan_film")
    x_cl_same = float(cl.ice_thickness(dT * t, h_a=h, latent=Lf))
    same("stefan", x_cl_same, x_ice, rel=1e-9, what="film, same Lf: climate.ice_thickness vs icephase.stefan_film")
    x_cl = float(cl.ice_thickness(dT * t, h_a=h))
    same("l_ice_climate", x_cl, x_ice, rel=1e-3, what="film, climate's own L_ICE: climate.ice_thickness vs icephase.stefan_film")


def test_stefan_front_inverts_exactly():
    sf = mod("slowforms")
    for K, R0, x in ((1e-8, 0.0, 0.1), (2e-9, 5e4, 0.3)):
        S = sf.front_S_of_x(x, K, R0)
        same("stefan_inv", sf.front_x_of_S(S, K, R0), x, rel=1e-9, what="front_x_of_S(front_S_of_x(x)) == x")


def test_ice_latent_heat_constants_agree_between_climate_and_icephase():
    cl, ip = mod("climate"), mod("icephase")
    same("l_ice_climate", cl.L_ICE, ip.ICE_ROW["Lf"] * 1e3, rel=1e-6, what="ice latent heat J/kg: climate.L_ICE vs icephase")
    same("ice_k", cl.K_ICE, ip.ICE_ROW["k"], what="ice k: climate vs icephase")
    same("ice_rho", cl.RHO_ICE, ip.ICE_ROW["rho"], what="ice rho: climate vs icephase")


def test_ice_row_numbers_species_vs_t6():
    sp, ip = mod("species"), mod("icephase")
    w = sp.get("water")
    same("ice_rho", w.get("rho_s", "ref"), ip.ICE_ROW["rho"], what="ice rho: species ref vs ICE_ROW")
    same("ice_k", w.get("k_s", "ref"), ip.ICE_ROW["k"], what="ice k: species ref vs ICE_ROW")
    # B3 closed the c difference (species_seed holds T6's 2.09 now) and added the emissivity; both must agree
    same("ice_c", w.get("c_s", "ref"), ip.ICE_ROW["c"], what="ice c: species ref vs ICE_ROW")
    same("ice_emiss", w.get("emiss_s", "ref"), ip.ICE_ROW["emissivity"], what="ice emissivity: species ref vs ICE_ROW")


# =====================================================================================
# (5) TRIDIAGONAL SOLVES
# =====================================================================================
def _tridiag_problem(rng, K, n):
    a = rng.uniform(-1.0, 0.0, (K, n))
    c = rng.uniform(-1.0, 0.0, (K, n))
    a[:, 0] = 0.0
    c[:, -1] = 0.0
    b = 1.0 - a - c + rng.uniform(0.1, 1.0, (K, n))
    d = rng.uniform(-5.0, 5.0, (K, n))
    return a, b, c, d


def test_tridiagonal_solves_in_thinheat_and_skinburn_agree():
    th, sk = mod("thinheat"), mod("skinburn")
    rng = np.random.default_rng(13)
    a, b, c, d = _tridiag_problem(rng, 7, 9)
    x_th = th.tridiag_solve(a, b, c, d)                       # (K, n): rows are systems
    x_sk = sk._thomas(a.T.copy(), b.T.copy(), c.T.copy(), d.T.copy())     # (n, N): columns are systems
    same("tridiag", x_th, x_sk.T, rel=1e-12, absol=1e-13, what="thinheat.tridiag_solve vs skinburn._thomas")
    # and both are THE solution
    for r in range(a.shape[0]):
        A = np.diag(b[r]) + np.diag(a[r, 1:], -1) + np.diag(c[r, :-1], 1)
        same("tridiag", np.linalg.solve(A, d[r]), x_th[r], rel=1e-10, absol=1e-12, what="vs dense solve")


# =====================================================================================
# (6) DAMAGE COMBINE
# =====================================================================================
def _combine_of(name):
    if name == "matdamage":
        return mod("matdamage").combine
    m = mod(name)
    return attr(m, "combine", "combine_damage", "_combine")


def _eval(f, *args):
    """f on arrays, or elementwise when f is scalar-only (a documented choice of some copies)."""
    try:
        return np.asarray(f(*args), dtype=float)
    except (TypeError, ValueError):
        a, b = np.broadcast_arrays(*[np.asarray(x, dtype=float) for x in args])
        return np.array([f(float(x), float(y)) for x, y in zip(a.ravel(), b.ravel())]).reshape(a.shape)


def _combine_grid():
    D = np.array([0.0, 0.01, 0.3, 0.5, 0.9, 0.998, 0.999])
    dD = np.array([0.0, 0.02, 0.3, 0.7, 1.0])
    return np.meshgrid(D, dD)


def _handler_nodes(name):
    """The statements inside `except ImportError:` handlers at the top level of a module: where the
    stand-in copies live."""
    tree = ast.parse(src_of(name))
    out = []
    for node in tree.body:
        if isinstance(node, ast.Try):
            for h in node.handlers:
                t = h.type
                names = [t.id] if isinstance(t, ast.Name) else [e.id for e in getattr(t, "elts", []) if isinstance(e, ast.Name)]
                if "ImportError" in names or "ModuleNotFoundError" in names:
                    out += h.body
    return out


def _fallback_func(name, *funcs):
    """A stand-in function defined in an `except ImportError:` handler, compiled alone."""
    tops = [n for n in ast.parse(src_of(name)).body if isinstance(n, ast.FunctionDef)]
    for st in _handler_nodes(name) + [t for t in tops if t.name.endswith("_copy")]:
        if isinstance(st, ast.FunctionDef) and st.name in funcs:
            ns = {"np": np, "math": math}
            exec(compile(ast.Module(body=[st], type_ignores=[]), f"{name}-standin", "exec"), ns)
            return ns[st.name]
    pytest.skip(f"{name} has no stand-in {funcs} in an except ImportError handler")


def _fallback_consts(name):
    """{name: value} of constants assigned in `except ImportError:` handlers (tuple assigns too)."""
    out = {}
    for st in _handler_nodes(name):
        if isinstance(st, ast.Assign):
            for tg in st.targets:
                try:
                    val = ast.literal_eval(st.value)
                except ValueError:
                    continue
                if isinstance(tg, ast.Name):
                    out[tg.id] = val
                elif isinstance(tg, ast.Tuple):
                    out.update(zip([e.id for e in tg.elts], val))
    return out


@pytest.mark.parametrize("who", ["mechanisms", "breaking", "bonestate", "matbeam", "vasculature"])
def test_damage_combine_is_the_same_function_as_matdamage(who):
    ref = _combine_of("matdamage")
    f = _combine_of(who)
    _same_combine(who, f, ref)


@pytest.mark.parametrize("who", ["matbeam", "vasculature"])
def test_the_standin_combine_inside_an_except_ImportError_is_the_same_function(who):
    """matbeam and vasculature import matdamage.combine and keep a local copy for when it is absent;
    the copy must be the same law."""
    _same_combine(who + " (stand-in)", _fallback_func(who, "combine", "_combine", "_combine_copy"), _combine_of("matdamage"))


def test_mechanisms_standin_combine_is_the_same_function(monkeypatch):
    mech = mod("mechanisms")
    if not hasattr(mech, "_matdamage"):
        pytest.skip("mechanisms._matdamage missing")
    monkeypatch.setattr(mech, "_matdamage", None)           # mechanisms reads it: force the local branch
    _same_combine("mechanisms (stand-in)", mech.combine, _combine_of("matdamage"))


def _same_combine(who, f, ref):
    Dg, dg = _combine_grid()
    key = "combine_breaking_repair" if who == "breaking" else f"combine_{who.split()[0]}"
    same(key, _eval(f, Dg, dg), _eval(ref, Dg, dg), rel=0.0, what=f"combine on a grid: {who} vs matdamage (damage rising)")
    # repairs (negative dD): a glued crack, a weld
    Dg2, dr = np.meshgrid(np.array([0.0, 0.1, 0.5, 0.9]), np.array([-0.05, -0.1, -0.5, -1.0]))
    same(key, _eval(f, Dg2, dr), _eval(ref, Dg2, dr), rel=0.0, what=f"combine with repairs: {who} vs matdamage")
    # scalars give scalars of the same value
    same(key, f(0.2, 0.3), ref(0.2, 0.3), rel=0.0, what=f"combine(0.2, 0.3): {who}")


def test_damage_combine_laws_hold_in_matdamage():
    f = _combine_of("matdamage")
    rng = np.random.default_rng(6)
    a, b, c = rng.uniform(0, 0.9, (3, 50))
    same("combine_laws", f(a, b), f(b, a), rel=0.0, what="commutative")
    same("combine_laws", f(f(a, b), c), f(a, f(b, c)), rel=1e-12, what="associative")
    assert float(np.max(f(a, 1.0))) <= 0.999


# =====================================================================================
# (7) MET AND ETA
# =====================================================================================
def test_met_and_eta_constants_come_from_energetics():
    en = mod("energetics")
    assert (en.MET_W_PER_KG, en.ETA) == (1.22, 0.22)
    ft = mod("fatigue")
    same("met_eta", ft.ETA, en.ETA, what="fatigue.ETA vs energetics.ETA")
    same("met_eta", ft.MET_W_PER_KG, en.MET_W_PER_KG, what="fatigue.MET_W_PER_KG vs energetics")
    # the stand-in copy inside fatigue must not drift from the owner
    st = getattr(ft, "STAND_IN", None)
    if st is not None:
        same("met_eta", st["ETA"], en.ETA, what="fatigue.STAND_IN ETA vs energetics")
        same("met_eta", st["MET_W_PER_KG"], en.MET_W_PER_KG, what="fatigue.STAND_IN MET vs energetics")
    ecc = getattr(en, "ECC_DIV", None)
    if ecc is not None and st is not None:
        same("met_eta", st["ECC_DIV"], ecc, what="eccentric divisor: fatigue stand-in vs energetics")


def test_stairs_uses_the_energetics_constants():
    en, stairs = mod("energetics"), mod("stairs")
    same("met_eta", attr(stairs, "ETA_UP", "ETA", "ETA_GROSS"), en.ETA, what="stairs eta vs energetics.ETA")
    same("met_eta", attr(stairs, "ECC_DIV"), en.ECC_DIV, what="stairs eccentric divisor vs energetics")
    same("met_eta", attr(stairs, "GRAV"), en.GRAV, what="stairs g vs energetics.GRAV")
    met = getattr(stairs, "MET_W_PER_KG", None)
    if met is not None:
        same("met_eta", met, en.MET_W_PER_KG, what="stairs MET vs energetics.MET_W_PER_KG")


@pytest.mark.parametrize("who,names", [
    ("stairs", {"_ETA": "ETA", "_ECC": "ECC_DIV", "_GRAV": "GRAV"}),
    ("fatigue", {"ETA": "ETA", "MET_W_PER_KG": "MET_W_PER_KG"}),
])
def test_the_standin_constants_for_energetics_have_not_drifted(who, names):
    """stairs and fatigue import energetics and keep the same numbers for when it is absent."""
    en = mod("energetics")
    mod(who)
    got = _fallback_consts(who)
    if not got:
        pytest.skip(f"{who} has no stand-in constants in an except ImportError handler")
    for local, owner in names.items():
        if local in got:
            same("met_eta", got[local], getattr(en, owner), what=f"{who} stand-in {local} vs energetics.{owner}")


def test_no_new_module_types_the_metabolic_constants_a_second_time():
    """Only energetics (owner), fatigue (documented stand-in) and the tests may hold 1.22 W/kg
    or eta 0.22 as a typed constant on an assignment line."""
    pat = re.compile(r"^\s*(?:MET_W_PER_KG|MET_W_KG|ETA|ETA_GROSS|ETA_MECH)\s*=\s*(?:1\.22|0\.22)\b", re.M)
    hits = [n for n in MAN.modules() if present(n) and pat.search(src_of(n))]
    assert set(hits) <= {"energetics", "fatigue", "stairs", "tissue"}, hits
    # tissue (T7a) typed its own 1.22 before energetics (T7b) existed: the same value, one owner at wiring
    ti, en = mod("tissue"), mod("energetics")
    same("met_eta", ti.MET_W_PER_KG, en.MET_W_PER_KG, what="tissue.MET_W_PER_KG vs energetics")


# =====================================================================================
# (8) COMPONENT KEY PACK AND UNPACK
# =====================================================================================
def _eval3(f, S, F, Z):
    """pack_key on arrays, or elementwise when the copy is scalar-only (layer_data's is)."""
    try:
        return np.asarray(f(S, F, Z))
    except (TypeError, ValueError):
        return np.array([f(int(a), int(b), int(c)) for a, b, c in zip(S.ravel(), F.ravel(), Z.ravel())],
                        dtype=np.uint32).reshape(S.shape)


def _key_modules():
    """The new modules that define pack_key (not just import it)."""
    found = []
    for n in MAN.modules():
        if present(n) and re.search(r"^def pack_key\b", src_of(n), re.M):
            found.append(n)
    return found


def test_pack_key_is_defined_in_one_place_or_equal_everywhere():
    owners = _key_modules()
    assert "species" in owners
    ref = mod("species")
    sp = np.array([0, 1, 17, 63, 64, 95, 96, 1000, 65535])
    fo = np.array([0, 1, 2, 3])
    sz = np.array([0, 1, 40, 255])
    S, F, Z = np.meshgrid(sp, fo, sz, indexing="ij")
    for n in owners:
        if n == "species":
            continue
        m = mod(n)
        same("key_pack", _eval3(m.pack_key, S, F, Z), ref.pack_key(S, F, Z), rel=0.0, what=f"pack_key arrays: {n} vs species")
        for s, f, z in ((0, 0, 0), (17, 3, 0), (65535, 3, 255), (21, 2, 64)):
            same("key_pack", m.pack_key(s, f, z), ref.pack_key(s, f, z), rel=0.0, what=f"pack_key({s},{f},{z}): {n} vs species")
            same("key_pack", m.unpack_key(ref.pack_key(s, f, z)), (s, f, z), rel=0.0,
                 what=f"unpack_key: {n} vs ({s},{f},{z})")
        k = ref.pack_key(S, F, Z)
        a = ref.unpack_key(k)
        flat = [m.unpack_key(int(v)) for v in k.ravel()]
        b = [np.array([t[i] for t in flat]).reshape(k.shape) for i in range(3)]
        for x, y in zip(a, b):
            same("key_pack", x, y, rel=0.0, what=f"unpack_key arrays: {n} vs species")


def test_the_key_layout_is_species_shl10_form_shl8_size():
    ref = mod("species")
    assert ref.pack_key(21, 2, 64) == (21 << 10 | 2 << 8 | 64)
    assert ref.unpack_key(ref.pack_key(65535, 3, 255)) == (65535, 3, 255)
    # every module that shifts a key by hand uses the same shifts
    for n in MAN.modules():
        if not present(n) or n in ("species", "species_seed"):
            continue
        for m in re.finditer(r">>\s*(\d+)\)?\s*&\s*3\b", src_of(n)):
            assert m.group(1) == "8", f"{n}: form shift {m.group(0)}"


# =====================================================================================
# (9) ZONES AND REGIONS
# =====================================================================================
def test_every_zone_has_a_bone_and_a_region_and_all_11_regions_are_reachable():
    ti, an = mod("tissue"), mod("anatomy")
    zones = set(ti.ZONES)
    assert len(zones) == 14 and len(ti.ZONES) == 14
    assert set(an.REGION_OF_ZONE) == zones, set(an.REGION_OF_ZONE) ^ zones
    assert list(an.ZONES) == list(ti.ZONES)
    assert len(an.REGIONS) == 11
    reached = {an.region_of_zone(z, a) for z in ti.ZONES for a in (None, 0.1, 0.9)}
    assert reached == set(an.REGIONS), set(an.REGIONS) ^ reached
    # zone_of_bone plus the distal split reach all 14 zones
    cover = set(ti.ZONE_OF_BONE.values()) | {z for z, _f in ti.DISTAL_OF_BONE.values()}
    assert cover == zones, zones ^ cover


def test_region_bone_and_zone_maps_close_the_loop():
    ti, an = mod("tissue"), mod("anatomy")
    bones = set(ti.ZONE_OF_BONE)
    assert set(an.BONE_OF_REGION.values()) == bones, set(an.BONE_OF_REGION.values()) ^ bones
    for z in ti.ZONES:
        bone = an.BONE_OF_REGION[an.REGION_OF_ZONE[z]]
        zones_of_bone = {ti.ZONE_OF_BONE[bone]} | ({ti.DISTAL_OF_BONE[bone][0]} if bone in ti.DISTAL_OF_BONE else set())
        assert z in zones_of_bone, f"zone {z} -> region {an.REGION_OF_ZONE[z]} -> bone {bone}, whose zones are {zones_of_bone}"


def test_winter_segment_fractions_in_tissue_are_the_body_sheets():
    ti, bs = mod("tissue"), mod("bodysheet")
    w = bs.WINTER
    alias = {"torso": "trunk", "shin": "shank"}
    for z in ti.ZONES:
        kind = z.split(" ", 1)[1] if z[:4] in ("left", "righ") else z
        same("winter", ti.WINTER_SHARE[z], w[alias.get(kind, kind)], rel=1e-9, what=f"Winter fraction of {z}: tissue vs bodysheet")


# =====================================================================================
# (10) THE BOOKS
# =====================================================================================
def test_every_ledger_term_is_declared_once_with_one_owner():
    names = [b["name"] for b in MAN.BOOKS]
    assert len(names) == len(set(names)), [n for n in names if names.count(n) > 1]
    topics = {"T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11"}
    for b in MAN.BOOKS:
        assert b["topic"] in topics, b
        assert b["kind"] in ("mass", "energy", "volume", "mass+energy"), b
        for w in b["writers"] + b["via"]:
            assert w in MAN.MODULES, f"term {b['name']}: writer {w} is not a manifest module"
        if not b["shared"]:
            owner_mods = [w for w in b["writers"] if MAN.MODULES[w] != b["topic"]]
            assert not owner_mods, f"term {b['name']} owned by {b['topic']} but written by {owner_mods}"
    for need in ("gone", "shed", "recast", "quant", "f32", "squeeze", "records_owned", "outside_booked",
                 "boiled_J", "melted_J", "frozen_J", "dissolved_J", "made", "mech_in", "pool", "stores",
                 "work_out", "pooled_ml", "edema_ml", "layer_mass", "layer_enthalpy", "slow_residual"):
        assert need in names, f"the plan's term {need} is not in the manifest"


def _writes(src):
    """[(term, function, line, op)] for each += / -= / = whose target names a ledger term."""
    tree = ast.parse(src)
    terms = {t for t in MAN.BOOK_NAMES if t not in ("gone", "shed", "pool", "stores")}
    terms |= set(MAN.SOURCE_ALIASES)
    out = []

    def term_of(node):
        while True:
            if isinstance(node, ast.Subscript):
                key = node.slice
                if isinstance(key, ast.Constant) and isinstance(key.value, str) and key.value in terms:
                    return key.value
                node = node.value
            elif isinstance(node, ast.Attribute):
                if node.attr in terms:
                    return node.attr
                node = node.value
            elif isinstance(node, ast.Name):
                return node.id if node.id in terms else None
            else:
                return None

    def walk(node, fn):
        for ch in ast.iter_child_nodes(node):
            f = ch.name if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)) else fn
            if isinstance(ch, ast.AugAssign):
                t = term_of(ch.target)
                if t:
                    out.append((MAN.SOURCE_ALIASES.get(t, t), fn, ch.lineno, "aug"))
            elif isinstance(ch, ast.Assign):
                for tg in ch.targets:
                    t = term_of(tg) if isinstance(tg, (ast.Subscript, ast.Attribute)) else None
                    if t:
                        out.append((MAN.SOURCE_ALIASES.get(t, t), fn, ch.lineno, "set"))
            walk(ch, f)

    walk(tree, "<module>")
    return out


def test_no_two_modules_claim_the_same_term():
    """Static scan: a new module writes a ledger term only if it belongs to the term's owner topic or
    the manifest lists it as a `via` caller of the owner's API."""
    bad = []
    seen = {}
    for name in MAN.modules():
        if not present(name):
            continue
        for term, fn, line, _op in _writes(src_of(name)):
            if MAN.MODULES[name] != MAN.owner_of(term) and name not in MAN.book(term)["via"]:
                bad.append(f"{name}.py:{line} ({fn}) writes {term}, owner {MAN.owner_of(term)} lists via {sorted(MAN.book(term)['via'])}")
            seen.setdefault(term, set()).add(MAN.MODULES[name])
    assert not bad, "\n".join(bad)
    for term, tops in seen.items():
        b = MAN.book(term)
        assert tops <= {b["topic"]} | {MAN.MODULES[v] for v in b["via"]}, (term, tops)


def test_no_function_books_both_work_out_and_mech_in():
    """The one-flow rule, by source: no function writes both books."""
    bad = []
    for name in MAN.modules():
        if not present(name):
            continue
        per_fn = {}
        for term, fn, line, _op in _writes(src_of(name)):
            if term in ("work_out", "mech_in"):
                per_fn.setdefault(fn, set()).add(term)
        # deposit() names both on purpose: it picks ONE by whether there is an actor (tested below)
        bad += [f"{name}.{fn}" for fn, ts in per_fn.items() if len(ts) > 1 and fn != "deposit"]
    assert not bad, bad


class _W(types.SimpleNamespace):
    pass


def test_deposit_books_one_flow_only():
    dep = attr(mod("mechanisms"), "deposit")
    # no actor: mech_in, never work_out
    w = _W(mech_in=0.0, work_out=0.0)
    assert dep(w, [], 12.5) == "mech_in"
    assert (w.mech_in, w.work_out) == (12.5, 0.0)
    # an actor: the actor's stores debited, work_out credited, mech_in untouched
    w = _W(mech_in=0.0, work_out=0.0)
    actor = {}
    book = dep(w, [], 7.0, actor=actor)
    assert book == "work_out" and (w.mech_in, w.work_out) == (0.0, 7.0)
    debited = []
    w = _W(mech_in=0.0, work_out=0.0)
    credited = []
    dep(w, [], 3.0, actor=actor, debit=lambda a, J: debited.append(J), work_out=lambda J: credited.append(J))
    assert debited == [3.0] and credited == [3.0] and w.mech_in == 0.0
    with pytest.raises(ValueError):
        dep(w, [], -1.0)


def test_energetics_bills_work_out_and_never_mech_in():
    """The actor's work goes to work_out. The code of energetics must not name mech_in as a name,
    an attribute or a key (the docstring may)."""
    tree = ast.parse(src_of("energetics"))
    hits = [n.lineno for n in ast.walk(tree)
            if (isinstance(n, ast.Attribute) and n.attr == "mech_in") or (isinstance(n, ast.Name) and n.id == "mech_in")
            or (isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant) and n.slice.value == "mech_in")]
    assert not hits, f"energetics.py names mech_in at lines {hits}"
    assert "work_out" in src_of("energetics")


# =====================================================================================
# (11) ID RANGES
# =====================================================================================
def test_id_blocks_do_not_collide():
    sp, sd = mod("species"), mod("species_seed")
    assert sd.SPECIES_LEGACY_END == 17 and sd.SPECIES_SEED_END == 64
    assert (sd.GAS_LO, sd.GAS_HI, sd.PACK_LO) == (64, 96, 96)
    assert sd.SOLID_ROWS == (0, 64) and sd.FLUID_ROWS == (64, 128) and sd.MIX_BASE == 128
    assert sd.NMAT_PAD == 64 and sd.NFLUID_PAD == 64
    assert sd.SOLID_ROWS[1] == sd.FLUID_ROWS[0] and sd.FLUID_ROWS[1] == sd.MIX_BASE
    assert sd.FLUID_ROWS[1] - sd.FLUID_ROWS[0] == sd.NFLUID_PAD
    ids = [s.id for s in sp.SPECIES if s is not None]
    assert len(ids) == len(set(ids))
    assert ids == sorted(ids)
    assert [i for i in ids if i < 17] == list(range(17))
    assert max(ids) < sd.GAS_LO, "no gas species in the seed yet: 64-95 are reserved"
    names = [s.name for s in sp.SPECIES if s is not None]
    assert len(names) == len(set(names))


def test_species_zero_to_sixteen_are_the_legacy_mats_and_fluids():
    sp, sd = mod("species"), mod("species_seed")
    for m in range(0, 12):
        slot = sp.MATS[m]
        assert slot.species == m, (m, slot)
    ice = sp.MATS[12]
    assert (ice.name, sd.MAT_ICE, ice.species) == ("ice", 12, sp.BY_NAME["water"])
    assert ice.form == sd.BOUND
    for f, s in sd.LEGACY_FLUID_SPECIES.items():
        assert sp.FLUIDS[f].species == s, (f, s)
    # fluid ids 0-8 are legacy, 9 and up new, nothing between
    assert [f for f in sd.LEGACY_FLUID_SPECIES] == list(range(9))
    assert min(sd.FLUID_SLOTS) == 9
    # skeleton mats 13-31 only; padded rows are named pad_
    used = [m for m in range(sp.NMAT_PAD) if sp.MATS[m].species is not None]
    assert used == list(range(len(used))), used
    assert max(used) < 32, used
    assert all(sp.MATS[m].name.startswith("pad_") for m in range(max(used) + 1, sp.NMAT_PAD))


def test_matrec_ids_are_the_species_mat_ids():
    mr, sp = mod("matrec"), mod("species")
    assert mr.ICE == 12 == sp.MAT_ICE
    assert mr.LEGACY_NMAT == 12
    names = {m.id: m.name for m in sp.MATS if m.species is not None}
    for r in mr.BASE.rows:
        if r.id is None:
            continue
        assert names.get(r.id) == r.key or (r.key == "air" and r.id == 0), \
            f"matrec row {r.key} has mat id {r.id}, species says {names.get(r.id)}"
    ice = mr.ingest_ice()
    if ice is not None:
        assert ice.row("ice").id == 12


def test_pure_rows_and_padded_tables_have_the_declared_ranges():
    sp, sd = mod("species"), mod("species_seed")
    assert sorted(sp.pure_rows()) == list(range(sd.MIX_BASE)), "pure rows are 0..127: solids 0-63, fluids 64-127"
    t = sp.padded_tables()
    for k, v in t.items():
        if hasattr(v, "shape") and v.ndim == 1:
            assert v.shape[0] == sd.NMAT_PAD or v.shape[0] == sd.NFLUID_PAD, (k, v.shape)
    assert t["NMAT"] == sd.NMAT_PAD and t["NFLUID"] == sd.NFLUID_PAD


def test_the_t1_legacy_tables_equal_the_t9_views_on_the_twelve_old_materials():
    """Two routes to the same old tables (T1 legacy_tables, T9 matrec.legacy_views); the oracle is the
    third (lane A-O). Here they are compared with each other."""
    sp, mr = mod("species"), mod("matrec")
    t, v = sp.padded_tables(), mr.legacy_views((5, 6, 7))
    checked = 0
    for k, b in v.items():
        if k in t and hasattr(b, "shape") and getattr(b, "ndim", 0) == 1:
            n = b.shape[0]
            same("legacy_view", np.asarray(t[k][:n], dtype=float), np.asarray(b, dtype=float), rel=1e-6,
                 what=f"{k}[:{n}]: species.padded_tables vs matrec.legacy_views")
            checked += 1
    assert checked >= 8, checked
    for name in ("MELT", "FREEZE"):
        for key, row in v[name].items():
            assert t[name][key] == row, (name, key, t[name].get(key), row)


LAYER_ALIAS = {"salt": "sodium_chloride", "clay": "kaolinite", "chalk": "calcite",
               "acid": "sulfuric_acid", "weak_acid": "acetic_acid"}


def test_layer_rows_use_species_ids():
    """T3's rows are keyed by species id (ck >> 10): ids below 64 must name the same stuff."""
    ld, sp = mod("layer_data"), mod("species")
    bad = []
    for k, row in ld.ROWS.items():
        if k < 64:                  # 200+ are layer-only stuffs (blood plasma, inks, ochre)
            s = sp.SPECIES[k] if k < len(sp.SPECIES) else None
            if s is None or s.name != LAYER_ALIAS.get(row["name"], row["name"]):
                bad.append((k, row["name"], None if s is None else s.name))
    assert not bad, bad
    assert min(k for k in ld.ROWS if k >= 64) >= 200, "layer-only stuffs live at 200 and up"


def test_mix_table_rows_start_after_the_pure_rows():
    mt = mod("mixtable")
    sd = mod("species_seed")
    first = attr(mt, "FIRST_MIX", "MIX_BASE")
    assert first == sd.MIX_BASE


# =====================================================================================
# (12) MODULE HYGIENE
# =====================================================================================
def test_module_hygiene_passes_with_all_new_files_present():
    """tests/test_module_hygiene.py, run on the sandbox tree as it is now (it reads every file
    of src/voxel and every test file)."""
    spec = importlib.util.spec_from_file_location("w13_hygiene", os.path.join(HERE, "test_module_hygiene.py"))
    hy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hy)
    assert [n for n in MAN.modules() if present(n)], "no new module present"
    for fn in ("test_every_global_a_function_reads_is_BOUND_in_its_own_module",
               "test_no_module_reaches_into_its_own_GLOBALS",
               "test_a_test_PATCHES_the_module_that_READS_what_it_patches"):
        getattr(hy, fn)()


# =====================================================================================
# (13) THE CRITIC'S POINTS
# =====================================================================================
@pytest.mark.parametrize("G,D_mm,L_mm", [(0.45, 4.83, 19.0), (0.6, 3.5, 25.0), (0.35, 6.0, 30.0)])
def test_screw_withdrawal_in_joinery_and_fasteners_agree(G, D_mm, L_mm):
    jo, fa = mod("joinery"), mod("fasteners")
    same("screw", fa.screw_withdrawal(G, D_mm / 1000.0, L_mm / 1000.0), jo.screw_withdrawal_standin(G, D_mm, L_mm),
         rel=1e-12, what="screw withdrawal: fasteners vs joinery stand-in")
    same("screw", jo.screw_withdrawal(G, D_mm, L_mm), fa.screw_withdrawal(G, D_mm / 1000.0, L_mm / 1000.0),
         rel=1e-12, what="joinery.screw_withdrawal (dispatch) vs fasteners")


@pytest.mark.parametrize("G,D_mm", [(0.45, 4.83), (0.6, 3.5), (0.35, 6.0)])
def test_dowel_mode_iv_formula_is_the_same_when_Rd_is_the_same(G, D_mm):
    """Same formula Z = (D^2/Rd) sqrt(2 Fe Fyb/3), Fe = 16600 G^1.84 psi."""
    jo, fa = mod("joinery"), mod("fasteners")
    Rd = 3.2
    z_f = fa.dowel_lateral_yield(G, D_mm / 1000.0, 1.0, Rd=Rd)       # a long screw: mode IV governs
    z_j = jo.dowel_lateral_yield_standin(G, D_mm, Rd=Rd)
    # both read Fe as 16600 G^1.84 psi times the one PSI (joinery used to type 114.45 MPa, 1.3e-5 off)
    same("dowel_formula", z_j, z_f, rel=1e-9, what="mode IV yield at equal Rd: joinery vs fasteners")


@pytest.mark.parametrize("G,D_mm", [(0.45, 4.83), (0.6, 3.5)])
def test_dowel_mode_iv_default_Rd_agrees(G, D_mm):
    """Decision N6: fasteners.py had the two ends of the NDS reduction term swapped (3.2 under 1/4 in,
    2.2 above); it now reads the table as printed (NDS-2018 Table 12.3.1B), the same function
    joinery uses. This pair used to differ (joinery 33 percent above fasteners at D 4.83 mm, 45
    percent at 3.5 mm: too much for any known-difference bound); there is no such entry any more,
    so a new disagreement fails."""
    jo, fa = mod("joinery"), mod("fasteners")
    same("dowel_rd_default", jo.dowel_lateral_yield_standin(G, D_mm), fa.dowel_lateral_yield(G, D_mm / 1000.0, 1.0),
         rel=1e-6, what="mode IV yield, each at its own default Rd")


def test_dowel_Rd_is_ONE_function_in_fasteners_record_data_and_joinery():
    """The reduction term is typed once, in fasteners.py; record_data and joinery hold the same
    function object, so they cannot drift (they used to be two copies, one with the ends swapped)."""
    fa, rd, jo = mod("fasteners"), mod("record_data"), mod("joinery")
    assert rd.dowel_Rd is fa.dowel_Rd and jo.dowel_Rd is fa.dowel_Rd
    assert rd.INCH == fa.INCH


@pytest.mark.parametrize("G", [0.35, 0.45, 0.6])
def test_dowel_mode_iv_default_Rd_agrees_over_every_screw_diameter(G):
    """The default yield of joinery's stand-in and of fasteners agree for every diameter of a screw
    or a nail (2 to 6.3 mm, under 1/4 in), across both breaks of K_D (0.17 in = 4.32 mm). From 1/4
    in fasteners.py switches to the bolt embedment 11200 G psi and the stand-in does not: that is a
    different row, not Rd (the stand-in is only ever asked for screws and nails)."""
    jo, fa = mod("joinery"), mod("fasteners")
    D = np.linspace(2.0, 6.3, 87)
    same("dowel_rd_default",
         [jo.dowel_lateral_yield_standin(G, d) for d in D],
         [fa.dowel_lateral_yield(G, d / 1000.0, 1.0) for d in D],
         rel=1e-9, what="mode IV yield over D 2 to 6.3 mm, each at its own default Rd")


def test_nail_withdrawal_row_is_the_same_family_as_the_screw_row():
    """Wood Handbook: nail 54.12 G^2.5 D, screw 108.25 G^2 D: at G = 0.5 the screw row per mm
    of thread is 108.25 * 0.25 = 27.06 vs 54.12 * 0.1768 = 9.57 per mm of nail (the screw
    holds 2.8 times a smooth nail, the classic ratio)."""
    jo, fa = mod("joinery"), mod("fasteners")
    ratio = fa.screw_withdrawal(0.5, 3e-3, 20e-3) / jo.nail_withdrawal(0.5, 3.0, 20.0)
    assert 2.0 < ratio < 4.0, ratio
    same("nail", jo.nail_withdrawal(0.5, 3.3, 38.0), 1199.7, rel=2e-4, what="the 1.2 kN nail")


@pytest.mark.parametrize("T", [5.0, 20.0, 50.0])
def test_water_viscosity_andrade_in_species_and_vogel_in_layers_agree(T):
    sp, ll = mod("species"), mod("layer_laws")
    w = sp.get("water")
    A, B = w.get("mu_A", "ref"), w.get("mu_B", "ref")
    andrade = A * math.exp(B / (T + 273.15))                 # mPa s
    vogel = ll.viscosity_water(T) * 1e3
    tol = 2e-3 if T == 20.0 else 0.015 if T == 50.0 else 0.12      # Andrade is fitted at 20 and 50 C only
    same("visc", andrade, vogel, rel=tol, what=f"water viscosity at {T} C (mPa s): species Andrade vs layer_laws Vogel")


def test_water_row_viscosity_in_layer_data_is_the_species_value_at_20C():
    ld, sp = mod("layer_data"), mod("species")
    A, B = sp.get("water").get("mu_A", "ref"), sp.get("water").get("mu_B", "ref")
    same("visc", ld.ROWS[12]["eta20"], A * math.exp(B / 293.15), rel=2e-3, what="eta20 of the layer water row vs species Andrade at 20 C")


def test_nacl_and_sucrose_mu_rel_tables_are_increasing_and_start_at_one():
    sp = mod("species")
    for n in ("sodium_chloride", "sucrose"):
        t = sp.get(n).get("mu_rel", "ref")
        x = [p[0] for p in t]
        y = [p[1] for p in t]
        assert x[0] == 0.0 and y[0] == 1.0 and x == sorted(x) and y == sorted(y), n


def test_matrec_blend_has_no_thermal_code_and_asks_mixrules():
    """Thermal numbers (density, c, k, viscosity) come from the mixrules registry. blend() must
    not hold a conductivity or heat-capacity law: static check on its source, and a behaviour
    check with a fake rule set that returns sentinels."""
    mr = mod("matrec")
    tree = ast.parse(src_of("matrec"))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "blend")
    # attribute reads of .k inside blend: only dom.k (the stand-in) and r.k handed to the rule in a dict
    ks = [n for n in ast.walk(fn) if isinstance(n, ast.Attribute) and n.attr == "k"]
    allowed = 0
    for n in ks:
        if isinstance(n.value, ast.Name) and n.value.id in ("dom", "r"):
            allowed += 1
    assert allowed == len(ks) and len(ks) <= 2, f"blend reads .k outside the stand-in and the rule hand-over: {len(ks)}"
    text = ast.get_source_segment(src_of("matrec"), fn)
    for forbidden in ("maxwell", "eucken", "harmonic", "series", "parallel", "wiener", "hashin", "bruggeman"):
        assert forbidden not in text.lower(), f"thermal mixing law ({forbidden}) inside blend"
    rec = mr.REC
    parts = [(rec["wood"], 0.5, "matrix"), (rec["iron"], 0.5, "grain")]
    # the REAL interface (mixrules.RULES[kind](Entries, slot) -> array, registry kinds rho/c/k/mu,
    # c in J/g/K): a fake that answers with sentinels and records what it was asked
    asked = []

    def fake(v):
        def rule(ent, slot):
            asked.append((ent, slot))
            return np.array([v])
        return rule

    rules = types.SimpleNamespace(
        make_entries=lambda n, cell, key, frac: ("entries", n, tuple(key), tuple(frac)),
        RULES={"rho": fake(777.0), "c": fake(1.25), "k": fake(0.4242), "mu": fake(9.0)})
    out = mr.blend(parts, rules)
    assert (out.rho, out.cp, out.k) == (777.0, 1250.0, 0.4242)       # c: J/g/K -> J/kg/K
    assert out.extra.get("viscosity") == 9.0
    assert len(asked) == 4 and all(a[0][0] == "entries" for a in asked), "blend did not ask all four rules"
    out2 = mr.blend(parts, types.SimpleNamespace(RULES={}))
    assert out2.extra.get("k_source", "").startswith("standin")


def test_matrec_blend_thermal_numbers_equal_the_mixrules_ones_when_it_exists():
    """The real mixrules run by hand on an Entries row (species key, form word, mass fraction)
    gives the same rho, c and k as blend. Random sweep over pairs of species-records and forms,
    so a wrong form map or a wrong unit shows up on more than the pair a person picked."""
    mr, mx, sp = mod("matrec"), mod("mixrules"), mod("species")
    names = [n for n in ("wood", "stone", "iron", "tin", "copper", "ice", "glass") if n in mr.REC
             and mr.REC[n].key in sp.BY_NAME]
    assert len(names) >= 3, names
    word = {"crystal": "crystal", "grain": "grain", "matrix": "bound", "bound": "bound"}
    rng = np.random.default_rng(13)
    tested = 0
    for _ in range(40):
        a, b = rng.choice(names, 2, replace=False)
        fa, fb = rng.choice(list(word), 2)
        wa = float(rng.uniform(0.05, 0.95))
        parts = [(mr.REC[a], wa, str(fa)), (mr.REC[b], 1 - wa, str(fb))]
        out = mr.blend(parts, mx)
        if "k_source" in out.extra:      # the record is not exactly its species: stand-in, tested elsewhere
            continue
        rows = sorted((int(sp.pack_key(sp.BY_NAME[r.key], sp.FORM_NAMES.index(word[f]))), w)
                      for r, w, f in parts)
        ent = mx.make_entries(1, np.zeros(2, np.int64), np.array([k for k, _ in rows], np.uint32),
                              np.array([w for _, w in rows]))
        same("blend_thermal", out.rho, float(mx.RULES["rho"](ent, "engine")[0]), rel=1e-12, what="blend rho vs mixrules")
        same("blend_thermal", out.cp, 1000.0 * float(mx.RULES["c"](ent, "engine")[0]), rel=1e-12, what="blend cp vs mixrules c")
        same("blend_thermal", out.k, float(mx.RULES["k"](ent, "engine")[0]), rel=1e-12, what="blend k vs mixrules k")
        tested += 1
    assert tested >= 10, "too few random pairs went through mixrules"


@pytest.mark.parametrize("who", ["tissue", "anatomy", "mechanisms"])
def test_body_numbers_come_from_bodysheet(who):
    m = mod(who)
    bs = mod("bodysheet")
    text = src_of(who)
    if who == "mechanisms" and not re.search(r"^[^#\n]*\b5\.6\b", text, re.M):
        pytest.skip("mechanisms.py types no body number (the kick 5.6 kg / 6 m/s is an argument): bodysheet is read at wiring")
    assert re.search(r"from \. import bodysheet|from \.bodysheet import|import bodysheet", text), \
        f"{who}.py does not import bodysheet"
    bound = [v for v in vars(m).values() if v is bs]
    assert bound or any(getattr(v, "__module__", None) == bs.__name__ for v in vars(m).values()), \
        f"{who} imports bodysheet but the name is not bound: the stand-in is running"


def test_anatomy_segment_fractions_are_winters_from_bodysheet():
    an, bs = mod("anatomy"), mod("bodysheet")
    w = bs.WINTER
    frac = getattr(an, "REGION_FRAC", None)
    if frac is None:
        pytest.skip("anatomy.REGION_FRAC missing")
    same("winter", frac["chest"] + frac["abdomen"], w["trunk"], rel=1e-9, what="chest + abdomen vs bodysheet trunk")
    same("winter", frac["head"], w["head"], rel=1e-9, what="head vs bodysheet")
    same("winter", sum(frac.values()), 1.0, rel=1e-9, what="region fractions sum")


# =====================================================================================
# (13) SHARED CONSTANTS: Lv, THE NaCl EUTECTIC, R_GAS, SIGMA
# =====================================================================================
CODATA_R = 8.314462618       # J/(mol K), CODATA 2018 exact (N_A k_B)
CODATA_SIGMA = 5.670374419e-8  # W/(m2 K4), CODATA 2018 exact


def _module_literals(names):
    """{module: {name: value}} for module-level assignments of a plain number to one of `names`,
    in every src/voxel file (read by the AST: nothing is imported, a non-literal is left out)."""
    out = {}
    for f in sorted(os.listdir(VOX)):
        if not f.endswith(".py"):
            continue
        try:
            tree = ast.parse(open(os.path.join(VOX, f)).read())
        except SyntaxError:
            continue
        for n in tree.body:
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) \
                    and n.targets[0].id in names:
                try:
                    v = ast.literal_eval(n.value)
                except (ValueError, SyntaxError):
                    continue
                if isinstance(v, (int, float)):
                    out.setdefault(f[:-3], {})[n.targets[0].id] = float(v)
    return out


def test_the_gas_constant_is_the_codata_value_in_every_module():
    """R_GAS is typed in several modules. The value is exact (CODATA 2018); a module may round it
    (skinburn types 8.314, 5.6e-5 off) but never differ by more than 1e-4."""
    found = _module_literals({"R_GAS"})
    assert len(found) >= 3, f"expected R_GAS in several modules, found {sorted(found)}"
    for m, d in found.items():
        same("r_gas", d["R_GAS"], CODATA_R, rel=1e-4, what=f"{m}.R_GAS vs CODATA")


def test_the_gas_constant_named_R_in_matterdata_is_codata():
    md = _module_literals({"R"})
    if "matterdata" not in md:
        pytest.skip("matterdata.R absent")
    same("r_gas", md["matterdata"]["R"], CODATA_R, rel=1e-9, what="matterdata.R vs CODATA")


def test_stefan_boltzmann_is_the_codata_value_in_every_module():
    found = _module_literals({"SIGMA", "SIGMA_SB"})
    assert len(found) >= 4, f"expected SIGMA in several modules, found {sorted(found)}"
    for m, d in found.items():
        for k, v in d.items():
            same("sigma", v, CODATA_SIGMA, rel=1e-6, what=f"{m}.{k} vs CODATA")


def test_NaCl_eutectic_is_one_point_in_mixdata_and_layer_data():
    """-21.1 C / 23.3 wt% (mixdata, CRC, one decimal) and -21.12 C / 0.2331 (layer_data, sourced
    liquidus): the same point to the rounding of the first."""
    md, ld = mod("mixdata"), mod("layer_data")
    (t_m, w_m), _src = md.TABLES["eutectic_ref"]["nacl"]
    same("nacl_eutectic", t_m, ld.T_EUTECTIC, rel=0.0, absol=0.05, what="eutectic T (C): mixdata vs layer_data")
    same("nacl_eutectic", w_m / 100.0, ld.WT_EUTECTIC, rel=0.0, absol=0.0005,
         what="eutectic mass fraction: mixdata (wt%/100) vs layer_data")
    same("nacl_eutectic", ld.LIQUIDUS_T[-1], ld.T_EUTECTIC, what="layer_data liquidus ends at the eutectic")
    same("nacl_eutectic", ld.LIQUIDUS_WT[-1] / 100.0, ld.WT_EUTECTIC, rel=1e-9,
         what="layer_data liquidus ends at the eutectic composition")


def test_latent_heat_of_vaporisation_agrees_at_100C_to_half_a_percent():
    """Water Lv at 1 atm, 100 C is 2257 J/g (CRC 2256.4, IAPWS 2256.5). The sim's LATENT is the
    rounded 2260; bodyair's body-range line (2501 - 2.37 T) gives 2264 there. The spread is
    0.3 percent and must stay under 0.5 (an evaporation budget is a heat budget: 0.5 percent of
    the heat of a boil-off is below the ledger's own tolerance)."""
    sp = mod("species")
    vals = {"species.Lv": float(sp.get("water").get("Lv", "ref")),
            "mixdata.LATENT_SIM": float(mod("mixdata").LATENT_SIM)}
    vals["bodyair.latent_heat(100)"] = float(mod("bodyair").latent_heat(100.0)) / 1000.0
    if present("record_data"):
        m = re.search(r"\(100\.0,\s*([0-9.]+)e3", src_of("record_data"))
        if m:
            vals["record_data.water latent"] = float(m.group(1))
    if present("materials"):
        m = re.search(r"BOIL,\s*LATENT\s*=\s*100\.0,\s*([0-9.]+)", open(os.path.join(ROOT, "src", "materials.py")).read()) \
            if os.path.exists(os.path.join(ROOT, "src", "materials.py")) else None
        if m:
            vals["materials.LATENT"] = float(m.group(1))
    ref = vals["species.Lv"]
    assert ref == 2257.0, ref
    for who, v in vals.items():
        same("lv", v, ref, rel=5e-3, what=f"Lv at 100 C J/g: {who} vs species")
    same("lv", vals["mixdata.LATENT_SIM"], 2260.0, rel=0.0, what="mixdata.LATENT_SIM is the engine's 2260")


def test_latent_heat_at_ambient_is_the_bodyair_line_not_the_engine_boil_value():
    """slowforms types 2450 J/g as the default latent of evaporation near 20 C; bodyair's line gives
    2453.6 there. The engine's 2260 is the value AT 100 C only: a use at ambient needs the line."""
    ba, sf = mod("bodyair"), mod("slowforms")
    m = re.search(r'"latent_J_g",\s*([0-9.]+)', src_of("slowforms"))
    assert m, "slowforms latent default not found"
    same("lv_ambient", float(m.group(1)), float(ba.latent_heat(20.0)) / 1000.0, rel=5e-3,
         what="slowforms default latent (J/g) vs bodyair.latent_heat(20 C)")
    assert abs(float(ba.latent_heat(100.0)) / 1000.0 - 2257.0) < 10.0, "the line stays near Lv at 100 C"


# =====================================================================================
# (14) THE EXISTING TESTS, UNCHANGED
# =====================================================================================
def _existing_test_files():
    base = os.path.join(MARK, "base19", "tests")
    return sorted("tests/" + f for f in os.listdir(base) if f.startswith("test_") and f.endswith(".py"))


def _summary(text):
    last = text.strip().splitlines()[-1] if text.strip() else ""
    return {("errors" if k.startswith("error") else k): int(n)
            for n, k in re.findall(r"(\d+) (passed|failed|errors?|skipped|deselected)", last)}


def _pytest_fast(tree_root, files):
    pre = []
    lock = os.path.join(MARK, "test.lock")
    cap = os.path.join(MARK, "cap.sh")
    if os.path.exists(lock):
        pre += ["flock", "-w", "6000", lock]
    if os.path.exists(cap):
        pre += [cap]
    cmd = pre + [sys.executable, "-B", "-m", "pytest", "-q", "-m", "not slow", "-p", "no:cacheprovider"] + files
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=tree_root)
    return _summary(r.stdout), r.stdout[-1500:]


def _log_summary(path):
    try:
        return _summary(open(path).read())
    except OSError:
        return {}


@pytest.mark.slow
def test_the_fast_tier_of_the_existing_tests_is_unchanged_from_base19():
    """Every test file that exists in base19 (the tree after wave 12), fast tier (-m "not slow"), in the
    base19 snapshot and in this tree: the pass and fail counts must be equal. (Re-based from base18 in the
    13a repair: the old logs compared the phase A sandbox with base18 and went stale at the merge.)

    Default: read the two logs w13b/REPAIR/{base19,live}_fast.log. If a log is missing, run that side now
    (under flock and cap.sh). W13_SKIP_FAST=1 skips this case."""
    if os.environ.get("W13_SKIP_FAST"):
        pytest.skip("W13_SKIP_FAST set: the fast-tier comparison is run by tools/w13_audit.py --fast")
    work = os.path.join(MARK, "w13b", "REPAIR")
    base = _log_summary(os.path.join(work, "base19_fast.log"))
    mine = _log_summary(os.path.join(work, "live_fast.log"))
    copy = os.path.join(MARK, "base19")
    if not base.get("passed"):
        if not os.path.isdir(copy):
            pytest.skip(f"no base19 copy at {copy} and no log")
        base, tail = _pytest_fast(copy, _existing_test_files())
        assert base.get("passed"), tail
    if not mine.get("passed"):
        mine, tail = _pytest_fast(ROOT, _existing_test_files())
    assert mine.get("passed") == base.get("passed") and mine.get("failed", 0) == base.get("failed", 0) \
        and mine.get("errors", 0) == base.get("errors", 0), f"live {mine} vs base19 {base}"


# =====================================================================================
# (15) THE EXISTING TESTS THAT LIST THE SOURCE TREE
# =====================================================================================
def test_existing_tests_that_list_src_still_pass_with_the_new_files():
    """A new module can break an old test only by being seen by it: a size budget, a hygiene scan, an
    "only these modules read X" scan. Every existing test that lists the tree runs here. A failure that
    needs a phase B edit of the old test is in tools/w13_manifest.py KNOWN_PHASE_B and is reported as
    xfail with the fix; any other failure is a real break by a new file."""
    base = os.path.join(MARK, "base18")
    if not os.path.isdir(base):
        pytest.skip("no base18 tree")
    ids = MAN.source_globbing_tests(base)
    assert len(ids) >= 10, ids
    pre = [os.path.join(MARK, "cap.sh")] if os.path.exists(os.path.join(MARK, "cap.sh")) else []
    r = subprocess.run(pre + [sys.executable, "-B", "-m", "pytest", "-q", "-rfE", "-m", "not slow",
                              "-p", "no:cacheprovider"] + ids,
                       capture_output=True, text=True, cwd=ROOT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    failed = {re.split(r"\s+-\s+", ln.split(None, 1)[1])[0].strip()
              for ln in r.stdout.splitlines() if ln.startswith(("FAILED", "ERROR")) and len(ln.split(None, 1)) > 1}
    unknown = sorted(f for f in failed if f not in MAN.KNOWN_PHASE_B)
    assert not unknown, f"a new file breaks an old test: {unknown}\n{r.stdout[-1500:]}"
    stale = sorted(f for f in MAN.KNOWN_PHASE_B if f in ids and f not in failed)
    assert not stale, f"KNOWN_PHASE_B entry now passes; remove it: {stale}"
    known = sorted(failed)
    if known:
        pytest.xfail("KNOWN PHASE B EDIT of an existing test: "
                     + "; ".join(f"{k}: {MAN.KNOWN_PHASE_B[k]}" for k in known))
