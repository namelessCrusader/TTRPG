"""SOLUTIONS (wave 13, T4, law L14): solubility, Henry, pH, sorption, soil water.

Small tables with one formula each, pure numpy, no World. `comp` is a float
array of grams with the species on the last axis (mixdata.ORDER). Temperature
in C. Nothing here moves matter: dissolved_target tells where equilibrium IS
(a TARGET); the rate to get there is L4's, and the energy-coupled solve for a
dissolving crystal is mixphase.phase_step.
"""
import numpy as np

from . import mixdata as D
from .mixwater import prep

_IW = D.INDEX["h2o"]
_IE = D.INDEX["ethanol"]


# -- solubility ------------------------------------------------------------------
_SOLC = {}


def _sol_curve(name):
    """(T knots, ln s knots) for a solubility table, with two far knots on the end
    slopes so one np.interp extends the curve log-linearly; None if no row. One
    point means constant in T. NaCl gets its hydrohalite line below 0.1 C."""
    key = (name, D.revision())
    if key in _SOLC:
        return _SOLC[key]
    t = D.TABLES["solubility"].get(name)
    p = D.TABLES["solubility_point"].get(name)
    if t is not None:
        ts, ls = np.array(D.SOL_T), np.log(np.array(t[0]))
        xs = np.concatenate([[-400.0], ts, [1000.0]])
        xl = np.concatenate([[ls[0] + (ls[1] - ls[0]) / (ts[1] - ts[0]) * (-400.0 - ts[0])], ls,
                             [ls[-1] + (ls[-1] - ls[-2]) / (ts[-1] - ts[-2]) * (1000.0 - ts[-1])]])
        out = (xs, xl)
    elif p is not None:
        out = (np.array([-400.0, 1000.0]), np.log(np.array([p[0][1], p[0][1]])))
    else:
        out = None
    _SOLC[key] = out
    return out


def has_solubility(name):
    return _sol_curve(name) is not None


def solubility_water(name, T):
    """Grams of solute per 100 g water at T (C): linear in ln s against T between
    the table points, extended log-linearly on the last two points (no clamp: a
    clamp would make candy sugar crystallise above 83 wt percent). NaCl below
    0.1 C follows the hydrohalite line."""
    c = _sol_curve(name)
    if c is None:
        raise KeyError(f"no solubility row for {name}")
    T = np.asarray(T, dtype=float)
    s = np.exp(np.interp(T, c[0], c[1]))
    if name == "nacl":
        (t0, s0), (t1, s1) = D.row("hydrohalite", "nacl")
        s = np.where(T < t1, s0 + (s1 - s0) / (t1 - t0) * (T - t0), s)
    return s


def solubility(name, T, cosolvent=0.0):
    """Solubility in g per 100 g SOLVENT at T (C). `cosolvent` is the ethanol
    mass fraction of the solvent f: ln s = (1-f) ln s_water + f ln s_ethanol
    (ethanol values at 25 C, constant in T). A solute with no ethanol row is
    unaffected by the cosolvent (flagged in coverage-style notes)."""
    sw = solubility_water(name, T)
    f = np.asarray(cosolvent, dtype=float)
    e = D.TABLES["solubility_ethanol"].get(name)
    if e is None:
        return sw
    return np.exp((1.0 - f) * np.log(sw) + f * np.log(e[0]))


def saturation_grams(comp, T):
    """(...,S) grams each species could hold dissolved at T in the solvent of this
    cell (water plus any ethanol); np.inf where the species has no solubility row."""
    comp = prep(comp)
    T = np.asarray(T, dtype=float)
    W = comp[..., _IW]
    E = comp[..., _IE]
    solv = W + E
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.where(solv > 0.0, E / np.where(solv > 0.0, solv, 1.0), 0.0)
    out = np.full(comp.shape, np.inf)
    for name in D.ORDER:
        if name in ("h2o", "ethanol") or not has_solubility(name):
            continue
        out[..., D.INDEX[name]] = solubility(name, T, f) * solv / 100.0
    return out


def dissolved_target(comp_dissolved, comp_crystal, T):
    """(dissolved_eq, crystal_eq, supersaturation): where equilibrium is. For each
    species with a solubility row dissolved_eq = min(total, saturation) and
    crystal_eq = total - dissolved_eq, so the two sum to the total (to one rounding
    step). Species without a row keep their forms. supersaturation = dissolved /
    saturation of the INPUT state (0 where there is no limit)."""
    d = prep(comp_dissolved)
    c = prep(comp_crystal)
    sat = saturation_grams(d, T)
    lim = np.isfinite(sat)
    tot = d + c
    deq = np.where(lim, np.minimum(tot, sat), d)
    ceq = np.where(lim, tot - deq, c)
    with np.errstate(invalid="ignore", divide="ignore"):
        ss = np.where(lim, d / np.where(lim, np.maximum(sat, 1e-300), 1.0), 0.0)
    return deq, ceq, ss


def dissolved_split(comp, T):
    """dissolved_target for crystal-free input (stand-alone use): (dissolved, crystal)."""
    comp = prep(comp)
    deq, ceq, _ = dissolved_target(comp, np.zeros_like(comp), T)
    return deq, ceq


def supersaturation(comp, T):
    """(...,S) dissolved / saturation for each limited species."""
    return dissolved_target(comp, np.zeros_like(prep(comp)), T)[2]


def heat_of_solution(name):
    """J per gram dissolved, + = endothermic (the solution cools)."""
    e = D.TABLES["hsol"].get(name)
    return 0.0 if e is None else e[0]


def _hsol_vec():
    return np.array([heat_of_solution(n) for n in D.ORDER])


def heat_of_split(d_old, d_new):
    """Joules ABSORBED (+) by the change in dissolved grams: sum hsol * (d_new - d_old)."""
    return ((prep(d_new) - prep(d_old)) * _hsol_vec()).sum(axis=-1)


_NUC = {}


def _nuc_setup():
    rev = D.revision()
    n = _NUC.get("n")
    if n is None or n[0] != rev:
        lim = [i for i, nm in enumerate(D.ORDER) if nm not in ("h2o", "ethanol") and has_solubility(nm)]
        n = (rev, lim, np.array([D.SPECIES[x].nucleation_ratio for x in D.ORDER]), _hsol_vec())
        _NUC["n"] = n
    return n


def nucleate_cols(d, c, cols, T):
    """nucleate on a column subset: d, c are (N,K) holding the species `cols`
    (global indices, ascending). The boil step compresses its arrays to the few
    species a cell holds; this is the same rule on those columns."""
    _rev, lim, nuc, hs = _nuc_setup()
    pos = {int(g): j for j, g in enumerate(cols)}
    present = [(i, pos[i]) for i in lim if i in pos and d[..., pos[i]].any()]
    if not present:
        return d, c, np.zeros(d.shape[:-1])
    T = np.asarray(T, dtype=float)
    iw, ie = pos.get(_IW), pos.get(_IE)
    solv = (d[..., iw] if iw is not None else 0.0) + (d[..., ie] if ie is not None else 0.0)
    f = 0.0
    if ie is not None and d[..., ie].any():
        with np.errstate(invalid="ignore", divide="ignore"):
            f = np.where(solv > 0.0, d[..., ie] / np.where(solv > 0.0, solv, 1.0), 0.0)
    d2 = d.copy()
    c2 = c.copy()
    released = np.zeros(d.shape[:-1])
    for i, j in present:
        sat = solubility(D.ORDER[i], T, f) * solv / 100.0
        over = d[..., j] > nuc[i] * sat
        moved = np.where(over, d[..., j] - sat, 0.0)
        d2[..., j] = d[..., j] - moved
        c2[..., j] = c[..., j] + moved
        released = released + moved * hs[i]
    return d2, c2, released


def nucleate(dissolved, crystal, T):
    """The boil step's crystal rule. Where dissolved_i exceeds nucleation_ratio_i
    times saturation, everything above saturation (ratio 1) becomes crystal.
    Returns (dissolved, crystal, released_J): the heat the crystallising gives
    back (+). Mass is conserved exactly (the crystal gets the difference)."""
    d = prep(dissolved)
    c = prep(crystal)
    return nucleate_cols(d, c, np.arange(d.shape[-1]), T)


# -- gases in water (Henry) -----------------------------------------------------------
def henry(gas, T, p, salt_molal=0.0):
    """Dissolved gas, mol per m3 of water: c = kH(T) p, kH(T) = kH25 exp(dlnk
    (1/T - 1/298.15)); `salting out`: log10(c0/c) = Ks c_salt (Ks in L/mol, c_salt
    taken as the molality). p in Pa."""
    k25, dl = D.row("henry", gas)
    Tk = np.asarray(T, dtype=float) + D.T0K
    kh = k25 * np.exp(dl * (1.0 / Tk - 1.0 / 298.15))
    ks = D.row("salting_out", "ks_l_per_mol")
    return kh * np.asarray(p, dtype=float) * 10.0 ** (-ks * np.asarray(salt_molal, dtype=float))


# -- pH --------------------------------------------------------------------------------
def pkw(T):
    t = sorted(D.TABLES["pkw"])
    return np.interp(np.asarray(T, dtype=float), t, [D.TABLES["pkw"][k][0] for k in t])


_ACIDS = ("acetic", "lactic", "citric", "h2so4")        # whole acids, by species
_CARBONATES = ("co2", "k2co3", "na2co3")
_ONE_CAT = ("naoh", "koh")
_TWO_CAT = ("caoh2", "k2co3", "na2co3")
_SPECTATORS = ("nacl", "kcl")
_PH_ITERS = 60


def _gam(z, I):
    """Davies activity coefficient of an ion of charge magnitude z."""
    sI = np.sqrt(I)
    return 10.0 ** (-D.row("davies", "A") * z * z * (sI / (1.0 + sI) - 0.3 * I))


def ph(comp, T=25.0):
    """(pH, I, ok) by charge balance on the hydrogen activity: sum cations - sum
    anions + [H+] - [OH-] = 0, Davies activity coefficients, 60 bisections on
    log10 a_H in [-15, 1], 4 passes for the ionic strength I. ok is False above
    I = 0.5 M. Concentrations are taken per kg of water. pKa slopes with T ignored."""
    comp = prep(comp)
    lead = comp.shape[:-1]
    c = comp.reshape(-1, comp.shape[-1])
    N = c.shape[0]
    Tf = np.broadcast_to(np.asarray(T, dtype=float), lead).reshape(-1)
    kg = np.maximum(c[:, _IW], 1e-300) / 1000.0
    m = lambda name: c[:, D.INDEX[name]] / D.SPECIES[name].M / kg
    acid_set = []                               # (total mol/kg, pKa tuple)
    for a in _ACIDS:
        acid_set.append((m(a), D.row("pka", a)))
    carb = sum(m(x) for x in _CARBONATES)
    acid_set.append((carb, D.row("pka", "co2")))
    cat_fixed = sum(m(x) for x in _ONE_CAT) + 2.0 * sum(m(x) for x in _TWO_CAT)
    spect = sum(m(x) for x in _SPECTATORS)
    # bases that bring cations: the cation of k2co3 / na2co3 is 2 per mole (counted above)
    kw = 10.0 ** (-pkw(Tf))
    lo = np.full(N, -15.0)
    hi = np.full(N, 1.0)
    I = np.zeros(N)
    out_I = I
    for _ in range(4):
        lo = np.full(N, -15.0)
        hi = np.full(N, 1.0)
        g1 = _gam(1.0, I)
        for _b in range(_PH_ITERS):
            x = 0.5 * (lo + hi)
            f, _anI = _balance(x, I, g1, kw, acid_set, cat_fixed)
            hi = np.where(f > 0.0, x, hi)
            lo = np.where(f > 0.0, lo, x)
        x = 0.5 * (lo + hi)
        _f, anI = _balance(x, I, g1, kw, acid_set, cat_fixed)
        h = 10.0 ** x
        Hp = h / g1
        OHm = kw / (h * g1)
        # ionic strength: all ions, 1/2 sum c z^2
        cat_z2 = sum(m(x_) for x_ in _ONE_CAT) + 4.0 * m("caoh2") + 2.0 * (m("k2co3") + m("na2co3"))
        I = 0.5 * (Hp + OHm + cat_z2 + anI + 2.0 * spect)
        out_I = I
    pH = -x
    return pH.reshape(lead), out_I.reshape(lead), (out_I <= 0.5).reshape(lead)


def _balance(x, I, g1, kw, acid_set, cat_fixed):
    """f = cations + [H+] - [OH-] - anion charge; also sum c z^2 of the acid anions."""
    h = 10.0 ** x
    anion = np.zeros_like(x)
    anI = np.zeros_like(x)
    for tot, pkas in acid_set:
        n = len(pkas)
        logw = [np.zeros_like(x)]
        for j in range(1, n + 1):
            gprev = _gam(float(j - 1), I) if j > 1 else np.ones_like(I)
            gj = _gam(float(j), I)
            logw.append(logw[-1] + (-pkas[j - 1] - x) + np.log10(gprev / gj))
        mx = np.maximum.reduce(logw)
        w = [10.0 ** (lw - mx) for lw in logw]
        s = sum(w)
        z1 = sum(j * w[j] for j in range(1, n + 1)) / s
        z2 = sum(j * j * w[j] for j in range(1, n + 1)) / s
        anion = anion + tot * z1
        anI = anI + tot * z2
    f = cat_fixed + h / g1 - kw / (h * g1) - anion
    return f, anI


# -- water in solids: sorption (GAB), potential, soil retention -------------------------
def _gab(host):
    if host not in ("wood",):
        raise KeyError(f"no sorption isotherm for {host!r} yet (grain, hide, bread are reserved)")
    return D.row("gab", host)


def _t_mult(T):
    return 1.0 + D.row("gab", "wood_t_coef") * (np.asarray(T, dtype=float) - 20.0)


def moisture_at_aw(host, aw, T=20.0):
    """Equilibrium moisture, g water per g dry host (GAB):
    w = wm C K aw / ((1 - K aw)(1 - K aw + C K aw)), times 1 + coef (T - 20)."""
    wm, C, K = _gab(host)
    a = np.asarray(aw, dtype=float)
    x = K * a
    return wm * C * x / ((1.0 - x) * (1.0 - x + C * x)) * _t_mult(T)


def aw_at_moisture(host, w, T=20.0):
    """Inverse of moisture_at_aw (closed form: a quadratic in K aw)."""
    wm, C, K = _gab(host)
    w = np.asarray(w, dtype=float) / _t_mult(T)
    b = wm * C - w * (C - 2.0)
    x = (-b + np.sqrt(b * b + 4.0 * (C - 1.0) * w * w)) / (2.0 * (C - 1.0) * np.maximum(w, 1e-300))
    return np.where(w > 0.0, x / K, 0.0)


def psi_from_aw(aw, T=20.0):
    """Water potential, Pa: psi = (R T / Vw) ln aw."""
    return D.R_GAS * (np.asarray(T, dtype=float) + D.T0K) / D.VW * np.log(np.asarray(aw, dtype=float))


def aw_from_psi(psi, T=20.0):
    return np.exp(np.asarray(psi, dtype=float) * D.VW / (D.R_GAS * (np.asarray(T, dtype=float) + D.T0K)))


_CM_WATER_PA = 98.0665          # 1 cm of water column, Pa


def retention(texture, psi):
    """van Genuchten water content theta (volume fraction) of a soil texture at
    water potential psi (Pa, negative): theta = theta_r + (theta_s - theta_r)
    [1 + (alpha h)^n]^-(1 - 1/n), h = -psi in cm of water."""
    tr, ts, alpha, n = D.row("vangenuchten", texture)
    h = np.maximum(-np.asarray(psi, dtype=float) / _CM_WATER_PA, 0.0)
    return tr + (ts - tr) * (1.0 + (alpha * h) ** n) ** (-(1.0 - 1.0 / n))
