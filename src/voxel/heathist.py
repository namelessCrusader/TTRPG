"""HEAT HISTORY (wave 15, lane 1e): ONE path-functional state law for steel.

What a bar is now depends on how it got hot and cooled, not on what it is at this moment. A
sample (a voxel cell, or a node of a slab) keeps a SMALL STATE of path functionals; a pure STEP
`step(s, T_now, dt)` moves it along the temperature path. Everything the world reads (hardness,
the temper ratio the edge law uses, ductility, grain) is a READOUT of that state. Nothing is a
mode: slow cooling through A1 anneals, air cooling normalises, a quench faster than the nose of
the C-curve makes martensite, and tempering is the same step below A1. The rows are data
(heathist_data.py); the equations are the same for any carbon or manganese steel.

State, per sample (float64 arrays in a dict; `new_state`):
  c, mn   composition in weight percent (fixed in stage 1)
  fg      austenite fraction (also the RETAINED austenite once cold)
  fM      martensite fraction
  (the rest, 1 - fg - fM, is soft product: ferrite, pearlite, bainite) with mean hardness HVs
  d       austenite grain diameter, micrometres
  Xd      Scheil additivity sum of the diffusional product, 0..1
  Tm      lowest temperature reached since the sample was last above Ms (Koistinen-Marburger)
  P       Hollomon-Jaffe parameter of the martensite, T (C + log10 t_h) / 1000
  Tp      the temperature of the last step (the path's previous point)

THE LAWS (each a closed form that a step reproduces whatever the step size, which tests check):
  Ac1, Ac3     Andrews 1965; Ms Andrews / Steven-Haynes (474), with a 33 K spread carried as data
  heating      fg relaxes to the lever rule fg_eq(T) between Ac1 and Ac3 at tau_a(T) (Arrhenius)
  grain        d^n grows by K(T) dt while austenite is present: additive in d^n, so the step IS
               the path integral. New austenite is fine (d0): d is the fraction-weighted mean.
  cooling      Scheil: dXd = dt/tau(T), with a C-curve nose; the austenite that converts is
               fg dXd/(1 - Xd), and it forms soft product of hardness set by the temperature
  martensite   Koistinen-Marburger, written as its path-exact increment: below Ms and below any
               colder point already reached, fg falls by exp(-b dT) and fM takes the rest
  tempering    Hollomon-Jaffe, inverted to equivalent time so a stepped path equals the closed
               form; P never falls (max)
  latent heat  enthalpy of the state h = Lg fg + Lm fM (J/g). `step` returns the change in h so
               the caller books it: thermal energy + m h is the conserved sum.

Mass-fraction bookkeeping is exact: fg + fM + fs = 1 after every step (tests).
KNOWN GAPS (ledger rows in switches): no tempering embrittlement band, no bainite below the nose
(the Scheil product cuts off at Ms), no cementite network above 0.77 C, martensite carbon of an
intercritical quench is the bulk carbon, no carbon exchange with the gas, no volume change.
"""
import numpy as np

from . import heathist_data as D

R = D.R_GAS
K0C = 273.15


# --------------------------------------------------------------------------------------------
# composition laws
# --------------------------------------------------------------------------------------------
def ac1(c, mn, ni=0.0, si=0.0, cr=0.0):
    a = D.AC1
    return a["base"] + a["Mn"] * mn + a["Ni"] * ni + a["Si"] * si + a["Cr"] * cr


def ac3(c, mn, ni=0.0, si=0.0, mo=0.0):
    a = D.AC3
    t = a["base"] + a["sqrtC"] * np.sqrt(c) + a["Ni"] * ni + a["Si"] * si + a["Mo"] * mo
    return np.maximum(t, ac1(c, mn) + 1.0)       # the two lines never cross (hypereutectoid: Acm)


def ms(c, mn, cr=0.0, ni=0.0, mo=0.0, variant=None):
    """Martensite start, C. Andrews below 0.6 percent C, Steven-Haynes above (the 474 row)."""
    a, s = D.MS_ANDREWS, D.MS_STEVEN_HAYNES
    ck = s["C"] if variant is None else variant
    lo = a["base"] + a["C"] * c + a["Mn"] * mn + a["Cr"] * cr + a["Ni"] * ni + a["Mo"] * mo
    hi = s["base"] + ck * c + s["Mn"] * mn + s["Cr"] * cr + s["Ni"] * ni + s["Mo"] * mo
    return np.where(np.asarray(c) < D.C_SWITCH, lo, hi)


def hv_base(c):
    return D.HV_BASE[0] + D.HV_BASE[1] * np.asarray(c, float)


def hv_martensite(c):
    return np.interp(c, D.HV_M_C, D.HV_M)


def c_hj(c):
    """Hollomon-Jaffe constant: 19.5 up to 0.4 C, 15 from 0.9 C, linear between."""
    x = np.clip((np.asarray(c, float) - D.CHJ_C_LOW) / (D.CHJ_C_HIGH - D.CHJ_C_LOW), 0.0, 1.0)
    return D.CHJ_LOW + (D.CHJ_HIGH - D.CHJ_LOW) * x


def g_temper(P):
    """Share of the martensite's excess hardness that is left after tempering parameter P."""
    return np.interp(P, D.G_P, D.G_G)


def _L_aust(c):
    x = np.clip(np.asarray(c, float) / 0.77, 0.0, 1.0)
    return D.L_AUST_J_G[0] + (D.L_AUST_J_G[1] - D.L_AUST_J_G[0]) * x


def enthalpy(s):
    """h_state, J per gram: what the state stores above the soft aggregate."""
    L = _L_aust(s["c"])
    return L * s["fg"] + D.L_MART_SHARE * L * s["fM"]


# --------------------------------------------------------------------------------------------
# constants derived once from the rows
# --------------------------------------------------------------------------------------------
def _grain_row():
    (T1, d1), (T2, d2) = D.GRAIN_FIT
    n, d0, t = D.GRAIN_N, D.GRAIN_D0_UM, 3600.0
    k1, k2 = (d1 ** n - d0 ** n) / t, (d2 ** n - d0 ** n) / t       # um^n / s
    T1k, T2k = T1 + K0C, T2 + K0C
    Q = R * np.log(k2 / k1) / (1.0 / T1k - 1.0 / T2k)
    K0 = k1 / np.exp(-Q / (R * T1k))
    return float(Q), float(K0)


GRAIN_Q, GRAIN_K0 = _grain_row()


def _nose_q():
    """Activation energy of the C-curve that puts the nose of the reference steel at NOSE_T_C:
    d/dT of [m ln(A - T) - Q/RT] = 0  gives  Q = m R Tn^2 / (A - Tn)."""
    A = float(ac3(D.NOSE_REF_C, D.NOSE_REF_MN))
    Tn = D.NOSE_T_C + K0C
    return D.NOSE_M * R * Tn * Tn / (A + K0C - Tn)


NOSE_Q = _nose_q()


def tau_nose(c, mn, d):
    """Seconds to transform at the nose of the C-curve. Carbon and manganese raise it
    (hardenability); a coarser austenite grain raises it (fewer nucleation sites)."""
    t = D.NOSE_TAU_S * np.exp(D.TAU_C_SLOPE * (np.asarray(c, float) - D.NOSE_REF_C))
    t = t * (1.0 + D.MN_GROSSMANN * mn) / (1.0 + D.MN_GROSSMANN * D.NOSE_REF_MN)
    return t * (np.maximum(d, 1.0) / D.GRAIN_REF_UM) ** D.HARDENABILITY_P


def _nose_T(A_K):
    m, Q = D.NOSE_M, NOSE_Q
    return (-Q + np.sqrt(Q * Q + 4.0 * m * R * Q * A_K)) / (2.0 * m * R)


def diff_rate(T_C, c, mn, d):
    """1/tau of the diffusional product at T (s^-1): Kirkaldy-type, normalised so that its
    maximum (the nose) is 1/tau_nose. Zero at and above Ae3 and below Ms."""
    A = ac3(c, mn) + K0C
    T = np.asarray(T_C, float) + K0C
    Tn = _nose_T(A)
    drive = np.clip((A - T) / (A - Tn), 0.0, None)
    shape = drive ** D.NOSE_M * np.exp(-NOSE_Q / R * (1.0 / T - 1.0 / Tn))
    return shape / tau_nose(c, mn, d)


def tau_aust(T_C):
    T = np.asarray(T_C, float) + K0C
    return D.TAU_A_REF_S * np.exp(D.QA_J / R * (1.0 / T - 1.0 / (D.TAU_A_REF_C + K0C)))


def hv_diffusional(T_C, c, mn):
    """Hardness of the soft product formed at T: coarse and soft near A1, fine and hard at the
    nose (pearlite spacing falls as 1/(A1 - T); strength with spacing^-1/2: a linear form here)."""
    a1 = ac1(c, mn)
    x = np.clip((a1 - np.asarray(T_C, float)) / (a1 - D.NOSE_T_C), 0.0, 1.0)
    hb = hv_base(c)
    return hb + (D.HV_FINE_RATIO * hb - hb) * x


# --------------------------------------------------------------------------------------------
# the state
# --------------------------------------------------------------------------------------------
def new_state(c, mn=0.5, T0=20.0, n=None):
    """A fresh sample set at rest at T0: annealed, no austenite, no martensite."""
    c = np.atleast_1d(np.asarray(c, float))
    n = len(c) if n is None else n
    c = np.broadcast_to(c, (n,)).copy()
    mn = np.broadcast_to(np.asarray(mn, float), (n,)).copy()
    return dict(c=c, mn=mn, fg=np.zeros(n), fM=np.zeros(n), HVs=hv_base(c),
                d=np.full(n, D.GRAIN_D0_UM), Xd=np.zeros(n), Tm=np.full(n, 1e9),
                P=np.zeros(n), Tp=np.full(n, float(T0)))


def hardness(s):
    """Vickers hardness of the sample as it is now."""
    fs = np.clip(1.0 - s["fg"] - s["fM"], 0.0, 1.0)
    hb = hv_base(s["c"])
    hm = hb + (hv_martensite(s["c"]) - hb) * g_temper(s["P"])
    return s["fM"] * hm + fs * s["HVs"] + s["fg"] * D.HV_RETAINED


def temper_ratio(s):
    """What the edge and strength laws multiply by: HV over the row's own base hardness.
    Exactly 1.0 for a sample that never passed Ac1 or Ms."""
    return hardness(s) / hv_base(s["c"])


def toughness_factor(s):
    """Multiplier on ductility and impact toughness (1 for the soft annealed state at the
    reference grain): (HV_base/HV)^q (hard martensite is brittle, tempering restores it) times
    the Hall-Petch grain term (d_ref/d)^1/2."""
    r = temper_ratio(s)
    g = (D.GRAIN_REF_UM / np.maximum(s["d"], 1.0)) ** D.GRAIN_HP
    out = (1.0 / np.maximum(r, 1e-6)) ** D.DUCT_Q
    return np.where(np.asarray(s["d"]) <= D.GRAIN_REF_UM, out, out * g)


def _piece(s, T_a, T_b, dt, idx):
    """One substep on the samples `idx`, the temperature going T_a -> T_b over dt.
    Mid-point temperature is used for the rates (the 3-point rule). Updates s in place and
    returns the change in the state's enthalpy (J/g) for those samples."""
    c, mn = s["c"][idx], s["mn"][idx]
    T = 0.5 * (T_a + T_b)
    h0 = enthalpy({k: s[k][idx] for k in ("c", "fg", "fM")})
    A1, A3, Ms = ac1(c, mn), ac3(c, mn), ms(c, mn)
    fg, fM, d, Xd, HVs = s["fg"][idx], s["fM"][idx], s["d"][idx], s["Xd"][idx], s["HVs"][idx]
    Tm, P = s["Tm"][idx], s["P"][idx]
    fs = np.clip(1.0 - fg - fM, 0.0, 1.0)

    # --- heating: austenite forms toward the lever rule
    feq = np.clip((T - A1) / (A3 - A1), 0.0, 1.0)
    up = feq > fg
    if up.any():
        tau = tau_aust(T)
        new = np.where(up, feq - (feq - fg) * np.exp(-dt / tau), fg)
        df = new - fg
        take = np.minimum(df, fs + fM)
        share = np.where(fs + fM > 0, take / np.maximum(fs + fM, 1e-300), 0.0)
        fM = fM * (1.0 - share)                       # austenite eats the martensite and soft product alike
        fs = fs * (1.0 - share)
        d = np.where(take > 0, (fg * d + take * D.GRAIN_D0_UM) / np.maximum(fg + take, 1e-300), d)
        fg = fg + take
        Xd = np.where(T > A1, 0.0, Xd)
        Tm = np.where(T > A1, 1e9, Tm)

    # --- grain growth in the austenite (d^n additive: the step is the path integral)
    hot = fg > 0.5
    if hot.any():
        K = GRAIN_K0 * np.exp(-GRAIN_Q / (R * (T + K0C)))
        dn = d ** D.GRAIN_N + K * dt
        d = np.where(hot, dn ** (1.0 / D.GRAIN_N), d)

    # --- cooling: diffusional product by Scheil additivity (T above Ms, below Ae3)
    cool = ((T_b < T_a) | (T < A1)) & (fg > 0) & (T < A3) & (T > Ms)
    if cool.any():
        rate = np.where(cool, diff_rate(T, c, mn, d), 0.0)
        Xn = np.minimum(Xd + rate * dt, 1.0)
        rem = np.maximum(1.0 - Xd, 1e-300)
        conv = np.where(Xd < 1.0, fg * (Xn - Xd) / rem, fg * 0.0)
        conv = np.where(Xd >= 1.0, np.where(cool, fg, 0.0), conv)   # the C-curve is finished
        hv_new = hv_diffusional(T, c, mn)
        HVs = np.where(conv > 0, (fs * HVs + conv * hv_new) / np.maximum(fs + conv, 1e-300), HVs)
        fs = fs + conv
        fg = fg - conv
        Xd = np.where(cool, Xn, Xd)

    # --- martensite: Koistinen-Marburger, path-exact increment below Ms and below the lowest point
    Tm = np.where(T_b > Ms, 1e9, Tm)                   # above Ms nothing is remembered
    ref = np.minimum(Tm, Ms)
    low = (T_b < ref) & (fg > 0)
    if low.any():
        frac = 1.0 - np.exp(-D.KM_B * np.where(low, ref - T_b, 0.0))
        dM = fg * frac
        P = np.where(dM > 0, fM * P / np.maximum(fM + dM, 1e-300), P)
        fM = fM + dM
        fg = fg - dM
    Tm = np.where((T_b < Ms) & (T_b < Tm), T_b, Tm)

    # --- tempering (below Ac1, martensite present, warm enough to matter)
    tp = (T_b < A1) & (fM > 0) & (T > 100.0)
    if tp.any():
        Tk = T + K0C
        C = c_hj(c)
        t_eq = 10.0 ** np.minimum(P * 1000.0 / Tk - C, 300.0)          # hours
        Pn = Tk * (C + np.log10(t_eq + dt / 3600.0)) / 1000.0
        P = np.where(tp, np.maximum(P, Pn), P)

    for k, v in (("fg", fg), ("fM", fM), ("d", d), ("Xd", Xd), ("HVs", HVs), ("Tm", Tm), ("P", P)):
        s[k][idx] = v
    h1 = enthalpy({"c": c, "fg": fg, "fM": fM})
    return h1 - h0


def step(s, T_now, dt, dT_max=20.0, max_sub=64):
    """Advance every sample along its temperature path from `s["Tp"]` to `T_now` over dt seconds
    (linear in time). Returns the change in enthalpy of each sample's state, J/g, for the
    caller to book (thermal energy goes down by m dh). A sample that is cold and has no
    austenite is skipped. The substeps are shared (the worst sample sets them)."""
    with np.errstate(all="ignore"):
        return _step(s, T_now, dt, dT_max, max_sub)


def _step(s, T_now, dt, dT_max, max_sub):
    T_now = np.broadcast_to(np.asarray(T_now, float), s["Tp"].shape)
    Tp = s["Tp"]
    dh = np.zeros_like(Tp)
    active = (np.maximum(Tp, T_now) > 100.0) | (s["fg"] > 0)
    idx = np.flatnonzero(active)
    if len(idx):
        span = float(np.abs(T_now[idx] - Tp[idx]).max())
        n = int(min(max_sub, max(1, np.ceil(span / dT_max))))
        edges = Tp[idx][None, :] + (T_now[idx] - Tp[idx])[None, :] * (np.arange(n + 1) / n)[:, None]
        for j in range(n):
            dh[idx] += _piece(s, edges[j], edges[j + 1], dt / n, idx)
    s["Tp"] = np.array(T_now, float)
    return dh


def oxide_step_v(x, T_C, dt, p=None):
    """layer_laws.oxide_step on arrays (the same two-regime law, the same arithmetic; a test
    compares them value by value). The film thickness x grows by dt at temperature T_C."""
    from . import layer_laws as LL
    p = LL.OXIDE if p is None else p
    x, T = np.asarray(x, float), np.asarray(T_C, float)
    xl = p["A"] * np.exp(-p["B"] / (T + 273.15))
    xj = xl * np.log1p(p["t_join"] / p["tau"])
    kp = p["kp_x"] * np.exp(-p["Q"] / R * (1.0 / (T + 273.15) - 1.0 / (p["T_x"] + 273.15)))
    hot = T >= p["T_x"]
    logreg = ~hot | (x < xj)
    u = x / xl
    t_eq = np.where(u < 700.0, p["tau"] * np.expm1(np.minimum(u, 700.0)), 1e300)
    xn = xl * np.log1p((t_eq + dt) / p["tau"])
    cross = hot & (xn > xj)
    t_left = dt - np.maximum(0.0, p["t_join"] - t_eq)
    xc = np.sqrt(xj * xj + kp * np.maximum(t_left, 0.0))
    par = np.sqrt(x * x + kp * dt)
    return np.where(logreg, np.where(cross, xc, xn), par)
