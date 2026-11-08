"""SKIN BURN: a thin 1-D skin profile under every hot 5 cm face (wave 13, T10 3.1).

A 5 cm flesh cell with a 3.7 mm skin node cannot see a burn that happens in the
top 2 mm in under a second. So each HOT (or damaged) face gets its own small
profile: 8 nodes, 10 mm deep, backward Euler, a tridiagonal (Thomas) solve done
for all faces at once. The host voxel keeps the heat totals; the face record is
a READOUT of the flux the heat law applied to that face. It writes nothing
back. It is energy-open by the little heat its profile holds above the host's
mean (`excess_enthalpy_J`), which is a diagnostic, not a conserved ledger.

Per node it integrates
  - the Moritz-Henriques damage  d(omega)/dt = P exp(-dE / R T)  for T >= 44 C,
  - the mirror cold-injury integral  d(omega_c)/dt = 1 / t_freeze(T)  below
    -0.5 C (a log-interpolated data row table).
The degree and the burned depth are READ from those profiles by comparing with
a table (data, not code). Heat pain is read at the basal node.

THE FOUR BOUNDARY KINDS (one exposure per face; the unit is a face, so a face
touching hot iron is never averaged with its cool neighbours):
  "gas" (alias "flame")  absorbed radiation a*q_rad + h_c (T_gas - T_s), minus
                         the skin's own radiation eps sigma (T_s^4 - T_env^4);
  "contact"              the interface temperature of two semi-infinite
                         bodies, T_i = (e_s T_s + e_o T_o) / (e_s + e_o), held
                         as a fixed surface temperature; a thin hot object
                         that passes `heat_cap_area` (J/m2K) cools as it gives
                         heat, otherwise the semi-infinite rule is used and
                         the face is marked `semi_infinite`;
  "liquid"               h (T_liq - T_s), h from a small table (water 1500,
                         oil 400 W/m2K), the scald and cold-water first aid;
  "steam"                condensing steam, h = 5000 W/m2K at 100 C.
An exposure may carry a `clothing` layer {thickness, k, rho_c, wet} as an extra
top node.

Plain numpy is the definition. Every batched operation is element-wise across
faces and the sums run along depth only, so the answer for a face does not
depend on which other faces are in the batch (tested bit for bit).

Numbers marked (u) are unverified: Torvi and Dale 1994 skin properties, the
steam h, the cold table, the region dermis depths. Moritz and Henriques 1947
for the damage constants. Du Bois for the body surface area.
"""
import math

import numpy as np

try:
    from .materials import BLEED as _BLEED
except ImportError:  # pragma: no cover - a stand-in with the same two numbers
    _BLEED = {"f0": 0.45, "k": 0.56}

SIGMA = 5.670374419e-8
R_GAS = 8.314
# Henriques (Moritz and Henriques 1947; realism map constants).
HEN_P = 3.1e98                 # 1/s
HEN_DE = 6.28e5                # J/mol
HEN_DE_R = HEN_DE / R_GAS      # 75,540 K
HEN_LNP = math.log(HEN_P)
T_DAMAGE = 44.0                # C: below this no omega is gained

# The thin skin: depths (mm) to layer bottoms. Node 1 is the basal layer
# (0.05-0.10 mm, centre 0.075 mm), where the pain sensor sits; node 5 ends the
# dermis at 2.0 mm.
BOUNDS_MM = (0.05, 0.10, 0.40, 1.00, 1.50, 2.00, 4.0, 10.0)
NODES = 8
BASAL = 1
DERMIS_BASE = 5
FACE_AREA_M2 = 0.0025          # one 5 cm face

# Properties by layer (Torvi and Dale 1994, (u) to re-check): k W/mK,
# rho*c J/m3K, perfusion w_b rho_b c_b W/m3K (toward the arterial T).
EPIDERMIS = {"k": 0.255, "rc": 3.96e6, "wb": 0.0}
DERMIS = {"k": 0.523, "rc": 3.96e6, "wb": 4750.0}
SUBCUTIS = {"k": 0.167, "rc": 2.5e6, "wb": 1500.0}
EPIDERMIS_MM = 0.10
T_ARTERIAL = 37.0
DEEP_K, DEEP_L = 0.5, 0.020    # the reservoir sits 20 mm of subcutis below

# Dermis end depth (mm) by region (u). Back is thick, palm and sole thicker,
# the face thin. Unknown names get the default.
DERMIS_MM_DEFAULT = 2.0
DERMIS_MM = {"back": 3.0, "palm": 3.0, "sole": 3.0, "face": 1.2, "head": 1.5,
             "hand": 1.2, "forearm": 1.5, "upper arm": 1.8, "thigh": 2.0,
             "shin": 1.8, "foot": 2.0, "torso": 2.0, "chest": 2.0,
             "abdomen": 2.0}

# Boundary data.
ABSORB = 0.94                  # skin absorptivity of flame radiation
EPS_SKIN = 0.95                # emissivity of skin (long wave)
H_STILL = 10.0                 # W/m2K, still air (the heat law's number)
SKIN_E = 1100.0                # skin effusivity sqrt(k rho c), (u)
LIQUID_H = {"water": 1500.0, "oil": 400.0}     # W/m2K, (u)
STEAM_H = 5000.0               # W/m2K at 100 C, condensing, (u)
STEAM_T = 100.0

# Degrees: (name, damage node, omega threshold) from the deepest down. Data.
DEGREES = ((3, DERMIS_BASE, 1.0), (2, BASAL, 1.0), (1, BASAL, 0.53))
# Cold injury degrees on omega_c: 2 deep frostbite, 1 frostnip.
COLD_DEGREES = ((2, DERMIS_BASE, 1.0), (1, BASAL, 1.0))
# Freezing: t_freeze(T) rows (T C, seconds), log-interpolated (u: shaped on
# wind-chill frostbite charts). Below the last row it stays at the last.
T_FREEZE = ((-30.0, 60.0), (-15.0, 300.0), (-5.0, 1800.0), (-0.5, 7200.0))
T_FREEZE_ONSET = -0.5
_FZ_T = np.array([r[0] for r in T_FREEZE])
_FZ_LN = np.log([r[1] for r in T_FREEZE])

# Area weights of the burn that LEAKS (shock): first degree does not.
TBSA_WEIGHTS = {1: 0.0, 2: 0.6, 3: 1.0}
# Shock: edema / V0 = 0.012 * sum_i dTBSA_w,i * (1 - exp(-(t - t_i) / 6 h)).
SHOCK_PER_PCT = 0.012
SHOCK_TAU_S = 6.0 * 3600.0
SHOCK_FOLD_S = 24.0 * 3600.0
SHOCK_CONST_T = -1e18          # an event at this "time" is a settled constant

PAIN_T0, PAIN_SPAN = 44.0, 11.0


# ----------------------------------------------------------------- the mesh --
def bounds_for(dermis_mm=DERMIS_MM_DEFAULT):
    """Depths (m) to layer bottoms with the dermis ending at `dermis_mm`."""
    s = (dermis_mm - EPIDERMIS_MM) / (DERMIS_MM_DEFAULT - EPIDERMIS_MM)
    e = EPIDERMIS_MM
    mm = (0.05, e, e + 0.30 * s, e + 0.90 * s, e + 1.40 * s, dermis_mm,
          dermis_mm + 2.0, dermis_mm + 8.0)
    return np.array(mm) * 1e-3


def props_at(depth_m, dermis_mm=DERMIS_MM_DEFAULT):
    """(k, rho*c, w_b rho_b c_b) at depths (m): epidermis, dermis, subcutis."""
    d = np.asarray(depth_m, dtype=float)
    epi = d < EPIDERMIS_MM * 1e-3
    der = (~epi) & (d < dermis_mm * 1e-3)
    out = []
    for key in ("k", "rc", "wb"):
        out.append(np.where(epi, EPIDERMIS[key],
                            np.where(der, DERMIS[key], SUBCUTIS[key])))
    return tuple(out)


def make_mesh(th, k, rc, wb, k_deep=DEEP_K, l_deep=DEEP_L):
    """A mesh of nodes from top down: thickness, conductivity, rho*c and
    perfusion per node. G joins neighbours through their halves, Gb joins the
    last node to the arterial reservoir, Gs the top node to its own surface."""
    th = np.asarray(th, dtype=float)
    k = np.asarray(k, dtype=float)
    G = 1.0 / (th[:-1] / (2.0 * k[:-1]) + th[1:] / (2.0 * k[1:]))
    Gb = 1.0 / (th[-1] / (2.0 * k[-1]) + l_deep / k_deep)
    return {"n": len(th), "th": th, "k": k, "C": np.asarray(rc, dtype=float) * th,
            "G": G, "Gb": Gb, "Gs": 2.0 * k[0] / th[0],
            "wbth": np.asarray(wb, dtype=float) * th}


def mesh_from_bounds(bounds_m, dermis_mm=DERMIS_MM_DEFAULT):
    b = np.asarray(bounds_m, dtype=float)
    th = np.diff(np.concatenate(([0.0], b)))
    k, rc, wb = props_at(b - th / 2.0, dermis_mm)
    return make_mesh(th, k, rc, wb)


_MESHES = {}


def mesh_for(dermis_mm=DERMIS_MM_DEFAULT):
    """The 8-node skin mesh for a dermis depth (cached; read only)."""
    key = float(dermis_mm)
    m = _MESHES.get(key)
    if m is None:
        m = mesh_from_bounds(bounds_for(key), key)
        m["bounds_m"] = bounds_for(key)
        _MESHES[key] = m
    return m


def graded_bounds(n, th0_m=1.25e-5, total_m=0.010):
    """n node bottoms with geometric growth, summing to `total_m`: the fine
    reference mesh the 8-node solution is checked against."""
    lo, hi = 1.0, 2.0
    for _ in range(200):
        r = 0.5 * (lo + hi)
        s = th0_m * (r ** n - 1.0) / (r - 1.0)
        lo, hi = (r, hi) if s < total_m else (lo, r)
    r = 0.5 * (lo + hi)
    return np.cumsum(th0_m * r ** np.arange(n))


def cloth_mesh(base, thickness, k, rho_c, wet=0.0):
    """The base mesh with one clothing node on top. `wet` is the water volume
    fraction: it adds water's heat capacity and a little conductivity (u)."""
    rc = rho_c + wet * 4.18e6
    kk = k * (1.0 + 2.0 * wet)
    th = np.concatenate(([thickness], base["th"]))
    ks = np.concatenate(([kk], base["k"]))
    rcs = np.concatenate(([rc], base["C"] / base["th"]))
    wbs = np.concatenate(([0.0], base["wbth"] / base["th"]))
    return make_mesh(th, ks, rcs, wbs)


# ---------------------------------------------------------------- the solver --
def _thomas(a, b, c, d):
    """Tridiagonal solve for all columns at once. Arrays are (n, N), a[0] and
    c[-1] unused. Rows are contiguous, sums run along depth only."""
    n = b.shape[0]
    cp = np.empty_like(b)
    dp = np.empty_like(b)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / m
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m
    x = np.empty_like(b)
    x[n - 1] = dp[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def make_bc(count):
    """A boundary for `count` columns: heat transfer hh toward Tref, a fixed
    flux q0, radiation eps toward Tenv, and where dm is 1 the top is held at
    Tdir through the surface conductance. All zero is an insulated top."""
    z = np.zeros(count)
    return {"hh": z.copy(), "Tref": z.copy(), "q0": z.copy(), "eps": z.copy(),
            "Tenv": z.copy() + 20.0, "dm": z.copy(), "Tdir": z.copy()}


def flux_bc(q_abs=0.0, h=H_STILL, T_ref=20.0, eps=0.0, T_env=20.0):
    """A one-column boundary from plain numbers (for tests and tools)."""
    bc = make_bc(1)
    bc["q0"][:] = q_abs
    bc["hh"][:] = h
    bc["Tref"][:] = T_ref
    bc["eps"][:] = eps
    bc["Tenv"][:] = T_env
    return bc


def advance(mesh, T, dt, bc, perf=None, T_core=T_ARTERIAL, on=None):
    """One backward-Euler step of the Pennes profile for N columns.
    T is (n, N) in C. perf (N,) scales perfusion by the circulating-blood
    fraction f; `on` (n, N) is 0 where a node has lost its perfusion
    (stasis). Returns (T_new (n, N), q_surface (N,) the net flux, W/m2, the
    boundary delivered into the top node during this step)."""
    n, N = T.shape
    G = mesh["G"]
    Cdt = mesh["C"][:, None] / dt
    wbt = mesh["wbth"][:, None] * np.ones((1, N))
    if perf is not None:
        wbt = wbt * np.asarray(perf, dtype=float)[None, :]
    if on is not None:
        wbt = wbt * on
    core = np.broadcast_to(np.asarray(T_core, dtype=float), (N,))
    b = Cdt + wbt
    if n > 1:
        b[:-1] += G[:, None]
        b[1:] += G[:, None]
    b[-1] += mesh["Gb"]
    d = Cdt * T + wbt * core[None, :]
    d[-1] += mesh["Gb"] * core
    a = np.zeros((n, N))
    c = np.zeros((n, N))
    if n > 1:
        a[1:] = -G[:, None]
        c[:-1] = -G[:, None]
    T0k = T[0] + 273.15
    Tek = bc["Tenv"] + 273.15
    hr = bc["eps"] * SIGMA * (T0k * T0k + Tek * Tek) * (T0k + Tek)
    dm = bc["dm"]
    free = 1.0 - dm
    h_all = (bc["hh"] + hr) * free
    b[0] += h_all + mesh["Gs"] * dm
    d[0] += (bc["hh"] * bc["Tref"] + hr * bc["Tenv"] + bc["q0"]) * free \
        + mesh["Gs"] * bc["Tdir"] * dm
    Tn = _thomas(a, b, c, d)
    t0 = Tn[0]
    q = ((bc["q0"] + bc["hh"] * (bc["Tref"] - t0) + hr * (bc["Tenv"] - t0))
         * free + mesh["Gs"] * (bc["Tdir"] - t0) * dm)
    return Tn, q


def steady_profile(mesh, T_surf, T_core=T_ARTERIAL, f=1.0):
    """The steady Pennes profile for a surface held at T_surf (8 floats)."""
    bc = make_bc(1)
    bc["dm"][:] = 1.0
    bc["Tdir"][:] = T_surf
    T = np.full((mesh["n"], 1), float(T_surf))
    Tn, _ = advance(mesh, T, 1e12, bc, perf=np.array([f]), T_core=T_core)
    return Tn[:, 0].copy()


# ----------------------------------------------------------- damage integrals --
_HEN_R44 = math.exp(HEN_LNP - HEN_DE_R / (T_DAMAGE + 273.15))


def henriques_rate(T_c):
    """d(omega)/dt (1/s) at temperature T_c (array or float); 0 below 44 C."""
    T = np.asarray(T_c, dtype=float)
    hot = T >= T_DAMAGE
    TK = np.where(hot, T, T_DAMAGE) + 273.15
    r = np.exp(HEN_LNP - HEN_DE_R / TK)
    # The Arrhenius value at 44 C (1.1e-5 /s) is taken off so the rate starts
    # at zero and has no step; at 60 C it changes the rate by 1e-5 only.
    return np.where(hot, np.maximum(r - _HEN_R44, 0.0), 0.0)


def freeze_rate(T_c):
    """d(omega_c)/dt (1/s): 1 / t_freeze(T) below -0.5 C, else 0."""
    T = np.asarray(T_c, dtype=float)
    cold = T < T_FREEZE_ONSET
    ln = np.interp(np.where(cold, T, T_FREEZE_ONSET), _FZ_T, _FZ_LN)
    return np.where(cold, np.exp(-ln), 0.0)


def contact_T(e_a, T_a, e_b, T_b):
    """Interface temperature of two semi-infinite bodies put together."""
    return (e_a * T_a + e_b * T_b) / (e_a + e_b)


def pain_heat(T_basal):
    """Heat pain 0..1 from the basal node: starts at 44 C, full at 55 C (u)."""
    return float(np.clip((T_basal - PAIN_T0) / PAIN_SPAN, 0.0, 1.0))


# ------------------------------------------------------------------- faces --
def face_key(bone, cell_idx, direction):
    """The key of a face in p["body"]["skin"]: bone name, index in the bone's
    own cell list, face direction in the body frame."""
    return "%s|%d|%s" % (bone, int(cell_idx), direction)


def dermis_mm_of(region, sheet=None):
    """Dermis depth for a region name (u), times sheet["skin_scale"]."""
    name = str(region).replace("_", " ").strip().lower()
    if name.startswith(("left ", "right ")):
        name = name.split(" ", 1)[1]
    d = DERMIS_MM.get(name)
    if d is None:
        d = DERMIS_MM_DEFAULT
        for key in sorted(DERMIS_MM):
            if key in name:
                d = DERMIS_MM[key]
                break
    scale = float(sheet.get("skin_scale", 1.0)) if sheet else 1.0
    return d * scale


def new_face(sheet=None, region="torso", T_start=33.0, T_core=T_ARTERIAL):
    """A face resting at the skin set point of the air it stands in: the steady
    profile for a surface at `T_start`. Arrays are float64 (8)."""
    dm = dermis_mm_of(region, sheet)
    T = steady_profile(mesh_for(dm), T_start, T_core)
    return {"v": 1, "region": str(region), "dermis_mm": dm, "T": T,
            "omega": np.zeros(NODES), "omega_c": np.zeros(NODES),
            "q_in": 0.0}


def prunable(face, T_cool=34.0):
    """A cool undamaged face carries no information and can be dropped."""
    return (face["T"][:DERMIS_BASE + 1].max() < T_cool
            and not face["omega"].any()
            and not face["omega_c"].any())


DEFAULT_EXPOSURE = {"kind": "gas", "T_gas": 20.0, "h_c": H_STILL,
                    "q_rad": 0.0, "T_env": 20.0}
T_SET = 33.0                   # the skin set point a new face rests at

_REST = {}


def rest_flux(dermis_mm=DERMIS_MM_DEFAULT, exposure=None):
    """The make-up flux (W/m2, into the surface) that keeps a face at rest in
    an exposure. A resting person holds the skin near its set point (blood,
    sweat and shivering together); this one number stands for that. It is the
    flux the steady profile for a 33 C surface needs, so that profile does not
    drift. `exposure` is a gas exposure (default: still air at 20 C). Without
    it, a face at rest would cool as if it were a dead body."""
    e = exposure or DEFAULT_EXPOSURE
    key = (float(dermis_mm), float(e.get("h_c", H_STILL)),
           float(e.get("T_gas", 20.0)), float(e.get("eps", EPS_SKIN)),
           float(e.get("T_env", 20.0)))
    q = _REST.get(key)
    if q is None:
        mesh = mesh_for(dermis_mm)
        t0 = float(steady_profile(mesh, T_SET)[0])
        tk, tek = t0 + 273.15, key[4] + 273.15
        hr = key[3] * SIGMA * (tk * tk + tek * tek) * (tk + tek)
        q = float(mesh["Gs"] * (T_SET - t0)
                  - key[1] * (key[2] - t0) - hr * (key[4] - t0))
        _REST[key] = q
    return q


def _row(e, Ts0, face):
    """One exposure -> (hh, Tref, q0, eps, Tenv, dm, Tdir, cap) as floats.
    Also keeps the contact bookkeeping on the face."""
    kind = e.get("kind", "gas")
    if kind in ("gas", "flame"):
        q0 = float(e.get("a", ABSORB)) * float(e.get("q_rad", 0.0)) \
            - float(e.get("evap_W_m2", 0.0))
        if e is DEFAULT_EXPOSURE:       # nothing given: the face is at rest
            q0 += rest_flux(face.get("dermis_mm", DERMIS_MM_DEFAULT))
        return (float(e.get("h_c", H_STILL)), float(e.get("T_gas", 20.0)), q0,
                float(e.get("eps", EPS_SKIN)), float(e.get("T_env", 20.0)),
                0.0, 0.0)
    if kind == "liquid":
        h = e.get("h")
        if h is None:
            h = LIQUID_H[e.get("liquid", "water")]
        return (float(h), float(e["T_liq"]), 0.0, 0.0, 20.0, 0.0, 0.0)
    if kind == "steam":
        return (float(e.get("h_c", STEAM_H)), float(e.get("T_gas", STEAM_T)),
                0.0, 0.0, 20.0, 0.0, 0.0)
    if kind == "contact":
        if "contact_T0" not in face:
            face["contact_T0"] = float(Ts0)
            face["contact_Q"] = 0.0
        T_o = float(e["T_o"])
        cap = e.get("heat_cap_area")
        if cap:
            # The object gives (or takes) heat until it meets the skin's
            # starting temperature: a hot one relaxes down to it, a cold one
            # relaxes up to it, neither passes it.
            T_s0 = face["contact_T0"]
            T_rel = T_o - face["contact_Q"] / float(cap)
            T_o = max(T_rel, T_s0) if T_o >= T_s0 else min(T_rel, T_s0)
            face["semi_infinite"] = False
        else:
            face["semi_infinite"] = True
        Ti = contact_T(float(e.get("e_s", SKIN_E)), face["contact_T0"],
                       float(e["e_o"]), T_o)
        return (0.0, 0.0, 0.0, 0.0, 20.0, 1.0, Ti)
    raise ValueError("unknown exposure kind: %r" % (kind,))


def _cloth_key(e):
    cl = e.get("clothing")
    if not cl:
        return None
    return (float(cl["thickness"]), float(cl["k"]), float(cl["rho_c"]),
            float(cl.get("wet", 0.0)))


def step(faces, dt, exposures=None, f=1.0, T_core=T_ARTERIAL):
    """Advance every face by dt seconds, in place. `exposures` is a list of
    one dict (or None for still air at 20 C) per face. f is the circulating
    blood fraction (scalar or per face); a node whose omega reached 1 has no
    perfusion any more. Faces are grouped by (dermis depth, clothing) and each
    group is solved in one batch. Exposures are not changed."""
    N = len(faces)
    if N == 0:
        return
    if exposures is None:
        exposures = [None] * N
    fv = np.broadcast_to(np.asarray(f, dtype=float), (N,))
    cv = np.broadcast_to(np.asarray(T_core, dtype=float), (N,))
    groups = {}
    for i in range(N):
        e = exposures[i] or DEFAULT_EXPOSURE
        groups.setdefault((faces[i]["dermis_mm"], _cloth_key(e)), []).append(i)
    for (dm_mm, ck), idx in groups.items():
        _step_group(faces, exposures, idx, dm_mm, ck, dt, fv, cv)


def _step_group(faces, exposures, idx, dm_mm, ck, dt, fv, cv):
    M = len(idx)
    base = mesh_for(dm_mm)
    mesh = base if ck is None else cloth_mesh(base, *ck)
    rows = []
    for i in idx:
        fc = faces[i]
        e = exposures[i] or DEFAULT_EXPOSURE
        kind = e.get("kind", "gas")
        if ck is None:
            if "T_cloth" in fc:
                del fc["T_cloth"]
            Ts0 = fc["T"][0]
        else:
            if "T_cloth" not in fc:
                fc["T_cloth"] = float(e["clothing"].get(
                    "T0", e.get("T_env", 20.0)))
            Ts0 = fc["T_cloth"]
        if kind != "contact" and "contact_T0" in fc:
            for key in ("contact_T0", "contact_Q", "semi_infinite"):
                fc.pop(key, None)
        rows.append(_row(e, Ts0, fc))
    arr = np.array(rows).reshape(M, 7)
    bc = {key: np.ascontiguousarray(arr[:, c]) for c, key in enumerate(
        ("hh", "Tref", "q0", "eps", "Tenv", "dm", "Tdir"))}
    sel = [faces[i] for i in idx]
    skin = np.ascontiguousarray(np.concatenate([fc["T"] for fc in sel])
                                .reshape(M, NODES).T)
    om = np.concatenate([fc["omega"] for fc in sel]).reshape(M, NODES).T
    oc = np.concatenate([fc["omega_c"] for fc in sel]).reshape(M, NODES).T
    T = skin if ck is None else np.vstack(
        [np.array([fc["T_cloth"] for fc in sel])[None, :], skin])
    on = (om < 1.0).astype(float)
    if ck is not None:
        on = np.vstack([np.ones((1, M)), on])
    Tn, q = advance(mesh, T, dt, bc, perf=fv[idx], T_core=cv[idx], on=on)
    sk = Tn if ck is None else Tn[1:]
    # (M, 8) row-major, so each face gets a view of its own row.
    Tr = np.ascontiguousarray(sk.T)
    # Adding zero changes no bit, so a batch with nothing hot (or cold)
    # skips the exponentials without changing any face's answer.
    if sk.max() >= T_DAMAGE:
        om = om + henriques_rate(sk) * dt
    if sk.min() < T_FREEZE_ONSET:
        oc = oc + freeze_rate(sk) * dt
    Or = np.ascontiguousarray(om.T)
    Cr = np.ascontiguousarray(oc.T)
    qd = (q * dt).tolist()
    for j, fc in enumerate(sel):
        fc["T"] = Tr[j]
        fc["omega"] = Or[j]
        fc["omega_c"] = Cr[j]
        fc["q_in"] += qd[j]
        if ck is not None:
            fc["T_cloth"] = float(Tn[0, j])
        if "contact_Q" in fc:
            fc["contact_Q"] += qd[j]


def excess_enthalpy_J(face, T_host, area_m2=FACE_AREA_M2):
    """Heat (J) the face profile holds above the host's mean temperature:
    the amount by which the readout is energy-open. A diagnostic."""
    C = mesh_for(face["dermis_mm"])["C"]
    return float(area_m2 * np.sum(C * (face["T"] - T_host)))


# ---------------------------------------------------------------- readings --
def degree(face, table=DEGREES):
    """Burn degree 0..3 by comparing omega with a table of (degree, node,
    threshold). Third degree is necrosis through the whole dermis."""
    om = face["omega"]
    for deg, node, thr in table:
        if om[node] >= thr:
            return deg
    return 0


def cold_degree(face, table=COLD_DEGREES):
    """Cold injury 0..2: 1 frostnip (basal), 2 deep frostbite (dermis base)."""
    oc = face["omega_c"]
    for deg, node, thr in table:
        if oc[node] >= thr:
            return deg
    return 0


def burn_depth_m(face):
    """Depth (m) to the bottom of the deepest node with omega >= 1; 0 if none."""
    bounds = mesh_for(face["dermis_mm"])["bounds_m"]
    hit = np.nonzero(face["omega"] >= 1.0)[0]
    return float(bounds[hit[-1]]) if hit.size else 0.0


def bsa_dubois(mass_kg, height_cm):
    """Du Bois body surface area, m2: 70 kg and 175 cm give 1.85."""
    return 0.007184 * mass_kg ** 0.425 * height_cm ** 0.725


def tbsa(faces, bsa_m2, weights=None):
    """(partial %, full %, weighted %) of the body surface burned. Partial is
    second degree, full is third; weighted counts what leaks (first degree
    does not). Each face is one 5 cm face of 25 cm2."""
    w = TBSA_WEIGHTS if weights is None else weights
    n2 = n3 = 0
    for fc in faces:
        d = degree(fc)
        n2 += d == 2
        n3 += d == 3
    a = FACE_AREA_M2 / bsa_m2 * 100.0
    return (n2 * a, n3 * a, (n2 * w[2] + n3 * w[3]) * a)


def shock_events(events, t_s):
    """Fraction of circulating blood lost to burn edema by time t_s: the sum
    over burn events [t_i, dTBSA_w,i] (percent) of
    0.012 * dTBSA_w,i * (1 - exp(-(t - t_i) / 6 h)). Burns accrue, so each
    event is its own term. A closed form: a time skip needs no ticks."""
    tot = 0.0
    for t_i, dpct in events:
        dt = t_s - t_i
        if dt > 0.0:
            tot += SHOCK_PER_PCT * dpct * (1.0 - math.exp(-dt / SHOCK_TAU_S))
    return tot


def fold_events(events, t_s, horizon_s=SHOCK_FOLD_S):
    """Events older than the horizon become ONE settled constant (an event at
    SHOCK_CONST_T, whose factor is exactly 1). The folded value is each term
    at the fold time, so the list stays short; what it forgets is the last
    growth, 2 percent of a term at 24 h (u). Returns a new list."""
    keep, const = [], 0.0
    for t_i, dpct in events:
        if t_s - t_i >= horizon_s and t_i > SHOCK_CONST_T / 2:
            const += dpct * (1.0 - math.exp(-(t_s - t_i) / SHOCK_TAU_S))
        elif t_i <= SHOCK_CONST_T / 2:
            const += dpct
        else:
            keep.append([t_i, dpct])
    return ([[SHOCK_CONST_T, const]] if const else []) + keep


def perfusion_after_loss(loss_frac):
    """The brain's oxygen supply left when `loss_frac` of the blood is gone,
    by the same law as physiology._perfusion (BLEED f0 and k)."""
    f = 1.0 - loss_frac
    x = (f - _BLEED["f0"]) / (1.0 - _BLEED["f0"])
    return float(np.clip(x, 0.0, 1.0) ** _BLEED["k"])
