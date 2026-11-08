"""VESSELS and CONDUITS: a cup of tea, a pot, a pipe, a siphon.

Stand-alone (wave 13, topic T2, spec sections 3.6, 3.8, 6.4 spill, 7.4). Nothing
here imports World. Plain numpy and the standard library are the definition.

Two scales (spec 3.8):
  CUP SCALE: contents under about one cell. The record OWNS the matter (class
  b): layers from the bottom, each with its species vector in ml, T, density.
  `owned()` is what the ledger term records_owned adds up.
  VOXEL SCALE: a hollow of 3x3x3 cells or more lives in the lattice; the record
  is then only the stratification split (not built here).

What it keeps:
  * Pour, spill and leak take the TOP layer first. Volumes are exact: every
    species of a layer is split by an exact subtraction, so ml per species
    adds back to the whole. Spills leave in whole PARCELS per tick; the
    remainder stays.
  * Spill over a lip is the weir law Q = (2/3) C_d b sqrt(2 g) H^1.5, solved in
    closed form for a vertical-walled cup (A dH/dt = -k H^1.5).
  * A hole in the wall leaks at min(orifice, slit) (breaking.leak_rate).
  * CONDUIT flow: v = sqrt(2 g dh / (1 + K + f L/D)) with f = 64/Re laminar,
    Blasius 0.316 Re^-0.25 turbulent, and max(64/Re, Blasius) over
    Re 2300-4000 so flow is monotone in pressure. Entrance and exit losses
    are K.
  * A SIPHON runs only while primed; air entering the line clears the prime;
    and it stops when the pressure at the summit reaches the vapour pressure
    of the liquid (about 10 m of water at sea level, less in warm water).

Numbers marked (u) are unverified. Numbers marked (c) were computed from the
formula shown.
"""
import math

from .breaking import exact_split, leak_rate

G = 9.80665
T_REF = 293.15
P_ATM = 101325.0
CD_WEIR = 0.6
PARCEL_ML = 5.0             # spill leaves in whole parcels of this size (u)
RE_LAM, RE_TURB = 2300.0, 4000.0
K_ENTRANCE, K_EXIT = 0.5, 1.0


# ── layers ────────────────────────────────────────────────────────────────────
def make_layer(species_ml, rho=1.0, T=T_REF, cp=4.18):
    """A layer: species -> ml, density g/ml, T in K, cp J/(g K)."""
    return {"sp": dict(species_ml), "rho": float(rho), "T": float(T), "cp": float(cp)}


def layer_ml(layer):
    return math.fsum(layer["sp"].values())


def layer_grams(layer):
    return layer_ml(layer) * layer["rho"]


def layer_J(layer):
    """Thermal energy above the 20 C reference, J."""
    return layer_grams(layer) * layer["cp"] * (layer["T"] - T_REF)


def new_vessel(area, lip_h, lip_w=0.05, parcel_ml=PARCEL_ML, layers=None):
    """A cup-scale vessel: inside cross-section `area` m2, lip at height lip_h m
    above the inside bottom, spout width lip_w m. `acc` is the spill (and leak)
    volume owed but not yet released as parcels."""
    return {"area": float(area), "lip_h": float(lip_h), "lip_w": float(lip_w),
            "parcel": float(parcel_ml), "layers": list(layers or []),
            "acc": 0.0, "acc_leak": 0.0,
            "head_T": T_REF, "head_gas_mol": 0.0, "vent_area": 0.0}


def total_ml(v):
    return math.fsum(layer_ml(l) for l in v["layers"])


def level(v):
    """Height of the surface above the inside bottom, m."""
    return total_ml(v) * 1e-6 / v["area"]


def owned(v):
    """What the cup's contents add to the ledger (records_owned): ml and grams
    per species, and J."""
    ml, grams = {}, {}
    for l in v["layers"]:
        for s, x in l["sp"].items():
            ml.setdefault(s, []).append(x)
            grams.setdefault(s, []).append(x * l["rho"])
    return {"ml": {s: math.fsum(x) for s, x in ml.items()},
            "grams": {s: math.fsum(x) for s, x in grams.items()},
            "J": math.fsum(layer_J(l) for l in v["layers"])}


def _same_kind(a, b, tol_T=0.5):
    return set(a["sp"]) == set(b["sp"]) and abs(a["rho"] - b["rho"]) < 1e-9 \
        and abs(a["T"] - b["T"]) < tol_T


def merge_layers(a, b):
    """Two layers of the same kind into one: ml add per species, T from the sum
    of the energies (so J is exact)."""
    sp = {}
    for k in list(a["sp"]) + [k for k in b["sp"] if k not in a["sp"]]:
        sp[k] = a["sp"].get(k, 0.0) + b["sp"].get(k, 0.0)
    ga, gb = layer_grams(a), layer_grams(b)
    rho = (ga + gb) / (math.fsum(sp.values()))
    cap = ga * a["cp"] + gb * b["cp"]
    T = T_REF + (layer_J(a) + layer_J(b)) / cap if cap > 0 else a["T"]
    return {"sp": sp, "rho": rho, "T": T, "cp": cap / (ga + gb)}


def add_layer(v, layer):
    """Put a layer into the vessel: denser liquids lie lower. A layer of the
    same kind as its neighbour (same species, density and T within 0.5 K)
    merges into it."""
    layers = v["layers"]
    k = len(layers)
    while k > 0 and layers[k - 1]["rho"] < layer["rho"] - 1e-12:
        k -= 1
    if k > 0 and _same_kind(layers[k - 1], layer):
        layers[k - 1] = merge_layers(layers[k - 1], layer)
    elif k < len(layers) and _same_kind(layers[k], layer):
        layers[k] = merge_layers(layers[k], layer)
    else:
        layers.insert(k, layer)


def take_from(layer, ml):
    """(taken, rest) of a layer: `ml` taken, every species shared by an exact
    subtraction (taken + rest == layer, bit for bit, per species)."""
    tot = layer_ml(layer)
    f = min(max(ml / tot, 0.0), 1.0) if tot > 0 else 0.0
    ta, re = {}, {}
    for s, x in layer["sp"].items():
        ta[s], re[s] = exact_split(x, f)
    base = {k: layer[k] for k in ("rho", "T", "cp")}
    return dict(base, sp=ta), dict(base, sp=re)


def take_top(v, ml):
    """Remove `ml` from the TOP layer first, then the next one down. Returns
    the list of fragments (each a layer dict). Never removes more than there
    is."""
    out = []
    need = float(ml)
    while need > 0 and v["layers"]:
        top = v["layers"][-1]
        have = layer_ml(top)
        if have <= need:
            out.append(v["layers"].pop())
            need -= have
        else:
            t, r = take_from(top, need)
            out.append(t)
            v["layers"][-1] = r
            need = 0.0
    return out


def pour(v, ml):
    """Tip `ml` out of the top. Same as take_top; the fragments are what lands
    in a second cup (add them there with catch)."""
    return take_top(v, ml)


def catch(v, fragments):
    """A cup catching parcels at its mouth: each fragment is added on top
    (denser ones sink below lighter layers)."""
    for f in fragments:
        add_layer(v, f)


# ── weir spill ────────────────────────────────────────────────────────────────
def weir_k(b, cd=CD_WEIR, g=G):
    """Weir constant k in Q = k H^1.5, k = (2/3) C_d b sqrt(2 g)."""
    return (2.0 / 3.0) * cd * b * math.sqrt(2.0 * g)


def weir_flow(H, b, cd=CD_WEIR, g=G):
    """Q = (2/3) C_d b sqrt(2 g) H^1.5, m3/s, for a head H over the lip."""
    return weir_k(b, cd, g) * max(H, 0.0) ** 1.5


def weir_head_after(H0, area, b, dt, cd=CD_WEIR):
    """Closed form for a vertical-walled cup: A dH/dt = -k H^1.5, so
    H(t) = (H0^-0.5 + k t/(2 A))^-2."""
    if H0 <= 0:
        return 0.0
    k = weir_k(b, cd)
    return (H0 ** -0.5 + k * dt / (2.0 * area)) ** -2


def spill_step(v, dt):
    """One tick of spill over the lip. The volume owed is added to `acc`; whole
    parcels leave from the top, never taking the level below the lip; the
    remainder stays. Returns the list of fragments that left."""
    H0 = level(v) - v["lip_h"]
    if H0 > 0:
        H1 = weir_head_after(H0, v["area"], v["lip_w"], dt)
        v["acc"] += v["area"] * (H0 - H1) * 1e6
    return _release(v, "acc")


def _release(v, key):
    p = v["parcel"]
    above = max(total_ml(v) - v["lip_h"] * v["area"] * 1e6, 0.0) if key == "acc" \
        else total_ml(v)
    n = int(min(math.floor(v[key] / p + 1e-12), math.floor(above / p + 1e-12)))
    if n <= 0:
        return []
    v[key] -= n * p
    return take_top(v, n * p)


# ── leaks (a hole in the wall) ────────────────────────────────────────────────
def layer_at(v, h):
    """Index of the layer holding height h above the bottom (the top one if h
    is above the surface)."""
    z = 0.0
    for k, l in enumerate(v["layers"]):
        z += layer_ml(l) * 1e-6 / v["area"]
        if h < z:
            return k
    return len(v["layers"]) - 1


def leak_step(v, hole_h, w, length, t_wall, dt, mu=1.0e-3):
    """Leak through a crack at height hole_h: min(orifice, slit) of the head
    above the hole. The liquid that leaves is the layer AT the hole. Returns
    the fragments released this tick (whole parcels; the rest is owed)."""
    head = level(v) - hole_h
    if head <= 0 or not v["layers"]:
        return []
    rho = v["layers"][layer_at(v, hole_h)]["rho"] * 1000.0
    v["acc_leak"] += leak_rate(w, length, t_wall, head, rho, mu) * dt * 1e6
    return _release_at(v, hole_h)


def _release_at(v, hole_h):
    p = v["parcel"]
    out = []
    while v["acc_leak"] >= p - 1e-12 and v["layers"]:
        k = layer_at(v, hole_h)
        lay = v["layers"][k]
        if layer_ml(lay) <= p:
            out.append(v["layers"].pop(k))
            v["acc_leak"] -= layer_ml(out[-1])
        else:
            t, r = take_from(lay, p)
            out.append(t)
            v["layers"][k] = r
            v["acc_leak"] -= p
    return out


# ── flow in a pipe ────────────────────────────────────────────────────────────
def friction_factor(Re):
    """Darcy f: 64/Re laminar (Re <= 2300); Blasius 0.316 Re^-0.25 turbulent
    (4000 to 1e5 and held beyond, (u)); in 2300-4000 the larger of the two, so
    the pressure needed rises monotonically with the flow (at Re 2300 they are
    0.0278 and 0.0456, (c))."""
    if Re <= 0:
        return math.inf
    lam = 64.0 / Re
    blas = 0.316 * Re ** -0.25
    if Re <= RE_LAM:
        return lam
    return max(lam, blas)


def conduit_speed(dh, L, D, rho=1000.0, mu=1.0e-3, k_in=K_ENTRANCE,
                  k_out=K_EXIT, g=G):
    """Mean speed through a pipe of length L and bore D under a head dh (m):
    dh = (1 + K + f L/D) v^2 / (2 g), solved by bisection (the right side is
    monotone in v). K = entrance + exit loss. Returns (v, Re, f)."""
    if dh <= 0:
        return 0.0, 0.0, math.inf
    K = k_in + k_out

    def head(v):
        Re = rho * v * D / mu
        return (1.0 + K + friction_factor(Re) * L / D) * v * v / (2.0 * g)
    lo, hi = 0.0, math.sqrt(2.0 * g * dh)
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if head(mid) < dh:
            lo = mid
        else:
            hi = mid
    v = 0.5 * (lo + hi)
    Re = rho * v * D / mu
    return v, Re, friction_factor(Re)


def conduit_flow(dh, L, D, **kw):
    """Volume flow, m3/s."""
    v, _, _ = conduit_speed(dh, L, D, **kw)
    return v * math.pi * D * D / 4.0


# ── vapour pressure and the siphon ────────────────────────────────────────────
def vapour_pressure_water(T_K):
    """Antoine, water 1-100 C: log10 P[mmHg] = 8.07131 - 1730.63/(233.426 + T_C).
    2.34 kPa at 20 C; 47.4 kPa at 80 C."""
    Tc = T_K - 273.15
    return 133.322 * 10.0 ** (8.07131 - 1730.63 / (233.426 + Tc))


def siphon_max_height(T_K=T_REF, rho=1000.0, p_atm=P_ATM, g=G):
    """The height of the summit above the surface at which a motionless column
    reaches the vapour pressure: (p_atm - p_v)/(rho g). 10.1 m of water at 20 C
    at sea level (c); 5.5 m at 80 C."""
    return (p_atm - vapour_pressure_water(T_K)) / (rho * g)


def siphon(z_surface, z_summit, z_out, L, D, z_in=None, T_K=T_REF, primed=True,
           rho=1000.0, mu=1.0e-3, p_atm=P_ATM, L_up=None, p_vap=None, g=G):
    """A siphon from a surface at z_surface over a summit at z_summit to an
    outlet at z_out (m). Runs only while primed. Air entering (the surface
    below the inlet z_in) clears the prime. It stops when the pressure at the
    summit falls to the vapour pressure. Returns {"v", "Q", "primed", "p_summit",
    "reason"}; the new `primed` state goes back into the record."""
    z_in = z_surface - 0.01 if z_in is None else z_in
    pv = vapour_pressure_water(T_K) if p_vap is None else p_vap
    out = {"v": 0.0, "Q": 0.0, "primed": primed, "p_summit": None, "reason": "ok"}
    if not primed:
        out["reason"] = "not primed"
        return out
    if z_surface < z_in:
        out.update(primed=False, reason="air")
        return out
    dh = z_surface - z_out
    if dh <= 0:
        out["reason"] = "no head"
        return out
    v, Re, f = conduit_speed(dh, L, D, rho, mu, g=g)
    up = (L / 2.0) if L_up is None else L_up
    p_s = p_atm - rho * g * (z_summit - z_surface) \
        - (1.0 + K_ENTRANCE + f * up / D) * rho * v * v / 2.0
    out["p_summit"] = p_s
    if p_s <= pv:
        out.update(primed=False, reason="vapour")
        return out
    out.update(v=v, Q=v * math.pi * D * D / 4.0)
    return out
