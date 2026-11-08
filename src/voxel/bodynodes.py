"""THE COMPACT BODY: 43 nodes (wave 13, T7 section 3.12).

Three nodes for each of the 14 body zones (SKIN, SHELL, CORE) and the central
blood pool: 3 x 14 + 1 = 43. It is the same physics as the voxel law
(bodyheat.py) and shares its code (tissue, thermoreg, clothing, bodyair,
digestion, energetics), but the zone is one lump, so a night costs a few
milliseconds instead of a million voxel updates. It is

  * the SKIP: `advance` carries a still body over minutes, applying the room
    its heat, water, O2 and CO2 through the SAME sink the ticks use;
  * the HAND-OVER target: `to_nodes` / `from_nodes` move the body between the
    lattice and the nodes without making or losing a joule.

WHAT A NODE IS (and the honest limit)
  SKIN   the lattice's own skin node: the thin surface layer of every
         flesh voxel that has an open face (Cs). Its temperature is the real
         skin temperature that radiation, sweat and burns read.
  SHELL  the bulk of the voxels that have at least one open face.
  CORE   the voxels that have none (the heart of the trunk and thighs).
         A zone too thin to have any (a hand at 5 cm) has a CORE of zero
         capacity, which the solver treats as the same temperature as its
         shell; that is the lattice's own resolution, said once.
  POOL   the central blood: core temperature. Solved together with the
         nodes, backward Euler, stable for any dt.
  The compact model takes zone areas, open-face counts, a_z, view factors,
  capacities and the links between zones FROM THE LATTICE (`build_geom`), so
  it uses the same coefficients as the kernel. What it cannot share is the
  per-voxel distribution inside a layer (it is uniform) and the lattice's
  5 cm air cell.

Marks: (u) unverified number; (d) design choice.
"""
import math

import numpy as np

from . import bodyair as BA
from . import digestion as DG
from . import energetics as EN
from . import thermoreg as TR
from . import tissue as TI

ZONES = TI.ZONES
NZ = TI.NZONES
LAYERS = ("skin", "shell", "core")
POOL = 3 * NZ
N_NODES = POOL + 1                       # 43
NODE_NAMES = tuple(f"{z}/{l}" for z in ZONES for l in LAYERS) + ("pool",)
AMBIENT = 20.0                           # the lattice's reference temperature, C
SIGMA = BA.SIGMA

# the lattice FLESH row (materials.py): 1000 kg/m3, 3.5 J/(g K), 0.37 W/(m K)
RHOC_FLESH = 1000.0 * 3.5 * 1000.0       # J/(m3 K)
K_FLESH = 0.37
SKIN_T_DEPTH = 126.0                     # materials.SKIN_T[0], s
SKIN_DEPTH = math.sqrt(K_FLESH / RHOC_FLESH * SKIN_T_DEPTH)       # 3.65 mm

# THE FITTED CONSTANTS (spec R1: "fitting is allowed, hiding is not"). Each is named,
# reported on its own, and recorded beside the human-data fixtures
# (tests/data/human_cases.json, block "fit"; a test checks they are equal). They were
# fitted on the STEADY cases only; the transients, humid, hot, immersion, dehydration and
# corpse cases were held out. The fit is the WEAKER, human-data-only one (the JOS-3
# oracle was not run, decision D8).
#  HC_FLESH, RAD_VIEW  multiply the ISO natural-convection law and the skin emissivity. They are
#                      1.0 and NOT fitted. (An earlier fit let them fall to 0.41 and 0.40 to make
#                      the cases fit, which hid a wrong heat balance: a nude body at -30 C stayed
#                      warm. Spec R1 is about the 5 cm AIR CELL of the lattice, not about people:
#                      if the lattice needs a calibration it is measured there, phase B/C6, and
#                      enters through the hook tables, never through the human-data fit.)
#  DRY_WETTEDNESS_FIT  the wetting of dry skin by diffusion. bodyair uses 0.06 (u 0.06-0.1);
#                      the fit scales ONLY the diffusion part of the evaporation, never the sweat.
#  PRM_SCALE           population scale on the controller constants of thermoreg.PARAMS (those
#                      ranges are a literature guess, u, in thermoreg.py); a person's own draw is
#                      multiplied by it, so people still differ.
DRY_WETTEDNESS_FIT = 0.016
HC_FLESH = 1.0
RAD_VIEW = 1.0
PRM_SCALE = {'sw_k_c': 0.662, 'sw_k_s': 2.255, 'sw_k_cs': 0.895, 'vd_k_dil': 0.88, 'vc_k_con': 2.056, 'sh_k1': 2.893, 'sh_k2': 0.25, 'sh_max_w_kg': 0.718, 'set_core': 1.0}
RUNOFF_K = 0.5                           # Gagge: the efficiency of sweating is 1 - RUNOFF_K w^2 (u, not fitted)
SHELL_VASO = 1.0                         # share of the shell blood flow (fat and outer muscle) that follows the zone vessel reflex: cold closes it, heat opens it (not fitted)
SHELL_VASO_CAP = 3.0                     # the reflex multiplies the shell flow by at most this
RMV_L_MIN = 8.5                          # physiology.BREATH["rmv_l_min"], at 1 MET
FILM_CAP_G_M2 = 30.0                     # water a skin can hold before it drips (u)
SUB_S = 30.0                             # longest sub-step of the skip, s
A_Z_RANGE = (0.4, 3.0)                   # outside this a zone is under-resolved


def nidx(zi, layer):
    return 3 * zi + layer


# ============================================================ geometry ======
_FACES = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])


def cell_layout(cells, zone_idx):
    """For a body's cells (N,3 ints): how many of the six neighbours are NOT
    body (open faces), the list of neighbour pairs (i, j) with i < j, and the
    (N,6) table of which faces (+x -x +y -y +z -z) are open. Pure numpy on a
    local grid."""
    cells = np.asarray(cells, dtype=np.int64)
    lo = cells.min(axis=0) - 1
    hi = cells.max(axis=0) + 2
    shape = tuple(hi - lo)
    grid = -np.ones(shape, dtype=np.int64)
    c = cells - lo
    grid[c[:, 0], c[:, 1], c[:, 2]] = np.arange(len(cells))
    n_open = np.zeros(len(cells), dtype=np.int64)
    open_dir = np.zeros((len(cells), 6), dtype=np.int64)   # which of the six faces are open
    pi, pj = [], []
    for k, f in enumerate(_FACES):
        nb = c + f
        j = grid[nb[:, 0], nb[:, 1], nb[:, 2]]
        n_open += (j < 0)
        open_dir[:, k] = (j < 0)
        sel = (j >= 0) & (np.arange(len(cells)) < j)
        pi.append(np.nonzero(sel)[0])
        pj.append(j[sel])
    return n_open, np.concatenate(pi), np.concatenate(pj), open_dir


def build_geom(cells, zone_idx, sheet, dx=0.05, rhoc=RHOC_FLESH, k=K_FLESH,
               f_self=None, c_vox=None):
    """Everything the compact model (and the hand-over) needs from a body in
    the lattice. `cells` (N,3), `zone_idx` (N,) in 0..13, `sheet` the person's
    Sheet. `c_vox` the lattice heat capacity of each voxel (J/K), default
    rho c dx^3.

    Per voxel (geom["vox"]): n_open, shell (bool), Cs_v (skin capacity), Cb_v,
    C_v, A_v (open area). Per zone: capacities of the three layers, volumes,
    open area A_open, true area A_true = AREA_SHARE x BSA, the area factor
    a_z = A_true / A_open (and `under` if it is outside [0.4, 3]), the H1
    conductance skin-to-shell G_sb = A_true / R_sc, the shell-to-core
    conduction G_bc, and the links between zones (shell to shell)."""
    cells = np.asarray(cells, dtype=np.int64)
    zi = np.asarray(zone_idx, dtype=np.int64)
    n = len(cells)
    n_open, pi, pj, open_dir = cell_layout(cells, zi)
    vol = dx ** 3
    C_v = np.full(n, rhoc * vol) if c_vox is None else np.asarray(c_vox, dtype=np.float64)
    shell = n_open > 0
    fill = np.clip(C_v / (rhoc * vol), 1e-6, 1.0)      # a partly filled voxel is a thinner thing
    V_v = C_v / rhoc                                   # the flesh volume it holds, m3
    A_v = n_open * dx * dx * np.sqrt(fill)             # (lattice: area x sqrt(fill))
    # the lattice's skin: a layer SKIN_DEPTH deep over the open faces, at most half a cell
    Cs_v = np.where(shell, np.minimum(rhoc * SKIN_DEPTH * A_v, 0.5 * C_v), 0.0)
    Cb_v = C_v - Cs_v
    zp = TI.zone_params(sheet)
    bsa = TI.bsa_dubois(sheet.mass_kg, sheet.height_m)
    g = {"dx": dx, "rhoc": rhoc, "k": k, "n_vox": np.zeros(NZ, dtype=np.int64),
         "C_skin": np.zeros(NZ), "C_shell": np.zeros(NZ), "C_core": np.zeros(NZ),
         "V_shell": np.zeros(NZ), "V_core": np.zeros(NZ), "A_open": np.zeros(NZ),
         "A_true": np.zeros(NZ), "a_z": np.zeros(NZ), "under": np.zeros(NZ, dtype=bool),
         "G_sb": np.zeros(NZ), "G_bc": np.zeros(NZ), "f_self": np.zeros(NZ),
         "links": [], "R_sc": np.zeros(NZ)}
    for z in range(NZ):
        m = zi == z
        if not m.any():
            continue
        g["n_vox"][z] = m.sum()
        g["C_skin"][z] = Cs_v[m].sum()
        g["C_shell"][z] = Cb_v[m & shell].sum()
        g["C_core"][z] = C_v[m & ~shell].sum()
        g["V_shell"][z] = V_v[m & shell].sum()
        g["V_core"][z] = V_v[m & ~shell].sum()
        g["A_open"][z] = A_v[m].sum()
        zpar = zp[ZONES[z]]
        g["A_true"][z] = zpar.area_m2
        g["R_sc"][z] = zpar.R_sc
        a = zpar.area_m2 / g["A_open"][z] if g["A_open"][z] > 0 else 0.0
        g["a_z"][z] = a
        g["under"][z] = not (A_Z_RANGE[0] <= a <= A_Z_RANGE[1])
        g["G_sb"][z] = zpar.area_m2 / zpar.R_sc
    # conduction: a face between two voxels is k dx^2/dx = k dx
    zz = {}
    for a, b in zip(pi, pj):
        za, zb = zi[a], zi[b]
        if za == zb:
            if shell[a] != shell[b]:
                g["G_bc"][za] += k * dx
        else:                              # any face between zones joins their shells
            key = (min(za, zb), max(za, zb))
            zz[key] = zz.get(key, 0.0) + k * dx
    g["links"] = [(a, b, G) for (a, b), G in sorted(zz.items())]
    g["f_self"] = np.zeros(NZ) if f_self is None else np.asarray(f_self, dtype=np.float64)
    g["vox"] = {"n_open": n_open, "shell": shell, "Cs_v": Cs_v, "Cb_v": Cb_v, "C_v": C_v,
                "A_v": A_v, "V_v": V_v, "zone": zi, "open_dir": open_dir}
    return g


# ---- a body with no lattice at all (the JOS-3 fixtures, the night of a sleeper) ----
# cross-section (m2) through which each neighbouring pair of zones conducts (d)
_LINK_AREA = {"head": 0.010, "upper arm": 0.006, "forearm": 0.004, "hand": 0.003,
              "thigh": 0.012, "shin": 0.006, "foot": 0.004}
# share of each zone's radiating surface that looks at the body itself (arm
# against the torso, thigh against thigh): the compact stand-in for the ray
# cast; its area-weighted mean is 1 - 0.71 (ISO f_eff of a standing body)
_F_SELF = {"head": 0.05, "torso": 0.12, "upper arm": 0.42, "forearm": 0.35, "hand": 0.30,
           "thigh": 0.45, "shin": 0.35, "foot": 0.30}
_LINKS = (("torso", "head"), ("torso", "{s} upper arm"), ("{s} upper arm", "{s} forearm"),
          ("{s} forearm", "{s} hand"), ("torso", "{s} thigh"), ("{s} thigh", "{s} shin"),
          ("{s} shin", "{s} foot"))


def geom_from_sheet(sheet, shell_frac=0.55):
    """The compact model with no lattice: capacities from the zone masses,
    areas from the sheet (a_z = 1), links from limb cross-sections. Used by
    the offline fixtures and by tests of the physics alone. Not exact to any
    lattice (the voxel mix of shell and core is a guess, `shell_frac`)."""
    zp = TI.zone_params(sheet)
    g = {"dx": 0.05, "rhoc": RHOC_FLESH, "k": K_FLESH, "n_vox": np.zeros(NZ, dtype=np.int64),
         "C_skin": np.zeros(NZ), "C_shell": np.zeros(NZ), "C_core": np.zeros(NZ),
         "V_shell": np.zeros(NZ), "V_core": np.zeros(NZ), "A_open": np.zeros(NZ),
         "A_true": np.zeros(NZ), "a_z": np.ones(NZ), "under": np.zeros(NZ, dtype=bool),
         "G_sb": np.zeros(NZ), "G_bc": np.zeros(NZ), "f_self": np.zeros(NZ),
         "links": [], "R_sc": np.zeros(NZ), "vox": None}
    for z, zn in enumerate(ZONES):
        p = zp[zn]
        thin = zn.endswith(("hand", "foot"))        # too small for a core at this resolution
        cs = p.area_m2 * RHOC_FLESH * SKIN_DEPTH
        rest = p.mass_kg * p.c_mix - cs
        frac = 1.0 if thin else shell_frac
        g["C_skin"][z] = cs
        g["C_shell"][z] = rest * frac
        g["C_core"][z] = rest - g["C_shell"][z]
        g["V_shell"][z] = p.volume_m3 * frac
        g["V_core"][z] = p.volume_m3 - g["V_shell"][z]
        g["A_open"][z] = g["A_true"][z] = p.area_m2
        g["R_sc"][z] = p.R_sc
        g["G_sb"][z] = p.area_m2 / p.R_sc
        g["f_self"][z] = _F_SELF[zn.split(" ", 1)[-1] if zn[:4] in ("left", "righ") else zn]
        g["G_bc"][z] = 0.0 if thin else K_FLESH * p.area_m2 * 0.8 / 0.04
    for side in ("left", "right"):
        for a, b in _LINKS:
            a, b = a.format(s=side), b.format(s=side)
            if side == "right" and b == "head":
                continue
            kb = b.split(" ", 1)[-1] if b[:4] in ("left", "righ") else b
            g["links"].append((ZONES.index(a), ZONES.index(b), K_FLESH * _LINK_AREA[kb] / 0.15))
    return g


# ================================================================ record ====
def new_heat_record(sheet, pid=0, name="p", T_pool=None):
    """p["heat"]: the plain-dict record the body law and the skip share.
    Deep-copy and snapshot safe (numbers, lists, dicts)."""
    sh = sheet.as_dict() if hasattr(sheet, "as_dict") else dict(sheet)
    prm = TR.person_params(pid, name)
    for k, f in PRM_SCALE.items():
        prm[k] = prm[k] * f
    return {
        "sheet": sh, "pid": pid, "name": name,
        "T_pool": 37.0 if T_pool is None else float(T_pool), "C_pool": TI.pool_capacity(TI.Sheet.from_dict(sh)),
        "prm": prm, "ctl": TR.new_state(), "flags": {"sweat": False, "shiver": False},
        "stores": DG.new_stores(sh["mass_kg"], sh["fat_frac"], sh["sex"]),
        "work": EN.new_book(), "clothes": [], "pending_J": 0.0, "set_offset": 0.0,
        "circ": 1.0, "film_g": [0.0] * NZ, "t_last": 0.0, "met": 1.0, "asleep": False,
        "alive": True, "fit": {}, "set_skin": dict(TR.SET_SKIN),
    }


def sheet_of(heat):
    return TI.Sheet.from_dict(heat["sheet"])


def comfort_T(geom, heat=None, T_core=36.9):
    """A comfortable resting temperature for every node (a start state):
    pool and core 36.9, shell between core and skin, skin at its set point."""
    T = np.zeros(N_NODES)
    for zi, zn in enumerate(ZONES):
        sk = TR.SET_SKIN[zn]
        T[nidx(zi, 0)] = sk
        T[nidx(zi, 1)] = sk + 0.6 * (T_core - sk) if not zn.endswith(("hand", "foot")) else sk + 0.25 * (T_core - sk)
        T[nidx(zi, 2)] = T_core - 0.2 if not zn.endswith(("hand", "foot")) else T[nidx(zi, 1)]
    T[POOL] = T_core
    return T


def caps_vector(geom, heat):
    """The 43 node capacities, J/K."""
    C = np.zeros(N_NODES)
    for z in range(NZ):
        C[nidx(z, 0)] = geom["C_skin"][z]
        C[nidx(z, 1)] = geom["C_shell"][z]
        C[nidx(z, 2)] = geom["C_core"][z]
    C[POOL] = heat["C_pool"]
    return C


def nodes_from_T(geom, heat, T):
    C = caps_vector(geom, heat)
    E = np.where(C > 0, C * (np.asarray(T) - AMBIENT), 0.0)
    return {"E": E, "C": C, "film_g": np.array(heat["film_g"], dtype=np.float64)}


def temps(nodes):
    """Node temperatures, C. A node of zero capacity (a core too thin to
    exist) is reported at the temperature of its shell."""
    C, E = nodes["C"], nodes["E"]
    T = np.where(C > 0, AMBIENT + E / np.where(C > 0, C, 1.0), np.nan)
    for z in range(NZ):
        if C[nidx(z, 2)] <= 0:
            T[nidx(z, 2)] = T[nidx(z, 1)]
    return T


def zone_means(nodes):
    """(skin, core) temperature of each zone, C: the skin node and the
    capacity-weighted mean of shell and core."""
    T = temps(nodes)
    C = nodes["C"]
    sk = np.array([T[nidx(z, 0)] for z in range(NZ)])
    cr = np.array([(C[nidx(z, 1)] * T[nidx(z, 1)] + C[nidx(z, 2)] * np.nan_to_num(T[nidx(z, 2)]))
                   / (C[nidx(z, 1)] + C[nidx(z, 2)]) for z in range(NZ)])
    return sk, cr


def mean_skin(nodes, geom):
    sk, _ = zone_means(nodes)
    a = geom["A_true"]
    return float(np.sum(sk * a) / np.sum(a))


# ============================================================ hand-over =====
def to_nodes(geom, heat, E_v, s_v):
    """From the voxels to the nodes, with NO energy made or lost.

    `E_v` the voxels' energy above 20 C (J, any dtype; summed in float64),
    `s_v` the skin offsets (K above the cell mean). Per voxel T = 20 + E/C,
    Ts = T + s, Tb = T - s Cs/Cb: skin node = sum Cs (Ts - 20), shell =
    sum Cb (Tb - 20) over the voxels with a skin, core = sum E over the
    others. The three add to sum(E) algebraically. The pool is the record's."""
    v = geom["vox"]
    E = np.asarray(E_v, dtype=np.float64)
    s = np.asarray(s_v, dtype=np.float64)
    C, Cs, Cb, sh, zi = v["C_v"], v["Cs_v"], v["Cb_v"], v["shell"], v["zone"]
    e_skin = Cs * E / C + Cs * s
    e_shell = np.where(sh, E - e_skin, 0.0)
    nodes_E = np.zeros(N_NODES)
    for z in range(NZ):
        m = zi == z
        nodes_E[nidx(z, 0)] = e_skin[m & sh].sum()
        nodes_E[nidx(z, 1)] = e_shell[m & sh].sum()
        nodes_E[nidx(z, 2)] = E[m & ~sh].sum()
    nodes_E[POOL] = heat["C_pool"] * (heat["T_pool"] - AMBIENT)
    return {"E": nodes_E, "C": caps_vector(geom, heat),
            "film_g": np.array(heat["film_g"], dtype=np.float64)}


def from_nodes(geom, heat, nodes):
    """From the nodes to the voxels. Every voxel of a layer takes the layer's
    temperature (the profile inside a layer is flat; said in the module doc),
    so sum(E_v) equals the node energies exactly in float64. Returns
    (E_v, s_v) in float64; `exact_round` puts them in the lattice dtype."""
    v = geom["vox"]
    C, Cs, Cb, sh, zi = v["C_v"], v["Cs_v"], v["Cb_v"], v["shell"], v["zone"]
    T = temps(nodes)
    E = np.zeros(len(C))
    s = np.zeros(len(C))
    for z in range(NZ):
        m = zi == z
        Ts, Tb, Tc = T[nidx(z, 0)], T[nidx(z, 1)], T[nidx(z, 2)]
        ms, mc = m & sh, m & ~sh
        if ms.any():
            # exact sums: skin and shell energies are the nodes', spread by capacity
            es = nodes["E"][nidx(z, 0)] * Cs[ms] / Cs[ms].sum()
            eb = nodes["E"][nidx(z, 1)] * Cb[ms] / Cb[ms].sum()
            E[ms] = es + eb
            Tv = AMBIENT + E[ms] / C[ms]
            Tsv = AMBIENT + es / Cs[ms]
            s[ms] = Tsv - Tv
        if mc.any():
            E[mc] = nodes["E"][nidx(z, 2)] * C[mc] / C[mc].sum()
    return E, s


def exact_round(values, dtype):
    """Round an array into `dtype` carrying the rounding error forward, so the
    sum of the stored values is the sum of the wanted ones to within ONE
    spacing of the dtype (not N). float64 in float64 is the identity."""
    vals = np.asarray(values, dtype=np.float64)
    if np.dtype(dtype) == np.float64:
        return vals.copy()
    out = np.empty(len(vals), dtype=dtype)
    carry = 0.0
    for i in range(len(vals)):
        want = vals[i] + carry
        out[i] = want
        carry = want - float(out[i])
    return out


# ======================================================== the physics =======
def zone_env(env, zi, T_skin_mean_others):
    """The environment of one zone: global values, overridden by env["zones"]."""
    zn = ZONES[zi]
    d = {"T_air": env.get("T_air", 20.0), "rh": env.get("rh", 0.5), "v": env.get("v", 0.0),
         "T_rad": env.get("T_rad", env.get("T_air", 20.0)), "R_cl": 0.0, "f_cl": 1.0,
         "T_mrt": None, "h_fixed": None, "exposed": 1.0}
    d.update(env.get("zones", {}).get(zn, {}))
    return d


def skin_loss(Ts, e, area, f_self, T_self, sweat_g, film_g, dt):
    """Heat the skin of a zone gives up in one second terms, at skin temperature
    Ts. Returns (dry_W, latent_W, detail). `e` = zone_env. dry_W is convection
    plus radiation (positive = loss); latent_W the evaporation (positive =
    the skin cools; negative = condensation warms it)."""
    Ta = e["T_air"]
    if e["h_fixed"] is not None:                      # immersion: a fluid touches the skin
        return e["h_fixed"] * area * (Ts - Ta), 0.0, {"evap_g": 0.0, "diffuse_g": 0.0,
                                                        "film_g": film_g, "drip_g": 0.0}
    if e["T_mrt"] is not None:
        Tr = e["T_mrt"]
    else:                                              # surroundings plus the body's own skin in view
        Tr = (((1.0 - f_self) * (e["T_rad"] + 273.15) ** 4 + f_self * (T_self + 273.15) ** 4) ** 0.25) - 273.15
    hc_nat = float(BA.h_c_nat(Ts - Ta)) * HC_FLESH
    R = e["R_cl"]
    eps = BA.EMISS_SKIN * RAD_VIEW
    if R <= 0.0:
        hc = max(hc_nat, float(BA.h_c_forced(e["v"])))
        dry = hc * area * (Ts - Ta) + eps * SIGMA * area * ((Ts + 273.15) ** 4 - (Tr + 273.15) ** 4)
    else:
        hc = hc_nat
        hr = float(BA.h_r(Ts, Tr, eps, 1.0))
        h = hc + hr
        Top = (hc * Ta + hr * Tr) / h
        dry = area * (Ts - Top) / (R + 1.0 / (e["f_cl"] * h))
    ev = evap_zone(Ts, e, area, hc, sweat_g, film_g, dt)
    return dry, ev["latent_w"], ev


def evap_zone(Ts, e, area, hc, sweat_g, film_g, dt):
    """Evaporation (or condensation) of one zone's skin film over `dt` s at
    skin temperature Ts, with the garment's vapour resistance. The one code
    path the compact model and the voxel law share. Returns the
    bodyair.evaporation_step dict with the diffusion part scaled by the fitted
    DRY_WETTEDNESS_FIT."""
    R = e["R_cl"]
    emax = float(BA.evap_max(Ts, e["T_air"], e["rh"], hc, I_cl=R, area_m2=area))
    runoff = 0.0
    if sweat_g > 0.0 and emax > 0.0:
        # Sweat is not shed evenly: where a zone makes more than the air can take, the surplus
        # runs off before it evaporates. Gagge's efficiency of sweating 1 - w^2/2, with
        # w = the evaporation this sweat would need over the most the air can take (u).
        w = min(sweat_g * float(BA.latent_heat(Ts)) / 1000.0 / dt / emax, 1.0)
        runoff = sweat_g * RUNOFF_K * w * w
    ev = BA.evaporation_step(sweat_g - runoff, film_g, emax, dt, Ts, FILM_CAP_G_M2 * area)
    if runoff > 0.0:
        ev = dict(ev)
        ev["drip_g"] += runoff
    if ev["diffuse_g"] > 0.0:                          # the fit applies to the diffusion only
        ev = dict(ev)
        ev["diffuse_g"] *= DRY_WETTEDNESS_FIT / BA.DRY_WETTEDNESS
        ev["latent_w"] = (ev["evap_g"] + ev["diffuse_g"]) * float(BA.latent_heat(Ts)) / 1000.0 / dt
    return ev


def shell_flow_factor(vessel):
    """The multiplier of a zone's shell blood flow (subcutaneous fat and the outer muscle) from the
    vessel reflex: SHELL_VASO of the flow follows the reflex (cold closes it, heat opens it up to
    SHELL_VASO_CAP), the rest is fixed. The one function the compact model and the voxel law share."""
    f = (1.0 - SHELL_VASO) + SHELL_VASO * np.clip(vessel, 0.0, SHELL_VASO_CAP)
    return float(f) if np.ndim(f) == 0 else f


def tissue_heat(geom, heat, T, sheet, zp, circ=1.0):
    """Basal heat of the shell and core of every zone (W) at their temperatures."""
    q = np.zeros(N_NODES)
    for z, zn in enumerate(ZONES):
        p = zp[zn]
        for lay, V in ((1, geom["V_shell"][z]), (2, geom["V_core"][z])):
            if V > 0:
                q[nidx(z, lay)] = float(TI.metab_power(p.q_m, V, T[nidx(z, lay)], sheet.bmr_scale))
    return q


def advance(heat, nodes, geom, env, dt, met=1.0, hour=12.0, sub=SUB_S, act_zone_W=None):
    """Carry the body over `dt` seconds in sub-steps of at most `sub` s.

    Mutates `nodes` (E, film_g) and `heat` (pool temperature, controller
    state, stores, flags). `env` = {"T_air", "rh", "v", "T_rad", optional
    "zones": {zone: {"R_cl", "f_cl", "T_mrt", "h_fixed", ...}}}; it may be a
    callable t -> env. Returns the quantities the room must be given, the
    SAME ones the tick path gives the same sink:

      vapour_g    water that left as vapour (negative = condensed from the air)
      latent_J    its latent heat (leaves the body with it)
      sensible_J  convection + radiation + breath, to the room
      heat_J      latent_J + sensible_J: what the sink adds
      o2_l, co2_l gas drawn from / added to the air (positive numbers each)
      drip_g      sweat that ran off (the pour topic places it)
      zone_W      {zone: mean W given to the room}
      events      [(verb, "on"|"off")] in order
      stores_J    chemical energy drawn from the stores
    """
    sheet = sheet_of(heat)
    zp = TI.zone_params(sheet)
    mass = sheet.mass_kg
    prm = heat["prm"]
    st = heat["stores"]
    T_all = temps(nodes)
    C = nodes["C"]
    out = {"vapour_g": 0.0, "latent_J": 0.0, "sensible_J": 0.0, "heat_J": 0.0, "o2_l": 0.0,
           "co2_l": 0.0, "drip_g": 0.0, "zone_W": {z: 0.0 for z in ZONES}, "events": [],
           "stores_J": 0.0, "metabolic_J": 0.0, "breath_water_g": 0.0, "sweat_g": 0.0,
           "tef_J": 0.0, "unmet_J": 0.0}
    t_done = 0.0
    stores_J0 = DG.stores_J(st)
    while t_done < dt - 1e-9:
        h = min(sub, dt - t_done)
        e_now = env(heat["t_last"]) if callable(env) else env
        _substep(heat, nodes, geom, e_now, h, met, hour + (heat["t_last"]) / 3600.0, zp,
                 sheet, mass, prm, st, out, act_zone_W)
        heat["t_last"] += h
        t_done += h
    out["heat_J"] = out["latent_J"] + out["sensible_J"]
    out["stores_J"] = stores_J0 - DG.stores_J(st)
    out["zone_W"] = {z: v / dt for z, v in out["zone_W"].items()}
    heat["T_pool"] = float(temps(nodes)[POOL])
    heat["film_g"] = nodes["film_g"].tolist()
    return out


def _substep(heat, nodes, geom, env, dt, met, hour, zp, sheet, mass, prm, st, out, act_zone_W):
    T = temps(nodes)
    C = nodes["C"]
    Tp = T[POOL]
    sk_T = {ZONES[z]: T[nidx(z, 0)] for z in range(NZ)}
    a_tot = float(np.sum(geom["A_true"]))
    Tsk_mean = float(sum(sk_T[ZONES[z]] * geom["A_true"][z] for z in range(NZ)) / a_tot)
    # --- controllers ---
    set_core = TR.set_points(prm, hour, heat["asleep"], heat["set_offset"])
    dehyd = DG.deficit_pct(st, mass)
    live = heat["alive"]
    if live:
        sig = TR.signals(Tp + TR.DT_BRAIN, sk_T, set_core, heat["set_skin"],
                         {ZONES[z]: geom["A_true"][z] for z in range(NZ)}, dehyd)
        _, ctl = TR.controllers(sig, prm, heat["ctl"], dt, mass, Tp, 1.0)
        out["events"] += TR.events(prm, ctl, mass, heat["flags"])
    else:                                             # the dead sweat, shiver and breathe no more
        ctl = TR.Outputs({z: 0.0 for z in ZONES}, {z: 1.0 for z in ZONES}, 0.0)
    # --- power: ONE source (energetics) ---
    q_t = tissue_heat(geom, heat, T, sheet, zp) if live else np.zeros(N_NODES)
    P_tissue = float(q_t.sum())
    mb = EN.met_b_of(P_tissue, mass) if P_tissue > 0 else 0.9
    floor = EN.rest_floor_w(P_tissue, mass, sheet.height_m, not heat["asleep"])
    P_gait = EN.p_gait(met, mb, mass, floor) if live else 0.0
    P_shiv = ctl.shiver_w
    P_act = 0.0
    zw = {ZONES[z]: geom["C_shell"][z] + geom["C_core"][z] for z in range(NZ)}
    Q_extra = np.zeros(N_NODES)                      # gait + shiver + acts, W per node
    for z, w in EN.heat_share(EN.USE["gait"], zw).items():
        _deposit(Q_extra, geom, ZONES.index(z), P_gait * w)
    for z, w in EN.heat_share(EN.USE["shiver"], zw).items():
        _deposit(Q_extra, geom, ZONES.index(z), P_shiv * w)
    if act_zone_W:
        for z, w in act_zone_W.items():
            _deposit(Q_extra, geom, ZONES.index(z), w)
            P_act += w
    # split each zone's extra between shell and core by volume
    P_total = P_tissue + P_gait + P_shiv + P_act
    # breath: ventilation follows P_total (shiver, fever and work raise it)
    V_E = RMV_L_MIN * P_total / (EN.MET_W_PER_KG * mass)
    T_in, rh_in = env["T_air"], env["rh"]
    br = BA.breath_loss(V_E, T_in, rh_in, T_core=Tp) if live else \
        {"sens_w": 0.0, "lat_w": 0.0, "water_g_s": 0.0}
    # --- stores pay for it ---
    sweat_g_z = {z: ctl.sweat_g_s[z] * dt for z in ZONES}
    sweat_tot = sum(sweat_g_z.values())
    film = nodes["film_g"]
    if live:
        dg = DG.step(st, dt, P_total, mass_kg=mass, water_loss_ml=0.0)
    else:
        dg = {"capacity": 1.0, "o2_L": 0.0, "co2_L": 0.0, "unmet_J": 0.0, "tef_J": 0.0}
    cap = dg["capacity"]
    out["o2_l"] += dg["o2_L"]
    out["co2_l"] += dg["co2_L"]
    out["metabolic_J"] += P_total * dt * cap
    out["unmet_J"] += dg["unmet_J"]
    tef_w = dg["tef_J"] / dt
    out["tef_J"] += dg["tef_J"]
    # --- node sources ---
    Q = np.zeros(N_NODES)
    Q += (q_t + Q_extra) * cap
    _deposit(Q, geom, ZONES.index("torso"), tef_w)
    # heat that the digestion made but the demand did not carry (DNL, TEF) is in tef_w
    # --- perfusion ---
    circ = heat["circ"]
    g = np.zeros(N_NODES)
    eps = np.zeros(N_NODES)
    for z, zn in enumerate(ZONES):
        p = zp[zn]
        for lay, V in ((1, geom["V_shell"][z]), (2, geom["V_core"][z])):
            n = nidx(z, lay)
            if V <= 0:
                continue
            w0 = p.w_b_core
            wb = w0 * (shell_flow_factor(heat["ctl"]["vessel"][zn]) if lay == 1 else 1.0)
            q_work_dens = Q_extra[n] * cap / V
            if q_work_dens > 0:                  # work hyperemia adds to the basal flow, capped on the UNSCALED basal
                wb += EN.hyperemia_w(w0, q_work_dens) - w0
            g[n] = TI.perfusion_g(wb, V, float(TI.perfusion_factor(T[n], 0.0, circ)))
            eps[n] = p.eps_cc
        ns = nidx(z, 0)
        dv = geom["A_true"][z] * TI.SKIN_M
        g[ns] = TI.perfusion_g(p.w_b_skin, dv, float(TI.perfusion_factor(T[ns], 0.0, circ)) * ctl_vessel(heat, zn))
        eps[ns] = p.eps_cc
    geff = g * (1.0 - eps)
    # --- skin: iterate the linearised environment loss ---
    Ts = np.array([T[nidx(z, 0)] for z in range(NZ)])
    zenv = [zone_env(env, z, Tsk_mean) for z in range(NZ)]
    Tnew = T.copy()
    lin = None
    for it in range(4):
        A = np.zeros((N_NODES, N_NODES))
        b = np.zeros(N_NODES)
        lin = []
        for z in range(NZ):
            ns, nb, nc = nidx(z, 0), nidx(z, 1), nidx(z, 2)
            area = geom["A_true"][z] * zenv[z]["exposed"]
            T0 = Tnew[ns]
            dry0, lat0, _ = skin_loss(T0, zenv[z], area, geom["f_self"][z], Tsk_mean, sweat_g_z[ZONES[z]], film[z], dt)
            dry1, lat1, _ = skin_loss(T0 + 0.05, zenv[z], area, geom["f_self"][z], Tsk_mean, sweat_g_z[ZONES[z]], film[z], dt)
            slope = max((dry1 + lat1 - dry0 - lat0) / 0.05, 0.0)
            lin.append((dry0, lat0, slope, T0))
            Gsb = geom["G_sb"][z]
            # skin node
            A[ns, ns] += C[ns] / dt + Gsb + geff[ns] + slope
            A[ns, nb] -= Gsb
            A[ns, POOL] -= geff[ns]
            b[ns] += C[ns] / dt * T[ns] - (dry0 + lat0 - slope * T0) + Q[ns]
            # shell node
            A[nb, nb] += C[nb] / dt + Gsb + geff[nb]
            A[nb, ns] -= Gsb
            A[nb, POOL] -= geff[nb]
            b[nb] += C[nb] / dt * T[nb] + Q[nb]
            if C[nc] > 0:
                Gbc = geom["G_bc"][z]
                A[nb, nb] += Gbc
                A[nb, nc] -= Gbc
                A[nc, nc] += C[nc] / dt + Gbc + geff[nc]
                A[nc, nb] -= Gbc
                A[nc, POOL] -= geff[nc]
                b[nc] += C[nc] / dt * T[nc] + Q[nc]
            else:
                A[nc, nc] += 1.0
                A[nc, nb] -= 1.0
        for (za, zb, G) in geom["links"]:
            na, nb_ = nidx(za, 1), nidx(zb, 1)
            A[na, na] += G
            A[na, nb_] -= G
            A[nb_, nb_] += G
            A[nb_, na] -= G
        # pool: breath is its sink
        A[POOL, POOL] += C[POOL] / dt
        b[POOL] += C[POOL] / dt * T[POOL] - (br["sens_w"] + br["lat_w"])
        for n in range(POOL):
            if geff[n] > 0:
                A[POOL, POOL] += geff[n]
                A[POOL, n] -= geff[n]
        Tnew = np.linalg.solve(A, b)
    # --- book the sub-step with the ACTUAL loss at the final temperature ---
    Ed = C * (Tnew - T)
    Ed[C <= 0] = 0.0
    evap_tot = diffuse_tot = 0.0
    lat_J = 0.0
    sens_J = 0.0
    drip = 0.0
    new_film = film.copy()
    for z in range(NZ):
        ns = nidx(z, 0)
        area = geom["A_true"][z] * zenv[z]["exposed"]
        dry, lat, ev = skin_loss(Tnew[ns], zenv[z], area, geom["f_self"][z], Tsk_mean,
                                 sweat_g_z[ZONES[z]], film[z], dt)
        d0, l0, slope, T0 = lin[z]
        lin_total = d0 + l0 + slope * (Tnew[ns] - T0)
        # the node equation used the linearised loss; give the difference to the skin
        # so the energy leaving the body is exactly what the sink is given
        corr = (lin_total - (dry + lat)) * dt
        Ed[ns] += corr
        evap_tot += ev["evap_g"]
        diffuse_tot += ev["diffuse_g"]
        lat_J += lat * dt
        sens_J += dry * dt
        out["zone_W"][ZONES[z]] += (dry + lat) * dt
        new_film[z] = ev["film_g"]
        drip += ev["drip_g"]
    nodes["film_g"] = new_film
    nodes["E"] = nodes["E"] + Ed
    ins_g = br["water_g_s"] * dt
    # body water: sweat made (it is on the skin now), the skin's own diffusion, breath
    st["water_ml"] -= sweat_tot + diffuse_tot + ins_g
    out["sweat_g"] += sweat_tot
    out["breath_water_g"] += ins_g
    out["vapour_g"] += evap_tot + diffuse_tot + ins_g
    out["latent_J"] += lat_J + br["lat_w"] * dt
    out["sensible_J"] += sens_J + br["sens_w"] * dt
    out["drip_g"] += drip
    out["zone_W"][ZONES[0]] += (br["sens_w"] + br["lat_w"]) * dt


def _deposit(Q, geom, zi, watts):
    """Put `watts` of work heat into a zone's shell and core nodes by their capacity, the way the
    voxel law shares it over the zone's voxels (a zone with no core puts it all in the shell)."""
    cs, cc = geom["C_shell"][zi], geom["C_core"][zi]
    f = cs / (cs + cc) if cs + cc > 0.0 else 1.0
    Q[nidx(zi, 1)] += watts * f
    Q[nidx(zi, 2)] += watts * (1.0 - f)


def ctl_vessel(heat, zn):
    return heat["ctl"]["vessel"][zn]


# ===================================================== energy bookkeeping ====
def node_energy(nodes):
    """Sum of all node energies, J above 20 C (pool included)."""
    return float(np.sum(nodes["E"]))


def kill_nodes(heat, nodes, geom):
    """Death in the compact model: the pool's energy goes into the torso and
    head shells (kept), the pool is left at 20 C with no coupling (circ = 0),
    controllers and stores stop. Energy is kept to the joule."""
    J = nodes["E"][POOL]
    nodes["E"][POOL] = 0.0
    zs = [ZONES.index("torso"), ZONES.index("head")]
    cs = np.array([nodes["C"][nidx(z, 1)] for z in zs])
    for z, c in zip(zs, cs):
        nodes["E"][nidx(z, 1)] += J * c / cs.sum()
    heat["T_pool"] = AMBIENT
    heat["circ"] = 0.0
    heat["alive"] = False
