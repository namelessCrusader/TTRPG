"""THE BODY LAW: a mixin that makes a body in the lattice a heat engine
(wave 13, T7 sections 2, 3.1-3.10).

Once a body second (every `bh_period` ticks; 40 = 1.000 s, staggered by pid)
each person with a heat record gets, on top of the lattice's own conduction:

  * Pennes bioheat in every flesh voxel, the central blood pool solved with
    the voxels in closed form (tissue.pennes_step), counter-current exchange,
    muscle hyperemia, perfusion as a function of tissue state;
  * ONE source of power (energetics): tissue q(T) + gait + shiver + acts, paid
    from float64 stores (digestion) that also hold the O2 and CO2 books;
  * the skin: sweat evaporation (signed, so condensation works), forced
    convection beyond the face law, the mean radiant temperature per zone from
    a ray cast, dermal perfusion with the vessel reflex;
  * breath through the AirSink (heat, vapour, O2, CO2);
  * thermoregulation (sweat, vessels, shiver) as 1 Hz controllers.

HOST CONTRACT. The mixin sits beside the lattice and asks it for:
  E (ndarray, ANY float dtype), heat_capacity() (J/K per voxel), _skin_field()
  (K above the cell mean), scale (voxel metres / 0.1), tick, shed, gone,
  persons. Optional: tick_s (default materials.TICK_S), bh_period, bh_env(p)
  -> {"T_air","rh","v","T_rad"}, bh_T_rad(unit_dir) -> C (radiant temperature
  of the world past the lattice edge), bh_set_hooks(pid, cells, zone_idx,
  tables) -> installs the face-law tables BH1 (core-to-skin conductance), BH2
  (convection law, emissivity, area factor) and BH3 (series clothing
  resistance) keyed by zone, airsink (an AirSink).
  Phase A: tested on a throwaway World subclass (zone map is a SIDE ARRAY, no
  carried field) and on the slab fixture in scenes_bodyheat.py. The real
  lattice hooks are phase B (BH1-BH3 in heat.py).

LEDGER (3.10). With the sign convention `shed -= J` for an injection:

    E_sum + shed + pool_J + stores_J + work_out  =  constant

except for the named injections (`bh_injected`: the warm-up). `bh_terms()`
lists every term. The sweat, breath and radiation leave through the sink;
heat the room gives the body (forced convection, mean radiant temperature)
enters as a negative sink heat.

Marks: (u) unverified; (d) design choice.
"""
import math

import numpy as np

from . import bodyair as BA
from . import bodynodes as BN
from . import digestion as DG
from . import energetics as EN
from . import thermoreg as TR
from . import tissue as TI
from .airsink import OpenWorldSink, deliver

try:                                     # clothes are optional: a nude body needs none
    from . import clothing as CL
except ImportError:                      # pragma: no cover
    CL = None

TICK_S_DEFAULT = 0.025
BODY_PERIOD = 40                          # ticks per body second at 0.025 s
T_INF = 20.0                              # the lattice radiates to a fixed 20 C
SIGMA = BA.SIGMA
FACES = BN._FACES

# core-temperature bands -> the share of consciousness left (3.11, u)
COLD_BAND = ((24.0, 0.0), (28.0, 0.05), (30.0, 0.3), (32.0, 0.8), (35.0, 1.0))
HOT_BAND = ((39.0, 1.0), (40.0, 0.9), (41.5, 0.4), (43.0, 0.0))

_RAY_DIRS = np.array([(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)
                      if (a, b, c) != (0, 0, 0)], dtype=np.float64)
_RAY_DIRS /= np.linalg.norm(_RAY_DIRS, axis=1)[:, None]
_RAY_STEP = 0.5                           # voxels per ray step
RAY_REFRESH_S = 120.0                     # where the rays end is looked at again this often, sim seconds


# ============================================================ zone maps =====
def zones_from_segs(segs):
    """(cells (N,3), zone_idx (N,)) from a person's segs {bone: (n,3) cells}:
    the zone of each voxel by its bone, the hand and the foot being the distal
    part of the forearm and shin bones by distance along the bone axis from
    the proximal (upper) end, measured in the REST pose (tissue.zone_of)."""
    cells, zones = [], []
    for bone in sorted(segs):
        c = np.asarray(segs[bone], dtype=np.int64)
        if len(c) == 0:
            continue
        zmin, zmax = c[:, 2].min(), c[:, 2].max()
        for row in c:
            along = (zmax - row[2]) / float(zmax - zmin + 1)
            z = TI.zone_of(bone, along)
            if z is None:
                continue
            cells.append(row)
            zones.append(TI.ZONE_INDEX[z])
    return np.array(cells, dtype=np.int64), np.array(zones, dtype=np.int64)


def zones_by_height(cells):
    """Fallback for a body with no segs: by height fraction of the flesh voxel
    list (head top 13 percent, torso, legs, arms by distance from the centre
    line; d). Hand and foot are not told apart."""
    cells = np.asarray(cells, dtype=np.int64)
    z = cells[:, 2]
    lo, hi = z.min(), z.max() + 1
    f = (z - lo) / float(hi - lo)
    cx = cells[:, 0].mean()
    side = np.where(cells[:, 0] < cx, "left", "right")
    out = np.zeros(len(cells), dtype=np.int64)
    for i in range(len(cells)):
        if f[i] >= 0.87:
            out[i] = TI.ZONE_INDEX["head"]
        elif f[i] >= 0.45:
            arm = abs(cells[i, 0] - cx) > 1.6
            if not arm:
                out[i] = TI.ZONE_INDEX["torso"]
            else:
                kind = "upper arm" if f[i] >= 0.65 else "forearm"
                out[i] = TI.ZONE_INDEX[f"{side[i]} {kind}"]
        else:
            kind = "thigh" if f[i] >= 0.23 else "shin"
            out[i] = TI.ZONE_INDEX[f"{side[i]} {kind}"]
    return cells, out


# ================================================================ sinks =====
class _WorldSink(OpenWorldSink):
    """The open-world sink bound to the host: heat goes to host.shed, grams and
    litres to host.gone. The tick path and the skip path call the same one."""

    def __init__(self, host, t_c, rh):
        super().__init__(gone=host.gone, t_c=t_c, rh=rh)
        self.host = host

    def add(self, vapour_g=0.0, heat_J=0.0, o2_l=0.0, co2_l=0.0, where=None):
        if vapour_g:
            self.gone["vapour"] = self.gone.get("vapour", 0.0) + vapour_g
        if o2_l:
            self.gone["o2_l"] = self.gone.get("o2_l", 0.0) + o2_l
        if co2_l:
            self.gone["co2_l"] = self.gone.get("co2_l", 0.0) + co2_l
        self.host.shed += heat_J


# ================================================================= mixin ====
class BodyHeatMixin:
    work_out = 0.0                        # joules of mechanical work given to the world

    # ---------------------------------------------------------------- setup -
    def _bh_cache(self):
        c = self.__dict__.get("_bh_runtime")
        if c is None:
            c = self.__dict__["_bh_runtime"] = {}
        return c

    def _bh_period(self):
        return int(getattr(self, "bh_period", BODY_PERIOD))

    def _bh_tick_s(self):
        return float(getattr(self, "tick_s", TICK_S_DEFAULT))

    def _bh_dx(self):
        return 0.1 * float(getattr(self, "scale", 0.5))

    def _airsink(self):
        s = getattr(self, "airsink", None)
        if s is not None:
            return s
        env = self._bh_env(None)
        return _WorldSink(self, env["T_air"], env["rh"])

    def _bh_env(self, p):
        f = getattr(self, "bh_env", None)
        d = {"T_air": 20.0, "rh": 0.5, "v": 0.0, "T_rad": 20.0}
        if f is not None:
            d.update(f(p))
        return d

    def bh_injected(self):
        """Named injections so far: {"body_warmup": J, ...} (the ledger test reads this)."""
        return self.__dict__.setdefault("_bh_inject", {})

    def body_heat_on(self, p, cells=None, zone_idx=None, sheet=None, warm=True, clothes=None,
                     segs=None, T_core=36.9):
        """Give a person a heat record. LAZY: nothing in add_person calls it.

        `cells` (N,3) are the body's voxels and `zone_idx` their zones (a SIDE
        ARRAY in phase A, kept in the record); from `p["segs"]` when not
        given, else by height. `warm` heats the flesh from the lattice's cold
        20 C to a comfortable state and books the joules as the named
        injection "body_warmup"."""
        if cells is None:
            segs = segs if segs is not None else p.get("segs")
            if segs is not None:
                cells, zone_idx = zones_from_segs(segs)
                if "origin" in p:
                    origin = np.asarray(p["origin"], dtype=np.int64)
                else:                              # scenes._person keeps segs as offsets from the body's corner
                    origin = self._bh_flesh_cells(p).min(axis=0)
                cells = cells + origin
            else:
                cells, zone_idx = zones_by_height(self._bh_flesh_cells(p))
        if zone_idx is None:
            cells, zone_idx = zones_by_height(cells)
        cells = np.asarray(cells, dtype=np.int64)
        idx = tuple(cells.T)
        C = np.asarray(self.heat_capacity(), dtype=np.float64)[idx]
        if sheet is None:
            mass = float(C.sum() / BN.RHOC_FLESH * 1000.0)       # kg: flesh volume x 1000 kg/m3
            sheet = TI.sheet("male", mass_kg=mass, height_m=float(
                (cells[:, 2].max() - cells[:, 2].min() + 1) * self._bh_dx()))
        h = BN.new_heat_record(sheet, p.get("pid", 0), p.get("name", "p"), T_pool=T_core)
        h["cells"] = cells
        h["zone"] = np.asarray(zone_idx, dtype=np.int64)
        h["clothes"] = list(clothes or [])
        h["tick_last"] = int(self.tick) - self._bh_period()
        h["set_skin_mean"] = float(np.mean(list(TR.SET_SKIN.values())))
        h["events"] = []
        h["a_z"] = []
        p["heat"] = h
        g = self._bh_geom(p, h, force=True)
        if warm:
            self._bh_warm(p, h, g, T_core)
        self._bh_hooks(p, h, g)
        return h

    def _bh_flesh_cells(self, p):                 # pragma: no cover - World only
        mask, sl = self._person_cells(p)
        c = np.argwhere(mask)
        off = np.array([s.start or 0 for s in sl])
        return c + off

    def _bh_geom(self, p, h, force=False):
        """The geometry of the body now (areas, a_z, capacities, links), cached
        outside the state against the cell list and the lattice capacity."""
        cache = self._bh_cache()
        key = (h["pid"], h["cells"].tobytes() if len(h["cells"]) < 20000 else id(h["cells"]))
        idx = tuple(h["cells"].T)
        C = np.asarray(self.heat_capacity(), dtype=np.float64)[idx]
        ent = cache.get(h["pid"])
        if ent is None or force or ent["key"] != key or not np.array_equal(ent["C"], C):
            sheet = BN.sheet_of(h)
            g = BN.build_geom(h["cells"], h["zone"], sheet, dx=self._bh_dx(), c_vox=C)
            ent = cache[h["pid"]] = {"key": key, "C": C, "geom": g,
                                     "zp": TI.zone_params(sheet),
                                     "zgrid": None, "rayc": None}
            h["a_z"] = [float(x) for x in g["a_z"]]
            h["under"] = [bool(x) for x in g["under"]]
        return ent["geom"]

    def _bh_warm(self, p, h, g, T_core):
        """Heat the flesh to a comfortable resting state: a named injection."""
        idx = tuple(h["cells"].T)
        nodes = BN.nodes_from_T(g, h, BN.comfort_T(g, h, T_core))
        E_v, s_v = BN.from_nodes(g, h, nodes)
        E0 = float(np.asarray(self.E[idx], dtype=np.float64).sum())
        self._bh_write(idx, E_v, s_v)
        gained = float(np.asarray(self.E[idx], dtype=np.float64).sum()) - E0
        h["T_pool"] = float(T_core)
        inj = self.bh_injected()
        inj["body_warmup"] = inj.get("body_warmup", 0.0) + gained + h["C_pool"] * (T_core - BN.AMBIENT)
        self.shed -= gained + h["C_pool"] * (T_core - BN.AMBIENT)

    def _bh_write(self, idx, E_v, s_v):
        dt = self.E.dtype
        self.E[idx] = BN.exact_round(E_v, dt).astype(dt, copy=False)
        sk = self._skin_field()
        sk[idx] = BN.exact_round(s_v, sk.dtype).astype(sk.dtype, copy=False)

    def _bh_hooks(self, p, h, g):
        """Hand the face-law tables to the host (BH1-BH3), if it takes them."""
        f = getattr(self, "bh_set_hooks", None)
        if f is None:
            return
        n = BN.NZ
        R_cl, f_cl = self._bh_clothing(h, 0.0)
        tables = {"gsb": g["a_z"] / np.maximum(g["R_sc"], 1e-9),         # BH1: W/(m2 lattice K)
                  "az": g["a_z"].copy(),                                  # BH2: area factor
                  "emiss": np.full(n, BA.EMISS_SKIN * BN.RAD_VIEW),       # BH2: 0.98 (x the lattice view calibration, 1.0 until C6 measures R1)
                  "conv_nat": np.full(n, 2.38 * BN.HC_FLESH),             # BH2: the ISO law (x the lattice calibration, 1.0 until C6)
                  "R_cl": R_cl, "f_cl": f_cl}                             # BH3
        f(h["pid"], h["cells"], h["zone"], tables)

    def _bh_clothing(self, h, v):
        n = BN.NZ
        R = np.zeros(n)
        f = np.ones(n)
        if CL is not None and h["clothes"]:
            for z, zn in enumerate(TI.ZONES):
                R[z] = CL.ensemble_resistance(h["clothes"], zn, v)
                f[z] = float(BA.f_cl_iso(R[z]))
        return R, f

    # --------------------------------------------------------- the law ------
    def _law_body_heat(self):
        period = self._bh_period()
        for p in self.persons:
            h = p.get("heat")
            if h is None:
                continue
            if (self.tick + p.get("pid", 0)) % period:
                continue
            self.bh_second(p, h)

    def bh_second(self, p, h):
        """One body step of `ticks since last` x tick_s seconds, any length
        (the pool is solved implicitly, so a skipped minute is stable)."""
        dt = (self.tick - h["tick_last"]) * self._bh_tick_s()
        h["tick_last"] = int(self.tick)
        if dt <= 0.0:
            return
        g = self._bh_geom(p, h)
        ent = self._bh_cache()[h["pid"]]
        zp = ent["zp"]
        sheet = BN.sheet_of(h)
        mass = sheet.mass_kg
        prm, st = h["prm"], h["stores"]
        idx = tuple(h["cells"].T)
        v = g["vox"]
        C, Cs, Cb, sh, zi, Av, Vv = v["C_v"], v["Cs_v"], v["Cb_v"], v["shell"], v["zone"], v["A_v"], v["V_v"]
        n = len(C)
        E = np.asarray(self.E[idx], dtype=np.float64)
        s = np.where(sh, np.asarray(self._skin_field()[idx], dtype=np.float64), 0.0)
        Tm = BN.AMBIENT + E / C
        Ts = np.where(sh, Tm + s, Tm)
        Tb = np.where(sh, Tm - s * Cs / np.where(sh, Cb, 1.0), Tm)
        Tp = h["T_pool"]
        env = self._bh_env(p)
        a_z = g["a_z"]
        # ---- zone skin means (area-weighted) and the controllers ----
        w_sk = np.where(sh, Av * a_z[zi], 0.0)
        zsk = np.zeros(BN.NZ)
        zarea = np.zeros(BN.NZ)
        np.add.at(zsk, zi, w_sk * Ts)
        np.add.at(zarea, zi, w_sk)
        zsk = np.where(zarea > 0, zsk / np.where(zarea > 0, zarea, 1.0), Tb[0])
        sk_T = {TI.ZONES[z]: float(zsk[z]) for z in range(BN.NZ)}
        a_true = {TI.ZONES[z]: float(g["A_true"][z]) for z in range(BN.NZ)}
        hour = 12.0 + self.tick * self._bh_tick_s() / 3600.0
        set_core = TR.set_points(prm, hour, h["asleep"], h["set_offset"])
        dehyd = DG.deficit_pct(st, mass)
        sig = TR.signals(Tp + TR.DT_BRAIN, sk_T, set_core, h["set_skin"], a_true, dehyd)
        if h["alive"]:
            _, ctl = TR.controllers(sig, prm, h["ctl"], dt, mass, Tp, 1.0)
            self._bh_events(p, h, ctl, mass)
        else:
            ctl = TR.Outputs({z: 0.0 for z in TI.ZONES}, {z: 1.0 for z in TI.ZONES}, 0.0)
        # ---- power: one source ----
        qz = np.array([zp[z].q_m for z in TI.ZONES])
        q_v = TI.metab_power(qz[zi], Vv, Tb, sheet.bmr_scale)
        if not h["alive"]:
            q_v = q_v * 0.0
        P_tissue = float(q_v.sum())
        mb = EN.met_b_of(P_tissue, mass) if P_tissue > 0 else 0.9
        met = h["met"]
        floor = EN.rest_floor_w(P_tissue, mass, sheet.height_m, not h["asleep"])
        P_gait = EN.p_gait(met, mb, mass, floor) if h["alive"] else 0.0
        P_shiv = ctl.shiver_w
        zw = {TI.ZONES[z]: float(C[(zi == z)].sum()) for z in range(BN.NZ)}
        work_v = np.zeros(n)                          # gait + shiver + continuous effort, W per voxel
        for power, row in ((P_gait, "gait"), (P_shiv, "shiver")):
            if power <= 0.0:
                continue
            for zn, wt in EN.heat_share(EN.USE[row], zw).items():
                z = TI.ZONE_INDEX[zn]
                m = zi == z
                if m.any():
                    work_v[m] += power * wt * C[m] / C[m].sum()
        # discrete acts: the joules billed since the last second, deposited exactly
        pend = h["work"]["pending_zone_J"]
        act_v = np.zeros(n)
        for zn, J in list(pend.items()):
            z = TI.ZONE_INDEX[zn]
            m = zi == z
            if m.any():
                act_v[m] += J / dt * C[m] / C[m].sum()
        act_J = float(sum(pend.values()))
        pend.clear()
        P_act_w = act_J / dt
        P_demand = P_tissue + P_gait + P_shiv
        if h["alive"]:
            dg = DG.step(st, dt, P_demand, mass_kg=mass)
        else:
            dg = {"capacity": 1.0, "o2_L": 0.0, "co2_L": 0.0, "unmet_J": 0.0, "tef_J": 0.0}
        cap = dg["capacity"]
        P_total = P_demand + P_act_w
        h["p_total_w"] = P_total
        h["P_tissue_w"] = P_tissue
        # ---- the bulk nodes ----
        Q = (q_v + work_v) * cap + act_v
        tef_w = dg["tef_J"] / dt
        if tef_w > 0.0:
            m = zi == TI.ZONE_INDEX["torso"]
            Q[m] += tef_w * C[m] / C[m].sum()
        w_core = np.array([zp[z].w_b_core for z in TI.ZONES])
        dens = (work_v * cap + act_v) / np.maximum(Vv, 1e-12)        # W/m3 of work heat
        vess_z = np.array([ctl.vessel[z] for z in TI.ZONES])
        w0v = w_core[zi]
        wb = w0v * np.where(sh, BN.shell_flow_factor(vess_z[zi]), 1.0) + (EN.hyperemia_w(w0v, dens) - w0v)
        circ = h["circ"]
        g_b = TI.perfusion_g(wb, Vv, TI.perfusion_factor(Tb, 0.0, circ))
        eps_z = np.array([zp[z].eps_cc for z in TI.ZONES])
        eps_b = eps_z[zi]
        # ---- the skin nodes (surface voxels) ----
        surf = np.nonzero(sh)[0]
        Qs = np.zeros(len(surf))
        # sweat, evaporation and condensation by zone, then shared out by area
        R_cl, f_cl = self._bh_clothing(h, env["v"])
        h["R_cl"] = R_cl.tolist()
        film = np.array(h["film_g"], dtype=np.float64)
        new_film = film.copy()
        lat_J = vap_g = diffuse_g = drip_g = 0.0
        sweat_made = 0.0
        sweat_zone = {z: ctl.sweat_g_s[z] * dt for z in TI.ZONES}
        e_z = []
        for z in range(BN.NZ):
            e = {"T_air": env["T_air"], "rh": env["rh"], "v": env["v"], "T_rad": env["T_rad"],
                 "R_cl": float(R_cl[z]), "f_cl": float(f_cl[z]), "T_mrt": None, "h_fixed": None}
            e_z.append(e)
            if g["A_true"][z] <= 0 or zarea[z] <= 0:
                continue
            Tz = sk_T[TI.ZONES[z]]
            hnat = float(BA.h_c_nat(Tz - env["T_air"])) * BN.HC_FLESH
            hc = max(hnat, float(BA.h_c_forced(env["v"]))) if R_cl[z] <= 0 else hnat
            ev = BN.evap_zone(Tz, e, g["A_true"][z], hc, sweat_zone[TI.ZONES[z]], film[z], dt)
            new_film[z] = ev["film_g"]
            drip_g += ev["drip_g"]
            vap_g += ev["evap_g"] + ev["diffuse_g"]
            diffuse_g += ev["diffuse_g"]
            sweat_made += sweat_zone[TI.ZONES[z]]
            lat_J += ev["latent_w"] * dt
            msk = np.nonzero(sh[surf] * (zi[surf] == z))[0]
            if len(msk):
                wts = Av[surf][msk]
                Qs[msk] -= ev["latent_w"] * wts / wts.sum()
        # forced convection beyond the face law (bare zones) and the radiant correction
        gain_J = 0.0
        self._bh_fself(h, g, ent)
        Tmrt4 = self._bh_mrt(p, h, g, ent, zsk, env)
        Ts_s = Ts[surf]
        zs = zi[surf]
        As = Av[surf] * a_z[zs]
        bare = R_cl[zs] <= 0.0
        hf = float(BA.h_c_forced(env["v"]))
        hn = BA.h_c_nat(Ts_s - env["T_air"]) * BN.HC_FLESH
        Q_forced = np.where(bare, As * np.maximum(hf - hn, 0.0) * (env["T_air"] - Ts_s), 0.0)
        Qs += Q_forced
        q_rad = np.zeros(len(surf))
        od = v["open_dir"][surf]
        af = np.sqrt(np.clip(C[surf] / (BN.RHOC_FLESH * self._bh_dx() ** 3), 1e-6, 1.0))
        face_a = self._bh_dx() ** 2 * af
        # per face: T^4 of what that face sees (6 normal groups per zone)
        t4 = Tmrt4[zs]                                       # (m, 6) in K^4
        gain4 = ((t4 - (T_INF + 273.15) ** 4) * od).sum(axis=1) * face_a * a_z[zs]
        h_r = BA.h_r(Ts_s, T_INF, BA.EMISS_SKIN * BN.RAD_VIEW, 1.0)
        # bare: the lattice radiates eps sigma (Ts^4 - Tinf^4); add eps sigma (Tmrt^4 - Tinf^4)
        q_rad = np.where(bare, BA.EMISS_SKIN * BN.RAD_VIEW * SIGMA * gain4, 0.0)
        # clothed: the series conductance carries the radiative share h_r/h of the linearised flux
        if (~bare).any():
            Tm_eff = (np.where(od.sum(axis=1) > 0, (t4 * od).sum(axis=1) / np.maximum(od.sum(axis=1), 1), 0.0)) ** 0.25 - 273.15
            hc_s = hn
            hh = hc_s + h_r
            ser = As * (h_r / np.maximum(hh, 1e-9)) / (R_cl[zs] + 1.0 / (f_cl[zs] * np.maximum(hh, 1e-9)))
            q_rad = np.where(bare, q_rad, ser * (Tm_eff - T_INF) * (od.sum(axis=1) > 0))
        Qs += q_rad
        gain_J += float(Q_forced.sum() + q_rad.sum()) * dt
        # dermal perfusion with the vessel reflex
        w_skin = np.array([zp[z].w_b_skin for z in TI.ZONES])
        vess = np.array([h["ctl"]["vessel"][z] for z in TI.ZONES])
        g_s = TI.perfusion_g(w_skin[zs], As * TI.SKIN_M,
                             TI.perfusion_factor(Ts_s, 0.0, circ) * vess[zs])
        eps_s = eps_z[zs]
        # ---- breath (a pool sink) ----
        V_E = BN.RMV_L_MIN * P_total / (EN.MET_W_PER_KG * mass)
        br = BA.breath_loss(V_E, env["T_air"], env["rh"], T_core=Tp) if h["alive"] else \
            {"sens_w": 0.0, "lat_w": 0.0, "water_g_s": 0.0}
        Q_o = br["sens_w"] + br["lat_w"]
        # ---- Pennes, bulk + skin nodes, the pool closed form ----
        T_all = np.concatenate([Tb, Ts_s])
        C_all = np.concatenate([np.where(sh, Cb, C), Cs[surf]])
        g_all = np.concatenate([g_b, g_s])
        Q_all = np.concatenate([Q, Qs])
        eps_all = np.concatenate([eps_b, eps_s])
        Tn, Tp1, B = TI.pennes_step(T_all, C_all, g_all, Q_all, dt, Tp, h["C_pool"], Q_o, eps_all)
        dEb = C_all[:n] * (Tn[:n] - Tb)
        dEs = C_all[n:] * (Tn[n:] - Ts_s)
        dE = dEb.copy()
        dE[surf] += dEs
        E1 = E + dE
        Tm1 = BN.AMBIENT + E1 / C
        s1 = np.zeros(n)
        s1[surf] = Tn[n:] - Tm1[surf]
        self._bh_write(idx, E1, s1)
        h["T_pool"] = float(Tp1)
        h["film_g"] = new_film.tolist()
        # ---- what the room is given, through the one sink ----
        sink = self._airsink()
        ins_g = br["water_g_s"] * dt
        st["water_ml"] -= sweat_made + diffuse_g + ins_g
        deliver(sink, vapour_g=vap_g, latent_J=lat_J, sensible_J=-gain_J,
                o2_l=0.0, co2_l=0.0, where=None)
        book = h["work"]                    # the gas of the acts billed since the last second
        act_o2, act_co2 = book.pop("o2_L", 0.0), book.pop("co2_L", 0.0)
        deliver(sink, vapour_g=ins_g, latent_J=br["lat_w"] * dt, sensible_J=br["sens_w"] * dt,
                o2_l=-(dg["o2_L"] + act_o2), co2_l=dg["co2_L"] + act_co2, where=None)
        h["drip_g"] = h.get("drip_g", 0.0) + drip_g
        # ---- readouts the rest of the sim reads ----
        a_sum = float(g["A_true"].sum())
        h["T_skin_mean"] = float((zsk * g["A_true"]).sum() / a_sum)
        h["set_core"] = set_core
        h["dehyd_pct"] = DG.deficit_pct(st, mass)
        h["hunger"] = DG.hunger(st)
        h["thirst"] = DG.thirst(st, mass)
        if h["clothes"]:                                   # wind and wetness change the garment's resistance
            self._bh_hooks(p, h, g)
        h["unmet_J"] = h.get("unmet_J", 0.0) + dg["unmet_J"]
        h["fit"] = {"cold": float(np.interp(Tp1, [c[0] for c in COLD_BAND], [c[1] for c in COLD_BAND])),
                    "hot": float(np.interp(Tp1, [c[0] for c in HOT_BAND], [c[1] for c in HOT_BAND]))}
        h["last"] = {"sweat_g": sweat_made, "latent_J": lat_J, "breath_sens_J": br["sens_w"] * dt,
                     "breath_lat_J": br["lat_w"] * dt, "gain_J": gain_J, "tef_J": dg["tef_J"],
                     "o2_L": dg["o2_L"], "co2_L": dg["co2_L"], "dt": dt, "P_total_w": P_total,
                     "capacity": cap, "Q_forced_J": float(Q_forced.sum() * dt),
                     "Q_rad_J": float(q_rad.sum() * dt)}
        return h["last"]

    def _bh_events(self, p, h, ctl, mass):
        for verb, edge in TR.events(h["prm"], ctl, mass, h["flags"]):
            h["events"].append((int(self.tick), verb, edge))
            if len(h["events"]) > 64:
                del h["events"][0]
            f = getattr(self, "bh_emit", None)
            if f is not None:
                f(p, verb, edge)

    # ---------------------------------------- mean radiant temperature -------
    def _bh_zgrid(self, h, ent):
        zg = ent["zgrid"]
        if zg is None:
            zg = np.full(self.shape, -1, dtype=np.int8)
            zg[tuple(h["cells"].T)] = h["zone"]
            ent["zgrid"] = zg
        return zg

    def _bh_rays(self, h, ent):
        """What each of the 26 rays of each zone hits first, found once per
        shape of the body (cached outside the state): kind 0 = leaves the
        lattice, 1 = another zone of this body (value = zone), 2 = some other
        solid (value = flat cell index). A ray skips its OWN zone."""
        # the walls may have changed (a door opened): look again every two sim minutes
        if ent["rayc"] is not None and (self.tick - ent.get("rayc_tick", 0)) * self._bh_tick_s() < RAY_REFRESH_S:
            return ent["rayc"]
        ent["zgrid"] = None
        zg = self._bh_zgrid(h, ent)
        shape = np.array(self.shape)
        cent = np.zeros((BN.NZ, 3))
        have = np.zeros(BN.NZ, dtype=bool)
        for z in range(BN.NZ):
            m = h["zone"] == z
            if m.any():
                cent[z] = h["cells"][m].mean(axis=0) + 0.5
                have[z] = True
        nr = len(_RAY_DIRS)
        pos = np.repeat(cent, nr, axis=0)
        dirs = np.tile(_RAY_DIRS, (BN.NZ, 1))
        zown = np.repeat(np.arange(BN.NZ), nr)
        kind = np.zeros(BN.NZ * nr, dtype=np.int8)
        val = np.zeros(BN.NZ * nr, dtype=np.int64)
        alive = np.repeat(have, nr)
        mat = self.mat
        steps = int(np.ceil(np.linalg.norm(shape) / _RAY_STEP)) + 2
        for _ in range(steps):
            if not alive.any():
                break
            pos[alive] += dirs[alive] * _RAY_STEP
            cell = np.floor(pos).astype(np.int64)
            out = ((cell < 0) | (cell >= shape)).any(axis=1) & alive
            alive &= ~out                                  # kind 0 stays: the world past the edge
            ins = np.nonzero(alive)[0]
            if len(ins) == 0:
                break
            c = cell[ins]
            solid = mat[c[:, 0], c[:, 1], c[:, 2]] != 0
            for k_, k in enumerate(ins):
                if not solid[k_]:
                    continue
                zc = int(zg[tuple(c[k_])])
                if zc == zown[k]:
                    continue
                if zc >= 0:
                    kind[k], val[k] = 1, zc
                else:
                    kind[k], val[k] = 2, int(np.ravel_multi_index(tuple(c[k_]), self.shape))
                alive[k] = False
        ent["rayc"] = (kind, val, dirs)
        ent["rayc_tick"] = int(self.tick)
        return ent["rayc"]

    def _bh_fself(self, h, g, ent):
        """The view factors the compact model takes from the lattice: for each
        zone, the share of what its open faces see that is the body itself
        (rays that hit another zone), weighted by open-face area. Stored in
        the geometry (and so in the compact model's inputs)."""
        if ent.get("fself_done"):
            return
        kind, val, dirs = self._bh_rays(h, ent)
        nr = len(_RAY_DIRS)
        hit = (kind == 1).reshape(BN.NZ, nr).astype(float)
        v = g["vox"]
        af = np.sqrt(np.clip(v["C_v"] / (BN.RHOC_FLESH * self._bh_dx() ** 3), 1e-6, 1.0))
        for z in range(BN.NZ):
            m = v["zone"] == z
            if not m.any():
                continue
            fa = (v["open_dir"][m] * af[m][:, None]).sum(axis=0)          # open face area per normal
            tot = fa.sum()
            if tot <= 0:
                continue
            f = 0.0
            for k, nrm in enumerate(FACES):
                sel = (_RAY_DIRS @ nrm) > 1e-9
                f += fa[k] * hit[z, sel].mean()
            g["f_self"][z] = f / tot
        ent["fself_done"] = True

    def _bh_mrt(self, p, h, g, ent, zsk, env):
        """T^4 (K^4) that each of the 6 outward-normal groups of each zone sees:
        26 rays from each zone's centroid, each ray skips its own zone, then
        hits the first solid (a body zone at its skin temperature, any other
        solid at its own) or leaves the lattice and sees the world. Where the
        rays go is cached against the body's shape; only the temperatures at
        their ends are read each second."""
        kind, val, dirs = self._bh_rays(h, ent)
        nr = len(_RAY_DIRS)
        T = np.empty(len(kind))
        e = kind == 0
        if e.any():
            T[e] = [self._bh_T_rad(dirs[k], env) for k in np.nonzero(e)[0]]
        z1 = kind == 1
        T[z1] = zsk[val[z1]]
        c2 = kind == 2
        if c2.any():
            Cap = np.asarray(self.heat_capacity(), dtype=np.float64).reshape(-1)
            Ef = np.asarray(self.E, dtype=np.float64).reshape(-1)
            T[c2] = BN.AMBIENT + Ef[val[c2]] / Cap[val[c2]]
        t4 = ((T + 273.15) ** 4).reshape(BN.NZ, nr)
        out6 = np.zeros((BN.NZ, 6))
        for k, nrm in enumerate(FACES):
            sel = (_RAY_DIRS @ nrm) > 1e-9
            out6[:, k] = t4[:, sel].mean(axis=1)
        return out6

    def _bh_T_rad(self, d, env):
        f = getattr(self, "bh_T_rad", None)
        return f(d) if f is not None else env["T_rad"]

    # ------------------------------------------------------- the hand-over --
    def bh_to_nodes(self, p):
        h = p["heat"]
        g = self._bh_geom(p, h)
        idx = tuple(h["cells"].T)
        return BN.to_nodes(g, h, np.asarray(self.E[idx], dtype=np.float64),
                           np.asarray(self._skin_field()[idx], dtype=np.float64))

    def bh_from_nodes(self, p, nodes):
        h = p["heat"]
        g = self._bh_geom(p, h)
        idx = tuple(h["cells"].T)
        E_v, s_v = BN.from_nodes(g, h, nodes)
        self._bh_write(idx, E_v, s_v)
        h["T_pool"] = float(BN.temps(nodes)[BN.POOL])
        h["film_g"] = nodes["film_g"].tolist()

    def bh_skip(self, p, dt, env=None, met=None):
        """The night skip: voxels -> nodes -> advance(dt) -> voxels, with the
        same sink as the ticks."""
        h = p["heat"]
        g = self._bh_geom(p, h)
        self._bh_fself(h, g, self._bh_cache()[h["pid"]])
        nodes = self.bh_to_nodes(p)
        e = self._bh_env(p) if env is None else env
        R_cl, f_cl = self._bh_clothing(h, e.get("v", 0.0))
        e = dict(e)
        e["zones"] = {TI.ZONES[z]: {"R_cl": float(R_cl[z]), "f_cl": float(f_cl[z])} for z in range(BN.NZ)}
        out = BN.advance(h, nodes, g, e, dt, met=h["met"] if met is None else met)
        self.bh_from_nodes(p, nodes)
        h["drip_g"] = h.get("drip_g", 0.0) + out["drip_g"]      # the same book the ticks write
        deliver(self._airsink(), vapour_g=out["vapour_g"], latent_J=out["latent_J"],
                sensible_J=out["sensible_J"], o2_l=-out["o2_l"], co2_l=out["co2_l"])
        self.shed += 0.0
        h["tick_last"] = int(self.tick)
        return out

    # ------------------------------------------------ acts, food, readouts ---
    def note_work(self, p, tag, J, kind, zone_weights=None):
        """Bill an act once (energetics.spend_J): the actor's stores pay, work_out is credited."""
        h = p["heat"]
        if zone_weights is None:
            g = self._bh_geom(p, h)
            zone_weights = {TI.ZONES[z]: float(g["C_shell"][z] + g["C_core"][z]) for z in range(BN.NZ)
                            if g["n_vox"][z] > 0}
        now = self.tick * self._bh_tick_s()
        self.work_out = float(self.work_out)
        return EN.spend_J(h, self, tag, J, kind, now, zone_weights, mass_kg=h["sheet"]["mass_kg"])

    def release_work(self, J):
        """Work an act put into the world leaves work_out (energetics.release_J);
        the caller deposits the same joules in E."""
        self.work_out = float(self.work_out)
        return EN.release_J(self, J)

    def p_total(self, p):
        return p["heat"].get("p_total_w", 0.0)

    def eat(self, p, food_rec):
        """Put food in the stomach. Its chemical energy enters the books as the
        named injection "food" (the food thing leaves the world through its own
        ledger); returns the mass eaten, grams."""
        J = DG.food_ge_kj(food_rec) * 1000.0
        inj = self.bh_injected()
        inj["food"] = inj.get("food", 0.0) + J
        self.shed -= J
        return DG.ingest(p["heat"]["stores"], food_rec)

    def drink(self, p, ml, name="water"):
        return self.eat(p, DG.food(name, ml))

    def bh_void(self, p):
        """Pass water and faeces: the matter leaves the body (the pour topic
        places it) carrying its chemical energy, which the books show as the
        named injection "excreta" going negative. Returns (urine_ml, urea_g,
        faeces_g)."""
        st = p["heat"]["stores"]
        ml, urea, ju = DG.void_bladder(st)
        gf, jf = DG.void_colon(st)
        inj = self.bh_injected()
        inj["excreta"] = inj.get("excreta", 0.0) - (ju + jf)
        self.shed += ju + jf
        return ml, urea, gf

    def body_temps(self, p):
        """Readouts, C: core (pool), hypothalamus, tympanic (head core), mean
        skin, and every zone's skin and core."""
        h = p["heat"]
        g = self._bh_geom(p, h)
        idx = tuple(h["cells"].T)
        v = g["vox"]
        E = np.asarray(self.E[idx], dtype=np.float64)
        s = np.where(v["shell"], np.asarray(self._skin_field()[idx], dtype=np.float64), 0.0)
        Tm = BN.AMBIENT + E / v["C_v"]
        Ts = np.where(v["shell"], Tm + s, Tm)
        Tb = np.where(v["shell"], Tm - s * v["Cs_v"] / np.where(v["shell"], v["Cb_v"], 1.0), Tm)
        zi = v["zone"]
        skin, core = {}, {}
        for z, zn in enumerate(TI.ZONES):
            m = (zi == z) & v["shell"]
            wt = v["A_v"][m]
            skin[zn] = float((Ts[m] * wt).sum() / wt.sum()) if m.any() else float("nan")
            mc = zi == z
            core[zn] = float((Tb[mc] * v["C_v"][mc]).sum() / v["C_v"][mc].sum()) if mc.any() else float("nan")
        a = g["A_true"]
        ms = float(sum(skin[zn] * a[z] for z, zn in enumerate(TI.ZONES) if not math.isnan(skin[zn])) / a.sum())
        return {"core": h["T_pool"], "hypothalamus": h["T_pool"] + TR.DT_BRAIN, "tympanic": core["head"],
                "skin_mean": ms, "skin": skin, "zone_core": core}

    def body_heat_death(self, p):
        """Death: the pool's energy goes into the torso and head voxels (kept);
        circ = 0 so nothing is perfused, no flag says so."""
        h = p["heat"]
        g = self._bh_geom(p, h)
        idx = tuple(h["cells"].T)
        J = h["C_pool"] * (h["T_pool"] - BN.AMBIENT)
        v = g["vox"]
        E = np.asarray(self.E[idx], dtype=np.float64)
        m = np.isin(v["zone"], [TI.ZONE_INDEX["torso"], TI.ZONE_INDEX["head"]])
        E = E.copy()
        E[m] += J * v["C_v"][m] / v["C_v"][m].sum()
        s = np.where(v["shell"], np.asarray(self._skin_field()[idx], dtype=np.float64), 0.0)
        self._bh_write(idx, E, s)
        h["T_pool"] = BN.AMBIENT
        h["circ"] = 0.0
        h["alive"] = False
        p["alive"] = False

    def bh_terms(self):
        """Every body term of the energy and mass books, in one dict."""
        pool = stores = pend = 0.0
        m_stores = unplaced = 0.0
        for p in self.persons:
            h = p.get("heat")
            if h is None:
                continue
            pool += h["C_pool"] * (h["T_pool"] - BN.AMBIENT)
            stores += DG.stores_J(h["stores"])
            # sweat that ran off the skin and urine that overflowed the bladder are
            # still matter of this body's books until the pour topic places them
            lost = h.get("drip_g", 0.0) + h["stores"]["leak_ml"]
            unplaced += lost
            m_stores += DG.stores_g(h["stores"]) + float(np.sum(h["film_g"])) + lost
            pend += float(sum(h["work"]["pending_zone_J"].values()))
        return {"pool_J": pool, "stores_J": stores, "pending_J": pend, "work_out": float(self.work_out),
                "injected": dict(self.bh_injected()), "stores_g": m_stores,
                "unplaced_g": unplaced}
