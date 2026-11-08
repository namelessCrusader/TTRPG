"""Proof scenes and the SLAB FIXTURE for body heat (wave 13, T7 section 8).

CRITIC 4 (plan): the real lattice hooks BH1-BH3 (heat.py) are phase B. In
phase A the voxel-side body tests run on a small lattice that implements the
SAME hook arguments itself, never on the unmodified World face law. That
lattice is `SlabWorld` below:

  * a dense box of voxels; the body is a set of flesh voxels, each with the
    lattice's own two nodes (a thin SKIN node, Cs, over the open faces, and
    the BULK, Cb), conduction between neighbouring voxels (k dx per face), and
    a face law that reads hook tables keyed by body zone:
        BH1  skin-to-bulk conductance per lattice area (a_z / R_sc),
        BH2  natural-convection law (2.38 dT^0.25), emissivity, area factor a_z,
        BH3  series clothing resistance on the combined h.
    With no tables it is the OLD lattice: H_CONV = 10 W/m2K, emissivity 0.95,
    a_z = 1, the sqrt(a t) path to the core. Stub tables with the old values
    give the same bits (a test says so).
  * the air is a reservoir at `T_air`; the lattice radiates to a fixed 20 C
    (the old world sink); heat that leaves goes to `shed`, or to `face_sink`
    when a test hangs a sealed-room double there.
  * E and the skin field take ONE dtype, `E_dtype` (default float64), and
    every test is parametrised by it.

The scenes (S1 phase-A form, S8, hyperemia) use only the laws; the T8-dependent
full forms (S1 nude 28 C, S2, S3, S5, S7) are listed in `PHASE_B_SCENES` for
C6 and are NOT phase A acceptance.
"""
import copy

import numpy as np

from . import bodyair as BA
from . import bodynodes as BN
from . import tissue as TI
from . import bodyheat as BH

FLESH = 4
SUB_FIT = 120.0                     # sub-step of the human-case runs, s
# A body lying on its back on a floor (what Henssge's nomogram is for, and so what the 'corpse'
# case kind means): the share of each zone's skin that still looks at the air. The back is
# against an insulating floor (u: back of the head 25 percent, of the trunk 45, of the legs
# 30, of an arm at the side 15, of a hand or foot 15-20).
LYING_EXPOSED = {"head": 0.75, "torso": 0.55, "upper arm": 0.85, "forearm": 0.85, "hand": 0.85,
                 "thigh": 0.70, "shin": 0.70, "foot": 0.80}
# The chamber experiments the rest cases stand for (Fanger, Stolwijk, Hardy and DuBois) used SEATED
# or reclining people: the back and the underside of the thighs touch the chair. The share of each
# zone's skin that still looks at the air (u: about 8 percent of the whole body is covered).
SEATED_EXPOSED = {"torso": 0.85, "thigh": 0.85}
WATER_H = 25.0 * 8.3                # W/m2K, immersion film (spec F6: 25 times air)
SIGMA = BA.SIGMA
T_INF = 20.0
H_CONV = 10.0                    # the old lattice constant, W/m2K
EMISS_OLD = 0.95
K_FLESH = BN.K_FLESH
SKIN_PATH = np.sqrt(K_FLESH / BN.RHOC_FLESH * 1060.0)       # materials.SKIN_T[1] path, m


def _t4(T):
    t = T + 273.15
    t2 = t * t
    return t2 * t2


class SlabWorld(BH.BodyHeatMixin):
    """The fixture lattice. Implements the host contract of BodyHeatMixin."""
    AMBIENT = 20.0

    def __init__(self, shape=(14, 10, 36), dx=0.05, T_air=20.0, rh=0.5, v=0.0,
                 E_dtype=np.float64, tick_s=1.0, bh_period=1):
        self.shape = tuple(shape)
        self.dx = float(dx)
        self.scale = dx / 0.1
        self.E_dtype = np.dtype(E_dtype)
        self.mat = np.zeros(self.shape, np.uint8)
        self.fill = np.zeros(self.shape)
        self.E = np.zeros(self.shape, self.E_dtype)
        self.skin = np.zeros(self.shape, self.E_dtype)
        self.tick = 0
        self.tick_s = float(tick_s)
        self.bh_period = int(bh_period)
        self.shed = 0.0
        self.gone = {}
        self.persons = []
        self.T_air, self.rh, self.v = float(T_air), float(rh), float(v)
        self.T_wall = {}                  # {(axis, sign): C} for the world past the box edge
        self.face_sink = None             # a sealed-room double takes the face flux when set
        self._body = None
        self._cap = None
        self.accept_hooks = True

    # ---- the host contract ------------------------------------------------
    def heat_capacity(self):
        if self._cap is None:
            self._cap = np.where(self.mat == FLESH, self.fill * BN.RHOC_FLESH * self.dx ** 3, 1.0)
        return self._cap

    def _skin_field(self):
        return self.skin

    def bh_env(self, p):
        return {"T_air": self.T_air, "rh": self.rh, "v": self.v, "T_rad": self.T_air}

    def bh_T_rad(self, d):
        ax = int(np.argmax(np.abs(d)))
        return self.T_wall.get((ax, int(np.sign(d[ax]))), self.T_air)

    def bh_set_hooks(self, pid, cells, zone_idx, tables):
        if not self.accept_hooks:
            return
        b = self._body
        z = b["zone"]
        b["az"] = tables["az"][z]
        b["emiss"] = tables["emiss"][z]
        b["conv_nat"] = tables["conv_nat"][z]
        b["R_cl"] = tables["R_cl"][z]
        b["f_cl"] = tables["f_cl"][z]
        b["gsb"] = tables["gsb"][z]
        b["hooked"] = True

    # ---- building ---------------------------------------------------------
    def add_body(self, cells, fill, zone_idx):
        cells = np.asarray(cells, dtype=np.int64)
        for c, f in zip(cells, fill):
            self.mat[tuple(c)] = FLESH
            self.fill[tuple(c)] = f
        self._cap = None
        n_open, pi, pj, open_dir = BN.cell_layout(cells, zone_idx)
        C = np.asarray(fill, dtype=np.float64) * BN.RHOC_FLESH * self.dx ** 3
        af = np.sqrt(np.clip(np.asarray(fill), 1e-6, 1.0))
        A = n_open * self.dx ** 2 * af
        sh = n_open > 0
        Cs = np.where(sh, np.minimum(BN.RHOC_FLESH * BN.SKIN_DEPTH * A, 0.5 * C), 0.0)
        self._body = {"idx": tuple(cells.T), "C": C, "Cs": np.where(sh, Cs, 1.0), "Cs0": Cs,
                      "Cb": C - Cs, "sh": sh, "A": A, "pi": pi, "pj": pj, "n": len(cells),
                      "zone": np.asarray(zone_idx, dtype=np.int64),
                      # the OLD lattice, until tables arrive
                      "az": np.ones(len(cells)), "emiss": np.full(len(cells), EMISS_OLD),
                      "conv_nat": np.zeros(len(cells)), "R_cl": np.zeros(len(cells)),
                      "f_cl": np.ones(len(cells)), "gsb": np.full(len(cells), K_FLESH / SKIN_PATH),
                      "hooked": False}

    # ---- the face law (plain numpy; the definition) -----------------------
    @staticmethod
    def face_loss(Ts, A, az, emiss, conv_nat, R_cl, f_cl, T_gas):
        """Watts leaving a skin node through its open faces. conv_nat == 0
        means the old constant H_CONV; else conv_nat x |dT|^0.25 (ISO). The
        order of operations is the same on both paths so stub tables give the
        same bits."""
        dT = Ts - T_gas
        hn = conv_nat * np.sqrt(np.sqrt(np.abs(dT)))
        h = np.where(conv_nat == 0.0, H_CONV, hn)
        Ae = A * az
        conv = h * Ae * dT
        rad = emiss * SIGMA * Ae * (_t4(Ts) - _t4(T_INF))
        bare = conv + rad
        # BH3: the garment sits in series on the COMBINED h; its outer surface radiates
        hr = 4.0 * emiss * SIGMA * (0.5 * (Ts + T_INF) + 273.15) ** 3
        hh = h + hr
        top = (h * T_gas + hr * T_INF) / hh
        ser = Ae * (Ts - top) / (R_cl + 1.0 / (f_cl * hh))
        return np.where(R_cl > 0.0, ser, bare)

    def step(self):
        if self._body is not None:
            self._tick_body(self._body)
        self._law_body_heat()
        self.tick += 1

    def _tick_body(self, b):
        idx, dt = b["idx"], self.tick_s
        E = np.asarray(self.E[idx], dtype=np.float64)
        sh, C, Cs, Cb = b["sh"], b["C"], b["Cs"], b["Cb"]
        s = np.where(sh, np.asarray(self.skin[idx], dtype=np.float64), 0.0)
        T = BN.AMBIENT + E / C
        Ts = np.where(sh, T + s, T)
        Tb = np.where(sh, T - s * Cs / np.where(sh, Cb, 1.0), T)
        n = b["n"]
        flux = K_FLESH * self.dx * (Tb[b["pi"]] - Tb[b["pj"]])
        cond = np.bincount(b["pj"], flux, n) - np.bincount(b["pi"], flux, n)
        loss = np.where(sh, self.face_loss(Ts, b["A"], b["az"], b["emiss"], b["conv_nat"],
                                           b["R_cl"], b["f_cl"], self.T_air), 0.0)
        gsb = b["gsb"] * b["A"]
        Ts1 = Ts + dt * (gsb * (Tb - Ts) - loss) / Cs
        E1 = E + dt * (cond - loss)
        T1 = BN.AMBIENT + E1 / C
        s1 = np.where(sh, Ts1 - T1, 0.0)
        self._store(idx, E1, s1)
        out = float(loss.sum() * dt)
        if self.face_sink is not None:
            self.face_sink.add(heat_J=out)
        else:
            self.shed += out

    def _store(self, idx, E1, s1):
        self.E[idx] = E1.astype(self.E_dtype)
        self.skin[idx] = s1.astype(self.E_dtype)

    # ---- a world as a value (the same bargain as ledgers.snapshot) ---------
    _NOT_STATE = ("_bh_runtime", "face_sink", "airsink")

    def snapshot(self):
        """Everything this world IS; the caches rebuild, the sinks stay."""
        state = {k: v for k, v in self.__dict__.items() if k not in self._NOT_STATE}
        return copy.deepcopy(state)

    def restore(self, snap):
        state = copy.deepcopy(snap)
        for k in [k for k in self.__dict__ if k not in self._NOT_STATE]:
            del self.__dict__[k]
        self.__dict__.update(state)
        self.__dict__.pop("_bh_runtime", None)

    # ---- the books --------------------------------------------------------
    def total_energy(self):
        """Lattice energy plus the named body terms (3.10)."""
        t = self.bh_terms()
        return (float(np.asarray(self.E, dtype=np.float64).sum()) + self.shed + t["pool_J"]
                + t["stores_J"] + t["pending_J"] + t["work_out"])


# ======================================================== the stock figure ===
def stock_figure(px=5, py=5, z0=1):
    """scenes._person's body, rebuilt without a World: (cells, fill, segs).
    396 voxels, 43.9 kg at 5 cm."""
    z = z0 - 1
    segs, fills = {}, {}

    def part(name, x0, x1, y0, y1, z1, z2, frac):
        c = np.array([(x, y, zz) for x in range(x0, x1) for y in range(y0, y1)
                      for zz in range(z1, z2)], dtype=np.int64)
        segs[name] = c
        fills[name] = frac
    for side, lx0 in (("left", px), ("right", px + 2)):
        part(f"{side} shin", lx0, lx0 + 1, py - 1, py + 2, z + 1, z + 8, 0.9)
        part(f"{side} thigh", lx0, lx0 + 1, py - 1, py + 2, z + 8, z + 15, 0.9)
    part("torso", px, px + 3, py - 2, py + 3, z + 15, z + 27, 0.9)
    for side, ax0 in (("left", px - 2), ("right", px + 3)):
        part(f"{side} upper arm", ax0, ax0 + 2, py - 1, py + 2, z + 23, z + 27, 0.85)
        part(f"{side} forearm", ax0, ax0 + 2, py - 1, py + 2, z + 18, z + 23, 0.85)
    part("head", px, px + 2, py - 1, py + 2, z + 27, z + 31, 0.9)
    cells, zone = BH.zones_from_segs(segs)
    # fill per voxel, in the same order zones_from_segs walked the bones
    fill = []
    for bone in sorted(segs):
        fill += [fills[bone]] * len(segs[bone])
    return cells, np.array(fill), zone, segs


def stock_sheet():
    return TI.sheet("male", mass_kg=43.9, height_m=1.5, fat_frac=0.17)


def person_world(E_dtype=np.float64, T_air=20.0, rh=0.5, v=0.0, clothes=None, warm=True,
                 pid=0, sheet=None, tick_s=1.0, bh_period=1, hooks=True):
    """A fixture lattice with the stock figure standing in it and a heat record.
    `hooks=False` leaves the OLD face law (the golden-same case)."""
    w = SlabWorld(E_dtype=E_dtype, T_air=T_air, rh=rh, v=v, tick_s=tick_s, bh_period=bh_period)
    w.accept_hooks = bool(hooks)
    cells, fill, zone, segs = stock_figure()
    w.add_body(cells, fill, zone)
    p = {"pid": pid, "name": "a", "alive": True}
    w.persons.append(p)
    w.body_heat_on(p, cells, zone, sheet or stock_sheet(), warm=warm, clothes=clothes)
    return w, p


def run(w, seconds):
    n = int(round(seconds / w.tick_s))
    for _ in range(n):
        w.step()


# ============================================================ scenes ========
PHASE_B_SCENES = {
    "S1_full": "nude person, 28 C still air, 2 h: needs T8 (no void radiation from the lattice) and the real hooks",
    "S2": "10 C for 1 h bare vs 1.5 clo coat: T8",
    "S3": "soaked coat at 5 C: T8",
    "S5": "four sleepers in a closed room, 8 h: T8 and the sealed-room region sink (C10)",
    "S7": "cold-water swim: T8 water body",
}


def scene_clothed_20C(minutes=20, E_dtype=np.float64, clo=1.0):
    """S1 phase-A form: a CLOTHED (1 clo) person in 20 C still air."""
    from . import clothing as CL
    cov = {z: 1.0 for z in TI.ZONES if not z.endswith(("hand", "foot")) and z != "head"}
    area = {z: 1.0 for z in cov}
    g = CL.garment("suit", "wool", CL.thickness_for_clo(clo), cov, area, layer=0)
    w, p = person_world(E_dtype=E_dtype, T_air=20.0, clothes=[g])
    run(w, minutes * 60)
    return w, p


def scene_work(met_steps=((600, 3.0), (600, 8.0)), E_dtype=np.float64, T_air=20.0):
    """Hyperemia: so many seconds at each MET. Returns the world, the person
    and a trace of (t, core, hottest muscle voxel, mean skin)."""
    w, p = person_world(E_dtype=E_dtype, T_air=T_air)
    h = p["heat"]
    tr = []
    t = 0
    idx = tuple(h["cells"].T)
    muscle = np.isin(h["zone"], [TI.ZONE_INDEX["left thigh"], TI.ZONE_INDEX["right thigh"],
                                 TI.ZONE_INDEX["left shin"], TI.ZONE_INDEX["right shin"]])
    for secs, met in met_steps:
        h["met"] = met
        for _ in range(int(secs / w.tick_s)):
            w.step()
            t += w.tick_s
            if int(t) % 20 == 0:
                Tv = BN.AMBIENT + np.asarray(w.E[idx], dtype=np.float64) / w._body["C"]
                tr.append((t, h["T_pool"], float(Tv[muscle].max()), float(Tv[muscle].mean())))
    return w, p, tr


# ================================================== HUMAN-DATA CASES =========
# The compact model run against published human data (tests/data/human_cases.json).
# The tolerances are written into that file BEFORE any fit and carry a hash;
# only the "fit" block may change afterwards. Fit on the "fit" cases (steady),
# test on the "held_out" ones (transients, clothing, humid, hot, immersion,
# dehydration, corpse).
import hashlib
import json


def mid_params(heat):
    """The population-average controller constants (the middle of each draw
    range, then the fitted scale), so a fixture is about 'a person', not about
    one lucky draw."""
    from . import thermoreg as TR
    prm = {k: 0.5 * (lo + hi) for k, (lo, hi) in TR.PARAMS.items()}
    for k, f in BN.PRM_SCALE.items():
        prm[k] = prm[k] * f
    heat["prm"] = prm
    return prm


def _std_person(mass=73.0, height=1.75, fat=None, sex="male"):
    sh = TI.sheet(sex, mass_kg=mass, height_m=height, fat_frac=fat)
    g = BN.geom_from_sheet(sh)
    h = BN.new_heat_record(sh, 1, "std")
    mid_params(h)
    return sh, g, h


def _clo_ensemble(clo, v=0.1):
    from . import clothing as CL
    cov = {z: 1.0 for z in TI.ZONES if not z.endswith(("hand", "foot")) and z != "head"}
    gar = CL.garment("suit", "wool", CL.thickness_for_clo(clo), cov, {z: 1.0 for z in cov})
    zones = {}
    for z in TI.ZONES:
        R = CL.ensemble_resistance([gar], z, v)
        zones[z] = {"R_cl": R, "f_cl": float(BA.f_cl_iso(R))}
    return zones


def _compact_run(inp, T0_env=None, minutes=None, state=None):
    """Run the compact model. Returns (state, trace). `state` = (sh, g, h, nodes)."""
    if state is None:
        sh, g, h = _std_person(inp.get("mass", 73.0), inp.get("height", 1.75), inp.get("fat"))
        nodes = BN.nodes_from_T(g, h, BN.comfort_T(g, h))
        if inp.get("dehydrated_pct"):
            h["stores"]["water_ml"] -= inp["dehydrated_pct"] / 100.0 * sh.mass_kg * 1000.0
        state = (sh, g, h, nodes)
    sh, g, h, nodes = state
    return state


def run_human_case(case, hour=12.0):
    """Execute one case with the module's current constants. Returns {name: value}."""
    kind = case["kind"]
    inp = case["inputs"]
    sub = SUB_FIT

    def env_of(T, rh, v, clo=0.0, water=False):
        e = {"T_air": T, "rh": rh, "v": v, "T_rad": T}
        zones = _clo_ensemble(clo, v) if clo > 0 else {}
        if water:
            for z in TI.ZONES:
                if z != "head":
                    zones[z] = {"h_fixed": WATER_H}          # water: 25 x the 8.3 of still air (spec F6)
        e["zones"] = zones
        if not water:                                      # seated: part of the skin is on the chair
            for z in TI.ZONES:
                k = z.split(" ", 1)[1] if z[:4] in ("left", "righ") else z
                if k in SEATED_EXPOSED:
                    zones.setdefault(z, {})["exposed"] = SEATED_EXPOSED[k]
        return e

    def settle(state, e, minutes, met=1.0):
        sh, g, h, nodes = state
        tot = {"metabolic_J": 0.0, "latent_J": 0.0, "sweat_g": 0.0, "dt": 0.0, "drip_g": 0.0,
               "vapour_g": 0.0}
        for _ in range(int(minutes / 10.0)):
            out = BN.advance(h, nodes, g, e, 600.0, met=met, hour=hour, sub=sub)
            for k in ("metabolic_J", "latent_J", "sweat_g", "drip_g", "vapour_g"):
                tot[k] += out[k]
            tot["dt"] += 600.0
        return tot

    def new_state():
        return _compact_run(inp)

    def snap(state):
        sh, g, h, nodes = state
        T = BN.temps(nodes)
        return {"core": float(T[BN.POOL]), "skin_mean": BN.mean_skin(nodes, g),
                "shiver_W": float(h["ctl"]["shiver"]),
                "sweat_Lh": float(sum(h["ctl"]["sweat"].values()) * 3.6)}

    if kind == "steady":
        st = new_state()
        e = env_of(inp["T_air"], inp["rh"], inp["v"], inp.get("clo", 0.0))
        settle(st, e, inp["settle_min"] - 60)
        tot = settle(st, e, 60)
        r = snap(st)
        r["heat_prod_W"] = tot["metabolic_J"] / tot["dt"]
        return r
    if kind == "step":
        st = new_state()
        e0 = env_of(inp["T0"], inp["rh"], inp["v"])
        settle(st, e0, 120)
        a = snap(st)
        e1 = env_of(inp["T1"], inp["rh"], inp["v"])
        settle(st, e1, inp["minutes"])
        b = snap(st)
        r = dict(b)
        r["core_rise"] = b["core"] - a["core"]
        r["core_drop"] = a["core"] - b["core"]
        r["skin_drop"] = a["skin_mean"] - b["skin_mean"]
        return r
    if kind == "pair_humid":
        res = []
        for rh in (inp["rh_lo"], inp["rh_hi"]):
            st = new_state()
            settle(st, env_of(inp["T0"], 0.4, 0.1), 120)
            tot = settle(st, env_of(inp["T1"], rh, inp["v"]), inp["minutes"])
            res.append((snap(st), tot))
        return {"delta_core": res[1][0]["core"] - res[0][0]["core"],
                "latent_ratio": res[1][1]["latent_J"] / max(res[0][1]["latent_J"], 1e-9)}
    if kind == "immersion":
        drops = []
        for fat in (inp["fat_lean"], inp["fat_fat"]):
            inp2 = dict(inp, fat=fat)
            st = _compact_run(inp2)
            sh, g, h, nodes = st
            settle(st, env_of(inp["T_air"], 0.5, 0.1), 120)
            a = snap(st)
            settle(st, env_of(inp["T_water"], 1.0, 0.0, water=True), inp["minutes"])
            drops.append(a["core"] - snap(st)["core"])
        return {"drop_lean": drops[0], "drop_fat": drops[1], "ratio": drops[1] / max(drops[0], 1e-9)}
    if kind == "dehydration":
        cores = []
        for d in (0.0, inp["dehydrated_pct"]):
            st = _compact_run(dict(inp, dehydrated_pct=d))
            settle(st, env_of(inp["T_air"], inp["rh"], inp["v"]), 120, met=1.0)
            settle(st, env_of(inp["T_air"], inp["rh"], inp["v"]), inp["minutes"], met=inp["met"])
            cores.append(snap(st)["core"])
        return {"delta_core": cores[1] - cores[0]}
    if kind == "corpse":
        st = new_state()
        sh, g, h, nodes = st
        e = env_of(inp["T_air"], 0.5, 0.0)
        settle(st, e, 120)
        BN.kill_nodes(h, nodes, g)
        e = dict(e, zones={z: {"exposed": LYING_EXPOSED[z.split(" ", 1)[1] if z[:4] in ("left", "righ") else z]}
                           for z in TI.ZONES})
        out = {}
        t = 0
        prev = None
        for hrs in inp["hours"]:
            while t < hrs * 60:
                BN.advance(h, nodes, g, e, 600.0, met=1.0, hour=hour, sub=120.0)
                t += 10
            T = BN.temps(nodes)
            out["T_%dh" % hrs] = float(T[BN.nidx(TI.ZONE_INDEX["torso"], 2)])   # the deep trunk: where a rectal probe sits
        return out
    raise ValueError(kind)


def case_tolerance_hash(cases):
    """Hash of everything a fit must NOT move: ids, roles, inputs, targets."""
    blob = json.dumps([{k: c[k] for k in ("id", "role", "kind", "inputs", "targets")} for c in cases],
                      sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def score_case(case, measured):
    """(ok, rows): every target [lo, hi] checked; rows = (name, value, lo, hi, ok)."""
    rows = []
    for name, (lo, hi) in sorted(case["targets"].items()):
        v = measured[name]
        rows.append((name, v, lo, hi, lo <= v <= hi))
    return all(r[4] for r in rows), rows


def cost(cases, roles=("fit",), margin=0.0):
    """Sum over the fit cases of how far each value is outside its band,
    in units of the band's width (0 when all are inside). `margin` shrinks
    each band by that share of its width at both ends, so a fit lands inside
    the band and not on its edge."""
    c = 0.0
    for case in cases:
        if case["role"] not in roles:
            continue
        m = run_human_case(case)
        for name, (lo, hi) in case["targets"].items():
            w = max(hi - lo, 1e-9)
            v = m[name]
            lo, hi = lo + margin * w, hi - margin * w
            c += (max(lo - v, 0.0) / w) ** 2 + (max(v - hi, 0.0) / w) ** 2
    return c


# ============================================ wave 13b: scenes on the real lattice
# These build a BodyWorld (bodyworld.py): the real World with body heat on. The
# slab scenes above run on the fixture; these run on the lattice, with its face
# law, its air and its walls. New scenes first (decision D7): none of the old
# scenes is changed by them.
def lattice_nude_20C(minutes=0.0):
    """A nude person alone in a 2 m room (40 x 40 x 40 voxels of 5 cm, stone
    floor, the world edge for walls, air at the lattice's 20 C). Returns
    (world, person). `minutes` of ticks are run before it returns (0: only
    built; the heat record appears in the first body second)."""
    from .bodyworld import BodyWorld
    from .materials import STONE
    from .scenes import _person, _Wants
    w = BodyWorld(40, 40, 40, voxel_cm=5)
    w.fill(0, 40, 0, 40, 0, 1, STONE)
    p = _person(w, 15, 15)
    w.policy = _Wants()
    for _ in range(int(minutes * 60 * 40)):
        w.step()
        p["events"].clear()
    return w, p


def lattice_hot_wall(minutes=3.0, wall_C=45.0):
    """S9 in a small form: a person two voxels from a wall held at `wall_C`
    (a dev thermostat on the wall voxels). The skin that faces the wall is
    warmer than the back (the mean radiant temperature per zone, section 3.9).
    Returns (world, person)."""
    from .bodyworld import BodyWorld
    from .materials import STONE
    from .scenes import _person, _Wants
    w = BodyWorld(40, 40, 40, voxel_cm=5)
    w.fill(0, 40, 0, 40, 0, 1, STONE)
    w.fill(25, 27, 0, 40, 1, 40, STONE)
    w.thermostats = [(x, y, z, float(wall_C)) for x in (25, 26) for y in range(40) for z in range(1, 40)]
    p = _person(w, 16, 16)
    w.policy = _Wants()
    for _ in range(int(minutes * 60 * 40)):
        w.step()
        p["events"].clear()
    return w, p
