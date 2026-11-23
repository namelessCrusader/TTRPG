"""SKIN FACES, wired (wave 15, S2 C; owner W15-Q3: burns PERSIST).

`skinburn` is the law: a 1-D skin profile of 8 nodes under one exposed face,
Moritz and Henriques damage, a degree and a depth read from it, a closed-form
shock. This file connects it to a body in the world. It is called by
`physiology._law_life` where the old body-wide scalar
`p["burn"] += gain * mean(clip(T - 55 C, 0, 400))` was.

THE OLD SCALAR averaged over every flesh cell of the body: a hand held in a
fire added one cell in 400, a person in a hot room a whole-body average. No
area, no depth, no place. Now a face is a flesh cell's side that meets gas, a
liquid or another solid, and it burns at its own exposure:

  gas      the neighbour cell's air: convection h_c = 25 W/m2K (EN 1991-1-2's
           coefficient for a surface exposed to fire (k); the heat law's 10 is
           still air) from its temperature. NO RADIANT FLUX: the heat law keeps none
           per face (heat.py `_face_flux` is an empty seam), so a flash burn at
           a distance is not seen (switch row skinworld.no_radiant_flux);
  liquid   a neighbour cell that holds fluid: the fluid's own h (materials.FLUID:
           water 1500, oil 500 W/m2K) times LIQUID_SKIN_BOOST = 20, at its
           temperature. Why 20: the heat law's h is a solid in a liquid; a hand in
           a liquid has the surface at the liquid's temperature within a second
           (a thin film against the skin's own effusivity), which is how Moritz and
           Henriques's table was taken. At h = 1500 a scald at 60 C needs 23 s for
           2nd degree, at 30000 7 s (their table: about 5 s) (calibration (u), the
           skin law is the same);
  contact  a neighbour solid: the interface temperature of two semi-infinite
           bodies (skinburn.contact_T) with the solid's effusivity.

A face is created only when its exposure is hot (gas or liquid at 38 C or more,
a solid at 44 C or more) and is KEPT: p["body"]["skin"] holds every face that
was ever hot, keyed (bone, cell index in the bone, direction in the body's own
frame), so a mark moves with the limb. Faces that are cool and unburned are
pruned (skinburn.prunable); a burned face is a record and stays. Cold injury
(omega_c) is not wired (a stretch goal).

WHAT READS IT
  p["burn"]       stays as a WRITTEN VIEW, the scalar whose old shock curve
                  (injury.shock_loss) gives the plasma loss the faces calculate:
                  loss = 0.012 x weighted %TBSA x (1 - exp(-t / 6 h)) summed over
                  the events when a face crossed into 2nd or 3rd degree
                  (skinburn.shock_events). So injury.flow_factor, the death text
                  and the legacy readers all see the new shock, and the old lines
                  "faints at 0.2, kills at 0.45" are gone (the new lines come
                  from the %TBSA). It only climbs.
  p["hand_state"][arm]  the squeeze a burned hand leaves: 1 - 0.5 x the share of
                  the forearm bone's faces burned (the hand rides in that bone;
                  grip.state_factor reads it) (u).
  p["pain_heat_m2"]     the area-weighted heat pain, m2 (a face's pain 0..1 at
                  its basal node times its area): the vagal drive's input.

TBSA is counted against the body's OWN surface faces in the lattice, not Du
Bois's figure: the lattice body is thinner than a person (a stick figure, 45 kg
at 1.35 m), and a share of ITS surface is what a burn takes. Face area is the
voxel's own.

PERSISTENCE: only a revival resets a burn (`revive_reset`). The lattice has no
revival of its own: it is the lab's setting (lab/world2 `perma`: a person who
dies stays dead; without it the next night's person is a fresh record), and a
fresh record has no faces. So no flag is read here; whoever revives a body calls
`revive_reset`. Healing is a later wave.

Numbers marked (u) are unverified. The gate is World.skin_faces (PhysiologyMixin,
default True); False runs the old scalar bit for bit.
"""
import numpy as np

from . import injury as IJ, skinburn as SB
from .materials import AIR, FLESH, FLUID, TICK_S

T_ACT_HOT = 38.0            # a gas or liquid this warm starts a face (u)
T_ACT_SOLID = 44.0          # a solid this warm starts a face: where damage starts (Moritz and Henriques)
WARM_T = 34.0               # a face this warm at its top six nodes is still settling (skinburn.prunable)
HAND_BURN_WEIGHT = 0.5      # a fully burned hand keeps half its squeeze (u)
GAS_H_FIRE = 25.0           # W/m2K, EN 1991-1-2 convective coefficient of a fire-exposed surface (k)
LIQUID_SKIN_BOOST = 20.0    # see the module note: skin in a liquid follows the liquid (u)
FLUID_FRAC = 0.25           # a cell holding more than this of its volume as fluid is liquid for the skin (u)
SKIN_STRIDE = 12            # faces step every 12th tick with dt x 12 (backward Euler, stable); error measured in tests
                            # the every-tick law is measured in tests/test_skin_faces_wired.py (u until then)
EV_FOLD_N = 24              # fold the shock events when the list passes this
DIRS_BODY = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))   # f, l, u, each both ways
LABEL = ("f+", "f-", "l+", "l-", "u+", "u-")


def _world_dirs(chest):
    """The six body-frame directions as lattice unit vectors for this chest vector
    (forward is the way the chest faces, lateral the way its right hand side runs)."""
    cx, cy = int(round(chest[0])), int(round(chest[1]))
    out = []
    for f, l, u in DIRS_BODY:
        out.append((f * cx + l * cy, f * cy - l * cx, u))
    return out


def _rec(p):
    return p.setdefault("body", {})


def skin_of(p):
    """The faces of a person, {key: face}, or {} for one never hot."""
    return ((p.get("body") or {}).get("skin")) or {}


def revive_reset(p):
    """A revival: the body starts afresh, so its burn record goes (and the view)."""
    b = p.get("body")
    if b:
        for k in ("skin", "skin_ev", "skin_tw", "skin_n", "skin_nb", "skin_cnt"):
            b.pop(k, None)
    p["burn"] = 0.0
    p.pop("hand_state", None)
    p.pop("pain_heat_m2", None)


def _bone_cells(world, p, flesh, sl):
    """{(x, y, z): (bone, index)} for the body's cells, from its segments (offsets from
    the body's own corner, the same order every tick) and the corner."""
    segs = p.get("segs") or {}
    own = world._claim_cells(p, flesh, sl)
    base = own.min(axis=0)
    look = {}
    for b in sorted(segs):
        c = np.asarray(segs[b]) + base
        for i, xyz in enumerate(c.tolist()):
            look[tuple(xyz)] = (b, i)
    return look, base


def _exposure(world, T, nb, mat_n):
    """The exposure of one face from its neighbour cell nb (a lattice index tuple),
    or None when the neighbour is not something that burns or when it is another
    body's flesh."""
    Tn = float(T[nb])
    if mat_n == AIR:
        fv = float(world.fvol[nb])
        if fv > FLUID_FRAC * world.cap:
            row = FLUID.get(int(world.fl[nb]))
            h = (row[6] if row is not None else SB.LIQUID_H["water"]) * LIQUID_SKIN_BOOST
            return {"kind": "liquid", "T_liq": Tn, "h": float(h)}, Tn >= T_ACT_HOT
        return {"kind": "gas", "T_gas": Tn, "h_c": GAS_H_FIRE, "T_env": 20.0}, Tn >= T_ACT_HOT
    if mat_n == FLESH:
        return None, False
    return {"kind": "contact", "T_o": Tn, "e_o": IJ.effusivity(mat_n)}, Tn >= T_ACT_SOLID


_TEMPLATES = {}


def _new_face(region, T_core):
    """A resting face for a region (a copy of a cached template: the steady profile
    is one solve per region and core temperature, not one per face)."""
    key = (SB.dermis_mm_of(region), round(T_core, 1))
    t = _TEMPLATES.get(key)
    if t is None:
        t = _TEMPLATES[key] = SB.new_face(None, region, T_core=T_core)
    return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in t.items()}


def step(world, p, T, flesh, sl):
    """Advance one person's faces by a tick and write the views. `flesh` is the mask of
    their cells over the box `sl`, `T` the lattice temperature. Does nothing for a body
    that has no face still settling and whose neighbourhood is nowhere hot."""
    stride = SKIN_STRIDE if getattr(world, "skin_stride", True) else 1
    if stride > 1 and (world.tick + p["pid"]) % stride:
        return                                   # not this person's tick; the next one covers it with dt x stride
    b = p.get("body")
    faces = (b or {}).get("skin") or {}
    nx, ny, nz = world.shape
    lo = [max((s_.start or 0) - 1, 0) for s_ in sl[:3]]
    hi = [min(s_.stop if s_.stop is not None else n, n) + 1 for s_, n in zip(sl[:3], (nx, ny, nz))]
    hot_near = float(T[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].max()) >= T_ACT_HOT
    warm = [k for k, f in faces.items() if f["warm"]]
    if not hot_near and not warm:
        return
    look, base = _bone_cells(world, p, flesh, sl)
    wd = _world_dirs(p.get("chest", (0.0, 1.0)))
    off = np.array([s_.start or 0 for s_ in sl[:3]])
    mat = world.mat
    todo = {}                                    # key -> (world cell, direction index)
    # (a) the faces that are still settling: their cell and side are in their record
    segs = p.get("segs") or {}
    for key in warm:
        bone, i, d = faces[key]["k"]
        seg = segs.get(bone)
        if seg is None or i >= len(seg):
            continue
        todo[key] = (tuple((np.asarray(seg[i]) + base).tolist()), d)
    # (b) every exposed face of the body, to count its surface and to start a face where
    # the neighbour is hot enough to
    rec = _rec(p)
    if hot_near:
        cells = np.argwhere(flesh) + off
        names = sorted(segs)
        bid = np.array([names.index(look[c][0]) if c in look else len(names)
                        for c in map(tuple, cells.tolist())])
        n_bone = np.zeros(len(names) + 1, np.int64)
        n_total = 0
        sh = np.array(flesh.shape)
        for d, vec in enumerate(wd):
            nbs = cells + np.array(vec)
            rel = nbs - off
            inbox = ((rel >= 0).all(axis=1) & (rel < sh).all(axis=1))
            own = np.zeros(len(cells), bool)
            own[inbox] = flesh[rel[inbox, 0], rel[inbox, 1], rel[inbox, 2]]
            inb = ((nbs[:, 0] >= 0) & (nbs[:, 0] < nx) & (nbs[:, 1] >= 0) & (nbs[:, 1] < ny)
                   & (nbs[:, 2] >= 0) & (nbs[:, 2] < nz))
            idx = np.nonzero(inb & ~own)[0]
            n_total += len(idx)
            n_bone += np.bincount(bid[idx], minlength=len(names) + 1)
            Tn = T[nbs[idx, 0], nbs[idx, 1], nbs[idx, 2]]
            for j in idx[Tn >= T_ACT_HOT].tolist():
                c = tuple(cells[j].tolist())
                bone, i = look.get(c, ("?", hash(c) & 0xFFFFFF))
                key = SB.face_key(bone, i, LABEL[d])
                if key not in todo:
                    todo[key] = (c, d)
        rec["skin_n"] = n_total
        rec["skin_nb"] = {n: int(n_bone[k]) for k, n in enumerate(names)}
    if not todo:
        return
    faces = rec.setdefault("skin", {})
    T_core = float(p["heat"]["T_pool"]) if p.get("heat") and "T_pool" in p["heat"] else SB.T_ARTERIAL
    f = max(p["blood_ml"] / p["blood0_ml"], 0.0) if "blood0_ml" in p else 1.0
    keys, exps, act = [], [], []
    for key, (c, d) in todo.items():
        v = wd[d]
        nbc = (c[0] + v[0], c[1] + v[1], c[2] + v[2])
        if not (0 <= nbc[0] < nx and 0 <= nbc[1] < ny and 0 <= nbc[2] < nz):
            continue
        e, active = _exposure(world, T, nbc, int(mat[nbc]))
        if e is None:
            continue
        face = faces.get(key)
        if face is None:
            bone, i, lab = key.split("|")
            face = _new_face(bone if bone != "?" else "torso", T_core)
            face["k"] = (bone, int(i), LABEL.index(lab))
            face["warm"], face["deg"] = True, 0
            faces[key] = face
        keys.append(key)
        exps.append(e)
        act.append(active)
    if not keys:
        return
    fcs = [faces[k] for k in keys]
    SB.step(fcs, TICK_S * stride, exps, f=f, T_core=T_core)
    Tb = np.array([fc["T"] for fc in fcs])
    om = np.array([fc["omega"] for fc in fcs])
    deg = np.where(om[:, SB.DERMIS_BASE] >= 1.0, 3, np.where(om[:, SB.BASAL] >= 1.0, 2, 0))
    pain = np.clip((Tb[:, SB.BASAL] - SB.PAIN_T0) / SB.PAIN_SPAN, 0.0, 1.0)
    # still settling: its exposure is hot now, or the heat is still in the skin
    warm_now = np.asarray(act) | (Tb[:, :SB.DERMIS_BASE + 1].max(axis=1) >= WARM_T)
    counts = rec.setdefault("skin_cnt", {"n2": 0, "n3": 0, "bone": {}})
    drop = []
    for j, fc in enumerate(fcs):
        was, now = fc["deg"], int(deg[j])
        if was != now:
            counts["n2"] += (now == 2) - (was == 2)
            counts["n3"] += (now == 3) - (was == 3)
            if (now >= 2) != (was >= 2):
                bn = fc["k"][0]
                counts["bone"][bn] = counts["bone"].get(bn, 0) + (1 if now >= 2 else -1)
            fc["deg"] = now
        fc["warm"] = bool(warm_now[j])
        if not fc["warm"] and not om[j].any():
            drop.append(keys[j])                  # cool and unburned: it carries nothing
    for k in drop:
        del faces[k]
    _readouts(world, p, rec, counts, float(pain.sum()))


def _readouts(world, p, rec, counts, pain_faces):
    """TBSA, the shock events, the p["burn"] view, the hand's state and the pain."""
    n_tot = max(int(rec.get("skin_n") or 0), 1)
    pct_w = (SB.TBSA_WEIGHTS[2] * counts["n2"] + SB.TBSA_WEIGHTS[3] * counts["n3"]) / n_tot * 100.0
    prev = rec.get("skin_tw", 0.0)
    ev = rec.setdefault("skin_ev", [])
    t_s = world.tick * TICK_S
    if pct_w > prev:
        ev.append([t_s, pct_w - prev])
        rec["skin_tw"] = pct_w
        if len(ev) > EV_FOLD_N:
            rec["skin_ev"] = ev = SB.fold_events(ev, t_s)
    loss = SB.shock_events(ev, t_s)
    view = float(np.interp(loss, IJ._SHOCK_Y, IJ._SHOCK_X))
    if view > p["burn"]:
        p["burn"] = min(view, 1.0)
    p["pain_heat_m2"] = pain_faces * (0.1 * world.scale) ** 2
    nb = rec.get("skin_nb") or {}
    for limb in (a for a in (p.get("chain") or {}) if a.endswith(" arm")):
        fb = (p["chain"][limb] or [None])[-1]      # the forearm bone: the hand rides in it
        nbn = counts["bone"].get(fb, 0)
        if nbn > 0:
            share = nbn / max(nb.get(fb, 0), nbn)
            p.setdefault("hand_state", {})[limb] = max(1.0 - HAND_BURN_WEIGHT * share, 0.0)
