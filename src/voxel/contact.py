"""CONTACT EVENTS AND THE SLIP LAW (wave 13c, lane C13; T3 spec 4.10, 4.13).

Two things the body work owed the surface layers.

1. A CONTACT EVENT. A foot comes down, a hand closes: two faces meet with a
   contact share, a pressure, a slide distance and a patch length. That is what
   `Layers.contact` takes, and nothing in the old sim made it. `touch` is the one
   writer: it moves the layers, keeps a small record of the contact (`w.contacts`,
   a ring) and says it as a typed event when something actually moved.
   `footfall` and `grasp` are the two callers.

2. THE SLIP LAW. A foot holds while the friction it needs is less than the friction
   the surface gives: required = tangential / normal force. Walking asks for about
   0.2 at normal pace (Redfern et al. 2001, "Biomechanics of slips", and the
   Perkins 1978 / Harper 1962 heel-strike measurements, 0.15 to 0.25; (u) for the
   exact figure) and more the faster you go (taken linear in speed). The surface
   gives `floor_mu`: the dry pair value, lowered by whatever lies on the floor, read
   from BOTH stores (the layer pool and the liquid film in the lattice) through the
   one formula `layer_laws.friction_mu`. When required reaches the surface, the heel
   goes: the share of the step's momentum that the friction could not carry leaves
   the body over its feet as a speed, backwards along the step. The old balance
   reflex (`_overbalanced`, `_balance_reflex`) then catches it or does not.
   Nothing here is a flag on a person.

PURE LAWS take the world as an argument and read it only; the two methods that
write are `touch` (the pool) and the callers in motion.py and hands.py.
"""
import functools
from collections import deque

import numpy as np

from . import layer_laws as LL
from . import matfric
from .layer_data import AMBIENT, SUBSTRATE, substrate
from .layers import TAG_BODY, cell_key, make_key
from .materials import AIR, FLESH, FRICTION, GRAVITY

RCOF_WALK = 0.20          # friction a normal walking step asks of the floor (u, 0.15 to 0.25)
FOOT_AREA = 0.02          # m2 of sole under one foot in single support
FOOT_LEN = 3              # cells of floor a foot covers along the step
FOOT_WIDE = 2             # and across it
FOOT_COVER = 0.8          # share of each covered floor face the sole touches
DUST_G = 1.0e-3          # g: a top layer on the sole below this is a smear that the next footfall leaves whole
SLIDE_HEEL = 0.0          # a clean footfall does not slide; a slip adds its own
WALK_MPS = 1.4
LUBE = {matfric.WET: 0.15, matfric.OILED: 0.08, matfric.ICED: 0.07}   # mu_lube of a lattice film (T3 spec 2.6)
REGION_FOOT, REGION_HAND = 8, 12      # body-region ids: 8 and 9 the feet, 12 and 13 the hands
RING = 256


def sole_key(p, foot):
    """The layer face of one of a person's soles (tag 2, a body region)."""
    return make_key(TAG_BODY, int(p["pid"]) * 64 + REGION_FOOT + (int(foot) & 1), 0)


def hand_key(p, arm):
    return make_key(TAG_BODY, int(p["pid"]) * 64 + REGION_HAND + (1 if arm == "right" else 0), 0)


def required_friction(speed_ratio):
    """Friction the foot needs at this speed (1.0 = walking pace)."""
    return RCOF_WALK * max(float(speed_ratio), 0.0)


def _pool_busy(w):
    L = getattr(w, "layers", None)
    return L if (L is not None and getattr(w, "surface_laws", False)) else None


@functools.lru_cache(maxsize=256)
def _pair_mu(floor, kinetic, flat):
    """Dry friction of a bare foot on this floor material (a table lookup, cached)."""
    table = (matfric.FLAT_S, matfric.FLAT_K) if flat else None
    return float(matfric.mu_pair(FLESH, floor, matfric.DRY, kinetic=kinetic, table=table))


def floor_mu(w, x, y, z, kinetic=False, dry=None):
    """Effective friction of the floor under a sole standing in cell (x, y, z): the
    top face of cell (x, y, z - 1). None when there is no floor there.

    mu_dry: the pair value when the material friction law is on (as hands.py reads
    it), else FRICTION (0.40); `dry` names it outright (the toppling pivot keeps its
    calibrated 0.40: see bodies._pivot_mu). Lowered by the layers on that face and by the liquid
    in the cell above it; the slipperier of the two stores wins."""
    if z <= 0 or not (0 <= x < w.shape[0] and 0 <= y < w.shape[1]) or z - 1 >= w.shape[2]:
        return None
    floor = int(w.mat[x, y, z - 1])
    if floor == AIR:
        return None
    mu_dry = FRICTION
    mats = getattr(w, "mat_laws", None)
    if dry is not None:
        mu_dry = float(dry)
    elif mats and "friction" in mats:
        mu_dry = _pair_mu(floor, bool(kinetic), "flat" in mats)
    # the liquid film of the lattice, in the air cell above the floor
    h_lat, lube = 0.0, None
    fv = float(w.fvol[x, y, z]) if w.mat[x, y, z] == AIR else 0.0
    if fv > 0.01 * w.cap:
        film = int(matfric.film_of(int(w.fl[x, y, z])))
        if film != matfric.DRY:
            h_lat = fv / w.cap * 0.1 * w.scale
            lube = LUBE[film]
    L = _pool_busy(w)
    sub = floor if floor in SUBSTRATE else None
    ra = substrate(sub)["rough_um"] * 1e-6
    if L is None:
        return mu_dry if lube is None else LL.friction_mu(mu_dry, lube, h_lat, ra)
    key = cell_key(x, y, z - 1, 4, w.shape)
    mu_l = L.grip(key, mu_dry, sub=sub)
    if lube is None:
        return mu_l
    return min(mu_l, LL.friction_mu(mu_dry, lube, h_lat, ra))


def slip_speed(mu_floor, speed_ratio):
    """The speed (m/s) the body is left leaning back at when the friction it needs
    is more than the floor gives. 0.0 while the foot holds.

    The foot needs mu_req; it gets mu_floor. The step's momentum the friction
    cannot carry is the share 1 - mu_floor / mu_req of the walking speed."""
    req = required_friction(speed_ratio)
    if req <= 0.0 or mu_floor >= req:
        return 0.0
    return WALK_MPS * max(speed_ratio, 0.0) * (1.0 - mu_floor / req)


# THE RECOVERY STEP ASKS FOR FRICTION TOO (wave 14b, lane H). A body whose
# extrapolated centre of mass is a distance d past its base is rescued by a step
# whose foot must push back through the floor. For an inverted pendulum of height
# h the leg force is along the line from the foot to the centre, so the foot needs
# a friction coefficient of tan(lean) = d / h: the horizontal force the floor must
# give for each newton it carries (Hof 2008 on the pendulum; Redfern et al. 2001
# for the same ratio at a heel strike, the "required coefficient of friction").
# A floor that gives less lets the foot slide out and the step buys nothing.
STEP_H_MIN = 0.25         # m: the shortest pendulum the law is asked of (as _xcom)


def step_friction_needed(past_m, height_m):
    """Friction a recovery step needs when the centre is `past_m` metres past the
    base and `height_m` above the foot: d / h (0.0 for a body that is not past)."""
    return max(float(past_m), 0.0) / max(float(height_m), STEP_H_MIN)


def step_holds(mu_floor, past_m, height_m):
    """Does the floor give the friction the recovery step needs? A floor that is
    not there (None) is not asked: the step's own 'foot over nothing' test rules."""
    return mu_floor is None or mu_floor >= step_friction_needed(past_m, height_m)


def touch(w, kind, actor, S, Ds, c_c, pressure, slide, L_p, sub_S=None, sub_D=None,
          weights=None, pattern=None, place=None, cause=None, Ts_S=AMBIENT, Ts_D=AMBIENT):
    """THE CONTACT EVENT: face S meets faces Ds. The layers move (mass exact), the
    contact is logged in `w.contacts` and, when matter moved, said as a typed
    "touches" event with result `kind`. Returns the grams moved."""
    L = _pool_busy(w)
    if L is None:
        return 0.0
    moved, _adj = L.contact(S, Ds, c_c, pressure, slide, L_p, sub_S=sub_S, sub_D=sub_D,
                            Ts_S=Ts_S, Ts_D=Ts_D, weights=weights, pattern=pattern)
    log = getattr(w, "contacts", None)
    if log is None:
        log = w.contacts = deque(maxlen=RING)
    log.append({"tick": w.tick, "kind": kind, "actor": actor, "S": int(S),
                "D": [int(d) for d in Ds], "c_c": float(c_c), "p": float(pressure),
                "s": float(slide), "L_p": float(L_p), "moved_g": float(moved)})
    if moved > 0.0:
        w.emit("touches", actor=actor, result=kind, place=place, cause=cause,
               data=(("moved_g", float(moved)),), site="contact:" + kind)
    return float(moved)


def footfall(w, p, fx, fy, zlo, facing, mass_kg, cause=None):
    """A foot comes down on the floor under cell (fx, fy, zlo). The sole carries what
    it has picked up; the floor faces under it carry what they have. A face with a
    stack gives to the sole (the adhesion law decides how much); the sole's top layer
    goes to the faces that had nothing. Cheap when nothing is there: no pool, or a
    clean sole on a clean floor, does no work at all."""
    L = _pool_busy(w)
    if L is None or zlo <= 0:
        return 0.0
    foot = p.get("_nfoot", 0)
    p["_nfoot"] = foot + 1
    sole = sole_key(p, foot)
    fxn, fyn = int(np.sign(facing[0])), int(np.sign(facing[1]))
    if fxn == 0 and fyn == 0:
        fxn = 1
    px, py = -fyn, fxn
    faces = []
    for a in range(-(FOOT_LEN // 2), FOOT_LEN // 2 + 1):
        for b in range(FOOT_WIDE):
            x, y = fx + a * fxn + b * px, fy + a * fyn + b * py
            if 0 <= x < w.shape[0] and 0 <= y < w.shape[1] and w.mat[x, y, zlo - 1] != AIR:
                faces.append((x, y, cell_key(x, y, zlo - 1, 4, w.shape), int(w.mat[x, y, zlo - 1])))
    if not faces:
        return 0.0
    loaded = [f for f in faces if L.stack(f[2])]
    carrying = bool(L.stack(sole))
    if not loaded and not carrying:
        return 0.0
    pr = mass_kg * GRAVITY / FOOT_AREA
    total = 0.0
    ev = None
    place = (float(fx), float(fy), float(zlo))
    for x, y, k, m in loaded:
        total += touch(w, "footfall", p["pid"], k, [sole], FOOT_COVER, pr, SLIDE_HEEL,
                       0.1, sub_S=m, cause=cause, place=place)
    clean = [f for f in faces if not L.stack(f[2])]
    if carrying and clean:
        # A SPENT TOP LAYER IS LEFT WHOLE. Each footfall gives up a share of the top layer,
        # which never reaches zero; a smear under a milligram is gone in the next print, and
        # the layer under it is what the next print shows (Holmes: chalk, then mud).
        top = L.stack(sole)[0]
        if sum(g for _c, g in L._comp(top)) < DUST_G:
            n = len(L.stack(sole))
            L.move(sole, clean[0][2], [1.0] + [0.0] * (n - 1))
        from . import marks
        total += touch(w, "footfall", p["pid"], sole, [f[2] for f in clean], FOOT_COVER, pr,
                       SLIDE_HEEL, 0.1, sub_D=clean[0][3],
                       pattern=marks.mirrored_top_pattern(L, sole), cause=cause, place=place)
    return total


def grasp(w, p, cells, arm):
    """A hand closes on a thing: the hand's layers meet the face of the held cell that
    looks toward it (the top face when the thing is below the hand). A hand with
    nothing on it, on a clean thing, does no work."""
    L = _pool_busy(w)
    if L is None or not len(cells):
        return 0.0
    hk = hand_key(p, arm)
    c = cells[int(np.argmax(cells[:, 2]))]
    key = cell_key(int(c[0]), int(c[1]), int(c[2]), 4, w.shape)
    if L.stack(key):
        return touch(w, "grasp", p["pid"], key, [hk], 0.6, 2e4, 0.0, 0.08,
                     sub_S=int(w.mat[tuple(c)]), place=tuple(float(v) for v in c))
    if L.stack(hk):
        return touch(w, "grasp", p["pid"], hk, [key], 0.6, 2e4, 0.0, 0.08,
                     sub_D=int(w.mat[tuple(c)]), place=tuple(float(v) for v in c))
    return 0.0
