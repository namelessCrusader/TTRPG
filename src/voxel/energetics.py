"""ENERGETICS: ONE SOURCE OF POWER (wave 13, T7 section 3.4).

A body makes heat from one place and is billed once:

    P_total = P_tissue + P_gait + sum(P_act) + P_shiv

  P_tissue  the basal heat of the tissues (Pennes q_m, tissue.py), which
            already carries fever and cold through Q10;
  P_gait    (MET - MET_b) x MET_W_PER_KG x mass. The sim's MET is the ACSM
            number and it already CONTAINS rest, so the part that is rest
            (MET_b = P_tissue / (1.22 m)) is taken off. At rest P_total is
            exactly 1.22 W/kg x m, with nothing counted twice;
  P_act     discrete acts (a lift, a swing, a shove, a drag): the joules the
            sim delivers to the world divided by the muscle efficiency;
  P_shiv    shivering (thermoreg.py).

THE ONE-FLOW RULE. A joule an actor delivers to a mechanism is debited from
the ACTOR's stores and credited to `work_out`, the reservoir of mechanical
energy the person has given the world. It is NEVER also booked as `mech_in`,
which is for releases with no actor (a spring let go). `spend_J` is the one
hook; `note_work` refuses a second bill for the same tag in the same second
(gait is billed by MET, so a step is never also billed as a lift).

These two numbers are the only copies in the code base; T10 (fatigue) and
T11 (stairs) import them:  MET_W_PER_KG and ETA.
"""
import math

import numpy as np

from . import digestion
from .tissue import ISO_MET_W_PER_M2, bsa_dubois

# ACSM: 1 MET = 3.5 ml O2/(kg min) = 3.5 x 20.9 J/ml / 60 = 1.22 W/kg.
MET_W_PER_KG = 1.22
# Gross muscle efficiency of positive work (draw 0.20-0.25). (u)
ETA = 0.22
# Aliases other topics may already use; they are the SAME numbers.
MET_W_KG = MET_W_PER_KG
ETA_GROSS = ETA
# Eccentric (lowering) work costs |W| / 1.2 metabolically (about 0.83 |W|).
ECC_DIV = 1.2
GRAV = 9.81

# ACSM walking and running: VO2 = 3.5 + 0.1 v (v in m/min), 0.2 v for running.
RUN_MPS = 2.2

# Kinds of billed work: positive work given to the world, or negative work
# the world gives back to the muscle (a load lowered, a fall caught).
POSITIVE = ("lift", "push", "drag", "swing", "throw", "carry_up", "climb")
NEGATIVE = ("lower", "catch")

# --- which zones do the work (shares of the heat; u) ------------------------
# Groups, not 14 names, so the table is the same for any zone naming.
USE = {
    "gait": {"thigh": 0.40, "shin": 0.35, "torso": 0.20, "uarm": 0.05},
    "swing": {"uarm": 0.30, "farm": 0.25, "torso": 0.45},
    "carry": {"uarm": 0.25, "farm": 0.25, "hand": 0.10, "torso": 0.40},
    "lift": {"uarm": 0.20, "farm": 0.15, "torso": 0.30, "thigh": 0.25, "shin": 0.10},
    "push": {"uarm": 0.15, "farm": 0.10, "torso": 0.25, "thigh": 0.30, "shin": 0.20},
    "shiver": {"torso": 0.50, "thigh": 0.25, "uarm": 0.075, "farm": 0.075,
               "shin": 0.05, "head": 0.05},
}
USE["drag"] = USE["push"]
USE["throw"] = USE["swing"]
USE["climb"] = USE["gait"]
USE["carry_up"] = USE["gait"]
USE["lower"] = USE["lift"]
USE["catch"] = USE["carry"]

_GROUPS = (("forearm", "farm"), ("upper arm", "uarm"), ("upperarm", "uarm"), ("uarm", "uarm"),
           ("farm", "farm"), ("hand", "hand"), ("thigh", "thigh"),
           ("shin", "shin"), ("foot", "foot"), ("head", "head"),
           ("torso", "torso"))


def group_of(zone):
    """The use-table group of a zone name ('thigh_l' -> 'thigh')."""
    z = zone.lower()
    for key, g in _GROUPS:
        if key in z:
            return g
    return "torso"


# ---------------------------------------------------------------- power ----
def met_of_speed(v_mps, grade=0.0, run_mps=RUN_MPS):
    """ACSM MET from horizontal speed (and fractional grade)."""
    v = max(v_mps, 0.0) * 60.0
    if v_mps > run_mps:
        vo2 = 3.5 + 0.2 * v + 0.9 * v * grade
    else:
        vo2 = 3.5 + 0.1 * v + 1.8 * v * grade
    return vo2 / 3.5


def met_power_w(met, mass_kg):
    return met * MET_W_PER_KG * mass_kg


def met_b_of(p_tissue_w, mass_kg):
    """The MET that the tissue term already is (about 0.9)."""
    return p_tissue_w / (MET_W_PER_KG * mass_kg)


def p_gait(met, met_b, mass_kg, floor_w=0.0):
    """The movement part of a MET number: what is left after rest. `floor_w`
    (default 0, the wave 12 law) is the least this can be: the muscle tone of
    an awake body (`rest_floor_w`)."""
    return max(max(met - met_b, 0.0) * MET_W_PER_KG * mass_kg, floor_w)


def rest_floor_w(p_tissue_w, mass_kg, height_m, awake=True):
    """N2 (a), wave 13b: a seated, awake body at rest makes the ISO 8996 / 7730
    level, 58.15 W/m2 of body surface (1 met in the ISO convention), not
    1.22 W/kg. The tissue term (P_tissue, the Pennes q_m sum) is the basal
    part, so the floor is the muscle tone and posture that an awake body adds on
    top of it. 1.22 W/kg x mass stays the unit of a MET for exercise: above rest
    the old `(MET - MET_b) x 1.22 x m` is larger and wins. Asleep, or
    unconscious, there is no floor (sleep is basal metabolism, about 0.95 of
    it; Aschoff and Wever 1958 (u))."""
    if not awake:
        return 0.0
    return max(ISO_MET_W_PER_M2 * bsa_dubois(mass_kg, height_m) - p_tissue_w, 0.0)


def load_term_w(mass_kg, load_kg, v_mps, grade_pct=0.0, terrain=1.0):
    """Pandolf (1977): the EXTRA power a carried load costs while walking:
    M(load) - M(0). Added once; a load carried at constant height has no
    mechanical work."""
    def m(l):
        return (1.5 * mass_kg + 2.0 * (mass_kg + l) * (l / mass_kg) ** 2
                + terrain * (mass_kg + l) * (1.5 * v_mps ** 2 + 0.35 * v_mps * grade_pct))
    return m(load_kg) - m(0.0)


def p_total(p_tissue_w, met, mass_kg, p_act_w=0.0, p_shiv_w=0.0, load_w=0.0):
    """The one total. met_b comes from the tissue term itself."""
    mb = met_b_of(p_tissue_w, mass_kg)
    return p_tissue_w + p_gait(met, mb, mass_kg) + load_w + p_act_w + p_shiv_w


def act_power(work_j, eta=ETA, kind="lift"):
    """What one act costs. `work_j` is the mechanical joules delivered to the
    world (positive kinds) or released by lowering (negative kinds, give the
    magnitude). Returns dict: met_J (drawn from the stores), work_out_J
    (signed credit to the mechanical-energy reservoir) and heat_J (made in
    the muscle). met_J - work_out_J - heat_J = 0 always."""
    w = abs(work_j)
    if kind in NEGATIVE:
        met = w / ECC_DIV
        # the potential energy released becomes heat in the muscle, and so
        # does the metabolic cost: both counted once.
        return {"met_J": met, "work_out_J": -w, "heat_J": w + met}
    if kind not in POSITIVE:
        raise ValueError("unknown kind of work: %r" % (kind,))
    met = w / eta
    return {"met_J": met, "work_out_J": w, "heat_J": met - w}


def heat_share(use, zone_weights):
    """Split 1.0 of heat between zones. `use` is a USE row (group -> share,
    summing to 1); `zone_weights` is {zone name: weight} (heat capacity, so a
    big voxel mass takes more). Shares sum to exactly 1 (the last takes the
    remainder)."""
    groups = {}
    for z, w in zone_weights.items():
        groups.setdefault(group_of(z), []).append((z, max(w, 0.0)))
    tot_use = sum(s for g, s in use.items() if g in groups)
    out = {z: 0.0 for z in zone_weights}
    if tot_use <= 0.0:
        n = len(zone_weights)
        return {z: 1.0 / n for z in zone_weights}
    for g, s in use.items():
        if g not in groups:
            continue
        zs = groups[g]
        wsum = sum(w for _, w in zs)
        for z, w in zs:
            out[z] += (s / tot_use) * (w / wsum if wsum > 0 else 1.0 / len(zs))
    # exact: put the rounding on the largest share
    err = 1.0 - sum(out.values())
    big = max(out, key=out.get)
    out[big] += err
    return out


# --------------------------------------------------- hyperemia (F12) --------
RHO_C_BLOOD = 3.8e6                  # J/(m3 K)
HYP_CAP = 20.0                       # blood flow rises at most 20x basal


def hyperemia_w(w_basal, q_work_w_m3, dt_av_k=1.5):
    """Local perfusion rate (1/s, as w_b) of working muscle: it rises with the
    work heat deposited in it, w = w_basal + q / (rho c_b dT_av), capped at
    HYP_CAP x basal (about 1 L of blood per 20 kJ of work heat). (u)"""
    w = w_basal + np.maximum(q_work_w_m3, 0.0) / (RHO_C_BLOOD * dt_av_k)
    out = np.minimum(w, HYP_CAP * w_basal)
    return float(out) if np.ndim(out) == 0 else out


# ------------------------------------------------------------ the bills ----
def new_book():
    """Per-person bill book, lives in p['heat']['work']."""
    return {"sec": -1, "tags": {}, "pending_zone_J": {}, "met_J": 0.0,
            "work_out_J": 0.0, "refused": 0}


def release_J(books, joules):
    """Mechanical work that an act delivered INTO the world (a drag rubs it into
    the floor as heat). It leaves `books.work_out`, and the caller puts the same
    joules into E. This module is the one writer of work_out (the one-flow rule)."""
    books.work_out -= joules
    return joules


def note_work(book, tag, now_s, kind="act"):
    """Register a bill. False (refused) if the same tag was already billed in
    this body second -- by any kind. The caller must NOT pay then."""
    sec = int(math.floor(now_s))
    if book["sec"] != sec:
        book["sec"] = sec
        book["tags"] = {}
    if tag in book["tags"]:
        book["refused"] += 1
        return False
    book["tags"][tag] = kind
    return True


def spend_J(heat, books, tag, work_j, kind, now_s, zone_weights=None,
            mass_kg=None, eta=ETA):
    """THE HOOK for discrete work (hands.py and the mechanisms call it).

    Debits the ACTOR's stores for the metabolic cost, credits `books.work_out`
    with the mechanical joules (signed), and leaves the muscle heat in
    heat['work']['pending_zone_J'] per zone for the body law to deposit.
    `books` is any object with a float `work_out`; it is never given
    `mech_in`. Returns the act_power dict, or None if the bill was refused
    (a second bill for the same tag in the same second)."""
    book = heat["work"]
    if not note_work(book, tag, now_s, kind):
        return None
    cost = act_power(work_j, eta, kind)
    st = heat["stores"]
    m = heat["sheet"]["mass_kg"] if mass_kg is None else mass_kg
    out = {"delivered_J": 0.0, "tef_J": 0.0, "heat_J": 0.0, "o2_L": 0.0, "co2_L": 0.0,
           "co2_g": 0.0, "o2_g": 0.0, "h2o_ml": 0.0, "unmet_J": 0.0}
    digestion._burn(st, cost["met_J"], 1.0, m, out)
    # the stores may be too empty to pay: what could not be paid is not
    # done -- the heat made is what was actually burned
    scale = out["delivered_J"] / cost["met_J"] if cost["met_J"] > 0 else 1.0
    books.work_out += cost["work_out_J"] * scale if kind in POSITIVE else cost["work_out_J"]
    heat_j = out["delivered_J"] - (cost["work_out_J"] * scale if kind in POSITIVE else 0.0)
    if kind in NEGATIVE:
        heat_j = out["delivered_J"] + abs(cost["work_out_J"])
    if zone_weights:
        sh = heat_share(USE[kind], zone_weights)
        pz = book["pending_zone_J"]
        for z, f in sh.items():
            pz[z] = pz.get(z, 0.0) + heat_j * f
    book["met_J"] += out["delivered_J"]
    book["work_out_J"] += cost["work_out_J"] * (scale if kind in POSITIVE else 1.0)
    book["o2_L"] = book.get("o2_L", 0.0) + out["o2_L"]
    book["co2_L"] = book.get("co2_L", 0.0) + out["co2_L"]
    cost = dict(cost, scale=scale, o2_L=out["o2_L"], co2_L=out["co2_L"], heat_J=heat_j)
    return cost
