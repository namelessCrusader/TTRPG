"""FIRST AID: the acts that write the one outside pressure (wave 15, S2 A).

There is NO bleeding rule per act. Pressing, binding and a tourniquet each only
make a pressure on the cut vessel (`vasculature` P_ext: the largest of them),
and the one flow law in `vasculature.step_wound` does the rest. An act here is a
RECORD on the wound (or the person) and the law that turns the record into
pascals; the menu rows that make a person do it are the menu lane's.

  PRESS      a hand on the wound. Force F over the pad's area, faded with the
             vessel's depth (vasculature.press_pa). It is a STATIC HOLD, so the
             hand tires by Rohmert's endurance (fatigue.step_static): at
             f = F / MVC = 0.5 a hand lasts 86 s, and when it gives out the
             pressure is gone and the clot restarts from the plug's own age.
  BIND       a strap or cloth round the limb: P = T / (r w) (vasculature.hoop_pa).
             The cloth has a tensile limit; pulled past it, it tears and is gone.
  TOURNIQUET a strap with a windlass. The cuff pressure is the strap's hoop
             pressure; an artery is shut only above AOP (vasculature.aop_pa,
             wider cuffs need less), a vein at any cuff above the venous
             pressure, and only for cuts DISTAL to it on the same limb. It can
             tear like any strap. Tissue under a closed cuff dies slowly
             (vasculature.isch_dead_frac): the dead share is a recorded state
             on the person, `p["isch"][bone] = {"t_s", "share", "dead"}`, and the
             limb law reads it (skeleton._isch_dead): a dead forearm cannot
             grip, a dead leg cannot bear weight. No limb is lost (W15-Q2).

Pure functions on plain dicts. Numbers marked (u) are unverified.
"""
import math

from . import fatigue as FT, vasculature as VA

SBP_PA = VA.MAP_0 * 4.0 / 3.0        # 16 kPa = 120 mmHg, the tourniquet's reference pressure


# -- records the acts make -----------------------------------------------------
def press(wound, F_N, area_m2, mvc_N):
    """A hand on a wound: F_N of force on a pad of area_m2, from a hand whose
    rested maximum is mvc_N. The record keeps the hand's own fatigue."""
    wound["press"] = {"F_N": float(F_N), "area_m2": float(area_m2),
                      "mvc_N": float(mvc_N), "fat": FT.new_state(), "F_eff": float(F_N)}
    wound["pressed"] = True                    # what mindlink and the old law read
    return wound["press"]


def release_press(wound):
    """Lift the hand. The clot keeps its age only while it holds on its own: the
    flow law reopens the cut and restarts the plug if the pressure was what held it."""
    wound.pop("press", None)
    wound.pop("pressed", None)


def bind(wound, T_N, r_m, w_m, T_max_N):
    """A strap or cloth of width w_m pulled to tension T_N round a limb of radius
    r_m, tearing at T_max_N."""
    wound["bind"] = {"T_N": float(T_N), "r_m": float(r_m), "w_m": float(w_m),
                     "T_max_N": float(T_max_N)}
    return wound["bind"]


def tourniquet_on(p, bone, along, T_N, r_m, width_m, T_max_N, now_s=0.0):
    """Apply a tourniquet on `bone` at `along` (0-1 down that bone from its
    proximal end). Its cuff pressure is the strap's hoop pressure."""
    tq = {"bone": bone, "limb": VA.limb_of_bone(bone), "along": VA.limb_along(bone, along),
          "pos": float(along), "order": VA.PROX_ORDER.get(VA._part_of(bone), 0.0), "T_N": float(T_N), "r_m": float(r_m), "width_m": float(width_m),
          "T_max_N": float(T_max_N), "pressure_Pa": VA.hoop_pa(T_N, r_m, width_m),
          "t0_s": float(now_s)}
    p.setdefault("tourniquets", []).append(tq)
    return tq


def tourniquet_off(p, tq):
    """Take a tourniquet off (the dead share it made stays: only healing, a
    later wave, brings tissue back)."""
    if tq in (p.get("tourniquets") or ()):
        p["tourniquets"].remove(tq)


# -- the pressure a wound's cut feels ------------------------------------------
def press_step(wound, dt):
    """Tire the pressing hand for dt seconds; returns the force on the wound now.
    A hand worn out (gam 1) or past its capacity lets go: the force is 0."""
    pr = wound.get("press")
    if not pr:
        return 0.0
    pr["fat"] = FT.step_static(pr["fat"], pr["F_eff"], pr["mvc_N"], dt)
    if not FT.holds(pr["fat"], pr["F_eff"], pr["mvc_N"]):
        pr["F_eff"] = 0.0
        wound.pop("pressed", None)
    return pr["F_eff"]


def strap_check(rec):
    """True while a strap's tension is under its tensile limit (a tourniquet or
    a bandage that is over it tears)."""
    return rec["T_N"] < rec["T_max_N"]


def cut_pext(cut, wound, tqs, sbp=SBP_PA):
    """The outside pressure on one cut, Pa: the largest of a press, a bandage's
    hoop pressure and every tourniquet on its limb that it lies distal to."""
    pe = 0.0
    pr = wound.get("press")
    if pr and pr["F_eff"] > 0.0:
        pe = max(pe, VA.press_pa(pr["F_eff"], pr["area_m2"], cut.get("depth_m", 0.0)))
    bd = wound.get("bind")
    if bd:
        pe = max(pe, VA.hoop_pa(bd["T_N"], bd["r_m"], bd["w_m"]))
    for tq in tqs:
        if tq["limb"] == cut.get("limb"):
            pe = max(pe, VA.tourniquet_pext(cut, tq, sbp=sbp, circ_m=2.0 * math.pi * tq["r_m"]))
    return pe


# -- ischaemia ----------------------------------------------------------------
def isch_step(p, tqs, dt, bones_of_limb, sbp=SBP_PA):
    """Tissue under a CLOSED cuff dies: for every tourniquet whose pressure
    reaches the artery's occlusion pressure, the bones distal to it on its limb
    (`bones_of_limb(limb) -> [bone, ...]`) accrue dt seconds, and the dead share
    is vasculature.isch_dead_frac of the time they have been shut off. The cuff's
    own bone counts for the share below the cuff (1 - position). Recorded on
    p["isch"]; it only grows (healing is a later wave)."""
    for tq in tqs:
        if tq["pressure_Pa"] < VA.aop_pa(sbp, 2.0 * math.pi * tq["r_m"], tq["width_m"]):
            continue                              # an open artery: the limb is perfused
        isch = p.setdefault("isch", {})
        for b in bones_of_limb(tq["limb"]):
            part_order = VA.PROX_ORDER.get(VA._part_of(b), 0.0)
            if part_order < tq["order"]:
                continue                          # above the cuff's bone
            share = 1.0 if part_order > tq["order"] else 1.0 - tq["pos"]
            rec = isch.setdefault(b, {"t_s": 0.0, "share": 0.0, "dead": 0.0})
            rec["t_s"] += dt
            rec["share"] = max(rec["share"], share)
            rec["dead"] = rec["share"] * VA.isch_dead_frac(rec["t_s"])
