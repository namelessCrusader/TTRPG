"""BLEEDING BY VESSEL, wired (wave 15, S2 A; owner W15-Q1: ON BY DEFAULT).

The law is `vasculature` (pure: vessel pressure, the open fraction, spasm, the
clot, the tourniquet). This file connects it to the body in the world and is
called by `physiology._law_bleed` and `physiology._blow`:

  make_cuts       when a blow opens a wound: where on the bone it is (along
                  and around, `skeleton._bone_frame`, in the bone's OWN frame)
                  and which vessels it crosses (`vasculature.crosses`). The cuts
                  live on the wound: w["cuts"].
  step_wound_cuts each tick, for one wound: the outside pressure from the acts
                  (`firstaid.cut_pext`), the hydrostatic height of the wound
                  under the heart, the body's heat and work, then the ONE flow
                  law (`vasculature.step_wound`). It returns the millilitres
                  lost, clamped to the blood there is, which `physiology`
                  pours onto the lattice by the same whole-millilitre path the
                  old capillary law used, so blood stays a conserved quantity.

The capillary ooze (400 ml/s per m2 of cut plane, clot 1.7e-6 m2 a second)
STAYS for every open wound. A vessel's flow is ADDED to it; a wound that
crosses no vessel (a bruise, a scratch, a hand) bleeds bit for bit as before.

The gate is World.vessel_bleed (class attribute on PhysiologyMixin, default
True). False runs the old law only, bit for bit: no geometry is kept, no cut
is made.

Numbers: the clot constants in vasculature (14 kPa mm, 180 s) are (u). The
first-aid check that stands in for a source: direct pressure for ten minutes is
what first-aid courses teach for a bleed that does not stop by itself
(American Red Cross first aid, "apply pressure for at least 5 to 10 minutes";
(k)); the radial artery (2.5 mm) then holds once the hand is lifted and a
femoral artery (8 mm) never does (tests/test_bleed_wired.py).
"""
import numpy as np

from . import firstaid as FA, vasculature as VA

Z_EVERY = 8                      # ticks between two reads of a wound's height under the heart


def mass_kg(p):
    """The body's own mass for the vessel sizes: its blood volume over 70 ml/kg
    (the world's body is as heavy as its own lattice flesh, a little under an adult)."""
    return p["blood0_ml"] / VA.ML_PER_KG


def make_cuts(world, p, wound, cell):
    """Fill a NEW open wound with its place and its cuts. `cell` is the struck
    lattice cell. A bone with no vessel row (a hand, a foot) or a body with no
    frame gets no cuts."""
    bone = wound.get("bone")
    region = VA.region_of_bone(bone) if bone else None
    if region is None:
        return
    fr = world._bone_frame(p, bone)
    if fr is None:
        return
    az, along = world._frame_az_along(fr, cell)
    if bone == "torso" and VA._AN.region_of_zone("torso", 1.0 - along) != "chest":
        return                    # a belly wound reaches no named vessel (organs: impact only)
    wound["az"], wound["along"] = az, along
    sheet = {"mass_kg": mass_kg(p)}
    tc = VA.tendon_cut(wound, region, sheet)
    if tc > 0.0:
        # a forearm cut through the flexor tendons: the grip law reads it (grip.tendon_cut)
        arm = VA.limb_of_bone(bone)
        cut = p.setdefault("body", {}).setdefault("tendon_cut", {})
        cut[arm] = min(cut.get(arm, 0.0) + tc, 1.0)
    cuts = VA.crosses(wound, region, sheet)
    if not cuts:
        return
    D = VA.limb_thickness_m(region, sheet)
    reg = region.split(" ", 1)[1] if region[:4] in ("left", "righ") else region
    for c in cuts:
        v = VA._AN.VESSEL_OF[c["vessel"]]
        c["bone"] = bone
        c["limb"] = VA.limb_of_bone(bone)
        c["depth_m"] = v["depth_mm"] / VA._AN.D_ADULT_MM[reg] * D
        c["along_bone"] = c["along"]
        c["along"] = VA.limb_along(bone, c["along"])
    wound["cuts"] = cuts
    wound["z_rel_m"] = _height_under_heart(world, p, fr, along)


def _height_under_heart(world, p, fr, along):
    """How far the wound is BELOW the heart, metres (negative above it)."""
    z_w = float(fr["o"][2] + (along - 0.5) * (fr["t1"] - fr["t0"]) * fr["axis"][2])
    return (fr["z_heart"] - z_w) * 0.1 * world.scale


def refresh_height(world, p, w, frames):
    """Re-read the wound's height under the heart every Z_EVERY ticks (`frames`: this
    person's bone frames read this tick, so two wounds on a bone cost one read)."""
    if world.tick % Z_EVERY:
        return
    if w["bone"] not in frames:
        frames[w["bone"]] = world._bone_frame(p, w["bone"])
    fr = frames[w["bone"]]
    # a body that has fallen has no frame (its segments are the standing ones): it lies flat,
    # so the wound is level with the heart (a stated softness)
    w["z_rel_m"] = _height_under_heart(world, p, fr, w.get("along", 0.5)) if fr is not None else 0.0


def p_ext_cuts(w, tqs):
    """The outside pressure on each of a wound's cuts, Pa."""
    return [FA.cut_pext(c, w, tqs) for c in w["cuts"]]


def step_wound_cuts(world, p, w, v_circ, dt, frames=None):
    """Advance one wound's cuts by dt seconds; returns the millilitres lost.
    `v_circ` is the blood still circulating (not pooled, not already owed)."""
    heat = p.get("heat")
    T_body = float(heat["T_pool"]) if heat and "T_pool" in heat else 37.0
    FA.press_step(w, dt)
    bd = w.get("bind")
    if bd and not FA.strap_check(bd):
        w.pop("bind")                              # the cloth tore
    tqs = p.get("tourniquets") or ()
    pe = p_ext_cuts(w, tqs)
    refresh_height(world, p, w, {} if frames is None else frames)
    ml = VA.step_wound(w["cuts"], v_circ, p["blood0_ml"], pe, p.get("met", 1.0), dt,
                       T_body, w.get("z_rel_m", 0.0), alive=p["alive"])
    return ml


def tourniquets_step(world, p, dt):
    """Tear the cuffs that are pulled past their strap, then let the tissue
    under a closed cuff die."""
    tqs = p.get("tourniquets")
    if not tqs:
        return
    for tq in [t for t in tqs if not FA.strap_check(t)]:
        tqs.remove(tq)                             # the strap tore: the limb is perfused again
    FA.isch_step(p, tqs, dt, lambda limb: [b for b in (p.get("segs") or {})
                                           if VA.limb_of_bone(b) == limb])
