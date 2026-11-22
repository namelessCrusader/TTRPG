"""Blows and wounds, bleeding, perfusion, breath, burns, faint and death."""
import numpy as np
from . import injury as IJ, matedge as ME
from .faintrates import o2_target
from .materials import (AIR, BLEED, BLOOD, BODY, FLESH, GRAVITY, O2_PER_L,
                        SMOKE_EXT, TICK_S, WOUND)

# The Mind's harm slot each faint names as its cause (V3, `_law_life`).
_HARM = {"the foul air": "air", "no air": "air", "blood loss": "wounds",
         "the burns": "burns", "the wounds": "wounds",
         "the cold": "cold", "the heat": "heat",      # (T7: the core temperature bands)
         "a blow": "wounds", "the pain": "wounds"}    # (T10, C7: a head blow; the vasovagal reflex)
# Air with less oxygen than this, as a fraction of fresh, is bad air even
# with no smoke in it: a sealed room the fire has used up.
BAD_AIR = 0.95
# THE BREATHING CLOCK (realism gap G9: rows B21-B24 and P11 of
# docs/realism-references.md). Blood oxygen used to chase the air with a
# 1.25 s time constant, a face with no air beside it died in under two
# seconds, a face under water breathed the oxygen the water cell still held,
# and smoke was read in grams PER VOXEL, which is eight times more at 10 cm
# than at 5 cm for the same air. Every number here is per litre, per kilogram
# or per second, so it means the same thing at any voxel size.
BREATH = {
    # THE BODY'S OXYGEN STORE, and what it burns. About 20 ml/kg is held in
    # the lungs, the blood and the muscle (~1.4 L in a 70 kg adult), and a
    # resting body burns 3.5 ml/(kg min) (one MET). With no air at all,
    # `blood_o2` (the store as a fraction of full) falls by their ratio:
    # 0.175 a minute at rest, so a held breath reaches the faint line (0.55)
    # in about 2.5 min from full -- B22: unconscious after 1-3 min.
    "store_ml_kg": 20.0, "vo2_ml_kg_min": 3.5,
    # AN UNCONSCIOUS BODY BURNS LESS: the muscles stop working and the
    # diving reflex slows the heart. Set from B22's own clock, not measured:
    # unconscious at 1-3 min, dead at about 10, so the store between the
    # faint line and the death line (0.30 of it) must last minutes, not the
    # 100 s it would at the waking rate. Half the resting rate makes it
    # 3.4 min; awake-then-faint dies at about 6 min, out-from-the-start at 8.
    "out_met": 0.5,
    # BREATHING, the blood follows the air with this time constant: the
    # lungs wash out to the new air in a few breaths and the blood follows.
    # At 5% O2 (B21, B-T2) the faint line comes at 36 s ("coma in under
    # 40 s") and death at 2.8 min; at 10% the faint comes at 79 s; at 14%
    # the blood settles at 0.67 and never faints. The thresholds themselves
    # (BODY faint_o2, death_o2) were already right; only the clock was not.
    "tau_s": 40.0,
    # THE MOUTH: air touching the top 20 cm of the body (standing, the head;
    # lying, the whole top of the body). Metres, so 4 layers at 5 cm (as it
    # always was) and 2 at 10 cm.
    "head_m": 0.2,
    # A cell more than half full of liquid gives no air (row B22).
    "flooded": 0.5,
    # WORK. Oxygen use and breathing rise with how fast the body moves:
    # ACSM's walking equation, VO2 = 3.5 + 0.1 v (ml/(kg min), v in m/min),
    # and its running one (0.2 v) above `run_mps`. Smoothed over `met_s`,
    # because a stride is not a sprint. Breath volume (for the CO dose)
    # rises with it: Purser's 8.5 L/min at rest, 25 for light work.
    "run_mps": 2.2, "met_s": 2.0, "max_mps": 8.0, "rmv_l_min": 8.5,
    # CARBON MONOXIDE, the Purser dose (row B23):
    #     %COHb gained = 3.317e-5 * ppm^1.036 * RMV (L/min) * t (min)
    # There is no CO field yet, so CO rides with the smoke: the sim's smoke
    # is 0.6 g per g of wood burned (SOOT, "all visible products"), and a
    # pre-flashover room fire makes about 0.04 g of CO per g burned (the
    # design yield fire engineers use for mixed furnishings; well-ventilated
    # wood alone makes ~0.004, a fire short of air up to 0.2). So
    # 0.04 / 0.6 g of CO per g of smoke. At the 0.12 g/L of a smoke-logged
    # room that is ~6900 ppm, and 30% COHb comes in ~4 min of light work.
    # Gas species retire this ratio.
    "co_per_smoke": 0.04 / 0.6, "co_g_l": 1.16,      # CO density, 20 °C
    "co_half_min": 320.0,         # COHb half-life breathing air (row B24)
    # WHAT COHb COSTS THE BRAIN. Each unit of COHb takes that haemoglobin
    # away AND makes the rest hold on to its oxygen, so it costs more than
    # its share. 1.5 puts Purser's incapacitation (30% COHb) on the faint
    # line and death at 50% COHb, inside the 50-70% the literature gives.
    # Coming round needs COHb back under 20% (`wake_o2`): from a collapse at
    # 30%, about three hours on room air. Not seconds after rescue.
    "co_cost": 1.5,
    # SMOKE AT THE MOUTH, as `p["smoke"]` (0 to 1): the extinction
    # coefficient there (SMOKE_EXT per g/L, 1/m) over this one, where you
    # cannot see a metre. The will layer coughs (`chokes`) above 0.05 of it,
    # which is sigma 0.15 /m: visibility still 20 m (at C = 3), well inside
    # P11's "most people who met smoke could still see 9 m or more" (no
    # numeric odour threshold was found). Per litre, so the same at every
    # voxel size (P-T4).
    "smoke_full_sigma": 3.0,
    # The reading rises at once and falls with this half-life when the air
    # clears: the throat clears in a breath. Kept short on purpose. At 5 s
    # a body that saw a fire 40 cm away and caught one whiff of it while it
    # was still answering the sight went on "choking" for seconds in clean
    # air, and answered that instead (tests/test_voxel.py `_alarm_room`).
    "cough_half_s": 1.0,
}


class PhysiologyMixin:

    def _blow(self, p, cell, joules, area, tip=None):
        """A blow of `joules` arriving on `area` square metres at `cell`.

        Met by each layer in turn — armour, skin, tissue — and what it does is
        read off energy per area at each one. Returns the wound it leaves and
        records it on the person. A wound is a fact about a BONE — it moves
        with the arm it is in, it does not stay in the cell where the blade
        met it — so it lives on the body and not on the lattice, where every
        law that moves matter would have to remember to carry it. (That is
        the exact class of bug the particle seam kept hitting.)
        """
        if p is None or not p["alive"] or joules <= 0.0:
            return None
        vox_m = 0.1 * self.scale
        bone = self._bone_at(p, cell)
        e, a = float(joules), max(float(area), 1e-9)
        worn = (p.get("worn") or {}).get(bone)
        armour = None
        if worn:
            # THE PLATE FIRST. An edge that cannot shear a plug out of it is
            # stopped, and what reaches the body is the same energy spread over
            # the plate — or, for mail, hardly spread at all.
            y = WOUND["yield_Pa"].get(worn["mat"], 0.0)
            need = WOUND["plug"] * y * a * worn["thick_m"]
            if e > need:
                e -= need                           # through, with what is left
                armour = "pierced"
            else:
                if worn.get("kind", "plate") == "plate":
                    spread = worn.get("area_m2", vox_m * vox_m)
                else:
                    spread = a * WOUND["mail_spread"]
                a = max(a, spread)
                armour = "turned"
        wound = {"bone": bone, "tick": self.tick, "joules": round(e, 2),
                 "area_m2": a, "depth_m": 0.0, "length_m": 0.0,
                 "armour": armour, "open": False, "through": False}
        # THE WOUND SPLIT (C8b): skin opens under the TIP (L x 2r) at the
        # rescaled constant, the cut deepens under the wedge area `a`. A new
        # edge opens at today's energy; a worn one needs more. A plate that
        # turned the blow spread it, so the edge is no longer the contact.
        if tip is None or armour == "turned":
            skin = WOUND["skin_Jm2"] * a
        else:
            skin = ME.skin_constant() * tip
        if e > skin:
            # THE EDGE IS IN. How deep is what is left over the pressure it
            # takes to keep cutting, capped at the limb it is cutting through.
            depth = (e - skin) / (WOUND["tissue_Pa"] * a)
            thick = self._thickness(p, bone) if bone else vox_m
            wound.update(depth_m=min(depth, thick),
                         length_m=a / WOUND["edge_w_m"], open=True,
                         through=bool(depth >= thick))
            self._blood0(p)                     # it has blood to lose now
        # WHATEVER IT DID, IT HURT. The tissue-damage integral is unchanged —
        # the same bruise threshold per contact area the fall law uses — so a
        # club and a blade of the same energy leave the same bruise budget and
        # differ in whether they ALSO open the body.
        b_hit = BODY["bruise_kJm2"] * 1000.0 * a
        if e > b_hit:
            self._wound(p, e - b_hit)
            if bone == "head":
                IJ.head_blow(p, e, self.tick)       # T10: the head's own knock-out
        if wound["open"] or e > b_hit or armour:
            p.setdefault("wounds", []).append(wound)
            # DEFERRED (`_defer`, not `emit`): see its docstring. `because`
            # is already set to the `hits` event's box by `_spend`, if this
            # blow came from one -- a wound caused by a fall carries
            # whatever _land_body set up instead, and one with neither just
            # carries the current (possibly None) cause, same as any emit.
            box = self._defer("wounds", target=("person", p["pid"]), place=cell,
                              result="through" if wound["through"]
                              else "open" if wound["open"] else "bruise",
                              why=armour, db=e, text="")
            # AND THE BODY KEEPS IT (V3): a faint from blood loss, or a death
            # from the wounds, ticks later, is caused by this. A box until
            # the tick ends (`_unbox` swaps in the id).
            self.minds[p["pid"]].harm["wounds"] = box
            self._boxed = True
        return wound

    def _blood0(self, p):
        """How much blood this body started with: ml per kg of its own flesh."""
        if "blood0_ml" not in p:
            comp, sl = self._person_cells(p)
            kg = float(self.smass[sl][comp].sum()) / 1000.0 \
                if comp is not None and comp.any() else 45.0
            p["blood0_ml"] = BLEED["ml_per_kg"] * kg
            p.setdefault("blood_ml", p["blood0_ml"])
        return p["blood0_ml"]

    def _perfusion(self, p):
        """How much of the blood's oxygen still reaches the brain (0..1)."""
        if "blood0_ml" not in p:
            return 1.0
        f = (p["blood_ml"] - p.get("pooled_ml", 0.0)) / max(p["blood0_ml"], 1e-9)
        x = (f - BLEED["f0"]) / (1.0 - BLEED["f0"])
        return float(np.clip(x, 0.0, 1.0) ** BLEED["k"])

    @staticmethod
    def _air_o2(p):
        """What the blood's oxygen is worth with its carbon monoxide: COHb
        takes haemoglobin away and makes the rest hold on (`co_cost`)."""
        co = max(1.0 - BREATH["co_cost"] * p.get("cohb", 0.0), 0.0)
        return p["blood_o2"] * co

    def _delivered_o2(self, p):
        """Oxygen the brain actually gets: what the blood carries, less what
        its carbon monoxide costs, times how much blood is still going round
        to carry it. Burn shock and the vasovagal outflow (T10, C7) take their
        share of the flow on the same road (`injury.flow_factor`, 1.0 for a
        body with neither)."""
        return self._air_o2(p) * self._perfusion(p) * IJ.flow_factor(p)

    def _mouth_air(self, p, flesh, sl, fw=None):
        """What the mouth breathes: (gas, frac, smoke_g_per_L).

        The air touching the TOP of the body (standing, head height; collapsed,
        the floor layer): `gas` is whether any air cell touches it, `frac` its
        oxygen as a fraction of fresh, and the smoke in grams per litre. Read
        only. It is `_law_life`'s own arithmetic moved into a method so that a
        twin (slowtwins.mouth_air) can be tested against it; a tick gives the
        same numbers it always did. `fw` is the body's cells when the caller
        has them already."""
        fresh = O2_PER_L * self.vox_l
        if fw is None:
            fw = self._claim_cells(p, flesh, sl, shift=False)
        zs = fw[:, 2]
        z_lo, z_hi = int(zs.min()), int(zs.max())
        top = max(int(round(BREATH["head_m"] / (0.1 * self.scale))) - 1, 0)
        # the mouth breathes the air touching the body's TOP — standing,
        # that's head height; collapsed, the floor layer. Asked in a small
        # box round the head and not over the whole body's box: the same
        # cells (the six neighbours of the top layers, none on the body
        # box's own faces), without six rolled copies of the box a tick.
        zb = max(z_hi - top, z_lo)
        cap_ = flesh[:, :, zb:z_hi + 1]
        hx = np.flatnonzero(cap_.any(axis=(1, 2)))
        hy = np.flatnonzero(cap_.any(axis=(0, 2)))
        bx, by, bz = flesh.shape
        win = (slice(max(int(hx[0]) - 1, 0), min(int(hx[-1]) + 2, bx)),
               slice(max(int(hy[0]) - 1, 0), min(int(hy[-1]) + 2, by)),
               slice(max(zb - 1, 0), min(z_hi + 2, bz)))
        head = flesh[win].copy()
        head[:, :, :zb - win[2].start] = False
        near = np.zeros_like(head)
        near[1:] |= head[:-1]
        near[:-1] |= head[1:]
        near[:, 1:] |= head[:, :-1]
        near[:, :-1] |= head[:, 1:]
        near[:, :, 1:] |= head[:, :, :-1]
        near[:, :, :-1] |= head[:, :, 1:]
        for axis, (s, n) in enumerate(zip(win, (bx, by, bz))):
            face = [slice(None)] * 3
            if s.start == 0:                         # the body box's own
                face[axis] = 0                       # faces are left out,
                near[tuple(face)] = False            # as they always were
            if s.stop == n:
                face[axis] = -1
                near[tuple(face)] = False
        at = tuple(slice((b.start or 0) + w_.start, (b.start or 0) + w_.stop)
                   for b, w_ in zip(sl, win))
        # AIR, NOT WATER. A cell more than half full of liquid gives no
        # air: a face under water used to breathe the oxygen the water
        # cell still held, so nobody could drown.
        near &= (self.mat[at] == AIR) \
            & (self.fvol[at] <= BREATH["flooded"] * self.cap)
        gas = bool(near.any())
        if gas:
            frac = float(self.o2[at][near].mean()) / fresh
            sm = float(self.smoke[at][near].mean()) / self.vox_l  # g/L
        else:
            frac, sm = 0.0, 0.0                      # buried or drowned:
        return gas, frac, sm

    def _breathe(self, p, gas, frac, smoke_gl):
        """One tick of breathing for `p`: the oxygen store, the CO dose, and
        the smoke at the mouth. `gas` is whether the mouth has any air at all
        (none: buried, or under water); `frac` its oxygen as a fraction of
        fresh air, `smoke_gl` its smoke in grams per litre."""
        dt = TICK_S
        # HOW HARD IT IS WORKING, from how fast it moves (`_eye`, set by the
        # caller this tick). A body that is out does no work, however fast
        # it is dragged.
        eye = p["_eye"]
        last = p.get("_met_at")
        p["_met_at"] = (eye[0], eye[1])
        if not p["awake"]:
            want = BREATH["out_met"]
        else:
            v = 0.0 if last is None else float(np.hypot(eye[0] - last[0],
                                                        eye[1] - last[1]))
            v = min(v * 0.1 * self.scale / dt, BREATH["max_mps"]) * 60.0
            want = (3.5 + (0.2 if v > BREATH["run_mps"] * 60.0 else 0.1) * v) \
                / 3.5
        met = p.get("met", 1.0)
        met += (want - met) * min(dt / BREATH["met_s"], 1.0)
        p["met"] = met
        # ONE SOURCE OF POWER (T7, wave 13b): a body with a heat record burns what
        # its heat law says it makes (rest, gait, shivering, acts), so a shivering
        # body breathes harder and a still one at the ISO rest a little more than
        # 1 MET. The smoothed walking MET above stays what the heat law reads.
        h_ = p.get("heat")
        if h_ and h_.get("p_total_w") and p["awake"]:
            met = max(met, h_["p_total_w"] / (1.22 * h_["sheet"]["mass_kg"]))
        if gas:
            # BREATHING: the blood follows the air at the mouth.
            p["blood_o2"] += (o2_target(frac) - p["blood_o2"]) * dt / BREATH["tau_s"]
            # and takes in its carbon monoxide (Purser), while the old dose
            # leaves through the same lungs with a 320 min half-life
            ppm = smoke_gl * BREATH["co_per_smoke"] / BREATH["co_g_l"] * 1e6
            rmv = BREATH["rmv_l_min"] * met
            gain = 3.317e-5 * ppm ** 1.036 * rmv * (dt / 60.0) / 100.0
            cohb = p.get("cohb", 0.0)
            out = cohb * np.log(2.0) / (BREATH["co_half_min"] * 60.0) * dt
            p["cohb"] = float(min(max(cohb + gain - out, 0.0), 1.0))
        else:
            # NO AIR: the store burns down at the rate the body uses it.
            use = met * BREATH["vo2_ml_kg_min"] / (BREATH["store_ml_kg"] * 60.0)
            p["blood_o2"] = max(p["blood_o2"] - use * dt, 0.0)
            p.setdefault("cohb", 0.0)
        # what the nose and the throat feel: rises at once, lingers
        now = min(SMOKE_EXT * smoke_gl / BREATH["smoke_full_sigma"], 1.0)
        p["smoke"] = max(now, p["smoke"] * 0.5 ** (dt / BREATH["cough_half_s"]))

    def _law_bleed(self):
        """Blood leaves through every open wound, and lands where it lands.

        It comes out of the body onto the LATTICE as a fluid, at the wound's
        current place — the lowest cell of the wounded bone with air beside
        it — so it pools, runs downhill, soaks in and can be followed like
        any other liquid. Nothing is subtracted that is not poured: blood
        that cannot find room stays in the body, which keeps blood a
        conserved quantity (`total_fluid(BLOOD)` counts both).
        """
        dt = TICK_S
        for p in self.persons:
            if not p["alive"] or p["safe"]:
                continue
            open_ = [w for w in (p.get("wounds") or ()) if w.get("open")]
            if not open_:
                continue
            b0 = self._blood0(p)
            pressure = max(p["blood_ml"] / b0, 0.0)
            for w in open_:
                plane = w["depth_m"] * w["length_m"]
                w["clot_m2"] = min(w.get("clot_m2", 0.0)
                                   + BLEED["clot_m2_s"] * dt, plane)
                gap = max(plane - w["clot_m2"], 0.0)
                if gap <= 0.0:
                    w["open"] = False            # sealed: it has stopped
                    continue
                if w.get("pressed"):             # a hand on it
                    gap *= 0.1
                ml = min(BLEED["ml_s_m2"] * gap * pressure * dt, p["blood_ml"])
                if ml <= 0.0:
                    continue
                w["_owe"] = w.get("_owe", 0.0) + ml
                if w["_owe"] < 1.0:              # pour in whole millilitres'
                    continue                     # worth, not dust every tick
                got = self._spill_from(p, w["bone"], BLOOD, w["_owe"])
                w["_owe"] -= got
                p["blood_ml"] -= got
                w["shed_ml"] = w.get("shed_ml", 0.0) + got

    def _spill_from(self, p, bone, fluid, ml):
        """Pour `ml` out of a body at the outside of `bone`. Returns what fit."""
        parts = self._bone_parts(p, [bone]) if bone else None
        if not parts or not len(parts[0]):
            return 0.0
        c = parts[0][np.argsort(parts[0][:, 2])]    # lowest first: it runs down
        nx, ny, nz = self.shape
        left = ml
        for x, y, z in c:
            for dx, dy, dz in ((0, 0, -1), (1, 0, 0), (-1, 0, 0),
                               (0, 1, 0), (0, -1, 0), (0, 0, 1)):
                X, Y, Z = x + dx, y + dy, z + dz
                if not (0 <= X < nx and 0 <= Y < ny and 0 <= Z < nz):
                    continue
                if self.mat[X, Y, Z] != AIR:
                    continue
                before = float(self.fvol[X, Y, Z])
                self.pour(int(X), int(Y), int(Z), fluid, left)
                left -= float(self.fvol[X, Y, Z]) - before
                if left <= 1e-9:
                    return ml
        return ml - left

    def _aloft_cells(self, p):
        """A body in the air, as `_person_cells` would give it on the floor:
        (a mask of its flesh over the box it is in, the slices of that box),
        read at the poses the free bodies have now (`body_of`), or None when
        none of this person's flesh is up there. The box is the whole height
        of the world over the few columns he spans, as a claim's is."""
        _on, up = self.body_of(p)
        if not len(up):
            return None
        nx, ny, nz = self.shape
        c = np.rint(up).astype(np.int64)
        c = c[(c[:, 0] >= 0) & (c[:, 0] < nx) & (c[:, 1] >= 0)
              & (c[:, 1] < ny) & (c[:, 2] >= 0) & (c[:, 2] < nz)]
        if not len(c):
            return None
        x0, y0 = int(c[:, 0].min()), int(c[:, 1].min())
        x1, y1 = int(c[:, 0].max()) + 1, int(c[:, 1].max()) + 1
        mask = np.zeros((x1 - x0, y1 - y0, nz), bool)
        mask[c[:, 0] - x0, c[:, 1] - y0, c[:, 2]] = True
        return mask, (slice(x0, x1), slice(y0, y1), slice(None))

    def _crumple(self, p, comp, sl):
        """AN UNCONSCIOUS BODY CRUMPLES. Nothing holds it up any more: the
        knees give first and the whole body drops at the rate of anything
        falling (a few tens of milliseconds a centimetre, `_crumple_step`),
        and only then does what is left tip over, the way a crouched body
        does, under real g (`_tip`). It does not stand stiff and go over like
        a plank, which took a second and a half and fell from a metre and a
        half. The legs' own length says how far the knees go (`_crouch_of`):
        nobody typed it here. A body with no skeleton to bend goes over at
        once."""
        legs = [k for k in self._limbs(p) if "leg" in k]
        if not p.get("segs") or len(legs) < 2:
            return self._crumple_over(p, comp, sl)
        fall = float(self._crouch_of(p))
        p["_crumple"] = {"t0": self.tick, "fall": fall,
                         "n": max(1, int(round(float(np.sqrt(
                             2.0 * fall / GRAVITY)) / TICK_S)))}
        self._crumple_step(p, comp, sl)

    def _crumple_step(self, p, comp, sl):
        """One tick of the crumple. The knees buckle through the time it takes
        to fall the depth they have (the weight is let go: free fall, so the
        drop goes as the square of the time); when they have gone as far as
        they will, or the lattice will not let them, the body goes over."""
        c = p["_crumple"]
        age = self.tick - c["t0"] + 1
        if age <= c["n"]:
            how = self._crouch(p, 1.5 * (age / c["n"]) ** 2)
            if how == "moved" or age == 1:
                return
        del p["_crumple"]
        comp, sl = self._person_cells(p)
        if comp is None or not comp.any():
            return
        self._crumple_over(p, comp, sl)

    def _crumple_over(self, p, comp, sl):
        """The tip that ends a crumple: toward the openest side, about the
        edge of the feet, starting from a lean that the buckled knees give
        (about a fifth of a radian), so it takes about the time a crouched
        body takes to go over."""
        cells = self._claim_cells(p, comp, sl)
        z_lo = int(cells[:, 2].min())
        zt = z_lo + int(0.6 * (int(cells[:, 2].max()) - z_lo))    # chest height
        best, pick = -1, None
        for axis, s in ((0, 1), (0, -1), (1, 1), (1, -1)):
            edge = int(cells[:, axis].max() if s > 0 else cells[:, axis].min())
            mid = int(round(float(cells[:, 1 - axis].mean())))
            free = 0
            for k in range(1, int(cells[:, 2].max()) - z_lo + 2):
                q = [0, 0]
                q[axis], q[1 - axis] = edge + s * k, mid
                if (0 <= q[0] < self.shape[0] and 0 <= q[1] < self.shape[1]
                        and self.mat[q[0], q[1], zt] == AIR):
                    free += 1
                else:
                    break
            if free > best:
                best, pick = free, (axis, s)
        axis, s = pick
        base = cells[cells[:, 2] <= z_lo + 1]
        pivot = int(base[:, axis].max() if s > 0 else base[:, axis].min())
        self._tip(p, cells, axis, s, float(pivot), float(z_lo), 0.0, lean=0.2)

    def _ground_under(self, pose, contact):
        """The landing surface: the material under the lowest cell the body came down
        on and the depth of that same material below it (`impact.ground_of`)."""
        low = np.asarray(pose)[np.asarray(contact, bool)]
        if not len(low):
            return None
        x, y, z = (int(v) for v in low[int(np.argmin(low[:, 2]))])
        if z < 1 or not (0 <= x < self.shape[0] and 0 <= y < self.shape[1]):
            return None
        m = int(self.mat[x, y, z - 1])
        d, zz = 0, z - 1
        while zz >= 0 and int(self.mat[x, y, zz]) == m:
            d, zz = d + 1, zz - 1
        return IJ.IM.ground_of(m, d * 0.1 * self.scale)

    def _injure_landing(self, q, joules, pose, contact, shares, box):
        """A landing's real injuries (T10): the closed-form chain of impact.py
        for this person's own sheet. It writes ko_until / organ_dead /
        fractures / pool_left_ml on the person (`injury.landing_effects`) and,
        if anything of that kind happened, names the landing as the harm a
        later faint or death is blamed on."""
        if q is None or not q["alive"]:
            return
        got = IJ.landing_effects(q, joules, pose, contact, shares, self.tick,
                                 ground=self._ground_under(pose, contact))
        if got is None:
            return
        o = got["out"]
        if (o["ko"] or o["dead"] or o["hemorrhage_ml"] > 0.0) and box is not None:
            self.minds[q["pid"]].harm["wounds"] = box
            self._boxed = True

    def _joint_why(self, shock, vag, p):
        """No single cause is under the faint line by itself: name the one
        that takes the most (the lowest share of the flow), blood loss when
        the blood is as low as any."""
        perf = self._perfusion(p)
        low = min(shock, vag, perf)
        if perf <= low:
            return "blood loss"
        return "the burns" if shock <= vag else "the pain"

    def _upright_m(self, fw):
        """Height in metres of a body's claimed cells (a lying body is under
        0.8 m)."""
        return (int(fw[:, 2].max()) - int(fw[:, 2].min()) + 1) * 0.1 * self.scale

    def _withdraw(self, p, T):
        """THE PAIN WITHDRAWAL (T10, row "withdraw" of reflex.py, decision D2).
        A hand on a thing reads the interface temperature of skin and thing
        (skinburn.contact_T); heat pain (0 at 44 C, 1 at 55 C) is the
        stimulus. Past the person's own threshold, raised by sleep or
        unconsciousness and drawn as a chance (`reflex_fires`), the hand
        lets go. A chosen hold ("hold_hot") or a catch ("fall_grab") silences
        it; the dead do nothing."""
        held = p["held"]
        cell, mat = held.get("cell"), held.get("mat")
        if cell is None or mat is None or mat == FLESH:
            return
        try:
            tc = float(T[tuple(cell)])
        except IndexError:
            return
        pain = IJ.hold_pain(mat, tc)
        if pain <= 0.0:
            self._reflex_done(p, "withdraw")
            return
        if self._reflex_gate(p, "withdraw") != "go":
            return
        if not self.reflex_fires(p, "withdraw", pain):
            return
        self._reflex_done(p, "withdraw")
        label = held.get("label", "thing")
        p["held"] = None
        self._reflex_fire(p, "withdraw", "releases", result="withdrew",
                          text=f"{p['name']} snatches the hot {label} out of the hand")

    def _wound(self, p, joules):
        """Impact energy a body absorbed beyond what tissue takes whole.
        Accumulates like the burn integral and reads against the same
        thresholds; wounds do not heal here any more than burns do."""
        if p is None or not p["alive"]:
            return
        p["hurt"] = min(p["hurt"] + joules / BODY["hurt_J"], 1.0)

    def _law_life(self):
        """PHYSIOLOGY — the body's slow chemistry, every constant from BODY
        and BREATH. Blood oxygen chases the air actually around the mouth, or
        burns down when there is none (`_breathe`); the carbon monoxide the
        smoke carries builds a dose that takes the blood's oxygen away and
        clears over hours, not seconds; skin hotter than
        hurt_T cooks in, integrating toward burn shock. Consequences are
        thresholds ON STATE, never scripted events: below faint_o2 the muscles
        let go — and a slack body is just an object, so it keels over by the
        same promote/topple machinery as a chopped tree."""
        if not self.persons or self.o2 is None:
            return
        fresh = O2_PER_L * self.vox_l
        T = self.T()
        nx, ny, nz = self.shape
        for p in self.persons:
            if not p["alive"] or p["safe"]:              # out of the building is
                continue                                # out of the simulation
            flesh, sl = self._person_cells(p)            # THIS person's body only
            if flesh is None:
                flesh = np.zeros((1, 1, 1), bool)
            aloft = False
            if not flesh.any():
                # A BODY IN THE AIR LIVES. It breathes the air it is in, burns
                # in the heat it passes through and bleeds, exactly as one on
                # the floor does: the clock does not stop because he is
                # falling (a man was relaunched every third tick over a gap
                # with his physiology frozen for the whole run).
                got = self._aloft_cells(p)
                if got is not None:
                    flesh, sl = got
                    aloft = True
                else:
                    if any((b["mats"] == FLESH).any() for b in self.bodies):
                        continue                         # mid-fall: body is in the air
                    p["alive"] = False
                    self.emit("gone", actor=p["pid"],
                              text=f"{p['name']} is gone",
                              to=p, site="life:gone")
                    continue
            # WHERE THIS BODY IS, kept up to date whether or not it is awake
            # enough to ask. `_eye` was written only by the will layer, and the
            # will layer skips anyone unconscious — so a man dragged out of a
            # fire was still HEARD from where he fainted, and at standing head
            # height while he lay on the floor. Measured: dragged eighteen
            # voxels to the door and located by everyone else back in the
            # burning room. Being carried is a thing that happens TO you.
            fw = self._claim_cells(p, flesh, sl, shift=False)
            p["_eye"] = (float(fw[:, 0].mean()) + (sl[0].start or 0),
                         float(fw[:, 1].mean()) + (sl[1].start or 0),
                         float(fw[:, 2].max()))
            zs = fw[:, 2]
            z_lo, z_hi = int(zs.min()), int(zs.max())
            gas, frac, sm = self._mouth_air(p, flesh, sl, fw)
            self._breathe(p, gas, frac, sm)              # (gas False: the held breath)
            # WHAT IS HURTING IT, kept for a faint or a death to name (V3).
            # Bad air and heat are blamed on the nearest fire (`_fire_near`):
            # the smoke came from some fire, and nothing finer is known.
            harm = self.minds[p["pid"]].harm
            bad_air = sm > 0.0 or frac < BAD_AIR
            p["skin"] = float(T[sl][flesh].max())   # what the body can FEEL
            hot = np.clip(T[sl][flesh] - BODY["hurt_T"], 0.0, 400.0)
            p["burn"] = min(p["burn"] + BODY["burn_gain"] * float(hot.mean()), 1.0)
            if bad_air or hot.max() > 0.0:
                fire = self._fire_near(p["_eye"])
                if fire is not None:
                    if bad_air:
                        harm["air"] = fire
                    if hot.max() > 0.0:
                        harm["burns"] = fire
            # WHAT KNOCKS A PERSON OUT (T10, wave 13c). The old rule read
            # one number, hurt + burn, against 0.2 and 0.45: a bruise was as
            # good as a bleed. Now each is a cause with its own road:
            #   blood loss and bad air   -> the delivered oxygen (as before)
            #   burn shock               -> the same, through flow_factor
            #   the vasovagal reflex     -> the same, through flow_factor
            #   a head blow              -> ko_until (a tick)
            #   a crushed organ / brain  -> organ_dead
            # The legacy p["hurt"] / p["burn"] stay as written views: pain and
            # shock read them, nothing faints from them alone.
            upright = (self._upright_m(fw) > IJ.UPRIGHT_M)
            IJ.vagal_step(p, upright, self.tick)
            IJ.pool_step(p)
            if p.get("held"):
                self._withdraw(p, T)
            ko = self.tick < p.get("ko_until", 0)
            o2 = self._delivered_o2(p)          # what reaches the brain
            air_o2 = self._air_o2(p)            # ...had it all its blood
            # THE CORE TEMPERATURE BANDS (T7 3.11, wave 13b): a body with a heat
            # record has a `fit` per side, the share of consciousness its core
            # temperature leaves (1 above 35 C or below 39 C, falling to 0 at
            # 24 C and at 43 C). It scales what reaches the brain and feeds the
            # SAME faint / wake hysteresis as oxygen; a share of 0 is death. A
            # body with no record has no `fit` and reads exactly as before.
            fit_t = 1.0
            hrec = p.get("heat")
            if hrec and hrec.get("fit"):
                fit_c, fit_h = hrec["fit"].get("cold", 1.0), hrec["fit"].get("hot", 1.0)
                fit_t = min(fit_c, fit_h)
                o2 = o2 * fit_t
            went_out = False
            if p["awake"] and (o2 < BODY["faint_o2"] or ko):
                p["awake"] = False
                went_out = True
                shock = IJ.flow_factor({"burn": p["burn"]})
                vag = IJ.flow_factor({"vagal": p.get("vagal", 0.0)})
                temp_out = fit_t < 1.0 and air_o2 * self._perfusion(p) * IJ.flow_factor(p) >= BODY["faint_o2"]
                why = ("a blow" if ko
                       else "the burns" if shock < BODY["faint_o2"]
                       else "the pain" if vag < BODY["faint_o2"]
                       else ("the foul air" if gas else "no air")
                       if air_o2 < BODY["faint_o2"]
                       else ("the cold" if hrec["fit"].get("cold", 1.0) <= hrec["fit"].get("hot", 1.0)
                             else "the heat") if temp_out
                       else self._joint_why(shock, vag, p))
                hurt_by = harm.get(_HARM[why])
                self.emit("faints", actor=p["pid"], why=why, cause=hurt_by,
                          link="carried" if hurt_by is not None else None,
                          text=f"{p['name']} slumps — {why}",
                          to=p, site="life:faints")
            elif (not p["awake"] and o2 > BODY["wake_o2"]
                    and not ko and not p.get("organ_dead")):
                # air alone can undo what air did. Burns cannot be undone —
                # the burn integral only climbs — so someone who went down from
                # heat stays down, and someone who went down from the air comes
                # round if the air comes good. The threshold used to trip one
                # way only, so a body carried into clean air stayed unconscious
                # for ever, which made rescuing anyone pointless. And it is the
                # DOSE that decides when: short of oxygen, the blood refills in
                # tens of seconds; poisoned by carbon monoxide, it takes hours
                # for the COHb to clear (B24), not the seconds it once took.
                p["awake"] = True
                p.pop("_crumple", None)         # (the legs come back, if any)
                p["emergency"] = False          # comes round with no plan
                p["goal"], p["_path"] = None, None
                self.emit("wakes", actor=p["pid"],
                          text=f"{p['name']} comes round",
                          to=p, site="life:wakes")
            # still upright and nothing holds him: unless he fell and the
            # walls stood him back up (`_lay_down`), and he has not been moved
            # since. Moved, he is a slack body again and may fall again.
            corner = (int(fw[:, 0].min()) + (sl[0].start or 0),
                      int(fw[:, 1].min()) + (sl[1].start or 0), z_lo)
            if p.get("_crumple") and not aloft:
                self._crumple_step(p, flesh, sl)
            elif (not p["awake"] and (went_out or self.tick % 8 == 0)
                    and not aloft
                    and z_hi - z_lo > int(0.8 / (0.1 * self.scale))
                    and p.get("_propped") != corner):
                self._crumple(p, flesh, sl)
            if (o2 < BODY["death_o2"] or p.get("organ_dead")
                    or (hrec is not None and fit_t <= 0.0)):
                p["alive"] = False
                # what stopped it: the burns or the wounds past the tissue
                # threshold, else the air, else the blood that is not there
                kind = ("wounds" if p.get("organ_dead")
                        else "burns" if IJ.flow_factor({"burn": p["burn"]}) < BODY["death_o2"]
                        else "air" if air_o2 < BODY["death_o2"]
                        else ("cold" if hrec["fit"].get("cold", 1.0) <= hrec["fit"].get("hot", 1.0) else "heat")
                        if fit_t <= 0.0 and self._delivered_o2(p) >= BODY["death_o2"]
                        else "wounds")
                hurt_by = harm.get(kind)
                self.emit("dies", actor=p["pid"], cause=hurt_by,
                          link="carried" if hurt_by is not None else None,
                          text=f"{p['name']} stops breathing",
                          to=p, site="life:dies")
