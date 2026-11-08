"""VASCULATURE: vessels, pressure, clot and pooling (wave 13, T10, lane A-T10b).

(The topic spec calls this "vessels"; tests/test_vessels.py is about
containers, so the file is vasculature.)

State: V = circulating blood (the existing blood_ml), V0 = 70 ml/kg x mass,
f = V / V0. Pressure falls as blood is lost:
    MAP_rel(f) = clip((f - 0.25) / (0.80 - 0.25), 0, 1), 0 when dead.
    MAP_0 = 12 kPa (90 mmHg), venous P_v = 1.3 kPa (u).
Perfusion to the brain keeps the existing numbers, clip((f - 0.45)/0.55)^0.56,
so the faint (36 % lost) and death (50.4 % lost) lines do not move.

A CUT VESSEL (a wound that reaches the vessel's depth across its azimuth):
    P_vessel = MAP_0 MAP_rel (1 + 0.08 (met - 1)) + P_hydro      artery
             = P_v MAP_rel + P_hydro                              vein
    P_hydro  = rho_b g (z_heart - z_wound), 10.4 kPa a metre, scaled by
               min(1, MAP_rel / 0.3) so an empty circulation has no column
    P_drive  = max(P_vessel - P_ext, 0)
    Q = Q_full g(f_open) (P_drive / P_ref) (d_eff / d)^2,  g(x) = x / (x + 0.5 (1 - x))
  Spasm: a muscular artery of d <= 4 mm narrows,
    d_eff = d sqrt(1 - 0.8 (1 - exp(-t / 30 s)))  -- and the narrower open
    width feeds the clot hold, not only the flow.
  Clot: the plug holds a pressure that grows with time and falls with the open
  width,  P_hold = 14 kPa (1 mm / d_open) (1 - exp(-t / 180 s)) x
  clip(2^((T_body - 37) / 10), 0.1, 1),  d_open = f_open d_eff.
  The cut SEALS when P_drive <= P_hold; if it later has P_drive > P_hold it
  REOPENS and the clot starts again.
  P_ext is the largest of a press (hand force over contact area, attenuated by
  depth), a bandage's hoop pressure T / (r w), tissue tamponade, and a
  tourniquet's cuff pressure for wounds DISTAL to it (an artery is occluded
  only when the cuff reaches AOP = SBP (1 + 0.1 circumference / width)).

THE OLD CAPILLARY LAW STAYS in physiology (400 ml/s per m2 of cut plane, clot
1.7e-6 m2 a second). Nothing here replaces it: these flows are ADDED for
wounds that cross a vessel.

CLOSED BLEEDING (fractures, organs) is pressure driven and saturates:
    d(pooled)/dt = (V_cavity - pooled) MAP_rel / 1800 s
the cavity by region (rib 125 ml ... pelvis 2500 ml), clamped to the blood
there is, zero when dead, slowed by a splint's tamponade.

ISCHAEMIA: dead tissue fraction under a tourniquet = 1 - exp(-(t / 8 h)^2)
(2 h gives 6.1 %).

Pure functions on plain dicts and floats. No World. Numbers marked (u) are
unverified.
"""
import math

try:
    from . import anatomy as _AN
except ImportError:                    # pragma: no cover
    _AN = None
try:
    from .matdamage import combine as _combine
except ImportError:                    # pragma: no cover - local copy (lane X compares them)
    def _combine(D, dD):
        r = 1.0 - (1.0 - D) * (1.0 - dD)
        return min(max(r, 0.0), 0.999)

MAP_0 = 12.0e3                 # Pa, 90 mmHg
P_VENOUS = 1.3e3               # Pa (u)
MAP_F_LO, MAP_F_HI = 0.25, 0.80
RHO_B_G = 10.4e3               # Pa per metre of blood column (rho_b g)
MET_PRESSURE = 0.08            # arterial pressure rises 8 % per MET above rest (u)
SPASM_D_MM = 4.0               # arteries up to this narrow
SPASM_DEPTH = 0.8              # at most to (1 - 0.8) of the area
SPASM_TAU_S = 30.0
HOLD_PA_MM = 14.0e3            # P_hold x d_open at full maturity
CLOT_TAU_S = 180.0
PRESS_DEPTH_M = 0.04           # a press fades with depth: exp(-depth / 0.04 m)
TQ_AOP_K = 0.1                 # AOP = SBP (1 + 0.1 circumference / width)
ISCH_SCALE_S = 8.0 * 3600.0
POOL_TAU_S = 1800.0
SPLINT_TAMPONADE = 0.5         # a full splint slows pooling by half (u)
ML_PER_KG = 70.0               # BLEED["ml_per_kg"]
PERF_F0, PERF_K = 0.45, 0.56   # BLEED["f0"], BLEED["k"]
FAINT_O2, DEATH_O2 = 0.55, 0.25  # BODY["faint_o2"], BODY["death_o2"]
Q_REF_KG = 70.0


# ── pressure and perfusion ───────────────────────────────────────────────────
def map_rel(f, alive=True):
    """Relative mean arterial pressure: 1 at f >= 0.8, 0 at f <= 0.25, 0 dead."""
    if not alive:
        return 0.0
    x = (f - MAP_F_LO) / (MAP_F_HI - MAP_F_LO)
    return min(max(x, 0.0), 1.0)


def perfusion(f):
    """How much of the blood's oxygen reaches the brain (0..1): the existing
    clip((f - 0.45) / 0.55) ^ 0.56."""
    x = (f - PERF_F0) / (1.0 - PERF_F0)
    return min(max(x, 0.0), 1.0) ** PERF_K


def f_at_perfusion(p):
    """The blood fraction at which perfusion equals p (the inverse)."""
    return PERF_F0 + (1.0 - PERF_F0) * p ** (1.0 / PERF_K)


def faint_loss_frac():
    """Fraction of the blood lost at the faint line (perfusion < 0.55): 36 %."""
    return 1.0 - f_at_perfusion(FAINT_O2)


def death_loss_frac():
    """Fraction lost at the death line (perfusion < 0.25): 50.4 %."""
    return 1.0 - f_at_perfusion(DEATH_O2)


def blood0_ml(mass_kg):
    """V0 = 70 ml/kg x mass."""
    return ML_PER_KG * mass_kg


def hydro_pa(z_rel_m, mrel):
    """The hydrostatic term: 10.4 kPa a metre of height of the heart above the
    wound (negative for a wound above the heart), scaled by min(1, MAP_rel/0.3)
    so an empty circulation has no column."""
    return RHO_B_G * z_rel_m * min(1.0, mrel / 0.3)


# ── what a wound crosses ─────────────────────────────────────────────────────
def limb_thickness_m(region, sheet):
    """The person's limb thickness in this region (m): the adult table scaled by
    the cube root of mass. A wound dict that carries "d_limb_m" overrides it."""
    reg = region.split(" ", 1)[1] if region[:4] in ("left", "righ") else region
    return _AN.D_ADULT_MM[reg] * 1e-3 * (sheet["mass_kg"] / Q_REF_KG) ** (1.0 / 3.0)


def _arc_overlap(c1, hw1, c2, hw2):
    """Length (degrees) of the overlap of two arcs on a circle, each given as
    centre and half-width."""
    d = abs((c1 - c2 + 180.0) % 360.0 - 180.0)          # centre distance on the circle
    lo = max(-hw1, d - hw2)
    hi = min(hw1, d + hw2)
    return max(0.0, hi - lo)


def crosses(wound, region, sheet):
    """The cuts a wound makes: a list of dicts, one per vessel it reaches.

    The wound has depth_m, length_m, az (degrees in the bone's own frame), along
    (0-1 from the joint). Its arc = min(360, length / (pi D) x 360) degrees
    centred on az (an edge across the limb, (u); a lengthwise slash is
    under-modelled). The vessel's depth is a FRACTION of the limb's thickness
    (so the stock 5 cm limb has its vessels 5 cm deep at most), its width and
    Q_full scale with the person (cube root and 2/3 power of mass).
        f_depth = clip((depth - (vessel_depth - d/2)) / d, 0, 1)
        f_arc   = overlap of the arcs over the vessel's width
        f_open  = f_depth f_arc
    A blunt blow (no depth) cuts nothing. Each cut is
    {vessel, kind, region, f_open, d_m, Q_full, along, seal_s, sealed, t}."""
    out = []
    reg = region.split(" ", 1)[1] if region[:4] in ("left", "righ") else region
    depth = float(wound.get("depth_m", 0.0))
    if depth <= 0.0:
        return out
    D = float(wound.get("d_limb_m") or limb_thickness_m(region, sheet))
    arc = min(360.0, float(wound.get("length_m", 0.0)) / (math.pi * D) * 360.0)
    az = float(wound.get("az", 0.0))
    size = (sheet["mass_kg"] / Q_REF_KG)
    for v in _AN.VESSELS:
        if v["region"] != reg:
            continue
        d_m = v["d_mm"] * 1e-3 * size ** (1.0 / 3.0)
        v_depth = v["depth_mm"] / _AN.D_ADULT_MM[reg] * D
        f_depth = min(max((depth - (v_depth - 0.5 * d_m)) / d_m, 0.0), 1.0)
        if f_depth <= 0.0:
            continue
        width = 2.0 * v["hw"]
        ov = _arc_overlap(az, 0.5 * arc, v["az"], v["hw"])
        f_arc = min(1.0, ov / width) if width > 0.0 else 0.0
        f_open = f_depth * f_arc
        if f_open <= 0.0:
            continue
        out.append(dict(vessel=v["name"], kind=v["kind"], region=region, f_open=f_open,
                        d_m=d_m, Q_full=v["Q_ml_s"] * size ** (2.0 / 3.0),
                        along=float(wound.get("along", 0.5)),
                        seal_s=0.0, sealed=False, t=0.0))
    return out


def tendon_cut(wound, region, sheet):
    """Fraction of the flexor tendons a forearm wound cuts (0-1): the depth over
    the tendons' band (8-25 mm of the adult forearm) times the arc over their
    width. Not read from "open"."""
    reg = region.split(" ", 1)[1] if region[:4] in ("left", "righ") else region
    if reg != "forearm":
        return 0.0
    depth = float(wound.get("depth_m", 0.0))
    if depth <= 0.0:
        return 0.0
    t = _AN.TENDONS[0]
    D = float(wound.get("d_limb_m") or limb_thickness_m(region, sheet))
    scale = D / (_AN.D_ADULT_MM[reg] * 1e-3)
    lo, hi = t["depth_mm_lo"] * 1e-3 * scale, t["depth_mm_hi"] * 1e-3 * scale
    f_depth = min(max((depth - lo) / (hi - lo), 0.0), 1.0)
    arc = min(360.0, float(wound.get("length_m", 0.0)) / (math.pi * D) * 360.0)
    ov = _arc_overlap(float(wound.get("az", 0.0)), 0.5 * arc, t["az"], t["hw"])
    return f_depth * min(1.0, ov / (2.0 * t["hw"]))


def internal_cut(organ, frac, sheet):
    """A cut for an internal organ (liver, spleen, lung, aorta, heart) from an
    impact: the same dict as `crosses` gives, full pressure inside, no depth to
    cross. It pools into the cavity rather than pouring: the caller decides."""
    o = _AN.ORGANS[organ]
    size = sheet["mass_kg"] / Q_REF_KG
    return dict(vessel=organ, kind=o["kind"], region=o["region"], f_open=min(max(frac, 0.0), 1.0),
                d_m=o["d_mm"] * 1e-3 * size ** (1.0 / 3.0),
                Q_full=o["Q_ml_s"] * size ** (2.0 / 3.0), along=0.5,
                seal_s=0.0, sealed=False, t=0.0, internal=True)


# ── the pressures the acts write ─────────────────────────────────────────────
def press_pa(F_N, area_m2, depth_m=0.0):
    """A hand on a wound: force over contact area, attenuated by the depth of the
    vessel, P = (F / A) exp(-depth / 0.04 m). 100 N on 4 cm2 is 250 kPa at the
    skin."""
    return (F_N / area_m2) * math.exp(-depth_m / PRESS_DEPTH_M)


def hoop_pa(T_N, r_m, w_m):
    """A bandage's hoop pressure P = T / (r w) (T the strap tension, r the limb
    radius, w the band width)."""
    return T_N / (r_m * w_m)


def aop_pa(sbp, circ_m, width_m):
    """Arterial occlusion pressure of a cuff: SBP (1 + 0.1 circumference / width)
    (a fit, u). A 4 cm strap on a 55 cm thigh needs 2.4 x SBP; a 10 cm band 1.5."""
    return sbp * (1.0 + TQ_AOP_K * circ_m / width_m)


def tourniquet_pext(cut, tq, sbp=MAP_0 * 1.33, circ_m=0.4):
    """The pressure a tourniquet puts on a cut: only for a wound DISTAL to the
    cuff (cut["along"] beyond tq["along"], measured from the joint). An artery
    is occluded only when the cuff reaches AOP; a vein is closed by any cuff
    above the venous pressure. tq = {pressure_Pa, width_m, along}."""
    if cut.get("along", 0.5) <= tq.get("along", 0.5):
        return 0.0
    p = float(tq.get("pressure_Pa", 0.0))
    if cut["kind"] == "a":
        return p if p >= aop_pa(sbp, circ_m, tq.get("width_m", 0.04)) else 0.0
    return p


def isch_dead_frac(t_s):
    """Tissue dead under a tourniquet after t seconds: 1 - exp(-(t / 8 h)^2).
    2 h gives 6.1 %."""
    return 1.0 - math.exp(-(t_s / ISCH_SCALE_S) ** 2)


def isch_dead_combine(D, t_s):
    """Fold the ischaemic death into a damage number with the one combine rule."""
    return _combine(D, isch_dead_frac(t_s))


# ── flow through a cut vessel ────────────────────────────────────────────────
def g_open(x):
    """The open-fraction shape of the flow, g(x) = x / (x + 0.5 (1 - x)): 0 shut,
    1 for a vessel cut right through."""
    return x / (x + 0.5 * (1.0 - x))


def d_eff_ratio(cut, t):
    """d_eff / d: the spasm of a muscular artery (d <= 4 mm) after t seconds,
    sqrt(1 - 0.8 (1 - exp(-t / 30))). Veins and big arteries do not narrow."""
    if cut["kind"] != "a" or cut["d_m"] * 1e3 > SPASM_D_MM:
        return 1.0
    return math.sqrt(1.0 - SPASM_DEPTH * (1.0 - math.exp(-t / SPASM_TAU_S)))


def temp_factor(T_body):
    """The clot is slower in the cold: clip(2^((T - 37) / 10), 0.1, 1)."""
    return min(max(2.0 ** ((T_body - 37.0) / 10.0), 0.1), 1.0)


def hold_pa(cut, t_clot, T_body=37.0):
    """The pressure the platelet plug holds at clot age t_clot (seconds):
    14 kPa (1 mm / d_open)(1 - exp(-t / 180)) x temperature factor,
    d_open = f_open d_eff. Zero at t = 0; falls as the open width grows."""
    d_open = cut["f_open"] * cut["d_m"] * d_eff_ratio(cut, cut.get("t", t_clot))
    if d_open <= 0.0:
        return math.inf
    return (HOLD_PA_MM * 1e-3 / d_open) * (1.0 - math.exp(-t_clot / CLOT_TAU_S)) * temp_factor(T_body)


def vessel_pressure(cut, f, alive=True, met=1.0, z_rel_m=0.0):
    """P_vessel inside the cut vessel: arterial MAP_0 MAP_rel (1 + 0.08 (met-1)),
    venous P_v MAP_rel, plus the hydrostatic term."""
    mr = map_rel(f, alive)
    if cut["kind"] == "a":
        base = MAP_0 * mr * (1.0 + MET_PRESSURE * (met - 1.0))
    else:
        base = P_VENOUS * mr
    return base + hydro_pa(z_rel_m, mr)


def flow_ml_s(cut, f, p_ext=0.0, alive=True, met=1.0, z_rel_m=0.0):
    """Q of an open cut at this moment (ml/s), ignoring the clot: Q_full g(f_open)
    (P_drive / P_ref) (d_eff/d)^2. The clot is `step_wound`'s."""
    P = vessel_pressure(cut, f, alive, met, z_rel_m)
    drive = max(P - p_ext, 0.0)
    ref = MAP_0 if cut["kind"] == "a" else P_VENOUS
    r = d_eff_ratio(cut, cut["t"])
    return cut["Q_full"] * g_open(cut["f_open"]) * (drive / ref) * r * r


def step_wound(cuts, V, V0, P_ext_by_cut, met, dt, T_body, z_rel_m, alive=True):
    """Advance the cuts of one wound by dt seconds; returns ml lost (at most V).

    Each cut dict is mutated: `t` (age, seconds), `seal_s` (the clot's age; it
    restarts when the cut reopens), `sealed`, `flow` (ml/s used). The flow of
    every cut uses the blood fraction at the start of the step, minus what the
    cuts before it took, and the total is clamped to V so the blood ledger
    stays exact. `P_ext_by_cut` is a list (or a dict by vessel name) of external
    pressures, Pa. `z_rel_m` is the height of the heart above the wound
    (positive: wound below the heart, as standing at the ankle); a cut may carry
    its own "z_rel_m". A dead body (alive False) bleeds nothing."""
    lost = 0.0
    V_run = V
    for i, cut in enumerate(cuts):
        pe = (P_ext_by_cut.get(cut["vessel"], 0.0) if isinstance(P_ext_by_cut, dict)
              else (P_ext_by_cut[i] if i < len(P_ext_by_cut) else 0.0))
        z = cut.get("z_rel_m", z_rel_m)
        f = V_run / V0 if V0 > 0.0 else 0.0
        P = vessel_pressure(cut, f, alive, met, z)
        drive = max(P - pe, 0.0)
        hold = hold_pa(cut, cut["seal_s"], T_body)
        sealed_now = drive <= hold
        if sealed_now:
            cut["sealed"] = True
            cut["flow"] = 0.0
        else:
            if cut["sealed"]:                    # reopened: the plug is lost
                cut["sealed"] = False
                cut["seal_s"] = 0.0
            q = flow_ml_s(cut, f, pe, alive, met, z)
            ml = min(q * dt, max(V_run, 0.0))
            cut["flow"] = ml / dt if dt > 0.0 else 0.0
            lost += ml
            V_run -= ml
        cut["t"] += dt
        cut["seal_s"] += dt
    return lost


# ── closed bleeding ──────────────────────────────────────────────────────────
def cavity_ml(region, sheet=None):
    """The cavity a closed bleed fills at most, ml: the ATLS midpoints by region
    (rib 125, forearm 400, humerus 600, shin 750, thigh 1500, pelvis 2500),
    scaled by mass over 70 kg."""
    base = _AN.CAVITY_ML[region]
    return base * (sheet["mass_kg"] / Q_REF_KG if sheet else 1.0)


def pool_step(region, V_pooled, f, dt, alive=True, splint=0.0, V_ml=None, sheet=None):
    """Blood that leaves circulation into a closed cavity during dt seconds, ml.
    d(pooled)/dt = (V_cavity - pooled) MAP_rel / 1800 s, solved exactly for the
    step; saturates at the cavity, zero when MAP_rel is 0 or dead, slowed by a
    splint's tamponade, and clamped to the blood there is (`V_ml`) so the
    ledger stays exact."""
    cap = cavity_ml(region, sheet)
    room = max(cap - V_pooled, 0.0)
    mr = map_rel(f, alive)
    if mr <= 0.0 or room <= 0.0:
        return 0.0
    rate = mr * (1.0 - SPLINT_TAMPONADE * min(max(splint, 0.0), 1.0)) / POOL_TAU_S
    ml = room * (1.0 - math.exp(-rate * dt))
    if V_ml is not None:
        ml = min(ml, max(V_ml, 0.0))
    return ml
