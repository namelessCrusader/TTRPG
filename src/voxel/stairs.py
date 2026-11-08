"""Stairs for authors, and what walking and falling on them costs (T11, 4.11).

Nothing here says "this is a stair": the route planner reads heights
(routing.py). This module holds

  * the residential limits as DATA (IRC R311.7) and a check for an author
  * the cell geometry of a flight at a given voxel size
  * the time a rise takes, from the power a body can climb with, and the
    integer step-cost table the planner prices edges with
  * the METABOLIC work of a climb or a descent (one law per regime, never
    summed with another), and the heat it makes
  * `tumble()`, a stand-in for a body falling down a flight until the friction
    and rolling of wave 15 exist, closed by two independent energy sums

Cell edge d = voxel_cm / 100 metres, so every length here is derived from it.
"(u)" = a number I could not verify; "(v)" = checked.
"""
import math

# The body-heat topic owns the efficiency and the gravity constant. Imported
# when it is there; the copy is the same number and a cross test checks it.
try:
    from .energetics import ETA as _ETA, ECC_DIV as _ECC, GRAV as _GRAV
except ImportError:                       # pragma: no cover - phase A stand-in
    _ETA, _ECC, _GRAV = 0.22, 1.2, 9.81

ETA_UP = _ETA                            # positive-work efficiency (u: Margaria-like)
ECC_DIV = _ECC                           # lowering costs |W| / 1.2 (T7, eta -1.2)
GRAV = _GRAV
C_LEVEL = 2.5                            # net J per (kg m) of level walking (u)
V_WALK = 1.4                             # m/s on the flat
P_STAIR = 250.0                          # W mechanical, for up to a minute (u)
P_SUSTAINED = 150.0                      # W after a minute: T10's critical power (u)
STEPS_PER_S = 1.8                        # the unhurried stair cadence (u)
STAIR_V_X = STEPS_PER_S * 0.254          # horizontal speed on stairs, m/s (v)

# IRC 2018 R311.7, in millimetres (v). Headroom 80 in = 2032 mm (v1 said 2036).
IRC = {"riser_max_mm": 196.85,           # 7.75 in
       "tread_min_mm": 254.0,            # 10 in
       "headroom_min_mm": 2032.0,        # 6 ft 8 in
       "uniform_mm": 9.5}                # 3/8 in between the largest and smallest
BLONDEL_MM = (600.0, 650.0)              # 2R + T, a comfort check (u)

# What becomes of the potential a descent loses: the share that goes to the
# feet and the rest to the floor under them (data, u). The sum is 1.
DESCENT_HEAT_SPLIT = {"feet": 0.5, "floor": 0.5}


# ── the geometry an author wants ────────────────────────────────────────────
def irc_check(riser_mm, tread_mm, headroom_mm=None, slack_mm=0.0):
    """The limits a flight breaks, as a list of words ([] = none). `slack_mm`
    is the rounding to whole cells the lattice forces (half a cell is the
    honest value), so a 4 x 5 flight at 5 cm (200 / 250 mm) is not flagged."""
    out = []
    if riser_mm > IRC["riser_max_mm"] + slack_mm:
        out.append("riser too high")
    if tread_mm < IRC["tread_min_mm"] - slack_mm:
        out.append("tread too short")
    if headroom_mm is not None and headroom_mm < IRC["headroom_min_mm"] - slack_mm:
        out.append("headroom too low")
    return out


def stair_cells(voxel_cm, riser_cells=4, tread_cells=5, n_steps=12,
                body_cells=None):
    """The cell geometry of a flight at this voxel size.

    Returns a dict: d_mm (cell edge), riser_mm, tread_mm, angle_deg, rise_cells
    (total), run_cells, headroom_cells (what a flight needs overhead: IRC 2032
    mm rounded UP), blondel_mm (2R + T), issues (irc_check with half a cell of
    rounding allowed), and steps: [(x_cell, standing_z_offset), ...] for an
    author: step i covers run cells [i * tread, (i + 1) * tread) and the floor
    to stand on rises (i + 1) * riser cells above the bottom landing.
    `body_cells` (the body's height in cells) is only echoed to say whether the
    headroom fits it."""
    d_mm = float(voxel_cm) * 10.0
    riser_mm = riser_cells * d_mm
    tread_mm = tread_cells * d_mm
    head = math.ceil(IRC["headroom_min_mm"] / d_mm)
    return {
        "d_mm": d_mm, "riser_mm": riser_mm, "tread_mm": tread_mm,
        "angle_deg": math.degrees(math.atan2(riser_mm, tread_mm)),
        "rise_cells": riser_cells * n_steps, "run_cells": tread_cells * n_steps,
        "headroom_cells": head,
        "blondel_mm": 2 * riser_mm + tread_mm,
        "issues": irc_check(riser_mm, tread_mm, head * d_mm, slack_mm=d_mm / 2.0),
        "body_fits": None if body_cells is None else body_cells <= head,
        "steps": [(i * tread_cells, (i + 1) * riser_cells) for i in range(n_steps)],
    }


# ── time ────────────────────────────────────────────────────────────────────
def climb_speed(mass_kg, strength_N=400.0, power_w=P_STAIR):
    """The fastest a body can rise, m/s: P / (m g), P scaled by (strength / 400).
    """
    return power_w * (strength_N / 400.0) / (mass_kg * GRAV)


def step_time(dz_m, d_m, mass_kg=70.0, strength_N=400.0, power_w=P_STAIR,
              v_walk=V_WALK):
    """Seconds for a move of one cell (d_m) with a rise of |dz_m|:
    max(d / v_walk, h / vz_max). A 20 cm riser at 250 W costs 0.55 s (v)."""
    return max(d_m / v_walk,
               abs(dz_m) / climb_speed(mass_kg, strength_N, power_w))


def unhurried_power(mass_kg=70.0, riser_m=0.20, per_s=STEPS_PER_S):
    """The mechanical power of the default stair cadence: steps/s x riser x
    m g. 1.8 x 0.20 x 70 x 9.81 = 247 W, which the 250 W cap agrees with."""
    return per_s * riser_m * mass_kg * GRAV


def step_table(voxel_cm, r_max=6, mass_kg=70.0, strength_N=400.0,
               power_w=P_STAIR, v_walk=V_WALK):
    """The integer step-cost table F[0 .. r_max], built once per voxel size.
    F[0] = 8 exactly (a flat edge costs today's 70 or 99); F[n] =
    round(8 t(n) / t(0)) from the time of a move with a rise of n cells, never
    below 8. The planner prices an edge c * F[|dz|] // 8."""
    d = voxel_cm / 100.0
    t0 = d / v_walk
    out = [8]
    for n in range(1, int(r_max) + 1):
        t = step_time(n * d, d, mass_kg, strength_N, power_w, v_walk)
        out.append(max(8, int(round(8.0 * t / t0))))
    return out


# ── work ────────────────────────────────────────────────────────────────────
def climb_work(mass_kg, dz_m, run_m, eta=ETA_UP, c_level=C_LEVEL):
    """The METABOLIC joules of going UP dz_m over run_m of level ground, net of
    rest (the body-heat lane already burns the basal rate).

        up:  E = m g dz / eta + C_level m run

    One law per regime: the climb is not also counted as Margaria plus a level
    cost plus Minetti (v1 counted it three times). 70 kg x 3 m over 3.75 m:
    mechanical 2.06 kJ, climbing part 9.36 kJ, level part 0.66 kJ, total 10.0
    kJ (v). Returns a dict with mech_J (m g dz, the potential gained, which stays
    OFF the ledger), vertical_J, level_J, metabolic_J and heat_J (what is
    made in the body: metabolic - mech)."""
    mech = mass_kg * GRAV * abs(dz_m)
    vert = mech / eta
    lev = c_level * mass_kg * abs(run_m)
    return {"mech_J": mech, "vertical_J": vert, "level_J": lev,
            "metabolic_J": vert + lev, "heat_J": vert + lev - mech}


def descent_work(mass_kg, dz_m, run_m, ecc_div=ECC_DIV, c_level=C_LEVEL):
    """The metabolic joules of going DOWN. The muscles absorb m g |dz|:

        down: metabolic = m g |dz| / 1.2 + C_level m run

    and the HEAT in the body is the potential lost plus the metabolic part:
    m g |dz| + metabolic. (The vertical part alone is 3.78 kJ for 70 kg x 3 m,
    v.) The potential is lost, so it is booked as heat: `fall_heat_J` is the
    joule count a caller deposits through the world's one deposit()."""
    mech = mass_kg * GRAV * abs(dz_m)
    vert = mech / ecc_div
    lev = c_level * mass_kg * abs(run_m)
    return {"mech_J": mech, "vertical_J": vert, "level_J": lev,
            "metabolic_J": vert + lev, "heat_J": mech + vert + lev,
            "fall_heat_J": mech, "heat_vertical_J": mech + vert}


def step_work(mass_kg, dz_m, run_m, **kw):
    """Climb or descent by the sign of dz_m (a flat step is the level part)."""
    return climb_work(mass_kg, dz_m, run_m, **kw) if dz_m >= 0 \
        else descent_work(mass_kg, dz_m, run_m, **kw)


# ── falling down a flight (a stand-in until wave 15: rolling, friction) ─────
def tumble(mass_kg, vx0, riser_m, tread_m, n_steps, e_n=0.3, mu=0.4,
           vz_stick=0.25, max_hops=2000):
    """A point mass that LANDS on a flight with tangential speed vx0 (m/s),
    bounces tread by tread to the bottom landing, and what that cost.

    FIRES only on a landing with tangential speed on a flight: a body standing
    on stairs, or arriving on a flat landing (n_steps == 0), does not tumble.
    The mass starts at the nosing (front edge) of the top tread, tread 0, with
    vertical speed 0 and tangential speed vx0; tread i is at -i R below it. Between
    impacts it flies under gravity (and may skip treads if vx t exceeds a
    tread). At each impact the vertical speed is reversed with the normal
    restitution e_n, and Coulomb friction acts on the horizontal motion:

        vx' = max(0, vx - mu (1 + e_n) vz)

    A bounce too small to leave the tread (< vz_stick) becomes a slide along it
    under mu g, which can stop the body. It ends at the first arrival on the
    bottom landing (that arrival is the landing event the body-injury law
    takes), or when it comes to rest on a tread (vx = 0 and no bounce left).

    THE ENERGY IS CLOSED TWICE, by calculations that share no line:
      speed_loss_J   per impact the kinetic energy before minus after, from the
                     speeds
      formula_loss_J per impact 1/2 m vz^2 (1 - e^2) + the friction impulse
                     (m (vx - vx')) times the mean tangential speed, and a
                     slide's mu m g L
    and  m g H + KE0 = sum of losses + KE_end  (H the height descended).

    Returns a dict: impacts (list of per-impact dicts), n_impacts, ke0, ke_end,
    H_m, speed_loss_J, formula_loss_J, exit (vx, vz) at the bottom or None,
    stopped_on (tread index) or None, tumbled (bool)."""
    g = GRAV
    m = float(mass_kg)
    ke0 = 0.5 * m * vx0 * vx0
    if n_steps <= 0 or vx0 <= 0.0:
        return {"impacts": [], "n_impacts": 0, "ke0": ke0, "ke_end": ke0,
                "H_m": 0.0, "speed_loss_J": 0.0, "formula_loss_J": 0.0,
                "exit": (vx0, 0.0), "stopped_on": None, "tumbled": False}
    R, T = float(riser_m), float(tread_m)
    n = int(n_steps)
    x, y, vx, vz = T, 0.0, float(vx0), 0.0          # at the nosing of the top tread; vz up
    tread = 0
    impacts = []
    speed_sum = form_sum = 0.0
    exit_v, stopped = None, None
    for _ in range(max_hops):
        # fly from (x, y, vx, vz) until the first tread (or the landing) is hit
        best_t, best_i = None, None
        for i in range(tread, n + 1):
            top = -i * R
            # y(t) = y + vz t - g t^2 / 2 = top, the descending root
            disc = vz * vz + 2.0 * g * (y - top)
            if disc < 0.0:
                continue
            t = (vz + math.sqrt(disc)) / g
            if t <= 1e-12 and not (i == tread and vz > 0.0):
                continue
            xi = x + vx * t
            lo = i * T
            hi = (i + 1) * T if i < n else float("inf")
            if lo - 1e-12 <= xi < hi:
                if best_t is None or t < best_t:
                    best_t, best_i = t, i
        if best_t is None:                            # (cannot happen; guard)
            stopped = tread
            break
        t, i = best_t, best_i
        x, y = x + vx * t, -i * R
        vz = vz - g * t                                # negative: going down
        tread = i
        if i == n:                                     # the bottom landing
            exit_v = (vx, vz)
            break
        vzi = -vz                                      # impact speed, positive
        vx_out = max(0.0, vx - mu * (1.0 + e_n) * vzi)
        vz_out = e_n * vzi
        ke_b = 0.5 * m * (vx * vx + vzi * vzi)
        ke_a = 0.5 * m * (vx_out * vx_out + vz_out * vz_out)
        sp = ke_b - ke_a
        fo = (0.5 * m * vzi * vzi * (1.0 - e_n * e_n)
              + m * (vx - vx_out) * (vx + vx_out) / 2.0)
        impacts.append({"tread": i, "vz_in": vzi, "vx_in": vx, "vx_out": vx_out,
                        "vz_out": vz_out, "speed_loss_J": sp, "formula_loss_J": fo})
        speed_sum += sp
        form_sum += fo
        vx, vz = vx_out, vz_out
        if vz < vz_stick:                              # too small to hop: slide
            if vz > 0.0:                               # the small bounce is lost
                impacts.append({"tread": tread, "settle": True, "vz_in": vz,
                                "vx_in": vx, "vx_out": vx, "vz_out": 0.0,
                                "speed_loss_J": 0.5 * m * vz * vz,
                                "formula_loss_J": m * g * (vz * vz / (2.0 * g))})
                speed_sum += 0.5 * m * vz * vz
                form_sum += m * g * (vz * vz / (2.0 * g))
            vz = 0.0
            if vx <= 0.0:
                stopped = tread
                break
            room = (tread + 1) * T - x
            v2 = vx * vx - 2.0 * mu * g * room
            if v2 <= 0.0:                              # stops on this tread
                sp = 0.5 * m * vx * vx
                fo = mu * m * g * (vx * vx / (2.0 * mu * g))
                impacts.append({"tread": tread, "slide": True, "vz_in": 0.0,
                                "vx_in": vx, "vx_out": 0.0, "vz_out": 0.0,
                                "speed_loss_J": sp, "formula_loss_J": fo})
                speed_sum += sp
                form_sum += fo
                vx = 0.0
                stopped = tread
                break
            v_out = math.sqrt(v2)
            sp = 0.5 * m * (vx * vx - v_out * v_out)
            fo = mu * m * g * room
            impacts.append({"tread": tread, "slide": True, "vz_in": 0.0,
                            "vx_in": vx, "vx_out": v_out, "vz_out": 0.0,
                            "speed_loss_J": sp, "formula_loss_J": fo})
            speed_sum += sp
            form_sum += fo
            vx = v_out
            x = (tread + 1) * T                        # leaves the edge, falls on
            # the next flight starts at the tread's edge: height unchanged
    reached = tread if stopped is not None else n
    H = reached * R
    if exit_v is not None:
        ke_end = 0.5 * m * (exit_v[0] ** 2 + exit_v[1] ** 2)
    else:
        ke_end = 0.5 * m * (vx * vx + vz * vz)
    return {"impacts": impacts, "n_impacts": len(impacts), "ke0": ke0,
            "ke_end": ke_end, "H_m": H, "speed_loss_J": speed_sum,
            "formula_loss_J": form_sum, "exit": exit_v, "stopped_on": stopped,
            "tumbled": True}
