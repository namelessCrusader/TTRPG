"""DAMAGE: the scalar D, the slow-overload clock, and the shatter threshold (wave 13, T9).

D is a number in [0, 0.999]. 0 is clean. It multiplies strength and stiffness
(see matrec.MatRec.at), lowers the shatter threshold, and is written by several
independent causes (impact, decay, fire, frost). Stand-alone: numpy only.

TWO KINDS OF DAMAGE, kept apart so nothing is counted twice:
  DENSE D_other   impact, decay, fire, repair. It lowers capacity in every
                  check. It lives in the lattice array `dmg` (float32).
  SPARSE creep clock y   slow overload (static fatigue). It decides WHEN a
                  loaded member fails and does not lower the capacity the
                  instant check uses. It is float64 and lives in a small dict.

`combine` below is THE ONE COPY of the combine rule. When this file is wired,
every other module imports it from here; lane X checks that no second copy
exists.

THE CREEP CLOCK (Kachanov-Rabotnov). A member at utilisation u (demand over its
NOMINAL capacity, which already holds temperature and D_other but no creep
reduction) obeys dD/dt = (u / (1 - D))^n / t0. With y = (1 - D)^(n + 1) this is
dy/dt = -(n + 1) u^n / t0, which does not depend on y. So it integrates exactly
for any u(t), path independent, and a piecewise-constant u is exact. The record
stores t_R1, the time to failure at u = 1, and t0 = t_R1 (n + 1); then
    y_new = y - u^n dt / t_R1 ,   failure at y <= 0 ,   t_fail = y t_R1 / u^n .
Wood: n = 24.7 and t_R1 = 300 s give 0.57 of the short-test strength for ten
years (the Wood Handbook duration-of-load rule) and 4,049 s at u = 0.9.
"""
import math

import numpy as np

D_MAX = 0.999              # at or above this a cell is made a piece (rubble)
ETA_IMPACT = 0.3           # (u) fraction of a blow's energy that opens crack surface
DURATION_CAP = 1.3         # (u) most a short load may raise wood's capacity
COMPRESSION_EXP = 0.5      # (u) a crack closes under compression: (1 - D)^0.5


def combine(D, dD):
    """Independent causes: 1 - (1 - D)(1 - dD), clipped to [0, 0.999].
    Each cause keeps its own survival fraction, so the order does not matter.
    A negative dD is a repair (a glued crack, a weld). Works on arrays."""
    r = 1.0 - (1.0 - np.asarray(D, float)) * (1.0 - np.asarray(dD, float))
    r = np.clip(r, 0.0, D_MAX)
    return float(r) if r.ndim == 0 else r


def mult_t(D):
    """Strength multiplier for tension, bending, shear, yield: 1 - D."""
    return 1.0 - np.asarray(D, float)


def mult_c(D):
    """Strength multiplier for compression: (1 - D)^0.5."""
    return (1.0 - np.asarray(D, float)) ** COMPRESSION_EXP


def mult_E(D):
    """Stiffness multiplier (Kachanov strain equivalence): 1 - D."""
    return 1.0 - np.asarray(D, float)


def impact_dD(e_J, gimp, face_area, eta=ETA_IMPACT):
    """Cumulative impact damage of a blow that did NOT reach the shatter
    threshold: dD = min(1, eta e / (gimp A)). A calibrated law, not Griffith's:
    for soda-lime glass the real crack energy KIc^2/E is 8 J/m2 and the table's
    gimp is 200, 25 times that; gimp is a game toughness chosen so the old
    shatter rule keeps its behaviour, and eta absorbs the gap. The energy is
    NOT taken out of the heat budget (the caller books it as heat as today)."""
    d = eta * np.asarray(e_J, float) / (np.asarray(gimp, float) * np.asarray(face_area, float))
    r = np.clip(d, 0.0, 1.0)
    return float(r) if r.ndim == 0 else r


def shatter_threshold(thr_old, D):
    """The shatter threshold lowered by the damage already there:
    thr_old (1 - D). `thr_old` is today's value (TOUGH x 1000 x area x flaw),
    computed by the caller in today's operand order and dtype. At D = 0 the
    factor is exactly 1.0 in the same dtype, so the result is BIT-IDENTICAL.
    A Python float stays a Python float; float32 stays float32."""
    if isinstance(thr_old, float) and np.ndim(D) == 0:
        return thr_old * (1.0 - float(D))
    t = np.asarray(thr_old)
    dt = t.dtype if t.dtype.kind == "f" else np.dtype(np.float64)
    one_minus = np.asarray(1.0, dt) - np.asarray(D, dt)
    r = t * one_minus
    return r if isinstance(thr_old, np.ndarray) else r[()]


def decay_D(mass_lost_fraction, rot_ratio=1.0):
    """Strength lost to decay or rust: rot_ratio times the mass lost, clipped to
    [0, 0.999]. The ratio comes from the record (wood 5.0 (u), default 1.0)."""
    r = np.clip(np.asarray(rot_ratio, float) * np.asarray(mass_lost_fraction, float), 0.0, D_MAX)
    return float(r) if r.ndim == 0 else r


# ── the creep clock ─────────────────────────────────────────────────────────
def _un(u, n):
    """u^n for u >= 0 (a member with no load does not age)."""
    return np.maximum(np.asarray(u, np.float64), 0.0) ** np.asarray(n, np.float64)


def creep_y_step(y, u, n, t_R1_s, dt):
    """y_new = y - u^n dt / t_R1. Exact for constant u over dt, float64.
    Failure is y <= 0. y starts at 1. Works on arrays (a sparse dict's values)."""
    r = np.asarray(y, np.float64) - _un(u, n) * np.asarray(dt, np.float64) / np.asarray(t_R1_s, np.float64)
    return float(r) if r.ndim == 0 else r


def creep_D(y, n):
    """Damage seen from the clock: 1 - y^(1/(n + 1)). For reporting, stiffness
    and sag. NOT for the instant check (that would count creep twice)."""
    yy = np.maximum(np.asarray(y, np.float64), 0.0)
    r = 1.0 - yy ** (1.0 / (np.asarray(n, np.float64) + 1.0))
    return float(r) if r.ndim == 0 else r


def time_to_failure(y, u, n, t_R1_s):
    """Seconds until y reaches 0 at a constant utilisation u: y t_R1 / u^n.
    Infinite for u <= 0. The slow-time scheduler schedules this instead of
    stepping."""
    un = _un(u, n)
    with np.errstate(divide="ignore"):
        r = np.asarray(y, np.float64) * np.asarray(t_R1_s, np.float64) / un
    r = np.where(un > 0.0, r, np.inf)
    return float(r) if r.ndim == 0 else r


def creep_advance(y, u, n, t_R1_s, dt):
    """One step of the clock with the failure inside the step located:
    returns (y_new, failed, t_fail) where t_fail is the time from the start of
    the step at which y reached 0 (None when it did not)."""
    t_left = time_to_failure(y, u, n, t_R1_s)
    y_new = creep_y_step(y, u, n, t_R1_s, dt)
    if y_new <= 0.0:
        return 0.0, True, float(min(t_left, dt))
    return y_new, False, None


def duration_factor(t_load_s, n, t_R1_s, cap=None):
    """Capacity gain for a load that lasts t seconds, from the same law:
    (t_R1 / t)^(1/n). Wood at 0.2 s: 1.34. `cap` (default none) clips it; the
    static check uses DURATION_CAP = 1.3 (u) for the transient part."""
    f = (np.asarray(t_R1_s, float) / np.asarray(t_load_s, float)) ** (1.0 / np.asarray(n, float))
    if cap is not None:
        f = np.minimum(f, cap)
    return float(f) if f.ndim == 0 else f


def transient_factor(t_load_s, n, t_R1_s):
    """duration_factor capped at DURATION_CAP, never below 1."""
    return float(max(1.0, duration_factor(t_load_s, n, t_R1_s, cap=DURATION_CAP)))


# ── the instant check and the clock together (the SEPARATION rule) ──────────
def instant_util(u_nominal, D_other, exponent=1.0):
    """Utilisation seen by the instant check: the nominal one divided by what
    D_other leaves of the capacity, (1 - D)^p (p = 1 for tension, bending and
    shear, 0.5 for compression). Creep is NOT in it. >= 1 is failure."""
    return np.asarray(u_nominal, float) / (1.0 - np.asarray(D_other, float)) ** exponent


def run_member(u, n, t_R1_s, dt, t_end, D_other=0.0, y0=1.0):
    """Step one member at constant nominal utilisation `u` until `t_end`.
    Returns (outcome, t) with outcome "instant" (the instant check failed at
    t = 0, because of D_other), "creep" (the clock ran out at t) or "holds"."""
    if instant_util(u, D_other) >= 1.0:
        return "instant", 0.0
    y, t = y0, 0.0
    while t < t_end:
        step = min(dt, t_end - t)
        y, failed, tf = creep_advance(y, u, n, t_R1_s, step)
        if failed:
            return "creep", t + tf
        t += step
    return "holds", t_end
