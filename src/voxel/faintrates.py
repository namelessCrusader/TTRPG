"""FAINT RATES AGAINST SOURCES (wave 14b, lane H). Pure: read-only checks of
what knocks a person out, set beside the real figures. Nothing here runs in a
tick; the laws are in physiology.py (`_law_life`) and injury.py, and
tests/test_faint_rates.py holds these numbers to the sources.

Each cause has a row: what the sim says, what the source says, and how they sit.
(u) marks a figure the lane could not check against the paper itself.

blood loss  ATLS classes: 15-30 percent lost is mild (the heart compensates),
            30-40 percent is hypotension and confusion (class III), over 40
            percent is life-threatening (class IV). The sim faints at 36 and dies
            at 50 percent lost (vasculature.faint_loss_frac). Inside the bands.
vasovagal   Fainting from any cause is common: about one person in three over a
            lifetime (Ganzeboom 2006, 35 percent by age 60 (u); Kapoor 2000).
            Pain and fear are a minority of the triggers (about 40 percent, u),
            so about 12 percent of people have fainted from a strong pain or fright.
            The sim's per-person dial (injury.vagal_threshold, log-normal, median
            0.6, sigma 0.6) puts the share who faint at a pain of 0.3 (750 J
            absorbed in one blow) at 12 percent, and the share at 0.134, the
            clip, at 0.5 percent: most need a lot, a few faint easily.
hypoxia     Time of useful consciousness (FAA / Ernsting): about 15 s at 40000 ft,
            45 s at 35000 ft, 90 s at 30000 ft, 4 min at 25000 ft, 10 min at
            22000 ft, 30 min at 18000 ft. The sim's blood follows the air with a
            40 s time constant toward the oxygen FRACTION, a straight line, which
            fainted people 20 times too early at 18000 ft (74 s against 25 min).
            Now it follows the haemoglobin curve (`o2_target`): 28000, 30000 and
            35000 ft are inside the FAA ranges, the heights where the line fell
            under the faint line (18000 to 22000 ft) no longer faint (the FAA
            says 10 to 30 minutes), 25000 ft never reaches the
            line (the FAA says 3 to 5 minutes; hyperventilation is not modelled)
            and 40000 ft is 29 s against 15 s. OPEN: see `HYPOXIA_GAP`.
head blow   Concussion: loss of consciousness in about 10 percent of them
            (McCrea 2003, 9 percent (u)), mostly seconds to minutes. The sim's
            ko_until is set by impact.head_result (not this lane's file).
burn shock  Burns past about 15 to 20 percent of the skin need fluid, and half of
            people die at about 60 percent in the middle years (Baux, u). The sim's
            p["burn"] is a stand-in for the burnt share (skinburn waits for the
            heat law's flux), so its lines 0.2 and 0.45 are not checked here.
"""
import math

# FAA time of useful consciousness: (altitude ft, seconds), the middle of the quoted range
TUC = ((18000.0, 25 * 60.0), (22000.0, 10 * 60.0), (25000.0, 4 * 60.0), (28000.0, 160.0),
       (30000.0, 90.0), (35000.0, 45.0), (40000.0, 17.0), (43000.0, 10.0))
P_SEA_KPA, P_WATER_KPA, O2_AIR = 101.325, 6.3, 0.2095


def pressure_kpa(alt_ft):
    """The standard atmosphere (ICAO), troposphere then the isothermal layer."""
    h = alt_ft * 0.3048
    if h <= 11000.0:
        return P_SEA_KPA * (1.0 - 2.25577e-5 * h) ** 5.25588
    return 22.632 * math.exp(-(h - 11000.0) / 6341.62)


def sea_level_equivalent_o2(alt_ft):
    """The oxygen fraction of sea-level air that gives the same inspired
    oxygen pressure as air at this altitude (lungs hold 6.3 kPa of water vapour)."""
    pio2 = (pressure_kpa(alt_ft) - P_WATER_KPA) * O2_AIR
    return pio2 / (P_SEA_KPA - P_WATER_KPA)


def model_faint_seconds(frac_of_fresh, tau_s=40.0, line=0.55, curve=True):
    """Seconds from full blood to the faint line, breathing air that holds this
    share of fresh air's oxygen (the closed form of `_breathe`'s relaxation toward
    `o2_target`; curve=False is the straight line it used to chase).
    inf when the blood settles above the line."""
    x = o2_target(frac_of_fresh) if curve else float(frac_of_fresh)
    if x >= line:
        return math.inf
    return -tau_s * math.log((line - x) / (1.0 - x))


def hypoxia_table():
    """[(altitude ft, sea-level-equivalent share of fresh air, FAA seconds, model
    seconds with the curve, model seconds on the old straight line)]"""
    return [(a, sea_level_equivalent_o2(a) / O2_AIR, t, model_faint_seconds(sea_level_equivalent_o2(a) / O2_AIR),
             model_faint_seconds(sea_level_equivalent_o2(a) / O2_AIR, curve=False))
            for a, t in TUC]


def vagal_share_below(drive, median=0.6, sigma=0.6, clip=2.5):
    """Share of people whose vagal threshold is under `drive` (hurt units): the
    log-normal of injury.vagal_threshold, clipped at +-clip sigma."""
    lo, hi = median * math.exp(-sigma * clip), median * math.exp(sigma * clip)
    if drive <= lo:
        return 0.0
    if drive >= hi:
        return 1.0
    z = math.log(drive / median) / sigma
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# the one place the distance to the sources is written down
HYPOXIA_GAP = ("25000 ft (and 22000) settle just above the faint line "
               "and the faint comes only from slow brain stores and a falling pump "
               "that the sim has not got: it needs a brain-store lag (u) as well as the curve.")


# THE OXYGEN CURVE (wave 14b). What the blood settles at, breathing air with this
# share of fresh air's oxygen. Haemoglobin loads along a Hill curve of the
# alveolar oxygen pressure (P50 3.5 kPa = 26.6 mm Hg, n 2.7: Severinghaus 1979),
# normal alveolar pressure 13.3 kPa at sea level (West, Respiratory Physiology).
# It is flat near fresh air (a room at 90 percent oxygen costs almost nothing)
# and falls steeply below about 5 kPa. The store is scaled so fresh air is 1.0.
# Hyperventilation, which lifts alveolar oxygen in real hypoxia, is not modelled
# (u), so the middle heights stay a little too slow to faint; see `hypoxia_table`.
PAO2_FRESH_KPA = 13.3
P50_KPA, HILL_N = 3.5, 2.7


def _hill(p_kpa):
    return p_kpa ** HILL_N / (p_kpa ** HILL_N + P50_KPA ** HILL_N)


FRESH_BAND = 0.95        # air this close to fresh is read as it always was (the field is float32)


def o2_target(frac):
    """The blood's oxygen store (0 to 1) breathing air with `frac` of fresh air's
    oxygen. Within FRESH_BAND of fresh air it is `frac` itself, as before (so a
    room of clean air changes nothing, bit for bit); below it follows the Hill
    curve, scaled to join the line at the band's edge. Above 1 it is 1."""
    x = float(frac)
    if x >= FRESH_BAND:
        return min(x, 1.0)
    if x <= 0.0:
        return 0.0
    return _hill(PAO2_FRESH_KPA * x) / _hill(PAO2_FRESH_KPA * FRESH_BAND) * FRESH_BAND
