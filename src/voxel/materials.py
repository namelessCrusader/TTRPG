"""What matter IS: material and fluid ids, property tables, physical constants, body data.

WAVE 13a. The per-material numbers of the old tables (SOLID, EMISS, SPAN, TOUGH, TSHOCK,
MELT, FREEZE, PYRO, O2_RATIO, SOOT, GLOW, BURN_AREA, TRANSMIT, MATNAME and the "ring"
of SOUND) now live in ONE place, the material record (src/voxel/matrec.py: one frozen
row per material, in SI, with its sources). The names in this file are VIEWS of it,
built once at import by `matrec.legacy_views`, with the very same numbers, types and
key order the literal tables had (tests/test_oracle_format.py and
tests/test_mattables.py prove it row by row). Edit a material's number there.

The lookup arrays (`_DENS_ARR` ...) are NMAT_PAD = 64 entries long, the liquid ones
NFLUID_PAD = 64 (src/voxel/mattables.py): the first twelve (nine) are the old arrays
byte for byte, ICE (id 12) has its row, and an id with no row copies AIR (NOFLUID).

NMAT (12) and NFLUID (9) count the ids the OLD tables know (`SOLID[m] for m in range(NMAT)`
is right, `np.zeros(NMAT)` for something a cell's id indexes is not). NMAT_ROWS (13) counts
the ids that have a row of the record; NMAT_PAD and NFLUID_PAD size every array a cell's
id indexes.
"""
import numpy as np

from . import matrec as _matrec
from . import mattables as _mattables
from .species_seed import NFLUID_PAD, NMAT_PAD   # (re-exported: the pad sizes are defined once, there)


NOFLUID, WATER, OIL, ACID, WEAK_ACID = 0, 1, 2, 3, 4
MTIN, MLEAD, MIRON = 5, 6, 7     # MOLTEN metals — a metal is one substance in
                                 # two states, and the states are ROWS
BLOOD = 8                        # what a body is carrying, until it is not

AMBIENT = 20.0                   # °C
# Voxel size is a World parameter (World(..., voxel_cm=5)). Per-voxel amounts scale
# with voxel VOLUME: capacity in ml, solid grams and the air's heat capacity. The
# heat and fire laws are written in SI per second and multiplied by TICK_S, so
# they do not depend on the voxel size or the tick.

AIR, WOOD, STONE, IRON, FLESH, ASH, GLASS, LEAF, CHAR = 0, 1, 2, 3, 4, 5, 6, 7, 8
TIN, LEAD = 9, 10
ROPE = 11                        # natural-fibre rope (manila, hemp): see SOLID
ICE = 12                         # the first id past the old twelve (wave 13). Its row is T6's
                                 # (icephase.ICE_ROW), held by the record; it is in none of the
                                 # old dict tables and no cell holds it before the ice law is on
NMAT = 12                        # ids the OLD dict tables know (SOLID, EMISS ...): see the docstring

# THE MATERIAL RECORD. MATREC is the shipped table (matrec.BASE, the twelve old
# materials) with ICE added: 13 rows, ids 0..12, never renumbered (a retired row keeps
# its id and names its successor). Without the ICE row the ids would shift, so a
# missing icephase is an error here, not a silent fallback.
MATREC = _matrec.ingest_ice()
if MATREC is None:
    raise ImportError("materials.py needs icephase.ICE_ROW: mat id 12 (ICE) has no row without it")
NMAT_ROWS = MATREC.nmat          # 13: ids that have a row of the record
_MOLTEN = (MTIN, MLEAD, MIRON)   # the three molten fluid ids matrec is handed (it imports nothing here)
# The old dict tables are the record's rows for ids 0..11 only.
_V = _matrec.legacy_views(_MOLTEN, MATREC, upto=NMAT)


def _in_legacy_order(view, order):
    """`view` with its keys in `order`, the order the old literal listed them in. Python
    dicts iterate in insertion order and some laws walk a table in it (`_law_melt` goes
    through MELT and FREEZE), so a different order could change the order two events of
    one tick are logged in. A key that `order` does not name, or names twice, is refused:
    a row added to or dropped from an old table is a change somebody must make on purpose."""
    if sorted(order) != sorted(view):
        raise ValueError("the old table lists %s, the record gives %s" % (sorted(order), sorted(view)))
    return {k: view[k] for k in order}


# The heat columns are SI (docs/realism-references.md, tables 2.2 and 2.3):
#   k       conductivity, W/(m·K). Air 0.026; pine 0.12-0.15; granite 1.7-4.0 and
#           limestone 1.3; pure iron 83.5 at 0 °C, 54.7 at 327 °C; tissue 0.29-0.45;
#           glass 0.8-0.96; charcoal 0.084; tin 66.8; lead 35.3. Ash and leaves are
#           UNVERIFIED (no source opened): loose, mostly air.
#   ignition  the surface temperature where the solid gives off burnable gas
#           (wood 300-365 °C piloted) — a SURFACE, not the voxel's average.
#   burn    heat of combustion, J per gram burned.
#   rate    the fastest a burning surface can give off fuel, g/(m²·s) of exposed
#           face. Cone peaks for wood are 87-280 kW/m², 5-17 g/(m²·s); the rate a
#           fire actually burns at is set by its heat balance (`PYRO`), and this
#           is only the ceiling.
#            density g/L  c J/g°C  k W/(m·K)  ignition °C  burn J/g  rate g/(m²·s)
SOLID = _V["SOLID"]              # the rows are in matrec.py. Notes that belong to a row:
                                 #   FLESH  a body: mostly water, can char
                                 #   ASH    what burning leaves behind
                                 #   GLASS  see-through, acid-proof
                                 #   LEAF   foliage: light, flashy fuel. FRESH leaves
                                 #          carry water in the voxel and refuse to light
                                 #          until the moisture boils off (no flag)
                                 #   CHAR   what burning LEAVES: charcoal. Black forever,
                                 #          lights hotter, burns slow and fierce: embers
                                 #   TIN    soft white metal, melts LOW; LEAD is heavier
                                 #          and melts a bit higher
                                 #   ROPE   natural-fibre rope: see below
# ROPE is twisted natural fibre, manila or hemp (the user's decision, 2026-10-16:
# a rope is a MATERIAL, not "wood under 15% full"). Density: 12 mm three-strand
# manila weighs 0.104 kg/m and breaks at 10.6 kN (Engineering ToolBox, "Manila
# Ropes - Strength"; Orion Ropeworks' sheet gives 11.2 kg/100 m and 10.6 kN):
# 0.104 kg/m over the 1.13 cm² section is 920 g/L. 885 is src/bonds' own rope
# row (5% under), kept so both sides read one density. UNVERIFIED (no source
# opened, cellulose-fibre orders of magnitude): c 1.4 J/(g K) (cotton 1.3,
# bast fibres 1.3-1.5); k 0.10 W/(m K), a twisted bundle with air in it, between
# loose fibre (0.04) and solid cellulose (about 0.25); piloted ignition 260 °C,
# a little under wood's (cotton about 255); 16.5 kJ/g, cellulose's heat of
# combustion (16-17.5); and wood's 20 g/(m²·s) ceiling.
# The last two columns are SI too. rate, g/(m²·s): a pool boils off as fast
# as its flame heats it (`_law_burn`), and kerosene pools burn at 25-39 because
# that is what their flame gives back; the rate is only a ceiling, for a smear
# too thin to have a skin of its own. h: how hard the liquid takes
# heat from a solid it wets, W/(m²·K): free convection of liquids is 50-1000,
# boiling 2,500-100,000, and a film of steam over red-hot iron only 200-500.
# One number per liquid stands in for that curve (G5, item 6, not built), a game
# choice: water 1500, oil about a third of it, as quench severities go (water
# 0.9-1.2, oil 0.3-0.4). Molten metals wet hard. UNVERIFIED for acid and blood.
#            density g/ml c J/g°C  viscosity(spread/tick)  ignition  burn J/g  rate  h
FLUID = {
    WATER:     (1.0,  4.2, 0.50, None, 0.0,  0.0, 1500.0),
    OIL:       (0.9,  2.0, 0.25, 250., 40000.0, 60.0, 500.0),
    ACID:      (1.8,  1.4, 0.40, None, 0.0,  0.0, 1500.0),  # "vitriol" — strong mineral acid
    WEAK_ACID: (1.05, 3.8, 0.48, None, 0.0,  0.0, 1500.0),  # "aqua debilis" — vinegar-class
    MTIN:      (6.98, 0.24, 0.6, None, 0.0,  0.0, 5000.0),  # molten tin: runs like water,
    MLEAD:     (10.66, 0.14, 0.6, None, 0.0, 0.0, 5000.0),  # scalds like fire
    MIRON:     (7.0,  0.82, 0.5, None, 0.0,  0.0, 5000.0),
    BLOOD:     (1.06, 3.6,  0.45, None, 0.0, 0.0, 1000.0),  # a little thicker than water
}
_FLIDS = (WATER, OIL, ACID, WEAK_ACID, MTIN, MLEAD, MIRON, BLOOD)
NFLUID = 1 + len(_FLIDS)         # 9: the fluid ids the OLD tables know (0 = none, then the eight)
_FDENS = [0.0] + [FLUID[f][0] for f in _FLIDS]
_FC = [1.0] + [FLUID[f][1] for f in _FLIDS]
_FK = [0.0] + [FLUID[f][6] for f in _FLIDS]
# REACTIONS — data, not law. The dissolution law (fluids.py) reads this table; a new
# acid (or ANY fluid-eats-solid interaction) is a ROW, never code. Same philosophy
# as the sandbox interaction sets (Powder Toy's element matrix), but each row is a
# conserved mass conversion:
#   (rate, ml of stock spent per g, J per g, residue)
# RATE is measured data: points (weight fraction of the reagent, mg per cm² of
# wetted face per hour), read in straight lines between the points and flat past
# the ends. So a threshold is two points close together, and a rate that grows
# with strength is a slope. It is NOT rate x potency: real acids have thresholds
# (vitriol is skin-corrosive from 15 wt%; steel is passive in it above ~60 wt%).
# RESIDUE is (material, share of a full voxel): what the solid becomes, instead of
# air, once only that much of it is left. None: it dissolves away.
# The numbers and their sources: docs/realism-references.md, table 2.5 (C1-C19).
# Which acid is which: ACID is vitriol, 98% sulfuric acid at potency 1. It chars
# wood (it pulls the water out of cellulose), eats limestone and dilute-strength
# iron, leaves strong-strength iron, charcoal and glass alone, and it does NOT
# dissolve flesh: an acid burn kills the tissue in place (coagulation) and removes
# nothing. The burn belongs to the body model, which does not exist yet (rulings
# case 2), so there is no FLESH row on purpose.
# WEAK_ACID is vinegar, 5% acetic acid at potency 1: harmless to skin, wood and
# leaves (vinegar is kept in wooden casks); it slowly etches carbonates and iron.
# Not modelled yet: the products. Dissolved mass leaves as smoke here, where it
# should be H2 from metals, CO2 from carbonates and a salt left in the liquid;
# that needs gas species (roadmap 107, list B).
REACTIONS = {
    # Dehydrates cellulose to carbon. 72% acid at 30 °C takes 0.25-0.85 mm wood
    # particles apart in <= 30 min: >= 0.014 mm/min, ~51 mg/(cm²·h) at 0.6 g/cm³
    # (derived; source 130), and it leaves 22-26% of the mass as black residue
    # (130). Wood tanks hold acid of pH >= 1.5 for decades (129), so dilute acid
    # does nothing; where between the two it starts is UNVERIFIED (no source):
    # here at 72%, the strength source 130 measured. The water pulled out (56% of
    # cellulose by mass) dilutes the acid: counted as 0.56 ml of stock spent per g.
    # Heat: that water's heat of dilution, ~0.031 mol/g x 16.3 kJ/mol (Table
    # B.11, n = 3 row, derived).
    (ACID, WOOD):       (((0.70, 0.0), (0.72, 51.0)), 0.56, 510.0, (CHAR, 0.25)),
    (ACID, LEAF):       (((0.70, 0.0), (0.72, 51.0)), 0.56, 510.0, (CHAR, 0.25)),
    #                   ^ leaves are cellulose too: the wood numbers, UNVERIFIED
    # Limestone. Calcite in 1 N HCl at 25 °C: 0.47 mg/(cm²·s) = 1690 mg/(cm²·h)
    # (135). 1 N is 4.8 wt% of sulfuric acid by acid strength. The rate is limited
    # by how fast acid reaches the face, so it grows with strength below that
    # (an inference, 135) and is held flat above it (UNVERIFIED: in vitriol a
    # gypsum crust should make it FALL with time, C9, source 137). 1 g of calcite
    # uses 0.54 ml of 98% acid and gives 141 J when gypsum forms (94, 134).
    # One STONE stands for limestone and granite alike until they are split (C8).
    (ACID, STONE):      (((0.0, 0.0), (0.048, 1690.0)), 0.54, 141.0, None),
    # Wood ash is alkaline, mostly carbonate: the limestone row. UNVERIFIED (C19).
    (ACID, ASH):        (((0.0, 0.0), (0.048, 1690.0)), 0.54, 141.0, None),
    # Mild steel in 1 M HCl at 30 °C: 5.74 mg/(cm²·h) (131; 132 has 0.7). Dilute
    # sulfuric acid is "extremely corrosive" to steel, but above ~60 wt% (below
    # 38 °C) steel is passive: < 0.02 in/yr, < 0.045 mg/(cm²·h) (133). Between
    # 4.8 and 59 wt% the rate is held flat (UNVERIFIED). Not modelled yet: hot
    # acid (above 38 °C, 60-93 wt% eats steel again, 133). Fe + 2H+: 0.97 ml of
    # 98% acid and 1.6 kJ per g of iron (94, 134).
    (ACID, IRON):       (((0.0, 0.0), (0.048, 5.7), (0.59, 5.7), (0.60, 0.02)),
                         0.97, 1600.0, None),
    # Vinegar. No opened source gives either rate (C8, C18): both UNVERIFIED.
    # Iron: ~1 mm/yr, 0.09 mg/(cm²·h). Carbonate: the 1 N rate scaled by the
    # free acid of vinegar (pH ~2.4), 7 mg/(cm²·h); under the C-T12 bound
    # (< 0.01 mm in 60 s). Spend is stoichiometry (derived): 2 mol acetic acid
    # per mol of iron or calcite, 0.00087 mol in a ml of 5% vinegar. The heats
    # are the strong-acid ones (UNVERIFIED for acetic acid).
    (WEAK_ACID, IRON):  (((0.0, 0.0), (0.05, 0.09)), 41.0, 1600.0, None),
    (WEAK_ACID, STONE): (((0.0, 0.0), (0.05, 7.0)), 23.0, 150.0, None),
    (WEAK_ACID, ASH):   (((0.0, 0.0), (0.05, 7.0)), 23.0, 150.0, None),
}
# MISCIBLE — which fluid pairs dissolve into each other, and which species the
# blend counts as. Water and the acids are one family (an acid IS water carrying
# a reagent — POTENCY tracks how much); the blend keeps the acid's name and the
# potency mixes by volume, water counting as potency zero. Oil joins nothing.
MISCIBLE = {frozenset({WATER, ACID}): ACID,
            frozenset({WATER, WEAK_ACID}): WEAK_ACID}
# STOCK — what POTENCY 1 of each acid is: the weight fraction of the reagent in
# it. Potency is the share of stock by VOLUME, the rest water, so the weight
# fraction the sources quote is w·ρ·p / (ρ·p + 0.998·(1 − p)), with ρ the stock's
# density: half-strength vitriol is about 63 wt%, not 49 (C12).
STOCK = {ACID: 0.983, WEAK_ACID: 0.05}
# DILUTION — vitriol meeting water gets HOT. The integral heat of mixing of
# sulfuric acid: kJ given off per mol of acid, against n = mol of water per mol
# of acid. Felder, Rousseau & Bullard Table B.11 (C11, source 139), only the rows
# the reference quotes; n = 0 is the acid alone. (molar mass g/mol, rows).
# Vinegar has no row: diluting acetic acid gives off almost nothing.
DILUTION = {ACID: (98.08, ((0.0, 0.0), (3.0, 48.99), (100.0, 73.97), (np.inf, 96.19)))}


def _potency_at(f, wt):
    """The potency at which acid f is `wt` (a weight fraction) strong."""
    w, rho = STOCK[f], FLUID[f][0]
    return 1.0 if wt >= w else 0.998 * wt / (rho * (w - wt) + 0.998 * wt)


def _mix_heat(f):
    """(potency, J) for one ml of acid f at that potency: the heat given off
    by making it from stock and water. Mixing two liquids gives off what the
    blend holds minus what the two parts held. The DILUTION rows are read in
    straight lines in n/(n + 100): that keeps the curve concave, so mixing any
    two strengths gives off heat and never takes it in, and it puts infinite
    dilution at 1. A 1:1 mix comes out at 0.86 kJ per ml of stock, as C11 has."""
    molar, rows = DILUTION[f]
    w, rho = STOCK[f], FLUID[f][0]
    xs = np.array([1.0 if n == np.inf else n / (n + 100.0) for n, _h in rows])
    ys = np.array([h for _n, h in rows])
    p = np.linspace(0.0, 1.0, 2001)
    acid = p * w * rho / molar                                    # mol per ml
    water = (p * (1.0 - w) * rho + (1.0 - p) * 0.998) / 18.015    # mol per ml
    n = water / np.maximum(acid, 1e-300)
    n0 = (1.0 - w) * molar / (w * 18.015)                         # the stock's own
    return p, 1000.0 * acid * (np.interp(n / (n + 100.0), xs, ys)
                               - np.interp(n0 / (n0 + 100.0), xs, ys))


# The rate points re-read against potency, which is what a cell stores.
_RATE_PTS = {k: (np.array([_potency_at(k[0], wt) for wt, _r in row[0]]),
                 np.array([r for _wt, r in row[0]]))
             for k, row in REACTIONS.items()}
_MIXHEAT = {f: _mix_heat(f) for f in DILUTION}
# Which way a freshly built body's CHEST points: shoulders along x, right arm
# on +x, so it faces +y. `_turn` keeps `p["chest"]` up to date after that.
CHEST0 = (0.0, 1.0)
# PHYSIOLOGY — what being alive means, as data. Blood chases the air it breathes;
# smoke rides along and blocks uptake (the carbon-monoxide effect); heat cooks
# skin above a threshold. Below faint the muscles let go; below death, the end.
BODY = {"breath": 0.02,          # blood O2 relaxes toward inhaled air at this rate
        "smoke_in": 0.02,        # inhaled smoke g -> choking load, per tick.
                                 # Deliberately CO-heavy: real smoke carries
                                 # carbon monoxide we can't track until gas
                                 # species exist, so soot grams stand proxy
                                 # for the whole toxic package (Purser FED
                                 # replaces this wholesale later)
        "smoke_out": 0.001,      # the load clears slowly in clean air
        "faint_o2": 0.55, "death_o2": 0.25,
        "wake_o2": 0.70,         # and consciousness COMES BACK when the blood
                                 # recovers — well clear of the fainting line,
                                 # not marginally past it, or a body pulled to
                                 # clean air would flicker on the threshold.
                                 # Hysteresis is the real thing here, not a
                                 # smoothing trick: recovery genuinely lags
        "hurt_T": 55.0,          # skin °C where tissue starts to cook
        "feel_T": 45.0,          # and where it is hot enough to NOTICE. Below
                                 # cooking on purpose: you feel a fire well
                                 # before it marks you, which is the whole use
                                 # of feeling it
        "burn_gain": 4e-6,       # damage per degree-over-threshold per tick
        "faint_burn": 0.2, "death_burn": 0.45,
        # A BLOW WOUNDS. Tissue takes damage far below the toughness that tears
        # it apart — TOUGH[FLESH] is gross failure, this is where bruising and
        # breakage begin, per contact area. Landing energy beyond it goes into
        # p["hurt"], the same integral shape as burns, read against the same
        # faint/death thresholds as one combined tissue damage. Wounds never
        # come back down, the same one-way street as burns.
        # calibration: set so a body's own jump landings cost nothing, a 1.5 m
        # drop onto stone bruises, ~3 m knocks out, ~6 m kills — scaled to
        # THIS body, which is light (28.5 kg, a known softness)
        "bruise_kJm2": 12.0,     # tissue injury threshold, kJ per m² of contact
        "hurt_J": 2500.0,        # absorbed joules beyond it that sum to 1.0
        # HANDS, as ONE cause. A body can put a bounded force on a thing it can
        # reach; what happens then is arithmetic, not a list of verbs. Lifting
        # fights gravity (m·g), shoving fights friction (µ·m·g), so the SAME
        # strength gives two different limits and nobody has to write them down
        # separately. ~400 N is what an unremarkable adult manages: about 40 kg
        # off the floor, and roughly 100 kg shoved along it.
        "strength_N": 400.0,
        # A FIST CAN PULL, NOT CLAMP. Strength says what a grip can hang on to;
        # this says what it can BALANCE. A load below the hold is a pendulum and
        # costs the wrist nothing; a load whose weight rides above the hold is
        # stood on the fist, and the wrist pays weight times lever to keep it
        # there. ~20 N·m is holding a 4 kg hammer level at half a metre — near
        # the edge of what an unremarkable wrist manages.
        "wrist_Nm": 20.0,
        # LEGS ARE NOT ARMS. Reusing the arm number for a jump gets a push
        # barely above the body's own weight, which is not a jump; a leg press
        # is several times what the same person can lift. This is the force the
        # legs put into the GROUND, and how far the body has to push over
        # (a crouch), which together are all a jump is: work done against
        # weight, turned into speed, turned into height. Nobody types a height.
        # WHAT LEGS CAN PUSH WITH. A muscle's force is its CROSS-SECTION times
        # a stress, which is why a thicker leg is a stronger one and why this
        # should never have been a flat number: a body twice as tall has four
        # times the leg section and eight times the mass, so with 1400 N for
        # everybody a giant could not jump at all — measured, 0.00 m/s.
        #
        # `legs_N` is the fallback for a body with no legs declared. The stress
        # is the part still guessed, and it is LOW for muscle (about 90 kPa
        # against a real 300) because this body is a stick figure: its legs are
        # thinner, relative to its height, than a person's are. Calibrated so
        # the humanoid comes out where it always was.
        "legs_N": 1400.0, "muscle_Pa": 93000.0,
        # A JOINT'S TORQUE IS NOT ITS LIMB'S LENGTH. A muscle pulls on a bone a
        # few centimetres from the joint, not at the far end of it — so a
        # shoulder's torque is its section times a stress times that SHORT
        # insertion, and deriving it from the limb's own length would make an
        # arm five times stronger than any arm.
        #
        # This is what the declared numbers already implied and nobody had
        # noticed: 60 N.m against a 698 N arm is a moment arm of 8.6 cm, which
        # is a real deltoid. The fraction below is that, over the arm's length.
        "lever_frac": 0.19,
        # HOW FAR A BODY SINKS BEFORE IT PUSHES OFF. `crouch_m` is the fallback
        # for a body with no legs declared; a body WITH legs is measured, and
        # this fraction is the only part of it still a guess — how much of a
        # leg's length a crouch uses, which the lattice does not know.
        "crouch_m": 0.25, "crouch_frac": 0.36,
        # A SHOULDER, in newton-metres. This is the only number a swing needs:
        # the segment's own inertia decides how fast it comes round, so a heavy
        # arm is slower than a light one and an arm holding something is slower
        # still, without anybody writing down a duration. ~60 N.m is an
        # unremarkable adult shoulder.
        "arm_Nm": 60.0,
        # AND A BACK IS NOT AN ARM. The hips carry the torso, the head and both
        # arms, so the same lever holds far more mass — a trunk extensor makes
        # a few hundred N.m, and this is the low end of an untrained adult. It
        # is the number that decides how far out a man may lean before his own
        # back, rather than his balance, says no.
        "back_Nm": 200.0,
        # HOW FAR A TRUNK BENDS AT ALL. Not a balance figure and not a muscle
        # figure — both of those are computed — but the joint's own stop, the
        # place where a spine has run out of spine. Asked to lean out, a body
        # asks for this and gets however much of it the world allows.
        "lean_max": 0.8,         # rad; ~46 degrees, and the lattice tears at 1.0
        # AND A TRUNK IS SLOWER THAN A SHOULDER. Much more mass on a much
        # shorter lever; nobody snaps into a bow. It is not a cosmetic figure:
        # at a shoulder's pace the first tick of a lean is 0.375 rad, which is
        # already past the angle a man can stand at, so the body would propose
        # a pose it could not keep and think better of it, for ever, and never
        # bend at all.
        "waist_wmax": 1.5,       # rad/s
        # A BODY HAS TWO ARMS AND A PREFERENCE. The dominant hand is stronger
        # and better practised, so it is what a body reaches with — but it is a
        # PREFERENCE, not a rule, and the other hand wins when the other hand
        # is plainly the one for the job. `off_hand_m` is how much further a
        # body will reach across itself rather than switch hands: past that
        # much difference in distance, being right-handed stops mattering.
        # That is a reason rather than a die, which matters because the sim
        # carries no die and because "sometimes the other hand" is not a
        # coin-toss in real bodies either — it is where the thing is.
        # About half a shoulder width, and that is the scale it should be on:
        # the cost of the off hand is that it is weaker, the benefit of the
        # near hand is not reaching across yourself, and those trade at roughly
        # the distance between your shoulders. Set wider than the shoulders it
        # is not a preference, it is a rule, and the off hand never wins.
        "off_hand_m": 0.10,
        "off_hand": 0.8,         # what the other hand is worth, in strength
        # WHEN THE HAND LETS GO. An arm swinging up from rest carries its hand
        # on a circle, so the hand's velocity is always at right angles to the
        # arm: hanging straight down it is going FORWARD, straight out in front
        # it is going UP, and half way between it is going forward and up at
        # 45 degrees. Which is the angle that throws a thing furthest, and it
        # falls out of the geometry rather than being aimed — a body throws
        # well because of where its shoulder is, not because it knows ballistics.
        "throw_rad": 0.785,      # pi/4 from hanging
        # CATCHING is stopping something, and stopping is force times time. A
        # hand gives as it closes — that is what makes a catch different from
        # a wall — and how long it gives for is the whole of why a cricket ball
        # can be caught and a brick at the same speed cannot.
        "catch_s": 0.12,         # seconds a closing hand takes to stop a thing
        # AND NOTHING IS CAUGHT BEFORE IT IS SEEN COMING. Stopping an arrow
        # takes 25 N, which any hand has; what no hand has is the 0.2 s it
        # takes to answer something (simple visual reaction time). An arrow
        # at 50 m/s crosses a room in less. Measured before this: a man
        # caught an arrow loosed from 1.4 m away, because the rule asked
        # only whether his arm was strong enough.
        "react_s": 0.2,
        # HOW FAR BACK A THROW STARTS. Nobody throws from their hip: the arm
        # goes back first, and the whole of what that buys is ARC — more of it
        # to accelerate through before the hand opens. It is not a separate
        # motion and needs no new law, only a swing that begins behind the body
        # instead of under it.
        "windup_rad": 0.9,
        # AND A MUSCLE CANNOT PULL AT ANY SPEED. Hill's force-velocity
        # relation: the faster a muscle is already shortening the less force
        # it makes, so torque fades to nothing at a top speed. Without it an
        # arm accelerates for as long as the swing lasts, and a LIGHT arm —
        # ours is light, see scenes._person — reaches speeds no shoulder can.
        # That is what let a BARE FIST splinter a fence post. It is a cause
        # rather than a speed limit typed in: the same relation is why an arm
        # carrying something heavy swings slow.
        "arm_wmax": 15.0,        # rad/s, a shoulder unloaded
        # A JOINT IS NOT A FRICTIONLESS BEARING. A real arm let go swings two
        # or three times and stops; a pendulum on a knife edge swings for
        # minutes. The difference is flesh working against itself, and this is
        # the fraction of a joint's angular speed that goes to heat each
        # second. GUESS — nothing in the lattice measures it and no source was
        # read for it; it is set to the value that gives a dropped arm the two
        # or three swings a dropped arm has. What it is NOT allowed to do is
        # lose the energy: what damping takes comes off the swing and goes onto
        # `shed`, so the books still close.
        "joint_damp": 2.2,       # 1/s
        "reach_m": 0.30}         # how far an arm goes. A grip is not
                                 # telekinesis: what is held has to stay within
                                 # reach or it is not held any more
# A BLOW, LAYER BY LAYER. What a strike does to a body is decided by one
# quantity — energy per unit of CONTACT AREA — met by each layer in turn:
# whatever the body is wearing, then skin, then the tissue under it. Cut and
# bruise are not two kinds of attack. They are the same energy arriving on a
# small area or a large one: an edge declares a small area (the axe's 2e-4
# m^2, sub-voxel shape the lattice cannot draw — Ruling 2 q1), a club's face
# or a fist arrives on a whole voxel face, and a breastplate turns the first
# into the second by spreading it. Every number here is a named guess with
# where it came from, not a knob.
WOUND = {
    # SKIN PARTS at about 0.1 J per square millimetre of contact — the figure
    # wound ballistics uses for skin penetration (Sellier & Kneubuehl). Below
    # it the energy bruises; above it the edge is IN.
    "skin_Jm2": 1.0e5,
    # AND THEN IT HAS TO CUT ITS WAY. Driving an edge through muscle takes a
    # pressure on the edge of about 1.5 MPa (a knife needs ~50-100 N on a
    # ~5e-5 m^2 contact). Energy left after the skin, over that pressure times
    # the contact area, is how deep it goes.
    "tissue_Pa": 1.5e6,
    # AN EDGE IS LONGER THAN IT IS WIDE. A declared edge area is length times
    # width; a honed edge meets flesh about 2 mm wide, so the same area says
    # how long the cut is. A 2e-4 m^2 axe bit opens ten centimetres.
    "edge_w_m": 2.0e-3,
    # PUNCHING A PLATE. An edge gets through armour by shearing out a plug:
    # about yield stress x contact area x thickness, times a plugging factor of
    # ~3. With wrought iron at 250 MPa and a 1.5 mm plate, an axe (2e-4) needs
    # ~225 J and bounces; a bodkin point (1e-5) needs ~11 J and goes through —
    # which is the real historical argument about arrows and plate, arrived at
    # rather than looked up.
    "yield_Pa": {IRON: 2.5e8},
    "plug": 3.0,
    # WHAT MAIL DOES NOT DO is spread a blow. Rings stop an edge the way a
    # plate does, but they are not stiff, so a blunt blow arrives on the body
    # hardly wider than it landed — why mail was worn over padding.
    "mail_spread": 4.0,
}
# BLEEDING. An open wound is a plane cut through tissue, and blood leaves
# through the vessels crossing it — so what bleeds is the area of that plane,
# at a pressure that falls as the blood runs out. Named guesses with their
# targets, not fitted knobs:
BLEED = {
    # ~70 ml of blood per kg of body (adult physiology). This body is light
    # (scenes._person), so it carries about three litres, not five.
    "ml_per_kg": 70.0,
    # FLOW THROUGH A CUT, per square metre of cut plane at full pressure. Set
    # so a serious limb wound — 10 cm long, 5 cm deep, 5e-3 m^2 — runs at
    # about 2 ml/s, which takes a light body to the fainting line in a few
    # minutes: the pace of a wound that needs a tourniquet, not a scratch.
    "ml_s_m2": 400.0,
    # CLOTTING SEALS A FIXED AREA A SECOND, so a scratch stops by itself and
    # a deep cut does not: the same process, a different ratio of what it can
    # close to what is open. About a square centimetre a minute.
    "clot_m2_s": 1.7e-6,
    # PERFUSION: how much of the blood's oxygen still reaches the brain as the
    # volume falls. The heart compensates, then fails — ATLS class III
    # (30-40% lost) is confusion and collapse, class IV (>40%) is lethal —
    # so ((f - f0) / (1 - f0)) ** k with f0 and k set to put the existing
    # faint and death lines at ~35% and ~50% lost. The oxygen thresholds are
    # the ones smoke already uses; bleeding just delivers less of it.
    "f0": 0.45, "k": 0.56,
}
FRICTION = 0.40                  # sliding, solid on solid. One number until a
                                 # scenario needs ice or grease; a per-pair
                                 # table is the honest end state, and this is
                                 # the value that makes the two limits above
                                 # come out where a person's really do
GRAVITY = 9.81
TICK_S = 0.025           # SECONDS IN A TICK. It was never written down, and two
                         # parts of the sim quietly assumed different answers —
                         # the will layer says "nobody re-deliberates every
                         # fortieth of a second" while the fall law dropped
                         # everything exactly one voxel a tick, which at 5 cm is
                         # a flat 2 m/s for a feather and an anvil alike. A
                         # clock is not a tuning knob; it is the thing that lets
                         # an acceleration mean anything.
AIR_DENS = 1.2           # kg/m^3 at room temperature. The air was always there
                         # as a chemistry (O2, smoke, pressure) and never as a
                         # MASS that something falling has to shove out of the
                         # way.
DRAG_CD = 1.1            # a blunt slab, near enough. Everything a parachute
                         # does is this number meeting a big area and a small
                         # mass; there is no canopy special case to write.
FALL_SUBSTEPS = 16       # most voxels a column may drop in one tick. Matter
                         # cannot move more than a cell per support sweep — the
                         # thing it might land on has to be re-asked each time —
                         # so a fast fall runs the sweep again within the tick.
                         #
                         # It was FOUR, and the note against it said that capped
                         # falls at 8 m/s. It did not. It capped the DESCENT and
                         # left the speed running: a body that cannot fall as
                         # fast as gravity is pulling it spends LONGER falling,
                         # and gravity goes on adding to it the whole time. So a
                         # long drop arrived too FAST, not too slow — measured,
                         # 27.8 m/s off a 20 m fall where free fall gives 19.8.
                         # Energy goes as v squared, so that landed with twice
                         # the blow it should have.
                         #
                         # Sixteen is honest to about 40 m and costs nothing:
                         # the substep loop already stops the moment no column
                         # has any fall left, so a world with nothing falling
                         # never runs a second sweep. Measured with a slab
                         # dropping through a furnished room: 1.01x a tick.
_REACTIVE = {f for (f, _m) in REACTIONS}
PLUME_REACH_M = 2.4              # meters of entrainment catchment: the air a
                                 # fire's plume can actually pull in — health
                                 # and the O2 debt read this same reach.
                                 # PHYSICAL length: a finer lattice blurs over
                                 # more cells, or the same bed fire suffocated
                                 # at 5 cm that thrived at 10 cm (measured)
BOIL, LATENT = 100.0, 2260.0     # water boils at 100 °C; real latent heat, J per gram
# RAD and LEAK are gone (G4, G5): RAD was a radiation term on EVERY voxel,
# buried ones too, and LEAK took 0.4% of every voxel's heat each tick, a 6 s
# time constant for anything. Heat now leaves only through exposed faces
# (`_law_conduct`).
# ── HEAT IN SI (docs/realism-references.md, G4 and G5) ───────────────────────
SIGMA = 5.670374e-8              # Stefan-Boltzmann, W/(m²·K⁴)
AIR_C = 1.2                      # heat capacity of air, J/(L·K) (ρ·cp, H12). It was
                                 # 50, a "thermal floor" added to every voxel, 41x
                                 # real. No floor is needed: every face law below is
                                 # clipped so no cell passes its neighbour (H18, H19)
H_CONV = 10.0                    # free convection, a surface to still air, W/(m²·K)
                                 # (2-25, H14)
EPS_GAS = 0.2                    # how strongly hot gas (a flame, a smoke layer)
                                 # radiates onto a cooler surface it touches.
                                 # UNVERIFIED: a game choice. A 5 cm sooty flame is
                                 # 0.1-0.3 thick optically; 0.2 puts a 1000 °C flame
                                 # licking a 300 °C wall at ~35 kW/m² (wall flames
                                 # give 20-40). Radiation across open air is G7

EDDY = 7500.0                    # hot air churns: an eddy diffusivity of
                                 # (T - 20) / EDDY m²/s (0.08 m²/s at 620 °C). The old
                                 # "k grows with temperature" stand-in for convection,
                                 # unchanged in what it does, now written in SI
FACE_CLIP = 0.15                 # no face moves more than this share of the smaller
                                 # side's heat capacity times the difference in one
                                 # tick. Six faces stay under 1, so no cell can pass
                                 # its hottest neighbour (H18) and a checkerboard
                                 # decays (H19)
# EMISSIVITY of each surface (H14): oxidised steel 0.79, wrought iron 0.94, oak
# 0.89, pine 0.95, glass 0.92-0.94, water 0.95-0.963. Stone, ash, char and flesh
# are UNVERIFIED (dark rough surfaces, near 0.9). Tin and lead are dull metals.
EMISS = _V["EMISS"]              # ROPE 0.9: a dark rough fibre surface (UNVERIFIED)
EMISS_FLUID = {WATER: 0.95, OIL: 0.9, ACID: 0.9, WEAK_ACID: 0.95, MTIN: 0.3,
               MLEAD: 0.3, MIRON: 0.4, BLOOD: 0.95}
# LIQUID conductivity, W/(m·K): heat between two wet cells. Water 0.61 (H20),
# oil 0.17; molten metals about half their solid. Acid and blood UNVERIFIED.
FLUID_K = {WATER: 0.6, OIL: 0.17, ACID: 0.35, WEAK_ACID: 0.6, MTIN: 30.0,
           MLEAD: 15.0, MIRON: 40.0, BLOOD: 0.5}
# THE SKIN. A voxel is 5-10 cm, but a fire lives in its outer few millimetres:
# under 50 kW/m² a pine surface reaches 300 °C in about 18 s while its middle
# is still cold. Averaged over the voxel, SI conduction would take minutes
# (F4, G4). So every solid with an exposed face carries one more number, its
# skin's temperature above the voxel's average (`World.skin`), and the skin
# trades heat with the faces it covers and, slowly, with the core.
# The skin is a layer sqrt(a*t) deep, a the material's diffusivity: SKIN_T[0]
# for its depth (what it holds) and SKIN_T[1] for its path to the core (how fast
# it drains). Fitted to the piloted-ignition times of wood (t_ig = 130 rho^0.73
# / (q - 11)^1.82 s, F4): 4.3 mm and 12.5 mm for pine, so 50 kW/m² lights it in
# about 30 s and 25 kW/m² in about 65 s, within the formula's spread, and the
# critical flux comes out near 12 kW/m². A material whose skin would be a
# quarter of the voxel or more (iron, tin, lead) is simply even all through.
SKIN_T = (126.0, 1060.0)         # seconds
# BURNING, as an energy balance (G4). A burning surface sits at its ignition
# temperature the way boiling water sits at 100 °C: heat beyond it turns solid
# into gas, at L_g joules a gram. The gas burns in the flame, which gives the
# surface back q_fl W/m² (never more than the flame makes). The rest of the flame
# goes up as hot gas into the open cells beside it, except CHI_R, its radiation,
# which leaves (radiation across air is G7). So a lone surface with no other
# heat falling on it burns slowly or goes out, and a crib, where the sticks heat
# each other, burns hard.
# Wood: heats of gasification 1.8-3.5 kJ/g; laminar wall flames give 25-35 kW/m².
# With these a slab burns at 4-8 g/(m²·s) on its own (F6: 2.9-9.6). LEAF burns
# from every leaf in the cell, not from its six faces: BURN_AREA counts it as
# several times its face area (UNVERIFIED, set so a dry crown burns out in
# tens of seconds, F18). FLESH is mostly water and barely keeps a flame.
#          L_g J/g  q_fl W/m²  O2 limit (share of fresh air)
PYRO = _in_legacy_order(_V["PYRO"], (WOOD, LEAF, FLESH, ROPE))   # ROPE: cellulose, as wood (UNVERIFIED)
PYRO_OIL = (1000.0, 20000.0, 15.0 / 21.0)   # a pool: 10-30 g/(m²·s), F14, F-T11
BURN_AREA = _V["BURN_AREA"]
CHI_R = 0.3                      # radiant share of a flame (0.15-0.6; wood 0.30, H17)
# THE FLAME'S TEMPERATURE IS NOT A NUMBER WE PICK; it comes from what the fire
# burns and the air it is given (it was a flat 1000 °C cap, COMBUST_CEIL and
# T_FLAME, which welding heat could never pass). A fuel gives
# `burn / O2_RATIO` joules for each gram of oxygen it uses (about 12-13 kJ for
# every organic fuel, F8). That heat, less the share a flame radiates away
# (CHI_R), warms everything that passes through the flame: the fuel gas, the
# oxygen and the nitrogen that came with it, and the extra air the flame
# mixes in. So
#     T = AMBIENT + (1 - CHI_R) * heat_per_g_O2 / (C_GAS * grams_heated)
# where grams_heated, for each gram of O2 burned, is 1/O2_RATIO of fuel plus
# LAMBDA_FREE times the 4.31 g of air that carries it (23.2% of air is O2),
# and more of it still when the air is thin (health < 1: a room that is
# running out of oxygen has a cooler flame).
AIR_O2_MASS = 0.232              # share of air's mass that is oxygen
C_GAS = 1.25                     # J/(g·K), hot combustion gas (H12; 1.1-1.3)
LAMBDA_FREE = 1.4                # a free flame mixes 1.4 times the air it burns.
                                 # CALIBRATED, not measured: it puts wood's
                                 # flame at about 1010 °C and a big oil pool's
                                 # at about 1190 °C (H28: 900-1000 under a metre,
                                 # 1100-1200 for large pools)


def flame_temp(heat, ratio, health=1.0, lam=LAMBDA_FREE, chi=CHI_R):
    """°C of the flame of a fuel with `heat` J per gram and `ratio` g of O2 per
    gram, in air whose oxygen is `health` of fresh and mixed in `lam` times the
    amount the fuel needs. At lam = 1 it is the flame of exactly enough air
    (about 1350 °C for wood), the most a fire of it can reach."""
    air = lam / AIR_O2_MASS / np.maximum(health, 0.3)
    return AMBIENT + (1.0 - chi) * (heat / ratio) / (C_GAS * (1.0 / ratio + air))


# IGNITERS, in watts of flame a surface held in them takes up (F24): a candle
# or a lamp wick is about 80-100 W of flame; a taper a few hundred; a torch
# 1-5 kW. A scene lights a fire with `World.ignite` and one of these, held for
# seconds, as a hand would; it does not drop joules into a cell.
IGNITER_W = {"candle": 80.0, "lamp": 100.0, "taper": 500.0, "torch": 2000.0}
GLOW_RED = 525.0                 # °C: the Draper point, where a hot thing first
                                 # glows by day-dark red light (the smith's
                                 # colour scale below reads up from here)
# The smith's colours (ASM; the colour of steel heated in a dim shop), °C from
# which each holds: what a person watching a bar reads off it.
GLOW_COLORS = ((525.0, "dull red"), (650.0, "dark cherry"), (750.0, "cherry"),
               (850.0, "bright cherry"), (900.0, "salmon"), (950.0, "orange"),
               (1050.0, "yellow"), (1200.0, "light yellow"), (1300.0, "white"))
EPS_FLAME = 0.4                  # a flame sheet licking a surface beside it: 35 kW/m²
                                 # onto a 300 °C face (wall flames 20-40 kW/m²)
# Flaming stops near 15% O2 (F11): a flame keeps its strength until the last
# quarter of the way from fresh air to its limit, then fails (`_o2_factor`). CHARCOAL glows on to 4-10% O2 and burns AT its surface,
# with no flame: (O2 limit, share of its heat kept by the glowing skin, and the
# kelvin above ignition it takes to reach its full rate).
GLOW = _V["GLOW"]
# Oxygen a burning face can pull in, g/(m²·s), at fresh air. A 1 m² face of a
# pool needs about 75 at 25 g/(m²·s); this only binds when the air is foul.
O2_FACE = 100.0
# STRENGTH as carrying-span: sideways overhang a material can hold before it breaks
# off, in 10 cm REFERENCE voxels (physical length — a finer world divides by scale).
# Vertical stacks always hold (v1). Ash can barely hold itself.
SPAN = _V["SPAN"]                # ROPE 1: a rope holds nothing out sideways: it hangs
# MELT — one substance, two states, three numbers: the temperature where the
# solid gives, which FLUID row it becomes, and the latent heat that pins the
# temperature there while it happens (same shape as boiling). FREEZE is the
# same table read backwards — molten metal below its point sets solid again,
# releasing the heat it swallowed. Real values: Sn 232 °C / Pb 327 / Fe 1538 —
# which is the whole story of a campfire foundry: tin runs, lead follows,
# iron only glows unless a blast lifts the bed past 1538 °C.
MELT = _in_legacy_order(_V["MELT"], (TIN, LEAD, IRON))      # tin runs first
FREEZE = {fl_id: (mt, m, lat) for m, (mt, fl_id, lat) in MELT.items()}
# IMPACT TOUGHNESS — the energy a face can absorb before BRITTLE failure, in
# kJ/m² (Charpy-class figures: glass ~0.1-0.3, wood ~5-10 along grain, mild
# steel 100+). Soft and granular things don't shatter — they deform — so flesh,
# ash and leaves carry big numbers. Landing energy is real: m·g·(drop height).
TOUGH = _V["TOUGH"]              # ROPE 30: fibre bends, it does not shatter
# THERMAL SHOCK — the ΔT one face of the material can carry before it cracks
# (soda-lime glass famously fails near 60-120 °C; ductile materials never do).
# Stone DOES spall in real fires (~400 °C) — deliberately left out until the
# house-fire scene is ready for collapsing walls.
TSHOCK = _V["TSHOCK"]
# What a thing is CALLED when a hand reaches for it — menu text, nothing else.
MATNAME = _in_legacy_order(_V["MATNAME"], (WOOD, STONE, IRON, GLASS, ASH, CHAR, LEAF,
                                           TIN, LEAD, ROPE))
# TRANSMIT — how much light gets through the MATERIAL ITSELF at one voxel thick,
# packed solid. This is genuinely a material property (glass is clear because of
# what it is, not how it was built), so unlike porosity a table is its right
# home. Only one entry is interesting; 5 cm of anything else stops light dead.
TRANSMIT = _V["TRANSMIT"]
# A PARTLY filled voxel blocks by COVERAGE, which needs no constant at all: if a
# fraction f of the cell is stuff, then f of the light hits it and (1-f) sails
# past. So a quarter-full hedge passes about three quarters of the view and a
# packed wall passes none, from the same one line — no "how opaque is foliage"
# row to argue about, and it tracks a hedge being cut back or a wall being eaten.
SEE_TAU = 2.0                    # optical depth a look can still penetrate
# A LIGHT SOURCE IS SEEN THROUGH MORE SMOKE than a lit object: Jin's visibility
# V = K / sigma has K = 8 for a light-emitting sign and K = 3 for a reflecting
# one (Jin, SFPE Handbook, visibility in smoke; (u) not re-checked). So the smoke
# part of the depth to a FLAME counts 3/8 as much. Solids still block it.
SEE_LIT_SMOKE = 3.0 / 8.0
                                 # (~13% of the light arrives)
# SMOKE_EXT — obscuration per (gram per liter) of smoke, per meter. Beer-Lambert,
# anchored to reported fire visibility: a smoke-logged room in this sim carries
# ~0.12 g/L, and this coefficient puts visibility there at ~3 m, which is what
# the fire-safety literature reports for a smoke-logged compartment. It is tied
# to the SOOT yield convention (WOOD 0.6 counts ALL visible products, ~40x real
# soot yield, because no gas-species field exists yet to hold the rest); gas
# species retires this for the standard 8.7 m^2/g against true soot mass.
SMOKE_EXT = 5.6
_SIX = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
# AIR THAT HOLDS A MILLIONTH OF A GRAM OF SOOT IS CLEAN AIR, and is not modelled
# as moving it. The first modelling decision of the active-region work rather
# than a provable skip, so it is made in the open and it was MEASURED, not
# guessed (Ruling 1's stopping rule: a region is inert when processing it would
# not change anything an outcome we care about could notice).
#
# Why it is needed at all: diffusion puts an infinitesimal trace of smoke in
# EVERY cell within ten ticks, so "smoke != 0" is true almost everywhere almost
# at once and tells you nothing — measured, 0 cells at tick 10 and 893,596 of
# 910,080 at tick 20. Where the smoke actually IS stays small for far longer.
#
# Measured in a burning town, share of ALL the smoke inside the box:
#   1e-6 g  -> 99.9%, box is 28% of the world     <- the knee, and what we use
#   1e-4 g  -> 91.1%, box is  9%
#   1e-3 g  -> 49.3%, box is  1.5%
# Nothing is destroyed below the line; it simply stops being carried, so every
# gram is still there and still counted.

SMOKE_STILL = 1e-6
SHOUT_DB = 90.0                  # a shout, one meter from the mouth
HEAR_DB = 40.0                   # where a shout stops being intelligible
# NOISE. Anything that hits anything makes a sound, and how loud is what
# fraction of the blow's energy the two materials turn into vibration they
# RING with rather than damp: glass and iron ring, stone less, wood thuds,
# flesh and leaves swallow it. The softer of the two decides — a cup dropped
# on a cushion — so it is the smaller fraction that counts. Radiated power is
# that energy over a ~50 ms impact; level at a metre is power level less 8 dB
# (a source on a floor radiating into a half-space). Named guesses, pinned by
# two anchors: a footstep, ~4 J of a walking body landing through flesh,
# comes out ~60 dB at a metre, and a pane of glass dropped on stone ~100.
SOUND = {
    "ring": _in_legacy_order(_V["SOUND_RING"], (GLASS, IRON, STONE, WOOD, CHAR, TIN, LEAD,
                                                FLESH, LEAF, ASH, ROPE)),   # ROPE: fibre swallows a blow, as leaves do
    "impact_s": 0.05,
    # A STEP is the body coming down a centimetre onto the leading foot, at
    # an ordinary walk. The energy of a heel strike goes as the square of how
    # fast the foot arrives, and that goes with walking speed — so a creep at
    # half the pace lands each foot with a quarter of the energy, 6 dB down.
    "step_drop_m": 0.01,
    "sneak_slow": 2.0,           # a creep is half a walk
    # NOTICING a noise in a quiet room is easier than making out words in it.
    "notice_db": 35.0,
    # AND WAKING TAKES MORE. Sleep studies put reliable awakening from a
    # single noise at about 55-65 dB at the ear.
    "wake_db": 60.0,
    # HOW SOUND GETS PAST THINGS (docs/realism-references.md, G1 and rows
    # A11-A15). A wall is charged by the mass law; these are the things that
    # are NOT walls, and the ways round one.
    #
    # Sound bends round an edge. ISO 9613-2 charges a path that has to go
    # round a barrier 10 log10(3 + 20 z / wavelength) dB, z being how much
    # longer the bent path is than the straight one, and never more than
    # 20 dB for one edge. The wavelength is that of 500 Hz, the band the
    # mass law is quoted at.
    "wave_m": 0.686,
    "bend_cap_db": 20.0,
    # A cell less full than this is a way through for sound going round.
    # A fuller one is a wall, and what leaks through ITS gaps is charged on
    # the straight path (by the energy the gaps let through).
    "open_fill": 0.5,
    # A body in the way is a shadow, not a wall: about 1.6 dB at 500 Hz and
    # 2.7-4 dB at 2-4 kHz for the head (interaural level difference; a proxy,
    # not a measured insertion loss).
    "shadow_db": 3.0,
    # Leaves are not a wall either. ISO 9613-2 charges dense foliage only on
    # a path of 10 m or more through it: 1 dB for 10-20 m, then 0.05 dB a
    # metre, to 200 m.
    "leaf_db_m": 0.05,
    # WORDS ARE MADE OUT OVER THE NOISE AROUND THEM. A quiet room is about
    # 35 dB; a scene can set `w.ambient_db` (a crowd, a brawl: 70-75). A
    # shout is understood down to about -3 dB signal-to-noise and lost below
    # it (0-3 dB is "satisfactory", below -3 "insufficient").
    "ambient_db": 35.0,
    "snr_db": -3.0,
}
PUSH_THROUGH = 0.4               # how full a voxel can be and still be shoved
                                 # through. A STAND-IN: what really decides this
                                 # is the body's strength against whatever holds
                                 # the material, and the force law will replace
                                 # it. Until then it at least reads the fill
                                 # instead of demanding pure air
# THE LOOKUP ARRAYS, one entry per material (fluid) id, for the per-voxel property
# gathers (`_DENS_ARR[mat]`). These used to be np.choose calls, which measured at 47%
# of ALL sim time on a 400k world: choose builds and broadcasts every branch, while
# fancy indexing gathers once. Since wave 13 they are NMAT_PAD = 64 (NFLUID_PAD = 64)
# long: the first twelve (nine) entries are the old arrays byte for byte, ICE is id 12,
# and an id with no row of the record copies AIR (NOFLUID); see mattables.py. Any array
# a cell's material (fluid) id indexes must be this long: `range(NMAT)` sizes the OLD
# dict tables only.
_VA = _matrec.legacy_views(_MOLTEN, MATREC, upto=NMAT_ROWS)    # the mechanical numbers of the 13 ids with a row
_PAD = _mattables.build(
    {"SOLID": SOLID, "FLUID": FLUID, "EMISS": EMISS, "EMISS_FLUID": EMISS_FLUID,
     "FLUID_K": FLUID_K, "MELT": MELT, "BOIL": BOIL, "LATENT": LATENT, "SKIN_T": SKIN_T,
     "NMAT": NMAT, "FLIDS": _FLIDS},
    [r for r in MATREC.by_id_list if r.id >= NMAT],
    {name: _VA[name] for name in _mattables.MECHANICAL},
    dict(zip(_matrec.MOLTEN_KEYS, _MOLTEN)))
_TOUGH_ARR = _PAD["_TOUGH_ARR"]            # kJ/m2
_TSHOCK_ARR = _PAD["_TSHOCK_ARR"]          # K one face can carry (inf: never cracks)
_SPAN_ARR = _PAD["_SPAN_ARR"]              # 10 cm reference voxels
_TRANSMIT_ARR = _PAD["_TRANSMIT_ARR"]      # light through one packed voxel (AIR 1.0)
_DENS_ARR = _PAD["_DENS_ARR"]              # g/L
_CSOLID_ARR = _PAD["_CSOLID_ARR"]          # J/(g K)
_KSOLID_ARR = _PAD["_KSOLID_ARR"]          # W/(m K)
_EMISS_ARR = _PAD["_EMISS_ARR"]
# The temperature each solid surface catches at (inf for one that never does).
_IGN_ARR = _PAD["_IGN_ARR"]
# The skin of each solid, from its diffusivity a = k / (rho c): its depth and its
# path to the core, both sqrt(a * t) in metres (SKIN_T above).
_ALPHA = _PAD["_ALPHA"]
_SKIN_DEPTH_ARR = _PAD["_SKIN_DEPTH_ARR"]
_SKIN_PATH_ARR = _PAD["_SKIN_PATH_ARR"]
# the liquids, by fluid id
_FDENS_ARR = _PAD["_FDENS_ARR"]            # g/ml
_FC_ARR = _PAD["_FC_ARR"]
_FK_ARR = _PAD["_FK_ARR"]                  # the quench-transfer column h of FLUID
_FEMISS_ARR = _PAD["_FEMISS_ARR"]
_FLK_ARR = _PAD["_FLK_ARR"]
# and of each liquid, by fluid id: a burning pool sits at its boiling point at
# the top while its bottom is still cool, and the same skin says so
_FALPHA = _PAD["_FALPHA"]
_FSKIN_DEPTH_ARR = _PAD["_FSKIN_DEPTH_ARR"]
_FSKIN_PATH_ARR = _PAD["_FSKIN_PATH_ARR"]
# MELT and FREEZE as arrays by id, for laws that gather instead of walking the dicts
# (inf / 0 where an id does not melt or set). Nothing reads them yet: ICE is in neither.
MELT_T, MELT_FL, MELT_LAT = _PAD["MELT_T"], _PAD["MELT_FL"], _PAD["MELT_LAT"]
FREEZE_T, FREEZE_MAT, FREEZE_LAT = _PAD["FREEZE_T"], _PAD["FREEZE_MAT"], _PAD["FREEZE_LAT"]
# Which row a padded id copies (a legacy id is its own).
LIKE_MAT, LIKE_FLUID = _PAD["LIKE_MAT"], _PAD["LIKE_FLUID"]
del _VA, _PAD

# THE SURFACE-FILM COLUMNS OF THE MATERIAL ROW (wave 13a, lane B5; decision D4).
# Two numbers of a surface used to be constants of the heat law: the free-convection
# film (H_CONV, one number for every surface) and the emissivity. They are DATA of the
# row now, one entry per material id (NMAT_PAD long), and the heat law reads them only
# when the world asks (`World.film_cols`, False): then a surface of any material gets
# the law its row names. With the switch off, which is every world today, the law reads
# H_CONV and `_EMISS_ARR` exactly as before.
#   _HNAT_ARR   W/(m2 K^1.25): 0 means "the constant film H_CONV"; > 0 means the
#               natural-convection law  h = _HNAT_ARR * |dT|**0.25  (ISO 7933 / 9920
#               style, Mitchell 1971 gives 2.38 for a standing person (u)).
#   _EMISS_FILM_ARR  the emissivity the law reads when the columns are live. Every
#               row equals `_EMISS_ARR` except where a row below says otherwise.
# What the FLESH row says is the law of the body (ISO 9920 gives skin 0.95-0.98 (u)); it is
# only data until 13b flips the switch for the body-heat topic.
FILM_COLS = {FLESH: {"hnat": 2.38, "eps": 0.98}}
_HNAT_ARR = np.zeros(NMAT_PAD, np.float32)
_EMISS_FILM_ARR = _EMISS_ARR.copy()
for _m, _row in FILM_COLS.items():
    _HNAT_ARR[_m] = _row["hnat"]
    _EMISS_FILM_ARR[_m] = _row["eps"]
del _m, _row
O2_PER_L = 0.28                 # grams of oxygen in one liter of fresh air
O2_RATIO = _V["O2_RATIO"]        # g O2/g fuel. ROPE 1.26: 16.5 kJ/g at 13.1 kJ per g of O2
# Every organic fuel gives about 13.1 kJ per gram of O2 (F8). FLESH was 1.4,
# which made it 5.7 kJ/g O2; its heat (8 kJ/g, mostly water) needs 0.6.
# SOOT — the VISIBLE fraction of burned mass. The rest leaves as clear gas
# (CO2 — unaccounted until gas species exist, same ledger-gap as steam).
# Charcoal's whole reason to exist is burning clean; green leaves billow.
SOOT = _V["SOOT"]                                         # solids, by mat id (ROPE 0.6: as wood, UNVERIFIED)
SOOT_OIL = 1.0                                            # (fluid ids are their
                                                          # own numbering)
O2_RATIO_OIL = 3.0
del _V                           # the record's views are bound above; the record itself is MATREC
