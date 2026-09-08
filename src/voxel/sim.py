"""The voxel physics core — an ACTUAL simulation (Ruling 1: model causes, never cases).

One voxel = a 10 cm cube = 1 liter. The world is dense numpy lattices over (x, y, z),
z up. Everything physical is a conserved amount:

  mat    what solid fills the voxel (air / wood / stone / iron)
  smass  grams of that solid remaining
  fl     which fluid sits here (none / water / oil) — one fluid per voxel for now
  fvol   milliliters of it
  E      thermal energy above ambient (J-ish); temperature is DERIVED: T = 20 + E/C

There are only four laws, and no special cases anywhere:
  1. fluids fall, then spread sideways by level difference (viscosity = how fast)
  2. heat conducts between neighbors by temperature difference
  3. combustion: where temperature >= a material's ignition point and fuel remains,
     mass converts at a temperature-driven rate and releases that material's heat.
     FIRE IS NOTHING BUT THIS HAPPENING — there is no on_fire flag.
  4. water above 100 °C boils away, and boiling COSTS energy (latent heat).

Consequences that fall out with zero case-code: wet wood will not light (its water
pins the voxel near 100 °C, below wood's 300 °C ignition); water quenches a wood
fire (cools below 300) but mostly fails against an oil fire (needs only 250 °C —
still hard — while the water boils off); a keg-sized oil pool is an inferno; fuel
runs out and the fire dies because the LAW has nothing left to convert.

Fluids are sparse (a puddle is ~dozens of voxels), so laws 1 and 4 walk an active
list in Python; laws 2 and 3 are dense numpy over the whole lattice. Deterministic:
no randomness anywhere.
"""
import numpy as np

NOFLUID, WATER, OIL, ACID, WEAK_ACID = 0, 1, 2, 3, 4
MTIN, MLEAD, MIRON = 5, 6, 7     # MOLTEN metals — a metal is one substance in
                                 # two states, and the states are ROWS

AMBIENT = 20.0                   # °C
# Voxel size is a World parameter (World(..., voxel_cm=5)). Per-voxel amounts scale
# with voxel VOLUME: capacity in ml, solid grams, the airbase thermal floor, and the
# burn rates (tuned at the 10 cm / 1 L reference size).

AIR, WOOD, STONE, IRON, FLESH, ASH, GLASS, LEAF, CHAR = 0, 1, 2, 3, 4, 5, 6, 7, 8
TIN, LEAD = 9, 10
NMAT = 11

#            density g/L  c J/g°C  k     ignition °C  burn J/g  burn g/(°C·tick)
SOLID = {
    AIR:   (1.2,   1.0,  0.6,  None, 0.0,  0.0),
    WOOD:  (600.0, 1.7,  2.0,  300., 16000.0, 0.03),   # real heat of combustion
    STONE: (2700., 0.8,  3.0,  None, 0.0,  0.0),
    IRON:  (7800., 0.45, 20.0, None, 0.0,  0.0),
    FLESH: (1000., 3.5,  1.0,  280., 8000.0, 0.01),    # a body: mostly water, can char
    ASH:   (150.,  0.8,  0.5,  None, 0.0,  0.0),       # what burning leaves behind
    GLASS: (2500., 0.8,  1.0,  None, 0.0,  0.0),       # see-through, acid-proof
    LEAF:  (200.,  2.4,  0.5,  240., 15000.0, 0.06),   # foliage: light, flashy fuel —
    CHAR:  (300.,  1.0,  0.8,  330., 30000.0, 0.01),   # what burning LEAVES: charcoal.
    TIN:   (7300., 0.23, 15.0, None, 0.0,  0.0),       # soft white metal, melts LOW
    LEAD:  (11340., 0.13, 8.0, None, 0.0,  0.0),       # heavier, melts a bit higher
}                                                      # Black forever, lights hotter,
                                                       # burns slow and fierce — embers                                                      # FRESH leaves carry water in the
                                                       # voxel and refuse to light until
                                                       # the moisture boils off (no flag)
#            density g/ml c J/g°C  viscosity(spread/tick)  ignition  burn J/g  rate  k
FLUID = {
    WATER:     (1.0,  4.2, 0.50, None, 0.0,  0.0, 30.0),   # k: boiling contact cools HARD
    OIL:       (0.9,  2.0, 0.25, 250., 40000.0, 0.03, 1.5),
    ACID:      (1.8,  1.4, 0.40, None, 0.0,  0.0, 2.0),    # "vitriol" — strong mineral acid
    WEAK_ACID: (1.05, 3.8, 0.48, None, 0.0,  0.0, 5.0),    # "aqua debilis" — vinegar-class
    MTIN:      (6.98, 0.24, 0.6, None, 0.0,  0.0, 15.0),   # molten tin: runs like water,
    MLEAD:     (10.66, 0.14, 0.6, None, 0.0, 0.0, 8.0),    # scalds like fire
    MIRON:     (7.0,  0.82, 0.5, None, 0.0,  0.0, 20.0),
}
_FLIDS = (WATER, OIL, ACID, WEAK_ACID, MTIN, MLEAD, MIRON)
_FDENS = [0.0] + [FLUID[f][0] for f in _FLIDS]
_FC = [1.0] + [FLUID[f][1] for f in _FLIDS]
_FK = [0.0] + [FLUID[f][6] for f in _FLIDS]
# REACTIONS — data, not law. The dissolution law below reads this table; a new
# acid (or ANY fluid-eats-solid interaction) is a ROW, never code. Same philosophy
# as the sandbox interaction sets (Powder Toy's element matrix), but each row is a
# conserved mass conversion: (rate g/face/tick @ ref scale, ml spent per g, J per g).
# Which acid is which: ACID is vitriol — it chars and devours organics, fizzes on
# metal, barely marks stone, and NEVER marks glass (why laboratories are glass).
# WEAK_ACID is tenfold-diluted stock: stings wood, harmless to nearly all else.
REACTIONS = {
    (ACID, WOOD):       (3.0, 0.4, 150.0),
    (ACID, FLESH):      (6.0, 0.4, 200.0),
    (ACID, ASH):        (8.0, 0.5, 50.0),
    (ACID, STONE):      (0.02, 0.6, 30.0),
    (ACID, IRON):       (0.3, 0.5, 250.0),   # metal fizzes HOT
    (ACID, LEAF):       (8.0, 0.4, 120.0),
    (ACID, CHAR):       (6.0, 0.5, 60.0),
    (WEAK_ACID, LEAF):  (0.8, 0.8, 30.0),
    (WEAK_ACID, WOOD):  (0.3, 0.8, 30.0),
    (WEAK_ACID, FLESH): (0.5, 0.8, 40.0),
    (WEAK_ACID, ASH):   (1.0, 0.8, 20.0),
}
# MISCIBLE — which fluid pairs dissolve into each other, and which species the
# blend counts as. Water and the acids are one family (an acid IS water carrying
# a reagent — POTENCY tracks how much); the blend keeps the acid's name and the
# potency mixes by volume, water counting as potency zero. Oil joins nothing.
MISCIBLE = {frozenset({WATER, ACID}): ACID,
            frozenset({WATER, WEAK_ACID}): WEAK_ACID}
# PHYSIOLOGY — what being alive means, as data. Blood chases the air it breathes;
# smoke rides along and blocks uptake (the carbon-monoxide effect); heat cooks
# skin above a threshold. Below faint the muscles let go; below death, the end.
# WILL — the reflex layer of a mind, as data: what a body notices and what it
# does about it WITHOUT any thinking. This is the scaffolding a language model
# later plugs into (it will pick from menus this layer generates); the reflexes
# themselves never wait on one.
# How the eyes sweep around the way the body is pointed, in radians — a whole
# turn in eighths. A head turns, and it turns ALL the way: a sweep of a
# quarter-circle either side leaves a body unable to notice anything behind it
# ever, and a person who walks east and stops does eventually look west.
GAZE_SWEEP = tuple(i * np.pi / 4 for i in range(8))
WILL = {"see_m": 12.0,           # how far the eyes take in the LAYOUT of a
                                 # place (not the same as noticing a flame:
                                 # you can see a room is a room much further
                                 # than you can register that it is alight)
        "fire_see_m": 4.0,       # a flame this close is NOTICED (if faced)
        "run_see_m": 2.5,        # someone bolting past this close is a warning
        "fov_deg": 120.0,        # eyes look WHERE THE FACE POINTS — a cone,
                                 # not a sphere; idle heads scan around
        "scan_every": 25,        # ticks per step of the gaze sweep
        "shout_hear_m": 14.0,    # a shout carries this far through open air
        "wall_cost_m": 4.0,      # each solid voxel in the way eats this much
        "walk_every": 3,         # ticks per walking step
        "step_m": 2.0,           # how far "straight on" and "to the left" mean.
                                 # A body that knows nowhere by name can still
                                 # say which WAY it wants to go, and that is a
                                 # real intention, not a step: the legs still
                                 # do the walking
        "arrive_m": 0.12,        # close enough to a goal to have got there.
                                 # Was 0.5 m — which is TEN voxels at 5 cm, so
                                 # a body "arrived" the moment it chose
                                 # anywhere nearby and never took a step, and
                                 # people were called safe nine voxels short of
                                 # the door they were running for
        "decide_every": 30}      # ticks before a body that stood pat will
                                 # weigh its options again — nobody
                                 # re-deliberates every fortieth of a second
# LIMBS — a body does several things at once because it HAS several parts.
# Legs go somewhere, hands hold something, a mouth speaks. This one fact is
# what makes "run for the door while shouting" expressible without anybody
# ever writing down a response called flee_shouting.
#
# There used to be such a response, next to flee and flee_answering, and the
# three of them were one intention (leave) crossed with three uses of a mouth.
# That is a cross-product wearing a table's clothes — Ruling 2 — and it grows
# multiplicatively: add "carrying a lamp" and every row splits again. The
# primitives do not multiply. Two acts may run together when they want
# different parts of the body, which is not a rule anyone writes per pair; it
# falls out of a body having parts.
LIMBS = ("legs", "hands", "mouth")
# Names for the eight ways a person can point. Labels for humans and for
# whatever is reading the menu; nothing in the sim reads them back.
_DIRS = ("east", "north-east", "north", "north-west",
         "west", "south-west", "south", "south-east")
# ACTS — everything a body knows how to do, and which part does it. `null` is
# the act of not using that part, which is always available and never a
# failure: hands doing nothing keep whatever they were holding.
ACTS = {"go":     {"limb": "legs",  "null": False},
        "jump":   {"limb": "legs",  "null": False},
        "stay":   {"limb": "legs",  "null": True},
        "hold":   {"limb": "hands", "null": False},
        "let_go": {"limb": "hands", "null": False},
        "keep":   {"limb": "hands", "null": True},
        "swing":  {"limb": "hands", "null": False},
        "say":    {"limb": "mouth", "null": False},
        "quiet":  {"limb": "mouth", "null": True}}
# REFLEXES — percept -> what each part of the body does about it. This is the
# part a character sheet edits, and it is now a sheet of a better shape: a
# brave character's "sees_fire" row can send the legs at the fire while the
# mouth still calls the warning, which the old one-response-per-percept table
# could not say at all. A person dict may carry its own "reflexes" override.
# The table is not consulted by the law: it is one POLICY among several,
# reading the same menus everything else reads.
REFLEXES = {"sees_fire":   {"legs": "go:exit", "mouth": "say:fire"},
            "hears_alarm": {"legs": "go:exit", "mouth": "say:coming"},
            "sees_runner": {"legs": "go:exit"},
            "chokes":      {"legs": "go:exit"}}
# LINES — the words a body has. A character sheet REPLACES this wholesale, so
# a character who only ever calls a warning has no answer to give.
LINES = {"fire":   "Fire! Fire! Get out!",
         "coming": "I hear you! I'm coming!"}
# Which lines are REPLIES, and to what. "I'm coming" presupposes something to
# come to, so a body that heard nothing is never offered those words. This is
# a fact about what the sentence DOES rather than about its wording, which is
# why it is kept apart from the wording: rewriting the line on a character
# sheet must not be able to lose it.
ANSWERS = {"coming": "hears_alarm"}
# What a body does with its legs when nothing is happening to it. Looking at
# what you have not seen is not filler: it is how a mind that only knows what
# it has looked at comes to know the building it is standing in, which is what
# makes running for a door it found honest later.
IDLE = {"legs": ("go:frontier", "go:roam", "go:step", "stay")}
# MENU_CAP — how many options a body may weigh at once, PER PART OF THE BODY.
# This is NOT a budget for whatever is picking: it is a model of attention. A
# person in a burning room weighs the door, the window and the child — not the
# forty places a full legality sweep would list. Modelling the limit is more
# true than pretending it is absent (Ruling 1: model the cause).
MENU_CAP = 7


class TablePolicy:
    """The reflex table, wearing the policy interface.

    A POLICY answers one question: given what the body knows and the list of
    things one part of it could legally do, which one? That is the whole
    contract —

        pick(situation: dict, menu: list[dict]) -> int   # index into menu

    called once per limb, with `situation["limb"]` saying which and
    `situation["chosen"]` carrying what the other parts have already committed
    to this tick — so a mouth may answer for legs that are already leaving
    without anybody enumerating the pair. It is the only seam a language model
    needs. This implementation reads REFLEXES; a Haiku-backed policy, and
    later a distilled local one, drop into the same slot. Costs nothing and
    waits on nothing, which is what keeps reactive play at zero model calls."""

    name = "table"

    def pick(self, situation, menu):
        limb = situation["limb"]
        want = []
        if situation["percept"] is not None:
            row = situation["reflexes"].get(situation["percept"]) or {}
            if row.get(limb):
                want = [row[limb]]
        else:
            want = list(IDLE.get(limb, ()))
        for tag in want:
            for i, opt in enumerate(menu):
                if opt["tag"] == tag:
                    return i
        return 0                     # the table named something unavailable
                                     # (no route to a door, say) — take the
                                     # null act instead of lying, which every
                                     # menu carries at index 0
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
        "legs_N": 1400.0, "crouch_m": 0.25,
        # A SHOULDER, in newton-metres. This is the only number a swing needs:
        # the segment's own inertia decides how fast it comes round, so a heavy
        # arm is slower than a light one and an arm holding something is slower
        # still, without anybody writing down a duration. ~60 N.m is an
        # unremarkable adult shoulder.
        "arm_Nm": 60.0,
        # AND A MUSCLE CANNOT PULL AT ANY SPEED. Hill's force-velocity
        # relation: the faster a muscle is already shortening the less force
        # it makes, so torque fades to nothing at a top speed. Without it an
        # arm accelerates for as long as the swing lasts, and a LIGHT arm —
        # ours is light, see scenes._person — reaches speeds no shoulder can.
        # That is what let a BARE FIST splinter a fence post. It is a cause
        # rather than a speed limit typed in: the same relation is why an arm
        # carrying something heavy swings slow.
        "arm_wmax": 15.0,        # rad/s, a shoulder unloaded
        "reach_m": 0.30}         # how far an arm goes. A grip is not
                                 # telekinesis: what is held has to stay within
                                 # reach or it is not held any more
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
FALL_SUBSTEPS = 4        # most voxels a column may drop in one tick. Matter
                         # cannot move more than a cell per support sweep — the
                         # thing it might land on has to be re-asked each time —
                         # so a fast fall runs the sweep again within the tick.
                         # Four caps it at 8 m/s, which is a two-storey drop.
_REACTIVE = {f for (f, _m) in REACTIONS}
PLUME_REACH_M = 2.4              # meters of entrainment catchment: the air a
                                 # fire's plume can actually pull in — health
                                 # and the O2 debt read this same reach.
                                 # PHYSICAL length: a finer lattice blurs over
                                 # more cells, or the same bed fire suffocated
                                 # at 5 cm that thrived at 10 cm (measured)
BOIL, LATENT = 100.0, 2260.0     # water boils at 100 °C; real latent heat, J per gram
COMBUST_CEIL = 1100.0            # flames saturate — reaction slows at flame temperature
RAD = 3e-9                       # radiation: heat leaps between neighbors ∝ T⁴ — this is
                                 # how a fire lights things it does not touch
LEAK = 0.004                     # per-tick fraction of E lost to the wider world
# STRENGTH as carrying-span: sideways overhang a material can hold before it breaks
# off, in 10 cm REFERENCE voxels (physical length — a finer world divides by scale).
# Vertical stacks always hold (v1). Ash can barely hold itself.
SPAN = {AIR: 0, WOOD: 12, STONE: 20, IRON: 30, FLESH: 6, ASH: 1, GLASS: 8,
        LEAF: 4, CHAR: 6, TIN: 16, LEAD: 10}
# MELT — one substance, two states, three numbers: the temperature where the
# solid gives, which FLUID row it becomes, and the latent heat that pins the
# temperature there while it happens (same shape as boiling). FREEZE is the
# same table read backwards — molten metal below its point sets solid again,
# releasing the heat it swallowed. Real values: Sn 232 °C / Pb 327 / Fe 1538 —
# which is the whole story of a campfire foundry: tin runs, lead follows,
# iron only glows (flames cap near 1100 °C).
MELT = {TIN: (232.0, MTIN, 59.0), LEAD: (327.0, MLEAD, 23.0),
        IRON: (1538.0, MIRON, 247.0)}
FREEZE = {fl_id: (mt, m, lat) for m, (mt, fl_id, lat) in MELT.items()}
# IMPACT TOUGHNESS — the energy a face can absorb before BRITTLE failure, in
# kJ/m² (Charpy-class figures: glass ~0.1-0.3, wood ~5-10 along grain, mild
# steel 100+). Soft and granular things don't shatter — they deform — so flesh,
# ash and leaves carry big numbers. Landing energy is real: m·g·(drop height).
TOUGH = {AIR: 1e9, WOOD: 8.0, STONE: 3.0, IRON: 150.0, FLESH: 50.0,
         ASH: 30.0, GLASS: 0.2, LEAF: 30.0, CHAR: 0.8, TIN: 60.0, LEAD: 40.0}
# THERMAL SHOCK — the ΔT one face of the material can carry before it cracks
# (soda-lime glass famously fails near 60-120 °C; ductile materials never do).
# Stone DOES spall in real fires (~400 °C) — deliberately left out until the
# house-fire scene is ready for collapsing walls.
TSHOCK = {GLASS: 120.0}
# What a thing is CALLED when a hand reaches for it — menu text, nothing else.
MATNAME = {WOOD: "wood", STONE: "stone", IRON: "iron", GLASS: "glass",
           ASH: "ash", CHAR: "char", LEAF: "leaves", TIN: "tin", LEAD: "lead"}
# TRANSMIT — how much light gets through the MATERIAL ITSELF at one voxel thick,
# packed solid. This is genuinely a material property (glass is clear because of
# what it is, not how it was built), so unlike porosity a table is its right
# home. Only one entry is interesting; 5 cm of anything else stops light dead.
TRANSMIT = {GLASS: 0.96}
# A PARTLY filled voxel blocks by COVERAGE, which needs no constant at all: if a
# fraction f of the cell is stuff, then f of the light hits it and (1-f) sails
# past. So a quarter-full hedge passes about three quarters of the view and a
# packed wall passes none, from the same one line — no "how opaque is foliage"
# row to argue about, and it tracks a hedge being cut back or a wall being eaten.
SEE_TAU = 2.0                    # optical depth a look can still penetrate
                                 # (~13% of the light arrives)
# SMOKE_EXT — obscuration per (gram per liter) of smoke, per meter. Beer-Lambert,
# anchored to reported fire visibility: a smoke-logged room in this sim carries
# ~0.12 g/L, and this coefficient puts visibility there at ~3 m, which is what
# the fire-safety literature reports for a smoke-logged compartment. It is tied
# to the SOOT yield convention (WOOD 0.6 counts ALL visible products, ~40x real
# soot yield, because no gas-species field exists yet to hold the rest); gas
# species retires this for the standard 8.7 m^2/g against true soot mass.
SMOKE_EXT = 5.6
SHOUT_DB = 90.0                  # a shout, one meter from the mouth
HEAR_DB = 40.0                   # where a shout stops being intelligible
PUSH_THROUGH = 0.4               # how full a voxel can be and still be shoved
                                 # through. A STAND-IN: what really decides this
                                 # is the body's strength against whatever holds
                                 # the material, and the force law will replace
                                 # it. Until then it at least reads the fill
                                 # instead of demanding pure air
_TOUGH_ARR = np.array([TOUGH[m] for m in range(NMAT)], np.float32)
_TSHOCK_ARR = np.array([TSHOCK.get(m, np.inf) for m in range(NMAT)], np.float32)
_DENS_ARR = np.array([SOLID[m][0] for m in range(NMAT)], np.float32)
# Lookup arrays for the per-voxel material/fluid property gathers. These used to
# be np.choose calls, which measured at 47% of ALL sim time on a 400k world:
# choose builds and broadcasts every branch, while fancy indexing gathers once.
_CSOLID_ARR = np.array([SOLID[m][1] for m in range(NMAT)], np.float32)
_KSOLID_ARR = np.array([SOLID[m][2] for m in range(NMAT)], np.float32)
_SPAN_ARR = np.array([SPAN[m] for m in range(NMAT)], np.float32)
_FDENS_ARR = np.array(_FDENS, np.float32)
_FC_ARR = np.array(_FC, np.float32)
_FK_ARR = np.array(_FK, np.float32)
_TRANSMIT_ARR = np.array([1.0 if m == AIR else TRANSMIT.get(m, 0.0)
                          for m in range(NMAT)], np.float32)
O2_PER_L = 0.28                  # grams of oxygen in one liter of fresh air
O2_RATIO = {WOOD: 1.33, FLESH: 1.4, ASH: 0.0, LEAF: 1.25, CHAR: 2.6}  # g O2/g fuel
# SOOT — the VISIBLE fraction of burned mass. The rest leaves as clear gas
# (CO2 — unaccounted until gas species exist, same ledger-gap as steam).
# Charcoal's whole reason to exist is burning clean; green leaves billow.
SOOT = {WOOD: 0.6, FLESH: 0.8, LEAF: 0.8, CHAR: 0.05}     # solids, by mat id
SOOT_OIL = 1.0                                            # (fluid ids are their
                                                          # own numbering)
O2_RATIO_OIL = 3.0


class World:
    def __init__(self, nx, ny, nz, voxel_cm=10):
        self.shape = (nx, ny, nz)
        self.scale = voxel_cm / 10.0              # length ratio to the 10 cm reference
        self.vox_l = self.scale ** 3              # liters per voxel (10 cm ref = 1 L)
        self.cap = 1000.0 * self.vox_l            # ml of fluid one voxel holds
        self.c_airbase = 50.0 * self.vox_l        # thermal floor of an "empty" voxel
        self.mat = np.zeros(self.shape, np.uint8)
        self.smass = np.zeros(self.shape, np.float32)
        self.fl = np.zeros(self.shape, np.uint8)
        self.fvol = np.zeros(self.shape, np.float32)
        self.fpot = np.zeros(self.shape, np.float32)    # POTENCY: fraction of the fluid
                                                        # that is still active reagent —
                                                        # spent/diluted acid pales toward
                                                        # inert sludge, never vanishes
        self.persons = []                               # living bodies: physiology state
        self.exits = []                                 # (x, y) places a fleeing
                                                        # body walks toward
        self.speech = []                                # (tick, name, text): what
                                                        # was said aloud, for the
                                                        # TTS/subtitle pass
        self.policy = TablePolicy()                     # WHO CHOOSES. Swap this
                                                        # for a model-backed one;
                                                        # the sim never knows the
                                                        # difference
        self.traces = []                                # one row per decision:
                                                        # situation, the whole menu
                                                        # offered, and the pick.
                                                        # The harvest starts here
        self.thermostats = []                           # (x, y, z, T): DEV blocks
                                                        # pinned at a temperature —
                                                        # heat without fire, for
                                                        # testing what heat does
        self.vfall = np.zeros(self.shape[:2], np.float32)  # m/s DOWNWARD, per column
        self.fdrop = np.zeros(self.shape[:2], np.float32)  # voxels owed but not yet
                                                        # taken — a thing slower than
                                                        # one voxel a tick still falls
        self.fallh = np.zeros(self.shape, np.float32)   # voxels of ACCUMULATED free
                                                        # fall — cashed in as impact
                                                        # energy on landing
        self.edge = np.zeros(self.shape, np.float32)    # DECLARED contact area, m²:
                                                        # "this edge concentrates its
                                                        # blow into X mm²" is a fact
                                                        # about a manufactured thing
                                                        # (Ruling 2 q1), same class as
                                                        # density. 0 = blunt: a strike
                                                        # spreads over the voxel face.
                                                        # An edge is SUB-VOXEL shape,
                                                        # the one thing 5 cm cells
                                                        # cannot draw
        self.E = np.zeros(self.shape, np.float32)
        self.smoke = np.zeros(self.shape, np.float32)   # grams of combustion gas per voxel
        self.o2 = None                                  # filled on first step, after the
        self.open_sky = True                            # scene is built (air gets fresh O2)
        self.air_region = None                          # compact id of each connected
        self._region_air = None                         # airspace; -1 in solids
        self.left_mass = 0.0                            # mass that walked OUT of
                                                        # the world through a door
        self._just_shattered = set()                    # cells emptied THIS pass:
                                                        # fragments must not plug
                                                        # the hole being made
        self._slack = None                              # support reach, cached
        self._slack_mat = None                          # ...against this layout
        self._slack_grip = frozenset()                  # ...and these held cells
        self._region_count = 0
        self._torque_solid = None                       # solids snapshot for the tip check
        self._torque_sleep = 0                          # cooldown after a wedged landing
        self.drops = []                                 # ballistic fluid parcels in flight:
        self.bodies = []                                # rigid bodies mid-TOPPLE (free
                                                        # of the grid until they land)
        self.pcell = None                               # [x,y,z, vx,vy,vz, ml, fluid]
        self.vcell = None                               # coarse gas pressure + velocity
        self.p_add = np.zeros(self.shape, np.float32)   # pressure injected this tick
        self.tick = 0

    # ── derived fields ───────────────────────────────────────────────────────
    def heat_capacity(self):
        c_solid = _CSOLID_ARR[self.mat]
        fmass = self.fvol * _FDENS_ARR[self.fl]
        c_fluid = _FC_ARR[self.fl]
        return self.smass * c_solid + fmass * c_fluid + self.c_airbase

    def T(self):
        return AMBIENT + self.E / self.heat_capacity()

    def porosity(self):
        """How freely gas crosses a voxel: THE SPACE THAT IS LEFT IN IT.

        Not a material row. `fill()` has always said it — "frac < 1 is a PARTIAL
        voxel: a stick is mostly air inside its cube" — and `smass` has always
        carried it; the gas laws simply threw it away by asking `mat == AIR`, a
        binary question about a continuous field. Solid mass over what the
        material would weigh packed solid IS the occupied fraction; what is left
        is void, and void is what gas moves through.

        So a shut door leaks because the SCENE says it fills 99% of its cells —
        a true statement about that door's geometry, at the one resolution the
        lattice cannot draw (a 1 cm undercut in a 5 cm voxel). Wood is not
        declared leaky; this door is. Nothing here knows what a door is.

        What falls out for free, none of it written: rubble and thatch breathe,
        a wall half-eaten by acid breathes through the loss, and wood GROWS more
        permeable as it burns away, feeding the fire that is thinning it."""
        packed = self.smass / np.maximum(_DENS_ARR[self.mat] * self.vox_l, 1e-9)
        void = np.clip(1.0 - packed, 0.0, 1.0)
        drowned = self.fvol > 0.5 * self.cap      # a flooded gap does not breathe
        return np.where(drowned, np.minimum(void, 0.02), void)

    def solid(self):
        return self.mat != AIR

    def burning(self):
        """Where combustion is HAPPENING right now (derived, not stored)."""
        T = self.T()
        out = (self.fl == OIL) & (self.fvol > 0) & (T >= FLUID[OIL][3])
        for m, spec in SOLID.items():
            if spec[3] is not None:
                out |= (self.mat == m) & (self.smass > 0) & (T >= spec[3])
        return out

    # ── placing things (scene setup) ─────────────────────────────────────────
    def fill(self, x0, x1, y0, y1, z0, z1, material, frac=1.0, edge=0.0):
        """frac < 1 is a PARTIAL voxel: a stick is mostly air inside its cube.
        edge > 0 declares a working edge: the m² this thing concentrates a blow
        into (an axe bit ~2e-4). A property of the OBJECT, not the material."""
        self.mat[x0:x1, y0:y1, z0:z1] = material
        self.smass[x0:x1, y0:y1, z0:z1] = SOLID[material][0] * frac * self.vox_l
        self.edge[x0:x1, y0:y1, z0:z1] = edge

    def pour(self, x, y, z, fluid, ml, pot=1.0):
        if fluid not in _REACTIVE:
            pot = 0.0                    # water has no potency to bring
        if self.mat[x, y, z] == AIR and self._mix(fluid, self.fl[x, y, z]) is not None:
            old = float(self.fvol[x, y, z])
            add = min(self.cap - old, ml)
            if add <= 0:
                return
            self.fpot[x, y, z] = (self.fpot[x, y, z] * old + pot * add) / (old + add)
            self.fl[x, y, z] = self._mix(fluid, self.fl[x, y, z])
            self.fvol[x, y, z] = old + add

    @staticmethod
    def _mix(f_in, f_at):
        """Species the target voxel holds after f_in joins f_at — or None if the
        two refuse each other (oil into water). Same fluid or empty: trivial."""
        if f_at in (NOFLUID, f_in):
            return f_in
        return MISCIBLE.get(frozenset({int(f_in), int(f_at)}))

    # ── the four laws ────────────────────────────────────────────────────────
    def _move_fluid(self, src, dst, ml):
        """Move ml of fluid src→dst, carrying its share of the source's heat and
        its potency: the destination blends by volume, so strong acid dripping
        into a water pool leaves a paler, weaker acid — never a hidden flag."""
        f = self.fl[src]
        f_out = self._mix(f, self.fl[dst])
        if f_out is None:
            return
        dens, c_f = FLUID[f][0], FLUID[f][1]
        c_src = (self.smass[src] * SOLID[int(self.mat[src])][1]        # scalar capacity of
                 + self.fvol[src] * dens * c_f + self.c_airbase)       # ONE voxel — never
        share = (ml * dens * c_f) / max(float(c_src), 1e-6)            # the whole lattice
        dE = float(self.E[src]) * min(share, 1.0)
        old = float(self.fvol[dst])
        self.fpot[dst] = (self.fpot[dst] * old + float(self.fpot[src]) * ml) / (old + ml)
        self.fvol[src] -= ml
        self.E[src] -= dE
        if self.fvol[src] <= 0.01:
            self.fvol[src] = 0.0
            self.fl[src] = NOFLUID
            self.fpot[src] = 0.0
        self.fl[dst] = f_out
        self.fvol[dst] += ml
        self.E[dst] += dE

    def _fly_drops(self):
        """Ballistic fluid parcels (the Noita trick): a splash throws real droplets
        that ARC through the air under gravity and rejoin the grid where they land.
        Deterministic — direction comes from the splash geometry, never RNG."""
        if not self.drops:
            return
        nx, ny, nz = self.shape
        kept = []
        for d in self.drops:
            d[5] -= 0.25                                 # gravity
            d[0] += d[3]; d[1] += d[4]; d[2] += d[5]
            x = min(max(int(round(d[0])), 0), nx - 1)
            y = min(max(int(round(d[1])), 0), ny - 1)
            z = min(max(int(round(d[2])), 0), nz - 1)
            landed = (self.mat[x, y, z] != AIR or self.fvol[x, y, z] > 1.0 or z == 0
                      or self.mat[x, y, z - 1] != AIR
                      or self.fvol[x, y, z - 1] > 0.7 * self.cap)
            if not landed:
                kept.append(d)
                continue
            zz = z                                       # deposit in the first open cell
            while zz < nz - 1 and (self.mat[x, y, zz] != AIR
                                   or self._mix(d[7], self.fl[x, y, zz]) is None):
                zz += 1
            fo = self._mix(d[7], self.fl[x, y, zz])
            if self.mat[x, y, zz] == AIR and fo is not None:
                room = self.cap - self.fvol[x, y, zz]
                add = min(d[6], max(room, 0.0))
                if add > 0:
                    old = float(self.fvol[x, y, zz])
                    self.fpot[x, y, zz] = (self.fpot[x, y, zz] * old
                                           + d[8] * add) / (old + add)
                    self.fl[x, y, zz] = fo
                    self.fvol[x, y, zz] += add
                    d[6] -= add
            if d[6] > 0.5:                               # remainder keeps falling
                d[0], d[1], d[2] = x, y, min(zz + 1, nz - 1)
                d[3] = d[4] = d[5] = 0.0
                kept.append(d)
        self.drops = kept

    def _law_head(self):
        """PRESSURE, hydrostatic half: p = rho*g*h. Within ONE connected body of
        the same liquid, the free surface seeks a single level — columns above
        the body's mean surface bleed into columns below it. This is why water
        poured into one tank RISES in the tank connected to it by a pipe.
        v1 limit: the transfer rate is fixed — a narrow pipe does not yet
        throttle it (that needs the full momentum field)."""
        for f in _FLIDS:
            cells = np.argwhere((self.fl == f) & (self.fvol > 1.0) & (self.mat == AIR))
            if len(cells) < 4:
                continue
            index = {(int(c[0]), int(c[1]), int(c[2])): i for i, c in enumerate(cells)}
            parent = list(range(len(cells)))

            def find(i):
                while parent[i] != i:
                    parent[i] = parent[parent[i]]
                    i = parent[i]
                return i

            for (x, y, z), i in index.items():
                for nb in ((x + 1, y, z), (x, y + 1, z), (x, y, z + 1)):
                    j = index.get(nb)
                    if j is not None:
                        ri, rj = find(i), find(j)
                        if ri != rj:
                            parent[ri] = rj
            bodies = {}
            for (x, y, z), i in index.items():
                bodies.setdefault(find(i), {}).setdefault((x, y), []).append(z)
            for cols in bodies.values():
                if len(cols) < 2:
                    continue
                surf = {}
                for (x, y), zlist in cols.items():
                    zt = max(zlist)
                    surf[(x, y)] = zt + float(self.fvol[x, y, zt]) / self.cap
                mean = sum(surf.values()) / len(surf)
                give = sorted(((h, c) for c, h in surf.items() if h > mean + 0.1),
                              reverse=True)
                take = []                                # only columns that CAN rise —
                for (x, y), h in surf.items():           # a roofed pipe is not a taker
                    if h >= mean - 0.1:
                        continue
                    tz = max(cols[(x, y)])
                    if self.fvol[x, y, tz] >= self.cap - 0.5:
                        tz += 1
                        if tz >= self.shape[2] or self.mat[x, y, tz] != AIR \
                                or self.fl[x, y, tz] not in (NOFLUID, f):
                            continue                     # sealed under a roof: skip
                    take.append((h, (x, y), tz))
                take.sort(key=lambda t: t[0])
                for (hg, (gx, gy)), (ht, (tx, ty), tz) in zip(give, take):
                    gz = max(cols[(gx, gy)])
                    amount = min(0.25 * self.cap * (hg - ht),
                                 float(self.fvol[gx, gy, gz]) - 1.0,
                                 self.cap - float(self.fvol[tx, ty, tz]))
                    if amount > 0.5:
                        self._move_fluid((gx, gy, gz), (tx, ty, tz), amount)

    def _law_flow(self):
        self._fly_drops()
        # density layering: a heavier fluid directly ABOVE a lighter one sinks — they
        # swap. This is WHY water fails against an oil fire: it slips underneath and
        # the burning oil rides up on top of it (no rule about fires anywhere).
        up, lo = (slice(None), slice(None), slice(1, None)), (slice(None), slice(None), slice(None, -1))
        dens = _FDENS_ARR[self.fl]
        m = ((self.mat[up] == AIR) & (self.mat[lo] == AIR)
             & (self.fvol[up] > 1.0) & (self.fvol[lo] > 1.0) & (dens[up] > dens[lo]))
        if m.any():
            for arr in (self.fl, self.fvol, self.E, self.fpot):
                a, b = arr[up], arr[lo]
                a[m], b[m] = b[m], a[m].copy()
        nx, ny, nz = self.shape
        active = np.argwhere(self.fvol > 0)
        order = np.argsort(active[:, 2])                    # settle lowest first
        for x, y, z in active[order]:
            if self.fvol[x, y, z] <= 0:
                continue
            if self.mat[x, y, z] != AIR:                # moisture held INSIDE a solid
                continue                                # (fresh leaves, soaked wood)
                                                        # stays put — it only boils away
            if z > 0 and self.mat[x, y, z - 1] == AIR:      # 1a: fall
                below = (x, y, z - 1)
                if self._mix(self.fl[x, y, z], self.fl[below]) is not None:
                    room = self.cap - self.fvol[below]
                    if room > 0:
                        self._move_fluid((x, y, z), below, min(self.fvol[x, y, z], room))
            if self.fvol[x, y, z] <= 0:
                continue
            resting = z == 0 or self.mat[x, y, z - 1] != AIR or self.fvol[x, y, z - 1] >= self.cap - 0.01
            if not resting:
                continue
            visc = FLUID[self.fl[x, y, z]][2]               # 1b: spread by level difference
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                p = (x + dx, y + dy, z)
                if not (0 <= p[0] < nx and 0 <= p[1] < ny) or self.mat[p] != AIR:
                    continue
                if self._mix(self.fl[x, y, z], self.fl[p]) is None:
                    continue
                diff = self.fvol[x, y, z] - self.fvol[p]
                if diff > 1.0:
                    self._move_fluid((x, y, z), p, min(diff * visc / 4.0, self.cap - self.fvol[p]))

    def _law_stir(self):
        """Potency evens out WITHIN a standing pool — reaction heat and density
        currents stir a real liquid, so the layer working the wood face is fed
        by the potent bulk above it instead of exhausting and playing dead."""
        for f in _REACTIVE:
            liq = (self.fl == f) & (self.fvol > 1.0) & (self.mat == AIR)
            if not liq.any():
                continue
            for axis in range(3):
                a = [slice(None)] * 3; b = [slice(None)] * 3
                a[axis], b[axis] = slice(None, -1), slice(1, None)
                a, b = tuple(a), tuple(b)
                m = liq[a] & liq[b]
                if not m.any():
                    continue
                lo = np.minimum(self.fvol[a][m], self.fvol[b][m])
                d = 0.1 * (self.fpot[a][m] - self.fpot[b][m]) * lo
                self.fpot[a][m] -= d / self.fvol[a][m]     # active ml conserved:
                self.fpot[b][m] += d / self.fvol[b][m]     # what one loses, one gains

    def _law_support(self):
        """Gravity + STRENGTH for solids. Support starts at the ground; a voxel sitting
        directly on a supported voxel INHERITS ITS CARRIER'S slack (clamped to its own
        span — ash on a stone lintel is still just ash); support that travels SIDEWAYS
        spends 1 span per hop. When the span runs out, the overhang BREAKS and falls —
        so an ash slab crumbles where the wood that held it burned away. Slack only ever
        DECAYS along a path: a plank resting on a shelf cannot lend the shelf reach it
        never had (the old full-restore-on-carry rule made any two-layer slab an
        infinite girder — one table leg carried 89% of the tabletop). SPAN is physical
        length: measured in 10 cm reference voxels, converted by the world's scale.
        Falling voxels carry their heat and clinging fluid, landing on solids or pools."""
        steps, moved_tot, airborne = None, None, None
        for _ in range(FALL_SUBSTEPS):
            r = self._support_once(steps)
            if r is None:
                break
            steps, moved, airborne = r
            moved_tot = moved if moved_tot is None else (moved_tot | moved)
            if not steps.any():
                break
        return self._cash_impacts(moved_tot, airborne)

    def _support_once(self, steps):
        """One sweep of the support law. Returns the columns' remaining fall
        budget, what moved, and what is STILL IN THE AIR — or None when
        nothing is falling any more.

        A column with several voxels of fall owed this tick comes back through
        here for each of them: matter may not skip over what it might have
        landed on, so the support question is asked again between every cell."""
        solid = self.mat != AIR
        hang = solid[:, :, 1:] & ~solid[:, :, :-1]           # any solid with air below?
        if not hang.any():
            return None                          # last tick's landings still pay
        # The slack field is a relaxation run to fixpoint over the WHOLE grid, and
        # it was 45% of all sim time. But it is a pure function of the material
        # layout: if not one voxel changed material since last tick, last tick's
        # answer is still exactly right. Comparing the layout costs ~0.2 ms
        # against ~134 ms to recompute, and it is correct by construction rather
        # than by remembering to invalidate — in a standing room nothing moves,
        # so this is skipped almost every tick.
        grip = self._grip_cells() if self.persons else frozenset()
        if self._slack_mat is not None \
                and np.array_equal(self._slack_mat, self.mat) \
                and grip == self._slack_grip:
            slack = self._slack
        else:
            slack = self._relax_slack(solid, grip)
            self._slack_mat, self._slack = self.mat.copy(), slack
            self._slack_grip = grip
        falling = solid & (slack < 0)
        if not falling.any():
            return None
        if steps is None:                        # once a tick: how fast is each
            steps = self._fall_speed(falling)    # column actually going?
        go = falling & (steps > 0)[:, :, None]
        if not go.any():
            return np.zeros_like(steps), np.zeros(self.shape, bool), falling
        moved = self._settle(solid, go)
        return np.maximum(steps - 1, 0), moved, falling

    def _fall_speed(self, falling):
        """How fast each falling column is going, in whole voxels this tick.

        THE AIR PUSHES BACK. A falling column shoves air out of its way, and
        what that costs is drag — half rho C_d A v-squared — against a weight
        of m g. The two balance at a terminal speed, and because the AREA is
        one voxel face however deep the column is, the same matter spread thin
        falls slower than the same matter balled up. That is the whole of a
        parachute, and there is nothing about parachutes in it: a canopy is a
        large area with very little mass behind it, which is a SHAPE, and the
        sim has always known the shape.

        Before this everything unsupported dropped exactly one voxel a tick,
        so a feather and an anvil fell alike and nothing ever accelerated."""
        vox_m = 0.1 * self.scale
        col = falling.any(axis=2)
        kg = (self.smass * falling).sum(axis=2) / 1000.0
        area = vox_m * vox_m
        drag = (0.5 * AIR_DENS * DRAG_CD * area
                * self.vfall * self.vfall / np.maximum(kg, 1e-9))
        self.vfall = np.where(
            col, np.maximum(self.vfall + (GRAVITY - drag) * TICK_S, 0.0), 0.0)
        self.fdrop = np.where(col, self.fdrop + self.vfall * TICK_S / vox_m, 0.0)
        steps = np.minimum(np.floor(self.fdrop), FALL_SUBSTEPS).astype(np.int32)
        self.fdrop -= steps
        return steps

    def _relax_slack(self, solid, grip=frozenset()):
        """Support reach, spread from the ground until it stops changing.

        A GRIP IS AN EDGE IN THIS GRAPH. Support relaxes from the ground up
        through material, and a hand is not material — so nothing on the
        lattice could ever hold a hanging man, and everything a hand carried
        had to rest on something. Cells a footed body holds within its
        strength (`_grip_cells`) seed as supported, exactly like the ground
        does: the load hangs from the body, and the body stands on its feet."""
        # STRENGTH FADES WITH THE MASS THAT IS LEFT. Span used to be read
        # straight off the material, so a trunk voxel eaten to a tenth of
        # itself carried like sound timber right up to the tick it became ash.
        # Reach is a property of the beam that is actually there: full span
        # while at least half the material remains, then falling away with
        # what is left. A burning tree now comes down onto its own fire,
        # a fire-thinned lintel drops its load, and scene `frac` fills (a
        # stick at 0.6) keep their strength — which is why the knee is at a
        # half and not at one.
        packed = np.clip(self.smass / np.maximum(
            _DENS_ARR[self.mat] * self.vox_l, 1e-9), 0.0, 1.0)
        span = np.round(_SPAN_ARR[self.mat] / self.scale
                        * np.clip(packed / 0.5, 0.0, 1.0)).astype(np.int16)
        slack = np.full(self.shape, -1, np.int16)
        slack[:, :, 0][solid[:, :, 0]] = span[:, :, 0][solid[:, :, 0]]
        if grip:
            gi = tuple(np.asarray(list(grip), np.int64).T)
            slack[gi] = np.where(solid[gi],
                                 np.maximum(slack[gi], span[gi]), slack[gi])
        while True:
            cand = np.full(self.shape, -1, np.int16)         # carried from directly below:
            cand[:, :, 1:] = np.where(                       # inherit the carrier's slack
                solid[:, :, 1:] & (slack[:, :, :-1] >= 0), slack[:, :, :-1], -1)
            nb = np.full(self.shape, -1, np.int16)           # sideways/hanging: span - 1
            nb[1:, :, :] = np.maximum(nb[1:, :, :], slack[:-1, :, :] - 1)
            nb[:-1, :, :] = np.maximum(nb[:-1, :, :], slack[1:, :, :] - 1)
            nb[:, 1:, :] = np.maximum(nb[:, 1:, :], slack[:, :-1, :] - 1)
            nb[:, :-1, :] = np.maximum(nb[:, :-1, :], slack[:, 1:, :] - 1)
            nb[:, :, 1:] = np.maximum(nb[:, :, 1:], slack[:, :, :-1] - 1)
            nb[:, :, :-1] = np.maximum(nb[:, :, :-1], slack[:, :, 1:] - 1)
            new = np.where(solid, np.maximum(cand, nb), -1).astype(np.int16)
            new = np.minimum(new, span)          # nothing projects MORE support than its
            new = np.maximum(new, slack)         # own material can carry (ash caps at 1)
            if (new == slack).all():
                break
            slack = new
        return slack

    def _settle(self, solid, falling):
        """Everything the support law does once it knows what is unsupported."""
        fdens = _FDENS_ARR[self.fl].astype(np.float32) * 1000.0   # fluid, g/L
        sdens = self.smass / self.vox_l                                  # this solid, g/L
        arrived = np.zeros(self.shape[:2], bool)             # which columns moved this tick
        moved = np.zeros(self.shape, bool)
        for z in range(1, self.shape[2]):                    # bottom-up: columns stay whole
            pool = self.fvol[:, :, z - 1] >= 0.5 * self.cap
            # a solid DENSER than the pool below SINKS into it and the displaced
            # fluid rides up into the vacated cell (the swap moves it) — stone
            # drops through a pond, wood is lighter than water and rests afloat
            sinks = sdens[:, :, z] > fdens[:, :, z - 1]
            m = falling[:, :, z] & (self.mat[:, :, z - 1] == AIR) & (~pool | sinks)
            arrived |= m
            if not m.any():
                continue
            for arr in (self.mat, self.smass, self.fl, self.fvol, self.E,
                        self.fpot, self.edge):
                lo, hi = arr[:, :, z - 1], arr[:, :, z]
                lo[m], hi[m] = hi[m], lo[m].copy()           # the cell and the air swap
            # the drop height RIDES with the voxel and grows — except through
            # liquid, where drag bleeds it away fast (a pond is a cushion, and
            # it also forgives the speed the fall brought INTO the water)
            self.fallh[:, :, z - 1][m] = np.where(
                pool[m], self.fallh[:, :, z][m] * 0.4, self.fallh[:, :, z][m] + 1.0)
            self.fallh[:, :, z][m] = 0.0
            moved[:, :, z - 1] |= m
            # SPLASH: the displaced fluid (now sitting where the solid was) is
            # shoved outward + thrown as droplets where neighbors have room
            for x, y in np.argwhere(m & pool):
                self._splash(int(x), int(y), z)
        # a LIGHTER solid that just STOPPED on a pool still SLAPS it — a log
        # hitting a pond splashes even though it will float. Only columns that
        # actually MOVED this tick slap (a resting float never re-splashes).
        sdens2 = self.smass / self.vox_l
        landed = (self.mat[:, :, 1:] != AIR) & (self.mat[:, :, :-1] == AIR) \
            & (self.fvol[:, :, :-1] >= 0.5 * self.cap) \
            & (sdens2[:, :, 1:] <= _FDENS_ARR[self.fl[:, :, :-1]] * 1000.0)
        for x, y, zl in np.argwhere(landed):
            if arrived[x, y] and zl + 1 < self.shape[2]:
                self._splash(int(x), int(y), int(zl), int(zl) + 1)
        return moved            # the caller cashes impacts once, after the last
                                # sweep of the tick — a thing part-way through a
                                # multi-voxel fall has not landed on anything

    def _cash_impacts(self, moved, airborne=None):
        """IMPACT: a voxel that fell and now CANNOT move has landed — its
        stored drop is cashed in as m·g·h against the material's toughness.
        Brittle stuff (glass, char) shatters; wood just thuds; a fall cushioned
        by a pond arrives with no height to cash."""
        self._just_shattered = set()
        rest = (self.mat != AIR) & (self.fallh > 0.5)
        if moved is not None:
            rest &= ~moved
        # STILL FALLING IS NOT LANDED. "Did not move this tick" used to be a
        # safe reading of "has come to rest", because everything unsupported
        # moved a voxel every single tick. Now that a thing can be going slower
        # than a voxel a tick, a light body in open air stops for a tick
        # between steps — and cashing its drop then shattered it in MID-AIR:
        # measured, a glass block bursting at z=8 on the way down to a pond it
        # never reached. What matters is whether the floor is under it.
        if airborne is not None:
            rest &= ~airborne
        if not rest.any():
            return
        vox_m = 0.1 * self.scale
        cells = np.argwhere(rest)
        e = (self.smass[rest] / 1000.0) * 9.81 * self.fallh[rest] * vox_m
        thr = _TOUGH_ARR[self.mat[rest]] * 1000.0 * vox_m * vox_m
        flaw = np.array([0.7 + 0.6 * self._flaw01(*c) for c in cells], np.float32)
        thr = thr * flaw                                 # every piece has its own
        for (x, y, z), broke, ov, joules in zip(cells, e > thr,
                                                e / np.maximum(thr, 1e-9), e):
            if broke:                                    # worst flaw — no two break
                self._shatter(int(x), int(y), int(z), float(ov))   # alike
            # AND WHAT IT LANDED ON TAKES THE SAME BLOW. Newton's third law: the
            # impulse is shared, so the floor is tested against its OWN toughness
            # with the same energy. Before this a falling anvil could only ever
            # hurt itself — it went through a glass table without marking it,
            # which is the wrong way round. A struck cell that shatters drops
            # whatever it was holding, so collapses cascade on their own.
            zb = int(z) - 1
            if zb < 0 or self.mat[int(x), int(y), zb] == AIR:
                continue
            below = int(self.mat[int(x), int(y), zb])
            tb = (_TOUGH_ARR[below] * 1000.0 * vox_m * vox_m
                  * (0.7 + 0.6 * self._flaw01(int(x), int(y), zb)))
            if joules > tb:
                self._shatter(int(x), int(y), zb, float(joules / max(tb, 1e-9)))
        self.fallh[rest] = 0.0

    def _splash(self, x, y, z, tz=None):
        """Shove the struck fluid cell outward — into lateral neighbors at level
        tz (its own level for a sinking entry; the level ABOVE for a surface
        slap, where the water erupts up-and-out around what hit it) — and throw
        droplet parcels. Deep in a pool every neighbor is full, so nothing
        sprays — the quiet deep falls out for free."""
        if tz is None:
            tz = z
        vol, f = float(self.fvol[x, y, z]), self.fl[x, y, z]
        if vol <= 1.0 or f == NOFLUID:
            return
        nx, ny = self.shape[0], self.shape[1]
        room = [(dx, dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
                if 0 <= x + dx < nx and 0 <= y + dy < ny
                and self.mat[x + dx, y + dy, tz] == AIR
                and self._mix(f, self.fl[x + dx, y + dy, tz]) is not None
                and self.fvol[x + dx, y + dy, tz] < 0.7 * self.cap]
        for (dx, dy) in room:
            part = min(0.4 * vol / len(room),
                       self.cap - float(self.fvol[x + dx, y + dy, tz]))
            if part > 0.5:
                self._move_fluid((x, y, z), (x + dx, y + dy, tz), part)
            if len(self.drops) < 2000:                   # and real droplets FLY:
                dml = 0.2 * vol / len(room)              # an arcing crown parcel
                if dml > 1.0:
                    self.fvol[x, y, z] -= dml
                    self.drops.append([float(x + dx), float(y + dy), float(tz + 1),
                                       0.8 * dx, 0.8 * dy, 0.55, dml, int(f),
                                       float(self.fpot[x, y, z])])

    @staticmethod
    def _flaw01(x, y, z):
        """Deterministic per-cell 'flaw' in [0, 1) — real brittle failure is
        governed by the worst microscopic flaw in the piece (Weibull's
        statistics), so neither the breaking point nor the fragments should
        come out uniform. No RNG: the same drop always breaks the same way."""
        return ((x * 73 + y * 151 + z * 211) % 17) / 16.0

    def _shatter(self, x, y, z, over=1.0):
        """Brittle failure: the voxel's cohesion is gone. The cell empties and
        its mass scatters as UNEVEN fragments of the same material — heavier
        shards near the break, slighter ones farther out, and the more the
        impact exceeded the material's toughness (`over`), the farther the
        spray reaches. Any fluid it held SPILLS where the cell stood. Boxed in
        on every side, the rubble stays wedged — a crushed cell deep inside a
        wall is still a wall of rubble."""
        m0 = int(self.mat[x, y, z])
        mass, E0 = float(self.smass[x, y, z]), float(self.E[x, y, z])
        f, fv, fp = int(self.fl[x, y, z]), float(self.fvol[x, y, z]), float(self.fpot[x, y, z])
        nx, ny, nz = self.shape
        ring1 = ((1, 0), (-1, 0), (0, 1), (0, -1))
        ring2 = ((2, 0), (-2, 0), (0, 2), (0, -2), (1, 1), (1, -1), (-1, 1), (-1, -1))
        offsets = ring1 + (ring2 if over >= 2.0 else ())
        targets = []                                     # (cell, weight): heavy
        if z > 0 and self.mat[x, y, z - 1] == AIR:       # shards drop, light ones
            targets.append(((x, y, z - 1), 1.6))         # fly — never evenly
        # ...but never INTO a hole that is being made in the same instant. When
        # a row of cells breaks together under one blow, each was scattering
        # fragments into its neighbours' just-emptied cells and refilling them,
        # so a shattered plate stayed a plate and an anvil sat on the wreckage
        # of the table it had just smashed. Fragments fall out of a break; they
        # do not queue up to plug it.
        fresh = self._just_shattered
        for dx, dy in offsets:
            tx, ty = x + dx, y + dy
            d = abs(dx) + abs(dy)
            if (tx, ty, z) in fresh:
                continue
            if 0 <= tx < nx and 0 <= ty < ny and self.mat[tx, ty, z] == AIR:
                targets.append(((tx, ty, z),
                                (0.4 + 1.2 * self._flaw01(tx, ty, z)) / (0.5 + d)))
        if not targets:
            return
        self._just_shattered.add((x, y, z))
        self.mat[x, y, z] = AIR
        self.smass[x, y, z] = 0.0
        self.E[x, y, z] = 0.0
        self.fallh[x, y, z] = 0.0
        self.edge[x, y, z] = 0.0             # rubble is blunt: an edge is made,
        self.fl[x, y, z] = NOFLUID           # and breaking unmakes it
        self.fvol[x, y, z] = 0.0
        self.fpot[x, y, z] = 0.0
        wsum = sum(wt for _t, wt in targets)
        for (tx, ty, tz), wt in targets:
            self.mat[tx, ty, tz] = m0
            self.smass[tx, ty, tz] += mass * wt / wsum
            self.E[tx, ty, tz] += E0 * wt / wsum
        if fv > 0 and f != NOFLUID:              # the held fluid pours out where
            self.fl[x, y, z] = f                 # the cell used to be — the flow
            self.fvol[x, y, z] = fv              # law takes it from there
            self.fpot[x, y, z] = fp

    def _law_crack(self):
        """THERMAL SHOCK — a brittle solid spanning a steep temperature
        difference cracks: the hot side expands, the cold side refuses.
        TSHOCK is data (the ΔT one face can carry); the conduction field
        already knows every face's ΔT, so this law only reads it. A cold
        vat licked by flame bursts; a gently warmed one is fine."""
        if not (np.isin(self.mat, list(TSHOCK)).any()):
            return
        T = self.T()
        hits = set()
        for axis in range(3):
            a = [slice(None)] * 3; b = [slice(None)] * 3
            a[axis], b[axis] = slice(None, -1), slice(1, None)
            a, b = tuple(a), tuple(b)
            dT = np.abs(T[a] - T[b])
            for s_, d_ in ((a, b), (b, a)):          # the brittle cell is at d_
                m = dT > _TSHOCK_ARR[self.mat[d_]]
                if not m.any():
                    continue
                off = [0, 0, 0]
                off[axis] = 1 if d_ is b else 0
                for x, y, z in np.argwhere(m):
                    hits.add((int(x) + off[0], int(y) + off[1], int(z) + off[2]))
        for x, y, z in hits:
            if self.mat[x, y, z] != AIR:
                self._shatter(x, y, z)

    def _topple(self, cells, axis, s, pivot, zb, owner=None):
        """PROMOTE a tipping cluster to a free rigid body (the sandbox lineage's
        promote/demote protocol): its voxels leave the grid and the body leans
        about the pivot line with real pendulum acceleration — slow at first,
        crashing at the end — until it lands and re-rasterizes."""
        cells = np.asarray(cells, np.float32)
        idx = tuple(cells.astype(np.int64).T)
        mats = self.mat[idx].copy()
        masses = self.smass[idx].copy()
        Es = self.E[idx].copy()
        fls = self.fl[idx].copy()                        # held moisture rides along
        fvols = self.fvol[idx].copy()
        fpots = self.fpot[idx].copy()
        edges = self.edge[idx].copy()
        self.mat[idx] = AIR                              # lift the body off the grid
        self.smass[idx] = 0.0
        self.E[idx] = 0.0
        self.fl[idx] = NOFLUID
        self.fvol[idx] = 0.0
        self.fpot[idx] = 0.0
        self.fallh[idx] = 0.0
        self.edge[idx] = 0.0
        m = np.maximum(masses, 1e-6)
        d_lat = cells[:, axis] - pivot
        d_z = cells[:, 2] - zb
        dxc = float((m * d_lat).sum() / m.sum()) * s     # COM lever, toward the overhang
        dzc = float((m * d_z).sum() / m.sum())
        body = {
            "cells": cells, "mats": mats, "masses": masses, "Es": Es,
            "fls": fls, "fvols": fvols, "fpots": fpots, "edges": edges,
            "axis": axis, "s": s, "pivot": float(pivot), "zb": float(zb),
            "theta": 0.0, "omega": 0.0,
            "phi0": max(float(np.arctan2(max(dxc, 0.1), max(dzc, 0.5))), 0.02),
            "L": max(float(np.hypot(dxc, dzc)), 2.0),
            # WHOSE BODY THIS IS, if it is anybody's. The sim is the thing that
            # picked this flesh up and it is the thing that puts it down, so it
            # KNOWS where the person went: making identity re-derive that from
            # adjacency afterwards is throwing away an answer we already hold,
            # and re-deriving it is what handed a fainting person their
            # rescuer's body.
            "owner": owner}
        # WEDGED check: if the body cannot even BEGIN to lean (its first sliver
        # of rotation already collides), it is stuck against something — put it
        # back and leave it until the world changes around it. Without this, a
        # blocked vat tip-landed-retipped every tick, forever.
        probe = np.round(self._body_pose(body, 0.1)).astype(np.int64)
        nx, ny, nz = self.shape
        inb = ((probe[:, 0] >= 0) & (probe[:, 0] < nx) & (probe[:, 1] >= 0)
               & (probe[:, 1] < ny) & (probe[:, 2] >= 0) & (probe[:, 2] < nz))
        wedged = int((~inb).sum()) + int((self.mat[tuple(probe[inb].T)] != AIR).sum())
        if wedged > max(2, len(cells) // 50):
            self.mat[idx] = mats                         # restore; snapshot stays valid,
            self.smass[idx] = masses                     # so no re-check until the
            self.E[idx] = Es                             # lattice actually changes
            self.fl[idx] = fls
            self.fvol[idx] = fvols
            self.fpot[idx] = fpots
            return
        self.bodies.append(body)
        self._torque_solid = None

    def _launch(self, cells, vel, owner=None):
        """PROMOTE a cluster to a body in free FLIGHT — the same promote/demote
        protocol as a topple, but travelling rather than turning.

        This is what having a velocity buys. A topple could only ever swing
        about a pivot it was already touching, so nothing in the sim could
        leave the ground: not a jump, not a thrown stone, not a swung axe. A
        flying body carries m/s, is pulled on by gravity, is pushed back on by
        the air it has to shove aside, and pays what it has left as energy
        when it arrives."""
        cells = np.asarray(cells, np.int64)
        idx = tuple(cells.T)
        body = {"cells": cells.astype(np.float32),
                "mats": self.mat[idx].copy(), "masses": self.smass[idx].copy(),
                "Es": self.E[idx].copy(), "fls": self.fl[idx].copy(),
                "fvols": self.fvol[idx].copy(), "fpots": self.fpot[idx].copy(),
                "edges": self.edge[idx].copy(),
                "fly": True, "vel": np.asarray(vel, np.float64).copy(),
                "off": np.zeros(3), "owner": owner,
                "axis": 0, "s": 1, "pivot": 0.0, "zb": 0.0,
                "theta": 0.0, "omega": 0.0, "phi0": 0.02, "L": 2.0}
        for arr, zero in ((self.mat, AIR), (self.smass, 0.0), (self.E, 0.0),
                          (self.fl, NOFLUID), (self.fvol, 0.0),
                          (self.fpot, 0.0), (self.fallh, 0.0),
                          (self.edge, 0.0)):
            arr[idx] = zero
        self.bodies.append(body)
        self._torque_solid = None
        self._slack_mat = None
        return body

    def _limb_cells(self, p, name, own=None):
        """Where one named part of this body actually is, right now.

        Segments are kept as offsets from the body's own corner, so they ride
        along when it walks, is shoved, or is carried out. A limb that has been
        swung and come to rest somewhere new updates its own offsets on landing.
        Returns None when there is no such limb, or nothing left of it."""
        segs = p.get("segs")
        if not segs or name not in segs:
            return None
        if own is None:
            comp, sl = self._person_cells(p)
            if comp is None or not comp.any():
                return None
            own = np.argwhere(comp)
            own[:, 0] += sl[0].start or 0
            own[:, 1] += sl[1].start or 0
        limb = (segs[name] + np.asarray(own).min(axis=0)).astype(np.int64)
        nx, ny, nz = self.shape
        ok = ((limb[:, 0] >= 0) & (limb[:, 0] < nx) & (limb[:, 1] >= 0)
              & (limb[:, 1] < ny) & (limb[:, 2] >= 0) & (limb[:, 2] < nz))
        limb = limb[ok]
        if not len(limb):
            return None
        limb = limb[self.mat[tuple(limb.T)] == FLESH]
        return limb if len(limb) else None

    def _swing(self, p, name, toward=None):
        """Put a LIMB in motion about its joint, driven by a muscle.

        This is the same rotation a toppling tree does — `_body_pose` has always
        turned a set of cells about an arbitrary pivot — with two things
        changed: the pivot is INSIDE the body, and the acceleration comes from a
        muscle instead of from gravity. That is the whole of what a shoulder is.

        It is not animation, and the distinction is sharp in a sim where
        everything reads the lattice. The arm's voxels leave their cells and
        arrive in new ones, so while it is coming round the arm really is
        somewhere else: it blocks what it now occupies, its mass sits where it
        now sits, and whatever stops it is paid the rotational energy it had.
        A pose that only changed the picture would need a picture to change,
        and there isn't one — the renderer reads `mat` like everything else."""
        segs, joints = p.get("segs"), p.get("joints")
        if not segs or name not in segs:
            return None
        # WHICH WAY. The arm hangs from the shoulder, so turning it about the
        # sideways axis sweeps it forward; the sign is which forward.
        dx, dy = toward if toward else p.get("facing", (1.0, 0.0))
        axis = 0 if abs(dx) >= abs(dy) else 1
        s = -1 if (dx if axis == 0 else dy) >= 0 else 1
        comp, sl = self._person_cells(p)
        if comp is None or not comp.any():
            return None
        own = np.argwhere(comp)
        own[:, 0] += sl[0].start or 0
        own[:, 1] += sl[1].start or 0
        limb = self._limb_cells(p, name, own)
        if limb is None:
            return None                       # the limb is not there any more
        # A HELD THING SWINGS WITH THE ARM — that is most of what holding a
        # tool is for. Its cells join the limb's rigid body, so its mass slows
        # the swing (Hill does the rest) and its edge is what arrives.
        obj = self._held_cells(p)
        if obj is not None and self._in_reach(limb, obj):
            limb = np.concatenate([limb, obj])
        jx, jy, jz = (joints[name] + own.min(axis=0)).astype(np.int64)
        vox_m = 0.1 * self.scale
        kg = self.smass[tuple(limb.T)] / 1000.0
        r = (np.abs(limb[:, axis] - float(jx if axis == 0 else jy))
             + np.abs(limb[:, 2] - float(jz))) * vox_m
        inertia = float((kg * r * r).sum())
        if inertia <= 0.0:
            return None
        b = self._launch(limb, (0.0, 0.0, 0.0), owner=p["name"])
        b["fly"] = False                      # it turns, it does not travel
        b["part"] = True                      # and it is PART of somebody
        b["axis"], b["s"] = axis, s
        b["pivot"] = float(jx if axis == 0 else jy)
        b["zb"] = float(jz)
        b["theta"], b["omega"] = 0.0, 0.0
        b["Nm"], b["I"] = BODY["arm_Nm"], inertia
        b["seg"] = name
        b["phi0"], b["L"] = 0.02, 2.0
        return b

    def _fly_pose(self, b, off=None):
        """Where a flying body's cells are, at its current offset."""
        off = b["off"] if off is None else off
        return b["cells"] + off

    def _drag_a(self, b):
        """Deceleration from the air, in m/s^2, along the way it is going.

        Frontal area is COUNTED, not declared: the cells that face the
        direction of travel are the ones doing the shoving. A body spread
        broadside to its own fall presents many of them and slows; the same
        matter end-on presents few and does not. That is a parachute, and also
        why a plank falls differently flat than edge-on."""
        v = float(np.linalg.norm(b["vel"]))
        if v < 1e-6:
            return np.zeros(3)
        vox_m = 0.1 * self.scale
        ax = int(np.argmax(np.abs(b["vel"])))            # the way it is going
        face = {tuple(int(c[i]) for i in range(3) if i != ax)
                for c in b["cells"]}                     # its shadow, that way
        area = len(face) * vox_m * vox_m
        kg = max(float(b["masses"].sum()) / 1000.0, 1e-9)
        mag = 0.5 * AIR_DENS * DRAG_CD * area * v * v / kg
        return -mag * (b["vel"] / v)

    def _body_pose(self, b, theta):
        """Rotated float positions of a body's cells at lean angle theta."""
        c, ax, s = b["cells"], b["axis"], b["s"]
        d_lat = c[:, ax] - b["pivot"]
        d_z = c[:, 2] - b["zb"]
        ct, st = np.cos(theta), np.sin(theta)
        lat = b["pivot"] + d_lat * ct + s * d_z * st
        zz = b["zb"] - s * d_lat * st + d_z * ct
        out = c.copy()
        out[:, ax] = lat
        out[:, 2] = zz
        return out

    def _law_bodies(self):
        """Advance every mid-topple body one tick; land the ones that arrive."""
        if not self.bodies:
            return
        nx, ny, nz = self.shape
        vox_m = 0.1 * self.scale
        still = []
        for b in self.bodies:
            if b.get("fly"):
                b["vel"] += (np.array([0.0, 0.0, -GRAVITY])
                             + self._drag_a(b)) * TICK_S
                off = b["off"] + b["vel"] * TICK_S / vox_m
                pose = np.round(self._fly_pose(b, off)).astype(np.int64)
                inb = ((pose[:, 0] >= 0) & (pose[:, 0] < nx) & (pose[:, 1] >= 0)
                       & (pose[:, 1] < ny) & (pose[:, 2] >= 0) & (pose[:, 2] < nz))
                hit = int((~inb).sum()) + \
                    int((self.mat[tuple(pose[inb].T)] != AIR).sum())
                # ANY contact is an arrival, for a flier. A topple tolerates a
                # few grazing cells because it is pivoting through its own
                # neighbourhood; a body in flight that tolerates them lands
                # driven INTO whatever it hit, and _land_body then shoves the
                # buried cells upward as debris — which comes out through the
                # top. Measured: a person landing from a jump arrived with 15
                # of their 342 voxels rearranged, and what a person has at the
                # top of them is their head. Stop at the last clear pose and
                # let the support law set them down the rest of the way.
                #
                # BUT A WALL IS NOT A FLOOR. What a wall takes is the sideways
                # speed; what stops a fall is something underneath. So a hit is
                # retried with the horizontal part removed: if straight down is
                # clear, the body is SCRAPING PAST — the wall keeps the
                # sideways speed and the fall goes on — and it arrives only
                # when that too is blocked. Known softness: the sideways energy
                # is absorbed by the wall unpaid; the honest version spends it
                # on the struck face, and matters once throwing does.
                if hit > 0:
                    landed = True
                    if abs(b["vel"][2]) > 1e-6 and \
                            (abs(b["vel"][0]) > 1e-6 or abs(b["vel"][1]) > 1e-6):
                        vv = np.array([0.0, 0.0, b["vel"][2]])
                        off_v = b["off"] + vv * TICK_S / vox_m
                        pv = np.round(self._fly_pose(b, off_v)).astype(np.int64)
                        inb_v = ((pv[:, 0] >= 0) & (pv[:, 0] < nx)
                                 & (pv[:, 1] >= 0) & (pv[:, 1] < ny)
                                 & (pv[:, 2] >= 0) & (pv[:, 2] < nz))
                        if not int((~inb_v).sum()) and \
                                not int((self.mat[tuple(pv[inb_v].T)] != AIR).sum()):
                            b["vel"] = vv
                            b["off"] = off_v
                            still.append(b)
                            landed = False
                    if landed:
                        self._land_body(b, None)  # arrived: pay what it carried
                else:
                    b["off"] = off
                    still.append(b)
                continue
            if b.get("Nm"):
                # DRIVEN by a muscle: angular acceleration is torque over the
                # limb's own inertia, so a loaded arm comes round slower and
                # nobody types how long a swing takes.
                w_s = b["omega"] / TICK_S                # rad/tick -> rad/s
                fade = max(0.0, 1.0 - w_s / max(BODY["arm_wmax"], 1e-9))
                w_s += (b["Nm"] * fade / b["I"]) * TICK_S
                b["omega"] = w_s * TICK_S
                theta_next = min(b["theta"] + b["omega"], np.pi / 2)
                pose = np.round(self._body_pose(b, theta_next)).astype(np.int64)
                inb = ((pose[:, 0] >= 0) & (pose[:, 0] < nx) & (pose[:, 1] >= 0)
                       & (pose[:, 1] < ny) & (pose[:, 2] >= 0) & (pose[:, 2] < nz))
                blocked = pose[inb][self.mat[tuple(pose[inb].T)] != AIR]
                if len(blocked) or int((~inb).sum()) or \
                        theta_next >= np.pi / 2 - 1e-6:
                    self._land_body(b, b["theta"], struck=blocked)
                else:
                    b["theta"] = theta_next
                    still.append(b)
                continue
            alpha = (1.0 / b["L"]) * max(float(np.sin(b["theta"] + b["phi0"])), 0.02)
            b["omega"] = min(b["omega"] + alpha, 0.35)   # and never faster than the
            if b["theta"] < 0.01:                        # wedge probe on the FIRST step
                b["omega"] = min(b["omega"], 0.06)
            theta_next = min(b["theta"] + b["omega"], np.pi / 2)
            pose = self._body_pose(b, theta_next)
            pi_ = np.round(pose).astype(np.int64)
            inb = ((pi_[:, 0] >= 0) & (pi_[:, 0] < nx) & (pi_[:, 1] >= 0)
                   & (pi_[:, 1] < ny) & (pi_[:, 2] >= 0) & (pi_[:, 2] < nz))
            hit = int((~inb).sum()) + int((self.mat[tuple(pi_[inb].T)] != AIR).sum())
            tol = max(2, len(b["cells"]) // 50)          # same tolerance as the wedge
            if hit > tol or theta_next >= np.pi / 2 - 1e-6:  # probe — a graze is not
                self._land_body(b, b["theta"] if hit > tol else theta_next)   # a landing
            else:
                b["theta"] = theta_next
                still.append(b)
        self.bodies = still

    def _land_body(self, b, theta, struck=None):
        """Demote: rasterize the body back onto the grid at its landing pose.
        Blocked cells pile upward as debris; the support law settles the rest.
        A landing at a barely-leant angle means the body is WEDGED — the torque
        law sleeps briefly so it retries occasionally, not every tick."""
        if theta is not None and theta < 0.15:   # a flying body has no lean to
            self._torque_sleep = max(self._torque_sleep, 20)   # be wedged at
        if b.get("part"):
            # A LIMB IS HELD AT ITS JOINT. Rasterising it wherever the swing
            # stopped leaves the far end nowhere near the body, and a lump of
            # flesh that touches nothing is not an arm any more — measured, a
            # person went from one piece to three and left 18 voxels of hand on
            # the floor. Hanging IS the rest state of something pivoted at one
            # end, so the arm comes back to it once the blow is spent. The
            # swing is a transient; what it is attached to is not.
            pose = np.round(b["cells"]).astype(np.int64)
        else:
            pose = np.round(self._fly_pose(b) if b.get("fly")
                            else self._body_pose(b, theta)).astype(np.int64)
        nx, ny, nz = self.shape
        put = []
        # IMPACT: the body's centre of mass FELL — that energy is paid out over
        # the cells that strike first, each judged by its own material's
        # toughness. A keeling vat of glass bursts on the shelf edge; a keeling
        # tree of wood just booms.
        vox_m = 0.1 * self.scale
        mtot = max(float(b["masses"].sum()), 1e-6)
        if b.get("Nm"):
            # A SWING ARRIVES WITH ITS ROTATIONAL ENERGY, and it arrives on
            # WHAT STOPPED IT — not on whatever happens to be underneath. A
            # falling thing is judged by the floor it meets; a swung thing is
            # judged by the thing it hit, which is the only difference between
            # leaning on an axe and swinging one.
            w_s = b["omega"] / max(TICK_S, 1e-9)          # rad/tick -> rad/s
            joules = 0.5 * b["I"] * w_s * w_s
        elif b.get("fly"):
            # IT ARRIVES WITH WHAT IT IS CARRYING. A flying body's blow is its
            # kinetic energy, not the height it happens to have lost — which is
            # the whole difference between a stone dropped on a plank and the
            # same stone thrown at it, and the difference between leaning on an
            # axe and swinging one.
            v = float(np.linalg.norm(b["vel"]))
            joules = 0.5 * (mtot / 1000.0) * v * v
        else:
            dh = float(((b["masses"] * b["cells"][:, 2]).sum()
                        - (b["masses"] * pose[:, 2].astype(np.float32)).sum()) / mtot) * vox_m
            joules = max(dh, 0.0) * 9.81 * (mtot / 1000.0)
        # A BLOW WOUNDS what it lands on and what lands. Flesh does not shatter
        # — TOUGH[FLESH] is a deforms-not-fragments number — but tissue takes
        # real damage far below that, and the same energy-per-contact-area
        # arithmetic that bursts glass is what breaks a leg. Anything beyond
        # the bruise threshold is absorbed by the BODY, not the cell.
        bruise = BODY["bruise_kJm2"] * 1000.0 * vox_m ** 2
        if struck is not None and len(struck):
            # spend it on the struck cells, each against its own toughness.
            # A small contact concentrates the same energy into a larger
            # stress. THE EDGE IS THE OBJECT'S OWN CLAIM to be smaller still:
            # an axe bit is sub-millimetre, which no 5 cm lattice can draw, so
            # a manufactured edge DECLARES the area it concentrates a blow
            # into — the same class of fact as its density. Which cell of the
            # swung thing made contact is below the lattice's resolution too,
            # so the sharpest edge the body carries is the one that strikes;
            # an axe is swung edge-first, which is a fact about how tools are
            # held, not a case about axes.
            ed = b["edges"][b["edges"] > 0] if "edges" in b else []
            a_hit = min(float(min(ed)) if len(ed) else vox_m ** 2, vox_m ** 2)
            per = joules / len(struck)
            for (sx, sy, sz) in struck:
                if self.mat[sx, sy, sz] == FLESH:
                    # flesh DEFORMS, it does not fragment — TOUGH[FLESH] said
                    # so in words while a big enough blow shattered feet into
                    # debris anyway. The whole overage is the person's wound:
                    # a swung fist, a landing body — the STRUCK person pays too
                    b_hit = BODY["bruise_kJm2"] * 1000.0 * a_hit
                    if per > b_hit:
                        self._wound(self._person_at(
                            (int(sx), int(sy), int(sz))), per - b_hit)
                    continue
                thr = (_TOUGH_ARR[self.mat[sx, sy, sz]] * 1000.0 * a_hit
                       * (0.7 + 0.6 * self._flaw01(int(sx), int(sy), int(sz))))
                if per > thr:
                    self._shatter(int(sx), int(sy), int(sz), per / max(thr, 1e-9))
        if b.get("part"):
            # the blow has already been spent on what stopped it. An arm does
            # take the same impulse back — Newton's third — but paying the
            # whole energy TWICE, once to the post and once to the hand, is not
            # that law, it is double counting.
            joules = 0.0
        contact = pose[:, 2] <= pose[:, 2].min() + 1
        e_per = joules / max(int(contact.sum()), 1)
        sore = 0.0                          # energy the faller's own flesh took
        order = np.argsort(pose[:, 2])      # kept: put[k] belongs to order[k],
        for i in order:                     # so a landing can be told apart
                                            # into flesh and carried tool
            dx = min(max(int(pose[i, 0]), 0), nx - 1)
            dy = min(max(int(pose[i, 1]), 0), ny - 1)
            dz = min(max(int(pose[i, 2]), 0), nz - 1)
            while dz < nz - 1 and self.mat[dx, dy, dz] != AIR:
                dz += 1                                  # blocked: pile upward as debris
            put.append((dx, dy, dz))
            self.mat[dx, dy, dz] = b["mats"][i]
            self.smass[dx, dy, dz] = b["masses"][i]
            self.E[dx, dy, dz] += b["Es"][i]
            self.edge[dx, dy, dz] = b["edges"][i]
            if b["fvols"][i] > 0:
                self.fl[dx, dy, dz] = b["fls"][i]
                self.fvol[dx, dy, dz] = b["fvols"][i]
                self.fpot[dx, dy, dz] = b["fpots"][i]
            if contact[i]:
                if b["mats"][i] == FLESH:
                    if e_per > bruise:      # flesh deforms, never fragments:
                        sore += e_per - bruise   # the overage is the wound
                    continue
                thr_i = (_TOUGH_ARR[b["mats"][i]] * 1000.0 * vox_m ** 2
                         * (0.7 + 0.6 * self._flaw01(dx, dy, dz)))
                if e_per > thr_i:
                    self._shatter(dx, dy, dz, e_per / thr_i)
        if sore > 0.0 and b.get("owner") is not None and not b.get("part"):
            # A FALL HURTS. _land_body always paid 1/2 m v² against what a body
            # STRUCK and nothing against the body — a man dropped onto stone
            # landed whole and entirely unbothered, and "pulled off a cliff"
            # was a change of address. The energy his own contact cells took
            # is his, by the same third-law sharing as the struck side.
            for q in self.persons:
                if q["name"] == b["owner"]:
                    self._wound(q, sore)
                    if sore / BODY["hurt_J"] >= 0.02:
                        q["events"].append(
                            f"t{self.tick}: {q['name']} lands hard")
                    break
        self._torque_solid = None                        # lattice changed: recheck
        if b.get("part") and b.get("seg") and b.get("owner") is not None:
            put_arr = np.array(put)
            mats_put = b["mats"][order]
            for q in self.persons:            # the limb came to rest somewhere
                if q["name"] == b["owner"] and q.get("segs") and q.get("_own"):
                    origin = np.min(np.array(list(q["_own"])), axis=0)
                    # only the FLESH is the limb — a swung tool came along for
                    # the ride and is not part of anyone's arm
                    q["segs"][b["seg"]] = put_arr[mats_put == FLESH] - origin
                    if q.get("held") and (mats_put != FLESH).any():
                        q["held"]["cell"] = tuple(map(
                            int, put_arr[mats_put != FLESH][0]))
                    break                     # say WHERE, or the next swing
                                              # reaches for cells that have gone
        if b.get("owner") is not None and not b.get("part"):   # hand the person
            for q in self.persons:                       # their own body, exactly
                if q["name"] == b["owner"]:              # where it came down
                    q["_own"] = frozenset(put)
                    q["_claim_tick"] = None
                    q["anchor"] = (int(round(float(np.mean([c[0] for c in put])))),
                                   int(round(float(np.mean([c[1] for c in put])))))
                    break

    def _person_at(self, cell):
        """Whose flesh is this? Ownership, not adjacency — the claim map."""
        for q in self.persons:
            if cell in (q.get("_own") or frozenset()):
                return q
        return None

    def _wound(self, p, joules):
        """Impact energy a body absorbed beyond what tissue takes whole.
        Accumulates like the burn integral and reads against the same
        thresholds; wounds do not heal here any more than burns do."""
        if p is None or not p["alive"]:
            return
        p["hurt"] = min(p["hurt"] + joules / BODY["hurt_J"], 1.0)

    def bodies_array(self):
        """Every mid-flight body voxel as (x, y, z, mat) for the renderer."""
        if not self.bodies:
            return np.zeros((0, 4), np.float32)
        parts = []
        for b in self.bodies:
            pose = self._fly_pose(b) if b.get("fly") \
                else self._body_pose(b, b["theta"])
            parts.append(np.column_stack([pose, b["mats"].astype(np.float32)]))
        return np.concatenate(parts).astype(np.float32)

    def _law_torque(self):
        """TIPPING. An object is a connected cluster of same-material solid voxels
        (a wood tree on a stone floor is its own body, RESTING on the floor).
        Two checks per rooted body, and one verdict — where the weight hangs
        (center of mass) must sit over what carries it:
          1. BASE: COM vs the support footprint on the ground. Fails -> the whole
             body topples 90 degrees about the footprint edge, toward the overhang.
          2. WAIST: wherever the body NARROWS sharply (a section under 40% of the
             plane above it — an axe notch, table legs under a top), everything
             above stands on that section; its COM must sit over the section's
             bbox. Fails -> the part above tears off and topples about the waist
             edge. This is why a tree with a thin hinge left FALLS TOWARD THE
             NOTCH, and a table that lost its east legs tips east.
        Runs only when the solid lattice changed. v1: instant quarter-turn, no
        arc; same-material contact merges bodies (wood-on-wood is one body)."""
        if self._torque_sleep > 0:
            self._torque_sleep -= 1
            return
        solid = self.mat != AIR
        if self._torque_solid is not None and bool((solid == self._torque_solid).all()):
            return
        self._torque_solid = solid.copy()
        lab = np.where(solid, np.arange(solid.size, dtype=np.int32).reshape(self.shape), -1)
        while True:                                      # same-material flood labeling
            new = lab.copy()
            for axis in range(3):
                a = [slice(None)] * 3; b = [slice(None)] * 3
                a[axis], b[axis] = slice(None, -1), slice(1, None)
                a, b = tuple(a), tuple(b)
                same = solid[a] & solid[b] & (self.mat[a] == self.mat[b])
                new[a] = np.where(same, np.maximum(new[a], lab[b]), new[a])
                new[b] = np.where(same, np.maximum(new[b], lab[a]), new[b])
            if (new == lab).all():
                break
            lab = new
        _, comp = np.unique(lab[solid], return_inverse=True)
        L = np.full(self.shape, -1, np.int32)
        L[solid] = comp.astype(np.int32)
        n = int(comp.max()) + 1 if comp.size else 0
        if n == 0:
            return
        nz = self.shape[2]
        xs, ys, zs = np.indices(self.shape)
        w = self.smass
        cx, cy, cz = xs[solid], ys[solid], zs[solid]
        cw = w[solid]
        # per-(cluster, z) stats in one bincount family
        ids = comp * nz + cz
        cnt = np.bincount(ids, minlength=n * nz).reshape(n, nz)
        msz = np.bincount(ids, weights=cw, minlength=n * nz).reshape(n, nz)
        mxz = np.bincount(ids, weights=cw * cx, minlength=n * nz).reshape(n, nz)
        myz = np.bincount(ids, weights=cw * cy, minlength=n * nz).reshape(n, nz)
        bx0 = np.full((n, nz), 10 ** 6); bx1 = np.full((n, nz), -(10 ** 6))
        by0 = np.full((n, nz), 10 ** 6); by1 = np.full((n, nz), -(10 ** 6))
        np.minimum.at(bx0, (comp, cz), cx); np.maximum.at(bx1, (comp, cz), cx)
        np.minimum.at(by0, (comp, cz), cy); np.maximum.at(by1, (comp, cz), cy)
        mass = msz.sum(axis=1)
        com_x = mxz.sum(axis=1) / np.maximum(mass, 1e-9)
        com_y = myz.sum(axis=1) / np.maximum(mass, 1e-9)
        below_L = np.full(self.shape, -2, np.int32)      # what each voxel STANDS on
        below_L[:, :, 1:] = L[:, :, :-1]
        below_solid = np.zeros_like(solid)
        below_solid[:, :, 1:] = solid[:, :, :-1]
        contact = solid & ((zs == 0) | (below_solid & (below_L != L)))
        cL = L[contact]
        fx0 = np.full(n, 10 ** 6); fx1 = np.full(n, -(10 ** 6))
        fy0 = np.full(n, 10 ** 6); fy1 = np.full(n, -(10 ** 6))
        fz = np.full(n, 10 ** 6)
        np.minimum.at(fx0, cL, xs[contact]); np.maximum.at(fx1, cL, xs[contact])
        np.minimum.at(fy0, cL, ys[contact]); np.maximum.at(fy1, cL, ys[contact])
        np.minimum.at(fz, cL, zs[contact])
        rooted = fx1 >= 0
        # RIDERS: a cluster of another material whose entire support stands on ONE
        # body (a leaf canopy around a trunk, a stick on a table) belongs to that
        # body when it tips — it rides along instead of hovering behind.
        sup_lo = np.full(n, 10 ** 9); sup_hi = np.full(n, -1)
        cnz = contact & (zs > 0)
        np.minimum.at(sup_lo, L[cnz], below_L[cnz])
        np.maximum.at(sup_hi, L[cnz], below_L[cnz])
        has_ground = np.zeros(n, bool)
        cg = contact & (zs == 0)
        if cg.any():
            has_ground[np.unique(L[cg])] = True

        def rider_mask(li):
            rs = np.nonzero(~has_ground & (sup_lo == li) & (sup_hi == li))[0]
            rs = rs[rs != li]
            return np.isin(L, rs) if len(rs) else None

        def whose(cells):
            """Which PEOPLE this cluster is made of.

            Flesh touching flesh is one cluster to the labeller, so a cluster
            can be nobody, one person, or two people in contact."""
            if not self.persons or not len(cells):
                return []
            want = set(map(tuple, np.asarray(cells, np.int64).tolist()))
            out = []
            for q in self.persons:
                self._person_cells(q)             # make sure the claim is today's
                if (q.get("_own") or frozenset()) & want:
                    out.append(q)
            return out

        def outside(comx, comy, x0, x1, y0, y1):
            ox = comx - x1 if comx > x1 + 0.5 else (comx - x0 if comx < x0 - 0.5 else 0.0)
            oy = comy - y1 if comy > y1 + 0.5 else (comy - y0 if comy < y0 - 0.5 else 0.0)
            return ox, oy

        tipped = 0
        for li in np.nonzero(rooted)[0]:
            if tipped >= 4:
                break
            # 1. BASE check: whole-body COM over the ground footprint
            ox, oy = outside(com_x[li], com_y[li], fx0[li], fx1[li], fy0[li], fy1[li])
            if ox != 0 or oy != 0:
                rm = rider_mask(li)
                body = (L == li) if rm is None else ((L == li) | rm)
                cells = np.argwhere(body)
                mine = whose(cells)
                if any(q["alive"] and q["awake"] for q in mine):
                    continue                     # A PERSON ON THEIR FEET HOLDS
                                                 # THEMSELVES UP. This law asks
                                                 # where a rigid lump's weight
                                                 # hangs, which is the right
                                                 # question for a wall and the
                                                 # wrong one for someone stood
                                                 # at a lip: it tipped them over
                                                 # for leaning. Balance is what
                                                 # the living do that the dead
                                                 # do not, and the sim already
                                                 # has the other half — a body
                                                 # that goes slack is toppled by
                                                 # _collapse, one whose footing
                                                 # is gone falls by _law_footing
                if 2 <= len(cells) <= 20000:
                    ow = mine[0]["name"] if mine else None
                    if abs(ox) >= abs(oy):
                        self._topple(cells, 0, 1 if ox > 0 else -1,
                                     int(fx1[li] if ox > 0 else fx0[li]),
                                     int(fz[li]), owner=ow)
                    else:
                        self._topple(cells, 1, 1 if oy > 0 else -1,
                                     int(fy1[li] if oy > 0 else fy0[li]),
                                     int(fz[li]), owner=ow)
                    tipped += 1
                continue
            # 2. WAIST check: sharp narrowings carry everything above them
            c = cnt[li]
            for lz in range(int(fz[li]), nz - 1):
                if c[lz] == 0 or c[lz + 1] == 0 or c[lz] >= 0.4 * c[lz + 1]:
                    continue
                am = msz[li, lz + 1:].sum()              # the body above the waist
                if am <= 0:
                    continue
                acx = mxz[li, lz + 1:].sum() / am
                acy = myz[li, lz + 1:].sum() / am
                ox, oy = outside(acx, acy, bx0[li, lz], bx1[li, lz],
                                 by0[li, lz], by1[li, lz])
                if ox == 0 and oy == 0:
                    continue
                rm = rider_mask(li)
                body = (L == li) if rm is None else ((L == li) | rm)
                cells = np.argwhere(body & (zs > lz))
                mine = whose(cells)
                if any(q["alive"] and q["awake"] for q in mine):
                    break                        # nor does a standing body tear
                                                 # in half at the waist: a person
                                                 # narrows at the neck, the waist
                                                 # and both knees, and every one
                                                 # of those reads as a notch
                if 2 <= len(cells) <= 20000:
                    ow = mine[0]["name"] if mine else None
                    if abs(ox) >= abs(oy):
                        self._topple(cells, 0, 1 if ox > 0 else -1,
                                     int(bx1[li, lz] if ox > 0 else bx0[li, lz]),
                                     lz + 1, owner=ow)
                    else:
                        self._topple(cells, 1, 1 if oy > 0 else -1,
                                     int(by1[li, lz] if oy > 0 else by0[li, lz]),
                                     lz + 1, owner=ow)
                    tipped += 1
                break                                    # one waist verdict per body

    def _law_conduct(self):
        T, C = self.T(), self.heat_capacity()
        k = _KSOLID_ARR[self.mat].astype(np.float32)
        # hot gas convects: rising, churning air moves heat far faster than still air.
        # Modeled as conductivity growing with temperature (standard trick, not a case).
        hot_air = self.mat == AIR
        k[hot_air] += np.maximum(T[hot_air] - AMBIENT, 0.0) / 60.0
        fk = _FK_ARR[self.fl].astype(np.float32)
        wet = (self.fvol > 0.1 * self.cap) & (self.mat == AIR)      # OPEN fluid on a solid:
        solid = self.mat != AIR                          # contact is governed by the FLUID
        for axis in range(3):                            # (a boiling film, not the timber)
            a = [slice(None)] * 3; b = [slice(None)] * 3
            a[axis], b[axis] = slice(None, -1), slice(1, None)
            a, b = tuple(a), tuple(b)
            kpair = np.minimum(k[a], k[b])
            contact = np.where(wet[a] & solid[b], fk[a], 0.0) + np.where(wet[b] & solid[a], fk[b], 0.0)
            kpair = np.maximum(kpair, contact)
            Tk = ((T[a] + T[b]) / 2.0) + 273.0              # radiation rides on the pair too:
            kr = RAD * Tk * Tk * Tk                         # ~0 at room temp, strong at flame temp
            # conduction scales with contact area / distance = LINEAR in voxel size;
            # radiation scales with contact area = size squared (found the hard way:
            # unscaled pair-k bled small voxels dry and nothing could ever ignite)
            q = (kpair * self.scale + kr * self.scale * self.scale) * (T[a] - T[b])
            q = np.clip(q, -0.2 * C[b] * np.abs(T[a] - T[b]),
                        0.2 * np.minimum(C[a], C[b]) * np.abs(T[a] - T[b]))
            self.E[a] -= q
            self.E[b] += q

    def _law_rise(self):
        """Buoyancy: hot gas rises. A hot air voxel hands much of its excess heat to the
        air above it; at the open top of the world, the plume leaves entirely. THIS is why
        a doused fire can stay out — the reheating cloud escapes upward instead of sitting."""
        T = self.T()
        lo = (slice(None), slice(None), slice(None, -1))
        up = (slice(None), slice(None), slice(1, None))
        both_air = (self.mat[lo] == AIR) & (self.mat[up] == AIR)
        dT = T[lo] - T[up]
        # only the GAS'S share of the cell's heat rises (c_airbase, the air's own
        # capacity) — a pool of liquid is not a plume, its heat stays put. Found the
        # hard way: rising 0.35*C_total blew a lamp's oil heat skyward and no taper
        # could ever light a pool. For pure air the two are identical.
        q = np.where(both_air & (dT > 0), 0.35 * self.c_airbase * dT, 0.0).astype(np.float32)
        self.E[lo] -= q
        self.E[up] += q
        top = (slice(None), slice(None), -1)
        sky = (self.mat[top] == AIR) & (self.fvol[top] < 0.1 * self.cap)
        self.E[top][sky] *= 0.5                          # open sky: the plume is gone

    def _gas_access(self):
        """Fire breathes: a voxel can only burn with gas nearby — itself or a neighbor
        that is open air not drowned in fluid. A water blanket or a full seal SMOTHERS
        combustion regardless of temperature (the oxygen field proper comes later;
        this is its contact condition)."""
        airish = (self.mat == AIR) & (self.fvol < 0.5 * self.cap)
        access = airish.copy()
        access[1:, :, :] |= airish[:-1, :, :]
        access[:-1, :, :] |= airish[1:, :, :]
        access[:, 1:, :] |= airish[:, :-1, :]
        access[:, :-1, :] |= airish[:, 1:, :]
        access[:, :, 1:] |= airish[:, :, :-1]
        access[:, :, :-1] |= airish[:, :, 1:]
        return access

    def _law_burn(self):
        T = self.T()
        damp = np.clip((COMBUST_CEIL - T) / COMBUST_CEIL, 0.0, 1.0)
        access = self._gas_access()
        # a fire ENTRAINS air — its plume drags in far more oxygen than diffusion
        # brings. Each burning voxel may draw an allowance from ITS OWN airspace,
        # scaled by how much oxygen that airspace still holds: a sealed room's fire
        # starves as the room empties, the room NEXT DOOR keeps its air unless a
        # doorway (or a burned-through wall) joins the two into one lung.
        air = self.mat == AIR
        fresh = O2_PER_L * self.vox_l
        surf = self._region_of_surface()                     # which airspace each voxel breathes
        cshape = tuple((s + self._CS - 1) // self._CS for s in self.shape)
        drawn_c = np.zeros(cshape, np.float64)               # entrained O2, owed by the
                                                             # fire's OWN neighborhood
        drawn_r = np.zeros(self._region_count, np.float64)   # and only ever paid by the
                                                             # fire's OWN airspace
        # entrainment health is LOCAL (this cell's air), not the room average —
        # so a fire beside the doorway keeps breathing the draft while the far
        # corner starves first, and the blaze TAPERS instead of dying as one
        o2c = self._cells(np.where(air, self.o2, 0.0))
        airc = self._cells(air.astype(np.float32))
        # a plume's CATCHMENT: it entrains from meters around, not from its own
        # shoebox — health and the debt below must use the SAME reach, or the
        # fire eats a small pocket, reads the vacuum as suffocation, and
        # strangles itself while the room is still full of air (measured).
        # The blur is AIR-WEIGHTED: a wall is not a region of zero oxygen, it
        # is simply not air — averaging it in as 0 suffocated every fire that
        # burned near a floor (measured: health 0.21 under a fresh sky)
        reach = max(2, int(round(PLUME_REACH_M / (0.1 * self.scale * self._CS))))
        o2b = self._blur_cells(o2c, reach)               # o2c is already per-SLOT
        airb = np.maximum(self._blur_cells(airc, reach), 1e-6)
        hc = np.clip((o2b / airb) / fresh, 0.0, 1.0)     # so o2b/airb = per-AIR-voxel
        health_vox = self._upcell(hc).astype(np.float32)
        # the cell blur cannot see a voxel-thin wall — ceiling each fire's
        # health by ITS airspace's true average, so a sealed room's fire still
        # starves as the room empties, whatever the far side of the wall holds
        reg_air = self.air_region[air]
        reg_frac = np.ones(max(self._region_count, 1), np.float32)
        if air.any():
            reg_frac = (np.bincount(reg_air, weights=self.o2[air],
                                    minlength=self._region_count)
                        / np.maximum(np.bincount(reg_air,
                                                 minlength=self._region_count), 1)
                        / fresh).astype(np.float32)
            reg_frac = np.clip(reg_frac, 0.0, 1.0)

        def ration(mask, ratio, dm_want):
            """O2-limit dm for the masked voxels; consume local pores first, log the
            entrained remainder against the voxel's own coarse-cell neighborhood
            (the plume pulls from the meter around it, not from the far corner)."""
            r = surf[mask]
            health = np.minimum(health_vox[mask], reg_frac[np.maximum(r, 0)])
            allow = 3.0 * self.scale ** 2 * health * (r >= 0)
            dm = np.minimum(dm_want, (self.o2[mask] + allow) / ratio)
            local = np.minimum(dm * ratio, self.o2[mask])
            self.o2[mask] -= local
            need = dm * ratio - local
            ok = (r >= 0) & (need > 0)
            if ok.any():
                cells = np.argwhere(mask)[ok] // self._CS
                np.add.at(drawn_c, tuple(cells.T), need[ok])
                np.add.at(drawn_r, r[ok], need[ok])
            return dm

        for m, (dens, _c, _k, ign, heat, rate) in SOLID.items():
            if ign is None:
                continue
            w = (self.mat == m) & (self.smass > 0) & (T >= ign) & access
            if w.any():
                want = np.minimum(self.smass[w], rate * self.scale ** 2 * (T[w] - ign) * damp[w])
                dm = ration(w, O2_RATIO.get(m, 1.3), want)   # no oxygen ration, no fire
                self.smass[w] -= dm
                self.E[w] += dm * heat
                self.smoke[w] += dm * SOOT.get(m, 0.7)       # only the soot SHOWS
                self.p_add[w] += dm * 0.8                    # hot gas EXPANDS — gently:
                                                             # the outward push must not
                                                             # blow away the fire's own air
                if m in (WOOD, FLESH):                       # the VOLATILES flame off
                    charring = ((self.mat == m) & (T >= ign)     # (~80% of the mass);
                                & (self.smass < 0.18 * dens * self.vox_l)  # what's left
                                & (self.smass > 0))                        # is charcoal:
                    self.mat[charring] = CHAR                    # black forever, embers
                gone = (self.mat == m) & (self.smass < 0.05 * dens * self.vox_l)
                self.mat[gone] = ASH                         # ...becomes gas + a little ash
                self.smoke[gone] += self.smass[gone] * 0.5
                self.smass[gone] *= 0.5
        ign, heat, rate = FLUID[OIL][3], FLUID[OIL][4], FLUID[OIL][5]
        dens = FLUID[OIL][0]
        o = (self.fl == OIL) & (self.fvol > 0) & (T >= ign) & access
        if o.any():
            want = np.minimum(self.fvol[o] * dens, rate * self.scale ** 2 * (T[o] - ign) * damp[o])
            dg = ration(o, O2_RATIO_OIL, want)
            self.fvol[o] -= dg / dens
            self.E[o] += dg * heat
            self.p_add[o] += dg * 0.8
            self.smoke[o] += dg * SOOT_OIL                   # oil burns SOOTY — it all
            out = (self.fl == OIL) & (self.fvol <= 0.01)     # leaves as smoke mass
            self.fl[out] = NOFLUID
            self.fvol[out] = 0.0
        if (drawn_c > 0).any():
            # collect the debt from the same catchment the health read, in
            # proportion to where the air actually IS — but never across a
            # wall: the cell blur gives the catchment its SHAPE (fading with
            # distance), while the airspace REGION bounds who can pay (a
            # voxel-thin wall is invisible at cell scale, and chamber B's air
            # must stay chamber B's). Gradients now form across a room, and
            # wind + diffusion must genuinely resupply a far corner — but a
            # plume never starves while its own catchment still holds air.
            # the debt is paid by the gas that actually PASSED THROUGH the
            # flame — the hot plume and the ceiling layer it feeds — far more
            # than by the cool inflow skimming the floor. Weighting by warmth
            # is what makes the HOT LAYER the depleted one (and the classic
            # room-fire profile: foul air descending from above, crawl-low
            # air surviving longest at the floor).
            hot_w = 1.0 + np.clip(T - AMBIENT, 0.0, 300.0) / 50.0
            wvox = self._upcell(self._blur_cells(drawn_c, reach)
                                .astype(np.float32)) * self.o2 * hot_w
            for r in np.nonzero(drawn_r > 0)[0]:
                in_r = air & (self.air_region == r)
                w_r = np.where(in_r, wvox, 0.0)
                wsum = float(w_r.sum())
                if wsum <= 1e-9:
                    continue
                take = np.minimum(w_r * (drawn_r[r] / wsum), self.o2)
                self.o2 -= take.astype(np.float32)

    def _law_boil(self):
        T, C = self.T(), self.heat_capacity()
        # pressure raises the boiling point (the pressure-cooker effect, stolen as
        # DESIGN from the sandbox lineage): a sealed hot room boils water later
        boil_at = BOIL + (1.0 * np.clip(self._upcell(self.pcell), 0.0, 18.0)
                          if self.pcell is not None else 0.0)
        w = (self.fl == WATER) & (self.fvol > 0) & (T > boil_at)
        if w.any():
            boil_g = np.minimum(self.fvol[w], (T[w] - boil_at[w] if self.pcell is not None
                                               else T[w] - BOIL) * C[w] / LATENT)
            self.fvol[w] -= boil_g
            self.E[w] -= boil_g * LATENT
            self.p_add[w] += boil_g * 0.1                # steam pushes (gently — an
                                                         # open puddle is no boiler)
            dry = (self.fl == WATER) & (self.fvol <= 0.01)
            self.fl[dry] = NOFLUID
            self.fvol[dry] = 0.0
            self.fpot[dry] = 0.0

    def _law_melt(self):
        """PHASE CHANGE for metals, both directions, energy-honest (the same
        shape as boiling): heat above the melt point converts solid grams to
        molten milliliters at the rate the excess energy can pay the latent
        heat — so a melting bar PINS near its melt point, exactly as boiling
        water pins near 100. The melt stays held in the bar's cell until the
        solid is gone, then the flow law takes the puddle. Cooling runs the
        table backwards: a molten pool below its point sets solid where it
        lies, giving the latent heat back — a splat the shape of wherever it
        finally froze."""
        T, C = self.T(), self.heat_capacity()
        for m, (mt, fl_id, lat) in MELT.items():
            w = ((self.mat == m) & (self.smass > 0) & (T > mt)
                 & np.isin(self.fl, (NOFLUID, fl_id)))
            if w.any():
                dm = np.minimum(self.smass[w], (T[w] - mt) * C[w] / lat)
                self.smass[w] -= dm
                self.E[w] -= dm * lat
                self.fl[w] = fl_id
                self.fvol[w] += dm / FLUID[fl_id][0]
                gone = (self.mat == m) & (self.smass <= 0.01)
                self.mat[gone] = AIR             # nothing left but the melt —
                self.smass[gone] = 0.0           # it flows from here
        for fl_id, (mt, m, lat) in FREEZE.items():
            w = ((self.fl == fl_id) & (self.fvol > 0) & (T < mt - 2.0)
                 & np.isin(self.mat, (AIR, m)))
            if w.any():
                g = np.minimum(self.fvol[w] * FLUID[fl_id][0],
                               (mt - 2.0 - T[w]) * C[w] / lat)
                self.fvol[w] -= g / FLUID[fl_id][0]
                self.E[w] += g * lat
                self.mat[w] = m
                self.smass[w] += g
                dry = (self.fl == fl_id) & (self.fvol <= 0.01)
                self.fl[dry] = NOFLUID
                self.fvol[dry] = 0.0

    def _law_acid(self):
        """Dissolution — combustion's quieter sibling, driven ENTIRELY by the
        REACTIONS table. Where a reactive fluid touches a solid face, solid mass
        converts to fumes at the table's rate; the fluid is SPENT doing it and the
        row's heat is released. Nothing else is special: the hole an acid bores is
        just mat -> AIR, so the fluid FALLS into it and keeps eating (support and
        flow laws do the rest). A new acid is a data row, never code."""
        solid = self.mat != AIR
        for f in _REACTIVE:
            has = (self.fl == f) & (self.fvol > 1.0) & (self.fpot > 0.01)
            if not has.any():
                continue
            rate = np.zeros(NMAT, np.float32)
            spend = np.zeros(NMAT, np.float32)
            heat = np.zeros(NMAT, np.float32)
            for (rf, rm), (r, sp, h) in REACTIONS.items():
                if rf == f:
                    rate[rm], spend[rm], heat[rm] = r, sp, h
            rmat = rate[self.mat]
            for axis in range(3):
                a = [slice(None)] * 3; b = [slice(None)] * 3
                a[axis], b[axis] = slice(None, -1), slice(1, None)
                a, b = tuple(a), tuple(b)
                for s_, d_ in ((a, b), (b, a)):              # fluid at s_, solid at d_
                    m = has[s_] & solid[d_] & (rmat[d_] > 0)
                    if not m.any():
                        continue
                    mats = self.mat[d_][m]
                    # rate scales with POTENCY — a half-spent acid eats at half
                    # speed, so a pool quenches ITSELF as it works and the last
                    # of the wood survives in a puddle of exhausted sludge
                    dm = np.minimum(self.smass[d_][m],
                                    rmat[d_][m] * self.scale ** 2
                                    * self.fpot[s_][m]
                                    * np.minimum(self.fvol[s_][m] / self.cap, 1.0))
                    active_ml = self.fpot[s_][m] * self.fvol[s_][m]
                    dm = np.minimum(dm, active_ml / np.maximum(spend[mats], 1e-6))
                    self.smass[d_][m] -= dm
                    # SPENT reagent stays in the pool as inert volume: potency
                    # falls, the liquid itself remains (and pales, on camera)
                    self.fpot[s_][m] = np.maximum(
                        (active_ml - dm * spend[mats]) / self.fvol[s_][m], 0.0)
                    self.smoke[d_][m] += dm                  # dissolved mass leaves as fumes
                    self.E[s_][m] += dm * heat[mats]
        gone = solid & (self.smass <= 0.01)
        if gone.any():
            self.mat[gone] = AIR                             # dissolved CLEAN — no ash
            self.smass[gone] = 0.0

    _CS = 4                                              # air-field cell = 4^3 voxels

    def _blur_cells(self, carr, passes):
        """Neighbor-mean blur on the coarse cell grid — the shape of a plume's
        entrainment field: strongest at the fire, fading over `passes` cells."""
        out = carr
        for _ in range(passes):
            m2 = out.copy(); c2 = np.ones_like(out)
            for axis in range(3):
                a = [slice(None)] * 3; b = [slice(None)] * 3
                a[axis], b[axis] = slice(None, -1), slice(1, None)
                a, b = tuple(a), tuple(b)
                m2[a] += out[b]; c2[a] += 1
                m2[b] += out[a]; c2[b] += 1
            out = m2 / c2
        return out

    def _cells(self, arr, how="mean"):
        """Downsample a voxel array onto the coarse cell grid."""
        cs = self._CS
        nx, ny, nz = self.shape
        px, py, pz = (-nx) % cs, (-ny) % cs, (-nz) % cs
        a = np.pad(arr, ((0, px), (0, py), (0, pz)))
        a = a.reshape(a.shape[0] // cs, cs, a.shape[1] // cs, cs, a.shape[2] // cs, cs)
        return a.mean(axis=(1, 3, 5)) if how == "mean" else a.sum(axis=(1, 3, 5))

    def _upcell(self, carr):
        """Broadcast a cell array back to voxel resolution."""
        cs = self._CS
        nx, ny, nz = self.shape
        return np.repeat(np.repeat(np.repeat(carr, cs, 0), cs, 1), cs, 2)[:nx, :ny, :nz]

    def _law_air(self):
        """PRESSURE, pneumatic half — a coarse LOSSY gas field (the falling-sand
        lineage's design, in 3D): pressure and velocity live on 4-voxel cells.
        Fires and boiling INJECT pressure (hot gas expands); velocity follows the
        pressure gradient; pressure follows the velocity divergence; both decay.
        Walls block. The open sky bleeds pressure away. The field then CARRIES
        smoke and oxygen (advection) — a fire now pushes its smoke through a
        doorway instead of politely diffusing."""
        if self.pcell is None:
            cshape = tuple((s + self._CS - 1) // self._CS for s in self.shape)
            self.pcell = np.zeros(cshape, np.float32)
            self.vcell = [np.zeros(cshape, np.float32) for _ in range(3)]
        # what the coarse pressure cells can breathe through, read from POROSITY
        # rather than a headcount of non-air voxels: a shut door that fills 97%
        # of its cells is a path, and the old binary read rounded its gap away.
        # The block stays HARD, not graded — these cells are 20 cm wide, so a
        # cell half-filled by a wall has that wall across its whole face, and
        # grading it by volume fraction let sealed rooms bleed pressure through
        # solid masonry (measured: a sealed fire stopped out-pressuring an open
        # one). Coarse geometry cannot support a soft answer here.
        cpor = self._cells(self.porosity())
        blocked = cpor < 0.05
        self.pcell += self._cells(self.p_add, "sum")
        self.p_add[:] = 0.0
        p, v = self.pcell, self.vcell
        # convection: warm cells RISE — this closes the loop that gives a room a
        # real breeze: up over the fire, out along the ceiling, down the cool
        # walls, back along the floor. Without it the wind only pushes outward
        # and smoke piles into corners.
        warm = self._cells(np.clip(self.T() - AMBIENT, 0.0, 500.0).astype(np.float32))
        v[2] += np.where(~blocked, warm * 0.0015, 0.0)
        for axis in range(3):                            # velocity chases the gradient
            a = [slice(None)] * 3; b = [slice(None)] * 3
            a[axis], b[axis] = slice(None, -1), slice(1, None)
            a, b = tuple(a), tuple(b)
            g = (p[a] - p[b]) * 0.12
            ok = ~blocked[a] & ~blocked[b]
            v[axis][a] += np.where(ok, g, 0.0)
            v[axis][b] += np.where(ok, g, 0.0)
            div_pair = v[axis][a] - v[axis][b]           # and pressure follows the flow
            p[a] -= 0.08 * np.where(ok, div_pair, 0.0)
            p[b] += 0.08 * np.where(ok, div_pair, 0.0)
        for arr in (p, *v):                              # diffuse + decay (lossy on
            m = arr.copy()                               # purpose — game-feel field)
            cnt = np.ones_like(arr)
            for axis in range(3):
                a = [slice(None)] * 3; b = [slice(None)] * 3
                a[axis], b[axis] = slice(None, -1), slice(1, None)
                a, b = tuple(a), tuple(b)
                m[tuple(a)] += arr[tuple(b)]; cnt[tuple(a)] += 1
                m[tuple(b)] += arr[tuple(a)]; cnt[tuple(b)] += 1
            arr *= 0.8
            arr += 0.2 * (m / cnt)
            arr *= 0.965
        p[blocked] = 0.0
        for axis in range(3):
            v[axis][blocked] = 0.0
        if self.open_sky:
            sky = ~blocked[:, :, -1]
            p[:, :, -1][sky] *= 0.3
        # advection: the wind carries the gases, upwind. A face passes what its
        # PORES allow — open air freely, a shut door at a hundredth. This is the
        # half that matters for a closed door: diffusion through a 1%-porous leaf
        # is nearly nil (measured), but a fire PRESSURISES the room it is in and
        # pushes its smoke through the gaps. That push is why a shut door is a
        # delay and not a seal.
        air = self.mat == AIR
        por = self.porosity()
        for axis in range(3):
            w = np.clip(self._upcell(v[axis]) * 0.25, -0.4, 0.4)
            a = [slice(None)] * 3; b = [slice(None)] * 3
            a[axis], b[axis] = slice(None, -1), slice(1, None)
            a, b = tuple(a), tuple(b)
            pair_ok = np.minimum(por[a], por[b])
            wface = w[a]
            # the wind carries SMOKE only, still. Second attempt at O2-advection
            # (2026-09-04), measured and rejected for a DIFFERENT reason than the
            # first: the lossy compressible field plus set-value boundaries acts
            # as a mass PUMP — the house-fire room sloshed itself down to 2% O2
            # with no fire burning at all. Composition moves by diffusion and by
            # buoyant OVERTURN in _law_o2 (hot-poor gas trades places with
            # cool-rich gas), which is local, conservative, and can't pump.
            for field in (self.smoke,):
                if field is None:
                    continue
                fwd = np.where(wface > 0, wface * field[a], 0.0) * pair_ok
                back = np.where(wface < 0, -wface * field[b], 0.0) * pair_ok
                field[a] += back - fwd
                field[b] += fwd - back
        # (the old 1.5x-fresh O2 clip is gone: clipping DESTROYED whatever the
        # wind compressed past the cap — the world was quietly losing oxygen
        # every tick. Compressed pockets are fine: burn health caps at 1, so
        # rich air is a reservoir, never a rate boost.)

    def _law_smoke(self):
        """Smoke is a gas: it seeps to neighbors, RISES hard through open air, and
        leaves at the open sky. The mass that burning removes from wood and oil
        travels here — conservation made visible."""
        air = self.mat == AIR
        por = self.porosity()
        # smoke is born INSIDE burning solids — it must escape through the surface
        # first (up if it can, sideways if it must), or it stays trapped in the wood
        lo = (slice(None), slice(None), slice(None, -1))
        up = (slice(None), slice(None), slice(1, None))
        esc = np.where(~air[lo] & air[up], 0.9 * self.smoke[lo], 0.0).astype(np.float32)
        self.smoke[lo] -= esc
        self.smoke[up] += esc
        for a, b in ((( slice(None, -1), slice(None), slice(None)), (slice(1, None), slice(None), slice(None))),
                     ((slice(None), slice(None, -1), slice(None)), (slice(None), slice(1, None), slice(None)))):
            for s_, d_ in ((a, b), (b, a)):
                esc = np.where(~air[s_] & air[d_], 0.5 * self.smoke[s_], 0.0).astype(np.float32)
                self.smoke[s_] -= esc
                self.smoke[d_] += esc
        for axis in range(3):                                # seep: even out with neighbors
            a = [slice(None)] * 3; b = [slice(None)] * 3
            a[axis], b[axis] = slice(None, -1), slice(1, None)
            a, b = tuple(a), tuple(b)
            q = 0.08 * (self.smoke[a] - self.smoke[b]) * np.minimum(por[a], por[b])
            self.smoke[a] -= q
            self.smoke[b] += q
        lo = (slice(None), slice(None), slice(None, -1))
        up = (slice(None), slice(None), slice(1, None))
        risable = air[lo] & air[up]                          # rise: buoyant, fast
        q = np.where(risable, 0.4 * self.smoke[lo], 0.0).astype(np.float32)
        self.smoke[lo] -= q
        self.smoke[up] += q
        # the CEILING JET: smoke that can rise no further pools against the
        # ceiling and RACES along it — buoyancy makes it an upside-down liquid
        # seeking an upside-down level. Without this the bank crawled at
        # diffusion pace and the far half of a burning room read 0.00 g
        # forever (measured; a real ceiling jet runs meters per second).
        pinned = np.zeros_like(air)
        pinned[:, :, :-1] = air[:, :, :-1] & ~air[:, :, 1:]  # air with solid above
        pinned[:, :, -1] = air[:, :, -1] & (not self.open_sky)
        for _ in range(2):
            for axis in (0, 1):
                a = [slice(None)] * 3; b = [slice(None)] * 3
                a[axis], b[axis] = slice(None, -1), slice(1, None)
                a, b = tuple(a), tuple(b)
                m = pinned[a] & pinned[b]
                q = np.where(m, 0.25 * (self.smoke[a] - self.smoke[b]),
                             0.0).astype(np.float32)
                self.smoke[a] -= q
                self.smoke[b] += q
        top = (slice(None), slice(None), -1)
        self.smoke[top][air[top]] *= 0.5                     # the open sky takes it

    def _air_regions(self):
        """Label the connected airspaces (6-connectivity, pure numpy flood by
        max-propagation). A fire can only breathe the air ITS OWN airspace holds —
        a sealed wall is a real wall, a doorway joins two rooms into one lung.
        Recomputed only when the set of air voxels changes (a wall burns through,
        a collapse opens a gap)."""
        air = self.mat == AIR
        if self._region_air is not None and bool((air == self._region_air).all()):
            return
        lab = np.where(air, np.arange(air.size, dtype=np.int32).reshape(self.shape), -1)
        while True:
            new = lab.copy()
            new[1:, :, :] = np.maximum(new[1:, :, :], lab[:-1, :, :])
            new[:-1, :, :] = np.maximum(new[:-1, :, :], lab[1:, :, :])
            new[:, 1:, :] = np.maximum(new[:, 1:, :], lab[:, :-1, :])
            new[:, :-1, :] = np.maximum(new[:, :-1, :], lab[:, 1:, :])
            new[:, :, 1:] = np.maximum(new[:, :, 1:], lab[:, :, :-1])
            new[:, :, :-1] = np.maximum(new[:, :, :-1], lab[:, :, 1:])
            new[~air] = -1
            if (new == lab).all():
                break
            lab = new
        _, compact = np.unique(lab[air], return_inverse=True)
        self.air_region = np.full(self.shape, -1, np.int32)
        self.air_region[air] = compact.astype(np.int32)
        self._region_count = int(compact.max()) + 1 if compact.size else 0
        self._region_air = air

    def _region_of_surface(self):
        """For every voxel, the airspace it touches (max over the 6 neighbors) —
        this is the air a burning SOLID voxel breathes from. -1: touches none."""
        r = self.air_region
        out = r.copy()
        out[1:, :, :] = np.maximum(out[1:, :, :], r[:-1, :, :])
        out[:-1, :, :] = np.maximum(out[:-1, :, :], r[1:, :, :])
        out[:, 1:, :] = np.maximum(out[:, 1:, :], r[:, :-1, :])
        out[:, :-1, :] = np.maximum(out[:, :-1, :], r[:, 1:, :])
        out[:, :, 1:] = np.maximum(out[:, :, 1:], r[:, :, :-1])
        out[:, :, :-1] = np.maximum(out[:, :, :-1], r[:, :, 1:])
        return out

    def _law_o2(self):
        """Oxygen: everywhere air reaches, including a solid's pores; fire eats it (in
        the burn law); movement evens it out. The mixing rate RISES with temperature —
        a fire violently stirs the air that feeds it (convective entrainment), cold air
        barely moves. An open sky refills the top; in a SEALED space (open_sky=False)
        the fire eats what the room holds, then stops — fuel or no fuel."""
        T = self.T()
        air = self.mat == AIR
        por = self.porosity()
        for axis in range(3):
            a = [slice(None)] * 3; b = [slice(None)] * 3
            a[axis], b[axis] = slice(None, -1), slice(1, None)
            a, b = tuple(a), tuple(b)
            k = np.clip(0.1 + (T[a] + T[b]) / 1600.0, 0.1, 0.45)
            k = k * np.minimum(por[a], por[b])       # a barrier passes what its PORES
            q = k * (self.o2[a] - self.o2[b])        # allow: a shut door breathes a
            self.o2[a] -= q                          # little, masonry almost nothing,
            self.o2[b] += q                          # a pane nothing at all
        # (the old 0.1-per-tick pull toward the REGION MEAN is gone: it made every
        # airspace an instantly-stirred lung — head air always equaled knee air,
        # and a fire's deficit teleported to every nose in the room.)
        # BUOYANT OVERTURN — the vertical transport that builds the two-layer
        # fire room (CFAST's physics, done per-pair): where warmer gas sits
        # UNDER cooler gas the column is unstable and the parcels trade places,
        # each keeping its composition — so the fire's hot, oxygen-poor exhaust
        # climbs to the ceiling and the cool fresh air it displaces slides down
        # to the floor. Crawl-low is this law's shadow. A stable column (cool
        # below hot) doesn't move: stratification, once built, PERSISTS.
        lo = (slice(None), slice(None), slice(None, -1))
        hi = (slice(None), slice(None), slice(1, None))
        dT = T[lo] - T[hi]                               # >0 = unstable: hot below
        f = np.where(air[lo] & air[hi],
                     np.clip(dT * 0.02, 0.0, 0.45), 0.0).astype(np.float32)
        for gas in (self.o2, self.smoke):        # a parcel is ONE gas: its oxygen
            q = f * (gas[lo] - gas[hi])          # and its soot travel together
            gas[lo] -= q
            gas[hi] += q
        # TWO-ZONE stirring — the CFAST room, made literal: every airspace is a
        # HOT SMOKY zone (the plume and the exhaust bank it feeds under the
        # ceiling) and a COOL CLEAR zone (everything else), each stirred fast
        # internally by its own buoyant circulation, exchanging across the
        # interface only by the diffusion and overturn above. The fire's debt
        # is hot-weighted (see the burn law), so the FOUL zone is the hot one —
        # and as smoke and heat bank downward the foul zone DESCENDS onto a
        # standing head while the floor stays breathable: crawl-low is this
        # structure's shadow, not a rule. It is also why a bed fire keeps
        # breathing: its cool zone is the whole room's fresh pool (a fire that
        # dug a local oxygen hole and died beside fresh air was the measured
        # alternative).
        if air.any():
            # the stirring is LOCAL — a plume circulates ITS room, not every
            # airspace its doorway connects to. (Zone-global mixing was tried
            # and it teleported the room's smoke into the whole yard outside,
            # diluting it to 0.00 g — the same crime as the old region mean.)
            # Each voxel relaxes toward the blurred mean of its OWN zone over
            # a room-scale reach, so a doorway is a gradient, not a wormhole.
            hotzone = (T > 45.0) | (self.smoke > 0.2)
            reach = max(2, int(round(PLUME_REACH_M / (0.1 * self.scale * self._CS))))
            sizes = np.bincount(self.air_region[air], minlength=self._region_count)
            for r in np.nonzero(sizes > 64)[0]:  # a pocket too small to hold a
                in_r = air & (self.air_region == r)          # circulation isn't stirred
                for zone in (True, False):
                    mz = in_r & (hotzone == zone)            # zone AND region: the blur
                    if not mz.any():                         # cannot see a thin wall,
                        continue                             # so the mask must
                    wt_c = np.maximum(self._blur_cells(
                        self._cells(mz.astype(np.float32)), reach), 1e-6)
                    for gas in (self.o2, self.smoke):
                        gm_c = self._blur_cells(
                            self._cells(np.where(mz, gas, 0.0)), reach)
                        local = self._upcell((gm_c / wt_c).astype(np.float32))
                        gas[mz] += 0.15 * (local[mz] - gas[mz])
        if self.open_sky:
            # an open world floats in an INFINITE fresh atmosphere: every
            # lattice edge that is air sits at ambient composition. (Top-only
            # refresh let a long fire drag the whole map toward extinction —
            # half-oxygen "open air", which no real backyard ever becomes.
            # Sealed scenes wall their edges in stone and never feel this.)
            fresh = O2_PER_L * self.vox_l
            for edge in ((slice(None), slice(None), -1),
                         (0, slice(None), slice(None)), (-1, slice(None), slice(None)),
                         (slice(None), 0, slice(None)), (slice(None), -1, slice(None))):
                self.o2[edge][air[edge]] = fresh

    # ── physiology: living bodies ────────────────────────────────────────────
    def add_person(self, x, y, name="the person", knows_world=True):
        """Register a living body anchored near (x, y). The FLESH voxels are the
        body; this struct is only the slow chemistry riding on them. One person
        per neighborhood for now — the flesh search is anchor-local."""
        p = {"anchor": (int(x), int(y)), "name": name, "blood_o2": 0.97,
             "smoke": 0.0, "burn": 0.0, "hurt": 0.0, "awake": True, "alive": True,
             "fleeing": False, "safe": False, "events": [],
             # WHAT THIS BODY HAS SEEN — not what the world contains. This is
             # the difference between a person and the sim: the sim holds the
             # register of exits, a person holds the ones they have laid eyes
             # on, and pathing runs on the latter.
             #
             # The default is True because the usual case is someone standing
             # in a place they live: they HAVE observed it, so knowing where
             # their own door is privileges nothing. Pass knows_world=False for
             # a stranger, and they must find the door by looking — which is
             # what makes wandering and looking around worth doing at all.
             "known": np.full(self.shape[:2], bool(knows_world))}
        self.persons.append(p)
        return p

    def _person_cells(self, p):
        """The person's OWN flesh — and OWN means owned, not merely connected.

        Bodies used to be identified by adjacency alone, so the moment two
        people touched they became one 3-voxel-wide, two-headed person: both
        resolved to the same 684 cells, both anchors converged, and anything
        that made them touch (carrying, dragging, a crowd in a doorway) broke
        them. Now the bodies are claimed once per tick in a fixed order, and a
        flood stops at flesh another person has already claimed. Two people can
        stand shoulder to shoulder and stay two people."""
        if p.get("_claim_tick") != self.tick:
            self._claim_bodies()
        return p.get("_claim_mask"), p.get("_claim_sl")

    def _claim_bodies(self):
        """Resolve every body once a tick, all of them AT THE SAME TIME.

        Two people in contact are genuinely one connected lump, so whoever
        floods first swallows both — order cannot fix that, and neither can
        barring last tick's cells, because a body that has taken a step is
        standing somewhere last tick's barrier does not cover. Claiming one
        after another therefore lets whoever runs first eat into a neighbour's
        NEW flesh: measured, 684 cells of two people split 497/187, and the
        one left with half a body could no longer walk.

        So identity persists through TIME and is settled FAIRLY. Each body is
        seeded with the cells it owned a moment ago, and every seed grows one
        voxel a round together. Contested flesh goes to whichever body was
        nearer to it last tick, with ties broken by a fixed order so the sim
        stays deterministic. No one claims before anyone else, so no one can
        claim out of anyone else.

        A seed can still go missing: a body that keels over is promoted to a
        free body and its flesh LEAVES the lattice for a few ticks. Falling
        back to "nearest flesh to the anchor" then hands it the body of whoever
        is standing over it — measured: a fainting person resolved to 42 cells
        of their would-be rescuer, and the rescuer spent a hundred ticks
        dragging a piece of themselves across the room while the real body lay
        where it fell. A body with nothing to grow from waits, and may adopt
        only flesh the grow left over."""
        prev = {id(q): q.get("_own", frozenset()) for q in self.persons}
        # AND NEITHER HAS SOMEONE IN MID-AIR. A body promoted whole to a free
        # body has no flesh on the lattice AT ALL for those ticks, so the seed
        # it falls back on is whatever unclaimed flesh is nearest — which is a
        # bystander. Measured: while one man fell off a ledge the man who
        # pushed him lost 21 voxels to him, then read as standing on a footprint
        # that was no longer under his own weight, and toppled for it. A LIMB in
        # flight is different: a body mid-swing is still standing there.
        aloft = {b["owner"] for b in self.bodies
                 if b.get("owner") and not b.get("part")}
        live = []
        for q in self.persons:
            # someone who walked out, or died and was cleared, has no body to
            # find. Left in, their anchor-flood reaches for the nearest flesh it
            # can see and takes SOMEBODY ELSE'S — measured: a person already
            # outside claimed a colleague's body from 35 voxels away, and the
            # colleague, barred from their own flesh, stopped dead.
            if q["safe"] or not q["alive"] or q["name"] in aloft:
                q["_claim_mask"], q["_claim_sl"] = None, None
                q["_claim_tick"], q["_own"] = self.tick, frozenset()
            else:
                live.append(q)
        if not live:
            return
        flesh = self.mat == FLESH
        FREE = np.int16(np.iinfo(np.int16).max)
        label = np.where(flesh, FREE, np.int16(-1)).astype(np.int16)
        for i, q in enumerate(live):
            for c in prev[id(q)]:
                if flesh[c]:
                    label[c] = i                 # what it was, still flesh
        # A BODY WITH NOTHING TO GROW FROM seeds on the nearest flesh nobody
        # else already claims. Two people who START in contact are one lump
        # with no history to divide it, and flooding them one at a time gives
        # the whole lump to whoever went first and NOTHING to the second, who
        # then has no body at all — measured, a rescuer who could not act
        # because they did not exist. Seeding both and letting the grow settle
        # it splits the lump by who was nearer, which is the same rule that
        # settles every other contested voxel. It is also why a fainting body
        # is not re-acquired out of its rescuer: with its own flesh off the
        # lattice, every cell near it is already claimed, so it seeds nothing.
        r = max(int(1.8 / max(0.1 * self.scale, 1e-9)), 8)
        for i, q in enumerate(live):
            if any(flesh[c] for c in prev[id(q)]):
                continue
            cand = np.argwhere(label == FREE)
            if not len(cand):
                break
            x0, y0 = q["anchor"]
            d = np.abs(cand[:, 0] - x0) + np.abs(cand[:, 1] - y0)
            near = cand[int(d.argmin())]
            if d.min() <= 2 * r:
                label[tuple(near)] = i
        while True:                              # every seed grows together
            free = label == FREE
            if not free.any():
                break
            best = np.full(self.shape, FREE, np.int16)
            lab = np.where(label >= 0, label, FREE)   # air is not a low label:
            for ax in range(3):                       # -1 leaking into the
                for d in (1, -1):                     # minimum made every
                    sh = np.full(self.shape, FREE, np.int16)   # SURFACE cell
                    src = [slice(None)] * 3; dst = [slice(None)] * 3  # unclaim-
                    src[ax] = slice(1, None) if d > 0 else slice(None, -1)
                    dst[ax] = slice(None, -1) if d > 0 else slice(1, None)
                    sh[tuple(dst)] = lab[tuple(src)]  # able, and bodies wasted
                    np.minimum(best, sh, out=best)    # away to four voxels
            take = free & (best < FREE)
            if not take.any():
                break                            # the rest touches nobody
            label[take] = best[take]
        for i, q in enumerate(live):
            mine = label == i
            if mine.any():                       # a whole-lattice mask, but the
                nx, ny, _nz = self.shape         # slice still starts at 0 so
                self._own_claim(q, mine,         # every caller's offset maths
                                (slice(0, nx), slice(0, ny), slice(None)))
            else:                                # off the lattice: no body to
                q["_claim_mask"], q["_claim_sl"] = None, None    # be found, and
                q["_claim_tick"], q["_own"] = self.tick, frozenset()  # none
                                                 # borrowed from anybody else


    def _own_claim(self, q, mask, sl):
        """Record what this body turned out to be, and return its cells."""
        got = np.argwhere(mask)
        got[:, 0] += sl[0].start or 0            # a whole-lattice mask has no
        got[:, 1] += sl[1].start or 0            # offset to add back
        cells = frozenset(map(tuple, got))
        q["_claim_mask"], q["_claim_sl"], q["_claim_tick"] = mask, sl, self.tick
        q["_own"] = cells
        return cells

    def _flood_person(self, p, taken=(), seed_from=(), adopt=True, only=None):
        """The connected component nearest this anchor, stopping at flesh that
        belongs to somebody else.

        `adopt` is permission to take flesh this body has no claim on — right
        for a body being seen for the first time, wrong for one whose own flesh
        has gone missing. `only` narrows the search to particular cells, which
        is how a body that has just landed picks itself back up out of the
        flesh the simultaneous grow left spare, and cannot reach for anyone."""
        x0, y0 = p["anchor"]
        nx, ny, nz = self.shape
        r = max(int(1.8 / (0.1 * self.scale)), 8)
        sl = (slice(max(x0 - r, 0), min(x0 + r, nx)),
              slice(max(y0 - r, 0), min(y0 + r, ny)), slice(None))
        flesh = self.mat[sl] == FLESH
        if only is not None:
            keep = np.zeros_like(flesh)
            for (tx, ty, tz) in only:
                lx, ly = tx - sl[0].start, ty - sl[1].start
                if 0 <= lx < flesh.shape[0] and 0 <= ly < flesh.shape[1]:
                    keep[lx, ly, tz] = True
            flesh &= keep
        if taken:                                    # someone else's already
            for (tx, ty, tz) in taken:               # — not part of this body
                lx, ly = tx - sl[0].start, ty - sl[1].start
                if 0 <= lx < flesh.shape[0] and 0 <= ly < flesh.shape[1]:
                    flesh[lx, ly, tz] = False
        if not flesh.any():
            return None, sl
        comp = np.zeros_like(flesh)
        for (sx, sy, sz) in seed_from:            # what this body WAS, still
            lx, ly = sx - sl[0].start, sy - sl[1].start   # flesh: grow from that
            if 0 <= lx < flesh.shape[0] and 0 <= ly < flesh.shape[1] \
                    and flesh[lx, ly, sz]:
                comp[lx, ly, sz] = True
        if not comp.any():                        # first sight of this body
            if not adopt:
                return None, sl
            cand = np.argwhere(flesh)
            d = np.abs(cand[:, 0] - (x0 - sl[0].start)) \
                + np.abs(cand[:, 1] - (y0 - sl[1].start))
            comp[tuple(cand[int(d.argmin())])] = True
        while True:                                  # flood restricted to flesh
            grown = comp.copy()
            grown[1:] |= comp[:-1]; grown[:-1] |= comp[1:]
            grown[:, 1:] |= comp[:, :-1]; grown[:, :-1] |= comp[:, 1:]
            grown[:, :, 1:] |= comp[:, :, :-1]; grown[:, :, :-1] |= comp[:, :, 1:]
            grown &= flesh
            if (grown == comp).all():
                break
            comp = grown
        return comp, sl

    def person_status(self, p):
        state = ("DEAD" if not p["alive"] else
                 "unconscious" if not p["awake"] else "awake")
        return (f"{p['name']}: blood O2 {100 * p['blood_o2']:.0f}%, smoke load "
                f"{p['smoke']:.2f}, burns {100 * p['burn']:.0f}%, wounds "
                f"{100 * p['hurt']:.0f}% — {state}")

    def _law_life(self):
        """PHYSIOLOGY — the body's slow chemistry, every constant from BODY.
        Blood oxygen chases the air actually around the mouth; inhaled smoke
        rides along and blocks uptake (the choking effect); skin hotter than
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
            if not flesh.any():
                if any((b["mats"] == FLESH).any() for b in self.bodies):
                    continue                             # mid-fall: body is in the air
                p["alive"] = False
                p["events"].append(f"t{self.tick}: {p['name']} is gone")
                continue
            zs = np.argwhere(flesh)[:, 2]
            z_lo, z_hi = int(zs.min()), int(zs.max())
            head = flesh.copy()                          # the mouth breathes the air
            head[:, :, :max(z_hi - 3, z_lo)] = False     # touching the body's TOP —
            near = np.zeros_like(head)                   # standing, that's head height;
            for axis in range(3):                        # collapsed, the floor layer
                near |= np.roll(head, 1, axis) | np.roll(head, -1, axis)
            near[0], near[-1] = False, False             # roll wraps: trim box edges
            near[:, 0], near[:, -1] = False, False
            near[:, :, 0], near[:, :, -1] = False, False
            near &= self.mat[sl] == AIR
            if near.any():
                frac = float(self.o2[sl][near].mean()) / fresh
                sm = float(self.smoke[sl][near].mean())
            else:
                frac, sm = 0.0, 0.0                      # face buried: nothing to breathe
            p["smoke"] = max(p["smoke"] + BODY["smoke_in"] * sm
                             - BODY["smoke_out"], 0.0)
            uptake = min(frac, 1.0) * max(1.0 - p["smoke"], 0.0)
            p["blood_o2"] += BODY["breath"] * (uptake - p["blood_o2"])
            hot = np.clip(T[sl][flesh] - BODY["hurt_T"], 0.0, 400.0)
            p["burn"] = min(p["burn"] + BODY["burn_gain"] * float(hot.mean()), 1.0)
            dmg = p["burn"] + p["hurt"]     # tissue damage is tissue damage:
            # cooked or crushed, the body it incapacitates is the same body,
            # so the two integrals read against ONE pair of thresholds
            if p["awake"] and (p["blood_o2"] < BODY["faint_o2"]
                               or dmg > BODY["faint_burn"]):
                p["awake"] = False
                why = ("the foul air" if dmg <= BODY["faint_burn"]
                       else "the burns" if p["burn"] >= p["hurt"]
                       else "the wounds")
                p["events"].append(f"t{self.tick}: {p['name']} slumps — {why}")
            elif (not p["awake"] and p["blood_o2"] > BODY["wake_o2"]
                    and dmg <= BODY["faint_burn"]):
                # air alone can undo what air did. Burns cannot be undone —
                # the burn integral only climbs — so someone who went down from
                # heat stays down, and someone who went down from the air comes
                # round if the air comes good. The threshold used to trip one
                # way only, so a body carried into clean air stayed unconscious
                # for ever, which made rescuing anyone pointless.
                p["awake"] = True
                p["emergency"] = False          # comes round with no plan
                p["goal"], p["_path"] = None, None
                p["events"].append(f"t{self.tick}: {p['name']} comes round")
            if (not p["awake"] and self.tick % 8 == 0
                    and z_hi - z_lo > int(0.8 / (0.1 * self.scale))):
                self._collapse(p, flesh, sl)             # still upright: nothing holds
            if p["blood_o2"] < BODY["death_o2"] or dmg > BODY["death_burn"]:
                p["alive"] = False
                p["events"].append(f"t{self.tick}: {p['name']} stops breathing")

    def _look_around(self, p, eye):
        """Sweep the eyes and REMEMBER what they fell on.

        A body's map of the world is what it has actually looked at — nothing
        else. Rays go out through the facing cone and stop where the view
        stops, so a wall casts a shadow of ignorance behind it and a doorway
        lets knowledge through. Turning the head fills the map in over time,
        and walking somewhere fills it in properly.

        This is what makes the exit list honest. Before it, every body picked
        the nearest door from the world's own register and pathed to it across
        rooms it had never entered — the sim's knowledge wearing a person's
        face. Now a stranger has to find the door."""
        nx, ny, _ = self.shape
        vox_m = 0.1 * self.scale
        R = max(int(WILL["see_m"] / max(vox_m, 1e-9)), 4)
        z = int(np.clip(round(eye[2]), 0, self.shape[2] - 1))
        nrays = max(48, int(2.0 * np.pi * R / 1.5))
        ang = np.linspace(0.0, 2.0 * np.pi, nrays, endpoint=False)
        fx, fy = p.get("gaze", p.get("facing", (1.0, 0.0)))
        fn = float(np.hypot(fx, fy)) or 1.0
        inside = (np.cos(ang) * fx + np.sin(ang) * fy) / fn >= \
            np.cos(np.radians(WILL["fov_deg"] / 2.0))
        ang = ang[inside]
        if not len(ang):
            return
        steps = np.arange(1, R + 1, dtype=np.float32)
        xi = np.clip(np.round(eye[0] + np.cos(ang)[:, None] * steps[None]),
                     0, nx - 1).astype(np.int32)
        yi = np.clip(np.round(eye[1] + np.sin(ang)[:, None] * steps[None]),
                     0, ny - 1).astype(np.int32)
        mat = self.mat[xi, yi, z]
        packed = np.clip(self.smass[xi, yi, z] / np.maximum(
            _DENS_ARR[mat] * self.vox_l, 1e-9), 0.0, 0.999)
        through = (1.0 - packed) + packed * _TRANSMIT_ARR[mat]
        tau = np.cumsum(-np.log(np.maximum(through, 1e-9)), axis=1)
        # a cell is seen if the view REACHED it: the depth BEFORE it is clear
        reach = np.concatenate([np.zeros((len(ang), 1), np.float32),
                                tau[:, :-1]], axis=1) < SEE_TAU
        p["known"][xi[reach], yi[reach]] = True
        p["known"][int(round(eye[0])), int(round(eye[1]))] = True

    def _sees(self, a, b):
        """Sight as OPTICAL DEPTH along the ray, not a list of allowed materials.

        Every voxel on the way dims the view by how much stuff is in it: the
        material's own opacity scaled by how full the cell actually is, plus the
        smoke hanging in it. Beer-Lambert; you can see if enough light survives.

        What this buys over the old material whitelist: a GLASS window is a
        window (it was opaque before), a sparse hedge dims instead of walling
        off, a doorway half-choked with rubble is half-blind, and — the one that
        was simply missing — SMOKE BLINDS. You could previously spot a runner
        clean through a smoke bank. The old "bodies don't wall off sight" case
        is gone with it: a body between you and the fire really does block it,
        and your own is skipped by geometry (you don't see your own eyelashes)
        rather than by exempting the material."""
        a = np.asarray(a, np.float32)
        b = np.asarray(b, np.float32)
        n = int(np.ceil(np.abs(b - a).max())) + 1
        if n <= 2:
            return True
        ts = np.linspace(0.0, 1.0, n)[1:-1]
        pts = np.round(a[None] + ts[:, None] * (b - a)[None]).astype(np.int64)
        near = (np.abs(pts - a[None]).max(axis=1) > 1.0)      # skip own eyelashes
        pts = pts[near]
        if not len(pts):
            return True
        nx, ny, nz = self.shape
        pts[:, 0] = np.clip(pts[:, 0], 0, nx - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, ny - 1)
        pts[:, 2] = np.clip(pts[:, 2], 0, nz - 1)
        idx = tuple(pts.T)
        vox_m = 0.1 * self.scale
        packed = np.clip(self.smass[idx] / np.maximum(
            _DENS_ARR[self.mat[idx]] * self.vox_l, 1e-9), 0.0, 0.999)
        # what gets past one voxel: the (1-f) that misses the stuff entirely,
        # plus the f that hits it and comes through anyway if it is glass
        through = (1.0 - packed) + packed * _TRANSMIT_ARR[self.mat[idx]]
        tau = float(-np.log(np.maximum(through, 1e-9)).sum())
        tau += float((SMOKE_EXT * (self.smoke[idx] / self.vox_l) * vox_m).sum())
        return tau < SEE_TAU

    @staticmethod
    def _in_cone(facing, eye, target):
        """Eyes point where the face points: the target must sit inside the
        field-of-view cone around `facing` (a 2D heading)."""
        vx, vy = target[0] - eye[0], target[1] - eye[1]
        n = float(np.hypot(vx, vy))
        if n < 1.5:
            return True                              # at arm's reach you just know
        cosang = (facing[0] * vx + facing[1] * vy) / n
        return cosang >= np.cos(np.radians(WILL["fov_deg"] / 2.0))

    def _hears(self, a, b):
        """Hearing in DECIBELS, by the acoustic MASS LAW.

        A shout leaves the mouth at ~90 dB. Distance spends it by spherical
        spreading (6 dB per doubling). A barrier spends it by its MASS PER AREA
        — that is the whole of it, which is why a lead sheet beats a thick
        curtain and why builders quote kg/m2. It is read off `smass`, so it is
        derived, not declared: 5 cm of masonry is ~135 kg/m2 and eats ~50 dB,
        the same thickness of pine is ~30 kg/m2 and eats ~37 dB, and a door
        hung with a gap eats less again because less mass is in the way.

        This replaces a flat "every solid voxel costs 4 m", which charged stone
        and pine and a leaf curtain exactly the same. The mass is summed over
        the whole path and the law applied ONCE — a wall's loss comes from its
        total mass, not per slice, or six voxels of stone would silence a
        thunderclap. Ears have no cone."""
        a = np.asarray(a, np.float32)
        b = np.asarray(b, np.float32)
        vox_m = 0.1 * self.scale
        dist_m = max(float(np.linalg.norm(b - a)) * vox_m, 1.0)
        level = SHOUT_DB - 20.0 * np.log10(dist_m)           # spreading
        n = int(np.ceil(np.abs(b - a).max())) + 1
        if n > 2:
            ts = np.linspace(0.0, 1.0, n)[1:-1]
            pts = np.round(a[None] + ts[:, None] * (b - a)[None]).astype(np.int64)
            nx, ny, nz = self.shape
            pts[:, 0] = np.clip(pts[:, 0], 0, nx - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, ny - 1)
            pts[:, 2] = np.clip(pts[:, 2], 0, nz - 1)
            kg = float(self.smass[tuple(pts.T)].sum()) / 1000.0
            sigma = kg / max(vox_m * vox_m, 1e-9)            # kg per square meter
            if sigma > 0.0:                                  # mass law at 500 Hz
                level -= max(0.0, 20.0 * np.log10(sigma * 500.0) - 47.0)
        return bool(level >= HEAR_DB)

    def _say(self, p, response):
        text = p.get("lines", LINES).get(response, LINES.get(response))
        if text:
            self.speech.append((self.tick, p["name"], text))
            p["events"].append(f"t{self.tick}: {p['name']} shouts: \"{text}\"")
            p["_shouted"] = self.tick

    def _walkable(self, cells):
        """Columns this body could STAND in: clear enough to occupy, and with
        something under them to stand on.

        The floor test is not a detail. Without it a column of open air reads
        as walkable, so the route planner would happily march a body straight
        out over a chasm and the legs would carry it — walking on nothing,
        because nothing ever asked what was underneath. It also makes a gap a
        real obstacle, which is what gives jumping across one a point."""
        zlo, zhi = int(cells[:, 2].min()), int(cells[:, 2].max())
        col = (slice(None), slice(None), slice(zlo, zhi + 1))
        packed = self.smass[col] / np.maximum(
            _DENS_ARR[self.mat[col]] * self.vox_l, 1e-9)
        clear = (packed < PUSH_THROUGH).all(axis=2)
        if zlo <= 0:
            return clear
        return clear & (self.mat[:, :, zlo - 1] != AIR)

    def _bearing(self, dx, dy):
        """Which way that is, in words. A label, so a menu row reads like
        something a person would say to themselves."""
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            return "here"
        return _DIRS[int(round(float(np.arctan2(dy, dx)) / (np.pi / 4))) % 8]

    def _reach_along(self, fit, ax_, ay_, dx, dy, reach):
        """The furthest this body could get going THAT WAY without turning.

        This is what makes "straight on" and "to the left" real destinations
        rather than steps. A body that has not seen anywhere worth naming can
        still have an intention about a direction, and the legs still do the
        walking — the difference between choosing a way to go and being asked
        to place your feet."""
        n = float(np.hypot(dx, dy))
        if n < 1e-9:
            return None
        dx, dy = dx / n, dy / n
        nx, ny = fit.shape
        last = None
        for k in range(1, int(reach) + 1):
            x, y = int(round(ax_ + dx * k)), int(round(ay_ + dy * k))
            if not (0 <= x < nx and 0 <= y < ny) or not fit[x, y]:
                break
            last = (x, y)
        if last is None or max(abs(last[0] - ax_), abs(last[1] - ay_)) < 2.0:
            return None                   # nowhere to get to that way
        return last

    def _places(self, p, cells, fit):
        """Every spot this body could NAME, drawn from its OWN map.

        A place is where an intention points. They are named, and the naming
        matters: "explore" told a body to go and look at something the SIM
        picked, which is a decision taken away from whoever is choosing. Now
        the unseen edges are listed one per direction, so going to look at the
        dark doorway north is a different choice from going to look at the
        unlit end of the hall, and the body makes it.

        An exit it has never laid eyes on is not a destination — it is not even
        an option — so a stranger in a burning house must find the door the
        hard way and a resident (add_person(knows_world=True)) runs straight
        for it. That difference used to be impossible to express: everyone read
        the world's register."""
        ax_ = float(cells[:, 0].mean()); ay_ = float(cells[:, 1].mean())
        vox_m = 0.1 * self.scale
        out = []

        def add(tag, label, xy):
            d = max(abs(xy[0] - ax_), abs(xy[1] - ay_))
            out.append({"tag": tag, "label": label, "xy": (int(xy[0]), int(xy[1])),
                        "away": d})

        for e in self.exits:                       # doors it has SEEN
            if p["known"][int(e[0]), int(e[1])]:
                add("go:exit", "the door " + self._bearing(e[0] - ax_, e[1] - ay_),
                    (e[0], e[1]))
        # THE UNSEEN EDGES: somewhere known that touches somewhere unknown.
        # Walking to one is how the map grows, and it is why a body that has
        # been dropped into a dark building does not simply stand in the dark.
        unseen = ~p["known"]
        edge = np.zeros_like(fit)
        edge[1:, :] |= unseen[:-1, :]; edge[:-1, :] |= unseen[1:, :]
        edge[:, 1:] |= unseen[:, :-1]; edge[:, :-1] |= unseen[:, 1:]
        front = np.argwhere(fit & edge & p["known"])
        if len(front):
            d = np.abs(front[:, 0] - ax_) + np.abs(front[:, 1] - ay_)
            near = max(8.0, WILL["arrive_m"] / max(vox_m, 1e-9) * 3)
            keep = front[d > near] if (d > near).any() else front
            kd = np.abs(keep[:, 0] - ax_) + np.abs(keep[:, 1] - ay_)
            seen_dirs = {}
            for i in np.argsort(kd):               # nearest first, one per way
                c = keep[i]
                w = self._bearing(c[0] - ax_, c[1] - ay_)
                if w not in seen_dirs:
                    seen_dirs[w] = c
            for w, c in list(seen_dirs.items())[:4]:
                add("go:frontier", "the unseen ground " + w, c)
        # STRAIGHT ON, AND TO EITHER SIDE — relative to the face, because that
        # is how a person says it. Offered only as far as the body could
        # actually get, so a wall in front removes "straight on" rather than
        # offering a walk into it.
        fx, fy = p.get("facing", (1.0, 0.0))
        for name, (dx, dy) in (("straight on", (fx, fy)),
                               ("to the left", (-fy, fx)),
                               ("to the right", (fy, -fx)),
                               ("back the way it came", (-fx, -fy))):
            tgt = self._reach_along(fit, ax_, ay_, dx, dy,
                                    WILL["step_m"] / max(vox_m, 1e-9))
            if tgt is not None:
                add("go:step", name, tgt)
        # SOMEWHERE IT KNOWS, a way off. Deterministic: the sim carries no die,
        # so which way a mind wanders comes from the tick and the name.
        spots = np.argwhere(fit & p["known"])
        if len(spots) > 1:
            d = np.abs(spots[:, 0] - ax_) + np.abs(spots[:, 1] - ay_)
            far = spots[d > max(6.0, np.percentile(d, 70))]
            if len(far):
                i = (self.tick // max(WILL["decide_every"], 1)
                     + len(p["name"])) % len(far)
                add("go:roam", "the floor "
                    + self._bearing(far[i][0] - ax_, far[i][1] - ay_), far[i])
        seen = {}                                  # two doors due east are two
        for o in out:                              # rows, so say which is which
            seen.setdefault(o["label"], []).append(o)
        for label, rows in seen.items():
            if len(rows) > 1:
                for r in rows:
                    r["label"] = f"{label} ({r['away'] * vox_m:.0f} m)"
        return out

    def _underfoot(self, cells):
        """Is there anything under this body's feet to push against?

        The FEET, deliberately: this answers whether the legs have something to
        drive into, which is what deciding to jump and deciding to brace both
        need. Whether a body is HELD UP is a different question with a different
        answer — see _contact — and running them together made a man teetering
        on a lip both unable to jump and unable to fall."""
        zlo = int(cells[:, 2].min())
        if zlo <= 0:
            return True
        return any(self.mat[int(c[0]), int(c[1]), zlo - 1] != AIR
                   for c in cells if int(c[2]) == zlo)

    def _contact(self, cells):
        """Which cells of this body are actually resting on something else.

        Not the bottom layer — ANY cell with something solid beneath it that is
        not part of this same body. A man at the very lip has his feet over air
        and a forearm still over the stone, and that forearm is genuinely
        touching the world. Standing on your own foot proves nothing."""
        cells = np.asarray(cells, np.int64)
        on_floor = cells[:, 2] <= 0                   # the floor of the world
        z1 = np.maximum(cells[:, 2] - 1, 0)
        under = self.mat[cells[:, 0], cells[:, 1], z1] != AIR
        ny, nz = self.shape[1], self.shape[2]
        key = (cells[:, 0] * ny + cells[:, 1]) * nz + cells[:, 2]
        return cells[on_floor | (under & ~np.isin(key - 1, key))]

    def _leap_speed(self, cells):
        """The fastest this body can leave the ground, in m/s. The legs put
        legs_N into the floor over a crouch; what is left after holding the
        body's own weight up becomes speed. A heavier body gets less out of the
        same legs, for free, because it is the same arithmetic."""
        kg = float(self.smass[tuple(np.asarray(cells).T)].sum()) / 1000.0
        if kg <= 0.0:
            return 0.0
        net = BODY["legs_N"] / kg - GRAVITY
        if net <= 0.0:
            return 0.0                        # too heavy to lift itself
        return float(np.sqrt(2.0 * net * BODY["crouch_m"]))

    def _leap_targets(self, p, cells, fit, vmax):
        """Where this body could LAND if it jumped, and how far that is.

        A jump straight up is a jump that goes nowhere, and going nowhere was
        all the legs could do: the whole point of having a velocity is that it
        has a direction. These are the furthest spots it could stand on in each
        way it could face, within the range its own legs allow — and crucially
        they are not filtered by whether it could WALK there, which is what
        makes leaping a gap different from stepping over one.

        Range at the best angle is v-squared over g. Nothing here picks a
        height or a distance; the legs and the mass do."""
        if vmax <= 0.0:
            return []
        vox_m = 0.1 * self.scale
        reach = int((vmax * vmax / GRAVITY) / max(vox_m, 1e-9))
        if reach < 2:
            return []
        ax_ = float(cells[:, 0].mean()); ay_ = float(cells[:, 1].mean())
        nx, ny = fit.shape
        fx, fy = p.get("facing", (1.0, 0.0))
        out = []
        for name, (dx, dy) in (("straight on", (fx, fy)),
                               ("to the left", (-fy, fx)),
                               ("to the right", (fy, -fx)),
                               ("back the way it came", (-fx, -fy))):
            n = float(np.hypot(dx, dy))
            if n < 1e-9:
                continue
            ux, uy = dx / n, dy / n
            best = None
            for k in range(2, reach + 1):
                x, y = int(round(ax_ + ux * k)), int(round(ay_ + uy * k))
                if not (0 <= x < nx and 0 <= y < ny):
                    break
                if fit[x, y]:
                    best = (x, y, float(k))       # the furthest good landing
            if best is not None:
                out.append({"label": name, "xy": (best[0], best[1]),
                            "away": best[2]})
        return out

    def _leap(self, p, cells, goal=None):
        """Push off — up, or up and ALONG.

        Aimed at somewhere, the body leaves at the angle that carries furthest
        and at exactly the speed that distance needs (v-squared = g times the
        range), so it lands where it meant to instead of hurling itself as hard
        as it can. A place further than the legs can reach is not offered."""
        vmax = self._leap_speed(cells)
        if vmax <= 0.0:
            return
        if goal is None:
            self._launch(cells, (0.0, 0.0, vmax), owner=p["name"])
            return
        ax_ = float(cells[:, 0].mean()); ay_ = float(cells[:, 1].mean())
        dx, dy = goal[0] - ax_, goal[1] - ay_
        n = float(np.hypot(dx, dy))
        if n < 1e-9:
            self._launch(cells, (0.0, 0.0, vmax), owner=p["name"])
            return
        d = n * 0.1 * self.scale
        v = min(float(np.sqrt(GRAVITY * d)), vmax)      # just far enough
        half = v / float(np.sqrt(2.0))                  # 45 degrees: best range
        self._launch(cells, (half * dx / n, half * dy / n, half),
                     owner=p["name"])

    def _in_reach(self, mine, theirs):
        """Is that thing close enough to get a hand to?

        Measured between the two SURFACES, not the two centres. A body is five
        voxels across, so a centre-to-centre limit is a different limit for a
        child and for a cart, and the arm is the same arm."""
        gap = max(abs(float(theirs[:, 0].mean()) - float(mine[:, 0].mean())),
                  abs(float(theirs[:, 1].mean()) - float(mine[:, 1].mean())))
        half = 0.5 * (self._span_xy(mine) + self._span_xy(theirs))
        return gap - half <= BODY["reach_m"] / max(0.1 * self.scale, 1e-9)

    @staticmethod
    def _span_xy(cells):
        """How wide that thing is on the floor, in voxels."""
        return float(max(np.ptp(cells[:, 0]), np.ptp(cells[:, 1])))

    def _resist(self, q, cells):
        """What it costs to move a body that would rather not be moved, in
        newtons.

        Friction is all a limp body gives you. A body that is AWAKE, alive and
        has something under its feet BRACES — it puts its own strength into the
        floor against you — and that is one number rather than a rule about
        rescues and a second rule about fights. Everything interesting falls
        out of the arithmetic: an unconscious man can be dragged and a standing
        one cannot, the same standing man CAN be moved by someone stronger, and
        he stops resisting the instant his feet have nothing to push against.
        Nobody wrote any of those three down.

        Before this, a grip was legal only on someone already unconscious. That
        was a case wearing a flag (Ruling 1): it made rescuing the only reason
        two people ever touched, and it made 'pull him off the ledge'
        unaskable — not hard, not physically refused, simply absent from the
        menu."""
        _lift, drag_N = self._effort(cells)
        if q["awake"] and q["alive"] and self._underfoot(cells):
            drag_N += q.get("strength_N", BODY["strength_N"])
        return drag_N

    def _within_reach(self, p, cells):
        """People this body could get a hand to: alive, and near.

        REACH decides whether a grip is possible; FORCE decides what the grip
        can then do, and it is asked again every tick in _haul. The force test
        used to live here, which quietly said a body knows before touching you
        whether you can be moved — and worse, it answered ONCE, so a brace
        that fails the moment a man's heel goes over an edge could never change
        the answer. Taking hold of someone stronger than you is a perfectly
        legal thing to do. It just does not move them."""
        out = []
        for q in self.persons:
            if q is p or not q["alive"] or q["safe"]:
                continue
            qc, qsl = self._person_cells(q)
            if qc is None or not qc.any():
                continue
            qcell = np.argwhere(qc)
            qcell[:, 0] += qsl[0].start
            qcell[:, 1] += qsl[1].start
            if self._in_reach(cells, qcell):
                out.append(q)
        return out

    def _objects_within_reach(self, p, cells):
        """Loose THINGS a hand could close on: solid, not flesh, near, and
        light enough that this body could at least drag them. The last is a
        model of attention rather than a law — every wall in the world is
        within reach of somebody and on nobody's menu. A thing is what
        _object_at says it is: whatever is joined to what you grabbed."""
        vox_m = 0.1 * self.scale
        R = max(int(round(BODY["reach_m"] / max(vox_m, 1e-9))), 1)
        nx, ny, nz = self.shape
        cells = np.asarray(cells)
        x0 = max(int(cells[:, 0].min()) - R, 0)
        x1 = min(int(cells[:, 0].max()) + R + 1, nx)
        y0 = max(int(cells[:, 1].min()) - R, 0)
        y1 = min(int(cells[:, 1].max()) + R + 1, ny)
        z0 = max(int(cells[:, 2].min()) - R, 0)
        z1 = min(int(cells[:, 2].max()) + R + 1, nz)
        box = self.mat[x0:x1, y0:y1, z0:z1]
        cand = np.argwhere((box != AIR) & (box != FLESH))
        if not len(cand):
            return []
        cand += np.array([x0, y0, z0])
        d = np.abs(cand[:, None, :] - cells[None, :, :]).max(axis=2).min(axis=1)
        cand = cand[d <= R]
        out, seen = [], set()
        strength = p.get("strength_N", BODY["strength_N"])
        for c in map(tuple, cand):
            if c in seen:
                continue
            obj = self._object_at(*c)
            seen.update(map(tuple, obj))
            if len(obj) >= 4000:
                continue                  # hit the flood cap: that is the world
            _lift, drag_N = self._effort(obj)
            if drag_N > strength:
                continue                  # could not even shift it: not a thing
                                          # worth a slot of attention
            m0 = int(self.mat[c])
            out.append({"cell": tuple(map(int, obj[0])), "mat": m0,
                        "label": MATNAME.get(m0, "thing")})
            if len(out) >= 2:
                break
        return out

    def _held_cells(self, p):
        """Where the held THING is now — re-flooded from the gripped cell,
        because objects have no registry: identity is adjacency, checked
        against the material the hand closed on. Gone (burned, shattered,
        knocked away) means gone."""
        held = p.get("held")
        if not held:
            return None
        x, y, z = held["cell"]
        nx, ny, nz = self.shape
        if not (0 <= x < nx and 0 <= y < ny and 0 <= z < nz) \
                or int(self.mat[x, y, z]) != held["mat"]:
            # IT MAY BE IN THE AIR — mid-swing, the thing is off the lattice
            # with the arm that swings it. A grip is not lost because it
            # moved (the same lesson _haul learned from fainting bodies).
            for b in self.bodies:
                if b.get("owner") == p["name"] and (b["mats"] != FLESH).any():
                    return None
            p["held"] = None
            return None
        obj = self._object_at(x, y, z)
        return obj.astype(np.int64) if len(obj) else None

    def _grip_holds(self, p, cluster):
        """Whether this body's grip can CARRY that load where it is: strength
        against weight, the holder's own feet against the floor — and a hand
        holds things UP, so a load starting at or above the holder's own crown
        is not hanging from anything. (Reach already ignores height, a stated
        softness: an ankle 1.5 m overhead can be GRIPPED, but not borne.)

        A FIST CAN PULL, NOT CLAMP. What hangs BELOW the hold is a pendulum
        and asks the wrist for nothing; a load whose weight rides ABOVE the
        hold is balanced on the fist, and the wrist pays weight times lever
        to keep it upright. An axe at the fist costs a few newton-metres and
        is held; a man gripped by the ankle is hundreds, and rotates out of
        the hand — the grip on the ankle survives, the CARRY does not."""
        if not p["alive"] or not p["awake"] or p["safe"]:
            return False
        mine, msl = self._person_cells(p)
        if mine is None or not mine.any():
            return False
        own = np.argwhere(mine)
        own[:, 0] += msl[0].start
        own[:, 1] += msl[1].start
        if not self._underfoot(own):
            return False
        cluster = np.asarray(cluster)
        if float(cluster[:, 2].min()) >= float(own[:, 2].max()):
            return False
        w_N, _ = self._effort(cluster)
        if w_N > p.get("strength_N", BODY["strength_N"]):
            return False
        # THE FIST IS AT THE END OF AN ARM, not wherever two bodies happen to
        # press together — measured against the nearest touching cell, a man
        # landed against his holder's chest read as gripped at the chest, and
        # hung there. The load is held at its cell nearest a fist.
        fists = []
        for k in (p.get("segs") or {}):
            if "arm" not in k:
                continue
            limb = self._limb_cells(p, k, own)
            if limb is not None and len(limb):
                fists.append([float(limb[:, 0].mean()),
                              float(limb[:, 1].mean()),
                              float(limb[:, 2].min())])       # a hand hangs at
        if not fists:                                         # the arm's foot
            zs = own[:, 2]
            fists = [[float(own[:, 0].mean()), float(own[:, 1].mean()),
                      float(zs.min()) + 0.55 * float(zs.max() - zs.min())]]
        d2 = ((cluster[None, :, :].astype(np.float64)
               - np.asarray(fists)[:, None, :]) ** 2).sum(-1)
        touch_z = float(cluster[int(d2.min(0).argmin()), 2])
        kg = self.smass[tuple(cluster.T)]
        com_z = float((cluster[:, 2] * kg).sum() / max(float(kg.sum()), 1e-9))
        lever_m = max(0.0, com_z - touch_z) * 0.1 * self.scale
        return w_N * lever_m <= BODY["wrist_Nm"]

    def _grip_cells(self):
        """Every cell a HAND is holding up, for the support law. A grip is an
        edge in the support graph: what a footed body holds, within its
        strength, is supported THROUGH the body — the way anything in a hand
        is — not through the lattice, which is why a hanging man does not
        need his flesh to span like a girder."""
        if not any(p.get("held") or p.get("dragging") for p in self.persons):
            return frozenset()
        out = set()
        for p in self.persons:
            clusters = []
            obj = self._held_cells(p)
            if obj is not None:
                clusters.append(obj)
            who = p.get("dragging")
            if who:
                q = next((r for r in self.persons if r["name"] == who), None)
                if q is not None and q["alive"] and not q["safe"]:
                    comp, sl = self._person_cells(q)
                    if comp is not None and comp.any():
                        qc = np.argwhere(comp)
                        qc[:, 0] += sl[0].start
                        qc[:, 1] += sl[1].start
                        clusters.append(qc)
            for cl in clusters:
                if self._grip_holds(p, cl):
                    out.update(map(tuple, np.asarray(cl)))
        return frozenset(out)

    def _take_up(self, p, own):
        """Bring a held thing to the HAND, if the arm can lift it. A liftable
        thing hangs from the fist — where a swing will find it, and where the
        grip (not the floor) carries it. Too heavy stays where it lies: still
        held, a grip is a constraint before it is a lift."""
        obj = self._held_cells(p)
        if obj is None:
            return
        w_N, _ = self._effort(obj)
        if w_N > p.get("strength_N", BODY["strength_N"]):
            return
        limb = self._limb_cells(p, "right arm", own)
        if limb is None:
            return
        hand = limb[np.argmin(limb[:, 2])]
        top = int(obj[:, 2].max())
        d = (int(hand[0]) - int(round(float(obj[:, 0].mean()))),
             int(hand[1]) - int(round(float(obj[:, 1].mean()))),
             int(hand[2]) - 1 - top)
        if d != (0, 0, 0) and self._shove(obj, *d):
            c = p["held"]["cell"]
            p["held"]["cell"] = (c[0] + d[0], c[1] + d[1], c[2] + d[2])

    def _haul_held(self, p):
        """Bring the held THING along, exactly as _haul brings a person: it
        trails toward the hauler, and an arm stretched past its reach is not
        holding anything. No brace, no resistance — things do not fight."""
        obj = self._held_cells(p)
        if obj is None:
            return
        mine, msl = self._person_cells(p)
        if mine is None or not mine.any():
            return
        hcell = np.argwhere(mine)
        hcell[:, 0] += msl[0].start
        hcell[:, 1] += msl[1].start
        if not self._in_reach(hcell, obj):
            p["held"] = None
            p["events"].append(f"t{self.tick}: {p['name']} loses hold of "
                               f"the {MATNAME.get(int(self.mat[tuple(obj[0])]), 'thing')} "
                               f"— an arm is only so long")
            return
        _lift, drag_N = self._effort(obj)
        if min(_lift, drag_N) > p.get("strength_N", BODY["strength_N"]):
            return                            # the grip holds; the pull fails
        dx = float(hcell[:, 0].mean()) - float(obj[:, 0].mean())
        dy = float(hcell[:, 1].mean()) - float(obj[:, 1].mean())
        if max(abs(dx), abs(dy)) <= 0.5 * (self._span_xy(hcell)
                                           + self._span_xy(obj)):
            return                            # already in hand
        order = ((int(np.sign(dx)), 0), (0, int(np.sign(dy)))) \
            if abs(dx) >= abs(dy) else ((0, int(np.sign(dy))), (int(np.sign(dx)), 0))
        for step in order:
            if step != (0, 0) and self._shove(obj, *step):
                c = p["held"]["cell"]
                p["held"]["cell"] = (c[0] + step[0], c[1] + step[1], c[2])
                return

    def _menu(self, p, cells, percept, limb, chosen):
        """Everything ONE PART of this body could actually do, right now.

        Legality is CHECKED, not assumed. A place is only offered if a
        breadth-first route to it really exists, so an option that gets picked
        can never fail to happen. That is the property worth paying for:
        whatever is choosing cannot invent a door, because it never sees one
        that isn't there.

        One menu per limb, rather than one menu of every combination. Three
        parts with eight, three and two things to do is forty-eight
        combinations and thirteen rows, and the thirteen say everything the
        forty-eight do. That is not a saving trick — it is the shape a body
        actually has, and it is why nothing has to be written down twice when
        a new way to use a mouth arrives.

        Index 0 is always the NULL act: the part does nothing, which is always
        legal and never a failure. Hands doing nothing keep their grip."""
        menu = []
        if limb == "legs":
            menu.append({"key": "stay", "tag": "stay", "verb": "stay"})
            # a CARRIED thing is part of the walker, to the planner as much as
            # to the legs: its column is not an obstacle to its own carrier,
            # or "straight on" vanishes from the menu the moment the fist
            # closes on a stick hanging in front of the chest
            obj = self._held_cells(p) if p.get("held") else None
            nav = np.concatenate([cells, obj]) if obj is not None \
                and not self._underfoot(obj) else cells
            fit, _start = self._fit_grid(nav, p.get("known"))
            ground = self._underfoot(cells)
            places = self._places(p, cells, fit)
            leaps = self._leap_targets(p, cells, fit, self._leap_speed(cells)) \
                if ground else []
            routes = self._routes(cells, [pl["xy"] for pl in places]
                                  + [lp["xy"] for lp in leaps],
                                  p.get("known"), fit=fit)
            if ground:                        # you cannot push off thin air
                menu.append({"key": "jump", "tag": "jump", "verb": "jump"})
            for lp in leaps:
                if routes.get(lp["xy"]):
                    continue                  # it could just WALK there. A leap
                                              # is worth weighing where the legs
                                              # cannot carry it — over a gap, not
                                              # across an open floor — and four
                                              # redundant leaps crowded a body's
                                              # whole attention out of the menu
                menu.append({"key": f"leap({lp['label']})", "tag": "leap",
                             "verb": "jump", "goal": lp["xy"],
                             "away": lp["away"]})
            for pl in places:
                route = routes.get(pl["xy"])
                if not route:
                    continue                  # no route it knows of: not legal
                menu.append({"key": f"go({pl['label']})", "tag": pl["tag"],
                             "verb": "go", "goal": pl["xy"], "route": route,
                             "away": pl["away"]})
        elif limb == "hands":
            held = p.get("dragging") or \
                ("the " + p["held"]["label"] if p.get("held") else None)
            menu.append({"key": "keep hold of " + held if held else "hands free",
                         "tag": "keep", "verb": "keep"})
            if held:
                menu.append({"key": "let go", "tag": "let_go", "verb": "let_go"})
            if self._limb_cells(p, "right arm", cells) is not None:
                # swinging at NOTHING is possible, just useless — the same way
                # shouting in an empty house is possible. What makes it legal
                # is having an arm, not having a target. A held thing swings
                # WITH the arm; that is most of what holding a tool is for
                menu.append({"key": "swing an arm" if not p.get("held") else
                             f"swing the {p['held']['label']}", "tag": "swing",
                             "verb": "swing"})
            for q in self._within_reach(p, cells):
                if q["name"] == held:
                    continue
                menu.append({"key": f"take hold of {q['name']}", "tag": "hold",
                             "verb": "hold", "who": q["name"]})
            if not p.get("held"):
                for ob in self._objects_within_reach(p, cells):
                    menu.append({"key": f"take hold of the {ob['label']}",
                                 "tag": "hold", "verb": "hold", "what": ob})
        elif limb == "mouth":
            menu.append({"key": "say nothing", "tag": "quiet", "verb": "quiet"})
            for name, text in p.get("lines", LINES).items():
                if not text:
                    continue                  # this body has no such words
                if ANSWERS.get(name) not in (None, percept):
                    continue                  # nothing to answer
                menu.append({"key": f"say({name})", "tag": f"say:{name}",
                             "verb": "say", "line": name})
        if len(menu) > MENU_CAP:              # attention is the scarce thing.
            want = (p.get("reflexes", REFLEXES).get(percept) or {}).get(limb) \
                if percept else None
            head, tail = menu[0], menu[1:]    # the null act never falls off
            keep = [o for o in tail if o["tag"] == want]
            rest = sorted((o for o in tail if o not in keep),
                          key=lambda o: o.get("away", 0.0))
            menu = ([head] + keep + rest)[:MENU_CAP]
        return menu

    def _decide(self, p, percept, cells, ax_, ay_, eye, vox_m):
        """Offer one menu per part of the body, let the policy pick from each,
        carry the whole program out, log it.

        The result is a PROGRAM — go here AND hold them AND say this — built
        by composition rather than looked up. Every combination the old table
        spelled out still happens, and combinations nobody spelled out happen
        too: holding someone while walking deeper in, calling a warning while
        wandering, taking hold with no idea where the door is."""
        # who would actually HEAR a shout. Not a legality gate — shouting in
        # an empty house is possible, just useless — but a fact a thinking
        # policy should get to weigh, so it rides along in the situation.
        heard = sum(1 for q in self.persons
                    if q is not p and q["alive"] and q["awake"]
                    and self._hears(eye, q.get("_eye", (*q["anchor"], eye[2]))))
        sit = {"tick": self.tick, "who": p["name"], "percept": percept,
               "reflexes": p.get("reflexes", REFLEXES),
               "smoke": round(float(p["smoke"]), 3),
               "blood_o2": round(float(p["blood_o2"]), 3),
               "burn": round(float(p["burn"]), 3),
               "hurt": round(float(p["hurt"]), 3),
               "holding": p.get("dragging") or
                          ("the " + p["held"]["label"] if p.get("held") else None),
               "knows_a_way_out": any(p["known"][int(e[0]), int(e[1])]
                                      for e in self.exits),
               "seen_of_the_world": round(float(p["known"].mean()), 3),
               "others_in_earshot": heard}
        menus, picks, chosen = {}, {}, {}
        for limb in LIMBS:
            menu = self._menu(p, cells, percept, limb, chosen)
            if not menu:
                continue
            i = self.policy.pick(dict(sit, limb=limb, chosen=dict(chosen)), menu)
            i = i if isinstance(i, int) and 0 <= i < len(menu) else 0
            opt = menu[i]
            menus[limb] = [o["key"] for o in menu]
            picks[limb] = opt
            chosen[limb] = opt["tag"]
        if not picks:
            return
        prog = " & ".join(picks[l]["key"] for l in LIMBS
                          if l in picks and not ACTS[picks[l]["verb"]]["null"])
        self.traces.append({"tick": self.tick, "who": p["name"],
                            "percept": percept, "situation": sit,
                            "menus": menus,
                            "pick": {l: o["key"] for l, o in picks.items()},
                            "tags": dict(chosen), "program": prog or "stay",
                            "by": getattr(self.policy, "name", "?")})
        p["_decided"] = self.tick
        if percept is not None:
            p["emergency"] = True        # committed: stop weighing the ordinary
        hands = picks.get("hands")
        if hands is not None:            # HANDS. Holding is not a rescue
            if hands["verb"] == "swing":
                self._swing(p, "right arm", toward=p.get("facing"))
                p["events"].append(f"t{self.tick}: {p['name']} swings an arm")
            elif hands["verb"] == "hold":  # no subsystem: a grip is REACH, and
                if "who" in hands:             # what it can then do is force —
                    p["dragging"] = hands["who"]   # the same _effort a shove
                    p["events"].append(f"t{self.tick}: {p['name']} takes hold "
                                       f"of {hands['who']}")   # uses, asked
                else:                              # again every tick
                    ob = hands["what"]
                    p["held"] = {"cell": ob["cell"], "mat": ob["mat"],
                                 "label": ob["label"]}
                    self._take_up(p, cells)    # to the fist, if the arm can
                    p["events"].append(f"t{self.tick}: {p['name']} takes up "
                                       f"the {ob['label']}")
            elif hands["verb"] == "let_go":
                p["dragging"], p["held"] = None, None
                p["events"].append(f"t{self.tick}: {p['name']} lets go")
        legs = picks.get("legs")
        if legs is not None and legs["verb"] == "jump":
            self._leap(p, cells, legs.get("goal"))
            p["goal"], p["fleeing"] = None, False
            p["events"].append(f"t{self.tick}: {p['name']} jumps")
        elif legs is not None and legs["verb"] == "go":
            # FLEEING means driven, not merely walking. Every `go` used to set
            # it, which was harmless while the only reason to walk was an
            # alarm and wrong the moment idling became a real choice: a person
            # strolling to the window then read to everyone else as someone
            # BOLTING PAST, and set off alarms that had no fire behind them.
            p["fleeing"] = percept is not None
            p["goal"] = legs["goal"]
            p["_path"], p["_planned"] = legs["route"], self.tick
            if percept:
                p["events"].append(f"t{self.tick}: {p['name']} startles "
                                   f"({percept.replace('_', ' ')}) and makes "
                                   f"for {legs['key'][3:-1]}")
            else:
                p["events"].append(f"t{self.tick}: {p['name']} makes for "
                                   f"{legs['key'][3:-1]}")
        elif legs is not None:
            p["goal"] = None
            if percept:
                p["events"].append(f"t{self.tick}: {p['name']} notices "
                                   f"({percept.replace('_', ' ')}) and stays put")
        mouth = picks.get("mouth")
        if mouth is not None and mouth["verb"] == "say":
            self._say(p, mouth["line"])

    def trace_outcomes(self):
        """Stamp every logged decision with how that body ENDED, and hand the
        rows back. This is the blessing signal the harvest needs: a pick made
        by a person who walked out is worth learning from, one made by a
        person who did not is worth learning against.

        It is not yet menu RECALL — knowing whether a better option was on
        the menu and passed over needs the counterfactual run, which we
        cannot do here. What this does give is the pairing without which no
        recall number can ever be computed: the whole menu, the pick, the
        end."""
        end = {}
        for p in self.persons:
            end[p["name"]] = ("safe" if p["safe"] else "dead" if not p["alive"]
                              else "down" if not p["awake"] else "still inside")
        for row in self.traces:
            row["outcome"] = end.get(row["who"], "unknown")
        return self.traces

    def _law_will(self):
        """The WILL layer, in three separate parts — this is the MENU SEAM.

        1. NOTICE. The body's senses fire a percept: it sees flame in its
           cone, coughs on smoke, hears a shout through a wall, watches
           someone bolt past. Physics decides this; nothing chooses.
        2. OFFER. The sim builds the MENU: every response this body could
           actually carry out this instant, each one checked against the
           world (a "flee" option exists only if a route to a door really
           exists). Nothing chooses here either.
        3. PICK. A POLICY takes the situation and the menu and returns one
           index. Today that is the reflex table and the answer is the same
           one the table always gave. Tomorrow it can be a language model,
           and later a model distilled from these very decisions.

        The split is the whole point. Because the menu is built from what the
        sim just computed, a picked option cannot be illegal — the mind
        supplies taste, never facts, and can never invent a door. Every pick
        is written to `self.traces` with the full menu beside it, which is
        what makes a harvest possible: you cannot learn from a choice without
        knowing what else was on offer."""
        # A ROOM WITH NO DOOR STILL HAS PEOPLE IN IT. This used to return unless
        # the world registered an exit, which quietly made every mind in the sim
        # conditional on there being somewhere to escape to: a scene with no door
        # — two men on a clifftop, say — had bodies that could not decide
        # anything at all, and not because anything physical stopped them.
        # Escaping is one thing a body might want, not the reason it has a will.
        if not self.persons:
            return
        vox_m = 0.1 * self.scale
        burning = self.burning() if any(p["alive"] and p["awake"]
                                        for p in self.persons) else None
        fire_at = np.argwhere(burning) if burning is not None and burning.any() else None
        for p in self.persons:
            if not (p["alive"] and p["awake"]) or p["safe"]:
                continue
            comp, sl = self._person_cells(p)
            if comp is None:
                continue
            cells = np.argwhere(comp)
            cells[:, 0] += sl[0].start
            cells[:, 1] += sl[1].start
            ax_, ay_ = (float(cells[:, 0].mean()), float(cells[:, 1].mean()))
            p["anchor"] = (int(round(ax_)), int(round(ay_)))
            eye = (ax_, ay_, float(cells[:, 2].max()))
            p["_eye"] = eye
            if "facing" not in p:
                p["facing"] = (1.0, 0.0)
            # THE HEAD IS NOT THE FEET. A gaze sweeps around whichever way the
            # body is pointed, so a person crossing a room still looks about as
            # they go. It used to sweep by turning `facing` itself, and only
            # while the body had nowhere to be — survivable while standing
            # still was common, and a blindfold the moment bodies always had
            # somewhere to go: measured, a witness with a clear window onto a
            # burning bench walked about in front of it for 260 ticks and never
            # once perceived the fire, because one vector was doing both jobs
            # and every step reset it.
            if self.tick % WILL["scan_every"] == 0 and not p.get("emergency"):
                p["gaze_i"] = (p.get("gaze_i", 0) + 1) % len(GAZE_SWEEP)
            off = 0.0 if p.get("emergency") else GAZE_SWEEP[p.get("gaze_i", 0)]
            fx, fy = p["facing"]
            cg, sg = np.cos(off), np.sin(off)
            gaze = (fx * cg - fy * sg, fx * sg + fy * cg)
            p["gaze"] = gaze                  # what the eyes are pointed at
            if self.tick % 4 == 0:            # look where the eyes point, and
                self._look_around(p, eye)     # REMEMBER it
            # SENSES RUN WHETHER OR NOT THE BODY IS BUSY. Having somewhere to
            # be is not the same as being in an emergency: a body strolling
            # across a room must still notice the room is alight. Gating the
            # percepts on "is moving" made a wandering person blind.
            if not p.get("emergency"):
                percept = None
                if fire_at is not None:
                    d = np.abs(fire_at[:, :2] - np.array([ax_, ay_])).max(axis=1)
                    close = fire_at[d * vox_m <= WILL["fire_see_m"]]
                    for c in close[np.argsort(
                            np.abs(close[:, :2] - np.array([ax_, ay_]))
                            .max(axis=1))][:25]:
                        if self._in_cone(gaze, eye, c) \
                                and self._sees(eye, c.astype(np.float32)):
                            percept = "sees_fire"    # an actual LINE to a flame
                            break                    # the face is TURNED toward
                if percept is None and p["smoke"] > 0.05:
                    percept = "chokes"                   # coughing IS noticing
                if percept is None:
                    for q in self.persons:
                        if q is p or not q["fleeing"] or q["safe"]:
                            continue
                        qeye = q.get("_eye", (*q["anchor"], eye[2]))
                        dq = max(abs(q["anchor"][0] - ax_),
                                 abs(q["anchor"][1] - ay_))
                        if (q.get("_shouted") is not None
                                and self.tick - q["_shouted"] < 30
                                and self._hears(eye, qeye)):
                            percept = "hears_alarm"      # ears have no cone and
                            break                        # a wall only MUFFLES
                        if (dq * vox_m <= WILL["run_see_m"]
                                and self._in_cone(gaze, eye, qeye)
                                and self._sees(eye, qeye)):
                            percept = "sees_runner"
                            break
                # a body with NOTHING happening to it still has a life. If no
                # percept fires it weighs its ordinary options — go and look at
                # what it has not seen, walk somewhere it has, or stand still —
                # through the same menu, picked by the same policy, written to
                # the same trace. Idling is a choice, not a gap between choices.
                # A PERCEPT DOES NOT WAIT ITS TURN. decide_every paces ordinary
                # deliberation — nobody re-weighs their afternoon every tick —
                # but an alarm must land the moment it arrives. Making percepts
                # queue behind the cooldown meant a shout could expire unheard
                # while the hearer was still inside its own 30-tick pause.
                due = self.tick - p.get("_decided", -10 ** 9) >= \
                    WILL["decide_every"]
                if percept is not None or (due and p.get("goal") is None):
                    self._decide(p, percept, cells, ax_, ay_, eye, vox_m)
            goal = p.get("goal")
            if goal is None:
                p["fleeing"] = False
                continue
            # arrival is measured from the BODY, not from its centre: you are
            # at the door when part of you is at the door. A doorway in an
            # outer wall has no room for a centre, so measuring from the middle
            # meant a body could stand IN the opening and not have arrived.
            reach = float(np.abs(cells[:, :2] - np.array(goal)).max(axis=1).min())
            if reach * vox_m <= WILL["arrive_m"]:
                if tuple(goal) in {tuple(e) for e in self.exits}:
                    p["safe"] = True
                    p["events"].append(f"t{self.tick}: {p['name']} reaches the doorway")
                    self._leave(p, cells)
                    who = p.get("dragging")
                    if who:                       # whoever came with them is out
                        for q in self.persons:    # too — that is all rescuing is
                            if q["name"] == who and q["alive"]:
                                q["safe"] = True
                                qc, qsl = self._person_cells(q)
                                if qc is not None and qc.any():
                                    qcell = np.argwhere(qc)
                                    qcell[:, 0] += qsl[0].start
                                    qcell[:, 1] += qsl[1].start
                                    self._leave(q, qcell)
                                p["events"].append(
                                    f"t{self.tick}: ...hauling {who} clear")
                        p["dragging"] = None
                else:                                # arrived somewhere ordinary:
                    p["goal"] = None                 # pick again next time round
                    p["fleeing"] = False
                    p["_path"] = None
                continue
            ex = goal
            if self.tick % WILL["walk_every"]:
                continue
            # walking is PLANNED, not greedy: legs that jam forever on the
            # first baffle aren't legs (greedy + wall-slide both livelocked,
            # measured). A breadth-first route over stand-able columns is the
            # body's motor competence — knowing the way around the table is
            # not thinking, any more than balance is.
            # WHAT IS CARRIED WALKS WITH THE BODY. A thing hanging from the
            # fist is one kinematic unit with the hand that holds it — walked
            # as separate matter it is a wall in front of its own carrier, and
            # the carrier jams on it forever (and the planner jams first, so
            # the carried cells count as the body's own there too). A held
            # thing still ON the floor (too heavy to lift) is not carried; it
            # TRAILS, like a dragged person does.
            obj = self._held_cells(p) if p.get("held") else None
            carried = obj is not None and not self._underfoot(obj)
            walk_cells = np.concatenate([cells, obj]) if carried else cells
            if (not p.get("_path")) or self.tick - p.get("_planned", -99) > 30:
                p["_path"] = self._plan_path(walk_cells, ex, p.get("known"))
                p["_planned"] = self.tick
            path = p.get("_path") or []
            while path and max(abs(path[0][0] - ax_), abs(path[0][1] - ay_)) < 1.0:
                path.pop(0)
            if not path:
                continue
            sx = int(np.sign(path[0][0] - ax_))
            sy = int(np.sign(path[0][1] - ay_))
            for step in ((sx, 0), (0, sy)) if abs(path[0][0] - ax_) >= \
                    abs(path[0][1] - ay_) else ((0, sy), (sx, 0)):
                if step == (0, 0):
                    continue
                if self._walk(walk_cells, *step):
                    p["facing"] = (float(step[0]), float(step[1]))
                    if carried:
                        c = p["held"]["cell"]
                        p["held"]["cell"] = (c[0] + step[0],
                                             c[1] + step[1], c[2])
                    self._haul(p, *step)
                    if not carried:
                        self._haul_held(p)
                    break
            else:
                p["_path"] = None                    # blocked mid-route: replan

    def _leave(self, p, cells):
        """Out means OUT. A body that reached the door walks through it and off
        the lattice, the way gas leaves at an open sky — its mass is accounted
        as having left the world, not destroyed inside it.

        This is not tidiness. Bodies have real depth now, so one person standing
        in a doorway they had already escaped through was a wall to everyone
        behind them: measured, a second body stuck three voxels short of the
        exit for 800 ticks because the first was still stood in it."""
        idx = tuple(np.asarray(cells).T)
        self.left_mass += float(self.smass[idx].sum())
        for arr in (self.mat, self.smass, self.E, self.fl,
                    self.fvol, self.fpot, self.fallh, self.edge):
            arr[idx] = 0
        p["_own"] = frozenset()
        p["_claim_mask"], p["_claim_tick"] = None, None
        self._slack_mat = None
        self._torque_solid = None

    def _haul(self, p, sx, sy):
        """Bring along whoever this body took hold of. The dragged body is not
        a passenger with special rules — it is a thing being shoved, moved by
        the same _shove that shifts a crate, and it is left behind the moment
        it will not shift (wedged, or grown too heavy to matter). When both are
        at the door, both are out; that is the only thing rescuing anyone ever
        needed to mean.

        WHAT IS DRAGGED FOLLOWS. It used to be shoved along the hauler's own
        step vector, which is a grip only while the two happen to move alike:
        once the load snagged for a tick, the hauler walked on and the pair
        drifted apart, still joined — measured, a rescuer at x=29 towing a body
        at x=36, seven voxels AHEAD of them and pulling it further. A load
        trails the thing pulling it, so the step is toward the hauler, and an
        arm that has been stretched past its reach is not holding anything."""
        who = p.get("dragging")
        if not who:
            return
        q = next((r for r in self.persons if r["name"] == who), None)
        if q is None or q["safe"] or not q["alive"]:
            p["dragging"] = None
            return
        # THEY MAY BE IN THE AIR. An unconscious body is promoted to a free body
        # while it keels over, and for those ticks it is off the lattice or only
        # half back on it. Grabbing at it then fails — and letting go on that
        # failure ended every rescue at precisely the moment the person being
        # rescued finished falling over, which is when they need carrying most.
        # Wait for them to come down; a grip is not lost because someone moved.
        if any((b["mats"] == FLESH).any() for b in self.bodies):
            return
        comp, sl = self._person_cells(q)
        if comp is None or not comp.any():
            p["dragging"] = None
            return
        cells = np.argwhere(comp)
        cells[:, 0] += sl[0].start
        cells[:, 1] += sl[1].start
        mine, msl = self._person_cells(p)
        if mine is None or not mine.any():
            return
        hcell = np.argwhere(mine)
        hcell[:, 0] += msl[0].start
        hcell[:, 1] += msl[1].start
        if not self._in_reach(hcell, cells):
            p["dragging"] = None                  # stretched past an arm: gone
            p["events"].append(f"t{self.tick}: {p['name']} loses hold of {who} "
                               f"— an arm is only so long")
            return
        # HE MAY BE OVER THE EDGE. With nothing underfoot, what an arm holds is
        # his WEIGHT and not the friction of a floor — the same _effort, its
        # other half. The support law now knows a grip carries load
        # (`_grip_cells`), so a man whose weight the arm can take HANGS from
        # it: no step is made, and "left holding him over the drop" is a state
        # of the world. Hauling him back up is not built. A load beyond the
        # arm — or one wholly above the holder's own crown, which no grip
        # bears — goes, exactly as before.
        if not self._underfoot(cells):
            if self._grip_holds(p, cells):
                if not p.get("_held_over"):
                    p["_held_over"] = True
                    p["events"].append(f"t{self.tick}: {p['name']} is left "
                                       f"holding {who} over the drop")
                return
            p["dragging"] = None
            p["events"].append(f"t{self.tick}: {who} goes over, and the grip "
                               f"is not enough to hold a hanging man")
            return
        p["_held_over"] = False
        # AND HE MAY NOT WANT TO COME. Asked every tick, because the answer
        # changes: the same pull that a braced man shrugs off for twenty ticks
        # succeeds the moment his footing is gone or someone stronger takes over.
        if self._resist(q, cells) > p.get("strength_N", BODY["strength_N"]):
            return                                # the grip holds; the pull fails
        dx = float(hcell[:, 0].mean()) - float(cells[:, 0].mean())
        dy = float(hcell[:, 1].mean()) - float(cells[:, 1].mean())
        if max(abs(dx), abs(dy)) <= 0.5 * (self._span_xy(hcell)
                                           + self._span_xy(cells)):
            return                                # already at their heels
        order = ((int(np.sign(dx)), 0), (0, int(np.sign(dy)))) \
            if abs(dx) >= abs(dy) else ((0, int(np.sign(dy))), (int(np.sign(dx)), 0))
        for step in order:
            if step != (0, 0) and self._shove(cells, *step):
                q["anchor"] = (int(round(float(cells[:, 0].mean()))) + step[0],
                               int(round(float(cells[:, 1].mean()))) + step[1])
                return
        p["dragging"] = None                      # it would not come; let go

    def _fit_grid(self, cells, known=None):
        """Columns this body could stand in, WHOLE — and knows about.

        A column is passable by how much is actually IN it, not by whether it is
        pure air: a body shoves through a hedge or a curtain and does not shove
        through a wall. Same continuous read as sight and sound — before this, a
        single leaf voxel was as impassable as masonry. PUSH_THROUGH is a
        stand-in for a force the body has not got yet; the force law replaces
        it with strength against what holds the material."""
        # ONE rule for where a body may stand, not two. This had its own copy
        # of the fill test, so teaching _walkable that a column needs a floor
        # under it taught the PLACES but not the PLANNER — and the planner is
        # the half that decides where the legs actually go.
        walk_ok = self._walkable(cells)
        if known is not None:
            walk_ok &= known          # you cannot plan a route through rooms
                                      # you have never seen
        # A BODY IS NOT A POINT. The route is walked by the body's CENTRE, but
        # the body is several voxels across, so a centre column is only usable
        # if the whole footprint fits there. Without this the planner happily
        # routes a centre into a column two voxels off a wall, the legs refuse
        # (correctly — the shoulder is in the masonry), and the walker jams
        # against the wall shuffling sideways forever. Measured on the
        # glasshouse: a body stuck 2 voxels short of its door for 500 ticks.
        cx = int(round(float(cells[:, 0].mean())))
        cy = int(round(float(cells[:, 1].mean())))
        for c in cells:                       # a body does not block ITSELF, and
            walk_ok[int(c[0]), int(c[1])] = True   # this has to be true BEFORE
                                              # the footprint test: its own flesh
                                              # is denser than PUSH_THROUGH, so
                                              # eroding first left a body unable
                                              # to stand where it was standing
        fit = walk_ok.copy()
        nx, ny = walk_ok.shape
        for dx, dy in {(int(c[0]) - cx, int(c[1]) - cy) for c in cells}:
            if dx == 0 and dy == 0:
                continue
            sh = np.zeros_like(walk_ok)
            xs_lo, xs_hi = max(0, -dx), min(nx, nx - dx)
            ys_lo, ys_hi = max(0, -dy), min(ny, ny - dy)
            if xs_lo < xs_hi and ys_lo < ys_hi:
                sh[xs_lo:xs_hi, ys_lo:ys_hi] = \
                    walk_ok[xs_lo + dx:xs_hi + dx, ys_lo + dy:ys_hi + dy]
            fit &= sh
        fit[cx, cy] = True                    # wherever it is now, it fits
        return fit, (cx, cy)

    def _routes(self, cells, targets, known=None, fit=None):
        """Every target's route, from ONE breadth-first sweep.

        A menu prices a dozen places at once, and asking the planner a dozen
        separate questions re-walks the same room a dozen times. One sweep
        records where every column was reached FROM, and each route is read
        back out of it — cheaper than the three separate searches this replaced,
        for four times the places."""
        from collections import deque
        if fit is None:
            fit, start = self._fit_grid(cells, known)
        else:
            start = (int(round(float(cells[:, 0].mean()))),
                     int(round(float(cells[:, 1].mean()))))
        nx, ny = fit.shape
        prev = {start: None}
        q = deque([start])
        while q:
            cur = q.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (cur[0] + dx, cur[1] + dy)
                if (0 <= nxt[0] < nx and 0 <= nxt[1] < ny
                        and nxt not in prev and fit[nxt]):
                    prev[nxt] = cur
                    q.append(nxt)
        cand = None
        out = {}
        for t in targets:
            goal = (int(t[0]), int(t[1]))
            if not fit[goal]:                 # a doorway in an outer wall has
                if cand is None:              # no room for a body's centre —
                    cand = np.argwhere(fit)   # aim at the nearest place that does
                if not len(cand):
                    out[(int(t[0]), int(t[1]))] = None
                    continue
                d = np.abs(cand[:, 0] - goal[0]) + np.abs(cand[:, 1] - goal[1])
                near = cand[int(d.argmin())]
                goal = (int(near[0]), int(near[1]))
            if goal not in prev:
                out[(int(t[0]), int(t[1]))] = None                     # no way through
                continue
            path, cur = [], goal
            while cur is not None:
                path.append(cur)
                cur = prev[cur]
            out[(int(t[0]), int(t[1]))] = path[::-1][1:]
        return out

    def _plan_path(self, cells, ex, known=None):
        """Breadth-first route from where this body is to one goal.
        Deterministic, 4-connected, replanned when the world changes underfoot."""
        return self._routes(cells, [ex], known)[(int(ex[0]), int(ex[1]))]

    def _object_at(self, x, y, z, cap=4000):
        """The connected thing that voxel belongs to — same material, flood
        filled. A 'thing' is not declared anywhere; it is whatever is joined to
        whatever you grabbed. (Which is also why a table with iron legs is two
        things: joints do not exist yet.)"""
        m0 = int(self.mat[x, y, z])
        if m0 == AIR:
            return np.zeros((0, 3), np.int32)
        nx, ny, nz = self.shape
        seen = {(x, y, z)}
        stack = [(x, y, z)]
        out = []
        while stack and len(out) < cap:
            cx, cy, cz = stack.pop()
            out.append((cx, cy, cz))
            for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0),
                               (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                t = (cx + dx, cy + dy, cz + dz)
                if t in seen or not (0 <= t[0] < nx and 0 <= t[1] < ny
                                     and 0 <= t[2] < nz):
                    continue
                if int(self.mat[t]) == m0:
                    seen.add(t)
                    stack.append(t)
        return np.array(out, np.int32)

    def _effort(self, cells):
        """What it costs this body to move that thing, in newtons: the weight
        to LIFT it, the friction to SHOVE it along the floor.

        This is the whole of 'hands'. Push, pull, drag, lift and press are not
        five verbs with five rules — they are one force meeting either gravity
        or friction, and the arithmetic decides which are possible. A grown
        person comes out able to shove a loaded chest they could never pick up,
        and — the case that matters — able to DRAG someone unconscious but not
        to carry them, which is exactly how it goes."""
        if not len(cells):
            return 0.0, 0.0
        kg = float(self.smass[tuple(cells.T)].sum()) / 1000.0
        return kg * GRAVITY, kg * GRAVITY * FRICTION

    def _shove(self, cells, dx, dy, dz=0):
        """Translate a thing by one voxel, carrying everything it holds. Refuses
        if any arriving cell is occupied by something that is not itself."""
        if not len(cells):
            return False
        nx, ny, nz = self.shape
        tgt = cells + np.array([dx, dy, dz], np.int32)
        if (tgt < 0).any() or (tgt[:, 0] >= nx).any() \
                or (tgt[:, 1] >= ny).any() or (tgt[:, 2] >= nz).any():
            return False
        own = {tuple(c) for c in cells}
        for t in map(tuple, tgt):
            if t not in own and int(self.mat[t]) != AIR:
                return False
        fields = (self.mat, self.smass, self.E, self.fl, self.fvol,
                  self.fpot, self.fallh, self.edge)
        idx, tdx = tuple(cells.T), tuple(tgt.T)
        held = [arr[idx].copy() for arr in fields]
        for arr in fields:
            arr[idx] = 0
        for arr, h in zip(fields, held):
            arr[tdx] = h
        self._torque_solid = None
        self._slack_mat = None
        return True

    def _walk(self, cells, sx, sy):
        """Translate a body one voxel sideways, SHOVING ASIDE what is light
        enough to shove.

        A step needs its arriving cells to be sparse enough to push through
        (the same fill read the route planner uses — legs and plans must agree,
        or the planner routes through a hedge the legs then refuse). Anything
        packed, and any real puddle, still stops the step.

        What gets shoved has to GO somewhere: the branches the body enters are
        moved into the cells the body leaves, so the hedge closes behind you and
        no leaf is quietly deleted. A rigid step vacates exactly as many cells as
        it enters, so the exchange always balances."""
        tgt = cells.copy()
        tgt[:, 0] += sx
        tgt[:, 1] += sy
        nx, ny, nz = self.shape
        if (tgt[:, 0] < 0).any() or (tgt[:, 0] >= nx).any() \
                or (tgt[:, 1] < 0).any() or (tgt[:, 1] >= ny).any():
            return False
        own = set(map(tuple, cells))
        new = set(map(tuple, tgt))
        entered = [c for c in map(tuple, tgt) if c not in own]
        for t in entered:
            packed = self.smass[t] / max(_DENS_ARR[self.mat[t]] * self.vox_l, 1e-9)
            if packed >= PUSH_THROUGH or self.fvol[t] > 1.0:
                return False
        vacated = [c for c in map(tuple, cells) if c not in new]
        fields = (self.mat, self.smass, self.E, self.fl, self.fvol,
                  self.fpot, self.edge)
        idx, tdx = tuple(cells.T), tuple(tgt.T)
        body = [arr[idx].copy() for arr in fields]
        shoved = [[arr[c] for c in entered] for arr in fields]
        for arr, h in zip(fields, body):                  # the body arrives
            arr[tdx] = h
        for arr, keep in zip(fields, shoved):             # the hedge closes behind
            for c, val in zip(vacated, keep):
                arr[c] = val
        self._torque_solid = None
        return True

    def _law_footing(self):
        """A PERSON IS A RIGID THING, so a person goes over as ONE thing.

        Support is relaxed column by column, which is exactly right for a wall
        and quite wrong for a man. Walked off his own ledge, he came down as
        five separate showers of flesh — every gram still there, the person
        gone — and hooked by a forearm on the lip he came apart the other way,
        the arm staying at z31 while his legs ran down to the floor like sand.
        A jumper never did either, and the reason is that a leap PROMOTES the
        body to a free rigid object first. That promotion was the piece missing
        everywhere else.

        The question is the FEET: a body whose lowest voxels have nothing under
        them is no longer standing on anything. Deliberately not the torque
        law's fuller test of weight-over-footprint, which reads a body already
        lying in a heap as hopelessly overbalanced — most of a slumped body
        rests on the rest of itself — and launched it into the ground it was
        already on, over and over.

        Going over an edge is a rotation about that edge, so a body leaves it
        moving OUTWARD as well as down, and by the time it has turned far
        enough to clear the lip its weight has fallen through about the
        overhang: that is where the sideways speed comes from, and it is the
        only reason the thing still touching is not struck at once. Without it,
        a man tipped off a lip with a forearm over the stone dropped one voxel,
        hit the very rock his arm had been on, and was set back down in the
        same place — three times a second, for ever.

        Nothing here is about ledges. A shove, a haul, a floor that burns out
        from underneath all arrive at the same test, and all pay the same
        1/2 m v² on landing."""
        if not self.persons:
            return
        aloft = {b.get("owner") for b in self.bodies}
        holders = {q["dragging"]: q for q in self.persons if q.get("dragging")}
        for p in self.persons:
            if not p["alive"] or p["safe"] or p["name"] in aloft:
                continue
            comp, sl = self._person_cells(p)
            if comp is None or not comp.any():
                continue
            cells = np.argwhere(comp)
            cells[:, 0] += sl[0].start
            cells[:, 1] += sl[1].start
            if self._underfoot(cells):
                continue                      # still stood on something
            h = holders.get(p["name"])
            if h is not None and self._grip_holds(h, cells):
                continue                      # he HANGS from the grip: a hand
                                              # that can lift him can hold him
            vel = [0.0, 0.0, 0.0]
            foot = self._contact(cells)       # ...but is anything still touching?
            if len(foot):
                kg = self.smass[tuple(cells.T)]
                tot = max(float(kg.sum()), 1e-9)
                dx = float((cells[:, 0] * kg).sum()) / tot - float(foot[:, 0].mean())
                dy = float((cells[:, 1] * kg).sum()) / tot - float(foot[:, 1].mean())
                axis = 0 if abs(dx) >= abs(dy) else 1
                o = dx if axis == 0 else dy
                vel[axis] = float(np.sign(o) * np.sqrt(
                    2.0 * GRAVITY * abs(o) * 0.1 * self.scale))
            self._launch(cells, vel, owner=p["name"])
            p["events"].append(f"t{self.tick}: {p['name']} loses their footing")

    def _collapse(self, p, comp, sl):
        """An unconscious body is a slack object: promote it as a free rigid
        body keeling toward the openest side — the same fall as a chopped tree."""
        xs0, ys0 = sl[0].start, sl[1].start
        cells = np.argwhere(comp).astype(np.int64)       # this person's body only
        cells[:, 0] += xs0
        cells[:, 1] += ys0
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
        self._topple(cells.astype(np.float32), axis, s, float(pivot), float(z_lo),
                     owner=p["name"])

    def step(self):
        if self.o2 is None:                      # first step: fresh air fills every pore
            self.o2 = np.full(self.shape, O2_PER_L * self.vox_l, np.float32)
        self._law_bodies()
        self._law_flow()
        self._law_stir()
        self._law_head()
        self._law_footing()                      # before support: a body with no
        self._law_support()                      # footing must leave as a BODY
        self._law_torque()
        self._law_conduct()
        self._law_crack()                        # thermal shock reads what conduction wrote
        self._law_rise()
        self._air_regions()                      # (re)label airspaces if walls/gaps changed
        self._law_burn()
        self._law_boil()
        self._law_melt()
        self._law_acid()
        self._law_smoke()
        self._air_regions()                      # acid may have bored new airspace
        self._law_o2()
        self._law_air()                          # the pneumatic field: pressure, wind
        self._law_life()                         # the slow chemistry of anyone alive
        self._law_will()                         # and their reflexes
        if self.thermostats:                        # dev heater blocks hold their
            C = self.heat_capacity()                # set temperature against all
            for (tx, ty, tz, tT) in self.thermostats:   # losses — heat without fire
                self.E[tx, ty, tz] = (tT - AMBIENT) * C[tx, ty, tz]
        Tk = self.T() + 273.0                       # every voxel radiates to the wider,
        self.E -= RAD * self.scale ** 2 * (Tk ** 4 - 293.0 ** 4) / 3.0   # cooler world — trivial when warm,
        self.E *= (1.0 - LEAK)                      # fierce when white-hot (caps flame temps)
        self.E = np.maximum(self.E, -AMBIENT * self.heat_capacity())   # nothing below 0 °C here
        self.tick += 1

    # ── totals (the conservation the tests watch) ────────────────────────────
    def total_fluid(self, f):
        return float(self.fvol[self.fl == f].sum()) \
            + sum(d[6] for d in self.drops if d[7] == f)   # parcels in flight count

    def total_wood(self):
        return float(self.smass[self.mat == WOOD].sum())
