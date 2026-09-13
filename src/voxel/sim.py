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
import copy

import numpy as np

# OPTIONAL ACCELERATOR. numpy cannot express the shape a cellular-automata
# engine wants — one sweep of the lattice applying many rules per cell — so it
# makes a full pass per arithmetic step instead, and we are memory-bandwidth
# bound: conduction alone builds about seventeen whole-lattice temporaries per
# axis. numba compiles one loop that does all of it per cell.
#
# It is OPTIONAL on purpose. The engine's only hard dependency is PyYAML, and
# the numpy path below stays the definition of what the law MEANS; the compiled
# one has to agree with it cell for cell, which is what
# `test_the_FAST_PATH_and_the_PLAIN_ONE_agree` exists to check. Without numba
# installed, nothing changes except the speed.
try:
    from numba import njit
    HAVE_NUMBA = True
except Exception:                        # pragma: no cover - depends on install
    HAVE_NUMBA = False

    def njit(*args, **kwargs):           # a decorator that does nothing
        def wrap(fn):
            return fn
        return wrap if not args or not callable(args[0]) else args[0]


@njit(cache=True)
def _conduct_pairs(E, T, C, k, fk, wet, solid, s, r, di, dj, dl):
    """One axis of the conduction law, fused: every face between neighbours
    visited ONCE, with nothing kept but the heat that crosses it.

    The arithmetic is written to match the numpy law step for step, including
    the order of the multiplications and of the two clip comparisons, because a
    float32 sum reordered is a float32 sum CHANGED — and the whole value of the
    plain path is that it stays the definition of what the law means.

    The heat is applied in two sweeps rather than one for the same reason: the
    numpy law subtracts from every left-hand cell and only then adds to every
    right-hand one, so a cell sees its own loss before its neighbour's gain."""
    nx, ny, nz = E.shape
    q = np.empty((nx - di, ny - dj, nz - dl), np.float32)
    half = np.float32(0.5)
    k0 = np.float32(273.0)
    lo_f = np.float32(-0.2)
    hi_f = np.float32(0.2)
    for i in range(nx - di):
        for j in range(ny - dj):
            for l in range(nz - dl):
                i2 = i + di
                j2 = j + dj
                l2 = l + dl
                ka = k[i, j, l]
                kb = k[i2, j2, l2]
                kp = ka if ka < kb else kb
                c = np.float32(0.0)
                if wet[i, j, l] and solid[i2, j2, l2]:
                    c += fk[i, j, l]
                if wet[i2, j2, l2] and solid[i, j, l]:
                    c += fk[i2, j2, l2]
                if c > kp:
                    kp = c
                ta = T[i, j, l]
                tb = T[i2, j2, l2]
                tk = (ta + tb) * half + k0
                kr = r * tk * tk * tk
                dt = ta - tb
                v = (kp * s + kr * s * s) * dt
                ad = dt if dt >= np.float32(0.0) else -dt
                ca = C[i, j, l]
                cb = C[i2, j2, l2]
                cm = ca if ca < cb else cb
                lo = lo_f * cb * ad
                hi = hi_f * cm * ad
                if v < lo:
                    v = lo
                if v > hi:
                    v = hi
                q[i, j, l] = v
    for i in range(nx - di):
        for j in range(ny - dj):
            for l in range(nz - dl):
                E[i, j, l] -= q[i, j, l]
    for i in range(nx - di):
        for j in range(ny - dj):
            for l in range(nz - dl):
                E[i + di, j + dj, l + dl] += q[i, j, l]


@njit(cache=True)
def _o2_diffuse(o2, T, por, di, dj, dl):
    """One axis of oxygen mixing, fused. Same shape as conduction and the same
    discipline: numpy's order of operations, kept step for step, and the loss
    swept over every cell before the gain is."""
    nx, ny, nz = o2.shape
    q = np.empty((nx - di, ny - dj, nz - dl), np.float32)
    base = np.float32(0.1)
    div = np.float32(1600.0)
    top = np.float32(0.45)
    for i in range(nx - di):
        for j in range(ny - dj):
            for l in range(nz - dl):
                i2 = i + di
                j2 = j + dj
                l2 = l + dl
                k = base + (T[i, j, l] + T[i2, j2, l2]) / div
                if k < base:
                    k = base
                if k > top:
                    k = top
                pa = por[i, j, l]
                pb = por[i2, j2, l2]
                k = k * (pa if pa < pb else pb)
                q[i, j, l] = k * (o2[i, j, l] - o2[i2, j2, l2])
    for i in range(nx - di):
        for j in range(ny - dj):
            for l in range(nz - dl):
                o2[i, j, l] -= q[i, j, l]
    for i in range(nx - di):
        for j in range(ny - dj):
            for l in range(nz - dl):
                o2[i + di, j + dj, l + dl] += q[i, j, l]


@njit(cache=True)
def _gas_overturn(o2, smoke, T, air):
    """Warm gas sitting UNDER cool gas trades places with it, carrying whatever
    it is made of. One parcel, so its oxygen and its soot move together and by
    exactly the same fraction — which is why both gases share one kernel and one
    buffer, and why oxygen goes first here as it does in the plain law."""
    nx, ny, nz = o2.shape
    f = np.empty((nx, ny, nz - 1), np.float32)
    rate = np.float32(0.02)
    cap = np.float32(0.45)
    zero = np.float32(0.0)
    for i in range(nx):
        for j in range(ny):
            for l in range(nz - 1):
                if air[i, j, l] and air[i, j, l + 1]:
                    v = (T[i, j, l] - T[i, j, l + 1]) * rate
                    if v < zero:
                        v = zero
                    if v > cap:
                        v = cap
                    f[i, j, l] = v
                else:
                    f[i, j, l] = zero
    for gas in (o2, smoke):
        q = np.empty((nx, ny, nz - 1), np.float32)
        for i in range(nx):
            for j in range(ny):
                for l in range(nz - 1):
                    q[i, j, l] = f[i, j, l] * (gas[i, j, l] - gas[i, j, l + 1])
        for i in range(nx):
            for j in range(ny):
                for l in range(nz - 1):
                    gas[i, j, l] -= q[i, j, l]
        for i in range(nx):
            for j in range(ny):
                for l in range(nz - 1):
                    gas[i, j, l + 1] += q[i, j, l]

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
        "forget": 0.985,         # what a memory of danger is worth next look.
                                 # A body that never forgets treats a fire it
                                 # saw an hour ago as a fire, and one that
                                 # forgets at once has no memory at all; this
                                 # is about a minute to half-weight at the
                                 # scan rate. Forgetting is a MODEL, not a
                                 # leak — see item 14
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
        "look_for": 25,          # ticks a body holds a chosen direction before
                                 # its head goes back to sweeping. About one
                                 # step of the sweep: long enough to have LOOKED
                                 # rather than glanced, short enough that a
                                 # decision to look is not a decision to stare.
        # HOW OFTEN A BODY DECIDES: a human REACTION TIME, not a number picked
        # to feel right. A simple visual reaction is about a quarter of a
        # second — see something, choose, begin to move — so that is what this
        # is, and it is DERIVED from `TICK_S` so a finer or coarser tick does
        # not quietly make everyone quicker or slower on the draw.
        #
        # It was 30 ticks (0.75 s), which is nearly three reaction times, and
        # that had consequences nobody had connected to it: a fall from a ledge
        # takes about 25 ticks, so a man who was not already holding the rail
        # got exactly ONE decision in the whole of his friend's fall and had to
        # spend it on the rail or on the catch. At a quarter second he gets
        # two, which is the difference between a rescue being possible and not.
        "react_s": 0.25,
        "decide_every": 10}      # filled in from `react_s` below, once TICK_S
                                 # exists. Ten is what 0.25 s comes to at the
                                 # tick this sim has always run at.
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
LIMBS = ("legs", "hands", "mouth", "waist", "eyes")
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
        "throw":  {"limb": "hands", "null": False},
        "catch":  {"limb": "hands", "null": False},
        "catch_who": {"limb": "hands", "null": False},
        "take":   {"limb": "hands", "null": False},
        "reach":  {"limb": "hands", "null": False},
        "pull_in": {"limb": "hands", "null": False},
        "say":    {"limb": "mouth", "null": False},
        "quiet":  {"limb": "mouth", "null": True},
        # A WAIST IS A PART OF THE BODY, so it gets a menu like every other
        # part. Leaning is not walking and it is not reaching: the feet stay,
        # the hands do whatever they were doing, and what moves is the trunk.
        # EYES ARE A PART OF THE BODY TOO, and looking is something a mind can
        # DECIDE to do. The sweep stays as the null act — a body that has
        # decided nothing is still turning its head — but "look behind you" is
        # now a choice a body makes and a row a harvest can learn from.
        "look":   {"limb": "eyes", "null": False},
        "about":  {"limb": "eyes", "null": True},
        "lean":   {"limb": "waist", "null": False},
        "upright": {"limb": "waist", "null": False},
        "hold_pose": {"limb": "waist", "null": True}}
# REFLEXES — percept -> what each part of the body does about it. This is the
# part a character sheet edits, and it is now a sheet of a better shape: a
# brave character's "sees_fire" row can send the legs at the fire while the
# mouth still calls the warning, which the old one-response-per-percept table
# could not say at all. A person dict may carry its own "reflexes" override.
# The table is not consulted by the law: it is one POLICY among several,
# reading the same menus everything else reads.
REFLEXES = {"sees_fire":   {"legs": "go:exit", "mouth": "say:fire"},
            "scorched":    {"legs": "go:exit", "mouth": "say:fire"},
            "hears_alarm": {"legs": "go:exit", "mouth": "say:coming"},
            "sees_runner": {"legs": "go:exit"},
            "chokes":      {"legs": "go:exit"},
            # nothing for the legs: a thing coming at you is a matter for the
            # hands, and a character sheet that wants it dodged says so
            "sees_thrown":  {"hands": "catch"},
            # and a falling PERSON is a different thing to see
            "sees_falling": {"hands": "catch"}}
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
            hit = [i for i, opt in enumerate(menu) if opt["tag"] == tag]
            if not hit:
                continue
            # TASTE, of the plainest kind there is. The table says WHAT to do;
            # among the ways of doing it, prefer the one that does not take you
            # past a fire you remember seeing, and then the nearest.
            #
            # This is the whole of why it matters: without it the pick was the
            # FIRST matching row, so which way an idle body wandered was decided
            # by the order `_places` happened to build its list in — a modulo of
            # the tick. Any change to the set of reachable places was a coin
            # flip over where a body went, and one of those flips walked a man
            # into a fire and killed him. A policy that prefers nothing is not
            # neutral; it is arbitrary, and arbitrary is not a thing a body does.
            return min(hit, key=lambda i: (menu[i].get("danger", 0.0),
                                           menu[i].get("away", 0.0)))
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
        "legs_N": 1400.0, "crouch_m": 0.25,
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
# A BODY DECIDES AT A REACTION TIME, and the tick length is what says how
# many ticks that is. Derived here rather than typed into `WILL`, because
# `TICK_S` is defined below it and because a finer or coarser tick must not
# quietly make everybody quicker or slower on the draw.
WILL["decide_every"] = max(1, int(round(WILL["react_s"] / TICK_S)))
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
_SWEEP_RAD = 0.08              # how finely a limb's turn is swept for
                               # things in the way. One voxel at the end
                               # of a 9-voxel arm is about 0.11 rad, so
                               # this cannot step over a wall.
_SIX = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))

SMOKE_STILL = 1e-6
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
        # WHAT HAS LEFT THE WORLD ALTOGETHER, in grams by material. A lattice
        # has edges; a thing thrown past one is outside, and the books only
        # balance if the world can say how much went. See `_left_world`.
        self.gone = {}
        # AND WHAT IT HAS SHED AS HEAT, in joules — radiated into the colder
        # universe outside the lattice, plus `LEAK`, less whatever the 0 °C
        # floor had to put back. See the end of `step`.
        self.shed = 0.0
        self.pcell = None                               # [x,y,z, vx,vy,vz, ml, fluid]
        self.vcell = None                               # coarse gas pressure + velocity
        self.p_add = np.zeros(self.shape, np.float32)   # pressure injected this tick
        self.tick = 0
        # DO NOT COMPUTE WHERE NOTHING IS HAPPENING. Most of a world is inert
        # most of the time, and a quiet room measured EXACTLY as expensive as a
        # burning one (98 ms/tick against 100). Every skip below has to be
        # provable, not merely plausible: a law is skipped only when the thing
        # it moves is not present at all, or cannot have changed since it last
        # ran. Set False to run every law over the whole world every tick —
        # which is what `test_ACTIVE_REGIONS_change_NOTHING` compares against.
        self.skip_quiet = True
        # Measure what the attention cap costs (see _decide). Off: it doubles
        # the work of deciding, and answers a question rather than doing one.
        self.recall_check = False
        # Use the compiled laws where there are any. Turned off, the world runs
        # on the numpy ones, which are what the laws MEAN — the compiled path
        # has to agree with them cell for cell, and a test says so.
        self.fused = HAVE_NUMBA
        self._cap_key = None                            # derived fields, cached on
        self._cap = self._por = None                    # what matter is where
        self._aircells = self._airblur = None           # and where the air is

    # ── derived fields ───────────────────────────────────────────────────────
    def heat_capacity(self):
        """What it costs to warm a voxel by a degree: its solid, its fluid, and
        the air in the space left over.

        CACHED against the four fields it is made of. It was being rebuilt
        TWELVE TIMES A TICK — a fifth of all sim time, burning or not — for a
        quantity that only changes when matter does: something burns away,
        melts, is poured, or walks. Comparing the four costs about a fifth of
        one rebuild, and it is right by construction because it compares the
        real state, rather than by everyone who writes a voxel remembering to
        invalidate it. Same bargain as the slack field, and the same reason.

        Callers only ever read it — checked — so one array is handed to all."""
        self._matter_moved()
        if self._cap is None:
            c_solid = _CSOLID_ARR[self.mat]
            fmass = self.fvol * _FDENS_ARR[self.fl]
            c_fluid = _FC_ARR[self.fl]
            self._cap = self.smass * c_solid + fmass * c_fluid + self.c_airbase
        return self._cap

    def _matter_moved(self):
        """Has any matter changed since the derived fields were last built? If
        it has, throw them away and remember the new state.

        One comparison serves both heat capacity and porosity, because both are
        made of the same four fields — so the price of checking is paid once
        however many derived fields come to depend on it."""
        k = self._cap_key
        if (k is not None and np.array_equal(k[0], self.mat)
                and np.array_equal(k[1], self.smass)
                and np.array_equal(k[2], self.fl)
                and np.array_equal(k[3], self.fvol)):
            return False
        self._cap = self._por = None
        self._cap_key = (self.mat.copy(), self.smass.copy(),
                         self.fl.copy(), self.fvol.copy())
        return True

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
        self._matter_moved()                      # cached beside heat capacity:
        if self._por is None:                     # same four fields, one check
            packed = self.smass / np.maximum(_DENS_ARR[self.mat] * self.vox_l, 1e-9)
            void = np.clip(1.0 - packed, 0.0, 1.0)
            drowned = self.fvol > 0.5 * self.cap  # a flooded gap does not breathe
            self._por = np.where(drowned, np.minimum(void, 0.02), void)
        return self._por

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

    def _limbs(self, p):
        """The body's named LIMBS — an arm counts ONCE, by the name of the
        whole arm, not once per bone. Everything outside this file asks for
        "right arm"; whether that is one bone or three is the body's own
        business, declared by whoever built it."""
        chain = p.get("chain") or {}
        bones = {b for bs in chain.values() for b in bs}
        out = list(chain)
        out += [k for k in (p.get("segs") or {}) if k not in bones
                and k not in chain]
        return out

    def _bones(self, p, name):
        """The bones of a limb, SHOULDER FIRST. A limb with no chain declared
        is one bone with its own name, which is why nothing had to change when
        arms grew elbows."""
        chain = (p.get("chain") or {}).get(name)
        return list(chain) if chain else [name]

    def _parents(self, p):
        """Which bone each bone hangs from.

        A limb's CHAIN already says most of this — the forearm hangs from the
        upper arm, that is what writing them in that order means — so it is
        read off there rather than declared a second time, which would be a
        second place to be wrong. `p["parent"]` says the rest: what the arms
        and the head hang from, which no chain covers because they are not in
        one another's chains."""
        par = dict(p.get("parent") or {})
        for bs in (p.get("chain") or {}).values():
            for a, b in zip(bs, bs[1:]):
                par.setdefault(b, a)
        return par

    def _kids(self, p):
        """Bone -> the bones hanging off it."""
        kids = {}
        for b, par in self._parents(p).items():
            kids.setdefault(par, []).append(b)
        return kids

    def _subtree(self, p, root):
        """`root` and everything hanging off it, nearest the body first.

        TURNING A JOINT CARRIES EVERYTHING BEYOND IT. That is what a joint is,
        and it is the whole reason a body needs a tree and not a list: leaning
        at the hips takes the head and both arms with it, and leaves the legs
        exactly where they are standing. A body with no parents declared is all
        roots, so this returns the one bone and nothing changes."""
        kids = self._kids(p)
        out, q = [], [root]
        while q:
            b = q.pop(0)
            if b in out:
                continue
            out.append(b)
            q.extend(sorted(kids.get(b, [])))   # sorted: the sim carries no die
        return out

    def _path_from(self, p, root, bone):
        """The bones from `root` out to `bone`, inclusive — the joints whose
        turns compose to say where that bone ends up."""
        par = self._parents(p)
        path, b = [], bone
        while True:
            path.append(b)
            if b == root:
                return path[::-1]
            b = par.get(b)
            if b is None or len(path) > 32:
                return None

    def _pivot_of(self, p, name):
        """The joint a whole limb turns about: for a chain, the one nearest the
        body. What a shoulder torque is measured at."""
        joints = p.get("joints") or {}
        for b in self._bones(p, name):
            if b in joints:
                return np.asarray(joints[b])
        return None

    @staticmethod
    def _joint_xform(pivot, zb, theta, sgn, bend=False):
        """One joint's move, as a 2x3 affine on (sideways, up).

        A HINGE TURNS. Exactly the rotation `_body_pose` has always done,
        written as a matrix so that joints can be COMPOSED — an elbow is the
        shoulder's turn applied to the forearm's turn, and matrices are how you
        say that without special cases.

        A SPINE DOES NOT TURN, IT BENDS, and on a lattice that difference is
        the difference between possible and impossible. A torso is a solid
        slab, and a rigid rotation of a solid slab is never injective on a
        grid: at EVERY angle some pair of its voxels rounds into one cell, so
        every lean was refused as undrawable and a body could not bend at all.

        A lean is not one bone swinging. A trunk is a stack of vertebrae and
        bending it is each slice sliding forward over the one below — a SHEAR.
        Which is exactly right for the thing it models, and has the property
        the lattice needs for free: every row moves by one constant, so within
        a row the map is a translation of integers, and rows never meet because
        their height does not change. Nothing can round into anything. The
        first-order shortening of a real bend (cos theta) is what it gives up,
        and that is second order in the angle."""
        if bend:
            # the sign is flipped against a hinge's, and for a reason rather
            # than to taste: `sgn` is set so that a POSITIVE angle carries a
            # bone's far end forward, and an arm's far end hangs BELOW its
            # shoulder while a torso's rises ABOVE its hips. Same convention,
            # opposite geometry.
            t = -float(np.tan(theta))
            return np.array([[1.0, sgn * t, -sgn * t * zb],
                             [0.0, 1.0, 0.0]])
        c, sn = float(np.cos(theta)), float(np.sin(theta))
        return np.array([[c, sgn * sn, pivot - c * pivot - sgn * sn * zb],
                         [-sgn * sn, c, zb + sgn * sn * pivot - c * zb]])

    def _chain_xform(self, pivots, thetas, upto, sgn, bends=None):
        """Where a bone's REST cells end up: its OWN joint turns first, then
        every joint above it carries the result along.

        This is forward kinematics and it is three lines, because all of a limb
        turns in one plane. Written about the REST pivots on purpose — the
        elbow MOVES when the shoulder turns, and composing in the rest frame is
        what saves having to track where it moved to."""
        M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        for j in range(upto, -1, -1):
            R = self._joint_xform(pivots[j][0], pivots[j][1], thetas[j], sgn,
                                  bool(bends[j]) if bends else False)
            M = R @ np.vstack([M, [0.0, 0.0, 1.0]])
        return M

    @staticmethod
    def _apply_xform(M, cells, axis):
        """Move a set of voxels by a 2x3 affine in the (axis, z) plane."""
        out = np.asarray(cells, np.float64).copy()
        lat, zz = out[:, axis].copy(), out[:, 2].copy()
        out[:, axis] = M[0, 0] * lat + M[0, 1] * zz + M[0, 2]
        out[:, 2] = M[1, 0] * lat + M[1, 1] * zz + M[1, 2]
        return out

    def _limb_parts(self, p, name, own=None):
        """One array of live cells PER BONE, shoulder first.

        Anything that has to know which bone a voxel belongs to asks here — a
        swung arm that comes to rest has to be able to say where its elbow went
        and not merely where its arm did."""
        segs = p.get("segs")
        bones = [b for b in self._bones(p, name) if b in (segs or {})]
        if not segs or not bones:
            return None
        return self._bone_parts(p, bones, own)

    def _bone_parts(self, p, bones, own=None):
        """The live cells of each NAMED bone, in the order asked for."""
        segs = p.get("segs") or {}
        if own is None:
            comp, sl = self._person_cells(p)
            if comp is None or not comp.any():
                return None
            own = np.argwhere(comp)
            own[:, 0] += sl[0].start or 0
            own[:, 1] += sl[1].start or 0
        base = np.asarray(own).min(axis=0)
        nx, ny, nz = self.shape
        out = []
        for b in bones:
            if b not in segs:
                return None
            c = (np.asarray(segs[b]) + base).astype(np.int64)
            ok = ((c[:, 0] >= 0) & (c[:, 0] < nx) & (c[:, 1] >= 0)
                  & (c[:, 1] < ny) & (c[:, 2] >= 0) & (c[:, 2] < nz))
            c = c[ok]
            if len(c):
                c = c[self.mat[tuple(c.T)] == FLESH]
            out.append(c)
        return out

    def _bone_angle(self, p, bone):
        """One bone's OWN joint angle, whichever limb happens to own it. A bone
        nobody poses — a head — is at rest, and rides on whatever is below it."""
        for limb, bs in (p.get("chain") or {}).items():
            if bone in bs:
                return self._angles((p.get("pose") or {}).get(limb, 0.0),
                                    len(bs))[bs.index(bone)]
        return float(np.atleast_1d(
            (p.get("pose") or {}).get(bone, 0.0))[0])

    def _limb_cells(self, p, name, own=None):
        """Where one named part of this body actually is, right now.

        Segments are kept as offsets from the body's own corner, so they ride
        along when it walks, is shoved, or is carried out. A limb that has been
        swung and come to rest somewhere new updates its own offsets on landing.
        A limb made of several bones answers with all of them.
        Returns None when there is no such limb, or nothing left of it."""
        parts = self._limb_parts(p, name, own)
        if not parts:
            return None
        limb = np.concatenate(parts)
        return limb if len(limb) else None

    def _swing(self, p, name, toward=None, back=0.0):
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
        bones = [b for b in self._bones(p, name) if b in (segs or {})]
        if not segs or not bones:
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
        parts = self._limb_parts(p, name, own)
        if not parts or not sum(len(c) for c in parts):
            return None                       # the limb is not there any more
        limb = np.concatenate(parts)
        # WHICH BONE EACH VOXEL IS. A whole arm swings as ONE rigid thing —
        # the elbow is locked by the same muscles that drive the shoulder —
        # but it has to come to rest as an arm with an elbow in it, or the
        # next pose reaches for bones that are no longer where they are said
        # to be. -1 is a carried tool: along for the ride, part of nobody.
        segof = np.concatenate([np.full(len(c), i, np.int32)
                                for i, c in enumerate(parts)])
        # A HELD THING SWINGS WITH THE ARM — that is most of what holding a
        # tool is for. Its cells join the limb's rigid body, so its mass slows
        # the swing (Hill does the rest) and its edge is what arrives.
        obj = self._held_cells(p)
        if obj is not None and self._in_reach(limb, obj):
            limb = np.concatenate([limb, obj])
            segof = np.concatenate([segof, np.full(len(obj), -1, np.int32)])
        piv = self._pivot_of(p, name)
        if piv is None:
            return None
        jx, jy, jz = (piv + own.min(axis=0)).astype(np.int64)
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
        b["segof"], b["bones"] = segof, bones
        # AND IT MAY START BEHIND THE BODY. `back` is how far, and it is
        # honoured only if the arm can actually BE there — a man in a doorway
        # with a wall at his shoulder throws from where he stands, and throws
        # worse, which is right.
        if back:
            nx, ny, nz = self.shape
            mine = {tuple(c) for c in limb}
            for tryback in (back, back * 0.5):
                at = np.round(self._body_pose(dict(b, cells=limb.astype(
                    np.float64)), -tryback)).astype(np.int64)
                if ((at < 0).any() or (at[:, 0] >= nx).any()
                        or (at[:, 1] >= ny).any() or (at[:, 2] >= nz).any()):
                    continue
                if any(tuple(t) not in mine and int(self.mat[tuple(t)]) != AIR
                       for t in at):
                    continue
                b["theta"] = -float(tryback)
                break
        b["phi0"], b["L"] = 0.02, 2.0
        return b

    def _still_joined(self, p, want, own, mine):
        """Is the limb, DRAWN AT THIS ANGLE, still one piece of one body?

        The other half of "undrawable", and the half that was missing. A line
        of voxels turned to anything but a right angle rounds to a STAIRCASE,
        and a staircase is not joined — its steps touch at corners only. So the
        check that two voxels must not round into one has a twin: they must not
        round APART either. Measured before this: an arm bent 1.2 rad came out
        in two pieces, still all its own mass, no longer all one arm.

        The honest answer is the same one the lattice has always given — the
        angle is not drawable, the flesh holds its last good pose, and the arm
        catches up at the next angle that IS. It is not a smoothing hack; at
        5 cm a one-voxel limb genuinely has only a handful of poses."""
        rest = {tuple(c) for c in own} - mine    # the body it hangs from
        S = {tuple(c) for c in want}
        seen = {c for c in S
                if any((c[0] + dx, c[1] + dy, c[2] + dz) in rest
                       for dx, dy, dz in _SIX)}
        if not seen:
            return False                      # it is not attached at all
        stack = list(seen)
        while stack:
            c = stack.pop()
            for dx, dy, dz in _SIX:
                n = (c[0] + dx, c[1] + dy, c[2] + dz)
                if n in S and n not in seen:
                    seen.add(n)
                    stack.append(n)
        return len(seen) == len(S)

    @staticmethod
    def _angles(theta, n):
        """One angle per bone. A single number turns the joint NEAREST THE BODY
        and leaves the rest straight — which is exactly what every caller meant
        back when an arm was one bone, so nothing had to be rewritten when it
        grew an elbow."""
        th = [float(theta)] if np.isscalar(theta) else [float(t) for t in theta]
        return (th + [0.0] * n)[:n]

    @staticmethod
    def _inv_xform(M):
        """The way back. The 2x2 part is a rotation, so its inverse is its
        transpose and there is nothing to solve."""
        A = M[:, :2].T
        return np.hstack([A, -A @ M[:, 2:3]])

    @staticmethod
    def _comp_xform(a, b):
        """Do b, then a."""
        return a @ np.vstack([b, [0.0, 0.0, 1.0]])

    def _turn(self, p, quarters):
        """Turn a body on its own feet, in QUARTER turns.

        A quarter turn is the only rotation a lattice can do EXACTLY. It is a
        permutation of the cells — (dx, dy) becomes (-dy, dx) and back again —
        so nothing rounds together and nothing rounds apart, and the two checks
        `_repose` needs for every other angle are not needed for this one at
        all. Everything else in this file that turns has to argue with the
        grid; this does not.

        Which is why a body could not turn until now, and why it matters that
        it can. The humanoid is built with its shoulders along x and its
        `facing` rotates freely, so it FACES ALONG ITS OWN SHOULDER LINE half
        the time — and an arm turns in the plane the body faces, so half the
        time that plane holds the torso too. That is one cause behind a
        wind-up that sweeps an arm through its own chest, a thicker arm that
        collides with its own body when it reaches, and a pose space that
        depends on which way a body happens to be pointed.

        It can be REFUSED, and that is a feature: a body in a space too tight
        to turn in does not turn. Corridors are like that."""
        comp, sl = self._person_cells(p)
        if comp is None or not comp.any():
            return False
        own = np.argwhere(comp)
        own[:, 0] += sl[0].start or 0
        own[:, 1] += sl[1].start or 0
        q = int(quarters) % 4
        if q == 0:
            return True
        cx = int(round(float(own[:, 0].mean())))
        cy = int(round(float(own[:, 1].mean())))

        def spin(cells):
            c = np.asarray(cells, np.int64).copy()
            dx, dy = c[:, 0] - cx, c[:, 1] - cy
            for _ in range(q):
                dx, dy = -dy, dx
            c[:, 0], c[:, 1] = cx + dx, cy + dy
            return c

        want = spin(own)
        nx, ny, nz = self.shape
        if ((want < 0).any() or (want[:, 0] >= nx).any()
                or (want[:, 1] >= ny).any() or (want[:, 2] >= nz).any()):
            return False
        mine = {tuple(c) for c in own}
        if any(tuple(t) not in mine and int(self.mat[tuple(t)]) != AIR
               for t in want):
            return False                      # no room to turn round in
        fields = (self.mat, self.smass, self.E, self.fl, self.fvol,
                  self.fpot, self.fallh, self.edge)
        src, dst = tuple(own.T), tuple(want.T)
        held = [arr[src].copy() for arr in fields]
        for arr in fields:
            arr[src] = 0
        for arr, h in zip(fields, held):
            arr[dst] = h
        # AND EVERY PART OF IT TURNS WITH IT. Segments, joints and rest shapes
        # are offsets from the body's own corner, and that corner has moved —
        # so each is rotated about the same centre and re-based, or the body
        # keeps a map of a shape it no longer has.
        base = want.min(axis=0)
        for book in ("segs", "rest", "joints"):
            d = p.get(book)
            if not d:
                continue
            origin = own.min(axis=0)
            for k, v in list(d.items()):
                v = np.asarray(v)
                flat = v.ndim == 1
                a = spin((v.reshape(1, 3) if flat else v) + origin) - base
                d[k] = a[0] if flat else a
        fx, fy = p.get("facing", (1.0, 0.0))
        for _ in range(q):
            fx, fy = -fy, fx
        p["facing"] = (float(fx), float(fy))
        p["_claim_tick"] = None               # it is a different shape now
        self._torque_solid = None
        self._slack_mat = None
        return True

    def _repose(self, p, name, theta, toward=None):
        """Put one limb where its ANGLES say it is, on the lattice, now.

        This is the difference between a pose and a picture. The limb is not
        lifted off the world, turned, and set back down — its voxels are moved
        from the cells they are in to the cells the angles ask for, so while the
        arm is out it really IS out: it blocks what it now occupies, its mass
        sits where it now sits, a hand at the end of it is somewhere new, and
        anything already in the way STOPS it.

        A LIMB IS A CHAIN OF BONES, each turning about its own joint, and the
        bones nearer the body carry the ones further out along with them. One
        angle per joint; a bare number means the joint nearest the body and the
        rest held straight. An arm with one bone is the same code with a chain
        of length one, which is why nothing above this had to learn about
        elbows.

        Answers WHY it could not, because the two reasons are not alike.
        "blocked" is the world saying no — something solid is there, and an arm
        that meets a post stops at the post. "undrawable" is the LATTICE saying
        no: a limb one voxel wide has only a handful of angles it can be drawn
        at, because at the others two of its voxels round into the same cell and
        a voxel of flesh would be destroyed by the act of moving. A limb is
        never allowed to cost the body mass, so those angles are skipped over —
        the arm holds its last drawable pose while the angle goes on, and
        catches up when the two agree again. At 5 cm that is not a workaround,
        it is what a thin thing turning on a coarse grid IS.

        Turned from the limb's REST shape rather than from wherever it happens
        to be, so angles do not accumulate the rounding of every angle before
        them — and, for a chain, so that the elbow can be turned about where it
        RESTS rather than about wherever the shoulder has just carried it."""
        segs, joints = p.get("segs"), p.get("joints")
        bones = [b for b in self._bones(p, name) if b in (segs or {})]
        if not segs or not bones or any(b not in (joints or {}) for b in bones):
            return "blocked"
        # EVERYTHING BEYOND THE JOINT COMES TOO. The limb's own bones are the
        # ones with angles; the moving SET is those plus whatever hangs off
        # them, which for an arm is nothing and for a torso is a head and two
        # arms. A body with no parents declared has a moving set of exactly its
        # own bones, which is why nothing above this changed.
        moving = [b for b in self._subtree(p, bones[0]) if b in segs]
        for b in bones:
            if b not in moving:
                moving.append(b)
        paths = {b: self._path_from(p, bones[0], b) or [b] for b in moving}
        comp, sl = self._person_cells(p)
        if comp is None or not comp.any():
            return "blocked"
        own = np.argwhere(comp)
        own[:, 0] += sl[0].start or 0
        own[:, 1] += sl[1].start or 0
        origin = own.min(axis=0)
        rest = p.setdefault("rest", {})
        for b in moving:
            if b not in rest:
                rest[b] = np.asarray(segs[b]).copy()
        cur_parts = self._bone_parts(p, moving, own)
        if cur_parts is None or any(len(c) != len(rest[b])
                                    for c, b in zip(cur_parts, moving)):
            return "blocked"                  # a bone has lost voxels; its rest
                                              # shape no longer describes it
        cur = np.concatenate(cur_parts)
        dx, dy = toward if toward else p.get("facing", (1.0, 0.0))
        axis = 0 if abs(dx) >= abs(dy) else 1
        sgn = -1 if (dx if axis == 0 else dy) >= 0 else 1
        th = self._angles(theta, len(bones))
        bend = set(p.get("bend") or ())    # which joints bend rather than turn
        pivot = {}
        for b in moving:
            j = (np.asarray(joints.get(b, joints[bones[0]]))
                 + origin).astype(np.float64)
            pivot[b] = (float(j[axis]), float(j[2]))

        def place(ang):
            """Every moving bone, at these angles: its own joint first, then
            every joint between it and the limb being turned."""
            held = dict(zip(bones, ang))
            out = []
            for b in moving:
                path = paths[b]
                out.append(self._apply_xform(
                    self._chain_xform([pivot[x] for x in path],
                                      [held.get(x, self._bone_angle(p, x))
                                       for x in path],
                                      len(path) - 1, sgn,
                                      [x in bend for x in path]),
                    (np.asarray(rest[b]) + origin).astype(np.float64), axis))
            return out

        parts = place(th)
        want = np.round(np.concatenate(parts)).astype(np.int64)
        # WHAT THE HAND HOLDS TURNS WITH THE HAND. Not an extra rule — it rides
        # rigidly with the bone at the END of the chain, carried from where the
        # old angles put that bone to where the new ones do. Without it an arm
        # could not reach out while holding anything, because the thing it held
        # was standing in the way of its own arm.
        ride = self._held_cells(p)
        if ride is not None and self._in_reach(cur, ride):
            # A HELD THING HANGS FROM THE FIST — it does not ride CLAMPED to
            # it. So it follows where the hand goes and keeps its own attitude:
            # gravity has a say in which way up a carried thing is, and the
            # wrist is a joint whether or not this sim models one yet. Turned
            # rigidly about the shoulder instead, a 38 kg block hanging below a
            # man's hand was swung up over his head by the act of putting his
            # arm out, which is not what carrying is.
            #
            # By WHERE THE HAND IS, not by what the angle reads: the two part
            # company whenever the lattice cannot draw an angle, because the
            # flesh holds its last good pose while the angle runs on and then
            # catches up several angles at once. Measured with the delta in
            # angles: the arm went out eight voxels and the block moved one, so
            # the lever never formed and the man kept his feet.
            #
            # The bone's voxels keep their order from `rest`, so cell 0 of the
            # far bone is the SAME piece of flesh before and after — which is
            # what makes this an honest displacement and not a guess.
            tip = moving.index(bones[-1])     # the limb's OWN far bone, which
            shift = (np.round(parts[tip][0]).astype(np.int64)   # is not the last
                     - cur_parts[tip][0])                       # of the subtree
                                              # once a head and arms come along
            ride_to = ride + shift
            cur = np.concatenate([cur, ride])
            want = np.concatenate([want, ride_to])
            moved_hold = (tuple(ride_to[0]), tuple(ride[0]))
        else:
            moved_hold = None
        nx, ny, nz = self.shape
        if ((want < 0).any() or (want[:, 0] >= nx).any()
                or (want[:, 1] >= ny).any() or (want[:, 2] >= nz).any()):
            return "blocked"
        flat = (want[:, 0] * ny + want[:, 1]) * nz + want[:, 2]
        if len(np.unique(flat)) != len(flat):
            return "undrawable"               # two voxels into one: mass lost
        mine = {tuple(c) for c in cur}
        if not self._still_joined(p, want[:len(want) - (len(ride) if
                                  moved_hold is not None else 0)], own, mine):
            return "undrawable"               # ...and two voxels into none
        blocked = [tuple(t) for t in want
                   if tuple(t) not in mine and int(self.mat[tuple(t)]) != AIR]
        if blocked:
            return "blocked"                  # something is in the way, and the
                                              # way is the world, not a picture
        # AND NOTHING WAS IN THE WAY ON THE ROAD THERE. Only the pose at the
        # END of a tick used to be checked, which is honest while a limb turns
        # a little at a time — and stops being honest the moment it does not.
        # The flesh waits at every angle the lattice cannot draw and then
        # catches up several angles at once, so a rail that happens to sit in
        # the undrawable part of the sweep was never touched by anything: the
        # arm was on one side of it, and then it was on the other.
        #
        # So the angles in between are swept. They do not have to be DRAWABLE
        # — two voxels rounding together is an artifact of drawing, not an
        # event in the world — they only have to be unoccupied. This is the
        # same rule a walking body already keeps: you may not arrive somewhere
        # by passing through something.
        was_th = self._angles((p.get("drawn") or {}).get(name, 0.0), len(bones))
        gap = max(abs(a - b) for a, b in zip(th, was_th)) if bones else 0.0
        if gap > _SWEEP_RAD:
            for f in np.arange(_SWEEP_RAD, gap, _SWEEP_RAD) / gap:
                mid = [w + (t - w) * float(f) for w, t in zip(was_th, th)]
                for pt_f in place(mid):
                    pt = np.round(pt_f).astype(np.int64)
                    if ((pt < 0).any() or (pt[:, 0] >= nx).any()
                            or (pt[:, 1] >= ny).any() or (pt[:, 2] >= nz).any()):
                        return "blocked"
                    hit = [tuple(t) for t in pt if tuple(t) not in mine
                           and int(self.mat[tuple(t)]) != AIR]
                    if hit:
                        return "blocked"      # it would have had to go through
        fields = (self.mat, self.smass, self.E, self.fl, self.fvol,
                  self.fpot, self.fallh)
        src, dst = tuple(cur.T), tuple(want.T)
        held = [arr[src].copy() for arr in fields]
        for arr in fields:
            arr[src] = 0
        for arr, h in zip(fields, held):
            arr[dst] = h
        # THE BODY'S CORNER CAN MOVE WHEN ONLY PART OF THE BODY DOES. Every
        # segment is kept as an offset from the min corner of the whole body,
        # and that corner is read fresh each time — so a lean, which carries a
        # torso and both arms forward and leaves the legs, can change which
        # voxel is the corner, and then EVERY offset on the body is out by one
        # and the arms are no longer where the body says they are. An arm
        # reaching never showed it: an arm goes forward, and the corner is
        # behind. Rebase, and nothing has to know.
        n_flesh = len(want) - (len(ride) if moved_hold is not None else 0)
        stay = np.array([c for c in own if tuple(c) not in mine], np.int64)
        allf = np.concatenate([stay.reshape(-1, 3), want[:n_flesh]])
        base = allf.min(axis=0)
        if not np.array_equal(base, origin):
            shift = origin - base
            for b in segs:
                segs[b] = np.asarray(segs[b]) + shift
            for b in rest:
                rest[b] = np.asarray(rest[b]) + shift
            for b in (joints or {}):
                joints[b] = np.asarray(joints[b]) + shift
        k = 0
        for b in moving:                       # each bone keeps its own offsets
            n = len(rest[b])
            segs[b] = want[k:k + n] - base
            k += n
        p.setdefault("drawn", {})[name] = list(th)    # where the flesh IS
        if max((abs(t) for t in th), default=0.0) > 1e-9:
            p.pop("_no_reach", None)          # it got somewhere: the question is
                                              # open again. Coming back to REST
                                              # is not getting somewhere — the
                                              # arm unwinding after a refused
                                              # reach ends in a successful move
                                              # to angle zero, and that cleared
                                              # the very memory it had just made
        if moved_hold is not None and p.get("held"):
            p["held"]["cell"] = moved_hold[0]      # the grip follows the thing
        # AND THE BODY IS NOT WHERE IT WAS. Identity is resolved once a tick
        # and kept; this moved flesh INSIDE that tick, so the kept answer is
        # now a picture of a body that has since bent. It only ever mattered
        # once a pose could move the body's own corner — everything read
        # afterwards was measured from a corner that had gone.
        p["_claim_tick"] = None
        self._torque_solid = None
        self._slack_mat = None
        return "moved"

    def _hold_torque(self, p, name, theta):
        """What the muscle must find to hold this limb at that angle: the limb's
        own weight times how far out its middle hangs from the joint. Zero
        hanging straight down, most of it held straight out — which is why an
        arm can be kept at your side all day and not at arm's length."""
        # EVERYTHING BEYOND THE JOINT hangs off it, not just the bones with
        # angles: what the hips have to hold up is a torso and a head and two
        # arms, and what a shoulder holds is an arm and whatever is in the hand.
        bones = [b for b in self._subtree(p, self._bones(p, name)[0])
                 if b in (p.get("segs") or {})] if self._bones(p, name) else []
        parts = self._bone_parts(p, bones) if bones else None
        limb = np.concatenate([c for c in parts if len(c)]) \
            if parts and any(len(c) for c in parts) else None
        piv = self._pivot_of(p, name)
        if limb is None or piv is None:
            return 0.0, 0.0
        comp, sl = self._person_cells(p)
        own = np.argwhere(comp)
        own[:, 0] += sl[0].start or 0
        own[:, 1] += sl[1].start or 0
        # the joint nearest the body, and the WHOLE limb's weight hanging off
        # it — a bent elbow is cheaper to hold out precisely because it brings
        # the mass back in, and that falls out of the lever with no new rule
        jx, jy, _jz = (piv + own.min(axis=0)).astype(np.int64)
        kg = self.smass[tuple(limb.T)] / 1000.0
        m = float(kg.sum())
        if m <= 0.0:
            return 0.0, 0.0
        cx = float((limb[:, 0] * kg).sum()) / m
        cy = float((limb[:, 1] * kg).sum()) / m
        lever = max(abs(cx - jx), abs(cy - jy)) * 0.1 * self.scale
        return m * GRAVITY * lever, m

    def _law_pose(self):
        """Limbs go where they are asked, at the speed a muscle can manage, and
        stay there while the muscle can hold them.

        A shoulder is a joint with a motor and a limit, which is a solved thing
        (box3d, MIT) — what is taken here is the shape of it rather than the
        code: an angle, a speed cap, and a torque the motor either has or does
        not. The cap is the same `arm_wmax` Hill's relation already uses for a
        swing, so an arm reaches out at a shoulder's pace and not instantly.

        And an arm held out is HELD OUT, at a cost. If the weight hanging off
        the joint asks more of the muscle than it has, the limb sinks back until
        the lever is short enough — so a limb can be kept at your side for ever
        and at arm's length only while something is paying for it. That is what
        makes a reach a physical act and not a pose in a picture."""
        for p in self.persons:
            want = p.get("reach")
            pose = p.get("pose")
            if not want and not pose:
                continue
            if not p["alive"] or p["safe"]:
                p["reach"] = {}
                want = None
            pose = p.setdefault("pose", {})
            for name in sorted(set(list(pose) + list(want or {}))):
                n = len(self._bones(p, name))          # sorted: the sim carries
                at = np.array(self._angles(            # no die, and a set of
                    pose.get(name, 0.0), n))           # names has no order
                goal = np.array(self._angles((want or {}).get(name, 0.0), n))
                if np.abs(goal).max() > 0.0:
                    need, _m = self._hold_torque(p, name, goal)
                    can = (p.get("torque_Nm") or {}).get(
                        name, p.get("arm_Nm", BODY["arm_Nm"]))
                    if need > can:
                        goal = np.zeros(n)    # too heavy to hold out there
                d = goal - at
                if np.abs(d).max() < 1e-9:
                    # ARRIVED IN ANGLE, BUT NOT IN FLESH. The command has been
                    # given in full and the body is not there — so THIS is how
                    # far this joint goes on this lattice, and the body settles
                    # to where it actually is rather than holding a number it
                    # can never be at.
                    #
                    # The angle outrunning the flesh is deliberate and right:
                    # it is how a limb crosses the angles that cannot be drawn
                    # and catches up at the next one that can. What was missing
                    # is the end of that story. A lean has a long undrawable
                    # tail, so a body's angle read 45.8 degrees while it was
                    # bent 19.3, for ever, and anything that believed the angle
                    # was wrong about the body.
                    dr = np.array(self._angles(
                        (p.get("drawn") or {}).get(name, 0.0), n))
                    if np.abs(dr - at).max() > 1e-9:
                        pose[name] = list(dr)
                        if want is not None and name in want:
                            want[name] = list(dr)
                    continue
                # HOW FAST THAT JOINT GOES. A property of the joint, like the
                # torque it can hold — a trunk is not a shoulder.
                step = (p.get("wmax") or {}).get(
                    name, BODY["arm_wmax"]) * TICK_S
                nxt = at + np.clip(d, -step, step)
                how = self._repose(p, name, nxt)
                if how == "blocked":
                    bent = self._reach_around(p, name, nxt, at)
                    if bent is not None:
                        pose[name] = list(bent)
                        continue
                    # the world said no, every way round. An arm that meets a
                    # post stops AT the post and does not go on wanting to be
                    # past it — and REMEMBERS, against the spot it was standing
                    # on and the way it was facing, so it does not ask the same
                    # question of the same wall every few ticks for ever.
                    (p.get("reach") or {}).pop(name, None)
                    p["_no_reach"] = (p["anchor"],
                                      tuple(p.get("facing", (1.0, 0.0))))
                    if not np.abs(at).max():
                        pose.pop(name, None)
                    continue
                if how == "moved":
                    # AND A POSE YOU CANNOT KEEP IS NOT A POSE YOU ADOPT. The
                    # same gate the muscle already has, with the other reason a
                    # body stops short: not "my back will not hold that" but
                    # "that puts me over my own toes". So a body leans as far
                    # as it can stand and no further, and going over is left to
                    # the things that really do take you over — a load, a shove,
                    # a floor that leaves. Balance measured on the FLESH; what
                    # a body carries is `_pulled_over`'s question, and the two
                    # are complementary rather than duplicated.
                    comp, sl = self._person_cells(p)
                    if comp is not None and comp.any():
                        c = np.argwhere(comp)
                        c[:, 0] += sl[0].start or 0
                        c[:, 1] += sl[1].start or 0
                        if self._underfoot(c) \
                                and self._overbalanced(p, c) is not None:
                            self._repose(p, name, at)      # think better of it
                            # AND HOLD THERE. The intention was "as far as I
                            # can", and this is how far — so it resolves to
                            # HERE rather than being abandoned. Dropped
                            # instead, the goal fell back to nothing and the
                            # body straightened up again the moment it reached
                            # the furthest it could stand, which is a strange
                            # thing to watch and a stranger thing to mean.
                            if want is not None and name in want:
                                want[name] = list(at)
                            continue
                pose[name] = list(nxt)        # the angle moves either way; the
                                              # flesh catches up at the next
                                              # angle the lattice can draw

    def _reach_around(self, p, name, th, at=None):
        """An arm that meets something does not simply stop. It BENDS.

        With one degree of freedom there is exactly one path to a place, so a
        man standing beside the person he is holding could not put his arm out
        at all — that person was standing in the one path. Real arms go round,
        and going round is what a second joint IS.

        The CHOICE was "reach out"; which way the bones get there is motor
        competence, the same kind of thing as knowing the way round a table. So
        this belongs here and not on any menu. It is a sampled search over the
        elbow, coarse on purpose: at 5 cm a one-voxel forearm has only a
        handful of drawable angles anyway, and `_repose` is what says which —
        the first bend that really fits is the one the arm takes."""
        bones = self._bones(p, name)
        if len(bones) < 2:
            return None                       # one bone has nowhere to bend
        # LEAST BEND FIRST — a body that can straighten its arm does. And the
        # shoulder is allowed to stay where it IS as well as go where it was
        # asked: getting the hand somewhere matters more than the angle the
        # shoulder happens to be at, which is the difference between a joint
        # and a dial. Fine steps because drawability on a 5 cm lattice is
        # SPIKY — a coarse ladder walks straight past the angles that fit.
        heads = [float(th[0])]
        if at is not None and abs(float(at[0]) - float(th[0])) > 1e-9:
            heads.append(float(at[0]))
        for head in heads:
            for bend in [s * b for b in np.arange(0.15, 1.8, 0.15)
                         for s in (1.0, -1.0)]:
                alt = [head, float(bend)] + [0.0] * (len(bones) - 2)
                if self._repose(p, name, alt) == "moved":
                    return alt
        return None

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

    _PERCELL = ("cells", "mats", "masses", "Es", "fls", "fvols", "fpots",
                "edges", "segof")

    def _release(self, b, thrown, at):
        """Let go of part of a swinging body, at the speed it was going.

        A throw is not a new kind of motion. The thing is already travelling —
        it has been going round on the end of an arm — and letting go only
        stops it being made to go round. So its speed is the speed it had,
        `omega` times how far out it was, and its direction is the tangent,
        which is where the hand was taking it anyway."""
        ax, sgn = int(b["axis"]), int(b["s"])
        vox_m = 0.1 * self.scale
        c = b["cells"][thrown]
        d_lat = float(c[:, ax].mean()) - b["pivot"]
        d_z = float(c[:, 2].mean()) - b["zb"]
        th = float(b["theta"])
        ct, st = float(np.cos(th)), float(np.sin(th))
        w_rad = float(b["omega"]) / max(TICK_S, 1e-9)
        vel = [0.0, 0.0, 0.0]
        vel[ax] = (-d_lat * st + sgn * d_z * ct) * w_rad * vox_m
        vel[2] = (-sgn * d_lat * ct - d_z * st) * w_rad * vox_m
        flier = {"cells": at[thrown].astype(np.float32),
                 "fly": True, "vel": np.asarray(vel, np.float64),
                 "off": np.zeros(3), "owner": None,
                 "axis": 0, "s": 1, "pivot": 0.0, "zb": 0.0,
                 "theta": 0.0, "omega": 0.0, "phi0": 0.02, "L": 2.0}
        for k in ("mats", "masses", "Es", "fls", "fvols", "fpots", "edges"):
            flier[k] = b[k][thrown].copy()
        for k in self._PERCELL:
            if k in b and b[k] is not None and len(b[k]) == len(thrown):
                b[k] = b[k][~thrown]
        # AND THE ARM IS LIGHTER NOW, so it comes round faster — which is what
        # anyone who has thrown something has felt.
        if len(b["cells"]):
            kg = b["masses"] / 1000.0
            r = (np.abs(b["cells"][:, ax] - b["pivot"])
                 + np.abs(b["cells"][:, 2] - b["zb"])) * vox_m
            b["I"] = max(float((kg * r * r).sum()), 1e-6)
        b["throw"] = False
        self.bodies.append(flier)
        for q in self.persons:                    # the hand is empty now
            if q["name"] == b.get("owner"):
                q["held"] = None
                q["events"].append(f"t{self.tick}: {q['name']} lets fly")
                break
        return flier

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
                # LET GO, if that is what this swing was for. Before the block
                # check, because a thrown thing leaves the hand and what the
                # ARM then hits is the arm's business.
                if b.get("throw") and b.get("segof") is not None \
                        and theta_next >= BODY["throw_rad"] \
                        and (b["segof"] == -1).any():
                    b["theta"] = theta_next
                    self._release(b, b["segof"] == -1,
                                  np.round(self._body_pose(b, theta_next)))
                    if not len(b["cells"]):
                        continue                  # nothing of it left to swing
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

    def _left_world(self, b, i):
        """One voxel of a body went past the edge of the lattice, and is gone.

        Recorded rather than dropped, because "mass is conserved" has to stay a
        statement you can CHECK: everything is on the lattice, or in a body in
        flight, or on this tally. A world with edges loses things out of them;
        a world that cannot say how much it lost is just wrong."""
        g = self.gone
        m = int(b["mats"][i])
        g[m] = g.get(m, 0.0) + float(b["masses"][i])
        g["E"] = g.get("E", 0.0) + float(b["Es"][i])
        if b["fvols"][i] > 0:
            g[("fluid", int(b["fls"][i]))] = \
                g.get(("fluid", int(b["fls"][i])), 0.0) + float(b["fvols"][i])

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
        # A ROLL IS NOT ONE LANDING. It is several, each taking a share of the
        # fall on a different part of you — feet, then hip, then back, then
        # shoulder — and tissue damage is a THRESHOLD, so a blow divided into
        # parts that each fall under it does no harm at all while the energy is
        # unchanged. That is force being momentum divided by the time taken to
        # lose it, written in the arithmetic this model already has.
        #
        # How many parts is read off the body: how many times its own crouch
        # goes into its own length, which is how far a body rolling over itself
        # has to travel before it stops. Nothing typed in.
        shares = 1
        if b.get("owner") is not None and not b.get("part"):
            who = next((q for q in self.persons
                        if q["name"] == b["owner"]), None)
            if who is not None and who.get("rolling"):
                tall = (float(pose[:, 2].max() - pose[:, 2].min()) + 1.0) * vox_m
                shares = max(1, int(round(tall / max(BODY["crouch_m"], 1e-9))))
                who["rolling"] = False
                who["events"].append(f"t{self.tick}: {who['name']} rolls with it")
        e_per = joules / float(shares) / max(int(contact.sum()), 1)
        sore = 0.0                          # energy the faller's own flesh took
        order = np.argsort(pose[:, 2])      # kept: put[k] belongs to kept[k],
        kept = []                           # so a landing can be told apart
        for i in order:                     # into flesh and carried tool
            # THE WORLD HAS EDGES, AND WHAT GOES PAST THEM IS OUTSIDE IT.
            # These three lines used to CLAMP: a cell thrown past the wall was
            # set down on the wall instead, several of them into the same
            # column, and each `smass =` overwrote the last. Mass went missing,
            # quietly, in the one field every test in the suite leans on.
            #
            # Clamping was never right anyway — it teleports matter. A thing
            # that leaves is gone, and saying so keeps the books: it is on the
            # lattice, or in a body, or on this tally, and the three add up.
            dx, dy, dz = int(pose[i, 0]), int(pose[i, 1]), int(pose[i, 2])
            if not (0 <= dx < nx and 0 <= dy < ny and 0 <= dz < nz):
                self._left_world(b, i)
                continue
            while dz < nz - 1 and self.mat[dx, dy, dz] != AIR:
                dz += 1                                  # blocked: pile upward as debris
            if self.mat[dx, dy, dz] != AIR:
                self._left_world(b, i)       # the column is full to the ceiling
                continue                     # and there is nowhere left to put it
            put.append((dx, dy, dz))
            kept.append(i)
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
                        sore += (e_per - bruise) * shares   # the overage, once
                    continue                                # per share of the
                                                            # blow, is the wound
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
            # EVERY CELL OF IT MAY HAVE LEFT THE WORLD — a limb swung off the
            # edge comes back as nothing, and `np.array([])` is not a list of
            # coordinates. Say the shape or the filter below has nothing to
            # filter.
            put_arr = np.array(put, np.int64).reshape(-1, 3)
            mats_put = b["mats"][np.array(kept, np.int64)] \
                if kept else b["mats"][:0]
            for q in self.persons:            # the limb came to rest somewhere
                if q["name"] == b["owner"] and q.get("segs") and q.get("_own"):
                    origin = np.min(np.array(list(q["_own"])), axis=0)
                    # only the FLESH is the limb — a swung tool came along for
                    # the ride and is not part of anyone's arm
                    # EACH BONE BACK TO ITS OWN OFFSETS. Writing the whole
                    # arm under the arm's name leaves the bones holding the
                    # cells they were in before the swing — which are empty —
                    # and the next pose moves nothing and loses the limb.
                    seg_put = b["segof"][np.array(kept, np.int64)] \
                        if b.get("segof") is not None and kept \
                        else np.where(mats_put == FLESH, 0, -1)
                    for i, bone in enumerate(b.get("bones") or [b["seg"]]):
                        q["segs"][bone] = put_arr[seg_put == i] - origin
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

    def _hot_window(self, pad=1):
        """The slab of the world heat can be in, grown by one voxel of room to
        move into — or None when nothing anywhere is above ambient.

        Heat travels one voxel per LAW THAT MOVES IT, and no further: every
        radiative term is between TOUCHING faces (see the pair loop below), and
        the only other heat path is each voxel's own small loss to the world
        outside, which cannot warm a neighbour. So a box around everything above
        ambient, grown by one, contains everything ONE such law can write.

        Not one voxel a TICK, which is what this first claimed and is wrong:
        conduction and buoyancy both move heat and both run in the same tick, so
        heat really does get two voxels from home before the tick is out. The
        measured symptom was a single cell — the sim left 3.4 J at z4 that the
        windowed run left at zero, on the very first tick, because conduction
        carried heat up to z3 and buoyancy then carried some of it to z4, one
        past the edge of a box drawn before either ran.

        So the window is asked for again between the two, rather than padded by
        two once. That costs one more scan of a field we have already touched,
        and it stays right when a third law that moves heat is added — which a
        comment saying "pad by two" would not.

        This is the half that matters for a big world: a town with one room
        alight answers "yes, something is hot" and would otherwise get no skip
        at all, because presence alone is a question about the WHOLE world."""
        return self._window(self.E != 0, pad)

    def _window(self, there, pad):
        """A box around everything `there` is true of, grown by `pad` — or None
        if it is nowhere. With skipping off, the whole world."""
        if not self.skip_quiet:
            return (slice(None), slice(None), slice(None))
        out = []
        for ax in range(3):
            line = there.any(axis=tuple(a for a in range(3) if a != ax))
            nz = np.flatnonzero(line)
            if not len(nz):
                return None
            out.append(slice(max(int(nz[0]) - pad, 0),
                             min(int(nz[-1]) + 1 + pad, self.shape[ax])))
        return tuple(out)

    def _law_conduct(self, sl=None):
        sl = sl if sl is not None else (slice(None),) * 3
        E = self.E[sl]                                   # a view: writing to it
        mat = self.mat[sl]                               # writes the world
        C = self.heat_capacity()[sl]
        T = AMBIENT + E / C
        k = _KSOLID_ARR[mat].astype(np.float32)
        # hot gas convects: rising, churning air moves heat far faster than still air.
        # Modeled as conductivity growing with temperature (standard trick, not a case).
        hot_air = mat == AIR
        k[hot_air] += np.maximum(T[hot_air] - AMBIENT, 0.0) / 60.0
        fk = _FK_ARR[self.fl[sl]].astype(np.float32)
        wet = (self.fvol[sl] > 0.1 * self.cap) & (mat == AIR)       # OPEN fluid on a solid:
        solid = mat != AIR                               # contact is governed by the FLUID
        if self.fused and HAVE_NUMBA:                    # (a boiling film, not the timber)
            sc, rd = np.float32(self.scale), np.float32(RAD)
            for di, dj, dl in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                _conduct_pairs(E, T, C, k, fk, wet, solid, sc, rd, di, dj, dl)
            return
        for axis in range(3):
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
            E[a] -= q
            E[b] += q

    def _law_rise(self, sl=None):
        """Buoyancy: hot gas rises. A hot air voxel hands much of its excess heat to the
        air above it; at the open top of the world, the plume leaves entirely. THIS is why
        a doused fire can stay out — the reheating cloud escapes upward instead of sitting."""
        sl = sl if sl is not None else (slice(None),) * 3
        E, mat = self.E[sl], self.mat[sl]
        T = AMBIENT + E / self.heat_capacity()[sl]
        lo = (slice(None), slice(None), slice(None, -1))
        up = (slice(None), slice(None), slice(1, None))
        both_air = (mat[lo] == AIR) & (mat[up] == AIR)
        dT = T[lo] - T[up]
        # only the GAS'S share of the cell's heat rises (c_airbase, the air's own
        # capacity) — a pool of liquid is not a plume, its heat stays put. Found the
        # hard way: rising 0.35*C_total blew a lamp's oil heat skyward and no taper
        # could ever light a pool. For pure air the two are identical.
        q = np.where(both_air & (dT > 0), 0.35 * self.c_airbase * dT, 0.0).astype(np.float32)
        E[lo] -= q
        E[up] += q
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
        # WHERE THE AIR IS does not change when the air changes — only when the
        # WALLS do. This was a full downsample of the lattice plus a wide blur
        # of it, rebuilt every tick, for a field that depends on nothing but
        # which voxels are air. Thrown away by `_air_regions`, which already
        # compares the air mask against last tick's and so already knows: no
        # extra comparison, and right by construction rather than by anyone
        # remembering that a burning wall changes where the air is.
        #
        # Keyed on the MATTER fields first, which was the obvious thing and the
        # wrong one — a fire eats wood, `smass` changes every tick, and the
        # cache was thrown away every tick for a field that had not moved.
        airc = self._aircells
        if airc is None:
            airc = self._aircells = self._cells(air.astype(np.float32))
        # a plume's CATCHMENT: it entrains from meters around, not from its own
        # shoebox — health and the debt below must use the SAME reach, or the
        # fire eats a small pocket, reads the vacuum as suffocation, and
        # strangles itself while the room is still full of air (measured).
        # The blur is AIR-WEIGHTED: a wall is not a region of zero oxygen, it
        # is simply not air — averaging it in as 0 suffocated every fire that
        # burned near a floor (measured: health 0.21 under a fresh sky)
        reach = max(2, int(round(PLUME_REACH_M / (0.1 * self.scale * self._CS))))
        o2b = self._blur_cells(o2c, reach)               # o2c is already per-SLOT
        airb = self._airblur
        if airb is None:
            airb = self._airblur = np.maximum(
                self._blur_cells(airc, reach), 1e-6)
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
        # A WORLD THAT ALREADY DIVIDES EVENLY NEEDS NO PADDING, and np.pad copies
        # the whole lattice whether it has anything to add or not. Most worlds
        # divide evenly — this was a spare full-size copy of a 4 MB array, made
        # eleven times a tick, for nothing.
        a = arr if (px or py or pz) == 0 else np.pad(arr, ((0, px), (0, py), (0, pz)))
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

    # The most voxels smoke moves along ONE axis in a single pass of the law
    # below: a lateral escape, a seep, and two turns of the ceiling jet. Counted
    # off the code, not guessed. Add a phase that shifts smoke sideways and this
    # must go up — and if you forget, the tick-by-tick guard says so, which is
    # why it is safe to have a number here at all.
    _SMOKE_REACH = 4

    def _law_smoke(self, sl=None):
        """Smoke is a gas: it seeps to neighbors, RISES hard through open air, and
        leaves at the open sky. The mass that burning removes from wood and oil
        travels here — conservation made visible.

        Run over a box around the smoke there IS, because this law makes some
        thirteen passes over a field that is empty nearly everywhere: a fire in
        one hut of a town had every pass sweeping the whole town."""
        sl = sl if sl is not None else (slice(None),) * 3
        smoke = self.smoke[sl]                    # a view: writing writes the world
        air = self.mat[sl] == AIR
        por = self.porosity()[sl]
        # smoke is born INSIDE burning solids — it must escape through the surface
        # first (up if it can, sideways if it must), or it stays trapped in the wood
        lo = (slice(None), slice(None), slice(None, -1))
        up = (slice(None), slice(None), slice(1, None))
        esc = np.where(~air[lo] & air[up], 0.9 * smoke[lo], 0.0).astype(np.float32)
        smoke[lo] -= esc
        smoke[up] += esc
        for a, b in ((( slice(None, -1), slice(None), slice(None)), (slice(1, None), slice(None), slice(None))),
                     ((slice(None), slice(None, -1), slice(None)), (slice(None), slice(1, None), slice(None)))):
            for s_, d_ in ((a, b), (b, a)):
                esc = np.where(~air[s_] & air[d_], 0.5 * smoke[s_], 0.0).astype(np.float32)
                smoke[s_] -= esc
                smoke[d_] += esc
        for axis in range(3):                                # seep: even out with neighbors
            a = [slice(None)] * 3; b = [slice(None)] * 3
            a[axis], b[axis] = slice(None, -1), slice(1, None)
            a, b = tuple(a), tuple(b)
            q = 0.08 * (smoke[a] - smoke[b]) * np.minimum(por[a], por[b])
            smoke[a] -= q
            smoke[b] += q
        lo = (slice(None), slice(None), slice(None, -1))
        up = (slice(None), slice(None), slice(1, None))
        risable = air[lo] & air[up]                          # rise: buoyant, fast
        q = np.where(risable, 0.4 * smoke[lo], 0.0).astype(np.float32)
        smoke[lo] -= q
        smoke[up] += q
        # the CEILING JET: smoke that can rise no further pools against the
        # ceiling and RACES along it — buoyancy makes it an upside-down liquid
        # seeking an upside-down level. Without this the bank crawled at
        # diffusion pace and the far half of a burning room read 0.00 g
        # forever (measured; a real ceiling jet runs meters per second).
        pinned = np.zeros_like(air)
        pinned[:, :, :-1] = air[:, :, :-1] & ~air[:, :, 1:]  # air with solid above
        at_roof = sl[2].stop in (None, self.shape[2])     # is the window's top
        pinned[:, :, -1] = air[:, :, -1] & (not self.open_sky) & at_roof
        for _ in range(2):
            for axis in (0, 1):
                a = [slice(None)] * 3; b = [slice(None)] * 3
                a[axis], b[axis] = slice(None, -1), slice(1, None)
                a, b = tuple(a), tuple(b)
                m = pinned[a] & pinned[b]
                q = np.where(m, 0.25 * (smoke[a] - smoke[b]),
                             0.0).astype(np.float32)
                smoke[a] -= q
                smoke[b] += q
        if at_roof:                                          # the open sky takes it
            top = (slice(None), slice(None), -1)             # — but only the real
            smoke[top][air[top]] *= 0.5                      # sky, not a box edge

    def _air_regions(self):
        """Label the connected airspaces (6-connectivity, pure numpy flood by
        max-propagation). A fire can only breathe the air ITS OWN airspace holds —
        a sealed wall is a real wall, a doorway joins two rooms into one lung.
        Recomputed only when the set of air voxels changes (a wall burns through,
        a collapse opens a gap)."""
        air = self.mat == AIR
        if self._region_air is not None and bool((air == self._region_air).all()):
            return
        self._aircells = self._airblur = None    # the walls moved: so did the air
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
        the fire eats what the room holds, then stops — fuel or no fuel.

        MIXING IS WINDOWED, and unlike smoke it needs no judgement to be. Air
        that is everywhere at the same composition has nothing to even out —
        every pair cancels — so the box around where the field is NOT fresh
        contains every exchange there is, and the skip is exact. Measured in a
        burning town, that box is 21% of the world at tick 20 and 63% at tick
        69: the deficit really does stay where the fire is, which is exactly
        what smoke does NOT do (an infinitesimal trace of soot reaches every
        cell within ten ticks, so `smoke != 0` is useless and smoke needs a
        stated threshold instead)."""
        T = self.T()
        air = self.mat == AIR
        por = self.porosity()
        fused = self.fused and HAVE_NUMBA
        # one hop per axis for diffusion; the overturn below adds one more in z
        mix = self._window(self.o2 != O2_PER_L * self.vox_l, 1)
        for axis in range(3):
            sl = mix if mix is not None else (slice(None),) * 3
            if fused:
                d = [0, 0, 0]; d[axis] = 1
                _o2_diffuse(self.o2[sl], T[sl], por[sl], d[0], d[1], d[2])
                continue
            a = [slice(None)] * 3; b = [slice(None)] * 3
            a[axis], b[axis] = slice(None, -1), slice(1, None)
            a, b = tuple(a), tuple(b)
            o2, Tw, pw = self.o2[sl], T[sl], por[sl]
            k = np.clip(0.1 + (Tw[a] + Tw[b]) / 1600.0, 0.1, 0.45)
            k = k * np.minimum(pw[a], pw[b])         # a barrier passes what its PORES
            q = k * (o2[a] - o2[b])                  # allow: a shut door breathes a
            o2[a] -= q                               # little, masonry almost nothing,
            o2[b] += q                               # a pane nothing at all
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
        # ASKED AGAIN. The diffusion above has just moved oxygen, so the box it
        # was drawn around is one voxel out of date in z — the same mistake the
        # heat window made on its first day, when conduction carried heat up one
        # cell and buoyancy carried it up another, one past the edge of a box
        # drawn before either ran. Smoke rides along, so its own spread counts
        # too.
        turn = self._window((self.o2 != O2_PER_L * self.vox_l)
                            | (self.smoke > SMOKE_STILL), 1)
        tsl = turn if turn is not None else (slice(None),) * 3
        if fused:
            _gas_overturn(self.o2[tsl], self.smoke[tsl], T[tsl], air[tsl])
        else:
            Tw, aw = T[tsl], air[tsl]
            dT = Tw[lo] - Tw[hi]                         # >0 = unstable: hot below
            f = np.where(aw[lo] & aw[hi],
                         np.clip(dT * 0.02, 0.0, 0.45), 0.0).astype(np.float32)
            for gas in (self.o2[tsl], self.smoke[tsl]):  # a parcel is ONE gas: its
                q = f * (gas[lo] - gas[hi])              # oxygen and its soot
                gas[lo] -= q                             # travel together
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
            # AN AIRSPACE WITH NOTHING IN IT TO STIR IS NOT STIRRED. Bullet's
            # island rule, which is the right one here for the same reason it is
            # right there: the thing that may sleep is a whole connected body,
            # never part of one, and anything that disturbs it wakes all of it.
            # An airspace IS an island — gas mixes room-wide, so a room is the
            # smallest thing that can be called still.
            #
            # A room every cell of which holds fresh air, no soot and no warmth
            # has one empty zone and one uniform zone, and relaxing a uniform
            # thing toward its own mean moves nothing. The cost of asking is two
            # comparisons and a count; the cost of not asking was a blur of the
            # WHOLE lattice, three times per zone, for every room in the world.
            # Measured in a building of twenty sealed rooms with a fire in one:
            # nineteen were asleep, holding 95% of all the air.
            live = np.ones(self._region_count, bool)
            if self.skip_quiet:
                still = air & ((self.o2 != O2_PER_L * self.vox_l)
                               | (self.smoke > 0.0) | (T > 45.0))
                live = np.bincount(self.air_region[still],
                                   minlength=self._region_count) > 0
            for r in np.nonzero(sizes > 64)[0]:  # a pocket too small to hold a
                if not live[r]:                  # circulation isn't stirred, and
                    continue                     # neither is a room at rest
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
             "known": np.full(self.shape[:2], bool(knows_world)),
             # AND WHAT IT COULD WALK THROUGH when it looked. `known` says
             # where the eyes have been; this says what they FOUND there, and
             # it is what the route planner runs on. Built on the first look,
             # because a body's height decides which columns it fits down and
             # there is no flesh on the lattice yet at this point.
             "free": None, "knew": bool(knows_world)}
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
            # WHERE THIS BODY IS, kept up to date whether or not it is awake
            # enough to ask. `_eye` was written only by the will layer, and the
            # will layer skips anyone unconscious — so a man dragged out of a
            # fire was still HEARD from where he fainted, and at standing head
            # height while he lay on the floor. Measured: dragged eighteen
            # voxels to the door and located by everyone else back in the
            # burning room. Being carried is a thing that happens TO you.
            fw = np.argwhere(flesh)
            p["_eye"] = (float(fw[:, 0].mean()) + (sl[0].start or 0),
                         float(fw[:, 1].mean()) + (sl[1].start or 0),
                         float(fw[:, 2].max()))
            zs = fw[:, 2]
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
            p["skin"] = float(T[sl][flesh].max())   # what the body can FEEL
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

    def _look_around(self, p, eye, cells):
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
        # AND WHAT WAS THERE, not merely that there was a there. Belief recorded
        # geometry only: a body could remember the shape of a room it had walked
        # through and nothing whatever about the fire in it, so when it came to
        # choose somewhere to go, every place it knew looked equally good. That
        # is why an idle body would stroll toward a blaze — not because it was
        # brave, but because nothing it remembered said otherwise.
        #
        # Read by COLUMN rather than at eye height, because light from a fire at
        # your feet reaches your eyes: what the ray decides is whether you can
        # SEE that far, which is the occlusion the loop above already did.
        #
        # And it FADES. A body that never forgets treats an hour-old fire as a
        # fire; one that forgets at once has no memory to speak of.
        danger = p.get("danger")
        if danger is None:
            danger = p["danger"] = np.zeros(self.shape[:2], np.float32)
        danger *= WILL["forget"]
        alight = self.burning().any(axis=2)
        np.maximum(danger, alight.astype(np.float32), out=danger)
        danger[xi[reach], yi[reach]] = np.where(
            alight[xi[reach], yi[reach]], 1.0, danger[xi[reach], yi[reach]])
        # AND WHETHER IT COULD GET THERE. This is the same god-channel as the
        # door register, one layer down and much better hidden: belief gated
        # WHERE a body had looked, and then the planner read the floor itself,
        # live, every time. So a corridor blocked by a beam that fell behind a
        # body's back was routed around instantly, by a body that never turned
        # its head — and a wall breached out of sight became a way through the
        # moment it was breached. What a body can plan over is now what it SAW
        # it could plan over, and nothing else.
        #
        # It does not fade. A floor is not a fire: the useful error is a body
        # that trusts a way it can no longer take and finds out by going, not
        # one that forgets the shape of a room it was standing in a moment ago.
        walk_now = self._walkable(cells)
        free = p.get("free")
        if free is None:                  # built on the FIRST look, because a
            free = p["free"] = (          # body's height decides which columns
                walk_now.copy() if p.get("knew")   # it fits down, and there is
                else np.zeros(self.shape[:2], bool))  # no flesh yet at signup
        free[xi[reach], yi[reach]] = walk_now[xi[reach], yi[reach]]
        # A BODY IS NOT A WALL TO ITSELF — the same rule the planner already
        # keeps, and it has to be kept HERE too or belief is poisoned by the
        # looker. Its own flesh is denser than PUSH_THROUGH, so every look
        # wrote the ground under its own feet down as blocked, it walked on,
        # and nothing ever looked back to correct it. Measured before this
        # line: a body crossing an empty 4 m hall laid 109 phantom walls
        # behind itself and believed every one of them. What it carries goes
        # with it, for the same reason and by the same mistake.
        free[cells[:, 0], cells[:, 1]] = True
        if p.get("held"):
            obj = self._held_cells(p)
            if obj is not None and len(obj):
                free[obj[:, 0], obj[:, 1]] = True

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
        real obstacle, which is what gives jumping across one a point.

        FLAT, and that is a known limit rather than an oversight: every column
        is tested at the body's own height and no other, so a kerb is a wall and
        a stair is a wall. Making it height-aware is item 5, and it was TRIED —
        see "What a step up cost" in the roadmap for why it came back out and
        what it has to wait for."""
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

    def _contact(self, cells, ignore=None):
        """Which cells of this body are actually resting on something else.

        Not the bottom layer — ANY cell with something solid beneath it that is
        not part of this same body. A man at the very lip has his feet over air
        and a forearm still over the stone, and that forearm is genuinely
        touching the world. Standing on your own foot proves nothing.

        And neither does standing on what you are CARRYING, which is what
        `ignore` is for. A man holding a block has his arm directly above it, so
        the block came back as part of his own footprint — which made his base
        as wide as his reach and said he could never be overbalanced by anything
        he could hold. What you carry is not what carries you."""
        cells = np.asarray(cells, np.int64)
        on_floor = cells[:, 2] <= 0                   # the floor of the world
        z1 = np.maximum(cells[:, 2] - 1, 0)
        under = self.mat[cells[:, 0], cells[:, 1], z1] != AIR
        ny, nz = self.shape[1], self.shape[2]
        key = (cells[:, 0] * ny + cells[:, 1]) * nz + cells[:, 2]
        mine = np.isin(key - 1, key)
        if ignore is not None and len(ignore):
            ig = np.asarray(ignore, np.int64)
            mine = mine | np.isin(key - 1,
                                  (ig[:, 0] * ny + ig[:, 1]) * nz + ig[:, 2])
        return cells[on_floor | (under & ~mine)]

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
        R = max(int(round(self._reach_of(p) / max(vox_m, 1e-9))), 1)
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

    def _in_flight_near(self, p, own):
        """Things in the air that a hand of this body could close on now.

        A catch is a grab at something that will not wait. So the test is the
        same reach a grab uses, asked of a body that has left the grid — which
        is why this needed nothing new once things could be thrown: a flier
        already has a position and a velocity, and a fist already has a place."""
        out = []
        if not self.bodies:
            return out
        reach = self._reach_of(p) / max(0.1 * self.scale, 1e-9)
        fists = np.asarray(self._fists(p, own), np.float64)
        for b in self.bodies:
            # A THING, NOT A BODY. An owned body is somebody's flesh — a person
            # falling, or a person hanging from your own fist — and neither of
            # those is a thing thrown at you. Without this, holding a man over
            # a drop made you perceive him as a missile every tick, which
            # crowded out every other percept you might have had about the
            # situation you were actually in. Catching a falling PERSON is a
            # real act and a different one (item 56).
            if not b.get("fly") or b.get("part") or b.get("owner") is not None:
                continue
            at = self._fly_pose(b)
            d = np.abs(at[None, :, :] - fists[:, None, :]).max(axis=2)
            if float(d.min()) <= reach:
                out.append(b)
        return out

    def _falling_near(self, p, own):
        """PEOPLE in the air a hand of this body could close on now.

        The twin of `_in_flight_near`, and separate from it on purpose: what
        you do about a falling friend is not what you do about a thrown stone,
        so they are two percepts and two rows. The arithmetic underneath is the
        same — a body in the air has a position and a velocity, a fist has a
        place, and `m v / t` says whether an arm can stop it."""
        out = []
        if not self.bodies:
            return out
        reach = self._reach_of(p) / max(0.1 * self.scale, 1e-9)
        fists = np.asarray(self._fists(p, own), np.float64)
        for b in self.bodies:
            who = b.get("owner")
            if not b.get("fly") or b.get("part") or who is None \
                    or who == p["name"]:
                continue
            at = self._fly_pose(b)
            d = np.abs(at[None, :, :] - fists[:, None, :]).max(axis=2)
            if float(d.min()) <= reach:
                out.append(b)
        return out

    def _catch_person(self, p, b, arm):
        """Close a hand on someone who is falling.

        The same sum as catching a stone — stopping is force times time — and
        then the same GRIP that has always been able to hold a hanging man.
        Nothing was added for this: a caught body goes back on the lattice
        where it was caught, and `_grip_cells` seeds what a hand holds as
        supported, so he hangs there because a hand is holding him and for no
        other reason."""
        m = float(b["masses"].sum()) / 1000.0
        v = float(np.linalg.norm(b["vel"]))
        if m * v / max(BODY["catch_s"], 1e-9) > self._hand_strength(p, arm):
            return False
        who = b.get("owner")
        b["vel"] = np.zeros(3)
        self._land_body(b, None)
        if b in self.bodies:
            self.bodies.remove(b)
        p["dragging"] = who
        return True

    def _catch(self, p, b, arm):
        """Close a hand on something in flight, if the hand can stop it.

        Stopping is force times time: a thing of mass m at speed v needs
        `m v / t` to be brought to rest in `t`, and a hand has only so much.
        That is the whole rule, and it is the same arithmetic as lifting — a
        catch that is too much for the arm simply does not happen, and the
        thing goes on its way past a hand that could not close on it."""
        m = float(b["masses"].sum()) / 1000.0
        v = float(np.linalg.norm(b["vel"]))
        if m * v / max(BODY["catch_s"], 1e-9) > self._hand_strength(p, arm):
            return False
        at = np.round(self._fly_pose(b)).astype(np.int64)
        b["vel"] = np.zeros(3)                # caught: it arrives with nothing
        self._land_body(b, None)
        if b in self.bodies:
            self.bodies.remove(b)
        nx, ny, nz = self.shape
        for c in at:
            cx, cy, cz = (min(max(int(c[0]), 0), nx - 1),
                          min(max(int(c[1]), 0), ny - 1),
                          min(max(int(c[2]), 0), nz - 1))
            if int(self.mat[cx, cy, cz]) != AIR:
                p["held"] = {"cell": (cx, cy, cz), "mat": int(self.mat[cx, cy, cz]),
                             "label": MATNAME.get(int(self.mat[cx, cy, cz]), "thing"),
                             "arm": arm}
                return True
        return True

    def _body_span(self, p, what):
        """A LENGTH THIS BODY ACTUALLY HAS, in metres, measured off its own
        voxels — not a number typed into `BODY` and hoped to match.

        This is the loop worth having: the sim already knows where every voxel
        of every limb is, so anything shaped like "how far can an arm reach" or
        "how wide is a pair of shoulders" has a ground truth sitting right
        there. Reading it instead of declaring it means a child, a giant and a
        one-armed man each get the right answer without anybody writing three
        rows, and it means the number cannot drift out of step with the body it
        describes — which `BODY["reach_m"]` silently had, at 0.30 m against an
        arm that is 0.45 m long.

        The constants stay as the FALLBACK, for a body with no segments
        declared. They are what a person is like when nobody has said."""
        got = (p.get("_span") or {}).get(what)
        if got is not None:
            return got
        vox_m = 0.1 * self.scale
        out = None
        arms = [a for a in self._limbs(p) if "arm" in a]
        if what == "reach" and arms:
            piv = self._pivot_of(p, arms[0])
            parts = self._limb_parts(p, arms[0])
            if piv is not None and parts:
                cells = np.concatenate([c for c in parts if len(c)]) \
                    if any(len(c) for c in parts) else None
                if cells is not None and len(cells):
                    comp, sl = self._person_cells(p)
                    if comp is not None and comp.any():
                        own = np.argwhere(comp)
                        own[:, 0] += sl[0].start or 0
                        own[:, 1] += sl[1].start or 0
                        j = np.asarray(piv) + own.min(axis=0)
                        out = float(np.abs(cells - j).sum(axis=1).max()) * vox_m
        elif what == "shoulders" and len(arms) > 1:
            mids = []
            for a in arms:
                parts = self._limb_parts(p, a)
                if parts and len(parts[0]):
                    mids.append(parts[0].mean(axis=0))
            if len(mids) > 1:
                out = float(np.abs(mids[0] - mids[1]).max()) * vox_m
        if out is None or out <= 0.0:
            out = BODY["reach_m"] if what == "reach" else BODY["off_hand_m"] * 2
        p.setdefault("_span", {})[what] = out
        return out

    def _reach_of(self, p):
        """How far THIS body's arm goes."""
        return self._body_span(p, "reach")

    def _hand(self, p, own, toward=None, free=False):
        """WHICH HAND does this.

        The dominant one, unless the other is plainly the one for the job: a
        thing on your left is nearer your left hand, and past enough difference
        that beats being right-handed. So a body uses its good hand most of the
        time and its other hand when the world asks for it — never because a
        random number said so, and the sim carries no die anyway.

        An arm that is not there, or already holding something, is not a
        candidate: an indisposed hand is not a choice, it is an absence."""
        arms = [a for a in self._limbs(p) if "arm" in a]
        if not arms:
            return None
        dom = str(p.get("handed", "right")) + " arm"
        # HALF THIS BODY'S OWN SHOULDER WIDTH, read off its own shoulders.
        # The cost of the off hand is that it is weaker; the benefit of the
        # near hand is not reaching across yourself; those trade at about the
        # distance between your shoulders, so a broad man switches hands later
        # than a narrow one and nobody writes that down twice.
        bias = 0.5 * self._body_span(p, "shoulders") / max(0.1 * self.scale, 1e-9)
        busy = (p.get("held") or {}).get("arm") if free else None
        best, score = None, None
        for a in sorted(arms):                # sorted: deterministic ties
            if a == busy:
                continue
            f = self._fist_of(p, a, own)
            if f is None:
                continue                      # nothing left of that arm
            sc = 0.0 if a == dom else bias
            if toward is not None:
                sc += float(np.linalg.norm(np.asarray(toward, np.float64)
                                           - np.asarray(f, np.float64)))
            if score is None or sc < score:
                best, score = a, sc
        return best

    def _hand_strength(self, p, arm):
        """What that hand can put out. The other one is weaker, which is most
        of what having a dominant hand MEANS."""
        full = p.get("strength_N", BODY["strength_N"])
        return full if arm == str(p.get("handed", "right")) + " arm" \
            else full * BODY["off_hand"]

    def _fist_of(self, p, name, own):
        """The HAND at the end of one arm: the voxel of the last bone furthest
        from the joint the whole limb hangs from.

        Said that way because it is true in every pose. It used to be "the mean
        x and y of the arm, at its lowest z", which is exactly right for an arm
        hanging straight down and exactly wrong for one held out — a horizontal
        arm's mean is its ELBOW. Measured: a man holding 38 kg at arm's length
        had the load's weight applied half way up his forearm, which halved the
        lever and let him keep his feet under a load that should plainly have
        taken him off them. Arms only hung down when that was written."""
        parts = self._limb_parts(p, name, own)
        piv = self._pivot_of(p, name)
        if not parts or piv is None or not len(parts[-1]):
            return None
        far = parts[-1]
        j = np.asarray(piv) + np.asarray(own).min(axis=0)
        d = np.abs(far.astype(np.int64) - j.astype(np.int64)).sum(axis=1)
        return [float(c) for c in far[int(np.argmax(d))]]

    def _fists(self, p, own):
        """Where this body's hands are. A body with no segments declared falls
        back to its own middle height."""
        fists = []
        for k in self._limbs(p):              # per ARM, not per bone: an elbow
            if "arm" not in k:                # is not a second hand
                continue
            f = self._fist_of(p, k, own)
            if f is not None:
                fists.append(f)
        if not fists:
            zs = own[:, 2]
            fists = [[float(own[:, 0].mean()), float(own[:, 1].mean()),
                      float(zs.min()) + 0.55 * float(zs.max() - zs.min())]]
        return fists

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
        fists = self._fists(p, own)
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
            for cl in self._held_clusters(p):
                if self._grip_holds(p, cl):
                    out.update(map(tuple, np.asarray(cl)))
        return frozenset(out)

    def _rub(self, p, own):
        """Dragging a thing over a floor HEATS BOTH OF THEM.

        Work is force times distance and friction is a force, so a thing hauled
        one voxel over the ground has turned `drag_N * voxel` joules of muscle
        into heat — there is nowhere else for it to have gone. Split evenly
        between the thing and what it is dragged over, because a rubbing pair
        is two surfaces and the third law does not care which one you were
        thinking about.

        And this needed NO NEW RULE. `E` is the field combustion already reads,
        so a thing dragged far enough over a rough floor gets hot, and a thing
        that gets hot enough catches. Nobody wrote "dragging can start a fire";
        it is what `mu N v` and a combustion law mean together.

        Known softness, and it is the body's: muscle is an unmodelled source of
        energy here. A man has `blood_o2` and no calorie budget, so the joules
        he puts into the floor come from nowhere the sim is counting."""
        vox_m = 0.1 * self.scale
        nz = self.shape[2]
        for cl in self._held_clusters(p):
            cl = np.asarray(cl, np.int64)
            if not len(cl):
                continue
            rub = self._contact(cl, ignore=own)
            if not len(rub):
                continue                      # carried, not dragged: no rubbing
            joules = float(self._effort(cl)[1]) * vox_m
            if joules <= 0.0:
                continue
            half = joules * 0.5 / len(rub)
            self.E[tuple(rub.T)] += half
            under = rub.copy()
            under[:, 2] -= 1
            under = under[under[:, 2] >= 0]
            if len(under):
                self.E[tuple(under.T)] += joules * 0.5 / len(under)

    def _haul_cost(self, p, own):
        """How much of this body's strength its load is already using.

        Two costs, because they are two different things. A thing LIFTED is
        held up, and what it costs is its weight. A thing TRAILED along the
        floor is not held up at all, and what it costs is friction — the same
        `_effort` a shove asks about, so a heavy thing on a rough floor is dear
        and the same thing on ice is not."""
        cost = 0.0
        for cl in self._held_clusters(p):
            cl = np.asarray(cl, np.int64)
            if not len(cl):
                continue
            lift_N, drag_N = self._effort(cl)
            cost += drag_N if len(self._contact(cl, ignore=own)) else lift_N
        return cost

    def _held_clusters(self, p):
        """Everything in this body's hands, thing or person, as cell arrays.
        Whether the grip can CARRY any of it is _grip_holds' question."""
        out = []
        obj = self._held_cells(p)
        if obj is not None:
            out.append(obj)
        who = p.get("dragging")
        if who:
            q = next((r for r in self.persons if r["name"] == who), None)
            if q is not None and q["alive"] and not q["safe"]:
                comp, sl = self._person_cells(q)
                if comp is not None and comp.any():
                    qc = np.argwhere(comp)
                    qc[:, 0] += sl[0].start
                    qc[:, 1] += sl[1].start
                    out.append(qc)
        return out

    def _overbalanced(self, p, own):
        """Is this body's OWN weight past its OWN feet — and which way.

        The plainest statement of balance there is, and it had no home in the
        sim until a body could bend. `_pulled_over` asks the same question
        about a LOAD and answers it with a lean allowance, because a man
        carrying something shifts his weight back against it. There is no
        allowance here: this IS his weight, and it is already out there.

        The foot span is the tolerance and needs no other — a contact patch
        three voxels long is a 15 cm foot, which is about what an ankle buys
        you. Measured on the shear: the centre of mass crosses the toes at
        0.3 rad, which is 17 degrees, which is roughly as far as anyone leans
        forward without moving their hips back or taking a step."""
        # ONLY A BODY THAT IS STANDING UP CAN BE TAKEN OFF ITS FEET. Said as
        # geometry rather than as a flag: a standing body is TALLER than it is
        # long, and a body lying down is not. That is the whole difference, and
        # it is the difference between toppling and having already toppled.
        #
        # Measured without it: a man lying unconscious across a floor rests on
        # the nine voxels of him that touch it, with the centre of his mass
        # fourteen voxels away along his own length — which reads exactly like
        # a man leaning too far, and the sim threw him across the room. What
        # should happen to the parts of him that are over nothing is that they
        # SAG, and that is the support law's question, not this one.
        tall = float(own[:, 2].max() - own[:, 2].min())
        if tall <= max(float(own[:, 0].max() - own[:, 0].min()),
                       float(own[:, 1].max() - own[:, 1].min())):
            return None
        foot = self._contact(own)
        # AND A HAND ON SOMETHING THE WORLD IS HOLDING UP IS A FOOT. This is
        # the same sum `_pulled_over` does, from the other end: a held thing
        # NOT resting on anything hangs from you and drags you over, and a held
        # thing that IS resting on something is holding itself up — so it can
        # hold you too. A man with a hand on the rail leans out over the lip;
        # a man with nothing to hold leans as far as his own toes allow and
        # stops. Same body, same back, different world.
        #
        # The base reaches the HAND, which is conservative: really you can pull
        # on a rail and go well past it, and that wants the arm's tension
        # rather than a wider footprint (item 47).
        grips = []
        if p.get("held") or p.get("dragging"):
            fists = np.asarray(self._fists(p, own), np.float64)
            for cl in self._held_clusters(p):
                cl = np.asarray(cl, np.int64)
                if not len(cl) or not len(self._contact(cl)):
                    continue                  # it hangs from HIM, not he from it
                d2 = ((cl[None, :, :].astype(np.float64)
                       - fists[:, None, :]) ** 2).sum(-1)
                grips.append([int(round(c))
                              for c in fists[int(d2.min(1).argmin())]])
        if grips:
            g = np.array(grips, np.int64)
            foot = np.vstack([foot, g]) if len(foot) else g
        if not len(foot):
            return None
        kg = self.smass[tuple(own.T)]
        m = float(kg.sum())
        if m <= 0.0:
            return None
        out = [0.0, 0.0]
        for ax in (0, 1):
            c = float((own[:, ax] * kg).sum()) / m
            lo, hi = float(foot[:, ax].min()), float(foot[:, ax].max())
            if c > hi:
                out[ax] = c - hi
            elif c < lo:
                out[ax] = c - lo
        return None if out == [0.0, 0.0] else tuple(out)

    def _pulled_over(self, p, own):
        """Which way what this body is holding drags it off its own feet — or
        None, if the pair of them still balance.

        Newton's third, asked with arithmetic already here. A hand carries its
        load's weight THROUGH the body into the floor, so the weight that has
        to sit over this body's feet is its own PLUS whatever hangs from its
        fists. A man dangling from your arm hangs beyond your toes, and what
        takes you over the lip after him is not a sideways tug — a load
        hanging still pulls straight DOWN — but that the two of you together
        no longer balance on the ground you are stood on.

        Only what the grip actually HOLDS UP counts, which is why this needs no
        threshold of its own: a load too heavy to lift, or one lying on the
        floor, is resting on the world and pulls nobody anywhere. It is also
        why a stick costs nothing and a man costs everything — the same sum,
        and the mass decides."""
        # AND ONLY WHAT IS ACTUALLY HANGING. A grip carries what the world is
        # not already carrying: a crate still stood on the floor weighs on its
        # own footprint, not on the hand resting against it. Counted anyway, a
        # man was tipped over by a block he had merely taken hold of, one tick
        # after he touched it and while it was still sat on the rock.
        loads = [np.asarray(cl) for cl in self._held_clusters(p)
                 if self._grip_holds(p, cl) and not len(self._contact(cl))]
        if not loads:
            return None
        foot = self._contact(own, ignore=np.concatenate(loads))
        if not len(foot):
            return None
        kg = self.smass[tuple(own.T)]
        m_s = max(float(kg.sum()), 1e-9)
        sx = float((own[:, 0] * kg).sum()) / m_s
        sy = float((own[:, 1] * kg).sum()) / m_s
        # THE WEIGHT COMES IN AT THE FIST, not at the load's own middle. A
        # thing hanging still pulls straight DOWN along the arm holding it, so
        # where its own mass happens to sit says nothing about the holder's
        # balance — that is the WRIST's question, and _grip_holds asks it.
        # Summed at the load's centre instead, a body trailing on the floor
        # behind a hauler read as hanging metres out in front of his toes.
        fists = np.asarray(self._fists(p, own), np.float64)
        m_l, lx, ly = 0.0, 0.0, 0.0
        for cl in loads:
            m_l_here = float(self.smass[tuple(cl.T)].sum())
            if m_l_here <= 0.0:
                continue
            d2 = ((cl[None, :, :].astype(np.float64)
                   - fists[:, None, :]) ** 2).sum(-1)
            hand = fists[int(d2.min(1).argmin())]     # the hand it hangs from
            m_l += m_l_here
            lx += hand[0] * m_l_here
            ly += hand[1] * m_l_here
        if m_l <= 0.0:
            return None
        lx, ly = lx / m_l, ly / m_l
        # AND THE HOLDER CAN LEAN. A person shifts their own weight back
        # against what they carry, and how far back is bounded by how big they
        # are — their own half-width, read off the body, not a number typed in.
        # Without it the sum says nobody may hold anything at arm's length,
        # which is plainly false; with it, whether a load takes you over is
        # its weight and how far out it hangs against YOUR weight and YOUR
        # build, which is the real answer and is different for every pair.
        lean = 0.5 * self._span_xy(own)
        out = [0.0, 0.0]
        for ax, (lo, hi, l_, s_) in enumerate(
                ((int(foot[:, 0].min()), int(foot[:, 0].max()), lx, sx),
                 (int(foot[:, 1].min()), int(foot[:, 1].max()), ly, sy))):
            if l_ > hi:
                arm, back, sign = l_ - hi, (hi - s_) + lean, 1.0
            elif l_ < lo:
                arm, back, sign = lo - l_, (s_ - lo) + lean, -1.0
            else:
                continue                      # the load is over the feet
            if m_l * arm > m_s * max(back, 0.0):
                out[ax] = sign * arm
        return None if out == [0.0, 0.0] else tuple(out)

    def _take_up(self, p, own):
        """Bring a held thing to the HAND, if the arm can lift it. A liftable
        thing hangs from the fist — where a swing will find it, and where the
        grip (not the floor) carries it. Too heavy stays where it lies: still
        held, a grip is a constraint before it is a lift."""
        obj = self._held_cells(p)
        if obj is None:
            return
        arm = (p.get("held") or {}).get("arm") \
            or self._hand(p, own, toward=obj.mean(axis=0))
        if arm is None:
            return
        w_N, _ = self._effort(obj)
        if w_N > self._hand_strength(p, arm):    # the off hand is weaker, and
            return                               # that is what dominance MEANS
        hand = self._fist_of(p, arm, own)
        if hand is None:
            return
        hand = np.array([int(round(c)) for c in hand], np.int64)
        top = int(obj[:, 2].max())
        # HANG IT BESIDE THE FIST, NOT THROUGH THE CHEST. Centred ON the hand, a
        # load any wider than a hand reaches back into the body of whoever is
        # holding it, and the move is refused for the perfectly good reason that
        # a man is already standing there — measured, a 38 kg block a 4000 N man
        # could obviously lift stayed on the floor being dragged instead.
        # Outward is away from the body's own middle, which is the side the arm
        # is on.
        cx, cy = float(own[:, 0].mean()), float(own[:, 1].mean())
        ax = 0 if abs(hand[0] - cx) >= abs(hand[1] - cy) else 1
        away = 1 if (hand[ax] - (cx if ax == 0 else cy)) >= 0 else -1
        beside = [int(hand[0]), int(hand[1])]
        beside[ax] += away * (int(np.ptp(obj[:, ax])) // 2 + 1)
        d = (beside[0] - int(round(float(obj[:, 0].mean()))),
             beside[1] - int(round(float(obj[:, 1].mean()))),
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
        danger = p.get("danger")
        if limb == "legs":
            menu.append({"key": "stay", "tag": "stay", "verb": "stay"})
            # a CARRIED thing is part of the walker, to the planner as much as
            # to the legs: its column is not an obstacle to its own carrier,
            # or "straight on" vanishes from the menu the moment the fist
            # closes on a stick hanging in front of the chest
            obj = self._held_cells(p) if p.get("held") else None
            nav = np.concatenate([cells, obj]) if obj is not None \
                and not self._underfoot(obj) else cells
            fit, _start = self._fit_grid(nav, p)
            ground = self._underfoot(cells)
            places = self._places(p, cells, fit)
            leaps = self._leap_targets(p, cells, fit, self._leap_speed(cells)) \
                if ground else []
            routes = self._routes(cells, [pl["xy"] for pl in places]
                                  + [lp["xy"] for lp in leaps],
                                  p, fit=fit)
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
                # HOW BAD THE WAY THERE LOOKS, from what this body remembers
                # seeing. A place is as dangerous as the worst step on the way
                # to it — which is the right question, and a cheaper one than
                # asking about the destination, because the route is already
                # in hand. The sim states the fact; whether to mind it is the
                # policy's business, and a body with no taste at all still
                # gets the same menu.
                risk = 0.0
                if danger is not None:
                    risk = float(max(danger[int(x), int(y)] for x, y in route))
                menu.append({"key": f"go({pl['label']})", "tag": pl["tag"],
                             "verb": "go", "goal": pl["xy"], "route": route,
                             "away": pl["away"], "danger": round(risk, 3)})
        elif limb == "hands":
            held = p.get("dragging") or \
                ("the " + p["held"]["label"] if p.get("held") else None)
            menu.append({"key": "keep hold of " + held if held else "hands free",
                         "tag": "keep", "verb": "keep"})
            if held:
                menu.append({"key": "let go", "tag": "let_go", "verb": "let_go"})
            # WHICH ARM, named on the row. An arm already out is the one the
            # body would bring in; otherwise a FREE hand reaches, and which
            # free hand is `_hand`'s business.
            out_arm = next((a for a in sorted(self._limbs(p)) if "arm" in a
                            and np.abs(np.atleast_1d(
                                (p.get("drawn") or {}).get(a, 0.0))).max() > 1e-9),
                           None)
            arm = out_arm or self._hand(p, cells, free=True) \
                or self._hand(p, cells)
            if arm is not None:
                # REACHING IS AN ACT, not a way of drawing the body. An arm put
                # out is OUT: it occupies what it occupies, its hand is a good
                # deal further from the feet, and holding it there costs the
                # shoulder the limb's weight times the lever. So a body gets to
                # choose it, and gets to stop.
                # WHERE THE ARM IS, not what the body meant. These part
                # company whenever a reach is abandoned — the world said no and
                # the intention was dropped, but the flesh is still out there.
                # Read off the intention, the menu then offered a man with his
                # arm stuck half out the chance to "reach out" and never once
                # the chance to bring it down.
                out = out_arm is not None
                if out or not p.get("_no_reach") == (p["anchor"],
                                                     tuple(p.get("facing", (1.0, 0.0)))):
                    # AND A REACH IT HAS ALREADY FOUND IT CANNOT MAKE is not on
                    # the menu. A picked option must be one that can happen; a
                    # body pinned against a wall re-decided to reach every few
                    # ticks for ever, and every one of those picks went into the
                    # trace as a decision that changed nothing. It is forgotten
                    # the moment the body moves or turns, because then it is a
                    # different question.
                    menu.append({"key": "pull the arm back in" if out else "reach out",
                                 "tag": "pull_in" if out else "reach",
                                 "verb": "pull_in" if out else "reach",
                                 "arm": arm})
                # swinging at NOTHING is possible, just useless — the same way
                # shouting in an empty house is possible. What makes it legal
                # is having an arm, not having a target. A held thing swings
                # WITH the arm; that is most of what holding a tool is for
                menu.append({"key": "swing an arm" if not p.get("held") else
                             f"swing the {p['held']['label']}", "tag": "swing",
                             "verb": "swing",
                             # what is held swings with the hand holding it
                             "arm": (p.get("held") or {}).get("arm") or arm})
                if p.get("held"):
                    # THE SAME SWING, LET GO OF. Nothing new happens to the
                    # arm; the only difference is whether the hand opens.
                    menu.append({"key": f"throw the {p['held']['label']}",
                                 "tag": "throw", "verb": "throw",
                                 "arm": (p.get("held") or {}).get("arm") or arm})
                # AND A HAND THAT IS EMPTY CAN CLOSE ON SOMETHING PASSING.
                # A FREE hand, which is not the same as having no hands full: a
                # body has two, and one of them being busy is exactly the
                # situation a rescue is. Written as "only when holding nothing"
                # it meant a man with a hand on the rail could not catch the
                # friend going past him, which is the one moment it was for.
                #
                # Offered only for a thing genuinely within reach right now, so
                # a picked catch is a catch that can happen — whether the arm
                # can STOP it is the act's own question, and a hand that cannot
                # is a hand the thing goes past.
                spare = self._hand(p, cells, free=True)
                if spare is not None:
                    for fb in self._in_flight_near(p, cells):
                        lab = MATNAME.get(int(fb["mats"][0]), "thing")
                        menu.append({"key": f"catch the {lab}", "tag": "catch",
                                     "verb": "catch", "arm": spare, "body": fb,
                                     "away": 0.0})
                    for fb in self._falling_near(p, cells):
                        menu.append({"key": f"catch {fb['owner']}",
                                     "tag": "catch_who", "verb": "catch_who",
                                     "arm": spare, "body": fb, "away": 0.0})
            for q in self._within_reach(p, cells):
                if q["name"] == held:
                    continue
                menu.append({"key": f"take hold of {q['name']}", "tag": "hold",
                             "verb": "hold", "who": q["name"]})
            # A THING AND A PERSON ARE NOT THE SAME ROW. They were both tagged
            # "hold", so a character sheet saying `hands: hold` meant "hold
            # whatever is nearest" — and a rescuer who wandered within reach of
            # the fire picked up a BURNING STICK instead of the man he had come
            # for, then stood there holding it until the smoke took him. What
            # the sheet meant was a person.
            if not p.get("held"):
                for ob in self._objects_within_reach(p, cells):
                    menu.append({"key": f"take hold of the {ob['label']}",
                                 "tag": "take", "verb": "hold", "what": ob})
        elif limb == "eyes":
            # THE SWEEP IS THE NULL ACT. A body that decides nothing goes on
            # turning its head, which is what it did before eyes were a limb —
            # so nothing in the sim got worse the day they became one.
            held = (p.get("look") or (0, 0, -99))
            looking = self.tick - held[2] < WILL["look_for"]
            menu.append({"key": "go on looking about", "tag": "about",
                         "verb": "about"})
            fx, fy = p.get("facing", (1.0, 0.0))
            for name, (dx, dy) in (("behind", (-fx, -fy)),
                                   ("to the left", (-fy, fx)),
                                   ("to the right", (fy, -fx))):
                if looking and abs(held[0] - dx) < 1e-6 and abs(held[1] - dy) < 1e-6:
                    continue                  # already looking that way
                menu.append({"key": f"look {name}", "tag": f"look:{name}",
                             "verb": "look", "dir": (float(dx), float(dy))})
        elif limb == "waist":
            if not self._bones(p, "lean") or "lean" not in (p.get("chain") or {}):
                return []                     # nothing here bends
            # THE SAME SHAPE AS HANDS. The null act KEEPS what the body is
            # doing — a trunk takes many ticks to bend and the body must be
            # able to go on bending — and straightening up is an act of its
            # own. Made the null act "stand up straight", a body cancelled its
            # own lean on the very next decision and never bent at all.
            # DRAWN, not posed. `pose` is the motor command and it runs on
            # through every angle the lattice cannot draw; `drawn` is where the
            # flesh actually IS. They part company badly here, because a lean
            # has a long undrawable tail — measured, a body whose angle read
            # 45.8 degrees was bent 19.3. Ask the flesh.
            bent = abs(float(np.atleast_1d(
                (p.get("drawn") or {}).get("lean", 0.0))[0])) > 1e-9
            going = bool((p.get("reach") or {}).get("lean"))
            menu.append({"key": "keep leaning" if going else
                         ("stay bent" if bent else "stand as it is"),
                         "tag": "hold_pose", "verb": "hold_pose"})
            if going or bent:
                menu.append({"key": "straighten up", "tag": "upright",
                             "verb": "upright"})
            if not going:
                menu.append({"key": "lean out", "tag": "lean", "verb": "lean"})
        elif limb == "mouth":
            menu.append({"key": "say nothing", "tag": "quiet", "verb": "quiet"})
            for name, text in p.get("lines", LINES).items():
                if not text:
                    continue                  # this body has no such words
                if ANSWERS.get(name) not in (None, percept):
                    continue                  # nothing to answer
                menu.append({"key": f"say({name})", "tag": f"say:{name}",
                             "verb": "say", "line": name})
        self._offered = len(menu)             # what a full sweep found, before
        if len(menu) > MENU_CAP:              # attention is the scarce thing.
            want = (p.get("reflexes", REFLEXES).get(percept) or {}).get(limb) \
                if percept else None
            head, tail = menu[0], menu[1:]    # the null act never falls off
            keep = [o for o in tail if o["tag"] == want]
            rest = sorted((o for o in tail if o not in keep),
                          key=lambda o: o.get("away", 0.0))
            # ONE OF EACH KIND FIRST, then the nearest of what is left.
            #
            # Sorted by distance alone, the cap would drop every row of a whole
            # KIND — measured, it cut all four roam spots because each was
            # further off than a step, and the body then could not choose to
            # wander at all however much it wanted to. Recall was 81%: one
            # decision in five, the cap was hiding the answer.
            #
            # This is a model of attention rather than a trick for the policy's
            # benefit. What a body notices is categories before instances — that
            # there is a door, that there is somewhere unseen, that there is a
            # place it could go — and then the nearest of each. It keeps nothing
            # because the policy prefers it: salience is not preference, and a
            # fire is worth noticing whether you mean to run at it or from it.
            first, later, got = [], [], {o["tag"] for o in keep}
            for o in rest:
                (later if o["tag"] in got else first).append(o)
                got.add(o["tag"])
            menu = ([head] + keep + first + later)[:MENU_CAP]
        return menu

    def _full_menu(self, p, cells, percept, limb, chosen):
        """The menu WITHOUT the cap — what a body would have to choose from if
        attention were free. Only ever used to measure what the cap costs."""
        was, globals()["MENU_CAP"] = MENU_CAP, 10 ** 6
        try:
            return self._menu(p, cells, percept, limb, chosen)
        finally:
            globals()["MENU_CAP"] = was

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
               "ground_it_trusts": round(float(p["free"].mean()), 3)
                                   if p.get("free") is not None else 0.0,
               "others_in_earshot": heard}
        menus, picks, chosen, seen, would = {}, {}, {}, {}, {}
        for limb in LIMBS:
            menu = self._menu(p, cells, percept, limb, chosen)
            if not menu:
                continue
            i = self.policy.pick(dict(sit, limb=limb, chosen=dict(chosen)), menu)
            i = i if isinstance(i, int) and 0 <= i < len(menu) else 0
            opt = menu[i]
            menus[limb] = [o["key"] for o in menu]
            seen[limb] = self._offered        # and how many there were to see
            if self.recall_check:
                # WHAT ATTENTION COST, measured rather than hoped. Build the
                # menu the body would have had with attention free, ask the
                # same policy, and record what it would have done. Off by
                # default: it doubles the work of deciding, and it exists to
                # answer one question — does the cap ever hide the answer?
                full = self._full_menu(p, cells, percept, limb, chosen)
                j = self.policy.pick(dict(sit, limb=limb, chosen=dict(chosen)),
                                     full) if len(full) > len(menu) else i
                j = j if isinstance(j, int) and 0 <= j < len(full) else 0
                would[limb] = full[j]["key"] if len(full) > len(menu) \
                    else opt["key"]
            picks[limb] = opt
            chosen[limb] = opt["tag"]
        if not picks:
            return
        prog = " & ".join(picks[l]["key"] for l in LIMBS
                          if l in picks and not ACTS[picks[l]["verb"]]["null"])
        self.traces.append({"tick": self.tick, "who": p["name"],
                            "percept": percept, "situation": sit,
                            "menus": menus,
                            "offered": seen,  # what a full sweep found, so the
                                              # cost of the cap is on the record
                            "would": would,   # and what it would have picked
                                              # with attention free, if asked
                            "pick": {l: o["key"] for l, o in picks.items()},
                            "tags": dict(chosen), "program": prog or "stay",
                            "by": getattr(self.policy, "name", "?")})
        p["_decided"] = self.tick
        if percept is not None:
            p["emergency"] = True        # committed: stop weighing the ordinary
        hands = picks.get("hands")
        if hands is not None:            # HANDS. Holding is not a rescue
            if hands["verb"] in ("swing", "throw"):
                b = self._swing(p, hands.get("arm") or self._hand(p, cells),
                                toward=p.get("facing"),
                                back=(BODY["windup_rad"]
                                      if hands["verb"] == "throw" else 0.0))
                if b is not None and hands["verb"] == "throw":
                    b["throw"] = True
                p["events"].append(
                    f"t{self.tick}: {p['name']} "
                    + ("throws the " + p["held"]["label"]
                       if hands["verb"] == "throw" and p.get("held")
                       else "swings an arm"))
            elif hands["verb"] == "hold":  # no subsystem: a grip is REACH, and
                if "who" in hands:             # what it can then do is force —
                    p["dragging"] = hands["who"]   # the same _effort a shove
                    p["events"].append(f"t{self.tick}: {p['name']} takes hold "
                                       f"of {hands['who']}")   # uses, asked
                else:                              # again every tick
                    ob = hands["what"]
                    p["held"] = {"cell": ob["cell"], "mat": ob["mat"],
                                 "label": ob["label"],
                                 # WHICH HAND TOOK IT, so the other stays free
                                 "arm": self._hand(p, cells, toward=ob["cell"],
                                                   free=True)}
                    self._take_up(p, cells)    # to the fist, if the arm can
                    p["events"].append(f"t{self.tick}: {p['name']} takes up "
                                       f"the {ob['label']}")
            elif hands["verb"] == "reach":
                p.setdefault("reach", {})[
                    hands.get("arm") or self._hand(p, cells)] = float(np.pi / 2)
                p["events"].append(f"t{self.tick}: {p['name']} reaches out")
            elif hands["verb"] == "pull_in":
                (p.get("reach") or {}).pop(
                    hands.get("arm") or self._hand(p, cells), None)
                p["events"].append(f"t{self.tick}: {p['name']} lowers the arm")
            elif hands["verb"] == "catch_who":
                fb = hands.get("body")
                if fb is not None and fb in self.bodies:
                    who = fb.get("owner")
                    got = self._catch_person(p, fb, hands.get("arm")
                                             or self._hand(p, cells))
                    p["events"].append(
                        f"t{self.tick}: {p['name']} "
                        + (f"catches {who}" if got
                           else f"grabs at {who} and cannot hold them"))
            elif hands["verb"] == "catch":
                fb = hands.get("body")
                if fb is not None and fb in self.bodies:
                    got = self._catch(p, fb, hands.get("arm")
                                      or self._hand(p, cells))
                    p["events"].append(
                        f"t{self.tick}: {p['name']} "
                        + ("catches the " + (p.get("held") or {}).get("label", "thing")
                           if got else "cannot hold what is coming, and it goes past"))
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
        eyes = picks.get("eyes")
        if eyes is not None and eyes["verb"] == "look":
            p["look"] = (eyes["dir"][0], eyes["dir"][1], self.tick)
            p["events"].append(
                f"t{self.tick}: {p['name']} looks {eyes['key'][5:]}")
        waist = picks.get("waist")
        if waist is not None and waist["verb"] == "lean":
            # AS FAR AS IT CAN, and the world decides how far. `_law_pose`
            # walks the angle up at the speed a trunk can manage and stops
            # where the back or the balance says stop — the body asks to lean
            # out, it does not ask for an angle it has no way of knowing.
            p.setdefault("reach", {})["lean"] = BODY["lean_max"]
            p["events"].append(f"t{self.tick}: {p['name']} leans out")
        elif waist is not None and waist["verb"] == "upright":
            (p.get("reach") or {}).pop("lean", None)
            if abs(float(np.atleast_1d(
                    (p.get("drawn") or {}).get("lean", 0.0))[0])) > 1e-9:
                p["events"].append(f"t{self.tick}: {p['name']} straightens up")

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
                self._decide_falling(p)      # in the air, and not out of choices
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
            # UNLESS IT HAS DECIDED TO LOOK SOMEWHERE. A chosen direction wins
            # over the sweep for as long as the body holds it — which is what
            # makes looking an ACT rather than a thing that happens to a head.
            want_look = p.get("look")
            if want_look is not None and not p.get("emergency") \
                    and self.tick - want_look[2] < WILL["look_for"]:
                gaze = (want_look[0], want_look[1])
            p["gaze"] = gaze                  # what the eyes are pointed at
            if self.tick % 4 == 0:            # look where the eyes point, and
                self._look_around(p, eye, cells)   # REMEMBER it
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
                if percept is None and p.get("skin", AMBIENT) > BODY["feel_T"]:
                    # BEING COOKED IS NOTICING, and it does not go through the
                    # eyes. Sight needs a clear line from the head, in the
                    # direction the head is turned — so a body STANDING ON a
                    # fire did not see it, because the fire was under its own
                    # feet and behind its own legs. Measured the day stepping up
                    # was built: a man idly wandered onto a burning crib and
                    # suffocated there over two hundred ticks without once
                    # perceiving the thing he was stood in.
                    percept = "scorched"
                if percept is None and self._falling_near(p, cells):
                    percept = "sees_falling"     # somebody is going past you
                if percept is None and self._in_flight_near(p, cells):
                    # SOMETHING IS COMING. Needs no sight cone and no distance
                    # rule: it is already within a hand's reach, which is the
                    # only range at which noticing is any use to you.
                    percept = "sees_thrown"
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
            # WHAT YOU DRAG, YOU PAY FOR. A man hauling an unconscious body
            # walked at exactly the pace of a man carrying nothing — the force
            # arithmetic said the haul was legal and then charged him nothing
            # for it, which made rescuing someone free and dragging a crate the
            # same act as strolling.
            #
            # A body has only so much to put out. What the load takes, the legs
            # do not get, and pace goes with what is left: hauling nothing is
            # the pace it always was, and at the very limit of what a man can
            # shift he barely moves — which is what being at your limit IS. The
            # floor of 5% is not a fudge for the model, it is a fence against
            # dividing by nothing.
            haul = self._haul_cost(p, cells)
            full = p.get("strength_N", BODY["strength_N"])
            pace = WILL["walk_every"] * full / max(full - haul, full * 0.05)
            # STARTS ONE SHORT, so that the very first tick completes a full
            # stride and the cadence of a body carrying nothing is exactly the
            # cadence it always had — 0, 3, 6, 9 and not 0, 2, 5, 8. That is
            # the property that made it safe to replace the modulo at all, and
            # being one out shifted every unladen body by a voxel.
            p["_stride"] = p.get("_stride", pace - 1.0) + 1.0
            if p["_stride"] < pace:
                continue
            p["_stride"] -= pace
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
                p["_path"] = self._plan_path(walk_cells, ex, p)
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
                    if haul > 0.0:
                        self._rub(p, cells)    # friction becomes heat, and heat
                                               # is what combustion reads
                    break
            else:
                p["_path"] = None                    # blocked mid-route: replan
                # AND LEARN IT. A body that remembers a way as clear will plan
                # the same way again the moment it replans, and jam against the
                # thing in it forever. Walking into something is a percept: you
                # find out the corridor is shut by shutting your shin in it.
                free = p.get("free")
                if free is not None:
                    for st in ((sx, 0), (0, sy)):
                        if st == (0, 0):
                            continue
                        bx = int(round(ax_)) + st[0]
                        by = int(round(ay_)) + st[1]
                        if 0 <= bx < free.shape[0] and 0 <= by < free.shape[1]:
                            free[bx, by] = False

    def _decide_falling(self, p):
        """A body in the air is not a body with nothing to decide.

        It has no cells on the lattice while it flies, which is why the will
        layer used to skip it entirely — and that quietly made falling the one
        thing in the sim nobody could do anything about. It cannot walk, and
        there is nothing under it to push against, so the menu is SHORT. But
        the one choice still open is how to meet the ground, and that is the
        difference between getting up and not.

        The same seam as every other decision: a menu of what is really
        possible, a policy that picks an index, and the whole menu written to
        the trace beside the pick."""
        if not (p["alive"] and p["awake"]) or p["safe"]:
            return
        b = next((x for x in self.bodies if x.get("owner") == p["name"]
                  and not x.get("part")), None)
        if b is None:
            return
        if self.tick - p.get("_fell", -99) < WILL["decide_every"]:
            return
        p["_fell"] = self.tick
        menu = [{"key": "stay as you are", "tag": "stay", "verb": "stay"}]
        if not p.get("rolling"):
            menu.append({"key": "roll on landing", "tag": "roll", "verb": "roll"})
        sit = {"tick": self.tick, "who": p["name"], "percept": "falling",
               "reflexes": p.get("reflexes", REFLEXES),
               "smoke": round(float(p["smoke"]), 3),
               "blood_o2": round(float(p["blood_o2"]), 3),
               "burn": round(float(p["burn"]), 3),
               "hurt": round(float(p["hurt"]), 3),
               "holding": p.get("dragging"),
               "falling": True,
               "knows_a_way_out": False, "seen_of_the_world": 0.0,
               "others_in_earshot": 0}
        i = self.policy.pick(dict(sit, limb="legs", chosen={}), menu)
        i = i if isinstance(i, int) and 0 <= i < len(menu) else 0
        opt = menu[i]
        self.traces.append({"tick": self.tick, "who": p["name"],
                            "percept": "falling", "situation": sit,
                            "menus": {"legs": [o["key"] for o in menu]},
                            "pick": {"legs": opt["key"]},
                            "tags": {"legs": opt["tag"]},
                            "program": opt["key"] if opt["tag"] != "stay" else "stay",
                            "by": getattr(self.policy, "name", "?")})
        if opt["verb"] == "roll":
            p["rolling"] = True
            p["events"].append(f"t{self.tick}: {p['name']} tucks to roll")

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

    def _fit_grid(self, cells, p=None):
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
        # THE FLOOR AS THIS BODY LAST SAW IT, not as it is. A mind with no
        # belief yet (or no mind at all — a bare route query) reads the world,
        # which is the honest default for a question nobody is asking.
        free = p.get("free") if p is not None else None
        walk_ok = self._walkable(cells) if free is None else free.copy()
        if p is not None:
            walk_ok &= p["known"]     # you cannot plan a route through rooms
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

    def _routes(self, cells, targets, p=None, fit=None):
        """Every target's route, from ONE breadth-first sweep.

        A menu prices a dozen places at once, and asking the planner a dozen
        separate questions re-walks the same room a dozen times. One sweep
        records where every column was reached FROM, and each route is read
        back out of it — cheaper than the three separate searches this replaced,
        for four times the places."""
        from collections import deque
        if fit is None:
            fit, start = self._fit_grid(cells, p)
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

    def _plan_path(self, cells, ex, p=None):
        """Breadth-first route from where this body is to one goal.
        Deterministic, 4-connected, replanned when the world changes underfoot."""
        return self._routes(cells, [ex], p)[(int(ex[0]), int(ex[1]))]

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
        1/2 m v² on landing.

        AND A BODY IS NOT ONLY ITS OWN WEIGHT. What it holds up rides on the
        same two feet, so a man still firmly stood on rock can be taken over
        by the one hanging from his arm — see _pulled_over. That is the other
        half of a grip being an edge in the support graph: the load hangs from
        the holder, and the holder answers for it."""
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
            over, why = None, "loses their footing"
            if self._underfoot(cells):
                # STOOD ON SOMETHING — but a body is not only its own weight.
                # What it holds up rides on the same two feet, and a man
                # dangling from an arm hangs well beyond the toes.
                over = self._pulled_over(p, cells)
                why = "is pulled off their feet by what they are holding"
                if over is None:
                    # AND A BODY CAN OVERBALANCE ALL BY ITSELF. Balance was
                    # only ever asked about what a man was CARRYING, so a man
                    # carrying nothing could not fall over however he stood —
                    # which nobody noticed while a body was one rigid block,
                    # because a block that stands up straight has its weight
                    # over its feet by construction. Give it a waist and the
                    # hole opens: measured, a man bent 46 degrees with his head
                    # fourteen voxels past his toes stood there indefinitely.
                    over = self._overbalanced(p, cells)
                    if over is None:
                        continue
                    why = "overbalances"
                # AND THE ARM COMES DOWN — all the way, at once. You cannot
                # keep a thing at arm's length while it is taking you over: the
                # reach is the first thing to go, which shortens the lever,
                # which is why a man goes over ONCE instead of being thrown off
                # his feet again on every tick he spends back on them.
                #
                # ALL THE WAY, and not a tick at a time. Holding a limb out is
                # work a standing body does, and a body going over has stopped
                # doing it — the arm falls, it does not unwind at a shoulder's
                # pace. Dropping only the INTENTION left `_law_pose` to walk the
                # angle home, and since the flesh waits at every angle the
                # lattice cannot draw, the man landed with the weight still out
                # and went over a second time, and a third.
                p["reach"] = {}
                for limb in list(p.get("pose") or {}):
                    if self._repose(p, limb, 0.0) == "moved":
                        (p["pose"]).pop(limb, None)
                        (p.get("drawn") or {}).pop(limb, None)
            else:
                h = holders.get(p["name"])
                if h is not None and self._grip_holds(h, cells):
                    continue                  # he HANGS from the grip: a hand
                                              # that can lift him can hold him
                foot = self._contact(cells)   # ...but is anything still touching?
                if len(foot):
                    kg = self.smass[tuple(cells.T)]
                    tot = max(float(kg.sum()), 1e-9)
                    over = (float((cells[:, 0] * kg).sum()) / tot
                            - float(foot[:, 0].mean()),
                            float((cells[:, 1] * kg).sum()) / tot
                            - float(foot[:, 1].mean()))
            vel = [0.0, 0.0, 0.0]
            if over is not None:
                ox, oy = over
                axis, o = (0, ox) if abs(ox) >= abs(oy) else (1, oy)
                vel[axis] = float(np.sign(o) * np.sqrt(
                    2.0 * GRAVITY * abs(o) * 0.1 * self.scale))
            self._launch(cells, vel, owner=p["name"])
            p["events"].append(f"t{self.tick}: {p['name']} {why}")

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

    def _busy(self, *fields):
        """Is there anything here at all for a law to move?

        A whole-array reduce costs about a tenth of a millisecond on a 300k
        world; the laws it stands in front of cost tens. Each question is asked
        IMMEDIATELY before the law it gates, never hoisted to the top of the
        tick — a fire that lights this tick makes its smoke this tick, and a
        stale answer would leave that smoke sitting still for a tick."""
        if not self.skip_quiet:
            return True
        return any(bool(f.any()) for f in fields)

    def _mixing(self):
        """Has the air anything to even out? Oxygen moves down a gradient, and
        a flat field has none — every pair cancels. The open sky is the other
        way in: it refills the top, which matters only if the world is below
        what fresh air holds."""
        if not self.skip_quiet:
            return True
        lo, hi = float(self.o2.min()), float(self.o2.max())
        return lo != hi or (self.open_sky and hi < O2_PER_L * self.vox_l)

    def _pushed(self):
        """Is any gas moving, being made to move, or about to be?

        Pressure and velocity decay toward nothing and stay there, and the
        gradient of a flat field is zero, so a still world carries nothing. But
        pressure is not only injected by `p_add`: WARM AIR MAKES ITS OWN WIND,
        and the law reads `T - AMBIENT` directly, because a plume is buoyancy
        before it is anything else.

        Found by the tick-by-tick guard, which is the whole reason to have one.
        Gated on the pressure fields alone, the first wisp of smoke a fire ever
        made came out wrong by 1.7e-5 on the very tick it appeared — because the
        world had been skipping its buoyant wind since tick zero, and the gate
        looked perfectly reasonable until something measured it."""
        if not self.skip_quiet:
            return True
        if bool(self.E.any()) or bool(self.p_add.any()):
            return True
        if self.pcell is None:
            return False
        return bool(self.pcell.any()) or any(bool(v.any()) for v in self.vcell)

    def step(self):
        if self.o2 is None:                      # first step: fresh air fills every pore
            self.o2 = np.full(self.shape, O2_PER_L * self.vox_l, np.float32)
        # DO NOT COMPUTE WHERE NOTHING IS HAPPENING. Measured before any of
        # this: a room with nothing lit in it cost 98 ms a tick and a room on
        # fire cost 100 — the whole world was being swept for heat that was not
        # there, smoke that did not exist and puddles nobody had poured.
        #
        # Every gate below is PROVABLE rather than plausible, which is the only
        # kind worth having: a law is passed over exactly when the thing it
        # moves is absent, so running it could not have changed one cell.
        # `self.E` is energy ABOVE ambient, so all-zero means every voxel sits
        # at 20 °C — no gradient to conduct, no dT to rise, nothing hot enough
        # to light, boil or melt, and no face carrying a shock. Set
        # `skip_quiet = False` to run everything over everything, which is what
        # `test_ACTIVE_REGIONS_change_NOTHING` compares every scene against.
        self._law_bodies()
        if self._busy(self.fvol) or self.drops:
            self._law_flow()                     # (drops in flight are fluid too)
            self._law_stir()
            self._law_head()
        self._law_footing()                      # before support: a body with no
        self._law_support()                      # footing must leave as a BODY
        self._law_torque()
        hot = self._hot_window()                 # ...and WHERE the heat is
        if hot is not None:
            self._law_conduct(hot)
            self._law_crack()                    # thermal shock reads what conduction wrote
            risen = self._hot_window()           # conduction just MOVED heat, so
            if risen is not None:                # ask again before moving it more
                self._law_rise(risen)
        self._air_regions()                      # (re)label airspaces if walls/gaps changed
        if self._busy(self.E):
            self._law_burn()
            self._law_boil()
            self._law_melt()
        if self._busy(self.fvol):
            self._law_acid()
        sooty = self._window(self.smoke > SMOKE_STILL,   # asked AFTER burning,
                             self._SMOKE_REACH)          # which is where it
                                                         # comes from
        if sooty is not None:
            self._law_smoke(sooty)
        self._air_regions()                      # acid may have bored new airspace
        if self._busy(self.E) or self._mixing():
            self._law_o2()
        if self._pushed():
            self._law_air()                      # the pneumatic field: pressure, wind
        self._law_life()                         # the slow chemistry of anyone alive
        self._law_will()                         # and their reflexes
        self._law_pose()                         # and where their limbs ended up
        if self.thermostats or self._busy(self.E):
            if self.thermostats:                    # dev heater blocks hold their
                C = self.heat_capacity()            # set temperature against all
                for (tx, ty, tz, tT) in self.thermostats:   # losses — heat without fire
                    self.E[tx, ty, tz] = (tT - AMBIENT) * C[tx, ty, tz]
            # AND WHAT THE WORLD SHEDS, IT SAYS IT SHED. These three lines are
            # the lattice's boundary with everywhere else, and they are right:
            # a voxel really does radiate into a colder universe, and that is
            # what stops a flame climbing for ever. But they were the only
            # place in the sim where a conserved quantity changed and nothing
            # wrote it down — so "energy is conserved" was not a statement
            # anyone could CHECK, the way mass became checkable when what left
            # the world started being counted.
            #
            # Measured while chasing friction heat: 1000 J left completely
            # alone in a closed room is 779 J sixty ticks later. Nothing was
            # wrong; nothing could say so either.
            was = float(self.E.sum())
            Tk = self.T() + 273.0                   # every voxel radiates to the wider,
            self.E -= RAD * self.scale ** 2 * (Tk ** 4 - 293.0 ** 4) / 3.0   # cooler world — trivial when warm,
            self.E *= (1.0 - LEAK)                  # fierce when white-hot (caps flame temps)
            self.E = np.maximum(self.E, -AMBIENT * self.heat_capacity())   # nothing below 0 °C here
            self.shed += was - float(self.E.sum())  # net: the floor above is a
                                                    # SOURCE, and nets in here
        self.tick += 1

    # ── a world as a value ───────────────────────────────────────────────────
    _NOT_STATE = ("policy",)          # a mind is not part of the world

    def snapshot(self):
        """Everything this world IS, as a value you can put back later.

        The state is a handful of arrays and a few plain lists, so a copy is
        cheap and — the point — EXACT: restore one and the next tick is the
        tick that would have followed. Three different things want it.

        A guard that compares two runs only at the END can say a divergence
        happened; one that can start both from the SAME state says which change
        caused it, with the build of the world removed as a variable. That is
        what makes it safe to rewrite how the laws are applied at all.

        A table wants to rewind and try the other thing. And a harvest of
        decisions is only honest if the run behind it can be played again.

        The policy is left out on purpose: it is a mind, not a world, and it may
        one day be a model whose weights dwarf the lattice."""
        return copy.deepcopy({k: v for k, v in self.__dict__.items()
                              if k not in self._NOT_STATE})

    def restore(self, snap):
        """Put a snapshot back, and keep whatever mind is in place now.

        Copied on the way in as well as the way out, so one snapshot can be
        restored as many times as you like — which is the whole use of it."""
        for k in [k for k in self.__dict__ if k not in self._NOT_STATE]:
            del self.__dict__[k]
        self.__dict__.update(copy.deepcopy(snap))

    # ── totals (the conservation the tests watch) ────────────────────────────
    def total_fluid(self, f):
        """Every millilitre of that fluid there is: in cells, in parcels still
        in flight, in bodies mid-topple, and past the edge of the world."""
        return float(self.fvol[self.fl == f].sum()) \
            + sum(d[6] for d in self.drops if d[7] == f) \
            + sum(float(b["fvols"][b["fls"] == f].sum()) for b in self.bodies) \
            + self.gone.get(("fluid", int(f)), 0.0)

    def total_mass(self, m):
        """Every gram of that material there is.

        ONE ANSWER, COUNTING EVERYWHERE IT CAN BE. Matter in this sim lives in
        three places — cells, bodies that have left the grid to topple or fly,
        and the tally of what went past the edge — and a total that reads only
        the first is not a total. It quietly said a swung axe had ceased to
        exist for as long as it was in the air.

        This had gone wrong in the small: `total_wood` counted the tally and
        nothing else did, which is worse than none of them counting it, because
        a conservation check that is right for one material and wrong for the
        rest fails at whichever moment is least convenient."""
        return (float(self.smass[self.mat == m].sum())
                + sum(float(b["masses"][b["mats"] == m].sum())
                      for b in self.bodies)
                + self.gone.get(int(m), 0.0))

    def total_wood(self):
        return self.total_mass(WOOD)

    def total_energy(self):
        """Every joule this world has had: what is in it, what is riding on
        bodies in flight, what went past the edge, and what it has radiated
        away. The only form in which "energy is conserved" is a question you
        can put to the sim and get a straight answer to."""
        return (float(self.E.sum())
                + sum(float(b["Es"].sum()) for b in self.bodies)
                + self.gone.get("E", 0.0) + self.shed)

    def total_lost(self):
        """Every gram that has gone past the edge of the world. Zero in any
        scene nobody is throwing things out of, and the only honest way to
        write a conservation check in one that is."""
        return sum(v for k, v in self.gone.items()
                   if isinstance(k, int) and k != AIR)
