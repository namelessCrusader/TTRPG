"""The decision contract: limbs, acts, reflexes, lines, attention cap, and the table policy."""
import zlib

import numpy as np

from .materials import SHOUT_DB, TICK_S

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
        # A FLAME IS NOTICED BY HOW BIG IT LOOKS, not by being inside a
        # radius. It was a fixed 4 m: a man 5 m from a burning bed, looking
        # straight at it, never noticed it at all. Now a flame in view counts
        # by the solid angle it fills (its area over the distance squared, in
        # steradians), at full weight in the middle of the view and at
        # `periphery` at the edge, and it is noticed once that passes this.
        # One 5 cm flame is noticed straight ahead to about 7 m, at the edge
        # of the view to about 4 m; a burning bench, across any room. No
        # light term yet (row P5): this is a lit room.
        "flame_sr": 5e-5,
        "periphery": 0.3,        # what the edge of the view is worth to it:
                                 # poor at detail, good at flicker and motion
        "run_see_m": 6.0,        # someone bolting past this close is a warning
        "runner_mps": 2.0,       # and BOLTING is a speed you can SEE: about
                                 # where people change from a walk to a run
                                 # (Hreljac 1993, ref 175, row B4). It was the
                                 # other body's `fleeing` flag, which read
                                 # another mind (row P20).
        # EYES LOOK WHERE THE FACE POINTS, in two zones (rows P1, P2). The
        # whole field is about 200 degrees, and anything in it can be
        # DETECTED: that something is there, that it moves, that it burns.
        # Only the middle 60 degrees IDENTIFIES: who it is, which thing it
        # is. The whole field was 120 degrees, and everything in it was seen
        # in full.
        "fov_deg": 200.0,
        "focus_deg": 60.0,
        # AND CLOSE TO, the edge of the view tells things apart too. Acuity
        # is a sixteenth of the centre's at 30 degrees off (row P2), so the
        # edge makes out a face at about a sixteenth of the distance the
        # centre does: faces are told apart well to 15 m and more (row P6),
        # so to about 1.5 m at the edge.
        "know_m": 1.5,
        "scan_every": 25,        # ticks per step of the gaze sweep
        "look_every": 4,         # ticks between two looks round (the layout,
                                 # the danger map, who and which fires)
        # HOW LONG A SEEN FIRE IS REMEMBERED: half its weight after this long
        # out of sight. A body that never forgets treats a fire it saw an hour
        # ago as a fire, and one that forgets at once has no memory at all.
        # Forgetting is a MODEL, not a leak — see item 14. The factor per look
        # (`forget`) is derived from this and the look period below; it was
        # typed in as 0.985, which is a half-life of 4.6 s against the "about
        # a minute" it was meant to be (row P21).
        "forget_half_s": 60.0,
        "forget": None,          # filled in below
        "shout_hear_m": 14.0,    # a shout carries this far through open air
        "wall_cost_m": 4.0,      # each solid voxel in the way eats this much
        # HOW FAST LEGS GO, in metres a second, and NOT in ticks a voxel. The
        # pace was "one voxel every 3 ticks", which is a speed only once you
        # say how big a voxel is: 0.67 m/s at 5 cm (half a real walk), 1.33
        # m/s at 10 cm, and slower again on a diagonal. People do not walk
        # faster in a coarser world. `_pace` turns these into ticks a voxel.
        # A comfortable walk is 1.10-1.65 m/s (Bohannon & Andrews 2011,
        # refs 172-173 of docs/realism-references.md, row B3).
        "walk_mps": 1.4,
        # A RUN, for a body that is fleeing. People change from a walk to a
        # run at about 2.06 m/s (Hreljac 1993, ref 175, row B4), so anything
        # slower is still a walk however frightened. 2.5 m/s is an easy run,
        # not a sprint (elite top speed is 12.4 m/s): ordinary running speeds
        # have no checked source yet, so this stays near the change-over.
        "run_mps": 2.5,
        "step_m": 2.0,           # how far "straight on" and "to the left" mean.
                                 # A body that knows nowhere by name can still
                                 # say which WAY it wants to go, and that is a
                                 # real intention, not a step: the legs still
                                 # do the walking
        "arrive_m": 0.12,        # close enough to a goal to have got there.
        "give_up": 80,           # ticks of getting no nearer before a body
                                 # stops trying to get somewhere. 2 s, about
                                 # 2.8 m at a walk.
        "baulk_for": 400,        # and how long it remembers that it could not,
                                 # so the menu does not hand the same
                                 # unreachable spot straight back.
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
        # HOW LONG A THING STAYS NEWS, under the "onset" attention law (3.5).
        # A body decides again when it perceives something it has not
        # perceived for this many ticks (a second at the usual tick): a fire
        # it has been watching all along is not news, the same fire seen
        # again after a minute away is. The same wait `perceive.LOOK_AGAIN`
        # gives a standing sight. Not read under "latch".
        "attend_gap": 40,
        # HOW LONG A BODY STAYS ALERT after something alarmed it, or after it
        # struck out itself: for this long it answers the next thing at the
        # speed of a body that is ready for it (`REACT`, "ready_s").
        "alert_s": 30.0,
        "decide_every": 10}      # filled in from `react_s` below, once TICK_S
                                 # (imported above, from materials.py) is in
                                 # hand. Ten is what 0.25 s comes to at the
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
            "sees_falling": {"hands": "catch"},
            # a noise is a question, not an alarm: the eyes go to it
            "hears_noise":  {"eyes": "look:noise"}}
# Percepts that turn the head and nothing else: the body looks, and goes on
# weighing the ordinary — it is not committed to anything.
ORIENTING = frozenset({"hears_noise"})
# HOW LONG BETWEEN PERCEIVING AND ACTING (rows P12-P16). A percept that would
# start a decision starts a DELAY first, counted from the moment of perception,
# and every person has their own: a sheet value of that name, or one drawn
# from the band for that person (`sheet`). So two people do not react on the
# same tick, and nobody reacts on the tick they perceive.
#
# - ready_s: a body that is ready for it (already fighting, already alarmed,
#   or `ready` on its sheet) answers in a simple reaction time, ~0.19 s to a
#   light and 0.16 s to a sound (Hick, ruler drop 0.15-0.30 s). A bare noise
#   is answered at this speed too: turning to look is the simplest act there
#   is (a saccade to the unexpected starts after ~0.2 s, row P14).
# - surprise_s: an ORGANISED act after a surprise. Driving data: 0.7 s
#   expected, 1.25 s unexpected, 1.5 s surprise, foot movement included
#   (row P13). A game band borrowed from it.
# - flinch_s: and before that, a surprised body FLINCHES at a thing coming at
#   it: the startle cascade turns the head in 60-120 ms and brings the arms up
#   in 125-195 ms (row P13). Here the head: the eyes go to it.
# - premove_s: and before LEAVING for a fire or an alarm, people take time to
#   look, gather and go to others (row P15). Real medians are minutes (home
#   drills 2:49, WTC 3-4 min); the Station nightclub crowd moved 24-30 s after
#   seeing the fire (row P16). The game band is 5-30 s from noticing to the
#   legs moving off (test P-T1 accepts 5-60 s), drawn on a log scale, so the
#   middle of it is about 12 s and a few people take much longer.
REACT = {"ready_s": (0.15, 0.30), "surprise_s": (1.0, 1.5),
         "flinch_s": (0.08, 0.15), "premove_s": (5.0, 30.0)}
# Never leaving a fire sooner than this after noticing it, however a sheet
# says otherwise (P-T1: never under 0.25 s).
PREMOVE_MIN_S = 0.25
# The percepts that start a PRE-MOVEMENT before the legs go: cues of a danger
# somewhere else, which a person stops to make sense of. Being burned or
# choked is not one of them: that is answered at once.
PREMOVE_ON = frozenset({"sees_fire", "hears_alarm", "sees_runner"})
# The percepts a surprised body flinches at first: a thing coming at it.
FLINCH_ON = frozenset({"sees_thrown", "sees_falling"})
# The percepts that cut a wait short: what is happening to the body itself,
# or coming at it. A body still making up its mind about a shout answers a
# flame on its own skin at once (at the speed of a ready body, as it is
# alarmed already). The rank orders what replaces what: a noise is replaced
# by any alarm, an alarm by any of these, and nothing by its equal.
URGENT = frozenset({"scorched", "chokes", "sees_thrown", "sees_falling"})
# WHAT A BODY DOES WHILE IT MAKES SENSE OF AN ALARM (the pre-movement, row
# P15), limb by limb, for the table: the eyes go to the cue, the hands take up
# what it holds dear (`dear` on its sheet: a thing's eid), the legs stay. The
# mouth does what the percept's own row says (a warning given is a
# pre-movement act too: "told by others" is how a fifth of people first hear
# of a fire, row P10). Going to someone it knows is not built yet.
PREMOVE = {"eyes": ("look:cue",), "hands": ("take:dear",), "legs": ("stay",)}
# And the words a body has for where to look, by what it noticed.
CUES = {"sees_fire": "the fire", "scorched": "the heat",
        "hears_alarm": "the shout", "sees_runner": "the one running",
        "sees_thrown": "what is coming", "sees_falling": "the one falling",
        "chokes": "the smoke"}


def sheet(p, key):
    """A person's own value for one of the `REACT` times, in seconds: the
    sheet's if it has one, else drawn from the band for THIS person.

    Drawn, not random: the sim has no random number generator (2.4), so the
    draw is a hash of the key, the pid and the name, the same every run, and
    different for two people even of the same name. `premove_s` is drawn on
    a log scale (its band spans six times); the others evenly."""
    v = p.get(key)
    if v is not None:
        return float(v)
    lo, hi = REACT[key]
    u = (zlib.crc32(f"{key}|{p.get('pid')}|{p.get('name')}".encode())
         & 0xFFFFFFFF) / 2.0 ** 32
    if key == "premove_s":
        return float(lo * (hi / lo) ** u)
    return float(lo + (hi - lo) * u)


def rank(percept):
    """What replaces what while a body waits to answer (`URGENT`): 0 a noise,
    1 any other percept, 2 an urgent one."""
    return 0 if percept in ORIENTING else 2 if percept in URGENT else 1


# LINES — the words a body has. A character sheet REPLACES this wholesale, so
# a character who only ever calls a warning has no answer to give.
LINES = {"fire":   "Fire! Fire! Get out!",
         "coming": "I hear you! I'm coming!"}
# THE VOICE has a loudness, as the law has: a spoken line is a sound of so many
# dB at a metre from the mouth, and the sound law (senses.py `_level`) decides
# who hears it. These are the levels a mouth is offered, quietest first; the
# law takes any number. Speech levels at 1 m, from the usual tables of vocal
# effort (ANSI S3.5 and the speech-communication texts): a whisper 30-40, a
# quiet voice about 50, a normal voice about 60, a raised voice about 75, a
# shout 85-90 (SHOUT_DB, the one number the sound law was first pinned to).
# The whisper is the top of its band, 40: words are made out only above
# HEAR_DB (40 dB at the ear, materials.py), so a whisper reaches a metre and
# not a step beyond, which is what a whisper is for; a quiet voice reaches
# about 3 m, a normal voice about 10 m, a raised voice about 55 m.
VOICE = (("whisper", 40.0), ("say quietly", 50.0), ("say", 60.0),
         ("call out", 75.0), ("shout", SHOUT_DB))
VOICE_DB = dict(VOICE)
# What a speaker does of a line when nobody tells it otherwise: an alarm is
# shouted, a reply to it is called, anything else is said. The menu offers
# the line at every level; this only says which row the reflex table takes.
LINE_VOICE = {"fire": "shout", "coming": "call out"}
# The word the event of a line is written with, by its level.
VOICE_VERB = {"whisper": "whispers", "say quietly": "says quietly",
              "say": "says", "call out": "calls out", "shout": "shouts"}
# THE LOMBARD REFLEX: a speaker raises the voice without deciding to when the
# noise around it rises. It starts at about 45 dB of background and lifts the
# voice about half a dB for each dB of noise above that (Lombard 1911; Lane
# and Tranel 1971: 0.3-0.7 dB/dB), up to what a throat can do (about 100 dB at
# a metre). The reflex adds to the level the speaker chose; the sound law then
# hears the louder voice. These numbers are data a reader compares against.
LOMBARD = {"from_db": 45.0, "slope": 0.5, "max_db": 100.0}
# Which lines are REPLIES, and to what. "I'm coming" presupposes something to
# come to, so a body that heard nothing is never offered those words. This is
# a fact about what the sentence DOES rather than about its wording, which is
# why it is kept apart from the wording: rewriting the line on a character
# sheet must not be able to lose it.
ANSWERS = {"coming": "hears_alarm"}
# AND WHAT HEARING ONE DOES TO YOU. A line is an act, and an act has an effect
# — which is not the same as the effect of having made a NOISE. Without this,
# hearing was "somebody nearby shouted within the last thirty ticks", the words
# went unread, and the reply to an alarm WAS an alarm: A shouts fire, B answers
# "I'm coming", C hears B and flees a fire it has no evidence of, answering as
# it goes. Two lines were enough to build a cascade out of.
#
# A line absent from this map is audible and means nothing — which is what a
# reply should do to a bystander.
MEANS = {"fire": "hears_alarm"}
# What a body does with its legs when nothing is happening to it. Looking at
# what you have not seen is not filler: it is how a mind that only knows what
# it has looked at comes to know the building it is standing in, which is what
# makes running for a door it found honest later.
IDLE = {"legs": ("go:frontier", "go:roam", "go:step", "stay")}
# The legs' rows that are a body going for a walk. Each is marked, when the
# body has nothing to react to, with whether the whole way stays in the room it
# stands in (`inside`, will.py `_room`). A body with nothing to do wanders in
# its own room: it leaves it by choice (any row, picked by a mind), by a drive
# (a reflex's way out) or by a job, and the table, which has none of those
# while it is idle, takes only the rows that stay.
WANDER = frozenset({"go:frontier", "go:roam", "go:step"})
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
    # With nothing to react to, the table wanders in the room it stands in
    # (`WANDER`): a chooser that is not the table, a mind or a scene's script,
    # has chosen what it picks, and is not held to the room.
    idle_in_room = True

    def pick(self, situation, menu):
        limb = situation["limb"]
        want = []
        phase = situation.get("phase")
        if phase == "flinch":
            # A FLINCH IS THE EYES GOING TO IT, and nothing the sheet says:
            # it comes before there is time to want anything
            want = ["look:cue"] if limb == "eyes" else []
        elif situation["percept"] is not None:
            row = situation["reflexes"].get(situation["percept"]) or {}
            # A WARNING IS GIVEN ONCE. Under the "onset" law a body decides
            # again while a percept goes on (its goal ended, say), and the
            # percept still says what the legs want; but shouting "Fire!" again
            # at every such decision is not what the reflex meant. The mouth
            # row is for NEWS. Under "latch" there is no `onset` key, a percept
            # is always news, and the row is read as it always was.
            if limb == "mouth" and not situation.get("onset", True):
                row = {}
            if row.get(limb):
                want = [row[limb]]
            if phase == "premove":
                # STILL MAKING SENSE OF IT: what `PREMOVE` says first, then
                # the percept's own row, except for the legs, which stay
                # until the pre-movement is over (they are asked again then)
                want = list(PREMOVE.get(limb, ())) \
                    + ([] if limb == "legs" else want)
        else:
            want = list(IDLE.get(limb, ()))
        idle = situation["percept"] is None
        for tag in want:
            hit = [i for i, opt in enumerate(menu) if opt["tag"] == tag
                   and not (idle and opt.get("inside") is False)]
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
# A BODY DECIDES AT A REACTION TIME, and the tick length is what says how
# many ticks that is. Derived here rather than typed into `WILL`, because
# `TICK_S` comes from materials.py and a finer or coarser tick must not
# quietly make everybody quicker or slower on the draw.
WILL["decide_every"] = max(1, int(round(WILL["react_s"] / TICK_S)))
# AND FORGETS AT A HALF-LIFE IN SECONDS, the same way: the factor a look is
# derived from the look period, so looking more often does not forget faster.
WILL["forget"] = 0.5 ** (WILL["look_every"] * TICK_S / WILL["forget_half_s"])
