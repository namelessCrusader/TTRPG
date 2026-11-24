"""Lab world, version 2: a village inn at night (plan v3, Workstream E).

A closed scene with people and no fixed goal. Not a puzzle.

    four rooms      the main room, the kitchen, the cellar, the yard
    five people     Tom (peasant), Sir Ada (knight), Oren (merchant),
                    Fenn (a thief; the others think he is a traveller),
                    Marta (the cook). A bandit may come in from the road.

WHAT IS TRUE IS DECIDED HERE, never by a model. A policy (policy.py) only
chooses from the options this file lists; this file decides what happens.

- TRAITS (bravery, greed, pride, kindness, temper) have a mean for each type
  of character. Each character is drawn around its type, and at each choice
  its traits vary again a little: a trait is a range, not a number.
- MOOD is momentum: anger at a person, fear, drink. Events raise it; it fades
  slowly. Anger adds force to a blow and takes away aim.
- ACTIONS TAKE TIME (turns). The world keeps running while they do. A blow,
  a threat, a shout or a theft in view stops what a character is doing, and
  it chooses again.
- A CHARACTER KNOWS what it saw, heard or was told, and nothing else. Every
  belief keeps the event it came from. Beliefs can be wrong: a lie is told.
- THINGS ARE MADE WHEN NEEDED: what is in the old chest is decided when
  someone first opens it (Workstream H, in its smallest form).
- EVERY EVENT is logged in the format lab/judge/signals.py reads.

FLAGS (learned-mind plan, section 6). Each is OFF by default; with none on,
the inn is exactly what it was (lab/world2/golden.py, and the digests in
lab/results/mindL/world_flags_off.json). A flag adds facts and menu acts to
the world; it never chooses for anyone.

    drives   seven needs, each a level in 0..1 (0 = met) with a set point and a
             weight per type (NATURE) and a drift per turn (RATE): energy
             (hunger, thirst), safety (fear, hp), company, standing, wealth,
             rest (fatigue) and curiosity (want of the new). New acts: sit
             down, sleep, look around. A sleeper sees nothing, hears what is
             loud in its own room and wakes on it.
    gossip   a person who SAW a deed (a harm, a kindness, a lie found out)
             may tell another of it; a hearer who does not distrust the
             teller changes its liking of the doer (reputation).
    trade    the cook holds the food and serves it. Anyone may ask another
             for a thing, or offer a thing of its own for it; the one asked
             answers by giving, accepting, or refusing (its own choice).
    watch    a theft by someone the watcher distrusts is seen more often.
    check    lies are found out more often: seeing the thing held openly, or
             being hidden, by another than the one named; searching the
             named culprit and finding nothing; a tale that contradicts what
             the hearer saw itself.
    known    the menu's numbers (and the menu itself) come only from what the
             chooser knows. Without it, some option features read hidden
             facts: `danger` sums the OTHERS' private anger, a room's `goal`
             counts the food, people and anger in a room the chooser cannot
             see, `first` on the chest says whether anyone opened it, and the
             flee and shout options appear on that hidden danger. With it:
             danger is the hostility the chooser PERCEIVED (harsh acts aimed
             at it, fading as anger fades; kind acts lower it) plus a visibly
             armed bandit and violence it perceived; a room's goal comes from
             where it last saw each thing and person (else the inn's layout at
             the start of the night); `first` from whether it saw the chest
             opened; threat, defend and shout read only perceived events;
             a sleeper's or a fallen person's things are the ones visible or
             believed held. A chooser also notices the loss of ANY thing it
             believed it held (not only its own) at its next choice, so the
             wealth need never drops unnoticed.

    reprisal  the interrupt loophole closed (2026-10-29). A harmful act aimed at
             a person (HARMFUL: strike, shove, threaten, insult, steal, accuse,
             search) takes at least HARM_MIN turns, whatever it is: the actor is
             committed and spends its time. Everyone who SEES one lowers its
             liking of the actor a little (WITNESS_COST), and with `drives` the
             actor's own standing and company suffer for each witness. An act
             that is interrupted keeps the turns already done: choosing the same
             act again within RESUME_SPAN turns takes only what is left, so being
             interrupted costs the interrupted one nothing but the choice. The
             target's anger, liking, the gossip and its own reply were there
             before; this flag adds the actor's cost and the resume. One rule for
             every harmful act, none for a single verb.
    wear     the same words from the same person mean less each time in a short
             span (diminishing returns). For `praise` and `insult` (WEAR_VERBS):
             each earlier one by the same actor to the same target within
             WEAR_SPAN turns multiplies the effect by WEAR (liking, anger, the
             slight to standing, lonely, the hostility perceived).
    NEW FLAGS (2026-11-08, the inn fix; all four OFF in "all" and "all2", ON in "all3"):
    meals    food in shareable PORTIONS (a pot of stew, a loaf). Whoever holds the pot may eat a portion,
             SERVE one to somebody (sharing costs a fed giver little and feeds a hungry one a lot), and the inn's
             house rules sell a meal for a thing of small value (`buy`) or for work (`work`). The supply is a
             dial (World(supply=...)). Needs `trade`.
    partner  one rule for everybody: every act aimed at a person carries the actor's TRUST in that person (its
             liking, the deeds it heard of, lies and robberies it remembers). The rules score reads it the same
             for all: kind acts go to the trusted, restraint toward the trusted, less for the distrusted.
    news     gossip by SALIENCE, not by sign: a deed is worth telling by how unusual it is among the deeds the
             teller has seen, how much it goes against its own opinion of the doer, and how big it is. A witness
             warms to one who helps in view as it cools to one who harms (the mirror of `reprisal`).
    calm     the causes of the rules cast's quarrels, found in step 0a2 (all-rules cast, 5 seeds, 30 nights):
             the cast made about 52 harsh acts a night (24 charges, 16 blows, 7 searches), mean liking -0.77.
             (1) A charge was repeated: habit forgot what one had said after 40 turns, so every believer accused
                 again and again. Now the memory of what one said lasts the night and fades slowly (ECHO_HALF), and
                 saying the same kind of thing to the same person pulls less, as the same words do.
             (2) A blow never spent the anger behind it, so a fight ran until someone fell. Acting on a grievance
                 now spends part of it (DISCHARGE), as eating spends hunger.
             (3) The rules score read neither one's wounds nor the company drive. A hurt one is now held back
                 from blows and threats, and a lonely one is drawn to warm acts (policy.flag_terms).
             (4) A charge about a thing one no longer thinks the culprit holds is moot (feature `moot`, and a
                 search only of one still suspected).
             (5) Reprisal charged an answer to a wrong as much as the wrong: every witness cooled to the one who hit
                 back. A witness now charges an answer a quarter of a first harm (PROVOKED_SHARE), which is the
                 main cause of the liking spiral (liking -0.77 -> -0.33 with the rest).
    The first six flags are what "all" means (as in the first test round, so
    its numbers can be redone); "all2" is every flag, these two included.

LIFETIME (lab/world2/lifetime.py): `carry` brings in each cast member's
traits, liking and memories from the night before; `night` numbers the night.
"""
from __future__ import annotations

import functools
import math
import random

ROOMS = {
    "the main room": ("the kitchen", "the yard"),
    "the kitchen": ("the main room", "the cellar"),
    "the cellar": ("the kitchen",),
    "the yard": ("the main room",),
}

TRAITS = ("bravery", "greed", "pride", "kindness", "temper")
KINDS = {  # the type's trait means, and how the others see it
    "peasant": ("a peasant", dict(bravery=.25, greed=.40, pride=.30, kindness=.60, temper=.40)),
    "knight": ("a knight", dict(bravery=.85, greed=.30, pride=.80, kindness=.50, temper=.50)),
    "merchant": ("a merchant", dict(bravery=.35, greed=.80, pride=.60, kindness=.35, temper=.40)),
    "thief": ("a traveller", dict(bravery=.50, greed=.90, pride=.40, kindness=.25, temper=.40)),
    "cook": ("the cook", dict(bravery=.45, greed=.40, pride=.50, kindness=.60, temper=.60)),
    "bandit": ("a bandit", dict(bravery=.80, greed=.90, pride=.50, kindness=.10, temper=.80)),
    "innkeeper": ("the innkeeper", dict(bravery=.50, greed=.40, pride=.45, kindness=.55, temper=.45)),
    "child": ("a child", dict(bravery=.30, greed=.30, pride=.25, kindness=.55, temper=.40)),
    "elder": ("an old woman", dict(bravery=.40, greed=.30, pride=.55, kindness=.60, temper=.35)),
}
SPREAD, STATE_SPREAD = 0.12, 0.08

CAST = [("Tom", "peasant", "the main room", ["the copper coin"]),
        ("Sir Ada", "knight", "the main room", ["the sword"]),
        ("Oren", "merchant", "the main room", ["the purse of silver", "the lute"]),
        ("Fenn", "thief", "the main room", ["the dagger"]),
        ("Marta", "cook", "the kitchen", ["the ladle"])]

# name: (where it starts, properties, value)
THINGS = {
    "the table": ("the main room", {"fixed"}, 0),
    "the hearth": ("the main room", {"fixed"}, 0),
    "the stool": ("the main room", set(), 0),
    "the mug of ale": ("the main room", {"ale"}, 0),
    "the dice": ("the main room", set(), 0),
    "the kitchen knife": ("the kitchen", set(), 1),
    "the iron pot": ("the kitchen", set(), 1),
    "the loaf of bread": ("the kitchen", {"food"}, 0),
    "the bowl of stew": ("the kitchen", {"food"}, 0),
    "the bucket of water": ("the kitchen", {"water"}, 0),
    "the barrel of ale": ("the cellar", {"fixed", "ale_source"}, 0),
    "the old chest": ("the cellar", {"fixed", "chest"}, 0),
    "the lantern": ("the cellar", set(), 1),
    "the well": ("the yard", {"fixed", "water_source"}, 0),
    "the axe": ("the yard", set(), 1),
    "the copper coin": (None, set(), 1),
    "the sword": (None, set(), 3),
    "the purse of silver": (None, set(), 10),
    "the lute": (None, {"music"}, 3),
    "the dagger": (None, set(), 1),
    "the ladle": (None, set(), 0),
    "the club": (None, set(), 0),
}
WEAPON = {"the sword": 4, "the axe": 4, "the dagger": 3, "the kitchen knife": 3, "the club": 3,
          "the stool": 2, "the iron pot": 2}
# what the old chest holds, decided when it is first opened
CHEST = [("some old silver coins", 6, 3), ("a rusty sword", 2, 2), ("a bundle of old letters", 0, 2),
         ("a bottle of wine", 1, 2), ("nothing but rags", 0, 3)]

DUR = {"walk to": 3, "flee to": 2, "take": 2, "steal": 3, "drop": 1, "give": 2, "eat": 4, "drink": 3,
       "strike": 2, "shove": 1, "threaten": 2, "insult": 1, "praise": 1, "tell": 3, "accuse": 2,
       "shout for help": 1, "help up": 2, "bandage": 5, "defend": 2, "hide": 2, "search": 3,
       "open": 3, "play": 6, "wait": 3}
LOUD = {"strike", "shout for help", "threaten", "accuse", "play", "flee to", "shove"}
HARMFUL = {"strike", "shove", "threaten", "insult", "steal", "accuse", "search"}
VIOLENT = {"strike", "shove"}
PHYSICAL = {"strike", "shove", "threaten"}

# ---------------------------------------------------------------- flags
BASE_FLAGS = ("drives", "gossip", "trade", "watch", "check", "known")      # what "all" means
FLAGS = BASE_FLAGS + ("reprisal", "wear")                                  # "all2" is all of them
NEW_FLAGS = ("meals", "partner", "news", "calm")                          # the inn fix; "all3" is FLAGS + these
ALL3 = FLAGS + NEW_FLAGS
LUCK_FLAGS = ("luck",)                                                    # uneven luck (2026-11-20); "all4" is ALL3 + this
ALL4 = ALL3 + LUCK_FLAGS
# luck: half of the night's expected food stops coming from the house pot and comes instead as each person's own CATCH
# (foraged, hunted, earned), the same expected size for everybody (LUCK_SHARE of the 20 rations, over the 4 who are not the
# house), but it varies from night to night and independently between people. The dial v is the chance of an empty night.
# Otherwise the catch is Gamma(shape 2) scaled so the mean stays put (mean catch = mean / (1 - v)), CV^2 = 1.5/(1-v) - 1
# (v 0.3: CV 1.07; v 0.5: CV 1.41). Grounding (u, from memory, not re-checked): Ache hunters bring home no meat on about 40 percent
# of days and their daily returns have a CV near 1 or more (Kaplan and Hill 1985, "Food sharing among Ache foragers");
# Hadza and !Kung daily individual returns are similarly lumpy (Hawkes, O'Connell, Blurton Jones 2001 (u); Cashdan 1985 (u)).
# Kaplan and Hill read this variance as the reason food is shared. A catch cannot be kept overnight (each night is a new world),
# so a lucky night's surplus is worth nothing to its holder unless shared: that is the part of the model that matters.
LUCK_SHARE = 0.5
LUCK_SHAPE = 2
NEED_FLAGS = ("tend", "allies", "labour")                                # the three needs only another person can meet (2026-11-20)
ALL5 = ALL4 + NEED_FLAGS                                                 # "all5" = all4 + these three
# tend: some harms cannot be treated by the one who has them (a wound on the back, a fever). An AILMENT is a level in 0..1 that
# feeds the safety need, grows while nobody treats it, and falls only when another person TENDS it. One law for everybody.
# Sizes (all (u), from memory, not re-checked): forager-horticulturalists are sick or hurt enough to stop work on about 10 percent
# of days (Hill and Hurtado 1996 on the Ache; Sugiyama and Chacon 2000 on the Shiwiar; Gurven et al. on the Tsimane), and
# survival of the badly hurt depends on being cared for (Sugiyama 2004). So 10 percent of nights a person starts with an ailment.
AIL_P = 0.10                  # the chance that a person starts a night with an ailment
AIL_LO, AIL_HI = 0.25, 0.60   # its starting level (uniform): a fifth to over half of the need scale
AIL_GROW = 0.0012             # untreated, it grows this much a turn (+0.36 over a 300-turn night: an infection left alone)
TEND_RELIEF = 0.35            # one tending (TEND_DUR turns of the helper's time) takes this off
TEND_MIN = 0.15               # below this an ailment is not worth a visit
TEND_TIRE = 0.04              # the helper tires (as walking tires, drives flag)
# allies: some dangers beat one person and lose to two or more acting together. A RAID: a beast or a bandit comes in from the
# road (the yard) at a turn of the night, hits one person in its room every RAID_ROUND turns, and goes for the people if the room
# is empty. Each "fend off" act that finishes costs it FEND_DMG in expectation (drawn 0.6..1.4 times). One fighter takes about
# 6.7 rounds to bring it down and is down itself after about 3.4 hits; two take 3.3 rounds, so the hits are shared. Joining costs
# risk: the joiner is in the room and gets hit. A raid comes on RAID_P of nights (u: chosen from the 14 percent of forager
# deaths that were violence (Bowles 2009, u) and that a raid is rarer than an ordinary night).
RAID_P = 0.15
RAID_T0, RAID_T1 = 30, 200    # the turn it comes in (uniform)
RAID_ROUND = 5
RAID_HP, RAID_DMG, FEND_DMG = 10.0, 1.8, 1.5
FEND_WARM = 0.05              # those who stand together like each other a little more
# labour: different people bring different food, and a person needs both. The inn's larder is staples (stew, bread). The luck
# catch of each person is split by that person's own dial: the bold (bravery) bring GAME, the careful bring GREENS, in the
# share of their bravery. Nobody is a hunter or a gatherer by class; the split is the same rule for everyone and moves with the
# dial. Eating has sensory-specific satiety: a ration of the same kind as the last one eaten relieves LABOUR_REPEAT of the usual
# (Rolls et al. 1981, variety raises intake by about 60 percent, u). So a person who lives on the larder alone eats at 60
# percent, and the other kinds are in other people's hands. Needs `luck` (the catch).
LABOUR_REPEAT = 0.6
STAPLE = "staple"
# 2026-11-20 (the reciprocity-conditions answer, S2.4, S2.5, S2.3): three more general laws, each behind its own flag, the same for every person.
LUMPY_FLAGS = ("lumpy", "stock", "choosy")
ALL6 = ALL5 + LUMPY_FLAGS                                                # "all6" = all5 + these three
# signs (2026-11-21, SV2): what one person PERCEIVES of another's hurt, fear and hunger. A state shows through signs scaled by the
# target's EXPRESSIVENESS (a dial drawn per person: a stoic shows little of the same pain) and is read with the observer's ACUITY (a
# dial drawn per person: it sets how noisy the reading is), and only by an observer in the same room who is awake and acts. Told
# about it (hearsay, the knowledge layer) is a HOOK, not built: World.perceived_need is the one place it would be added. The flag adds
# the function and the dials and nothing else: it changes no choice and draws nothing from the world's own random generator.
SIGNS_FLAGS = ("signs",)
# drives2 (mindL6 stage A): with `signs`, the visible signs also carry THIRST (what an observer reads of another person's thirst), so a mind can learn by watching a person drink.
# Nothing else changes with the flag: the body and the laws are the same.
DRIVES2_FLAGS = ("drives2",)
# body2 (mindL6 stage B1, 2026-11-23): the library's body numbers (responses/2026-11-23-mark1-drives-and-cultural-learning.md, D-a to D-e). Off: nothing here is read.
#   WATER LEDGER. The need (0..1) covers thirst ONSET: 0..1 percent of body mass lost (the answer: "the need covers only about 0.8-1.3 percent"). The RESERVE
#   (LETHAL_RESERVE days, drawn down at the cap) covers everything from there to death. Death is at 15-20 percent of body WATER (Tipton and Montgomery 2025,
#   doi:10.1113/ep093194); body water is 55-75 percent of body mass (adult 60 percent, u: between the elder's 55 and the infant's 75), so death is at 17.5 x 0.60 = 10.5
#   percent of body mass (u: the middle of both ranges). The reserve maps LINEARLY onto the loss: lost_pct = 1.0 x need + (10.5 - 1.0) x dep, dep = 1 - reserve/full (u: linear).
#   Anchors by percent of body MASS lost: first symptoms 1.4-1.6 (headache, concentration; Ganio 2011, Armstrong 2012, Johnson 2014 cards), cognition -0.28 SD above 2
#   (Wittbrodt and Millard-Stafford 2018), headache and irritability above 5, weakness and faintness at 8-10 (Craig and Cummings 1966 via Tipton), death 10.5 (this mapping).
#   FELT LADDER (STEPS, not a slope: the answer, L3). Rung heights are (u); the thresholds are the anchors. The top rung is 1.0 = FELT_GAIN.
WATER_LOSS_NEED_PCT, WATER_LOSS_END_PCT = 1.0, 10.5
WATER_RUNGS = ((1.5, 0.15), (2.0, 0.30), (5.0, 0.60), (8.0, 0.85), (10.0, 1.00))
#   FOOD LEDGER. Death in complete starvation at 40-60 days (the 45 days stays). Body WEIGHT lost at death 40 percent (u: Keys' men kept living at over 25 percent; the
#   fatal loss is not in the library). Linear in dep. Anchors by percent of body weight lost: no cognitive loss to 6, capacity falls past 10, fatigue, apathy, irritability and
#   extreme weakness at over 25 (Tipton and Montgomery 2025; Keys 1950 via Franklin 2005). Rung heights (u).
FOOD_LOSS_END_PCT = 40.0
FOOD_RUNGS = ((6.0, 0.10), (10.0, 0.30), (25.0, 0.70), (40.0, 1.00))
#   REPAYMENT. A 24 h water deficit was gone within 20 minutes of drinking (Rolls 1980). A turn is 4.8 minutes (a night is a day of 300 turns): the thirst reserve is
#   repaid in REPAY_TURNS = 20 / 4.8 turns once the need is below its cap. Hunger keeps the old give-back (a day a day: no card number, u).
REPAY_TURNS = 20.0 / 4.8
#   EATING. (1) A meal raises thirst: EAT_THIRST per ration (u: the human size is in no card; mouse thirst neurons rise from the first bite, Zimmerman 2016; booked from the
#   meal's water cost: about 22 percent of water intake comes from food, a ration of about 600 kcal costs about 0.07 L, 0.05 of the thirst scale where a drink of 0.85 L is 0.5).
#   (2) The dehydrated eat less: at 2.9 percent of body mass lost men ate 0.65 of their rehydrated intake (Perez-Luco 2019): a ration relieves hunger by factor
#   1 - 0.35 x lost_pct / 2.9, never below EAT_FLOOR (u). The ration is used up (the pot is in whole rations): the rest is waste (u).
EAT_THIRST, EAT_ANOREXIA, EAT_REF_PCT, EAT_FLOOR = 0.05, 0.35, 2.9, 0.35
#   INNATE SET. CRY: a graded sign = CRY_GAIN x the sum of unmet needs (hunger and thirst above their usual level, hurt, fear), times a gain that halves a year after
#   CRY_AGE = 2 (u). CARERS CANNOT HEAR ITS CAUSE (35 percent against 33 chance, Parsons/Ahrens-Leuchter 2023): below CRY_CAUSE_AGE (u) the hunger and thirst signs are hidden and only
#   the cry shows; the carers' serve options for a DEPENDENT (under DEPENDENT_AGE = 4 (u): the youngest age children are sent for water, Stieglitz 2012) are driven by the cry.
#   TASTE: sweet-good, bitter-bad (Berridge 2000): a signed taste for what is consumed, read by the mind as a first valence (a prior with a pseudo-count). Values (u): ale -0.5
#   (bitter), food +0.4, water 0.
CRY_GAIN, CRY_AGE, CRY_CAUSE_AGE, DEPENDENT_AGE = 1.0, 2.0, 2.0, 4.0
TASTE_ALE, TASTE_FOOD, TASTE_WATER = -0.5, 0.4, 0.0
BODY2_FLAGS = ("body2",)
#   BODY SHEETS (D-e, L8): elder work 0.50 -> 0.85 (Tsimane men 10-20 percent below peak at 60+, women flat; grandfathers' net food giving peaks in their 60s); child metab 0.70 kept
#   (1,738-1,811 kcal a day at 8 against 2,400 is 0.72-0.75). Elder ill, ill_len and recov stay (u: no morbidity-by-age card; frailty is 10.7 percent at 65+, a check, not an input).
BODY2_SHEET = {"elder": dict(work=0.85)}


def water_lost_pct(need, dep):
    """Percent of body mass lost to water, from the thirst need (0..1) and the reserve fraction lost (dep, 0..1)."""
    return WATER_LOSS_NEED_PCT * min(1.0, max(0.0, need)) + (WATER_LOSS_END_PCT - WATER_LOSS_NEED_PCT) * min(1.0, max(0.0, dep))


def food_lost_pct(dep):
    """Percent of body weight lost to hunger, from the reserve fraction lost."""
    return FOOD_LOSS_END_PCT * min(1.0, max(0.0, dep))


def rung(pct, rungs):
    """The felt level (a step) at `pct` lost: the height of the highest rung whose threshold is reached."""
    lvl = 0.0
    for thr, h in rungs:
        if pct >= thr:
            lvl = h
    return lvl


def cry_gain(age):
    return CRY_GAIN * (0.5 ** max(0.0, age - CRY_AGE))


def cry_level(hunger, thirst, hurt, fear, age):
    """The cry (0..1): graded by the sum of unmet needs, the same whatever the cause."""
    unmet = max(0.0, hunger - LETHAL_START) + max(0.0, thirst - LETHAL_START) + hurt + fear
    return min(1.0, cry_gain(age) * unmet)
# hard (2026-11-21, the realistically hard inn, owner decision (A)): the hardships set FROM THE SOURCE RATES instead of by taste. Source:
# the Library answer "responses/2026-11-21-mark1-preindustrial-hardship-rates.md" (S1 to S6). One night is one day. With the flag off nothing
# here is read. With it on, `ill`, `luck` and `raid` given to World() are ignored: the source rates below are used.
HARD_FLAGS = ("hard",)
#  1. LAID UP. An ailment is an EPISODE that lasts across nights. While one lasts the person is laid up: it cannot do what exerts the body
#     (the EXERT acts: walk, flee, strike, shove, play, search, tend, fend off) or work, and cannot reach the counter (buy), and it brings
#     no catch. It can still ask and be served and tended, and everything that needs only a word or a hand. Others must feed and tend it.
#     Prevalence about 10 percent of days; episodes mean 8.7 days, 91 percent over 3 days (Gurven et al. 2012, Tsimane,
#     doi:10.1016/j.exger.2012.05.006, 508 bedridden adults; 99.4 percent of them were fed by others); 13 percent last a month or more
#     (Sugiyama 2004, Shiwiar, doi:10.1002/ajpa.10325, a different group). The length is a two-part lognormal (a bulk and a heavy tail)
#     fitted to these three shares (lab/results/mindL4/hardinn/hardinn_fit_duration.py): mean 8.70, P(n > 3) 0.910, P(n >= 30) 0.130, where
#     n = whole nights, max(1, round(x)). The paper's own SD (20.9) is NOT matched: this law has SD 11.4. The three shares and the SD do
#     not fit together in one law with mean 8.7 (the best four-way fit misses the mean by 20 percent).
HARD_DUR = (0.859922, 1.419916, 0.133144, 3.58703, 0.138655)     # weight of the bulk, then (mu, sigma) of the bulk and of the tail (of ln days)
HARD_PREVALENCE = 0.10                                          # of person-days laid up (Gurven 2012; "about 10 percent of days")
#  Onset hazard per HEALTHY person-night, from prevalence = E[n] / (E[n] + (1 - h) / h), so h = 1 / (1 + E[n] (1 - P) / P). The Library's
#  1.1 percent a day is prevalence / duration, which ignores that the sick cannot fall ill again; h comes out about 1.26 percent.
#  (hard_duration_mean() and hard_onset() below compute them from HARD_DUR.)
#  2. FOOD. About half of hunts fail (Tsimane 45 percent, Gurven et al. 2015, doi:10.1007/s10745-015-9764-y; Ache "about every other time",
#     Codding et al. 2011, doi:10.1098/rspb.2011.1458). The chance of an empty catch is 0.48. A hit is a Gamma(shape 2) size (LUCK_SHAPE) with the
#     mean kept. Person-day CV^2 = (1 + 1/2) / 0.52 - 1 = 1.885, CV 1.37 (the field: 1.1 for Ache calories, 1.6 for a Tsimane household, S2).
HARD_EMPTY = 0.48
#  3. RAIDS. Rare. Violent death is 0.03 percent (late medieval Europe, Eisner 2003) to 1 percent a person-year (precontact Hiwi, Hill et al.
#     2007, doi:10.1016/j.jhevol.2006.11.003); a force raid kills about 1.1 percent of the warriors it meets (Turkana, Mathew and Boyd 2011,
#     doi:10.1073/pnas.1105604108). Deaths a night in a five-person inn = 5 r / 365; deaths per raid = 5 x 0.011; so raids a night = r / (365 x
#     0.011). At r = 1 percent: 0.0025 a night (one raid in 400 nights); at 0.03 percent: 0.000075. The higher end is used (the most danger
#     the sources support).
HARD_RAID = 0.01 / (365 * 0.011)
#  4. COLD: unmeasured in the sources (S5): left as it was.
# lethal (2026-11-21, owner D1 lever (c): neglect as deadly as it really is). ONE general law for every need that has a clock: a need held at its
# EXTREME (the cap, 1.0) draws down a RESERVE of days; a need below the extreme gives the days back at the same pace; a reserve at zero is death.
# No per-cause script: thirst and hunger are two rows of one table. The laid-up die because they cannot reach water or food unless someone
# brings it: the death comes from the same law. A person's hunger, thirst and reserves CARRY from night to night (without the flag each night
# starts fresh). Total time without the thing, from a usual level (0.325) to death, in days (LETHAL_DAYS):
#   thirst 5 days. Humans die without water in about 3 to 7 days, depending on heat, effort and health (v: the "rule of threes" literature,
#     Wikipedia "Rule of threes (survival)", Healthline, Live Science, accessed 2026-11-21; popular summaries, no single clinical paper). Mid-range 5.
#   hunger 45 days. A healthy adult on water alone lasts about 30 to 60 days; the 1981 Irish hunger strikers died at 46 to 73 days (v: Wikipedia
#     "Hunger strike", "1981 Irish hunger strike"; well-fed young men with water and salt, so the inn's 45 is the lower end). The owner's brief: 30 to 60.
#   The reserve is that total minus the time the drift takes to carry the need from 0.325 to the cap (RESERVE below).
# RATE MULTIPLIER while ill: the days are used up 1 + ail times as fast (u: fever raises water loss and metabolic rate, about 13 percent a degree
#   in the literature (k); ail is 0..1 so a full ailment doubles the rate; not fitted to anything).
# GIVE-BACK: a day away from the extreme returns a day of reserve (u: refeeding and rehydration take about as long as the deficit did).
# UNTREATED AILMENTS: no case-fatality datum for untreated common illness or injury was found in the Library or the search; the death is DERIVED
#   from thirst and hunger while laid up. Sugiyama (2004) ASSUMES that a month or more of disability is fatal without food from others (an
#   assumption, not a measurement); here a month laid up and unfed kills by the hunger row (45 days, about 22 with the full-ailment rate: 44.25 / 2), and unwatered by the thirst row.
# lethal2 (2026-11-22): three general fixes found by the first lethal check, each behind its own flag (off: exactly as before).
#   bring    ONE act brings any consumable to a person who cannot fetch it: `serve` now covers a drink as well as a portion of a pot. A drink comes from
#            a source (the well, the barrel) or a loose drink, fetched on a trip whose time is the walk there and back (the carer stays put in the model;
#            only the time is paid). The laid up can ask for a drink as they ask for a portion. A drink and a meal are one act with one effect each.
#   raidlaw  a raid's blows go to those who can still resist (the standing); a person on the floor is out of the fight and is not struck again; when
#            nobody is standing the raiders go. Before, a raid struck the fallen until they were dead (about 25 blows a raid, 1.6 deaths a raid). Each blow
#            a raid is a killing raid with chance RAID_KILL (its first blow is mortal), the one constant, read from the source rate below.
#   selfcare the partner scripts' background holds the SAME self-care every chooser has (see policy.self_care): they ask, buy and work for their own food and drink.
LETHAL2_FLAGS = ("bring", "raidlaw", "selfcare")
RAID_KILL = 5 * 0.011   # a raid kills 5 x 0.011 = 0.055 people (Turkana, 1.1 percent a warrior, see HARD_RAID): the source's own unit is the RAID, so a raid
#                         is a killing raid with this chance, and its first blow is then mortal. Deaths per raid follow the source by construction, whatever
#                         the number of blows. Measured cause of the old 1.6 deaths per raid (300 raids): a person on the floor was struck again until dead
#                         (with the raid law and no mortal blow: 0 deaths in 300 raids, 21.6 blows and 0.8 people knocked down per raid).
SOURCE_PROPS = frozenset({"water_source", "ale_source"})
DRINK_PROPS = frozenset({"water", "ale", "water_source", "ale_source"})
LETHAL_FLAGS = ("lethal", "perma")     # lethal: the law above. perma: a person who dies stays dead (no next-night revival); the lab's PERMANENT-DEATH variant.
LETHAL_DAYS = {"thirst": 5.0, "hunger": 45.0}
LETHAL_START, LETHAL_TURNS = 0.325, 300
LETHAL_DRIFT = {"thirst": 0.004, "hunger": 0.003}
LETHAL_RESERVE = {k: LETHAL_DAYS[k] - (1.0 - LETHAL_START) / (LETHAL_DRIFT[k] * LETHAL_TURNS) for k in LETHAL_DAYS}
LETHAL_EXTREME = 0.999

# field (2026-11-22, owner "makes sense"): the FOOD SUPPLY and the BANDIT's arrival set from the sources, not by taste or to pass a test. Needs `hard` (it
# builds on the hard world's empty catch 0.48, its Gamma(2) size and its laid-up people). With the flag off nothing here is read.
#  1. DEMAND. A person's hunger drifts 0.9 a night (LETHAL_DRIFT x LETHAL_TURNS) and a ration takes PORTION = 0.25 off it, so a person needs
#     NEED_RATIONS = 3.6 rations a day to stay level. In kcal, one need unit is the adult's daily need: Tsimane adult men's mean daily energy intake
#     is 2,592 kcal (Library card doi:10.7554/elife.62883) and the brief's adult range is 2,000-2,800 (Pontzer et al. 2012, Hadza total energy
#     expenditure, about 2,600 for men and 1,900 for women, from memory (u)); the middle, 2,400 kcal, is used, so a ration is 667 kcal. (The older
#     "1,100 kcal a need unit" note in the meals block made the inn feed people about 40 percent of a real day; the dynamics are unchanged, only the kcal label.)
#  2. SUPPLY. The old inn made about 15 rations a night for 5 people who need 18: the counter (`stock`) is refilled by the cook alone (COOK_MAKES = 5 a
#     night; the 20 rations of STOCK_ were a night-0 stock, halved by `luck`), the four others bring a catch of mean LUCK_SHARE x 20 / 4 = 2.5 rations (0.69 of a
#     day's need), and the laid-up bring nothing. Measured in the lethal2 sweep (60 lives, 43,800 nights): 13.5 rations eaten a night, 80 percent of
#     the need. Everybody lived on the edge of the 45-day reserve, a SUPPLY deficit, not a sharing or self-care one.
#     Now: the MEAN supply equals the MEAN demand, the steady state of any stable population (the Tsimane household mean production, 5,700 kcal a day, is
#     the consumption of the household it feeds, Gurven et al. 2015). Counter refill C (the cook's COOK_MAKES, unchanged) plus the catches of the producers
#     who are well (prevalence HARD_PREVALENCE are laid up and bring nothing) equal the demand of the cast:
#         C + n_prod x (1 - HARD_PREVALENCE) x m = n x NEED_RATIONS    ->  m = 3.61 rations (1.00 of a person's day) for each producer's well day.
#     The field's VARIANCE is the hard world's, unchanged: empty chance 0.48 (Tsimane hunts 45 percent, Ache about half), person-day CV 1.37 (field
#     1.1 Ache, 1.6 Tsimane household), Gamma(2) size. `m` is the mean of the whole-ration catch (the integer rounding is corrected, field_catch_mean()).
#     Work at the counter (WORK_YIELD) is labour turned into food and is NOT counted in the mean supply (it adds about 3 percent, measured).
#     So any shortfall that still happens comes from the variance (a household-day's production is below its need on most days), access and choice.
#     ENERGY IS CONSERVED (found by the first smoke run of this flag: with a supply of 101 percent of the need in rations, a ration took only 0.65-0.81 of its
#     nominal 0.25 off the hunger, so the cast was relieved of 63-89 percent of its drift and 58-87 percent of days reached the extreme). Cause: the LABOUR_REPEAT
#     satiety law cuts a ration of the kind just eaten to 60 percent of its relief (Rolls et al. 1981, "variety raises intake about 60 percent", u): a food
#     that is eaten but takes only 60 percent off the need destroys kcal. Rolls measured INTAKE (appetite), not absorption; a kcal is a kcal. Under `field` a ration
#     relieves PORTION whatever was eaten before (the discount stays for the old worlds). Game and greens still exist as kinds; only the vanishing of energy is gone.
#     NOTHING IS WASTED ON THE SATIATED (second finding of the smoke run): the cast ate 104 percent of the need in rations but took only 80 percent off its hunger,
#     because a serve (or a buy, or work) fed a person whose hunger was already at or below FULL and the ration was used up with no relief (Fenn ate 141 percent
#     of the need, Marta 119 percent, while Tom, Sir Ada and Oren ate 81-90 percent). Under `field` a person eats only when the hunger left can take a whole ration
#     (hunger >= PORTION, self.full): a smaller appetite leaves the portion in the pot. That removes the partial-ration waste too (a ration eaten at hunger 0.15 wasted 0.10).
#     CHECK OF THE BALANCE (smoke runs, 3 lifetimes x 120 nights): production without work 102 percent of the demand; of it about 20 percent spoils (lumps that nobody
#     could eat in time: the perishable-catch law, half-life 90 turns, u), the work at the counter adds about 13 percent, and the cast EATS 98 percent of its need.
#     Raising the production to cover the spoilage (to 136 percent) only raised the spoilage (to 31 percent of production), so no loss factor is added: the supply meets the
#     demand in the net sense too, and who is still short is a matter of timing, access and choice (see the report).
#  3. THE BANDIT. The old arrival chance was 0.6 a night (unsourced). The sources fix DEATHS, not arrivals: violent death 0.03 percent (late-medieval Europe,
#     Eisner 2003) to 1 percent a person-year (precontact Hiwi, Hill et al. 2007); no source measures non-lethal threats (the library says so). The HARD world
#     already spends the top of the band, 1 percent, on raids (Turkana unit). Here the top-of-band budget R_VIOL = 0.01 a person-year is the total of ALL
#     outside violence and is SPLIT: half to raids (HARD_RAID x RAID_SHARE) and half to the lone bandit; the half-and-half is an unsourced choice, backed only
#     by Eisner's finding that most late-medieval killings were quarrels between acquaintances, not strangers' attacks. The bandit's arrival chance per night is
#         b = R_VIOL x (1 - RAID_SHARE) x n / (365 x d),   d = deaths per arrival, MEASURED in this world (FIELD_D_BANDIT, a calibration run of forced bandit nights).
FIELD_FLAGS = ("field",)
NEED_RATIONS = LETHAL_DRIFT["hunger"] * LETHAL_TURNS / 0.25      # PORTION = 0.25 is defined further down; 3.6 rations a person a day
FIELD_KCAL_DAY = 2400.0                                          # a person's daily need; a ration is FIELD_KCAL_DAY / NEED_RATIONS kcal
R_VIOL = 0.01                                                    # the violent-death budget of the whole inn, per person-year (top of the library band)
RAID_SHARE = 0.5                                                 # the share of R_VIOL spent on raids; the rest is the bandit's
# Measured 2026-11-22: 12 lives x 1,000 fresh nights (seeds 5001-5006 and 5101-5106 x 2,000 nights, `calib_bandit.py`), the bandit forced to arrive, the cast all
# `rules` (Fenn P-need, Oren P-hard), nights with a raid excluded: 22 people killed by the bandit in 23,969 nights = 9.18e-4 per arrival (Poisson 95% CI 5.8e-4 to 1.4e-3).
# Fresh nights (no wound carried in) so a bandit that meets a person already hurt kills a little more often than this: the rate is a floor, the sweep checks the total.
FIELD_D_BANDIT = 22 / 23969


@functools.lru_cache(maxsize=None)
def field_catch_mean(v=HARD_EMPTY, target=None, n_prod=4, n=5, prevalence=HARD_PREVALENCE, counter=5):
    """The `mean` argument of World._luck under `field`: the continuous mean that makes the MEAN WHOLE-RATION catch of a well producer equal
    m = (n x NEED_RATIONS - counter) / (n_prod x (1 - prevalence)). A catch is int(Gamma(2, scale) x mean / (1 - v) + 0.5) with chance 1 - v, else 0."""
    import math
    m = target if target is not None else (n * NEED_RATIONS - counter) / (n_prod * (1.0 - prevalence))

    def cdf(x, th):                       # Gamma(shape 2, scale th)
        return 1.0 - math.exp(-x / th) * (1.0 + x / th) if x > 0 else 0.0

    def got(mu):                          # E[int(G x mu / (1 - v) + 0.5)] x (1 - v), summed over the whole-ration steps
        th = mu / (2.0 * (1.0 - v))
        return (1.0 - v) * sum(1.0 - cdf(k + 0.5, th) for k in range(0, 400))
    lo, hi = 0.0, 4.0 * m
    for _ in range(80):
        mid = (lo + hi) / 2.0
        lo, hi = (mid, hi) if got(mid) < m else (lo, mid)
    return (lo + hi) / 2.0


def field_bandit_p(n=5):
    """The chance of a bandit's arrival on a night, from the violent-death budget and the measured deaths per arrival."""
    assert FIELD_D_BANDIT, "run the bandit calibration first"
    return min(1.0, R_VIOL * (1.0 - RAID_SHARE) * n / (365.0 * FIELD_D_BANDIT))
EXPR_MEAN, EXPR_SD, EXPR_LO, EXPR_HI = 0.7, 0.2, 0.15, 1.0     # expressiveness: the share of a state's size that shows
ACU_MEAN, ACU_SD, ACU_LO, ACU_HI = 1.0, 0.3, 0.4, 1.6          # acuity: the reading's noise is SIGN_NOISE / acuity
SIGN_NOISE = 0.12
# lumpy: some acquisitions come in packages larger than one person can use before they spoil. A person's catch is no longer a
# smooth draw: with chance LUMP_P the day's hunt or harvest is a hit and brings the whole day's expected food at once, one package of
# mean / LUMP_P rations (about 8, which is 2 need units, more than the 0.9 a person needs in a night), at a turn of the night; otherwise
# nothing. The expected catch is unchanged. Food spoils (a time law on a fresh catch): each ration has a half-life of SPOIL_HALF turns
# (u: fresh meat or greens left in the warm for a day-night; a night is 300 turns). The house's staples are cooked and kept and do not
# spoil. So the holder can use only part of a package, the rest is worth nothing to it unless it is shared, and the hungry at the
# door value it fully: marginal value to the donor low, to the recipient high (Kaplan and Hill; Winterhalder; Gurven 2004, S2.4).
# The giving rule reads one new fact for everybody: `spare`, the share of the pot the holder could not eat before it spoils.
LUMP_P = 0.30
LUMP_T0, LUMP_T1 = 1, 150
SPOIL_HALF = 90
SPOIL_RATE = 1.0 - 0.5 ** (1.0 / SPOIL_HALF)
# stock: the counter (the house's stew and bread) is a finite stock that carries from night to night. Nothing refills it but
# people's own work: the cook makes COOK_MAKES rations a night, and every `work` done at the counter puts WORK_YIELD back after the
# worker takes its WORK_PORTIONS. It cannot hold more than STOCK_CAP times the nominal pot. So a run of bad nights can leave it empty.
STOCK_CAP = 2
COOK_MAKES = {"the pot of stew": 3, "the loaf of bread": 2}
WORK_YIELD = 2
# choosy: the partner rule that already governs whom one serves, gives to and trades with (trust, distrust, liking from one's own
# history with that person) also governs whom one tends, defends, bandages and helps up. The same weight as serve's.
CHOOSY_W = 2.0
CHOOSY_ACTS = frozenset({"tend", "defend", "bandage", "help up"})
HARM_MIN = 3             # reprisal: the least turns a harmful act aimed at a person takes
WITNESS_COST = 0.04      # reprisal: a witness's liking of the actor falls by this
WITNESS_SLIGHT = 0.03    # reprisal (+drives): the actor's standing / company worsen by this per witness
RESUME_SPAN = 20         # reprisal: an interrupted act can be taken up again within this many turns
# meals: a ration takes PORTION off the hunger need. One need unit is about 1,100 kcal a day-night (a night's drift of
# 0.9 is about 1,000 kcal), so a ration of 0.25 is about 275 kcal (u): a bowl of stew of 290 g at 95 kcal/100 g, or a
# 100 g slab of bread at 265 kcal/100 g. The pot holds 12 bowls (3.5 kg), the loaf 8 slabs (0.8 kg): 20 rations = 5.0 need units, against about 4.6 that
# five people need to stay at their set point through a night (0.9 drift + 0.3 start - 0.3 set point, each). All (u).
PORTION = 0.25
FULL = 0.10             # meals: below this hunger a person is full and eats no more
STOCK = {"the pot of stew": 12, "the loaf of bread": 8}
MEAL_PORTIONS, WORK_PORTIONS = 2, 1     # a thing paid buys a meal (two rations); an hour's chores earn one
WORK_TIRE, CHORE_RELIEF = 0.06, 0.04    # work tires the worker, and eases the one it helps
PAYABLE = (1, 3)                        # the values a house will take for a meal (no change is given)
# partner: the weight of trust in the score of an act aimed at a person (the same for every actor)
PARTNER_W = 1.5
BENEFIT_ACTS = {"give", "praise", "help up", "bandage", "defend", "gossip", "offer", "accept", "serve", "buy", "work", "tend"}
RESTRAINT_ACTS = {"strike", "shove", "threaten", "insult", "steal", "accuse", "search"}
HELPED = {"give", "help up", "bandage", "defend", "serve", "tend"}   # news: a witness warms to the one who does these
SURPRISE_BITS = 3.0     # news: a deed 1 in 8 of what the teller has seen is fully surprising
# calm
WOUND_W = 2.0           # calm: the rules score reads one's own wounds (the safety drive), as it reads hunger
PROVOKED_SHARE, PROVOKED_SPAN = 0.25, 60   # calm: a witness charges an answer to a wrong this share of a first harm
ECHO_HALF = 150         # calm: the memory of what one has said halves every 150 turns (half a night)
DISCHARGE = 0.5         # calm: a harmful act aimed at a person spends this share of the actor's anger at it
LONELY_W, LONELY_SET = 1.5, 0.3      # calm: weight of the company drive (above the set point) on warm acts
WARM_ACTS = {"praise", "gossip", "serve", "give"}
WEAR_VERBS = {"praise", "insult"}
WEAR, WEAR_SPAN = 0.5, 40   # wear: each repeat within the span halves the effect
DUR.update({"tend": 6, "fend off": RAID_ROUND})
DUR.update({"sit down": 4, "sleep": 12, "look around": 2, "gossip": 3, "ask": 1, "offer": 1,
            "accept": 2, "refuse": 1, "serve": 2, "buy": 2, "work": 10})
DRIVES = ("energy", "safety", "company", "standing", "wealth", "rest", "curiosity")
# NATURE: for each type, each drive's (weight, set point). The error of a drive
# is max(0, need - set point); a weight of 0 means the type does not care.
NATURE = {
    "peasant": dict(energy=(1.0, .30), safety=(1.2, .15), company=(0.8, .30), standing=(0.5, .40),
                    wealth=(0.0, 1.0), rest=(0.8, .40), curiosity=(0.4, .50)),
    "knight": dict(energy=(0.8, .35), safety=(0.6, .30), company=(0.5, .40), standing=(1.2, .20),
                   wealth=(0.0, 1.0), rest=(0.6, .45), curiosity=(0.5, .40)),
    "merchant": dict(energy=(0.9, .30), safety=(1.0, .20), company=(0.6, .35), standing=(0.8, .30),
                     wealth=(1.2, .20), rest=(0.7, .40), curiosity=(0.3, .50)),
    "thief": dict(energy=(0.9, .30), safety=(0.9, .25), company=(0.3, .50), standing=(0.3, .50),
                  wealth=(1.3, .20), rest=(0.6, .45), curiosity=(0.8, .30)),
    "cook": dict(energy=(0.8, .30), safety=(0.9, .20), company=(1.1, .25), standing=(0.7, .30),
                 wealth=(0.0, 1.0), rest=(0.9, .35), curiosity=(0.4, .45)),
    "bandit": dict(energy=(0.8, .35), safety=(0.6, .35), company=(0.1, .70), standing=(0.6, .30),
                   wealth=(1.4, .10), rest=(0.5, .50), curiosity=(0.5, .40)),
    # family inn (flag `family`; only the family cast uses these three)
    "innkeeper": dict(energy=(0.9, .30), safety=(0.9, .20), company=(0.9, .30), standing=(0.8, .30),
                      wealth=(0.0, 1.0), rest=(0.8, .40), curiosity=(0.4, .50)),
    "child": dict(energy=(1.1, .25), safety=(1.3, .15), company=(1.0, .25), standing=(0.2, .50),
                  wealth=(0.0, 1.0), rest=(0.8, .40), curiosity=(0.9, .30)),
    "elder": dict(energy=(0.9, .30), safety=(1.1, .20), company=(0.9, .30), standing=(0.5, .40),
                  wealth=(0.0, 1.0), rest=(0.9, .35), curiosity=(0.3, .50)),
}
WANT = {"merchant": 13, "thief": 6, "bandit": 6}      # held value that meets the wealth need
# drift per turn, when nothing serves the need (hunger and thirst keep their own)
RATE = {"rest": 0.002, "curiosity": 0.004, "company": 0.004, "company_met": 0.006, "standing_fade": 0.995}
EXERT = {"walk to": 0.01, "flee to": 0.02, "strike": 0.02, "shove": 0.01, "play": 0.02, "search": 0.01,
         "tend": TEND_TIRE, "fend off": 0.02}
# how being the TARGET of an act moves standing (slight) and company (lonely)
SLIGHT = {"insult": .20, "accuse": .25, "search": .20, "threaten": .15, "strike": .10, "shove": .10,
          "refuse": .15, "praise": -.20, "give": -.15, "help up": -.10, "bandage": -.10, "defend": -.10,
          "accept": -.10, "gossip": -.05, "serve": -.15, "tend": -.10}
KIND_ACTS = {"praise", "give", "help up", "bandage", "defend", "accept", "gossip", "serve", "tend"}
HARSH_ACTS = {"insult", "refuse", "threaten", "strike", "shove", "accuse", "search"}
# deeds a witness may tell of (gossip): +1 kind, -1 harm, -1.5 a lie found out
DEED = {**{a: -1.0 for a in HARMFUL}, "give": 1.0, "help up": 1.0, "bandage": 1.0, "defend": 1.0,
        "praise": 0.5, "refuse": -0.5, "lie": -1.5, "serve": 1.0, "tend": 1.0}
PAST = {"strike": "struck", "shove": "shoved", "threaten": "threatened", "insult": "insulted",
        "steal": "robbed", "accuse": "accused", "search": "searched", "give": "gave a gift to",
        "help up": "helped up", "bandage": "bandaged", "defend": "defended", "praise": "praised",
        "refuse": "refused", "lie": "lied to", "serve": "fed", "tend": "tended"}
# known: how much a perceived act aimed at me says about the actor's hostility
# toward me (a kind act lowers it); it fades as anger fades (0.985 a turn)
HOSTILE = {"strike": .6, "shove": .5, "threaten": .4, "steal": .4, "insult": .25, "accuse": .25,
           "search": .25, "refuse": .1, "give": -.3, "help up": -.3, "bandage": -.3, "defend": -.2,
           "praise": -.1, "accept": -.1, "serve": -.3, "tend": -.3}
START_ROOM = {n: room for n, _, room, _ in CAST}

# ---------------------------------------------------------------- family (2026-11-23, mindL5 stage A; docs/family-design.md Fam3)
# A family-run inn with lodgers: an innkeeper couple (the wife is the cook), a child, an elderly parent (the wife's mother) and three unrelated lodgers.
# Nothing here says who is kin: the world holds ages and body sheets only. The household's PAST (who lived with whom, who cared for whom) lives in
# lab/mindL/household.py. With the flag off nothing here is read and the old inn is bit for bit as before.
FAMILY_FLAGS = ("family",)
FAMILY_CAST = [("Jon", "innkeeper", "the main room", ["the club"]),
               ("Mara", "cook", "the kitchen", ["the ladle"]),
               ("Pip", "child", "the main room", []),
               ("Gran", "elder", "the kitchen", []),
               ("Tom", "peasant", "the main room", ["the copper coin"]),
               ("Sir Ada", "knight", "the main room", ["the sword"]),
               ("Oren", "merchant", "the main room", ["the purse of silver", "the lute"])]
FAMILY_AGE = {"Jon": 38, "Mara": 36, "Pip": 7, "Gran": 66, "Tom": 28, "Sir Ada": 45, "Oren": 40}
# Body sheets by age class, each a multiplier on the adult's. ALL (u): from memory, not re-checked against the library; a request goes to it (report, needs).
#   metab    ration: FAO/WHO/UNU 2004 energy requirements, about 1,700 kcal a day at 7 years and about 2,000 at 60+ against an adult's 2,400 here -> 0.70, 0.83.
#   work     share of an adult's food-getting (the catch, work at the counter): children's foraging returns are a small fraction of an adult's until the
#            teens, and returns fall in old age (Kaplan 1994; Gurven and Kaplan 2006 on Tsimane and Hadza returns by age) -> 0.15 and 0.5.
#   strength share of an adult's blow (grip strength at 7 is about a third of an adult's, at 70 about 0.7) -> 0.35, 0.7.
#   ill      the chance that a healthy person falls ill on a night, times (morbidity rises with age: Tsimane, Gurven et al.) -> elder 2.0.
#   ill_len  an episode's length, times (slower recovery) -> elder 1.5.   recov: a tending takes this share of the usual off an ailment -> elder 0.7.
FAMILY_BODY = {"child": dict(metab=0.70, work=0.15, strength=0.35, ill=1.0, ill_len=1.0, recov=1.0),
               "elder": dict(metab=0.83, work=0.50, strength=0.70, ill=2.0, ill_len=1.5, recov=0.7)}
FAMILY_ADULT = dict(metab=1.0, work=1.0, strength=1.0, ill=1.0, ill_len=1.0, recov=1.0)


LAID_UP_BARRED = frozenset(EXERT) | {"work", "buy"}


def _hard_cdf(x):
    """P(episode length < x days) under HARD_DUR (x > 0)."""
    w, m1, s1, m2, s2 = HARD_DUR
    z = lambda m, sg: 0.5 * (1.0 + math.erf((math.log(x) - m) / (sg * 2 ** 0.5)))
    return w * z(m1, s1) + (1 - w) * z(m2, s2)


def hard_duration_pmf(kmax=4000):
    """P(n nights) for n = 1..kmax, n = max(1, round(x)) (the discrete law the world draws from)."""
    out, prev = [], 0.0
    for k in range(1, kmax + 1):
        c = _hard_cdf(k + 0.5)
        out.append(c - prev)
        prev = c
    out[0] = _hard_cdf(1.5)
    return out


def hard_duration_mean():
    return sum((k + 1) * q for k, q in enumerate(hard_duration_pmf()))


def hard_onset(prevalence=HARD_PREVALENCE):
    """The chance a healthy person falls ill on a night so that `prevalence` of nights are spent laid up."""
    return 1.0 / (1.0 + hard_duration_mean() * (1.0 - prevalence) / prevalence)


def hard_duration_draw(u1, u2, z):
    """One episode length in nights from two uniforms and a standard normal (fixed draws, so a stream does not depend on the branch)."""
    w, m1, s1, m2, s2 = HARD_DUR
    m, sg = (m1, s1) if u1 < w else (m2, s2)
    return max(1, int(math.exp(m + sg * z) + 0.5))


class Person:
    def __init__(self, name, kind, room, rng):
        self.name, self.kind, self.room = name, kind, room
        mean = KINDS[kind][1]
        self.base = {t: min(1, max(0, rng.gauss(mean[t], SPREAD))) for t in TRAITS}
        self.traits = dict(self.base)
        self.anger = {}            # at whom, fades
        self.fear = 0.0
        self.drunk = 0.0
        self.hunger = rng.uniform(0.15, 0.5)
        self.thirst = rng.uniform(0.15, 0.5)
        self.hp = 10.0
        self.down = False
        self.dead = False
        self.doing = None          # the action under way
        self.aff = {}              # liking of each other person, -1..1
        self.bel = {}              # ("where", thing) / ("took", thing) -> (value, source)
        self.perceived = set()     # every event id this person saw or heard
        self.pending = []          # events of others perceived since the last choice
        self.insulted_by = set()
        self.opened_chest = False
        # used only when a flag is on (no random draws here, so the night is unchanged)
        self.fatigue, self.boredom, self.lonely, self.slight = 0.10, 0.30, 0.20, 0.10
        self.glimpsed, self.examined, self.visited, self.kinds_seen = set(), set(), {room}, set()
        self.deeds = []            # deeds this person SAW (and may tell of): dicts, newest last
        self.told = set()          # (hearer, deed key) already told
        self.rep = {}              # name -> sum of deed signs heard about them (gossip)
        self.liars = {}            # name -> times caught in a falsehood
        self.memory = []           # lasting records across nights (deaths, lies found out, thefts)
        self.requests = {}         # asker -> {"thing", "pay", "t"}
        self.doubted = set()       # told-beliefs already dropped after a failed search
        # used only under the `known` flag: what p perceived of the others
        self.seen_at = {}          # name -> room where p last saw or heard it (None: unknown, dead)
        self.seen_open = False     # p saw (or did) the old chest opened tonight
        self.hostile = {}          # name -> (perceived hostility toward p, turn it was set)
        self.partial = {}          # reprisal: (verb, target, obj) -> (turns done, turn interrupted)
        self.said = {}             # calm: option text -> turns it was done tonight
        self.said_kind = {}        # calm: (verb, object) -> turns
        self.ail = 0.0             # tend: the ailment level (0 = none); nothing in the old inn sets it
        self.last_kind = None      # labour: the kind of the last ration eaten tonight
        self.reserve = dict(LETHAL_RESERVE)   # lethal: days left before each need kills
        self.cause = None          # lethal: the need that killed it
        self.age = 30.0            # family: years (a body datum; the old inn never reads it)
        self.metab = self.work = self.strength = self.ill = self.ill_len = self.recov = 1.0    # family: the body sheet (FAMILY_BODY); 1.0 = the adult of the old inn
        self.work_acc = 0.0        # family: work at the counter is made in whole rations; the fraction carries
        self.tasted = None         # body2: the last thing consumed (what is on the tongue: the sense of taste reads the world's signed taste of it)

    @property
    def public(self):
        return KINDS[self.kind][0]

    def acts(self):
        return not (self.down or self.dead)

    def anger_up(self, who, amount):
        """Anger at one person rises, up to 1. Without the limit, two fighters
        grew so angry that neither could hit the other, for ever."""
        self.anger[who] = min(1.0, self.anger.get(who, 0) + amount)


class World:
    def __init__(self, seed=0, bandit=None, turns=300, flags=(), carry=None, night=0, supply=1.0, luck=0.3,
                 ill=AIL_P, raid=RAID_P, young=None):
        self.flags = frozenset(BASE_FLAGS if flags == "all" else FLAGS if flags == "all2" else
                               ALL3 if flags == "all3" else ALL4 if flags == "all4" else
                               ALL5 if flags == "all5" else ALL6 if flags == "all6" else flags)
        bad = self.flags - set(FLAGS) - set(NEW_FLAGS) - set(LUCK_FLAGS) - set(NEED_FLAGS) - set(LUMPY_FLAGS) - set(SIGNS_FLAGS) - set(HARD_FLAGS) - set(LETHAL_FLAGS) - set(LETHAL2_FLAGS) - set(FIELD_FLAGS) - set(FAMILY_FLAGS) - set(DRIVES2_FLAGS) - set(BODY2_FLAGS)
        assert not bad, f"unknown flags {sorted(bad)}"
        self.night = night
        self.full = PORTION if "field" in self.flags else FULL     # field: a meal is a whole ration, eaten only when the hunger left can take it (no partial meal is wasted)
        self.rng = random.Random(seed)
        rng = self.rng
        self.t = 0
        self.people = {}
        self.family = "family" in self.flags
        self.cast = FAMILY_CAST if self.family else CAST
        self.start_room = {n: room for n, _, room, _ in self.cast}
        for name, kind, room, held in self.cast:
            self.people[name] = Person(name, kind, room, rng)
            if self.family:
                b = {**FAMILY_ADULT, **FAMILY_BODY.get(kind, {})}
                q = self.people[name]
                q.age = float(FAMILY_AGE[name])
                if "body2" in self.flags:                      # body2: the library's sheets (elder work up)
                    b = {**b, **BODY2_SHEET.get(kind, {})}
                if young and name in young:                    # a childhood: this person is a child of that age (years): the child body sheet
                    b = {**FAMILY_ADULT, **FAMILY_BODY["child"]}
                    q.age = float(young[name])
                q.metab, q.work, q.strength, q.ill, q.ill_len, q.recov = (b[k] for k in ("metab", "work", "strength", "ill", "ill_len", "recov"))
        self.things = {}
        for name, (at, props, value) in THINGS.items():
            self.things[name] = {"at": at, "props": set(props), "value": value, "owner": None,
                                 "hidden": False}
        for name, kind, room, held in self.cast:
            for th in held:
                self.things[th].update(at=name, owner=name)
        for a in self.people.values():
            for b in self.people.values():
                if a is not b:
                    a.aff[b.name] = max(-1, min(1, rng.gauss(0.1, 0.25)))
        self.chest = None              # decided when first opened
        self.words = []                # wear: (turn, actor, target, verb) of the words said
        self.guard = {}                # person -> (defender, until)
        self.log = []
        self.n = 0
        # A bandit comes in from the road in about 60% of the nights.
        if bandit is None:
            bandit = rng.random() < (field_bandit_p(len(self.cast)) if "field" in self.flags else 0.6)
        self.bandit_at = rng.randint(turns // 3, 2 * turns // 3) if bandit else None
        if carry:                       # the same cast as last night: no random draw here
            for name, c in carry.items():
                p = self.people[name]
                p.base, p.traits = dict(c["base"]), dict(c["base"])
                p.aff.update(c["aff"])
                p.memory, p.rep, p.liars = c["memory"], dict(c["rep"]), dict(c["liars"])
                p.deeds, p.told = list(c["deeds"]), set(c["told"])
        if "trade" in self.flags:       # the cook holds the food and serves it
            cook = next(p.name for p in self.people.values() if p.kind == "cook")
            for n in ("the loaf of bread", "the bowl of stew"):
                self.things[n].update(at=cook, owner=cook)
        self.gone = set()              # perma: the dead of earlier nights (they stay dead)
        self.lethal = "lethal" in self.flags
        for name, c in (carry or {}).items():
            if c.get("gone") and "perma" in self.flags:        # perma: a death is final
                self.gone.add(name)
                self.people[name].dead = self.people[name].down = True
            elif self.lethal and "body" in c:                  # lethal: the body's state goes on from night to night (a revived person starts afresh)
                b = self.people[name]
                b.hunger, b.thirst, b.reserve = c["body"]["hunger"], c["body"]["thirst"], dict(c["body"]["reserve"])
        self.portions = {}             # meals: thing -> rations left
        if "meals" in self.flags:
            assert "trade" in self.flags, "meals needs trade (the cook holds the food)"
            cook = next(p.name for p in self.people.values() if p.kind == "cook")
            del self.things["the bowl of stew"]          # a pot to share instead of one bowl
            self.things["the pot of stew"] = {"at": cook, "props": {"food"}, "value": 0, "owner": cook,
                                              "hidden": False}
            self.portions = {n: max(1, round(k * supply)) for n, k in STOCK.items()}
        self.stock = dict(STOCK)       # the full size of each pot (what `left` is a share of)
        self.catch = {}                # luck: person -> rations they brought in tonight
        self.laid = {}                 # hard: person -> nights of the episode left, counting tonight (the laid-up are the keys)
        self.hard = "hard" in self.flags
        if self.hard:                  # the source rates replace the dials given
            assert "tend" in self.flags and "luck" in self.flags, "hard needs tend (ailments) and luck (the catch)"
            luck, raid = HARD_EMPTY, HARD_RAID
            if "field" in self.flags:  # the raid keeps its half of the violent-death budget (the bandit has the other half)
                raid = HARD_RAID * RAID_SHARE
            self._episodes(seed, carry)
        self.perish = {}               # lumpy: fresh item -> the turn it came in (it spoils from then on)
        self.rot = {}                  # lumpy: item -> the fraction of a ration spoiled and not yet taken off
        if "lumpy" in self.flags:
            assert "luck" in self.flags, "lumpy needs luck (the catch is what comes in packages)"
        self.kind_of = {}              # labour: food item -> its kind (the larder's are STAPLE)
        if "luck" in self.flags:
            assert "meals" in self.flags, "luck needs meals (a catch is eaten and shared by the portion)"
            self._luck(seed, supply, luck)
        if "stock" in self.flags:
            assert "meals" in self.flags, "stock needs meals (the counter is the house's pots)"
            house = next((c["house"] for c in (carry or {}).values() if "house" in c), None)
            if house is not None:        # the counter keeps what was left last night, and the cook's work adds to it
                for n, k in COOK_MAKES.items():
                    self.portions[n] = min(int(STOCK_CAP * self.stock[n]), house[n] + k)
        if "choosy" in self.flags:
            assert "partner" in self.flags, "choosy extends the partner rule"
        self.threat = None             # allies: the raid in the house now (a dict), or None
        self._raid_at = None
        if "tend" in self.flags and not self.hard:
            self._ailments(seed, ill)
        if "allies" in self.flags:
            self._raid_plan(seed, raid)
        if "labour" in self.flags:
            assert "luck" in self.flags, "labour needs luck (the catch is what is split)"
        self._sg_seed = seed
        if "signs" in self.flags:       # the sign dials: from their own generator (never the world's), kept for the lifetime by carry
            for p in self.people.values():
                r = random.Random(f"signs/{seed}/{p.name}")
                p.express = min(EXPR_HI, max(EXPR_LO, r.gauss(EXPR_MEAN, EXPR_SD)))
                p.acuity = min(ACU_HI, max(ACU_LO, r.gauss(ACU_MEAN, ACU_SD)))
            for name, c in (carry or {}).items():
                if "dials" in c:
                    self.people[name].express, self.people[name].acuity = c["dials"]
        self.start_at = {n: th["at"] for n, th in self.things.items()}   # the inn as each night sets it
        for p in self.people.values():
            self._look(p, "start")

    def _luck(self, seed, supply, v):
        """luck: the house pot shrinks to (1 - LUCK_SHARE) of its stock, and each person who is not the house brings a catch.
        The draws come from their own generator (seed only), so they draw nothing from the world's and are the same for every
        chooser that plays the same night. `v` is the chance of an empty night."""
        assert 0 <= v < 1
        for n in self.portions:
            self.portions[n] = max(1, round(STOCK[n] * supply * (1 - LUCK_SHARE)))
            self.stock[n] = self.portions[n]
        people = [p for p in self.people.values() if p.kind != "cook"]
        lumps = {}
        mean = LUCK_SHARE * sum(STOCK.values()) * supply / len(people)
        if "field" in self.flags:            # the mean catch that meets the demand (see FIELD above); the variance is the hard world's
            mean = field_catch_mean(v, n_prod=len(people), n=len(self.people), counter=sum(COOK_MAKES.values())) * supply
        rng = random.Random(f"luck/{seed}")
        fam_m = None
        if self.family and "field" in self.flags:    # family: supply = demand with each person's own ration (metab) and share of work: m per unit of work
            demand = NEED_RATIONS * sum(q.metab for q in self.people.values())
            fam_m = (demand - sum(COOK_MAKES.values())) / ((1.0 - HARD_PREVALENCE) * sum(q.work for q in people))
        for p in people:
            mean_p = mean if fam_m is None else field_catch_mean(v, target=fam_m * p.work) * supply
            if v == 0:
                got = mean_p
            else:
                got = 0.0 if rng.random() < v else rng.gammavariate(LUCK_SHAPE, 1 / LUCK_SHAPE) * mean_p / (1 - v)
            when = 0
            if "lumpy" in self.flags:        # a hit brings everything in one package, at a turn of the night (own generator)
                lr = lumps.setdefault("r", random.Random(f"lump/{seed}"))
                hit, when = lr.random() < (1 - v if self.hard else LUMP_P), lr.randint(LUMP_T0, LUMP_T1)
                got = mean_p / LUMP_P if hit else 0.0
                if self.hard:                # hard: the package is a Gamma(2) size with the mean kept (CV 1.37 with the 0.48 empty)
                    got = lr.gammavariate(LUCK_SHAPE, 1 / LUCK_SHAPE) * mean_p / (1 - v) if hit else 0.0
            if p.name in self.laid or p.name in self.gone:   # hard: a laid-up person brings nothing (nor does a dead one) (the draws above are still made: same streams)
                got = 0.0
            k = int(got + 0.5)
            self.catch[p.name] = k
            if "labour" in self.flags:       # the bold bring game, the careful greens, by the share of the dial
                g = int(k * p.base["bravery"] + 0.5)
                parts = [(f"{p.name}'s game", "game", g), (f"{p.name}'s greens", "greens", k - g)]
            else:
                parts = [(f"{p.name}'s catch", STAPLE, k)]
            for name, kind, n in parts:
                if n > 0:
                    self.things[name] = {"at": p.name, "props": {"food"}, "value": 0, "owner": p.name, "hidden": False}
                    if when:                 # lumpy: it is not in the house until it comes in
                        self.things[name]["at"] = None
                        self.perish[name] = when
                    self.portions[name] = n
                    self.stock[name] = n
                    self.kind_of[name] = kind

    def _ailments(self, seed, p_ill):
        """tend: each person starts the night with an ailment with chance p_ill, at a level drawn the same way for all.
        The draws come from their own generator (seed only), in the fixed order of the cast: nothing is taken from the
        world's own draws, and every chooser playing the same night gets the same ailments."""
        rng = random.Random(f"ailment/{seed}")
        for p in self.people.values():
            if rng.random() < p_ill:
                p.ail = rng.uniform(AIL_LO, AIL_HI)
            else:
                rng.random()

    def _episodes(self, seed, carry):
        """hard: who is laid up tonight. An episode in the carry goes on (its level of ailment is carried too); a healthy person falls ill
        with the onset chance, the length drawn from HARD_DUR. The draws come from their own generator per person and night (seed
        and name only, a fixed number each), so every chooser playing the same night meets the same ailments."""
        h = hard_onset()
        for p in self.people.values():
            r = random.Random(f"hard/{seed}/{p.name}")
            u0, u1, u2, z, lv = r.random(), r.random(), r.random(), r.gauss(0, 1), r.uniform(AIL_LO, AIL_HI)
            if p.name in self.gone:
                continue
            c = (carry or {}).get(p.name, {}).get("ill")
            if c and c[0] > 0:
                self.laid[p.name], p.ail = c[0], c[1]
            elif u0 < h * p.ill:
                self.laid[p.name], p.ail = max(1, int(hard_duration_draw(u1, u2, z) * p.ill_len + 0.5)), lv

    def _raid_plan(self, seed, p_raid):
        """allies: whether a raid comes tonight and when (own generator, seed only; the same for every chooser)."""
        self._rr = random.Random(f"raid/{seed}")
        if self._rr.random() < p_raid:
            self._raid_at = self._rr.randint(RAID_T0, RAID_T1)
        self._raid_kind = self._rr.choice(["the wolf", "a raider"])

    # ------------------------------------------------------------ helpers
    def here(self, p):
        return [q for q in self.people.values() if q is not p and q.room == p.room and not q.dead]

    def held(self, p, visible_only=False):
        return [n for n, th in self.things.items() if th["at"] == p.name
                and not (visible_only and th["hidden"])]

    def loose(self, room):
        return [n for n, th in self.things.items() if th["at"] == room]

    def weapon(self, p):
        w = [n for n in self.held(p) if n in WEAPON]
        return max(w, key=WEAPON.get) if w else None

    def danger(self, p):
        """How threatening the others here are to p: anger at p, a weapon,
        and violence in the last few turns. Under `known`: the hostility p
        perceived, a visibly armed bandit, violence p perceived."""
        if "known" in self.flags:
            return self._danger_known(p)
        d = 0.5 if self.threat and self.threat["room"] == p.room else 0.0     # allies: a raid in the room
        for q in self.here(p):
            if q.acts():
                d += max(0.0, q.anger.get(p.name, 0) - 0.2) + (0.3 if self.weapon(q) and q.kind == "bandit" else 0)
        recent = [e for e in self.log[-12:] if e["action"] in VIOLENT and e.get("done", True)
                  and self.people.get(e["actor"]) and self.people[e["actor"]].room == p.room]
        return min(1.0, d + 0.3 * len(recent))

    def recently_harmed(self, victim, turns=20, acts=None, knower=None):
        """Who harmed `victim` in the last turns (for defend, rescue). With a
        `knower` under `known`: only the harms the knower perceived or did."""
        acts = acts or HARMFUL
        return {e["actor"] for e in self.log[-120:] if e["t"] >= self.t - turns and e.get("target") == victim
                and e["action"] in acts and e.get("done", True) and self._knows(knower, e)}

    def _knows(self, p, e):
        """Under `known`: p perceived the event or did it. Else always true."""
        return p is None or "known" not in self.flags or e["actor"] == p.name or e["id"] in p.perceived

    def hostility(self, p, name):
        """How hostile p has SEEN `name` be toward it (known flag), 0..1."""
        h, t0 = p.hostile.get(name, (0.0, self.t))
        return h * 0.985 ** (self.t - t0)

    def _danger_known(self, p):
        d = 0.5 if self.threat and self.threat["room"] == p.room else 0.0
        for q in self.here(p):
            if q.acts():
                armed = any(n in WEAPON for n in self.held(q, visible_only=True))
                d += max(0.0, self.hostility(p, q.name) - 0.2) + (0.3 if armed and q.kind == "bandit" else 0)
        recent = [e for e in self.log[-12:] if e["action"] in VIOLENT and e.get("done", True) and self._knows(p, e)
                  and self.people.get(e["actor"]) and self.people[e["actor"]].room == p.room]
        return min(1.0, d + 0.3 * len(recent))

    def _believed_held(self, p, q):
        """What p sees q hold or believes q holds (known flag)."""
        seen = set(self.held(q, visible_only=True))
        return [n for n in self.things if n in seen or p.bel.get(("where", n), (None,))[0] == q.name]

    # ------------------------------------------------------------ flags: drives, trust, the lifetime
    @staticmethod
    def asleep(p):
        return bool(p.doing) and p.doing["verb"] == "sleep"

    def wealth(self, p):
        """What p holds of value. Under `known`, what p BELIEVES it holds:
        a loss counts once p notices it (at its next choice, _check_own)."""
        if "known" in self.flags:
            return sum(th["value"] for n, th in self.things.items()
                       if p.bel.get(("where", n), (None,))[0] == p.name)
        return sum(self.things[n]["value"] for n in self.held(p))

    def drives(self, p):
        """Each drive's need, 0..1 (0 = met). Read by a mind; written by no one
        but this file."""
        return {"energy": max(p.hunger, p.thirst),
                "safety": 1.0 if p.dead else max(p.fear, min(1.0, max(0.0, (10 - p.hp) / 10)), p.ail),
                "company": p.lonely, "standing": p.slight,
                "wealth": max(0.0, 1 - self.wealth(p) / WANT.get(p.kind, 10)),
                "rest": p.fatigue, "curiosity": p.boredom}

    def perceived_need(self, obs, tgt):
        """signs: what `obs` (a name) perceives of `tgt`'s hurt, fear and hunger, each 0..1, or None (flag off, not in the same room,
        the observer cannot act or sleeps, the target is dead). reading = expressiveness(tgt) * state + noise(sd = SIGN_NOISE /
        acuity(obs)), clipped. The noise is a function of (night seed, turn, observer, target), so a reading repeats and the world's
        own generator is untouched. HOOK (not built): hearsay, a reading from what was TOLD, would be added here."""
        if "signs" not in self.flags:
            return None
        o, t = self.people.get(obs), self.people.get(tgt)
        if o is None or t is None or t.dead or o is t or o.room != t.room or not o.acts() or self.asleep(o):
            return None
        state = {"hurt": max(min(1.0, max(0.0, (10 - t.hp) / 10)), t.ail), "fear": t.fear, "hunger": t.hunger}
        if "drives2" in self.flags:                              # thirst shows too (the last key: the other signs' noise draws are unchanged)
            state["thirst"] = t.thirst
        if "body2" in self.flags:                                # body2: the cry (graded, cause blind); an infant's hunger and thirst are not readable, only its cry
            state["cry"] = cry_level(t.hunger, t.thirst, state["hurt"], t.fear, t.age)
            if t.age < CRY_CAUSE_AGE:
                del state["hunger"]
                state.pop("thirst", None)
        r = random.Random(f"sign/{self._sg_seed}/{self.t}/{obs}/{tgt}")
        sd = SIGN_NOISE / getattr(o, "acuity", ACU_MEAN)
        ex = getattr(t, "express", EXPR_MEAN)
        return {k: min(1.0, max(0.0, ex * v + r.gauss(0.0, sd))) for k, v in state.items()}

    def drive_error(self, p, need=None):
        """The weighted error of p's drives, by its type's NATURE. A drop in it
        is the homeostatic reward (Keramati and Gutkin)."""
        need = need or self.drives(p)
        return sum(w * max(0.0, need[d] - s) for d, (w, s) in NATURE[p.kind].items())

    def distrust(self, p, name):
        """How much p distrusts `name`, from p's own memories: dislike, bad
        deeds heard of (gossip), falsehoods caught. 0..1."""
        return min(1.0, max(0.0, -p.aff.get(name, 0)) + 0.2 * max(0.0, -p.rep.get(name, 0))
                   + 0.5 * p.liars.get(name, 0))

    def carry_out(self, names=None):
        """What each cast member takes into the next night (lifetime.py)."""
        names = names or [n for n, _, _, _ in self.cast]
        cast = set(n for n, _, _, _ in self.cast)
        out = {}
        for n in names:
            p = self.people[n]
            deeds = [d for d in p.deeds if d["actor"] in cast and d["target"] in cast][-6:]
            keys = {d["key"] for d in deeds}
            out[n] = {"base": dict(p.base), "aff": {k: v for k, v in p.aff.items() if k in cast},
                      "memory": p.memory, "rep": {k: v for k, v in p.rep.items() if k in cast},
                      "liars": {k: v for k, v in p.liars.items() if k in cast}, "deeds": deeds,
                      "told": sorted(t for t in p.told if t[0] in cast and t[1] in keys)}
            if "signs" in self.flags:                            # the sign dials are the person's own for the whole life
                out[n]["dials"] = (p.express, p.acuity)
            if self.lethal and not p.dead:                       # the body goes on (a person who died comes back afresh, or not at all under perma)
                out[n]["body"] = {"hunger": p.hunger, "thirst": p.thirst, "reserve": dict(p.reserve)}
            if p.dead and "perma" in self.flags:                 # a death is final under perma
                out[n]["gone"] = True
            if self.hard:                                        # an episode goes on into the next night (nights left after tonight)
                out[n]["ill"] = (max(0, self.laid.get(n, 0) - 1), p.ail)
            if "stock" in self.flags and p.kind == "cook":      # the counter's stock goes into the next night with the cook
                out[n]["house"] = {k: self.portions[k] for k in COOK_MAKES}
        return out

    def _rested(self, p, amount):
        p.fatigue = min(1.0, max(0.0, p.fatigue - amount))

    def _novel(self, p, amount):
        p.boredom = max(0.0, p.boredom - amount)

    # ------------------------------------------------------------ options
    def options(self, p):
        """Every action the world permits p now, with the numbers the rule
        policy and the trait bias read (`f`)."""
        W = self.things
        out = []

        tcache = {}

        def add(verb, text, target=None, obj=None, **f):
            if partner and target in self.people and "trust" not in f:
                if target not in tcache:
                    tcache[target] = self.trust(p, target)
                f["trust"] = tcache[target]
            dur = DUR[verb]
            if reprisal and verb in HARMFUL and target in self.people:
                dur = max(dur, HARM_MIN)
            out.append({"verb": verb, "text": text, "target": target, "obj": obj, "f": f,
                        "dur": dur})
            return out[-1]

        danger = self.danger(p)
        others = self.here(p)
        # p is under suspicion: accused or searched lately
        suspected = float(any(e["target"] == p.name and e["action"] in ("accuse", "search") and e["done"]
                              for e in self.log[-80:] if e["t"] >= self.t - 40 and self._knows(p, e)))
        known = "known" in self.flags
        reprisal = "reprisal" in self.flags
        partner = "partner" in self.flags
        for r in ROOMS[p.room]:
            add("walk to", f"walk to {r}", target=r, goal=self._goal(p, r))
            if danger > 0.2:
                add("flee to", f"flee to {r}", target=r, danger=danger)
        for n in self.loose(p.room):
            th = W[n]
            if "fixed" in th["props"]:
                continue
            add("take", f"take {n}", obj=n, gain=th["value"] / 10, weapon=WEAPON.get(n, 0) / 4,
                theft=float(th["owner"] not in (None, p.name)), mine=float(th["owner"] == p.name),
                witnesses=len([q for q in others if q.acts()]))
        mine = self.held(p)
        for n in mine:
            th = W[n]
            if "food" in th["props"] and not (n in self.portions and p.hunger <= self.full):   # a full stomach holds no more
                add("eat", f"eat a portion of {n}" if n in self.portions else f"eat {n}", obj=n, need=p.hunger * self._eff(p, n))
            if "ale" in th["props"]:
                add("drink", f"drink {n}", obj=n, need=p.thirst, ale=1.0)
            if "water" in th["props"]:
                add("drink", f"drink from {n}", obj=n, need=p.thirst)
            if "music" in th["props"]:
                add("play", f"play {n}", obj=n)
            add("drop", f"drop {n}", obj=n)
            stolen = th["owner"] not in (None, p.name)
            if stolen and not th["hidden"]:
                add("hide", f"hide {n} under your cloak", obj=n, stolen=1.0)
            if n in self.portions:                  # meals: a pot is shared by the portion (serve), not handed over
                continue
            for q in others:
                if q.acts():
                    add("give", f"give {n} to {q.name}", target=q.name, obj=n,
                        gain=th["value"] / 10, returning=float(th["owner"] == q.name),
                        like=max(0, p.aff.get(q.name, 0)))
                    if "trade" in self.flags and p.requests.get(q.name, {}).get("thing") == n \
                            and p.requests[q.name]["pay"] is None:
                        out[-1]["f"].update(asked=1.0, distrust=self.distrust(p, q.name))
        for n in self.loose(p.room):
            props = W[n]["props"]
            if "ale_source" in props:
                add("drink", f"drink from {n}", obj=n, need=p.thirst, ale=1.0)
            if "water_source" in props:
                add("drink", f"drink from {n}", obj=n, need=p.thirst)
            if "chest" in props and not p.opened_chest:
                add("open", f"open {n}", obj=n,
                    first=float(not p.seen_open) if known else float(self.chest is None))
        w = self.weapon(p)
        for q in others:
            harmed = self.recently_harmed(q.name, acts=PHYSICAL, knower=p)
            f = dict(anger=p.anger.get(q.name, 0), dislike=max(0, -p.aff.get(q.name, 0)),
                     like=max(0, p.aff.get(q.name, 0)), down=float(q.down),
                     threat=float(self._is_threat(p, q)), danger=danger,
                     rich=float(any(W[n]["value"] >= 5 for n in self.held(q, visible_only=True))),
                     insulted=float(q.name in p.insulted_by))
            if q.acts():
                if w:
                    add("strike", f"strike {q.name} with {w}", target=q.name, obj=w, armed=1.0, **f)
                add("strike", f"strike {q.name}", target=q.name, armed=0.0, **f)
                add("shove", f"shove {q.name}", target=q.name, **f)
                add("threaten", f"threaten {q.name}", target=q.name, **f)
                add("insult", f"insult {q.name}", target=q.name, **f)
                add("praise", f"praise {q.name}", target=q.name, **f)
                add("search", f"search {q.name}", target=q.name,
                    suspect=float(self._suspects(p, q)), **f)
            else:                                   # lying on the floor: its things can be taken
                for n in (self._believed_held(p, q) if known else self.held(q)):
                    add("steal", f"take {n} from {q.name}", target=q.name, obj=n, gain=W[n]["value"] / 10,
                        witnesses=len([o for o in others if o.acts() and o is not q]), **f)
            if q.down or q.hp < 7:
                add("bandage", f"bandage {q.name}", target=q.name, need=1.0 if q.down else 0.5, **f)
            if q.down and not q.dead:
                add("help up", f"help {q.name} up", target=q.name, need=1.0, **f)
            if q.acts() and harmed - {p.name}:
                add("defend", f"defend {q.name}", target=q.name, attacked=1.0, **f)
            if q.acts():
                sleeping = "drives" in self.flags and self.asleep(q)     # a sleeper's things can all be taken
                for n in ((self._believed_held(p, q) if known else self.held(q)) if sleeping
                          else self.held(q, visible_only=True)):
                    if sleeping or (W[n]["value"] >= 1 and W[n]["owner"] == q.name):
                        add("steal", f"steal {n} from {q.name}", target=q.name, obj=n,
                            gain=W[n]["value"] / 10,
                            witnesses=len([o for o in others if o.acts() and o is not q]), **f)
            # what p believes about thefts: accuse, tell the truth, or lie
            for (kind, thing), (culprit, _src) in list(p.bel.items()):
                if kind != "took" or culprit == p.name:
                    continue
                if culprit == q.name:
                    extra = {}
                    if "calm" in self.flags:     # a charge about a thing p no longer thinks the culprit holds is moot
                        extra["moot"] = float(p.bel.get(("where", thing), (None,))[0] != q.name)
                    add("accuse", f"accuse {q.name} of taking {thing}", target=q.name, obj=thing,
                        mine=float(W[thing]["owner"] == p.name), **extra, **f)
                elif q.acts():
                    add("tell", f"tell {q.name} that {culprit} took {thing}", target=q.name, obj=thing,
                        truth=1.0, useful=float(W[thing]["owner"] == q.name),
                        dislike_culprit=max(0, -p.aff.get(culprit, 0)), said=culprit, **f)
            # whoever holds a stolen thing can blame someone else for it: a lie
            if q.acts():
                for n in mine:
                    if W[n]["owner"] in (None, p.name):
                        continue
                    thinks = p.bel.get(("took", n), (None,))[0]        # blaming whom p believes took it is not a lie
                    blame = [g for g in self.here(p) if g is not q and g.name not in (W[n]["owner"], thinks)]
                    for g in sorted(blame, key=lambda o: p.aff.get(o.name, 0))[:2]:
                        add("tell", f"tell {q.name} that {g.name} took {n}", target=q.name, obj=n,
                            lie=1.0, dislike_scapegoat=max(0, -p.aff.get(g.name, 0)), said=g.name,
                            suspected=suspected, owner=float(W[n]["owner"] == q.name), **f)
        if self.flags:
            self._flag_options(p, add, others, danger)
        if danger > 0.2 or any(self.recently_harmed(q.name, 6, knower=p) for q in others):
            add("shout for help", "shout for help", danger=danger)
        add("wait", "wait")
        # the same text twice would be the same option: keep the first
        seen, uniq = set(), []
        mine_ev = [e for e in self.log[-80:] if e["actor"] == p.name and e["done"] and e["t"] >= self.t - 40]
        done = [e["text"] for e in mine_ev]
        done_kinds = [(e["action"], e["parts"].get("thing")) for e in mine_ev]
        last = mine_ev[-1] if mine_ev and mine_ev[-1]["t"] >= self.t - 10 else None
        undo = {"take": ("drop",), "steal": ("drop",), "drop": ("take",)}
        calm = "calm" in self.flags
        barred = LAID_UP_BARRED if p.name in self.laid else ()
        for o in out:
            if o["verb"] in barred:          # hard: a laid-up person cannot do what exerts the body, work, or reach the counter
                continue
            if o["text"] not in seen:
                seen.add(o["text"])
                # habit: undoing what one just did (drop right after take) pulls less;
                # without it the model policies looped take, drop, take
                o["f"]["undo"] = float(bool(last) and o["verb"] in undo.get(last["action"], ())
                                       and o["obj"] == last["parts"].get("thing"))
                o["f"]["repeat"] = float(done.count(o["text"]))     # habit: the same act again pulls less
                o["f"]["repeat_kind"] = float(sum(1 for (v_, ob) in done_kinds if v_ == o["verb"] and ob == o["obj"]))
                if calm:                    # calm: one remembers what one said all night, fading slowly
                    o["f"]["repeat"] = self._echo(p.said.get(o["text"]))
                    o["f"]["repeat_kind"] = self._echo(p.said_kind.get((o["verb"], o["obj"]))) + \
                        (self._echo(p.said_kind.get((o["verb"], o["target"]))) if o["target"] else 0.0)
                uniq.append(o)
        return uniq

    def _flag_options(self, p, add, others, danger):
        """The menu acts the flags add. Facts only: whether to take them is
        the chooser's business."""
        W = self.things
        awake = [q for q in others if q.acts() and not self.asleep(q)]
        if "drives" in self.flags:
            add("sit down", "sit down", fatigue=p.fatigue, danger=danger)
            add("sleep", "sleep", fatigue=p.fatigue, danger=danger, others=float(len(awake)),
                distrust=max([self.distrust(p, q.name) for q in awake] + [0.0]))
            new = [n for n in self.loose(p.room) if n not in p.examined] + \
                  [q.name for q in others if q.name not in p.examined]
            room_things = len(self.loose(p.room)) + len(others)
            add("look around", "look around", unseen=len(new) / max(1, room_things), boredom=p.boredom)
        if "gossip" in self.flags:
            for q in awake:
                fresh = [d for d in reversed(p.deeds)
                         if not (q.name in (d["actor"], d["target"]) or (q.name, d["key"]) in p.told)]
                news = "news" in self.flags
                if news:        # the most salient deeds, whatever their sign (a stable sort: ties keep the newest first)
                    fresh.sort(key=lambda d: -self._salience(p, d))
                for d in fresh[:3]:
                    sign = DEED[d["action"]]
                    extra = {"sal": self._salience(p, d)} if news else {}
                    add("gossip", f"tell {q.name} that {d['actor']} {PAST[d['action']]} {d['target']}",
                        target=q.name, deed=d, bad=float(sign < 0), good=float(sign > 0),
                        dislike_actor=max(0, -p.aff.get(d["actor"], 0)), like_actor=max(0, p.aff.get(d["actor"], 0)),
                        like=max(0, p.aff.get(q.name, 0)), dislike=max(0, -p.aff.get(q.name, 0)), **extra)
        if "trade" in self.flags:
            mine = [n for n in self.held(p) if W[n]["value"] >= 1]
            for q in awake:
                dq = self.distrust(p, q.name)
                like = max(0, p.aff.get(q.name, 0))
                for n in self.held(q, visible_only=True):
                    props = W[n]["props"]
                    food, drink = float("food" in props), float(bool(props & {"ale", "water"}))
                    need = max(food * p.hunger * self._eff(p, n), drink * p.thirst)
                    if n in self.portions:       # meals: one asks for a portion; the counter sells (buy, work)
                        add("ask", f"ask {q.name} for a portion of {n}", target=q.name, obj=n, need=need,
                            gain=0.0, like=like, distrust=dq)
                        continue
                    if food or drink or W[n]["value"] >= 1:
                        add("ask", f"ask {q.name} for {n}", target=q.name, obj=n, need=need,
                            gain=W[n]["value"] / 10, like=like, distrust=dq)
                    for m in mine:
                        if m != n:
                            add("offer", f"offer {q.name} {m} for {n}", target=q.name, obj=n, pay=m, need=need,
                                gain=W[n]["value"] / 10, cost=W[m]["value"] / 10, like=like, distrust=dq)
                # the answers to what q asked of p (giving is the "give" option)
                r = p.requests.get(q.name)
                if r and r["thing"] in self.held(p):
                    if r["pay"] is not None and r["pay"] in self.held(q):
                        add("accept", f"accept {q.name}'s {r['pay']} for {r['thing']}", target=q.name,
                            obj=r["thing"], pay=r["pay"], gain=W[r["thing"]]["value"] / 10,
                            get=W[r["pay"]]["value"] / 10, like=like, distrust=dq)
                    add("refuse", f"refuse {q.name}", target=q.name, obj=r["thing"], like=like,
                        dislike=max(0, -p.aff.get(q.name, 0)), distrust=dq)

        if "meals" in self.flags:
            self._meal_options(p, add, awake)
        if "bring" in self.flags:
            self._bring_options(p, add, awake)
        if "tend" in self.flags:             # tend the sick; ask to be tended (an ailment cannot be treated by its owner)
            for q in others:
                if q.ail > TEND_MIN:
                    r = p.requests.get(q.name, {})
                    add("tend", f"tend {q.name}", target=q.name, need=q.ail, asked=float(r.get("thing") == "care"),
                        like=max(0, p.aff.get(q.name, 0)), anger=p.anger.get(q.name, 0), distrust=self.distrust(p, q.name))
            if p.ail > TEND_MIN:
                for q in awake:
                    add("ask", f"ask {q.name} to tend you", target=q.name, obj="care", need=p.ail, gain=0.0,
                        like=max(0, p.aff.get(q.name, 0)), distrust=self.distrust(p, q.name))
        th = self.threat
        if th and th["room"] == p.room and p.acts():     # allies: fend off the raid in the room, alone or with those who do
            fighters = [q for q in awake if q.doing and q.doing["verb"] == "fend off"]
            hit = any(e["action"] == "raid hits" and e["target"] == p.name for e in self.log[-12:])
            add("fend off", f"fend off {th['name']}", obj=th["name"], allies=float(len(fighters)),
                here=float(len(awake)), hit=float(hit), hp_left=th["hp"] / RAID_HP)

    def _distance(self, a, b):
        """Rooms between a and b (breadth first); None if not joined."""
        seen, edge, d = {a}, [a], 0
        while edge:
            if b in edge:
                return d
            nxt = []
            for x in edge:
                for r in ROOMS[x]:
                    if r not in seen:
                        seen.add(r)
                        nxt.append(r)
            edge, d = nxt, d + 1
        return None

    def _bring_options(self, p, add, awake):
        """bring: ONE act for any consumable a person cannot fetch (a drink; a portion is `serve` in _meal_options). The carer fetches a drink from a
        source (the well, the barrel) or a loose drink and gives it: it costs the serve act plus the walk there and back. One cannot carry what one
        cannot walk to (laid up). The cannot-fetch person asks the same way (`ask`)."""
        src = []
        for n, th in self.things.items():
            if th["at"] in ROOMS and th["props"] & DRINK_PROPS:
                d = self._distance(p.room, th["at"])
                if d is not None:
                    src.append((n, d))
        laid = p.name in self.laid
        for n, d in src:
            if d and laid:
                continue
            for q in awake:
                dep = self.dependent(q)
                if dep:                                     # body2: a dependent is brought WATER when it cries (cause blind: whether or not it is thirst)
                    if "ale" in self.things[n]["props"] or "ale_source" in self.things[n]["props"] or self.seen_need_of(q, q.thirst) <= 0.0:
                        continue
                elif q.name not in self.laid or q.thirst <= FULL:
                    continue
                r = p.requests.get(q.name, {})
                o = add("serve", f"bring {q.name} a drink from {n}", target=q.name, obj=n,
                        asked=float(r.get("thing") == n and r.get("pay") is None), seen_need=self.seen_need_of(q, q.thirst) if dep else (q.thirst if q.thirst > 0.6 else 0.0),
                        own=p.thirst, left=1.0, like=max(0, p.aff.get(q.name, 0)), distrust=self.distrust(p, q.name))
                o["dur"] = DUR["serve"] + 2 * d * DUR["walk to"]
        if laid and p.thirst > FULL:
            for n, d in src:
                for q in awake:
                    add("ask", f"ask {q.name} for a drink from {n}", target=q.name, obj=n, need=p.thirst, gain=0.0,
                        like=max(0, p.aff.get(q.name, 0)), distrust=self.distrust(p, q.name))

    def _meal_options(self, p, add, awake):
        """meals: serve a portion from a pot p holds; buy a meal or work for one at a pot another holds."""
        W = self.things
        for n in self.held(p):
            if n not in self.portions or self.portions[n] <= 0:
                continue
            left = min(1.0, self.portions[n] / self.stock[n])
            extra = {"spare": self.spare(p, n)} if "lumpy" in self.flags else {}
            for q in awake:
                r = p.requests.get(q.name, {})
                asked = float(r.get("thing") == n and r.get("pay") is None)
                add("serve", f"serve {q.name} a portion of {n}", target=q.name, obj=n, asked=asked,
                    seen_need=self.seen_need_of(q, q.hunger), own=p.hunger, left=left,
                    like=max(0, p.aff.get(q.name, 0)), distrust=self.distrust(p, q.name), **extra)
        pay = [m for m in self.held(p) if PAYABLE[0] <= W[m]["value"] <= PAYABLE[1] and m not in self.portions]
        for q in awake:
            for n in self.held(q, visible_only=True):
                if n not in self.portions or self.portions[n] <= 0:
                    continue
                dq = self.distrust(p, q.name)
                for m in pay:
                    add("buy", f"pay {q.name} {m} for a meal from {n}", target=q.name, obj=n, pay=m, need=p.hunger * self._eff(p, n),
                        cost=W[m]["value"] / 10, distrust=dq)
                add("work", f"work for {q.name} for a share of {n}", target=q.name, obj=n, need=p.hunger * self._eff(p, n),
                    fatigue=p.fatigue, distrust=dq)

    def _goal(self, p, room):
        """What a room offers p: food, drink, whoever p is angry at, loot, safety."""
        if "known" in self.flags:
            return self._goal_known(p, room)
        things = [n for n, th in self.things.items() if th["at"] == room]
        people = [q for q in self.people.values() if q.room == room and q is not p and not q.dead]
        food = any("food" in self.things[n]["props"] for n in things)
        drink = any(self.things[n]["props"] & {"ale", "ale_source", "water_source", "water"} for n in things)
        foe = max([p.anger.get(q.name, 0) for q in people] + [0])
        loot = any(self.things[n]["value"] >= 5 for n in self.things
                   if self.things[n]["at"] in [q.name for q in people])
        safe = 1.0 - min(1.0, sum(max(0, q.anger.get(p.name, 0) - 0.2) for q in people))
        company = min(1.0, len(people) / 3)
        g = dict(food=float(food), drink=float(drink), foe=foe, loot=float(loot), safe=safe,
                 company=company, chest=float(room == "the cellar" and self.chest is None))
        if "drives" in self.flags:
            g["new"] = float(room not in p.visited)
        return g

    def _goal_known(self, p, room):
        """The same, from what p knows: where it last saw each thing (fixed
        things, and anything it has not seen tonight: where the night put it),
        where it last saw or heard each person (else where the cast starts),
        and the hostility it perceived."""
        def at(n):
            b = p.bel.get(("where", n))
            return b[0] if b else self.start_at.get(n)
        things = [n for n in self.things if at(n) == room]
        people = [q for q in self.people.values() if q is not p
                  and p.seen_at.get(q.name, self.start_room.get(q.name)) == room]
        food = any("food" in self.things[n]["props"] for n in things)
        drink = any(self.things[n]["props"] & {"ale", "ale_source", "water_source", "water"} for n in things)
        foe = max([p.anger.get(q.name, 0) for q in people] + [0])
        names = {q.name for q in people}
        loot = any(self.things[n]["value"] >= 5 for n in self.things if at(n) in names)
        safe = 1.0 - min(1.0, sum(max(0, self.hostility(p, q.name) - 0.2) for q in people))
        g = dict(food=float(food), drink=float(drink), foe=foe, loot=float(loot), safe=safe,
                 company=min(1.0, len(people) / 3),
                 chest=float(room == "the cellar" and not (p.seen_open or p.opened_chest)))
        if "drives" in self.flags:
            g["new"] = float(room not in p.visited)
        return g

    def _is_threat(self, p, q):
        """q struck, shoved or threatened p, or someone p likes, lately; or q
        is a bandit. Words (an accusation, an insult) do not make a threat."""
        if q.kind == "bandit":
            return True
        for e in self.log[-40:]:
            if e["actor"] == q.name and e["action"] in PHYSICAL and e.get("done", True) and self._knows(p, e):
                v = e.get("target")
                if v == p.name or (v in p.aff and p.aff[v] > 0.2):
                    return True
        return False

    def _suspects(self, p, q):
        if "calm" in self.flags:         # still suspects q only of what it believes q still holds
            return any(k == "took" and v == q.name and p.bel.get(("where", th), (None,))[0] == q.name
                       for (k, th), (v, _s) in p.bel.items())
        return any(k == "took" and v == q.name for (k, _), (v, _s) in p.bel.items())

    # ------------------------------------------------------------ the turn
    def step(self, policy):
        self.t += 1
        if self.bandit_at == self.t:
            self._bandit_arrives()
        if self._raid_at == self.t:
            self._raid_arrives()
        if self.perish:
            self._spoil()
        if self.threat and self.t > self.threat["t0"] and (self.t - self.threat["t0"]) % RAID_ROUND == 0:
            self._raid_round()
        for p in self.people.values():
            self._drift(p)
        order = list(self.people.values())
        self.rng.shuffle(order)
        for p in order:
            if not p.acts():
                p.doing = None
                continue
            if p.doing is None:
                self._choose(p, policy)
            else:
                p.doing["left"] -= 1
                if p.doing["left"] <= 0:
                    self._finish(p)

    def _spoil(self):
        """lumpy: a package comes in at its turn, and from then on its rations spoil by a half-life (no random draw)."""
        for n, t0 in self.perish.items():
            if t0 > self.t or self.portions.get(n, 0) <= 0:
                continue
            th = self.things[n]
            if t0 == self.t:
                th["at"] = th["owner"]
                continue
            if th["at"] is None:
                continue
            self.rot[n] = self.rot.get(n, 0.0) + self.portions[n] * SPOIL_RATE
            k = min(int(self.rot[n]), self.portions[n])
            if k >= 1:
                self.rot[n] -= k
                self.stock[n] = self.stock[n] * (self.portions[n] - k) / self.portions[n]    # `left` is a share: spoilage does not change it
                self.portions[n] -= k
                if self.portions[n] <= 0:
                    th["at"] = None

    def spare(self, p, item):
        """lumpy: the share of what is left of `item` that its holder could not eat before it spoils (0 for what keeps)."""
        if item not in self.perish or self.portions.get(item, 0) <= 0:
            return 0.0
        usable = (max(0.0, p.hunger - self.full) + 0.003 * SPOIL_HALF) / PORTION
        return max(0.0, 1.0 - usable / self.portions[item])

    def _drift(self, p):
        p.hunger = min(1.0, p.hunger + 0.003 * p.metab)
        p.thirst = min(1.0, p.thirst + 0.004)
        p.fear *= 0.98
        p.drunk *= 0.995
        for k in list(p.anger):
            p.anger[k] *= 0.985
        g = self.guard.get(p.name)
        if g and g[1] < self.t:
            del self.guard[p.name]
        if "drives" in self.flags and not p.dead:
            p.fatigue = min(1.0, p.fatigue + RATE["rest"])
            p.boredom = min(1.0, p.boredom + RATE["curiosity"])
            liked = sum(1 for q in self.here(p) if q.acts() and p.aff.get(q.name, 0) > 0.1)
            p.lonely = max(0.0, p.lonely - RATE["company_met"] * min(2, liked)) if liked else \
                min(1.0, p.lonely + RATE["company"])
            p.slight *= RATE["standing_fade"]
        if p.ail > 0 and not p.dead:     # tend: left alone it grows
            p.ail = min(1.0, p.ail + AIL_GROW)
        if self.lethal and not p.dead:   # lethal: a need at its extreme draws down its reserve; away from it the reserve comes back
            dt, m = 1.0 / LETHAL_TURNS, 1.0 + p.ail
            for k, lvl in (("thirst", p.thirst), ("hunger", p.hunger)):
                if lvl >= LETHAL_EXTREME:
                    p.reserve[k] -= m * dt
                    if p.reserve[k] <= 0:
                        p.dead = p.down = True
                        p.doing, p.cause = None, k
                        break
                else:
                    back = LETHAL_RESERVE[k] / REPAY_TURNS if (k == "thirst" and "body2" in self.flags) else dt     # body2: water is repaid in about 20 minutes
                    p.reserve[k] = min(LETHAL_RESERVE[k], p.reserve[k] + back)
        if p.requests:                  # a request not answered in 30 turns lapses
            p.requests = {k: r for k, r in p.requests.items() if r["t"] >= self.t - 30}

    def _choose(self, p, policy):
        # a trait is a range: this choice's traits vary around the character's own
        p.traits = {t: min(1, max(0, self.rng.gauss(p.base[t], STATE_SPREAD))) for t in TRAITS}
        self._look(p, "sight")
        self._check_own(p)
        opts = self.options(p)
        pc, pa = policy.probs(self, p, opts)
        i = self.rng.choices(range(len(opts)), weights=pc)[0]
        o = opts[i]
        left = o["dur"]
        if "reprisal" in self.flags:             # an interrupted act is taken up where it stopped
            done_, t_ = p.partial.pop((o["verb"], o["target"], o["obj"]), (0, -RESUME_SPAN - 1))
            if self.t - t_ <= RESUME_SPAN:
                left = max(1, left - done_)
        p.doing = {**o, "left": left, "choice": {
            "options": [x["text"] for x in opts], "chosen": i,
            "p_char": [round(x, 5) for x in pc], "p_avg": [round(x, 5) for x in pa],
            "seen": list(p.pending), "traits": {k: round(v, 3) for k, v in p.traits.items()}}}
        p.pending = []
        if left <= 1:
            self._finish(p)

    def _finish(self, p):
        d, p.doing = p.doing, None
        ok, info = self._apply(p, d)
        ev = self._emit(p, d, done=ok, **info)
        if "calm" in self.flags and ok:
            p.said.setdefault(d["text"], []).append(self.t)
            p.said_kind.setdefault((d["verb"], d["obj"]), []).append(self.t)
            if d["target"]:
                p.said_kind.setdefault((d["verb"], d["target"]), []).append(self.t)
        if self.flags and ok:
            self._after(p, d, ev)

    def interrupt(self, p, by):
        if p.doing and p.acts():
            d, p.doing = p.doing, None
            if d["verb"] == "sleep":    # woken: the turns slept still rested it
                self._rested(p, 0.05 * (d["dur"] - d["left"]))
            elif "reprisal" in self.flags and d["dur"] - d["left"] > 0:
                p.partial[(d["verb"], d["target"], d["obj"])] = (d["dur"] - d["left"], self.t)
            self._emit(p, d, done=False, why=f"stopped by {by}")

    # ------------------------------------------------------------ effects
    def _apply(self, p, d):
        """Re-check that the action is still possible, then do it. Returns
        (done, extra fields for the log)."""
        v, W = d["verb"], self.things
        q = self.people.get(d["target"]) if d["target"] in self.people else None
        if q is not None and (q.room != p.room or q.dead):
            return False, {"why": f"{q.name} is not here"}
        o = d["obj"]
        if v in ("walk to", "flee to"):
            p.room = d["target"]
            return True, {}
        if v == "take":
            if W[o]["at"] != p.room:
                return False, {"why": "it is gone"}
            W[o]["at"] = p.name
            return True, {}
        if v == "steal":
            if W[o]["at"] != q.name:
                return False, {"why": "it is gone"}
            W[o].update(at=p.name, hidden=True)
            return True, {"sneaky": True}
        if v == "drop":
            W[o].update(at=p.room, hidden=False)
            return True, {}
        if v == "give":
            if W[o]["at"] != p.name:
                return False, {"why": "not held"}
            W[o].update(at=q.name, hidden=False)
            if W[o]["owner"] == p.name:
                W[o]["owner"] = q.name
            q.aff[p.name] = min(1, q.aff.get(p.name, 0) + 0.3 + 0.3 * (W[o]["owner"] == q.name))
            q.anger[p.name] = max(0, q.anger.get(p.name, 0) - 0.3)
            return True, {}
        if v == "eat":
            if W[o]["at"] != p.name:
                return False, {"why": "not held"}
            if o in self.portions:                       # meals: one portion of the pot
                if not self._feed(p, 1, o):
                    return False, {"why": "none left"}
                return True, {}
            p.hunger = max(0, p.hunger - 0.6)
            W[o]["at"] = None
            return True, {}
        if v == "drink":
            self._sip(p, o, d["f"].get("ale"))
            return True, {}
        if v == "strike":
            victim = q
            dfd = self.guard.get(q.name)
            if dfd and dfd[0] in self.people and self.people[dfd[0]].room == p.room and dfd[0] != p.name:
                victim = self.people[dfd[0]]              # the defender is in the way
            anger = p.anger.get(q.name, 0)
            aim = max(0.3, 0.75 - 0.25 * anger - 0.2 * p.drunk + (0.2 if victim.down else 0)
                      - (0.25 * max(0.0, p.fatigue - 0.6) if "drives" in self.flags else 0)
                      + (0.2 if "drives" in self.flags and self.asleep(victim) else 0))
            if self.rng.random() > aim:
                return True, {"result": "missed"}
            dmg = WEAPON.get(o, 1) if d["f"].get("armed") else 1
            dmg *= (1 + 0.5 * anger) * self.rng.uniform(0.3, 0.8)
            dmg *= p.strength
            victim.hp -= dmg
            victim.fear = min(1, victim.fear + 0.15 * (1.2 - victim.traits["bravery"]))
            res = {"result": f"hit {victim.name}", "damage": round(dmg, 1)}
            if victim is not q:
                res["blocked_by"] = victim.name
            if victim.hp <= 4 and not victim.down:
                self.interrupt(victim, "a blow")       # its unfinished action goes in the log
                victim.down = True
                res["result"] = f"knocked {victim.name} down"
            if victim.hp <= -2:                         # killing takes more than one blow
                victim.dead = victim.down = True
                res["result"] = f"killed {victim.name}"
            return True, res
        if v == "shove":
            if self.rng.random() < 0.3 and not q.down:
                q.down = q.hp <= 5
            return True, {}
        if v in ("threaten", "insult", "praise", "accuse", "shout for help", "wait", "play"):
            if v == "threaten":
                q.fear = min(1, q.fear + 0.35 * (1.2 - q.traits["bravery"]))
            if v == "insult":
                q.insulted_by.add(p.name)
            if v == "praise":
                q.aff[p.name] = min(1, q.aff.get(p.name, 0) + 0.15 * self._wear(p.name, q.name, v))
            return True, {}
        if v == "tell":
            return True, {}                                  # beliefs change on hearing (_perceive)
        if v in ("help up", "bandage"):
            if v == "bandage":
                q.hp = min(10, q.hp + 4)
            if not q.dead:
                q.down = False if v == "help up" or q.hp > 4 else q.down
                q.hp = max(q.hp, 1.0) if v == "help up" else q.hp
            q.aff[p.name] = min(1, q.aff.get(p.name, 0) + 0.4)
            return True, {}
        if v == "defend":
            self.guard[q.name] = (p.name, self.t + 8)
            return True, {}
        if v == "hide":
            W[o]["hidden"] = True
            return True, {}
        if v == "search":
            found = [n for n in self.held(q) if W[n]["hidden"]]
            for n in found:
                W[n]["hidden"] = False
            return True, {"found": found}
        if v == "open":
            p.opened_chest = True
            if self.chest is None:                           # made when needed
                names, _, weights = zip(*CHEST)
                self.chest = self.rng.choices(names, weights=weights)[0]
            val = dict((n, v_) for n, v_, _ in CHEST)[self.chest]
            if val >= 5 and "some old silver coins" not in W:
                W["some old silver coins"] = {"at": p.name, "props": set(), "value": val,
                                              "owner": None, "hidden": False}
            return True, {"found": self.chest}
        if self.flags:
            return self._apply_flag(p, d, q)
        raise ValueError(v)

    def _apply_flag(self, p, d, q):
        v, W, o = d["verb"], self.things, d["obj"]
        if v == "sit down":
            self._rested(p, 0.15)
            return True, {}
        if v == "sleep":
            self._rested(p, 0.6)
            return True, {}
        if v == "tend":                                      # tend: another person's care takes the ailment down
            q.ail = max(0.0, q.ail - TEND_RELIEF * q.recov)
            q.aff[p.name] = min(1, q.aff.get(p.name, 0) + 0.4)
            return True, {"relief": TEND_RELIEF * q.recov}
        if v == "fend off":                                  # allies: one blow at the raid; the raid is gone when it is spent
            th = self.threat
            if th is None or th["room"] != p.room:
                return False, {"why": "it is gone"}
            hit = FEND_DMG * self._rr.uniform(0.6, 1.4)
            th["hp"] -= hit
            for x in self.here(p):
                if x.acts() and not self.asleep(x) and x.doing and x.doing["verb"] == "fend off":
                    x.aff[p.name] = min(1, x.aff.get(p.name, 0) + FEND_WARM)      # those who stand together
                    p.aff[x.name] = min(1, p.aff.get(x.name, 0) + FEND_WARM)
            res = {"result": f"hurt {th['name']}", "damage": round(hit, 1)}
            if th["hp"] <= 0:
                res["result"] = f"drove off {th['name']}"
                self._raid_event("drives off", f"{th['name']} is driven off", th["room"], result=res["result"])
                self.threat = None
            return True, res
        if v == "look around":
            new = [n for n in self.loose(p.room) if n not in p.examined] + \
                  [x.name for x in self.here(p) if x.name not in p.examined]
            p.examined.update(new)
            self._novel(p, 0.02 + 0.04 * len(new))
            self._look(p, "sight")
            return True, {"found": new}
        if v == "gossip":
            return True, {}                                  # the hearer changes on hearing (_learn)
        if v in ("ask", "offer"):
            q.requests[p.name] = {"thing": o, "pay": d["f"].get("pay"), "t": self.t}
            return True, {}
        if v == "accept":
            pay = d["f"]["pay"]
            if W[o]["at"] != p.name or W[pay]["at"] != q.name:
                return False, {"why": "not held"}
            W[o].update(at=q.name, owner=q.name, hidden=False)
            W[pay].update(at=p.name, owner=p.name, hidden=False)
            q.requests.pop(p.name, None)
            p.requests.pop(q.name, None)
            for a, b in ((p, q), (q, p)):
                a.aff[b.name] = min(1, a.aff.get(b.name, 0) + 0.1)
            return True, {"traded": pay}
        if v == "refuse":
            p.requests.pop(q.name, None)
            q.aff[p.name] = max(-1, q.aff.get(p.name, 0) - 0.1)
            return True, {}
        if v == "serve":                                     # meals: a portion from the pot p holds; bring: a drink fetched for q
            if o not in self.portions and "bring" in self.flags and W[o]["props"] & DRINK_PROPS:
                if W[o]["at"] is None:
                    return False, {"why": "none left"}
                self._sip(q, o, bool(W[o]["props"] & {"ale", "ale_source"}))
            elif o not in self.portions or W[o]["at"] != p.name or not self._feed(q, 1, o):
                return False, {"why": "none left"}
            q.aff[p.name] = min(1, q.aff.get(p.name, 0) + 0.3)        # as a gift (give)
            q.anger[p.name] = max(0, q.anger.get(p.name, 0) - 0.3)
            return True, {}
        if v == "buy":                                       # meals: a thing for a meal, at the house's price
            m = d["f"]["pay"]
            if W[m]["at"] != p.name or W[o]["at"] != q.name or not self._feed(p, MEAL_PORTIONS, o):
                return False, {"why": "not held"}
            W[m].update(at=q.name, owner=q.name, hidden=False)
            for a, b in ((p, q), (q, p)):
                a.aff[b.name] = min(1, a.aff.get(b.name, 0) + 0.1)
            return True, {"traded": m}
        if v == "work":                                      # meals: chores for a ration
            if W[o]["at"] != q.name or not self._feed(p, WORK_PORTIONS, o):
                return False, {"why": "none left"}
            if "stock" in self.flags and o in COOK_MAKES:        # the counter is refilled by work
                cap = int(STOCK_CAP * self.stock[o])
                put = WORK_YIELD
                if self.family:                              # family: a child's or an elder's work puts back less (whole rations; the fraction carries)
                    p.work_acc += WORK_YIELD * p.work
                    put = int(p.work_acc + 1e-9)
                    p.work_acc -= put
                self.portions[o] = min(cap, self.portions[o] + put)
                self.things[o]["at"] = q.name
            p.fatigue = min(1.0, p.fatigue + WORK_TIRE)
            self._rested(q, CHORE_RELIEF)
            for a, b in ((p, q), (q, p)):
                a.aff[b.name] = min(1, a.aff.get(b.name, 0) + 0.1)
            return True, {}
        raise ValueError(v)

    def body_loss(self, person):
        """body2: (percent of body mass lost to water, percent of body weight lost to hunger) from the need and the reserve (the ledger above)."""
        dt = 1.0 - person.reserve["thirst"] / LETHAL_RESERVE["thirst"]
        dh = 1.0 - person.reserve["hunger"] / LETHAL_RESERVE["hunger"]
        return water_lost_pct(person.thirst, dt), food_lost_pct(dh)

    def eat_factor(self, person):
        """body2: the share of a ration's relief a person gets: 1 when hydrated, 0.65 at 2.9 percent of body mass lost, never below EAT_FLOOR."""
        m = self.body_loss(person)[0]
        return max(EAT_FLOOR, 1.0 - EAT_ANOREXIA * m / EAT_REF_PCT)

    def dependent(self, q):
        """body2: a person who cannot yet fetch for itself and is brought what it needs (a young child)."""
        return "body2" in self.flags and q.age < DEPENDENT_AGE

    def seen_need_of(self, q, own):
        """The urgency a carer reads off q. A dependent shows only its cry (the cause is not heard); everyone else shows the need itself, above 0.6."""
        if self.dependent(q):
            c = cry_level(q.hunger, q.thirst, max(min(1.0, max(0.0, (10 - q.hp) / 10)), q.ail), q.fear, q.age)
            return c if c > 0.6 else 0.0
        return own if own > 0.6 else 0.0

    def taste(self, item):
        """body2: the signed taste of what is consumed (sweet and savoury good, bitter bad, water neutral)."""
        props = self.things.get(item, {}).get("props", ())
        if "ale" in props or "ale_source" in props:
            return TASTE_ALE
        if "food" in props:
            return TASTE_FOOD
        return TASTE_WATER

    def _sip(self, person, item, ale):
        """One drink (the same effect whether it is drunk or brought): thirst down; ale also makes drunk and a mug is used up."""
        if "body2" in self.flags:
            person.tasted = item
        person.thirst = max(0, person.thirst - 0.5)
        if ale:
            person.drunk = min(1, person.drunk + 0.3)
            if "ale" in self.things.get(item, {}).get("props", ()):
                self.things[item]["at"] = None

    def _eff(self, person, item):
        """labour: the share of a ration's relief that `item` would give `person` now (1 unless it is the kind it just ate)."""
        if "labour" not in self.flags or item not in self.portions or "field" in self.flags:
            return 1.0
        return LABOUR_REPEAT if person.last_kind == self.kind_of.get(item, STAPLE) else 1.0

    def _feed(self, person, n, item):
        """meals: `person` eats up to n rations of `item`. Returns how many it got (0 if the pot is empty)."""
        n = min(n, self.portions.get(item, 0))
        if n <= 0:
            return 0
        if "field" in self.flags and person.hunger < self.full:      # a person whose hunger cannot take a whole ration does not eat: the portion stays in the pot (nothing is wasted on the satiated)
            return 0
        self.portions[item] -= n
        if "body2" in self.flags:        # body2: the dehydrated eat less of a ration, and a meal raises thirst
            f = self.eat_factor(person)
            person.tasted = item
            person.thirst = min(1.0, person.thirst + EAT_THIRST * n)
            if "labour" in self.flags:
                k = self.kind_of.get(item, STAPLE)
                for _ in range(n):
                    person.hunger = max(0.0, person.hunger - PORTION * f * (LABOUR_REPEAT if person.last_kind == k and "field" not in self.flags else 1.0))
                    person.last_kind = k
            else:
                person.hunger = max(0.0, person.hunger - PORTION * n * f)
            if self.portions[item] <= 0:
                self.things[item]["at"] = None
            return n
        if "labour" in self.flags:       # sensory-specific satiety: the same kind again relieves less
            k = self.kind_of.get(item, STAPLE)
            for _ in range(n):
                person.hunger = max(0.0, person.hunger - PORTION * (LABOUR_REPEAT if person.last_kind == k and "field" not in self.flags else 1.0))
                person.last_kind = k
        else:
            person.hunger = max(0.0, person.hunger - PORTION * n)
        if self.portions[item] <= 0:
            self.things[item]["at"] = None
        return n

    def trust(self, p, name):
        """partner: how much p trusts `name`, -1..1, from what p already has: its liking, the deeds it heard of
        (rep), the lies it caught, the robberies and the killing it remembers. One rule for everybody."""
        bad = sum(1 for m in p.memory if m.get("who") == name and m.get("what") == "robbed")
        bad += 2 * sum(1 for m in p.memory if m.get("by") == name and m.get("what") == "died")
        t = p.aff.get(name, 0.0) + 0.1 * p.rep.get(name, 0) - 0.5 * p.liars.get(name, 0) - 0.2 * min(3, bad)
        return max(-1.0, min(1.0, t))

    def _salience(self, p, d):
        """news: how worth telling a deed is. Unusual among the deeds p has seen (its surprisal, in bits), against
        p's own opinion of the doer, and big. The same for a good deed and a bad one."""
        sign = DEED[d["action"]]
        same = sum(1 for x in p.deeds if (DEED[x["action"]] > 0) == (sign > 0))
        surprise = min(1.0, -math.log2((same + 1) / (len(p.deeds) + 2)) / SURPRISE_BITS)
        against = min(1.0, 2 * max(0.0, -math.copysign(1, sign) * p.aff.get(d["actor"], 0)))
        return abs(sign) / 1.5 * (0.6 * surprise + 0.4 * against)

    def _echo(self, turns):
        """calm: how much of what p said before still weighs on it, from the turns it said it."""
        return sum(0.5 ** ((self.t - t0) / ECHO_HALF) for t0 in turns) if turns else 0.0

    def _after(self, p, d, ev):
        """What a done act does to the actor itself, under the flags."""
        v = d["verb"]
        if "wear" in self.flags and v in WEAR_VERBS and d["target"] in self.people:
            self.words.append((self.t, p.name, d["target"], v))
        if "drives" in self.flags:
            p.fatigue = min(1.0, p.fatigue + EXERT.get(v, 0))
            if v == "drink":
                self._rested(p, 0.03)                         # drinking slowly rests
            if v in ("walk to", "flee to") and p.room not in p.visited:
                p.visited.add(p.room)
                self._novel(p, 0.15)
            if v == "open":
                self._novel(p, 0.4 if ev.get("found") and self.chest and p.name == self._first_opener() else 0.15)
        if "calm" in self.flags and v in HARMFUL and d["target"] in self.people and d["target"] in p.anger:
            p.anger[d["target"]] *= 1.0 - DISCHARGE           # acting on a grievance spends part of it
        if v in ("give", "serve") and "trade" in self.flags:
            q = self.people[d["target"]]
            if p.requests.get(q.name, {}).get("thing") == d["obj"]:
                del p.requests[q.name]                         # served
        if v == "search" and "check" in self.flags:
            self._checked(p, self.people[d["target"]], ev)
        if v == "gossip":
            p.told.add((d["target"], d["f"]["deed"]["key"]))
        if "known" in self.flags:            # p knows where the things it just handled went
            for n in (d.get("obj"), d["f"].get("pay")):
                if n in self.things and v in ("take", "steal", "drop", "give", "eat", "drink", "hide", "accept", "buy"):
                    at = self.things[n]["at"]
                    if at is None:
                        p.bel.pop(("where", n), None)
                    else:
                        p.bel[("where", n)] = (at, ev["id"])
            if v == "open":
                p.seen_open = True
            for n in self.held(p):           # and what it holds after it (the chest's coins)
                p.bel[("where", n)] = (p.name, ev["id"])

    def _wear(self, actor, target, verb):
        """The effect of these words now: WEAR to the power of how many of the
        same words the same actor said to the same target within WEAR_SPAN."""
        if "wear" not in self.flags or verb not in WEAR_VERBS:
            return 1.0
        n = sum(1 for (t, a, b, v) in self.words if a == actor and b == target and v == verb
                and self.t - t <= WEAR_SPAN)
        return WEAR ** n

    def _first_opener(self):
        return next((e["actor"] for e in self.log if e["action"] == "open" and e["done"]), None)

    # ------------------------------------------------------------ log, perception, beliefs
    def _emit(self, p, d, done=True, **info):
        c = d.get("choice", {})
        target = d.get("target")
        tk = self.people[target].kind if target in self.people else ("room" if target in ROOMS else None)
        ev = {"id": f"e{self.n}", "t": self.t, "actor": p.name, "kind": p.kind,
              "traits": c.get("traits", {k: round(v, 3) for k, v in p.traits.items()}),
              "action": d["verb"], "target": target, "target_kind": tk, "text": d["text"],
              "parts": {"with": d["obj"]} if d.get("obj") and d["verb"] in ("strike",) else
                       ({"thing": d["obj"]} if d.get("obj") else {}),
              "room": p.room, "done": done, "valid": True}
        for k in ("options", "chosen", "p_char", "p_avg", "seen"):
            if k in c:
                ev[k] = c[k]
        said = d["f"].get("said") if d["verb"] == "tell" else None
        if said:
            thing = d["obj"]
            real = self._real_taker(thing)
            # "true": what the world says. "lie": the speaker KNOWS it is false
            # (passing on a false rumour it believes is not a lie).
            ev["facts"] = {"said": f"{said} took {thing}", "true": real == said, "who": said, "thing": thing,
                           "lie": bool(d["f"].get("lie"))}
        if d["verb"] == "gossip":
            dd = d["f"]["deed"]
            ev["facts"] = {"said": f"{dd['actor']} {PAST[dd['action']]} {dd['target']}", "true": True,
                           "who": dd["actor"], "deed": dd["action"], "victim": dd["target"],
                           "source": f"n{dd['key'][0]}:{dd['key'][1]}"}
        ev.update({k: v for k, v in info.items() if k != "sneaky"})
        self.n += 1
        self.log.append(ev)
        if done:
            self._perceive(ev, sneaky=info.get("sneaky", False))
        return ev

    def _real_taker(self, thing):
        for e in self.log:
            if e["action"] in ("steal", "take") and e.get("done") and e["parts"].get("thing") == thing \
                    and self.things[thing]["owner"] not in (None, e["actor"]):
                return e["actor"]
        return None

    def note(self, p, action, **info):
        """An event that is not a choice: finding something out, arriving."""
        d = {"verb": action, "text": info.pop("text", action), "target": info.pop("target", None),
             "obj": info.pop("obj", None), "f": {}}
        return self._emit(p, d, **info)

    def _perceive(self, ev, sneaky=False):
        actor = self.people[ev["actor"]]
        room = ev["room"]
        loud = ev["action"] in LOUD
        drives = "drives" in self.flags
        for q in self.people.values():
            if q is actor or q.dead:
                continue
            sees = q.room == room and not q.down
            asleep = drives and self.asleep(q)
            if asleep:
                sees = False                     # a sleeper sees nothing
            if sees and sneaky:                  # a theft is seen only sometimes
                chance = 0.3 if q.name == ev["target"] else 0.35
                if "watch" in self.flags and self.distrust(q, actor.name) >= 0.3:
                    chance = 0.6                 # a distrusted one is watched
                sees = self.rng.random() < chance * (1 - 0.5 * q.drunk)
            hears = (not sees) and loud and (room in ROOMS[q.room] or (asleep and q.room == room))
            if not (sees or hears):
                continue
            q.perceived.add(ev["id"])
            q.pending.append(ev["id"])
            if "known" in self.flags:
                self._perceive_known(q, ev)
            if drives and (ev["actor"], ev["action"]) not in q.kinds_seen:
                q.kinds_seen.add((ev["actor"], ev["action"]))
                self._novel(q, 0.03)
            if asleep and hears:                 # a loud noise wakes a sleeper
                self.interrupt(q, ev["id"])
            if sees:
                self._learn(q, ev, actor)
            else:
                q.fear = min(1, q.fear + 0.05 * (1 - q.traits["bravery"]))
            # something important stops what q is doing: harm to q, a blow in
            # sight or earshot, a shout, a theft it saw
            aimed = ev["target"] == q.name and ev["action"] in HARMFUL
            if aimed or ev["action"] in VIOLENT or ev["action"] == "shout for help" or (sneaky and sees):
                self.interrupt(q, ev["id"])

    def _perceive_known(self, q, ev):
        """What q takes from any event it saw or heard (known flag): where the
        actor is, who died, the chest opened, hostility aimed at q."""
        a = ev["action"]
        q.seen_at[ev["actor"]] = ev["room"]
        res = ev.get("result") or ""
        if res.startswith("killed "):
            q.seen_at[res[len("killed "):]] = None
        if a == "open":
            q.seen_open = True
        if ev["target"] == q.name and a in HOSTILE:
            h = max(0.0, min(1.0, self.hostility(q, ev["actor"]) + HOSTILE[a] * self._wear(ev["actor"], q.name, a)))
            q.hostile[ev["actor"]] = (h, self.t)
        if a == "accept" and ev["target"] == q.name:       # q's pay went, the thing came
            for n, holder in ((ev.get("traded"), ev["actor"]), (ev["parts"].get("thing"), q.name)):
                if n in self.things:
                    q.bel[("where", n)] = (holder, ev["id"])

    def _learn(self, q, ev, actor):
        a, v, thing = ev["action"], ev["target"], ev["parts"].get("thing") or ev["parts"].get("with")
        tr = q.traits
        if a in ("take", "steal", "give", "drop") and thing in self.things:
            holder = {"take": actor.name, "steal": actor.name, "give": v, "drop": ev["room"]}[a]
            self._contradict(q, thing, holder, ev["id"])      # before the new sight replaces the old belief
            q.bel[("where", thing)] = (holder, ev["id"])
            if a in ("steal", "take") and self.things[thing]["owner"] not in (None, actor.name):
                q.bel[("took", thing)] = (actor.name, ev["id"])
                if self.things[thing]["owner"] == q.name:
                    q.anger_up(actor.name, 0.5 * (1 + tr["temper"]))
                    q.aff[actor.name] = q.aff.get(actor.name, 0) - 0.4
                    q.memory.append({"night": self.night, "t": self.t, "what": "robbed", "who": actor.name,
                                     "thing": thing})
        if a == "search" and ev.get("found"):
            for n in ev["found"]:
                self._contradict(q, n, v, ev["id"])
                q.bel[("where", n)] = (v, ev["id"])
                if self.things[n]["owner"] not in (None, v):
                    q.bel[("took", n)] = (v, ev["id"])
        if a == "tell" and v == q.name and "facts" in ev:
            # believed if q does not distrust the teller
            if q.aff.get(actor.name, 0) > -0.3:
                old = q.bel.get(("took", ev["facts"]["thing"]))
                if not (old and old[1].startswith("e") and self._saw(q, old[1])):
                    q.bel[("took", ev["facts"]["thing"])] = (ev["facts"]["who"], ev["id"])
        if a in HARMFUL and v == q.name:
            wear = self._wear(actor.name, q.name, a)
            hurt = (0.45 if a in VIOLENT else (0.25 if a in PHYSICAL else 0.15)) * wear
            q.anger_up(actor.name, hurt * (1 + tr["temper"] + q.drunk))
            q.aff[actor.name] = max(-1, q.aff.get(actor.name, 0) - hurt)
            if a in VIOLENT:
                q.fear = min(1, q.fear + 0.3 * (1 - tr["bravery"]))
            if a == "insult":
                q.anger_up(actor.name, 0.2 * tr["pride"] * wear)
        elif a in HARMFUL and v in q.aff and q.aff[v] > 0.2:
            q.anger_up(actor.name, 0.25 * (tr["kindness"] + tr["temper"]))
            q.aff[actor.name] = max(-1, q.aff.get(actor.name, 0) - 0.15)
        if a in VIOLENT:
            q.fear = min(1, q.fear + 0.1 * (1 - tr["bravery"]))
        if a in ("help up", "bandage", "defend") and v in q.aff and q.aff[v] > 0:
            q.aff[actor.name] = min(1, q.aff.get(actor.name, 0) + 0.1)
        if self.flags:
            self._learn_flag(q, ev, actor)

    def _learn_flag(self, q, ev, actor):
        """What q takes from an event it SAW, under the flags."""
        a, v, thing = ev["action"], ev["target"], ev["parts"].get("thing")
        if "drives" in self.flags and v == q.name:
            wear = self._wear(actor.name, q.name, a)
            q.slight = min(1.0, max(0.0, q.slight + SLIGHT.get(a, 0) * wear))
            if a in KIND_ACTS:
                q.lonely = max(0.0, q.lonely - 0.15 * wear)
            elif a in HARSH_ACTS:
                q.lonely = min(1.0, q.lonely + 0.05 * wear)
        if "reprisal" in self.flags and a in HARMFUL and v in self.people and v != q.name and ev.get("done"):
            # a witness thinks the less of one who does harm; the doer pays for being seen
            share = PROVOKED_SHARE if "calm" in self.flags and self._provoked(q, actor, v, ev) else 1.0
            q.aff[actor.name] = max(-1, q.aff.get(actor.name, 0) - WITNESS_COST * share)
            if "drives" in self.flags:
                actor.slight = min(1.0, actor.slight + WITNESS_SLIGHT * share)
                actor.lonely = min(1.0, actor.lonely + WITNESS_SLIGHT * share)
        if "news" in self.flags and a in HELPED and v in self.people and v != q.name and ev.get("done"):
            # the mirror of reprisal: a witness warms to one who helps in view, and the helper gains standing
            q.aff[actor.name] = min(1, q.aff.get(actor.name, 0) + WITNESS_COST)
            if "drives" in self.flags:
                actor.slight = max(0.0, actor.slight - WITNESS_SLIGHT)
                actor.lonely = max(0.0, actor.lonely - WITNESS_SLIGHT)
        if "gossip" in self.flags and a in DEED and v in self.people and v != q.name and ev.get("done"):
            self._witness(q, actor.name, a, v, ev["id"])
        if a == "gossip" and v == q.name and "facts" in ev:
            f = ev["facts"]
            if self.distrust(q, actor.name) < 0.3 and f["who"] != q.name:
                # reputation: a bad deed told lowers q's liking of the doer, a
                # good one raises it; more when the victim is someone q likes
                sign = DEED[f["deed"]]
                w = 0.12 * (1 + (q.aff.get(f["victim"], 0) > 0.2))
                q.aff[f["who"]] = max(-1, min(1, q.aff.get(f["who"], 0) + w * sign))
                q.rep[f["who"]] = q.rep.get(f["who"], 0) + sign
                q.aff[actor.name] = min(1, q.aff.get(actor.name, 0) + 0.03)     # confided in
            elif f["who"] == q.name and DEED[f["deed"]] < 0:
                q.aff[actor.name] = max(-1, q.aff.get(actor.name, 0) - 0.1)
        if a == "gossip" and v != q.name and ev["facts"]["who"] == q.name and DEED[ev["facts"]["deed"]] < 0:
            q.aff[actor.name] = max(-1, q.aff.get(actor.name, 0) - 0.1)       # overheard being run down
        if "check" in self.flags and a == "hide" and thing in self.things:
            self._contradict(q, thing, actor.name, ev["id"])
            q.bel[("where", thing)] = (actor.name, ev["id"])
        if "check" in self.flags and a == "tell" and v == q.name and "facts" in ev:
            old = q.bel.get(("took", ev["facts"]["thing"]))
            if old and old[0] != ev["facts"]["who"] and old[1].startswith("e") and self._saw(q, old[1]):
                self._caught(q, actor.name, ev["facts"]["thing"], ev["id"], "it contradicts what I saw")

    def _provoked(self, q, actor, target, ev):
        """calm: witness q saw `actor` answer a wrong: the target had harmed the actor within PROVOKED_SPAN turns
        (q saw it), or, for a charge or a search, q itself believes the target took the thing. A witness
        condemns the one who began a quarrel, not the one who answers it."""
        if ev["action"] == "accuse":
            return q.bel.get(("took", ev["parts"].get("thing")), (None,))[0] == target
        if ev["action"] == "search" and any(k == "took" and v == target for (k, _), (v, _s) in q.bel.items()):
            return True
        return any(e["actor"] == target and e.get("target") == actor.name and e["action"] in HARMFUL and e.get("done")
                   and e["id"] in q.perceived for e in self.log[-80:] if e["t"] >= self.t - PROVOKED_SPAN)

    def _witness(self, q, actor, action, target, eid):
        q.deeds.append({"key": (self.night, eid), "actor": actor, "action": action, "target": target})
        if len(q.deeds) > 12:
            if "news" in self.flags:     # forget the oldest of the commoner kind, so a rare good deed is kept
                good = [x for x in q.deeds if DEED[x["action"]] > 0]
                bad = [x for x in q.deeds if DEED[x["action"]] <= 0]
                q.deeds.remove((good if len(good) > len(bad) else bad)[0])
            else:
                del q.deeds[0]

    def _caught(self, q, liar, thing, eid, why):
        """q finds that `liar` told it something false."""
        q.liars[liar] = q.liars.get(liar, 0) + 1
        q.memory.append({"night": self.night, "t": self.t, "what": "caught lying", "who": liar, "thing": thing,
                         "why": why})
        if "gossip" in self.flags:
            self._witness(q, liar, "lie", q.name, eid)

    def _checked(self, p, q, ev):
        """The searcher's own lesson (check flag): what it found, or that a
        tale it was told about q does not hold."""
        for n in ev.get("found", []):
            self._contradict(p, n, q.name, ev["id"])
            p.bel[("where", n)] = (q.name, ev["id"])
            if self.things[n]["owner"] not in (None, q.name):
                p.bel[("took", n)] = (q.name, ev["id"])
        for (k, n), (who, src) in list(p.bel.items()):
            if k != "took" or who != q.name or n in self.held(q) or (n, src) in p.doubted:
                continue
            src_ev = next((x for x in self.log if x["id"] == src), None)
            if src_ev and src_ev["action"] == "tell":
                p.doubted.add((n, src))
                del p.bel[("took", n)]
                p.aff[src_ev["actor"]] = max(-1, p.aff.get(src_ev["actor"], 0) - 0.2)
                self.note(p, "doubt", text=f"doubt that {q.name} took {n}", obj=n,
                          facts={"learns": f"{q.name} does not have {n}", "contradicts": src})

    def _saw(self, q, eid):
        e = next((x for x in self.log if x["id"] == eid), None)
        return e is not None and e["action"] != "tell"

    def _contradict(self, q, thing, holder, eid):
        """q believed, because it was TOLD, that someone else took the thing,
        and now sees who has it: the lie is found out."""
        b = q.bel.get(("took", thing))
        if not b:
            return
        who, src = b
        src_ev = next((x for x in self.log if x["id"] == src), None)
        if src_ev and src_ev["action"] == "tell" and holder in self.people and holder != who \
                and self.things[thing]["owner"] != holder:
            liar = src_ev["actor"]
            q.bel[("took", thing)] = (holder, eid)
            q.aff[liar] = max(-1, q.aff.get(liar, 0) - 0.5)
            q.anger_up(liar, 0.3)
            self._caught(q, liar, thing, src, f"saw that {holder} has it")   # memory only; no draw
            self.note(q, "notice", text=f"notice that {holder} has {thing}",
                      facts={"learns": f"{holder} has {thing}", "contradicts": src})

    def _look(self, p, src):
        """p sees what lies in the room and what the others here hold openly."""
        for n, th in self.things.items():
            if th["at"] == p.room:
                p.bel[("where", n)] = (p.room, src)
        check = "check" in self.flags and src == "sight"
        for q in self.here(p):
            for n in self.held(q, visible_only=True):
                if check:                     # a tale is checked against who holds it openly
                    self._contradict(p, n, q.name, "sight")
                p.bel[("where", n)] = (q.name, src)
        if "drives" in self.flags:
            new = [n for n, th in self.things.items() if th["at"] == p.room and n not in p.glimpsed] + \
                  [q.name for q in self.here(p) if q.name not in p.glimpsed]
            if new:
                p.glimpsed.update(new)
                self._novel(p, 0.02 * len(new))
        for n in self.held(p):
            p.bel[("where", n)] = (p.name, src)
        if "known" in self.flags:
            here = {q.name for q in self.here(p)}
            for name in here:
                p.seen_at[name] = p.room
            for name in list(p.seen_at):
                if p.seen_at[name] == p.room and name not in here:
                    p.seen_at[name] = None           # gone, p does not know where
            for name, room in self.start_room.items():    # not here, where the night started
                if room == p.room and name not in here and name != p.name and name not in p.seen_at:
                    p.seen_at[name] = None

    def _check_own(self, p):
        """p notices that something of its own is gone (under `known`:
        anything it believed it held)."""
        known = "known" in self.flags
        for n, th in self.things.items():
            if (known or th["owner"] == p.name) and th["at"] != p.name \
                    and p.bel.get(("where", n), (None,))[0] == p.name:
                p.bel[("where", n)] = ("gone", "own")
                p.fear = min(1, p.fear + 0.1)
                for q in self.here(p):
                    p.anger_up(q.name, 0.1)
                self.note(p, "notice", text=f"notice that {n} is gone", obj=n,
                          facts={"learns": f"{n} is gone"})

    def _raid_event(self, action, text, room, victim=None, loud=True, **info):
        """allies: an event of the raid (not a person's choice). Those in the room see it and are stopped by it; those next
        door hear it. It is logged like any event, with the raid as the actor."""
        th_name = (self.threat or {}).get("name", info.pop("name", "a raider"))
        ev = {"id": f"e{self.n}", "t": self.t, "actor": th_name, "kind": "raid", "traits": {}, "action": action,
              "target": victim, "target_kind": "person" if victim else None, "text": text,
              "parts": {}, "room": room, "done": True, "valid": True}
        ev.update(info)
        self.n += 1
        self.log.append(ev)
        for q in self.people.values():
            if q.dead:
                continue
            asleep = self.asleep(q)
            sees = q.room == room and not q.down and not asleep
            hears = not sees and loud and (room in ROOMS[q.room] or (q.room == room and asleep))
            if not (sees or hears):
                continue
            q.perceived.add(ev["id"])
            q.pending.append(ev["id"])
            if "known" in self.flags:
                q.seen_at[th_name] = room
            if sees:
                q.fear = min(1, q.fear + 0.04 * (1.2 - q.traits["bravery"]))
            else:
                q.fear = min(1, q.fear + 0.05 * (1 - q.traits["bravery"]))
            if q.room == room or asleep:
                self.interrupt(q, ev["id"])
        return ev

    def _raid_arrives(self):
        self.threat = {"name": self._raid_kind, "room": "the yard", "hp": RAID_HP, "t0": self.t,
                       "mortal": "raidlaw" in self.flags and self._rr.random() < RAID_KILL}
        self._raid_event("raid arrives", f"{self._raid_kind} comes in from the road", "the yard")

    def _raid_round(self):
        """allies: the raid hits one person in its room, or goes for the people. It is beaten when its hp is spent
        (see fend off)."""
        th = self.threat
        law = "raidlaw" in self.flags          # the blows go to those who can still resist; the fallen are out of the fight
        live = [q for q in self.people.values() if not q.dead and (q.acts() or not law)]
        here = [q for q in live if q.room == th["room"]]
        if not here:
            rooms = ROOMS[th["room"]]
            best = max(rooms, key=lambda r: sum(1 for q in live if q.room == r))
            if any(q.room == best for q in live):
                th["room"] = best
                self._raid_event("raid prowls", f"{th['name']} goes into {best}", best)
            elif law and not live:
                self._raid_event("drives off", f"{th['name']} goes, nobody is left to fight", th["room"], result="left")
                self.threat = None
            return
        v = self._rr.choice(here)
        dmg = RAID_DMG * self._rr.uniform(0.5, 1.5)
        v.hp -= dmg
        v.fear = min(1, v.fear + 0.15 * (1.2 - v.traits["bravery"]))
        res = f"hit {v.name}"
        if v.hp <= 4 and not v.down:
            v.down = True
            res = f"knocked {v.name} down"
        if v.hp <= -2 or th.get("mortal"):
            th["mortal"] = False
            v.dead = v.down = True
            res = f"killed {v.name}"
        self._raid_event("raid hits", f"{th['name']} hits {v.name}", th["room"], victim=v.name, result=res,
                         damage=round(dmg, 1))

    def _bandit_arrives(self):
        b = Person("the bandit", "bandit", "the yard", self.rng)
        for q in self.people.values():
            b.aff[q.name] = -0.3
            q.aff["the bandit"] = -0.3
        self.people[b.name] = b
        self.things["the club"].update(at=b.name, owner=b.name)
        self.note(b, "arrive", text="come in from the road")

    # ------------------------------------------------------------ for the models
    def describe(self, p, persona=None):
        """What p knows, in words, from p's side. No hidden facts."""
        s = [f"It is night at the village inn. {p.name} is {p.public}, in {p.room}."]
        if persona:
            s.append(persona)
        here = self.here(p)
        if here:
            s.append("Here: " + "; ".join(
                f"{q.name} ({q.public}{', lying on the floor' if q.down else ''}"
                f"{', holding ' + ', '.join(self.held(q, visible_only=True)) if self.held(q, visible_only=True) else ''})"
                for q in here) + ".")
        things = [n for n in self.loose(p.room)]
        if things:
            s.append("Things here: " + ", ".join(things) + ".")
        mine = self.held(p)
        s.append(f"{p.name} holds " + (", ".join(mine) if mine else "nothing") + ".")
        feel = []
        if p.hunger > 0.6:
            feel.append("hungry")
        if p.thirst > 0.6:
            feel.append("thirsty")
        if p.fear > 0.4:
            feel.append("afraid")
        if p.drunk > 0.4:
            feel.append("drunk")
        if p.hp < 7:
            feel.append("hurt")
        mad = [n for n, a in p.anger.items() if a > 0.4]
        if mad:
            feel.append("angry at " + " and ".join(mad))
        if feel:
            s.append(f"{p.name} feels " + ", ".join(feel) + ".")
        recent = [e for e in self.log if e["id"] in p.perceived][-5:]
        if recent:
            s.append("Recently: " + " ".join(
                f"{e['actor']} {'did ' if not e['done'] else ''}{e['text']}"
                f"{' (' + e['result'] + ')' if e.get('result') else ''}." for e in recent))
        for (k, thing), (who, _src) in p.bel.items():
            if k == "took" and who != p.name:
                s.append(f"{p.name} believes {who} took {thing}.")
        return " ".join(s)
