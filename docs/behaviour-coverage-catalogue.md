# Behaviour coverage catalogue: draft for review

Draft, 2026-11-20 (from the lab and docs run, checked by a critic). Every real behaviour must be POSSIBLE and TUNABLE in the sim. Not yet run: the reachable counts come from reading the code. Open questions are in section 9.

```text
BEHAVIOUR COVERAGE CATALOGUE: DRAFT 1 (design, for your review)
Written 2026-11-20, lane mindL3. Nothing here has been run. No file in src/, tests/ or tools/ was touched. Not hashed.
Sources: (u) = from memory, not looked up. Many are old classics (author and year I am fairly sure of; page, title and effect are not). Before anything is hashed, a lookup pass is needed, as with the registry.

PLAIN WORDS FIRST
- You said: every behaviour people show in real life must be POSSIBLE in the sim in similar circumstances, and we must be able to TWEAK it. So the old pattern list (28 "more X goes with more Y" rules) stops being a target to match. It becomes a CATALOGUE of behaviours. Each entry gets three checks:
  1. POSSIBLE: somewhere in the space of dials, history, state and world settings, the behaviour shows up. If nowhere, we do not blame the mind. We write down which law, act or sense is missing. That list is the real product: it tells the engine waves what to build.
  2. TUNABLE: a named dial moves how often it shows up, in one direction.
  3. TYPICAL BY DEFAULT: how often it shows up in the default cast and default settings, set beside a rough real-world "common / occasional / rare / very rare". Report only. Never used to select or train the mind.
- The evidence bar is low on purpose: show that the behaviour EXISTS and name the circumstances it happens in. No effect sizes.
- The catalogue is open-ended. It grows from seed lists (section 3), from the books (section 7), and from what playtests show.
- This draft seeds it with 131 entries: the 28 old patterns, Brown's human universals, animal ethogram categories, rare and extreme behaviours, and group behaviours.

THE HEADLINE (from reading the inn's code, not from running it)
- 131 entries. Reachable by reading: 33 Y (every act, sense and law is there; the search has not run, so "Y" means "should be found", not "was found"); 54 P (one named piece is missing, mostly a log field); 44 N (a missing act, law, sense or world setting).
- Missing pieces across the 98 non-Y entries (an entry can miss two): 29 need a new ACT, 30 a new LAW, 11 a new SENSE or marker, 22 a new WORLD SETTING (bigger cast, death in the night, children, groups, weather), 12 only a LOG field, 3 a designed probe, 2 a mind-side change.
- So the biggest gaps are not about the mind. They are: no groups or coalitions, no promises or commitments, no greeting, touch or apology acts, no death or grief law, no addiction, fear-freeze or pain law, and a cast of five. Section 8 turns these into a missing-law log.

----------------------------------------------------------------------
1. THE TABLE
----------------------------------------------------------------------
Columns: ID | behaviour | circumstances it happens in | what the inn has or lacks | NOW | seed source.
NOW codes. Y = every act, sense and law exists, so a search should find it. P:x = one thing missing; x is log (the recorder of section 5 fixes it), probe (needs a designed two-clone probe), sense (a marker or sense the menu lacks), law (a world law), act (a menu act), world (a different cast or setting), mind (the mind cannot yet represent it). N:x = more than a small addition, x as above.
[registry n] marks old pattern n (its X, Y and source stay in docs/learned-mind-registry-draft.md).


A. RECIPROCITY AND COOPERATION
A1 | Direct reciprocity (kind for kind) | repeated meetings, same two people | all exist (kind acts, memory of B) | Y | Axelrod+Hamilton 1981 (u); Fehr+Gaechter 2000 (u) [registry 1]
A2 | Indirect reciprocity (kind to those seen being kind) | a witness sees B help a third person | gossip flag, perceived set | P:log | Wedekind+Milinski 2000 (u) [registry 2]
A3 | Generosity after one mistake | long kind history then one harsh act | no "mistake" marker (proxy: harm while drunk or interrupted) | P:sense | Wu+Axelrod 1995 [registry 3]
A4 | Forgiveness over time | time passes since a harm, no further harm | anger fade exists | Y | McCullough+others 2010 [registry 4]
A5 | Food sharing with the hungry | one has food, a neighbour is hungry, asked or seen | give, ask, serve, hunger drive | Y | Kaplan+Hill 1985 (u); Gurven 2004 (u)
A6 | Tolerated theft / begging accepted | a holder has more than it needs, a needy one takes or asks | steal, ask, refuse | Y | Blurton Jones 1984 (u)
A7 | Gift giving to build a bond | before asking a favour, or after a quarrel | give, offer; no "gift raises liking" check on the giver side | P:law | Mauss 1925 (u)
A8 | Hospitality to a stranger | a newcomer arrives tired and hungry | serve, give; no newcomer arrival in a night | P:world | Brown 1991 (u)
A9 | Mutual aid in injury | one is down or bleeding, others near | help up, bandage, defend | Y | Hamilton 1964 (u) for kin; Trivers 1971 (u)
A10 | Division of labour (a trade-off of tasks) | work, chores, cook and serve exist; no choice of trade by skill | work, serve | P:law | Brown 1991 (u)
A11 | Promise made and kept | a person says it will do X later, and does | no promise act, no commitment record | N:act+law | Brown 1991 (u) [promise]
A12 | Promise broken, then reputation loss | a person said it would, did not; others hear | promise record, gossip | N:act+law | Brown 1991 (u)
A13 | Fair-share rules (equal split, protest at unfairness) | a split of food between people with equal claim | ask, give; no split act | N:act | Henrich and others 2001 (u)
A14 | Free riding and its punishment | one takes the shared food and does no work | work, steal, accuse, reprisal flag | Y | Fehr+Gaechter 2002 (u)
A15 | Trade at a price (buy, offer for a thing) | two people with unequal holdings | buy, offer, accept, refuse | Y | Brown 1991 (u)

B. PARTNERS AND BONDS
B1 | Approach follows liking | a choice among people in reach | aff exists | Y | Barclay+Willer 2007 [registry 5]
B2 | Exit after harm (leave the harmer's room) | harmed by B, B in the same room | walk to, flee to | Y | Barclay 2013 (u) [registry 6]
B3 | Attachment and distress at separation | a close partner leaves or is silent | company drive is generic, not per person | P:sense | Bowlby 1969 [registry 16]
B4 | Grief after a permanent loss | a close partner dies or leaves for good | a dead flag exists; no death act in the menu, no per-person weight | N:law+world | Parkes+Weiss 1983 (u) [registry 17]
B5 | Mourning behaviour (a funeral, a shared silence) | a death among the group | no ritual act, no death in a night | N:act+world | Brown 1991 (u)
B6 | Friendship (stable pairs seeking each other) | many nights, mutual kind acts | aff, company, nights | P:world | Brown 1991 (u)
B7 | Pair bond / courtship | two people with a sexual or romantic drive | no such drive, no act | N:law+act | Brown 1991 (u)
B8 | Kin favouritism (nepotism) | a kin tie is known | no kin relation in the cast | N:world | Hamilton 1964 (u)
B9 | Greeting and leave-taking | meeting or parting | no greeting act ("praise" is not it) | N:act | Brown 1991 (u)
B10 | Allogrooming / comforting touch | a partner is tense or hurt | no touch act (bandage only) | N:act | Dunbar 1991 (u)
B11 | Consolation after a fight (third-party) | a bystander near a loser | help up, praise | P:sense | de Waal+Aureli 1996 (u)
B12 | Reconciliation after a fight (the two make up) | two who fought, afterwards | give, praise, no apology act | P:act | de Waal+van Roosmalen 1979 (u)
B13 | Reputation as a partner (chosen for what others say) | a choice among partners with gossip about them | gossip, trade | Y | Sylwester+Roberts 2010 [registry 5]

C. RISK AND DECISION BIAS
C1 | Energy-budget risk sensitivity | short of need: take the gamble; above it: the sure thing | no outcome-variance on the menu | P:sense | Caraco and others 1980 [registry 7]
C2 | Switch at the mind's own set point | same, per type | same | P:sense | Kahneman+Tversky 1979 [registry 8]
C3 | Taking under need (steal, threaten) | deficit high, few witnesses | steal, threaten, witness set | Y | Wilson+Daly 1997 (u) [registry 9]
C4 | Loss aversion in choices | equal gain and loss on offer | no priced gamble on the menu | N:act | Brown+others 2024
C5 | Sunk-cost persistence | effort already spent on a task that is failing | work, search, wait; no "task" record | P:sense | Arkes+Blumer 1985 (u)
C6 | Thrill and novelty seeking | curiosity drive high, a risky unknown option | look around, open (chest) | Y | Zuckerman 1979 (u)
C7 | Gambling with dice | the dice exist, a stake | dice are an object with no act | N:act | Brown 1991 (u)
C8 | Overconfidence (picking a fight beyond one's strength) | a weak party attacks a strong one | strike, threaten, strength estimate | P:sense | Johnson+Fowler 2011 (u)

D. DOMINANCE AND STATUS
D1 | Winner effect | just won a contest | no "contest outcome" marker | P:log | Rutte and others 2006 [registry 10]
D2 | Loser effect | just lost a contest | same | P:log | Hsu and others 2006 [registry 11]
D3 | Linear order forms | many contests among 5 or more | yields not logged as such | P:log | de Vries 1995 (u) [registry 12]
D4 | Prestige vs dominance routes | influence gained by being sought vs by fear | ask, offer, threaten | P:log | Cheng and others 2013 [registry 13]
D5 | Submission and appeasement display | facing a stronger threatening party | no submit act (give, refuse only) | P:act | Tinbergen 1963 (ethogram)
D6 | Bullying of the weak | a weak target, no helpers present | insult, shove, strike, witnesses | Y | Olweus 1993 (u)
D7 | Boasting / display of strength | an audience, a high standing drive | no display act (play, praise) | N:act | Zahavi 1975 (u)
D8 | Humility and self-effacement | a high-status person among the low | no act of self-lowering | N:act | Brown 1991 (u)
D9 | Leadership (others follow and defer) | a shared danger, one person acts first | no follow or defer act; walk to the same room only | N:act+law | Brown 1991 (u)
D10 | Deference to elders / rank | a rank known, a clash of wants | no rank, no age | N:world | Brown 1991 (u)
D11 | Tyranny (one holds others by threat, others cannot resist) | one strong party, others scattered | strike, threaten, steal; no coalition among the weak | P:world | Boehm 1999 (u)
D12 | Reverse dominance (the group levels a bully) | many weak, one strong bully | no coalition act | N:act | Boehm 1999 (u)

E. LEARNING, HABIT, ADDICTION, RITUAL
E1 | Secondary reinforcement | a person paired with relief | pairings in log | P:log | Kelleher+Gollub 1962 (u) [registry 14]
E2 | Intermittent reward resists extinction | gifts stop after two schedules | designed probe | P:probe | Humphreys 1939 (u) [registry 15]
E3 | Primacy | same acts, order flipped | designed probe | P:probe | Asch 1946 (u) [registry 27]
E4 | Recency and decay | older acts weigh less | designed probe | P:probe | Anderson+Schooler 1991 (u) [registry 28]
E5 | Habit (the same act repeated in the same cue) | same room, same time, same need | no habit trace; mind memory only | P:mind | Wood+Neal 2007 (u)
E6 | Addiction (craving, tolerance, withdrawal) | a drug (ale) repeatedly drunk | drink exists; no tolerance, no craving, no withdrawal law | N:law | Robinson+Berridge 1993 [wanting vs liking]
E7 | Superstition (an act kept after a chance pairing) | chance reward after an act | no random reward channel | N:world | Skinner 1948 (u)
E8 | Ritual (a fixed sequence repeated, no direct effect) | stress, a group, a recurring cue | no sequence act; lute "play" is the nearest | N:act+law | Boyer+Lienard 2006 (u)
E9 | Hoarding (taking more than needed) | scarcity, then plenty | take, hold; value of held things; no storage limit | P:law | Vander Wall 1990 (u); Frost+Gross 1993 (u)
E10 | Imitation of the successful | watch another get relief by an act | perceived set; mind learns from own acts only | P:mind | Chartrand+Bargh 1999 (u)
E11 | Teaching (showing another how) | a skilled and an unskilled person | no skill record | N:law+act | Brown 1991 (u)

F. LOSS, ANGER, PUNISHMENT, REVENGE
F1 | Betrayal costs more from the trusted | harm from a liked partner | aff | P:log | Bohnet+Zeckhauser 2004 (indirect) [registry 19]
F2 | Costly punishment of an offence | an offence, retaliation costs the punisher | strike, reprisal | Y | Fehr+Gaechter 2002 (u) [registry 20]
F3 | Revenge (delayed answer to a wrong) | a wrong, a safe chance later | anger memory, strike, steal | Y | McCullough 2001 (u); Boehm 1984 (u)
F4 | Revenge cycle / feud (A hits B, B hits A, loops up) | equal parties, a kin or group backing | no group backing; two-person loop possible | P:world | Boehm 1984 (u)
F5 | Redirected aggression (hit a safe third party) | angry at a strong party, a weak one present | anger vector, strike | Y | ethology (u)
F6 | Displacement activity (an irrelevant act under conflict) | drive conflict, e.g. fear vs hunger | wait, look around | P:law | Tinbergen 1963
F7 | Envy (resentment of another's holding) | another holds what one wants | wealth drive, steal, insult | Y | Brown 1991 (u)
F8 | Jealousy | a bonded partner favours a third person | no exclusive bond | N:law | Brown 1991 (u)
F9 | Shame (withdrawal after exposure) | a lie found out, in front of others | check flag; no shame response | N:law | Brown 1991 (u)
F10 | Pride and the insult response | an insult from an equal in public | insult, strike, pride trait | Y | Nisbett+Cohen 1996 (u)
F11 | Mediation (a third party stops a fight) | a fight, a neutral present | defend, shout, no go-between act | P:act | Brown 1991 (u)
F12 | Mercy (stopping when the loser yields) | a winner with a chance to finish | no yield act | N:act | Tinbergen 1963
F13 | Atonement / making amends | the harmer after a harm | give, no apology act | P:act | McCullough 2001 (u)
F14 | Loss looms larger than gain (anger after theft vs liking after a gift) | same value taken, same value given | steal, give, anger | P:log | Kahneman+Tversky 1979; extrapolated (weak) [registry 18]

G. BODY STATE, STRESS, STABLE DIFFERENCES
G1 | Hunger and harshness | energy deficit | needs | P:log | Bushman and others 2014 [registry 21]
G2 | Fatigue, harshness, less helping | rest deficit | needs | P:log | Kamphuis and others 2012 (u) [registry 22]
G3 | Drink and harshness; audience effect shrinks | ale drunk | drink level | P:log | Bushman+Cooper 1990 (u) [registry 23]
G4 | Own load and less helping | high need, another in need | needs | P:log | Darley+Batson 1973 (u) [registry 24]
G5 | Rank-order stability of harshness | many nights, 8 or more minds | nights | P:world | Roberts+DelVecchio 2000 [registry 25]
G6 | Stable across situations, less than across nights | situations varied | dials | P:world | Mischel 1968 (u) [registry 26]
G7 | Panic freezing (stillness under overwhelming threat) | close, inescapable danger | wait exists; no freeze law (threat locks action) | N:law | Marx and others 2008 (u); Roelofs 2017 (u)
G8 | Fight, flight, or faint | danger scaled to body power | flee, strike; fainting is wave 14b | P:law | Cannon 1915 (u)
G9 | Pain avoidance and limping | wounded | hp exists; pain as a state is not | N:law | Brown 1991 (u)
G10 | Self-medication (drink to dull fear or pain) | fear high, ale near | drink, safety need | Y | Khantzian 1985 (u)
G11 | Exhaustion collapse | rest need at its limit | sleep, rest drive | Y | Brown 1991 (u)
G12 | Illness behaviour (withdrawal when sick) | infection | no disease | N:law | Hart 1988 (u)

H. INFORMATION, DECEPTION, SPEECH
H1 | Lying for gain | a gain from a false tale | tell exists with lies | Y | Byrne+Whiten 1988 (u)
H2 | Lie detection and the liar's cost | the lie contradicts what the hearer saw | check flag | Y | Brown 1991 (u)
H3 | Gossip (telling of others' deeds) | a deed seen, a listener | gossip | Y | Dunbar 1996 (u)
H4 | Reputation management (hiding a bad deed) | a deed to hide, witnesses near | hide, steal | P:law | Brown 1991 (u)
H5 | Confession | a guilty person, a trusted listener | tell exists | P:law | Brown 1991 (u)
H6 | Secrecy and trust in a secret | a secret told to one person | tell has no "private" marker | N:sense | Brown 1991 (u)
H7 | Rumour spread and distortion | a tale passed through 4+ tellers | tell, hearsay beliefs; no drift | P:law | Allport+Postman 1947 (u)
H8 | Eavesdropping | a conversation in the next room | hearing exists for loud acts only | P:sense | Brown 1991 (u)
H9 | Curiosity / exploring the unknown | the curiosity drive | look around, open | Y | Berlyne 1960 (u)
H10 | Humour and joking | a group at ease | no humour act | N:act | Brown 1991 (u)
H11 | Music and singing together | company need, an instrument | play (lute) exists; no group act | P:act | Brown 1991 (u)
H12 | Dance | a group, music | no act | N:act | Brown 1991 (u)
H13 | Storytelling | a night, a listener | tell is facts, not stories | N:act | Brown 1991 (u)

I. MORAL EXTREMES
I1 | Self-sacrifice (accept harm so another survives) | a third person in danger, the helper at risk | defend, bandage; hp loss | P:law | Oliner+Oliner 1988 (u); Hamilton 1964
I2 | Heroism (risk for a stranger with no gain) | a stranger in danger, one can act | defend, shout | P:sense | Franco+Blau+Zimbardo 2011
I3 | Cowardice (leave a friend in danger) | danger, a friend in it, an exit | flee to | Y | Brown 1991 (u)
I4 | Martyrdom (accept death for a cause) | a cause or creed, a threat of death | no creed, no cause record | N:law | Atran 2003 (u); Whitehouse 2018 (u)
I5 | Betrayal of a friend (harm the one who trusts) | a big gain, a trusting partner | steal, accuse, aff | Y | Brown 1991 (u)
I6 | Obedience to harm on an order | a person in authority orders harm | no order act, no authority | N:act+law | Milgram 1963
I7 | Murder | high anger, a weapon, low cost | strike with a weapon; death law exists | Y | Brown 1991 (u)
I8 | Mass cruelty on a group member | a group with a shared enemy | no group | N:world | Staub 1989 (u)
I9 | Tolerating abuse (staying with a harmer) | dependence on the harmer | company and trade dependence | P:world | Brown 1991 (u)

J. GROUPS AND CROWDS
J1 | Crowd panic (a rush for the exit, crush) | many in one room, a danger | 5 people is too few; no crowd physics | N:world | Helbing and others 2000
J2 | Social contagion of fear | a person flees, others see | flee seen, safety drive | P:law | Brown 1991 (u)
J3 | Coalitions (two against one) | three or more, shared interest | no join act, no shared goal record | N:act+law | Harcourt+de Waal 1992 (u)
J4 | Scapegoating (blame the weak for a loss) | a loss, no culprit, a weak party | accuse, gossip; no group | P:world | Glick 2005 (u)
J5 | Ostracism (the group excludes one) | a norm broken, a group | no group, no shared decision | N:act+world | Brown 1991 (u)
J6 | Conformity | a majority act against one's view | perceived set; no norm | P:sense | Asch 1951 (u)
J7 | Bystander effect (less help with more watchers) | a person down, many present | help up; cast of 5 | P:world | Darley+Latane 1968 (u)
J8 | Social loafing | a shared task | work; no shared task | N:law | Latane 1979 (u)
J9 | Mobbing a predator or intruder | a threat to the group | no group | N:world | Curio 1978 (u)
J10 | In-group favouritism | two labelled groups | no labels | N:world | Tajfel 1971 (u)
J11 | Collective decision (vote or talk till agreed) | a shared choice | no act | N:act | Brown 1991 (u)
J12 | Alarm calling | danger seen by one | shout for help | Y | Seyfarth and others 1980 (u)

K. SUBSISTENCE AND BODY (ethogram)
K1 | Feeding, drinking | hunger, thirst | eat, drink | Y | ethogram
K2 | Resting, sleeping | rest need | sit down, sleep | Y | ethogram
K3 | Seeking shelter / a warm place | cold | no weather in the inn | N:law | ethogram
K4 | Caching food | plenty then scarcity | take, drop | P:law | Vander Wall 1990 (u)
K5 | Cooking and preparing food | raw food, a fire | serve; cooking is a voxel law | N:world | Brown 1991 (u)
K6 | Tool use | a task a tool helps | weapons only | P:law | Brown 1991 (u)
K7 | Healing the sick or hurt | a wounded person | bandage | Y | Brown 1991 (u)
K8 | Play (rule-bound or mock fight) | spare energy, a peer | play (lute) | P:act | Burghardt 2005 (u)
K9 | Vigilance (scanning) | danger likely | look around | Y | Beauchamp 2015 (u)
K10 | Territory and property defence | a held object | steal, defend, hide | Y | Brown 1991 (u)
K11 | Inheritance of goods | a death | no death turn in the cast | N:world | Brown 1991 (u)
K12 | Child care | an infant | no infant | N:world | Brown 1991 (u)

----------------------------------------------------------------------
2. THE THREE TESTS PER ENTRY
----------------------------------------------------------------------
Every entry first gets a DETECTOR and a CIRCUMSTANCE CARD, written before any run.
- Detector: a short rule over the typed event log plus the choice record (section 5). It says "this behaviour happened here". Example, H1 (lying for gain): a TELL whose content the teller's own beliefs mark false, and the teller's wealth or standing need falls within 20 turns. The detector is checked on a synthetic log first: one log with the behaviour planted, one without. It must fire on the first and stay silent on the second.
- Circumstance card: the ranges the real-world circumstances allow. Example, F3 (revenge): wrong done, a later turn where the wronged party is safe, no more than a night later. A found setting must sit INSIDE the card. This stops the search from "finding" a behaviour in an absurd world (everyone at 1 hp, no witnesses ever).

TEST 1, POSSIBLE (two levels).
 Level W, the WORLD can express it. A scripted probe forces the acts (the actor is made to do X) in the card's circumstances, and we read what the world does. Does the act exist? Does the world respond as the real circumstance would (a promise recorded, a death mourned, a freeze locked)? Forcing is a probe of the world only. It is not used on the mind. If level W fails, the entry goes to the missing-law log (section 8) with the missing item classed as act, law, sense or world setting. The mind is not at fault.
 Level M, the MIND reaches it unforced. A search over four spaces:
   dials   : traits (bravery, greed, pride, kindness, temper), the NATURE weights and set points of the seven drives, and the mind's own learning dials (learning rate, decay, partner weight PARTNER_W, anger discharge DISCHARGE, memory half-life ECHO_HALF, wear WEAR, lonely weight LONELY_W);
   history : a scripted prefix of events the actor lives through before the test window (kind then harsh, a theft, a death);
   state   : starting need levels, drink, hp, anger toward each person, beliefs;
   world   : cast size, who is present, scarcity (food in stock, STOCK), flags on or off (gossip, trade, watch, check, known, reprisal), room layout.
 Method: start at the card's own setting. Then a space-filling sample (Latin hypercube, 200 settings), then a local search from the best 5 (coordinate steps). Each setting runs 5 lives per seed, 3 seeds. The objective is the detector's rate in the test window. This is a PROBE of the world and the mind. Nothing here trains the mind or enters selection (principle: select for what persists). Budget per entry: about 2,000 night-lives at nice, 3 workers at most, under the shared lock. 131 entries then cost on the order of 260,000 night-lives. That is an estimate (u), not measured; the first 10 entries will give the real cost. Cheap entries (many events per night) are run first.
 Outcomes: FOUND (setting stored, inside the card; seeds that show it listed), NOT FOUND IN BUDGET (not the same as impossible; goes to triage below).
 Triage of NOT FOUND, in order: (a) does level W pass? If not, a missing act, law, sense or world setting. (b) Does the detector fire on a random chooser at a rate above a floor? If yes, the detector is too lax (fix the detector). (c) With level W passing, does a forced-act probe followed by free choice show the next acts that the real circumstance would bring (a follow-up)? If not, a missing law on the response side. (d) Otherwise the mind cannot represent the needed state or choice. That goes to the mind-side list.
 Guard against degenerate finds: a setting is rejected if it lies outside the card, outside the NATURE bounds, or if the behaviour appears only because of a bug signature (for instance, a replayed interrupted act, which is known from the calm round; the detector must exclude "resumed" acts).

TEST 2, TUNABLE.
 Pick the dial named in the entry (or the nearest 3). Sweep it over 5 levels, 10 seeds each, rest at the default. Report the rate at each level. Passed if: (i) the rank correlation of level and rate has the same sign in at least 8 of 10 seeds and the 95 percent interval over seeds excludes 0; (ii) the rate at the top level differs from the bottom by at least 0.02 in absolute terms (so it is a real move, not a nudge). Direction is monotone only; shape is report-only. "The dial moves it" is the claim; which dial is named in advance, from the entry, and not picked after seeing results.
 If no dial moves it, the result is TUNABLE: NO. That also goes to the missing-law log, since a behaviour with no handle is a hole in the dials.

TEST 3, TYPICAL BY DEFAULT.
 Default cast, default flags, 10 seeds, 600 nights (as in round 2). Report the rate per night per mind. Place it in one of four bins (common, occasional, rare, very rare) beside the entry's real-world bin, set by hand from the source or marked (u). Never scored, never in any loss. See section 6 for how it is read.

A cheap summary line per entry: "W:pass  M:found(setting 17)  T:yes(dial=pride, 0.02->0.11)  D:occasional vs real occasional".

----------------------------------------------------------------------
3. THE EVIDENCE BAR AND THE SEED LIST
----------------------------------------------------------------------
THE BAR. The behaviour EXISTS and the circumstances it happens in. That is all.
- One credible source is enough: a field study, an ethogram entry, a lab study, a historical record, or a standard textbook. An effect size is not asked for.
- Tiers, shown on the entry: field (observed in the wild or a society), lab (a controlled study), review (reviews or a handbook), depicted (only in fiction, see section 7), or common knowledge (Brown's list says it is universal). Fiction is the lowest tier: it shows what authors find plausible, not that people do it. A depicted-only entry stays in the catalogue but is marked, and a real source is looked for.
- The (u) mark is kept. A lookup pass can upgrade it. No entry is dropped for a (u).
- Contested findings stay in (hunger-aggression G1, the loser effect D2). The bar is existence in some circumstance, not agreement.

THE SEED LIST (what is in the table, and how it was drawn).
 (1) The 28 registry patterns: A1-A4, B1-B4 (B3 is registry 16, B4 is registry 17), B2, B13 (reputation partner choice, from the registry 5 source), C1-C3, D1-D4, E1-E4, F1, F2, F14, G1-G6. Their "more X, more Y" form becomes the TUNABLE direction; their old sealed set stays sealed for the "default matches" report of section 6. All 28 are in the table; a script check found none missing.
 (2) Donald Brown, Human Universals (1991), the Appendix list of universals. I used it from memory (u), not the book, so only the entries I am fairly sure are on it are here: language and speech, music, dance, play, gossip, humour and joking, ritual, taboo, kin groups and nepotism, reciprocity and gift giving, sharing of food, hospitality, division of labour, status and prestige, leadership, in-group and out-group, conflict and its mediation, revenge and redress of wrongs, murder, fear of death, mourning, funerals, deception, empathy, envy, jealousy, shame, pride, greeting and parting, promises, property and inheritance, trade, fairness rules, healing, cooking, tools, shelter, child care, courtship and pair bonds, and territory. Where Brown gives only the category, the entry says only what the category says. A lookup pass should check each against the real list; I expect a few entries to be missing from my memory.
 (3) Ethogram categories from animal behaviour (Tinbergen 1963 four questions; the standard ethogram handbooks): feeding, drinking, resting and sleep, locomotion, grooming, play, agonistic displays (threat, attack, submission, flight, appeasement), affiliative contact, food sharing, sexual and parental behaviour, vigilance, alarm calls, displacement activities, redirected aggression, reconciliation and consolation (de Waal), caching, tool use, mobbing, exploration, territorial defence, stereotypy under stress. K1-K12, D5, B10-B12, F5-F6 and J9, J12 come from here. Stereotypy under stress is NOT yet an entry; add it (pacing, a repeated act in a bad world). See the "gaps in the seed" note below.
 (4) Rare and extreme behaviours: martyrdom (I4), tyranny (D11), revenge cycles (F4), panic freezing (G7), self-sacrifice (I1), betrayal of a friend (I5), heroism (I2), cowardice (I3), deception (H1), forgiveness (A4), grief (B4), hoarding (E9), addiction (E6), ritual (E8), plus murder (I7), obedience to harm (I6) and mass cruelty (I8).
 (5) Group behaviours: crowds (J1 panic, J2 contagion), coalitions (J3), scapegoating (J4), ostracism (J5), conformity (J6), bystander effect (J7), social loafing (J8), mobbing (J9), in-group favouritism (J10), collective decision (J11).
 Gaps in the seed (add in the first review): stereotypy under stress; learned helplessness (give up after unavoidable harm); the endowment effect; mere exposure (liking from repeated contact); the "watching eyes" effect on kindness; nostalgia and homesickness; boredom-driven mischief; apology and its acceptance; trust falls and risk-taking in a partner; spite (hurting another at a cost to oneself). The first three are the cheapest to add (all reachable with existing acts).

DETAIL BY FAMILY: circumstances, what the inn or voxel sim would need, and why NOW is what it is.
 The table gives one line each. Here are the entries where the reasoning is not obvious.

 A. Reciprocity and cooperation. A1-A4 and A9, A14, A15 have every act (give, praise, help up, bandage, serve, refuse, steal, trade, reprisal). A5 (food sharing) and A6 (tolerated theft) are reachable: the cook holds food, hungry minds ask. A11-A13 are the real gap: the inn has no "promise" act and no record of a commitment. A promise needs three things: an act that records "I will do X by time T", a world law that notes when it is kept or broken, and a sense that lets others learn of it (gossip carries it). That is one law, and A12 (the reputation cost of breaking it) comes free from the existing gossip channel. A13 (fair split) needs a split act: "divide the food among those present". That is an act, not a law; the existing hunger drive already scores the result.

 B. Partners and bonds. B1, B2, B13 are in the log. B3 (separation distress) needs a per-person company need. The inn's company drive is generic: "any company" meets it. The missing sense is "the particular person is gone". B4 and B5 need a death or permanent exit inside a night; the dead flag exists on the person, but no menu path leads to it except violence, and nothing models the effect on a bonded partner. The wave plans' bleeding and injury work (wave 15) would supply the dying. The mourning law itself would be one rule: a person's weight on a lost partner decays slowly, from the prior contact (registry 17). B7 (courtship) needs a pair-bond drive and an act; this is a large addition and may stay out of scope of an inn. B8 needs kin relations in the cast. B9 and B10 are cheap acts (a greeting; a comforting touch). B12 is one act (an apology) plus its acceptance, which already exists as "accept".

 C. Risk. C1, C2 need the menu to show an outcome spread per option (registry 7's new field). C3 is reachable (steal when needy, with witnesses). C4 and C7 need a priced gamble act (dice exist as an object with no act). Whether they should exist in an inn is a design call; dicing is a real inn behaviour, so I lean yes. C5 (sunk cost) needs a "task in progress" record to measure what was spent; the mind does not hold one.

 D. Dominance and status. D1-D4 need a contest table (the registry's). D5 (submission) is missing as an act: today a threatened person can only give, refuse or flee. Real appeasement is a stand-down signal that ends a fight without damage, and the ethogram lists it as central. D7-D10 need acts and ranks. D9 (leadership) needs a follow act ("go where X goes"); without it, "others follow" is only the accident that they walk the same way. D11 (tyranny) is reachable in form (one strong party with a weapon, others weak), but its persistence needs the weak to be unable to ally (J3), which is why it is P:world. D12 (reverse dominance, Boehm) needs the coalition act.

 E. Learning, habit, addiction, ritual. E1, E3, E4 need probes or logs. E2 needs the two-clone probe from the registry. E6 (addiction) is a law gap: ale reduces a need, but there is no tolerance (need more for the same relief), no craving (a drive that rises with the drug's absence) and no withdrawal. The wanting/liking split of Robinson and Berridge (1993) gives the shape: a wanting signal that grows with use while liking falls. That is a pair of state variables and one law, and it falls out of the voxel body's existing energy ledger only partly (alcohol's effects on the body are not in it). E8 (ritual) needs a fixed-sequence act and a law tying it to stress; the cheap version is a "repeat a sequence" act whose only effect is to lower fear. That is a special case in disguise if built that way; the honest version needs ritual to arise from the mind's own habit and anxiety. I flag this as the hardest entry in the family; probably P:mind. E9 (hoarding) needs a holding limit or a cost to holding (weight, theft risk) so that hoarding is a choice rather than free.

 F. Loss, anger, punishment, revenge. F2, F3, F5, F7, F10 are reachable now. F4 (feud) is a two-person loop in the inn; real feuds need group backing (J3). F6 (displacement) is a law: a conflict between two drives that nearly tie should produce an irrelevant act (the ethogram's classic); in the inn the mind picks the max score, so there is no near-tie behaviour except by noise. F8-F9 are law gaps (jealousy needs an exclusive bond, shame a response to exposure). F11 (mediation) needs a go-between act. F12 (mercy) needs a "yield" act.

 G. Body state and stable differences. G1-G4 are the old patterns. G7 (panic freezing) is a missing LAW with a clean physiological form: under inescapable close threat, the fast fear response suppresses movement (tonic immobility is documented in animals and people; Marx and others 2008 (u)). A general version is a rule on the body's action selection: when fear is above a bound AND the escape options are blocked, the "act" rate falls. It must be a general law about the body's action gate, and not a flag on "panic". G8 (fight, flight or faint) depends on wave 14b fainting. G9 (pain) is a missing state: hp is not pain. G12 (illness behaviour) needs disease; out of scope for the inn but in the voxel sim's future.

 H. Information. H1-H3 and H9 exist (tell with lies, the check flag, gossip, curiosity). H4 needs "hide a deed": hiding exists for objects only. H6 (secrecy) needs a private marker on a tell. H7 (rumour distortion) needs a drift law in hearsay (a tale changes as it passes). H10-H13 (humour, dance, storytelling) are acts without a clear effect on a drive today: they matter through the company and curiosity drives, so each would be a "social" act of low cost that relieves company. The risk is making each one a special case. I recommend ONE general "shared activity" act (together with others, relieves company and curiosity, costs time) instead of four.

 I. Moral extremes. I3, I5, I7 are reachable now: flee from a friend in danger; steal from a trusting partner; kill with a weapon. I1 (self-sacrifice) needs the sim to let a helper accept a likely hp loss, which "defend" already does; it is P:law because nothing makes the sacrifice reach a stranger more than a friend. I2 (heroism) is the same for a stranger, plus a sense (does the mind see that a stranger is in danger?). I4 (martyrdom) needs a creed: a belief the person holds that can outrank its own safety drive. That conflicts with the mind-side principle that drives are the only weights. The consistent reading: a belief about a bond or standing (the people of my side) acts through the standing and company drives, with weights from history. The entry stays N:law until that design is settled. I6 (obedience) needs an order act and an authority; Milgram's finding comes from a role setting, so the world needs the roles first.

 J. Groups and crowds. A cast of five cannot show crowds. J1 needs perhaps 30 to 100 people in a room, and a crowd physics (voxel sim's momentum, exits). That is a world setting plus the mind speed to run 100 minds (the latency budget matters: reactive mode costs zero NPC LM calls, so the crowd minds must be the rules or learned-small kind). J3 (coalitions) is the central gap in the family: an act "join X against Y" that raises the allies' joint weight, and a belief of shared interest. J4, J5, J10 depend on it. J6 needs a norm or majority sense. J7 needs a larger cast than five to show "more watchers, less help". J12 exists (shout for help). J2 (fear contagion) is a sense law: seeing someone flee raises one's own fear.

 K. Subsistence and body. K1, K2, K7, K9, K10 exist. K3 (shelter from weather) belongs to the voxel sim (humidity and temperature are wave 14a). K5 (cooking) is a voxel law (combustion and heat transfer exist). K6 (tool use) is partly there (weapons). K4 (caching) and E9 (hoarding) are the same law. K8 (play): the lute's "play" is an act with an effect on company; true play (a mock fight with no damage) needs the damage gate: a blow with a tiny force and a mutual signal. K11 and K12 need death and children in the cast.

NOTE ON REACHABILITY AND THE VOXEL SIM. The table scores against the inn (lab/world2, an act-menu world). The voxel sim has real bodies, heat, fire, and soon contact and exertion (wave 14b). Some N entries (K3 shelter, K5 cooking, G8 faint, G9 pain, G12 illness) flip when the voxel world carries the minds. So the table has a second column to fill later: NOW-voxel. This draft scores only the inn, because that is where minds run.

----------------------------------------------------------------------
4. THE ENTRY TEMPLATE
----------------------------------------------------------------------
 id:            e.g. F3
 behaviour:     one plain sentence; no trait labels ("proud", "coward"); only what can be seen in events
 tier:          field | lab | review | depicted | common knowledge   (+ (u) if unchecked)
 source:        author, year, place (journal, book)
 circumstance card: the ranges (needs, who is present, what happened before, time window)
 detector:      the rule over the log; unit tests with a planted and an empty log
 dial(s):       named in advance; the nearest 3
 W / M / T / D: the four results above
 missing:       act | law | sense | world | log | mind | probe | none, and what exactly
 default bin:   the real-world bin and the measured bin (report only)
 first seen:    the seed list, the books, or a playtest
 status:        draft | detector written | W done | M done | T done | D done | retired

Retiring: an entry is retired only if the lookup pass finds no credible source that the behaviour exists. Not because the sim cannot do it.

----------------------------------------------------------------------
5. THE STATE-AT-CHOICE RECORDER (spec)
----------------------------------------------------------------------
WHY. The registry found that almost every pattern needs the actor's state at the moment of choice, and the world's event log does not hold it (it keeps id, t, actor, kind, traits, action, target, room, done, options text, chosen, p_char, p_avg, seen, facts). The catalogue also needs it for detectors, for the sweep reports, and for the books mapping. One recorder serves all of them.

WHERE. Lab-side only. It reads the world's state through the same view the mind has plus a separately labelled ORACLE block the mind never sees. The world is not changed. It writes one row per choice (a choice = a turn on which a mind picks from a menu), in a compact column file next to the event log, keyed by the event id.

FIELDS (each row).
 identity   : run id, seed, night, turn t, actor, actor type, mind id, flag set (the world's FLAGS), world-settings hash, dial vector (traits; NATURE weights and set points per drive; mind learning dials). The dial vector is stored once per life and referenced.
 needs      : the seven drive levels (energy, safety, company, standing, wealth, rest, curiosity), their errors above set point, plus hunger and thirst separately, the weighted total error.
 body       : hp, fear, pain (null until a pain state exists), fatigue (rest need), drink level, whether asleep, down, interrupted, what the actor is doing and its remaining turns, held things and their value.
 social     : for every other person: the actor's liking (aff), anger, distrust flag, the turns since that person's last kind act and last harsh act toward the actor, the counts of kind and harsh acts from and to that person in the last 40 and 200 turns, and whether the person is present.
 beliefs    : the actor's named beliefs, each with its content, the event it came from (seen, heard, told) and the turn. Alongside, in the ORACLE block, whether each belief is true. The mind never reads the oracle block; it is for the lab's tests and for lies and errors.
 perception : the ids of events perceived this turn (the perceived set); the people present; who could have seen the planned act (witnesses: the people present and awake).
 options    : for each option on the menu: act, target, the features shown to the chooser (value, danger, goal, need served, witness count), and, from the mind, its forecast mean and variance of the outcome (the new field registry 7 needs), its score components, the pick probability. And the index chosen. Whether the menu was cut or batched.
 history    : the last 20 events involving the actor (as actor or target), by id.
 outcomes   : filled in later, after the fact: at +1, +10 and +40 turns, the change in the actor's need errors, hp and each person's aff, and what the target did next. This is what lets a pattern's Y be measured without hand-written probes.
 rooms      : each person's room every 5th turn (and at every change), so that "time in the same room" (B2, D2) can be measured.
 contests   : a small table built from the log by the lab (initiator, target, outcome, turn). Outcome rules: yield (target gives, flees or falls after a threat or blow), and so on. Not a world field; a derived table.
 pairings   : a table of (contact event, drive error drop within 10 turns), for E1.

COST. About 1 to 2 KB a row if names are small integers (u, a guess). A night has a few hundred choices across five minds (u). Run storage for 10 seeds x 600 nights is then in the low gigabytes, which is why dev runs keep every row but tournament-scale runs keep one row in k (k=10, set before the run), and the probes keep all rows. Rows are compressed column files. Check: the disk budget on the machine; the .cache area is not git-tracked and the repo is PUBLIC, so nothing here (and nothing from the books) goes into the repo.

RULES. (1) The recorder never feeds back: a mind cannot read its own recorder. (2) It must not change a single event: the run with the recorder on must give the same event digest as the run with it off (the inn already has golden digests; reuse them). (3) The ORACLE columns are named oracle_* so a test can grep that the mind never touches them. (4) Fields are append-only and versioned, so old rows still read.

----------------------------------------------------------------------
6. THE ALIEN (FLIPPED) CONTROL: ITS NEW ROLE
----------------------------------------------------------------------
THEN. The alien registry was a pass-fail test of the mind: a mind that matched the real directions and failed the mirror list "looked human". Its weakness (registry section 4): the mirror scores are exact opposites, so most of the information is in the silent patterns.
NOW. The catalogue asks a different question (can the behaviour happen, and can it be tuned). That question does not need an alien. The alien keeps three smaller jobs.
 1. A POPULATION-DEFAULT CHECK, not a possibility test. For every entry that has a direction (more X, more Y; the 28 and others), the default cast under default settings gets one of three labels: MATCH (the real direction, interval excludes 0), CONTRADICT (the flipped direction), SILENT. The label says "the default population is off the human tendency here", which is a fact about the DEFAULT SETTINGS, not a failure of possibility. An entry that CONTRADICTs by default is only a problem if it is also not TUNABLE or not POSSIBLE; otherwise the remedy is a different default, chosen by a person.
 2. A SENSITIVITY CHECK of the detectors. The flipped version of a pattern is generated by a world run with a mirrored rule (for example, a world where kindness is punished). The detector must read the flip as the opposite sign. If it cannot, the detector is blind and its results mean nothing. This is the same job as the old "alien" test, moved to where it was always strongest: it tests the ruler, not the mind.
 3. The cheap-pattern filter, kept: seven null choosers (random, hermit, always-harsh, always-giving, rules, bandit, PPO) are scored on each entry's default direction. If 3 or more nulls match it, the default match is "cheap" (any survival learner gets it) and is not read as a sign of a human-like mind. Reported with the label, kept out of any headline count.
REPORTING. Three numbers per family, never one: matches, contradictions, silent. The old rule holds: a family with matches plus contradictions below half its entries is "silent", not a pass. No pass or fail on the whole catalogue; there is no whole-catalogue score.
TENSION TO DECIDE (question 4 below): "default off the human tendency, then move the default" is a fit to human data if a loop does it. If a person decides it by reading the report and edits a default, it is authoring, and is logged. I propose that: the loop never touches a default, and every change of a default is a dated, signed line in the catalogue's change log.
SEALED SET. The old sealed set (3, 8, 12, 13, 15, 18, 23, 26, 27) is kept for one purpose: the final "default matches" report, because that is the only place where fitting could leak. It does not affect possibility or tunability, since those are not scored against human data.

----------------------------------------------------------------------
7. HOW THE BOOKS FEED THE CATALOGUE (decision 5c)
----------------------------------------------------------------------
WHAT THE BOOKS HAVE. The decision-point cases (lab/books; a case holds the narrative, the chooser, KEY, NEEDS, BELIEFS, HOLDS, PERSONALITY, dialogue, a list of options, the option the book took, and the raters' acceptable set; the "hard subset" is where the options alone do not decide). Cases come from public-domain books (Crusoe, Huck Finn, Dracula, and so on) and from books held only locally (the LitRPG and cultivation books). Public repo rule: only case ids and catalogue ids may leave the cache; no text and no case data go into the repo.
WHAT THEY ARE EVIDENCE OF. Fiction is not behaviour data. A depicted choice shows that the author, and readers, find the behaviour plausible in those circumstances. That is "depicted", the lowest tier, and an entry backed only by it is marked so. That is still useful, because the catalogue aims at coverage of what stories contain, which is also what the player will want from a D&D-like game.
THREE USES.
 1. TAGGING (evidence, and seeds for new entries). Each case's chosen option (the book's own) is mapped to a catalogue id from the controlled list of entries. A case that maps adds a depicted citation (case id) to that entry's evidence. A case that maps to nothing is a NEW SEED: one line for the review. The share of cases that map is a direct measure of how well the catalogue covers what stories show (coverage of the catalogue itself). First-pass tagging: keyword and option-type rules on the option text plus a small number of hand tags; a rater pass on a sample (the existing rater pipeline, blind to the mapping) to estimate agreement. No GPU is needed for rules; an LM tag pass is optional and only after the GPU is free.
 2. CIRCUMSTANCE CARDS. A case's NEEDS become drives (hunger to energy, a wish for a place to standing), BEL becomes beliefs, HOLDS becomes held things, PERS becomes trait dials, and the narration gives the scene. This is the circumstance that goes on the entry's card (section 2), with the real behaviour that happened. It gives the card a realistic range and the search a realistic starting setting. A case that cannot be translated into inn terms (it needs a sea, a ship, a spell) is not discarded: it goes on the missing-law log with the missing item named (e.g. "ocean travel", "oath to a god"). That is the way the books tell us which gaps matter most to the play we want.
 3. THE HARD SUBSET AS THE BEST EVIDENCE OF CONTEXT-DEPENDENCE. By the decision-5(b) definition, the hard subset is the cases where options alone fail: the choice follows from the character's state, beliefs and history. Those cases give the sharpest circumstance-to-behaviour evidence, and the same cases can score a TYPICAL-BY-DEFAULT (does the mind's pick, given the translated state, fall inside the raters' acceptable set?). That score is reported, but never for tuning or selection (the books are a test, per earlier rulings).
WHAT TO KEEP AWAY FROM. (a) No selection on book cases. (b) A tag is not a measurement of how often the behaviour occurs in life; a long book has more "betrayal" than a life does. Book frequencies are never read as real-world frequencies, and are never used for the TYPICAL bin. (c) Adversarial filtering of options (decision 5a) comes later and only if a clean benchmark is needed; the catalogue does not need it.
OUTPUT. One table per book family: case id, catalogue id (or NEW), circumstance summary in inn terms, translatable yes/no, missing item. Kept in the cache, never in the repo.

----------------------------------------------------------------------
8. THE MISSING-LAW LOG (the main product)
----------------------------------------------------------------------
Every entry that fails level W, or has no dial, adds a line: entry id, item class (act, law, sense, world, log, probe, mind), the exact item, what else needs it. From this draft, by reading only (so provisional), grouped by what one addition would unlock. The grouping is the useful part, since it shows which small number of general additions covers most gaps, which fits the principle of general laws, not per-behaviour flags.
 1. GROUPS AND ALLIANCES (act: join/back X; a shared-interest belief; larger cast). Unlocks J3, J4, J5, J9, J10, J11, D11, D12, F4, A11 partly, I8. About 11 entries.
 2. COMMITMENTS (act: promise or oath; a law that records it, marks kept or broken, and carries it through gossip). Unlocks A11, A12, I4 (partly), I6 (partly), and underlies leadership D9 and ritual E8. About 5 entries.
 3. DEATH AND LOSS (a death inside a night; a per-person bond weight that decays slowly; inheritance). Unlocks B4, B5, K11, B3 (separation), F8 (needs an exclusive bond). About 5 entries.
 4. SOCIAL SIGNAL ACTS (greet, comfort, apologise, submit, stand down, mediate). Cheap acts that end or soften conflict. Unlocks B9, B10, B12, D5, F11, F12, F13. 7 entries. These are the ethogram's heart (appeasement, reconciliation), and their absence is probably why the inn thrashes in fights.
 5. SHARED ACTIVITY (one general act, together with others, relieves company and curiosity). Unlocks H10-H13, K8, partly E8. 6 entries.
 6. BODY LAWS (fear-freeze gate; pain as a state distinct from hp; addiction state: wanting vs liking, tolerance; illness). Unlocks G7, G9, E6, G12, F6 (displacement under drive conflict). 5 entries. Some depend on voxel wave 14b (fainting, exertion).
 7. MARKERS AND SENSES (a "mistake" marker, a per-person company need, a private-tell marker, outcome variance per option, a hearing law for overheard talk). Unlocks A3, B3, H6, H8, C1, C2, C5. 7 entries.
 8. WORLD SETTINGS (a bigger cast, children, kin, weather, a larger room for crowds; at least 30 minds in one room, which costs the mind speed budget). Unlocks J1, J7, G5, G6, B6, B8, K12, K3.
 9. LOGS AND PROBES (the recorder of section 5 plus the contest and pairing tables, and the two-clone probes). Unlocks 12 P:log entries and 3 P:probe. Cost: lab-only, zero world change. This is the step to do first, because it makes the 12 log entries Y.
Reading from the table, the first, second and fourth groups, plus the recorder, cover roughly 35 of the 98 non-Y entries (counted by hand, with overlaps) at the cost of about five general additions. That is a hint for the next wave plan, and it is only a read of the code; the real result is what the search finds.
Not logged here: the magic entries. Magic (decision list, 'one change') is a later set of entries: spell effects must be derivable from the laws, so their "possible" test is the faithfulness check of the magic plan, not this one.

----------------------------------------------------------------------
9. HOW TO BUILD IT, AND WHAT I NEED FROM YOU
----------------------------------------------------------------------
ORDER (all lab-side; nothing touches src/, tests/ or tools/, which wave 14a owns now).
 1. The recorder of section 5 and its digest check. About a day (u).
 2. Detectors and cards for the 33 Y entries and the 12 P:log entries (45 entries), with synthetic-log tests. A small set of 10 first: A1, A4, B1, B2, C3, D6, F2, F3, H1, I3.
 3. The search harness (section 2), and its first run on those 10. That gives the real cost per entry and a first look at POSSIBLE and TUNABLE.
 4. A lookup pass for the (u) sources; upgrade the tier.
 5. The books tagging (section 7) on the public-domain cases first.
 6. The missing-law log into the wave planner.
It all runs on CPU. Heavy runs go through the shared lock, nice, 3 workers at most.
RISKS. (a) A big catalogue is an attractive target for over-fitting: the rule is that no part of it enters a loss, a reward or a selection. (b) A "possible" via a degenerate setting is worth nothing; the card and the null-chooser floor guard it. (c) 131 detectors is a lot of code; many share parts (act sets, windows, contests), and the first 10 will show whether the shape is right before the rest are written. (d) Each missing item added to the inn risks becoming a special case (a "ritual" act, a "panic" flag). The test for every addition: is it one general law that other entries also use (section 8 groups), and does it give a handle the mind can use without a trait flag?

QUESTIONS FOR YOU.
 1. Scope: is 131 entries right, or should the first catalogue be only the Y and P entries (87), with the N entries kept as a wish list?
 2. The seed list gaps in section 3 (stereotypy, learned helplessness, endowment, mere exposure, apology, spite): add all?
 3. Do you want the missing-law log (section 8) to feed the next wave plan directly, with groups (1), commitments (2) and social signals (4) first?
 4. Default settings: is a human-made edit to a default, after reading the default-check report, within the "tweak it as needed" you asked for? Or must defaults stay untouched?
 5. Book evidence is "depicted", lowest tier. Is that right, or should fiction count the same as a lab study for this catalogue?
 6. The magic entries: a later separate set, with the faithfulness check as their "possible" test. Agreed?
```
