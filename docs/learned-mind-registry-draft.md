 # Learned-mind pattern registry: draft for review

Draft 2 plus a third check, 2026-11-08. It is **not hashed**: you review it first. It is report-only: nothing here is ever used in selection or tuning.

```text
LEARNED-MIND PATTERN REGISTRY, DRAFT 2 (fix round done; not hashed; for your review)
Written 2026-11-08, lane R. Nothing here has been run. No file in src/, tests/ or docs/ was touched.

PLAIN WORDS FIRST
- A "pattern" is a direction, not a number: more of X goes with more (or less) of Y. We never ask the mind to hit a human figure.
- X and Y are numbers the inn already logs (need levels, liking, witnesses, turns since an act). No labels like "proud" or "coward".
- 28 patterns. For each one there is a mirror-image "alien" pattern with the direction flipped. A good mind should match the real list and fail the alien list. A test that cannot tell them apart says nothing.
- Nothing here is ever in the loss or in tuning. Selection sees only persistence. These are report-only.
- DRAFT 2 NOTE: a skeptical check found wrong sources and an overstated 'measurable now' column. Both are fixed; see section 8 (Fix round). Only about 14 sources were searched, and only their abstracts were seen.
- SOURCES: about 14 were searched (abstracts only) and are marked 'confirmed by search'; the rest are from memory and marked (u). A third check (main session, from the second skeptic) corrected patterns 10, 11, 18, 19 and 28; see section 9. The authors, years and the direction of each finding I am fairly sure of. Effect sizes I am not. Before hashing, someone should look each one up (a web pass of about 28 lookups). "Evidence" says how strong the finding is as I remember it.

----------------------------------------------------------------------
1. THE TABLE
----------------------------------------------------------------------
"Now?" = can the inn measure it today from its logs. Y = yes, P = partly (a small recorder addition or a designed probe), N = no.
"Seal" = held-out probe, never looked at until the final report (section 5).

ID  Family        Pattern (more X -> Y)                                           Evidence            Now?  Seal
1   Reciprocity   Kind acts received from B -> more kind acts back to B           replicated          Y
2   Reciprocity   Kind acts B did to others that I saw -> more kindness to B      replicated, small   P
3   Reciprocity   Longer kind history of B -> less retaliation after one harsh    replicated          P     S
                  act from B (generosity after a mistake)
4   Reciprocity   More turns since B's harm -> more chance of a kind act to B     single paper         P
5   Partner       Higher liking of B -> more approach and asks to B               replicated          P
6   Partner       B's harm -> less time in the same room as B (exit)              theory+tests (u)      P
7   Risk          Energy budget below need -> choose the higher-variance option;  animals: replicated; P
                  above need -> the lower-variance option                         humans: mixed
8   Risk          The switch point sits at the mind's own set point, not at a     replicated (u)      P     S
                  fixed level (reference dependence)
9   Risk          More energy and wealth deficit -> more taking (steal, threaten, single, observ. (u) P
                  ask hard); more witnesses -> less taking                       country (u)
10  Dominance     A recent win -> more contests started, also against others      reviews, animals       P
11  Dominance     A recent loss -> fewer contests, more avoidance                  reviews, animals       P
12  Dominance     Yields between minds form a linear order (transitive triads     replicated, animals P     S
                  above chance)
13  Dominance     At equal influence, holders of prestige (sought out) are liked  single + replicated (u) P  S
                  more than holders by fear (threat then give)
14  Acquired      More pairings of B with drive relief -> more approach to B;     replicated, old     P
                  gain slows with each pairing
15  Acquired      Approach to a once-rewarding B dies out SLOWER if the reward    replicated, robust  N     S
                  was intermittent than if it was every time
16  Acquired      More prior contact with B -> more seeking of B and more         replicated          P
                  company need when B is gone
17  Acquired      After B is lost, the weight on B falls steadily; the more prior  mixed (u)           N
                  contact, the slower
18  Loss          A loss of value (theft) -> bigger anger and retaliation than    extrapolated (weak) P     S
                  the same value given -> gratitude
19  Loss          A harm from a well-liked B -> bigger drop in liking and bigger  no source yet       P     cand
                  response than the same harm from a stranger
20  Loss          Retaliation rises with the size of the offence and falls with   replicated, strong  P
                  its cost to me, even when it has no future benefit
21  State         More hunger (energy deficit) -> more harsh acts                 single, contested      P
22  State         More fatigue (rest deficit) -> more harsh acts, less helping    weak, mixed (u)    P
23  State         More drink -> more harsh acts, and the witness effect shrinks   meta-analysis       P     S
24  State         More own total need -> fewer costly kind acts (give, help up)   single, replicated  P
                  to others                                                       in part (u)
25  Individual    A mind's rank among minds on harsh-act share is stable from     meta-analysis       P
                  night block to night block
26  Individual    Rank-order stability across SITUATIONS is above 0 but below     replicated          P     S
                  stability across nights of the same situation
27  Order         Same acts, different order: the first acts weigh more in the    replicated          P     S
                  liking of B (primacy)
28  Order         An act's pull on later behaviour toward B falls steadily with   replicated          P
                  the turns since it (recency / decay)

Counts: 28 patterns. Measurable now (corrected against world.py, third check): 1 Y, 25 P, 2 N. Almost every P needs ONE recorder addition, the state snapshot at choice (see the log note below). Sealed: 9 (19 unsealed, third check). Families as you asked: reciprocity (1-4), partner choice (5-6), risk under need (7-9), dominance and prestige (10-13), acquired values (14-17), loss aversion and betrayal (18-20), state (21-24), individual differences (25-26), order effects (27-28).

The inn's logs, as I read world.py (_emit, lines about 965-991), CORRECTED in draft 2:
- Per event the world stores: id, t, actor, kind, traits, action, target, room, done, options (text), chosen, p_char, p_avg, seen (the events the actor perceived before choosing), plus facts for tell and gossip.
- It does NOT store, at the moment of choice: the actor's seven need levels, liking (aff) of anyone, anger, drink, rest, hp. It does not store option features (liking, witnesses) either.
- "Who SAW an act" is not on the event. It can be rebuilt from each person's perceived set (workable, not logged today).
- So only patterns that use acts alone (who did what to whom, and when) are measurable now: 1 and 28. Everything that needs a state at choice time is P: it needs ONE mind-side recorder line per choice (needs, aff to each person, anger, drink, rest, hp, witnesses present). This is a recorder addition on the lab side; the world does not change.
- From mind-side files (lab/mindL): partner-model values, the weights on each drive, the valence signal. A "measured weight on standing" comes from a swap test (set the weight to another mind's, see the action change), not from reading a number.
The words "win", "yield", "mistake", "variance" are not logged today. Each "P" below says what to add.

----------------------------------------------------------------------
2. HOW TO READ AN ENTRY (the one-page example you asked for)
----------------------------------------------------------------------
Every entry has the same fields. Example, pattern 4:

 id:        R4
 statement: for a given actor A and partner B, the more turns since B's last harmful act toward A, the higher the chance that A's next act toward B is kind.
 X:         turns since B's last act in HARMFUL aimed at A, capped at 200.
 Y:         1 if A's next act aimed at B is in KIND_ACTS, else 0 (acts that are neither are dropped).
 control:   A's liking of B at that moment, and A's anger at B (so we do not just measure that anger fades).
 statistic: Spearman rho between X and Y inside each life, after taking out the control by ranking residuals; then the sign across seeds.
 direction: positive.
 tolerance: a life counts only if it has at least 30 (X, Y) pairs; a seed set passes if the sign test and the interval both point positive.
 falsifier: a rho at or below 0 with an interval that excludes 0 on the negative side.
 source:    McCullough and others 2010, Emotion 10(3):358-376 (author list u); Wu and Axelrod 1995, J Conflict Resolution 39:183-189 (confirmed). Evidence: single paper with several samples.
 status:    needs the state snapshot for the controls.

Every pattern below follows that shape, in short form: WHY (the idea), X, Y, DIRECTION, SOURCE, NOW?

----------------------------------------------------------------------
3. THE 28 PATTERNS
----------------------------------------------------------------------

RECIPROCITY AND ITS LIMITS

1. Direct reciprocity.
   X: number of kind acts (KIND_ACTS) from B to A in the last 40 turns. Y: chance A's next act toward B is kind.
   Direction: Y rises with X.
   Source (u): Axelrod and Hamilton 1981 (tit for tat); Fehr and Gaechter 2000 (reciprocal responses, public goods). Replicated.
   Now: yes. Warning from round 3 step 0a: in today's inn a kind act gets a kind reply only 3 percent of the time (background rate). A mind that fits the world will fail this. That is a finding about the world, not a flaw in the pattern. It is the one pattern where we EXPECT a fail until the world is fixed.

2. Indirect reciprocity (reputation).
   X: number of kind acts by B toward third parties that A saw or was told of (perceived set, gossip tells), minus harsh ones. Y: chance A is kind to B on first contact after.
   Direction: Y rises with X.
   Source (u): Nowak and Sigmund 1998 (image scoring, a model); Wedekind and Milinski 2000 (experiment: people give more to those who gave before). Replicated in lab games, small effect. (Sylwester and Roberts 2010 was moved to pattern 5: it shows cooperators gain PARTNERS, which is partner choice, not reputation raising giving.)
   Now: partly (needs the witness set and the state snapshot). Step 0a found gossip carries 88 bad deeds per good one, so the X range is narrow. Need: a log of what A saw about B toward third parties, per turn.

3. Generosity after a mistake. (SEALED)
   X: A's count of kind acts from B in the last 100 turns before B's one harsh act. Y: chance A retaliates (a HARSH_ACT to B) within 20 turns.
   Direction: Y falls with X.
   Source: Wu and Axelrod 1995, J Conflict Resolution 39:183-189 (generous tit for tat under noise; confirmed by search, abstract only); Van Lange and others 2002 (generosity under noise). Replicated in lab games.
   Now: partly. The inn does not mark a "mistake". Use a proxy: a harsh act that was interrupted, or came while B's drink was high or aim was low. Need: log the actor's drink at the act (it exists in state) and treat "isolated harsh act after 5+ kind acts" as the test cell.

4. Forgiveness over time.
   X: turns since B's last harmful act aimed at A (capped at 200). Y: A's next act toward B is kind (1) or not (0).
   Direction: Y rises with X, controlling for liking and anger.
   Source: McCullough and others 2010, 'On the Form and Function of Forgiving: Modeling the Time-Forgiveness Relationship and Testing the Valuable Relationships Hypothesis', Emotion 10(3):358-376 (journal, volume and title found by search; author list from memory, u). Several samples inside one paper; abstract only seen. Evidence: single paper, several samples. Valued relationships forgive faster: this is also why liking is the control.
   Now: partly. Turns since B's harm is in the event log; the liking and anger controls need the state snapshot. Without the controls it is Y but can only show that anger fades. This is the worked example in section 2.

PARTNER CHOICE

5. Approach follows liking.
   X: A's liking of B (aff, -1..1). Y: share of A's social acts (ask, offer, walk to, praise, gossip) that go to B among the people in reach.
   Direction: Y rises with X.
   Source: Barclay and Willer 2007, 'Partner choice creates competitive altruism in humans', Proc R Soc B 274:749-753 (found by search; abstract only); Sylwester and Roberts 2010, 'Cooperators benefit through reputation-based partner choice in economic games', Biology Letters (title from memory, u); Roberts 1998 (u). Evidence: replicated lab games.
   Now: partly. Who A acts toward is in the event log; A's liking (aff) at that moment is not (state snapshot).

6. Exit after harm.
   X: size of B's harm to A (sum of SLIGHT of acts aimed at A in the last 40 turns). Y: share of A's turns in the same room as B over the next 40 turns.
   Direction: Y falls with X.
   Source (u): Barclay 2013 (partner choice); Baumard, Andre and Sperber 2013 (mutualistic morality). Evidence: theory with some experiments, not a clean replication (downgraded in draft 2).
   Now: partly. Room is logged per event, not per turn. Need: record each mind's room every turn (or every 5th).

RISK UNDER NEED

7. Energy-budget risk sensitivity.
   X: A's energy budget gap = (sure option's expected energy gain) minus (energy needed to reach the set point before the night's end). Y: chance A takes the higher-variance option when the options differ in variance and have the same mean.
   Direction: negative gap (sure option falls short) gives more risk; positive gap gives less. Ordinal: Y falls as the gap rises.
   Source: Caraco, Martindale and Whittam 1980, Anim Behav 28:820-830 (juncos; confirmed by search); (u below); Stephens 1981; Kacelnik and Bateson 1996 (review). Replicated in animals; in humans mixed (Mishra and Lalumiere 2010 find it in some tasks).
   Now: partly. The menu does not carry an outcome spread, and the energy gap is not logged (needs the state snapshot plus a NEW outcome-variance field). Need: at each choice, log each option's forecast mean and variance from the mind's own model, and mark option pairs with equal means.

8. The switch sits at the set point. (SEALED)
   X: A's energy level relative to its OWN set point (NATURE gives a different set point by type). Y: as in 7.
   Direction: the preference flips where A crosses its set point, and the crossing moves with the type's set point. Ordinal and single crossing: Y(below set point) > Y(above set point) for each type, with no second crossing.
   Source (u): Kahneman and Tversky 1979 (reference dependence); Caraco and others 1980 as above. Replicated in animals; humans mixed.
   Statement (draft 2, ordinal only): (a) for each type, rate of the high-variance option is higher below its set point than above; (b) across types, the crossing level rises with the set point (rank correlation across types). Report-only: the crossing is not required to sit at a given number.
   Now: partly. Same recorder as 7. Test: compare the crossing level across types; it should track their set points (0.15 to 0.35), not sit at one number.

9. Taking under need.
   X1: energy deficit and wealth deficit of A; X2: witnesses of the act (size of perceived set). Y: chance A picks steal, threaten, or hard-ask when it could.
   Direction: Y rises with X1 and falls with X2.
   Source (u): Wilson and Daly 1997 (inequality and homicide, single-region study); Becker 1968 (economics of crime, theory); Fehr and Gaechter 2002 (u). Evidence: observational for the need part; single study for homicide. Draft 1 said 'replicated'; this was too strong.
   Now: partly. The act is logged; need levels and witnesses need the snapshot and the rebuilt witness set. Note: round 3 step 0a found the extractor loses even with reprisal off, so the sign on X1 may be flat in this world.

DOMINANCE AND PRESTIGE ROUTES

10. Winner effect.
   X: whether A "won" its last contest (B yielded, fled, fell, or gave after a threat) in the last 60 turns. Y: chance A starts a new contest (threaten, strike, shove) against someone else.
   Direction: Y is higher after a win.
   Source: Rutte, Taborsky and Brinkhof 2006, 'What sets the odds of winning and losing?', TREE 21:16-21 (confirmed by search; abstract only; winner and loser effects in animals); Mazur and Booth 1998 is about testosterone, not win-loss behaviour, so it is dropped as a source. Evidence: REVIEWS of many animal studies (Rutte 2006; Hsu, Earley and Wolf 2006), with no pooled effect size; human evidence weak.
   Now: partly. "Win" is not logged. Define it from existing events: threaten followed by give; a blow that makes the target flee or fall. Need: a contest table (initiator, target, outcome, turn).

11. Loser effect.
   X: whether A lost its last contest in the last 60 turns. Y: chance A starts a contest; also share of turns in the room of the winner.
   Direction: contests fall, avoidance rises. 'The loser effect lasts longer or is stronger' is often reported in the reviews but not universal: REPORT-ONLY, never scored.
   Source (u): Rutte and others 2006 as above; Hsu, Earley and Wolf 2006, Biol Rev 81:33-74 (confirmed by search). Evidence: reviews of animal studies, no pooled effect size (Hsu, Earley and Wolf 2006 is titled 'Modulation of aggressive behaviour by fighting experience: mechanisms and contest outcomes'); human evidence weak.
   Now: partly. Same contest table.

12. A linear order forms. (SEALED)
   X: the yield matrix over many nights (who gave way to whom). Y: Landau's h, and the share of transitive triads.
   Direction: h and the transitive share are above what a shuffled matrix gives (shuffle who beat whom, keep totals).
   Source (u): de Vries 1995 (test of linearity); Chase and others 2002 (orders from identical animals through chance wins and winner effects); Appleby 1983 (transitivity in pecking orders).
   Now: partly. Needs the contest table and at least 6 to 10 pairs with 5 or more contests each. The inn has 5 minds of different types, so this tests an order, not the "equal individuals" claim. That claim needs experiment E-D (five identical-nature minds) from the doc.

13. Two routes to standing, and who is liked. (SEALED)
   X: a mind's route share = (threat-then-give events won) / (those plus times it was asked or offered to). Y: the mean liking others hold of it (mean aff toward it).
   Direction: at the same level of influence (number of times others yield to or ask it), liking is higher for the low-route-share (prestige) holder.
   Source (u): Henrich and Gil-White 2001 (prestige vs dominance); Cheng, Tracy, Foulsham, Kingstone and Henrich 2013, JPSP 104:103-125 (confirmed by search; abstract only). Evidence: single study with partial replication.
   Now: partly. Needs yields and asks counted per mind, over a cast, many nights. Weak with one focal mind and four rules minds.

ACQUIRED VALUES

14. Secondary reinforcement.
   X: number of pairings of B (contact) with a drop in A's weighted drive error within 10 turns. Y: A's approach to B (walk to, ask, stay in the room) when A is not in need.
   Direction: Y rises with X, with falling gains per pairing (concave). Placebo check: a person who never co-occurs with relief must show no rise.
   Source (u): Pavlov; Kelleher and Gollub 1962 (review of conditioned reinforcement); Berridge and Robinson 2003 (cue wanting). Replicated, old.
   Now: partly. The pairings are in the logs. The concern table of doc 9.2 is not built, so today this tests behaviour only. Needs: a pairing log (contact event, error drop, turn).

15. Intermittent rewards are harder to extinguish. (SEALED)
   X: schedule of B's gifts: every time versus about one in three. Y: nights of approach to B after the gifts stop.
   Direction: approach lasts longer after the intermittent schedule.
   Source (u): Humphreys 1939; Mackintosh 1974 (the partial reinforcement extinction effect). Replicated, very robust in animals.
   Now: no. Needs a designed probe: two clone minds, one gift-giver on each schedule, then the giver goes silent (doc experiment E-A). Cheap: a few CPU hours.

16. Attachment and separation.
   X: prior contact with B (turns in the same room plus kind acts exchanged). Y: after B leaves or is silent, A's rise in company need and its walking to where B was.
   Direction: Y rises with X.
   Source (u): Bowlby 1969; Ainsworth and others 1978 (strange situation); Harlow 1958 (contact over feeding). Replicated.
   Now: partly. Company need is generic in the inn, not per person. Need: a per-person absence log and a search/follow count.

17. Grief fades slowly, slower with more contact.
   X: prior contact with B at the time B is lost (dies, leaves for good). Y: the weight A still puts on B over the following nights.
   Direction: the weight falls steadily, and the half-life is longer for larger X.
   Source (u): Parkes and Weiss 1983 (bereavement); Bonanno 2004 (most people recover; wide spread). Mixed.
   Now: no. Needs deaths or permanent exits of cast members inside a life, and the concern module. Listed so it is not forgotten; it joins the score only when the module exists.

LOSS AVERSION AND BETRAYAL RESPONSE

18. Loss looms larger than gain. (SEALED)
   X: value of a thing (THINGS value) that A lost by theft versus a thing of equal value A received as a gift. Y: A's anger rise and its retaliation chance (loss) versus A's liking rise and kind reply (gain).
   Direction: the loss response is larger than the gain response, at equal value. Ratio above 1 is the test. 'Humans give about 2' is REPORT-ONLY context, never scored.
   Source (u): Kahneman and Tversky 1979; Brown, Imai, Vieider and Camerer 2024, J Economic Literature 62:485-516 (meta-analysis of loss aversion in CHOICES over gambles; mean 1.955; confirmed by search). Evidence: the meta-analysis covers choice; anger and retaliation after a theft is an EXTRAPOLATION, a weak fit.
   Now: partly. Value is known for each thing. Needs matched cells (same value, theft versus gift) which are few; group by value tercile.

19. Betrayal costs more from the trusted. (CANDIDATE, unsealed: no direct source yet)
   X: A's liking of B just before B's harmful act. Y: A's retaliation chance and A's drop in liking of B in the next 20 turns.
   Direction: Y rises with X, for harms of equal size.
   Source: NOT YET FOUND. Bohnet and Zeckhauser 2004 (JEBO 55:467-484) is about one-shot betrayal aversion toward STRANGERS and only partly replicates (Fetchenhauer and Dunning; Hong and Bohnet); Koehler and Gershoff 2003 is about harm from products meant to protect. Neither tests prior liking. Evidence: indirect. UNSEALED: this stays a candidate until a source on harm from close versus distant others is found.
   Now: partly. Needs A's liking of B BEFORE the act. That is the state snapshot; it is not logged today (draft 1 said yes; wrong). It stays sealed because its code is tested on synthetic data only. Note: whether the DROP in liking scales with prior liking is a world rule. Score the retaliation chance as the mind's own measure and report the drop as a world check only.

20. Costly punishment.
   X1: size of the offence (SLIGHT of the act toward A). X2: A's cost of retaliating (witnesses who would lower A's liking, B's strength or weapon, A's hp). Y: chance A retaliates.
   Direction: Y rises with X1 and falls with X2. REPORT-ONLY (a level, not scored): Y stays above zero on the last night, when retaliation can bring no future gain.
   Source (u): Fehr and Gaechter 2002 (altruistic punishment); Henrich and others 2006 (across societies). Evidence: replicated lab games and cross-cultural; strong.
   Now: partly. The offence type is in the event log; witnesses, B's strength and A's hp need the snapshot. Night index is available.

EFFECT OF STATE ON AGGRESSION AND HELPING

21. Hunger and harshness.
   X: A's energy deficit. Y: A's share of harsh acts (insult, threaten, shove, strike, refuse) among its social acts.
   Direction: Y rises with X.
   Source (u): Bushman, DeWall, Pond and Hanus 2014 (low blood sugar and aggression, one lab and a spouse study). Evidence: single study (107 couples, search confirmed; criticised by Lange and Kurzban, so contested). Weakest evidence in the list; flagged. Citation: Bushman, DeWall, Pond and Hanus 2014, PNAS (confirmed by search; abstract only).
   Now: partly (needs the snapshot).

22. Fatigue and harshness, helping.
   X: A's rest deficit (a QUANTITY of rest). Y1: share of harsh acts. Y2: share of kind acts that cost A something (give, help up, bandage).
   Direction: Y1 rises with X; Y2 falls.
   Source (u): Kamphuis, Meerlo, Koolhaas and Lancel 2012, review of poor sleep and aggression, Sleep Medicine (not searched, u). Supporting single study: Barnes, Lucianetti, Bhave and Christian 2015, AMJ 58(5): leaders' sleep, daily abusive supervision and work-unit engagement. That is ONE 10-day experience-sampling study, and its abstract supports sleep QUALITY, not quantity. It is NOT about ethical behaviour (draft 1 was wrong).
   Evidence: weak. The review is the main support and I have not read it; the one study is about quality. The inn's rest is a quantity, so the match is loose. Flagged as weak, a candidate to drop.
   Now: partly (needs the snapshot).

23. Drink raises harshness and shrinks the audience effect. (SEALED)
   X: A's drink level. Y: share of harsh acts; and the slope of Y on witnesses.
   Direction: Y rises with X; the (negative) slope on witnesses gets smaller in size with more drink.
   Source (u): Bushman and Cooper 1990 (meta-analysis of alcohol and aggression); Steele and Josephs 1990 (alcohol myopia). Evidence: meta-analysis (u, not searched).
   Now: partly. Drink is in the world state but not in the log (snapshot). Note the blow's aim is already worse with drink by world rule; score the CHOICE of a harsh act, not the damage.

24. Own need load and helping.
   X: A's total weighted drive error. Y: chance A gives, helps up or bandages when B is in need (hp low, or asked).
   Direction: Y falls with X.
   Source (u): Darley and Batson 1973 (hurry and the Good Samaritan, one study, partly replicated); Cunningham and others on mood and helping (u).
   Now: partly (snapshot). Check separately that the effect is not just "no time left", by using only choices with the same turn cost.

INDIVIDUAL DIFFERENCES THAT STAY STABLE

25. Rank-order stability.
   X: night block (nights 1-100 versus 101-200, and so on). Y: each mind's share of harsh acts, its share of social acts, its energy deficit.
   Direction: Spearman rank correlation of minds across blocks is above 0, and is higher when more nights are pooled (the SIGN of the gain is scored; the Spearman-Brown shape is report-only).
   Source: Roberts and DelVecchio 2000, Psych Bull 126:3-25 (meta-analysis; about .31 in childhood to .74 in middle age; confirmed by search, abstract only); Epstein 1979 (aggregating over occasions).
   Now: partly. Needs 8 or more different minds (an evolved population or archetypes) run in the same world. One focal mind gives no rank order.

26. Stable across situations, but less than across nights. (SEALED)
   X: situation (fed versus hungry nights, watched versus unwatched, ordinary versus shock). Y: the same share of harsh acts.
   Direction: rank correlation across situations is above 0 and below the correlation across two nights of the same situation. Both parts needed.
   Source (u): Mischel 1968 (person and situation); Fleeson 2001 (within-person variation); Funder and Colvin 1991 (cross-situation consistency). Replicated.
   Now: partly. It depends on pattern 25's population AND its recorder (draft 1 said yes, but 25 is P).
   Scored as two parts: 26a r_situation > 0; 26b r_night > r_situation.

ORDER EFFECTS

27. Primacy. (SEALED)
   X: order of the same acts by B toward A: kind, kind, harsh versus harsh, kind, kind. Y: A's liking of B and A's next act toward B after the third.
   Direction: the first acts weigh more: liking after (kind, kind, harsh) is above liking after (harsh, kind, kind).
   Source (u): Asch 1946 (impression formation); Anderson 1965 (weighted averaging). Replicated, though recency shows up when judging after delay. The sealed test uses the IMMEDIATE-judgement cell only.
   Now: partly. Not a natural log pattern. A designed probe like the L-tests: two partners with permuted histories. Score the mind's partner model and its actions, not the world's aff.

28. Recency and decay.
   X: turns since an act of B's. Y: how much the act changes A's next act toward B (compare A's choice toward B with and without that act in the history).
   Direction: the pull falls steadily with X (monotone; scored as an ordering of the lag bins. 'Close to a power law' is report-only).
   Source (u): Anderson and Schooler 1991 (power-law forgetting); Hogarth and Einhorn 1992 (belief-adjustment). Replicated.
   Now: partly. Y is a counterfactual (the choice with and without the act), so it needs a designed probe (as in the L-tests) or a model-side regression; it cannot be read from the log alone.

----------------------------------------------------------------------
4. THE ALIEN CONTROL
----------------------------------------------------------------------
IDEA. Same X and Y for every pattern, direction flipped. The "alien registry" is the world as it would be if people were backwards. A mind that is a real learner should match the real list more than the alien list. A mind that matches both equally, or that scores the same on a random chooser, shows the test is empty.

Flips (A = alien):
 1  A: more kind acts from B -> FEWER kind acts back.
 2  A: seeing B do kind acts to others -> LESS kindness to B.
 3  A: longer kind history of B -> MORE retaliation after one harsh act.
 4  A: more turns since harm -> LESS chance of a kind act.
 5  A: higher liking -> LESS approach.
 6  A: B's harm -> MORE time in B's room.
 7  A: budget short -> take the LOWER variance; budget met -> the higher.
 8  A: (a) the high-variance rate is LOWER below the set point than above; (b) across types the crossing level FALLS as the set point rises (negative rank correlation).
 9  A: more deficit -> LESS taking; more witnesses -> MORE taking.
10  A: a win -> FEWER contests.
11  A: a loss -> MORE contests, less avoidance.
12  A: a NON-linear order (cyclic triads above chance).
13  A: at equal influence, fear-route holders are liked MORE.
14  A: more pairings with relief -> LESS approach.
15  A: intermittent rewards extinguish FASTER.
16  A: more prior contact -> LESS seeking after loss.
17  A: more prior contact -> FASTER fade of the weight on a lost partner.
18  A: gain response larger than the loss response (ratio below 1).
19  A: harm from a trusted partner -> LESS response than from a stranger.
20  A: larger offence -> LESS retaliation; higher cost -> MORE retaliation.
21  A: more hunger -> FEWER harsh acts.
22  A: more fatigue -> fewer harsh acts, MORE costly helping.
23  A: more drink -> fewer harsh acts, and a STRONGER audience effect.
24  A: more own need -> MORE costly helping.
25  A: minds' ranks flip between blocks (negative rank correlation).
26  A: (26a) r_situation < 0; (26b) r_situation > r_night. Each half flipped separately.
27  A: in the immediate-judgement cell, liking after (harsh, kind, kind) is ABOVE liking after (kind, kind, harsh) (recency, the mirror of primacy).
28  A: an act's pull GROWS with the turns since it.
Check of flips (draft 2): 1-7, 9-12, 14-25 and 28 are clean sign flips. 8, 26 and 27 were not true mirrors in draft 1 and are rewritten above. 22's alien flips both halves, which is right. 3's and 19's flips are about the same cell as the real test.

THREE WAYS THIS CAN FOOL US, AND THE FIXES
(a) Mirror scores are exact opposites. If a mind matches 15 of 28 and contradicts 3, its alien score is exactly 3 minus 15. So "real beats alien" is true for any mind with more matches than contradictions, and says little by itself. The information is in the SILENT patterns (interval includes 0). So we report three numbers, not one: matches, contradictions, silent. A run is informative only if matches plus contradictions is at least 14 of 28. Below that, the whole suite is silent and we say so.
(b) Some flips are what a pure survival learner does anyway (pattern 6, 9, 20 in this inn: avoiding the retaliator is cheap and wins). The null choosers settle this: score random, hermit, always-harsh, always-giving, rules, the bandit and PPO on the real list. For each pattern, count how many nulls match it. A pattern that 3 or more of the 7 nulls match is "cheap" (non-diagnostic): kept in the report, dropped from the headline count. This is the check that would have caught round 2's mind, which provoked the kind partner rather than the retaliator (an anti-pattern 1 and 20 behaviour that still looked sensible from the inn's costs).
(c) A second alien that is not an exact mirror: the SHUFFLED alien. Keep each pattern's direction, but swap its Y with another pattern's Y within the same family (for example, hunger -> more kind acts instead of more harsh acts, with the sign kept). It is not the negative of the real score, so it gives a separate floor. A mind that scores as well on the shuffled alien as on the real list is matching "any drive-sensible direction", not the human one.

PASS RULE (proposed). The mind passes the control if: (i) matches minus contradictions on the headline set is above that of every null chooser, by a paired interval over seeds; (ii) it beats its own shuffled-alien score by at least 4 patterns; (iii) the informative count is at least 14. Pass means "consistent with these patterns". It does not mean "appropriate", and the (mostly WEIRD, lab) source bias is named, not removed.

----------------------------------------------------------------------
5. HOW EACH PATTERN IS SCORED
----------------------------------------------------------------------
STATISTIC. Always ordinal (shapes and levels named in the entries, such as concave, power law, ratio about 2, Spearman-Brown, are REPORT-ONLY and never scored; the scored part is a sign or an order): a Spearman rho or Kendall tau inside each life between X and Y (for tests with two X, a rank regression with each X in turn). A pattern with a flip or a crossover (8, 26) is scored as an ordering of two group means, not as a fit. Binary Y (a kind act or not) is turned into a rate per X bin (4 bins, equal counts), and the order of the bins is the test.
LIFE-LEVEL RULE. A life gives a number for a pattern only if it has at least 30 (X, Y) pairs (at least 8 per bin). Else that life is "no data". If fewer than 60 percent of lives have data, the pattern is "silent, thin", not a fail.
SEED LEVEL. Each life gives a sign (+, -, no data). Across seeds: a sign test plus a 95 percent t interval on the life-level rho. Match = both point in the predicted direction and the interval excludes 0. Contradict = the interval excludes 0 on the wrong side. Else silent.
NUMBER OF SEEDS.
- Development patterns (18 of them): 10 seeds, 600 nights, as in round 2. A 10/10 sign test is p = 0.002 two-sided; that just misses the Holm threshold for 28 tests (0.05/28 = 0.0018). So development results are treated as "looks", with Holm applied to the look ledger, and not as pass/fail claims.
- Sealed probes (10): 16 seeds, because 15 of 16 agreeing is p = 0.0005, below the first Holm step, and 16 leaves room for 2 or 3 seeds with no data. The pre-registered rule is: at least 15 of 16 point the same way (15/16 is p = 0.0005; 14/16 is p = 0.004 and would miss the Holm step) and the interval excludes 0. This is strict on purpose: a sealed pattern that only 13 or 14 seeds show is reported as 'leaning', not as a match.
- Designed probes (15, 27, and 12 in the E-D version): the effect is within one comparison, 20 pairs of clones (not 20 lives of one mind), because the cost of each is small.
WHAT IS NOT SCORED. Levels. We never test "the rate is 12 percent". Only signs, orders, ratios (pattern 18 ratio above 1) and crossovers.
LOOK LEDGER. Every time anyone opens a pattern's result, one look is logged with the date and the run it was on. Holm correction is over all looks to date. A pattern looked at more than 3 times during development is retired from the sealed set and replaced from the spare list (below).
SPARE LIST (not in the 28, ready to swap): mere-exposure liking (Zajonc 1968; Bornstein 1989 meta-analysis) (u); bystander effect on helping a fallen person (Fischer and others 2011 meta-analysis) (u); watching-eyes effect on kindness (weak, Northover and others 2017 meta-analysis) (u); effort justification (Aronson and Mills 1959; Norton, Mochon and Ariely 2012 on effort and value) (u); endowment effect in trade refusals (Kahneman, Knetsch and Thaler 1990; contested) (u).

HELD-OUT PROBES (SEALED): 3, 8, 12, 13, 15, 18, 19, 23, 26, 27. They are chosen to differ in KIND: two reciprocity/loss-of-trust (3, 19), one risk (8), two dominance (12, 13), one acquired (15), one loss (18), one state (23) and two stability/order (26, 27). The rule is that nobody (not the loop, not the tuning, not a person reading a report to decide what to change) sees their results until the final held-out report. Their code is written and unit-tested against synthetic data with a known direction, never against inn runs. If one is looked at early, the look is logged and the pattern moves to the development set.
DEVELOPMENT SET (18): 1, 2, 4, 5, 6, 7, 9, 10, 11, 14, 16, 17, 20, 21, 22, 24, 25, 28. They may be read during development to find world or mind bugs. They are still never in any loss, reward or selection: "development" means "may be looked at", not "may be tuned on".

----------------------------------------------------------------------
6. SKIPPED, AND WHAT WOULD UNLOCK THEM
----------------------------------------------------------------------
- Ultimatum-game fairness, trust-game amounts: no money-split act in the inn. Skipped, not forced.
- Similarity attraction, in-group favouritism: the inn has no groups or features to be similar on.
- Pattern 17 (grief) and 15 (extinction): need designed probes and the concern module.
- Reprisal for third-party harm ("moral outrage" at harm to someone else): only if gossip carries enough kind and harsh deeds; step 0a says it does not yet.
- Culture, norms, religion: out of scope (doc 9.7).

----------------------------------------------------------------------
7. WHAT I NEED FROM YOU
----------------------------------------------------------------------
1. Are 18 and 21 (loss aversion ratio, hunger and aggression) worth keeping? 21 is the weakest evidence in the list; I kept it because you named hunger. 18 depends on enough matched theft and gift cells.
2. The pass rule in section 4 (three numbers, null choosers, shuffled alien). Is it too strict?
3. Do you want the sealed set chosen by me (above) or by a coin flip from the 28 with the kind constraint? A coin flip is more honest; mine is more balanced.
4. Allow the lookups: a full-text pass on every source. Draft 2 checked about 14 by search (abstracts only). Not yet searched: Kamphuis 2012, Fehr and Gaechter 2000 and 2002, Nowak and Sigmund 1998, Henrich and Gil-White 2001, Wilson and Daly 1997, Darley and Batson 1973, Bushman and Cooper 1990, Steele and Josephs 1990, Anderson and Schooler 1991, Chase 2002, de Vries 1995, and others. Most (u) tags remain.
5. Recorder additions (all in the lab-side log, none in the world). The big one, new in draft 2: a STATE SNAPSHOT at each choice (seven needs, aff to each person, anger, drink, rest, hp, who is present). Then: per-turn room, a contest table, option forecast means and variances (new outcome-variance field), a pairing log, a witness set per act. About 6 fields. Is that all right, given that the world must not change?

----------------------------------------------------------------------
8. FIX ROUND (draft 2, answering the skeptical check)
----------------------------------------------------------------------
Kept: all 28 patterns, the families, the sealed set (10) and the development set (18), the alien control idea, the pass rule, the scoring plan.
Fixed:
1. Pattern 4 source (high). Now cited with the real journal, volume, pages and title (Emotion 10(3):358-376). Evidence downgraded from 'replicated' to 'single paper, several samples'.
2. Pattern 22 source (high). Barnes and others 2015 is a leaders' sleep, abusive supervision and engagement study (AMJ 58(5)), one 10-day experience-sampling study, supporting sleep quality only. It no longer claims 'ethical behaviour', 'reviews' or 'quantity'. Marked weak and a candidate to drop. The review claim now rests on Kamphuis 2012, which is not yet searched.
3. 'Measurable now' (high). Rechecked against world.py _emit. The event log holds id, t, actor, kind, traits, action, target, room, done, options text, chosen, p_char, p_avg, seen, and facts. It does NOT hold needs, aff, anger, drink, rest or hp at the choice, nor option features. Count is now 2 Y (patterns 1 and 28), 24 P, 2 N (15, 17), instead of 12 Y. Patterns 5, 9, 19 to 24 and others moved to P: they need one STATE SNAPSHOT line per choice. Pattern 26 moved to P because it depends on 25. 'Who saw it' is rebuilt from each person's perceived set, noted as not logged. Pattern 19 stays sealed but is P (needs liking before the act).
4. Sources found real by search (abstract only): Wu and Axelrod 1995; Rutte and others 2006; Bushman and others 2014 (with the Lange and Kurzban criticism noted); Caraco and others 1980; Cheng and others 2013; Brown and others 2024; Bohnet and Zeckhauser 2004; Roberts and DelVecchio 2000; Barclay and Willer 2007; Sylwester and Roberts 2010; Hsu and others 2006; Koehler and Gershoff 2003. Full texts were not read.
5. Loose citations. Mazur and Booth: author line marked unconfirmed. Sylwester and Roberts moved from pattern 2 to pattern 5 (partner choice). Barclay and Willer given its real title. Pattern 6 and 9 evidence wording lowered.
6. Alien flips (medium). 8, 26 and 27 rewritten as true mirrors (see section 4). 8 and 26 are split into two sub-claims each, flipped separately.
7. Ordinal leaks (low). 'About 2', 'power law', 'Spearman-Brown', 'concave' and the 'positive Y on the last night' level are marked REPORT-ONLY. Pattern 7 now lists the new outcome-variance recorder field.
8. Still not done: most (u) tags (about 16 of 28 patterns still rest on unchecked sources). That is the full-text lookup pass in section 7 question 4. Nothing here has been run on the inn.

----------------------------------------------------------------------
9. THIRD CHECK (main session, 2026-11-08, from the second skeptic's findings)
----------------------------------------------------------------------
- 10, 11: evidence relabelled from "meta-analysis" to reviews of animal studies (no pooled effect). Mazur and Booth (testosterone) dropped. "The loser effect lasts longer" is report-only.
- 18: the loss-aversion meta-analysis covers choices over gambles. The anger and retaliation version is an extrapolation, marked a weak fit.
- 19: no direct source. The cited studies test betrayal aversion toward strangers, or harm from protective products. UNSEALED until a source on close versus distant harm is found.
- 28: needs a designed probe; it is not readable from the log ("partly", not "yes"). So only pattern 1 is measurable now.
- Header: sources are about 14 searched (abstracts) and the rest from memory (u).
- The sealed set loses 19, leaving 9. The 16-seed and Holm figures in section 7 should be recomputed for 9.
```
