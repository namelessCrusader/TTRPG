# Knowledge: what each person believes, and how it spreads

Scoped with the owner on 2026-11-21 (decisions K1-K11). It builds on the plan-cache decisions Q15-Q20 (docs/plan-cache-design.md, section 15b). It is to be built in WAVE 16 together with the plan cache, after wave 15 and before magic.

Prior work: two library answers and two open requests in the research library (~/Desktop/Fun Stuff/Library):
- responses/2026-11-21-mark1-knowledge-transmission.md (78 papers);
- the open requests requests/2026-11-21-mark1-belief-representation.md and requests/2026-11-21-mark1-belief-systems-games.md.

## The standing direction

- The long-term aim is a real simulation of a brain. Every process below is a SCAFFOLD behind a replaceable mind interface.
- The literature's measured effects become ACCEPTANCE TESTS that any later cognition model must pass. They are not rules.
- Few primitives, each with a source and a test. The owner: "the hardest part would be modelling the complex cognitive processes in human brains into smaller primitives."
- General laws only: one belief structure and one transmission law for every person and every kind of knowledge. No per-topic rules, no trait flags, no god-channel.

## K1. What a belief is

- RECORDS AT THE BOUNDARY, LEARNING INSIDE.
  - A belief is a typed claim, in the same closed language as the plan cache's success tests (registered quantities, pointers to world things, ordinal values).
  - The claim is what is logged, passed between people and tested against the world.
  - Behind the interface, a mind may store beliefs any way it likes: a typed table first, a learned memory later that still reads and writes typed claims at the boundary.
  - Free text is only how a belief is SHOWN.
- It starts from the existing `Belief` (src/mind/api.py) and store (src/voxel/beliefs.py), extended to FOUR KINDS:
  1. how to do something (plans);
  2. how the world works (laws);
  3. what happened or what is (events, places, holdings);
  4. who someone is (people).
- A claim the language cannot express is logged as a missing quantity.
- Note: the representation needs MUCH more work, especially a learned store for the learned mind.

## K2. What a belief carries

1. **The claim.**
2. **Confidence:** how sure the holder privately is.
3. **Sources:**
   - how it was acquired (saw, did, told, shown, rumour, inferred), from whom and when;
   - the COUNT OF DISTINCT people heard from, the best single predictor of adoption in the literature;
   - source memory and content memory fade SEPARATELY ("Tom told me" can become "someone told me" while the claim stays). Corrected by the library's second pass: the evidence that source fades faster is mixed, so the two rates START EQUAL and are both logged, and the effect is tested, not assumed.
4. **Evidence:** tests and sightings, each with its outcome AND the perceived reason.
5. **Salience:** surprise and emotion. It drives telling and survives retelling.
6. **History:** the belief it replaced. Records only.
7. **The link to the true event.** Records only, NEVER visible to the mind.

The HEDGE ("apparently", "I think") belongs to the TELLING, not the belief. A hedge can be lost on retelling (K4), and a liar can state firmly what they privately doubt.

## K3. How a belief passes

- NOTHING HAPPENS AUTOMATICALLY.
  - Telling, warning, gossiping, teaching, showing, asking and lying are acts the mind chooses, each with its real cost.
  - Listening is perception through the sound and sight models: overhearing, noise, distance, walls and deafness all apply.
- Incentives for spreading come only from the mind's own drives and learned values: company, standing, trading information, protecting people it values, curiosity for asking. Never a rumour rule.
- People who acted on something pass it on more often than people who only heard it (about 0.5 against 0.1 per neighbour per round in one field study).

## K4. What a retelling loses

Three primitives, each rate tuned so that retelling chains match the measured numbers:
1. **The teller's memory.** Each part of a belief fades at its own rate: source fastest, then detail, then gist, and salience last.
2. **The channel.** The sound model decides what physically gets through.
3. **The listener.** Limited attention per telling, and gaps filled from the listener's own beliefs (drift toward the stereotype).

Hedges drop without a rule. The listener sets confidence from the hedge and trust, but the memory of the hedge fades like the source, and a retelling hedges from current confidence.

**Close to lossless is possible at a cost, never exactly lossless.**
- Careful telling: repetition, read-back, slow speech; it costs time and attention.
- Trained memory: mnemonics and verse, a trained skill.
- THE WRITTEN WORD, a physical record. It keeps the claim, the hedge and the source, but:
  - it needs literacy and materials;
  - it decays physically (fading, fire, water);
  - it gains small scribal errors per copy;
  - it still passes the reader's attention and interpretation;
  - it can be false, forged or out of date.
- Records are ONE kind of thing across knowledge, plans and magic (the recorded spells of magic Q8).
- Warning from the literature: written theories passed down a chain did not speed progress, and they narrowed what the next person tried.

## K5. Believing what you are told

- Believing is NOT a chosen act; it happens, like perception.
- One process, whose inputs are:
  - the number of distinct sources (the second and third raise adoption sharply, levelling by the fourth);
  - their agreement;
  - the listener's own confidence;
  - the cost of testing it yourself;
  - TRUST in the teller, learned from their track record (did their past claims hold?);
  - discounting reports from those you disagree with.
- A credulity dial is drawn from a distribution and shaped by history and state (tiredness, fear). There are no "gullible" or "sceptic" types.

## K6. Lies

- No special mechanics. A lie is telling a claim you do not hold, or stating firmly what you doubt.
- Detection:
  - INCONSISTENCY with what the listener saw, believes or hears from others (the strong signal);
  - weak physical tells from the body model (arousal, sweat, gaze, voice) seen through the senses, at about 54 percent accuracy, near chance (u). Practice suppresses tells; knowing a person's baseline helps.
- Discovery lowers track-record trust, and the discovery itself can spread.
- Beliefs about others' beliefs, to ONE level ("Tom thinks I was at the mill"). They also serve teaching (not telling people what they know) and secrets.

## K7. Correction

- ONLY through what the person perceives.
  - A seen cause corrects the belief behind that step (the plan cache's per-step supports say which).
  - An unseen cause is blamed by the person's own reasoning: the least-sure belief, or an explanation they already hold. So WRONG BLAME, superstitions and folk theories persist.
- A weakened belief with no alternative explanation keeps a foothold until a better story arrives.
- A plan that works can carry a wrong belief about why it works, and success never corrects it; only a test that varies the neglected factor does.
- Deliberate testing is a chosen act (curiosity, doubt).
- The after-run report of a plan is a test of the beliefs its steps rely on.

## K8. Forgetting

- Power-law forgetting per part of a belief.
- Use (acting on it, telling it, being reminded) refreshes it, and spaced use builds lasting strength. Salience slows fading; practised skills fade slowest.
- Below recall the belief is forgotten. Tip-of-the-tongue comes from the same curve.
- A cap stays only as a cost limit, set high, and how often it binds is measured. BEL_CAP 64 with oldest-first eviction is to be replaced.

## K9. Offscreen people (built later, once there are several settlements)

- Per locality and belief, keep the holder count and PAIR-LEVEL co-knowledge (who knows it together with whom). A summary that treats people as independent overstates spread.
- Anyone with a rare or newly seeded belief stays explicit.
- People entering the scene are drawn from the summary.
- Validate against explicit runs over many seeds (final share, time to half the locality, pair co-knowledge), AND on interventions (a plan taught, a false belief tested), not only on free spread.
- Travellers carry part of their repertoire and learn by the same rule as everyone, so nothing merges. A locality's "common knowledge" is a statistic over its people.

## K10. The player and the character

- The CHARACTER'S beliefs decide what they can do: the menu and plans come from them, labelled by the character's own experience.
- The PLAYER may still attempt anything legal by free intent, and the sim judges it by physics.
- The character's beliefs are shown in plain words, with confidence and source.
- Free text invokes only what the character has shown proficiency in. Knowing THAT is not knowing HOW.

## K11. The build (wave 16, with the plan cache)

The first slice is onscreen only, built in the main sim first, with a matching version in the lab inn afterwards:
- the four kinds with the seven fields;
- the acts (tell, ask, warn, teach, show, lie);
- the three retelling primitives;
- believing, forgetting and correction;
- one level of beliefs about beliefs;
- written records as objects.

**Acceptance tests from the literature** (numbers from lab tasks and toy models, so the tolerances need setting):
- retelling chains: length falls about 55 percent at the first retelling and about 77 percent by the third, leaving about a quarter of the details;
- hedges drop within one or two retellings, and acts get more specific;
- social content survives best;
- adoption rises with the second and third distinct contacts and levels off by the fourth;
- trust follows the track record;
- lie detection is near 54 percent;
- a false belief that nobody tests survives;
- forgetting follows a power law, and spacing helps;
- a skill survives in proportion to the number of distinct holders a learner can sample, not the headcount.

**MARK1's own measurements** (no paper gives them):
- the fidelity of a multi-step plan with its reasons, per channel (expect telling to keep goals and lose reasons);
- false-belief correction by testing in a small population with lossy transfer;
- later, the validated offscreen summary.

## Refinements from the library (2026-11-21)

From responses/2026-11-21-mark1-belief-representation.md and responses/2026-11-21-mark1-belief-systems-games.md (54 papers). They CONFIRM the design and refine it. None of these changes an owner decision.

- **The record is Talk of the Town's belief facet** (Ryan et al. 2015: value, strength summed from evidence, evidence with source, place and time, a predecessor, an accuracy flag), with three fixes it lacked:
  1. **Count TIMES HEARD and DISTINCT PEOPLE HEARD FROM as two separate fields.** Repetition by one person must not count as several sources; the truth effect (repetition raises belief) then stays measurable against the world.
  2. Source memory and content memory fade separately (already K2).
  3. No per-topic tables: drift goes toward the hearer's prior, and salience comes from the hearer's drives (already K4).
- **Trace to cause, with an assumption-based truth maintenance system** (de Kleer 1986).
  - Each belief carries the SOURCE EVENTS it rests on, as a short list of (event, how, teller, tick).
  - A claim found false later flags everything derived from it.
  - Retraction is the costly part. Use the paper's fix: add a defeasibility assumption and contradict that instead.
  - Today's one-source-per-belief store is the trivial special case.
- **One revision law for every fact** (Talk of the Town):
  - the first evidence is adopted;
  - supporting evidence adds strength;
  - telling a belief slightly strengthens it for the teller;
  - contradicting evidence is adopted only if stronger, and otherwise kept as a CANDIDATE that gathers evidence and wins once it is stronger.
  - The AGM postulates (success, minimal change) serve as audit checks on logged changes.
- **Confidence:**
  - Log the INITIAL confidence at first encoding as its own field; it is the informative one (Wixted and Wells 2017). Calibration by how a belief was acquired is then a measurable curve.
  - For quantities, an interval that WIDENS with time and narrows when sensed is a cheap, general decaying confidence (Funge, Tu and Terzopoulos 1999).
- **Two emergent effects to test:**
  - THE SLEEPER EFFECT: if a source's credibility fades on its own schedule, rumours from poor sources gain acceptance later.
  - SOURCE CONFUSION (Johnson et al. 1993): people judge the source at recall, and err. A separate mechanism may blur `how` and the source while keeping the value; an explicit `how` field idealises people.
- **The learner's interface:**
  - typed claims read as key-value slots, write(key, value) and read(query), so the inside can change while the interface stays (key-value memory networks);
  - learned internals give no records, so LOG ONLY WHAT CROSSES THE INTERFACE;
  - the features a learner gets: how, the distinct-source count, repetitions, time since the source, the teller's trust, and its own initial confidence.
- **Beliefs about others:** one level per person plus a shared "popular belief" (Teutenberg and Porteous 2015). That popular belief is the locality statistic of K9.
  - Full epistemic planning is undecidable in general.
  - Bayesian theory of mind costs about 0.86 to 28 s a step, so it is OFFLINE ONLY.
- **LLM memory streams stay OUT of reactive play.** They fail exactly these requirements: wrong speaker attribution, embellished memories, and a model call per update.
- **The records-only truth link has precedent in every working system** (the Dwarf Fortress spy's true identity "to keep the game honest").
- **THE MAIN DESIGN RISK IS LEGIBILITY, not CPU.** Shipped games cut knowledge systems for player legibility and authored control. Tarn Adams avoided fallible memory because it "would appear to arise from bugs if not artfully expressed". So show WHY someone believes something (owner question K12).
- **Cheap audits on the logs:**
  - each NPC act is checked against that NPC's own believed state (plans consistent with a character's beliefs were judged most believable);
  - a "story sifter" query layer over the TRUE records finds rumour chains and lies forming, without giving any mind access to them.
- **Offscreen (K9):** materialise detail lazily but consistently with what was already seen (alibi generation: 20,000 agents, about 52 kB of precomputed data, about 9 ms an alibi). Avoid "decide the event, invent the cause later" (Caves of Qud), which leaves no truth to audit.
- **Firsts for MARK1** (no source reports them):
  - the per-character cost of a provenance-carrying store at a few hundred people;
  - whether players read misremembering and lies as intended or as bugs;
  - retelling with decay between NPCs in a running game.

## K12. Legibility: why someone believes something (owner, 2026-11-21)

- IN THE WORLD during play: the player asks "who told you that?", and the NPC answers from its OWN source memory, which may have faded or be confused ("someone at the market, I think"). A wrong belief always has a reason that can be dug up in the world.
- An OPTIONAL AFTER-SESSION REVEAL, read from the true records (the story sifter): how a rumour really travelled, who lied, where a story changed. It never feeds back into any mind, and it shows no spoilers during play.
- A DEBUG INSPECTOR for checking the sim, off by default.

## Second pass from the library (2026-11-21)

From requests/2026-11-21-library-mark1-beliefs-second-pass.md (64 papers) and requests/2026-11-21-library-mark1-knowledge-second-pass.md (38 papers). MARK1's answers are in each note.

- **Source against content fade:** separate parameters, starting equal, both logged (K2 corrected above).
- **Confidence as evidence counts:**
  - each belief keeps (r, s), supporting and contradicting, plus its acquisition time;
  - the expectation is (r+1)/(r+s+2), and uncertainty falls as evidence grows (subjective logic, the beta form);
  - independent tellers are fused by ADDING counts; tellers whose claims trace to the SAME origin are DEPENDENT and are AVERAGED. That is the formal form of "times heard" against "distinct people".
- **Teller trust:**
  - counts per (holder, teller), updated when a told claim or plan is later tested (TRAVOS; Yu and Singh);
  - a distrusted teller adds uncertainty, not contrary belief;
  - trust RECOVERS through the same forgetting law, because old evidence fades; there is no separate rule.
  - Gossip moves belief weighted by the teller's confidence, never equally per teller.
- **Forgetting:** a power law in odds form, p = a/(a + t^b), computed AT READ from the acquisition time, with no per-tick update. Rubin and Wenzel: across 210 data sets the exponential fits worst. Refit for game time; Brom's character-memory parameters are a reference check.
- **Write rules with no model call:**
  - the same key with a new value closes the old row and opens a new one (MemStrata);
  - a retracted source cascades its dependants to "needs verification" or "historical only" (PACMI);
  - nothing is deleted.
- **Procedures (the plan cache):**
  - a retold plan is scored at TWO LEVELS per hop and channel, structure (sub-goals and order) kept better than details;
  - steps whose effect the learner can see pass better;
  - a responsive holder's demonstration has the highest fidelity (a complex knot 4.60 of 5 taught against 2.67 shown as an end state); telling has the highest substitution rate;
  - losing a prerequisite makes its dependants unusable (Kolodny 2015).
- **Gossip, a CORRECTION of an earlier reading:** field gossip is 63 percent "definitely true" and 35.9 percent negative (Dores Cruz et al., N = 309, 5,284 events). So the lab inn's 88:1 bad-to-good ratio points at the inn's SALIENCE RULE, not at gossip as such. To be checked in the realistically-hard-inn work.
- **False rumours:**
  - a hoax dies when forgetting is fast relative to its credibility;
  - retractions rarely reach holders of the false belief, so per false belief log how many holders could ever perceive a contradiction;
  - an offscreen summary may stand in only where beliefs fade fast.
- **Emergent tests added:**
  - a frequent liar comes to believe the lie (Ryan's "declaration", from the K2 revision law);
  - the sleeper effect;
  - the hoax threshold;
  - the one-step attractor regression per scalar property along a retelling chain;
  - recoverable source facts per hop.
- **Offscreen and history:** Talk of the Town needed "several hours" per town to simulate knowledge over a whole pre-history. It simulated the last week and implanted the rest, and the K9 plan does the same.
- **Legibility:** show forgetting on a timeline in the character's knowledge view (Shadows of Doubt). Ryan calls intrigue hidden behind menus "the Tale-Spin effect".


## Full-text upgrades from the library (2026-11-21, after the owner's PDFs)

- **Sleeper effect** (Kumkale and Albarracín 2004, full):
  - delayed change from a discounted message is d = 0.08 overall, the same as no message;
  - an effect of about 0.25 appears only when the discounting cue comes AFTER the message and the hearer can and will think about it;
  - relative fading of source against message was never estimated. This backs "start the two fades equal".
- **Source monitoring** (Johnson, Hashtroudi and Lindsay 1993, full):
  - the source is INFERRED AT RETRIEVAL from perceptual, contextual, semantic, affective and cognitive-operation cues, with shifting criteria; there are no decay curves;
  - ASKING FOR THE SOURCE at test removed eyewitness suggestibility. For K12: when the player asks "who told you?", the NPC's answer goes through the same inference, and the asking itself makes them more careful (a testable effect).
- **Mesoudi and Whiten 2004** (full): along 4-person chains, low-level detail falls linearly while medium- and high-level summary RISES, and event order is kept almost perfectly (4 of 404 retained propositions out of order). This backs the two-level retention of plans and stories.
- **Bad News** (full): a 140-year town is generated in about 5 MINUTES, which is the cost precedent for a coarse pre-history. Confidence in a fact decays without reinforcement, and characters spread false facts unknowingly or lie. There is no player study of false beliefs.
- **Open Design Challenges** (full): there is NO knowledge-propagation challenge in it, only the "Tale-Spin effect" (depth the player never sees). An earlier card note was wrong.
- **Tarn Adams, "Characterization and Emergent Narrative in Dwarf Fortress"** (full, doi:10.1515/9783839453452-007):
  - about 50 personality facets, about 30 values, needs and stress;
  - a TWO-LAYER MEMORY in which each season's strongest events per category become PERMANENT and change personality;
  - summed happiness caused tantrum spirals (a warning for any summed-mood design);
  - numbers are hidden and personality is shown as text.

## The uncanny: general laws from the library (2026-11-22)

From responses/2026-11-22-mark1-uncanny-coverage.md (88 papers). All are general laws of perception, memory, retelling and fear. None is a "spooky" flag.
1. **A perceptual criterion.** Seeing a signal in noise is a CRITERION SHIFT: the criterion falls with the expected base rate, darkness, a told expectation, and lost control (a 25 to 75 percent base rate moved false alarms from 0.15 to 0.26; belief added about 9 percent per SD). "Haunted" rooms work this way: dim, featureless rooms plus a hint gave 94 percent an odd sensation, while EMF and infrasound did nothing (French 2009). Reports track local cues (exterior light, room size).
2. **Memory scaled by SURPRISE in context,** saturating beyond 2 to 3 violations.
   - The counter-intuitive advantage is small (d 0.19 to 0.57) and shows mainly after a delay and in stories.
   - Social-schema violations beat physics ones, and agents add.
   - Counter-intuitive content is remembered but BELIEVED far less (d about 2). Belief follows witnesses who ACT on it.
   - REFINED after the full texts (2026-11-22, library note pdf-upgrades):
     - ONE VIOLATION PER ITEM transmits best: in Barrett and Nyhof's 3-generation retelling chains, one-violation items went from 3.98 to 2.72 against bizarre 3.22 to 1.29 and ordinary 2.28 to 0.89 (about 2x survival), and 35 against 17 percent in oral recall at 3 months;
     - 37.5 percent of kept bizarre items DRIFTED INTO counter-intuitive ones;
     - COMBINING TWO violations in ONE item lowered recall (Boyer and Ramble, adults);
     - at STORY level, 2 to 3 anomalous items (successful Grimm tales);
     - the "small effect" holds only for list recall.
     So: one change per thing, a few strange things per story, which fits MARK1's one change to physics.
3. **Lossy, emotion-biased retelling:**
   - copy error about 0.1 per hop (about 11 percent of online copies mutate);
   - a credible source dropped in about 37 percent of retellings and invented in about 3 percent;
   - DISGUST predicts pass-on (low-disgust variants died out within 4 hops);
   - hazard claims are believed more (d about 0.55), and urban legends are about 3:1 hazard stories;
   - tale types spread by descent more than by distance.
   This refines K4 and K5.
4. **A fear state with separable parts:** unpredictable threat gives a sustained state; imminence switches planning to reflex; dread over a known wait makes people prefer a worse but sooner harm (79 percent took the shorter wait). Enjoyment is an inverted U in PER-SCARE fear (quadratic beta -0.05); the whole-visit relation is flat, so test per event, not per episode. Observers can read fear, not enjoyment. Disgust lowers enjoyment.
5. **Feature-mismatch aversion:** the uncanny valley (meta g = 1.01, or trimmed g = 0.68 after correcting a strong publication bias; human side 1.34, non-human 0.64) is driven by MISMATCH among features, not by category ambiguity.

Also:
- a sleep-paralysis "visitor" law: lifetime 7.6 percent in the general population and 28 percent in students; onset rises with sleep debt, stress and trauma; the content follows local belief;
- delayed self-feedback as a sensed presence.

Catalogues for coverage checks: Thompson's Motif-Index (23 chapters; C Tabu, D Magic, E the Dead, F Marvels, G Ogres) and ATU tales of magic 300 to 749 are free tables (CC BY-SA); Christiansen's migratory legends. An LLM motif tagger (F1 0.85) can measure coverage and drift in logs, offline only.
- Rumour cascades (Friggeri et al. 2014, full): 62 percent of rumour cascades were false; a correcting link makes a reshare 4.4 times likelier to be deleted, but 45 to 59 percent of reshares come AFTER the correction. Corrections arrive late.

