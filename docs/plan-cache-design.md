# Plan-cache design

Design note, 2026-11-20. An off-menu intent becomes a plan of existing actions; the sim checks it; it is cached with re-adjudication. Open questions with defaults are in section 15.

```text
PLAN-CACHE DESIGN NOTE
Mark_1, written 2026-11-20. Design only: nothing here is built, and nothing in src/, tests/ or tools/ was touched.
Agreed in principle by the user on 2026-10-30. The magic decision Q15 (2026-11-19) is folded in (section 13).

Marks: (u) = unverified, from memory or a guess, to be checked before it is relied on.
"Proposed default" = a number I chose to start with. It is a dial to tune, not a measured fact.


0. THE ONE-PARAGRAPH VERSION
-------------------------------------------------------------------------------
A player says something the action menu does not cover ("smoke out the bees"). A model
proposes a PLAN built only from actions the world already has. The sim runs it. The sim,
not the model, says whether it worked: a success test written in a small closed language
is checked against what really happened, and the energy/mass/momentum ledgers must balance.
A plan that worked is remembered, stripped of this scene's names and keyed by the CAUSES
that made it work. Next time a similar situation comes up, the plan is reused with no model
call. Every reuse is re-checked by the sim, and the entry can be doubted, narrowed, replaced
or retired at any time (re-adjudication). If a plan cannot even be built from existing actions,
a general law or verb is missing, and that is logged as a ticket. The cache remembers WHICH
ACTIONS served an INTENT. It never remembers outcomes. Outcomes always come fresh from the laws.


1. WHAT PROBLEM THIS SOLVES, AND WHAT IT MUST NOT BECOME
-------------------------------------------------------------------------------
Problem. The menu is derived from properties (docs/data-engine-design.md D2), so it already
covers a lot. But a player can still say things whose answer is a SEQUENCE ("light damp straw,
carry it under the hive") or a COMPOSITION that no single menu line names. Today those either
fail or go to the propose channel (D4), which writes nothing and logs an "affordance-gap ticket".
The plan cache is the layer above that: it gives the sequence case a working path, and it keeps
the ticket log for the case where the world truly lacks something.

Fit with the existing design (read: docs/data-engine-design.md sections 1, 2 (D1-D13), 3.2, 3.7,
5):
- D1 sim is authoritative: the plan only SELECTS among legal actions. It never writes an effect.
- D4 propose channel writes nothing; unmapped proposals become tickets. Same here.
- D6 no numeric literals from the author: plans and tests use ordinals or point at world
  quantities.
- D7 tag registry: a test may only talk about quantities the sim actually reads.
- D3 bounded horizon (2-3 beats, revised on new perception): plans are short and are re-checked
  at each step.
- Section 5 "menu-select is the depth-1 case of grammar-constrained derivation": a plan is a
  small complete derivation, one step deeper than a menu pick. The cache stores derivations.
- Decision 2 of the open list ("composed tree from engine-listed pieces", src/core/pieces.py):
  the plan language IS that piece grammar. Every step is checked per node, and a bad piece is
  refused BY NAME. We add nothing new to the engine language.

What it must NOT become (the user's own rules, so these are tests, not hopes):
- Not a table of cases. docs/rulings.md Ruling 1 (model causes, never cases) and Ruling 2 (a
  property table is a hiding place for cases). A cache entry is a case wearing a library's
  clothes unless it is keyed by causes and unless the laws can retire it. Sections 5 and 9 are
  built around this.
- Not a way around the laws. A plan never creates an effect. A plan the laws cannot support
  simply fails its test.
- Not a latency cost. docs/MEMORY latency budget: reactive mode costs 0 NPC model calls; only
  the player parse and the addressed NPC's reply. A cache HIT costs 0 model calls. A MISS costs
  at most one proposal call (plus at most 2 short repairs), and only for the player, only when
  the menu did not cover the text.
- Not a special-case subsystem. Flags enable capabilities, never replace cognition. The cache
  stores nothing that overrides a character's choices (section 8).


2. VOCABULARY (plain)
-------------------------------------------------------------------------------
Intent      what the player wants, in a CLOSED vocabulary the sim can test (section 4).
Plan        a short tree of existing actions (verb + roles), with typed holes instead of names.
Hole        a role described by properties ("a smouldering fuel in hand"), not by id.
Situation key  the features of the scene that mattered. Used to decide "is this similar?".
Aim         the success test: a small declarative formula over world quantities.
Entry       one cache record (section 3).
Witness     a stored, replayable scene in which the entry was tried (section 9.2).
Touched laws  the laws (rules, channels) that actually fired when the plan ran. Used for the
            law version and for key derivation.
Ticket      a log record "an intent needed a piece the world lacks".
Re-adjudication  re-judging an entry against the current laws and new evidence (section 9).


3. THE ENTRY: DATA STRUCTURE
-------------------------------------------------------------------------------
One entry is plain JSON (the repo's rule: frozen types hold no dict; the storage form is plain
data, like api.Job and the record.py rows). Field by field:

  id              short stable hash of (intent canon + plan shape). Not the situation.
  intent          {canon, samples, language}
                    canon:   the goal in the closed vocabulary (section 4). This is the lookup key.
                    samples: up to 8 player phrasings that mapped here (for audit and training).
  key             the situation key (section 5): holes, actor, env, social, touched.
  plan            the plan tree (section 6).
  aim             the success test (section 7). Includes the universal clauses by reference.
  laws            {stamp: {law_id: version}, epoch}  the law version (section 9.1).
  state           provisional | established | doubted | retired | negative   (section 9.3)
  tally           {successes, failures, aborted, preconds_failed, situations: [hashes],
                   proposers: [ids], last_used, last_audited}
  cost            running medians from the ledgers: {ticks, energy_J, injury_band, items_used}
  alternatives    ids of other entries with the same intent (a Pareto set, section 8.3)
  witnesses       list of scene recipes (section 9.2)
  provenance      {proposer: {kind: model|player|pack|taught, model_id, prompt_hash},
                   first_seen: {tick, scenario, seed}, evidence: real|rehearsal,
                   parent: id of an entry this one was revised from, or null}
  history         NOT stored inside the entry. It is the append-only event log (section 10);
                  the entry state is derived from it. This keeps "why is it established?"
                  answerable for ever, and it is the same idea as the project's rule that
                  model output is always logged as typed deltas.

Worked example (bees). Player text: "smoke out the bees". Canon goal: lower(behavior.aggression,
of: ?swarm) AND hold(absent(?swarm, region: ?hive_cavity)) AND never(hp_loss(actor) above light).
After a successful run the entry reads, in words:

  intent   lower aggression of ?swarm and clear ?hive
  key      holes: ?swarm = animal(group, flying, nesting_in ?hive)
                  ?hive  = structure(holds ?swarm, reachable)
                  ?fuel  = thing(flammable, moisture band damp..wet, mass band light)   [in hand or near]
                  ?flame = thing(emits fire OR can ignite ?fuel)                          [in hand or near]
           actor: free_hand >= 1, not on_fire, hp band >= moderate
           env:   humidity band >= dry..humid (any), wind band <= breeze (smoke must stay close)
           touched: combustion(smouldering), smoke(gas production), perception(irritant), behavior(flee)
  plan     (then (ignite ?fuel with ?flame)  (take ?fuel)  (move reach-of ?hive)
                 (hold ?fuel at ?hive_cavity_below))
  aim      see section 7 for the exact text.

Note what is NOT in it: ids, names, coordinates, tick numbers, the number of bees, the story.


4. THE INTENT: FROM FREE TEXT TO A CLOSED GOAL
-------------------------------------------------------------------------------
The player's text goes to the SAME small model call the reactive player parse already uses. If
the parse finds a menu option or a pieces-grammar line, the cache is never involved. Only when
it does not, the same call (no extra call) is asked for a second output: a CANONICAL INTENT.

Canonical intent grammar (closed, no free words):
  goal   := effect ( AND effect )* [ AND guard ]*
  effect := verb-of-change(quantity-ref [, role])
  verb-of-change := raise | lower | stop | start | move | open | close | hold | transfer |
                    reveal | hide | join | separate
  quantity-ref := (channel, role, optional region)   channel in the coupling graph channels:
                    heat, fire, fluid, hp, life, structure, perception, disposition, behavior
                    (src/core/data/couplings.yaml) plus later channels as the world grows
  role := a typed variable ?x with a property predicate (tags, material tags, ordinal bands)

Why closed: if the model may write anything, nothing can be tested. If the intent cannot be
written in this vocabulary, that is information: the world has no quantity for it. That becomes
a ticket of kind "missing quantity" (section 11), not a guess.

Retrieval needs no model: exact match on canon, then match on the "shape" of canon (same effect
verbs and channels, roles generalised). An embedding of the canon text is used only to
suggest extra CANDIDATES when exact match finds none (a bi-encoder is already the planned
model, D9). A candidate must still pass the precondition check (section 5.4). So a wrong
embedding match wastes a check; it cannot cause a wrong plan.


5. THE SITUATION KEY: WHAT FEATURES MATTER
-------------------------------------------------------------------------------
5.1 The idea: key by CAUSES, not by looks.
  Explanation-based generalisation (Mitchell, Keller, Kedar-Cabelli 1986, "Explanation-Based
  Generalization: A Unifying View", Machine Learning 1(1); (u) exact page): take ONE successful
  run, take the chain of rules that fired (the explanation), replace the specific objects by
  variables, and keep as preconditions only the conditions those rules actually read. That is
  what a key should be. The features that mattered are exactly the ones the laws read on the
  way to the goal.
  The sim already declares its coupling graph with rule ids (couplings.yaml, edges via: [rule ids]).
  So the run's rule trace tells us which laws were touched and which conditions they read.
  (u) I have not checked that every run emits the rule ids with their read conditions. If not, the
  first build step is to add a "read set" to the existing typed events; that is a sim-side change
  for the wave that owns the sim, not for this module.

5.2 Feature families (all read through the existing perception/belief layer, so the key never
    uses god-channel facts; the character-relevant view is the same belief slice the menu uses):
  A. HOLES: for each role variable, its required properties:
     - tags and material-template tags (flammable, wooden, container, person, animal, ...);
     - ordinal bands of quantities (moisture dry/damp/wet; temperature cold/warm/hot/above-ignition;
       mass light/moderate/heavy; amount little/some/much) - band edges come from the law that
       reads them (e.g. the band edge for "damp" is the sorptivity/ignition threshold), never typed
       in by the cache;
     - reach band (touch/arm/throw/earshot/sight).
  B. ACTOR: capabilities the plan's steps need (free hand count, can_see, can_lift band), body
     bands (strength, fatigue, injury, temperature), items held by property.
  C. ENVIRONMENT: only channels that the TOUCHED laws read. Humidity band, wind/air-motion band,
     light band, ambient temperature band, ground/support. If no touched law reads wind, wind is
     not in the key. This is what stops the key from growing without limit.
  D. SOCIAL: only if disposition/behavior channels are touched: who is near, relationship band,
     whether observed. Never names.
  E. LAWS: the version stamp of the touched laws (section 9.1).
  NEVER in a key: entity ids, names, coordinates, tick or time of day (unless a touched law reads
  a clock), story text, the player's identity.

5.3 Two key levels.
  - FINE key: the full set of the features above for one run. Stored in the witness and in the
    history, to say "tried here".
  - COARSE key (the entry key): the precondition. Starts as the leaves of the explanation, with
    each band widened to the band edges the touched law reads. It is then changed by evidence:
      * widened when successes in varied situations show a feature never mattered (drop it), like
        shrinking a hypothesis from the specific side (Mitchell 1982 version spaces, "Generalization
        as search", Artificial Intelligence 18; (u) page);
      * narrowed when a failure shows a feature mattered (add it as a precondition), section 9.4.

5.4 Matching (bind) at reuse time. Given the player's intent and the current scene:
  1. Exact/shape retrieval by canon gives candidate entries.
  2. For each, try to BIND the holes: ask the backend's bind() (the same legality predicate the
     menu uses, D8) for things satisfying each hole predicate. No binding = not applicable.
  3. Check actor and env predicates. A failed predicate is recorded as a "precondition failed",
     not as a plan failure (it does not hurt the entry).
  4. Order what is left: established before provisional, then lower cost, then fewer
     failures, then id (no randomness; the sim has no RNG, choosers use seed_of).
  5. If a hole has several matching things, the character's normal chooser picks among them (the
     plan FILTERS the menu, it does not pick for the character, section 8).

5.5 Cost of a coarse key that is wrong.
  Too narrow: low hit rate and many model calls. Measured by "calls per 100 intents" (section 12).
  Too wide: reuse in the wrong situation and a failure. Measured by the failure-after-hit rate and
  by audits. Both are visible numbers, so the widening/narrowing rules have an objective to tune.


6. THE PLAN: STEPS, HOLES, CONTROL
-------------------------------------------------------------------------------
6.1 Step. verb + role slots, exactly the Option shape the mind already uses (api.Option: catalog
    verb id and roles; or the pieces form verb + args). A slot is one of:
      - a hole variable ?name (bound at run time, section 6.2);
      - a literal only for things that are not entities: direction names, ordinals
        (light/moderate/heavy, little/some/much, all), spoken words (speak stays the one
        free-text field).
    No numbers (D6). Amounts point at world quantities ("all", "carried amount") or are ordinals.

6.2 Late binding. The existing chain (Job.then, Mind._chain) requires the next option to be legal
    in the current menu, otherwise the chain is dropped and the actor is asked again. A plan has
    the same need, but its later steps often name things that do not exist yet (the smouldering
    bundle appears only after step 1). So a step is a CONSTRAINT: verb + role predicates. When
    its turn comes, the engine filters the actor's current menu (backend.bind) to options that
    satisfy it:
      - none match: the step is not feasible now -> plan run stops with reason "step N not
        feasible: <missing>". The player gets a diegetic line ("you have nothing to carry it in").
      - one matches: it is taken (nobody is asked).
      - several match: the chooser picks among only those (character dials still count).
    A step may bind a variable to "the thing made by step k" (a result reference). That is the
    only forward reference allowed.

6.3 Control. Use the pieces grammar and no more: then (sequence), if with conditions from the
    listed pieces (near, has, carrying; extend only with evidence). Guard semantics as today: a false
    condition skips its step quietly. A bounded wait step ("wait little/some") is added when a plan
    needs time to pass (smoke spreading). No loops in phase 1-4. If evidence shows a loop is
    needed ("keep fanning until clear"), add `repeat <ordinal> until <aim clause>` with a hard cap.
    Stopping rule from Ruling 1: add a construct only when two cases the user wants different would
    otherwise be the same.

6.4 Size. At most 6 steps (proposed default; it matches the bounded-horizon finding that 2-3
    beats works and unbounded planning overruns sight). Longer needs are broken into sub-plans
    that call other cache entries by intent (a step can be "achieve <canon>", which itself uses
    the cache). Cycles are refused (an entry may not call itself through any path).

6.5 Legal by construction. Every leaf must resolve to a real Option from the current menu
    (pieces.index). A piece that does not resolve is refused BY NAME (PieceError). This is the same
    guarantee as menu-select: a plan cannot do anything the actor could not do one step at a time.


7. THE SUCCESS TEST ("AIM"): LANGUAGE
-------------------------------------------------------------------------------
7.1 Goals of the language: closed, declarative, readable by the player in plain words, checkable by
    reading registered world quantities, with no numeric literals and no code.

7.2 Clauses.
  reach  Q  CMP  TARGET        the state when the plan run ends and the settle window has passed
  change Q  DIR  AMOUNT        the difference from the baseline read when the plan started
  hold   PRED  for DURATION    the predicate stays true for an ordinal duration (so a momentary
                               success does not count)
  never  PRED                  a guard during the whole run (a side-effect limit)
  within BUDGET                limits on time, energy, injury, items consumed (ordinal bands)
  Q       := (channel, role, optional region)   a REGISTERED quantity
  CMP     := above | below | gone | present | in-state(S)
  TARGET  := an ordinal, "baseline", or a POINTER to a world quantity, e.g. "below the flee
             threshold of ?swarm" or "below the ignition threshold of ?fuel". Pointers are the D6
             form 1 (point at a world quantity), so the test moves when the law's constant moves.
  DIR     := up | down | to-zero | to-nonzero
  AMOUNT  := an ordinal (little/some/much) or "any"

7.3 Registered quantities. A quantity can be tested only if the sim reads it (same spirit as the D7
    tag registry: a name nothing reads is dead weight). The test compiler refuses an unregistered
    quantity BY NAME and raises a ticket of kind "missing quantity". This keeps the test language
    from becoming a back door for invented statuses ("blinded", "charmed").

7.4 Universal clauses. The cache adds these to every aim itself. The model cannot leave them out and
    cannot weaken them:
  - ledgers: energy, mass and momentum books balance within the sim's existing tolerance over the run
    (the ledgers from waves 13a/13b: the energy ledgers, body heat, digestion, combustion);
  - every step was legal when taken (checked by the existing start/legality test);
  - the actor is alive and not in an injury band above the intent's allowance (default: no worse
    than "moderate" unless the intent says it wants harm);
  - bounded time: the plan ended within its step and time budget;
  - no unintended death or large collateral loss of anything the actor values (reads the existing
    drives and relationships; the exact predicate is part of phase 3).

7.5 Test authoring and hardening. The model drafts the intent goal (section 4) and the aim comes
    from it mechanically plus the universal clauses. A model-written extra clause is allowed but
    is hardened before an entry is admitted:
  1. NULL CONTROL: run the do-nothing plan (the catalog null act) on a snapshot. The aim MUST
     FAIL. If the aim passes with no action, it is vacuous and the entry is refused.
  2. ABLATION: for each step, run the plan with that step removed. If the aim still passes, the
     step is not causal and is removed from the plan (delta-debugging style minimisation, Zeller
     and Hildebrandt 2002, "Simplifying and Isolating Failure-Inducing Input", IEEE TSE (u)).
     If removing a step changes nothing about cost either, it is extraneous. A plan whose every
     step is needed is what gets stored.
  3. PLAYER-READABLE: the aim is rendered in plain words ("I'll count this as a success if the
     bees leave the hive and you are not badly stung"). Showing it to the player before the first
     run is optional (an open question, section 15).

7.6 Example aim for the bees:
     reach  behavior.aggression(?swarm)   below   flee-threshold(?swarm)
     hold   absent(?swarm, region: ?hive_cavity)   for  some
     never  hp_loss(actor)   above   light
     within time: some ; energy: little
     + universal clauses.


8. HOW IT PLUGS INTO THE EXISTING CHOOSE PATH
-------------------------------------------------------------------------------
8.1 Today (src/mind/loop.py): Mind.decide(backend, who, reason) -> backend.bind(who) -> rule_scores
    -> core.menu -> DecisionRequest -> chooser (rules / model / player) -> Decision -> backend.start
    -> Job; Job.then holds a player's chain; Mind._chain re-checks that the next option is legal.
    The reactive mode keeps NPCs on rules with 0 model calls.

8.2 Two plug-in points. No existing function changes meaning.

  POINT A. The player's off-menu text (the main case; new file, proposed name src/mind/plans.py).
    Flow:
      1. player text; the existing parse fails to land on a menu option;
      2. the same call also yields the canonical intent (section 4); no extra call;
      3. retrieval + bind (section 5.4). A hit gives a bound plan with no model call;
      4. a miss goes to the PROPOSER (one call; section 8.4);
      5. a PlanRun object (kept in Mind next to `jobs`, plain data so snapshots copy it) executes
         the steps as normal jobs, one at a time, through Mind.decide with a menu FILTER (the
         plan's current step constraint) and the normal chooser. backend.start re-checks legality
         against the live world, as it does now;
      6. at the end the aim is evaluated; the entry is created or updated; an event is logged;
      7. the player's turn comes back normally.
    The PlanRun reuses Job.then's idea (the next step runs when the last is "done") but with
    late binding. If a step fails or is interrupted, the Mind asks the actor again with the
    reason, as it does today ("failed:..."), and the PlanRun is closed as ABORTED or FAILED.

  POINT B. A cached plan as a macro option in menus (later, phase 5).
    A plan whose preconditions hold in the current scene is offered as ONE option in the menu
    (group "plan", verb "plan", text like "smoke out the hive"), exactly the way parts.AFFORDS hang
    options on objects. It is a derived option, not an authored verb (D2). NPCs and players then
    choose it with the same choosers, the same rule scores (the plan option gets features from its
    cost and aim: expected effort, risk band), and the same pre-rank cap (core.menu cap, ~7). A
    plan option is never forced. This is how an NPC could "know" a plan: only entries the NPC has
    in its personal repertoire (section 13.3) are offered.

8.3 Alternatives and character. Several entries can serve one intent (a quick but risky one, a slow
    safe one, a stealthy one). They are kept as a Pareto set by their cost vector (ticks, energy,
    injury, noise, items). Each becomes its own option, and the character's dials (caution,
    patience, etc.) choose among them through the existing rule scores. This respects "traits are
    dials plus history plus state" and "flags enable capabilities, never replace cognition". The
    cache never picks for a character.

8.4 The proposer. A new small protocol beside api.Chooser, proposed name Proposer:
      propose(intent_canon, piece_table, scene_text, failure_context=None) -> text in pieces grammar
    It sees only what the actor perceives (the perception-filtered table, D8 and the god-channel
    guard), plus the intent. It gets the error text on a bad piece and may repair at most twice.
    It is counted and metered like the model chooser (the loop's _Metered counts remote calls per
    tick), so a test can assert the budget. The proposer can be a local model, a larger model used
    offline in the lab, or a scripted fake for tests. The big model's job stays "taste", not
    "legality" (D10): legality is guaranteed by the grammar and the sim.

8.5 What is logged for each use (record.py row, one extra field "plan"): entry id, entry state,
    law stamp, binding (ids, for the replay), steps taken, the aim result per clause, the ledger
    result, the cost read, and whether the model was called. Replays need the cache state; the
    cache version hash goes in the run header.


9. RE-ADJUDICATION
-------------------------------------------------------------------------------
9.1 Law version.
  Every law module and every coupling-graph rule has a version string (a hash of its data and
  code constants, proposed). An entry stores the versions of ITS TOUCHED LAWS only (from the run
  trace), not one global number. When a law changes, only entries whose touched set includes it are
  marked STALE. A change that cannot be attributed (an engine-wide change) bumps an `epoch`, which
  stales everything. The idea is dependency-directed: re-justify only what depended on the changed
  thing (Doyle 1979, "A Truth Maintenance System", Artificial Intelligence 12; (u) page).

9.2 Witnesses. A witness is not a saved world (too big). It is a RECIPE: scenario builder name +
  scenario version + parameters + world seed + the actor id + the tick the attempt began + the
  actions of the run + the outcome hash. The sim is deterministic and has no random number
  generator, so the recipe replays the same scene exactly. Each entry keeps up to 5 witnesses
  (proposed default): the first real success, the most different successes by key distance, and
  the most informative failures. When a scenario's code version changes and a recipe no longer
  replays, the witness is marked UNREPLAYABLE and rebuilt from the scenario factory; if it cannot
  be rebuilt it is dropped, and an entry left with 0 witnesses cannot be established.

9.3 Entry states.
  provisional   proposed and passed at least one run; used with a visible hedge (the plan runs, the
                result is judged as always); needs evidence.
  established   enough varied successes (9.5). Preferred at retrieval.
  doubted       a trigger fired; kept, but ranked lower and re-checked before use.
  retired       replaced, or failed re-adjudication with no narrower key that still works. Kept in
                the log, never served.
  negative      "we asked K times and no plan of existing actions achieves this intent here". Cached
                so the model is not asked again for the same hopeless intent and key. Negative entries
                are re-adjudicated by the same triggers (a law change may make it possible).

9.4 The five triggers and what each does.

  T1  A LAW CHANGE.  Entries with a stale touched law go to the re-run queue. For each entry: replay
      its witnesses under the new laws (no model call; the aim and universal clauses are the
      judge). All witnesses pass: stamp the new version, log "survived change X". Some fail: go to T2
      handling. All fail: retire, then (queue) ask the proposer once with the failure as context.
      Ordering: cheapest first, most-used first. Runs in the background or at law-change time in the
      lab, never inside a player's turn.

  T2  A FAILURE IN A NEW SITUATION.  A reuse ran and the aim failed (not aborted, not a failed
      precondition). Steps:
      a. Classify the failure: step infeasible / ran but test failed / ledger violated / aborted by
         an outside event. Aborted runs are NOT counted against the plan (an interrupted run says
         nothing about the plan); they are logged and counted separately.
      b. Find the cause by asking the SIM, not the model: take the failing scene and a success
         witness, list the key features that differ, and run one-feature-at-a-time swaps on
         snapshots (a counterfactual probe; the sim is deterministic, so a single run per feature
         is an exact answer). The features whose swap flips the result are the causal ones.
         Cost: one rehearsal per differing feature (proposed cap 8 probes; more differences go to
         step c directly).
      c. NARROW the key: add the causal feature as a precondition (excluding the failure's band).
         If no single feature explains it, or no narrower key keeps any success, ask the proposer
         for an alternative plan in the failing situation with the failure as context (one call,
         the failure's aim result and the probe findings in the prompt). A new plan becomes a
         separate provisional entry with parent = the old one. The old entry stays valid in the
         narrower key.
      d. Log everything. A failure-after-hit counts toward the entry's `doubted` state: 2 failures
         in the last 5 uses (proposed) -> doubted.

  T3  A DISPUTE.  By the player (a plain command, "that shouldn't work" or "that's not what I
      meant") or by another player in a shared session. A dispute is sorted by what it is about,
      because three different things can be wrong:
        - WRONG PLAN: the plan did not do what was meant -> treat as T2 on the recorded run.
        - WRONG TEST: the aim says success but the player says it was not -> the aim is the
          suspect. The entry is doubted, the aim is shown in words, and the proposer is asked to
          restate it. Pending the answer the entry is not served.
        - WRONG LAW: "that outcome is unrealistic" -> this is not a cache matter. It goes to the
          rulings queue (docs/rulings.md) as a case for the user, with the run attached. The cache
          entry is doubted until the ruling lands. The sim stays authoritative, so the cache never
          "fixes" physics.
      Disputes from the user outrank the automatic checks. The entry is never deleted by a dispute,
      only doubted or retired through the same flow, so a mistaken dispute can be undone from the
      log.

  T4  RANDOM AUDITS.  Each established entry is audited about once per A uses or D days of
      play-time since last audit (proposed: every 50 uses or each law-stamp epoch, whichever first).
      The audit replays a witness and runs 2 PERTURBED scenes: the witness with non-key features
      varied inside their bands (property-test style), plus one at the key's band edge. A failure
      is handled as T2. Sampling is seeded (seed_of(world seed, entry id, audit counter)) so a run
      replays. Audit load is capped per session (proposed default: at most 20 rehearsals per
      session), and run only when the lock is free in the lab / idle in play.

  T5  A BETTER PLAN FOUND LATER.  Every success records its cost vector (ticks, energy, injury,
      noise, items) from the ledgers. A challenger (a new proposal for the same intent, or an entry
      from a different intent that happens to also satisfy this aim) is run on the incumbent's
      witnesses. If it passes them all with a better vector on at least one axis and no worse than
      a tolerance on the others, it is added to the Pareto set (8.3). If it dominates, the
      incumbent becomes retired with "superseded by" and is kept as an alternative while it still
      has users. Challengers are generated only when idle and only for entries used often.

9.5 From provisional to established.
  An entry becomes established when ALL hold (proposed defaults):
   - at least 3 REAL successes (rehearsal successes do not count toward this number);
   - in at least 3 distinct situations: distinct = their fine keys differ on at least 2 features
     that are not in the first run's key, or at least 2 different scenario families;
   - at least 1 of those successes by a different actor from the proposer;
   - at least 1 success at an easy band edge (so the key edges are tested at least once);
   - no unresolved failure, no pending dispute, no stale touched law;
   - at least 2 witnesses replayable.
  It goes back to doubted on the T2/T3 rules and back to provisional if its law stamp goes stale and
  a witness fails.
  Why "varied": three successes in the same room prove the room. Three successes in different
  setups prove the CAUSES. This is also the "select for what persists" rule applied to plans.

9.6 Guard against poisoning by failure and by luck.
  The sim has no random draw, so a run is determined by the scene. A "lucky" success can still be
  an accident of this scene (an NPC wandered in and helped). The null control and ablation (7.5)
  catch much of it. The varied-situations rule catches the rest.


10. STORAGE
-------------------------------------------------------------------------------
10.1 Two stores.
  - SEED entries: hand-vetted entries that ship with a pack, as data (YAML under src/mind/data/,
    like verbs.yaml; proposed src/mind/data/plans/<pack>.yaml). They are what a content author
    writes by hand, and they are tests of the format. They enter as state `established` only if
    they pass their own witnesses.
  - PLAY entries: created during play. They are play data, not source. They live in a per-save
    directory (proposed: <save>/plans/), and are git-ignored (the repo is public; no caches or
    play data; docs rule from the user's notes).

10.2 Format. Append-only JSON-lines event log + an in-memory index built at load.
  Events: entry_created, used(binding, result), preconds_failed, aborted, state_changed(from,to,why),
  key_narrowed(feature), key_widened(feature), aim_restated, witness_added/dropped/unreplayable,
  law_stamped, audit(result), disputed(kind), challenger(result), retired(by), ticket_linked.
  The entry's current state is the fold of its events (the project's way: logs are the truth, state
  is derived; same shape as src/mind/record.py rows). Compaction writes a snapshot file every N
  events (N = 10,000, proposed). A single writer holds a file lock; the lab's parallel runs write
  their own logs and a merge step combines them (no shared writes).
  I expect hundreds to a few thousand entries per world (u). That is small enough for plain
  JSON-lines and a dict. If it passes ~50,000 entries (u), move the index to sqlite3 (standard
  library) and keep the event log as it is.

10.3 Scope. The entry says what WORKS in this world's laws, so it is shareable between characters
  and between saves with the same law stamps. WHO KNOWS it is a separate record: the personal
  repertoire (section 13.3). That split is what lets Q15 and NPC knowledge work.

10.4 Retention. Retired entries and their events stay in the log; their witnesses are dropped after
  30 days of play-time or at compaction (proposed). Unused provisional entries (no use in the last
  200 player intents) are retired and kept only as log lines.

10.5 Determinism and replay. Retrieval, ordering, audit sampling and narrowing choices use no hidden
  randomness. The run header records the cache snapshot hash. A replayed run reads the same entries
  and gets the same plan bindings. (Replay needs seed_of, as the choosers do.)


11. THE MISSING-LAW LOG (TICKETS)
-------------------------------------------------------------------------------
11.1 When a ticket is raised. All of these, and each has a different diegetic line for the player:
  a. MISSING PIECE: the proposed plan uses a verb/slot value that is not on the piece table, or no
     composition of listed pieces can do the thing (PieceError names the piece). The world lacks an
     action. Ticket kind: missing_action.
  b. MISSING QUANTITY: the intent needs a quantity the sim has no register for (7.3). Kind:
     missing_quantity.
  c. MISSING EDGE: all pieces exist and the plan is legal and runs, but the aim fails because no
     coupling edge carries the effect (e.g. smoke does not occlude sight; couplings.yaml already
     has this as a declared missing edge). Kind: missing_edge. This is the signal that the cause is
     missing from the laws.
  d. IMPOSSIBLE BY THE LAWS: the laws are complete enough, and the intent contradicts them. Not a
     ticket. It is a NEGATIVE entry (9.3) after K=3 diverse failed proposals (proposed).
  e. INFEASIBLE NOW: the plan exists but this scene lacks something (no flame source). Not a
     ticket, not a failure of the entry. Diegetic: "you have nothing to light it with." It is a
     precondition failure.

11.2 Ticket record: {id, kind, intent canon, the missing piece/quantity/edge (normalised),
     situation fine key, the nearest listed pieces, the failing aim clause, count, first_seen,
     last_seen, sample player phrases, proposer output}. De-duplicated on (kind, normalised missing
     thing). Ranked by count and by how many distinct intents were blocked.

11.3 Where the log goes. It feeds the existing affordance-gap backlog: the `missing:` list of
     src/core/data/couplings.yaml, which today is hand-kept. A ticket that reaches a count threshold
     (proposed: 5 distinct players-or-runs) is shown in the next wave plan as a candidate law. The
     metric "unmapped-proposal rate over time, should decline" (data-engine-design.md section 7) is
     measured from these tickets.

11.4 When the world gains the missing piece, linked tickets and negative entries are retried
     automatically (a law change, T1). Linked intents are re-asked once, in the background.

11.5 Promotion. An entry used very often and always with the same key is a sign that the world
     wants a derived menu option or a composite for it. It is flagged "promote?" in the same list.
     The cache is the way improvisation matures into the menu, not a permanent second rulebook.


12. TESTS
-------------------------------------------------------------------------------
12.1 Unit.
  - Aim parser/evaluator: each clause type; refuses unregistered quantities, numeric literals,
    unknown pointers, each by name.
  - Generalisation: run a plan on scene A, generalise, then bind on scene A again: the same plan and
    the same result. Bind on scene A with all ids renamed: same.
  - Band edges come from the law: change a law constant in a test copy, and the edge moves.
  - Key never contains ids, names, coordinates or ticks (a scan over every entry).
  - Retrieval order is deterministic; identical logs give identical picks.

12.2 Budget (the latency rule, as tests).
  - Reactive mode over N ticks with a counting fake model: 0 NPC calls, as now.
  - A cache hit: 0 model calls. A miss: exactly 1 proposal call (+<=2 repairs), only on the
    player's turn.
  - A plan run adds no model call per step.

12.3 Honesty of the test (the poisoning tests).
  - Null control: an intent whose aim passes with no action is refused (vacuous).
  - Ablation: a plan with a padded extra step is stored without it.
  - Aborted runs (an interrupt) do not change the tally of failures.
  - Ledger violation: a mutated sim that leaks energy makes the plan fail admission; the universal
    clause cannot be removed by the proposer's output.
  - A scripted adversarial proposer (tries a free-energy loop, a vacuous test, a plan that wins by
    killing the observer) gets refused or retired.

12.4 Re-adjudication.
  - Law change: only entries whose touched set contains the law are re-run (a spy counts replays).
  - T2: a scene differing in one feature (humidity band changed) makes a plan fail; the counterfactual
    probe names that feature; the entry narrows to exclude it; the other scene still hits.
  - T3: each dispute kind goes to the right place; a "wrong law" dispute creates a rulings case and
    never edits the sim.
  - T4: audit sampling is reproducible from the seed and capped.
  - T5: a cheaper plan joins the Pareto set; a dominated one is superseded.
  - Provisional to established obeys 9.5 exactly (property tests over synthetic histories).

12.5 Missing-law tickets.
  - An intent needing an absent verb raises one ticket with a normalised signature; asking again
    increments, not duplicates.
  - Add the verb in a test catalogue: the ticket's retry runs and resolves it.
  - Infeasible-now is NOT a ticket. Impossible-by-laws becomes a negative entry after K tries.

12.6 No-special-case scans (like the Q11 pack scan and the tag registry CI invariant).
  - Every step of every entry resolves to a verb in the catalogue. No entry names an element, a
    material or a creature in a plan step (they appear only inside predicates as properties).
  - An entry can only name registered tags/quantities/channels (D7 style).
  - Renaming/metamorphic: relabel entities and rephrase the player text; same entry, same outcome.

12.7 Integration (the playtest harness, python -m src.core session; runs after each slice).
  - Scripted off-menu intents on existing scenarios (the bees, "cool the room", "dam the stream",
    "signal the guard from a distance") run end to end. First run records a miss and a provisional
    entry; later runs hit.
  - Measures, reported each wave: hit rate; model calls per 100 player intents; miss->success rate;
    failure-after-hit rate; false-establish rate (established entries that later fail an audit);
    re-adjudication churn per law change; tickets per 100 intents (should fall); time per
    rehearsal (u, to be measured).

12.8 Determinism. Same seed + same cache snapshot + same player script => identical rows.


13. MAGIC DECISION Q15: PROVEN COMPOSITIONS BECOME TEXT-INVOCABLE
-------------------------------------------------------------------------------
13.1 The decision (user, 2026-11-19): menus at first. Later, intricate interactive casting
  interfaces (draw glyphs, say spells, compose music, write formulae, gestures). FREE TEXT only
  for what the player has already shown proficiency in through those interfaces or manual methods
  (a proven composition becomes invocable by text, like the plan cache), or for what the character
  has implicit proficiency in (innate techniques set by the pack). Free text must NOT let anyone do
  anything ("I cast fireball, but where no one can see it"). The player's own performance in the
  interface supplies the pattern quality; the character's skill sets the noise floor and limits.

13.2 Same cache, different plan type. An entry's plan can be a COMPOSITION in the universal magic
  grammar (decision 8.4: regions and shapes, forms of change, strength over time, source, triggers,
  combinators), i.e. a complete grammar-constrained derivation. This is the "existing actions" of
  magic: the grammar nodes. Everything else in the entry is the same:
    intent  the desired physical effect in the closed intent vocabulary (a quantity change in a
            region: "raise heat in region R by much");
    key     caster state (M available band, body heat headroom, fatigue), medium used, local field
            band, target properties (touched laws); no names of spells;
    plan    the composition AST with typed holes (target region, amount ordinal);
    aim     the Aim language over physical quantities (energy delivered to region, area, duration)
            plus the faithfulness clauses the magic design already requires (ledger booked: work +
            light + waste = 1; hand-off equivalence is a test of the law, not of the entry);
    laws    the magic law pack version, so an engine change to M retires stale compositions;
    provenance  who performed it, in which interface, with what performance quality.
  The names "fireball" etc. are cultural labels in a naming table (decision Q10). They are text
  synonyms for the canonical intent, not engine names.

13.3 Personal repertoire (the gate). Separate from the world cache (10.3), each character has a
  repertoire: {entry id -> proficiency record}. A record holds: first performed (tick), number of
  performances, best and typical pattern quality (from the player's interface performance, bounded by
  the character's skill), last performed, interface used, decay state.
  Text invocation of an entry is allowed only if ALL hold:
   1. the character has a repertoire record for it with enough proven performances (proposed: at
      least 3 successful performances, the last within a decay window), OR the entry is an
      implicit technique granted by the pack for this character (data, set by the pack, never a
      flag on the engine);
   2. the entry is currently valid (not stale/retired) and its preconditions bind in the scene;
   3. the caster has the resources now (M, body heat headroom, energy), checked by the same
      magic kernel as any cast.
  What a text invocation does: it REPLAYS the stored composition. It does not let the text change
  the composition. So:
   - the casting time, the tell others can perceive, the waste signature and the energy cost are the
     stored ones (the tell cannot be hidden by wording: "in a way no one can see" is not an
     invocable parameter, because it is not a hole of any composition that was proven);
   - the free parameters are ONLY the holes the proving varied over (target, amount ordinal within
     the proven range). Outside the proven range the text is refused, or runs with an error that
     gives a neighbouring valid spell (the existing rule that errors give neighbours);
   - the pattern quality is the character's current ability, with the proven best as a ceiling
     and the character's noise floor and state (fatigue, injury, stress) as the limit. It is not the
     player's skill at the moment of typing. A tired or injured character casts the text spell
     worse.
   - the cast goes through the same physics: the cache gives the PATTERN, the magic law computes the
     effect from it fresh. So a text cast is exactly as faithful as a hand-built cast (the
     hand-off equivalence and metamorphic tests of the faithfulness list apply to cache casts too).
  Decay: proficiency falls without practice, so chunks that are not used fade. This matches the
  design that practice chunks a family into macros and that a tradition's "elements" are its learned
  chunks (Q10): a tradition's established entries ARE its element-like macros; teaching one is
  copying an entry to another character's repertoire with provenance "taught by" and a lower
  starting proficiency (the student must still perform it to prove it).

13.4 Improvised casting and recorded casting (decision Q8). The cache stores the PATTERN. Whether a
  character can cast it with no record needs memorised knowledge (the repertoire above). A scroll,
  engraving or recorded tune is an object that holds the pattern (the "implicit slot"); reading it
  is a normal action with its own time and wear. The two ways to cast stay as decided; the cache is
  the memory behind the first and the content of the second.

13.5 Q15 tests (added to section 12).
  - Unproven: text invocation of a never-performed composition is refused (diegetically).
  - Proven: allowed; the replay has identical casting time, tell, waste signature and cost to the
    stored composition.
  - No hiding: no wording can remove or shrink the tell or the cost. A scan: the invocation grammar
    has no slot that touches the tell.
  - Range: a target outside the proven range is refused or errors to a neighbour.
  - Skill/state: lower the character's skill or raise fatigue in a test and the same text cast gets
    noisier or fails; the player's typing never changes pattern quality.
  - Faithfulness: a cache cast of Q joules gives the same downstream physics as an ordinary Q-joule
    source (the same A/B test as for hand-built casts); renaming the text synonym changes nothing.
  - Decay: proficiency falls with disuse and the text cast gets refused below the threshold.
  - Teaching: a copied entry starts below the threshold and needs a performance.
  Magic phases depend on the first magic slices (the field and its ledger, the kernel); the cache
  side is ready for it as soon as entries can hold a composition AST (phase 6 below).


14. PHASED BUILD PLAN
-------------------------------------------------------------------------------
Each phase ends with a gate. The gates use numbers; all thresholds are proposed defaults.

PHASE 0. Checks (about half a day (u)). No new code in the cache yet.
  - Confirm that a run can export the rule ids with the conditions they read (needed for 5.1 and
    9.1). If not, ask the sim owner to add a "read set" to the typed events.
  - Confirm a cheap snapshot and a deterministic rehearsal API on both backends (rooms: turns;
    voxel: ticks). Measure the cost of one rehearsal (u).
  - List the registered quantities available for tests (from couplings.yaml and the body/ledger
    modules). Missing ones become the first tickets.
  Gate: a short report with those three answers.

PHASE 1. Data, Aim, store, hand entries. No model.
  Entry/Plan/Aim types, the Aim parser/evaluator, the universal clauses, the JSON-lines store with the
  index, seed entries from YAML, retrieval + bind (Point A, hit path). Player text maps to canon by a
  scripted table in tests.
  Gate: 10 hand entries on existing scenarios hit and run; unit and determinism tests pass; hit costs
  0 model calls.

PHASE 2. The run path.
  PlanRun in Mind, late-binding steps, menu filter, abort vs failure, aim evaluation at the end,
  history events, the record.py row. A scripted fake Proposer.
  Gate: 5 off-menu intents run end to end under the playtest harness; budget tests pass; aborted runs
  do not count.

PHASE 3. Proposal and admission.
  The real Proposer (local model), the named-bad-piece repair loop (<=2), rehearsal, null control,
  ablation pruning, generalisation (EBG) into holes, provisional entries, the ticket log (11).
  Gate: on a set of 30 off-menu intents across 3 scenarios, (a) every admitted entry passes null
  control and ablation, (b) an unbuildable intent set produces tickets, not entries, (c) calls per
  intent on the second ask are 0, (d) a count of how many of the 30 produced an entry is reported
  honestly (no target set before measuring).

PHASE 4. Re-adjudication.
  Law stamps and touched sets, witness recipes and replay, T1 to T5, counterfactual probes, narrowing
  and widening, establishment rule 9.5, audits (capped), the dispute command and its sorting.
  Gate: a real law constant is changed in a test copy; only touched entries re-run; one entry
  narrows correctly by probe; false-establish rate on a synthetic adversarial set is 0.

PHASE 5. Menu integration and knowledge.
  Plan macro options in menus (Point B), Pareto alternatives, the personal repertoire, NPC knowledge
  through gossip/belief (source "told"/"saw"), sub-plans ("achieve <canon>") with cycle refusal, and
  the "promote?" report.
  Gate: the menu-size test still holds (padding the world 10x leaves menus equal); NPC calls stay at
  0 in reactive mode; a plan option appears only for characters who know it.

PHASE 6. Magic compositions (Q15). Needs the first magic slices and the kernel.
  Composition AST as a plan type, magic law stamps, repertoire proficiency and decay, text invocation
  rules (13.3), teaching.
  Gate: the Q15 tests of 13.5 pass, and the faithfulness tests still pass with cache casts.

PHASE 7 (optional, later). Use the cache as training data. Every established entry is a complete
  derivation sequence blessed by the sim, so it supervises the grammar-constrained derivation model
  at every node (data-engine-design.md section 5). This is the same idea as the trace harvest (bless
  only what advanced the world) with a better bar: a falsifiable intent test.

Order and dependencies: 0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6. Phases 1 and 2 are small. Phase 4 is the
largest and the one that decides whether the cache is trustworthy. Do not skip the gate of phase 4
before phase 5, because phase 5 puts entries into NPC menus where a bad entry is seen by everyone.
Nothing here may start until wave 14a is finished, because it needs a quiet src/mind and sim.


15. OPEN QUESTIONS FOR THE USER (short, each with my default)
-------------------------------------------------------------------------------
1. Show the success test to the player before the first try? Default: show it in one plain line,
   because a misread intent is the most common failure and the player can fix it for free.
2. Is the cache global per world (shared by all characters, knowledge gated per character) or per
   character only? Default: world-level entries plus personal repertoires, as in 10.3 and 13.3.
3. How much may a model be trusted for the intent canon? Default: it only fills a closed grammar; if it
   cannot, that is a missing-quantity ticket, not a guess.
4. Should a plan ever be allowed to run when the aim is "unknown" (no testable quantity)? Default: no.
5. Should a provisional entry be served at all before it is established? Default: yes for the player
   who proposed it (the player's own idea), and for others only as a hedged option.

15b. OWNER DECISIONS (2026-11-21), replacing the defaults above where they differ
-------------------------------------------------------------------------------
- Supports and partial repair (6.x / 9.x): YES. Each step records what it achieves for later steps;
  a failing link is repaired alone and the sim re-judges the repaired plan.
- Q1 above (show the success test first): NO, against the default. The plan runs; the after-run
  report states what physically changed, so a misread goal shows there and the player's correction
  re-files the entry under the right intent.
- Q2 above (scope): a nuanced world-plus-personal split, TO BE SCOPED. The world record is the
  sim's memo of what works and is known by no one. Knowledge lives in per-person BELIEFS (true or
  false; partial plans or claims about laws; with confidence and provenance). Cultural knowledge,
  rumours and specialised knowledge are one transmission law along different paths (everyone,
  friendships, master-apprentice) with per-channel fidelity. There is no stored "local base": a
  locality's common knowledge is a statistic over its people, so travellers carry their own beliefs
  and nothing merges. Offscreen people materialise from a compact locality summary checked against
  explicit people on facts. Library request: requests/2026-11-21-mark1-knowledge-transmission.md.
- Q3 above (the model and the intent canon): the closed grammar BY DEFAULT; a missing quantity is a
  ticket, never a guess. Approximation or invented conditions come to the owner case by case, as
  rulings.
- Q4 above (an untestable aim): the steps RUN as ordinary legal actions with their real effects;
  the report makes no success claim; nothing is cached; the missing quantity is logged. (Rituals
  and superstitions stay playable: "that makes for fun plays".)
- Q5 above (provisional entries): usable at once by ANYONE WHO KNOWS the plan, judged by the sim
  every time, labelled in the menu by the character's own experience ("you've done this once"),
  never by the sim's entry state. A plan someone was TOLD succeeds less often because the transfer
  can fail: the lower rate comes from what was transferred (missing steps or supports, wrong
  details) and from the lack of practice, not from a flat penalty.



16. REAL SOURCES AND HOW I USED THEM
-------------------------------------------------------------------------------
(All from memory; no web check was done in this pass. Titles and authors I am fairly sure of. Page
numbers, numbers and exact claims are not. Check before quoting.)
- Fikes, Hart, Nilsson 1972, "Learning and Executing Generalized Robot Plans", Artificial
  Intelligence 3. The STRIPS-era macro-operators: a plan generalised and stored for reuse. (u detail)
- Mitchell, Keller, Kedar-Cabelli 1986, "Explanation-Based Generalization: A Unifying View", Machine
  Learning 1. Key by the leaves of the explanation. (used in 5.1)
- Mitchell 1982, "Generalization as Search", Artificial Intelligence 18. Narrow from the general side,
  widen from the specific side. (used in 5.3)
- Doyle 1979, "A Truth Maintenance System", Artificial Intelligence 12. Dependency-directed
  re-justification. (used in 9.1) (u exact pages)
- Zeller and Hildebrandt 2002, "Simplifying and Isolating Failure-Inducing Input", IEEE Transactions on
  Software Engineering. Delta debugging for the ablation step. (u)
- Kolodner 1993, "Case-Based Reasoning" (book); Aamodt and Plaza 1994, "Case-Based Reasoning:
  Foundational Issues, Methodological Variations, and System Approaches", AI Communications 7. The
  retrieve-reuse-revise-retain cycle is the shape of this note. (u pages)
- Wang et al. 2023, "Voyager: An Open-Ended Embodied Agent with Large Language Models", arXiv 2305.16291.
  A skill library of verified, reusable programs retrieved by description; self-verification before a
  skill is stored. Our difference: the verifier is the sim and the ledgers, and entries carry a
  generalised key and re-adjudication. (u number)
- Ellis et al. 2021, "DreamCoder", PLDI. Library learning by compressing solutions into reusable
  abstractions; this is the same pressure as "promote?" (11.5) and as chunking in Q10. (u)
- Shinn et al. 2023, "Reflexion". Failure context given back to the proposer; used in T2c. (u)
- Kambhampati et al. 2024, "LLM Modulo" (arXiv 2402.01817 (u)): models propose, external verifiers
  decide. This is D10 and the rule that the sim, not the model, judges.
- OpenWorldRPG (openworldrpg.ai, github SihanXU/openworld, MIT). Model-written rules validated and
  cached per type. We study the idea only and never its code; our layer sits one level above the
  laws and never lets a model write a law (project notes: openworld-study, plan-cache-layer).
- In-house: docs/data-engine-design.md (D1-D13, section 3.2 pieces grammar, 3.7 menu size, section 5
  grammar-constrained derivation, section 7 metrics), src/core/pieces.py (table/index/parse/PieceError),
  src/core/knn.py and src/core/trace.py (the blessed-trace store), src/mind/loop.py (decide, _two,
  _chain, _Metered), src/mind/choosers.py (draw, seed_of, Player, chain), src/mind/api.py (Option,
  Job.then, Why, DecisionRequest), src/core/data/couplings.yaml (channels, via rule ids, missing),
  docs/rulings.md (Ruling 1, Ruling 2), w14/decisions.md (decisions 8.4, Q8, Q10, Q15, faithfulness
  list).


17. SUMMARY OF THE DESIGN IN TEN LINES
-------------------------------------------------------------------------------
1. A plan is a short tree of EXISTING actions with typed holes, in the pieces grammar. Legal by
   construction; bad pieces refused by name.
2. The intent is a closed-vocabulary goal over registered quantities. If it cannot be written, that
   is a ticket.
3. The success test (Aim) is a small closed language plus universal ledger and safety clauses the
   model cannot weaken. It must fail on the do-nothing plan, and every step must be needed.
4. The key is built from the leaves of the rule trace (EBG), widened and narrowed by evidence; it
   never holds ids, names, coordinates or time.
5. A hit costs 0 model calls. A miss costs one proposal call. NPC reactive mode stays at 0.
6. A cached plan filters the actor's menu; the character's own chooser still decides.
7. Entries are stored as an event log; state is derived. Seed entries ship as data; play entries are
   git-ignored.
8. Re-adjudication: law change (touched laws only), failure in a new situation (sim-made
   counterfactual probes, then narrow), dispute (sorted into plan/test/law), seeded audits, challengers.
9. Unbuildable intents become de-duplicated tickets feeding the affordance-gap backlog; impossible
   ones become negative entries.
10. Magic Q15 uses the same cache: a proven composition is a replayable entry in the character's
    repertoire, invocable by text only with its stored tell, cost and time, within its proven range,
    at the character's current skill.
```
