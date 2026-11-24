# The science of magic: axioms, questions, test cases

Started with the owner on 2026-11-23. This is the working method for turning MARK1's magic into an exact science: what magic is, why it works, and what follows. docs/magic-design.md holds the design and its research. This file holds the walk.

**How we work:**
- Settle one question at a time, in the order of the tree below. Each question gets options, a recommendation and its consequences, and names the test cases it decides.
- Check every answer against the CASE BANK. An answer that gives a silly result on any case is wrong or incomplete.
- If a later answer conflicts with an earlier one, reopen the earlier one explicitly. Nothing changes silently.
- Where real physics, psychology or other systems can inform a question, send a library request (from: mark1).
- Where a law can be simulated, test it in a small lab PROTOTYPE before it enters the engine.
- Record decisions here with dates, and mirror them in the decisions log.
- The case bank grows over time. The owner wants it very extensive: everyday life, crafts, combat, culture, abuse and exploits, and the fiction tropes in docs/coverage/*.tsv (4,128 rows) are a ready source.

## 1. Axioms (settled)

- **A1. The One Change** (magic-design section 1): a patterned structure can move energy and momentum at a distance through a finite, conserved, out-of-equilibrium field M. The pattern steers a flow; it does not make energy.
- **A2. Arrangement, not bulk** (library, 2026-11-23; Carroll, arXiv:2101.07884): the field couples only through structures whose effect depends on their arrangement (a brain, a carving), never as a bulk force on ordinary matter. Otherwise everyday physics would already have detected it.
- **A3. Every joule booked.** Magic hands off to the existing laws, and a magic joule acts like any other joule.
- **A4. A spell is a mechanism on existing fields** (MS3): for example, "isolate the flammable gas x metres away in direction y and spark it z seconds later". It is never "the mana bar went down, the enemy went boom".
- **A5. Realism by default; game rules as packs.**
- **A7. Cases emerge; they are never coded** (owner, 2026-11-23: "if we implement it right, all these things would naturally arise from the structure of the simulation"). The case bank is a set of EMERGENCE TESTS: each case must arise from the laws, with no case-specific code.
- **A8. The zero-magic limit** (2026-11-24, with Q2.5): with no field (M = 0) or no coupling, every law reduces EXACTLY to the engine's non-magic laws (a correspondence principle). A mind with no carriers, or in a world without magic, works exactly as a real mind; magic is an addition to real life, never a requirement of it. Test: with magic off, the lab's mind tests and golden traces are bit for bit unchanged.
- **A6. Magic only allows change that could exist without it** (owner, 2026-11-23, with Q0). Every end state is a lawful physical state that some ordinary process could reach, and every step balances (Q0). Magic changes WHICH lawful change happens, and where, when and how fast. It never adds to the set of lawful states. This is the catalyst principle: a catalyst opens a pathway and changes a rate, never the equilibrium. Teleport is a pack doorway because the end state is lawful but no ordinary path is instant; true creation of mass is ruled out.

## 2. Decisions so far

- MS1 (unfamiliar effects): practice raises efficiency, so unfamiliar patterns waste more heat. Skill has an intrinsic part and an explicit part (writing the pattern in detail buys specificity, a game of player skill). The layers run from the field, through interfaces and many notations, up to spells. The default history has gods giving people compiled thematic spells that are intuitive but opaque. (magic-design addendum)
- MS2: a magic index per material (Snell, partial reflection, waveguides, lenses). Density-dependent index later. How much material matters depends on the interface.
- MS3: signatures ride whichever channel carries the tell. The tell and waste go to any physical channel. Spells are mechanisms (A4).
- MS4: every object resonates by its real body. No object is a focus by type; nobody needs a staff.
- MS5: engravings multiplexed by wavelength. Crosstalk later.
- Q1.0 (the level of theory), 2026-11-24: **(b), AN EFFECTIVE THEORY PLUS A WRITTEN DEEPER THEORY.** Magic is simulated at the engine's level (classical continuum physics, with modern physics entering as data and information theory as limits). A GENERATING THEORY is written once and checked for consistency, never run each tick: it produces the effective numbers (why carriers exist, why each substance has its magic lines, the speed c_M, the planetary dynamo). Template: physicists' hidden fifth forces carried by a new charge (some screened in dense matter, such as chameleon fields). It holds the brute fact (why the field exists) and gives in-world science a path: folk lore, effective laws, then the deeper theory. A pack may swap the deeper theory.
- Q0 (the frame), REVISED 2026-11-23: **(f), FIVE LAYERS**, "more complete for now" (owner). This supersedes the first answer, a conservation checklist, which is now layer 1's default setting.
  1. SYMMETRIES decide WHAT IS CONSERVED (Noether): every symmetry is kept by default, so energy, momentum, angular momentum, centre of mass, charge and particle number are all conserved. A pack may break a NAMED symmetry in a NAMED place (for example, no momentum conservation inside a ley line, so levitation needs no reaction partner there, and only there).
  2. A THERMODYNAMIC BUDGET decides WHETHER IT IS AFFORDABLE, with information as fuel (the generalised second law, Sagawa and Ueda): free energy available plus temperature times the information held must cover the free-energy change the outcome needs, plus losses. Improbable outcomes pay for the order they create (A6 made exact, R1). Luck has a price per bit of bias (R4).
  3. CAUSAL REACH decides WHETHER IT CAN GET THERE: a signal path from the pattern to the target that carries enough information in time (channel capacity within c_M; the ray march and the channel table).
  4. COMPUTATION decides WHAT THE THINKING COSTS: bits held, operations, and Landauer's minimum per erased bit. Searching and divination pay the bits needed to single out the answer; complexity is a real limit (a long chaotic forecast is out of reach at any power).
  5. THE ONE CHANGE is HOW IT HAPPENS: the mechanism the simulation runs.
  - The owner's A6 ("it can only allow change that can exist without it in some way") stands, and follows from layers 1-2. The pattern is a catalyst that wears.
- Q1.1 (what the field is and what carries its charge), REVISED 2026-11-23: **(b) NOW, (c) LATER.** It merges the first Q1.1 answer (energy with a phase, kept as the waves' property) with the reframed Q1.3.
  - (b) BUILT LIKE ELECTROMAGNETISM: some particles carry a small MAGIC CHARGE or MAGIC MOMENT. In ordinary matter the carriers point every which way and cancel, so A2 is EXPLAINED (only aligned or organised carriers couple), not assumed.
  - What organises carriers: a crystal lattice, an evolved organ, an engraving's geometry, a brain's activity patterns (R2).
  - CHARGED MATERIALS store nothing: their carriers are aligned and emit their own field, as a lodestone does. Heating past a critical temperature scrambles them and erases the charge.
  - MS2's INDEX comes from how easily a material's carriers respond to the field.
  - ABSORPTION jostles carriers, and the jostling ends as warmth (Q1.3).
  - (c) LATER EXTENSION: stable field STATES (normal, dead, wild regions separated by walls, which can flip when pushed hard).
- Q1.2 (where the field comes from), REVISED 2026-11-23: **(d)**, "coming towards something more doable" (owner). It keeps the first answer (natural flows, ancient deposits, inflow as a pack) and gives each part a physical cause.
  - (c) THE PHYSICS, A LAW OF CARRIER MOTION: field is made wherever carriers are moved or organised (crystals growing, living organs working, storms separating charge) and above all by THE PLANET'S CHURNING CORE, a dynamo like Earth's magnetic one. The energy is taken from each process and booked.
    - LEY LINES are the planet's own field lines, mappable like magnetic north.
    - There are POLES, a SLOW DRIFT over centuries, and rare REVERSALS.
    - Local sources (life, storms, crystals) add to the global field.
  - (b) THE CHEAP ESTIMATE for distant regions (the coarse twin, SC5): a fixed small share of local dissipation becomes field (R3).
  - ANCIENT DEPOSITS are NATURAL MAGIC LODESTONES (carriers aligned long ago), mineable, and erased by heat.
  - INFLOW from other worlds or beings stays a pack.
- Q2.1 (what lets a brain organise its carriers), 2026-11-23: **(d), A BLEND**, "the most sensible for now" (owner).
  - CORE, NEURAL SYNCHRONY: the brain holds a trace of carriers, and organised, synchronised activity moves them coherently, as aligned, synchronous cortical neurons make the field an EEG reads. Coupling = carrier count (inherited) x the alignment of neural geometry (development) x synchrony (trainable).
  - So concentration helps; strong emotion raises synchrony but blurs the pattern (MC6); fatigue and drugs disrupt it; meditation-style training raises it.
  - THE BODY ADDS A LITTLE: some carriers in the body, so trained movement and breath help modestly; gestures matter a little physically by default, without a magic language.
  - BEASTS evolve dedicated organs (C21). In humans there is no special organ, only natural variation in carrier density and brain geometry (the genetics layer's dial).
  - Predictions: deep sleep gives at most diffuse warmth; dreams give small wild leaks; animals couple less; plants make field slowly and cannot cast; a crystal-circuit machine could couple in principle (later).
- Q2.2 (what lets matter couple; how an engraving works), 2026-11-23: **(d), ALL THREE.**
  - THE BASE is four MATERIAL CLASSES, like electronics: CONDUCTORS (free carriers; traces, shields), INSULATORS (bound carriers that respond; they set the index), "MAGNETS" (aligned carriers; lodestones and charged gems that store and emit) and SWITCHABLE materials (like semiconductors; the basis of engraved logic).
  - An ENGRAVING is a real CIRCUIT: conductor traces cut or inlaid into a substrate. Purity and fewer lattice defects mean less loss and a higher Q.
  - CHARGING and some engravings work by WRITING a pattern of aligned carriers (like magnetic tape). Writing takes energy and a strong local field; heat or a strong pulse erases it.
  - LIVING MATTER actively organises carriers, so living wood, bone and blood couple better than dead. Creature materials carry order built by life, which FADES after death unless a craft fixes it (curing, tanning, firing).
  - The owner: these consequences should arise from the simulation's structure (A7).
- Q2.3 (where a person's coupling comes from), 2026-11-23: **(b), LIKE REAL HUMAN APTITUDES.** Moderate heritability (about half, over many genes, the genetics layer's model), plus rare large-effect variants (prodigies born to ordinary parents), plus development (childhood more plastic; MC3 critical periods), plus practice (training raises synchrony, with diminishing returns). Bloodlines show up only STATISTICALLY; a pack may raise heritability. Beasts' organs are species data built from the same laws and materials; evolving species over deep time belongs to world-history generation later.
- Q3.1 (what a pattern is, physically), 2026-11-23: **(c), BOTH LEVELS.** The owner: "the grammar happens by sentient thought, but what's moving and making things happen is pure wave physics."
  - THE THOUGHT LEVEL: the grammar is how minds compose and hold a spell.
  - THE CAUSAL LEVEL: what acts is a designed WAVEFORM (frequencies, phases across space, timing) meeting matter's carriers. It SELECTS by resonance (each substance's magic absorption lines, in the material record, the absorption side of MS3's lines), PUSHES by momentum transfer, HEATS by absorption, and SENSES by reflection.
  - THE SIMULATION executes the grammar's operations (fast), with each operation's selectivity, cost, precision and errors taken from the wave physics. Errors land on the substance with the NEAREST resonance. A substance whose signature the caster does not know cannot be targeted (Q0's information requirement).
- Q3.2 (what intent is; what fills the unspecified parts), 2026-11-23: **(d).**
  - INTENT IS A GOAL, as in motor control: the caster intends an outcome, and the mind's LEARNED INVERSE MODEL of how its own field affects the world turns the goal into a pattern, as reaching turns "cup in hand" into muscle commands. The model's quality is MS1's intrinsic skill, grown by practice; it is the same machinery as the learned mind's model of its own acts.
  - EXPLICIT OVERRIDES: writing specific parts yourself is MS1's explicit skill (the player's game of skill).
  - CULTURE AND GIFTED SPELLS supply ready-made models for common goals, so "fire" works for someone taught it.
  - NOISE fills only what neither the model nor the caster settles.
  - So the frightened child (clear goal, poor model) gets a huge crude result; the untrained adult (no model) gets little or nothing; the trained mage gets a precise one.
- Q3.3 (is there a line below which nothing happens), 2026-11-23: **(c), THE LINE COMES FROM NOISE.** Coupling is continuous, but every effect must beat the THERMAL NOISE of the matter it acts on (Q0 layer 2: each bit costs at least a temperature-set minimum). An incoherent intent is lost in the noise, as the brain's own real electric field is far too weak to move anything. There is no hand-set threshold.
  - Idle thoughts and daydreams never leak. Fear's high synchrony lifts a child's drive above the noise.
  - COLLECTIVE RITUAL works by SYNCHRONY: drives in phase add (Q1.1), so many people chanting in unison reach what none could alone, and out-of-step wishes cancel.
- Q3.4 (magic without deliberate intent), 2026-11-23: **(c) for INTRINSIC magic.** Any strong goal casts, conscious or not (accidental magic in children and strongly coupled untrained adults under extreme emotion); training includes emotional control; practised patterns become automatic reflexes (a startled mage's ward). Accidental magic fades through adolescence as emotional regulation matures. Dreams stay below the noise for almost everyone.
  - The owner's open conflict: if magic is change in existing fields, how does a goal such as "escape" map onto them? Is intent thematic, with the fields needing it clearer? Is it people's understanding? Taken up as Q3.5.
- Q2.4 (does the field act back on the mind), 2026-11-24: **(d).** (b) A WEAK LOOP for every coupled mind, like ephaptic coupling (the brain's own weak field nudges neural timing; Anastassiou 2011, Frohlich and McCormick 2010): a mostly SUBCONSCIOUS felt sense (mood, unease, pressure, intuition) that training can make conscious; technique acts inward; places affect thought; minds entrain one another; shared belief bends the field; strong raw pulses stun or disorder nearby minds; the mind stays in the brain. PLUS an optional evolved DEDICATED SENSE ORGAN (some beasts, rarely people), clear and directional like a bird's magnetic compass. (c) THE MIND PARTLY MADE OF THE FIELD (ghosts, possession, places remembering people) is a later extension, weighed with R5 and R8; spirits need only R8.
- Q2.5 (where the interface touches the mind), 2026-11-24: **(b), ONLY THE MIND'S EXISTING GENERIC PORTS.** The learned mind stays one learner; magic touches it through SENSE (one more channel with acuity dials; the organ is a sharper channel), ACT (casting is one more act family; the mind's world model of its own acts IS Q3.2's inverse model; costs through the body's drives), BODY STATE (field nudges move only the variables emotion, fatigue and drugs already move: arousal, choice noise, precision; plus one new variable, SYNCHRONY; never reward or beliefs directly) and OTHERS (casting perceived through its tells). Fear of mages, addiction, guilds and superstitions must EMERGE; learning magic is the same machinery as lab stage B's childhood. No field-to-reward link.
- Q3.5 (must a caster understand the physics), 2026-11-24: **(b), THE IMPLICIT MODEL IS ENOUGH; EXPLICIT THEORY EXTENDS REACH.** Understanding is mostly IMPLICIT (the learned world model of one's own acts, Q2.5: fast, partly subconscious, reliable only within experience, confidently wrong outside it); EXPLICIT theory and formulae are slow but can design effects never tried, roughly and wastefully at first (MS1). New effects come by EXPLORATION (trying variations, paid in time, energy and mishaps through the same learner) or by THEORY. No wishes (a wish pack only). A strong drive with a poor model gives a crude noisy release on its own, so no coded crude default is needed. Two kinds of mage emerge: the intuitive practitioner and the scholar.
  - THE OWNER'S ADDITION: PATTERN-MATCHING MAGIC SYMBOLS WITHOUT UNDERSTANDING HAS SOME EFFECT TOO. It emerges three ways: a DRAWN OR CARVED copy is matter geometry and works as far as its substrate and precision allow (C06, C46); a symbol TRACED BY GESTURE OR CHANTED moves the body's small carrier share in its shape (Q2.1), a weak physical effect; a symbol HELD IN MIND partly organises neural activity into its shape, a weak, imprecise version. Understanding adds precision and flexibility on top. This is also how learning starts: watching teaches the movement far more than the outcome (library: 0.77 against 0.17 SD), so apprentices copy symbols first and understand later.
- Q4.1 (how a pattern steers energy), 2026-11-24: **(d), ONE ACCOUNTING OF THREE ROUTES.** A spell's energy splits into the VALVE (uphill, drawn from a store: field, body or charged material), TRIGGERS (downhill, released from the world's own metastable stores: fuel, strain, poised masses, reactions; the spell pays only the nudge and the information) and the DEMON (sorting what is there by information, bounded by temperature x information, paid in bits and Landauer waste heat). Working with nature is cheap and against it expensive; knowing where the world stores energy makes a mage stronger for the same reserves.
- Q2.6 (why sapient beings have access; why interfaces work), 2026-11-24: **(d), THREE LAYERS.** PHYSICS: all organised matter couples a little. BIOLOGY: evolution tuned it, strongly in some beasts, weakly in humans (the felt sense and some coupling), with a natural spread (Q2.3). CULTURE: interfaces are TECHNOLOGIES built on that capacity, as writing is built on vision and language (neuronal recycling, Dehaene): spells, notations, chants and formulae work because generations refined them to produce patterns the field responds to and a mind can hold; failures died out, superstitions survived where feedback was noisy. In the default world the gods' gifts (MS1) started the culture layer. No special magic language is needed: symbols are magical as good tools are, by selection.
- Q4.2 (speed, range and aim), 2026-11-24: **(c), SPEED SET BY MEDIUM AND FREQUENCY** (dispersion, from Q1.1 and MS2). A pulse's momentum is its energy over its speed, so slow magic means huge recoil; in the default world typical casting frequencies are light-like in air (strong strikes fast, recoil small). Waves are slower in dense, strongly coupled matter (late and weak through stone); low frequencies crawl (wards and hums spread slowly); complex spells smear with distance, so simple ones carry far. A pack may make air slow (visible, dodgeable bolts with real recoil). Range and aim stay as designed: spreading, absorption, near field efficient and quiet, diffraction (fine aim at range needs short wavelength and aperture, where antennas and staffs help, MS4), and information (you can only aim at what you can locate).
- Q4.3 (sensing and what it reveals), 2026-11-24: **(b), PASSIVE AND ACTIVE, WITH RECIPROCITY.** Passive sensing (feel fields, read tells) is silent. Active sensing (send a pulse, read reflections: location, shape, material) makes every ping a tell, so stealth against sight is a real choice (passive against active sonar). Detection is limited by noise (Q3.3): the sensor's coupling, the signal and the background, so loud nodes hide people. Quiet casting (short range, low power, near field) hides the tell at the cost of reach. FORENSICS emerges from Q2.2: a strong spell writes a faint pattern into nearby magnet-class materials, as cooling lava records Earth's field (archaeomagnetic dating), fading with time and heat.
- Q5.1 (how magic hands its energy to the ordinary world), 2026-11-24: **(c), CARRIER RESONANCES PLUS A WEAK LINK TO ELECTROMAGNETISM.** Each channel has a physical route from matter's resonant absorption (Q3.1): HEAT (absorption ending as jostling), PUSH (momentum), LIGHT (excited carriers relaxing, like fluorescence: MS3's light share and cold light), CHEMISTRY (exciting one bond until it breaks or reacts, like laser-driven chemistry: Q4.1's trigger), SOUND (pressure from rapid pushes). PLUS a tiny MIXING with ordinary electromagnetism (as proposed for dark photons): strong pulses spark wires and drive currents through nerves (Q2.4's stun), lightning raises the local field (C29's cause), compass needles twitch near strong casting (an in-world detector, C30's path). The mixing is one constant, tiny so A2 holds; packs may change it.
- Q5.2 (how deep into matter magic reaches), 2026-11-24: **(b), THE NUCLEUS TOO, AT ITS REAL COST AND WITH ITS REAL HAZARDS.** Conjuring is gathering (A6) and copying is gathering plus information (an exact copy costs the original's full information: a pure crystal is cheap, an exact person impossible). Changing an element is lawful but pays the real nuclear price (energies about a million times chemistry's; radioactive products; real: a few thousand atoms of gold made from bismuth in a particle accelerator, 1980). So transmutation is possible in principle, out of reach in practice except with vast stores, and dangerous in a way nobody in the world understands. Everyday alchemy is chemistry because chemistry is affordable, not by rule. The owner is "not exactly sure how" magic reaches the nucleus: the mechanism belongs to the deeper theory (Q1.0); a candidate is nuclear resonance at gamma-like frequencies (real: nuclear resonance fluorescence), which needs enormous energy density.
- Q5.3 (magic acting on other minds), 2026-11-24: **(b), PRECISE, BUT ONLY THROUGH THE PORTS.** Like brain stimulation (flashes, disrupted speech, small mood and sleep effects; never inserted thoughts): a skilled caster can target calm or agitation, drowsiness or alertness (synchrony rhythms), focus or confusion, and INDUCED PERCEPTS (a flash, a sound only the target hears), which the target's own mind interprets. Beliefs change only through perception and learning, never by being written (emergent conditioning: a mage who always induces calm comes to be liked, and loses it when found out). CONTENT-LEVEL magic (planting beliefs, memories, commands) needs a model of that specific brain's code (Q0 information layer): out of reach for strangers, conceivable only through deep long bonds, perhaps never by default; a pack. RESISTANCE and CONSENT fall out of forced synchrony (Q2.4): a focused or trained mind holds its own rhythm and resists; a willing mind falls into step on purpose, cutting the energy needed by orders of magnitude; distance and conducting caps shield. Depiction follows the content rules (CV7).
- Q5.4 (magical healing), 2026-11-24: **(c), ROUTE 1 PLUS ROUTE 2 AS PATTERN GUIDANCE** (refines healing-design HL4). Route 1 feeds the body's own healing (energy, warmth, perfusion, immune support) by the same repair law. Route 2 GUIDES THE BODY'S OWN REGENERATION with positional signals, as bioelectric patterns guide regeneration in real tissue (Levin's lab: flatworms and frogs). The body supplies cells, mass (food) and time; speed is capped by cell division. The template must come from somewhere real (Q0 information): the surrounding healthy tissue (small wounds can heal scarless with modest skill), the mirror-image part (a finger regrows from its twin), or the healer's knowledge. Forcing growth without a good pattern gives a disordered, tumour-like lump. Big injuries need food and time; elders heal slower; skill improves outcomes; the dead stay dead (no living cells to guide; the mind was in the brain).
- Q6 (explicit limits), 2026-11-24: **(a) NO EXPLICIT LIMITS BY DEFAULT; (b) GAME LIMITS AS PACKS.** The library's eight cost laws and five guards are consequences: upkeep is the held state's own loss rate; tipping is the trigger route; draw caps come from local density and refill; downhill/uphill is valve against trigger; triggered spells draw from a real store when they fire; link strength is information; capacity is feature size and channel capacity; the bit floor is the budget layer; doorways carry energy and momentum (Q10); runaway coupling saturates at the local store; holders saturate like magnets. The one limit the booking makes visible is the HOLDER'S OWN HEAT: 1 MJ of overhead warms a 70 kg body by about 4 C, near fatal heatstroke, so large effects need external transducers, stores or patience (a natural cooldown with no cooldown rule). What the player sees as "mana" is the character's felt sense of heat, fatigue and the local field, to the precision the character could know (MC7). Mana pools, cooldowns and spell slots exist only in packs.
- Q5.5 (force at a distance), 2026-11-24: PAUSED. The owner found the options (radiated push, magnet-like static grip, slow-magic packs) shallow: mechanism stories patched one at a time. In a world with magic it is fully part of the physics, and we cannot derive a made-up fundamental physics when we cannot fully do so for real physics. Correction recorded: radiated push is weak at light-like speeds (force = power / speed, so holding 50 kg would take about 5 GW), so a light-like "force bolt" acts through heat or a gas blast.
- METHOD CHANGE (owner, 2026-11-24): stop inventing mechanisms; first choose a ROOT FRAMEWORK from which magic's laws derive, then re-derive earlier decisions as its consequences. Candidates (section 8): least action and effective field theory, the gauge principle (a "why" for the field), Onsager/GENERIC and port-Hamiltonian systems, axiomatic thermodynamics, resource theories, constructor theory, operational and process theories, maximum entropy, digital physics, the free energy principle, assembly theory. Several are already implicit: Q0 is a resource or constructor-theoretic statement; the grammar is a process theory; the ports are port-Hamiltonian; the coarse twins are maximum-entropy estimates. Library request 2026-11-24-mark1-foundational-frameworks.md (go given). Then questions FQ1 (the root), FQ2 (which others become tools or consequences), FQ3 (does the gauge principle give the field's why).
- CORRECTIONS from the library's frameworks answer (2026-11-24, Library responses/2026-11-24-mark1-foundational-frameworks.md S5):
  - Q0 layer 2 exact form: W_in >= dF - kT * I_used, with I the MUTUAL information between record and target, only the part actually used, on average for one bath; erasing the record later costs at least kT * I, so the full cycle obeys the ordinary second law; the information fuel decays as the correlation decays.
  - A6 restated: under exact conservation a perfectly returned, uncorrelated catalyst enables nothing (arXiv:2306.00798). The pattern must be slightly changed or left correlated: its WEAR is what pays for what it enables, a booked output of every cast.
  - A2 and screening: chameleon-type screening hides a force by bulk density, which A2 rejects; keep it only for field walls (Q1.1 (c)) and the density-dependent index (MS2).
  - The field must be a VECTOR (spin-1) field: a scalar (spin-0) field couples to all matter with one sign, contradicting A2. The formal write-up's scalar is to be replaced.
  - Q4.2 "light-like in air" holds only with a quasi-static field solve plus ray-marched pulses, or an implicit solver: an explicit update on 5 cm voxels at 25 ms is stable only up to about 1.15 m/s.
  - Q5.5 can reopen: a static grip is field stress (paid in setup energy and losses), not radiated power; the 5 GW figure is for radiated push only.
- SOURCED REFINEMENTS from the library's science answer (2026-11-24, Library responses/2026-11-23-mark1-magic-science-foundations.md):
  - Q2.1 REDEFINED: coupling uses band-limited STRUCTURED coherence, not raw synchrony amplitude. Coherent sources add as N and random ones as sqrt(N) (Nunez: 1% of cortical columns in step outweigh the other 99% about 3 to 1; a synchronous group wins once it exceeds sqrt(N_total), about 0.3%). Anaesthesia and deep sleep RAISE raw amplitude while destroying structure, so they cannot cast. Numbers: 0.2-1 nA m per mm2 of cortex; 6-20 cm2 of synchronous cortex is visible at the scalp.
  - Q2.4: ephaptic entrainment at 0.74 mV/mm at 1 Hz but 5.58 mV/mm at 30 Hz (a frequency penalty); making neurons fire needs 25-50x more, so a stun needs fields of several V/m.
  - Q2.3: training grows with the LOG of practice hours (meditation gamma d about 0.4-0.8, r about 0.33 with log hours; neurofeedback g about 0.3; some people never learn).
  - Organs (C21, C53): one stacking law (cells in series set voltage, columns in parallel set current; an eel about 6,000 cells of 0.1 V); weakly electric fish spend 3-22% of metabolism; evolved at least six times from muscle; sensors limited by thermal noise and integration time (sharks: 5 nV/cm).
  - Charged materials (Q1.1, Q2.2, Q4.3): a stored state lasts tau0 exp(E/kT) (E/kT about 40 for ten years); writing needs a field above a coercivity that falls with temperature; ONE erasure law with three routes (past the Curie point, slow thermal decay, a strong or alternating field); Curie points iron 1,043 K, magnetite 858 K, nickel 627 K.
  - Dynamo (Q1.2): 25-65 microtesla analogue; westward drift about 0.2 degrees a year; poles move 0-60 km a year; about 2.2 reversals per million years (Poisson, with long quiet superchrons), each 1-10 thousand years; excursions drop the field to about 6% for under a thousand years; 0.1-5 TW of loss fed by 5-16 TW of core heat.
  - Information: kT ln2 = 2.87 zJ per bit at 300 K; fast erasure pays extra; real demons reach 28-98.5% of the bound; 1 kJ would need 3.5e23 bits, so information governs order and luck, never energy budgets (C33).
  - Q3.2 strengthened: motor noise grows with command size and is left uncorrected where the goal does not care (minimal intervention). Q7.2: working memory holds about 4 chunks per level; experts enlarge chunks; notation is external storage and adds no power. Q3.4 corrected: the best arousal FALLS as the task gets harder (the original Yerkes-Dodson data), so intricate spells need calm.
- FQ1 (the root), 2026-11-24: **(d), WHY THE FIELD EXISTS IS THE ROOT: magic is made real from the bottom up, as a LARGE RESEARCH PROJECT** (owner). The field EMERGES from an underlying substrate, as real electromagnetism-like gauge fields emerge from local constraints in organised matter (spin ice's Coulomb phase with its magnetic monopoles; string-net condensation). The library's GENERIC root with a gauge field becomes the DERIVED EFFECTIVE LAYER that the engine runs (Q1.0), with its numbers derived from the substrate instead of chosen. Program: substrate research, toy-model simulations in the lab, coarse-graining to the effective theory, and the consequences the effective theory misses. First scoping question RQ1: where the substrate lives.
- RQ1 (where the substrate lives), 2026-11-24: **(a) A HIDDEN MEDIUM FILLING ALL OF SPACE** (working hypothesis): the field is its collective behaviour, ordinary matter sits inside it, and it carries magic through vacuum, air and matter with a material index (fits MS2, Q4.2 and ley lines). RIVAL HYPOTHESES kept and compared in the toy-model stage: (b) the field as a collective mode of a hidden property of ordinary matter (empty space magically dead, thin air weak); (c) a sparse background population of special particles whose density varies by place.
- RQ2 (classical or quantum substrate), 2026-11-24: **(c), BOTH, IN ROLES.** The written deeper theory is QUANTUM (modelled on quantum spin ice and string-net condensation, where emergent light-like waves and charges are known); the research and simulation model is CLASSICAL WITH INERTIA (simulable at scale), tested against what the quantum theory says should emerge. The owner: at the macro scale the quantum substrate's effects APPEAR classical, so the classical model is the macro interpretation (the correspondence limit). Disagreements are findings: an effect needing the quantum substrate is rarer and stranger in the world. First experiment R2.1: a classical constrained lattice with inertia; measure whether waves travel, how fast, and whether its defects obey a Coulomb law.
- RQ3 (what kind of medium; what a carrier is), 2026-11-24: **(b) THE FIRST RESEARCH COMPARISON, with (c) A MIX OF MEDIA AS ITS EXPECTED CONSEQUENCE.** The owner: "even in existing scientific theory I feel the correct answer is some mix of all that exists, and that would be what we come at; it is hard to write that down in the first go, but we must try." The kinds of medium (rule-following magnet, string-net, superfluid, elastic solid, hybridising medium, topological medium, fractional charges, induced-only) are to be compared by library survey and toy simulation, judged against the target list R0 and the case bank. A first attempt at writing the mix is working hypothesis H1 (section 9). Working assumptions across all media: the THREE PORTALS (binding, mixing, energy exchange) are the only ways the medium touches ordinary physics; charges are free and bound where the medium allows.
- FM1 (what makes a symbol work), 2026-11-23: **(b) by default.** Minds work by MEANING (any notation that is understood); shaped matter works by GEOMETRY (like circuits); voice and gesture are channels for a mind. (c), sounds and gestures that work by their physical form, is open: default or pack (Q8.4).

## 3. The question tree (dependency order)

- **Q0. The frame.** Is a spell possible if and only if every conserved quantity and the information it needs balance (the library's constraint layer), with the One Change as the mechanism?
- **Q1. The field.**
  - Q1.1 What kind of quantity is M: a scalar energy, energy with a phase, several components, or no store at all (pure control)?
  - Q1.2 Where M comes from and where it goes; why it is finite and out of equilibrium.
  - Q1.3 Why everyday life does not notice it (A2), and what that rules out.
- **Q2. Coupling: what can touch the field, and why.**
  - Q2.1 Minds: what about a brain makes it a transducer; the coupling dial; animals, plants, devices and artificial minds.
  - Q2.2 Matter: geometry at the field's scale; crystals, purity, lattice order.
  - Q2.3 Bodies: coupling organs evolved by beasts; inheritance (with the genetics layer).
- **Q3. Pattern and intent.**
  - Q3.1 What a pattern is: information in bits, and the grammar.
  - Q3.2 What intent is in a mind, and what fills the parts left unspecified (priors, habits, noise).
  - Q3.3 How specific intent must be. Is thinking "fire" enough?
  - Q3.4 Emotion, arousal, the unconscious and sleep. A child's intent; accidental magic.
  - Q3.5 Must the caster understand the physics? (OPEN, the owner is thinking, 2026-11-23.) The owner: it depends on WHAT AN INTERFACE IS, whether it is intrinsic to sapient beings, and whether people can FEEL the field, or whether magic is like a science where specific shaping of the field makes it work. All of these are interconnected, and the answer affects the rest. The owner's versions:
    - V1 PHYSICAL CHANGES MOVE THE FIELD (magic as a science). This is the CURRENT position: Q1.1 carriers are particles, Q2.1 synchronised brain activity physically moves them, Q2.2 engravings are circuits; the mind is special only as a fine, self-reshaping arrangement (R2). Choosing it reopens nothing.
    - V2 THE MIND SHAPES THE FIELD DIRECTLY. The mind would be something beyond its brain's physics. It would reopen Q1.1 and Q2.1 and loosen A2, so it suits a pack (a spirit or dualist setting) better than the default.
    - V3 THE MIND IS PARTLY MADE OF MAGIC (the middle). It can be added to V1 as a FEEDBACK LOOP. Real analogue: ephaptic coupling, where the brain's own electric field slightly affects its neurons. Then people can FEEL the field (the inner sense, MS1's intrinsic part); posture, breath, chant and ritual act INWARD (they change the caster's own field and so the caster's state) as well as outward; strong ambient fields affect thought (unease or intoxication near nodes).
    - The choice is mainly whether to ADD V3's loop to V1, and whether V2 exists as a pack. Q3.5's reach question (understanding only; the field finds a way; understanding plus a crude evolved default plus a small paid search) and Q4.3 (sensing) depend on it.
- **Q4. Steering: how a pattern moves the flow.**
  - Q4.1 What "steer" means physically: a valve or catalyst, Maxwell's demon, Landauer's cost per bit.
  - Q4.2 Range, aim and timing: c_M, the index (MS2), diffraction, near and far field.
  - Q4.3 Sensing: coupling both ways, detection and tells.
- **Q5. What magic changes in the existing fields.**
  - Q5.1 Which channels M couples into (heat, motion, light, sound, chemical bonds, electricity, others) and their splits (MS3).
  - Q5.2 Matter: moving mass, transmutation limits, conjure as gathering.
  - Q5.3 Minds: neural bias, resistance and consent.
  - Q5.4 Life: healing through the existing repair laws.
- **Q6. Costs and limits:** the library's eight candidate laws and five guards (requests/2026-11-23-library-mark1-magic-principles.md) and MC1-MC8.
- **Q7. Interfaces.**
  - Q7.1 Media as channels (rate, noise, tell).
  - Q7.2 Notations and working memory: why formulae exist; many notations compiling to one grammar.
  - Q7.3 Engravings and devices: circuits (MS5).
- **Q8. Open slots from earlier answers.**
  - Q8.4 FM1(c): are physically resonant sounds and gestures in the default world, or a pack?
- **Q9. Learning and culture.**
  - Q9.1 Skill: intrinsic and explicit (MS1).
  - Q9.2 Gifted spells and taking them apart (MS1's cultural path).
  - Q9.3 How magical knowledge spreads, errs and is lost (the knowledge design).
- **Q10. Beyond the One Change:** doorways and teleport, rewind, conjure, planes. Earlier owner decisions apply: bounded versions now, unbounded after a playable MVP, true creation off by default.

## 3a. The canonical pure models (reference for hybrids; owner asked 2026-11-23)

Each model is consistent on its own; a hybrid is a choice of which one carries the weight. They are sorted by where the cause lives.

| # | Model | The cause lives in | What an interface is | Can you feel it? | A parrot recites a spell | Examples |
|---|---|---|---|---|---|---|
| 1 | Naturalist: magic as physics | Matter; the right physical configuration moves the field | Any physical setup: brains, engravings, devices | Only through organs or instruments that couple | Nothing, unless its sounds are physically right | qntm's Ra; Sanderson's Allomancy |
| 2 | Idealist: magic as mind | The will; minds shape the field directly | Aids to focus; optional | Yes, directly | Nothing | Psionics; the Force |
| 3 | Linguistic: magic as code | Information; reality responds to correct symbols and true names | The language itself; correctness is everything | Not needed | Works if exact | Earthsea; the naming in The Name of the Wind |
| 4 | Animist: magic as relationship | Other beings act for you | Social: asking, offering, naming, bonding | You perceive the beings | Only if the spirit is fooled or amused | Shamanism; clerics; warlocks |
| 5 | Constructivist: magic as belief | Collective belief; reality bends to consensus | A paradigm | As certainty | Nothing | Mage: The Ascension; Pratchett's gods |
| 6 | Vitalist: magic as inner energy | A cultivated life energy in the body | The body: breath, posture, stages | Yes, as a bodily sense | Nothing | Cultivation novels; Beware of Chicken |
| 7 | Transactional: magic as exchange | A strict ledger | Whatever pays the price | Not needed | Only if it pays | Fullmetal Alchemist |
| 8 | Systemic: the rules are the physics | An adjudicating System | Menus, classes, levels | Through the interface | Whatever the System allows | LitRPG; Dungeon Crawler Carl |

- What each pure form breaks in our axioms and cases:
  - (1) breaks none; it is the current spine.
  - (2) breaks A2.
  - (3) breaks FM1's translation case (C08) and the parrot case (C04).
  - (4) is lawful when the beings are real minds in the world (MC4).
  - (5) breaks A5 (reality depends on opinion).
  - (6) is lawful if its energy is real; it is V3's neighbour and the cultivation pack's model.
  - (7) is a constraint, not a source of causes (Q0 layer 2).
  - (8) is allowed only as an in-world fact (the earlier CV decision).
- WHERE MARK1 SITS (2026-11-23): (1) as the spine with (7) as its ledger; (4) through real beings; (6) reachable by adding V3's loop; (2), (3), (5) and (8) are packs. The open blend is how much of (6), and whether any of (3).

## 3c. The generative stack (owner direction, 2026-11-23; the derivation is PROPOSED, to confirm)

The owner: the pure models are not rivals. There is a version where each builds on the others. Magic is intrinsic to how the world works (1); that is what magic is and where it resides. The mind can be an interface, harder to use because it is flexible, maybe subconscious; that is how it is reached. Magic must be interfaceable like code; that is how it is specified. Relationships, belief, storage, exchange and Systems come from composition. Keep some mysticism while staying explainable.

Three axes: SUBSTANCE (what magic is), ACCESS (how it is reached), SPECIFICATION (how it is written). Composition builds the rest. The proposed derivation, bottom up:
1. SUBSTANCE, magic as physics (model 1): the field and its carriers (Q1.1). Ordinary matter cancels, so NOTHING HOLDS MAGIC BY DEFAULT; holding is an achievement.
2. EXCHANGE (model 7) follows: conservation and the budget (Q0).
3. HOLDING: aligned carriers hold magic (lodestones, Q1.1). People can learn to align their own, a practised skill; that IS inner energy (model 6): cultivation is training the body into a charged material.
4. ACCESS THROUGH MATTER: engravings and devices, precise but fixed (Q2.2).
5. ACCESS THROUGH MINDS: synchronised activity couples (Q2.1), mostly SUBCONSCIOUSLY through the learned model (Q3.2), which its owner cannot see into. Flexible but noisy, so precise mind-casting is harder than a carving. Model 2 appears weakened: the mind shapes magic, through its brain.
6. SPECIFICATION, code (model 3): the grammar is the language; formulae are source code a mind interprets; engravings are code compiled into matter that runs whoever carved it.
7. RELATIONSHIP (model 4): any beings that hold social constructs and move magic make animism by themselves (bonds, offerings, pacts).
8. BELIEF (model 5): shared belief synchronises many minds (ritual, Q3.3), and with the V3 loop their coupled patterns bias the local field, so near believers reality bends toward the consensus, within the budget, never against physics.
9. SYSTEMS (model 8) by composition: a vast compiled pattern that ordinary minds use through simple thoughts (menus, classes, levels). Gifted spells (MS1) are a small one; a LitRPG System is a huge one; it can be wrong because it is an artefact.
10. SPIRITS AND GODS: if minds can be partly made of magic, minds made wholly of self-sustaining field patterns (R8) are possible, so animism's beings emerge too.
- Other families beyond the eight also fall out: correspondence ("like affects like", R10 shared modes) and probability or luck (Q0 layer 2). Narrative magic stays a pack.
- NEW INGREDIENTS this needs beyond what is decided: (i) the V3 FEEDBACK LOOP (the mind's own field and the ambient field act back on thought), needed for the inner sense, subconscious access, belief bending the field, and spirits; (ii) self-sustaining computing field patterns (only for field-native beings, later).
- MYSTICISM lives in KNOWLEDGE AND COMPLEXITY, not in the laws: exact in the engine, known to no one in the world (cultures hold competing, partly wrong theories, MC2; characters see only what they could know, MC7); access is subconscious, so it feels like intuition; the loop of minds, beliefs and field is chaotic; the deepest layer is a brute fact (why the field exists), with layers no culture has found yet. Explainable, but not explained to anyone in the world.

## 3b. Dependency map (to build)

The tree above gives the ORDER; the owner noted that decisions are interconnected. Each decision will list the earlier decisions it RELIES ON, so a changed answer shows at once which others move. To fill in when Q3.5 is settled.

## 4. The case bank (first 26; expectations are PROVISIONAL until the questions they test are settled)

| id | case | a sensible outcome (provisional) | tests |
|---|---|---|---|
| C01 | A frightened 6-year-old cornered by a dog | If the child couples at all: a big, crude, unaimed release (the curtains catch, or the dog is thrown back), then exhaustion | Q2.1, Q3.4 |
| C02 | An untrained adult thinks "fire" hard | Usually nothing; with some coupling, a weak, wasteful, wild flicker | Q3.2, Q3.3 |
| C03 | A trained mage thinks "fire" | Works, efficiently, for the practised case only | Q3.2, Q9.1 |
| C04 | A parrot recites a spell perfectly | Nothing measurable: a parrot couples weakly, and rote sound moves only the body's small carrier share (revised with Q3.5) | Q2.1, Q3.5, Q8.4 |
| C05 | A recording of a chant is played | Nothing in the default world; perhaps something in a resonant-sound pack | Q8.4 |
| C06 | An illiterate mason copies a ward rune precisely | The ward works | Q2.2, Q7.3 |
| C07 | A scholar copies the same rune sloppily | It fails, or misfires as a neighbouring pattern | Q2.2, Q7.3 |
| C08 | A spellbook is translated into another language | Works for a reader who understands it; ambiguity in translation costs reliability | Q2.1, Q7.2 |
| C09 | English words are carved in stone | Nothing | Q2.2 |
| C10 | A fireball made by gathering gas and sparking it | The energy is chemical and limited by the oxygen there. Small cast cost, smoke, waste heat, a visible tell | Q4.1, Q5.1, A4 |
| C11 | A big spell cast inside a drained crater | Weak, or fails; draws from farther away; the hole refills slowly | Q1.2, Q6 |
| C12 | A gifted (compiled) spell used in an odd situation | Works as its maker built it, poorly matched; cannot be adapted on the spot | Q9.2 |
| C13 | A scholar takes a gifted spell apart over years | Learns its dials, and can then vary it | Q9.2 |
| C14 | Two casters race a spell and its counterspell | Interference needs phase knowledge; timing and speed decide | Q1.1, Q4.2 |
| C15 | A ward's hidden second function is found by probing at another frequency | Discoverable by systematic probing (MS5) | Q7.3 |
| C16 | Healing a deep cut | Speeds the body's own repair laws within their limits; costs the body and the caster | Q5.4 |
| C17 | Telekinesis lifts a 50 kg rock | The reaction is booked to the ground or the caster; holding it costs | Q4.1, Q5.2 |
| C18 | Calming a frightened horse | Works (arousal is a port); a command does not; resisted by a strong state; consent lowers the cost for people | Q5.3 |
| C19 | A sleeping person casts in a dream | To be decided (Q3.4) | Q3.4 |
| C20 | Casting from a lead-lined room, or underwater | Attenuated by the materials' index and loss (MS2) | Q4.2 |
| C21 | A beast evolved an M organ (like an eel's electric organ) | It couples by biology, with costs paid by its body | Q2.3 |
| C22 | A non-mage uses an engraved battery-and-trigger device | Works, because matter couples and the store holds the energy | Q2.2, Q7.3 |
| C23 | A child raised by mages against one raised without magic | Different abilities from culture, priors and plasticity, not from a flag | Q9 |
| C24 | Someone tries to teleport with no doorway | Impossible in the base world; with a doorway pack, potential energy and momentum are conserved | Q0, Q10 |
| C25 | A villager is sure a charm works, and it does nothing physical | Only a placebo effect, unless they cast, when belief lowers their noise (MC6) | Q7, MC6 |
| C26 | A novice tries a 0.9 GJ effect from the body alone | Delivers only what the body supplies (about 0.37 GJ of fat at most), and harms the body | Q6, A3 |
| C27 | A sealed cave with no sources | The field slowly leaks away to heat, and magic fades there | Q1.2, Q1.3 |
| C28 | A city over-uses its region's field for generations | Regional depletion, a tragedy of the commons; people go farther, or mine deposits | Q1.2, Q6 |
| C30 | A natural philosopher carries a fine thermometer along a ley line | (provisional, Q1.3) a small, consistent warm anomaly, so the field can be found by science | Q1.3 |
| C31 | A wild magical surge in an empty desert, and the same near a crystal grove | (provisional) only warmth in the desert; light and effects near the grove | Q1.3 |
| C32 | A skeptic tests a mage with ordinary instruments | Finds nothing directly, only the effects (heat, motion, light) | Q1.3, A2 |
| C33 | A gambler loads dice by magic | Possible, at a minimum energy per bit of bias; heavy bias over many throws costs a lot | Q0 layer 2 |
| C34 | Finding one hidden person in a city by divination | Costs the bits needed to single them out; a link, a name or a trace cuts the cost | Q0 layers 3-4 |
| C35 | Forecasting the weather a month ahead by magic | Out of reach at any power (chaos and computation) | Q0 layer 4 |
| C36 | A charged crystal is heated in a forge | It loses its charge past a critical temperature (its carriers scramble) | Q1.1 |
| C37 | A cage of a material whose carriers move freely | Blocks spells, like a Faraday cage | Q1.1, Q4.2 |
| C38 | Why one person couples strongly and another barely | How well a brain can organise its own carriers: biology, development and practice, not a flag | Q1.1, Q2.1 |
| C39 | A lodestone crystal on a pivot | Points along the field: a magical compass for navigation | Q1.2 |
| C40 | A famous ley line over centuries | Drifts off its ancient temples; the offset dates them | Q1.2 |
| C41 | A global field reversal | A rare, world-scale upheaval in magic | Q1.2 |
| C42 | A mage with a head injury | Keeps power but loses fine control (damaged geometry) | Q2.1 |
| C43 | Months of meditation-style training | Coupling rises (synchrony is trainable) | Q2.1 |
| C44 | A mage under deep anaesthesia | Cannot cast: raw synchrony rises but structured coherence is destroyed (revised with the library's science answer) | Q2.1 |
| C45 | A crystal-circuit machine with organised currents tries to cast | Possible in principle; to be decided with Q2.2 and devices | Q2.1, Q2.2 |
| C46 | A correct rune cut into plain granite, and the same rune inlaid with conductor ore | The granite one does nothing; the inlaid one works | Q2.2 |
| C47 | A dragon-bone wand over decades | Weakens unless it was cured | Q2.2 |
| C48 | A switchable crystal in an engraving | Acts as a logic gate | Q2.2, Q7.3 |
| C49 | A strong magical pulse near written enchantments | Wipes them, like erasing a tape | Q2.2 |
| C50 | Children of two strong mages | Above average on the whole, widely spread; some ordinary | Q2.3 |
| C51 | A gifted child born to non-mage parents | Happens rarely (a large-effect variant) | Q2.3 |
| C52 | Someone who starts training as an adult | Reaches less than someone trained from childhood | Q2.3 |
| C53 | An eel-like beast's field organ | Species data from ordinary laws and materials, paid by its body | Q2.3 |
| C54 | Isolating one gas from a mix of similar gases | Errors grab the gas with the nearest resonance | Q3.1 |
| C55 | A caster tries to target a substance they have never studied | Cannot, until they learn its signature | Q3.1, Q0 |
| C56 | Probing a sealed jar's resonances | Identifies what is inside: magical spectroscopy | Q3.1, Q4.3 |
| C57 | Two substances with identical resonances | Cannot be separated by magic, a lawful limit | Q3.1 |
| C58 | A skilled caster overrides their habitual fireball's timing | Works, at an effort cost | Q3.2 |
| C59 | Someone taught "light" tries "dim light" | Easy light, but dim light needs new learning | Q3.2, Q9 |
| C60 | A caster aims at a goal they cannot picture (curing an illness they do not understand) | Fails or goes astray | Q3.2, Q5.4 |
| C61 | A crowd each silently wishes for rain | Nothing (each drive is below the noise and out of phase) | Q3.3 |
| C62 | A hundred people chant in unison | A small effect none could reach alone | Q3.3, Q1.1 |
| C63 | One person daydreams | No measurable effect | Q3.3 |
| C64 | A strongly coupled untrained adult in a rage | A rare flare | Q3.4 |
| C65 | A trained mage is startled | Raises a ward before thinking | Q3.4 |
| C66 | A child's accidental magic through adolescence | Fades as emotional control matures | Q3.4 |
| C67 | A mage on calming herbs | Leaks less and casts more weakly | Q3.4, MC6 |
| C68 | A sensitive near a powerful node | Feels uneasy without knowing why | Q2.4 |
| C69 | Two mages work side by side | They fall into step; group casting gets easier | Q2.4, Q3.3 |
| C70 | A sensitive in a dead zone | Feels flat or dull | Q2.4 |
| C71 | Years of training | The vague feeling becomes a conscious, directional sense | Q2.4, MS1 |
| C72 | A powerful caster's raw pulse | Stuns or disorders the people nearby | Q2.4 |
| C73 | A widow feels her dead husband's presence at home | Grief and memory by default; a faint imprint only if R5 is adopted; real only under the later (c) | Q2.4, R5 |
| C74 | A beast with a field-sensing organ | Tracks a mage by their field | Q2.4, Q2.3 |
| C75 | A world with no magic | Every mind behaves exactly as without the magic laws | A8 |
| C76 | A sensitive and a node that unsettles them | Learns to avoid it from experience, not code | Q2.5 |
| C77 | A village that has seen mages cause harm | Fear of mages emerges, with no coded prejudice | Q2.5 |
| C78 | A child in a casting family | Learns to feel and cast through the same childhood learning as drinking | Q2.5, Q9.1 |
| C79 | A person with no carriers | Untouched by field nudges: magically deaf | Q2.5, A8 |
| C80 | People and nodes | Node-seeking emerges only if nodes serve a drive through the body | Q2.5 |
| C81 | An illiterate village healer with decades of practice | Excellent within her practice; fails on a new kind of injury | Q3.5 |
| C82 | A scholar who has never cast designs a new effect on paper | The first cast works roughly and wastefully | Q3.5, MS1 |
| C83 | A caster tries variations on a known spell | Slow progress toward a new effect, with occasional mishaps | Q3.5 |
| C84 | A caster believes a gesture is necessary because it once coincided with success | It does nothing, except through their confidence | Q3.5 |
| C85 | A desperate parent wishes their child cured, with no model of the illness | Nothing useful, or something crude and harmful | Q3.5, Q0 |
| C86 | An apprentice traces a master's symbol by rote | A weak, imprecise version of the effect | Q3.5 |
| C87 | A symbol drawn in sand versus inlaid in conductor ore | The sand copy does very little; the inlaid one works | Q3.5, Q2.2 |
| C88 | Lighting a stove, and heating a room with no fuel | A few joules (a trigger), against the full heat (the valve) | Q4.1 |
| C89 | Tipping a poised boulder, and lifting it | Little, against its full potential energy | Q4.1 |
| C90 | Freezing a cup of water by sorting its molecules | Possible; costs bits, and the heat must go somewhere | Q4.1, Q0 |
| C91 | Cooling a room in a heat wave | The heat must be dumped elsewhere; it cannot be destroyed | Q4.1, A3 |
| C92 | A mage who knows where the gas line runs, against a stronger mage who does not | The knowing mage wins | Q4.1 |
| C93 | A spell notation refined over centuries, and its ancient form | The refined one works better | Q2.6 |
| C94 | An isolated people discovers magic independently | A very different interface that works as well for its users | Q2.6 |
| C95 | An interface built for human minds, used by a non-human mind | Works poorly; interfaces fit their users | Q2.6 |
| C96 | Coupled people in a culture with no magic tradition | Feel the vague sense and have occasional accidents, put down to ghosts or luck | Q2.6, Q2.4 |
| C97 | A lost technique and an old carving that still works | The technique is rediscovered from the carving | Q2.6, Q2.2 |
| C98 | A spell cast through a stone wall | Arrives late and weakened | Q4.2 |
| C99 | A complex spell and a simple one cast far | The complex one smears out; the simple one carries | Q4.2 |
| C100 | Striking a target the caster cannot see | Needs a link, a name or a trace | Q4.2, Q0 |
| C101 | The same effect by touch and at 30 m | Touch is efficient and nearly undetectable; at range it is lossy and visible | Q4.2 |
| C102 | A slow-magic pack | Bolts can be seen and dodged; a big one knocks its caster over | Q4.2 |
| C103 | An active scan of a house where people hide | Finds them, and reveals the scanner to any sensitive among them | Q4.3 |
| C104 | A careful mage crosses a guarded estate | Senses only passively and stays hidden | Q4.3 |
| C105 | Investigators at a crime scene weeks later | Read the faint trace a spell wrote into an iron gate | Q4.3, Q2.2 |
| C106 | A mage casts quietly to stay hidden | Reaches only a few metres | Q4.3 |
| C107 | A fugitive mage beside a ley line | The background drowns out their tells | Q4.3, Q3.3 |
| C108 | A spell on an iron lock in a wooden door | Rusts the lock through by driving the iron-oxygen reaction; the wood is untouched | Q5.1, Q3.1 |
| C109 | A strong pulse near copper wiring | The wiring sparks | Q5.1 |
| C110 | A compass beside a powerful casting | The needle twitches | Q5.1 |
| C111 | A mage makes light | Light without heat, at a limited efficiency | Q5.1, MS3 |
| C112 | Sensitives near a lightning strike | Feel a brief surge in the field | Q5.1, Q2.4 |
| C113 | Turning lead into gold | Lawful; needs an enormous store; the product is dangerously radioactive | Q5.2 |
| C114 | A swindler's magical "gold" | A gilded alloy, exposed by a density test | Q5.2 |
| C115 | Conjuring water from the air in a desert | Slow, limited by the humidity | Q5.2, Q4.1 |
| C116 | An exact copy of a person, and a rough copy of a pure crystal | The first is impossible (information); the second is easy | Q5.2, Q0 |
| C117 | An old alchemist who once truly transmuted metal | Died of a wasting sickness nobody understood | Q5.2 |
| C118 | Making a frightened horse obey a command | Calming works; the command does not | Q5.3 |
| C119 | The same entrainment on a trained, focused mind and an untrained one | The focused mind resists; the untrained one is overwhelmed | Q5.3, Q2.4 |
| C120 | Calming a willing patient and an unwilling one | The willing one costs a fraction of the energy | Q5.3 |
| C121 | A caster makes a flash appear in someone's vision | The target sees it and decides for themselves what it was | Q5.3 |
| C122 | A mage who always induces calm when present | Comes to be liked by conditioning, and loses it when found out | Q5.3, Q2.5 |
| C123 | A cap of conducting material | Blocks mind magic | Q5.3, Q2.2 |
| C124 | Planting a false memory in a stranger | Out of reach | Q5.3, Q0 |
| C125 | An intuitive healer speeds a small cut | It heals faster and still scars | Q5.4 |
| C126 | A skilled healer closes a deep cut using the skin around it as the template | Heals without a scar | Q5.4 |
| C127 | Regrowing a lost finger | Weeks, plenty of food, and the matching finger as the template | Q5.4 |
| C128 | A healer speeds a badly set break | It heals crooked, faster | Q5.4 |
| C129 | Forcing growth without a good pattern | A disordered, tumour-like lump | Q5.4 |
| C130 | A starving patient, healed by magic | Barely heals | Q5.4, Q4.1 |
| C131 | Raising the dead | Impossible unless a pack adds it | Q5.4 |
| C132 | A mage channels too much power through their own body | Collapses with heatstroke | Q6, Q4.1 |
| C133 | Keeping a forge hot by magic | Costs its full heat loss, continuously | Q6 |
| C134 | A weak mage uses a carved battery-and-trigger device | Releases far more power than their body could handle | Q6, Q2.2 |
| C135 | Two resonators set to feed each other | Grow until they drain the local field, then stop | Q6 |
| C29 | A thunderstorm passes | The local field rises for a while (MC5), and casting is easier during and after | Q1.2 |

## 5. Research to request (library)

- Q4.1: information thermodynamics (Maxwell's demon, Szilard, Landauer, Bennett; feedback control at the thermodynamic limit).
- Q7.2 and Q3.3: working memory limits and external notation (Cowan's about 4 chunks, extended cognition, the history of notations in mathematics and music).
- Q3.2 and Q3.3: how specific motor and action intentions are, and how the brain fills in the rest (motor control theory, predictive processing).
- Q3.4: arousal against precision, emotion regulation in children, and acting in sleep.
- Q2.1 and Q2.3: biological sensing and emitting organs as analogues (electroreception, electric organs, magnetoreception), and their energy costs.
- Done: the survey of about 50 magic principles (Library surveys/2026-11-23-magic-principles.md).
- SENT 2026-11-23: requests/2026-11-23-mark1-magic-science-foundations.md (F-a to F-i: EEG field generation and synchrony, meditation training, biological field organs, magnetic recording and Curie points, planetary dynamos, information thermodynamics, working memory and notation, intention specificity and arousal). It waits for the owner's go.

## 6. Lab prototypes (to test candidate laws numerically)

- P1 a toy field: a 2D grid of M with sources, leak and diffusion, and ray-marched pulses through a per-material index (C11, C14, C20).
- P2 a pattern as a valve: a pattern of k bits steering a flow, with a Landauer cost per bit (Q4.1).
- P3 completing an intent: a pattern with unspecified slots filled from a learned prior or from noise, measuring efficiency and wildness (C02, C03, Q3.3).

## 7. Revisits: richer answers to consider later (owner, 2026-11-23: "keep these in mind, we will consider them later")

- R1 (A6, Q0) THE PRICE OF IMPROBABILITY: a change costs its energy plus the order it creates (free energy, energy minus temperature times entropy), so improbable lawful outcomes cost more. This is folded into Q0 option (f), layer 2, if adopted.
- R2 (FM1, A2) EVERYTHING COUPLES BY ARRANGEMENT: a brain is a fine arrangement that can reshape itself from symbols, an engraving a coarse fixed one. Meaning matters only because a mind turns it into its own neural arrangement. One law replaces two kinds of transducer.
- R3 (Q1.2) THE FIELD COMES FROM DISSIPATION: wherever free energy is used up (sunlight absorbed, metabolism, storms, volcanic heat), a fixed small share becomes field. Every natural source falls out of one rule.
- R4 (A6, Q4.1) LUCK AS LAWFUL MAGIC: biasing which lawful outcome happens (dice, weather, a coin), at a minimum energy per bit of bias (Landauer). This is folded into Q0 (f) if adopted.
- R5 THE FIELD REMEMBERS: strong patterns leave faint imprints in the field or in absorbing matter (haunted places, echoes of rituals, reading a room's history).
- R6 (Q1.2, MC1) MINDS AND WORSHIP AS SOURCES: attention or collective belief produces field. Relevant to gods; needs a perpetual-motion guard.
- R7 (MS2) RICHER MATERIALS: direction-dependent indexes split a spell in two (birefringence); frequency-dependent indexes make prisms that sort spells by frequency.
- R8 (Q1.1) SELF-SUSTAINING PACKETS: a non-linear field holds stable lumps that persist (will-o'-the-wisps, standing spells).
- R9 (MS3) A PERSONAL ACCENT: signatures identify the caster; a skilled caster can forge another's.
- R10 (MS4) SYMPATHY FROM SHARED MODES: objects with similar mode sets are linked, a physical basis for "like affects like".

Also reframed by the owner (2026-11-23): **Q1.3 is not "matter stores the field".** Physics never stores a field in a material; particles carry their own charge or moment and emit their own field (magnets, charges, excited states). So Q1.3 becomes: WHAT CARRIES THE FIELD'S CHARGE, and what absorbing the field does to those carriers. This reopens Q1.1-Q1.3 at the depth of the revised Q0.

## 8. Foundational frameworks (candidates for the root; a first map, 2026-11-24)

| Framework | Derived from | What it would give magic |
|---|---|---|
| Least action (Lagrangian, Hamiltonian) | a variational principle; the classical limit of quantum mechanics | forces from energy; conservation via Noether |
| Effective field theory | the renormalisation group: large scales depend on small-scale details only through a few numbers | all allowed terms, unknowns as dials (a mass term gives magic a built-in range) |
| Gauge principle | demanding a symmetry hold locally forces a connecting field to exist | a candidate WHY for the field |
| Onsager, GENERIC | microscopic time-reversal symmetry plus the second law | dissipation and automatic cross-effects (a magical thermocouple) |
| Axiomatic thermodynamics (Caratheodory, Lieb-Yngvason) | the relation "which states can be reached from which"; entropy derived | an order on states = which changes are possible |
| Resource theories | free operations; everything else a resource; convertibility rules | what spells are possible with which resources (quantum computing has a resource theory literally called magic) |
| Constructor theory (Deutsch, Marletto) | laws as possible and impossible tasks; a constructor causes a change and keeps its ability | A1 almost verbatim: a pattern is a constructor that wears |
| Operational and reconstructed theories (Hardy; Chiribella, D'Ariano, Perinotti) | theories rebuilt from operational axioms | a principled in-world science of what can be measured |
| Process theories (Abramsky, Coecke) | physics as composition of processes, as diagrams | the spell grammar is already one |
| Port-Hamiltonian systems, bond graphs | energy-conserving interconnection through power ports | generic interfaces with replaceable insides, guaranteed to conserve |
| Maximum entropy (Jaynes) | statistical mechanics as inference under constraints | coarse twins |
| Digital physics (reversible cellular automata, lattice gases) | laws as local reversible update rules | exact conservation on the voxel grid |
| Free energy principle (Friston) | persisting systems minimise a bound on surprise | a candidate basis for intent and the mind's interface |
| Assembly theory (Cronin, Walker) | complexity as minimum steps to build | costs of copying and of patterns that make patterns |

## 9. Working hypothesis H1: one medium, many phases (first attempt, 2026-11-24; a draft to be stress-tested, not a decision)

The candidate media are not rivals if they are PHASES or DUAL DESCRIPTIONS of one medium. Real physics already has such a structure.

**H1.** The medium filling space is a compact U(1) lattice gauge theory coupled to charged matter (the medium's own defects, plus the carriers bound in ordinary matter), at a spacing far below the voxels. Lattice gauge theories of this kind have a known phase diagram (Creutz, Jacobs and Rebbi 1979; Fradkin and Shenker 1979), and each phase is a kind of magic region:
- the COULOMB PHASE: free charges, a long-range field, travelling waves. This is ORDINARY MAGIC (everything decided so far), with ley lines as the planet's field lines;
- the CONFINED PHASE: field lines bunch into strings with tension, so opposite charges can never be separated (as quarks cannot). Magic acts only through organised pairs and strings; ley lines are physical bundles of strings;
- the HIGGS (SCREENED) PHASE: the field is expelled, as a superconductor expels a magnetic field (the Meissner effect), and has only short range. This is a DEAD ZONE, and a ward could be a small screened region. Field can enter only as QUANTISED FLUX TUBES (Abrikosov vortices), so ley lines through a dead zone are thin, quantised threads.
- Fradkin and Shenker showed the Higgs and confined regions are smoothly connected, so regions can shade into each other.
- WATCH (library, 2026-11-24, to be confirmed): with CHARGE-1 matter the Higgs and confined regions are one connected phase, not two (Fradkin-Shenker complementarity); only CHARGE-2 or higher condensing matter gives a distinct (Z_q topological) screened phase with sharp boundaries, as in real superconductors, where Cooper pairs carry charge 2. So whether dead zones are SHARPLY BOUNDED phases or a smooth crossover from string regions depends on the charge of the medium's condensing matter: a candidate dial. R2.1 uses charge-1 matter; a charge-2 run is a natural R2.2.

The other media appear as dual descriptions or limits:
- the SUPERFLUID picture is the dual of the charge picture (vortex-charge duality), and the flux tubes of the screened phase are its vortices: magic as flow, ley lines as vortex filaments;
- the ELASTIC-SOLID picture is dual to a tensor gauge theory (fracton-elasticity duality, Pretko and Radzihovsky 2018): stress as force (Q5.5), dislocations as charges, places that work-harden;
- the RULE-FOLLOWING MAGNET (spin ice) is a classical Coulomb phase (Henley 2010): the classical macro model of RQ2;
- TOPOLOGICAL phases (edge states at boundaries; fractional charges) arise if the medium has topological order, the string-net extension (Levin and Wen 2005), which is also RQ2's quantum deeper theory.

What sets the local phase: the medium's local coupling and the density of charged matter, both of which energy, temperature and organised matter can change. Which way they go (for example, whether hot places are ordinary and cold, quiet depths are dead, or the reverse) is for the toy models to find, not to choose.

What H1 already explains: Q1.1 (c) stable field states (= phases); dead zones and wards (= screening); ley lines (field lines, strings or flux tubes by phase); R8 self-sustaining packets (vortices, flux tubes, solitons); R5 memory (slow dynamics near the transitions and in frustrated, glassy regimes); A2 (neutral matter cancels; confinement hides free charge); charge conservation (topological).

What H1 does not yet say: why the medium has these couplings (the brute fact moves one level down); which phase each place is in; how organised matter, minds included, binds and aligns carriers; the numbers.

**Testing H1:** a library survey of the phases, dualities and dynamics of these media (and anything H1 misses); toy simulations of a classical compact U(1) lattice gauge theory with matter and inertia (Monte Carlo for phases, Hamiltonian dynamics for waves), measuring phases, the force law, waves and flux tubes; then the target list R0 and the case bank.

## 10. R0: the target list (2026-11-24)

Every candidate medium, and H1 in particular, must reproduce these in its effective (macro) limit. Each line names the decision it comes from. A candidate that fails one is wrong or incomplete; a target that turns out to be impossible in every candidate is reopened explicitly.

| id | target | from |
|---|---|---|
| T1 | A conserved magic charge, exactly (by symmetry or topology); charges appear and vanish only in opposite pairs | Q1.1, A3, FQ1 |
| T2 | A long-range field in ordinary regions (Coulomb-like at a distance), unless a region's phase screens or confines it | Q1.1, Q1.2, H1 |
| T3 | Travelling waves with a phase (interference, coherence, resonance), fast enough to be light-like in air at casting frequencies, with speed set by medium and frequency | Q1.1, Q4.2, MS2 |
| T4 | Neutral, disordered bulk matter is silent: only organised moments of carriers couple | A2, Q1.1 |
| T5 | Material classes from how carriers respond: index, absorption, conduction and screening, remanence with a critical temperature, switching | MS2, Q2.2 |
| T6 | Exactly three portals to ordinary physics: binding (force on and energy shifts of matter), weak mixing with electromagnetism, energy exchange (pairs created by dissipation, recombination to heat) | Q5.1, RQ3 |
| T7 | Sources and large-scale structure: natural flows make field; a planetary, dynamo-like field with lines, poles, drift and rare reversals | Q1.2 |
| T8 | Fluctuation and dissipation from one friction: whatever absorbs magic emits magical noise at its temperature (the casting threshold) | Q3.3 |
| T9 | Static field stress as force at a distance (a grip like a magnet's), paid in setup energy and losses, not running power | Q5.5 (reopened) |
| T10 | Distinct, stable kinds of region (ordinary, dead or screened, others), with boundaries, that can change when pushed hard | Q1.1 (c), H1 |
| T11 | The zero-magic limit: with the couplings off, the engine's ordinary physics exactly | A8 |
| T12 | Energy and momentum conserved including the medium; entropy never decreases | A3, Q0, GENERIC |
| T13 | Only lawful end states; any catalysis pays by wear or correlation | A6 |
| T14 | An effective layer that runs on 5 cm voxels at 25 ms ticks (a quasi-static near-field solve plus ray-marched pulses, or an implicit solve) | S4.1 of the frameworks answer |
| T15 | Macro effects that appear classical, from a quantum deeper theory | RQ2 |
| T16 | Every case in the case bank, as an emergence test at the effective level | A7 |

Wanted but not required (to be discovered, not imposed): self-sustaining packets (R8), field memory (R5), ley lines as strings or quantised flux tubes, places that harden with use, mirror cross-effects (a magical thermocouple).
