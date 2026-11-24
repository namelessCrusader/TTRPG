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
- **A6. Magic only allows change that could exist without it** (owner, 2026-11-23, with Q0). Every end state is a lawful physical state that some ordinary process could reach, and every step balances (Q0). Magic changes WHICH lawful change happens, and where, when and how fast. It never adds to the set of lawful states. This is the catalyst principle: a catalyst opens a pathway and changes a rate, never the equilibrium. Teleport is a pack doorway because the end state is lawful but no ordinary path is instant; true creation of mass is ruled out.

## 2. Decisions so far

- MS1 (unfamiliar effects): practice raises efficiency, so unfamiliar patterns waste more heat. Skill has an intrinsic part and an explicit part (writing the pattern in detail buys specificity, a game of player skill). The layers run from the field, through interfaces and many notations, up to spells. The default history has gods giving people compiled thematic spells that are intuitive but opaque. (magic-design addendum)
- MS2: a magic index per material (Snell, partial reflection, waveguides, lenses). Density-dependent index later. How much material matters depends on the interface.
- MS3: signatures ride whichever channel carries the tell. The tell and waste go to any physical channel. Spells are mechanisms (A4).
- MS4: every object resonates by its real body. No object is a focus by type; nobody needs a staff.
- MS5: engravings multiplexed by wavelength. Crosstalk later.
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
  - Q3.5 Must the caster understand the physics?
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

## 4. The case bank (first 26; expectations are PROVISIONAL until the questions they test are settled)

| id | case | a sensible outcome (provisional) | tests |
|---|---|---|---|
| C01 | A frightened 6-year-old cornered by a dog | If the child couples at all: a big, crude, unaimed release (the curtains catch, or the dog is thrown back), then exhaustion | Q2.1, Q3.4 |
| C02 | An untrained adult thinks "fire" hard | Usually nothing; with some coupling, a weak, wasteful, wild flicker | Q3.2, Q3.3 |
| C03 | A trained mage thinks "fire" | Works, efficiently, for the practised case only | Q3.2, Q9.1 |
| C04 | A parrot recites a spell perfectly | Nothing (FM1 (b)) | Q2.1, Q8.4 |
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
| C18 | Calming a frightened horse | Possible with coupling to its mind; resisted by a strong state; consent matters for people | Q5.3 |
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
| C44 | A mage under deep anaesthesia | Cannot cast (synchrony is disrupted) | Q2.1 |
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
