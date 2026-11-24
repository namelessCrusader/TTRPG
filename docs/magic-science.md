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
- **A6. Magic only allows change that could exist without it** (owner, 2026-11-23, with Q0). Every end state is a lawful physical state that some ordinary process could reach, and every step balances (Q0). Magic changes WHICH lawful change happens, and where, when and how fast. It never adds to the set of lawful states. This is the catalyst principle: a catalyst opens a pathway and changes a rate, never the equilibrium. Teleport is a pack doorway because the end state is lawful but no ordinary path is instant; true creation of mass is ruled out.

## 2. Decisions so far

- MS1 (unfamiliar effects): practice raises efficiency, so unfamiliar patterns waste more heat. Skill has an intrinsic part and an explicit part (writing the pattern in detail buys specificity, a game of player skill). The layers run from the field, through interfaces and many notations, up to spells. The default history has gods giving people compiled thematic spells that are intuitive but opaque. (magic-design addendum)
- MS2: a magic index per material (Snell, partial reflection, waveguides, lenses). Density-dependent index later. How much material matters depends on the interface.
- MS3: signatures ride whichever channel carries the tell. The tell and waste go to any physical channel. Spells are mechanisms (A4).
- MS4: every object resonates by its real body. No object is a focus by type; nobody needs a staff.
- MS5: engravings multiplexed by wavelength. Crosstalk later.
- Q0 (the frame), REVISED 2026-11-23: **(f), FIVE LAYERS**, "more complete for now" (owner). This supersedes the first answer, a conservation checklist, which is now layer 2's default setting.
  1. SYMMETRIES decide WHAT IS CONSERVED (Noether): every symmetry is kept by default, so energy, momentum, angular momentum, centre of mass, charge and particle number are all conserved. A pack may break a NAMED symmetry in a NAMED place (for example, no momentum conservation inside a ley line, so levitation needs no reaction partner there, and only there).
  2. A THERMODYNAMIC BUDGET decides WHETHER IT IS AFFORDABLE, with information as fuel (the generalised second law, Sagawa and Ueda): free energy available plus temperature times the information held must cover the free-energy change the outcome needs, plus losses. Improbable outcomes pay for the order they create (A6 made exact, R1). Luck has a price per bit of bias (R4).
  3. CAUSAL REACH decides WHETHER IT CAN GET THERE: a signal path from the pattern to the target that carries enough information in time (channel capacity within c_M; the ray march and the channel table).
  4. COMPUTATION decides WHAT THE THINKING COSTS: bits held, operations, and Landauer's minimum per erased bit. Searching and divination pay the bits needed to single out the answer; complexity is a real limit (a long chaotic forecast is out of reach at any power).
  5. THE ONE CHANGE is HOW IT HAPPENS: the mechanism the simulation runs.
  - The owner's A6 ("it can only allow change that can exist without it in some way") stands, and follows from layers 1-2. The pattern is a catalyst that wears.
- Q1.1 (what kind of quantity M is), 2026-11-23: **(b), ENERGY WITH A PHASE.** Each voxel holds an amplitude and a phase; energy is the amplitude squared. Interference, standing waves, coherence and resonance are native: a counterspell is a wave in opposite phase, and casters working in step add up. The density still spreads, leaks and refills. Several components are a pack option; a no-store pure-control field is a low-magic pack (it would reverse the sources decision).
- Q1.2 (where M comes from), 2026-11-23: **(d).** (a) NATURAL FLOWS make it by default: sunlight, life's metabolism, the planet's inner heat and storms each turn a tiny share of their own energy into M, booked as taken from them. It stays out of equilibrium because it is pumped where those flows are strong and leaks to heat everywhere, so MC5's cycles follow. Plus (b) ANCIENT CONCENTRATIONS as places and materials (nodes, crystals of fossil M) that can be mined and run out. Plus (c) INFLOW from other worlds or beings as a pack, booked as an import. The owner: it makes magic "a finite resource more than just a fixed number": rich places, mineable deposits, regional over-use.
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
| C29 | A thunderstorm passes | The local field rises for a while (MC5), and casting is easier during and after | Q1.2 |

## 5. Research to request (library)

- Q4.1: information thermodynamics (Maxwell's demon, Szilard, Landauer, Bennett; feedback control at the thermodynamic limit).
- Q7.2 and Q3.3: working memory limits and external notation (Cowan's about 4 chunks, extended cognition, the history of notations in mathematics and music).
- Q3.2 and Q3.3: how specific motor and action intentions are, and how the brain fills in the rest (motor control theory, predictive processing).
- Q3.4: arousal against precision, emotion regulation in children, and acting in sleep.
- Q2.1 and Q2.3: biological sensing and emitting organs as analogues (electroreception, electric organs, magnetoreception), and their energy costs.
- Done: the survey of about 50 magic principles (Library surveys/2026-11-23-magic-principles.md).

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
