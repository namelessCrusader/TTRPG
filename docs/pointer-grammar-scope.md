# Pointer grammar: design scope

How should a small model choose a structured action like
`take(the iron key, from: the chest)`? This document lists the open
questions, the options for each, and the experiment that settles each one.
It does not choose answers. Each question gets an experiment and a pass
mark, and the experiments choose.

It refines three existing commitments rather than replacing them:
`docs/data-engine-design.md` D9 ("bi-encoder over the candidate set"), its
factoring paragraph (autoregressive verb, then target, then params, with
pointer attention), and "grammar-constrained derivation" as the long-term
arc. Roadmap items 84, 86, 88, 89 and 98 hold the evidence.

## 0. What we already know

Measured in `lab/` (roadmap 98):

- **Flat option lists degrade with size.** Laya picks the right item from 4
  options 19 times in 20, from 16 options 11 in 20, and from 64 options 3 in
  20. This is not the token budget: raising the budget from 192 to 1,400
  tokens changed nothing. Laya's own embedding shortlist did worse.
- **Zero-shot, choosing "what to deal with next" is word matching.** Laya
  scores 33-42% and a content-word-overlap matcher 32-34%. No phrasing of
  the room made Laya beat the matcher on both single decisions and whole
  rooms. Zero-shot Laya escaped 0 of 30 rooms.
- **Reading works.** A player line's intent: 60% against 10% chance.
- **Direction is a skill only in isolation.** "The chest is 5 west" gives
  "step west" 8 of 8 times. In a room with 14 things, it does not surface.

Already in the code:

- **The sim factorizes by limb.** Legs, hands, mouth, eyes and waist are
  asked in turn, each told what the others chose ("already chosen"). That
  is parallel composition of actions (roadmap 83-84).
- **Every option carries its structure.** `pieces.py` stores `verb` and
  `args` on each option. The voxel sim's `ACTS` table has 20 verbs over the
  5 limbs.

So the grammar already exists as data. What is missing is the model that
derives within it.

## 0b. Revision 1 (2026-09-26): use a pretrained task, do not train a new one

Two objections from review changed the core proposal.

**"One latent per entity" is not free.** "The rusty key" or "a squat iron
strongbox" is several tokens, and what they mean in a scene lives in a
transformer's activations over the whole context, which cannot be
precomputed for every combination. There are three known ways to get one
latent object per entity:

| | how | what you get | the catch |
| --- | --- | --- | --- |
| a | a sentence-embedding model pools the phrase into one vector, trained for that | one cacheable vector per name | blind to context: the key does not know it is in the chest |
| b | run an encoder over the scene and take a marker or span activation per entity (GLiNER's spans, Laya's [MASK] before each option) | one contextual vector per entity | recomputed for every decision; fine at ~30 ms for 512 tokens |
| c | **store nothing**: a language model scores each entity's NAME, token by token, as the filler of a slot in context | the pointer score itself | one prefix pass plus a few tokens per candidate, with a prefix cache |

**A combiner trained from scratch has no zero-shot ability.** We cannot
pretrain, only fine-tune for format. So the scorer should be a model whose
**pretraining task already is** "fill typed slots from a legal set in
context". Two are:

- **Function calling with constrained decoding.** Small instruct LMs (Qwen3
  is trained for tool use) already emit `take(object=..., from=...)` from
  context, and already read histories written as tool call plus tool result.
  Constrain each argument to the sim's legal fillers, and each slot is a
  pointer whose scores are the LM's own next-token probabilities. The LM's
  attention is the combiner; the grammar is the constraint.
- **Masked slot filling**, which is exactly ModernBERT's pretraining
  ("Kael takes [MASK] from [MASK]"). Clean for one-token fillers, costlier
  for multi-token ones.

So the hand-designed scorer families of Q2 (a to e) become the fallback, used
only if a format-tuned pretrained model fails.

**Measured first (`lab/calib.py`).** One need, one right item, N-1
distractors ("Kael needs to cut the rope" → the knife). Right out of 20:

| options | chance | word matcher | Laya | Laya calibrated | Qwen3-0.6B scoring legal names | the same, calibrated |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 25% | 7 | **17** | 16 | 14 | 17 |
| 16 | 6% | 5 | 13 | **15** | 12 | 11 |
| 32 | 3% | 2 | 8 | 9 | **10** | 6 |
| 64 | 2% | 1 | 4 | 5 | **10** | **10** |
| 80 | 1% | 3 | 3 | 5 | **9** | 7 |

"Calibrated" subtracts each option's score with the need replaced by "do
something". Laya chose "the door" from a state reading "banana", which is
a prior over option strings, and this is the standard way to divide such a
prior out (Zhao et al. 2021; Holtzman et al. 2021).

- Laya is best on short lists. Calibration helps it a little at 64 and 80
  (from 4 and 3 to 5 and 5): its large-list failure is not only the prior.
- **A general 0.6B language model, scoring only the legal names, degrades
  far less:** 14/20 at 4 options, still 9-10/20 at 64 and 80, where Laya
  has fallen to 3-4. That is the zero-shot direction wanted: the pretrained
  task already does the thing, so fine-tuning only has to teach the format.
- Calibrating the LM did not help. Twenty cases per cell is small: read
  differences of 2-3 as noise.

**Data for format fine-tuning,** on the other machine:

1. Procedural generators with experts, like `lab/rooms.py` (every label
   derivable from the text shown).
2. The sim's own harvest: situations, menus and outcomes (`seam.py`).
3. A big-model teacher for language: paraphrases of player lines, and
   labels for social choices, checked by the sim.
4. The corpora already in the repo: `docs/corpus_review/dm_corpus.jsonl` and
   `docs/trajectories*`, which hold player lines paired with the programs
   the engine ran.

**Latency.** The latency budget allows model calls only for the player's
parse and the addressed NPC's reply. A constrained LM slot filler fits the
player-parse path directly. Per-tick NPC choices stay with reflexes, tables
and the kNN harvest; the LM can teach them offline.

### A generic verb inventory (draft, to settle)

About two dozen verbs, each with typed roles, grouped by the limb that does
them. The per-limb null act is always offered. Roles: **patient** (what is
acted on), **instrument** (with what), **source** (from where), **goal**
(to where, or at what), **recipient** (to whom), **topic** (about what),
plus **line** (the words said) and **direction**.

| limb | verbs (roles) | covers the sim's `ACTS` |
| --- | --- | --- |
| legs | go(goal), follow(patient), flee(source), jump(goal), climb(patient), hide(goal), *stay* | go, jump, roll, stay |
| hands | take(patient, source?), drop(patient), put(patient, goal), give(patient, recipient), open(patient, instrument?), close(patient), use(instrument, patient), hold(patient), let_go(patient), throw(patient, goal), strike(patient, instrument?), catch(patient), search(patient), read(patient), *keep* | hold, let_go, swing, throw, catch, catch_who, take, reach, pull_in, keep |
| mouth | say(line, recipient?), ask(recipient, topic), tell(recipient, topic), offer(recipient, patient, for?), threaten(recipient, for?), call(for), *quiet* | say, quiet |
| eyes | look(goal or direction), watch(patient), *about* | look, about |
| waist | crouch, lean(direction), *upright* | lean, upright, hold_pose |

A verb applies to whatever the sim says affords it. `open` is offered for
anything with a closed state, and `use(instrument, patient)` for any held
thing and anything within reach. So the list stays short while the
situations it covers grow with the world.

## 1. The questions

| | question | why it is open |
| --- | --- | --- |
| Q1 | **Representation.** What is embedded, by what model, and cached how? Embeddings or model outputs? | D9 says "encode each menu entry", but the menu changes every tick. |
| Q2 | **Combination.** How is a slot scored, given the verb and the slots already filled? | A dot product cannot condition on the verb, the role, or a pair. "Which key fits which lock" is a relation between two candidates. |
| Q3 | **Order.** Verb first, object first, or any order? | The player often names the object and leaves the verb implied. An NPC's attention picks the object first. |
| Q4 | **Memory.** How does what happened (what the character did, touched, was told, saw) get back in? | Text history in the state is costly, and it leaks answers in the form the model was trained to recognize (the user's point). |
| Q5 | **Structure.** Is `verb(role: filler, ...)` the right unit of choice at all? | Primitive steps failed. Goal-level choices may be what the model should make, with the sim doing the rest. |

### Q1. Representation

**Split every candidate into what it IS and what it is NOW.**

- **Static identity.** Name, kind and description ("the rusty key", "a squat
  iron strongbox"). Embedded once per distinct string and cached forever.
  This is where language knowledge enters: a model never trained on
  "strongbox" still places it near "chest".
- **Dynamic state.** Where it is relative to me, open, locked, held by whom,
  burning, what is known to be inside, and when I last touched it. **The sim
  computes these as numbers and tags, not text.** The sim knows them
  exactly, they change every tick, and putting them in text forces a fresh
  encoder pass for every entity on every tick.
- An entity's vector is `g(static_embedding, dynamic_features)`, where `g` is
  a small learned network. The situation vector is built the same way: the
  character's own state and drives as features, plus a cached embedding of
  its goal text.

The rule: **text is for identity, and the sim's numbers are for state.**
That makes precomputing possible. Only new strings ever need an encoder
pass.

**Embeddings or model outputs?** Both, each for its own job:

| representation | used for | example |
| --- | --- | --- |
| frozen encoder hidden states | what the learned head combines | the embedding of "the strongbox" |
| a decision model's typed outputs | features, only where language must be read | Laya's P(threat) for a player's line; the type of a newly described object |
| a small LM's generated text | speech only | the NPC's line |

Anything the sim already knows should never be asked of a model. Outputs
are low-dimensional and already decided, so they can gate or annotate but
cannot be composed. Embeddings can be composed.

Candidate static encoders: Laya's ModernBERT-large encoder (395M); a small
sentence-embedding model (tens of millions of parameters, trained for
similarity); a small decoder's hidden states (Qwen3-0.6B). CLM is the
precedent for the pattern: a frozen 8B backbone plus a ~20M trained head.

**E1 (probes).** Cache the static embeddings of every entity string in both
name pools. Fit linear probes on the TRAIN pool for the facts the grammar
needs: kind (container, door, item, person), "could open a lock", "could
hold a key", "can be carried". Score on the TEST pool.
*Pass:* 90% or better on unseen names for each fact. Cheap; runs on the
laptop.

### Q2. Combination: the "grammar of combining"

The factored policy:

    P(action | S) = P(verb | S) · Π_k P(slot_k = e | S, verb, slots filled so far)

Each slot's candidates are the legal fillers the sim offers for that role.
For `use`, the instrument must be something held and the target something
adjacent that can be opened. Legality is enforced by construction, as with
the menus.

The families of scoring functions, from cheapest to most expressive:

| | scorer | can express | precedent |
| --- | --- | --- | --- |
| a | dot: `q · e` | relevance to the situation only | D9 as written; DRRN for text games |
| b | role-bilinear: `(W_role q_k) · e`, where `q_k` is updated after each choice from (verb, role, chosen entity) | conditioning on the verb and on earlier choices | AlphaStar's autoregressive action heads; KG-A2C |
| c | plus a pairwise term `r(e, e_j, pair features)` | compatibility between two candidates: key and lock, note and box, container and contents | relation networks |
| d | a small transformer over [situation, verb, filled slots with role embeddings, all candidates], scoring each candidate | candidates seeing each other and the filled slots (listwise, as Laya is) but over vectors, so 1,000 candidates are 1,000 positions, not 1,000 squeezed strings | Set Transformer; pointer networks |
| e | one small module per **role** (instrument, patient, source, goal), shared across verbs | whether "instrument" learned on use(key, door) carries over to cut(knife, rope) | Neural Module Networks |

The grammar is **three things**: the role inventory, which roles each verb
takes, and a learned combiner. The first two are declared, and the sim
already has them. Only the combiner is learned.

Pair features (c) have to come from the sim: `fits(key, lock)`,
`names(note, box)`, `inside(x, y)`, `owns(person, thing)`. **If the sim does
not expose a relation, no combiner can learn it.** Auditing which pair
features exist is part of this work.

**E2 (one slot, large lists).** Target selection with 5, 10, 30, 100, 300
and 1,000 candidates, in-distribution and with unseen names. Compare the
constrained LM slot (0b), zero-shot and after format fine-tuning, against
flat Laya (raw and calibrated) and the word matcher. Scorer families a, b and
d run only if the LM falls short. First numbers are in 0b.
*Pass:* accuracy at 1,000 is at least 90% of accuracy at 10, and at least 20
points above the word matcher.

**E3 (several slots, compositional).** A SCAN/gSCAN-style split: train on
verb-object and role-kind combinations with some held out; test on the
held-out combinations and on unseen names. Compare b, c, d and e.
*Pass:* held-out-combination accuracy at least 80% of in-distribution.

### Q3. Order

| order | fits |
| --- | --- |
| verb first | how language puts it; AlphaStar |
| object first | attention and salience; how the sim builds menus (an option exists because an object affords it) |
| any order: a masked-slot model predicts any missing slot from the others, most confident first | both jobs with one model: parsing (the player fixed some slots) and deciding (the NPC fixes all of them) |

**E4.** The same data under all three decoders. Measure accuracy, and
robustness when the player's line fixes some of the slots.
*Pass:* take the simplest decoder within 2 points of the best.

### Q4. Memory

| | how the past gets in | cost | leak risk |
| --- | --- | --- | --- |
| a | text history in the state ("A moment ago: ...") | tokens every tick | high: it states what to do in the form the model was trained on |
| b | **sim-derived per-entity features**: tried to open it and it held, what is known to be inside, who said what about it, when last touched | a few numbers per entity | low: facts about the thing, not advice |
| c | **event embeddings**: each event is a filled frame (verb, roles, outcome, time) encoded by the SAME grammar, and attached to the entities it involves; the slot scorer attends over a candidate's recent events | one small vector per event | low |
| d | a recurrent hidden state | opaque | unknown, and hard to debug |

(c) has a property worth testing on purpose. **An action and an observation
share one representation.** What the model outputs, a filled frame, is what
it later reads back as memory. "I took the key from the chest" and "the
fisherman took the key from the chest" are the same object with a
different agent.

Precedents: KG-A2C builds a knowledge graph from observations and uses it
to constrain and inform object choice. Recurrent Entity Networks keep a
memory slot per entity.

**E5 (tasks that need the past).** The key's place was told 20 ticks ago;
the door was tried and held; a person lied earlier. Compare a, b and c. The
word matcher gets the same memory, as the leak control.
*Pass:* history-dependent decisions at in-distribution accuracy, with at
most k events per entity, and the word matcher does not match it.

### Q5. Structure: what is the unit of choice?

| | structure | example | measured |
| --- | --- | --- | --- |
| S1 | primitive motor | step west; take the key | fails (roadmap 98); belongs in the motor layer |
| S2 | predicate-argument frame with roles | take(patient: key, source: chest) | matches language and `pieces.py` |
| S3 | object first, then what it affords | the chest → open | matches how the menus are generated |
| S4 | goal as a state change | key: in the chest → in hand; door: locked → unlocked | the sim's planner and motor competence turn it into steps; the model's choice is purely semantic |
| S5 | hierarchy: goal → frame → motor | `Plan := Step+`, each step a derivation | the "go to them first, then hold" hole (items 81, 89) |

The principle to test: **the model decides at the level the sim cannot
decide for itself.** The sim knows how: paths, reach, legality, physics.
The model decides what and what for: which goal, which target. Primitive
steps failed because they asked the model to do the sim's job.

**E6 (closed loop).** The randomized rooms, then the voxel harvest scenes.
Same expert, same data budget, action spaces S1 to S5. Measure escape rate,
milestones, steps, and learning curves at 1k, 10k and 100k decisions, on
unseen names and held-out compositions.
*Pass:* the structure with the best held-out success at equal data. On a
tie, the one needing fewer model decisions per episode (latency budget).

## 2. Testbed

**Tasks.**

- **T1: grounding a player's line** into a frame over the entities present.
  Templated, then paraphrased (paraphrasing by a large model, on the other
  machine), with pronouns, indirect references and near-miss distractors,
  so word overlap does not solve it.
- **T2: an NPC's choice**, from the voxel sim's own situations and menus
  (the `seam.py` spelling) and from the rooms.
- **T3: closed loop**: rooms, then the grid, then voxel scenes.

**Splits.** In-distribution; unseen names (TRAIN and TEST pools share
nothing, as in `lab/rooms.py`); held-out compositions (verb-object,
role-kind); length (lists up to 1,000, plans longer than any in training);
unseen room types.

**Controls in every table.** Random; the word matcher (the leak check); the
expert (the ceiling); zero-shot Laya and Open-Jev as reference points.

## 3. Compute

- **Embeddings are computed once**, as inference. Heads train on the cached
  vectors in seconds to minutes. This runs on the laptop, one job at a time,
  under the heat watchdog.
- **Anything that fine-tunes an encoder, or runs the GPU for more than about
  10 minutes, goes to the other machine** as a packaged kit: generators, the
  expert, caches, training and evaluation scripts, and a README with exact
  commands.
- **Inference target.** One frozen-encoder pass per new string, and per
  tick for the situation. Head cost well under 1 ms per slot for 1,000
  candidates.

## 4. Order of work

Each stage has its pass mark. A failed gate stops that line and is reported,
not tuned around.

1. **E1**: representation probes → choose the encoder(s).
2. **E2**: one slot at scale → choose the combiner family.
3. **E3 and E4**: frames and order → the grammar.
4. **E5**: memory.
5. **E6**: structure, in closed loop.
6. **Wire into the sim**: `pieces.py` verbs and args, the `seam.py`
   spelling, the harvest.

## 5. Risks

- **Relations the sim does not expose** cannot be learned. Audit the pair
  features before E3.
- **The teacher is the ceiling.** The room expert is one policy. For social
  choices (T2), labels must come from the sim's outcomes or from a
  big-model teacher (roadmap 77, 88).
- **Exposure bias** in slot-by-slot decoding. Train on noisy prefixes, as
  the noise in `lab/rooms.py` already does.
- **Templated language overfits.** Paraphrase on the other machine.
- **Leakage through features.** "You know the key is in it" on an option is
  a fact, and it is also nearly the answer. The word matcher on every table
  is what catches this.

## 6. Decisions needed

1. **Which task comes first, T1 or T2?** Recommendation: T1, grounding the
   player's line. It is the model call the latency budget keeps, and its
   answer is in the text, so it is a fair first test.
2. **What will the other machine take?** Heads only, or encoder fine-tuning
   too?
3. **Where to design the grammar.** Take the voxel sim's `ACTS` and
   `pieces.py` as the verb inventory now, or design it on the rooms first and
   port it?

## References

Pointer Networks (Vinyals, Fortunato and Jaitly, 2015). Set Transformer
(Lee et al., 2019). Neural Module Networks (Andreas et al., 2016). SCAN
(Lake and Baroni, 2018). gSCAN (Ruis et al., 2020). DRRN (He et al., 2016).
Jericho (Hausknecht et al., 2020). KG-A2C (Ammanabrolu and Hausknecht, 2020).
Wolpertinger, for large discrete action spaces (Dulac-Arnold et al., 2015).
AlphaStar's autoregressive action heads with pointer selection (Vinyals et
al., 2019). SayCan (Ahn et al., 2022). Recurrent Entity Networks (Henaff et
al., 2017). FrameNet and PropBank for semantic roles. CLM
(github.com/Contrastive-LM/CLM). Open-Jev
(github.com/Zefan-Cai/Open-Jev). Laya
(huggingface.co/convaiinnovations/laya).
