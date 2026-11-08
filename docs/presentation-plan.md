# How to present the world to a model: the next three steps

Notes for after the presentation battery on "things" (`lab/methods/present.py`,
roadmap 102). Each step lists the options, the best guess, how to test it, and
what result would change the plan.

What we know when this was written (2026-09-28):

- The models fail mostly on **knowing what things can do**. The verb alone
  does as well as the whole need, and removing the prior helps little (the
  ablation, first two models).
- Wrong picks are things that **belong to the same activity** as the need (the
  rope, the lamp oil, the padlock), and each model has **favourite items**.
- **Unrelated scene text** pulls some models (GLiNER-Decide, bev-decider, Von).
- **Long options that all start with the same words** confuse the listwise
  models. Writing each option as the whole action fell from 55-62% to 33-37%.
- **A short list, then a careful choice**, helps on long lists.

## 1. People and places: how to show them

### People

The "people" kind asks for an action aimed at one of the people present. The
new failure it can show is **the right action at the wrong person** ("pull the
bandit to safety"). The first model's numbers suggest that this happens about
twice as often as chance.

**Ways to show people** (one change at a time, from "name only"):

| name | what the model sees |
| --- | --- |
| name only (now) | "the bandit", "the child" |
| state | "the bandit (armed, shouting)", "the child (crying, held by the bandit)" |
| relation to me | "the child (a stranger's child)", "the guard (keeps the peace here)" |
| card | one line per person, fixed fields: doing / holding / state / relation to me |
| triples | "the bandit holds the child. the guard watches the square." |
| descriptor in the option | "pull the child, whom the bandit holds, to safety" |
| target first | "the bandit: hit" instead of "hit the bandit" |

**Ways to choose:**

- flat list (now)
- **whom first, then what:** two steps, as in the pick-list, where the player
  selects a target and then an action
- **what first, then whom**

**Best guess.** Two things will cut the wrong-person errors most:

1. **Cards with state and relation.** They give each person a short
   description the model can bind to.
2. **Choosing whom first.** Each question becomes small and clean: "who needs
   attention?" and then "what do I do to them?"

Descriptors inside the options will help the one-at-a-time models (openjev,
Qwen), which see only one option at a time and so need the target described
in it. For the listwise models, long options will hurt again, as the whole
action did.

**Leak control.** A descriptor must be a fact the world would show (armed,
crying, holding the child), never a label ("the threat", "the victim"). Run
the word-overlap baseline on every format. Also run a **descriptors only**
check: the people are described but the key sentence is removed. If that
alone finds the answer, the descriptors give it away.

**What would change the plan.** If "whom first" does not help, binding is not
the problem. Then the model does not know which action fits which person, and
that is a training target: pairs of the same action at the right and the
wrong person.

### Places

**Ways to show places:**

- name only (now)
- **features:** "the well (water, in the square, stone)"
- **distance:** "two rooms away", "across the river"
- **area groups:** inside the inn / the village / outside
- **what I believe is there:** "you think the cellar is empty"

**Best guess.**

- **Features** help, as the tags did for things. They must be physical
  features ("water", "high up", "roofed"), not purposes ("good for hiding"),
  or they leak.
- **Distance** changes which place is best. That is a trade-off, not a label,
  so it belongs in the world test (section 2), not here.
- **Area groups** help long lists the way "kind first" should for things.

## 2. The inn: prose against entity cards

The inn has no labels: there is no single correct action. So first make it
testable with the same harness as Workstream A:

1. **Decision snapshots.** Freeze about 150 choice points from the inn logs,
   using the rules and the model runs, with a mix of calm, theft, fight and
   bandit moments.
2. **An acceptable set for each snapshot.** Label each option good, possible
   or bad. The large model does this offline and you audit a sample, as with
   the other labels.
3. Then run every format on the same snapshots, one change at a time, with
   the overlap baseline.

**Ways to describe the scene** (from `World.describe`, prose, now):

| name | what the model sees |
| --- | --- |
| prose (now) | "It is night at the village inn. Tom is a peasant, in the kitchen. Here: ..." |
| entity cards | a block per person and thing: where, doing, holding, state, relation to me, what I believe about them |
| belief record | the character's beliefs as JSON |
| recent events as lines | "Fenn took the purse from Oren. Oren shouted." (one line each, newest last) |
| summary | "Oren is angry at Fenn because Fenn stole his purse." |
| filtered | only what bears on the character's drives now (hunger, anger, fear, greed); the rest is left out |
| order | the self and goals first, the scene first, or the question last |

**Ways to show the options** (the inn has 25-40 options, most aimed at a
person):

- flat (now)
- grouped by target
- whom first, then what (as in section 1)
- short list first (the pipeline that won on long lists)

**Best guess.**

- **Filtering beats every layout for the encoders** (Laya, GLiNER-Decide,
  bev-decider). Unrelated text halves their accuracy, and Laya has a
  512-token limit.
- **Events as short lines beat a prose summary** for who did what to whom.
- **Entity cards help the listwise encoders.** The decoders (Qwen) do as well
  with prose.
- **Grouping by target plus whom first** matters more than the scene format.
  This is the same binding problem as in the people kind.

**Also measure in the live inn.** Run the best two formats and prose through
the model policy, then compare the layer 3 numbers (surprising and in
character against odd), the story patterns, and the type behaviour.

**What would change the plan.** If the formats score the same on snapshots,
the model is not reading the scene. Then the scene should go in as data:
latents and a bias from the world, as in section 3.

## 3. Not text: embeddings and slot latents (training track)

This needs training, so it runs on your machine, not the laptop. Only format
fine-tuning is allowed, and something must work zero-shot first.

**The idea (the pointer grammar, roadmap 99):**

- **One vector for each entity,** made from its card by an encoder and kept
  between turns. It changes only when the entity changes.
- **One vector for the need or drive,** and one for each verb.
- **A small head scores an action slot by slot.** It picks the verb, then the
  target, then the instrument. Each slot points at one entity vector.
- **The head is trained** on the static cases, the inn snapshots (labels from
  the large model, offline) and your ratings.

**Zero-shot steps we can take now, with text only:**

1. **Factorized scoring.** Score the verb list and the target list separately
   with the existing models, and add the two scores.
   - *Best guess:* it reduces wrong-person errors, because each list is short.
     But it fails where the verb and the target interact: "hit the bandit" is
     good and "hit the child" is bad. The fraction of cases where it fails
     measures how much of the job needs a learned combination.
2. **Cached entity readings.** Encode each entity's card once, and each option
   as verb + cached target. This is "read apart" with a better text for each
   entity.
   - *Best guess:* still weak (bge was 26-60% on things), but it sets the
     floor the trained head must beat.
3. **The capability hint as a first step.** Have Qwen write the capability a
   need asks for ("something sharp"), then match options to it.
   - Only worth it if the oracle hint in the battery helps a lot.

**Training, in order:**

1. **Format fine-tuning on text.** Use the formats that win in sections 1-2,
   with hard negatives from the failure analysis:
   - things from the same activity (the rope for "cut the rope");
   - the model's own favourites;
   - the right action at the wrong person.
2. **Tools and what they do.** Short pairs of need and tool, including the
   unusual-but-workable ones from the creativity labels. This goes at the gap
   the ablation found.
3. **Only then, the slot head over entity vectors.** Compare it with the
   fine-tuned text model on the same snapshots.

**Personality, in the same track.** The trait bias (your idea: the model's
score plus a weight on each option from the traits) is the first thing to
learn. The weights for each trait and option feature can be fitted to your
ratings in the rating rounds. Learned prefix vectors for traits come after,
because the review found steering weak below about 2B.
