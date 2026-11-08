# Making personalities more pronounced: the design

This combines the two reviews with our own results:

- `docs/persona-steering-review.md`: persona vectors, steering, trained
  conditioning (94 sources).
- `docs/mech-interp-review.md`: probes, patching, sparse autoencoders, bias
  fixes (111 sources).

The reviews hold the evidence and the references. This page holds the
decisions and their order. Written 2026-09-29.

## What we know

- **Personality as text in the input does almost nothing** to a small
  listwise chooser. GLiNER2.5-Decide: 0.3% of choices surprising and in
  character, and the types came out backwards (roadmap 102). The review
  agrees: small encoders read instructions as keywords (FollowIR).
- **A trait bias added to each option's score works:** 10.4%, 3 times the
  rule policy. The review names it as a known, sound pattern: adding scores in
  log space, as in DExperts and classifier-free guidance. Its weak point is
  that the option features and the weights are written by hand.
- **The model under the bias has poor sense.** It spends a third of its
  choices taking and dropping things, and it fights about 2.5 times as often as
  the rules. A pronounced personality on top of poor sense is still noise. So
  sense and personality are two separate layers, and both must improve.
- **"Steering is weak below 2B" is untested for our use,** not disproven.
  Steering moves models down to 124M. No study measured personality steering
  of forced-choice option scores below 2B. Where steering was measured, it was
  most reliable on forced choice.
- **Traits bleed into each other,** and the effect of strength can turn over at
  high strength. What a model says and what it chooses can come apart. So we
  measure choices only, as we already do.

## The design: three layers

```
final score of an option = sense + personality + habit

sense        = the chooser's score with NO persona       (a model, or cached)
personality  = γ · Σ weight(trait, feature) · trait_now · feature(option)
habit        = the repeat penalty                        (already in world2)

trait_now    = the character's trait, drawn again at each choice
               (a trait is a distribution; already in world2)
γ            = the strength, drawn per scene from the character's range
Only options whose sense score is near the best (within δ) can be moved by
personality. This is the plausibility guard, and it keeps a trait from
turning nonsense into a choice.
```

Why this shape:

1. **It costs no model call for personality.** Most characters get zero model
   calls on most turns (the speed rule). A bias outside the model works even
   when the sense score comes from a cache or from rules.
2. **It works with every chooser.** Encoders, cross-encoders and decoders all
   give a sense score. Only the decoders can also be steered from the inside,
   and that comes later (below).
3. **Each part can be learned separately.** The features, the weights and γ
   can each be learned, and each can be tested one factor at a time.

### Where the features come from (in place of hand tags)

- **NLI trait entailment,** cached for each option in context. The premise is
  the action in its scene ("Tom strikes the bandit who holds the child"). The
  hypotheses are "This is a brave act.", "This is a greedy act." and so on.
  openjev computes them offline or at load time, not every turn.
- **Projection on a trait axis** in an embedding space, as a cheaper second
  source.
- **The world's own tags** (violent, helps, takes, flees), as now.
- **Check before use:** the features must agree with the large model's trait
  labels on a sample. Watch for word overlap (roadmap 102) and for the wrong
  person ("strike the child" is not brave).

### Where the weights come from

- **Now:** hand weights (the rules policy's trait part).
- **Next:** fitted to large-model labels of what each character type would do
  in a decision snapshot, then checked on held-out scenes.
- **Later:** refined from your ratings in the rating rounds (Workstream C).

### Inside the model (decoders first, only when the model runs anyway)

1. **Guidance on option scores.** score = s0 + γ(s+ − s−), where s0 has no
   persona, s+ the persona and s− its opposite. It uses the same guard. Test
   the sign first: does s+ − s− rank brave options above cowardly ones?
2. **A steering vector for each trait** (persona-vector style): the mean
   difference of middle-layer activations between "brave" and "cowardly"
   versions of our own scenes, added with strength α drawn per scene. Check
   the signal-to-noise of the pairs first. Qwen3-0.6B first.
3. **Trained, off-laptop:** BiPO or LoReFT on option preferences for Qwen; a
   learned prefix, or FiLM on the option embeddings, for the encoders.

## The sense layer: what the reviews point to

These come from the failure analysis, the ablation and the interpretability
review. They are listed here because personality cannot fix poor sense.

- **The scene description:** filter to what bears on the character's drives,
  and give recent events as short lines (`docs/presentation-plan.md`).
- **Taking back what you just did** ("drop" after "take") counts as a repeat.
- **Item priors:** a per-option bias table, estimated from blank-need scores.
  It costs nothing during play. Calibration from the kept scores is checked
  offline first.
- **Affordance knowledge** (the main gap). First a quiz and probes, to find
  whether the model knows "a sword cuts" but does not use it, or does not know
  it. Then either training pairs of tool and use (does not know), or an
  inference step from capability to thing (knows, does not use).
- **Laya's list-length collapse** is not a cut-off artifact (nothing is cut up
  to 32 options, where it has already fallen from 70% to 18%). Test: block
  attention between options, as bev-decider does by design.

## The shared prerequisite: a bank of decision snapshots

Both the inn formats (`docs/presentation-plan.md`, step 2) and the personality
measures need the same thing: about 150 frozen choice points from the inn,
with the options, the scene, and an acceptable set labelled offline. Build it
first.

## How we measure

- **More pronounced:**
  - *Dose-response slope:* move one trait from 0 to 1, hold the others at 0.5,
    and measure the change in how much of the draw goes to options that express
    the trait.
  - *Type separation:* how different two types' choices are on the same
    snapshots.
  - *The plan's type ranges* (a knight brave in more than 70% of threats).
  - *Layer 3:* the "surprising and in character" rate.
- **Without losing sense:**
  - the acceptable-option rate on snapshots and on the Workstream A cases;
  - the layer 3 "odd" rate;
  - the layer 2 nonsense rate at matched variety.
- **Control:**
  - the share of snapshots that move the wrong way;
  - bleed between traits (a 5 × 5 table);
  - whether each character's choices over many nights match its target mean
    and spread;
  - cost: calls, milliseconds, memory, temperature.
- **Pass rule (proposed, for you to set):** at least double the bias policy's
  slope, or beat 10.4% on layer 3, while losing no more than 3 points of
  acceptable options and adding no more than 2 points of odd choices.

## Order of work

| step | what | GPU | needs |
| --- | --- | --- | --- |
| 0 | the snapshot bank, labelled offline | no | the inn logs (already run) |
| 1 | baselines on the snapshots: random, rules, text, bias; the sign of the persona effect for each model | small | 0 |
| 2 | sense fixes: the filtered description, the undo-as-repeat rule, the prior table | small | 0 |
| 3 | automatic features (NLI trait entailment, embedding projection) in place of hand tags | yes | 0 |
| 4 | learned weights, and γ drawn per scene, with the guard | no | 0, 3 |
| 5 | guidance on option scores (Qwen, openjev) | yes | 1 |
| 6 | a steering vector per trait, Qwen3-0.6B, then 1.7B | yes | 1 |
| 7 | affordance probes and activation patching on the failure pairs | yes | none |
| 8 | trained conditioning (off-laptop) | other machine | 3, 4 |

Stop a line early if step 1 shows a model's persona sign is wrong, or if the
pair signal is low for a trait (skip steering for it).
