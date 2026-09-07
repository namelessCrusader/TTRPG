# Trained local DM — direction & decision record

Captured from a grilling session (2026-08). This is a **direction**, not a spec, and
every expensive piece in it is **gated behind a measurement** — nothing here is
greenlit to build until the gate before it passes. Grill this doc; don't trust it.

Companion docs: [SMALL_MODEL_HARNESS](SMALL_MODEL_HARNESS.md),
[SMALL_MODEL_COMPETENCE](SMALL_MODEL_COMPETENCE.md), [OPEN_DESIGNS](OPEN_DESIGNS.md).

## The reframe

Everything to date makes a *frozen* small model competent by **selection** (kNN over
its own scored traces — measured working, +5% lift). The new direction is different
in kind: a **trained, self-contained, small (<500M) local DM** — no big-model
dependency at runtime. Big models (Opus/Sonnet/Haiku) become the *offline* engine
that generates training data by playing the real sim; the sim stays the authority;
the small model is the artifact.

## Decisions (the tree we walked)

1. **The DM authors, it does not adjudicate.** Adjudication stays deterministic in
   the sim (`apply()` + reactions). An LM-DM that decided outcomes would un-win the
   whole project. What the DM adds is *content/situations*, which the sim then runs.

2. **Artifact = a trained small model, not an off-the-shelf selector.** Off-the-shelf
   caps out at constrained selection; training is what buys authoring range.

3. **Replace, not complement — but big models generate the data.** End-state is a
   self-contained local DM. Runtime has no big model. Big models are used *offline*
   to author scenarios and to play the real sim producing grounded trajectories.

4. **Data is GROUNDED, through a WIDER interface.** Every training trajectory runs
   through the real sim (so it transfers). The current narrow interface was a
   concession to small-model fragility; big models don't need the training wheels,
   so the interface widens — but see constraint (7).

5. **Validity ≠ competence.** A CFG/grammar guarantees *valid* output, never *good*
   output. The measured failure (100%-"walk" collapse in `corpus_eval`) was a
   competence failure among perfectly valid actions. Grammar can't fix it.

6. **The keystone: upgrade the reward signal.** Today "blessed = committed
   coherently" (reached `apply_all`, not refused). Walk almost always commits, so
   the signal over-credits walk and *training on it amplifies the collapse*. The fix
   is **"blessed = advanced the world"** — distance-to-goal dropped, a meaningful
   state change, a clock toward resolution. The sim already holds these ingredients.
   This single change filters the training data AND gives any RL a real reward.

7. **"Open interface" = a finite, compositional, GROUNDED vocabulary — not free
   text.** Three constraints must hold at once:
   - *grounded* → resolves through the effect algebra (`apply()`), or it won't transfer;
   - *open* → enough range that big models don't produce walk-heavy data;
   - *compositional* → creativity is *novel combinations of a finite vocabulary*
     (like language), so a small model learns the vocabulary + rules and generalizes
     to combinations it never saw.
   Concretely: grow the verbs/macros over the effect algebra, add fine deltas where
   the algebra is too thin. This is **sim/algebra engineering, not prompt
   engineering.** The CFG's real job is to make creativity compositional (hence
   learnable), not merely valid.

8. **Generalization is measured by train/test ENVIRONMENT splits, not by size.** One
   room proves nothing; a *big* room isn't the answer either. The answer is a
   *distribution* of small varied rooms with a **held-out test set**, and you report
   the **generalization gap**. (Procgen's method.)

9. **Supervised first; add DAgger/reward only where a measurement demands it.**
   Behavior cloning on good data is better than today — but it has a *named* failure
   (covariate shift: a too-good expert never demonstrates recovery from the mistakes
   the clone will make → the clone drifts off-manifold). Whether it bites is
   empirical and depends on state-space coverage. Measure drift; escalate only if.

10. **Architecture is the LAST thing decided, and the measurements scope it.** If
    supervised-on-good-data over a covering env distribution holds up, the multi-head
    / RL / bespoke-architecture machinery may be unnecessary. Don't pre-commit.

## The gate (do this before anything downstream)

**Build the generalization experiment, measure, then decide.**

1. **Procedural environment generator** — a *distribution* of small varied rooms
   (our `seed.py` is already flagged to shrink into layer-map authoring). Split into
   train / held-out test.
2. **Multi-room big-model harness** — extend the parallel, tag-namespaced session
   harness (already built; ran 4 persona agents) so Opus/Haiku plays across the
   distribution and harvests **grounded** trajectories per room.
3. **Measure** with `corpus_eval`: walk-collapse per bucket, the **generalization
   gap** (train vs held-out), and **drift** when a fitted model drives the sim for real.
4. Only if the gap/drift is bad do we reach for DAgger (harness relabels the clone's
   drifted states — the big-model expert is right there) or outcome-reward RL.

Everything below step 4 (the <500M architecture, CFG-constrained decoding, multi-head
training, quantization/speed) is **gated on these numbers** and premature until then.

## First gate run (2026-08) — what it taught us

Ran the full pipeline: `generate` → 5 Haiku agents play procedural rooms (3 train
seeds 0/1/2, 2 held-out test seeds 50/51) → harvest → `generalization_eval.py`.

- **The generator is solid.** All 5 agents reported coherent, playable rooms, sound
  physics, reactive NPCs, no crashes. The procedural distribution works.
- **The generalization number was underpowered** (+0% gap on 2 train / 7 test
  anchors — every raw pick was already correct, so no lift/gap could manifest). Not
  a real answer; the harvest was too thin. *Why* it was thin is the finding:
- **THE HARVEST IS STRUCTURALLY SOCIAL-DOMINATED.** ~1 verb trace across all five
  combat/fire-heavy sessions; everything else was social. The model only *scores*
  (and thus harvests) the **ambiguous residual the sim hands it**:
  - explicit free-text verbs resolve **lexically** → no scoring → **no capture**;
  - menu play is deterministic → **no capture**;
  - only *ambiguous* free-text verbs AND **every addressed-NPC reply** get scored.
  Competent agents type explicit commands, so verb-selection barely harvests; social
  replies harvest every time.

**Implication (reframes decision 6/7 above):** the trainable, harvestable,
*generalizable* decision is the **social reply** — fixed-vocab stances (cower/defy/
warm/…) that recur across every room, so they CAN transfer. Verb-parse is mostly
**deterministic lexical resolution the sim already does** — little to learn, and it
barely produces training signal anyway. This lands exactly on the latency-budget
warm slots: *parse* (mostly deterministic) + *reply* (the real learnable decision).
So the walk-collapse worry was partly aimed at a decision the model rarely even
makes. **Next informative run: a SOCIAL-heavy harvest across many rooms** (talk a
lot, everywhere) to get a real *social* generalization gap on fixed-vocab buckets.

## Open risks (grill these)

- **Coverage vs. openness pull opposite ways.** A wider interface enlarges the state
  space the expert must cover to prevent clone drift. Widen and cover, or drift wins.
- **Reward design is the real hard part.** "Advanced the world" is easy to say; a
  concrete, non-gameable sim-progress metric is not yet designed.
- **Distribution match.** Opus-authored *scenarios* must still be sim-adjudicable end
  to end, or the "grounded" claim leaks.
- **Latency budget still binds** ([[latency-budget]]): reactive per-tick NPC cognition
  must remain 0 LM calls. Whatever we train cannot run the hot loop; the sim owns it.
