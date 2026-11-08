# Persona steering for small choosers: a deeper review

2026-09-29. Follows `docs/character-methods-review.md` (62 sources). That
review asked "what exists besides persona text?". This one asks a narrower
question: **how do we make a trait more pronounced and controllable in a model
that CHOOSES among options, at 0.3B to 2B, on a 6 GB GPU?** It does not repeat
the game-AI and psychology material of the earlier review. Score calibration
and diagnosing wrong picks are covered in `docs/mech-interp-review.md`
(Sections 3.13, 3.14, 3.16); this review covers personality only.

What we measured (roadmap entries 100 to 102, `lab/world2/policy.py`):

| policy | "surprising and in character" (layer 3) |
| --- | --- |
| random | 0% (and 60% odd choices) |
| rules (utility AI, hand-written) | 3.6% |
| persona as a sentence in the input (GLiNER2.5-Decide) | 0.3%, types backwards |
| model score + hand-written trait bias | 10.4% |

**How sources were used.** Every source below was found by web search and
opened (abstract page, HTML or PDF). Numbers come from the source text. Where a
detail could not be checked, the text says so. Many 2026 sources are arXiv
preprints and have not been peer reviewed. Numbers in brackets point to
Section 7.

---

## 1. Summary

1. **Our additive trait bias is a known, sound pattern.** Adding a trait score
   to the model's option score in log space is a "product of experts", the same
   arithmetic as DExperts and classifier-free guidance [76, 74]. Its only weak
   point is that the option features and weights are hand-written; both can be
   derived from a model or learned [47, 1].
2. **"Steering is weak below 2B" needs restating.** Steering does move small
   models, down to 124M [32] (refusal: 1.8B and 2B [14]). But at about 2B it
   loses to prompting on generation benchmarks [34, 54], small Qwen models break
   early when generating [32], and **no study measured personality steering of
   forced-choice option scores below 2B**. It is unmeasured for our use, not
   disproven.
3. **Forced choice is where steering is most reliable,** with monotonic effects
   as strength grows [42, 11]. But a persona description in the prompt cancelled
   steering completely at 70B [42], and effects vary input by input, with up to
   about half the inputs moving the wrong way in some datasets [27].
4. **Trait strength can be dialled, but traits are not independent.** Several
   3B to 34B studies report dose-response correlations above 0.9 [3, 47]. But
   steering one trait moves others even after the directions are made
   orthogonal [43], strength has an inverted-U effect [39], and 14% of 275
   roles got worse as strength grew [8].
5. **Talk and action come apart.** Persona text changes what a model says about
   itself more than what it does [56]. Steering "forgiveness" made a model's
   words more forgiving and its game moves less forgiving [7]. Measure choices
   only, as the lab already does.
6. **Small encoders and rerankers read instructions as keywords.** In FollowIR,
   retrieval models under about 3B that were not instruction-tuned did not follow
   instructions [70]. This matches our 0.3% result. For encoders the evidence
   points to learned conditioning (a per-user learned prefix, LoRA with user
   embeddings, LoReFT on RoBERTa) [63, 65, 58], or keeping the bias outside the
   model. We found no study that steers a DeBERTa or NLI chooser by adding a
   vector to its activations.
7. **The cheapest zero-shot upgrade for our decoders and NLI models is
   guidance on option scores:** neutral score + γ × (score with persona − score
   with the opposite persona), applied only among options the neutral model
   finds plausible. γ can be drawn per scene to make a trait a distribution.
   This is our proposal, assembled from classifier-free guidance, context-aware
   decoding and contrastive decoding [74, 75, 77]. We found no test of it on
   choosers.
8. **The tools fit the laptop.** nnsight and pyvene hook any PyTorch model,
   Qwen3 and DeBERTa included [88, 60]; repeng and steering-vectors target
   Hugging Face decoders and should work with Qwen3 (not tested by us)
   [86, 87]. Sparse-autoencoder (SAE) weights exist for
   Qwen3-1.7B-Base and Qwen3.5-2B-Base, not for 0.6B or 0.8B [92], and SAE
   steering loses to simple baselines [34, 36].

---

## 2. Terms

| term | meaning |
| --- | --- |
| activation | A number a layer computes inside the model for each token. |
| residual stream | The running sum of each layer's output. Every layer reads it and adds to it. Most steering adds a vector here. |
| steering vector | A fixed vector added to the residual stream at one or more layers, to push behaviour one way. |
| coefficient (α, strength) | How many times the steering vector is added. Negative values push the other way. |
| contrastive pairs | Two inputs that differ only in the trait (brave vs cowardly). Their activation difference gives a direction. |
| mean difference (MoD, DiffMean) | Average of (activation with trait − activation without). The simplest steering vector. |
| persona vector | A mean-difference vector for a character trait, built from system prompts that bring the trait out or suppress it [1]. |
| forced choice | The model must pick from a fixed list. Our choosers work this way. |
| log-probability | The log of the probability a decoder gives a text. We score each option by it. |
| logit | The raw score before it becomes a probability. |
| guidance (CFG) | Score with a condition, pushed further away from the score without it: s_uncond + γ(s_cond − s_uncond) [74]. |
| plausibility guard | Only let a re-scoring term reorder options that the plain model already rates close to the best [77]. |
| SAE (sparse autoencoder) | A model trained to split activations into many sparse "features", some of which a human can name. |
| LoRA | A small low-rank add-on to the weights. Training changes only the add-on. |
| ReFT, LoReFT | Training a small edit to the activations (not the weights) at chosen layers and tokens [58]. |
| prefix / prompt tuning | Training a few "virtual tokens" that the frozen model reads first. |
| FiLM | Scale and shift each feature by amounts computed from a conditioning vector [66]. |
| cross-encoder | Reads two texts together and outputs one score (our NLI models: premise = situation, hypothesis = question + option). |
| listwise chooser | Reads the situation and all options together and scores them all at once (GLiNER2.5-Decide, Laya). |
| trait bleed | Steering one trait also moves others. |
| anti-steerable input | An input on which steering moves behaviour the wrong way [27]. |
| dose-response | How the behaviour changes as strength goes up. |
| density distribution | A trait seen as the spread of a person's momentary states, with a stable mean and a stable width [81]. |

---

## 3. Methods

### 3.0 The size question: does "steering is weak below 2B" still hold?

The earlier review found steering results at 1.5B and 2B only, with prompting
winning at 2B. New evidence, grouped by what it shows:

**Steering moves small models.**
- GPT-2 (124M) to Llama-3.2-3B, nine models, emotion steering. An external
  emotion classifier saw the intended shift in 37 of 40 scenarios (92%). On
  GPT-2, anger probability fell from 0.945 to 0.041 between strengths 0.01 and
  0.02: a sharp flip, not a slope [32].
- A refusal direction exists and works in all 13 chat models tested, from Qwen
  1.8B and Gemma 2B up to 72B. It was computed from 128 harmful and 128 harmless
  instructions [14].
- On Gemma-2-2B (replicated on five models from 1.4B to 8B, with demos on
  Llama-3.2-1B and OLMo-2-1B), a "linear accessibility" measure predicted where
  mean-difference steering works, ρ = 0.86 to 0.91 [31]. The concepts were
  factual and word tasks, not personality.
- A hobby repository steers the style of Qwen3-0.6B. A unit-length vector had no
  effect. A raw difference vector at α = 2 cut answers from 149 to 16 words.
  α ≥ 3 broke coherence. Best layers were 12 to 17 [33]. (Not peer reviewed.)

**But it is fragile, and prompting often wins at 2B.**
- AxBench, Gemma-2-2B, open-ended generation, score 0 to 2 [34]: prompting
  0.70 to 0.73; LoReFT (trained) 0.70 to 0.72; mean difference 0.18 to 0.30;
  SAE features 0.15 to 0.18.
- Big Five on Gemma-2-2B [54]: prompting beat steering on all five traits
  (for example, conscientiousness: prompting +0.81, steering +0.29;
  neuroticism: +0.97 against +0.50).
- In generation, small Qwen models degrade early. Qwen2.5-1.5B perplexity grew
  12× (base) and 24× (instruct) from strength 0.005 to 0.05, with Chinese tokens
  appearing. Gemma-2-2B stayed coherent [32].
- Across 36 models from 1.5B to 70B, function vectors recovered 5-shot
  performance in only 20% of model-task pairs (52% after a parameter search)
  [29].

**Verdict.** The claim holds for open-ended generation at about 2B. It does not
hold as a blanket claim: steering moves behaviour even at 124M. For our case,
forced-choice scoring of options for personality below 2B, **there is no
direct evidence either way**. Two facts change the comparison for us:
1. Prompting did not work for our listwise chooser (0.3%), so steering does not
   have to beat a strong prompt baseline there.
2. We score options and do not generate text, so the "explosive" failure (bad
   text) shows up in a different way: as scores that stop tracking what makes
   sense. We must measure that directly (Section 5).

### 3.1 Persona text in the input (the baseline)

- **What it is.** "Tom is timid and kind." in the model's input.
- **Evidence.** Persona injection moves self-reports, with little or
  inconsistent effect on behaviour [56]. Retrieval models under about 3B that
  were not instruction-tuned did not follow instructions. They used them as
  keywords, and scored better when given only the keywords [70]. Our
  GLiNER2.5-Decide result (0.3%, types backwards) is consistent with keyword use.
- **Interaction with steering.** At 70B, a character description in the prompt
  made steering have no effect in forced choice [42]. In Llama-2 7B/13B chat,
  steering added to the effect of a system prompt [11]. So the interaction must be
  measured, not assumed.
- **Plugs into option scoring:** as the "with persona" score in guidance (3.2).

### 3.2 Output-level guidance on option scores

- **What it is.** Run the model twice (with and without a condition) and combine
  the two scores. Four versions:
  - **Classifier-free guidance (CFG) for LLMs** [74]:
    log P̂ = log P(w | c) + γ [log P(w | c) − log P(w | c̄)], where c̄ is empty or a
    "negative prompt".
  - **Context-aware decoding** [75]: (1 + α) log p(y | c, x) − α log p(y | x).
    It amplifies whatever the context changes.
  - **Contrastive decoding** [77]: the score of a large "expert" minus that of a
    small "amateur", but only over tokens the expert finds plausible.
  - **DExperts** [76]: base logits + α (expert − anti-expert), with small experts
    steering a larger base model.
- **Needs.** No training. Two forward passes per choice (three with a negative
  persona).
- **Fits laptop.** Yes. The lab's "text" policy already runs both passes.
- **Small-model evidence.** CFG helped GPT-2, Pythia (160M to 12B) and LLaMA;
  γ = 1.5 was the default. On 5 of 9 tasks, a model with CFG did not differ
  significantly from a model twice its size without CFG; of the other 4, 2
  favoured CFG and 2 the bigger model [74]. Contrastive decoding used OPT-125M as the amateur [77].
  Scoring multiple-choice answers against a prior (domain-conditional PMI [79])
  and contextual calibration [80] fixed option biases in GPT-2 and GPT-3.
- **Evidence for personas.** With a negative prompt (the default system prompt
  versus the edited one, for example "write a sad response"), CFG at γ = 3 was
  preferred for following the system prompt in 75% of human comparisons, with no
  loss of relevance to the user prompt (52%) [74]. We found no study that
  applies guidance to a character persona in a choice task.
- **Controllability.** Continuous (γ). Per character (the persona sentence). A
  distribution: draw γ per scene.
- **Failure modes.** It amplifies the sign of the persona effect. If the text
  effect is backwards (as in GLiNER), guidance makes it more backwards. Large γ
  lets the persona term choose nonsense; contrastive decoding's plausibility
  guard exists to stop this [77].
- **Plugs into option scoring (our proposal):**
  `s(o) = s0(o) + γ · [s+(o) − s−(o)]`, where s0 has no persona, s+ has "Tom is
  brave", s− has "Tom is cowardly" (or none). Apply it only to options with
  s0(o) ≥ max s0 − δ. Draw γ from the character's trait distribution.

### 3.3 Additive trait bias, lexicons and logit biasing

- **What it is.** Add a trait-dependent score to each option. This is our "bias"
  policy. Two published relatives work on tokens:
  - **UPLex** [47] multiplies each next-token probability by
    (1 + α Σ β_t L[w]^t), where L is a per-word, per-trait lexicon built without
    labels from model answers to situational-judgement questions (200 refined
    questions per trait). β_t sets each Big Five trait.
  - **SWAI** [78] biases high-scoring tokens by corpus log-odds, only within the
    model's top-K candidates.
- **Needs.** A lexicon or feature table. UPLex builds one without labels.
- **Fits laptop.** Yes. No model call beyond the one we already make. Zero calls
  if the model is not run at all.
- **Evidence.** UPLex, 7B to 34B models: Pearson correlations between β_t and
  measured trait intensity "consistently greater than 0.9", perplexity stable,
  several traits combine [47]. Our
  lab: 10.4% "surprising and in character" (hand-written weights).
- **Controllability.** Continuous, per trait, per character. A distribution:
  draw the trait value per scene (the lab already does).
- **Failure modes.** Only as good as its features: a missing feature means a
  missing reason. Word-level lexicons may reward word overlap, which is already
  our models' main failure (roadmap 102).
- **Plugs into option scoring:** directly; this is what we have. Section 4
  lists ways to derive features and weights instead of writing them.

### 3.4 Contrastive steering vectors (ActAdd, CAA, RepE, persona vectors, mean-centring)

- **What it is.** Build a direction from contrastive pairs and add α times it to
  the residual stream.
  - ActAdd: one prompt pair ("Love" − "Hate") [10].
  - CAA: average over many A/B answer pairs, taken at the answer token, then
    added at all token positions after the instruction; layer 13 was best in
    Llama-2-7B chat, 14 or 15 in 13B [11].
  - RepE: population-level directions for honesty and more [12].
  - Mean-centring: subtract the mean of all activations from the target mean [20].
  - Persona vectors [1]: Claude writes 5 pairs of trait-eliciting and
    trait-suppressing system prompts and 40 questions. The vector is the mean
    difference of response-token activations; the best layer is picked by
    testing.
  - Theory and tests on Llama-2-7B: the mean difference is the optimal vector
    for moving negatives to positives, and it beat PCA and classifier
    directions on most tasks [22].
- **Needs.** Tens to hundreds of contrastive examples. Forward passes only.
- **Fits laptop.** Yes. Qwen3-1.7B in bf16 is about 3.4 GB of weights.
- **Small-model evidence.** See 3.0. For personality specifically, the smallest
  models we found are Gemma-2-2B [54] and Qwen3-4B / Qwen2.5-3B [3, 6].
- **Personality evidence.**
  - Persona vectors [1] (Qwen2.5-7B, Llama-3.1-8B): steering raises the trait
    with coefficient. Projecting the last prompt token onto the vector predicts
    trait expression before any text is written (r = 0.75 to 0.83). Large
    inference-time coefficients lower MMLU.
  - PERSONA [3] (Qwen2.5 3B/7B/14B, Qwen3-4B, Llama-3.1-8B, Ministral-8B): α
    from −1 to +2 gives a correlation above 0.9 between α and trait score; adding
    vectors composes traits; PersonalityBench 9.60 against 9.61 for fine-tuning.
  - Big Five by regression (Llama-3.3-70B) [42]: forced choice moved
    monotonically with α, in steps, not smoothly; |α| ≤ 0.4 stayed coherent;
    open-ended text barely changed; a character description in the prompt
    cancelled steering entirely.
  - In games (Qwen2.5-7B, layer 20, 500 prompt-question pairs per trait) [7]:
    altruism steering raised dictator-game gifts from $15 (β = 0) to up to $55;
    the strongest effects were for β between 0 and 3. β was tested from −5 to
    5; at large |β| quality fell (less variety, reasoning that did not match the
    choice, switching between English and Chinese). In the trust game,
    forgiveness steering made the words more forgiving and the moves less
    forgiving.
  - As a scorer (Qwen3-4B, Qwen3-32B, gpt-oss-20b, |α| = 2) [6]: persona steering
    changed how the model GRADED answers ("evil" and "impolite" graded harshly,
    "optimistic" leniently). The 4B model shifted most (mean absolute shift 0.172
    against 0.021 for 32B). This is the closest published case to steering a
    scorer, which is what our choosers are.
- **Controllability.** Continuous α; per character; a distribution by drawing α
  per scene. The response is often an inverted U: trait expression rises, peaks,
  then falls, while coherence falls steadily [39, 8].
- **Failure modes.**
  - Unreliable per input. Llama-2-7b-chat and Qwen-1.5-14b-chat, 40 datasets:
    in some datasets almost half the inputs were anti-steerable; part of the
    effect came from a bias toward answer position (A vs B) [27]. Our choosers
    already have position and favourite-item biases (roadmap 102).
  - Steering works when the pairs agree in direction. High cosine similarity
    between the pair differences, and a high signal-to-noise ratio, predict
    success (r = 0.76 across 38 datasets, 3B to 14B) [28, 30]. Both can be
    computed before steering.
  - Trait bleed: see 3.10.
  - Side effects: in a safety benchmark (Gemma-2-2B, Llama-3.1-8B, Qwen2.5-7B),
    steering one behaviour degraded social behaviours by up to 76% [38]. Side
    effects can be forecast from the model's baseline representations [40].
- **Plugs into option scoring.** Add α·v during the scoring pass. Every option's
  log-probability (or NLI logit) changes. Draw α from the character's trait
  distribution for each choice. No extra model call.

### 3.5 Variants of activation steering

| variant | idea | evidence (model sizes) | use for us |
| --- | --- | --- | --- |
| ITI [13] | Shift a few attention heads chosen by probes. | Alpaca 7B TruthfulQA 32.5% → 65.1%, from a few hundred examples. | Finer than whole-layer steering. No small-model data. |
| Angular steering [21] | Rotate activations in a 2D plane instead of adding. Angle is the dial. | 3B to 14B; benchmarks kept "across the steering circle"; 3B showed interference at some angles. | Bounded strength (an angle cannot blow up the norm). None below 3B. |
| Conceptors [19] | A soft projection matrix instead of a vector; AND/OR/NOT combine goals. | GPT-J 6B: antonyms 52.1% vs 20.5% for addition. | Combining traits. Matrix is 17 MB at width 2,048. |
| CAST [23] | Apply the behaviour vector only when the input matches a "condition" vector (cosine). | Refusal rules ("if about X, refuse"). | An "if-then" personality: brave only when a child is threatened. |
| BiPO [18] | Optimise the vector so the preferred answer gets higher log-probability than the dispreferred one; multiplier sampled from {−1, +1}. | Llama-2-7b-chat, Mistral-7B; 307 to 642 training pairs per behaviour; beat CAA; ±1, ±2 graded; vectors add. | Its loss is option scoring. Train on large-model option preferences. |
| GCAD, PSR [24, 25] | Copy how a prompt acts (attention-level, token-specific strength) instead of one fixed vector. | GCAD (Qwen2.5-7B, Llama-3.1-8B): multi-turn coherence drift −18.6 → −1.9; turn-10 trait expression 78.0% → 93.1%. PSR sizes not recorded. | Keeps coherence over long sessions. Complex. |
| HyperSteer [26] | A hypernetwork writes the steering vector from a text description. | "On par with steering-via-prompting" on AxBench. | Persona text → vector without per-trait data. Needs training. |

### 3.6 Function vectors, in-context vectors, task vectors

- **What it is.** Compress demonstrations into a vector and patch it in at a
  middle layer [15, 16, 17]. In-context vectors were tested on style transfer and
  role-playing, and combine by vector arithmetic [16].
- **Evidence.** Function vectors are robust in the original models (GPT-J,
  Llama-2 13B and 70B) [15], but across 36 models from 1.5B to 70B they worked in
  only 20% of model-task pairs by default [29].
- **For us.** Weak fit. They transfer a task, not a trait strength. A "brave
  character's choices" ICV is possible but unproven.

### 3.7 SAE feature steering

- **What it is.** Find an SAE feature for the trait and add its decoder
  direction.
- **Evidence.** On AxBench, SAE steering scored lowest (0.15 to 0.18 on
  Gemma-2-2B) [34]. Picking features by their effect on the OUTPUT, not by what
  activates them, gave 2 to 3× better steering, level with supervised methods
  [35]. Google DeepMind found linear probes beat SAE probes out of distribution
  and "deprioritised" SAE research [36]. Anthropic (Claude 3 Sonnet, 29
  features) found a "sweet spot" of −5 to +5 and off-target effects (a
  gender-bias feature raised age-bias scores by 13%) [37]. An OCEAN study with
  SAE features used an 8B model [57].
- **For us.** SAEs exist for Qwen3-1.7B-Base and Qwen3.5-2B-Base (all layers,
  residual stream, 32K width) [92], not for 0.6B or 0.8B. Our models are
  post-trained or fine-tuned, so base-model SAEs may not match. Use for
  inspection, not as the first steering method.

### 3.8 Neuron edits and weight arithmetic

- **NPTI** [46]. Find neurons whose activation differs between opposite trait
  answers (about 20,000 in LLaMA-3-8B), then raise one side and lower the
  other. Matched fine-tuning on PersonalityBench. 7B to 9B only.
- **Personality vectors by model merging** [48]. Fine-tune a model per trait,
  subtract the base weights, add the difference scaled by α to any model of the
  same family. Continuous intensity, traits compose, transfer across models.
  Llama-3.1-8B and Qwen2.5-7B.
- **PAS** [53]. Searches for the most useful attention heads and shifts them per
  user, from the PAPI data of 300,000 people. Beat DPO and PPO at 1/6 of PPO's
  time. Llama-3-8B and 70B.
- **For us.** Merging suits a fixed per-character model, not a per-scene
  distribution. It needs one fine-tune per trait (off-laptop). NPTI and PAS are
  forms of activation steering with extra search.

### 3.9 ReFT and LoReFT (trained activation edits)

- **What it is.** Learn a low-rank edit to hidden states at chosen layers and
  positions; the model's weights stay frozen [58].
- **Needs.** Labelled examples and backpropagation through the frozen model.
- **Evidence.** 15 to 65 times fewer parameters than LoRA [58]. It also works for
  classification with small encoders: RoBERTa-base (125M) and RoBERTa-large
  (350M) on GLUE, level with other efficient methods [58]. The pyreft demo
  trains an "emoji-only" persona from 10 examples in 36 seconds (on a 7B model)
  [59]. On AxBench, LoReFT at 2B scored as well as prompting (0.70 to 0.72) [34].
- **Fits laptop.** Training for 0.6B and the encoders probably fits in 6 GB
  (estimate, not measured). 1.7B is tight; train off-laptop.
- **Controllability.** Trained for one behaviour. A strength dial is not built
  in. Scaling the learned edit by a trait value is our extension and is
  untested.
- **Plugs into option scoring.** It changes the scorer. It can be trained
  directly on "which option this character picks" labels.

### 3.10 What the personality papers say about linearity, strength and interference

- **Linear?** Mostly. Big Five regression directions (70B) are nearly
  orthogonal and separate trait adjectives cleanly [42]. Persona space in
  27B to 70B models is low-dimensional (4 to 19 components explain 70%), with one
  shared "Assistant axis" [2].
- **Strength dialable?** Yes, within a window. Examples: r > 0.9 for most
  traits [3], r > 0.9 [47], graded ±1/±2 [18], a rotation angle [21]. In forced choice the response
  is step-like because the output is discrete [42]. Past the window, coherence
  collapses [42, 7, 32].
- **Interference.** Steering one Big Five trait moves others even after hard
  orthonormalisation, which also weakens the steering (LLaMA-3-8B, Mistral-8B)
  [43]. In persona vectors, negative traits and humour moved together, opposite
  to optimism [1]. Naively adding several vectors made models collapse; steering
  traits one after another, each on the already-steered stream, fixed much of it
  (7B to 8B) [45]. Of 171 trait pairs in Qwen3-8B, all 40 destructive pairs
  involved two traits that were NOT already natural for the model [4].
- **Not every character is steerable.** In a social-simulation study (OLMo-3-7B,
  275 roles), most roles improved with strength (median per-role r = +0.98), but
  14% got worse on every measure as strength rose. The authors screen every role
  over a coefficient grid before use [8].
- **Training beats prompting at 8B+.** SFT and DPO on 100,000 dialogues beat
  prompting on BFI and IPIP-NEO (LLaMA-3-8B and 70B) [49]. PEFT was strongest on
  Gemma-2-2B for some traits, but cost other skills [54].

### 3.11 LoRA adapters, mixtures and soft personas

| method | what | data | model |
| --- | --- | --- | --- |
| Neeko [61] | One LoRA block per character (rank 4 each), gated by a role embedding; new characters by fusing or adding blocks. | 9 characters, about 1.6K scenes each | LLaMA-2 7B |
| P-Tailor / P-React [50] | A LoRA expert per Big Five trait, mixed by routing, with a loss that keeps experts distinct. | OCEAN-Chat (size not checked) | not checked |
| Machine Mindset [51] | SFT, then DPO, per MBTI type. | "hundreds of thousands" of entries (not checked) | Baichuan, LLaMA-2 (sizes not checked) |
| PersonaPKT [62] | A learned prefix per persona, initialised from a prefix shared across personas. | "a small number of" dialogues per persona; < 0.1% parameters | backbone not checked |

For us, LoRA fits a fixed character. A continuous trait dial needs either one
adapter per trait with a scaled mix [48, 50], or trait values fed in as input
(Section 3.12).

### 3.12 Conditioning encoders and cross-encoders

This matters because most of our choosers are encoders.

- **Instructions do not work well as text.** See FollowIR [70] in 3.1.
- **Learned per-user conditioning works on encoders.**
  - UserAdapter [63]: a learned virtual "prefix" token per user steers RoBERTa's
    representations for sentiment; under 0.5% extra parameters per user; about
    a dozen examples per new user.
  - UserIdentifier [64]: a fixed random string per user, prepended, with one
    shared model trained end to end; up to 13% better than a prefix-tuning
    method.
  - PLoRA [65]: user knowledge injected into LoRA for BERT, RoBERTa and
    Flan-T5, with zero-shot and few-shot settings for new users.
  - Speaker embeddings in dialogue models improved consistency [67].
  - FiLM [66] conditions any feature map on a vector (from vision; not tested on
    choosers).
- **Interventions inside encoders change predictions.** Pushing BERT's
  representation across a linear boundary changed its agreement predictions
  (AlterRep) [68]. LEACE erases a linear concept in closed form, for example
  gender in BERT [69]. LoReFT trains edits on RoBERTa [58].
- **Steering LLM rankers.** RankSteer [71] steered 7B to 8B pointwise LLM
  rankers with 5 anchor queries and 3 role-instruction pairs. Role instructions
  and query-document content were processed "largely independently", with role
  signals in early layers. The largest gain we read in its tables was nDCG@10
  from 0.508 to 0.636 (we read this as Mistral-7B on TREC DL 2020; the column
  assignment was not fully checked).
- **Gap.** We found no study that adds a behaviour or persona vector to a
  DeBERTa-style chooser or an NLI cross-encoder. It is mechanically the same
  (hooks on hidden states). It is untested.

### 3.13 Game, NPC and "trait as distribution" work

- **Persona vectors in games** [7]: see 3.4 (dictator $15 → $55; rhetoric and
  strategy can move in opposite directions).
- **Prompted Big Five in repeated prisoner's dilemma** (GPT-3.5, GPT-4o, GPT-5)
  [84]: agreeableness was the main driver of cooperation; the personality
  prompt biased behaviour rather than fixing it (our paraphrase).
- **Across contexts** [83]: prompted Big Five agents behaved differently by
  context, which the authors read through Whole Trait Theory.
- **npc-steering-plus** [85], a hobby project: six affect numbers that decay to
  a persona's baseline set the strength of valence, arousal and dominance
  vectors in Qwen3.5-9B (4-bit, layer 15). Close to our "state from trait mean
  plus momentum" design. Not evaluated in a paper.
- **Trait as distribution.** Fleeson [81]: in three experience-sampling studies
  over 2 to 3 weeks, the typical person showed "nearly all levels of all
  traits"; each person's mean was almost perfectly stable; the width of the
  spread was a stable difference between people; and variability in
  extraversion reflected how strongly people reacted to situational cues. Whole
  Trait Theory adds that the spread is produced by goals, beliefs and
  situations [82].
  **We found no paper that draws the steering strength per decision to produce
  such a distribution.** The pieces exist: continuous dose-response (3.10), and
  context-predicted per-turn coefficients in PERSONA's "Persona-Flow" [3].
  Fleeson's point about cues suggests the draw should depend on the situation
  (if-then), not be pure noise; CAST [23] is one way to make it conditional.

### 3.14 Tools for a 6 GB GPU

| tool | what it does | Qwen3 decoders | DeBERTa / NLI encoders | note |
| --- | --- | --- | --- | --- |
| repeng [86] | Train a control vector from pairs "in less than sixty seconds"; export GGUF for llama.cpp | HF decoders (docs show Mistral) | not stated | Does not work with MoE models. |
| steering-vectors [87] | CAA-style train and apply with a multiplier | HF LMs "GPT, LLaMa, Gemma, Mistral, Pythia, and many more" | not stated | |
| nnsight [88] | Read and edit any module's activations in a trace | yes (any PyTorch model) | yes (any PyTorch model) | Local or remote. |
| pyvene, pyreft [60, 59] | Declarative interventions; ReFT training | yes | yes ("any PyTorch model") | |
| TransformerLens [89] | Hooked models for analysis | `qwen3`, `qwen3_5` adapters | `bert` adapter; no DeBERTa adapter found | Checked the `main`, `dev` and `dev-4.x` trees on 2026-09-29. `docs/mech-interp-review.md` lists deberta adapters; we could not find them. Use nnsight or pyvene for DeBERTa. |
| SAELens [90] | Train and load SAEs; encode/decode works with any PyTorch model | via hooks | via hooks | |
| Qwen-Scope [92] | Official SAEs, all layers | Qwen3-1.7B-Base, Qwen3.5-2B-Base (and larger) | none | No 0.6B or 0.8B. |
| Neuronpedia [91] | Browse SAE features; has a steering page | a Qwen3-1.7B page exists | none found | Page listing was truncated. |
| EasySteer [93] | Steering on vLLM, 10.8 to 22.3× faster than other frameworks | vLLM models | no | For serving, not needed in the lab. |
| AISteer360 [94] | IBM toolkit: input, weight, activation and output steering in one pipeline (includes CAST) | HF native | not stated | |

### 3.15 Comparison

"Small-model evidence" lists the smallest size with a published result for any
behaviour, and for personality where known. "Sense risk" is the main way the
method could make choices stop making sense.

| method | needs | fits laptop at play | small-model evidence | strength dial | per character | distribution | sense risk | plugs into option scoring |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| persona text | a sentence | yes, 1 call | weak; keywords under 3B [70]; our 0.3% | no | yes | no | low | as the input |
| guidance on scores [74, 75, 77] | nothing | yes, 2 to 3 calls | CFG from 160M [74]; not on personas | γ | yes | draw γ | amplifies wrong signs; guard needed | s0 + γ(s+ − s−) |
| additive bias (ours), lexicon [47] | features + weights | yes, 0 extra calls | our 10.4%; UPLex 7B+ | trait value | yes | draw trait | missing features | already does |
| contrastive vector (CAA, persona vector) [1, 11] | pairs, forward passes | yes | 124M emotion [32]; 2B personality, weaker than prompting [54] | α | yes | draw α | per-input sign flips [27]; context override [42] | edit during scoring |
| BiPO [18] | preference pairs, backprop | 0.6B likely; else off | 7B | α | yes | draw α | as above | trained on option preferences |
| angular / conceptors / CAST [21, 19, 23] | pairs | yes | 3B / 6B / not recorded | angle / matrix / condition | yes | draw | fewer blow-ups (angle) | edit during scoring |
| SAE features [35, 92] | SAE weights | 1.7B / 2B only | 2B; weakest on AxBench [34] | yes | yes | draw | off-target effects [37] | edit during scoring |
| neuron / merge [46, 48] | search or per-trait fine-tune | merge: yes at play | 7B+ | α | yes | merge: no | capability loss | changed model |
| LoReFT [58] | labels, backprop | train off; run on | RoBERTa-base 125M (GLUE) | not built in | per model | no | low if trained on our labels | trained scorer |
| LoRA, mixtures [61, 50] | labels, backprop | train off; run on | 7B for personas | mix weights | per adapter | weak | skill loss [54] | trained scorer |
| learned prefix / user embedding [63, 64, 65] | labels | train off; run on | RoBERTa, BERT | scale (untested) | yes | draw (untested) | low | trained scorer input |
| FiLM on option embeddings [66] | labels | train off; run on | none for text choosers | trait vector | yes | draw trait | low | head on the scorer |

---

## 4. What this means for our choosers

One rule first. Our latency budget allows **zero model calls for most
characters** on most ticks; only the addressed character's reply uses a model.
So the additive bias (3.3) is the main path for most characters, and anything
inside a model helps only when that model runs. Every ranking below keeps that
in mind.

### 4.1 Decoder LM scoring options by log-probability (Qwen3-0.6B, Qwen3-1.7B)

1. **Guidance on option scores, with a plausibility guard.** Score each option
   three ways: no persona (s0), positive persona (s+), opposite persona (s−).
   Use s0 + γ(s+ − s−) only among options with s0 ≥ max s0 − δ. Draw γ from the
   trait distribution. *Why first:* no training; it cannot damage the model's
   internals; CFG helped from 160M up [74]; the negative-prompt form improved
   system-prompt following [74]; the guard comes from contrastive decoding [77].
   *Check first:* the sign. Measure whether s+ − s− ranks trait-tagged options
   the right way before amplifying it.
2. **A contrastive steering vector per trait, α drawn per scene.** Mean
   difference [22], middle layers (sweep), built from our own scenes with
   "Tom is brave / Tom is cowardly" system prompts, persona-vector style [1].
   Tools: repeng or steering-vectors, or nnsight. Compute the pair
   signal-to-noise ratio first [30]. *Why second:* forced choice is where
   steering is most reliable [42, 11], and there is no extra call. *Risks:*
   unmeasured below 2B for personality; small Qwen models degrade early [32];
   long scene text may override it [42]; per-input sign flips [27]. Do not
   combine it with persona text until the interaction is measured.
3. **A BiPO vector trained on option preferences.** Large-model labels give
   "this character prefers option A over B in this scene"; BiPO's loss raises
   log p(A) − log p(B) directly, with a signed multiplier [18]. BiPO used 300 to
   650 pairs per behaviour at 7B. Likely trainable on the laptop for 0.6B (an
   estimate); off-laptop for 1.7B. *Alternative:* LoReFT trained on the same
   labels [58].

### 4.2 NLI cross-encoder (openjev, Qwen3.5 2B / 0.8B)

1. **Use the NLI model to MAKE the trait features, then keep the additive
   bias.** Premise: the action in context ("Tom strikes the bandit who holds a
   knife"). Hypotheses: "This is a brave act.", "This is a greedy act.", and so
   on. The entailment scores become option features; weights are learned from
   large-model labels (Section 5, E3 and E4). Cache per option in context. *Why first:*
   it keeps the design that already works (10.4%), removes hand-written
   features, and costs nothing per tick for characters whose options are
   cached. NLI models classify text against arbitrary label hypotheses zero-shot
   [72]. *Risks:* hypothesis wording changes results; our models are pulled by
   word overlap (roadmap 102); entity binding ("strike the child") must be
   checked. This is our proposal; untested.
2. **Guidance on entailment scores** (persona premise against neutral premise),
   as in 4.1 item 1. *Risk:* FollowIR suggests a small cross-encoder may treat
   the persona as keywords [70], so s+ − s− may measure word overlap, not the
   trait. Test the sign.
3. **Activation steering inside the cross-encoder** (hooks through nnsight or
   pyvene). Exploratory. The closest evidence is steering of 7B to 8B LLM rankers
   with a handful of anchors [71]. Qwen-Scope has SAEs for Qwen3.5-2B-Base [92],
   but openjev is fine-tuned, so features may not transfer.

### 4.3 Listwise encoder (GLiNER2.5-Decide 340M, Laya 421M, bev-decider 0.4B)

1. **Keep the bias outside the model and learn it.** Features from the world's
   tags, from cached NLI trait scores (4.2 item 1, computed offline by openjev),
   or from projecting option text onto trait axes in an embedding space
   (SemAxis-style [73]; persona-vector projection predicts trait expression,
   r = 0.75 to 0.83 at 7B to 8B [1]). Weights learned from large-model labels.
   *Why first:* it is the only option with zero added calls, and it builds on
   the 10.4% result.
2. **Train trait conditioning into the encoder.** A learned prefix vector per
   trait, scaled by the drawn trait value (UserAdapter-style [63], PersonaPKT
   [62]), or FiLM on the option embeddings from the trait vector [66]. Train
   off-laptop on large-model-labelled snapshots. *Why second:* per-user prefixes
   and user embeddings work on RoBERTa and BERT [63, 64, 65]; scaling them for
   strength is untested.
3. **LoReFT with a trait-scaled edit** [58]. It works on RoBERTa-base and -large
   for classification; the scaling is our extension.

**Do not** put the persona as text in these models' input and expect a trait
effect. Our 0.3% and FollowIR [70] agree.

---

## 5. Experiment plan (lab, one factor at a time)

Run in `lab/world2` and on the Workstream A cases, with the existing layer-2 and
layer-3 measures. One model on the GPU at a time. Same seeds for every arm.

### 5.1 Measures

**"More pronounced":**
- **Dose-response slope.** Hold four traits at 0.5. Set the fifth to 0, 0.25,
  0.5, 0.75, 1. Over a fixed bank of decision snapshots from world2 logs,
  measure the probability mass on options tagged as expressing that trait
  (bravery: defend, strike a threat; against it: flee, hide). Report the change
  in log-odds per 0.1 trait. Baselines: rules, bias (10.4%), text.
- **Type separation.** Jensen–Shannon divergence between the action
  distributions of two character types on the same snapshots; and the plan's
  type ranges (for example, knight brave over 70%; rules reach 19%).
- **Layer-3 "surprising and in character"** rate (existing).

**"Without losing sense":**
- Acceptable-option rate on Workstream A cases with the method on, at neutral
  and at extreme traits, against the method off.
- Layer-3 "odd" rate; layer-2 nonsense rate at matched variety.

**Control quality:**
- Monotonicity: Spearman correlation between strength and behaviour per
  snapshot; the fraction of anti-steerable snapshots [27].
- Trait bleed: a 5 × 5 matrix of slopes, steered trait against measured trait.
- Distribution fit (only with sampled strength): per character, the mean and
  spread of the trait-expressing choice rate across nights, against the target.
- Cost: model calls per choice, ms per choice, peak VRAM, GPU temperature.

**Pass rule (proposed):** a method passes if it at least doubles the bias
policy's dose-response slope, or raises layer 3 above 10.4%, while losing no more
than 3 points of acceptable-option rate and adding no more than 2 points of odd
choices. The user should set these thresholds.

### 5.2 Steps

| step | changes one thing | models | question |
| --- | --- | --- | --- |
| E0 | nothing: record baselines | all | Random, rules, text, bias on the same seeds. Add the text arm for Qwen3 decoders and openjev (only GLiNER is measured). Record the SIGN of the persona effect per trait. |
| E1 | guidance γ ∈ {0, 1, 2, 4, 8}, guard off | Qwen3-0.6B, 1.7B, openjev 0.8B | Does s0 + γ(s+ − s−) raise the slope? What happens to sense? |
| E2 | E1's best γ, guard δ on | same | Does the guard keep sense at the same slope? |
| E3 | automatic features instead of hand-written, weights fixed | bias policy | NLI trait features [72] and embedding projection [73, 1] against hand tags. Agreement with large-model trait labels first. |
| E4 | learned weights instead of hand weights | bias policy | Fit weights to large-model choice labels; test on held-out scenes. Watch label flattening (earlier review). |
| E5 | steering vector, α sweep and layer sweep, one trait | Qwen3-0.6B, then 1.7B | Pair SNR [30] first. Slope, sense, anti-steerable fraction. Then two traits for bleed [43]. |
| E6 | fixed α → α drawn per scene | best of E1/E5 | Do per-character distributions match target mean and spread (Fleeson [81])? Compare pure noise with a cue-dependent draw. |
| E7 (off-laptop) | trained conditioning | encoders; Qwen3-1.7B | Prefix/FiLM on GLiNER-Decide; BiPO or LoReFT on Qwen. Same measures. |

Stop early if E0 shows the persona sign is wrong for a model (skip E1/E2 for it)
or if E5's SNR is low for a trait (skip steering for that trait).

---

## 6. Open questions

1. Does a steering vector move option log-probabilities for trait-tagged
   options at 0.6B to 1.7B at all? No published result exists.
2. Does the long scene description override steering, as a character
   description did at 70B [42]?
3. For the Qwen3 decoders and openjev, is the persona-text effect the right sign?
   For GLiNER it was backwards.
4. Can an NLI model score "this action is brave" in context, including who the
   action is aimed at, well enough to replace hand-written features?
5. Is trait bleed in option space like trait bleed in text [43], or smaller
   because options are short?
6. Should the per-scene strength be pure noise, or depend on cues (Fleeson's
   reactivity finding [81])?
7. Do Qwen-Scope SAEs from base models [92] work on our post-trained Qwen3-1.7B
   and on the fine-tuned openjev?
8. Two to three passes per choice for guidance: acceptable for the addressed
   character only? The laptop's heat budget must be measured.
9. Large-model labels flatten variety (earlier review). Will learned weights
   inherit that?

**Not verified.** Model sizes in the mean-centring paper [20]; the base models
and data size of Machine Mindset [51] and P-React [50]; PersonaPKT's backbone
[62]; the models tested in SWAI [78] and PSR [25]; the full Neuronpedia SAE list
for Qwen3-1.7B [91]; whether steering-vectors and repeng support encoders; the
exact table column behind the RankSteer number [71]; the wording of [84] (the
PMC page could not be read directly). The
Qwen3-0.6B repository [33] and npc-steering-plus [85] are unreviewed hobby
projects. The Whole Trait Theory journal page was blocked, so [82] is the
author's lab page. Memory fits in Section 4 are estimates, not measurements.

---

## 7. References

*Persona vectors and follow-ups*
1. Chen, Arditi, Sleight, Evans, Lindsey (2025). Persona Vectors: Monitoring and Controlling Character Traits in Language Models. https://arxiv.org/abs/2507.21509
2. Lu, Gallagher, Michala, Fish, Lindsey (2026). The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models. https://arxiv.org/abs/2601.10387
3. Feng, Zhao, Zhong, Huang, Gu, Kong, Feng, Qin (2026). PERSONA: Dynamic and Compositional Inference-Time Personality Control via Activation Vector Algebra. ICLR 2026. https://arxiv.org/abs/2602.15669
4. Zeng, Emami, Choi (2026). What Models Express, Suppress, and Resist: Auditing Open-Weight LLMs with Persona Vectors. https://arxiv.org/abs/2607.13162
5. Pai, Wang, Lu, Sun, Lee, Chang (2025). BILLY: Steering Large Language Models via Merging Persona Vectors for Creative Generation. https://arxiv.org/abs/2510.10157
6. Wu, Henriksson (2026). Persona Matters: Effects of Activation Steering on Short Answer Generation and Scoring. https://arxiv.org/abs/2604.07102
7. Sun, Zhang (2026). Persona Vectors in Games: Measuring and Steering Strategies via Activation Vectors. https://arxiv.org/abs/2603.21398
8. Song et al. (2026). Role Steering of Language Models for Social Simulations. https://arxiv.org/abs/2608.00023
9. Wang, Dupré la Tour, Watkins, Makelov, Chi, Miserendino, Wang, Rajaram, Heidecke, Patwardhan, Mossing (2025). Persona Features Control Emergent Misalignment. https://arxiv.org/abs/2506.19823

*Steering methods*
10. Turner et al. (2023, v5 2024). Steering Language Models With Activation Engineering (ActAdd). https://arxiv.org/abs/2308.10248
11. Rimsky (Panickssery), Gabrieli, Schulz, Tong, Hubinger, Turner (2024). Steering Llama 2 via Contrastive Activation Addition. ACL 2024. https://arxiv.org/abs/2312.06681
12. Zou et al. (2023). Representation Engineering: A Top-Down Approach to AI Transparency. https://arxiv.org/abs/2310.01405
13. Li, Patel, Viégas, Pfister, Wattenberg (2023). Inference-Time Intervention: Eliciting Truthful Answers from a Language Model. NeurIPS 2023. https://arxiv.org/abs/2306.03341
14. Arditi, Obeso, Syed, Paleka, Panickssery, Gurnee, Nanda (2024). Refusal in Language Models Is Mediated by a Single Direction. https://arxiv.org/abs/2406.11717
15. Todd, Li, Sharma, Mueller, Wallace, Bau (2024). Function Vectors in Large Language Models. ICLR 2024. https://arxiv.org/abs/2310.15213
16. Liu, Ye, Xing, Zou (2024). In-context Vectors. ICML 2024. https://arxiv.org/abs/2311.06668
17. Hendel, Geva, Globerson (2023). In-Context Learning Creates Task Vectors. Findings of EMNLP 2023. https://arxiv.org/abs/2310.15916
18. Cao et al. (2024). Personalized Steering of Large Language Models: Versatile Steering Vectors Through Bi-directional Preference Optimization (BiPO). https://arxiv.org/abs/2406.00045
19. Postmus, Abreu (2024). Steering Large Language Models using Conceptors. https://arxiv.org/abs/2410.16314
20. Jorgensen, Cope, Schoots, Shanahan (2023). Improving Activation Steering in Language Models with Mean-Centring. https://arxiv.org/abs/2312.03813
21. Vu, Nguyen (2025). Angular Steering: Behavior Control via Rotation in Activation Space. NeurIPS 2025. https://arxiv.org/abs/2510.26243
22. Im, Li (2025). A Unified Understanding and Evaluation of Steering Methods. https://arxiv.org/abs/2502.02716
23. Lee, Padhi, Natesan Ramamurthy, et al. (2025). Programming Refusal with Conditional Activation Steering (CAST). ICLR 2025. https://arxiv.org/abs/2409.05907
24. Kang, Liu, Ma, Huang, Tan, Jiang (2026). Prompt-Activation Duality: Improving Activation Steering via Attention-Level Interventions. https://arxiv.org/abs/2605.10664
25. Heyman, Vandeputte (2026). Steer Like the LLM: Activation Steering that Mimics Prompting. ICML 2026. https://arxiv.org/abs/2605.03907
26. Sun, Baskaran, Wu, Sklar, Potts, Geiger (2025). HyperSteer: Activation Steering at Scale with Hypernetworks. https://arxiv.org/abs/2506.03292

*Reliability, size and SAEs*
27. Tan et al. (2024). Analysing the Generalisation and Reliability of Steering Vectors. NeurIPS 2024. https://arxiv.org/abs/2407.12404
28. Braun, Eickhoff, Krueger, Bahrainian, Krasheninnikov (2025). Understanding (Un)Reliability of Steering Vectors in Language Models. https://arxiv.org/abs/2505.22637
29. Da Silva, Sethuraman, Rajagopal, Hajishirzi, Kumar (2025). Steering off Course: Reliability Challenges in Steering Language Models. ACL 2025. https://arxiv.org/abs/2504.04635
30. Bhattarai, Alhanai (2026). Signatures of Steerability in Activation Space of Language Models. https://arxiv.org/abs/2609.14151
31. Billa (2026). Predicting Where Steering Vectors Succeed. https://arxiv.org/abs/2604.15557
32. Jeong (2026). Extracting and Steering Emotion Representations in Small Language Models: A Methodological Comparison. https://arxiv.org/abs/2604.04064
33. aygp-dr (2026). qwen3-steering (GitHub repository, Qwen3-0.6B). https://github.com/aygp-dr/qwen3-steering
34. Wu, Arora, Geiger, Wang, Huang, Jurafsky, Manning, Potts (2025). AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders. ICML 2025. https://arxiv.org/abs/2501.17148
35. Arad, Mueller, Belinkov (2025). SAEs Are Good for Steering – If You Select the Right Features. EMNLP 2025. https://arxiv.org/abs/2505.20063
36. Smith, Rajamanoharan, et al. (2025). Negative Results for SAEs On Downstream Tasks and Deprioritising SAE Research. Google DeepMind. https://www.lesswrong.com/posts/4uXCAJNuPKtKBsi28/negative-results-for-saes-on-downstream-tasks
37. Anthropic (2024). Evaluating Feature Steering: A Case Study in Mitigating Social Biases. https://www.anthropic.com/research/evaluating-feature-steering
38. Siu, Crispino, Park, Henry, Wang, Liu, Song, Wang (2025). SteeringSafety. https://arxiv.org/abs/2509.13450
39. Bas, Novak (2025). What Can We Actually Steer? A Multi-Behavior Study of Activation Control. https://arxiv.org/abs/2511.18284
40. Ong, Sim, Zhang, Sun (2026). Forecasting Side Effects of Activation Steering. https://arxiv.org/abs/2608.11227
41. Deshpande, Giyahchi, Padmanabhan, Akoglu, Rumshisky (2026). Activation Steering for Synthetic Data Generation: The Role of Diversity in Downstream Safety Detection. https://arxiv.org/abs/2605.28664

*Personality*
42. Frising, Balcells (2025). Linear Personality Probing and Steering in LLMs: A Big Five Study. https://arxiv.org/abs/2512.17639
43. Bhandari, Naseem, Nasim (2026). Do Personality Traits Interfere? Geometric Limitations of Steering in Large Language Models. Findings of ACL 2026. https://arxiv.org/abs/2602.15847
44. Bhandari, Fay, Selvaganapathy, Datta, Naseem, Nasim (2025). Activation-Space Personality Steering: Hybrid Layer Selection for Stable Trait Control in LLMs. https://arxiv.org/abs/2511.03738
45. Hoppe, Khachaturov, Mullins, Meng (2026). Controllable and Explainable Personality Sliders for LLMs at Inference Time. https://arxiv.org/abs/2603.03326
46. Deng, Tang, Yin, Yang, Zhao, Wen (2025). Neuron-based Personality Trait Induction in Large Language Models. ICLR 2025. https://arxiv.org/abs/2410.12327
47. Li, Liu, Wu, Dou, Wang, Lv, Wang, Zheng, Huang (2025). UPLex: Fine-Grained Personality Control in Large Language Models via Unsupervised Lexical Modulation (earlier title: UBPL). Findings of EMNLP 2025. https://arxiv.org/abs/2310.16582
48. Sun, Baek, Kim (2025). Personality Vector: Modulating Personality of Large Language Models by Model Merging. https://arxiv.org/abs/2509.19727
49. Li, Liu, Liu, Zhou, Diab, Sap (2025). BIG5-CHAT: Shaping LLM Personalities Through Training on Human-Grounded Data. ACL 2025. https://arxiv.org/abs/2410.16491
50. Dan, Zhou, Chen, Tian, He (2024/2025). P-React (v1 title: P-Tailor): Mixture of Specialized LoRA Experts for Personality Traits. https://arxiv.org/abs/2406.12548
51. Cui, Lv, Wen, Wang, Tang, Tian, Yuan (2023). Machine Mindset: An MBTI Exploration of Large Language Models. https://arxiv.org/abs/2312.12999
52. Jiang, Zhang, Cao, Breazeal, Roy, Kabbara (2024). PersonaLLM. Findings of NAACL 2024. https://arxiv.org/abs/2305.02547
53. Zhu et al. (2024). Personality Alignment of Large Language Models (PAPI, PAS). https://arxiv.org/abs/2408.11779
54. Handa, Wu, Koshiyama, Treleaven (2025). Personality as a Probe for LLM Evaluation: Method Trade-offs and Downstream Effects. https://arxiv.org/abs/2509.04794
55. Cho, Cheong (2025). Scaling Personality Control in LLMs with Big Five Scaler Prompts. https://arxiv.org/abs/2508.06149
56. Han, Kocielnik, Song, Debnath, Mobbs, Anandkumar, Alvarez (2025). The Personality Illusion: Revealing Dissociation Between Self-Reports & Behavior in LLMs. https://arxiv.org/abs/2509.03730
57. Courtis, Hu (2026). Mechanistic Personality Analysis of LLMs: Steering Personality via Latent Feature Interventions. https://arxiv.org/abs/2606.28770

*Trained interventions*
58. Wu, Arora, Wang, et al. (2024). ReFT: Representation Finetuning for Language Models. https://arxiv.org/abs/2404.03592
59. Stanford NLP. pyreft. https://github.com/stanfordnlp/pyreft
60. Stanford NLP. pyvene. https://github.com/stanfordnlp/pyvene
61. Yu et al. (2024). Neeko: Leveraging Dynamic LoRA for Efficient Multi-Character Role-Playing Agent. EMNLP 2024. https://arxiv.org/abs/2402.13717
62. Han, Guo, Jung, Yao, Zhang, Liu, Guo (2023). PersonaPKT: Building Personalized Dialogue Agents via Parameter-efficient Knowledge Transfer. SustaiNLP 2023. https://arxiv.org/abs/2306.08126

*Encoders and conditioning*
63. Zhong, Tang, Wang, Yin, Duan (2021). UserAdapter: Few-Shot User Learning in Sentiment Analysis. Findings of ACL 2021. https://aclanthology.org/2021.findings-acl.129/
64. Mireshghallah, Shrivastava, Shokouhi, Berg-Kirkpatrick, Sim, Dimitriadis (2021). UserIdentifier: Implicit User Representations for Simple and Effective Personalized Sentiment Analysis. https://arxiv.org/abs/2110.00135
65. Zhang, Wang, Yu, Xu, Zhang (2024). Personalized LoRA for Human-Centered Text Understanding. AAAI 2024. https://arxiv.org/abs/2403.06208
66. Perez, Strub, de Vries, Dumoulin, Courville (2018). FiLM: Visual Reasoning with a General Conditioning Layer. AAAI 2018. https://arxiv.org/abs/1709.07871
67. Li, Galley, Brockett, Spithourakis, Gao, Dolan (2016). A Persona-Based Neural Conversation Model. ACL 2016. https://arxiv.org/abs/1603.06155
68. Ravfogel, Prasad, Linzen, Goldberg (2021). Counterfactual Interventions Reveal the Causal Effect of Relative Clause Representations on Agreement Prediction. CoNLL 2021. https://arxiv.org/abs/2105.06965
69. Belrose, Schneider-Joseph, Ravfogel, Cotterell, Raff, Biderman (2023). LEACE: Perfect Linear Concept Erasure in Closed Form. NeurIPS 2023. https://arxiv.org/abs/2306.03819
70. Weller, Chang, MacAvaney, Lo, Cohan, Van Durme, Lawrie, Soldaini (2024/2025). FollowIR: Evaluating and Teaching Information Retrieval Models to Follow Instructions. https://arxiv.org/abs/2403.15246
71. Wang, Chen, Verberne (2026). RankSteer: Activation Steering for Pointwise LLM Ranking. https://arxiv.org/abs/2602.03422
72. Yin, Hay, Roth (2019). Benchmarking Zero-shot Text Classification: Datasets, Evaluation and Entailment Approach. EMNLP 2019. https://arxiv.org/abs/1909.00161
73. An, Kwak, Ahn (2018). SemAxis. ACL 2018. https://arxiv.org/abs/1806.05521

*Output-level methods*
74. Sanchez, Fan, Spangher, Levi, Ammanamanchi, Biderman (2023). Stay on Topic with Classifier-Free Guidance. https://arxiv.org/abs/2306.17806
75. Shi, Han, Lewis, Tsvetkov, Zettlemoyer, Yih (2023). Trusting Your Evidence: Hallucinate Less with Context-aware Decoding. https://arxiv.org/abs/2305.14739
76. Liu, Sap, Lu, Swayamdipta, Bhagavatula, Smith, Choi (2021). DExperts: Decoding-Time Controlled Text Generation with Experts and Anti-Experts. ACL 2021. https://arxiv.org/abs/2105.03023
77. Li et al. (2023). Contrastive Decoding: Open-ended Text Generation as Optimization. ACL 2023. https://arxiv.org/abs/2210.15097
78. An, Park, Jin, Han (2026). Steering Language Models Before They Speak: Logit-Level Interventions (SWAI). https://arxiv.org/abs/2601.10960
79. Holtzman, West, Shwartz, Choi, Zettlemoyer (2021). Surface Form Competition: Why the Highest Probability Answer Isn't Always Right. https://arxiv.org/abs/2104.08315
80. Zhao, Wallace, Feng, Klein, Singh (2021). Calibrate Before Use: Improving Few-Shot Performance of Language Models. ICML 2021. https://arxiv.org/abs/2102.09690

*Games and trait distributions*
81. Fleeson (2001). Toward a Structure- and Process-Integrated View of Personality: Traits as Density Distributions of States. JPSP 80(6), 1011–1027. http://simine.com/407/readings/Fleeson_2001.pdf
82. Fleeson Lab. Whole Trait Theory (summary of Fleeson & Jayawickreme 2015 and later). https://fleesonlab.wordpress.com/whole-trait-theory/
83. Han, Kwon, Gratch (2026). Personality Expression Across Contexts: Linguistic and Behavioral Variation in LLM Agents. https://arxiv.org/abs/2602.01063
84. Sakai, Yokoyama, Tateishi, Ichinose (2026). Effects of personality steering on cooperative behavior in large language model agents. Scientific Reports. https://pmc.ncbi.nlm.nih.gov/articles/PMC13462106/
85. Azimn. npc-steering-plus (GitHub repository). https://github.com/Azimn/npc-steering-plus

*Tools*
86. vgel. repeng. https://github.com/vgel/repeng
87. steering-vectors. https://github.com/steering-vectors/steering-vectors
88. NDIF. nnsight. https://github.com/ndif-team/nnsight
89. TransformerLens. https://github.com/TransformerLensOrg/TransformerLens
90. SAELens. https://github.com/jbloomAus/SAELens
91. Neuronpedia, Qwen3-1.7B. https://www.neuronpedia.org/qwen3-1.7b
92. Qwen Team (2026). Qwen-Scope: Turning Sparse Features into Development Tools for Large Language Models. https://arxiv.org/abs/2605.11887
93. Xu et al. (2025). EasySteer: A Unified Framework for High-Performance and Extensible LLM Steering. https://arxiv.org/abs/2509.25175
94. Miehling, Natesan Ramamurthy, Venkateswaran, et al. (2026). AI Steerability 360: A Toolkit for Steering Large Language Models. https://arxiv.org/abs/2603.07837
