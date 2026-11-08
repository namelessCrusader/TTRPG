# Why the small models pick wrong: a mechanistic-interpretability review

2026-09-29. A literature review for the mind lab (`lab/`). It goes with
roadmap items 98 to 102, `lab/results/methods/failures.md`,
`lab/results/methods/ablate.md` and `docs/character-models-plan.md`
(Workstream A).

**The question.** Small models choose a character's next action from a list.
We have measured HOW they fail. Which interpretability methods can show WHY?
Which fix does each one point to: training data, an inference change, or an
intervention inside the model?

**Sources.** We opened every source we cite, on 2026-09-29. For arXiv papers
we read the arXiv record (title, authors, date, abstract). For some we also
read the HTML text. For tools we read the repository page or the model card.
We quote a number only when we saw it in the source. Where we could not
check a detail, we say so. Numbers in brackets point to Section 7.

**What we measured (from our own files).**

| failure | where measured | size of effect |
| --- | --- | --- |
| Object/activity lure: the rope for "cut the rope" | `failures.md` | Wrong picks sit near the need's object in meaning at about 2x chance and near its verb at about 1.7x. |
| Affordance gap | `ablate.md` | With the verb alone, GLiNER2.5-Decide and bev-decider do as well as with the whole need (60-63% at 16 options). The object alone gives 20-23%. |
| Item priors | `failures.md` | Each model has favourite items. It picks them whatever the need, at 3-7x chance. |
| Distraction | `failures.md` | Words in unrelated scene text pull GLiNER2.5-Decide, bev-decider, Von and bge at 3-4x. |
| Position and order | `failures.md`, `report.md` | Qwen3-1.7B with a numbered list: first option 3.7x. Laya: first 2.0x, last 1.8x. Von listwise: last 2.8x. Laya keeps only about 40% of its picks when a 16-option list is reordered. |
| List length | `report.md` | From 4 to 128 options: Laya listwise 70% to 7%, Qwen3-1.7B numbered list 77% to 5%, GLiNER2.5-Decide 78% to 28%, bev-decider 75% to 35%, openjev 2B (one option at a time) 88% to 32%. |
| Wrong person | `failures.md` (people) | GLiNER2.5-Decide: the right action aimed at the wrong person is 2.2x chance among its wrong picks (30 wrong picks only). |

Each cell in those tables has about 60 cases. The 95% range is about plus or
minus 12 points. Small effects need paired designs (Section 5).

---

## 1. Summary

The findings that matter most for us, one line each.

1. **Test "knows" before "uses".** Language models learn object-action links
   that are written down. They are weak at links that follow from a
   property ("a sword has an edge, so it cuts") [6][7][9][13]. Our "verb alone
   = whole need" result and the unusual-tool drop (roadmap 101) fit this. So
   the first experiment is a direct quiz and a probe for affordances.
2. **A probe shows that information is present, not that it is used**
   [1][2][3][4]. Pair every probe with an intervention: erase the direction
   (amnesic probing, LEACE) [4][5], or patch it.
3. **Activation patching on minimal pairs is the workhorse.** A minimal pair
   is two inputs that differ in one thing. It works on every model type
   through plain PyTorch hooks. It needs only forward passes, so it fits the
   laptop. Use "symmetric token replacement" (swap one word for another), not
   added noise. Use a score difference as the measure [16][17].
4. **Attribution patching maps a whole model in two forward passes and one
   backward pass.** It is an approximation. Treat its map as a list of
   suspects, and confirm the top ones with real patching [18][19][28].
5. **For the listwise encoders, blocking attention between options is the
   most direct test of the list-length collapse.** It is also a candidate fix:
   bev-decider already uses such a mask by design [108], and it keeps 35% at
   128 options where Laya keeps 7%.
6. **ModernBERT, Laya's base, uses global attention in only every third
   layer. The other layers see only 128 tokens around each token** [43]. An
   option far from the need can reach the need only through the global
   layers. This is a concrete, testable suspect for Laya's position and
   length effects.
7. **Position bias in decoders has known parts and cheap fixes.** The parts:
   a prior on the option labels ("A", "1") [70], "correct letter" attention
   heads [31][32], MLP value vectors that favour the first option [77], and
   attention sinks and U-shaped attention [45][75]. The fixes: estimate and
   remove the label prior (PriDe) [70], average over orders [72], or change
   the attention so that options are processed as a set [73][74].
8. **An item prior can be removed with a per-option bias table.** It is
   estimated offline from many cases (batch calibration) [82][84]. It costs
   no model call at play time. The ablation says this gains little for
   bev-decider (+7 points) and GLiNER2.5-Decide (-2 points) at 16 options, so
   it is not their main problem.
9. **Decoders bind a person to their state with "binding ID" vectors**
   [64][66]. They retrieve bound things by position, by word and by a direct
   pointer. The positional route gets noisy in the middle of long lists [67].
   Interchange interventions test this directly. We found little work on
   binding in bidirectional encoders.
10. **Feature tools now exist for our exact decoders.** circuit-tracer
    transcoders exist for Qwen3-0.6B (their config names `Qwen/Qwen3-0.6B`,
    the checkpoint the lab uses) and for Qwen3-1.7B [54][58]. Qwen's own SAEs
    exist for Qwen3-1.7B-Base and Qwen3.5-2B-Base, the family openjev is built
    on [57]. We found none for DeBERTa or ModernBERT.
11. **SAE features are not better than linear probes for downstream tasks**
    [61][62]. Use SAEs to explore and to debug data. Do not make them the
    main diagnostic.
12. **Knowledge editing (ROME, MEMIT) is a poor way to add many affordance
    facts.** Edits do not spread to related facts [91]. Many edits in a row
    make the model forget [92]. Where a fact is stored does not tell you where
    to edit it [90]. Prefer fine-tuning on contrast sets and hard negatives.
13. **Fine-tuning mostly strengthens or wraps what a model already has**
    [68][93][94]. So hard-negative training will likely fix the lure only
    where the affordance knowledge is already present. Check with probes
    before and after training.
14. **Tools.** TransformerLens 4 has adapters for Qwen3, Qwen3.5 and BERT
    [101]. It has none for DeBERTa or the ModernBERT encoder (only a
    ModernBERT decoder); checked on main and dev on 2026-09-29. nnsight, pyvene and baukit work on any
    PyTorch module [102][103][106], so they also cover our custom packages
    (laya, gliner2, bev-decider). inseq does not support encoder-only
    classifiers [105].

---

## 2. Terms

| term | meaning |
| --- | --- |
| residual stream | The running vector at each token position that every layer reads from and adds to. |
| hidden state, activation | The vector a layer outputs for one token. |
| minimal pair | Two inputs that differ in one controlled thing, such as "cut the rope" and "cut the bread". |
| probe | A small classifier trained to read a property from hidden states. "Linear" means one weight layer. |
| control task | A fake task with random labels per word. A good probe does well on the real task and badly on the control task [1]. |
| selectivity | Real-task accuracy minus control-task accuracy [1]. |
| MDL probing | Probing that measures how many bits the probe needs to learn the labels. Fewer bits means the property is easier to read [2]. |
| amnesic probing, LEACE | Remove a property from the hidden states, then see if the model's output changes. LEACE is a closed-form linear eraser [4][5]. |
| activation patching | Run the model on input A. Replace one activation with its value from input B. See how much the output moves [15][16]. |
| clean, corrupted | The two inputs of a patch. "Clean" gives the behaviour we study; "corrupted" does not. |
| denoising, noising | Patch clean into corrupted (is this part SUFFICIENT to restore the behaviour?) or corrupted into clean (is it NECESSARY?) [16]. |
| causal tracing | The ROME form of patching: corrupt the subject with noise, then restore one state at a time [15]. |
| attribution patching | A gradient approximation of activation patching for all parts at once [18]. |
| path patching | Patching that changes only one path between two parts, not everything downstream [20][21]. |
| circuit | The small set of parts and connections that carry out one behaviour. |
| causal scrubbing | A test of a circuit hypothesis: resample every activation the hypothesis says does not matter, and check the behaviour stays [22]. |
| interchange intervention | Put an internal value from one input into a run on another input, to test if that value plays a hypothesised role [23]. |
| DAS | Distributed alignment search. It learns a rotation so that a small subspace, not single neurons, carries a variable [24]. |
| logit lens, tuned lens | Read an intermediate layer through the output layer (logit lens [34]) or through a trained per-layer map (tuned lens [35]). |
| attention knockout | Block attention from some tokens to others and measure the output change [30]. |
| SAE | Sparse autoencoder. It rewrites an activation as a sum of a few "features" from a large dictionary [47][49]. |
| transcoder | Like an SAE, but it predicts an MLP's output from its input. Cross-layer transcoders (CLT) write to many later layers [50][52]. |
| crosscoder | A dictionary shared across layers or across two models. Used to compare models ("model diffing") [51]. |
| attribution graph | A graph of features and their direct effects on each other for one prompt, built from transcoders [52]. |
| binding ID | A vector that marks which attribute belongs to which entity [64]. |
| calibration | A fixed change to scores that removes a known bias, such as a prior over options [80]. |
| PMI | Pointwise mutual information. Here: an option's score with the need, minus its score without it [81]. |
| steering vector | A vector added to activations at run time to push the model's behaviour [99][100]. |
| ReFT | Training a small intervention on hidden states instead of the weights [98]. |
| hard negative | A wrong option that looks right. A lure is a hard negative. |
| contrast set | Test cases made by small edits that change the right answer [96]. |
| logit difference | The score of the right option minus the score of the lure. The recommended patching measure [16]. |

---

## 3. Methods

Our models fall into four types. The type decides which methods apply.

| type | our models | what gives the score |
| --- | --- | --- |
| D: decoder, log-probability | Qwen3-0.6B, Qwen3-1.7B | Next-token log-probability of the option's words. The listwise form reads a numbered list and scores the label. |
| C: decoder cross-encoder | openjev 0.8B / 2B (Qwen3.5 backbone) [110] | A 3-way NLI head (entailment, neutral, contradiction) on the last token. One pass per option. |
| M: decoder, masked listwise | bev-decider-0.4B (first 20 layers of Qwen3-0.6B, merged LoRA, 2-layer head) [108] | All options in one pass. Each option sees only the query and itself. |
| E: bidirectional encoder, listwise | Laya (ModernBERT-large, 512 tokens) [107], Von (ModernBERT-large), GLiNER2.5-Decide (DeBERTa-v3-large, 340M) [109] | All options in one pass. Options can attend to each other. A head reads each option's marker or label tokens. |

Three facts about the architectures matter below.

- **DeBERTa** (GLiNER2.5-Decide's base) gives each token two vectors, one for
  content and one for relative position, and attends with both [44]. So
  position enters every attention score, not only the input.
- **ModernBERT** alternates attention. "Every third layer employs global
  attention ... the remaining layers use a 128 token, local sliding window
  attention" [43]. The large model has 28 layers [43].
- **Qwen3.5-2B** has 24 layers in the pattern 6 x (3 x Gated DeltaNet, then 1
  x gated attention) [111]. Gated DeltaNet is a linear-attention layer with a
  running state, not a normal attention map. Attention-map methods apply only
  to its 6 full-attention layers.

### 3.1 Minimal pairs and contrast sets (the base of every experiment)

- **Question.** Which single change in the input changes the pick?
- **How.** Build pairs that differ in one thing: the object ("cut the rope" /
  "cut the bread"), the position of the right option, the filler text, or
  which person has which state. Contrast sets are the same idea used as a
  test set; models score up to 25% lower on them [96].
- **Needs.** Our case generator (`lab/methods/cases.py`). No GPU beyond normal
  scoring.
- **Fit.** All types. It is already how `ablate.py` works.
- **Fix it points to.** The pairs become training data (counterfactually
  augmented data) [95]. Training on original plus edited examples reduced
  reliance on spurious signals in [95].

### 3.2 Probing: linear probes, control tasks, MDL

- **Question.** Is a property readable from the hidden states? For us: does
  the state for "the chisel" encode "can cut"?
- **How.** Collect hidden states per layer for each item name. Train a linear
  classifier per affordance. Hold out whole ITEMS, not only item-need pairs,
  so the probe cannot memorise item identity. Report selectivity against a
  control task [1]. Report MDL, which separates trained from random
  representations better than accuracy does [2].
- **What it cannot say.** A probe shows the information is there. It does not
  show the model uses it [3][4]. Amnesic probing found that probing accuracy
  was not correlated with task importance in BERT [4].
- **Needs.** One forward pass per item per template. Probes train on the CPU.
- **Fit.** All types. For D, C and M, read the last token of the item name.
  For E, read the item's tokens inside the list.
- **Prior work on our property.** Neural representations "only learn
  associations that are explicitly written down"; they miss affordances that
  follow from properties [6]. PIQA: humans 95%, large pretrained models 77%,
  partly from reporting bias [7]. Text2Afford (15 affordance classes): weak on
  uncommon affordances; few-shot fine-tuning helped [9]. Affordance20Q (2026):
  models about 20 points below humans when they must infer affordances from
  physical properties [13]. Models' knowledge of everyday objects' parts is
  fragmented and inconsistent (54-59% accurate) [12]. Object substitution
  needs utility, context and physical state [10]. MacGyver: models propose
  physically infeasible uses [11]. PROST: option order changes the answers,
  and scale helps little [8]. Entity state is linearly readable in BART and
  T5, and editing it changes generation [14].
- **Fix it points to.** If the affordance is encoded but not used: an
  inference change (say the property in the option) or light training (ReFT,
  LoRA) that routes existing knowledge. If it is not encoded: training data
  with the facts.

### 3.3 Erasure: amnesic probing and LEACE

- **Question.** Does the model's pick DEPEND on a property?
- **How.** Remove the property's linear direction from the hidden states at
  one or all layers (LEACE gives a closed-form eraser that stops every linear
  classifier from reading it [5]). Re-score. If accuracy on "cut" needs falls
  and other needs do not, the model uses "can cut".
- **Needs.** The probe data from 3.2. One extra forward pass per case with a
  hook.
- **Fit.** All types. Cheap.
- **Fix it points to.** A confirmed direction can be strengthened at run time
  (a steering vector, 3.12) or targeted in training.

### 3.4 Activation patching and causal tracing

- **Question.** Which layer and which token position carry the information
  that decides the pick?
- **How.** Take a minimal pair. Run the clean input and the corrupted input.
  For each layer and each span of tokens (need verb, need object, each option,
  filler), copy the activation from one run into the other. Measure the
  change in the score difference (right option minus lure). ROME's causal
  tracing found that middle-layer MLPs at the last subject token mediate
  factual recall in GPT [15].
- **Good practice [16][17].** Replace tokens symmetrically (a word for a word
  of the same length); do not add Gaussian noise, which is "highly sensitive
  to the noise level" [16]. Use the logit difference, not a probability,
  which saturates [16]. Run both directions: denoising tests sufficiency,
  noising tests necessity, and they can find different parts [16].
- **Needs.** Forward passes: layers x spans per pair. For Laya: 28 layers x
  about 6 spans = about 170 passes per pair, at about 20 ms each. That is
  seconds per pair.
- **Fit.** All types, through hooks. For Qwen, turn off the KV-cache warm
  start while patching, so every position is recomputed. For openjev's
  Gated DeltaNet layers, patch the layer output; the running state inside is
  harder to reach (not checked in any tool).
- **Fix it points to.** A located layer tells us where to put a steering
  vector or a ReFT intervention (3.12). It does NOT reliably tell us where to
  edit weights [90].

### 3.5 Attribution patching (AtP, AtP*, EAP-IG)

- **Question.** The same as 3.4, for every head, neuron or edge at once.
- **How.** One forward pass on each input and one backward pass. Estimate each
  patch as (clean - corrupted activation) x gradient [18]. AtP* fixes two
  causes of false negatives [19]. EAP-IG (with integrated gradients) finds
  more faithful circuits than plain edge attribution [28].
- **Limits.** It works poorly for big activations such as whole residual
  streams, for saturated attention and for strong LayerNorm effects [18]. Its
  author calls it "exploratory, hypothesis-generating" [18].
- **Needs.** A backward pass: gradients of activations, not of weights. For
  Qwen3-0.6B and the 400M encoders this should fit 6 GB with short inputs (we
  have not measured it).
- **Fit.** All types.
- **Fix it points to.** Candidate heads and neurons for a masking fix
  (UniBias-style [78]) once confirmed by real patching.

### 3.6 Path patching, automatic circuit discovery, causal scrubbing

- **Question.** What is the full circuit, and is our story about it complete?
- **How.** Path patching changes one path between two parts [21]. It traced a
  26-head circuit in GPT-2 small (the IOI circuit) [20]. ACDC automates the
  search; it recovered a known circuit with 68 of 32,000 edges [27]. Causal
  scrubbing tests a finished hypothesis by resampling everything it says is
  irrelevant [22].
- **Needs.** Many forward passes. A clear hypothesis first.
- **Fit.** Possible for Qwen3-0.6B and the encoders. Costly in time, not in
  memory.
- **When to use.** Late. Only after 3.4 and 3.5 have found a strong, narrow
  effect worth a full circuit.

### 3.7 Interchange interventions and DAS

- **Question.** Does a hidden VARIABLE exist, such as "which option fits the
  need" or "which person is in danger"? Where is it stored?
- **How.** Write the variable into a small causal model. Swap the internal
  value between two inputs. If the output follows the swap, the model has
  that variable there [23]. DAS learns the subspace by gradient descent, so
  the variable need not sit in single neurons [24]. Boundless DAS scaled this
  to a 7B model [25]. pyvene implements both [103].
- **Warning.** A learned subspace can change the output through a "dormant
  parallel pathway", not the real one [26]. Check it against plain patching.
- **Needs.** A few hundred minimal pairs. Training a rotation: small, fits the
  laptop for sub-1B models.
- **Fit.** All types. Best for binding and position (Section 4.6, 4.7).
- **Fix it points to.** Interchange intervention TRAINING (IIT) trains the
  model so that a chosen subspace holds a chosen variable; it beat multi-task
  learning and data augmentation on three tasks [33]. A found subspace can
  also be projected out at run time.

### 3.8 Causal mediation (direct and indirect effects)

- **Question.** How much of an effect goes through a given part, and how much
  goes around it?
- **How.** Split the total effect of an input change into a direct effect and
  indirect effects through chosen parts. On gender bias this found the effect
  concentrated in few parts [29].
- **Fit.** All types; it was done on pre-trained Transformers including
  encoders [29].
- **Use for us.** Item priors and distraction: is the pull carried by a few
  heads (maskable) or spread out (needs training)?

### 3.9 Logit lens, tuned lens, Patchscopes

- **Question.** At which layer does the model "decide", and what does a middle
  layer already predict?
- **How.** Pass an intermediate state through the final output layer (logit
  lens [34]) or through a trained per-layer map (tuned lens, more reliable
  [35]). Patchscopes generalise this: patch a state into a second prompt and
  let the model describe it [36].
- **Fit.** D only as published: it needs an LM head. For C, M and E we can
  apply the classifier head to each layer, or train a per-layer map to the
  final score (our adaptation of the tuned lens; not in the literature we
  read).
- **Use for us.** (a) Is "cut" a top prediction at the knife's tokens (a quick
  affordance check)? (b) When does a first-option bias appear? MLP value
  vectors that favour "A" were found this way in GPT-2 [77].

### 3.10 Attention analysis and attention knockout

- **Question.** Which tokens does the decision read from?
- **Limits of looking at attention.** Attention weights often do not match
  gradient importance, and different weights can give the same output [37].
  Whether attention can explain depends on the definition and the test [38].
  Information mixes across layers, so raw maps mislead; attention rollout and
  flow track it better [39]. Some BERT heads do track syntax and coreference
  [40].
- **Knockout, not looking.** Block attention from one set of tokens to another
  and measure the output change. This found where facts flow into the
  prediction in decoders [30].
- **Fit.** D and M: edit the attention mask per layer. E: edit the
  bidirectional mask. ModernBERT's local layers already restrict attention,
  so knockout matters most in its global layers. C: only the 6 full-attention
  layers of Qwen3.5-2B have a mask to edit.
- **What cross-encoders do inside.** A study of a retrieval cross-encoder found
  it computes term-frequency and IDF-like signals in early-to-middle layers
  and combines them like BM25 (a classic word-matching score) [41]. NLI
  models trained on MNLI, including BERT, rely on word overlap [42]. Both
  predict an object lure: the option that shares words or meaning with the
  need scores high.
- **Fix it points to.** A mask that stops options attending to each other
  (like bev-decider [108]) or to filler.

### 3.11 SAEs, transcoders, crosscoders, attribution graphs

- **Question.** Which human-readable features are active on a lure, and how do
  they connect to the score?
- **What they are.** An SAE rewrites an activation as a few features from a
  large dictionary; features are often more interpretable than neurons
  [47][49]. Feature steering changes behaviour (the "Golden Gate" example)
  [48]. Transcoders replace an MLP with a sparse, wider MLP, which makes
  circuits easier to trace [50]. Cross-layer transcoders and attribution
  graphs build a per-prompt graph of feature-to-feature effects [52].
  Crosscoders share one dictionary across layers or across two models [51].
- **Limits.** Attribution graphs freeze attention, so they cannot explain how
  attention patterns are chosen [52]. That is exactly where position bias and
  list interference live. On an 18-layer model the largest CLT matched the
  model's output on about 50% of prompts [52]. In the Claude 3.5 Haiku study
  the method gave satisfying insight for about 25% of prompts [53]. DeepMind's
  team found SAE-based probes generalise worse than dense linear probes, and
  deprioritised SAE research [61]. AxBench found SAEs "not competitive" with
  prompting, fine-tuning or difference-in-means for steering and detection
  [62]. SAEs were useful for finding spurious correlations in a dataset [61].
- **What exists for our models.**
  - circuit-tracer supports Qwen3 at 0.6B, 1.7B, 4B, 8B and 14B with per-layer
    transcoders, and has TransformerLens and nnsight backends [54]. The
    transcoders for 0.6B (`mwhanna/qwen3-0.6b-transcoders-lowl0`) list
    `Qwen/Qwen3-0.6B`, MLP input to MLP output, 28 layers [58]. A 1.7B set
    exists; we did not open its config. The 0.6B repository is 108.8 GB with
    its feature files [58]; check what circuit-tracer downloads before
    starting.
  - BluelightAI released cross-layer transcoders for Qwen3-0.6B-Base and
    Qwen3-1.7B-Base (December 2025) [59]. They are for the base models, not
    the post-trained ones we use.
  - Qwen-Scope (Qwen team, May 2026) released residual-stream TopK SAEs for
    all layers of Qwen3-1.7B-Base (28 layers, 32K features, 50 or 100 active)
    and Qwen3.5-2B-Base (24 layers, 32K features), among 14 sets [57].
  - SAEs trained on a base model usually transfer to its chat model, with
    exceptions (Gemma v1 2B), and a base SAE can be fine-tuned cheaply [60].
    Whether they transfer to our post-trained Qwen3-1.7B, and to openjev's
    NLI fine-tune, is not known. Measure reconstruction and the change in
    the score first.
  - Neuronpedia hosts Qwen3-4B transcoders, not the smaller sizes [63].
  - The large open SAE suites cover other families: Gemma 2 (Gemma Scope,
    all layers of 2B and 9B) [55] and Llama-3.1-8B (Llama Scope, 256 SAEs)
    [56]. They show what a full release looks like; none of them is for our
    models.
  - We found no public SAEs or transcoders for DeBERTa or ModernBERT.
- **Fit.** D: yes (circuit-tracer, Qwen-Scope). C: possibly (Qwen-Scope
  3.5-2B, transfer unknown). M: maybe (bev-decider is the first 20 layers of
  Qwen3-0.6B plus a merged LoRA; transfer unknown). E: we would have to train
  our own SAE, which belongs on the other machine.
- **Fix it points to.** A feature that carries the lure can be clamped at run
  time, or its examples added to training data. Given [61][62], confirm any
  feature with a plain probe and patching first.

### 3.12 Model diffing

- **Question.** What did fine-tuning change inside the model?
- **How.** Crosscoders find features shared by two models and features only
  one has [51]. A simpler route: patch activations from the fine-tuned model
  into the original (cross-model activation patching, CMAP) [68]; or re-run
  the probes of 3.2 on both.
- **What the literature says fine-tuning does.** Fine-tuning improved an
  existing entity-tracking circuit rather than building a new one [68]. It
  often learns a thin "wrapper" over existing capabilities that a little
  more training can undo [93]. DPO "bypassed" toxic capabilities rather than
  removing them [94].
- **Fit.** All types for probes and CMAP. Crosscoders need training off the
  laptop.
- **Use for us.** After training on hard negatives: did the model learn
  affordances (probe gains on held-out items), or a wrapper that down-weights
  the object (no probe gain, and a fragile fix)?

### 3.13 Score corrections: calibration

These are not interpretability methods. They are cheap fixes that follow from
a mechanism, and they are measured the same way.

- **Contextual calibration.** Score a content-free input ("N/A") and correct
  toward uniform; up to 30.0 points better on GPT-3 [80]. Our `blank` and
  `calibrated` columns in `ablate.md` are this idea.
- **Domain-conditional PMI.** Divide an option's probability by its
  probability in a task-only context, to stop options with common wording
  winning [81]. For us, the object-only and verb-only runs give such contexts.
- **Domain-context calibration.** Build the content-free input from random
  in-domain words, not an empty string; up to 37% Macro-F1 better [84].
- **Batch calibration.** Estimate the bias from the batch of test inputs
  themselves; zero-shot and inference-only [82].
- **Prototypical calibration.** Fit clusters of scores per label and assign
  by cluster [83].
- **Noisy channel.** Score P(situation | option) instead of P(option |
  situation); more stable with imbalanced labels [85]. It costs one pass per
  option over the situation.
- **Fit.** All types. Offline from kept scores, no GPU.
- **Latency.** A per-option bias table adds no model call at play time. Any
  method that needs extra passes per decision (permutations, channel scoring)
  multiplies play-time cost; see the latency budget.

### 3.14 Steering vectors and representation fine-tuning (ReFT)

- **Question.** Can a fixed or small learned change to the activations fix the
  failure?
- **How.** CAA takes the mean difference of residual activations between
  contrasting examples, and adds it at run time [99]. ActAdd does this from
  one prompt pair [100]. ReFT trains a low-rank intervention on hidden
  states; LoReFT was "15x-65x more parameter-efficient" than LoRA [98]. AxBench
  proposes a rank-1 ReFT that did well on steering and detection [62].
- **Fit.** All types (pyvene has a ReFT library [98][103]). Steering is
  inference only. ReFT training on sub-1B models may fit 6 GB; long runs go to
  the other machine.
- **Fix it points to.** A direct inference-time or training fix at the layer
  that 3.4 found.

### 3.15 Knowledge editing and knowledge neurons

- **What they are.** ROME edits one fact with a rank-one change to a
  middle-layer MLP [15]. MEMIT edits thousands of facts in GPT-J and
  GPT-NeoX [87]. AlphaEdit projects the change into the null space of kept
  knowledge; 36.7% better on average across editing methods [88]. Knowledge
  neurons are MLP neurons in BERT found by attribution; changing them
  updated or erased a fact [89].
- **Limits.** Where causal tracing locates a fact does not predict which
  layer is best to edit [90]. Edits do not propagate to related facts; an
  in-context baseline did best [91]. Sequential edits cause gradual, then
  sudden, forgetting [92].
- **Fit to our problem.** Poor. Affordances are many-to-many ("anything with
  an edge cuts rope"), not one subject-relation-object triple. The published
  work is on decoders (and BERT cloze for knowledge neurons), not on our
  classifier heads. We did not find editing work on NLI cross-encoders or
  listwise encoders.
- **Better route.** Put the facts in the input (the "in-context" baseline
  that won in [91]), or fine-tune on contrast sets.

### 3.16 Tools that fit a 6 GB GPU

| tool | what it does | our model types | notes |
| --- | --- | --- | --- |
| TransformerLens 4 (TransformerBridge) [101] | Hooks, caching, patching helpers | D (qwen3 adapter), C (qwen3_5 adapter: "Hybrid linear-attention (GatedDeltaNet) + full-attention"), E: bert only; no deberta or ModernBERT-encoder adapter (only modernbert_decoder), checked on main and dev on 2026-09-29 | We did not check whether it wraps sequence-classification heads or our custom packages (laya, gliner2, bev-decider). |
| nnsight [102] | Read and write any internal value in a forward pass | All ("any PyTorch model") | Local, or remote on NDIF. The safest choice for custom packages. |
| pyvene [103] | Interventions, interchange interventions, DAS, trainable interventions; ReFT is built on it | All ("various PyTorch modules") | The tool for Sections 3.7 and 3.14. |
| baukit [106] | `Trace`, `TraceDict`: hook a named layer | All (plain PyTorch hooks) | Minimal. Enough for probes and patching. |
| SAELens [104] | Train and load SAEs | Inference "works with any PyTorch-based model" | Training for encoders is not described. |
| circuit-tracer [54] | Attribution graphs, feature interventions | D (Qwen3 0.6B to 14B transcoders) | TransformerLens backend is faster; nnsight backend covers more models. Colab (15 GB) was their target for Gemma-2-2B; 0.6B should be smaller, but we have not measured it. |
| Neuronpedia [63] | Web UI for features and graphs; open source | Qwen3-4B hosted, not our sizes | Local install asks for "at least 16GB RAM". |
| inseq [105] | Input attribution (integrated gradients, occlusion, contrastive attribution) | D only ("ForCausalLM" and encoder-decoder) | Not for encoder-only classifiers. Contrastive attribution ("why the lure and not the tool?") fits Qwen well. |

### 3.17 The method table

Cost is per minimal pair unless noted. "Fits" means on our 6 GB laptop GPU,
one model at a time, short runs.

| method | question it answers | D | C | M | E | cost | fits |
| --- | --- | --- | --- | --- | --- | --- | --- |
| minimal pairs, contrast sets | which input change moves the pick | yes | yes | yes | yes | 2 scorings | yes |
| behavioural affordance quiz | does the model know "X can do V" when asked directly | yes | yes | yes | yes | 1 scoring per item-verb | yes |
| linear probe + control task + MDL | is the property readable, and where | yes | yes | yes | yes | 1 pass per item; CPU probes | yes |
| amnesic probing, LEACE | is the property used | yes | yes | yes | yes | 1 extra pass | yes |
| activation patching | which layer and position decide | yes | yes (layer outputs) | yes | yes | layers x spans passes | yes |
| attribution patching, EAP-IG | same, for every part at once (approximate) | yes | yes | yes | yes | 2 forward + 1 backward (IG: more) | likely (not measured) |
| path patching, ACDC, causal scrubbing | the full circuit, and is the story complete | yes | partly | yes | yes | many passes | slow but yes |
| interchange interventions, DAS | is there a variable, in which subspace | yes | yes | yes | yes | small training | yes (sub-1B) |
| causal mediation | how much of the effect goes through a part | yes | yes | yes | yes | like patching | yes |
| logit / tuned lens, Patchscopes | when the decision forms | yes | adapted | adapted | adapted | 1 pass; tuned lens needs training | yes |
| attention knockout | which token-to-token reads matter | yes | 6 full-attention layers only | yes | yes | 1 pass per knockout | yes |
| SAEs, transcoders, attribution graphs | which readable features drive a pick | yes (Qwen3) | maybe (Qwen3.5 SAEs, transfer unknown) | maybe | no public dictionaries | graph per prompt; SAE training is heavy | inference yes; training no |
| model diffing (CMAP, probes, crosscoders) | what fine-tuning changed | yes | yes | yes | yes | passes on two models | yes (crosscoders: no) |
| calibration family | remove a prior or form bias | yes | yes | yes | yes | offline | yes (no GPU) |
| steering, ReFT | a small fix at a located layer | yes | yes | yes | yes | inference or small training | yes (short runs) |
| ROME, MEMIT, AlphaEdit | edit stored facts | yes (as published) | untested | untested | knowledge neurons: BERT cloze only | per-edit solves | yes, but poor fit |

### 3.18 Which method tells us what, per failure mode

| method | object lure | affordance gap | item priors | distraction | position and order | list length | wrong person |
| --- | --- | --- | --- | --- | --- | --- | --- |
| minimal pairs | swap the object only | swap the tool for an unusual one | same option, many needs | filler vs neutral filler | rotate the list | same case, 4 vs 128 options | swap the persons' states |
| affordance quiz | — | knows vs uses (main test) | — | — | — | — | — |
| probes + control | is "similar to the object" a strong direction? | is "can cut" encoded, per layer | is a "favourite" direction in the option state? | — | is position linearly readable in option states? | does the right option's state degrade with list size? | is "who is in danger" readable at each name? |
| erasure (LEACE) | remove object similarity; does the lure lose? | remove "can cut"; does accuracy on cut needs fall? | remove the prior direction | — | remove the position direction | — | — |
| activation patching | where the object's pull enters | where the tool's fit is computed | where the prior is added | where filler is read | where position enters the score | where the long list corrupts the right option | where the binding is stored |
| attribution patching | candidate heads for the pull | candidate MLPs for the fit | candidate heads/FFN vectors | candidate heads reading filler | candidate "correct label" heads | candidate cross-option heads | candidate heads for retrieval |
| DAS, interchange | a "matches the object" variable | a "fits the need" variable | — | — | an ordering-ID subspace | — | binding-ID vectors (main test) |
| lens | does the lure's word appear early? | does "cut" appear at the knife? | is the favourite high at every layer? | — | when does the first-option bias appear? | — | — |
| attention knockout | block option-to-object reads | — | — | block option-to-filler reads | block reads near the edges | block option-to-option reads (main test) | block option-to-name reads |
| SAEs, graphs (Qwen) | features for the rope as object | affordance features, if any | features for favourite items | features fired by filler | limited: attention is frozen | limited: attention is frozen | limited |
| calibration | object-conditional PMI | — | per-option bias table (main fix) | — | PriDe label prior | — | — |
| steering, ReFT | subtract a lure direction | add a "fits" direction | subtract a prior direction | — | subtract position direction | — | — |
| editing | — | poor fit | — | — | — | — | — |

---

## 4. Failure mode by failure mode

Each part: what we know, the methods that would diagnose it, the concrete
experiment (inputs, what to measure, what each result means), and the fix it
points to.

The measure in every patching experiment is the score difference S = score(a
right option) - score(the model's wrong pick), from the kept scores. For
decoders scored by log-probability, use the sum over the option's
tokens as well as the mean, because the mean is what `one_mean` uses and it
lowered accuracy.

### 4.1 Object/activity lure (the rope, the apple, the lamp oil, the padlock)

**What we know.** Wrong picks sit near the need's object in meaning at about
2x chance. In the ablation, the object alone gives 20-23% at 16 options (6%
chance), so the object carries real signal about the task. But in
GLiNER2.5-Decide and bev-decider the object pull explains only 12-22% of
failures. The literature predicts this lure in two of our model types:
cross-encoders compute a word-matching score like BM25 [41], and NLI models
lean on word overlap [42].

**Diagnose with.** Minimal pairs, activation patching, attention knockout,
and, for Qwen, contrastive attribution (inseq) and attribution graphs.

**Experiment A (cheap, all models).** Keep the verb and the list. Change only
the object: "cut the rope" / "cut the bread" / "cut the net". Keep lists
where the lure (the rope) appears in all three.
- Measure: the lure's rank and S in each version.
- If the lure drops when the object changes: the pull is object matching.
- If the lure stays high: it is an item prior (4.3), not the object.

**Experiment B (one model: Qwen3-0.6B, then Laya).** Patching on the pair
("cut the rope", "cut the bread"), with the rope option present in both.
Denoise: copy states from the "bread" run into the "rope" run, one layer and
one span at a time (need-object span, lure-option span, right-option span,
last token).
- Measure: how much S recovers.
- If recovery comes from the need-object span in early layers: the lure is
  surface matching, read directly from the words. A mask or a rewording can
  fix it.
- If recovery comes from middle layers at the last token (D) or the option
  marker (E): the model compares meanings late. Hard-negative training is
  needed.

**Experiment C (encoders).** Attention knockout: stop option tokens reading
the need-object tokens, in global layers only, then in all layers.
- If the lure loses more than the right tool: the pull flows by attention from
  the object. If both lose equally: the object is also needed to find the
  right tool (the 20-23% object-only score suggests it is).

**Fix it points to.**
- Inference, no training: object-conditional PMI. Score = full score minus
  lambda x object-only score, with lambda fitted on half the cases. The
  object-only scores are already kept by `ablate.py`. This is the
  domain-conditional PMI idea [81] applied to the object. It is untested.
- Training: hard negatives that are the need's own object and its near
  neighbours, and contrast sets that change the object but not the tool
  [95][96][97].

### 4.2 Affordance knowledge gap (what a thing can do)

**What we know.** The verb alone does as well as the whole need, so the
object is not the main problem. Every model loses 10 to 42 points when the
obvious tool is replaced by an unusual one that still works (roadmap 101).
Several all-model failures need a property-to-use step: the sword to cut the
rope, the hairpin for the lock, the mirror to signal a ship. This matches
[6]: models learn links that are written down and miss those implied by
properties.

**Diagnose with.** A behavioural quiz, probes with control tasks and MDL,
erasure, and the logit lens on Qwen.

**Experiment A: the affordance quiz (cheap, every model).** Build an item x
verb table from the item pools in `cases.py` and the verbs of the 38 needs in
`ablate.py`. Ask each pair alone, in each model's own form: Qwen, "Can
you cut a rope with a sword? Answer:" scored yes vs no; openjev, the premise
"Kael has a sword." and the hypothesis "Kael can cut a rope with it."; Laya
and GLiNER, a yes/no question.
- Measure: AUC against our labels, split by obvious vs unusual tools.
- If the quiz is right where the list pick was wrong: the model KNOWS, but
  does not USE it in the list. Go to Experiment C and fixes of kind (a).
- If the quiz is also wrong, mostly on unusual tools: the knowledge is
  missing. Fixes of kind (b).

**Experiment B: probes (Qwen3-0.6B first, then Laya).** Collect hidden states
per layer for each item in a neutral frame ("In the storeroom there is a
chisel."). Train one linear probe per affordance ("cuts", "holds water",
"gives light", "is long") and one per property ("has an edge", "is hollow",
"is hard"). Split by ITEM. Add a control task with random labels per item
[1] and report MDL [2]. Baselines: layer 0, and bge-small vectors.
- If properties are readable but affordances of unusual items are not: the
  model has the parts but does not compose them. Fix: say the property in the
  option ("the chisel (sharp edge)"), or train on property-to-use
  examples.
- If nothing beats the control task: the knowledge is absent at this size.

**Experiment C: erasure.** Erase the "cuts" direction with LEACE [5] at the
best probe layer. Re-run the cut needs and five other needs.
- If only cut needs fall: the model uses this direction. It can be
  strengthened (steering, ReFT).
- If nothing falls: the direction exists but the pick does not use it. The
  decision is being made from something else, such as word association.

**Fix it points to.**
- (a) Knowledge present but unused: an inference change (properties in the
  options; ask the quiz first and feed its answer as a feature, which costs
  extra passes), or light training (ReFT or LoRA on hard negatives).
- (b) Knowledge absent: training data. A large model writes item x verb x
  property facts; the sim checks what it can. Hold out items to test
  generalisation, using the TRAIN and TEST pools. Text2Afford saw gains from
  few-shot fine-tuning [9].
- Not recommended: ROME or MEMIT for this (Section 3.15).

### 4.3 Item priors (favourites picked whatever the need)

**What we know.** Favourites are picked at 3-7x chance. They are measured on
one half of the cases and counted on the other half. Removing the blank-need
score gains little: bev-decider +7 points, GLiNER2.5-Decide -2 points, at 16
options. For Qwen, `one_cal` beats `one` by 0 to 10 points at most sizes; it
loses 8 points for Qwen3-1.7B at 64 options.

**Diagnose with.** The kept scores, a probe on the option state, lens, and
UniBias-style search for biased parts [78].

**Experiment A (no GPU).** For each option name, compute its mean score over
all cases where it is wrong (a batch-calibration estimate [82]). Fit on half
the cases, test on the other half. Also try domain-context calibration with
random in-domain words as the "blank" need [84].
- If accuracy rises and the favourite lift falls: a fixed per-option bias
  table fixes it at zero play-time cost.
- If not: the prior depends on context (for example, firewood wins for any
  need about warmth or fire), which is closer to the lure of 4.1.

**Experiment B (one model).** Regress the blank-need score on the option's
hidden state at each layer. Then erase that direction and re-score.
Attribution patching between a favourite and a non-favourite at the same
position can name the heads or MLP vectors that add the bonus. UniBias masks
such parts at inference [78]; in GPT-2, first-option bias sat in MLP value
vectors and was fixed by editing them [77].

**Fix it points to.** A per-option bias table first. If the prior sits in a
few parts, mask them. If it is spread out, rebalance the training data so no
item is the answer more often than others.

### 4.4 Distraction by unrelated text

**What we know.** Words in the filler pull four models at 3-4x. Irrelevant
context also lowers accuracy in large models, and an instruction to ignore it
helps them [86].

**Diagnose with.** Minimal pairs, attention knockout, activation patching on
the filler span.

**Experiment.** Three versions of each long-situation case: the original
filler; a neutral filler of the same length with no item words; no filler.
- Measure: S, and how often the pick shares a word with the filler.
- Then knock out attention from option tokens to filler tokens.
- If knockout gives back most of the loss: the filler is read by attention
  from the options. A mask fix is possible for the encoders.
- If knockout does little but neutral filler helps: the filler changes the
  need's own representation (the whole context shifts). The fix is in the
  input.

**Fix it points to.** The sim controls the input, so the cheapest fix is to
give the model only relevant facts (the `view` formulation of roadmap 98).
Then: train with distractor text; or mask options from filler.

### 4.5 Position bias, order sensitivity, shared prefixes

**What we know.** Qwen3-1.7B with a numbered list favours the first option
(3.7x). Laya favours first (2.0x) and last (1.8x); Von favours last (2.8x).
Laya keeps only about 40% of its picks when a 16-option list is reordered.
bev-decider and openjev are order-invariant by design [108][110].
GLiNER2.5-Decide shows no first or last lift (1.0 and 0.8), although DeBERTa
puts relative position into every attention score [44]; it is a useful
contrast case for Laya.
Averaging over 4 orders helped Laya little (`laya_en:perm4`: 33% vs 27% at 16
options, 8% vs 7% at 128).

**Known mechanisms in decoders.** A prior on the option IDs ("A", "1"),
not the place, drives much of the selection bias [70]. Models differ in
whether they can bind a symbol to its option at all [79]. "Correct letter"
heads read an "Nth item in an enumeration" feature [31]. Middle-layer
attention heads decide the letter and later layers amplify it [32]. First-
option bias in GPT-2 sat in MLP value vectors [77]. Initial tokens attract
attention whatever their content (attention sinks) [45], and attention is
U-shaped over the input [75]. The causal mask creates position-specific
hidden states; scaling one dimension reduced the bias [76]. The causal mask
and relative position encodings are named as sources, and PINE removes them
between documents [73].

**Suspected mechanism in Laya (a hypothesis).** ModernBERT's local layers
see 128 tokens [43]. The first options sit close to the need, so they get
the need's information in every layer; later options get it only in the
global layers, about a third of the 28. The last options sit next to the end-of-input token, which
BERT-style heads attend to a lot [40].

**Experiment A (no GPU beyond existing runs).** From `laya_en:perm4` and
`laya_en:all`, compute each right option's token distance from the need, as
well as its index. Fit accuracy on both.
- If token distance explains the first-option effect better than index: the
  local window is the suspect. Knock out the global layers' reads to confirm.

**Experiment B (Qwen listwise).** PriDe: estimate the prior over option IDs
from a few permuted cases, and divide it out [70].
- If the first-option lift disappears: the bias is a label prior. PriDe is a
  cheap fix. If not: patch the list-position states (Experiment C).

**Experiment C (one model).** Take a case with the right option first and the
same case with it eighth. Patch the right option's states from the first run
into the eighth, layer by layer. Then run DAS to find a low-rank
"ordering ID" subspace [66].
- If a small subspace carries the position effect: project it out at
  run time (an intervention fix).

**Shared prefixes.** When options share a long prefix ("go to the door",
"go to the door (locked)"), a mean over tokens is dominated by the shared
tokens; in encoders the option's marker may pool mostly the prefix. Test:
score only the differing tokens (decoders); knock out the prefix tokens'
reads (encoders).

**Fix it points to.** For decoders: PriDe [70], permutation averaging
[71][72], PINE [73] or set-based prompting [74], which changes the attention
mask and position ids so the output does not depend on order, without
training ("the impact on expected accuracy is small" [74]). Or score one
option at a time (as openjev does). For Laya: training with random orders, or
an option-isolating mask like bev-decider's (Laya would need retraining for
it; untested).

### 4.6 List-length collapse (4 to 128 options)

**What we know.** Every model falls. Some of the fall is plain statistics:
with more wrong options, more of them are near-misses, and the highest of
many noisy scores is more often wrong. openjev scores each option alone and
still falls from 88% to 32%. So only the EXTRA fall of listwise models is an
architecture effect. Laya listwise falls from 70% to 7%. Qwen3-1.7B with a
numbered list falls from 77% to 5%, while the same model scoring options one
at a time keeps 27%.

**Known mechanisms.** Softmax attention flattens as the number of items
grows, so the top item gets less weight [46]. Information in the middle of
long inputs is used less [69]; positional retrieval of bound items gets
noisy in the middle [67].

**First check (no GPU).** Count Laya's tokens at 64 and 128 options. Laya's
input is 512 tokens [107]. If options past the limit are cut, part of the
collapse is truncation, not reasoning. `run.py` already notes that Laya
"cannot fit very long lists".

**Experiment A (no GPU).** A null model for the statistical part. For each
pointwise model, take each case's per-option scores from the 16-option run.
Draw extra wrong options' scores from the same model's scores on those items
in other cases, to reach 32, 64 and 128. Predict accuracy.
- The gap between predicted and measured accuracy is the architecture effect.
  Apply the same null to Laya's 16-option scores.

**Experiment B (Laya, then GLiNER2.5-Decide).** Attention knockout: block
option-to-option attention. Keep option-to-need and option-to-itself.
- If accuracy at 64 and 128 rises toward bev-decider's (38% and 35%): the
  collapse is interference between options. The mask is also the fix
  (retraining would make it native).
- If accuracy falls: Laya needs to compare options, and the collapse is
  dilution or distance.

**Experiment C.** Patch the right option's final states from the 4-option
run into the 128-option run (same need, same right option).
- If S recovers at the last layers only: the head's comparison step fails.
  If it recovers at middle layers: the option's own reading of the need
  degrades with length (distance or dilution).

**Fix it points to.** A shortlist then a choice (already the best at 128,
roadmap 102). Options that see only the query (bev-decider's design).
Tournaments of small groups. Scalable-Softmax changes attention for long
inputs but needs pretraining or retrofitting [46]; not practical for us.

### 4.7 Wrong-person binding ("pull the bandit to safety")

**What we know.** GLiNER2.5-Decide aims the right action at the wrong person
at 2.2x chance, on only 30 wrong picks. Other models are queued.

**Known mechanisms (decoders).** Models attach a "binding ID" vector to an
entity and to its attribute, and match them [64]. A low-rank "ordering ID"
subspace carries the order of entities and changes the binding when edited
[66]. Retrieval mixes three routes: by position, by the bound word, and by a
direct pointer; the positional route is noisy in the middle of longer lists
[67]. Propositional probes read the right bindings even when the output is
wrong, so a model may hold the right world state and still answer wrong [65].
Fine-tuning improved an existing entity-tracking circuit [68]. All of this is
on decoders (Pythia, LLaMA and others). We found no study of binding in
bidirectional encoders.

**Experiment A (cheap, all models).** Swap which person has which state
("Pim is drowning; the bandit stands on the bank" vs the reverse). Keep the
option text fixed.
- Measure: does the pick follow the state or the name?
- If the pick stays with the same name: the model binds by name prior or by
  position, not by state.

**Experiment B (Qwen3-1.7B).** Interchange intervention [23][64]: patch the
activations at the person-name tokens from the swapped story into the
original run.
- If the pick moves to the other person: the binding is carried at the names,
  as binding IDs predict. Then fit DAS for a low-rank binding subspace [24].
- If the pick moves only when the state tokens are patched: binding is read
  from the state side (the lexical route of [67]).

**Experiment C (propositional probe, Qwen).** Train a probe for "X is in
danger" at each name token [65].
- If the probe is right when the pick is wrong: the scene is understood; the
  failure is in reading it out. Fixes at the option side (name the state in
  the option) should work.

**Fix it points to.** Inference: put each person's state next to their name
(cards), or put the state in the option ("pull Pim, who is drowning, to
safety"); this uses the lexical route [67]. Training: contrast sets that swap
states between people [96].

---

## 5. A prioritized experiment plan for the lab

Rules for every step: one model at a time; the cheapest step first; short GPU
runs (the laptop runs hot); kept scores always; a paired design where
possible. Paired means the same case in two conditions; compare per-case S
differences (a sign test or a paired bootstrap), not two accuracies with
plus or minus 12-point ranges. Every table keeps the word-overlap baseline.

| # | step | model | GPU | inputs | measure | decision |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | Token counts at every list size | Laya, GLiNER2.5-Decide | none | existing cases | tokens vs 512 | If lists are cut, report truncation separately from collapse. |
| 1 | Offline corrections | all with kept scores | none | `*.scores.jsonl`, `*.ablate.jsonl` | accuracy and lifts with per-option bias table [82]; object-conditional PMI [81]; domain-context blank [84]; fit on half, test on half | Keep any that gains on the test half. The bias table costs nothing at play time. |
| 2 | Null model for list length | pointwise models, then Laya | none | 16-option scores | predicted vs measured accuracy at 32-128 | Tells how much of the collapse is architecture. |
| 3 | Token distance vs index | Laya | none | `laya_en:perm4`, `laya_en:all` | accuracy by distance and by index | If distance wins, the local window is the suspect (4.5). |
| 4 | Affordance quiz | Qwen3-0.6B, then openjev 0.8B, GLiNER2.5-Decide, Laya | short | item x verb pairs, obvious vs unusual | AUC vs labels | Knows vs uses (4.2). |
| 5 | Object-swap and state-swap pairs | every model | normal scoring | cut the rope / bread / net; swapped person states | lure rank, S, pick follows name or state | Object pull vs prior; binding by state or by name. |
| 6 | Probes + control tasks + MDL | Qwen3-0.6B, then Laya | forward only, save states to disk | items in a neutral frame | selectivity and MDL per layer; hold out items | Is "can cut" or "has an edge" encoded, and where. |
| 7 | LEACE erasure at the best layer | same | 1 extra pass | step 6 directions | accuracy per need type | Is the direction used. |
| 8 | Option-to-option knockout | Laya, then GLiNER2.5-Decide | 1 pass per case | 16, 64, 128 lists | accuracy change | Interference vs dilution (4.6). |
| 9 | Activation patching heatmap | Qwen3-0.6B (nnsight or TransformerLens), then Laya | layers x spans per pair | object pairs, position pairs, filler pairs | S recovered per layer and span | Where each failure enters. |
| 10 | Attribution patching / EAP-IG | Qwen3-0.6B | 2 fwd + 1 bwd | same pairs | per-head, per-MLP scores | Suspects only; confirm top 5 with step 9. |
| 11 | Interchange intervention for binding | Qwen3-1.7B | forward only | people pairs | pick follows patched names? | Binding ID or lexical route (4.7). |
| 12 | PriDe and set-based prompting | Qwen3-1.7B listwise | a few permutations | numbered lists | first-option lift, accuracy | Cheap position fix for decoders. |
| 13 | DAS for ordering and binding subspaces | Qwen3-0.6B, then Laya | small training | position and people pairs | interchange accuracy; check vs plain patching [26] | A subspace to project out. |
| 14 | Attribution graphs on 5-10 failing cases | Qwen3-0.6B with circuit-tracer | measure memory first | "cut the rope", "reach the apple", "pour the oil" | features on the path to the lure | Exploration; confirm any feature with a probe. |
| 15 | Qwen-Scope SAE check | Qwen3-1.7B, then openjev 2B | forward only | any cases | reconstruction error and change in S when the SAE output replaces the layer | Only if transfer is good, look for lure features. |
| 16 | Fixes that need training | off the laptop | — | contrast sets, hard negatives, random orders, swapped states | held-out-item accuracy; probes and CMAP before and after [68] | Did training add knowledge or a wrapper [93]. |

Steps 0 to 3 need no GPU. Steps 4 to 9 are forward passes of the size the lab
already runs. Heavy work (SAE training, crosscoders, fine-tuning) goes to the
other machine.

---

## 6. Open questions

1. Do SAEs trained on Qwen3-1.7B-Base and Qwen3.5-2B-Base [57] work on our
   post-trained Qwen3-1.7B and on openjev's NLI fine-tune? Base-to-chat
   transfer usually works [60]; a classifier fine-tune is untested.
2. Do the Qwen3-0.6B transcoders [58] work on bev-decider, which is the first
   20 layers of Qwen3-0.6B with a merged LoRA [108]?
3. How do bidirectional encoders bind attributes to entities? We found the
   binding literature only for decoders.
4. Can Laya's head accept an option-isolating mask without retraining? It was
   trained with options that see each other.
5. Is "affordance" one direction, one per verb, or not linear at all? The
   probe design in 4.2 will tell, if the control tasks hold.
6. How do we patch the running state of a Gated DeltaNet layer (openjev)? We
   found a TransformerLens adapter for Qwen3.5 [101], but did not check what
   its hooks expose.
7. Some "all models fail" cases may have generous labels (a rolling pin for
   pond ice, a frying pan for a nail; `failures.md`). Mechanistic results on
   those cases would explain a label error, not a model error. Audit them
   before using them as minimal pairs.
8. Which fixes fit the latency budget? A bias table and a mask cost nothing
   extra. Permutation averaging, channel scoring and the quiz-as-feature
   multiply model calls.
9. Does the object-only score help or hurt as a correction? The object carries
   task information (20-23% alone), so subtracting all of it may remove good
   signal. Fit lambda; do not assume 1.

**What we could not verify.**
- The causal scrubbing post body did not load. We read its sequence page,
  which gives the authors and a summary [22].
- The two large transformer-circuits.pub pages for "Towards Monosemanticity"
  and "Scaling Monosemanticity" were too large to load. We cite Anthropic's
  summary pages instead [47][48].
- The Qwen-Scope blog page returned no content. We used the arXiv record, two
  model cards and the Hugging Face model listing [57].
- We did not open the config of the Qwen3-1.7B transcoders [58], nor check
  transcoder widths or GPU memory for circuit-tracer on our laptop.
- We did not check whether TransformerLens wraps sequence-classification
  heads, or our custom packages [101].
- Laya's head details come from the lab's own code ("option-marker head") and
  its model card [107]; we found no paper for it.
- For several arXiv papers we give the venue only when the record stated it.

---

## 7. References

All opened on 2026-09-29.

**Probing and affordance knowledge**

1. Hewitt, Liang (2019). Designing and Interpreting Probes with Control Tasks. EMNLP. https://arxiv.org/abs/1909.03368
2. Voita, Titov (2020). Information-Theoretic Probing with Minimum Description Length. https://arxiv.org/abs/2003.12298
3. Belinkov (2021). Probing Classifiers: Promises, Shortcomings, and Advances. Computational Linguistics. https://arxiv.org/abs/2102.12452
4. Elazar, Ravfogel, Jacovi, Goldberg (2021). Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals. TACL. https://arxiv.org/abs/2006.00995
5. Belrose et al. (2023). LEACE: Perfect linear concept erasure in closed form. https://arxiv.org/abs/2306.03819
6. Forbes, Holtzman, Choi (2019). Do Neural Language Representations Learn Physical Commonsense? CogSci. https://arxiv.org/abs/1908.02899
7. Bisk et al. (2020). PIQA: Reasoning about Physical Commonsense in Natural Language. AAAI. https://arxiv.org/abs/1911.11641
8. Aroca-Ouellette, Paik, Roncone, Kann (2021). PROST: Physical Reasoning about Objects through Space and Time. Findings of ACL. https://arxiv.org/abs/2106.03634
9. Adak, Agrawal, Mukherjee, Aditya (2024). TEXT2AFFORD: Probing Object Affordance Prediction abilities of Language Models solely from Text. CoNLL. https://arxiv.org/abs/2402.12881
10. Agrawal, Prabhakar, Goyal, Liu (2024). Physical Reasoning and Object Planning for Household Embodied Agents (COAT). TMLR. https://arxiv.org/abs/2311.13577
11. Tian et al. (2024). MacGyver: Are Large Language Models Creative Problem Solvers? NAACL. https://arxiv.org/abs/2311.09682
12. Gu, Dalvi Mishra, Clark (2023). Do language models have coherent mental models of everyday things? ACL. https://arxiv.org/abs/2212.10029
13. Jiang, Yang, Li, Pujara (2026). AFFORDANCE20Q: Evaluating Affordance Reasoning from Physical Properties. EMNLP Findings. https://arxiv.org/abs/2606.14240
14. Li, Nye, Andreas (2021). Implicit Representations of Meaning in Neural Language Models. ACL. https://arxiv.org/abs/2106.00737

**Causal methods**

15. Meng, Bau, Andonian, Belinkov (2022). Locating and Editing Factual Associations in GPT (causal tracing, ROME). NeurIPS. https://arxiv.org/abs/2202.05262
16. Heimersheim, Nanda (2024). How to use and interpret activation patching. https://arxiv.org/abs/2404.15255 (text read at https://arxiv.org/html/2404.15255)
17. Zhang, Nanda (2024). Towards Best Practices of Activation Patching in Language Models: Metrics and Methods. ICLR. https://arxiv.org/abs/2309.16042
18. Nanda. Attribution Patching: Activation Patching At Industrial Scale (blog post). https://www.neelnanda.io/mechanistic-interpretability/attribution-patching
19. Kramár, Lieberum, Shah, Nanda (2024). AtP*: An efficient and scalable method for localizing LLM behaviour to components. https://arxiv.org/abs/2403.00745
20. Wang, Variengien, Conmy, Shlegeris, Steinhardt (2022). Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small. https://arxiv.org/abs/2211.00593
21. Goldowsky-Dill, MacLeod, Sato, Arora (2023). Localizing Model Behavior with Path Patching. https://arxiv.org/abs/2304.05969
22. Chan, Garriga-Alonso, Goldowsky-Dill, Greenblatt, Nitishinskaya, Radhakrishnan, Shlegeris, Thomas (Redwood Research, 2022). Causal Scrubbing (sequence). https://www.alignmentforum.org/s/h95ayYYwMebGEYN5y ; post: https://www.alignmentforum.org/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-redwood-research (body did not load)
23. Geiger, Lu, Icard, Potts (2021). Causal Abstractions of Neural Networks. NeurIPS. https://arxiv.org/abs/2106.02997
24. Geiger, Wu, Potts, Icard, Goodman (2023). Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations (DAS). https://arxiv.org/abs/2303.02536
25. Wu, Geiger, Icard, Potts, Goodman (2023). Interpretability at Scale: Identifying Causal Mechanisms in Alpaca (Boundless DAS). NeurIPS. https://arxiv.org/abs/2305.08809
26. Makelov, Lange, Nanda (2023). Is This the Subspace You Are Looking for? An Interpretability Illusion for Subspace Activation Patching. https://arxiv.org/abs/2311.17030
27. Conmy, Mavor-Parker, Lynch, Heimersheim, Garriga-Alonso (2023). Towards Automated Circuit Discovery for Mechanistic Interpretability (ACDC). NeurIPS. https://arxiv.org/abs/2304.14997
28. Hanna, Pezzelle, Belinkov (2024). Have Faith in Faithfulness: Going Beyond Circuit Overlap When Finding Model Mechanisms (EAP-IG). COLM. https://arxiv.org/abs/2403.17806
29. Vig et al. (2020). Causal Mediation Analysis for Interpreting Neural NLP: The Case of Gender Bias. https://arxiv.org/abs/2004.12265
30. Geva, Bastings, Filippova, Globerson (2023). Dissecting Recall of Factual Associations in Auto-Regressive Language Models (attention knockout). EMNLP. https://arxiv.org/abs/2304.14767
31. Lieberum et al. (2023). Does Circuit Analysis Interpretability Scale? Evidence from Multiple Choice Capabilities in Chinchilla. https://arxiv.org/abs/2307.09458
32. Wiegreffe et al. (2025). Answer, Assemble, Ace: Understanding How LMs Answer Multiple Choice Questions. ICLR. https://arxiv.org/abs/2407.15018
33. Geiger et al. (2022). Inducing Causal Structure for Interpretable Neural Networks (interchange intervention training). https://arxiv.org/abs/2112.00826

**Lenses, attention and architectures**

34. nostalgebraist (2020). interpreting GPT: the logit lens. https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens
35. Belrose et al. (2023). Eliciting Latent Predictions from Transformers with the Tuned Lens. https://arxiv.org/abs/2303.08112
36. Ghandeharioun, Caciularu, Pearce, Dixon, Geva (2024). Patchscopes. ICML. https://arxiv.org/abs/2401.06102
37. Jain, Wallace (2019). Attention is not Explanation. NAACL. https://arxiv.org/abs/1902.10186
38. Wiegreffe, Pinter (2019). Attention is not not Explanation. EMNLP. https://arxiv.org/abs/1908.04626
39. Abnar, Zuidema (2020). Quantifying Attention Flow in Transformers. https://arxiv.org/abs/2005.00928
40. Clark, Khandelwal, Levy, Manning (2019). What Does BERT Look At? An Analysis of BERT's Attention. BlackboxNLP. https://arxiv.org/abs/1906.04341
41. Lu, Chen, Eickhoff (2025). Pathway to Relevance: How Cross-Encoders Implement a Semantic Variant of BM25. https://arxiv.org/abs/2502.04645
42. McCoy, Pavlick, Linzen (2019). Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in Natural Language Inference (HANS). ACL. https://arxiv.org/abs/1902.01007
43. Warner et al. (2024). Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder (ModernBERT). https://arxiv.org/abs/2412.13663 (text read at https://arxiv.org/html/2412.13663)
44. He, Liu, Gao, Chen (2020). DeBERTa: Decoding-enhanced BERT with Disentangled Attention. https://arxiv.org/abs/2006.03654
45. Xiao, Tian, Chen, Han, Lewis (2024). Efficient Streaming Language Models with Attention Sinks. ICLR. https://arxiv.org/abs/2309.17453
46. Nakanishi (2025). Scalable-Softmax Is Superior for Attention. https://arxiv.org/abs/2501.19399

**Sparse dictionaries, circuits and model diffing**

47. Anthropic (2023). Towards Monosemanticity: Decomposing Language Models With Dictionary Learning (summary page). https://www.anthropic.com/research/towards-monosemanticity-decomposing-language-models-with-dictionary-learning
48. Anthropic (2024). Mapping the Mind of a Large Language Model (summary of Scaling Monosemanticity). https://www.anthropic.com/research/mapping-mind-language-model
49. Cunningham, Ewart, Riggs, Huben, Sharkey (2023). Sparse Autoencoders Find Highly Interpretable Features in Language Models. https://arxiv.org/abs/2309.08600
50. Dunefsky, Chlenski, Nanda (2024). Transcoders Find Interpretable LLM Feature Circuits. NeurIPS. https://arxiv.org/abs/2406.11944
51. Lindsey, Templeton, Marcus, Conerly, Batson, Olah (2024). Sparse Crosscoders for Cross-Layer Features and Model Diffing. https://transformer-circuits.pub/2024/crosscoders/index.html
52. Ameisen, Lindsey, Pearce et al. (2025). Circuit Tracing: Revealing Computational Graphs in Language Models. https://transformer-circuits.pub/2025/attribution-graphs/methods.html
53. Lindsey et al. (2025). On the Biology of a Large Language Model. https://transformer-circuits.pub/2025/attribution-graphs/biology.html
54. Hanna, Piotrowski, Lindsey, Ameisen (2025). circuit-tracer: A New Library for Finding Feature Circuits. BlackboxNLP. https://aclanthology.org/2025.blackboxnlp-1.14.pdf ; repository: https://github.com/safety-research/circuit-tracer
55. Lieberum et al. (2024). Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2. https://arxiv.org/abs/2408.05147
56. He et al. (2024). Llama Scope: Extracting Millions of Features from Llama-3.1-8B with Sparse Autoencoders. https://arxiv.org/abs/2410.20526
57. Deng et al. (Qwen team, 2026). Turning Sparse Features into Development Tools for Large Language Models (Qwen-Scope). https://arxiv.org/abs/2605.11887 ; model cards: https://huggingface.co/Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50 and https://huggingface.co/Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50 ; listing: https://huggingface.co/api/models?author=Qwen&search=SAE-Res
58. Hanna (mwhanna). Qwen3 transcoders for circuit-tracer. Listing: https://huggingface.co/api/models?author=mwhanna&search=qwen3 ; 0.6B config: https://huggingface.co/mwhanna/qwen3-0.6b-transcoders-lowl0/raw/main/config.yaml ; repository info: https://huggingface.co/api/models/mwhanna/qwen3-0.6b-transcoders-lowl0
59. BluelightAI (2025). Cross Layer Transcoders for the Qwen3 LLM Family. https://www.lesswrong.com/posts/cW9AdDm2DZtbXBnvt/cross-layer-transcoders-for-the-qwen3-llm-family ; listing: https://huggingface.co/api/models?author=bluelightai
60. Kissane, robertzk, Conmy, Nanda (2024). SAEs (usually) Transfer Between Base and Chat Models. https://www.lesswrong.com/posts/fmwk6qxrpW8d4jvbd/saes-usually-transfer-between-base-and-chat-models
61. Google DeepMind mech interp team (2025). Negative Results for SAEs On Downstream Tasks and Deprioritising SAE Research. https://www.alignmentforum.org/posts/4uXCAJNuPKtKBsi28/negative-results-for-saes-on-downstream-tasks
62. Wu, Arora, Geiger et al. (2025). AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders. https://arxiv.org/abs/2501.17148
63. Neuronpedia (open source). https://github.com/hijohnnylin/neuronpedia ; Qwen3-4B page: https://www.neuronpedia.org/qwen3-4b

**Binding, position and order**

64. Feng, Steinhardt (2023). How do Language Models Bind Entities in Context? https://arxiv.org/abs/2310.17191
65. Feng, Russell, Steinhardt (2024). Monitoring Latent World States in Language Models with Propositional Probes. https://arxiv.org/abs/2406.19501
66. Dai, Heinzerling, Inui (2024). Representational Analysis of Binding in Language Models (ordering ID). https://arxiv.org/abs/2409.05448
67. Gur-Arieh, Geva, Geiger (2025). Mixing Mechanisms: How Language Models Retrieve Bound Entities In-Context. ICLR 2026. https://arxiv.org/abs/2510.06182
68. Prakash, Rott Shaham, Haklay, Belinkov, Bau (2024). Fine-Tuning Enhances Existing Mechanisms: A Case Study on Entity Tracking. ICLR. https://arxiv.org/abs/2402.14811
69. Liu et al. (2023). Lost in the Middle: How Language Models Use Long Contexts. TACL. https://arxiv.org/abs/2307.03172
70. Zheng, Zhou, Meng, Zhou, Huang (2024). Large Language Models Are Not Robust Multiple Choice Selectors (PriDe). ICLR. https://arxiv.org/abs/2309.03882
71. Pezeshkpour, Hruschka (2023). Large Language Models Sensitivity to The Order of Options in Multiple-Choice Questions. https://arxiv.org/abs/2308.11483
72. Tang, Zhang, Ma, Lin, Ture (2024). Found in the Middle: Permutation Self-Consistency Improves Listwise Ranking in Large Language Models. NAACL. https://arxiv.org/abs/2310.07712
73. Wang et al. (2024). Eliminating Position Bias of Language Models: A Mechanistic Approach (PINE). https://arxiv.org/abs/2407.01100
74. McIlroy-Young, Brown, Olson, Zhang, Dwork (2024). Order-Independence Without Fine Tuning (set-based prompting). NeurIPS. https://arxiv.org/abs/2406.06581
75. Hsieh et al. (2024). Found in the Middle: Calibrating Positional Attention Bias Improves Long Context Utilization. Findings of ACL. https://arxiv.org/abs/2406.16008
76. Yu et al. (2025). Mitigate Position Bias in Large Language Models via Scaling a Single Dimension. Findings of ACL. https://arxiv.org/abs/2406.02536
77. Li, Gao (2025). Anchored Answers: Unravelling Positional Bias in GPT-2's Multiple-Choice Questions. Findings of ACL. https://arxiv.org/abs/2405.03205
78. Zhou, Feng, Zhu, Qian, Mao (2024). UniBias: Unveiling and Mitigating LLM Bias through Internal Attention and FFN Manipulation. NeurIPS. https://arxiv.org/abs/2405.20612
79. Robinson, Rytting, Wingate (2023). Leveraging Large Language Models for Multiple Choice Question Answering (multiple choice symbol binding). ICLR. https://arxiv.org/abs/2210.12353

**Calibration and distraction**

80. Zhao, Wallace, Feng, Klein, Singh (2021). Calibrate Before Use: Improving Few-Shot Performance of Language Models. ICML. https://arxiv.org/abs/2102.09690
81. Holtzman, West, Shwartz, Choi, Zettlemoyer (2021). Surface Form Competition: Why the Highest Probability Answer Isn't Always Right (domain-conditional PMI). https://arxiv.org/abs/2104.08315
82. Zhou et al. (2024). Batch Calibration: Rethinking Calibration for In-Context Learning and Prompt Engineering. ICLR. https://arxiv.org/abs/2309.17249
83. Han, Hao, Dong, Sun, Wei (2022). Prototypical Calibration for Few-shot Learning of Language Models. https://arxiv.org/abs/2205.10183
84. Fei, Hou, Chen, Bosselut (2023). Mitigating Label Biases for In-context Learning (domain-context calibration). ACL. https://arxiv.org/abs/2305.19148
85. Min, Lewis, Hajishirzi, Zettlemoyer (2022). Noisy Channel Language Model Prompting for Few-Shot Text Classification. ACL. https://arxiv.org/abs/2108.04106
86. Shi et al. (2023). Large Language Models Can Be Easily Distracted by Irrelevant Context. ICML. https://arxiv.org/abs/2302.00093

**Editing, fine-tuning, training data and steering**

87. Meng, Sen Sharma, Andonian, Belinkov, Bau (2022). Mass-Editing Memory in a Transformer (MEMIT). https://arxiv.org/abs/2210.07229
88. Fang et al. (2025). AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models. ICLR. https://arxiv.org/abs/2410.02355
89. Dai et al. (2022). Knowledge Neurons in Pretrained Transformers. ACL. https://arxiv.org/abs/2104.08696
90. Hase, Bansal, Kim, Ghandeharioun (2023). Does Localization Inform Editing? NeurIPS. https://arxiv.org/abs/2301.04213
91. Cohen, Biran, Yoran, Globerson, Geva (2024). Evaluating the Ripple Effects of Knowledge Editing in Language Models. TACL. https://arxiv.org/abs/2307.12976
92. Gupta, Rao, Anumanchipalli (2024). Model Editing at Scale leads to Gradual and Catastrophic Forgetting. Findings of ACL. https://arxiv.org/abs/2401.07453
93. Jain et al. (2023). Mechanistically analyzing the effects of fine-tuning on procedurally defined tasks ("wrappers"). https://arxiv.org/abs/2311.12786
94. Lee et al. (2024). A Mechanistic Understanding of Alignment Algorithms: A Case Study on DPO and Toxicity. https://arxiv.org/abs/2401.01967
95. Kaushik, Hovy, Lipton (2020). Learning the Difference that Makes a Difference with Counterfactually-Augmented Data. ICLR. https://arxiv.org/abs/1909.12434
96. Gardner et al. (2020). Evaluating Models' Local Decision Boundaries via Contrast Sets. https://arxiv.org/abs/2004.02709
97. Robinson, Chuang, Sra, Jegelka (2021). Contrastive Learning with Hard Negative Samples. ICLR. https://arxiv.org/abs/2010.04592
98. Wu, Arora, Wang, Geiger, Jurafsky, Manning, Potts (2024). ReFT: Representation Finetuning for Language Models. https://arxiv.org/abs/2404.03592
99. Panickssery, Gabrieli, Schulz, Tong, Hubinger, Turner (2023). Steering Llama 2 via Contrastive Activation Addition. https://arxiv.org/abs/2312.06681
100. Turner et al. (2023). Steering Language Models With Activation Engineering (ActAdd). https://arxiv.org/abs/2308.10248

**Tools**

101. TransformerLens. https://github.com/TransformerLensOrg/TransformerLens ; adapters checked: https://github.com/TransformerLensOrg/TransformerLens/blob/main/transformer_lens/model_bridge/supported_architectures/qwen3.py , https://github.com/TransformerLensOrg/TransformerLens/blob/main/transformer_lens/model_bridge/supported_architectures/qwen3_5.py ; listing (bert, modernbert_decoder; no deberta): https://api.github.com/repos/TransformerLensOrg/TransformerLens/contents/transformer_lens/model_bridge/supported_architectures
102. nnsight (Fiotto-Kaufman et al.). https://github.com/ndif-team/nnsight
103. Wu et al. (2024). pyvene: A Library for Understanding and Improving PyTorch Models via Interventions. https://arxiv.org/abs/2403.07809
104. SAELens. https://github.com/decoderesearch/SAELens
105. inseq. https://github.com/inseq-team/inseq
106. baukit. https://github.com/davidbau/baukit

**Our models (cards)**

107. Laya (ModernBERT-large, 421M, 512 tokens). https://huggingface.co/convaiinnovations/laya
108. bev-decider-0.4B (20 layers of Qwen3-0.6B, merged LoRA, 2-layer head; "choice-order invariance by construction"). https://huggingface.co/avbiswas/bev-decider-0.4B
109. GLiNER2.5-Decide (DeBERTa-v3-large, 340M). https://huggingface.co/fastino/GLiNER2.5-Decide ; GLiNER2 paper (Zaratiana et al., 2025): https://arxiv.org/abs/2507.18546
110. openjev (Qwen3.5 cross-encoder, 3-way NLI head, last-token pooling). https://huggingface.co/AlexWortega/openjev
111. Qwen3.5-2B model card (24 layers; 6 x (3 x Gated DeltaNet, 1 x gated attention)). https://huggingface.co/Qwen/Qwen3.5-2B
