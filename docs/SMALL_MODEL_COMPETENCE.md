# Small-Model Competence — the training & reliability playbook

Product of a 6-cluster deep sweep (prompt scaffolds, distillation/fine-tuning,
ensembling, constrained generation, retrieval exemplars, calibration/abstention)
against our Qwen3-0.6B selector on 6GB. Companion to SMALL_MODEL_HARNESS.md.

HEADLINE: the ONE technique with direct published evidence at our exact scale
(0.8B) is kNN-Prompting — frozen model as feature extractor, decision by
non-parametric kNN over our own log. ~60 LOC numpy. Build it first.

I'll synthesize this for your engine. This is analysis work — no tools needed, just ranking against what you've already built.

# Synthesis: Small-Model Selector Techniques for Mark_1

## 1. DEDUPE — Canonical Technique List

The 30-odd findings collapse to **11 distinct techniques**:

| # | Canonical technique | Appears as |
|---|---|---|
| A | **Reason-free, constrain-late** (never emit structure from the 0.6B) | reason-then-format, CRANE-answer-slot, grammar-redundancy |
| B | **Distill teacher rationales into weights, deploy as label-only classifier** | Distilling Step-by-Step, SCoTD, seq-KD |
| C | **RFT on sim-verified selections** (log = corpus, sim = perfect verifier) | RFT, STaR loop-discipline |
| D | **DPO on good-vs-bad option pairs from same menu** | V-STaR |
| E | **Decompose hard pick → chain of atomic PMI picks** | least-to-most, DialCoT, step-back hierarchy |
| F | **kNN-Prompting** (LM logits as features → external kNN over log) | kNN-Prompting, task-adapted metric |
| G | **Soft self-consistency** (prob-weighted vote over K sampled scoring passes) | soft-SC, hard-SC, adaptive-consistency |
| H | **Second Guess** (option-perturbation answer-stability abstention) | Second Guess |
| I | **P(True)/P(IK) single-token confidence probe** | P(True), self-eval |
| J | **Signal fusion** (margin + P(True) [+ hidden-state probe] via tiny logistic reg) | entropy-insufficient, 2-axis, difference-of-means probe, multi-layer ensemble, conformal thresholding |
| K | **Voice-lane digit/meta LogitsProcessor mask** (jump-forward, no retry loop) | jump-forward, DOMINO token-healing, HF-LogitsProcessor feasibility |

LoRA/QLoRA feasibility and LR-to-batch ratio are **substrate/config**, not standalone techniques — they attach to C/D.

---

## 2. RANKED SHORTLIST (competence-gain-per-line, sub-1B selector)

### #1 — kNN-Prompting (F) — **the standout**
**Port:** You already cache PMI denominators + do batched option scoring. Build a datastore `{kl_key: logprob-vector-over-option/stance-tokens}` from your logged `(situation → correct-choice)` tapes. At decision time you already compute the query distribution; add a KL-kNN lookup (k=8–64, numpy) and blend: `final = α·selector_logprob + (1−α)·kNN_vote`. ~60–100 LOC, numpy only, no server. Grows automatically as the log grows.
**Expected gain:** +3.5 (4-shot) to +7 (8-shot) over calibration baselines, and it's the **only** technique here with *direct published evidence at 0.8B*, scaling monotonically to 1024 anchors.
**Evidence strength:** STRONG. Tested at your exact scale, bypasses ICL-emergence entirely (model is a frozen feature extractor). This is the highest-confidence prompt-side win you have not yet built.

### #2 — RFT on sim-verified selections (C)
**Port:** `mine_rft.py` — walk `world.log` tapes; for each Selector decision emit `(assembled_prompt, option_str)`; keep rows whose downstream deltas pass an author-written `good_outcome(deltas)` predicate (start permissive: "not refusal-only AND scene advanced"). Dedup by `(situation-signature, option)`. Plain HF LoRA SFT over your exact `llm.py _prompt` format. **Retrain from base Qwen3-0.6B each round**, stop at golden-eval saturation.
**Expected gain:** RFT's law is "weaker/smaller models gain the most"; LLaMA-7B GSM8K 35.9→41.7. Your task is far easier than GSM8K and your verifier is **free and perfect** (the sim) — the exact spot where RFT is cheap for you and expensive for everyone else.
**Evidence strength:** STRONG mechanism, but no sub-1B *selection*-RFT number exists — the gain size is an educated extrapolation.

### #3 — Second Guess abstention (H)
**Port:** After your margin-gate flags a narrow pick, run **one** extra scoring pass with a synthetic "none/unsure" option appended and option order permuted. If argmax flips → route to ask-back instead of committing. Gate behind existing margin-gate so confident turns stay at 1 call. No training, no logprobs needed — argmax only.
**Expected gain:** −10.8% Composite Risk at just 2 forward passes, beating P(True) and entropy. **Gains grow as base accuracy drops** (inverse fit) — precisely the 0.6B regime.
**Evidence strength:** STRONG for your size class (paper is *about* 2–8B, low-accuracy models gained most). Best single abstention primitive here.

### #4 — Signal fusion for the ask-back gate (J)
**Port:** Log per-turn `[continuation margin, option-entropy, P(True)]` + sim ground-truth label ("committed action later reverted/contested"). Fit `LogisticRegression(C=0.1)` on StandardScaler'd features; use its output as the ask-back gate **instead of a raw margin threshold**. Re-fit offline nightly. ~20 LOC sklearn.
**Expected gain:** TriviaQA AUROC 0.826→0.863; the point is it tells you **when margin is lying** — critical because your forced-token stance prefixes artificially sharpen the distribution and inflate raw-margin confidence.
**Evidence strength:** MEDIUM-STRONG, with an honest caveat: fusion helped on TriviaQA, *hurt* on BioASQ/MedicalQA. Must validate on your turn-distribution. Qwen showed the worst "confidently-wrong" collapse, so this matters more at 0.6B.

### #5 — Soft self-consistency over sampled scoring passes (G)
**Port:** You already produce per-option continuation scores. Sample K∈{3,5} variants of the gated option-set or K stance-prefixes; **accumulate each option's score and argmax the sum** (soft, not hard vote). Reuse the margin-gate to early-stop (adaptive-consistency): easy gates cost 1 pass, ambiguous ones spend up to 5.
**Expected gain:** Soft-SC beats hard-SC +6.6 on WebShop; k=5 soft ≈ k=10 hard. Its killer property: **works when no discrete majority forms** (86% of sparse-action cases at k=5) — exactly a noisy small selector's failure mode.
**Evidence strength:** MEDIUM. Mechanism is size-agnostic (aggregates the model's own probs), but all numbers are 7–70B and gains grow with size — expect a smaller absolute lift at 0.6B. Worth measuring because it's near-free on your existing scorer.

### #6 — DPO on same-menu good/bad pairs (D)
**Port:** `mine_dpo.py` — for each logged decision with ≥1 verifier-passing AND ≥1 verifier-failing option **in the same menu**, emit `(prompt, chosen=good, rejected=bad)`. TRL `DPOTrainer` + LoRA, β=0.1. VRAM: policy+ref at 0.6B fp16 ≈ 1.2+1.2GB, trivially inside 6GB. Adapter drops straight into `pick_scored`.
**Expected gain:** V-STaR +4–17% absolute; crucially a **DPO ranker beats a classifier-head ORM for N>4 candidates** — your exact 5–10-way regime. It directly sharpens the PMI margin your gate already reads.
**Evidence strength:** MEDIUM-STRONG mechanism; do this **after** RFT (it needs the failing-option labels RFT mining already surfaces, so it's incremental cost).

### #7 — Voice-lane digit/meta mask (K)
**Port:** Replace your post-hoc retry+regex sanitizer on the voice lane with a hand-written HF `LogitsProcessor` masking digit/dice/stat token-ids **during** decode. ~15–30 LOC, zero dependency (no grammar library needed for a fixed ban-set). Your forced stance-opener becomes a length-1 grammar → jump-forward for free.
**Expected gain:** Guarantees "words-not-numbers" in one pass (kills the reseeded-retry→canned-fallback loop), and net-*faster* per jump-forward evidence (~6–9ms/tok vs 15–16). Keep it OFF the prose body (CRANE), ON only the bans.
**Evidence strength:** STRONG on mechanism; this is the *only* genuinely additive constrained-decoding port for your architecture — the rest of that cluster validates what you already do.

**Placement note:** F, C, H, J are the core four. G, D, K are strong seconds. I'd stop the shortlist at 7; everything below is either redundant or a config-knob.

---

## 3. THE TRAINING QUESTION — concrete recipe

**Verdict up front: YES, worth it — but as *cheap SFT you already have the data for*, not as a research project. Do it in this order, and only escalate if prompting-side (kNN) saturates.**

### Realistic recipe

**Method:** RFT → (optionally) DPO. Both are **plain SFT/preference LoRA** — no RL rollouts, no reward model, no giant teacher required, because **the sim is your verifier**. This is the single fact that makes training cheap for you specifically. The usual small-model RFT killer ("a small model can't verify its own outputs") does not bite: your verifier is authoritative and free.

**Precision:** Use **16-bit LoRA, NOT 4-bit QLoRA.** A 0.6B in fp16 is ~1.2GB — you are nowhere near VRAM-bound on 6GB, and NF4 rounding noise would perturb the **tight PMI margins** your selector runs on. QLoRA's entire reason-for-being (squeeze a big model onto a small card) doesn't apply at 0.6B. Skip bitsandbytes.

**Config:**
- LoRA on `q,k,v,o,gate,up,down`, **r=16, α=32** (drop to r=8 if margins look jittery post-train)
- **LR 1e-4 to 2e-4** (higher end), effective batch ~8 (`per_device=2, grad_accum=4`) — keep the LR-to-batch ratio **high**, per SmolTulu: sub-2B reasoning-shaped tasks want a *higher* ratio than big-model recipes, confirmed down to 135M. Copying a big-model LR silently underperforms.
- 1–3 epochs, cosine decay
- **Retrain from base every round; stop at golden-eval saturation** (STaR/RFT discipline — prevents overfit, which is where the 33B model in the RFT paper got zero gain)

**VRAM:** ~1.2GB base (fp16) + LoRA adapters + short-sequence activations (your prompts are mostly <512 tok). Comfortably inside 6GB with headroom; Unsloth optional for ~2× speed but unnecessary.

**Steps/time:** Your corpus is a finite own-log; candidate mining is **1 forward-pass-per-option with zero generation**, so mining the whole log is *minutes*, not the 10–62 A100-hrs the text-reasoning RFT paper needed. Training: a few hundred to low-thousands of steps = **tens of minutes on the 6GB card**.

**Critical dataset hygiene:**
- Dedup `(situation-signature, option)` so common turns don't dominate (RFT distinct-path lesson)
- For DPO, pairs must differ **only by option within the same turn** — teaches option-discrimination, not context memorization
- Reuse the *exact* `llm.py _prompt` chat-template render for train and infer; **assert the template hash at load** (template drift silently corrupts PMI)
- Gate every promotion on the 59-case golden eval + a held-out log slice; pick by **eval, not train loss**

### Is it worth it vs pure prompting?

**Do kNN-Prompting (#1) first** — it's prompting-side, needs no training, has direct 0.8B evidence, and consumes the *same* `(situation, correct-choice)` log. If kNN closes the gap, you may not need to train at all.

**Then train** if kNN saturates, because:
- Your data is **free and already logged** (typed-delta replay corpus)
- Your verifier is **free and perfect** (the sim) — the expensive part for everyone else
- The scaling law explicitly favors you: "RFT brings more improvement for *less performant* LLMs"
- It's *cheap SFT*, not RL — tens of minutes, inside your card

**Do NOT** build the distillation/teacher path (B) unless a specific class of turns has *ambiguous* good-outcomes the sim can't label — that's the only niche where a 7B teacher's judgment beats the free sim label, and it's an offline-only labeling job (never co-resident with your 6GB serving path).

---

## 4. REJECTED (one line each)

| Technique | Reason |
|---|---|
| In-context CoT / zero-shot CoT | Emergent >~100B; **actively hurts** <10B — produces fluent-illogical chains scoring below plain prompting. Your scorer has no trace to read anyway. |
| Self-ask / plan-and-solve | Same emergence wall; all evidence at 175B/PaLM-2L. No runtime port. |
| Free-form self-verification ("are you sure?") | Verification tracks capacity; 7–13B self-verifiers score ~F1 51% (coin-flip). A 0.6B is below the floor. Your sim validity-filter already verifies for free. |
| Expert/amateur contrastive decoding | Needs a *large* expert to pair with the amateur; you have no expert (0.6B *is* the amateur). Also *hurts* factual recall −2.1 — bad for an authoritative front-end. |
| STaR self-generated rationalization at runtime | 6B needed 36 iterations to crawl to 10.7% GSM8K; violates 0-generation latency budget. Keep only the loop-discipline (already folded into RFT). |
| Runtime self-consistency generation (SCoTD-style) | N generative samples/decision blows the latency budget; teacher-side data-filter only. |
| DoLa layer-contrast | Its own Table 17: fails on small LMs ("layers lack distinction"). One throwaway A/B on Qwen's 28 layers at most — do not budget for a win. |
| Grammar library for the composer | Strictly *weaker* than your existing referent-checked slot-selection (selection guarantees valid referents, grammar can't) + adds a dependency the repo avoids. |
| 4-bit QLoRA for the 0.6B | NF4 noise perturbs tight PMI margins; a 0.6B doesn't need quantization on 6GB. Use 16-bit LoRA. |
| Hidden-state correctness probe as **sole** gate | At 0.6B peaks ~0.70 AUROC and is "erratic layer-to-layer"; usable only as a *supplementary* fusion feature, not standalone. |
| Autocontrastive decoding | Needs fine-tuned early-exit heads and **increases** hallucination — wrong direction. |
| Generic-embedding kNN for the *label* | KATE kNN_roberta=50.2 (near-useless for label prediction); embeddings are for *fetching exemplars*, not deciding. The LM's own logits (F) are the task-aware space. |

---

## 5. ALREADY-HAVE — validated by this literature

- **Scoring-only `pick_scored` (no generation, no JSON)** ← validated by "Capacity, Not Format": the format tax is *worst* at your size class (Haiku −36pp). Your architecture pays **zero** format tax. **Never regress to "emit JSON."**
- **Two-stage classify-then-select** ← validated by step-back hierarchy: coarse-then-fine scoring reduces branching factor and improves PMI separability.
- **Composer's dependency-ordered slot selection** ← validated by least-to-most + grammar-constrained-gen papers: you already banked the +42.5pp Qwen3-0.6B Bash-grammar win by a *stronger* route (selection guarantees valid **referents**, not just valid syntax).
- **Validity-filtered option sets (dry-run against forked sim)** ← this is a *perfect free verifier*, the exact asset that makes RFT/DPO cheap for you and expensive for everyone else.
- **Margin-gate → ask-back** ← the correct scaffold onto which Second Guess, soft-SC early-stop, and conformal thresholding all attach.
- **Forced-token stance prefixes** ← a length-1 grammar (jump-forward); validated — but note they *sharpen the distribution* and can inflate raw-margin confidence, which is exactly why signal-fusion (#4) matters.
- **`enable_thinking=False`** ← correct; keep it, and add the CoT-tripwire assertion (CoT preamble must NOT beat plain scored prompt on golden set).
- **Joint-tokenization token-healing for PMI options** ← validated by DOMINO (mis-tokenized masking costs up to −11pt, and hurts a 0.6B *most* since its mass is thin).
- **CRANE answer-slot rule** ← you already obey it: composer slots are extraction (safe to mask), nothing forces answer-before-context.

---

## 6. THE SINGLE HIGHEST-LEVERAGE NEXT TECHNIQUE

**Build kNN-Prompting (F).**

Why it wins the tiebreak over RFT:
1. **Only technique here with direct evidence at your exact scale (0.8B)** — everything else extrapolates down from 7B+.
2. **Zero training, zero new infra** — reuses the PMI-scoring pipeline and cache you already have; the datastore is literally your existing `(situation → correct-choice)` log keyed by the model's own output distribution.
3. **Structurally sidesteps the ICL-emergence wall** — the 0.6B is used only as a frozen feature extractor (produce a distribution that moves with the label, which even tiny LMs do); the actual decision is a non-parametric kNN *outside* the model.
4. **Self-improving for free** — accuracy scales monotonically with anchor count up to 1024, and your log grows every session, so the selector gets better with play at no marginal engineering cost.
5. **~60–100 LOC of numpy** — highest competence-gain-per-line in the entire corpus.

Concrete first cut: build the datastore over your `social.py _classify` decisions (cache the logprob vector over the stance/option-token positions per logged correct choice), add KL-divergence kNN (k=8), blend `α·selector + (1−α)·kNN_vote`, sweep α on the 59-case golden eval. If it moves eval there, extend the same datastore to the composer's hardest ambiguous verb/target picks.

RFT is the highest-leverage *training* move and should be the very next thing after kNN — but kNN is strictly first because it's free evidence at your scale and it de-risks whether you need training at all.
