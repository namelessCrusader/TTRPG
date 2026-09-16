# Small-Model Harness — the usability playbook

Product of a 6-cluster sweep (KoboldAI/KoboldCpp, SillyTavern, TextWorld/Jericho/
CALM/BALROG, Voyager/NetPlay, AI-Dungeon clones, decoding research), synthesized
against this engine: sim-authoritative, Qwen3-0.6B selector + 0.5B voice, 6GB.

The thesis this sweep confirmed: THE HARNESS IS THE MODEL. Every ecosystem that
had to make small models work converged on the same shape we built — and the
gaps they close for us are ranked below.

# 1. DEDUPE — canonical forms

The 33 findings collapse to 14 distinct mechanisms:

1. **Conditional lore injection** (Kobold World Info = ST World Info = AI-Dungeon-clone cards): gate context fragments on triggers; only relevant entries cost tokens.
2. **Depth-positioned injection** (Kobold Author's Note = ST @Depth = clone andepth): steer at the recency hotspot, not the prompt top.
3. **Cache-first prompt layout** (ContextShift = ST sticky effects = Jericho fixed sections = shared-prefix batching): stable prefix, append-only middle, volatile tail only.
4. **Grammar-constrained decoding** (GBNF = llguidance): mask invalid tokens; parseable by construction.
5. **Backtrack phrase banning** (banned_strings) + **DRY anti-repetition** + **deterministic sanitizer** (Clover regex pipeline): flavor-lane output hygiene, three layers of one idea.
6. **Validity filtering by state-diff** (Jericho Alg. 1 = Voyager verify-gate = sim-as-critic): execute candidates against forked state; keep only ones that change the world.
7. **Template+slot action factorization** (Jericho).
8. **Typed per-call info diet** (TextWorld EnvInfos = Jericho handicap reporting).
9. **Propose/rank split with logged-play training data** (CALM).
10. **Batched one-pass option scoring** (DRRN batch = llama.cpp shared-prefix) + **precomputed DC-PMI priors** + **token healing**: the scoring-math stack.
11. **Event-gated LM invocation** (NetPlay) + **stall detection** (NetPlay = Clover loop-revert).
12. **Outcome-first forced prefix** (Clover d20).
13. **Input canonicalization** (thadunge2/Clover).
14. **Embedding retrieval/cache** (Voyager skill library + option pruning).

# 2. RANKED SHORTLIST (port plans)

**1. One-forward-pass menu selection: shared-prefix batching + compile-time DC-PMI priors + token healing.**
Restructure the composer's scorer around llama.cpp KV reuse: evaluate the turn context once, fork one sequence per menu option, score all continuations in a single `llama_batch`. Compute each authored option's domain-prior logprob (`P(y|domain_string)`) at pack-compile time and store it in the pack, so runtime PMI is numerator-only. Tokenize prompt+option jointly and score the suffix span (implicit token healing — no trailing-space artifacts flipping close margins). Add: assert the chat-template hash of each checkpoint at load (template drift silently corrupts PMI calibration), and margin-gate — if top1−top2 margin is below a golden-eval-tuned threshold, route to the existing ask-back instead of accepting. Net: a full menu selection costs ~1 forward pass with full PMI accuracy. Seam: composer/scorer.

**2. Validity-filtered menus (Jericho world-change detection).**
Before building each affordance menu, dry-run every candidate through the effect bus against a forked sim state; drop options emitting zero or refusal-only deltas. The 0.6B only ever ranks options pre-verified to do something (DRRN's 10.7% vs 6.1% was exactly this pruning). Free byproduct: the dry-run's failure delta is the text of the diagnostic refusal for absent options. Seam: director → menu builder, 0 LM calls.

**3. Slot/depth/order prompt assembler with cache-stable layout.**
Every fragment declares {slot, depth-from-end, order, budget-tier}. Layout: immutable rules/persona prefix → scene block (mutates only on scene transitions; accept one reprocess per scene) → event tail → a <50-token director-note slot at depth 2-3 lines above the answer point (current mood word, tone directive, menu framing). Budget admission is deterministic: pinned > tier order, truncate oldest-first, never mid-fact. Deterministic placement also keeps PMI scores comparable across turns. A/B insertion depth on the 59-case eval. Seam: composer core.

**4. Scene-graph-gated lore cards (World Info minus the keywords).**
We beat keyword scanning: the sim knows scene membership. Per-entity qualitative cards injected iff the entity is in the current scene graph; AND-gates on effect-bus flags for conditional facts; ST-style mutually-exclusive card variants (hostile-card vs friendly-card) selected by sim state; NOT-gates to suppress stale lore; hard per-tier token caps (50/100/150). Add an explicit `links:` field for bounded 1-hop pulls (faction card when its NPC is present) — authored graph traversal, no keyword recursion. Sticky-within-scene so the assembled prefix is byte-stable and prompt caching engages. Seam: composer, 0 LM calls.

**5. Input canonicalization + speech/action split.**
Deterministic pre-classifier normalizer: collapse to canonical second-person form, complete punctuation, detect quoted input. Quoted speech at an addressed NPC skips intent classification entirely — it's dialogue by construction, saving the reactive-mode call. Run golden eval raw vs canonicalized to measure. Seam: front of the social pipeline, pure regex.

**6. Outcome-first forced prefix for flavor narration (Clover d20).**
The sim's seeded roll and typed delta already decide the outcome; render margin-of-success into a graded verb/adverb prefix ("You masterfully…" / "You start to…") and force-decode the 0.5B from it. Constraint lives in committed tokens — the one channel a tiny model reliably obeys — not in instructions. Seam: flavor lane, template table only.

**7. Sampler-level enforcement on the generative lanes.**
Flavor voice: banned_strings with rewind-and-resample for RP slop, meta-leakage, and crucially numeric/stat patterns ("HP", digit+damage) — enforcing words-not-numbers mechanically; DRY sampler with sequence breakers covering our dialogue markup and NPC names; Clover-style sanitizer (sentence truncation, quote balancing) as the last pass. Ask-back questions: grammar-constrain to a template GBNF so clarifications are machine-parseable. Free if serving via koboldcpp/llama.cpp. Seam: flavor lane + ask-back emitter.

**8. Event-gated invocation + stall detection.**
NPC behaviors are authored multi-tick scripts; define a small typed interrupt vocabulary (addressed-by-player, belief contradiction, qualitative threshold crossed, script precondition broke) as the ONLY triggers for an LM selection — one call amortized over many ticks, reactive mode stays 0-call by construction. Stall rule: k consecutive selections of the same intent yielding empty/identical deltas → suppress that option, force ask-back/diagnostic refusal, log a typed `fallback` delta and flag it as a candidate golden-eval case. Seam: director.

# 3. REJECTED

- **Voyager embedding skill-cache + embedding option pruning**: needs embeddings infra we don't run; validity filtering + scene gating already prune deterministically. (The verify-then-store idea survives keyless: freeze verified novel adjudications as new authored menu entries keyed by exact intent-template + situation signature.)
- **CFG negative prompting**: duplicates KV/compute on a 6GB card for an optional lane; banned_strings + DRY buy the same hygiene cheaper.
- **Speculative decoding**: draft-model overhead likely exceeds gain on an already-fast 0.5B (findings' own conclusion).
- **Recursive keyword activation**: keyword-hop chains are a workaround for lacking a world model; our typed entity graph with an explicit `links:` field replaces it (folded into #4).
- **Keyword scanning generally**: scene-graph membership strictly dominates string matching in an authoritative sim.
- **Multi-paraphrase adaptive voting**: extra scoring passes violate the latency budget; keep only the margin *gate* (into #1) routing to ask-back at 0 cost.
- **Letter-label (MCSB) scoring**: sub-1B models lack symbol binding; would silently degrade toward chance.
- **CALM's learned DRRN re-ranker / fine-tuning**: not harness code; parked as a future use of the delta log.
- **Timed cooldown/delay effects**: solves keyword-monopoly problems we don't have; sticky-within-scene (in #4) is the only piece that pays.

# 4. ALREADY-HAVE VALIDATIONS

- **Continuation/cloze PMI scoring**: Robinson & Wingate confirm it's the correct regime for sub-1B (MCSB is a big-model skill). Keep; add the letter-vs-cloze A/B arm to the eval as a tripwire for future fine-tunes.
- **Affordance menus (choice over authored actions)**: DRRN-vs-TDQN is the empirical proof; #2 completes it by guaranteeing choice-set validity.
- **Two-stage classify-then-select**: exactly CALM's propose/rank split, which matched oracle-action agents at ~100M params. Jericho's template+slot factorization says structure stage-2 as template + scene-grounded slots so 59 golden cases cover templates × slot-types.
- **Qualitative words-not-numbers prompts**: NetPlay's structured-buckets finding independently confirms; #7 adds mechanical enforcement.
- **Diagnostic refusal / ask-back**: is BALROG's graceful-fallback protocol; harden by logging every fallback as a typed delta and capping fed history.
- **Constrained output for classification**: "Let Me Speak Freely?" shows format constraints *help* classification-shaped tasks — we are entirely on the winning side.
- **Replayable typed-delta log**: is a ClubFloyd-style (state, chosen-action) dataset accumulating for free — the future fine-tune corpus.

# 5. THE BIGGEST LEVER

The lever not yet pulled is treating the assembled prompt as an engineered, sim-computed artifact rather than a template: every ecosystem surveyed spent years approximating with keywords, embeddings, and critic calls what an authoritative sim computes exactly — which entities are present (retrieval), which actions can succeed (validity), what the outcome was (adjudication), and where a fact must sit to be attended to (placement). Wiring those four through the composer — scene-gated cards, dry-run-filtered menus, forced outcome prefixes, and depth-slotted steering over a byte-stable cached prefix — means the 0.6B never retrieves, never judges, never adjudicates, and never parses ambiguity; it only discriminates among 5-10 pre-verified options in one batched forward pass against precomputed priors. That compounds every finding into one economic fact: a full game turn costs approximately one prompt evaluation, and every token in it is load-bearing.
