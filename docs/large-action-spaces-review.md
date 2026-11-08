# Large action spaces: a literature review

Simplified Technical English. 2026-09-28. Companion to
`docs/pointer-grammar-scope.md`. **The question.** A small model must pick a
character's action from a legal menu of 5 to 500+ options. Several options
can be acceptable. How do other fields let a model choose among tens to
thousands of actions? Which methods fit our game? **Sources.** We checked
every source on 2026-09-28. For arXiv papers, we read the arXiv record
(title, authors, date) and searched the paper text for each number we
quote. For other sources, we used a web search and the publisher page. We
give a venue only when the paper or its record states it. Numbers in
brackets point to Section 12.

## 1. The problem

| limit | value |
| --- | --- |
| GPU memory at play time | 6 GB laptop: models of about 0.3B to 2B parameters |
| model calls at play time | only the player's parse and the addressed character's reply |
| menu | the sim gives legal options: "take the rusty key", "ask the fisherman about the key" |
| menu size | 5 to 500+ |
| right answer | no single one; an acceptable set of one or more options |

What we measured:

| chooser | 4 options | 16 options | 128 options |
| --- | --- | --- | --- |
| Laya, listwise (ModernBERT [61] reads all options in one pass) | 70% | — | 7% |
| Laya, pairwise (two options at a time) | — | 57% | — |
| 1.7B causal LM, scores each option alone | — | — | 27% (more compute) |

Two more facts. Small decoders prefer the last items of a numbered list.
Distracting text in the situation halves accuracy for all models. We now
test three structures: a tournament (groups of 4–8, winners regroup), a
shortlist (embedding retrieval, then rerank) and pairwise knockouts.

## 2. Terms

| term | meaning |
| --- | --- |
| option, menu | One legal action; the list of them. |
| listwise | The model reads all options at once. It picks one or ranks them. |
| pointwise | The model scores each option alone. We then compare the scores. |
| pairwise | The model compares two options. |
| setwise | The model picks the best of a small set (3 to 9 options). |
| cross-encoder | One model pass reads the situation and one option together. |
| bi-encoder (two-tower) | Situation and option become separate vectors. A dot product compares them. We can cache option vectors. |
| shortlist, recall@K | A cheap first stage keeps K options. Recall@K is the chance that the K hold an acceptable option. |
| position bias | A preference for an option because of its place in the list. Recency bias prefers the last places. |
| calibration | A change to scores. It makes them match true chances, or it removes a known bias. |
| conformal prediction | A method that turns scores into a set. The set holds a right answer with a chosen probability. It needs held-out labeled cases. |
| affordance | What an object lets an agent do in the current state. |
| action masking | Setting the chance of an illegal action to zero. |
| distillation | Training a small model to copy a large one. |
| prefix cache | Reuse of the computation for shared text (the situation) across all options. |

## 3. Six ways to handle a big menu

| way | idea | examples |
| --- | --- | --- |
| A. Filter | Rules or a learned model remove impossible or useless options. | OpenAI Five [31], AE-DQN [27], masking [28], page pruning [1][6] |
| B. Shortlist | A cheap scorer keeps K options. A strong scorer picks among them. | MindAct [1], YouTube [39], Wolpertinger [24], CALM [36] |
| C. Factor | Choose the kind of action first. Then choose its arguments. | OpenAI Five [31], AlphaStar [32], templates [34][35], UFO [7] |
| D. Small groups | Compare 2 to 20 options at a time. Winners regroup, or a sort runs. | MindAct [1], TourRank [55], Setwise [53], PRP [54], ListT5 [51] |
| E. Score each | Score every option alone. No list is in the prompt, so no position bias. | SayCan [15], GLAM [38], monoT5 [45], GENRE [60] |
| F. Generate, then map | The model writes free text. A mapper finds the nearest legal option. | Zero-shot planners [16], Infer–Retrieve–Rank [59] |

Most working systems stack several ways. Calibration (Section 8) applies to all of them.

## 4. Computer-use and web agents

A web page holds over 1,000 elements on average [1]. A desktop accessibility
tree can hold tens of thousands of tokens [6]. No agent shows a model the
full page as one flat list.

| method | how it cuts or structures the choice | needs | sizes shown | fits 0.3–2B laptop? | reported failures |
| --- | --- | --- | --- | --- | --- |
| MindAct [1] | Rules drop invisible elements (1,135 to 580; target kept 94.7%). An 86M DeBERTa cross-encoder keeps the top 50 (recall@50 85–89%). An LM answers multiple choice in groups of 5 plus "None". Selected elements regroup until one remains. | Fine-tuned ranker and chooser (Flan-T5 220M to 3B) on 2,000+ tasks | ~1,000 → 50 → groups of 5 | Yes | The ranker alone as chooser: 26.8% element accuracy; two stages: 55.1%. Zero-shot Flan-T5-XL fails. GPT-3.5 picks "None" too often. Random grouping moves scores by under 1 point. |
| SeeAct [2] | The model first writes what it wants to do. It then picks the element from text choices: MindAct's top 50, in groups of 17. | GPT-4V; no training | 50 candidates | Idea yes; model no | Numbered image marks fail: the model invents labels and links boxes to wrong labels. Best method stays 20–30 points below oracle grounding. |
| Set-of-Mark [3] | Numbered marks drawn on image regions. The model answers with a mark. | Segmenter plus large VLM | Object-centric images | No | Fails on dense screens [2][6]. |
| WebArena, VisualWebArena [4][5] | Accessibility tree: a pruned page tree with an ID per element. Actions are primitives on IDs: click, hover, type, scroll, tab and page operations. | Large LLM | Real web apps | Format yes | GPT-4 agent: 14.41% task success; humans 78.24%. |
| OSWorld [6] | Filters the tree by tag, visibility and text. Optional marks on the screenshot. | Large VLM | 369 tasks | Filter yes | Best model 12.24%; humans 72.36%. Marks hurt GPT-4V. The authors blame dense, high-resolution screens. |
| AgentOccam [8] | Removes actions that LLMs misuse (tab operations, scrolling). Adds note, stop, branch and prune. Simplifies the observation. | No training | WebArena | Yes (a design rule) | Design alone adds 26.6 points over a plain agent. |
| UFO [7] | Two levels: one agent picks the application. A second agent picks a labeled control, then an operation. | GPT-4V | 9 Windows apps | Hierarchy yes | We did not extract failure rates. |
| Grounding models [9][10][11][12] | A model maps a text description to a screen point. This replaces the list. | GUI pretraining (UGround: 10M elements, 1.3M screenshots) | Screen benchmarks | ShowUI is 2B | Grounding is the main bottleneck [2][6]. |
| Interface design [13][14] | Few, clear commands with feedback, built for the model. | Design work | Code repos; a shop with 1.18M products | Yes | WebShop best agent 29%; humans 59%. |

**Lessons.** MindAct is our shortlist plus tournament, and it works at small
sizes. "Say what you want, then pick" separates intent from grounding.
Visual marks do not scale to crowded screens. The action set design is a
lever by itself.

## 5. Robotics

| method | how it cuts or structures the choice | needs | sizes shown | fits laptop? | reported failures |
| --- | --- | --- | --- | --- | --- |
| SayCan [15] | Scores every skill: LM probability of the skill text times a value function ("can it succeed here?"). | PaLM 540B; one trained value function per skill | 551 skills; 101 instructions | Scoring yes; quality unknown at 1B | Plan 84%, execution 74%. Without the value function: 67%. Generate-then-map by embeddings: 74%. 65% of errors come from the LM. Fails on negation. Stops early on long tasks. |
| Zero-shot planners [16] | The LM writes a free-text step. Sentence-RoBERTa (355M) maps it to the nearest admissible action by cosine similarity. Score = best cosine + β × log-probability over k samples. A threshold ends the plan. | GPT-3 175B, Codex; no training | VirtualHome | Mapper yes; planner no | Executability rose from 18% to 79%. Correctness fell. |
| ProgPrompt [19] | Lists available actions as Python imports and objects as a list. The LM writes a program with assertions. | GPT-3 | VirtualHome, tabletop | Idea yes | Needs a strong code LM. |
| Code as Policies [18] | The LM writes code over primitive APIs. The API is the action set. | Codex | Real robots | No | Needs a strong code LM. |
| Inner Monologue [17] | After each step, success checks and scene text go back to the LM. It replans. | InstructGPT, PaLM | 3 domains | Idea yes | Fixes errors after they happen; does not cut the menu. |
| Grounded Decoding [20] | Decodes tokens that score high under the LM and under grounded models (affordance, safety). | Grounded models per domain | Robot tasks | Yes, if grounded models are cheap | — |
| KnowNo [21] | The LM writes options A–D plus "E) not listed". Next-token scores feed conformal prediction. If the set holds more than one option, the robot asks for help. Extends to several acceptable options. | PaLM-2L; about 400 calibration cases; no fine-tuning | 4–5 options per step | Yes (method is model-agnostic) | The guarantee needs calibration cases from the same distribution. |
| RT-2 [22] | Actions become text tokens: 8 numbers, 256 bins each. Decoding samples only valid action tokens. | Large VLA co-fine-tuning | 6,000 trials | No | — |
| Options [23] | Skills with their own start and stop rules. A policy chooses skills, not primitive steps. | Theory | — | Yes (a design) | — |

**Lessons.** SayCan's product of "sounds right" and "can succeed" is the
closest match to our menu. Our sim already supplies the "can succeed" part
as legality. Generate-then-map trades correctness for legality. We do not
need that trade, because the sim gives the list. KnowNo shows how to
return a calibrated set when several options are acceptable.

## 6. Reinforcement learning and text games

A text game with a 700-word vocabulary allows 240 billion four-word
commands. Only about a hundred are valid in a given state [34].

| method | how it cuts or structures the choice | needs | sizes shown | fits laptop? | reported failures |
| --- | --- | --- | --- | --- | --- |
| Wolpertinger [24] | The actor outputs a point in action-embedding space. The k nearest real actions come back. A critic picks the best of the k. | Prior action embeddings; RL | Up to 1M actions; recommenders with 49, 835 and 13,138 items | Idea yes | The refine step (k > 1) adds robustness to a poor embedding. k of 5–10% of actions performs like the full set. |
| Action representations [25][26] | Learn action embeddings; act in embedding space. Handles unseen actions. | Training | Large real sets; unseen tool sets | Yes | — |
| AE-DQN [27] | A network predicts which actions the parser will reject. A contextual bandit removes them with confidence bounds. | A rejection signal | Zork, over 1,000 actions | Yes | Removes only invalid actions. Our sim already does this. |
| Masking, action shaping [28][29] | Set illegal actions to zero. Remove pointless actions by domain rules. | Rules | Strategy and video games | Yes | Masking matters more as the invalid space grows. |
| Factored heads [30][31][32] | Choose the action type, then each argument, each conditioned on earlier choices. OpenAI Five: hand-written filters leave 8.1 primary actions on average; then a unit (of 189), an offset (of 81) and a delay. AlphaStar: about 10^26 legal actions per step; an autoregressive head with a pointer network. | Large-scale RL | Dota: 8,000–80,000 actions per step | Structure yes | — |
| DRRN [33] | Embeds the state and each action text separately. An interaction function scores each pair. | RL | Small text games | Yes | Needs the valid list. |
| Templates, KG-A2C [34][35] | Pick a template ("put _ in _"), then fill blanks. KG-A2C masks fillers with a graph of seen objects. | RL; game templates | Zork1: 237 templates, about 10^8 actions | Structure yes | The template space is still huge without masks. |
| CALM [36] | GPT-2, trained on human play logs, proposes the top 30 actions. DRRN re-ranks them. | Human transcripts; RL | 28 games | Yes | k = 10 is much worse; top-10 lists are mostly simple moves. |
| Word-vector affordance [37] | Word vectors suggest verbs that fit a noun ("what can you do with a rock?"). | Word2vec | Text games | Yes | Helps "most of the time"; adds compute per step. |
| GLAM [38] | The LM scores each action by the probability of its tokens. A softmax gives a policy. PPO fine-tunes the LM. | Online RL | 3 to 9 actions | Yes (Flan-T5 80M, 780M, 3B) | Tested only on small action sets. |

**Lessons.** Large game systems never show one head all actions. They
filter by rule, then factor into "what kind" and "on what". Nearest
neighbours in embedding space need a refine step after them.

## 7. Retrieval and ranking

| method | how it cuts or structures the choice | needs | sizes shown | fits laptop? | reported failures |
| --- | --- | --- | --- | --- | --- |
| Two-stage recommender [39] | Candidate generation from millions. A heavier model ranks the survivors. | Logged data | Millions → hundreds | Pattern yes | — |
| Bi-encoders [40][41] | Encode query and item separately. Cache item vectors. Find nearest neighbours. | Contrastive training | SBERT: 10,000 sentences in 5 s instead of 65 h | Yes | Less accurate than cross-encoders [43]. |
| ColBERT, Poly-encoder [42][43] | Middle ground: cached item encodings plus light interaction. | Training | Poly: 10, 20 and 100 candidates; 10,000 articles | Yes | Cross-encoders win on accuracy but are slow. |
| Cross-encoders [44][45][46] | Read query plus one item; output a score. BERT multiple choice: one pass per option, one shared softmax. | Fine-tuning | SWAG: 4 options | Yes (110M–3B) | Pointwise scores need calibration across items [53]. |
| RankGPT [47] | An LLM ranks a window of 20. The window slides by 10 from the bottom. A distilled 440M student beats a 3B supervised ranker. | ChatGPT, GPT-4; distillation | Top 100 | Student yes | Listwise metrics can drop over 50% when input order changes [54]. |
| RankVicuna, RankZephyr, FIRST [48][49][50] | Open 7B listwise rerankers. FIRST reads a full ranking from the first output token's logits; 50% faster. | Distillation from GPT | Top 100 | No (7B) | — |
| ListT5 [51] | Encodes each passage with the query separately; the decoder reads all (Fusion-in-Decoder). m-ary tournament sort with m = 5. | Fine-tuning T5-base (220M) or 3B | Top 100 | Yes | Reports no lost-in-the-middle effect. |
| Set-Encoder [52] | Each passage is encoded with the query. Shared [INT] tokens pass information between passages. Equal position codes make order irrelevant. | Trained on 8 passages, then 100 per query by distillation | 100 at once | Yes (110M, 330M) | — |
| Setwise [53] | Pick the best of c documents (c = 3 to 9) from label logits. This drives a heap sort. | No training | Top 100; Flan-T5 780M, 3B, 11B | Yes | Listwise generation is far worse at 780M and 3B. Larger c is faster but less accurate. Setwise resists input order. |
| PRP [54] | Compare two items in both orders. Inconsistent answers count as a tie. Aggregate by win counts, heap sort or sliding window. | No training | Top 100; 3B to 20B | Yes (3B shown) | All pairs cost O(N²). Answers are not always transitive. A 20B model gave mostly unusable listwise outputs. |
| TourRank [55] | Seeded groups; the model picks the top m of n per group. Stages 100 → 50 → 20 → 10 → 5 → 2. Order is shuffled in each group. Repeat tournaments; sum points. | No training | Top 100; 7B–13B open models, GPT | 7B shown | One tournament is coarse. Two beat RankGPT; ten were best. |
| Lost in the Middle [56] | A finding: accuracy is U-shaped over the position of the answer. | — | 10, 20, 30 documents | — | GPT-3.5 with the answer in the middle scored below no documents at all. Encoder-decoders are robust only within their training length. |
| Permutation self-consistency [57] | Shuffle the list m times. Merge the rankings (Kemeny). | No training; m calls | Sorting, reranking | Yes | Five shuffles give 67% of the gain of twenty. Theory covers random order noise only. |
| LLMRank [58] | An LLM ranks 20 recommendation candidates. Rank 3 times with shuffles, then merge. | ChatGPT | 20 | — | Accuracy drops when the target sits in the last places. Popularity bias. |
| Infer–Retrieve–Rank [59] | The LM writes guesses. A frozen retriever maps them to labels. The LM re-ranks. | Tens of labeled examples | Thousands of labels | Partly | — |
| GENRE, constrained decoding [60][62][63] | Generate the option's name under a prefix tree of valid names (constrained beam search). | Seq2seq fine-tuning (406M) | About 6M Wikipedia titles | Yes | Generating numeric IDs instead of names lost 20.3 points. |

**Lessons.** Ranking research reached our results first. Small models fail
at "generate a ranking" [53][54]. They hold up when they compare 2 to 9
items and we read their output scores. Position bias needs shuffling and
merging. Encoders that read each item separately resist position bias
[51][52].

## 8. Calibration and choice bias

| method or finding | what it does | needs | fits laptop? | notes |
| --- | --- | --- | --- | --- |
| Contextual calibration [64] | Score a content-free input ("N/A"). Rescale so its prediction is uniform. | One extra pass | Yes | Found majority-label, recency and common-token biases (GPT-2 1.5B, GPT-3). |
| Domain PMI [65] | Divide an option's probability by its probability in a neutral context. | One extra pass per option | Yes | Fixes competition between surface forms of one answer. |
| Batch calibration [66] | Estimates the bias from the batch of test inputs. | No labels | Yes | Documents failure cases of earlier methods. |
| PriDe [67] | Permute options on about 5% of cases. Estimate the prior for option IDs. Divide it out. | Small unlabeled sample | Yes | 20 LLMs, 2 to 5 options. The bias comes mainly from ID tokens (A/B/C/D), less from position. |
| Symbol binding [68] | Measures agreement across option orders. | — | — | GPT-2 (1.5B) and InstructGPT Curie scored near chance: they cannot use numbered-choice prompts. |
| Order sensitivity [69] | Reordering options changes accuracy by 13–75%. | — | — | Bias is largest when the top 2–3 options are close. |
| Both-order judging [70][71] | Judge both orders; combine. | 2× calls | Yes | Order changes let Vicuna-13B "beat" ChatGPT on 66 of 80 queries. |
| MMLU-Pro [72] | 4 → 10 options with harder distractors. | — | — | Accuracy fell 16–33%; harder questions also contribute. |
| Irrelevant context [73] | Adds irrelevant sentences to math problems. | — | — | Accuracy drops sharply. Self-consistency and an "ignore irrelevant text" instruction help. |
| Temperature scaling [74] | One parameter rescales scores to match true frequencies. | Held-out labeled cases | Yes | Simple and strong. |
| Conformal prediction [75] | Turns scores into a set with a coverage guarantee. | Held-out labeled cases | Yes | Assumes test cases resemble calibration cases. |
| Self-consistency [76] | Sample several answers; take the majority. | Many calls | Costly | Also a fix for distraction [73]. |

## 9. What the fields agree on

1. **Nobody gives a small model hundreds of options in one pass.** MindAct
   goes ~1,000 → 580 → 50 → groups of 5 [1]. OpenAI Five filters to about
   8 primary actions [31].
2. **Groups stay small.** Working group sizes: 5 [1][51], 3 to 9 [53], 10 to
   20 with 7B+ models [47][55], 17 with GPT-4V [2]. In these papers, the
   smaller the model, the smaller the group.
3. **Small models compare well but rank badly.** Listwise generation is
   weak at 780M and 3B [53]. A 20B model gave unusable lists [54]. Pairwise
   and setwise with output scores hold at all sizes tested [53].
4. **Names beat labels.** Small models cannot bind "3" to the third option
   [68]. GENRE lost 20.3 points with IDs instead of names [60]. Score the
   option text, not a label token.
5. **Position bias is everywhere; shuffling fixes part of it.** It is worst
   when the top options are close [69]. That is our near-miss case. Five
   shuffles give most of the gain [57].
6. **Train at the list size you test.** Set-Encoder trains on 100
   passages [52]. Encoder-decoders resist position effects only within
   their training length [56]. If Laya trained on short lists, this
   predicts its drop.
7. **Factor the action.** Type first, then arguments [30][31][32][34].
8. **Sets, not single answers, fit acceptable sets.** KnowNo gives a
   calibrated set with several acceptable options [21].

## 10. What this means for us

The latency budget allows model calls in two places only. They are the
player's parse and the addressed character's reply. The pipeline below runs
there. It can also label data offline for the kNN harvest.

### 10a. Methods to test, ranked

| rank | method | training? | evidence | what to vary | our prediction |
| --- | --- | --- | --- | --- | --- |
| 1 | **Shuffled small-group tournament with points.** Laya picks in groups of 4 (its best size). Seed groups by a cheap score; shuffle inside groups; top 2 advance; repeat 2–3 times; sum points. | None | [1][53][55][57] | group 4 vs 8; top-1 vs top-2; 1 vs 3 repeats; seeded vs random | Beats one-pass Laya from 32 options up. At 128: groups of 4, top-2 is 63 reads in 6 batched rounds. |
| 2 | **Pointwise text scoring with calibration.** A 0.6B–1.7B LM scores each option's text after the situation. Use a prefix cache, length normalization, and a content-free prior [64][65]. | None | [15][38][60]; our `lab/calib.py` | raw vs length-normalized vs PMI; 0.6B vs 1.7B | Flattest curve over list size. Best as the shortlist scorer for rank 1. |
| 3 | **Factored choice from the sim's structure.** Options carry `verb` and `args`. Choose among verbs first (keep the top 2), then among arguments. | None | [7][31][32][34] | on vs off at 64+ options; verb score = best child vs verb text | Example: 500 options over 20 verbs become 20 + 25 choices. Risk: an early wrong verb. |
| 4 | **Pairwise final, both orders.** All pairs among the last 4–8; win counts; ties when orders disagree. | None | [54][70][71] | final 4 vs 8; pairwise vs Laya group-of-4 with 4 cyclic shifts | Helps near-misses. Eight finalists cost 56 reads in 1 round. |
| 5 | **Trained shortlist ranker.** Fine-tune a small cross-encoder (ModernBERT-base) on (situation, option) → acceptable, with near-miss negatives from the harvest. | Yes (off-laptop) | [1][44][46] | K = 8, 16, 32; report recall@K | Needed if rank 2's recall@16 is low. Laya's own embedding shortlist did worse (`pointer-grammar-scope.md` §0). |
| 6 | **Retrain Laya for long lists.** Train on 32–128 options with random order and near-miss negatives. Encode each option with the situation; share one softmax [46], or use [INT] tokens [52] or Fusion-in-Decoder [51]. Distill from ranks 1–4 [47]. | Yes (off-laptop) | [46][47][51][52] | train size 8 vs 128; test 4–128 | Raises the 128-option score without loss at 4. |

**Test factors.** Record acceptable@1 and recall@K for shortlist stages.
Also record survival per round, reads and rounds per decision, laptop
latency, and agreement across shuffles [68].

| factor | levels | literature predicts |
| --- | --- | --- |
| list size | 4, 8, 16, 32, 64, 128 | One-pass listwise decays [56][72]. Tournaments decay with depth. Pointwise stays flatter. |
| option length | short vs long phrasing | Longer options force fewer per group [53]. Raw LM scores depend on surface form and string frequency [64][65]; normalize them. |
| position | acceptable first, middle, last | U-shape [56]; last-place drop [58]; recency in small decoders [64]. Shuffled methods should be flat. |
| near-miss distractors | 0, 1, 3 near-misses | Bias peaks when the top options are close [69]. Pairwise both-order finals should help most here. |
| situation length | short, long, long with distractors | Accuracy halves with distracting text [73]. Test pruning the situation [8] and repeating the need after the options [56]. |

### 10b. One pipeline

```
sim legal menu (5 to 500+)
 0. legality from the sim only; no preference rules                  [27][28]
 1. factor: if > 64 options, choose among verbs, keep the top 2      [31][32]
 2. shortlist: pointwise LM or trained ranker keeps K = 16 to 32     [1][15]
 3. tournament: groups of 4, seeded, shuffled, top 2 advance,
    2 to 3 repeats, points summed                                     [55][57]
 4. final: pairwise, both orders, among the top 4 to 8               [54]
 5. calibrated draw: temperature-scaled probabilities -> conformal
    set with >= 1 acceptable option at rate 1 - e -> weighted draw   [21][74][75]
```

| menu size | stages used |
| --- | --- |
| up to 4 | Laya once per cyclic shift (4 reads), then stage 5 |
| 5 to 16 | stages 3, 4, 5 |
| 17 to 64 | stages 2, 3, 4, 5 |
| over 64 | stages 1 to 5 |

Stage 5 fits "several acceptable". Fit the temperature and the conformal
threshold on about 400 labeled situations, as KnowNo did [21]. Then draw
from the set with the character's trait weights. If the set is empty, take
the top option. Measure coverage (the set holds an acceptable option) and
mean set size.

### 10c. Risks

1. **Errors compound over rounds.** If each round keeps an acceptable option
   with chance p, d rounds keep it with about p^d. With p = 0.7 and d = 4,
   that is 0.24. Top-2 advance and repeats reduce this. Measure survival
   per round.
2. **The shortlist caps accuracy.** No later stage recovers an option that
   the shortlist dropped. Report recall@K before anything else.
3. **Latency grows with rounds.** Reads in one round batch together. Rounds
   run in sequence. Count rounds, not only reads.
4. **Bias survives small groups.** Groups of 4 still carry order and label
   effects [67][69]. Shuffle every repeat. Prefer scoring option text to
   label tokens [60][68].
5. **Merging fixes random noise, not consistent errors.** The theory in
   [57] covers random order noise. A model that always prefers a near-miss
   will still prefer it after merging.
6. **Pairwise answers can cycle.** PRP notes no transitivity guarantee [54].
   Use win counts, not a single elimination path, in the final.
7. **Calibration drifts.** Conformal guarantees need calibration cases like
   the test cases [21][75]. Recalibrate when menus or models change.
8. **Filters must not replace cognition.** Stage 0 removes only what the sim
   marks illegal. Preferences stay with the chooser (the magic_duel lesson).
9. **Benchmark gap.** Synthetic near-misses may not match real sim menus.
   With twenty cases per cell, a difference of 2–3 cases is noise
   (`pointer-grammar-scope.md`).
10. **Situation text is a separate problem.** No stage above fixes
    distracting context [73]. Test situation pruning on its own.

## 11. What we left out or could not check

- We could not open the Nature full text for AlphaStar [32]. We verified
  the record by web search. The action-space size and the pointer-network
  head come from DeepMind's blog [32].
- When a paper's record states no venue, we list it as arXiv. Some of
  these papers later appeared at conferences.
- We found no paper that sweeps group sizes below 5 for sub-1B encoders.
  Setwise tests c = 3 to 9 with Flan-T5 [53]. Our group-of-4 choice rests
  on our own Laya numbers.

## 12. References

1. Deng, Gu, Zheng, et al. (2023). Mind2Web: Towards a Generalist Agent for the Web. NeurIPS 2023. https://arxiv.org/abs/2306.06070
2. Zheng, Gou, Kil, Sun, Su (2024). GPT-4V(ision) is a Generalist Web Agent, if Grounded. arXiv. https://arxiv.org/abs/2401.01614
3. Yang, Zhang, Li, et al. (2023). Set-of-Mark Prompting Unleashes Extraordinary Visual Grounding in GPT-4V. arXiv. https://arxiv.org/abs/2310.11441
4. Zhou, Xu, Zhu, et al. (2023). WebArena: A Realistic Web Environment for Building Autonomous Agents. arXiv. https://arxiv.org/abs/2307.13854
5. Koh, Lo, Jang, et al. (2024). VisualWebArena: Evaluating Multimodal Agents on Realistic Visual Web Tasks. ACL 2024. https://arxiv.org/abs/2401.13649
6. Xie, Zhang, Chen, et al. (2024). OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments. arXiv. https://arxiv.org/abs/2404.07972
7. Zhang, Li, He, et al. (2024). UFO: A UI-Focused Agent for Windows OS Interaction. arXiv. https://arxiv.org/abs/2402.07939
8. Yang, Liu, Chaudhary, et al. (2024). AgentOccam: A Simple Yet Strong Baseline for LLM-Based Web Agents. ICLR 2025. https://arxiv.org/abs/2410.13825
9. Cheng, Sun, Chu, et al. (2024). SeeClick: Harnessing GUI Grounding for Advanced Visual GUI Agents. arXiv. https://arxiv.org/abs/2401.10935
10. Gou, Wang, Zheng, et al. (2024). Navigating the Digital World as Humans Do: Universal Visual Grounding for GUI Agents. ICLR 2025. https://arxiv.org/abs/2410.05243
11. Lu, Yang, Shen, Awadallah (2024). OmniParser for Pure Vision Based GUI Agent. arXiv. https://arxiv.org/abs/2408.00203
12. Lin, Li, Gao, et al. (2024). ShowUI: One Vision-Language-Action Model for GUI Visual Agent. arXiv. https://arxiv.org/abs/2411.17465
13. Yang, Jimenez, Wettig, et al. (2024). SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering. arXiv. https://arxiv.org/abs/2405.15793
14. Yao, Chen, Yang, Narasimhan (2022). WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents. NeurIPS 2022. https://arxiv.org/abs/2207.01206
15. Ahn, Brohan, Brown, et al. (2022). Do As I Can, Not As I Say: Grounding Language in Robotic Affordances. arXiv. https://arxiv.org/abs/2204.01691
16. Huang, Abbeel, Pathak, Mordatch (2022). Language Models as Zero-Shot Planners: Extracting Actionable Knowledge for Embodied Agents. arXiv. https://arxiv.org/abs/2201.07207
17. Huang, Xia, Xiao, et al. (2022). Inner Monologue: Embodied Reasoning through Planning with Language Models. arXiv. https://arxiv.org/abs/2207.05608
18. Liang, Huang, Xia, et al. (2022). Code as Policies: Language Model Programs for Embodied Control. arXiv. https://arxiv.org/abs/2209.07753
19. Singh, Blukis, Mousavian, et al. (2022). ProgPrompt: Generating Situated Robot Task Plans using Large Language Models. arXiv. https://arxiv.org/abs/2209.11302
20. Huang, Xia, Shah, et al. (2023). Grounded Decoding: Guiding Text Generation with Grounded Models for Embodied Agents. arXiv. https://arxiv.org/abs/2303.00855
21. Ren, Dixit, Bodrova, et al. (2023). Robots That Ask For Help: Uncertainty Alignment for Large Language Model Planners. CoRL 2023. https://arxiv.org/abs/2307.01928
22. Brohan, Brown, Carbajal, et al. (2023). RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control. arXiv. https://arxiv.org/abs/2307.15818
23. Sutton, Precup, Singh (1999). Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning. Artificial Intelligence 112(1–2):181–211. https://doi.org/10.1016/S0004-3702(99)00052-1
24. Dulac-Arnold, Evans, van Hasselt, et al. (2015). Deep Reinforcement Learning in Large Discrete Action Spaces. arXiv. https://arxiv.org/abs/1512.07679
25. Chandak, Theocharous, Kostas, et al. (2019). Learning Action Representations for Reinforcement Learning. ICML 2019. https://arxiv.org/abs/1902.00183
26. Jain, Szot, Lim (2020). Generalization to New Actions in Reinforcement Learning. ICML 2020. https://arxiv.org/abs/2011.01928
27. Zahavy, Haroush, Merlis, et al. (2018). Learn What Not to Learn: Action Elimination with Deep Reinforcement Learning. NeurIPS 2018. https://arxiv.org/abs/1809.02121
28. Huang, Ontañón (2020). A Closer Look at Invalid Action Masking in Policy Gradient Algorithms. FLAIRS 35 (2022). https://arxiv.org/abs/2006.14171
29. Kanervisto, Scheller, Hautamäki (2020). Action Space Shaping in Deep Reinforcement Learning. IEEE CoG 2020. https://arxiv.org/abs/2004.00980
30. Tavakoli, Pardo, Kormushev (2017). Action Branching Architectures for Deep Reinforcement Learning. AAAI 2018. https://arxiv.org/abs/1711.08946
31. OpenAI: Berner, Brockman, et al. (2019). Dota 2 with Large Scale Deep Reinforcement Learning. arXiv. https://arxiv.org/abs/1912.06680
32. Vinyals, Babuschkin, Czarnecki, et al. (2019). Grandmaster level in StarCraft II using multi-agent reinforcement learning. Nature 575:350–354. https://www.nature.com/articles/s41586-019-1724-z ; blog: https://deepmind.google/blog/alphastar-mastering-the-real-time-strategy-game-starcraft-ii/
33. He, Chen, He, et al. (2015). Deep Reinforcement Learning with a Natural Language Action Space. ACL 2016. https://arxiv.org/abs/1511.04636
34. Hausknecht, Ammanabrolu, Côté, Yuan (2019). Interactive Fiction Games: A Colossal Adventure. arXiv. https://arxiv.org/abs/1909.05398
35. Ammanabrolu, Hausknecht (2020). Graph Constrained Reinforcement Learning for Natural Language Action Spaces. ICLR 2020. https://arxiv.org/abs/2001.08837
36. Yao, Rao, Hausknecht, Narasimhan (2020). Keep CALM and Explore: Language Models for Action Generation in Text-based Games. EMNLP 2020. https://arxiv.org/abs/2010.02903
37. Fulda, Ricks, Murdoch, Wingate (2017). What can you do with a rock? Affordance extraction via word embeddings. IJCAI 2017. https://arxiv.org/abs/1703.03429
38. Carta, Romac, Wolf, et al. (2023). Grounding Large Language Models in Interactive Environments with Online Reinforcement Learning. ICML 2023 (PMLR 202). https://arxiv.org/abs/2302.02662
39. Covington, Adams, Sargin (2016). Deep Neural Networks for YouTube Recommendations. RecSys 2016. https://www.semanticscholar.org/paper/Deep-Neural-Networks-for-YouTube-Recommendations-Covington-Adams/5e383584ccbc8b920eaf3cfce3869da646ff5550
40. Karpukhin, Oğuz, Min, et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering. EMNLP 2020. https://arxiv.org/abs/2004.04906
41. Reimers, Gurevych (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP 2019. https://arxiv.org/abs/1908.10084
42. Khattab, Zaharia (2020). ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT. SIGIR 2020. https://arxiv.org/abs/2004.12832
43. Humeau, Shuster, Lachaux, Weston (2019). Poly-encoders: Transformer Architectures and Pre-training Strategies for Fast and Accurate Multi-sentence Scoring. ICLR 2020. https://arxiv.org/abs/1905.01969
44. Nogueira, Cho (2019). Passage Re-ranking with BERT. arXiv. https://arxiv.org/abs/1901.04085
45. Nogueira, Jiang, Lin (2020). Document Ranking with a Pretrained Sequence-to-Sequence Model. arXiv. https://arxiv.org/abs/2003.06713
46. Devlin, Chang, Lee, Toutanova (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv. https://arxiv.org/abs/1810.04805
47. Sun, Yan, Ma, et al. (2023). Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents. EMNLP 2023. https://arxiv.org/abs/2304.09542
48. Pradeep, Sharifymoghaddam, Lin (2023). RankVicuna: Zero-Shot Listwise Document Reranking with Open-Source Large Language Models. arXiv. https://arxiv.org/abs/2309.15088
49. Pradeep, Sharifymoghaddam, Lin (2023). RankZephyr: Effective and Robust Zero-Shot Listwise Reranking is a Breeze! arXiv. https://arxiv.org/abs/2312.02724
50. Reddy, Doo, Xu, et al. (2024). FIRST: Faster Improved Listwise Reranking with Single Token Decoding. arXiv. https://arxiv.org/abs/2406.15657
51. Yoon, Choi, Kim, et al. (2024). ListT5: Listwise Reranking with Fusion-in-Decoder Improves Zero-shot Retrieval. ACL 2024. https://arxiv.org/abs/2402.15838
52. Schlatt, Fröbe, Scells, et al. (2024). Set-Encoder: Permutation-Invariant Inter-Passage Attention for Listwise Passage Re-Ranking with Cross-Encoders. ECIR 2025. https://arxiv.org/abs/2404.06912
53. Zhuang, Zhuang, Koopman, Zuccon (2023). A Setwise Approach for Effective and Highly Efficient Zero-shot Ranking with Large Language Models. SIGIR 2024. https://arxiv.org/abs/2310.09497
54. Qin, Jagerman, Hui, et al. (2023). Large Language Models are Effective Text Rankers with Pairwise Ranking Prompting. NAACL 2024. https://arxiv.org/abs/2306.17563
55. Chen, Liu, Zhang, et al. (2024). TourRank: Utilizing Large Language Models for Documents Ranking with a Tournament-Inspired Strategy. WWW 2025. https://arxiv.org/abs/2406.11678
56. Liu, Lin, Hewitt, et al. (2023). Lost in the Middle: How Language Models Use Long Contexts. TACL. https://arxiv.org/abs/2307.03172
57. Tang, Zhang, Ma, et al. (2023). Found in the Middle: Permutation Self-Consistency Improves Listwise Ranking in Large Language Models. NAACL 2024. https://arxiv.org/abs/2310.07712
58. Hou, Zhang, Lin, et al. (2023). Large Language Models are Zero-Shot Rankers for Recommender Systems. ECIR 2024. https://arxiv.org/abs/2305.08845
59. D'Oosterlinck, Khattab, Remy, et al. (2024). In-Context Learning for Extreme Multi-Label Classification. arXiv. https://arxiv.org/abs/2401.12178
60. De Cao, Izacard, Riedel, Petroni (2020). Autoregressive Entity Retrieval. ICLR 2021. https://arxiv.org/abs/2010.00904
61. Warner, Chaffin, Clavié, et al. (2024). Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference. arXiv. https://arxiv.org/abs/2412.13663
62. Geng, Josifoski, Peyrard, West (2023). Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning. EMNLP 2023. https://arxiv.org/abs/2305.13971
63. Willard, Louf (2023). Efficient Guided Generation for Large Language Models. arXiv. https://arxiv.org/abs/2307.09702
64. Zhao, Wallace, Feng, et al. (2021). Calibrate Before Use: Improving Few-Shot Performance of Language Models. ICML 2021. https://arxiv.org/abs/2102.09690
65. Holtzman, West, Shwartz, et al. (2021). Surface Form Competition: Why the Highest Probability Answer Isn't Always Right. arXiv. https://arxiv.org/abs/2104.08315
66. Zhou, Wan, Proleev, et al. (2023). Batch Calibration: Rethinking Calibration for In-Context Learning and Prompt Engineering. ICLR 2024. https://arxiv.org/abs/2309.17249
67. Zheng, Zhou, Meng, et al. (2023). Large Language Models Are Not Robust Multiple Choice Selectors. ICLR 2024. https://arxiv.org/abs/2309.03882
68. Robinson, Rytting, Wingate (2022). Leveraging Large Language Models for Multiple Choice Question Answering. ICLR 2023. https://arxiv.org/abs/2210.12353
69. Pezeshkpour, Hruschka (2023). Large Language Models Sensitivity to The Order of Options in Multiple-Choice Questions. arXiv. https://arxiv.org/abs/2308.11483
70. Wang, Li, Chen, et al. (2023). Large Language Models are not Fair Evaluators. arXiv. https://arxiv.org/abs/2305.17926
71. Zheng, Chiang, Sheng, et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. NeurIPS 2023 Datasets and Benchmarks. https://arxiv.org/abs/2306.05685
72. Wang, Ma, Zhang, et al. (2024). MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark. NeurIPS 2024 Datasets and Benchmarks. https://arxiv.org/abs/2406.01574
73. Shi, Chen, Misra, et al. (2023). Large Language Models Can Be Easily Distracted by Irrelevant Context. ICML 2023. https://arxiv.org/abs/2302.00093
74. Guo, Pleiss, Sun, Weinberger (2017). On Calibration of Modern Neural Networks. ICML 2017. https://arxiv.org/abs/1706.04599
75. Angelopoulos, Bates (2021). A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification. arXiv. https://arxiv.org/abs/2107.07511
76. Wang, Wei, Schuurmans, et al. (2022). Self-Consistency Improves Chain of Thought Reasoning in Language Models. ICLR 2023. https://arxiv.org/abs/2203.11171
