# Can small models learn near misses? A literature review

2026-09-29. Companion to `lab/methods/learnable.py` and
`lab/methods/finetune_gliner.py`. Web review only: no code runs, no GPU.

**The question.** Kael needs to cut the rope. Which thing can he use? With
no training, every model we test (GLiNER2.5-Decide, Qwen3-0.6B/1.7B) is at
chance when the wrong options only LOOK useful (a butter knife, a comb, a
whetstone). The models pick what the need is ABOUT (the rope), not what does
the job. The worry: a zero-shot null says nothing about what training can
do, and we know of no case where a small model learned this. What does the
literature say, for and against?

**Sources.** I opened every source I cite. For arXiv and ACL papers I read
the PDF text and searched it for each number I quote. Numbers read off a
figure, or not checked, are marked. Numbers in brackets point to Section 4.

**Terms.**

| term | meaning |
| --- | --- |
| near miss | a wrong option that looks right: same kind of object, or linked to the need by topic |
| zero-shot | the model is used as it is, with no training on the task |
| fine-tune | further training of the whole model (or of a LoRA adapter) on task examples |
| LoRA | a small add-on to a frozen model; only the add-on is trained |
| probe | a small classifier (often linear) trained on FROZEN model vectors; the model itself does not change |
| random split | train and test examples are mixed at random, so test items and needs were seen in training |
| held-out split | whole objects (or whole needs, or whole tasks) are kept out of training |
| hard negative | a wrong example chosen because it is close to a right one |
| shortcut | a cue that predicts the label in the data but is not the skill (for example, word overlap) |
| selectivity | probe accuracy on the real task minus probe accuracy on a random "control" task |

## 1. Summary

1. **Small encoders do learn physical and affordance judgements when you fine-tune them.** On PIQA, RoBERTa-large (355M) reaches 77.1% (chance 50%, humans 94.9%). About 60% of PIQA pairs differ by only 1 or 2 words, so these are near misses [1]. With 124 examples, RoBERTa-large goes from 0.55 to 0.77 on object affordances, but those test objects were also in training [5]. So "no small model ever learned this" is not correct.
2. **A zero-shot null is weak evidence against learnability.** In a similar trap for BERT-base, about 400 targeted counterexamples (0.1% of the training data) moved accuracy from 0.28 to 0.73, and the gain carried over to a construction the model never saw (0.33 to 0.83) [14]. The authors call this a "missed connection": the information was in the model, but the training data never showed how to use it.
3. **Our exact failure is documented.** BERT-large and RoBERTa-large can tell a related object from an unrelated one, but they cannot tell an afforded use from a non-afforded one (a shirt versus glasses, to dry your feet). Of the models tested, only GPT-3 (175B) could, and it showed only about a third of the human effect [6]. Property tests lose up to 25 points when the wrong concept is similar to the right one [7].
4. **Fine-tuned models often learn a shortcut, not the skill.** On PIQA, a fine-tuned RoBERTa-large scores 0.795 with the goal and 0.735 without it [2]. So most of what it learned is "which answer is sensible in general", not "which answer fits this goal". For us, the same shortcut is an item prior: "a knife is useful". Training on PIQA also made a 3B model slightly worse on a different physical test [3].
5. **Results on held-out objects are mixed, and held-out tasks are the weak point.** With objects held out, fine-tuned BERT still matches objects to their affordances (micro-F1 0.88, but always guessing the most common answer gets 0.82). It fails to link affordances to properties (0.37; random guessing gets 0.23) [4]. A fine-tuned T5-11B loses more than 18 points on unseen objects. A 117M model grounded in a simulator loses only about 3 points [10].
6. **Hard negatives help small models a lot, but they are easy to over-read.** With one hard negative per question, a BERT-base retriever goes from 52.1 to 65.0 top-5 accuracy [15]. With swapped-word hard negatives, CLIP goes from 0.46 to 0.86, where plain fine-tuning gives 0.36 [16]. But a large part of the CLIP gain disappears (to less than 10 points) on a test whose negatives were made in a different way [17]. And when many hard negatives are really positives, they make the model worse than using no hard negatives: 26.0 against 32.4 MRR@10 (a ranking score) [18].
7. **Probes on frozen vectors can read object properties for unseen objects, but only moderately.** On unseen nouns, a linear SVM on BERT-large vectors gets F1 63 to 64 for properties, against 40 for GloVe [9]. Probes trained on 14 objects beat zero-shot on unseen objects [8]. A high probe score does not prove the model uses the information [20]. Probes need a control task [19].
8. **No one has published fine-tuning results for GLiNER2 or GLiNER2.5-Decide on a task like ours.** In the same model family, 8 examples per label add 0.11 to 0.21 F1 [24]. GLiNER handles new label types only after training on about 13,000 of them [25]. We have 38 needs, so a test on new needs asks a lot of the data.

## 2. Evidence

### 2.1 Table

"—" means the study did not test it. "RS" = random split. "HO" = held out.

| dataset / study | model and size | trained how | RS result | HO objects / tasks | near-miss / adversarial | URL |
| --- | --- | --- | --- | --- | --- | --- |
| PIQA [1] | RoBERTa-large 355M; BERT-large 340M; GPT 124M | full fine-tune, 16k pairs | test 77.1 / 66.8 / 69.2; human 94.9; chance 50 | — (split made by adversarial filtering, not by object) | ~60% of pairs differ by 1–2 words. RoBERTa by pivot word: spoon 90%, water 75%, freeze 66%; before/after/top/bottom ≈ chance | https://arxiv.org/abs/1911.11641 |
| Shortcutted Commonsense [2] | RoBERTa-large; COMET(BART) | fine-tune, 5 seeds | PIQA dev 0.795 | cross-task, zero-shot: PIQA-trained → CSQA 0.552 (chance 0.2), → ARC 0.230 (chance 0.25); CSQA-trained → PIQA 0.687 (chance 0.5) | goal removed: 0.735. Word-swap attacks: drop over 35% on PIQA. 13.2% of PIQA dev overlaps pretraining text | https://aclanthology.org/2021.emnlp-main.113/ |
| PROST [3] | BERT, RoBERTa, ALBERT, GPT-2, T5; UnifiedQA to 3B | zero-shot; UnifiedQA-3B also fine-tuned on PIQA | 4-way, chance 25. Best pretrained macro 31.8; RoBERTa-L 31.3 | UnifiedQA-3B trained on PIQA: PROST 47.4 early, 45.3 at best PIQA (87.1) | mean gap of 17.6 points between a question and its most/least inverse | https://arxiv.org/abs/2106.03634 |
| Physical commonsense [4] | BERT (size not checked); ELMo; GloVe | BERT fully fine-tuned; MLP on the others | — | objects HO (411/103; 64/16). Object–affordance µF1 0.88 (majority 0.82, human 0.80). Object–property 0.74. Affordance–property 0.37 (random 0.23) | — | https://arxiv.org/abs/1908.02899 |
| Text2Afford [5] | BERT-large; RoBERTa-large; LLMs to 40B | NLI zero-shot; few-shot fine-tune on 124 examples | RoBERTa-L 0.55 → 0.77; BERT-L 0.46 → 0.58; LLMs <0.55 | NOT HO: all 62 objects appear in training | errors on uncommon affordances ("table salt" → sit on, from "table") | https://arxiv.org/abs/2402.12881 |
| Affordances [6] | BERT-large, RoBERTa-large (340M); GPT-3 175B | zero-shot measures | — | — | afforded vs non-afforded: no effect in BERT/RoBERTa; GPT-3 yes, ~1/3 of human effect. Related vs afforded: detected | https://pages.ucsd.edu/~bkbergen/papers/cogsci_2022_nlm_affordances_final.pdf |
| COMPS [7] | 22 pretrained LMs | zero-shot minimal pairs | random wrong concept: 0.71–0.90 | — | similar wrong concept: 0.56–0.71 (up to 25 points lower). With distracting text: below chance for many models | https://arxiv.org/abs/2210.01963 |
| CoDa (colour) [8] | GPT-2, RoBERTa, ALBERT (base–xxl); CLIP | zero-shot; MLP probe on frozen vectors | — | objects HO: with 14 training objects, every model beats its zero-shot score (JSD, a distance between distributions) | — | https://arxiv.org/abs/2110.08182 |
| Noun properties [9] | BERT-large, RoBERTa-large (averaged masked-mention vectors) | linear SVM per property, frozen | — | nouns HO. X-McRae F1 62.8–64.1 (GloVe 40.1); CSLB 46.5–50.9 (GloVe 31.8) | — | https://arxiv.org/abs/2012.07580 |
| PIGLeT [10] | 117M LM + 17M dynamics model trained in a simulator; T5 small–11B | fine-tune | PIGLeT >80% (all attributes right) | T5-11B: ~77 seen, more than 18 points lower on unseen objects. PIGLeT: ~3 points lower | — | https://arxiv.org/abs/2106.00188 |
| TANGO [11] | graph network + ConceptNet vectors (not an LM) | trained on demonstrations in a simulator | plan success (home) 92.3 | new tool swapped in: 81.6 (baseline that ignores the scene: 29.3; FastText vectors: 74.6). Usual tool removed: 77.5 | unrelated object swapped in: 59.7 | https://arxiv.org/abs/2105.04556 |
| NEWTON [12] | BERT-base (after SWAG) | fine-tune on 5k/10k/40k explicit-attribute questions | — | accuracy on an unseen implicit-reasoning track rises with more samples (numbers only in a figure; not read) | — | https://arxiv.org/abs/2310.07018 |
| Commonsense transfer [13] | RoBERTa ensemble | fine-tuned per benchmark | >80% average in-domain | "clear evidence" of weak transfer across 5 benchmarks (per-pair numbers not extracted) | — | https://arxiv.org/abs/2011.09159 |
| Syntactic augmentation [14] | BERT-base | MNLI + 101 / 405 / 1215 generated examples | MNLI unchanged | unseen construction (relative clauses): 0.33 → 0.83 | lexical-overlap trap: 0.28 → 0.73 | https://arxiv.org/abs/2004.11999 |
| DPR [15] | BERT-base dual encoder | in-batch negatives + 1 BM25 hard negative | NQ dev top-5 52.1 → 65.0; top-20 70.8 → 77.3 | — | the hard negatives share words with the question but lack the answer | https://arxiv.org/abs/2004.04906 |
| NegCLIP [16] | CLIP ViT-B/32 | fine-tune on COCO with word-swapped negatives | downstream tasks about the same | VG-Relation 0.63 → 0.81 (plain fine-tune 0.63) | COCO order 0.46 → 0.86 (plain fine-tune 0.36) | https://arxiv.org/abs/2210.01936 |
| SugarCrepe [17] | CLIP models incl. NegCLIP | re-test with new, debiased negatives | — | — | text-only "blind" models beat CLIP on older tests. NegCLIP gains shrink to <10 points; largest when train and test negatives are the same type | https://arxiv.org/abs/2306.14610 |
| RocketQA [18] | dual and cross encoders | hard negatives, with and without denoising | MS MARCO MRR@10: in-batch 32.4; hard negatives 26.0; denoised hard negatives 36.4 | — | 70% of top-ranked "negatives" they checked were really positives | https://arxiv.org/abs/2010.08191 |
| GLiClass [24] | DeBERTa-v3 base/large, ModernBERT | 8 examples per label; LoRA post-training (r = 384 to 1536) | +0.106 (large) to +0.209 F1 over zero-shot | — | — | https://arxiv.org/abs/2508.07662 |
| GLiNER [25] | DeBERTa-v3 | trained on Pile-NER (~13k entity types), random negative types | — | beats ChatGPT zero-shot on out-of-domain NER (per paper) | — | https://arxiv.org/abs/2311.08526 |
| SmolLM2 [26] | SmolLM2-1.7B, Llama3.2-1B, Qwen2.5-1.5B (base) | zero-shot, answer likelihood | PIQA 77.6 / 74.8 / 76.1 | — | PIQA pairs are mostly 1–2 word edits [1] | https://arxiv.org/abs/2502.02737 |
| VEC [27] | BERT, RoBERTa, OPT 125M–175B | zero-shot; distil CLIP into OPT | BERT and RoBERTa near random on visual and "embodied" concepts | distillation: +3.4 points (around 52 → 56) | — | https://arxiv.org/abs/2305.14057 |
| COAT [30] | 6B–13B open LLMs, GPT-3.5, PaLM | zero-shot; PaLM only fine-tuned | substitute object: PaLM 98.6 (2 options) → 92.7 (5) | — | context task, Mistral-7B: 76.2 (2 options) → 47.2 (5) | https://arxiv.org/abs/2311.13577 |
| MacGyver [28] | GPT-4, Claude 2, PaLM 2, Llama 2 | zero-shot, free text | — | — | LLMs often propose physically infeasible uses | https://arxiv.org/abs/2311.09682 |
| CreativityBench [29] | GPT-5 family, Gemini, Qwen3 4B–32B | zero-shot, free text | right object: 0.36–0.62; right object AND part: 0.12–0.26 (Qwen3-4B 0.19) | — | distractors with SIMILAR affordances made tasks EASIER than dissimilar ones | https://arxiv.org/abs/2605.02910 |

### 2.2 What the table says, question by question

**Q1. Were small models fine-tuned on this kind of task, and did it work?**
Yes. RoBERTa-large on PIQA [1], BERT on object affordances [4], RoBERTa-large
on Text2Afford [5] and BERT on NEWTON [12] all improve with training. No
source fine-tunes a small model on "pick the object that does the job, from
a list of look-alikes" with held-out needs. The creative tool-use benchmarks
(MacGyver [28], CreativityBench [29], COAT [30]) test large models zero-shot
only. COAT fine-tunes only PaLM.

**Q2. Does it generalize to held-out objects, held-out tasks and near misses?**
- Held-out objects: often yes, for direct object–affordance links [4] and
  object properties [8][9]. It works best when the training has a grounded
  signal: a simulator [10][11] or a knowledge-graph vector [11].
- Held-out tasks: weak. PIQA training does not carry over to PROST [3].
  Transfer between commonsense benchmarks is mixed: some pairs are well
  above chance, and one is below it [2][13]. Linking a property to an
  affordance is the hardest part [4].
- Near misses: this is where zero-shot models fail [6][7]. Training can fix
  a near-miss trap, and the fix can carry over [14]. But the fix can also
  be a shortcut that only works on negatives made the same way [17].
- Shortcuts: an answer-only model does almost as well on PIQA [2]. Text-only
  "blind" models beat vision models on older hard-negative tests [17].

**Q3. Are affordances linearly readable from frozen small encoders, and does
that predict fine-tuning?** Object properties are moderately readable, even
for unseen nouns [9], and with very few training objects [8]. I found no
probe of "object × need fit" (as opposed to a property of the object alone).
Across 25 models, probe scores predicted GLUE (a general language benchmark)
fine-tuning scores. The error was 40–80% smaller than baselines [21]. That
compares models; it does not show that one probe predicts whether one task
can be learned. Two warnings: a probe can find information the model does
not use [20], and a flexible probe can memorize. Use a control task and
report selectivity [19].

**Q4. Do hard negatives teach near-miss discrimination?** They help in small
retrievers [15] and in CLIP [16]. The CLIP ablation matters: fine-tuning
without hard negatives made the order tests WORSE (0.46 → 0.36), and with
them the score rose to 0.86 [16]. Two limits: the gain shrinks when the test
negatives come from a different generator [17], and false negatives hurt [18].

**Q5. GLiNER / GLiNER2 / small decision models.** The GLiNER2 paper reports
only zero-shot results. Its classifier reads all labels in one pass [22].
The GLiNER2.5-Decide model card says: 340M, DeBERTa-v3-large, 60.2% on its
own 17-domain decision benchmark, "not a general-purpose model. It does not
reason" [23]. It gives no fine-tuning results. The nearest evidence is GLiClass,
a GLiNER-style classifier: big few-shot gains, LoRA ranks of 384 to 1536 [24].
GLiNER learned new label types from about 13k training types, with random
negative types added [25].

## 3. What this predicts for our tests

**Our data.** 1,500 lists: 1,260 "things" lists (about 23 options each) and
240 creative lists (10 options each). 38 needs, 186 things, about 1.3
acceptable options per list. The fine-tune holds out 8 needs and 15 things.
GLiNER-Decide reads the situation and all options together. So each option
vector already "sees" the need.

### 3.1 Predictions

| test | what the literature predicts | why | surprising if |
| --- | --- | --- | --- |
| 5a probe, **random** split, all three tasks | high, clearly above the shuffled control | items and needs repeat, so the probe can memorize (item, need) pairs or item priors [2][19] | it is near chance: then the vectors or the pipeline lose the option identity |
| 5a probe, **items** split, acceptable vs bad | above chance | properties of unseen objects are readable from frozen vectors [8][9]; object–affordance links hold on unseen objects [4] | it is at chance. That goes against [4][8][9]; check for a bug before drawing a conclusion |
| 5a probe, **items** split, acceptable vs near miss | lower than vs bad; perhaps only a little above chance | similar negatives cost up to 25 points zero-shot [7]; "related" is encoded but "afforded" is not [6] | it is as high as vs bad |
| 5a probe, **needs** split, acceptable vs bad | modest; can be inflated by "generally useful thing" | the answer-only shortcut [2]; only ~30 training needs, against ~13k label types for GLiNER [25] | — |
| 5a probe, **needs** split, acceptable vs near miss | near chance | new tasks transfer poorly [3][13]; property-to-affordance links are the hardest part [4] | **clearly above chance with a good control: the most informative positive result** |
| 5a unusual-but-workable vs bad | the lowest of the "acceptable" tasks | uncommon affordances fail most [5][29] | it is as easy as the obvious answers |
| 5b new head on frozen vectors, new needs | beats GLiNER's own head on acceptable vs bad; small or no gain on near misses | a missed connection can be fixed without changing the encoder [14], but only if the information is there | it wins clearly on near misses: then the encoder already "knows", and only the head is wrong |
| 5d LoRA fine-tune, seen needs / new things | clear gain, including on near misses | fine-tuning works in distribution [1][5]; hard negatives teach contrast [15][16] | no gain at all: suspect label noise, too little data, or a training bug |
| 5d LoRA fine-tune, new needs | a smaller gain; mostly on bad options, less on near misses | held-out tasks are the weak point [3][13] | a large near-miss gain on new needs |

### 3.2 Controls the literature says to add

1. **Answer-only baseline** [2]. Score each thing with no need: a probe on
   the thing alone, or on a neutral situation. If it is close to the full
   probe on the items or needs split, the probe learned "useful things", not
   fit. `finetune_gliner.py` already lists a "no need gives an even output"
   term as a later step.
2. **A control task that can be memorized** [19]. The shuffled-label control
   stays near 0.5 by construction. Also give each THING one fixed random
   label, then probe on the random split. The gap between the real score and
   this score (selectivity) shows how much of the random-split score is
   memory.
3. **A false-negative check** [18]. Some hand-picked near misses would pass
   with a person: the file (cut the rope), the belt or the scarf (tie up the
   thief), the bench (climb to the high window). The sim may reject them
   because of a property that the option text does not state. If so, no
   text model can learn that rule from text. Before training, have a person
   rate a sample of the near misses.
4. **Near misses from a second source for the test** [17]. If the training
   and test near misses come from one hand-made list, the model can learn
   the style of that list. Make some test near misses in a different way
   (another author, or another rule).
5. **Count needs, not only examples.** New-need generalization is the claim
   that matters. With 30 training needs, a failure on new needs may mean
   "too few needs", not "cannot learn". Plot the new-needs score against
   the number of training needs (for example 10, 20, 30) before you decide.
6. **The format for Qwen.** Base 1–2B models get 75–78% zero-shot on PIQA,
   where the answer is a written procedure [26]. Our options are bare nouns.
   If Qwen is at chance with bare nouns, write each option as the action
   ("cut the rope with the butter knife") and score that. Then the null
   tests the model, not the prompt format. I did not check whether the lab
   has already tried this.

### 3.3 In one line

The literature predicts: probes and fine-tunes work on seen needs and on
new things, near misses are harder, and new needs are the real test.
Near-miss discrimination on NEW needs, with a clean control, would be a new
result. I found no published example of it at this model size.

**Other notes.** Our LoRA rank (r = 8) is far below GLiClass (r = 384 for
the base model) [24]. Their data and goal were very different, so this is
not evidence that r = 8 is too small. If the fine-tune does not move even
the training needs, try a larger rank before you conclude anything.

## 4. References

1. Bisk, Zellers, Le Bras, Gao, Choi. PIQA: Reasoning about Physical Commonsense in Natural Language. AAAI 2020. https://arxiv.org/abs/1911.11641
2. Branco, Branco, Silva, Rodrigues. Shortcutted Commonsense: Data Spuriousness in Deep Learning of Commonsense Reasoning. EMNLP 2021. https://aclanthology.org/2021.emnlp-main.113/
3. Aroca-Ouellette, Paik, Roncone, Kann. PROST: Physical Reasoning about Objects through Space and Time. Findings of ACL 2021. https://arxiv.org/abs/2106.03634
4. Forbes, Holtzman, Choi. Do Neural Language Representations Learn Physical Commonsense? CogSci 2019. https://arxiv.org/abs/1908.02899
5. Adak, Agrawal, Mukherjee, Aditya. Text2Afford: Probing Object Affordance Prediction abilities of Language Models solely from Text. CoNLL 2024. https://arxiv.org/abs/2402.12881
6. Jones, Chang, Coulson, Michaelov, Trott, Bergen. Distributional Semantics Still Can't Account for Affordances. CogSci 2022. https://pages.ucsd.edu/~bkbergen/papers/cogsci_2022_nlm_affordances_final.pdf
7. Misra, Rayz, Ettinger. COMPS: Conceptual Minimal Pair Sentences for testing Robust Property Knowledge and its Inheritance in Pre-trained Language Models. EACL 2023. https://arxiv.org/abs/2210.01963
8. Paik, Aroca-Ouellette, Roncone, Kann. The World of an Octopus: How Reporting Bias Influences a Language Model's Perception of Color. EMNLP 2021. https://arxiv.org/abs/2110.08182
9. Li, Bouraoui, Camacho-Collados, Espinosa-Anke, Gu, Schockaert. Modelling General Properties of Nouns by Selectively Averaging Contextualised Embeddings. IJCAI 2021. https://arxiv.org/abs/2012.07580
10. Zellers, Holtzman, Peters, Mottaghi, Kembhavi, Farhadi, Choi. PIGLeT: Language Grounding Through Neuro-Symbolic Interaction in a 3D World. ACL 2021. https://arxiv.org/abs/2106.00188
11. Tuli, Bansal, Paul, Mausam. TANGO: Commonsense Generalization in Predicting Tool Interactions for Mobile Manipulators. IJCAI 2021. https://arxiv.org/abs/2105.04556
12. Wang, Duan, Fox, Srinivasa. NEWTON: Are Large Language Models Capable of Physical Reasoning? Findings of EMNLP 2023. https://arxiv.org/abs/2310.07018
13. Kejriwal, Shen. Do Fine-tuned Commonsense Language Models Really Generalize? 2020. https://arxiv.org/abs/2011.09159
14. Min, McCoy, Das, Pitler, Linzen. Syntactic Data Augmentation Increases Robustness to Inference Heuristics. ACL 2020. https://arxiv.org/abs/2004.11999
15. Karpukhin et al. Dense Passage Retrieval for Open-Domain Question Answering. EMNLP 2020. https://arxiv.org/abs/2004.04906
16. Yuksekgonul, Bianchi, Kalluri, Jurafsky, Zou. When and Why Vision-Language Models Behave like Bags-of-Words, and What to Do About It? ICLR 2023. https://arxiv.org/abs/2210.01936
17. Hsieh, Zhang, Ma, Kembhavi, Krishna. SugarCrepe: Fixing Hackable Benchmarks for Vision-Language Compositionality. NeurIPS 2023 Datasets and Benchmarks. https://arxiv.org/abs/2306.14610
18. Qu et al. RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering. NAACL 2021. https://arxiv.org/abs/2010.08191
19. Hewitt, Liang. Designing and Interpreting Probes with Control Tasks. EMNLP 2019. https://arxiv.org/abs/1909.03368
20. Ravichander, Belinkov, Hovy. Probing the Probing Paradigm: Does Probing Accuracy Entail Task Relevance? EACL 2021. https://arxiv.org/abs/2005.00719
21. Zhu, Shahtalebi, Rudzicz. Predicting Fine-Tuning Performance with Probing. EMNLP 2022. https://arxiv.org/abs/2210.07352
22. Zaratiana, Pasternak, Boyd, Hurn-Maloney, Lewis. GLiNER2: An Efficient Multi-Task Information Extraction System with Schema-Driven Interface. 2025. https://arxiv.org/abs/2507.18546
23. Fastino. GLiNER2.5-Decide model card (read 2026-09-29). https://huggingface.co/fastino/GLiNER2.5-Decide
24. Stepanov, Shtopko, et al. GLiClass: Generalist Lightweight Model for Sequence Classification Tasks. 2025. https://arxiv.org/abs/2508.07662
25. Zaratiana, Tomeh, Holat, Charnois. GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer. 2023. https://arxiv.org/abs/2311.08526
26. Ben Allal, Lozhkov, Bakouch, et al. SmolLM2: When Smol Goes Big — Data-Centric Training of a Small Language Model. 2025. https://arxiv.org/abs/2502.02737
27. Li, Xu, Dong, Zheng, Liu, Kong, Sun. Can Language Models Understand Physical Concepts? EMNLP 2023. https://arxiv.org/abs/2305.14057
28. Tian et al. MacGyver: Are Large Language Models Creative Problem Solvers? NAACL 2024. https://arxiv.org/abs/2311.09682
29. Qian, Ha, et al. CreativityBench: Evaluating Agent Creative Reasoning via Affordance-Based Tool Repurposing. Preprint, 2026. https://arxiv.org/abs/2605.02910
30. Agrawal et al. Physical Reasoning and Object Planning for Household Embodied Agents (COAT). TMLR 2024. https://arxiv.org/abs/2311.13577

**Not verified or uncertain.** The BERT size in [4] (not stated in the text
I searched). NEWTON's [12] fine-tuning numbers (in a figure only). The
per-pair transfer numbers in [13] (not extracted). The seen/unseen numbers
for T5-11B in [10] come from the paper's text ("77%", "over 18%"). The
table I extracted was garbled and may show slightly different figures. The
"22 pretrained LMs" in [7] was not checked model by model. Some venues I
checked on the PDF or the landing page: [2], [3], [4], [5], [7], [10], [11],
[12], [14], [16], [17], [18], [21], [28], [30]. The others ([1], [8],
[9], [15], [19], [20], [27]) are from search results or memory and are not
checked.
