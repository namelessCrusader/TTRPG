# Character methods: a literature review

Simplified Technical English. 2026-09-28. Workstream B of
`docs/character-models-plan.md`. **The question.** Personality as text in a
model's input is weak. What else exists? Is it useful for us? **Sources.**
We found each source with a web search and checked its title, authors, year
and link. When we could not read a source, we say so. Numbers in brackets
point to Section 13.

## 1. The problem

**Targets.** A peasant is cowardly on average, but sometimes brave. A dragon
is loud, proud and frightening. A knight forces his idea of justice on
others. Characters sometimes do unexpected but valid things (throw a key
away in anger). Many actions can make sense.

| limit | value |
| --- | --- |
| GPU memory at play time | 6 GB: models of about 0.4B to 2B parameters |
| speed rule | No model call for per-tick choices. Only the reply of the character that the player speaks to. |
| training data | almost none |
| large model (Claude) | offline only: data source and judge |
| training | fine-tuning on another machine; no pretraining |
| what the world gives | a legal action menu, and an exact belief record per character |

| term | meaning |
| --- | --- |
| trait, state | A trait is a lasting tendency (bravery). A state is how the character is now. |
| utility score | A number that says how much a character wants one action now. |
| response curve | A function that changes one input (for example, danger) into a score from 0 to 1. |
| weighted random | Choose at random, but give high scores a higher chance. |
| activation, steering vector | Activations are the numbers a model computes inside itself for each word. A steering vector is fixed numbers that we add to them. It pushes the output toward a trait. |
| LoRA | A small add-on that fine-tuning changes. The base model stays the same. |
| distill | Train a small, fast thing (model, tree, table) to copy a slow, large thing. |
| story sifting | Search an event log for small stories that match a pattern. |

## 2. Language-model characters

**Generative Agents** [1]. Twenty-five agents lived in a town like The
Sims. GPT-3.5-turbo ran them. Each agent kept a memory stream: a plain-text
log of what it saw. It wrote reflections (conclusions from memories) and
made day plans. A hundred human raters ranked interview answers. The full
design ranked highest. Each removed part lowered the rank. Reported faults:
failed memory retrieval, invented details, and too formal speech.

**Simulations of 1,000 people** [2]. The team interviewed 1,052 people for
two hours each. Each agent read one interview. On a social survey, agents
reached 85% of a person's own two-week consistency. Agents built from
demographic labels did worse. **Lesson:** a rich history beats a label. Our
belief record and event log are that history.

**Silicon samples** [3]. GPT-3 read thousands of real survey backstories.
Its answers matched each group's spread. This fits group averages, not one
character in one scene.

**Known faults of persona text.**

| fault | evidence |
| --- | --- |
| caricature | GPT-4 simulations of some groups exaggerate the traits that define the persona [4]. |
| flattening | Four models misportray groups and remove variation inside groups [5]. |
| too little variation | ChatGPT personas gave less varied answers than real survey data [6]. |
| weak effect | Persona variables explain under 10% of variation in human labels. Persona prompts help a little [7]. |
| tuning removes variety | Tuning from human preference (RLHF) reduces output diversity [8]. |
| words, not deeds | Persona text changes what a model says about itself. Its effect on behavior is small or inconsistent [9]. |
| state-blind | In a data set of people, 74% of variation was state and 26% trait. Models followed trait and ignored state [11]. |

Prompts can shape a model's personality test scores. Larger
instruction-tuned models do this best [10]. But test scores do not predict
actions [9]. **This agrees with our finding.** Text changes the talk, not
reliably the choice.

**Extensions.** Character-LLM [12] fine-tuned LLaMA-7B on "experiences"
from GPT-3.5-turbo. "Protective experiences" taught confusion about things
outside the character's world, like our belief record. LLM raters scored
interviews. Humanoid Agents [13] added needs, emotion and closeness.

## 3. Changing a model's internal numbers

**Activation addition** [14]. Take the activations for "Love" and for
"Hate". Subtract them. Add the difference during generation. The text
becomes more loving, with no training. The first demo used GPT-2-XL, a 1.5B
model [15]. Later versions of the paper test Llama-3 and OPT.

**Contrastive activation addition (CAA)** [16]. Average the difference over
many answer pairs, with and without a behavior. A coefficient sets strength
and direction. Tested on Llama 2 Chat, with multiple-choice questions and
open-ended text. **Representation engineering** [17] finds directions for
honesty and power-seeking. Tested on LLaMA-2-Chat 7B, 13B and 70B.

**Persona vectors** [18]. A pipeline starts from a trait description in
plain words. It contrasts prompts that bring out the trait with prompts that
suppress it. Tested on Qwen2.5-7B and Llama-3.1-8B, for evil, sycophancy,
hallucination and four more traits. An LLM judge, checked against humans,
scored trait expression. The authors note: only two mid-size models.

**Problems found by others:**
- Steering works on some inputs and fails on others. Some concepts break
  when the prompt changes [19].
- On Gemma-2-2B and 9B, plain prompting beat every steering method.
  Fine-tuning came second [20].
- On Gemma-2-2B and Llama-3-8B (Big Five traits), prompting gave a strong
  match at low cost. LoRA matched best but hurt other skills. Steering was
  in the middle [21].

**Learned prompts.** Prefix-tuning learns about 0.1% extra parameters as
"virtual words". On GPT-2 and BART it beat fine-tuning with little data
[22]. Prompt tuning on T5 matched full tuning only past billions of
parameters [23]. P-Tuning v2 puts learned prompts at every layer. It works
at all tested sizes [24]. LoRA is the common fine-tuning add-on [25].
**Small models:** steering has results at 1.5B [15] and 2B [20, 21]. We
found no checked study of trait steering at 0.4B to 0.6B.

## 4. Game character AI

**Utility AI** [26, 27]. Each input goes through a response curve to a score
from 0 to 1. The scores combine into one score per action. The agent picks
by weighted random from all actions or the top few [26]. Curves remove the
hard jumps of if-then rules. Dave Mark's Infinite Axis Utility System lets a
behavior use any number of scored inputs. It is data-driven. It shipped in
Guild Wars 2: Heart of Thorns [27].

**Needs-based AI (The Sims)** [28]. Needs fall over time. Objects advertise
actions and the needs they fill. The agent scores each advertisement
against its needs. It picks the best, or by weighted random from the top
three. The author calls weighted random "a good compromise": predictable,
but not "unpleasantly deterministic". Per-agent weights and filters give
personality.

**GOAP** [29]. F.E.A.R. used a state machine with three states. A planner
(A* search) chained actions with preconditions and effects toward a goal.

**BDI** [30]. An agent has beliefs, desires and intentions. A blindly
committed agent never drops an intention. A single-minded agent drops it
when it believes the goal is done or impossible. The paper describes OASIS,
an air-traffic system tested at Sydney airport. **For us:** commitment is
emotional momentum.

**OCC and ALMA** [31, 32]. In OCC, emotions come from appraisals. Events
are good or bad for my goals. Acts are praiseworthy or blameworthy. Objects
are appealing or not. In ALMA, Big Five traits set a default mood
(pleasure, arousal, dominance). Emotions push the mood. It slowly returns to
the default. ALMA leaves evaluation to future work.

| game | trait and mood system |
| --- | --- |
| Dwarf Fortress [33] | About 50 facets from 0 to 100, in seven bands. Lowest bravery: "a coward, completely overwhelmed by fear". Values and needs are separate. The wiki names no psychology model. |
| RimWorld [34] | Thoughts set a mood target. Mood moves to it (+12 per hour up, -8 down). Break risk below 35%, 20%, 5%. Traits move the threshold: Iron-willed -18%, Volatile +15%. |
| Crusader Kings III [35] | Hidden AI values: boldness, compassion, greed, honor, rationality, vengefulness, zeal and more. High boldness: less affected by dread, more war on equal or stronger rulers. |

**Lesson.** A break threshold models the brave peasant. Fear controls him
until pressure crosses a line. Then a rare action happens.

## 5. Social simulation

**Comme il Faut and Prom Week** [36, 37]. Weighted rules score how much each
character wants each social exchange. The other character has rules to
accept or reject. Prom Week has about 4,900 rules.

**Versu** [38]. A social practice is a recurring situation, such as a
dinner. It offers actions, but never controls the characters. Each character
picks by utility over its desires. **This is our affordance menu,** with
utility on top.

**Ensemble** [39]. Volition rules raise or lower each character's wish to
act toward another. Trigger rules change the world state.

**Talk of the Town** [40]. Characters hold mental models made of belief
facets. They observe, tell, forget, misremember and lie. **This is our
belief record,** made fallible. Ryan's thesis argues for curation: sift the
simulation for stories [41].

**Rules plus language models.** Paradise extends Ensemble with language
models [42]. We could not read its full text. In Slice of Life [43], the
social engine decides actions and state. An LLM writes dialogue from that
symbolic state. The authors report a first case study only.

## 6. Personality models

The Big Five (OCEAN) are openness, conscientiousness, extraversion,
agreeableness and neuroticism. HEXACO changes these and adds
honesty-humility [44]. **Do traits predict behavior?**

- For one act, weakly. Correlations are usually .20 to .30 [45].
- For life outcomes, well. Traits predict death, divorce and job level. They
  do as well as social class and IQ [48].
- People show nearly all levels of each trait in everyday life. But each
  person's average is very stable. The width of the spread is stable too
  [47]. **A trait is a distribution, not a number.**
- Personality shows as stable if-then patterns [46]. Example: "if teased,
  then angry".

**Which traits do games use?** Concrete traits tied to actions: bravery,
nerves, boldness, greed [33-35]. ALMA uses the Big Five only for mood [32].
LLM studies use the Big Five mainly for questionnaires [10, 21]. **For us:**
keep fantasy traits (bravery, pride, greed, temper, justice). Give each a
mean and a spread. Add if-then modifiers.

## 7. Game masters and stories

**Façade** [49]. A drama manager picks the next story beat from about 200.
Each beat has preconditions and effects. The aim is a tension arc. A later
study asked players how they saw the AI [50].

**PaSSAGE** [51]. It learns weights for five player styles (fighter, power
gamer, tactician, storyteller, method actor) from choices. It picks story
events to fit. A user study found more enjoyment for some player types.

**Directors.** Left 4 Dead's AI Director adapts each playthrough [52].
RimWorld's Cassandra sends threats in on/off cycles. Randy is random [53].

**Story sifting.** Ryan et al. named sifting as an open design problem [54].
Felt is a query language and rules engine for sifting [55]. Winnow finds
stories while they happen [56]. "Select the Unexpected" ranks events by how
unlikely they are [57]. Expressive range analysis plots many outputs on
metric charts to show variety and bias [58]. **For us:** sifting counts
interesting scenes. Unlikeliness measures "unexpected". Expressive range
measures variation per character. None needs a model.

## 8. Distillation and judging

- A 770M T5 learned from a large model's reasons. It beat few-shot 540B
  PaLM on a benchmark, with 80% of the data [59].
- VIPER copies a neural policy into a decision tree that can be checked [60].
- Eureka: a large model wrote reward code. Simulation feedback improved it.
  It beat human experts on 83% of 29 tasks [61].
- A strong LLM judge agreed with humans over 80% of the time. It has
  position, length and self-preference biases [62].

## 9. Comparison

"Mean + spread" asks: can it give "cowardly on average, sometimes brave"?
"Fits laptop" is about memory. The speed rule still limits model calls.

| approach | needs | fits laptop at play | mean + spread | tested on behavior in context? | distill offline |
| --- | --- | --- | --- | --- | --- |
| persona text [7, 9, 10] | a prompt | yes; one call per choice | weak; low variation [6] | mostly surveys, self-reports | yes, as labels |
| generative-agent loop [1, 13] | large model, many calls | no | only through text | yes: town, human rankings [1] | yes, as labels |
| history-grounded agents [2] | long personal history | retrieval yes; calls no | per person, no knob | yes: vs. real people | yes, as labels |
| fine-tune, LoRA [12, 25] | generated data; GPU elsewhere | yes | strong match; costs skills [21] | interviews, LLM raters [12] | yes |
| steering [14-18] | contrast pairs; model internals | yes; 1.5B-2B shown | a real knob: coefficient | multiple choice, judge scores; unreliable [19]; prompting won at 2B [20] | only via data |
| learned prompts [22-24] | labeled set; training | yes | one prefix per level | text tasks, not characters | only via data |
| utility AI [26, 27] | authored curves | yes, very fast | strong: curves set mean, weighted random sets spread | shipped games; no formal study found | is a table; large model can write curves |
| needs-based [28] | needs, advertising objects | yes | per-agent weights; top-3 random | shipped game (The Sims) | is a table |
| GOAP [29] | actions with preconditions, effects | yes | weak | shipped game | plans can be cached |
| BDI commitment [30] | beliefs, goals, plans | yes | sets stubbornness | industrial systems | yes |
| OCC + ALMA mood [31, 32] | appraisal rules | yes | strong: mood returns to trait default | not evaluated [32] | yes |
| trait/mood games [33-35] | trait numbers, thresholds | yes | strong: rare breaks | shipped games | is a table |
| social rule engines [36-39] | many authored rules | yes | rule weights | shipped games | is a table |
| belief models [40] | belief facets | yes | not a choice method | design paper | not needed |
| drama manager, player model [49-53] | beats or events | yes | pacing only | user studies [50, 51] | yes |
| story sifting [55-58] | patterns | yes, on the log | measures only | tools, demos | yes |

## 10. What this means for us

Test these three in the lab world, in this order. Baselines in all tests:
random valid choice, and persona text.

**Test 1: trait distributions + utility scores (no training, no model).**

1. Each trait has a mean and a spread [47]. Peasant bravery: low mean.
2. At each scene, draw a state. If-then modifiers change it [46].
   Example: "child in danger: bravery up".
3. Appraisal makes emotions. Emotions push mood. Mood decays [31, 32].
4. Response curves score each menu option [26, 27].
5. Pick by weighted random from the top few [28].
6. A break threshold opens rare actions, such as throwing the key [34].
7. A commitment rule keeps the action until a stop condition [30].

**Why first.** It needs zero model calls per tick, so it obeys the speed
rule. It has the "mean + spread" knob by design. It sets the bar for every
model. A large model writes the curves offline; the sim and judge check
them, as in Eureka [61].

**Test 2: large-model labels, distilled into weights or a tree.**

1. Take many belief-record snapshots from lab runs.
2. The large model gives several good choices per character, with weights.
3. Fit the Test 1 weights to these labels, or fit a small tree [60].
4. Compare with hand-written weights on new scenes.

**Why.** It turns the large model's sense of context into a fast table. No
model runs at play time. Watch the label spread: large models flatten
variation [5, 6].

**Test 3: persona text vs. steering vs. LoRA (addressed character only).**

1. One model of 0.5B to 1.5B for all arms. Arm A: persona text.
2. Arm B: a CAA vector per trait [16]. Draw its coefficient from the trait.
3. Arm C: a LoRA trained on large-model data, on the other machine [25].
4. Score actions and lines. Never score self-reports [9].

**Why.** It answers the team's question directly. Expect little: prompting
beat steering at 2B [20], and steering is unreliable [19]. Run it last.

## 11. How the approaches combine

```
world (voxel sim)       -> legal action menu        (Versu-style offers [38])
belief record           -> what the character knows (Talk of the Town [40])
appraisal (OCC)         -> emotion -> mood, decays  (ALMA [32])
trait mean + spread     -> this scene's state       (Fleeson [47])
if-then modifiers       -> adjusted weights         (Mischel & Shoda [46])
utility curves          -> score per option         (IAUS [27])
weighted random, break  -> the action; rare acts    (Sims [28], RimWorld [34])
commitment rule         -> momentum                 (BDI [30])
small model (addressed) -> words for the chosen act (Slice of Life [43])
sifting + judge         -> "interesting", variety   ([56], [57], [58])
```

Game AI gives structure and variation. The belief record gives context.
Models work offline: they write curves, label scenes and judge. At play, a
small model only words the addressed character's act. Steering can color
that line with the mood. Sifting and expressive range measure the result.
The dragon has high pride weights on "roar" and "boast". The knight's rule
reads his beliefs: "if I believe X wronged Y, then punish X".

## 12. Open risks

- **Curve tuning is work.** Many curves interact. Large-model curves can be
  wrong in ways that look right.
- **Features limit context.** A score sees only the inputs we give it. A
  missing belief input means a missing reason.
- **Label flattening.** Large-model labels may lack variation [5, 6, 8].
  Draw trait states before asking for labels. Check the spread.
- **Judge bias.** LLM judges have position and length biases [62]. They may
  reward caricature [4]. Your ratings must calibrate the judge.
- **Self-report trap.** A character that says "I am brave" proves nothing
  [9]. Measure actions only.
- **Small-model steering is unknown** below 1.5B. It may fail on new prompts
  [19] or hurt fluency.
- **Per-act noise.** Traits predict single acts weakly [45]. Use many seeds.
  **Lab is not voxel:** check tuned curves again after the move.
- **Not verified:** the full text of Paradise [42]; how CK3 sets AI values
  from traits [35].

## 13. References

1. Park, O'Brien, Cai, Morris, Liang, Bernstein (2023). Generative Agents: Interactive Simulacra of Human Behavior. UIST 2023. https://arxiv.org/abs/2304.03442
2. Park, Zou, Kamphorst, et al. (2024). Generative Agent Simulations of 1,000 People. arXiv. https://arxiv.org/abs/2411.10109
3. Argyle, Busby, Fulda, Gubler, Rytting, Wingate (2023). Out of One, Many: Using Language Models to Simulate Human Samples. Political Analysis 31. https://www.cambridge.org/core/journals/political-analysis/article/abs/out-of-one-many-using-language-models-to-simulate-human-samples/035D7C8A55B237942FB6DBAD7CAA4E49
4. Cheng, Piccardi, Yang (2023). CoMPosT: Characterizing and Evaluating Caricature in LLM Simulations. EMNLP 2023. https://aclanthology.org/2023.emnlp-main.669/
5. Wang, Morgenstern, Dickerson (2025). Large language models that replace human participants can harmfully misportray and flatten identity groups. Nature Machine Intelligence 7. https://www.nature.com/articles/s42256-025-00986-z
6. Bisbee, Clinton, Dorff, Kenkel, Larson (2024). Synthetic Replacements for Human Survey Data? The Perils of Large Language Models. Political Analysis 32. https://doi.org/10.1017/pan.2024.5
7. Hu, Collier (2024). Quantifying the Persona Effect in LLM Simulations. ACL 2024. https://aclanthology.org/2024.acl-long.554/
8. Kirk et al. (2024). Understanding the Effects of RLHF on LLM Generalisation and Diversity. ICLR 2024. https://arxiv.org/abs/2310.06452
9. Han, Kocielnik, Song, Debnath, Mobbs, Anandkumar, Alvarez (2025). The Personality Illusion: Revealing Dissociation Between Self-Reports & Behavior in LLMs. arXiv. https://arxiv.org/abs/2509.03730
10. Serapio-García, Safdari, et al. (2025). A psychometric framework for evaluating and shaping personality traits in large language models. Nature Machine Intelligence. https://www.nature.com/articles/s42256-025-01115-6
11. Harry, Ngong, Nweke, Feng, Near (2026). Beyond Fixed Psychological Personas: State Beats Trait, but Language Models are State-Blind. arXiv. https://arxiv.org/abs/2601.15395
12. Shao, Li, Dai, Qiu (2023). Character-LLM: A Trainable Agent for Role-Playing. EMNLP 2023. https://aclanthology.org/2023.emnlp-main.814/
13. Wang, Chiu, Chiu (2023). Humanoid Agents: Platform for Simulating Human-like Generative Agents. EMNLP 2023 Demos. https://aclanthology.org/2023.emnlp-demo.15/
14. Turner, Thiergart, Leech, Udell, Vazquez, Mini, MacDiarmid (2023). Activation Addition: Steering Language Models Without Optimization (later title: Steering Language Models With Activation Engineering). arXiv. https://arxiv.org/abs/2308.10248
15. Turner et al. (2023). Steering GPT-2-XL by adding an activation vector. LessWrong. https://www.lesswrong.com/posts/5spBue2z2tw4JuDCx/steering-gpt-2-xl-by-adding-an-activation-vector
16. Rimsky, Gabrieli, Schulz, Tong, Hubinger, Turner (2024). Steering Llama 2 via Contrastive Activation Addition. ACL 2024. https://aclanthology.org/2024.acl-long.828/
17. Zou et al. (2023). Representation Engineering: A Top-Down Approach to AI Transparency. arXiv. https://arxiv.org/abs/2310.01405
18. Chen, Arditi, Sleight, Evans, Lindsey (2025). Persona Vectors: Monitoring and Controlling Character Traits in Language Models. arXiv (Anthropic). https://arxiv.org/abs/2507.21509
19. Tan et al. (2024). Analysing the Generalisation and Reliability of Steering Vectors. NeurIPS 2024. https://arxiv.org/abs/2407.12404
20. Wu, Arora, Geiger, Wang, Huang, Jurafsky, Manning, Potts (2025). AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders. ICML 2025. https://arxiv.org/abs/2501.17148
21. Handa, Wu, Koshiyama, Treleaven (2025). Personality as a Probe for LLM Evaluation: Method Trade-offs and Downstream Effects. NeurIPS 2025 workshop. https://arxiv.org/abs/2509.04794
22. Li, Liang (2021). Prefix-Tuning: Optimizing Continuous Prompts for Generation. ACL 2021. https://aclanthology.org/2021.acl-long.353/
23. Lester, Al-Rfou, Constant (2021). The Power of Scale for Parameter-Efficient Prompt Tuning. EMNLP 2021. https://aclanthology.org/2021.emnlp-main.243/
24. Liu et al. (2022). P-Tuning: Prompt Tuning Can Be Comparable to Fine-tuning Across Scales and Tasks. ACL 2022. https://aclanthology.org/2022.acl-short.8/
25. Hu et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022. https://arxiv.org/abs/2106.09685
26. Mark, Dill (2010). Improving AI Decision Modeling Through Utility Theory. GDC 2010 AI Summit. https://www.gdcvault.com/play/1012410/Improving-AI-Decision-Modeling-Through
27. Mark (2013). Architecture Tricks: Managing Behaviors in Time, Space, and Depth. GDC 2013. https://www.gdcvault.com/play/1018040/Architecture-Tricks-Managing-Behaviors-in ; IAUS: https://www.gameai.com/iaus.php
28. Zubek (2010). Needs-Based AI. Game Programming Gems 8. Draft: https://robert.zubek.net/publications/Needs-based-AI-draft.pdf
29. Orkin (2006). Three States and a Plan: The A.I. of F.E.A.R. GDC 2006. https://www.gamedevs.org/uploads/three-states-plan-ai-of-fear.pdf
30. Rao, Georgeff (1995). BDI Agents: From Theory to Practice. ICMAS-95. https://cdn.aaai.org/ICMAS/1995/ICMAS95-042.pdf
31. Ortony, Clore, Collins (1988). The Cognitive Structure of Emotions. Cambridge University Press. https://archive.org/details/isbn_9780521386647
32. Gebhard (2005). ALMA: A Layered Model of Affect. AAMAS 2005. https://dl.acm.org/doi/10.1145/1082473.1082478
33. Dwarf Fortress Wiki. Personality facet (DF2014). https://dwarffortresswiki.org/index.php/DF2014:Personality_facet
34. RimWorld Wiki. Mood; Traits. https://rimworldwiki.com/wiki/Mood ; https://rimworldwiki.com/wiki/Traits
35. CK3 Wiki. Characters (AI personality). https://ck3.paradoxwikis.com/Characters
36. McCoy, Treanor, Samuel, Wardrip-Fruin, Mateas (2011). Comme il Faut: A System for Authoring Playable Social Models. AIIDE 2011. https://ojs.aaai.org/index.php/AIIDE/article/view/12454
37. McCoy, Treanor, Samuel, Reed, Mateas, Wardrip-Fruin (2013). Prom Week: Designing past the game/story dilemma. FDG 2013. http://www.fdg2013.org/program/papers/paper13_mccoy_etal.pdf
38. Evans, Short (2014). Versu—A Simulationist Storytelling System. IEEE TCIAIG 6(2). https://ieeexplore.ieee.org/document/6648395/
39. Samuel, Reed, Maddaloni, Mateas, Wardrip-Fruin (2015). The Ensemble Engine: Next-Generation Social Physics. FDG 2015. http://www.ben-samuel.com/wp-content/uploads/2015/09/FDG2015-The-Ensemble-Engine-Next-Generation-Social-Physics.pdf
40. Ryan, Summerville, Mateas, Wardrip-Fruin (2015). Toward Characters Who Observe, Tell, Misremember, and Lie. EXAG workshop, AIIDE 2015. https://eis.ucsc.edu/papers/ryanEtAl_TowardCharactersWhoObserveTellMisrememberLie.pdf
41. Ryan (2018). Curating Simulated Storyworlds. PhD thesis, UC Santa Cruz. https://escholarship.org/uc/item/1340j5h2
42. Kelly, Mateas, Wardrip-Fruin (2024). Paradise: An Experiment Extending the Ensemble Social Physics Engine with Language Models. FDG 2024. https://doi.org/10.1145/3649921.3659841
43. Treanor, Samuel, Nelson (2025). Slice of Life: A Social Physics Game with Interactive Conversations using Symbolically Grounded LLM-Based Generative Dialogue. FDG 2025. https://dl.acm.org/doi/10.1145/3723498.3723806
44. Ashton, Lee (2007). Empirical, Theoretical, and Practical Advantages of the HEXACO Model of Personality Structure. PSPR 11. https://journals.sagepub.com/doi/10.1177/1088868306294907
45. Mischel (1968). Personality and Assessment. Wiley. https://archive.org/details/personalityasses0000misc
46. Mischel, Shoda (1995). A cognitive-affective system theory of personality. Psychological Review 102(2). https://pubmed.ncbi.nlm.nih.gov/7740090/
47. Fleeson (2001). Toward a structure- and process-integrated view of personality: Traits as density distributions of states. JPSP 80(6). https://doi.org/10.1037/0022-3514.80.6.1011
48. Roberts, Kuncel, Shiner, Caspi, Goldberg (2007). The Power of Personality. Perspectives on Psychological Science 2(4). https://journals.sagepub.com/doi/abs/10.1111/j.1745-6916.2007.00047.x
49. Mateas, Stern (2003). Façade: An Experiment in Building a Fully-Realized Interactive Drama. GDC 2003. https://users.soe.ucsc.edu/~michaelm/publications/mateas-gdc2003.pdf
50. Mehta, Dow, Mateas, MacIntyre (2007). Evaluating a Conversation-centered Interactive Drama. AAMAS 2007. https://users.soe.ucsc.edu/~michaelm/publications/mehta-aamas2007.pdf
51. Thue, Bulitko, Spetch, Wasylishen (2007). Interactive Storytelling: A Player Modelling Approach. AIIDE 2007. https://ojs.aaai.org/index.php/AIIDE/article/view/18780
52. Booth (2009). The AI Systems of Left 4 Dead. AIIDE 2009 keynote slides. https://steamcdn-a.akamaihd.net/apps/valve/2009/ai_systems_of_l4d_mike_booth.pdf
53. RimWorld Wiki. AI Storytellers. https://rimworldwiki.com/wiki/AI_Storytellers
54. Ryan, Mateas, Wardrip-Fruin (2015). Open Design Challenges for Interactive Emergent Narrative. ICIDS 2015. https://link.springer.com/chapter/10.1007/978-3-319-27036-4_2
55. Kreminski, Dickinson, Wardrip-Fruin (2019). Felt: A Simple Story Sifter. ICIDS 2019. https://link.springer.com/chapter/10.1007/978-3-030-33894-7_27
56. Kreminski, Dickinson, Mateas (2021). Winnow: A Domain-Specific Language for Incremental Story Sifting. AIIDE 2021. https://ojs.aaai.org/index.php/AIIDE/article/view/18903
57. Kreminski, Dickinson, Wardrip-Fruin, Mateas (2022). Select the Unexpected: A Statistical Heuristic for Story Sifting. ICIDS 2022. https://link.springer.com/chapter/10.1007/978-3-031-22298-6_18
58. Smith, Whitehead (2010). Analyzing the Expressive Range of a Level Generator. PCG Workshop 2010. https://dl.acm.org/doi/10.1145/1814256.1814260
59. Hsieh et al. (2023). Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes. Findings of ACL 2023. https://aclanthology.org/2023.findings-acl.507/
60. Bastani, Pu, Solar-Lezama (2018). Verifiable Reinforcement Learning via Policy Extraction. NeurIPS 2018. https://proceedings.neurips.cc/paper/2018/hash/e6d8545daa42d5ced125a4bf747b3688-Abstract.html
61. Ma et al. (2024). Eureka: Human-Level Reward Design via Coding Large Language Models. ICLR 2024. https://arxiv.org/abs/2310.12931
62. Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. NeurIPS 2023. https://arxiv.org/abs/2306.05685
