# Question primitives: a review

Simplified Technical English. 2026-09-30. Mind lab. **The question.** A
person who acts in a scene asks himself many small questions. What kinds of
question are there? Which kinds can our small classifier answer well? How do
we chain them without a wrong early answer that spoils the choice?
**Sources.** We opened each source in Section (e). When we could read only an
abstract, a record or a search summary, we say so. Numbers in brackets point
to Section (e). Lab numbers come from `docs/mind-lab-todo.md`.

## Main points

1. Fourteen kinds of question cover the sources well (Section a). Two things
   sort them: whose mind the question is about (the world, Tom, another
   person), and the time (before, now, after).
2. The simulation (sim) knows the right answer for seven kinds, and partly
   for six. Only one kind has no sim answer at all: Tom's own feeling and
   tendency (G10). **Rule: do not ask the model a question that the sim can
   answer.** Use those questions only to measure the model.
3. The model's weak points in the literature match our lab data: roles (who
   did what to whom), other people's minds, negation, numbers, time order,
   pronouns, and "the model picks the words that the scene names" (Section
   b.2). The relevance check (yes/no, one sentence) is in the strong group.
4. When no single answer is right, measure consistency (same answer for the
   same meaning), sensitivity (a different answer when a cause changes), and
   the value for the final choice (Section c).
5. Errors add up along a chain. Keep more than one answer, pass
   probabilities as soft weights, and use an answer only when its confidence
   is above a gate that the costs set (Section d).

## Terms

| term | meaning |
| --- | --- |
| zero-shot | The model answers with labels that it did not see in training. |
| encoder, cross-encoder | An encoder reads the whole input at once and gives numbers, not new text. A cross-encoder reads the text and the label together. Our chooser is this kind of model. |
| NLI (entailment) | Natural language inference. Does sentence A make sentence B true? Many zero-shot classifiers use it: "the text entails 'this is about sport'". |
| label | One answer phrase in the list (for example, "stopping someone"). |
| sim | Our game simulation. It holds the true state, the event log, and a belief record per character. |
| known answer | The sim can compute the right answer. |
| unknown answer | No single right answer exists (feelings, dispositions). Several answers can be good. |
| appraisal | How a person judges an event for himself: is it new, good or bad, who caused it, can I cope. |
| theory of mind (ToM) | Thinking about what another person sees, knows, believes, or wants. |
| first-order, second-order belief | "Mara thinks X" is first order. "Bren thinks that Mara thinks X" is second order. |
| false belief | A person believes something that is not true now (he did not see the change). |
| affordance | What a thing lets you do (a rope lets you tie, a bucket lets you carry water). |
| precondition | Something that must be true before an action can happen. |
| paraphrase | The same meaning in other words. |
| contrast set, minimal pair | Two inputs that differ in one small, meaningful way. |
| counterfactual | "What if one fact were different?" |
| consistency | The model gives answers that agree with each other. |
| calibration | A model is calibrated when "70% sure" is right 70% of the time. |
| confidence gate | Use the answer only when the model's confidence is above a threshold. Else, do not use it (abstain). |
| hard filter, soft weight | A hard filter deletes options. A soft weight changes their scores and can be outweighed. |
| top-k | Keep the k best answers, not only the best one. |
| lure | A word in the scene that pulls the model to a label that shares the word. |
| P@k | Precision at k. The share of the top k outputs that judges accept. |
| oracle | The right answer, given by hand or by the sim, for comparison. |

## Lab numbers and where they fall

| lab question | result | group |
| --- | --- | --- |
| "Is this sentence about what Kael must do?" (yes/no, per sentence) | 97% | G1 relevance |
| "What does this situation call for?" (9 phrases) | 60% | G11 purpose |
| "What is the bandit in this scene?" (7 roles) | 22% | G3 roles |
| "What does Kael need for this?" (38 capability phrases) | 60% | G12 means |
| "How does Tom feel?" / "What does Tom want to do?" after "Tom is hot-tempered" | 45 of 45 trait-consistent | G10 own feeling |
| "What is still missing?" (4 phrases, two kinds of state) | 33% and 83% | G14 preconditions |

---

## (a) A grouping of question kinds

### a.1 The scene for the examples

The village square at noon. A bandit holds a knife near Mara, the baker.
Bren, the guard, stands at the gate. Tom, the blacksmith, is hot-tempered.
Tom holds a hammer. A rope and a bucket lie by the well.

### a.2 Two axes

The sources sort questions along two axes. ATOMIC splits its questions by
person (PersonX or others) and by time (before or after the event) [10].
GLUCOSE splits its ten dimensions into five before the event and five after
it [12]. Appraisal theory adds a third axis: the order of the checks
(relevance first, then implications, then coping, then norms) [20, 21].

| axis | values |
| --- | --- |
| whose mind | the world (facts), Tom (self), another person |
| time | before (cause, precondition), now (state, role), after (consequence) |
| answer format | YN = yes or no. PICK = one of up to 8 phrases. POINT = one person or thing from the scene. |
| answer source | K = known (sim computes it). P = partly known. U = unknown. |

### a.3 Summary

| # | group | typical question | format | answer source | expected reliability (our prediction from the sources) |
| --- | --- | --- | --- | --- | --- |
| G1 | Relevance and attention | "Does this matter for Tom now?" | YN | K (mostly) | high (lab 97%) |
| G2 | Scene facts and time | "Who holds the knife?" | POINT, YN | K | high for plain lookups; low with negation, numbers, time order, pronouns |
| G3 | Roles: who did what to whom | "Who is in danger?" | POINT, YN | K | low (lab 22%) |
| G4 | Cause, agency, intent | "Did the bandit mean to hurt Mara?" | POINT, YN | P | medium for one agent's motive; low for blame among several people |
| G5 | Consequence and prediction | "What will happen if Tom does nothing?" | PICK, YN | P (physics K, people U) | medium for the main agent; low for effects on others |
| G6 | Value for Tom's goals, urgency | "Is this good or bad for Tom?" | YN, PICK of 3 | P | not measured; likely medium |
| G7 | Norms and judgment | "Would the village approve if Tom helps?" | YN, POINT | P (law K, morals U) | not measured |
| G8 | What others know and believe | "Does Bren know about the knife?" | YN, POINT | K | low |
| G9 | What others want and feel | "What does the bandit want?" | PICK | P (drives K, feelings U) | low to medium |
| G10 | Tom's own feeling and tendency | "How does Tom feel?" | PICK | U | high consistency; the scene effect is not yet tested |
| G11 | Purpose: what the scene calls for | "What does this call for?" | PICK, YN | P | medium (lab 60%) |
| G12 | Means: what things can do | "Which thing here can cut rope?" | POINT, YN | K | medium for common uses; low for unusual uses and long lists (lab 60% over 38) |
| G13 | Feasibility and coping | "Can Tom stop this alone?" | YN | K | not measured; physical reasoning is weak in the sources |
| G14 | Preconditions, obstacles, progress | "What is still missing?" | PICK, POINT, YN | K | low to medium (lab 33%, 83%) |

The "expected reliability" column is our reading of the sources and lab data.
It is a prediction to test, not a measurement.

### a.4 The groups

#### G1. Relevance and attention

- **Asks:** Does this fact matter to Tom now? Is it new to him?
- **Examples:**
  - "Is this sentence about Tom?"
  - "Does this sentence matter for what Tom must do now?"
  - "Is this new to Tom?"
  - "Is this sentence about a danger?"
- **Format:** YN, one sentence at a time.
- **Answer source:** Mostly known. The sim knows which facts touch Tom's
  drives, goals and place. "Is it new" comes from Tom's belief record.
- **Why it is a kind:** Scherer puts the relevance check (novelty and goal
  relevance) first in the sequence of appraisal checks [20, 21]. ATOMIC asks
  a relevance gate before a content question ("Does this event affect people
  other than PersonX?", then "What do they likely want to do next?") [10].
- **Evidence:** Lab 97%. Graesser's verification questions have short answers
  ("yes", "no", "maybe") [1].
- **Use:** A first filter. Pass its probability on as a weight. Do not delete
  sentences with it (Section d).

#### G2. Scene facts and time

- **Asks:** What is where? Who has what? Is it so? How many? What came first?
- **Examples:**
  - "Where is the bandit?"
  - "Who holds the knife?"
  - "Is the smithy door open?"
  - "Is there more than one guard in the square?"
  - "What happened first: the shout or the crash?"
- **Format:** POINT or YN. Do not use numbers as labels. Ask a yes/no
  threshold ("more than one?") instead.
- **Answer source:** Known (sim state and event log).
- **Why it is a kind:** The 5W1H questions (who, what, when, where, why, how)
  [4]. The Li and Roth answer types HUMAN, LOCATION, ENTITY, NUMERIC [7].
  Graesser's concept completion, feature specification and quantification
  [1, 2]. bAbI tasks 1 to 14 (where, yes/no, counting, lists, negation,
  "maybe", pronouns, time) [8]. MC-TACO's five time kinds: duration, order,
  typical time, frequency, stationarity [72].
- **Evidence:** Plain fact questions are easy. GPT-3 answers ToMi fact
  questions 90% to 100% right, and belief questions at most 60% [34]. But
  small wording changes
  break reading models. CheckList, on a BERT reading model: 100% failure when
  the question has a negation, 82.9% on before/after, 100% on simple
  pronouns, 67.3% on comparisons with an antonym [55]. BERT's word pieces
  code numbers less precisely than other embeddings [71].
- **Use:** The sim answers. Use this group as a test bed: the sim grades the
  model on many scenes at no cost.

#### G3. Roles: who did what to whom

- **Asks:** Which person has which part in an event?
- **Examples:**
  - "Who is threatening Mara?"
  - "Who is in danger?"
  - "Who is only watching?"
  - "Who struck first?"
  - "Is the bandit a danger to others?" (YN form, one person per question)
- **Format:** POINT (a person), or YN per person and role.
- **Answer source:** Known. The event log holds the actor and the target of
  each act.
- **Why it is a kind:** QA-SRL writes each role as a question ("Who finished
  something?", "What did someone finish?") [9]. bAbI task 5 asks "Who gave
  the cake to Fred?" [8]. ATOMIC and GLUCOSE keep the agent apart from the
  others [10, 12].
- **Evidence:** This is a known weak point. CheckList: "Richard bothers
  Elizabeth. Who is bothered?" fails 60.8% of the time; with three people,
  95.7% [55]. BERT struggles with role-based event prediction [68]. NLI
  models give the same label when the word order is shuffled [70]. SocialIQA
  errors pick "answers pertaining to the wrong participants" [13]. GPT-3
  "often confuses which participant is being asked about" [34].
- **Lab:** 22% over 7 abstract roles.
- **Use:** The sim answers. If the model must answer, name the person in the
  question and ask YN per person. Abstract role labels ("a danger to
  others") share few words with the scene, so overlap cannot help (b.2).

#### G4. Cause, agency and intent (looking back)

- **Asks:** Why did this happen? Who caused it? Was it on purpose?
- **Examples:**
  - "Why is Mara shouting?"
  - "Who broke the window?"
  - "Did the bandit mean to hurt Mara?"
  - "Was it an accident?"
- **Format:** POINT (an event or a person); YN for intent.
- **Answer source:** Partly. The sim knows the event that caused the state
  and the actor. It knows the goals of non-player characters. It does not
  know the player's intent.
- **Why it is a kind:** Graesser's causal antecedent (a deep-reasoning
  question) [1]. GLUCOSE dimensions 1 to 5 (the event, the emotion or drive,
  the location, the possession, and other attributes that cause or enable X)
  [12]. ATOMIC xIntent and xNeed [10]; ATOMIC-2020 xReason [11]; ConceptNet
  Causes and MotivatedByGoal [15]. The agency check in Scherer (is it the
  self, another person, or chance?) and accountability in Lazarus [20, 21].
  Socratic questions that probe reasons and evidence [6]. bAbI task 20: "Why
  did John go to the kitchen? hungry" [8].
- **Evidence:** On abductive inference ("what is the best explanation?") the
  best model scored 68.9% and humans 91.4% [81]. In SocialIQA, questions on
  motivations and needs before the event were easier for BERT-large [13].
  Agency plus a bad event gives a tendency to aggress; control, more than
  agency, drives the tendency to repair [21].
- **Use:** The sim supplies the cause and the actor. The model may add the
  motive of one named agent.

#### G5. Consequence and prediction (looking forward)

- **Asks:** What happens next? What happens if Tom does X?
- **Examples:**
  - "What will happen if Tom does nothing?"
  - "What will the bandit do next?"
  - "Will the rope hold Tom's weight?"
  - "If Tom shouts, what will Bren do?"
  - "Will the fire spread to the smithy?"
- **Format:** PICK (up to 8 outcomes) or YN ("Will X happen?").
- **Answer source:** Physical outcomes are known: the sim can run forward.
  Reactions of people are unknown.
- **Why it is a kind:** Graesser's causal consequence and expectational
  questions [1]. ATOMIC xEffect, oEffect, xWant, oWant [10]. GLUCOSE
  dimensions 6 to 10 [12]. Endsley's third level of situation awareness,
  projection (search summary only, see [27]). Socratic questions on
  implications and consequences [6].
- **Evidence:** In ATOMIC, human judges accepted generated effects on others
  (oEffect) only 23% to 26% of the time, and reactions of PersonX (xReact)
  64% to 72% [10]. PIQA (physical outcomes): humans 95%, large models 77%
  [16]. SocialIQA gives a mixed picture for effects: the figure says effects
  are easier, the text says involuntary effects are harder [13].
- **Use:** The sim answers physical questions. The model may predict the main
  agent's next act; treat its answers about others as weak.

#### G6. Value for Tom's goals, and urgency

- **Asks:** Is this good or bad for Tom? How much? Must he act now?
- **Examples:**
  - "Is this good or bad for Tom?"
  - "Does this put Tom's smithy at risk?"
  - "Does this help what Tom wants?"
  - "Must Tom act now, or can he wait?"
- **Format:** YN, or PICK of 3 (good, bad, neither).
- **Answer source:** Partly. If drives and goals are sim state (hunger,
  safety, property), the sim can compute "good or bad for the goal". How
  much Tom cares is personality, so it is unknown.
- **Why it is a kind:** Scherer's intrinsic pleasantness, goal conduciveness
  and urgency checks [20, 21]. Lazarus's motivational congruence [20].
  Scherer and Moors define valence as intrinsic pleasantness plus goal
  congruence [21]. OCC rates events by desirability and objects by
  appealingness [22]. ConceptNet Desires and NotDesires [15].
- **Evidence:** We found no study that tests encoder classifiers on appraisal
  labels. Yin et al. include emotion and situation labels in zero-shot
  benchmarks [76]. Scherer and Moors report evidence that novelty is judged
  before intrinsic valence [21].
- **Use:** A good early step in an appraisal chain (Section d.4). The sim can
  give the "for which goal" part.

#### G7. Norms and judgment

- **Asks:** Is this right? Is it allowed? Would others approve? Who is to
  blame?
- **Examples:**
  - "Is it right for Tom to hit the bandit?"
  - "Is it allowed to carry a blade in the square?"
  - "Would the village approve if Tom helps Mara?"
  - "Who is to blame for the fight?"
  - "Would Tom be ashamed to run away?"
- **Format:** YN; POINT for blame.
- **Answer source:** Partly. Village law can be sim data (known). A personal
  moral view changes from character to character (unknown).
- **Why it is a kind:** Scherer's normative significance check (own
  standards and social norms) and fairness [20, 21]. OCC rates actions of
  agents by praiseworthiness [22]. Graesser's judgmental questions [1, 2].
  Socratic questions about viewpoints ("How would other groups respond?")
  [6]. Social Chemistry 101 gives 292k rules of thumb with 12 dimensions,
  for example social judgment, cultural pressure, legality and moral
  foundations [79].
- **Evidence:** Not measured for encoders in our sources.
- **Use:** The sim gives the law. The model gives the character's own view,
  evaluated with Section c.

#### G8. What others know and believe

- **Asks:** What does another person see, know or believe?
- **Examples:**
  - "Does Bren know the bandit has a knife?"
  - "Did the bandit see Tom?"
  - "Where does Mara think her purse is?"
  - "Who knows about the stolen bread?"
  - "Does the bandit think that Tom will fight?" (second order)
- **Format:** YN; POINT (a place or a person).
- **Answer source:** Known. The sim keeps a belief record per character.
- **Why it is a kind:** The Sally-Anne false-belief test and its text
  versions: ToMi [28, 34], BigToM [29], Hi-ToM [30], OpenToM [31], FANToM
  [32].
- **Evidence:** This is the clearest case of "the sim knows, the model does
  not":
  - ToMi can be solved by dataset biases; the authors built a controlled
    version [28]. On it, GPT-3's best was 60% on mental-state questions,
    against 90% to 100% on fact questions [34].
  - BigToM: forward belief (percept to belief) is easy for strong models.
    Backward belief (infer the belief from an action) is the hardest; most
    models are "far below chance" and anchor on a belief that the text
    states [29].
  - Hi-ToM: from order 0 (reality) to order 4, joint accuracy goes "from near
    perfect to near zero". Models also prefer answers at the start or the end
    of the story [30].
  - FANToM: models do better on belief multiple-choice questions than on the
    "who knows this?" list questions that the belief depends on. The authors
    call this illusory theory of mind [32].
  - Small changes that keep the logic of the task can reverse the results
    [33].
- **Use:** Never ask the model. Read the belief record.

#### G9. What others want, feel and will do

- **Asks:** The wants, feelings and attitude of another person.
- **Examples:**
  - "What does the bandit want?"
  - "How does Mara feel?"
  - "Will Bren help if Tom asks?"
  - "How will Mara feel about Tom if he helps?"
  - "Is the bandit afraid?"
- **Format:** PICK (up to 8).
- **Answer source:** Partly. The drives and goals of non-player characters
  are sim state. Their feelings and attitudes are unknown, unless the sim
  models them.
- **Why it is a kind:** ATOMIC oReact, oWant, oEffect [10]. SocialIQA
  question types (wants, reactions, descriptions, motivations, needs,
  effects) [13]. OCC's "fortunes of others" emotions [22]. OpenToM attitude
  questions [31].
- **Evidence:** GPT-3 "performs consistently better on agent-centric
  questions, compared to other-oriented questions" [34]. ATOMIC effects on
  others score lower than the agent's own reaction [10]. OpenToM: attitude
  questions are among the hardest; chain-of-thought helps them only a little
  [31].
- **Use:** The sim gives drives and goals. Name the other person in the
  question and in each label.

#### G10. Tom's own feeling, disposition and action tendency

- **Asks:** How does Tom feel? What does he want to do? What does this say
  about him?
- **Examples:**
  - "How does Tom feel?"
  - "What does Tom want to do?"
  - "Does Tom want to move toward the bandit or away from him?"
  - "What does this say about Tom?"
  - "How strongly does Tom feel this?"
- **Format:** PICK (up to 8 feelings, or action tendencies such as approach,
  avoid, attack, help, freeze).
- **Answer source:** Unknown. This is the personality answer. A trait gives a
  spread of likely answers, not one answer.
- **Why it is a kind:** ATOMIC xReact, xWant, xAttr [10]. Frijda: an emotion
  is a readiness for a kind of action (an action tendency) [24]. Scherer and
  Moors: appraisal results lead to action tendencies [21]. Fleeson: a trait
  is a distribution of states; "the typical individual regularly and
  routinely manifested nearly all levels of all traits" [82].
- **Evidence:** In ATOMIC, the agent's own reaction (xReact) had the highest
  human-judged precision of the nine relations (64% to 72% P@10) [10]. A 2026
  study found that 74% of the variance in people's profiles is state and 26%
  is trait, and that language models are "state-blind": they follow the
  trait and ignore the situation [83].
- **Lab:** 45 of 45 answers fit the trait. This can mean that the model reads
  the trait well, or that it ignores the scene. Section c.3 gives the test.
- **Use:** The main job of the model. Feed it the sim's appraisal facts (G1,
  G4, G6, G13) and read a distribution, not one label.

#### G11. Purpose: what the situation calls for

- **Asks:** What goal fits now? What kind of act?
- **Examples:**
  - "What does this situation call for?"
  - "What is Tom trying to achieve?"
  - "What matters most right now?"
  - "Is this a time to talk or to act?"
  - "Does this call for getting help?" (YN form)
- **Format:** PICK (up to 8), or YN per purpose.
- **Answer source:** Partly. In test scenes the author sets the call. In open
  play, several calls are valid.
- **Why it is a kind:** Graesser's goal orientation (deep reasoning) [1].
  ConceptNet MotivatedByGoal [15]. bAbI task 20, agent's motivations [8].
  Frijda's action tendencies [24]. The Socratic "questions about the
  question" ask for the purpose [5, 6].
- **Evidence:** Lab 60% over 9 phrases. A small trained head gets 85% on new
  needs. As a sentence in the scene, the model's call made the final choice
  worse (38% to 30%) (lab data).
- **Use:** A soft weight on options, as in lab item A3.

#### G12. Means: what things can do (affordance)

- **Asks:** Which thing serves this need? What can this thing do?
- **Examples:**
  - "What does Tom need for this?"
  - "Which thing here can cut rope?"
  - "What can carry water?"
  - "Can the hammer break the lock?"
  - "What is the bucket for?"
- **Format:** POINT (a thing from the scene), or YN per thing.
- **Answer source:** Known (the sim's property system).
- **Why it is a kind:** ConceptNet UsedFor ("the purpose of A is B"),
  CapableOf, HasProperty, ReceivesAction, AtLocation [15]. ATOMIC-2020
  ObjectUse: about 165k object uses, some of them unusual ("wear a popcorn
  bucket as a hat") [11]. Graesser's instrumental/procedural questions [1].
  PIQA [16]. SayCan multiplies the language model's "is this skill useful?"
  by a value function's "can this skill succeed here?" [18].
- **Evidence:** PIQA: humans 95%, models 77% [16]. Text2Afford: pretrained
  models are weak on uncommon affordances; few-shot fine-tuning helps [17].
  COMET reached 91.7% precision at top 1 on ConceptNet [14].
- **Lab:** 60% over 38 phrases. 38 is far above 8 labels.
- **Use:** The sim answers. If the property system has no entry, the model
  may rank common uses. Split long lists (Section d.3).

#### G13. Feasibility and coping

- **Asks:** Can Tom do it? Can he control it? Can he live with the result?
- **Examples:**
  - "Can Tom reach the bandit before he runs?"
  - "Is Tom strong enough to hold the door?"
  - "Can Tom stop this alone?"
  - "Can Tom fix this later?"
  - "Is it too late to help?"
- **Format:** YN.
- **Answer source:** Known (legal actions, reach, strength, time).
- **Why it is a kind:** Scherer's coping potential check (control, power,
  adjustment) [20, 21]. Lazarus's secondary appraisal: coping potential and
  future expectancy [20]. SayCan's value function gives the chance of success
  [18]. bAbI size reasoning and path finding [8].
- **Evidence:** No encoder study in our sources. Physical reasoning [16] and
  numbers [71] are weak.
- **Use:** The sim answers. The answer is an input to G10: Scherer and Moors
  report that control, more than agency, drives the tendency to repair harm
  [21].

#### G14. Preconditions, obstacles and progress

- **Asks:** What must be true first? What is missing? What is in the way? Is
  it done?
- **Examples:**
  - "What is still missing?"
  - "What must happen first?"
  - "What stands in Tom's way?"
  - "Is the task done?"
  - "What step comes next?"
- **Format:** PICK (up to 8), POINT, or YN per precondition.
- **Answer source:** Known (preconditions and state).
- **Why it is a kind:** Graesser's enablement questions [1]. GLUCOSE's
  location, possession and attribute states that enable X [12]. ATOMIC xNeed
  [10]. ConceptNet HasPrerequisite and ObstructedBy [15]. ATOMIC-2020
  HinderedBy, about 106k hindrances to goals [11]. ProPara tracks where each
  thing is and whether it still exists [80]. QDMR (Break) writes a question
  as ordered steps [38].
- **Evidence:** Earlier models did poorly on ProPara state tracking [80]. In
  the lab, the model picks the thing that the scene names ("Dry wood is laid
  in the hearth, but it is not lit" gives the firewood 100%). This matches
  the word-overlap findings in b.2.
- **Lab:** 33% and 83% on two kinds of state.
- **Use:** The sim answers. It can write the state as "what is missing".

---

## (b) Sources for each group

### b.1 Group to source map

"Roots" are the taxonomies that name the kind. "Model evidence" is what the
sources measured.

| group | roots (question taxonomies, psychology) | datasets and relations | model evidence |
| --- | --- | --- | --- |
| G1 relevance | Scherer relevance check [20, 21]; Lazarus primary appraisal [20]; Graesser verification [1, 2] | ATOMIC relevance gate [10] | lab 97% |
| G2 facts, time | 5W1H [4]; Li and Roth classes [7]; Graesser concept completion, feature specification, quantification [1, 2]; Bloom "remember" [3] | bAbI 1-14 [8]; MC-TACO [72]; ProPara [80] | CheckList negation, time, pronouns [55]; numbers [67, 71]; ToMi fact questions easy [34] |
| G3 roles | 5W1H "who" [4]; Li and Roth HUMAN [7] | QA-SRL [9]; bAbI 5 [8]; ATOMIC X vs others [10] | CheckList SRL [55]; Ettinger [68]; Sinha [70]; SocialIQA [13]; Sap 2022 [34] |
| G4 cause, intent | Graesser causal antecedent [1]; Socratic reasons [5, 6]; Scherer agency, Lazarus accountability [20, 21] | GLUCOSE 1-5 [12]; ATOMIC xIntent, xNeed [10]; ATOMIC-2020 xReason [11]; ConceptNet Causes [15]; ART [81] | ART 68.9% vs 91.4% [81]; SocialIQA motivations easier [13] |
| G5 consequence | Graesser causal consequence, expectational [1]; Socratic implications [6]; Endsley projection [27] | ATOMIC xEffect, oEffect [10]; GLUCOSE 6-10 [12]; PIQA [16] | ATOMIC oEffect low [10]; PIQA 77% vs 95% [16] |
| G6 value, urgency | Scherer pleasantness, conduciveness, urgency [20, 21]; Lazarus congruence [20]; OCC desirability, appealingness [22] | ConceptNet Desires [15]; ATOMIC-2020 NotDesires [11] | none found for encoders |
| G7 norms | Scherer norms, fairness [21]; OCC praiseworthiness [22]; Graesser judgmental [1, 2]; Socratic viewpoints [6]; Bloom "evaluate" [3] | Social Chemistry 101 [79] | none found for encoders |
| G8 others' beliefs | Socratic viewpoints [6] | ToMi [28]; BigToM [29]; Hi-ToM [30]; OpenToM [31]; FANToM [32] | weak: [28-34] |
| G9 others' wants, feelings | OCC fortunes of others [22] | ATOMIC oReact, oWant [10]; SocialIQA [13]; OpenToM attitude [31] | agent > others [34]; attitude hardest [31] |
| G10 own feeling, tendency | Frijda [24]; Scherer and Moors path to action tendency [21]; Lazarus [20]; Fleeson [82] | ATOMIC xReact, xWant, xAttr [10]; COMET [14] | xReact best in ATOMIC [10]; state-blind LMs [83]; lab 45/45 |
| G11 purpose | Graesser goal orientation [1]; Socratic "questions about the question" [5, 6] | ConceptNet MotivatedByGoal [15]; bAbI 20 [8]; zero-shot "situation" labels [76] | lab 60% |
| G12 means | Graesser instrumental/procedural [1]; Bloom "apply" [3] | ConceptNet UsedFor, CapableOf, HasProperty [15]; ATOMIC-2020 ObjectUse [11]; PIQA [16]; Text2Afford [17]; SayCan [18]; admissible actions [19] | PIQA [16]; Text2Afford [17]; lab 60% |
| G13 feasibility | Scherer coping potential [20, 21]; Lazarus secondary appraisal [20] | SayCan value function [18]; bAbI 17-19 [8] | weak physical and number skills [16, 71] |
| G14 preconditions | Graesser enablement [1]; QDMR steps [38] | GLUCOSE enabling states [12]; ATOMIC xNeed [10]; ConceptNet HasPrerequisite, ObstructedBy [15]; ATOMIC-2020 HinderedBy [11]; ProPara [80] | lab 33%, 83%; word overlap [66, 77] |

Notes on the taxonomies:

- **Graesser and Person** sort tutoring questions into 18 categories. Six are
  "deep reasoning" questions: causal antecedent, causal consequence, goal
  orientation, instrumental/procedural, enablement, expectational [1]. The
  paper says that verification questions often combine with another category
  [1]. So each of our groups also has a yes/no form. We read the 1994 paper,
  but its category table is an image; we took the category list and the
  definitions from a 2026 paper that reprints them [2].
- **Bloom's taxonomy** sorts learning goals: remember, understand, apply,
  analyze, evaluate, create. Scholars debate whether these levels form a
  strict order [3]. It sorts thinking effort more than question content, so
  it adds little to our grouping.
- **5W1H** (who, what, when, where, why, how) comes from rhetoric and
  journalism [4].
- **Socratic questions** (Paul): clarification, assumptions, reasons and
  evidence, viewpoints and perspectives, implications and consequences, and
  questions about the question [5, 6].
- **Li and Roth** define 6 coarse and 50 fine answer types for factual
  questions [7]. These are types of the answer, not of the thought, so they
  map only to G2 and G3.
- **ATOMIC** has nine if-then relations: xIntent, xNeed, xAttr, xEffect,
  xReact, xWant, oEffect, oReact, oWant (x = PersonX, o = others) [10].
  **ATOMIC-2020** has 23 relations in three groups. Physical entity:
  ObjectUse, AtLocation, MadeUpOf, HasProperty, CapableOf, Desires,
  NotDesires. Event centered: IsAfter, HasSubEvent, IsBefore, HinderedBy,
  Causes, xReason, isFilledBy. Social interaction: the nine ATOMIC relations
  [11]. **COMET** learns to generate these relations [14].
- **GLUCOSE** has ten causal dimensions about an event X. Before X: an event,
  an emotion or basic drive, a location state, a possession state, other
  attributes. After X: an event, an emotion, a change of location, a change
  of possession, other changes [12].
- **ConceptNet** relations that we use: UsedFor, CapableOf, HasProperty,
  AtLocation, Desires, CausesDesire, HasPrerequisite, MotivatedByGoal,
  Causes, HasSubevent, ReceivesAction, ObstructedBy, CreatedBy, PartOf,
  MadeOf [15].
- **Appraisal theory.** Lazarus: primary appraisal (relevance, congruence
  with goals) and secondary appraisal (accountability, coping potential,
  future expectancy). Scherer: relevance, then implications (cause, goal
  conduciveness, urgency), then coping potential (control, power), then
  normative significance, in this order [20]. Scherer and Moors review the
  evidence for the order and for the paths from appraisal to action
  tendencies [21]. **OCC** sorts 22 emotion types by three branches: events
  (desirability), actions of agents (praiseworthiness), objects
  (appealingness). Ortony later proposed about ten types for believable
  characters [22]. We read OCC through [22], not through the 1988 book.
  **Frijda**: emotions are action tendencies [24]; we opened only a record of
  the 1989 paper [23], not its text.
- **Situation awareness** (Endsley): we read the abstract [27]. The three
  levels (perception, comprehension, projection) come from a search summary;
  we did not read them in the paper.
- **Game precedents.** Smith proposes emotion models for non-player
  characters, with acceptance tests from literary character analysis [25].
  Cai et al. run Scherer's appraisal as steps (trigger extraction, appraisal,
  emotion update) in a multi-agent role-play system [26]. We read only their
  abstracts.

### b.2 What zero-shot classifiers get wrong, by language pattern

| pattern | evidence | effect for us | groups |
| --- | --- | --- | --- |
| Word overlap, mention lure | HANS: NLI models use lexical overlap, subsequence and constituent shortcuts [66]. Stress tests: word-overlap distraction [67]. Zero-shot NLI classifiers "heavily rely on spurious lexical patterns"; results vary much across datasets; BERT without NLI tuning is competitive [77]. Misprimes distract models [69]. A hypothesis alone predicts the label in 67% of SNLI [75]. SocialIQA models learn "lexical associations" [13]. FANToM wrong answers overlap more with the text by design, and models fall for them [32]. | The model picks the label that shares words with the scene. Abstract labels ("a danger to others") get no help. | G3, G12, G14, G8 |
| Negation | BERT is insensitive to negation [68]; "Birds cannot [MASK]" ~ "Birds can [MASK]" [69]; negation is rare in NLI data and hard for transformers [73]; large LMs too [74]; CheckList 67.5% to 100% failure [55]; negation words correlate with a label in NLI data [75]. | Do not put "not" in labels or questions. Write the missing thing as a need ("He still needs a flame"). | G2, G14, G7 |
| Who did what to whom | CheckList subject/object 60.8% and 95.7% failure [55]; role-based prediction [68]; word-order insensitivity: 98.7% of MNLI examples have a shuffle that still gets the gold label [70]; wrong participant [13, 34]. | Roles from the sim. Name one person per question. | G3, G4, G9 |
| Numbers, comparisons | Numerical stress test [67]; BERT's word pieces code numbers less precisely [71]; CheckList "least worried" 91.3% failure; "more optimistic" vs "more pessimistic" 100% [55]. | Turn numbers into YN thresholds from the sim. | G2, G13 |
| Time order | CheckList before/after 82.9% failure [55]; MC-TACO ~20% below humans [72]; SocialIQA wrong timing [13]; bAbI time task [8]. | Order from the event log. Say "before" and "after" in plain, fixed forms. | G2, G4, G5 |
| Pronouns | CheckList he/she, his/her, former/latter 100% failure [55]; bAbI 11, 13 [8]. | Repeat names; no pronouns in the scene text. | all |
| Other minds vs facts | Fact questions easy, belief questions hard [34]; backward belief below chance [29]; higher order collapses [30]; list "who knows" harder than choice [32]. | Beliefs from the belief record. | G8, G9 |
| Self vs others | Agent-centric better than other-oriented [34]; ATOMIC xReact vs oEffect [10]. | Ask about Tom; supply others from the sim. | G9, G10 |
| Label prior, wording, order | Models prefer some answers with no input; a content-free input ("N/A") measures and removes this [58]. Different words for one answer split the probability ("computer" vs "PC") [59]. LLMs prefer some option positions [57]. | Measure the prior per label and subtract it (lab item A4). Avoid near-synonyms in one list. Shuffle the order in tests. | all PICK |
| Long lists | Li and Roth: one label right 91.0% (coarse) and 84.2% (fine); the right label in the kept set (5 or fewer) 98.8% and 95.0% [7]. | Keep the top few; split 38 phrases into a hierarchy or into YN per thing. | G12, G11 |
| Ambiguous items | Models are near perfect on high-agreement NLI items and near chance on low-agreement ones [63]. | Expect low accuracy where people disagree; score those items with c.4. | G7, G9, G10, G11 |

We found no source that tests GLiNER-family models on these patterns.
GLiNER2 is an encoder with a schema interface for entities, classification
and structured extraction [78]. The name "GLiNER2.5-Decide" and the size 340M
come from our lab, not from [78]. For scale: SocialIQA reports BERT-large, a
340M encoder, at 64.5% (humans 84.4%) after full training [13].

---

## (c) How to evaluate a question kind with no known right answer

### c.1 Why one gold label fails

Several answers can be right. ATOMIC says its annotations "correspond to a
distribution over likely inferences" [10]. People disagree on many NLI items,
and the disagreement stays when more raters join [64, 63]. Plank argues that
this variation is signal, not noise [65]. Fleeson shows that one person shows
most levels of a trait over time [82]. So we score a question kind by
properties that do not need one gold label.

### c.2 Consistency: the same meaning gives the same answer

These checks need no label.

1. **Paraphrase invariance.** Ask the question in 3 to 5 wordings. ParaRel
   uses 328 paraphrases of 38 relations and finds poor consistency in all
   models [51]. CALUM finds inconsistent answers "even for semantically
   identical inputs" [52]. Measure: the share of top answers that agree, and
   the distance between the answer distributions.
2. **Label wording and order invariance.** Shuffle the label order. Swap one
   label for a near-synonym. LLMs prefer some option positions; PriDe
   estimates the preference from a few permutations [57]. Different words
   for the same answer split the probability [59].
3. **Format agreement.** Ask the PICK question, then ask each label as YN
   ("Is Tom angry?"). The two must agree. FANToM counts an item right only
   when all its question formats are right; models score much lower on that
   joint score [32]. Binary and list formats fail in different ways [32].
4. **Negation and implication consistency.** If the top answer is "angry",
   then "Is Tom angry?" must be yes and "Is Tom calm?" must be no. Ribeiro et
   al. penalize a model that says "red" to "What color is the rose?" and "no"
   to "Is the rose red?" [53]. CheckList has the same idea as invariance
   tests [55].
5. **Chain consistency.** Answers along a chain must fit. OpenToM checks that
   a fine location answer agrees with the coarse answer it depends on, and
   finds many unfaithful pairs [31]. For us: a bad event that another person
   caused on purpose (G4, G6) should raise "wants to strike back" more than
   the same event caused by chance. Agency or blame plus a bad event gives a
   tendency to aggress [21].

### c.3 Sensitivity: the answer moves for the right reasons

1. **Contrast sets (minimal pairs).** Change one fact that should flip the
   answer. "The bandit hits Mara" against "The bandit gives Mara bread".
   Contrast sets cut model scores by up to 25% [54]. Measure the flip rate
   on should-flip pairs, and the stay rate on should-stay pairs (CheckList
   INV and DIR tests [55]).
2. **Directional tests.** More danger should not make "afraid" less likely.
   CheckList calls these directional expectation tests [55].
3. **Counterfactual factors.** Use a two-factor grid: trait (hot-tempered,
   timid, kind) times scene (threat to Tom, threat to a friend, an insult, a
   gift). Both factors should move the answer. Kaushik et al. show that
   models trained on original data fail on minimally revised data, and the
   reverse [56]. Language models can be "state-blind": they follow the trait
   and ignore the situation [83]. This is the test for the lab's 45 of 45.
4. **Content-free baseline.** Ask the question with an empty scene or "N/A".
   An answer that does not differ from this prior gives no information about
   the scene [58].
5. **Lure test.** Add one sentence that names a wrong label. The answer
   should not move. Irrelevant context lowers LLM accuracy much [45];
   misprimes distract BERT [69].

### c.4 Soft targets: a set or a distribution

1. **Acceptable set.** For each item, list all acceptable answers (from raters
   or from a stronger model). Score "top answer is in the set". ATOMIC and
   COMET use this kind of human validity judgment (P@k) [10, 14].
2. **Distribution match.** Collect many ratings per item and compare the
   model's distribution to the human one. ChaosNLI collects 100 ratings per
   item and shows that models do not match the human distribution [63].
3. **Split by agreement.** Report scores for high-agreement and low-agreement
   items apart [63].
4. **Calibration.** A calibrated model is right as often as its confidence
   says. Modern networks are often over-confident; temperature scaling (one
   number that softens or sharpens the scores) fixes much of it [60]. BERT
   and RoBERTa are fairly calibrated in domain [61]. Without gold labels,
   calibrate against the acceptable set.

### c.5 A stronger model as a judge

A strong LLM judge reached over 80% agreement with human preferences, about
the same as agreement between humans [62]. The same study found position
bias, verbosity bias and self-enhancement bias [62]. So: let the judge give
an acceptable set, not one label; shuffle the label order for the judge
too; check the judge against a human sample.

### c.6 Human rating rounds

Ask raters "Is this answer valid for this scene?" (YN), not "What is the
answer?". ATOMIC used this form; raters judged gold annotations valid 86.2%
of the time [10]. Use three or more raters per item and keep all ratings
[65]. Smith proposes acceptance tests for non-player character emotion built
from literary character analysis [25].

### c.7 Value for the final choice (the decisive test)

The final choice often has a known answer: the sim blesses it. So test each
question kind by its effect on the final choice, in four arms:

| arm | what the chooser gets |
| --- | --- |
| none | no preliminary answer |
| model | the model's preliminary answer |
| oracle | the right preliminary answer (sim or author) |
| shuffled | a preliminary answer from another scene |

If "model" is not better than "shuffled", the question adds noise. If
"oracle" is not better than "none", the question kind does not help the
choice at all. Lanham et al. show that models differ in how much they use an
intermediate step [44], so measure this dependence and do not assume it.

### c.8 A proposed test card per question kind

These thresholds are our proposal, not values from the sources.

| check | pass (proposal) |
| --- | --- |
| known-answer accuracy (on the sim's graded cases) | 85% or more for YN; 70% or more for PICK |
| paraphrase agreement (top answer, 4 wordings) | 80% or more |
| order invariance (top answer, 3 shuffles) | 90% or more |
| PICK vs YN agreement | 80% or more |
| contrast pairs: flip on should-flip / stay on should-stay | 70% / 90% or more |
| distance from the content-free answer | clearly above zero on most scenes |
| lure test: answer unchanged | 90% or more |
| final choice: model arm minus shuffled arm | 5 points or more |

---

## (d) Error cascades in chained questions

### d.1 Evidence

- **Pipelines compound errors.** Finkel et al.: a greedy "1-best" pipeline
  passes only the best output of each stage, so "errors propagate and
  compound at each stage". Semantic role labeling fell from 91.2 to 80.0
  F-measure when real parses replaced gold parses. Passing k-best lists
  gives "useful improvements". Their own method samples from the whole
  distribution at each stage, and gains on role labeling and on textual
  entailment [42].
- **Keeping a set helps.** Li and Roth's classifier keeps up to 5 coarse
  classes (enough to cover 0.95 of the probability) and passes them all to
  the fine classifier. The right class is in the kept set 98.8% (coarse) and
  95.0% (fine) of the time, against 91.0% and 84.2% for one label [7].
- **Composition fails even when the parts are right.** Self-ask measures a
  "compositionality gap": the model knows both facts but fails the two-hop
  question. The gap stays near 40% as GPT-3 grows [35]. Error rates grow with
  the number of steps [46].
- **The parts can fail when the whole is right.** Multi-hop models often fail
  the single-hop sub-questions of questions that they get right [41]. FANToM
  and OpenToM find the same pattern for belief questions [31, 32]. So a chain
  can add errors that a direct choice avoids.
- **Early mistakes snowball.** A model "over-commits to early mistakes" and
  then makes more. When asked apart, ChatGPT and GPT-4 spotted 67% and 87% of
  their own mistakes [47].
- **A wrong preliminary answer acts as a distractor.** Irrelevant context
  lowers accuracy much [45]. In our lab, the model's "calls for" answer as a
  sentence in the scene made the choice worse (38% to 30%) (lab data).
- **Dependence varies.** Models sometimes use their intermediate steps and
  sometimes ignore them [44].
- **Decomposition methods.** Least-to-most prompting solves subproblems in
  order and reached 99% on SCAN, against 16% with chain-of-thought [36].
  Successive prompting asks one simple question at a time and gained about 5
  F1 on DROP [37]. QDMR writes a question as ordered steps with operators
  such as SELECT, FILTER, PROJECT, AGGREGATE, COMPARATIVE, BOOLEAN [38].
  Unsupervised decomposition also helps multi-hop QA [40]. These methods use
  large generative models; for our encoder the lesson is the step design,
  not the prompts.

### d.2 The arithmetic

If each step is right with chance p and the steps fail independently, a
chain of k hard picks is all right with chance p to the power k. Two steps at
60% give 36%. Three steps at 97% give 91%. So only strong steps can be
chained with hard picks.

Our rule: a wrong answer costs more than a right one gains. Let a right
preliminary answer gain G and a wrong one cost C. Using an answer with
confidence p pays off only when p times G is more than (1 - p) times C, that
is, when p is more than C / (C + G). If C = 2G, the gate is p > 0.67. Chow
describes this kind of optimal reject rule and its error-reject trade-off
[49]. The gate needs calibrated confidence (c.4).

### d.3 Ways to limit cascades

1. **Do not ask what the sim knows.** Seven groups are known, six partly
   known (Section a). A sim answer is right every time. SayCan takes its
   feasibility score from a value function, not from the language model [18].
2. **Keep alternatives.** Pass the top few answers or the whole distribution
   [7, 42]. Conformal prediction gives a set that contains the right answer
   with a chosen probability, for example 90%, with no assumptions about the
   model [50].
3. **Soft weights, not hard filters or sentences.** Add a weight times the
   log probability of the preliminary answer to each option's score. SayCan
   multiplies the "useful" and "possible" probabilities and picks the best
   product [18]. A hard filter makes a wrong answer final. (Lab item A3 tests
   this for "calls for".)
4. **Confidence gates.** Use an answer only above the gate from d.2. A
   trained calibrator beats raw softmax for abstention: 56% of questions
   answered at 80% accuracy, against 48% [48].
5. **Agreement gates.** Ask 2 or 3 paraphrases; use the answer only when they
   agree. Self-consistency samples many reasoning paths and takes the most
   common answer [43]. For an encoder, paraphrases play the role of samples.
6. **Rescore whole chains.** Score the final choice under each kept
   preliminary answer, weight by its probability, and sum. DecompRC scores
   each whole decomposition (sub-questions and answers) to pick the final
   answer [39].
7. **Short chains, simple steps.** Prefer YN steps (lab 97%) to PICK steps
   (lab 60%). Turn a long list into a hierarchy (coarse, then fine) or into YN
   per item [7].
8. **Remove the prior.** Subtract each label's answer on a content-free
   input, so that favourite labels do not flow down the chain [58, 59] (lab
   item A4).
9. **Write the state without lures.** Say what is missing, not what is
   present and done; avoid negation (b.2; lab item B2).
10. **Test the chain as a whole.** Use the four arms of c.7 for the chain, not
    only for each step.

### d.4 A proposed chain for Tom

This follows Scherer's order of checks [20, 21] and runs the appraisal as
steps, as Cai et al. do with an LLM [26]. The sim does every step that it
can. The model does only the last step.

| step | question | who answers | passed on as |
| --- | --- | --- | --- |
| 1 | Is this new, and does it matter to Tom? (G1) | sim (model as a check) | weight per sentence |
| 2 | Who is doing what to whom? (G3) | sim | fixed facts |
| 3 | Is it good or bad for Tom's goals, and is it urgent? (G6) | sim for the goal; model for "how much" | probability |
| 4 | Who caused it? On purpose? (G4) | sim | fixed facts |
| 5 | Can Tom control it? (G13) | sim | fixed facts |
| 6 | Is it allowed? Would the village approve? (G7) | sim for the law; model for Tom's view | probability |
| 7 | How does Tom feel, and what does he tend to do? (G10) | model | distribution over tendencies |
| 8 | Final choice from the legal menu | model plus weights from steps 1 to 7 | the action |

Only steps 3, 6 and 7 carry model uncertainty. Each passes a distribution,
behind a confidence gate, as a soft weight. Test the chain with c.7.

---

## (e) References

Each entry gives the title, the authors, the year and the URL that we opened.
"Abstract only" means that we read the abstract page, not the full text.

1. Question Asking During Tutoring. A. C. Graesser, N. K. Person. 1994
   (American Educational Research Journal 31(1)). https://gwern.net/doc/psychology/spaced-repetition/1994-graesser.pdf
   (text read; the category table is an image and did not extract).
2. Analysis of Types of Inquiries in Student-AI Interaction: A case study of
   two CS2 tasks. M. Amoozadeh, A. Alipour. 2026. https://arxiv.org/pdf/2608.17919
   (Table 2 reprints the 18 Graesser categories).
3. Bloom's taxonomy. Wikipedia. Accessed 2026-09-30. https://en.wikipedia.org/wiki/Bloom%27s_taxonomy
4. Five Ws. Wikipedia. Accessed 2026-09-30. https://en.wikipedia.org/wiki/Five_Ws
5. The Thinker's Guide to the Art of Socratic Questioning. R. Paul, L. Elder.
   2006. https://www.criticalthinking.org/files/SocraticQuestioning2006.pdf
6. The Six Types of Socratic Questions (course handout after R. W. Paul;
   author not stated). https://dr-hatfield.com/human_development/docs/6_type_socratic_questions.pdf
7. Learning Question Classifiers. X. Li, D. Roth. 2002 (COLING).
   https://aclanthology.org/C02-1150/ and https://aclanthology.org/C02-1150.pdf ;
   class list: https://cogcomp.seas.upenn.edu/Data/QA/QC/definition.html
8. Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks
   (bAbI). J. Weston, A. Bordes, S. Chopra, A. M. Rush, B. van Merriënboer,
   A. Joulin, T. Mikolov. 2015. https://arxiv.org/abs/1502.05698
9. Question-Answer Driven Semantic Role Labeling: Using Natural Language to
   Annotate Natural Language. L. He, M. Lewis, L. Zettlemoyer. 2015 (EMNLP).
   https://aclanthology.org/D15-1076/
10. ATOMIC: An Atlas of Machine Commonsense for If-Then Reasoning. M. Sap,
    R. Le Bras, E. Allaway, C. Bhagavatula, N. Lourie, H. Rashkin, B. Roof,
    N. A. Smith, Y. Choi. 2019. https://arxiv.org/abs/1811.00146
11. COMET-ATOMIC 2020: On Symbolic and Neural Commonsense Knowledge Graphs.
    J. D. Hwang, C. Bhagavatula, R. Le Bras, J. Da, K. Sakaguchi,
    A. Bosselut, Y. Choi. 2021 (AAAI). https://arxiv.org/abs/2010.05953
12. GLUCOSE: GeneraLized and COntextualized Story Explanations.
    N. Mostafazadeh, A. Kalyanpur, L. Moon, D. Buchanan, L. Berkowitz,
    O. Biran, J. Chu-Carroll. 2020 (EMNLP). https://arxiv.org/abs/2009.07758
13. SocialIQA: Commonsense Reasoning about Social Interactions. M. Sap,
    H. Rashkin, D. Chen, R. Le Bras, Y. Choi. 2019 (EMNLP).
    https://arxiv.org/abs/1904.09728
14. COMET: Commonsense Transformers for Automatic Knowledge Graph
    Construction. A. Bosselut, H. Rashkin, M. Sap, C. Malaviya,
    A. Celikyilmaz, Y. Choi. 2019 (ACL). https://arxiv.org/abs/1906.05317
    (abstract only).
15. ConceptNet 5.5: An Open Multilingual Graph of General Knowledge.
    R. Speer, J. Chin, C. Havasi. 2016. https://arxiv.org/abs/1612.03975
    (abstract only); relation definitions: https://github.com/commonsense/conceptnet5/wiki/Relations
16. PIQA: Reasoning about Physical Commonsense in Natural Language. Y. Bisk,
    R. Zellers, R. Le Bras, J. Gao, Y. Choi. 2020 (AAAI).
    https://arxiv.org/abs/1911.11641 (abstract only).
17. Text2Afford: Probing Object Affordance Prediction abilities of Language
    Models solely from Text. S. Adak, D. Agrawal, A. Mukherjee, S. Aditya.
    2024 (CoNLL). https://arxiv.org/abs/2402.12881 (abstract only).
18. Do As I Can, Not As I Say: Grounding Language in Robotic Affordances
    (SayCan). M. Ahn et al. (45 authors). 2022. https://arxiv.org/abs/2204.01691
    and https://say-can.github.io/
19. Language Models as Zero-Shot Planners: Extracting Actionable Knowledge for
    Embodied Agents. W. Huang, P. Abbeel, D. Pathak, I. Mordatch. 2022.
    https://arxiv.org/abs/2201.07207 (abstract only).
20. Appraisal theory. Wikipedia. Accessed 2026-09-30. https://en.wikipedia.org/wiki/Appraisal_theory
21. The Emotion Process: Event Appraisal and Component Differentiation.
    K. R. Scherer, A. Moors. 2019 (Annual Review of Psychology 70).
    https://ppw.kuleuven.be/okp/_pdf/Scherer2019TEPEA.pdf
22. The Relationship Between Emotion Models and Artificial Intelligence.
    C. Bartneck, M. J. Lyons, M. Saerbeck. 2017. https://arxiv.org/abs/1706.09554
    (full text read for the OCC summary).
23. Relations among emotion, appraisal, and emotional action readiness.
    N. H. Frijda, P. Kuipers, E. ter Schure. 1989 (Journal of Personality and
    Social Psychology 57). Record only, no abstract or text:
    https://api.semanticscholar.org/graph/v1/paper/DOI:10.1037/0022-3514.57.2.212?fields=title,authors,year,abstract,venue
24. Nico Frijda. Wikipedia. Accessed 2026-09-30. https://en.wikipedia.org/wiki/Nico_Frijda
    (summary of his action-tendency theory).
25. Start Your EM(otion En)gine: Towards Computational Models of Emotion for
    Improving the Believability of Video Game Non-Player Characters.
    G. M. Smith. 2023. https://arxiv.org/abs/2307.10031 (abstract only).
26. From Triggers to Emotions: A CPM-Grounded Appraisal Multi-Agent for
    Dynamic Emotional Evolution in Persona-Based Dialogue. J. Cai, S. Liu,
    A. Rehman, Y. Guo, Q. Tian, T. Dolby, S. Green, C. Cox, X. Yang. 2026.
    https://arxiv.org/abs/2607.07824 (abstract only).
27. Toward a Theory of Situation Awareness in Dynamic Systems. M. R. Endsley.
    1995 (Human Factors). https://trid.trb.org/View/426377 (abstract only;
    the three levels are from a search summary, not verified in the paper).
28. Revisiting the Evaluation of Theory of Mind through Question Answering
    (ToMi). M. Le, Y-L. Boureau, M. Nickel. 2019 (EMNLP-IJCNLP).
    https://aclanthology.org/D19-1598/ (abstract only).
29. Understanding Social Reasoning in Language Models with Language Models
    (BigToM). K. Gandhi, J-P. Fränken, T. Gerstenberg, N. D. Goodman. 2023.
    https://arxiv.org/abs/2306.15448 and https://arxiv.org/pdf/2306.15448
30. HI-TOM: A Benchmark for Evaluating Higher-Order Theory of Mind Reasoning
    in Large Language Models. Y. He, Y. Wu, Y. Jia, R. Mihalcea, Y. Chen,
    N. Deng. 2023. https://arxiv.org/abs/2310.16755 and https://arxiv.org/pdf/2310.16755
31. OpenToM: A Comprehensive Benchmark for Evaluating Theory-of-Mind Reasoning
    Capabilities of Large Language Models. H. Xu, R. Zhao, L. Zhu, J. Du,
    Y. He. 2024. https://arxiv.org/abs/2402.06044 and https://arxiv.org/pdf/2402.06044
32. FANToM: A Benchmark for Stress-testing Machine Theory of Mind in
    Interactions. H. Kim, M. Sclar, X. Zhou, R. Le Bras, G. Kim, Y. Choi,
    M. Sap. 2023 (EMNLP). https://arxiv.org/abs/2310.15421 and https://arxiv.org/pdf/2310.15421
33. Large Language Models Fail on Trivial Alterations to Theory-of-Mind
    Tasks. T. Ullman. 2023. https://arxiv.org/abs/2302.08399 (abstract only).
34. Neural Theory-of-Mind? On the Limits of Social Intelligence in Large LMs.
    M. Sap, R. Le Bras, D. Fried, Y. Choi. 2022 (EMNLP).
    https://arxiv.org/abs/2210.13312 and https://arxiv.org/pdf/2210.13312
35. Measuring and Narrowing the Compositionality Gap in Language Models
    (self-ask). O. Press, M. Zhang, S. Min, L. Schmidt, N. A. Smith, M. Lewis.
    2023 (Findings of EMNLP). https://arxiv.org/abs/2210.03350 and https://arxiv.org/pdf/2210.03350
36. Least-to-Most Prompting Enables Complex Reasoning in Large Language
    Models. D. Zhou, N. Schärli, L. Hou, J. Wei, N. Scales, X. Wang,
    D. Schuurmans, C. Cui, O. Bousquet, Q. Le, E. Chi. 2023 (ICLR).
    https://arxiv.org/abs/2205.10625 (abstract only).
37. Successive Prompting for Decomposing Complex Questions. D. Dua, S. Gupta,
    S. Singh, M. Gardner. 2022. https://arxiv.org/abs/2212.04092
    (abstract only).
38. Break It Down: A Question Understanding Benchmark (QDMR). T. Wolfson,
    M. Geva, A. Gupta, M. Gardner, Y. Goldberg, D. Deutch, J. Berant. 2020. https://arxiv.org/abs/2001.11770 and https://arxiv.org/pdf/2001.11770
39. Multi-hop Reading Comprehension through Question Decomposition and
    Rescoring (DecompRC). S. Min, V. Zhong, L. Zettlemoyer, H. Hajishirzi.
    2019 (ACL). https://arxiv.org/abs/1906.02916 (abstract only).
40. Unsupervised Question Decomposition for Question Answering. E. Perez,
    P. Lewis, W. Yih, K. Cho, D. Kiela. 2020.
    https://arxiv.org/abs/2002.09758 (abstract only).
41. Do Multi-Hop Question Answering Systems Know How to Answer the Single-Hop
    Sub-Questions? Y. Tang, H. T. Ng, A. K. H. Tung. 2020.
    https://arxiv.org/abs/2002.09919 (abstract only).
42. Solving the Problem of Cascading Errors: Approximate Bayesian Inference
    for Linguistic Annotation Pipelines. J. R. Finkel, C. D. Manning,
    A. Y. Ng. 2006 (EMNLP). https://aclanthology.org/W06-1673/ and https://aclanthology.org/W06-1673.pdf
43. Self-Consistency Improves Chain of Thought Reasoning in Language Models.
    X. Wang, J. Wei, D. Schuurmans, Q. Le, E. Chi, S. Narang, A. Chowdhery,
    D. Zhou. 2023 (ICLR). https://arxiv.org/abs/2203.11171 (abstract only).
44. Measuring Faithfulness in Chain-of-Thought Reasoning. T. Lanham et al.
    (30 authors, last E. Perez). 2023. https://arxiv.org/abs/2307.13702
    (abstract only).
45. Large Language Models Can Be Easily Distracted by Irrelevant Context.
    F. Shi, X. Chen, K. Misra, N. Scales, D. Dohan, E. Chi, N. Schärli,
    D. Zhou. 2023 (ICML). https://arxiv.org/abs/2302.00093 (abstract only).
46. Faith and Fate: Limits of Transformers on Compositionality. N. Dziri et
    al. (16 authors, last Y. Choi). 2023. https://arxiv.org/abs/2305.18654
    (abstract only).
47. How Language Model Hallucinations Can Snowball. M. Zhang, O. Press,
    W. Merrill, A. Liu, N. A. Smith. 2023. https://arxiv.org/abs/2305.13534
    (abstract only).
48. Selective Question Answering under Domain Shift. A. Kamath, R. Jia,
    P. Liang. 2020. https://arxiv.org/abs/2006.09462 (abstract only).
49. On optimum recognition error and reject tradeoff. C. K. Chow. 1970 (IEEE
    Transactions on Information Theory). Record and abstract:
    https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/TIT.1970.1054406?fields=title,authors,year,abstract,venue
50. A Gentle Introduction to Conformal Prediction and Distribution-Free
    Uncertainty Quantification. A. N. Angelopoulos, S. Bates. 2021.
    https://arxiv.org/abs/2107.07511 (abstract only).
51. Measuring and Improving Consistency in Pretrained Language Models
    (ParaRel). Y. Elazar, N. Kassner, S. Ravfogel, A. Ravichander, E. Hovy,
    H. Schütze, Y. Goldberg. 2021. https://arxiv.org/abs/2102.01017
    (abstract only).
52. Accurate, yet inconsistent? Consistency Analysis on Language Understanding
    Models. M. Jang, D. S. Kwon, T. Lukasiewicz. 2021.
    https://arxiv.org/abs/2108.06665 (abstract only).
53. Are Red Roses Red? Evaluating Consistency of Question-Answering Models.
    M. T. Ribeiro, C. Guestrin, S. Singh. 2019 (ACL).
    https://aclanthology.org/P19-1621/
54. Evaluating Models' Local Decision Boundaries via Contrast Sets.
    M. Gardner, Y. Artzi, V. Basmova, J. Berant, B. Bogin, and 21 more. 2020. https://arxiv.org/abs/2004.02709 (abstract only).
55. Beyond Accuracy: Behavioral Testing of NLP models with CheckList.
    M. T. Ribeiro, T. Wu, C. Guestrin, S. Singh. 2020.
    https://arxiv.org/abs/2005.04118 and https://arxiv.org/pdf/2005.04118
56. Learning the Difference that Makes a Difference with
    Counterfactually-Augmented Data. D. Kaushik, E. Hovy, Z. C. Lipton. 2020
    (ICLR). https://arxiv.org/abs/1909.12434 (abstract only).
57. Large Language Models Are Not Robust Multiple Choice Selectors. C. Zheng,
    H. Zhou, F. Meng, J. Zhou, M. Huang. 2024 (ICLR).
    https://arxiv.org/abs/2309.03882 (abstract only).
58. Calibrate Before Use: Improving Few-Shot Performance of Language Models.
    T. Z. Zhao, E. Wallace, S. Feng, D. Klein, S. Singh. 2021 (ICML).
    https://arxiv.org/abs/2102.09690 (abstract only).
59. Surface Form Competition: Why the Highest Probability Answer Isn't Always
    Right. A. Holtzman, P. West, V. Shwartz, Y. Choi, L. Zettlemoyer. 2021. https://arxiv.org/abs/2104.08315 (abstract only).
60. On Calibration of Modern Neural Networks. C. Guo, G. Pleiss, Y. Sun,
    K. Q. Weinberger. 2017. https://arxiv.org/abs/1706.04599 (abstract
    only).
61. Calibration of Pre-trained Transformers. S. Desai, G. Durrett. 2020
    (EMNLP). https://arxiv.org/abs/2003.07892 (abstract only).
62. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. L. Zheng,
    W-L. Chiang, Y. Sheng, and 10 more. 2023.
    https://arxiv.org/abs/2306.05685 (abstract only).
63. What Can We Learn from Collective Human Opinions on Natural Language
    Inference Data? (ChaosNLI). Y. Nie, X. Zhou, M. Bansal. 2020.
    https://arxiv.org/abs/2010.03532 (abstract only).
64. Inherent Disagreements in Human Textual Inferences. E. Pavlick,
    T. Kwiatkowski. 2019 (TACL). https://aclanthology.org/Q19-1043/
65. The 'Problem' of Human Label Variation: On Ground Truth in Data, Modeling
    and Evaluation. B. Plank. 2022 (EMNLP). https://arxiv.org/abs/2211.02570
    (abstract only).
66. Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in Natural
    Language Inference (HANS). R. T. McCoy, E. Pavlick, T. Linzen. 2019.
    https://arxiv.org/abs/1902.01007 (abstract only).
67. Stress Test Evaluation for Natural Language Inference. A. Naik,
    A. Ravichander, N. Sadeh, C. Rose, G. Neubig. 2018 (COLING).
    https://arxiv.org/abs/1806.00692 (abstract only).
68. What BERT is not: Lessons from a new suite of psycholinguistic diagnostics
    for language models. A. Ettinger. 2019 (revised 2020). https://arxiv.org/abs/1907.13528
    (abstract only).
69. Negated and Misprimed Probes for Pretrained Language Models: Birds Can
    Talk, But Cannot Fly. N. Kassner, H. Schütze. 2020.
    https://arxiv.org/abs/1911.03343 (abstract only).
70. UnNatural Language Inference. K. Sinha, P. Parthasarathi, J. Pineau,
    A. Williams. 2021. https://arxiv.org/abs/2101.00010 (abstract only).
71. Do NLP Models Know Numbers? Probing Numeracy in Embeddings. E. Wallace,
    Y. Wang, S. Li, S. Singh, M. Gardner. 2019.
    https://arxiv.org/abs/1909.07940 (abstract only).
72. "Going on a vacation" takes longer than "Going for a walk": A Study of
    Temporal Commonsense Understanding (MC-TACO). B. Zhou, D. Khashabi,
    Q. Ning, D. Roth. 2019 (EMNLP). https://arxiv.org/abs/1909.03065 and https://arxiv.org/pdf/1909.03065
73. An Analysis of Natural Language Inference Benchmarks through the Lens of
    Negation. M. M. Hossain, V. Kovatchev, P. Dutta, T. Kao, E. Wei,
    E. Blanco. 2020 (EMNLP). https://aclanthology.org/2020.emnlp-main.732/
74. Language models are not naysayers: An analysis of language models on
    negation benchmarks. T. H. Truong, T. Baldwin, K. Verspoor, T. Cohn. 2023.
    https://arxiv.org/abs/2306.08189 (abstract only).
75. Annotation Artifacts in Natural Language Inference Data. S. Gururangan,
    S. Swayamdipta, O. Levy, R. Schwartz, S. R. Bowman, N. A. Smith. 2018. https://arxiv.org/abs/1803.02324 (abstract only).
76. Benchmarking Zero-shot Text Classification: Datasets, Evaluation and
    Entailment Approach. W. Yin, J. Hay, D. Roth. 2019.
    https://arxiv.org/abs/1909.00161 (abstract only).
77. Issues with Entailment-based Zero-shot Text Classification. T. Ma,
    J-G. Yao, C-Y. Lin, T. Zhao. 2021 (ACL, short).
    https://aclanthology.org/2021.acl-short.99/
78. GLiNER2: An Efficient Multi-Task Information Extraction System with
    Schema-Driven Interface. U. Zaratiana, G. Pasternak, O. Boyd,
    G. Hurn-Maloney, A. Lewis. 2025. https://arxiv.org/abs/2507.18546
    (abstract only).
79. Social Chemistry 101: Learning to Reason about Social and Moral Norms.
    M. Forbes, J. D. Hwang, V. Shwartz, M. Sap, Y. Choi. 2020 (EMNLP).
    https://arxiv.org/abs/2011.00620 (abstract only).
80. Tracking State Changes in Procedural Text: A Challenge Dataset and Models
    for Process Paragraph Comprehension (ProPara). B. Dalvi Mishra, L. Huang,
    N. Tandon, W. Yih, P. Clark. 2018. https://arxiv.org/abs/1805.06975
    (abstract only).
81. Abductive Commonsense Reasoning (ART). C. Bhagavatula, R. Le Bras,
    C. Malaviya, K. Sakaguchi, A. Holtzman, H. Rashkin, D. Downey, W. Yih,
    Y. Choi. 2019. https://arxiv.org/abs/1908.05739 (abstract only).
82. Toward a Structure- and Process-Integrated View of Personality: Traits as
    Density Distributions of States. W. Fleeson. 2001 (Journal of Personality
    and Social Psychology 80). http://simine.com/407/readings/Fleeson_2001.pdf
83. Beyond Fixed Psychological Personas: State Beats Trait, but Language
    Models are State-Blind. T. Harry, I. Ngong, C. Nweke, Y. Feng, J. Near.
    2026. https://arxiv.org/abs/2601.15395 (abstract only).

**Not verified.** We could not open the OCC book (Ortony, Clore and Collins,
1988) or Clore and Ortony (2013); OCC facts come from [22]. We could not open
the full Frijda (1989) paper [23] or Endsley (1995) [27]. For Graesser [1],
the category table did not extract; the list comes from [2]. Venue names in
parentheses come from the pages that we opened. Where a page did not name
the venue, we left it out.
