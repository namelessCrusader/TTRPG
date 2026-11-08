# Plan v3: How should small models run characters in the world?

Written in Simplified Technical English. Short sentences. One idea in each
sentence. Each term has one meaning in all of the plan (Section 3). Numbers
marked "(proposed)" are for your notes.

## 1. Context

### What we measured so far

- Laya (a ready-made choice model) reads the purpose of a line well.
- Laya chooses badly from long lists: 17 in 20 with 4 options, 4 in 20 with
  64 options. We do not know why.
- In the locked room, Laya did no better than a word-matching rule. We do not
  know why.
- Qwen3-0.6B scored each option alone and did better with long lists. This
  can be the scoring method. It can also be the model's deeper pretraining.
  The test did not separate the two.

### What was wrong with the earlier plans

- **They assumed one correct answer.** A character has valid actions, some
  information about its situation, and a history. Many actions can make
  sense. A character with the trait "tries to do what is right" will try to
  do what it believes is right. That is also personality.
- **They did not isolate causes.** When a test fails, we must know which
  part failed.
- **They compared models and methods at the same time.** A difference can
  come from either.
- **The locked room tested the wrong thing.** It tested problem solving that
  we never specified to the model. It did not test social behavior or
  personality. Most of its problems (no layout, no view, no memory) would
  also occur in an open room.

### Decisions already made

- The player selects a thing or a place. The actions for it are shown in
  groups. The player can also make complex actions (Section 7).
- The new test world is built in the lab first.
- A large model can be used offline, never during play: as a judge, as a
  data source, to make embeddings, and to decide effects for training data.
- Work on the laptop first. One GPU job at a time. Training can move to your
  machine.

## 2. Principles

1. **No single correct answer.** We judge a choice on three questions. Is it
   valid (the world permits it)? Does it make sense for this character, with
   what it knows and its history? Does it add to the scene? Many choices can
   pass all three.
2. **Change one thing at a time.** Each test changes one factor. All other
   factors stay the same.
3. **Separate the model from the method.** Test each method on the same
   model. Test each model with the same method.
4. **Research before we choose.** For personality, we first read what exists.
   Then we choose what to test.
5. **The world keeps running.** A character's action takes time. The world
   does not wait. A character can stop and change its action, but its
   momentum (physical or emotional) continues.
6. **One GPU job at a time.** Before each run, we check that the model fits
   in GPU memory. We run on the GPU, not the CPU.

## 3. Terms

| term | meaning |
| --- | --- |
| world | the program that decides what is true. First the lab world, later the voxel simulation. |
| thing | one object, place or person in the world. |
| option | one complete action that the world permits now. |
| good option | an option that is valid and makes sense for this character now. One case can have many good options. |
| acceptable set | all good options for one case. |
| score | the number that the model gives to one option. |
| method | the way we get scores from a model (Section 4). |
| factor | one property of a test case that we can change: list size, option length, and others (Section 4). |
| trait | one part of personality: bravery, greed, pride, kindness, temper. |
| mood | the trait values at this moment. Events change it. |
| momentum | what continues when a character changes its action: its speed, or its anger. |
| belief record | what one character knows or thinks about each thing. The world keeps it exactly. |
| judge | a small model that rates a scene or a choice: "makes sense" and "interesting". |
| rating round | one cycle: the judge chooses a few samples, you rate them, we improve the judge. |

## 4. Workstream A: Which method, and why does Laya get worse?

**Question.** Why does Laya get worse with more options? Is it the method,
the model, or the task?

### The methods

| method | how it works | models |
| --- | --- | --- |
| all together | all options in one input; the model scores them together | Laya (now); Qwen3 with a numbered list |
| one at a time | one input for each option; the model scores that option alone | Laya with a yes/no question; Qwen3; Open-Jev (its normal method) |
| two at a time | compare two options; the winner goes on to the next pair | Laya; Qwen3 |
| read apart | read the situation and each option separately; compare the two readings; option readings are kept and reused | Laya's reading part; a sentence-matching model; the CLM method |
| shortlist, then choose | "read apart" makes a short list of 10; then "one at a time" chooses | combination |

In "one at a time", the model reads the situation one time and keeps that
reading. Each option adds only its own words. Readings of options that come
back often are kept between turns (warm start).

### The models

| model | size | longest input | why |
| --- | --- | --- | --- |
| Laya, English | 421M | 512 word pieces | the model we tested |
| Laya, multilingual | 322M | 1,024 word pieces | the same family with a longer input |
| Von | 395M | not stated | another model of the same type |
| Qwen3-0.6B | 0.6B | 32,000 | general model, one-at-a-time scoring |
| Qwen3-1.7B | 1.7B | 32,000 | the same family, deeper: shows the effect of model size |
| Open-Jev-2B | 2B | 4,096 | trained for choices; one at a time by design |

This table separates the method from the model. Laya English against Laya
multilingual shows the effect of the input length. Qwen3-0.6B against 1.7B
shows the effect of size in one family. Each model runs with several methods.

### The factors

Each test changes one factor. All other factors stay at a fixed middle value.

| factor | values |
| --- | --- |
| list size | 4, 8, 16, 32, 64, 128 |
| option length | short name; name with a short description |
| position of the good options | first, middle, last |
| similarity of the other options | unrelated; same type; nearly the same meaning |
| situation length | short (2 lines); long (a full scene) |
| size of the acceptable set | 1, 2, 4 good options |

### The test cases

Three kinds of cases, because a method can work for one kind and not for
another:

1. Things: "Kael must cut the rope." Which things can he use?
2. People: "A peasant sees a bandit threaten a child." What can the peasant
   do?
3. Places: "Smoke fills the kitchen." Where can the cook go?

**Labels.** The large model labels each option in each case as good,
possible, or bad. You check 50 cases (proposed). If you disagree with
more than 20% (proposed), we fix the labels before we use them.

### What we measure

- How often the chosen option is in the acceptable set.
- How much of the score falls on the acceptable set.
- The rank of the first good option.
- The time for each choice, and the GPU memory.

### How we run it

- One model and one method at a time.
- Before each run: check that the model fits in GPU memory.
- Each run writes each result when it is done. A paused run can continue.
- The heat guard pauses a run when the CPU is too hot.

### What it tells us

| result | meaning |
| --- | --- |
| Laya "one at a time" does as well as Qwen3 | The cause is the method, not the model. |
| Laya multilingual does better than Laya English | The short input is part of the cause. |
| Qwen3-1.7B does much better than 0.6B with the same method | Model size is part of the cause. |
| All methods fail when the other options are nearly the same | The task is hard, not the method. |
| Results depend on position | The model has a position bias. Use "one at a time" or mix the order. |

**Output.** A table of method × model × factor, and a short report: the
method and model to use for each kind of case.

## 5. Workstream B: Research on characters and human simulation

**Question.** Text in the input is a weak way to give a model personality,
history and a room layout. What else exists? Is it useful for us?

**Method.** A short literature review. For each approach, we answer:

- What does it need: data, training, computer power?
- Does it fit the laptop?
- Can we control it? Can it give "cowardly on average, sometimes brave"?
- Was it tested on behavior that must make sense in context?

**Approaches to review:**

| area | examples |
| --- | --- |
| language-model characters | Generative Agents (memory, reflection, plans); simulations of 1,000 real people; "silicon samples" (a model answers as a described person) |
| changing the model's internal numbers | steering directions, persona directions, learned prefix vectors |
| game character AI | utility AI (scores for needs); BDI (beliefs, desires, intentions); OCC emotion model |
| social simulation | Comme il Faut and Prom Week (social rules), Versu (social practices), Ensemble |
| personality models | Big Five, HEXACO (which traits to use) |
| game masters and stories | drama managers (Façade), player models, story sifting (Felt, Winnow) |

**Output.** A table of approaches, and two or three to test in the lab world
(Workstream F).

## 6. Workstream C: The judge, taught in rating rounds

**Question.** How can we rate thousands of scenes when you can rate only a
few?

**The loop:**

1. The large model writes a starting set: short scene summaries and single
   choices, with labels for "makes sense for this character" and
   "interesting". It includes bad and boring examples on purpose. About
   500 (proposed).
2. A small judge model learns from this set on the laptop.
3. The judge chooses about 10 samples where it is least sure, or where it
   disagrees with the large model.
4. You rate these 10 samples on a rating page. The page shows one sample at
   a time.
5. We measure agreement: your ratings against the judge's ratings from
   before step 4.
6. The large model uses your ratings to fix the set: it corrects wrong
   labels and writes more samples like the ones where it was wrong.
7. Go back to step 2.

**Stop rule.** The judge agrees with your new ratings at least 75% of the time
(proposed) in two rounds one after the other.

**Signals that the judge also reads,** from the event log of each run:

- Story patterns: written patterns of interesting events (a reversal, a
  betrayal, a rescue, a lie that is found out, a coward who acts bravely).
- Surprise: how badly a simple "average character" predicts what happened,
  when the events still make sense.
- Chains: how often one character's action changes another character's
  choice.
- Variety: how different the runs are when only the random seed changes.

**The rating page.** A private web page that keeps your ratings. You can use
it on the laptop or a phone. I read the ratings from it. (A small
command-line tool is the other option.)

## 7. Workstream D: How the player acts, and how actions take time

**Question.** What can the player do, and how does the player choose it?

**Selection.**

- The player selects a thing, a person or a place.
- The actions for it are shown in groups of similar actions: move, handle,
  fight, talk, use a skill or magic.
- A group shows only the actions that are possible with the selected thing.
- The player can build a complex action: an action, with what, how (for
  example carefully, or in anger), and what to do next.

**First task: the action catalog.** Write down which kinds of actions the
player needs. Use the scenario list of roadmap 95: fights, stealth, talk,
traps, doors, water, fire, crafting. Put each kind in a group. This is a
design document. You give notes on it before we build anything.

**Actions take time.**

- Each action has a duration.
- While it runs, the rest of the world acts.
- A character can stop its action and choose again, at each turn. It cannot
  do this when it is in the air or cannot act.
- Momentum continues after a change: physical speed, and emotion (anger
  fades slowly).
- The voxel simulation already has part of this: a body that is committed to
  an action, and a body that falls. The "world keeps running" design (the
  job system, and a stop when something important is seen) is the model.

## 8. Workstream E: The test world (lab world, version 2)

Not a puzzle. A closed scene with people and no fixed goal.

- **Scene.** A village inn at night (proposed): main room, kitchen, cellar,
  yard, each with a layout.
- **Characters.** For example: a peasant, a knight, a merchant, a thief, a
  cook. Later a dragon arrives. Each has wants, traits, a mood and a history.
- **Observation.** Each character sees only what is in view. It hears only
  what is near.
- **Belief record.** The world keeps what each character saw, heard or was
  told. Beliefs can be wrong.
- **Actions take time,** and can be stopped, as in Workstream D.
- **Traits change effects.** The world decides how traits and mood change
  an action. Example: anger adds force and takes away accuracy. Fear makes a
  character faster when it runs, and makes it drop things more often. The
  character chooses what to try. The world decides how well it goes.
- **Things made when needed.** The contents of a chest are decided when
  someone opens it (Workstream H).
- **Event log.** All actions and results.

We copy ideas from the old engine (`src/core/mind.py`: traits, seeing deeds,
gossip, choice by draw). We do not import it.

**Check.** 1,000 turns with random characters: no errors, each action is
valid, and each belief agrees with what the character saw, heard or was
told.

## 9. Workstream F: Characters with personality

**Question.** Which method from Workstream B gives behavior that makes sense
in context, with personality and with variation?

- Test the two or three methods chosen in Workstream B. Always include
  "personality as text" as the baseline, and random valid choice.
- Characters draw their action from the scores. They do not always take the
  highest score. So they sometimes do unexpected but valid things.
- Traits have an average and a range. A peasant is cowardly on average, but
  sometimes desperate and brave.

**Measure.**

- The judge's "makes sense" rating.
- Type behavior in ranges that you set. Example (proposed): a peasant acts
  bravely in 10 to 25% of threats, a knight in more than 70%.
- The judge's "interesting" rating, and variety between seeds.

## 10. Workstream G: How characters speak

**The decision.** A choice model cannot write words. How does a character
speak?

| option | how | problem |
| --- | --- | --- |
| a) restricted grammar | The meaning is a speech act and a content. Fixed sentence patterns change it into words. | Can sound stiff. |
| b) line bank | Offline, the large model writes many lines with gaps, for each speech act, content type and personality. In play, the choice model picks one of about 10 lines, and the gaps are filled from the meaning. | The bank must be large enough. |
| c) small language model | It writes words from the meaning. | Speed; it can add facts that are not true. |

In all three options, the listener gets the meaning, not the words. So the
world always knows what was said, and a lie is a meaning that does not agree
with the world.

**Test.** For each option, we measure: variety, fit to the character (judge),
meaning kept (a second model reads the words back), and time. We start with
option b, because it keeps the choice model's task small.

## 11. Workstream H: Deciding effects and making things when needed

This is a large part of what a game master does.

- **Making things (J4).** When a character opens a chest, the world asks:
  what is in it? When someone knocks, who is at the door? A generator makes
  a short list of candidates that fit the place and what the characters
  already know. The choice model picks one, with the story in mind. The
  large model makes the generator's catalog offline, with embeddings.
- **Undefined effects (J3).** When an action has no world rule (magic, a
  strange use of a thing, "rule of cool"), the model picks an effect from
  candidates in the world's effect form: add a mark, change a number, move a
  thing, make a thing, start an event. The world applies only permitted
  effects, inside limits. The old engine's effect system
  (`src/core/effects.py`, `src/core/data/reactions.yaml`) is the model.
- **Check.** The same action in the same situation gives similar results.
  The judge rates "fair" and "fun".

## 12. Order of work

| order | workstream | GPU | depends on |
| --- | --- | --- | --- |
| 1 | A: methods and models (one job at a time) | yes | — |
| 1 | B: literature review | no | — |
| 1 | D: action catalog (design document for your notes) | no | — |
| 2 | E: lab world v2 | no | D |
| 3 | C: judge and rating rounds | small | E (scenes to rate) |
| 4 | F: characters with personality | yes | A, B, C, E |
| 5 | G: speech | yes | A, C |
| 6 | H: effects and things made when needed | yes | A, C, E |
| 7 | move to the voxel simulation | — | own plan |

Items with order 1 can go in parallel, because only A uses the GPU.

## 13. Limits and risks

- **Speed during play.** Your speed rule allows no model call for character
  choices during play. Only the reply of the character that the player
  speaks to is permitted. Question 1 below asks how to handle this.
- **Laptop.** One GPU job at a time. Long runs and training go to your
  machine.
- **Labels from a large model** have their own bias. Your checks in
  Workstream A and the rating rounds in Workstream C control this.
- **Research methods may not work in small models.** Steering, for example,
  is tested mostly in larger models. Workstream F finds this out.
- **The lab world is not the voxel simulation.** A method must be checked
  again after the move.

## 14. Questions for your notes

1. **Speed rule.** Character choices by a model during play break the
   current rule. Which option?
   - a) The model runs only for characters near the player. Others use rules.
   - b) Run the model offline. Put its behavior into fast tables for play.
   - c) Change the rule.
2. **Rating tool.** A private web page, or a command-line tool?
3. **Rating load.** About 10 samples in each round. How many rounds in a
   week?
4. **First scene.** Is the village inn good? Which character types first?
5. **Traits.** Bravery, greed, pride, kindness, temper. Or a standard set
   (Big Five)?
6. **Speech.** Start with the line bank (option b)?
7. The numbers marked "(proposed)".

## 15. Files (code in `lab/`, documents in `docs/`; nothing in `src/` changes)

- `lab/methods/`: Workstream A: each method, the factor tests, the run
  script (one job at a time, GPU memory check, resumable).
- `docs/character-methods-review.md`: Workstream B.
- `docs/action-catalog.md`: Workstream D.
- `lab/world2/`: Workstream E.
- `lab/judge/`: Workstream C: the starting set, the judge, the rating
  rounds.
- `lab/persona/`: Workstream F.
- `lab/speech/`: Workstream G.
- `lab/tools/heat_guard.sh`: the heat guard.

Reused: `lab/calib.py` (`LM.scores`: one-at-a-time scoring), `lab/models.py`
(Laya adapter), `lab/speak.py` (wording and the check for invented names),
`lab/vendor/Open-Jev` (installed, with its own environment).

## 16. How we check the work

- Each workstream writes its results to `lab/results/`, with error ranges.
- For each step, we read 20 samples by hand.
- The voxel simulation tests still pass, because `src/` does not change.
