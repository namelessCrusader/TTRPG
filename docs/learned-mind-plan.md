# Plan: a learned mind for Mark_1 characters (from the Buddy seed)

Status: a plan, written 2026-10-18 at the user's request (decision 7a). It runs after wave 11b. Nothing here is built.

Sources: the user's Buddy brief (`~/Desktop/Fun Stuff/Buddy/seed`), the wave 11b book results (lab/results/books/), and the inn world (lab/world2).

## 1. The question

Can a character choose like a person when its choices come from what it has LIVED, and not from a sentence about its personality?

Two results from wave 11b point here:
- GLiNER reads the scene well. It uses needs (-3.5 points when removed), beliefs (-2.2) and the key fact (+2.4 when placed last).
- GLiNER does not use who the character is. Removing the persona changes nothing (+0.8). Removing the history changes nothing (+0.4). A trait sentence is words to it, not a reason.

The Buddy rule fits this: "nothing pretrained may choose actions; a pretrained model is only the sensory cortex". So the frozen model reads, and values that the character learned from its own life choose.

## 2. Granularity: what we write, and what is learned

The user's point on the stuck bodies (decision 2) and the Buddy rule 2 give one principle: we write the levels BELOW the capability we study, and the capability itself must come from the loop.

| level | who writes it | example |
| --- | --- | --- |
| the world and its facts | we write it (the sim is the authority) | what is possible, what happens, who saw it |
| the menu of acts | we write it (the engine lists the acts the world allows) | "give Oren the coin", "lie to Marta" |
| reflexes, every tick | we write them | balance (a foot over a gap carries no load), flinch, orienting to a loud event |
| drives and their set points | we write them (this is "nature") | hunger, safety, company, standing |
| learning rules | we write them | the delta rule, actor-critic, habit growth |
| which act, when, toward whom | LEARNED from the character's own life | trust in Fenn, the habit of paying debts, fear of the cellar |

This is a deliberate change from Buddy rule 2. Buddy writes only wheel speeds and a buzz. Mark_1 characters need acts like "bribe the guard", and to learn those from motor commands would take far too many trials (the Buddy brief says prior work stalls on this). So the menu acts are our "motor primitives". They are facts that the world allows, not behaviour with a meaning written in.

## 3. What exists already

| Buddy module | what Mark_1 has now |
| --- | --- |
| drives | hunger, thirst, fear, hp, drink (lab/world2/world.py `Person`, `_drift`); needs and physiology in src/voxel |
| sensors | the event log with causes; what each person saw or heard (`perceived`, `pending`) |
| memory of people | liking (`aff`, drawn at random at the start), anger (fades), beliefs with their source event (`bel`), lies |
| the menu | `World.options(p)`, with numeric features for each option (`o["f"]`: gain, theft, witnesses, like, threat, ...) |
| habits | `policy.habit` (written by hand: repeats and undoing pull less) |
| episodic memory | the trace harvest (src/voxel/harvest.py), the kNN datastore idea |
| choosers to compare | `random`, `rules`, `text:M`, `bias:M`, `appraise:M` (lab/world2/policy.py) |

A rules night of the inn takes 0.19 s (300 turns, 5 people). So 1,000 nights take about 3 minutes on the CPU. A lifetime is cheap.

## 4. The mind

One mind for each character. Each module has its own learning rule and its own speed. All are small (numpy, linear or tabular), and learn online, one event at a time.

| module | what it learns | rule | speed |
| --- | --- | --- | --- |
| H. homeostasis | nothing (written): each drive's error gives the reward and biases the act | reward = the drop in weighted drive error | — |
| V. valence | how good or bad each person, thing and place is, from what followed it | delta rule on the drive change after the cue (Pavlovian) | fast |
| A. action selection | which act pays in which situation | actor-critic on the reward-prediction error, over the menu features | medium |
| K. habit | what this character tends to do here | strength grows with each repeat, fades with disuse | slow |
| W. world model | what an act does to my drives | one-step prediction, delta rule | medium |
| P. partner models, ONE PER PERSON | how THIS person answers my acts toward them (pays back, hits back, lies, tells others) | online logistic regression per partner and act kind | fast |
| E. episodic memory | what happened, where, with whom, and what it did to me | one-shot store; nearest-neighbour lookup; replay into A and K at the end of each night ("sleep") | one shot |
| C. curiosity | where my models still learn | intrinsic bonus from the learning progress of W and P; habituation lowers it | medium |

The choice: softmax over the menu of (actor score + habit + valence of the target + a one-step lookahead with W and P). The frozen reader (GLiNER, or the engine's own option features) only turns the scene and the options into features.

ORDER comes free. Each event updates the values when it happens, so "betrayed by Fenn, then saved by Fenn" leaves a different trust in Fenn than "saved, then betrayed". The book work could build only 3 such order pairs; here we can make as many as we want.

PERSONALITY has two parts:
- nature: the drive weights and set points (a greedy character feels the loss of coin more), written per character type;
- nurture: everything in V, A, K, P and E, which comes from the character's life.

## 5. Drives and reward

The inn has hunger, thirst, fear and hp. The social side needs more. Proposed drives, each with a set point and a decay:
- energy (hunger, thirst);
- safety (fear, hp);
- company (time near people who are kind to me; it decays when alone or shunned);
- standing (how others treat me: praise, insults, trust shown, being searched or accused);
- wealth, for the types that care (merchant, thief): what I hold of value.

Reward = the drop in the weighted error of the drives (Keramati and Gutkin, homeostatic RL). There is no designer reward like "+1 for a kind act". Other people matter to the mind only through what they do to its drives (Buddy rule 3).

## 6. Changes to the world (needed first)

A learner becomes human-like only if the world's consequences are. Today a lie is rarely found out, so a learner would learn that lying pays. Before the mind, the inn needs:
1. A LIFETIME: the same cast, night after night (about 1,000 nights). The room and the things reset each night. The minds, and what people remember of each other, stay.
2. CONSEQUENCES that travel: gossip (a person tells others what it saw); reputation (liking changes for those who hear); people who refuse to trade with, or watch, someone they distrust; lies found out by checking (the existing `_contradict`, more often).
3. SCRIPTED PARTNERS for the tests (Buddy's P0-P3), each rule written down and reported:
   - P-kind: pays back a gift with chance p;
   - P-hard: hits back when insulted with chance p;
   - P-switch: one of these, with its rule changed at set nights (the betrayal tests);
   - P-deaf: never answers (the control: nothing should be learned about it).

## 7. Null models (Buddy rule 6: no claim without one)

Every result is compared on the same world, the same drives and the same compute with:
- N1, a bandit over the acts (contextual, linear on the same features);
- N2, one monolithic RL policy (PPO) on the same features and reward. stable-baselines3 is not in the lab venv; it must be installed (a small, CPU-only package);
- the existing choosers: `rules` and `bias:gdecide`.

A capability counts only if the mind shows it and the nulls do not, with seeds stated.

## 8. Tests

| test | what we do | what counts as a pass |
| --- | --- | --- |
| L0 keeps itself well | a lifetime in the plain inn | the drive error stays as low as `rules` or lower, and far below random |
| L1 a self from a life | clones: the same nature, different lives (different partners) | stable, different behaviour after the lives part, explained by what each lived; the nulls drift or do not part |
| L2 one partner, one way | a P-kind and a P-hard partner in the same lifetime | the mind treats each its own way, and holds both without mixing them up |
| L3 betrayal | P-switch: a trusted partner starts to betray | the mind's trust and acts follow within a few nights; the SECOND switch is learned faster (learning to learn) |
| L4 drives, not a designer | the same mind with a designer reward instead | the behaviour that depends on context appears only with the drives (the Buddy E4 hypothesis) |
| L5 order | twins who live the same events in a different order | they end with different trust and different acts toward that person; a summed-history mind cannot |
| L6 people judge | nights of the learned cast against `rules` and `bias:gdecide`, the same seeds | the user's blind side-by-side rating (plan v5's rating page), with the marked lines for each |

Controls against fooling ourselves (Buddy's Clever Hans list): shuffle the partners' answers; break the contingency in the middle of a run; P-deaf must show no learned behaviour toward it.

## 9. Order of work and cost

1. The world changes in section 6 (lab/world2, CPU). Tests in lab.world2.test_world.
2. The mind, module by module, in a new lab/mindL/ (lab never imports src). L0 after H, A and K; then V, P, E, W, C. Each module is added with an ablation run (one change at a time).
3. The nulls N1 and N2.
4. L1 to L5: CPU only. A lifetime of 1,000 nights is minutes. A sweep of 10 seeds x 5 tests x 4 choosers is a few hours, under the lock, the cap and the cooling gate.
5. L6: a short GPU run for `bias:gdecide`, under the GPU lock and the heat guard; the user rates.
6. Only if the mind wins: a port to the voxel world through the mind link (src/voxel/mindlink.py), with the same tests in voxel scenes.

GPU: only for the gdecide comparison and, if used, the frozen reader's features (cached once). No fine-tuning of any pretrained model.

## 10. Risks

- TOO SLOW TO LEARN. This is why prior work stalls. Our answer is the high-level menu, cheap nights, and episodic memory for one-shot learning. If L0 needs more than about 10,000 nights, the plan is wrong.
- THE WORLD TEACHES THE WRONG LESSON. If the consequences are thin, the mind learns to lie and steal. Section 6 comes first for this reason, and L6 (people judge) is the final test.
- A NULL MATCHES IT. If the bandit or PPO does as well, the modules add nothing. That is a real result, and we report it.
- IT DOES NOT CARRY TO NEW SITUATIONS. A mind that lived only in the inn cannot answer a book case about a dungeon. The book cases keep judging the READER; this plan judges the CHOOSER in the sim.

## 11. Decisions for the user

1. The drives: the five in section 5, or a different set?
2. One lifetime = about 1,000 nights with the room reset each night and the minds kept. Agree?
3. Install stable-baselines3 in the lab venv for the PPO null?
4. The frozen reader: start with the engine's own option features (exact, free), and add GLiNER features as a second step? (Recommended.)
5. When: right after wave 11b, or after the next engineering wave?
