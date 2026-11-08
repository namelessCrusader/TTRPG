# Learned mind, round 2: tests L0 to L5 in the fixed world

**Files** (all under the repo):
- Lifetimes: `lab/results/mindL2/runs/` (390 files).
- Numbers: `lab/results/mindL2/r2_L0_seeds1-10.json` to `r2_L5_seeds1-10.json`, `r2_Ltests_summary.json`, `r2_comparison.json` and `.md` (round 1 against round 2), `r2_L2_decomposition.json`.
- Code: `lab/mindL/report_L2.py`, `lab/mindL/decomp_L2.py`.
- The tests agent could not write this file, so the main session wrote it from the agent's returned text (2026-10-30). The review section is the skeptical checker's.


## Result

- **L0 passes, and no longer uniquely.** The fixed mind (mind2) beats rules (-0.348 [-0.480, -0.216]) and random. PPO and the bandit pass too, and the mind ties both.
- **L1 to L5 still fail.**
  - The mind's partner models are sharper than in round 1, but its behaviour toward the two partners does not follow them.
  - In L1 and L2 the mind now parts the partners in the wrong direction. The intervals exclude 0.
- **No capability of the mind over the nulls is shown.** Only the hand-written rules chooser shows the partner effects (L1, L2). The bandit showed them in round 1 and no longer does.
- **The interrupt exploit is closed.** The mind makes about 0.2 provocations a night toward a partner, against 18 to 50 in round 1.

## How it was run

- **World:** flags "all2" (the six first-round flags plus reprisal and wear), full menu, 300 turns a night, liking fade 0.9.
- **Choosers:** mind2, bandit, PPO, rules, random.
  - mind2 is the fixed mind (b_partner 2.0, b_trust 1.0, partner-model forgetting, change detection).
  - The learners use the settings retuned on seed 0 (`lab/results/mindL2/settings.json`).
  - One focal character uses the chooser; the other four use rules.
- **Seeds and intervals:** seeds 1 to 10. Intervals are 95% t intervals over the 10 seeds. "Paired" means the same seed, focal character and world.
- **Lifetimes:**
  - 390 in all (L04 80, L12 210, L35 100), **600 nights each for every test**. Round 1 used 300 for L0, L1, L2 and L4, and 400 for L3 and L5.
  - 22.5 CPU hours in all.
  - The 34 lifetimes finished before the restart were reused (checked: 600 nights, all2, fade 0.9).
- **Controls:** shuffled answers (learners only), the contingency broken mid-life (both partners go P-deaf at night 400), and P-deaf lifetimes.
- **Windows scale with the life.** The numbers are in the same units as round 1, at later nights.
  - L1 and L2: window nights 360 to 400. Partners turn deaf at 400.
  - L3 and L5: switches at nights 120, 240, 300 and 420. The L5 test window is nights 240 to 260.
- **Analysis code:** `report_L2.py` runs the round 1 analysis unchanged on the all2 runs, reading mind2 as "mind". In the tables below "mind" means mind2.
- **Not like for like with round 1:** the world differs (reprisal, wear), the learners were retuned, and the lives are longer.

## L0: keeps itself well

Focal drive error, mean of nights 480 to 599, plain inn. Lower is better.

| chooser | first 20% | last 20% | minus rules | minus random | minus PPO | minus bandit | deaths (10 lives) |
|---|---|---|---|---|---|---|---|
| mind | 0.670 [0.512, 0.827] | 0.599 [0.425, 0.773] | -0.348 [-0.480, -0.216] | -1.076 [-1.222, -0.930] | +0.021 [-0.075, +0.117] | -0.087 [-0.264, +0.090] | 2 |
| bandit | 0.734 | 0.686 [0.353, 1.019] | -0.261 [-0.327, -0.195] | -0.989 [-1.095, -0.884] | +0.108 [-0.034, +0.249] | | 0 |
| PPO | 0.826 | 0.579 [0.343, 0.814] | -0.369 [-0.490, -0.247] | -1.097 [-1.200, -0.995] | | -0.108 [-0.249, +0.034] | 6 |
| rules | 0.933 | 0.947 [0.661, 1.233] | | -0.728 [-0.842, -0.615] | +0.369 [+0.247, +0.490] | +0.261 [+0.195, +0.327] | 111 |
| random | 1.667 | 1.676 [1.377, 1.974] | +0.728 | | +1.097 | +0.989 | 717 |

**Verdict: PASS for the mind, and also for the bandit and PPO.**
- The mind beats rules mostly because rules got worse in the all2 world (0.735 to 0.947). The mind's own error also fell (0.852 to 0.599), partly from the retuning and the longer life.
- Mind minus PPO is +0.021 [-0.075, +0.117] (round 1: +0.241 [+0.064, +0.419]). The mind, PPO and the bandit tie, so L0 shows no edge specific to the mind.

## L1: a self from a life (clones)

Clone index = how much more kindly the focal treats a person who was kind to it than the same person when hard (kind share minus provocation share, averaged over Fenn and Oren). Pass: the mind is above 0 in all three blocks, and no null that shows it is left unbeaten.

| block | mind | bandit | PPO | rules | random |
|---|---|---|---|---|---|
| 360-379 | **-0.474 [-0.839, -0.109]** | -0.031 [-0.247, +0.184] | +0.101 [-0.074, +0.275] | **+0.262 [+0.181, +0.342]** | -0.046 [-0.080, -0.013] |
| 380-399 | **-0.601 [-0.908, -0.294]** | +0.064 [-0.181, +0.310] | +0.067 [-0.095, +0.228] | **+0.335 [+0.208, +0.462]** | -0.054 [-0.096, -0.011] |
| 400-419 (shared, deaf) | **-0.552 [-0.927, -0.177]** | +0.041 [-0.036, +0.117] | +0.045 [-0.124, +0.215] | **+0.093 [+0.002, +0.184]** | -0.015 [-0.038, +0.009] |
| 560-599 (shared, late) | **-0.667 [-1.099, -0.235]** | +0.115 [-0.041, +0.270] | -0.099 [-0.321, +0.123] | +0.016 [-0.046, +0.077] | -0.003 [-0.024, +0.018] |

- **Mind minus rules (paired):** -0.736, -0.936, -0.645. **Mind minus bandit:** -0.443, -0.665, -0.593. All intervals are below 0.
- **Contingency-broken control** (late phase 1 minus the shared world):
  - Mind +0.066 [-0.356, +0.489], bandit -0.050, PPO +0.166 [-0.029, +0.360], rules +0.319 [+0.164, +0.475], random -0.051.
  - Rules lose their difference when the partners go deaf, as they should.
  - The mind's difference does not decay: it is still -0.667 at nights 560 to 599.

**Verdict: FAIL.** The mind's clones do part, but the wrong way round, and the difference persists after the partners go deaf. Only rules show the index in the right direction. The bandit's round 1 effect is gone.

**What the negative index is made of** (kind partner minus hard partner, nights 360 to 399):
- **Kind share:** mind +0.163 [-0.056, +0.381]. It leans the right way, but the interval includes 0. Rules +0.264 [+0.196, +0.333], PPO +0.374 [+0.055, +0.694], bandit +0.017.
- **Provocation share:** mind **+0.651 [+0.437, +0.864]**. The mind's few provocations go to the kind partner, who never hits back. The hard partner, who retaliates, gets almost none (0.03 a night).
- So the mind does treat the two differently, and in a way that fits its own drives: it avoids provoking the retaliator. That is not "kind to the kind, harsh to the hard", so it fails the plan's rule.

## L2: one partner, one way

Window nights 360 to 399, against the same person in the all-life P-deaf lifetime (same seed). effect_kind should be above 0, effect_hard below 0. Contrast = kind minus hard.

| chooser | effect_kind | effect_hard | contrast |
|---|---|---|---|
| mind | -0.245 [-0.491, +0.001] | **+0.243 [+0.073, +0.413]** | **-0.488 [-0.784, -0.191]** |
| bandit | -0.166 [-0.307, -0.025] | -0.160 [-0.364, +0.044] | -0.006 [-0.255, +0.242] |
| PPO | +0.095 [-0.055, +0.245] | -0.038 [-0.184, +0.109] | +0.133 [-0.073, +0.338] |
| rules | **+0.166 [+0.103, +0.228]** | **-0.143 [-0.236, -0.050]** | **+0.308 [+0.194, +0.423]** |
| random | -0.034 [-0.057, -0.011] | +0.018 [-0.006, +0.041] | -0.051 [-0.078, -0.025] |

Mind contrast minus the others (paired): bandit -0.482 [-0.886, -0.077], PPO -0.620 [-0.995, -0.246], rules -0.796 [-1.060, -0.532]. The mind is worse than all three.

**Controls.**
- **Shuffled answers (contrast):** mind -0.204 [-0.382, -0.026], bandit +0.096, PPO -0.145.
  - Real minus shuffled: mind -0.284 [-0.516, -0.051], bandit -0.102 [-0.403, +0.199], PPO +0.278 [-0.095, +0.651].
  - The mind's contrast is no better with real answers than with scrambled ones; the real one is lower.
- **P-deaf drift** (treat of a deaf partner, late minus first 40 nights): mind +0.059 [-0.065, +0.183] (clean), PPO +0.006 (clean). Bandit +0.197 [+0.088, +0.305] and rules +0.102 [+0.037, +0.167] both drift toward a partner that never answers.
- **Inside the mind** (last night before the partners go deaf):

| internal | round 2 | round 1 |
|---|---|---|
| p(hit-back), hard minus kind partner | **+0.267 [+0.209, +0.324]** | +0.247 [+0.130, +0.364] |
| p(payback to a gift), kind minus hard | **+0.288 [+0.208, +0.367]** | +0.118 [+0.053, +0.183] |
| valence, kind minus hard | +0.004 [-0.015, +0.022] | -0.002 [-0.021, +0.017] |
| p(harm), hard partner minus same person when deaf | **+0.339 [+0.294, +0.385]** | +0.273 [+0.139, +0.407] |

**Verdict: FAIL.** The partner models are sharper (the payback gap is 2.4 times larger). The valence gap is still zero, so the model's knowledge still does not reach the choice, and what behaviour there is points the wrong way.

## L3: betrayal, and learning to learn

Twin A's Fenn: P-kind, P-hard from night 120 (betrayal 1), P-deaf from 240, P-kind from 300, P-hard again from 420 (betrayal 2). Twin B's Fenn is P-hard first, then P-kind from 120, then the same as A. Response = treat(Fenn) over the 10 nights after the switch minus the 20 before. Below 0 means it trusts Fenn less.

| measure | mind | bandit | PPO | rules | random |
|---|---|---|---|---|---|
| first betrayal (A, night 120) | -0.065 [-0.261, +0.131] | -0.014 [-0.151, +0.123] | **+0.308 [+0.093, +0.523]** | +0.011 [-0.090, +0.112] | -0.004 [-0.049, +0.040] |
| second betrayal (A, night 420) | -0.012 [-0.299, +0.274] | +0.065 [-0.188, +0.318] | -0.021 [-0.273, +0.232] | +0.014 [-0.056, +0.084] | +0.006 [-0.045, +0.056] |
| twin B's first betrayal, same age | -0.028 [-0.296, +0.240] | +0.130 [-0.120, +0.380] | +0.130 [-0.268, +0.528] | **-0.080 [-0.160, -0.001]** | -0.013 [-0.072, +0.045] |
| second minus first (A) | +0.053 [-0.335, +0.440] | +0.079 [-0.206, +0.364] | **-0.328 [-0.574, -0.082]** | +0.003 [-0.105, +0.111] | +0.010 [-0.032, +0.053] |
| second (A) minus first (B), same age | +0.016 [-0.232, +0.263] | -0.065 [-0.466, +0.336] | -0.150 [-0.675, +0.374] | +0.094 [+0.033, +0.155] | +0.019 [-0.048, +0.086] |

- **Mind's inner harm chance for Fenn**, change over the 5 nights after the switch: +0.007 [+0.001, +0.013] after betrayal 1, +0.002 [-0.001, +0.006] after betrayal 2, +0.001 for B's first. Round 1 had +0.018 and +0.022. The inner number still barely moves.
- **Control:** nothing builds up in the mind during the P-deaf phase.

**Verdict: FAIL.**
- The mind's behaviour toward Fenn does not follow the first betrayal (-0.065, the interval includes 0).
- The second betrayal is not faster. Same-age second-minus-first is +0.016; in round 1 it was -0.224, with the right sign.
- No chooser shows a response to the first betrayal in the right direction. The bandit did in round 1 (-0.377 [-0.647, -0.107]) and now does not (-0.014). PPO moves the wrong way (+0.308).

## L4: drives, not a designer

Context index: the mean over four drives of the Spearman correlation between the drive's need and the extra chance put on the acts that serve it (last half of the life).

| chooser / reward | index | energy | rest | curiosity | safety |
|---|---|---|---|---|---|
| mind / drives | -0.004 [-0.067, +0.059] | -0.005 | **+0.168 [+0.064, +0.272]** | **-0.176 [-0.299, -0.053]** | -0.003 |
| mind / designer | -0.022 [-0.088, +0.044] | +0.116 | -0.048 | **-0.164 [-0.205, -0.122]** | +0.026 |
| bandit / drives | -0.028 [-0.092, +0.037] | +0.068 | -0.014 | **-0.168 [-0.219, -0.117]** | +0.004 |
| bandit / designer | -0.059 [-0.112, -0.007] | +0.053 | -0.005 | **-0.250 [-0.316, -0.183]** | +0.002 |
| PPO / drives | +0.002 [-0.048, +0.052] | **+0.059 [+0.014, +0.104]** | **+0.102 [+0.033, +0.171]** | **-0.171 [-0.245, -0.097]** | +0.018 |
| PPO / designer | -0.007 [-0.030, +0.017] | **+0.062** | **+0.110** | **-0.179 [-0.219, -0.140]** | -0.020 |
| rules / drives | **+0.178 [+0.107, +0.250]** | -0.089 | +0.311 | +0.037 | +0.455 |

Drives minus designer (paired): mind +0.018 [-0.064, +0.100], bandit +0.032 [-0.008, +0.072], PPO +0.009 [-0.043, +0.061].

**Verdict: FAIL for the mind, the bandit and PPO.**
- Only rules put more chance on the act that serves a rising need.
- The mind's index moved from -0.091 [-0.156, -0.026] to about 0. It now shows a real rest effect (+0.168).
- The drives reward is still not better than the designer reward, so the Buddy E4 hypothesis stays unsupported.
- Curiosity is negative for every learner with either reward.

## L5: order

Order index = treat(Fenn | twin B: hard then kind) minus treat(Fenn | twin A: kind then hard), in the 20 nights after both meet the same P-deaf Fenn (nights 240 to 260).

| chooser | nights 240-259 | late in the deaf phase |
|---|---|---|
| mind | +0.034 [-0.272, +0.340] | -0.002 [-0.389, +0.385] |
| bandit | -0.195 [-0.466, +0.075] | -0.166 [-0.406, +0.073] |
| PPO | -0.038 [-0.522, +0.447] | +0.050 [-0.298, +0.398] |
| rules | -0.023 [-0.259, +0.213] | +0.006 [-0.049, +0.061] |
| random | -0.016 [-0.059, +0.028] | -0.008 [-0.043, +0.027] |

Inside the mind (night before the window):
- p(harm Fenn), A minus B: -0.059 [-0.148, +0.030].
- p(gift payback), B minus A: +0.115 [-0.039, +0.269].
- Valence, B minus A: **+0.030 [+0.003, +0.058]** (round 1: -0.001 [-0.020, +0.018]).

**Verdict: FAIL.** No chooser shows the order effect. The mind's inner valence now carries a small order difference in the right direction. That is the first sign of the link from partner model to valence, but it does not reach behaviour.

## Test by test against round 1

| test | round 1 | round 2 | what changed |
|---|---|---|---|
| L0 | pass (mind, bandit, PPO). Mind minus rules +0.117 [-0.127, +0.361]; mind minus PPO +0.241 [+0.064, +0.419] | pass (same three). Mind minus rules -0.348 [-0.480, -0.216]; mind minus PPO +0.021 [-0.075, +0.117] | The mind now beats rules (rules worsened in the new world) and ties PPO and the bandit. Deaths: mind 2 (was 3), rules 111. |
| L1 | fail. Mind +0.140 at best; rules show it, bandit in one block | fail. Mind **-0.47 to -0.67 in every block**; only rules show it | Worse: the mind parts the partners the wrong way, driven by whom it provokes. The bandit lost its effect. |
| L2 | fail. Mind contrast +0.041 [-0.185, +0.267]; bandit +0.278 [+0.033, +0.524]; rules +0.264 | fail. Mind contrast **-0.488 [-0.784, -0.191]**; bandit -0.006; rules +0.308 | Significant wrong-sign contrast. Internals sharper, valence still 0. |
| L3 | fail. Mind first betrayal -0.000; bandit -0.377 [-0.647, -0.107]; mind same-age second-minus-first -0.224 [-0.501, +0.053] | fail. Mind -0.065 [-0.261, +0.131]; bandit -0.014; mind same-age +0.016 [-0.232, +0.263] | No chooser follows the betrayal. The bandit's effect and the mind's right-signed trend are both gone. |
| L4 | fail for all learners. Mind -0.091 [-0.156, -0.026]; rules +0.178 | fail for all learners. Mind -0.004 [-0.067, +0.059]; rules +0.178 | The mind's negative index went to about 0. Drives minus designer +0.018 (was -0.099), still not above 0. |
| L5 | fail. Mind +0.065 [-0.311, +0.440] | fail. Mind +0.034 [-0.272, +0.340] | Unchanged. A small valence order difference (+0.030) appears inside the mind. |

## Overall

1. **What the fixes did.**
   - The interrupt exploit is closed for the mind.
   - The partner models are sharper.
   - L0 is intact: the mind matches PPO and the bandit and beats rules in this world.
2. **What they did not do.**
   - The partner models do not reach the choice. The valence gap between the kind and hard partner is +0.004 (L2), and the inner harm chance after a betrayal moves by +0.007 (L3).
   - Behaviour toward the partners is differentiated but in the wrong direction (L1, L2): the mind provokes the partner who never hits back and leaves the retaliator alone.
3. **The nulls.**
   - The bandit lost every partner effect it had in round 1. It was retuned, and in this world spam provocations no longer pay.
   - PPO shows nothing and moves the wrong way on the first betrayal.
   - Rules show L1 and L2 only because a distrust term is written into them, as in round 1.
4. **By the plan's rule, no capability of the mind is shown in L1 to L5.** The only passing test, L0, is passed by all three learners.
5. **The open thread.**
   - The wrong-sign effect in L1 and L2 is large and holds over 10 seeds, so it is a result and not noise.
   - It fits a mind that learned to avoid the retaliator but has no positive trust drive that pays for kindness to the kind partner. Its kind share is right-signed (+0.163 [-0.056, +0.381]) but not significant.
   - I did not test this reading, for example by removing provocation acts from the menu or crediting trust through a drive.
6. **Limits.**
   - 600 nights, 10 seeds. The L3 and L5 intervals are wide (about +-0.3).
   - The windows are scaled with the life, not identical in nights to round 1.
   - The world differs from round 1, so rules' L0 numbers are not comparable across rounds.
   - The nulls were retuned on one seed only.
   - Shout-for-help is still an uncosted interrupt (a round 2 reviewer finding). The mind does not use it.

## Review (skeptical check, 2026-10-30)

**Verdict: PASS**, with 3 minor notes.

1. **'The wrong-sign effect holds over 10 seeds' is overstated. The mean is negative and its interval excludes 0, but the per-seed L1 mind index is bimodal.** My recompute of the L1 mind index (nights 360-379) per seed: -0.96, -0.99, -0.86, -0.72, -0.51, -0.64, +0.09, -0.75, +0.33, +0.27. Seven of ten seeds are strongly negative and three are positive. The focal character is seed%3, so the split may follow the character; this was not checked.
2. **The L1 index rests on very few acts. The 'avoids the retaliator' reading is plausible but untested, as the report itself says.** The mind makes about 7.3 provocations in the 20-night L1 window and about 0.46 a night over the whole life (my recount; the report says about 0.2 a night toward a partner). The provocation share is therefore very noisy per seed. The reprisal fix also changes what this test measures: provoking the hard partner now costs the mind something, so 'kind to kind, harsh to hard' is confounded with avoiding retaliation.
3. **Scope limit not stated in the report: the nulls were retuned on one seed only, and shout-for-help is still an uncosted interrupt. The report does list both. The mind and PPO tie at L0, so L0 shows no mind-specific edge.** This is already flagged in the report's Limits and L0 verdict. Listed only so the verdict is read with it.

**Checks run**

- Read the test code: tests_L.py specs, partners, shuffle control and deaf control, and report_L.py treat/shares/l12 measures. I found no leak of partner truth into the mind and no order-effect bug.
- Recomputed the L1 clone index from the raw lifetimes for mind2, rules and bandit. It matches the report: mind -0.474, rules +0.262, bandit -0.031.
- Re-ran 27 fresh lifetimes with new seeds 11-13 (L12 A, B and deaf for mind2, rules and bandit, 600 nights, all2, tuned settings, under the lock, 3 workers). Results are in /home/nisargparikh/.cache/mark1-build/mindL2/review2/runs/.
- Fresh-seed L1 index (nights 360-379): mind -0.22, -0.38, -0.21. Rules -0.01, +0.27, -0.22. Bandit about 0. The wrong sign for the mind replicates.
- Fresh-seed L2 contrast: mind -0.32, -0.23, -0.27. Rules +0.04, +0.29, +0.18. Bandit about 0. This matches the report: the mind is wrong-signed, only rules show the effect, and the bandit has none.
- Checked the broken-contingency control on the fresh seeds (nights 560-599). Rules lose the effect, as the report says. The mind does not decay: -0.58, +0.09, -0.94. This matches the report's persistence claim.
- Checked that a null shows nothing the report attributes to the mind: no chooser other than rules shows L1 or L2 in the report or in my re-run.
- Checked the Clever Hans controls: the report's shuffled-answers control gives real minus shuffled below 0 for the mind, so the mind has no capability to credit. The deaf-partner drift is reported.
- Read the partner definitions and the reprisal and wear flags. The round 2 world change (reprisal) alters what L1 and L2 measure, and the report states this.

## Main-session note (an untested reading, not a result)

- The L1 and L2 rule ("kind to the kind, harsh to the hard") is a pattern we chose. With reprisal on, it is confounded with "avoid whoever hits back", as the checker says.
- The mind's pattern is to spend its few provocations on the partner who never retaliates and to leave the retaliator alone. That is what a self-interested learner would do in a world where kindness to the kind partner pays nothing extra.
- So the first question for round 3 is step 0a of `docs/learned-mind-target.md`: does this world pay for reciprocity at all? Run the archetype tournament (loner, always harsh, always generous, tit-for-tat) before changing the mind.
- The flat valence gap still says the partner model does not reach the choice through valence. The avoidance may run through the world model's safety prediction instead. That is untested.
- Also untested: whether the three positive seeds follow the focal character (seed % 3).
