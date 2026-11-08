# Learned mind: tests L0 to L5

## Result

- **L0 passes.** The mind keeps itself fed about as well as rules and far better than random, but PPO and the bandit pass too, and PPO is lower than the mind.
- **L1 to L5 fail.** The mind's partner models do learn who is kind and who is hard (the numbers inside the mind show it). What the character does toward each partner does not follow.
- **No capability of the mind over the nulls was shown.** The bandit tracks the partners better than the mind does. Rules do too, but only because a distrust term is written into them.

## How it was run

- **Seeds and intervals:**
  - Seeds 1 to 10 for every chooser and test (seed 0 was the tuning seed).
  - A seed fixes the world draws, the focal character and the partners' random draws.
  - Intervals are 95% t intervals over the 10 seeds, because nights within one life are not independent.
  - "Paired" means same seed, same focal character, same world.
- **Choosers:** mind (full AKVWPEC), bandit, PPO, rules, random. One focal character uses the chooser and the other four use rules.
- **Settings:** the learners use the settings tuned on seed 0 and were not retuned in the honest world. The world is flags "all" including `known`, with the full menu and 300 turns a night. The shout-and-shove ban was not used.
- **Lifetime length: 300 nights for L0, L1, L2 and L4, and 400 nights for L3 and L5. Not 1,000.**
  - The sweep took about 3.3 hours of wall time on 5 workers (14.2 CPU hours).
  - 1,000 nights would have made each sweep run well past 2 hours.
  - PPO's error roughly halves between the first and last fifth of its 300 nights, so a null result here means "not shown in 300 nights", not "cannot be learned".
- **Code edit:** the shuffled-answers control was run only for the three learners (mind, bandit, PPO), since rules and random do not learn. That made L12 210 lifetimes instead of 250, via a one-line change in `tests_L.py`.
- **Resuming after the session limit:**
  - The 16 fade-0.9 L04 lifetimes already on disk were checked (fade field 0.9, 300 nights each) and kept.
  - The other 64 of the 80 L04 lifetimes were run, then L12 and L35.
  - The earlier fade-1.0 L04 run was kept as a sensitivity check.

### The fade: decided 0.9

Overnight, each liking moves 10% of the way back to its night-0 draw.

- **Why:** with no fade, liking saturates at plus or minus 1 within a few dozen nights.
  - In the partial fade-1.0 partner lifetime I inspected (mind and PPO, seed 1, Ada), Fenn sat at +1.0 and Oren at -1.0 from night 10 to night 299, whatever they did.
  - A pinned number cannot change when a partner betrays, so L3 and L5 could not have shown anything for any chooser.
  - With the fade the world step measured mean liking of -0.18 in nights 100 to 199, against -0.30 to -0.40 without it.
- **Cost:** the fade applies to every chooser, but it makes forgetting cheap, which shortens how long a betrayal is remembered.
- **Sensitivity:** L0 and L4 were also run at fade 1.0 (80 lifetimes) and give the same picture.
  - L0, mind minus rules: +0.138 [-0.124, +0.401] at fade 1.0, against +0.117 at 0.9.
  - L4, mind index: -0.092 at fade 1.0, against -0.091 at 0.9.
  - The partner tests were not run at fade 1.0.

### Warning for every test with the mind

The interrupt exploit from the mind step is still in the world and the mind still uses it.

- **Provocations a night:** the mind makes about 18 to 50, the bandit 30 to 50, PPO 1 to 2 and rules about 0.01.
- **Effect on the tests:** the P-hard partner hits the mind back about 0.8 times a night, and "who gets the provocations" is spread thin over a huge number of acts.
- **Status:** I did not fix it, because the mind step left that to you.

### Treatment measure for L1 to L5

treat(X) over a block of nights is the share of the focal's kind acts that go to X, minus the share of its insults, blows, shoves and threats that go to X. It runs from -1 to +1.

---

## L0: keeps itself well

**Measured.** Focal drive error (the homeostatic error over the seven drives), mean of nights 240 to 299, 10 seeds, plain inn. Lower is better.

| chooser | first 20% | last 20% | minus rules | minus random | minus PPO | minus bandit | deaths (10 lives) |
|---|---|---|---|---|---|---|---|
| mind | 0.842 [0.653, 1.031] | 0.852 [0.642, 1.061] | +0.117 [-0.127, +0.361] | -0.828 [-1.076, -0.580] | +0.241 [+0.064, +0.419] | -0.077 [-0.201, +0.047] | 3 |
| bandit | 1.109 [0.958, 1.261] | 0.929 [0.807, 1.050] | +0.194 [-0.087, +0.474] | -0.751 [-1.025, -0.476] | +0.318 [+0.130, +0.507] | | 43 |
| PPO | 1.425 [1.086, 1.765] | 0.610 [0.412, 0.809] | -0.125 [-0.270, +0.021] | -1.069 [-1.209, -0.929] | | -0.318 [-0.507, -0.130] | 83 |
| rules | 0.758 [0.470, 1.046] | 0.735 [0.435, 1.035] | | -0.945 [-1.051, -0.838] | +0.125 [-0.021, +0.270] | -0.194 [-0.474, +0.087] | 110 |
| random | 1.712 [1.399, 2.025] | 1.679 [1.366, 1.993] | +0.945 [+0.838, +1.051] | | +1.069 [+0.929, +1.209] | +0.751 [+0.476, +1.025] | 808 |

**Rule.** Error as low as rules or lower, and far below random. The mind passes: its gap to rules includes 0 and its gap to random is -0.83. The bandit and PPO pass too.

**In plain words.**
- **Looks after itself:** the mind does this like the hand-written rules, and much better than random.
- **No edge:** PPO is lower than the mind (mind minus PPO +0.24, interval above 0), and the mind and the bandit tie.
- **Does not improve:** the mind starts best of the learners (0.84) but stays flat over 300 nights, while PPO falls from 1.43 to 0.61.
- **Deaths:** the mind dies far less often (3 against 110 for rules and 83 for PPO).

This agrees with the mind step: no module adds measurable value at L0.

## L1: a self from a life (clones)

**Measured.** Same seed and nature, lived twice with the partner roles swapped.

- Life A: Fenn is P-kind(0.9) and Oren is P-hard(0.9). Life B is the reverse.
- Both partners go P-deaf at night 200, which is the shared world and also breaks the contingency.
- Clone index = how much more kindly the focal treats a person who was kind to it than the same person when hard, averaged over Fenn and Oren.
- Pass: the mind shows an index above 0 in all three blocks, and no null that shows it is left unbeaten.

| block | mind | bandit | PPO | rules | random |
|---|---|---|---|---|---|
| nights 160-179 | -0.047 [-0.301, +0.207] | +0.162 [-0.102, +0.425] | -0.114 [-0.231, +0.003] | **+0.259 [+0.182, +0.335]** | -0.026 [-0.067, +0.015] |
| nights 180-199 | +0.140 [-0.104, +0.383] | **+0.357 [+0.161, +0.554]** | -0.130 [-0.226, -0.034] | **+0.265 [+0.170, +0.360]** | -0.013 [-0.044, +0.018] |
| nights 200-219 (shared) | +0.173 [-0.150, +0.496] | +0.088 [-0.069, +0.246] | -0.052 [-0.209, +0.105] | **+0.181 [+0.131, +0.231]** | -0.011 [-0.042, +0.020] |

- **Mind minus rules, paired:** -0.306 [-0.610, -0.002], -0.126 [-0.417, +0.166], -0.008 [-0.333, +0.317].
- **Mind minus bandit:** -0.209, -0.217, +0.085. All three intervals include 0.
- **Mind minus PPO:** +0.067, +0.270 [+0.018, +0.521], +0.225. Only block 2 is above 0, and that is because PPO is below 0 there.

**Control (contingency broken).** The index in nights 180-199 minus nights 260-299 is:
- mind +0.132 [-0.065, +0.329], with little to lose
- bandit +0.340 [+0.138, +0.542]
- rules +0.277 [+0.121, +0.433]
- random -0.012

The bandit and rules had a real difference and lost it when the partners went deaf, as they should.

**Verdict: FAIL.** The mind's clones do not part. Only rules show the index in all three blocks, and the bandit shows it in one.

**In plain words.** Two copies of the mind with the same nature and different lives did not end up treating Fenn and Oren differently in a measurable way. Rules do part, but they have a hand-written distrust term, so that is not a learned self.

## L2: one partner, one way

**Measured.** In nights 160 to 199, the focal's treatment of the kind partner (or the hard partner) minus its treatment of the same person in the all-life P-deaf lifetime with the same seed. effect_kind should be above 0 and effect_hard below 0. Contrast = effect_kind minus effect_hard.

| chooser | effect_kind | effect_hard | contrast |
|---|---|---|---|
| mind | -0.026 [-0.190, +0.138] | -0.067 [-0.319, +0.186] | +0.041 [-0.185, +0.267] |
| bandit | +0.071 [-0.125, +0.267] | -0.207 [-0.418, +0.005] | **+0.278 [+0.033, +0.524]** |
| PPO | -0.062 [-0.109, -0.015] | +0.061 [+0.001, +0.120] | -0.123 [-0.212, -0.034] |
| rules | **+0.231 [+0.147, +0.315]** | -0.033 [-0.102, +0.036] | **+0.264 [+0.188, +0.340]** |
| random | -0.019 [-0.038, +0.001] | +0.002 [-0.023, +0.026] | -0.020 [-0.047, +0.006] |

Mind minus others on the contrast, paired:
- bandit: -0.237 [-0.550, +0.076]
- rules: -0.223 [-0.498, +0.052]
- PPO: +0.164 [-0.052, +0.380]

**Controls.**
- **Shuffled answers** (each answer credited to a random cast member):
  - Contrast: mind -0.016 [-0.339, +0.307], bandit +0.045, PPO -0.104.
  - Real minus shuffled: mind +0.057 [-0.328, +0.441], bandit **+0.233 [+0.012, +0.454]**, PPO -0.019.
  - The bandit's contrast is real, because it falls when the answers are scrambled.
- **P-deaf drift:** mind -0.058 [-0.233, +0.116], bandit -0.082, PPO -0.044 [-0.060, -0.028], rules +0.039, random +0.001. The control is clean for the mind: nothing builds up toward a deaf partner.
- **Inside the mind** (last night before the partners go deaf):
  - Chance of a hit-back from the hard partner minus the kind partner: **+0.247 [+0.130, +0.364]**.
  - Chance of a payback to a gift, kind minus hard: **+0.118 [+0.053, +0.183]**.
  - Harm chance for the hard partner minus the same person when deaf: **+0.273 [+0.139, +0.407]**.
  - Valence, kind minus hard: -0.002 [-0.021, +0.017], which is zero.

**Verdict: FAIL.** The mind's behaviour shows no contrast. The bandit and rules show one.

**In plain words.**
- The mind's partner models work. It comes to expect a hit from the hard partner and a payback from the kind one, and expects neither from the deaf one.
- The choice step does not use that knowledge: the behavioural contrast is +0.04 and the valence stays at zero for both partners.
- The bandit does turn it into different treatment.

## L3: betrayal, and learning to learn

**Measured.**
- Twin A's Fenn is P-kind, turns P-hard at night 80 (betrayal 1), P-deaf from 160, P-kind from 200, and P-hard again at night 280 (betrayal 2).
- Twin B's Fenn is P-hard first, then P-kind from night 80, then the same as A.
- Response = treat(Fenn) over the 10 nights after the switch minus the 20 nights before. Below 0 means it trusts Fenn less.
- Pass: the mind responds to the first betrayal; its second response is faster than that of the same-age twin B meeting its first (second A minus first B below 0); and no null that does both is left unbeaten.

| measure | mind | bandit | PPO | rules | random |
|---|---|---|---|---|---|
| first betrayal (A, night 80) | -0.000 [-0.121, +0.121] | **-0.377 [-0.647, -0.107]** | -0.006 [-0.176, +0.164] | -0.026 [-0.143, +0.092] | +0.030 [-0.025, +0.086] |
| second betrayal (A, night 280) | -0.070 [-0.379, +0.238] | **-0.116 [-0.212, -0.020]** | -0.030 [-0.124, +0.064] | +0.014 [-0.069, +0.096] | +0.030 [-0.031, +0.091] |
| twin B's first betrayal, same age | +0.154 [-0.003, +0.311] | **-0.162 [-0.254, -0.069]** | +0.091 [-0.101, +0.283] | -0.041 [-0.227, +0.146] | -0.025 [-0.070, +0.020] |
| second minus first (A) | -0.070 [-0.409, +0.268] | +0.261 [-0.047, +0.568] | -0.024 [-0.213, +0.165] | +0.040 [-0.058, +0.137] | -0.000 [-0.091, +0.091] |
| second (A) minus first (B), same age | -0.224 [-0.501, +0.053] | +0.045 [-0.015, +0.105] | -0.121 [-0.331, +0.089] | +0.054 [-0.107, +0.216] | +0.056 [-0.014, +0.125] |

- **Mind minus bandit, first betrayal:** +0.377 [+0.064, +0.690]. The mind is worse: it does not drop.
- **Mind's inner harm chance for Fenn**, change over 5 nights after the switch: +0.018 [-0.011, +0.047] after betrayal 1, +0.022 [-0.028, +0.072] after betrayal 2.
- **Control:** the P-deaf phase (nights 160 to 199) is the control for broken contingency, and nothing was learned there except the bandit's carry-over (see L5).

**Verdict: FAIL.**
- The mind does not follow the first betrayal.
- The second is not shown to be faster. The sign is the right way, -0.224 [-0.501, +0.053], but the interval includes 0.
- Only the bandit follows the betrayal, and it does not show a faster second one.

**In plain words.** When a trusted partner starts hitting back, the mind's behaviour toward that partner does not change within 10 nights, and its inner number barely moves. The "learning to learn" effect is suggested in sign only and is not a result.

## L4: drives, not a designer

**Measured.** A context index: per drive that has serving acts, the Spearman correlation between the need and how much more chance the chooser puts on the serving acts than their share of the menu. It uses the last half of the life and is averaged over four drives.

- Energy: eat, drink. Rest: sit down, sleep. Curiosity: look around, open. Safety: flee to.
- "Drives" is the homeostatic reward.
- "Designer" is a reward of +1 per kind act, -1 per harsh act or theft, and +0.5 for eat, drink, sit down, sleep and look around, whatever the character needs.
- Pass: the drives-reward index is above 0 and above the designer-reward index (paired), and no null shows it too.

| chooser / reward | index | energy | rest | curiosity | safety |
|---|---|---|---|---|---|
| mind / drives | -0.091 [-0.156, -0.026] | +0.079 [-0.076, +0.234] | -0.102 [-0.239, +0.035] | -0.164 [-0.269, -0.059] | -0.091 [-0.152, -0.029] |
| mind / designer | +0.008 [-0.084, +0.100] | +0.184 [-0.046, +0.414] | +0.041 [-0.107, +0.189] | -0.141 [-0.199, -0.083] | -0.022 [-0.091, +0.047] |
| bandit / drives | +0.000 [-0.091, +0.092] | +0.129 [-0.076, +0.334] | +0.037 [-0.027, +0.101] | +0.012 [-0.112, +0.135] | -0.090 [-0.208, +0.029] |
| bandit / designer | +0.025 [-0.042, +0.092] | +0.160 [-0.040, +0.361] | +0.066 [+0.003, +0.128] | -0.146 [-0.173, -0.119] | +0.020 [-0.056, +0.096] |
| PPO / drives | +0.036 [-0.015, +0.087] | +0.035 [-0.012, +0.083] | +0.175 [+0.081, +0.268] | -0.105 [-0.195, -0.016] | +0.039 [-0.035, +0.114] |
| PPO / designer | -0.021 [-0.053, +0.011] | +0.064 [-0.025, +0.153] | +0.084 [+0.054, +0.114] | -0.207 [-0.255, -0.158] | -0.027 [-0.090, +0.037] |
| rules / drives | **+0.178 [+0.108, +0.248]** | -0.082 [-0.128, -0.035] | +0.304 [+0.265, +0.344] | +0.046 [+0.021, +0.072] | +0.443 [+0.182, +0.703] |

- **Drives minus designer, paired:** mind -0.099 [-0.202, +0.004], bandit -0.025 [-0.116, +0.067], PPO +0.057 [-0.013, +0.127].
- **Random:** it has no index, because its correlation is undefined.

**Verdict: FAIL for the mind, the bandit and PPO.** Only rules put more chance on the act that serves a rising need.

**In plain words.**
- Nothing learned here chooses by what it needs. The mind's index is slightly negative.
- With the drives reward the mind is no more need-dependent than with the designer reward.
- The hypothesis that context-dependent behaviour appears only with drives (Buddy E4) is not supported in 300 nights.

## L5: order

**Measured.** Twin A lives Fenn kind then hard, twin B hard then kind. Both then meet the same P-deaf Fenn from night 160. Order index = treat(Fenn | B) minus treat(Fenn | A). Above 0 means the twin whose last phase was kind treats Fenn better. A mind that only sums its history could not show it.

| chooser | nights 160-179 | nights 180-199 |
|---|---|---|
| mind | +0.065 [-0.311, +0.440] | +0.193 [-0.174, +0.560] |
| bandit | -0.091 [-0.453, +0.271] | +0.049 [-0.131, +0.229] |
| PPO | -0.097 [-0.234, +0.041] | -0.106 [-0.271, +0.058] |
| rules | +0.070 [-0.038, +0.178] | -0.067 [-0.141, +0.007] |
| random | +0.023 [-0.007, +0.053] | -0.005 [-0.056, +0.045] |

**Inside the mind** (night before the window):
- Harm chance for Fenn, A minus B: +0.006 [-0.144, +0.156].
- Payback chance, B minus A: +0.042 [-0.034, +0.117].
- Valence, B minus A: -0.001 [-0.020, +0.018].

**Verdict: FAIL.** No chooser shows the order effect. The mind's point estimates have the right sign but the intervals are very wide, so this is a no-result rather than evidence against. Inside the mind there is no order difference either.

**In plain words.** Two minds that lived the same events in a different order did not end up treating the same person differently.

## Overall

1. **L0:** the mind keeps itself alive and fed, with the fewest deaths of any chooser (3, against 43 to 110 for the bandit, PPO and rules). PPO has the lower drive error and the bandit ties it.
2. **Where it breaks:** the partner models learn the right facts, but behaviour and valence do not follow. The weak link looks like the step from the model to the choice. My guess is that the choice score is dominated by the actor and the habit, which have settled on the interrupt exploit. I did not test this.
3. **The nulls:** the bandit tracks the partners better than the mind does, and the shuffle control confirms its effect is real. Rules do too, through their hand-written distrust term. By the plan's rule, no capability of the mind counts.
4. **Controls:** they worked. Shuffled answers removed the bandit's contrast, P-deaf produced no learned treatment in the mind, and breaking the contingency removed the bandit's and rules' difference.
5. **Limits:** 300 to 400 nights, learners not retuned, the interrupt exploit still open, and the fade at 0.9. With 10 seeds the L3 and L5 intervals are wide.

The next steps I would try are removing the interrupt lock, giving the partner terms real weight in the choice, and rerunning L1 to L5.

## Review (independent re-check, 2026-10-29): the results stand, with four corrections

- **"The bandit is better" is too strong.** The bandit SHOWS the partner effects and the mind does not. Only the L3 first-betrayal gap between them excludes 0 (+0.377 [+0.064, +0.690]).
- **The L0 pass is weak.** It means only that the gap to rules includes 0. The mind is worse than PPO (+0.241 [+0.064, +0.419]). Its low death count may come from the interrupt exploit, so it is not a skill.
- **The partner models learn fixed partners only.** They do not follow a change of partner: the harm chance moves by about +0.02 after a betrayal (L3), and by +0.006 in L5.
- **The exploit is still open.** So the L1-L5 failures show the design failing WITH the exploit in place, not that the design cannot work. The bandit makes a similar number of provocations and still shows effects, so the exploit does not explain everything. The "model-to-choice link is weak" guess is untested.
- **Checks the reviewer ran:**
  - the tables match the JSON files;
  - no partner label leaks into the mind;
  - the controls behave;
  - one new-seed rerun (seed 11) falls inside the reported range.
