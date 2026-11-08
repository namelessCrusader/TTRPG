# Learned mind, round 3, step 0a: does the world pay?

Written 2026-10-30 by the main session from the lane's report (`report_0a.txt`, the full text with all tables and the menu mapping) and the skeptical check (`~/.cache/mark1-build/mindL3/0a_verify/verify_report.txt`, verdict PASS).
- Code: `lab/mindL/tournament.py`, `tournament_report.py`, `test_tournament.py` (15 tests).
- Results: 490 lifetimes, 100 nights each, 1.96 CPU hours, in this directory.
- No world file was changed.

## Answer

**The inn pays for peace, not for reciprocity.** One of the five pass rules passes, and that one is hollow.

| Rule | Result | Numbers (10 seeds, paired, 95% t) |
|---|---|---|
| Oracle gap at least 0.1 | FAIL | 0.061 [0.011, 0.111]. The most any policy can win back from the social drives is 0.130. |
| Hermit and always-harsh lose to rules | FAIL | Always-harsh loses (+0.308). The hermit WINS (-0.235 [-0.421, -0.050]). |
| Always-giving loses to tit-for-tat | FAIL | -0.075 [-0.286, +0.136]: no detectable difference. |
| q·b > c | PASS, hollow | +0.090 with q = "a third party was present". With q = "a third party's opinion moved", q_eff = 0.02 and the result is -0.013. |
| Extractor wins only where reprisal is weak | FAIL | It wins nowhere: worse than rules in all 40 cells, even with reprisal off. |

## What is missing in the world

1. **Only harm is answered.** A harsh act gets a harsh reply within 40 turns 55% of the time. A kind act gets a kind reply 3% of the time, about the background rate.
2. **Reputation exists for harm, not for kindness.** Gossip carries 88 bad deeds for each good one. A third party's opinion of a giver moves after 2% of kind acts.
3. **The hand-written cast is hostile.** `rules` makes 10.4 harsh acts a night. With reprisal on, an all-rules cast ends at a mean liking of -0.77 between its members.
4. **Hunger dominates, and food barely reaches anyone.** The cook holds 2 portions a night and eats them herself; the other four eat almost nothing. Hunger is 0.32 to 0.35 of every stance's error, and 69% of the blind error in the L2 setting. Asking the cook is the one social act the world clearly pays for.
5. **The price of provoking the retaliator is small.** About 0.15 per provocation, which is below what a learner can see through its own noise.

## What this means for round 2

The round-2 mind avoided the retaliator, provoked rarely, and did not reward the kind partner. That fits what this world pays for. Fix the world before the mind (the plan's rule: stop at the first failure; each failure names its fix).

## Checker's notes (all accepted)

- Rule 4 passes only with the loosest q.
- Rule 3 is "no detectable difference", not "cannot pass".
- 40 oracle lifetimes come from an older recorder. Re-running them gives identical numbers.
- Some wording goes past the data:
  - Hunger is not the largest error for always-giving; wealth is (0.361 against 0.319).
  - The feeder's edge is weaker on fresh seeds: -0.057 [-0.122, +0.007].
