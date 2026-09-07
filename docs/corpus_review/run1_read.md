# Run-1 quality read — the 10 v3 findings, re-scored on the real engine

_Corpus: `_run1/corpus.jsonl`, 17 rows, vault seed 12, beats t0–t20. Protagonist AI-composed
(decision-2 piece trees); all NPCs on mind.py. One refusal (named piece, zero time cost)._

| # | v3 finding | run-1 verdict |
|---|---|---|
| 1 | perception tracked, ignored | **FIXED by construction** — menu built from the fresh slice; 0/17 rows target anything unseen |
| 2 | simultaneity authored; player wrote own death | **FIXED** — the sim killed the player (t20 "the light goes out of you"); the author never narrated anyone |
| 3 | range/possession unenforced | **FIXED** — range structural; the vial travelled with Sly as `has_prize` the whole run |
| 4 | stakes decorative (fire never touches an agent) | **MOSTLY FIXED** — soaked_oil + spark = roaring flame + scald (fire→hp lives); NEW same-family gap: cell acid never hurt a person (ticket 5) |
| 5 | escalation re-improvised per beat | **FIXED** — sim-owned constants; hp fell 10→8→4.8 consistently, no ad-hoc numbers anywhere |
| 6 | disposition ratchets instead of responds | **FIXED** — it responded: −0.1 for pestering, −0.25 + hostility 0.8→1.0 for a witnessed attack |
| 7 | premise/trace divergence | **FIXED** — the goal was real and contested; the story stayed the premise (vial, guard, rival) to the end |
| 8 | engine corrupts LM inputs (silent refusals) | **FIXED** — the one refusal named its piece ('ignite'), cost no time, and the author adapted. (Row 1 still shows the pre-fix `\|`-swallow bug — kept as history.) |
| 9 | affordance set tiny | **BETTER, still thin socially** — 6 distinct verbs actually used (speak/move/attack/pour/ignite + a refused combo); but no give/yield path exists, so a persuaded NPC *could not* hand the vial over (ticket 2 family) |
| 10 | register leaks (numbers in dialogue) | **FIXED** — every spoken line is diegetic ("the oil at your feet meets my torch"); no stat reads leaked. Rally is still missing: Pip stayed `afraid` forever |

**Score: 8 fixed, 2 partial (4, 9).** Both partials are world-content gaps, not constraint
gaps — the same shape as the brood finding: the ceiling is the world, not the pen.

New tickets from this run: (4) tree legality is parse-time only — a step unlocked by an earlier
step ((then (pour oil) (ignite))) is refused; (5) acid→living-flesh coupling missing;
(6) "sluices the oil away" and "goes up in flame" fired the same tick — ordering.
