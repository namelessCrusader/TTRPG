# Gap analysis report, version 2: how much of the wanted magic existing physics allows, and what the rest costs

Version 2, 2026-11-27. It re-applies five explicit owner rules (R1 to R5, section 2) to every one of the 4,128 rows of the first pass, then recomputes every table. The first pass is kept as classified_v1.tsv and report_v1.md (and the other v1 tables with the suffix _v1). The method is in docs/magic-gap-analysis.md; the decided physics is in docs/magic-science.md. Documents and small scripts only; no simulation.

## 0. The answer

- Of 4,128 rows, 275 are not physical effects and 83 are too thin to classify (v1: 275 and 69), so 3,770 are physical wishes. Tiers now: T0 2,163 (57.4%), T1 804 (21.3%), T2 577 (15.3%), T3 226 (6.0%). The big move is doorways: 169 T3 rows became T2 with A8 (rule R1), so T3 falls from 371 to 226, and T2 rises from 458 to 577.
- The T2 rows (counting the owner rows as needing A7) need A8 (a doorway, 258 rows), A7 (246), A6 (100) or A5 (11). A9 and A10 are gone: no row needs them. A2, A3 and A4 are still needed by no row.
- **The exact minimal cover of every T1 and T2 row is A1, A5, A6, A7, A8** (cost 26 units counting A1; 14 beyond A1), the same under either reading of the owner rows. Without A5 (11 rows, all low confidence) it is A1, A6, A7, A8 and covers 99.2% of the rows; without A5 and with the owner reading ghosts as belief it is still A1, A6, A7, A8 but A7 then serves only 19 rows.
- **A7 now hangs on one owner choice.** 227 rows are labelled "A7-or-T0 (owner)" (ghosts, possession, patrons, spirits, gods). If the owner reads them as belief, culture, misperception or A1 effects they drop out: 197 become T0 and 30 stay T2 only because they also need A6 or A8. Only 19 rows need A7 whatever the owner chooses (a mind that must exist outside its brain: stored minds, body swap, merging, a thinking object). v1 said 275 rows needed A7; v2 says 19 firmly, 227 on the owner's choice.
- A1 alone enables 804 of 1381 T1+T2 rows (58.2%) with the owner rows counted as A7, or 67.9% with them dropped to T0. Adding A8 first is the best single step in both readings (A8 76.0% and 89.2%). A1, A6, A7, A8 reaches 99.2%; high and medium confidence rows only, A1, A6, A7, A8 reaches 100%.
- Confidence: low-confidence rows among T1 and T2 fall from 642 of 1264 (50.8%) to 466 of 1381 (33.7%). The T1 share of that fall is a rule-based review (rows raised from low to medium because their stated reason names an A1 mechanism), not new evidence; see section 9.

## 1. What changed from v1

Of 4,128 rows, 251 changed tier (6.1%) and 3,877 kept their tier. Every changed row carries the old tier (tier_v1), the rule that moved it (rules) and a note (note2) in classified.tsv.

| v1 tier to v2 tier | rows |
|---|---|
| T3 to T2 | 169 |
| T2 to T1 | 26 |
| T2 to T3 | 21 |
| T1 to T0 | 12 |
| T1 to U | 11 |
| T2 to U | 3 |
| T2 to T0 | 3 |
| T1 to T3 | 3 |
| T1 to T2 | 2 |
| T0 to T2 | 1 |

Tier counts, v1 against v2:

| tier | v1 | v2 | change |
|---|---|---|---|
| T0 | 2149 | 2163 | +14 |
| T1 | 806 | 804 | -2 |
| T2 | 458 | 577 | +119 |
| T3 | 371 | 226 | -145 |
| X | 275 | 275 | +0 |
| U | 69 | 83 | +14 |

What each rule did:

| rule | rows touched | effect |
|---|---|---|
| R1 doorways | 170 | 169 T3 rows to T2 A8 (bounded version lawful, unbounded kept as the T3 version), plus one T0 row (a nightly door to a hidden plane). Rows that remain T3 are those whose bounded version is not a doorway or does not exist. |
| R2 mind effects | 49 | 18 T2-A7 rows to T1 A1 (plus 6 antimemetic rows, section 8) (drive, arousal, sleep and percept effects, bonded animals, dream-percept machines), 6 dream-place rows to T2 A1+A8, and the 19 strict-A7 rows kept at A7 with a stated reason. |
| R3 belief rows | 227 | 227 rows labelled A7-or-T0 (owner), counted separately (section 4). |
| R4 T1 review | 327 | All 327 low-confidence T1 rows re-checked (section 9): 199 T1 confirmed, raised to medium, 100 T1 kept low, 12 to T0, 2 to T2 (A6), 3 to T3, 11 to U. |
| R5 other | 30 | The 30 antimemetic and cognitohazard rows re-tiered (section 8): A9 now needs 0 rows. Phasing moved from A10 to A8. |

The earlier skeptical check found three defects. Doorway rows split between T2-A8 and T3: fixed by R1, and a final script check finds no row that lists A8 with a lawful bounded version and is still T3 except 10 deliberate exclusions (creation from nothing, omnipotence, reality rewrite, time-loop and body-copy resets, mind transfer). T1 inflated: the review removes 28 rows from T1 and raises 199 from low to medium on a stated criterion. T2-A7 over-reach: 26 rows moved to T1 A1 (18 mind-effect rows and 8 antimemetic rows) and the A7 rows are split into strict and owner. The 79.2% tier agreement was measured on v1 and has not been re-measured on v2; the rules above remove the three systematic disagreements, but a fresh blind second pass on a sample is still the way to claim the 80% bar.

## 2. The five rules as applied

- **R1 doorways.** A doorway, plane access, teleport or summoning whose bounded version conserves energy and momentum (a flux-carrying join between places, Q10) is T2 with A8. Only the unbounded version (instant, free, faster than light, to the past) is T3. Every A8 row records both (columns bounded_version, bounded_tier, unbounded_version, unbounded_tier): bounded T2, unbounded T3.
- **R2 mind effects.** Induced percepts, mood, arousal, calm, panic, drowsiness, confusion, stuns and entrainment are A1 (Q5.3: magic acts on minds through their ports). Only effects that need a mind to exist outside its brain are A7.
- **R3 belief rows.** Ghost, possession, patron and spirit rows whose effects can be explained as belief, culture, misperception or A1 effects are labelled "A7-or-T0 (owner)" (column label, owner=1) and counted separately.
- **R4 T1 review.** A row is T1 only if it needs magic and A1's control, sensing and energy at a distance suffice; otherwise T0 (no magic) or T2.
- **R5 other.** The 30 antimemetic and cognitohazard rows are re-tiered under R2. Content-selective erasure of specific memories needs a model of that brain's code (Q5.3), so it is T3 for strangers. Phasing goes to A8.

How I drew the strict-A7 line between R2 and R3 (the one judgement the rules left open): a row is strict A7 when its effect is a thinking agent that must live in a non-brain carrier or merge with another mind (Chucky, a lich in a phylactery, a sentient sword, body swap, the Instrumentality), because belief or A1 percepts cannot reproduce that effect. A ghost that haunts, a possessing demon, a patron god, an angel or a familiar spirit is an owner row, because the observable effects (apparitions, a changed host, power that tracks a cult) are reproducible by A1 percepts, A1 control and belief. This keeps the owner's choice small and legible; the 19 strict rows are listed in classified.tsv (A7class = strict).

## 3. Tiers by family and lane

Tier counts use the primary tier, so owner rows are counted as T2 here (the owner column counts them). Rows can sit in several lanes, so lane totals exceed 4,128. Full family table: matrix_family_tier.tsv.

| lane | rows | T0 | T1 | T2 | T3 | X | U | owner | T1+T2 | share T0 | share T1+T2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bestiaries | 260 | 159 | 55 | 38 | 5 | 3 | 0 | 19 | 93 | 0.612 | 0.358 |
| creepypasta | 277 | 200 | 8 | 58 | 5 | 6 | 0 | 19 | 66 | 0.722 | 0.238 |
| cyberpunk_transhumanism_postapoc_mecha | 452 | 290 | 89 | 46 | 21 | 6 | 0 | 30 | 135 | 0.642 | 0.299 |
| dreams_fairy_tales_and_liminal | 390 | 151 | 50 | 67 | 13 | 29 | 80 | 5 | 117 | 0.387 | 0.3 |
| folklore | 411 | 255 | 72 | 68 | 8 | 8 | 0 | 37 | 140 | 0.62 | 0.341 |
| horror | 292 | 183 | 38 | 56 | 7 | 8 | 0 | 29 | 94 | 0.627 | 0.322 |
| litrpg_cultivation | 337 | 125 | 96 | 34 | 20 | 62 | 0 | 12 | 130 | 0.371 | 0.386 |
| mythology_pantheons_religion | 345 | 178 | 62 | 43 | 21 | 41 | 0 | 34 | 105 | 0.516 | 0.304 |
| scifi | 285 | 185 | 50 | 15 | 21 | 14 | 0 | 2 | 65 | 0.649 | 0.228 |
| scp | 311 | 126 | 63 | 60 | 41 | 18 | 3 | 16 | 123 | 0.405 | 0.395 |
| superpowers | 392 | 160 | 141 | 41 | 37 | 13 | 0 | 17 | 182 | 0.408 | 0.464 |
| videogame_mechanics_and_loops | 325 | 117 | 85 | 34 | 30 | 59 | 0 | 3 | 119 | 0.36 | 0.366 |
| weirdfiction | 235 | 130 | 28 | 57 | 10 | 10 | 0 | 26 | 85 | 0.553 | 0.362 |

Lane against addition (T1 and T2 rows, owner rows counted as A7; A1 is implied for all):

| lane | A1 | A5 | A6 | A7 | A8 |
|---|---|---|---|---|---|
| bestiaries | 93 | 2 | 8 | 22 | 8 |
| creepypasta | 66 | 1 | 20 | 19 | 23 |
| cyberpunk_transhumanism_postapoc_mecha | 135 | 4 | 3 | 32 | 9 |
| dreams_fairy_tales_and_liminal | 117 | 0 | 6 | 5 | 59 |
| folklore | 140 | 0 | 23 | 44 | 10 |
| horror | 94 | 1 | 18 | 35 | 7 |
| litrpg_cultivation | 130 | 0 | 1 | 15 | 18 |
| mythology_pantheons_religion | 105 | 0 | 0 | 34 | 14 |
| scifi | 65 | 1 | 1 | 4 | 10 |
| scp | 123 | 0 | 2 | 16 | 42 |
| superpowers | 182 | 0 | 8 | 17 | 19 |
| videogame_mechanics_and_loops | 119 | 1 | 2 | 4 | 28 |
| weirdfiction | 85 | 2 | 20 | 26 | 13 |

## 4. The owner's one choice: A7 or T0

227 rows are A7-or-T0 (owner). Reading them as T0 removes A7 from 197 rows outright; 30 keep a T2 tier through A6 or A8 (for example a ghost on an underworld plane). The largest families among them: Chaos (Warhammer 40,000) (9), Ghosts (8), Weird West (7), Great Old Ones (6), Jinn (6), Ceremonial magic (6), Magic (Shadowrun) (6), Demons (5).

| reading of the owner rows | T0 | T1 | T2 | T1+T2 | A7 needed by | minimal cover |
|---|---|---|---|---|---|---|
| owner picks A7 (rows need A7) | 2163 | 804 | 577 | 1381 | 246 rows (19 strict + 227 owner) | A1, A5, A6, A7, A8 |
| owner picks T0 (belief reading) | 2360 | 804 | 380 | 1184 | 19 rows (strict only) | A1, A5, A6, A7, A8; or A1, A6, A8 for 97.6% of the rows |

So the choice moves 197 rows between T0 and T2 and changes how much A7 buys, not which additions the full cover needs: the 19 strict rows (including 2 of the A5 rows) keep A7 in the cover either way. If the owner also declines the 19 strict rows as out of scope, the cover is A1, A5, A6, A8. Listing: classified.tsv, filter label = "A7-or-T0 (owner)".

## 5. Coverage curve

For each number k of additions beyond A1, the best set (most rows enabled, ties by lower cost) from the 7 candidates A2 to A8, by exhaustive search. A row is enabled when every addition it needs is in the set. Cost is in the units of cost_table.tsv (unchanged from v1). A9 and A10 are not candidates because no row needs them.

**owner rows counted as needing A7 (primary).** 1381 T1+T2 rows; 915 of them high or medium confidence. File coverage_curve.tsv.

| k | best set (all rows) | rows | share | cost beyond A1 | best set (high+medium only) | rows | share |
|---|---|---|---|---|---|---|---|
| 0 | A1 | 804 | 58.2% | 0 | A1 | 680 | 74.3% |
| 1 | A1,A8 | 1050 | 76.0% | 4 | A1,A8 | 830 | 90.7% |
| 2 | A1,A7,A8 | 1270 | 92.0% | 8 | A1,A7,A8 | 881 | 96.3% |
| 3 | A1,A6,A7,A8 | 1370 | 99.2% | 11 | A1,A6,A7,A8 | 915 | 100.0% |
| 4 | A1,A5,A6,A7,A8 | 1381 | 100.0% | 14 | A1,A5,A6,A7,A8 | 915 | 100.0% |
| 5 | A1,A2,A5,A6,A7,A8 | 1381 | 100.0% | 18 | A1,A2,A5,A6,A7,A8 | 915 | 100.0% |

Greedy order (all rows): A1 (804), A8 (1050), A7 (1270), A6 (1370), A5 (1381). High+medium: A1 (680), A8 (830), A7 (881), A6 (915).

**owner rows read as belief (T0).** 1184 T1+T2 rows; 870 of them high or medium confidence. File coverage_curve_ownerT0.tsv.

| k | best set (all rows) | rows | share | cost beyond A1 | best set (high+medium only) | rows | share |
|---|---|---|---|---|---|---|---|
| 0 | A1 | 804 | 67.9% | 0 | A1 | 680 | 78.2% |
| 1 | A1,A8 | 1056 | 89.2% | 4 | A1,A8 | 830 | 95.4% |
| 2 | A1,A6,A8 | 1156 | 97.6% | 7 | A1,A6,A8 | 864 | 99.3% |
| 3 | A1,A6,A7,A8 | 1173 | 99.1% | 11 | A1,A6,A7,A8 | 870 | 100.0% |
| 4 | A1,A5,A6,A7,A8 | 1184 | 100.0% | 14 | A1,A5,A6,A7,A8 | 870 | 100.0% |
| 5 | A1,A2,A5,A6,A7,A8 | 1184 | 100.0% | 18 | A1,A2,A5,A6,A7,A8 | 870 | 100.0% |

Greedy order (all rows): A1 (804), A8 (1056), A6 (1156), A7 (1173), A5 (1184). High+medium: A1 (680), A8 (830), A6 (864), A7 (870).

The curve compared with v1 (v1: A1 63.8%, then A7 82.8%, A6 90.2%, A8 96.7%): the first addition is now A8, not A7, because R1 moved 169 doorway rows into T2, and A7's share shrinks because of R2 and R3. On high and medium rows alone the first addition is again A8, and A1 + A8 + A7 reach 96.3%.

## 6. The exact minimal cover

Each row needs all of its listed additions together, so the smallest set that covers every T1 and T2 row is the union of what the rows need; the exhaustive search over all 128 subsets of A2 to A8 confirms nothing smaller covers 100%.

| rows | scenario | exact minimal set | cheapest set for 90% | 95% | 99% | 100% |
|---|---|---|---|---|---|---|
| all T1+T2 | owner = A7 | A1, A5, A6, A7, A8 | A1, A7, A8 | A1, A6, A7, A8 | A1, A6, A7, A8 | A1, A5, A6, A7, A8 |
| all T1+T2 | owner = T0 | A1, A5, A6, A7, A8 | A1, A6, A8 | A1, A6, A8 | A1, A6, A7, A8 | A1, A5, A6, A7, A8 |
| high+medium only | owner = A7 | A1, A6, A7, A8 | A1, A8 | A1, A7, A8 | A1, A6, A7, A8 | A1, A6, A7, A8 |
| high+medium only | owner = T0 | A1, A6, A7, A8 | A1, A8 | A1, A8 | A1, A6, A8 | A1, A6, A7, A8 |

The only reason A5 stays in the full cover is 11 rows, every one low confidence (Colour Out of Space, Ghroth, the Warp, warp storms, the Beams, the Crimson King, Fire elemental, Fumecrux, Reality glitch, Energy beings, Sentient sword). A5 has no support from high or medium rows. The need patterns are in matrix_need_patterns.tsv:

| needs | rows (owner = A7) | rows (owner = T0) |
|---|---|---|
| A1 | 804 | 804 |
| A1,A8 | 246 | 252 |
| A1,A7 | 214 | 17 |
| A1,A6 | 72 | 96 |
| A1,A6,A7 | 24 | 0 |
| A1,A5 | 7 | 7 |
| A1,A7,A8 | 6 | 0 |
| A1,A6,A8 | 4 | 4 |
| A1,A5,A8 | 2 | 2 |
| A1,A5,A7 | 2 | 2 |

## 7. Variety per addition

Cost units are unchanged. "Rows" counts T1 and T2 rows needing the addition; "families" the distinct catalogue families of 598; "kinds" the 12 keyword mechanism kinds (a coarse proxy). "Only" is rows needing nothing beyond A1 plus this addition. Owner = A7 scenario:

| addition | cost | rows | families | kinds | only rows | only families | families per cost | kinds per cost |
|---|---|---|---|---|---|---|---|---|
| A1 | 12 | 1381 | 335 | 12 | 804 | 235 | 27.92 | 1.0 |
| A5 | 3 | 11 | 9 | 7 | 7 | 6 | 3.0 | 2.33 |
| A6 | 3 | 100 | 56 | 10 | 72 | 43 | 18.67 | 3.33 |
| A7 | 4 | 246 | 116 | 12 | 214 | 107 | 29.0 | 3.0 |
| A8 | 4 | 258 | 116 | 8 | 246 | 110 | 29.0 | 2.0 |
| A7 strict only | 4 | 19 | 14 | 3 | 17 | 12 | 3.5 | 0.75 |
| A7 owner rows only | 4 | 227 | 108 | 12 | 197 | 100 | 27.0 | 3.0 |

Owner = T0 scenario (A7 keeps only its strict rows):

| addition | cost | rows | families | kinds | families per cost | kinds per cost |
|---|---|---|---|---|---|---|
| A1 | 12 | 1184 | 305 | 12 | 25.42 | 1.0 |
| A5 | 3 | 11 | 9 | 7 | 3.0 | 2.33 |
| A6 | 3 | 100 | 56 | 10 | 18.67 | 3.33 |
| A7 | 4 | 19 | 14 | 3 | 3.5 | 0.75 |
| A8 | 4 | 258 | 116 | 8 | 29.0 | 2.0 |
| A7 strict only | 4 | 19 | 14 | 3 | 3.5 | 0.75 |

Reading. A8 and A7 tie at 29 families per cost unit when the owner rows count, ahead of A6 (18.7) and A5 (3.0), with A1 at 27.9 for 12 units. A6 opens the most kinds per cost (3.3). If the owner reads ghosts as belief, A7 falls to 3.5 families per cost unit and A8 (29.0) and A6 (18.7) lead; A7 then buys only 14 families. A8 opens 8 kinds (doorways, places, time-slip, summoning and the dream places) at the price of one flux-carrying join, and is the only addition that turns T3 rows into T2 (169 rows). A2, A3 and A4 unlock no catalogue row; their case rests on the research and on composition in the toy simulations, which the catalogue cannot show.

## 8. Doorways, minds and the 30 antimemetic rows

**Doorways (R1).** T2 rows needing A8: 258 (v1: 84). Of these 169 came from T3. For all of them the bounded version is T2 and the unbounded version is T3 (recorded per row). Bounded: a flux-carrying join between places, energy and momentum booked, finite pockets, travel at or below light speed. Unbounded (T3): free or instant relocation, faster-than-light travel or signalling, a doorway to the past, creation of a place from nothing. T3 rows kept at T3 are those with no lawful doorway reading: time travel and loops, creation from nothing and omnipotence, exact copies and uploads of a mind or body, true foresight, content-selective brain writes, FTL and metric engineering. T3 now has 226 rows.

**Mind effects (R2).** 18 rows moved from A7 to A1: imposed drives (Domination, Mind Control, Controller), bonded animals and familiars (synchrony through the ports), sleeper-percept rows (Dream invasion, shared dream, Inception, Paprika, Freddy Krueger, dream creatures), the succubus (energy drain plus dream percepts), Bird Box (a percept that triggers panic), astral space as sensing, and the wish-jewel as charged matter with coarse sensing. Six dream-place rows (Dreamlands, Dreamscape, Dream-Quest, Dream Cycle and the like) are now T2 A1+A8: the sleeper gets percepts, the place is a bounded joined region. Six rows that read or write specific thoughts or commands (Clairtelepathy, Detect thoughts, Soul search, Neuralyzer, Neurohacking, Geass) became T3 for strangers; the coarse version is A1.

**The 30 antimemetic and cognitohazard rows (R5).** After re-tiering: T0 3, T1 8, T2 1, T3 15, U 3. A rage or panic trigger, a general amnesia place, a hallucination and a neurological hazard from reading are percept or coarse memory effects through the ports, so T1 A1 (8 rows: SCP-096, -1128, -2571, -4133, -4921, -7390, -8031, -8066). Rows whose harm or forgetting depends on the content (one memory, one secret, one object, a puzzle that transforms the solver) need a model of that brain's code and are T3 for strangers (15 rows). Rows that are culture or a computer worm are T0 (3); rows too thin to classify are U (3); one stone-ghost row joins the owner rows. **A9 is not adopted and no row needs it.** Phasing (1 row) went to A8 as a local doorway that travels with the body; A10 is dropped.

## 9. The T1 review and what is still uncertain

All 327 T1 rows that v1 marked low confidence were re-checked against R4. Outcome: T1 confirmed, raised to medium 199, T1 kept low 100, to T0 12, to T2 (A6) 2, to T3 3, to U 11. The 12 moved to T0 need no magic (for example Destruction of the Moon, a radar blocker, a sensor that reads mood coarsely, a brain-machine interface, blunt drug amnesia, visions and glossolalia as culture); 3 replay whole recorded experiences and are T3 by the content-write rule; 2 hold memories or a patch in the field and are T2 A6; 11 "Object N" rows have no stated effect and are U. A row was raised from low to medium only when its reason names an A1 mechanism (guided regeneration, steered matter or force, light bending, induced percepts, sensing, synchrony, charged matter) and its mechanism text is specified; that is a consistency check against R4, not new evidence, so it should not be read as the same confidence that a blind second classification would give.

Low-confidence rows now: T0 208, T1 124, T2 342, T3 71, X 6, U 83 (total 834; v1 1,032). Among T1+T2: 466 of 1381. What is still uncertain, in order of what changes the answer:

1. **The owner choice on A7** (section 4). It decides whether A7 is a 19-row addition or a 246-row addition.
2. **T2 rows are still mostly low confidence** (342 of 577): 108 involve a doorway (many are Backrooms "Level N" and SCP place rows with "(u)" mechanisms), 195 need A7 and 66 need A6 (a row can need more than one). The three additions are well supported by the structure of the rows but not by their individual descriptions.
3. **A5**: 11 rows, all low. If none holds, A5 leaves the cover.
4. **The 124 T1 rows that are still low (100 kept low by the R4 review, 24 moved down from T2 by R2)** and the "(u)" rows generally. A blind second pass on a random sample of 400 rows is the way to re-measure tier agreement against the 80% bar; v2 has not been re-measured.
5. **My strict-versus-owner line for A7** (section 2) is a judgement; the owner can move rows across it by editing A7class.

## 10. The case bank

cases_classified.tsv is unchanged except C19 (a sleeping person casts in a dream): under R2 the sleeper's own brain produces the dream percepts and the casting is the person's own A1 control, so C19 is now T1 A1 rather than T2 A7, and the only case that needed an undecided addition now needs none. C24 (teleport with no doorway) stays T3: the unbounded teleport is T3 and its bounded version is the A8 doorway pack.

## Files (all beside this report)

- classified.tsv (v2): all 4,128 rows, merged.tsv columns plus tier, tier_v1, needs, needs_v1, label (A7-or-T0 (owner)), owner, A7class, tier_if_owner_T0, needs_if_owner_T0, bounded and unbounded versions and tiers, confidence, conf_v1, rules, note2 and one 0/1 column per addition. classified_v1.tsv and report_v1.md are the first pass; other first-pass tables carry the suffix _v1.
- coverage_curve.tsv, coverage_greedy.tsv (owner = A7) and coverage_curve_ownerT0.tsv, coverage_greedy_ownerT0.tsv (owner = T0), each with all rows and high+medium columns.
- matrix_additions.tsv, matrix_need_patterns.tsv, matrix_kind_addition.tsv, variety_per_cost.tsv (owner = A7) and the same with _ownerT0; matrix_family_tier.tsv, matrix_lane_tier.tsv, matrix_lane_addition.tsv; cost_table.tsv; other_additions.tsv; lowconf.tsv; cases_classified.tsv.
