# Gap analysis report: how much of the wanted magic existing physics allows, and what the rest costs

Produced 2026-11-27 from the seven classifier files and the case-bank file, by documents and small scripts only (no simulation). The method is in docs/magic-gap-analysis.md; the decided physics is in docs/magic-science.md. The consolidation script is /home/nisargparikh/.cache/mark1-build/gap/consol.py and the report script rep.py beside it. Every table here is also written as TSV in this folder.

## 0. The answer in six lines

- Of the 4,128 catalogue rows, 275 are not physical effects (tools, meta-features, narrative) and 69 are too thin to classify, so 3,784 rows are physical wishes. Of those, 2,955 (78.1%) need nothing beyond the base A1: 2,149 (56.8%) are ordinary physics, biology and culture with no magic at all, and 806 (21.3%) are ordinary physics steered by A1 (control, sensing and energy at a distance). 458 (12.1%) need a further addition, and 371 (9.8%) need a broken law.
- This matches the expectation that the largest magic class is ordinary physics steered. It is 63.8% of the 1,264 rows that need magic at all (T1 plus T2), and by construction A1 alone enables all of T1; the informative question is what the 458 T2 rows need.
- The 458 T2 rows need A7 (a strong mind-field coupling), A6 (lasting imprints), A8 (a doorway), A5 (self-sustaining packets), or a "content-selective neural effect" that no listed addition supplies. The exact minimal set that covers every T1 and T2 row as classified is A1, A5, A6, A7, A8 and two small new candidates (A9, A10). I propose that neither new candidate be adopted (section 2), which gives the minimal set A1, A5, A6, A7, A8 over 1,234 rows.
- Three additions carry almost all the coverage. A1 plus A6, A7 and A8 enables 1,222 of 1,264 T1+T2 rows (96.7%); A5 adds only a further 11 rows.
- No catalogue row needs A2 (magnetic charge), A3 (condition-dependent pairing) or A4 (a non-abelian symmetry). They bring 0 unlocked rows at a combined cost of about 15 units. Their case rests on the bottom-up research (RQ4, sharp-bordered dead zones, threads), not on the catalogue; the catalogue cannot yet tell whether they open variety, which is what the toy simulations are for (method step 5).
- The largest single lever is the owner's decision on A7, the mind partly made of the field. It alone unlocks 275 rows (240 of them needing nothing else) in 124 families, but 221 of those 275 are low confidence, because a ghost, possession or patron row can equally be read as belief and culture (T0). Reading them that way removes most of A7's demand.

## 1. Completeness check

Every one of the 4,128 rows of docs/coverage/merged.tsv appears exactly once in docs/coverage/gap/classified.tsv (checked by script: 4,128 distinct entry names in the catalogue, 4,128 classifier rows, the two sets identical, no duplicates, no family mismatches). No row was missing, so none was re-classified for completeness. The classifier files split as 590, 590, 590, 590, 590, 590 and 588 rows. classified.tsv carries all merged.tsv columns, then the final tier, the raw tier, the additions the row needs (A1 implied for every T1 and T2 row), the raw additions text, the bounded-version note, confidence, reason, the classifier number, a note where I changed a row, and one 0/1 column per addition A1 to A10.

I changed 71 rows during normalisation and kept the raw values beside them (section 2). Classifier 6 reported a late fix (Nanopunk, T2 to T0); the file I read already holds the fix.

Final tiers by classifier (after normalisation):

| classifier | T0 | T1 | T2 | T3 | X | U | rows |
|---|---|---|---|---|---|---|---|
| 1 | 322 | 102 | 115 | 29 | 22 | 0 | 590 |
| 2 | 370 | 111 | 42 | 49 | 18 | 0 | 590 |
| 3 | 290 | 114 | 125 | 46 | 15 | 0 | 590 |
| 4 | 345 | 112 | 50 | 53 | 30 | 0 | 590 |
| 5 | 260 | 134 | 60 | 44 | 92 | 0 | 590 |
| 6 | 327 | 134 | 45 | 60 | 24 | 0 | 590 |
| 7 | 235 | 99 | 21 | 90 | 74 | 69 | 588 |

Classifier 5 has 92 X rows (15.6%, against 3 to 5% elsewhere); the rows are source works, grimoires and LitRPG mechanics, so the share is plausible for its families rather than a sign of loose work. Classifier 5 itself warned that group assignments were loose, so its 134 T1 and 60 T2 rows are the first to re-check. Classifier 7 holds the Backrooms entities, which explains its 69 under-specified rows.

## 2. The "other" additions, normalised

The classifiers wrote seven different "other:" texts on 102 rows. They reduce to four groups:

| group | raw texts | rows | proposal | why |
|---|---|---|---|---|
| U: unspecified anomalous entity | "unspecified anomalous traits" (19) and the same with "the creature body itself is T0" (50) | 69 | Not an addition. Re-tiered to U (under-specified) and kept out of the cover. | These are Backrooms entities with no stated effect; "other" was a placeholder, not a missing law. They are low confidence. If the owner wants them counted, the classifier note says each would be T0 (creature body) or an A8 doorway case. |
| Transmutation at nuclear cost | "bulk transmutation at nuclear cost" and "bulk body transmutation at nuclear cost" (Tetsuo, Shapechange) | 2 | Fits A1. Re-tiered T2 to T1. | Decision Q5.2 makes transmutation lawful at its real nuclear cost, and classifier 1 and 3 already tier alchemy rows T1. The two rows were inconsistent with their siblings. |
| A9 candidate: content-selective neural effect | "content-selective neural effect" (29), "content-selective memory suppression" (1) | 30 | Recommend NOT adopting. If a candidate is wanted, call it A9. | All 30 are SCP antimemetic, cognitohazard and infohazard rows (an effect that depends on the meaning of what is perceived or remembered). The decisions already rule the physical route out: Q5.3 allows only the ports (flashes, mood, sleep, never inserted thoughts) and C124 says a false memory is out of reach. Two lawful readings remain: a stimulus whose harm comes through ordinary perception and the brain's own processing is T0, and a rewrite or erasure of memory without perception is T3. Adding a mechanism that reads or writes meaning would be a large new addition (about 4 cost units) for 2.4% of the T1+T2 rows. The owner should look at these 30 rows (lowconf.tsv) and choose T0, T3 or X for each. |
| A10 candidate: selective suppression of contact forces | Phasing (1 row) | 1 | Not an addition. Fold into A8 (a local doorway that travels with the body) or leave T3. | One row does not justify a new field or term. |

The tier changes are in classified.tsv (column note) and other_additions.tsv. Unnormalised, the cover needs two more additions (A9, A10); normalised, it does not.

## 3. Tiers by catalogue family and lane

Rows counted once each. The 13 lanes are the catalogue's source lanes; a row listed in several lanes is counted in each (so lane totals exceed 4,128). The share columns are of all rows in the lane.

| lane | rows | T0 | T1 | T2 | T3 | X | U | T1+T2 | share T0 | share T1+T2 |
|---|---|---|---|---|---|---|---|---|---|---|
| cyberpunk_transhumanism_postapoc_mecha | 452 | 287 | 93 | 42 | 24 | 6 | 0 | 135 | 0.635 | 0.299 |
| folklore | 411 | 255 | 71 | 66 | 11 | 8 | 0 | 137 | 0.62 | 0.333 |
| superpowers | 392 | 160 | 140 | 32 | 47 | 13 | 0 | 172 | 0.408 | 0.439 |
| dreams_fairy_tales_and_liminal | 390 | 152 | 56 | 16 | 68 | 29 | 69 | 72 | 0.39 | 0.185 |
| mythology_pantheons_religion | 345 | 173 | 66 | 43 | 22 | 41 | 0 | 109 | 0.501 | 0.316 |
| litrpg_cultivation | 337 | 125 | 94 | 21 | 35 | 62 | 0 | 115 | 0.371 | 0.341 |
| videogame_mechanics_and_loops | 325 | 115 | 85 | 9 | 57 | 59 | 0 | 94 | 0.354 | 0.289 |
| scp | 311 | 123 | 55 | 85 | 30 | 18 | 0 | 140 | 0.395 | 0.45 |
| horror | 292 | 183 | 36 | 53 | 12 | 8 | 0 | 89 | 0.627 | 0.305 |
| scifi | 285 | 184 | 51 | 7 | 29 | 14 | 0 | 58 | 0.646 | 0.204 |
| creepypasta | 277 | 199 | 9 | 39 | 24 | 6 | 0 | 48 | 0.718 | 0.173 |
| bestiaries | 260 | 159 | 53 | 37 | 8 | 3 | 0 | 90 | 0.612 | 0.346 |
| weirdfiction | 235 | 130 | 28 | 49 | 18 | 10 | 0 | 77 | 0.553 | 0.328 |

The lanes that most need magic (T1 plus T2 share) are SCP (45%), superpowers (44%), folklore (33%) and bestiaries and litrpg_cultivation (about 34%). Creepypasta (17%), dreams and fairy tales (18%, with the 69 Backrooms rows counted as U) and sci-fi (20%) need the least. The most doorway-bound lane is dreams, fairy tales and liminal spaces (68 T3 rows); the most non-physical are litrpg_cultivation (62 X) and the video-game lane (59 X).

Lane against addition, counting T1 and T2 rows (A1 is implied for every row):

| lane | A1 | A5 | A6 | A7 | A8 | A9 | A10 |
|---|---|---|---|---|---|---|---|
| bestiaries | 90 | 2 | 8 | 24 | 5 | 0 | 0 |
| creepypasta | 48 | 1 | 20 | 19 | 4 | 0 | 0 |
| cyberpunk_transhumanism_postapoc_mecha | 135 | 4 | 2 | 36 | 2 | 0 | 0 |
| dreams_fairy_tales_and_liminal | 72 | 0 | 3 | 13 | 0 | 0 | 0 |
| folklore | 137 | 0 | 23 | 45 | 7 | 0 | 0 |
| horror | 89 | 1 | 18 | 37 | 2 | 0 | 0 |
| litrpg_cultivation | 115 | 0 | 1 | 19 | 1 | 0 | 0 |
| mythology_pantheons_religion | 109 | 0 | 0 | 35 | 13 | 0 | 0 |
| scifi | 58 | 1 | 1 | 6 | 0 | 0 | 0 |
| scp | 140 | 0 | 2 | 15 | 38 | 30 | 0 |
| superpowers | 172 | 0 | 7 | 21 | 7 | 0 | 1 |
| videogame_mechanics_and_loops | 94 | 1 | 2 | 7 | 0 | 0 | 0 |
| weirdfiction | 77 | 2 | 20 | 28 | 5 | 0 | 0 |

The catalogue has 598 families (the full table is matrix_family_tier.tsv). The 15 with the most T1 plus T2 rows:

| family | rows | T0 | T1 | T2 | T3 | X | U | T1+T2 |
|---|---|---|---|---|---|---|---|---|
| Spells | 57 | 0 | 46 | 0 | 11 | 0 | 0 | 46 |
| Matter and energy tech | 102 | 66 | 29 | 1 | 4 | 2 | 0 | 30 |
| Cultivation concepts | 47 | 10 | 23 | 4 | 10 | 0 | 0 | 27 |
| Magic (Shadowrun) | 26 | 1 | 16 | 8 | 1 | 0 | 0 | 24 |
| Force | 23 | 2 | 20 | 1 | 0 | 0 | 0 | 21 |
| Occult | 46 | 21 | 14 | 6 | 0 | 5 | 0 | 20 |
| Grimm | 60 | 37 | 14 | 4 | 5 | 0 | 0 | 18 |
| Motif D | 21 | 0 | 15 | 3 | 3 | 0 | 0 | 18 |
| Zone artifacts (STALKER) | 19 | 1 | 18 | 0 | 0 | 0 | 0 | 18 |
| Mythological objects | 34 | 9 | 15 | 1 | 7 | 2 | 0 | 16 |
| Undead | 26 | 10 | 10 | 6 | 0 | 0 | 0 | 16 |
| Body | 56 | 40 | 12 | 3 | 1 | 0 | 0 | 15 |
| undead | 24 | 9 | 10 | 5 | 0 | 0 | 0 | 15 |
| Lovecraft concepts | 36 | 15 | 5 | 9 | 5 | 2 | 0 | 14 |
| Miracles | 34 | 18 | 13 | 1 | 1 | 1 | 0 | 14 |

Families of 8 or more rows that are almost wholly ordinary physics (highest T0 share):

| family | rows | T0 | T1+T2 | share T0 |
|---|---|---|---|---|
| beast | 29 | 29 | 0 | 1.0 |
| Wasteland creatures (Fallout) | 18 | 18 | 0 | 1.0 |
| Divination | 16 | 16 | 0 | 1.0 |
| Social | 15 | 15 | 0 | 1.0 |
| Wasteland systems (Fallout) | 15 | 15 | 0 | 1.0 |
| Psych | 14 | 14 | 0 | 1.0 |
| Cryptid | 24 | 23 | 1 | 0.958 |
| Fortean | 19 | 18 | 1 | 0.947 |
| Metro mutants | 13 | 12 | 1 | 0.923 |
| Internet legends | 12 | 11 | 0 | 0.917 |

Families of 8 or more rows with the highest broken-law (T3) share:

| family | rows | T3 | share |
|---|---|---|---|
| Backrooms levels | 37 | 37 | 1.0 |
| Time loops | 12 | 10 | 0.83 |
| Backrooms | 16 | 13 | 0.81 |
| Time | 14 | 11 | 0.79 |
| FTL and spacetime | 21 | 15 | 0.71 |
| Resurrection | 11 | 7 | 0.64 |
| Omni | 11 | 5 | 0.45 |
| Reality | 26 | 11 | 0.42 |
| LitRPG | 8 | 3 | 0.38 |
| Space | 25 | 9 | 0.36 |

## 4. Coverage matrix: which rows need which addition

A row "needs" an addition when the classifier listed it. A1 is implied for every T1 and T2 row. "Only it" means the row needs that addition and nothing else beyond A1. The last two columns are T3 rows whose bounded lawful version lists the addition.

| addition | what it is | rows needing it (T1+T2) | of which T1 | of which T2 | rows needing ONLY it beyond A1 | low confidence | T3 listing | T3 with a bounded version |
|---|---|---|---|---|---|---|---|---|
| A1 | base gauge field, three portals | 1264 | 806 | 458 | 806 | 642 | 57 | 55 |
| A2 | magnetic charge, dyons | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| A3 | condition-dependent charge-2 pairing | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| A4 | non-abelian symmetry | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| A5 | field self-interaction | 11 | 0 | 11 | 7 | 11 | 0 | 0 |
| A6 | lasting imprints | 95 | 0 | 95 | 70 | 61 | 9 | 6 |
| A7 | strong mind-field coupling | 275 | 0 | 275 | 240 | 221 | 10 | 9 |
| A8 | doorway | 84 | 0 | 84 | 72 | 29 | 192 | 177 |
| A9 | content-selective neural effect (proposed, not adopted) | 30 | 0 | 30 | 30 | 30 | 0 | 0 |
| A10 | selective contact-force suppression (proposed, not adopted) | 1 | 0 | 1 | 1 | 1 | 0 | 0 |

Rows by exact need pattern (T1 and T2; the full list is matrix_need_patterns.tsv):

| needs | rows |
|---|---|
| A1 | 806 |
| A1,A7 | 240 |
| A1,A8 | 72 |
| A1,A6 | 70 |
| A1,A9 | 30 |
| A1,A6,A7 | 24 |
| A1,A7,A8 | 9 |
| A1,A5 | 7 |
| A1,A5,A7 | 2 |
| A1,A5,A8 | 2 |
| A1,A6,A8 | 1 |
| A1,A10 | 1 |

The T2 rows split into one dominant combination, A1 with A7 (240 rows), then A8 (72), A6 (70), the A9 group (30, section 2), A6 with A7 (24), A7 with A8 (9) and a handful needing A5. Two-addition rows are few (38 of 458 T2), so the additions are nearly independent: adopting one rarely depends on another.

Mechanism kinds against addition. The kind is a keyword bucket over the row's entry, mechanism and law text, a coarse proxy that I wrote (the rules are in consol.py); "other kind" is every row the keywords did not catch, mostly rows whose mechanism is only "(u)".

| kind | A1 | A5 | A6 | A7 | A8 | A9 | A10 |
|---|---|---|---|---|---|---|---|
| creature and nature | 19 | 0 | 0 | 3 | 0 | 0 | 0 |
| doorway and place | 109 | 3 | 2 | 25 | 73 | 0 | 0 |
| energy and elements | 88 | 2 | 0 | 3 | 2 | 0 | 0 |
| force and motion | 99 | 1 | 3 | 1 | 0 | 0 | 0 |
| ghost, spirit and patron | 287 | 1 | 53 | 188 | 2 | 2 | 0 |
| healing and body | 108 | 0 | 3 | 2 | 0 | 0 | 1 |
| imprint, curse and rune | 27 | 0 | 10 | 1 | 0 | 0 | 0 |
| matter and making | 105 | 1 | 4 | 6 | 0 | 0 | 0 |
| mind and memory | 138 | 1 | 0 | 21 | 1 | 27 | 0 |
| other kind | 201 | 2 | 13 | 18 | 6 | 1 | 0 |
| sensing and information | 44 | 0 | 3 | 4 | 0 | 0 | 0 |
| time and fate | 39 | 0 | 4 | 3 | 0 | 0 | 0 |

## 5. Minimal set cover and the coverage curve

The cover is over the T1 and T2 rows. A row is covered by a set of additions when every addition it needs is in the set, so this is a set-inclusion count, not an "any one of" cover. With 9 candidates beyond A1 there are only 512 subsets, so I enumerated all of them: the results are exact, not greedy. Ties are broken by lower cost (cost table in section 6).

**Smallest sets that reach a coverage target** (as classified, 1,264 rows, including A9 and A10):

| target | smallest set (beyond A1) | rows covered | share |
|---|---|---|---|
| 90% | A1 + A6, A7 | 1140 | 90.2% |
| 95% | A1 + A6, A7, A8 | 1222 | 96.7% |
| 97% | A1 + A5, A6, A7, A8 | 1233 | 97.5% |
| 99% | A1 + A6, A7, A8, A9 | 1252 | 99.1% |
| all rows | A1 + A5, A6, A7, A8, A9, A10 | 1264 | 100.0% |

To cover every T1 and T2 row exactly, the set is A1, A5, A6, A7, A8, A9, A10: seven additions. With my proposal from section 2 (A9 rows sent to T0, T3 or X by the owner, A10 folded into A8) the universe is 1,234 rows and the minimal exact cover is A1, A5, A6, A7, A8: five. Without A5 it covers 1,223 of 1,234 (99.1%). A2, A3 and A4 appear in no minimal set because no row needs them.

**Coverage curve** (the best set of each size, exact, as classified):

| additions beyond A1 | best set | T1+T2 rows enabled | share | extra cost units |
|---|---|---|---|---|
| 0 | A1 | 806 | 63.8% | 0 |
| 1 | A1,A7 | 1046 | 82.8% | 4 |
| 2 | A1,A6,A7 | 1140 | 90.2% | 7 |
| 3 | A1,A6,A7,A8 | 1222 | 96.7% | 11 |
| 4 | A1,A6,A7,A8,A9 | 1252 | 99.1% | 15 |
| 5 | A1,A5,A6,A7,A8,A9 | 1263 | 99.9% | 18 |
| 6 | A1,A5,A6,A7,A8,A9,A10 | 1264 | 100.0% | 20 |

The curve is the same as the greedy order, so greedy loses nothing here (coverage_greedy.tsv): A1 806 (63.8%), then A7 (82.8%), A6 (90.2%), A8 (96.7%), A9 (99.1%), A5 (99.9%), A10 (100%). The first row is partly tautological (T1 is defined as needing only A1). The informative reading is the T2 part: after A1, A7 enables 240 rows, A6 a further 94 (including A6 with A7), A8 a further 82, A5 about 11 in all.

**Sensitivity to confidence.** Restricting to the 622 high and medium confidence rows changes the order: A1 alone then covers 77.0%, the best single addition is A8 (85.9%), then A7 (94.5%), and A1, A6, A7, A8 cover all 622. So the three-addition conclusion is robust, but the rank of A7 is not: most of A7's rows are low confidence. Proposal-adjusted figures (A9 dropped, A10 into A8):

| set | rows (as classified) | share of 1,264 | rows (adjusted) | share of 1,234 |
|---|---|---|---|---|
| A1 only | 806 | 63.8% | 806 | 65.3% |
| A1, A7 | 1046 | 82.8% | 1046 | 84.8% |
| A1, A8 | 878 | 69.5% | 879 | 71.2% |
| A1, A6 | 876 | 69.3% | 876 | 71.0% |
| A1, A6, A7, A8 | 1222 | 96.7% | 1223 | 99.1% |
| A1, A5, A6, A7, A8 | 1233 | 97.5% | 1234 | 100.0% |

## 6. Variety per addition and variety per unit of cost

**Cost.** The method doc defines cost as new fields, charges, terms and free numbers. It gives no figures, so these are my estimates for the owner to correct: each field, charge, term and number counts 1. A1 is large because it brings the field, its electric charge, three portal terms plus the brain coupling and the weak back-action, and the matching numbers. The ranking below is also shown with every addition costed equally, to see whether the estimates matter.

| addition | fields | charges | terms | numbers | total |
|---|---|---|---|---|---|
| A1 | 1 | 1 | 5 | 5 | 12 |
| A2 | 0 | 1 | 1 | 2 | 4 |
| A3 | 1 | 1 | 1 | 2 | 5 |
| A4 | 1 | 1 | 2 | 2 | 6 |
| A5 | 0 | 0 | 1 | 2 | 3 |
| A6 | 0 | 0 | 1 | 2 | 3 |
| A7 | 0 | 0 | 2 | 2 | 4 |
| A8 | 0 | 0 | 2 | 2 | 4 |
| A9 | 0 | 0 | 2 | 2 | 4 |
| A10 | 0 | 0 | 1 | 1 | 2 |

**Unlocked variety.** For each addition: the T1 and T2 rows it unlocks, the distinct families (of 598) and distinct mechanism kinds (of 12, the keyword proxy) among them. "Only" restricts to rows needing nothing else beyond A1, which is the variety the addition opens by itself.

| addition | cost | rows | families | kinds | families (only it) | kinds (only it) | T3 rows given a bounded version |
|---|---|---|---|---|---|---|---|
| A1 | 12 | 1264 | 347 | 12 | 226 | 12 | 55 |
| A2 | 4 | 0 | 0 | 0 | 0 | 0 | 0 |
| A3 | 5 | 0 | 0 | 0 | 0 | 0 | 0 |
| A4 | 6 | 0 | 0 | 0 | 0 | 0 | 0 |
| A5 | 3 | 11 | 9 | 7 | 6 | 5 | 0 |
| A6 | 3 | 95 | 53 | 9 | 41 | 9 | 6 |
| A7 | 4 | 275 | 124 | 12 | 113 | 11 | 9 |
| A8 | 4 | 84 | 69 | 5 | 62 | 5 | 177 |
| A9 | 4 | 30 | 28 | 3 | 28 | 3 | 0 |
| A10 | 2 | 1 | 1 | 1 | 1 | 1 | 0 |

**Ranking by unlocked families per cost unit** (the main measure; kinds per cost and the equal-cost ranking beside it):

| rank | addition | cost | families | families per cost | kinds per cost | rank by kinds per cost | rank if all costs equal (by families) |
|---|---|---|---|---|---|---|---|
| 1 | A7 | 4 | 124 | 31.0 | 3.0 | 1 | 2 |
| 2 | A1 | 12 | 347 | 28.92 | 1.0 | 5 | 1 |
| 3 | A6 | 3 | 53 | 17.67 | 3.0 | 2 | 4 |
| 4 | A8 | 4 | 69 | 17.25 | 1.25 | 4 | 3 |
| 5 | A9 | 4 | 28 | 7.0 | 0.75 | 6 | 5 |
| 6 | A5 | 3 | 9 | 3.0 | 2.33 | 3 | 6 |
| 7 | A10 | 2 | 1 | 0.5 | 0.5 | 7 | 7 |

A7 buys the most families per unit of cost (31, against 28.9 for A1 itself); A6 and A8 are next at about 17.5; A5, A9 and A10 are poor. On kinds per cost A6 and A7 tie at 3.0 (A6: 95 rows over nine kinds, A7: 275 rows over all twelve), then A5 at 2.3; the kinds measure is coarse and the spread of A7 is partly because the keyword rules put many ghost-like entries in other buckets. A8 opens only about five kinds (doorways, planes, teleport, travel) but is the only addition that gives bounded versions to T3 rows: 177 of the 371 T3 rows list it with a lawful bounded version, which is the biggest effect of any addition on the broken-law class. With all costs equal the order by families becomes A1, A7, A8, A6, A9, A5, A10: A1 and A7 stay on top and A8 passes A6, so the top group does not depend on my cost estimates. A2, A3 and A4 rank last with zero rows. "Variety" here counts catalogue rows and families; it is a proxy for demand, not for what composition will produce in the simulations, so it undercounts any addition whose gain is in combinations (A2 to A4 may be like that).

## 7. The case bank

All 135 cases (C01 to C135) are classified in cases_classified.tsv with the doc's expected outcome beside them. Tiers: T0 23, T1 106, T2 1, T3 5, X 0. Additions: 106 cases need only A1, 1 needs A7 (C19), 1 needs A8 (C24), and 27 need none. The case bank is nearly all A1 because it was written from the decided physics, so it confirms the One Change on the cases (133 of 135, 98.5%, need at most A1 or are impossible by law; the other two are C19 and C24) but says little about the further additions.

The five T3 cases are as the doc says: C24 (teleport without a doorway: a doorway pack needs A8), C35 (a month of weather, beyond chaos), C116 (an exact copy of a person, impossible by information), C124 (a false memory, writes beliefs) and C131 (raising the dead, unless a pack adds it). C45 is low confidence and the doc leaves it open.

**Cases whose expected outcome needs an addition the decisions have not made** (the decided set is A1, plus A2 and A3 as the RQ4 combination with tests pending, plus doorways as packs under Q10 with the mechanism open; A4 to A7 are not decided):

| case | what | class | needs | flag | note |
|---|---|---|---|---|---|
| C05 | A recording of a chant is played | T0 | - | doc outcome conditional or provisional | a recording has no coupled mind, so no field control |
| C19 | A sleeping person casts in a dream | T2 | A7 | needs undecided: A7 | dream casting is undecided and may need strong mind coupling |
| C24 | Someone tries to teleport with no doorway | T3 | A8 | needs doorway pack (Q10): design open | teleport needs a doorway; bounded version with booked energy exists |
| C30 | A natural philosopher carries a fine thermometer along a ley | T1 | A1 | doc outcome conditional or provisional | field losses show as a small warm anomaly, found by thermometer |
| C31 | A wild magical surge in an empty desert, and the same near a | T1 | A1 | doc outcome conditional or provisional | surge energy dissipates as heat or light depending on materials |
| C45 | A crystal-circuit machine with organised currents tries to c | T1 | A1 | doc outcome conditional or provisional | a device with organised currents might couple; undecided |
| C73 | A widow feels her dead husband's presence at home | T0 | - | doc outcome conditional or provisional | grief and memory by default; an imprint would need A6 |
| C80 | People and nodes | T0 | - | doc outcome conditional or provisional | node-seeking only if it serves a drive |
| C131 | Raising the dead | T3 | - | doc outcome conditional or provisional | raising the dead is not a state ordinary physics reaches |

Reading: only C19 (casting in a dream) needs an undecided addition as classified (A7), and the doc already marks it "to be decided". C24 needs A8, which Q10 settles only as packs. C73 (the widow feels her dead husband) classified T0, which matches the doc's "grief and memory by default"; its doc text is conditional on R5 (A6) and on Q2.4 (c) (A7), so the case is a ready test for A6 and A7 once decided. C05, C30, C31, C45, C80 and C131 have provisional or conditional doc outcomes but do not need an undecided addition. C131 needs "a pack"; no addition is listed. Two classifier calls to check: C45 (a crystal-circuit machine casting, T1 at low confidence) and C135 (two resonators feeding each other until the field drains, T1 A1; it looks like a self-sustaining loop but the doc says it stops when the local field is drained, so A5 is not needed).

## 8. Low-confidence rows for the owner

1,032 rows (25.0%) are marked low confidence, listed in full in lowconf.tsv (sorted by tier, then needed additions, then family). By tier: T0 193, T1 327, T2 315, T3 122, X 6, U 69. The share of each tier that is low confidence is shown below; the figures that matter most are the T2 and T1 rows because they decide the cover.

| tier | rows | low confidence | share |
|---|---|---|---|
| T0 | 2149 | 193 | 9.0% |
| T1 | 806 | 327 | 40.6% |
| T2 | 458 | 315 | 68.8% |
| T3 | 371 | 122 | 32.9% |
| X | 275 | 6 | 2.2% |
| U | 69 | 69 | 100.0% |

By addition (T1 and T2 rows only):

| addition | rows needing it | low confidence | share |
|---|---|---|---|
| A1 | 1264 | 642 | 50.8% |
| A5 | 11 | 11 | 100.0% |
| A6 | 95 | 61 | 64.2% |
| A7 | 275 | 221 | 80.4% |
| A8 | 84 | 29 | 34.5% |
| A9 | 30 | 30 | 100.0% |
| A10 | 1 | 1 | 100.0% |

Where to look first, in order of what changes the answer:

1. **The A7 rows (221 low-confidence of 275).** Ghosts, patrons, possession, liches, spirit-substrate beings and believer-fed gods. The classifiers noted that the tier depends on whether the owner treats these as A6 or A7 effects or as belief and culture. This is the question that decides whether A7 is needed. Families with the most A7 rows are in the families table of matrix_family_tier.tsv and lowconf.tsv (filter needs containing A7).
2. **The 30 A9 rows (all low confidence).** Antimemetic and cognitohazard SCP rows; choose T0, T3 or X (section 2).
3. **The 11 A5 rows (all low confidence).** These are the only support for A5 (self-sustaining packets). They are:
   - Colour Out of Space (Lovecraft concepts), needs A1,A5: Light or field outside known states, self-sustaining and spreading, needs field self-interaction.
   - Ghroth (Outer Gods), needs A1,A5: A self-sustaining organised packet needs field self-interaction.
   - Reality glitch (Reality), needs A1,A5: a local zone with different effective constants needs a self-interacting field state
   - The Beams (Dark Tower) (Weird West), needs A1,A5: Lines or beings of self-sustaining field structure need field self-interaction.
   - Crimson King (Weird West), needs A1,A5: Lines or beings of self-sustaining field structure need field self-interaction.
   - Fire elemental (elemental), needs A1,A5: A self-sustaining organised packet needs field self-interaction.
   - Fumecrux, living fumes (spirit), needs A1,A5: A self-sustaining organised packet needs field self-interaction.
   - Energy beings (Aliens and bio), needs A1,A5,A7: A matterless self-sustaining pattern with a mind needs field self-interaction and A7.
   - Sentient sword (intelligent item) (Items), needs A1,A5,A7: a mind in an object needs the mind partly made of a field packet
   - The Warp (Immaterium) (Chaos (Warhammer 40,000)), needs A1,A5,A8: Field self-interaction and a doorway-like region give field storms and gates.
   - Warp storm (Chaos (Warhammer 40,000)), needs A1,A5,A8: Field self-interaction and a doorway-like region give field storms and gates.
4. **The 69 under-specified Backrooms rows (U).** They have no stated effect; fill in or leave out.
5. **The T1 and T2 boundary.** Classifier 2 flagged reshaping, mind steering and ghost-like entities as blurry; 325 T1 rows are low confidence. If many of them are really T2, A1 alone covers less than the first row of the curve says.
6. **Classifier 5 and classifier 3 group assignments.** Classifier 5 admits loose group assignment by index list; classifier 3 assigned tiers in bulk by group, with default tiers for SCP rows with a "(u)" mechanism. Spot checks on their T1 and T2 rows are worth more than on the others.

## 9. What this says about the One Change, and what to decide next

- The expectation holds: after the roughly 57% that is ordinary, the largest magic class is ordinary physics steered by one control channel at a distance plus energy sources (T1, 806 rows, 21% of physical wishes). One change (A1) covers it.
- The residue is small and has a shape: 458 T2 rows, 420 of them (92%) needing at least one of three things (a mind partly made of the field, lasting imprints, doorways), each with a decision already queued (Q2.4 (c), R5, Q10). Two further queued items, A5 (R8) and A9, are tiny.
- The catalogue does not justify A2, A3 or A4 on coverage. If the owner wants them, the reason must come from the research or from what the simulations produce, and this analysis should be rerun on the simulation output (method step 5).
- If the owner decides ghosts and possession are belief (T0), A7 drops out and the cover is A1, A6, A8 (and A5 for a few rows).
- Caveats: all tiers are by classifier judgment on short descriptions, 25% of rows are low confidence, the kinds are a keyword proxy, and the cost units are my estimates.

## Files (all beside this report)

- classified.tsv: all 4,128 rows, all merged.tsv columns plus the classification.
- matrix_family_tier.tsv, matrix_lane_tier.tsv, matrix_lane_addition.tsv: tiers by family and by lane, and lane against addition.
- matrix_additions.tsv, matrix_need_patterns.tsv, matrix_kind_addition.tsv: the addition matrices.
- coverage_curve.tsv, coverage_greedy.tsv: the cover.
- variety_per_cost.tsv, cost_table.tsv: the variety and cost.
- other_additions.tsv: the "other" groups.
- cases_classified.tsv: the case bank.
- lowconf.tsv: low-confidence rows.
- class_3.tsv: an earlier copy of classifier 3 output, left in place.
