# Gap analysis report, version 3: how much of the wanted magic existing physics allows, and what the rest costs

Version 3, 2026-11-27. It applies four more owner rules (M1 to M4, section 2) to every one of the 4,128 rows of version 2, after a check found rule inconsistencies (version 2 passed tier agreement at 94% but treated the same kind of row differently in different places). Version 2 is kept as classified_v2.tsv and report_v2.md (other v2 tables carry the suffix _v2; first-pass tables carry _v1). The method is in docs/magic-gap-analysis.md; the decided physics is in docs/magic-science.md. Documents and small scripts only; no simulation.

## 0. The answer

- Of 4,128 rows, 275 are not physical effects and 83 are too thin to classify, so 3,770 are physical wishes. Tiers now: T0 2,163 (57.4%), T1 787 (20.9%), T2 618 (16.4%), T3 202 (5.4%). Version 2 had T1 804, T2 577, T3 226. 44 rows changed tier; 92 rows changed in tier, needs or A7 class.
- **The exact minimal cover of every T1 and T2 row is still A1, A5, A6, A7, A8** (26 cost units counting A1; 14 beyond A1), under either reading of the owner rows and on all rows. On high and medium confidence rows only it is A1, A6, A7, A8 (A5 has only 11 rows, all low confidence). A2, A3, A4, A9 and A10 are needed by no row.
- **A7 is now a bigger addition, and the owner choice now matters less.** 91 rows need A7 as a mind outside its brain (rule M1, bounded version lossy), 15 need it as strong coupling between minds (rule M2), and 179 stay "A7-or-T0 (owner)" because belief, culture or A1 percepts can explain them. A7 serves 285 rows if the owner rows count as A7, and 106 rows if the owner reads them as belief (version 2: 246 and 19). The owner choice moves 157 rows between T0 and T2 (version 2: 197).
- Curve, owner rows as A7: A1 alone enables 787 of 1405 T1+T2 rows (56.0%); A8 is the best single addition (73.6%), then A7 (92.0%), then A6 (99.2%), then A5 (100%). Owner rows as T0: A1 63.1%, A8 83.3%, A7 91.0%, A6 99.1%. On high and medium rows, A1, A6, A7, A8 reaches 100% in both readings.
- Variety per cost: A7 (31.75 families per unit) now edges A8 (29.75) when owner rows count; if the owner reads ghosts as belief, A7 falls to 14.75 and A8 leads clearly.
- Confidence: low-confidence rows among T1 and T2 are 476 of 1405 (33.9%; version 2: 466 of 1381, 33.7%). The rows moved by M1 to M4 keep the confidence they had; none was raised. Tier agreement (94%, version 2) was measured before these rules and has not been re-measured.

## 1. What changed from v2

44 of 4,128 rows changed tier (1.1%); 48 more changed needs or A7 class without changing tier. Every changed row is listed in moved_v2_to_v3.tsv and carries its rule (rules), the old tier (tier_v2) and a note (note3) in classified.tsv. The 19 rows already strict in version 2 keep their tier; they now record bounded = lossy.

| v2 tier to v3 tier | rows |
|---|---|
| T3 to T2 | 26 |
| T1 to T2 | 16 |
| T1 to T3 | 1 |
| T2 to T3 | 1 |

| tier | v2 | v3 | change |
|---|---|---|---|
| T0 | 2163 | 2163 | +0 |
| T1 | 804 | 787 | -17 |
| T2 | 577 | 618 | +41 |
| T3 | 226 | 202 | -24 |
| X | 275 | 275 | +0 |
| U | 83 | 83 | +0 |

| rule | rows touched | what it did |
|---|---|---|
| M1 minds outside brains | 72 | 47 owner rows to strict A7, 17 T3 rows to T2 A7 (uploads, transfers, merging, stored minds, a thinking object), 1 T1 row to T2 A7 (Freddy Krueger, also M2), plus 19 rows already strict that now record bounded = lossy. The 7 doorway-adjacent rows below also carry M1. |
| M2 T1 reasons naming A7, A6 | 17 | 14 T1 rows to T2 A7 (strong coupling beyond A1's ports), 1 T1 row to T2 A6 (Mongo), and 1 T2 row (Dream Cycle) gaining A7 beside A8. |
| M3 doorway-adjacent | 9 | 7 rows to T2 A1+A7+A8 (lossy mind record plus the original body through a doorway) and 2 rows to T2 A1+A8 (Chuanyue, Matter transmitter); the unbounded version stays T3. |
| M4 memory erasure of strangers | 2 | SCP-6766 (T2 owner row to T3) and SCP-2571 (T1 to T3). |

## 2. The four new rules as applied

- **M1 minds outside brains.** A mind that persists, moves or acts outside its brain (a ghost that thinks, a lich and phylactery, a soul stone, body swap, transmigration, reincarnation with memory, mind upload, merging, a thinking object, possession by a disembodied mind) is T2 with A7, and its bounded version is "lossy" (Q0: an exact copy needs the full information, so exact duplication or perfect transfer is the unbounded T3 version). Rows whose effects are explainable as belief, culture or A1 effects stay "A7-or-T0 (owner)". Columns for these rows: bounded = lossy, bounded_tier = T2, unbounded_version = exact duplication or perfect transfer of the full information, unbounded_tier = T3, A7class = strict.
- **M2 T1 rows naming A7, A6, A8 or strongly coupled minds in their own reason** are re-tiered to match the reason. A7class = coupling for rows whose effect is a strong coupling between minds (a shared dream, a link into a sleeper, a bonded familiar, mind control, a merged group); these need A7 under either owner reading.
- **M3 doorway-adjacent rows** (Chuanyue, Soulkiller, Agent Smith, Teletransportation paradox, Digital immortality and those like them) apply R1 (a bounded lawful version with A8 is T2, otherwise T3) and M1 together: a lossy mind record (A7) and the original body through a flux-carrying doorway (A8) is T2 A1+A7+A8; an exact copy or perfect transfer is T3.
- **M4 content-selective memory erasure of strangers** is T3 by R5 (it needs a model of that brain's code), whatever else the row is.

How I drew the M1 line against the owner rows (the judgement the rule leaves open): a row is strict when its own text names (a) a disembodied mind possessing, riding or seizing a host, (b) a ghost, soul or spirit of a particular dead or living person acting with its own intent (a revenant, a vengeful ghost, a ghost lover from the grave), (c) reincarnation or transmigration with memory, (d) a mind in an object or place-bound carrier (doll, mask, skull, box, machine, a thinking snowman), (e) a mind leaving its body (astral projection, soul flight), or (f) a stolen or stored soul. A row stays an owner row when it is a patron, god, angel, outsider demon, elemental, nature spirit or cosmic entity, a hunt, omen or procession, a haunting loop that only replays, a poltergeist force, a drain, an exorcism or rite (mediumship, planchette, evocation), a realm, or a pact, because apparitions, a changed host, power that tracks a cult and rites are reproducible by A1 percepts, A1 control and belief. T0 rows that are plain belief, folklore or ordinary engineering (for example Zar possession, Wendigo, Android, Brain in a vat) were reviewed and left alone.

### Every row M1 moved

**Owner rows to strict A7 (47):** Annabelle; Astral projection; BEN Drowned; Beyond the Wall of Sleep; Brahmarakshasa; Churel; Daemonic possession; Deadites; Doppelgeist; Dybbuk; Dybbuk box; E200 Malevolent return from the dead; E300 Friendly return from the dead; E600 Reincarnation; Hantu, hantu-like vengeance spirits; Haunted clockwork and wyrwood; Human possession in science fiction; Hungry ghost; Kitsunetsuki; Ligeia; Lost Hearts; Majora's Mask; Morella; Multi-Possession; Onryō; Possessed (Chaos); Possession; Possession by a demon; Reincarnation; Revenant (Weird West); Robert the Doll; Rusalka; SCP-035 Possessive Mask; SCP-2113 Haunted Liopleurodon Skull; SCP-4161 METEMPSYCHOSIS.; Seizing a body; Soul flight; Soul theft (sleep, picture); Soul-Puppetry; The Ghost Pirates; The Horla; The Thing on the Doorstep; Toyol; Vengeful ghost; Vetala; Visitant, soulrider; Wiedergaenger.

**T3 to T2 A1+A7, lossy (17):** Andersen: The Snowman; Bilocation; Cortical stack and re-sleeving; Duplicate Merging; Eldar soulstone; Fairy tale: Koschei the Deathless; Great Race of Yith; Immortal by phylactery; Infomorph; Mind transfer (body swap); Mind uploading; Mind uploading in fiction; Reincarnation with memory; The Shadow out of Time; Transmigration; Uploaded mind horror; Whole brain emulation.

**T1 to T2 A1+A7, lossy (1):** Freddy Krueger.

**Already strict in v2, now recording bounded = lossy (19):** Body swap; Chucky (killer doll); E700 The soul as bird/insect/breath; E710 External soul (Koschei, heart hidden); Energy beings; Golden Throne; Human Instrumentality Project; Jikininki; Ka and Ba; Koschei the Deathless; Lich; Nascent soul; Nascent soul escape; Possessed car (Christine); Revenant; Sentient sword (intelligent item); Slappy the Dummy; Soul-eater (soul stealing); Wicked Blade.

### Every row M2, M3 and M4 moved

**M2 T1 to T2 A1+A7, coupling (14):** Astral space; Bird Box; Cintamani; Controller; Dominate person; Dream and nightmare creatures; Dream sharing machine; Familiar (witch's animal); Familiar (wizard); Inception (film); Mind Control; Paprika (anime); Shared dream; Succubus / incubus. **M2 T1 to T2 A1+A6 (1):** Mongo. **M2 T2 gains A7 (1):** Dream Cycle (Lovecraft).

**M3 T3 to T2 A1+A7+A8 (7):** Agent Smith; Altered Carbon (stacks and sleeves); Digital immortality; Puppet Master (Ghost in the Shell); Relic engram (Johnny Silverhand); Soulkiller (engram upload that kills); Teletransportation paradox. **M3 T3 to T2 A1+A8 (2):** Chuanyue; Matter transmitter.

**M4 to T3 (2):** SCP-2571 Cragglewood Park; SCP-6766 To Know All Stones.

## 3. Tiers by family and lane

Tier counts use the primary tier, so owner rows are counted as T2 here (the owner column counts them). Rows can sit in several lanes, so lane totals exceed 4,128. Full family table: matrix_family_tier.tsv.

| lane | rows | T0 | T1 | T2 | T3 | X | U | owner | T1+T2 | share T0 | share T1+T2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bestiaries | 260 | 159 | 53 | 40 | 5 | 3 | 0 | 16 | 93 | 0.612 | 0.358 |
| creepypasta | 277 | 200 | 8 | 58 | 5 | 6 | 0 | 13 | 66 | 0.722 | 0.238 |
| cyberpunk_transhumanism_postapoc_mecha | 452 | 290 | 86 | 58 | 12 | 6 | 0 | 27 | 144 | 0.642 | 0.319 |
| dreams_fairy_tales_and_liminal | 390 | 151 | 46 | 73 | 11 | 29 | 80 | 5 | 119 | 0.387 | 0.305 |
| folklore | 411 | 255 | 71 | 69 | 8 | 8 | 0 | 25 | 140 | 0.62 | 0.341 |
| horror | 292 | 183 | 36 | 59 | 6 | 8 | 0 | 18 | 95 | 0.627 | 0.325 |
| litrpg_cultivation | 337 | 125 | 94 | 38 | 18 | 62 | 0 | 10 | 132 | 0.371 | 0.392 |
| mythology_pantheons_religion | 345 | 178 | 61 | 44 | 21 | 41 | 0 | 34 | 105 | 0.516 | 0.304 |
| scifi | 285 | 185 | 50 | 21 | 15 | 14 | 0 | 0 | 71 | 0.649 | 0.249 |
| scp | 311 | 126 | 62 | 59 | 43 | 18 | 3 | 12 | 121 | 0.405 | 0.389 |
| superpowers | 392 | 160 | 140 | 44 | 35 | 13 | 0 | 12 | 184 | 0.408 | 0.469 |
| videogame_mechanics_and_loops | 325 | 117 | 83 | 38 | 28 | 59 | 0 | 3 | 121 | 0.36 | 0.372 |
| weirdfiction | 235 | 130 | 28 | 59 | 8 | 10 | 0 | 19 | 87 | 0.553 | 0.37 |

Lane against addition (T1 and T2 rows, owner rows counted as A7; A1 is implied for all):

| lane | A1 | A5 | A6 | A7 | A8 |
|---|---|---|---|---|---|
| bestiaries | 93 | 2 | 8 | 24 | 8 |
| creepypasta | 66 | 1 | 20 | 19 | 23 |
| cyberpunk_transhumanism_postapoc_mecha | 144 | 4 | 3 | 44 | 16 |
| dreams_fairy_tales_and_liminal | 119 | 0 | 6 | 12 | 59 |
| folklore | 140 | 0 | 23 | 45 | 10 |
| horror | 95 | 1 | 18 | 38 | 7 |
| litrpg_cultivation | 132 | 0 | 2 | 17 | 19 |
| mythology_pantheons_religion | 105 | 0 | 0 | 35 | 14 |
| scifi | 71 | 1 | 1 | 9 | 11 |
| scp | 121 | 0 | 2 | 15 | 42 |
| superpowers | 184 | 0 | 8 | 20 | 19 |
| videogame_mechanics_and_loops | 121 | 1 | 2 | 8 | 28 |
| weirdfiction | 87 | 2 | 20 | 28 | 13 |

## 4. The owner's one choice: A7 or T0

179 rows are A7-or-T0 (owner) (v2: 227; 47 moved to strict by M1, 1 to T3 by M4, and the rest are patrons, gods, angels, outsider spirits and the like). Reading them as T0 removes A7 from 157 rows outright; 22 keep a T2 tier through A6 or A8. The largest families among them: Chaos (Warhammer 40,000) (7), Great Old Ones (6), Jinn (6), Ceremonial magic (6), Magic (Shadowrun) (6), Weird West (6), Ghosts (5), Angels and demons (5).

| reading of the owner rows | T0 | T1 | T2 | T1+T2 | A7 needed by | minimal cover |
|---|---|---|---|---|---|---|
| owner picks A7 (rows need A7) | 2163 | 787 | 618 | 1405 | 285 rows (91 strict + 15 coupling + 179 owner) | A1, A5, A6, A7, A8 |
| owner picks T0 (belief reading) | 2320 | 787 | 461 | 1248 | 106 rows (91 strict + 15 coupling) | A1, A5, A6, A7, A8 |

The choice now moves 157 rows between T0 and T2. It no longer decides whether A7 is a small or a large addition: even on the belief reading A7 serves 106 rows and 59 families, and it is in the minimal cover under both readings and on high and medium rows. Listing: classified.tsv, filter label = "A7-or-T0 (owner)".

## 5. Coverage curve

For each number k of additions beyond A1, the best set (most rows enabled, ties by lower cost) from the 7 candidates A2 to A8, by exhaustive search. A row is enabled when every addition it needs is in the set. Cost is in the units of cost_table.tsv (unchanged). A9 and A10 are not candidates because no row needs them. Both owner readings, on all rows and on high plus medium confidence rows.

**owner rows counted as needing A7 (primary).** 1405 T1+T2 rows; 929 of them high or medium confidence. File coverage_curve.tsv.

| k | best set (all rows) | rows | share | cost beyond A1 | best set (high+medium only) | rows | share |
|---|---|---|---|---|---|---|---|
| 0 | A1 | 787 | 56.0% | 0 | A1 | 678 | 73.0% |
| 1 | A1,A8 | 1034 | 73.6% | 4 | A1,A8 | 828 | 89.1% |
| 2 | A1,A7,A8 | 1293 | 92.0% | 8 | A1,A7,A8 | 895 | 96.3% |
| 3 | A1,A6,A7,A8 | 1394 | 99.2% | 11 | A1,A6,A7,A8 | 929 | 100.0% |
| 4 | A1,A5,A6,A7,A8 | 1405 | 100.0% | 14 | A1,A5,A6,A7,A8 | 929 | 100.0% |
| 5 | A1,A2,A5,A6,A7,A8 | 1405 | 100.0% | 18 | A1,A2,A5,A6,A7,A8 | 929 | 100.0% |

Greedy order (all rows): A1 (787), A8 (1034), A7 (1293), A6 (1394), A5 (1405). High+medium: A1 (678), A8 (828), A7 (895), A6 (929).

**owner rows read as belief (T0).** 1248 T1+T2 rows; 902 of them high or medium confidence. File coverage_curve_ownerT0.tsv.

| k | best set (all rows) | rows | share | cost beyond A1 | best set (high+medium only) | rows | share |
|---|---|---|---|---|---|---|---|
| 0 | A1 | 787 | 63.1% | 0 | A1 | 678 | 75.2% |
| 1 | A1,A8 | 1040 | 83.3% | 4 | A1,A8 | 828 | 91.8% |
| 2 | A1,A7,A8 | 1136 | 91.0% | 8 | A1,A7,A8 | 868 | 96.2% |
| 3 | A1,A6,A7,A8 | 1237 | 99.1% | 11 | A1,A6,A7,A8 | 902 | 100.0% |
| 4 | A1,A5,A6,A7,A8 | 1248 | 100.0% | 14 | A1,A5,A6,A7,A8 | 902 | 100.0% |
| 5 | A1,A2,A5,A6,A7,A8 | 1248 | 100.0% | 18 | A1,A2,A5,A6,A7,A8 | 902 | 100.0% |

Greedy order (all rows): A1 (787), A8 (1040), A7 (1136), A6 (1237), A5 (1248). High+medium: A1 (678), A8 (828), A7 (868), A6 (902).

Compared with v2 (A1 58.2%, A8 76.0%, A7 92.0%, A6 99.2% with owner rows as A7; A1 67.9%, A8 89.2%, A6 97.6%, A7 99.1% with owner rows as T0): the order is now A1, A8, A7, A6, A5 in both readings. In the belief reading A7 now comes before A6 because M1 and M2 added 87 A7 rows that do not depend on the owner's choice; in v2 A6 was second.

## 6. The exact minimal cover

Each row needs all of its listed additions together, so the smallest set that covers every T1 and T2 row is the union of what the rows need; the exhaustive search over all 128 subsets of A2 to A8 confirms nothing smaller covers 100%.

| rows | scenario | exact minimal set | cheapest set for 90% | 95% | 99% | 100% |
|---|---|---|---|---|---|---|
| all T1+T2 | owner = A7 | A1, A5, A6, A7, A8 | A1, A7, A8 | A1, A6, A7, A8 | A1, A6, A7, A8 | A1, A5, A6, A7, A8 |
| all T1+T2 | owner = T0 | A1, A5, A6, A7, A8 | A1, A6, A8 | A1, A6, A7, A8 | A1, A6, A7, A8 | A1, A5, A6, A7, A8 |
| high+medium only | owner = A7 | A1, A6, A7, A8 | A1, A6, A8 | A1, A7, A8 | A1, A6, A7, A8 | A1, A6, A7, A8 |
| high+medium only | owner = T0 | A1, A6, A7, A8 | A1, A8 | A1, A6, A8 | A1, A6, A7, A8 | A1, A6, A7, A8 |

A5 stays in the full cover for 11 rows, every one low confidence (Colour Out of Space, Ghroth, the Warp, warp storms, the Beams, the Crimson King, Fire elemental, Fumecrux, Reality glitch, Energy beings, Sentient sword); it has no support from high or medium rows. The need patterns are in matrix_need_patterns.tsv:

| needs | rows (owner = A7) | rows (owner = T0) |
|---|---|---|
| A1 | 787 | 787 |
| A1,A8 | 247 | 253 |
| A1,A7 | 245 | 88 |
| A1,A6 | 73 | 89 |
| A1,A6,A7 | 24 | 8 |
| A1,A7,A8 | 14 | 8 |
| A1,A5 | 7 | 7 |
| A1,A6,A8 | 4 | 4 |
| A1,A5,A7 | 2 | 2 |
| A1,A5,A8 | 2 | 2 |

## 7. Variety per addition

Cost units are unchanged. "Rows" counts T1 and T2 rows needing the addition; "families" the distinct catalogue families of 598; "kinds" the 12 keyword mechanism kinds (a coarse proxy). "Only" is rows needing nothing beyond A1 plus this addition. Owner = A7 scenario:

| addition | cost | rows | families | kinds | only rows | only families | families per cost | kinds per cost |
|---|---|---|---|---|---|---|---|---|
| A1 | 12 | 1405 | 334 | 12 | 787 | 233 | 27.83 | 1.0 |
| A5 | 3 | 11 | 9 | 7 | 7 | 6 | 3.0 | 2.33 |
| A6 | 3 | 101 | 57 | 10 | 73 | 44 | 19.0 | 3.33 |
| A7 | 4 | 285 | 127 | 12 | 245 | 118 | 31.75 | 3.0 |
| A8 | 4 | 267 | 119 | 9 | 247 | 111 | 29.75 | 2.25 |
| A7 strict only | 4 | 91 | 50 | 7 | 74 | 47 | 12.5 | 1.75 |
| A7 coupling only | 4 | 15 | 12 | 6 | 14 | 12 | 3.0 | 1.5 |
| A7 owner rows only | 4 | 179 | 94 | 12 | 157 | 84 | 23.5 | 3.0 |

Owner = T0 scenario (A7 keeps its strict and coupling rows):

| addition | cost | rows | families | kinds | families per cost | kinds per cost |
|---|---|---|---|---|---|---|
| A1 | 12 | 1248 | 310 | 12 | 25.83 | 1.0 |
| A5 | 3 | 11 | 9 | 7 | 3.0 | 2.33 |
| A6 | 3 | 101 | 57 | 10 | 19.0 | 3.33 |
| A7 | 4 | 106 | 59 | 8 | 14.75 | 2.0 |
| A8 | 4 | 267 | 119 | 9 | 29.75 | 2.25 |
| A7 strict only | 4 | 91 | 50 | 7 | 12.5 | 1.75 |
| A7 coupling only | 4 | 15 | 12 | 6 | 3.0 | 1.5 |

Reading. With owner rows counted, A7 (31.75 families per unit, 127 families) edges A8 (29.75, 119 families) and both lead A6 (19.0) and A5 (3.0); A1 is 27.83 for 12 units. A6 opens the most kinds per cost (3.33). On the belief reading A7 falls to 14.75 families per unit (59 families, 8 kinds) and A8 (29.75) leads. This is the same shape as version 2, with the A7 figure higher because M1 and M2 added 87 A7 rows that do not depend on the owner. A8 remains the only addition that turns T3 rows into T2 for doorways (170 rows through R1, plus 9 through M3), and M1 adds a second such step for A7 (24 T3 rows now T2). A2, A3 and A4 unlock no catalogue row; their case rests on the research and on composition in the toy simulations, which the catalogue cannot show.

## 8. Doorways, minds and the antimemetic rows

**Doorways (R1, M3).** T2 rows needing A8: 267 (v2: 258). The 9 M3 rows join them; for each the bounded version is T2 and the unbounded version T3, recorded per row. T3 now has 202 rows: time travel and loops, creation from nothing and omnipotence, exact copies of bodies, true foresight, content-selective brain writes (including SCP-6766 and SCP-2571 now), FTL and metric engineering, reading or writing thoughts of strangers.

**Minds outside brains (M1, M2).** One lawful shape now covers all of these rows: a mind that exists, moves or acts outside its brain needs A7, a lossy record or carrier is the bounded version (T2), and the exact or perfect version is T3 (the same Q0 argument that makes an exact copy impossible). This puts mind upload, whole brain emulation, mind transfer, Great Race of Yith, cortical stacks, phylactery and soul-stone rows, reincarnation with memory, merging and a thinking snowman at T2 A7 together with the lich, the sentient sword and Chucky. Rows that only couple minds (shared dreams, a bonded familiar, mind control, a controller, astral space) are T2 A7 with A7class = coupling because their own reasons name a coupling beyond A1's ports.

**The antimemetic and cognitohazard rows (R5, M4).** After M4: T0 3, T1 7, T3 17, U 3. A9 is not adopted and no row needs it. SCP-2571 and SCP-6766 join the content-selective rows at T3.

## 9. What is still uncertain

Low-confidence rows now: T0 208, T1 109, T2 367, T3 61, X 6, U 83 (total 834). Among T1+T2: 476 of 1405. In order of what changes the answer:

1. **The M1 line against the owner rows** (section 2). The 47 rows moved are my judgement, applied by name; the owner can move rows across by editing A7class. Even if all of them went back to owner rows, the cover would not change; only how many rows A7 serves would.
2. **Some M2 rows are read by their stated reason, not by their effect.** Bird Box (a percept that triggers panic) and Astral space were T1 in v2 for good reasons and are now T2 A7 only because their recorded reason names A7; if the owner prefers the effect over the reason they go back to T1. Dream Cycle gained A7 for the same reason, while the other dream-place rows (Dreamlands, Dreamscape, Dream-Quest) have a reason naming "mind-field coupling" for inhabitants and were left T2 A8 only, since M2 covers T1 rows.
3. **Judgement calls inside M3.** Matter transmitter and Chuanyue were given A8 only (no mind-outside-brain), and the six uploader and copier rows were given A7 and A8 together because their recorded reason names a doorway; this adds A8 to rows that are about a mind, and a different reading (A7 only) would not change the cover.
4. **T2 rows are still mostly low confidence.** The additions are well supported by the structure of the rows but not by their individual descriptions. A5 is 11 rows, all low; if none holds, A5 leaves the cover.
5. **Tier agreement has not been re-measured since the M rules.** A blind second pass on a random sample of 400 rows is still the way to claim the 80% bar; the M rules were applied by name, not blind.

## 10. The case bank

cases_classified.tsv is unchanged from v2. C116 (an exact copy of a person) stays T3, consistent with M1: the exact copy is the unbounded T3 version, and a rough or lossy record is the T2 A7 version. C19 (a sleeper casts in a dream) stays T1, because the dreamer's own brain makes the percepts and no mind exists outside a brain; the shared-dream rows moved by M2 are a different case (two minds coupled). C124 (a false memory in a stranger) stays T3 by R5 and M4.

## Files (all beside this report)

- classified.tsv (v3): all 4,128 rows, with tier, tier_v1, tier_v2, needs, label, owner, A7class (strict, coupling or owner), the owner-as-T0 reading, bounded and unbounded versions and tiers, confidence, rules (R1 to R5, M1 to M4), note2, note3 and one 0/1 column per addition. classified_v2.tsv and report_v2.md are version 2; other version 2 tables carry _v2; first-pass tables carry _v1.
- moved_v2_to_v3.tsv: every row whose tier, needs or A7 class changed, with the rule and a note.
- coverage_curve.tsv, coverage_greedy.tsv (owner = A7) and coverage_curve_ownerT0.tsv, coverage_greedy_ownerT0.tsv (owner = T0), each with all rows and high+medium columns.
- matrix_additions.tsv, matrix_need_patterns.tsv, matrix_kind_addition.tsv, variety_per_cost.tsv (owner = A7) and the same with _ownerT0; matrix_family_tier.tsv, matrix_lane_tier.tsv, matrix_lane_addition.tsv; cost_table.tsv; other_additions.tsv; lowconf.tsv; cases_classified.tsv.

## Known residual issues (main session, 2026-11-27)

A final check of v3 found the numbers and covers correct, but the line between A1 (mind-to-mind effects through the ports) and A7 (minds outside brains) is still drawn unevenly in about a dozen rows:
- familiars, dream eaters, hive minds and group minds, mesmerism: some moved to A7, similar ones stayed A1;
- ghost-lover and revenant rows: listed as strict in the text, still "owner" in the data;
- reincarnation, draugr and a moving reflection: still T1.

These do not change the covers (A1, A6, A7, A8, plus A5 on 11 weak rows) or the finding that A2-A4 are needed by no row. The clean rule already exists in decision Q5.3: shared states and induced percepts through the ports are A1; CONTENT (commands, beliefs, memories) in a specific brain needs that brain's code, so it is out of reach for strangers and a pack otherwise; a mind persisting or acting outside its brain is A7. The next pass, if any, applies Q5.3 to these rows. Classification is a judgement tool here, not a measurement: confidence is low on 476 of 1,405 T1+T2 rows.
