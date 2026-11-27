# Gap analysis report, version 4: what the wanted magic costs in bits, and the cheapest set that covers it

Version 4, 2026-11-27. It re-costs the additions in bits (the library answer of 2026-11-27, R1.7, R2.4, R3, R4.3, R5), renames them E1 to E8, splits X into X-N and X-L, applies the library's seven tier corrections, and applies owner decision QSG (spirits, ghosts and gods as a ladder of self-sustaining field patterns). Version 3 is kept as report_v3.md, classified_v3.tsv and the other tables with the suffix _v3 (v2 and v1 files keep their suffixes). Method: docs/magic-gap-analysis.md; decided physics: docs/magic-science.md. Documents and small scripts only; no simulation. Every judgement is marked (u); every cost figure below is the library's reading made into a number, not a computed field theory.

## 0. The answer

- **The chosen set is E1 + E-ladder + E8 + E3, 75.5 bits.** E1 (the base) 32.5; E-ladder 17.4 given E3 (21.4 alone); E8 (doorways) 14.8; E3 (charge-2 pairing) 6.8 net (10.8 alone, minus a 4.0-bit discount it gives the ladder's imprints). It covers all 330 families with T1 or T2 rows (1,377 rows) and leaves none to the uncovered-family penalty. E3 is in the set only because R0's target T10 (sharp-bordered regions) needs charge-2 matter; no catalogue row needs it.
- **Without R0 as a hard constraint the set is E1 + E-ladder + E8 at 68.7 bits.** The 6.8-bit gap is above the library's 3.6-bit bar, so whether R0 T10 holds is a real decision, not noise (section 4).
- **E2 (magnetic charge and dyons) and E4 (non-abelian symmetry) are in no optimum.** Neither unlocks any catalogue row. E2 would pay for itself by unlocking half a family at the library's price (4.6 bits against 9.22), E4 by unlocking 2.1 families (19.0 bits). Their case is the physics and the toy simulations, as in v3.
- **The cover is robust.** The same set, with the ladder covering all its rungs and every family covered, wins in 14 of the 22 scenarios of section 5 (18 of 22 if the four ladder-capped scenarios are counted, since they keep the same four additions but leave 11.9 to 38.8 family-equivalents uncovered): prices from 18.4 down to 0.5 bits per family, every free number and tuning doubled or halved, low-confidence rows removed, the former owner rows read as belief. The four exceptions to the 18 are two price collapses (0.2 and 0.1 bits per family) and two scenarios with R0 not hard (E3 drops out). In 3,000 random draws that scale every judgement item by 0.5 to 2 times, the ladder and the doorway are in the optimum every time and E3 in 81% (the draws where R0 was hard). Break-even prices: rungs 3 and 4 together (mind coupling and full field minds, 12.4 bits for 38.8 family-equivalents) fall away below 0.32 bits per family (rung 3 alone is 0.42, but rung 4 depends on it, so the pair is the unit that drops out, and the price sweep keeps both down to 0.35), the doorway below 0.21, the lower rungs below 0.12.
- **Two of the cheapest-looking additions in v3 were the most understated.** The E-ladder costs 21.4 bits against 10 flat units for A5, A6 and A7 together, and 10.0 of the 21.4 is one tuned number (mind-selective coupling). E8 costs 14.8 against 4, and 6.0 of that is debts (support energy, chronology, loop test).
- **Variety per bit now puts E8 first, narrowly.** Shapley credit above E1: E-ladder 79.6 families, E8 69.4 (of 330; together 149.1 beyond E1's 180.9). Per marginal bit: E8 4.71, E-ladder 4.57. Rao-Stirling disparity is 0.915 and 0.900 (all wanted families: 0.933). Both are far above E2, E3 and E4 (zero).
- **Tiers (4,132 rows; 4 are splits of time stop):** T0 2,197, T1 794, T2 583, T3 194, X-N 275, X-L 6, U 83. The wanted set (T1+T2) is 1,377 rows in 330 families; 447 of those rows (32.5%) are low confidence. E1 alone enables 794 rows (57.7%) and 180.9 family-equivalents (54.8%).
- **Where the owner decides (section 7):** R0 T10 (6.8 bits), E2 (4.6 bits), E4 (19.0 bits), and how high the ladder goes (rungs 3 and 4 hold 184 of the 337 ladder rows, 138 of them low confidence).

## 1. What changed from v3

### 1.1 Renaming: A-labels to E-labels

The additions are E1 to E8 (E for extension) so they no longer clash with the axioms A1 to A8. The data files of v3 keep their A-labels (the _v3 files); the v4 files use E-labels.

| v3 label | v4 label | meaning |
|---|---|---|
| A1 | E1 | the base field with electric magic charge, carried by organised matter, three portals (binding, mixing, energy exchange) |
| A2 | E2 | magnetic charge and dyons |
| A3 | E3 | condition-dependent charge-2 pairing |
| A4 | E4 | non-abelian symmetry |
| A5 | E5, a rung of E-ladder | self-sustaining field patterns |
| A6 | E6, a rung of E-ladder | imprints (the field remembers) |
| A7 | E7, rungs of E-ladder | minds outside brains (and strong coupling between minds) |
| A8 | E8 | doorways |
| A9, A10 | not adopted | no row needs them (other_additions.tsv) |

In the data, E5, E6 and E7 are the one flag EL (the E-ladder). The v3 columns A1 to A10 are replaced by E1, E2, E3, E4, EL, E8; the v3 tier and needs are kept as tier_v3 and needs_v3.

### 1.2 Tiers

| tier | v3 | v4 | change |
|---|---|---|---|
| T0 | 2163 | 2197 | +34 (owner rows placed as belief) |
| T1 | 787 | 794 | +7 (3 luck rows, 4 time-stop splits) |
| T2 | 618 | 583 | -35 (34 to T0, Phasing to T3) |
| T3 | 202 | 194 | -8 (6 to X-L, 3 luck rows to T1; Phasing comes in from T2; the two time-stop rows are replaced by their T3 part (c)) |
| X (v3) / X-N (v4) | 275 | 275 | all 275 are narrative, tool or meta: X-N |
| X-L | 0 | 6 | impossible under any laws |
| U | 83 | 83 | unchanged |
| rows | 4128 | 4132 | the two time-stop rows each became three |

Every changed row is in moved_v3_to_v4.tsv with its rule (v4_rule) and note (v4_note) in classified.tsv. Version 3's own change list is moved_v2_to_v3.tsv.

### 1.3 X split and the library's corrections (R4.3)

**X-N** (narrative, tool or meta): all 275 old X rows. **X-L** (impossible under any laws): only rows that claim UNRESTRICTED knowledge or foresight, by Wolpert's result (no physical device can infallibly know all questions; a restricted class can be inferred). A seer of one kind of event stays T1 or T3. Six rows move to X-L; all six judgements are (u):

| row | why X-L |
|---|---|
| Omniscience | claims knowing everything |
| Recording angel | an all-seeing record of all deeds |
| D1300 Magic mirror that answers truth | answers any question truthfully (u) |
| Precognition | unqualified true future sight (u) |
| Prescience | unqualified true foreknowledge (u) |
| Prophecy (true foresight) | unqualified true foresight (u) |

Rows that stay T3 under the caveat: Cassandra, Minority Report, M300 Prophecies, SCP-6300 Prophecies, D1810 Magic knowledge (each claims a restricted or story-specific foresight, which is backward signalling and so T3), and Claircognizance (knowing without a channel; not clearly unrestricted).

| R4.3 correction | rows | what was done |
|---|---|---|
| 1. Omniscience T3 to X-L | 3 | Omniscience, Recording angel, D1300 mirror |
| 2. Infallible precognition T3 to X-L; backward signalling stays T3 | 3 | the three unqualified foresight rows (u) |
| 3. Time stop: three rows T1, T1, T3 | 2 rows become 6 | Time stop and Time Stop (spell), each split into (a) perceptual speed-up T1 (drag and heat bound it; about 1 GW at 1,000 times), (b) stasis by cooling T1, (c) true clock stop T3 |
| 4. Size change: constant mass is T3 | 2 | Size change and Shrinking keep T1 (feedstock or shed mass) with the note |
| 5. Phasing's "other: selective contact suppression" is T3 (Pauli), not a new-addition finding | 1 | Phasing T2 to T3, bounded version T2 E1+E8 (micro-doorways); the doorway-version rows (Intangibility, Intra-Phasing, Potion of Phase Through Wall, Phase spider) stay T2 E8 |
| 6. Transmutation stays T1 with a nuclear-scale cost | 3 | Alchemy, Philosopher's stone, Transmutation noted (about 5e9 J per gram of gold); Projection (alchemy) stays T3 |
| 7. Teleport and doorways pay the chronology cost and loop test | 266 | every T2 row needing E8 is flagged; the cost is in E8's price (section 3) |
| bounded luck at a price per bit is T1 | 3 | Luck, Probability Manipulation, Fortune and luck cultivators T3 to T1; free unbounded biasing stays the T3 version |

### 1.4 The ladder (QSG)

E5, E6 and E7 are rungs of ONE addition, the E-ladder: a self-sustaining field pattern with a capacity limit. The rungs: 1 imprint (replay and fade, no agency), 2 reactive pattern (simple drives, like a flame), 3 mind fragment (partial memory and intent, lossy), 4 full field mind. Every v3 row that needed A5, A6 or A7 now needs EL and carries a rung. The 270 former owner (179) and strict A7 (91) rows were all placed. Placement rule (a judgement for every row, (u), recorded in the rung column of classified.tsv and listed in ladder_rows.tsv):

- **Rung 3, strict rows (91):** a disembodied mind possessing, riding or seizing a host, the ghost of a named person acting with intent, reincarnation or transmigration with memory, a mind in an object, a mind leaving its body, a stored or stolen soul, uploads and transfers (lossy, per M1). Medium confidence.
- **Former owner rows (179), low confidence:** 34 stay as belief and are placed T0 (see below); 145 go on rungs: rung 1 residual hauntings and replays (23), rung 2 elementals, nature spirits, poltergeists, familiars, bound small spirits (44), rung 3 ghosts and spirits with purpose, underworld rulers, soul-eaters (30), rung 4 gods, angels, demons and cosmic entities with a believer-sustained or vast substrate (48).
- **A6-only rows (77)** go to rung 1 (curses, taboos, wards, bonds: the field remembers), **A5 rows (9 not already placed)** to rung 2, **coupling rows (15)** to rung 3 (they need the ladder's mind coupling, not a pattern outside a brain).
- **Stay T0 as belief or ritual (34, "fully belief or E1 effects"):** Baron Samedi, Barong/Rangda, Cailleach, Casting the Runes, Cat Sìth, Chinese dragon, Damnification, Demonomancy, Exorcism, Faithifery, Luna Game, Mediumship, Noaidi's drum, Oh Whistle and I'll Come to You, Pact with the Devil, Ritual of summoning, SCP-231, The Yellow Sign, Sekhmet, Lamashtu, Fuji (planchette), Guardian angel, Hantu Raya, Evocation, Bornless Ritual, Eye of Ra, Dakini, Heavenly host, Jonah Hex, Deadlands (Reckoners), Wynonna Earp, Hellsing (Alucard), Divine intervention, Warlock pact (patron). These are rites, pacts, invocations, cult-tracked power or thinly described patrons; the being they invoke has its own row on a rung.

| rung | rows (T1+T2) | families | low confidence | what it is |
|---|---|---|---|---|
| 1 imprint | 100 | 61 | 58 | A6-only rows 77, former owner 23 |
| 2 reactive pattern | 53 | 36 | 46 | former owner 44, A5 rows 9 |
| 3 mind fragment | 136 | 73 | 91 | strict A7 91, owner 30, coupling 15 |
| 4 full field mind | 48 | 28 | 47 | former owner 48 (gods, angels, demons, outsiders) |
| total (E-ladder) | 337 | 141 | 242 | 317 need only E1+EL, 20 also need E8 |

E8 rows: 266 (246 with E1 alone, 20 with the ladder). Needs patterns now (matrix_need_patterns.tsv): E1 only 794, E1+EL 317, E1+E8 246, E1+E8+EL 20.

## 2. Tiers by lane

Primary tiers; rows can sit in several lanes. The last columns count T1+T2 rows needing the ladder and the doorway. File: matrix_lane_tier.tsv; family table: matrix_family_tier.tsv.

| lane | rows | T0 | T1 | T2 | T3 | X-N | X-L | U | T1+T2 | need_E_ladder | need_E8 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bestiaries | 260 | 159 | 53 | 40 | 5 | 3 | 0 | 0 | 93 | 33 | 8 |
| creepypasta | 277 | 203 | 8 | 55 | 5 | 6 | 0 | 0 | 63 | 32 | 23 |
| cyberpunk_transhumanism_postapoc_mecha | 452 | 294 | 86 | 54 | 12 | 6 | 0 | 0 | 140 | 47 | 16 |
| dreams_fairy_tales_and_liminal | 390 | 151 | 46 | 73 | 11 | 29 | 0 | 80 | 119 | 18 | 59 |
| folklore | 411 | 262 | 71 | 62 | 7 | 8 | 1 | 0 | 133 | 53 | 10 |
| horror | 292 | 184 | 36 | 58 | 6 | 8 | 0 | 0 | 94 | 51 | 7 |
| litrpg_cultivation | 337 | 128 | 95 | 35 | 17 | 62 | 0 | 0 | 130 | 16 | 19 |
| mythology_pantheons_religion | 345 | 187 | 61 | 35 | 20 | 41 | 1 | 0 | 96 | 26 | 14 |
| scifi | 285 | 185 | 50 | 21 | 14 | 14 | 1 | 0 | 71 | 10 | 11 |
| scp | 311 | 127 | 62 | 58 | 43 | 18 | 0 | 3 | 120 | 16 | 42 |
| superpowers | 392 | 165 | 142 | 38 | 31 | 13 | 3 | 0 | 180 | 20 | 18 |
| videogame_mechanics_and_loops | 329 | 119 | 87 | 36 | 27 | 59 | 1 | 0 | 123 | 8 | 28 |
| weirdfiction | 235 | 133 | 28 | 56 | 8 | 10 | 0 | 0 | 84 | 43 | 13 |


## 3. Cost in bits

### 3.1 The rule

Following the library (R1.6, R2.4): each discrete choice costs log2(options); each integer log2(1+|m|); each real number log2(range/window), where the window is the interval inside which the wanted cases still emerge. A number that must be small with no protecting symmetry has a narrow window, so tuning is priced automatically (a number tuned to one part in D costs log2 D bits). Numbers shared by two additions are paid once. Costs are marginal: the same item can be cheaper once another addition is held. The six currencies are columns of the table: **F** new fields (3 bits for the kind of field, plus log2(1+degrees of freedom)); **S** imposed symmetries (3 bits for the group, plus log2(1+|charge|); accidental and topological ones cost 0 and are listed as predictions); **O** all terms the symmetries allow up to the working order (Ginzburg-Landau order: quartic in fields, second order in gradients; multipole order for couplings), each of which is either set to zero by a symmetry (0 bits), fixed by a target (0 bits) or a free number (N); **N** free numbers; **T** tuning; **D** discrete choices and consistency debts (the fields, terms and numbers that consistency forces, paid in the other columns' units and shown here). Prior ranges and windows are in decades on a log scale. Every item is a judgement (u) unless it follows from the rule alone.

### 3.2 Totals by currency

| addition | F | S | D | N | T | standalone (given E1) | marginal in the chosen set |
|---|---|---|---|---|---|---|---|
| E1 base | 9.2 | 7.0 | 2.0 | 14.4 | 0.0 | 32.5 | 32.5 |
| E2 magnetic charge, dyons | 0.0 | 0.0 | 4.6 | 1.6 | 0.0 | 6.2 | 4.6 (condensation map shared with E3) |
| E3 charge-2 pairing | 0.0 | 1.6 | 1.6 | 5.6 | 2.0 | 10.8 | 6.8 (gives the ladder a 4.0-bit discount) |
| E4 non-abelian | 10.0 | 3.0 | 0.0 | 6.6 | 2.0 | 21.6 | 19.0 (gives the ladder a 2.6-bit discount) |
| E-ladder (rungs 1 to 4) | 0.0 | 0.0 | 2.6 | 8.8 | 10.0 | 21.4 | 17.4 |
| E8 doorway | 0.0 | 0.0 | 8.0 | 6.8 | 0.0 | 14.8 | 14.8 |
| chosen set E1+E3+E-ladder+E8 | | | | | | | 75.5 |

Against v3's flat units (E1 12, E5+E6+E7 10, E8 4, E3 5): the base is 2.7 times, the ladder about 2 times and the doorway 3.7 times the flat figure. File: cost_summary_bits.tsv; every item: cost_items_bits.tsv.

### 3.3 Terms allowed (currency O), by order

| addition | terms allowed at the working order | status |
|---|---|---|
| E1 | field kinetic (F^2); carrier kinetic, mass (m_c) and quartic; binding phi-squared times H (lambda); kinetic mixing eps F_M.F_EM; energy exchange (pair creation and recombination, kappa); theta_M F F~; mixed theta F_M F~_EM; Lorentz-violating design speed c_M; spurion coupling b P(x) | all paid in N except: field mass 0 (target T2), theta_M and mixed theta 0 (forbidden by T-invariance, E1.S2), kinetic mixing is generated at one loop (compulsory when carriers carry ordinary charge, so priced as a number with a one-loop window, not as a tuned small number) |
| E2 | monopole mass and fugacity; Witten effect (dyon charge theta/2pi); mixed theta | theta restricted to {0, pi} (both T-invariant): 1 bit; dyon labels up to (1,1); strings need quantised electric flux and are NOT bought |
| E3 | pair-condensate quartic and gradient terms; pairing temperature; coupling of the pairing to local carrier density | charge-1 condensation must stay off in the sharp zones: the one tuned number |
| E4 | adjoint quartic; vev; Skyrme-type higher-derivative term for knots; dimension-5 mixing operator | forbids the dimension-4 mixing, so E1's mixing becomes a dimension-5 term with a scale (R6.4): paid inside E4 |
| E-ladder | two quartic self-terms (0 numbers if generated by light-carrier loops); slow imprint variable or pinned flux; capacity C; strong mind coupling; mind-selective coupling | the selective coupling has no protecting symmetry: tuned |
| E8 | not a term of any order: an identification of two regions (library R1.7) | priced as a discrete structure, four numbers and three debts |

### 3.4 Derivation of every cost

**E1, the base** (32.5 bits)

| item | currency | bits | judgement | derivation |
|---|---|---|---|---|
| E1.F1 | F | 4.58 | u | gauge field M: 3 bits (one of 8 field kinds) + log2(1+2 dof) |
| E1.F2 | F | 4.58 | u | charged carrier field: 3 + log2(1+2 dof) |
| E1.S1 | S | 4.0 | - | imposed U(1)_M gauge symmetry: 3 bits (one of 8 groups) + log2(1+abs(q)), q=1 |
| E1.S2 | S | 3 | u | T-invariance of the magic sector (forbids theta_M and the mixed theta term: the named symmetry R0 T6 needs) |
| E1.D1 | D | 1 | u | discrete: carriers carry ordinary charge (yes) so the mixing portal is generated, 1 bit |
| E1.D2 | D | 1 | u | discrete: compact U(1) (monopoles as defects), 1 bit |
| E1.N1 | N | 2.0 | u | g_M: prior 8 decades, window 2 decades (casting works, instruments do not see it) |
| E1.N2 | N | 2.46 | u | epsilon (mixing): prior 11 decades (dark-photon benchmark span), window 2 decades (one-loop band 1e-2..1e-1 x g_M); generated, not tuned |
| E1.N3 | N | 2.32 | u | binding lambda: prior 10 decades, window 2 |
| E1.N4 | N | 2.0 | u | carrier mass m_c: prior 12 decades, window 3 (sets Debye and gap scales at casting energies) |
| E1.N5 | N | 2.58 | u | energy-exchange rate kappa: prior 12 decades, window 2 |
| E1.N6 | N | 1.42 | u | design speed c_M (one Lorentz-violating number): prior 8 decades, window 3 (light-like in air) |
| E1.N7 | N | 1.58 | u | spurion coefficient b of the arrangement function P(x) (axiom A2): prior 6 decades, window 2; per-material numbers are not free |
| E1.N8 | N | 0 | - | field mass m_M = 0 (massless, fixed by target T2); theta_M = 0 (fixed by T-invariance) |

**E2, magnetic charge and dyons** (6.2 bits alone; 4.6 given E3; 2.6 given E4, which contains the monopoles)

| item | currency | bits | judgement | derivation |
|---|---|---|---|---|
| E2.D1 | D | 1 | u | theta_M in {0, pi} (both T-invariant, keeps three portals): 1 bit |
| E2.D2 | D | 2 | - | dyon labels allowed up to (q_e,q_m) <= (1,1): log2(1+1)+log2(1+1) [only if E4 absent] |
| E2.D3 | D | 1.58 | - | what condenses where: electric, magnetic or dyonic (t Hooft): log2(3); the same map as E3 (paid once if both) [paid once if E2 and E3 are both held] |
| E2.N1 | N | 1.58 | u | monopole mass/fugacity scale: prior 12 decades, window 4 [only if E4 absent] |
| E2.F1 | F | 0 | - | no new field: monopoles are defects of the compact lattice; Dirac quantisation fixes the unit (R1.7) |

**E3, condition-dependent charge-2 pairing** (10.8 bits alone)

| item | currency | bits | judgement | derivation |
|---|---|---|---|---|
| E3.S1 | S | 1.58 | - | charge-2 condensate: integer log2(1+abs(q)), q=2; the Z2 topological order is accidental (free), listed as a prediction |
| E3.F1 | F | 0 | - | pair condensate of E1 carriers counts 0 fields (R1.7) |
| E3.D1 | D | 1.58 | - | what condenses where (shared with E2) [paid once if E2 and E3 are both held] |
| E3.N1 | N | 2.0 | u | pairing temperature T_c: prior 6 decades, window 1.5 |
| E3.N2 | N | 1.0 | u | pairing strength (sets xi against lambda: type II): prior 4 decades, window 2 |
| E3.N3 | N | 1.0 | u | quartic beta of the pair condensate: prior 4 decades, window 2 |
| E3.N4 | N | 1.58 | u | condition dependence (T_c against local carrier density or organisation): prior 6, window 2 |
| E3.T1 | T | 2.0 | u | debt/tuning: charge-1 carriers must NOT condense in the sharp zones: m^2 of the charge-1 field positive and above the pair scale; prior 4 decades, window 1 |

**E4, non-abelian symmetry** (21.6 bits alone)

| item | currency | bits | judgement | derivation |
|---|---|---|---|---|
| E4.F1 | F | 5.0 | u | non-abelian gauge multiplet (two extra vectors): 3 + log2(1+3) |
| E4.F2 | F | 5.0 | u | adjoint scalar: 3 + log2(1+3) |
| E4.S1 | S | 3 | u | non-abelian group and breaking pattern replaces U(1)_M: 3 bits for the discrete choice (net) |
| E4.N1 | N | 2.0 | u | vev v: prior 8 decades, window 2 |
| E4.N2 | N | 1.0 | u | adjoint quartic: prior 4 decades, window 2 |
| E4.N3 | N | 1.58 | u | higher-derivative (Skyrme) coefficient for stable knots: prior 6, window 2 |
| E4.N4 | N | 2.0 | u | dimension-5 mixing scale Lambda (mixing portal moves from dimension 4 to 5, R6.4): prior 10 decades, window 2.5 |
| E4.T1 | T | 2.0 | u | knot stability window: Skyrme and quartic must balance: prior 4, window 1 |

**E-ladder, self-sustaining field patterns with a capacity limit** (all four rungs: 21.4 alone, 17.4 given E3, 18.8 given E4)

The imprint item is the cheapest available way to hold memory: pinned flux (E3 held, 1.0 bit), frozen monopoles (E2 held, 2.0 bits) or hysteresis near a transition (5.0 bits). A slow field with a shift symmetry (12.3 bits) is dearer and never used. Rungs 1 and 2 cost 9.0 bits alone (5.0 imprint + 2.6 self-sustaining + 1.4 capacity) and 5.0 given E3; rung 3 adds 11.4 (mind coupling 1.4 + selective tuning 10.0); rung 4 adds 1.0.

| item | currency | bits | judgement | derivation |
|---|---|---|---|---|
| EL.imprint.E3 | N | 1.0 | u | imprint memory = pinned flux (needs E3): 0-1 number (depinning threshold), 1 bit [used when E3 is held] |
| EL.imprint.E2 | N | 2.0 | u | imprint memory = frozen monopoles (needs E2): 1-2 numbers, 2 bits [used when E2 is held] |
| EL.imprint.alone | N | 5.0 | u | imprint memory by hysteresis near a transition: two numbers, one tuned (prior 4 decades, window 0.5 = 3 bits) + 2.0 bits for the second number (prior 4 decades, window 1 decade = log2(4)); a slow field with a shift symmetry would be 12.3 bits and is not used |
| EL.self.alone | D | 2.6 | u | self-sustaining pattern (E5): two quartic terms generated by light-carrier loops, 0 numbers; debt: m_c window narrowed 3 decades to 1 = 1.6 bits; discrete choice of the two terms 1 bit [only if E4 absent] |
| EL.cap | N | 1.42 | u | capacity limit C (bits per cell the field can hold; sets the top rung): prior 8 decades, window 3 |
| EL.mind.g | N | 1.42 | u | strong mind coupling g_mind (coherent neural moments): prior 8 decades, window 3 [only if the ladder reaches rung 3] |
| EL.mind.T | T | 10.0 | u | mind-SELECTIVE coupling (brains over other organised matter): no protecting symmetry, so tuned: Delta = 1e3 gives log2 = 10 bits (Barbieri-Giudice as a Bayes factor) [only if the ladder reaches rung 3] |
| EL.r4 | N | 1.0 | u | rung 4 (full field minds): window of C narrowed to allow deep-node accumulation: 1 bit [only if the ladder reaches rung 4] |

**E8, doorways** (14.8 bits). The three debts are the library's (R1.7, R3.3): a static traversable throat needs negative energy (so the doorway is made dynamic and its support work is booked), moving mouths make a time machine (forbidden by the engine's global tick as a preferred frame, plus the loop test "no sequence of doorway moves returns a body to an earlier tick"). Chronology and loop test together are 4.0 bits, support energy 2.0, the debts 6.0 of 14.8.

| item | currency | bits | judgement | derivation |
|---|---|---|---|---|
| E8.D1 | D | 2.0 | u | discrete structure: identification map of two regions and type (paired, fixed far end, dialled): log2(4) |
| E8.N1 | N | 1.58 | u | throat radius limit: prior 6 decades, window 2 |
| E8.N2 | N | 2.0 | u | opening energy per area: prior 8, window 2 |
| E8.N3 | N | 1.58 | u | lifetime without support: prior 6, window 2 |
| E8.N4 | N | 1.58 | u | maximum separation range: prior 6, window 2 |
| E8.D2 | D | 2.0 | u | debt: static traversable throat needs negative energy; made dynamic: support work per area-second, one number (prior 8, window 2) |
| E8.D3 | D | 3.0 | u | chronology debt: moving mouths make a time machine; forbidden by the global tick as a preferred frame (one imposed condition, 3 bits) |
| E8.D4 | D | 1.0 | u | loop test: no sequence of doorway moves returns a body to an earlier tick (a checked constraint on mouth motion): 1 bit |

### 3.5 Interactions (marginal costs)

| pair | effect | bits |
|---|---|---|
| E3 held, ladder added | imprints become pinned flux: 5.0 to 1.0 | -4.0 |
| E2 held, ladder added | imprints by frozen monopoles: 5.0 to 2.0 | -3.0 |
| E4 held, ladder added | self-sustaining term comes with the gauge self-coupling: 2.6 to 0 | -2.6 |
| E2 and E3 both held | the condensation map is paid once | -1.6 |
| E4 held, E2 added | monopoles are contained in E4: dyon labels and monopole scale not paid | -3.6 |
| E4 added | E1's mixing moves to dimension 5 with a scale | +2.0 inside E4's 21.6 |
| ladder to rung 2 only | no mind coupling, no selective tuning, no rung 4 | -12.4 |

## 4. The cover

### 4.1 Constraints and prices

**Hard constraints (R0 and the case bank).** E1 is required by every target T1 to T9 and T11 to T15 (the field, waves, silent bulk, material classes, three portals, sources, fluctuation and dissipation, grip, zero-magic limit, conservation, lawful end states, the effective layer, the quantum deeper theory). **T10 (distinct, stable kinds of region with sharp borders) needs charge q of at least 2 (library emergent-media answer, correction 2; H2 change 2), so E3 is required if R0 holds** (u: no other route to sharp borders is known; if one is found T10 stops forcing E3). T16, the case bank: of 135 cases, 107 are T1 and 23 T0 and need at most E1; the 5 T3 cases are packs or named doorways (C24, a teleport with no doorway, is the one that names E8). So every constraint is met by {E1, E3}.

**Price per uncovered family.** e = log2(598) = 9.22 bits per family (the library's suggested value, R2.4; log2 of the 598 catalogue families), with a sensitivity range of 0.5 to 2 times that (4.6 to 18.4) and a sweep down to 0.05. A family counts by its share: each uncovered T1 or T2 row costs e divided by the number of T1 and T2 rows its family has, so a family with one of three rows uncovered costs a third of e. (In the integer reading, a family is covered if any of its rows is enabled; the Shapley table gives both.) An addition earns its place only if it removes at least bits/e families.

**Search.** Exact over all 32 subsets of E2, E3, E4, E-ladder, E8 (E1 always in). Total = additions' bits (with sharing and discounts) + e times uncovered family-equivalents. File: cover_all_subsets_bits.tsv.

### 4.2 Result

| rank | set | additions (bits) | uncovered families | total bits at 9.22 |
|---|---|---|---|---|
| 1 | E1+E3+E8+E-ladder (R0 holds) | 75.5 | 0 | **75.5** |
| 2 | E1+E2+E3+E8+E-ladder | 80.1 | 0 | 80.1 |
| 3 | E1+E3+E4+E8+E-ladder | 94.5 | 0 | 94.5 |
| 4 | E1+E3+E-ladder (no doorway) | 60.7 | 70.8 | 713.4 |
| memo | E1+E8+E-ladder (R0 not hard) | 68.7 | 0 | 68.7 |

No other set is within 1.44 bits of the winner, so there are no ties. The winner leads the nearest alternative that differs in a family-covering addition by 637.9 bits; the nearest set that differs only in a no-family addition (E2) by 4.6 bits, above the library's 3.6-bit bar for a winner. The one close rival is the R0-soft set, 6.8 bits cheaper, which is above the bar too: dropping E3 is a real option if T10 is dropped.

The set against v3: the minimal cover A1, A5, A6, A7, A8 becomes E1, E-ladder, E8. The three rungs of the ladder share one capacity limit, one imprint mechanism and one set of debts, so the mind rungs (v3's A7) no longer look like a separate cheap addition: they cost 11.4 of the ladder's 21.4 bits.

### 4.3 Coverage curve in bits

Best coverage of the 330 families (fractional) within a budget, from the exact subset search. File: coverage_curve_bits.tsv (R0 not forced) and coverage_curve_bits_R0.tsv.

| bits | set | R0 satisfied | family-equivalents covered | share | families with at least one row covered |
|---|---|---|---|---|---|
| 32.5 | E1 | no | 180.9 | 54.8% | 233 |
| 43.3 | E1+E3 | yes | 180.9 | 54.8% | 233 |
| 47.3 | E1+E8 | no | 249.1 | 75.5% | 287 |
| 58.0 | E1+E3+E8 | yes | 249.1 | 75.5% | 287 |
| 54.0 | E1+E-ladder | no | 259.2 | 78.5% | 283 |
| 60.7 | E1+E3+E-ladder | yes | 259.2 | 78.5% | 283 |
| 68.7 | E1+E8+E-ladder | no | 330.0 | 100% | 330 |
| 75.5 | E1+E3+E8+E-ladder | yes | 330.0 | 100% | 330 |

The first 32.5 bits (the base) give 54.8% of the family-equivalents; the doorway adds 20.7 points for 14.8 bits; the ladder adds 23.7 points for 21.4 bits alone; both together reach 100%. E2 and E4 add bits and no coverage, so they never sit on the front.

### 4.4 Price sweep

How low must the price per family fall before an addition stops paying? The ladder's top is chosen as part of the search. File: price_sweep.tsv.

| price_per_family_bits | best_set | ladder_top_rung | total_bits | additions_bits | uncovered_family_mass |
|---|---|---|---|---|---|
| 9.22 | E1+E3+E8+EL | 4 | 75.5 | 75.5 | 0.0 |
| 4.61 | E1+E3+E8+EL | 4 | 75.5 | 75.5 | 0.0 |
| 2.0 | E1+E3+E8+EL | 4 | 75.5 | 75.5 | 0.0 |
| 1.0 | E1+E3+E8+EL | 4 | 75.5 | 75.5 | 0.0 |
| 0.5 | E1+E3+E8+EL | 4 | 75.5 | 75.5 | 0.0 |
| 0.35 | E1+E3+E8+EL | 4 | 75.5 | 75.5 | 0.0 |
| 0.3 | E1+E3+E8+EL | 2 | 74.7 | 63.1 | 38.8 |
| 0.25 | E1+E3+E8+EL | 2 | 72.8 | 63.1 | 38.8 |
| 0.2 | E1+E3+EL | 2 | 69.8 | 48.3 | 107.7 |
| 0.15 | E1+E3+EL | 2 | 64.5 | 48.3 | 107.7 |
| 0.1 | E1+E3 | - | 58.2 | 43.3 | 149.1 |
| 0.05 | E1+E3 | - | 50.7 | 43.3 | 149.1 |

The break-even price of each piece (bits it costs divided by the family-equivalents only it covers, given the rest of the set) is in section 7.

## 5. Robustness

Re-runs of the exact search under the stated cost uncertainties and with rows removed. Total bits include the penalty for families left uncovered. File: robustness_bits.tsv.

| scenario | wanted_rows | best_set | total_bits | additions_bits | uncovered_families | sets_within_1.44_bits |
|---|---|---|---|---|---|---|
| base (e=9.22, R0 hard) | 1377 | E1+E3+E8+EL | 75.5 | 75.5 | 0.0 | 1 |
| e=4.6 (x0.5) | 1377 | E1+E3+E8+EL | 75.5 | 75.5 | 0.0 | 1 |
| e=18.4 (x2) | 1377 | E1+E3+E8+EL | 75.5 | 75.5 | 0.0 | 1 |
| e=1.0 | 1377 | E1+E3+E8+EL | 75.5 | 75.5 | 0.0 | 1 |
| e=0.2 | 1377 | E1+E3 | 73.1 | 43.3 | 149.1 | 2 |
| e=0.1 | 1377 | E1+E3 | 58.2 | 43.3 | 149.1 | 1 |
| costs: every free number and tuning x2 | 1377 | E1+E3+E8+EL | 119.0 | 119.0 | 0.0 | 1 |
| costs: every free number and tuning x0.5 | 1377 | E1+E3+E8+EL | 53.7 | 53.7 | 0.0 | 1 |
| ranges doubled (+1 bit per real number) | 1377 | E1+E3+E8+EL | 94.5 | 94.5 | 0.0 | 1 |
| ranges halved (-1 bit per real number) | 1377 | E1+E3+E8+EL | 56.5 | 56.5 | 0.0 | 1 |
| mind-selective tuning x3 (30 bits) | 1377 | E1+E3+E8+EL | 95.5 | 95.5 | 0.0 | 1 |
| mind-selective tuning x0.3 (3 bits) | 1377 | E1+E3+E8+EL | 68.5 | 68.5 | 0.0 | 1 |
| doorway debts x3 (chronology etc.) | 1377 | E1+E3+E8+EL | 87.5 | 87.5 | 0.0 | 1 |
| R0 not hard (T10 dropped) | 1377 | E1+E8+EL | 68.7 | 68.7 | 0.0 | 1 |
| low-confidence rows removed | 930 | E1+E3+E8+EL | 75.5 | 75.5 | 0.0 | 1 |
| low-confidence rows removed, R0 not hard | 930 | E1+E8+EL | 68.7 | 68.7 | 0.0 | 1 |
| former owner rows read as belief (removed) | 1232 | E1+E3+E8+EL | 75.5 | 75.5 | 0.0 | 1 |
| ladder tops out at rung 2 (RQ5 low) | 1377 | E1+E3+E8+EL | 421.3 | 63.1 | 38.8 | 1 |
| ladder tops out at rung 3 | 1377 | E1+E3+E8+EL | 184.2 | 74.5 | 11.9 | 1 |
| ladder rung 2, e=1.0 | 1377 | E1+E3+E8+EL | 101.9 | 63.1 | 38.8 | 1 |
| ladder rung 2, owner rows removed, e=1.0 | 1232 | E1+E3+E8+EL | 86.3 | 63.1 | 23.3 | 1 |
| high+medium confidence rows only | 930 | E1+E3+E8+EL | 75.5 | 75.5 | 0.0 | 1 |

Reading:
- **Prices.** The set is unchanged from e = 18.4 down to 0.35 (price sweep, section 4.4). At 0.3 and 0.25 the ladder is cut to rungs 1 and 2; at 0.2 and 0.15 the doorway goes; at 0.1 only the base and E3 remain. The price would have to fall to about 3% of the library's value before the top of the ladder was dropped. (The scenario table fixes the ladder's top at rung 4, so its rows at 0.2 and 0.1 show the doorway and ladder both dropped.)
- **Costs.** Doubling or halving every free number and tuning, or adding or subtracting a bit for every real number (the library's "every range doubled and halved"), changes the total (53.7 to 119.0) and never the set. Tripling the mind-selective tuning to 30 bits, or tripling the three doorway debts, changes nothing: the set pays by hundreds of bits of penalty avoided.
- **Rows.** Removing all 447 low-confidence rows, keeping only high and medium confidence rows (930 rows, 280 families), or reading all former owner rows (145) as belief leaves the set. The ladder stays even with rungs 3 and 4 removed entirely, because rungs 1 and 2 (153 rows) and the A6 and A5 rows still need it (63.1 bits).
- **Ladder height (RQ5).** If the medium tops out at rung 2, the ladder costs 63.1 bits in total for the set and leaves 38.8 family-equivalents (rungs 3 and 4) without cover: they become packs at the price. At rung 3 the set costs 74.5 and leaves 11.9.
- **R0.** Dropping T10 (or finding another route to sharp borders) removes E3: 68.7 bits. This is the single decision that moves the set, and it moves it by one addition.
- **Random draws.** 3,000 draws, each scaling every judgement item by a factor between 0.5 and 2 (log-uniform) and the price by 0.5 to 2, with the ladder's top drawn from rungs 2, 3 or 4, the wanted set from all rows or high and medium rows only, and R0 hard in 80% of draws. The ladder and the doorway were in the optimum in 100% of draws; E3 in 80.8% (every draw where R0 was hard); E2 in 2.2% (it appears only when R0 is soft and the cheaper imprint route outweighs its own cost); E4 in 0%. In 3,000 draws that vary only the costs and the price (R0 hard, all rows), the optimum was E1+E3+E8+E-ladder every time.

## 6. Variety

Each addition's credit is its Shapley value over all subsets of the six additions: the value of a subset is the number of family-equivalents whose rows it enables (a row needs every addition it lists), and an addition's credit is its marginal families averaged over all join orders. Over all six players E1 holds 255.0 (it is needed by every row); with E1 given, the five others split the remaining 149.1. Files: shapley_variety_per_bit.tsv, marginal_by_subset_size.tsv.

| addition | rows needing it | families needing it | marginal bits (in the chosen set) | Shapley, six players | Shapley above E1 (families) | same, integer families | above E1 per marginal bit | Rao-Stirling disparity of unlocked families |
|---|---|---|---|---|---|---|---|---|
| E1 | 1,377 | 330 | 32.5 | 255.0 | (base) | (base) | | 0.933 (all wanted) |
| E8 doorway | 266 | 119 | 14.8 | 34.9 | **69.4** | 50.5 | **4.71** | 0.900 |
| E-ladder | 337 | 141 | 17.4 | 40.0 | **79.6** | 46.5 | **4.57** | 0.915 |
| E3 | 0 | 0 | 6.8 | 0.0 | 0.0 | 0.0 | 0 | |
| E2 | 0 | 0 | 4.6 | 0.0 | 0.0 | 0.0 | 0 | |
| E4 | 0 | 0 | 19.0 | 0.0 | 0.0 | 0.0 | 0 | |

- **Rank by variety per bit:** E8 (4.71) then E-ladder (4.57 against its marginal 17.4 bits; 3.72 against its standalone 21.4); E3, E2, E4 have no catalogue variety. With integer families the lead widens (E8 3.41, E-ladder 2.67). The two are close on the fractional count; both are far above the zero-row additions.
- **Order dependence.** The ladder's marginal families depend only a little on what is held: averaged over the other additions it is 78.3, 79.0, 79.6, 80.3 for 0, 1, 2, 3 others (range 78.3 to 80.9); E8's is 68.1, 68.8, 69.4, 70.1. The 20 rows needing both make the small rise. Order dependence is small: both additions sit on E1 and only 20 rows need both.
- **Complements.** Mean number of additions beyond E1 a row needs: 794 rows 0, 563 rows 1, 20 rows 2 (mean 0.44). Valence is low, so the Hausmann-Hidalgo convexity warning does not bite here.
- **Disparity.** Rao-Stirling index on a TF-IDF distance between families' mechanism and reason text (families weighted by wanted rows): 0.915 for the ladder, 0.900 for E8, 0.933 across all wanted families. Both additions open families about as different from each other as the whole catalogue is (u: text distance is a rough proxy; a distance by lane or kind is the library's other option).
- **Totalitarian column** (R3.1: every process an addition allows will happen, so list them, wanted or not) (u):

| addition | allowed processes (wanted or not) | allowed but absent, and why |
|---|---|---|
| E1 | induction and radiation; screening by carrier plasma; mixing-driven sparks in wires and nerve stimulation; pair creation by strong fields (energy exchange); casting-threshold noise (T8) | bulk force on neutral matter: absent by axiom A2 (spurion rule) |
| E2 | Witten effect and magnetoelectric cross-effects; monopole pair creation in extreme fields; dyon bound states | free monopoles at everyday energy: heavy (a mass number); string tension: needs quantised electric flux, not bought |
| E3 | sharp borders, half-quantum threads, Aharonov-Bohm sensing, vortex lattices and flux creep, borders that melt above T_c | charge-1 condensation in the sharp zones: held off by the one tuned mass |
| E4 | knots as particles, braiding memory, extra massive vectors | decay of knots: set by the Skyrme balance (tuned) |
| E-ladder | imprints of every strong event accumulate; reactive patterns persist where fuel and drive exist; mind fragments and field minds form where capacity allows; patterns also form around non-minds (computers) to the extent the selectivity is imperfect; patterns can be fed by worship (R6) | runaway patterns: held by the capacity limit C; unselective coupling: the 10-bit tuning |
| E8 | any two regions can be joined; potential-energy and momentum differences between mouths drive flows (booked); a doorway is a heat pump or pressure equaliser | closed loops: forbidden by the tick (loop test); static throats: need support work |

## 7. Marginal families: where the owner's preferences decide

Method: for each piece of the chosen set, the break-even price is its marginal bits divided by the family-equivalents only it covers, given the rest. A piece is "near the line" if the break-even is within a factor of a few of the adopted price (9.22). File: marginal_components.tsv.

| component | marginal_bits | family_mass_it_alone_covers | break_even_bits_per_family | ratio_to_9.22 |
|---|---|---|---|---|
| E8 doorway (given E1,E3,EL) | 14.75 | 70.76 | 0.209 | 0.023 |
| E-ladder rungs 1-2 (imprint, self-sustaining pattern, capacity; given E3) | 5.02 | 42.11 | 0.119 | 0.013 |
| E-ladder rung 3 (mind coupling + mind-selective tuning) | 11.42 | 26.95 | 0.424 | 0.046 |
| E-ladder rung 4 (full field minds, capacity) | 1.00 | 11.89 | 0.084 | 0.009 |
| E-ladder as one addition | 17.43 | 80.95 | 0.215 | 0.023 |
| E3 charge-2 pairing (net of its 4.0-bit discount on imprints) for catalogue rows | 6.75 | 0.00 | inf | inf |
| E2 magnetic charge and dyons (given chosen set) | 4.58 | 0.00 | inf | inf |
| E4 non-abelian (given chosen set) | 18.98 | 0.00 | inf | inf |

**Nothing is near the line at the library's price.** The highest break-even is rung 3 alone at 0.42 bits per family, 4.6% of 9.22 (rungs 3 and 4 together, the unit that can actually drop out, 0.32); the ladder as a whole is 0.22 (2.3%), E8 0.21 (2.3%). A family would have to be worth less than half a bit for these to be dropped (price sweep above). The catalogue rows give no close call. So where do the owner's preferences actually decide? In four places:

1. **R0 T10 and E3: 6.8 bits for zero catalogue families.** In family terms it is 0.73 of one family at 9.22 (6.755 / 9.224). The decision is whether sharp-bordered regions (T10) are required (then E3 enters and gives the ladder a 4.0-bit discount) or optional (then the set is 68.7 bits and charge-1 matter keeps the Fradkin-Shenker smooth crossover).
2. **E2: 4.6 bits for zero catalogue families** (half a family). The decision is whether magnetoelectric cross-effects and dyons are wanted as a kind of magic the catalogue does not have.
3. **E4: 19.0 bits for zero families** (2.1 families), and it forbids the dimension-4 mixing. The decision is whether stable knots and braiding memory are wanted as new kinds of magic, or as the substrate of the upper ladder rungs (RQ5). It also gives the ladder a 2.6-bit discount, which does not close the gap.
4. **How high the ladder goes (RQ5).** Rungs 3 and 4 cost 12.4 bits and cover 38.8 family-equivalents (184 rows, 138 of them low confidence); rungs 1 and 2 cost 5.0 given E3 and cover 42.1. If the toy simulations show the medium cannot hold complex self-maintaining structures, the 38.8 family-equivalents become packs.

**The families that sit on the weakest ground** (not at the price line, but resting on the least-supported pieces; the first to fall if the price or the evidence drops):

- **Families resting wholly on rungs 3 and 4:** 15 families, 25 rows (14 of them low confidence): Technology (Warhammer 40,000) 4, BoC demons 3, Angels and demons 3, Transhumanism 3 (the uploader rows, 1 of 3 low), Egypt 2, Motif A (gods of sky, sea, earth), Soul-external (Koschei), Nature spirits (Rusalka), possessive mask, ghost/reptile, artifact/ritual, deity/kaiju, cadaver/electronic, cadaver/indestructible, DCC deities. Families with a large share on rungs 3 and 4: Chaos (Warhammer 40,000) 9 of 12, Great Old Ones 6 of 8, Cursed media, Dream phenomena, Motif E. File: marginal_families.tsv (91 families have at least one row on rung 3 or 4).
- **Families resting wholly on the doorway:** 47 families, 106 rows: Backrooms levels 37, Backrooms 14, Mythos places 5, Liminal spaces 2, extradimensional and portal families, Isekai concepts, Fae. Chronology is the doorway's debt, so a stricter reading of "a place reached by a doorway" would drop these first. File: marginal_families_E8.tsv (119 families have a doorway row).
- **Rows needing both** the ladder and the doorway (20): Hades, Hel, Valhalla, Astral plane, Gallu, and the seven uploader rows (Agent Smith, Soulkiller, Altered Carbon, Digital immortality, Puppet Master, Relic engram, Teletransportation paradox), among others. They are covered only if both are held.

## 8. What is still uncertain

1. **Rounding convention.** Each addition's cost is the sum of its ledger items (each item to its stated digits), rounded to 0.1; totals are sums of those rounded figures (32.5 + 10.8 + 14.8 + 17.4 = 75.5). Using exact logarithms throughout gives 75.4; no decision changes at that difference.
2. **Every cost is a judgement.** The structure is the library's (six currencies, bits, marginal, shared numbers once); the prior ranges and windows are mine and are marked (u). The result does not depend on them (section 5), but the figures (75.5 bits) should not be read to a bit. The costs are the library's reading of each addition (R1.7), not computed field theories; consistent writings of E1 to E8 would firm them up. One consequence: since E1's mixing is generated, "kinetic mixing can be forbidden" (library R6) is used only as a design choice (carriers carry ordinary charge).
3. **The rung placements are my judgement by row name** (section 1.4), low confidence for the former owner rows and medium for the strict ones. They move the rows between rungs and demote 34 to T0; they do not change which additions are chosen, even with rungs 3 and 4 removed (section 5).
4. **The uncovered-family price is a prior, not a measurement.** At 9.22 bits nothing is close; the ladder and doorway would drop only below 0.22 bits per family.
5. **E3 enters only if R0 T10 holds and has no other route.** If sharp borders can be had another way, E3 is not needed.
6. **The six X-L rows rest on reading generic foresight as unrestricted** (three of the six are marked (u)). A narrower reading returns Precognition, Prescience and Prophecy to T3, which changes no cover.
7. **Rows still at v3's confidence.** The T1/T2 confidence is 447 low of 1,377 (32.5%). Tier agreement (94% in v2) has not been re-measured since the M rules or these changes. The residual A1/A7 line issues listed at the end of report_v3.md (a dozen rows of familiars, hive minds, ghost-lover and revenant rows, reincarnation and draugr) are not fixed in this pass: the ladder placement treats the A7-class rows as they stand.
8. **Coupling rows.** Coupling rows (15) sit at rung 3 only because they need the ladder's mind coupling; under Q5.3 many shared-state effects through the ports are E1. A narrower reading would move some of them to T1.

## Files (all beside this report)

- classified.tsv (v4): all 4,132 rows: tier (v4), tier_v3, needs (E-labels), needs_v3, one 0/1 column per addition (E1, E2, E3, E4, EL, E8), rung and rung_conf, xclass (X-N, X-L), v4_rule and v4_note, with all v3 columns kept. classified_v3.tsv and report_v3.md are version 3; v2 and v1 files keep their suffixes.
- moved_v3_to_v4.tsv: every changed row. ladder_rows.tsv: every row placed on a rung or demoted to T0.
- cost_items_bits.tsv (every cost item with currency, bits, judgement and derivation), cost_summary_bits.tsv.
- cover_all_subsets_bits.tsv, coverage_curve_bits.tsv, coverage_curve_bits_R0.tsv, price_sweep.tsv, robustness_bits.tsv.
- shapley_variety_per_bit.tsv, marginal_by_subset_size.tsv, marginal_components.tsv, marginal_families.tsv, marginal_families_E8.tsv.
- matrix_need_patterns.tsv, matrix_family_tier.tsv, matrix_lane_tier.tsv, other_additions.tsv, lowconf.tsv, cases_classified.tsv (E-labels). The v3 coverage, matrix and variety tables are the _v3 files.

## Known issues after the final check (main session, 2026-11-27)

A last check found only small slips; no decision changes:
- section 4.2: the rank-4 row is mislabelled; by cover_all_subsets_bits.tsv the next sets satisfying R0 are E1+E3+E4+E8+EL (94.5 bits) and then all six additions (95.5 bits);
- section 4.3: 78.5% should read 78.6%, and the doorway adds 20.6 points, not 20.7 (rounding);
- the "3,000 random draws" robustness figures cannot be recomputed: the script was not saved. The 22 named scenarios do check out (14 full-ladder wins, 4 ladder-capped wins, 4 exceptions: two price collapses and two runs with R0 soft).
