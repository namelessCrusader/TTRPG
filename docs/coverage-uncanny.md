# Coverage of the uncanny: what MARK1 can and cannot build

Source: `docs/coverage/merged.tsv` (4,128 deduplicated entries from 13 crawl lanes; per-lane summaries sit beside it as `docs/coverage/<lane>.txt`).
Caveat: most mechanisms were classified from titles and family, not article text (u). Dedup is by normalised name only, so synonyms (Vampire / Vampire (folk)) can still be counted twice. Treat all counts as soft, good to roughly +/-5 percent.

## 1. Totals

| Verdict | Rows | Share |
|---|---|---|
| COVERED (works from the layers) | 2,152 | 52% |
| PACK (data only) | 1,426 | 35% |
| LAW (one missing general law) | 308 | 7% |
| SECOND (needs a second change) | 105 | 3% |
| EXCLUDED (meta / narrative / nonsense) | 137 | 3% |

**87% is buildable today or with pack data. 7% needs one new general law each. About 2.5% needs a second change to physics.** The single-change design survived 13 lanes, including the five completeness lanes (LitRPG, mythology, cyberpunk, video-game mechanics, dreams) that were added specifically to attack its weak spots.

By layer (an entry can use several): M 1,879 | K 1,656 | P 1,493 | S 985 | X 628 | G 572. (50 EXCLUDED/meta rows have no layer.)

| Layer | Rows | COVERED | PACK | LAW | SECOND |
|---|---|---|---|---|---|
| M | 1,879 | 999 | 707 | 138 | 35 |
| K | 1,656 | 1,005 | 417 | 159 | 32 |
| P | 1,493 | 734 | 624 | 128 | 5 |
| S | 985 | 589 | 333 | 30 | 5 |
| X | 628 | 347 | 137 | 86 | 53 |
| G | 572 | 196 | 348 | 26 | 1 |

Reading it: K is the layer with the most LAW rows (159), more than P (128) or M (138). The gaps are mostly about **minds**, not matter. X is where SECOND concentrates (53 of 105): time, wishes and fate all want more than a rebuild.

## 2. Per genre (lane of first discovery)

| Lane | Rows | Covered | Pack | Law | Second | Excl. |
|---|---|---|---|---|---|---|
| Superpowers | 382 | 242 | 81 | 15 | 32 | 12 |
| Folklore | 338 | 225 | 90 | 9 | 6 | 8 |
| Bestiaries | 223 | 115 | 89 | 11 | 5 | 3 |
| Horror | 198 | 148 | 25 | 14 | 6 | 5 |
| Weird fiction | 201 | 119 | 56 | 15 | 3 | 8 |
| Creepypasta | 240 | 129 | 90 | 13 | 3 | 5 |
| SCP | 311 | 155 | 83 | 51 | 8 | 14 |
| Sci-fi | 271 | 175 | 27 | 48 | 8 | 13 |
| LitRPG / cultivation | 337 | 110 | 195 | 24 | 3 | 5 |
| Mythology / religion | 341 | 183 | 133 | 10 | 4 | 11 |
| Cyberpunk / post-apoc / mecha | 452 | 164 | 240 | 44 | 2 | 2 |
| Video-game mechanics | 310 | 172 | 58 | 21 | 19 | 40 |
| Dreams / fairy tales / liminal | 387 | 158 | 200 | 18 | 0 | 11 |

(Cross-lane rows, about 130, are counted under the first lane that found them, so the table sums to the total.)

- **Superpowers.** Covers most movement, force, energy, body and element powers as M momentum and energy moves plus G. Pack data: specific powers with a store and limit. Needs a law: luck bias, phasing, light redirection. SECOND: the cluster of 32 is omni-anything, reality warping, fate, probability without bound, time powers. Excluded: genre-shifting, narration, meta powers.
- **Folklore.** The best fit of all (66% covered). Pacts, wards, curses, name-power, changelings, haunting and possession are M patterns and bonds; rumour mutation is K. Pack data: regional creatures. LAW: pattern infection (vampirism, wendigo), true creation. SECOND: fixed fate, true prophecy. Excluded: Thompson motif-index headings.
- **Bestiaries.** Mostly body plus pack data (a monster is a body plan, a diet and a patron). LAW: alternate biochemistry, amorphous bodies.
- **Horror.** 75% covered; the fear is almost all K. LAW: sleep/dream body mode (Freddy), face/agent anomaly detector. SECOND: time-loop horror. Excluded: meta-horror, Final Girl.
- **Weird fiction / Lovecraft.** Cosmic outsiders through resonant links are covered. SECOND: Azathoth, Yog-Sothoth (true omnipotence), Hounds of Tindalos and Yith (time). LAW: mind-state-gated planes, meme entities.
- **Creepypasta.** Covered or pack. Slender Man needs the infohazard channel; the Backrooms are planes plus K (see below). SECOND: Weeping Angels-style observer-gated motion.
- **SCP.** The big LAW lane (51). Antimemetics, cognitohazards, memetic kill-agents and computation entities all need K-layer laws. Object classes are EXCLUDED as labels, but they map to institution-level "containment priority" (S).
- **Sci-fi.** The biggest pure-physics gap: planetary and orbital scale, delta-v, light delay, nuclear energy, radiation, mass-energy conversion. SECOND: FTL, warp, ansible, time dilation.
- **LitRPG / cultivation.** 58% PACK. Systems, levels, classes, skills, achievements, realms and tribulations are reaction-packs over the event log (the proven `cultivation.yaml` route). LAW: bounded death reversal, M-driven tissue repair, an acquired-sense HUD with defined proxies. SECOND: rebirth/regression to the past, Heavenly Dao as an abstract judge.
- **Mythology / religion.** Gods are the cleanest fit to "believer-linked substrate that weakens when faith breaks"; sacrifice becomes upkeep. Miracles reduce to P plus K. Grimoires and circles are pack data. LAW: age-reset, long-decay revival, pattern re-seeding across death, true creation (cornucopia, Draupnir). SECOND: Moirai/Norns fate, true prophecy. Divination is a K sink (cold reading); good test of the model.
- **Cyberpunk / post-apoc / mecha.** 53% PACK; STALKER's Zone is almost pure pack data on a region of elevated M; Warhammer Chaos is covered by faith-substrate beings. LAW (44): computation substrate for nets and AI hosts, mind copy rules, brain-state recording (braindance), Essence (a shared body/M capacity ledger), nuclear and radiation. SECOND: precogs, the wish-granter.
- **Video-game mechanics.** 40 EXCLUDED (UI, commerce, balance, dev tools). Notable: dynamic difficulty and level scaling are excluded on principle, because the world must not bend toward the player. SECOND (19): time travel to the past, time stop, Wish, Power Word Kill. Respawn and lives are PACK from a bind-point store plus rebuild.
- **Dreams / fairy tales / liminal.** No SECOND rows. Grimm and Andersen need no new law (bargains, curses with release clauses, name-power, helpers). Dreams and the Backrooms converge on ONE missing law: a plane rendered from minds.

## 3. Where the fear and excitement come from

Keyword tally of the fear_source column over all 4,128 rows (an entry can count in several): **K about 2,080 (50%)**, **body about 905 (22%)**, **S about 260 (6%)**. M and X are almost never the source of dread. They are the mechanism; K and the body supply the feeling.

Recurring patterns, in order of how many entries use them:

1. **The unknown rule (K).** A thing acts by a law the character does not know: Zone anomalies, Backrooms levels, SCP "do not look", cursed objects, fae bargains. Fear grows with unpredictability, which MARK1 already prices (fear from unpredictability, noisy senses). Nothing extra is needed; the pack author just must not explain the rule.
2. **The misperceived shape (K).** Mimics, doppelgangers, changelings, shadow people, faces in noise, the uncanny. The narrator shows only what a character perceives, so a mimic is scary exactly when a perception channel fails. Wants the "agent-presence expectation violation" law (face detector) to be general.
3. **The rumour that grows (K + S).** Slender Man, urban legends, cryptids, local curses, "someone told me". Rumours that mutate and false beliefs that survive untested are built; creepypasta lives here. S adds taboo and reputation.
4. **The wrongness of a body (P/G).** Parasites, infection, possession, mutation, implants eating the self (Essence, cyberpsychosis), petrification, flaying. Pain and disgust are the body layer; loss of bodily control is possession and parasites.
5. **The trusted thing betrayed (S).** Institutions that grade or watch you (DCC's System, the Foundation-like organisation), patrons with an agenda, contracts you did not choose, a god that wants feeding.
6. **Loss of self (K).** Copies, uploads, memory edits, infohazards, dream invasion. These are the K-layer LAW gaps: they are exactly the horror the corpus cares most about and the engine cannot yet express.
7. **Scale without a mind (K failure).** Azathoth, cosmic indifference: no mind to model, so fear comes from the character's failure to model it.

**Excitement** (superpowers, tech, LitRPG, cultivation, mecha) is mostly competence and growth: K anticipation of the next threshold, admiration, the learning history. The mechanism is thresholds in store dials plus the event log.

## 4. Ranked missing general laws (biggest wins first)

Counts are approximate row clusters from LAW plus SECOND rows.

1. **Information attack channel on minds (~40).** A sensed pattern sets a bias, a link or a timer in a perceiver (percept-as-payload); conditions that fire on knowing or hearing (semantic triggers); a meme as a self-propagating entity. Unlocks Slender Man, Snow Crash, Bird Box, Videodrome, cognitohazards, name curses, knowledge-gated wards.
2. **Memory-record edit and erase (~17), plus mind copy / brain-state encode-decode / copy identity (~25).** Memories are targeted edits on K state; a brain pattern is read or written over an M link. Unlocks antimemetics, braindance, upload, Soulkiller, telepathy, soul search, resleeving, tulpas.
3. **Bounded pattern rebuild with a mass budget (~20).** Heal, regrow, polymorph, shrink, petrify, age-reset, death-reversal from a stored snapshot (respawn, resurrection, phylactery). This is one extension of X (rebuild with booked matter and M energy and a tissue-growth rate cap); it is the single most common effect in LitRPG and D&D. Needs a rule for how the snapshot is made and kept.
4. **Computation substrate and machine-state interface (~22).** Circuits, networks and recording media as in-world places that can host entities, with corruption. Unlocks AI hosts, the net, ICE, technopathy, VR, the System as a machine mind, cyberbrains.
5. **Planet and astronomical scale (~26).** Orbital mechanics, delta-v, light delay, climate and colony ticks. A scale and slow-tick question, not new physics. Mostly sci-fi.
6. **Planes with their own rules (~25).** A plane rendered from sleepers' minds with persistence and write-back; a plane clock offset; lazy infinite plane generation; plane snapshot-restore with one carried pattern (the buildable time loop); pocket volume with conserved mass (bag of holding); scale-changing doors; plane-wide scheduled events. Unlocks Dreamlands, Inception, Backrooms, fairyland, Wonderland, loops with memory.
7. **Nuclear energy release and radiation dose (~8).** Zone radiation, Fallout, Godzilla, transmutation by binding energy.
8. **Pattern infection / self-replication with conserved mass (~8).** Vampirism spread, The Thing, nanotech, grey goo, wendigo.
9. **Sleep and dream as a body mode where perception intrudes (~6).** Sleep paralysis hag, Freddy, shadow people, night terrors.
10. **Luck / chaotic-outcome bias as a tiny M effect with a ledger (~6)**; **crowd panic and mass flight (~4)**; **local time-rate change or rate-freeze ward (~4)**; **light redirection (~3)**; **contact decoupling / phasing (~3)**; **shared body/M capacity ledger (Essence) (~3)**; **face and agent anomaly detector (~3)**; **EM/IR sensors (~3)**.
11. Singles: gravity as a field, mass-field coupling, alternate biochemistry, vacuum biology, relativistic velocity, observer-dependent state, higher-dimensional voxels, doorway graphs with cycles, names and likenesses as link keys, amorphous bodies, an oath-enforcement outcome rule, a rule-inverting effect.

**SECOND-change clusters (105 rows, ~2.5%).** Time to the past and precognition (loops, regression, Hounds of Tindalos, Yith, prophecy, fixed fate); omni powers (omnipotence, true wishes, omnifabrication); reality, fate and probability unbounded; FTL and spacetime (warp, ansible, time dilation); observer-gated motion (Weeping Angels, statue-style); teleport fusion (The Fly). Several move to LAW if the owner accepts a bounded version: luck with a ledger, a rate-freeze ward, observer-gated motion as a perception-triggered rule, plane-reset loops with a carried pattern (see OWNER_QUESTIONS, question 2).

**EXCLUDED (137)** are mostly design pressure, not content: object classes (containment priority, S), Roko's basilisk (infohazard test for law 1), UI and balance terms, alignment, Thompson motifs, meta powers, Dreamtime (excluded as living sacred cosmology of real peoples, not as nonsense).

## 5. Authoring test cases for the pack interface

Each line: what a pack file must express. Form, patterns and stores, triggers, links, mind, seeded rumours. The spread: 8 COVERED, 9 PACK, 8 LAW, 3 SECOND (a pack can still express the failing case, which tests the validator's error messages).

1. **Zeus-style sky patron (COVERED).** Form: no body or a huge avatar; a store fed by believer links; lightning pattern with a drain; mind with arbitrary-wrath drives; rumours of omens.
2. **Burner anomaly (PACK).** Stationary heat-plume cell; trigger on mass entering; heat-flux effect; no mind; rumour "the air shimmers over the old factory".
3. **Dracula-style vampire (PACK, infection law for spread).** Humanoid body with G variant dials (light sensitivity, blood need); patron-host pattern; trigger on invitation (semantic condition); mind with hidden agenda; rumours seeded in a village.
4. **Wendigo (LAW: infection).** Possessing pattern plus starvation drive; spreads by cannibalism; needs the conserved-mass infection law; rumour of tracks.
5. **Slender Man-style stalker (LAW: infohazard).** Tall figure form; perception-triggered link set when seen; sets a stalking drive in the viewer; no direct harm; seeded rumour is the main vector.
6. **Backrooms Level N (LAW: plane laws).** A plane with an adjacency graph, lazy generation, unlit light dials, a plane clock; entry by noclip doorway; no inhabitants or a hostile mind.
7. **Stalker Zone region (PACK).** Region with elevated M; list of anomalies with triggers; artifacts as charged materials; mutants as G packs; stalker culture rumours.
8. **DCC Crawler System (PACK).** An institution pack: a machine mind with sensors and a ledger; reaction-pack of levels, achievements, loot boxes triggered on the event log; audience patron economy; the "killing pays" incentive as author data.
9. **Cultivation realm ladder (COVERED).** Store dial with thresholds, tribulation as a threshold discharge, a sect with rules, pills as charged materials, demonic cultivation as a different bond.
10. **Respawn bind-point (LAW: rebuild with snapshot).** Stored body pattern at a bind point; restore at booked cost after death; cultural fact in the pack: a bound, rebuilt people.
11. **Fae bargain (COVERED).** Pact bond with a release clause, name-power as a link key, gift with drain, mind that obeys the letter; rumour of the rules of hospitality.
12. **Changeling (COVERED).** Rough copy by X rebuild swapped into a family; slow tell in behaviour; a social trigger (iron, name); seeded rumour "the baby changed".
13. **Cursed monkey's paw (COVERED with a patron).** Charged material with a wish bond and a patron interpreting the wish at bounded cost; limits come from the patron, not from true wish power.
14. **Philosopher's stone (COVERED/LAW).** Charged material that lowers rebuild cost; transmutation tied to binding energy; age-reset as the failing LAW case.
15. **Soulkiller engram upload (LAW: copy identity).** Brain-state read over a link; the copy is a mind pattern on a substrate; a rule for whether the source survives; the pack expresses the dilemma, not the answer.
16. **Cordyceps parasite (COVERED).** Organism with a behaviour-hijack pattern on a host; spore link; zombie-like gait; seeded rumour of a sickness upriver.
17. **The Thing (LAW: infection with mass).** Organism-copy law with a conserved mass; mimicry of a victim; trigger on blood test; paranoia from K.
18. **Weeping Angel-style statue (SECOND, bounded form).** Body with a motion rule gated by being perceived; test whether the validator accepts "observer-gated" as a perception-triggered rule or rejects it with a clear error.
19. **Shadow person / Old Hag (LAW: sleep mode).** Not an entity at all but a body mode in which hypnagogic perception intrudes; a culture pack gives it a name (S).
20. **Freddy-style dream killer (LAW: shared plane).** Entity with a plane link into sleepers; dream plane rendered from minds with write-back to the body; rumour as the trigger.
21. **Dreamlands (LAW: plane).** Plane with its own clock offset, mind-driven contents, gated entry by sleep state; inhabitants as faith-substrate beings.
22. **SCP-style antimemetic (LAW: memory edit).** Pattern that removes the memory of the observer; keep only item number and the invented replacement entity; seeded rumour that no one can repeat.
23. **Warhammer-style daemon engine (COVERED).** Machine body with a patron-host pairing; faith-substrate patron; possession drift dial; rumours of cult mechanics.
24. **Evangelion AT Field (COVERED).** Ward pattern bound to a pilot link with a store and a drain; a mecha body in P; a neural sync dial as drift.
25. **Hounds of Tindalos (SECOND).** Pursuer that follows through time; the pack expresses it as a bounded plane-link pursuer instead and the validator must report what is missing.
26. **Genie granting wishes (SECOND).** True wish power fails; the pack expresses it as a patron with a bounded cost and a literal-minded mind; the failing field is "creation off by default".
27. **Bag of holding (LAW: pocket plane).** Pocket plane volume with conserved mass; opening is a doorway; pack states capacity and the pull-out rule.
28. **Kaiju (PACK).** Giant body in P (scaling laws), G variant for atomic breath (needs nuclear law, flagged LAW), a rampage drive; the rumour is the city's fear before arrival; crowd panic law flagged.

## 6. Licensing note

- **SCP** content is CC BY-SA. The coverage tables keep only the item number, title and link, and our own one-line summaries; no article text. MARK1 invents its own entities, inspired by the mechanisms, so share-alike does not apply. If we ever ship an SCP-derived item, attribute and share-alike apply.
- **Slender Man** (Eric Knudsen / Something Awful), **Siren Head**, some **creepypasta characters**, **Backrooms** (partly community CC BY-SA, partly owned) and **fandom-wiki** content (DCC, Beware of Chicken, Warhammer, STALKER, Fallout, Shadowrun) are owned or share-alike: we model mechanisms and invent our own names and designs. Named games, books and shows are used only as labels in the coverage tables.
- **Public domain** is safe to use directly: folklore, Grimm, Andersen, Lovecraft (largely), Poe, M. R. James, mythology, religious texts (as traditions, with the sensitivity note below).
- **Sensitive content.** Real-world ritual harm (child sacrifice, albino body-part trade, child witchcraft accusations) is kept as S-layer fear sources only; Dreamtime and living sacred traditions are excluded as content. (u: how the owner wants these handled; see OWNER_QUESTIONS.)

## Owner decisions (2026-11-22, CV1-CV8)

- **CV1, the order of the missing laws:**
  1. bounded rebuild (heal, polymorph, respawn);
  2. information attacks on minds;
  3. planes with their own rules.
  All of them come AFTER the normal simulation build (waves 15, 16a, 16b), as magic or as engine mods.
- **CV2, bounded versions of "second change" tropes:** build them now, and work toward the unbounded cases once there is a playable MVP:
  - a plane-reset loop with one carried pattern;
  - luck as a small, paid field bias on chaotic outcomes;
  - a rate-slowing ward that draws process energy;
  - observer-gated motion as perception plus behaviour.
- **CV3, bounded versions of the hard cases** (the unbounded ones wait for the pack interface):
  - travel into A VERSION of the past: a plane rebuilt from a stored snapshot, the traveller copied in, the original "dies", and the main timeline untouched;
  - TRUE WISHES: the closest lawful plan of magic that brings the wish about (monkey's-paw literalism emerges);
  - OMNIPOTENCE: standard magic extended at ruinous, unsustainable cost.
- **CV4, cognitohazards for the player:** real. The narrator shows the character's ALTERED experience, under a per-pack dial the owner controls.
  - INCOMPREHENSION DISTRESS: unresolved prediction error raises stress.
  - Resistance comes through habituation, understanding, training, a tolerance-of-uncertainty dial, and coping.
- **CV5, rewards for killing:**
  - possible as in-world facts, NOT the default;
  - a System entity only in packs;
  - the NATURAL version: real loot, food and status, and a slain magical being's stored M absorbed by nearby minds with coupling.
- **CV6, difficulty scaling:**
  - only as an in-world being that chooses to scale its challenges (a pack dungeon AI);
  - by default difficulty EMERGES: people settle and gather by danger, knowledge of danger spreads with its distortions, and survival of the fittest makes dangerous places dangerous.
- **CV7, sensitive real-world material:**
  - possible to build, as a fear source and as optional content, CENSORED BY DEFAULT;
  - the engine supports the full range of human behaviour;
  - no gruesome real-world material shipped for now;
  - nothing illegal or outside Claude's limits is ever built, whatever the dials;
  - living sacred traditions under MC8.
- **CV8, the first packs:** the 28 test cases, then one full genre (folklore).
