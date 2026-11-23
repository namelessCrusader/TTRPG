# MARK1 architecture and play-loop audit (synthesis)

Date 2026-11-22. Documents only: nothing was run, edited or committed. Built from five lane reports in `docs/arch-audit/`
(A worlds, B minds, C play loop, D code, E dev infrastructure). Each lane did not re-propose anything already in
`decisions.md`; this report does the same. Item ids (A1, B3, C2, D1, E1) point back to the lane files for the full text
(what exists with file:line, problem, options, cost, risk). Costs: S = a day or less, M = a few days, L = a wave.
All costs are estimates. Nothing was timed or measured by the lanes unless stated.

## 1. The picture in one page

1. **There are four worlds that share almost no behaviour.** Of the 34 catalog verbs, the voxel runs 14, the rooms inn 26,
   and both only 7. The lab inn has 13 verbs the catalog does not have (serve, ask, offer, sleep, ...).
2. **The rooms inn is a stale copy** of the lab inn (806 lines against 2,148; 14 flag groups behind). Nothing catches the
   drift. Everything the learned mind was selected on since 2026-11-08 exists only in the lab.
3. **The voxel has bodies but no needs, no appraisals, and no verbs to act on them.** The body can digest a meal and cannot
   be asked to eat. Hunger and thirst are `null` in the catalog. There is no liking, trust, debt or ownership claim anywhere
   in `src/voxel`. So neither the rules mind nor the learned mind has anything to want there.
4. **There is no play loop for the voxel world.** But `src/mind` already is one (player chooser, chains, addressed NPC),
   and the rooms inn can be played today. Missing: a narrator, saves on disk, a typed-input path, and a decision on who
   holds control while `World.step()` waits for the player.
5. **The learned mind is about 890 lines of world-free numpy** that can move into `src` once the mind interface can hold
   state. But it has not beaten the rules in any lab result (rules +0.38, effects -0.06; survival check not met).
   Port the interface, not the claim.
6. **The code is a 38-mixin chain** with 35 silently shadowed names, 20 attributes written by several mixins, and a tick
   that is one hand-ordered list. Files are huge (`heat.py` 137 KB, `will.py` 113 KB, `matter.py` 103 KB). There is no
   declared "what does this law read and write".
7. **Feedback is slow.** The full suite is 4,911 tests, about 62 minutes, serial, under one lock. xdist is installed and
   unused. The SLOW list was measured on 7 files and is stale. Work areas and agent practice caused real incidents this
   week (a 4 h polling loop, a lab chain holding the lock, a half-finished lane left in the live tree).
8. **76 files of wave 15 are uncommitted** with no snapshot. Any split, registry or shell validated against that moving
   base is validated against sand.

## 2. Findings by area

### 2.1 Worlds and their relation (lane A)

| id | finding | status | prio | level |
|---|---|---|---|---|
| A1 | The catalog is not the contract. 34 verbs; voxel 14, rooms 26, both 7; lab 37 (13 not in catalog); docs count 28 / 34 / 37. No parity test. Eating is a body law no act calls. | PA6 schedules "re-express the verbs"; the 13 verbs, parity test, single count NEW | HIGH | technical |
| A2 | Body needs are not drives in the voxel (`mindlink.py:298` reports only safety and harm; `bodyheat.py:595` already computes `DG.hunger`). The inn's drift is per turn, the body runs in joules. | SC1/SC6 decided; needs function and agreement test NEW | HIGH | technical |
| A3 | Rooms is a fork pinned at sha `0cea5952`; the lab file is `66084f1b`. `tests/test_rooms.py:81` cannot see the drift. | NEW | HIGH | **owner** |
| A4 | Social laws (liking, anger, trust, reputation, gossip) live in the World object, once, in the lab. A world-side `liking` is a fact no one perceived. "stance" means posture in 87 voxel files. | K1-K12 and AG1 partly; home and unit rule NEW | HIGH | technical (principle decided; one-line nod) |
| A5 | Rooms as the coarse twin: no SI units, no horizon, no materialise. The inn "night" is 24 h (300 turns x 288 s); the voxel inn night is hours. | SC5/SC7/Fam2 schedule the twin; class T/S split NEW | HIGH for unit rule, rest later | **owner** (instrument vs game part) |
| A6 | The inn and voxel scenes are hard-coded Python (3,466 lines of scenes). AU5 does not list scenes or casts. | NEW | MED, HIGH at the twin | technical |
| A7 | `src/core` (6,065 lines) called dormant but still cited by plan-cache design; README still describes it as the game; 34 test files run it in the suite. Per-file keep/port/retire table in the lane file. | NEW | HIGH (freeze), ports after 16a/16b | **owner** (overrules "does not delete") |
| A8 | The three packs (dcc, cultivation, farming) and the 132-line composer eval set are the only proof cases and are on the old format. | NEW | MED | technical |
| A9 | Three play drivers; only the rooms session plays through `src/mind`. | NEW | HIGH (cheap) | technical |
| A10 | Four belief stores and four event formats; judge and survival reports read only the inn's row format. | K1-K2 schedule schema; exporter NEW | MED | technical |
| A11 | No shared conformance suite for a Backend. | NEW | MED | technical |
| A12 | Inn flag sets accreted into a second engine; aliases (all5 vs all6) hide which arm a result used. | NEW | MED | technical |
| A13 | Ownership and value are god fields in the inn and absent in the voxel. Ownership must be a claim (AG1). Do not add `owner` to voxel things. | AG1/AG6 with 16b; removal NEW | MED, HIGH at first trade scene | technical |
| A14 | Supply, luck, raids, ailments are calibrations kept in prose comments; nothing links them to what will replace them. | NEW | LOW | technical |
| A15 | Names mean two things (World, mind, stance, night, turn). It already caused a wrong comparison. | NEW | LOW | technical |

### 2.2 Minds (lane B)

| id | finding | status | prio | level |
|---|---|---|---|---|
| B1 | The Chooser protocol is stateless, blind to refs and outcomes, and batched over sub-menus; it cannot hold a learner. Add a stateful Mind protocol (observe/decide/close/sleep/state/explain) with a ChooserMind adapter. | NEW | P1 | technical |
| B2 | Extract the world-free learners; replace inn-hardwired tables (VERBS, one-hot people, 43 KEYS, DRIVES/NATURE, turn-based time) with catalog-declared channels, dials and seconds. | NEW | P1 | technical (location = owner, see O2) |
| B3 | The voxel has no needs vector, so the learned mind has no reward there. Depends on 15c and 16a. | NEW | P1 | technical |
| B4 | Where the learned mind runs first (plumb on rooms now; voxel after 15c/16a). | NEW | P1 | **owner** |
| B5 | Where shared learner code lives (the isolation rule). Merged with A3. | NEW | P1 | **owner** |
| B6 | Mind conformance suite (determinism, snapshot, no-hidden-fact, LM-call ledger). Merge with A11. | NEW | P1 | technical |
| B7 | Cost: nothing measured (ms per decision, `bind()` per voxel decision, memory of `Mind.rows`). | NEW | P2 | technical |
| B8/B9 | Rules live in four places, one a trait table; the rule veto cuts the menu the learned mind could see. One data-driven rules mind, full menu. | NEW | P2 | technical |
| B10-B11 | Learned stores should project to typed Beliefs; learning boundaries are world events (sleep). | K1/K12 + NEW | P2 | technical |
| B12 | The LM boundary: a model only for player parse, wording a typed NPC reply, and optional narration; never for NPC decisions, perception, belief or adjudication. No live LM code exists in `src`. Add an engine-wide LM ledger; demote `featured`/Combined/Gate to an eval package. | NEW | P1/P2 | **owner** (O3, O4) |
| B13-B20 | Player autopilot hook (SC6), interrupts as one lens (PA3), `will.py` hook protocol, beliefs/reflex seams, judge signals, one "who chooses" slot, verb list as data, mark and retire `core`/`harvest.py`. | mixed | P2-P4 | technical |
| gate | A stated rule for "a learned mind replaces rules for this character". None met today. | NEW | P1 | **owner** |

### 2.3 Play loop (lane C)

| id | finding | status | prio | level |
|---|---|---|---|---|
| C1 | The playable MVP in two steps: MVP-0 on the rooms inn now, MVP-1 on one voxel scene after wave 15 batch 2 is committed. Neither is blocked by 15c, 15d, 15e, 16a, 16b. | NEW milestone (CV2, CV3 wait on it undefined) | P0 | **owner** |
| C2 | Narrator = a pure function of the character's own percept rows, beliefs and self-state; per-channel renderers and phrase packs as data. Gives K10 and MC7 by construction; lint plus leak test enforce it. | K10/MC7 sched; module NEW | P0 | technical (model-written prose = owner) |
| C3 | The player's question arrives inside `World.step()` (`will.py:806-930`). A parked world worker thread (or an equivalent hand-back) decides save, undo, replay and every front end. | NEW | P0 | technical |
| C4 | Turns, queue, `wait` with halt on salience. Which body acts are involuntary reflexes? | PA3 sched; shell NEW | P1 | owner for the reflex rule (at MVP-1) |
| C5 | Input in three tiers (numbered, lexical, model); port `pieces.py`; speech is closed speech acts until 16b's claim language. | PA2/plan cache sched | P0/P1 | owner for the speech limit |
| C6 | Save = true state + decision log + version stamp, not replay from seed. `World.snapshot()` is in-memory and leaves the mind out. | NEW | P1 | owner (compat promise, undo) |
| C7-C10 | One session-log schema; text front end and Frontend protocol; `src/play` layout with data packs. | NEW | P0/P1 | technical |
| C11 | Names and strangers leak under K10; needs a leak test. | NEW | P1 | technical |
| C12 | Play latency budgets, unmeasured. World cost per game second on this laptop is unknown (risk R2). | principle sched | P1 | owner sets the numbers after measuring |
| C13 | Scenario start, end and death. No voxel scene is a place you live in (risk R1). | NEW | P1 | owner (death rule, first scenario) |
| C14 | Agent playtest per wave; voxel version NEW. | old | P2 | technical |

### 2.4 Code architecture (lane D)

| id | finding | status | prio | level |
|---|---|---|---|---|
| D1 | World is a 38-mixin chain; hidden coupling by name (`WillMixin` calls into 17 mixins), 35 shadowed names, `_tick_end` defined in 7 mixins with no check that each calls `super()`. Option A: law registry (declared reads, writes, gate, after, twin, cost), staged, current order frozen as golden order. Option B: lint only. | NEW (prerequisite of AU2) | HIGH | **owner** (A vs B; I recommend A with the lint first) |
| D2 | A field table (name, dtype, unit, owner law, ledger, lazy, in snapshot, in golden) generates snapshot, restore, golden hash and the lazy list; touch-trace check of declared reads and writes. | NEW | MED | technical |
| D3 | Split big files by law, `motion.py` first (two mixins in one file), then `heat.py`, `will.py`, `mechlaw.py`. Only after the tree is committed, one file per lane. | NEW | MED-HIGH | technical |
| D4 | Layer rule as a test; package cycle `mpm` <-> `voxel` (one import line breaks it); move the mind-calling part of `will.py` behind `mindlink`. | NEW | MED | technical |
| D5 | AU5 data move, with two cautions: bit-exact load check, and numba constant capture. | AU5 + NEW cautions | HIGH | technical |
| D7 | Test tiers and per-lane locks (code side). Merge with E1/E3. | NEW | HIGH | technical |
| D8 | A four-call session API: `observe`, `act`, `advance(until)`, `save/load`. Decide before 16a ends. Merge with C3/C10: it is the same seam. | implied; API NEW | MED | technical |
| D9 | Numba hygiene (warm-up, kernel files, cache dir). Rust deferred until a measured need. | NEW | LOW | owner question later (shipping to people without Python) |
| D10d | Pin the `_law_*` call order with one test. Minutes of work. | NEW | MED, do now | technical |

### 2.5 Development infrastructure (lane E)

| id | finding | status | prio | level |
|---|---|---|---|---|
| E1 | Three test tiers from one timing file (T0 smoke at most 60 s; T1 fast at most 8 min with `-n 3..4`; T2 full), "affected tests" per lane. 4 workers x 2.5 GB is the memory ceiling on 14 GB RAM. | partly | P1 | owner nod on laptop load (heat) |
| E2 | Fact goldens (EA6) for behaviour; bit hashes only for refactors; declared diffs machine-read. | EA6 + NEW | P2 | technical |
| E3 | Slot pool by class instead of one `flock`. | NEW | P1 | technical |
| E4 | One sandbox builder; stop copying blender and `lab/vendor`. | NEW | P1 | technical |
| E5 | Switch ledger as data, in the smoke tier. | NEW/AU5 | P2 | technical |
| E6 | Agent practice: no `pgrep -f`, a background job registry, a pause protocol, git worktrees per lane, machine-checkable lane reports. | NEW | P1 (W1-W4), P2 (W5-W6) | technical |
| E7 | Lane commits on a local branch with owned paths and a prepush scan. | NEW | P1/P2 | **owner** (commit rule) |
| E9 | Disk: `~/.cache/mark1-build` is 62 GB (not lab); about 45 GB freeable with no loss; `lab/vendor` venvs 10.6 GB. | NEW | P1 | owner for deleting vendor venvs |
| E10-E12 | Tests by law, mutation harness; `make smoke/law/check/wave`; copy `decisions.md` out of a cache dir into the repo. | NEW | P1-P3 | technical |

## 3. Shared root causes (why the list is shorter than it looks)

1. **No declared seams.** Verbs (A1), needs (A2/B3), beliefs (A10), laws (D1), fields (D2), session (C3/D8) are each a
   convention held in code. Remedy in each case is the same shape: a declared table plus a conformance test. This is
   one habit, applied six times.
2. **Two copies of the inn and a hold on the isolation rule** (A3, A5, B2, B4, B5, A12, A4). One decision (O2 below)
   unlocks seven items.
3. **The mind has no stateful interface and the voxel has no mind-facing surface** (B1, B3, A2, A4, A13). One interface
   change (B1) plus one needs function (A2) unlocks the learned mind and the rules mind on both worlds.
4. **The live tree is shared, uncommitted and moving** (E6, E7, D10c, C-R4). Every structural change (splits, registry,
   shell on voxel) is unsafe until it is committed and each lane has its own checkout.
5. **A slow, serial, single-lock loop** (E1, E3, E4). It is why agents wait, why lanes pick the wrong tests, and why
   refactors are expensive.
6. **Old engine left half-alive** (A7, A8, A9, B20, D4, D10b, C9). A freeze plus a per-file table removes five findings.
7. **Units** (A2, A5, A15). Per turn versus per second caused a wrong comparison already and will invalidate any
   agreement test if left.

## 4. ONE RANKED LIST OF FIXES

Rank is by (unblocks play or silently invalidates results) then cost. "S" = scheduled already; "N" = new.

| # | fix | refs | S/N | cost | why this rank |
|---|---|---|---|---|---|
| 1 | **Commit the live tree** (wave 15 b1-b2 once end-checked), then work in per-lane worktrees or local lane branches. | E6 W5, E7, D10c, C R4 | N | S + owner nod | Everything below touches files that are dirty now. |
| 2 | **Test tiers and slots:** timing file, T0 smoke <= 60 s, T1 fast with `-n 3..4`, slot pool by class, `make smoke/law/check/wave`, `python -P` and the sandbox builder. | E1, E3, E4, E11, D7 | partly | M | Every other item is paid for in this loop. |
| 3 | **Agent hygiene:** `waitfor.sh`, job registry, pause protocol, name rule for WORK, copy `decisions.md` into the repo. | E6 W1-W4, E12 | N | S (2 h) | Cheap; prevents repeats of this week's incidents. |
| 4 | **Cheap guards now:** byte-hash drift test for the rooms fork (it fails today, that is the finding); step-order pin test; layer-rule test and the `mpm`/`voxel` cycle check; import test freezing `src.core`; fix the README and the empty `python3` file. | A3, D10d, D4, A7, A15 | N | S | A day buys four silent-failure alarms. |
| 5 | **MVP-0 shell on the rooms inn** (`src/play`: session API of 4 calls with a world worker thread, narrator v0 with phrase packs and a leak test, input tiers 1-2, save bundle, session log, text front end, agent CLI). Rooms session becomes the agent playtest harness at once. | C1, C2, C3, C5-C8, C10, A9 | N (milestone) | L | The only way to find out what play is like; touches new files only, no `test.lock`. |
| 6 | **The mind interface:** stateful Mind protocol with ChooserMind adapter; one conformance suite for Backends and Minds; an engine-wide LM ledger. | B1, B6, A11, B12 | N | M | Unlocks the learned mind and every backend-agreement test. |
| 7 | **One inn, one copy** (per owner answer O2): move the current inn into `src/rooms`, flags-on goldens first; conditions in data with a switch-ledger row per flag. | A3, A12, B5 | N | M | Until done, all lab results are stranded in the lab. |
| 8 | **Units rule and the needs function:** seconds everywhere; `Turn.seconds` passed to every rate; `needs(self_state)` in `src/mind` filled from the rooms drives now and from the body later; first agreement test (inn night vs voxel inn night). | A2, A5, B2, B3 | N | M | The learned mind's survival measure is meaningless without it. |
| 9 | **Verb parity:** put the 13 inn verbs in the catalog, parity table as a test, one verb count; voxel programs ride on 16a. | A1 | N + PA6 | M (+ 16a) | The catalog becomes the contract. |
| 10 | **Law registry, staged:** lint first (shadowed names, `super()` in hooks, multi-writer attributes), then registry with the current order frozen as the golden order, then the field table. | D1, D2 | N | M then L | Prerequisite of AU2 (mods) and of safe agent edits. |
| 11 | **Relations store in `src/mind`** (percept-fed attitudes: liking, trust, anger, debts, claims) with the rename `stance` -> `attitude`. | A4, A13 | partly | M-L | Appraisal belongs to the mind (no god channel). Do with 16a/16b. |
| 12 | **Split by law:** `motion.py` first, then `heat.py`, `will.py`, `mechlaw.py`, `matter.py`; move the mind-calling part of `will.py` behind `mindlink`. Between waves only. | D3, D4, B15 | N | L | Editability; safe only after 1 and 10. |
| 13 | **Narrator for the voxel, `wait` with halt, one voxel scenario (MVP-1)**; measure cost per game second and set the latency budget with the owner. | C1, C12, R1, R2 | N | L | Needs item 1 and wave 15 batch 2 committed. |
| 14 | **Port from `src/core`, then retire** (`pieces.py`, composer and eval set, session protocol, bless rule, `address()`, oracle, `reach.py`); extract the three packs as case cards. | A7, A8 | N | M slices | After 16a/16b; each port moves its tests, the suite shrinks. |
| 15 | **Learned mind into `src`:** extract the world-free learners, catalog-declared channels, per-character opt-in, a written acceptance gate. | B2, B4, gate | N | M-L | Plumbing now on rooms; voxel life after 15c/16a. |
| 16 | **The LM boundary** (parse, wording, optional narration; demote `featured`/Combined/Gate to an eval package). | B12 | N | M | Enforces the latency budget in code. |
| 17 | **Rooms as a coarse twin:** class T/S split, stand-in table with sources, horizon list, materialise; scenes and casts as data with a voxel exporter. | A5, A6, A14 | SC5/SC7 + N | L | At the family step. |
| 18 | **Belief schema unification and `ev_to_row` exporter**; ownership as a claim. | A10, A13 | K1-K2 + N | M | With 16b. |
| 19 | **Disk clean-up** (about 45 GB freeable; vendor venvs after owner nod). | E9 | N | S | Housekeeping; 56 GB free today. |
| 20 | **Fact goldens, mutation harness, tests by law, numba hygiene, glossary.** | E2, E10, D9, A15 | EA6 + N | M | Later quality work. |

## 5. THE SMALLEST PLAYABLE MVP

**One sentence.** A person starts a scenario from a short list, reads what their character perceives as prose, acts by
number or a short typed phrase, queues a few steps and lets time pass with interruptions, talks to an NPC, saves, quits and
resumes; every session leaves a log that replays. NPCs and the world run by their own laws with zero model calls; a model is
called only for a typed phrase the matcher cannot place and for an addressed NPC's reply, and everything works with both
switched off.

**What it contains.**
- `src/play`: a shell over the Backend protocol with the four-call session API (`observe`, `act`, `advance(until)`,
  `save/load`), the world running in a parked worker thread so the player's question can wait inside `World.step()`.
- A narrator: a pure function of the character's own percept rows, beliefs and self-state, with phrase packs as data;
  never a tick, an id or a raw number unless asked.
- Input tiers 1 (numbered) and 2 (lexical: the `pieces.py` grammar ported onto mind Options).
- A save bundle: true state + decision log + version stamp. A session log schema that replays to the same world hash.
- A text front end (always available) and an agent CLI over the same session API.
- Scenario files (a premise paragraph, premade sheets), an end/death rule.

**What it is not (cut on purpose).** No character creator, no quest engine, no graphics, no real-time control, no free-text
conversation understanding (needs 16b's claim language), no magic, no multi-settlement world.

**Two steps.**
- **MVP-0, on the rooms inn, start now.** The inn is the only world with persons, needs, trade, theft, violence, speech
  and a finished Backend; it runs in cheap turns and has goldens. Depends on nothing in the build order; it touches only new
  files and fast tests, so it can proceed while wave 15 is paused.
- **MVP-1, one voxel scene.** Depends on wave 15 batch 2 being merged, end-checked and committed, and on one scenario that
  is a place you live in (a workshop or forge is recommended: 14b and wave 15 physics are strongest there). Uses today's
  voxel verbs. 15c, 15d, 15e, 16a and 16b improve it but do not block it.

**Where it sits in the build order.**

| build step | relation to the MVP |
|---|---|
| now (paused) | MVP-0 on rooms, in parallel; plus items 1-4 above first. |
| wave 15 batch 2, 3, 4 (healing) | Batch 2 committed is the gate for MVP-1. Batches 3-4 add bodies to the narrator as data. |
| 15c solids | Not required. Needed for the voxel needs vector (item 8) and richer bodies. |
| 15d light | Narrator works in the dark as data once light channel exists. |
| 15e gas fire | Not required. Smoke and heat percepts become narrator data. |
| 16a primitives | PA2(a) list of perceived things and PA2(b) typed phrase to primitives replace tier 2 matching; shortcuts as plans; voxel programs for the 13 inn verbs; the session API's `act` is born here. |
| 16b knowledge and plans | Tier 3 plan cache; K10 belief-labelled menus; K12 "who told you?"; free-text speech. |
| magic | After; a pack-level content job, not a shell change. |

**What must come first.** (1) Commit the tree (items 1-2); (2) the leak test and no-digit test written before the narrator
(acceptance tests A2 and A5 of lane C); (3) a measurement of world cost per game second on this laptop (risk R2: if a
3-second job costs several wall seconds, play stalls whatever the shell does); (4) the decision of who holds control while
`step()` waits (C3).

**Acceptance tests (stated first).** An agent plays 30 decisions on the inn and 30 on a voxel scene through the CLI and the
log replays to the same world hash; a leak test (hide an entity by wall, dark or distance, over 100 seeded states, including
known names): no narrator line names it; the whole scripted playthrough runs with models disabled; reactive turns cost 0 model
calls (counted by the existing `Mind.remote` meter); save, quit, load, continue gives the same hash as an uninterrupted run.

## 6. Proposed placement of the new work

```
src/mind     (stdlib + numpy only, knows no backend)
   catalog, menus, choosers, Mind protocol (stateful) + ChooserMind adapter, needs(), relations (attitudes, claims),
   narrator inputs, pieces.py (ported), LM ledger, conformance suite, learned learners (world-free numpy; O2)
src/rooms    the coarse world: the CURRENT inn moves here (O2), scene pack as data, class T/S laws
src/voxel    the truth; mindlink fills SelfState from the body; law registry + field table; session seam is a 4-call API
src/play     NEW: session (4 calls + worker thread), narrator + phrase packs, input tiers, save/log, text front end, agent CLI
src/core     frozen; ports leave file by file; retired at the end
lab/         experiments only; imports src.mind and src.rooms one way (O2); keeps its own venv for torch
tools/ + Makefile   test tiers, slots, sandbox, job registry, waitfor
docs/        glossary (World, mind, attitude vs stance, day vs night, turn)
```

## 7. What to keep as is

- The voxel physics as the truth, and the principle that the coarse world is a twin, not a second truth (SC4, SC5).
- `src/mind` as the neutral interface: it imports nothing from voxel, rooms, core or lab (checked by grep), and
  `src/rooms` imports only `src/mind` and the stdlib.
- The record and replay crossing for the decision model (JSONL cards, lab never imports `src` today) and the catalog's
  `backends:` structure.
- The two-level menu (scope then act) and the tournament merge.
- The compat-goldens idea (hold a port byte for byte to a reference) and the voxel switch ledger discipline (owner, reason,
  `deleted_when`), which the inn should copy.
- The inn laws and their lessons (five quarrel causes, meals and `serve`, bring-to-the-bedridden, one self-care function,
  sourced supply and bandit rate); move the laws, not the file.
- Lazy-law seam, slow twins, fact-goldens plan (EA6) and the strong end-to-end bit goldens.
- The learned mind code and tournaments as the place experiments run, but not its claim.

## 8. Decisions I take myself (technical)

Merge D8, C3 and C10 into one session seam. Merge A11 and B6 into one conformance suite. Do the byte-hash drift test, step-order
pin, layer-rule test and `python -P` today. Appraisals live in the mind, percept-fed (principle already decided: no god
channel). Seconds are the unit and a turn is a stated jump length. Rename social `stance` to `attitude`. Model-free rules
mind gets the full menu with no rule veto. Test parallelism is trialled at `-n 3`, not 16. Order of splits: `motion.py`,
`heat.py`, `will.py`, `mechlaw.py`, `matter.py`. Data move (AU5) gets a bit-exact load check and a numba-constant check.

## 9. Not measured by any lane (verify before relying)

Learned mind ms per decision in either world; `bind()` cost per voxel decision; memory of `Mind.rows` per hour; world cost per
game second on this laptop; whether the 12 off-by-default lab options are dead; the lab's current per-night time with all
modules on; the 4-worker memory and heat limit on 14 GB RAM.

## 10. Owner questions raised (details in `~/.cache/mark1-build/arch-audit-questions.txt`)

1. Build the MVP in two steps, rooms inn first, in parallel with the paused waves?
2. May the lab import `src.mind` and `src.rooms` (and a shared numpy-only learner package) one way, so there is one inn?
3. Is the rooms world a lab instrument first and an offscreen game part later (twin claim only for class T laws)?
4. Learned mind: opt-in per character, with a written gate before it replaces rules; plumb on rooms first?
5. Models in play: only parse and addressed-NPC wording; narration by templates with model polish as an opt-in dial; `featured` demoted to eval?
6. Freeze `src/core` now and delete it in stages as each piece is ported?
7. Law registry staged (lint first), or lint only?
8. Local lane commits on a working branch (no push, no attribution), and may the 10.6 GB vendor venvs be archived?

Deferred to the start of MVP-1 (not asked now): save compatibility promise and undo dial; death of the player's body (end
with reveal vs play on); reflex rule while a queued job runs; first voxel scenario; latency budget numbers after measuring;
dialogue before 16b as menu-like speech acts.
