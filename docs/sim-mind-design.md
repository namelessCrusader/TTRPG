# Sim and mind: how decision models will choose actions in the voxel world

**Status:** design, 2026-10-01. Nothing in `src/`, `tests/`, `lab/` or `tools/` is changed yet.

**What this file is.** It merges five topic designs into one plan:

1. split `src/voxel/sim.py` into modules;
2. an entity layer (things with identity and names);
3. typed events, perception and beliefs;
4. the mind interface (`src/mind`) with two backends;
5. particle (MPM) speed and the seam between particles and voxels.

Section 3.0 settles the places where the five designs disagreed.

**Line references.**
- `sim.py:N` means `src/voxel/sim.py` in today's working tree: 8,794 lines, sha256 `4d0b87e91545…`. It has uncommitted changes on top of 6120b6e, the last commit that touched it (HEAD is 54fcc60).
- After the split (step S2), code is found by symbol, for example `hands.py::_object_at`. The generated `docs/sim-split-map.md` turns old `sim.py:N` references into new locations.
- The prototypes cited here are in this session's scratchpad, under `scratchpad/design/`. Step S1 copies the ones that later steps need into `tools/protos/`, so that they survive a `/tmp` cleanup.

---

## 1. Context

### 1.1 What the decision models need

The lab measured GLiNER2.5-Decide (340M parameters, about 30 ms per choice). Each finding below becomes a requirement on the engine.

| Lab finding | What the engine must do |
|---|---|
| It picks from a list of short option texts, given a scene text. Accuracy is 62% with 16 options and 47% with 64. | Give menus of **16 options or fewer**, after a rules pre-rank. The menus offer tasks, not limb moves. |
| A sentence that names a useless thing makes it pick that thing (100%). | Write a **clean input**: name the people and what they hold, the character's own state, the last few events it perceived, and at most 2 relevant beliefs. Say what is **missing** for a task. Never list inert things. |
| It reads stated facts well (where, goal: 100%). It infers badly (danger, fault, false belief). | Facts, beliefs, perception and "what is missing" come **from the engine**, never from the model. |
| Its common sense lives in words ("the axe", "the rope"). | Things need **names**. "The iron" throws that knowledge away. |
| A capability hint ("cut: something with a sharp edge") raises accuracy from 62% to 75%. | Verbs declare the capabilities they need, and the engine computes them (for example `sharp` from the `edge` field). |
| Personality chooses among the options the facts allow. | Traits act as a **guard and a bias** inside the legal menu, never against it. |
| Real play (`docs/dnd-scenes-review.md`): most turns are obvious; wrong beliefs come from what a character perceived; others see an act start and react; a cause log explains acts. | Perception produces beliefs, including wrong ones. Acts are events that others can perceive. Every act has a cause. |

### 1.2 What the code is today

| Fact | Where |
|---|---|
| Three engines share no code. **v2** (`src/core`, about 6k lines) is dormant. It has a typed `Effect` on one log, perception, derived menus and YAML verb packs. **v3** (`src/voxel` + `src/mpm`, about 12k lines) is active. **The inn** (`lab/world2`, 731 lines) is a lab testbed. | `src/core/effects.py:23,71`; `src/core/engine.py:38`; `lab/world2/world.py` |
| One class holds every law. `World` has 191 methods: fluids, heat, burning, air, acid, free bodies, skeleton, physiology, senses, menus, will, speech and ledgers. | `sim.py:908-8794` |
| There is no entity layer. A thing is a same-material flood fill, so "identity is adjacency". Labels are material words. A body holds one item, and reach offers at most 2 things. | `_object_at` `sim.py:8318`; `_held_cells` `6138`; `MATNAME` `804`; reach stop `6134` |
| Body ownership is re-derived every tick by a flood fill. Two people who fall together can merge (bug 87). | `_claim_bodies` `sim.py:5004` |
| Minds work per limb. Every 10 ticks (one tick is 25 ms), `_decide` asks the policy one menu per limb (legs, hands, mouth, waist, eyes). Each menu is legality-checked and capped at 7, and index 0 is the null act. | `_decide` `sim.py:7386`; `_menu` `7112`; `LIMBS` `348`; `MENU_CAP` `438`; `decide_every` `764` |
| Control is inverted: `step()` calls the policy. | `_law_will` `sim.py:8671` → `_decide` → `policy.pick` `7426` |
| Events are free text in 35 places (`p["events"]`, `w.speech`, `w.sounds`). The harvest and the scenes clear them after every step. | `grep -c '\["events"\].append' src/voxel/sim.py` = 35; `harvest.py:174-176` |
| The particle solver costs 451 ms of a 466 ms tick for one 56-particle plank at 5 cm voxels. | measured, `scratchpad/design/mpm2/budget5cm.py` |
| Memory is tight: about 1.6 GB free while this was written, with 5 GB of swap in use. | `free -m` |

### 1.3 What this design does

- **It keeps the voxel world as the one world** where models will act. The inn becomes a second, cheap backend for building and testing the mind interface.
- **It improves structure first:** a split into modules, one golden check, and test tiers.
- **Then it adds what the models need:** things with names, typed events, perception and beliefs, and a task-level mind interface.
- **It makes the particle solver about 100 times cheaper for small objects** (P1–P3 measured; P10 projected). It also fixes the seam between particles and voxels, which today loses heat and duplicates matter.

**It does not:**
- delete `src/core`;
- add demotion (particles back to voxels): roadmap 93, decision 3, rules it out (`docs/voxel-roadmap.md:5978-5985`);
- run a model live without the user's approval (question Q3).

### 1.4 Terms used in this file

| Term | Meaning |
|---|---|
| **golden check** | `tools/golden.py`. It steps a set of small worlds ("cases") and hashes every World field after every tick. **SAME** means bit-identical. **DIFFERS at X** names the first case, tick and field that changed. |
| **E step / P step** | An **E** (exact) step must give SAME on every golden case. A **P** (physics or behaviour) step declares the cases it may change. Its checks are tests, tolerances and ledgers. |
| **mixin** | A class that holds a group of World methods in its own module. `World` inherits from all of them. |
| **MPM** | The particle solver (`src/mpm`). Matter that falls or topples is "promoted" from voxels to particles. Each tick it is "rasterized" (drawn back) onto the voxel lattice as `moving` cells. |
| **seam** | `src/mpm/matter.py`. It is the code that moves matter between the lattice and the particles. |
| **numba** | A compiler for Python numeric loops. The sim already uses it when it is installed (`sim.py:48-57`) and keeps a plain numpy path as the definition. |
| **pid / eid / ev_id** | `pid` is a person's index in `w.persons`. `eid` is an entity (thing) id. `ev_id` is an event id. See section 3.0, item 1. |
| **news** | A percept whose source event this mind has not had recently. Only news makes a body decide again. |
| **reflex tier** | The per-limb menus and `TablePolicy` (`sim.py:441-486`). It reacts in the same tick and makes 0 model calls. |
| **job** | A task the mind chose, for example "take the axe". The sim carries it out over many ticks through per-limb rows. |
| **chooser** | What picks an option from a menu: Rules, Random, Gate, Combined, Replay, Remote (the model) or Player. |
| **wave** | A set of build steps that run at the same time and own disjoint files (section 4). |

---

## 2. The target architecture

### 2.1 The picture

```
  ┌────────────────────────────────────────────────────────────────────────────────┐
  │ MODEL CHOOSER SERVER (lab venv, a separate process; offline by default)          │
  │  GLiNER2.5-Decide: rank(scene text, <=16 option texts) -> scores                 │
  │  lab/methods/score_requests.py (record -> score -> replay)  |  lab/serve.py (opt-in)│
  └──────────────────────────────▲─────────────────────────────────────────────────┘
                                 │ JSONL cards: requests out, decisions back
  ┌──────────────────────────────┴─────────────────────────────────────────────────┐
  │ MIND INTERFACE  src/mind  (standard library only; imports no backend, no lab)    │
  │  perceive -> beliefs(<=2) -> menu(pre-rank <=16, canonical order, null act)      │
  │  -> describe(clean text) -> chooser -> act -> job (start/act/finish, interrupt)  │
  │  choosers: Rules, Random, Gate, Combined, Replay, Remote, Player                 │
  │  cause log (Why per decision) · task-level harvest · verbs.yaml catalog          │
  └───────────▲────────────────────────────────────────────────▲───────────────────┘
              │ Backend protocol                                │ Backend protocol
  ┌───────────┴────────────┐                    ┌───────────────┴────────────────────┐
  │ ROOMS  src/rooms       │                    │ VOXEL LINK  src/voxel/mindlink.py   │
  │ port of the lab inn    │                    │ VoxelBackend + MindPolicy           │
  │ (rooms, traits, beliefs)│                    │ (prepare / limbs / pure pick)       │
  └────────────────────────┘                    └───────────────▲────────────────────┘
                                                                 │ reads
  ┌──────────────────────────────────────────────────────────────┴─────────────────┐
  │ PERCEPTION AND BELIEFS  perceive.py, beliefs.py                                  │
  │  w.minds[pid]: perceived, got, bel (<=64 keys), harm, attention state            │
  │  saw / heard / felt / told / did records; beliefs carry their source event       │
  ├─────────────────────────────────────────────────────────────────────────────────┤
  │ TYPED EVENTS  events.py: Ev(id, verb, actor, target, instrument, phase, result,  │
  │  place, origin, cause, link) · w.events ring · w.emit · w.because · w.why        │
  ├─────────────────────────────────────────────────────────────────────────────────┤
  │ ENTITY LAYER  things.py, label.py: w.eid field, registry, name(), props(),       │
  │  _law_identity (splits, deaths, debris), thing() authoring                       │
  ├─────────────────────────────────────────────────────────────────────────────────┤
  │ REFLEX TIER  will.py, policy.py: per-limb menus, REFLEXES, TablePolicy,          │
  │  attention law (latch | onset). 0 model calls. Falling picks always land here.   │
  ├─────────────────────────────────────────────────────────────────────────────────┤
  │ PHYSICS LAWS  World mixins: fields fluids heat air support bodies skeleton pose  │
  │  persons physiology senses motion hands ledgers · kernels.py (numba)             │
  │  + src/mpm: MLS-MPM (windowed, numba) and the seam (promote, rasterize,          │
  │  write-back, matter events)                                                      │
  └─────────────────────────────────────────────────────────────────────────────────┘
```

**Import rules:**
- `src/mind` imports nothing from `src/voxel`, `src/rooms` or `lab`.
- `src/mpm` imports nothing from `src/voxel`.
- `lab` and `src` never import each other. They exchange JSONL files.
- `src/core` stays dormant, and no new code imports it. Its ideas come back in the new layers:

  | `src/core` idea | where it returns |
  |---|---|
  | the `Option` dry run | `Option.ok` / `Option.missing` |
  | verb packs as data | `verbs.yaml` |
  | the typed `Effect` with a cause | `Ev` with `cause` |

### 2.2 One tick, in order

Today's call sites are given in brackets. The hook names are added by step S3.

1. `_tick_start` resets the current cause.
2. `matter.unrasterize` [8631]. From X3 on it first writes the lattice's edits back into the particles. Then `matter.step` [8632].
3. Bodies, fluids, grips, footing, support and torque [8633-8641].
4. `matter.rasterize` [8642], then `_after_raster`, which turns matter events into typed events.
5. Conduction, crack, rise, burn, boil, melt, acid, smoke, O2 and air [8643-8667].
6. `_law_identity`, new, placed before `_law_bleed` [8668]: hygiene, splits, debris and deaths of things.
7. `_law_bleed`, then `_law_life` [8668-8669].
8. `_law_perceive` replaces `_law_hear` [8670]. In order:
   - fire bookkeeping, computing `burning()` once, under the gate `_law_will` uses today [7649-7650];
   - `_law_hear`, unchanged;
   - heard and sight records, the "looked" pass, and sightings of things.
9. `_law_will` [8671]:
   - the attention law decides who decides;
   - `_decide` builds per-limb menus;
   - the policy's hooks `prepare` and `limbs` run, then a pure `pick`;
   - acts run inside `w.because(decision)`.
10. Pose, joints and radiation [8672-8697].
11. `_tick_end` flushes `breaks` and fire ends. Then `tick += 1` [8698].

### 2.3 One decision, end to end

1. Something happens. The sim writes an event, for example `hits` (actor = the attacker's pid, target = `("person", 1)`, cause = the `swings` event).
2. `_law_perceive` writes a `felt` record into the victim's mind. The record carries the event id.
3. The attention law sees **news** (an event id this mind has not had). The body decides.
4. `MindPolicy.prepare` checks the running job against the news.
   - A STANDING interrupt (salience above the job's threshold, and the kind is in the verb's `interrupt_on` list) stops the job.
   - The mind interface builds a request: the menu is pre-ranked to 16 or fewer options, and `describe` writes the scene text.
5. The chooser answers.

   | chooser | when it is used |
   |---|---|
   | Rules | every NPC by default; 0 model calls |
   | Remote | featured characters, and the character the player addressed |
   | Replay | runs that re-apply recorded model decisions |

6. The chosen job compiles into per-limb matchers plus `p["intent"]` (aim, flee, take, wake_at). Only the limbs a step needs are asked.
7. The act runs, and the sim writes typed events. The job's step test reads those events: "a `takes` event by me since the job started".
8. The decision row (request, menu, cut options, pick, chooser, `Why`) goes to the cause log and to the task-level harvest.

### 2.4 Invariants (each one has a test)

- A chooser sees only legal options. `act()` checks the option again against the live world.
- An NPC menu has 16 options or fewer, always includes the null act, and has unique option texts. A player's menu is never cut.
- A chooser never sees payloads, belief truth, ticks, raw numbers, entity ids or the reflex table.
- Facts come from the engine. A heard sound never reveals who made it, unless the listener knows that voice and understood the words.
- Reactive mode makes **0** model calls on every tick, including interrupt ticks. Featured mode has at most 1 outstanding request per featured character (memory note *latency-budget*).
- No flag replaces cognition. The attention law is a world-level law that applies to every body (memory note *no-special-case-subsystems*).
- The sim has no random number generator. Choosers that need one seed it from `(world seed, who, request id)`.
- `snapshot()` never copies the mind or a second World.

---

## 3. Decisions by topic

### 3.0 Where the five designs disagreed, and the ruling

| # | Conflict | Ruling |
|---|---|---|
| 1 | Three things were called "eid": an entity id (entity design), an event id (events design) and a particle entity (MPM design). | **`eid`** means an entity (thing) id only: the int32 field `w.eid` and the particle field `mpm.eid`. An **event id** is `Ev.id`, stored as `ev_id` on sounds and bodies. A person is a **`pid`**, its index in `w.persons`. That index is stable because persons are only ever appended (`sim.py:4987`). Each promote call stamps a **`batch`** serial on its particles. |
| 2 | Two fixes for the snapshot "ghost World": a deepcopy memo sentinel (split design C4) and `Matter.__getstate__` (MPM design P0b). | **The memo sentinel** in `ledgers.py` (step S5). It also handles restoring into a different World, and it is prototyped with a test (`design/ghost_test.py`). The MPM version is dropped. |
| 3 | Three perception stores: `p["perceived"]` and `p["thing_seen"]` (entity), `w.minds[pid]` (events), and `p["news"]` plus `p["_attn"]` (mind). | **One store: `w.minds[pid]`**, a slotted `Mind` held outside the person dict. Sightings of things become beliefs `("at", ("thing", eid))`. The news ring and the attention table become Mind fields. The policy writes only `p["intent"]` and `p["job"]`. |
| 4 | Two typed logs: `w.acts` and `w.thing_events` (entity), and `w.events` (events). | **One log, `w.events`.** Identity changes are verbs: `splits`, `breaks`, `catches_fire`, `ashes`, `melts`, `dissolves`, `gone`. |
| 5 | Job steps were checked by new flags (`_took`, `_swung`, `_arrived`, `_gave_up`, `_struck`, and a body serial `bid`). | **Typed events instead**: the actor's events since the job started. `swings` carries `result`, because today the text is written even when `_swing` returns None (`sim.py:7476-7487`). Being struck is a `hits` event that targets me. The serial is `b["ev_id"]`. The cost is that the voxel mind link (M7) waits for the events lane (V2, V3). |
| 6 | Particle identity was planned twice: entity carry sites 15–17 and MPM step P5a. | **The MPM lane owns every `src/mpm` edit.** Rasterize uses the entity rule: a drawn cell takes the eid with the most mass *among particles of the cell's material*, ties going to the lower eid. Only `_law_identity` creates new eids on a split. MPM supplies the facts. |
| 7 | Matter events: MPM step P7 (`MatterEvent`) and an events-design hook in `Matter`. | **One producer, one consumer.** `matter.events()` produces them, and the events layer converts them into `w.emit` calls at `_after_raster`. |
| 8 | The guard that stops hands taking particle matter appeared twice (entity A0, MPM P6). | **Entity step E0 owns it.** |
| 9 | Names: `w.name(eid, fallback)` (entity) against `w.names[(mat, edge)]` plus `_label` (mind). | **The entity layer only.** The first voxel mind link (M8) runs with Rules and does not need names. Model runs wait for named scenes (E5). |
| 10 | New `sit` keys for things (entity), none (events), `onset` (mind). | **No per-limb `sit` keys for things.** The model reads the task-level `describe` from `src/mind`. `sit["onset"]` exists only when `attention == "onset"`. The seam spelling becomes v2 only when names reach the menu rows (E5). |
| 11 | Three golden tools: `golden.py`, `golden_ev.py` and the mind design's S0 decision goldens. | **One `tools/golden.py`.** It leaves out, by name, fields that are absent from the base recording (and lists them), and it gains `--except` and the `duel150` decision case. |
| 12 | Two perception slots: `_law_notice_things` and `_law_perceive`. | **One `_law_perceive`** at `sim.py:8670`. |
| 13 | Sleep for particles (MPM P8), which the roadmap names as next-phase work (`docs/voxel-roadmap.md:6039-6041`). | **Question Q4.** Nothing depends on it. |
| 14 | Turning control around. | **Keep the inversion.** 58 call sites set `.policy`. Add optional hooks (`prepare`, `limbs`, `on_restore`) and keep `pick` a pure function. |

### 3.1 Split `src/voxel/sim.py` into modules

**Decisions:**
- **One atomic change, generated by a tool.**
  - `src/voxel/` gains 15 mixin modules plus `kernels.py` (numba kernels), `materials.py` (tables) and `policy.py` (`WILL`, `LIMBS`, `ACTS`, `REFLEXES`, `MENU_CAP`, `TablePolicy`).
  - `sim.py` keeps `World.__init__` (909-1033) and `step` (8610-8698), which fixes the law order. It re-exports all 102 old names explicitly, because `import *` skips names that start with `_`.
  - `sim.py` shrinks from 8,794 to 375 lines. The largest new module is `pose.py`, at 1,155 lines.
  - The method-to-module map is in `design/splitmap.json`. `will.py` holds `_menu`, `_full_menu`, `_decide`, `_law_will`, `_decide_falling` and `trace_outcomes`. `hands.py` holds the reach, grip and haul code. `persons.py` holds `_claim_bodies`.
- **Every cross-module call is `self.x()`,** so no mixin imports another. Imports run one way: `kernels`/`materials` → `policy` → mixins → `sim` → `scenes`/`demo`/`room` → `harvest`.
- **Sha pin.** `split.py` and `carve.sh` refuse any `sim.py` whose hash is not `4d0b87e9…`. Comments are re-homed by explicit ranges, and `astsame.py` proves that the ASTs are identical (191 methods, 4 class attributes, 80 top-level statements) and that no new module imports `.sim`.
- **One test edit.** `test_an_ARM_CANNOT_SWEEP_THROUGH_what_it_would_CLEAR_at_the_end` (`tests/test_voxel.py:2914`) patches `S._SWEEP_RAD`. Its import becomes `src.voxel.pose`. The edit is found by content, not by line, because the same import at 3841 must not change. The rule is: patch the module that *reads* the global.
- **A permanent hygiene test,** `tests/test_module_hygiene.py` (0.3 s), with three rules:
  1. every global a function reads is bound in its own module;
  2. there is no `globals()` and no `global` statement outside an allow-list;
  3. tests patch only the module that reads the global.
- **Follow-up commits** (wave 1, steps S4 and S5):
  - **C3:** `_menu(…, cap=None)` replaces the `globals()["MENU_CAP"]` swap in `_full_menu` (`sim.py:7377-7384`). This is also the hook for menus of 16 or fewer.
  - **C4:** the ghost fix.
  - **C5:** `_crouch` reads an undefined `joints` (`sim.py:6564-6565`), in a branch that would raise today. The fix is `joints = p.get("joints")`, plus a test that forces the branch.
  - **C6:** comments that describe positions ("below", "this file").
  - **C7:** delete 7 dead methods: `solid`, `_chain_xform`, `_apply_xform`, `_inv_xform`, `_comp_xform`, `_joint_xform` and `_flood_person`.

**What the split buys:**
- the decision seam becomes two small files (`will.py` 917 lines, `policy.py` 227);
- workstreams stop colliding in one file;
- numba caches stop going stale: about 1.2 s of recompiling per process after every edit of `sim.py`;
- a golden tool every later optimization can use.

**What it does not buy:**
- no speed (fire_alarm 59.4 s against 57.9 s);
- the mixins are not seams: `WillMixin` calls 38 methods in 7 other mixins;
- control is still inverted.

**Evidence** (scratch, under the lock):

| check | result |
|---|---|
| golden on the full carve | 34 of 34 cases SAME |
| hash-seed probe | 16 more cases SAME |
| the 5 voxel test files on the carve | 173 passed in 1,213 s serial, 1.04 GB peak memory |
| `harvest.to_file(range(32), workers=1)` | `broke == 0` (178 rows) |
| negative control: `_full_menu` moved out of `will.py` | golden `recall_hall` DIFFERS; hygiene rule 2 fails. The *current* cap test still passes, which is why S1 strengthens it. |

**Risks:**
- **Freeze.** Any edit to `sim.py` between S0 and S2 makes the carve refuse. Recovery means re-running `deps.py` and `splitmap.py`, re-checking the comment ranges, re-pinning the hash and re-recording golden (about 3 minutes).
- **What golden proves.** Golden proves only the paths its cases reach. A branch that no case enters is covered only by the identical-AST proof.
- **numba caches.** The 36 numba cache files under `src/voxel/__pycache__/sim.*` become orphans, so `carve.sh` deletes them. The first pooled harvest after the carve compiles once in every worker, so run golden first to warm the cache.
- **Things the split keeps exactly as they are.** The `arms` scene no longer swings its pendulum under `step()`: the arm goes from 1.2 to 0.0 rad between ticks 108 and 114, and `_limb_cells` is never called. The body work must own this. The split preserves it bit for bit.

### 3.2 Particle speed and the seam (MPM)

**Measured today** (512-particle block, ms per substep):

| case | today | active window (exact) | window + numba (exact) |
|---|---|---|---|
| block, 13,824 nodes | 7.21 | 5.17 | 1.96 |
| 400k-node house grid | 155.65 | 5.44 | 2.14 |
| tree via `promote`, 3,662 particles, 112k nodes | 41.42 (8.9 s per tick) | 28.98 | 14.80 |

- **The SDF rebuild** (signed distance to walls, `core.py:235-278`, with `np.apply_along_axis` at 240) costs 6,072 ms at 400k cells today. The numba version costs 208 ms. A windowed field costs 4.7–7.3 ms at any world size.
- **The seam loses heat.** A heated promoted block's energy ledger drifts −22,883 J, against −2,889 J for a block that was never promoted. The write-back prototype gives −2,889 J.
- **The seam duplicates matter.** A lattice shove of drawn particle matter took 600 g of wood to 1,200 g (`design/review/drawn_shove.py`). Lift, carry and drag all go through `_shove` (`sim.py:7073, 7107, 8207`).
- **One case crashes.** The `MAT` rows for ash, char and tin have 6 fields (`src/mpm/core.py:102-104`), and `promote` unpacks 7 (`src/mpm/matter.py:104`). A toppling charred cluster raises ValueError.

**Decisions.** Build steps: P0a–P3 are X1, P4–P5b are X2, P6 is X3, P7 is X4, P8 is X5, P9 is X6 and P10 is X7. P11 is deferred.

| id | change | class |
|---|---|---|
| P0a | Add the rigid column for ash, char and tin. Proposed values are 0.0, 1.0 and 0.9 (Q4). | P (today it crashes) |
| P0c | `Matter._walls` (`matter.py:190-197`) rebuilds when the wall content **inside the particles' box** changes, not on a whole-world count. Today a moved cell can be missed, and a far-away fire triggers a 1.7 s rebuild. | P |
| P0d | Inventory: which tests and scenes promote. There is no repo change. | — |
| P1 | Active-region grid in `MPM.substep` (`core.py:641-860`), plus an early return when there are no particles. | E |
| P2 | Rewrite the einsum at `core.py:688` once as an explicit sum (drift 1.55e-15 m over 400 substeps). Then numba P2G/G2P in `src/mpm/kernels.py`, with the import shim copied from `sim.py:48-57`. | E after the one redefinition |
| P3 | numba distance transform (φ bit-exact). Windowed φ and ∇φ, rebuilt every substep with margin 4. Drop the dense `phi`/`pgrad`/`phi_node`/`pgrad_node` fields (`core.py:198-201`, 64 B per cell). **Keep** `wall` and the `add_box` flaw field. Count `deep_hits` for particles more than 4 cells inside a wall. | E |
| P4 | `vol0 = mass / density` in `promote` (`matter.py:110, 118-122`). The tree drops from 215 to 86 substeps per tick. The parting stress of 40%-full spokes rises 1.58×. | P, owner review |
| P5a | A registry of per-particle fields; `mpm.eid` and `mpm.batch`; `promote` reads `w.eid` when it exists; sparse rasterize (33.8 → 1.7 ms at 400k cells, bit-identical) with the eid rule of 3.0 item 6; `matter.entities()` (cells, centroid, speed, aloft, material mix, carried fluid, shape extents). Delete the dead owner path (`matter.py:50-51, 95-99, 180, 346-355`). Keep the `aloft_owners()` signature, so `support.py` does not change. | E |
| P5b | Object-local flaws in `promote`: seeded by eid and indexed from the cluster's corner. | P |
| P6 | **Write-back.** Between `rasterize` and `unrasterize`, the lattice may change *what* a drawn cell holds, never *where* it is. `unrasterize` first applies each drawn cell's ΔE (by mass share), its `smass` ratio, its fluid deltas, a CHAR/ASH retype, removal (when the cell became AIR or foreign) and rubble→`dmg`. Ledgers include the pending deltas. With no edits it is exact. | P |
| P7 | `matter.events()`: crack, part, struck (edge-triggered, with hysteresis) and touch (on onset). Two flags, both off by default: `seam_sounds` (to `_noise`) and `seam_blows` (to `_spend`). | E (the tap), P (the flags) |
| P8 | Sleep without demotion: an entity at rest moves to a dormant store. It is still drawn and still written back, but costs no solver time. It wakes on overlap, a change in its box, write-back or an impulse. | P, **Q4** |
| P9 | Islands: P2G/G2P per group of overlapping boxes inside one substep. | E with a global dt |
| P10 | The whole substep in one numba kernel. The model is 0.82 µs per particle-substep, so 4–5 ms per tick for the plank. It waits on roadmap decision 2 (bonds against the continuum). LAPACK `svd`/`eigvalsh`/`solve` (`core.py:573, 626, 514`) is first redefined as a fixed-sweep numpy algorithm. | redefinition, then E |
| P11 | Demotion. Deferred by roadmap 93, decision 3. The prototype (one cell per particle, max fill 1.00, ledger neutral) is kept as evidence. | — |

**Budget for characters** (5 cm voxels, 25 ms ticks):

| | 56-particle plank, ms per tick | falling tree, ms per tick |
|---|---|---|
| today | 451 | about 8,900 |
| after P1–P4 | about 90 | about 1,300 |
| after P10 | 4–5 | 260–330 |
| asleep (P8) | about 0 | about 0 |

The non-solver part of a 5 cm tick with one person is about 15 ms. Real time therefore allows about 10 ms of solver, or about 100 awake particles.

**GPU:** none. The chooser holds about 2 GB of the 6 GB card. The plan is CPU numba with serial kernels, and parallelism across harvest processes.

**Risks:**
- P6 changes every heated seam scene, so renders must be judged again. Combustion heat is not in `total_energy` (`sim.py:4425` against `8779-8787`), so energy tests must avoid fire.
- The lattice's own energy ledger drifts −2,889 J in a small torch scene with no seam. That is unexplained, and it limits how tight seam energy tests can be.
- The windowed SDF is inexact only for particles more than 4 cells deep inside a wall. They are counted in `deep_hits`.
- Sleep wake rules may miss support that changes outside the box. A periodic recheck is the fallback.
- P10's exactness depends on numba not fusing multiply-adds. The plain path stays the definition, and a guard compares the two.
- P4 and P5b change fracture (tree branches), so they need owner review.
- The `seam_*` flags change what people hear and suffer, so harvests must be re-baselined.
- Until people become colliders or demotion-on-act exists (Q4), a person cannot pick up a fallen particle object. The act fails with "it is still moving".

### 3.3 The entity layer: identity, names and properties

**Decisions:**
- **One join rule, one key.** Two touching solid cells belong to one thing exactly when their *key* is equal:

  ```python
  key = np.where(self._thing_lut[self.eid], 256 + self.eid.astype(np.int64), self.mat)
  ```

  - Person eids and eid 0 map to `mat`.
  - With nothing named, the key *is* `mat`, so all behaviour is unchanged. `design/label_golden.py` reproduced the three legacy labellers bit for bit on 60 worlds.
  - A named axe (iron head, wood haft) is one thing. An unnamed bar touching an unnamed iron post still merges with it, as today, until it is named or taken (E4, adopt-on-take).
- **Storage.**
  - `w.eid` is an int32 field aligned with `mat`: 4 B per cell, 1.6 MB for a 400k-cell house.
  - `w.things` is the registry (plain dicts, so snapshots work). Ids are never reused.
  - `_thing_lut` maps an eid to "is a non-person thing".
  - `_thing_gen` is bumped whenever a thing's cell moves.
- **Fix six defects first (E0).** Identity would inherit them:

  | defect | where |
  |---|---|
  | `_repose` and `_crouch` do not carry `edge`. A held tool goes blunt during a reach, and flesh that moves in inherits the edge. | `sim.py:2782-2783`, `6549-6550` |
  | The topple wedge-restore drops `edge`. | cleared at `1837`, never restored at `1866-1871` |
  | Melt and acid leave `edge` and `rubble` in emptied cells. | `4519-4520`, `4581-4582` |
  | Shatter targets land on a stale edge. | `1762-1773` |
  | Hands can take and shove drawn particle matter, and that duplicates it. | `6024-6026`, `8359-8380` |
  | `b in self.bodies` can raise with several bodies in flight. | `6229-6230`, `6251-6252` |

  Every-tick hygiene clears `edge`, `rubble` and `eid` in cells that became AIR. It costs 29 µs on a quiet tick at 400k cells. E0 is a behaviour change: swings after a reach become sharp again.
- **One labeller.** `src/voxel/label.py` is a numba union-find, `label(mask, key, canon=max|min, wrap)`. Its root is the canonical member, so the label *arrays* equal the legacy ones.

  | caller | mask | key | canon | wrap |
  |---|---|---|---|---|
  | torque (`sim.py:4057-4069`) | `mat != AIR` | `mat` | max | no |
  | reach (`6054-6083`) | `solid & ~flesh & ~moving` | `mat` | min | **yes** |
  | identity pass | (see `_law_identity`) | none | min | no |

  - Speed: whole-house torque labelling goes from 1,895 ms to 2.1 ms. The reach box goes from 5.62 ms to 0.25 ms.
  - The legacy numpy routines stay as the fallback when `not w.fused`.
  - A numba `_object_at` returns the same cells in the same order, at 0.066 µs per cell against 1.12 µs.
  - Free-fall labelling (`sim.py:1521-1532`) lives inside the flag-gated `_free_fall_to_seam`, so it is left alone until Q4 decides that method's fate.
- **Carry `eid` at every site that moves matter.** That is 14 lattice sites: settle, shatter, topple, launch (plus the swing at `2310-2326`), turn, repose, crouch, release, loose, land, `_left_world`, leave, shove and walk. The particle sites belong to MPM P5a. A ledger test per site, with a mutation run, proves each one.
- **`_law_identity`** runs before `_law_bleed`:
  - it clears empty cells;
  - it splits dirty things into pieces, where lattice components plus solver `cid` groups each count as a piece;
  - the heaviest piece keeps the eid;
  - children are named from declared parts ("the haft of the axe") or as "a piece of the plank";
  - there is one debris child per parent;
  - **the grip follows the piece in the hand**;
  - a census records deaths with their cause (`_left_world` and `_leave` record losses, so a death is never missed);
  - every change is a typed event.
- **Things in the solver are named but cannot be taken.** There is no demotion (roadmap 93). `id_audit` counts how often this limit bites.
- **Properties** come from `w.props(eid)`, computed on the thing's cells and cached per `(tick, _thing_gen)`: `mass_g`, `integrity`, `sharp` (the smallest positive `edge`), `hot_c` (per cell, the `heat_capacity` formula), `burning` (per cell), `charred`, `ashes`, `wet_ml`, `contains`, `can_hold_ml` (from `_cavity`, factored out of `matter.py:141-170`), `broken`, `in_solver`, `held_by` and `liftable_by(p)`.
- **Names.** `w.name(eid, fallback)` returns the fallback unchanged for eid 0, so unnamed text stays byte-identical.
  - State words come in priority order: "the burning axe", "the ashes of the chair", "the charred chair", "the broken axe".
  - Lookalikes get words relative to the body ("on your left", "at your feet"), then ordinals. Metres are never used: `_places` prints "(0 m)" within reach (`sim.py:5802`).
  - Names reach all 11 text sites: `7186, 7237, 7244, 7262, 7282, 7408, 7484, 7502, 7544, 7093, 6260`.
- **Holding by eid.** For a named thing, the grip is valid while `eid[cell] == held eid`. The pointer cell and `mat` come from the same voxel. The swing landing (`sim.py:3752-3754`) and the catch (`6253-6262`) use the eid only for named things, and keep today's code otherwise.
- **Authoring.**

  ```python
  with thing(w, "axe", kind="tool", parts={"head": IRON, "haft": WOOD},
             says={"affords": ["cut", "chop"]}):
      ...
  ```

  - `thing()` stamps every cell whose `mat` or `smass` changed. That is the same diff `part()` uses (`scenes.py:457-459`), so direct writes such as `_grow_tree`'s canopy are caught.
  - `mass0` and the affords audit are taken when `thing()` exits.
  - A mismatch (a "cut" hint on a blunt thing) goes to `w.id_audit` and never becomes a capability.
  - `_person` stamps the body.
- **Bodies.** A body eid could replace `_claim_bodies` (`body = (eid == p.eid) & (mat == FLESH)`), but it does **not** fix bug 87, which comes from bodies interleaving on landing (`docs/voxel-roadmap.md:5409-5550`). So:
  - the fuzz harness is rebuilt first (EF);
  - an audit reports claim/eid disagreements by cause (E6);
  - the replacement (E7) waits for "a landing must not put two bodies in the same space" (`docs/voxel-roadmap.md:5544-5547`).

**Risks:**
- E0 moves outcomes, so each failing test must be read, never re-tuned.
- An eid index rebuild costs 722 µs at 400k cells. Several carriers in a big world add milliseconds. The fallback is a per-thing bounding-box search.
- Names change with state ("the axe" becomes "the broken axe"). Beliefs key on eid, but a model may read the two names as two things unless the prose shows continuity.
- Split rules across the seam are new logic with rare triggers.
- The reach wrap-around hides a same-material thing on the far face of the reach box until E4 removes the wrap (a menu change).
- Seam v2 starts a new dataset. Harvest tooling must refuse to mix spellings.

### 3.4 Typed events, perception, beliefs and the cause log

**Decisions:**
- **The record.** `Ev` is a frozen, slotted dataclass in `src/voxel/events.py`. It has no numpy.

  ```python
  Ev(id, tick, verb, actor, target, instrument, phase, result, why,
     place, origin, db, visible, dur, cause, link, text, data)
  ```

  | field | contents |
  |---|---|
  | `verb` | a key of a closed `VERBS` table (data) |
  | `actor` | a pid, or None for the world |
  | `target` | `("person", pid)`, `("thing", eid)`, `("fire", ev_id)` or `("mat", m, cell)`; ints only |
  | `phase` | start, done or stopped |
  | `origin` | the float point a sound starts from; it equals `w.sounds[i]["at"]` |
  | `link` | call, carried or near |
  | `text` | the legacy line; it is **never** shown to a model |

  - `__deepcopy__` returns self: 4,096 records copy in 0.84 ms.
  - `w.events` is a ring of 4,096 records.
  - `w.event_sink` and `w._epoch` are not world state. `restore` bumps the epoch, so ids never repeat within an epoch.
- **The API.**

  ```python
  w.emit(verb, …, text=, to=)
  ```

  Its only legacy side effect is the old `p["events"]` append, made at the same position. Cost: 1.0 µs. `with w.because(ev_id, link):` sets the current cause and restores it in `try/finally`.
- **The 35 legacy sites are converted byte for byte.** Gate: `grep -c '\["events"\].append' src/voxel/sim.py` (or its modules) = 0. New events have no text:

  | events | where |
  |---|---|
  | `decides` | `7456`, `8010` |
  | `steps` | `7939` |
  | `lands` | `1652`, `3670` |
  | `hits` | `_spend`, `3529` |
  | `wounds` | `_blow`, `3926-3927` |
  | `breaks` | `_shatter`, collected after the early return at `1748-1749` and flushed sorted at tick end |
  | `topples` | `1806`, `1571` |
  | `looses` | `3342` |
  | `melts`, `sets`, `dissolves` | aggregated per material |
  | `ignites`, `catches_fire`, `burns_out` | fire bookkeeping |
- **Attribution rules:**
  - The actor of `hits` is the owner's pid only for **limb** bodies. For thrown fliers it is `b["by_pid"]`. For whole-body falls it is None, with `instrument = ("person", faller)` and `cause` = the `falls` event.
  - Pids are stamped where the person dict is in hand: `2325`, `3336`, `5913`, `5919`, `8533`. There is no lookup by name, because default names repeat ("the person", `sim.py:4963`).
- **Fire comes from one predicate.**
  - `lit = burning()` is computed once in `_law_perceive`, under the exact gate of `sim.py:7649-7650` (anyone alive and awake), and handed to `_law_will` as `w._burning_now`. A broader gate would shift `_matter_gen` and break golden.
  - Fire boxes have hysteresis: `burns_out` follows 10 unlit ticks.
  - Measured: identical events with `skip_quiet` True and False.
- **Hearing stays as it is.** `_law_hear`'s body is not changed. One line next to the `_level` call (`sim.py:5563`) writes a `heard` record for every audible pair.
  - The batched version was measured exact (0 of 1,036 decisions differ) but **not faster** (157 against 151 µs), so it is not adopted.
  - Sight records use one batched `rays()` call (100% agreement with `_sees` on 64 rays).
  - The "looked" pass runs every 4 ticks (the cadence of `sim.py:7691`).
  - All of this is write-only: today's percept chain (`7697-7764`) is unchanged until a flagged later step.
- **Who perceives:**

  | person | records | `p["noise"]` and waking |
  |---|---|---|
  | dead or safe | none | none (`5534-5535`) |
  | asleep | heard only | as today |
  | unconscious | none | as today |
- **No identity leaks.** A `heard` record has `actor_seen = None`. The exception is a `says` event that was understood (at least `HEAR_DB` = 40 dB, `sim.py:851`) by a listener who already knows that voice. A voice is learned when one `says` is both seen and heard. `("at", ·)` beliefs come only from `saw` or `looked`.
- **Percept hooks** at `7707` (sees_fire), `7719`/`7721` (falling, thrown), `7746` (hears_alarm), `7755` (sees_runner) and `7764` (hears_noise) give every percept a source event id. That id is what the attention law uses as "news".
- **Beliefs.** `Mind.bel[key] = (value, ev_id, how, tick)`.
  - Keys: `("at", pid)`, `("holds", pid)`, `("state", pid)`, `("hurt_by", victim)`, `("alight", fire)`, `("said", pid)` and `("voice", pid)`. From V4 on, `("at", ("thing", eid))` is added.
  - At most 64 keys; the oldest goes first, and `voice` is never evicted.
  - A `told` belief never overrides a fresher `saw` (the lab rule, `lab/world2/world.py:616-621`).
  - `notices_missing` waits for the entity layer, because today's `known` map is a 2D sweep at eye height (`sim.py:5346, 5369-5370`) and would report people lying behind a bench as missing.
- **Misreads and claims** are data hooks, off by default.
  - A misread draw is keyed on content (splitmix64 of seed, pid, tick, verb, actor, place), so the sim keeps no RNG. The log keeps the truth.
  - A claim (`says fire`) carries `true` and `grounded`. `grounded` comes from the speaker's own fire records. A liar is just a character sheet plus a policy.
- **The cause log** has three link kinds:
  - **call:** synchronous callees inherit the cause;
  - **carried:** an id stored on a body, a particle batch, or a Mind's act and harm slots;
  - **near:** a cause found by place from `w._disturbed`, shown as "probably because".

  `w.why(ev_id)` walks the chain, for example `stops_breathing ← wounds ← hits ← swings ← decides`. Trace rows gain one plain-JSON key, `row["mind"]` (scenes `json.dump` every row: `scenes.py:651-657, 762-766`).
- **Readers.** `perceive.render(rec, viewer)` writes text from the viewer's side ("someone", "a crash to the north-east") and never uses `Ev.text`. `beliefs.view(w, p)` returns plain dicts shaped like the mind interface's `Percept` and `Belief`.

**Risks:**
- Fire is not observed while everyone sleeps, so `ignites` can be logged late.
- "near" causes are heuristics.
- Sight records use start-of-tick geometry.
- Particle matter makes no sound and touches no one until the P7 flags are on (`matter.py:327-331`).
- The 4,096-record ring bounds how far back a job's step test can look (an estimate: about 400 ticks with 8 people). Step tests also read the Mind's own `did` records.
- `row["mind"]` grows `traces.json`.

### 3.5 The mind interface and its two backends

**Decisions:**
- **`src/mind` is task-level and uses only the standard library.**
  - Types (`src/mind/api.py`): `Ref`, `Percept`, `Belief`, `SelfState`, `Option(verb, roles, text, group, duration, interruptible, features, ok, missing, key)`, `Menu(options, blocked, full, cut, prior, prior_avg, habit, groups)`, `Job`, `JobEvent`, `Why`, `DecisionRequest`, `Decision`, and the protocols `Chooser` and `Backend`.
  - Frozen types hold no dict fields.
  - A `Job` is plain data, so snapshots can copy and pickle it.
- **The verb catalog** is `src/mind/data/verbs.yaml`, drafted in `design/mind_iface/verbs_v2.yaml`.
  - Fields: `id`, `group`, `null_act` (YAML reads a key named `null` as None), `roles` with property tests, `needs` (capabilities), `serves` (drives), `hint`, the `told` templates, `duration`, `commit_after`, `interrupt_on`, `flags` and per-backend programs.
  - `check_catalog_v2.py` runs every structural check. The v2 draft passes.
- **Menus.**
  - The pre-rank keeps options by score until 95% of the rules' probability mass or 12 options. It then adds the null act, one option for each verb not yet kept, and fills up to 16. Measured mass cut: mean 0.64%, p99 4.3%.
  - The score is `max(prior, prior_avg) + habit`, so habit is counted once.
  - **Canonical order** is group, then verb, then text, never score. With canonical order the rules' favourite is first in 8% of menus, against 100% when sorted by score.
  - The cut is recorded.
  - Recall is audited **offline with the model**, never with the rules' own scores (M9).
- **Choosers:**

  | chooser | rule |
  |---|---|
  | Rules | softmax((prior + habit)/T), T = 0.2. Equals the lab's `Rules` (`lab/world2/policy.py:144-150`). |
  | Combined | softmax(z/τ + β·(prior − prior_avg)/T + habit/T), behind the plausibility guard. Equals the lab's bias mode (`policy.py:233-243`). Only z comes from the model. |
  | Gate(g) | The rules pick when their top probability is at least g. At g = 0.7, 35% of decisions skip the model. |
  | Random, Replay, Player | Player gets the full grouped menu, may chain options (`Job.then`), and never gets the model. |

  `MindConfig.mode` is one of:
  - `reactive` (the default): Rules for every NPC, 0 remote calls;
  - `featured`: the model for named characters and the one the player addressed;
  - `all`: offline evaluation only.
- **Determinism per chooser:**

  | chooser | guarantee |
  |---|---|
  | Rules, Random, Gate→Rules | Replay exactly. |
  | Remote, deterministic mode | Blocks with a 60 s watchdog that aborts the run, with no fallback. One request per forward pass. Replayable only from its decision log. |
  | Remote, realtime mode | Falls back to the rules' argmax, logged as a fallback. |
- **The voxel link** (`src/voxel/mindlink.py`). Control stays inverted, and `MindPolicy` adds optional hooks read with `getattr`:
  - `prepare(w, p, percept)` does all state changes: step tests (typed events), interrupts on news, starting a pending decision, making a request, and compiling the step into per-limb matchers and `p["intent"]`;
  - `limbs(…)` returns only the limbs the step needs, so a woken decide never stops a walk: the legs' null act clears the goal (`sim.py:7570-7571`);
  - `pick` is pure. It matches rows **by payload** (goal xy, object cell or eid, who), never by label, because place labels change as a body moves (`5748-5749, 5789-5802`). Falling decides go to the reflex table. The second pick under `recall_check` (`7431-7440`) therefore changes nothing;
  - `on_restore` cancels pending requests;
  - a step that gets no matching event within 3 decide periods fails the job;
  - in the `_duel` prototype, the task-level mind asked **8 questions in 150 ticks**. The per-limb policies faced 71–111 limb menus.
- **The attention law** (`World.attention`, a world-level setting for every body).
  - `"latch"` is today's behaviour and the default. Under it, golden is SAME.
  - `"onset"` runs the senses every tick for every awake body. A body decides again only on **news** (a source event id not seen for `attend_gap` = 40 ticks), when its goal ends, or at `intent.wake_at`. A percept that persists never triggers a decide. `emergency` clears when the goal ends.
  - The mind link requires `"onset"`. Simply skipping the latch was measured to decide 120 times in 120 ticks and never reach the door.

  Measured in the smoky room, 120 ticks:

  | | decides | gets out |
  |---|---|---|
  | latch, no exit known | 1 | no |
  | onset, no exit known | 2 | **yes** (it re-decides when its goal ends) |
  | onset, exit known | 1 | yes |

  Making onset the default changes scene traces (swing 84→48, ledge 44→87, rescue 40→97), so it is question Q2.
- **Motive in voxel.** Voxel bodies have no traits today. A declared character sheet `p["sheet"] = {"harm": {name: w}, "traits": {…}}` drives `rule_scores`:

  | option | score |
  |---|---|
  | strike X | `harm[X]` × (1.5 if armed, else 0.5) |
  | take | 0.8 × the drive the thing serves |
  | flee | safety × (1 − route danger) |
  | shout | safety, when others can hear |
  | walk | 0.1 |
  | wait | 0.2 |

  Traits bias strike against flee through `prior − prior_avg` (Q5).
- **The rooms backend** (`src/rooms`) is a fork of `lab/world2/world.py`. Its header records the lab file's sha1.
  - `step()` takes no policy, and `_choose` is split into `needs_decision` and `start`.
  - Event rows keep the raw lab verbs, so `lab/judge/signals.py` (HELP/HARM at `63-66`) still counts them. `why` goes to a separate decisions JSONL.
  - Compat mode must match lab goldens byte for byte. Seeds 0–2 give 438, 425 and 449 events.
  - Start events (for loud or violent verbs) go to a separate stream. They never call `_learn` and never draw from the RNG.
  - Measured: inn menus have a median of 27 options and a maximum of 73. A needs sentence ("hungry, no food held") would fire in 49% of decisions.
- **The model crossing.** The default is **record, score, replay**:
  - `src/mind/record.py` writes lab case-shaped JSONL (situation, question, options, full_options, cut);
  - `lab/methods/score_requests.py` scores it offline, one process at a time;
  - `Replay` applies the decisions by request id.

  A live socket (`lab/serve.py` + `src/mind/remote.py`) is opt-in (Q3). It is used only after the server's memory is measured, and the run refuses to start if server plus sim memory exceeds free RAM minus 1 GB.

**Risks:**
- Canonical order has its own position bias (the "move" group comes first). M9 measures it with permutations.
- A needs sentence could act as a lure, and the 62→75% hint gain was measured on an item-choice framing. M9 runs an A/B test of describe.
- The fork drifts from the lab inn.
- GPU float order makes model runs replayable only from logs.
- Character-sheet drives keep voxel personalities scripted.
- Bug 87 can hit any two-person scene with falls. M8 has no falls, and every mind-link test asserts there are 2 persons on every tick.
- The strike stand-off depends on stance: `_square_up` refuses a body that holds something (`sim.py:2516-2520`). M10 calibrates it.

### 3.6 Verification tools and test tiers

- **`tools/golden.py`** (from `design/golden.py`): 27 quick cases (83 s) and 7 slow ones (105 s).
  - It hashes every World field except `policy`, plus "taps" (`_full_menu`, ledgers, the render path).
  - S1 adds:
    - **leaving out names that are absent from the base recording**: new World attributes and new dict keys are listed as `NEW: …` and are not compared;
    - `--except CASES`;
    - `MARK1_REPO` sandboxes;
    - the decision case `duel150`, the `tests/test_wounds._duel` geometry (`tests/test_wounds.py:154-180`) run for 150 ticks.
  - Coverage: the quick set misses only the 7 dead methods and the flag-gated `_free_fall_to_seam`.
- **`tools/stepcheck.sh`** (new in S1) gives each step its own sandbox (section 4.1).
- **Test tiers** (S6).
  - `tests/conftest.py` marks tests slow from a measured `SLOW` set. The C2 gate's `--durations=0` gives 1,212 s over 157 timed tests, and marking 51 of them leaves 88 s.
  - A misspelled `SLOW` entry raises `pytest.UsageError`.
  - There is no root `pytest.ini`, because it would capture the lab's tests.
  - **Never combine `-m` with node ids:** pytest then deselects them and exits 5. Every gate treats exit 5 as a failure.
- **The harvest guard.** `harvest.to_file(…, range(32), workers=1)` must give `broke == 0`, because `_one` turns exceptions into rows (`harvest.py:209-214`).
- **Memory.** Run one heavy process at a time under the shared lock. A golden process peaks at 350–470 MB. The 5 voxel test files peak at 1.04 GB. Measure memory before using `-n 2`.

---

## 4. Build plan

### 4.1 How several agents work at once

- **Ownership rule.** In a wave, every file is owned by exactly one step. A step edits only the files it owns. New tests go into new files owned by that step. Only S2 and S3 edit `src/voxel/sim.py`. Only the integrator edits `tools/golden.py`, `tests/conftest.py` and `docs/sim-split-map.md` after S1.
- **Sandbox rule.** Other agents' half-done edits must never affect a check. `tools/stepcheck.sh <id> "<declared cases>" <owned paths…>` does three things:
  1. once, at the step's start, it unpacks `git archive HEAD` (read-only) into `$WORK/<id>/head` and records a golden base there;
  2. at each check, it builds `$WORK/<id>/mine` from `head` plus the step's owned files copied from the live tree;
  3. it runs `golden.py check --all` in `mine` and passes when every undeclared case is SAME.

  Tests for a step also run inside `mine`.
- **Golden rule.** An E step declares "none". A P step declares the cases it may change, and explains every DIFFERS line in its report (the field and the first tick).
- **Wave end** (the integrator):
  1. run `stepcheck.sh wave<N>` over the whole live tree: only the declared cases may differ;
  2. run the fast tier, plus the full 5 voxel files if the wave had a P step;
  3. run the harvest guard;
  4. re-record the base;
  5. commit (Q1). HEAD is then the next wave's start.
- **The lock.** Every python run that builds a world or runs tests goes through `flock "$LOCK"`. Waiting is normal: verification is serial even when editing is parallel.
- **Mind lane.** `src/mind`, `src/rooms` and their tests touch no voxel file. They may run in parallel with every wave after S0, phase 0 included, because S1 narrows `carve.sh`'s clean-tree check to the paths the carve reads. A mind-lane step's files go into a wave-end commit only when that step's own tests pass.

### 4.2 Common commands

```bash
R="/home/nisargparikh/Desktop/Fun Stuff/Mark_1"; cd "$R"
# LOCK = the shared lock file; WORK = a scratch directory (both set by the orchestrator)
VOX="tests/test_voxel.py tests/test_wounds.py tests/test_sound.py tests/test_projectiles.py tests/test_mpm.py tests/test_module_hygiene.py"
T()       { flock "$LOCK" python3 -m pytest -q -p no:cacheprovider "$@"; }   # tests (node ids: never with -m)
G()       { flock "$LOCK" python3 tools/golden.py "$@"; }                    # record | check | coverage | list
FAST()    { T $VOX -m "not slow"; }                                          # after S6; about 90 s
HARVEST() { flock "$LOCK" python3 -c "from src.voxel import harvest as h; g = h.to_file('$WORK/h32.jsonl', range(32), workers=1); assert g['broke'] == 0, g"; }
tools/stepcheck.sh <id> "<declared cases | none>" <owned paths…>              # sandboxed golden check
```

Below, `T`, `G`, `FAST` and `HARVEST` mean these shell functions.

### 4.3 Waves at a glance

The steps of one wave own disjoint files. "M lane" steps (M1–M6) run alongside, in order, starting right after S0.

| wave | steps (owned files, short form) |
|---|---|
| **phase 0** (serial) | S0 user commit → S1 tools and guards → S2 the carve → S3 hook points |
| **1** | S4 cleanup (will, pose, kernels, materials, policy, fields, skeleton, persons, hygiene) · S5 ghost (ledgers) · S6 tiers (conftest) · X1 MPM correctness and speed (src/mpm) · EF body fuzz (tools/fuzz_bodies.py) · EB vessel pins (tests/test_vessels.py) · Q0 profile (no files) |
| **2** | E0 entity defects (pose, support, heat, fluids, hands, motion, things) · V0 event core (events, ledgers, persons) · X2 particle identity, fill, flaws (src/mpm) |
| **3** | E1 labeller and inert eid (label, support, bodies, pose, motion, hands, fields, things) · V1 events in will, senses, physiology · X3 write-back (src/mpm) |
| **4** | E2 registry, identity pass and names in menus (things, will) · V2 events in bodies, hands, motion, support, heat, fluids, pose, physiology · X4 matter events and `_cavity` (src/mpm) |
| **5** | E3 holding by eid and authoring (hands, bodies, persons, scenes, things) · V3 perception (perceive, senses, will, physiology, events) · X5 sleep, if approved (src/mpm) |
| **6** | M7 attention law and policy hooks (will, policy, ledgers) · V4 beliefs and sightings of things (beliefs, perceive, events, harvest) · E4 props, reach, adopt, torque (things, hands, support) |
| **7** | V5 cause log and readers (will, beliefs, perceive, events) · E5 named scenes and seam v2 (scenes, seam) · X6 islands (src/mpm) |
| **8** | M8 voxel mind link, M2a (mindlink, harvest) · E6 body audit (persons, things, tools/fuzz_bodies.py) |
| **9** | M9 offline model gate (user, lab venv) · M10 approach, M2b (mindlink, motion) · X7 fused substep, if decided (src/mpm) |
| **10** | M11 live model, if approved (lab/serve.py, src/mind/remote.py) · E7 body eid, gated on bug 87 (persons) · E8 one `CARRIED` tuple (support, bodies, pose, motion, hands) |

**Critical path to a model choosing in voxel:** S0–S3 → V0 → V1 → V2 → V3 → M7 → V5 → M8 (Rules) → E5 (names) → M9 → M11 (live) or Replay (offline). The mind lane (M1–M6) is finished long before M8 needs it.

### 4.4 The steps

Each step lists **what**, **owns** (the only files it may edit), **verify**, **depends** and its **wave**. "Golden: E" means `tools/stepcheck.sh <id> none <owned>` must pass. Paths under `src/voxel/` are written without the folder.

---

**S0. Commit today's tree (the user).**
- **What:** Commit `src/` and `tests/` as they are. That covers `sim.py` (1,133 changed lines), the untracked `src/mpm/` and `tests/test_{mpm,projectiles,wounds}.py`. Freeze `src/voxel` and `src/mpm` until S2's gate passes (about 30 minutes).
- **Owns:** nothing.
- **Verify:** `git status --porcelain -- src tests` prints nothing. `git show HEAD:src/voxel/sim.py | sha256sum` starts with `4d0b87e91545`.
- **Depends:** none. **Wave:** phase 0.

**S1. Tools and guards before the move.**
- **What:**
  - Add `tests/test_module_hygiene.py` (rules 1–3, with allow-lists for `_crouch`/`joints` and `_full_menu`).
  - In `test_the_ATTENTION_CAP_never_hides_the_answer` (`tests/test_voxel.py:2560`), add a spy: the longest uncapped menu must be longer than `MENU_CAP`.
  - In `test_a_HARVEST_BUYS_SITUATIONS_not_rows` (`4855`), replace the vacuous check at `4881-4882` with `r["spelling"] == seam.SPELLING`.
  - Commit `tools/golden.py` with the S1 additions of 3.6, `tools/stepcheck.sh`, and `tools/split/{deps,splitmap,split,astsame,namecheck,reexport,wheremap,slowlist}.py` plus `carve.sh` and `gate.sh`. `LOCK` and `MARK1_REPO` come from the environment.
  - Narrow `carve.sh`'s clean check to `src/voxel src/mpm` and the 5 voxel test files.
  - Copy the prototypes later steps cite into `tools/protos/`: `ghost_test.py`, `label_golden.py`, `eid_proto2.py`, `index_cost.py`, `review/*.py`, `mpm/exact_check.py`, `mpm2/*.py`, `events/*.py`, `onset/{patch_onset,patch_hooks,probe_onset,mind_duel,duel_baseline}.py`, `onset/sim_mind.diff`, `proto/*.py` and `mind_iface/{verbs_v2.yaml,check_catalog_v2.py}`.
- **Owns:** `tests/test_module_hygiene.py`, `tests/test_voxel.py`, `tools/golden.py`, `tools/stepcheck.sh`, `tools/split/*`, `tools/protos/*`.
- **Verify:**
  - `T tests/test_module_hygiene.py` gives 3 passed.
  - `T "tests/test_voxel.py::test_the_ATTENTION_CAP_never_hides_the_answer" "tests/test_voxel.py::test_a_HARVEST_BUYS_SITUATIONS_not_rows"` passes (about 30 s).
  - `G record $WORK/base0.json --all`, then `G check $WORK/base0.json --only pond,rewind,duel150` prints ALL SAME.
  - `src/` is untouched.
- **Depends:** S0. **Wave:** phase 0.

**S2. The carve (exclusive: no other voxel or MPM work runs).**
- **What:** Run `tools/split/carve.sh "$R" "$WORK/carve"`, then `tools/split/gate.sh "$R" "$WORK/carve"`. This writes:
  - 18 new modules plus the shrunk `sim.py`;
  - the one test edit (by content);
  - `docs/sim-split-map.md`;
  - and it removes the orphan numba caches.

  To bisect a failure, use `split.py <pristine> <tmp> --only fields,ledgers`.
- **Owns:** `src/voxel/{sim,kernels,materials,policy,fields,fluids,heat,air,support,bodies,skeleton,pose,persons,physiology,senses,motion,hands,will,ledgers}.py`, `tests/test_voxel.py`, `docs/sim-split-map.md`.
- **Verify:** `gate.sh` exits 0:
  - import OK;
  - astsame IDENTICAL (191/4/80);
  - namecheck clean;
  - re-export 102→117 names, with 0 missing and 0 changed;
  - golden `--all` ALL SAME;
  - the 6 voxel test files pass (keep `durations.txt` for S6);
  - harvest `broke == 0`.

  Then the user commits.
- **Depends:** S1. **Wave:** phase 0.

**S3. Hook points in `sim.py` (the last edit of `sim.py` in this plan).**
- **What:**
  - Add three stub mixins to `World`'s bases: `ThingsMixin` (`things.py`), `EventsMixin` (`events.py`) and `PerceiveMixin` (`perceive.py`).
  - In `__init__`, add `self._init_things()`, `self._init_events()` and `self.attention = "latch"`.
  - In `step`:
    - `self._tick_start()` first;
    - `self._after_raster()` after `matter.rasterize`;
    - `self._law_identity()` before `_law_bleed`;
    - `self._law_perceive()` in place of `self._law_hear()` (the stub calls `_law_hear`);
    - `self._tick_end()` before `self.tick += 1`.
  - Every stub is a no-op.
- **Owns:** `sim.py`, `things.py`, `events.py`, `perceive.py`.
- **Verify:** Golden E (`--all`). `T tests/test_module_hygiene.py "tests/test_voxel.py::test_ACTIVE_REGIONS_change_NOTHING" "tests/test_voxel.py::test_the_FAST_PATH_and_the_PLAIN_ONE_agree"`. Re-export check unchanged.
- **Depends:** S2. **Wave:** phase 0.

---

**S4. Cleanup after the carve.**
- **What:**
  - **C3:** `_menu(…, cap=None)`, with `cap = MENU_CAP if cap is None else cap`; `_full_menu` returns `self._menu(…, cap=10**6)`; empty the `globals()` allow-list.
  - **C5:** in `_crouch`, `joints = p.get("joints")`, plus a test that forces `base != origin`.
  - **C6:** fix stale positional comments (`sim.py:44, 241, 333, 760-763, 2421` before the split).
  - **C7:** delete the 7 dead methods. Delete or keep `_free_fall_to_seam` per Q4. If deleted, `support.py` joins this step's files.
- **Owns:** `will.py`, `pose.py`, `kernels.py`, `materials.py`, `policy.py`, `fields.py`, `skeleton.py`, `persons.py` (and `support.py` only if Q4 says delete), `tests/test_module_hygiene.py`, `tools/split/namecheck.py`, `tests/test_pose.py` (new).
- **Verify:**
  - Golden E (`recall_hall` includes the `tap:_full_menu`).
  - `T tests/test_module_hygiene.py tests/test_pose.py "tests/test_voxel.py::test_the_ATTENTION_CAP_never_hides_the_answer"`. The new `_crouch` test raises NameError before the fix.
  - `grep -rn "solid()\|_chain_xform\|_flood_person" src tests tools` finds nothing.
  - `golden.py coverage` lists nothing, or only `_free_fall_to_seam`.
- **Depends:** S3. **Wave:** 1.

**S5. The snapshot ghost fix (a hard prerequisite for any model in `w.policy`).**
- **What:** Add a module-level sentinel `_THIS` in `ledgers.py`. `snapshot` uses `deepcopy(state, {id(self): _THIS})`, and `restore` uses `deepcopy(snap, {id(_THIS): self})`. Add a test: with a policy whose `__deepcopy__` raises, snapshot and restore work, `w.matter.w is w`, and restoring into a second World gives `w2.matter.w is w2`.
- **Owns:** `ledgers.py`, `tests/test_snapshot.py` (new).
- **Verify:**
  - `tools/stepcheck.sh S5 rewind ledgers.py`: only `rewind` differs, at `matter`, on its first post-restore row.
  - `T tests/test_snapshot.py` fails on HEAD and passes after the fix.
  - The tests that restore, by node id with no `-m`: `T tests/test_voxel.py::<name>` for `test_WHAT_YOU_HOLD_UP_STANDS_ON_YOUR_FEET_TOO` (1988), `test_a_WORLD_can_be_PUT_BACK_exactly_as_it_was` (2206), `test_the_FAST_PATH_and_the_PLAIN_ONE_agree` (2266), `test_a_body_DOES_NOT_WALL_ITSELF_IN_behind_its_own_back` (2715), `test_an_ELBOW_puts_the_hand_where_ONE_BONE_never_could` (2779), `test_a_POSE_that_rounds_a_limb_APART_is_not_a_POSE` (2859), `test_a_SPINE_BENDS_rather_than_SWINGING` (3038), `test_a_MAN_CROUCHES_and_the_DEPTH_is_his_own_LEGS` (4168) and `test_a_SHOULDER_IS_A_SOCKET_INSIDE_THE_BODY_not_a_hinge_on_the_skin` (4738).
- **Depends:** S3. **Wave:** 1.

**S6. Test tiers.**
- **What:** `tests/conftest.py` registers the `slow` marker and applies it from `SLOW = {"file::test", …}`, generated by `tools/split/slowlist.py $WORK/carve/durations.txt 90`. It raises `pytest.UsageError` for a missing name.
- **Owns:** `tests/conftest.py`.
- **Verify:**
  - `FAST` runs serially in about 90 s.
  - A slow node id given without `-m` runs.
  - A misspelled `SLOW` entry errors.
  - `lab/` tests are unaffected.
  - Record peak memory with `/usr/bin/time -v` before trying `-n 2`.
- **Depends:** S2. **Wave:** 1.

**X1. MPM correctness and exact speed.**
- **What:** First P0d, the inventory (a scratch pytest plugin wraps `Matter.promote`; per-scene seam timings; no repo change). Then:
  - **P0a:** the MAT rows;
  - **P0c:** the box-content wall check;
  - **P1:** the active window, with the early return for no particles;
  - **P2:** the explicit sum, then numba P2G/G2P;
  - **P3:** numba EDT, windowed φ/∇φ, the dense fields dropped, `deep_hits`.
- **Owns:** `src/mpm/core.py`, `src/mpm/matter.py`, `src/mpm/kernels.py` (new), `tests/test_mpm.py`, `tests/test_mpm_seam.py` (new).
- **Verify:**
  - `tools/stepcheck.sh X1 "pole_mpm,seam_fall" src/mpm tests/test_mpm.py tests/test_mpm_seam.py`. P1, P2 and P3 alone must leave every case SAME.
  - `T tests/test_mpm.py tests/test_mpm_seam.py`.
  - New tests:
    - `test_ACTIVE_WINDOW_changes_NOTHING`: `array_equal` on x, v, C, F, Jp, dmg, cid and strength after **every** substep against `window=False`, on block, plank, glass (600 substeps, at least 1 broken) and bound glass;
    - an MPM fast path against the plain path, every substep;
    - a World guard: the hinged pole and the vessel built with `compiled` False and True, every tick;
    - a randomized windowed-SDF field test;
    - promoting every `_NAMES` material does not raise, and `total_mass` holds over 5 ticks;
    - a person walking next to a falling plank: each tick, `mpm.wall[box] == (w.mat[box] != AIR)`.
  - The seam tests found by P0d pass.
  - Benchmarks: block 7.21 → ≤2.0 ms per substep; 400k grid 155.65 → ≤2.2 ms.
- **Depends:** S2. **Wave:** 1.

**EF. Body fuzz harness.**
- **What:** `tools/fuzz_bodies.py`: `harvest.world(seed)` (`harvest.py:73`) plus two `scenes._person` bodies, 300 ticks. It checks the invariants at `docs/voxel-roadmap.md:5166-5172` and `5285-5288`: no shared voxel, one connected lump, no NaN, nothing aloft for ever, and mass conserved when nothing burns.
- **Owns:** `tools/fuzz_bodies.py`.
- **Verify:** `flock $LOCK python3 tools/fuzz_bodies.py --worlds 5` reports its time. Run 150 worlds only if 5 take under 30 s. On today's code it reproduces bug 87 in about 38 of 150 worlds (`docs/voxel-roadmap.md:5377`).
- **Depends:** S2. **Wave:** 1.

**EB. Pin the vessel behaviour before anything refactors it.**
- **What:**
  - Test 1: promote a filled glass vat. Right after the promote, `w.matter.fluid(WATER)` is about what was poured, and the hollow's lattice fluid is 0.
  - Test 2: fill a cup, an open box, a chair, a pipe and a closed box, promote each, and pin which cells' fluid rides on the particles (the carry mask of `matter.py:141-170`) as stored expected values. X4's `_cavity` refactor must keep them.
- **Owns:** `tests/test_vessels.py` (new).
- **Verify:** `T tests/test_vessels.py` passes on HEAD. In a scratch copy, disabling the carry block (`matter.py:141-170`) makes both tests fail.
- **Depends:** S2. **Wave:** 1.

**Q0. Profile the target scene (no repo change).**
- **What:** Run `cProfile` for 40 ticks of `step()` in two worlds: a 100×80×50 house with 4 people and no fire (today about 268 ms per tick), and 8 harvest worlds. Write the top 15 functions to `$WORK/profile_step.txt`. Later steps use it to pick optimizations. Known candidates: `_decide` 13.8% and `_menu` 10.0% of a harvest tick (the mind link asks 1 limb menu per decide, not 4.5); `_objects_within_reach` 9.0% (the E1 labeller); `_claim_bodies` (E7).
- **Owns:** nothing.
- **Verify:** The file exists and its numbers are cited by the step that acts on them.
- **Depends:** S2. **Wave:** 1.

---

**E0. Entity defects first (P step).**
- **What:** The six fixes in 3.3:
  - add `edge` to the `_repose` and `_crouch` field lists;
  - restore edges in the topple wedge path;
  - `_unmake(mask, cause)` for melt and acid;
  - `edge = 0` on shatter targets;
  - `~moving` in reach (`cand`/`solid`) and in the `_object_at` flood, and `_shove`/`_walk` refuse `moving` cells;
  - remove bodies by identity in `_catch_person`/`_catch`.

  Also:
  - `_init_things` sets `_id_prev_mat`;
  - the `_law_identity` stub does the every-tick AIR hygiene for `edge` and `rubble`;
  - `_state_of` (`tests/test_voxel.py:2094-2104`) gains `edge` and `rubble`.
- **Owns:** `pose.py`, `support.py`, `heat.py`, `fluids.py`, `hands.py`, `motion.py`, `things.py`, `tests/test_things.py` (new), `tests/test_voxel.py`.
- **Verify:**
  - `tools/stepcheck.sh E0 any <owned>`, with every DIFFERS line explained by one of the six fixes.
  - `T tests/test_things.py` covers: the edge stays on the tool through a reach and back (`tools/protos/review/edge_probe3.py`); the forearm inherits no edge; taking drawn matter is refused and `total_mass(WOOD)` is unchanged (`drawn_shove.py`); two fliers are caught in one tick.
  - Then the full `T $VOX` (about 20 minutes serial). Read every failing test before any re-tuning.
- **Depends:** S3, S4 (`pose.py`). **Wave:** 2.

**V0. The event core.**
- **What:** In `events.py`:
  - `Ev` (frozen, slotted, with `__deepcopy__` returning self, and `to_json`);
  - `EventLog` (a ring of 4,096, with `get` and `since`);
  - the `VERBS` table with `VERB_ID`;
  - `hash01` (splitmix64);
  - the slotted `Mind` (`perceived` deque(32), `got` dict, `bel` capped at 64, `harm`, `said_ev`/`go_ev`/`hold_ev`/`lean_ev`, and the attention fields `attn` and `news`);
  - `emit`, `because` (`try/finally`), `why`, `_init_events` (`w.events`, `w.minds`, `w.seed = 0`, `w._cover = None`), and `_tick_start` resetting `_cause`.

  In `ledgers.py`, `_NOT_STATE` gains `event_sink` and `_epoch`, and `restore` bumps `_epoch`. In `persons.py`, `add_person` sets `p["pid"] = len(persons)` and appends a `Mind`.
- **Owns:** `events.py`, `ledgers.py`, `persons.py`, `tests/test_events.py` (new).
- **Verify:**
  - Golden E (new names are listed as `NEW:`).
  - `T tests/test_events.py tests/test_snapshot.py` covers:
    - after trimming the ring, ids and `get()` still work;
    - deepcopy shares the records;
    - snapshot/restore keeps the log and bumps the epoch;
    - `emit` appends legacy text only when given `to=`;
    - `because()` restores after an exception;
    - `json.dumps(ev.to_json())` works.
- **Depends:** S3, S5 (`ledgers.py`), S4 (`persons.py`). **Wave:** 2.

**X2. Particle identity, fill and flaws.**
- **What:** In this order: **P5a** (exact), then **P4** and **P5b** (physics). See the table in 3.2.
- **Owns:** `src/mpm/core.py`, `src/mpm/matter.py`, `tests/test_mpm.py`, `tests/test_mpm_seam.py`.
- **Verify:**
  - P5a: stepcheck with `none` declared. Sparse rasterize is bit-identical to the world-sized version. Two promoted clusters keep distinct eids for 100 ticks. The rasterize eid rule matches brute force on 30 random sets.
  - P4/P5b: `tools/stepcheck.sh X2 "pole_mpm,seam_fall" …`. The tree drops from 215 to 86 substeps per tick. Before/after counts of `dmg`, parted particles and canopy cells still attached, for `tree_fell` and `tree_hinge`, are written to the report for owner review. The same cluster promoted at two heights gets equal strengths (rel 1e-12).
  - `T tests/test_mpm.py tests/test_mpm_seam.py`.
- **Depends:** X1. **Wave:** 2.

---

**E1. One labeller and an inert eid.**
- **What:**
  - **L:** `label.py` (numba union-find plus the legacy fallbacks, used when `not w.fused`). Switch torque (`support.py::_law_torque`) and reach (`hands.py::_objects_within_reach`). Add a numba `_object_at` with the same push order.
  - **A1:** `_init_things` adds `eid`, `things`, `_thing_lut`, `_next_eid`, `_stamping`, `_thing_gen` and `_id_losses`. `fill(…)` writes `_stamping`. Carry `eid` at the 14 lattice sites, without changing the landing pointer. Bump `_thing_gen` when a thing cell moves. `_left_world` and `_leave` record losses. Hygiene also clears `eid`. `promote` passes `w.eid` (already supported by P5a). `_state_of` gains `eid`.
- **Owns:** `label.py` (new), `support.py`, `bodies.py`, `pose.py`, `motion.py`, `hands.py`, `fields.py`, `things.py`, `tests/test_label.py` (new), `tests/test_things.py`, `tests/test_voxel.py`.
- **Verify:**
  - Golden E. `T tests/test_label.py` asserts that the labels are equal **arrays** to each legacy routine on random masks, including masks that touch the box faces, and that the flood port returns the same cell **order**.
  - In `tests/test_things.py`, a stamped 40×14×50 scene runs 200 ticks with `w.id_check = True`: the index always matches a recomputation, and `eid == 0` wherever `mat == AIR`.
  - `T "tests/test_voxel.py::test_the_FAST_PATH_and_the_PLAIN_ONE_agree"`, then `FAST`.
- **Depends:** E0, X2 (P5a). **Wave:** 3.

**V1. Events in `will.py`, `senses.py` and `physiology.py`.**
- **What:**
  - Convert the legacy sites in these modules to `w.emit(verb, …, text=<identical>, to=<same dict>)` at the same position:
    - `_law_life`: 5262, 5310, 5322, 5328;
    - `_law_hear`: 5572;
    - `_say`: 5636;
    - `_decide`: 7483–7598, 15 sites;
    - `_law_will`: 7819, 7831, 7838, 7851;
    - `_decide_falling`: 8019.
  - `swings` carries `result` (launched or not).
  - Acts store `go_ev`, `hold_ev` and `lean_ev` on the Mind.
  - `_noise` still builds and returns its dict. `_say` keeps both of its appends and stamps `snd["ev_id"]`. The `steps` event goes at 7939.
- **Owns:** `will.py`, `senses.py`, `physiology.py`, `events.py`, `tests/test_events.py`.
- **Verify:**
  - Golden E: `p["events"]`, speech and sounds are byte-identical.
  - `w.sounds[i]` (tick, at, db, who) equals a pre-step recording for `throw` and `fire_alarm`, apart from `ev_id`.
  - `grep -c '\["events"\].append'` is 0 in the three modules.
  - `T tests/test_events.py tests/test_sound.py`.
  - With `w._cover` on, every converted site is hit by golden plus: `T tests/test_voxel.py::test_a_body_GIVES_UP_on_somewhere_it_cannot_get_to tests/test_voxel.py::test_someone_can_be_DRAGGED_out_and_it_is_a_choice` (the sites at 7831 and 7851).
- **Depends:** V0, S4 (`will.py`). **Wave:** 3.

**X3. Write-back: the heat seam.**
- **What:** P6, as in 3.2. The mover guard is already in E0.
- **Owns:** `src/mpm/matter.py`, `src/mpm/core.py`, `tests/test_mpm_seam.py`.
- **Verify:**
  - `tools/stepcheck.sh X3 "pole_mpm,seam_fall" …`. Seam tests with no heat must stay SAME.
  - New tests:
    - the torch scene's energy ledger with the block promoted equals the never-promoted control within 1 J (prototype −2,888.6 J both);
    - a hot promoted block burns and loses mass (prototype 900 → 737.5 g);
    - shoving a drawn object leaves `total_mass` unchanged;
    - melting a drawn iron cell keeps `total_mass + total_fluid`.
- **Depends:** X2, X1 (P0a). **Wave:** 3.

---

**E2. Registry, identity pass, and names in the menus.**
- **What:**
  - **B1:** in `things.py`, add `register`, `stamp`, `entity_at`, `cells_of` (the index keyed on `_thing_gen`) and `name(eid, fallback, …)`. Make `_law_identity` complete (dirty eids, pieces, heaviest keeps, part-named children, one debris child, the grip follows, forced census, identity events through `w.emit`). Add per-cell T and burning helpers.
  - **A2:** an identity ledger test per carry site, using hand-stamped bodies for the body-only sites.
  - **Text:** the `will.py` text sites (7186, 7237, 7244, 7262, 7282, 7408, 7484, 7502, 7544) use `self.name(e, legacy)`.
- **Owns:** `things.py`, `will.py`, `tests/test_things.py`.
- **Verify:**
  - Golden E (nothing is registered in any case).
  - `T tests/test_things.py` covers:
    - a burned-through plank gives "the plank" and "a piece of the plank";
    - a haft that breaks while held moves the grip to "the haft of the axe";
    - a shattered bottle gives one "the pieces of the bottle";
    - melted tin dies "melted";
    - a bar thrown off the edge dies even when the lattice did not change that tick;
    - a half-promoted thing is two pieces;
    - the per-cell T equals `T()[idx]`;
    - snapshot round trip.
  - Mutation check: for each of the 14 sites, a scratch copy without `eid` in that site's list fails its own scenario (about 5 minutes under the lock).
- **Depends:** E1. **Wave:** 4.

**V2. Events in the physical modules.**
- **What:**
  - The remaining legacy sites:
    - `_release` 3338;
    - `_land_body` 3665, 3726;
    - `_haul_held` 7092;
    - `_law_grips` 8122, 8171, 8186, 8190;
    - `_law_footing` 8534.
  - Pid stamps at 2325, 3336 and 8533.
  - `hits` (`_spend`), `lands` (`_cash_impacts` 1652, `_land_body` 3670), `wounds` (`_blow`), `breaks` (via `_broke`, flushed and sorted in `_tick_end`), `topples`, `looses`, and `melts`/`sets`/`dissolves`.
  - `because()` wrappers in `_spend`, `_land_body`, `_law_crack` and `_cash_impacts`.
  - `w._disturbed` and `_cause_near` for "near" links.
- **Owns:** `bodies.py`, `hands.py`, `motion.py`, `support.py`, `heat.py`, `fluids.py`, `pose.py`, `physiology.py`, `events.py`, `tests/test_events.py`.
- **Verify:**
  - Golden E on `--all`, including `seam_fall` and `pole_mpm`.
  - `grep -rc '\["events"\].append' src/voxel` totals 0.
  - `T tests/test_events.py` covers:
    - a thrown iron's `hits` names the thrower;
    - a whole-body topple onto someone gives `hits` with actor None, the faller as instrument and `falls` as cause;
    - a pane drop gives exactly one `breaks`;
    - the event lists are equal with `skip_quiet` True and False;
    - a new weak-holder test reaches 8190 ("goes over").
  - `T tests/test_wounds.py tests/test_projectiles.py`.
- **Depends:** V1, E1. **Wave:** 4.

**X4. Matter events and `_cavity`.**
- **What:**
  - P7: the per-eid impulse/KE tap; crack and part from `dmg`/`cid`; edge-triggered `struck`; `touch` on onset; `matter.events()`; the `seam_sounds`/`seam_blows` flags, read by the voxel side later.
  - Factor `_cavity(cells, need_fluid)` out of `promote` (`matter.py:141-170`).
- **Owns:** `src/mpm/*`, `tests/test_mpm*.py`.
- **Verify:**
  - Golden E with the tap on.
  - A dropped block gives exactly one `struck` above X J, with speed within 10% of √(2gh).
  - A block on a block is reported in the same tick.
  - A resting pane emits nothing.
  - Tap overhead is under 5%.
  - `T tests/test_vessels.py` still passes against the factored `_cavity`.
- **Depends:** X3. **Wave:** 4.

---

**E3. Holding by eid, and authoring.**
- **What:**
  - **B2:** reach and `_object_at` use the key of 3.3. The take row takes `cell` and `mat` from `where[0]`. `_held_cells` uses `cells_of` for a thing. The swing landing (`_land_body`) and the catch use the eid rule only for named things. The held dict gains `eid` and `name` only when eid > 0. The hands text sites (6260, 7093) use `name()`.
  - **C1:** `thing()` in `scenes.py` (a diff stamp on exit, `mass0`, the affords audit). `w.stamp`. `_person` registers and stamps its body before the re-base (`scenes.py:494-505`). The seed-and-grow in `_claim_bodies` adopts unstamped flesh.
- **Owns:** `hands.py`, `bodies.py`, `persons.py`, `scenes.py`, `things.py`, `tests/test_things.py`.
- **Verify:**
  - Golden E (nothing named).
  - `T tests/test_things.py`: a named iron-head + wood-haft axe is one thing in reach, with rows "take hold of the axe", "swing the axe" and "throw the axe". It is held for 30 ticks through a reach, swung (still held), thrown, and caught by another person, with `held["eid"]` stable throughout. Two named bottles render with relative words.
  - The 9 tests that fill flesh by hand and then call `add_person` (the calls are at `tests/test_voxel.py:544, 567, 701, 756, 911, 960, 2694, 3803, 4968`), then `FAST`.
- **Depends:** E2. **Wave:** 5.

**V3. Perception.**
- **What:** `perceive.py::_law_perceive`:
  - (a) fire bookkeeping under the 7649 gate, handing over `w._burning_now` (`will.py` reads it at 7649–7651);
  - (b) `_law_hear` unchanged, plus `heard` records at the `_level` line and `p["noise"]["ev_id"]`;
  - (c) batched sight records;
  - (d) `did` and `felt` records;
  - (e) the "looked" pass.

  Also:
  - the percept hooks at 7707–7764, which give each percept a source event id;
  - slump and stop-breathing causes taken from `Mind.harm`;
  - the `_after_raster` adapter from `matter.events()` to `w.emit`.

  `_law_perceive` never calls anything that caches or claims.
- **Owns:** `perceive.py`, `senses.py`, `will.py`, `physiology.py`, `events.py`, `tests/test_perceive_voxel.py` (new).
- **Verify:**
  - Golden E on `--all` (this proves `_matter_gen`, noise and wakes are unchanged).
  - `T tests/test_perceive_voxel.py` covers:
    - a footstep through a wall gives `heard` with `actor_seen None` and no `at` belief;
    - a sleeper's wake has a record whose cause is the sound's event;
    - an unconscious listener gets no records, but `p["noise"]` still updates;
    - one fire gives exactly one `ignites` and one `burns_out`;
    - `rays()` agrees with `_sees` on 64 of 64 rays;
    - the hearer's `hears_alarm` points at the shouter's `says`.
- **Depends:** V2, X4. **Wave:** 5.

**X5. Sleep without demotion (only if Q4 says yes).**
- **What:** P8, as in 3.2.
- **Owns:** `src/mpm/*`, `tests/test_mpm_seam.py`.
- **Verify:**
  - The hinged pole goes dormant at about t52, and `matter.step` is about 0 ms afterwards.
  - Ledgers are exact across sleep and wake.
  - Removing the shelf cell under a sleeping plank makes it fall the same tick.
  - Positions are within 1 cell of a run that never sleeps, at t80.
- **Depends:** X4. **Wave:** 5.

---

**M7. The attention law and the policy hooks.**
- **What:**
  - In `will.py`:
    - `attention == "onset"` evaluates every sense each tick;
    - news = a percept whose source event id is not in `Mind.attn` within `attend_gap`;
    - the trigger is `news or (due and goal is None) or wake_due`;
    - `emergency` clears when the goal ends;
    - `sit["onset"]` exists only under onset;
    - `prepare`/`limbs` hooks before the limb loop;
    - `_decided` is set even when no limb is asked;
    - `toward = intent.aim or facing`;
    - `fleeing |= intent.flee`;
    - `_menu` keeps the row matching `intent["want"][limb]` and adds a take row for `intent["take"]` (via `_object_at`, `_in_reach` and `_effort`; the 2-thing stop stays).
  - In `policy.py`: `WILL["attend_gap"] = 40`, and TablePolicy uses its mouth row only on news.
  - In `ledgers.py`: `restore` calls `policy.on_restore(world)`.
- **Owns:** `will.py`, `policy.py`, `ledgers.py`, `tests/test_voxel_attention.py` (new).
- **Verify:**
  - Golden E (latch default), including `duel150`.
  - `T tests/test_voxel_attention.py` covers:
    - the smoky-room probe under onset: ≤ 2 decides in 120 ticks, and the body gets out;
    - a test per hook with a linked stub policy;
    - a key-hygiene grep: `intent` and `job` were not already used as `p[...]` keys;
    - a snapshot of a linked world deep-copies and its job pickles.
  - Harvest cost under onset stays within +10% (time `harvest.to_file(range(24))` both ways).
  - `T tests/test_voxel.py::test_the_ATTENTION_CAP_never_hides_the_answer tests/test_voxel.py::test_the_FAST_PATH_and_the_PLAIN_ONE_agree`.
- **Depends:** V3, S4 (C3). **Wave:** 6.

**V4. Beliefs, sightings of things, misreads and claims.**
- **What:**
  - `beliefs.py`: update rules per verb; `told` never overrides a fresher `saw`; the 64-key cap; `looked` writes.
  - Things: `_law_perceive` adds `entities_seen` (the nearest 8 named things, then the cone and `_sees`) as `("at", ("thing", eid))` beliefs, and `w.has_within_reach(p, prop)` for `sharp`, `holds_liquid`, `burning` and `liftable`.
  - `MISREAD` and `CLAIMS`, both off by default.
  - `harvest.run` sets `w.seed`.
- **Owns:** `beliefs.py` (new), `perceive.py`, `events.py`, `harvest.py`, `tests/test_beliefs_voxel.py` (new).
- **Verify:**
  - Golden E.
  - `T tests/test_beliefs_voxel.py` covers:
    - a thrown hit is felt by the victim, seen by a witness (`hurt_by` = the thrower) and heard behind a wall (no identity);
    - `told` never overrides a fresher `saw`;
    - a thing moved while the person faced away is still believed at its old place;
    - the key count stays ≤ 64 over 2,000 ticks;
    - the same seed gives the same misreads across a snapshot;
    - an honest witness is `grounded=True`, and a scripted liar is `grounded=False, true=False`.
- **Depends:** V3, E3. **Wave:** 6.

**E4. Properties, reach, adopt-on-take and torque.**
- **What:**
  - **B3:** `props(eid)`, cached per `(tick, _thing_gen)`, using X4's `_cavity`.
  - **D1:** lift the 2-thing reach stop behind `w.reach_things_cap` (default 2), add "away" to the rows, and flip the default in a separate commit.
  - **D1b:** drop the reach wrap.
  - **D3:** adopt-on-take for unnamed things.
  - **D4:** torque on the key.
- **Owns:** `things.py`, `hands.py`, `support.py`, `tests/test_things.py`.
- **Verify:**
  - B3: an axe with edge 2e-4 is sharp; a filled vat reports `contains` and `can_hold_ml > 0`; a burning plank renders "the burning plank"; the affords audit flags a "cut" hint on a blunt thing.
  - D1, D1b and D3 are P steps that declare `harvest_*,recall_hall,arms,rescue`. Their reports diff menus and traces against cap 2 with `recall_check` on. A new test: two same-material things on opposite faces of the reach box both appear.
  - D4: a named axe leaning on its haft tips as one cluster, and an unnamed world's torque label arrays are identical.
  - `FAST`.
- **Depends:** E3, X4. **Wave:** 6.

---

**V5. Cause log and readers.**
- **What:**
  - `_decide` and `_decide_falling` emit `decides` (cause = the source event id of the percept) and run their acts inside `because(dec, "call")`.
  - Trace rows gain `row["mind"]` (plain JSON).
  - `w.why`.
  - `perceive.render(rec, viewer)` and `beliefs.view(w, p)`, which returns `Percept`/`Belief`-shaped dicts with `recent(k=5)` and `beliefs_for(menu, k=2)`.
  - The optional `policy.observe(view)` before the limb loop.
- **Owns:** `will.py`, `beliefs.py`, `perceive.py`, `events.py`, `tests/test_cause_log.py` (new).
- **Verify:**
  - Golden E (the new key is left out by name, so the spelled seam text is unchanged).
  - `T tests/test_cause_log.py` covers:
    - `json.dumps` succeeds on every row of `fire_alarm` cut to 120 ticks, with `_save` stubbed;
    - `why()` on the hearer's `goes` walks `goes ← decides(hears_alarm) ← says ← decides(sees_fire) ← ignites`;
    - `render` never names a person who is not in `actor_seen`/`target_seen`;
    - TablePolicy and Roam runs are unchanged.
- **Depends:** M7, V4. **Wave:** 7.

**E5. Named scenes and seam v2.**
- **What:** New scenes with an axe, a bottle and a plank, authored with `thing()`. In `seam.py`, `HIDDEN` gains `eid` and `eids`, and `SPELLING` becomes `"mark1-menu-v2"`. Harvest refuses to mix spellings in one file.
- **Owns:** `scenes.py`, `seam.py`, `tests/test_seam_v2.py` (new).
- **Verify:**
  - `T tests/test_voxel.py::test_a_DECISION_IS_SPELLED_THE_SAME_WAY_EVERY_TIME` (`tests/test_voxel.py:4800`) passes.
  - For 3 harvest seeds, the v2 text equals the v1 text captured before the bump, except for the rules header line.
  - Harvest rows of a named scene show nouns and no integer ids.
  - The integrator re-records golden for the spelling change.
- **Depends:** E4, V4. **Wave:** 7.

**X6. Islands.**
- **What:** P9.
- **Owns:** `src/mpm/*`, `tests/test_mpm.py`.
- **Verify:** Two separated falling objects: the island path is `array_equal` with the single window on every substep, and the cost scales with particle count, not with the distance between the objects.
- **Depends:** X4. **Wave:** 7.

---

**M8. The voxel mind link, M2a (Rules chooser).**
- **What:** `mindlink.py`:
  - `VoxelBackend`:
    - `bind` over things in reach, exits and people, keyed by eid, or by cell for unnamed things, with held things excluded;
    - `rule_scores` from `p["sheet"]`;
    - percepts from `beliefs.view`;
    - beliefs (way out, fire, could not reach).
  - `MindPolicy`: `prepare`, `limbs`, a pure `pick`, `on_restore`; payload matching; falling decides to the reflex table; failure after 3 decide periods; trace rows with `by: "mind"`.
  - `harvest.py` skips `by: "mind"` rows.
- **Owns:** `mindlink.py` (new), `harvest.py`, `tests/test_mindlink.py` (new).
- **Verify:** `T tests/test_mindlink.py` in the `_duel` geometry (attacker at x=17, axe at 22–24 with edge 2e-4, defender at 31, 150 ticks, `attention="onset"`):
  - the argmax Rules chooser opens a wound;
  - at least 18 of 20 sampled request seeds do;
  - the defender's next request has the reason `interrupt:struck`;
  - at most 1 limb menu per decide, and at most 2 requests in any tick;
  - 2 persons on every tick;
  - a deterministic replay;
  - only catalog voxel verbs run;
  - reactive mode makes 0 remote submits.
- **Depends:** V5, M5, E3. **Wave:** 8.

**E6. Body-eid audit (no gate on zero).**
- **What:** With `w.id_audit` on, `_claim_bodies` logs claimed-minus-eid and eid-minus-claimed per person, tagged `aloft`, `reserved`, `drift`, `merge` or `other`.
- **Owns:** `persons.py`, `things.py`, `tools/fuzz_bodies.py`.
- **Verify:** A histogram over the EF worlds plus the 12 claim-dependent tests (`tests/test_voxel.py:1150, 1703, 1862, 1988, 3518, 3941, 3995, 4552, 4738, 5003, 5068, 5124`) goes into the report. Every `other` case is read.
- **Depends:** E3, EF. **Wave:** 8.

---

**M9. Offline model gate (the user runs it in the lab venv, one process).**
- **What:** Score the recorded M5 (rooms) and M8 (voxel) requests with GLiNER2.5-Decide:
  - (a) the new describe against the lab describe;
  - (b) pre-rank recall of the model's full-menu pick, and the "surprise" score with and without the cut;
  - (c) canonical order against 4 permutations;
  - (d) model–rules agreement on gated decisions;
  - (e) `lab/serve.py` memory in CPU and GPU modes.
- **Owns:** `lab/methods/score_requests.py`.
- **Verify:** A report with the five numbers. Go on to M11 only if recall ≥ 95%, the new describe ≥ the lab describe, and the server fits in free RAM minus 1 GB.
- **Depends:** M6, M8. **Wave:** 9.

**M10. Approach, M2b.**
- **What:**
  - `go:seek` rows from `intent["seek"]` in `_places`;
  - a stand-off band per stance, from a calibration test that sweeps `stand` (`tests/test_wounds.py:155-163`: unsquared 0.55–0.75 m, squared up 0.40–0.45 m);
  - a `sees_attack` start percept from a visible harmful job start;
  - more than one held item.
- **Owns:** `mindlink.py`, `motion.py`, `tests/test_mindlink.py`.
- **Verify:**
  - The calibration reproduces the energy-against-distance band for both stances.
  - The attacker approaches, takes the weapon and opens a wound from at least 1.5 m away.
  - No fall happens next to the attacker.
  - A deterministic replay.
- **Depends:** M8, E4. **Wave:** 9.

**X7. The fused substep (only after roadmap decision 2).**
- **What:** P10.
- **Owns:** `src/mpm/*`, `tests/test_mpm.py`.
- **Verify:**
  - A one-time tolerance check of the fixed-sweep algorithm against LAPACK on every `tests/test_mpm.py` scene. After that, fused against numpy is `array_equal` every substep.
  - Through `World.step` at 5 cm: the plank beside a person costs ≤ 10 ms of solver per tick (today 451 ms).
- **Depends:** X6, Q4. **Wave:** 9.

**M11. Live model (opt-in, only if Q3 says yes).**
- **What:** `lab/serve.py` (JSON lines over a Unix socket; one request per forward pass; the duty sleep between requests) and `src/mind/remote.py` (a standard-library client with a watchdog). Featured mode uses `Combined(sense=Remote)` and the gate. Rooms run first, then voxel.
- **Owns:** `lab/serve.py`, `src/mind/remote.py`, `tests/test_mind_remote.py`.
- **Verify:**
  - With an in-process fake server: a round trip, a timeout (realtime falls back, deterministic aborts), and a cancelled reply is dropped. No GPU is used in tests.
  - Live runs report model calls per turn, agreement with rules, and the lab's layer-3 signals against the Rules baseline on the same seeds.
- **Depends:** M9. **Wave:** 10.

**E7. Body eid replaces the claim (gated on the bug-87 landing fix).**
- **What:** `_person_cells` reads the eid mask behind `w.body_by_eid`. Then delete the claim grow, `_carry_own` and the `_own` frozensets.
- **Owns:** `persons.py`.
- **Verify:** The full `T $VOX`, plus EF with both invariants.
- **Depends:** E6 and the roadmap-87 fix (`docs/voxel-roadmap.md:5544-5547`). **Wave:** 10.

**E8. One `CARRIED` tuple.**
- **What:** `CARRIED = ("mat", "smass", "E", "fl", "fvol", "fpot", "edge", "fallh", "rubble", "eid")`, with `_lift`/`_put` used at all 14 lattice sites. `_lift` bumps `_thing_gen` (built in wave 11a; the user kept it there on 2026-10-16, because moving the bump to `_put` changes nine golden counters and nothing else).
- **Owns:** `support.py`, `bodies.py`, `pose.py`, `motion.py`, `hands.py`.
- **Verify:** A P step. Each field newly carried at a site that omitted it (`fallh`/`rubble` in `_walk`; `rubble` in `_shove`, `_turn`, `_repose`, `_crouch`; `fallh` in the topple wedge restore) is re-measured by the E2 ledger tests and the full `T $VOX`.
- **Depends:** E2, M10 (`motion.py`). **Wave:** 10.

---

**The mind lane: M1–M6 (after S0, in order, one agent).** None of these touch `src/voxel`.

| step | what | owns | verify | depends |
|---|---|---|---|---|
| **M1** | API types, catalog loader, `verbs.yaml` | `src/mind/{__init__,api,catalog}.py`, `src/mind/data/verbs.yaml`, `tests/test_mind_catalog.py` | `T tests/test_mind_catalog.py`: every check of 3.5; frozen types hash; `grep -rn "import lab\|from lab" src/` finds nothing | none |
| **M2** | Inn goldens: `lab/world2/golden.py` runs `World.step(Rules())` for 300 turns on seeds 0–2 and writes `tests/data/inn_rules_seed{0,1,2}.jsonl`, with the lab file's sha1 in a header | `lab/world2/golden.py`, `tests/data/inn_rules_seed*.jsonl` | 438, 425 and 449 events; content hashes pinned in `tests/test_rooms.py` | none |
| **M3** | The rooms port (fork) | `src/rooms/{__init__,world,rules,backend}.py`, `tests/test_rooms.py` | `T tests/test_rooms.py`: compat mode is byte-identical to M2; the lab's 4 checks (`lab/world2/test_world.py:20-61`) are ported; start records never call `_learn` and never draw from the RNG; under 2 s | M1, M2 |
| **M4** | Neutral functions: perceive, beliefs ≤ 2, menu with pre-rank and canonical order, blocked options and needs sentences; `describe` (spelling `mark1-mind-v1`) | `src/mind/{core,describe}.py`, `tests/test_mind_core.py` | On 6 inn nights: menus ≤ 16 with the null act; the favourite is first in about 8% of menus; mean mass cut < 1%; needs sentences appear; no "Things here", no digits, ≤ 2 beliefs, ≤ 5 percepts; 0 duplicate option texts; the same input gives the same bytes | M3 |
| **M5** | Loop, choosers and CLI: `MindConfig` modes; the per-tick remote counter; `python -m src.rooms run/--play/session` | `src/mind/{loop,choosers}.py`, `src/rooms/__main__.py`, `tests/test_mind_loop.py` | `python -m src.rooms run --seeds 6 --turns 300 --chooser rules` completes; every started job was in its menu; two runs are identical; reactive mode makes 0 remote submits on every tick; every job row has a `Why`; `lab/judge/signals.py` `run_numbers`, run in the lab venv on the output files, gives help and harm > 0 (equal to the lab's in compat mode) | M4 |
| **M6** | Record and replay: a `Recorder` for lab case-shaped rows; the `Replay` chooser; `lab/methods/score_requests.py` | `src/mind/record.py`, `lab/methods/score_requests.py`, `tests/test_mind_record.py` | Record a night, replay it with a fake decisions file: a byte-identical run; a missing id raises in deterministic mode; the test never imports `lab` | M5 |

### 4.5 What can start right now

- **The user:** S0 (commit), then answer Q1–Q5. Only Q1 blocks the build; Q2–Q5 block later steps.
- **Agents, after S0:**
  - one agent runs S1 → S2 → S3 (phase 0);
  - one agent runs the mind lane M1–M6;
  - P0d (the inventory at the start of X1) and Q0 (profile) need no repo change. They run whenever the lock is free.

---

## 5. Open questions for the user

1. **Commits and process.** The build needs a commit at S0 and at the end of every wave. HEAD is the base that each step's sandbox and golden check compare against. May the integrator agent make these commits, after showing you the wave report? If not, each wave waits for you to commit.
2. **Attention law.** Should `attention = "onset"` become the default for every body? That means re-blessing the scenes: swing traces go 84→48, ledge 44→87 and rescue 40→97, and one ledge body ends one cell over. After that, delete the latch law. Until you say yes, onset is opt-in, mind-linked worlds turn it on world-wide, and the harvest stays on latch.
3. **The model crossing.** Record, score and replay only? That is the default and needs no change to the lab isolation rule. Or also a live socket (`lab/serve.py`), which amends that rule and must fit in free RAM minus 1 GB (about 1.6–2 GB is free today)?
4. **Particle decisions:**
   - (a) Build sleep now (X5), although roadmap 93 lists it as next-phase work?
   - (b) Decide bonds or continuum before the fused substep (X7)?
   - (c) How should a person pick up a fallen particle object: demotion-on-act, or the person as a collider? Until then the act fails with "it is still moving".
   - (d) Rigid values for ash, char and tin: 0.0, 1.0 and 0.9?
   - (e) Delete or enable `_free_fall_to_seam`? Nothing sets its flag (`sim.py:1000, 1399`), but the alchemist docstring describes it as live (`scenes.py:388-395`).
5. **Backends and motive:**
   - (a) Keep `src/rooms` as a fork of the lab inn, synced only through the M2 goldens?
   - (b) Use a declared character sheet (`p["sheet"]`) as voxel motive until a disposition model exists?
   - (c) In featured mode, which characters get the model, and should the gate be g = 0.7?
