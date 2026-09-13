# Where every number came from

A roadmap entry that says "caps falls at 8 m/s" and one that says "measured,
779 J of 1000 after sixty ticks" read exactly alike, and only one of them is
worth trusting. Six entries did not survive being checked on 2026-09-12 — the
fall cap had the sign backwards, the humanoid's mass was stale by 10 kg, two
described bugs that did not exist — and every one of them had sat for weeks
looking like a known quantity.

So: **every number in this engine says where it came from.** This file is the
ledger. The kinds are:

| kind | means |
|---|---|
| **WORLD** | a physical constant. Look it up; it is the same for everyone. |
| **MEASURED** | a probe was run and this is what it said. The probe is named. |
| **DERIVED** | computed from something else the sim already knows. Cannot drift. |
| **DECLARED** | a fact about a particular body or object, said by whoever built it. |
| **GUESS** | nobody has measured it. Said out loud so it can be doubted. |

## The loop worth having

The best number is one nobody types. The sim knows where every voxel of every
limb is, so anything shaped like "how far can an arm reach" has a GROUND TRUTH
sitting in the lattice already, and reading it beats declaring it three ways
over:

- it cannot drift out of step with the body it describes;
- a child, a giant and a one-armed man each get the right answer with nobody
  writing three rows;
- and the law stops being a claim about people in general and becomes a claim
  about THIS person, which is what the sim is for.

**It had already drifted.** `BODY["reach_m"]` said 0.30 m; the humanoid's arm,
measured off its own voxels, is **0.45 m**. Nothing was wrong with either
number — they had simply never been asked to agree.

Converted so far:

| was typed | now read from | value |
|---|---|---|
| `BODY["reach_m"]` = 0.30 | the arm's own far end from its own shoulder | **0.45 m** |
| `BODY["off_hand_m"]` = 0.10 | half the body's own shoulder separation | 0.10 m |
| `WILL["decide_every"]` = 30 | `react_s / TICK_S` | 10 ticks |

The constants stay as the FALLBACK, for a body with no segments declared. They
are what a person is like when nobody has said.

Still typed, and each one is a loop waiting to be closed:

| number | what it should be read from |
|---|---|
| `BODY["crouch_m"]` 0.25 | the leg's own length — how far this body can actually sink |
| `BODY["lean_max"]` 0.8 | the angle past which the lattice cannot draw this spine (measured 0.34) |
| `BODY["arm_wmax"]`, `arm_Nm` | a muscle's cross-section, which is voxels of flesh |
| `WILL["step_m"]` 2.0 | a stride, which is a leg length |

## Time and the world

| name | value | kind | why |
|---|---|---|---|
| `GRAVITY` | 9.81 m/s² | WORLD | |
| `TICK_S` | 0.025 s | DECLARED | the clock everything else is derived against; 40 Hz |
| `AIR_DENS` | 1.2 kg/m³ | WORLD | at room temperature |
| `DRAG_CD` | 1.1 | WORLD | a blunt slab, near enough |
| `AMBIENT` | 20 °C | DECLARED | the world's resting temperature |
| `FRICTION` | 0.40 | WORLD | sliding, solid on solid; one number until materials need their own |
| `FALL_SUBSTEPS` | 16 | MEASURED | 4 made a 20 m fall arrive 41% too fast (twice the energy). 16 is honest to ~40 m and costs 1.01x a tick — `test_a_LONG_FALL_arrives_at_the_speed_gravity_gives_it` |
| `RAD`, `LEAK` | 3e-9, 0.004 | GUESS | the lattice's boundary with a colder universe. Tuned so flames cap; the SHAPE is Stefan-Boltzmann, the scale is not measured. `total_energy()` counts what they take |
| `COMBUST_CEIL` | 1100 °C | WORLD | flames saturate about here |
| `SEE_TAU` | 2.0 | GUESS | optical depth a look still penetrates |
| `PLUME_REACH_M` | 2.4 m | MEASURED | a physical length, not a cell count: the same bed fire suffocated at 5 cm and thrived at 10 cm before this was in metres |

## A body

| name | value | kind | why |
|---|---|---|---|
| `react_s` | 0.25 s | WORLD | a simple visual reaction: see, choose, begin to move |
| `decide_every` | 10 ticks | DERIVED | `react_s / TICK_S`. Was 30 (0.75 s), nearly three reaction times |
| `reach_m` | 0.30 m | DERIVED per body | fallback only; read off the arm at 0.45 m |
| `off_hand_m` | 0.10 m | DERIVED per body | fallback only; half the shoulder span |
| `off_hand` | 0.8 | GUESS | what the other hand is worth. Nobody has measured it here |
| `catch_s` | 0.12 s | GUESS | how long a closing hand gives. It is the whole of why a ball can be caught and a brick cannot |
| `throw_rad` | π/4 | WORLD | the angle a projectile goes furthest from. Geometry, not taste |
| `windup_rad` | 0.9 | GUESS | how far back a throw starts |
| `arm_Nm` | 60 N·m | WORLD | an unremarkable adult shoulder |
| `back_Nm` | 200 N·m | WORLD | low end of an untrained adult trunk extensor |
| `arm_wmax` | 15 rad/s | WORLD | Hill's force-velocity: torque fades to nothing at top speed |
| `waist_wmax` | 1.5 rad/s | GUESS | a trunk is slower than a shoulder. At a shoulder's pace the first tick of a lean is already past what a man can stand at |
| `lean_max` | 0.8 rad | GUESS | the spine's own stop. The LATTICE stops this body at 0.34 — item 49 |
| `strength_N` | 400 N | WORLD | what an adult can shift |
| `legs_N`, `crouch_m` | 1400 N, 0.25 m | WORLD / GUESS | a jump is legs_N over a crouch; the crouch should be read off the leg |
| `MENU_CAP` | 7 | MEASURED | a model of attention, not a budget. Recall 81% → 100% when the cap kept one of each KIND first — "What attention was costing" |

## Measured thresholds

These exist because a probe was run. Each names its probe.

| name | value | probe |
|---|---|---|
| `SMOKE_STILL` | 1e-6 | the first wisp of smoke was wrong by 1.7e-5 when `_law_air` was gated on pressure alone |
| `BODY["feel_T"]` | 45 °C | the `scorched` percept: a body standing ON a fire could not feel it |
| `WILL["forget"]` | 0.985 | about a minute to half-weight at the scan rate. A body that never forgets treats an hour-old fire as a fire |
| `_SWEEP_RAD` | 0.08 rad | three quarters of a voxel at the end of a nine-voxel arm, so a sweep cannot step over a wall |

## What this file is for

Two things, and the second matters more.

**So a reader can tell which lines to trust.** A GUESS is not a failing — most
of them are the right shape and an unknown scale, and saying so is what lets
the next person go and measure one instead of assuming it was measured already.

**So the loops get closed.** Every row above marked GUESS or DECLARED where the
sim already holds the ground truth is a law waiting to be driven by the world
instead of by a person's opinion. That is the direction: the sim's own facts
should set its own numbers, and a number that cannot be derived should have to
say why.
