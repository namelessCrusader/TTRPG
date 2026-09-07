# Voxel core — gaps, stolen ideas, build queue

The end-state this queue builds toward is scoped in `docs/voxel-full-scope.md`
(what physics says per subsystem, what we run, the references to copy from).

What the simulation cannot do yet, why, and in what order we close it. Sibling of
`rulings.md` (which holds the case law); this holds the capability map. Updated as
scenarios expose new lies.

## Where we are

Laws: fluid fall/spread, density sinking + splash displacement, support (span
strength), torque (base + waist tipping), conduction + T⁴ radiation, buoyant rise
(gas share only), combustion with per-airspace O2, boiling, dissolution via the
REACTIONS data table, smoke transport. Materials: wood, stone, iron, flesh, ash,
glass. Fluids: water, oil, vitriol, weak acid. ~250 tests. Scenes and clips in
`src/voxel/scenes.py` / `renders/`.

## Known lies, each with its missing cause

| Symptom (seen in a clip) | Missing cause | Closed by |
|---|---|---|
| ~~A shut door is a perfect gas seal — no smoke, no oxygen, ever~~ | **FIXED (2026-09-07)**: the gas laws asked `mat == AIR`, a binary question about a continuous field, so the fill fraction `smass` has always carried was thrown away. `porosity()` now DERIVES void from solid mass over packed density; `_law_o2` seep, `_law_smoke` seep and `_law_air` advection all scale by it. Two hardcodes died with it (O2's flat 0.01 through every solid; smoke's hard air-only gate). A door leaks because the SCENE says it fills 97% of its cells. Free consequences nobody wrote: rubble and thatch breathe, acid-eaten walls breathe through the loss, and burning wood opens up as it thins (measured: void 0.000 → 0.163 at the charred face). See Ruling 2 | done — but see the row below |
| A shut door leaks, but far too little: ~450× tighter than an open doorway for a 10% gap | a gap is an ORIFICE, not a porous medium. Void fraction is the right shape for one face, but a 2-voxel-thick door makes three faces in series, and an orifice does not get tighter because the wall is thicker. Kozeny–Carman would make this *worse*, not better — it describes packed beds, not holes | an aperture treatment: conductance along a contiguous porous run takes the MINIMUM face, not the product. One formula, no per-object cases. Alternative (rejected for now as a case in disguise): a CFAST-style room-to-room leak law |
| A burst vessel's water "vanishes" — soaked into the shard heap (drop_test) | _shatter scatters rubble into the cells the spill pooled in; fluid inside a solid isn't rendered | shard placement could prefer dry cells; or render soaked rubble darker |
| Acid fumes and fire smoke are the SAME gas | one untyped `smoke` field | gas species (#2) |
| Lamp chars shelf at 240 °C forever, never ignites | no pyrolysis: real wood at 200–250 °C slowly converts to char that ignites LOWER; we have a fixed 300° line | time-at-temperature chemistry (#5) |
| Torch handle burns like a bare stick; flames have zero height | combustion happens IN the fuel voxel; real flames are burning gas ABOVE it | flame-above-fuel, needs fuel vapor (#4 after #2) |
| Torch over gasoline wouldn't flash | no evaporation → no flammable vapor | gas species (#2) |
| Hinged tree falls flat in ONE tick — no visible gravity arc | topple is an instant quarter-turn; no free-body state | free bodies (#3) |
| Straight-cut symmetric tree lands on its stump and stands | perfectly balanced = stable in our statics; real trees always have asymmetry | fine as physics; scenes should build asymmetric trees |
| Water levels never equalize through a pipe; splash has no flying droplets | no pressure, no momentum — flow is fall + level-diff seep | pressure + momentum (#1) |
| Doorway-joined rooms share air INSTANTLY; a keyhole = an archway | per-region mean-mix ignores aperture | pressure + momentum (#1) |
| Sealed-room fire builds no pressure; no explosions, no drafts | same | pressure + momentum (#1) |
| Wood-on-wood is ONE body to the torque law | object identity by material adjacency only | free bodies / object ids (#3) |
| ~~A landing body "shatters" — crown scatters into debris~~ | **BUILT**: TOUGH column (kJ/m², Charpy-class) — fallen voxels carry real drop height, landings cash m·g·h per contact cell; free bodies pay their COM drop; ponds cushion (drag bleeds fallh). TSHOCK column: thermal shock cracks flame-licked glass. Fluid in a burst vessel spills | done (2026-09-03) |
| ~~A 90%-burned trunk voxel carries like solid timber, then flips to ash~~ | **FIXED (2026-09-07)**: span is no longer read straight off the material — it scales with the fraction of the voxel that is still there, full reach down to half mass and decaying below. Measured cantilever reach by fill: 1.0/0.8/0.6/0.5 all hold to x=29, then 0.4→24, 0.3→19, 0.2→15. The knee sits at a half so scene `frac` fills keep their strength. A burning tree can now come down onto its own fire | done |
| Fire climbs a trunk far slower than real bark fires | combustion in bulk wood only; no surface-flame spread / flame height | flame-above-fuel (#4) |
| A held torch burns the holder's hand only in theory | no flesh heat-damage model | entities layer (with the humanoid) |

## Stolen designs (Powder Toy, adapted to 3D — ideas and constants only, never code; GPLv3)

1. **Phase-transition quadruple.** Every material: (lowT, highT, lowP, highP) →
   target material. Ice↔water↔steam, lava↔stone as DATA. Slots straight into our
   REACTIONS philosophy; pressure thresholds activate once #1 exists.
2. **Polymorphic carriers.** Their LAVA remembers what it melted (`ctype`) and
   freezes back into it. One "molten-X" fluid + one "frozen-X" solid multiplies
   the interaction space without new materials. Needs one extra per-voxel id.
3. **Pressure coupling.** Ignition chance and boiling point shift with local
   pressure. Cheap once #1 exists; big behavioral payoff (their explosions and
   geology all hang off this).
4. **Their element list as a menu.** Salt/salt-water, thermite, plants/growth,
   virus, soap... each is a handful of REACTIONS rows for us, not a law. Their
   inline rate constants are calibration references.
5. **What NOT to copy:** no conservation (cloners/voids), stochastic heat,
   per-element if-chain reactions, 2D-specific particle tricks.

## What we may copy, from where (so each build starts from proven logic)

License decides HOW MUCH transfers: permissive (MIT/zlib/BSD/Apache) = port the
actual logic/code with an attribution note; LGPL/GPL = ideas and constants only,
reimplement from spec; closed = published talks only.

| Source | License | Take | Feeds queue item |
|---|---|---|---|
| box3d / Box2D (Erin Catto) | MIT | CODE OK: semi-implicit Euler integrator, sequential-impulse contact solver, soft constraints, sub-stepping scheme; his GDC papers/talks document all of it | free bodies (#3) |
| Bullet | zlib | CODE OK: compound-of-boxes collision, island-based sleeping, CCD ideas; also the pre-decided escape-hatch library (pybullet) | free bodies (#3) |
| Principia | BSD-3 | CODE OK: the component vocabulary — sensors → signal cables → logic gates → actuators, decoupled from the solver; entity/group/connection model | devices layer (#6) |
| Rapier | Apache-2 | CODE OK: cross-platform determinism practices (strict IEEE-754 discipline), snapshot/restore design | free bodies (#3), replay |
| The Powder Toy | GPLv3 | IDEAS ONLY: air-field update order (taken already), phase quadruple, ctype polymorphism, element menu + rate constants as calibration | pneumatic (done), reactions (#2, rows) |
| Noita | closed | TALK ONLY (GDC 2019): the promote/demote protocol — pixels→rigid body→re-pixelate, velocity handoff, when to promote | free bodies (#3) |
| Luanti/Minetest | LGPL | IDEAS ONLY: falling_node (voxel becomes entity mid-fall, voxel again on landing) as the minimal free-body form; per-material behavior flags | free bodies (#3) |
| Veloren | GPL-3 | IDEAS ONLY: voxel-structure-with-pose vs voxel-terrain coarse collision (ships over terrain) | free bodies (#3), vehicles someday |
| Terasology | Apache-2 | CODE OK: world-behaviors-as-optional-modules structure | pack architecture |

## Where material property ROWS come from (surveyed 2026-09)

Goal: new materials and reactions are data rows. Verdicts, best first:

| Source | Covers | Machine access | Verdict |
|---|---|---|---|
| **FSRI MaP database** (materials.fsri.org, github.com/ulfsri/fsri_materials_database) | 100+ real household/building materials: measured k, heat capacity, density, cone-calorimeter ignition/HRR, heats of combustion, decomposition | CSV + JSON per material, on GitHub | **PRIMARY for solids** — literally fire-model input data; check repo LICENSE before vendoring |
| PubChem PUG-View REST | pure chemicals: heat of combustion, flash point, autoignition, density | full REST API, open | secondary, for LIQUIDS/reagents; annotation strings need parsing; nothing for "oak" |
| NIST Chemistry WebBook | authoritative thermochemistry | no official API; SRD copyright | manual cross-check only |
| GESTIS (IFA) | ignition temps/flash points, ~9,400 substances | unofficial REST only | gap-filler |
| Engineering ToolBox | best wood/stone/metal k, cp tables | none; copyrighted site | hand-transcribe individual values with a `source` note; no scraping |
| MatWeb | alloys/polymers | ToS bans bulk export | skip |
| Wikipedia (list articles + infoboxes, CC BY-SA) | scattered everything | parseable | gap-filler. The Category:Chemical_reactions tree is prose about reaction TYPES — no data |
| FireBID (UMD), sedaoturak/data-resources index | fire-oriented material props; open-dataset index | varies | worth a skim for extra rows |

**The reaction-rate finding**: no open dataset maps (fluid, solid) → attack
rate. Real kinetics databases (NIST) are gas-phase elementary reactions —
useless here. Chemical-compatibility charts (qualitative A/B/C/D ratings) are a
defensible ordinal PRIOR: map A→0, B→slow, C→fast, D→very fast, then hand-tune.
So REACTIONS stays hand-curated; that is the honest state of the art.

## Build queue (dependency-ordered)

1. **Pressure + momentum field — BUILT (first cut).** Three pieces:
   hydrostatic head (`_law_head`: one connected liquid body seeks one surface —
   vessels equalize; v1 limit: pipe width doesn't throttle the rate yet);
   ballistic droplets (`_fly_drops`: splash parcels arc under gravity and rejoin
   the grid; conservation counts them in flight); pneumatic field (`_law_air`:
   coarse 4-voxel cells with lossy pressure+velocity, fires and steam inject
   pressure, wind advects smoke and O2, sealed rooms hold pressure, boiling
   point shifts with pressure). Still open: pressure pushing SOLIDS (blast),
   aperture-limited liquid rates.
2. **Gas species** — the smoke field becomes (species, grams): smoke, acid fume,
   water vapor, fuel vapor. Closes the water cycle (boil → vapor → condense),
   separates acid smog from fire smoke, feeds #4.
3. **Free bodies** — a toppling/falling/thrown cluster leaves the grid as a body
   with pose + velocity, advances under gravity with real angular acceleration,
   re-rasterizes on landing. Fixes the too-fast tree, enables throwing, carried
   objects (the torch in a hand), impact. **Survey verdict: hand-roll in numpy**
   (own collision world = the voxel grid; syncing terrain into pybullet/Jolt/
   Rapier is more code than the integrator; determinism stays ours). Copy the
   Noita promote/demote protocol: cluster → body (mass/inertia from voxels,
   velocity handed over) → re-rasterize on impact. Escape hatch if scope grows
   to joints/ropes/vehicles: pybullet (zlib, mature bindings); re-check Rapier's
   PyPI status then. Watch: erincatto/box3d (MIT, alpha, no Python yet).
   Mechanism vocabulary to mine later: Bithack/principia (BSD-3) — sensors →
   signal cables → actuators, decoupled from the physics solver.
4. **Flame above fuel** — burning fuel emits fuel vapor (#2); vapor + O2 + heat
   combusts IN THE AIR: flames get height, torches burn at the head only, lamps
   ignite shelves at honest distances, gasoline flashes.
5. **Pyrolysis** — wood held at 200–300 °C slowly converts to char (lower
   ignition); the lamp shelf eventually DOES catch, hours-scale fire causes work.
6. **Electricity** — even TPT special-cases all of it; ours would be a data-pack
   layer (conductors, sources, switches). Parked until a scenario demands it.

## Perception reads continuous fields — BUILT (2026-09-07)

The Ruling-2 sweep, applied to the three places that asked a binary question of a
continuous field. Demo: `glasshouse` scene + `renders/glasshouse/`.

- **Sight is optical depth.** `_sees` accumulated a material whitelist
  (`AIR | FLESH`); it now accumulates Beer-Lambert optical depth. A partial voxel
  blocks by COVERAGE, which needs no constant — fraction `f` of the cell is stuff,
  so `f` of the light hits it. Only `TRANSMIT = {GLASS: 0.96}` is a table, because
  transparency genuinely is a material property. Fixes: glass was as blind as
  masonry, a hedge walled off sight completely, and **smoke did not obscure at all**
  (a listed open issue). The "bodies don't wall off sight" case is gone — the
  viewer's own body is skipped by geometry, not by exempting flesh.
- **Hearing is the acoustic mass law.** `_hears` charged a flat 4 m per solid
  voxel — pine, masonry and a leaf curtain identical. Now: 90 dB shout, spherical
  spreading, and barrier loss from MASS PER AREA read off `smass`
  (`20·log10(σf) − 47`, one law applied to the path's total mass, not per slice).
  5 cm masonry ≈ 135 kg/m² ≈ 50 dB; the same pine ≈ 30 kg/m² ≈ 37 dB. Measured in
  the scene: Bram hears through the pine door, and stone kills the same shout.
- **Passage reads fill.** `_plan_path` and `_walk` both demanded pure AIR, so one
  leaf voxel was as impassable as masonry — and they disagreed after the first
  fix, the planner routing through a hedge the legs then refused. Both now use
  `PUSH_THROUGH` (a stand-in until the force law). `_walk` also **displaces what
  it shoves**: entered cells' contents move to vacated cells, so the hedge closes
  behind you and no leaf is deleted (locked by a conservation test).
- **Pressure reads porosity** for its block threshold, so a 97%-filled door is a
  path. Kept HARD, not graded: grading the coarse 20 cm cells by volume fraction
  let sealed rooms bleed pressure through masonry (a test caught it).

## Information leaks — what a mind may know (audited 2026-09-07)

**Ablation run, and now a test** (`test_a_body_learns_of_the_fire_ONLY_by_perceiving_it`):
the glasshouse with its window bricked up. Identical fire — 181 burning voxels
either way, same 0.008 g of smoke through the door — but with masonry in place of
glass the witness never sees it, takes NO decision, says nothing, and never moves.
Sight leaks nothing.

**But geometry is still omniscient, and events are not the only channel:**
- `ex = min(self.exits, ...)` — a body picks the nearest exit from a GLOBAL list.
  Nobody has to have seen a door to know exactly where it is. A stranger in an
  unfamiliar building would not.
- `_plan_path` runs its BFS over the WHOLE `walk_ok` grid — perfect knowledge of
  the entire floor plan, including rooms never entered. Bodies route around
  baffles they cannot see.
- Consequently the menu is built from omniscient pathing: `_menu` calls
  `_plan_path`, so "flee" is offered on the strength of a route the body has no
  business knowing.

The fix is the one `src/core` already has and the voxel minds lack: a BELIEF of
the world (what this body has actually seen), with pathing and exit choice run
against the belief rather than the lattice. Core item 1 already binds contact
options to *fresh belief nodes*; the voxel side is behind it. Until then the
honest description is: **perception gates EVENTS, not GEOMETRY.**

## Two body bugs the RENDER caught and the tests did not (2026-09-07)

- **A body is not a point.** `_plan_path` routed the body's CENTRE, but a body
  is ~5 voxels across, so a door set in an outer wall is a column the centre can
  never occupy — its shoulder would be in the masonry. The planner routed there,
  the legs correctly refused, and the walker jammed against the wall shuffling
  sideways for the rest of the run (measured: stuck 2 voxels short of its door
  for 500 ticks — this was the long-standing "Bram never gets out"). Fixed by
  eroding the walkable grid by the body's own footprint before searching, aiming
  at the nearest column the body actually fits when the goal itself is too tight,
  and measuring **arrival from the body rather than its centre** — you are at the
  door when part of you is at the door. Bram now leaves at t226.
- **Fainting tripped one way.** A body carried out of the smoke stayed
  unconscious for ever, so rescuing anyone was pointless. Air can undo what air
  did: `wake_o2` (0.70 against `faint_o2` 0.55) brings a body round with real
  hysteresis, and never for burns, whose integral only climbs. Not modelled: it
  wakes where it fell — getting back to its feet is not a thing bodies can do.

## Belief and impetus — BUILT (2026-09-07)

Two asks that turned out to be one build: bodies must not hold information they
never observed, and bodies must do something when nothing is happening. If a body
only knows what it has seen, then **background activity is what builds that
knowledge** — you know where the door is because you once walked through it.

- **`p["known"]`** — a per-body map of columns it has actually looked at.
  `_look_around` casts the facing cone through the same opacity the sight law
  uses, so a wall casts a shadow of ignorance and a doorway lets knowledge
  through. Turning the head fills it in; walking fills it in properly.
- **Pathing and exit choice run on belief.** `_plan_path` masks by `known`;
  `_goals` offers `exit` only for doors this body has seen. A door never laid
  eyes on is not a destination — it is not even a menu option.
  `add_person(knows_world=)` states it per character: the default True is the
  usual case (someone standing in a place they live HAS observed it), and
  `False` gives a stranger who must find the way.
- **Impetus.** With no percept the body still weighs `explore` / `wander` /
  `stay` through the same menu, picked by the same policy, written to the same
  trace. Idling is a choice, not a gap between choices. Measured: a stranger
  goes from 44% to 98% of the floor known, choosing `explore` each time; a
  resident wanders instead, having nothing left to find.
- **Three real bugs found doing it.** `arrive_m` was 0.5 m = TEN voxels at 5 cm,
  so a body "arrived" the moment it chose anywhere nearby and never took a step
  (and people were called safe nine voxels short of the door). Percepts had to
  wait out the 30-tick deliberation cooldown, so a 30-tick shout could expire
  unheard. And gating the senses on "is moving" made a wandering body blind to
  fire — having somewhere to be is not the same as being in an emergency.

Still omniscient, now the only one left: nothing models *forgetting*, and belief
records geometry only — a body does not remember that the hall was full of smoke
a minute ago, only that the hall exists.

## Speed — 4x, measured not guessed (2026-09-07)

Profiled first. 400k-voxel world was **630 ms/tick (1.6 ticks/sec)**.

- `np.choose` was **47% of all sim time** — it evaluates and broadcasts every
  branch before selecting. Nine sites replaced with fancy indexing
  (`_ARR[self.mat]`) → 295 ms/tick. Purely mechanical, 2.1x.
- `_law_support`'s slack relaxation was then 45% — a fixpoint iteration over the
  whole grid, every tick, for a field that is a pure function of the material
  layout. Cached against `mat`: comparing costs ~0.2 ms, recomputing ~134 ms, and
  in a standing room it is skipped almost every tick → **158 ms/tick, 6.3/sec**.
  Correct by construction (compares real state) rather than by invalidation.
- Test suite fell 164 s → 88 s as a side effect.

Not yet done, in order of expected value: **active regions** (most of a world is
inert; Powder Toy / Noita dirty-rects — an order of magnitude on big quiet worlds,
and the thing that reaches the >10 runs/sec design target); **kernel fusion** via
numba or Taichi (we are memory-bandwidth bound — ~30 laws each making several full
passes; a CA engine sweeps once and applies many rules per cell, which numpy
cannot express); `heat_capacity()` still rebuilt 12x/tick. GPU (CuPy) is a real
option but not the next lever: state is ~1.6 MB, so kernel-launch overhead would
dominate, and VRAM is budgeted for the language model.

**Activity criterion, agreed** (the "bacteria" question — in reality nothing is
inert, so the threshold is a modelling decision, the same judgement as Ruling 1's
stopping rule): a region is inert when processing it would not change a modelled
quantity by enough for any outcome we care about to notice. Heat is provable (no
gradient, no flux — exactly zero). **Radiation is the trap**: it is not
neighbour-local, so a source must wake everything within its REACH — adjacency for
conduction, line of sight for radiation, earshot for sound. Gas must freeze by
whole connected AIRSPACE, never by chunk, since gas mixes room-wide. Pressure
last. And the guard is a differential test — same scene with and without active
regions, trajectories equal to tolerance — written BEFORE the optimisation.

## The menu seam — BUILT (2026-09-06)

`_law_will` used to notice and choose in one expression. It is now three parts:
**notice** (senses fire a percept — physics, no choosing), **offer**
(`_menu` builds every response this body could carry out *right now*), **pick**
(a POLICY returns an index). Options are intentions, not steps; which door and
which way round the table stays motor competence (`_plan_path`).

- **Legality is checked, not assumed.** A `flee` row is offered only if a BFS
  route to a door really exists, so a picked option cannot fail to execute.
  Wall the room and the menu is `["stay"]` — the mind is never shown a door
  that isn't there. This is what makes illegal action *impossible* rather than
  discouraged, and it is the same bet as core D8/D2, one layer down.
- **The policy contract is one method**: `pick(situation, menu) -> int`.
  `TablePolicy` reads REFLEXES and reproduces the old behaviour exactly (all
  40 voxel tests unchanged). A model-backed policy drops into the same slot;
  reactive play stays at zero model calls because the table costs nothing.
- **Every decision logs a trace row**: situation, the *whole menu*, the pick,
  the policy that made it. `trace_outcomes()` stamps how that body ended
  (safe / down / dead / still inside) — the blessing signal. Scenes dump
  `traces.json` beside `speech.json`.
- **`MENU_CAP = 7`** — stated as a model of attention, not a token budget. A
  person in a burning room weighs a handful of things. Trimming keeps whatever
  the body's own reflex table would have picked.
- **`decide_every = 30`** ticks — a body that stood pat re-weighs its options
  later; nobody re-deliberates every fortieth of a second.

Open, in the order they will bite:
- **Menu recall is not measured yet.** The rows pair menu↔pick↔outcome, which
  is the pairing recall needs, but knowing a *better* option was offered and
  passed over needs the counterfactual run. Until then a capped menu can drop
  the right option invisibly — the failure mode to watch, not list length.
- **Exploration slots.** If the picker only ever sees the top of the salience
  ranking, a model distilled from these traces inherits the ranker's blind
  spots permanently. Reserve a slot or two for low-ranked options so the
  harvest contains out-of-rank picks that *worked*.
- **Uncertainty gating.** Don't spend a model call when the top option wins by
  a wide margin; call only when the top few are close. Cuts calls and
  concentrates the harvest on the hard cases.
- **Salience ranking itself** — with four responses nothing needs ranking yet.
  It becomes real the moment responses attach to objects (fight the fire, shut
  the door, drag the fainted one), which needs hands.
- **Menu breadth comes from objects, not verbs.** Follow core `parts.AFFORDS`:
  options ride on the things attention already selected, so the cross product
  is never built.
