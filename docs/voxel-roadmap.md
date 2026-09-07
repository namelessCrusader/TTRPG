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
| A 90%-burned trunk voxel carries like solid timber, then flips to ash | strength is by material ID, not remaining MASS — should fade: full span at ≥50% mass, decaying to ash-like below (gate at 50% so scene `frac` fills keep their strength); with it, a burning tree eventually falls onto its own fire | mass-scaled strength |
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
