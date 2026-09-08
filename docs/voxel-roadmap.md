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

## TODO — open, in the order I would take them (2026-09-07)

Momentum exists now (`TICK_S`, `vfall` + drag, `_launch` free flight, `jump`).
These are what it opened up and what it left behind.

**Momentum, finishing the four cases**

1. **`roll` — a landing spread over TIME.** Force is the momentum change divided
   by how long it takes to make it, so a body that keeps moving and stops over
   many ticks is struck far less hard than one that stops dead. Needs a landing
   that can take more than one tick. Test written: `test_ROLLING_on_landing_...`
2. **`swing` — hands putting force behind a moving mass.** Impulse = force x
   time, so a held thing leaves the hand with speed and arrives as `1/2 m v^2`
   over the short distance an edge takes to stop. `_launch` and the kinetic
   landing already exist; what is missing is the act, and a contact area small
   enough that the same energy lands as a much larger stress. Test written:
   `test_an_axe_SWUNG_bites_...`
3. **Throwing falls out of (2)** — the same impulse with a different aim. Free.
4. **Catching.** A percept for a thing in flight, plus reach. Cheap once bodies
   have trajectories, which they now do.
5. **Walkers cannot change elevation at all** — not a step up, not a stair, not a
   ladder. Unrelated to momentum and cheap; blocks every scene with a level
   change.
6. **Friction heat: `mu N v` into `E`.** `FRICTION` is still only a force
   threshold. Since `E` is the field combustion already reads, rubbing could
   START A FIRE with no new rule. Wants sliding contact between free bodies.
7. **`FALL_SUBSTEPS = 4` caps falls at 8 m/s.** Honest below it, clipped above.
   Raising it costs a support sweep per sub-step; the fix is to sweep only the
   columns that are actually moving.
8. **A flying body that leaves the world loses mass** — `_land_body` clamps
   out-of-bounds cells and several can pile into one. Only reachable by
   launching something out of the lattice, but it is a conservation hole.

**Bodies and identity**

9. **Hauling costs the hauler nothing** — no speed penalty for dragging 28 kg.
   `strength_N` already says by how much it should.
10. **A felled body travels much further than a topple should** (x=16 to a
    centre at x=35 for a body ~24 voxels tall).
11. **Bodies wake exactly where they fell**, with no account of having been
    moved while unconscious.
12. **The humanoid is light** — 28.5 kg for 1.5 m, so it jumps ~1 m and can be
    carried too easily. A stick figure at 5 cm fills less of its outline than a
    person does; the force law is right and the body is thin.

**Minds**

13. **The menu policy work.** No model is wired to `pick` yet. Exploration
    slots, uncertainty gating, and menu RECALL are all unbuilt — recall matters
    most, because a capped menu can drop the right option invisibly.
14. **Belief records geometry only, and nothing forgets.** A body remembers
    where walls are, never that a door was open or that a room was alight.
15. **An "eyes" limb.** The gaze sweep is a fixed cycle; looking is not yet
    something a mind can CHOOSE to do, which it should be once anything has a
    reason to look somewhere in particular.

**World**

16. **Active regions** — most of a room is inert air resweeping itself. The
    design is settled and unbuilt; XCube's sparse voxel hierarchy (VDB, MPL-2.0)
    is the shape to reimplement.
17. **Gas species; flame above fuel; pyrolysis** — items 2, 4, 5 above.
18. **The door still leaks ~450x too little** — an aperture is an orifice, not a
    porous medium.

**Next up:** item 19 below (segments and one driven joint). It closes the axe
test that is already written, and unblocks the crouch, the swing and the fold on
landing at the same time.

19. **Segments and joints — the body is one rigid lump.** No crouch, no leg
    bend, no arm swing: it translates as a block, which is why a jump reads
    wrong even with every voxel where it belongs. Same missing cause as swinging
    an axe (an arm needs a shoulder) and rolling (a body has to fold).

    **It is not animation, and the distinction is sharp here.** Everything in
    this sim reads the lattice — sight, blocking, walking, collision, mass,
    torque, combustion. There is no separate model to keep in sync. So a pose
    written into `mat`/`smass` IS an environmental effect: the limb occupies
    different cells, its mass sits somewhere new, and if it moved fast it
    carries `1/2 m v^2` into whatever it meets. A pose written only into the
    picture would be animation, and we would have to go out of our way to build
    that.

    **The danger is a SCRIPTED pose, not a drawn one.** "Crouch for 5 ticks,
    then extend" authors the timing and the depth — Ruling 1, a case wearing a
    body's clothes, and a heavy person would crouch as fast as a light one.
    Model the muscle TORQUE and the rest falls out: the segment's own inertia
    decides how fast it turns, so a loaded arm swings slower and nobody types a
    duration.

    **We already have exactly one joint, and it is physical.** `_body_pose`
    rotates a set of cells about an arbitrary pivot and `_land_body` writes them
    back and pays the energy. The only thing that makes it a topple rather than
    a limb is that gravity supplies the acceleration and the pivot sits at the
    object's base. Put the pivot inside the body, drive it with a muscle, and
    the same code is a shoulder.

    **The test that settles animation-versus-physics:** a swung arm knocks over
    a candle that a still arm does not touch. If the pose were animation, the
    candle never moves.

    **Scope:** segments as data on the body (`scenes._person` already builds
    legs, torso, arms and head separately — it just forgets which is which),
    one driven joint reusing the topple machinery. NOT a constraint solver;
    that is box3d/Bullet territory on the copy list, and only worth reaching for
    once joints must RESIST each other. Known limit of the cheap version: no
    reaction forces between segments, so an arm pushing something heavier than
    itself will not be properly resisted — the same gap as having no contact
    solver anywhere else.

**Opened by the ledge work (2026-09-08)**

20. **A GRIP SHOULD CARRY LOAD.** Today a body pulled over an edge always falls,
    because a hand is not material and support is relaxed from the ground up
    THROUGH material — so nothing in the lattice can hold a hanging man. An arm
    that can lift 40 kg ought to be able to hold him, and then "he pulled him off
    and was left holding him over the drop" becomes a real outcome instead of a
    foregone one. Needs the grip to be an edge in the support graph. It is also
    what would let two people roped together drag EACH OTHER off, which is the
    version of this question with any drama in it.

21. ~~**A fall should hurt.**~~ **BUILT (2026-09-08)** — see "A fall hurts"
    below.

22. **A flier stops dead on any contact.** `hit > 0` is an arrival, which is
    right for the ground and wrong for the rock a falling body is scraping past.
    What a wall takes is the sideways speed; what stops a fall is something
    underneath. Cheap fix: on a hit, retry the same step with the horizontal part
    removed, and land only if that is blocked too.

23. **Grab an OBJECT, not just a person.** `p["dragging"]` names a person. The
    constraint is the same for a stick; what it needs is for the held thing to be
    identified the way a person is (a claimed cluster) rather than by name. This
    is the missing half of the axe test — see "What 5 cm voxels cost us".

## A filter for 3D research links (2026-09-07)

Assessed on request, and the verdict was the same three times, so it is worth
writing the reason down rather than re-deriving it per link.

| Link | What it is | Use to us |
|---|---|---|
| [XCube](https://research.nvidia.com/labs/toronto-ai/xcube/) | generative 3D scenes on a sparse voxel hierarchy (VDB) | the DATA STRUCTURE, for active regions — not the generation |
| [InfiniCube](https://research.nvidia.com/labs/toronto-ai/infinicube/) | generates large dynamic driving scenes; its own page says geometry and appearance, no physics | none |
| [NVIDIA RTR voxels](https://research.nvidia.com/labs/rtr/tag/voxels/) | three rendering papers (ray-box intersection, GPU voxelization, cone tracing) | none — our render is not the bottleneck |
| [WorldSculpt](https://alaya-lab.github.io/WorldSculpt/) | images + instance masks + 3D boxes into one complete MESH per object, composed rigidly | none directly; see below |

**Why they keep missing.** All of it is about making worlds LOOK right —
generation, reconstruction, rendering. This project's premise is that a world
BEHAVES right, and behaviour comes from conserved quantities under laws, which
none of these produce. A mesh has a shape and no causes: no density, no ignition
point, no toughness, no fill fraction. Importing one would give us silhouettes we
would then have to assign materials to, and assigning the materials IS the work.

WorldSculpt grazes a real gap — its premise is that a scene is individually
complete OBJECTS rather than one soup, and `_object_at` floods by material
adjacency, so a table with iron legs is two things and two touching crates are
one. But it is GIVEN the instance masks; its contribution is completing what the
camera could not see. We have no images, and we need the separation for torque
and grabbing, not for viewing.

**Where research does help, narrowly:**

1. **Data structures for speed** — sparse voxel hierarchies (active regions).
2. **Solvers** — box3d (MIT) and Bullet (zlib), already on the copy list, for
   joints and contacts.
3. **Articulated-object priors** — the live one, given item 19. Scene
   reconstruction is the static version of this and does not carry joints; the
   thing to look at is work that predicts or catalogues ARTICULATION. SAPIEN's
   PartNet-Mobility is the first to check, because it carries actual joint
   definitions rather than geometry alone. **Unverified** — confirm what it
   covers and under what license before leaning on it.

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

## Hands, as one force — PART BUILT (2026-09-07)

Not five verbs. `_effort(cells)` returns what a thing costs to LIFT (its weight)
and to SHOVE (friction × weight); `BODY["strength_N"] = 400` (an ordinary adult,
~40 kg up, ~100 kg along) decides which are possible. `_object_at` is the thing
you grabbed — whatever is connected to it, same material. `_shove` translates it
with everything it holds. Push/pull/press/drag stop being separate rules.
Measured: an iron anvil (62 kg) cannot be lifted and can be slid; a wicker
basket goes straight up.

**Object identity, forced by this and now built.** Bodies were identified by
adjacency, so two people who touched became one two-headed person — measured,
both resolving to the same 684 cells with both anchors converging. Identity now
persists through TIME: each body is claimed once a tick, seeded from the cells
it owned last tick, barred from flesh another body owned. Two people stand
shoulder to shoulder and stay two people. (Small known bleed: a ~7% overlap at
the contact face from the one-tick lag.)

**And out means out.** A body that reached the door now leaves the lattice, its
mass accounted as having left. Bodies have real depth now, so one person standing
in a doorway they had already escaped through was a wall to everyone behind them
(measured: a second body stuck 3 voxels short for 800 ticks). Two related bugs
found with it: a person who had left still ran an anchor-flood and claimed a
COLLEAGUE'S body from 35 voxels away, freezing them; and the footprint erosion
ran before "own columns count", so a body's own flesh made its own column
unwalkable and it could not stand where it stood.

**The humanoid got a body.** Built one voxel thin it weighed 10 kg — a person you
could pick up like a cat, which quietly voided every question about lifting and
carrying. With a real chest and limbs it measures 28.5 kg. Still light (about a
ten-year-old) because a stick figure at 5 cm fills far less of its outline than a
person does; stated as a known softness, not fudged. Depth went into y, not x, so
every doorway in every scene is unchanged.

**Dragging someone out — WORKS (2026-09-07).** A rescuer takes hold (legal only
because the force arithmetic says this body can shift that one) and hauls them to
the door with the same `_shove` that slides a crate. It is a CHOICE: the same
world with an ordinary character sheet leaves the body where it lies, and the
pick goes through the menu and into the traces like any other. Locked by test.

The blocker was subtle and worth remembering: a fainting body is promoted to a
FREE body while it keels over, so for a few ticks it is off the lattice or only
half back on it (measured: 342 cells, then 0, then 42). Grabbing at it then fails
— and letting go on that failure ended every rescue at exactly the moment the
person being rescued finished falling over. A grip is not lost because someone
moved; `_haul` now waits out the fall.

Still soft: hauling a body costs the hauler nothing — same walking speed, no
extra effort. Dragging 28 kg should slow you down, and the strength number is
already there to say by how much.

## Impact damages what it lands on (2026-09-07)

`_cash_impacts` tested only the FALLING voxel against its own toughness, so an
anvil could hurt itself but never the glass table it went through. The struck
cell is now tested against ITS toughness with the same energy — Newton's third
law, the impulse is shared. Measured: an iron block from z=40 sinks through a
glass plate and is held by a plank or a slab.

Fixed alongside, and it was the real blocker: cells shattering in the same pass
were scattering fragments **into each other's just-emptied cells**, refilling
the hole as fast as it was made, so a shattered plate stayed a plate.
`_just_shattered` excludes them for the pass — fragments fall out of a break,
they do not queue up to plug it.

Still short of the truth: the anvil stops one voxel down rather than reaching
the floor, because the fragments land beneath and become a new bearing layer.
Rubble should not carry like a slab; mass-scaled span helps but shards keep
enough. Wants free bodies (#3) to fall properly.

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

## Actions are a grammar, not a list — BUILT (2026-09-07)

The response table had `flee`, `flee_shouting` and `flee_answering` in it. That is
ONE intention (leave) crossed with three uses of a mouth, and the cross-product is
the thing that grows: add "carrying a lamp" and every row splits again. It is
Ruling 2 in a new costume — a table hiding cases — and the fix is the same shape.
Decompose by the BODY, not by the situation.

**What replaced it.** A body has parts: `LIMBS = ("legs", "hands", "mouth")`. Each
part gets its own menu, each menu is checked for legality on its own, and a PICK is
one option per part. The result is a program — `go(the door east) & take hold of
Fallen` — composed rather than looked up.

| was | is |
|---|---|
| `flee` | legs `go(the door …)` |
| `flee_shouting` | legs `go(the door …)` & mouth `say(fire)` |
| `flee_answering` | legs `go(the door …)` & mouth `say(coming)` |
| `drag_them_out` | hands `take hold of X` & legs `go(the door …)` |
| `explore` | legs `go(the unseen ground north)` |
| `wander` | legs `go(the floor south-west)` |

Seven compound responses became four verbs. Eight, three and two things to do with
the three parts is 48 combinations and 13 menu rows, and the 13 say everything the
48 do. Nothing has to be written twice when a new use of a mouth arrives, and
combinations nobody wrote now happen: holding someone while walking DEEPER in,
calling a warning while wandering, taking hold with no idea where the door is.

**Places are named, and that was a decision being taken away from whoever picks.**
`explore` sent a body to a frontier the SIM chose. Now the unseen edges are listed
one per compass direction, so "go and look at the dark end north" is a different
choice from "go and look west", and the mind makes it. Direction is a place too:
`straight on`, `to the left`, `to the right`, `back the way it came`, each reaching
as far as the body could actually get that way (`WILL["step_m"]`), so a body that
knows nowhere by name can still have an intention. The legs still do the walking —
choosing a way to go is not the same as being asked to place your feet.

**One BFS prices every place.** `_routes` sweeps once and reads each route out of
the parent map; `_plan_path` is now a one-target call to it. Cheaper than the three
separate searches it replaced, for four times the places.

**A reflex row is now per-limb**, which the old sheet could not express at all:
`{"sees_fire": {"legs": "go:exit", "mouth": "say:fire"}}`. A rescuer's sheet says
`{"legs": "go:exit", "hands": "hold"}` — one row, two parts of a body, and no
`drag_them_out` anywhere in the sim.

**Bug this surfaced: a grip was telekinesis.** `_haul` shoved the held body along
the HAULER's step vector, which is a grip only while the two happen to move alike.
Once the load snagged for a tick the pair drifted apart, still joined — measured, a
rescuer at x=29 towing a body at x=36, seven voxels AHEAD and pulling it further.
A load trails what pulls it: the step is now toward the hauler, and an arm
stretched past `BODY["reach_m"]` is not holding anything. `_in_reach` measures
between SURFACES (`_span_xy`), not centres, so the same arm is the same arm whether
it reaches for a child or a cart — and the hardcoded 6-voxel reach died with it.

## Matter has no momentum — the one gap under six wanted cases (2026-09-07)

Asked for: a jump; catching something in the air; jumping off a cliff and ROLLING
so it doesn't kill you; a parachute; air catching in things; two metals rubbed
together getting hot. Six requests, ONE missing quantity.

**What the sim carries today.** Gas has velocity — `vcell` holds coarse pressure
and a velocity field on 4-voxel cells, and the wind carries smoke. **Solids do
not.** A falling voxel accumulates `fallh`, a count of voxels dropped, and at rest
cashes `m·g·h` against toughness in ONE tick. That is energy bookkeeping standing
in for a velocity, and it is why every one of the six is impossible:

| wanted | why it can't happen |
|---|---|
| jump | no vertical impulse — and `_walk` cannot change elevation AT ALL, not even a step up |
| catch a falling thing | nothing to predict: a free body has a position, not a trajectory. No percept for it either |
| roll on landing | the whole `m·g·h` cashes in one tick; rolling IS spreading it over a longer stop |
| parachute | drag needs `v`. The sim already knows the cross-section — the shape is countable — so this is the one that would nearly fall out |
| air catching in something | gas velocity never pushes on solid mass; the coupling is one-way |
| friction heat | `FRICTION` is a force threshold only. Heating is `µ·N·v` — a POWER, so it needs a sliding speed |

Ruling 1's stopping rule: *model a quantity explicitly at the moment two cases the
user wants to be different would otherwise be the same.* Founding case 1 parked
speed ("tipping-by-walking wants SPEED, which the grid does not carry"). Six named
cases now need it. It is earned.

**Scoping, so it stays computable.** Velocity does NOT go on every voxel — that is
a full solid solver and the fluid field is already coarse for exactly this reason.
It goes on the OBJECT: `self.bodies` (free bodies mid-fall) already exist, are few,
and now have identity through time. A falling thing gets one velocity, not 342.

Proposed order, each step a cause with several consequences:
1. `v` on free bodies; `fallh` becomes `v = √(2gh)` and stops being a special case.
2. Landing over TIME (stop distance) → rolling; upward impulse → jumping. Needs
   `_walk` to handle elevation first, which is worth doing on its own.
3. Drag against the existing gas velocity, using the cross-section the voxel shape
   already gives → parachutes, wind catching a cloak, terminal velocity.
4. `µ·N·v` into `E` at sliding contacts → rubbed metal heats, and since `E` is the
   field combustion already reads, friction can START A FIRE with no new rule.
5. Catching: a percept for a moving thing, plus reach. Cheap once (1) exists.

## The rescue test was green for the wrong reason (found 2026-09-07)

Reported on 2026-09-07 as working. It was not. Probing the committed version
(68f23ac) with the same scenario the test uses:

```
t15  Hero (26.0, 12.0, 342 cells) | Fallen (25.0, 12.0,  42 cells)
t30  Hero (31.0, 12.0, 342 cells) | Fallen (19.0, 12.0,   3 cells)
t60  Hero (41.0, 12.0, 342 cells) | Fallen (27.0, 12.0,   3 cells)
HEAD: hero safe True, down safe True
```

The rescuer hauled a **three-voxel fragment** to the door out of a 342-voxel
body, and `down["safe"] = True` was set because the hauler arrived while
`dragging` still held a NAME. The test asserted the flag. The body never moved.

**Root cause, and it is not the rescue code.** A body that faints is promoted to
a free body, so its flesh LEAVES the lattice for a few ticks. `_person_cells`
then finds nothing to grow from and falls back to "nearest flesh to the anchor" —
which is the person standing over it. The fainting body was re-acquired out of
its rescuer: 42 cells of the hero, then whittled to 3.

**Five fixes. It works now, and the test counts BODIES rather than reading a flag**
(`not (w.mat == FLESH).any()` — both people left the lattice; in the world with an
ordinary character sheet, 300+ voxels of person are still lying on the floor).

1. *A grip is not telekinesis.* `_haul` shoved the load along the HAULER's step
   vector, so once the load snagged the pair drifted apart still joined —
   measured, a rescuer at x=29 towing a body at x=36, seven voxels AHEAD of them.
   A load now trails what pulls it, and an arm stretched past `BODY["reach_m"]`
   is holding nothing. `_in_reach` measures between SURFACES, killing a
   hardcoded 6-voxel constant.
2. *Nobody claims before anybody else.* Claiming one body after another lets
   whoever runs first grow into a neighbour's NEW cells — barring last tick's
   cells cannot help, because a body that stepped is standing somewhere the
   barrier does not cover. Measured: 684 cells of two people split 497/187, and
   the one left with half a body could no longer walk. Every body is now seeded
   with what it owned and all seeds grow ONE VOXEL A ROUND TOGETHER; contested
   flesh goes to whoever was nearer last tick. A body with no seed waits and may
   adopt only what the grow left spare, so nobody is re-acquired out of anyone.

3. *A body with no history seeds on its ANCHOR.* Two people who start in contact
   are one lump with nothing to divide it, so flooding them one at a time gave
   the whole lump to whoever went first and NOTHING to the second — measured, a
   rescuer who could not act because they did not exist. Both now seed and the
   grow settles it by who was nearer, the same rule as every other contested
   voxel.
4. *The sim knows where it put the body.* `_topple` promotes a fainting person's
   flesh off the lattice and `_land_body` writes it back down — so the sim
   HOLDS the answer identity was busy re-deriving from adjacency. A free body
   now carries `owner`, and landing hands those exact cells back. Re-deriving
   what you already know is what let a fainting person be assembled out of
   their rescuer.
5. *The head is not the feet.* A gaze swept by turning `facing`, and only while
   a body had nowhere to be. Once idling became a real choice bodies always had
   somewhere to go, and every step reset the one vector doing both jobs: a
   witness at a clear window onto a burning bench walked about in front of it
   for 260 ticks and never perceived the fire. `p["gaze"]` now sweeps a whole
   turn in eighths around whichever way the body is pointed.

**And one semantic bug the grammar exposed.** `fleeing` was set by every `go`,
which was harmless while the only reason to walk was an alarm. With idling a real
choice, a person strolling to the window read to everyone else as someone BOLTING
PAST — `sees_runner` fired all over the house with no fire behind it, and the
rescuer's whole decision was spent on a false alarm before the real one arrived.
`fleeing` now means DRIVEN, not merely walking.

Measured, end to end: `rescuer=False: hero out True, fallen out False` /
`rescuer=True: hero out True, fallen out True`, carrying 342 cells. 278 tests pass.

## Momentum — the four cases, two of them BUILT (2026-09-07)

Four tests were written first, as the specification: a person jumps; a person
falls off a ledge and rolls; a wide thing falls slower than a compact one; an
axe swung bites where the same axe pressed does not. Two pass. Two are
`xfail(strict)`, so they shout the day they work.

**A clock, at last.** `TICK_S = 0.025` was never written down and two parts of
the sim assumed different answers — the will layer says "nobody re-deliberates
every fortieth of a second" while the fall law dropped everything exactly one
voxel a tick, which at 5 cm is a flat 2 m/s for a feather and an anvil alike.
An acceleration cannot mean anything without it.

**Falling has a speed.** `vfall` (m/s, per column) accelerates under gravity and
is resisted by drag: half rho C_d A v-squared against m g. The frontal area is
one voxel face however DEEP the column is, so the same matter spread thin falls
slower than the same matter balled up — measured, 25x25x1 against 5x5x25 of
identical mass:

| fill | spread | balled | ratio |
|---|---|---|---|
| 0.10 | 46 ticks | 40 ticks | 1.15 |
| 0.03 | 63 ticks | 41 ticks | 1.54 |
| 0.01 | 98 ticks | 43 ticks | 2.28 |

That is a parachute, with nothing about parachutes in it. It only appears once
the one-voxel-per-tick cap is gone: terminal velocity for a light sheet is ~2.1
m/s and the cap was 2.0, so the effect was being clipped away entirely.
`FALL_SUBSTEPS = 4` lets a fast column take several sweeps within a tick — the
support question has to be re-asked between every cell, or matter skips over
what it would have landed on.

**Bodies can leave the ground.** `_launch` promotes a cluster to a free body in
FLIGHT — the same promote/demote protocol as a topple, travelling rather than
turning. It carries m/s, is pulled on by gravity, is pushed back on by air
(`_drag_a` COUNTS the frontal cells rather than being told an area), and pays
`1/2 m v^2` where it arrives, which is the difference between a stone dropped on
a plank and the same stone thrown at it. `jump` is a legs act, legal only with
something underfoot, and the height is a consequence: `legs_N = 1400` over a
25 cm crouch against the body's own weight. Legs are not arms — reusing
`strength_N` gives a push barely above body weight, which is not a jump.

**Two bugs the change exposed, both older than it.**

- `_settle` cashed impacts itself and returned the result, so the caller's
  `moved` mask arrived as None and every falling voxel read as landed.
- "Did not move this tick" was a safe reading of "has come to rest" only while
  everything unsupported moved every tick. A thing slower than a voxel a tick
  pauses between steps, and cashing its drop then SHATTERED IT IN MID-AIR:
  measured, a glass block bursting at z=8 on the way down to a pond it never
  reached. Impact now asks whether the thing is still airborne.

**Five existing tests had tick budgets that encoded the old constant fall** (8
ticks for a drop that now takes ~25). Their claims are unchanged — unsupported
things fall, supported things do not — only the time given.

**Still to come:** `roll` (a landing spread over TIME rather than taken whole —
force is the momentum change divided by how long it takes to make it), and
`swing` (hands putting the body's force behind a moving mass, arriving as
kinetic energy over the short distance an edge takes to stop). Both now have
the field they were missing.

## Renders of the new capabilities (2026-09-07)

Two clips, both at 640x360, in `renders/`. Neither could have been made a day
earlier — not because nothing would draw, but because nothing would MOVE that
way.

- **`renders/jumper/`** — a person jumps, repeatedly, and a crate falls beside
  them. Two things to watch: the body leaves the ground at all (legs put 1400 N
  into the floor over a crouch; what is left after holding its own weight up
  becomes speed), and the crate does not descend at a stately voxel a tick — it
  starts from rest and picks up speed. The jump goes through the ordinary menu:
  a `_Wants` policy pins the taste so the clip is about physics, but `jump` is
  still only OFFERED to a body with something underfoot.
- **`renders/parachute/`** — a sheet and a wad of the same matter, the same
  mass, dropped together. Measured in the run: **the wad is down at t43, the
  sheet at t89.** By still 18 the wad is already standing on the floor and the
  sheet is barely a third of the way. Nothing in the scene mentions parachutes.

**Known rough edge:** the renderer's camera looks at `nz * 0.30`, which is low
for a world built tall to drop things through — the first few frames of the
parachute clip are an empty floor while the objects are still above the top of
frame. Framing the actual CONTENT rather than the lattice would fix it for every
tall scene at once.

## Wide vs wider, and a jump that goes somewhere (2026-09-07)

**The sheet pair, which is a sharper test than the first one.** Drag was proved
by dropping a sheet against a wad. That only shows shape matters. The pair that
shows we modelled the RIGHT thing holds one variable at a time:

| | 8x8 sheet | 20x20 sheet | |
|---|---|---|---|
| same thickness (frac 0.02) | 73 ticks, 96 g | **73 ticks**, 600 g | 6x the mass, identical speed |
| same load (240 g) | 54 ticks | **109 ticks** | spread thinner, twice as slow |

The first row is the invariant: terminal speed balances drag against weight and
BOTH grow with area, so mass per unit area is what decides, not size. An
implementation that read drag off a footprint would make the bigger sheet
slower and would still have passed the wide-versus-balled test convincingly.
The second row is why a canopy works: hold the load fixed, spread it wider,
come down slower. Nobody told the sim either rule.

**The jump now goes somewhere.** It launched straight up and landed where it
started, which is barely a jump — the point of having a velocity is that it has
a direction. Aimed at a place, the body leaves at 45 degrees (the angle that
carries furthest) at exactly the speed that distance needs, `v^2 = g d`, so it
lands where it meant to instead of hurling itself as hard as it can. Anywhere
further than the legs allow is not offered.

**Two lies this uncovered, both about where a body may stand:**

- `_walkable` never asked whether there was a FLOOR under a column, only whether
  the column itself was clear. A chasm read as walkable, so the planner would
  march a body straight out over one and the legs would carry it — walking on
  nothing. Fixing it is also what makes a gap an obstacle, and so what gives
  jumping across one a point.
- `_fit_grid` had its own copy of that rule, so the fix reached the PLACES and
  not the PLANNER. One rule now, in one place.

**And a menu lesson.** Offering four leap directions unconditionally crowded
`MENU_CAP` and starved the walking options — the ordinary person in the rescue
scenario stopped getting out of the burning room, because their attention was
full of jumps. A leap is only offered to a spot no route reaches. An option that
is a worse way of doing something already on the menu is not a choice, it is
noise, and the cap is a model of attention rather than a budget.

Clip: `renders/leap/` — a person crosses a 50 cm chasm they cannot walk over.
Measured flight: launch at t0 with v = (2.57, 0, 2.57) m/s, a clean parabola,
landing at x=37 on the far ledge at t22. Then it leaps BACK, because once it is
across, the only landing spot no route reaches is the one it came from.

## The jumper's face — a real bug, and a real limitation (2026-09-07)

Two separate complaints about the same clip, and only one of them is a bug.

**The bug: landing DISFIGURED the body.** `_land_body` writes a body's cells
back onto the lattice and pushes any blocked cell upward as debris, which is
right for a toppling tree arriving in rubble. A flier inherited the topple's
tolerance — a few grazing cells are not a landing — so a jumper came down
already driven several voxels INTO the floor, and those buried cells got shoved
up through the body. What is at the top of a person is their head. Measured: 15
of 342 voxels rearranged, every gram still present.

Any contact is now an arrival for a flier: it stops at the last clear pose and
the support law sets it down the rest of the way (free fall as lattice matter
was verified shape-preserving — 342 cells, zero out of place). The jump test
now checks SHAPE and not only mass, counting matter in flight as well as matter
on the grid, because a jumper spends most of a jump off the lattice and mass
alone never noticed the face being rearranged.

This also closes the conservation hole where a flier leaving the world lost
mass: out-of-bounds now counts as contact, so nothing flies off the edge.

**The limitation, which is NOT fixed: the body is one rigid lump.** No crouch,
no leg bend, no arm swing — it translates as a block, and that is why the jump
looks wrong even when every voxel is where it should be. That is not a jump
problem. It is the same missing cause as swinging an axe (an arm needs a
shoulder) and rolling on landing (a body has to fold): **there are no joints**,
which `_object_at` has admitted from the start — "a table with iron legs is two
things: joints do not exist yet."

Faking it by squashing the cells on take-off would be animation, and Ruling 1
says model the cause. `scenes._person` already builds the body as separate
limbs; making those segments real, with joints between them, is the piece that
unblocks jumping, swinging and rolling at once. It is the next big thing.

## Segments and a driven joint — BUILT (2026-09-07)

Item 19. The body is no longer one rigid lump: it has named parts, and one of
them can move by itself.

**A body knows what it is made of.** `scenes._person` already built legs, torso,
arms and head separately and then forgot which was which. It now records each
piece as it builds it, and keeps them as OFFSETS from the body's own corner, so
segments ride along when it walks, is shoved, or is carried out. The sim could
not work this out for itself — every voxel is just FLESH and nothing in the
lattice says where a shoulder is — but the scene that built the body knows
exactly, and how a thing was built is the object's own business to declare
(Ruling 2, question 1).

**A joint is the topple, moved inside.** `_body_pose` has always turned a set of
cells about an arbitrary pivot; `_land_body` has always written them back and
paid the energy. The only things that made it a TOPPLE rather than a LIMB were
that gravity supplied the acceleration and the pivot sat at the object's base.
`_swing` moves the pivot to the shoulder and hands the acceleration to a muscle:
`alpha = torque / I`, where I is the limb's own `sum(m r^2)`. `BODY["arm_Nm"] =
60` is the only number, and it is a shoulder's torque, not a duration — the
limb's inertia settles how fast it comes round, so a heavier arm is slower and a
loaded one slower still, with nobody writing down how long a swing takes.

**The blow lands on WHAT STOPPED IT.** A falling thing is judged by the floor it
meets; a swung thing has to be judged by the thing it hit, or leaning on an axe
and swinging one are the same event. `_land_body` now takes the struck cells and
spends `1/2 I omega^2` across them, each against its own toughness.

**Measured, and it is the test that settles animation versus physics:**

```
hands = keep : 0 broken voxels, 2175 g
hands = swing: 4 broken voxels, 2175 g
```

Same body, same post, same everything else; one of them swings. A pose is
animation if it only changes the picture — and there is no picture here. Every
law reads the lattice, so while the arm comes round it really is somewhere else:
it occupies what it now fills, its mass sits where it now sits, and the post is
broken by the rotational energy the arm was carrying. Mass is conserved through
it; breaking is not losing. Swinging goes through the menu as a hands act like
any other, and swinging at NOTHING is legal — useless, but legal, the same way
shouting in an empty house is.

Clip: `renders/swing/`.

**Known limits, none of them hidden:**

- **No return to rest.** A swung arm stays where it stopped, because once it is
  back on the lattice it is ordinary attached flesh and the support law holds it
  there. Its segment offsets are updated on landing so the NEXT swing still
  finds it, but a body that swings repeatedly windmills.
- **No reaction between segments.** An arm pushing something heavier than itself
  is not resisted properly — the same gap as having no contact solver anywhere.
  That is where box3d (MIT) or Bullet (zlib) would come in, and only then.
- **Still no crouch, and so still no natural-looking jump.** Legs have joints
  now in the same sense arms do, but nothing drives them; a jump is still a
  whole-body launch. `_swing` is general — it takes any named segment — so the
  work left is which torques a jump applies, not new machinery.
- **The axe test remains xfail.** It needs a body to HOLD an object, which is
  hands-on-things rather than hands-on-people, and does not exist yet. The swing
  itself is done and proven.

### The swing tore the arm off (found and fixed same day)

Caught by eye in `renders/swing/`, not by a test: the hand came apart along with
the post. It was not shattering — flesh stayed at 342 cells and 38138 g with
ZERO broken voxels throughout. Counting connected lumps found it:

```
before the swing: flesh in 1 piece  [315]
after  the swing: flesh in 3 pieces [324, 12, 6]
```

Eighteen voxels of arm, detached and lying on the floor. **A limb was being
rasterised wherever the swing happened to stop**, and at the end of a sweep the
far end of an arm is nowhere near the body it belongs to. A lump of flesh that
touches nothing is not an arm.

**A limb is held at its joint.** Hanging IS the rest state of something pivoted
at one end, so the arm returns to it once the blow is spent — the swing is a
transient, and what it is attached to is not. That also retires the "no return
to rest" limitation and the windmilling that came with it.

**And the blow is no longer paid twice.** A swing was delivering its full energy
to the struck cells AND then paying the same energy again into its own contact
cells, as if the arm had fallen on something. An arm does take the impulse back
— Newton's third — but that is not the same as double counting, so the self-hit
is dropped until there is a reason to model it properly.

Now measured: **one piece throughout**, and repeated swings chew further in —
4 broken post voxels after the first, 8 after the second. The test asserts the
piece count, since nothing was checking that a person who swings is still one
person afterwards.

### A bare fist should not splinter a fence post (2026-09-07)

It did. Cause: the muscle made the same torque however fast the arm was already
going, so it accelerated for the whole sweep and a LIGHT arm — ours is light,
see `scenes._person` — reached about 15 m/s at the knuckle, roughly twice a real
punch and so four times the energy.

**Hill's force-velocity relation**: the faster a muscle is already shortening,
the less force it makes, fading to nothing at a top speed. One constant
(`BODY["arm_wmax"] = 15` rad/s, a shoulder unloaded), and it is a cause rather
than a speed limit typed in — the same relation is why an arm carrying something
heavy swings slow. Measured after: **peak knuckle ~3.8 m/s, post undamaged.**

**The test got better for it.** It used to assert only that a swung arm marks
what a still one does not, which motion alone explains. Now it discriminates on
MATERIAL: the same fist at the same speed shatters glass (0.2 kJ/m2) and leaves
a wooden post (8.0) untouched, and breaks neither itself nor the arm. Two
answers from one blow, and nobody wrote either down. Clip: `renders/swing/`,
where the pane goes from 32 to 55 broken voxels over three swings while the post
stays at zero.

**Also dropped: the blow was being paid twice** — in full to the struck cells
AND again into the arm's own contact cells, as if the arm had fallen on
something. An arm does take the impulse back (Newton's third), but that is not
the same as double counting.

## What 5 cm voxels cost us, and what they do not (2026-09-07)

Raised as: "is it possible to grab the stick? the voxel is costing us dexterity."

**It does, and the line is worth drawing precisely.** At 5 cm a hand is one or
two voxels and there are no fingers, so a grip cannot be modelled as GEOMETRY —
there is nothing to close around anything.

**But a grip does not have to be geometry.** It is a constraint: what is held
moves with the hand. The sim already does exactly this for dragging a person
(`p["dragging"]` plus `_haul`, reach-limited). Extending it from people to
objects is small, and what decides whether a grip is possible is force
(`_effort`, which exists) and size (can a hand close on it) — both real
questions the lattice can answer. So "grab the stick" is buildable now, and it
is the missing half of the axe test.

**What the resolution genuinely costs.** Anything where SUB-VOXEL shape decides
the outcome: a key in a lock, a knot, where the fingers go — and **an edge**. An
axe bites because its edge is sub-millimetre, and at 5 cm we cannot represent
that at all. An earlier note here claimed sharpness came out of the contact
being "a few voxels"; that is weaker than it sounded, because a 5 cm contact is
not sharp by any measure.

Three ways out, and the first is the one to take:

1. **Declare the contact area as a property of the OBJECT.** "This edge
   concentrates its blow into X mm2" is a fact about a manufactured thing, the
   same class as its density — legitimate under Ruling 2 question 1, and needed
   anyway before an axe can differ from a club of the same mass.
2. Finer voxels near hands only (multi-resolution). Expensive, and it buys
   little the constraint model does not.
3. Finer voxels everywhere. Costs everything, everywhere, for one problem.

## Can a man try to pull another off a cliff? — BUILT (2026-09-08)

Asked as: "alright can a man try to pull another off a cliff?"

**No, and for one line's worth of reason.** Every physical piece was already
here — take hold, walk, the load trails on the same `_shove` a crate uses, a
falling body, energy paid on landing. What was missing was permission:
`_within_reach` skipped anyone `q["awake"]`, so a grip was legal ONLY on someone
already unconscious. That is a case wearing a flag (Ruling 1). It made rescue the
only reason two people ever touched each other in this sim, and it made the
question not hard but UNASKABLE: the row was never on the menu to pick.

**The cause that replaces it.** A body that is awake, alive and has something
under its feet BRACES — it puts its own strength into the floor against you:

    resist = friction(their mass)  +  their strength, if they are braced

One line, and three stories fall out with none of them written down: an ordinary
man cannot shift an equal (112 N of friction + 400 N of brace against 400 N of
pull), a stronger man shifts the same equal, and an ordinary man shifts someone
weaker. The unconscious case every rescue in the suite depends on is the SAME
line with the brace at zero, which is why it never needed its own rule either.

**And the force test moved.** It used to be answered once, at the moment of
grabbing, inside the menu — which says a body knows before touching you whether
you can be moved. It is asked every tick in `_haul` now, because the answer
changes: the same pull a braced man shrugs off for twenty ticks succeeds the
moment his heels leave the floor.

Scene `ledge`, test `test_a_BRACED_man_cannot_be_PULLED_off_a_LEDGE_...`: two men
on a ledge 1.5 m up, two below with hold of an ankle each, walking away. 700 N
takes his man over. 400 N does not move his, and the arm doing the pulling runs
out of length and lets go.

### Four bugs it took the whole way down to find

None of these were about pulling. All four were sprung by one person being
airborne while another stood nearby, which no scene had done before.

1. **A person fell apart.** Support is relaxed column by column — right for a
   wall, wrong for a man: walked off his own ledge he came down as FIVE separate
   showers of flesh, every gram present and the person gone. A jumper never did
   that, because a leap PROMOTES the body to a free rigid object first, and that
   promotion was missing everywhere else. New `_law_footing`, before the support
   law: a body whose feet have nothing under them leaves as one body, and pays
   the same 1/2 m v² on arrival. Not a rule about ledges — a shove, a haul, or a
   floor burning out from underneath all reach the same test.

2. **The torque law tipped people over for standing still, and stole their
   names.** `_law_torque` promoted with no owner, so a toppling person came back
   as anonymous meat and whoever stood nearest inherited their flesh — measured,
   one man reduced to 22 of his 342 voxels. Two fixes: a topple carries its
   owner, and a person on their FEET is not tipped by that law at all. Balance
   is what the living do and the dead do not; the sim already has the other half
   (`_collapse` for a body gone slack, `_law_footing` for one with no footing).

3. **A man in mid-air seeded his identity on a bystander.** `_claim_bodies` falls
   back to "nearest unclaimed flesh" for a body with no history — and a body
   promoted whole to a free body has NO flesh on the lattice for those ticks, so
   the nearest unclaimed flesh is the person standing over him. Measured: while
   one man fell, the man who pulled him lost 21 voxels to him and then read as
   standing on a footprint no longer under his own weight. Fixed: someone whose
   whole body is aloft gets no claim at all. A LIMB in flight is different — a
   body mid-swing is still standing there — so `part` bodies are exempt.

4. **A man dropped and re-caught, three times a second, for ever.** Tipped off a
   lip with a forearm still over the stone, he fell one voxel, struck the very
   rock his arm had been on, and was set back down where he started. Going over
   an edge is a ROTATION about that edge, so a body leaves it moving outward as
   well as down, and by the time it has turned enough to clear the lip its weight
   has fallen through about the overhang: `v = sqrt(2 g o)`, no constant typed in.

**A softness worth naming:** reach is measured across the floor and ignores
height, so a man on the ground can take an ankle 1.5 m above him. At that height
it is about right. At five metres it would be nonsense.

**Also fixed, in passing:** `_law_will` returned unless the world registered an
exit, which made every mind in the sim conditional on there being somewhere to
escape to. Two men on a clifftop could not decide anything at all, and nothing
physical was stopping them. Escaping is one thing a body might want, not the
reason it has a will.

**Not rendered.** Frames generate fine; Blender is not on PATH in this session.

## A fall hurts — BUILT (2026-09-08)

Item 21. `_land_body` paid 1/2 m v² against what a body STRUCK and nothing
against the body, so a man dropped onto stone landed whole and entirely
unbothered, and "pulled off a cliff" was a change of address.

**The wound is the energy the tissue absorbed.** Tissue takes damage far below
the toughness that tears it apart: `TOUGH[FLESH]` is gross failure,
`BODY["bruise_kJm2"]` is where bruising and breakage begin, per contact area.
Landing energy beyond it goes into `p["hurt"]` — one integral, the same shape
as burns, and the two read against ONE pair of faint/death thresholds as
combined tissue damage (`dmg = burn + hurt`). No new thresholds; wounds never
come back down, the same one-way street as burns. The struck side pays too:
flesh under a landing body, or in the way of a swung fist, takes the same
overage through `_person_at` ownership.

**The bug underneath, and it inverted the scale.** At high energy the landing
arithmetic SHATTERED flesh — a man dropped 6 m had his feet scatter as debris
and the rest of him walked away with hurt=0.000, because the shatter branch ate
exactly the energy the wound should have carried. The harder the landing, the
less it hurt (measured: 3 m → 0.295, 4.5 m → 0.129, 6 m → 0.000). The TOUGH
table's own comment says flesh "deforms, not fragments"; the big number was the
mechanism and it failed at exactly the energies that matter. Flesh now never
takes the shatter branch — the whole overage is the person's wound.

Measured after, rigid landings onto stone: 0.5 m nothing, 1.5 m bruised
(hurt 0.08, walks away), 3 m knocked out, 4.5 m dead, 6 m dead — monotone, and
every gram conserved. Calibration is stated in BODY, scaled to this body's
known lightness.

The `roll` xfail was rewritten against the wound model — its old assertions
were written in grams of shattered flesh, which flesh deforming can never
produce, so the spec could never have flipped. It now asserts the thing rolling
is FOR: the same 3 m fall, rigid, knocks a body out; rolled, it stays conscious
and pays less.
