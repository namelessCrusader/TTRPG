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

## Stolen designs (OpenWorld — closed source, ideas only) (2026-10-29)

Evidence and detail live in three notes:
- `docs/openworld-reference.md`: gameplay, from their public wiki;
- `docs/openworld-engine-notes.md`: simulation, storage, AI tiers;
- `docs/openworld-rendering-and-craft.md`: rendering, and how their code is written.

**The filter: nothing on this list hand-codes an interaction.** Each item is one of two kinds:
- plumbing (timing, budgets, storage, rendering) that no scene's outcome depends on;
- the SAME law run cheaper (a closed form of L1, an L11 ledger), with its parameters taken from the cells it summarises.

Their per-case shortcuts stay out: recipes, multipliers, taming dice, compressed time, per-species brain code, no per-voxel fluids. See openworld-reference.md §3 and openworld-engine-notes.md §3.

**Now: plumbing, golden SAME.**
1. **Per-law timing.** Mean, p99 and max ms per tick for each law, against a stated budget, printed by stepcheck.
   - Add a slow-tick line: when a tick runs over budget, one log line splits it by law (heat, fluids, fire, air, minds, seam, labels) and gives the active-set sizes. This is their "GEL" hitch line.
   - Nothing in src/voxel times itself today.
2. **Budget, partial, log.** Every flood fill or model call that can grow gets a cap: the air labeller, island labels, perception rays, mind calls.
   - At the cap it uses the partial result and logs that it did.
   - A silent cap is a lie; a logged one is a measurement.
3. **Hysteresis on every on/off threshold:** sleep and wake, activity windows, and later the mind tiers. Wake at one level and sleep at a lower one, so nothing flickers at an edge.
4. **Invariants worded as rules, citing the ruling.** Their example: "the Chunk edge must stay a multiple of the light cell (W-05)". Our ruling and D ids go into assert and log text.
5. **Magic + version headers** on saved worlds and on the `.npz` frames the renders read.
6. **One switch per law:** an environment variable that turns it off, for bisecting a bad scene. None exist today.

**With wave 13's one material record.**
7. **Retire, never renumber.**
   - Each material gets a stable key, and a retired one names its successor.
   - Why it matters: goldens and snapshots store ids, so renumbering silently rewrites old worlds.

**With wave 15** (clocks, scheduler, skip-ahead), or wave 13 if L11 moves forward:
8. **Simulation level of detail.**
   - Voxel laws run where someone watches.
   - The far tier is the same laws in closed form.
   - Work hands back to the voxel laws when attention arrives.
   - Their ledgers (one record per slow process, with a due time, and no per-tick scan) are L11's shape.
9. **A room far from attention is one heat node.**
   - The air-region label gives the volume, the wall areas and the openings.
   - UA comes from the cells' real conductivity and the openings; C comes from their real heat capacity.
   - The room relaxes toward the steady state (ΣGᵢTᵢ + Q)/ΣGᵢ with time constant C/ΣG, exactly.
   - OpenWorld solves only the steady state; we add the real lag.
   - Test: the fireplace-and-open-door scene, voxel law against far tier, on the facts the scene is about.

**With wave 15 and "later: offscreen people"** (see the latency budget):
10. **Mind activity tiers.**
    - Active minds get model calls.
    - Resting minds run on static drives, with zero calls.
    - Dormant minds are an L11 summary.
    - Promotion goes by perception salience, the same lens as the two-channel halt, with hysteresis as in item 3.

**With wave 17 (animals, packs) and "later: world at scale".**
11. **Lazy population.** A region is peopled when it is first perceived, not when the world is built.
12. **Niches as world predicates.** A creature or encounter appears only where world facts allow it: dark, near water, lava within 4 m. There is a cap per area. Encounters are seeded from state, not scripts.
13. **Progress as predicates over the event log.** A pack's quest step is "an event matching this pattern was seen", with wildcards allowed. This fits cultivation.yaml-style packs.
14. **A world = seed + edits.**
    - Once scenes are generated, store the seed and the cells that play changed.
    - Derived state (labels, islands, caches) is always rebuildable from the cells.

**Scenario design, any time.**
15. **Hazard-matched gates.** A place is closed by a physical hazard (smoke, flood, cold, dark) that a found or made thing answers, never by a key flag.

**Renders.**
16. **Merge faces before the Blender export:** binary greedy meshing in numba on 64-cell rows. It cuts polygon counts on flat surfaces by orders of magnitude.
17. **A coarser display grid for wide shots,** sampled from the 5 cm sim. Close-ups keep the 5 cm grid.
18. **If a live view ever comes:**
    - a 20-40 cm display grid in 64-cell chunks;
    - packed quads and vertex pulling;
    - face-direction ranges;
    - sky and block light with a day multiplier;
    - distance-ordered streaming with hysteresis and per-frame budgets;
    - far heightfield rings with hole masks.

    See openworld-rendering-and-craft.md §1 and §4.

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
| OpenWorld (T. Pascard) | closed | IDEAS ONLY, from its public wiki, its shipped data files and dashboard, and the strings in its programs. Never its code, assets or text. Its builds stay outside the repo (~/re/openworld) | "Stolen designs (OpenWorld)" above |

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

## A NUMBER IN THIS FILE THAT WAS NEVER MEASURED SHOULD SAY SO (2026-09-12)

Five entries did not survive being measured today, and they had one thing in
common: each was written down as a fact, with no measurement behind it, and
each then sat for weeks reading like a known quantity.

| item | what it said | what is true |
|---|---|---|
| 7 | `FALL_SUBSTEPS = 4` caps falls at 8 m/s, clipped above | it caps the DESCENT and lets the speed run — a 20 m fall arrived **41% too fast**, at twice the impact energy |
| 10 | a felled body travels much further than a topple should | the distance is exactly right for a rigid topple about the toe; the bug is that a slack body is modelled rigid |
| 12 | the humanoid is 28.5 kg | **38.1 kg** — 28.5 was before the chest and limbs were given depth |
| 39 | a plank held overhead is not accounted for in a doorway | nothing can be held overhead; a carried thing hangs DOWN and stays inside the body's own height |
| 42 | an arm can pass through a rail | true, but the first probe measured a rail that had already fallen over, and the fix had to be re-verified against one that stands |

None of them were lies. Every one was a reasonable inference written in the
same voice as the measured lines around it, and that is exactly the problem: a
reader — including me, weeks later — cannot tell which is which.

**So: state the measurement, or state that there isn't one.** "Measured, X"
appears on most pages of this file and it is the thing that makes it worth
reading. An entry without it should say "unmeasured" out loud, and an entry
that gets measured should be corrected in place rather than quietly fixed.

The same applies to docstrings. Item 12's 28.5 kg was stale in
`scenes._person` too, where it had been sitting under the words "it measures".

## DECISIONS WAITING ON A HUMAN (2026-09-12)

Two things are blocked on a judgement rather than on work. Written here rather
than left in a conversation, because a decision nobody wrote down is a decision
that gets made again.

**D1. Should the humanoid's limbs be THICKER?** (item 41.) **TRIED
2026-09-12, and the answer is no — not on its own.** Written up in full under
"What widening the arms actually did" below. The short of it:

| | before | 2-voxel arms |
|---|---|---|
| hand places (one bone / elbow) | 2 / 7 | **5 / 30** |
| mass | 38.1 kg | **43.9** (against ~47 for a real one) |
| walking footprint | 6 voxels | 7 |
| **the suite** | 107 pass | **15 FAIL** |
| the wind-up facing +x | refused | **still refused** |
| how far he leans, free | 10.7 deg | **8.6** — worse |

It buys the pose ENUMERATION and the mass, and costs the reach ACT. Do not do
it alone; it is a symptom of D3.

**D4 — ANSWERED 2026-09-12: two axes on every joint, built.** The freedom of
movement was never going to come from which way the body is built; that only
chooses which single plane an arm swings in. Every joint carries a PAIR now —
swing (in the plane the body faces) and spread (out of it) — so an arm can
reach sideways, across its own chest, and overhead-and-out. See "A joint is not
a hinge" below. The build question below is still open and now matters less.

**D4 (the build). WHAT DOES `facing` MEAN, GIVEN HOW THE BODY IS BUILT?** The humanoid
has its shoulders along x, and scenes almost always set `facing = (1, 0)` —
PARALLEL to its own shoulder line. That is the root of the three findings, and
measurement has now narrowed it to one line of build rather than a redesign:

| built facing | 0 turns | 1 | 2 | 3 |
|---|---|---|---|---|
| **+x**, along its own shoulder line | refused | refused | refused | refused |
| **+y**, across its shoulders | **-0.9** | **-0.9** | **-0.9** | **-0.9** |

**Turning PRESERVES the relationship.** It was never an anisotropy — whether a
body can wind up depends on how it was BUILT, and then holds in every direction
it can turn to. So the fix is in the build, and `_turn` carries it everywhere.

Three ways, and this one IS a taste question:

- **(a) Default `facing` to (0, 1).** One line, free, measured to work. Costs:
  every scene that assumes (1, 0), and bodies start facing "north" for no
  reason a reader can see.
- **(b) Rebuild `_person` with shoulders along y.** Anatomically honest and
  keeps `facing = (1, 0)` meaning "east", which is what a reader expects.
  Costs: the footprint's orientation flips, so doorways tuned to a 6-wide-in-x
  body need re-checking — the same class of cost that made widening fail 15.
- **(c) Leave the build; make turning carry the meaning.** Bodies turn before
  walking. Costs: ticks on every journey, and the timing of every walking test.

My read is (b) — it is the only one where `facing = (1, 0)` means what it says
AND the arm is in a plane without the torso in it. But (a) is free, and this is
a choice rather than a measurement.

**D3. A BODY CANNOT TURN.** ~~Open~~ **BUILT (2026-09-12)** as `_turn`: quarter
turns, exact, refusable when there is no room. What remains of D3 is D4 (which
way it should be built) and whether walking should turn a body automatically. Its geometry is fixed when it is built while
`facing` rotates freely, so the humanoid FACES ALONG ITS OWN SHOULDER LINE half
the time. Real anatomy puts the shoulders left-right and the face forward —
perpendicular — so an arm is beside the torso on one axis and swings in a plane
that does not contain it. Ours has them parallel, which is why:

- a wind-up sweeps the arm through its own chest (and widening makes that
  worse, not better);
- a thicker arm collides with its own torso during a reach;
- the extra arm mass hangs forward of the toes and the man overbalances sooner;
- and which poses a body has at all depends on its compass bearing.

Turning the shoulders with the heading fixes all four at the root. It is the
bigger piece of work and it is the RIGHT one — D1 without it is treating a
symptom, and the 15 red tests say so out loud.

**D2. Is a CROUCH symmetric?** (item 45.) Bending a knee moves everything
ABOVE it, so the body has to hang from the feet rather than from the hips — and
two feet on the ground give two paths up to the hips, which can disagree. If
both knees bend by the same amount the two agree and the problem goes away.
That is a claim about bodies: an uneven crouch is a LUNGE, and we do not model
lunges. Cheap to build once said, wrong to slide in unsaid.

## TODO — open, in the order I would take them (2026-09-07)

Momentum exists now (`TICK_S`, `vfall` + drag, `_launch` free flight, `jump`).
These are what it opened up and what it left behind.

**Momentum, finishing the four cases**

1. ~~**`roll` — a landing spread over TIME.**~~ **BUILT (2026-09-11).** Force is the momentum change divided
   by how long it takes to make it, so a body that keeps moving and stops over
   many ticks is struck far less hard than one that stops dead. Needs a landing
   that can take more than one tick. Test written: `test_ROLLING_on_landing_...`
2. ~~**`swing` — hands putting force behind a moving mass.**~~ **ALREADY
   BUILT**, and this entry was stale: the act is on the hands menu, `_swing`
   drives it, the edge declares its own contact area (`fill(edge=...)`), and
   `test_an_axe_SWUNG_bites_where_the_same_axe_PRESSED_does_not` has been
   passing. Sixth entry this session that did not survive being checked.
3. ~~**Throwing falls out of (2)**~~ **BUILT (2026-09-12)**, and it did fall
   out of it. See "A throw is a swing that lets go" below.
4. ~~**Catching.**~~ **BUILT (2026-09-12)** — and it was cheap, exactly as
   this said. See "A catch is a grab at something that will not wait" below.
5. **Walkers cannot change elevation at all** — not a step up, not a stair, not
   a ladder. TRIED 2026-09-11 and taken back out; it works, and it is blocked
   behind item 13 rather than behind geometry. See "What a step up cost".
6. ~~**Friction heat: `mu N v` into `E`.**~~ **BUILT (2026-09-12)**, and it
   cost almost nothing once hauling had a force to charge for — the sliding
   contact it wanted turned out to be a man dragging something. See "Friction,
   and the joules nobody was counting" below.
7. ~~**`FALL_SUBSTEPS = 4` caps falls at 8 m/s.**~~ **FIXED (2026-09-12), and
   the note had the sign wrong.** It did not cap the speed, it capped the
   DESCENT — so long falls arrived too FAST. See "Falls were arriving too hard"
   below. Raised to 16, which costs 1.01x a tick.
8. ~~**A flying body that leaves the world loses mass**~~ **FIXED
   (2026-09-12).** And it was reachable more easily than the note thought: not
   by launching anything, but by TOPPLING a tall thing standing at the edge.
   See "What leaves the world" below.

**Bodies and identity**

9. ~~**Hauling costs the hauler nothing**~~ **FIXED (2026-09-12).** 92 ticks
   across a room empty-handed, 175 dragging lead. See "What you drag" below.
10. **A felled body falls like a PLANK, not like a person.** Re-measured
    2026-09-12 and the original note had it the wrong way round: a 30-voxel
    body stood at x17 lands centred at x34.4, spanning x18..46. That is not
    too far — it is exactly right for a rigid body pivoting about its leading
    TOE, which is what `_topple` does. The distance was never the bug.

    The bug is the model. `_collapse`'s own docstring says "an unconscious body
    is a SLACK object" and then promotes it as a rigid one — so a man who
    faints goes over stiff as a felled tree, head landing 28 voxels from his
    feet. A person crumples: the knees go, the trunk folds, and the head comes
    down near the feet.

    Newly feasible, and this is why it is worth restating rather than closing:
    unconsciousness is the ABSENCE OF MUSCLE, and `_law_pose` already gates
    every pose on the torque a muscle has. A body with no muscle should sag to
    wherever gravity puts its joints — which is a solver, but a small one, and
    the joints exist now.
11. ~~**Bodies wake exactly where they fell**~~ **FIXED (2026-09-12)**, and
    the truth was worse than the note: they were not merely waking there, they
    were BEING THERE the whole time. `_eye` — where other bodies hear and see
    you from — was written only by the will layer, and the will layer skips
    anyone unconscious. Measured on the rescue: a man dragged eighteen voxels
    to the door was located by everyone else back in the burning room, at
    standing head height, while he lay on the floor. Being carried is a thing
    that happens TO you; a body does not have to be awake to be somewhere.
    Test: `test_a_body_MOVED_WHILE_UNCONSCIOUS_is_where_it_was_PUT`.
12. **The humanoid is light** — **re-measured 2026-09-12: 38.1 kg, not the
    28.5 kg this said.** A real 1.5 m person is about 47 kg, so it is at 81% of
    one rather than 60%, and the item is milder than it read. Still light, and
    still for the same reason: a stick figure at 5 cm fills less of its own
    outline than a person does, so the force law is right and the body is thin.
    Related to D1 — thickening the limbs would fix the mass and the pose space
    with one change.

**Minds**

13. **The menu policy work.** Taste: BUILT 2026-09-11. Menu RECALL: MEASURED
    and FIXED 2026-09-11 (81% -> 100%, see below). Still open: no model is wired
    to `pick`, no exploration slots, no uncertainty gating.
14. ~~**Belief records geometry only, and nothing forgets.**~~ **BUILT
    (danger 2026-09-11, the FLOOR 2026-09-12).** A body remembers where it saw
    fire and the memory fades; and it now plans over the floor AS IT LAST SAW
    IT, rather than reading the live lattice through a mask of where it had
    looked. See "The floor a body remembers" below. Still open: it remembers
    that a door is THERE, not that it was open, locked, or on fire.
15. ~~**An "eyes" limb.**~~ **BUILT (2026-09-12)** — and the item's own
    condition had come true: belief has contents worth checking now. A body
    that chooses to look behind it finds a fire behind it at tick 1 where one
    waiting for the sweep takes 50. See "Looking is something a mind can do"
    below.

**World**

16. ~~**Active regions**~~ **HALF BUILT (2026-09-11)**: a quiet world went
    98 ms/tick to 1 ms. A busy one barely moved, because the gas laws will not
    window — see items 27 and 29.
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

20. ~~**A GRIP SHOULD CARRY LOAD.**~~ **BUILT (2026-09-08)** — see "A grip
    carries load" and "What you hold up stands on your feet too" below. The
    reaction is built; the roped-pair drama is NOT, and it turns out to want a
    body that can reach and lean, not a bigger force. Rolled into 24.

21. ~~**A fall should hurt.**~~ **BUILT (2026-09-08)** — see "A fall hurts"
    below.

22. ~~**A flier stops dead on any contact.**~~ **BUILT (2026-09-08)**: the
    cheap fix as designed — a hit is retried with the horizontal part removed,
    and is an arrival only if straight down is blocked too. A block thrown at
    a tall wall slides down its face and lands at the bottom, re-rasterising
    where it LANDS rather than where it grazed (locked by test). Stated
    softness: the wall absorbs the sideways energy unpaid, which starts to
    matter once throwing is built.

23. ~~**Grab an OBJECT, not just a person.**~~ **BUILT (2026-09-08)** — see
    "A grip carries load" below. The axe test is green.

24. ~~**A BODY THAT CAN HOLD A POSE.**~~ **BUILT (2026-09-11)** — `_law_pose`,
    `p["pose"]`, `_repose`. Chains of bones and elbows followed on 2026-09-12.
    Original note:  The one missing piece behind three
    separate things now: an arm that stays where the swing left it, a crouch, and
    a man reaching out over a lip to take hold of someone below — which is what
    would let a load pull its holder over, and is the only reason it cannot. A
    limb re-rasterised each tick from a joint angle, rather than promoted off the
    lattice and put back. This is the next real build.

**From the reference harvest (2026-09-11), in the order I would take them**

25. ~~**SNAPSHOT AND RESTORE.**~~ **BUILT (2026-09-11)** — `World.snapshot()` /
    `restore()`, policy excluded.
    Original note:  (Rapier, Apache-2, code may be ported.) The whole
    world is a handful of numpy arrays plus a list of dicts, so this is small —
    and it is the TOOL the next three items need. Today the differential guard
    can only compare two runs at the END; with snapshots it compares them at
    every tick and names the first tick and the first cell that diverged. Also
    what a table needs to rewind, and what makes a trace harvest honest.
    **Doing this first on purpose: it is the thing that makes 26 and 27 safe.**

26. ~~**KERNEL FUSION.**~~ **BUILT (2026-09-11)** — optional numba kernels,
    bit-identical to the numpy path, ~8%.
    Original note:  (No reference helps — this is ours.) The biggest lever
    left for a world that is actually doing something: thirty laws each making
    several passes over the same arrays, when a cellular-automata engine sweeps
    once and applies many rules per cell. We are memory-bandwidth bound and
    numpy cannot express the fused form. numba or Taichi. Wants 25 as its guard:
    fuse, then diff tick-by-tick against the unfused run.

27. ~~**ISLAND SLEEPING.**~~ **BUILT (2026-09-11)** — an airspace is an island;
    the dungeon went 311 ms/tick to 78.
    Original note:  (Bullet, zlib, code may be ported.) Bullet already
    solved what active regions ran into: bodies in contact form an ISLAND, an
    island sleeps when everything in it has been quiet a while, and touching one
    wakes the WHOLE island, never half. An airspace is an island. Take the
    structure and the wake discipline. Blocked on one modelling decision that is
    not ours to guess: whether distant air is modelled at the same fidelity.

28. ~~**A REVOLUTE JOINT.**~~ **BUILT (2026-09-11)** — an angle, a speed cap
    (`arm_wmax`) and a torque the muscle either has or has not (`arm_Nm`).
    Original note:  (box3d, MIT, code may be ported.) This is item 24 with
    a name and a proven implementation. A shoulder is a revolute joint with a
    motor and limits; we already have the motor (`arm_Nm`, faded by Hill) and the
    inertia (`I = sum m r^2`), and what is missing is the constraint that keeps
    the far end attached while the angle is driven — sequential impulses with
    soft constraints, documented in Erin Catto's GDC papers.

**D5. HOW MUCH OF THE WORLD IS SIMULATED PROPERLY, AND WHERE?**
(Answers items 29 and 38; decided 2026-09-13.)

**The rule is: detail follows CONSEQUENCE, not distance.** Far-away things are
cheap by default, and anything that can reach the player is paid for in full —
including things that are cheap right up until they are not.

What that has to survive, in the user's words: an earthquake started a mile
off, a bullet fired from out of sight, and a scope that looks somewhere far
away and expects it to be there properly when it does. So:

- **cheap is the default, everywhere far.** Coarser grid, cheaper laws, or
  both. The only requirement is that the answer roughly makes sense — a fire
  two rooms away must still burn out, still make smoke, still weaken a beam.
- **a WINDOW can be opened anywhere,** at full fidelity, and moved. A scope is
  one. So is a player walking somewhere. So is anything the far field says has
  become interesting.
- **and coarse must be able to HAND OVER.** The hard part is not running two
  fidelities, it is the seam: a bullet leaving a fine region into a coarse one
  and arriving somewhere real, an earthquake computed coarsely that has to
  shake a room that is being simulated properly. Conservation has to hold
  ACROSS the seam or the sim quietly leaks — and this file already has the
  machinery for saying so (`gone`, `shed`, `total_mass`).

**What it means in play, which is the point:**

| | |
|---|---|
| a fire in the next building | burns coarsely, spreads coarsely, and is exactly right about whether the building is still standing an hour later |
| someone shouts a mile away | never computed at all — no path, no consequence |
| a bullet from out of sight | the SHOT is a consequence that reaches you, so the region it travels through is refined along its path |
| a scope on that far ridge | opens a window: that patch runs properly for as long as it is looked at |
| an earthquake | coarse everywhere, fine where it meets anything that can be broken |

**THE PROMOTION RULE** (the user's, 2026-09-13, and it is sharper than what
was here before): *when something local does a thing that needs information
from — or makes a change to — somewhere farther away that is running coarse,
find the SMALLEST area of effect and promote that to fine.*

Three properties worth spelling out, because they are what make it a rule
rather than a wish:

- **It is driven by the ACT, not by a watcher.** Nothing has to patrol the far
  field asking whether it has become interesting. A body fires, looks, shouts,
  or shakes the ground; the act names what it needs, and that names the region.
  A coarse region nobody is interacting with is never examined and costs
  nothing, which is the whole point of it being coarse.
- **Smallest area of effect.** A bullet promotes a corridor, not a county. A
  scope promotes the cone it can see, not everything at that range. Getting
  this wrong in the generous direction is how a level-of-detail scheme quietly
  becomes no level-of-detail scheme.
- **Information and change are the same question.** Reading distant state and
  writing it both need the state to be true at the fidelity the reader expects.
  A scope that looks at a coarse hillside and a bullet that arrives at one are
  the same failure — the answer was computed at a resolution the asker was not
  told about.

What it needs from the engine, all of which exists in some form: a region that
can be re-gridded without losing what it holds (`fill` and `_air_regions`
already think in regions), a conservation check ACROSS the seam (`total_mass`,
`total_energy`, `gone`, `shed`), and acts that can state their own reach — which
the menu seam already forces them to do, because a menu row that cannot say
what it touches cannot be checked for legality either.

29. **DOES DISTANT AIR GET THE SAME FIDELITY?** A modelling decision in Ruling
    1's stopping-rule family, and the reason a burning town costs what it does:
    its airspace count is ONE, because outdoor air really is all connected, so a
    fire in one hut shares air with the whole town. True, and the reason gas
    cannot be made cheap by drawing a box round it. What a coarser distant air
    would cost has to be MEASURED, not assumed.

30. **PHASE TRANSITIONS AS ONE TABLE.** (Powder Toy, GPLv3, ideas only.) `MELT`
    and `BOIL` are two mechanisms doing one job. Their (lowT, highT, lowP, highP)
    -> target table collapses both into data, and the pressure half now has a
    pressure field to key on, which it did not when this was first written.

31. ~~**SPARSE VOXEL HIERARCHY — MEASURE FIRST.**~~ **MEASURED, and the
    answer is NO — see below.** (XCube, the data structure only.)
    A 1.02M-voxel town is 30+ MB of dense arrays swept by thirty laws. But a
    sparse structure only pays if most of the world is EMPTY, and ours is mostly
    solid ground and air that gas actually uses. Measure occupancy before
    believing it.

**Found by the experiments themselves (2026-09-11), in the order I would take them**

32. ~~**`_take_up` does not always lift a load to the fist.**~~ **FIXED
    (2026-09-11)** — it placed the load CENTRED on the hand, so anything wider
    than a hand reached back into the holder's own chest and the move was
    refused for the good reason that a man was standing there. It hangs BESIDE
    the fist now. Measured while
    trying to close item 28's payoff: a 38 kg lead block, a holder with 4000 N
    of strength, and the block stayed sitting on the ledge being DRAGGED rather
    than carried — so the balance sum rightly ignored it (a load resting on the
    world weighs on its own footprint). Until this works, "he was left holding
    it over the drop" cannot be reached from the grab, only set up by hand.

33. ~~**OXYGEN'S THRESHOLD, measured like smoke's was.**~~ **BUILT
    (2026-09-11), and it needed no threshold at all — see below.** Item 29 is answered for
    smoke and open for oxygen, which is 32% of a burning town. Unlike smoke,
    oxygen's deficit IS local — the exact box is 21% of the world at tick 20 and
    63% at tick 69 — and a 1e-4 line holds 99.8% of the deficit in 28%. So this
    one can probably be windowed EXACTLY, no judgement needed. Measure first.

34. **`_law_burn` and `_law_crack` are still whole-world** (part done: its
    coarse air field is cached now — see below). Crack is cheap and
    already guarded. Burn is the hard one and probably cannot be windowed at
    all: its oxygen accounting is per-AIRSPACE, the same reason gas cannot.

35. **A LEAN, and a CROUCH.** ~~A lean~~ **BUILT (2026-09-12)** — see "A
    spine bends" below. A CROUCH is still open, and it is the harder half: it
    needs the tree ROOTED AT THE FEET (bend a knee and everything above comes
    down), and two feet on the ground make a loop rather than a tree. `_leap`
    still takes `crouch_m` as a constant for want of a real one.

36. ~~**An arm cannot sweep through what it is reaching for.**~~ **BUILT
    (2026-09-12): limbs are CHAINS of bones now, and the humanoid has elbows.**
    See "What an elbow bought, and what it exposed" below. The stated case —
    a man beside the person he is holding — turned out to be right as it was:
    nobody's arm goes through a person. What an elbow really buys is measured
    there instead.

37. ~~**`roll` — a landing spread over TIME**~~ **BUILT (2026-09-11), and it
    was the oldest open item in the file.** See below. (Was item 1.)
    Force is momentum change divided by how long it takes to make it, so a body
    that keeps moving and stops over many ticks is struck far less hard. Needs a
    landing that can take more than one tick. Test written and xfailing.

38. **An OPEN HALL costs 450 ms/tick** — 100x80x50 with nothing in it but
    people. Island sleeping cannot help: it IS one island, one airspace, and
    the gas laws sweep all of it. Noticed while pricing item 14's floor
    memory (which cost 0.2%; the hall cost everything else). This is the real
    shape of item 29 — the question is not whether to window the gas laws but
    whether distant air deserves the same fidelity at all.

39. ~~**A body plans a doorway at ITS OWN height, carrying or not.**~~
    **NOT A BUG — measured 2026-09-12.** Nothing can be held overhead: a
    carried thing hangs DOWN from the fist, and a 1 m plank taken up by a 1.5 m
    man occupies z1..19 inside his own z1..30. There is no clearance to account
    for that the footprint does not already cover.

    It becomes live the moment a body can raise a load above its own head,
    which an elbow now nearly allows — a bent arm reaches z30 on a body whose
    crown is z30. Worth re-checking when anything can lift.

    **Filed from reading the diff rather than from a test, and wrong.** That is
    the third roadmap entry this session that did not survive being measured.

40. **`free` records a PERSON standing somewhere as blocked ground**, and
    clears it on the next look that way. That is right as far as it goes, and
    it means a body can plan through a crowd it has not looked at recently.
    Whether a remembered person should be a soft cost rather than a wall is a
    real modelling question, not a bug.

41. **A LIMB ONE VOXEL THICK CAN ONLY BE DRAWN AT RIGHT ANGLES**, and the
    humanoid's arms are one voxel thick in x and three in y. Measured: 2 hand
    positions with one bone, 7 with an elbow; at TWO voxels thick it is 5 and
    **30**. So posture is limited by the body's thickness, not by how many
    joints it has — and it is anisotropic, because a man facing east and the
    same man facing north turn their arms in planes of different thickness.
    Widening the arm changes the walking footprint and therefore every doorway
    in every scene, so it is a decision to take deliberately. See "What an
    elbow bought" below.

42. ~~**An arm can pass THROUGH a rail if the angles either side of it are
    undrawable.**~~ **FIXED (2026-09-12)** — the angles in between are swept
    now. See "An arm went through a wall" below.

43. ~~**`_reach_around` gives up and the reach is cancelled**~~ **FIXED
    (2026-09-12)**, and it was two bugs. See "A question already answered"
    below.

44. ~~**A GRIP SHOULD COUNTERBALANCE A LEAN.**~~ **BUILT (2026-09-12)** —
    10.7 degrees to 45.8 with a hand on the post. See "A hand on the rail"
    below.

45. **A CROUCH NEEDS THE TREE ROOTED AT THE FEET.** Bend a knee and everything
    ABOVE comes down — which is the tree inverted, and two feet on the ground
    make a loop rather than a tree. A symmetric crouch resolves the loop (both
    knees the same), but that is a modelling claim to make deliberately rather
    than to slide in. Item 35's remaining half.

47. **PULLING ON A RAIL SHOULD BEAT HOLDING ONE.** The base reaching the
    HAND is not a fudge — it is exactly right for a contact that can only
    PUSH, which is what "the centre of mass must lie within the contact hull"
    means. Going past it needs the hand to PULL, and then the limit is the
    arm's tension against the toppling moment: `strength_N` times the hand's
    height above the toes. Same arithmetic `_pulled_over` already does, and it
    would make reaching over a lip depend on how strong the arm is. **Blocked
    behind 48**: the payoff is a rescuer holding on with one hand and reaching
    with the other, and there is only one working hand.

48. ~~**A BODY HAS TWO ARMS AND ONLY ONE OF THEM CAN DO ANYTHING.**~~
    **BUILT (2026-09-12)** — a dominant hand, and the other when the other is
    the one for the job. See "Which hand" below. Still one thing held at a
    time: two independent grips would want `p["held"]` to be per-arm, and that
    is a bigger change than this was.

49. ~~**`pose` RUNS ON WHERE `drawn` CANNOT FOLLOW, for ever.**~~ **FIXED
    (2026-09-12).** The angle outrunning the flesh is deliberate and right —
    it is how a limb crosses the angles the lattice cannot draw and catches up
    at the next one that can. What was missing was the END of that story: when
    the command has been given in full and the body is still not there, THAT is
    how far this joint goes here, and the angle settles to where the flesh is.
    Asked and at agree once a body has stopped moving. Test:
    `test_a_BODY_STOPS_ASKING_for_a_pose_it_can_never_be_IN`.

50. **ONE OBJECT HELD AT A TIME** — narrower than this said. `p["held"]`
    (a thing) and `p["dragging"]` (a person) are separate slots, so a body can
    already hold a post in one hand and a falling man in the other, and does:
    see "One hand on the rail" below. What it cannot do is hold TWO objects, or
    two people. That wants `held` to be per-arm, and then "hands" as one limb
    stops being the right shape — two hands doing two different things is two
    menus.

58. ~~**THREE QUARTERS OF THE HARVEST IS "THIS PART DID NOTHING".**~~
    **DECIDED 2026-09-12: keep them.** A model that never sees "do nothing"
    cannot choose it, and will fidget — reaching, speaking and looking about
    every quarter second because it has never been shown a body at rest. The
    imbalance is handled on the LEARNER's side, per limb, so each limb's nulls
    are weighed only against that limb's own alternatives. Nothing changes in
    the sim, which is the right side of the seam.

    **The original:**

60. **THREE QUARTERS OF THE HARVEST IS "THIS PART DID NOTHING".** Measured
    on the rescue, 2026-09-12: 4 decisions x 5 limbs = 20 picks logged, of
    which **15 are null acts** — the mouth saying nothing, the waist standing
    as it is, the eyes going on looking about. It was three limbs a week ago
    and is five now, so today made it worse.

    This is not a bug. A body genuinely does nothing with its mouth most of the
    time, and a menu that hid that would be lying about what was on offer. But
    a learner trained on these rows sees mostly "do nothing", which is the
    class-imbalance problem arriving before anyone has written a learner — and
    it is much easier to decide what to do about it now than after a model has
    been trained on it.

    The options are all outside the sim, which is the right side of the seam:
    weight the rows, filter to decisions where SOMETHING changed, or learn per
    limb rather than per body. Worth choosing deliberately.

59. ~~**A BODY CAN LAND AND JUMP AGAIN INSIDE ONE TICK.**~~ **FIXED
    (2026-09-12).** A landing costs time now, and the time is the body's own:
    a landing is absorbed by sinking through a crouch, and how long that takes
    under gravity is `sqrt(2h/g)` — 9 ticks for a man, 13 for a giant, and no
    constant of its own. A leaper told to leap for ever went from **0 of 120
    ticks with its feet on the lattice to 36**, which is exactly what it was at
    the old 0.75 s reaction time: the gather restores the rhythm without giving
    back the reaction.

    **The original:**

61. **A BODY CAN LAND AND JUMP AGAIN INSIDE ONE TICK.** Found by item 57:
    at a quarter-second reaction a leaper lands and re-decides to leap before
    the tick is out, so its flesh is never on the lattice at a tick boundary at
    all. The physics is right — it crosses the gap either way — but there is no
    recovery in a landing, and a body that has just taken a fall should not be
    as ready to jump as one that has been standing still. It is the same shape
    as `_fell`, which already keeps a falling body from re-deciding every tick.

    It also broke a test, and the test was wrong rather than the change: it
    sampled only flesh ON the lattice, which silently assumed a body spends
    most ticks standing. ASK WHERE THE MATTER IS, not where it is standing.

57. ~~**A BODY DOES ONE HAND-THING PER DECISION, AND DECISIONS ARE 30 TICKS
    APART.**~~ **FIXED (2026-09-12).** `decide_every` is DERIVED now, from
    `WILL["react_s"]` — 0.25 s, a simple visual reaction: see, choose, begin to
    move — over `TICK_S`. That comes to 10 ticks where it used to be a typed
    30, which was nearly three reaction times.

    Derived rather than typed so a finer or coarser tick cannot quietly make
    everybody quicker or slower on the draw, which is the loop worth having:
    the sim's own clock sets how often its minds get to think.

    What it was about: a fall from a ledge takes about 25 ticks, so at 30 a man
    who was not ALREADY holding the rail got exactly ONE decision in the whole
    of his friend's fall and had to spend it on the rail or on the catch. At a
    quarter second he gets two. (Measured after: the rescue still does not
    complete from a standing start — the free hand has to find him within
    reach at the right moment as well — so this was necessary and not
    sufficient. Worth saying, because "I changed the number and the scenario
    still fails" is exactly the sort of thing that goes unsaid.)

52. **THE SUITE IS TEN MINUTES**, 104 tests. Profiled 2026-09-12: the 15
    slowest are 390 s of the 586, and there is no single offender — it is a
    long tail of multi-body scenes that must run hundreds of ticks before the
    thing under test happens. Trimming them one at a time is slow work and
    risks weakening them.

    **The cheap answer is `pytest-xdist`**, which is not installed. The tests
    share nothing, so `-n 16` on this machine would put the wall time at the
    length of the SLOWEST SINGLE TEST — 62 s — which is ten minutes down to
    about one for one dev-only dependency and no change to any test. Not
    installed without asking, because adding a dependency is the sort of thing
    that should be somebody's decision rather than a side effect.

    **And it needs no dependency to prove**: `pytest` has always taken a list
    of test ids, so splitting the collected list across N shells and waiting
    gives the same thing by hand. Measured 2026-09-12, 111 tests:

    | | wall |
    |---|---|
    | one process | **11:47** |
    | 8 shells, split round-robin | **3:38** |

    Which also shows what xdist would add over the hand-rolled version:
    round-robin splitting does not know what a test COSTS, so the eight chunks
    came in between 47 s and 218 s and the slowest one is the whole wall time.
    xdist hands work out as workers free up, so it lands near the 62 s floor.

    The floor after that is `test_reflexes_flee_and_alarm_spreads_by_shout_or
    _sight` at 62 s, and THAT one would be worth trimming.

    A script lives in this session's scratchpad, not in the repo: a suite that
    is only fast when somebody knows the trick is still a ten-minute suite.

53. ~~**`total_wood()` COUNTS WHAT LEFT THE WORLD AND THE OTHER TOTALS DO
    NOT.**~~ **FIXED (2026-09-12)** — `total_mass` counts cells, bodies in
    flight, and the tally. Which also fixed a quieter one: a swung axe used to
    cease to exist for as long as it was in the air.

56. ~~**CATCHING A FALLING PERSON IS A DIFFERENT ACT.**~~ **BUILT
    (2026-09-12)**, a few hours after it was filed, because everything it
    needed had just been built for other reasons. See "Somebody goes over the
    edge, and somebody catches them" below.

55. **A THROW IS ONE SHOULDER THROUGH A QUARTER TURN.** ~~The wind-up~~
    **BUILT (2026-09-12)**, and it works — 4.11 m/s to 5.28, a 28% gain, where
    there is room for one. **But there usually is not, and that is D1 again**:
    the arm winds back through its own chest. See "A wind-up, and a third
    reason to widen the arms" below.

    Still open: the TRUNK unwinding into the throw. The waist exists, so this
    is now the same rotation about one more joint — but it wants the arm's
    swing to compose with the waist's, which the bone tree can express and
    `_swing` (which promotes ONE limb to a rigid body) cannot yet.

54. ~~**NO RENDERS THIS SESSION.**~~ **MY ERROR (2026-09-13).** Blender is
    not on PATH and never needed to be: it is vendored at `tools/blender/`
    (4.2.23 LTS) and runs. I recorded "no renders are possible in this
    environment" without looking in the repo, and that stood for two days.

    The lesson is the same one as the unmeasured numbers, one floor down: a
    blocker written down without being checked is worse than an open question,
    because it stops anybody looking.

62. ~~**A GRIP ON AN ANCHORED THING TRIES TO CARRY IT.**~~ **FIXED
    (2026-09-13).** What the world is holding up does not come with you: a post
    set in the ground is not carried by the hand on it, the hand is
    CONSTRAINED by the post. So the post stays and the ARM has to reach — a
    pose that would put the fist further from it than an arm is long is
    refused.

    | a man at the lip | leans to | stopped by |
    |---|---|---|
    | nothing to hold | 10.7 deg | his own balance |
    | a hand on the post | **20.4 deg** | his arm's length |
    | what the spine could draw | 45.5 deg | — |

    The number barely moved (19.3 to 20.4) and the REASON changed completely,
    which is the point: "the post refuses to move" is the sort of right answer
    that stops being right the moment anything changes.

46. ~~**`_law_pose` poses and un-poses to test balance.**~~ **HALF DONE
    (2026-09-12)**: `_repose` takes `dry` now, so every check can be run
    without anything moving — which is what made `_bend_max` possible at all.
    The balance gate still poses and un-poses, because balance needs the
    body's centre of mass and that wants the prospective cells rather than a
    verdict. Smaller than it was.

    **The original:**

64. **REACH IS BOX TO BOX, WHICH IS COARSE.** It is three-dimensional now,
    but it compares BOUNDING BOXES — so two 1.5 m bodies on either side of a
    1.5 m ledge are "within an arm", because their boxes overlap vertically
    even though the parts that would touch are two metres apart. The honest
    measure is FIST to nearest cell, which every piece of exists (`_fists`,
    `_fist_of`) and which would also let an arm RAISE to reach something above
    it — the thing the old floor-only measure was really standing in for.

65. **TWO PEOPLE CAN HOLD THE SAME PERSON AND THE SIM KEEPS ONE.**
    `holders = {q["dragging"]: q for q in self.persons if q.get("dragging")}`
    is keyed by the DRAGGED name, so when two people have hold of one man the
    last in list order silently wins and the other's grip is ignored by
    `_law_footing`. Found in the rescue scene, where it happened not to matter
    because the rescuer came last — which is luck, not design. A tug-of-war
    over one body is exactly the case the scene exists to show.

63. **`_law_pose` poses and un-poses to test balance.** The gate reposes to the
    new angle, asks `_overbalanced`, and reposes back if the answer is no — two
    moves of flesh on the failing tick, and the world briefly holds a pose
    nobody adopted. Correct, because nothing else runs in between, and worth
    replacing with a dry-run once anything else wants one.

## A filter for 3D research links (2026-09-07)

Assessed on request, and the verdict was the same three times, so it is worth
writing the reason down rather than re-deriving it per link.

| Link | What it is | Use to us |
|---|---|---|
| [XCube](https://research.nvidia.com/labs/toronto-ai/xcube/) | generative 3D scenes on a sparse voxel hierarchy (VDB) | the DATA STRUCTURE, for active regions — not the generation |
| [InfiniCube](https://research.nvidia.com/labs/toronto-ai/infinicube/) | generates large dynamic driving scenes; its own page says geometry and appearance, no physics | none |
| [NVIDIA RTR voxels](https://research.nvidia.com/labs/rtr/tag/voxels/) | three rendering papers (ray-box intersection, GPU voxelization, cone tracing) | none — our render is not the bottleneck |
| [WorldSculpt](https://alaya-lab.github.io/WorldSculpt/) | images + instance masks + 3D boxes into one complete MESH per object, composed rigidly | none directly; see below |
| [Programmable Cellular Automata](https://arxiv.org/abs/2609.06102) (2026-09-09) | CA written as readable Python — local functions, a decision function, and GLOBAL functions over whole states; tested on level generation | no code to take; it independently CONFIRMS a design choice we already paid for — see below |
| [Principal Bundle Geometry of Qualia](https://doi.org/10.1093/pnasnexus/pgag261) (2026-09-09) | consciousness theory: qualia as relations under a symmetry group, not intrinsic properties | none. It is a theory of what experience IS, not of how to build a perceiver, and our percepts are already read off continuous fields |
| [Programmable World Model](https://alaya-lab.github.io/pwm/) (2026-09-11) | an explicit rule-driven state executor in FRONT of a video model; boxes compiled to pixel controls, video model renders only | the closest link yet, and it AGREES with us — see below. Its offer is a prettier renderer and scene authoring, neither of which is a bottleneck |

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

Not yet done, in order of expected value: ~~**active regions**~~ **BUILT
2026-09-11, 98 ms to 1 ms on a quiet world — see "Do not compute where nothing is
happening" below; the SPATIAL half is still open**; **kernel fusion** via
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

## A grip carries load — BUILT (2026-09-08)

Items 20 and 23, which turned out to be one build: a hand that can hold a stick
up is the same hand that can hold a hanging man, and both needed the support
law to know it.

**The grip is an edge in the support graph.** Support relaxes from the ground
up through material, and a hand is not material — so nothing on the lattice
could hold a hanging man, and everything a hand carried had to rest on
something. `_grip_cells` now seeds what a body holds as supported, exactly like
the ground: the load hangs from the body, and the body stands on its feet. The
slack cache keys on the grip set as well as the layout, so it stays correct by
construction. Four gates, each a cause: the holder must be footed (nothing to
brace against, nothing held up); the load must be within `strength_N` (an arm
that lifts 40 kg holds 40 kg); a hand holds things UP — a load starting at
or above the holder's own crown hangs from nothing; and A FIST CAN PULL, NOT
CLAMP — what hangs below the hold is a pendulum and costs the wrist nothing,
but a load whose weight rides above the hold is balanced on the fist, and the
wrist pays weight times lever to keep it there (`wrist_Nm`, ~20: a 4 kg hammer
held level at half a metre). The lever is measured from the FIST — the bottom
of an arm segment, where a hand hangs — to the load's centre of mass, at the
load cell nearest that fist; measured against the nearest touching cell
instead, a man who fell against his holder's chest read as gripped at the
chest and hung there. The axe at the fist costs a few newton-metres and is
carried; a man gripped by the ankle is hundreds, and rotates out of the hand —
which is why the braced-ledge test still ends with its man on the ground: his
puller stands BELOW him, holding an ankle with a whole man above it.
Measured: a man whose floor is carved away HANGS at the lip, unhurt,
for as long as his holder stands there (`test_a_GRIP_CARRIES_LOAD_...`); the
same man unheld is a wound on the ground — and `_haul` now says "left holding
him over the drop" instead of letting go.

**Objects are held the way people are.** `p["held"]` grips a CLUSTER — re-flooded
from the gripped cell each use, because objects have no registry: identity is
adjacency, checked against the material the hand closed on. The menu offers
nearby loose things ("take hold of the wood") capped at two, filtered to what
this body could at least shift — a model of attention, since every wall is
within reach of somebody and on nobody's menu. A liftable thing is carried to
the fist and hangs there by the grip; too heavy stays where it lies, still
held, and TRAILS like a dragged person when the holder walks. A carried thing
is one kinematic unit with the walker — walked as separate matter it is a wall
in front of its own carrier, and the carrier jams on it: once in `_walk`, and
once more in the PLANNER, where the stick's column deleted "straight on" from
its own carrier's menu.

**A held thing swings with the arm.** Its cells join the limb's rigid body, so
its mass slows the swing (Hill does the rest) and what it carries is what
lands. The grip survives the swing — mid-arc the thing is off the lattice with
the arm, and a grip is not lost because it moved (the same lesson `_haul`
learned from fainting bodies).

**An edge is declared, and the axe test is green.** `fill(edge=...)` puts a
contact area in m² on a voxel — "this edge concentrates its blow into X mm²",
a fact about a manufactured thing, the same class as its density (Ruling 2 q1),
because an edge is sub-voxel shape a 5 cm lattice cannot draw. The struck
threshold uses the sharpest edge the swung body carries (which face struck is
below the lattice's resolution too; an axe is swung edge-first, a fact about
how tools are held). Rubble is blunt: shattering a thing unmakes its edge.
Measured: the same swing that chews a trunk with a declared 2 cm² axe bit
(`pressed 0 voxels, swung >0, mass conserved`) leaves it untouched blunt — and
the take-up and every swing go through the menu like any other act.

**Known limits, stated:** a hanging man cannot be hauled back up; the wall that
takes a flier's sideways speed absorbs its energy unpaid; and carrying costs
the carrier nothing, the same softness hauling has.


## What you hold up stands on your feet too — BUILT (2026-09-08)

The residual from item 20: the load did not pull back on the holder. It does
now, and the answer to "and does he go over with him?" turned out to be NO, for
a reason worth having found.

**The rule.** A grip is an edge in the support graph, so the load hangs from the
holder — and the weight that has to sit over the holder's feet is his own PLUS
whatever hangs from his fists. That is the same moment balance every leaning
thing in the sim already gets: the load's weight times how far out it hangs,
against the holder's weight times how far back he can put it. How far back is
READ OFF THE BODY — his own half-width, because that is as far as a person can
shift their weight — so a small man is taken over by a load a big one shrugs at,
and no threshold is typed in. Without the lean term the sum says nobody may hold
anything at arm's length, which is plainly false.

**Two corrections it needed, each a real error.**

- **The weight comes in at the FIST, not at the load's own centre.** A thing
  hanging still pulls straight DOWN along the arm holding it; where its mass sits
  is the WRIST's question, which `_grip_holds` already asks. Summed at the load's
  centre, a body trailing on the floor behind a hauler read as hanging out past
  his toes and tipped him over.
- **Only what is actually hanging counts.** What the world still carries weighs
  on its own footprint. Before this the holder was tipped by a block he had
  merely taken hold of, one tick after touching it, while it sat on the rock.

Measured (`test_WHAT_YOU_HOLD_UP_STANDS_ON_YOUR_FEET_TOO`): a 38 kg man on a lip
holds 89 kg hanging from his fist and stays; 128 kg takes him over with it. Same
man, same grip, same lip — the mass is the only difference. Nothing fires while
either block is still stood on the rock.

**And the honest part: this can almost never fire for a hanging PERSON, and the
reason is not a missing force.** The fist sits about 5 cm outside the toes,
because these bodies stand with their arms at their sides and cannot reach. A
load hanging that nearly straight above your foot does not tip you however heavy
it is — it presses you into the floor. Correct statics, and it means the tipping
load has to be roughly three times the holder's own weight, which `strength_N`
rules out long before balance does.

What pulls a real rescuer over a cliff is that they REACH OUT over the edge,
putting their own weight out there with the load. These bodies have no pose to
do that with. So the roped-pair drama is blocked by the SAME missing piece as
the persistent-pose limb work — a body that can hold a pose other than the one
it was built in — and not by anything about grips or forces. Worth knowing
before building a counterweight of special cases to fake it.

## Can we LEARN the rules? — assessed, no (2026-09-09)

Asked as: "i am thinking if we can use [The Well](https://polymathic-ai.org/the_well/)
to learn parts of the voxel simulation rules."

**The Well** is 16 datasets, 15 TB, of numerical PDE simulation — turbulence,
Rayleigh-Bénard convection, MHD, acoustic scattering, supernovae — packaged with
PyTorch loaders and FNO baselines. It exists to train deep-learning SURROGATES:
networks that predict the next state of a solver you already have.

**A learned law is a case table with smooth interpolation, which is the exact
thing Ruling 1 forbids.** Everything this engine is worth comes from outcomes not
being stored: the same fist shatters glass and leaves a wooden post untouched,
and neither answer is written down anywhere. Fit a network to outcomes and you
have written all of them down at once, in a form nobody can read, argue with, or
write a ruling about.

Three concrete losses, in the order they would hurt:

1. **Conservation.** Every test in the suite leans on it — `_flesh_grams` before
   and after, "breaking is not losing: every gram is still there", mass held
   while a body was dragged as a fragment, while glass shattered mid-air, while
   an arm was torn off. Not conserving is the well-known failure mode of neural
   surrogates, and it is the single invariant this project leans on hardest.
2. **Unseen combinations.** The wins are all "nobody wrote either of them down."
   A surrogate is only as good as its training distribution; ours is the whole
   point of not having one.
3. **The docs are the spec.** "Measured, X" appears on nearly every page. A
   learned law cannot be measured against a reason.

**And the domain does not even line up.** The Well is continuum PDE on regular
grids of floats. This is a discrete material lattice: integer material ids,
per-voxel mass and fill fraction, fluid volumes, combustion chemistry, toughness,
rigid-body promotion. There is no FLESH, WOOD, GLASS or LEAD anywhere in it, and
no person. There is nothing to transfer.

**Where learning belongs, and it is already built.** The menu seam. `policy.pick`
gets the situation and the checked menu and returns an index; every pick is
written to `self.traces` with the full menu beside it, precisely so a harvest is
possible — you cannot learn from a choice without knowing what else was on offer.
A model there supplies TASTE and can never invent a door. Learning the physics
inverts the one architectural decision this project has been most careful about:
the sim is authoritative, the model chooses.

**The one narrow case for a surrogate, stated fairly.** `_law_air` (pressure,
wind) and smoke transport are the only genuinely PDE-shaped parts, and they are
the least conservation-critical. But speed is not currently the blocker, the last
hot spot (`_relax_slack`, 45% of tick time) was fixed EXACTLY by caching on the
material layout rather than approximately, and the next levers are active regions
and kernel fusion — both exact. If a surrogate is ever wanted there, the training
data has to come from THIS sim, because no public dataset contains our materials
or our laws; at that point it is distillation of a solver we already have, which
is a speed optimisation for later, not a way to get rules we do not have.

### Programmable Cellular Automata — a confirmation, not a technique

[arXiv 2609.06102](https://arxiv.org/abs/2609.06102) (Khalifa, Nasir, Siper,
James, Togelius) writes cellular automata as readable Python split into local
functions, a decision function, and GLOBAL functions over the whole state. Its
headline finding: global functions cut the iterations needed, and **some problems
cannot be solved by purely local rules at all**.

That is worth recording because we found it the expensive way, five times, and it
is the reason this engine is not a pure CA:

- `_relax_slack` — support is a fixpoint relaxation, because "is this held up"
  cannot be answered in one local pass.
- `_law_torque` — connected components, then a whole-body centre of mass.
- `_law_footing` — the same question per person, whole-body.
- `_claim_bodies` — one simultaneous, fair, whole-lattice flood.
- `_air_regions`, `_routes` — global labelling and breadth-first search.

So: no code to take (their target is level generation), but independent evidence
that the local/global split we pay for is the right shape, and a name for it.

## Cost that does not depend on voxel size — the want is right (2026-09-11)

Raised with [PWM](https://alaya-lab.github.io/pwm/): "there exists a world where we
manage rough states and the model predicts the world state, that's constant time
regardless of voxel size — we just have to manage rough world state."

**PWM is the closest link anyone has sent, and it argues our side.** Its four
parts are a state executor that keeps world state under rules, a DETERMINISTIC
compiler that projects that state into pixel-aligned controls, a video model that
renders observations, and a language model that writes the world spec. The model
never owns the state. Its complaint about ordinary video world models is close to
verbatim our own thesis: state "lives implicitly in visual histories, difficult to
inspect, edit, or verify", and "the world must persist beyond the current view,
where off-screen entities and latent facts are silently forgotten."

Mapped onto what is here:

| PWM part | Ours |
|---|---|
| state executor, rules | `src/voxel/sim.py` — and ours is materials and conserved quantities, not boxes |
| deterministic compiler | `tools/render/render_voxels.py` |
| video model | would replace Blender |
| language model writing the spec | would replace hand-written `scenes.py` |

So what is actually on offer is a prettier renderer and scene authoring from a
description. The render is 640x360 headless and has never been the bottleneck,
and authoring scenes is where the thinking happens. Worth revisiting if a scene
ever has to be built from a photograph.

**The constant-time claim is true and the constant is the problem.** A forward
pass does not care how many voxels there are — but it is paid in the one resource
already spoken for: 6 GB of VRAM, with reactive play budgeted at zero model calls
per NPC. The sim is 158 ms/tick on a 400k-voxel world on the CPU. A video model
doing a 30-second rollout is not competing with that; it is competing with the
language model for the card.

**And the deeper reason, which is not about cost.** The voxels are not the
picture. They are the reasons. A fist shatters glass and leaves a post because
energy met toughness over a contact area on specific cells; water beats a wood
fire and loses to an oil fire because there is a fuel field. Roughen the state and
you must choose which of those facts survive the roughening — and every one that
does not becomes a case somebody writes down by hand again. That is the property
table, arrived at from the other direction.

**But the want is right, and there are two exact routes to it, both already ours.**

1. **Active regions.** Most of a world is inert; do not compute where nothing
   changes. Same win, no correctness given up, criterion already agreed above
   (heat is provable, radiation is the trap, gas freezes by whole airspace), and
   expected to be the order-of-magnitude lever on big quiet worlds.
2. **A coarse far field, under the SAME laws.** `voxel_cm` is already a World
   parameter and 31 places read `self.scale` — spans, drag area, toughness area,
   radiation, reach — so a 20 cm town and a 5 cm room are the same code at two
   settings, not two engines. Walking back into a room then gives you the room,
   rather than a plausible guess at it. Which is exactly what PWM says video
   models get wrong.

The honest summary: cost decoupled from resolution is the right thing to want.
The route is fewer voxels computed, not a model guessing what the voxels would
have said.

## Do not compute where nothing is happening — BUILT, 98 ms to 1 ms (2026-09-11)

Measured first, and the number made the case on its own: a room with **nothing
lit in it cost 98 ms a tick, and a room on fire cost 100**. The whole world was
being swept every tick for heat that was not there, smoke that did not exist and
puddles nobody had poured. On a 316k-voxel two-room world:

| | before | after |
|---|---|---|
| quiet room | 98 ms/tick (10.2 ticks/s) | **1 ms/tick (1237 ticks/s)** |
| burning room | 100 ms/tick | 101 ms/tick |

The design target of >10 runs/sec is met by a factor of a hundred for a world
that is not doing anything, and not at all for one that is. That is the honest
shape of this win and the reason the next step matters.

**The guard was written first, as agreed.** `test_ACTIVE_REGIONS_change_NOTHING`
runs five scenes twice — once with every law sweeping the whole world every tick,
once with the skips on — and compares `mat`, `smass`, `E`, `fl`, `fvol`, `fpot`,
`smoke`, `o2` and every person's state, cell for cell. The scenes are picked for
the ways a skip can be wrong, not for variety: a fire (which heats what it does
not touch), water over the fuel, a person in the room, and — the one that matters
— **a room that starts quiet and is disturbed only at tick 30**, because that is
where a skip becomes a bug. `skip_quiet = False` turns it all off.

**What was actually built is not regions, and that is why it cannot miss
anything.** Each gate asks whether the thing a law moves exists ANYWHERE in the
world, not whether it exists nearby. `self.E` is energy above ambient, so all
zero means every voxel sits at 20 °C: no gradient to conduct, no dT to rise,
nothing hot enough to light, boil or melt, no face carrying a shock. There is no
boundary for an interaction to be missed across, because no part of the world is
being looked at in isolation. Each question is asked immediately before the law
it gates and never hoisted — a fire that lights this tick makes its smoke this
tick, and a stale answer would leave that smoke sitting still.

### The rule for going spatial, before anything spatial is written

Raised, correctly, while this was being built: *"we will have to do some smart
modelling decisions so we don't miss when an actual interaction happens."*

That is the whole difficulty of the next step, and the governing rule is:

> **A skip is keyed to the REACH of what is being skipped, never to a box.**

And the practical form that makes it safe: **wake on the SOURCE, not on the
target.** Whatever injects energy, mass or pressure marks everything it can
reach as awake; a sleeping region is never asked to notice on its own behalf,
because a sleeper that has to check whether it should wake is not asleep.

Reach, per thing, in this sim:

| | reaches | so a sleeper is woken by |
|---|---|---|
| conduction, and the radiation riding on it | ONE voxel a tick — checked: every radiative term here is neighbour-PAIR (`_law_conduct`), plus a per-voxel loss to the outside. Nothing leaps a gap | adjacency, one voxel of pad |
| burning, boiling, melting, cracking | one voxel (they light, heat or read their own neighbours) | adjacency |
| gas — oxygen, smoke, pressure | the whole connected AIRSPACE, room-wide | never freeze by chunk; freeze a whole airspace or none of it |
| falling, toppling, a body losing its footing | the column below, and what it lands on | the support field already answers this |
| sight and earshot (`_law_will`) | as far as the ray or the sound carries | this is the real trap, and it is in the MIND layer, not the matter layer |

So the next increment — a heat window, the bounding box of `E != 0` padded by one
voxel — is provably exact for the matter laws, because heat cannot travel further
than a voxel a tick here. It is worth doing because a town with one fire in it
currently answers "yes, something is hot" and gets no skip at all. What it must
NOT be extended to without more thought is gas and perception, for the reasons in
the table.

### The spatial half, and the wall it runs into (2026-09-11)

Heat travels one voxel a tick and no further — checked, not assumed: every
radiative term is between TOUCHING faces, and the only other path is each
voxel's own loss to the outside, which cannot warm a neighbour. So `_hot_window`
draws a box around everything above ambient, grows it by one, and conduction and
buoyancy run inside it. Provably exact; the guard agrees.

Two derived fields are now cached on the four matter fields they are made of, by
the same compare-the-real-state bargain as the slack field. `heat_capacity()` was
being rebuilt TWELVE TIMES A TICK for something that only changes when matter
does — a fifth of all sim time, burning or not. Porosity rides on the same check.

| | whole world | skipping | |
|---|---|---|---|
| quiet room, 317k voxels | 98 ms/tick | **1 ms/tick** | ~100x |
| burning two rooms, 317k | ~87 ms/tick | 71 ms/tick | 1.2x |
| burning town, 1.02M voxels, 1.7% of it hot | ~340 ms/tick | ~250 ms/tick | 1.3x |

(Medians of repeated runs. A single run came back at 249 and 612 ms for the
whole-world column and was an outlier — worth saying, because one measurement is
not a measurement.)

**And there the wall is, and it is a modelling question rather than a coding
one.** In the burning town the remaining time is gas — oxygen 32%, pressure 18%,
burning 17%, smoke 14% — and gas will not window. It mixes across a whole
connected airspace, and the town's airspace count is **one**: outdoor air really
is all connected, so there is no geometric boundary to exploit. It is not that
the implementation sweeps too much; it is that the model says a fire in one hut
shares air with the whole town, which is true and is also why a fire cannot be
made cheap by drawing a box round it.

So the remaining levers are not spatial:

1. **Decide that distant air is not modelled at the same fidelity.** A modelling
   decision in Ruling 1's stopping-rule family, and the only real answer for a
   world the size of a town. What it costs has to be measured, not assumed.
2. **Kernel fusion** (numba/Taichi). Thirty laws each making several full passes
   over the same arrays; we are memory-bandwidth bound. A CA engine sweeps once
   and applies many rules per cell, which numpy cannot express. This is the
   biggest untaken lever for a BUSY world.
3. `_law_burn` and `_law_crack` are still whole-world. Crack is cheap and already
   guarded. Burn is the genuinely hard one: its oxygen accounting is per-AIRSPACE,
   so it cannot be windowed by geometry for the same reason gas cannot.

## What to take from the references, now (consolidated 2026-09-11)

Every source we have ever listed, re-read against what the code can actually use
TODAY rather than against the queue as it stood when the list was written. Two
things have changed the answers: free rigid bodies exist, and active regions
exist. Ranked by what is unblocked, not by how interesting the source is.

**1. Bullet's ISLAND SLEEPING — zlib, code may be ported.** Now the live one.
Active regions just ran into exactly the problem Bullet solved: what may sleep,
and who wakes it. Bullet's answer is that bodies in contact form an ISLAND, an
island sleeps when everything in it has been below a threshold for a while, and
touching one wakes the whole island — never half of it. That is the same rule
arrived at above the hard way ("wake on the source, not the target"; "freeze a
whole airspace or none of it"), already proven in an engine people ship. An
airspace IS an island. Take the structure and the wake discipline.

**2. box3d's REVOLUTE JOINT — MIT, code may be ported.** The biggest unblock on
the board. Item 24 wants a limb that holds a pose; a shoulder is a revolute joint
with a motor and limits, which is a solved, documented, permissively-licensed
thing. We already have the motor (`arm_Nm` with Hill's fade) and the inertia
(`I = sum m r^2`); what is missing is the constraint that keeps the far end
attached while the angle is driven — which is precisely what sequential impulses
with soft constraints are for. Erin Catto's GDC papers document the whole scheme.
This is the route to the arm that stays where the swing left it, the crouch, and
a man reaching out over a lip.

**3. Rapier's SNAPSHOT/RESTORE — Apache-2, code may be ported.** Cheap here,
because the entire world is a handful of numpy arrays plus a list of dicts. Worth
doing for three separate reasons: the differential guard gets much stronger (diff
two runs at any tick, not just at the end), a TTRPG wants to rewind and try again,
and deterministic replay is how a trace harvest stays honest.

**4. Powder Toy's PHASE QUADRUPLE — GPLv3, ideas only.** `MELT` and `BOIL` are
two mechanisms doing one thing. Their (lowT, highT, lowP, highP) -> target
material table collapses both into data, and we now HAVE a pressure field to key
the pressure half on, which we did not when this was first written.

**5. XCube's SPARSE VOXEL HIERARCHY (VDB) — the data structure only.** Moves from
"someday" to "measure it": a 1.02M-voxel town is 30+ MB of dense arrays swept by
thirty laws, and the busy case is memory-bandwidth bound. But a sparse structure
helps only if most of the world is EMPTY, and ours is mostly solid ground and
air that gas actually uses. Measure occupancy before believing it.

**Already taken, so strike them off:** Noita's promote/demote protocol IS
`_launch`/`_land_body`; Luanti's `falling_node` is the same mechanism; Powder
Toy's air-field update order is `_law_air`.

**Still not yet, and why:** Principia's device vocabulary (no devices layer yet);
Terasology's module structure (belongs to the packs work, not the voxel core);
PWM's scene authoring (the render is not a bottleneck and authoring is where the
thinking happens); The Well (a learned law is a case table); the 3D generation and
reconstruction papers (they make worlds LOOK right, and ours must BEHAVE right).

**The one thing no reference gives us** is the next lever for a busy world:
kernel fusion. Thirty laws each making several passes over the same arrays, when
a CA engine sweeps once and applies many rules per cell. numba or Taichi, and
nobody's source code helps — it is a rewrite of how our own laws are applied.

## Item 25: a world is a value — BUILT (2026-09-11)

`snapshot()` copies everything the world IS; `restore()` puts it back, as often
as you like. The policy is deliberately left out: a mind is not a world, and it
may one day be a model whose weights dwarf the lattice.

It was taken first on purpose, as the tool the next items need — and it paid for
itself before those items were even started.

**The guard it made possible found two real bugs the same afternoon.** The old
differential guard compared two runs at the END, against a threshold relative to
the final values. The new one starts both runs from ONE snapshot — so how the
world was built cannot be the difference — and compares them EVERY TICK, so a
failure names the tick and the field. Both of these had been sitting green:

1. **Heat does not travel one voxel a tick. It travels one voxel per LAW that
   moves it, and two of them run in the same tick.** Conduction carried heat up
   one cell and buoyancy carried some of it up another, one past the edge of a
   window drawn before either ran. The symptom was 3.4 J in a single cell, on
   TICK ZERO, against a field of 2325 — far too small for an end-state check with
   a relative threshold to notice, and a real hole. Fixed by asking for the
   window again between the two laws rather than padding by two once, so it stays
   right when a third heat-moving law is added.

2. **`_law_air` is driven by TEMPERATURE, not only by injected pressure.** Warm
   air makes its own wind — the law reads `T - AMBIENT` straight off the lattice,
   because a plume is buoyancy before it is anything else. Gated on the pressure
   fields alone, the first wisp of smoke a fire ever made came out wrong by
   1.7e-5 on the very tick it appeared, because the world had been skipping its
   buoyant wind since tick zero. The gate looked perfectly reasonable until
   something measured it.

Both are the shape of mistake this whole idea is prone to: a gate that is
obviously right, and is not. Neither was found by reading. The lesson is not
"be careful" — it is that **a skip is only as trustworthy as the thing watching
it**, and the watcher has to compare every tick from a shared start.

Fixing the second one cost speed: `_law_air` now runs whenever anything at all is
warm, which in a burning world is always. The busy-world win fell from about 1.4x
to about 1.3x. That is the right trade and worth writing down as one.

## Item 29 answered for smoke, and item 26 is blocked on a dependency (2026-09-11)

### "smoke != 0" is true almost everywhere almost at once

Windowing the smoke law the way heat was windowed bought nothing, and the reason
is the interesting part. Measured in the burning town:

| tick | cells with any smoke at all | box around them |
|---|---|---|
| 10 | 0 | — |
| 20 | 893,596 of 910,080 | **100% of the world** |
| 69 | 910,080 | 100% |

Diffusion puts an infinitesimal trace in every cell within ten ticks. So "is
there smoke here" is true nearly everywhere and says nothing, while "where the
smoke IS" stays small for far longer. Heat does not behave this way — its box
was 0.5% of the town at tick 10 and still only 4% at tick 69 — which is why the
same trick worked there and not here.

### So the threshold is a MODELLING decision, and it was measured

`SMOKE_STILL = 1e-6` grams of soot in a voxel: below it, the air is not modelled
as carrying it. Share of ALL the smoke that stays inside the box:

| line | box | holds |
|---|---|---|
| 1e-6 g | 28% of the world | **99.9%** |
| 1e-4 g | 9% | 91.1% |
| 1e-3 g | 1.5% | 49.3% |

1e-6 is the knee rather than a round number. Nothing is destroyed below the line
— it stops being carried, so every gram is still there and still counted — and
the tick-by-tick guard still passes, because a cell holding under a microgram
cannot drift by more than a microgram.

This is the first skip in this work that is a judgement rather than a proof, so
it is written where it is made, with its measurement beside it.

### Item 26 (kernel fusion) needs a dependency this project does not have

Checked: no numba, no taichi, no cython, no cupy, and `requirements.txt` says
"Core engine needs only PyYAML". True fusion — one sweep applying many rules per
cell — cannot be expressed in numpy, so item 26 is a dependency decision before
it is an engineering one, and not one to take unilaterally. numba pulls in LLVM.

What is left without it, and where the time now goes in a burning town:
oxygen 32%, pressure 18%, burning 17%, smoke 14%. All three of the big ones are
global for the same real reason — a fire breathes its whole airspace, and the
town's airspace count is one. The same threshold trick could be measured for
oxygen, which is the honest next experiment if the dependency is declined.

Standing numbers after all of this (medians of repeated runs):

| | whole world | skipping |
|---|---|---|
| quiet room, 317k voxels | 98 ms/tick | **1 ms/tick** |
| burning two rooms, 317k | ~90 ms/tick | ~73 ms/tick |
| burning town, 1.02M voxels | ~335 ms/tick | ~245 ms/tick |

## Item 26: numba, and the honest number it bought (2026-09-11)

Taken as an OPTIONAL accelerator: `pip install numba`, and without it the engine
runs on plain numpy and behaves identically. numpy stays the definition of what
every law MEANS; the compiled kernels exist only to be quicker, and a test
(`test_the_FAST_PATH_and_the_PLAIN_ONE_agree`) compares the two every tick on a
world with fire, fuel, water, smoke, oxygen and a person in it, and demands they
match EXACTLY. Not nearly — a tolerance there would quietly license a second
physics.

**Exact turned out to be achievable, which was not obvious.** A float32 sum
reordered is a float32 sum changed, so the kernels keep numpy's order of
operations step for step: the same multiplication order, the same two clip
comparisons in the same direction, and the loss swept over every cell before the
gain is, because that is what `E[a] -= q` then `E[b] += q` actually does. The
result is bit-identical over 60 ticks across E, oxygen, smoke and mass.

Fused so far: conduction (all three axes), oxygen diffusion, buoyant overturn.

**And it bought about 8%.** On a 1.02M-voxel burning town: 215 ms/tick on plain
numpy, 199 ms/tick compiled.

**Why so little, which is the part worth keeping.** The premise of this item was
"thirty laws each sweeping the same arrays, where a cellular-automata engine
sweeps once". That was true when it was written and is no longer a good
description of where the time goes, for two reasons:

1. **Windowing already took the cost out of everything easy to window.**
   Conduction now runs on a box that is 4% of the town, so fusing it saved
   almost nothing — the first measurement was 1.02x. We optimised a law that
   had already been made cheap by other means.
2. **What is left is not neighbour sweeps.** It is coarse-grid block means
   (`_cells`), per-airspace region work, and the two-zone stirring's blurs —
   different shapes, some of which cannot be made bit-identical at all, because
   numpy's `mean` uses pairwise summation and a plain loop does not.

One pure-numpy fix found on the way was worth nearly as much as the compiler:
`_cells` called `np.pad` whether or not there was anything to pad, copying the
whole 4 MB lattice eleven times a tick for worlds whose shape already divides
evenly. Skipping that is free and bit-identical.

**Where the 1.02M town stands now**, against ~338 ms/tick when this session
started: **199 ms/tick**, and a quiet world is 1 ms. The remaining big three are
oxygen's two-zone stirring, burning, and the pressure field — all global because
a fire breathes its whole airspace, which is item 29's question and not a
compiler's problem.

**Is the dependency worth keeping?** It is optional, it is locked to the numpy
path by a test, and it is 8%. Worth keeping only if more of the engine gets
fused later; worth dropping without argument if the dependency is ever
inconvenient, and nothing but speed changes.

## Item 27: an airspace is an island — BUILT, 4x on a building (2026-09-11)

Bullet's rule, and it is the right one here for the same reason it is right
there: **the thing that may sleep is a whole connected body, never part of one,
and anything that disturbs it wakes all of it.** Gas mixes room-wide, so a room
is the smallest thing that can be called still.

**Where it bites.** `_law_o2`'s two-zone stirring looped over every airspace and,
for each one, blurred the WHOLE lattice three times per zone. That is
O(rooms x world) — so a building got dramatically more expensive per voxel than
open ground, which is precisely backwards. A 420k-voxel building of twenty sealed
rooms cost more per tick than a 1.02M-voxel town.

**The rule.** A room every cell of which holds fresh air, no soot and no warmth
has one empty zone and one uniform zone, and relaxing a uniform thing toward its
own mean moves nothing. Two comparisons and a count decide it, once, for all
rooms at a time. Measured with a fire in one room of twenty: **nineteen asleep,
holding 95% of all the air in the building.**

| 420k voxels, 20 sealed rooms, one alight | |
|---|---|
| whole world | 311 ms/tick |
| skipping | 87 ms/tick |
| skipping + compiled laws | **78 ms/tick** |

**4x**, and it is the case that matters: a building is the shape this engine is
for. The town benchmark could never have shown it, because a town's outdoor air
is ONE airspace — which is why measuring the right scene mattered more than
writing the right code.

The guard gained a scene for it: two rooms with no way between them and a fire in
one, so the far room is an airspace nothing has disturbed — the case the rule is
allowed to sleep through and must not get wrong.

### Standing numbers, end of the speed work

| | before any of it | now |
|---|---|---|
| quiet room, 317k voxels | 98 ms/tick | **1 ms/tick** |
| building, 420k, 20 rooms, 1 alight | 311 ms/tick | **78 ms/tick** |
| burning town, 1.02M voxels | ~338 ms/tick | **~199 ms/tick** |

The town is the weakest of the three and for an honest reason: one airspace, one
fire in it, and gas that really does mix across the whole thing. Nothing in the
remaining work makes that cheap — only item 29's modelling decision would.

## Item 28: a body that can hold a pose — BUILT (2026-09-11)

Item 24 renamed, with a proven shape behind it. A shoulder is a revolute joint
with a motor and a limit (box3d, MIT); what is taken is the SHAPE of that — an
angle, a speed cap, and a torque the motor either has or has not — rather than
the code, because the solver underneath ours is a lattice and not a contact
graph.

**What was wrong before.** A limb could only be FLUNG. `_swing` promoted it off
the lattice, turned it as a free body, and put it back at REST, so an arm could
never simply BE somewhere. Three things were waiting on that: the arm that stays
where a swing left it, a crouch, and a man leaning out past his own toes.

**What is built.** `p["pose"][limb]` is an angle that persists; `_repose` puts
the limb's voxels where that angle says, ON the lattice, moving them from the
cells they are in to the cells they should be in. `_law_pose` advances each angle
toward what the body asked for, at `arm_wmax` — the same shoulder speed Hill's
relation already caps a swing with — and refuses to hold an angle whose lever
asks more of the muscle than it has. `reach` and `pull the arm back in` are menu
rows like any other.

It is not animation, and the test is the same one that settled the swing: there
is no picture here to change. **A wall stops it** — measured, an arm reaching
into stone gets as far as the stone and no further, and takes no bite out of it.
What the hand holds turns with the hand, which is the same rotation about the
same joint and most of what holding a thing is for.

**And the lattice gets a say, which was the surprise.** A limb ONE VOXEL WIDE
can only be drawn at a handful of angles: at the rest of them two of its voxels
round into the same cell, and moving would destroy a voxel of flesh. Measured on
a nine-voxel arm, 21.5 degrees is undrawable while 20 and 25 are fine. So the
angle runs on smoothly and the flesh catches up at the next angle that CAN be
drawn — the arm moves in clicks. That is not a workaround for a coarse grid;
it is what a thin thing turning on one IS, and it is the same honesty as
declaring an edge rather than pretending to draw it.

Measured: an arm goes from hanging to fully out in about four ticks (a tenth of
a second, which is a shoulder's pace), holds there indefinitely, keeps all 27 of
its voxels and every gram of the body, and stays attached.

### What did NOT land, and why it is worth saying

The reach was meant to close a loop from two days ago: a load hanging from an
extended fist has a long lever, so "he leaned out after him and went over too"
should become possible. It does not yet, for two reasons found by trying:

1. **An arm cannot sweep through what it is reaching for.** Standing beside the
   man he is holding, the holder's arm is blocked by that man — correctly — so
   he lowers it again. Reaching out works when there is somewhere to reach INTO.
2. **The thing has to be hanging from the fist, and `_take_up` did not lift it**
   in that geometry, so it sat on the ledge being dragged rather than carried.
   A load resting on the world weighs on its own footprint, so the balance sum
   rightly ignored it.

Neither is a flaw in the pose model; both are the next thread. And the reach a
rescuer actually needs is not this one anyway — it is leaning the TORSO out over
a lip, or crouching, or lying down. The arm alone cannot do it, because at rest
the arm already hangs straight down and there is nowhere further down to go.
That is the honest shape of what a shoulder buys and what it does not.

## Item 28's payoff, landed — and two bugs it needed first (2026-09-11)

A body that can hold a pose was built so that a man could make a LEVER out of
his own arm. Getting there took two fixes, and both were the same kind of
mistake: a rule that was obviously right and quietly wasn't.

1. **`_take_up` put the load CENTRED on the hand.** So anything wider than a
   hand reached back into the chest of whoever was holding it, and the move was
   refused — for the perfectly good reason that a man was already standing
   there. Measured: a 38 kg block that a 4000 N man could obviously lift stayed
   on the floor being dragged. It hangs beside the fist now, on the side the arm
   is on.

2. **What you carry came back as part of what carries you.** A man holding a
   block has his arm directly above it, so `_contact` counted the block as part
   of his own FOOTPRINT — which made his base as wide as his reach and said he
   could never be overbalanced by anything he was strong enough to hold. The
   block widened his footprint to exactly cover his own fist, so the overhang
   was always zero. `_contact` takes an `ignore` now, and the balance sum passes
   it what he is holding.

**And the arm comes down.** Overbalanced, a man was launched, landed still
holding the thing at arm's length, and was overbalanced again — every tick,
for ever. You cannot keep a thing at arm's length while it is taking you over,
so the reach is the first thing to go. He goes over once.

**The pair of tests this leaves** are the two halves of one moment, and neither
could have been written a day ago:

- `test_the_SAME_WEIGHT_at_ARMS_LENGTH_takes_a_man_off_his_feet` — the mass
  held still, the ARM moved. 38 kg at his side: nothing. The same 38 kg on the
  end of an arm: over he goes, once.
- `test_WHAT_YOU_HOLD_UP_STANDS_ON_YOUR_FEET_TOO` — the arm held still, the
  MASS moved. 64 kg at his side: he carries it. 128 kg: over he goes.

Measured on the way: at his side this man carries anything up to 64 kg without
trouble, and at arm's length 38 kg is already too much. Nobody wrote either
number; they are his weight and his build against the load's weight and its
lever.

## Item 33: oxygen windows EXACTLY, and needed no judgement (2026-09-11)

The opposite result to smoke's, and the contrast is the useful part. Air all at
one composition has nothing to even out — every pair cancels — so the box around
where the field is NOT fresh contains every exchange there is, and the skip is
provable. No stated threshold, no modelling decision.

It works because oxygen's deficit stays where the fire is, and smoke's does not.
Measured in the same burning town:

| | tick 20 | tick 69 |
|---|---|---|
| box around any smoke at all | 100% of the world | 100% |
| box around any oxygen deficit | 21% | 63% |

Diffusion carries an infinitesimal trace of soot everywhere within ten ticks, so
`smoke != 0` is true almost everywhere and says nothing. Oxygen is consumed
rather than produced, so its deficit is a hole that spreads slowly from one place
instead of a trace that reaches everywhere at once.

**Honest about the payoff:** it helps a fire that has just started and stops
helping as the deficit spreads. Measured on the town — a fire just lit, 186
ms/tick with the box at 28%; well alight, 215 ms/tick with the box at 67%. Kept
because it is exact and degrades to a no-op, not because it is a headline.

Also fixed while windowing it: the overturn moves oxygen AGAIN in z after the
diffusion has, so it asks for its own box — the same mistake the heat window
made on its first day, caught this time before it could be made.

## Item 37: a roll — BUILT, and it was the oldest open item (2026-09-11)

Written as item 1 on 2026-09-07 with a strict-xfail test waiting for it. That
test now passes, and **the suite has no xfails left at all.**

**A ROLL IS NOT ONE LANDING.** It is several, each taking a share of the fall on
a different part of you — feet, then hip, then back, then shoulder. Tissue damage
is a THRESHOLD, so a blow divided into parts that each land under it does no harm
at all, while the energy is unchanged and every gram is still there. That is
force being momentum divided by the time taken to lose it, written in the
arithmetic this model already had.

How many parts is READ OFF THE BODY: how many times its own crouch goes into its
own length, which is how far a body rolling over itself travels before it stops.
About six, for this body. Nothing typed in.

| fall onto stone | landed rigid | rolled |
|---|---|---|
| 0.5 m | nothing | nothing |
| 1.5 m | bruised (0.08) | nothing |
| 3 m | knocked out (0.30) | walks away |
| 4.5 m | knocked out (0.52) | walks away |
| 6 m | nearly dead (0.74) | a scratch (0.02) |
| 8 m | DEAD | out cold, alive |
| 11 m | dead | dead |

**And it is not a get-out-of-jail card**, which the test now guards: dividing a
blow only helps while the parts land under the threshold, and past that the
arithmetic runs out on its own. Rolling buys about one band — you survive roughly
twice the height, and are knocked out where you would have been killed.

### The other half: a falling body could not decide anything

To choose a roll you must choose it while FALLING, and the will layer skipped
anyone in the air — a flying body has no cells on the lattice for it to read, so
it was passed over entirely. That quietly made falling the one thing in this sim
nobody could do anything about.

`_decide_falling` gives it the same seam as every other decision: a short menu of
what is really possible (there is nothing to walk on and nothing to push
against), a policy that picks an index, and the whole menu written to the trace
beside the pick. It is also the thing item 4 (catching something in flight) has
been waiting for.

### Deciding in mid-air is an INTERFACE question, not a physics one (2026-09-11)

Raised on seeing `_decide_falling`: "picking an option while jumping is just a
question of how we provide an interface — let a character act on previous or
predefined behaviour, and let the player press a button to give the next input,
like roll, while the character is in the air."

That is right, and it is worth writing down because it says what the sim owes and
what it does not. The sim's job is to OFFER: while a body is falling there really
are only two things it can do, and both are on the menu with the whole menu
written to the trace. Who answers, and when, is the layer above:

- **a standing preference** — "always roll if you can" on the character sheet, so
  a body that is never asked still behaves like itself;
- **a queued input** — what the player last said, still in force;
- **a live interjection** — a button pressed during the flight, which is only
  meaningful because a fall takes real ticks (a 3 m drop is about thirty of
  them, and the decision window is every `WILL["decide_every"]`).

None of those is a change to the sim: `policy.pick(situation, menu)` already
takes all three, because it only ever asked for an index. The one thing the sim
had to fix was that it was not asking at all while a body was in the air.

It also sets the shape of the real-time question later on: a menu that appears
and expires is a prompt, and how long the player has to answer it is the tick
budget, which is measurable rather than a matter of taste.

## Item 31: a sparse structure would save nothing here — MEASURED (2026-09-11)

The idea was XCube's sparse voxel hierarchy (VDB): store only the non-empty part
of a big world. Measured occupancy before believing it:

| | voxels | air | solid |
|---|---|---|---|
| the town | 1,024,000 | 87.6% | 12.4% |
| the building | 420,000 | 48.2% | 51.8% |

**And that settles it, because our "empty" is not empty.** The air carries
oxygen, smoke, pressure and temperature, and the gas laws need every cell of it;
the solid is what the matter laws need. Between them, every voxel in the world is
wanted by some law. A sparse structure pays when most of a world is genuinely
nothing, and in a sim whose whole subject is fire and air, nothing is nothing.

It would only start to pay after item 29's decision — if distant air stops being
modelled at the same fidelity, the part that stops being modelled becomes real
empty space, and then a structure that skips it means something. So this is not
"no", it is "not until a modelling decision makes it true".

## Item 34, part one: the air field a fire reads was rebuilt every tick

`_law_burn` downsamples the whole lattice to find where the air is, then blurs it
wide, to work out what a plume can draw on. That field depends on nothing but
WHICH VOXELS ARE AIR — not on the oxygen in them, not on the fire.

Cached, and thrown away by `_air_regions`, which already compares this tick's air
mask against last tick's and so already knows when walls have moved. No extra
comparison, and right by construction rather than by anyone remembering that a
burning wall changes where the air is.

**Keyed on the matter fields first, which was the obvious thing and the wrong
one:** a fire eats wood, `smass` changes every tick, and the cache was thrown
away every tick for a field that had not moved. Worth writing down because it is
the third time this session a cache has been keyed on more than it depended on.

Burning town, 1.02M voxels: about **200 ms/tick**, from ~338 at the start of the
day. The rest of `_law_burn` is still whole-world and probably has to be — its
oxygen accounting is per-airspace, for the same real reason gas is.

## What a step up cost — item 5 TRIED, and taken back out (2026-09-11)

Walking is flat: every column is tested at the body's own height and no other,
so a kerb is a wall and a stair is a wall. Making it height-aware was built,
worked, and was reverted. Writing down why, because the reasons are worth more
than the code was.

**What was built and did work.** `_walkable` asking each column for the highest
floor within a step of the feet; `_walk` lifting a step when the flat one is
blocked; `BODY["step_m"] = 0.30` as a declared property of a body, the same
class of fact as its strength. Measured: a body walked up a kerb of 5, 10, 20
and 30 cm to a door on the raised half, and was stopped by 40 cm — the declared
limit doing exactly what it says, with nothing anywhere deciding what a kerb is.

**What it could not do.** A STAIRCASE. The planner sees one step's worth of
height, because the mask is a single plane: treads at 20, 40, 60 and 80 cm are
one tread visible and the rest a wall. Doing it properly means routing in 2.5D —
a standable surface per column, edges between neighbours within a step — which
is a rewrite of `_fit_grid`, `_routes`, `_plan_path` and `_places`, not an
addition to them.

**And four real things it exposed, each a bug that was already there:**

1. **A body climbed its own shins.** Its own flesh is solid, so the floor test
   under its own feet found its own legs, and it levitated up a wall it had
   walked into, a voxel a tick.
2. **A body that steps up could never get down.** Allowing only up-steps left a
   man stranded on the first thing he climbed — measured, standing on a burning
   crib, feeling it at 57 °C, deciding to leave and having nowhere to go, while
   his skin went to 135 °C.
3. **A body could not FEEL a fire.** Sight needs a clear line from the head in
   the direction the head is turned, so a body standing ON a fire did not
   perceive it: the flames were under its own feet and behind its own legs. KEPT
   — `BODY["feel_T"] = 45` and a `scorched` percept, below the temperature that
   marks tissue, because you feel a fire well before it burns you and that is
   the whole use of feeling it.
4. **A rescuer picked up a burning stick.** A character sheet saying
   `hands: hold` matched "take hold of Fallen" and "take hold of the wood"
   alike, so whichever came nearest won, and a man who had come to drag someone
   out stood holding a brand until the smoke took him. KEPT — a thing is `take`
   and a person is `hold`, so the sheet means what it says.

**Why it came out.** Not geometry. Making a woodpile walkable changed which
column an IDLE body wandered to, and that reshuffled a calibrated scene into a
man strolling into a fire. The roam target is picked by
`(tick // decide_every + len(name)) % len(far)` — a deterministic stand-in for
taste with no preference in it at all, so any change to the set of reachable
places is a coin flip over where a body goes.

So item 5 is not really blocked behind 2.5D routing. **It is blocked behind item
13** — a policy with taste, that would rather not walk toward a fire. Until
something is choosing, every scene is balanced on an arbitrary modulo, and
widening the world's walkable set will keep knocking scenes over.

## The plainest taste there is — items 13 and 14, started (2026-09-11)

Came straight out of the step-up failure, which ended with: "item 5 is blocked
behind item 13, not behind geometry." So item 13.

**Belief recorded geometry only.** `p["known"]` said a column had been laid eyes
on and nothing about what was in it — so a body could walk through a burning room
and remember its SHAPE and not the fire. When it came to choose somewhere to go,
every place it knew looked equally good. That is why an idle body would stroll
toward a blaze: not bravery, nothing it remembered said otherwise.

`p["danger"]` now records where it saw flame, read by COLUMN rather than at eye
height — light from a fire at your feet still reaches your eyes, and whether you
could see that far is the occlusion the ray already settled. **And it fades**
(`WILL["forget"]`, about a minute to half-weight): a body that never forgets
treats an hour-old fire as a fire, one that forgets at once has no memory at all.
Forgetting is a model, not a leak.

**Every `go` row now carries how bad the way there looks** — the worst step on
the route, which is the right question and a cheaper one than asking about the
destination, because the route is already in hand. The SIM states the fact.
Minding it is the policy's business, and a policy with no taste gets the same
menu.

**And the table policy minds it.** It used to take the FIRST row matching the tag
the reflex named, so which door a fleeing body ran for, and which way an idle one
wandered, came down to the order `_places` happened to build its list in — a
modulo of the tick. **A policy that prefers nothing is not neutral; it is
arbitrary, and arbitrary is not a thing a body does.** Now: least dangerous
first, nearest to break the tie.

Locked by two tests. One is the sim end: a witness remembers the fire at x4-10
y8-16, remembers the far corner as fine, and forgets a fire that has gone out.
The other is the seam itself, with a menu built by hand — the longer way round
beats the short way past a fire; with nothing to choose between them, the nearer
one wins; and an idle body does not wander toward a fire it remembers, which is
the whole of what was wrong.

**What this does NOT do.** There is still no model at `pick`, no exploration
slots, no uncertainty gating, and menu RECALL is still unmeasured — a capped menu
can still drop the right option invisibly. And danger is the only thing belief
records; a body cannot yet remember where the door was, only that it saw it.

## What attention was costing — RECALL measured, 81% then 100% (2026-09-11)

`MENU_CAP` is a model of attention rather than a budget for whatever is
choosing: a person in a burning room weighs the door, the window and the child,
not the forty places a full legality sweep would list. Modelling the limit is
more true than pretending it is absent.

But a limit that quietly removes the thing a body would have done is not a model
of attention — it is a bug with a comment on it. Nobody knew which this was,
which is why it sat on the list as "recall matters most, because a capped menu
can drop the right option invisibly".

**So measure it.** `World.recall_check` (off by default, it doubles the work of
deciding) builds the menu the body would have had with attention free, asks the
SAME policy, and records what it would have done. Recall is how often that is
what it actually did.

| | |
|---|---|
| legs decisions that hit the cap | **57%** |
| most options ever offered at once | 13, against a cap of 7 |
| recall, before | **81%** — one decision in five, the cap hid the answer |
| recall, after | **100%** |

**What was wrong.** The cap kept the nearest rows, and every roam spot is
further off than a step — so it cut all four of them at once and a body could
not choose to wander at all, however much it wanted to. The measured miss was
exactly that: picked `go(straight on)`, would have picked
`go(the floor south-west)`.

**What fixed it is a better model, not a bigger cap.** ONE OF EACH KIND first,
then the nearest of what is left. A body notices categories before instances —
that there is a door, that there is somewhere it has not seen, that there is a
place it could go — and then the nearest of each.

And it keeps nothing because the policy would prefer it. **Salience is not
preference**, and the distinction is what lets the cap stay in the sim while
taste stays in the policy: a fire is worth noticing whether you mean to run at
it or away from it.

`traces` now carry `offered` (what a full sweep found) beside `menus`, so the
cost of attention is on the record for every decision a harvest ever reads.


## The floor a body remembers — the last god-channel in the planner (2026-09-12)

Item 14 said belief recorded geometry only. It was worse than that, and the bad
half was hiding one layer down from the half anybody could see.

`p["known"]` is where a body has LOOKED. That gate was real and it did its job:
a stranger cannot aim at a door it has never laid eyes on. But the planner did
this:

    walk_ok = self._walkable(cells)   # THE LATTICE, right now, everywhere
    walk_ok &= known                  # masked to where the body had looked

So `known` said WHERE, and the world itself said WHAT — live, every tick. A
body's route ran over a floor it was not looking at. The door register had been
taken away from minds months ago; the FLOOR under it never was.

**What that cost, measured.** Two-room house, a resident who knows the
building, a door in the far room behind a baffle. Wall off the way to that door
— two rooms away, through a doorway, with nobody within sight of it:

| | exit rows the instant the far wall goes up |
|---|---|
| reading the floor live | *(none)* |
| the floor it remembers | `the door east` |

Every body in the building re-planned around that wall on the next tick,
without one of them turning its head. A blind man in the cellar knew.

**The fix is the same shape as the door register.** `p["free"]` — what the eyes
FOUND where they went, filled in by `_look_around` from the same rays that fill
in `known`, and it is what `_fit_grid` runs on. `_fit_grid`, `_routes` and
`_plan_path` now take the PERSON rather than a bare `known` array, which is
also the honest signature: a route is somebody's, or it is the sim's.

Now the resident keeps walking toward a door it cannot reach, crosses two
rooms, comes within sight of the wall, and finds out — **132 ticks committed to
a route that no longer existed.** That is not a body being stupid. That is the
only way a body could possibly know.

**It does not fade, and danger does.** A fire is an event and a floor is a
fact: the useful error is a body that trusts a way it can no longer take and
finds out by going, not one that forgets the shape of a room it was standing in
a moment ago. Two memories, two half-lives, and the difference is a claim about
the world rather than a tuning knob.

**The shin is a sense organ.** A remembered-clear column that is now blocked
would be re-planned identically the moment the legs jam on it, forever. So a
failed step writes belief: you find out the corridor is shut by shutting your
shin in it. `_walk` fails on exactly three things — a packed cell, a real
puddle, the world's edge — and all three are contact, so this is a percept and
not a peek. And it self-heals: a person standing in the way is recorded as
blocked and cleared again by the next look in that direction, because the body
is already facing the thing it just walked into.

**And the first thing a remembered floor got wrong was the LOOKER.** Flesh is
denser than anything a body can shove through, so every look wrote the ground
under its own feet down as blocked — and then the body walked on, and nothing
ever looked back to correct it. Measured on an empty 4 m hall: **109 phantom
walls**, laid along its own path, every one of them believed, and the first of
them exactly where it had been standing when it first opened its eyes.

The planner already had this rule (`_fit_grid`: "a body does not block
ITSELF"). Belief needed it at the moment of LOOKING, and that is the general
shape of the bug: the one thing guaranteed to be in front of a body's eyes is
its own body, so any map built from looking has to say what a looker is.
109 -> 0. Test: `test_a_body_DOES_NOT_WALL_ITSELF_IN_behind_its_own_back`.

**What it costs.** A look now reads the floor once (it never did before).
Measured on a 100x80x50 hall: one `_walkable` pass is 0.79 ms, a look happens
every 4 ticks per person, so four people add 0.80 ms to a 472 ms tick — **0.2%**.
Nothing to optimise, and worth recording because the obvious worry (belief costs
a full-lattice pass now) is simply wrong at these sizes.

**The general lesson, third time now.** A fact reaching a mind through a
DERIVED quantity is still the sim wearing a person's face. The exits were
obvious. The floor under the exits was not, because the code that read it was
called `_walkable` and looked like physics. Ask of every field a mind reads:
*who looked?*

Tests: `test_a_body_PLANS_OVER_THE_FLOOR_IT_SAW_and_finds_out_by_going`,
`test_WHAT_YOU_WALK_INTO_you_learn`.


## What an elbow bought, and what it exposed (2026-09-12)

Item 36 wanted one thing (an arm that goes round an obstacle) and the build
gave three: a general chain, a bug that had been hiding behind having only one
joint, and a measurement that says the lattice is the limit rather than the
joint model.

**A limb is now a CHAIN of bones.** `p["chain"]` is declared by whoever built
the body — `{"right arm": ["right upper arm", "right forearm"]}` — and
everything outside the scene still asks for "right arm" and never learns how
many bones that is. A chain of length one is the old behaviour exactly: the
twelve arm and pose tests passed unchanged before the humanoid was split, and
again after.

The kinematics are three lines, because all of a limb turns in one plane. Each
joint's turn is written as a 2x3 affine — the same rotation `_body_pose` has
always done — and a bone's cells are carried by its OWN joint first, then by
every joint above it. Composing in the REST frame is what saves tracking where
the shoulder has dragged the elbow to. Verified against `_body_pose` cell for
cell, both axes, both signs, five angles.

**A bare number still means the shoulder**, with the rest held straight, which
is what every existing caller meant. That is why `reach out` did not have to
change, and why the menu did not either: reaching is a CHOICE, and which way
the bones get there is motor competence — the same distinction `_law_walk`
already draws when it says knowing the way round a table is not thinking.

**An arm that meets something BENDS.** `_reach_around` searches the elbow, least
bend first, when the straight pose is refused by the world; it may also hold
the shoulder where it is rather than where it was asked, because getting the
hand somewhere matters more than what the shoulder reads — that is the
difference between a joint and a dial. Fine steps, because drawability on this
lattice is SPIKY and a coarse ladder walks straight past the angles that fit.

### The bug an elbow uncovered: two voxels rounding APART

`_repose` has always refused a pose where two voxels round into the SAME cell,
because that would destroy flesh. It never asked the opposite question. A line
of voxels turned to anything but a right angle rounds to a **staircase**, and a
staircase touches only at its corners — which, in a sim that decides what a
THING is by 6-connectivity, is not one limb. Measured: an arm bent 1.2 rad came
out **in two pieces**, every gram present, no longer an arm.

One joint never showed it because one joint has almost nothing but right angles
to work with. `_still_joined` is the twin check, and the answer it gives is the
one the lattice always gave: not drawable, hold the last good pose, catch up at
the next angle that agrees.

### What it bought, counted

Hand positions reachable — cells the hand can actually be put in, over the whole
range of the joints, drawable AND joined:

| arm thickness in the plane it turns in | one bone | with an elbow |
|---|---|---|
| **1 voxel (what the humanoid has)** | **2** | **7** |
| 2 voxels | 5 | **30** |
| 3 voxels | 5 | 9 |

Two things fall out of that table, and the second is the more important.

**A one-boned man at 5 cm has TWO poses**: hanging down, and straight out.
Everything in between rounds to something that is not a joined arm. An elbow
makes it seven. So the elbow is not a refinement — without it the arm is a
switch.

**The binding constraint is limb THICKNESS, not joint count.** At two voxels
the elbow buys 30 places instead of 7. At three it collapses again to 9,
because a thicker arm runs into the body and into itself. And it is
ANISOTROPIC: our arm is 1 voxel across in x and 3 in y, so a man facing east
and the same man facing north do not have the same poses available. That is a
lattice artifact wearing a body, and it is now written down (item 41) rather
than discovered again later.

Not changed today, on purpose: widening the arm in x changes the walking
footprint, and the humanoid's whole point is that "every doorway in every scene
is unchanged". That is a scene-wide decision, not a drive-by one.

### Three things that were only ever true of an arm hanging down

Splitting the arm did not break these. It made them visible, which is the
useful thing a change of this size does.

**The FIST was the arm's mean.** `_fists` read a hand as "the mean x and y of
the arm, at its lowest z" — exactly right for an arm hanging straight down and
exactly wrong for one held out, because a horizontal arm's mean is its ELBOW.
Measured: a man holding 38 kg at arm's length had the load applied half way up
his forearm, which halved the lever and let him keep his feet under a weight
that should plainly have taken him off them. A hand is now **the voxel of the
last bone furthest from the joint the limb hangs from** — true hanging, held
out, or bent.

**A held thing was CLAMPED to the arm, not hanging from it.** `_repose` turned
the load rigidly about the shoulder, so putting an arm out swung a 38 kg block
from below the hand to above the head. A thing held in a fist hangs: it follows
where the hand goes and keeps its own attitude, because gravity has a say in
which way up a carried thing is and the wrist is a joint whether or not this
sim models one yet.

And it is carried by **where the hand IS, not by what the angle reads**. The two
part company at every undrawable angle: the flesh holds its last good pose while
the angle runs on, then catches up several angles at once. Carried by the change
in ANGLE, the arm went out eight voxels and the block moved one.

**An arm came down at a shoulder's pace after its owner was knocked over.**
Holding a limb out is work a standing body does, and a body going over has
stopped doing it — the arm falls. Unwound a tick at a time (and waiting at every
undrawable angle on the way), the man landed with the weight still out and went
over a second time, and a third. It now goes home in one move.

All three had one shape: a rule written when there was only one pose to write it
for.

Tests: `test_an_ELBOW_puts_the_hand_where_ONE_BONE_never_could`,
`test_an_ARM_BENDS_round_what_it_cannot_reach_THROUGH`,
`test_a_POSE_that_rounds_a_limb_APART_is_not_a_POSE`.


## An arm went through a wall, and the wall was unmarked (2026-09-12)

Found while probing the elbow, and it had been true of every reach since poses
existed. Only the pose at the END of a tick was ever checked against the world.

That is honest while a limb turns a little at a time. It stops being honest the
moment it does not — and it does not, constantly, because the flesh waits at
every angle the lattice cannot draw and then catches up several angles at once.
So a wall sitting in the undrawable part of a sweep was never touched by
anything. The arm was on one side of it, and then it was on the other.

**A waist-high wall is exactly that shape**: the arm CLEARS it once horizontal,
so the final pose is perfectly legal, and the only way to that pose is through
the stone. Measured, with the sweep turned off:

| wall standing to | where the hand got to | wall |
|---|---|---|
| z20 | **x21** — the far side | whole |
| z24 | **x21** — the far side | whole |
| z26 | **x21** — the far side | whole |

Nothing was destroyed, nothing was displaced, no law was broken. The arm simply
was not in the room for the part of the journey that mattered.

**The fix is a swept volume**, at `_SWEEP_RAD = 0.08` rad — about three
quarters of a voxel at the end of a nine-voxel arm, so it cannot step over a
wall. The samples do NOT have to be drawable: two voxels rounding together is
an artifact of drawing, not an event in the world. They only have to be
unoccupied. That is the same rule a walking body has always kept — you may not
arrive somewhere by passing through something.

It swept from `drawn` rather than from `pose` for the same reason the held load
does: the angle and the flesh part company at every undrawable step, and it is
the FLESH that has to get there.

Test: `test_an_ARM_CANNOT_SWEEP_THROUGH_what_it_would_CLEAR_at_the_end`, which
turns `_SWEEP_RAD` off and checks the arm really does walk through the wall
without it — a guard that is worthless if the thing it guards stops being true.

**And a scene lesson, twice in one day.** Both times I built an obstacle
FLOATING — a rail at z20-26 with nothing under it, posts starting at z23 — and
both times the support law quietly put it on the floor before the probe looked.
The first reading said the arm reached past a wall; there was no wall there any
more. In a sim where everything obeys the same laws, THE SCENE OBEYS THEM TOO:
a probe's furniture has to be built to stand up, or the probe is measuring a
different world than the one it describes.


## A question already answered (2026-09-12)

Item 43 looked like trace noise. It was a break in the menu's one guarantee.

**A picked option must be one that can happen.** That is the whole value of the
seam — whatever is choosing cannot invent a door, because it never sees one that
is not there. A man pinned against a wall was breaking it: he decided to reach
out, the world refused, the intention was dropped, and a few ticks later the
same option was on the same menu again. For ever.

| | reach picks, 400 ticks |
|---|---|
| no memory | **14 of 14 decisions** |
| it remembers | **1 of 14** |

Every one of those fourteen went into `traces` as a decision that changed
nothing — exactly the kind of row a harvest must never learn from, because it
teaches that reaching at a wall is what bodies do.

So a body remembers that it could not, **against the spot it was standing on
and the way it was facing**, and forgets the moment either changes — because
then it is a different question about a different wall. The same shape as the
shin learning a corridor is shut, and the same shape as `p["free"]`: a thing
found out by trying, held only as long as it is still the same question.

One trap on the way in. The memory is made when a reach is refused, and the arm
then unwinds to rest — and unwinding ENDS in a successful move, to angle zero.
Clearing the memory on any successful move therefore cleared the memory it had
just made, every time. Coming back to rest is not getting somewhere.

**And the second bug, which the first was hiding.** The menu read the INTENTION
— "is a reach wanted" — and called that "is the arm out". They part company at
exactly the moment this item is about: the world says no, the wanting stops,
and the flesh is still out there. A man with his arm stuck half out was offered
the chance to reach out and never once the chance to bring it down. It reads
`pose` now — where the arm IS, not what the body meant.

That is the third time today the same correction has been needed: the FIST read
an intention of a pose, a held load rode on an angle instead of a hand, and now
a menu row read a wish instead of a limb. **Ask the flesh.**

Tests: `test_a_REFUSED_REACH_is_not_asked_again_until_something_CHANGES`
(which turns the memory off and checks the old behaviour really was 14 of 14),
`test_an_arm_LEFT_OUT_is_offered_the_way_BACK_IN`.


## A spine bends, it does not swing (2026-09-12)

Item 35's first half. Three things had to be true before a man could bend, and
only one of them was about leaning.

### A torso cannot be ROTATED at all

The elbow's machinery was in place and it refused every lean, correctly. A
rigid rotation of a SOLID slab is never injective on a grid: at every angle
some pair of its voxels rounds into one cell, and `_repose` is right to call
that undrawable, because a body may not pay mass to change its shape. A thin
arm has a handful of angles that work. A twelve-voxel torso has none.

So the model was wrong, not the check. **A lean is not one bone swinging.** A
trunk is a stack of vertebrae, and bending it is each slice sliding forward
over the one below — a SHEAR. That is what the thing actually is, and it has
the property the lattice needs for free: every row moves by one constant, so
within a row it is a translation of integers, and rows never meet because their
height does not change. Nothing can round into anything, at any angle, ever.
What it gives up is the `cos theta` shortening of a real bend, which is second
order in the angle.

`p["bend"]` names the joints that shear rather than turn; a hinge is still a
hinge. Drawable from 0 to 0.8 rad, mass exact, one lump throughout. At 1.0 rad
the shear tears the body and `_still_joined` says so.

### Turning a joint carries everything BEYOND it

Which means the body is a tree, not a list. `_repose` walks one now. A limb's
CHAIN already says which bone hangs from which — the forearm from the upper arm
— so parentage is read off there rather than declared twice, and `p["parent"]`
says only what no chain covers: the head and the arms hang from the torso.

The legs are deliberately NOT children of the torso. Leaning bends a man at the
waist and leaves his feet where they are standing, which is the whole point of
a lean.

### Balance was only ever asked about what a body CARRIES

Nobody noticed for months, because a body was one rigid block and a block
standing up straight has its weight over its feet by construction. Give it a
waist and the hole opens at once: measured, a man bent 46 degrees with his head
**fourteen voxels past his toes** stood there indefinitely.

`_overbalanced` asks the plainest question there is — is this body's own weight
past its own feet — and `_law_pose` uses it as a gate beside the one the muscle
already had. **A pose you cannot KEEP is not a pose you adopt.** So a body leans
as far as it can stand and holds there:

| | |
|---|---|
| centre of mass crosses the toes at | **0.22 rad** |
| where a body asked to "lean out" settles | **0.188 rad (10.7 degrees)** |
| what the back is doing there | 18 N.m of the 200 it has |

The angle is written down nowhere. It falls out of where this body's mass sits
over this body's feet, so a heavier head or a longer foot gives a different one.
And the BACK is not what stops it, which is the right answer: leaning forward
is a balance problem long before it is a strength problem.

**Only a body that is standing up can be taken off its feet**, and that is said
as geometry rather than as a flag: a standing body is taller than it is long.
Without it, a man lying unconscious across a floor rested on the nine voxels of
him that touched it with his centre of mass fourteen voxels away along his own
length — which reads exactly like a man leaning too far, and the sim threw him
across the room. What should happen to the parts of him that are over nothing
is that they SAG, and that is the support law's question.

### And a waist is a part of the body, so it gets a menu

`LIMBS` is four now. The null act KEEPS what the body is doing, because a trunk
takes many ticks to bend and the body has to be able to go on bending;
straightening up is an act of its own. Made the null act "stand up straight", a
body cancelled its own lean on the very next decision and never bent at all —
the same shape as hands keeping or letting go, and for the same reason.

A trunk is also SLOWER than a shoulder (`waist_wmax`, 1.5 rad/s against 15).
Not cosmetic: at a shoulder's pace the first tick of a lean is 0.375 rad, which
is already past what a man can stand at, so the body proposed a pose it could
not keep and thought better of it, for ever, and never bent at all.

### Two bugs it flushed out

**The body's corner moves when only part of the body does.** Every segment is
kept as an offset from the min corner of the whole body, read fresh each time —
so a lean, which carries a torso and both arms forward and leaves the legs, can
change which voxel is the corner, and then every offset on the body is out by
one. An arm reaching never showed it: an arm goes forward, and the corner is
behind.

**Identity is resolved once a tick and kept.** `_repose` moves flesh INSIDE
that tick, so everything read afterwards was measured against a picture of a
body that had since bent. It only ever mattered once a pose could move the
corner.

Tests: `test_a_SPINE_BENDS_rather_than_SWINGING`,
`test_a_body_LEANS_AS_FAR_AS_IT_CAN_STAND_and_no_further`,
`test_LEANING_is_a_CHOICE_like_any_other`.


## A hand on the rail (2026-09-12)

Item 44, and it is what a lean is FOR. Item 35 said it in the first place:
"leaning out over a lip means moving the TORSO". A lean that stops at 10.7
degrees does not reach anybody.

Nothing new was declared. `_pulled_over` already did this sum, from the other
end — a held thing NOT resting on anything hangs from you and drags you over,
so a held thing that IS resting on something is holding itself up, and can hold
you. A grip was already an edge in the support graph; balance already read the
foot contact. The two had simply never met.

| a man at the lip | how far he leans | what stops him |
|---|---|---|
| nothing to hold | **0.188 rad — 10.7 degrees** | his own balance |
| a hand on the post | **0.338 rad — 19.3 degrees** | the LATTICE |

Same body, same back, same spine. The world is the only difference.

**And the first reading of this was wrong, in the way I have been fixing all
day.** It said 45.8 degrees, because it read `pose` — which is the motor
command, and runs on through every angle the lattice cannot draw. `drawn` is
where the flesh IS. They part company badly on a lean, because a lean has a
long undrawable tail: measured, a body whose angle read 45.8 degrees was bent
19.3. Every test and every menu row about a body's SHAPE reads `drawn` now.
**Ask the flesh** — third time today, and this time I was the one who forgot.

The man holding on is not straining his balance at all when he stops. He has
run out of LATTICE: past about 0.34 rad the shear tears a one-voxel-wide arm
off its own shoulder and `_still_joined` refuses it, correctly. So `lean_max`
at 0.8 rad is a stop the body can never reach, and the binding constraint is
item 41 again — limb thickness, not anatomy.

It is CONSERVATIVE, and worth being honest about: the base reaches the hand, so
a grip buys exactly the ground up to the fist. Really you can pull on a rail and
go well past it, because the arm is in tension and the sum is a force balance
rather than a wider footprint — which would make a reach over a lip depend on
how strong the arm is, which is the point. That is item 47.

Test: `test_a_HAND_ON_THE_RAIL_lets_a_man_lean_out_over_the_LIP`.


## Which hand (2026-09-12)

Item 48. Seven places named `"right arm"`, so the left arm was flesh, mass, a
lever and a fist that nothing could ever be done with — `_fists` found it,
`_hold_torque` weighed it, a lean carried it, and it could not pick anything
up. A man could not hold the rail and reach over the lip, which is the one
thing the lean was built for.

**A body has two arms and a PREFERENCE.** The dominant one is stronger and
better practised, so it is what a body reaches with. But a preference is not a
rule, and the other hand wins when the other hand is plainly the one for the
job — a thing on your left is nearer your left hand.

**That is a reason, not a die**, which matters twice. The sim carries no die and
is not about to grow one. And "sometimes the other hand" is not a coin toss in
real bodies either: it is where the thing is.

`off_hand_m` is what the switch costs — **half a shoulder width**, and it
belongs on that scale for a reason. The cost of the off hand is that it is
weaker; the benefit of the near hand is not reaching across yourself; those
trade at roughly the distance between your shoulders. Set wider than the
shoulders (0.25 m was the first guess, against 0.2 m of shoulder) it stops
being a preference and becomes a rule, and the off hand never wins once.

| what is in front of a right-handed body | which hand |
|---|---|
| nothing in particular | right |
| a thing by the left fist | **left** |
| a thing by the right fist | right |
| a thing straight ahead | right |

A left-handed body gives the mirror image, because `handed` is a fact about the
body like its mass — declared, not learned. There is nothing to learn about it;
we already know it.

**And the rescuer works now.** A right-handed man takes the post with his right
hand and reaches with his left; a left-handed man does the reverse. `p["held"]`
records WHICH hand took it, and a hand that is already full is not a candidate
— an indisposed hand is not a choice, it is an absence. The off hand is also
weaker (`off_hand`, 0.8), which is most of what having a dominant hand means:
what it can pick up is measured against that hand rather than the body's best.

Still one thing held at a time. Two independent grips want `p["held"]` per arm,
and that is a bigger change than this one.

Tests: `test_a_body_USES_ITS_DOMINANT_HAND_and_the_other_when_that_is_the_one`,
`test_ONE_HAND_HOLDS_THE_POST_while_the_OTHER_REACHES`.


## What leaves the world (2026-09-12)

Item 8, and it was reachable more easily than the note guessed. A flier never
gets out — `_law_bodies` stops it at the last clear pose, which is why the
first probe threw a block at 60 m/s through a 24-voxel world and it landed back
where it started. A TOPPLE does get out: a tall thing standing at the edge
swings its top clean past the wall.

`_land_body` clamped those cells. A voxel swung past the wall was set down ON
the wall instead, several of them into the same column, and `smass =` overwrote
rather than added. So mass went missing, quietly, in the one field every test
in the suite leans on.

**Clamping was never right anyway.** It teleports matter to the edge and calls
that a landing — a thing that leaves is GONE. Saying so is what keeps "mass is
conserved" a statement you can CHECK: every gram is on the lattice, or in a
body in flight, or on `self.gone`, and the three add up.

Measured on a tree toppled off the edge of a 30-voxel world:

| | |
|---|---|
| wood before | 11880 g |
| on the lattice after | 11700 g |
| **recorded as having left** | **180 g** |
| sum | 11880 g |

And nothing was teleported into the wall to make the sum work, which the test
checks separately — a conservation sum that passes because the matter was
shoved somewhere wrong is worse than one that fails.

`total_wood()` counts the tally, and `total_lost()` is the general question.
Zero in any scene nobody is throwing things out of.

Test: `test_what_LEAVES_THE_WORLD_is_counted_not_lost`.


## What you drag, you pay for (2026-09-12)

Item 9. The force arithmetic said a haul was legal and then charged nothing for
it, so a man towing an unconscious body walked at exactly the pace of a man
carrying nothing. Rescuing someone was free; dragging a crate across a room was
the same act as strolling across it.

A body has only so much to put out, and what the load takes the legs do not
get. Pace goes with what is left.

| crossing 30 voxels of room | ticks |
|---|---|
| empty-handed | **92** |
| dragging a block of lead | **175** |

**The number is written down nowhere.** It is this load's friction against this
body's strength — a lighter load or a stronger man gives a different one, and
the same load on ice would give another again, because `_effort` is the same
question a shove asks.

**Two costs, because they are two different things.** A thing LIFTED is held
up, and what it costs is its weight. A thing TRAILED along the floor is not
held up at all, and what it costs is friction. `_haul_cost` asks which by
looking at whether the load is touching the world — the same test
`_pulled_over` and `_overbalanced` use, now doing a third job.

The walk gate stopped being `tick % walk_every` and became a per-body stride
that accumulates. With nothing hauled it lands on exactly the ticks the modulo
did, which is the property that made it safe to change: the pace of a man
carrying nothing is the pace he always had.

And the rescue still works. It takes him longer, which is the point.

Test: `test_WHAT_YOU_DRAG_YOU_PAY_FOR`.


## Friction, and the joules nobody was counting (2026-09-12)

Item 6 had been waiting on "sliding contact between free bodies". It turned out
the sliding contact was a man dragging something, which item 9 had just given a
force: work is force times distance, so a thing hauled one voxel over the ground
has turned `drag_N * voxel` of muscle into heat. There is nowhere else for it to
have gone. Split evenly between the thing and the floor, because a rubbing pair
is two surfaces and the third law does not care which one you had in mind.

**No new rule was needed.** `E` is the field combustion already reads, so a
thing dragged far enough over a rough floor gets hot, and a thing hot enough
catches. Nobody wrote "dragging can start a fire"; it is what `mu N v` and a
combustion law MEAN together.

The magnitudes are honestly small, and that is the physics rather than a
shortfall: dragging a 200 kg block of iron three metres puts about 500 J into
the world, which moves a block that size by a fraction of a degree. Friction
fires want speed and pressure this scene does not have.

### The bug it found, which was the better result

The first probe said the heat was going missing — 2562 J put in, 535 J present.
It was not going missing. **The world sheds heat and never said how much.**

    Tk = self.T() + 273.0          # every voxel radiates to the wider,
    self.E -= RAD * ... (Tk**4 - 293**4) / 3.0    # cooler world
    self.E *= (1.0 - LEAK)

Both lines are right. A voxel really does radiate into a colder universe and
that is exactly what stops a flame climbing for ever. But they were the only
place in the sim where a conserved quantity changed and nothing wrote it down —
so "energy is conserved" was not a statement anyone could CHECK, in the way
mass became checkable when what left the world started being counted.

Measured: **1000 J left completely alone in a closed room is 779 J sixty ticks
later.** Nothing was wrong. Nothing could say so either.

`self.shed` counts it, `total_energy()` adds up what is in the world, what is
riding on bodies in flight, what went past the edge, and what was radiated
away — and it comes to what you started with.

**The same shape as item 8, one field over.** Ask of every conserved quantity:
when it leaves, who writes it down?

### And a cadence that had to be exactly right

Replacing `tick % walk_every` with a per-body stride was safe only if an
unladen body kept the cadence it always had. Mine started one short of that and
stepped at ticks 0, 2, 5, 8 instead of 0, 3, 6, 9 — every body in the sim a
voxel out of step, which surfaced as a crash in a belief test whose pillar was
now being built on top of the body. Starting the accumulator at `pace - 1` puts
it back exactly.

Tests: `test_DRAGGING_A_THING_HEATS_IT_and_the_floor`,
`test_the_WORLD_SAYS_HOW_MUCH_HEAT_IT_SHEDS`.


## Falls were arriving too hard (2026-09-12)

Item 7 said `FALL_SUBSTEPS = 4` capped falls at 8 m/s — honest below, clipped
above. Measured, it does no such thing.

The cap is on how many voxels a column may DROP in a tick, not on how fast it
is going. So a body that cannot fall as fast as gravity is pulling it spends
LONGER falling, and gravity goes on adding to it the whole time. The speed does
not get clipped; it gets extra.

| drop | arrives at | gravity gives | |
|---|---|---|---|
| 6 m | 11.24 m/s | 10.80 | +4% |
| 12 m | 18.46 m/s | 15.31 | **+21%** |
| 20 m | 27.81 m/s | 19.78 | **+41%** |

Energy goes as v squared, so **a 20 m fall landed with twice the blow it should
have**. In a sim where the first question anybody asked was whether a man can
pull another off a cliff, that is not a rounding error.

**And the fix was free, which is why the note's suggested fix was not needed.**
It proposed sweeping only the moving columns to make more substeps affordable.
That work is already done, in a different form: the substep loop stops the
moment no column has any fall left, so a world with nothing falling never runs
a second sweep. Measured with a slab dropping through a furnished room,
4 substeps against 16:

| | 4 | 16 |
|---|---|---|
| nothing falling | 0.26 ms/tick | 0.30 |
| something falling | 139.48 ms/tick | **140.71 (1.01x)** |

Sixteen is honest to about 40 m. Eight would have done for 20; the headroom is
cheap and cliffs are not all the same height.

**The lesson is about the note, not the code.** "Honest below 8 m/s, clipped
above" was a guess written down as a measurement, and it sat in the file for
weeks reading like a known quantity. A number in this file that was never
measured should say so.

Test: `test_a_LONG_FALL_arrives_at_the_speed_gravity_gives_it`.


## A throw is a swing that lets go (2026-09-12)

Item 3 said "throwing falls out of (2) — the same impulse with a different aim.
Free." It was right, and it is the nicest kind of item to close: nothing new
was modelled.

A thrown thing is ALREADY TRAVELLING. It has been going round on the end of an
arm, and letting go only stops it being made to go round. So its speed is the
speed it had — `omega` times how far out it was — and its direction is the
tangent, which is where the hand was taking it anyway. `_release` splits the
object's cells out of the swinging body and hands them to the flier machinery
that already existed for jumps and falls.

**The release angle is geometry, not aim.** A hand on a circle moves at right
angles to the arm: hanging straight down it is going forward, straight out in
front it is going up, and half way between it is going forward and up at 45
degrees — the angle that throws a thing furthest. A body throws well because of
where its shoulder is, not because it knows any ballistics. Measured release:
**52 degrees above horizontal.**

And the arm is lighter once the thing has gone, so it comes round faster after
the release than before it — which is what anyone who has thrown something has
felt, and which nobody had to write down.

| | speed leaving the hand | how far it flew |
|---|---|---|
| 0.97 kg | 4.92 m/s | 2.45 m |
| 1.95 kg | 4.11 m/s | 2.15 m |
| 3.90 kg | 3.12 m/s | 1.40 m |
| 11.70 kg | 2.02 m/s | 0.82 m |

Nobody wrote down how fast a throw is. It is the arm's torque against what the
arm is carrying — Hill's relation and a moment of inertia, the same two things
that decide how fast an axe swings.

**And it is WEAK, honestly so.** A person throws a 1 kg stone at 10-15 m/s and
ours manages 4.92. The ceiling is 6.75 m/s: Hill caps the shoulder at 15 rad/s
and the hand is 0.45 m out. The quarter arc from hanging does not even reach
that. A real throw starts behind the head and has the legs, hips and trunk in
it before the arm is — and the waist exists now, so a wind-up and a trunk
unwinding into the swing would cost no new law. Item 55.

Tests: `test_a_THROW_is_a_SWING_that_lets_go`,
`test_a_LIGHTER_THING_is_THROWN_HARDER`.


## A catch is a grab at something that will not wait (2026-09-12)

Item 4, and it cost almost nothing — which is what the item predicted, and
which only became true this afternoon when throwing gave the sim something to
catch.

**Nothing new was modelled.** A flier already has a position and a velocity; a
fist already has a place; `reach_m` already says how far a hand goes. So the
percept is not a sight cone and not a distance rule: the thing is within a
hand's reach, which is the only range at which noticing it is any use to you.

**The one law is that stopping is force times time.** A thing of mass m at
speed v needs `m v / t` to be brought to rest in the time a closing hand gives,
and a hand has only so much. That is why a cricket ball can be caught and a
brick at the same speed cannot, and it is the same arithmetic as lifting — so a
catch that is too much for the arm simply does not happen, and the thing goes
past a hand that could not close on it.

| | needs | |
|---|---|---|
| 0.97 kg at 4 m/s, 3000 N arm | 32 N | caught |
| 26.3 kg at 4 m/s, 3000 N arm | 878 N | caught |
| 7.8 kg at 4 m/s, **200 N arm** | 260 N | **goes past** |

`catch_s` is 0.12 s — how long a closing hand gives. It is the whole of what
makes a catch different from a wall, and it is the only number this needed.

**And a man who does not try to catch still NOTICES.** `sees_thrown` fires
whatever the hands then do, which is the seam working as intended: the percept
is physics, the response is taste, and a character sheet that would rather duck
says so in its own row.

Tests: `test_a_THROWN_THING_can_be_CAUGHT_and_only_if_chosen`,
`test_WHAT_A_HAND_CANNOT_STOP_goes_past_it`.


## Looking is something a mind can do (2026-09-12)

Item 15 said an eyes limb should wait "until anything has a reason to look
somewhere in particular". That condition came true earlier today: belief now
holds where the fire was, which floor a body trusts, and which reaches it has
already found impossible — all of it gathered by a head that turned on a timer
and could not be aimed.

`LIMBS` is five. **The sweep is the NULL act**, which is the property that made
this safe: a body that decides nothing goes on turning its head exactly as it
did before eyes were a limb, so nothing in the sim got worse the day they
became one. What is new is that "look behind you" is a row, picked by the same
policy and written to the same trace as every other act.

**Measured on a man with his back to a fire:**

| | noticed it at |
|---|---|
| sweeping, as before | tick **50** |
| choosing to look behind | tick **1** |

Both get there. The difference is fifty ticks of a room filling with smoke, and
it is the difference between a mind that looks and one that waits to be shown.

A chosen direction holds for `look_for` (25 ticks, about one step of the sweep)
and then the head goes back to sweeping — long enough to have LOOKED rather
than glanced, short enough that deciding to look is not deciding to stare. An
emergency overrides it: a body with something urgent in front of it does not
keep its head turned away.

**And it is a row a harvest can learn from**, which is the real point. Until
now every trace said what a body did about what it happened to see. They can
now say what it chose to look at first — which is most of what tells a careful
person apart from a lucky one.

Test: `test_LOOKING_is_something_a_MIND_CAN_DO`.


## A wind-up, and a third reason to widen the arms (2026-09-12)

Nobody throws from their hip. The arm goes back first, and the whole of what
that buys is ARC — more of it to accelerate through before the hand opens. No
new law: the same swing, beginning behind the body instead of under it.

It is honoured only if the arm can actually BE there. A man with a wall at his
shoulder throws from where he stands and throws worse, which is right.

**And so, for the same reason, does a man whose own chest is in the way.**

| | wind-up | leaves at |
|---|---|---|
| facing +x — arms fore-and-aft | **refused** | 4.11 m/s |
| facing +y — arms to the sides | -0.9 rad | **5.28 m/s** |

The humanoid's arms sit either side of its torso in X, and an arm turns in the
plane the body FACES. Facing east, that plane holds the torso and the arm both,
so winding back sweeps the arm through the chest and `_swing` correctly refuses
it. Facing north, the arm and torso are separated along an axis the rotation
does not touch, and the wind-up is free.

So **the same man is a better thrower facing one way than another**, and the
code is not wrong — the BODY is. This is the third independent finding pointing
at D1:

1. an arm has 2 drawable poses with one bone and 7 with an elbow, but **30** if
   it were two voxels thick;
2. a lean stops at 19.3 degrees because the ARM tears off the shoulder, not
   because the man is off balance;
3. and now: whether a body can wind up to throw depends on its compass bearing.

Three separate capabilities, one voxel of arm. That is no longer a note about
posture — it is the single change that would move the most.

Test: `test_a_WIND_UP_THROWS_HARDER_where_there_is_room_for_one`.


## What widening the arms actually did (2026-09-12)

D1 had been sitting as a question for the user to answer blind, which is the
wrong shape for a question that can be MEASURED. So it was measured: arms
widened from one voxel to two in x, full suite, then reverted.

**What it bought** — exactly what was predicted, on the real humanoid rather
than a synthetic one: hand places 2 to 5 with one bone and 7 to **30** with an
elbow, and mass 38.1 kg to 43.9 against the ~47 kg a real 1.5 m person weighs,
which would have closed item 12 as a side effect. The walking footprint grew by
**one** voxel, not the two I expected.

**What it cost: 15 of 107 tests.** And the interesting thing is that they are
not all the scene-tuning failures I assumed. Three kinds:

| | |
|---|---|
| `396 == 369` | an exact voxel count — trivially updated |
| "the leaper lands on the FAR ledge" | a wider body cannot clear a gap tuned to a narrower one — scene work |
| **"an arm put out REACHES: its far end went from x16 to x16"** | **the arm stops reaching AT ALL** |

That last one is not scene tuning. A thicker arm sweeping from hanging to
horizontal collides with its own torso on the way, because the arm turns in the
plane the body faces and the torso is in that plane too.

**Which is the real finding, and it is not about thickness.** The humanoid
faces along its own shoulder line. Widening the arm makes the collision worse
rather than better, moves its mass forward of the toes so the free lean gets
WORSE (10.7 degrees to 8.6), and does not free the wind-up at all.

So: the pose enumeration improves because enumeration asks only whether a pose
can be DRAWN; the acts get worse because an act has to SWEEP there. D1 alone
buys the picture and loses the motion.

**The experiment was worth more than the decision it was meant to settle.** It
turned "should we widen the arms" — which I could not answer and neither could
anyone else without running it — into "a body cannot turn its shoulders", which
is D3 and is the thing actually in the way.


## Somebody goes over the edge, and somebody catches them (2026-09-12)

Item 56 was filed this afternoon as the honest half of a bug — `sees_thrown`
was firing for people, so it was made to ignore them, and catching a falling
person was written down as the real act that deserved its own row. It is built
now, and it needed almost nothing, because everything it wanted had been built
for other reasons in the preceding hours.

**Two percepts, not one.** `sees_thrown` is a thing; `sees_falling` is a
person. The arithmetic underneath is identical — a body in the air has a
position and a velocity, a fist has a place, and `m v / t` says whether an arm
can stop it — but what you do about a falling friend is not what you do about a
thrown stone, so they are two rows and a character sheet can answer them
differently.

**And nothing at all was added for the HOLDING.** A caught body goes back on
the lattice where it was caught, and `_grip_cells` has seeded what a hand holds
as supported since the day grips became support edges. So he hangs there
because a hand is holding him, and for no other reason — and if the hand cannot
carry him, the same law that has always dropped a hanging man drops him.

| a man steps off the lip | his lowest voxel |
|---|---|
| nobody catching him | 40 -> **1**, and "lands hard" |
| a man who chooses to catch | 40 -> **40**, held at the lip |

**Four things met here that were built separately**: a percept seam that
separates noticing from responding; fliers with real trajectories; a hand whose
strength decides what it can stop; and a grip that is an edge in the support
graph. None of them was built for this. That is the whole argument for modelling
causes rather than cases — nobody wrote a rescue, and a rescue happened.

Test: `test_a_FALLING_PERSON_can_be_CAUGHT`.


## One hand on the rail (2026-09-12)

A FREE hand is not the same as having no hands full. A body has two, and one of
them being busy is exactly the situation a rescue IS — so a catch offered only
"when holding nothing" meant a man with a hand on the rail could not catch the
friend going past him, which is the one moment it was for.

| | holds | catches | faller ends at |
|---|---|---|---|
| both hands empty | - | the faller | z40, at the lip |
| **a post already in one hand** | **iron** | **the faller** | **z40, at the lip** |

Item 50 turned out to be narrower than it was written: `held` (a thing) and
`dragging` (a person) have always been separate slots, so holding a post and
holding a man were never in competition. What is still impossible is two
OBJECTS, or two people.

**And the probe found a timing fact worth more than the fix** (item 57). A body
does ONE hand-thing per decision and decisions are 30 ticks apart, while a fall
from a ledge takes about 25. So a man who is not already holding the rail
cannot take hold of it AND catch his friend: there is one decision in the
window and he must spend it on one or the other.

That is arguably right — you cannot do two things at once either — and arguably
an artifact of `decide_every` being flat. A real body decides faster when
something is happening, and the percept seam already knows when that is.

Test: `test_ONE_HAND_ON_THE_RAIL_and_the_OTHER_CATCHES_HIM`.


## A joint is not a hinge (2026-09-12)

D4 asked which way a body should be built so it can move freely. The answer was
that the build only ever chooses WHICH SINGLE PLANE an arm swings in, and one
plane is not freedom of movement however well you choose it. A shoulder is not
a hinge.

**Every joint carries a pair now**: swing, in the plane the body faces, and
SPREAD, about the perpendicular horizontal axis — out of that plane altogether.
An arm can go sideways, across its own chest, and overhead-and-out. Within one
joint the spread applies first and the swing on top, which is the order a
shoulder does it in and the order that keeps "swing" meaning exactly what it
meant yesterday.

The transform went from a 2x3 affine on (sideways, up) to a 3x4 on (x, y, z),
because **two rotations in different planes do not compose in two dimensions**.
The horizontal coordinate a given turn does not touch passes through unchanged,
which is what makes one axis alone provably the old transform with a row added
— checked first, both axes, both signs, hinge and shear, five angles: **0
disagreements in 40 cases**. Without that every pose in the sim would have
shifted underneath the change.

A bare number still means "the joint nearest the body swings, and nothing
else", so `reach out` and every existing caller were untouched.

**And `_reach_around` has two ways round now** rather than one: bend the elbow
IN the plane, or spread the whole arm OUT of it. Elbow first, because bending
an arm is cheaper than swinging the whole of it sideways.

### The 250x that was hiding behind it

The change looked expensive — a ten-test subset went from about 100 s to 444.
It was not the axes. Profiled: `_object_at`, the flood fill that answers "what
is this voxel part of", was **84% of the run and called 4576 times in thirty
ticks**, with fifty million set operations behind it.

`_object_at` stops at 4000 cells. A stone ledge is sixteen thousand. So the
memo never covered it, and **every candidate voxel standing on that ledge paid
another four-thousand-cell flood**. Lengthening a body's reach from 0.30 m to
0.45 (itself a fix, from reading the arm instead of declaring it) tripled the
candidates and so tripled that.

"A thing is what is joined to what you grabbed", and JOINED only has to be
answered inside the box a hand can reach into. Labelling that box is one
vectorised pass and answers for every candidate at once.

| the two-hand catch scene, 30 ticks | |
|---|---|
| flooding per candidate | **100.3 s** |
| labelling the box once | **0.40 s** |

And the suite came out FASTER than before the axes were added at all — chunks
of 14-127 s against 31-160 before — because the flood problem was always there
and only the longer reach made it loud enough to find.

**Two bugs in the fix, and the second is the interesting one.** A component
that merely TOUCHES the box edge is not necessarily the world — a post is
nineteen voxels tall and leaves any box drawn round one man's reach — so those
fall back to a real flood, once each rather than once per candidate. And the
labelling first merged **iron into the stone ledge it stands on**, because it
labelled "solid and not flesh" where `_object_at` has always meant SAME
MATERIAL: a table with iron legs is two things, because joints do not exist
yet. The post read as the world and could not be picked up.

**On kernels.** The obvious reading of a 100-second profile is "compile it".
That would have made a wrong algorithm faster. Kernels earn their place where
the work is a dense regular sweep — conduction, oxygen mixing, the support
relaxation, which is where numba already is — and not on a pointer-chasing
flood fill in Python. Fix the algorithm; then compile what is left.


## What a picture found that a hundred tests did not (2026-09-13)

The first render since the body was rebuilt. Three bugs in an afternoon, none
of which 113 passing tests had anything to say about.

**Reach ignored height.** `_in_reach` measured across the FLOOR, a softness
stated in the `ledge` docstring on the grounds that "an ankle 1.5 m overhead is
about an arm away; at five metres it would be nonsense". It was nonsense at
1.5 m: a man standing on a ledge held somebody lying on the ground twenty
voxels below him. It compounded with `_span_xy`, which allows for how WIDE
both things are — and a body lying down is its own length wide, so falling
over made a man reachable from much further away. True of his hand; not true
of the rest of him.

**A grip was only re-tested when its holder WALKED.** The check lived inside
`_haul`, which runs on a step, so a holder standing still never asked again —
260 ticks of holding someone he could not possibly be holding. Whether an arm
still reaches is a question about NOW, and nothing about it is about stepping.
`_law_grips` asks it every tick.

**And a units bug I introduced fixing it**: `_reach_of` returns METRES and the
new parameter wanted VOXELS, so passing 0.45 where 9 was meant made every grip
in the sim let go on the tick after it was made. The scene said so immediately
— "the puller loses hold at t1" — which is the kind of obvious wrongness a
picture shows and an assertion does not.

**What the scene shows now:**

    t12  the one pulled overbalances
    t12  the rescuer catches the one pulled
    t32  the rescuer loses hold of the one pulled — an arm is only so long

He catches him, holds him twenty ticks, and loses him as the puller drags him
out of reach. A tug-of-war with an outcome, and none of it written down: the
scene was built expecting a clean catch.

**The lesson is the one this file keeps learning in different clothes.** Tests
check what somebody thought to ask. A picture is the whole state at once, and
it cannot be politely silent about the part nobody asked about.

---

## 66. A limb one voxel thick cannot be turned, and the fix was the ROUNDING

**Why this is here:** the body was asked to move as freely as a real one does
at 5 cm. Measured first, before anything was built — every bone, both turning
planes, 31 angles each, alone, with nothing else in the way:

    left shin    swing : .u..uuu.uuuu.uuuujuuuuuuuu.u..   8/31
    left shin    spread: .jjjjjjjjuujjjuujjjujuujjjjjj.   2/31
    torso        swing : uuuuuuuuuuuu.uuuuuuuuuuuuu.u.u   3/31

`u` is two voxels rounding into one cell; `j` is the bone rounding into a
staircase whose steps touch at corners only. **A torso could be turned through
3 of 31 angles.** This was never a body that could move.

**Two separate causes, and the big one was not the shape.** Making bones
thicker helped the tearing and made the crowding WORSE — 3x3 scored 9/31 where
2x2 scored 16/31 — because the fault was `np.rint` on a rotation, which is not
injective on a lattice at any angle.

**Three shears are.** `x += round(k*z)` slides a whole row onto a whole row, so
it is a bijection for any k, and three shears make a rotation (Paeth 1986, the
raster-rotation algorithm — a public technique, reimplemented here, not copied).
Same bones, same 31 angles:

    bone            rint      shear
    1x3x7 swing     9/31      31/31
    3x5x12 swing    4/31      31/31      (the torso)
    2x3x7 spread   16/31      28/31

Mass conservation stops being a check that can fail and becomes a property of
the arithmetic. WHERE it lands is the cost: three roundings drift about one
voxel at the far corner of a 1.5 rad turn (measured: 1.04).

**So the rotation is still preferred and the shears are the fall-back**, taken
only when the rotation cannot draw the pose AND the shear pose is one the body
can actually be in. That last condition was learned the hard way: without it
the fall-back traded "the lattice cannot draw this" for "the post is in the
way", and a left-handed man beside a post lost the ability to reach out at all
— his flesh was fine, his arm was simply being put down a voxel from where it
aimed. **A fall-back that can only turn a refusal into a success cannot
regress anything.**

**What is NOT done, and the number that says why.** The shears fix the crowding
but not the tearing: a bone one voxel across still has nothing holding its rows
together when it turns in the plane it is thin in. So these arms swing freely
forward and back (31/31) and barely at all out to the SIDE (5/31).

Two voxels across fixes it — measured 27/31 — and is the truer body besides:
a real upper arm is about 10 cm through, one voxel is a matchstick, and the man
goes from 38.1 kg to 43.9 kg against the ~47 kg his height asks for. **It was
built, measured, and backed out**, because it takes the body from 5 voxels
across to 7 and fourteen scenes are built around a man 25 cm wide:

    two people who touch stay two people      a braced man cannot be pulled
    an arm reaches out and a wall stops it     a grip carries load
    an arm bends round what it cannot reach    a hand on the rail
    ... 14 in all

None of those failures were the body being wrong. They were scenes measured
against a narrower man. **That is a decision about the scenes, taken with the
scenes open, not a change to slip in under a limb fix** — so the arms stay thin
and the sideways reach stays poor until it is made deliberately.

## 67. A crouch is not a limb pose, and at 5 cm it is not a fold either

**Why this is here:** "crouch sure if you can do it lets do it."

**First measurement:** bending a knee on its own is impossible — 1 of 289 knee
angles drawable, and the one was standing up straight. Rotating a leg about its
ankle swings the HIP away from a torso that has not moved, so the leg comes off
the body. *You cannot bend a knee without the hips moving.*

**Second measurement, after making it one composite motion** (shin forward,
thigh back, body follows): the geometry works — the hip drops 2 voxels at
0.5 rad, 4 at 0.9, 8 at 1.3 — but the knee crease crowds 2 to 9 cells of 42,
and the drift that puts them there pushes the foot through the floor. **The
bend is finer than the grid.** A knee crease at 5 cm is one cell wide.

**So the crouch is a COMPRESSION, which is what is actually visible at this
size.** Each leg keeps its foot planted, every cell comes down in proportion to
how far up the leg it sits, and everything above the hips drops by exactly what
the legs lost — so hip and waist move together by construction and the body
cannot come apart at the waist. Measured, and mass drift is 0.00000 kg at every
angle:

    knee     sank    leg height
    0.2rad     0       14
    0.4rad     1       13
    0.6rad     2       12
    0.8rad     4       10
    1.0rad     5        9
    1.2rad     5        9   <- capped

**The cap is `_crouch_of`, not a number in this file** — 0.252 m read off this
body's own legs, so a tall man sinks further than a child and nobody writes it
twice. That is the self-referential loop, closed: the legs decide the crouch.

**Two bugs worth keeping.** `_angles` spreads one number over the joint nearest
the body, which is right for an arm and wrong for two legs — one number that
way is a man bending one knee, which is a LUNGE. Symmetric-by-default (D2) had
to be written where the legs are, not where the angles are; pass a pair and you
get the lunge on purpose. And `_still_joined` looked for the part of the body
that STAYED PUT to hang the moved part from — but a crouch moves every bone
there is, so it found nothing to hang from and reported a sinking body as one
that had come apart. When the whole body goes, joined means joined to ITSELF.

**Flesh bunches, it does not vanish.** Where two bones want one cell the
surplus goes to the nearest free cell (`_bunch`) instead of the pose being
refused or the voxel dropped. A folded knee thickens where a real one does, and
the mass books stay shut. `None` — nowhere within reach for it to go — is a
real refusal: flesh pressed into a gap smaller than itself.

**And the picture caught one more, as usual.** The crouch scene put a stone
beam overhead for him to duck under. A slab with nothing holding it up FALLS,
so the beam came down through the place he was crouching into, and `_crouch`
reported `blocked` on the way back up — for thirty-odd ticks I read that as the
crouch failing to reverse. It was the ceiling landing on him. The offending
cells were STONE at z22 and z26, in mid-air, inside the man.

Nothing was wrong with the body and nothing was wrong with the support law:
the scene was wrong, and only the scene was wrong. Put the beam on four posts
and he goes all the way down and all the way back up, 38.138 kg at every one
of forty-one frames. `renders/crouch/`.

## 68. A shoulder is a joint, not a dial — and the angle is the truth

**Why this is here:** the user's idea — "other kinds of voxels, circular ones,
which can rotate on their axis" — and their pick of where to put the first one.

**First, a correction to what was promised.** I said a round pivot would end
the staircase because the limb would stop being rounded cumulatively. That was
wrong: `_repose` already draws each bone once from its canonical rest shape.
The staircase comes from a SINGLE rounding of a rotation, which is
non-injective however few times you do it. A round joint does not remove it.

**What it buys instead is better.** The body's truth stops being "where the
voxels are" and becomes "what the joints are at". A joint holds a real angle
and a real angular speed; the voxels are a DRAWING of that. So a limb can BE
at an angle the lattice cannot draw, and reach, torque and momentum are read
off the joint rather than off the staircase. That decouples *can this body do
it* from *can the grid draw it*, which is behind most of the trouble in
entries 66 and 67.

**A joint with nothing holding it is turned by gravity** (`_law_joints`). The
moment is `M g d sin(theta)`, the limb's own inertia resists it, and the arm
accelerates, swings through the bottom, and settles. All three numbers are
counted off the voxels that are there — no row, no table:

    left arm, measured:  I = 0.1673 kg m2   M = 2.87 kg   d = 0.200 m
    period those ask for:                   1.083 s
    period the swing takes:                 1.117 s      (ratio 1.031)

The 3% is damping and finite amplitude, both of which lengthen a real
pendulum's period. **Nobody typed how long an arm takes to fall.**

**And it closed a flag doing a force's job.** Before this a limb stayed exactly
where `_law_pose` put it, so a dead man held his arm out at shoulder height for
ever and `alive` was the only thing holding it up. Now there is simply no
muscle on the joint and gravity wins.

**The mistake in between is worth keeping.** The first version released a joint
when its limb left `reach` — and a refused reach is POPPED as a note about what
this body will ask for NEXT, not as a release. So a live man's arm flopped the
instant he gave up on a wall, and the control case of item 46's test went from
40 of 40 to 4 of 40. *Wanting is not a force.* What holds a limb up is muscle
tone, which a living waking body already spends energy on every tick, so
`alive and awake` is the condition — losing consciousness IS losing tone, which
is a cause and not a flag.

**Damping is a GUESS and it is not allowed to be free.** `joint_damp` = 2.2/s
buys a dropped arm two or three swings, and nothing in the lattice measures it.
What it takes off the swing goes onto `shed` as heat, so the energy book still
closes — measured, 3.09 J out of one dropped arm.

**What is still to do, in order:**

- **the driven case** — `_law_pose` is still kinematic (`at + clip(d, ±step)`).
  It should apply muscle TORQUE and let the angle be an outcome. Everything it
  needs is already here: `_swing_of` gives the inertia, `_torque_of` gives the
  muscle, and `_law_bodies` already integrates `alpha = Nm/I` for a swing.
- **loaded joints** — a leg standing on the floor is not a pendulum, and that
  is a question about CONTACT, which `_law_joints` already asks. What it does
  not yet do is pass the load DOWN the chain.
- **the round voxel proper** — axis, angle, omega as three world fields rather
  than a record on the person, so a door hinge, an axle and a capstan are the
  same thing as a shoulder. Deliberately held back until there is a second
  user, so the shape of those fields is decided by two cases and not one.

## 69. A loaded arm is a slower arm — and the load was never in the sum

**Why this is here:** `_law_pose` was the last kinematic thing about a body.
Entry 68 gave an UNHELD joint real dynamics; this gives the held one a real
speed.

**Two faults, and the first was a docstring telling the truth about code that
was not.** `_hold_torque` says a shoulder holds "an arm and whatever is in the
hand" — and counted body segments only. So a man holding an anvil at arm's
length paid exactly what a man holding nothing paid. The lever was right and
the mass on the end of it was missing.

Fixed by making "everything beyond this joint" ONE answer (`_hanging_cells`),
so the muscle's effort and the joint's inertia are read off the same voxels and
cannot drift apart again.

**The second: a ceiling was being used as a rate.** `arm_wmax` is Hill's
force-velocity limit — 15 rad/s, a shoulder unloaded. `_law_pose` stepped every
limb at exactly that, so every angle was crossed at the same speed whatever was
being carried. Now the cap is a cap, and what a joint MANAGES is

    w = min(wmax, sqrt(2 * (torque / inertia) * angle left))

— the torque it has over the inertia it must shift, and then only as fast as it
can still stop in the angle it has left to stop in.

**Measured, end to end through `_law_pose`, same man and same 0.6 rad reach:**

    holding   0.0 kg  ->  2 ticks   (50 ms)
    holding   1.0 kg  ->  3 ticks   (75 ms)
    holding   7.8 kg  ->  7 ticks  (175 ms)

And at the sizes a rescue actually involves:

    arm alone          I = 0.167 kg m2   a 1 rad reach:  75 ms
    + 11.7 kg iron     I = 2.936        314 ms
    + 38.3 kg lead     I = 10.405       591 ms

**Nobody typed any of that.** The anvil is in `_hanging_cells`, so it is in the
inertia, so it is in the time. A weak man is slower for the same reason, and a
thicker arm is slower than a thin one, and none of it is a second rule.

**In play:** snatching something heavy out of the way is no longer free. Two
men reaching for the same thing is settled by what each is already carrying.
And a man laden enough is slow enough to be beaten to it — which is a tactical
fact the sim now has without anybody writing a tactics table.

## 70. Two bugs a picture found, and why the arms are still thin

**Why this is here:** "render something to show me the new body". The scene was
built, and before it rendered anything it failed twice.

### A joint that climbed a ladder to nowhere, for ever

A limb crosses angles it cannot be drawn at and catches up at the next one it
can — right, and deliberate. What was missing is what happens when there is no
next one. Measured, a man told to raise his arm while facing along x:

    pose  0.375  0.750  0.900  0.000  0.375  0.750  0.900  0.000  ...
    drawn 0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  ...

**His arm never moved a single voxel**, and the loop ran as long as the sim
did. The command climbs, the flesh does not follow, the angle "arrives", the
body settles back to where the flesh actually is, and the next tick asks again.

Fixed the way this sim already handles a wall it cannot reach past: the body
LEARNS it. Keyed by which way it was facing, because that is what decides which
angles can be drawn at all. This bug was there before any of today's work —
`6b48071` runs it too — and no test caught it because every test asks whether a
body arrives, not whether it gives up.

### A body held together by its flesh, with no skeleton

Support travels sideways one span per hop and flesh does not span far, so a man
was a pile of meat: an arm held straight out was a cantilever of it, within one
hop of breaking and never measured because nothing had leaned on it.

A wider arm leaned on it. One more voxel put the outer column one hop past what
flesh spans, and `_law_support` **tore three cells off the end of a reaching
hand and dropped them on the floor** — mass conserved exactly, 396 cells before
and after, man in two pieces. Neither the flesh nor the span was wrong. A limb
is carried by BONE, and there is no bone in the lattice.

`_grip_cells` already said this about what a hand HOLDS — *"a hanging man does
not need his flesh to span like a girder"* — and never said it about the man.
Now a body with something under its feet seeds its own cells the way the ground
does. Same edge, same graph; that is what a skeleton is.

**Under its FEET, and nothing looser.** Seeding anyone merely in CONTACT seeds a
man in mid-fall the instant he brushes a wall, and then he never lands —
measured, it broke falling, catching and being carried at once, 18 failures
against 15.

### And the arms are still one voxel across

The case for widening got much stronger and the answer is still not yet:

| | drawable angles, right arm |
|---|---|
| facing along y | 30 of 30 |
| facing along x | **1 of 30** |
| facing along x, arms 2 voxels | 15 of 30 |

That is not a limit, it is a **paralysis in half the directions a man can
stand**, and it is the body's largest remaining fault. Widening costs 14 scenes
recalibrated — he goes from 25 cm across the shoulders to 35, and from 38.1 kg
to 43.9 — and the failures are the ordinary kind: loads picked to sit "either
side of what a 38 kg man can balance", two people stood a voxel apart, a gap
sized to a narrower leaper.

**The skeleton fix removed the one reason it could not be done at all.** What
is left is bookkeeping with the scenes open, which is the user's call and not a
thing to slip in under a limb fix.

### What the scene shows

`renders/arms/` — three men told to hold an arm out. Measured at t0:

    the empty hand      arm inertia 0.167 kg m2   (2.9 kg off the joint)
    the laden man       arm inertia 7.209 kg m2  (29.2 kg off the joint)

The empty hand is at 1.20 rad by t4. The laden man climbs 0.22, 0.42, 0.59,
0.74 and **stops at 0.41–0.81** — 26 kg at arm's length is past what his
shoulder will hold, which is `_hold_torque` refusing, and it only refuses now
because the load is finally in the sum (entry 69). The third holds 1.20 until
he is put under at t70, and then there is no muscle on the joint and his arm
falls and swings.

## 71. The hold that exempted the only case it was for

**Why this is here:** "a person just breaks in rescue".

**First, what is NOT happening.** Nobody comes apart. Every frame of the
scenario, measured: 1026 flesh cells in lumps of exactly 342, 342, 342 — three
whole men, on the lattice and in the air alike. The renderer was suspected too
and cleared: rendering one frame with surface culling OFF differs from the
culled one by 21 pixels of 921,600. The picture is honest.

**What the picture was honestly showing** is this, at t16:

    the puller        x12-16  z 1-30    on the ground
    the one pulled    x18-22  z30-59    UPRIGHT, in mid-air
    the rescuer       x30-36  z31-60    on the ledge

A man standing on nothing, with the man supposedly holding him eight voxels
further off than an arm reaches. Two of them stack in the frame and read as one
figure broken in the middle.

**The cause is one line in `_law_grips`** — the law written to fix exactly this
class of bug, entry 65, "a hold is re-earned every tick, not only when its
owner walks":

    if any((b["mats"] == FLESH).any() for b in self.bodies):
        continue                      # someone is mid-fall: see `_haul`

While ANY flesh in the world was mid-fall, the check was skipped for EVERY
holder, and the case handed to `_haul` — which only runs when somebody takes a
step. **So the one hold that most needs re-earning, on a man in the air, was
the one hold never re-examined.** Measured:

    t12  caught          12 voxels away   (an arm reaches 9)
    t16                  14 voxels
    t24                  18 voxels
    t32  let go          32 voxels — a metre and a half

and t32 is when he LANDED. He was not released because the arm ran out; he was
released because he stopped being a flying body and became checkable again.

**A body off the lattice still has cells and still has somewhere to be.** Ask
where they are, and the rescuer lets go at t20 while the man is still falling —
which is what "an arm is only so long" was always supposed to mean.

**The general lesson, and it is the second time today.** Both of today's worst
bugs were an escape hatch in a law that was itself written to close a hole:
`_law_grips` exempted mid-fall, and `_law_pose`'s "the angle outruns the flesh"
had no end. A law with a `continue` in it for the hard case has not covered the
hard case — it has documented that nobody looked.

## 72. Rendering all twenty-three scenes, to see what had quietly rotted

**Why this is here:** "render everything again, just to see previous stuff
works well." Scenarios are not in the test suite, so nothing has been checking
them. Three things had rotted, and none of them would ever have failed a test.

### 1. `fire_alarm` had been dying on its last line

    KeyError: 'menu'

The menu seam grew LIMBS — a body decides with its hands and its legs at once —
and the trace row's `menu` became `menus`. The scenario prints its harvest at
the end and was never updated, so it ran all 700 ticks, wrote all 60 frames,
and then crashed. Every frame it produced was correct. **The failure was after
the work, which is why nobody noticed it was failing.**

### 2. The tree was blue

The canopy holds its own weight of WATER, on purpose — it is sap, and it is why
a fresh canopy resists burning. The renderer drew a full-size cube wherever the
fluid field was set, without asking whether the cell was air:

    "water": fl == WATER,

So 2966 leaf voxels rendered as 2966 cubes of water over the top of the green,
and the tree came out blue. Poured fluid only ever enters AIR — `pour` checks —
so fluid found inside a solid is something the solid is CARRYING, and the solid
is what you should see. One `& (mat == AIR)` on each of water, oil and acid.

The sim was right the whole time. **The mass was conserved, the sap was doing
its job, and the picture was lying about it.**

### 3. Every frame of every scene is 373 MB and the repo is 55

The `.npz` dumps were already ignored, with a note that "the rendered
mp4/stills/traces beside them are kept" — but the per-frame PNGs were being
committed too, and a full sweep would have taken the repo up sevenfold. The
frames are a by-product: the scenario plus the Blender pass rebuilds any of
them exactly. Now ignored; what is kept is what a person actually looks at —
the mp4 and four stills a scene, 22 MB for all twenty-three.

### What the sweep found working

All 23 run. Measured across every frame of every scene: **no person is ever in
pieces.** Flesh goes to zero in `fire_alarm` and `glasshouse` with no char and
no ash beside it, which is both men reaching an exit — the scenes doing exactly
what they were built to ask.

    acid_bath  alchemist  arms  burning_tree  crouch  drop_test  fire_alarm
    forge  glasshouse  house_fire  jumper  lamp_shelf  leap  ledge  parachute
    pond  rescue  swing  torch_pillar  tree_fell  tree_hinge  two_rooms  vessels

**The lesson is the file's oldest one, in new clothes.** A test asks what
somebody thought to ask. Three things here were wrong for a long time and no
assertion could have caught any of them, because two were about what a picture
SHOWS and the third only happened after the last frame was already on disk.

## 73. Three complaints, and the third was a law with no direction

**Why this is here:** "i dont see an acid bath video, in alchemist the glass
bottle breaks weird doesnt shatter, the arms look werid on the arms render".

### The missing videos were mine

Four scenes — acid_bath, pond, tree_fell, vessels — were re-rendered after the
fluid fix (entry 72) and I ran the render pass without the encode pass, so their
mp4s were deleted and never rebuilt. Nothing to do with the sim.

### The bottle: a law that had no idea which way is up

The first two suspects were wrong and worth recording. Glass mass is conserved
EXACTLY — 48750 g at every tick, no drift — so nothing was being destroyed. And
`_shatter` really was running, 29 times. The cell count rising 156 -> 234 at
constant mass is shattering working: fewer grams a cell, spread wider.

What the picture showed was a bottle that punched a hole through the shelf and
stopped halfway, hanging in its own hole:

    z23-24  ....GGGG........GGGG....    part still up top
    z22     wwwwww.wwwwwwwwwwwwwwww     a HOLE in the shelf
    z19-21  ....GGGG................    a chunk hanging THROUGH it
    z18     ........................    and nothing at all beneath it

Isolated, the rule underneath it is stark: **a glass block touching only the
underside of a shelf hangs there for ever.** Support relaxed outward through
all six neighbours, so a cell inherited it from ABOVE exactly as readily as
from below, and anything against a ceiling was glued to it.

Being held from above takes a bond in TENSION. The only thing in this sim that
has one is a hand, and `_grip_cells` already seeds those — which is why a
hanging man still hangs and a wedged bottle now does not. Everything else rests
on what is under it, or on what is beside it while its own material can span.

### And a second fault found on the way: fragments that spanned like panes

`_shatter` opens with "the voxel's cohesion is gone" and then handed the
fragments back as ordinary material. Glass spans 8 reference voxels — sixteen
cells at 5 cm — which is right for a window and nonsense for the shards of one.

So a piece is now marked as a piece: it rests on what is under it and holds
nothing. **The flag has to RIDE WITH THE MATTER**, which took a while to see —
without that a shard fell exactly one voxel, arrived as ordinary glass, and the
span it was never supposed to have caught it again.

### The arms are thin, and that is what looks wrong

Nothing is broken in the arms scene: the angles are right (1.2 rad from hanging
puts a hand forward and 21 degrees below horizontal, which is what it does),
the flesh matches the angle, and the man is one piece throughout. The arm is
ONE VOXEL across — 5 cm — so it draws as a blade rather than a limb, and at
full extension it reads as a slab floating beside the body.

Rendered both ways, same scene, same tick: the two-voxel arm reads as an arm.
This is the same fault as entries 66 and 70 (30 of 30 drawable angles facing
one way, ONE of 30 facing the other) arriving for the third time, now as
something a person can simply see. **It is still not done, because it is still
14 scenes recalibrated, and that is a decision to take with the scenes open.**

## 74. The arms are two voxels across, and what it cost to get there

**Why this is here:** "do it." Entries 66, 70 and 73 each ended with the same
unfinished sentence — the arms are one voxel across, which is a paralysis in
half the directions a man can stand, and fixing it means re-measuring the
scenes. This is that.

    right arm, drawable angles     one voxel      two voxels
    facing along y                 30 of 30        15 of 15
    facing along x                  1 of 30         7 of 15

He is 43.9 kg where he was 38.1, and 35 cm across the shoulders where he was
25. **Ten tests had to be re-measured, and not one of them was wrong.**

### Four were built through something

A body is 7 voxels across now, and several scenes placed people and obstacles
with the old 5 in mind. Two men six apart shared a column. The rescuer stood
INSIDE the post he was meant to be holding. The puller's arm was built into the
ledge face. A block of lead was built through the holder's own hand. These are
not interesting; they are arithmetic, and the comments in the tests say so.

### Four were thresholds fitted to the old body

| | was | is | why |
|---|---|---|---|
| load he can balance | 64 / 128 kg | 128 / 179 kg | heavier AND a wider base |
| load that tips him at arm's length | 38 kg | 68 kg | the same |
| fatal rolled fall | 8 m | 7 m | a fall is his own weight times height |
| a man lying down, top of | z11 | z14 | he is as tall lying as he is broad |

Each of those is the sim being right about a different man. The one that most
deserved saying out loud: **an ordinary 400 N grip can no longer hold him** —
43.9 kg is 430 N — so the grip test now gives its holder the strength to close
that gap and asserts the 400 N man drops him.

### And two were the sim being genuinely short

**`_reach_around` never touched the shoulder.** It took the angle it was asked
for and hunted the elbow and the spread underneath it, so an arm that could not
get out AT THAT SHOULDER ANGLE gave up — while a pose with the same elbow and a
shallower shoulder sat there unfound. Measured with a wall four cells off: 9
poses drawable and unblocked, and the search returned None. A man who cannot
reach at full stretch reaches less far; it tries that now.

**`_law_pose` only went round the WORLD, never the LATTICE.** Bending round was
tried when a pose came back `blocked` and not when it came back `undrawable`,
which was left to the "angle outruns the flesh" path — and with a wall four
cells off and a two-voxel arm there is no later angle to catch up at. Measured:
183 of 210 poses undrawable, 18 blocked, 9 clear, and the arm never tried one
of the 9 because nobody told it the refusal came from the grid.

Both of those were true before the arms changed. The wider arm found them.

### A build oddity worth writing down

`_person` always offsets the arms in **x**. A body facing +x therefore wears
them FORE AND AFT rather than side by side, with one hand permanently nearer
whatever it faces. With a one-voxel arm that was survivable; with two, the far
arm is walled in between its own torso and anything it faces, and a left-handed
man could not free-hand anything at any distance — measured at three
separations, all of them. The post test is turned ninety degrees for now. **The
real fix is for the arms to be laid out across the facing, not along it**, and
that is a change to `_person`, not to a scene.

### And a livelock I shipped in 73 and found here

Removing support-from-above was right and it broke something else. A vessel on
a shelf edge topples, meets the shelf at 0.41 rad, is set down, is found
unbalanced on the next tick, and topples again — for ever. The wedge cooldown
existed and fired only under 0.15 rad, which catches the ones that never start.
**A topple that does not get halfway over has met something**, so the threshold
is 0.7 now: alchemist went from 862 saved frames to 217, and a real topple runs
to pi/2 and is untouched.

## 75. Asking the turn BACKWARDS — built, measured, and not landed

**Why this is here:** "use whatever from existing cutting edge research", and
two papers — [Love Handles](https://arxiv.org/abs/2608.17930) (sparse
deformation handles with compact support) and
[Aokana](https://arxiv.org/abs/2505.02017) (SVDAG, LOD, streaming for voxels).
Aokana answers a question we have not reached: it is LOD for DRAWING, and D5 is
LOD for SIMULATING. Love Handles is the skinning family, and reading it against
this file exposed something plain.

**We forward-map, and every drawing fault in this file is a consequence.**
`_repose` asks "where does each rest cell GO" — source to destination — and a
forward map on a lattice has exactly two failure modes: two sources land on one
cell, and no source lands on a cell. Those are "two voxels into one" and the
staircase, which between them account for entries 66, 67, 70 and most of 74.

Asked backwards there are no failure modes to have. Walk the cells the limb
could occupy, put each through the INVERSE turn, keep it if it lands in the
rest shape. Every cell gets exactly one answer. This is how texture mapping has
worked since the 1980s.

**Measured, isolated bones, 31 angles, both planes:**

    bone           forward+rint      backward warp
    1x3x7 swing        9/31              31/31
    2x3x7 spread      16/31              31/31
    3x5x12 swing       4/31              31/31     (the torso)
    1x3x7 spread       3/31               6/31     (genuinely disconnected)

Cells kept: 0.97 to 1.02 of the rest count — so a turned limb covers a few more
or fewer cells and **mass is conserved by density, not by counting**. Measured
in the body: 0.00058 g of drift over 30 poses, which is float32 and nothing
else. (`total_mass` reports 3.9 g of "drift" here and that is its own float32
summation — one ulp at 43875 g. Worth remembering before chasing one again.)

**In the body it got most of the way and stopped.** Undrawable went from the
dominant refusal to **0 of 93 facing +y and 11 of 93 facing +x**.

**And then the seams.** Warping bone by bone closes every hole INSIDE a bone
and opens the joins BETWEEN them: each bone rounds against its own rest shape
and adjacent bones stop touching. Measured: **83 of 461 accepted poses left the
man in pieces**, and an arm came back with 45 of its 54 cells.

Warping the whole limb as one field — one pass over the cells, a cell is flesh
if ANY bone claims it, first claim wins — fixed the simple swings (15 of 15
angles, 0 in pieces) and **did not fix the compound poses**: the same 83 of 461
over the full sweep of swings, spreads and elbows.

**Why, and what it needs.** First-claim attribution gives every cell to exactly
one bone, and at a bent joint the two bones' images pull apart because nothing
says a cell near the joint belongs to BOTH. That is precisely what skinning
weights are for: a cell's transform is a BLEND of the bones near it, weighted,
and the seam is continuous because the blend is. Love Handles is a method for
choosing those weights with compact support, which is the property we want — an
arm's handle must not drag the torso.

**So the next step is weights, not another attribution rule**, and it is a real
piece of work rather than a tweak. Reverted for now; the repo is green at 123.
The measurements above are the reason to come back to it.

## 76. A shoulder is a socket inside the body

**Why this is here:** continuing from 75, which said the next step was skinning
weights. It was not. A third paper came in on the way —
[evoxels](https://arxiv.org/pdf/2507.21748v1), differentiable physics on voxel
grids for microstructure design — and the honest read is that it answers a
question we do not have: it is micron-scale PDE work, and making this sim
differentiable would mean numpy to torch plus differentiating through
shattering, toppling and menu picks. The one real hook is that this repo is
full of GUESS constants and a differentiable sim can FIT them, and for ten
scalars that is a coarse sweep, not a framework.

**What the skinning prototype actually found.** Blending the inverses (weights
in destination space, against each bone's deformed axis) took the sweep from
83 of 461 accepted poses in pieces to **25 of 1089**. Then the weight power and
the distance floor made no difference at all across two orders of magnitude —
which was the clue. The sweep varies the SHOULDER only, so both bones carry the
same transform and there is nothing to blend. The 25 were never bone-to-bone
seams.

Every one of them was an arm past 1.2 rad, swung toward the horizontal. **The
arm was letting go of the man**, and the reason is that the joint was at the
TOP OF THE ARM — the surface where the arm meets the body. A thing turned about
its own surface swings that surface away.

**One cell inward, toward the middle of the body: nought of 1089.**

And it is where a shoulder is. A ball in a socket under the deltoid, not a
hinge on the skin.

    drawable shoulder poses, of 93      facing y    facing x
    joint on the arm's surface            33          35
    joint one cell inside the body        62          89
    (backward warp, surface joint)        62          82

**The socket alone beats the whole backward warp**, and costs three test
recalibrations against eleven. A one-line anatomical correction, found by
building the sophisticated thing and watching which of its knobs did nothing.

**What it changed, all measured, all real:**

- an arm now reaches OVER a metre-high wall (x21, arm top z27, wall top z20,
  stone untouched) where it used to stop at it
- the axe swing comes round on a longer arc: from x18 the head passed the trunk
  instead of meeting it, 0 voxels chewed against 53 from x17
- **the fulcrum moved IN, so the lever every held thing hangs on got longer** —
  what he can balance at arm's length went from 128/179 kg back down to 77/102.
  Wider arms had made him harder to tip; the socket makes him easier. The man
  did not change; where his arm turns about did.

**The warp is still the right idea and is still not landed** (entry 75 has the
numbers). It is worth less than it was, because the socket took most of what it
was going to buy.

## 77. The menu-select model has an architecture and no data

**Why this is here:** [jevlike](https://github.com/vinnylarouge/jevlike) (MIT,
Minimal Labs) — "takes a piece of text and a list of N text options, returns
one probability for each option" in ONE pass. Each option is a query, it
attends over the context, a shared dot-product scorer turns each option-context
pair into a score, softmax over the options.

**That is `policy.pick(situation, menu) -> int`, exactly.** Same inputs, same
output, no decoding and no grammar to constrain — the "0 violations by
construction" the ablation wanted, for free. Their reported speed is the other
half: one pass against a small decoder writing 400 tokens was about 100x. That
is the shape the latency budget needs, and MIT means it can be ported with
attribution rather than reimplemented from the paper.

**So the architecture question looks settled, and then you count the data.**

Every scenario with people in it, traces harvested, outcomes stamped:

    scenario      rows    secs        rows/sec
    jumper          36     1.8          20.0
    leap             9     0.7          12.9
    arms            63    10.8           5.8
    ledge           24     4.9           4.9
    swing           24     3.2           7.5
    rescue          37    11.8           3.1
    glasshouse      43   199.9           0.2
    fire_alarm       4    79.3           0.05
    house_fire       2   131.0           0.02
    two_rooms        0    61.3           0
    parachute        0     9.8           0

    TOTAL 242 rows in ~530 s
    menu sizes 1-7 (mode 2), ~5 menus a row, one per limb
    outcomes: 47 safe, 195 still inside — no dead, no unconscious
    percepts: 205 none, 27 falling, 5 sees_falling, 4 sees_fire, 1 sees_runner

**And 242 rows collapse to 51 DISTINCT (menu, pick) pairs.**

That is the finding. It is not a volume problem you can fix by running longer:
the teacher is a deterministic reflex table, so the same menu always draws the
same pick, and a student can learn nothing that is not in those 51 rows. Any
architecture memorises that in a minute.

**Three things have to change before an architecture matters:**

- **Situations, not scenarios.** Eleven hand-built worlds is eleven worlds.
  This wants randomised ones by the thousand.
- **A teacher worth copying.** A reflex table can only teach itself. This is
  what "big models generate grounded data, the sim stays authoritative" was
  for.
- **An outcome signal that discriminates.** 47 safe against 195 still-inside,
  with nobody dead and nobody unconscious, barely separates anything.

**And the throughput table says how to build the harvester.** Decisions are
dense where bodies keep re-deciding — falling, jumping, catching — and almost
absent in the long fire scenarios, because `emergency` commits a body and it
stops weighing the ordinary. 20 rows a second from `jumper` against 0.02 from
`house_fire`, a thousandfold. **Harvest short physical worlds, not slow fires.**

**The number to watch is distinct (menu, pick) pairs, not rows.**

## 78. Worlds by the thousand, and the number that says whether it worked

**Why this is here:** entry 77 found the model has an architecture and no data.
This is the data.

### First, the seam had to have a spelling

A teacher labelling a decision and a student learning from that label must see
the SAME bytes, or the label is attached to something the student is never
shown. `src/voxel/seam.py` is that text, and three rules shape it:

- **Deterministic to the byte.** Prompt caching is a prefix match and a byte
  that moves invalidates everything after it — a labelling run that silently
  stops hitting cache costs about ten times what it should.
- **The reflex table is not in it.** `situation["reflexes"]` is the policy's own
  answer key. Give it to a teacher and the teacher reads the answer off the
  page; give it to a student and it learns to look the answer up rather than to
  decide. Dropped in the serialiser rather than at the call site so nobody can
  forget. `tick` and `who` go too — facts about the RUN, not the situation.
- **Stable and varying are different functions.** `rules()` is cached,
  `situation()` is not, and they are never concatenated in this file.

The spelled row is also written AS EACH LIMB IS ASKED, because `chosen` is a
running state: a harvest that reads `menus` and `pick` off the finished trace
gets the menu and the answer and loses the half of the context that made the
answer make sense.

### Then the census got sharper, and the news was worse

Re-measuring the scenarios through the spelling:

    809 rows that are actually a CHOICE (41 had one option — not decisions)
     67 distinct (percept, limb, options) -> pick RULES explain them
    659 distinct spelled texts feed those 67 rules

Adding every body-state field splits 67 classes into 413 and resolves ONE
ambiguity. **The input is ten times richer than the function.** A model trained
on this learns "read the menu, ignore the body" — precisely the wrong lesson,
and an argument FOR an LM teacher rather than against one, since the reflex
table is a lookup on percept and cannot use state it never reads.

### And then worlds instead of scenes

`src/voxel/harvest.py`. Small rooms, short runs, randomised ground, drops,
gaps, posts, things to hand, sometimes a fire, sometimes a second person,
sometimes a door, and a body that does not arrive fresh.

Two choices earn their place:

- **`Roam`, an exploring policy.** `_Wants()` with no argument wants nothing,
  so the body chooses the null act every tick — 25 worlds gave 143 rows and 24
  situations. Picking off the menu at random, leaning away from doing nothing,
  gave 41 from the same 25. A random policy is not a good policy and is not
  meant to be: what a harvest buys is where the body ENDS UP, and the label
  comes from the teacher later.
- **Menus of one are dropped.** 41 of 850 rows offered a single option. That is
  not a decision and it is not a training row.

**One thousand worlds, 703 seconds:**

    7736 rows
     832 distinct situations        (against 67 from eleven scenarios)
    outcomes: 7135 inside, 555 safe, 41 down, 5 dead

Twelvefold on the number that matters, an outcome signal that finally
separates, and still climbing at a thousand — 780 at 900 worlds, 832 at 1000.

### And the harvest found a bug on its seventieth world

`_law_grips` — written THIS WEEK to fix "a hold on a falling man is the one
that must be re-earned" — asked a body for the `off` that only a flier carries.
A body mid-TOPPLE has a `theta` instead. Nothing in twenty-three scenarios had
ever had a man holding somebody who tipped over rather than fell.

**That is the argument for random worlds in one line.** The scenarios test what
somebody thought to ask; a thousand worlds ask what nobody thought of.

## 79. Bugs found by asking the sim to be social

**Why this is here:** working out how to structure actions for bargaining, and
looking at the speech path to see what is there. Two bugs, one root.

**A reply to an alarm was itself an alarm.** Measured: B says *"I hear you!
I'm coming!"* and A perceives `hears_alarm`. The sim's only two lines trigger
each other, so any shout cascades — A shouts fire, B answers, C hears B and
flees a fire it has no evidence of, answering as it goes.

**And ears carried the eyes' guard.** The loop skipped anybody with
`not q["fleeing"]` before testing either sense. Seeing somebody RUN reasonably
requires them to be running; hearing them shout does not. Measured: a body
shouting "Fire! Fire! Get out!" while standing still was inaudible at any
range, and the same shout from the same place became audible the moment the
shouter started running.

**One root: `_shouted` was a timestamp.** The words went into `self.speech` and
nothing ever read them — the sim modelled the fact of a noise, not what was
said. It is now `(tick, line)`, `MEANS` says what hearing a given line does to
you, and a line nobody has given a meaning is audible and means nothing, which
is what a reply should do to a bystander.

**Two more, elsewhere, both found the same way — by doing something no scenario
scripted:**

- `_law_grips` asked a body for the `off` that only a flier carries; a body
  mid-TOPPLE has a `theta`. Found on the harvester's 70th random world, and
  written THIS WEEK.
- `fb in self.bodies` compared a body dict against every other, and a body dict
  holds numpy arrays, so the comparison came back an array Python could not
  evaluate. Found by putting eight people in one room. Two places.

**Four bugs this week, none of them found by a test.** Two by random worlds,
one by a crowd, one by reading the speech path with a question in mind. The
suite is 127 tests and catches regressions; it has never once found something
nobody had thought to ask about. That is not a criticism of the suite — it is
what a suite is — but it is the argument for running worlds nobody designed.

## 80. The queue, re-ordered by what we now know

Ordered by what BLOCKS what, not by what is interesting. Every claim below was
measured this week.

### First — because they spoil work done after them

**1. Credit assignment is flat, and it mislabels every row we harvest.**
`trace_outcomes` stamps the person's FINAL state on every decision in the run:
a choice at t=5 and one at t=400 get identical credit, and 92% of harvested
rows read "still inside", which mostly means the run ended. HiPER's phrasing
for why this is worse than sparse: flat credit "must propagate across the
entire trajectory... leading to unstable optimization". **Every row harvested
before this is fixed is waste**, which is why it goes first and not last.
Cheapest honest version: record the tick of the decision and of the terminal
event, and let credit fall off with the distance between them.

**2. Walkers cannot change elevation — not 5 cm.** Measured: a body told to
walk to a door behind a one-voxel step goes x5 to x9, stops, and stands there
with its feet at z1 for two hundred ticks. So every ledge and gap the harvester
builds is traversable DOWNWARD ONLY, and situation diversity — the one number
the harvest is judged on — is capped by it. Roadmap item 5 says this was tried
in September and taken back out behind item 13; item 13 has since been built.

**3. `TOUGH` has no temperature term.** All four readings are
`_TOUGH_ARR[mat]`; no `T`, no `E`, anywhere. Iron is as hard at 1200 C as at
20. The forge scene MELTS metal and nothing ever SOFTENS, so there is no window
to work in — which is what smithing is. `tough_eff = TOUGH[mat] * f(T/T_melt)`
has the shape `span = SPAN[mat]/scale * clip(packed/0.5, 0, 1)` already uses.
Small, and it makes a whole craft domain simulable rather than decorative.

### Then — the model path, which cannot start without them

**4. The teacher loop.** An API call constrained to a menu INDEX: one output
token, no grammar to violate, the rules half of the spelling cached and the
situation half varying. About $0.00025 a label, so $2.50 for ten thousand — the
teacher is not the bottleneck, rate limits and wall-clock are.

**5. And the check on it**: re-run the table from entry 78 against the
teacher's labels. Under the reflex policy, 67 rules explain 809 choices and
adding every body-state field resolves ONE ambiguity — the state is a
distractor. If the LM teacher's labels are no more state-dependent than that,
the teacher is not worth its cost and we have learned it cheaply.

### Then — the direction, which is a design question before it is code

**6. Wanting.** An offer means nothing to a body that wants nothing. Drives
live in the old `src/core` engine, so this is the seam between the two engines
rather than new physics.

**7. Speech acts with slots** — `say(offer|ask|refuse, about=<thing in view>,
to=<person in earshot>)`, filled by the same conditioned menu loop the limbs
already use. No parse step: the AST is built by selection and every leaf is a
reference the sim resolved.

**8. Options.** Wanted for horizon (one decision spanning 2000 ticks instead of
400 decisions) — and wanted by item 1, which needs boundaries to aggregate
returns over. Note what options do NOT buy: the 2000 ticks of physics still
run. That lever is separate, and it is that physics steps at 40 Hz while
decisions happen at 4.

### Last — physics polish, which blocks none of the above

`_law_pose` is still kinematic (68); loaded joints do not pass load down the
chain (68); the round voxel is a record on the person rather than three world
fields (68); the backward warp is measured and not landed (75); gas species,
flame-above-fuel and pyrolysis (2, 4, 5); the door leaks 450x too little (18);
a felled body falls like a plank (10).

### And a standing note about how bugs are found here

Four this week, none by a test: a toppling body asked for a flier's `off` (70th
random world), `fb in self.bodies` comparing dicts of numpy arrays (eight
people in a room), a reply heard as a fresh alarm, and a shout inaudible unless
the shouter was running. The suite has 127 tests and has never once found
something nobody thought to ask about. That is what a suite is — and it is the
argument for spending compute on worlds nobody designed.



## 81. Four bugs behind one arbitrary number, and the one that hid the rest

The step-up (item 5) was taken back out because widening the walkable set
knocked calibrated scenes over, and the post-mortem blamed the roam target:

> `i = (self.tick // decide_every + len(p["name"])) % len(far)` — a
> deterministic stand-in for taste with no preference in it at all.

So the plan was to give roam a preference and then re-try the step-up. Giving
it one — offering far spots BY BEARING, one per direction, the way
`go:frontier` already does — broke four tests. Chasing them down found that
the modulo was not the disease. It was the **dressing on four wounds**, and it
had been holding them shut by churning the goal every ten ticks so that no
body ever stayed anywhere long enough for anything to be visibly wrong.

### What was kept, and what was not

Each change was applied ALONE to HEAD and the failures attributed, because
toggling one thing at a time out of a pile of four cannot separate an
interaction from a cause:

| change | breaks |
| --- | --- |
| roam by bearing | all three calibrated scenes |
| god-channel removed | `REMEMBERS_WHERE_THE_FIRE_WAS` |
| already-arrived options dropped | `DRAGGED` |
| roam deleted entirely | `DRAGGED` |

`DRAGGED` breaks under ANY change to the leg menu — it is the razor-thin
scene, and its own comment says so. But roam-by-bearing broke a scene nothing
else touched, and that is the verdict on it: **it is not a fix, it is a
different arbitrary rule.** Nearest-far-per-bearing is as much a stand-in for
taste as an index is, it just fails differently, and it costs a scene to
install. Reverted. The roam target stays arbitrary and the post-mortem's
blocker — item 13, a policy with actual taste — genuinely still stands.

What is kept is the four bugs it exposed.

### Wound 1 — a body knew where every fire in the world was

```python
danger *= WILL["forget"]
alight = self.burning().any(axis=2)
np.maximum(danger, alight.astype(np.float32), out=danger)   # <- whole lattice
danger[xi[reach], yi[reach]] = np.where(alight[...], 1.0, ...)  # <- what it SAW
```

The third line set every burning column on the lattice to full danger, one
line before the fourth did the same job gated on sight. Measured: a witness
sealed in a room at x10, behind solid stone, carried danger 1.0 over a fire at
x30 it had no way of knowing about. `_places` scores routes against this map,
so bodies were routing around fires nobody had told them about.

This is the god-channel the ablation took out of the percept, one layer down
and much better hidden — the percept was honest, the BELIEF was not.

### Wound 4 — and it could not simply be deleted

Deleting it left a witness in an open room, twelve cells from a burning bench
in plain view, remembering **nothing at all**. The sight-gated line had never
worked. Every scrap of fire memory in the sim had been coming from the
god-channel, which is exactly why nobody noticed.

The reason is in the ray. `_look_around` marches its fans across a SINGLE
HORIZONTAL PLANE at eye height, and smoke fills a room from the ceiling down,
so a witness standing in the smoke layer had every ray stopped at the first
cell — measured, 335 rays reaching 335 cells — while the fire at its feet was
in plain view underneath. The comment three lines above had always claimed
otherwise:

> Read by COLUMN rather than at eye height, because light from a fire at your
> feet reaches your eyes

That was the intent and only half of it was built. Light now gets its own
reach, taken over the band from this body's FEET to its EYES, where a column
counts as clear if any height in that band is clear. It sees under smoke, over
a bench and through a doorway; it does not see through a full-height wall,
which is opaque at every height; the band stops at the feet so a body upstairs
cannot see a fire through its own floor, and at the eyes so it cannot see
through the ceiling.

And the looker had to be excluded from its own occlusion. Over a band, a
seven-voxel torso is opaque at every height in it, so the first version blinded
every body with its own chest. `free` already keeps this rule — *a body is not
a wall to itself* — and it has to be kept here too.

### Wound 2 — a third of the leg menu was options that did nothing

`_reach_along` refuses to name a spot closer than **2 cells from the body's
centre**. Arrival triggers at `arrive_m / vox_m` = **2.4 cells from the body's
nearest cell**. A body three cells across is therefore ALREADY ARRIVED at
"straight on" the moment it is offered: it takes the option, arrives on the
same tick, clears the goal, is offered the same clipped spot next decision,
and takes it again.

Measured over 40 harvest worlds: **49 of 139 leg options with a target — 35% —
were places the body was already standing.** Two numbers that are supposed to
mean the same thing, measured from the centre in one place and from the edge
in the other, a whole body apart.

The cost is not only the livelock. It is a third of a seven-slot attention
budget spent on options indistinguishable from standing still, which means the
same situation teaches a policy both answers. Fixed in `add()`, which every
place-option goes through, using the same arithmetic that judges arrival so
the two cannot drift apart again.

After, on the same 40 worlds: **86 options with a target, 0 of them no-ops** —
against 139 with 49 no-ops before. The junk is gone and the real options are
all but unchanged, 90 to 86. An earlier draft of this entry claimed 139 real
options where there had been 127, which compared two inconsistent code states
and was simply wrong; the honest figure is that this fix removes junk and adds
nothing.

### And it makes the harvest metric go DOWN

Item 78 says the number to watch is distinct SITUATIONS rather than rows.
Measured over the same 200 seeds, before and after:

| | rows | situations |
| --- | --- | --- |
| HEAD | 1364 | 280 |
| after | 1187 | 254 |

Both fell, and that is what should happen. A situation is `(percept, limb, the
options offered)`, so a menu carrying a junk option is a DIFFERENT situation
from the same menu without it — and junk options generate a great many
menu-shapes that exist for no reason but the bug. Removing them removes
twenty-six situations that were never real.

So the headline metric can be inflated by a defect, and a harvest that grew
because the menu got noisier would read exactly like one that grew because the
world got richer. That is item 78's lesson taken a turn further: rows can lie,
and so can distinct rules. What cannot lie is whether an option, offered, does
anything.

### Wound 3 — nothing ever gives up

The only thing that cleared a goal was ARRIVING at it. No stall rule, no
progress check, no abandoning a place you cannot get to. The modulo hid this
by changing the goal every ten ticks whatever happened.

With a stable target the witness walked into the far corner of a room and
spent **180 ticks** stepping into a wall two cells in front of it — and from
that corner it could not see the window, so it never learned the bench was
alight and the scene it was built to test quietly stopped testing anything.

Fixed, and it needed two halves. **Getting no nearer** is the test rather than
time spent, because a long walk is not a failed one; half a cell of progress
counts, so a body shuffling round a corner is not mistaken for one stuck.
After `WILL["give_up"]` = 80 ticks (2 s, about 27 walking steps) of no
progress, the goal is dropped.

Dropping it is not enough on its own. The menu is built from the same map and
offers the same spot at the next decision, so the body sets off again and
nothing has changed — the give-up has to be REMEMBERED. `p["_baulked"]` holds
the places this body tried and could not reach, and `add()` skips them for
`WILL["baulk_for"]` = 400 ticks. That is the ordinary kind of knowledge — *I
tried that and could not get there* — and it fades, because a doorway that was
blocked ten seconds ago may not be now.

Tested against the honest version of the witness: a body, and somewhere it can
name and cannot reach, because the way there is bricked up behind its back
after it sets off. It believes the far side is walkable because it SAW it,
which is exactly right — belief is what it looked at, and it has not looked
since.

### What this says about the method

Four real bugs, all of them downstream of one number nobody had picked on
purpose, and all four invisible while it churned. The suite's 127 tests passed
throughout: the scenes were calibrated ON the churn, so they measured a world
in which no body ever stood still long enough to be stuck, and one of them —
`DRAGGED` — passed because its hero happened to be within arm's reach of the
body on the floor at the instant it first saw the fire. Improve the vision so
it sees the fire thirty ticks earlier and the luck runs out.

### And one thing that is not a bug, noted where it will be needed

`DRAGGED` is the scene that broke under every leg-menu change, and working out
why turned up a gap in the POLICY rather than the physics. A rescuer's sheet
says `{"legs": "go:exit", "hands": "hold"}`. That is a flat percept-to-act
table with one row per limb, so it can say *hold whatever is in reach* and it
cannot say **go to them first, and then hold**. Whether anyone gets rescued
therefore depends on where the rescuer happens to be standing at the instant
the percept fires — which at HEAD was within arm's reach by luck, and with
sharper eyes is thirty ticks earlier and six cells short.

Sequencing is the thing a hierarchical action space is for, and it is the same
gap as the one item 80 records for bargaining: an intention that takes several
decisions to carry out has nowhere to live. Not fixed here — the physics was
the job — but this is the cheapest existing demonstration that it is real, and
it costs nothing to keep the scene as the place it shows up.

Wounds 1 and 4 are the pair worth remembering. A god-channel does not only
give a body knowledge it has not earned — it **removes the pressure that would
have made the honest path work**. The sight-gated line was dead code for as
long as the cheat sat above it, and the only way to find that out was to take
the cheat away and watch everything go quiet.

The general lesson is the one item 78 was already making, arriving from the
other side: **an arbitrary constant does not just make a wrong decision, it
hides every bug downstream of the decision being made at all.**


## 82. Two bodies in contact break each other — a step that forgot to say so

Chasing the last failure out of item 81 walked into something much older, and
much worse, than the thing it was chasing. It is pre-existing — measured
identically at HEAD and after every fix in 81 — and nothing in the suite had
ever asked about it, because no scene had ever put three people in a room.

### What happens

One person in a room is fine: 396 flesh voxels, never airborne, for as long as
you like. Add people:

| people | airborne ticks (of 200) | cells each, all starting at 396 |
| --- | --- | --- |
| 1 | 0 | `[396]` |
| 2 | 0 | `[432, 360]` |
| 3 | 103 | `[569, 0, 369]` |

Two bodies that come into contact **exchange flesh**: one gains 36 voxels and
the other loses the same 36. Three bodies, and one of them is consumed
entirely — 0 cells — while another walks around carrying 569.

`_person_cells` was the first suspect and it was the wrong one. `add_person`'s
docstring still says *"One person per neighborhood for now — the flesh search
is anchor-local"*, but that has not been true for a long time: `_claim_bodies`
resolves every body once a tick, seeds each with the cells it owned a moment
ago, grows every seed one voxel a round TOGETHER so nobody claims before
anybody else, and settles contested flesh by who was nearer. It is careful
code and it is full of measurements from the last time somebody fought this.

The hole is that **a step never told it.** `_walk` and `_shove` translate a
body's voxels across the lattice and update nothing else, so at the next tick
the seed still pointed at the cells the body had just LEFT — air now, filtered
out of the seed — while the body's whole LEADING EDGE stood there owned by
nobody. A neighbour's seed was nearer to that edge than the walker's own
shrunken one, and grew into it. Every step gave a few voxels away.

A landing already did this correctly: when a free body comes down, `_own` is
rewritten to exactly where the flesh came to rest. Walking is just the other
way a body moves, so it gets the same rule — `_carry_own` displaces `_own` by
the same step the flesh took.

| | before | after |
| --- | --- | --- |
| 1 person | `[396]`, 0 airborne | `[396]`, 0 airborne |
| 2 people | `[432, 360]` | `[396, 396]` |
| 3 people | `[569, 0, 369]`, 103/200 airborne | `[396, 396, 396]`, 0 airborne |

The floating torso was downstream of this and not a separate bug. A body
holding less than its own feet fails its balance test, `_law_falls` calls
`_launch` on the half it was handed, and that half comes to rest on its own
legs — the landing test finds the cells below it occupied, so it bobs between
`off.z` of -0.12 and -0.37 for ever. Fix the ownership and it never happens.

### What it is NOT

Mass was never being destroyed. The first measurement here said 131625 units
became 104175, and that was a mistake in the measuring: a free body keeps its
mass under the key `masses`, the probe summed `smass`, and everything in
flight counted as zero. Counted properly the total is 131625 at t0 and 131625
at t299, with `left_mass` at zero throughout.

Worth writing down because it is the second time in this session a measurement
was wrong in a way that pointed at a far more alarming conclusion than the
truth, and what caught it was dumping the object's keys instead of trusting
the name I expected to find.

### Why it mattered

The next thing on the list after the physics is people talking to each other —
bargaining, warning, refusing. Every one of those scenes is two or more people
standing close enough to be in contact, which is exactly the configuration
that made them trade limbs. It was invisible because no scene in the suite had
ever put three people in a room; `test_people_in_a_CROWD_keep_their_own_BODIES`
now does, and checks the voxel count, the airborne count and the mass.


## 83. Somebody else built the menu seam

`archerhume.com/posts/jevs-architecture-unmasked` is one person's analysis of
a system that has not been released, inferred from an API and an adapter, with
the author flagging his own mixture-of-experts reading as speculation. Read it
as that. What makes it worth an entry is that the shape he describes is the
shape we arrived at from the other end, and where it DIFFERS is where our
decisions are.

### The same seam

> shared state encoding · isolated question branches · listwise option
> processing · direct probability readout · ends at prefill, not decoding

That is `policy.pick(situation, menu) -> int`. One encode of the situation,
one branch per limb, options that see each other before the answer, an INDEX
out instead of text. His sharpest line is the one item 76 was built around:

> final answers depend on the complete choice set, not independent scoring

which is exactly why a harvest row carries the whole menu beside the pick, and
why `census` counts `(percept, limb, options)` rather than rows.

### The compute argument is the teacher-budget argument

23,000 tokens of state against 5,000 questions: encoded once it is ~65k
tokens, asked separately it is over 100 million. Ours is smaller — one
situation, five limbs — but it is the same split, and `seam.py` already makes
it for the same reason: `rules()` is the stable half that caches and
`situation()` is the half that varies, kept in different functions and never
concatenated, because a prompt cache is a prefix match.

The constraint that prompted this was "contain the teacher to the smallest
output space so we can query it more". This says the structure that makes a
harvest cheap is the structure that makes LABELLING cheap, and we have already
paid for it.

### Calibration is sitting there unused

He reads the training signal as proper scoring against outcomes — log loss or
Brier — so that a probability means the frequency it claims, and notes the
`confidence` field is post-hoc arithmetic off the max probability rather than
anything learned.

Every harvested row already carries `outcome`: `still inside`, `safe`, `down`.
That is the signal, and nothing reads it yet.

### Where it does NOT help, which is the part to remember

There is nothing in it about hierarchy, planning, action spaces, sequencing or
small models. It is a decision classifier: one question, one distribution,
done. That is precisely the hole item 81 fell into — a rescuer's sheet can say
`hands: hold` and cannot say *go to them first, then hold*, so whether anybody
is saved turns on where the rescuer happened to be standing. A JEV-shaped
model gives us a better `pick`. It does not give us the thing that scene
needs, and adopting the architecture would not quietly solve it.

### One divergence to make on purpose

He gives a reason for keeping question branches ISOLATED:

> Asking whether the customer is angry should not change which queue receives
> the ticket.

We do the opposite deliberately. Each limb is told what the others have
settled on — `already chosen: legs=go:exit` — and that is how "walk to the
door WHILE holding them" composes out of two rows of a table instead of being
written down as a compound response. Isolation buys calibration and
parallelism; the coupling buys whole-body programs.

Worth stating plainly because the two are in tension and the tempting move is
to take his architecture wholesale and lose the composition without noticing.
If the limbs are ever made independent it should be because the coupling was
measured and found not to pay, not because somebody else's ticket-router had
a good reason to be stateless.


## 84. The pointer is the mechanism; Jev-like is the wrapper

Asked whether keeping the POINTER NETWORK framing fits with a Jev-shaped
model. It does, and they are not alternatives: they are the same thing named
at two different layers, and saying so settles several open questions at once.

### Why they are the same thing

A pointer network (Vinyals et al. 2015) makes its output distribution by
attending over the INPUT and reading the attention as the answer, so the
output space is whatever was put in front of it — variable in length, and
addressed by content rather than by an index agreed in advance.

Jev's readout has to work the same way. The options ride in the input (each
branch carries its own instructions and allowed answers), a shared scorer maps
each option's hidden state to one logit, and the softmax runs over however
many there are. A fixed `W` over a fixed label set cannot do
"runtime-defined semantic decisions" or "cardinality up to 255" — there is
nothing to enumerate at training time. Add the listwise step, where options
see each other before scoring, and it is a pointer network with candidate
self-attention.

So: **pointer is the mechanism, Jev-like is the deployment shape** — typed,
prefill-only, shared state, calibrated.

### Three reasons the pointer framing is the one to keep, and they are ours

**It is Ruling 2.** A fixed classification head over `{go:exit, go:roam,
go:step, take, hold, say, ...}` is a property table, and a property table is a
hiding place for cases. A pointer does not enumerate; it scores what it is
handed.

**The meaning is in the ARGUMENT, not the verb.** A real leg menu:

```
['stay', 'jump', 'leap(to the right)', 'go(straight on)',
 'go(back the way it came)']
```

Across one harvest that is 57 distinct option strings over 40 verb tags, and
the gap is small only because harvest worlds are tiny. In a real scene it is
`go(the door north-east)` against `go(the floor south)` — same verb, different
act. A tag classifier collapses those into one class and throws away exactly
what item 81 spent a day establishing: which cell an option points at is where
the decision lives.

**It survives the sim changing.** The roam labels changed twice in one
session. A pointer scores option text it has never seen; a fixed head needs
retraining every time the menu grows. For a project whose claim is emergent
composition, the option set is MEANT to keep growing.

### What Jev adds that a pointer does not

Calibration. A pointer network says nothing about whether its softmax means
anything; RLCD's proper scoring rule says the probability equals the frequency
it claims. Every harvested row already carries `outcome` — `still inside`,
`safe`, `down` — and nothing reads it.

### What to refuse, and why it is free

Branch isolation (see item 83). Our limbs are told what the others chose, and
that belongs in the pointer's QUERY vector, not in the option encodings:
condition the query on `chosen`, score the same options. The coupling and the
pointer are orthogonal, so keeping whole-body programs costs nothing
architecturally.

### Where this already was, and the hole it leaves

The model arc in `docs/data-engine-design.md` is already
bi-encoder → grammar-constrained AST derivation, with menu-select as depth-1.
A bi-encoder's dot product IS a pointer scorer, so this is not a new
direction — it is a name for the one already chosen.

"Depth-1" is the hole. Item 81 found a rescuer's sheet that can say
`hands: hold` and cannot say *go to them first, then hold*, so whether anyone
is saved turned on where the rescuer happened to be standing. A depth-1
pointer has the same limit as the flat table it replaces. Neither pointer
networks nor Jev say anything about sequencing — that is what the grammar half
of the arc is for, and it is still unbuilt.


## 85. A fuzzer, and the three things it found that were not bugs

Five bugs in item 81 and 82 came from running configurations nobody scripted,
so the method was worth automating rather than repeating by hand. `fuzz.py`
and `fuzz2.py` in the scratchpad build harvest worlds, drop extra people into
them, run them for a few hundred ticks and assert things that must be true in
ANY world:

- no NaN or infinity in `smass`, `E`, `fvol`, `fpot`
- a living body that is neither out of the world nor airborne HAS flesh
- nothing stays in the air for ever
- nobody holds one goal for longer than four times `WILL["give_up"]`
- solid mass is conserved when nothing burns

### The one real bug: a landing that places nothing

`_land_body` gives a landed body back to its owner and computes the new anchor
as the mean of the cells it placed. Fast enough, every cell of the landing
pose is off the lattice, `put` is empty, the mean of nothing is NaN, and
`int(NaN)` raises. **One world in 120, at tick 1.** The limb branch eight
lines above already guards exactly this — *"a limb swung off the edge comes
back as nothing"* — and the branch for the man it belongs to did not.

The fix is that he keeps the anchor he had. The FIRST fix was wrong and is
worth recording: it reasoned that an empty landing meant the body had gone
past the edge for good, and marked him dead. Measured, all 291 voxels were
still in `self.bodies`, `gone` was empty, and forty ticks later the flight came
down and put 41 of them back. An empty landing is a landing that did not take.

### And three false alarms, which cost more than the bug did

**"Mass is being destroyed."** 131625 units became 104175. The probe summed
`smass` off a free body, whose key is `masses`, so everything in flight
counted as zero. Mass was exactly conserved.

**"Energy is created from nothing."** Nine worlds in fifty. Every one was a
body falling or a body dragging something. `total_energy()` counts the
lattice, bodies in flight, what went past the edge and what was radiated — and
NOT gravitational potential, and NOT muscle. `_haul_cost` says so itself: *"a
man has blood_o2 and no calorie budget, so the joules he puts into the floor
come from nowhere the sim is counting."* **Mass is a closed ledger; energy is
not.** A conservation check that does not know this reports the sim as broken
when it is the checker that is, so the reasoning now sits in `fuzz2.py` where
the assertion used to be.

**"A man was thrown off the edge of the world."** Above.

### The pattern worth keeping

Four wrong measurements in one session, every one of them pointing at a more
alarming conclusion than the truth, and every one the same mistake: inferring
a cause from an effect instead of reading the state. What caught all four was
dumping the actual object — its keys, its counts, its fields — rather than
trusting the name I expected to find in it.

That is an argument for the fuzzer rather than against it. A checker makes the
claim precise enough to be wrong, which is how a wrong claim gets found in
minutes instead of being written into a roadmap and believed.


## 86. GLiClass: the menu-select model, already built, Apache-2.0

`arxiv.org/abs/2508.07662` — *GLiClass: Generalist Lightweight Model for
Sequence Classification Tasks*, Stepanov, Shtopko, Vodianytskyi, Lukashov,
Yavorskyi, Yaroshenko. Paper is CC BY-NC-ND; the CODE at
`github.com/knowledgator/gliclass` is **Apache-2.0**, with pretrained
checkpoints trained on synthetic data and stated as usable commercially. That
puts it in the port-with-attribution bucket rather than the ideas-only one.

### Why it is the thing, and not merely adjacent to it

GLiNER's trick is that the LABEL SET RIDES IN THE INPUT as text, so one
forward pass scores an arbitrary set of labels chosen at runtime. GLiClass
adapts it from spans to whole sequences. That is the menu seam:

| GLiClass | Mark_1 |
| --- | --- |
| text | `seam.situation()` |
| labels (runtime, arbitrary) | the menu's option strings |
| per-label scores, one pass | `pick` = argmax |

`seam.py` already emits exactly this pair of strings, for its own reasons.

### It also settles an open decision, against us

The paper's rationale rejects three alternatives, and the third one is the
model arc in `docs/data-engine-design.md`:

- generative LLMs — *"inconsistent instruction following and computational
  inefficiency"*
- cross-encoders — *"must process text-label pairs sequentially, significantly
  reducing efficiency with large label sets"*, i.e. one forward pass per option
- embedding methods — *"struggle with complex scenarios involving logical and
  semantic constraints"*

**We planned a bi-encoder.** In a bi-encoder the option is encoded
independently of the situation and the two meet only at a dot product, so the
option vector never sees the fire. `go(the door north-east)` is an excellent
act or a fatal one depending entirely on where the fire is and what the legs
already chose — which is precisely a "logical and semantic constraint". Item
84 argued the pointer must be able to condition on `chosen`; a bi-encoder
cannot, and a uni-encoder can.

So the arc becomes **uni-encoder (GLiNER-shaped) → grammar-constrained
derivation**. Same depth-1 ceiling as item 84 describes; a better depth-1.

### What it does not solve

Sequencing, again. It is one question, one distribution over the options in
front of it — no hierarchy, no options-over-time, nothing that says *go to them
first, then hold*. Item 81's rescuer is still stuck.

### The experiment that costs nothing

A pretrained `gliclass` checkpoint, zero-shot, over the harvest rows already on
disk, scored against `TablePolicy`'s picks. No teacher, no labels, no
training. It answers whether the plumbing works — whether the seam text and
the option strings are a well-formed question — before anything is spent on
labelling. Checkpoints are BERT-scale, which fits the 6 GB budget.

And openjev's number (item 83) says what to expect: a FROZEN 4B reading typed
options got 0.845 against the closed service's 0.883. Most of the value is in
the interface, not the training.


## 87. Two men who fall together end up sharing a body — OPEN

The fuzzer grew two invariants aimed squarely at item 82's class, because that
was the bug that took longest to find:

- no two people own the same voxel
- a person's owned cells are ONE connected lump

The first has never fired. The second fires in **16 of 60 harvest worlds**,
and item 82 is not as fixed as it looked.

### What is measured, and is certain

Harvest seed 34. Two people, `one` at (4,9) and `two` at (13,9), correctly
separate at t0 — x1-7 and x10-16, 396 voxels each. The world drops both of
them. At t18, `one` is re-launched carrying **692 voxels**, most of them the
other man's, and `two` is left with 25. They come down interleaved, x0-9
against x5-12, and hold 658 against 131 for the rest of the run.

Flesh is conserved throughout: 792 voxels, start to finish. This is an
OWNERSHIP failure, not a matter one.

The mechanism at the moment it happens, from instrumenting the claim:

```
[claim t18] one: own=396 of which flesh=396  aloft=False
[claim t18] two: own=181 of which flesh=0    aloft=False
[LAUNCH t18] owner=one cells=692
```

`two`'s deed points at 181 cells that hold no flesh at all, so it seeds
nothing; `one` seeds 396 and the grow gives it everything else going. That is
exactly item 82's shape — matter and deed separated — in the LANDING path
rather than the walking one.

### The cause, found

Wrapping every law and watching the deed cells gave it in two lines:

```
[land t18] two deed set to 396 cells, flesh now = 396
[land t18] two deed set to 181 cells, flesh now = 0
```

`_land_body` runs TWICE for the same man in one tick, because **a person can
own more than one free body** — and the second one carries no flesh at all.
The deed is assigned wholesale, `_own = frozenset(put)`, so the flesh-less
body's landing replaces the good 396-cell claim his real body set moments
before. With no live seed he claims nothing at the next tick and the neighbour
grows into everything.

### And the obvious fix makes it worse, which is the useful part

Built and measured: keep only the man's own FLESH, and ADD to what he has
already put down this tick instead of replacing it. It fixes the deed exactly
— both men hold 396 through every landing.

Then they bounce. Owning 396 each and landed INTERLEAVED, neither has footing,
so `_law_footing` launches both again every third tick. Seed 34 goes from
settling to **airborne 400 ticks out of 400**. Taken back out.

So the merge was HIDING a second defect. Two bodies that come down in the same
space have no footing and the sim has nowhere to put them; owning each other's
flesh gave them one lump that did. Fixing ownership alone trades a wrong answer
for no answer.

**That forces the order of work.** The landing has to stop putting two men in
one place before the deed can be made honest. Doing them in the other order
produces a livelock, and this is now measured rather than guessed.

Not guessing further. The diagnosis is worth more than another patch bolted on
in the dark.

### Two changes made along the way, kept because they are right anyway

Neither fixes the above. The suite is green with both (131 tests).

**An empty landing no longer wipes the deed.** `_own = frozenset(put)` ran
BEFORE the empty-landing guard from item 85, so a man halfway through coming
down — 75 voxels on the floor, 25 still in the air — lost his claim on what he
had already put down. Placing nothing must change nothing.

**Someone in the air keeps their deed.** `_claim_bodies` cleared `_own` for
anyone aloft, which threw away the only record of whose flesh is whose across
a landing — and a landing is not atomic. Flesh belonging to an aloft owner is
now marked -2 in the label grid: neither free to be grown into nor a seed to
grow from, it waits for him to finish coming down. Someone who WALKED OUT or
DIED still loses the deed, because they are not coming back for it.

### How widespread, and what else is wrong: nothing else

150 harvest worlds, 300 ticks, an extra two people dropped into each, against
every invariant the fuzzer has:

> **38 of 150 worlds break exactly one invariant, and it is always this one.**

No mass violation. No NaN. No stuck goal. Nothing airborne for ever. No two
people sharing a voxel. Every failure is *"X is in pieces: N of M voxels
connected to the rest"*, and the M is regularly 754 or 658 — one man holding
most of two. **This is the only remaining invariant violation in the sim**, and
it is in a quarter of worlds.

### A third attempt, also taken back out

*A man is not rubble.* Every cell that finds its target blocked rises on its
own until it meets air, which is right for a collapsing wall and wrong for a
person: two bodies coming down together thread their cells through each
other's gaps. So a whole body should look for ONE shift that fits all of it —
the shoulder does not go somewhere the hip could not follow — and rubble,
carried tools and severed limbs keep the per-cell rise because they really are
loose matter.

Built, and it made things worse: bodies stopped landing at all. Then measured,
and the measurement did not support the story — the shift it computes is **0 in
38 of 45 landings and 1 in one**, so the rule is very nearly a no-op, and yet
seed 27 went from 77 airborne ticks in 300 to 288. An effect that does not
follow from the change means the path is not understood well enough to be
edited, so it was reverted rather than tuned until the number looked right.

Three attempts now: fix the deed (livelock), keep the deed while aloft
(neutral), land as a rigid whole (worse, inexplicably). What they have in
common is that each assumed it knew why the landing behaves as it does. The
next attempt should start by reading `_land_body` and `_law_footing` together
until the interaction is actually understood, because every cheap fix has
either done nothing or traded this bug for a worse one.

## THE DIAGNOSIS ABOVE IS WRONG — read on

Everything above treats this as an ownership bug. It is not, and the way to
find that out was the thing three failed patches skipped: reading
`_land_body` and `_law_footing` together until the interaction made sense,
instead of editing whichever line looked guilty.

### What the sim actually says, when asked

Instrumenting `_law_footing` to report WHY each body is launched, on the same
seed 34:

```
t15  one  zlo=1, 396 cells  -> "one overbalances"    -> launched
t16  two  zlo=1, 396 cells  -> "two overbalances"    -> launched
t18  one  zlo=1, 692 cells  -> "one overbalances"    -> launched
```

The reason is **overbalancing, not footing** — and at t15 and t16 both men
have their feet on the ground and **396 cells each, correctly owned, before
any merge has happened**. The 692-voxel merge at t18 is DOWNSTREAM: it is what
two bodies bouncing in the same few columns eventually do to each other.

### The loop, in five lines

1. A body lands upright on a tiny contact patch.
2. `_overbalanced` says its weight is past its feet — correctly.
3. `_law_footing` answers with `_launch`, which TRANSLATES and keeps the pose.
4. It lands in the same upright pose.
5. Back to 2.

Measured across the bounces: `tall=29, long=11, contact_cells=4`, constant,
tick after tick — four voxels holding up six hundred and fifty-eight.

### And the escape is already written, and never reached

`_overbalanced` opens with the right idea:

> ONLY A BODY THAT IS STANDING UP CAN BE TAKEN OFF ITS FEET. Said as geometry
> rather than as a flag: a standing body is TALLER than it is long, and a body
> lying down is not.

A body that has gone over is exempt, which is exactly right. But nothing ever
rotates it flat: the answer to overbalancing is a translation, so **a man who
is falling over never falls over.** `tall` stays 29 and `long` stays 11 for
ever, and the exemption that would end the loop is unreachable.

`_law_footing`'s own docstring warned about this shape for a different test —
*"launched it into the ground it was already on, over and over"* — and
`_overbalanced` walked into it from the other side.

### Why the ownership patch made things worse

Because the merge was accidentally dampening this. A 658-voxel lump of two
interleaved men sometimes passes the balance test where 396 alone does not, so
giving each man his own body back put BOTH of them in the loop: seed 34 went
from 285 airborne ticks in 300 to 400 in 400. The patch was right and the
world underneath it was broken.

### The topple was built, measured, and taken out — and here is what it cost

*If your feet are on something you rotate about them; if they are on nothing
you fall.* One rule, using the promotion `_collapse` already uses for an
unconscious body. It worked on the thing it was aimed at:

| seed | airborne before | after | ownership |
| --- | --- | --- | --- |
| 27 | 77/300 | **53** | — |
| 12 | 84/300 | **76** | `[523, 0]` -> `[523, 47]` |
| 9 | — | 91/300 | **`[396]`**, one whole correct body |
| 23 | — | 212/300 | `[0, 2]` -> **`[428, 314]`** |

And it broke five tests, for two separate reasons, one of them fatal:

**It loses flesh.** *"flesh deforms, it does not fragment: every gram stays
(43875 -> 41925 g)"* — 1950 grams gone. A toppling body sweeps a quarter
circle through space, and every cell of that arc that cannot be placed goes to
`_left_world`. A launch translates and mostly finds room; a rotation does not.
Mass conservation is the invariant this whole approach is justified by, so
this is disqualifying on its own.

**It un-breaks ledges.** *"a STRONGER man drags the same braced man off it
(z=29)"* — he stays up. Rotating about your feet keeps you attached to the
thing you are stood on, which is exactly wrong at the lip of a drop, where
going over your feet means leaving. The distinction is not "are your feet on
something" but "is there anything to topple ONTO", and that is a different
question from the one this rule asks.

Narrowing it to `overbalances` only, leaving `_pulled_over` alone, did not
help: all five still failed. The self-overbalance path is what carries a man
over a ledge.

### What is still true, and worth the next attempt

The diagnosis stands and is the real find: **a man who is falling over never
falls over.** `_overbalanced` exempts a body that has gone over — *"a standing
body is TALLER than it is long, and a body lying down is not"* — and nothing
in the sim can put him in that state, because every response to losing balance
is a translation that preserves the pose.

Two things the next attempt has to answer that this one did not:

1. **Where does the flesh go.** A rotation needs somewhere to sweep through,
   and in a 1.3 m room there is not room. Either the topple has to refuse when
   the arc is blocked, or `_land_body` has to stop paying blocked cells to
   `_left_world`.
2. **Ledge or floor.** Going over your feet ONTO ground is a topple; going
   over them into air is a fall. One test, asked before choosing.

And a third thing, found on the way and separate from both: a body draped
diagonally — `x4-25, y0-10, z1-23`, touching the world at **2 cells out of
396** — passes the standing guard by one voxel, `tall=22` against `long=21`. A
body at 45 degrees has tall about equal to long, so the proxy fails exactly
where it matters, and the draped man is toppled again and re-draped. What
should happen to him is what `_law_footing` already says: *"the parts of him
that are over nothing SAG, and that is the support law's question, not this
one."* That is a third law, and nobody has read it yet.

### The fix, and the order, corrected

1. **An overbalanced body must TOPPLE, not fly.** `_law_footing` should promote
   it to a rotation about its contact edge — the machinery exists, `_land_body`
   already takes a `theta` and the topple path is what a chopped tree uses —
   so that it ends up lying down, where `_overbalanced` already knows to leave
   it alone. This is the one to do first and it is probably the whole bug.
2. **Then re-measure ownership.** The interleaving may simply stop once bodies
   come to rest instead of raining on each other. If it does not, the deed
   patch from this entry is written and measured and ready.

The ownership work in items 82 and 87 is not wasted — `_carry_own` fixed a real
and separate defect — but the thing to fix next is the bouncing, and it was
three patches away from where everyone was looking.

### What the fix will need, in this order

1. **A landing must not put two bodies in the same space.** `_land_body` piles
   its own blocked cells upward as debris, which is right for rubble and wrong
   for a man: it interleaves him with whoever is already there instead of
   refusing, so neither of them can stand. This is the one to do first.
2. **Then the deed follows the matter** — only his flesh, accumulated across
   the several bodies one man can land in a tick. The patch for this is written
   and measured and waiting on (1).

### And what it says about the fuzzer

Four hours of scene-driven work never touched this. Two invariants and sixty
worlds found it in under a minute. The connectivity check is the one that
earns its keep — "is this person one piece" is cheap, and every ownership bug
so far has ended with somebody in two.


## 88. DECIDED: GLiClass as the base, used the way Jev is used — AFTER the sim

The model direction is settled, and its schedule with it.

**The base is GLiClass** (item 86): GLiNER-shaped, labels ride in the input, one
forward pass scores whatever menu it is handed, Apache-2.0 with pretrained
BERT-scale checkpoints that fit the 6 GB budget. It is the pointer network of
item 84 with the listwise property of item 83, already built.

**The shape to use it in is Jev's** (item 83): typed decisions, shared state
encoded once with a branch per limb, probabilities read straight out, no
generation. Jev is closed and we cannot see inside it — so the references are
the evidence, and there are three kinds worth keeping apart:

- TypeSafe's own blog. Primary, and thin on mechanism. Their words on RLCD:
  *"where the claims require extraordinary evidence."* Their 0% hallucination
  figure *"is not empirical"* — true by construction, as ours is.
- Third-party analysis (archerhume). Inference from an API and an adapter.
- `openjev`, MIT. An independent build of the INTERFACE pattern, explicitly
  *"does not reproduce Jev's undisclosed model or training"* — and its number
  is the one that matters: a FROZEN 4B at 0.845 against the closed service's
  0.883.

Triangulate from all three, try things, keep what measures well. Nobody outside
TypeSafe knows what Jev is, so treat every claim about its internals as a
hypothesis to test rather than a design to copy.

**And NOT YET.** The simulation is the gate. Every one of these needs decisions
worth labelling, and today's work is the argument: eight bugs in one session,
seven of them changing what ends up on a menu or who is standing where. A
harvest taken before the sim is right is a dataset of a broken world, and the
seam is spelled `mark1-menu-v1` precisely so that re-labelling is a version
bump and not a guess about which rows are still good.

The first experiment when the gate opens costs nothing: a pretrained gliclass
checkpoint, zero-shot, over the harvest already on disk, scored against
`TablePolicy`'s picks. No teacher, no training — it only asks whether the seam
text and the option strings are a well-formed question.


## 89. GLiFormer — the hierarchy half, for when we get there

`knowledgator.com/research/gliformer`, same people as GLiClass (item 86).
Noted now because it is the FIRST thing read that touches the hole items 83,
84 and 86 all left open, and we will not remember why it mattered by the time
we need it.

**What it adds: structure without generation.** A schema-conditioned DeBERTa
encoder where runtime labels match against ANCHORS standing for entities,
relations, classes or records — and for hierarchical tasks it *"grounds field
values directly in the source, assigns them to unordered record anchors, and
predicts parent-child relations before deterministically assembling nested
JSON"*.

That is the shape the grammar half of the model arc needs. Everything else we
have read is depth-1: one question, one distribution over what is in front of
you. Item 81's rescuer is stuck at depth-1 — a sheet can say `hands: hold` and
cannot say *go to them first, then hold*. Parent-child relations assembled
deterministically is a way to say the second thing without a decoding loop.

**What to hold back.** No licence and no code stated — a PDF and an email
address, against GLiClass's Apache-2.0 and public checkpoints. So GLiClass
stays the base we build on (item 88) and this is the shape to watch. Their
headline is 91.10% F1 against their own GPT-5.6-luna baseline at 91.96%, so
the pitch is cost and not accuracy; the 95.8x speedup is an ESTIMATE, stated
as "under autoregressive assumptions"; and 547 ms median on CPU against 69 ms
on GPU is fine for a turn and not for anything reactive.

### And what reading actually costs us, measured

Worth writing down beside it, because it decides how much model we need:

| | |
| --- | --- |
| situation text, varies per decision | 37 words median, 56 max |
| stable half, cached once a run | 144 words |
| **total per decision** | **~180 words, ~250 tokens** |
| options per menu | 3 median, 7 max |
| distinct option strings ever seen | 57 |
| distinct word types in all situation text | 413, with ONE hapax |

**~256 tokens of context and a 413-word closed vocabulary.** The ten commonest
words are `nothing, 0.000, 0, 1, 1.000, percept, hurt, burn, smoke, blood_o2`
— that is not prose, it is a FORM. Reading a situation is reading a
fixed-format record.

So the reading splits in two, and only half of it needs a language model:

- **The situation needs almost nothing.** Any BERT-scale encoder has orders of
  magnitude more capacity than a 413-word form asks for.
- **The options are the whole reason for pretrained language.** `go(the door
  north-east)` against `go(the floor south)` is where the meaning is, and the
  option set GROWS as the sim grows — those strings changed twice in one day.
  Scoring an option string never seen in training is exactly the GLiNER trick,
  and it is the only part of this that needs inherited language.


## 90. Nine renders, and the two things no assertion was ever going to catch

The fuzzer had gone quiet — 150 worlds, one failure kind, all of it bug 8 —
and the suite was green at 131. That is exactly the moment to render, because
an invariant only catches what somebody thought to assert, and every visual
bug this project has ever had came from watching a video: the glass that broke
weird, the arms that looked wrong, the person who broke in rescue.

Nine scenes, serially, 640x360, frames -> PNGs -> mp4, with the encode run
unconditionally after every re-render because skipping it once deleted four
mp4s. All nine came through clean. The bugs were in what they SAID.

### A forge cannot reach forging temperature

`forge` prints its own state every 100 ticks, and it never melts the lead its
docstring promises will run:

```
t100: lead 45.4 kg solid, 0 ml molten, 220 C
t300: lead 45.4 kg solid, 0 ml molten, 253 C
t600: lead 45.4 kg solid, 0 ml molten, 263 C
```

Lead melts at 327. Iron stalls at 256 against a melting point of 1538. Tin is
the only one that runs, and only because 232 is under the ceiling.

Where the heat stops, measured:

| | temperature |
| --- | --- |
| coals elsewhere in the bed | ~505 C |
| coals TOUCHING the bar | ~375 C, fallen from 740 |
| the lead bar itself | 258 C, steady from t200 |

The metal digs a **cold pocket**. It cools its own neighbours by 365 degrees,
the rest of the bed does not resupply them, and the bar then sits 117 degrees
below even the pocket. Lead's melting point falls BETWEEN the bar and the
coals touching it, so there is no arrangement of this scene in which it melts.

Sealed, the whole thing dies at 48 C — combustion needs the open sky for
oxygen — so the bed really is burning, 1063 cells of it, at 880 C, for 700
ticks, with a lead bar buried in it at 266.

This is underneath item 80's third entry. `TOUGH` having no temperature term
would matter if metal ever got hot; **it never gets hot**, so smithing is
impossible one layer earlier than anyone had looked.

### Smoke does not rise

`house_fire` exists to show one thing, and says so: *"smoke banks down from the
ceiling, the air at head height goes foul while the knee-level air stays
breathable (crawl low)"*. Measured at t899, 1690 g of smoke in a room 43
voxels tall:

| band | smoke |
| --- | --- |
| z1-10, the floor | 305 g |
| z10-20 | 356 g |
| z20-30 | 231 g |
| z30-44, the ceiling | 286 g |

Flat. Marginally MORE at knee height than at the ceiling. The smoke diffuses
evenly through the whole room instead of banking, so beside the person it is
0.003 g a cell — too thin to blind anybody, poison anybody or read on any
instrument. Hence the readouts after 850 ticks of house fire: `skin 20 C`,
`burns 0%`, `smoke load 0.00`, blood O2 83%.

The inverted oxygen follows from it and is not a second bug: the fire is low —
lamp at z1, bed at z5-7 — so it eats the air near the floor, and with no smoke
layer to foul the head there is nothing to make the high air worse. O2 reads
76% at head against 57% at the knee. The physics is self-consistent; what is
missing is BUOYANCY.

**And it undercuts an assumption made earlier today.** Item 81's wound 4 was
found because a witness standing in a smoke layer had all 335 of its rays
stopped at the first cell, and the fix gives light its own reach so a body can
see UNDER the smoke. That fix is right and the measurement was real — but it
was taken in a 50x26x40 room, where the same smoke is concentrated enough to
be opaque. In a house-sized room there is no layer to see under. Seeing under
smoke is only worth anything once smoke banks.

### Still open, seen in the logs, not yet chased

- `tree_fell` asks its own question and answers it badly: *"tree top at z=45
  (was 54) — did it TOPPLE or just sink?"* It sank nine voxels. That is item
  80's "a felled body falls like a plank", and it is the same thread as the
  topple work reverted in item 87.
- `alchemist` holds `glass standing 298` flat for 800 ticks with acid in
  contact, and `drop_test` holds `glass cells 194 (intact 6)` flat from t20.
  Neither is obviously wrong; both want somebody who knows what the scene is
  for to look at the video.

### What the sweep is worth

Two bugs and two leads, in scenes that pass every test in the suite and every
invariant in the fuzzer, found by reading what the scenes print about
themselves. Neither bug is subtle once seen — a forge that cannot melt lead, a
fire whose smoke does not rise — and neither was findable any other way,
because no assertion in the codebase says smoke should be thicker at the
ceiling than at the floor.


## 91. Everything that bounces is the same bug, and it has no memory

Six complaints came back off the renders. Two of them — glassware popping in
and out, a tree's leaves glitching — turn out to be the same defect as the
bodies in item 87, and measuring it finished the diagnosis that item started.

### The glass is not shattering and is not being created

`alchemist` holds exactly **298 glass cells at every tick**. What moves is
where they are:

```
tick 725:  298 on lattice |   0 in flight | 0 free bodies
tick 758:  124 on lattice | 174 in flight | 2 free bodies
tick 800:  298 on lattice |   0 in flight | 0 free bodies
```

A 174-cell vessel is promoted to a free body, lands, is promoted again. On
screen that is popping out of existence and back. `burning_tree` does the same
with 685 leaves in one lump.

### And the cure is already written, and is only a delay

`_law_torque` topples a lump whose centre of mass leaves its contact
footprint. `_land_body` already knows this can loop, and says so:

> a vessel on a shelf edge stopped dead at 0.41 every time, was set down, was
> found unbalanced again on the next tick, and toppled again, for ever. It
> cost alchemist 862 saved frames instead of 130

The fix was `theta < 0.7 -> self._torque_sleep = 20`. Measured now, it works
exactly as designed and the loop survives it:

```
t69   70 glass  theta=0.06  [wedged -> sleep 20]
t70   53 glass  theta=0.41  sleep=19
t92  101 glass  theta=0.06  [wedged -> sleep 20]
t115 117 glass  theta=0.06  ...
```

**22 of 23 landings are caught as wedged.** The cycle is 23 ticks: sleep 20,
wake, find the same unbalanced lump, topple again. Sleeping does not change
the configuration, so waking finds it unchanged.

Two other things the numbers show. A topple of `theta=0.06` is 3.4 degrees —
a twitch, not a fall, and nothing distinguishes it from a real one. And
`_torque_sleep` is GLOBAL: one wedged vessel stops the torque law for every
object in the world.

### The one sentence that covers all of it

**The sim answers "this is unstable" by promoting it to a free body, promoting
does not make it stable, and nothing remembers that the answer has already
been tried.**

- a body: `_law_footing` sees overbalance, launches, it lands in the same
  attitude, repeat (item 87)
- a vessel: `_law_torque` sees the COM outside the footprint, topples, it
  lands still outside, repeat
- 685 leaves: the same

Item 81's wound 3 was this exact shape one layer up — *nothing ever gives up*
on a goal it cannot reach — and the fix there needed two halves: notice no
progress, and REMEMBER it, because otherwise the menu offers the same place
again next tick. `p["_baulked"]` is that memory for goals. The torque and
footing laws have no equivalent.

### The fix this points at

Not a longer sleep. A memory, per lump rather than global: a topple that lands
wedged in the same place as the last one has proved it cannot fall, and should
settle until something disturbs it. That is `_baulked` for matter.

Two design questions it needs answering first, which is why it is written down
rather than built at the end of a long session:

1. **What identifies a lump across ticks?** Goals are coordinates and hash
   easily. A lump is a set of cells that changes every time it moves. A
   per-cell field — a twelfth, beside `edge` — is the grid-native answer and
   costs nothing to read.
2. **What clears it?** A settled lump must wake when something strikes it,
   burns it, or removes what it rests on. Too eager and the bug returns; too
   lazy and the world goes rigid.

## 92. Matter comes off the grid: MPM, and the three bugs between here and a falling plank

The user drew three things and asked whether the sim could do them. Read
against the code, they got three different answers:

- **a glass slab breaking on a table edge** — mostly works. `_shatter`
  scatters mass into the 4 neighbours (8 more when `over >= 2`), marks them
  `rubble`, and rubble takes support only from directly beneath, so pieces
  past the edge do fall to the floor. What is missing is that shards are
  exactly as big as the parent's own cells, and that **flying fragments wound
  nobody** — `_shatter` moves mass, never energy.
- **a bar melting over an overhang** — does not work at all. `_law_melt`
  converts grams to millilitres in place and the cell stays `mat != AIR` until
  `smass <= 0.01`, while `_law_support` reads PRESENCE, not amount. A
  99%-melted cell holds its full span and then vanishes. Step, never droop.
- **a plank tipping off a table and landing turned** — no such thing exists.
  Of eleven per-cell fields not one is directional. `_topple` knows the
  rotation while the body is in flight, which is what `I` is for, and
  `np.round` throws it away on landing.

The verdict the user reached, and it is right: adding damage, offset and
facing as three more per-cell fields is building a bad particle out of three
worse ones, each needing its own law to talk to the other eleven. That is
Ruling 2 with extra steps.

**So matter comes off the grid.** The important half of the sentence is that
MPM KEEPS the lattice — particles carry the matter, the grid is scratch for
the momentum solve. Fire, conduction, smoke, oxygen, fluid, acid, sight and
every mind go on reading exactly what they read today.

It also dissolves item 91. Promote/demote exists because matter has to leave
the lattice to move properly; the bounce is that promoting does not make it
stable and nothing remembers. MPM is what you get when matter STAYS promoted.
There is no cycle left to loop.

### What is built

`src/mpm/core.py` — MLS-MPM in 3-D numpy, ported from Hu et al. (SIGGRAPH
2018, MIT) and the Taichi examples (Apache-2.0), both checked before taking
anything. `src/mpm/scenes.py`, `tools/render/render_mpm.py`,
`tests/test_mpm.py`.

Measured, not assumed: free fall matches one-half g t squared to three
decimals; mass is exact through an impact; restitution falls monotonically
with damping (0.3 / 1.0 / 3.0 gives e = 0.56 / 0.46 / 0.23).

### The three bugs, each found by a measurement that contradicted the story

1. **Pure elasticity cannot lose energy.** The block bounced HIGHER the
   stiffer it was made and was still airborne after a second and a half. Real
   matter has internal friction; a Kelvin-Voigt viscous stress is the cause,
   and it reads the strain RATE, so free fall stays undamped — which is what
   the first test guards.
2. **Damping has a step limit of its own.** It is a diffusion, and `cfl_dt`
   only knew about wave speed. Past about `damp = 0.5` the deformation
   gradient went to infinity inside a dozen substeps and the SVD stopped
   converging: a blow-up that read as a bad material and was a bad time step.
3. **A sticky table is a clamp.** Zeroing velocity at wall nodes glued
   whatever touched it, and the symptom was exact — a plank with most of
   itself off a table bobbed between 4 and 19 degrees for three seconds. A
   clamped cantilever oscillates; a resting one tips.

### And a fourth, which is the one worth remembering

With a proper surface collider the soft plank fell, and the STIFF one froze at
1.6 degrees — starting in free fall and stopping dead.

The first explanation was contact smearing: the P2G stencil reaches one and a
half cells, so particles out to x=1.45 still touch the table-edge node at
x=1.30, and the plank's centre of mass sat at exactly 1.45. It balanced on the
lie. Shifting the plank 10 cm further out tipped it to 44 degrees, which fit.

**Halving the grid spacing did not help** — 5120 particles, 688 seconds, still
0.5 degrees. That killed the explanation and found the real one: normals came
from a single gradient of the mask, so only the SKIN of a wall had one and
every interior node was hard-frozen. A particle's stencil reaches inside the
solid it rests on, so a plank standing on a table was gathering "hold still"
from the middle of the table. Blurring the mask before taking the gradient
gives every wall node a normal, and the plank then goes 0 -> 14 -> 50 -> 87
degrees and slides clear: the user's drawing, including the part where it ends
up on its end.

Worth keeping because the plausible diagnosis and the true one predicted the
same result for the cheap test and opposite results for the expensive one.

### Not yet answered

- **Cost.** 640 particles at E = 3.3e7 Pa is 235 substeps per 25 ms tick and
  about 85 seconds a scene. Real wood is 1.1e10 and would want a thousand.
  Softening is the standard graphics compromise and what it buys is stated:
  things ring slower than real matter. Whether that is affordable at scene
  size is untested.
- **People.** `_claim_bodies`, `_own`, footing and pose are grid-native in a
  way the glass is not, and what a person IS in particle terms is unanswered.
  Bodies stay on the lattice until it is.
- The glass and the melting bar, which are the other two drawings.

> **Correction, 2026-09-20.** Every standalone MPM measurement in item 92 was
> made with `add_wall` one cell too tall (`slice(..., ceil(hi)+1)`), so each
> "floor to 0.2 m" was really 0.3 m and each "table to 1.2 m" was 1.3 m. The
> half-cell-low distance field of the time put the surface at 0.25 / 1.25 and
> hid it. Every qualitative result stands — restitution monotone in damping,
> the plank tipping, damage localising at the edge, breakage rising with drop
> height — but the drops were 5 cm shorter and rest heights 5 cm higher than
> the text says. Found while marrying MPM to the lattice (item 94), where the
> lattice's own walls are the collider and there is no `add_wall` to be wrong.

## 93. DECIDED: what MPM is for, and the shape of the phase that proves it

Grilled to a shared understanding on 2026-09-20, after a day of building
MLS-MPM (item 92) and a second day finding five plumbing bugs that each
presented as physics. Seven decisions, in dependency order.

**1. MPM is for FACTS — and motion is a fact.** The product is a DM
narrating a simulated world, so the sim's output is what a DM can say. MPM
earns its place on fragment size (a shard smaller than a voxel cannot exist
on the lattice, so cannot be thrown or lodge in anyone), on where things end
up, and on whether a thing broke at all under a real load. The first draft
of this dismissed motion as cosmetic and the user corrected it: `tree_fell`
SINKS nine voxels instead of toppling, the alchemist vat bounces for ever, a
toppled vessel lands where it started. Those are wrong facts — "the tree sank
into the ground" — because in a sim, motion decides what happens next.

**2. Explicit bonds, replacing the continuum.** Object type IS a bond
structure: wood is strong along the grain and weak across, glass is uniform
and brittle, rope is a chain, cloth a sheet. "What holds this together"
becomes data rather than a property table, which is Ruling 2 done properly.
And crack propagation — measured as MISSING in the continuum, 2 of 92
columns severed, threefold refinement moving it only to 3.1% — arrives as a
consequence: a bond breaks, its neighbours take the load, they break. The
user's reason is Ruling 1 applied to method choice: mimic the cause closely
and the cases stop needing code. Cost: bonds REPLACE the per-particle SVD
rather than adding to it, so plausibly neutral; numba is a prerequisite,
not an optimisation. To be measured before anything is built on it.

**3. Two one-way doors, no demotion.** Static bulk — walls, floor, ground —
is voxel by NEVER BEING PROMOTED. Disturbed matter becomes particles and
NEVER COMES BACK. Voxels are ~13x cheaper in memory and identical in compute
when asleep, so the saving demotion would buy is real in principle and zero
in practice until a whole world has burned. What demotion would COST is the
one property that matters: a thing never returns different from how it
settled. Rasterising a plank leaning at 87 degrees back to a staircase is
item 91's bounce in a new coat.

**4. Done = (a) + (b).** (a) the three drawings — plank tipping off a table,
glass breaking across an edge, a bar sagging as it melts. (b) the five voxel
scenes that are broken BECAUSE of voxel stiffness — `tree_fell`, alchemist,
`drop_test`, `arms`, `crouch` — fixed through the seam. (b) is the test that
the marriage works; (a) is part of why the voxel scenes fail. People are
next phase, on the strength of a feasibility answer: a person is today a
kinematic puppet on the lattice, and in a particle world that puppet is a
MOVING COLLIDER — `wall` plus a velocity — with grips as attachment
constraints. The mind is untouched. A struck person promotes to a free body
for the fall, which is what `_law_footing` already does.

**5. Offline-correct, cost measured.** This phase's scenes are rendered and
judged, not bounded. Done includes a stated number — microseconds per awake
particle per substep, with numba — so the next phase (sleeping, live) has a
target rather than a hope. "Get it right, then make it cheap."

**6. The free-body protocol is deleted.** `_launch`, `_topple`,
`_land_body`, `_law_bodies`, the wedge probe, `_torque_sleep` — about a
thousand lines and a dozen roadmap items — replaced by "this cluster leaves
the lattice as particles." Two representations of moving matter is worse
than either. The voxel tests split: those asserting an OUTCOME (the tree
fell, the vat burst, the man landed hurt) are the contract and must keep
passing; those asserting a MECHANISM (a body has a theta, a landing is
wedged) break by construction and are rewritten or removed. Listed before
any are touched.

**7. The seam is built now, sized to what MPM needs.** The user's regret
was not having replaceable modules behind a generic interface; the seam is
the smallest thing that makes that true. The voxel sim talks to "the thing
that moves matter" through four calls — `promote(cells, mats, masses, E)`,
`step(dt)`, `rasterize() -> (mat, smass, E)`, `collide(cells, velocity)` —
and MPM is the first implementation. The free-body protocol is not wrapped
as a second one; a seam with one side is a contract, not a plugin system.
Generalised only when a second solver actually arrives.

### Order of work

1. **The seam interface** — an afternoon; everything plugs into it.
2. **`tree_fell` through the seam with the continuum MPM that exists** — a
   topple, which the plank already proves. Marries the two halves in days
   and de-risks the whole plan before the method changes.
3. **Bonds** — the method change, before the fracture scenes, so they marry
   to the final method once rather than twice.
4. **The fracture scenes** — `drop_test`, alchemist — which need bonds.
5. **`arms`, `crouch`** — which need the person-as-collider, the small
   version.
6. **The melting bar** — which needs the heat seam: particles carrying
   energy, exchanged through the lattice field.
7. **Cost measured at every step, with numba.**

### What this phase does NOT do

Sleeping. Live budgets. People as bonded particles. Demotion. A second
solver. Each is named here so that when one is reached for, it is reached
for on purpose.

## 94. The seam is in, a tree falls through it, and the seven things found on the way

Order-of-work steps 1 and 2 of item 93. `src/mpm/matter.py` is the seam —
`promote`, `step`, `unrasterize`/`rasterize`, the ledger queries — and
`_topple` goes through it for any cluster that is nobody's. A hinged pole
tips, lands, slides, stops, and lies toward its notch: both tree tests pass
with their only edit being `not w.bodies` -> `not w.matter.aloft()`.

**The voxel suite went 9 failures -> 0, and not one mechanism-test needed
rewriting.** Item 93 budgeted a dozen. The tests that read `theta` and
`.bodies` exercise `_launch`, which is untouched this phase; every failure was
one of two seam bugs below, and both were in the ledger and the routing, not
in the tests.

### Seven fixes, none of them physics

Each presented as physics — a pole that would not stop, glass that would not
break, a body that vanished — and each was a convention crossing the seam:
a unit, an index, an ownership, a statistic. The physics was right once the
plumbing was. In order found:

1. **The domain clamp was a floor.** `np.clip(x, 2 dx, ...)` held particles
   two cells up: an invisible, frictionless floor a cell ABOVE a one-cell
   floor. The felled pole rested on it and slid for ever at 0.689 m/s,
   `hit = 0 of 62`. The standalone block test had been PASSING on it — its
   fixture built no wall at all. Now half a cell, the stencil's true need.
2. **The distance field was half a cell low.** `_sample` mapped position to
   index as `x/dx`, treating node i as the cell's low corner; the EDT is
   between cell CENTRES, so the surface sat at 0.05 for a floor whose face is
   at 0.10. Every particle would have sunk 5 cm into everything. `x/dx - 1/2`.
3. **`add_wall` was one cell too tall** — `slice(..., ceil(hi)+1)`. Hidden by
   #2, which put the surface back near where it was asked for. Every
   standalone measurement in item 92 was on tables 5 cm taller than stated
   (annotated there).
4. **Promoted matter was jittered by a quarter of a cell**, inherited from
   `add_box` where it breaks a fresh lattice's standing wave. A voxel is
   flat-faced; the quarter turned the pole's underside into 2.5 cm teeth and
   it rested on the floor through ONE particle at a time — one hit per
   substep of 29 along the face. Five percent keeps the face; 3-6 hits.
5. **The ledger counted drawn cells twice.** `total_mass` summed the lattice
   AND `matter.mass()`, and rasterized particle matter is on both. Flesh at
   exactly 2x. `moving` cells are the solver's grams and are counted there.
6. **A passed-out man became particles.** `_topple` was redirected wholesale,
   so `_collapse`'s keel went through the seam, the body left the lattice,
   and the mind lost it — nobody ever came round. People stay on the lattice
   this phase (item 93, decision 4): the seam takes a cluster only when
   `owner is None`; an owned one keeps the old path, restored from git.
7. **`aloft` asked whether ANY particle moved.** A settled pole twitches at a
   mean of 0.006 m/s with one particle now and then past 0.02; that called it
   airborne. A body question gets a body statistic: the mean, which falls two
   orders of magnitude between falling and lying still.

### Glass at the corrected geometry

56 of 480 cracked, 56 within 20 cm of the table edge, 0 beyond 40 cm. The
localisation is total, and the overhang broke off and reached the floor.

### Contact, measured four ways at once

The MPM suite's three failures after the geometry fix were all contact, and
a single block dropped on a floor plus a slab dropped across a table edge,
each run with one contact component switched off at a time, said this:

| grid constraint on | block rest (want 0.400) | bounce e | slab shove |
|---|---|---|---|
| the face node       | 0.498 | 0.48 | 1.37 m/s |
| inside nodes only   | 0.386 | 0.39 | 1.39 |
| nothing (particle only) | 0.298 | 0.00 | 0.49 |

Three facts, each one a fix:

- **The grid must hold the body.** Per-particle contact alone corrects a
  particle after it has crossed the surface; the layer above keeps arriving
  through the grid, and an undamaged, unyielding block compacted to sixty
  percent of its height with its bottom pinned exactly on the face.
- **On nodes strictly INSIDE, not the face node.** The stencil reaches a
  cell and a half, so a body held at the face node rests a full cell high.
  Inside-only rests at 0.386 and keeps the bounce. The old one-cell-too-tall
  `add_wall` had been supplying exactly this by accident.
- **Read the field at the nodes.** `phi[i]` is cell i, centred at
  (i + 1/2) dx; the momentum grid's node i is at i dx. Reading cell values
  for nodes shifts every surface half a cell.

And one that was NOT a fix, recorded because its story was so plausible:
projecting onto the distance field's unit normal at interior nodes near a
convex corner gives a diagonal, and blocking each axis on its own by the
sign of its own gradient component removes the diagonal — correct for
axis-aligned voxel geometry, kept — and the slab still went sideways at
1.16 m/s. Four mechanisms for that shove have now been proposed and killed
by measurement: contact smearing, the blurred normal, the corner node, the
interior wedge. What finally answered it was a probe on the grid itself at
the first-contact substep, splitting the x-momentum on every node into the
`m v` term and the `affine` term, before and after the constraint:

    all nodes    px(m v) = +5.30   px(affine) =  0.00   pre = +5.30  post = +6.47
    inside       px(m v) = -0.22   px(affine) = -0.95   pre = -1.17  post =  0.00
    free         px(m v) = +5.52   px(affine) = +0.95   pre = +6.47  post = +6.47

**The +x was already in the particle velocities before the grid touched
them.** It came from the one per-particle contact substep just before: a
particle resting on the tabletop within a cell of the edge samples a normal
that trilinearly BLENDS the corner cell's diagonal, arrives at -6 m/s
straight down against (0.7, 0, 0.7), and leaves at +1.8 m/s sideways.
Twenty of them is +5 kg m/s in one substep. The grid then adds +1.17 by
zeroing the leftward stress-reaction at held nodes — correct Coulomb, and
physical (a body pushing on a held patch moves the other way) — which is why
every grid configuration made it WORSE and none of them CAUSED it. The
mechanism I had fixed on the grid, per-axis blocking, I had left undone on
the particle side. Same fix, same rule: block each component by the sign of
its own gradient component, and push out along the dominant axis rather
than the blended diagonal.

The four killed mechanisms were all about the corner, and the corner was
right; they were wrong about WHICH corner — the one in the field, sampled at
particles, not the one on the grid.

And it was never a transient. Run to rest instead of to the tick the test
happened to measure, a slab with seven tenths of its length on the table
and its centre of mass 20 cm inside the edge LEFT THE TABLE in every
configuration: with the grid constraint 290 of 336 supported particles
still up at t=28, none at t=40, all 480 on the floor by t=89; without it,
314, then 229, then none by t=60. The question the test asked at t=28 was
the right question about the wrong instant.

Per-axis blocking at the particle, the fix the probe seemed to point at,
changed nothing — far-end velocities identical to three decimals — and
that was the fifth mechanism killed. What settled it was a per-substep
x-momentum BUDGET through contact rather than a snapshot at its start:

    sub   total_px  d_px | affine(free)  constraint(inside)  particle
      0      +6.5   +1.2 |      +0.9          +1.2            0.0
     20     +73.0  +42.6 |      +4.6          +4.4            0.1
     40    +135.2  +23.7 |      +2.7          +1.8            0.0
    100    +201.5   +6.7 |      +1.5          +0.9           -0.4

`constraint(inside)` matches `d_px` substep for substep; `particle` is
nothing. The +x enters through the grid constraint at nodes under the
tabletop, every substep, and it enters because the slab is BENDING over the
edge: its bottom fibre in compression pushes -x onto nodes inside the table,
its top fibre in tension pulls +x onto free nodes, and the stencil straddles
the surface. Friction at the inside nodes — with a normal impulse from a
6 m/s arrival, `keep` is zero — glues the contact patch and eats the -x
half of an internal pair; the +x half stays and moves everything else. The
far end, airborne (`far_vz = +3.26`), flips up and toward the pivot like the
handle of a rake stepped on at the head.

That is correct physics for a rubber slab. `stiff = 1e-3` glass IS a
rubber slab — the same softening artifact named on the first day, "a
compliant slab drapes over an edge." Real glass does not whip; it sits, or
it breaks.

Measured, and the softening story is only a small part of it. At 1e-3 the
bottom layer of the slab nearest the edge — the contact patch, which should
be the one thing held still — slides at +0.5 to +1.1 m/s and LIFTS from
z = 1.222 to 1.280, off the table. At ten times the stiffness it still
slides at +0.6 and lifts to 1.31; the slab's centre goes from +1.25 to
+0.83 m/s. Stiffness moves the number a third. The collider owns the rest.

And the collider's fault is the stencil: a node's reach is a cell and a
half, so inside nodes go on constraining a slab that has bounced 8 cm
clear of the surface, and grid FRICTION at those nodes — with the normal
impulse of a 6 m/s arrival, `keep` is zero — deletes every tangential
component every substep. A table pushing sideways on something it is no
longer touching. The division that follows: the grid holds the body UP,
normal component only, and never holds ON; friction lives at the particle,
which knows whether it is actually touching — and per-particle friction is
already what stopped the felled pole.

### Contact, settled: the node does not guess its face

That division did not survive the next measurement, and neither did the
four rules tried after it. Each was a node deciding which face its matter
was against, and a node one cell across cannot know. The table's corner
node has two faces:

| rule at the node | block rest / e | slab landing kick | slab stays? | pole |
|---|---|---|---|---|
| hold both faces (original)         | 0.386 / 0.38 | +1.19 | no | ok |
| hold flat faces only (`|g| > 0.9`) | 0.386 / 0.38 | +0.05, then +1.15 | no | ok |
| hold the face driven hardest       | — | +1.29 | no | ok |
| nothing; particle stops itself before P2G | 0.316 / 0.00 | +0.19 | yes | ok |
| flat by grid, corner by what the particle was stopped against | 0.387 / 0.36 | +0.33 → +1.1 | no | ok |
| **node refuses the directions its matter TOUCHES in** | **0.394 / 0.26** | **+0.70 (friction on the bulge)** | **no, at this stiffness** | **ok** |

Each row was killed by a probe, and each probe was a cheaper question than
the theory it killed:

- *Hold both faces* — a per-node budget put every gram of the shove on
  corner nodes (gradient 0.71, 0.71) and none on flat ones. A slab sliding
  -x across the top edge had its -x deleted by a node whose contact face
  was the top.
- *Flat faces only* — the kick vanished at landing and came back two
  ticks later, now at the particle collider. The strip within a cell of the
  edge was held by nothing at the grid; the particle collider pushed it
  out, the grid handed it -6 m/s again, and it paid impact-scale friction
  every substep: +78 kg m/s in one tick, all at x = 1.46.
- *Particle stops itself before P2G, no grid* — the slab stayed, and the
  block never rebounded: the floor's reaction never enters. Deleting
  momentum at the particle leaves the stress with nothing to push
  against; the block crept down 8 cm and its internal chatter grew until a
  tick cost 387 substeps at rest.
- *Corner inherits from what the particle was stopped against* — the flat
  rule stops the flat part's particles first, so corner nodes are told
  nothing, the edge strip sinks while the rest bounces, and every drop
  height from 0.4 m up ended with all 480 on the floor.

The rule that stands: the geometry is axis-aligned, so which face a
particle touches is a LOOKUP, not a gradient — a particle within a tenth
of a cell of a wall cell along an axis, and not itself in one, touches
that face. The signal rides P2G beside mass and momentum, and a node
refuses only the directions its own matter is touching in. A flat top
reads "something I hold is touching a face below it" — not most of it,
since a face node holds the touching layer and the free layers above.
Momentum is still deleted at the NODE, after the stress force has been
added; that is what makes the support elastic. Friction lives at the node
too, sized by the normal impulse the node actually removed; without it
nothing penetrated far enough to feel the particle's, and the felled pole
slid off the world. Half a cell of skin held matter three centimetres in
the air (rest 0.434); a tenth is a centimetre.

Under it: block rest 0.394, e = 0.26; pole lands and stops; the pane's
damage sits at the edge (34 of 49 within 10 cm of it, where the
inside-node rule with an elastic bounce had 42 of 150 — bulk shatter on
impact); a slab landing beside the corner with no overhang takes no
sideways kick.

What is left on the straddling slab is friction on the impact bulge: with
every friction off the landing kick is -0.001 m/s; on, +0.70, all of it
in the tick of impact. A soft slab's underside spreads on landing; the
supported side spreads against friction, the overhang spreads free, and
the difference is a push toward the overhang. It is a third the size at
ten times the stiffness and would be nothing at glass's real stiffness.
That is the softening knob, named on the first day, and not the
collider. The shove test now drops the slab 5 cm short of the edge, which
is the claim the collider can make; the straddling scene is recorded here
and belongs to the stiffness/cost item.

### Three scenes re-rendered through the seam — and one silently broken

Re-rendering the back catalogue was not a victory lap. It found a bug no
test had: **the seam was DELETING FLUID.**

A vat is glass cells around AIR cells holding acid. `_topple` hands the
glass to `promote()`; the acid stays on the lattice, as it always did. But
`rasterize()` then drew the particles' fluid — which is none — over every
cell they landed in, hollow included, and `unrasterize()` cleared the
fluid of every cell it had drawn. A third of the alchemist's acid
(1000 ml of 3000) was gone two ticks after the shelf cracked, and the
cascade the scene exists to show never started: no spill, no fumes, no
eaten wood. The scene printed `acid 2000/3000` for eight hundred ticks and
nothing in 131 tests objected, because the tests ask about mass and about
bodies, and this was fluid VOLUME in cells nobody was watching.

The fix is one sentence: the seam writes its fluid only into cells
carrying none, and remembers which cells those were, so `unrasterize`
clears only what it wrote. Acid now 2964 of 3000 (the remainder eaten, as
it should be), wood 79200 → 76831 g, fumes throughout.

And writing the missing test found the OTHER half, which no scene showed:
`total_fluid` summed `fvol[(fl == f) & ~moving]`, excluding every cell the
seam had drawn into. That is the right rule for MASS — the seam's own
matter is in those cells and `matter.mass()` counts it — but fluid
ownership is finer. A vat's hollow holds liquid the LATTICE owns while the
glass around it has become particles, and the falling vat draws solid back
over that hollow; `moving` there means "a solid of ours passed through",
not "the liquid is ours". The total therefore flickered between 500 ml and
0 from tick to tick depending on whether a particle happened to cover the
cell that instant. Both halves are the same sentence: **the seam owns
exactly the fluid it wrote, and nothing else** — `drawn_fluid`, not
`drawn`, not `moving`.

Worth keeping, because it is the fourth time this phase: **the ledgers
that caught the double-counted flesh, and the suites that caught the lost
body, both passed this.** A conserved quantity nobody totals is not
conserved, and `total_fluid` existed the whole time — nothing compared it
across the seam. Now `test_FLUID_A_TOPPLING_VESSEL_CARRIES_is_not_...`
does, tick by tick and in both directions, so invention fails as loudly as
loss.


- **tree_fell** — topples toward the notch, breaks into chunks on the way
  down, lies beside its canopy. Nothing sinks.
- **tree_hinge** — falls toward the notch, and the middle of the fall shows
  what `stiff = 1e-4` is: the trunk DRAPES over the hinge like a hose
  before it parts. Wood at a ten-thousandth of its modulus is rubber. Real
  wood hinges and cracks. This is the stiffness knob of item 93's cost
  step, now visible in a picture rather than a number, and it is the
  strongest argument yet for bonds (decision 2), which carry stiffness
  without the wave-speed step limit.
- **alchemist** — the cascade runs again once the fluid bug above is
  fixed: vat A tips off the cracked shelf, the acid spills down the
  shelving, pools on the bench and floor, and eats wood for eight hundred
  ticks while fumes rise off it. This one DOES go through the seam (one
  cluster promoted, 52 particles), and the vat falls and lands as a body
  rather than as a column of cells.
- **drop_test** — does NOT go through the seam at all, and never did: a
  column with nothing under it falls straight down by the support law,
  cell by cell, the very voxel free-fall this phase exists to replace. The
  landing shatter reduces cell mass without changing the material, so the
  vat looked whole before and looks whole now. Item 93 already queues
  drop_test and alchemist for the seam after bonds; the support law's
  free-column fall should hand its cluster to `promote()` the way
  `_topple` does. Not done here.

### Two representations, for now

`_launch` and `_law_bodies` remain for flights, jumps, limb swings and any
cluster with an owner. That is the transition item 93 allowed and it ends at
order-of-work step 5, when `arms` and `crouch` migrate and people become the
moving collider. Until then a moving thing takes one of two paths, and which
one is decided by a single `if owner is None`.

### Cost, offline — and who sets it

The pole: 62 particles, 43 substeps a tick, about a tenth of a second a tick.

`tree_fell` at 5 cm, profiled over three ticks after the tree has left the
lattice: **3261 particles, 215 substeps a tick, 6.1 s a tick** — 8.7 µs per
particle per substep, up from the 6 µs of a plain block because every
particle now pays an SVD, an eigen-decomposition and a fracture check. A
full run is about fifteen minutes.

Two things in that number worth more than the number:

- **The 215 is the CANOPY'S.** Wood's step limit is 4.7e-4 s; the stiffened
  leaf at `damp = 3.0` has a VISCOUS limit of 1.3e-4, and the scene steps at
  the pace of its most restrictive material. The softest, most-damped thing
  in a scene sets the bill for everything in it. It stopped being "not
  this phase's job" one run later: the stiff-canopy `tree_fell` timed out
  at 1700 s. Leaf damping is now 1.0, wood's own value, which puts the
  wood back in charge of the step. Per-material substepping is the real
  fix and is still deferred.
- **Where a substep goes:** about half in the 27-neighbour P2G/G2P loops,
  21% in the SVD, 6% in `eigvalsh`, the rest einsum and clip. All of it is
  numpy overhead on three thousand particles. That is the numba case stated
  in item 93 — and bonds (decision 2) delete the SVD outright rather than
  speeding it up.

The tree was also 3661 cells, not the ~1200 estimated: 2966 leaves to 696
wood. Four fifths of a tree is canopy, and the canopy is what it costs.

### Confirmed

**Voxel suite: 132 of 132**, through the seam, under the settled contact
model — 131 plus the fluid-conservation test the re-render earned.
**MPM suite: 12 of 12.** Three tests were re-stated to what was
measured rather than what was hoped: the pane allows one stray far crack
in twenty (the elastic bounce meeting the weakest flaw; 1 of 56); the
shove test drops its slab 5 cm short of the edge and allows 0.25 m/s
(measured 0.21, a fifth of the corner node's 1.12, and it stays on the
table); the harder-fall test asks for an order of magnitude between its
low and high drops and lets the top two agree within noise (8, 89, 83 —
at the highest drop the damage moved INTO the hinge, 81 of 83 within
20 cm, rather than growing).

`tree_fell`, rendered: **it topples toward the notch.** The trunk hinges,
breaks into chunks on the way down and lies beside its canopy at z = 19
cells of 45; nothing sinks. The render reads the lattice after
`rasterize()`, so the fallen trunk shows as stacked blocks rather than the
particles it is — a particle renderer for the seam is a small item, not
this one.

## 95. DECIDED: the scenario list, non-magic first (2026-09-22)

A survey of what comes up at a D&D table or in a LitRPG, against what the
sim can do. The user's call: **build the non-magic scenarios one at a time,
and keep magic and the engineering changes as todos.** Each build lands as
a test plus a scene, and the scene starts only from things someone in the
world could do — drop, push, strike, pour, ignite — never from the author
deleting voxels.

### Build order, non-magic

0. **Green first.** Free fall through the seam, shape-matched clusters and
   vessels that carry their contents went in together and broke about a
   dozen voxel tests and tripled suite time. Nothing is built on red.
1. **Wounds.** A blade or a club on flesh and on armour: a cut opens flesh,
   a blow bruises through it, armour turns an edge and spreads a blow. The
   most common event in D&D and the sim has none of it.
2. **Bleeding.** Blood as a fluid leaving a wound; losing enough of it
   drops a person; pressure on the wound slows it.
3. **Projectiles and cover.** Arrows, bolts and thrown things as small
   fast bodies; what they hit decides whether they stop, stick or pass.
   A thrown oil flask is this plus the vessels from step 0.
4. **Sound and stealth.** Impacts, breaking glass and footsteps make
   noise that travels through air and is muffled by walls; sleepers wake,
   guards turn. Speech already carries.
5. **Light.** A torch or lamp lights a radius and sight is limited by it.
6. **Water and bodies.** Drowning, swimming, sinking in armour.
7. **Cold.** Water freezes, ice is slippery and thin ice breaks; a body
   loses heat — hypothermia, and the campfire that stops it.
8. **Mechanisms.** Doors forced, barred and wedged; locks; levers,
   pressure plates, portcullises; pit traps and floors that give way.
9. **Explosions.** A powder keg as a fast-burning source feeding the air
   field that already exists.
10. **Heavy things moving.** A rolling boulder, a rope and what hangs on
    it, a knocked-out pillar bringing a ceiling down.
11. **The rest.** Encumbrance, scent and tracking, brewing with reagents
    at temperature, crafting quality at the forge.

### Todo: magic (deferred)

Every spell is a SOURCE acting through laws that already exist, never a
special case (Ruling 1): fire bolt and fireball (heat + pressure), cone of
cold (heat sink; needs step 7), lightning (needs electricity and
conduction through metal), gust and telekinesis (air field, body force),
wall of stone and create water (conjured matter — needs a source/sink
ledger so conservation still holds), force barriers (a collider made of
nothing), feather fall (the parachute's drag), monsters of other stuff
(acid slimes, stone golems, undead that do not breathe, fire breath), and
the LitRPG System reaching into the physical world (DCC floor collapse,
cultivation, farming weather).

### Todo: engineering

- Conservation checked every tick in debug runs, for every material,
  fluid and energy total — every seam bug this phase was a total nobody
  added up.
- Split `sim.py` (8,000 lines, 187 methods) into law modules with declared
  reads and writes, so law order is explicit.
- One material record for both representations; derive span and
  toughness from modulus, strength and density (Ruling 2).
- SI units inside the sim, converted only at the edges.
- Rigid bonded clusters by default for solids; continuum only where
  deformation matters.
- Port the solver's hot loops to numba or taichi.
- Two test tiers: fast invariants in seconds, slow scenes separately.
- People as articulated bodies in the same solver as everything else.
- A sparse grid for static bulk, dense only where things are active.
- Render particles directly, not the lattice they are drawn back onto.
- Define the interface between the DM layer and the sim.

## 96. Wounds and bleeding: one quantity, three layers, and a fluid (steps 0-2)

### Step 0, green again

Free fall through the seam broke twelve voxel tests and tripled suite time.
Every failure was about falling, because the lattice fall law already
carries air drag, splashes and the blow to whatever gets landed on, and the
particle solver has none of those. It is behind a switch now
(`World.free_fall_to_seam`, off) until the solver has them.

The shape-matched clusters had a real bug the tree tests caught: moving
positions toward the rigid fit fought the solver's own elastic stress, and
every round of that fight added speed. A toppled pole landed, all 62 pieces
broke at once, and it was flung back across the world at 7.7 m/s. The fix
projects VELOCITIES onto rigid motion instead, weighted by mass times
rigidity. That conserves momentum and angular momentum exactly and can
only remove kinetic energy (the change is sum m*r*(r-2)|e|^2 <= 0). Bonds
part on impact stress, rho*c*dv against strength, which is the rule the
fracture law already uses. The pole now topples, lands toward the notch
and rests. Both suites went back to green and back to ten minutes.

### Step 1, wounds

A blow is energy arriving on a contact area, met by armour, then skin,
then tissue. There is no "slashing" or "bludgeoning" in the code:

- **Skin** parts at 0.1 J/mm^2 of contact (the wound-ballistics figure).
  An axe's declared edge (2e-4 m^2) needs 20 J. A club's 5 cm face needs
  250 J, so the same 60 J opens one and bruises with the other.
- **Depth** is the energy left over, divided by the ~1.5 MPa it takes to
  keep an edge moving through muscle, times the contact area. It is capped
  at the limb's own thickness, and a cut that reaches it goes through.
- **Armour** is a declared sub-voxel layer on a bone, because a 1.5 mm
  plate is below what a 5 cm lattice can draw. That is the same argument
  that let an edge declare its area (Ruling 2 q1). An edge gets through
  only by shearing a plug out: yield x area x thickness x 3. An axe needs
  about 225 J and is turned. A bodkin point needs about 11 J and goes
  through. That is the real historical argument about arrows and plate,
  arrived at rather than looked up. Plate spreads what it stops over its
  own area. Mail stops the edge but hardly spreads the blow, which is why
  it was worn over padding.
- **Weakness** is not a penalty table. A cut is a plane through the limb,
  and muscle past it pulls on nothing, so the limb's section minus the cut
  gives a smaller force from the same section-times-stress every limb's
  strength is already made of. A cut arm swings weaker, a cut leg pushes
  less, and the other side is untouched.
- **Where** a wound is: on the person, located on a bone, not on the
  lattice. A wound moves with the arm it is in, and a per-cell field would
  have to be carried by every law that moves matter. That is the class of
  bug the particle seam kept hitting.

Found on the way, and kept as a test because it is true: **too close to
cut.** The swing builds speed through its arc, so the same axe lands 6.7 J
on a man 45 cm away and 38.6 J on one at 70 cm, the edge of reach. The
first bounces off his forearm. The second opens his upper arm 6 cm deep.
Nobody wrote that. It is the swing law meeting the wound law.

### Step 2, bleeding

Blood is a fluid, the ninth. An open wound bleeds through its cut plane
(depth x length) at a pressure that falls as the blood runs out. Blood is
poured onto the lattice at the wound's current place, the lowest cell of
the wounded bone with air beside it. From there the fluid law runs it
down to the floor and pools it. `total_fluid(BLOOD)` counts blood still
inside people, and a test holds it constant to the millilitre.

Clotting seals a fixed area per second, about a square centimetre a
minute. So a 10 cm scratch stops in about three minutes and a deep cut
would take an hour and a half, which is the same process at a different
ratio. A hand pressed on a wound cuts the flow tenfold. The physics is in;
the action of pressing is a todo.

Blood loss is a shortage of oxygen, not a hit point. The brain gets the
blood's oxygen times a perfusion factor that falls as volume falls, and
faints and dies on the same lines as a man in smoke. The perfusion curve's
two constants are set to put those lines at the ATLS classes: collapse at
about a third lost, death past a half.

### Still open from these two steps

- Pressing on a wound, and tourniquets, as actions someone can choose.
- Broken bones from blunt blows. Energy past a bone's toughness should
  stop the limb bearing load.
- A limb cut through should come off. The wound says `through`, and the
  lattice does not yet detach the cells.
- Blood is not yet miscible with water, does not dry, and does not stain.


## 97. Projectiles, sound, and three bugs the suite found in them (steps 3-4)

### Step 3, projectiles and cover

An arrow is one cell of matter in flight, released by `loose()` with a speed,
an edge area and a frontal area it declares, because a shaft is narrower than
any cell. Nothing about arrows is written anywhere else.

- **Swept flight.** A tick is 25 ms and an arrow at 50 m/s crosses forty
  cells in one. Checked only where it ended up, it passed through a man
  untouched. Flight is now walked a cell at a time and stops at the first
  thing in the way.
- **What stops it pays.** A thing with no owner spends its energy on what
  it hit, through the same wound and shatter laws as a swing. The scrape
  (a wall takes the sideways part of the motion) now pays that part too; it
  used to vanish, and the first arrow loosed met a forearm and dropped
  gently at the man's feet.
- **Cover and armour fall out.** A plank takes the arrow. A bodkin (a small
  edge) goes through plate and a broadhead does not. Drag reads the
  declared area, so an arrow keeps its speed and a cube of the same mass
  does not.
- **Catching needs time.** Catching was a force rule, and men caught arrows
  at 50 m/s. It now also needs a reaction time, 0.2 s.
- Two engine bugs found on the way: `_person_at` read claims from before
  this tick's contacts, and a menu dedupe compared rows holding arrays by
  value.

### Step 4, sound and stealth

Anything that strikes anything makes a noise. How loud depends on how much
the two materials ring rather than damp, over a 50 ms impact. Glass on stone
lands at about 105 dB and a footstep on stone at about 60. The noise travels
by the mass law the shout already used: a stone wall muffles more than
wood, read off mass per area rather than a table. A sleeper wakes at 60 dB
and an awake body notices 35. Its eyes go toward the noise. Creeping halves
the pace and quarters the energy of each footfall, which is 6 dB, and that
halves the distance at which a guard can notice it.

Two fixes to the ear were needed before any of that measured true. The ear
sat inside the head, and a footstep started inside the walker's legs, so the
mass law charged 46 dB for the listener's own skull. Neither body's own
flesh is a wall now.

### What the full suite found (six failures, three causes)

- **A bare noise outranked a meaning.** A shout is also a noise, and
  `hears_noise` was checked before the loop that reads what a line MEANS.
  So "Fire!" arrived as "something, over there", and the hearer turned to
  look instead of running. Bare noise is now checked last.
- **Every percept committed the body.** A committed body stops sensing. A
  man heard a crackle, turned to look, and was frozen for good: he burned
  without ever seeing the fire he had turned toward. `ORIENTING` percepts
  (just `hears_noise` so far) turn the head and commit to nothing.
- **The fall tests were standing at the edge of the world.** A man knocked
  out in a 24-cell world, when he is 30 cells long lying down, keeled into
  the edge and left cells of himself outside it. That is booked to `gone`,
  which is correct: the edge is a void on purpose, and two tests say so
  (what LEAVES THE WORLD is counted). My first fix made the edge stop a
  topple, and it broke exactly those two. The real fix was to widen the
  fall and grip test worlds so nobody lies down against the edge.
- **A noise re-decided the whole body.** A pane broken by a man's own fist
  kept him deciding again every tick as the shards landed. He relaunched
  the swing before the last one came down, until a cell of the arm came
  away. Three changes: a blow's noise belongs to the striker (like a
  footstep), a noise asks only the eyes, and "swing" is not offered while
  that man's arm is still in flight.
  Asking only the eyes also exposed a test that had leaned on the old
  freeze. In the grip test, the unheld man only ever stood still at the lip
  because a noise stopped him. Now he wanders off before the floor goes,
  so the test tells both men to stay put.

Also found: a slumped man propped against the world edge overbalanced
3,925 times in the duel. `_overbalanced` now asks whether the way he would
go over is blocked by solid matter that is not his, or by the edge. If it
is, he is leaning, and leaning is stable.

### Bodies turn to walk (raised by the renders)

Every person was built with shoulders along x and never rotated. `facing`
was only a direction for the eyes and the swing, so every render showed a
crowd facing one way, and a man walking east walked along his own shoulder
line. `_turn` (exact quarter turns, a permutation of cells) existed, and
nothing called it.

Now a body squares up to where its next step goes, before it takes the
step, and the turn uses up that stride. `p["chest"]` records which way the
flesh faces (built facing +y, right arm on +x), and `_turn` keeps it
current. A body standing still keeps whatever way the scene built it, which
is also how a scene says which way someone faces. A body holding or
dragging something, or being dragged, does not turn, because what it holds
would be left behind.

I tried squaring up every body every tick first. It broke 17 tests. Nearly
all of them were scenes built with a man facing along his shoulder line,
so his right arm was in front of him with the target placed there.
Squared up, the arm is to his side and a forward swing misses a post dead
ahead. That is truer, but it is a different question from those tests.
Turning only to walk kept them. What still moved:

- A test that stepped once under the default policy (so he roamed, so he
  turned) and then read geometry off the built frame: the shoulder-socket
  and crouch tests. They now stand still. Turned bodies crouch fine in all
  four orientations; checked.
- The rescue scene depends on placement (item 81), and turning moved the
  lucky spot. Of ten hero positions swept, only (16,12) gets both men out.
- A walker that roams can end where it started. The sleeper test now
  measures how far he got, not where he ended.

**Open, found here:** dragging a turned, unconscious man out left two of
his foot cells on the floor (a separate lump). It shows only in the one
orientation measured, and is not yet diagnosed. The per-owner claims flood
fill of item 82 is the first suspect.

**Also open:** a man wedged against the edge of the world re-tips every
~45 ticks. Harmless since nothing is lost, and seen only in worlds narrower
than a man lying down.

## 98. The mind lab: the social half, tested apart from the sim

`lab/` is a separate environment (its own venv, its own requirements) that
never imports `src/` and is never imported by it. What crosses is a card
file: a state as text, typed questions (Laya's form: `choice`, `score`,
`noul`), and gold answers. The sim will write cards in the same format when
it can produce social situations. Until then `lab/cards/social_v1.py` holds
60 hand-labelled ones.

- **READ** (30 cards): what is a line doing, to whom, how hostile is it,
  and does it ask for something?
- **MIND** (30 cards): what does this person do next, how afraid are they,
  and do they believe it? These are a GM's call, written so the state
  settles the answer. They are provisional until the sim can score
  outcomes.

Every score prints beside the guessing baseline. Two leaks were found
through it before any model ran. The right target was usually listed
first, and `sort_keys` re-sorted the options after they were shuffled.

### First numbers, zero-shot, one model at a time on the 6 GB card

| | intent | hostility | asks (auc) | reply | fear | believes (auc) | ms/card | VRAM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| guess | 10% | 40% | — | 40% | 53% | — | 0 | 0 |
| Laya | **60%** | 47% | 0.70 | 27% | 57% | 0.46 | 62 | 2.4 GB |
| Von | 50% | 43% | **0.89** | 20% | 23% | 0.67 | 232 | 3.0 GB |
| GLiClass | 37% | 20% | 0.46 | 37% | 17% | 0.46 | 48 | 0.6 GB |
| Qwen3-0.6B picker | 47% | 23% | 0.63 | 10% | 27% | 0.64 | 447 | 1.2 GB |

**READING is solved well enough to build on; JUDGING is not.** Every model
beats guessing on what a line is doing, and Laya reads intent at six times
chance. None beats "always comply" on what a person does next. That is the
split the design already predicted: parsing has a right answer that lives
in the words, while judgement lives in the state and in what follows from
it. So decisions come from the sim, and these models do what they are good
at: reading a player's line into typed acts. MIND is the gap to label (see
item 88: a teacher, then the sim's own outcomes).

A zero-shot yes/no is often ranked right and cut wrong. Von put requests
above non-requests 89% of the time and still called most of them no.
Accuracy at 0.5 cannot see that, so `run.py` reports AUC as well.

### Speaking, then hearing it back

`lab/speak.py` takes a MIND card's decided act, has a small LM say it in
one line, has Laya read only the line, and checks with GLiNER for names the
scene never held.

| voice | act carried | fear carried (±1) | invents a name | ms/line |
| --- | --- | --- | --- | --- |
| Qwen3-0.6B | 43-53% | 90-93% | ~3% | 225-264 |
| Reyna-RP 0.5B | 40% | 97% | 0% | 644 |

Qwen narrates ("I feel terrified", "I strike first") rather than speaking.
Reyna speaks, but often from the wrong side of the scene (m05 answers as
the stranger). Fear survives the round trip almost always; the act gets
through about half the time. The next lever is the voice rather than the
reader: give it the decided act as words to paraphrase, not as a
description to act out.

### Not in the lab yet

- **jevlike**: an architecture and trainer with no general checkpoint.
  Useful once there is data (the labelled MIND set).
- **GLiFormer**: no public code or weights.
- **Jev**: closed.

### Judged case by case instead of against the key

An answer key only measures agreement with whoever wrote it. So the models
answered freely: the deciders pick, and the voices choose AND speak
(`lab/free.py`, as `DO: / SAY:`). Every answer was then read against the
person and the scene, side by side in `lab/results/review_v1.md`. Each got a
verdict: ok, arguable, or wrong. Verdicts and reasons are in
`lab/verdicts/judge_v1.py`.

| | reasonable (ok + arguable) | ok only |
| --- | --- | --- |
| Laya, deciding | 67% | 11/30 |
| GLiClass, deciding | 67% | 14/30 |
| Von, deciding | 47% | 8/30 |
| Qwen picker, deciding | 47% | 4/30 |
| **Qwen3-0.6B voice, free** | **67%** | 11/30 |
| Reyna 0.5B voice, free | 23% | 2/30 |

What that shows that the key hid:

- **The deciders are better than 27% said, and they miss in one shared
  way.** Laya and GLiClass pick something defensible two times in three.
  Where all four fail together, the right act turns on a TRAIT, not the
  surface of the scene. Every one of them has the healer sworn to heal
  refuse the dying man (m12). Every one misreads praise given to a vain
  host (m04). Every one has a lone merchant defy three drawn bows (m18).
  They read the threat words and miss the person.
- **Each model has a habit, and the habit is most of its errors.** The Qwen
  picker answers "call for help" 28 times out of 30. Von reads nearly every
  line as "deceive" and answers "lie" to half the MIND cards. That looks
  like scoring by option length or prior, not reading.
- **A small generating voice reads character better than the encoders.**
  Qwen, deciding for itself, got m04 and m12 right where every decider was
  wrong. It fails differently: it narrates, answers as the wrong person
  (m30), or ignores the physical facts (m08: the goblin runs out of a dead
  end).
- **Reyna is not usable.** It breaks character ("as an AI language
  model"), speaks as the other person, and lectures.

So the split is: an encoder reads the player's words into typed acts; a
small decoder, given the person and the scene, is the better judge of
character; and the sim checks whatever either says against what is
physically possible, which is exactly what m08 needed.

### Laya plays a grid

The quiz gave Laya our generic verbs. The real test is to give it the kind
of menu our sim produces and watch it act. `lab/grid.py` is a small 2D world
with that shape. Each tick every awake person gets one state and one menu
per limb (legs, hands, mouth). Every option is built from what is really
there and carries its argument: "go to the door (locked)", "take hold of
Pim", "offer your silver coin for Mira's loaf of bread". The world decides
what happens: fire spreads by a rule, a locked door opens only for the key,
a trade happens only if the other side accepts it. Each scenario states
its good outcome in world terms, and I checked that each one can be won by
playing it by hand. `lab/show.py` prints a run as a play-by-play.

| scenario (good outcome) | random (5 seeds) | Laya (3) | what Laya actually did |
| --- | --- | --- | --- |
| fire_escape (gets out) | 1/5 | 3/3 | walks to the door |
| douse (fire out) | 3/5 | 3/3 | **fetches both buckets, walks to the fire, throws; meaningful** |
| rescue (carries Pim out) | 0/5 | 0/3 | walks past her unconscious brother and out the door |
| locked_door (key, unlock, out) | 0/5 | 0/3 | walks into the locked door for 40 ticks and never looks at the key |
| market (one trade) | 0/5 | 0/3 | hands the loaf and coin back and forth every tick; with memory, the "fair-minded" baker ends up hitting him (0.94) |
| guard (no single answer) | — | — | the guard, whose goal mentions the door, walks out through it at tick 1 |

**It matches words, not situations.** Every win is a scenario where the
next right act is also the option whose words best match the goal. Brenna
wants to save her inn from the fire, and "throw the water on the fire" is
right there. Every loss needs a step first that the goal does not name:
the key before the door, lifting Pim before leaving. The guard's goal says
"door", so he goes to the door.

**It does not use feedback.** I gave every body one line saying what it did
last tick and what came of it ("your legs could not take you any further").
Nothing changed: Kael read it forty times and went for the door forty
times. It also shouts "Fire!" every tick in an empty room.

This is the depth-1 hole of items 84, 86 and 89 made concrete. A pointer
scorer, however good, answers "which of these best fits", not "what has
to happen first". Laya is the right reader for a menu and the wrong thing
to drive a body. The sim's own layering already assumes that: reflexes and
drives choose, and the model reads and speaks. Sequencing (GLiFormer's
parent-child records, or a planner over the menu) is what sits between.

**What Laya actually reads.** I held the locked door fixed and varied only
the state (`--prose` renders it as sentences):

| state given | door | key |
| --- | --- | --- |
| our key: value form, as played | 0.49 | 0.18 |
| the same, after 39 ticks against the door and told "your legs could not take you any further" | 0.50 | 0.12 |
| the form with the goal rewritten to "get the key" | 0.52 | 0.26 |
| **empty string**, or "banana" | 0.30 | 0.19 |
| one sentence: "Kael wants the key. The key is 4 north-west. He picks up keys." | 0.16 | **0.74** |

It is not truncation: 359 tokens against a 512 limit. It leans toward
"go to the door" from the option text alone, before it has read anything.
The state moves that lean only when a sentence all but names the answer.
Writing the whole state as prose did not help: it lost the douse it had
won, swinging between "go to the bucket" and "go toward the fire" every
tick as each step changed which one was nearer, and never picked a bucket
up. So the goal-word matching above is too generous. It is mostly a prior
over option strings, with the state as a weak nudge.

### A real room, primitive actions (`lab/room.py`)

The grid's options each read as an answer, which measured wording. So I
built an 18x9 inn room with a written description, furniture (bed, table,
chest, washstand, shelf, fireplace), loose things (letter, candle, knife,
rope, bread), an old woman who answers questions, and a sleeping dog. The
iron key is inside the closed chest. The letter on the table says where
it is, and so does the old woman if asked. Every option is a primitive:

- legs: step north, south, east or west
- hands: take, drop, open, close, search, read, and "use X on Y" for every
  held thing on every adjacent thing
- mouth: quiet, greet, ask about the key, the door or the chest

Unlocking is "use the iron key on the door", offered beside "use the
candle on the door". Getting out takes 7 links, and each is recorded as a
milestone. A scripted player gets out in 28 ticks through the same menus.

All three questions in one call came to 1,139 tokens against Laya's 512
limit, so most of the input would have been cut unseen. Each limb is
therefore asked alone (at most 508 tokens), and the log records the count.

| 3 seeds each | learned where the key is | reached the chest | anything further |
| --- | --- | --- | --- |
| random | 5/5 (of 5) | 1/5 | 1/5 opened and took the key, never reached the door |
| Laya | 0/3 | 0/3 | — |
| Laya, "what next?" then each limb asked about that | 0/3 | 0/3 | — |
| random, do-nothing options removed | 3/3 | 2/3 | — |
| Laya, do-nothing options removed | 0/3 | 0/3 | — |
| Laya "what next?", do-nothing removed | 0/3 | 0/3 | — |

What it did:

- **Given the null act, it takes it, 80 ticks of 80.** Stay, keep,
  quiet. This is not list position: moved last, "stay" still wins (0.38).
- **With no "stay" on the menu, the four steps are nearly even** (0.32 /
  0.29 / 0.22 / 0.17), whatever the room says. With the do-nothing
  options removed, it walks north to the wall, east to the corner, and
  steps east/west in that corner for the rest of the run.
- **Asked "what next?" first, it answers "the door" 80 times of 80,** then
  stays (or, with stay removed, picks up the candle, puts it down, and
  uses it on the table, over and over).
- **It has the atomic skill in isolation.** "Kael wants to reach the
  chest. The chest is 5 west of Kael." gets "step west": 8 of 8 right
  across four directions and two distances, at 0.46 to 0.94. In the room,
  among fourteen things at their distances, that skill does not surface.

**Verdict, zero-shot:** in a real room with composable actions Laya is
inert, or walks into a corner. It does worse than random, which at least
asks the old woman and sometimes reaches the chest. Its one demonstrated
competence is a single short mapping: one target, one relation, one
question. Anything that needs picking the relevant thing out of a scene,
or holding a sub-goal across ticks, is absent. To drive a body it would
need training on our own traces, and the sequencing layer would still sit
above it.

### Can a formulation rescue zero-shot? (`lab/rooms.py` + `lab/suite.py`)

Randomized rooms. A TRAIN and a TEST name pool share no container, item,
key or person. An expert uses only what the state shows, and escapes
400/400. Four formulations in a 2x2: `opts` makes each option say what it
brings you to ("one step closer to the chest"); `view` gives detail only for
what is near or known about. Every formulation is run with a word-overlap
matcher as the leakage control. All scores are on TEST-pool rooms.

| formulation, picker | target | legs | at the key's container (of 30) | took the key | out |
| --- | --- | --- | --- | --- | --- |
| any, random | 11-13% | 22-24% | 11-15 | 3-8 | 0 |
| base, words | 34% | 23% | 13 | 3 | 0 |
| base, Laya | 33% | 42% | 7 | 2 | 0 |
| opts, words | 32% | 29% | **24** | 6 | 0 |
| opts, Laya | 39% | 41% | 1 | 0 | 0 |
| view, Laya | **42%** | 43% | 5 | 2 | 0 |
| both, words | 32% | 30% | 22 | 9 | 0 |
| both, Laya | 38% | 43% | 1 | 0 | 0 |

No formulation passes the rule (Laya beats the word matcher on both
decisions and rooms). On "what next" Laya is the word matcher, give or take
eight points. Enriching the options helps the matcher reach the chest (13 to
24 of 30) and makes Laya worse (7 to 1). Random reaches the chest more often
than Laya does.

The user's diagnosis is the right one. Supplying what happened, in a form
the model can use, turns into writing the answer in the form it was trained
to recognize. Laya was trained on one-shot decisions whose answer is in the
text. A room is sequential, and its purpose lives in history. The room
measures the formulation more than the model. Laya should be tested on
the slots our design actually gives a model: grounding a player's line
into an act and a target among many present entities, and one NPC reaction
over the sim's real menu. A room is a fair test only for a model trained
on sequential decisions, such as Open-Jev (Doom and WikiRacing in its
training tasks).

Heat: this run pushed the CPU to 96 C despite a 4-thread cap and a
watchdog that pauses the job at 93 C. Long GPU jobs belong on the other
machine.


## 99. The pointer grammar: a scope, not a decision (2026-09-24)

Roadmap 98 showed that flat option lists and zero-shot picking will not
carry the model direction. So the questions D9 left open are now written
up as a scope with experiments: `docs/pointer-grammar-scope.md`. The
questions are what to embed and cache (Q1), how slots combine beyond a
dot product (Q2), which slot is decided first (Q3), how the past gets back
in (Q4), and whether `verb(role: filler)` is the right unit of choice at all
(Q5). Each has an experiment (E1 to E6) and a pass mark, run in that order,
with a word-overlap matcher as the leak check on every table.


## 100. DECIDED: how small models should run characters (plan v3, 2026-09-28)

The model work now follows `docs/character-models-plan.md`. It is written in
Simplified Technical English, as asked, and was reviewed three times. What
changed from the pointer-grammar scope (item 99):

- **No single correct answer.** A choice is judged on three questions: is it
  valid, does it make sense for this character, does it add to the scene.
  Every test case carries a set of acceptable options, not one.
- **Isolate causes.** Why Laya degrades with list size is tested one factor
  at a time, and every model runs under several scoring methods (all
  together, one at a time, two at a time, read apart, shortlist then
  choose). That separates the method from the model.
- **Player input is a grouped pick-list,** so no model reads free text from
  the player. The model jobs are running characters, wording speech,
  deciding undefined effects ("rule of cool") and making things when
  needed (what is in the chest).
- **Personality is researched before it is chosen.** Text in the input is
  the baseline, not the answer.
- **A judge learns from the user in rating rounds.** A large model writes
  the starting set; the judge asks about the samples it is least sure of.
- **The locked room is retired as a test.** It tested unspecified problem
  solving, not social behavior.


## 101. Creativity tests, three layers (2026-09-28)

Creative = unexpected AND still works AND in character. Random is not
creative, so every test pairs "unusual" with "valid".

- **Layer 1, resourcefulness** (`lab/methods/creative.py`, 8 options, 60
  cases per cell). The obvious tool is replaced by an unusual one that
  still works (the chisel cuts the rope, the yoke weighs the silver). The
  unusual labels are hand-picked per need, with a "doubt" list kept out of
  every list; `lab/results/methods/creative_labels.md` is for the audit.
  Result: every model loses 10 to 42 points when the tool is unusual. Best:
  openjev 2B, 50% (obvious 77%). Most models are at about 35%. When the
  other options are near misses (they only LOOK useful), every model is at
  or below chance (5 to 20% against 12%).
- **Layer 2, variety without nonsense** (16 options: 2 obvious, 3 unusual, 11
  useless; the runner keeps every score). The report reads the draw at equal
  nonsense (10/20/30%), so the result does not depend on how a model scales
  its scores. No model reaches 10% nonsense even when greedy (best: 17 to
  20%). At 30% nonsense, openjev 2B gives 2.6 different workable things in
  10 draws (of 5) and 25% unusual draws; the others give 2.0 to 2.4 and 14
  to 20%. A uniform draw is 19% unusual, so models favor the obvious tool,
  and more temperature buys mostly nonsense.
- **Layer 3, scenes** (`lab/judge/signals.py`, tests in
  `lab/judge/test_signals.py`). It reads a world's event log: surprise
  against an "average character" (after "Select the Unexpected"), split into
  choices the character's own personality explains and choices it does not;
  story patterns (revenge, betrayal, rescue, lie found out, coward acts
  bravely, gift returned) ranked by rarity across runs; reaction chains;
  variety between seeds per character type. The docstring fixes the event
  log format the lab world must write. It cannot run until the lab world
  (Workstream E) exists.


## 102. The lab world, scene creativity, and why the models pick wrong (2026-09-28)

**Lab world v2** (`lab/world2/`, plan Workstream E, built with the plan's
proposed defaults while the action catalog waits for notes). A village inn
at night: four rooms, five people (peasant, knight, merchant, a thief the
others take for a traveller, cook), and a bandit who comes in from the road
in about 60% of the nights. Traits have a mean per type and vary at each
choice; anger, fear and drink are momentum that fades; actions take turns
and a blow, threat, shout or seen theft stops them; each belief keeps the
event it came from; a lie is judged by what the teller believes; the old
chest's contents are made when first opened. The world check (1,000 random
turns, beliefs traceable, one chest) passes (`lab/world2/test_world.py`).
Tuning found and fixed: anger with no ceiling (two fighters missed each
other for ever), a gift that did not change owner (the new owner was
treated as a thief), words that counted as a physical threat (a knight
killed a cook who accused him), repeated accusations with no habituation,
and lies found out never (a new sight replaced the old belief before the
check).

**Four ways to choose** (`lab/world2/policy.py`): random; rules (utility AI,
no model); a model with the personality as a sentence; and a model with a
trait bias ADDED to each option's score (the user's idea: the model says
what makes sense, the traits weight the options; the weights could be
learned later). Habituation is a separate term that every policy adds.

**Layer 3 on the no-model policies** (20 nights each). Random: 0% surprising
choices that fit the character, 60% odd ones, and story patterns everywhere
(22 "revenges" a night): noise looks like drama until "does it make sense"
is asked. Rules: 3.6% surprising and in character (the bandit robbing, the
thief lying to shift blame, the timid merchant fleeing), 5.6% odd.
"In character" had to be tightened to "the character itself gives it at
least 5%, and twice the average": a 1-in-1000 draw is not in character even
when the traits made it 27 times likelier. Three measures had to change to
separate the two policies: patterns are counted per night, a reaction must
be aimed at the people of the event seen, and variety compares what each
type did night to night. The rules miss the plan's type ranges (knight
brave 19%, not over 70%): they have no sense of duty.

**Why the models pick wrong** (`lab/methods/failures.py`, every wrong pick in
Workstream A against a random wrong option from the same list). The main
failure: models pick what the need is ABOUT, not what does the job: the rope
to cut the rope, the apple to reach the apple, the copper coins to hide the
stolen coins, the lamp oil to pour oil into the lamp, the nails to fix a
nail, the padlock for a locked chest. By meaning, wrong picks sit near the
need's object at about 2x chance (35-47% of wrong picks) and near its verb at
about 1.7x (tools that sound right and are not). Every model also has a
prior: favourite items it picks whatever the need, at 3-7x chance (GLiNER-
Decide: firewood, the wooden box, the padlock; Laya: weapons; openjev 2B:
containers). Words in unrelated scene text pull GLiNER-Decide, bev-decider,
Von and bge at 3-4x. Position bias is only in the listwise decoders (Qwen
numbered list: first 3.7x) and Laya's own list (first 2.0x, last 1.8x). The
cases every model fails also show generous labels (a rolling pin "good" for
breaking pond ice; a frying pan for a nail), for the audit.

**The ablation that tests the causes** (`lab/methods/ablate.py`, queued): the
same lists with the verb only, the object only, no need at all, and each
option written as the whole action ("cut the rope with the rope" reads as
absurd). From the kept scores it also works out contextual calibration
(the full score minus the no-need score).

**People and places** (plan section 4, `lab/methods/cases_kinds.py`): 1,140
cases each. "People" aims actions at the people present, so the same action
at the wrong person ("hit the child") is the entity-binding test. Queued on
the main models.

**Short list, then choose** (`lab/methods/pipeline.py`, done). GLiNER-Decide
keeps 8, Qwen3-1.7B two-at-a-time chooses: 60/55/52% at 32/64/128 options,
against 53/53/40% for the best single model. The gain is on long lists (+12
at 128, at the edge of the noise), at a fraction of Qwen's time there.
Keeping 8 beats keeping 16; Qwen is the better finalist than openjev 2B.

**Ablation, first two models** (GLiNER-Decide, bev-decider): the verb alone
does as well as the whole need (60-63% at 16 options), so the object is not
what pulls them wrong; the gap is knowing what things can do. Removing the
prior gains little (+7 / -2). Writing each option as the whole action hurts
(33-37%).

**The presentation battery** (`lab/methods/present.py`, queued). Until now
every test gave a prose situation, a question and short option names. This
changes one thing at a time: how the need is given (in the question,
key-value, JSON, second person, an order, a capability hint), how options are
given (description, the world's tags, kind, numbers), and how query and
options combine (Qwen sentence completion, openjev hypothesis wordings, kind
first then thing). The word-overlap baseline is at chance on every format
except the tags (55-57%), which leak by words.

Paused on 2026-09-28 at night; `lab/methods/resume_all.sh` restarts every
queue where it stopped.

**Next, after the battery:** `docs/presentation-plan.md` notes the ways and
best guesses for (1) how to show people and places (cards, relations, "whom
first, then what"), (2) the inn's scene description (made testable with
labelled decision snapshots; prose against cards, events as lines, a
filter), and (3) the non-text track (entity vectors and a slot head, with
factorized scoring as the zero-shot step we can run now).

**Personality and interpretability, reviewed (2026-09-29).** Two reviews:
`docs/persona-steering-review.md` (94 sources) and `docs/mech-interp-review.md`
(111 sources; its TransformerLens claim was corrected: no DeBERTa or
ModernBERT-encoder adapter). The combined design is `docs/personality-design.md`:
final score = sense (the chooser, no persona) + γ · learned trait weights ·
option features + habit, with a plausibility guard and γ drawn per scene. It
costs no model call for personality, so it fits the speed rule. Features are
to come from cached NLI trait entailment instead of hand tags; weights from
large-model labels, then from the user's ratings. Steering inside the model
(guidance on option scores, persona-style vectors) is tried on the Qwen
decoders only, after a sign check. Laya's collapse with list length is not a
cut-off artifact: nothing is cut up to 32 options, where it is already at 18%.
The shared first step for both the inn formats and the personality measures
is a bank of about 150 labelled decision snapshots.

**Findings at the second pause (2026-09-29, 12:34).** The ablation finished for
all 8 models; the presentation battery for 7 (not onli2b, qwen17:two, the
hier runs for bev and Laya); people for the 8 fast models; places for
GLiNER-Decide only.
- GLiNER2.5-Decide (the chooser we settle on), 4 to 128 options: things
  78->28%, places 80->42%, people 70->7%. Long scene text: people 40->27%.
  The verb alone does as well as the whole need; removing its prior does
  nothing. The capability hint gives +15/+13 (16/64 options); key-value
  lines +6 at 16. Everything added to the options hurts it (descriptions
  -14, kinds -14, numbers -22/-27); tags are no better than word matching.
  Choosing the kind first, then the thing, collapses it (28%/27%).
- Models use different parts of the need: GLiNER-Decide and bev-decider the
  verb; Von and Qwen two-at-a-time the verb and the object; Laya mostly its
  prior. Removing the prior helps openjev 0.8B (+10), Qwen3-1.7B (+8/+10)
  and Laya (+11), hurts openjev 2B (-12/-20). Writing options as whole
  actions hurts the listwise models (GLiNER 62->37) but not Qwen
  two-at-a-time (72->68). Qwen is very sensitive to wording: as an order,
  one_cal falls from 50% to 3%.


## 103. Plan v4: small questions first, capture, priors, learnability (2026-09-29)

The user's notes on the findings (examples of every task, preliminary
questions to prepare the evidence, an entropy correction for favourite
items, scene focus, position bias, and whether near misses can be learned)
became plan v4. Built, with the queue written and not started
(`lab/methods/v4_queue.sh`; to-do list in `docs/mind-lab-todo.md`):

- `lab/methods/examples.py` -> `lab/results/methods/examples.md`: the full
  input of two cases per task space, with what each model receives.
- `--capture` (`run.py`, `score.py`): GLiNER-Decide saves each option's
  vector (the row its classifier scores) and the text's vector, through
  forward hooks. Checked on the CPU: the saved vectors give back the saved
  logits (difference 0.0004) and scores.
- `lab/methods/stages.py`: preliminary questions and how their answers feed
  forward (a phrase, a filter or a soft weight; never a hard gate). Each
  experiment has the arms none, oracle, model and wrong: capability (38
  phrases), people (7 roles, 9 "calls for"), personality (feeling and urge
  against persona text; the measure is the confront shift between a
  hot-tempered and a timid Tom), relevance, "what is still missing",
  soft kind-first, a speech demo.
- `lab/methods/prior.py`: a fair-die probe, and three corrections (the
  item's mean, an even spread fitted by iterative scaling, the item's mean
  vector removed before the classifier), fitted on half the needs, with a
  check that character survives. Self-test: a planted 4x favourite lift
  falls to 0.9, accuracy goes from 53% to 87%, and a trait shift survives
  (+0.20 before, +0.23 after).
- `lab/methods/cases_state.py`: 6 multi-step needs with the state written out
  (the firewood is right when the hearth is empty); the other step's things
  are the key distractors.
- `lab/methods/learnable.py`: probes on the captured vectors (random / new
  items / new needs, with a shuffled control), a head trained in place of the
  classifier on needs it never saw, and one head against one per kind of
  question.
- `lab/methods/finetune_gliner.py`: GLiNER2's own trainer with LoRA, data
  built with 8 needs and 15 things held out; for the user's machine.
  `LAB_GLINER_ADAPTER` lets the runner score with a trained adapter.
- Not relevant to the character models: CuACD (convex pieces of meshes).
  Noted for the simulation, with Taichi for GPU MPM.

**Learnability review** (`docs/learnability-review.md`, 30 sources). Small
models do learn object uses when trained (RoBERTa-large 77% on PIQA; about
400 targeted counterexamples lifted BERT-base from 0.28 to 0.73). But our
exact failure is published: related objects are told apart from unrelated
ones, usable from unusable ones are not. The prediction: probes and
fine-tunes work on seen needs and new things; near misses on NEW needs are
the real test, and a clean positive there would be new. Its controls are now
in `learnable.py` (answer only, memorise, a second source of near misses, a
curve over the number of needs), and `nearmiss_audit.md` asks the user to
rate 60 hand-picked near misses before training.


## 104. Plan v4 results (queue run 2026-09-29, 22:34-23:28, no errors)

**The knowledge is in GLiNER-Decide's vectors; the readout is the problem.**
Probes on the captured option vectors, on NEEDS NEVER SEEN in training:
acceptable against bad AUC 0.94, against hand-picked near misses 0.86,
against near misses from a second source 0.91, unusual-but-workable against
bad 0.90. Controls: the thing alone (no need) gets 0.44-0.74, bge on
"need: thing" 0.56-0.58, shuffled labels 0.5; memorising gets 0.87 on the
random split (so only the new-needs numbers count). Training on 10 needs
already gives 0.90. But a per-option head used to CHOOSE only moves the top
pick from 61% to 66% on new needs: the next step is a head that compares
the options of a list, or the LoRA.

**Favourite items are removed at no cost, and character stays.** Things:
the favourite lift falls from 2.96 to 1.22 (even spread; 62%, against 63%)
or 1.84 (item mean; 65%). Places: 3.18 to 0.98 at +2 points. The
hot-against-timid confront shift is kept (+0.28 to +0.26). The fair-die
probe: GLiNER-Decide spreads "pick a thing at random" almost evenly (0.96);
Von 1.3 puts 100% on "6", Laya 100% on "20" of a d20, Qwen 79% on the
kitchen knife.

**Personality carried by preliminary questions works.** Confront share,
hot-tempered minus timid: persona text +0.17, the model's own feeling and
urge +0.28, the right feeling and urge +0.30. The model's appraisal is
accurate (hot: angry, stand up to them, 45 of 45). The overlap baseline
shows no shift, so it is not word matching. Cost: valid picks fall from 76%
to about 50%; the plausibility guard of the personality design is needed.

**Preliminary questions for sense mostly do not pay yet: first answers are
not reliable enough, and a wrong one costs a lot.**
- Capability: GLiNER picks the right one 60% (top two 77%); a wrong hint
  drops the choice to 27%, so the model arm lands at 58% against 62%
  (break-even about 73%; the right hint gives 75%). A head on new needs does
  not help (55%). The capability must come from data: the verb inventory.
- People: roles are right only 22%; "calls for" 60% (a head: 85% on new
  needs). The right evidence helps with unrelated distractors (50% to 73%)
  but hurts with wrong-person lists (40% to 22%): naming everyone adds word
  cues.
- Relevance: 97% of sentences judged right, but the misses fall on the
  sentence that states the need (dropped in 28 of 120 lists), so the choice
  falls from 42% to 36%. Fix: never drop the need.
- Scene state: GLiNER picks the thing the state MENTIONS. "Dry wood is laid
  in the hearth" gives the firewood 100%; "Kael has a quill and ink" gives
  the quill and ink 100%. The mention lure at its plainest. The right
  "still needs ..." hint recovers step A from 39% to 67%.
- Kind first as a soft bonus, and a "will that work?" re-check of the top
  five: no gain. The speech demo works but its facts (split clauses) are
  crude.

## 105. CPU follow-ups on the saved v4 data, and a battery of question kinds (2026-09-30)

All on the CPU, from what plan v4 saved; no new model runs. Reports:
`lab/results/methods/learnable.md` (A1, A6), `combine.md` (A2, A3, A5),
`prior_offsets.json` (A4).

**A head that sees the whole list does not unlock the choice.** Trained on
the frozen option vectors, on needs it never saw: 67% right, against 61% for
GLiNER's own choice and 66% for a per-option head. On lists with a near miss:
57% against 51%. The control (each thing's average vector, no need) gets 9%,
so the head learned fit, not "useful things". The vectors separate the
options well (AUC 0.86-0.94), but a new head alone adds only 6 points. The
next step for choosing is the fine-tune (the user runs it elsewhere).

**The stage heads, re-checked with needs held out:** capability AUC 0.92,
top pick 55% (GLiNER 58%); "calls for" AUC 0.86, top pick 75% (GLiNER 61%).
The earlier 0.99 was memorisation.

**The per-person role question (22%) could not be answered.** For 125 of the
175 people it asked about, the person is not in the scene text at all (they
are only in the options). Named people: 56% right; unnamed: 9%.

**"Calls for" as a weight works; as a sentence it did not.** Bonus = l x log
p(the option's call). With the head's call: near-miss lists 28% to 50%,
unrelated distractors 50% to 60%, 8 options 60% to 67%, 16 options 38% to
40%. With the right call: 62%, 85%, 75%, 67%. It cannot help on the
wrong-person lists (40% to 25% with the head): a call does not say WHOM.

**Relevance: the model ranks the sentences perfectly; only the threshold was
wrong.** AUC 1.00 (sentences that matter against filler). At p(yes) of 0.3
or more, every sentence that matters is kept in every scene, and 67% of the
filler is dropped; at 0.4, 95% and 100%. The top-label rule (0.5) dropped the
need sentence in 23% of scenes. Caution: the filler is a fixed, plain list.

**The personality guard gives a trade-off, not both.** No setting keeps
valid picks at 70% or more with a shift of +0.20 or more. The appraisal with
a hard guard (g 2, d 0.5) gives +0.16 at about 70% valid (text: +0.17 at
58-71%); g 2, d 0.3 gives +0.38 at 51-60%. New finding: with no persona the
DRAW puts only 43% of its probability on valid options (the pick is valid
76%). A hard guard raises the draw to 67%. The inn samples, so this matters.
Only 45 lists from 15 scenes: more scenes are needed.

**The prior correction is a chooser flag.** `LAB_PRIOR=
lab/results/methods/prior_offsets.json` (7,918 items from 3,540 lists, item
mean, alpha 1). Checked: the flag gives the same numbers as prior.md.

**A battery of question kinds** (`lab/methods/primitives.py`, 2,307
questions, 12,172 cases with variants; not run yet). The user asked for
the kinds of question a person asks themselves, grouped, to find the ones
the model answers reliably and build chains only from those. 17 groups with
known answers (from the inn world's state, the item data, the scene roles,
false-belief stories, the state cases) and 4 with unknown answers (feeling,
urge, coping, approach). Each question is asked in 3 wordings, in reverse
order, with filler, and (for names, things, places) with a lure sentence.
Unknown answers are judged by direction (a trait sentence must raise the
expected answer), by stability, and by a trait x scene grid with an empty
scene (a "state-blind" model follows the trait and ignores the scene). A
literature review with 83 sources supports the grouping:
`docs/question-primitives-review.md`.

## 106. GPU follow-ups: the question battery, the mention lure, and the inn (2026-09-30, 21:57-23:34, no errors)

Reports: `lab/results/methods/primitives.md`, `lure.md`, `stages.md`
(capability2, relevance2, personality2), `lab/results/world2b/compare.md`.

**GLiNER-Decide mostly matches what the scene NAMES against the options.**
- One sentence that names a useless thing ("There is the bucket by the
  door.") makes it pick that thing in 100% of lists; right picks fall from
  69% to 0%.
- In the battery, "Someone mentions X" breaks every kind of question:
  "Where is Tom?" 100% to 33%, presence 75% to 21%, false belief 68% to 8%.
- "Who is in danger or hurt?" and "Whose fault is this?" answer "Tom" (the
  last person named: "Tom is there.") in 24 of 29 and 17 of 20.

**Which kinds of question it can answer.** Reading what is stated: place
100%, goal 100%. Norms (yes/no) 85%, possession 79%, presence 75%. Most other
kinds 55-70%; inference is weak: danger 39%, inner state 56%, roles as
entities 10%, fault 0%. Yes/no leans to "yes" (right when the answer is "no":
14-50%). False belief fails (where will she look, when she was away: 10%).
Answers are stable across wordings and orders (mostly 85-100%): consistent,
but often wrong. Being "sure" does not rescue the weak kinds.

**State-blind, as the review warned.** With no trait sentence the feeling
follows the scene (5 of 5: glad at a gift, sad at a death). With any trait
sentence the trait word wins in every scene (hot: angry at a gift; timid:
afraid everywhere). So the earlier 45 of 45 was the trait word read back.

**Writing the input as data works, and matches the oracle.**
- The state as only what is missing: step A 39% to 82%, step B 31% to 55%.
- The capability from a verb table: 62% to 75% (16 options), 47% to 60%
  (64), equal to the right hint. (The table was written knowing the needs.)
- Relevance kept at p(yes) 0.35: 42% to 48% (the perfect filter: 47%).

**Personality on 28 scenes.** Temper: the model's feeling and urge +0.24,
the trait sentence +0.14 (repeats v4). Kindness: +0.20 both. Valid picks still
fall. Given state-blindness, the gain is the urge's words matching option
words, not a judgement of the scene.

**The inn.** The take/drop fix cuts the trait-bias policy's reversals by 76%
(68 to 16 a night). The new appraisal policy collapses: 2,675 accusations in
10 nights (each of Oren, Sir Ada and Fenn accuses Tom about 42 times a night)
and the take/drop loop comes back (154 a night). Its 56% "surprising, in
character" is an artifact of a sharp draw, not creativity. The hard guard
(d 0.5) leaves 1-3 options and g 2 sharpens them, so the habit term cannot
break a loop. The trait-bias policy stays the best model policy: 12.8%
surprising and in character against 9.9% odd; revenge 14.3, rescue 27.4 a
night.

**What this means.** The model should not be asked to infer or appraise.
The engine knows the facts and the traits; it should write a small, clean
input (what is missing, the capability, no named distractors) and carry
personality as data. The model's job is the last pick. The fine-tune should
train against the mention lure (named but useless things as hard negatives)
and the yes bias.

## 107. Todo: the realism gaps, slow processes, and mechanics (2026-10-02)

Three lists the user agreed to. They go into the waves of
`docs/sim-mind-design.md` after events and perception, because each one must
write named facts and events, and the model reads only those.

### A. The ten biggest realism gaps

From `docs/realism-references.md` section 3. Each fix lands with its
`tests/test_realism_*.py` file (section 4 there), so it is checked against the
real range.

1. **G1 Hearing.** Speech is not understood past 2–3 m or through any wall:
   `_hears` counts the speakers' own heads as a 40–46 dB wall (the roadmap 97
   bug). A doorway costs 56 dB, not 10–17. No sound around corners.
2. **G2 Reaction.** Everyone reacts on the same tick and in the same way. Real
   reaction is about 0.2 s, and people often take minutes to leave a fire.
   Fire is noticed only within 4 m. The view cone is 120°, not 200–220°. A
   person who flees never looks around again.
3. **G3 Walking.** 0.67 m/s, not 1.1–1.65. No running. Speed depends on the
   voxel size.
4. **G4 Fire clock.** A 5 cm wood voxel burns in about 4 s; oil burns 100–500×
   too fast; a room reaches megawatts in 22 s. Each law is off by a different
   factor, so one rescale does not fix it.
5. **G5 Cooling.** Hot things go cold in seconds (`LEAK` 6.25 s; k about 400×
   too high). A glowing bar goes dark in under 3 s; the forge stalls at 258 °C.
6. **G6 Crash.** Toppling tin, char or ash crashes the particle solver (MAT
   rows have 6 fields, 7 are unpacked). Also in X1.
7. **G7 Radiant heat and smoke layers.** No heat across air, so nobody feels a
   bonfire and fire never jumps to the next house. Smoke does not layer, so
   crawling low does not help.
8. **G8 Acid.** 10³–10⁷× too fast. Vitriol dissolves living people; vinegar
   dissolves wood.
9. **G9 Breathing clock.** Nobody drowns (water counts as air). A buried face
   dies in 1.7 s. A smoke victim wakes 5–12 s after rescue.
10. **G10 Body numbers.** Throws 2–3× too slow, jumps 1.6–1.9× too high, arms
    2.6× too heavy, no knockout.

### B. Slow and cheap processes

**Three rules make them cheap:**
- Each process has its own clock. Fast laws run every tick; slow laws run
  every N ticks with a larger step.
- Compute only where something happens: active windows (as heat does now),
  touching faces for chemistry, scheduled events for slow changes.
- Skip ahead when it is quiet. When no fast law is active, move the slow
  clocks forward with the exact formula (an exponential decay has one), so a
  night of sleep costs a few steps.

**The processes:**

| Process | Where it runs | Model | Cost |
|---|---|---|---|
| Wind and gusts | the 20 cm air cells | an outdoor edge with a mean wind plus random gusts. The wind bends flames, pushes light things and people (drag), cools hot things faster, and spreads fire downwind | almost none; the air field exists |
| Gas species (CO, CO₂, steam, H₂, a poison gas) | the air cells | a few amounts per cell, carried by the air field. Replaces soot as the stand-in for CO | about 1/64 of a voxel field each |
| Contact chemistry (acid, rust, lime, mixing) | touching faces | one general reaction table: reactants, products, a rate that depends on temperature, the heat. Burning becomes one row of it | only where two reactants touch |
| Body clocks (tiredness, hunger, thirst, sleep, poison, infection, healing) | inside each person | a few numbers per person, updated each second or minute, with standard models (for example a one-compartment drug model) | a few numbers per person |
| Slow world (rot, fermenting, plant growth) | each object | a timer whose speed depends on heat and wetness (biological rates rise about 2–3× per +10 °C). Schedule the moment of change, not each tick | one scheduled event per object |

**What each one gives the decision models:**
- Wind: "a gust", "the candle went out".
- Gases: a new sense, smell ("the cellar smells of rotten eggs").
- Body clocks: true inner states ("Tom is tired and thirsty"). These give real
  answers to the lab's "how does X feel" and "what would X try" questions.
- New acts: shelter, open a window, eat, sleep, bandage.

**Order:**
1. A research pass first, as for `docs/realism-references.md`: wind, gas
   chemistry, slow biology, and the mechanics in list C.
2. Wind and gas species. They reuse the air field and also close G7 and G9.
3. The general reaction table. It takes in the acid fixes (G8).
4. Body clocks.
5. The slow world.

### C. Mechanics: direction, turning, give, and water on bodies

**The larger gap.** The sim keeps mass and energy well. It does not keep
momentum or turning (angular momentum) across kinds of matter. Things must
push on each other, and the push has a direction, a turning part and a give.
Each pair needs a force exchange both ways: solid on solid, solid on water,
solid on air, a person on any of them.

**What exists now.** Matter moves in three ways, and each way has part of the
physics:
- **Lattice** (`support.py`): columns fall straight down with air drag;
  impacts pay energy against toughness. No direction, no spin, no give.
  Tipping (`_law_torque`) is an instant quarter-turn about one edge.
- **Flying bodies** (`bodies.py`): a velocity, gravity and drag. No spin
  (`_fly_pose` only adds an offset). Rigid, no give.
- **Particle solver** (`src/mpm`): real momentum, turning, bending, denting,
  breaking and friction; shape matching keeps both momenta. Toppled clusters
  that no person owns go here. Free fall into it is off
  (`free_fall_to_seam = False`). It has no water and no people.
- **Water** (`fluids.py`): volumes in cells. It finds its level and layers by
  density. It has no momentum, so a moving thing feels no water force (except
  the cushion when it lands) and makes no waves.

**Why the tree's leaves fall in bulk (likely cause, not yet confirmed by a
run).** An object is a connected set of cells of the SAME material
(`_law_torque`). When the trunk tips, the wood goes to the particle solver.
The leaves are a different material, so they stay on the lattice. They lose
their support and drop straight down column by column, with the drag of one
cell face per column: a brick of leaves. A real crown is mostly air. Its
leaves hang on twigs that bend, and each leaf has a large area for its weight,
so it flutters and falls slowly.

**Why a stone cannot skip.** A skip needs the stone's tilt, its spin (which
keeps the tilt steady), and a water force that grows with the speed squared
on the wetted face. The best tilt is about 20° (Clanet, Hersen and Bocquet,
Nature 2004). The sim has none of the three.

**The plan: each object at the cheapest level that keeps its facts.**

| Level | For | State |
|---|---|---|
| Voxels | walls, floor, ground (never disturbed) | as now |
| Rigid object | stones, planks, barrels, weapons, boulders | one position, orientation, velocity, spin and inertia per object; contacts with friction and bounce |
| Bonded object | trees (trunk, branches, twigs, leaf clusters), ropes, chains, furniture, people (the skeleton is already a bone tree) | rigid pieces joined by bonds that bend and break |
| Small light pieces | leaves, sparks, ash, droplets | particles with high drag; they come off a bond when it breaks |
| Continuum (MPM) | what really flows or deforms: sagging metal, shattering glass, mud, clay | as now, after X1 |
| Water | pools, rivers | cell volumes, plus a surface height field for waves, plus forces on what is in it: buoyancy, drag, lift. The water takes the momentum back as waves and splash |

- **Cost.** Rigid and bonded objects cost per object, not per cell. A tree of
  a few hundred bonded pieces is small. MPM stays the expensive level, used
  only where it is needed.
- **Check.** Momentum and angular momentum totals across all levels, as
  `ledgers.py` does for mass and energy.
- **Decision.** This is the bonds decision of roadmap 93 (item 2), which was
  deferred. The tree and the stone say it is needed.

**Order:**
1. Rigid objects with full turning and contacts (stone, plank, boulder).
2. Bonds (tree, rope, chain), then people as bonded bodies.
3. Water forces on bodies (float, sink, skip), then waves.

**Reference tests to write (numbers from the research pass):**
- A flat stone thrown fast at a low tilt bounces on water more than once; a
  round stone does not.
- A felled tree's leaves fall slower than its trunk and spread out.
- A thrown axe turns in flight.
- A boulder on a slope rolls, and does not slide.
- A plank floats, and a tipped plank turns back flat.
- A crown hit by wind sways and comes back.

## 108. The sim-mind build, phase 0 to wave 5 (2026-10-01 to 2026-10-03)

The build of `docs/sim-mind-design.md`, with no commits: each wave is checked
against a snapshot of the tree before it (golden state hashing over 35 cases,
sandboxed step checks, the full suite, the harvest guard). Paused after wave 5
at the user's request.

| Stage | What landed |
|---|---|
| Phase 0 | `sim.py` (8,794 lines) split into about 20 mixin modules, behaviour identical. `tools/golden.py`, `tools/stepcheck.sh`. `src/mind` (catalog, perceive, beliefs, menu, describe, loop, choosers, record/replay) and `src/rooms` (the inn as a backend). |
| Wave 1 | Cleanup; snapshot ghost fix; fast/slow test tiers (`tests/conftest.py`); MPM numba solver (400k grid 143 → 2 ms per substep; tin/char/ash crash fixed); vessel pins; body fuzzer (`tools/fuzz_bodies.py`); profile. |
| Wave 2 | Six entity defects; the event core (`events.py`); particle eid, fill and flaws (a 0.8 safety factor on the viscous step stops fake trunk cracks). |
| Wave 3 | One labeller (`label.py`) and an eid on all matter; events in will, senses and physiology; the heat seam; air regions on the labeller. House tick 397 → 98 ms. |
| Wave 4 | Thing registry and names ("the haft of the axe"); events for hits, landings, wounds, breaks and topples with actor and cause; particle collision events. |
| Wave 5 | Holding by eid (a two-material axe is one thing); perception (heard, saw, did, felt, looked records, each tied to its cause event); R1, acid from measured rates (gap G8). |

Tests went from 394 to 666 (plus 3 strict expected failures), and the full
suite from 21 to 10 minutes.

**Found on the way, for later steps:**
- Bodies often keep a detached piece of flesh (103 of 150 fuzz worlds), and two
  people share cells in 43 of 150 (bug 87). One no-burn world lost 28.9 kg. → E6.
- A felled tree's crown separates from the trunk as its own block and never
  sheds a leaf: leaf is a different material from wood and has no tensile
  strength. → 107 list C (bonds, leaves as light pieces).
- G1 confirmed by a test: an alarm is heard at 1 m, not at 1.5 m. Sight rays
  start inside the viewer's own head, so close fires and passing runners are
  missed. → R2 (proposed with wave 6).

**Next:** wave 6 is M7 (the attention law), V4 (beliefs, sightings of things,
misreads and claims, off by default), E4 (properties, reach, adopt-on-take,
torque), with R2 (hearing and sight) alongside. Wave 8 links a chooser to
voxel bodies through `src/mind` for the first time.
