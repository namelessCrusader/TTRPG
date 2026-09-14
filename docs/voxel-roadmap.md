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
