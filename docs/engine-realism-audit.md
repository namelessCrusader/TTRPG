# Engine realism audit: early choices we can do differently or better

Date 2026-11-22. Status: a plan to read, not a build. Nothing was run, edited or committed (the pause is respected).
Owner's question: "let's think of ALL such early engine choices we can do differently or better for realism", asked right after deciding that rigid shape-keeping was a poor early choice.

Basis: five lane audits in `docs/engine-audit/` (A matter, B heat/fluid/light/sound, C bodies, D time/scale/numerics, E the switch ledger and the (u) constants), about 100 items in all, each with file:line, why it was made, the realism it costs, alternatives with sources, cost, status and priority. This report merges them, removes the overlaps, ranks them, and places the new ones in the build order. The lane files hold the full text and the citations; this file does not repeat them.

Marks used in the lanes and here: (v) checked by reading the repo or a source this session, (k) from knowledge and not re-checked, (u) a guess or an estimate. Almost every literature citation is (k): fetch it through Babbel before building (the request `2026-11-22-mark1-deformation-and-scale.md` is already out; light, cardiovascular and pharmacokinetic sources need a second one). Every cost figure is (u) until measured.

---

## 1. The short answer

The engine is better than its early choices. The conservation laws, the creep clock, edge wear, the bone beam check, vessel bleeding, Pennes heat, the Purser dose, the twin protocol and the ledger are sound. What went wrong early falls into a small number of root causes, and most of the 100 items are symptoms of them. The same lesson as the rigid voxels applies again and again: **a cheap stand-in was put where a law or a field belongs, and it was never a flagged, owned task to remove it.**

Ten root causes, in order of how much they cost (section 2 has the symptoms of each):

1. **Stiffness was paid for in time-steps** (softened moduli, authored `rig`, bond budget). Decided in part; the integrator is the open half.
2. **The grid is the only geometry** (a cell is a mass fraction with no normal and no offset). One root behind thin plates, thin films, rope radius, shards, finger-sized things, curved-surface area.
3. **Laws that were built but are switched off, with stand-ins doing their job** (gas flow and momentum, the eddy term, the capped rise). The single largest fire-scene error is here and no batch owns the flip.
4. **One number where a field belongs**: light, room radiation, the sun, convection, smoke, sound, ambient noise, weather.
5. **A rigid stand-in for a compliant person** (11 blocks, a hop for a gait, a broken leg that still bears weight).
6. **Physiology closed by one scalar** (no heart, no pharmacology, death as a threshold).
7. **Rates that depend on the clock and the cell size**, with nothing that checks it.
8. **Conservation by accounting, not by construction** (float32 energy, eight crutches that break "every joule booked").
9. **Three solvers, three sets of material numbers** (friction 0.40, stone weaker than wood, SPAN).
10. **Analysis replaced by a veto** (structures, collapse, granular matter) and **a world that is one dense box** (no village, no distance).

The ten best fixes by impact over cost (full list in section 3):

1. Switch the buoyant gas flow and momentum on in every world and delete its stand-ins (F1).
2. A convergence harness, fact goldens and one cell-size and tick constant, tests only, built first (F2).
3. One energy-hole sweep: a single "matter change" gate and a lint (F3).
4. One material record read by lattice, bonds and particles, with pair friction and restitution (F4).
5. The decided rod skeletons and elastic shape-keeping, with an implicit or position-based integrator and no softening (F5).
6. A light field with radiant exchange and the sun as heat, and a contrast law for sight (F8).
7. A lumped heart, pressure and brain-perfusion model, with death as a process (F9).
8. One pharmacokinetic law for alcohol, poison, medicine and potion (F10).
9. Shape for things below a cell: thing records, a shape class, a staircase area correction (F11) and a real convection law (F12).
10. Gait from the person, then an articulated skeleton and an adult default body (F13, F17, F18).

Eight of the choices here are real trade-offs and are in `~/.cache/mark1-build/engine-audit-questions.txt`. Everything else I decided myself and list in section 7 so you can veto it.

---

## 2. Root causes: one fix, many symptoms

| Root | What was chosen early | Symptoms (lane ids) | The one fix | Scheduled? |
|---|---|---|---|---|
| R1 Stiffness in time-steps | MPM moduli x1e-4, strength x sqrt, `rig` per material, bond budget 2e5 piece-substeps, symplectic Euler | A3, A4, A5, A6, A21, A13 (restitution fitted by ZETA_C), D12, D5 (cost shapes fidelity), B3(b) | Rod primitive plus elastic shape-matching with stored and booked energy; XPBD or implicit integrator; integrator damping booked as heat; no softening on stiff solids | Model decided 2026-11-22. Integrator, ledger rows for the softening and budget, bond islands (partly, wave 15 list): NEW |
| R2 Grid is the only geometry | Cell = material id + mass; fill is read as a thickness, an area, a hinge | A1, A2, A10, A20, A22, B1, B4, B5(length), B26(area), C2, C7, C19(extremities), D11 | Thing-level records (thickness, width, normal, shape class) as the source of section and area; oriented partial cell (normal + offset, 3 bytes) for ground, walls and floors | Thin-plate heat area: owner YES, wave 15 batch 3. Everything else: NEW |
| R3 Built but off | `GAS_FLOW` and `GAS_MOMENTUM` False by default; legacy rise, ceiling jet constant, eddy conductivity | B6, B7, B8(f), D2, E2.1, E (`gas.eddy_heat`, `gas.drag_mom`, `cold.plume_k`, `cold.legacy_pressure_mean`, `body.p_follow`), B25 | Flip on, delete the stand-ins in one wave, book the flow's kinetic energy | W15-Q8 names the goal; no batch owns the flip: NEW as an owned task |
| R4 One number, not a field | One lux; room = one radiant node; sun heats nothing; h = 10; one soot scalar; one sound band; one ambient dB | B2, B5, B13, B18, B21, B22, B23, B24, C22, C23, C25, D20 | A shared patch set (longwave and 4 shortwave bands) carrying radiation and light; a species bundle for gas; sound bands and an ambient from sources | 14a R1 only reads the sun lux. All else NEW |
| R5 Rigid stand-in for a compliant person | 11 voxel blocks; two-speed hop; step-or-grab reflex; fractures written, never read | C1, C2, C4, C6, C7, C8, C9, C10, C11, E2.5 | One articulated skeleton (same rod and joint primitive as trees), voxels drawn from it; gait from the person | Not scheduled. Must be decided before 16a |
| R6 Physiology by one scalar | Blood pressure linear in volume, `o2` with fixed faint and death lines, `drunk` set by nothing | C12, C13, C17, C18, C20, C19(c,d) | Two small primitives: a lumped cardiovascular model and one pharmacokinetic law | Not scheduled |
| R7 Clock and cell-size dependence | Per-tick and per-voxel constants; fixed law order; strides `tick % N` | D1, D2, D3, D4, D16, D18, B3, B19 | A convergence harness (dt/2, dx/2) plus unit tags and one `VOX_M`, then multi-rate scheduling | Harness machinery exists (twinharness); use for dt and dx: NEW |
| R8 Conservation by accounting | float32 E, quant/f32/recast books, eight energy-hole crutches | B3, B8(f), D6, D7, D8, E2.2, E2.6 | One `matter_change` gate plus a lint now; float64 conserved fields; integer quanta only if audit shows booked rounding hides real errors | Not scheduled |
| R9 Material numbers disagree | Friction 0.40 in two solvers, MPM wood 10 MPa above stone 8 MPa, SPAN floor, WET_FACTOR 0.7 | A7, A13, E2.3, E(`mat.*`), E(u) rank 1 | Everything reads `matrec`; the legacy overrides deleted one declared change at a time | "D5 one by one", no wave named |
| R10 Veto, not analysis; one dense box | Static vertical beam veto; collapse = cells falling; no granular law; world = one 6 m cube | A11, A12, A15, A16, A17, A18, D10, D14, D21, D22 | Load cases and a member graph on change; slump rule then collapse to bonds; sparse chunked world with a detail ladder | Soil and digging "later" (W17), world at scale "later": positions NEW |

Two further small roots: **fast movers and blows with a point** (D19 swept tests, A23 penetration, A9 one impact law, C15 head rotation: one event law plus one DDA traversal, which also fixes the sampled sight rays C24), and **default bodies fudged to a stick figure** (C3, C5, C16, C27: one generator `sheet(age, sex, genome)`).

---

## 3. One ranked list of fixes (impact over cost)

Rank mixes realism gain, cost and real-time risk. Dependencies are in section 5. "Owner" means it appears in the question file. Cost marks are (u).

| # | Fix | Covers | Status | Cost and real-time risk | Owner? |
|---|---|---|---|---|---|
| F1 | Gas flow and momentum ON in every world; delete `_law_rise`, RISE_MAX, ceiling-jet constant, two-zone stir, eddy conductivity, legacy pressure mean, `p_follow`; book the flow's kinetic energy; check GAS_ENTRAIN 0.03 against Morton-Taylor-Turner before the flip | B6, B7, B8(f), D2, E2.1 | W15-Q8 goal, NO batch, NEW as owned task | 2 to 5 ms in a fire, quiet worlds sleep; port the projection to Rust or multigrid. Declared diffs on every fire scene | no (cost ceiling already "optimise until it fits") |
| F2 | Convergence harness (run each golden at dt/2 and dx/2 on the twin harness), FACT goldens beside bit goldens, one `VOX_M` and `TICK_S`, unit tags | D1, D9, D18, D11, D16 | NEW (machinery exists) | Test time only. Expect many small hidden-constant findings. Build first so every later change is measured | no |
| F3 | Energy-hole sweep: every law that changes capacity, mass or species calls ONE `matter_change` gate; a lint fails any write to E or mass outside it; find the writer of a negative E instead of clipping it | E2.2 (matter_rule_partial, seep_no_heat, push.work_unbooked, adj_ignored, temp_clip, work_billing, steam_to_gone, scale_unbooked) | NEW | Runtime 0, about 10 laws touched. Blocks magic's hand-off-equivalence test and long runs | no |
| F4 | One material record read by lattice, bonds and MPM: wood grain and species in MPM, stone above wood, compressive crush, pair friction (static, kinetic, film) and per-pair restitution in bonds and MPM, rolling resistance, delete WET_FACTOR in favour of the film model | A7, A13, E2.3, E(u) rank 1 and 2 | "D5", no wave | Data work plus declared diffs. Prerequisite of F5 (stiffness from moduli means nothing on wrong E) | no (SPAN retirement stays owner-gated) |
| F5 | Rod skeletons for slender things, elastic shape-matching for the rest, stored and booked elastic energy, implicit or XPBD bonds, islands, no softening on stiff solids; rods also carry plastic set (bent bar) and bows; ledger rows for the softening and the budget | A3, A4, A5, A6, A21(rides on it), D5, D12 | Model DECIDED; integrator, ledger rows, set and bow scope NEW | Rods about 1 to 5 ms per 1000 (u); the risk is the exact-by-construction ledger (kernel.py) which must be re-proved | decided; two small scope adds in my choices |
| F6 | Winter wiring: clothing resists and emits in series on the lattice path; `pour` takes a temperature; outdoor humidity follows the climate; film reads temperature (ice on a floor); `soak` default | E2.4 | NEW, small | Each under a day (u) | no |
| F7 | A broken bone carries no load (read `bonestate` in `bears`/footing); landing by contact bone, not cell axis | E2.5, `injury.fractures_unread`, `landing_mode` | NEW, small | Table lookup | no |
| F8 | Light and radiant exchange on one patch set (20 cm patches, view factors rebuilt on change): longwave + 4 shortwave bands, sun as light AND heat, reflection by a few bounces, photometric sources (candle, torch, forge glow); sight by contrast, acuity and adaptation (Blackwell, Hecht, CIE glare) | B2, B21, B22, C22, C23(a,c), B26 (sun) | NEW (map P1 "owner"), unscheduled | 1 to 3 ms amortised, cached on change; the sight consumers change everywhere: a declared-diff wave | YES (bands, true darkness, sun as heat) |
| F9 | Lumped cardiovascular model (Guyton, Heldt: 3 to 6 compartments, baroreflex, gravity) giving MAP, pulse, brain perfusion; faint and death lines become emergent; death a process with a dying interval | C12, C13, C19(d), C20 (VO2 lag later) | NEW (wave 15 builds bleeding on the linear map) | A dozen floats per person at 1 Hz. Bleeding calibration re-checked (declared) | no |
| F10 | One pharmacokinetic law: compartments, absorption routes, elimination (first order, Michaelis-Menten for ethanol), effect as Emax onto existing dials; alcohol, poison, venom, antidote, medicine, potion are rows | C17, C26 (dials), HL2 and alchemy read it | NEW (old wave 16 never built it) | A few floats per active substance at 1 Hz. Keep doses abstract (CV7 content rule) | no |
| F11 | Shape for things below a cell: thing records as the source of thickness, width and normal for heat AND mechanics; shape class (slab, rod, powder) per partial cell; staircase correction (1/(|nx|+|ny|+|nz|)); oriented partial cell for ground, walls, floors; sub-cell rope radius | A1, A2, A10(debris layer), A20, A22, B1, B4, B26(area), C2, C19(b), C7 (hand record) | Thin-plate heat part: batch 3 (owner YES). The rest NEW | Near zero per tick; computed on change. Fold the staircase and shape class into the SAME batch 3 declared diff so every quench scene changes once | no (my choice) |
| F12 | Real convection law: h = (h_nat^3 + h_forced^3)^(1/3) with orientation, length from shape class and gas speed; exact two-body exponential relaxation where the face clip bites | B5, B3(b), B19 | Forge bellows only done; general NEW | One cube root per exposed face in the numba or Rust kernel; same pass as F11 | no |
| F13 | Gait from the person (Froude, leg length, age, load, fatigue, gradient), bounded acceleration, chosen effort, foot-contact event stream (feeds footfall sound, slips, trips); balance tiers (ankle, hip, step), swing-foot trips | C8, C10, C9 (later) | "Gait, speed, stairs" is map P1 effort L; speed law NEW | Small per walker; goldens for chases move | no |
| F14 | One gas-phase combustion path: fuel vapour in the gas store, flash point as lower flammability limit, ignition piloted vs auto, fuel rows (straw, paper, cloth, thatch, tallow, coal), char oxidation and smoulder, species soot/CO/CO2 with equivalence-ratio yields, thermodynamic room pressure per closed airspace; flame correlation becomes a test | B8(1), B9, B10, B11, B12, B13, B14 | NEW (map P1/P2) | One scalar per fuel species in the active box; medium real-time risk; verify by Heskestad, McCaffrey, crib tests | YES (combustion depth, blasts) |
| F15 | Granular law: lattice slump rule at the angle of repose with cohesion and bulking, pulled forward ahead of collapse; Drucker-Prager MPM only for energetic events | A11, A12 (snow bearing) | W17 "later"; position NEW | About 0.5 ms with 1000 active cells (u) | no (my choice, section 7) |
| F16 | Structural analysis on change: load cases (dead, live, wind from the push law, water and soil pressure, impulsive with dynamic factor), connection capacity (nails, mortar), member graph for inclined members and frames, audit at scene build, collapse handed to bonds with a debris heap via F15 | A15, A16, A17, A2(b) | NEW (veto items in ledger only) | 3 to 10 ms on a layout change; collapse bursts 30 to 60 ms per tick for 1 to 3 s (needs the policy in F33) | in Q7 (burst policy) |
| F17 | Adult default body (take D7), muscle force from cross-section and composition, strength by area scaling not mass, one `sheet(age, sex, genome)` generator, bone strength by age and sex | C3, C5(b), C16, C27 | Fam3 child sheets only | Zero run cost, one big re-measure of falls and kick | YES (every old result moves) |
| F18 | Articulated skeleton (about 20 segments, joint ranges, passive and active stiffness, ligament record), voxels drawn from it; hand as a record (aperture, pads, dexterity); neck, ankle, wrist exist | C1, C2, C4, C6, C7, E2.5 | NEW; decide before 16a | Microseconds per body; the risk is migrating pose, skeleton, hands, contact and support behind one seam | YES |
| F19 | Hearing: octave bands (6) with per-band mass law, ambient from live sources, sustained sources (fire, river, forge), reverberation per airspace, air absorption, arrival delay queue shared with magic and blasts, voice directivity, source power as a booked loss | B23, B24, C25, D20 | Lombard done; rest NEW (map P1) | 6 to 10 ms for 80 pairs before caching (u); cache by source, sleep sub-audible | no (my choice) |
| F20 | float64 for conserved lattice fields (E first), in the same fused kernel pass as F11 and F12 | B3(a), D6, E2.6 | NEW | +4 bytes per voxel; bandwidth cost shrinks once fused; every golden rebased | YES (with F2's fact goldens) |
| F21 | Multi-rate law scheduling by timescale, phase-staggered, implicit conduction on its own clock; alternate law order (Strang); one dependency-graph scheduler | D3, D4, D17 | Slow clocks done for slow things only | M to L; flat worst-case tick cost; needs F2 first | no |
| F22 | Rust kernels and threads with a fixed tile decomposition and a pinned libm, person structure-of-arrays table, profile-first; cross-machine and parallel determinism | D13, D8 | Direction scheduled (W15-Q8, D4) | Profile first; per kernel M | no (my choice, veto note) |
| F23 | Night and twin coverage: a twin per slow thing, a hazard process (Gillespie via hash01), a rule for tier-1 approximation in play | D14 | Framework scheduled; hazard and rule NEW | L overall, M for the hazard | YES (tier in play) |
| F24 | Mind constants stored per second; every lab result labelled "per turn at 288 s"; rooms inn as the coarse twin of an inn in the voxel world | D15 | SC4/SC5/Fam2 partly; rule NEW | Labels are free; the twin is L | scope of the twin: my choice (do labels now, twin at family step) |
| F25 | Sparse chunked level-of-detail world (chunks of 32^3, uniform chunks cost one value, exact flux at borders), delta snapshots | D10, D22, D21, D18 | "Later" in the map; storage NEW | XL; settle the design before more laws bake in whole-array indexing | YES (world scope) |
| F26 | Water: shallow-water (virtual pipes) for open water, viscosity as a flux cap, Torricelli outflow, property functions of temperature and composition (density(T), Antoine) | B15, B17, A18 | Map P2, no wave | 0.03 to 2.5 ms measured in the map | in Q5 |
| F27 | Weather as boundary conditions: wind, rain, snow, fog from the climate clock; pressure and altitude into boiling | B18, B14 | NEW | Per event; near zero when calm | in Q5 |
| F28 | Fast movers: swept DDA traversal for anything over a cell per tick, closed-form penetration depth for a point blow; the same traversal replaces the sampled sight rays | D19, A23, C24 | NEW (weapons wave) | Event based | no |
| F29 | One impact law: contact event with the pair's restitution and energy-to-damage; fragment size from energy; sub-cell debris layer (shards, splinters) | A9, A10, A22 | Map P2 | Microseconds per breakage | no |
| F30 | Medical depth records (tendon, nerve, lung, ligament) read by the existing laws; head kinematics; chemoreflex ventilation | C14, C15, C18, C4(c) | HL1 assumes nerve records exist | Data plus one rule | YES (in Q8) |
| F31 | Senses beyond sight and hearing (smell as a scalar on the gas carrier, touch, pain location, proprioception); gaze as an act with saliency; acuity dials | C23(b), C26 | K10, SV2 partly | Small per channel; grows the mind's input | YES (Q8) |
| F32 | Cloth as a thin record plus a sparse XPBD sheet on demand; rope capstan and hitches as records; cyclic fatigue (Basquin and Miner) on rods; abrasion; crack record for panes; snow load and ice thickness | A19, A20, A21, A22, A12 | P3 mostly | Bounded by what is awake | no |
| F33 | Real-time policy: report p99 and worst tick; decide whether a heavy tick slows the world, softens, or only runs paused | D5 | NEW | Policy | YES (Q7) |
| F34 | Gas solver details: Deardorff or Smagorinsky closure instead of a constant alpha, variable density and cp(T), flame by emergent burn, nested 5 cm box or sub-grid plume near small fires | B8(2,3,4), B10, B25 | NEW | Table and a few multiplies; nested box limited to 1 m | no |

Low priority, kept visible: any-angle routes (C9), exact bone mechanics detail and growth plates (C16), VO2 kinetics and fuel mix (C20), wetting angle per surface (B16), splash by energy (B20), the retired 10 cm production path (D11, folded into F2), delayed snapshots (D22, with F25), the demotion of promoted matter (A8, see section 7), thermoregulation held-out cases and acclimation (C19: run the held-out validation, no new model), the fall multibody reference (C11, already a decided lab item).

---

## 4. The early choices by subsystem

Compact index: id | choice (file:line) | realism it costs | better | status | priority | owner. Full text, sources and costs are in the lane file named at the head of each table.

### 4.1 Matter and mechanics (A_matter.txt)

| Id | Choice | Realism cost | Better | Status | Prio | Owner |
|---|---|---|---|---|---|---|
| A1 | Cell = mass fraction (store.py:97, matlaws MIN_FILL) | Thin things have no normal, area or section axis; no sub-cell debris | Thing records, then oriented partial cells (F11) | Heat part scheduled; rest NEW | High | Technical |
| A2 | Cubic faces, staircase surfaces (support.py:35) | Slopes, rafters as staircases, rolling hops | Normal field from A1; 3-D line members (F16) | NEW | Med | Technical |
| A3 | MPM x1e-4 stiffness, strength x sqrt (core.py:434, :828, :844) | Static deflection 1e4 too large; no ringing or bounce; contact time wrong; one ledger row missing | Stop using MPM for stiff solids (F5); add the ledger row now | Partly | High | Decided in part |
| A4 | Authored `rig` per material, velocity projection that only loses energy (core.py:109-146) | No spring-back, no bow, no sag, rig is a taste | Elastic shape-matching with stored, booked energy, XPBD compliance (F5) | Decided | High | Technical |
| A5 | Slender things: nothing gives sway | No sway, set, bow, canopy loading | Rod skeleton with plastic hinge, weak-link breaking, rasterise to lattice, wind, point loads, booked energy | Decided; scope adds NEW | High | Decided |
| A6 | Bond BUDGET softens bonds to fit (store.py:56-72) | Same trunk stiffer alone than in a forest: a hidden non-locality | Implicit or XPBD bonds, islands, lumped far leaves (F5) | Partly (wave 15 list) | Med-High | Technical |
| A7 | Three solvers, three material sets; MPM wood has no grain | Same plank breaks at three loads; stone anvil weaker than a stick | One record, one softening function, compressive crush (F4) | Partly ("D5") | Med-High | Technical |
| A8 | Moving thing = one same-material cluster, promotion never undone | Composite objects, quantised draw, resting things cost memory | Demotion with an exact write-back | No | Low-Med | Owner rule (decision 3); see section 7 |
| A9 | Lattice fall energy from drop height, drag as one voxel face (support.py:79-104, :323) | Three paths for "something hit something"; voxel-size dependent thresholds | One impact law (F29) | Partly | Med | Technical |
| A10 | Fragments one cell or the cell's own size (core.py:6-8, support.py:398) | No shards, splinters, dust, grain splitting | Fragment size from energy, debris surface layer, grain-aware split | NEW | Med | Technical |
| A11 | No granular law in lattice or MPM | No piles, cave-ins, spill, collapse heap | Lattice slump rule now, Drucker-Prager later (F15) | W17 | High | Technical |
| A12 | Snow, ice not mechanical | No snow load, no ice bearing | Snow depth store feeding load and friction | P3 | Low | Technical |
| A13 | One mu 0.4, one damped contact, spheres (store.py:30-50) | Same slide on ice and stone; no stiction, no rolling resistance | Pair data, Hunt-Crossley or soft contact, rolling resistance, box members (F4) | Partly (wave 15) | Med-High | Technical |
| A15 | Beam check is a static vertical veto on changed cells (matlaws) | No wind, water, soil, dynamic loads, connections, foundations, scene-build audit | Load cases on change, audit at build (F16) | Partly | High | Technical in my choice |
| A16 | Collapse = cells falling straight down | No toppling, no progressive failure, no debris cone | Failed members to bonds, heat via A11 (F16) | Map P2 | Med-High | Burst policy in Q7 |
| A17 | Horizontal runs, not a member graph; deflection unused | No frames, trusses, shear walls; no sag | Sparse direct-stiffness on change, cached by layout hash (F16) | NEW | Med | Technical |
| A18 | Liquids with no momentum (fluids.py:1015) | No flow speed, no throttling, no force on walls | Virtual pipes, Torricelli, column velocity (F26) | Map items | Med | Scope in Q5 |
| A19 | Cloth has no mechanics (clothing.py thermal only) | No sails, smothering, sheet ropes | Thin record, sparse XPBD on demand (F32) | P3 | Low-Med | Technical |
| A20 | Rope is a chain of 5 cm cells (store.py:77) | Cannot wrap a 3 cm pole, no capstan, no knots | Curve with radius, capstan closed form, hitch records (F32, F11) | Wave 15 | Med | Technical |
| A21 | No cyclic fatigue, abrasion, wear of faces | Bows, hinges, paperclips, rope chafe never fail | Basquin and Miner on rod cycles (F32) | NEW | Low-Med | Technical |
| A22 | Damage is one scalar per cell | No crack direction or stress raiser | Crack record per pane, segments per rod | NEW | Low | Technical |
| A23 | No penetration law | Shots through doors and bodies decided by authored numbers | Poncelet or Recht-Ipson closed form (F28) | NEW | Med | Weapons plan |

### 4.2 Heat, fire, gas, fluids, light, sound (B_heat_fluid.txt)

| Id | Choice | Realism it costs | Better | Status | Prio | Owner |
|---|---|---|---|---|---|---|
| B1 | Face area sqrt(fill), staircase surfaces (heat.py:1161) | Plates wrong, powder and straw too slow, round things lose and gain heat 1.27 to 1.7x | Shape class, staircase correction (F11) | Plate scheduled | High | Technical |
| B2 | Radiation from fires and surfaces over 500 C only; sun shortwave passed as 0.0; room = one node (heat.py ~107, outside.py) | Sun heats nothing; stoves do not warm by radiation; cold windows wrong | Patches, view factors, two or more bands (F8) | NEW | High | Bands in Q3 |
| B3 | float32 E, explicit clipped face flux (materials.py:557) | Quench and forge seconds too slow; small drains lost | float64 in the active window; exponential two-body relaxation where clipped (F12, F20) | NEW | Med | Q6 |
| B4 | Thin things only as records for iron, wood, glass | A 2 cm plank, leaf, cloth use the cell's time scale (50 mm / thickness)^2 off | Make "thickness" a property of matter in a cell (F11) | Partly | Med | Technical |
| B5 | Constant h = 10 (materials.py:544) | No wind chill, no up-facing plate, no stirred bath | Churchill blend of natural and forced (F12) | Forge only | High | Technical |
| B6 | Eddy conductivity in hot gas | Room fire upside down | Deleted by F1 | With B7 | High | Technical |
| B7 | Gas flow and momentum off by default (air.py:15, gasflow.py) | No crawl low, no neutral plane, no wind, no draft | Flip, delete stand-ins (F1) | W15-Q8, no batch | High | Cost ceiling decided |
| B8 | Boussinesq, no room pressure, constant alpha, non-conserving advection, KE unbooked | No sealed-room pressure, blast, variable gas properties, joules off books | Low-Mach thermodynamic pressure per region; Deardorff; book KE; blast as event law (F14, F34) | NEW | High (1), Med (2-4), Med-High (5) | Blasts in Q4 |
| B9 | Combustion as a surface flux (heat.py:2019) | No flash point, flammable gas, backdraft, dust, autoignition vs pilot | Gas-phase path with fuel vapour (F14) | NEW | High | Q4 |
| B10 | Flame by correlation (FLAME_PAY_MAX) | Double counts, wrong when under-ventilated | Emergent flame, correlation as a test | With B9 | Low | Technical |
| B11 | One wood, no fine fuels | Thatch, straw, paper, tallow burn as pine or not at all | Fuel rows as data and the thin ignition law | Map P1 | Med-High | Technical |
| B12 | No smoulder, no embers | Fires do not return; no CO from smoulder | Char-oxidation law | Map P2 | Med | Technical |
| B13 | Smoke is one scalar with soot counting all visible products | Yield independent of fuel and ventilation; toxic effects from two calibrations | Species soot, CO, CO2, steam, HCN if needed (F14) | Partly | High | Technical |
| B14 | Fixed boil point and ambient pressure | No mountains, no sealed-pot pressure | P_ATM per world, Clausius-Clapeyron | NEW | Low | Technical |
| B15 | Liquids as cell volumes with no velocity | No currents, boats, throttling, viscosity, sloshing | Virtual pipes plus viscosity cap (F26) | Map P2 | High for water scenes | Scope in Q5 |
| B16 | One wetting value per liquid | Waxed cloak, rag, glass behave alike | Contact angle per pair | NEW | Low | Technical |
| B17 | Eight liquids, constant properties | No density(T), no lake inversion, no viscosity | Property functions from the mixture engine | NEW | Med | Technical |
| B18 | No wind, rain, snow, fog, pressure systems | No rain on fires, no snow load, no wind on smoke | Weather as boundary conditions (F27) | NEW | Med | Scope in Q5 |
| B19 | Fixed 25 ms, clip not implicit | See B3 | See B3 | NEW | Med | Technical |
| B20 | Drops and splashes by tabled speeds | Splash does not follow impact energy | Bounded MPM burst when needed | NEW | Low | Technical |
| B21 | Light does not exist | No darkness, torches, stealth by dark, glare | Light on the patch set (F8) | NEW, map P1 owner | High | Q3 |
| B22 | Sight a binary optical-depth cut | No contrast, adaptation, glare, recognising | Contrast threshold, acuity dial (F8) | NEW | High | Technical |
| B23 | One sound band, static ambient, no reverb | Muffled voices, masking by crowd and fire, echo, far travel absent | Octave bands, ambient from sources, Sabine per airspace (F19) | Partly | Med-High | Technical |
| B24 | Sound levels are named guesses, unbooked | Bellows, boil, pour, fire roar are silent | Power as a booked loss to heat | NEW | Low | Technical |
| B25 | Gas grid fixed at 20 cm | Candle, breath, kettle plume resolved as 20 cm | Sub-grid plume or a 1 m nested box | NEW | Med | Technical |
| B26 | Evaporation and humidity residue | Weather humidity not wired; salt pans | Shift to climate dew point; B1 area | Partly | Low | Technical |

### 4.3 Bodies and senses (C_bodies.txt; citations (k), check before building)

| Id | Choice | Realism it costs | Better | Status | Prio | Owner |
|---|---|---|---|---|---|---|
| C1 | 11 rigid voxel blocks (scenes.py:514-690) | No neck, spine, hand, foot, finger, wrist; trunk does not twist | Articulated skeleton, same rod primitive as trees (F18) | NEW | High | Q1 |
| C2 | Pose by re-rasterising 5 cm voxels (pose.py) | Quantised angles, 10 cm limbs, jitter | Capsules written as fractional fill | NEW | Low alone, Med with C1 | Technical |
| C3 | Default 43.9 kg stick figure, fudged constants (scenes.py:514, materials.py:344) | Calibrations are adult numbers on a 2/3 body; strength linear in mass | Adult default, PCSA, area scaling (F17) | Partly (Fam3) | High | Q2 |
| C4 | Joints are hard stops plus a damper | No sprain, dislocation, joint locks, passive stiffness | Ligament record on the rod primitive | NEW | Med-High | Technical |
| C5 | Muscle is a few scalars, no posture dependence | Lift strength ignores posture (2 to 5x); one scalar for grip, push, lift | One torque row per joint axis with torque-angle and force-velocity | Partly | Med | Technical |
| C6 | Strike energy = force x stroke | No kinetic chain | Falls out of C1 + C5 | NEW | Low | Technical |
| C7 | Hand is a lump with one N_EFF (grip.py:41) | No lock-picking, knots, three-finger draw, burned tight hand | Hand record (grasp taxonomy, dexterity dial) | Partly (grip law) | Med | Q1 |
| C8 | Walk and run are two speeds, instant starts (policy.py:75, motion.py:907) | Child, elder, giant, fit runner all the same; no head start | Speed from leg length, age, load, gradient, acceleration, effort (F13) | Partly | High | Technical |
| C9 | 8-neighbour octile routes | Staircase paths | Any-angle paths | NEW | Low | Technical |
| C10 | One step-or-grab reflex | No ankle or hip strategy, no trip response, no sensory noise | Three tiers, swing foot, sensory dials | Partly | Med | Technical |
| C11 | Fall injury as a 1-D fuse chain tuned to two population numbers | Orientation only by discrete modes | Offline multibody as a reference | Decided (lab) | Low | No |
| C12 | Blood pressure linear in volume, no heart (vasculature.py:8) | No pulse, compensated shock, orthostatic faint, tachycardia | Lumped cardiovascular model (F9) | NEW | High | Technical |
| C13 | Consciousness and death are one scalar o2 (physiology.py:693) | No dying interval, no cause interaction | Per-cause state, death as organ failure integral | Partly (map P2) | Med | Technical |
| C14 | Bones, vessels, skin only | No tendon, nerve, lung, organ by piercing | Function records addressed by wound depth (F30) | HL1 needs it | Med | Q8 |
| C15 | Head blows by a linear spring | No rotation, concussion state, delayed bleed | Head kinematics into BrIC | NEW | Med | CV7 check |
| C16 | One integrity number per bone, age by a threshold | Hip fractures of the old, stress fractures | Density x geometry by age and sex | Partly | Low | Technical |
| C17 | No pharmacology, `drunk` set by nothing (reflex.py:137) | No taverns, poison, medicine, anaesthesia, potions | One PK law (F10) | NEW | High | Technical |
| C18 | Open-loop breathing | No CO2 drive, breath-hold, hyperventilation | Grodins chemoreflex | Map P2 | Low | Technical |
| C19 | Thermoregulation fitted on steady cases only | Transients, immersion, acclimation untested; fingers too coarse | Run held-out cases; zone data; plasticity dial | Partly | Med | Technical |
| C20 | Steady-state metabolism | No O2 lag, no second wind | First-order VO2 plus W' | NEW | Low | Technical |
| C22 | Sight has no light (policy.py:30, senses.py:456) | Every place lit; no dungeon darkness | Light field (F8) | NEW | High | Q3 |
| C23 | Sight yes/no, horizontal cone, scripted gaze | See up and down free, no acuity, no attention | Acuity law, 3-D cone, saliency gaze | Partly | Med | Q8 for attention |
| C24 | Sight rays sampled, not traversed | Pose-dependent leaks at corners | Supercover DDA (F28) | NEW | Low | Technical |
| C25 | Hearing one level, one 40 dB line | No spectrum, direction, ageing, masking | Bands, SII audibility (F19) | Partly | Med | Technical |
| C26 | No smell, touch, proprioception, pain location | Smoke smelled before seen, animals without noses | One channel record per sense (F31) | Partly | Med | Q8 |
| C27 | Static body sheet | No growth, ageing, composition-based dimorphism | One generator (F17) | Partly | Low until family step | Technical |

### 4.4 Time, scale, numerics (D_time_scale.txt)

| Id | Choice | Realism it costs | Better | Status | Prio | Owner |
|---|---|---|---|---|---|---|
| D1 | One fixed tick, no convergence check (materials.py:491) | Answers change with dt and dx, found by accident | Convergence harness (F2) | NEW | High | Technical |
| D2 | Per-tick and per-voxel constants in legacy gas laws | Smoke and oxygen depend on tick and cell | Delete via F1 and test it | W15-Q8 | Med | Technical |
| D3 | Strides `tick % N`; conduction over-sampled 7000x | Aliased worst-case ticks, late effects | Multi-rate scheduler, implicit conduction (F21) | Partly | Med-High | Technical |
| D4 | Law order is a hidden cause | One-tick head start for the world | Strang alternate order, order as data | NEW | Low-Med | Technical |
| D5 | Real time is a mean | Cost shapes fidelity silently | Report p99, choose a policy (F33) | NEW | Med | Q7 |
| D6 | float32 lattice state (sim.py:190) | Slow drains lost, ulp leaks | float64 conserved fields (F20) | NEW | Med-High | Q6 |
| D7 | Conservation by accounting (ledgers.py:37) | A real error can hide behind "f32" | Integer quanta by construction, only if justified | NEW | Med | Q6 |
| D8 | Determinism machine-bound | Replays diverge on another CPU or with threads | Fixed reduction trees, pinned libm, Rust (F22) | Partly | Med now, High before sharing saves | Technical |
| D9 | Bit goldens brake every real improvement | Mass declared diffs | Fact goldens beside bit goldens (F2) | NEW | Med | Technical |
| D10 | One dense box (sim.py:196) | No village, yard, road, arrow flight, fire to next roof | Sparse chunks plus detail ladder (F25) | Later | High | Q5 |
| D11 | 5 cm with a dead 10 cm path | Code cost of `scale` | Keep 10 cm only as coarse rung of the harness | NEW | Low | Technical |
| D12 | Stiffness in time-steps | See R1 | Implicit, XPBD, rods, islands (F5) | Model decided | High | Technical |
| D13 | One thread, numpy temporaries, Rust unused | Speed is realism: gates stay off because of cost | Profile-first Rust and threads, person SoA table (F22) | Direction scheduled | High | Distribution note |
| D14 | Night jump only where a twin exists | Waiting is slow or frozen where no twin exists | Twin per slow thing, hazard process, tier rule (F23) | Partly | High | Q7 |
| D15 | Two clocks: tick 25 ms and mind turn 3 to 5 minutes | Mind results do not transfer; fade 10x too fast was caught once | Seconds as the unit; rooms inn as coarse twin (F24) | Partly | High | Technical now, scope later |
| D16 | Optional short day | A fast planet confounds body and calendar | Forbid `day_s` other than 86400 outside tests | NEW | Low | Technical |
| D17 | Scheduler and law loop are two machines | New laws must join both by hand | One read/write dependency graph (F21) | NEW | Med | Technical |
| D18 | `0.1 * self.scale` in about 199 places | Silent unit errors | `VOX_M`, unit tags, scale-invariance test (F2) | NEW | Med | Technical |
| D19 | Things faster than a cell per tick (FALL_SUBSTEPS 16) | Bullets and sword tips pass through thin matter | Swept DDA (F28) | NEW | Med | Technical |
| D20 | Signals instant | No thunder delay, echo | Delayed-effect queue (F19) | NEW | Low, Med with D10 | Technical |
| D21 | Offscreen people full or nothing | Frozen or expensive buildings | SC4 as decided | Scheduled | High | Decided |
| D22 | Snapshots are whole-world copies | Large saves, undo cost | Delta snapshots on chunks | NEW | Low-Med | Technical |

### 4.5 The switch ledger and the (u) constants (E_ledger.txt)

| Id | Choice | Realism it costs | Better | Status | Prio | Owner |
|---|---|---|---|---|---|---|
| E2.1 | Gas momentum gate off, four dependents | Fire scenes wrong | F1 in one row's condition | NEW as task | High | Technical |
| E2.2 | Eight energy-hole crutches | Magic faithfulness cannot be tested; long runs drift | F3 | NEW | High | Technical |
| E2.3 | Friction 0.40, MPM inversion, SPAN floor | Boulders weaker than logs, wrong repose angle | F4 | NEW named wave | High | SPAN retirement owner-gated |
| E2.4 | Cold and clothing rows | Naked and clothed lose equal heat; January water at 20 C | F6 | NEW | High | Technical |
| E2.5 | The rigid person: `fractures_unread`, slip speed, lunge | Broken leg bears weight | F7 and F13 and F18 | Partly | High (F7), Med (legs) | Legged model with Q1 |
| E2.6 | float32 E, fresh-air band | Hidden small errors | F20 | NEW | Med | Q6 |
| E2.7 | One optical-depth threshold, one Cd | Dim, small and lit targets alike; sphere 2.5x off | Contrast law, Cd per shape | NEW | Low-Med | Technical |
| E hygiene H1 to H5 | No wave column; deleted_when never checked; 28 gates with no flip date; E_dtype, lattice cp, zero of E, tick and grid have no rows; (u) constants not tabulated | The ledger cannot say which crutches have no home | Add `wave`, `flip_when`, a `check` that fails a met condition; split `craft.missing_laws` into six rows; dedupe duplicates | NEW | Med | Technical |

The (u) constants to source first, ranked by play load (E section 5): friction pairs and WET_FACTOR; wood strength retention and the stone stand-in; impact and head thresholds (HIC 2500 as fatal looks high); digestion; thermoregulation fits; tissue tables; the fight lane numbers; GAS_ENTRAIN (0.03 against 0.09 to 0.16 in the literature, check before the F1 flip); smithing constants (already requested).

---

## 5. Where the new fixes go in the build order

Current order: wave 15 batch 2 (paused, partly merged) -> batch 3 (stairs hull, thin-plate area, ferment slice, default-on flips) -> batch 4 (healing, twins) -> 16a (primitives) -> 16b (knowledge and plan cache) -> magic slices; family step and the lab alongside; soil and digging "later".

Principle: **one declared-diff event per kernel.** Every golden is bit-exact today, so each fix that moves numbers costs a rebase. Group them, and let F2's fact goldens judge the diffs by facts (ignition time, flame height, core temperature, drop energy) rather than by bits.

| Step | Add | Why here | Depends on |
|---|---|---|---|
| After the pause lifts, before any batch | F2 (harness, fact goldens, `VOX_M`, unit tags); ledger hygiene H1 to H5; F24 labels (per turn at 288 s) | Tests and labels only, no behaviour change; every later step is measured by it | Nothing |
| Batch 2 (finish) | Nothing new. Sweep the stale ledger rows after the merge | Do not enlarge a paused batch | |
| Batch 3, 3a | Stairs hull as planned; F7 (broken bone carries no load; landing by bone) | Same files (stairs, bonestate, injury) | |
| Batch 3, 3b: the heat declared-diff event | F11 heat part (thin plate, shape class, staircase), F12 (convection, exponential relaxation), F20 (float64 E) in the same fused kernel pass | heat.py is already open for the thin plate; every quench and forge scene changes once | F2 |
| Batch 3, 3c: the flips | F3 first (so the flips book correctly), then F1 (gas flow on, stand-ins deleted, KE booked), F6 (winter rows), the DATA part of F4 (one record, deletions of the inversion and WET_FACTOR) | The flips are the batch's own job; F3 and F4a are data and lints, no new loops | F2, F3 before F1 |
| Batch 4 | Healing as planned, plus F9 (cardiovascular, death as a process) and F10 (pharmacokinetics) and the nerve, tendon, lung records of F30 | HL1 repairs nerves and needs perfusion; HL2 (honey, wine) reads substance rows; one physiology pass | F3 |
| NEW wave 15c, "solids", before 16a | F5 (rods, elastic shape-matching, implicit or XPBD bonds, islands, no softening), F4 solver part (pair friction, restitution, rolling), F29 (one impact law, debris layer), F11 mechanical part (thing records, oriented cells for ground and walls), rope radius and capstan, then the body seam: F18 first slice (neck, hands, feet as rod segments, hand record) and F17 (adult default) at the same rebase | The rod primitive is shared by trees, planks and bones; the body must be on it before 16a builds six motor primitives on 11 blocks | F4 data, F11, F2 |
| 16a | F13 (gait, balance tiers, swing foot), C5 torque by posture, verbs on the hand record, F28 swept DDA for the point-blow primitive | The primitives need person-dependent speed and joint torque | 15c |
| NEW wave 15d, "light and radiant", before 16b | F8 (patch set, sun as heat, light, contrast sight), C23 acuity and 3-D cone | 16b's knowledge layer uses sight as a channel; magic's waste light needs the bands chosen once | F1, F11, Q3 |
| 16b | F19 (hearing bands, ambient from sources, reverberation, delay queue), F23 (hazard process, tier rule), F24 (b) first coarse-twin tests, F31 as decided | K3 and K4 rely on the sound and sight laws; they need these to be physical | 15d |
| NEW wave 15e, "gas-phase fire", before magic slice one | F14 (vapour, species, fuel rows, pressure, smoulder), F34, the blast event law if Q4 says so | A fireball from a gem needs the fire it hands to; smoke is the main killer | F1, F8 (light of flame), F3 |
| Soil and digging wave (W17) | F15 pulled to its start (lattice slump rule), then F16 (loads, member graph, collapse to bonds), scene-build audit | Collapse leaves no heap without F15; wind and water loads need the push law and fluid velocity | F5, F15, F26 (for water loads) |
| By scene demand, with the world-scope answer | F26 water, F27 weather, F25 chunks (settle the storage interface NOW, build when the first village or outdoor scene arrives) | Scope calls (Q5) | F25 design before more laws bake in whole-array indexing |
| Performance track, parallel to everything | F22 (profile first, Rust face kernel, gas projection, person table), F21 (multi-rate, alternate order) | Speed is what lets the flips stay on; each kernel keeps the "bit for bit against Python" rule | F2 |

Dependency spine, for reference: F2 -> F3 -> F1 -> F14; F4(data) + F11 -> F5 -> F18 -> 16a; F1 + F11 -> F8 -> 16b; F5 + F15 + F26 -> F16.

---

## 6. What to keep as it is, and why

- **The 25 ms tick.** It resolves every body law (reflex 90 to 120 ms is 4 to 5 ticks). Do not shorten it; schedule the slow laws on their own clocks and keep the fast ones at tick or finer (D3).
- **Conduction stability.** r = alpha dt / dx^2 is about 2e-4 for iron at 5 cm: nowhere near the limit. The problem is over-sampling, not instability.
- **The 5 cm lattice with records for sub-voxel things (the "levels" principle).** Do not add a finer sub-grid (A1c is the heavy option). Give things geometry as data (F11). The 20 cm gas grid is also fine; add a sub-grid plume correction only where a small fire needs it.
- **Counter-based chance (`hash01`), the twin protocol, the SI-to-lattice seam in `matter.py`.** Right shape. Use them more, do not redesign.
- **Exact-by-construction bond conservation, until the integrator changes.** It is valuable; F5 must re-prove it, not drop it silently.
- **Creep clock (Kachanov-Rabotnov, exact integration), Archard edge wear, Euler and Johnson buckling and brittle or ductile bending in `matbeam`, the weak-link breaking law, per-pair friction data in `matfric` (apart from WET_FACTOR), sleep for rested matter, the one material record `matrec` itself.** Sound laws; extend them (rods reuse the breaking law per segment).
- **The ledger as a system** (gates, crutches, estimates, `deleted_when`). Honest and mostly well formed; fix hygiene, not design.
- **The beam veto as a safety, until the scene-build audit exists.** It never destabilises old scenes.
- **The body laws that have a law:** Pennes heat, Purser dose, bone beam check, fuse-chain falls (validated later against an offline multibody, already a decided low-priority lab item), vessel bleeding, XcoM balance, three-compartment fatigue.
- **The cellular liquid for still water.** Hydrostatics, exact mass and sleep work. Add flow speed only where water moves (F26).
- **float32 for derived fields** (smoke, oxygen). A relative 6e-8 is fine; only the conserved fields need more (F20).
- **Per-tick elasticity (FEM) stays rejected** (realism-map). The structural proposal is event-driven coarse beams, not per-tick FEM.
- **Octile routes, the fall multibody reference, the retired 10 cm path as a coarse rung:** low impact; leave until something needs them.

---

## 7. What I decided as technical (veto any), and the owner-level questions

Decided by me, with the reason:

1. **Granular matter: lattice slump rule first, MPM Drucker-Prager only for energetic events.** It avoids needing the demotion of promoted matter (A8), which decision 3 forbids as a standing rule, and keeps books exact. If a scene later needs MPM sand handed back to cells, I will ask then.
2. **Structure: audit at scene build, wind and water loads on structures, connection capacity, all as declared diffs.** It follows "every law on in every world". SPAN retirement stays owner-gated.
3. **Rods carry a permanent set and bows** (A5 items 1 and 6). Small scope, same primitive.
4. **Hearing: 6 octave bands, cached, sub-audible sources asleep, with the single-band path as the golden-identical case for a flat spectrum.**
5. **Rust kernels with threads on a fixed tile decomposition and a pinned libm; no more than physical cores minus 2** (heat on this laptop is a known constraint). The shipping and toolchain question is left for when anyone else builds.
6. **Mind constants per second, labels now; the coarse twin of the inn at the family step** (not before).
7. **`day_s` other than 86400 forbidden outside tests.** Debris layer for sub-cell shards (modest harm model). Penetration depth as a closed form, with weapons depth left to the weapons plan.
8. **The softening and the bond budget are deleted only after F5 re-proves conservation**; meanwhile give the softening and the bond budget ledger rows now.

Owner-level (8, in `~/.cache/mark1-build/engine-audit-questions.txt`, one at a time):

| Q | Choice | Recommendation |
|---|---|---|
| Q1 | Body representation: articulated skeleton with voxels as skin, hands as records | Yes, phased, before 16a |
| Q2 | Take D7: adult default body, area-scaled strength | Yes, in the same rebase as the body seam |
| Q3 | Light: a physical field, true darkness, the sun as light and heat, spectral bands | Yes, 4 shortwave and 2 thermal bands, once |
| Q4 | Combustion depth and blasts | Gas-phase path now; blasts as an event law with weapons |
| Q5 | World scope: inn, village or region; outdoor water and weather | Design chunked storage now, build for a village when a scene needs it |
| Q6 | Precision and conservation: float64, integer quanta, or booked float32 | float64 conserved fields; integers only if the audit finds them necessary |
| Q7 | Real-time policy and what "wait" may approximate | Slow the world for watched heavy bursts; paused play pays nothing; tier-1 approximation allowed only with a measured bound |
| Q8 | Character depth the player can miss or suffer: attention, smell and touch, tendon, nerve, lung | Yes to smell, touch, nerve, tendon, lung; attention after 16b |

---

## 8. Cautions and disagreements between the lanes

- **Citations are from memory.** Lane A has a few (v) web checks (Kugelstadt and Schomer 2016, Klar et al. 2016, FastLSM); lanes B, C, D and E are almost all (k). Fetch through Babbel before any number goes into code. Second library request needed: radiative exchange and light (Modest, Siegel and Howell, IES, Blackwell, Hecht), cardiovascular lumped models (Guyton, Heldt), pharmacokinetics (Widmark, Gabrielsson and Weiner), combustion (FDS reference, Drysdale, SFPE chapters), structural load cases and progressive collapse.
- **Lane C could not find the light field or pharmacology in decisions.md; I confirmed** (grep) that neither a light field, a heart model nor pharmacology is scheduled; only a smoke-sight read of the sun lux (14a R1) exists.
- **Costs are estimates.** The lanes agree on the scale (a few ms each) but none was measured. The real-time risks that matter are F1 (gas projection in a fire), F5 (rods and an implicit solver in a forest), F8 (a second field per room), F19 (bands x ear pairs) and F16 (collapse bursts over 25 ms). Each needs a profile before it is committed to.
- **Lane overlaps resolved here:** gas flow flip (B6, B7, D2, E2.1 became F1); float32 (B3, D6, E2.6 became F20, with D7 kept as a separate owner choice); light (B21, B22, C22, C23); softening and integrator (A3 to A6 and D12 became F5); material record (A7, A13, E2.3 became F4). Lane A proposes oriented cells, lane B proposes a shape class: they are the two halves of F11.
- **A friction-wave name in the ledger (W5b) does not exist**; rewrite that `deleted_when`. A few ledger rows may already be met after the batch 2 merge (`mix.crystal_mat21`, `body.work_billing`, `injury.bleed_grip_unwired`, `edge.temper_one`, `stairs.mechanism_rows`, `contact.reflex_foothold`): verify and shrink them then.
- **Nothing here is started.** All of it waits for your word on the pause, and the ranking should be re-read after the first harness run (F2), which will probably add findings of its own.

## Owner decisions (2026-11-22, EA1-EA8)

- **EA1, the body:** ONE ARTICULATED SKELETON (about 20 segments, joint ranges, passive and active stiffness, ligaments) on the SAME ROD AND JOINT PRIMITIVE as trees and planks; voxels as a skin derived from the pose; hands as records. Phased (neck, hands and feet first), the first slice in a NEW WAVE 15c before 16a. Fractures must be read.
- **EA2, the default body:** the ADULT male and female sheets, in the same pass as EA1. Muscle force from cross-section (about 300 kPa), strength by area; falls and the kick re-measured once; goldens rebased once.
- **EA3, light:** a PHYSICAL LIGHT FIELD on the radiant-exchange patch set, 4 shortwave plus 2 thermal bands, sight by contrast and dark adaptation, TRUE DARKNESS ON (torches needed), as a NEW WAVE 15d before 16b. NATURAL SOURCES are physical emitters: the sun with sky, twilight and seasons; the moon as reflected sunlight with phases (the MC5 orbit); stars and skyglow; lightning; incandescence by temperature (Planck); bioluminescence as biology; aurora and meteors; magic light.
- **EA4, fire:** GAS-PHASE COMBUSTION (fuel vapour, flash point, ignition, smoulder, species soot, CO and CO2, room pressure) as a NEW WAVE 15e before the first magic slice. BLASTS when weapons and explosives are planned.
- **EA5, world size:** the CHUNK STORAGE INTERFACE designed now, to SCALE TO REGION AND CONTINENT SIZE (the owner: ideally humongous, like OpenWorld, made affordable by coarse processing). Build toward a village when the first outdoor scene is wanted. A smaller area done correctly first.
- **EA6, precision:** FLOAT64 for conserved fields plus FACT GOLDENS, with batch 3's heat event. Integer quanta only if the leak sweep or the convergence harness shows rounding hiding real errors.
- **EA7, heavy ticks and waiting:** SLOW THE WATCHED WORLD for heavy ticks (no silent softening; report p99 and the worst tick). APPROXIMATE TWINS with measured drift budgets for waiting, logged. Both to be improved.
- **EA8, depth:**
  - MEDICAL function records (nerve, tendon, lung, ligament) and MORE SENSES (smell, touch, proprioception, pain location) now;
  - ATTENTION (gaze as an act) after 16b.

**The build order now:** wave 15 batch 2 (paused) -> batch 3 (thin plates, float64 plus fact goldens, the gas-flow flip F1, the convergence harness F2, the energy-leak sweep F3) -> batch 4 (healing) -> 15c solids (one material record, rods plus elastic shape-keeping, the skeleton and adult bodies, medical records, the heart and perfusion model, pharmacokinetics) -> 15d light (and the touch, proprioception and pain senses) -> 15e gas-phase fire (and smell) -> 16a primitives -> 16b knowledge and plans -> magic.
