# Realism references for the voxel sim

Date: 2026-10-02. Status: research and comparison only. Nothing in `src/` was changed.

This file compares the voxel sim with measured real-world values. It covers nine
domains: mechanics, heat, fire, fluids, chemistry, materials, the body, sound and
perception. Each domain has one table. Section 3 ranks the gaps. Section 4 lists the
reference tests for later build waves. Section 5 lists the sources.

## 1. How to read this

### 1.1 Which code

- All line numbers point to the stable snapshot `build/base0` (`src/voxel/sim.py`,
  `src/mpm/core.py`, `src/mpm/matter.py`, `src/voxel/scenes.py`). The live `src/` is
  being refactored, so its line numbers will drift. Use the **symbol name** as the
  stable key and the line number as a hint.
- `sim.py:724` means `src/voxel/sim.py`, line 724. `core.py` and `matter.py` are in
  `src/mpm/`. `scenes.py` is `src/voxel/scenes.py`.
- "Roadmap 90" means the entry numbered 90 in `docs/voxel-roadmap.md`.

### 1.2 Units

- Real values are in SI. Temperatures are in °C. Sound levels are dB SPL (re 20 µPa);
  dBA is A-weighted.
- The sim uses these units:

| Sim quantity | Unit and meaning |
|---|---|
| `TICK_S` | 0.025 s. One second is 40 ticks (sim.py:725). |
| `voxel_cm` | Edge of one voxel. The `World` default is 10 cm. Scenes with people use 5 cm. `scale = voxel_cm / 10`, `vox_m = 0.1 * scale`. |
| `E` | Heat in joules **above ambient** (`AMBIENT` = 20 °C, sim.py:200). |
| `smass` | Solid mass in grams per voxel. |
| fluid | Volume in ml per voxel. One voxel holds `cap = 1000 * scale**3` ml. |
| `smoke` | Grams of combustion gas **per voxel** (not per litre). |
| Per-tick constants | Many laws use a fraction or an amount **per tick** at the 10 cm reference size. A per-tick rate `r` is `r / TICK_S` per second. |

- One conversion is used often in this file. The conduction law moves
  `k * scale * ΔT` joules per face per tick. Its SI equivalent is
  `k_SI = k * scale / (vox_m * TICK_S) = 400 * k` W/(m·K), at **any** voxel size.
  The reverse is `k_sim = k_SI * 0.0025`.

### 1.3 What "ok for a game" means

The player must not notice the difference. As a rule:

- Times must be within a factor of about 1.5 of the real value.
- Thresholds (heights, temperatures, forces, levels) must be within about 20%.
- When the measured range in the source is wider than this, the source range is the band.

### 1.4 Verdicts and tags

| Verdict | Meaning |
|---|---|
| matches | Inside the real range, or inside the game band. |
| ok for game | Outside the real range, but the player will not notice it. |
| wrong | Outside the game band. A player will notice it. |
| missing | The sim does not model it. |
| unknown | Not measured in the sim yet, or no real value was found. |

| Tag | Meaning |
|---|---|
| (derived) | Worked out from the code or from a formula. No sim run was done. |
| (unverified) | No opened source states this value. Do not build a test on it without a source. |
| (secondary) | The source is a summary of other work. The primary was not opened. |

### 1.5 How the numbers were checked

Nine researchers collected the values. A second reviewer then checked each row
against its source and against the snapshot code. This file applies every correction
from that check. A value that the check could not trace is dropped or carries a tag.
The writer opened the sources that the check added (Section 5 marks them).

## 2. Comparison tables

"Src" is the number of the source in Section 5. A dash means no outside source (the
value is an internal invariant or a derivation).

### 2.1 Mechanics and motion (M)

Throws, jumps, leaps and fall injury are in 2.7 (Body). Glass drop height, beam
strength and trunk strength are in 2.6 (Materials).

| # | Quantity | Real value | Src | Sim value | Verdict |
|---|---|---|---|---|---|
| M1 | Gravity | 9.80665 m/s² standard (local 9.78–9.83) | 1 | `GRAVITY` 9.81 (sim.py:724); MPM `GRAVITY` (core.py:39) | matches |
| M2 | Free fall from 1 / 2 / 3 m | 0.452 / 0.639 / 0.782 s; impact 4.43 / 6.26 / 7.67 m/s. Drag on a 5 cm wood cube adds less than 2% (derived). | 1 | `_fall_speed` (sim.py:1408), `FALL_SUBSTEPS` 16 (sim.py:740). Derived: 18 / 26 / 32 ticks (0.45 / 0.65 / 0.80 s), 4.36 / 6.20 / 7.52 m/s. | matches |
| M3 | Landing energy | ½·m·v² at contact | 1 | `_cash_impacts` uses m·g·fallh·vox_m (sim.py:1647) and ignores drag. Overstatement ≈ 2gh/v_t². A frac-0.1 wood cell (v_t ≈ 6.7 m/s) lands 4.4× too hard from 10 m and 8.8× from 20 m. A full wood cell (v_t ≈ 21 m/s) is barely affected (derived). | ok for game |
| M4 | Air density and drag | Air 1.204 kg/m³ at 20 °C. Cd: cube 0.8, square flat plate 1.17, rectangular box 2.1, standing person 1.0–1.3. Cube 1.05 is often quoted (unverified). Disk 1.17 (NACA TN 253). | 2, 3, 4, 5, 6 | `AIR_DENS` 1.2 (sim.py:733), `DRAG_CD` 1.1 (sim.py:737), frontal area from voxel faces. MPM bodies have no air drag. | matches |
| M5 | A person keels over (faint) | 0.7–1.2 s from loss of balance to impact (falls with protective responses). The same paper's lab forward falls took 0.60–0.84 s. A rigid 1.5 m rod from 2–11° takes 0.9–1.5 s and lands at ≈ 4.4 rad/s (upper-bound idealisation). | 7, 8 | Owned bodies use a per-tick pendulum in `_law_bodies`: alpha = sin(θ+φ0)/L per tick², omega ≤ 0.35 rad/tick (sim.py:3496–3497). It has no g. Its effective g is vox_m/TICK_S² = 80 m/s² at 5 cm, 160 at 10 cm. Derived: 11–22 ticks (0.28–0.55 s), lands at 13.4–14 rad/s. Injury uses COM drop × g (sim.py:3630), so the hurt amount is right and only the motion is wrong. | wrong |
| M6 | Pole or tree topple | Rod θ'' = (3g/2H)·sinθ (eq. 3). A 2 m pole from rest: 2.78 s from 0.1°, 1.93 s from 1°, 1.53 s from 3°, 1.27 s from 6° (time depends on the log of the start angle). Impact angular speed 3.84 rad/s (2 m), 3.13 rad/s (3 m); tip 7.7 / 9.4 m/s. On a floor with µ 0.4 the base slips near 51.5° (derived from eq. 3). | 8 | `_law_torque` (sim.py:4035) → `_topple` (sim.py:1806) → MPM with true g, Coulomb µ 0.4 (core.py:151) and shape matching. Never measured. | unknown |
| M7 | Friction | Static: wood–wood 0.25–0.5, oak–oak 0.62 (kinetic 0.48), wood–stone 0.2–0.4, wood–concrete 0.62, wood–brick 0.6, masonry–brick 0.6–0.7, cast iron–oak 0.49, leather–oak 0.61 (kinetic 0.52), ice–ice 0.1 (kinetic 0.02). | 9 | `FRICTION` 0.40 for every pair (sim.py:719); MPM friction 0.4 (core.py:151). On the lattice it is only a static shove threshold (`_effort`, sim.py:8344). Nothing on the lattice slides; a landing flier loses its sideways speed at once. | ok for game (wood); low for stone on stone; no ice or grease |
| M8 | Arrow drag and speed | Arrow Cd on the shaft section: 1.45–1.6 laminar, 2.6–2.65 turbulent (5.25 mm target arrow; an extrapolation for a 10 mm war bolt). War-arrow launch 47.2 m/s for 102 g from a 144 lbf yew bow. A 150 lbf replica shot 53.6 g to 328 m and 95.9 g to 249.9 m. | 12, 13 | `arrow_trap` bolt 60 g at 50 m/s, `area_m2` 8e-5 (scenes.py:1322–1323), `_drag_a` (sim.py:3254) with Cd 1.1 → CdA 8.8e-5 m². Derived: 49.25 m/s and 0.845 m drop after 20 m. | ok for game (drag 1.4–2.3× low) |
| M9 | Bounce (restitution) | Dropped 92 cm onto concrete: wooden ball 0.60, steel ball 0.60, glass marble 0.66 (student data). Wooden block 0.40–0.55 (no citation in source). | 20, 21 | Lattice landings have e = 0 (`_cash_impacts` sim.py:1625, `_land_body` sim.py:3577). MPM wood block e ≈ 0.26 (roadmap 94). | ok for game, but the two paths disagree |
| M10 | Tick length and integration | No constant. A body must not move past the collision sweep in one tick. | 1 | `TICK_S` 0.025 (sim.py:725). Falls re-check every cell; flights are swept cell by cell; MPM sub-steps by CFL. Semi-implicit Euler adds ≈ g·t·dt/2 of drop (+0.05 m over a 20 m bolt flight). | matches (tests must allow the tick bias) |
| M11 | Glass shard scatter | ≈ 3 m in forensic work (unverified; no primary opened) | — | `_shatter` (sim.py:1715) spreads mass to 4 neighbour cells, 12 when overload ≥ 2 | unknown |

### 2.2 Heat transfer (H)

The `k` column is converted with `k_SI = 400 * k` (see 1.2). Diffusivities and
"times faster" figures are derived.

| # | Quantity | Real value | Src | Sim value | Verdict |
|---|---|---|---|---|---|
| H1 | Wood conductivity | Pine 0.12–0.15, oak 0.17 W/(m·K); diffusivity ≈ 1.2e-7 m²/s | 23, 24 | `SOLID[WOOD]` k 2.0 (sim.py:212) → 800 W/(m·K) | wrong (≈ 5000×) |
| H2 | Stone conductivity | Granite 1.7–4.0, limestone 1.26–1.33, sandstone 1.7, marble 2.08–2.94, dense brick 1.31 | 23 | `SOLID[STONE]` k 3.0 (sim.py:213) → 1200 | wrong (300–950×) |
| H3 | Iron conductivity | Pure iron 83.5 at 0 °C, 54.7 at 327 °C, 28.2 at 927 °C; wrought 59, cast 52, 0.5% C steel 54 | 25, 24 | `SOLID[IRON]` k 20 (sim.py:214) → 8000, no temperature dependence | wrong (100–160× at room T, ≈ 280× at 927 °C) |
| H4 | Glass conductivity | 0.8–0.96 | 23, 24 | `SOLID[GLASS]` k 1.0 (sim.py:217) → 400 | wrong (≈ 430×) |
| H5 | Flesh conductivity | Epidermis 0.25, dermis 0.45, tissue 0.293, blood 0.582 | 26 | `SOLID[FLESH]` k 1.0 (sim.py:215) → 400; `FLUID[BLOOD]` contact k 0.5 (sim.py:235) | wrong (≈ 1000×) |
| H6 | Air conductivity | 0.0262 at 25 °C | 23, 27 | `SOLID[AIR]` k 0.6 (sim.py:211) → 240, plus (T−20)/60 for hot air (sim.py:4272) | wrong for still air (≈ 9000×); the hot-air boost is a fair stand-in for convection |
| H7 | Charcoal conductivity | 0.084 | 23 | `SOLID[CHAR]` k 0.8 (sim.py:219) → 320 | wrong (≈ 3800×) |
| H8 | Contrast iron : stone : wood | ≈ 350–550 : 9–27 : 1 | 23, 25 | 20 : 3 : 2, so 10 : 1.5 : 1 | wrong. Ruling 1's "metal answers fire faster than stone" holds only in direction. |
| H9 | Conductance across a face | Two half-cells in series give the harmonic mean 2·ka·kb/(ka+kb). A solid–gas face needs a convective h. | — (derived) | `kpair = min(ka, kb)` (sim.py:4284; kernel sim.py:84–86) | wrong (up to 2× low) |
| H10 | Specific heat | kJ/(kg·K): oak 2.0, pine 2.5, granite 0.79, limestone 0.91, iron 0.46, steel 0.49, glass 0.84, lead 0.13, tin 0.24, charcoal 1.0, water 4.184, skin 3.39, whole body 2.98–3.47 | 31, 28, 32 | `SOLID` c (sim.py:211–221): wood 1.7, stone 0.8, iron 0.45, flesh 3.5, glass 0.8, char 1.0, tin 0.23, lead 0.13; `FLUID` water 4.2 (sim.py:228), blood 3.6 (sim.py:235) | matches |
| H11 | Density | kg/m³: oak 720, charcoal 300–400, granite 2600–2800, steel 7820, cast iron 7200, glass 2400–2800, lead 11350, tin 7280, tissue ≈ 1050–1200 | 33, 26 | `SOLID` density in g/L = kg/m³ (sim.py:211–221): wood 600, stone 2700, iron 7800, flesh 1000, glass 2500, char 300, tin 7300, lead 11340 | matches (flesh 5–15% light) |
| H12 | Heat capacity of air | ρ·cp ≈ 1.2 J/(L·K) (cp 1.006 kJ/(kg·K)) | 27, 81 | `c_airbase = 50 * vox_l` J/K added to **every** voxel (sim.py:914, used at sim.py:1054). Air gets 50 J/(L·K). Full wood gets +5%, full char +17%, ash +42%; thin partial voxels are dominated by it. The code calls it the "thermal floor of an empty voxel". The stability motive is unverified. | wrong (41×) |
| H13 | Latent heats | Water 2257 kJ/kg; fusion Fe 247, Pb 23 (NEA: 23.8 ± 0.7), Sn 59 kJ/kg | 28, 34, 87 | `BOIL, LATENT` 100 °C, 2260 J/g (sim.py:772); `MELT` (sim.py:789) | matches |
| H14 | Heat lost to the surroundings | Free convection h 2–25 W/(m²·K) plus ε·σ·(T⁴−T∞⁴), from **exposed faces only**. ε: oxidised steel 0.79, wrought iron 0.94, oak (planed) 0.89, pine 0.95, glass 0.92–0.94, water 0.95–0.963. A fully exposed 5 cm iron cube at 1000 °C loses ≈ 1.9 kW (derived). | 29, 35 | `LEAK` 0.004 (sim.py:776): `E *= (1 − LEAK)` on every voxel each tick (sim.py:8694). Time constant TICK_S/LEAK = 6.25 s for any size, exposure or material. A 5 cm iron voxel at 1000 °C leaks ≈ 70 kW (derived). Measured: 1000 J alone in a closed room is 779 J sixty ticks later (roadmap, "Friction, and the joules nobody was counting"). | wrong (≈ 35× at 1000 °C, > 250× at 100 °C, unbounded for buried cells) |
| H15 | Radiation to the outside | ε·σ·A·(T⁴−T∞⁴) from exposed faces only | 35 | `RAD` 3e-9 (sim.py:774): `E −= RAD·scale²·(Tk⁴−293⁴)/3` on every voxel (sim.py:8693), ≈ 70× one black face, buried voxels too. constants.md says it was tuned "so flames cap". | wrong |
| H16 | Radiation between neighbours | Linearised h = 4σT³: 12 W/(m²·K) at 100 °C, 151 at 600 °C, 366 at 900 °C. It crosses clear air, not opaque solids. | 35 | `kr = RAD·Tk³·scale²` per face (sim.py:4288–4293) ≈ 53× black body (19,400 vs 366 at 900 °C). It acts on every face pair, also inside iron, stone and wood. At 900 °C and 5 cm it adds 1.21 J/K per tick, more than wood's whole conduction. Its share grows with voxel size. | wrong |
| H17 | Radiant heat at a distance | Point source q = χr·Q/(4πR²), χr 0.15–0.60; it over-predicts close to the fire (good to 5% only for L/D > 2.5). ISO 13571 time to pain t = 4.2·q^−1.9 min (44 s at 2.5 kW/m², 12 s at 5, 3 s at 10), ±25%. Second-degree burn t = 6.9·q^−1.56 min. ISO ignores radiant dose below 2.5 kW/m². Fire Service Academy table: > 30 s tolerated at 2.5 kW/m². A 180–200 °C hot layer gives ≈ 2.5 kW/m². | 36, 37, 38 | Not modelled. Heat crosses air only through neighbour pairs, hot-air mixing and `_law_rise`. | missing |
| H18 | Hot-air clip | A face flux must never push a cell past its source temperature. | — (derived) | The clip is asymmetric: [−0.2·C[b], +0.2·min(C)]·abs(ΔT) (sim.py:4294–4295; kernel sim.py:105). A formula replica took an air cell at 600 °C, with 1000 °C iron on its +x, +y, +z faces, to 1763 °C in one tick; the mirrored geometry gave 840 °C (derived, not seen in a scene). | wrong (bug) |
| H19 | Stability of the update | An explicit 3-D update needs a per-face fraction of 1/6 or less. | — (derived) | 0.2 per face with T frozen across all three axes (sim.py:4267). A replica checkerboard in uniform 400 °C air at 5 cm grew ×1.4 per tick (derived). Hot-air mixing depends on grid only between ≈ 114 K and ≈ 564 K excess. | wrong (bug) |
| H20 | Liquid on a solid | Water k 0.6065. h: free convection of liquids 50–1000, forced 100–20,000, boiling or condensing 2,500–100,000 W/(m²·K). | 28, 29, 30 | Contact k `FLUID[WATER]` 30 (sim.py:228; wet mask sim.py:4274) → h ≈ 240,000 at 5 cm, 120,000 at 10 cm. `FLUID[OIL]` 1.5 → 12,000 at 5 cm. | wrong (above the real ceiling, grid dependent, no vapour blanket) |
| H21 | A glowing bar cools | A 25 mm steel bar in still air cools 800 → 500 °C at 0.89 °C/s, ≈ 340 s. Glow ends near 525 °C (Draper point). A 50 mm square bar takes ≈ 2× longer, ≈ 680 s (lumped V/A estimate, derived). | 44, 45 | Replica of the loss lines (sim.py:8693–8694) without conduction to air: a 5 cm iron voxel goes 800 → 500 °C in at most 2.6 s (2.8 s at 10 cm). RAD alone gives ≈ 20 s, so LEAK dominates (derived). | wrong (≥ 130×) |
| H22 | A hot drink cools | Coffee 85.6 °C → 62.2 °C at 13 min, 57.2 °C at 18 min, 40 °C at 48 min (room 21.7 °C; cup size and material not stated; a real open cup also loses heat by evaporation). 99 g water in a cup: τ = 1039 s (a one-run teaching note). | 46, 82 | Replica: a 5 cm water voxel cools 90 → 60 °C in at most 3.5 s | wrong (≈ 250×) |
| H23 | Heating a bar in a forge | Hot forging 950–1250 °C. ASM guideline: no more than 5 min per inch of section, so ≈ 10 min for 5 cm. | 47, 48 | The `forge` scene (scenes.py:340–354) runs 700 ticks = 17.5 s. Its bar reaches 258 °C in 5 s (too fast), then stays flat for 12 s (LEAK plateau). Iron stalls near 256 °C (roadmap 90). 17.5 s is too short to expect forging heat; the plateau is the defect. | wrong |
| H24 | Wood as an insulator | Standard fire (E119): ignition ≈ 2 min, char 0.8 mm/min for 8 min then 0.6 mm/min, char base 300 °C, ≈ 180 °C 6 mm deeper, heat reaches only 31–35 mm. | 49 | Effective diffusivity 7.5e-4 m²/s at 5 cm: heat reaches ≈ 21 cm into wood in 60 s. With LEAK the profile settles to a 6.8 cm decay length. A point 15 cm below an 800 °C top sits ≈ 86 K above ambient (derived). | wrong |
| H25 | Plume above a fire | FM Global: ΔT0 = 0.09325·T∞·Qc^(2/3)·(z−z0)^(−5/3) K, u0 = 1.179·Qc^(1/3)·(z−z0)^(−1/3) m/s, with Qc the **convective** HRR. Flame zone ≈ 3.5·T∞ ≈ 770 °C. Qc 20 kW: ≈ 205 K and 3.2 m/s at 1 m; ≈ 64 K and 2.5 m/s at 2 m. | 50 | `_law_rise` moves 0.35·c_airbase·ΔT up per tick (sim.py:4314); the open top halves E (sim.py:4319); `_law_air` moves smoke, not heat (sim.py:4650). Derived: bulk heat rises 0.7 m/s at 5 cm, 1.4 m/s at 10 cm, and carries 35,000–70,000 W/(m²·K) per kelvin against ≈ 3,600 for a real 3 m/s plume. | wrong (plume ≈ 10× too cool for its HRR; grid dependent) |
| H26 | Hot layer and flashover | Flashover at an upper layer of 500–600 °C with ≈ 20 kW/m² at the floor, 3–5 min after a small flame. NIST TN1600 room flashed over at ≈ 2.5 MW. | 51, 39 | `house_fire` (scenes.py:770) runs 900 ticks = 22.5 s. Smoke by height band is flat: 305 / 356 / 231 / 286 g (roadmap 90). Note: "skin 20 °C at 21 s" is not evidence. A real occupant is also unhurt 21 s after a lamp lights a bed. | wrong (no hot layer can form; see H12, H14) |
| H27 | Quenching | One oil at 60 °C, unagitated, standard 12.5 × 60 mm Inconel probe from 850 °C: 600 °C at 5.4–8.2 s, 400 °C at 8.4–12.6 s, 200 °C at 35.6–55.8 s, peak 86–103 °C/s. Water: > 150 °C/s. Severity H: agitated water 0.9–1.2, standard oil 0.3–0.4. A 50 mm bar in 14% brine from 850 °C: core 472.5 °C at 30 s, 287.7 °C at 50 s, 181 °C at 75 s. | 52, 53, 54, 142 | Contact k water 30, oil 1.5 (sim.py:228–229): water is 20× oil. Replica: a 5 cm iron voxel with one wet face has τ 0.74 s (2.97 s at 10 cm); fully immersed 0.12 s (0.49 s at 10 cm). No vapour-blanket stage. | wrong (≈ 100× too fast; grid dependent) |
| H28 | Flame temperature cap | Flame zone ≈ 900 °C for fires under ≈ 1 m, 1100–1200 °C for large pools; post-flashover rooms 900–1000 °C, rarely above 1200 °C. Charcoal burns above 1100 °C, up to ≈ 1260 °C with a blower. Forge welding needs 1400–1500 °C for pure iron (900–1100 °C for high-carbon steel). | 67, 50, 117, 83 | `COMBUST_CEIL` 1100 (sim.py:773); burning slows by (1100 − T)/1100 (sim.py:4338). Forge bed measured ≈ 880 °C. No forced draft. | ok for game (bellows missing) |

Skin pain, burns, body heat and hot-air injury are in 2.7 (Body). Boiling, freezing and
evaporation are in 2.4 (Fluids).

### 2.3 Combustion and fire (F)

| # | Quantity | Real value | Src | Sim value | Verdict |
|---|---|---|---|---|---|
| F1 | Piloted ignition of wood | Surface 300–365 °C at fluxes high enough for direct flaming (hardwoods 300–311 °C, softwoods 349–364 °C, oven-dry) | 40 | `SOLID[WOOD]` ignition 300 °C (sim.py:212), tested on the whole-voxel temperature (sim.py:4420). No pilot flame needed. | matches (but a voxel average, not a surface) |
| F2 | Slow-heating ignition | ≈ 250 °C after long, barely sufficient heating, piloted or not; glowing comes first. Knowledge of ignition by contact with hot bodies is poor. | 40 | Fixed 300 °C and no pyrolysis. The lamp shelf held at 240 °C chars "forever" and never ignites (roadmap "Known lies"; `lamp_shelf` scenes.py:111). | ok for game (a beam beside a hearth lights ≈ 50 °C late) |
| F3 | Fast radiant auto-ignition | 380–600 °C (≈ 530 °C at 40 kW/m²; ≈ 600 °C radiant, transient) | 40, 55 | No pilot / auto distinction (`_gas_access` sim.py:4321) | ok for game (early only under fast radiant heat with no flame near) |
| F4 | Time to ignite under radiant flux | t_ig = 130·ρ^0.73/(q − 11.0)^1.82 s (q in kW/m², RMS error 64%). For ρ 600: ≈ 114 s at 25 kW/m², 43 s at 35, 18 s at 50. Design minimum flux 12.5 kW/m²; minimum for long exposures 4.3 kW/m² (one plywood study, needs checking). | 40 | Radiation couples only neighbours (sim.py:4289); wood conducts at 800 W/(m·K) | unknown (needs line-of-sight radiation first) |
| F5 | Char rate of wood | 0.8 mm/min for 8 min, then 0.6 mm/min | 55, 49 | O2-capped surface voxel: 3·scale² g O2 per tick (sim.py:4405) ÷ 1.33 = 0.564 g/tick. With its char phase (sim.py:4431–4433) a 75 g, 5 cm voxel is gone in ≈ 150 ticks = 3.7 s (derived). | wrong (≈ 1000× fast) |
| F6 | Mass loss and HRR per area | 2.9–9.6 g/(m²·s) at 18–55 kW/m² imposed flux (Table 18-3). Peak cone HRR 87–280 kW/m². Flames die below a critical mass loss of ≈ 4 g/(m²·s). | 49, 56 | ≈ 9 kg/(m²·s) through one 25 cm² face, ≈ 144 MW/m² at 16 kJ/g. The kinetic term 0.03·scale²·(T−300)·damp peaks at 1.09 g/tick near 700 °C (sim.py:4422). (derived) | wrong (≈ 900–3000×); no extinction limit |
| F7 | Heat of combustion of wood | Gross ≈ 20 MJ/kg oven-dry; effective ≈ 65% of gross; two-pallet fire 502 MJ / 30.37 kg = 16.5 MJ/kg | 55, 56, 57 | `SOLID[WOOD]` 16000 J/g; the last 18% burns as CHAR at 30000 J/g (sim.py:212, 219, 4431). Overall ≈ 17.8 kJ/g of initial mass (derived). | matches |
| F8 | Heat per gram of O2 | 13.1 kJ/g O2 (within 5% for organic solids). Charcoal ≈ 11.4 (29.6 MJ/kg over ≈ 2.6 g O2/g); pure carbon 12.3 by stoichiometry. | 55, 61 | heat / `O2_RATIO`: WOOD 12.0, LEAF 12.0, CHAR 11.5, OIL 13.3, FLESH 5.7 kJ/g O2 (sim.py:212–219, 898, 905) | matches, except FLESH (wrong: set `O2_RATIO[FLESH]` ≈ 0.6 g/g, not a higher heat) |
| F9 | Oxygen demand | Wood CH1.7O0.74 → 1.32 g O2/g; carbon 2.67 g/g; triolein ≈ 2.9 g/g | 58 | `O2_RATIO` WOOD 1.33, CHAR 2.6 (sim.py:898); `O2_RATIO_OIL` 3.0 (sim.py:905) | matches |
| F10 | Oxygen in fresh air | 21 vol% = 23.14 wt% → 0.278 g/L | 59 | `O2_PER_L` 0.28 (sim.py:897) | matches |
| F11 | Oxygen limit for flaming | Flaming above ≈ 15% O2; smouldering only at 4–15%; pyrolysis only below 4% (particleboard; limits fall as flux rises). Self-sustained smoulder needs ≈ 10%. | 60 | No threshold. Burn rate is linear in local O2 (sim.py:4384, 4405). The test `test_sealed_room_fire_suffocates_with_fuel_left` (tests/test_voxel.py:181–193, voxel_cm 10) only proves the fire takes room air below ≈ 24% of fresh (≈ 5 vol% O2). The true end point is unmeasured. | wrong |
| F12 | Charcoal | HHV 29.6 / LHV 28.4 MJ/kg; autoignition 349 °C | 61, 62 | `SOLID[CHAR]` 330 °C, 30000 J/g (sim.py:219) | matches |
| F13 | Lamp oil | Vegetable oil: flash point 327 °C (canola), fire ("flame") point ≈ 379 °C, autoignition 398–435 °C (averages 406 soybean, 424 canola, 435 olive). Kerosene: flash point 38–72 °C, autoignition 210 °C. HHV: vegetable oil 40.5, kerosene 46.2 MJ/kg. | 63, 64, 65, 61 | `FLUID[OIL]` ignition 250 °C, 40000 J/g (sim.py:229) | heat matches; ignition wrong (80–130 °C low for vegetable oil, far too high for kerosene). Decide which oil "lamp oil" is. |
| F14 | Oil pool burning rate | Kerosene, large pool: 0.039 kg/(m²·s), kβ 3.5 m⁻¹; a 0.3 m pool ≈ 0.025 (derived). Spills 1 mm deep average about half the rate of 20 mm pools. Unconfined spills burned at ≈ 20% of pool rates (Gottuk 2000, cited). | 66 | O2-capped oil voxel: 0.25 g/tick over 25 cm² = 4.0 kg/(m²·s) (sim.py:229, 4441–4451) | wrong (≈ 100–500×) |
| F15 | HRR to flashover | Q_fo ≈ 750·A_o·√H_o kW. For the `house_fire` door (0.5 × 1.85 m): ≈ 0.94 MW (derived). | 69, 51, 68 | Flashover cannot happen (H12, H14–H16) | missing |
| F16 | Time to flashover | Natural-material ("legacy") 3.7 × 3.7 × 2.4 m rooms: 29:30 and 34:15 min; the 4.0 × 5.5 m legacy room never flashed over. Modern synthetic rooms: 3:20–4:45 min. Their openings (2.4 × 2.1 and 3.0 × 2.1 m) and fuel loads (11–23 kg/m²) are much larger than `house_fire`'s. | 70 | `house_fire`: room 3.9 × 2.9 × 2.15 m. Its smoke implies at least ≈ 1.1–1.7 MW from wood over 22.5 s, plus up to ≈ 0.5 MW from oil (derived; the roadmap's height bands sum to 1178 g, its text says 1690 g). | wrong (≈ 100× fast) |
| F17 | Furniture and crib HRR | Two pallets + 1 kg paper, four wick ignitions: 934 ± 42 kW at 3.93 min, out at 39.52 min, 30.37 of 33.00 kg lost, 502 MJ, 16.53 MJ/kg. Cotton chair 370 kW at 910 s; PU chair 1990 kW at 260 s. | 57, 70 | No scene logs HRR | wrong (growth 10–100× fast, derived); an HRR readout is missing |
| F18 | Burning tree | Dry 2.1 m Douglas fir: 7362 kW 31 s after a match; the watered tree did not sustain ignition in 3 tries. Dry 3.4–5.8 m trees: 7.5–42 MW within 7–10 s, 35–71% of mass left, shallow trunk char, χr 0.30 ± 0.03. Under 30% moisture they burned well, over 70% not at all. | 71, 72, 73 | `burning_tree` (scenes.py:267): LEAF at 40% fill with sap ≈ 75% moisture, dry basis (derived; scenes.py:75–77). `SOLID[LEAF]` 240 °C, 15000 J/g (sim.py:218). Trunk 20 × 20 cm solid wood; the docstring expects it may burn through. | ok for game (moisture mechanism right). LEAF heat is 0–29% low depending on convention. Trunk burn-through in minutes is the wrong outcome (indirect evidence). |
| F19 | Smoke opacity | Flaming soot: 8.7 ± 1.1 m²/g. Solid wood soot yield ≈ 0.015 g/g (Tewarson f_p 0.031) → ≈ 0.13 m²/g of wood burned. Dry-tree crown fires only 0.003–0.005 g/g. Visibility S = C/K. | 74, 58, 72 | `SOOT[WOOD]` 0.6 (sim.py:902) × `SMOKE_EXT` 5.6 (sim.py:825) → 0.0034 m²/g of wood | wrong (≈ 39× too clear for furniture and beams; 11× for tree fires) |
| F20 | Smoke layer in a room | Hot smoke pools under the ceiling. Zukoski model for the `house_fire` room with 100 kW: lower edge at 1.5 m after 7–10 s and at 1.0 m after 17–23 s (derived; the model deviates 36–43% from experiments). | 75 | Flat profile 305 / 356 / 231 / 286 g (roadmap 90). Likely causes (unmeasured): the room never warms (H12, H14, H15); the cool zone is stirred over 12 coarse cells, the whole room height (hot-zone test sim.py:4912); the 0.2 g smoke threshold is per voxel, not scaled by `vox_l`. | wrong |
| F21 | Water on burning oil | Water sinks, flashes to steam (≈ 1700× volume) and throws out burning oil ("slopover"). A 1 kg oil + 1 L water demo sequence lasts 2.4 s. A separate chip-pan demo sends flame to the ceiling. | 77, 78, 141 | Water sinks under oil (sim.py:1280–1289). Boiling only adds `p_add += boil_g * 0.1` (sim.py:4491). No steam species. `test_water_fails_against_oil_fire` only checks that the oil keeps burning. | wrong (outcome right, flare-up and thrown oil missing) |
| F22 | Water: latent heat and cooling | 2256.4 kJ/kg; 20 °C water to steam absorbs ≈ 2.59 MJ/kg | 76 | `LATENT` 2260 (sim.py:772), water c 4.2 (sim.py:228) | matches |
| F23 | Moisture and ignition | Moisture raises ignition temperature only ≈ 2 °C per %; the energy to drive the water off is the big effect | 40, 72 | Water held in a voxel pins it near 100 °C until boiled off (`_law_boil` sim.py:4479) | matches |
| F24 | Igniters | One candle ≈ 80 W | 68 | Scene "wick", "taper", "torch": 2500 J/tick = 100 kW (scenes.py:794, 129, 280). A "match" in the tests is 800 J/tick = 32 kW. | wrong (≈ 1000× a candle; rescale from real HRR when the laws move to SI) |
| F25 | Flame height | Heskestad L = 0.235·Q^(2/5) − 1.02·D (m, kW). 100 kW over a 0.3 × 0.3 m pool (D_eff 0.34 m): ≈ 1.14 m (derived). | 79 | Combustion happens only inside fuel voxels. Flames have zero height (roadmap "Known lies"). | missing |
| F26 | Time until a home is untenable | NIST home tests, at 1.5 m outside the room of origin: 217 ± 67 s for flaming fires, 4316 ± 1256 s smouldering, 1477 ± 249 s cooking. "Untenable" was set for sensitive people and was mostly reached by smoke obscuration in modern homes. | 80 | `house_fire` was never run past 22.5 s | unknown |

**There is no single fire clock.** Each law is fast by a different factor. The "real
seconds per tick" that would make each per-tick constant correct are (derived):
wood burning 23–75 s; char front 28–38 s; oil pool 2.6–12 s; conduction in wood
120–170 s, in air ≈ 230 s, in stone 7.5–18 s, in iron 3.4–4.3 s; neighbour radiation
≈ 1.3 s. A single global rescale therefore cannot fix fire. Each law needs its own
conversion to SI.

### 2.4 Fluids, phase change and molten metal (L)

Heat loss (`LEAK`) and the cup cooling time are in 2.2 (H14, H22). Blood volume and
bleeding are in 2.7.

| # | Quantity | Real value | Src | Sim value | Verdict |
|---|---|---|---|---|---|
| L1 | Melting points and latent heats | Sn 231.93 °C, 59.2 J/g; Pb 327.46 °C, 23.0 J/g (NEA 23.8 ± 0.7); Fe 1538 °C, 247 J/g. Grey cast iron melts lower (≈ 1150–1200 °C, unverified); its fusion heat is 96 kJ/kg. | 84, 85, 86, 87, 88 | `MELT` (sim.py:789–790); `FREEZE` is the same table with 2 K hysteresis (sim.py:791, 4522) | matches |
| L2 | Solid metal density and c | White tin 7.289 g/cm³, 0.228 J/(g·K); lead 11.348, 0.129; iron 7.874, 0.449 | 84, 85, 86 | `SOLID` TIN (7300, 0.23) sim.py:220; LEAD (11340, 0.13) sim.py:221; IRON (7800, 0.45) sim.py:214 | matches |
| L3 | Molten tin | ρ 6979 kg/m³ at Tm; η 1.87 mPa·s; σ ≈ 0.56 N/m | 89 | `FLUID[MTIN]` 6.98 g/ml, c 0.24, fluidity 0.6, k 15 (sim.py:232) | matches (no surface tension) |
| L4 | Molten lead | ρ = 11367 − 1.1944·T → 10.65 g/cm³ at 600.6 K; η 2.70 mPa·s (eq. 2.63); σ 0.451 N/m (eq. 2.20); k ≈ 9.2 + 0.011·T ≈ 15.8 W/(m·K) | 87, 84 | `FLUID[MLEAD]` 10.66, 0.14, 0.6, k 8 (sim.py:233) | matches |
| L5 | Molten iron | η 4.7–6.2 mPa·s at Tm (5.85 at 1811 K); σ ≈ 1.88 N/m; ρ 6.98 g/cm³ | 90, 86 | `FLUID[MIRON]` 7.0, c 0.82, 0.5, k 20 (sim.py:234). Liquid c not separately checked. | matches |
| L6 | Viscosity range | Water 1.0016 mPa·s at 20 °C (0.28 at 100 °C); olive oil 80.5 cP at 20 °C (33.5 at 40 °C); kerosene 1.64 cP (300 K); H2SO4 26.7 cP; molten Sn / Pb / Fe 1.9 / 2.7 / 5.9 mPa·s. Kinematic range ≈ 350× (olive oil 8.8e-5 m²/s, molten lead 2.5e-7 m²/s). | 91, 92, 93, 94 | The "viscosity" column is a fluidity: WATER 0.50, OIL 0.25, ACID 0.40, WEAK_ACID 0.48, MTIN/MLEAD 0.6, MIRON 0.5, BLOOD 0.45 (sim.py:228–235), used as diff·visc/4 per tick (sim.py:1320) | ok for game (order right; range 2.4× instead of ≈ 350×) |
| L7 | Puddle thickness and spill area | Equilibrium puddle h = 2·λc·sin(θ/2), independent of size. Water λc 2.71 mm → 1–4 mm deep, 0.25–1.0 m²/L. Measured spills: mean depth 0.72 ± 0.34 mm (0.22–2.4 mm); lube oil 0.5 L on smooth (permeable) concrete 0.57 m² at 0.88 mm; on coated concrete 0.22 m² at 2.30 mm; rule of thumb 1.4 m²/L up to 95 L. Blood pools on tile: 1.44 ± 0.19 mm. Molten lead λc 2.07 mm → splat ≈ 3–4 mm (derived). | 95, 96, 66, 115, 87 | Spreading stops when neighbours differ by 1 ml or less (sim.py:1319): a cone, not a pancake. Area ∝ V^(2/3)·cell area (derived): 1 L water → ≈ 0.71–0.76 m² at 5 cm, 2.85 m² at 10 cm; 100 ml → 0.15 m²; 10 L → 3.2 m². Every fluid, molten metal too, ends at the same area. | ok for game at 5 cm for 1–10 L; wrong in scaling (×4 per voxel-size doubling, area ∝ V^(2/3) not V, metal does not bead) |
| L8 | Spread speed | Spills grow fast for ≈ 30 s, then slowly; most reach ≥ 90% of maximum by 300 s | 66 | Front moves at most one cell per tick (sim.py:1290–1320). 90% of final area in 1.5 s (1 L water), 2.0 s (oil), 5.4 s (10 L) (derived). | ok for game (no slow creep) |
| L9 | Pouring and outflow | Free fall; Torricelli v = √(2gh), discharge coefficient ≈ 0.65 | 97 | Fluid falls one voxel per tick (sim.py:1300): 2 m/s at 5 cm, 4 m/s at 10 cm. `_law_head` moves 0.25·cap·Δh per tick between any two columns of a body, with no pipe throttle (sim.py:1271–1272). | ok for game (a 1 m pour takes 0.5 s vs 0.45 s). No jets, waves or sloshing. |
| L10 | Splash droplets | g = 9.81 m/s² | 97 | `_fly_drops`: d[5] −= 0.25 voxel/tick² (sim.py:1186) → 20 m/s² at 5 cm, 40 m/s² at 10 cm | wrong |
| L11 | Boiling flux ceiling | Critical heat flux ≈ 1000 kW/m² for water at 1 atm; nucleate boiling at 10–30 °C superheat. Leidenfrost point ≈ 193 °C: a drop vanished almost at once at 168 °C but lasted 152 s at 202 °C. | 99, 100, 101 | Contact k 30 → ≈ 600 W/K per 5 cm face: ≈ 7 MW/m² at 30 K superheat, 24 MW/m² at 100 K, no ceiling, no Leidenfrost reversal | wrong |
| L12 | Steam | Saturated steam 1.67 m³/kg (≈ 1670× the liquid). Water trapped under molten metal can explode. Molten lead dripped into water is a routine shot-making method. | 102, 103, 143 | Boiled grams are deleted; `p_add += 0.1 * g` (sim.py:4491). No vapour species. | missing |
| L13 | Heat inside a liquid | Water k 0.606 plus convection; molten lead ≈ 15.8 W/(m·K) | 28, 87 | Two fluid cells are both `mat == AIR`, so they conduct with air's k plus the hot-air bonus (sim.py:4268–4276; numba path sim.py:61–110). Contact k applies only to fluid on a solid, and "wet" needs fvol > 0.1·cap (sim.py:4274): 5 mm deep at 5 cm, 10 mm at 10 cm. Molten metal and water exchange heat only through air k. | wrong (a kettle boils its bottom cell; thin films never count as wet; grid dependent) |
| L14 | Only water boils | Blood, vinegar and diluted acid are mostly water (derived) | — | `_law_boil` acts only on WATER (sim.py:4485). Water + ACID becomes ACID (sim.py:265), so diluted acid can never boil. | wrong |
| L15 | Evaporation below boiling | g = Θ·A·(xs − x), Θ = 25 + 19v kg/(m²·h). Water at 20 °C, air at 0.5 m/s: 0.169 kg/(m²·h), so a 1 mm film dries in ≈ 6 h. | 104 | None (sim.py:4485) | missing (fine for minute-long scenes) |
| L16 | Freezing | Fusion 334 kJ/kg. Stefan ice growth h ≤ 0.035·√S m (S in °C·days); real ice ≈ 0.7 of that. | 88, 105 | No ICE. `E = max(E, −AMBIENT·C)`, "nothing below 0 °C here" (sim.py:8695). | missing |
| L17 | Solidification | Chvorinov t = B·(V/A)^n, n 1.5–2. Al alloy in sand: B = 45 min/in² (≈ 7.0 min/cm²). A separate depth-fit example gives 31.25 min for a 4 in bar ("real castings take somewhat longer"). | 106, 107 | From LEAK alone a 5 cm MLEAD voxel at 400 °C sets in ≈ 5 s (≈ 1.4 s to 325 °C plus ≈ 3.3 s latent; derived). A 3.5 mm lead splat on stone should take ≈ 6 s, a 5 cm block many minutes (semi-infinite contact estimate, derived). | ok for a splat; wrong for a block |
| L18 | Tin + lead | Fully miscible liquids; 63/37 melts at 183 °C, 60/40 at 188 °C | 108 | `MISCIBLE` has only water + acid and water + weak acid (sim.py:265–266) | missing |
| L19 | Water into wood | Absorption coefficient, softwoods: 10–16 g/(m²·s^0.5) along the grain, 1–7 across. Maximum moisture 267% (SG 0.30) to 44% (SG 0.90). | 109 | Not modelled. `_law_flow` skips fluid held in solids (sim.py:1297); `pour()` refuses solid cells (sim.py:1131–1141). | missing (fine for spills; needed for soaked torches and wet kindling) |
| L20 | Cloth | A cotton towel holds 2–3× its weight (5–6× for very absorbent ones); the fibre itself ≈ 25% (manufacturer's article) | 110 | No cloth material (`NMAT` 11, sim.py:207) | missing |
| L21 | Blood viscosity | 3.5–5.5 cP at high shear, ≈ 60 cP at 0.1 /s (strongly shear-thinning); plasma 1.2–1.3 cP | 111 | BLOOD fluidity 0.45 vs water 0.50 (sim.py:235) | ok for game (pools should stall sooner) |
| L22 | Blood density and c | Density ≈ 1060 kg/m³. c ≈ 3617 J/(kg·K) (unverified). | 118 | `FLUID[BLOOD]` 1.06 g/ml, 3.6 J/(g·K) (sim.py:235) | matches |
| L23 | Shed blood | Deposited drops clot 2–8 min after deposition. Whole blood in glass at 37 °C (Lee-White) clots within 20 min. A ≤ 9 ml pool on tile at 22.5 °C was still drying 8 h 37 min after it formed. | 115, 114 | Blood stays a free liquid for ever: no gel, no drying, not miscible with water (roadmap 96) | missing |
| L24 | Acid as a liquid | 98% H2SO4: ρ 1.84 g/ml, c ≈ 1.38 J/(g·K) (from 2.54 kJ/(L·K)), η 26.7 cP, boils at 337 °C. Below 29% H2SO4 the density is 1.00–1.25. | 94, 121 | `FLUID[ACID]` 1.8, 1.4, fluidity 0.40 (sim.py:230). A blend keeps ACID's ρ and c whatever its potency (`heat_capacity` sim.py:1052–1054), so each ml of water that joins the acid gains 0.8 g. | ok for pure acid; wrong for blends (mass not conserved) |

### 2.5 Chemistry: acids and reactions (C)

The code calls `ACID` "vitriol" (concentrated sulfuric acid). Rates in the sim are per
cm² of contact face and do not depend on voxel size. The removal times assume fresh
acid at potency 1; one 5 cm acid voxel (125 ml) can dissolve at most 312 g of flesh,
250 g of iron or 208 g of stone (sim.py:4570–4576).

| # | Quantity | Real value | Src | Sim value | Verdict |
|---|---|---|---|---|---|
| C1 | Strong acid on skin | GHS Skin Corr. 1A (H2SO4 ≥ 15%): visible necrosis through the epidermis into the dermis after ≤ 3 min of exposure. H2SO4 (UN1830) and 34–37% HCl (UN1789) are DOT Packing Group II: full-thickness destruction after > 3 min and up to 60 min. Epidermis ≈ 0.09 mm typical, 0.6 mm on the sole; forearm skin ≈ 1.3 mm. Damage continues while the acid stays. Acids cause coagulation necrosis: the dead layer stays in place; tissue is not removed. | 119, 120, 121, 122, 123, 124, 125 | `REACTIONS[(ACID, FLESH)]` (6.0 g/face/tick, 0.4 ml/g, 200 J/g) (sim.py:250) → 2.4 g/(cm²·s), ≈ 24 mm/s. A 5 cm flesh voxel turns to AIR, its mass to smoke (sim.py:4577–4581), in ≈ 2–3 s. `_law_acid` (sim.py:4535) does not exclude a living body, so a foot in a vitriol puddle dissolves (read from code, not run). Rulings case 2 says acid-hurts-people stays unmodelled. | wrong (3,000–60,000× fast; removes tissue; breaks rulings case 2) |
| C2 | Vinegar on skin | Acetic acid ≤ 20% is not classified for skin. 3.3% and 10% on rabbit skin for 4 h: not irritating. 10% in a 4 h human patch test: slightly irritating. Vinegar is 4–10% acetic acid. | 126 | `REACTIONS[(WEAK_ACID, FLESH)]` (0.5, 0.8, 40) (sim.py:258) → ≈ 2 mm/s; a 5 cm flesh voxel goes in ≈ 25 s | wrong (no tissue loss expected) |
| C3 | Vitriol on wood | Some species resist dilute mineral and organic acids (hardwood and sapwood less). Wood tanks hold liquids down to pH 1.5 for decades. Concentrated H2SO4 dehydrates cellulose to black carbon (tissue paper chars "instantly"). 72% acid at 30 °C hydrolyses 0.25–0.85 mm poplar particles in ≤ 30 min and leaves 21.8–25.9% acid-insoluble lignin. | 128, 129, 94, 130 | `REACTIONS[(ACID, WOOD)]` (3.0, 0.4, 150) (sim.py:249) → ≈ 20 mm/s; a 5 cm voxel goes in 2.5 s and dissolves "clean, no ash" (sim.py:4581) into smoke (sim.py:4577). Measured: the alchemist scene ate 2369 g of wood in 22 s (roadmap 94). | wrong (rate 10⁴–10⁵× fast, an unverified inference; residue should be CHAR) |
| C4 | Vinegar on wood and leaves | No structural effect on any game time scale | 128, 129 | `(WEAK_ACID, WOOD)` 0.3 (sim.py:257): a cask wall goes in ≈ 25 s. `(WEAK_ACID, LEAF)` 0.8 (sim.py:256): ≈ 16 mm/s. | wrong |
| C5 | Acid on iron | Mild steel, 1 M HCl, 30 °C, no inhibitor: ≈ 0.7–5.7 mg/(cm²·h) (≈ 8–64 mm/yr; two studies disagree). Carbon steel in 60–100% H2SO4 below 100 °F: < 0.02 in/yr (< 0.5 mm/yr), but 78–93% acid is still a "severe concern" and steel is unsuitable for 60–93% above 100 °F. Below ≈ 60%: "extremely corrosive". DOT counts > 6.25 mm/yr at 55 °C as corrosive to metal. | 131, 132, 133, 120 | `REACTIONS[(ACID, IRON)]` (0.3, 0.5, 250) (sim.py:253) → 432,000 mg/(cm²·h), 0.154 mm/s; linear in potency (sim.py:4566–4569). Removing a 975 g iron voxel needs ≈ 0.5 L or more of potent acid. | wrong (75,000–600,000× vs 1 M HCl; the concentration trend is inverted) |
| C6 | What dissolving iron makes | Fe + 2H⁺ → Fe²⁺ + H₂: ΔH ≈ −89.1 kJ/mol ≈ 1.6 kJ/g Fe. H₂ 0.018 mol/g (≈ 0.40 L/g at 0 °C), flammable. Acid used ≈ 0.97 ml/g of 98% H2SO4 or ≈ 3.0 ml/g of 37% HCl. The iron stays in the liquid as a salt. | 134, 94, 122 | Heat 250 J/g, spend 0.5 ml/g (sim.py:253). The dissolved mass becomes smoke (sim.py:4577). No H₂. | wrong (heat 6× low, spend 2–6× low, wrong products) |
| C7 | Limestone in strong acid | Marble disk in 1 N HCl, 25 °C, 500 rpm, 800 psig (CO2 kept dissolved): 9.4e-6 mol HCl/(cm²·s) ≈ 0.47 mg CaCO3/(cm²·s) ≈ 1.7 µm/s. A drop of 10% HCl on limestone fizzes for a few seconds. | 135, 127 | `REACTIONS[(ACID, STONE)]` (0.02, 0.6, 30) (sim.py:252) → 8 mg/(cm²·s), ≈ 0.03 mm/s; a 5 cm voxel in ≈ 28 min. No CO2; the stone goes to smoke. | ok for game for HCl-class acid (17× the stirred 1 M rate); products wrong |
| C8 | Granite vs limestone | Granite's minerals do not dissolve in weak acids. Vinegar or lemon juice etches polished marble in under 60 s (trade source). Granite in strong acid: no source opened (unverified). | 136, 127 | One STONE material (sim.py:205, 213). No `(WEAK_ACID, STONE)` row. | missing |
| C9 | Gypsum armour | Gypsum coats calcite in sulfate acid and passivates it. Limestone columns fed pH 2 Fe(III)–SO4 solution became inert within weeks. | 137 | A steady rate × potency | missing (matters because ACID is vitriol) |
| C10 | Acid used by limestone | 1 g CaCO3 uses 0.54 ml of 98% H2SO4 or ≈ 1.67 ml of 37% HCl. ΔH ≈ 152 J/g with dissolved Ca²⁺, ≈ 141 J/g when solid gypsum forms. CO2 0.44 g/g. | 94, 134 | Spend 0.6 ml/g, heat 30 J/g (sim.py:252) | matches for spend; products wrong |
| C11 | Heat of diluting vitriol | Integral heat of mixing (Felder, Rousseau & Bullard Table B.11): −48.99 kJ/mol H2SO4 at 3 mol H2O per mol, −73.97 at 100, −96.19 at infinite dilution. A 1:1 volume mix (≈ 3.1 mol/mol) releases ≈ 0.86 kJ per ml of 98% acid; full dilution ≈ 1.77 kJ/ml (derived). Water poured onto acid forms a hot top layer that can boil and spray acid. | 139, 94, 134, 138 | `MISCIBLE` {WATER + ACID → ACID} (sim.py:265–266). Potency blends by volume at once (pour sim.py:1139, `_move_fluid` sim.py:1166). No heat. | missing |
| C12 | How rate depends on concentration | Thresholds, not proportions: H2SO4 is Skin Corr. 1A from 15%; acetic acid ≤ 20% is unclassified; steel is slow in > 60% H2SO4 below 38 °C and fast in dilute acid. Calcite in HCl is mass-transfer limited, so roughly proportional (an inference). | 121, 126, 133, 135 | Every rate is linear in potency (sim.py:4566–4569). Potency mixes by volume of 98% stock, so pot 0.5 ≈ 64 wt% and pot 0.2 ≈ 31 wt%: w = 0.983·1.84p / (1.84p + 0.998·(1 − p)) (derived). | wrong for skin and iron |
| C13 | What `WEAK_ACID` is | Vitriol diluted 10× by volume ≈ 17 wt% H2SO4: still Skin Corr. 1A and "extremely corrosive" to steel. Vinegar (4–10% acetic): no skin damage, stored in wooden casks, etches carbonates only. | 94, 121, 126 | Commented "vinegar-class" (sim.py:231) and "tenfold-diluted stock" (sim.py:247). Two opposite substances. | wrong (incoherent) |
| C14 | Vitriol on charcoal | Concentrated H2SO4 makes carbon from carbohydrates. Only hot concentrated acid oxidises carbon (C + 2H2SO4 → CO2 + 2SO2 + 2H2O). | 94 | `REACTIONS[(ACID, CHAR)]` (6.0, 0.5, 60) (sim.py:255): ≈ 80 mm/s at any temperature | wrong (backwards) |
| C15 | Glass and lead vessels | Sulfuric acid is stored in glass. Lead resists it (Wikipedia, tagged "citation needed"). | 94 | No `(ACID, GLASS)` or `(ACID, LEAD)` row | matches |
| C16 | Vitriol fumes | H2SO4 vapour pressure < 0.001 mmHg at 25 °C. Visible gas from acid work is steam, CO2 (carbonates), H2 (metals in dilute acid) or SO2 (hot concentrated acid). | 94 | The "fumes throughout" of the alchemist scene (roadmap 94) come from the smoke ledger | wrong |
| C17 | Rust in air | ISO 9223 first-year carbon steel loss: C1 ≤ 1.3, C2 1.3–25, C3 25–50, C4 50–80, C5 80–200, CX 200–700 µm/yr | 140 | IRON has no reaction with water or O2 | missing (irrelevant until the world runs for days) |
| C18 | Vinegar on iron | Acetic acid corrodes mild steel. No verified rate for 5% vinegar. | — | No `(WEAK_ACID, IRON)` row | missing (unverified) |
| C19 | Acid on wood ash | Wood ash is alkaline (it should neutralise acid and give CO2). No opened source here (unverified). | — | `(ACID, ASH)` (8.0, 0.5, 50) sim.py:251 and `(WEAK_ACID, ASH)` sim.py:259 dissolve ash into smoke | unknown |

### 2.6 Material strength and fracture (S)

The MPM solver runs with softened stiffness: `stiff` 1e-4 (core.py:151, matter.py:43).
Strength is scaled by √stiff (core.py:379, matter.py:111). The yield strain is not
scaled (core.py:585–592). The flaw factor spans 0.45–2.0 only at single extreme grid
nodes; 98% of particles fall in 0.89–1.53 (derived from `_flaw_field`, core.py:307–334).

| # | Quantity | Real value | Src | Sim value | Verdict |
|---|---|---|---|---|---|
| S1 | Wood stiffness along the grain | 8.5–13.4 GPa (Table 5-3; raise 10% for E_L: 9.4–14.7 GPa). 3800 m/s is a worked example (E 12.4 GPa, ρ 480), not a general value. | 10 | `MAT['wood']` E 1.1e10 (core.py:75). With stiff 1e-4 the P-wave speed is ≈ 54 m/s instead of ≈ 5400. | matches (table); softening is S20 |
| S2 | Wood across the grain | E_T/E_L ≈ 0.03–0.09, E_R/E_L ≈ 0.05–0.20 (balsa excluded); transverse can be 1/20 of longitudinal | 10 | Isotropic, ν 0.35 (core.py:75); the lattice WOOD row has no grain | missing (cannot both split and snap) |
| S3 | Wood strength along the grain | MOR 59–99 MPa at 12% MC (green 34–57). Tension parallel 45–121 MPa green (+13% softwood, +32% hardwood when dry). Compression parallel 33–50 MPa. | 10 | `MAT['wood']` strength 1.0e7 Pa (core.py:75) × √stiff = 1e5 Pa × flaw → ≈ 89–153 kPa for 98% of particles. Impact parting speed ≈ 3.1 m/s nominal, 2.7–4.7 m/s for 98% of particles (core.py:519–522; derived). | wrong for static loads (impact speed kept, static margin cut 100×) |
| S4 | Wood across the grain, strength | Tension perpendicular 2.1–5.5 MPa; compression perpendicular 3–7 MPa; mode I toughness 0.22–0.55 MPa·m^½ | 10 | One isotropic strength | missing |
| S5 | Wood impact bending | Handbook "impact bending": a 22.7 kg hammer dropped from rising heights until rupture or 152 mm deflection. Dry: 0.46 m (eastern white pine), 0.79 m (coast Douglas-fir), 1.09 m (northern red oak). A comparative value, not a one-blow energy. | 10 | `TOUGH[WOOD]` 8 kJ/m² (sim.py:796) × one face × flaw 0.7–1.3 (sim.py:1654–1655, 3572) → 20 J (14–26 J) per 5 cm cell; an axe edge (2e-4 m²) → 1.6 J | ok for game (a 5 cm post survives a punch; thin boards need a fill-aware threshold) |
| S6 | Wood damping | Log decrement 0.02–0.1 (damping ratio 0.003–0.016) | 10 | `MAT['wood']` damp 1.0 (core.py:75; η core.py:418). Bending modes ζ ≈ 0.001–0.002 (under-damped), long modes over-damped; for bound clusters (rig 1.0) `_hold_shape` sets the damping | ok for game |
| S7 | Soda-lime glass | ρ 2500 kg/m³, E 70 GPa, ν 0.2, characteristic bending strength 45 MPa (EN 572-1); container glass 72 GPa, 2.52 g/cm³; K_Ic 0.7–0.8 MPa·m^½ | 144, 145, 146, 147, 19 | `MAT['glass']` (2500, 7e10, 0.22, strength 5e7) (core.py:78); `SOLID[GLASS]` 2500 (sim.py:217). `TOUGH[GLASS]` 0.2 kJ/m² is a contact energy, not a fracture energy. | matches |
| S8 | Glass bottle drop | A **filled** 250 ml bottle (wall 1.6–3.2 mm) breaks at 45–56 cm dropped flat onto concrete, and at 30–43 cm when its side hits the edge of an angle iron (the inventor's own test, in a patent). An impact-resistant lens must survive a 16 g steel ball from 1.27 m (≈ 0.2 J). | 17, 18 | Lattice: a full 5 cm glass cell (312 g) breaks when m·g·h > 0.5 J × flaw 0.7–1.3 (sim.py:1647, 1654), so from 0.11–0.21 m onto any solid floor (0.06–0.11 m at 10 cm); the floor's hardness is not used. A liquid layer cushions the fall (fallh × 0.4 per liquid cell, sim.py:1603–1604), so a pond drop survives. MPM: median ≈ 0.93 m (0.51–1.50 m), lower bounds (derived). At a true thin-wall fill both paths scale as 1/fill and become too tough. Impact energy counts only the cell's own mass, not the contents (sim.py:1647; matter.py:122). | wrong (depends on path, voxel size and fill) |
| S9 | Glass pane, human impact | EN 12600 swings a 50 kg impactor from 190 / 450 / 1200 mm (≈ 93 / 221 / 589 J); annealed float glass breaks in pattern A | 148, 149 | Any blow over 0.5 J breaks a 5 cm glass cell (sim.py:3572). A 4 mm pane is drawn as a 5 cm cell. | ok for game (windows break right; glass blocks too fragile) |
| S10 | Stone | Granite 2620 kg/m³, 36 GPa, tensile 8.4 MPa; basalt 15.6; marble 6.3; sandstone 4.4. UCS: granite 182 (49–324), limestone 121, sandstone 90 MPa. ASTM minimum MOR: 2.8 MPa (low-density limestone) to 10.3 MPa (granite). | 150, 151, 152, 11 | `MAT['stone']` (2700, 5e10, 0.25, strength 8e6) (core.py:76) | matches |
| S11 | Stone under impact | Dynamic split tensile strength 16.7–48.5 MPa (granite) and 18.9–79.8 MPa (basalt) at 0.4–27 /s. A flat 1 m landing makes ≈ 21–43 MPa of compression (lower on an equal stone floor). Real damage starts at corners and edges. | 150, 151 | MPM bond parting: ρ·c_P·abs(Δv) > strength·√stiff·flaw (core.py:518–521), no sign, no rate gain → bound stone loses its landing face from 1.6–4.7 cm (derived). A second law cracks particles on principal tension (core.py:621–628). Lattice: `TOUGH[STONE]` 3 kJ/m² → 1.6–2.9 m for a 5 cm cell. Plain drops stay on the lattice (`free_fall_to_seam` False, sim.py:1000); only topples use MPM. | wrong (derived; the two paths disagree 30–700×) |
| S12 | Wrought iron | UTS 234–372 MPa, yield 159–221 MPa, E 193 GPa, SG 7.5–7.9 | 153 | `MAT['iron']` E 2e11, yield strain 2e-3 (core.py:77) → 400 MPa before softening, 40 kPa after (core.py:585–592) | wrong (dents at 1/100 of the real impact speed; a bar sags under its own weight; predicted) |
| S13 | Grey cast iron | UTS 152–431 MPa, compression 572–1293 MPa, E 69–145 GPa, almost no shock resistance; K_Ic 11–19 MPa·m^½ | 154, 155 | One ductile IRON only | missing (cauldrons and pots crack, not dent) |
| S14 | Mild steel toughness | S275JR: Charpy V ≥ 27 J at 20 °C, yield ≥ 275 MPa, UTS 430–580 MPa | 156 | `TOUGH[IRON]` 150 kJ/m² (sim.py:796) → 375 J per 5 cm cell | ok for game |
| S15 | Lead | ρ 11348, E 16 GPa, ν 0.44, UTS 12–17 MPa. Yield: a few MPa (unverified). Creep at room temperature (unverified). | 84 | `MAT['lead']` yield strain 4e-4, rig 0.3 (core.py:80): softened yield 640 Pa against 5.6 kPa self-weight stress at the base of a 5 cm ingot | wrong (an ingot should slump; predicted) |
| S16 | Tin | 7.29–7.31 g/cm³, E 50 GPa, ν 0.36, yield 12–15 MPa, UTS 14–18 MPa, elongation 40–50%; yield strain ≈ 2.7e-4 (derived) | 85, 157 | `MAT['tin']` yield strain 2e-3 → 100 MPa (7–8× too strong). **The row has 6 fields** while `add_box` (core.py:361) and `Matter.promote` (matter.py:104) unpack 7. `ash` and `char` (core.py:102–103) have the same defect. Confirmed by import: `{'ash': 6, 'char': 6, 'tin': 6}`. | wrong (crash on promote) |
| S17 | Cortical bone | E ≈ 17 GPa along the bone; tension 130, compression 190 MPa (along); 50 / 130 MPa across | 158 | No bone (`NMAT` 11) | missing |
| S18 | A leg breaks in bending | Cadaver legs (mean age 57): static 241 N·m, 59.2 J; dynamic (9.48 kg at 5.55 m/s) 408 N·m with 68.6 J absorbed; dynamic 69% stronger | 160 | Blunt energy above 30 J per 5 cm cell (`bruise_kJm2` 12, sim.py:521) goes to hurt at 2500 J per unit (sim.py:522) | missing (no fracture state) |
| S19 | Muscle stiffness | AFM on intact cells 24.7 kPa (Mathur, cited); fibres 30–39 kPa relaxed, 66–97 active, 105–115 in rigor. These are transverse indentation values, not a bulk impact modulus. Bone J ≈ 31 kJ/m² is per crack area and cannot be compared with the sim's contact energies. | 161, 159 | `MAT['flesh']` E 1e6, yield strain 5e-2, rig 0.05 (core.py:79); softened 100 Pa | ok for the lattice; promoted flesh would puddle (predicted) |
| S20 | Softening | Real wave speeds 3800–5400 m/s; the first bending mode of a 2 m × 5 cm plank ≈ 23 Hz (derived) | 10 | `stiff` 1e-4: waves 43–54 m/s, plank mode ≈ 0.23 Hz. Glass test scenes use 1e-3 (mpm scenes.py:50–67). | ok for motion of bound clusters with rig ≥ 0.9; wrong as a basis for strength or yield |
| S21 | A plank under a person | δ = F·L³/(48·E·I), σ = F·L·y/(4·I). 25 cm wide plank, 80 kg at mid-span: 5 cm thick over 2 m sags 3.7–5.9 mm at 3.8 MPa; over 4 m 30–47 mm at 7.5 MPa. A 2.5 cm board over 3 m sags 101–160 mm at 22.6 MPa; over 4 m 30 MPa against MOR 34–59 MPa. Floors aim for L/360. (derived) | 162, 163, 10 | `SPAN` only: no bending and no load (sim.py:780, 1461–1496). An MPM bound cluster's deflection was never measured. | missing |
| S22 | How far a member reaches | Self-weight cantilever reach L = √(σ·h/(3·ρ·g)) for a 10 cm deep member: wood 18–24 m, glass 7.8, wrought iron 8.4, stone 2.7–4.3, tin 2.4, lead 1.3 m (derived). A 5 cm stone slab cantilevered 1 m carries 1.6 MPa; 2 m carries 6.4 MPa. | 10, 11, 144, 150, 153, 157 | `SPAN` in 10 cm cells: WOOD 12, STONE 20, IRON 30, GLASS 8, TIN 16, LEAD 10 (sim.py:780). Scaled by packing, clip(packed/0.5) (sim.py:1461–1462). Support spreads from both ends, so a clear gap of up to 2 × SPAN holds (2.4 m for wood). Vertical stacks always hold. | wrong (wood far too short; stone > wood is inverted; no load, no depth) |
| S23 | A falling trunk | The bending moment peaks at L/3 at m·g·L·sinθ/27. A 2.75 m trunk of 10 cm section peaks at ≈ 99 kPa near horizontal, 1/600–1/1000 of MOR, so a sound trunk does not break in the air (derived). Chimneys of weak masonry break at 1/3–1/2 of their height. | 164, 8 | Scaled wood strength ≈ 89–153 kPa against ≈ 99 kPa: marginal. Roadmap 94: `tree_fell` "breaks into chunks on the way down" (before shape matching; not re-run). | unknown (probably wrong; measure) |
| S24 | Felling hinge and landing | The hinge is 80% of the diameter long and 10% thick and holds the tree for most of its fall. Old-growth Douglas-fir: 3.8–13.8% of volume broken in conventional felling, 3.6–10.4% felled uphill. Small trees should break much less (extrapolation). | 165, 166 | Wood bonds part on landing above ≈ 2.7–4.7 m/s (derived); a 2.75 m trunk tip lands near 8–9 m/s | wrong (derived) |
| S25 | Breaking a board by hand | ≈ 5 J breaks a pine board, ≈ 1.6 J a concrete block; trained hands 10–14 m/s, contact ≈ 5 ms (secondary; board thickness not stated) | 170 | Fist on wood: 20 J (14–26 J) per 5 cm cell (sim.py:3549, 3572). Measured fist ≈ 3.8 m/s; the post is untouched (roadmap). | ok for game (secondary) |
| S26 | Fired clay | One white stoneware: MOR 2500 psi with a crazing glaze, 8000 psi with a fitted glaze; a vitrified cone-1 body > 10,000 psi (69 MPa). Lab clay rods fired at 400–600 °C: E 0.35–0.45 GPa (0.72–0.84 with spicules). Village earthenware is fired hotter (no figure found). | 167, 168 | No ceramic; vessels are GLASS | missing |
| S27 | Lake ice | Freshwater ice flexural strength 590 kPa (cantilever) and 780–790 kPa (3-point) on average; literature 0.5–2.2 MPa | 169 | No ICE | missing |

### 2.7 The human body: strength, movement, injury, bleeding (B)

The scene body (`scenes._person`, scenes.py:415–494) is 396 FLESH cells at 5 cm,
43.9 kg, about 1.5 m tall. Roadmap measurements made on the older 28.5 kg body are
marked "old body".

| # | Quantity | Real value | Src | Sim value | Verdict |
|---|---|---|---|---|---|
| B1 | Segment masses (share of body mass) | Hand 0.6%, forearm 1.6% (forearm + hand 2.2%), upper arm 2.8%, whole arm 5.0%; thigh 10.0%, leg 4.65%, foot 1.45%, whole leg 16.1%; head + neck 8.1%; trunk 49.7% | 171 | Each arm 5.74 kg = 13.1% (upper arm 5.8%, forearm + hand 7.3%); each leg 4.73 kg = 10.8%; torso 46.2%; head 6.2% (derived from the build) | wrong (arm 2.6×, leg 0.67×) |
| B2 | Segment lengths (H = 1.5 m) | Upper arm 0.186·H, forearm 0.146·H, hand 0.108·H; shoulder joint to joint 0.259·H = 0.39 m; hip height 0.53·H = 0.80 m | 171 | Arm 9 voxels = 0.45 m (upper 0.20, forearm 0.25); legs 0.70 m; shoulder joints 0.25 m apart (scenes.py:452–488) | ok for game (arm ≈ 30% short, shoulders ≈ 36% narrow) |
| B3 | Walking speed | Comfortable 1.10–1.65 m/s; meta-analysis group means 0.943–1.434 m/s | 172, 173 | One voxel per `WILL['walk_every']` = 3 ticks (sim.py:297; pace sim.py:7874): 0.67 m/s on a grid axis at 5 cm, ≈ 0.47 m/s on a 45° route, 1.33 m/s at 10 cm. A turn costs a whole stride (`_square_up`). | wrong |
| B4 | Running | Preferred walk-to-run change at 2.06 m/s (energetic optimum 2.24 m/s). Elite top speed 12.4 m/s. Ordinary running speeds: no opened source (unverified). | 175, 174 | No run. Fleeing uses the walk pace (sim.py:7860–7886). | missing |
| B5 | Vertical jump | Countermovement jump, hands on hips: men 16–20 y 35.3 cm (n 1764), men 21–30 y 34.6–34.9 cm (small groups), women 21–35 y 22.8–27.9 cm; healthy adults 29.9 ± 6.3 cm. Takeoff ≈ 2.1–2.6 m/s. | 176, 15 | `_leap_speed` (sim.py:5845): legs 0.015 m² × `muscle_Pa` 93 kPa (sim.py:554) = 1395 N over a crouch of 0.36 × 0.70 m = 0.252 m on 43.9 kg → takeoff 3.33 m/s, rise 0.565 m (derived). Jumps are aimed, so this is a ceiling. `muscle_Pa` was calibrated on the old body. A 0.30 m jump needs ≈ 950 N, i.e. `muscle_Pa` ≈ 63 kPa (derived). | wrong (1.6–1.9× too high) |
| B6 | Standing long jump | 117.3 ± 11.9% of body height, start line to back heel (n 31) → ≈ 1.76 m at 1.5 m | 16 | `_leap` aims at v = √(g·d) at 45°, capped by vmax (sim.py:5922); COM reach at most vmax²/g = 1.13 m. The toe-to-heel vs COM correction is unverified. | ok for game |
| B7 | Throw speed | Five untrained men (Cross 2004). One thrower: 19.5 m/s with 57 g, 8.5 m/s with 3.4 kg (a factor 2.3). The Table III fits give 11.5–14.5 m/s at 1 kg, 17.2–21.2 m/s at 0.16 kg, 9.0–11.6 m/s at 1.95 kg. Forearm only: 9.5 m/s (tennis ball), 6.3 m/s (0.73 kg). | 14 | `_release` (sim.py:3296) uses ω·r with `arm_wmax` 15 rad/s (sim.py:644) at 0.45 m → 6.75 m/s ceiling for light objects (forearm-only level). Old body (roadmap): 0.97 kg left at 4.92 m/s and flew 2.45 m; 1.95 kg at 5.28 m/s with a wind-up. | wrong (2–3× slow; range ≈ 6× short) |
| B8 | Lifting | NIOSH load constant 23 kg (safe for 75% of women and 90% of men under ideal conditions). Median peak grip 51 kg (men), 31 kg (women); grip is squeeze force, not lift. Maximal two-hand lift: unverified. | 177, 178 | `strength_N` 400 (sim.py:529): ≈ 40 kg off the floor, ≈ 100 kg shoved. No fatigue. | ok for game |
| B9 | Shoulder torque | Maximum isometric flexion 77.2 N·m at 90°, 43.2 N·m at 160° (14 healthy men) | 179 | Swing: `arm_Nm` 60 (sim.py:575). Hold: `_torque_of` (sim.py:6342–6364) = 0.015 m² × 93 kPa × 0.45 m × `lever_frac` 0.19 = 119 N·m. `lever_frac` was derived for the old one-voxel arm; the arm section then doubled. | ok for swing; wrong for hold (one shoulder, two strengths) |
| B10 | Punch | Olympic boxers: hand 9.14 ± 2.06 m/s, force 3427 ± 811 N, effective mass 2.9 kg, energy delivered to the head 17.2 ± 8.9 J, head 58 ± 13 g and 6343 ± 1789 rad/s². Rear-hand force: novices 2381 N, elite 4800 N. | 180, 181 | Old arm (roadmap 2026-09-07): knuckle 3.8 m/s. Current arm estimate 4–6 m/s, ≈ 32 J (derived, not run). The swing inertia uses a Manhattan radius (sim.py:2320–2321), which lowers speed. | wrong for speed; delivered energy is the right order |
| B11 | Knockout | Secondary thresholds via Walilko 2005: concussion at 4500 rad/s² (Ommaya), 50% probability above 1800 rad/s² (scaled primate data), severe injury 18000 rad/s²; translational 200 g; HIC 250 for mild injury | 180 | No head pathway. A knockout needs 0.2 × `hurt_J` 2500 = 500 J above threshold (sim.py:522), ≈ 25× a real punch. | missing |
| B12 | Blood volume | 70 ml/kg (men), 65 ml/kg (women); ≈ 5 L in a typical adult | 182 | `BLEED['ml_per_kg']` 70 (sim.py:701) × flesh mass (`_blood0` sim.py:3910) = 3071 ml | matches |
| B13 | Haemorrhage classes (ATLS) | I up to 15%; II 15–30%; III 30–40% (anxious, confused, BP down); IV over 40% (confused, lethargic) | 183, 112 | `_perfusion` (sim.py:3920) with f0 0.45, k 0.56 (sim.py:717) against `faint_o2` 0.55 and `death_o2` 0.25: faint at 36.1% lost, death at 50.4%, no waking until loss < 25.9% (derived). Death is instant when the line is crossed. | ok for game (faint early in class III) |
| B14 | Bleeding from a major vessel | Complete cut of the femoral artery and vein: fatal within 10 min (case report). Resting flow 434.4 ml/min in the common femoral artery, 172.5 ml/min in the superficial femoral (7.2 and 2.9 ml/s). These are intact-artery flows, not bleed rates. Severe bleeding can kill "within minutes". | 184, 185, 187, 116 | Flow = `ml_s_m2` 400 × cut plane × pressure (sim.py:3962); no vessels. Largest limb wound 5e-3 m² → 2 ml/s: faint 13.2 min, death 23.6 min. Torso 0.015 m²: 3.9 / 6.1 min (derived). | wrong (a femoral cut is 2–3× too slow) |
| B15 | A small cut stops bleeding | Ivy bleeding time 3–10 min (10 mm × 1 mm cut); Duke ear prick 2–5 min | 113 | `clot_m2_s` 1.7e-6 (sim.py:710) seals a fixed area per second (sim.py:3955): the Ivy cut in 5.9 s, a 3 mm × 10 cm cut in 2.9 min, a 10 × 5 cm deep wound in 49 min (roadmap 96 says "an hour and a half") (derived) | wrong (small cuts 30–100× fast) |
| B16 | Pressure and tourniquet | Needed but not applied: 0% survival (n 5) vs 87% with a tourniquet. Applied before shock: 90% survival vs 10% after. | 186 | `w['pressed']` cuts flow 10× (sim.py:3961) but no action sets it. No tourniquet. A cut-through limb does not come off. | missing (the actions) |
| B17 | Burn threshold and rate | Injury starts when the basal epidermis reaches ≈ 44 °C. The damage rate rises ≈ 1.8× per +1 °C. 44 °C 5–6 h, 45 °C 2 h, 47 °C 20 min. Moritz & Henriques (2nd / 3rd degree): 49 °C 8 / 10 min, 51 °C 2 / 4.2 min, 55 °C 17 / 30 s, 60 °C 3 / 5 s. Burns can incapacitate within minutes; burn death comes minutes to hours later. | 188, 189, 190, 191 | `hurt_T` 55 °C (sim.py:505); `burn_gain` 4e-6 per °C over per tick (sim.py:510), linear, on the **mean** over-temperature of all flesh voxels (sim.py:5296–5297), clipped at 400 K over. A hand voxel at 60 °C would take ≈ 27 h to cause a faint (derived). Death at `death_burn` 0.45 is instant (sim.py:5326). | wrong (threshold 11 °C high; linear; whole-body mean) |
| B18 | Heat pain | Skin pain at 42–45 °C (≈ 43 °C C fibres, 45.7 °C Aδ fibres); TRPV1 activates above 43 °C | 188, 41, 251 | `feel_T` 45 °C (sim.py:506) against the hottest flesh voxel (sim.py:5295) | matches |
| B19 | Body heat | Resting metabolism ≈ 104 W (58 W/m² × 1.8 m²); walking at 5 km/h ≈ 360 W; skin baseline ≈ 30 °C | 43, 41 | Flesh starts at `AMBIENT` 20 °C. Nothing makes body heat. A body must warm 25 K to feel heat, not 11–15 K. | missing (cold and hypothermia also impossible) |
| B20 | Hot air | ISO 13571, unclothed: t = 5e7·T^−3.4 min (7.9 min at 100 °C, 4.3 at 120, 2.0 at 150, 0.8 at 200), ±25%. Fire Service Academy table: 15 / 7 / 5 / 2 min. Purser: t = exp(5.1849 − 0.0273·T) min (11.6 min at 100 °C, 3.0 at 150); skin pain limits exposure above 121 °C. Heat stroke: core above 40 °C with CNS dysfunction. | 38, 191, 192 | No air-temperature, heat-flux or core-temperature path. Only flesh voxel temperature feeds the burn integral (sim.py:5295–5297). | missing |
| B21 | Low oxygen | 16% O2: impaired thinking; 14%: abnormal fatigue; 12.5%: very poor judgement; below 10%: unconsciousness and death; 4–6%: coma in under 40 s | 193 | `blood_o2` relaxes toward inhaled O2/fresh at `breath` 0.02 per tick (sim.py:490, 5294): τ 1.25 s. Faint below 0.55 (≈ 11.5% O2), death below 0.25 (≈ 5.3%). At 5% O2: faint 1.1 s, death 5.3 s; at 10%: faint 2.5 s (derived). | wrong (thresholds right; 10–40× fast; death instant) |
| B22 | Drowning and breath-hold | Surface struggle 20–60 s; unconscious after 1–3 min under water; death ≈ 10 min. Submersion 0–5 min: 10% death or poor outcome after hospital treatment; 6–10 min: 56%. | 194 | No breath store. A face with no AIR neighbour faints in 0.75 s and dies in 1.7 s (sim.py:5290). A submerged face counts water-filled cells as AIR that still hold fresh O2 (sim.py:5285, 8612), so a body breathes under water (read from code, not run). | wrong (nobody drowns; a buried face dies at once) |
| B23 | Carbon monoxide dose | %COHb = 3.317e-5·ppm^1.036·RMV·t (RMV 25 L/min for light work); incapacitation near 30% COHb. 1600 ppm: death within 1 h; 6400 ppm: 25–30 min; 12800 ppm: 1–3 min. | 191, 195 | Soot grams stand in for CO: load += `smoke_in` 0.02 × grams per voxel − `smoke_out` 0.001 per tick (sim.py:491, 497, 5288–5292). It grows only above 0.05 g per voxel: 400 g/m³ at 5 cm, 50 g/m³ at 10 cm. `house_fire` at 21 s: load 0.00 (roadmap 90). | wrong (uncalibrated, 8× grid dependent) |
| B24 | Clearing CO | COHb half-life: 320 min on room air, 80 min on 100% O2, 23 min hyperbaric | 196 | A smoke victim wakes ≈ 5–12 s after reaching clean air (load from 0.45 to below 0.30 at 0.001 per tick, then ≈ 1 s O2 lag; derived) | wrong (> 1000× fast) |
| B25 | Fall injury | Current field triage (2021): fall > 3 m (10 ft), any age, is a moderate-risk criterion. Hospital series: survivors fell 5.2 m on average, non-survivors 8.6 ± 2.3 m (Agalar 1999); critical border for survivable injury 6–7 m, mean 7.2 m (Richter 1996). These series exclude deaths at the scene, so they understate lethality. 100% mortality at 8 storeys or more (25–30 m). A head-first fall from the first floor kills ≈ 50%. LD50 ≈ 15 m, LD90 ≈ 26 m (secondary). | 197, 22, 198 | `_land_body` (sim.py:3636): each contact cell absorbs 30 J (`bruise_kJm2` 12 × 0.0025 m²); the excess / `hurt_J` 2500 adds to hurt; faint above 0.2, death above 0.45 (sim.py:511). Rigid, feet first, 12 foot cells, 43.9 kg: knocked out at ≈ 2.0 m, dead at ≈ 3.45 m (derived, not run). The roadmap fall figures are stale (old body, before the `FALL_SUBSTEPS` fix) and disagree with each other (BODY comment "≈ 6 m kills"; "4.5 m dead"; roll table "4.5 m knocked out (0.52)" although death is at 0.45). | wrong (derived; ≈ 2–4× too lethal) |
| B26 | Oxygen use and fatigue | 1 MET = 3.5 ml O2 per kg per min | 250 | Bodies use no O2 and never tire; a load can be held for ever | missing (low) |

### 2.8 Sound (A)

| # | Quantity | Real value | Src | Sim value | Verdict |
|---|---|---|---|---|---|
| A1 | Shout level at 1 m | Shouted speech: men 89 dBA, women 82, children 82 (SD 7–9). Vocal-effort scale: shouting 84, maximal shout 90, maximal individual shout 96 dB(A). Field shouts: 84–95 dB (male), 82–92 (female), mean 88.8 ± 4 dB; 7 ± 1 dB quieter directly behind the caller. | 199, 200, 221 | `SHOUT_DB` 90 (sim.py:850), the same for every speaker and in every direction | ok for game (adult male) |
| A2 | Normal and quiet speech | At 1 m: casual 50–53, normal 55–58, raised 63–65, loud 71–76 dBA; whisper 36 dB(A) | 199, 200 | Not modelled. `LINES` are shouts (sim.py:410); `_say` always emits `SHOUT_DB` (sim.py:5632–5643). | missing |
| A3 | Who understands a spoken line | A listener 3 m away in open air hears a shout at ≈ 80 dB | — | `_hears` decides what a line means (percept sim.py:7741, count sim.py:7400). It calls `_level(eye, qeye, SHOUT_DB)` with own=None (sim.py:5519–5521). Both eyes are head-top flesh cells, so the ray charges the mass law for the speakers' own heads: 40–46 dB (formula replica, two `_person` bodies 3 m apart). Result: a shout is understood only within ≈ 1.6–3.2 m in open air, and never through any wall. This is the bug roadmap 97 fixed only in `_law_hear` (sim.py:5549–5563). The only heard-line test uses 0.5 m, where the 1 m clamp hides it (tests/test_voxel.py:4922–4924). | wrong (bug; confirm with A-T1) |
| A4 | Spreading loss | −6 dB per doubling of distance in free field | 201 | level = db1m − 20·log10(d), d ≥ 1 m (sim.py:5612–5613) | matches |
| A5 | Understanding speech at a distance | Field shouts at ≈ 88 dB: 50% of words understood at d_i = 5619·e^(−0.0978·dB_amb) m (forest): ≈ 300 m at 30 dB ambient, 112 m at 40, 42 m at 50, ≈ 6 m at 70 (the last is an extrapolation). Audible but not understood to ≈ 270–330 m in quiet forest. Attenuation ≈ 1/r^2.35. Interspeech 2018: words ≈ 95% at 30 m (loud speech), 85% at 60 m and 75% at 90 m (shouted; listeners could see the speaker). A normal voice of 65.3 dBA in 41.6 dBA of natural noise: 77.8% at 11 m, 35.9% at 33 m. SNR 0–3 dB is "satisfactory", 3–9 "good", below −3 "insufficient". | 221, 219, 220, 200 | Fixed `HEAR_DB` 40 (sim.py:851). No ambient level, no SNR, no excess attenuation. Through `_law_hear` a shout is noticed to 562 m and wakes a sleeper to 31.6 m in free field. | wrong (no masking) |
| A6 | Background noise | Suburban homes 41 dBA inside / 48 outside; urban 48 / 61; restaurants 68.9 / 72.7 / 76.5 dB (low / medium / high occupancy); WHO bedroom guideline 30 dB LAeq, 45 dB LAmax per event | 199, 200, 217 | None. A brawl, a fire or other shouts never mask anything. | missing |
| A7 | Voices rise in noise (Lombard) | L_speech at 1 m = 55 + c·(L_noise − 45), c 0.5–0.7 above 45 dB | 200, 218 | Voices never rise | missing |
| A8 | Mass law | TL = 20·log10(m·f) − 46.8 (SI, random incidence). Real panels fall short near and above the coincidence frequency. | 208 | TL = max(0, 20·log10(σ·500) − 47) on the total path mass (sim.py:5626–5629) | matches (formula) |
| A9 | Timber panel | 25 mm floor (12 kg/m²) ≈ 28 dB; 50 mm boarding (24 kg/m²) ≈ 32 dB | 209 | One 5 cm WOOD voxel (30 kg/m²) costs 36.5 dB | ok for game (2–5 dB high) |
| A10 | Masonry wall | Precast concrete: 100 mm STC 49, 150 mm 55, 200 mm 58. 200 mm concrete (460 kg/m²) ≈ 52 dB. Hollow block 4 / 6 / 8 / 10 in: STC 44 / 46 / 48 / 50. | 210, 209, 211 | STONE 135 kg/m² per 5 cm: 1 voxel 49.6 dB, 2 voxels 55.6 dB, 4 voxels 61.6 dB | ok for game (4–10 dB high) |
| A11 | Doors | Hollow-core door STC 20–25; gasketed solid door 35–40; gaps can drop an STC 52 door to 21. In homes, closing a door adds ≈ 10 dBA. | 212, 210, 205 | A 5 cm pine door costs 36.5 dB. A 97% filled door costs 36.2 dB. A shout behind it reaches at most 53.5 dB, below `wake_db` 60, so it never wakes anyone. | wrong (12–25 dB too high) |
| A12 | Gaps and openings | Composite TL = 10·log(ΣS / Σ S·10^(−TL/10)): each opening passes sound fully, so a 1% opening caps TL near 20 dB | 213, 210 | One straight ray; partial fill averaged by mass, not by transmitted energy | wrong |
| A13 | Sound around corners | Thin-barrier loss is capped at 20 dB (single diffraction, ISO 9613-2) and 25 dB (double). In homes, an adjacent room through an open doorway is 10–17 dBA down. | 207, 214, 205 | One straight ray (`_level`, sim.py:5593–5630). Around the corner of a 10 cm stone wall a listener loses 55.6 dB, even with an open doorway 1 m away. | wrong |
| A14 | A person in the way | Head shadow (interaural level difference) 1 + (f/1000)^0.8·sinθ dB: ≈ 1.6 dB at 500 Hz, 2.7–4.0 dB at 2–4 kHz. A proxy for body shadow only (unverified as an insertion loss). | 215 | A third person's head is 45 kg/m² per voxel (frac 0.9): one same-height bystander costs 46.1–49.6 dB | wrong |
| A15 | Foliage | ISO 9613-2 counts foliage only on paths of 10 m or more: 1 dB at 500 Hz for 10–20 m of dense foliage, 0.05 dB/m for 20–200 m. A barrier needs ≥ 10 kg/m² and no gaps. | 207 | LEAF 200 g/L (sim.py:218). The scene hedge (frac 0.3, 4 voxels deep, scenes.py:699) costs 28.6 dB; a 1 m hedge at frac 0.3 costs 42.5 dB. | wrong |
| A16 | Reverberation | Past the critical distance d_c = 0.057·√(γ·V/RT60) the level stays roughly constant | 206 | Free field only | ok for game (once A3 is fixed) |
| A17 | Air absorption | 20 °C, 70% RH: 2.8 dB/km at 500 Hz, 5.0 at 1 kHz, 9.0 at 2 kHz, 22.9 at 4 kHz | 207 | None | ok to omit under ≈ 100 m |
| A18 | Metal drop | Ingots dropped 0.9–1.2 m into metal bins: 105–110 dBA at the operator (a level over time); into empty bins > 115 dBA. Forge hammer peaks 138–143 dB, max 147–148 dB (unweighted peak). | 203 | Iron on iron, ring 1e-3 (sim.py:862; `_noise` sim.py:5579–5591): a 5 kg ingot dropped 1 m gives ≈ 112 dB at 1 m | ok for game (within ≈ 10 dB) |
| A19 | A swung metal blow | No measured sword-clash level found. The metal-drop data are the closest proxy. | 203 | A swing's body holds the arm's FLESH; `_spend` takes min(ring) over every material in it (sim.py:3536–3537), so it rings as FLESH (1e-7): a 20–60 J sword clash gives 68–73 dB, a hammer on an anvil ≈ 72–75 dB (derived) | wrong (≈ 35–40 dB below the metal proxy) |
| A20 | Church bell | Bell chamber, 1–2 m from 8–12 bells rung continuously: Leq 110–117 dBA. Outside the tower at 10–50 m: 64–76 dBA. | 202 | No bell. Every sound lives 2 ticks = 50 ms (sim.py:5530). | ok for a peak; sustained sources missing |
| A21 | Fire crackle | A recorded wood crackle played at 42–48 dBA (one peak of 58) at the pillow for 30 s woke 91% of adults; 92% of them woke within the first 30 s. Lab playback level, not a measured fire. | 204 | Fire never calls `_noise`. Glass cracked by heat is silent (`_shatter` sim.py:1715). | missing |
| A22 | Waking from sleep | ≈ 75 dBA at the pillow wakes most adults; 55–60 dBA wakes some in a quiet room. Voice or T-3 signal: mean 59 dBA; high-pitch alarm 72.5 dBA. 5 of 24 young adults (20%) did not wake reliably at 60 dBA (linked to sleep loss). 85% of 6–17 year olds slept through one or both 60 dBA presentations. 95% of awakenings came within 32 s. | 205, 247 | `wake_db` 60 (sim.py:875): one 50 ms sound at 60 dB or more wakes anyone at once (sim.py:5569–5573) | ok for game (unimpaired adults, voice signals); age, drink, sleep debt and duration missing |
| A23 | Noticing a noise | The ear's absolute threshold is ≈ 0 dB SPL near 1 kHz; in practice background noise sets the limit. WHO single-event guideline: 45 dB LAmax indoors. | 216, 217 | `notice_db` 35 (sim.py:872); the exact source cell is stored (sim.py:5577) | ok for quiet rooms; no masking |
| A24 | Footsteps | No verified measurement | — | 43.9 kg × 9.81 × 0.01 m = 4.3 J → 61.3 dB walking, 55.3 dB creeping (sim.py:869–870, 7939–7942). Ring is always FLESH, so floor and footwear do not matter. | unknown |
| A25 | Water surface | Air–water impedance ratio up to 3600 → almost 30 dB of loss | 222 | Fluid is not in `smass`, so water is acoustically transparent | missing |
| A26 | Fire in the sound path | A compartment fire changed a source's received level by more than 10 dB | 223 | None | missing (low) |
| A27 | Dead constants | — | — | `WILL['shout_hear_m']` 14 and `WILL['wall_cost_m']` 4 (sim.py:295–296) are used nowhere | clean up |

### 2.9 Perception and reaction (P)

| # | Quantity | Real value | Src | Sim value | Verdict |
|---|---|---|---|---|---|
| P1 | Horizontal field of view | ≈ 200–220° in total; binocular overlap ≈ 114–120°. Per eye: 60° nasal, 107° temporal, 70° up, 80° down. | 224, 225, 226 | `WILL['fov_deg']` 120 (sim.py:285) is the **whole** cone, ±60° (`_in_cone` sim.py:5509–5517; `_look_around` sim.py:5350). "Arm's reach" in `_in_cone` is 1.5 voxels (sim.py:5514), 7.5 cm at 5 cm. | wrong |
| P2 | Peripheral detection | Acuity is ½ at 2° off-axis, ⅓ at 4°, 1/16 at 30°. The periphery stays good at detecting flicker and motion. | 226 | Everything in the cone with optical depth below `SEE_TAU` counts as fully seen (sim.py:5505–5506, 5517) | missing (a two-zone model is enough: detect to ≈ ±100°, identify within ≈ ±30°) |
| P3 | Vertical field of view | ≈ 130–150° | 226, 224 | Azimuth only (sim.py:5512–5517) | ok for game |
| P4 | Noticing a flame | On a dark night a candle at ≈ 2.6 km is at the naked-eye limit. (A room fire in plain view at any indoor distance is reasoning, not a source value.) | 227 | `WILL['fire_see_m']` 4.0 (sim.py:283) as a Chebyshev radius (sim.py:7699–7708), so 5.66 m on the diagonal. Only the 25 nearest burning voxels are tested (sim.py:7704). `_look_around` still writes the fire into `p['danger']` to `see_m` 12 m, so the body avoids the fire but never alarms. | wrong |
| P5 | Light level | Full moon 0.05–0.3 lux; living room 50 lux; office 320–500 lux. Scotopic vision 1e-6–1e-3 cd/m². Surfaces under a full moon are ≈ 0.005–0.03 cd/m², which is mesopic: colour and acuity are reduced, not absent (derived). Cones adapt in 9–10 min; full dark adaptation takes 20–30 min. | 228, 229, 230 | No illumination term in `_sees` or `_look_around` (sim.py:5330–5506). Roadmap 95 lists "5. Light" as not built. | missing |
| P6 | Identifying a person | Eyewitness identification is worse at 20–50 m than at 4–15 m. A simulated far distance (≈ 172 ft) dropped discriminability from 1.77 to 0.58. | 231 | Percepts carry no identity doubt | missing (low) |
| P7 | Seeing through smoke | V = C/σ. C = 2 for walls, doors and other non-luminous objects (Jin's own recommendation); C = 5–10 for light-emitting signs (FDS uses 3 and 8 as conventions). Flaming soot Km = 8700 m²/kg. | 232, 233 | `_sees` adds `SMOKE_EXT`·c·vox_m per sample (sim.py:5505); visible if τ < `SEE_TAU` 2 (sim.py:816). So V·σ = 2 for every target. Samples are taken per Chebyshev step, so diagonal rays undercount τ by up to 1.41× (flat) or 1.73× (3-D). | ok for objects; wrong for flames and lamps (should be C 5–10) and for diagonals |
| P8 | Smoke and wayfinding | Escape becomes hard above σ 0.15/m (≈ 13 m visibility) for strangers, 0.5/m (≈ 4 m) for people who know the building. Rasbash: 10 m. German guidance: 10–20 m. In irritant wood smoke, visibility falls toward zero near σ ≈ 1.2/m (eyes close). | 232, 233 | `_look_around` (sim.py:5330–5433) has no smoke term. Layout, exits and fires are learned through any smoke out to 12 m. | wrong |
| P9 | Turning back in smoke | Of those who met smoke, 62.7% (Bryan) and 60% (Wood) moved through it; 29.9% and 26% of those turned back. Of the turn-backs, 54.1–66% could see 1.8 m or less, and 88–94% could see 9 m or less. | 234 | Routes ignore smoke. The emergency latch stops all re-deciding (sim.py:7473, 7697). | missing |
| P10 | First cue of a fire | Bryan (569 people): smell of smoke 26.0%, told by others 21.3%, noise (source not specified) 18.6%, told by family 13.4%, saw smoke 9.1%, saw fire 8.1%, explosion 1.1%, felt heat 0.7% | 234 | Percepts: `sees_fire`, `scorched`, `chokes`, `hears_alarm`, `sees_runner` (sim.py:7699–7752). Combustion makes no sound. | wrong (smell effectively absent; vision of fire is the main path) |
| P11 | Smell of smoke | No numeric odour threshold found. Most people who met smoke could still see 9 m or more. | 234 | `chokes` fires at load > 0.05 (sim.py:7726), read from grams per voxel: ≈ 0.4 g/L at 5 cm (sim visibility ≈ 0.9 m), 0.05 g/L at 10 cm (≈ 7 m) | wrong (late; 8× grid dependent) |
| P12 | Simple reaction time | ≈ 190 ms to a visual stimulus, 160 ms to a sound. Ruler drop 0.15–0.30 s (11–44 cm fall). | 235, 236 | A percept calls `_decide` on the same tick (sim.py:7770–7778): 0–25 ms to decide, 75–150 ms to the first step. `decide_every` 10 ticks (sim.py:764) paces only idle re-deliberation. Catching a thing is gated 0.2 s from **launch** (`BODY['react_s']` sim.py:629, check sim.py:6246), not from when it is seen. | wrong (≈ 10× fast) |
| P13 | Choice and surprise | Hick's law RT = a + b·log2(n + 1). Drivers: 0.7 s expected, 1.25 s unexpected, 1.5 s surprise (brake data, includes moving the foot). Startle cascade: blink 20–40 ms, head 60–120, arms 125–195, legs 145–395 ms. Trained responses 0.1–2 s; novel decisions 8–10 s. | 235, 237, 238, 245 | No dependence on menu size (`MENU_CAP` 7, sim.py:438) or on surprise | missing |
| P14 | Orienting | A saccade to an unexpected stimulus starts after ≈ 200 ms; gaze shifts over ≈ 20° bring in the head | 239 | `hears_noise` is in `ORIENTING` (sim.py:404): the gaze jumps to the source on the same tick and holds 25 ticks | ok for game (no delay, no flinch) |
| P15 | Time before people start to leave | Unannounced home drills: mean 2:49 with a good alarm, 8:35 with a poor one, 5:19 in winter. Offices: 0:36–1:03. WTC 1993 Tower 1: mean 11:02, median 5:00. WTC 2001: medians 3–4 min. Real fires are often slower than drills. Only 20–25% thought the alarm might be a real fire. | 240, 241 | Zero. `REFLEXES` send the legs to `go:exit` on the percept tick (sim.py:390–399). `p['emergency']` latches (sim.py:7473) and blocks all later sensing and deciding (sim.py:7697) until the body faints and wakes (sim.py:5320). | wrong |
| P16 | Visible fire in a crowd | Station nightclub (filmed): danger recognised 24 s after ignition, the crowd moving at ≈ 30 s, alarm at 41 s, unsurvivable at ≈ 90 s. Bradford (televised): a glow at 15:44, the whole stand burning within ≈ 270 s; some spectators cheered at cameras while others escaped. | 242, 243, 244 | Inside 4 m and in the cone: flight on the same tick. Beyond 4 m: never, unless told or a "runner" passes within 2.5 m (sim.py:284, 7752). | wrong |
| P17 | Response mix | Commentary on sudden severe disasters (not a fire study): 10–15% stay calm and act; ≈ 75% are "stunned and bewildered" and act reflexively; 10–15% are counterproductive (weeping, confusion, paralysing anxiety) | 245 | Every body runs the same `REFLEXES` row | missing |
| P18 | Panic | True panic is rare; people gather information and often help others | 246 | No panic state; bodies warn and rescue | matches |
| P19 | Smell does not wake sleepers | Peppermint never woke sleepers; pyridine woke 45% of stage 2 trials, 0% of stage 4, a third of REM; tones woke ≥ 75% | 248 | Sleepers skip `_law_will` (sim.py:7653); only noise wakes them | matches |
| P20 | Seeing someone run | A runner is seen by speed, in the field of view | — | `sees_runner` reads `q['fleeing']` (sim.py:7748), which is set when the other body's walk was started by a percept (sim.py:7560). That reads another mind. Fleeing bodies also walk at the idle pace. | wrong (god channel) |
| P21 | Memory of a seen fire | No literature value. The code comment intends "about a minute to half-weight" (sim.py:288–294). | — | `danger *= 0.985` per look (sim.py:5398), one look every 4 ticks (sim.py:7691): half-life 4.6 s, 13× faster than the comment. A 60 s half-life needs ≈ 0.99885 per look. | wrong (against its own intent) |
| P22 | Seeing a thrown thing | A thing is perceived by sight (in the cone, in line of sight) or by sound | — | `sees_thrown` and `sees_falling` need no cone and no line of sight (sim.py:7716–7721). Catching a falling person has no reaction gate (`_catch_person` sim.py:6213–6233). | wrong (god channel) |
| P23 | Alcohol | Choice reaction time worsened at 50 and 80 mg/100 ml blood alcohol. Arousal thresholds rose at 0.05–0.08% BAC (Ball & Bruck, cited; dB values not extracted). | 249, 247 | No per-person perception or reaction | missing |

## 3. The biggest gaps, ranked by what a player notices first

The order uses two questions: how often does a player meet the gap, and how plainly
does it look wrong? Each gap names the cause (symbols and lines), the fix it
suggests, and the tests in Section 4 that check it. "(derived)" means the size of
the gap comes from code arithmetic, not from a run.

### G1. A spoken line is not understood beyond about 2–3 m

- **What the player sees.** "Fire!" shouted across a room, into the next room, or past
  a person goes unheard. A shout through a closed door never wakes a sleeper.
- **Cause.** `_hears` (sim.py:5519–5521) charges the mass law for the speaker's and
  the listener's own heads: 40–46 dB (formula replica). `_law_hear` got this fix
  (roadmap 97); `_hears` did not. Beyond that, `_level` (sim.py:5593–5630) traces one
  straight ray: an open doorway around a corner costs 55.6 dB (real 10–17 dB), a
  bystander's head 46–50 dB (real ≈ 2–4 dB), a hedge 28–43 dB (real ≈ 0–1 dB), a
  plank door 36.5 dB (real 10–25 dB). There is no background noise, so a brawl
  never masks a shout either. Rows A3, A5, A11–A15.
- **Fix.** Make `_hears` and `_law_hear` call one shared level function with the ear
  lift and the own-body exclusion. Add an indirect path through openings, with the
  barrier loss capped at 20 dB. Weight partly filled cells by transmitted energy, not
  by mass, so a 1–3% gap caps a door near 15–20 dB. Do not charge the mass law for
  flesh and leaves; use ≈ 1–4 dB of body shadow and the ISO foliage rule. Add an
  ambient level and an SNR test (shout passes at 0–6 dB SNR, fails below ≈ −3 dB).
- **Tests.** A-T1 to A-T9.

### G2. Everyone reacts at once and in the same way; a fire 5 m away is not noticed

- **What the player sees.** All villagers bolt on the tick a "Fire!" is heard. A
  villager idles 5 m from a burning bed. Once a body flees, it never looks again.
- **Cause.** A percept calls `_decide` on the same tick (sim.py:7770–7778), so
  reaction is 0–25 ms (real ≈ 0.19 s simple, 1–1.5 s surprise). `REFLEXES` send the
  legs to `go:exit` at once (sim.py:390–399); real people take tens of seconds to
  minutes before they start to leave. `p['emergency']` latches and blocks all later
  sensing and deciding (sim.py:7473, 7697). `WILL['fire_see_m']` is a fixed 4 m
  radius (sim.py:283). `WILL['fov_deg']` 120 is the whole cone (real ≈ 200–220°).
  Rows P1, P4, P12, P13, P15–P17.
- **Fix.** Start a reaction delay at the moment of perception for every percept
  (≈ 0.2 s primed; ≈ 1–1.5 s for an organised act after surprise; a flinch under
  0.2 s). Add a pre-movement phase drawn from a distribution (look, gather, go to
  family). Replace the 4 m radius with line-of-sight salience (flame size and
  brightness against the light level). Detect in a ≈ ±100° cone; identify in a
  narrower one. Let a committed body keep sensing and re-deciding (remove the latch).
- **Tests.** P-T1, P-T2, P-T5, P-T7, P-T8, P-T9.

### G3. People walk at half speed and cannot run

- **What the player sees.** Villagers stroll out of a burning house.
- **Cause.** One voxel per `WILL['walk_every']` = 3 ticks (sim.py:297, 7874):
  0.67 m/s on a grid axis at 5 cm, ≈ 0.47 m/s on a diagonal, 1.33 m/s at 10 cm. Real
  walking is 1.1–1.65 m/s. People change to a run near 2.06 m/s. Rows B3, B4.
- **Fix.** Set the pace in m/s (≈ 1.4 walk) and convert it to ticks per voxel with
  `voxel_cm`. Count a diagonal step as √2 of a voxel. Add a run gait (≥ 2.1 m/s) for
  fleeing.
- **Test.** B-T7.

### G4. Fire burns on the wrong clock

- **What the player sees.** A bed is gone in seconds. A dropped lamp's oil burns out
  at once. A room reaches megawatts in 22 s.
- **Cause.** The combustion constants are per tick and never use `TICK_S`. Wood
  surface mass loss is ≈ 9 kg/(m²·s) against a real 2.9–9.6 g/(m²·s); a 5 cm wood
  voxel burns in ≈ 3.7 s (derived) against 40–80 min of real charring. An oil pool
  burns at ≈ 4 kg/(m²·s) against ≤ 0.04. Igniters are 100 kW "wicks" (a candle is
  ≈ 80 W). The error differs per law (see the note under 2.3). Rows F5, F6, F14,
  F16, F17, F24.
- **Fix.** Write each law in SI per second and multiply by `TICK_S`: burn rate per
  exposed area, the O2 allowance per face, k in W/(m·K), loss through exposed faces,
  radiation with σ. Add a sub-voxel surface model per face (surface temperature or
  char depth); without it, SI conduction makes ignition at least 5× too slow (derived).
  Add a heat-release readout, HRR = Σ heat · Δmass / `TICK_S`, so scenes can be
  checked against calorimetry. Rescale igniters from real HRR. Land this together
  with G5 and G7: constants.md says `RAD` and `LEAK` were tuned so flames cap, so a
  partial fix will make fires run away or die.
- **Tests.** F-T1 to F-T4, F-T11.

### G5. Hot things go cold in seconds

- **What the player sees.** A bar from the forge stops glowing in about 2 s. A cup of
  tea is cold in seconds. The forge cannot melt lead. Molten lead freezes at once. A
  quench is over in a fraction of a second.
- **Cause.** `LEAK` 0.004 (sim.py:776) takes 0.4% of every voxel's heat each tick
  (sim.py:8694): a 6.25 s time constant for any size, exposure or material. `RAD`
  radiates from every voxel, buried ones too (sim.py:8693). The `k` column is 400×
  k_SI (wood 800 W/(m·K), iron 8000). `c_airbase` gives air 41× its real heat
  capacity (sim.py:914). Liquid contact k gives h far above the boiling ceiling and
  depends on voxel size. Rows H1–H9, H12, H14–H16, H20–H24, H27, L11.
- **Fix.**
  1. Lose heat only through exposed faces: q = h·A·ΔT + ε·σ·A·(T⁴ − T∞⁴), with h
     5–25 W/(m²·K) and ε per material. Remove `LEAK` and the volumetric `RAD` term.
  2. Set `k_sim = k_SI × 0.0025`: iron ≈ 0.13–0.2, stone ≈ 0.003–0.01, wood ≈ 0.0004.
  3. Give air 1.2 J/(L·K). Keep a floor only where stability needs it, and only for air.
  4. Use the harmonic mean across a face, and a convective h at solid–gas faces.
  5. Fix the asymmetric clip (sim.py:4294) and keep the explicit per-face fraction at
     1/6 or less (or sub-step) (H18, H19).
  6. Use a boiling curve for liquid contact: a ceiling near 1 MW/m² and a Leidenfrost
     drop near 193 °C. Let fluid–fluid faces use the liquid's own k.
- **Tests.** H-T1, H-T2, H-T4 to H-T9, L-T4, L-T5.

### G6. Toppling tin, charcoal or ash crashes the sim

- **What the player sees.** A crash when a burnt tree or a tin ingot falls over.
- **Cause.** `MAT['ash']`, `MAT['char']` and `MAT['tin']` have 6 fields
  (core.py:102–104). `add_box` (core.py:361) and `Matter.promote` (matter.py:104)
  unpack 7. Confirmed by import. Row S16.
- **Fix.** Add the seventh field (`rig`) to the three rows. Set tin's yield strain
  near 2.7e-4.
- **Test.** S-T5.

### G7. Heat does not cross the air, and smoke does not layer

- **What the player sees.** Nobody can warm their hands at a campfire or flinch from
  a bonfire. Fire never jumps a lane to the next house. Smoke fills a room evenly, so
  crawling low does not help. A torch over oil (rulings case 7) cannot ignite it.
- **Cause.** No line-of-sight radiation; radiation acts only between touching voxels
  (sim.py:4288–4293). `_law_air` moves smoke but not heat (sim.py:4650). The plume
  carries 10–20× too much heat per degree at 2–4× too low a speed (derived). Smoke by
  height band in `house_fire` is flat (roadmap 90). Rows H16, H17, H25, H26, F20.
- **Fix.** Add a radiation pass from hot surfaces (above ≈ 300 °C) to exposed faces
  within a few metres, along line-of-sight rays (reuse the rays perception casts),
  with χr ≈ 0.3 of the HRR. Move heat with the gas parcel in `_law_air`. Scale the
  smoke "hot zone" threshold (sim.py:4912) by `vox_l`. Then measure the layer again.
- **Tests.** H-T3, F-T5, F-T6, F-T8.

### G8. Acid is fantasy acid

- **What the player sees.** A foot in a vitriol puddle dissolves in seconds. A plank
  vanishes in 2.5 s. A vinegar cask dissolves itself. An iron lock melts in minutes.
- **Cause.** `REACTIONS` rates (sim.py:249–259) are 10³–10⁷× too fast. The
  `(ACID, FLESH)` row removes a living body's tissue, which breaks rulings case 2.
  `WEAK_ACID` attacks wood, leaves and flesh. Every dissolved gram becomes smoke
  (sim.py:4577). Vitriol eats charcoal. Rates are linear in potency (sim.py:4566–4569).
  Rows C1–C6, C12–C14.
- **Fix.** Set each row's rate from SI data (mg per cm² per hour). Remove
  `(ACID, FLESH)` removal until the body-parts model exists; then model a burn depth
  that grows with contact time, with no mass removed. Make `WEAK_ACID` acetic acid:
  no flesh, wood or leaf rows; slow carbonate and iron rows. Make vitriol leave CHAR
  on wood. Put products where they belong: dissolved salt in the liquid, H₂ from
  metals, CO₂ from carbonates. Replace linear potency with a table that has
  thresholds (skin from 15 wt%; steel passive above ≈ 60 wt%). Add the heat of
  dilution from Table B.11.
- **Tests.** C-T1 to C-T8.

### G9. Breathing has no clock: nobody drowns, and smoke does nothing

- **What the player sees.** A villager walks on the pond floor and lives. A face
  pressed into a bed dies in under 2 s. Smoke never knocks anyone out; when it does,
  the victim comes round seconds after rescue.
- **Cause.** `BODY['breath']` 0.02 per tick gives a 1.25 s time constant
  (sim.py:490, 5294). Water-filled cells count as air with fresh O2 (sim.py:5285,
  8612). Smoke dose is in grams per voxel, so it changes 8× with voxel size
  (sim.py:5288), and it clears in ≈ 5–12 s against a real CO half-life of 320 min.
  Rows B21–B24, P11.
- **Fix.** Give the body an O2 store that falls at ≈ 3.5 ml/(kg·min) at rest, so a
  held breath lasts 1–3 min to unconsciousness. Give flooded cells no air. Replace
  the soot proxy with a Purser CO dose (%COHb) that clears with a 320 min half-life.
  Read every gas per litre, not per voxel.
- **Tests.** B-T1, B-T2, B-T8, P-T4.

### G10. Body numbers: throws slow, jumps high, arms heavy, no knockout

- **What the player sees.** A thrown stone lands 2–3 m away. A villager jumps like an
  athlete. A tavern brawl can never end in a knockout.
- **Cause.** `arm_wmax` 15 rad/s at a 0.45 m arm caps the hand at 6.75 m/s
  (sim.py:644), which is forearm-only level. `muscle_Pa` 93 kPa (sim.py:554) was set
  for the old body. Each arm is 13% of body mass instead of 5% (`scenes._person`).
  `lever_frac` (sim.py:557–563) was not re-derived when the arm was doubled. Injury
  is one joules integral (`hurt_J` 2500, sim.py:522), so a knockout needs ≈ 25× a
  real punch. Rows B1, B5, B7, B9–B11.
- **Fix.** Rebuild `_person` to the Winter mass fractions. Set `muscle_Pa` near
  63 kPa for a 0.30 m jump (derived). Add the kinetic chain (trunk, elbow, wrist) so
  a 1 kg stone leaves at 11–15 m/s. Re-derive `lever_frac`. Add a head pathway keyed
  on head angular acceleration (≈ 2000–6000 rad/s², secondary thresholds).
- **Tests.** B-T9, B-T11, B-T12, B-T15, B-T16.

### G11. Falls kill too easily, and a fainting person drops like a stone

- **What the player sees.** A villager faints and hits the floor in a quarter of a
  second. A 3.5 m fall from a roof kills.
- **Cause.** Owned bodies topple with a per-tick pendulum that has no g
  (sim.py:3496–3497): 0.28–0.55 s and 14 rad/s, and the answer changes with voxel size.
  Fall injury (`bruise_kJm2`, `hurt_J`, sim.py:521–522) was calibrated for a 28.5 kg
  body; on the 43.9 kg body a rigid feet-first fall knocks out at ≈ 2.0 m and kills at
  ≈ 3.45 m (derived). Real survival extends to 6–7 m; the 50% death height is
  ≈ 12–15 m (secondary). Rows M5, B25.
- **Fix.** Topple owned bodies with θ'' = (3g/2H)·sinθ in SI, or hand them to the
  MPM like unowned clusters. Recalibrate the fall injury on the current body against
  the fall data. Add fractures, so that feet-first falls break legs instead of
  knocking people out.
- **Tests.** M-T2, B-T6.

### G12. Water on a burning lamp does nothing dramatic

- **What the player sees.** A bucket on a burning lamp boils quietly under the oil.
  Molten lead poured into the quench tub makes no hiss and no steam.
- **Cause.** No steam species; boiling adds only `p_add += 0.1` per gram
  (sim.py:4491). Molten metal and water exchange heat only through air k
  (sim.py:4268–4276). Rows F21, L12, L13.
- **Fix.** Track steam as a gas with volume (≈ 1670×). Let rapid boiling under oil
  throw out oil parcels. Use the boiling contact law on fluid–fluid faces.
- **Tests.** F-T9, L-T5, C-T14.

### G13. The breaking height depends on the code path and voxel size, not the material

- **What the player sees.** A bottle survives a topple from a shelf but breaks when
  knocked straight down 15 cm. A stone block shatters when dropped a few centimetres
  through the MPM path. A felled trunk breaks in the air. A 3 m plank bridge loses its
  middle with nobody on it; a person on a 2 m plank never bends it.
- **Cause.** The lattice and the MPM use different rules (S8, S11). Strength is
  scaled by √stiff for static loads too (S3, S23). The impact rule ignores
  compression vs tension and the rate gain of rock (S11). The yield strain is not
  scaled (S12, S15). `SPAN` is a fixed reach with no load or depth (S21, S22).
  Impact energy counts only one cell's mass (S8).
- **Fix.** Drive both paths from one material record (roadmap 95 todo). Use the whole
  body's mass, contents included, for impact energy. Judge quasi-static stress
  against the real strength; judge the impact part as √(E_real·ρ)·Δv, with sign and a
  rate gain for rock. Treat yield as a stress. Replace `SPAN` with a beam check
  (σ = M/S against MOR, with M from the actual load). Carry fill into MPM as
  thickness.
- **Tests.** S-T1 to S-T7.

### G14. Bleeding has no vessels; small cuts stop in seconds; nobody can press a wound

- **Cause.** Flow scales only with the cut plane (`ml_s_m2` 400, sim.py:3962).
  Clotting seals a fixed area per second (`clot_m2_s`, sim.py:710). `w['pressed']`
  exists (sim.py:3961) but no action sets it. Shed blood never gels or dries.
  Rows B14–B16, L23.
- **Fix.** Add vessels per body region (a cut femoral artery kills in under 10 min).
  Make clotting depend on vessel size (3–10 min for a small cut). Add press and
  tourniquet actions. Let shed blood gel in 2–20 min.
- **Tests.** B-T3, B-T4, B-T5, L-T7.

### G15. Burns use the wrong threshold, the wrong law and the wrong place; bodies are cold

- **Cause.** `hurt_T` 55 °C (sim.py:505), a linear `burn_gain`, the mean over all
  flesh voxels (sim.py:5296–5297), and 5 cm flesh cubes where real burns happen in
  the top millimetre. Flesh starts at 20 °C with no metabolism. Rows B17–B20.
- **Fix.** Use a local damage rate from ≈ 44 °C that rises ≈ 1.8× per °C. Add a
  sub-voxel skin layer heated by the surface flux. Start flesh at a 37 °C core and
  ≈ 30–33 °C skin, with ≈ 100 W of metabolism. Delay burn death.
- **Tests.** B-T13, B-T14.

### G16. Liquids: spill area depends on voxel size, and every liquid spreads alike

- **Cause.** A fixed 1 ml slope stops spreading (sim.py:1319). Droplet gravity is in
  lattice units (sim.py:1186). Only water boils (sim.py:4485). Nothing evaporates.
  Temperature cannot fall below 0 °C (sim.py:8695). `MISCIBLE` lacks tin + lead, acid +
  weak acid and blood + water (sim.py:265–266). Rows L7, L10, L14–L18.
- **Fix.** Stop spreading at a film thickness h = 2·λc·sin(θ/2) per fluid (water
  1–4 mm, molten lead ≈ 3–4 mm). Set droplet gravity to `GRAVITY·TICK_S²/vox_m` per
  tick². Let water-based fluids boil. Add evaporation, ICE, and the missing mixtures.
- **Tests.** L-T1 to L-T3, L-T8 to L-T12.

### G17. Sight: smoke does not hide the layout, flames look like walls, memory fades in seconds

- **Cause.** `_look_around` has no smoke term (sim.py:5330–5433). `SEE_TAU` 2 is used
  for every target (sim.py:816). Diagonal rays undercount smoke. `WILL['forget']`
  gives a 4.6 s half-life against its own "about a minute". `sees_runner` reads the
  other body's mind (sim.py:7748); `sees_thrown` needs no sight (sim.py:7716–7721).
  Rows P7, P8, P20–P22.
- **Fix.** Use the `_sees` smoke term in the `_look_around` rays. Use τ < 5–10 for
  self-luminous targets. Weight ray samples by Euclidean length. Set the forget factor
  near 0.99885 per look. Detect runners by seen speed, and thrown things by sight or
  sound.
- **Tests.** P-T3, P-T7, P-T10.

### G18. Sound details: swung metal rings like flesh; fire is silent; sleep is one switch

- **Cause.** `_spend` takes min(ring) over the swinging body, which includes the arm
  (sim.py:3536–3537). Fire never calls `_noise`. Every sound lives 50 ms
  (sim.py:5530). One 60 dB sample wakes anyone (sim.py:5569–5573). Rows A19–A22.
- **Fix.** Take ring from the contact cells or the tool. Add sustained sources (fire
  crackle, bells). Make waking a probability over exposure time, with thresholds per
  person (age, sleep debt, drink).
- **Tests.** A-T5, A-T11, A-T12, A-T14.

### G19. Smaller mechanics gaps

- Lattice landings never bounce (M9). One friction value for every pair and no lattice
  sliding (M7). Arrow drag is 1.4–2.3× low (M8). Landing energy ignores drag (M3).
  Semi-implicit Euler drops things ≈ g·t·dt/2 early (M10).
- **Tests.** M-T4 to M-T6.

## 4. Reference tests

These tests become `tests/test_realism_<domain>.py` in later build waves. One file
covers one domain. Where two researchers proposed the same test, the tests are merged
and placed in one file only.

Rules for every file:

- Run each test at `voxel_cm` 5. Where the test says so, run it at 10 as well; the two
  results must agree within the stated band. A result that depends on voxel size is a
  failure even when one size passes.
- Convert ticks with `TICK_S` (40 ticks = 1 s). Long real-time tests need long runs
  (10 min = 24,000 ticks). Mark them slow and keep them out of the default run.
- "Today" gives the current result. Most are derived from code, not run. A test that
  "passes today" is a regression guard.
- A test that needs a law that does not exist yet (radiation across air, a body O2
  store, a STONE split) says so. Keep it as an expected failure until that law lands.
- Priorities: **high** = a player meets it in the first scenes; **medium** = a player
  meets it in a specific scene; **low** = detail or long-run behaviour.

### 4.1 `tests/test_realism_mechanics.py`

1. **M-T1 Drop timing** — high
   - Scene: 5 cm world, stone floor. Release a 1-cell wood block and a 4×4×4 stone cube
     with their bottoms 1.0, 2.0 and 3.0 m above the floor (lattice fall).
   - Measure: ticks from release to rest on the floor; `vfall` on the last moving tick.
   - Expected: 18 / 26 / 31–32 ticks ± 1 (0.45 / 0.64 / 0.78 s); impact speed
     4.43 / 6.26 / 7.67 m/s ± 5%, after allowing the tick bias.
   - Source: 1. Today: passes (derived). Regression guard.
2. **M-T2 A person keels over** — high
   - Scene: `_person` on an open stone floor, no wall within 2 m, made to faint so that
     `_collapse` topples the body. Run at 5 cm and at 10 cm.
   - Measure: ticks from the first lean to landing; head speed at impact.
   - Expected: 0.6–1.2 s (24–48 ticks); the rigid-rod value 0.9–1.5 s is an upper
     bound. Head speed 4–8 m/s. The 5 and 10 cm runs agree within 20%.
   - Source: 7, 8. Today: 0.28–0.55 s at 13.4–14 rad/s, voxel-size dependent (derived).
3. **M-T3 Pole topple through the seam** — high
   - Scene: 5 cm world, stone floor. A 2.0 m and a 3.0 m unowned wooden pole, 2×2
     cells, built with a definite lean so that its COM is clearly outside the footprint
     (a one-cell offset of the top half is not enough: the COM stays inside).
   - Measure: the COM angle about the pivot edge at promote; time from promote to the
     tip's first floor contact; angular speed and tip speed at contact; whether the
     base slips backward.
   - Expected: time within ±25% of the rod equation θ'' = (3g/2H)·sinθ integrated from
     the measured start angle. Angular speed at impact 3.84 rad/s (2 m) and 3.13 rad/s
     (3 m) ± 20%; tip 7.7 / 9.4 m/s ± 20%. The base starts to slip near 50° (µ 0.4).
     Trunk integrity is tested in S-T2.
   - Source: 8. Today: never measured.
4. **M-T4 Shove a crate** — medium
   - Scene: a 40 kg wooden crate on (a) a wooden floor and (b) a stone floor.
     Lattice: push it. Seam: promote it with `vel = (3, 0, 0)`.
   - Measure: lattice breakaway push force; seam slide distance to rest.
   - Expected: breakaway 0.25–0.62 × weight (static bounds; the values are static
     coefficients). Slide from 3 m/s: wood on wood (kinetic 0.3–0.48) 0.95–1.5 m; wood
     on stone 1.1–2.3 m (static-coefficient bound) ± 30%. Later: ice (µ 0.02) ≈ 23 m.
   - Source: 9. Today: breakaway 0.40 × weight passes; the lattice has no sliding
     (record as missing).
5. **M-T5 Block bounce** — low
   - Scene: a 20 cm wooden cube dropped 1 m onto stone, once by lattice fall and once
     as an MPM free body.
   - Measure: height of the first rebound.
   - Expected: 10–30 cm (e 0.3–0.55); accept 5–35 cm. The two paths agree within 10 cm.
   - Source: 20, 21. Today: lattice 0 cm, MPM ≈ 7 cm.
6. **M-T6 Bolt down a range** — low
   - Scene: loose a 60 g wooden bolt (`area_m2` 8e-5) level at 50 m/s from 1.5 m; then
     at 40° on open ground.
   - Measure: speed and drop after 20 m of level travel; range at 40°.
   - Expected: speed 47.5–49.0 m/s; drop 0.78–0.86 m (the band allows the tick bias);
     range 180–330 m.
   - Source: 12, 13. Today: 49.25 m/s and 0.845 m (derived): fails on speed (drag
     1.4–2.3× low).

Throws, jumps, leaps and falls are in 4.7. The bottle drop is in 4.6.

### 4.2 `tests/test_realism_heat.py`

1. **H-T1 A glowing bar goes dark** — high
   - Scene: open sky, 5 cm. An iron bar 1 × 1 × 8 voxels on two stone voxels, set to
     800 °C; nothing else hot. Repeat at 10 cm.
   - Measure: time for the middle voxel to fall below 525 °C, 500 °C and 60 °C.
   - Expected: 800 → 500 °C in ≈ 680 s (2× the 25 mm round bar's ≈ 340 s; lumped
     estimate); accept 170–1400 s. The glow lasts more than 2 min. 5 and 10 cm agree
     within 25%.
   - Source: 44, 45. Today: at most 2.6 s (derived).
2. **H-T2 The forge melts lead and heats iron** — high (slow)
   - Scene: like `forge`, open sky, 5 cm. (a) One LEAD voxel (≈ 1.4 kg) buried in a
     char bed burning at 800–900 °C (or thermostat blocks at 850 °C). (b) An iron bar
     8 voxels long with its last 2 voxels in the hottest part of the bed.
   - Measure: (a) peak lead temperature, time to the first molten ml and to fully
     molten. (b) peak temperature of the buried iron and time to 950 °C. Run
     24,000–48,000 ticks.
   - Expected: (a) the lead passes 327.5 °C and melts fully in 0.3–10 min (≈ 89 kJ
     at 0.5–3 kW absorbed). A plateau below 327 °C fails. (b) The iron reaches
     ≥ 950 °C within ≈ 20 min (ASM: ≤ 5 min per inch, ≈ 10 min for 5 cm). A plateau
     below 900 °C fails.
   - Source: 47, 48, 84, 87. Today: lead flat at 258–263 °C, iron at ≈ 256 °C
     (roadmap 90). LEAK alone takes ≈ 9 kW from the lead voxel at 327 °C (derived).
3. **H-T3 Radiant heat at a distance** — high (needs line-of-sight radiation)
   - Scene: open ground; a wood crib burning at a measured HRR Q (≈ 1 MW). Flux probes
     face the fire at 1, 2, 2.5, 4 and 5 m, with a clear line of sight, upwind. A pine
     board faces the fire at 1 m.
   - Measure: incident flux on each probe (kW/m²), not voxel skin temperature (a 5 cm
     flesh voxel needs ≈ 20 min at 3.8 kW/m² to reach 45 °C); time to ignite the board.
   - Expected: q = χr·Q/(4πR²), χr 0.15–0.6 (0.30 ± 0.03 for wood and tree fires): for
     1 MW and χr 0.3, ≈ 3.8 kW/m² at 2.5 m and ≈ 1 at 5 m; accept ×/÷ 2 at R ≥ 2 m.
     The 1 m value (≈ 24 kW/m²) is an upper bound only. Board: ignites in ≈ 1–4 min
     (the ignition rule gives ≈ 130 s at 24 kW/m²). Once a skin layer exists: time to
     pain 4.2·q^−1.9 min ± 25%.
   - Source: 36, 38, 40, 72. Today: no heat crosses the air gap.
4. **H-T4 A hand on the cold end** — high
   - Scene: (a) an iron bar 12 voxels long in air, one end held at 1100 °C for 3 min.
     (b) a wooden stick 6 voxels long with its top voxel held at 250 °C (below
     ignition) or replaced by char, for 5 min.
   - Measure: temperature 40 cm from the hot end of the bar; 15 cm below the top of
     the stick.
   - Expected: (a) under 45 °C after 3 min (textbook conduction gives < +1 K at 40 cm).
     (b) under 30 °C after 5 min (heat reaches only 31–35 mm behind a char front).
   - Source: 23, 25, 49, 41. Today: decay lengths ≈ 12 cm (iron) and ≈ 6.8 cm (wood)
     within seconds (derived). Note: an 800 °C top would ignite the next wood voxel
     (≈ 395 °C), so do not use 800 °C.
5. **H-T5 Quench: water vs oil** — medium
   - Scene: iron at 850 °C, a piece at least 5 voxels across (or apply a Biot
     correction), dropped into (a) oil at 60 °C and (b) water at 20 °C. Repeat at 10 cm.
   - Measure: centre temperature over time; times to 600, 400 and 200 °C; the ratio of
     oil time to water time.
   - Expected: for a 50 mm piece in brine, core 472.5 °C at 30 s, 287.7 °C at 50 s,
     181 °C at 75 s (accept ×/÷ 2). For a 12.5 mm probe in still oil: 600 °C at
     5.4–8.2 s, 400 °C at 8.4–12.6 s, 200 °C at 35.6–55.8 s. Water is faster than oil;
     a ratio band of 2–6 is an estimate (unverified). 5 and 10 cm agree within 25%.
   - Source: 52, 53, 54, 142. Today: time constants 0.12–0.74 s at 5 cm,
     0.49–2.97 s at 10 cm; contact ratio 20 (derived).
6. **H-T6 A hot drink on the inn table** — medium (slow)
   - Scene: water at 86 °C in a small cup on a wooden table, still room air at 20 °C.
     Cup size (250 ml) and material are assumed.
   - Measure: time to 60 °C and to 40 °C.
   - Expected: 60 °C in 7–30 min; 40 °C in 25–100 min. Matching the curve also needs
     evaporation (L-T9), not only a loss fix.
   - Source: 46, 82. Today: at most 3.5 s to 60 °C (derived).
7. **H-T7 The plume above a hearth, at two voxel sizes** — medium
   - Scene: a steady wood fire on an open hearth; HRR from the sim's burn rate; open
     sky. Run at 5 and 10 cm.
   - Measure: time-averaged centreline gas temperature rise and heat-front speed at
     1 m and 2 m above the fuel.
   - Expected: within ×/÷ 2 of the FM Global plume with Qc = (1 − χr)·Q; for Qc 20 kW:
     ≈ 205 K and 3.2 m/s at 1 m, ≈ 64 K and 2.5 m/s at 2 m. 5 and 10 cm agree within
     25%.
   - Source: 50. Today: rise 0.7 / 1.4 m/s; plume ≈ 10× too cool (derived).
8. **H-T8 Three rods: iron, stone, wood** — low
   - Scene: three rods of 5 voxels, side by side in still air, one end of each held at
     300 °C.
   - Measure: temperature rise 10 cm from the hot end after 60 s and 600 s.
   - Expected (upper bounds, no surface loss): at 600 s steel +124 K (pure iron
     +153 K), granite +0.3 to +10 K, wood ≈ 0. At 60 s steel +4 K, iron +16 K, others
     0. Accept iron within ×/÷ 2, stone < +15 K, wood < +2 K, and the order iron ≫
     stone ≫ wood.
   - Source: 23, 31, 33. Today: k ratio 20 : 3 : 2.
9. **H-T9 Conduction law integrity** — high
   - Scene: (a) an air cell at 600 °C with 1000 °C iron on its +x, +y, +z faces, and the
     mirror image. (b) uniform 400 °C air at 5 cm with a ± 10 K checkerboard.
   - Measure: (a) temperatures after one tick in both geometries; (b) the checkerboard
     amplitude over 10 ticks.
   - Expected: (a) no cell ends hotter than its hottest neighbour, and both geometries
     give the same result. (b) the pattern decays.
   - Source: internal invariant (second law). Today: (a) 1763 °C vs 840 °C; (b) grows
     ×1.4 per tick (formula replica). Any retuning of k depends on this test.

### 4.3 `tests/test_realism_fire.py`

Add the HRR readout first: HRR = Σ heat of combustion × mass burned / `TICK_S`.

1. **F-T1 Slab burning rate per area** — high
   - Scene: open sky, stone floor, 5 cm. A wood slab 8 × 8 × 2 voxels
     (0.4 × 0.4 × 0.1 m). Heat its top at ≈ 50 kW/m² for 60 s, then stop.
   - Measure: the actual flux into the top face (ΔE/Δt/area); mass loss and HRR per
     exposed area from 2 to 10 min after ignition.
   - Expected: 2.9–9.6 g/(m²·s), HRR 60–280 kW/m²; accept ×/÷ 2. Measure mass loss,
     not char depth (5–8 mm is smaller than a voxel).
   - Source: 49, 56. Today: ≈ 9 kg/(m²·s) (derived).
2. **F-T2 Two-pallet crib, free burn** — high (slow)
   - Scene: open sky. A slatted crib 1.2 × 1.0 × 0.3 m of ≈ 30 kg of wood, one-voxel
     sticks and gaps. Ignite like NIST: paper and four small wicks, one per quadrant
     (or widen the time-to-peak band).
   - Measure: HRR over time, time to peak, peak HRR, time until flaming stops, fraction
     of mass burned, heat released per kg.
   - Expected: peak 0.5–2 MW at 2–8 min; flaming ends at 20–60 min; ≥ 85% burned;
     14–18 MJ/kg. NIST measured 934 ± 42 kW at 3.93 min, out at 39.52 min, 30.37 of
     33.00 kg, 502 MJ, 16.53 MJ/kg.
   - Source: 57.
3. **F-T3 Lamp spilled on a wooden bed** — high (slow)
   - Scene: `house_fire` as built (room 3.9 × 2.9 × 2.15 m, door 0.5 × 1.85 m), with
     the igniter rescaled to ≈ 100 W, an HRR log and a column of gas thermometers. Run
     ≥ 24,000 ticks.
   - Measure: time for HRR to reach 100 kW; time for the top 0.3 m of gas to reach
     600 °C; time to untenable conditions at 1.5 m (gas > 60 °C, visibility < 5 m or
     O2 < 15%).
   - Expected: HRR < 0.5 MW for the first 3 min. Untenable in ≈ 3–10 min (NIST's
     217 ± 67 s is a fast bound from modern homes). No flashover before 10 min;
     flashover needs ≈ 0.94 MW for this door; legacy rooms took 29.5–34 min or never.
     The game may compress time by at most 2–3×.
   - Source: 69, 70, 80. Today: ≈ 1.1–1.7 MW from wood within 22 s (derived).
4. **F-T4 A sealed room fire goes out near 15% O2** — high
   - Scene: sealed stone box, 2 × 2 × 2 m inside, 5 cm; a 3 kg wood crib lit by a
     small source.
   - Measure: O2 vol% near the fuel and in the room when flaming falls below 5% of its
     peak; wood burned; whether char keeps glowing.
   - Expected: flaming stops at 13–16% O2 (62–76% of fresh). Wood burned ≈ 0.48 kg
     (accept 0.3–0.7 kg) = (0.21 − 0.15)/0.21 × 2224 g O2 ÷ 1.33. Char may smoulder
     on to 4–10% O2.
   - Source: 60, 59. Today: no flaming limit; the existing test only shows O2 below
     ≈ 5 vol%.
5. **F-T5 Smoke and heat layer in a room** — high
   - Scene: the `house_fire` room with the door closed and a steady ≈ 100 kW fire
     (measured HRR) at floor centre. A standing and a crouching person. Run ≥ 2,400
     ticks (60 s), 12,000 for the person check.
   - Measure: smoke and temperature by height band every 5 s; the height where smoke
     falls to half the ceiling value; skin (or air at face height) of both people.
   - Expected: the layer edge reaches 1.5 m in 4–20 s and 1.0 m in 10–46 s (Zukoski
     7–10 s and 17–23 s, ± 40% model error, then ×/÷ 2). At 30 s, top-0.5 m smoke is
     ≥ 3× the bottom-0.5 m smoke, and the ceiling band's temperature rise is ≥ 2× the
     floor band's. The crouching person stays below 45 °C longer than the standing one.
   - Source: 75, 51, 37. Today: flat (305 / 356 / 231 / 286 g).
6. **F-T6 Fire jumps the alley** — medium (slow; needs radiation and flame reach)
   - Scene: two timber houses 1.8 m apart. A fire in a 3.7 × 4.3 × 2.4 m room of
     house A, with a 0.6 × 0.9 m window facing house B.
   - Measure: flux on B's wall at window height and at 4.9 m; time from flames leaving
     A's window to B igniting.
   - Expected: < 30 kW/m² at window height; 20–70 kW/m² (peak ≈ 140) at 4.9 m. B
     ignites 0.5–5 min after window flames appear (NIST: OSB ignited 80 s after flames
     came out; flames touched the wall 9–15 s after the window broke). A campfire under
     100 kW must not ignite B.
   - Source: 39.
7. **F-T7 Dry tree vs fresh tree** — medium
   - Scene: `_grow_tree` at 5 cm, twice: canopy moisture ≈ 10% and ≈ 100% (dry basis).
     ≈ 2 kW under the canopy for 10 s, then off.
   - Measure: time to peak HRR, foliage burned, char depth in the 20 cm trunk, whether
     burning continues after the source stops.
   - Expected: dry: peak 5–30 s after ignition, ≥ 80% of foliage burned, the trunk not
     burned through (char ≤ 2 cm), 35–70% of tree mass left. Fresh: no sustained canopy
     fire. Times ×/÷ 3.
   - Source: 71, 72.
8. **F-T8 Radiant-panel ignition time** — medium (needs radiation across air)
   - Scene: a hot plate (8 × 8 voxels) facing a 0.4 × 0.4 × 0.1 m wood block across a
     one-voxel air gap; plate at 380, 540 and 700 °C (≈ 10, 25, 50 kW/m² as a black
     body).
   - Measure: the actual flux into the wood face; ticks to the first burning voxel.
   - Expected: t_ig within ×/÷ 2.5 of 130·ρ^0.73/(q − 11)^1.82 at the measured flux:
     ≈ 40–300 s at 25 kW/m², 6–50 s at 50 kW/m², no flaming in 20 min at 10 kW/m².
   - Source: 40.
9. **F-T9 A bucket on burning oil vs on burning wood** — high
   - Scene: a 0.3 × 0.3 m pan with 1–2 L of oil burning for ≥ 60 s; pour 0.2–0.5 L of
     water from 30 cm above, under a 2.4 m ceiling. Control: the same water on a wood
     crib of similar HRR.
   - Measure: oil mass outside the pan within 2–3 s; HRR in the 3 s before and after;
     height of burning oil or flame; steam volume.
   - Expected: oil: ≥ 10% of the oil leaves the pan, HRR spikes ≥ 2× within 3 s (a demo
     sequence lasts 2.4 s), a flame column ≥ 1 m ("to the ceiling" is qualitative),
     steam ≈ 1700× the water volume. Wood: HRR falls > 50% within 10 s and stays down.
     Magnitudes ×/÷ 3; the kind of outcome is what counts.
   - Source: 77, 78, 141, 102. Today: a quiet boil under the oil.
10. **F-T10 Smoke blindness per gram burned** — medium
    - Scene: sealed 2 × 2 × 2 m box; burn 400 g of wood; let the smoke mix.
    - Measure: visibility along a 2 m path with the sim's own sight rule (`SMOKE_EXT`,
      `SEE_TAU` 2).
    - Expected: ≈ 0.3 m for structural wood (soot yield 0.015 g/g × 8.7 m²/g); 0.3–1.2 m
      covers yields of 0.004–0.015; accept ×/÷ 2.
    - Source: 74, 58. Today: ≈ 12 m (derived).
11. **F-T11 Oil pool burning rate and flame height** — medium
    - Scene: open sky; a 0.3 × 0.3 m stone tray with 2 L of oil, lit by a small source.
      Second case: 0.3 L spilled on a stone floor.
    - Measure: steady burning rate per area; burnout time; flame height (needs flames
      above the fuel).
    - Expected: tray 10–30 g/(m²·s) (kerosene ≈ 25; vegetable oil lower); burnout
      10–40 min; HRR ≈ 50–100 kW; flame ≈ 0.8–1.2 m (Heskestad). Floor spill: 0.2–0.8 ×
      the pool rate. Accept ×/÷ 2.
    - Source: 66, 79. Today: ≈ 4 kg/(m²·s), flame height 0.
12. **F-T12 Heat per gram of O2** — low (static check on the tables)
    - Measure: heat ÷ `O2_RATIO` for every burnable SOLID and FLUID row.
    - Expected: WOOD, LEAF, OIL 13.1 ± 10% (11.8–14.4 kJ/g O2); CHAR 11.3–12.5; wood O2
      demand 1.30–1.36 g/g; carbon 2.6–2.7 g/g.
    - Source: 55, 58, 61. Today: FLESH 5.7 fails; fix `O2_RATIO[FLESH]` ≈ 0.6.
13. **F-T13 Flame temperature ceiling** — low
    - Scene: the steady crib from F-T2, and a 1 m oil pool.
    - Measure: hottest gas or fuel temperature in steady burning; flame-tip temperature.
    - Expected: flame zone 800–1000 °C for fires under 1 m, up to 1100–1200 °C for large
      pools; never above ≈ 1250 °C; flame tips 300–550 °C.
    - Source: 67.

### 4.4 `tests/test_realism_fluids.py`

1. **L-T1 Molten lead splat on stone** — high
   - Scene: pour 100 ml of MLEAD at 400 °C from 20 cm onto a cold stone floor. Run at 5
     and 10 cm.
   - Measure: final solid area and mean thickness; time from first contact to solid.
   - Expected: thickness 2.9–4.0 mm (from λc = 2.07 mm, contact angle 90–150°), area
     0.025–0.035 m². Accept 2–6 mm, 0.017–0.05 m², and the same result within ± 50% at
     both sizes. Solid in 1–60 s (derived; ≈ 6 s for a contact estimate).
   - Source: 87, 95. Today: spreads like water (L7).
2. **L-T2 A bucket spilled on the floor** — high
   - Scene: tip 1 L and 10 L of water onto a level, smooth stone floor. Run at 5 and
     10 cm.
   - Measure: wetted area (cells with ≥ 0.1 mm), mean depth, time to 90% of the final
     area.
   - Expected: depth 1–4 mm, so 0.25–1.0 m² per L. Area per litre is the same for 1 L
     and 10 L within ± 30%. 5 and 10 cm agree within ± 30%. Most of the area within
     30 s, ≥ 90% by ≈ 300 s.
   - Source: 66, 95, 96. Today: 0.71 m²/L at 5 cm, 2.85 at 10 cm; area ∝ V^(2/3);
     90% in 1.5 s (derived).
3. **L-T3 Lamp oil spill area** — low
   - Scene: pour 0.5 L of OIL onto a smooth stone floor at 20 °C.
   - Measure: final area and mean depth.
   - Expected: 0.2–1.2 m², depth 0.4–2.5 mm (lube oil: 0.57 m² at 0.88 mm on smooth
     concrete, 0.22 m² at 2.30 mm on coated concrete). Area changes ≤ 30% with voxel size.
   - Source: 66.
4. **L-T4 Water on a hot plate** — medium
   - Scene: an iron plate held at 110, 130, 170, 200 and 250 °C, with one 125 ml water
     cell on it.
   - Measure: time for the cell to boil dry at each temperature.
   - Expected: heat flux at most ≈ 1 MW/m² (≈ 2.5 kW per 5 cm face), so boil-dry at
     130 °C takes ≥ ≈ 2 min; accept 1–10 min. Above 193 °C the cell lasts **longer**
     than at 150–170 °C (the drop data: almost instant at 168 °C, 152 s at 202 °C).
   - Source: 99, 100, 101. Today: ≈ 20–30 s at 130 °C, shorter as the plate gets
     hotter (derived).
5. **L-T5 Molten lead into a water bucket** — medium
   - Scene: closed and insulated (heat loss off). 1 L of water at 20 °C in a glass
     vessel; drop in 100 g of MLEAD at 400 °C.
   - Measure: final water temperature; lead state; water boiled at first contact;
     energy closure.
   - Expected: the lead ends solid; the water ends at 21.8 ± 0.3 °C (≈ 7.4 kJ, less the
     vessel's share); energy closes within 5%; some local boiling and steam at first
     contact.
   - Source: 87, 84, 98. Today: heat crosses lead–water faces only through air k; with
     heat loss off the water creeps toward ≈ 21.5–21.8 °C; no steam.
6. **L-T6 Water cooling energy balance** — low
   - Scene: an inert stone block at 600 °C; pour 500 ml of water onto it.
   - Measure: grams of water boiled against the energy taken from the stone.
   - Expected: boiled mass = ΔE / 2256 J/g, after 4.2 J/(g·K) to warm the water to
     100 °C, within ± 5%. The water never goes above ≈ 100 °C.
   - Source: 76.
7. **L-T7 Shed blood gels and dries** — medium (slow)
   - Scene: pour 50 ml of BLOOD onto a wooden floor at 20 °C.
   - Measure: time until the pool stops flowing; time until it is dry; pool thickness.
   - Expected: gels in 1–30 min (drops clot 2–8 min after landing; Lee-White ≤ 20 min).
     Dries in 1–24 h (a game band, unverified; a ≤ 9 ml tile pool was still drying at
     8 h 37 min). Thickness ≈ 1.44 ± 0.19 mm on a hard floor.
   - Source: 115, 114. Today: never gels or dries.
8. **L-T8 A splash obeys g** — medium
   - Scene: drop a 15 cm stone block into a 30 cm pond. Run at 5 and 10 cm.
   - Measure: fitted vertical acceleration of the droplet parcels in `self.drops`.
   - Expected: 9.81 m/s² ± 10% at both sizes.
   - Source: 97. Today: 20 m/s² (5 cm), 40 m/s² (10 cm).
9. **L-T9 A puddle evaporates** — low (slow)
   - Scene: a 1 m² film of water 1 mm deep at 20 °C, air at 0.5 m/s, humidity ratio
     0.0098 kg/kg, 2 h of world time.
   - Measure: water lost per m² per hour.
   - Expected: 0.169 kg/(m²·h) ± 50%.
   - Source: 104. Today: 0.
10. **L-T10 Wood soaks up a puddle** — low
    - Scene: a 2 mm puddle on a pine floor (side grain); a pine post standing in water
      (end grain); 10 min.
    - Measure: water taken into the wood per m².
    - Expected: m = Aw·√t: side grain 25–170 g/m², end grain 245–390 g/m²; ×/÷ 2.
    - Source: 109. Today: 0.
11. **L-T11 Tin and lead make solder** — low
    - Scene: 63 g of tin and 37 g of lead in a crucible, heated to 260 °C, then cooled.
    - Measure: one liquid or two; freezing temperature.
    - Expected: one liquid that freezes at 183 ± 3 °C.
    - Source: 108. Today: two layers that freeze at ≈ 230 and ≈ 325 °C.
12. **L-T12 Pond ice** — low (slow)
    - Scene: a 30 cm pond that starts at 0–1 °C, air held at −10 °C for 24 h.
    - Measure: ice thickness.
    - Expected: 2–11 cm (Stefan upper bound 11 cm; real ≈ 0.7 of that).
    - Source: 105. Today: temperature cannot fall below 0 °C.

### 4.5 `tests/test_realism_chemistry.py`

1. **C-T1 Vitriol on skin** — high
   - Scene: 5 cm. A person's foot in a one-voxel ACID puddle (potency 1.0), or a bare
     FLESH slab 4 × 4 × 2 under the same pool. Run ≥ 7,200 ticks (3 min); ideally 60 min.
   - Measure: flesh voxels turned to AIR; flesh grams lost; burn depth (needs a new
     per-face burn-depth quantity).
   - Expected: no flesh voxel removed (the dead layer stays). Burn depth 0.1–1.3 mm at
     3 min and ≥ 1.3 mm by 60 min of contact (Packing Group II: full thickness in more
     than 3 and up to 60 min). Depth keeps growing while the acid stays. Any voxel
     removed within 10 min fails. Until the body-parts model exists, rulings case 2
     applies: the pass is "no flesh removed".
   - Source: 119, 120, 121, 122, 123, 125. Today: ≈ 24 mm/s; a voxel is gone in ≈ 2–3 s.
2. **C-T2 Vinegar hand soak** — high
   - Scene: a FLESH slab or a hand in WEAK_ACID (potency 1.0) for 2,000 ticks;
     extrapolate to 4 h.
   - Measure: flesh grams lost; voxels removed.
   - Expected: 0 g lost. Mild redness at 10% is acceptable once a skin state exists.
   - Source: 126. Today: ≈ 2 mm/s.
3. **C-T3 Vinegar cask** — high
   - Scene: an open wooden tub with one-voxel walls and a 3 × 3 × 3 cavity full of
     WEAK_ACID. 2,000 ticks.
   - Measure: wood mass loss per game day (extrapolated); leaks.
   - Expected: < 0.1% of the wood per game day; no leak.
   - Source: 128, 129. Today: a wall goes in ≈ 25 s.
4. **C-T4 Vitriol on a plank** — high
   - Scene: a 5 cm wood slab 4 × 4 × 1 under a one-voxel ACID pool (potency 1.0).
     2,400 ticks.
   - Measure: wood recession; share of reacted mass left as CHAR; smoke made; when the
     plank fails.
   - Expected: the surface turns black within seconds; recession ≤ 1 mm/min (generous);
     ≥ 20% of the reacted mass stays as char (species dependent); the plank holds for
     ≥ 10 game minutes.
   - Source: 94, 130, 128. Today: ≈ 20 mm/s, no residue.
5. **C-T5 Water into vitriol, and the acids mix** — high
   - Scene: case A: an equal volume of WATER poured onto ACID (potency 1). Case B: ACID
     poured into WATER. Case C: WEAK_ACID poured onto ACID. 200 ticks each.
   - Measure: energy released per ml of acid; boiling or spatter at the interface;
     whether the liquids blend.
   - Expected: energy = n·[ΔHm(n_final) − ΔHm(n_initial)] from Table B.11: ≈ 0.86 kJ per
     ml of 98% acid at 1:1 (± 30%), ≈ 1.77 kJ/ml at full dilution. Case A boils and
     spits at the interface; case B does not. This needs a mixing rate; today blends are
     instant. Case C: the two acids blend into one liquid.
   - Source: 139, 94. Today: no heat; ACID and WEAK_ACID stay as separate layers.
6. **C-T6 Acid on iron: concentration sweep** — high
   - Scene: three iron slabs 4 × 4 × 1 under one-voxel ACID pools at potency 1.0
     (98 wt%), 0.5 (≈ 64 wt%) and 0.2 (≈ 31 wt%). 2,400 ticks each.
   - Measure: iron loss in mg/(cm²·h); gas made.
   - Expected: potency 1.0 ≤ 0.05 mg/(cm²·h) (< 0.5 mm/yr). Potency 0.5 also slow below
     38 °C. Potency 0.2 and below fast: 0.7–600 mg/(cm²·h). The dilute rate is ≥ 10× the
     concentrated rate. Visible H₂.
   - Source: 131, 132, 133. Today: 432,000 mg/(cm²·h) at potency 1.0, linear in potency.
7. **C-T7 Limestone vs granite** — medium (needs a STONE split)
   - Scene: two 2 × 2 × 2 stone blocks, one limestone, one granite; once with HCl-class
     acid at ≈ 1 M, once with 98% vitriol. 2,400 ticks.
   - Measure: mass loss in mg/(cm²·s) for each; change of rate over time.
   - Expected: limestone in 1 M HCl 0.47 mg/(cm²·s) ×/÷ 3. Granite ≤ 1% of limestone
     (unverified for strong acid). Limestone in vitriol: the rate falls as gypsum forms
     (time constant unverified; test only that it falls).
   - Source: 135, 136, 137. Today: 8 mg/(cm²·s) for any stone, steady.
8. **C-T8 Fumes ledger** — medium
   - Scene: dissolve 100 g of iron and, separately, 100 g of limestone in an ACID pool.
   - Measure: smoke made; gas made by species; mass held in the liquid.
   - Expected: smoke ≤ 5% of the dissolved mass. Iron gives ≈ 3.6 g of H₂ (≈ 40 L at
     0 °C), flammable. Limestone gives ≈ 44 g of CO₂ (≈ 22 L). The rest stays in the
     liquid (or as solid gypsum with sulfuric acid).
   - Source: 94, 122, 134. Today: all dissolved mass becomes smoke.
9. **C-T9 Neutralise a spill with limestone** — low
   - Scene: 500 ml of ACID (potency 1) over stone rubble until the acid is spent or stops.
   - Measure: ml of acid used per g of stone; heat per g.
   - Expected: at most 0.54 ml/g ± 20% for 98% H2SO4 (a gypsum crust may stop it early;
     the product is solid gypsum); 1.67 ml/g if the acid stands for 37% HCl. Heat
     ≈ 141–152 J/g ± 50%.
   - Source: 94, 134. Today: 0.6 ml/g (passes on spend).
10. **C-T10 Heat of dissolving iron** — low
    - Scene: an iron slab under an acid pool; dissolve exactly 100 g.
    - Measure: energy added to the pool per g of iron.
    - Expected: ≈ 1.6 kJ/g ± 30%.
    - Source: 134. Today: 250 J/g.
11. **C-T11 Vitriol in glass and lead vessels** — low
    - Scene: a GLASS vat and a LEAD vat, each holding ACID (potency 1), 2,000 ticks.
    - Expected: 0 g of wall lost.
    - Source: 94. Today: passes. Regression guard.
12. **C-T12 Vinegar on a marble altar** — low (needs a STONE split)
    - Scene: a polished limestone or marble block wetted with WEAK_ACID; a granite block
      as control.
    - Expected: visible etch and fizz on marble within 60 s with < 0.01 mm lost; granite
      unchanged.
    - Source: 136, 127. Today: no reaction.
13. **C-T13 Rust on a wet blade** — low (slow)
    - Scene: a wet IRON voxel in open air for one game week.
    - Expected: ≤ 0.1% of the mass lost (C3: 25–50 µm per year). An orange skin is
      acceptable.
    - Source: 140. Today: passes on mass.
14. **C-T14 Molten metal meets water** — low
    - Scene: (a) molten lead poured in drops into a water tub; (b) water trapped under a
      pool of molten lead (a wet mould).
    - Expected: (a) quench and boiling, no explosion (this is how lead shot was made);
      (b) violent steam expansion that throws metal. Graded by kind only.
    - Source: 143, 103. Today: neither.

### 4.6 `tests/test_realism_materials.py`

1. **S-T1 Stone knee-high drop** — high
   - Scene: a 20 × 20 × 20 cm stone block, bound as one cluster, dropped flat onto a
     stone floor from 0.05, 0.2, 0.5 and 1.0 m. Run once through the MPM path
     (`free_fall_to_seam=True`; the default is False, sim.py:1000) and once on the lattice.
   - Measure: share of particles damaged or separated; lattice cells shattered.
   - Expected: none at 0.05 and 0.2 m; ≤ 5% (edge and corner chips) at 0.5 and 1.0 m.
     The two paths agree within 2× on the height where damage starts. (A 1 m landing
     makes ≈ 21–43 MPa of compression against a UCS of 49–324 MPa.)
   - Source: 150, 151. Today: MPM damage from 1.6–4.7 cm (derived); lattice 1.6–2.9 m.
2. **S-T2 A felled tree stays whole** — high
   - Scene: a notched 2.75 m tree (as `tree_hinge`) and a 6 m version, with a hinge 80%
     of the diameter long and 10% thick, falling onto flat ground.
   - Measure: trunk bonds parted before the tip touches the ground; time from a 5° lean
     to landing; tip speed at contact; share of the trunk detached after landing.
   - Expected: no trunk bonds parted in the air (at most the weakest few % near 1/3 of
     the height). Fall time 1.3–2.0 s for 2.75 m (rigid rod 1.57 s; a heavy canopy is up
     to √1.5 slower) ± 20%. Tip ≈ 8–9 m/s. After landing, ≤ 5% of the trunk detached
     (extrapolated from 4–14% for large old-growth trees).
   - Source: 164, 8, 165, 166. Today: "breaks into chunks on the way down" (roadmap 94,
     before shape matching).
3. **S-T3 Plank bridge** — high
   - Scene: a wooden plank 25 cm wide and 5 cm thick (and a 2.5 cm board drawn at fill
     0.5) on two stone blocks across clear gaps of 2, 3 and 4 m. First empty, then with
     an 80 kg person at mid-span for 10 s.
   - Measure: does it stay up; mid-span sag; does it break.
   - Expected: empty, all spans hold. 5 cm plank with the person: sag ≤ 1 cm at 2 m,
     1–2 cm at 3 m, 3–5 cm at 4 m; none break. 2.5 cm board: 3 m sags 10–16 cm and
     holds; 4 m breaks or sags > 25 cm. Sag ×/÷ 2; break or no break must be right at
     2 m and 4 m. The 2.5 cm case needs fill carried into MPM geometry.
   - Source: 162, 10, 163. Today: gaps up to 2.4 m hold, 3–4 m lose the middle, load
     changes nothing.
4. **S-T4 Bottle drop** — high
   - Scene: a glass bottle drawn at its true wall fill, **filled** with water, dropped on
     its side onto stone from 0.2, 0.35, 0.5, 0.75 and 1.0 m; then across a stone or
     iron edge; then into a 30 cm pond from 1 m. Run as a straight lattice fall and as an
     MPM topple off a shelf, at 5 and 10 cm.
   - Measure: intact or broken at each height; the first height that breaks it.
   - Expected: flat: intact at 0.2 m, broken by 0.45–0.56 m (± 0.15 m), always broken at
     1.0 m. On an edge: breaks at 0.30–0.43 m. Into the pond: intact. Both paths and both
     voxel sizes agree within 0.2 m.
   - Source: 17. Today: a full glass cell breaks at 0.11–0.21 m on any solid floor; at
     true wall fill both paths become too tough; contents add no impact energy. The pond
     case passes.
5. **S-T5 Metals keep their shape, and nothing crashes** — high
   - Scene: a 10 × 10 × 5 cm lead ingot and a tin ingot on the floor; a 1 m × 5 × 5 cm
     iron bar across supports 0.9 m apart; a char cell and an ash cell. Promote all to
     MPM and run 400 ticks.
   - Measure: exceptions on promote; ingot height loss; permanent sag of the bar.
   - Expected: no `ValueError` for tin, char or ash. Height loss and sag ≤ 1 mm
     (self-weight stress 5.6 kPa in lead and ≈ 1 MPa in iron is far below real yield).
   - Source: 84, 85, 153, 157. Today: tin, char and ash crash; lead and iron predicted
     to slump and sag.
6. **S-T6 Stone slab overhang** — low
   - Scene: 5 cm stone slabs cantilevered 1 m and 2 m from a wall, no load. Parameterise
     by stone grade.
   - Expected: 1 m holds (≈ 1.6 MPa). 2 m (≈ 6.4 MPa) breaks the weakest stone that meets
     the spec (low-density limestone, MOR ≥ 2.8 MPa) and holds for granite-class stone
     (≥ 10.3 MPa).
   - Source: 11. Today: always holds.
7. **S-T7 Glass shelf under a pot** — medium
   - Scene: a glass shelf 1 m × 30 cm × 1 cm (fill 0.2) on brackets 0.9 m apart, bare and
     then with a 5 kg pot at the centre, 10 s.
   - Measure: cracked particles or shattered cells; the MPM member's thickness.
   - Expected: no break (≈ 1.5 MPa bare plus ≈ 2.2 MPa from the pot, against 45 MPa).
     The MPM shelf must be 1 cm thick; a fill-0.2 cell now promotes as a 5 cm slab at a
     fifth of the density, which would pass for the wrong reason.
   - Source: 144, 162. Today: scaled glass strength 0.23–1 MPa (derived).
8. **S-T8 Hammer on a stick, in steps** — medium
   - Scene: a 22.7 kg weight dropped on the middle of one 5 × 5 cm wooden stick on
     supports ≈ 71 cm apart (geometry unverified), from 0.2 m up in 0.1 m steps, on the
     same stick, until it breaks or sags 152 mm.
   - Measure: the height of the last drop.
   - Expected: with one WOOD row, 0.3–1.2 m. With per-species rows: pine (0.46 m) <
     Douglas-fir (0.79 m) < oak (1.09 m). Use the heights as a rank order, not as a
     one-blow energy.
   - Source: 10.
9. **S-T9 A blow to the shin** — medium (needs a bone layer)
   - Scene: a standing person struck mid-shin by a 9.5 kg mass at 5.55 m/s (≈ 146 J),
     and by a 2 kg club at 4, 6 and 8 m/s.
   - Measure: tibia bending moment (from the striker's effective mass); whether a
     fracture state appears (the leg no longer bears load).
   - Expected: fracture above ≈ 400 N·m (dynamic; ≈ 240 N·m static). The 146 J case
     breaks the leg. The club at 4–6 m/s bruises only; at 8 m/s it is uncertain.
   - Source: 160. Today: no fracture state.
10. **S-T10 Hand against board, post and pane** — low
    - Scene: a fist at 4 and 10 m/s on (a) a thin pine board on supports 25 cm apart,
      (b) a 10 × 10 cm fence post, (c) a 4 mm window pane in a frame.
    - Expected: the board breaks above ≈ 5 J (secondary; thickness not given); the post
      never breaks; the pane breaks at both speeds.
    - Source: 170, 148.
11. **S-T11 A dropped plank settles** — low
    - Scene: a 1 m bound wooden plank dropped flat 10 cm; repeat unbound as a check.
    - Measure: largest particle deviation from the rigid fit after 0.1 s; time to rest.
    - Expected: bound: deviation < 1 cm and at rest within 0.5 s. Unbound at stiff 1e-4 is
      expected to fail; record it as the reason to keep rig ≥ 0.9.
    - Source: 10.
12. **S-T12 Thin ice breaks** — low (needs ICE)
    - Scene: a person walks onto pond ice 2, 5 and 10 cm thick.
    - Measure: bending stress in the ice; does it break.
    - Expected: it breaks when the stress passes 0.5–2.2 MPa (lake-ice averages 0.59 MPa
      cantilever, 0.78–0.79 MPa three-point).
    - Source: 169.

### 4.7 `tests/test_realism_body.py`

1. **B-T1 Drowning clock** — high (needs a body O2 store)
   - Scene: a person standing in the pond with the head 20 cm under water; a second copy
     unconscious face-down in a 10 cm puddle. Pull one copy out at 90 s.
   - Measure: ticks to "slumps" and to "stops breathing"; whether the copy pulled out at
     90 s comes round.
   - Expected: unconscious at 60–180 s (accept 30–240 s); dead at 4–10 min (accept
     3–15 min); a body pulled out before ≈ 4 min survives. A face pressed into a solid
     follows the same breath-hold clock.
   - Source: 194. Today: nobody drowns; a buried face dies in 1.7 s.
2. **B-T2 A room with little oxygen** — high
   - Scene: a sealed 3 × 3 × 2.5 m room with the O2 field set to 14%, 10% and 5% of air;
     a person standing inside. One run per level.
   - Measure: time to faint and to death.
   - Expected: 14%: no faint within 10 min. 10%: faint within 1–10 min. 5%: coma in
     10–60 s; death not before ≈ 2 min.
   - Source: 193. Today: at 5% faint 1.1 s, death 5.3 s (derived).
3. **B-T3 Bleeding out from the thigh** — high
   - Scene: an edged blow (declared 2e-4 m² edge) deep into the thigh, through the main
     vessel, no aid. A second copy gets a shallow 1 cm cut.
   - Measure: faint time, death time, blood fraction lost at each; total blood conserved.
   - Expected: deep cut: faint in 1–5 min, death in 3–15 min (case report: < 10 min).
     Faint at 30–45% lost, death at 40–60%. With 3 L of blood and 6–32 ml/s, class III
     (900–1200 ml) comes in 0.5–3.5 min (derived; "a cut artery bleeds like its intact
     flow" is an unverified assumption). Shallow cut: stops by itself with < 5% lost.
   - Source: 184, 183, 187, 116. Today: faint 13.2 min, death 23.6 min (derived).
4. **B-T4 A small cut stops by itself** — high
   - Scene: a standing person with one wound 10 mm long and 1 mm deep on the forearm; no
     pressure.
   - Measure: time until `w['open']` is False; blood shed.
   - Expected: 3–10 min (accept 1–15 min).
   - Source: 113. Today: 5.9 s.
5. **B-T5 Pressure and tourniquet** — high (needs the actions)
   - Scene: the deep thigh wound of B-T3. A second person presses on it at 30 s; in a
     variant, puts a tourniquet above it.
   - Measure: survival; blood lost; time until flow is under 0.1 ml/s.
   - Expected: pressed or tied before collapse: survives with < 30% lost (accept < 35%).
     A tourniquet cuts limb flow to < 5% at once. The untreated copy dies.
   - Source: 186. Today: no action sets `w['pressed']`.
6. **B-T6 Fall ladder, feet first** — high
   - Scene: a person dropped rigid and feet first onto stone from 1, 2, 3, 4.5, 6, 10, 15
     and 25 m, in a world wide enough to lie down in. A head-first drop from 4 m.
   - Measure: hurt, awake and alive after landing.
   - Expected: conscious at 3 m and below (maybe injured); seriously hurt at 6 m; the
     50% death height falls in 8–20 m (≈ 12–15 m, secondary); dead at 25 m. Head first
     from 4 m: dead or unconscious.
   - Source: 22, 197, 198. Today: knocked out at ≈ 2.0 m, dead at ≈ 3.45 m (derived).
7. **B-T7 Walk and flee 20 m** — high
   - Scene: a flat 20 m corridor. A person walks to the far door, along a grid axis and
     along a diagonal; then flees the same distance from a fire. Run at 5 and 10 cm.
   - Measure: mean speed of each.
   - Expected: walk 1.2–1.5 m/s (accept 1.1–1.65) on both routes; flee ≥ 2.1 m/s, which
     needs a running gait (accept 2.1–7 m/s; ordinary run speeds unverified). The same at
     both voxel sizes.
   - Source: 172, 173, 175. Today: 0.67 m/s (axis), ≈ 0.47 m/s (diagonal) at 5 cm;
     1.33 m/s at 10 cm; fleeing is no faster.
8. **B-T8 Smoke in the bedroom, and the rescue** — high (slow; needs a CO dose)
   - Scene: the `house_fire` bedroom with the bed burning. One occupant stands at the far
     wall, one crawls. A rescuer carries the standing one out when they collapse.
   - Measure: time from flaming ignition to collapse for each; time to come round in
     clean air.
   - Expected: the standing occupant collapses 3–6 min after ignition (accept 2–10 min);
     the crawler lasts longer; the rescued victim stays down for ≥ 1 min (CO half-life
     320 min on air). Dose per litre, not per voxel.
   - Source: 191, 196. Today: smoke load 0.00 at 21 s; recovery ≈ 5–12 s (derived).
9. **B-T9 Standing jump** — medium
   - Scene: a person jumps straight up from a stone floor, nothing overhead.
   - Measure: takeoff speed; rise of the mean height of the flesh cells; airtime.
   - Expected: rise 0.25–0.45 m (takeoff 2.2–3.0 m/s) ± 25%; airtime 0.45–0.6 s.
   - Source: 176, 15. Today: 0.565 m (derived).
10. **B-T10 Leap a gap** — low
    - Scene: a person on a ledge facing gaps of 0.5, 1.0, 1.5 and 2.0 m to a ledge of the
      same height.
    - Measure: the widest gap offered as a leap and cleared.
    - Expected: clears 1.0 m; 1.5 m is borderline; never 2.0 m. COM travel 1.2–1.8 m.
    - Source: 16.
11. **B-T11 Throw ladder** — medium
    - Scene: the same thrower throws 0.06, 0.16, 0.73, 1.0, 1.95 and 3.4 kg objects, with
      a wind-up, on a flat 30 m field.
    - Measure: release speed for each mass; carry of the 1 kg stone to first contact.
    - Expected: ≈ 19.5 m/s at 57 g and ≈ 8.5 m/s at 3.4 kg (one thrower); 17.2–21.2 m/s at
      0.16 kg, 11.5–14.5 m/s at 1 kg, 9.0–11.6 m/s at 1.95 kg (Table III fits); ± 30% per
      point. 1 kg carry 10–18 m (accept ≥ 7 m).
    - Source: 14. Today: ≈ 4.9 m/s at 1 kg (old body), 6.75 m/s ceiling. Re-measure on
      the current body first.
12. **B-T12 Punch speed and knockout** — medium (needs a head pathway)
    - Scene: a fist swung into a wooden post; then 10 punches to an opponent's head.
    - Measure: peak knuckle speed; energy delivered; head angular acceleration; stun or
      knockout.
    - Expected: knuckle 5–10 m/s (accept 4–11); energy delivered ≈ 17 ± 9 J per punch.
      A knockout is possible, keyed on head angular acceleration (≈ 2000–6000 rad/s²,
      secondary thresholds), not on joules.
    - Source: 180. Today: no knockout is possible; speed not measured on the current arm
      (old arm 3.8 m/s).
13. **B-T13 Hot surface and hot water** — medium (needs a skin layer and a 37 °C core)
    - Scene: a hand held against iron, and dipped in water, held at 45, 49, 55, 60, 65
      and 70 °C.
    - Measure: ticks until "feels it" and until burn damage starts on that hand.
    - Expected: felt at once at ≥ 43–45 °C. Burn starts: 49 °C 8–10 min, 55 °C 17–30 s,
      60 °C 3–5 s, 65 °C ≈ 2 s, 70 °C ≤ 1 s (accept ×/÷ 2). The rate rises steeply
      (≈ 1.8× per °C), and the damage stays on the hand.
    - Source: 188, 190, 42, 41. Today: no damage below 55 °C; damage averaged over the
      whole body.
14. **B-T14 Hot air, no flame** — medium
    - Scene: a person in a closed room with the air held at 60, 100, 120, 150 and 200 °C,
      no radiant source.
    - Measure: time to incapacitation.
    - Expected: between the ISO 13571 curve (7.9 / 4.3 / 2.0 / 0.8 min at
      100 / 120 / 150 / 200 °C) and the Fire Service Academy table (15 / 7 / 5 / 2 min),
      ×/÷ 1.5. 60 °C: no incapacitation within 30 min.
    - Source: 38, 191. Today: no hot-air path; with air holding 41× its real heat the
      result would be far too harsh.
15. **B-T15 Segment masses** — low
    - Scene: build `scenes._person` and weigh each bone from `smass`.
    - Expected: each arm 4–6%, each leg 14–18%, trunk 45–52%, head 6–9% (± 30% of Winter).
    - Source: 171. Today: arm 13.1%, leg 10.8%.
16. **B-T16 Lift and hold** — low
    - Scene: lift 20, 40 and 80 kg sacks; hold a 4 kg hammer level at arm's length; hold
      the arm out at 90° and 160° with a load.
    - Expected: 20 kg lifted; 40 kg marginal and 80 kg not lifted (unsourced game
      targets). Shoulder hold 43–77 N·m by angle (accept 40–90). Swing and hold use one
      number.
    - Source: 177, 179. Today: swing 60 N·m, hold ≈ 119 N·m.

### 4.8 `tests/test_realism_sound.py`

1. **A-T1 `_hears` agrees with `_law_hear`** — high
   - Scene: two standing people 3 m apart in an open room, along ±x, ±y and both
     diagonals; then 30 m apart outdoors.
   - Measure: `_level` as used by `_hears` and as used by `_law_hear`; whether a spoken
     line is understood.
   - Expected: both give 90 − 9.5 = 80.5 ± 1 dB at 3 m in every direction, and ≥ 40 dB
     at 30 m. A line is understood at 3 m. (The existing heard-line test uses 0.5 m,
     where the 1 m clamp hides the fault.)
   - Source: internal consistency; roadmap 97. Today: `_hears` gives ≈ 34–40 dB at 3 m
     (formula replica).
2. **A-T2 Around an open doorway** — high
   - Scene: two rooms with a 10 cm stone wall between them and an open doorway 1 m wide.
     The shouter is in room A, 2 m from the doorway, with no line of sight. The listener
     is in room B, 3 m past the doorway, around the corner.
   - Measure: level at the listener; whether `_hears` is True.
   - Expected: 50–68 dB ± 10 dB (a rough band; the room corrections may double-count
     distance). `_hears` is True.
   - Source: 205, 214, 207. Today: ≈ 20 dB through `_level`, less through `_hears`.
3. **A-T3 A shout through a closed door wakes a sleeper** — high
   - Scene: a bedroom with a closed 5 cm plank door, with and without a one-voxel gap
     under it. The shouter stands 1 m outside and shouts "Fire!" again and again. An
     unimpaired adult sleeps 3 m inside.
   - Measure: door insertion loss; tick at which the sleeper wakes.
   - Expected: door loss 10–25 dB ± 5; level at the pillow 50–70 dB; the sleeper wakes
     within 30 s of the first shout.
   - Source: 205, 212, 210. Today: the door costs 36.5 dB; at most 53.5 dB reaches the
     room; the sleeper never wakes.
4. **A-T4 A bystander is not a wall** — high
   - Scene: a tavern. Shouter and listener 4 m apart, with one and then two people on the
     line between them.
   - Measure: drop in level caused by the bystanders.
   - Expected: 0–6 dB for one (a head-shadow proxy, unverified as insertion loss); ≤ 10 dB
     for two. The line is still understood at 4 m.
   - Source: 215, 214. Today: 46–50 dB per bystander's head.
5. **A-T5 A fire wakes a sleeper by sound, not smell** — high (needs fire sound)
   - Scene: an adult asleep, door shut. (a) Smoke only, from the next room, no flame in
     the room. (b) A wood fire lit 3 m away in the same room. Add a child sleeper.
   - Measure: whether and when each sleeper wakes, and what woke them; the fire's level
     at the pillow.
   - Expected: (a) no waking by smell. (b) most adult runs wake from the crackle well
     before ≈ 217 s (NIST time to untenable, flaming fires, from ignition). The child
     mostly sleeps on. The crackle's real level is unverified; the lab played 42–48 dBA
     and woke 91%.
   - Source: 204, 247, 248, 80. Today: (a) passes; (b) the sleeper never wakes unless
     something falls or breaks.
6. **A-T6 A hedge does not silence a shout** — high
   - Scene: two people 4 m apart outdoors with the scene hedge (LEAF frac 0.3, 4 voxels
     deep), and then a 1 m dense hedge, between them, blocking the view.
   - Measure: drop in level caused by the hedge.
   - Expected: 0–3 dB.
   - Source: 207. Today: 28.6 dB (scene hedge) to 42.5 dB.
7. **A-T7 Wall loss by mass** — medium
   - Scene: shouter and listener 2 m apart with one partition between them: (a) 5 cm
     pine, (b) 10 cm stone, (c) 20 cm stone.
   - Expected: (a) 28–34 dB, (b) 44–52 dB, (c) 52–58 dB; ± 5 dB.
   - Source: 209, 210, 211. Today: 36.5, 55.6, 61.6 dB.
8. **A-T8 A gap under the door** — medium
   - Scene: the pine door of A-T3, (i) airtight and (ii) with a one-voxel gap at the
     bottom (2–3% of its area). The listener is off the line of the gap.
   - Expected: (i) 25–37 dB; (ii) 12–20 dB.
   - Source: 213, 210. Today: the gap changes the loss by < 1 dB.
9. **A-T9 A brawl masks speech** — medium (needs ambient noise)
   - Scene: a tavern with a brawl or crowd at 70–75 dBA. A shout from 5 m; a normal voice
     (58 dBA at 1 m) from 2 m.
   - Measure: SNR at the listener; whether each line is understood.
   - Expected: the shout (SNR ≈ 0 to +6 dB) is understood. The normal voice (SNR ≈ −18 to
     −23 dB) is not. Lines start to fail below ≈ −3 dB SNR.
   - Source: 200, 199. Today: no masking; every line is heard.
10. **A-T10 How far a shout carries outdoors** — medium
    - Scene: open ground. Evaluate `_level` analytically at 25, 50, 100, 200, 300 and
      400 m, with the ambient level as a parameter (30, 40, 50 dB).
    - Expected: understood to ≈ 100–300 m at 30–40 dB ambient, to ≈ 40 m at 50 dB, much
      less in a tavern; d_i = 5619·e^(−0.0978·dB_amb) m ± 50% (forest data).
    - Source: 221, 219, 220. Today: 316 m whatever the noise.
11. **A-T11 Kinds of sleeper** — medium
    - Scene: one sleeper exposed to a 30 s sound at the pillow of 50, 60, 75 and 85 dBA.
      An unimpaired adult, a child, a sleep-deprived adult.
    - Expected: adult: most wake at 55–60 dBA (voice-type signal), nearly all by 75.
      Child (6–17): ≥ 80% sleep through 60 dBA. Sleep-deprived adult: ≥ 20% fail at 60.
      (Drink: unverified.) Waking takes seconds to a minute (95% within 32 s), not one tick.
    - Source: 205, 247. Today: one 50 ms sample at 60 dB wakes anyone at once.
12. **A-T12 Swung metal rings as metal** — medium
    - Scene: sword on sword, and a hammer on an anvil, at 20–100 J.
    - Expected: ≥ 95 dB at 1 m, consistent with iron ring and the metal-drop proxy
      (105–110 dBA).
    - Source: 203. Today: 68–75 dB, because the swing rings as FLESH (derived).
13. **A-T13 A metal drop** — low
    - Scene: a 5 kg iron block dropped 1 m onto an iron anvil; listener 1–2 m away.
    - Expected: 105–115 dB at 1 m ± 6 dB (the proxy is a level over time, not one drop).
    - Source: 203. Today: ≈ 112 dB. Regression guard.
14. **A-T14 A bell keeps ringing** — low
    - Scene: an iron or bronze bell of 200–700 kg struck once; listeners at 1–2 m and at
      30 m outside.
    - Expected: audible for more than 1 s after the strike. Compare levels only with
      single-bell data (the source gives 110–117 dBA Leq for 8–12 bells rung together,
      64–76 dBA outside). The clapper energy is unverified.
    - Source: 202. Today: every sound lasts 50 ms.
15. **A-T15 Shout level** — low
    - Scene: one person shouts.
    - Expected: 82–90 dB at 1 m (men ≈ 88–89, women and children ≈ 82); ≈ 7 dB less
      directly behind the speaker.
    - Source: 199, 200, 221. Today: 90 dB in every direction for everyone.

### 4.9 `tests/test_realism_perception.py`

1. **P-T1 A visible fire across the room** — high
   - Scene: a lit 8 × 5 m room at 5 cm, clear air. An idle, awake person; a wooden bench
     is lit at 3, 6 or 10 m (three runs), straight ahead and at a 45° bearing. No shout,
     no smoke yet.
   - Measure: seconds from the first burning voxel in line of sight (within ± 100° of
     the gaze) to the first fire percept; seconds from that percept to the legs moving away.
   - Expected: noticed within 5 s at every distance and bearing. Move-off 5–60 s after
     noticing (game median 5–30 s; Station nightclub 24–30 s); never under 0.25 s.
   - Source: 242, 227. Today: instant flight at 3 m; no percept at 6 or 10 m (derived).
2. **P-T2 "Fire!" in the house: time before leaving** — high (needs A-T1 fixed)
   - Scene: a two-storey house, evening, 10 awake people in different rooms. One person in
     the hall shouts "Fire!". No smoke or flame is visible from the other rooms.
   - Measure: for each listener, the delay from hearing to leaving the room; whether they
     first did a pre-movement act (look into the corridor, pick something up, go to family).
   - Expected: real medians are minutes (home drills 2:49; WTC 3–4 min). Game compression:
     median 10–60 s; the slowest ≥ 3× the median; ≥ 50% do a pre-movement act; nobody
     under 0.25 s.
   - Source: 240, 241. Today: everyone who hears leaves within one tick.
3. **P-T3 A smoke corridor: Jin's law** — high
   - Scene: a 15 m corridor with uniform smoke at σ = `SMOKE_EXT`·g/L = 0.3, 1.0 and
     3.0 /m. Targets: a burning lamp and a standing person at 1–15 m. Rays along an axis,
     at 45° and on a 3-D diagonal.
   - Measure: the furthest distance where `_sees` is True for each target; the furthest
     cell `_look_around` marks as known; V·σ.
   - Expected: person and walls V·σ ≈ 2 (accept 1.5–4); flame V·σ 5–10 (accept 4–12).
     `_look_around` obeys the same limit. The same V·σ on every bearing.
   - Source: 232, 233. Today: V·σ = 2 for both; diagonals see 1.4–1.7× further;
     `_look_around` sees 12 m through any smoke.
4. **P-T4 Smell before sight, at any voxel size** — high
   - Scene: a smouldering fire in a closed side room, door ajar; a person sits in the next
     room facing away. Run the same scene at 5 and 10 cm.
   - Measure: smoke at the nose (σ in 1/m, and visibility at C = 3) when the first smoke
     percept fires.
   - Expected: the percept fires while visibility is still ≥ 9–10 m (σ ≤ ≈ 0.3 /m). The
     trigger σ agrees within 20% at both voxel sizes.
   - Source: 234. Today: ≈ 0.4 g/L at 5 cm vs 0.05 g/L at 10 cm, an 8× gap.
5. **P-T5 Tavern brawl: ready vs surprised** — high
   - Scene: two people 1 m apart. Run A: B is already fighting A, and A swings. Run B: a
     bystander 1 m to B's side throws a mug at B without warning.
   - Measure: seconds from the start of the swing (or the mug's release) to B's first
     defensive or catching motion.
   - Expected: ready: 0.15–0.35 s (simple reaction ≈ 0.19 s). Surprised: a reflex flinch
     under 0.2 s; an organised defence after ≈ 1–1.5 s (a game band borrowed from driving
     data). A fast surprise blow lands first.
   - Source: 235, 237, 238. Today: catches are gated 0.2 s from launch; a mug from 1 m
     arrives sooner, the single catch attempt fails, and the emergency latch blocks a retry.
6. **P-T6 Ruler drop** — medium (run it before trusting it)
   - Scene: a person standing with an open hand; a 30 cm stick held between thumb and
     finger, released with no warning.
   - Measure: how far the stick falls before the grip closes.
   - Expected: 11–44 cm (0.15–0.30 s); typical 16–24 cm.
   - Source: 236. Today: probably fails (one catch attempt timed from launch; `let_go`
     may not create a flier with a launch tick). Read from code, not run.
7. **P-T7 A runner at the edge of sight** — medium
   - Scene: a person standing, facing +y, with the gaze sweep held at zero. Another body
     runs past 3 m away, crossing at bearings of 90° and 150° from the facing.
   - Measure: whether `sees_runner` fires and how soon.
   - Expected: seen at 90° within 0.2–0.5 s; not seen at 150° (footsteps may be heard).
     The cone edge lies at 95–110°. The percept comes from seen speed, not from the
     runner's `fleeing` flag.
   - Source: 224, 226, 239. Today: cone ± 60°, `run_see_m` 2.5 m: neither is seen.
8. **P-T8 Turning back in smoke** — medium (needs the latch removed)
   - Scene: 20 people (half know the building) must reach the only exit through a 12 m
     corridor where smoke thickens from σ 0 to 3 /m.
   - Measure: share who enter the smoke; share of those who turn back; visibility (C = 3)
     at each turn-back.
   - Expected: ≈ 60% of those who meet smoke go through it; 26–30% of them turn back; of
     the turn-backs, 54–66% at ≤ 1.8 m visibility and 88–94% at ≤ 9 m; ± 15 points.
   - Source: 234. Today: nobody turns back.
9. **P-T9 A crowd's response mix** — medium
   - Scene: 20 people in a tavern; an oil lamp falls and a 1 m flame flares in plain view.
   - Measure: over the first 10 s, classify each person as acting with purpose, slow or
     reflexive, or counterproductive (weeping, confusion, frozen with anxiety); count
     antisocial shoving.
   - Expected: a soft target only (the source is a commentary on severe disasters): a few
     act at once, most act slowly or by reflex, a few are counterproductive; antisocial
     panic ≈ 0; some help others. Wide bands.
   - Source: 245, 246. Today: every person within 4 m runs the same reflex at once.
10. **P-T10 Remember the fire** — medium
    - Scene: a person sees a burning bench, then walks into the next room out of sight.
      Nothing forces an emergency.
    - Measure: `p['danger']` at the fire's columns 10 s and 60 s after the last sight;
      whether the idle person walks back toward it.
    - Expected: weight ≥ 0.5 at 60 s (the code's own stated intent); no wandering back
      during the event.
    - Source: internal design intent (sim.py:288–294). Today: 0.22 at 10 s, 1e-4 at 60 s.
11. **P-T11 A night fire across the village** — medium (needs a light model)
    - Scene: outdoors at full moon (0.05–0.3 lux). A house burns 80 m away in line of
      sight. An unlit person stands still 20 m away.
    - Measure: whether and when the fire is noticed; whether the unlit person is
      detected and identified.
    - Expected: the fire is noticed within a few seconds once in view. The unlit person
      is detected as a shape, not identified; colour and detail are reduced (mesopic),
      not absent.
    - Source: 227, 228, 229, 231. Today: 4 m fire notice, 12 m layout sight, no light term.
12. **P-T12 Startle at a crash** — low
    - Scene: a seated person faces away; a glass pane shatters on stone 3 m behind.
    - Measure: time to the flinch; time until the gaze points at the source.
    - Expected: blink or flinch at 20–120 ms (arms 125–195 ms); gaze on the source at
      0.2–0.6 s (the upper bound is a game band).
    - Source: 238, 239. Today: the gaze snaps on the same tick; no flinch.

### 4.10 Count of reference tests

| File | High | Medium | Low | Total |
|---|---|---|---|---|
| test_realism_mechanics.py | 3 | 1 | 2 | 6 |
| test_realism_heat.py | 5 | 3 | 1 | 9 |
| test_realism_fire.py | 6 | 5 | 2 | 13 |
| test_realism_fluids.py | 2 | 4 | 6 | 12 |
| test_realism_chemistry.py | 6 | 2 | 6 | 14 |
| test_realism_materials.py | 5 | 3 | 4 | 12 |
| test_realism_body.py | 8 | 5 | 3 | 16 |
| test_realism_sound.py | 6 | 6 | 3 | 15 |
| test_realism_perception.py | 5 | 6 | 1 | 12 |
| **All** | **46** | **35** | **28** | **109** |

## 5. References

Every URL below was opened by a researcher, by the checker or by the writer of this
file. "W" marks a source the writer opened while applying the checker's corrections.
Where only an abstract could be read, the entry says so. Sources that could not be
opened (for example an NRC report on lead yield, the IT'IS tissue database, and the
primary forensic papers on glass shards) are not listed; the values that depended on
them are marked (unverified) in the tables.

### 5.1 Mechanics

| # | Title | URL | Used for |
|---|---|---|---|
| 1 | NIST CODATA, standard acceleration of gravity | https://physics.nist.gov/cgi-bin/cuu/Value?gn | g; fall times (M1, M2, M10) |
| 2 | Engineering ToolBox, Air density and specific weight | https://www.engineeringtoolbox.com/air-density-specific-weight-d_600.html | Air density (M4) |
| 3 | Engineering ToolBox, Drag coefficient (W) | https://www.engineeringtoolbox.com/drag-coefficient-d_627.html | Cube 0.8, plate 1.17, box 2.1, person 1.0–1.3 (M4) |
| 4 | NASA Glenn, Shape effects on drag | https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/shape-effects-on-drag/ | Flat-plate drag (M4) |
| 5 | Aerospaceweb, question q0231 | https://aerospaceweb.org/question/aerodynamics/q0231.shtml | Disk Cd 1.17 (NACA TN 253) (M4) |
| 6 | Wikipedia, Drag coefficient | https://en.wikipedia.org/wiki/Drag_coefficient | Cd ranges (M4) |
| 7 | Forward-fall study, J. Biomechanics 2023 (PMC10257944) | https://pmc.ncbi.nlm.nih.gov/articles/PMC10257944 | 700–1200 ms falls; lab falls 597–836 ms (M5, M-T2) |
| 8 | Varieschi & Kamiya, Toy models for the falling chimney, Am. J. Phys. 71, 1025 (2003) | https://arxiv.org/pdf/physics/0210033 | Rod topple equation; chimney break point (M5, M6, S23) |
| 9 | Engineering ToolBox, Friction coefficients | https://www.engineeringtoolbox.com/friction-coefficients-d_778.html | Friction pairs (M7, M-T4) |
| 10 | USDA FPL Wood Handbook GTR-190 (2010), ch. 5, Mechanical properties of wood | https://www.precisebits.com/PDF/USFS_mechanical_properties_of_wood.pdf | Wood E, MOR, tension, toughness, damping, impact bending (S1–S6, S20–S22) |
| 11 | BNP Media / Natural Stone Institute, testing when specifying natural stone (p. 3) | https://continuingeducation.bnpmedia.com/architect/courses/natural-stone-institute/the-importance-of-testing-when-specifying-natural-stone/3/ | ASTM stone MOR minimums (S10, S22, S-T6) |
| 12 | Miyazaki et al., Aerodynamic properties of an archery arrow (manuscript) | https://archerycoachesguildblog2.wordpress.com/wp-content/uploads/2015/06/tl_arfd2.pdf | Arrow drag coefficient (M8) |
| 13 | Wikipedia, English longbow | https://en.wikipedia.org/wiki/English_longbow | War-arrow speed and range (M8) |
| 14 | Cross, Physics of overarm throwing, Am. J. Phys. 72, 305 (2004) | https://www.physics.usyd.edu.au/~cross/PUBLICATIONS/25.%20Throwing.pdf | Throw speed vs mass (B7, B-T11) |
| 15 | Polar Vantage V2 countermovement-jump validity study (PMC9650676) | https://pmc.ncbi.nlm.nih.gov/articles/PMC9650676/ | Jump height 29.9 ± 6.3 cm (B5) |
| 16 | Standing long jump study (PMC12225348) | https://pmc.ncbi.nlm.nih.gov/articles/PMC12225348/ | Long jump 117% of height (B6, B-T10) |
| 17 | US 9,169,042 B2, Energy absorbing container | https://patents.google.com/patent/US9169042B2/en | Bottle break heights (S8, S-T4) |
| 18 | 21 CFR 801.410, impact-resistant lenses | https://www.law.cornell.edu/cfr/text/21/801.410 | 16 g ball from 1.27 m (S8) |
| 19 | Wikipedia, Soda–lime glass | https://en.wikipedia.org/wiki/Soda%E2%80%93lime_glass | Glass E, density, K_Ic (S7) |
| 20 | The Physics Factbook, Coefficients of restitution | https://hypertextbook.com/facts/2006/restitution.shtml | Bounce of wood, steel, glass (M9, M-T5) |
| 21 | SAM, Coefficient of restitution guide | https://www.samaterials.com/content/coefficient-of-restitution.html | Wooden block 0.40–0.55 (no citation in source) (M9) |
| 22 | Ehrnthaller et al. 2014, 35 m vertical free fall (PMC4227496) | https://pmc.ncbi.nlm.nih.gov/articles/PMC4227496/ | Fall heights of survivors and non-survivors (B25, B-T6) |

### 5.2 Heat

| # | Title | URL | Used for |
|---|---|---|---|
| 23 | Engineering ToolBox, Thermal conductivity of common materials | https://www.engineeringtoolbox.com/thermal-conductivity-d_429.html | k of wood, stone, glass, air, charcoal (H1–H8) |
| 24 | HyperPhysics, Thermal conductivity table | http://hyperphysics.gsu.edu/hbase/Tables/thrcn.html | k cross-check (H1, H3, H4) |
| 25 | Engineering ToolBox, Metals thermal conductivity | https://www.engineeringtoolbox.com/thermal-conductivity-metals-d_858.html | Iron k vs temperature (H3, H8) |
| 26 | Effective thermal conductivity of human skin (PMC8953946) | https://pmc.ncbi.nlm.nih.gov/articles/PMC8953946/ | Skin and blood k; tissue density (H5, H11) |
| 27 | Engineering ToolBox, Air properties | https://www.engineeringtoolbox.com/air-properties-d_156.html | Air k and ρ·cp (H6, H12) |
| 28 | Wikipedia, Properties of water | https://en.wikipedia.org/wiki/Properties_of_water | Water k, c, latent heat (H10, H13, H20) |
| 29 | Incropera, typical heat transfer coefficients (MTU copy) | https://pages.mtu.edu/~fmorriso/cm3120/typical_values_heat_transfer_coefficient.pdf | h ranges (H14, H20) |
| 30 | Engineering ToolBox, Convective heat transfer | https://www.engineeringtoolbox.com/convective-heat-transfer-d_430.html | Boiling h (H20) |
| 31 | Engineering ToolBox, Specific heat of solids | https://www.engineeringtoolbox.com/specific-heat-solids-d_154.html | c of solids (H10) |
| 32 | The specific heat of the human body is lower than previously believed (PMC10274559) | https://pmc.ncbi.nlm.nih.gov/articles/PMC10274559/ | Body c (H10) |
| 33 | Engineering ToolBox, Densities of solids | https://www.engineeringtoolbox.com/density-solids-d_1265.html | Densities (H11) |
| 34 | Engineering ToolBox, Fusion heat of metals | https://www.engineeringtoolbox.com/fusion-heat-metals-d_1266.html | Fusion heats (H13) |
| 35 | Engineering ToolBox, Emissivity coefficients | https://www.engineeringtoolbox.com/emissivity-coefficients-d_447.html | ε values (H14–H16) |
| 36 | PDH Online M312, Radiant heat flux (NUREG-1805 ch. 5) | https://pdhonline.com/courses/m312/Radiant%20Flux.pdf | Point-source radiation, χr (H17, H-T3) |
| 37 | Engineers Australia, Tenability criteria practice note | https://www.engineersaustralia.org.au/sites/default/files/2024-01/tenability-criteria-practice-note_0.pdf | 2.5 kW/m² pain asymptote; hot layer flux (H17, H26) |
| 38 | European Fire Safety Alliance, ISO 13571 / SFPE / AEGL comparison | https://www.europeanfiresafetyalliance.org/wp-content/uploads/2021/02/20200916-BA-Appendix-1-Comparison-between-the-ISO-standard-SFPE-handbook-and-AEGL.pdf | ISO pain, burn and hot-air times; Fire Service Academy table (H17, B20) |
| 39 | NIST TN 1600, Residential structure separation fire experiments | https://www.govinfo.gov/content/pkg/GOVPUB-C13-61ebeb4a9065a1584c3a9b0561bbb525/pdf/GOVPUB-C13-61ebeb4a9065a1584c3a9b0561bbb525.pdf | House-to-house flux and ignition (H26, F-T6) |
| 40 | Babrauskas, Ignition of wood: a review of the state of the art | https://doctorfire.com/pages/wood_ign.pdf | Ignition temperatures and times (F1–F4, F23) |
| 41 | PLOS ONE, The fine tuning of pain thresholds | https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0010269 | Pain at 43.0 / 45.7 °C; skin baseline (B18, B19) |
| 42 | A.O. Smith Technical Bulletin 34 (CPSC burn chart) | https://www.hotwater.com/info-center/technical-bulletins/bulletin-34.html | Scald times (B-T13) |
| 43 | Engineering ToolBox, Metabolic rate | https://www.engineeringtoolbox.com/met-metabolic-rate-d_733.html | Body heat (B19) |
| 44 | US 9,777,354 B2 (steel bar cooling in air) | https://patents.google.com/patent/US9777354B2/en | 0.89 °C/s bar cooling (H21, H-T1) |
| 45 | Wikipedia, Draper point | https://en.wikipedia.org/wiki/Draper_point | Glow ends near 525 °C (H21) |
| 46 | Coffee cooling data (U. Manitoba / UNL course project) | https://server.math.umanitoba.ca/~coopers5/pastcourses_unl_materials/project1_coffee_data.pdf | Cup cooling curve (H22, H-T6) |
| 47 | Wikipedia, Forging | https://en.wikipedia.org/wiki/Forging | Forging temperatures (H23) |
| 48 | New Equipment Digest, Material heating times | https://www.newequipment.com/plant-operations/article/21923688/material-heating-times | ASM 5 min per inch (H23, H-T2) |
| 49 | USDA FPL Wood Handbook FPL-GTR-282 (2021), ch. 18, Fire safety of wood construction | https://research.fs.usda.gov/download/treesearch/62268.pdf | Char rate, heat penetration, mass loss (H24, F5, F6) |
| 50 | FM Global, Tamanini 2018, Fire plume and ceiling layer correlations | https://www.fm.com/-/media/project/publicwebsites/fm/files/resources/research-technical-reports/rtr-plume-layer-transition.pdf | Plume temperature and speed (H25, H-T7) |
| 51 | Wikipedia, Flashover | https://en.wikipedia.org/wiki/Flashover | Flashover criteria and timing (H26, F15) |
| 52 | MacKenzie 2017, Understanding the cooling curve test (Thermal Processing) | https://thermalprocessing.com/wp-content/uploads/2017/201701/0117-ASM.pdf | Oil quench probe times (H27) |
| 53 | MetallurgyZone, Quenching steel guide (secondary) | https://metallurgyzone.com/quenching-steel-complete-guide/ | Quench severity H (H27) |
| 54 | Knife Steel Nerds, How fast do you have to quench | https://knifesteelnerds.com/2019/02/25/how-fast-do-you-have-to-quench-hardenability-of-steel/ | Quench rates (H27) |

### 5.3 Fire

| # | Title | URL | Used for |
|---|---|---|---|
| 55 | White & Dietenberger, Wood Handbook GTR-190 (2010), ch. 18 | https://research.fs.usda.gov/download/treesearch/37415.pdf | Ignition, char, heat of combustion, 13.1 kJ/g O2 (F3, F5, F7, F8) |
| 56 | Characterisation of the fire behaviour of wood (PMC9308566) | https://pmc.ncbi.nlm.nih.gov/articles/PMC9308566/ | Cone HRR; critical mass loss (F6, F7) |
| 57 | NIST Fire Calorimetry Database, Test5_2Pallet_r1 | https://www.nist.gov/el/fcd/multiple-item-transient-combustion-calorimetry/test52palletr1 | Two-pallet HRR (F7, F17, F-T2) |
| 58 | Tewarson 2008, Smoke emissions in fires (Fire Safety Science 9) | https://publications.iafss.org/publications/fss/9/1153/view/fss_9-1153.pdf | Wood formula, O2 demand, soot yield (F9, F19) |
| 59 | Engineering ToolBox, Air composition | https://www.engineeringtoolbox.com/air-composition-d_212.html | O2 per litre (F10) |
| 60 | Richter, Jervis, Huang & Rein 2021, Effect of oxygen on the burning rate of wood | https://ira.lib.polyu.edu.hk/bitstream/10397/92504/1/Richter_Oxygen_Charring_Rate.pdf | Flaming and smouldering O2 limits (F11, F-T4) |
| 61 | Engineering ToolBox, Fuels: higher and lower calorific values | https://www.engineeringtoolbox.com/fuels-higher-calorific-values-d_169.html | Charcoal and oil heating values (F8, F12, F13) |
| 62 | Engineering ToolBox, Autoignition temperatures | https://www.engineeringtoolbox.com/fuels-ignition-temperatures-d_171.html | Charcoal autoignition (F12) |
| 63 | Wikipedia, Flash point | https://en.wikipedia.org/wiki/Flash_point | Canola flash point (F13) |
| 64 | Buda-Ortins 2010, Auto-ignition of cooking oils (U. Maryland) | https://drum.lib.umd.edu/bitstream/handle/1903/11333/Buda_Ortins_ResearchPaper.pdf?sequence=2 | Oil fire and autoignition points (F13) |
| 65 | Engineering ToolBox, Flash points of fuels | https://www.engineeringtoolbox.com/flash-point-fuels-d_937.html | Kerosene flash point (F13) |
| 66 | Mealy, Benfer & Gottuk, Fire dynamics and forensic analysis of liquid fuel fires (NIJ 238704) | https://www.ojp.gov/pdffiles1/nij/grants/238704.pdf | Pool burning rates; spill areas and depths (F14, L7, L8, F-T11) |
| 67 | Babrauskas, Temperatures in flames and fires | https://doctorfire.com/temperatures-in-flames-and-fires/ | Flame temperatures (H28, F-T13) |
| 68 | NIST, Fire dynamics | https://www.nist.gov/el/fire-research-division-73300/firegov-fire-service/fire-dynamics | Candle power; flashover (F15, F24) |
| 69 | Kweon, Kim & Yang 2016, J. Mater. Sci. Eng. B 6 | https://www.davidpublisher.com/Public/uploads/Contribute/57d66d49521b0.pdf | Flashover HRR coefficient 750 (F15) |
| 70 | Kerber 2012, Analysis of changing residential fire dynamics | https://fsri.org/sites/default/files/2021-07/Analysis_of_Changing_Residential_Fire_Dynamics_and_Its_Implications_on_Firefighter_Operational_Timeframes.pdf | Legacy and modern room flashover times; chair HRR (F16, F17) |
| 71 | NIST, Fire hazards of dry versus watered Christmas trees | https://www.nist.gov/publications/fire-hazards-dry-versus-watered-christmas-trees | Dry vs watered tree (F18, F-T7) |
| 72 | NIST TN 2327r1 (2025), Christmas tree fires | https://nvlpubs.nist.gov/nistpubs/TechnicalNotes/NIST.TN.2327r1.pdf | Tree HRR, χr 0.30, soot yield, moisture limits (F18, F19, F23) |
| 73 | NIST FCD, Dry_Pine_Tree | https://www.nist.gov/el/fcd/dry-tree-fire-hazard/drypinetree | Tree fire data (F18) |
| 74 | Thunderhead Engineering, Smoke visibility and obscuration | https://www.thunderheadeng.com/docs/2026-1/pyrosim/examples/how-to/smoke-visibility-obscuration/ | Soot extinction 8.7 m²/g; visibility constant (F19, F-T10) |
| 75 | MMEP 10(6), Smoke-filling dynamics in compartment fires: a two-zone approach | https://www.iieta.org/journals/mmep/paper/10.18280/mmep.100604 | Zukoski smoke filling (F20, F-T5) |
| 76 | Engineering ToolBox, Water properties (heat of vaporization) | https://www.engineeringtoolbox.com/water-properties-d_1573.html | 2256.4 kJ/kg (F22, L-T6) |
| 77 | Wikipedia, Chip pan | https://en.wikipedia.org/wiki/Chip_pan | Slopover; chip-pan demos (F21, F-T9) |
| 78 | Gizmodo, high-speed video of a water drop on hot oil (Penn State) | https://gizmodo.com/watch-what-happens-to-this-drop-of-water-when-it-hits-h-1728975481 | Explosive steam expansion (F21) |
| 79 | De la Cruz-Ávila et al., Luminous flame height correlation (arXiv 2008.12209) | https://arxiv.org/pdf/2008.12209 | Heskestad flame height (F25, F-T11) |
| 80 | NIST, Smoke alarm Q&As (TN 1455) | https://www.nist.gov/system/files/documents/2017/05/09/SmokeDetectors_Q-As_Feb2008.pdf | Time to untenable (F26, F-T3, A-T5) |
| 81 | Engineering ToolBox, Air specific heat | https://www.engineeringtoolbox.com/air-specific-heat-capacity-d_705.html | Air cp (H12) |

### 5.4 Fluids

| # | Title | URL | Used for |
|---|---|---|---|
| 82 | Casaburo, Newton cooling time constant by Arduino (arXiv 2107.09527) | https://arxiv.org/pdf/2107.09527 | Cup τ = 1039 s (H22, H-T6) |
| 83 | Wikipedia, Forge welding | https://en.wikipedia.org/wiki/Forge_welding | Welding temperatures (H28) |
| 84 | Wikipedia, Lead | https://en.wikipedia.org/wiki/Lead | Lead properties (L1, L2, S15) |
| 85 | Wikipedia, Tin | https://en.wikipedia.org/wiki/Tin | Tin properties (L1, L2, S16) |
| 86 | Wikipedia, Iron | https://en.wikipedia.org/wiki/Iron | Iron properties (L1, L2, L5) |
| 87 | OECD-NEA, Handbook on LBE and lead properties, ch. 2 (2007) | https://www.oecd-nea.org/science/reports/2007/pdf/chapter2.pdf | Molten lead ρ, η, σ, k, fusion heat (L1, L4, L-T1, L-T5) |
| 88 | Engineering ToolBox, Latent heat of melting | https://www.engineeringtoolbox.com/latent-heat-melting-solids-d_96.html | Fusion heats, ice (L1, L16) |
| 89 | Humrickhouse, Liquid tin properties (INL/CON-17-41428) | https://inldigitallibrary.inl.gov/sites/sti/sti/8059219.pdf | Molten tin (L3) |
| 90 | Turchi, Viscosity and surface tension of metals (LLNL-TR-750486) | https://www.osti.gov/servlets/purl/1438687/ | Molten iron (L5) |
| 91 | Engineering ToolBox, Water viscosity | https://www.engineeringtoolbox.com/water-dynamic-kinematic-viscosity-d_596.html | Water viscosity (L6) |
| 92 | Nierat, Musameh & Abdel-Raziq 2014, Olive oil viscosity | https://staff-old.najah.edu/sites/default/files/Temperature-Dependence_of_Olive_Oil_Viscosity.pdf | Olive oil viscosity (L6) |
| 93 | Engineering ToolBox, Absolute viscosity of liquids | https://www.engineeringtoolbox.com/absolute-viscosity-liquids-d_1259.html | Kerosene viscosity (L6) |
| 94 | Wikipedia, Sulfuric acid | https://en.wikipedia.org/wiki/Sulfuric_acid | Acid properties, grades, dilution, products (L24, C3, C6, C11–C16) |
| 95 | Mugele & Heikenfeld, Electrowetting, ch. 1 (Wiley sample) | https://application.wiley-vch.de/books/sample/3527412298_c01.pdf | Puddle thickness formula (L7, L-T1, L-T2) |
| 96 | Wikipedia, Capillary length | https://en.wikipedia.org/wiki/Capillary_length | Capillary length (L7) |
| 97 | Wikipedia, Torricelli's law | https://en.wikipedia.org/wiki/Torricelli%27s_law | Outflow; g (L9, L10, L-T8) |
| 98 | Engineering ToolBox, Specific heat of water | https://www.engineeringtoolbox.com/specific-heat-capacity-water-d_660.html | Water c (L-T5) |
| 99 | Wikipedia, Critical heat flux | https://en.wikipedia.org/wiki/Critical_heat_flux | Boiling ceiling (L11, L-T4) |
| 100 | Wikipedia, Nucleate boiling | https://en.wikipedia.org/wiki/Nucleate_boiling | Nucleate range (L11) |
| 101 | Wikipedia, Leidenfrost effect | https://en.wikipedia.org/wiki/Leidenfrost_effect | Leidenfrost point (L11, L-T4) |
| 102 | Engineering ToolBox, Saturated steam properties | https://www.engineeringtoolbox.com/saturated-steam-properties-d_101.html | Steam volume (L12, F-T9) |
| 103 | Wikipedia, Steam explosion | https://en.wikipedia.org/wiki/Steam_explosion | Molten metal and water (L12, C-T14) |
| 104 | Engineering ToolBox, Evaporation from water surfaces | https://www.engineeringtoolbox.com/evaporation-water-surface-d_690.html | Evaporation rate (L15, L-T9) |
| 105 | Frederking 2018, Predicting ice thickness for engineering applications (IAHR) | https://nrc-publications.canada.ca/eng/view/ft/?id=fce1c616-994a-4f9f-8634-6d20be17c19d | Stefan ice growth (L16, L-T12) |
| 106 | KSU notes, Prediction of solidification time: Chvorinov's rule | https://faculty.ksu.edu.sa/sites/default/files/prediction_of_solidification_time1.pdf | Chvorinov's rule (L17) |
| 107 | Askeland text excerpt (U. Babylon) | https://www.uobabylon.edu.iq/eprints/publication_5_9952_901.pdf | Mould constant; depth-fit example (L17) |
| 108 | Wikipedia, Solder | https://en.wikipedia.org/wiki/Solder | Tin–lead eutectic (L18, L-T11) |
| 109 | Glass & Zelinka, Wood Handbook FPL-GTR-282, ch. 4, Moisture relations | https://research.fs.usda.gov/download/treesearch/62243.pdf | Water absorption by wood (L19, L-T10) |
| 110 | Uchino, The little science behind towel absorbency (manufacturer) | https://en.uchino.co.jp/journal/%e3%82%bf%e3%82%aa%e3%83%ab%e3%81%ae%e5%90%b8%e6%b0%b4%e5%8a%9b%e3%81%ae%e3%83%97%e3%83%81%e7%a7%91%e5%ad%a6%e3%81%a83%e3%81%a4%e3%81%ae%e3%82%bf%e3%82%a4%e3%83%97%e5%88%a5%e3%81%8a%e3%81%99%e3%81%99/ | Towel absorbency (L20) |
| 111 | Nader et al. 2019, Blood rheology (PMC6842957) | https://pmc.ncbi.nlm.nih.gov/articles/PMC6842957/ | Blood viscosity (L21) |
| 112 | StatPearls, Hemorrhagic shock | https://www.ncbi.nlm.nih.gov/sites/books/NBK470382/ | Blood volume; ATLS classes (B13) |
| 113 | Wikipedia, Bleeding time | https://en.wikipedia.org/wiki/Bleeding_time | Ivy and Duke times (B15, B-T4) |
| 114 | Hoffman & Snyder 1966, The ground-glass clotting time | https://www.ccjm.org/content/ccjom/33/3/107.full.pdf | Lee-White ≤ 20 min (L23, L-T7) |
| 115 | Smith, Nicloux & Brutin 2020, A new forensic tool to date human blood pools | https://www.nature.com/articles/s41598-020-65465-4 | Drop clotting, pool drying, pool height (L7, L23, L-T7) |
| 116 | Lewis et al. 1990, Common femoral artery volume flow (PubMed 2317678; abstract read through NCBI E-utilities) | https://pubmed.ncbi.nlm.nih.gov/2317678/ | Femoral flow at rest and peak (B14, B-T3) |
| 117 | Wikipedia, Charcoal (W) | https://en.wikipedia.org/wiki/Charcoal | 1100 °C; ≈ 1260 °C with a blower (H28) |
| 118 | Wikipedia, Blood (W) | https://en.wikipedia.org/wiki/Blood | Blood density 1060 kg/m³ (L22) |

### 5.5 Chemistry

| # | Title | URL | Used for |
|---|---|---|---|
| 119 | OSHA 29 CFR 1910.1200, Appendix A | https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.1200AppA | Skin corrosion categories (C1, C-T1) |
| 120 | 49 CFR 173.137, corrosive packing groups | https://www.govinfo.gov/content/pkg/CFR-2012-title49-vol2/pdf/CFR-2012-title49-vol2-sec173-137.pdf | Full-thickness times; corrosive to metal (C1, C5) |
| 121 | Labbox, Sulfuric acid 95–98% SDS | https://labbox.es/wp-content/uploads/FDS/SDS_EN_10526.pdf | 1A from 15%; Packing Group II; viscosity (C1, C12, L24) |
| 122 | Hydrochloric acid 34–37% SDS (Minnesota Dept of Health) | https://www.health.state.mn.us/communities/environment/envlab/sdsinfo/HclBna.pdf | HCl class; H₂ from metals; density (C1, C6) |
| 123 | Palao et al. 2009, Chemical burns: pathophysiology and treatment | https://www.prevor.com/wp-content/uploads/publications/burns-juil2009-chemical-burns-pathophysi.pdf | Acid burns progress with contact time; first aid (C1) |
| 124 | Wikipedia, Human skin | https://en.wikipedia.org/wiki/Human_skin | Skin thickness (C1) |
| 125 | Wikipedia, Epidermis (W) | https://en.wikipedia.org/wiki/Epidermis | Epidermis 31–597 µm, typical ≈ 90 µm (C1) |
| 126 | Health Canada, Acetic acid hazardous substance assessment | https://www.canada.ca/en/health-canada/services/environmental-workplace-health/occupational-health-safety/workplace-hazardous-materials-information-system/hazardous-substance-assessments/acetic-acid.html | Vinegar on skin (C2, C12, C13) |
| 127 | geology.com, The acid test for carbonate minerals | https://geology.com/minerals/acid-test.shtml | Fizz on limestone (C7, C8) |
| 128 | USDA FPL Wood Handbook GTR-190 (2010), full text | https://www.precisebits.com/PDF/fpl_gtr190.pdf | Wood resistance to chemicals (C3, C4) |
| 129 | Wood Tank & Pipe, Wooden tank advantages | https://www.woodtankandpipe.com/wooden-tank-advantages/ | Acid tanks pH ≥ 1.5 for decades (C3, C4) |
| 130 | Bhagia et al. 2016, Robustness of two-step acid hydrolysis | https://cellulosicbiomasslab.engr.ucr.edu/sites/default/files/2019-02/robustness_of_two-step_acid_hydrolysis_procedure.pdf | 72% acid on wood particles; lignin residue (C3, C-T4) |
| 131 | Sci. Rep. 2025, IDTT inhibitor for mild steel in 1 M HCl | https://www.nature.com/articles/s41598-025-95104-9 | 5.74 mg/(cm²·h) blank (C5) |
| 132 | Sci. Rep. 2022, nonanedioic acid derivative inhibitor for mild steel in HCl (PMC8933592) (W) | https://pmc.ncbi.nlm.nih.gov/articles/PMC8933592/ | Blank i_corr 667 µA/cm² at 303 K (C5) |
| 133 | US 4,154,791 A, Inhibition of sulfuric acid attack on carbon steel | https://patents.google.com/patent/US4154791A/en | Steel in strong vs dilute H2SO4 (C5, C12, C-T6) |
| 134 | Standard-state thermodynamic values at 298.15 K | https://conradnaleway.net/ThermoData.PDF | Reaction enthalpies (C6, C10, C11) |
| 135 | Lund, Fogler, McCune & Ault 1975, Dissolution of calcite in HCl | https://backend.production.deepblue-documents.lib.umich.edu/server/api/core/bitstreams/f6705606-d10f-45b6-ab4b-2c5c0d6082e1/content | Limestone dissolution rate (C7, C-T7) |
| 136 | Slabwise, Which stones are most prone to etching (trade source) | https://slabwise.com/guides/what-stones-are-most-prone-to-etching-and-how-to-explain-this | Granite vs marble (C8, C-T12) |
| 137 | Soler et al., The passivation of calcite by acid mine water (OSTI 970048) | https://www.osti.gov/servlets/purl/970048 | Gypsum armour (C9, C-T7) |
| 138 | Wikipedia, Standard enthalpy of formation | https://en.wikipedia.org/wiki/Standard_enthalpy_of_formation | Formation enthalpies (C11) |
| 139 | Harvey Mudd E82 lecture, Reference states, ΔHs, ΔHm (Felder, Rousseau & Bullard Table B.11) (W) | https://pages.hmc.edu/spjut/E82/lec/FirstLawLeftovers.pdf | Integral heat of mixing of H2SO4 (C11, C-T5) |
| 140 | ISO 9223:2012 preview (iTeh) | https://cdn.standards.iteh.ai/samples/53499/e1f1aefb0a5446ac8308e3ddfce1db8b/ISO-9223-2012.pdf | Rust rates (C17, C-T13) |
| 141 | Wikipedia, Boilover | https://en.wikipedia.org/wiki/Boilover | Slopover; 2.4 s demo (F21, F-T9) |
| 142 | Kobasko, Heat transfer fundamentals concerning quenching (IntechOpen) | https://www.intechopen.com/chapters/88502 | 50 mm bar quench in brine (H27, H-T5) |
| 143 | Wikipedia, Shot tower (W) | https://en.wikipedia.org/wiki/Shot_tower | Molten lead into water as routine (L12, C-T14) |

### 5.6 Materials

| # | Title | URL | Used for |
|---|---|---|---|
| 144 | AGC Planibel technical data sheet 01/2023 | https://www.agc-yourglass.com/sites/default/files/2023-01/TDS_Planibel_0123_EN.pdf | Glass E, ν, 45 MPa (S7, S-T7) |
| 145 | Wikipedia, List of physical properties of glass | https://en.wikipedia.org/wiki/List_of_physical_properties_of_glass | Container glass (S7) |
| 146 | MakeItFrom, Soda-lime float glass | https://www.makeitfrom.com/material-properties/Soda-Lime-Float-Glass | Glass tensile range (S7) |
| 147 | Wikipedia, Fracture toughness | https://en.wikipedia.org/wiki/Fracture_toughness | Glass K_Ic (S7) |
| 148 | Pressglass, Glass classification based on user safety (EN 12600) | https://www.pressglass.com/wp-content/uploads/2021/08/Glass_classification_based_on_user_safety_EN_RGB.pdf | Pendulum impact energies (S9, S-T10) |
| 149 | Pelfrene et al. 2016, Challenging Glass 5 | https://proceedings.challengingglass.com/index.php/cgc/article/download/162/161/782 | EN 12600 test (S9) |
| 150 | Padmanabha et al., Dynamic split tensile strength of rocks (arXiv 2110.10072) | https://arxiv.org/pdf/2110.10072 | Stone static and dynamic strength (S10, S11, S-T1) |
| 151 | Amadei, CVEN5768 notes 8 | https://ceae.colorado.edu/~amadei/CVEN5768/PDF/NOTES8.pdf | Rock UCS (S10, S11) |
| 152 | Materials 2023, sandstone strength (PMC10179767) | https://pmc.ncbi.nlm.nih.gov/articles/PMC10179767/ | Sandstone tension and flexure (S10) |
| 153 | Wikipedia, Wrought iron | https://en.wikipedia.org/wiki/Wrought_iron | Wrought iron strength (S12, S22) |
| 154 | Wikipedia, Gray iron | https://en.wikipedia.org/wiki/Gray_iron | Cast iron strength (S13) |
| 155 | Srinivasan & Seetharamu, Fracture toughness of metal castings (InTech) | https://www.intechopen.com/chapters/37419 | Cast iron K_Ic (S13) |
| 156 | thyssenkrupp, S275JR data | https://www.thyssenkrupp-materials-processing-europe.com/en/c-steel/hot-strip/s235jr-s355k2/s275jr | Charpy energy (S14) |
| 157 | Goodfellow, Tin | https://www.goodfellow.com/usa/material/metals/tin | Tin yield and creep (S16, S22) |
| 158 | WikiMSK, Bone biomechanics | https://wikimsk.org/wiki/Bone_Biomechanics | Bone E and strength (S17) |
| 159 | Koester, Ager & Ritchie 2008, The true toughness of human cortical bone | https://www2.lbl.gov/ritchie/Library/PDF/how_tough_bone.pdf | Bone toughness (S19) |
| 160 | Schreiber, Crandall et al. 1997, Static and dynamic bending strength of the leg (IRCOBI) | https://www.ircobi.org/wordpress/downloads/irc1997/pdf_files/1997_6.pdf | Leg fracture moment and energy (S18, S-T9) |
| 161 | Ogneva et al. 2010, Transversal stiffness of rat soleus fibres by AFM (PMC2814213) | https://pmc.ncbi.nlm.nih.gov/articles/PMC2814213/ | Muscle stiffness (S19) |
| 162 | Engineering ToolBox, Beams: stress and deflection | https://www.engineeringtoolbox.com/beam-stress-deflection-d_1312.html | Beam formulas (S21, S-T3, S-T7) |
| 163 | IRC Table R301.7, deflection limits | https://iccsafe.org/wp-content/uploads/bcac/IRC_Table_R301-7_New_11-05-12pdc.pdf | L/360 (S21) |
| 164 | Engineering Guidance, The falling chimney | https://www.engineeringguidance.com/the-falling-chimney | Moment in a falling trunk (S23, S-T2) |
| 165 | OSHA Logging eTool, Felling: the hinge | https://www.osha.gov/etools/logging/manual-operations/felling/cuts/hinge | Hinge proportions (S24, S-T2) |
| 166 | Hunt & Henley 1981, USDA FS GTR PNW-122 | https://www.fs.usda.gov/pnw/pubs/pnw_gtr122.pdf | Felling breakage (S24, S-T2) |
| 167 | Digitalfire, Fired strength | https://digitalfire.com/glossary/fired+strength | Stoneware MOR (S26) |
| 168 | Siliceous spicules and fracture resistance of pottery (PMC4550886) | https://pmc.ncbi.nlm.nih.gov/articles/PMC4550886/ | Low-fired clay E (S26) |
| 169 | Suominen et al. 2021, Flexural strength of freshwater ice in Saimaa area (POAC21-075) (W) | https://www.poac.com/Proceedings/2021/POAC21-075.pdf | Lake ice strength (S27, S-T12) |
| 170 | Karate Research, The physics of karate (secondary summary of Feld, McNair & Wilk) | https://karateresearch.com/2025/10/16/the-physics-of-karate-a-chapter-in-modern-karate-history/ | Board-breaking energy (S25, S-T10) |

### 5.7 Body

| # | Title | URL | Used for |
|---|---|---|---|
| 171 | Winter, Biomechanics and motor control of human movement, ch. 4 (anthropometry) | https://courses.grainger.illinois.edu/me481/sp2021/Anthro-Winter.pdf | Segment masses and lengths (B1, B2, B-T15) |
| 172 | Bohannon & Andrews 2011, Normal walking speed: a descriptive meta-analysis | https://www.csp.org.uk/journal/article/physiotherapy-september-2011/normal-walking-speed-descriptive-meta-analysis | Walking speed (B3, B-T7) |
| 173 | Wikipedia, Preferred walking speed | https://en.wikipedia.org/wiki/Preferred_walking_speed | Walking speed (B3) |
| 174 | Wikipedia, Footspeed | https://en.wikipedia.org/wiki/Footspeed | Elite top speed (B4) |
| 175 | Hreljac 1993, Preferred and energetically optimal gait transition speeds (Med Sci Sports Exerc, PMID 8231761; abstract read through Europe PMC) (W) | https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=hreljac%20gait%20transition%20speeds%201993&format=json&resultType=core | Walk-to-run 2.06 m/s (B4, B-T7) |
| 176 | Effect of age and sex on lower-extremity power, Scand J Med Sci Sports 2026 | https://doi.org/10.1111/sms.70322 | Jump heights by age and sex (B5, B-T9) |
| 177 | CCOHS, NIOSH lifting equation | https://www.ccohs.ca/oshanswers/ergonomics/niosh/assessing.html | 23 kg load constant (B8, B-T16) |
| 178 | Dodds et al. 2014, Grip strength across the life course | https://doi.org/10.1371/journal.pone.0113637 | Grip strength (B8) |
| 179 | Maximum shoulder torque during standing arm flexion, J Funct Morphol Kinesiol | https://doi.org/10.3390/jfmk11010020 | Shoulder torque (B9, B-T16) |
| 180 | Walilko, Viano & Bir 2005, Biomechanics of the head for Olympic boxer punches | https://www.sportmedicine.ru/articles/biomechanics_of_the_head_for_olympic_boxer_punches_to_the_face.pdf | Punch speed, energy, head acceleration, injury thresholds (B10, B11, B-T12) |
| 181 | Smith et al. 2000, Punch force in boxers, J Sports Sci | https://doi.org/10.1080/02640410050074377 | Rear-hand force (B10) |
| 182 | Wikipedia, Blood volume | https://en.wikipedia.org/wiki/Blood_volume | 70 / 65 ml/kg (B12) |
| 183 | LITFL, Major haemorrhage in trauma | https://litfl.com/major-haemorrhage-in-trauma/ | ATLS classes (B13, B-T3) |
| 184 | Fatal exsanguination after complete transection of femoral vessels, J Forensic Sci 2023 | https://doi.org/10.1111/1556-4029.15241 | Death within 10 min (B14, B-T3) |
| 185 | American College of Surgeons, Stop the Bleed press release (2023) | https://www.facs.org/media-center/press-releases/2023/stop-the-bleed-is-the-cpr-of-bleeding/ | "Within minutes" (B14) |
| 186 | Kragh et al. 2009, Survival with emergency tourniquet use | https://doi.org/10.1097/SLA.0b013e31818842ba | Tourniquet survival (B16, B-T5) |
| 187 | Liang 2020, Doppler flow measurement of lower extremity arteries (AJR, PMID 31670583; abstract read through Europe PMC) (W) | https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:31670583&format=json&resultType=core | Femoral resting flow (B14, B-T3) |
| 188 | Foster et al. 2021, Health Physics (PMC8300848) | https://doi.org/10.1097/HP.0000000000001447 | Burn thresholds; ×1.8 per °C (B17, B18, B-T13) |
| 189 | Martin & Falder 2017, Burns review | https://doi.org/10.1016/j.burns.2017.04.003 | 44 °C basal epidermis threshold (B17) |
| 190 | antiscald.com, burn time and temperature chart (Moritz & Henriques) | https://antiscald.com/index.php?route=information%2Finformation&information_id=15 | Scald times (B17, B-T13) |
| 191 | Purser 1989, Modelling toxic and physical hazard in fire (Fire Safety Science 2) | https://publications.iafss.org/publications/fss/2/391/view/fss_2-391.pdf | CO dose; hot-air incapacitation; burn incapacitation (B17, B20, B23, B-T8, B-T14) |
| 192 | Classic and exertional heatstroke, Nat Rev Dis Primers 2022 | https://doi.org/10.1038/s41572-021-00334-6 | Heat stroke definition (B20) |
| 193 | US CSB, Hazards of nitrogen asphyxiation (2003) | https://www.csb.gov/assets/1/6/nitrogen_asphyxiation_safety_bulletin_(6-11-03).pdf | Effects of low O2 (B21, B-T2) |
| 194 | Wikipedia, Drowning | https://en.wikipedia.org/wiki/Drowning | Breath-hold and drowning times (B22, B-T1) |
| 195 | Engineering ToolBox, Carbon monoxide | https://www.engineeringtoolbox.com/carbon-monoxide-d_893.html | CO ppm vs time (B23) |
| 196 | Wikipedia, Carbon monoxide poisoning | https://en.wikipedia.org/wiki/Carbon_monoxide_poisoning | COHb half-life (B24, B-T8) |
| 197 | American College of Surgeons, National guideline for field triage of injured patients, 2021 (W) | https://www.facs.org/media/2zdltley/field-triage-handout-vfinal-revise.pdf | Fall > 10 ft criterion (B25, B-T6) |
| 198 | Notebook on fall LD50, quoting Appalachian SAR (secondary) | http://www.conovers.org/ftp/NOTEBOOK/Falls.TXT | LD50 ≈ 15 m, LD90 ≈ 26 m (B25, B-T6) |

### 5.8 Sound

| # | Title | URL | Used for |
|---|---|---|---|
| 199 | Olsen 1998, Average speech levels and spectra (summary of Pearsons, Bennett & Fidell 1977) | https://files.ceqanet.lci.ca.gov/276991-2/attachment/2IHwmgJ6SJZB1WNWDl_8jY9tA4GnUZz0AQza16efeHQK_9wElzk6GqIMJTGb7a4rymVZ1Er7eIbk-5n20 | Speech levels; home noise (A1, A2, A6) |
| 200 | Rindel 2019, Restaurant acoustics (Acoustics in Practice 7(1)) | https://euracoustics.org/documents/8/01_Restaurant_acoustics.pdf | Vocal effort; restaurant noise; Lombard; SNR ratings (A1, A2, A5–A7) |
| 201 | Engineering ToolBox, Required voice level at distance | https://www.engineeringtoolbox.com/voice-level-d_938.html | Spreading loss (A4) |
| 202 | Bassett 2005, The acoustics of church bell-towers for change ringing | https://acoustics.asn.au/conference_proceedings/AAS2005/papers/67.pdf | Bell levels (A20, A-T14) |
| 203 | NIOSH HHE 2007-0075-3251, Hammer forge noise (2016) | https://stacks.cdc.gov/view/cdc/172411/cdc_172411_DS1.pdf | Metal impact levels (A18, A19, A-T12, A-T13) |
| 204 | Bruck & Brennan 2001, Recognition of fire cues during sleep | https://vuir.vu.edu.au/450/1/fire_cues-_brennan1-_boston.pdf | Crackle wakes 91% (A21, A-T5) |
| 205 | CPSC-ES-0502 (2004), smoke alarm audibility and waking | https://www.cpsc.gov/s3fs-public/pdfs/blk_pdf_alarm1.pdf | Room-to-room losses; waking thresholds (A11, A13, A22, A-T2, A-T3, A-T11) |
| 206 | Wikipedia, Critical distance | https://en.wikipedia.org/wiki/Critical_distance | Reverberant field (A16) |
| 207 | ISO 9613-2:1996 (copy in a South Dakota PUC docket) | https://puc.sd.gov/commission/dockets/electric/2019/el19-003/KMExhibit9.pdf | Air absorption; foliage; barrier caps (A13, A15, A17, A-T6) |
| 208 | Irvine, Acoustic transmission loss, Rev F | https://www.vibrationdata.com/tutorials/aco_tl.pdf | Mass law (A8) |
| 209 | CCCBR Towers & Belfries Committee, Sound control in belltowers | https://cccbr.org.uk/wp-content/uploads/2016/03/sound-control-notes.pdf | Timber and concrete TL (A9, A10, A-T7) |
| 210 | PCI Designer's Notebook, Acoustics | https://www.enterpriseprecast.com/wp-content/uploads/2023/05/Acoustics.pdf | Concrete STC; gaps (A10–A12, A-T8) |
| 211 | Wikipedia, Sound transmission class | https://en.wikipedia.org/wiki/Sound_transmission_class | STC table; block walls (A10, A11) |
| 212 | Trustile, Understanding STC ratings | https://www.trustile.com/technical-information/understanding-stc-ratings | Door STC (A11, A-T3) |
| 213 | Arup Strutt, Composite TL | https://strutt.arup.com/help/Building_Acoustics/CompositeTL_BA.htm | Openings in a partition (A12, A-T8) |
| 214 | Arup Strutt, Barrier attenuation | https://strutt.arup.com/help/Environmental_Noise/EnviroBarrierAtten.htm | Barrier loss cap (A13, A-T2, A-T4) |
| 215 | Wikipedia, Sound localization | https://en.wikipedia.org/wiki/Sound_localization | Head shadow (A14, A-T4) |
| 216 | Wikipedia, Absolute threshold of hearing | https://en.wikipedia.org/wiki/Absolute_threshold_of_hearing | Hearing threshold (A23) |
| 217 | WHO, Community noise (1995) | https://www.nonoise.org/library/whonoise/whonoise.htm | Bedroom guidelines (A6, A23) |
| 218 | DPA Microphones, Facts about speech intelligibility | https://www.dpamicrophones.com/mic-university/background-knowledge/facts-about-speech-intelligibility/ | Speech above noise (A7) |
| 219 | Meyer et al., Loud and shouted speech perception at variable distances in a forest (Interspeech 2018 abstract) (W) | https://www.isca-archive.org/interspeech_2018/meyer18b_interspeech.html | Word recognition at 30–90 m (A5, A-T10) |
| 220 | Meyer, Dentel & Meunier 2013, Speech recognition in natural background noise (PLOS ONE) (W) | https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0079279 | Normal voice at 11–33 m (A5, A-T10) |
| 221 | Finlay 2024, Voice calling detection distance in land search and rescue (J. Search & Rescue 7(2)) (W) | https://journalofsar.com/wp-content/uploads/2024/11/1-Finlay-voice-calling-detection.pdf | Shout levels; intelligibility vs ambient noise; directivity (A1, A5, A-T10, A-T15) |
| 222 | Discrete metasurface for extreme sound transmission through the water–air interface (arXiv 2308.01043) (W) | https://arxiv.org/pdf/2308.01043 | Air–water loss ≈ 30 dB (A25) |
| 223 | Head-related transfer function measurements in a compartment fire (arXiv 2108.11465) (W) | https://arxiv.org/abs/2108.11465 | Fire changes received level > 10 dB (A26) |

### 5.9 Perception

| # | Title | URL | Used for |
|---|---|---|---|
| 224 | Wikipedia, Field of view | https://en.wikipedia.org/wiki/Field_of_view | Field of view (P1, P3, P-T7) |
| 225 | Wikipedia, Visual field | https://en.wikipedia.org/wiki/Visual_field | Per-eye limits (P1) |
| 226 | Wikipedia, Peripheral vision | https://en.wikipedia.org/wiki/Peripheral_vision | Peripheral acuity and motion (P1–P3, P-T7) |
| 227 | Krisciunas & Carona 2015, At what distance can the human eye detect a candle flame? | https://arxiv.org/abs/1507.06270 | Flame detection at night (P4, P-T1, P-T11) |
| 228 | Wikipedia, Lux | https://en.wikipedia.org/wiki/Lux | Light levels (P5, P-T11) |
| 229 | Wikipedia, Scotopic vision | https://en.wikipedia.org/wiki/Scotopic_vision | Night vision range (P5) |
| 230 | Wikipedia, Adaptation (eye) | https://en.wikipedia.org/wiki/Adaptation_(eye) | Dark adaptation (P5) |
| 231 | Simulated viewing distance and the eyewitness confidence–accuracy relation (PMC9240149) | https://pmc.ncbi.nlm.nih.gov/articles/PMC9240149/ | Identification at distance (P6, P-T11) |
| 232 | Jin 1997, Studies on human behavior and tenability in fire smoke (IAFSS) | https://publications.iafss.org/publications/fss/5/3/view/fss_5-3.pdf | Visibility law; tenability; irritant smoke (P7, P8, P-T3) |
| 233 | Borger et al. 2024, A waypoint-based approach to visibility (arXiv 2404.11439) | https://arxiv.org/pdf/2404.11439 | Visibility constants; guidance (P7, P8) |
| 234 | ABCB FCRC-TR 97-11, Selected literature reviews on human behaviour in fire | https://www.abcb.gov.au/sites/default/files/resources/2020//FCRC-TR-97-11-Selected-literature-reviews-human-behaviour-fire.pdf | First cues; moving through smoke (P9–P11, P-T4, P-T8) |
| 235 | Wikipedia, Mental chronometry | https://en.wikipedia.org/wiki/Mental_chronometry | Reaction times; Hick's law (P12, P13, P-T5) |
| 236 | Topend Sports, Ruler drop test | https://www.topendsports.com/testing/tests/reaction-stick.htm | Ruler-drop norms (P12, P-T6) |
| 237 | Visual Expert, Driver reaction time (Green 2000) | https://www.visualexpert.com/Resources/reactiontime.html | Expected vs surprise reaction (P13, P-T5) |
| 238 | Wikipedia, Startle response | https://en.wikipedia.org/wiki/Startle_response | Startle latencies (P13, P-T12) |
| 239 | Wikipedia, Saccade | https://en.wikipedia.org/wiki/Saccade | Saccade latency (P14, P-T7, P-T12) |
| 240 | Proulx & Fahy 1997, The time delay to start evacuation (IAFSS) | https://publications.iafss.org/publications/fss/5/783/view/fss_5-783.pdf | Pre-movement times (P15, P-T2) |
| 241 | NIST NCSTAR 1-7, WTC occupant behavior | https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=101046 | WTC pre-movement (P15, P-T2) |
| 242 | NIST, Key findings of the Station nightclub fire investigation | https://www.nist.gov/el/key-findings-and-recommendations-improvement-nist-investigation-station-nightclub-fire | Recognition delay with visible fire (P16, P-T1) |
| 243 | SFPE, Fire Protection Engineering magazine, 2006 Q3 | https://www.sfpe.org/publications/fpemagazine/fpearchives/2006q3/2006q34 | Station nightclub timeline (P16) |
| 244 | Wikipedia, Bradford City stadium fire | https://en.wikipedia.org/wiki/Bradford_City_stadium_fire | Televised fire timeline (P16) |
| 245 | Leach 2004, Why people freeze in an emergency | https://www.hptc-pro.com/wp-content/uploads/2014/01/Why-People-Freeze.pdf | Response mix; decision times (P13, P17, P-T9) |
| 246 | Fire Adapted Communities, The panic myth | https://fireadaptednetwork.org/the-panic-myth-what-does-the-research-say-and-what-can-practitioners-do/ | Panic is rare (P18, P-T9) |
| 247 | Bruck & Ball 2007, Optimizing emergency awakening to audible smoke alarms | https://vuir.vu.edu.au/676/1/optimizingBruck.pdf | Waking thresholds; risk factors (A22, P23, A-T5, A-T11) |
| 248 | Carskadon & Herz 2004, Minimal olfactory perception during sleep (abstract read through NCBI E-utilities) | https://pubmed.ncbi.nlm.nih.gov/15164891/ | Smell does not wake (P19, A-T5) |
| 249 | Grant, Millar & Kenny 2000, Blood alcohol concentration and psychomotor effects (Br J Anaesth, PMID 11103181; abstract read through Europe PMC) (W) | https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=Grant%20Millar%20Kenny%202000%20blood%20alcohol%20choice%20reaction%20time%20Br%20J%20Anaesth&format=json&resultType=core | Alcohol and reaction time (P23) |
| 250 | Nutrients 2016, metabolic equivalent (1 MET = 3.5 ml O2/kg/min) | https://doi.org/10.3390/nu8070438 | Body O2 use (B26) |
| 251 | Wikipedia, TRPV1 (W) | https://en.wikipedia.org/wiki/TRPV1 | Heat activation above 43 °C (B18) |
