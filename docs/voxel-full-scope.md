# Voxel core — the fully scoped target

What the finished physical layer looks like, subsystem by subsystem. Each section
has three rungs: WHAT PHYSICS SAYS (the real model, with the reference to copy
from), WHAT WE RUN TODAY (the approximation and its known lies), and the TARGET
(the voxel-affordable version we are building toward). Ruling 1 applies at every
rung: model the causes, never the cases — an upgrade replaces a tuned constant
with a derivation, never with a flag.

The guard against "fixing one scenario breaks another": every scenario's promises
are pinned as tests (tests/test_voxel.py), and every upgrade re-renders the
scenario suite. Realism ratchets forward; it is not allowed to slosh.

## 0. Time and units — the foundation everything else stands on

- PHYSICS: one clock. Rates, velocities, and energies are only comparable when
  they share a Δt and SI units.
- TODAY: two clocks. Combustion is calibrated so a tick ≈ 1–3 s (char creep
  matches real mm/s); gravity (0.25 voxel/tick²) implies a tick ≈ 40 ms. Falls
  and fires disagree by ~50×. Splash velocities, topple ω caps, drop gravity are
  all tuned by eye.
- TARGET: declare `TICK_S` once. Derive gravity in voxels/tick², burn g/tick,
  conduction J/tick, drop and body velocities from it. Every constant that
  cannot be derived gets a `# calibration:` comment naming its source.

## 1. Gas phase (air, smoke, oxygen, vapors)

- PHYSICS: compressible Navier–Stokes with buoyancy; species carried by
  advection–diffusion. The gold-standard open implementation is **NIST FDS
  (Fire Dynamics Simulator)** — open source, and its Technical Reference Guide
  is a complete recipe for LES fire flow, combustion, and species transport.
  The cheap-but-honest fallback is **NIST CFAST**, a two-zone model: every room
  is a hot upper layer + cool lower layer with a neutral plane — exactly the
  "smoke banks down, crawl low" structure.
- TODAY: coarse lossy pressure+velocity cells (4³ voxels), buoyancy as a tuned
  constant (0.0015·warm), wind advects SMOKE ONLY. Oxygen moves by pair
  diffusion plus a 0.1/tick pull toward the REGION MEAN — an instantly stirred
  lung. Lies: head air = knee air always (no crawl-low), no fresh river through
  a doorway, a fire's deficit teleports to every nose in the airspace, the
  lattice boundary is a sealed terrarium with a one-layer sky vent.
- TARGET (two steps):
  1. Kill the region mean-mix. O2 (and each gas species) rides the SAME wind
     field as smoke, with buoyant stratification: hot depleted gas up, cool
     fresh air down — the CFAST two-layer structure emerging on the voxel grid.
     (The recorded failure to avoid repeating: naive O2 advection smothered
     fires 21-vs-1 before local entrainment health existed. Rebalance with A/B
     scenario measurements, not by feel.)
  2. Gas species (roadmap #2): smoke becomes (species, grams) — O2, CO2, CO,
     H2O vapor, soot, fuel vapor, acid fume. Closes mass conservation for
     boiling (today boiled water's mass VANISHES into a pressure nudge), splits
     acid smog from fire smoke, enables flame-above-fuel (volatiles burn in the
     gas), CO poisoning, steam scalds, backdrafts.

## 2. Combustion

- PHYSICS: pyrolysis (solid → volatiles + char, Arrhenius-rated) feeding
  gas-phase flaming combustion; references: FDS combustion chapter, the SFPE
  Handbook of Fire Protection Engineering. Wood is ~80% volatiles / ~20% fixed
  carbon — our char split is already this datum.
- TODAY: solid burns in place when T ≥ ignition with O2 rationed from its
  airspace; real heats of combustion; char conversion below 18% mass. Lies: no
  flame above the fuel (a torch's fire sits ON the handle), no smolder-vs-flame
  distinction, no fuel vapor (no flash fires), radiation constant is tuned.
- TARGET: pyrolysis emits fuel-vapor species; the EXOTHERMIC step moves to the
  gas phase where vapor meets O2 — flames become a place in the air. Smolder =
  the surface char oxidation path (already half-built as CHAR). Ignition
  becomes an Arrhenius rate, killing the hard threshold.

## 3. Liquids and solutions

- PHYSICS: liquids are mixtures. Composition determines density, viscosity,
  heat capacity, color, reactivity (rate ∝ concentration, Arrhenius in T),
  boiling (Raoult's law for mixtures), miscibility (polarity). References: CRC
  Handbook, any physical-chemistry text; PubChem for per-species scalars.
- TODAY: one fluid id + volume + ONE solute scalar (fpot) per voxel; MISCIBLE
  is a decree table; bulk properties ignore concentration; only WATER boils
  (a case in the law!); boiled mass is not conserved. fpot is honest as a
  2-species mixture but cannot say "salt AND acid in this water", and a
  half-water acid still weighs and flows like vitriol.
- TARGET: per-voxel composition vector (species → grams), starting with 3–4
  slots. Bulk properties mass-weighted from per-species data rows. REACTIONS
  keyed by species with rate ∝ concentration. Boiling reads a per-species
  boiling point + latent heat column and emits vapor species (needs gas
  species first, or mass vanishes). fpot collapses into this cleanly — it IS
  the 2-slot case. MISCIBLE dies, replaced by a polarity column.

## 4. Solid mechanics — strength, fracture, impact

- PHYSICS: stress = load / cross-section vs material strength (compressive,
  tensile); beams by Euler–Bernoulli; brittle fracture by the Griffith
  criterion; impact by toughness (Charpy/Izod tables: glass ~0.1–0.3 kJ/m²,
  wood ~5–10 along grain, mild steel 100+); thermal shock when ΔT stress beats
  tensile strength (soda-lime glass cracks near ΔT ≈ 60–90 °C, borosilicate
  ~160). Data: MatWeb-class tables, Engineering ToolBox, FSRI MaP.
- TODAY: strength is SPAN (overhang hops); load MASS is invisible (anvil =
  feather); nothing shatters — landing energy is discarded; a 3-leg table
  cannot tip from load. Objects are same-material clusters (a table with iron
  legs is two objects); no joints.
- TARGET (ordered):
  1. IMPACT TOUGHNESS (building now): falling voxels accumulate real drop
     height; landing energy per contact cell vs toughness × face area decides
     shatter; free bodies use their COM potential-energy drop. Glass bursts,
     wood thuds. Fluid inside a burst container SPILLS.
  2. THERMAL SHOCK (building now): a ΔT column for brittle materials; the
     conduction field already knows every face's ΔT.
  3. Mass-scaled strength: column load / cross-section vs compressive strength
     (kills anvil-=-feather).
  4. Object identity with joints: connected components across materials with a
     joint-strength table; topple and impact operate on assemblies.

## 5. Physiology

- PHYSICS: oxygen via the hemoglobin dissociation curve (Hill equation); CO
  binds Hb ~250× stronger; the fire-engineering standard is **Purser's
  Fractional Effective Dose (FED)** model (SFPE Handbook chapter) — incapacity
  and death as integrated dose of CO, CO2, low O2, heat, with published
  coefficients. Burns by heat flux dose (Henriques integral).
- TODAY: blood O2 relaxes toward the air at the body's top quarter; smoke load
  blocks uptake (a stand-in for CO, since CO doesn't exist as a species yet);
  burn integral over skin T; thresholds for faint/death. Honest shape, made-up
  coefficients.
- TARGET: adopt Purser FED verbatim once gas species exist (CO, CO2 real);
  Hill-curve mapping for O2; burn model on heat flux. The collapse-on-faint
  promote/topple stays — that part is already causal.

## 6. Parked (scoped, not scheduled)

- Electricity: conductivity column, sources, resistive heating (Principia's
  component vocabulary for devices).
- Light/sight: line-of-sight through smoke density for NPC senses.
- Sound: event loudness + distance for NPC attention.
- Weather: rain as drop spawner, wind as boundary condition, temperature cycle.

## The wild-scenario test (why full scope matters)

Scenarios the fpot regime cannot express but the composition+species+mechanics
target can: brine evaporating to salt crust; quicklime hissing in rain; a
backdraft when a door opens on a smoldering room; a glass of water shattering
in a bonfire and quenching a hand-span of embers; chlorine-ish heavy fume
hugging the floor while hot smoke rides the ceiling; tempering a blade (heat →
quench → toughness column changes); a dam of ash washing away as slurry. Each
falls out of the same few laws + data rows — none gets its own code.

## Reference stack (all free online)

| Domain | Reference |
|---|---|
| Fire/gas | NIST FDS Technical Reference Guide; NIST CFAST (two-zone) |
| Toxicity/incapacitation | Purser, "Toxicity Assessment of Combustion Products" (SFPE Handbook) — the FED model |
| Material fire data | FSRI MaP database (measured CSVs, GitHub) |
| Chemistry scalars | PubChem PUG-View; CRC Handbook; NIST WebBook (manual) |
| Impact/strength | Charpy/Izod tables (MatWeb-class), Engineering ToolBox, Griffith criterion |
| Mixtures | Raoult's law, standard phys-chem texts |
| Sandbox craft | Noita GDC talk (promote/demote), Powder Toy (ideas only, GPL) |
