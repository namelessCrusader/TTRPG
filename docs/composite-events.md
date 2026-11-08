# Composite events and the basic laws

*Read-only study, 2026-10-18. No repo file was changed. Scratch files are in `/home/nisargparikh/.cache/mark1-build/composite/`: `matrix.tsv` (169 events × laws) and `analyse.py`.*

---

## 1. The answer in one paragraph

About 22 general laws, plus burning (kept as its own law) and one input layer for the outside world, are enough for farming, smithing, tanning and 160 other composite events to emerge with no craft code. We broke 169 events from nine families (metal and stone, food, fibre and wood, growing, body and medicine, fire and household, building and machines, nature, information) into basic events. Then we attacked the law set with 25 new events. The core is 12 laws: heat, liquids, gas, burning, transport through solids, rate chemistry, mechanics, separating and joining, slow time, senses, mixture equilibria and outside inputs. Every craft needs several of these, but together they finish only 7 events alone. After the core, each further law unlocks 6 to 22 events. Three pieces of plumbing come before any law, and they matter as much as the laws. The first is a **composition vector** per cell, so leather, rust, lime and steel are states, not new materials. The second is **records for anything thinner than a cell** (a blade, a hide, a coin, a letter, a vessel), with one hand-over rule that keeps mass and energy exact. The third is a **surface-layer stack** per face. The set cannot do some things on its own. Market prices and rumour across a town are not physics; they need an offscreen summary of many minds. Magic comes later, as packs on top. Wave physics (real acoustics, optics on a grid, detonation) stays as closed-form stand-ins by design. And many numbers are still unverified and must be sourced before they become tests.

---

## 2. The basic laws

How to read this section:
- Ids L1 to L13 keep your numbering. L14 to L22 are new. **Bn** is burning, kept as its own law (decision 4). **Fx** is the outside world: an input layer, not a law.
- Sub-items (L2a to L2d, L5g) are separate build items inside one law family.
- Costs are at 5 cm cells and 25 ms ticks. **(m)** means measured in `docs/realism-map.md` §5; every other cost is an estimate.
- Levels: **W** world law per tick; **W-ev** world law on an event; **W slow** a slow clock; **O** offscreen closed form; **M** mind or act; **R** reflex.

### L1 Heat
- **Does:** conduction; convection; radiation between faces, across air (G7), to a cold sky and from the sun; glow. Phase change by enthalpy, with latent pinning: melt, freeze, boil, condense, sublimate. The boiling curve on wet faces (film boiling, nucleate boiling, Leidenfrost). Phase temperatures come from L14, never from constants.
- **State:** E and T per cell; skin T; node temperatures inside records.
- **Parameters:** c, k, ρ, emissivity, absorptivity, latent heats.
- **Have:** `_law_conduct` (heat.py:431); face radiation to a fixed 20 °C (heat.py:518); `_law_boil` (heat.py:1101), which deletes the steam; `_law_melt` (heat.py:1123), metals only; a 0 °C floor (sim.py:385).
- **Add:**
  - an ICE row, and remove the floor;
  - a sky sink and solar gain;
  - G7 radiation across air;
  - the boiling curve;
  - steam handed to L2b, not deleted;
  - phase points read from L14;
  - an implicit 1-D solve per record (the explicit limit at 0.75 mm nodes is about 23 ms, right at the tick).
- **Cost:** radiation ≈1 ms/tick (m); sky and solar under 0.1 ms; records 10–50 µs each. **Level:** W.

### L2a Bulk liquids
- **Does:** lattice flow plus a 2-D height field with momentum. Buoyancy and drag on things, members and bodies. Orifice and weir flow for leaks and doorways. Force on oriented faces (lift, see §5). Buoyancy also in gas.
- **State:** fluid id and volume; column depth and velocity.
- **Parameters:** ρ, viscosity, surface tension.
- **Have:** `_law_head` and `_law_flow` (fluids.py:97, 170). The drowning clock is right. A person cannot enter water.
- **Add:** water sleep, the height field, buoyancy per member and body, a person in water, orifice flow, and face forces.
- **Cost:** height field 0.026–2.5 ms (m); still water ≈0. **Level:** W; O.

### L2b Gas carrier with species
- **Does:** a 20 cm low-Mach buoyant projection. Species (O2, CO, CO2, water vapour, CH4, tar, odours, pathogen quanta) each sink or rise by their own density. Forced draft from bellows, breath and wind.
- **State:** velocity, T and a species vector per coarse cell.
- **Parameters:** molar mass, diffusivity.
- **Have:** `_law_o2` and `_law_smoke` (air.py:296, 190); `_law_rise` (heat.py:548). Smoke is 39× too clear today.
- **Add:** lane 7 projection; species; vapour; draft.
- **Cost:** 2–5 ms converged; +0.033 ms per species (m). **Level:** W.

### L2c Dispersed phases
- **Does:** anything carried in a fluid has a size and a density, so it has a settling or rising speed and a lift threshold. It moves with the flow and settles where the flow slows. This covers embers, sparks, drops, snow drift, silt, dust, a flour cloud, chaff, rising cream, lees, and pathogens riding water.
- **State:** densities per class per coarse cell, or particles.
- **Have:** drops (sim.py:244); MPM particles.
- **Add:** class densities with settling and lift; ember particles.
- **Cost:** ≈0.05 ms per class. **Level:** W; O.

### L2d Confined gas chambers
- **Does:** a chamber record obeys p = nRT/V. Gas comes from L4 yields and L1 heating, and leaves through vents (orifice or choked flow). It pushes a moving boundary (a ball, a bellows plate). It bursts when the hoop stress p·r/t passes the wall's strength, then hands over to the blast event. A room can act as a chamber for a premixed burn (§5). It also holds pore steam.
- **State:** n, T, V and vent area per chamber.
- **Have:** pressure parcels inside `_law_boil`.
- **Add:** chamber records, venting, and the burst hand-off.
- **Cost:** µs per chamber; blast 10.6 ms per event (m). **Level:** W-ev.

### Bn Burning (its own law)
- **Does:** flaming, glowing and smouldering behind an O2 gate. Each flame is an entity with power Q, height, light and a blow-off speed.
  - Fineness σ (surface over volume) sets when thin fuel ignites.
  - Flame temperature comes from the energy balance plus forced draft.
  - CO yield depends on the burn mode.
  - A premixed mode (§5) adds flammability limits, minimum ignition energy and a flame speed, for dust and gas explosions.
  - It shares the L4 row format.
- **State:** σ, char depth per face, fuel moisture (from L3).
- **Have:** `_law_burn` (heat.py:613); PYRO; GLOW for char only; the BURN_AREA fudge (materials.py:562); COMBUST_CEIL 1100 (materials.py:486); 100 kW scene tapers.
- **Add:** the flame entity; σ in place of BURN_AREA; derived flame temperature and bellows (which removes COMBUST_CEIL); smouldering and the smoulder-to-flame transition; CO by mode; the premixed mode.
- **Cost:** burning 12.8 ms in house_fire (m), unchanged; flames under 1 ms. **Level:** W.

### L3 Transport of species through solids and pores
- **Does:** Fick or Darcy transport of water, solutes, gases and interstitial atoms through any host. The host can be porous (soil, wood, hide, bread, ash) or dense (carbon in steel). The driving force is chemical potential, which for water is water potential or water activity aw.
  - It covers wicking, capillary rise, sorption toward the air's equilibrium moisture, drying below 100 °C, leaching, osmosis, infiltration, gas diffusing into ore lumps and mortar, and diffusion bonds.
  - Microbe densities ride the flow (§5).
- **State:** bound moisture w per cell; node profiles of water and solutes in records.
- **Parameters:** D(T) per (species, host) pair; permeability; sorption isotherm (from L14).
- **Have:** porosity from smass (fields.py:54, 136); one solute slot `fpot` (sim.py:173); FLUIDS lane step 4 is planned.
- **Add:**
  - the w field;
  - evaporation by mass transfer;
  - sorption;
  - a solute vector;
  - Fick in hosts with the erf closed form;
  - 2-D soil columns.
- **Cost:** ≈0.1 ms/tick on a 1 Hz slow clock; µs per record. **Level:** W slow; O.

### L4 Rate chemistry
- **Does:** reaction rows. Reactants can be gas, liquid, solid, dissolved or bound. Products can be gas, solute, bound solid or residue.
  - Each row has an Arrhenius rate and a heat of reaction.
  - Driver columns: temperature, concentration, light, catalyst (an enzyme that denatures by its own row), accumulated work, and a saturable Michaelis-Menten form.
  - Gates read from L14: aw, pH, equilibrium CO2 pressure.
  - Electrochemical pair rows act between two touching hosts that share an electrolyte (§5).
- **What it covers:** pyrolysis, Boudouard, ore reduction, lime burning and slaking, rust, saponification, Maillard browning, denaturation, ink oxidation, bleaching, alcohol elimination.
- **Have:** REACTIONS (materials.py:111, fluids.py:288): fluid eats solid, linear in concentration, no temperature term, products go to smoke.
- **Add:** the full row format.
- **Cost:** under 0.5 ms/tick on the active set; offscreen otherwise. **Level:** W slow; O.

### L5 Mechanics
- **Does:** support, contact, friction per material pair (wet, oiled, filmed), capstan friction, impact, fracture with a damage scalar that accumulates, and bonds.
  - A beam, column and thrust check tells compression from tension and takes live loads and sideways loads (m·a).
  - Strength comes from the section that is left after char, rot, rust, notches or spalling.
  - Lost mechanical work becomes heat in the two contact skins.
  - Contact pressure is computed, for example a hand on a wound, or a wrap with hoop pressure P = T/(r·w).
  - Small things below cell size move as rigid bodies (Newton-Euler) (§5).
- **Parameters:** E, ν, tensile and compressive strength, toughness, a table of μ per pair.
- **Have:** SPAN, which is load-blind (support.py:12); `_law_torque` (support.py:546); `_shatter`; bonds with grain; FRICTION is one number, 0.40 (materials.py:437); `_rub` (hands.py:483); a declared `edge` (sim.py:212).
- **Add:** the one material record; μ per pair; the beam check; damage; strength from the remaining section; contact heat; rigid bodies below cell size.
- **Cost:** checks on change ≈0; a collapse 30–60 ms/tick for 1–3 s. **Level:** W-ev.

### L5g Granular matter and soil (a branch of L5)
- **Does:** Mohr-Coulomb strength on effective stress, τ = c' + (σ − u)·tanφ'. Pore pressure u comes from L3. Cohesion falls with water and rises with freezing or sintering.
  - It gives slump to the angle of repose, the Rankine critical height for a cut wall, slope safety per column, shear on a snow weak layer, bearing and sinkage (Bekker, Terzaghi), bulking and compaction.
  - Grain and flour heaps follow it too.
- **State:** packing (void fraction), moisture.
- **Parameters:** c', φ', unit weight γ by texture.
- **Have:** nothing. All ground is STONE.
- **Add:** soil and snow rows; slump on active cells; slope checks per column.
- **Cost:** ≈0 per tick (it runs on change or hourly); an avalanche runs MPM for that event only. **Level:** W-ev; O.

### L6 Rheology (shape change)
- **Does:** flow stress σ = σ_y(T, composition, structure, strain) + η(T)·(strain rate)ⁿ.
  - The same law covers hot metal (Zener-Hollomon, work hardening), glass (VFT viscosity), wet clay (Bingham, by water content), dough, putty, wax, and creep (lead, ice, a bow taking a set).
  - Volume is conserved on a shape record.
  - It relaxes L15 residual stress, which is what annealing is.
  - It gives indentation hardness (a bite, a file).
- **State:** shape record, plastic strain.
- **Parameters:** fits for σ_y and η per material.
- **Have:** none. MPM forging is forbidden (realism map §4).
- **Add:** the fits, and a blow converted to displaced volume.
- **Cost:** µs per blow. **Level:** W-ev.

### L7 Structure state
- **Does:** the internal structure of a material, driven by temperature history, cooling rate, accumulated work (J/kg), moisture and chemistry.
  - Examples: steel phases (JMAK with Scheil additivity, Koistinen-Marburger, Hollomon-Jaffe tempering), setting and curing (mortar, plaster, glue, fired clay, gels, sealing wax), crystallising (sugar, fat, stale bread), sintering, the shrinkage temperature Ts of collagen, gluten, and snow metamorphism.
  - It also holds the fibre network: grain orientation, twist, bonded fraction, entanglement.
  - Derived outputs: hardness, toughness, stiffness, a strength factor, permeability.
- **State:** a few named fractions per record or cell.
- **Have:** none.
- **Add:** kinetics rows, and derived hardness exposed in `props` (things.py:523).
- **Cost:** µs per record on a slow clock. **Level:** W slow; O.

### L8 Living tissue
- **Does:** metabolism and energy, water and nutrient stores. A transport fluid (blood, sap, lymph) with one compartment per substance, which is pharmacokinetics.
  - Plants grow by intercepted light × radiation-use efficiency, by degree-days and by photoperiod.
  - Healing: wound strength by fraction; bone healing gated by the strain across the gap.
  - An immune kill term proportional to perfusion.
  - Death is a process, with a window for revival. Chemistry runs on after death.
  - Reproduction. Heredity (a parameter mix) is deferred.
- **State:** records per person and per body region; plot cohorts for plants.
- **Parameters:** species sheets.
- **Have:** `_breathe`, `_law_bleed`, `_perfusion`, `_law_life` (physiology.py). Flesh sits at 20 °C.
- **Add:** body heat (decision 1), stores, compartments, healing, immune response, death as a process, plant cohorts.
- **Cost:** a few µs per person (m); plots 0.18 ms per 400² cells per sim-hour (m). **Level:** W at 1 Hz; O.

### L9 Populations as density fields
- **Does:** microbes, fungi, insects, vermin, game, fish, weeds and bee colonies are densities, not bodies.
  - Growth depends on T (Ratkowsky), aw, pH, salt, O2, food, light and inhibitors, with a logistic cap.
  - Their products and heat go through L4 rows (acid, ethanol, CO2, toxins).
  - Heat kills them by D and z values.
  - They diffuse, move toward food (taxis) and prey on each other (Holling II).
  - An in-host form runs against the L8 immune term.
  - A spawn and absorb seam turns density into individual bodies near people, keeping the count.
- **Have:** none.
- **Cost:** ≈0.02 ms/tick; mostly offscreen. **Level:** O; W slow.

### L10 Separating and joining
- **Does:**
  - **Cutting and splitting** per stroke: energy over specific cutting energy, with grain direction.
  - **Grinding,** which raises surface area.
  - **Archard wear,** V = K·F·s/H: sharpening, dulling, filing, the file test, chafe, scrubbing. The edge radius is state.
  - **Fast penetration** layer by layer (§5).
  - **Joining:** weld strength from temperature, pressure, time and oxide; glue cure; nail and peg withdrawal; seams; lashings; knots; caulk.
  - Every joint is a record with a capacity and a failure law.
- **Have:** a swing shatters a whole cell; `edge` is constant; bonds can join and untie (store.py:593–655).
- **Add:** removal per stroke, Archard wear, penetration, joint records.
- **Cost:** µs per event. **Level:** M then W-ev.

### L11 Slow time (the execution layer)
- **Does:** a scheduler plus a library of closed forms:
  - erf (Fick) for a slab;
  - first-order Arrhenius;
  - Baranyi microbial growth over a temperature schedule;
  - Stefan ice growth;
  - √t carbonation;
  - ISO 9223 rust;
  - drying periods;
  - SEIR epidemics;
  - Omori aftershocks;
  - adaptation (L22).

  It hands a process back to the world laws when someone watches.
- **Have:** an integer tick, and the slow-row clock in `_law_acid`.
- **Cost:** 0.5 µs per event (m). **Level:** O.

### L12 Signal fields and senses
- **Does:**
  - **Light** as W/m² by band: lumens for sight, PAR for plants, heat for L1, a rate driver for L4.
  - **Mirrors and lenses:** light from small sources is traced through mirror and lens records, so a burning glass focuses sunlight and a fish appears where the bent ray says (§5).
  - **Sound** with an arrival delay of distance / 343 m/s.
  - **Odour and taste** species with thresholds.
  - **Touch:** pressure, temperature, and weight change against a Weber fraction.
  - **Attention load.**
  - **Long range:** horizon distance and contrast.
- **Have:** `_sees` by optical depth (senses.py:569); `_law_hear` (senses.py:724). There is no light, and every line is spoken at 90 dB.
- **Add:** all of the above.
- **Cost:** light 1–3 ms/tick (m); hearing batched 1–2 ms; 0.05 ms per odour species. **Level:** W on change.

### L13 Surface layers and marks
- **Does:** a stack of layers per face, item or body region. Examples: oxide and temper film, scale, rust, char or spall depth, soot, creosote, wax, grease, blood, plating, paint, ink, chalk, makeup, frost, dew, poison residue, an ash cover, and relief marks (seal, stamp, footprint, disturbed ground).
  - Layers carry mass, composition and adhesion.
  - They grow by condensation, settling, contact transfer and freezing. They shrink by evaporation, dissolving, burning and abrasion.
  - A liquid layer above its holding limit drains downhill and drips (§5).
  - Contact transfer copies relief as a mirror image (§5).
  - Layers change emissivity, friction, ignition and appearance, and can carry content (L19).
- **Have:** nothing. The heat skin is the only per-face state.
- **Add:** a sparse dict keyed by (host, face).
- **Cost:** ≈0 (driven by events). **Level:** W-ev; O.

### L14 Mixture equilibria (new)
- **Does:** state functions of composition and temperature, not rates:
  - phase points: boiling-point rise, freezing-point drop, liquidus and solidus (bronze, slag, iron-carbon), eutectics (salt and ice at −21.1 °C);
  - vapour-liquid equilibrium (the boiling point of syrup, the enrichment of spirit in a still);
  - solubility;
  - Henry's law;
  - water potential and aw;
  - pH;
  - gas-solid equilibria (limestone at 898 °C under 1 atm of CO2; the carbon potential of CO/CO2);
  - electrode potentials for L4 pair rows.
- **Why it is a law:** L1 knows only pure fixed points, and L4 knows only how fast. Without L14 a syrup never passes 100 °C, distilling enriches nothing, salt does nothing to ice, and no preservation method has a gate.
- **Have:** DILUTION and MISCIBLE only (materials.py:156–197).
- **Cost:** µs, run when composition changes. **Level:** derived.

### L15 Free strain and self-stress (new)
- **Does:** every solid wants a free strain: α·ΔT + β·Δmoisture (direction-dependent by grain) + the volume change of a phase change + pore pressure.
  - Volume-change examples: ice +9%, martensite +1–4%, quartz at 573 °C, rust ×2–4, slaking lime ×2, drying clay −4–8%, shrinking collagen, felting.
  - Where the strain is restrained, stress σ ≈ E·ε*/(1−ν) goes to L5: crack, spall, burst, clamp or jam.
  - L6 relaxes it.
- **Replaces:** TSHOCK and `_law_crack`.
- **One law covers:** fire-setting, quench cracks, a cracked bell, a pot blowing out in the kiln, exploding river stones, Prince Rupert's drops, frost bursting a jar, rawhide bindings, a stuck door, a boat that "takes up", a shrunk-on iron tyre.
- **Cost:** 0.1–0.5 ms/tick on hot or wet active sets. **Level:** W; O.

### L16 Mechanisms (new)
- **Does:**
  - **Joint rows:** fixed, hinge, slide, rolling, rope over a pulley, gear. Each has friction.
  - **Catch rows** hold until a trigger: a force threshold (latch, bar, trap trigger, gun sear), a shape match (key against lock), a count (clock alarm) or a displacement (tripwire).
  - **Ratios** with an efficiency: levers, gears, pulleys.
  - **Springs** whose stiffness comes from L5 geometry and E, and whose elastic limit comes from L6 and L7 (§5).

  It is solved quasi-statically on each change, and by XPBD while parts move.
- **Have:** wrist lever statics (hands.py:401). Doors are cells at fill 0.97.
- **Cost:** µs per change. **Level:** W-ev.

### L17 Tissue function (new)
- **Does:** capacities computed at 1 Hz from local state. The capacities: force, reflex latency and gain, nerve conduction, breathing drive, consciousness, pain, gut motility, and the thermoregulation setpoint. The local state: T, O2, stores, structure, fatigue, sleep pressure, and bound drug by an Emax/Hill curve. An antidote shifts EC50. Plants close their stomata the same way.
- **Why it is a law:** this is what makes drunk, numb, sleepy, feverish and poisoned into physics instead of flags. The reflex contract reads it.
- **Have:** MET from movement only.
- **Cost:** µs per person. **Level:** W at 1 Hz.

### L18 Electromagnetism (new, mostly event-level)
- **Does:**
  - A resistive solve on coarse cells when a source fires (lightning; later galvanic cells or a magic pack). Conductivity comes from material, moisture and T. Joule heat goes into thin layers in L1, and current through a body goes to L8 and L17.
  - Persistent magnetisation on iron records (§5): set by stroking with a lodestone, lost above the Curie point and by hammering. Torque in an Earth field from Fx, so a compass works.
- **Cost:** 5–20 ms once per lightning stroke; µs per magnet. **Level:** W-ev.

### L19 Content by code (new, perception level)
- **Does:** any perceivable pattern (a mark, a sound sequence, a light, a flag, smoke puffs, a token) has a **form**, which must survive the channel, and **content** in a code K, with a claimed source and a true maker.
  - A reader who resolves the form and knows K gets claims.
  - Literacy means knowing a script and a language.
  - Ownership and debts are belief keys with sources, not a world ledger.
- **Have:** `_claim` and `_told` (beliefs.py:313–375), for the single claim "fire" only.
- **Cost:** µs per read. **Level:** M; percept.

### L20 Recognition by features (new, perception level)
- **Does:** features are derived from state, one channel at a time: a face minus whatever covers it, voice, gait, the style of a hand, seal relief, coin mass, colour and ring, ink age. They are measured with noise from the channel and from expertise, then matched against learned exemplars, giving identity with a confidence.
- **Replaces:** `_misread` rates (beliefs.py:277) and `_knows_voice` (beliefs.py:124).
- **Cost:** µs per look.

### L21 Vibration and sound emission (new, from the attack)
- **Does:** every impact, pluck, scrape or splash emits a sound source.
  - **Pitch** is the lowest mode of the struck record, by closed form: a string √(T/μ)/2L; a bar or plate from E, ρ and thickness; an air column c/4L; a cavity by Helmholtz.
  - **Level** is a share of the work L5 already books.
  - **Decay** is a loss factor that rises with damage.
- **Gives:** a cracked bell sounds dull, a coin rings or thuds, a hollow wall answers a knock, and a lute can be tuned. No waves are resolved.
- **Have:** `SOUND["ring"]` (materials.py:658) gives loudness only, at a fixed 500 Hz.
- **Cost:** µs per event. **Level:** W-ev.

### L22 Adaptation by use (new, from the attack)
- **Does:** each capacity parameter drifts toward a target set by a running integral of its load. It has a time constant of days to months, a ceiling from the sheet, and decays with disuse.
- **Covers:** drug tolerance, muscle from training, calluses, acclimatisation to heat and altitude, and skill from practice. This is the physical base of LitRPG levelling.
- **Have:** "skill per act" on the sheet, but no rule ever changes it.
- **Cost:** µs per person. **Level:** O, using the L11 closed form.

### Fx Outside inputs (an input layer, not a law)
- **Does:** sun and calendar, T_out(t), sky temperature, humidity, wind with seeded gusts, rain, snow and hail rates, tide harmonics, ground acceleration, the lightning schedule, and the Earth's magnetic field.
  - Weather moves as fronts with leading signs, so people can forecast it from clouds and wind (§5).
  - Every input is booked into the ledgers with a cause.
- **Have:** AMBIENT 20 °C (materials.py:10).
- **Cost:** µs per tick. **Level:** W boundary; O.

### Current code the study says must change

| # | code | problem | fix |
|---|---|---|---|
| 1 | COMBUST_CEIL 1100 (materials.py:486) | Welding heat (1250–1400 °C) and the tuyere (≈1300 °C) can never be reached. | Derive flame temperature, and add bellows. Fix the forge diagnosis (H23 "wrong") first. |
| 2 | `_law_boil` deletes the steam | No condensation, steam burns or distilling. Syrup is stuck at 100 °C. Nothing dries below 100 °C. This blocks about 40 events. | L1 hands steam to L2b; L3 adds evaporation; L14 sets the boiling point. |
| 3 | One `fpot` slot (sim.py:173) | Salt, sugar, ethanol and acid cannot coexist. | A solute vector. |
| 4 | REACTIONS (materials.py:111) | No temperature term, no products, no solubility limit, fluid-on-solid only. | The L4 row format. |
| 5 | TSHOCK={GLASS:120} | A case table (its 120 K is still plausible). | Fold into L15. |
| 6 | FRICTION 0.40; load-blind SPAN | One friction for every pair; supports ignore load. | μ per pair; the beam check. |
| 7 | `edge` is constant (sim.py:212) | Sharpness never changes. | A view of edge radius × edge length. |
| 8 | No ICE; 0 °C floor | Nothing freezes. | Decision 3. |
| 9 | One OIL row | Lamp oil and cooking fat flash about 200 K apart. | Split the rows. |
| 10 | `src/core/data/packs/farming.yaml` | Tilling is a tag; crops grow +0.2 per watered tick. This is exactly the case code the principle retires. | Keep it as an old-engine reference only. |

---

## 3. Shared state, and the rule for things smaller than a cell

### What to store
**Store only what changes and cannot be derived.** Derive everything else from composition and structure: aw, pH, Ts, hardness, colour, strength, conductivity, phase points.

**Per lattice cell (5 cm).** It keeps mat, smass, E/T, skin T and its fluid. Add:

| field | type | meaning |
|---|---|---|
| comp_id | int32 (−1 = row default) | Index into a sparse table of ≈16–24 species fractions. `mat` becomes "dominant species", so old laws keep working. |
| w | float32 | Bound moisture (g water per g dry). The frozen share comes from enthalpy. |
| dmg | float16 | Damage 0–1 (cracks, spall, decay); a strength multiplier. |
| solute index | int32 | Replaces `fpot`; points to the fluid's solute vector. |

That is about 5 MB at 0.4 M cells. Face layers live in a sparse dict. Residual stress and pore pressure live on the active set only.

**Per 20 cm gas cell:** velocity, T, species vector, dispersed-class densities, light.

**Per outdoor column (2-D, made into voxels only near people):**
- ground height and slope;
- soil layers (texture, w, ice, packing) and the water table;
- surface water depth and velocity;
- snow layers (thickness, density, grain type, T, weak-layer shear);
- lake ice thickness;
- fuel load, σ and moisture by size class;
- the plot cohort;
- marks.

**Per thing or record** (one schema; a field exists only when used):
- geometry class and real dimensions;
- composition per layer, and mass;
- a 1-D node profile (T, w, solutes, phase fractions, residual stress);
- L7 scalars;
- edge radius;
- L9 populations;
- L13 layers;
- containment host and held_by;
- L16 rows;
- L19 content;
- L2d chamber.

**Per body region:**
- perfusion and ischaemia time;
- a vessel table (diameter, depth, artery or vein, clot or tie);
- covering layers (armour, cloth, bandage, splint);
- a bone record (integrity, angle, gap, callus);
- microbe density;
- marks;
- skin and extremity nodes.

**Per person:**
- core T and setpoint;
- energy, water and nutrient stores;
- one compartment per substance;
- sleep pressure, fatigue, immune strength, heartbeat;
- features (L20), codes known (L19), exemplars, skills and adapted capacities (L22).

**World:** clock, calendar, latitude, weather state, tides, the aftershock and lightning schedules, the magnetic field.

### The sub-cell rule (one rule for every family)

1. **Voxels only when the decisive shape is at least one cell** (a beam, a wall, a bell, a loaf, a haystack). Anything thinner or smaller is a **record** on a host: a cell, thing, body region, container or plot.
2. **The host holds the totals** (mass, energy, grams of each species). The record only says how they are spread. Every exchange crosses the host's face laws, so the ledgers stay exact. *Test: the ledgers match across every hand-over.*
3. **Times use the real thickness, never the cell size.** t ~ L²/D, Biot, Stefan and erf all read the record's L. A 2 mm blade, a 4 mm hide, 0.1 mm paper, a 30 mm herring, a 1 cm snow weak layer and a 1 mm cambium channel each behave at their own size.
4. **Record classes:**

   | class | examples | what it carries |
   |---|---|---|
   | SECTION | bar, blade, beam, prop | segments × 5–9 through-thickness nodes; implicit solve |
   | SHEET | hide, cloth, paper, pot or glass wall, garment, frost | thickness, area, node profile |
   | STRAND | rope, thread, wick, lashing | diameter, path, twist |
   | CONDUIT | siphon, reed, straw, still pipe, blowpipe | bore, length, contents; 1-D flow between two ends; must be primed |
   | LOT | grain, seeds, beans, coins, shot, berries, dice | a count, a distribution and one representative profile |
   | VESSEL | cup, pot, cask, jar, still, keg, churn | contents vector, wall, headspace chamber |
   | JOINT | mortar, seam, nail, weld, glue line, knot, caulk, boat seam gap | capacity, cure fraction |
   | MECHANISM | lock, gun, clock, trap, mill, cart | L16 rows |
   | FACE LAYER | (L13) | |
   | DENSITY | (L9) | |
   | COLUMN | outdoors | |

   Lot transforms are analytic:
   - the share of the density histogram lighter than the brine floats;
   - threshing detaches the share whose bond is weaker than the blow;
   - winnowing splits the lot by terminal velocity against the wind.
5. **Never declare HARD, SHARP, SET, CURED, DONE, TANNED or SPOILED.** Read them from state:
   - hardness = carbon × martensite fraction, then tempering;
   - edge = edge radius × edge length;
   - tanned through = mid-node tannin above a threshold;
   - safe = a 6-log kill at the coldest node, or pH below 4.6 all through, or aw below 0.85.

   Thresholds are data that a reader compares against, never flags.
6. **Hand-over.** A record becomes lattice matter only when it spills, breaks, tips or grows past cell size (a cup spills, a candle tips, an ember lands in tinder). It goes back to a record when it shrinks below. The voxel footprint is re-stamped only when its size crosses a cell boundary.
7. **Living cohorts** carry stage, biomass, a parameter sheet and a viability spread. An individual is split out (eid, body, mind) only when someone interacts with it, and the count is kept both ways.
8. **Refuse these Ruling 2 traps:** PADDY, TILLED, SALTED_MEAT, LEATHER, RUSTY, ROTTEN, "good seed", a hay-ignition row, "poisoned", "cursed coin". Each one is base matter plus state.

---

## 4. Composite events, family by family

Notation: → means "then". Law ids are in brackets. Numbers marked (u) are unverified.

### 4.1 Metal, stone, glass and ceramics (18)

| event | chain of basic events | decision facts |
|---|---|---|
| E1 fire-setting a rock face | fire heats the face to 400–600 °C [Bn, L1] → the hot skin wants +0.3% and the cold core refuses: ≈200 MPa against 5–15 MPa tensile [L15] → spalls, damage builds [L5] → CO2 pools in the drift, wait it out [L2b, L12] → the pick removes damaged rock [L10] | Is it cracked enough to dig? Is the air breathable? How long until the fumes clear? Is the roof sound? |
| E2 tunnel, sap, shore | dig [L10, L5g] → props carry the roof [L5] → fire the props: char 0.6–0.8 mm/min, the section shrinks, the wall falls [Bn, L5] → seepage floods it [L3, L2a] → CO2 or firedamp [L2b, L4] | Will the roof hold without this prop? How long until it drops? Is water rising? Can I bring a lamp in? |
| E3 bloomery smelt | charcoal [L4] → roast the ore [L4, L15] → bellows 1200–1300 °C [L2b, Bn, L8] → Boudouard CO → ore reduced Fe2O3→Fe [L4, L3] → fayalite slag melts ≈1200 °C [L14, L2a] → bloom sinters [L7] → hot hammering [L6] | Is it hot enough (slag running)? Is there a bloom, and how much iron? Is CO poisoning me? |
| E4 casting | alloy melts over a mushy range [L14, L1] → a cold crucible cracks [L15] → dry the mould [L3] → pour; superheat sets how far it runs [L2a] → a wet mould makes a steam explosion [L2d] → solidifies, shrinks, hot tears [L1, L15] | Is it hot enough to pour? Is the mould dry? Has it set? Is it sound (ring test)? |
| E5 forging | working heat 950–1250 °C [Bn, L1] → read the colour [L12] → a 30–75 J blow moves ≈0.4 cm³ hot, 4–8× less cold [L6, L5] → work hardening and cracked edges when cold [L6, L7] → scale [L4, L13] → fatigue and noise [L8, L12] | Is it hot enough? Will it crack? How many heats are left? Is my arm tired? |
| E6 forge weld | welding heat 1300–1400 °C [Bn] → flux turns scale to slag [L4, L14] → hammer the faces together: diffusion bond [L6, L3, L10] → oxide or a cold face gives a weak "cold shut" [L10] | Is it at welding heat? Did the weld take (bend test)? Can I mend the sword with this fire? |
| E7 carburize / decarburize | pack in charcoal at 900–950 °C [Bn, L14] → carbon diffuses √(Dt), ≈0.7 mm in 8 h [L3, L11] → an oxidising fire strips carbon [L3, L4] | Will this iron harden at all? How deep is the case? Why is the edge soft? |
| E8 quench | austenitize 780–850 °C [L1, L7] → plunge; the boiling curve sets the cooling [L1] → beat the pearlite nose → martensite [L7] → volume change cracks or warps it [L15, L5] → a big piece in a small oil bath lights the oil [Bn] | Is it hard (does a file skate)? Did it crack? Is it warped? Water or oil? |
| E9 temper, anneal, file test | reheat 180–350 °C [L1] → the oxide film colour tracks its thickness [L4, L13, L12] → martensite tempers by Hollomon-Jaffe [L7] → slow cooling anneals [L7] → a file skates or bites [L10] | When to stop (the colour)? Springy or brittle? Can a file cut it? |
| E10 sharpen and dull | a whetstone removes metal by Archard [L10] → the edge radius falls to the grit floor [L10] → a dry wheel draws the temper [L1, L7] → use rolls or chips the edge [L10, L6, L5] → contact area feeds the blow law [L5] | Is it sharp? Will it hold through the fight? Is it chipped? |
| E11 cut through bars | file or saw for hours [L10, L11] → the rasp is audible [L12] → or acid for days [L4] → or bend it with a lever: ≈330 N·m to hinge it [L5, L6] → rust or a notch cuts the section [L5] | How many nights? Will the guard hear? Can I bend it instead? |
| E12 coin assay | weigh [L5] → density by displacement [L2a] → bite or scratch [L6] → touchstone streak plus nitric acid [L13, L4] → ring test [L21] → a clipped edge [L10] | Is it gold? Is it light or clipped? What is it worth? |
| E13 pottery | temper the clay; water sets plasticity [L6] → dry, shrink 4–8% [L3, L15] → fire slowly; trapped steam blows the pot out [L1, L3, L15] → dehydroxylation, quartz inversion, sintering [L4, L7] | Is it dry enough to fire? Is the kiln hot enough? Will it hold water? |
| E14 stones in the fire | a wet porous stone's pore water boils → pore pressure bursts it [L2d, L15, L5] → stone-boiling cracks the stones [L1, L15] | Are these stones safe for the fire ring? Will the cup crack on the coals? |
| E15 glass | the batch melts [L4, L1] → blow it at 1e3 Pa·s (VFT) [L6] → anneal near 1e12 Pa·s [L6, L15] → unannealed glass shatters later; Prince Rupert's drop [L15, L5] → thermal shock at ≈60 K [L15] | Soft enough to shape? Will it survive cooling? Break the window to escape? |
| E16 lime cycle | burn limestone (1.78 kJ/g, −44% mass) [L4, L14] → CO2 floods the area [L2b, L8] → slaking +1.16 kJ/g; wet lime can fire a ship and burst casks [L4, Bn, L15] → alkali burn [L4, L8] → carbonation ≈1 mm/√day [L3, L4, L11] | Is it fully burnt? Is lime safe in a leaking hold? Has the mortar set? |
| E17 a wall or arch holds | lay with mortar [M] → mortar strength grows with carbonation [L4, L3, L11] → the thrust line stays inside the section [L5] → rams accumulate damage [L5] → frost and rust jack it apart [L15] | Will it hold? Is it safe under the arch? Is the mortar set? |
| E18 rust | a water film plus O2 at ISO 9223 rates, gated by RH [L4, L3, Fx] → rust takes ×2–4 the volume and seizes the part [L15] → the section shrinks [L5] → oil, heat or vinegar frees it | Will the old lock break? Is the chain safe to climb? Is the barrow sword usable? |

### 4.2 Food and drink (20)

| event | chain | decision facts |
|---|---|---|
| F1 boil water | the fire heats the pot base through a 2 mm wall [Bn, L1] → 2 L takes 4–11 min [L1] → pinned at the boiling point, which L14 raises [L1, L14] → 1 min rolling boil kills pathogens [L9] → boils dry in ≈50 min, scorches [Bn, L4] → steam burns on lifting the lid [L1, L8] | Is it boiling? Safe to drink? Will it boil dry? Is the handle hot? |
| F2 roast meat | radiation onto the surface [L1] → surface pinned near 100 °C while wet [L3, L1] → core 60 °C in 8–15 min a side for 3 cm [L1] → proteins denature, collagen shrinks [L4, L7] → Maillard, char [L4, Bn] → fat drips and flares → kill at the coldest node [L9] | Done in the middle? Burnt outside? Safe? |
| F3 fry / pan fire | oil at 160–190 °C [L1] → food water flashes to steam [L1] → crust forms, oil soaks in [L3, L4] → past the smoke point, then the flash point: a pan fire [L4, Bn] → water on it causes a slopover; a lid smothers it [L2a, Bn] | Hot enough (sizzle)? Smoking? Will a lid put it out? |
| F4 bread | flour takes up water [L3] → kneading builds gluten (work dose) [L7] → yeast makes CO2 [L9, L4, L14] → the dough rises [L6] → oven 220–250 °C [L1] → starch gelatinises, the crust browns [L7, L4] → core 96–99 °C at 30–40 min → stales, moulds on day 4–6 [L7, L9, L11] | Has it risen? Is it done? How long will it keep? Is the kitchen too hot? |
| F5 coffee | roast to 200–230 °C [L1, L3, L4] → first and second crack from internal gas [L2d, L5, L21] → grind [L10] → extract 18–22% at 90–96 °C [L3] → caffeine half-life 5 h [L8, L17] | Roast done? Too bitter? Will I sleep? |
| F6 ale | steep and germinate barley [L3, L8] → kiln [L3] → mash at 62–72 °C; amylases denature [L4] → boil with hops [L1, L9] → ferment: 0.51 g ethanol per g sugar, heat [L9, L4] → CO2 pools in the cellar [L2b, L8] → a closed cask bursts [L2d] → sours [L9] | Is it beer yet? How strong? Is the cellar air safe? |
| F7 wine / vinegar | crush [L10] → ferment ≈10 days at 28 °C; sticks at 35 °C [L9] → a vat heats itself [L4, L1] → ≈50 m³ CO2 per 1000 L [L2b] → rack off the lees [L2c] → air plus Acetobacter makes vinegar [L9, L4] | Has it turned? Is the cellar air safe? Tampered with? |
| F8 distil and age brandy | a 10% wash boils at ≈92 °C with ≈50% ABV vapour [L14] → condenses in the worm [L1] → cuts: methanol in the heads [L14, L8] → vapour fire risk [Bn] → ages in oak; angel's share 2%/yr [L3, L4, L11] → the brand is a claim [L13, L19] | How strong? Safe? Is it really 7 years old? Will it light? |
| F9 cheese | warm the milk [L1] → lactic acid drops the pH [L9, L14] → rennet sets a gel [L4, L7] → cut and cook; whey leaves [L3, L7] → salt it [L3, L14] → ripen for months [L9, L11] | Will it keep? Good mould or bad? Is it ready? |
| F10 butter | cream rises (Stokes) [L2c] → ripen [L9] → 10–15 °C partly crystalline fat [L7] → churning work inverts the emulsion [L7] → rancidity [L4, L12] | Has it come? Is it rancid? Why won't it churn? |
| F11 salt-cure | salt dissolves to saturation [L14] → salt in, water out; a 30 mm herring reaches 4.7% in 8.7 h [L3] → aw falls, bacteria stop below 0.91 [L9, L14] → keeps a year cool and dry → soak before eating [L3] | Cured through? How long will it keep? Must I soak it? |
| F12 smoke food | smouldering hardwood [Bn] → smoke species deposit on the surface [L3, L2b] → antimicrobial skin, drying [L9, L4] → a case-hardened surface leaves the core wet [L3, L9] → fat fires and CO [Bn, L8] | Smoked enough? Cooked? Is the smokehouse full of CO? |
| F13 dry food | slice thin (time ∝ thickness²) → evaporation by mass transfer [L3, L2b] → aw below 0.85, then below 0.6 [L14] → humid air brings it back and it moulds [L3, L9] → rain wets the batch [Fx] | Dry enough to keep? Damp or weevily? |
| F14 pickle | salt 2–2.5% draws a brine [L14, L3] → lactic bacteria in sequence; pH 3.4–3.6 in 3–6 weeks [L9, L4] → pH 4.6 blocks botulinum [L9] → film at the air surface, softening [L9, L7] | Safe (pH through)? Off, or harmless film? Ready? |
| F15 jam, honey, mead | boil down [L3] → the boiling point rises to 104–105 °C at 65% sugar [L14] → pectin gels [L7] → a hot-filled jar seals by vacuum [L9, L1] → honey above ≈19% water ferments [L9] | Has it set? Will it keep? Hot enough to burn? |
| F16 cold store, ice cream | heat leaks into the store [L1, Fx] → ice melts [L1] → spoilage slows [L9] → foods freeze at −1 to −3 °C [L14] → salt plus ice falls toward −21 °C [L14, L1] → churned cream freezes fine [L7] | How long will the ice last? Will salt and ice freeze the cream? |
| F17 butcher | kill and bleed [L5, L8, L2a] → gut within ≈1 h, or gut bacteria invade [L10, L9] → pH 7.0 → 5.5, rigor 24–48 h [L8, L4] → skin, joint [L10] → chill below 7 °C [L1, L9] | Is the meat good? How much is there? Is it in rigor? |
| F18 food spoils | starting load → growth by T, aw, pH, salt and O2 (Baranyi) [L9, L11] → odour, slime, toxins [L4, L12] → bulk stores heat themselves [L9, L1] → rancid, stale, pests [L4, L7] | Can I eat it (toxins may not smell)? How long will it keep? |
| F19 forage, make food safe | find by season and light [L12, L8] → identify by features [L12] → leach acorns, boil beans ≥10 min at 100 °C [L3, L4] → dose and delay: amatoxin, lectins [L8, L17] | Edible? Lethal dose? Processed enough? When do symptoms show? |
| F20 eat and drink | pour and mix [L2a] → a poison dissolves [L14] → taste thresholds [L12] → gut absorption, Widmark distribution, elimination [L8, L4] → effects [L17] → food poisoning onset [L9] | Is it watered or poisoned? How drunk will I get? When will it make me sick? |

### 4.3 Fibre, hide and wood (20)

| event | chain | decision facts |
|---|---|---|
| X1 bark tanning | flay and flesh [L10] → lime 1–2 weeks: swelling, hair loosens [L3, L4, L9] → leach bark [L10, L3] → tannin diffuses t~L²/D, binds collagen, Ts rises 60→75–85 °C over 6–15 months [L3, L4, L7, L11] → stuff and work while drying [L3, L7] | Is it leather yet (mid-node tannin)? Weeks left? Supple or a board? |
| X2 salt a hide, or it rots | bacteria race by temperature and aw [L9] → proteases make the hair slip, rot gas [L4, L12] → salt drives aw toward 0.75 [L3, L14] → or dry it, or freeze it [L3, L1] | Will the fur hold? How long can I wait? Does it stink? |
| X3 rawhide | dry on a frame [L3] → fibres bond and stiffen [L7] → a wet strip shrinks and clamps a haft [L15, L10] → rewetting loosens it [L3] → boiled to eat [L4, L8] | Will the axe head stay on? Is the binding wet? Can I eat it? |
| X4 cuir bouilli; ruined boots | wet leather [L3] → past the wet Ts it shrinks and densifies [L4, L7, L15] → press and dry it hard [L5, L3] → dry leather tolerates more heat [L7] | Hard enough to turn a blow? Will drying the boots by the fire ruin them? |
| X5 ret flax | grow [L8] → pectin-eating microbes: ≈6 days in water, 3–6 weeks as dew [L9, L4] → stink [L12] → break and scutch: brittle core shatters, fibre survives [L5, L10] → hackle [L10, L7] | Retted enough? Will one more day ruin it? |
| X6 spin, rope, knots | twist turns friction into cohesion [L7, L5] → a knot keeps 40–75% [L5] → chafe [L10, L1] → burns from ≈260 °C [Bn] → rots damp [L9] | Will it hold me with this knot? How long to chafe or burn through? Is it rotten? |
| X7 weave and use cloth | weave sets density and cover [L7] → bleach in the sun [L4, L12] → tears along a thread [L5] → a blanket smothers a fire [Bn] → linen burns, wool chars [Bn] | Will the sail hold? Will the blanket smother the fire? |
| X8 dye | scour [L3, L4] → alum mordant [L3, L4] → dye binds [L3, L4] → indigo vat: reduced, then oxidised in air [L9, L4] → wash fastness [L3] → colour read under light [L12] | Will the colour run in rain and give the disguise away? |
| X9 felt / shrink wool | warm alkaline water opens the scales [L3, L4] → directional friction plus work tangles the fibres [L5, L7] → irreversible shrink 20–50% (u) [L15] | Did washing ruin the garment? |
| X10 sew | pierce [L10] → seam strength = min(thread, tear-out) [L5] → wax and swelling seal it [L3, L15] | Will the seam hold? Is the boot watertight? |
| X11 wet clothes chill | water wicks in [L3] → pore conductivity ×23 [L1] → evaporation cools [L1, L3, Fx] → shiver, core falls [L8, L17] → dry by the fire, or scorch [Bn] | How long until hypothermia? Strip or keep it on? |
| X12 season timber | free water leaves first [L3, L11] → below the fibre saturation point it shrinks 8% / 4% / 0.1% [L15] → surface checks [L5] → strength rises as it dries [L5] → a swollen door, a tight cask [L15] | Is the timber ready? Why is the door stuck? Will the cask leak? |
| X13 carpentry | split along the grain (3 against 60 MPa) [L5] → hew and saw [L10] → joints, pegs, nails [L10] → load check [L5] → seasons loosen it [L15] | Will the ladder or bridge hold? How long with these tools? |
| X14 hide glue | hydrolyse collagen [L4] → gels at 35–38 °C [L7] → apply at 60 °C, clamp [L5] → cures by drying [L3, L7] → damp and heat re-soften it, mould [L7, L9] | Is the glue set? Will it hold in rain? |
| X15 charcoal mound | stack and cover; cover porosity sets the air [L5, L2b] → starved burn plus pyrolysis; yield 42% at 400 °C, 33% at 500 °C [L4, Bn] → read the smoke colour [L12] → about a week to cool; opening it hot re-lights it [L1, Bn] | Is it done? Is it safe to open? How much charcoal? |
| X16 tar and pitch | heat resinous wood at 300–450 °C closed [L4] → vapours condense and drain [L1, L2a] → boil down to pitch [L1, L14] → caulk and coat [L3, L10, L9] → hot pitch burns skin [L8] | Will the boat leak? Will the torch burn in rain? |
| X17 friction fire | μNv into the contact [L5, L1] → wear makes dust [L10] → water pins it at 100 °C [L3, L1] → char glows at ≈370 °C [L4, Bn] → blow it into tinder [L2b, Bn] → costs body energy [L8] | Is the wood dry? Am I strong enough? How long will it take? |
| X18 paper | rot and beat the rags [L9, L10] → form on a mould [L2a] → fibres bond as they dry [L7, L3] → size it [L3, L7] → wet paper is weak; ignites near 230 °C [Bn] | Will the letter survive rain or fire? How long to burn the ledger? |
| X19 ink, seal, tamper | gall plus iron ink [L3, L4] → write [L3, L13, L19] → darkens in air [L4] → seal wax softens at 60–75 °C [L7, L6] → a warm blade lifts the seal; the reseal blurs it [L1, L13] → rain makes ink bleed [L3] → forgery check [L20] | Opened? Forged? Can I read it? |
| X20 rot in timber and rope | wet above 20–30% moisture [L3] → fungi eat strength faster than mass [L9, L4] → fails under load [L5] → soft to a knife, musty [L12, L10] | Will the floor, bridge or rope hold me? Probe it first? |

### 4.4 Growing (20)

| event | chain | decision facts |
|---|---|---|
| G1 clear land | fell: notch, hinge, topple [L10, L5] → grub the stump [L10, L5, L5g] → brush dries [L3] → burn it [Bn] → ash leaches potash, pH rises [L3, L4] | Which way will it fall? Is the brush dry? Will the fire spread? |
| G2 till; frozen ground | blade against penetration resistance [L5, L6] → bulk density 1.6 → 1.2 [L5g] → spoil slumps [L5g] → frozen ground needs fire or a pick [L1, L5, Fx] → wet clay smears [L6] | Can I dig today? How deep in an hour? Ready to sow? |
| G3 seed sort and sprout | brine at specific gravity 1.13 [L14] → full seeds sink [L2a] → soak to 25–30% [L3] → respiration, the root emerges [L8] → too hot, or no O2: it rots [L8, L9] | Which seeds will grow? Has the bucket gone sour? |
| G4 paddy and irrigation | cut terraces [L5g] → puddling cuts conductivity [L6, L3] → channel water [L2a] → evapotranspiration 4–7 mm/day, seepage [L2a, L3, L1] → a storm breaches a bund [L2a, L5] | Will the paddy hold water? How many buckets? Will the bunds hold? |
| G5 growing season | transplant shock [L8, L3] → degree-days above the base [L8, L11] → canopy light capture × RUE [L8, L12] → water between −33 and −1500 kPa [L3] → harvest index 0.3–0.5 [L8] | When will it be ripe? Is it thirsty? How much yield? |
| G6 weeds and pests | seed bank [L9] → crop and weeds split light, water and N [L8, L3] → weeding [L10] → pests develop by degree-days [L9] → a cat eats rat density [L9] | Is weeding worth it? Do I need a cat? |
| G7 frost and drought | clear-night canopy 2–4 K below air [L1, Fx] → tender tissue freezes at −1 to −3 °C; hardened wheat survives −15 °C [L14, L8] → cover, smudge or flood [L1] → drought closes stomata [L3, L8, L17] | Will tonight's frost kill them? Cover, smoke or flood? |
| G8 harvest, thresh, winnow | sickle [L10] → dry in shocks to 14–16% [L3, Fx] → rain sprouts the ears [L8] → threshing detaches the weak-bonded share [L5, L10] → winnowing splits by terminal velocity [L2c] | Harvest now or wait? Dry enough to bag? |
| G9 grain storage | equilibrates to 13–14% at 65–70% RH [L3, L14] → wet grain respires and self-heats [L8, L9, L1] → mould above aw 0.70–0.75 [L9] → weevils, rats [L9] → days of food = energy / need [L8, L11] | Is it keeping? How much can I sell and still last? |
| G10 mill rice | dehull [L10] → winnow [L2c] → polish off the bran, and its thiamine [L10] → brown rice goes rancid [L4] → beriberi on polished rice [L8] | Brown or white? Why are the villagers sick? |
| G11 hay and compost | field-dry to under 20% [L3, Fx] → a wet stack respires [L8, L9] → traps heat, 55–65 °C [L1] → oxidation above 70 °C, smoulder, flames when opened [L4, Bn] → compost kills pathogens [L9] | Dry enough to stack? Is the stack heating? |
| G12 hens and eggs | feed to stores [L8] → laying needs 14–16 h of light [L8, L12] → an egg loses water; an old egg floats [L3, L2a] → a hen incubates at 37.5 °C for 21 days [L1, L8] | Is this egg fresh? Will they hatch? Keep her or eat her? |
| G13 slaughter and preserve | bleed [L8] → chill below 7 °C [L1] → rigor [L8, L7] → butcher [L10] → salt diffuses, osmosis [L3, L14] → smoke [L4] → jars or the ice pit [L9] | Is the meat good? Has the fish taken the salt? |
| G14 ice house | lake ice grows by Stefan [L1, L11] → cut blocks [L10] → pack in sawdust [L1] → a summer leak of ≈28 W per m³ lasts ≈4 months [L1] → cold slows rot [L9] | Will the ice last until summer? |
| G15 maple syrup | sap runs on freeze-thaw days [L8, Fx] → tap [L10] → rule of 86; ≈12 kg wood per litre [L1, Bn] → done at +4 °C above water's boiling point [L14] → colour darkens, sugar sand [L4, L7] | Is the sap running? Is one tree worth the fuel? Is it done? |
| G16 dairy | let-down [L8] → cream rises [L2c] → sours in 12–24 h warm [L9] → curdles at pH 4.6 [L7, L14] → butter, cheese [L7, L3, L9] | Still sweet? Will it curdle? |
| G17 bees | colony holds 34–35 °C [L8, L9] → nectar fanned below 18.6% water [L3, L14] → smoke masks the alarm pheromone [L12] → stings dose venom [L17] → winter needs 20–30 kg [L8] | Is it safe to open? How much honey can I take? |
| G18 fishing | fish activity by water temperature [L8] → bait odour plume [L12, L2a] → hook pierces [L10] → line tension against break load [L5] → spoils fast [L9] → pond stock [L9] | Where to fish? Will the line hold? Is the ice safe? |
| G19 tracks, traps, predators | a foot deforms snow or mud into a mark [L6, L13] → scent fades [L12] → age read from edge sharpness [L12] → snare, deadfall or pit as stored energy plus a catch [L16, L5] → a predator tests the coop for gaps [L9] | How old are the tracks? Is the coop fox-proof? Where to set the trap? |
| G20 forestry and firewood | split along the grain [L10, L5] → air-dry about a year per inch [L3] → shrinks and checks [L15] → green wood burns smoky [Bn] → rot [L9] → coppice regrowth [L8] | Dry enough to burn? Will this beam crack? |

### 4.5 Body and medicine (18)

| event | chain | decision facts |
|---|---|---|
| B1 stop a bleed | the cut crosses vessels by depth [L5, L10] → flow ∝ (blood pressure − applied pressure) [L8] → clotting rate falls with cold [L4] → hand pressure 40 kPa, wrap hoop pressure, tourniquet with an ischaemia clock [L5, L11] → tie off a vessel [L10] | How fast is it bleeding? Is the tourniquet tight enough? How long has it been on? |
| B2 cautery | heat the iron, read its glow [L1, L12] → contact conducts heat [L1] → Henriques denaturation coagulates to the vessel's depth [L4, L8] → withdrawal reflex or holding down [R, L5] → dead tissue raises infection risk [L9] | Is the iron hot enough? Will it stop the bleeding? |
| B3 wound care, healing | irrigate; spirits at 60–90% kill [L2a, L9] → boil the needle [L9] → stitch [L10] → honey or salt dressing by aw [L3, L9] → strength 3% at 1 week, 80% at 3 months [L8, L11] → reopens under load [L5] | Is it clean? Will it reopen if he fights today? |
| B4 infection | inoculum from soil, blade or bite [L9] → growth against immune kill ∝ perfusion [L9, L8] → established above 1e5 CFU/g [L9] → fever setpoint rises [L17] → tetanus, gangrene, sepsis [L17, L8] → debride, drain, amputate [L10] | Is it festering? How long until it kills him? |
| B5 amputation | tourniquet → cut [L10] → saw bone (130 MPa) [L10, L5] → tie vessels [L10] → shock [L8, L17] → the body's mass and balance change [L5] → a peg leg as a support [L10, L5] | Can he walk, climb, keep his feet on deck? Will he survive the shock? |
| B6 fracture and splint | bending moment beats 241–408 N·m [L5] → muscle pull angles the fragments [L17] → a broken bone bears no load [L5] → reduce with traction [L5, L12] → splint [L10, L5] → healing gated by gap strain (2% / 10%) [L8, L11] | Is it broken? Can he walk with a splint? When will it bear weight? |
| B7 herbal remedy | plant composition [L8] → dry [L3] → grind [L10] → decoction or tincture: D/size², T [L3, L4, L14] → dose = volume × concentration → PK [L8] → Hill effect; opiates depress breathing [L17] | How strong? How much to give? Will it stop his breathing? |
| B8 poisoning | dissolve [L14] → taste or smell thresholds [L12] → absorb, distribute [L8] → effect by target [L17] → vomit or charcoal within 1 h [L3] → antidote shifts EC50 [L17] | Is this cup safe? Poisoned or ill? Will purging help? |
| B9 venom | fang punctures [L5, L10] → depot → lymph flow rises with movement [L8] → necrosis [L4] → paralysis or failed clotting [L17] → puncture marks [L13] | Keep still or run? Bind it? Sting or bite? |
| B10 drunk | ethanol fraction × volume [L2a] → gut absorption [L8] → zero-order elimination 0.015 %/h (Michaelis-Menten) [L4] → effects by BAC [L17] → loses heat faster [L1] → aspiration risk [L8] → tolerance [L22] | How drunk is he? Can he still fight? |
| B11 hunger and thirst | food energy [L8] → glycogen gone in 12–24 h [L8] → spend 1400–2400 kcal/day [L8, L1] → water loss; death at 15–20% [L8, L3] → starvation, scurvy [L8, L17] → food poisoning [L9] | Can I eat this? How many days will rations last? |
| B12 cold exposure | heat loss by wind, wet and ground [L1, Fx] → wet clothes lose insulation [L3] → shivering, vessels narrow [L17] → cold hands lose grip [L17] → frostbite at −0.5 to −2 °C [L1, L8] → core falls ≈0.17 °C/h wet at 5 °C [L8] | How long can I stay out? Will I lose fingers? |
| B13 heat exposure | metabolism plus sun [L8, L1] → sweat limited by humidity [L3, L2b] → dehydration [L8] → collapse above 39.2 °C, stroke above 40.5 °C [L17] → cold-water cooling [L1] | March at noon? Is he collapsing? |
| B14 drowning and rescue | marginal buoyancy [L2a] → cold-shock gasp [R] → silent drowning response → O2 store runs down [L8] → death as a process, longer when cold [L8, L1] → reach with a pole or rope [L5, L10] → CPR [L8] | Can I reach him? How long has he been under? Is reviving him worth trying? |
| B15 sleep and fatigue | Borbely S and C [L8, L11] → falls asleep in quiet, dark and warm [L12, L1] → sleep loss slows reflexes [L17] → fatigue endurance curves [L8] → woken by sound, then inertia [L12, L17] | Who is fit to stand watch? Will this noise wake him? |
| B16 contagion | shedding: Wells-Riley quanta, fomites, water, vectors [L9, L2b, L2a, L5] → die-off outside a host [L9] → dose, incubation [L9, L8] → fever, cough [L17] → SEIR offscreen [L11] | Is it catching? Is the well safe? Burn the bedding? |
| B17 bloodletting | band raises the vein [L5] → lancet [L10] → venous flow into a basin [L8, L2a] → pressure falls [L8] → fainting reflex → plasma refills in a day, red cells in weeks [L8] | Will bleeding help or kill him? |
| B18 corpse and forensics | death as a process [L8] → Henssge cooling [L1] → lividity, rigor [L2a, L4] → decay, bloat; drowned bodies rise [L9, L2a] → marks read under light [L13, L12] → natron, salt, cold [L3, L9] → pyre [Bn] | How long has he been dead? Drowned or stung? Will the body be found? |

### 4.6 Fire, light and household (18)

| event | chain | decision facts |
|---|---|---|
| H1 friction fire | as X17 [L5, L1, L10, L3, L4, Bn, L2b, L8] | Is my wood dry? How many tries can I manage? |
| H2 flint and steel | char cloth by closed pyrolysis [L4] → hard steel sheds a burning curl; soft iron smears [L10, L7] → ≈27 mJ spark [L4, L1] → a thin spot on char cloth glows [Bn] → blow the tinder into flame [L2b, Bn] | Does my striker spark? Is the char cloth dry? |
| H3 lay a fire | stack [L5] → tinder ignites in ≈1.5 s (thin) [Bn] → kindling in ≈24 s (thick) [L1] → damp wood steams and smokes [L3] → crib feedback [L1, L2b] → coal bed [Bn] | Will it catch? Is it smoking from damp? |
| H4 bank the fire | cover the coals with ash [L5] → O2 diffusion through the ash limits the glow, ≈150 g charcoal per night (u) [L3, Bn] → ash insulates [L1] → alive by morning or not [L11] | Will it last till morning? Must I borrow fire? |
| H5 carry fire | an ember pot smoulders through vent holes [Bn, L2b] → a torch burns fuel held in its wrap [L3, Bn] → walking speed fans or kills it [L2b] → drips light thatch [L2c] → the light marks the carrier [L12] | Will it last the walk? Who can see me? |
| H6 candles | render tallow [L1, L10] → dip; layers freeze on [L1, L13] → melt pool [L1] → wax climbs the wick [L3] → ≈80 W, ≈1 cd [Bn, L12] → wick trimming, guttering, a liquid layer runs down [L4, L2b, L13] → as a clock [L11] | How long will the light last? Will the draught blow it out? |
| H7 oil lamps | fill [L2a] → wick rise [L3] → 50–100 W [Bn, L12] → the glass gets hot or cracks [L15] → a spill becomes a pool fire; the oil type sets the danger [L2a, Bn] | Full enough for the watch? Kerosene or olive oil? |
| H8 light and darkness | lumens → E = I/d² → reading needs lux; a face is identified at ≈0.8 lux at 4 m [L12] → light leaks under doors → beacon range [L12, L11] → dark adaptation | Will they see me? Can I read the letter here? |
| H9 heat a house | radiation from the hearth [L1] → stack draught ≈3.8 Pa/m [L2b] → a cold flue smokes the room [L2b] → masonry stores heat [L1] → walls lose to T_out [Fx] → a jug bursts on a cold night [L15] | Warm enough to sleep? Why is the room smoky? |
| H10 chimney fire | tar condenses below ≈121 °C [L13, L1] → ignites [Bn] → roars above 1000 °C, throws sparks [L2c] → the flue cracks, the joists catch [L15, L5] → sweeping removes the deposit [L10] | Is the chimney roaring? Is the roof catching? |
| H11 CO brazier | glowing charcoal under low O2 makes CO [L4, Bn] → odourless CO fills the room [L2b] → COHb, drowsiness [L8] → only symptoms are perceived [L12] | Why am I drowsy? Open the window? |
| H12 self-heating rags or hay | drying oil oxidises [L4] → the pile insulates [L1] → runaway above the critical size (emerges) [L4, L1] → smoulders, then flames [Bn] | Safe to leave the rags in a heap? |
| H13 fight a house fire | notice [L12] → fetch and throw water [L5, L2a, L8] → ≤26 MJ per bucket; film boiling [L1] → steam scalds [L2b] → smother below ≈15% O2 [Bn] → pull down thatch [L10, L5] → water on oil causes a slopover | Is the water enough, or run? Which house do we pull down? |
| H14 wells, carrying water | groundwater fills to the water table [L3] → windlass [L16, L5] → carry cost [L8] → well water as a cold store [L1] → a privy fouls it [L3, L9] → falling in [L5, L8] | Is the water safe? How much can I carry? |
| H15 ice harvest | Stefan growth [L1, L11] → Gold's formula for bearing [L5] → saw and haul [L10, L2a] → pack in sawdust [L1] → wet sawdust insulates worse [L3] | Thick enough to cross? Ice left for the feast? |
| H16 soap | leach lye from ash [L3] → strength by density (egg floats) [L2a] → causticise [L4] → saponification 1–2 h hot [L4] → salt out [L4, L2a] → cure 4–6 weeks [L3] → lye burns [L4, L8] | Is the lye strong enough? Is it safe on skin? |
| H17 laundry | soak [L3] → lye and soap [L4] → boil [L1] → beat [L10] → wring [L5] → line-dry by Penman, 2 h to 3 days [L3, Fx] → sun bleaching [L4, L12] → a sad iron scorches above ≈200 °C (u) [L1, L4] | Is it dry yet? Is the stain gone? |
| H18 stains, hiding a crime | blood soaks into pores and dries [L3, L13] → hot water sets it, cold and salt dissolve it [L4] → scrubbing lifts the surface but not the pores [L10, L3] → planing removes it [L10] → smell persists [L9, L12] → seen in good light [L12] | Clean enough to fool a visitor? Blood between the boards? |

### 4.7 Building, moving and machines (18)

| event | chain | decision facts |
|---|---|---|
| K1 dig a pit or tunnel | pick loosens [L10, L5] → shovel; bulks ×1.2–1.3 [L5, L8] → spoil slumps [L5g] → wall stands to Hc = 4c/γ; rain caves it [L5g, L3] → seepage [L3] → CO2 pools [L2b] → 0.25–0.75 m³/h [L11] → fresh ground shows [L13, L12] | How long? Will the side cave? Is there air at the bottom? |
| K2 build a wall | lay with friction [L5] → lime carbonates ≈1 mm/√day [L4, L3, L11] → thrust line and crushing [L5] → ram damage [L5] → frost and washout [L15, L3] → a palisade burns [Bn] | Will it stop them? Is the mortar still green? |
| K3 roof and thatch | rain [Fx] → a steep roof sheds along the stems [L2a, L3] → rot [L9] → dry straw is a fine fuel [Bn] → snow load fails the rafters [L5] | Will it keep us dry? Catch from an ember? Hold this snow? |
| K4 a well's bad air | dig and line [L10, L5g] → seepage [L3] → blackdamp pools [L2b] → a candle dies below 15–16% O2 [Bn, L12] → collapse in minutes [L8] → windlass [L16] → fouling [L9] | Is there air down there? Go down after him? |
| K5 bridge | load against beam strength; rope tension wL²/8s [L5] → creaks [L12] → wet or icy planks slip [L5, L3] → cut strands [L10] → char weakens it [Bn] → hidden rot [L9] | Will it hold me? Will it burn before they cross? |
| K6 doors | hinge [L16] → latch, bar, a wedge that locks itself [L16, L5] → barricade friction [L5] → a kick breaks the jamb or the strike first [L5] → axe [L10] → fire burns through in 40–60 min [Bn] → smoke and sound under it [L2b, L12] → swells and sticks [L15] | Will the bar hold? How many kicks? Can they hear us? |
| K7 locks | key profile matches the wards [L16] → picking: binding order is felt [L16, L12, L17] → audible [L12] → pry, smash, file [L16, L5, L10] → rust seizes it [L4] → wax impression [L13, L6] | How long with my hands shaking? Is it rusted? Can I copy the key? |
| K8 levers and tackle | mechanical advantage [L16] → fulcrum bearing [L5g] → nail withdrawal [L5] → pulley efficiency 0.9ⁿ [L16] → rope strength [L5] → the body's pull [L8] | Can I lift it alone? Is the rope strong enough? |
| K9 carts | axle [L16] → rolling resistance from sinkage in wet ground [L5g, L3] → greased axle [L5] → a horse ≈0.75 kW [L8] → a slope, brake heat, tipping [L5, L1] → a shrunk-on tyre [L15] | Can we outrun them with this load? Through the mud? |
| K10 rafts and boats | buoyancy, metacentric height [L2a, L5] → dry seams gape, then the planks swell shut [L3, L15] → caulking [L10, L7] → a hole lets in 3.7 L/s; bailing manages 0.5–1 L/s [L2a, L8] → rowing, sailing [L2a, Fx] | Will it float with all of us? Can we bail faster than it fills? |
| K11 mills | head and flow [L2a] → wheel power ηρgQh [L2a, L16] → gears [L16] → grinding [L10] → flour-dust explosion [L2c, Bn premixed] → a windmill gives ½ρAv³Cp [Fx]; brake fires [L1] | How much flour a day? Will a dropped lamp set the dust off? |
| K12 traps | stored energy plus a catch [L5, L16] → concealment [L12, L13] → trigger threshold [L16] → deadfall, snare by capstan friction, pit cover fails [L5] → trigger dries and swells [L3, L15] | Is there a trap here? How hard to trip it? |
| K13 bows | draw 450–700 N, store 160–260 J [L16 spring from L5] → hold time is fatigue [L8] → arrow ≈53 m/s [L16, L5] → crossbow spanning [L16] → wet string and glue soften; a strung bow takes a set [L3, L7, L6] | Can I draw it? Will it go through mail? Is my string wet? |
| K14 black powder | grind and corn [L10] → hygroscopic, 1.5% water in 24 h [L3] → ignites at 300–450 °C [L1] → its own oxidiser, ≈330 cm³ gas per g [L4] → fuse 100–131 s/m [L4, L12] → a keg bursts by hoop stress [L2d, L5] | Is the powder dry? How long is the fuse? How far is safe? |
| K15 flintlock | load 15–20 s → sear [L16] → spark into the pan [L4] → damp misfires [L3] → barrel pressure, ≈400–450 m/s [L2d] → report and smoke [L12] → a double charge bursts it [L2d] | Will it fire in rain? Can I reload before he reaches me? |
| K16 bombs | casing bursts by hoop stress [L2d, L5] → fragments [L5, L8] → blast by scaled distance [blast event, L12, L8] → breaching [L5] → triggers [L16] | How far is safe? Will it open the door? |
| K17 siege mining | dig [L10, L5g] → props as columns [L5] → fire the props [Bn] → collapse [L5] → countermining by ground sound [L12] → bad air [L2b, L8] | Will it bring the wall down, and when? Can they hear us? |
| K18 clockwork | wind the spring [L16] → gear train [L16] → pendulum period [L5]; drifts ≈0.5 s/day per K [L15] → runs down [L11] → alarm catch [L16] → ticking [L21] → read the dial [L19] | How long until it goes off? Which wheel do I jam? |

### 4.8 Nature and disaster (18)

| event | chain | decision facts |
|---|---|---|
| N1 lightning | strike the tallest thing within ≈90 m [Fx] → current takes the wet bark or the dry sapwood [L18, L3] → sap flashes to steam, the trunk splits [L1, L2d, L5] → ignites dry fuel [Bn] → thunder delayed 3 s/km [L21, L12] → step voltage into a person [L18, L8] | Is that tree the tallest? Is the storm close? Is CPR worth trying? |
| N2 caught in rain | rain rate [Fx] → lands on column tops [L2a] → clothes, fuel and powder absorb it [L3] → body chill [L1, L8, L17] → wet fuel and powder fail [Bn, L3] → rain masks sound [L12] → tracks wash out [L13] | Will my powder fire? Can they hear us? |
| N3 flood | catchment run-off [Fx, L11] → height field [L2a] → doorway weir flow → wading limit h·v ≈0.6 m²/s [L2a, L5] → carts float at 0.38–0.69 m [L2a] → grain soaks, mud walls soften [L3, L9, L5g] → silt [L2c] | Can I wade it? Will the mud wall hold? Is the well safe? |
| N4 wildfire | fuels dry to equilibrium moisture [L3, Fx] → spread by contact, radiation and wind tilt [Bn, L1, L2b] → embers start spot fires [L2c] → radiant heat at a person [L1, L8] → smoke [L2b, L12] | Which way is it running? Is the burnt ground safe? |
| N5 radiation frost | sky 20–30 K below air [Fx, L1] → frost forms at the dew point [L1, L2b] → a jar bursts [L15] → blossom kill at −2 to −4 °C [L8] → cover, smudge, sprinkle [L1] | Will it frost tonight? Cover the seedlings? |
| N6 thin ice | h ~ √(freezing degree-days) [L1, L11] → Gold P = A·h² [L5] → falls in [L5] → cold shock 1-10-1 [L17] → ice edge breaks under the elbows [L5] | Is it thick enough? How long can he last in there? |
| N7 blizzard | snow 50–150 kg/m³, settles [Fx, L7] → drifts [L2c] → whiteout [L12] → roof load 2.5 kPa/m [L5] → walking cost, footprints fill in [L8, L13] → snow cave [L1] | Can we follow the tracks? Will the roof hold? |
| N8 avalanche | facets grow under a temperature gradient [L7] → 33–45° slopes [L5g] → a person triggers the slab [L5g] → granular flow [L5] → debris sinters [L7] → burial clock [L8] → probe and dig [L12] | Is this slope loaded? Where is he buried? How long do we have? |
| N9 landslide | infiltration raises pore pressure [L3, Fx] → FS below 1 (Caine threshold) [L5g] → precursors: cracks, springs, leaning trees [L12] → runout [L5g, L2a] → burial [L5] | Leave the house tonight? Is the road passable? |
| N10 earthquake | ground acceleration [Fx] → things tip when a/g > width/height [L5] → walls fail out of plane [L5] → crush [L8] → spilled lamps start fires [Bn] → dust [L2c] → aftershocks by Omori [L11] | Run or shelter? Is someone alive under there? |
| N11 gale | q = ½ρv² [Fx, L2b] → trees snap or uproot [L5] → roofs lift [L5] → debris projectiles [L5] → people lose balance near 25 m/s [L5, R] → wind chill [L1] | Will that tree hit the house? Can I stand on the roof? |
| N12 hail | stones at terminal speed: 2.5 cm ≈1.4 J [Fx, L5] → tiles crack at ≈1.6 J [L5] → crops shredded [L8] → injury [L8] → melt, rot [L1, L9] | Get under cover? How much crop is lost? |
| N13 drought | ET0 3–7 mm/day [Fx, L3] → soil water to the wilting point; stomata close [L3, L17] → yield response [L8] → wells drop [L2a] → stock die [L8, L9] | How long does the water last? Which field to save? |
| N14 day, night, seasons | sun position [Fx] → sky light [L12] → solar heating, night cooling [L1] → T_out curve [Fx] → degree-days, photoperiod, leaf fall [L8] → circadian sleep [L8] | How long until dark? How cold tonight? Time to plant? |
| N15 rot and rust over years | wet wood above 20–30% [L3] → brown rot [L9] → ISO rust D = r·tᵇ [L4, L11] → section left decides failure [L5] → soft to a knife, flaking [L12, L13] | Will the bridge, floor or chain hold? |
| N16 animal populations | logistic growth, predation [L9] → rats breed where grain is accessible [L9] → snow drives wolves to the village [L9, Fx] → droppings, gnawing [L13] → spawn individuals near people [L9 seam] | Rats in the grain? Will wolves come tonight? |
| N17 tide | harmonics, spring and neap tides [Fx] → shore height field [L2a] → wading against a current [L2a, L5] → the boat floats off [L2a] | How long until the causeway floods? |
| N18 fog | cooling to the dew point [Fx, L1] → droplets [L2b] → visibility = 3.912/σ [L12] → sound carries [L12] → lifts after dawn [L1] | Can I see the shore? Call out or wait? |

### 4.9 Information, money and deception (19)

| event | chain | decision facts |
|---|---|---|
| I1 pay money | coins from a purse [L5] → count and identify by stamp and light [L12, L13, L20] → hand to hand; a dropped coin rings [L5, L21] → witnesses [L12] → the deal becomes claims [L19] | Is it enough? Who saw? Is this coin good? |
| I2 test a coin | look [L12, L20] → bite: gold is Mohs 2.5–3 [L6] → ring against thud [L21] → weigh: gilt lead ≈4.9 g against 8.4 g [L5, L12] → touchstone plus nitric acid [L13, L4] → a clipped edge [L10] | Real and full weight? Will testing insult him? |
| I3 write a letter | quill, ink, light [L10, L12] → strokes wick into fibres [L3, L13, L19] → dries 10 s–2 min; smudges [L3] → iron gall darkens in hours [L4] → fold and seal | Who can read this? Is it dry? What does it give away if caught? |
| I4 seal and reseal | wax softens ≈75–80 °C (u) [L1, L7] → signet relief [L6, L13] → sets in 20–60 s → breaking it is evidence [L5] → a warm blade lifts it; remelting blurs it [L1, L10, L13] → inspection [L20] | Has it been opened? Whose seal? |
| I5 forge a document | learn the exemplar [L12] → cast the seal [L7, L6] → imitate the hand with error [L19] → age the ink [L4, L3] → the reader compares features [L20] → believed with the claimed source [L19] | Will it pass this reader? Is the ink too fresh? |
| I6 read by candle; hidden ink | 1 cd gives ≈11 lux at 30 cm [L12] → legibility by size and contrast [L12] → decode [L19] → lemon ink browns before the paper does [L4, L1] | Can I read it here? Is there more (heat it)? |
| I7 destroy a document | paper ignites in ≈0.5 s [Bn] → tear: the fragments keep their content [L10] → soak: carbon ink runs [L3] → smoke smell [L2b, L12] | Is it gone for good? Will the smoke give me away? |
| I8 maps | survey with bearings [L19] → read into beliefs [L12] → compass [Fx, L18] → telescope [L12] → false or old map entries [L19] | Where is X? How far do I trust it? |
| I9 signs, paint, chalk | paint soaks in [L3, L13] → linseed cures in 1–3 days [L4, L7] → legible to ≈300–500× the letter height (u) [L12] → chalk wipes off [L10] → old paint shows through [L13] → fades [L4] | What does the sign say from here? Will my partner see the chalk mark but not the watch? |
| I10 disguise | spirit gum, dye, putty [L3, L10, L6, L4] → sweat and rain loosen it [L3, L1] → seen with blur ≡ distance; ≈0 recognition at 45 m [L12] → feature match per channel [L20] | Will he know me at this distance? Is the disguise holding? |
| I11 hide and search | out of sight lines [L12] → dig or lift a flag; traces left [L5g, L6, L13] → oilskin keeps it dry [L3] → traces fade [L3] → search coverage and probability [L12] → a hollow knock [L21] | Can anyone see me hide it? Does the spot look disturbed? |
| I12 lockbox and key | key matches [L16] → cut the cord [L10] → pick [L16] → force: loud and visible [L5, L12] → wax copy [L6, L13] | Do I have the key? Pick quietly or break it? |
| I13 poison in a cup | arsenic dissolves ≈20 g/L [L14, L2a] → taste thresholds [L12] → who touched the cup [L12] → dose [L8] → onset 30 min to hours [L8] → effect, antidote [L17] | Who poured it? How much did I drink? When will it act? |
| I14 drugged hood | volatile soaks the cloth, evaporates [L3, L1] → builds up in the headspace [L2b] → minutes, not seconds, of breathing [L8] → effect, recovery [L17] | How long must I hold him? Is he faking? |
| I15 contact poison | residue transfers on touch [L13, L3] → skin uptake in 10–30 min (u) [L3, L8] → rash or systemic effect [L8] → spreads to the face or food [L13] → wash [L3] | Did I touch it? How long before I must wash? |
| I16 smoke signal | damp leaves give white smoke [Bn, L4] → the plume bends in wind [L2b, Fx] → puffs with a blanket [Bn] → contrast and horizon [L12] → decode [L19] → enemies see it too | Will they see it? Who else will? |
| I17 sound signals | source level: whistle 111–143 dB [L12, L21] → propagation, absorption [L12] → above ambient [L12] → pattern decoded [L19] | Will they hear it? Will the enemy? |
| I18 visual signals | flag, lantern [L13, Bn] → line of sight, 1/d², blinds block [L12] → resolution [L12] → decode [L19] → a lit window gives you away [L12] | Can they see the light from the road? |
| I19 pickpocket, palm | distract [L12 attention] → reach or cut [L10, L5] → touch and weight thresholds [L12] → palmed: hidden by the hand [L12] → pass off [L5] | Is he looking? Will he feel it? |

### Which laws carry the most events (out of 169)

| law | events | law | events | law | events |
|---|---|---|---|---|---|
| L11 slow time | 130 | L9 populations | 59 | L17 tissue function | 23 |
| L1 heat | 116 | L2b gas | 56 | L6 rheology | 22 |
| L3 transport | 116 | L2a liquids | 56 | L5g granular | 15 |
| L5 mechanics | 102 | Fx inputs | 41 | L2c dispersed | 14 |
| L8 living | 96 | L7 structure | 40 | L16 mechanisms | 14 |
| L4 chemistry | 90 | L13 layers | 40 | L19 codes | 13 |
| L12 senses | 82 | L14 equilibria | 36 | L2d chambers | 7 |
| L10 sep/join | 81 | L15 free strain | 27 | L20 recognition | 6 |
| Bn burning | 60 | | | L18 electric | 1 |

L16, L19 and L20 have few events but carry the heaviest book themes: doors (376 cases), money (582–916 hits), locks (129), documents (227) and disguise (201). The median event uses 8 laws.

---

## 5. The attack: 25 new events against the law set

We tried 25 events not used to build the set:
- **9 were covered** with no change.
- **10 were covered with a gap.** Each needed one small amendment.
- **6 were not coverable.** Five needed a new law or a large extension. One is not a world law at all.

Every fix below is already folded into §2.

### Not coverable before the fix

| event | why it failed | fix, now in the law set |
|---|---|---|
| Tune a lute (also: ring a bell, knock a wall, thump a melon, a cracked pot sounds dull) | Sound sources were "data", so pitch would be a case table. | **New L21:** vibration and sound emission. Pitch is the lowest mode of the struck record; decay rises with damage. |
| Light a fire with a burning glass (also: spear a fish, signal with a mirror, a camera obscura) | Light on 20 cm cells cannot focus onto a 1 mm spot. | **L12 amendment:** trace light from small sources through mirror and lens records. Note: Piggy's glasses were for short sight; a concave lens cannot light a fire. The laws should refuse the book here. |
| Magnetise a needle for a compass | No law has magnetism. | **L18 widened** to electromagnetism: persistent magnetisation, Curie loss, Earth field from Fx. |
| Flour-mill dust explosion (also firedamp, a gas-filled room, still vapour) | Burning had no premixed mode and no pressure rise in a room. | **Bn premixed mode** (flammability limits, minimum ignition energy, flame speed), plus **the room as an L2d chamber**. |
| Build tolerance to a poison (also training, calluses, acclimatisation, getting better at a craft) | No law changes a capacity from its own history. | **New L22:** adaptation by use. It is the physical base of LitRPG levelling. |
| Grain prices climb in a hard winter | The physics supplies the scarcity, but a price is what minds agree to. | **Not a law.** It needs an offscreen summary of many minds: market clearing and rumour spread, with the same spawn/absorb seam as L9. |

### Covered with a gap (the amendments)

| event | gap | amendment |
|---|---|---|
| Drop a stone down a well and count; flash to bang | No arrival delay; no splash sound | L12 delay of distance / 343 m/s; L21 |
| Sail upwind (also kites, oars, rudders, a draught slams a door) | Scalar drag gives no lift | Force on oriented faces: normal pressure C_N(α) plus skin friction (L2a, L2b) |
| Siphon, breathe through a reed, drink through a straw | Tubes are far below a cell | CONDUIT record class |
| Loaded dice, coin flip, thrown knife, spinning top | Small things had no rotation | Newton-Euler motion for small records (L5) |
| Solar still (also dew, rain sheeting, wax running, blood trickling) | Layers could not flow | Liquid layers drain and drip (L13, L2a) |
| Sky lantern, hot-air balloon, a bloated body | Buoyancy only in liquids | Buoyancy in gas, counting the chamber's own gas |
| Read the sky for a storm | A seeded weather state cannot be forecast without reading the random generator | Weather as fronts with leading signs (Fx) |
| A book stops a bullet; arrows against mail | L10 cuts by slow strokes only | Fast penetration layer by layer, with the projectile deforming (L10) |
| Clean silver with foil; iron nails in copper sheathing | Reactions were local to one spot | Electrochemical pair rows (L4, with L14 potentials) |
| Printing press, seal impressions, mirror writing | Contact transfer did not copy shape | Mirrored relief transfer; L19 form from geometry |

### Covered with no change
- **Salt an icy road.** Liquidus from L14; melting draws heat, so the road gets colder; nothing melts below −21 °C.
- **Embalm with natron.**
- **Mixing cleaners makes a toxic gas.**
- **A locked room held by an ice block.** The ice melts, the latch drops, and a damp patch is left behind.
- ***To Build a Fire.*** Snow from the boughs quenches the fire, and cold fingers cannot strike a match.
- **Romeo's feigned death.** It falls out of L17 plus touch thresholds in candlelight.
- **Floating an egg to test freshness.**
- **Holmes reading mud on boots.** This works through the L13 layer order.
- **John Snow and the Broad Street pump.**

Two small notes came out of these: L13 layers must carry mass, composition and adhesion, and L9 densities must ride L3 and L2a flows.

### One strict catch from your own examples
**"Temper a spring."** If L16 spring stiffness were a declared row, tempering could not change it, and that is a case table. So stiffness now comes from L5 (E and geometry), and the elastic limit comes from L6 and L7. A soft spring takes a set. The same applies to bows.

---

## 6. Build order, mapped to the waves

**Step 0, before any law.** Build three pieces of plumbing:
- the composition vector, with `mat` as a view;
- the record schema with the hand-over invariant;
- the surface-layer dict.

They are plumbing, not laws, and everything after depends on them.

The waves follow `docs/realism-map.md` §6, with four pull-forwards this study recommends:
1. Composition, L14 and the L11 scheduler into wave 13.
2. Pour, buoyancy of things and a person in water into wave 15; the height field stays in 16.
3. Soil (L5g) from wave 17 to 15.
4. The weather state machine (cheap data) into wave 15.

The counts come from re-running `matrix.tsv` against this plan. They are approximate (±1 law per event).

| wave | laws finished (on top of the planned lanes) | new events possible | total | crafts and events that now work end to end |
|---|---|---|---|---|
| 12 (engineering lanes) | Bn parts (flame, char depth, G7, boiling curve, SI igniters); L2b (GAS lane 7); L3 start (film, sorption); L5 bond fixes; reflex contract (L17 seed) | 0 | 0 | None end to end yet. Fire, wet wood and smoke layers become right, and that is the base. Add the COMBUST_CEIL fix here if possible. |
| 13 Ground truths | Step 0; L1 (ICE, no floor, sky and solar, steam kept); L5 one material record, μ per pair, beam check, damage; L14; L3 full (evaporation, solutes in solids); L4 rows; L11 scheduler; L16 joints and catches (doors, locks); L8 body heat and stores; L13 layers; Fx temperatures | 3 | 3 | Lay a fire; bank the fire overnight; carburize iron. Doors, latches and bars work, but their full events also need light and cutting. |
| 14 Seeing and hearing | L12 light, odour, touch, delay, optics; L19 codes; L20 recognition; L21 vibration; Fx sun and calendar | 9 | 12 | Charcoal burning; the CO brazier; light and darkness; paying money; reading by candle and hidden ink; maps; smoke, sound and flag signals. Also the lute and the burning glass. |
| 15 Time and scale | L10 cutting, Archard, joints; L9 populations; L17 tissue function; L22 adaptation; L7 structure; L15 free strain; L5g soil (pulled); L2a pour and buoyancy (pulled); Fx weather | 101 | 113 | **The big payoff.** Salt-cure, smoke, dry, pickle, jam, cheese, roast, fry, boil water safely, distil; bark tanning, rawhide, rope, cloth, dye, felt, seasoning, carpentry, glue, friction fire; clear land, grain storage, ice house, slaughter, hens; all of body and medicine; flint and steel, candles, lamps, house heating, wells, soap, laundry, stains; digging, walls, roofs, bridges, doors, carts, rafts, traps, clockwork; fire-setting, quench, temper, lime, mortar, rust; rain, thin ice, avalanche, landslide, gale, hail, rot; writing, poison, pickpocketing. Also tolerance and training (L22), and toxic gas from cleaners. |
| 16 Matter, water, weapons | L2c dispersed; L2d chambers and blast; L6 rheology; height field | 40 | 153 | **Smithing at last:** bloomery, casting, forging, forge welding, sharpening, assaying, pottery, glass. Bread, coffee, ale, wine, butter, tar. Locks fully, mills, bows, powder, flintlocks, bombs. Flood, wildfire embers, blizzard drift, earthquake. Coin testing, seals, forgery, disguise, hide-and-search. The dust explosion. |
| 17 Living and packs | L8 plant cohorts, crops in 2-D | 15 | 168 | **Farming at last:** seed sorting, the growing season, weeds and pests, frost on crops, harvest and threshing, rice milling, hay heat, maple syrup, bees, flax, foraging, forestry regrowth, drought, seasons. |
| later | L18 lightning and magnetism; offscreen mind summary; heredity | 1 | 169 | Lightning, the compass, markets and rumour, breeding. |

**Reading this table:**
- **Wave 15 is too big for one wave.** Split it in two:
  - **15a:** L10, L7, L15, L2a pour and buoyancy.
  - **15b:** L9, L17, L22, L5g, weather.
- **Smithing needs L6 (wave 16) and farming needs plant L8 (wave 17).** Both are your named examples. To prove the principle sooner, pull one of them earlier with a narrow first slice:
  - **Smithing slice:** a SECTION record, L6 for hot iron only, and L7 steel kinetics. Proof scene: quench a blade, file-test it.
  - **Farming slice:** one plot cohort with degree-days and the FAO-56 water balance. Proof scene: salt-water seed sorting, where the sinkers sprout.
- **First proof scenes per law** are events whose only missing piece is that law:
  - L9: the ice house lasts about 4 months under sawdust and about 2 weeks without it; grain at 18% moulds while grain at 13% keeps.
  - L15: frost bursts a jar.
  - L13: blood stains.
  - L16: a door bar against kicks.
  - L19: the black spot.

  Each is a no-special-code test.

---

## 7. Open questions for you

1. **Pull work forward?** Should the composition vector, L14 and the L11 scheduler move into wave 13, and soil, pour and buoyancy into wave 15? Almost every craft is gated by them.
   **Recommend: yes.** They are cheap (µs per change), and without them each craft needs new material ids.

2. **Records for thin things?** Should everything thinner than a cell (blades, hides, coins, letters, cups, seams) live as records on a host, with one invariant that the ledgers match across every hand-over?
   **Recommend: yes, one record schema for all families.** It is the same mechanism as the planned cooking slab, cup record and thin skin, so we build it once.

3. **Remove COMBUST_CEIL and fix the forge first?** This moves golden fire cases.
   **Recommend: yes,** in the wave 12 FIRE lane or at the start of wave 13, as a declared golden diff. Smithing cannot happen while 1100 °C is a hard cap.

4. **L22 and LitRPG "systems".** Should physical adaptation (training, tolerance, practice) be the base that pack systems (DCC levels, cultivation realms) build on?
   **Recommend: yes.** L22 is world physics, and a pack adds its own stores and borders as data on top, the way cultivation.yaml already does. Packs never replace it.

5. **When a book's physics is wrong** (Piggy's short-sight glasses lighting a fire, an impossible ice load), what wins?
   **Recommend: the laws win.** Record each such case as a ruling and mark it "book physics wrong" in the judge set, so a model is never trained to copy the error.

6. **Markets and rumour.** Should these be an offscreen summary of many minds (market clearing, SEIR-style spread of beliefs) rather than a world law?
   **Recommend: yes,** built after wave 17, with the same spawn/absorb seam as L9, so that a person walking into the market meets real minds again.

*Every number marked (u), and the unverified figures the families listed (steel's hot flow stress, Archard K, tannin diffusivity, flintlock misfire rates, brown-rot strength loss, electrical conductivities and others), must be sourced before it becomes a test value.*