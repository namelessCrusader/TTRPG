# Realism references, part 2: wind, gases, reactions, body clocks, slow world, contacts

Date: 2026-10-02. Status: research and comparison only. Nothing in `src/` was changed.

This file is the research pass that roadmap 107 asks for before its lists B and C are
built. It compares the voxel sim with measured real-world values in six domains: wind
and weather, gas species, contact chemistry, body clocks, the slow world, and
mechanics (contacts, turning, give, water on bodies). Section 2 has one table per
domain. Section 3 ranks the gaps. Section 4 lists the reference tests. Section 5
lists the sources. Section 6 gives the cost of each domain per tick.

The first pass is `docs/realism-references.md`. This file does not repeat it. Where a
topic overlaps, a row or a gap gives the first-pass id (for example M7, L15, G9).

## 1. How to read this

### 1.1 Which code

- Code is cited as `file:symbol`, for example `air.py:AirMixin._law_air` or
  `materials.py:LEAK`. The files are in `src/voxel/` unless the name starts with
  `mpm/`. Line numbers are not given, because the split into mixins is still moving
  them.
- The sim was read at the working tree of 2026-10-02 (after the split into `air.py`,
  `heat.py`, `fluids.py`, `support.py`, `bodies.py`, `physiology.py`, `motion.py`,
  `senses.py` and `materials.py`).
- "Roadmap 107" means entry 107 in `docs/voxel-roadmap.md`.

### 1.2 Units, verdicts, tags and the game band

- Units, the verdicts (matches, ok for game, wrong, missing, unknown), the tags
  (derived, unverified, secondary) and the game band are the same as in sections
  1.2–1.4 of the first pass. In short: times within a factor of about 1.5, thresholds
  within about 20%, or the source's own range when that is wider.
- `TICK_S` is 0.025 s (`materials.py:TICK_S`). One second is 40 ticks. One hour is
  144,000 ticks. One day is 3,456,000 ticks. A slow process cannot run per tick; see
  Section 6.
- The air field cell is `_CS` = 4 voxels on a side (`air.py:AirMixin._CS`): 20 cm at
  `voxel_cm` 5, 40 cm at 10.
- For slow processes, "ok for a game" also allows a time that is right to within a
  factor of 1.5 **on the scale the player meets it**. Bread that goes mouldy in 4 days
  instead of 6 is fine. Bread that goes mouldy in 4 minutes is wrong.

### 1.3 Ids

- Row ids per domain: W (wind), GS (gas species), R (reactions), BC (body clocks,
  the domain roadmap 107 calls B2), SW (slow world), K (contacts and water on bodies).
  GS and BC are used so that rows do not clash with the first pass's gap ids G1–G19
  and its row B2.
- New gaps continue the first pass's numbering: G20 onwards.
- Test ids: W-T, GS-T, R-T, BC-T, SW-T, K-T and KW-T (water on bodies).
- A source number such as 12 is from Section 5 of this file. A source number with the
  prefix I-, such as I-9, is from Section 5 of the first pass.

### 1.4 How the numbers were checked

Six researchers collected the values, one per domain. Each one opened every source it
cites and copied only the values it read there. A value whose source could not be
opened carries the tag (unverified). The writer then checked each sim value against
the code. Sim values marked (derived) come from reading the code and doing the
arithmetic. No sim run was done for this file.

## 2. Comparison tables

"Src" is the number of the source in Section 5. A dash means no outside source (the
value is derived or an internal invariant). Each section starts with what the sim has
now, read from the code.

### 2.1 Wind and weather (W)

**What the sim has now.** The air field (`air.py:AirMixin._law_air`) is a coarse
pressure and velocity field on cells of 4 × 4 × 4 voxels (`_CS`). It has no units:
`pcell` and `vcell` are not pascals or metres per second.

- Only fires and warm air drive it. `_law_burn` and `_law_boil` add to `p_add`, and
  every warm cell adds `warm * 0.0015` to the upward velocity.
- Every tick each value is blurred (0.8 of itself plus 0.2 of the neighbour mean) and
  then multiplied by 0.965. A velocity with no source therefore halves in ≈ 20 ticks
  (0.5 s) (derived: ln 2 / −ln 0.965 = 19.5 ticks).
- The wind carries **smoke only**. A face passes at most 0.4 of a cell's smoke per
  tick (`np.clip(... * 0.25, -0.4, 0.4)`). That caps transport at 0.4 voxel per tick:
  0.8 m/s at 5 cm and 1.6 m/s at 10 cm (derived). A 10 m/s wind would need 12 voxels
  per tick at 5 cm.
- O2 is not advected, on purpose: an earlier try pumped a room down to 2% O2 (comment
  in `_law_air`). Heat is not advected either; heat rises by `heat.py:_law_rise`
  (first pass G7).
- There is no weather. No edge of the world sets a mean wind, nothing gusts, and the
  open sky only bleeds pressure (`p[:, :, -1] *= 0.3`).
- The wind pushes no solid, no person, no flame and no loose leaf. It does not cool
  anything faster (cooling is the flat `materials.py:LEAK`, first pass G5). It does not
  change how fire spreads (`_law_burn` has no wind term).

A conversion used below: weather data give the wind at 10 m. Over open grass
(z0 = 0.03 m) the log law gives 0.60 of it at 1 m, 0.70 at 1.75 m and 0.72 at 2 m
(W1, derived). People feel the 1.75 m value.

| # | Quantity | Real value | Src | Sim value | Verdict |
|---|---|---|---|---|---|
| W1 | Wind against height | Log law u(z) = (u*/k)·ln(z/z0), k ≈ 0.41. WMO surface wind = 10 m over z0 = 0.03 m. Ratio to the 10 m wind over open grass: 0.60 at 1 m, 0.67 at 1.5 m, 0.70 at 1.75 m, 0.72 at 2 m (derived). | 1, 2 | No mean wind | missing |
| W2 | Roughness length z0 | Open sea 0.0002 m; mud flats, snow 0.005; open grass 0.03; low crops 0.10; high crops 0.25; parkland, bushes 0.5; suburb or forest 1.0; city centre ≥ 2 m (Davenport–Wieringa). | 2 | None | missing |
| W3 | Gusts against the mean | 3 s gust over the 10 min mean, at 10 m: 1.66 inland, 1.52 on a coast with offshore wind, 1.38 with onshore wind, 1.23 at sea (1.75 / 1.60 / 1.45 / 1.30 over a 1 h mean). Turbulence intensity ≈ 0.17 at 10 m over z0 = 0.03 m. | 1 | None | missing |
| W4 | How long a gust lasts | WMO reports the highest 3 s average. A 3 s gust is a "wind run" of ≈ 50–100 m in strong wind. Means are 10 min averages. | 3 | None | missing |
| W5 | Beaufort scale on land (10 m, 10 min mean) | F2 1.6–3.3 m/s: "leaves rustle". F3 3.4–5.4: "leaves and small twigs in constant motion". F4 5.5–7.9: "raises dust and loose paper; small branches are moved". F5 8.0–10.7: small trees sway. F6 10.8–13.8: large branches move, "umbrellas used with difficulty". F7 13.9–17.1: "whole trees in motion; inconvenience felt when walking against the wind". F8 17.2–20.7: "breaks twigs off trees; generally impedes progress". F9 20.8–24.4: slight structural damage. F10 24.5–28.4: "trees uprooted". | 4, 5, 6 | None | missing |
| W6 | Wind and people (at 1.75 m) | Land Beaufort at pedestrian height: 5.6–7.5 m/s "force of wind felt on body, danger of stumbling"; 7.6–9.7 "difficult to walk steadily"; 12.1–14.5 "great difficulty with balance in gusts"; 14.6–17.1 "people blown over". Gusts: 15 m/s for 2 s can unbalance people; 20 m/s is dangerous even for the young; 23 m/s blows people over. A rise from 12 to 23 m/s in 2–3 s forced some people to their hands and knees. Criteria: U + 3σ > 15 m/s (elderly) and > 20 m/s (everyone). | 7, 8 | The wind pushes nobody. `motion.py:_overbalanced` reads only the body's own weight and what it holds. | missing |
| W7 | Wind force on a standing person | Drag area C_d·A ≈ 0.84 m² facing the wind, ≈ 0.46 m² side-on, 0.19–0.28 m² crouched, 0.11 m² lying (read from a figure; which drawing is front-on is the researcher's reading). C_d 1.0–1.3. F = ½ρv²·C_d·A: 50 N at 10 m/s, 113 N at 15 m/s, 202 N at 20 m/s facing (derived). Walking at 1.25 m/s into an 18.5 m/s wind raised O2 use from 0.8 to 3.1 L/min, with the square of the wind speed (abstract). | 9, 10, 11 | `bodies.py:_drag_a` drags only bodies in flight | missing |
| W8 | Wind that puts out a candle or a torch | No opened source (unverified). Leads only: an oscillating flow of ≈ 0.5 m/s put out a 20 W candle; small firebrand flames went out above ≈ 5 m/s. | — | No candle; igniters are 100 kW scene sources (first pass F24) | missing (unverified) |
| W9 | Rothermel wind factor | φ_w = C·U^B·(β/β_op)^−E with C = 7.47·exp(−0.133·σ^0.55), B = 0.02526·σ^0.54, E = 0.715·exp(−3.59e-4·σ); U is the midflame wind in ft/min, σ in ft²/ft³. Spread R = I_R·ξ·(1 + φ_w + φ_s)/(ρ_b·ε·Q_ig). The authors now advise not to cap the wind effect. | 12 | `_law_burn` has no wind term | missing |
| W10 | Grass fire spread against wind | CSIRO model, cured grass at 5% moisture: 0.75 m/s at a 10 m wind of 10 km/h, 1.56 m/s at 20 km/h, 2.28 m/s at 30 km/h (derived from the Cheney equations). Experiments: 0.29–2.07 m/s at winds of 2.9–7.1 m/s; consistent forward spread needs ≥ 5 km/h. Rule of thumb: spread ≈ 20% of the 10 m wind (mean error 51%). | 13, 14 | Fire spreads by contact conduction and neighbour radiation only (first pass G4, G7) | missing |
| W11 | Forest and shrub fire spread | Forward spread ≈ 10% of the 10 m open wind in dry fuel (118 wildfires; mean error < 50%); not for grass (abstract). | 15 | None | missing |
| W12 | Cooling by wind | Body: h_c = 12.1·√v W/(m²·K) (ISO 7730): 8.6 at 0.5 m/s, 12.1 at 1, 17.1 at 2, 27.1 at 5 m/s (derived). Still air: h_c = 1.26·ΔT^0.33, ≈ 2.7 W/(m²·K) at ΔT 10 K. The well-known h = 10.45 − v + 10·√v is in kcal/(m²·h·°C), not W/(m²·K). | 16, 17, 18 | `LEAK` takes 0.4% of every voxel's heat per tick, whatever the wind (first pass H14, G5) | wrong (no dependence on air speed) |
| W13 | Wind chill | T_wc = 13.12 + 0.6215·T − 11.37·v^0.16 + 0.3965·T·v^0.16 (T in °C, v in km/h at 10 m), for T ≤ 10 °C and v > 4.8 km/h. 0 °C at 20 km/h → −5.2; −10 °C at 30 km/h → −19.5 (derived). Exposed skin can freeze in 10–30 min at a wind chill of −28 to −39, in 5–10 min at −40 to −47. | 19, 20, 21 | None (no body heat: first pass B19) | missing |
| W14 | Wind that lifts dust and sand | Bagnold: u*_t = 0.10·√((ρ_p − ρ_a)/ρ_a·g·d). Quartz 100 µm: u*_t ≈ 0.15 m/s, ≈ 5.5 m/s at 10 m over sand; 250 µm: 0.23 m/s, ≈ 8.1 m/s (derived). Measured: saltation from 8.0 m/s (at 0.22 m height); dust starts to rise near 4 m/s. | 22, 23 | None | missing |
| W15 | Wind that moves leaves and paper | Leaves rustle at 1.6–3.3 m/s; leaves and twigs move all the time at 3.4–5.4 m/s; dust and loose paper are raised at 5.5–7.9 m/s (10 m), 3.9–5.5 m/s at 1.75 m. | 4, 7 | LEAF is a solid voxel; nothing lifts it | missing |
| W16 | Draught through an open door | Two-way flow Q = (C_d·W/3)·√(g·h³·ΔT/T), C_d 0.61. A 0.9 × 2.0 m door with ΔT 10 K: 0.31 m³/s each way, mean 0.34 m/s, ≈ 0.51 m/s at the top and bottom edges (derived). Room air away from the door: < 0.1 m/s. | 24 | Warm air rises (`warm * 0.0015`), cool air overturns (`_law_o2`). The doorway flow has not been measured. | unknown |
| W17 | When a draught is felt | ISO 7730 draught rate DR = (34 − t)·(v − 0.05)^0.62·(0.37·v·Tu + 3.14) %. At 20 °C and Tu 40%: 18% dissatisfied at 0.15 m/s, 35% at 0.25 m/s (derived). The category limits are unverified. | 16 | No sense of air movement | missing |
| W18 | Tree sway | Natural frequency n0 = 0.0766 + 3.1219·Dbh/H² Hz (Dbh in cm, H in m; secondary): 0.31 Hz for 30 cm × 20 m, 0.39 Hz for 10 cm × 10 m (derived). Internal damping ratio < 0.05. Measured fruit trees 0.63–1.5 Hz. | 25 | Wood in the solver is rigid (`MAT['wood']` rig 1.0); nothing loads it with wind | missing |
| W19 | Wind that breaks trees | ≈ 42 m/s (1 s maximum) breaks more than half of the trees, nearly independent of size and species; no damage reported below 20 m/s; all forests damaged above ≈ 40 m/s. Doubling the height raises it only 9%. | 26 | None | missing |
| W20 | How fast the wind can carry smoke | Real winds are 1–20 m/s and more | 5 | At most 0.4 voxel per tick: 0.8 m/s at 5 cm, 1.6 m/s at 10 cm (derived from `_law_air`) | wrong (cap 1–2 orders too low for outdoor wind) |
| W21 | How long a wind lasts | A mean wind is held up by the weather; only the gusts come and go (W3, W4) | 3 | With no fire or warmth, `vcell` halves every ≈ 0.5 s (derived) | missing (no source of wind) |

### 2.2 Gas species (GS)

**What the sim has now.** The air holds two gas fields, both in grams **per voxel**:
`o2` and `smoke` (`sim.py`, `air.py`). There is no CO, CO2, H2, steam, H2S or SO2.

- Burning adds `dm * SOOT[m]` to `smoke` (wood 0.6, leaf and flesh 0.8, char 0.05,
  oil 1.0) and takes `dm * O2_RATIO[m]` from `o2` (`heat.py:HeatMixin._law_burn`,
  `materials.py:SOOT`, `materials.py:O2_RATIO`). The rest of the products leave the
  books. For wood, 1 g of fuel and 1.33 g of O2 go in; 0.6 g of "smoke" comes out;
  1.73 g of CO2 and water vapour are not counted (derived). The comment at
  `materials.py:SOOT` says so ("CO2 — unaccounted until gas species exist").
- Acid adds every dissolved gram to `smoke` (`fluids.py:FluidsMixin._law_acid`; first
  pass C16). Boiling deletes the water and adds only `p_add += 0.1·g`
  (`heat.py:HeatMixin._law_boil`; first pass L12, G12).
- Smoke moves by `air.py:AirMixin._law_smoke`: it escapes from solids, seeps 0.08 per
  face per tick, **rises 0.4 per tick wherever air is above air, hot or cold**, runs
  along the ceiling, and halves at the open sky. `_law_o2` overturns warm-under-cool
  columns and stirs a hot zone and a cool zone. `_law_air` advects smoke only.
- One cutoff stops smoke being carried: `materials.py:SMOKE_STILL` = 1e-6 g per voxel.
- A body reads `o2` and `smoke` in the air touching its top three voxels of flesh
  (`physiology.py:PhysiologyMixin._law_life`). Smoke raises a load that blocks O2
  uptake (`BODY['smoke_in']` 0.02, `BODY['smoke_out']` 0.001 per tick). The will layer
  fires the percept `chokes` when the load passes 0.05 (`will.py`). There is no smell.
- First-pass rows and gaps that already cover gases: B21–B24 and G9 (breathing, CO
  dose, CO half-life), F19 and G7 (soot yield, smoke layer), P11 (smell of smoke),
  L12 and G12 (steam).

A conversion used below: a gas at φ ppm by volume holds φ·1e-6·ρ grams per litre,
with ρ from GS9. A 5 cm voxel holds 0.125 L of air, a 10 cm voxel 1 L.

| # | Quantity | Real value | Src | Sim value | Verdict |
|---|---|---|---|---|---|
| GS1 | Incapacitation dose of CO (ISO 13571) | FED = Σ φCO·vCO2/35,000·Δt (φ in ppm, Δt in min); incapacitation at FED = 1; uncertainty ≈ 35%. CO2 multiplier vCO2 = exp(%CO2/5): 1.49 at 2%, 2.72 at 5% (derived). So 1,000 ppm CO alone reaches FED 1 in 35 min, and in ≈ 13 min with 5% CO2 (derived). | 27 | No CO. Soot grams stand in for CO (`BODY['smoke_in']`, `physiology.py:_law_life`). First pass B23, G9. | missing |
| GS2 | Purser FED with all terms (SFPE, as in FDS) | FED_tot = (FED_CO + FED_CN + FED_NOx + FLD_irr)·HV_CO2 + FED_O2. FED_CO = ∫ 2.764e-5·C^1.036 dt (C in ppm, t in min): 1,000 ppm gives FED 1 in ≈ 28 min (derived). HV_CO2 = exp(0.1903·%CO2 + 2.0004)/7.1. FED_O2 = ∫ dt / exp[8.13 − 0.54·(20.9 − %O2)]: time to incapacitation ≈ 140 min at 15% O2, 28 min at 12%, 9.4 min at 10%, 3.2 min at 8% (derived). Low O2 is not multiplied by HV_CO2. | 28, 29 | One soot load and one blood-O2 value. Blood O2 relaxes to inhaled O2 with a 1.25 s time constant (first pass B21). | missing (the FED terms are the replacement the `BODY` comment already names) |
| GS3 | CO uptake (Stewart) | %COHb = 3.317e-5·ppm^1.036·RMV·t (RMV in L/min, t in min), valid up to 1 h; incapacitation at ≈ 30% COHb for light work (RMV 25 L/min). CO binds haemoglobin 230–270× more strongly than O2. Blood CO half-life 2–6.5 h; extra O2 shortens it. | 29, 30 | No COHb. The soot load clears in ≈ 5–12 s (first pass B24). | missing (see G9) |
| GS4 | CO effects by level and time | 200 ppm: slight headache in 2–3 h. 800 ppm: dizziness and nausea within 45 min, insensible within 2 h. 1,600 ppm: death within 30 min. 6,400 ppm: death in < 20 min. 12,800 ppm: death in < 3 min (secondary table). NIOSH: 1,000–1,200 ppm for 1 h is unpleasant but not dangerous; 1,500–2,000 ppm may be dangerous after 1 h. SFPE (light activity): life-threatening 6,000–8,000 ppm at 5 min, 1,400–1,700 ppm at 30 min. | 31, 32, 27 | None | missing |
| GS5 | Exposure limits (NIOSH IDLH) | CO 1,200 ppm; CO2 40,000 ppm (4%); H2S 100 ppm; SO2 100 ppm. | 32, 33, 34, 35 | None | missing |
| GS6 | Public emergency limits (EPA AEGL-2 / AEGL-3, 10 min · 60 min · 8 h, ppm) | CO: AEGL-2 420 · 83 · 27; AEGL-3 1,700 · 330 · 130. H2S: AEGL-2 41 · 27 · 17; AEGL-3 76 · 50 · 31. SO2: AEGL-2 0.75 at all times; AEGL-3 30 · 30 · 9.6. | 36, 37, 38 | None | missing |
| GS7 | CO2 by level | 3.3–5.4% for 15 min: deeper breathing. 7.5% for 15 min: "inability to breathe", headache, dizziness, disorientation. ≥ 10%: unconsciousness after 15 min. NIOSH: 70,000–100,000 ppm (7–10%) for a few minutes gives unconsciousness. SFPE: life-threatening 7–8% at 5 min, 6–7% at 30 min. | 31, 33, 27 | No CO2. Burnt mass beyond soot is deleted (derived above). | missing |
| GS8 | H2S by level | Odour 0.01–1.5 ppm (OSHA); more offensive at 3–5 ppm; 20 ppm fatigue, headache; 100 ppm loss of smell after 2–15 min; 100–150 ppm olfactory paralysis; 500–700 ppm collapse in 5 min; 700–1,000 ppm "knockdown" in 1–2 breaths, death within minutes. | 39, 34, 40 | None | missing |
| GS9 | Density at 20 °C, 1 atm, and which way it goes | Air 1.205 kg/m³. Relative to air: CO 0.97 (neutral), CO2 1.53 (pools low), H2S 1.19 (pools low), SO2 2.26 (pools low), H2 0.07 (rises fast), CH4 0.6 (rises). Steam at 100 °C 0.598 kg/m³, ≈ 0.50 of 20 °C air. Air at 300 °C ≈ 0.62 kg/m³ (ideal gas, derived). ICSC: CO2 "may accumulate in lowered spaces causing a deficiency of oxygen". | 41, 42, 43, 44 | Smoke rises 0.4 of a cell per tick through any air-over-air pair at any temperature (`_law_smoke`). No gas has a density. | wrong for cold smoke; missing for the species |
| GS10 | Odour thresholds (the new sense) | H2S 0.0005–0.3 ppm, "strong odor of rotten eggs" (EPA odour awareness 0.01 ppm). SO2 0.3–1 ppm, "very strong, pungent". NH3 5 ppm, "pungent, suffocating". CO, CO2, H2 and CH4 are odourless. Wood smoke: no threshold found (unverified). | 40, 37, 45, 46, 42, 43, 47 | No smell. The only gas percept is `chokes` (load > 0.05). | missing |
| GS11 | Smell fades and lies | H2S: olfactory fatigue at 100 ppm in 2–15 min; olfactory paralysis at 100–150 ppm. So the most dangerous level can smell of nothing. | 39, 48 | None | missing |
| GS12 | Smell thresholds against the sim's cutoff | 0.0005 ppm H2S ≈ 7e-10 g/L; 0.3 ppm ≈ 4.3e-7 g/L (derived from GS9 and GS10). | 41, 40 | `SMOKE_STILL` 1e-6 g per voxel = 8e-6 g/L at 5 cm, 1e-6 g/L at 10 cm. At 5 cm, a species read with the smoke cutoff stops moving ≈ 20–11,000× above the level a nose can detect (derived). | wrong if reused (each species needs its own cutoff) |
| GS13 | Flammability limits in air (vol %) and autoignition | CO 12.5–74%, 605 °C. H2 4–75%, 560 °C. H2S 4–44% (NPG; ICSC 3.9–45.5%), 260–270 °C. CH4 5–15%, 537 °C. CO2 and SO2 do not burn. H2S "may travel to a source of ignition and flash back" and burns to SO2. | 42, 43, 47, 48 | No gas burns. `smoke` is not a fuel. | missing |
| GS14 | H2 behaviour | Rises at up to ≈ 20 m/s ("almost 45 mph"); ignition energy 0.02 mJ (natural gas 0.29). DOE: it "would briefly collect on the ceiling". | 47 | No H2. Acid on iron makes no H2 (first pass C6). | missing |
| GS15 | CO yield of wood, well ventilated | 0.004 g/g for small Douglas-fir samples (Tewarson, as quoted by NIST). Burning dry 3–6 m trees: CO 0.047–0.101 g/g (mean 0.081), soot 0.0034–0.0054 g/g, CO2 1.21–1.93 g/g. The classic red-oak set (CO2 1.27, soot 0.015) was not opened here (unverified). | 49 | `SOOT[WOOD]` 0.6 g/g of "smoke", no CO, no CO2 (first pass F19) | missing (CO); wrong (soot 0.6 vs 0.004–0.015) |
| GS16 | CO yield of a starved fire (equivalence ratio > 1) | Nearly constant at ≈ 0.2 g CO per g fuel. Ponderosa pine 0.14 g/g (3.2 vol % CO in the hood). Mulholland's design value: 0.3 g/g for under-ventilated fires. CO rises once φ > ≈ 0.5. | 50 | `ration` in `_law_burn` cuts the burn rate when O2 runs short; the products do not change | missing (this is the reason a closed-room fire kills) |
| GS17 | Charcoal brazier in a closed space | 800 g of glowing charcoal in a 19 m³ room: CO above 3,000 ppm within 2 h. A closed garage simulation: 750–1,100 ppm. | 51 | CHAR burns at 30 kJ/g with soot 0.05; it makes almost no "smoke" and no CO | missing |
| GS18 | Molecular diffusion in air (cm²/s, 20 °C) | H2 0.756; H2O 0.242; CO 0.208; CO2 0.16; SO2 0.124 (296 K); NH3 0.228. Diffusion alone spreads CO2 1 m in ≈ 8.7 h and 5 m in ≈ 9 days (t = L²/2D, derived). | 52, 53 | Smoke seep 0.08 per face per tick: D = 0.08·dx²/`TICK_S` = 80 cm²/s at 5 cm, 320 cm²/s at 10 cm. O2: 100–450 cm²/s at 5 cm, ×4 at 10 cm (derived). | ok for game as **room mixing** (next row), not as molecular diffusion; voxel-size dependent |
| GS19 | How fast a room mixes | Natural convection in a 31 m³ room: 7–10 min with a heated wall, 80–100 min when nearly isothermal (abstract only). Fan-stirred: 2.4–23 min, t ∝ P^(−1/3). An equivalent D = L²/2t over 3.5 m is ≈ 10–150 cm²/s for natural convection (derived). | 54, 55 | Cold-air O2 mixing at 5 cm ≈ 100 cm²/s → a 3.5 m room mixes in ≈ 10 min; at 10 cm in ≈ 2.5 min (derived) | ok for game at 5 cm; voxel-size dependent |
| GS20 | Steam | 1 g of water at 20 °C becomes 1.673 L of steam at 100 °C (≈ 1670×). Latent heat 2256 kJ/kg. Steam condensing on skin gives up its latent heat "rapidly raising the temperature of the skin". | 44, 56 | Boiled water is deleted (first pass L12). | missing (see G12) |

**Note on smoke logging.** Hot smoke rises because it is hot (GS9: air at 300 °C has half
the density of air at 20 °C). Once it cools to room temperature it no longer rises,
and gases heavier than air sink. The sim's rise rule ignores temperature, so a cold
cloud of smoke still climbs at 0.4 cells per tick. A CO2 or H2S field that copied
this rule would rise out of a cellar instead of filling it.

### 2.3 Contact chemistry for one general reaction table (R)

**What the sim has now.** Three separate laws change what matter is:

- `materials.py:REACTIONS` is a table keyed by (fluid, solid). Each row is (rate in
  g per face per tick at the 10 cm reference, ml of fluid spent per g, J per g).
  `fluids.py:FluidsMixin._law_acid` reads it. The rate is linear in potency and does
  **not** depend on temperature. Every dissolved gram becomes `smoke`. There are
  rows for ACID and WEAK_ACID only. The first pass covers these rows (C1–C19, G8).
- `heat.py:HeatMixin._law_burn` is a second, separate law. Its rate is
  `rate * scale**2 * (T - ign) * damp` grams per tick: linear in the temperature above
  ignition, zero below it, capped by the O2 ration. WOOD and FLESH turn to CHAR when
  less than 18% of a full voxel's mass is left, and to ASH below 5%.
- `heat.py:HeatMixin._law_melt` and `_law_boil` change phase.

There is no solid–gas reaction (rust), no solid–liquid reaction other than acid
(quicklime and water, salt and water), no solid that reacts on its own when hot
(pyrolysis, gunpowder), and no fluid–fluid reaction (acid and base). Roadmap 107
asks for one table: reactants, products, a rate that depends on temperature, and
the heat. The rows below give the real data for that table. The acid rows of the
first pass (C1–C19) are not repeated.

How to read the rate column. A rate given per area (g/(m²·s)) becomes a per-tick
amount through the contact area of one face: `vox_m² × TICK_S` (6.25e-5 m²·s at
5 cm). A first-order rate k (1/s) acts on the mass of the voxel. Arrhenius means
k = A·exp(−Ea/(R·T)) with T in kelvin and R = 8.314 J/(mol·K).

| # | Reaction | Reactants → products (g per g) | Rate law, real parameters | Heat | Src | Sim value | Verdict |
|---|---|---|---|---|---|---|---|
| R1 | Wood pyrolysis (no flame) | Oak → gas + tar + char, three parallel first-order reactions; char ≈ 0.30 of the wood at 329–392 °C (0.299 ± 0.028) | A_gas 1.44e4 /s, E 88.6 kJ/mol; A_tar 4.12e6 /s, E 112.7; A_char 7.38e5 /s, E 106.5 (fitted at 325–385 °C). Half of the wood has reacted after ≈ 24 min at 300 °C, 4.0 min at 350 °C, 0.87 min at 400 °C (derived). Extrapolated: ≈ 3.2 h at 250 °C, ≈ 37 h at 200 °C (outside the fitted range). | Between −613 and +1680 kJ/kg in the literature; endothermic below 250 °C, exothermic at 250–340 °C | 57, 58, 59 | Nothing happens below `SOLID[WOOD]` ignition 300 °C. At 350 °C with air, a 5 cm voxel (75 g) burns half its mass in ≈ 3.7 s (derived from `_law_burn`, with its `damp` factor). No pyrolysis without flame. First pass F2, F5, G4. | wrong (≈ 65× fast at 350 °C; nothing below 300 °C) |
| R2 | Slow charring at low temperature | Wood → char over months | Matchstick samples: 150 °C for 165 days lost 65% of weight, "like charcoal"; 140 °C for 320 days lost 45%. Ignition ≈ 250 °C after minutes to hours; long-term ignition recorded down to 77 °C (steam-pipe fires 3 months to 15 years after installation). | — | 60 | None. A shelf held at 240 °C never changes (first pass F2). | missing (only for scenes that run for days) |
| R3 | Char yield against temperature | Oven-dry wood → charcoal: 42% at 300 °C (68% fixed C), 33% at 500 °C (86%), 30% at 700 °C (92%). Exothermic breakdown starts near 280 °C. | — | — | 61 | 18% of a full voxel's mass becomes CHAR (`_law_burn`, the `0.18 * dens * vox_l` test) | wrong (≈ 40% low; minor) |
| R4 | Charcoal glowing (kinetic regime) | C + O2 → CO2 and CO | Douglas-fir biochar, first order in remaining mass: A 3.76e7 /s, E 126 kJ/mol (333–433 °C); red oak A 3.12e6 /s, E 111. Half-life ≈ 11 min at 350 °C, 110 s at 400 °C, 39 s at 433 °C (derived, fine powder). Above ≈ 500 °C oxygen transport controls the rate (apparent E 8–27 kJ/mol). Whole-wood char step: E 183 kJ/mol (abstract). | see R5 | 62, 63 | `SOLID[CHAR]` ignites at 330 °C; rate `0.01 * scale**2 * (T - 330)` g/tick. A 5 cm CHAR voxel (37.5 g) at 400 °C loses half in ≈ 4.2 s (derived, with `damp`). Linear in T, not Arrhenius. | wrong (≈ 25× fast against powder kinetics; wrong law) |
| R5 | Heat and products of char burning | C + 2.664 O2 → 3.664 CO2; C + 1.332 O2 → 2.332 CO. CO2/CO by mole at 0.21 atm O2: 1.38 at 400 °C, 0.49 at 600 °C, 0.25 at 800 °C (Tognotti correlation, derived), so a glowing ember makes mostly CO at its surface. | — | −32.76 kJ/g C to CO2; −9.20 kJ/g C to CO | 64, 65, 66 | `SOLID[CHAR]` 30,000 J/g, `O2_RATIO[CHAR]` 2.6; products: soot 0.05 g/g, nothing else (first pass F12) | matches (heat and O2); missing (CO, CO2) |
| R6 | Rust under water | Fe + O2 + H2O → FeOOH (1 g Fe → 1.591 g FeOOH, using 0.430 g O2 and 0.161 g H2O; derived) | Bare steel, design rates per surface: fresh water, permanently immersed 0.025 mm/yr; top 300 mm of fresh water 0.050; sea water 0.035 (permanent) to 0.075 (low-water zone); undisturbed soil 0.015. Quiet sea water, measured: 0.1–0.2 mm/yr, pits several times deeper; "governed by diffusion of oxygen to the surface". | ≈ −7.3 to −7.5 kJ/g Fe (derived from ΔfH of goethite −559.5 kJ/mol, hematite −824.2) | 67, 68, 69, 70 | IRON has no reaction with water or O2 | missing (first pass C17 gives the ISO 9223 air rates) |
| R7 | Flash rust | Wet bare steel shows rust "within 0.5 hour or longer, depending on environmental conditions". Yellow-brown and adherent at first, red-brown and loose later. | first visible colour in ≥ 0.5 h | — | 71 | None | missing (this is the rust a player sees) |
| R8 | Iron powder heater | Iron powder + water + salt + carbon (a hand warmer) | Reaches its maximum in ≈ 20 min; peak up to 74 °C; modest heat "for many hours" (secondary) | as R6 | 72 | None | missing (low) |
| R9 | Quicklime and water (slaking) | CaO + 0.321 H2O → 1.321 Ca(OH)2 (derived) | EN 459-2 test (lime in excess water in a Dewar): time to 60 °C is 0.4–2.2 min for soft-burnt lime, 0.5–10.9 min medium-burnt, 1.3–22.2 min dead-burnt; peak 63.5–85.5 °C in the test. In bulk, "up to some 300 °C" (secondary). SDS: "can release heat sufficient to ignite combustible materials". Slaked lime pH 12.45. | −64.0 to −65.2 kJ/mol = −1.14 to −1.16 kJ/g CaO | 73, 74, 75, 76, 77 | No lime | missing |
| R10 | Acid and base | H⁺ + OH⁻ → H2O | Diffusion-limited, k = 1.3e11 /(M·s): effectively instant on contact (secondary) | −55.8 kJ/mol (−3.10 kJ per g of water formed; derived from Cerutti, abstract); −57.6 kJ/mol (secondary) | 78, 79 | No base fluid. `MISCIBLE` blends water into acid with no heat (first pass C11). | missing |
| R11 | Acid on wood ash | Wood ash: pH 10.4 (9–13.5); calcium carbonate equivalent 25–60% (40–50 in a second source); CaCO3 43% (22–92%). CaCO3 + 2H⁺ → 0.440 g CO2 per g; K2CO3 → 0.318 g CO2 per g (derived). Fresh ash leachate is ≈ 92% hydroxide (abstract). | Calcite, acid path: log k = −0.30 mol/(m²·s) at pH 0, E 14.4 kJ/mol, first order in H⁺. The low E means transport control: a powder in strong acid fizzes for seconds. | Hydroxide part as R10; carbonate part not verified here (first pass C10: ≈ 152 J/g CaCO3) | 80, 81, 82, 83 | `REACTIONS[(ACID, ASH)]` 8.0 g/face/tick, 0.5 ml/g, 50 J/g: ash dissolves into smoke and spends acid | ok for game on speed (a powder); wrong products (no CO2, no salt). Settles first pass C19. |
| R12 | Salt dissolving | NaCl(s) → Na⁺ + Cl⁻ in water. Solubility 36.0 g per 100 g water at 25 °C, 36.8 at 50 °C, 38.1 at 80 °C. | Surface reaction far from saturation: log k = −0.21 mol/(m²·s), E 7.4 kJ/mol → 36 g/(m²·s) ≈ 1 mm/min of recession (derived; the fast, stirred limit). In still water diffusion makes it slower; no measured still-water rate was found (unverified). | +3.87 kJ/mol = +66 J/g: the water cools slightly (secondary) | 84, 83, 85 | No salt | missing |
| R13 | Black powder | 75 KNO3 / 15 charcoal / 10 S → 55.9% solids (K2CO3, K2SO4, K2S …), 43.0% gas (CO2, N2, CO, H2S, H2, CH4), 1.1% water (secondary). Gas ≈ 271 cm³/g. | Deflagrates, never detonates. Inside a grain ≈ 6 cm/s (secondary); the flame jumps between grains at ≈ 9 m/s (secondary). Confined: ≈ 170 m/s (coarse) to ≈ 950 m/s (fine). Ignition: 125 °C (US Army manual) against 200–464 °C (an uncited spec sheet): the sources disagree. Burn rate at 1 atm of ≈ 0.4 cm/s: unverified. | Heat of explosion 684 cal/g = 2.86 kJ/g (spec sheet, no primary); ≈ 3 MJ/kg (secondary) | 86, 87, 88 | None | missing |
| R14 | How rates change with temperature | Arrhenius k = A·exp(−Ea/RT). Rates rise ≈ 2–3× per +10 °C for common Ea; Q10 = 2 at 25–35 °C means Ea ≈ 53 kJ/mol, Q10 = 3 means ≈ 84 kJ/mol (derived). | — | — | 89, 90 | `_law_acid` reads no temperature. `_law_burn` is linear in T − ign and zero below ign. | wrong form |
| R15 | Burning sulfur (brimstone) | S + 0.998 O2 → 1.998 SO2 (derived) | — | −296.8 kJ/mol = −9.26 kJ/g S | 91 | None | missing (gives the SO2 smell of GS10) |
| R16 | Lime kiln | CaCO3 → 0.560 CaO + 0.440 CO2 (derived) | Equilibrium pCO2 = exp(16.3 − 19,130/T) atm, so 1 atm at ≈ 900 °C (derived); kilns run at 850–950 °C; lime is made "above 825 °C". | +178 kJ/mol = +1.78 kJ/g CaCO3 (absorbs heat) | 92, 77 | None | missing (low) |

**One table can hold all of these.** Each row needs: two reactants and how they meet
(solid on gas, solid in liquid, liquid with liquid, or one solid alone when hot),
products as grams per gram (solids, fluids and gas species), a rate law (an Arrhenius
pair, or a measured flux at one temperature plus a Q10, or "transport limited"), and
the heat in J/g. Burning fits as the row "fuel + O2 → CO2 + H2O + CO + soot", with the
CO share taken from the O2 supply (GS16). The acid rows of the first pass fit as
rows with a concentration threshold (first pass G8).

### 2.4 Body clocks (BC; the domain roadmap 107 calls B2)

**What the sim has now.** A person is a dict made in `persons.py:add_person`
(`blood_o2`, `smoke`, `burn`, `hurt`, `awake`, `alive`, …). `physiology.py` adds
`blood_ml` and `wounds`. Every tick, for every person, `PhysiologyMixin._law_life`
does breathing, the smoke load, burns, faint, waking and death, and `_law_bleed` does
bleeding and clotting. That is the whole of the body's chemistry.

- **Sleep** is a flag, `p['asleep']`, that a scene sets. A noise of 60 dB at the ear
  clears it (`senses.py:SensesMixin._law_hear`, `SOUND['wake_db']`). Nothing makes a
  body sleepy, and a woken body is at full function at once.
- **Damage only climbs.** `burn` and `hurt` never fall. The `_wound` docstring says
  "wounds do not heal here any more than burns do". Clotting stops bleeding
  (`BLEED['clot_m2_s']`), but the wound stays.
- **No inner temperature.** Flesh starts at `AMBIENT` 20 °C and makes no heat (first
  pass B19, G15). `p['skin']` is the hottest flesh voxel.
- **No stores.** No energy, water, food, alcohol, drug, fatigue or infection.
- First-pass rows that touch this domain: B19 (body heat), B21–B24 and G9 (breath,
  CO), B26 (O2 use and fatigue), P19 and P23 (sleep and alcohol in perception), A22
  and G18 (waking).

| # | Quantity | Real value | Src | Sim value | Verdict |
|---|---|---|---|---|---|
| BC1 | Sleep pressure (Borbély two-process model) | Process S rises during waking with time constant 18.2 h and falls during sleep with 4.2 h (Daan, Beersma & Borbély 1984, as used by Skeldon et al.). Process C is a 24 h sinusoid, amplitude ≈ 0.10–0.12. Sleep starts when S reaches the upper threshold (example values H0+ 0.60–0.67) and ends at the lower one (H0− 0.17), each shifted by C. | 93, 94 | None. `asleep` is set by the scene. | missing |
| BC2 | Sleep need | Adults: ≥ 7 h a night. More than 9 h may suit young adults and people recovering from sleep debt. | 95 | None | missing |
| BC3 | Falling asleep | Sleep latency 0–30 min is good; 46–60 min is poor; > 60 min is poor at all ages. | 96 | None | missing |
| BC4 | Sleep inertia | Performance back to baseline within ≈ 15–30 min of waking; full recovery takes at least 1 h; subjective alertness keeps rising for up to 2 h. Worse when woken near the circadian low. | 97 | A woken body acts at full function on the same tick | missing |
| BC5 | Performance after sleep loss | 17 h awake ≈ the impairment of BAC 0.05%; 24 h ≈ 0.10%. From hour 10 to hour 26 awake, tracking performance falls 0.74% per hour (≈ +0.004% BAC per hour). Replicated: 17–19 h equals or exceeds BAC 0.05%. Fourteen nights of ≤ 6 h equal up to 2 nights of total deprivation, and the subjects were "largely unaware" of it (abstract). | 98, 99, 100 | None | missing |
| BC6 | How long a muscle can hold | Endurance time ET = 21.92·f^−1.98 s, f = fraction of maximum force (pooled; shoulder b0 14.86, b1 −1.83). 20% → 8.8 min; 50% → 86 s; 80% → 34 s; 100% → 22 s (derived). | 101 | `strength_N` 400, `wrist_Nm` 20 and `arm_Nm` 60 are limits with no clock (`materials.py:BODY`). A load within them is held for ever (first pass B26). | missing |
| BC7 | Recovery after a sprint | PCr resynthesis: fast half-time 21–22.5 s, slow > 170 s. After a 30 s all-out sprint: 19.7% of rest at the end, 65% after 90 s, 85.5% after 6 min. PCr falls below 40% within 10 s of intense work; the aerobic system needs ≈ 45 s to reach full output. | 102 | No run gait (first pass B4), no energy store | missing |
| BC8 | Walking all day | Army foot march: 32 km in 8 h at 4 km/h. Forced-march maximum 56 km in 24 h, 96 km in 48 h. Halts: 15 min after the first 45 min, then 10 min in every 50. | 103 | Walkers never tire (walking speed: first pass G3) | missing |
| BC9 | Sustainable work rate | Average aerobic load over an 8 h day should not exceed ≈ 30% of maximum (ILO/WHO recommendation, second-hand). | 104 | None | missing |
| BC10 | Sweating | 0.5–2.0 L/h is common. Hard work at ≈ 1000 W needs ≈ 1.2 L/h of sweat evaporated. Daily sweat loss ≈ 3.5–8.8 L/d in athletes. | 105 | None | missing |
| BC11 | Dehydration | > 2% of body weight lost as water lowers physical and mental performance. > 5%: headache, irritability. 8–10%: "performance becomes significantly impaired". Death at 15–20% of body water, "in as little as 3 days in hot climates". Minimum need at rest ≈ 1.5 L/d (≈ 1 L drunk, the rest from food and metabolism); ≥ 4 L/d walking in heat. Reference intake 3.7 L/d (men), 2.7 L/d (women). | 105, 106, 107 | None | missing |
| BC12 | Hunger and starvation | Liver glycogen is gone after 12–24 h of fasting. Weight loss ≈ 0.9 kg/day in week 1, 0.3 kg/day by week 3 (abstract). With water, normal-weight people survive "up to 2 to 2.5 months"; "serious complications and death occur especially from the fortieth day on"; serious problems begin near 18% weight loss. Aerobic capacity holds to 10% loss, then falls 4% per further 1%. Minnesota experiment: 24 weeks at ≈ 1800 kcal/d, most lost ≥ 25%. Energy use ≈ 2400 kcal/d active, ≈ 1400 kcal/d at rest. | 108, 109, 110, 106, 111 | None | missing |
| BC13 | Cold water | Cold shock in water < 15 °C: gasping within 30 s, over in 2–3 min. "1-10-1": 1 min of cold shock, 10 min of useful movement, 1 h before hypothermia can bring unconsciousness. Swimmers at 10 °C: 1 of 10 failed at 61 min, 4 more were pulled out before 90 min at a core of 35 °C (abstract). Core cooling (Hayward): 3.7 °C/h in 5 °C fresh water, 4.7 °C/h at 0 °C; modelled survival 136 min at 5 °C. Survival table (not peer-reviewed): 0–4 °C exhaustion 15–30 min, survival 30–90 min; 4–10 °C 30–60 min, 1–3 h; 10–16 °C 1–2 h, 1–6 h. | 112, 113, 114, 115, 116 | No core temperature. A flesh voxel in water takes the water's temperature through `FLUID[WATER]` contact k 30 in seconds (first pass H20, G5). Faint and death read only O2 and damage. | missing |
| BC14 | Hypothermia stages | Mild 35–32 °C, moderate 32–28 °C, severe < 28 °C (core). | 117 | None | missing |
| BC15 | Cold air | Shivering people in wet light clothing at 5 °C with wind: core fell 0.17 °C/h over 3 h (abstract). Snow burial: 0.7 °C/h (1.2 °C/h with high CO2) (abstract). | 118, 119 | None | missing |
| BC16 | Heat illness | Core > 38 °C is "undesirable" for workers; heat-exhaustion collapse risk ≈ 25% at 39.2 °C; near 40 °C "most individuals are in imminent danger". Exertional heat stroke: CNS dysfunction with core > 40.5 °C. Cold-water immersion cools ≈ 0.2 °C/min; within 30 min of collapse it gave 0% deaths in > 2000 cases. Work limit: RAL = 59.9 − 14.1·log10 M (°C-WBGT, M in W). | 120, 121 | Burns only, from flesh voxels above `hurt_T` 55 °C (first pass B17, B20) | missing |
| BC17 | Alcohol in the blood (Widmark) | Standard drink = 14 g of ethanol (US). BAC (g%) = 0.8·A·100/(W·1000·r) − β·t, A in ml of ethanol, W in kg; Widmark's r ≈ 0.7 (men), 0.6 (women), or r = 1.0181 − 0.01213·BMI (men). β = 0.0155 ± 0.0029 %/h (95% of people 0.010–0.022). Peak ≈ 40 min after beer on an empty stomach; BAC can keep rising after drinking stops. Example: 4 standard drinks for a 70 kg man → ≈ 0.11% at once, ≈ 0.08% two hours later (derived). | 122, 123, 124, 125 | None (first pass P23) | missing |
| BC18 | Effects by BAC (%) | 0.05–0.07: euphoria, slower reactions. 0.07–0.10: impaired reaction, attention and judgement. 0.10–0.20: drowsy, disoriented. 0.20–0.30: staggering. 0.30–0.40: stupor or unconscious. > 0.40: coma, possible death. Blackout probability = 2.21·BAC − 0.18. | 123 | None | missing |
| BC19 | A one-compartment poison model | C(t) = F·D·ka/(V·(ka − ke))·(e^(−ke·t) − e^(−ka·t)), t½ = ln2/ke. Paracetamol: peak within 90 min, V ≈ 0.9 L/kg, t½ 2.0–2.5 h (> 4 h with liver injury). Toxic above ≈ 150 mg/kg or 7.5 g. Nomogram: 200 µg/ml at 4 h falling to 25 at 16 h (a 4 h half-life, derived); treatment line 150 µg/ml at 4 h. Liver enzymes rise after 24–36 h; injury peaks at 3–5 days. Caffeine: tmax 30–120 min, V ≈ 0.67 L/kg, t½ 2–8 h. Example: 10 g of paracetamol in a 70 kg man gives at most D/V ≈ 160 µg/ml (derived). | 126, 127, 128, 129, 130 | None | missing |
| BC20 | Wound healing | Haemostasis at once; fibroblasts arrive in 2–3 days; proliferation lasts up to 3 weeks; collagen builds for 4–5 weeks; remodelling goes on for a year or more. Tensile strength: 3% at 1 week, 20% at 3 weeks, peak 80% of uninjured skin at 3 months. | 131, 132 | Wounds never heal; `hurt` and `burn` never fall (`physiology.py:_wound`) | missing |
| BC21 | Infection | Tetanus: incubation 3–21 days, ≈ 8 on average; ≈ 1 in 10 die even in intensive care. Gas gangrene: incubation usually < 24 h; mean survival after onset ≈ 15 h (traumatic); untreated "universally fatal". Wound infections in one high-incidence cohort were found on days 9–16. Septic shock: 79.9% survival with antibiotics in the first hour of low blood pressure, then 7.6% less for each hour of delay (abstract). | 133, 134, 135, 136 | None | missing |

**What a body clock needs from the sim.** Most rows above are one number per person
that moves at a known rate: S (sleep pressure), water, energy, BAC, a drug level, a
core temperature, a healing fraction per wound. Each has a closed form between events
(an exponential or a straight line), so a sleeping night can be skipped in a few steps
(Section 6).

### 2.5 Slow world (SW)

**What the sim has now.** Nothing changes with time alone. Matter changes only by
heat (burning, boiling, melting), by acid, and by force. Food, drink and bodies do
not exist as kinds of matter (a dead body is FLESH and stays as it is). Wood can hold
water as fluid inside the solid voxel (`fl`, `fvol`), and that water leaves only by
boiling: `fluids.py:_law_flow` skips fluid held inside a solid, and nothing
evaporates (first pass L15, L19, G16). Fresh LEAF holds water the same way. There
is no plant growth, no rot and no fermentation.

Roadmap 107 plans one scheduled event per object, with a speed that depends on heat
and wetness ("biological rates rise about 2–3× per +10 °C"). The rows below test that
assumption (SW2).

| # | Quantity | Real value | Src | Sim value | Verdict |
|---|---|---|---|---|---|
| SW1 | Food left out | Bacteria grow fastest at 4.4–60 °C (the "danger zone") and can double "in as little as 20 minutes". Perishable food should not be out of the fridge for more than 2 h, or 1 h above 32 °C. At 4.4 °C spoilage is gradual; at −18 °C bacteria are dormant. | 137 | None | missing |
| SW2 | How spoilage speed depends on temperature | Square-root model: √r = b·(T − T0), T0 a minimum temperature of the organism; it also fits moulds and yeasts. Chilled fish: relative rate (1 + 0.1·T)² against 0 °C, for −3 to +15 °C. Fish at 20–30 °C spoil ≈ 25× faster than at 0 °C. So the effective Q10 is ≈ 4 between 0 and 10 °C and ≈ 2.2 between 15 and 25 °C (derived from the fish model and from raw-milk keeping times). A general "Q10 = 2–3" for spoilage: no opened source (unverified). | 138, 139 | None | missing. Roadmap 107's "2–3× per +10 °C" holds above ≈ 15 °C and is too low near 0 °C: use the square-root law. |
| SW3 | Bread goes mouldy | Wheat bread at ≈ 22 °C in a plastic bag or bread box: visible mould from day 4–6. At 8 °C in a plastic bag: no mould within the 10-day test. A sourdough mixed bread: mould only after day 10. | 140 | None | missing |
| SW4 | Milk sours | Pasteurised milk in the fridge (below 7 °C): 10–21 days; ≈ 2 weeks at 4–7 °C. Raw milk at room temperature "will clabber" in 12–24 h (a homesteading magazine; weak). Codex: cool raw milk within 2 h to ≤ 6 °C. | 141, 142, 143, 144 | None | missing |
| SW5 | Meat goes off | Days until slime on fresh meat: 10 at 0 °C, 4 at 3 °C, 3 at 5 °C, 2 at 10 °C, 1 at 16 °C. Off-odour at 10⁶ bacteria per cm², slime at 10⁸. Ground pork rejected after 267 h at 0 °C, 147 h at 5 °C, 79 h at 10 °C, 54 h at 15 °C. Aerobic red meat: 15 days at 1 °C, 6.9 days at 7 °C. At 20–30 °C: no opened source (unverified). | 145, 146, 147 | None | missing |
| SW6 | Drying and salting preserve | Most microbes do not grow below water activity 0.91 (staphylococci and moulds are the exceptions). Fresh meat > 0.98; dry sausage 0.85–0.93; dried ham and jerky usually < 0.88. A whole country ham keeps at room temperature "for up to 1 year"; 6 months is the traditional cure. | 148, 149 | None | missing |
| SW7 | What fermenting makes | Glucose → 2 ethanol + 2 CO2: 0.511 g ethanol and 0.489 g CO2 per g of sugar (derived). Yeast gets close to this; ≈ 4% of the yield goes to glycerol, the rest of the gap to yeast growth. | 150 | None | missing |
| SW8 | Brewing beer | Ales ferment at 15–25 °C, lagers at 6–14 °C. A lab ale at 20 °C takes ≈ 6 days; a lager at 10 °C ≈ 10 days. ABV = (OG − FG) × 131.25. A recreated beer of the 1570s was ≈ 5% ABV; workers drank 5–10 pints a day. Medieval "small beer" strength: unverified. | 151, 152, 153 | None | missing |
| SW9 | Making wine | Ferments at 28 °C finished in ≈ 10 days. Must at 26 °Brix gave 14.8–15.1% ethanol. Fermentation goes from sluggish (11 °C) to normal to stuck (35 °C) (abstract); at 35 °C many yeasts die. | 154, 155, 156, 157 | None | missing |
| SW10 | Green wood dries | 25 mm lumber to 20% moisture, air-dried: southern pines 30–150 days, northern red oak 70–200, white oak 80–250. Red oak stacked on 1 June: 62 days; on 1 January: 144 days. Fibre saturation ≈ 30%; air-dried wood ends at 20–25%. Firewood: "at least six months" to < 20%; green wood can hold 100% or more (dry basis). | 158, 159, 160 | Water held in a wood voxel stays until it boils. Wet wood is slow to light (that part matches first pass F23), but it never dries. | missing |
| SW11 | Wet cloth dries | Fabric drying from both sides in still air (20 °C, 65% RH, 0.1 m/s): 1.6 g/(m²·min) ≈ 0.096 kg/(m²·h) (derived). First pass L15 gives evaporation 0.169 kg/(m²·h) for water at 0.5 m/s. Hours for laundry on a line: unverified. | 161 | No cloth (first pass L20); no evaporation | missing |
| SW12 | Hay dries in the field | Alfalfa is 75–80% water when cut. It takes 1–2 days to reach 60% and 3–5 days to reach 20%. Small bales are made below 20%, large ones below 14%. | 162, 163 | None | missing (low) |
| SW13 | Grass grows | Irish pasture in summer: 55–65 kg dry matter per ha per day (≈ 6 g/m² a day, derived); 60% of the year's grass grows in summer. A new leaf every 4–5 days at peak, ≈ 30 days in midwinter. Rest after grazing: 10–14 days in a cool spring, 40–60 days in a hot dry summer. Ryegrass leaves grow 52.9 µm per °C·h above 0 °C: ≈ 19 mm a day at 15 °C (derived). | 164, 165, 166, 167 | None | missing |
| SW14 | Crops ripen | Growing degree days (base 0 °C), wheat: emergence 125–160, flowering 807–901, mature 1538–1665; barley mature 1269–1522. Spring wheat ≈ 140 days to maturity, winter wheat ≈ 170. Wheat germinates at 4–37 °C, best at 12–25 °C. | 168, 169 | None | missing |
| SW15 | A dead body: the first two days | Core temperature holds for "several hours", then falls 1–1.5 °C/h for ≈ 12 h. Rigor begins in 1–2 h, is complete by 12 h, and passes over the next 24–48 h. Livor shows in 1–3 h and is complete in 6–8 h. Putrefaction shows outside in 12–24 h, and is advanced "as early as 24 to 36 hours in summer". | 170, 171 | A dead body is FLESH that keeps its temperature and shape until something moves it | missing |
| SW16 | A dead body: decomposition | Accumulated degree days explain ≈ 80% of the variation (abstract). Megyesi: ADD = 10^(0.002·TBS² + 1.81), and the error is a factor, not ± 388 ADD (secondary). Vass: PMI (days) = 1285·(decomp/100)/(0.0103·T·RH); full soft-tissue loss ≈ 166 days at 15 °C and 50% RH (derived). Stage bands (warm): fresh 1–10%, bloat 11–35%, decay 36–85%, mostly skeleton 86–100%. Pig carcasses at 24–27 °C: fresh 1 d, bloat 1–2 d, active decay 2–8 d, advanced 5–12 d; all stages done in 12–18 d. Temperate stage lengths: unverified. | 172, 173, 174, 175, 176, 177 | None | missing |
| SW17 | Wood rots in wet ground | Decay needs wood above fibre saturation (≈ 30%); fungi cannot grow at ≤ 20%; soaked wood is too wet. Decay is slow below 10 °C and above 35 °C and stops near 2 °C and 38 °C. Untreated southern pine stakes: ≈ 1 year in Panama, 1.8–3.6 years in Mississippi, Florida and Louisiana, ≈ 6 years in Wisconsin (abstract page). | 178, 179 | None | missing (low) |

**The pattern.** Every row is a timer whose rate depends on temperature (SW2, SW14,
SW16, SW17) and on wetness (SW6, SW10, SW17), with a few named stages. Degree-days
(SW14, SW16) are the simplest form: add max(T − T_base, 0) each hour, and fire the
next stage when the sum passes its threshold. Between two changes of temperature the
time to the next stage has a closed form, so it can be scheduled once (Section 6).

### 2.6 Mechanics: contacts, turning, give, water on bodies (K)

**What the sim has now.** Matter moves in four ways (roadmap 107, list C). Each has
part of the physics. The details below were read from the code; none was run.

- **Lattice** (`support.py`). `_law_support` drops unsupported columns with gravity
  and drag (`_fall_speed`: area = one voxel face per column, `DRAG_CD` 1.1). Columns
  fall straight down: no sideways speed, no spin, no bounce (e = 0, first pass M9).
  `_law_torque` tips a cluster of one material whose centre of mass leaves its
  footprint or a narrow waist. An unowned cluster goes to the particle solver
  (`_topple` → `matter.promote`). A person's body keeps the old per-tick pendulum
  (first pass M5, G11).
- **Rubble and ash piles.** `_relax_slack` carries a rubble cell only from directly
  below, and a vertical stack always holds. So a one-voxel column of rubble or ash
  stands for ever: the angle of repose is 90° (derived).
- **Friction.** `materials.py:FRICTION` 0.40 is used only for the shove limit
  (`hands.py`, `kg * GRAVITY * FRICTION`). The particle solver uses one Coulomb value,
  `mpm/core.py:MPM.__init__` `friction=0.4`, for every pair (first pass M7, G19).
  Nothing on the lattice slides or rolls.
- **Flying bodies** (`bodies.py`). `_launch` lifts a cluster off the lattice with a
  velocity (a jump, a throw). `_law_bodies` integrates velocity with gravity and
  `_drag_a` (frontal area counted along the main axis of motion, `DRAG_CD` 1.1 for
  every shape, no lift). `_fly_pose` adds only an offset: **fliers do not spin**
  (`theta` and `omega` stay 0). The hit test is `self.mat[...] != AIR`, and liquid
  lives in AIR cells, so **a thrown stone flies through a pond with air drag only and
  stops on the bottom** (derived).
- **Particle solver** (`mpm/core.py`, `mpm/matter.py`). Real momentum and spin.
  `_hold_shape` projects each bound cluster onto rigid motion and keeps both
  momenta. Damping per material (`MAT[...]` `damp`) sets the bounce (wood block
  e ≈ 0.26, first pass M9). There is **no air drag** and **no water** in the solver.
  A canopy is bound to its trunk at `rig` 0.15 and has no strength, so "leaves do not
  fall off a falling tree" (comment at `MAT['leaf']`). Free fall into the solver is
  off (`sim.py` `free_fall_to_seam = False`).
- **Water** (`fluids.py`). Volumes in AIR cells. `_law_flow` lets fluid fall and
  spread by level difference. `_law_head` levels each connected pool: it sorts the
  columns above the mean surface and those below it, then pairs them **by rank, not
  by distance** (a `zip` of the two sorted lists). A level change at one end of a pond
  therefore shows at the far end on the same tick (derived). There is no momentum and
  no wave. `_splash` moves 40% of the struck cell sideways and throws drops at fixed
  lattice speeds (first pass L10).
- **Floating.** `support.py:_settle` compares one solid cell's density with the fluid
  cell below it. A denser cell swaps down through the pool; a lighter one rests **on
  top of** the water, fully above the surface. There is no draft (a plank does not
  sit 60% under), no upward push on a sunk light object, no tipping back, and each
  voxel floats or sinks on its own. FLESH (1000 g/L) and WATER (1.0 g/ml) are equal,
  and the test is `sdens > fdens`, so a loose flesh cell rests on water (derived). A
  fall through a pool keeps 0.4 of its drop height per cell (`fallh * 0.4`).
- **Ropes and chains.** No material and no bonds. Roadmap 93 deferred bonds.

| # | Quantity | Real value | Src | Sim value | Verdict |
|---|---|---|---|---|---|
| K1 | Friction of pairs not in the first pass | Oak on oak along the grain: static 0.62, sliding 0.48; across: 0.54 and 0.32 (0.072 greased). Wood on wood wet 0.2. Leather on oak 0.61 / 0.52; leather on metal 0.6 dry, 0.4 wet, 0.2 greasy. Masonry on masonry 0.7–0.9; granite on granite 0.6; limestone 0.75. Iron on iron dry 1.0, greased 0.15–0.20; cast iron sliding dry 0.15, greased 0.07. | 180, 181 | `FRICTION` 0.40 for every pair, used only for the shove limit (`hands.py`); solver friction 0.4 (first pass M7) | ok for game for wood and stone; wrong for iron, greased and wet pairs |
| K2 | Friction on ice | Static: wood on ice 0.05, steel on ice 0.04. Skate blade sliding 0.005–0.006; curling stone 0.017. Ice on ice, slow sliding at −10 °C: 0.43–0.60 (rate-strengthening below ≈ 1e-5 m/s, then falling ≥ 4× by 5 cm/s). A sphere on ice: μ = 1.5e-4·exp(11.5 kJ/mol / RT): ≈ 0.03 at −10 °C, ≈ 0.44 at −100 °C (derived). Ice on ice 0.15–0.76 across 98–263 K is unverified. | 181, 182, 183, 184 | No ice (first pass L16) | missing |
| K3 | Wet stone | Water is an anti-lubricant on hard minerals and a lubricant on sheet minerals (clean surfaces, Horn & Deere): quartz 0.10 dry → 0.23 wet; feldspar 0.11 → 0.76; calcite 0.14 → 0.68; mica 0.43 → 0.23; talc 0.36 → 0.16. | 185 | Wetness never changes friction | missing (low) |
| K4 | Skin on things | Dry skin: μ > 0.2, typically ≈ 0.5, not pressure dependent. Moist or wet skin: typically > 1, rising as pressure falls (abstract). | 186 | One `FRICTION` for a grip | missing |
| K5 | Slipping when walking | Each +0.01 of required friction raises the odds of a slip 1.7×; slip probability 3–95% across the measured range (abstract). The typical required value of 0.17–0.23 is unverified. | 187 | Walking never slips | missing |
| K6 | Bounce of rock on the ground (rockfall) | Normal restitution: paving 0.37–0.42; bedrock 0.33–0.37; talus 0.30–0.33; soft soil 0.28–0.32. Tangential: 0.78–0.92. Quarry faces: limestone 0.32 normal, 0.71 tangential. | 188 | Lattice e = 0 (first pass M9). Solver: the bounce is set by each material's `damp`; the sideways part pays Coulomb friction 0.4. | missing on the lattice; unknown in the solver |
| K7 | Bounce of hard on hard | Steel ball on steel 0.63–0.69 at 1.7–2.4 m/s (falls with speed). Steel ball on granite 0.89–0.92. Granite spheres of 1.3 t: 0.8 ± 0.06 at 0.1–1.5 m/s. | 189, 190 | As K6 | missing on the lattice |
| K8 | Bounce of wood on wood | Identical wood blocks: e = 0.45 (0.41–0.47). A wood piece on a plastic plate: 0.53. | 191 | Solver wood block ≈ 0.26 (first pass M9); lattice 0 | ok for game in the solver; missing on the lattice |
| K9 | Bounce of ice | Frosted ice spheres at 0.06–0.22 m/s: mean e = 0.45 (0–0.84); frost lowers e by up to ≈ 30%. Above a critical speed ice breaks and e falls. | 190 | No ice | missing (low) |
| K10 | Roll, slide or tip | A ball rolls without slipping while tan θ ≤ 3.5·μ; a solid cylinder while tan θ ≤ 3·μ (derived from the textbook condition). A block on a slope tips first if b/h < μ, slides first if b/h > μ (derived statics). Rolling-friction coefficient of boulders (rockfall): rock 0.40–0.45, grass 0.45–0.65 (bigger blocks lower), soft debris 0.70–0.80, mud 0.85. Sliding friction of rock on slopes 0.56–0.58. | 192, 193, 185 | Lattice: a boulder whose centre of mass is over its footprint never moves on any slope (`_law_torque`; derived). It cannot roll or slide. The solver can roll a promoted cluster. | wrong (lattice) |
| K11 | Angle of repose | Dry sand 34–35°, damp sand 40°, wet sand 45°; gravel 40° (crushed 45°); dry ashes 33°, wet ashes 30°; loose earth 28°; bark and wood refuse 45°; snow 38°. Leaf litter: no source (unverified). | 194, 195 | Rubble and ash stand in vertical columns: 90° (`_relax_slack`; derived) | wrong |
| K12 | Seeds falling | Maple samara: 1.03–1.10 m/s, spinning at ≈ 81 rad/s; autorotation starts after ≈ 1 m of fall. Dandelion: 0.39 m/s. | 196, 197 | No seeds. A light piece falls as in K13. | missing |
| K13 | Leaves and light flat things falling | Paper shapes in air: 0.9–2.1 m/s when they spin or helicopter, 2.4–3.5 m/s when they plummet edge-first. Plates fall steadily below Re ≈ 100; a leaf is at Re ≈ 10³. They flutter, and tumble when I* > 0.2–0.4. Lobed leaves settle 15–30% slower than discs (in water). A tree leaf's speed in air and the spread of leaf fall: unverified. | 198, 199, 200 | A full LEAF voxel falls with the drag of one face: terminal 12.2 m/s at 5 cm (25 g), 17.2 m/s at 10 cm (200 g) (derived from `_fall_speed`; a cell filled to f falls √f as fast). No flutter, no sideways drift. In the solver, leaves have no drag and never part from a tree. | wrong (≈ 10× fast) |
| K14 | Skipping a stone | Best tilt ≈ 20° to the water (Clanet, Hersen & Bocquet, via a re-analysis). Model for a 0.1 kg, 10 cm stone: minimum speed ≈ 1–2 m/s; spin ≫ 14 /s (a few turns a second) keeps the tilt; bounces ≈ V²/(2·g·μ·l): ≈ 6 at 5 m/s, ≈ 17 at 8 m/s, ≈ 38 at 12 m/s; first skip ≈ 2.25 m at 8 m/s. Record 88 skips (secondary). | 201, 202, 203 | A flier treats water as air (hit test `mat != AIR`) and has no tilt or spin. It sinks to the bottom with no water force (derived). | missing |
| K15 | A thrown knife or axe turns | One full turn per ≈ 2 m for a 30 cm knife and ≈ 3.5 m for an axe with a 30 cm handle; the distance per turn depends on the throwing radius, not on the speed. Knife speed 13.9–16.9 m/s. Axe-throwing lines: 12 ft (3.66 m) for a hatchet (≤ 1.36 kg), 15 ft for a big axe. An axe thrown at ≈ 10–13 m/s turns ≈ 3–4 times a second (derived). | 204, 205, 206 | `_fly_pose` adds only an offset; `theta` and `omega` stay 0. A thrown axe arrives in the pose it left the hand. | missing |
| K16 | Does a person float? | Body density with lung air removed: men 1.059, women 1.025 (abstract). With full lungs ≈ 0.98 (floats); breathed out ≈ 1.04 (sinks) (derived, assumed lung volumes). | 207 | FLESH 1000 g/L equals WATER, and `_settle` sinks only a denser cell, so a loose flesh cell rests on the surface (derived). No breath effect, no swimming. | wrong |
| K17 | A person in armour | Field plate armour 30–50 kg (abstract) or 15–25 kg (secondary); the sources disagree. Walking in armour costs 2.1–2.3× the energy. With 20 kg of iron a 70 kg man's mean density is ≈ 1.2 even with full lungs (derived): he sinks. | 208, 209 | Worn armour is a record on the person (`p['worn']`, read by `physiology.py:_blow`) with no mass | missing |
| K18 | Wood in water | Specific gravity (dry mass over volume): white oak 0.60 green, 0.68 seasoned; red oak 0.56 / 0.63; eastern white pine 0.34 / 0.35. Green oak passes 1.0 when its moisture exceeds ≈ 67% (derived): very green oak can sink. | 210 | WOOD 600 g/L. A wood cell rests **on top of** the water with no draft. Held water does not change its density for floating. | wrong (no draft) |
| K19 | A plank floats flat | Box hull: GM = KB + BM − KG, BM = b²/(12·d). A uniform plank of width b, thickness t and density ratio s floats flat if (b/t)² > 6·s·(1 − s) (derived). So every real plank floats flat; a square log rolls corner-up for 0.21 < s < 0.79. | 211 | No righting moment: each cell floats or sinks alone | missing |
| K20 | Drag of a body in water | Towed swimmer: 34 N at 1.2 m/s, 49 N at 1.4, 64 N at 1.6, 91 N at 1.8 m/s; C_d on frontal area 0.61–0.73. Added mass: a sphere carries half its displaced water, a cylinder all of it. | 212, 213 | Fliers: none. Lattice falls through a pool keep 0.4 of their drop height per cell. | missing |
| K21 | A stone dropped into water | A wettable sphere makes no air cavity below a few m/s (none at 5 m/s); a water-repellent one does. The threshold does not depend on the size. | 214 | `_splash` moves 40% of the struck cell sideways and throws drops at fixed lattice speeds (first pass L10) | unknown |
| K22 | Waves and ripples | Deep water c = √(g·λ/2π), group speed c/2; shallow water c = √(g·h): 2.2 m/s at 0.5 m deep, 3.1 m/s at 1 m (derived). Ripples are slowest at 0.23 m/s, at λ ≈ 1.7 cm. | 215, 216 | No waves. `_law_head` pairs high and low columns by rank, so a level change crosses a pond in one tick (derived). | wrong |
| K23 | Rope | Manila break load: 12 mm 1,132 kg; 16 mm 1,969 kg; 20 mm 3,040 kg; 24 mm 4,334 kg. Safe load: factor 12 (table) or ≥ 4 (field rule). An overhand knot cuts strength ≈ 55%; tarred hemp ≈ 80% of untarred; sisal ≈ 80% of manila. Stretch at break and wet strength: unverified. | 217, 218, 219 | No rope, no bonds (roadmap 93) | missing |
| K24 | Chain | Grade 30 proof-coil chain: 3/8 in (10.3 mm) works to 2,800 lb (1,270 kg), breaks at 11,200 lb (49.8 kN); 1/2 in breaks at 85.8 kN. Break = 4 × working load; proof = 2 ×. | 220 | No chain | missing (low) |

The first pass already covers: drag of a person in air and of an arrow (M4, M8), the
pole topple (M6), the felling hinge and a falling trunk (S23, S24), wood strength
(S1–S5), the bounce of a dropped block (M9) and the friction table for wood and stone
(M7). The crown of a felled tree has no row of its own: no source gave a number for
how twigs give on landing. The physics is K13 (light pieces) plus bonds that bend
and break (G23).

## 3. The biggest gaps, ranked by what a player notices first

The order uses the same two questions as the first pass: how often does a player meet
the gap, and how plainly does it look wrong? The decision models read only named facts
and events, so each gap also says which facts and events the fix would give a
character. Event names follow the style of `docs/sim-mind-design.md` (`ignites`,
`breaks`). Where a fix depends on a first-pass gap, the gap says so.

### G20. Nobody gets tired, sleepy, hungry or thirsty

- **What the player sees.** A guard stands watch for three days without sleep. A
  traveller crosses a desert without a drink. The model asks "how does Tom feel?" and
  the sim has no true answer.
- **Cause.** The person dict (`persons.py:add_person`) has no store but blood O2,
  smoke, burn and hurt. Sleep is a flag that a scene sets (`p['asleep']`). Rows BC1–BC5,
  BC10–BC12.
- **Fix.** Add `physiology.py:_law_clocks`, run once per game second (every 40
  ticks), with one number per clock: sleep pressure S and circadian phase C (two-process
  model, τ 18.2 h awake, 4.2 h asleep), body water (sweat 0.5–2 L/h by work and heat),
  energy (≈ 1400 kcal/d at rest, ≈ 2400 active). Each clock has a closed form between
  events, so a quiet night is one step (Section 6). Add the acts sleep, eat and drink.
- **New facts and events.** Facts: "Tom is tired" (S high, or > 16 h awake), "Tom is
  thirsty" (deficit > 1% of body weight), "Tom is weak with thirst" (> 2%), "Tom is
  hungry" (12–24 h without food). Events: `falls_asleep`, `wakes`, `drinks`, `eats`.
  Sleep loss feeds the reaction delay of first pass G2 (17 h awake ≈ BAC 0.05%).
- **Tests.** BC-T1, BC-T2, BC-T3, BC-T8, BC-T9, BC-T10, BC-T18.

### G21. There is no wind

- **What the player sees.** A storm scene where nothing moves. Smoke from a burning
  barn drifts at walking pace whatever the weather. No candle ever gutters. A shout
  cannot be carried off, and a door left open makes no draught.
- **Cause.** `air.py:_law_air` has no source but fire and warmth, and every value
  decays (×0.965 per tick, half-life ≈ 0.5 s). Advection is capped at 0.4 voxel per
  tick (0.8 m/s at 5 cm). Rows W1–W5, W16, W20, W21.
- **Fix.** In `air.py`, add an outdoor boundary wind: a mean speed and direction at
  10 m, a roughness z0 per ground type, the log profile below 10 m, and a gust series
  with a 3 s scale and a gust factor near 1.5–1.7. Hold the open-sky edge cells at this
  wind instead of letting it decay. Move smoke (and the new gases, G25) with a
  semi-Lagrangian step on the air cells, so speed is not capped by the cell size.
  Outdoors, where nothing reads the wind, compute it on demand from the formula.
- **New facts and events.** Facts: "the wind is strong from the north", "a draught
  comes under the door". Events: `gusts`, `blows_out` (a candle; the threshold needs a
  source, W8).
- **Tests.** W-T1, W-T2, W-T3, W-T4, W-T12, W-T13, W-T16.

### G22. Thrown things never turn, and fly through water

- **What the player sees.** A thrown axe flies flat and hits with whatever face it left
  the hand with. A knife never tumbles. A skimmed stone drops straight through the
  pond to the bottom, as if the water were air.
- **Cause.** `bodies.py:_fly_pose` adds only an offset; a flier's `theta` and `omega`
  stay 0. `_law_bodies` stops a flier only where `mat != AIR`, and liquid lives in AIR
  cells, so water is invisible to it. Rows K14, K15, K20.
- **Fix.** Roadmap 107 list C, step 1: make a flier a rigid object with orientation,
  spin and inertia. At release, give it the spin the swing gives it: the distance per
  turn is set by the throwing radius (≈ 3.5 m for an axe). In `_law_bodies`, let a
  flier meet water: a force ½ρv²·C_d on the wetted face, with a lift part that depends
  on the tilt, plus buoyancy. A flat stone at ≈ 20° tilt with spin then skips; a round
  one sinks. Judge a hit by the face that arrives (edge or flat).
- **New facts and events.** "The axe struck handle-first", "the stone skipped four
  times". Events `turns` (in flight), `skips`, `splashes`, `sinks`.
- **Tests.** K-T5, K-T6, KW-T3, KW-T4.

### G23. A felled tree's leaves drop like bricks, and nothing flutters

- **What the player sees.** A tree is felled and its canopy comes down as a block or
  rains straight down column by column. An autumn leaf falls as fast as a stone.
- **Cause.** A LEAF voxel is a dense lump with the drag of one face: 12–17 m/s
  terminal (row K13). In the solver the canopy is bound to the trunk with no strength
  and no air drag, so leaves never part and never flutter (`mpm/core.py:MAT['leaf']`).
  The likely cause of the bulk drop is in roadmap 107 list C. Rows K12, K13; first
  pass S24.
- **Fix.** Roadmap 107 list C, step 2: bonds. Make the tree a set of rigid pieces
  (trunk, branches, twigs) joined by bonds that bend and break (the break strength from
  first pass S3). Make leaves light pieces with high drag and a flutter term, released
  when their twig bond breaks. A leaf should fall at ≈ 1–2 m/s and drift with the wind
  (G21).
- **New facts and events.** "Leaves are falling", "the crown hit the ground and
  branches snapped". Events `snaps` (a branch), `sheds`.
- **Tests.** K-T7, K-T8, W-T14.

### G24. Rubble and ash stand in columns; boulders never roll; nothing slides

- **What the player sees.** A burned house leaves towers of ash one voxel wide. A
  boulder sits on a steep slope. A crate on a tilted plank never slides off.
- **Cause.** `support.py:_relax_slack` carries rubble from directly below, and a
  vertical stack always holds, so the angle of repose is 90°. The lattice has no
  sliding or rolling: `_law_torque` moves a cluster only when its centre of mass
  leaves its footprint. Rows K10, K11.
- **Fix.** In `support.py`, let a rubble or granular cell slip sideways and down when
  the slope of its pile is above the material's angle of repose (sand 34°, gravel 40°,
  ash 30–33°). Hand a boulder or a crate on a slope to the rigid-object level (G22) and
  let the roll/slide/tip rules of K10 decide, with rolling resistance by ground type.
- **New facts and events.** "The boulder is rolling down the hill", "the crate slid
  off". Events `rolls`, `slides`, `slumps` (a pile).
- **Tests.** K-T1, K-T2, K-T3.

### G25. There is no smell, and every gas is soot

- **What the player sees.** A cellar of rotten eggs smells of nothing. A brewer goes
  down to a cellar full of CO2 and feels nothing. A gas that is heavier than air rises
  like smoke. Acid, boiling and burning all make the same grey "smoke".
- **Cause.** The air carries only `o2` and `smoke` (`air.py`). `_law_smoke` makes smoke
  rise 0.4 of a cell per tick at any temperature. There is no smell sense
  (`senses.py`); the only gas percept is `chokes` (load > 0.05, `will.py`). The cutoff
  `SMOKE_STILL` (8e-6 g/L at 5 cm) is far above a nose's threshold for H2S
  (7e-10 to 4e-7 g/L). Rows GS7–GS14, GS18–GS20.
- **Fix.** In `air.py`, give each air cell a few species (CO, CO2, water vapour, H2,
  H2S, SO2, plus soot) at 1/64 of the voxel count. Let buoyancy come from each
  parcel's density (temperature and composition), not from a fixed rise rule. Give
  each species its own "still" cutoff, set from its odour threshold. Add
  `senses.py:_law_smell`: read the species at the nose once a second, compare with the
  odour thresholds of GS10, and fade a smell that stays (H2S: 2–15 min above
  100 ppm). Put products in the reaction table (G35) so acid on chalk makes CO2 and
  sulfur makes SO2.
- **New facts and events.** "The cellar smells of rotten eggs", "a sharp smell of
  burning sulfur", "the air feels close" (CO2 3–5%). Event `smells` with a source
  direction. CO, CO2 and H2 give no smell, which is what makes them dangerous.
- **Tests.** GS-T4, GS-T5, GS-T6, GS-T7, GS-T8, GS-T10, GS-T11, GS-T12.

### G26. Floating is per voxel: planks sit on top, people neither swim nor sink

- **What the player sees.** A plank rests fully above the water like a tray. A man in
  plate armour floats like anyone else. A sunk log stays on the bottom. A dropped
  stone makes no ripple; a pond levels across its whole width in one tick.
- **Cause.** `support.py:_settle` compares one cell's density with the fluid cell
  below it: denser sinks, lighter rests on top. There is no draft, no upward push on a
  sunk light thing, and no righting moment. Armour has no mass (`p['worn']`).
  `fluids.py:_law_head` pairs columns by rank, not by distance, and water has no
  momentum. Rows K16–K22.
- **Fix.** Roadmap 107 list C, step 3. For each rigid or bonded object, sum buoyancy
  over its submerged cells (ρ·g·V), apply it at the centre of buoyancy (this gives the
  draft and the righting moment of K19), and add water drag (C_d ≈ 0.6–0.7 for a body,
  plus added mass). Give a person a breath state (full lungs float, empty sink, K16)
  and give worn armour its mass. Add a surface height field for ponds with
  c = √(g·h), and let objects push it (waves) and feel it.
- **New facts and events.** "The plank floats", "Tom is treading water", "the man in
  mail went under". Events `floats`, `sinks`, `surfaces`, `ripples`.
- **Tests.** KW-T1, KW-T2, KW-T5 to KW-T9.

### G27. A closed-room fire or a brazier makes no CO

- **What the player sees.** A family sleeps beside a charcoal brazier in a shut room
  and wakes up fine. A smouldering house fire kills nobody who stays out of the smoke.
- **Cause.** `_law_burn` turns burnt mass into soot (wood 0.6 g/g, char 0.05 g/g) and
  deletes the rest. Its products do not change when the fire runs short of oxygen.
  Rows GS15–GS17, R5.
- **Fix.** Make burning a row of the reaction table (G35) with products CO2, H2O, CO
  and soot. Take the CO share from the local O2 supply: ≈ 0.004–0.08 g/g well
  ventilated, ≈ 0.2 g/g when starved; a glowing ember makes mostly CO at its surface
  (R5). Dose people with the Purser FED (GS1–GS3), which also closes first pass G9.
- **New facts and events.** "Anna has a headache" (COHb ≈ 20%), "the room feels
  stuffy". Events `poisoned` (cause: CO), `slumps`.
- **Tests.** GS-T1, GS-T2, GS-T3, GS-T9.

### G28. Holding and carrying cost nothing

- **What the player sees.** A man holds a door shut against a crowd all night. A
  villager carries a 40 kg sack at arm's length for an hour. A sprinter never slows.
- **Cause.** `BODY['strength_N']`, `wrist_Nm` and `arm_Nm` are limits with no clock
  (`materials.py:BODY`, `hands.py`). There is no energy or fatigue store (first pass
  B26). Rows BC6–BC9.
- **Fix.** In `physiology.py`, give each limb a reserve that drains at the
  Frey-Law rate (endurance time 21.92·f^−1.98 s, f = load over maximum) and refills
  with the PCr half-times (≈ 21 s fast, > 170 s slow). `hands.py` reads the reserve:
  when it is empty the grip opens. Whole-body work drains the energy clock of G20 at
  ≤ 30% of maximum for an 8 h day.
- **New facts and events.** "Tom's arms are shaking", "Tom is out of breath". Events
  `drops` (cause: tired), `rests`.
- **Tests.** BC-T5, BC-T6, BC-T7, W-T6.

### G29. Wounds never heal

- **What the player sees.** A man cut in a brawl a month ago still carries the full
  wound. Nobody falls ill from a dirty wound.
- **Cause.** `hurt` and `burn` only climb; the `_wound` docstring says so
  (`physiology.py`). Clotting stops the bleeding but not the wound. Rows BC20, BC21.
- **Fix.** In `physiology.py`, give each wound a healing fraction that follows the
  phases of BC20 (3% of strength at 1 week, 20% at 3 weeks, 80% at 3 months) and lowers
  `hurt`. Give a dirty wound an infection timer (tetanus 3–21 days; gas gangrene
  < 24 h), drawn from the wound's facts, not from a die.
- **New facts and events.** "The cut is scabbed over", "the wound is red and hot",
  "Tom has lockjaw". Events `heals`, `festers`. New act: bandage.
- **Tests.** BC-T16, BC-T17.

### G30. Drink and poison have no clock

- **What the player sees.** A tavern brawl between sober men after eight pints. A
  poisoned cup does nothing.
- **Cause.** No compartment for a substance in the body. Rows BC17–BC19; first pass
  P23.
- **Fix.** In `physiology.py`, one compartment per substance: absorption ka,
  elimination ke (alcohol: zero order, 0.010–0.022 %/h; paracetamol: t½ 2–2.5 h),
  volume V per kg. Map BAC to the effects of BC18 (reaction delay, balance, sleep) and a
  poison level to its organ events.
- **New facts and events.** "Tom is drunk", "Tom is staggering", "Anna feels sick".
  Events `drinks`, `passes_out`, `poisoned`.
- **Tests.** BC-T14, BC-T15.

### G31. Fire ignores the wind

- **What the player sees.** A grass fire spreads as fast upwind as downwind. A
  burning barn never sets fire to the barn downwind first.
- **Cause.** `_law_burn` has no wind term; fire spreads only by contact conduction and
  neighbour radiation (first pass G4, G7). Rows W9–W11.
- **Fix.** After first pass G4 and G7 land: tilt the plume and the radiant view
  factor with the wind, and preheat fuel downwind. Check the result against the
  CSIRO grass model (0.75–2.28 m/s at 10–30 km/h) and the 10% forest rule. A per-fuel
  wind factor (Rothermel φ_w) is the fallback if the tilted plume does not reach it.
- **New facts and events.** "The fire is running before the wind". Event
  `fire_spreads` with a direction.
- **Tests.** W-T7, W-T8.

### G32. Cold and heat have no body clock

- **What the player sees.** A man falls through the ice and climbs out unharmed half
  an hour later. A smith works all day in a furnace room without distress.
- **Cause.** No core temperature and no metabolism (first pass B19, G15). Flesh voxels
  take the temperature of what they touch. Rows BC13–BC16, W12, W13.
- **Fix.** In `physiology.py`, one core temperature per person with a heat balance:
  metabolism (≈ 100 W at rest, more with work), skin loss through h_c = 12.1·√v (W12)
  in air and the much larger loss in water (core −3.7 °C/h in 5 °C water), and sweat
  (G20). Thresholds: hypothermia stages of BC14; collapse risk from 39.2 °C, heat stroke
  above 40.5 °C. Cold shock and loss of hand use in cold water follow 1-10-1.
- **New facts and events.** "Tom is shivering", "Tom's hands are numb", "Anna is
  overheated". Events `cold_shock`, `collapses` (cause: heat or cold).
- **Tests.** BC-T11, BC-T12, BC-T13, W-T9, W-T10.

### G33. The wind does not push, cool or sway

- **What the player sees.** A man stands upright on a cliff edge in a gale. A sail or
  a cloak does nothing. Trees stand still in a storm and never break.
- **Cause.** No wind force on bodies (`motion.py:_overbalanced` reads weight and load
  only); flat `LEAK` cooling; rigid trees with no wind load. Rows W6, W7, W12, W14,
  W15, W18, W19.
- **Fix.** In `motion.py`, add the drag force ½ρv²·C_d·A (A from the body's own cells
  facing the wind, C_d ≈ 1.0–1.3) to the balance test, so gusts above ≈ 15 m/s can push
  a person over. Use the air speed in the face heat loss of first pass G5 (h_c grows
  with √v). Load bonded trees (G23) with the same drag, and break trunks near 42 m/s.
  Lift loose light pieces (leaves, paper, ash) above their Beaufort thresholds.
- **New facts and events.** "Tom staggers in the gale", "the cloak is torn from his
  hands". Events `staggers`, `blown_over`, `uprooted`.
- **Tests.** W-T5, W-T9, W-T11, W-T14, W-T15.

### G34. Food, drink and bodies never change

- **What the player sees.** Bread baked a month ago is fresh. A corpse in the cellar
  stays as it fell. Ale never ferments, and a cask of wine is wine from the start.
- **Cause.** No food or drink kinds and no slow timers. Rows SW1–SW9, SW15, SW16.
- **Fix.** Add a slow-world scheduler (a heap of events in `things.py`): each
  perishable object holds its stage and the game time of its next stage. Compute that
  time from the square-root law (SW2) or from degree-days (SW14, SW16) at the object's
  temperature and wetness. Reschedule only when the temperature or wetness moves out
  of a band. Stages write facts and can release gases (G25): CO2 from fermenting, the
  smell of decay.
- **New facts and events.** "The bread is mouldy", "the milk has turned", "the ale is
  ready", "the cellar smells of death". Events `spoils`, `ferments`, `rots`.
- **Tests.** SW-T1 to SW-T7, SW-T10 to SW-T15.

### G35. Contact chemistry is acid-only and blind to temperature

- **What the player sees.** A sword left in the rain never rusts. Quicklime and water
  do nothing. Black powder does not exist. Acid eats wood at the same rate when frozen
  and when boiling.
- **Cause.** `materials.py:REACTIONS` holds only (acid, solid) pairs with a fixed rate,
  and `_law_acid` turns every product into smoke. Burning is a separate law, linear in
  T − ign. Rows R1–R16; first pass G8.
- **Fix.** Replace `REACTIONS`, `_law_acid` and the rate part of `_law_burn` with one
  table and one law (a new `chemistry.py`): each row has reactants and how they meet,
  products in g per g (solids, fluids and gas species), an Arrhenius pair (or a
  measured flux plus a Q10), and the heat. Run fast rows on touching faces; schedule
  slow rows (rust) as events. Fold in the first-pass acid fixes (concentration
  thresholds).
- **New facts and events.** "The blade is rusting", "the lime is steaming hot", "the
  powder flashes". Events `rusts`, `reacts`, `deflagrates`.
- **Tests.** R-T4 to R-T15.

### G36. Wood never dries, and never chars without a flame

- **What the player sees.** A wet log by the hearth stays wet for ever. A beam above a
  stove for a winter is as good as new; a shelf at 240 °C never scorches.
- **Cause.** Water held in a solid leaves only by boiling (`fluids.py:_law_flow`,
  `heat.py:_law_boil`; first pass L15, L19). Nothing changes wood below its 300 °C
  ignition (first pass F2). Rows R1–R3, SW10, SW11.
- **Fix.** Add pyrolysis as a reaction row (Thurner kinetics, char ≈ 30%) that runs
  without oxygen and below ignition. Add evaporation from held water through exposed
  faces (first pass L15) with the fibre-saturation and 20% limits of SW10, slow enough
  to schedule as an event.
- **New facts and events.** "The beam is scorched", "the firewood is seasoned". Events
  `chars`, `dries`.
- **Tests.** R-T1, R-T2, R-T3, SW-T8, SW-T9.

### G37. Smaller contact gaps: one friction for all, no slipping, no rope

- One friction value for every pair, dry or wet (K1–K4). Iron on iron should grip at
  ≈ 1.0 dry and 0.15–0.2 greased; wood on ice at 0.05; wet skin more than dry.
- Walking never slips (K5). An icy step or a greasy floor should give a `slips` event
  when the friction needed passes the friction there.
- Lattice landings never bounce (first pass M9). Rock on ground has a normal
  restitution of 0.28–0.42 and keeps most of its sideways speed (K6–K9).
- No rope or chain (K23, K24). With bonds (G23) they become a chain of pieces with a
  break load: 12 mm manila ≈ 1.1 t, halved by a knot.
- **New facts and events.** "The floor is slick", "the rope parted". Events `slips`,
  `parts`.
- **Tests.** K-T4, K-T9, K-T10, K-T11.

## 4. Reference tests

These tests become seven new files in later build waves. The rules of the first pass
(its Section 4) apply: run at `voxel_cm` 5, and at 10 where the test says so; convert
ticks with `TICK_S`; mark long tests slow; keep a test whose law does not exist yet as
an expected failure. Priorities: **high** = a player meets it in the first scenes;
**medium** = in a specific scene; **low** = detail or long-run behaviour.

Two more rules apply here:

- A test of a slow process (hours to years) must run with the skip-ahead of Section 6,
  never tick by tick. BC-T18 and SW-T15 check that the skip changes nothing.
- A test that reads a fact or an event (`tired`, `smells`, `rolls`) checks the name and
  the cause the event carries, not only the number behind it.

### 4.1 `tests/test_realism_wind.py`

1. **W-T1 Wind against height** — high
   - Scene: open sky, flat ground with z0 = 0.03 m (grass), no buildings. Set the
     outdoor wind to 10 m/s at 10 m. Run 2,400 ticks (60 s).
   - Measure: mean air speed in m/s at 1.0, 1.75 and 2.0 m, far from the edges.
   - Expected: 6.0, 7.0 and 7.2 m/s ± 10% (log law). The same at 5 and 10 cm.
   - Source: 1, 2. Today: no wind (needs an outdoor wind law).
2. **W-T2 Gusts** — high
   - Scene: as W-T1, 10 min (24,000 ticks; slow).
   - Measure: the highest 3 s mean at 1.75 m, divided by the 10 min mean; how long
     each gust lasts.
   - Expected: gust factor 1.5–1.8 (1.66 inland). Gusts last a few seconds, a wind run
     of tens of metres.
   - Source: 1, 3. Today: missing.
3. **W-T3 Smoke rides the wind** — high
   - Scene: a smoky fire outdoors in a 5 m/s wind at 2 m height. Run 400 ticks.
   - Measure: speed of the smoke front 5–20 m downwind.
   - Expected: 4–6 m/s (within 20% of the local wind). The same at 5 and 10 cm.
   - Source: — (advection; W20). Today: capped at 0.8 m/s at 5 cm and 1.6 m/s at 10 cm
     (derived).
4. **W-T4 Candle in a draught** — medium
   - Scene: a candle-sized flame (≈ 80 W, first pass F24) in steady air at 0.1, 0.3,
     1, 3 and 6 m/s. Then a torch.
   - Measure: whether the flame survives 60 s; the `blows_out` event.
   - Expected: the candle survives 0.1–0.3 m/s and leans; it goes out somewhere between
     0.5 and 5 m/s. A torch survives longer than a candle. The band is wide because
     no source was opened (W8): find one before making the test strict.
   - Source: unverified. Today: no candle.
5. **W-T5 Standing in a gale** — high
   - Scene: a person standing on open ground. Steady wind at 1.75 m of 5, 10, 15 and
     20 m/s, then the same means with gusts (W3).
   - Measure: wind force on the body; whether the person can stand and walk into it;
     the events `staggers` and `blown_over`.
   - Expected: force 30–60 N at 10 m/s facing or side-on (C_d·A 0.46–0.84 m²). Walking is
     unsteady above ≈ 7.6 m/s. Balance is lost in gusts of ≈ 12–15 m/s for some people.
     Blown over in gusts of ≥ 15–23 m/s; never at a steady 10 m/s.
   - Source: 9, 7, 8. Today: no wind force.
6. **W-T6 Walking into the wind costs breath** — low
   - Scene: a person walks at 1.25 m/s into a wind of 18.5 m/s.
   - Measure: energy or O2 use against calm air.
   - Expected: ≈ 4× (0.8 → 3.1 L/min), rising with the square of the wind speed.
   - Source: 11 (abstract). Today: no energy clock (needs G20).
7. **W-T7 Grass fire runs downwind** — high
   - Scene: 10 cm voxels; a strip of cured grass 4 m wide and 40 m long (needs a grass
     fuel); 10 m wind 10, 20 and 30 km/h along the strip; light one end.
   - Measure: steady forward rate of spread.
   - Expected: 0.75, 1.56 and 2.28 m/s, each within × 1.5 (the model's own error is
     ≈ 50%). Below 5 km/h the head fire does not spread steadily.
   - Source: 13, 14. Today: no wind term; the fire clock is also wrong
     (first pass G4).
8. **W-T8 Forest fire 10% rule** — low
   - Scene: a dry wood of small trees, 10 m wind 20 km/h.
   - Measure: forward rate of spread.
   - Expected: ≈ 0.56 m/s (10% of the wind), within × 2.
   - Source: 15 (abstract). Today: missing.
9. **W-T9 Wind cools faster** — medium
   - Scene: a warm body (37 °C core, needs first pass G15) in 0 °C air at 0.5 m/s and
     at 5 m/s.
   - Measure: convective heat loss per m² of skin.
   - Expected: the 5 m/s loss is √10 ≈ 3.2× the 0.5 m/s loss (pass 2.5–4), with
     h_c = 12.1·√v.
   - Source: 16. Today: `LEAK` ignores air speed.
10. **W-T10 Wind chill** — low
    - Scene: bare skin at −10 °C air with 30 km/h wind; then air giving a wind chill of
      −40.
    - Measure: time to the `frostbite` fact.
    - Expected: no frostbite within 30 min at a wind chill of −19.5; frostbite in
      5–10 min at −40.
    - Source: 21, 19. Today: missing.
11. **W-T11 Leaves, paper and dust lift** — medium
    - Scene: loose dry leaves, paper and fine sand (100 µm) on open ground. Ramp the
      10 m wind from 1 to 10 m/s over 5 min.
    - Measure: the wind at which each first moves.
    - Expected: leaves at 3.4–5.4 m/s; paper and dust at 5.5–7.9 m/s (at 1.75 m:
      3.9–5.5); sand saltation near 8 m/s.
    - Source: 4, 7, 22, 23. Today: nothing lifts.
12. **W-T12 Draught through a doorway** — medium
    - Scene: a room at 30 °C, outside at 20 °C, one door 0.9 × 2.0 m open. Run 2,400
      ticks.
    - Measure: volume flow each way; air speed at the top and bottom of the door; air
      speed in the middle of the room.
    - Expected: 0.31 m³/s each way ± 30%; ≈ 0.5 m/s near the top and bottom edges;
      < 0.1 m/s in the room. The same at 5 and 10 cm.
    - Source: 24. Today: unknown (not measured).
13. **W-T13 A person feels a draught** — low
    - Scene: a seated person at 20 °C in air at 0.05, 0.15 and 0.25 m/s.
    - Measure: the `draught` percept.
    - Expected: none at 0.05 m/s; reported by some at 0.15 (DR ≈ 18%); by more at 0.25
      (DR ≈ 35%).
    - Source: 16. Today: no such sense.
14. **W-T14 A tree sways and comes back** — medium
    - Scene: one tree, trunk 10 cm across and 10 m tall, in calm air. One 5 s gust of
      10 m/s, then calm.
    - Measure: sway frequency; decay of the sway; final lean.
    - Expected: ≈ 0.39 Hz (within × 2; the formula is secondary and fruit trees measure
      0.63–1.5 Hz). The sway dies away. The tree ends upright. (Roadmap 107 list C.)
    - Source: 25. Today: no wind load; needs bonds.
15. **W-T15 A storm breaks trees** — low
    - Scene: a stand of trees of mixed size, 1 s maximum winds of 20, 30, 42 and 50 m/s.
    - Measure: fraction of trunks broken.
    - Expected: none at 20 m/s; less than half at 30; about half near 42; most at 50.
      Tree size changes the threshold little (+9% for twice the height).
    - Source: 26. Today: missing.
16. **W-T16 The wind keeps blowing, and stays outside** — high
    - Scene: W-T1, with a closed stone hut on the ground. No fire. Run 2,400 ticks.
    - Measure: mean wind at 2 m outside; mean air speed inside the hut.
    - Expected: outside within 10% of the set wind for the whole run; inside
      < 0.1 m/s (only leaks).
    - Source: 3, 24. Today: a wind with no source halves in ≈ 0.5 s.

### 4.2 `tests/test_realism_gases.py`

1. **GS-T1 A brazier in a closed room** — high
   - Scene: a sealed room of ≈ 19 m³ with 800 g of glowing CHAR. Run 2 h of game time
     (skip ahead when nothing else moves).
   - Measure: CO in ppm at 1.5 m; COHb of a person in the room.
   - Expected: CO above 3,000 ppm within 2 h. A person in the room passes FED_CO = 1
     (incapacitation) before the end. With a leaky door: 750–1,100 ppm.
   - Source: 51, 27. Today: no CO; char makes 0.05 g of soot per g.
2. **GS-T2 CO dose clock** — high
   - Scene: a person held in uniform air with 1,000 ppm CO and 0% CO2; then 1,000 ppm
     CO with 5% CO2.
   - Measure: time to incapacitation (FED = 1).
   - Expected: 28–35 min (± 35%); 10–14 min with 5% CO2.
   - Source: 27, 28, 29. Today: no CO (first pass G9).
3. **GS-T3 Low oxygen clock between the first-pass levels** — medium
   - Scene: a person in uniform air at 15% O2, then at 12% O2, no CO. (First pass B-T2
     already covers 14%, 10% and 5%.)
   - Measure: time to faint.
   - Expected: ≈ 140 min at 15% and ≈ 28 min at 12% (± 35%), from the FED_O2 term.
   - Source: 28, 27. Today: blood O2 follows the air with a 1.25 s time
     constant (first pass B21).
4. **GS-T4 CO2 fills a cellar** — high
   - Scene: a cellar 2 m deep under a room, joined by an open hatch. Release 2 kg of
     CO2 at the cellar floor (or let a fermenting vat make it, SW-T6). No heat. Run
     10 min.
   - Measure: CO2 vol % at 0.2 m above the cellar floor and at 1.5 m in the room above.
   - Expected: the cellar floor holds ≥ 5× the room's head-height level. Nothing climbs
     out by a "smoke rises" rule. A person who climbs down smells nothing but gets the
     warning signs of GS7 (deeper breathing at 3–5%).
   - Source: 43, 42, 31. Today: no CO2; smoke would rise.
5. **GS-T5 Hydrogen goes up** — medium
   - Scene: release 10 g of H2 at the floor of a closed room 3 m high.
   - Measure: H2 by height after 30 s.
   - Expected: most of it in the top 0.5 m ("would briefly collect on the ceiling").
   - Source: 47, 43. Today: no H2.
6. **GS-T6 Rotten eggs in the cellar** — high
   - Scene: a 30 m³ cellar with a source of H2S at 1 mg/min. A person at 3 m. Run
     until the local level passes 150 ppm (raise the source if needed).
   - Measure: the tick of the first `smells` event and the local ppm then; when the
     smell fades.
   - Expected: first smelled between 0.0005 and 0.3 ppm. At > 100 ppm the smell fades
     within 2–15 min (olfactory fatigue), so the person "smells nothing" while the
     level is dangerous.
   - Source: 40, 39, 48. Today: no smell.
7. **GS-T7 Cold smoke does not climb** — medium
   - Scene: a puff of smoke at 20 °C in the middle of a still room; then the same puff
     at 300 °C.
   - Measure: ratio of smoke in the top 0.5 m to the bottom 0.5 m after 60 s.
   - Expected: cold puff: ratio below 2 (it spreads, it does not climb). Hot puff:
     ratio above 5.
   - Source: 41 (ideal gas, derived). Today: smoke rises 0.4 cell per
     tick at any temperature.
8. **GS-T8 A room mixes in minutes, not seconds** — medium
   - Scene: a closed room 3.5 × 3.7 × 2.4 m at one temperature. A 1 g tracer puff in one
     corner.
   - Measure: time until the spread over 9 sample points falls below 10% of the mean.
   - Expected: 7–100 min (natural convection; ≈ 80–100 min when isothermal). Results at
     5 and 10 cm agree within × 1.5.
   - Source: 54, 55. Today: ≈ 10 min at 5 cm and ≈ 2.5 min at 10 cm
     (derived from the O2 mixing rate).
9. **GS-T9 A starved fire makes CO** — high
   - Scene: `house_fire` with the door shut until the hot layer falls below 15% O2;
     then the door open.
   - Measure: CO yield (g CO made per g fuel burned) in each phase.
   - Expected: shut (fuel-rich): 0.14–0.3 g/g. Open and well ventilated: 0.004–0.1 g/g.
   - Source: 50, 49. Today: no CO.
10. **GS-T10 Gas that burns** — low
    - Scene: a closed cellar with H2S at 3% and at 5% by volume; then CO at 10% and at
      15%. A torch at the far end.
    - Measure: whether a flame runs through the gas.
    - Expected: H2S: no flame at 3%, flame at 5%. CO: no flame at 10%, flame at 15%.
    - Source: 42, 43. Today: no gas burns.
11. **GS-T11 The books close for gases** — high
    - Scene: burn 100 g of wood in a sealed stone box with enough air.
    - Measure: mass of all products (CO2, H2O, CO, soot, ash, char) against fuel burned
      plus O2 used.
    - Expected: equal within 1%. CO2 1.2–1.9 g per g of wood.
    - Source: 49, 64. Today: 1.73 g per g of fuel leaves the books
      (derived).
12. **GS-T12 Steam rises from a boiling pot** — medium
    - Scene: boil 100 ml of water in an open pot in a closed room.
    - Measure: steam volume made; where it goes; condensation on cold walls.
    - Expected: ≈ 167 L of steam at 100 °C (1 g → 1.673 L). It rises (density ≈ 0.5 of
      room air) and condenses on cold surfaces, giving back 2,256 J/g.
    - Source: 44. Today: boiled water is deleted (first pass G12).

### 4.3 `tests/test_realism_reactions.py`

1. **R-T1 Wood bakes to char without flame** — high
   - Scene: a 5 cm WOOD voxel held at 350 °C (thermostat) in a box with no oxygen. Run
     10 min (24,000 ticks).
   - Measure: time until half of the wood has reacted; mass left as CHAR at the end.
   - Expected: half in 2.7–6 min (4.0 min ± × 1.5); char 25–35% of the wood.
   - Source: 57, 58. Today: no change without oxygen; with air it
     burns half its mass in ≈ 3.7 s (derived).
2. **R-T2 A shelf near a stove for months** — low
   - Scene: a wood shelf held at 150 °C for 165 game days (skip ahead).
   - Measure: mass lost; material.
   - Expected: 40–90% lost; the shelf has become char. At 107 °C over the same time it
     only darkens.
   - Source: 60. Today: nothing happens below 300 °C.
3. **R-T3 Char yield against temperature** — medium
   - Scene: wood voxels baked to completion at 300, 500 and 700 °C without oxygen.
   - Measure: char mass as a fraction of the dry wood.
   - Expected: 42%, 33% and 30% (± 20%).
   - Source: 61. Today: 18% at any temperature.
4. **R-T4 An ember lasts** — medium
   - Scene: one 5 cm CHAR voxel held at 400 °C in fresh air.
   - Measure: time to half of its mass.
   - Expected: at least 75 s (powder kinetics give 110 s; a lump is slower because O2
     must reach it).
   - Source: 62. Today: ≈ 4.2 s (derived).
5. **R-T5 What an ember makes** — medium
   - Scene: as R-T4 at 400 °C and at 800 °C.
   - Measure: CO and CO2 made; heat released per g of carbon.
   - Expected: CO2/CO by mole ≈ 1.4 at 400 °C and ≈ 0.25 at 800 °C (within × 1.5).
     Heat 32.8 kJ/g C if all goes to CO2, 9.2 kJ/g C to CO.
   - Source: 66, 64, 65. Today: no CO or CO2.
6. **R-T6 A wet blade rusts** — high
   - Scene: an iron blade wetted with water at 20 °C, left in air.
   - Measure: the tick of the first `rusts` event (a visible colour change); mass lost
     per cm² per day.
   - Expected: first rust after ≈ 0.5 h (not before 20 min). Mass loss below
     1 mg/(cm²·day) (fresh-water rates are ≈ 0.1 mg/(cm²·day), derived). A dry blade does
     not rust in a day.
   - Source: 71, 67. Today: no rust.
7. **R-T7 Rust in a year** — low
   - Scene: iron plates in fresh water and in sea water for 1 game year (skip ahead).
   - Measure: thickness lost per surface.
   - Expected: fresh water 0.025–0.05 mm; sea water 0.035–0.2 mm.
   - Source: 67, 68. Today: no rust.
8. **R-T8 Quicklime and water** — medium
   - Scene: 1 kg of CaO in a wooden bucket. Case A: add 3 L of water. Case B: add only
     0.35 L.
   - Measure: time to 60 °C; peak temperature; product mass; the bucket wall.
   - Expected: A: 60 °C within 0.4–11 min; total heat 1.14–1.16 MJ; 1.32 kg of slaked
     lime. B: the paste gets far hotter (up to ≈ 300 °C, secondary) and can char or
     light the wood.
   - Source: 75, 73, 76, 77. Today: no lime.
9. **R-T9 Vitriol on wood ash** — medium
   - Scene: an ACID pool poured over a heap of ASH.
   - Measure: CO2 made; acid used; what is left.
   - Expected: CO2 0.32–0.44 g per g of carbonate in the ash; the acid's potency falls;
     a salt stays in the liquid; no "smoke". The fizzing ends within seconds to a
     minute.
   - Source: 80, 83. Today: ash becomes smoke (first pass C19).
10. **R-T10 Acid meets lye** — low
    - Scene: equal volumes of a strong acid and a strong base (needs a base fluid).
    - Measure: heat and speed.
    - Expected: 55.8 kJ per mol of water formed, released at once.
    - Source: 78. Today: no base.
11. **R-T11 Salt in water** — medium
    - Scene: 100 g of rock salt in 1 L of water at 20 °C, still; then stirred.
    - Measure: time to dissolve; water temperature; how much more salt the water takes.
    - Expected: all 100 g dissolves (saturation ≈ 360 g/L). Stirred recession at most
      ≈ 1 mm/min; still water slower. The water cools ≈ 1.6 °C (derived from +66 J/g).
    - Source: 84, 83, 85. Today: no salt.
12. **R-T12 Black powder deflagrates** — medium
    - Scene: 100 g of black powder in a loose train, lit at one end; then 100 g sealed
      in a small wooden cask.
    - Measure: gas made; solid residue; heat; what happens to the cask.
    - Expected: gas ≈ 27 L at STP (≈ 271 cm³/g); residue 50–60%; heat ≈ 2.9 kJ/g; no
      detonation. The sealed cask bursts. Ignition: the sources disagree (125 °C against
      200–464 °C), so do not test the ignition temperature yet.
    - Source: 87, 86, 88. Today: no gunpowder.
13. **R-T13 Every rate follows Arrhenius** — high
    - Scene: a unit test of the reaction table. For each row, compute its rate at
      25 and 35 °C.
    - Measure: ratio of the two rates.
    - Expected: exp(Ea/R·(1/298 − 1/308)) for the row's Ea; a row with Ea 53 kJ/mol gives
      2.0 ± 0.1. No row is temperature-blind.
    - Source: 89, 90. Today: `_law_acid` reads no temperature.
14. **R-T14 Every row closes its books** — high
    - Scene: a unit test of the reaction table.
    - Measure: products mass against reactants mass; heat against ΔH.
    - Expected: mass equal within 0.1%; heat within 1% of the row's ΔH (R1–R16 values).
    - Source: 64, 74, 91. Today: acid rows turn mass
      into smoke.
15. **R-T15 Brimstone** — low
    - Scene: burn 10 g of sulfur in a closed 30 m³ room.
    - Measure: SO2 made; ppm; the occupant's percepts.
    - Expected: ≈ 20 g of SO2 (≈ 245 ppm, derived: above the IDLH of 100); heat 9.26 kJ/g;
      the occupant `smells` it within seconds of 1 ppm.
    - Source: 91, 35, 45. Today: no sulfur.

### 4.4 `tests/test_realism_body_clocks.py`

1. **BC-T1 A long day makes a body sleepy** — high
   - Scene: a person wakes at 07:00 and stays up. Skip ahead when quiet.
   - Measure: process S by the hour; the `tired` fact; the hour sleep starts when the
     person may choose.
   - Expected: S rises as 1 − e^(−t/18.2 h) toward its upper level (± 5%). `tired`
     appears after ≥ 16 h awake. Sleep starts between 22:00 and 01:00.
   - Source: 93. Today: no sleep pressure.
2. **BC-T2 A night's sleep restores** — high
   - Scene: BC-T1, then 8 h of sleep in a quiet room.
   - Measure: S during sleep; the number of sim steps the night costs.
   - Expected: S falls as e^(−t/4.2 h) (± 5%). A quiet night costs at most 100 steps
     (skip-ahead).
   - Source: 93; roadmap 107 list B. Today: missing.
3. **BC-T3 Sleep loss slows a body** — high
   - Scene: one person after 17 h awake and after 24 h awake.
   - Measure: reaction delay and error rate (the G2 reaction law of the first pass).
   - Expected: the same impairment as BAC 0.05% (17 h) and 0.10% (24 h): ≈ 0.74% worse
     per hour from hour 10.
   - Source: 98, 99. Today: missing.
4. **BC-T4 Woken groggy** — medium
   - Scene: a sleeper woken by a 70 dB noise.
   - Measure: reaction delay at 0, 15, 30 and 60 min after waking.
   - Expected: impaired at once; back near baseline in 15–30 min; full by ≥ 60 min.
   - Source: 97. Today: full function on the waking tick.
5. **BC-T5 Holding a load** — high
   - Scene: a person holds a load out at 20%, 50% and 100% of shoulder strength.
   - Measure: time until the arm drops it (`drops` event).
   - Expected (shoulder fit 14.86·f^−1.83 s): 100% ≈ 15 s; 50% ≈ 53 s; 20% ≈ 4.7 min;
     each within × 1.5 (the pooled fit gives 22 s, 86 s and 8.8 min).
   - Source: 101. Today: held for ever.
6. **BC-T6 Sprint and recover** — medium
   - Scene: a person runs all out for 30 s, then rests (needs a run gait, first pass G3).
   - Measure: sprint capacity during rest.
   - Expected: ≈ 20% at the end of the sprint; ≈ 65% after 90 s; ≈ 85% after 6 min.
   - Source: 102. Today: no run, no store.
7. **BC-T7 A day's march** — medium
   - Scene: a person walks 8 h with the army halts.
   - Measure: distance; the `exhausted` fact.
   - Expected: ≈ 32 km at 4 km/h with no `exhausted`. A forced march past ≈ 56 km in 24 h
     gives `exhausted`.
   - Source: 103. Today: no fatigue (walking speed: first pass G3).
8. **BC-T8 Thirst in the heat** — high
   - Scene: a person working at ≈ 1000 W in 30 °C air with no water.
   - Measure: sweat rate; water deficit; when `thirsty` appears; when performance
     falls; when the person collapses.
   - Expected: sweat 0.5–2 L/h (≈ 1.2 L/h). `thirsty` before a 2% loss. Performance
     falls past 2% of body weight (≈ 1.4 L for 70 kg, within 1–2 h). Severe at 8–10%.
     Death at 15–20% of body water, possible within ≈ 3 days in heat.
   - Source: 105, 106. Today: no water store.
9. **BC-T9 Water at rest** — medium
   - Scene: a resting person at 20 °C for 3 days, free to drink.
   - Measure: water drunk per day.
   - Expected: 1.5–3.7 L/d (minimum need ≈ 1.5 L/d).
   - Source: 106, 107. Today: missing.
10. **BC-T10 Fasting** — medium
    - Scene: a person with water but no food for 30 days (skip ahead).
    - Measure: `hungry`; weight lost per day; performance; survival.
    - Expected: `hungry` within 12–24 h. ≈ 0.9 kg/day in week 1, ≈ 0.3 kg/day by week 3.
      Aerobic performance holds to 10% weight loss. No death before day 40.
    - Source: 108, 109, 110, 106. Today: missing.
11. **BC-T11 Fallen into a cold river** — high
    - Scene: a person in 5 °C water with the head above the surface, unable to climb
      out.
    - Measure: phases and core temperature.
    - Expected: cold shock (gasping) in the first 1–3 min; useful arm movement lost by
      ≈ 10 min; core falls ≈ 3.7 °C/h (± 30%); unconscious after ≈ 1 h or more; death
      within 1–3 h (model: ≈ 136 min).
    - Source: 112, 113, 115, 116. Today: no core temperature;
      also first pass G9.
12. **BC-T12 A cold night outdoors** — medium
    - Scene: a person in wet light clothes at 5 °C with wind, shivering, for 3 h.
    - Measure: core temperature.
    - Expected: falls ≈ 0.17 °C/h (within × 2). Mild hypothermia only below 35 °C.
    - Source: 118, 117. Today: missing.
13. **BC-T13 Heat stroke** — medium
    - Scene: a person working hard above the WBGT limit (RAL = 59.9 − 14.1·log10 M).
    - Measure: core temperature; the facts `overheated` and `confused`; cooling in cold
      water.
    - Expected: collapse becomes likely near 39.2 °C (≈ 25%); heat stroke with confusion
      above 40.5 °C. Cold water cools ≈ 0.2 °C/min.
    - Source: 120, 121. Today: missing.
14. **BC-T14 An evening at the tavern** — high
    - Scene: a 70 kg man drinks 4 standard drinks (4 × 14 g) in 1 h.
    - Measure: BAC by the minute; behaviour facts.
    - Expected: peak ≈ 0.09–0.11% (r 0.7), 30–90 min after the first drink; then falls
      0.010–0.022 %/h. Slower reactions from 0.05%; staggering at 0.20%; stupor at
      0.30%.
    - Source: 123, 122. Today: no alcohol (first pass P23).
15. **BC-T15 Poisoned** — medium
    - Scene: a 70 kg person swallows 10 g of paracetamol.
    - Measure: blood level by the hour; the liver injury events.
    - Expected: peak within 90 min, at most ≈ 160 µg/ml (D/V); half-life 2–2.5 h, longer
      if the liver is hurt; injury shows after 24–36 h and peaks at 3–5 days if not
      treated.
    - Source: 127, 129, 128. Today: missing.
16. **BC-T16 A cut heals** — medium
    - Scene: a 10 cm cut on an arm, cleaned.
    - Measure: wound strength as a fraction of skin; `hurt`.
    - Expected: ≈ 3% at 1 week, ≈ 20% at 3 weeks, ≈ 80% at 3 months; `hurt` falls.
    - Source: 131. Today: wounds never heal.
17. **BC-T17 A dirty wound** — low
    - Scene: a deep puncture with soil in it, not cleaned.
    - Measure: when infection facts appear.
    - Expected: tetanus signs after 3–21 days (≈ 8). Gas gangrene, if it comes, within
      ≈ 24 h. Septic shock: each hour without treatment lowers survival ≈ 7.6%.
    - Source: 133, 134, 136. Today: missing.
18. **BC-T18 Skipping ahead changes nothing** — high
    - Scene: one person through 8 quiet hours, run (a) in 1 s steps and (b) with the
      closed-form skip.
    - Measure: S, water, energy, BAC and wound healing at the end.
    - Expected: the two runs agree within 0.1%.
    - Source: — (roadmap 107 list B). Today: no clocks.

### 4.5 `tests/test_realism_slow_world.py`

1. **SW-T1 Bread on the shelf** — high
   - Scene: a loaf in a bread box at 22 °C; another at 8 °C. Skip ahead.
   - Measure: day of the `mouldy` event.
   - Expected: day 4–6 at 22 °C; not within 10 days at 8 °C.
   - Source: 140. Today: no food.
2. **SW-T2 Meat in the larder** — high
   - Scene: fresh meat at 0, 5, 10 and 16 °C.
   - Measure: days to `smells_off` and to `slimy`.
   - Expected: slime on day 10, 3, 2 and 1 (each within × 1.5). The smell comes first.
   - Source: 145, 146. Today: missing.
3. **SW-T3 The temperature law** — high
   - Scene: the same perishable at 0, 10, 15 and 25 °C.
   - Measure: spoilage rate ratios.
   - Expected: 10 °C/0 °C ≈ 4 and 25 °C/15 °C ≈ 2.2 (within 25%), as the square-root
     law gives. A flat Q10 of 2–3 fails the first ratio.
   - Source: 139, 138. Today: missing.
4. **SW-T4 Milk left out** — medium
   - Scene: raw milk at 22 °C; pasteurised milk at 4 °C.
   - Measure: time to `sour`.
   - Expected: raw: 12–24 h (weak source). Pasteurised, cold: 10–21 days.
   - Source: 143, 141. Today: missing.
5. **SW-T5 A cured ham keeps** — medium
   - Scene: a dry-cured ham (water activity < 0.88) at 20 °C for 6 months.
   - Measure: spoilage events.
   - Expected: none within 6 months (whole country ham: up to 1 year).
   - Source: 148, 149. Today: missing.
6. **SW-T6 Ale ferments** — medium
   - Scene: 20 L of wort at 20 °C with yeast, in a cellar.
   - Measure: days to finish; ethanol and CO2 made; ABV.
   - Expected: done in ≈ 6 days (× 1.5). Per g of sugar: ≈ 0.49 g ethanol (0.511 less
     ≈ 4%) and ≈ 0.49 g CO2. ABV = (OG − FG) × 131.25 ± 0.3. The CO2 gathers low in the
     cellar (GS-T4).
   - Source: 152, 150. Today: missing.
7. **SW-T7 Wine at the wrong temperature** — low
   - Scene: grape must at 11, 25 and 35 °C.
   - Measure: fermentation state.
   - Expected: sluggish at 11 °C; normal at 25 °C (≈ 10 days); stuck at 35 °C.
   - Source: 156, 157, 154. Today: missing.
8. **SW-T8 Firewood seasons** — medium
   - Scene: split green oak stacked outdoors in summer weather; another stack in
     winter.
   - Measure: moisture content by the month; how well it lights.
   - Expected: below 20% after ≥ 6 months; lumber 25 mm thick reaches 20% in ≈ 60 days
     in summer and ≈ 140 days in winter. Wet wood lights slowly (first pass F23).
   - Source: 159, 160. Today: held water leaves only by boiling.
9. **SW-T9 A wet cloak dries** — medium
   - Scene: a wet cloak (needs cloth) hung in still air at 20 °C, 65% RH; then in
     2 m/s wind.
   - Measure: drying rate.
   - Expected: ≈ 1.6 g/(m²·min) both sides in still air (± 50%); faster in wind
     (first pass L15: Θ = 25 + 19·v).
   - Source: 161, I-104. Today: no cloth, no evaporation.
10. **SW-T10 Grass grows back** — low
    - Scene: a grazed pasture in May-like weather, and in midwinter weather.
    - Measure: days to a 3-leaf sward; daily growth.
    - Expected: 12–15 days in May, ≈ 90 days in midwinter; 55–65 kg DM/ha/day in summer.
    - Source: 165, 164. Today: no growth.
11. **SW-T11 Wheat ripens** — low
    - Scene: wheat sown in a field with a daily temperature record.
    - Measure: growing degree days at maturity.
    - Expected: 1,538–1,665 GDD (base 0 °C).
    - Source: 168. Today: missing.
12. **SW-T12 After a death** — medium
    - Scene: a body at 20 °C after death.
    - Measure: core temperature; rigor; livor; the first smell of decay.
    - Expected: a plateau of a few hours, then 1–1.5 °C/h. Rigor from 1–2 h, complete by
      12 h, gone after 24–48 h. Livor complete in 6–8 h. Smell of decay (a gas
      species) after 12–36 h.
    - Source: 170, 171. Today: missing (and bodies are at
      20 °C to begin with: first pass B19).
13. **SW-T13 Decomposition by degree-days** — low
    - Scene: a body at a steady 15 °C and 50% RH.
    - Measure: decomposition fraction against time.
    - Expected: ≈ 18% at 30 days; ≈ 100% at ≈ 166 days (Vass, ± 50%).
    - Source: 175, 173. Today: missing.
14. **SW-T14 A fence post rots** — low
    - Scene: untreated pine posts in wet ground at a mean 20 °C and at a mean 8 °C; a
      dry post.
    - Measure: years to failure.
    - Expected: 1.8–3.6 years warm; ≈ 6 years cool; the dry post (≤ 20% moisture) never
      rots.
    - Source: 179, 178. Today: missing.
15. **SW-T15 One event per object** — high
    - Scene: 1,000 loaves in a larder for 10 days at one temperature; then a fire
      warms the room for 1 h.
    - Measure: scheduled events; per-tick work while quiet; the reschedule.
    - Expected: one pending event per loaf; no per-tick work for the loaves while the
      temperature is steady; the warm hour moves the events earlier, and only for the
      loaves it warmed.
    - Source: — (roadmap 107 list B). Today: missing.

### 4.6 `tests/test_realism_contacts.py`

1. **K-T1 A boulder rolls down a hill** — high
   - Scene: a near-round granite boulder (≈ 1 m) at rest on a grassy slope of 25°; a
     granite cube of the same mass beside it.
   - Measure: whether each moves; rolling or sliding (spin × radius against speed);
     where the boulder stops on a run-out of 5°.
   - Expected: the boulder rolls without sliding (tan 25° = 0.47 ≤ 3.5 × 0.56). The cube
     stays (tan 25° < μ 0.56 and < b/h = 1). On the 5° run-out the boulder stops within a
     few metres (rolling resistance on grass 0.45–0.65).
   - Source: 192, 185. Today: neither moves (derived from
     `_law_torque`).
2. **K-T2 Slide or tip on a tilting board** — medium
   - Scene: a tall wooden block (b/h = 0.3) and a flat one (b/h = 2) on an oak board
     that tilts by 1° a second.
   - Measure: the angle at which each moves, and how.
   - Expected: the tall block tips near 17° (tan θ = 0.3); the flat block slides near
     27–32° (tan θ = μ_s 0.5–0.62), ± 3°.
   - Source: 193, 180. Today: neither slides; the tall one tips only
     when its centre of mass leaves its footprint.
3. **K-T3 A heap finds its slope** — high
   - Scene: pour 0.1 m³ of dry sand, of gravel and of dry ash onto a flat floor from one
     point; also let a wall of ash 1 voxel thick and 1 m tall stand.
   - Measure: the slope of each heap's side.
   - Expected: sand 34–35°, gravel 40–45°, ash 30–36° (± 3°). The ash wall slumps into a
     heap.
   - Source: 194, 195. Today: 90°, the column stands.
4. **K-T4 Friction by pair** — medium
   - Scene: push a 10 kg block across a floor: iron on iron dry, iron on iron greased,
     wood on ice, leather on wet metal. (Wood and stone are first pass M-T4.)
   - Measure: breakaway force over weight.
   - Expected: 1.0, 0.15–0.20, 0.05 and 0.4 (± 20%).
   - Source: 181, 180. Today: 0.40 for all.
5. **K-T5 A thrown axe turns** — high
   - Scene: a person throws a 1.2 kg axe (30 cm handle) overarm at a target 3.66 m away.
   - Measure: spin rate; angle turned at the target; which part strikes; the struck
     area used by `_blow`.
   - Expected: about one full turn by the target (≈ 3.5 m per turn, ± 20%); 3–4 turns a
     second at 10–13 m/s. A throw from twice as far arrives at a different angle. The
     blade strikes only when the angle is right; otherwise the handle (a blunt area).
   - Source: 204, 205, 206. Today: no spin; the axe arrives in its
     release pose.
6. **K-T6 A thrown knife turns faster** — medium
   - Scene: as K-T5 with a 30 cm knife at 14 m/s.
   - Measure: distance per full turn.
   - Expected: ≈ 2 m per turn (± 20%), shorter than the axe's.
   - Source: 204. Today: no spin.
7. **K-T7 A felled tree's leaves fall slower and spread** — high
   - Scene: `tree_hinge` felled in still air; then in a 3 m/s wind.
   - Measure: time for the leaves to reach the ground against the trunk; area they
     land on.
   - Expected: the trunk lands in 1.3–2.0 s (first pass S-T2). Loose leaves keep falling
     for longer, at ≈ 1–2 m/s (paper shapes 0.9–3.5 m/s). They land over a wider area than
     the crown; downwind in the wind run. No block of leaves stays in the air or lands
     as one lump.
   - Source: 198, 199; roadmap 107 list C. Today: leaves drop in bulk
     (likely cause in roadmap 107).
8. **K-T8 A single leaf flutters** — medium
   - Scene: one leaf (a thin light piece, ≈ 5 × 8 cm) dropped from 5 m in still air;
     one maple seed.
   - Measure: mean descent speed; sideways drift; spin.
   - Expected: leaf 0.9–2.1 m/s with sideways swings (flutter or tumble); seed
     1.0–1.1 m/s, spinning at 50–110 rad/s after ≈ 1 m of fall.
   - Source: 198, 196. Today: a LEAF voxel falls at up to 12–17 m/s
     (derived).
9. **K-T9 Rock bounces off rock** — low
   - Scene: drop a granite block 1 m onto (a) bedrock and (b) soft soil; roll one in
     at 3 m/s along the ground first.
   - Measure: normal and tangential speed ratios at the first impact.
   - Expected: normal 0.33–0.37 (bedrock) and 0.28–0.32 (soft soil); tangential
     0.78–0.92. The lattice and the solver agree within 0.1.
   - Source: 188. Today: lattice 0; solver unknown.
10. **K-T10 Slipping on ice** — medium
    - Scene: a person walks across a pond frozen at −10 °C; then across dry stone.
    - Measure: `slips` events.
    - Expected: slips on ice (μ well below the friction a step needs); no slips on dry
      stone.
    - Source: 181, 184, 187. Today: no ice; walking never
      slips.
11. **K-T11 A rope takes a load** — low
    - Scene: a 12 mm manila rope hung from a beam; load it in steps of 100 kg; then a
      rope with an overhand knot.
    - Measure: load at which it parts.
    - Expected: ≈ 1,130 kg (± 20%); with the knot ≈ 45% of that.
    - Source: 217, 219. Today: no rope (needs bonds).

### 4.7 `tests/test_realism_water_bodies.py`

1. **KW-T1 A plank floats flat, part under** — high
   - Scene: a pine plank 1 m × 20 cm × 5 cm dropped flat into a still pond; then dropped
     edge-first.
   - Measure: draft; final orientation; time to settle.
   - Expected: draft ≈ 35–40% of the thickness (s ≈ 0.35–0.4 for seasoned pine). The
     edge-first plank turns flat and stays flat ((b/t)² = 16 > 6·s·(1 − s)).
   - Source: 210, 211. Today: the plank rests on top of the water; no
     righting.
2. **KW-T2 A square log rolls corner-up** — low
   - Scene: a square-section log of density ratio 0.5 placed flat in water.
   - Measure: final orientation.
   - Expected: it rolls to float corner-up (unstable flat for 0.21 < s < 0.79).
   - Source: 211 (derived criterion). Today: no rotation.
3. **KW-T3 A flat stone skips; a round one sinks** — high
   - Scene: a flat stone (0.1 kg, 10 cm across) thrown at 8 m/s, 10° above the water,
     tilted 20°, with a spin of ≥ 3 turns a second. Then the same throw with no spin, and
     a round stone.
   - Measure: number of skips; distance to the first skip.
   - Expected: flat and spinning: several skips (the model gives ≈ 17 at 8 m/s; accept
     ≥ 5); first skip ≈ 2 m away. No spin: the stone tips and stops within 1–2 skips.
     Round stone: no skip; it sinks at the first touch.
   - Source: 202, 201. Today: every stone passes into the water as if
     it were air (derived).
4. **KW-T4 The skip needs speed** — medium
   - Scene: KW-T3 flat stone at 0.5, 1, 2 and 5 m/s.
   - Measure: skips.
   - Expected: none at 0.5 m/s; the first skips appear between 1 and 2 m/s; ≈ 6 at
     5 m/s (model; accept 3–10).
   - Source: 202. Today: missing.
5. **KW-T5 Who floats** — high
   - Scene: a person in deep water with full lungs; the same person breathed out; the
     same person in 20 kg of plate.
   - Measure: whether each floats; the `floats` and `sinks` events.
   - Expected: full lungs: floats with the head near the surface (density ≈ 0.98).
     Breathed out: sinks slowly (≈ 1.04). In armour: sinks (≈ 1.2).
   - Source: 207, 208 (densities derived). Today: flesh equals water;
     armour has no mass.
6. **KW-T6 Pulling a body through water** — medium
   - Scene: tow a person-sized body through still water at 1.2 and 1.8 m/s.
   - Measure: drag force.
   - Expected: 34 N and 91 N (± 30%); C_d on the frontal area 0.6–0.75.
   - Source: 212. Today: no water force.
7. **KW-T7 A stone dropped in a pond** — medium
   - Scene: a 0.5 kg stone dropped from 1 m into a pond 1 m deep.
   - Measure: ring of ripples; their speed; splash.
   - Expected: rings spread out; long waves at ≈ √(g·h) ≤ 3.1 m/s and short ripples no
     slower than 0.23 m/s. A wettable stone at ≈ 4.4 m/s makes a splash without a big air
     cavity.
   - Source: 215, 216, 214. Today: no waves; drops at
     fixed lattice speeds.
8. **KW-T8 A wave takes time to cross a pond** — high
   - Scene: a pond 10 m long and 0.5 m deep. Raise the water at one end by 5 cm at once.
   - Measure: when the far end starts to rise.
   - Expected: after ≈ 10 / 2.2 = 4.5 s (± 20%). The same at 5 and 10 cm voxels.
   - Source: 215. Today: on the next tick (`_law_head` pairs by rank;
     derived).
9. **KW-T9 Green oak sinks, seasoned oak floats** — low
   - Scene: a block of green white oak (moisture 80%) and one of seasoned oak (12%).
   - Measure: floats or sinks.
   - Expected: green sinks (density > 1.0 once moisture > ≈ 67%); seasoned floats.
   - Source: 210 (derived). Today: wood floats whatever its water.

### 4.8 Count of reference tests

| File | High | Medium | Low | Total |
|---|---|---|---|---|
| test_realism_wind.py | 6 | 5 | 5 | 16 |
| test_realism_gases.py | 6 | 5 | 1 | 12 |
| test_realism_reactions.py | 4 | 7 | 4 | 15 |
| test_realism_body_clocks.py | 8 | 9 | 1 | 18 |
| test_realism_slow_world.py | 4 | 6 | 5 | 15 |
| test_realism_contacts.py | 4 | 5 | 2 | 11 |
| test_realism_water_bodies.py | 4 | 3 | 2 | 9 |
| **All** | **36** | **40** | **20** | **96** |

## 5. References

Every URL below was opened by a researcher in this pass, except where the entry
says otherwise. "Abstract only" means only the abstract could be read. "Secondary"
means the source summarises other work and the primary was not opened. "Local copy"
means the text was read from a copy downloaded earlier, not fetched again. Sources
that could not be opened are not listed (among them Clanet, Hersen & Bocquet 2004;
Penwarden 1973; Hunt et al. 1976; Moore & Maguire 2004; Lentink et al. 2009; Field et
al. 1997; Di Blasi 2008; Sasse 1985; Adolph 1947; Peel 1997). Values that depended on
them are marked (unverified) in the tables. "Used for" lists the rows and tests that
cite the source.

### 5.1 Wind and weather

| # | Title | URL | Used for |
|---|---|---|---|
| 1 | Harper, Kepert & Ginger 2010, Guidelines for converting between various wind averaging periods in tropical cyclone conditions (WMO/TD-1555) | https://www.systemsengineeringaustralia.com.au/download/WMO_TC_Wind_Averaging_27_Aug_2010.pdf | W1, W3, W-T1, W-T2 |
| 2 | WMO, Guide to Instruments and Methods of Observation (WMO-No. 8, 2014), Annex 1.B siting classification: z0 table (Davenport, adapted by Wieringa) | https://www.eoas.ubc.ca/courses/atsc303/Instruments/wmo_guides/CIMO_Guide_2014-Met_Site_Classification.pdf | W1, W2, W-T1 |
| 3 | WMO, Guide to Meteorological Instruments and Methods of Observation (WMO-No. 8, 7th ed., 2008): gust and mean-wind definitions | https://www.weather.gov/media/epz/mesonet/CWOP-WMO8.pdf | W4, W21, W-T2, W-T16 |
| 4 | US National Weather Service (Miami), Beaufort wind scale with land descriptions | https://www.weather.gov/mfl/beaufort | W5, W15, W-T11 |
| 5 | Wikipedia, Beaufort scale (m/s bands) (secondary) | https://en.wikipedia.org/wiki/Beaufort_scale | W5, W20 |
| 6 | UK Met Office, Beaufort wind force scale (knots) | https://weather.metoffice.gov.uk/guides/coast-and-sea/beaufort-scale | W5 |
| 7 | Blocken & Carmeliet 2004, Pedestrian wind environment around buildings: literature review (J. Thermal Envelope & Building Science 28), preprint; quotes Lawson & Penwarden 1975, Bottema 1993, Murakami 1980, Hunt et al. 1976 | https://www.urbanphysics.net/2004_JTEBS_WINDreview_preprint.pdf | W6, W15, W-T5, W-T11 |
| 8 | Schriever 1976, Ground-level winds around tall buildings, Canadian Building Digest CBD-174 (NRC Canada) | https://nrc-publications.canada.ca/fra/voir/td/?id=8ecc4b1f-12aa-4f47-8056-976c2bb494aa | W6, W-T5 |
| 9 | Hoerner 1965, Fluid-Dynamic Drag, p. 3-14 and Fig. 25 (drag areas of a man) | https://ia800606.us.archive.org/17/items/FluidDynamicDragHoerner1965/Fluid-dynamic_drag__Hoerner__1965_text.pdf | W7, W-T5 |
| 10 | Engineering ToolBox, Drag coefficient (person standing 1.0–1.3) (secondary) | https://www.engineeringtoolbox.com/drag-coefficient-d_627.html | W7 |
| 11 | Pugh 1971, The influence of wind resistance in running and walking, J. Physiol. (abstract only) | https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=TITLE:%22influence%20of%20wind%20resistance%20in%20running%20and%20walking%22&resultType=core&format=json | W7, W-T6 |
| 12 | Andrews 2018, The Rothermel surface fire spread model and associated developments, USDA FS RMRS-GTR-371 | https://www.fs.usda.gov/rm/pubs_series/rmrs/gtr/rmrs_gtr371.pdf | W9 |
| 13 | Cruz, Alexander & Kilinc 2022, Wildfire rates of spread in grasslands under critical burning conditions, Fire 5:55 (restates Cheney, Gould & Catchpole 1998) | https://pdfs.semanticscholar.org/0e66/0f20134d92598b343893d39201317c9a59d0.pdf | W10, W-T7 |
| 14 | Sullivan 2009, A review of wildland fire spread modelling 2: empirical and quasi-empirical models (arXiv 0706.4128) | https://arxiv.org/pdf/0706.4128 | W10, W-T7 |
| 15 | Cruz & Alexander 2019, The 10% wind speed rule of thumb for estimating a wildfire's forward rate of spread, Annals of Forest Science 76:44 (abstract only) | https://www.frames.gov/catalog/57791 | W11, W-T8 |
| 16 | ISO 7730:2005 preview (iTeh): convective coefficient h_c and draught rate; the same h_c in the CBE pythermalcomfort code | https://cdn.standards.iteh.ai/samples/39155/9632a0563ca742209edb45856ff69296/ISO-7730-2005.pdf | W12, W17, W-T9, W-T13 |
| 17 | Viswanathan et al. 2025, Impact of human body shape on free convection heat transfer, PLOS ONE | https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0318842 | W12 |
| 18 | Engineering ToolBox, Convective heat transfer (h = 10.45 − v + 10·√v, in kcal/(m²·h·°C)) (secondary) | https://www.engineeringtoolbox.com/convective-heat-transfer-d_430.html | W12 |
| 19 | Wikipedia, Wind chill (2001 JAG/TI formula, metric) (secondary) | https://en.wikipedia.org/wiki/Wind_chill | W13, W-T10 |
| 20 | US National Weather Service, Wind chill chart | https://www.weather.gov/safety/cold-wind-chill-chart | W13 |
| 21 | Government of Canada, Wind chill index and frostbite risk | https://www.canada.ca/en/services/environment/weather/severeweather/wind-chill-index.html | W13, W-T10 |
| 22 | Kok, Parteli, Michaels & Bou Karam 2012, The physics of wind-blown sand and dust, Rep. Prog. Phys. 75 (arXiv 1201.4353) | https://arxiv.org/pdf/1201.4353 | W14, W-T11 |
| 23 | Roney & White 2004, Definition and measurement of dust aeolian thresholds, JGR 109 F01013 | https://research.engineering.ucdavis.edu/wind/wp-content/uploads/sites/17/2014/03/Ronet-White-2004.pdf | W14, W-T11 |
| 24 | Jiang & Chen 2003, Buoyancy-driven single-sided natural ventilation in buildings with large openings, Int. J. Heat Mass Transfer 46 (author PDF) | https://engineering.purdue.edu/~yanchen/paper/2003-3.pdf | W16, W-T12, W-T16 |
| 25 | Shin et al. 2016, Gust effect factors and natural sway frequencies of trees (ICCM2016); restates Moore & Maguire 2004 (secondary) | https://www.sci-en-tech.com/ICCM2016/PDFs/1778-6188-1-PB.pdf | W18, W-T14 |
| 26 | Virot, Ponomarenko, Dehandschoewercker, Quéré & Clanet 2016, Critical wind speed at which trees break, Phys. Rev. E 93:023001 | https://yakari.polytechnique.fr/Django-pub/documents/virot2016rp-2pp.pdf | W19, W-T15 |

### 5.2 Gas species

| # | Title | URL | Used for |
|---|---|---|---|
| 27 | European Fire Safety Alliance / NL Fire Service Academy 2020, Appendix 1: Comparison between the ISO standard, SFPE handbook and AEGL (reproduces ISO 13571 and Purser & McAllister 2016) (secondary for ISO) | https://www.europeanfiresafetyalliance.org/wp-content/uploads/2021/02/20200916-BA-Appendix-1-Comparison-between-the-ISO-standard-SFPE-handbook-and-AEGL.pdf | GS1, GS4, GS7, GS-T1, GS-T2, GS-T3 |
| 28 | McGrattan et al., Fire Dynamics Simulator User's Guide, NIST SP 1019, sect. 17.10.9 (Purser FED) | https://www.fse-italia.eu/PDF/ManualiFDS/FDS_User_Guide.pdf | GS2, GS-T2, GS-T3 |
| 29 | Purser 1989, Modelling toxic and physical hazard in fire, Fire Safety Science 2:391–400 (read from a local copy; URL inferred) | https://publications.iafss.org/publications/fss/2/391/view/fss_2-391.pdf | GS2, GS3, GS-T2 |
| 30 | ATSDR, Interaction profile, Appendix A: background information for carbon monoxide | https://atsdr.cdc.gov/interaction-profiles/media/pdfs/ip12-a.pdf | GS3 |
| 31 | Fine, Cavanagh, Afonja & Binions 2010, Metal oxide semi-conductor gas sensors in environmental monitoring, Sensors 10:5469, Tables 2–3 (secondary compilation) | https://www.ebi.ac.uk/europepmc/webservices/rest/PMC3247717/fullTextXML | GS4, GS7, GS-T4 |
| 32 | NIOSH, IDLH documentation: carbon monoxide | https://www.cdc.gov/niosh/idlh/630080.html | GS4, GS5 |
| 33 | NIOSH, IDLH documentation: carbon dioxide | https://www.cdc.gov/niosh/idlh/124389.html | GS5, GS7 |
| 34 | NIOSH, IDLH documentation: hydrogen sulfide | https://www.cdc.gov/niosh/idlh/7783064.html | GS5, GS8 |
| 35 | NIOSH, IDLH documentation: sulfur dioxide | https://www.cdc.gov/niosh/idlh/7446095.html | GS5, R-T15 |
| 36 | US EPA, AEGL results: carbon monoxide | https://www.epa.gov/aegl/carbon-monoxide-results-aegl-program | GS6 |
| 37 | US EPA, AEGL results: hydrogen sulfide | https://www.epa.gov/aegl/hydrogen-sulfide-results-aegl-program | GS6, GS10 |
| 38 | US EPA, AEGL results: sulfur dioxide | https://www.epa.gov/aegl/sulfur-dioxide-results-aegl-program | GS6 |
| 39 | OSHA, Hydrogen sulfide: hazards (concentration table) | https://www.osha.gov/hydrogen-sulfide/hazards | GS8, GS11, GS-T6 |
| 40 | ATSDR, Toxicological profile for hydrogen sulfide, ch. 4 (odour threshold, Table 4-2) | https://www.atsdr.cdc.gov/toxprofiles/tp114-c4.pdf | GS8, GS10, GS12, GS-T6 |
| 41 | Engineering ToolBox, Gases: densities (secondary) | https://www.engineeringtoolbox.com/gas-density-d_158.html | GS9, GS12, GS-T7 |
| 42 | NIOSH Pocket Guide: CO (npgd0105), CO2 (npgd0103), H2S (npgd0337), SO2 (npgd0575) | https://www.cdc.gov/niosh/npg/npgd0105.html | GS9, GS10, GS13, GS-T4, GS-T10 |
| 43 | ILO/WHO International Chemical Safety Cards via INCHEM: H2 (0001), CO2 (0021), CO (0023), H2S (0165), CH4 (0291) | https://www.inchem.org/documents/icsc/icsc/eics0001.htm | GS9, GS10, GS13, GS-T4, GS-T5, GS-T10 |
| 44 | NIST Chemistry WebBook, thermophysical properties of water (IAPWS-95), isobar at 1 atm | https://webbook.nist.gov/cgi/fluid.cgi?Action=Data&Wide=on&ID=C7732185&Type=IsoBar&Digits=5&P=1.01325&THigh=300&TLow=100&TInc=50&RefState=DEF&TUnit=C&PUnit=bar&DUnit=kg%2Fm3&HUnit=kJ%2Fkg&WUnit=m%2Fs&VisUnit=uPa*s&STUnit=N%2Fm | GS9, GS20, GS-T12 |
| 45 | ATSDR, Medical management guidelines: sulfur dioxide | https://wwwn.cdc.gov/Tsp/MMG/MMGDetails.aspx?mmgid=249&toxid=46 | GS10, R-T15 |
| 46 | ATSDR, Medical management guidelines: ammonia | https://wwwn.cdc.gov/TSP/MMG/MMGDetails.aspx?mmgid=7&toxid=2 | GS10 |
| 47 | US DOE EERE, Hydrogen: similar but different (safety fact sheet) | https://www1.eere.energy.gov/hydrogenandfuelcells/pdfs/h2_safety_fsheet.pdf | GS10, GS13, GS14, GS-T5 |
| 48 | ATSDR, Medical management guidelines: hydrogen sulfide | https://wwwn.cdc.gov/tsp/MMG/MMGDetails.aspx?mmgid=385&toxid=67 | GS11, GS13, GS-T6 |
| 49 | Johnsson et al. 2025, The burning characteristics of 3 m to 6 m dry Douglas-fir trees, NIST TN 2327r1 (read from a local copy) | https://doi.org/10.6028/NIST.TN.2327r1 | GS15, GS-T9, GS-T11 |
| 50 | Pitts 1995, The global equivalence ratio concept and the formation mechanisms of carbon monoxide in enclosure fires, Prog. Energy Combust. Sci. 21:197–237 | https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=911660 | GS16, GS-T9 |
| 51 | German Federal Institute for Risk Assessment (BfR) 2013, Grilling with charcoal is definitely not an indoor pursuit (press release) | https://www.bfr.bund.de/en/press-release/grilling-with-charcoal-is-definitely-not-an-indoor-pursuit/ | GS17, GS-T1 |
| 52 | Engineering ToolBox, Air: diffusion coefficients of gases in excess of air (secondary) | https://www.engineeringtoolbox.com/air-diffusion-coefficient-gas-mixture-temperature-d_2010.html | GS18 |
| 53 | Burgess 2024, Self-diffusion and binary-diffusion coefficients in gases, NIST TN 2279, Table 1a | https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=957163 | GS18 |
| 54 | Baughman, Gadgil & Nazaroff 1994, Mixing of a point source pollutant by natural convection flow within a room, Indoor Air 4 (abstract only) | https://indoor.lbl.gov/publications/mixing-point-source-pollutant-natural | GS19, GS-T8 |
| 55 | Gadgil, Lobscheid & Abadie, LBNL-51413 (Atmos. Environ. 2003): mixing times 2.4–23 min in a fan-stirred room; with Lobscheid & Gadgil (OSTI 791816) | https://osti.gov/scitech/servlets/purl/820651 | GS19, GS-T8 |
| 56 | Prediction of steam burns severity using Raman spectroscopy on ex vivo porcine skin, Sci. Rep. 2018 (PMC5932075) | https://www.ebi.ac.uk/europepmc/webservices/rest/PMC5932075/fullTextXML | GS20 |

### 5.3 Contact chemistry

| # | Title | URL | Used for |
|---|---|---|---|
| 57 | Thurner, Mann & Beck 1980, Kinetic investigation of wood pyrolysis, DOE report (OSTI 6809951) | https://www.osti.gov/servlets/purl/6809951 | R1, R-T1 |
| 58 | Thurner & Mann 1981, Kinetic investigation of wood pyrolysis, Ind. Eng. Chem. Process Des. Dev. 20(3) (abstract only) | https://www.osti.gov/biblio/5354651 | R1, R-T1 |
| 59 | Sinha, Jhalani, Ravi & Ray, Modelling of pyrolysis in wood: a review, SESI Journal (IIT Delhi) | https://web.iitd.ac.in/~ravimr/Publications/IndianJournals/sesi-sanjiv.pdf | R1 |
| 60 | Babrauskas, 'Pyrophoric carbon' and long-term, low-temperature ignition of wood (McNaughton 1945 data) | https://doctorfire.com/pages/low_temp_wood1.pdf | R2, R-T2 |
| 61 | FAO 1987, Simple technologies for charcoal making, Forestry Paper 41, ch. 4 | https://www.fao.org/4/x5328e/x5328e05.htm | R3, R-T3 |
| 62 | Peterson & Brown 2020, Oxidation kinetics of biochar from woody and herbaceous biomass, Chem. Eng. J. 401:126043 (OSTI 1648151) | https://www.osti.gov/servlets/purl/1648151 | R4, R-T4 |
| 63 | Branca & Di Blasi 2004, Global intrinsic kinetics of wood oxidation, Fuel 83(1) (abstract only) | https://www.osti.gov/etdeweb/biblio/20406889 | R4 |
| 64 | NIST Chemistry WebBook, carbon dioxide: gas-phase thermochemistry | https://webbook.nist.gov/cgi/cbook.cgi?ID=C124389&Mask=1 | R5, GS-T11, R-T5, R-T14 |
| 65 | NIST Chemistry WebBook, carbon monoxide: gas-phase thermochemistry | https://webbook.nist.gov/cgi/cbook.cgi?ID=C630080&Mask=1 | R5, R-T5 |
| 66 | Murphy & Shaddix 2006, Combustion kinetics of coal chars in oxygen-enriched environments, Combust. Flame 144 (Tognotti CO/CO2 correlation) | https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=916934 | R5, R-T5 |
| 67 | Cowie 2009, Steel corrosion rates in water and soil, Steel Construction New Zealand CTG1006 | https://scnz.org/wp-content/uploads/2020/12/CTG1006.pdf | R6, R-T6, R-T7 |
| 68 | Powell (Nickel Institute) 2020, Overview of the corrosion behaviour of metals in seawater, ICorr webinar | https://www.icorr.org/wp-content/uploads/2020/06/2020-04-30-MCF-ICorr-Webinar-Carol-Powell-of-Nickel-Institute-Corrosion-of-Metal-in-Seawater.pdf | R6, R-T7 |
| 69 | Majzlan, Grevel & Navrotsky 2003, Thermodynamics of Fe oxides part II, Am. Mineral. 88 (abstract only) | https://asu.elsevierpure.com/en/publications/thermodynamics-of-fe-oxides-part-ii-enthalpies-of-formation-and-r/ | R6 |
| 70 | NIST Chemistry WebBook, water: condensed-phase thermochemistry; with Wikipedia, Iron(III) oxide, for ΔfH of hematite (secondary) | https://webbook.nist.gov/cgi/cbook.cgi?ID=C7732185&Mask=2 | R6 |
| 71 | Frenzel 2009, How to inspect for flash rust, WJTA Conference | https://www.wjta.org/images/wjta/Proceedings/Papers/2009/E1%20Frenzel.pdf | R7, R-T6 |
| 72 | Wikipedia, Hand warmer (secondary) | https://en.wikipedia.org/wiki/Hand_warmer | R8 |
| 73 | National Lime Association 2007, Fact sheet: properties of lime (heats of reaction) | https://www.lime.org/wp-content/uploads/fact-properties2007rev.pdf | R9, R-T8 |
| 74 | NIST Chemistry WebBook, CaO and Ca(OH)2: condensed-phase thermochemistry | https://webbook.nist.gov/cgi/cbook.cgi?ID=C1305788&Mask=2 | R9, R-T14 |
| 75 | Vola et al. 2019, Reactivity and overburning tendency of quicklime burnt at high temperature, ZKG 10/2019 (EN 459-2 test data) | https://www.cimprogetti.com/K_2019_EN_042.pdf | R9, R-T8 |
| 76 | Carmeuse, Quicklime safety data sheet (2019) | https://www.carmeuse.com/sites/default/files/2020-01/SDS%20-%20Quicklime.pdf | R9, R-T8 |
| 77 | Wikipedia, Calcium oxide (secondary) | https://en.wikipedia.org/wiki/Calcium_oxide | R9, R16, R-T8 |
| 78 | Cerutti, Ko, McCurdy & Hepler 1978, Can. J. Chem. 56: ΔH° of ionization of water (abstract only) | https://www.osti.gov/etdeweb/biblio/5708009 | R10, R-T10 |
| 79 | Wikipedia, Self-ionization of water (recombination rate) (secondary) | https://en.wikipedia.org/wiki/Self-ionization_of_water | R10 |
| 80 | University of Georgia Extension B1142, Best management practices for wood ash as agricultural soil amendment | https://fieldreport.caes.uga.edu/publications/B1142/best-management-practices-for-wood-ash-as-agricultural-soil-amendment/ | R11, R-T9 |
| 81 | Virginia Cooperative Extension 452-510, wood ash liming value | https://www.pubs.ext.vt.edu/452/452-510/452-510.html | R11 |
| 82 | Etiegni & Campbell 1991, Physical and chemical characteristics of wood ash, Bioresource Technol. 37 (abstract only) | https://www.osti.gov/etdeweb/biblio/5086807 | R11 |
| 83 | Palandri & Kharaka 2004, A compilation of rate parameters of water–mineral interaction kinetics, USGS Open-File Report 2004-1068 | https://pubs.usgs.gov/of/2004/1068/pdf/OFR_2004_1068.pdf | R11, R12, R-T9, R-T11 |
| 84 | Pinho & Macedo 2005, Solubility of NaCl, NaBr and KCl in water, methanol, ethanol and their mixed solvents, J. Chem. Eng. Data 50:29 | https://seniorchem.com/solubility-NaCl.pdf | R12, R-T11 |
| 85 | Wikipedia, Enthalpy change of solution (secondary) | https://en.wikipedia.org/wiki/Enthalpy_change_of_solution | R12, R-T11 |
| 86 | Wikipedia, Gunpowder (secondary) | https://en.wikipedia.org/wiki/Gunpowder | R13, R-T12 |
| 87 | OCS, Technical specification summary: black powder (no primary source cited) | https://www.ocsresponds.com/ref/hazlist/TechSpecs.pdf | R13, R-T12 |
| 88 | US Army TM 9-1300-214, Military Explosives (1984/1990), ch. 10 | https://www.epa.gov/sites/default/files/2015-03/documents/9546041.pdf | R13, R-T12 |
| 89 | Wikipedia, Arrhenius equation (secondary) | https://en.wikipedia.org/wiki/Arrhenius_equation | R14, R-T13 |
| 90 | Wikipedia, Q10 (temperature coefficient) (secondary) | https://en.wikipedia.org/wiki/Q10_(temperature_coefficient) | R14, R-T13 |
| 91 | NIST Chemistry WebBook, sulfur dioxide: gas-phase thermochemistry | https://webbook.nist.gov/cgi/cbook.cgi?ID=C7446095&Mask=1 | R15, R-T14, R-T15 |
| 92 | Grasa et al., Hydrogen-assisted calcination of CaCO3 (PMC13449209): equilibrium CO2 pressure and ΔH | https://pmc.ncbi.nlm.nih.gov/articles/PMC13449209/ | R16 |

### 5.4 Body clocks

| # | Title | URL | Used for |
|---|---|---|---|
| 93 | Skeldon, Dijk & Derks 2014, Mathematical models for sleep-wake dynamics (PLoS ONE; arXiv 1311.1734): two-process parameters after Daan, Beersma & Borbély 1984 | https://arxiv.org/pdf/1311.1734 | BC1, BC-T1, BC-T2 |
| 94 | Skeldon & Dijk 2025, The complexity and commonness of the two-process model (PMC12176636) | https://pmc.ncbi.nlm.nih.gov/articles/PMC12176636/ | BC1 |
| 95 | Watson et al. 2015, Recommended amount of sleep for a healthy adult (AASM/SRS consensus) | https://aasm.org/resources/pdf/pressroom/adult-sleep-duration-consensus.pdf | BC2 |
| 96 | Ohayon et al. 2017, National Sleep Foundation's sleep quality recommendations, Sleep Health 3:6–19 | https://fatiguemanagersnetwork.org/wp-content/uploads/Ohayon-et-al.2017_National-Sleep-Foundations-Sleep-Quality-Recommendations.pdf | BC3 |
| 97 | Hilditch & McHill 2019, Sleep inertia: current insights, Nat. Sci. Sleep | https://pmc.ncbi.nlm.nih.gov/articles/PMC6710480/ | BC4, BC-T4 |
| 98 | Dawson & Reid 1997, Fatigue, alcohol and performance impairment, Nature 388:235 (third-party copy) | https://understandingaf447.com/extras/Alcohol-Fatigue.pdf | BC5, BC-T3 |
| 99 | Williamson & Feyer 2000, Moderate sleep deprivation produces impairments equivalent to alcohol intoxication, Occup. Environ. Med. 57:649 | https://pmc.ncbi.nlm.nih.gov/articles/PMC1739867/ | BC5, BC-T3 |
| 100 | Van Dongen, Maislin, Mullington & Dinges 2003, The cumulative cost of additional wakefulness, Sleep 26:117 (abstract only) | https://academic.oup.com/sleep/article-abstract/26/2/117/2709164 | BC5 |
| 101 | Frey-Law & Avin 2010, Endurance time is joint-specific, Ergonomics 53:109, Table 2 | https://pmc.ncbi.nlm.nih.gov/articles/PMC2891087/ | BC6, BC-T5 |
| 102 | McMahon & Jenkins 2002, Factors affecting the rate of phosphocreatine resynthesis following intense exercise, Sports Med. 32:761 | https://paulogentil.com/pdf/Factors%20Affecting%20the%20Rate%20of%20Phosphocreatine%20Resynthesis%20Following%20Intense%20Exercise.pdf | BC7, BC-T6 |
| 103 | US Army FM 21-18, Foot Marches (1990), ch. 4 | https://www.globalsecurity.org/military/library/policy/army/fm/21-18/fm21-18_ch4.pdf | BC8, BC-T7 |
| 104 | Dinh-Dang, Khafagy, Krause & Harris-Adamson 2023, Applied Ergonomics 106:103886 (cites the ILO/WHO 30% workload limit) | https://stacks.cdc.gov/view/cdc/231307/cdc_231307_DS1.pdf | BC9 |
| 105 | Sawka et al. 2007, ACSM position stand: exercise and fluid replacement, MSSE 39:377 | https://eclass.hmu.gr/modules/document/file.php/YD238/%CE%86%CF%81%CE%B8%CF%81%CE%B1%20%CE%91%CE%B8%CE%BB%CE%B7%CF%84%CE%B9%CE%BA%CE%AE%CF%82%20%CE%94%CE%B9%CE%B1%CF%84%CF%81%CE%BF%CF%86%CE%AE%CF%82/Exercise%20and%20fluid%20replacement.pdf | BC10, BC11, BC-T8 |
| 106 | Tipton & Montgomery 2025, The physiology of survival: fluid and food, Exp. Physiol. | https://discovery.ucl.ac.uk/id/eprint/10213215/1/Montgomery_Experimental%20Physiology%20-%202025%20-%20Tipton%20-%20The%20physiology%20of%20survival%20%20Fluid%20and%20food.pdf | BC11, BC12, BC-T8, BC-T9, BC-T10 |
| 107 | Institute of Medicine 2005, Dietary reference intakes for water, potassium, sodium, chloride and sulfate, ch. 4 | https://www.nationalacademies.org/read/10925/chapter/6 | BC11, BC-T9 |
| 108 | Longo & Mattson 2014, Fasting: molecular mechanisms and clinical applications, Cell Metab. | https://pmc.ncbi.nlm.nih.gov/articles/PMC3946160/ | BC12, BC-T10 |
| 109 | Kerndt, Naughton, Driscoll & Loxterkamp 1982, Fasting: the history, pathophysiology and complications, West. J. Med. (abstract only) | https://pmc.ncbi.nlm.nih.gov/articles/PMC1274154/ | BC12, BC-T10 |
| 110 | Gétaz et al. 2012, Hunger strike among detainees, Swiss Med. Wkly 142:w13675 | https://smw.ch/index.php/smw/article/download/1579/2042/5390 | BC12, BC-T10 |
| 111 | Kalm & Semba 2005, They starved so that others be better fed: the Minnesota experiment, J. Nutr. 135:1347 | https://sites.pitt.edu/~nancyp/uhc-1510/MinnesotaStarvationExperiment.pdf | BC12 |
| 112 | International Life Saving Federation 2016, Medical position statement MPS-12: cold water immersion | https://www.ilsf.org/wp-content/uploads/2018/11/MPS-12-2016-Cold-Water.pdf | BC13, BC-T11 |
| 113 | Giesbrecht, Cold Water Boot Camp: 1-10-1 (educational site, not peer-reviewed) | https://www.coldwaterbootcamp.com/pages/1_10_60v2.html | BC13, BC-T11 |
| 114 | Tipton, Eglin, Gennser & Golden 1999, Immersion deaths and deterioration in swimming performance in cold water, Lancet (abstract only) | https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=TITLE:%22Immersion%20deaths%20and%20deterioration%20in%20swimming%20performance%20in%20cold%20water%22&resultType=core&format=json | BC13 |
| 115 | Junaid & Mahmud-Or-Rashid 2024, Heliyon (Hayward 1975 cooling rates; modelled survival) | https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11234009/fullTextXML | BC13, BC-T11 |
| 116 | US Search and Rescue Task Force, Cold water survival (web table, no source cited, not peer-reviewed) | https://www.ussartf.org/cold_water_survival.htm | BC13, BC-T11 |
| 117 | Pyzocha 2020, Hypothermia and frostbite: WMS guidelines, Am. Fam. Physician | https://www.aafp.org/pubs/afp/issues/2020/1101/p571.html | BC14, BC-T12 |
| 118 | Helland et al. 2025, Wilderness Environ. Med. (PMID 41134980) (abstract only) | https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:41134980&resultType=core&format=json | BC15, BC-T12 |
| 119 | Grissom et al. 2004, Hypercapnia increases core temperature cooling rate during snow burial, J. Appl. Physiol. (abstract only) | https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=TITLE:%22Hypercapnia%20increases%20core%20temperature%20cooling%20rate%20during%20snow%20burial%22&resultType=core&format=json | BC15 |
| 120 | NIOSH 2016, Criteria for a recommended standard: occupational exposure to heat and hot environments (DHHS 2016-106) | https://www.cdc.gov/niosh/docs/2016-106/pdfs/2016-106.pdf | BC16, BC-T13 |
| 121 | Casa et al. 2015, NATA position statement: exertional heat illnesses, J. Athl. Train. 50(9) | https://pmc.ncbi.nlm.nih.gov/articles/PMC4639891/ | BC16, BC-T13 |
| 122 | NIAAA, What is a standard drink? | https://www.niaaa.nih.gov/alcohols-effects-health/what-standard-drink | BC17, BC-T14 |
| 123 | Perry, Doroudgar & Van Dyke 2017, Ethanol forensic toxicology, J. Am. Acad. Psychiatry Law 45:429 | https://jaapl.org/content/jaapl/45/4/429.full.pdf | BC17, BC18, BC-T14 |
| 124 | Searle 2015, Alcohol calculations and their uncertainty, Med. Sci. Law 55:58 | https://pmc.ncbi.nlm.nih.gov/articles/PMC4361698/ | BC17 |
| 125 | NIAAA, Understanding the dangers of alcohol overdose | https://www.niaaa.nih.gov/publications/brochures-and-fact-sheets/understanding-dangers-of-alcohol-overdose | BC17 |
| 126 | Weiss 2023, Is the one-compartment model with first order absorption a useful approximation?, Pharm. Res.; with MetricGate docs, Bateman function | https://pmc.ncbi.nlm.nih.gov/articles/PMC10547630/ | BC19 |
| 127 | Hodgman & Garrard 2012, A review of acetaminophen poisoning, Crit. Care Clin. 28:499 | https://www.lvhn.org/sites/default/files/uploads/research_education/EmergencyMedicineMedStudents/Hodgman%20MJ%20-%20A%20review%20of%20acetaminophen%20poisoning.pdf | BC19, BC-T15 |
| 128 | Merck Manual Professional, Acetaminophen poisoning (rev. 2025) | https://www.merckmanuals.com/professional/injuries-poisoning/poisoning/acetaminophen-poisoning | BC19, BC-T15 |
| 129 | Dart et al. 2023, Management of acetaminophen poisoning in the US and Canada: a consensus statement, JAMA Netw. Open 6(8) | https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2808062 | BC19, BC-T15 |
| 130 | Gundert-Remy 2015, Pharmacokinetic and pharmacodynamic effects of caffeine, EFSA stakeholder meeting slides | https://www.efsa.europa.eu/sites/default/files/event/documentset/150305-p09.pdf | BC19 |
| 131 | Almadani, Vorstenbosch, Davison & Murphy 2021, Wound healing: a comprehensive review, Semin. Plast. Surg. 35:141 | https://pmc.ncbi.nlm.nih.gov/articles/PMC8432991/ | BC20, BC-T16 |
| 132 | Guo & DiPietro 2010, Factors affecting wound healing, J. Dent. Res. | https://pmc.ncbi.nlm.nih.gov/articles/PMC2903966/ | BC20 |
| 133 | CDC, Tetanus: clinical overview | https://www.cdc.gov/tetanus/hcp/clinical-overview/index.html | BC21, BC-T17 |
| 134 | Hussain et al. 2024, Clostridial myonecrosis review, Microorganisms 12(7):1464 | https://pmc.ncbi.nlm.nih.gov/articles/PMC11278868/ | BC21, BC-T17 |
| 135 | Alemayehu, Azene & Mihretie 2023, surgical site infection cohort, BMC Infect. Dis. | https://pmc.ncbi.nlm.nih.gov/articles/PMC10193810/ | BC21 |
| 136 | Kumar et al. 2006, Duration of hypotension before initiation of effective antimicrobial therapy, Crit. Care Med. 34:1589 (abstract only) | https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=TITLE:%22Duration%20of%20hypotension%20before%20initiation%20of%20effective%20antimicrobial%20therapy%22&resultType=core&format=json | BC21, BC-T17 |

### 5.5 Slow world

| # | Title | URL | Used for |
|---|---|---|---|
| 137 | USDA FSIS, How temperatures affect food (fact sheet, hosted copy) | https://www.healthygallatin.org/wp-content/uploads/2013/04/FSHTempAffects.pdf | SW1 |
| 138 | Ratkowsky, Olley, McMeekin & Ball 1982, Relationship between temperature and growth rate of bacterial cultures, J. Bacteriol. 149:1 (abstract) | https://pmc.ncbi.nlm.nih.gov/articles/PMC216584 | SW2, SW-T3 |
| 139 | FAO, Quality and quality changes in fresh fish, ch. 6 (relative rate of spoilage) | https://www.fao.org/4/v7180e/v7180e07.htm | SW2, SW-T3 |
| 140 | Impact of storing condition on staling and microbial spoilage behavior of bread, Foods 2021 (PMC7824337) | https://pmc.ncbi.nlm.nih.gov/articles/PMC7824337/ | SW3, SW-T1 |
| 141 | Cornell CALS Milk Quality Improvement Program 2023, Pasteurized versus ultra-pasteurized milk (fact sheet) | https://cals.cornell.edu/sites/default/files/2023-04/pasteurization-fact-sheet-v1.pdf | SW4, SW-T4 |
| 142 | Machado et al. 2017, Frontiers in Microbiology 8:302 (milk shelf life) | https://www.frontiersin.org/journals/microbiology/articles/10.3389/fmicb.2017.00302/full | SW4 |
| 143 | Goat Journal, Difference between sour milk and spoiled milk (homesteading magazine, not peer-reviewed) | https://goatjournal.iamcountryside.com/home-dairy/difference-between-sour-milk-and-spoiled-milk/ | SW4, SW-T4 |
| 144 | FAO/WHO 2006, Benefits and potential risks of the lactoperoxidase system of raw milk preservation | https://www.fao.org/4/a0729e/a0729e00.pdf | SW4 |
| 145 | FAO, Guidelines for slaughtering, meat cutting and further processing (T0279E) | https://www.fao.org/4/t0279e/t0279e03.htm | SW5, SW-T2 |
| 146 | Koutsoumanis et al. 2006, Appl. Environ. Microbiol. 72:124 (ground pork shelf life) | https://pmc.ncbi.nlm.nih.gov/articles/PMC1352189/ | SW5, SW-T2 |
| 147 | EFSA BIOHAZ Panel 2016, Growth of spoilage bacteria during storage and transport of meat, EFSA Journal 14(3):4523 | https://pmc.ncbi.nlm.nih.gov/articles/PMC11848000/ | SW5 |
| 148 | Principles of preservation of shelf-stable dried meat products (FSIS document, hosted by Univ. of Wisconsin) | http://www.meathaccp.wisc.edu/validation/assets/Principles%20for%20preservation.pdf | SW6, SW-T5 |
| 149 | USDA FSIS, Ham and food safety (hosted by Utah State Extension) | https://extension.usu.edu/fscreate/files/2020-22-staff/FFU_Ham_and_Food_Safety.pdf | SW6, SW-T5 |
| 150 | van Aalst et al. 2022, Synthetic and Systems Biotechnology (ethanol yield and glycerol) | https://pmc.ncbi.nlm.nih.gov/articles/PMC8792080/ | SW7, SW-T6 |
| 151 | Vidgren, Multanen, Ruohonen & Londesborough 2010, Appl. Environ. Microbiol. (ale and lager temperatures) | https://pmc.ncbi.nlm.nih.gov/articles/PMC2878602/ | SW8 |
| 152 | Thesseling et al. 2019, A hands-on guide to brewing and analyzing beer in the laboratory | https://pmc.ncbi.nlm.nih.gov/articles/PMC9286407/ | SW8, SW-T6 |
| 153 | Trinity College Dublin 2023, Drunk adventures in sixteenth-century brewing (news, research by S. Flavin) | https://www.tcd.ie/news_events/articles/2023/drunk-adventures-in-sixteenth-century-brewing/ | SW8 |
| 154 | Ganucci et al. 2018 (PMC6053494), wine co-fermentations at 28 °C | https://pmc.ncbi.nlm.nih.gov/articles/PMC6053494 | SW9, SW-T7 |
| 155 | Panprivech et al. 2015, Cabernet Sauvignon cold-soak study (PMC6272619) | https://pmc.ncbi.nlm.nih.gov/articles/PMC6272619/ | SW9 |
| 156 | Coleman, Fish & Block 2007, Temperature-dependent kinetic model for nitrogen-limited wine fermentations, Appl. Environ. Microbiol. (abstract only) | https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=PMCID:PMC2074923&resultType=core&format=json | SW9, SW-T7 |
| 157 | Ciani et al. 2016, Frontiers in Microbiology 7:555 (yeast at 35 °C) | https://www.frontiersin.org/journals/microbiology/articles/10.3389/fmicb.2016.00555/full | SW9, SW-T7 |
| 158 | USDA Forest Products Laboratory 1999, Air drying of lumber, FPL-GTR-117 | https://research.fs.usda.gov/download/treesearch/5730.pdf | SW10 |
| 159 | Simpson & Hart 2000, Estimates of air drying times for several hardwoods and softwoods, FPL-GTR-121 | https://www.fpl.fs.usda.gov/documnts/fplgtr/fplgtr121.pdf | SW10, SW-T8 |
| 160 | Cornell Cooperative Extension (Warren Co.), Storing and drying firewood | https://warren.cce.cornell.edu/natural-resources/heating-with-wood/storing-and-drying-firewood | SW10, SW-T8 |
| 161 | Jerkovic et al. 2025, Drying performance of fabrics on the human body (PMC12155929) | https://pmc.ncbi.nlm.nih.gov/articles/PMC12155929/ | SW11, SW-T9 |
| 162 | Orloff, Hay curing, baling and storage, UC ANR Intermountain Alfalfa Management, ch. 12 | https://ucanr.edu/?legacy-file=117607.pdf&legacy-file-path=sites%2FSiskiyou_County_Coop_Extension%2Ffiles%2F | SW12 |
| 163 | Sheaffer et al. 2022, Managing risk at first alfalfa harvest, UMN Crop News | https://blog-crop-news.extension.umn.edu/2022/05/managing-risk-at-first-alfalfa-harvest.html?m=1 | SW12 |
| 164 | Teagasc 2026, Finding the optimum summer stocking rate; Grass growth rebounds in 2025; Grass10 grazing management | https://teagasc.ie/news--events/daily/finding-the-optimum-summer-stocking-rate/ | SW13, SW-T10 |
| 165 | AHDB, Understanding grass growth for beef rotational grazing | https://ahdb.org.uk/knowledge-library/understanding-grass-growth-for-beef-rotational-grazing | SW13, SW-T10 |
| 166 | Penn State Extension, Four steps to rotational grazing | https://extension.psu.edu/four-steps-to-rotational-grazing | SW13 |
| 167 | Yates et al. 2019, Frontiers in Plant Science 10:344 (ryegrass leaf elongation) | https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2019.00344/full | SW13 |
| 168 | Miller, Lanier & Brandt 2001, Using growing degree days to predict plant stages, Montana State Univ. Extension MT200103AG | https://landresources.montana.edu/soilfertility/documents/PDF/pub/GDDPlantStagesMT200103AG.pdf | SW14, SW-T11 |
| 169 | Acevedo, Silva & Silva, Wheat growth and physiology, FAO Bread Wheat (Y4011E) | https://www.fao.org/4/y4011e/y4011e06.htm | SW14 |
| 170 | Shedge & Zubair, Postmortem changes, StatPearls (NBK539741) | https://www.ncbi.nlm.nih.gov/sites/books/NBK539741/ | SW15, SW-T12 |
| 171 | Eden, Das & Thomas, Algor mortis, StatPearls (NBK534875) | https://www.ncbi.nlm.nih.gov/sites/books/NBK534875/ | SW15, SW-T12 |
| 172 | Megyesi, Nawrocki & Haskell 2005, Using accumulated degree-days to estimate the postmortem interval, J. Forensic Sci. 50:618 (abstract only) | https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.1520/JFS2004017&resultType=core&format=json | SW16 |
| 173 | Wilson et al. 2019, Forensic Sci. Int.: Synergy 1:204 (quotes the Megyesi equation) (secondary) | https://pmc.ncbi.nlm.nih.gov/articles/PMC7219161 | SW16, SW-T13 |
| 174 | Smith et al. 2023, J. Forensic Sci. commentary on the Megyesi error term (PMC10092795) | https://pmc.ncbi.nlm.nih.gov/articles/PMC10092795/ | SW16 |
| 175 | Strack & Smith 2023, Forensic Sci. Int.: Reports 7:100321 (applies the Vass 2011 formula) | https://eprints.bournemouth.ac.uk/38695/1/1-s2.0-S2665910723000166-main.pdf | SW16, SW-T13 |
| 176 | University of Tennessee LEIC, Post-mortem interval calculation (after Vass 2011) | https://leic.tennessee.edu/post-mortem-interval-calculation | SW16 |
| 177 | Schlosberg et al. 2025 (PMC12444604), pig carcass decomposition in Queensland | https://pmc.ncbi.nlm.nih.gov/articles/PMC12444604/ | SW16 |
| 178 | Clausen 2010, Biodeterioration of wood, Wood Handbook ch. 14, FPL-GTR-190 | https://research.fs.usda.gov/download/treesearch/37418.pdf | SW17, SW-T14 |
| 179 | Woodward, Hatfield & Lebow 2011, Comparison of wood preservatives in stake tests: 2011 progress report, FPL (abstract page) | https://research.fs.usda.gov/treesearch/39936 | SW17, SW-T14 |

### 5.6 Contacts and water on bodies

| # | Title | URL | Used for |
|---|---|---|---|
| 180 | RoyMech, Coefficients of friction (table) | https://roymech.org/Useful_Tables/Tribology/co_of_frict | K1, K-T2, K-T4 |
| 181 | Engineers Edge, Coefficient of friction table | https://www.engineersedge.com/coeffients_of_friction.htm | K1, K2, K-T4, K-T10 |
| 182 | The Physics Factbook, Coefficients of friction for ice (secondary compilation) | https://hypertextbook.com/facts/2004/GennaAbleman.shtml | K2 |
| 183 | Schulson 2015, Friction and fracture of ice (review, BSEE TAP 714ac) | https://www.bsee.gov/sites/bsee.gov/files/tap-technical-assessment-program/714ac.pdf | K2 |
| 184 | Liefferink, Hsia, Weber & Bonn 2021, Friction on ice: how temperature, pressure, and speed control the slipperiness of ice, Phys. Rev. X 11:011025 | https://journals.aps.org/prx/pdf/10.1103/PhysRevX.11.011025 | K2, K-T10 |
| 185 | Rocscience RocFall2 docs, Dynamic friction and rolling friction coefficients table (Horn & Deere 1962; Azzoni et al. 1995; Chau et al. 1998) | https://www.rocscience.com/help/rocfall/documentation/slope/materials/reference-tables/dynamic-friction-and-rolling-friction-coefficients-table | K3, K10, K-T1 |
| 186 | Derler & Gerhardt 2012, Tribology of skin, Tribol. Lett. 45:1 (abstract only) | https://research.tue.nl/en/publications/tribology-of-skin-review-and-analysis-of-experimental-results-for | K4 |
| 187 | Beschorner, Albert & Redfern 2016, Required coefficient of friction during level walking is predictive of slipping, Gait Posture 48:256 (abstract only) | https://pubmed.ncbi.nlm.nih.gov/27367937/ | K5, K-T10 |
| 188 | Rocscience RocFall2 docs, Coefficient of restitution table (Pfeiffer & Bowen 1989; Robotham et al. 1995) | https://www.rocscience.com/help/rocfall/documentation/slope/materials/reference-tables/rocscience-coefficient-of-restitution-table-in-rocfall | K6, K-T9 |
| 189 | Sandeep et al., Experimental study on the coefficient of restitution of grain against block interfaces, Can. Geotech. J. (accepted manuscript) | https://discovery.ucl.ac.uk/10082944/1/Sandeep_et_al_TRS_Manuscript_Revised_COPY.pdf | K7 |
| 190 | Güttler, Heißelmann, Blum & Krijt 2012, Normal collisions of spheres: a literature survey (arXiv 1204.0001) | https://arxiv.org/pdf/1204.0001 | K7, K9 |
| 191 | Alonso-Marroquin et al., Mechanical properties of polygonal wood particles (arXiv 1405.3049), Tables 3.2–3.3 | https://arxiv.org/pdf/1405.3049 | K8 |
| 192 | OpenStax, University Physics Vol. 1, 11.1 Rolling motion | https://openstax.org/books/university-physics-volume-1/pages/11-1-rolling-motion | K10, K-T1 |
| 193 | Moore et al., Mechanics Map 6.2: Slipping vs tipping (LibreTexts) | https://eng.libretexts.org/Bookshelves/Mechanical_Engineering/Mechanics_Map_(Moore_et_al.)/06:_Friction_and_Friction_Applications/6.02:_Slipping_vs_Tipping | K10, K-T2 |
| 194 | Engineering ToolBox, Angles of repose (secondary) | https://www.engineeringtoolbox.com/dumping-angles-d_1531.html | K11, K-T3 |
| 195 | Wikipedia, Angle of repose (secondary) | https://en.wikipedia.org/wiki/Angle_of_repose | K11, K-T3 |
| 196 | Nave et al. 2020, Wind dispersal of natural and biomimetic maple samaras (arXiv 2010.12553) | https://arxiv.org/pdf/2010.12553 | K12, K-T8 |
| 197 | Cummins et al. 2018, A separated vortex ring underlies the flight of the dandelion, Nature 562:414 | https://strathprints.strath.ac.uk/79669/1/Cummins_etal_Nature_2018_A_separated_vortex_ring_underlies_the_flight.pdf | K12 |
| 198 | Howison et al. 2019, Physics driven behavioural clustering of free-falling paper shapes, PLoS ONE | https://pmc.ncbi.nlm.nih.gov/articles/PMC6594606/ | K13, K-T7, K-T8 |
| 199 | Andersen, Pesavento & Wang 2005, Unsteady aerodynamics of fluttering and tumbling plates, J. Fluid Mech. 541:65 | https://dragonfly.tam.cornell.edu/publications/2005_JFM_Andersen_Pesavento_Wang_a.pdf | K13, K-T7 |
| 200 | Biviano & Jensen 2025, Settling aerodynamics is a driver of symmetry in deciduous tree leaves, J. R. Soc. Interface | https://pmc.ncbi.nlm.nih.gov/articles/PMC12055283/ | K13 |
| 201 | Nagahiro & Hayakawa 2004, A theoretical and numerical approach to 'magic angle' of stone skipping (arXiv physics/0411125); Clanet, Hersen & Bocquet 2004 not opened | https://arxiv.org/pdf/physics/0411125 | K14, KW-T3 |
| 202 | Bocquet 2003, The physics of stone skipping, Am. J. Phys. 71:150 (arXiv physics/0210015) | https://arxiv.org/pdf/physics/0210015 | K14, KW-T3, KW-T4 |
| 203 | Wikipedia, Stone skipping (record, citing Guinness) (secondary) | https://en.wikipedia.org/wiki/Stone_skipping | K14 |
| 204 | KnifeThrowing.info, Physics of knife throwing | https://www.knifethrowing.info/physics_of_knife_throwing.html | K15, K-T5, K-T6 |
| 205 | World Axe Throwing League, 2025 gameplay rules | https://worldaxethrowingleague.com/2025-gameplay-rules/ | K15, K-T5 |
| 206 | Cross 2004, Physics of overarm throwing, Am. J. Phys. 72:305 (read from a local copy) | https://www.physics.usyd.edu.au/~cross/PUBLICATIONS/25.%20Throwing.pdf | K15, K-T5 |
| 207 | Timson & Coffman 1984, Body composition by hydrostatic weighing at total lung capacity and residual volume, Med. Sci. Sports Exerc. 16:411 (abstract only) | https://pubmed.ncbi.nlm.nih.gov/6493022/ | K16, KW-T5 |
| 208 | Askew, Formenti & Minetti 2012, Limitations imposed by wearing armour on Medieval soldiers' locomotor performance, Proc. R. Soc. B 279:640 (abstract only) | https://pubmed.ncbi.nlm.nih.gov/21775328/ | K17, KW-T5 |
| 209 | Wikipedia, Plate armour (secondary) | https://en.wikipedia.org/wiki/Plate_armour | K17 |
| 210 | Kretschmann 2010, Mechanical properties of wood, Wood Handbook ch. 5, FPL-GTR-190, Table 5-3a (read from a local copy) | https://www.fpl.fs.usda.gov/documnts/fplgtr/fplgtr190/chapter_05.pdf | K18, KW-T1, KW-T9 |
| 211 | Ozgul, Ship construction and stability II: calculating KB, BM and metacentric diagrams (course notes) | https://ilginozgul.academy/wp-content/uploads/2021/12/ship-stb.-wihtout-video.pdf | K19, KW-T1, KW-T2 |
| 212 | Morais et al., Understanding the importance of drag coefficient assessment ... of swimmers, J. Hum. Kinet., Table 1 | https://pmc.ncbi.nlm.nih.gov/articles/PMC11079927/ | K20, KW-T6 |
| 213 | Techet / Epps, MIT 2.016 Hydrodynamics, Derivation of added mass around a sphere | https://web.mit.edu/2.016/www/handouts/Added_Mass_Derivation_050916.pdf | K20 |
| 214 | Duez, Ybert, Clanet & Bocquet 2007, Making a splash with water repellency, Nature Physics 3:180 (arXiv cond-mat/0701093) | https://arxiv.org/pdf/cond-mat/0701093 | K21, KW-T7 |
| 215 | Wikipedia, Dispersion (water waves) (secondary) | https://en.wikipedia.org/wiki/Dispersion_(water_waves) | K22, KW-T7, KW-T8 |
| 216 | Wikipedia, Capillary wave (secondary) | https://en.wikipedia.org/wiki/Capillary_wave | K22, KW-T7 |
| 217 | Buyrope (UK), Natural manila rope data sheet (EN ISO 1181 break loads) | https://www.buyrope.co.uk/pdf/BR-PDF-MANILA-ROPE-Data-Sheet.pdf | K23, K-T11 |
| 218 | Engineering ToolBox, Manila ropes: strength (safety factor 12) (secondary) | https://www.engineeringtoolbox.com/manila-rope-strength-d_1512.html | K23 |
| 219 | CSU Fresno, AGET 120 course notes: Rope (adapted from a US Army rigging manual) | https://www.agedweb.org/courses/aget120/rope.pdf | K23, K-T11 |
| 220 | H-Lift, Grade 30 proof coil chain (ASTM A413) | https://www.h-lift.com/products/grade-30-proof-coil-chain-astm-a413 | K24 |

## 6. Cost

Roadmap 107 list B gives three rules: (1) each process has its own clock; (2) compute
only where something happens; (3) skip ahead when it is quiet, with the exact formula.
The table applies them to each domain. "Air cell" is the 4 × 4 × 4 voxel cell of
`_law_air`: 1/64 of the voxel count. Counts are per tick (25 ms) unless the clock
column says otherwise. All figures are estimates from the data sizes; nothing was
timed.

| Domain | Own clock | Where it runs | State | Work per active unit | Quiet skip |
|---|---|---|---|---|---|
| Wind (W) | Every tick near people, fire and smoke; otherwise on demand | Air cells in the windows that smoke and heat already use; the outdoor edge | Per world: mean speed, direction, z0 per ground type, one gust series. Per air cell: the 4 values that exist now. | Per air cell: one semi-Lagrangian gather per carried field. Per person outdoors: one drag sum over the cells facing the wind. Per exposed hot face: one h(v) look-up. | Outdoors with nothing to carry, the wind is a formula (log profile × gust(t)); read it at a point when someone asks. Cost ≈ 0. |
| Gases (GS) | Every tick where a species is above its own cutoff; smell once a second | Air cells | ≈ 6 float32 species per air cell ≈ 6/64 ≈ 0.1 of one voxel field | Per active air cell and species: advection, density overturn and diffusion (as smoke now). Per person, once a second: read ≈ 6 numbers at the nose; update 3 FED sums. | A uniform species has no gradient and costs nothing (the O2 window rule). Venting is an exponential; skip it. |
| Reactions (R) | Fast rows every tick; slow rows as events | Touching faces of two reactants only | One table row per pair, read-only | Per active face: one exp() (or a look-up by temperature bin), one mass move, one heat add | Slow rows (rust, pyrolysis below 200 °C, drying) are scheduled: compute the time to the next visible change once, and redo it only when the face's temperature or wetness changes band. |
| Body clocks (BC) | Once a second (every 40 ticks); faint and breath stay per tick | Inside each person | ≈ 12 numbers per person: S, C phase, water, energy, 4 limb reserves, core T, BAC, 2 drug amounts; 2 per wound | ≈ 50 arithmetic operations per person per second | While asleep or idle, solve each clock for the time it crosses its next threshold and jump there. A quiet 8 h night ≤ 100 steps (BC-T2). |
| Slow world (SW) | Events only | Each perishable or growing object | Per object: stage, rate, time of next stage. One heap. | One heap insert, O(log n), when the stage changes or the object's temperature or wetness leaves its band | Nothing per tick for an object at rest. 1,000 loaves cost 1,000 pending events and no work while the larder is steady (SW-T15). |
| Contacts (K) | Every tick while moving; sub-steps at hard contacts | Each rigid or bonded object that is awake | Per rigid object ≈ 20 numbers (position, orientation, velocity, spin, inertia). Per bond: 2 ends, rest pose, strength. | Per object: integrate + one contact pass. Per bond: one force. Per light piece (leaf): a drag update. | A body at rest sleeps (Bullet's island rule, already used for air in `_law_o2`); it wakes when touched. |
| Water on bodies (K) | Every tick for a body in water; waves only where the surface moves | Submerged cells of each body; pond surface columns | Per pond column: height and flow (2 numbers) | Per body: count submerged cells (buoyancy) and one drag term. Per active surface column: one shallow-water update. Waves need sub-steps: c = √(g·h) is 3.1 m/s at 1 m depth, so a 5 cm column needs dt ≤ 0.016 s, 2 sub-steps a tick; 3 at 2 m depth (derived). | A flat pond sleeps; a resting float sleeps. |

**What a quiet day costs.** A day is 3,456,000 ticks. At the roadmap target of more
than 10 runs a second, no domain may do work per tick for a thing that is not
changing. With the skips above, a village at night costs about: wind on demand only,
gases zero (nothing is above cutoff), reactions zero (rust is scheduled), body clocks
one jump per sleeper, slow world one pending event per object, contacts zero (all
asleep). That is the target the BC-T18 and SW-T15 tests guard.
