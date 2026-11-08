# Realism map: what Mark_1 can simulate, what it costs, and what to leave alone

Date: 2026-10-18. Tree: base18.

Sources: nine domain studies (fire, water, solids, bodies, senses, matter, living, environment, energy) and one critic pass. Every probe ran on this laptop under the shared lock. No repo file was edited. Probes and raw numbers are in `/home/nisargparikh/.cache/mark1-build/realism/`.

---

## 1. The short answer

We can simulate most of what book decisions turn on, from real laws, on the CPU, with the GPU left for the models. That includes:
- a room fire with a hot layer, flames of the right height, radiant heat and real smoke;
- bodies that balance, fall, tire, bleed and burn by measured laws;
- sound and light that decide who hears and sees whom;
- water as films, vessels, ponds and rivers;
- drinks, poisons, alcohol, powder and blasts;
- the slow world (hunger, healing, spoilage, seasons) on schedules that cost nothing while nothing changes.

Almost every one of these laws costs microseconds per person, or about 1 ms per tick. What keeps it affordable is one rule: put each thing at the cheapest level that keeps its facts. That means a 20 cm grid for gas and light, an event for a blast or a door, a closed form for a night, and a record for a cup.

What we should not try is anything that resolves waves, folds or shocks at 5 cm: CFD combustion, sound waves, global illumination, 3-D water, per-tick elasticity, detonation, cloth drape, muscle-driven ragdolls, per-voxel biology, or weather as a fluid. Each costs 10 to 1000 times more and decides nothing the cheap law misses.

The real limit today is not any law. It is two engineering facts:
- each person triggers passes over the whole lattice;
- ticks cannot cover hours.

Those two fixes come before most of the realism.

### What the research found that changes the plan

| finding | measured | consequence |
|---|---|---|
| The room fire is upside down | house_fire at 30 s: 121 °C near the floor, 28 °C under the ceiling. The plume over the bed falls to 20 °C by 2 m (correlation: about 290 K above ambient). | Heat is not carried by moving gas. "Crawl low", the deadly top of a doorway, banking down and flashover cannot come out right until it is. |
| A 2 mm puddle is a wall | A walker stops at a water strip and stays there for 500 ticks (`motion.py:930`). | A bucket thrown at a fire can trap the thrower. The fix is small. |
| Still water is the most expensive thing per cell | A still 2 m pond costs 130 ms/tick. | No scene can hold more than a bathtub. An exact sleep fixes it. |
| The thrown-plank bug is numerical | Plank leaves the floor with 3.3x its energy. At h/1.25 it is a normal lossy bounce. | The contact damping is missing from the step bound. It is a one-line fix plus tests, and friction is not the cause. |
| The stuck seeds are the missing balance reflex | Seed 56: an unconscious man over a 10 cm gap, relaunched every 3 ticks, physiology frozen. Seed 86: a man at the lip. Seed 119: a held thing counted as support. | These are not three bugs. They are one missing layer. |
| Falls are about 3x too lethal and do the wrong harm | A 2 m feet-first drop knocks a man out; 4 m kills. Survivors average 5.2 m. Nothing breaks. | Every ledge choice is judged wrongly. |
| Each person sweeps the whole world | One idle person: 14.8 ms at 0.4 M voxels, 77 ms at 1.6 M, 250-331 ms at 5.1 M. | This, not memory, limits world size. |
| A night costs hours of compute | house4 runs at 0.76x real time, so 8 h of sim takes 5-10 h. | "Wait", "sleep" and "until dawn" need an event-driven skip. |
| Every line is a shout | `senses.py:1045` speaks every line at 90 dB. | Nobody can whisper or overhear by accident. |
| Light does not exist | Sight only checks what blocks the view. | A dark room hides nothing. |

Two earlier notes needed correcting:
- **Cold.** The sim can already go down to 0 °C. What keeps it warm is that the heat sinks pull toward a fixed 20 °C.
- **The forge.** Its slowness is not a missing pore-radiation term. That term already exists (`heat.py:493-497`), so the forge needs a fresh diagnosis.

### What the books ask for

The v2 case set has 2,335 cases: Locke 868, DCC 625, Lattes 430, Treasure Island 412. Counts below are keyword hits in situation plus options, not verified physics decisions.

| theme | cases | theme | cases |
|---|---|---|---|
| money, ownership | 582 | firearms | 169 |
| door | 376 | wounds, bleeding | 170 |
| run, flee | 358 | climb | 139 |
| fight | 276 | carts, horses | 134 |
| boats, ships | 256 | wine, brandy | 132 |
| documents (letters, maps, ledgers) | 227 | locks, keys | 129 |
| window, glass | 210 | falls | 123 |
| drink | 194 | disguise, appearance | 122 |
| bombs, explosives (DCC) | 188 | sleep | 116 |

About 12% of options carry a fire or heat act. About 5% of key facts are about environment (dark, weather, cold, digging, terrain, time).

**Note:** Beware of Chicken, Mother of Learning, Science of Cultivation and The Wandering Inn are **not** in the v2 set. Crops and magic packs therefore rest on the north star, not on data yet.

---

## 2. The principle of levels

Every phenomenon is decided at one of four levels. Each runs at a different cost. Each level has its own author.

| level | when it runs | what it decides | who writes it | cost rule |
|---|---|---|---|---|
| **World law** | every tick, or on change, only where something can change | what happens: heat, flow, support, tipping, tissue damage, sound and light spreading | we write the law | microseconds to about 1 ms; sleeps when quiet |
| **Reflex** | every tick, below the mind | what a body does without deciding: shift weight, step, grab, pull back from pain, hold breath, gasp, startle, wake, open a tired grip, raise the voice in noise | we write the reflex; each person's numbers come from their sheet | 0 model calls; runs only when triggered |
| **Mind** | at decision points, gated by salience | what a person chooses: where to step, whether to cross, carefully or not, shout or whisper, douse or flee | the learned mind or the rules, picking from a menu we write | the only place model calls happen |
| **Offscreen summary** | on a schedule or a time skip | what happened while nobody watched: a night, a harvest, a rumour, a fire in an empty house | closed forms and zone models we write | microseconds per object per event |

### The balance example (the user's 2026-10-18 point)

A real body does not put full weight on a foot over a gap. Here is how that splits across the levels:

1. **World law, every tick: support and tipping.** Weight goes only on contacts that can bear it. The extrapolated centre of mass is compared with the hull of those contacts (XcoM = x + v/ω0, Hof 2005). Once the centre leaves the hull, the body tips as a rigid pendulum about the hull edge, under real g. This is exact, given support. Prototype cost: 157 µs per person, against 218 µs for today's check.
2. **Reflex, every tick: what keeps a person up.**
   - Take the load off a foot that is over nothing (90-120 ms latency).
   - Step to the nearest firm cell if the capture point is within reach.
   - Grab a firm hold within arm's reach.
   - Otherwise fall with the arm out or roll.
   - An unconscious or sleeping body has no reflex. It crumples where it stands; it does not go over like a plank.
   - Latency and step length come from each person's sheet. Drink, fatigue and injury degrade them.
3. **Mind: the real choices.** Whether to step into the gap at all. Whether to cross the plank. Whether to walk "carefully" (short steps, test the footing, hand on the rail, wider margin). Whether to jump.
4. **Offscreen:** nothing. Balance happens only in scenes.

### One contract for every reflex

The domains proposed about ten reflexes separately: balance, grab, pain withdrawal, breath-hold, the drowning response, the cold-shock gasp, startle, raising the voice in noise, waking, and a grip opening on fatigue. They should share one contract:
- latency per person, from the sheet, drawn deterministically;
- **suppressed by chosen intent:** a foot goes into a gap only if he chose that step; "grab the burning pan anyway" overrides the withdrawal;
- degraded by drink, fatigue and sleep debt;
- absent when the person is unconscious;
- 0 model calls;
- each firing writes an event, so the mind can perceive what its body did.

---

## 3. The map, domain by domain

**Legend**
- **Fidelity:** *exact* (an exact or fitted physical law), *keeps facts* (an approximation that keeps the facts a scene is about), *stand-in* (a fitted or authored number), *skip*.
- **Level:** W = world law every tick; W-ev = world law on an event or a change; R = reflex; M = mind decision; O = offscreen summary.
- **P** = priority (1 is highest). **Effort:** S, M, L, XL.

Where several domains proposed the same thing, it appears once, under one owner. The merges:
- burns → bodies;
- water into wood, and oil slopover → water;
- fire as a percept, all of light, and fire crackle → senses;
- balance, carrying and climbing → bodies;
- body energy, hunger and thirst → bodies;
- soil and digging, ice and cold, calendar and sun, and rain → environment;
- drinks below one voxel, poisons, food, spoilage, rust and rot, and glow colour → matter;
- the gas carrier for steam, odours, species, qi, disease and wind → the fire domain's gas-velocity item, since all of them depend on it.

### 3.1 Fire

| phenomenon | now | the law | cost | fidelity | level | P | effort |
|---|---|---|---|---|---|---|---|
| **Gas carries heat** (plume, ceiling jet, hot layer, doorway flow) | Room is hottest at the floor. Heat crosses gas only by an eddy stand-in and a capped rise. The 20 cm air field carries smoke only, and it is lossy (it pumped mass when it tried to carry O2). | Buoyant low-Mach velocity on the 20 cm air cells. Pressure projection solved to a tolerance (multigrid or PCG). Limited upwind flux advection (van Leer) of heat, smoke and O2. Substeps in the flame zone. Checked against the Heskestad plume, Alpert ceiling jet, Zukoski filling, MQH and the doorway neutral plane. | 0.7-1.25 ms measured, but with 20 unconverged iterations; expect 2-5 ms. Replaces part of the air, smoke and O2 laws (25 ms today). | keeps facts (20 cm is coarse; entrainment off by tens of %) | W | 1 | L |
| **Flame height** (F25) | Combustion only in fuel cells. A 5 kW spill never reaches the bed 20 cm up. Scenes start pre-lit. | World-law fire labelling (not perception's boxes). Heskestad L = 0.235Q^0.4 − 1.02D; ×2 against a wall, ×4 in a corner; flame runs out under the ceiling. Flame cells heat what they touch, paid from that fire's own heat. | under 1 ms | keeps facts | W | 1 | M |
| **Radiant heat across air** (G7) | Missing. Radiation acts only between touching faces; the rest is shed from the world. | Multi-point source along each flame, capped near the flame by the solid-flame model. Hot surfaces above 500 °C clustered per 20 cm cell (with a cap on the count). Hot-layer term once gas moves. Line-of-sight rays with smoke attenuation. Energy lands in the receivers' skins, people included. | 2.2-5.5 ms per pass, every 4-8 ticks ≈ 1 ms/tick with 4 flame points. Measure on a fully involved room. | keeps facts | W every 4-8 ticks | 1 | M |
| **Smoke opacity, soot** (F19) | Smoke is 39x too clear. | Per-fuel soot yield, 8.7 m²/g on soot; visibility K/σ (Jin, an empirical correlation). Opacity is read from soot alone. | Constant ≈0. A separate soot scalar costs about 6 ms today, cheap after the gas item. It retunes CO, coughing, Purser and smell. | keeps facts | W, read by sight | 1 (after gas) | M |
| **Fine fuels and igniters** | Only wood, leaf, rope, char and flesh burn. "Wicks" are 100 kW. | Thin materials (paper, straw, cloth, thatch): t_ig = ρcδΔT/q″. Candle 80 W, torch 1-5 kW, lamp 50-100 W, each with a flame. Blowing out = a critical air speed. | ≈0 | keeps facts | W; acts M | 1 | M |
| **Char depth, embers, glow** | Whole cell switches to char at 18% mass. Rope never chars. | Char depth from mass already lost; char added to the skin's heat path (gives the 0.8 → 0.6 mm/min slowdown). Bond strength ∝ remaining section. Glow colour from surface temperature (Draper 525 °C, cherry ~800, yellow ~1000). | ≈0 | keeps facts | W; glow is a percept | 2 (in wave 12) | S |
| **Smouldering, CO as its own gas** | Only char glows. CO is a fixed share of smoke. | Cellulosic surface oxidation below flaming (1.5-3 mm/min), turning to flame when O2 rises. CO field with yields by burning mode (smoulder yields unverified). | about 0.2 ms after gas | keeps facts | W; O for long smoulders | 2 | M |
| **Ventilation, flashover, opening a door on a fire** | O2 is not carried by flow. Flashover is impossible. | Emerges from gas velocity plus radiation. Tests: Kawagoe, Babrauskas Q_fo, MQH, layer at 500-600 °C with 20 kW/m² at the floor. Backdraft later (needs a fuel-gas species). | tests only | keeps facts | W; door acts M | 2 | M |
| **Long fires offscreen** | 2.2 s of wall time per sim second | Slow laws on longer clocks; a two-zone room model (CFAST style) handed back to the voxel laws when someone looks | µs per room; the cost is in the handover | keeps facts | O | 2 | L |
| **Wind on fire, embers** | No wind | Wind from the gas item's boundary; Thomas flame tilt; embers as light particles. Rothermel only offscreen for wildland. | small (a town ember storm is not) | keeps facts | W; O | 3 | L |
| **Clothing on fire** | Bodies have no clothing | A cloth layer per body region; rolling or a blanket cuts its O2 | small | stand-in | W; M/R | 3 | M |

### 3.2 Water

| phenomenon | now | the law | cost | fidelity | level | P | effort |
|---|---|---|---|---|---|---|---|
| **Walking through shallow water** | Any cell over 1 ml stops the step | Liquid is shoved, like the hedge exchange. Wading drag ½ρC_dAv² on the legs; buoyancy cuts the weight on the feet. | under 0.1 ms | keeps facts | W; gait R | 1 | S |
| **Still water costs nothing** (sleep for fluids) | Still 2 m pond: 130 ms/tick | Dirty set walked in the same low-first order; cached connected bodies; numba port | ≈0 when still | exact, but the sweep changes cells within a tick and head is non-local, so SLEEP-style lockstep tests are needed | W | 1 | M |
| **Puddle spread, drop gravity** | Spills end as cones whose area depends on voxel size. Drops fall at 20 m/s² at 5 cm. | Each liquid stops at its own film, h = 2λ_c sin(θ/2) (water 1-4 mm, oil under 1 mm). A creep term as the front thins. Drop gravity in SI. | unchanged | keeps facts | W | 1 | S |
| **Water soaking into wood, rope, char; a wet skin that will not light** | Water runs through the crib to the floor (F-T9 wood: −34% HRR at 10 s, real is above 50%) | Retained film per face. Sorption i = S√t (wood 10-16 g/m²s^½ along the grain, 1-7 across). Skin moisture pins the surface at 100 °C while the core stays dry. Drying by evaporation. | about 0.5 ms (estimate) | keeps facts | W; O for hours | 1 | M |
| **The boiling curve** (quench, Leidenfrost) | One heat-transfer number per liquid | Nukiyama curve per wet hot face: Rohsenow, Zuber CHF about 1.1 MW/m², Bromley film boiling past about 200 °C. F-T9 needs it, because water on 300-600 °C wood is in film boiling. | under 0.1 ms | keeps facts | W | 1 | S |
| **Water vapour and steam** | Boiled water is deleted (a ledger gap). No humidity. | One vapour field on the gas carrier. Magnus saturation; condensation releases 2260 J/g (the steam-burn fact); evaporation by mass transfer. | under 1 ms where active | exact law on an approximate carrier | W; O drying | 1 (after gas) | M |
| **Pour, fill, drink, douse** | No such acts | The hands tilt the held vessel and existing laws move the liquid. Torricelli outflow. A throw gives the drops the hand's speed. | ≈0 | exact | M; hands R | 1 | M |
| **Water on burning oil** (slopover) | Strict xfail | Steam under a liquid ejects it as drops on the existing drop path. Energy = pΔV × an efficiency fitted to the F-T9 bands. | negligible | stand-in | W | 2 | M |
| **Open water as a height field** (ponds, rivers, floods, waves) | No momentum. A narrow pipe does not throttle. | Virtual-pipes shallow-water model; seams to the lattice and to drops | 0.026 ms (2 m) to 2.5 ms (20 m) per tick, measured; volume drift 0 | keeps facts | W; O river levels | 2 | L |
| **Buoyancy and drag on things** | Cells only sink or float per column | Archimedes per bonded member plus quadratic drag against the water's velocity (gives the righting moment); hull flood fill | under 0.1 ms per 1000 members | exact | W | 2 | M |
| **A person in water** | Cannot enter water. The drowning clock is right. | Buoyancy with lung volume (0.98 vs 1.05 g/ml). Swimming = force at an O2 cost, with skill. Instinctive drowning response as a reflex: silent, cannot grab a rope. | under 0.1 ms | keeps facts | W; R; M | 2 | L |
| **Standing in a current** | None | Drag against friction, and moment about the downstream foot. This is the balance test with one more force. | flops | keeps facts | W+R; M | 2 | M |
| **Mixing fixes** (blood, solder, blends) | Blend keeps acid's density, so mass is not conserved | Volume-weighted blend density; new rows; clotting timer | ≈0 | exact | W | 3 | S |

### 3.3 Solids

| phenomenon | now | the law | cost | fidelity | level | P | effort |
|---|---|---|---|---|---|---|---|
| **Bonded things landing** (thrown-plank bug) | Contact damping is not in the step bound: plank gains 3.3x energy; a sideways throw reverses | Damping in the bound, h ≤ (2/ω)(√(1+ζ²) − ζ), counting every member that can touch. Friction capped per piece, not per member. Later: implicit soft contacts (Catto soft step or XPBD). | +30% bond substeps (plank 9 → 12 ms, crown 135 → 175 ms). Soft contacts ≈0. | exact | W (substeps) | 1 | S |
| **One material record** | Strength differs by solver; stone and wood inverted; MPM strength scaled by √stiff | One record (E, ν, ρ, tensile and compressive strength, toughness, restitution, μ per pair), read by lattice, bonds and MPM | lookups | exact data | — | 1 (blocks beams, fracture, friction, cutting, soil) | M |
| **Doors, hinges, latches, bars, locks** | Doors are cells at fill 0.97 | Named thing with one hinge axis. A quarter turn is an exact cell permutation; a sweep check means a body behind it blocks it. Latch, bolt, bar and lock hold forces; a key eid opens the lock. Holding it shut compares the pushes on each side. Breaking it down fails the latch first. Gap per angle feeds smoke and air. | event-driven | keeps facts | W-ev; M | 1 | M |
| **Beam check in place of SPAN** | Load changes nothing | Euler-Bernoulli per horizontal run. Load includes people. Fill is thickness. Bending stress against strength; sag; crushing check. | 3-6 ms per layout change | keeps facts | W-ev | 2 | M |
| **Collapse** | Columns fall straight down | Failed members go to bonds as 20 cm pieces. Offscreen: a debris cone at the angle of repose. | 30-60 ms/tick for 1-3 s (500 pieces); about 5 ms offscreen | keeps facts | W while moving; O | 2 | L |
| **Fracture** | Depends on code path and voxel size | Impact stress ρcΔv against strength from the one record; calibrated on S-T1, S-T4 and EN 12600 | lookups | keeps facts | W-ev | 2 | M |
| **Friction per pair, sliding and rolling** | 0.40 everywhere; nothing slides | μ_s and μ_k table, wet and greased modifiers. Crates and boulders become single rigid bonded pieces when pushed. Slip event. | table free; 1-3 ms per crate | exact | W | 2 | M |
| **Bond islands** | One global substep and one sleep | Each connected island gets its own substep and sleep | saves cost | exact | W | 2 (before ropes, collapse) | M |
| **Ropes** | 7.1 ms/tick for 1 m; never sleeps; no tie, cut or climb | XPBD or implicit links; tension against break load, knot at 45-55% | under 0.5 ms (estimate; measure first) | keeps facts | W; M | 2 | M |
| **Trees: test what is built** | S-T2 and K-T7 tests not written | Write the tests, fix what fails (hinge, landing) | test time | exact | W | 2 | S |
| **Chopping and cutting** | An axe blow shatters a whole 75 g cell | Mass removed per blow, along or across the grain. A notched member fails by the beam check. e_cut must be calibrated against chop times. | 0 per tick | keeps facts | W-ev; M | 2 | M |
| **Prying, levers, nails** | None | Lever statics; nail withdrawal strength (NDS) | 0 | exact | W-ev; M | 3 | S |
| **Cloth as a material** | None | CLOTH row as a thin sheet; a "covers" relation; smothering through O2; sheets tied into a rope | 0 | stand-in | W; M | 3 | S |

### 3.4 Bodies

| phenomenon | now | the law | cost | fidelity | level | P | effort |
|---|---|---|---|---|---|---|---|
| **Body numbers** (G10) | 1.5 m and 43.9 kg; arms 13% of mass each; a 0.565 m jump | Winter's segment fractions, muscle about 63 kPa, adult size. Must come before falls, carrying, buoyancy, energy stores and knockout. | 0 | keeps facts | data | 1 | M |
| **Footing and tipping** | Any sole cell over anything counts as standing; a leaf holds a man up; launch slides without rotating; an aloft body's physiology is frozen | XcoM against the hull of load-bearing contacts; tip about the hull edge; a held thing is not support; physiology runs while aloft | 157 µs/person (≈20 µs in numba); sleeps when unchanged | exact given support (SPAN is load-blind until the beam check) | W | 1 | M |
| **Balance reflex and crumple** | No posture reflex; the mind's clock is too slow | Unload a foot over a gap; compensatory step; grab; arm-out fall; unconscious bodies crumple (crouch, then topple) | about 150 µs, only when the margin is small | keeps facts | R | 1 | M |
| **Topple under gravity** | α = (1/L)sin per tick², about 10x too fast and voxel-dependent | θ″ = mgd sin θ / I_pivot | trivial | exact | W | 1 | S |
| **Fall injury by region** | One global "hurt"; no fractures | Landing force per region against tolerances (tibia, femoral neck 3-4 kN, HIC). Loss of function: a broken shin bears no load. Calibrated to survivors averaging 5.2 m and LD50 about 12-15 m. | per landing | keeps facts | W-ev | 1 (after G10) | M |
| **Fatigue and strength** | A grip holds forever | Three-compartment fatigue per limb (endurance 21.9·f^−1.98 s); critical power plus W′ | 11 µs/tick | keeps facts | W; grip opens R | 1 | M |
| **Bleeding** | Small cuts stop 30-100x too fast; femoral cuts kill 2-3x too slowly; "pressed" exists but no act sets it | Vessel map per region; flow from mean arterial pressure; clot time by vessel size; press, tourniquet and bandage acts. Pulse engine as an offline oracle only. | µs | keeps facts | W; M | 1 | M |
| **Burns** (owner) | Mean over all flesh above 55 °C, so a hand in the fire is diluted by the whole body | Thin 1-D skin (6 nodes) per surface cell, plus the Henriques integral (P 3.1e98 s⁻¹, ΔE/R 75,000 K); %TBSA with death hours later (Baux); pain per ISO 13571; withdrawal reflex. The thick 3.7 mm node undercounts a dash through flame. | about 64 µs/person while hot | exact, on the thin skin | W; R | 1 | M |
| **Gait, speed, stairs** | Walk 1.4 m/s, run 2.5; flat world (a kerb is a wall); instant starts | Speeds from the person and the ground; acceleration ≤ μg; turns v²/r ≤ μg; 2.5-D standable surfaces; crawl. One rewrite together with route bounding. | about the same as routes today | keeps facts | M gait; legs every tick | 1 | L |
| **Carrying, lifting** | One strength number, no cost | Joint torques by posture (hands, shoulder, back); Pandolf cost | 0 | keeps facts | W; M | 2 | S |
| **Climbing** | No act | Stances in static equilibrium (Bretl-Lall); holds found from lattice edges once per change | small, climbers only (the hold scan must beat 14.5 ms) | keeps facts | M target; motor every tick; grab R | 2 | L |
| **Knockout, concussion** | No path through the head | Head angular acceleration or HIC per blow; threshold per person | per blow | keeps facts | W-ev | 2 | M |
| **Breath, drowning, toxic gas** | O2 store and Purser CO done | Purser FED over CO, HCN and CO2 species; breath-hold break point; gasp | ≈0 extra | exact | W | 2 | S |
| **Core temperature** | Flesh sits at 20 °C, no metabolism | Two-node Gagge model on the person; boundary flux to the skin cells (decision 1); shivering, sweating; cold shock 1-10-1; hypothermia stages | µs | keeps facts | W at 1 Hz | 2 | M |
| **Energy and water stores** | None | float64 glycogen and fat (float32 loses a resting drain entirely). BMR × MET; muscle 18-25% efficient, the rest becomes heat; O2 ↔ energy 20.1-20.9 kJ/L; water balance. | about 50 flops per person-second | keeps facts | W at 1 Hz; O | 2 | S |
| **Sleep and alertness** | One 60 dB sample wakes anyone | Borbély two-process model; sleep inertia; sleep loss slows reflexes | trivial | keeps facts | O; M | 2 | S |
| **Infection, fever, healing** | Wounds never heal | Contamination set by the cause; bacterial load against the immune response; fever; healing at 3% → 20% → 80% strength (1 wk / 3 wk / 3 mo) | ≈0, scheduled | keeps facts | W scheduled; M | 2 | M |
| **Death as a process** | Instant at a threshold | Brain hypoxia integrated over time (window unverified); a "dying" state where acts still matter | trivial | keeps facts | W | 2 | S |
| **Cold-water immersion** | — | Cold-shock gasp reflex; core cooling; wet cloth | ≈0 | keeps facts | R; W | 3 | M |

### 3.5 Senses and knowing

| phenomenon | now | the law | cost | fidelity | level | P | effort |
|---|---|---|---|---|---|---|---|
| **Hearing cost** | 0.27 ms per sound-ear pair; 21 ms for 20 sounds × 4 ears | One numba kernel for all pairs, same law, golden SAME | about 1-2 ms for 80 pairs (estimate) | exact | W | 1 | M |
| **Speech loudness** | Every line is a 90 dB shout | Level at 1 m chosen with the line: whisper 36, normal 55-58, raised 63-65, shout 84-90 dBA. Raised voice in noise (Lombard) is a reflex. | 0 | exact | M; W; R | 1 | S |
| **Partial understanding** | Words or nothing | SII audibility in three bands: a noise, voices without words, the words; a misreading draw in the middle band | 0 | keeps facts | W | 1 | S |
| **Sustained sounds, room ambient, fire crackle** | Fire is silent; sounds live 2 ticks; one ambient level for the world | Sources with power and duration. Fire level ∝ burn rate (stand-in value). Sabine reverberant level per air region. | µs per region; needs the batch | keeps facts (fire level is a stand-in) | W; waking R | 1 | M |
| **Light** (owner) | None; every place is lit | Point sources: E = I·cosθ/d² × line-of-sight transmittance (smoke, glass). Fire sources come from the world-law fire labels, not perception's boxes. Flame efficacy 0.16-0.3 lm/W (candle). Lamps. Sky from environment. Indirect light per room by the lumen method. Magic light as lumens declared by a pack. | 20 cm grid 0.15-0.25 ms per source, cached on change; about 1-3 ms/tick | keeps facts | W on change | 1 | L |
| **Detecting vs identifying** | Full light assumed; layout sight ignores smoke | Contrast threshold against adaptation (Blackwell). A face identified at about 0.8 lux at 4 m. Self-luminous things seen through τ < 8. Layout learned only where lit or touched. | under 0.5 ms | keeps facts | W per look | 1 | M |
| **Smelling smoke** | Only coughing | A "smells burning" percept below the cough threshold, with direction and adaptation | 0 | stand-in (calibrate after the soot fix) | W; R; M | 1 | S |
| **Others' visible state** | Nobody owns it | Bleeding, drunk, burned, not breathing and unconscious as percepts, gated by light and range | ≈0 | keeps facts | per look | 1 | S |
| **How old a belief is** | Age not shown | Show the age and channel ("a minute ago, heard"); the mind discounts | 0 | exact | mind input | 2 | S |
| **Dark adaptation, glare** | None | Hecht curve (cones about 10 min, full 20-30 min); CIE disability glare | µs | keeps facts | R; W | 2 | S |
| **Heat felt; feeling a door** | Only "scorched" above 45 °C | Warm, hot and pain bands; radiant 1 / 2.5 / 10 kW/m²; a touch act | ≈0 (rides on G7) | keeps facts | W; R; M | 2 | M |
| **Touch in the dark** | None | Touched cells write layout belief; bumping into someone | 0 | exact | W | 2 | S |
| **Impact sounds by contact pair** | Whole-body minimum ring | Ring of the striking pair; floor and footwear; creeping | 0 | keeps facts (footstep levels stand-in) | W | 2 | S |
| **Waking over time** | One sample decides | Hazard rate above a personal threshold, accumulating over a sustained sound | 0 | keeps facts | R | 2 | S |
| **Hearing damage after a blast** | None | Threshold shift or eardrum rupture lowers what is heard | 0 | keeps facts | W | 2 | S |
| **Recognition, disguise** | — | Face, voice, clothing and gait as channels, each blockable (hood, dark, distance); feeds misidentification | small | keeps facts | per look | 2 | M |
| **Odour species, animal noses** | None | Scalars on the gas carrier; per-species thresholds and adaptation | about 0.05 ms per species | keeps facts | W | 3 (2 if animal minds come soon) | M |
| **Forgetting over days, news travel** | None | Power-law retention at time skips; a carrier graph with distortion per hop | µs | stand-in | O | 3 | M |
| **Limits of attention** | None | Detection threshold rises outside the focus with task load | 0 | stand-in | R | 3 | S |

### 3.6 Matter and chemistry

| phenomenon | now | the law | cost | fidelity | level | P | effort |
|---|---|---|---|---|---|---|---|
| **Solutes in liquids, cups and flasks** | One potency slot per cell | A solute vector per fluid cell (exact volume blend), plus a vessel record for contents below one voxel. A spill turns the record into lattice fluid. Pours and sips are events, so "who touched my cup" can be believed. | +0.1-0.3 ms per active solute | exact | W; M | 1 | M |
| **Alcohol** | None | Widmark with first-order uptake; elimination 0.015 %/h. Effects change reflex numbers (reaction time, sway, speech clarity). Spirits burn by a flash-point table. Disinhibition is NOT written as a rule. | ≈0 | keeps facts | W at 1 Hz → R; M | 1 | M |
| **Black powder, fuse, wet powder, a burst keg** | Nothing carries its own oxidiser | POWDER row with its own oxidiser; a moisture gate; keg pressure against wall strength becomes a low-order blast; fuse burns at 100-131 s/m and writes a "fuse burning, about N s" belief | ≈0 | keeps facts | W; M | 1 | M |
| **Blast** | None | Kingery-Bulmash by scaled distance. Shielding from UFC 3-340-02 barrier data (the acoustic barrier rule overstates cover). Confinement factor. Breakage by toughness, impulse on bodies, eardrum and lung thresholds, Gurney fragments into the projectile law. | 10.6 ms, once per blast | keeps facts | W-ev; duck R; M | 1 | L |
| **Poisons and drugs** | None | One-compartment pharmacokinetics plus a dose-response curve on named targets; antidote rows; fantasy and System poisons as data rows | ≈0 | keeps facts (fantasy rows are stand-ins) | W at 1 Hz; M | 2 | M |
| **Reaction table extension** | Fluid-on-solid rows only | Contact type, products, temperature dependence, row clocks. Burning stays its own law (decision 4). | 0.46 ms today | exact | W; O for slow rows | 2 | M |
| **Gas products** (H2, CO2, SO2) | All go to "smoke" | Species on the gas carrier; flammability windows; smell thresholds | 0.033 ms per species (coarse) | keeps facts | W | 2 (after gas) | M |
| **Acid on skin** | Excluded on purpose (Ruling 2) | Dose = contact time × concentration; rinsing stops it | ≈0 | keeps facts | W; M | 2 (needs the ruling lifted) | S |
| **Forge heat** | Metal warms about 0.1 K/s, an estimated 4-5x too slow; nothing melts | Diagnose first. Pore radiation already exists (k_rad ≈ 4 W/m·K). Suspects: the bed-to-metal face, the 30 kg block's skin and core, or the bed mean against the temperature at the face. Then bellows as a forced draft. | ≈0 | keeps facts | W; M | 2 | S |
| **Cooking** (owner) | No food | 1-D slab inside each food thing (5 cm cells lose "raw in the middle"). Cook value; protein sets at 60-75 °C; crumb done at 96-99 °C; browning above 140 °C; burnt hands over to the burn law. | under 0.05 ms for 200 items | keeps facts | W slow clock; O | 2 | M |
| **Spoilage, fermentation, rust, rot, drying** | None | Scheduled stages: Ratkowsky √ law, water-activity gate, 0.511 g ethanol per g of sugar, ISO 9223 rust, wood rot above fibre saturation | 0 per tick | keeps facts | O | 3 | S |
| **Metalwork** | None | Thing state; working windows; quench hardening (Koistinen-Marburger); tempering (Hollomon-Jaffe) | 0 | keeps facts | M; O | 3 | L |
| **Glass: shock and softening** | Cracks above 120 K; never softens | Shock threshold by thickness; viscosity curve for sagging (a stand-in) | 0 | keeps facts / stand-in | W | 3 | S |

### 3.7 Living world

| phenomenon | now | the law | cost | fidelity | level | P | effort |
|---|---|---|---|---|---|---|---|
| **Crops and pasture** (BoC) | Stand-in in the old text engine | Per-plot 2-D state: growing degree-days, FAO-56 water balance, AquaCrop yield, frost kill, a fertility stock; voxel canopy only near people; qi as a pack multiplier. As voxels a hectare would be 8 GB. | 0.18 ms per 400×400 update, once per sim-hour | keeps facts | O; M | 2 (1 after a BoC pilot) | M |
| **Trees: structure and seasons** | One hand-built tree | Space colonisation plus pipe model at generation; leaf-out and leaf-fall by degree-days; wood moisture | one-off; events | keeps facts | O; W-ev | 2 | M |
| **Animal bodies** | None; the skeleton is already data | Same body laws with a declared skeleton; balance for N feet; flocks as capsules, flesh only on contact; a hen at 5 cm is a stand-in | 2.7-3.9 ms per full body in a small room; about 1 per house until windowing | keeps facts | W; R; M | 2 | L |
| **Animal minds** | None | The learned mind with a species menu and drives. Rescorla-Wagner, habituation. Escape economics and flight zone as drives or reward, NOT as the policy. Flocking as a steering reflex. "Awakening" is a capability flag that widens the menu. | per decision | keeps facts | M; R | 2 | M |
| **Contagion** | None | Wells-Riley with a quanta species on the gas carrier; contact events; SEIR per person; town SIR offscreen | about 0.02 ms/tick | keeps facts | W; O; M | 2 (after gas and clocks) | M |
| **The corpse** | Flesh that stays as it was | Cooling from the core model; rigor and livor scheduled; decay by degree-days; a decay gas; flies as a stand-in density | ≈0 | keeps facts | W-ev; O | 2 | M |
| **Livestock life cycle** | None | Scheduled from husbandry data (laying needs 14-16 h of light; 21-day incubation); broodiness is a drive | 0 | keeps facts | O; M | 2 | S |
| **Hay that heats itself** | None | Moisture-gated microbial heat; 65 °C danger, 85 °C flames | ≈0 | keeps facts | W scheduled | 3 | M |
| **Vermin, pests** | None | Density fields with logistic growth and diffusion | small | stand-in | W every N ticks | 3 | M |
| **Wild populations, game** | None | Surplus production per region | 0 | stand-in | O | 3 | S |
| **Riding, draft animals** | None | Offscreen travel and haul tables; in-scene riding skipped | — | skip in-scene | O | 3 | XL |

### 3.8 Environment, time and scale

| phenomenon | now | the law | cost | fidelity | level | P | effort |
|---|---|---|---|---|---|---|---|
| **Per-person passes over the whole lattice** | One idle person: 14.8 / 77 / 250-331 ms at 0.4 / 1.6 / 5.1 M voxels | Window support, torque, slack, `burning()`, `T()`, labelling, the sight band and routes to boxes around bodies and changes, as SLEEP did for heat | removes cost; target about 15 ms per person at any world size | exact (golden SAME) | W | 1 | M |
| **Clocks, scheduler, skip-ahead** | No clock above the tick; a night costs 5-10 h of compute | Slow clocks; a heap with each object's next event; closed forms between events; jump while every fine law is provably asleep; the momentum design's two-channel halt test | 0.5 µs per event; a quiet night in ms | exact only when provably quiet; room closed forms are approximate | O | 1 | L |
| **Calendar, sun, sky** | Integer tick only | NOAA/Meeus sun position; sky lux bands; moon phase; diurnal temperature (Parton-Logan); weather generated or authored | µs | exact (sun), keeps facts (sky, weather) | W slow | 1 | S |
| **Outdoor temperature, below 0 °C, ice** | Cells can reach 0 °C, but sinks pull toward a fixed 20 °C; a hard floor at 0 °C; no ICE | A separate boundary temperature T_out(t) read by the sinks and by buoyancy, with a fixed reference. Remove the 0 °C floor. Quiet gates test "deviates from the boundary" or "changed last tick", not E ≠ 0. ICE row at 334 J/g; Stefan ice growth; Gold's load formula (secondary source). | low, if quiet worlds stay quiet | exact | W; O for ice growth | 1 | M |
| **Fog** | None | Koschmieder V = 3.912/σ added to the sight sum; onset from the weather state | ≈0 | exact (onset is a stand-in) | W; O | 2 | S |
| **Wind** | Driven only by fire | Outdoor log-law wind with seeded gusts, analytic outside the bubble and a boundary of the gas projection inside. Drag on relative velocity; forced cooling; wind chill; flame tilt. Gusts push the balance reflex. | ≈0 outside | keeps facts | O; W; R; M | 2 | M |
| **Rain** | None (a lattice film would cost 64-291 ms/tick) | 2-D column store; Green-Ampt infiltration; ponds become lattice water only where they matter; fuel moisture; masking noise (level unverified) | 1-2 ms per million columns per sim-second | keeps facts | O; W slow | 2 | M |
| **Soil, digging, heaps** | All ground is stone | Loam, sand, clay and gravel rows holding water. A lattice slump rule at the angle of repose (not MPM: decision 3 forbids demotion). Digging as an act with a tool and a rate (sources conflict by 5x). Spoil heap; a "freshly dug" flag. | ≈0 when idle | keeps facts | W on active cells; M | 2 | M |
| **Trench and slope failure** | None | Rankine critical height checked as the dig deepens; release to the slump rule | µs | keeps facts | W-ev | 3 | M |
| **Terrain at scale** | Flat boxes; routes sweep the whole plane | Heightmap at 0.5-1 m; Tobler walking speed; Pandolf cost; coarse A* | ms per query | keeps facts | W coarse; O; M | 2 | L |
| **Large worlds** (level of detail) | One dense array | Voxel bubble, rooms graph and region graph, with sparse diffs; demote only when provably quiet | O(bubble) | the demotion is exact; rooms and regions are approximate | O; W | 2 | XL |
| **Offscreen people** | Simulated in full or not at all | Rooms-backend agents with jobs; guarded claims; halt test | µs | keeps facts | O; M | 2 | L |
| **Snow; seasons and rivers; room temperature over hours** | None | 2-D snow store; climate normals; river reservoir; a 5R1C model per room offscreen | ≈0 | keeps facts | O | 3 | M |

### 3.9 Energy and magic

| phenomenon | now | the law | cost | fidelity | level | P | effort |
|---|---|---|---|---|---|---|---|
| **One energy ledger** | Thermal only. Fuel is not a stock. Muscle work comes from nowhere. Thermostats and scene igniters are unbooked. | Stocks: thermal, chemical (mass × heat of combustion), mechanical, latent, body, arcane. Every change outside a law goes through `inject(quantity, amount, cells, cause)`. Wood checks out at 18.5 kJ/g against a real 18-20. | 0.95 ms per audit (every tick in tests, once a second normally) | exact | W | 1 | M |
| **Universes and borders** | None in voxel; the old pack creates qi from nothing | Each universe is a set of named conserved quantities. A border row gives from, to, ratio, rate, where, who and impurities, all booked. | under 1 ms | exact bookkeeping | W where active; O | 2 (1 after a pilot) | M |
| **Personal store** (mana, qi) | None | float64 store and capacity; regeneration as a border; upkeep; overdraw on the body | 0 | keeps facts | W at 1 Hz; M; a drive | 2 | S |
| **Spells as converters** | None; the 0 °C floor blocks cold | A row gives cost, joules per unit (from the pack) and target, then calls an existing law: heat, impulse (work = J·Δv/2), fill, sink or pressure | per cast | exact, given the pack's rates | M cast; W effect | 2 | L |
| **Ambient qi field** | None | A conserved scalar on the coarse gas cells; sources only through borders; wards set permeability | 0.038 ms per species | keeps facts | W slow; O | 2 | M |
| **Packs on the voxel engine** | Packs exist only in src/core | Quantities, borders, acts, thresholds, percepts and narration as data. DCC's System XP in a separate non-physical ledger. | per row | exact bookkeeping | W; M; O | 2 | L |
| **Stores in things** (crystals, potions) | None | A float64 store per thing, with a closed-form leak | 0 | keeps facts | W-ev; M | 2 | S |
| **Arcane senses** | None | Field gradient and others' stores, attenuated by distance and wards; a concealment act | under 0.1 ms | keeps facts | per look; M | 2 | S |
| **Advancement, healing magic, force constructs, time loop** | Old core only, or none | Training law and System events; a border into the healing law; a pinned bonded piece charged to the store; snapshot and restore with the looper's memory kept | ≈0 | stand-in / keeps facts / stand-in / exact | O; W; M | 3 | S-M |

### 3.10 Not covered by any domain (found by the critic)

| phenomenon | cases | proposed owner | the law | P | effort |
|---|---|---|---|---|---|
| Money, ownership, value | 582 | social and world layer | A non-physical claims ledger plus prices | 1 | M |
| Documents (letters, maps, ledgers, forgeries) | 227 | things and senses | A content field on a thing; reading gated by light and literacy; forgery quality as a perceived fact | 1 | S |
| Locks and keys | 129 | solids (with doors) | Lock state; a matching key eid; picking = skill × time; forcing by latch strength | 1 | S |
| Melee beyond `strike` | fight: 276 | bodies | Parry and block contact; armour per region; reach and weapon mass; blow energy from the corrected body numbers; block and flinch reflex | 2 | L |
| Firearms | 169 | matter and solids | Ball speed; wet-powder misfire (the powder moisture gate); reload time; report; flash. An event plus the projectile law. | 2 | M |
| Bows, crossbows | 57 | solids | Draw force to arrow speed; penetration through the existing wound law | 2 | S |
| Boats and ships | 256 | water | Sail force, rowing against fatigue, hull in waves, boarding, anchors; voyages offscreen | 2 | L |
| Carts | 134 | solids | A rigid object with rolling resistance by ground; travel offscreen | 2 | S |
| Containers as hiding places | — | things and senses | Inside a container means hidden from sight; sound passes through the walls | 2 | S |
| One skill record per person per act | — | bodies and mind | Shared by swimming, cooking, climbing, picking, sailing, with a practice rule | 2 | M |

---

## 4. What NOT to simulate, and the stand-in that keeps the facts

| do not simulate | why not (measured or estimated) | use instead |
|---|---|---|
| CFD of combustion (FDS-style LES) at 5 cm | Far over budget, and 5 cm buys nothing at book grain | Buoyant projection at 20 cm, flame correlations, a zone model offscreen |
| Finite-volume radiation every tick, or surface-to-surface radiosity | About 100x the measured line-of-sight cost | Multi-point flame sources, clustered hot surfaces and a layer term, every 4-8 ticks |
| Soot chemistry, multi-step pyrolysis, full moisture transport in wood | Uncertain and costly | Per-fuel yields, char depth from mass lost, a skin moisture term |
| Shock or detonation on the grid | 250-500 substeps, 2.5-5 s per blast | A Kingery-Bulmash event with a confinement factor and Gurney fragments |
| Sound as waves (FDTD) | Step of 84 µs at 5 cm, about 1 s per tick | Rays with edge bending, composite transmission loss, Sabine per room |
| Global illumination at 5 cm | Direct light alone is 27-73 ms per source at 5 cm | Coarse direct light plus a lumen-method term per room |
| 3-D water (FLIP, SPH or MPM water) | Tens of ms for a small pond, against 0.03 ms for a height field | Lattice cells, height field, drops, cup records, a vapour field |
| Elasticity (FEM) every tick | 229-333 ms per solve of the house | Beam theory with fill as thickness, run on a change |
| Crack waves at real stiffness | About 1000x the substeps | Impact stress against strength |
| MPM as a strength model; plastic forging in MPM | Softened stiffness makes yield meaningless | Bonds and the beam check; MPM for flow only; forging as an act |
| Cloth drape | No decision depends on folds | A cloth row and a "covers" relation |
| Muscle-driven ragdolls (SIMBICON, Euphoria, OpenSim) | Slow, uncertain at 5 cm, and the split is cleaner with written reflexes | Written reflexes plus kinematic gait |
| Pulse, BioGears or JOS-3 at runtime for every NPC | Cost per NPC | Use them offline as calibration oracles |
| Injury dice (AIS) | The sim has no dice | Deterministic thresholds varied by each person's sheet |
| Per-voxel biology, microbes per cell, ecology | 3.2 ms per call even for a trivial per-voxel pass; 8 GB per hectare of voxel crops | Per-object schedules, 2-D plots, density fields |
| Weather as a fluid, erosion, lightning electrostatics | 80-200 ms per law per tick on a large fine grid | A weather state machine per day, authored terrain, a scheduled strike event |
| Rothermel as its own fire-spread law | A special-case subsystem (the no-special-case rule) | Spread from flames, radiation and wind on voxel fuels |
| Animal choices written as rules (flight zone, escape economics) | Breaks "author below the capability" | Drives and reward terms in the animal mind |
| Real alchemy chemistry; a universal mana-to-joule constant | No law exists | Pack data rows; a joule rate declared per pack, or none |
| Divination, narrativium, true names | Story rules, not world laws | The DM and author layer, logged as such |
| Electricity and circuits | A new solver for rare scenes | Skip; if needed, a booked heat line along the least-resistance path |
| In-scene riding | Two coupled articulated bodies (XL) | Offscreen travel and haul tables |
| Using the GPU for the sim | Its 6 GB belongs to the models | CPU and numba only |

---

## 5. The budget

A real-time tick is 25 ms. Measured numbers are marked (m). The rest are estimates.

### Typical scenes

| scene | now | after the priority-1 realism items | notes |
|---|---|---|---|
| Quiet inn night (house4, 4 people, 0.4 M voxels) | 33 ms (m), 0.76x real time | about 42-47 ms | Adds light 1-3, hearth (burn, flame, radiation, projection) 3-5, balance 0.15/person, audit 1 (tests only). If bodies hold 37 °C flesh in the voxels, add about 6 ms more, because the air and heat laws stay awake (decision 1). Hearing batch saves 1-2. |
| house_fire (about 220 kW, 400 k voxels) | 54.7 ms (m) | about 45-60 ms | The gas velocity replaces the rise, smoke jets and part of air and O2, but burn (12.8) and conduct (5.3) stay. Realistic floor for the core laws: about 30-35 ms. Flame, radiation, light and soot/CO add the rest. |
| Still 2 m pond | 130 ms (m) | under 5 ms | Exact sleep, then a height field at 0.03 ms (m) |
| Plank thrown onto stone | 9 ms (m) | 12 ms (damping fix) or about 9 (soft contacts) | Correct landing |
| Fallen tree crown | 135-138 ms (m) | about 175 ms with the damping fix; islands drop quiet crowns to 0 | Islands are priority 2 |
| Yard 10 × 8 m, 1 idle person (1.6 M voxels) | 77 ms (m) | about 20 ms (target) after windowing | Windowing is exact (golden SAME) |
| Harvest jumper world | 3-6 ms | +0.16 ms per person for balance | Stays well inside budget |
| One 8-hour night | 5-10 h of compute | milliseconds if provably quiet; ticks otherwise | A burning candle keeps the world awake |

### Measured or estimated cost of each new law

| law | cost |
|---|---|
| Gas projection, house at 20 cm | 0.7-1.25 ms (m, unconverged); expect 2-5 ms converged |
| Radiation pass, 39k faces × 4 points | 2.2-5.5 ms (m), every 4-8 ticks ≈ 1 ms/tick |
| Flame height | under 1 ms |
| Light, one source on the 20 cm grid | 0.15-0.25 ms (m), cached |
| Hearing, batched | about 1-2 ms for 80 pairs, against 21 ms (m) for 80 pairs now |
| Balance check | 157 µs per person (m); about 20 µs in numba |
| Fatigue / Henriques / thin skin | 11 / 4.5 / 59 µs (m) |
| Energy audit | 0.95 ms (m) |
| Blast | 10.6 ms (m), once per explosion |
| Height-field water | 0.026-2.5 ms (m) |
| One gas species on the coarse cells | 0.033-0.038 ms (m) |
| Scheduler | 0.5 µs per event (m) |

### What this says

The new laws are cheap. With the map as proposed, fire scenes stay roughly where they are, and social scenes get 25-40% slower. **No scene with people meets 25 ms today, and none will after the realism work alone.** The ceiling is per-person overhead:
- whole-lattice passes per person;
- `_look_around` at 3.1 ms per look;
- `_routes` at 0.49 s per call in large worlds;
- `_objects_within_reach` at 14.5 ms per call.

Those must be cut first, and they are cut by exact windowing, not approximations. The batch harvest does not need real time, but the playtest loop and the 10-runs-per-second target do.

---

## 6. The recommended engineering wave, and the waves after

### Wave 12: the engineering wave

There are seven lanes, each owning its own files. Lanes 1, 2, 4 and 5 have no dependencies and start at once. Lane 3 starts with its contract. Lanes 6 and 7 can slip into wave 13 without blocking anything.

**Lane 1: SCALE.** Files: `support.py`, `fields.py`, `label.py`, the lit band in `perceive.py`.
- Steps:
  1. Profile one idle person at 0.4 / 1.6 / 5.1 M voxels.
  2. Window `_law_torque` and `_relax_slack` to boxes around bodies and changed cells. Remove the per-call `np.indices`.
  3. Cache `burning()` and `T()` once per tick.
  4. Make `label` incremental or windowed.
  5. Bound the sight band.
- `_routes` is not in this lane. It goes into the stairs rewrite in wave 13.
- Checks:
  - golden SAME;
  - lockstep tests in the style of `test_sleep_regions`;
  - one idle person at 1.6 M voxels drops from 77 ms toward about 20 ms.

**Lane 2: BONDS.** Files: `src/bonds/kernel.py`, `src/bonds/store.py`, `tests/test_bonds.py`.
- Steps:
  1. Put the contact damping into the step bound, counting every member that can touch at once.
  2. Cap friction per piece.
  3. Write the new tests below.
  4. Re-run tree_fell and tree_hinge.
  5. Measure an XPBD rope (probe only).
  6. Islands, if there is time.
- Checks:
  - a plank dropped on stone never gains energy, with restitution in the wood band 0.3-0.45;
  - a plank thrown sideways never changes the sign of its sideways speed, and stops within about one cell;
  - declared golden diffs only;
  - fuzzer clean;
  - plank scene ≤ 12 ms, crown ≤ 175 ms.

**Lane 3: BODY (the balance reflex).** Files: `motion.py`, `bodies.py`, a new `reflex.py`, the aloft line in `physiology.py`, tests.
- Steps:
  1. Write the reflex contract (section 2).
  2. Footing law: load-bearing contacts, hull, XcoM. A held thing is not support.
  3. Balance reflex: unload the foot, step, grab, fall.
  4. Unconscious bodies crumple.
  5. Topple under real g.
  6. Physiology keeps running while a body is aloft.
  7. Shove liquid out of the way instead of refusing the step (`motion.py:930`; drag comes later).
- Checks:
  - fuzzer, 150 worlds: 0 bodies stuck aloft (seeds 56, 86 and 119 named);
  - a person placed with one foot over a gap shifts weight and steps back without falling;
  - a pushed body past its margin steps;
  - an unconscious body crumples within about 1 s instead of tipping like a plank;
  - topple time of a rigid 1.5 m body within the band of the pendulum law;
  - a walker crosses a 2 mm water strip;
  - harvest unchanged except for declared cases;
  - added cost ≤ 0.2 ms per person.

**Lane 4: MINDS AT REST.** Files: `will.py` (where the idle target is chosen), the hearing kernel in `senses.py` and `kernels.py`.
- Steps:
  1. Idle wandering picks targets inside the person's own air region (from `label.py`) and among cells they believe free. A person leaves the room only by choice, a drive or a job.
  2. Batched hearing in numba, the same law exactly.
- Checks:
  - N idle people for 10 sim-minutes never leave their room unless they chose to;
  - hearing golden SAME;
  - `_law_hear` for 20 sounds × 4 ears drops from 21 ms to ≤ 3 ms.

**Lane 5: FIRE.** Files: `heat.py`, fuel rows in `materials.py`, igniters in `scenes.py`, `tests/test_realism_fire.py`.
- Steps:
  1. World-law fire labelling (Q and D per fire).
  2. Flame height (F25), with wall, corner and ceiling.
  3. Flame cells heat what they touch, paid from that fire's own heat.
  4. Radiation G7: flame points and clustered hot surfaces (capped), smoke attenuation, every 4-8 ticks, landing in skins.
  5. Char depth from mass lost; char in the skin path; rope chars; glow readout.
  6. The boiling curve on wet hot faces.
  7. Igniters in SI. Remove pre-lit starts where a flame now reaches.
- Radiation onto people lands as heat, but the old burn law stays until wave 13.
- Checks:
  - a 5 kW fire over a 0.15 m pool gives a flame about 0.3 m tall;
  - a spill lights the bed 20 cm above it;
  - the crib test F-T2 passes;
  - ignition time against the critical-flux curve (about 12 kW/m²);
  - char rate 0.6-0.8 mm/min;
  - the energy books close (no energy created);
  - flame under 1 ms; radiation ≤ 1.5 ms/tick, measured once on a fully involved room.

**Lane 6: FLUIDS.** Files: `fluids.py`, tests.
- Steps:
  1. Exact sleep for `_law_flow` and `_law_head`. The dirty set must spread within the sweep and over whole connected bodies. Use a numba port in the same order.
  2. Equilibrium film per liquid.
  3. Drop gravity in SI.
  4. Retained film, sorption, and skin moisture.
- Commit the sleep separately: it is golden SAME, and the rest is not.
- Checks:
  - sleep is golden SAME;
  - a still 2 m pond ≤ 2 ms;
  - L-T1, L-T2, L-T3 and L-T8 pass;
  - mass ledger exact;
  - joint check with lane 5 at merge: F-T9 wood shows HRR down by 50% or more at 10 s.

**Lane 7: GAS** (behind a switch, off by default). Files: `air.py`, a new gas-flow file; the switch is one line where laws are registered, merged last.
- Steps:
  1. Buoyant projection on the 20 cm cells, solved to a tolerance (multigrid or PCG).
  2. Limited advection of heat, smoke and O2.
  3. Substeps in the flame zone.
  4. Measure the converged cost.
- Checks:
  - house_fire at 30 s is hotter under the ceiling than near the floor;
  - layer sharpness against Zukoski filling;
  - Alpert ceiling jet;
  - doorway neutral plane;
  - mass and energy exact;
  - a cost report.
- Only when these pass does the switch turn on, and `_law_rise` and the smoke jet constants go.

**Merge order:** SCALE → hearing batch → fluids sleep (all three golden SAME), then BONDS, BODY, FIRE, the rest of FLUIDS, and GAS with the switch off.

**Wave checks:**
- full suite;
- golden with declared diffs;
- fuzzer, 150 worlds;
- fast-tier time;
- a timing table per scene against section 5.

### The waves after

| wave | name | content | depends on |
|---|---|---|---|
| 13 | Ground truths | One material record (and beam check started); boundary temperature, below 0 °C, quiet gates; G10 body numbers; body heat per decision 1; burns on thin skin; fall injury by region; fatigue; bleeding with press and tourniquet; doors, latches and locks; smoke opacity and soot (if GAS landed); pain-withdrawal and grab reflexes; others' visible state; stairs (2.5-D) together with route bounding | wave 12 |
| 14 | Seeing and hearing | Calendar, sun, sky; light field; detecting vs identifying; speech loudness and partial understanding; sustained sounds, room ambient and crackle; smelling smoke; dark adaptation; touch in the dark; reading documents; recognition and disguise | wave 13 fire labels, soot |
| 15 | Time and scale | Clocks, scheduler, skip-ahead; energy ledger with booked inject; body clocks (energy and water stores, sleep, healing, infection); bond islands; ropes on XPBD; friction and loose rigid objects; chopping; carts | waves 12-13 |
| 16 | Matter, water, weapons | Water vapour; solutes and cup records with pour and drink acts; alcohol; powder and fuse; blast; firearms and bows; melee; oil slopover stand-in; height field, buoyancy, a person in water; boats | GAS, wave 15 clocks |
| 17 | Living and packs (after BoC and MoL pilot chunks) | Soil and digging; crops in 2-D; animal bodies and minds; magic borders, stores, spells, voxel packs; the corpse; contagion | pilot results, waves 14-16 |
| later | World at scale | Level of detail, offscreen people, wind, rain, terrain, seasons | waves 12, 15 |

The books lane (fix the pilot set, then recipe v2) runs in parallel to all of this.

The OpenWorld transfers in docs/voxel-roadmap.md ("Stolen designs (OpenWorld)") are slotted by wave:

| Items | When |
|---|---|
| 1-6, 15-17 | any time |
| 7 | wave 13 |
| 8-10 | wave 15, or wave 13 if L11 moves forward |
| 11-14 | wave 17 and "later" |

---

## 7. Open decisions for the user

1. **Where does body heat live?** Choices: on the person (a core and shell model with heat flowing through the skin cells), or in the flesh voxels at 37 °C.
   - **Recommend: on the person.**
   - Warm flesh voxels keep the heat and air laws awake around every body (about +6 ms/tick in every scene with people), and they stop the night skip from ever firing.
   - Burns still read the real skin cells.

2. **The reflex contract.** One shared layer, with per-person latency, step length and sprint speed drawn deterministically from the sheet. Suppressed by chosen intent. Degraded by drink, fatigue and sleep. Absent when unconscious.
   - **Recommend: yes.**
   - On lost capability, **recommend:** remove a capability outright when the structure is gone (a broken shin bears no load; a cut tendon cannot grip), and weaken it by the law otherwise. Never by a flag.

3. **Cold: when and how.** **Recommend:**
   - In wave 13, add a boundary temperature T_out(t) read by the heat sinks and buoyancy.
   - Keep the reference fixed, rather than moving it.
   - Remove the 0 °C floor.
   - Change the quiet gates.

   This unblocks ice, winter, hypothermia, cold spells and the Cold scenario together.

4. **Is burning a row in the reaction table?** Roadmap 107 B says yes; the matter study says no.
   - **Recommend: keep `_law_burn` as its own law and share only the row format.** Its energy balance (skin temperature, flame feedback, entrainment per airspace) keeps facts that a rate row would lose, and folding it would move every golden fire case.
   - Record this as a ruling.

5. **Magic and farming before data?** **Recommend:** cut Beware of Chicken and Mother of Learning pilot chunks first, then build crops and voxel packs to what those decisions actually read. Rules for when they are built:
   - each pack declares its own joule rate, or none where the book never commits;
   - DCC's System XP and loot live in a separate non-physical ledger;
   - qi drawn from breath is tied to real O2 uptake, so a smoky sealed room really is worse for cultivating.

6. **How should speaking softly or loudly appear in the menu?** **Recommend:** one `say` act with a level parameter. The menu shows it as three or four plain rows ("whisper", "say", "say loudly", "shout"), because menu-select is depth 1 today. Later, the pointer grammar can take the level as an argument.

Default unless you object: time of day is a running world clock. Each scene sets its date, latitude and start time.