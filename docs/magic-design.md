# Mark_1 magic design

> Draft from the magic research run (2026-11-19): five research lanes, a critic and a synthesis. The research reports are in `docs/magic-research/`. NOTE: the wave numbers in the build plan (section 12) follow the old realism-map plan, not the current wave 14 and 15 plan (WORK/w14/decisions.md). The open decisions are in section 11.

Synthesis of the five research lanes (physics, biology, chemistry, engineering, fiction), the critic's report, and decisions 8.0 to 8.4. Date 2026-11-19. Research and writing only: no sim changes.

How to read the marks.
(v) = checked against the real source by web search in this research run.
(k) = from my own knowledge of the literature, believed right.
(u) = unverified, a textbook value from memory, or my own estimate. Check before it goes into a data file.
"Design default" = a number I chose to make the world work. It is data, not physics. Packs may change it.

Where the critic found an error in a lane report, this document uses the corrected number and says so.

## 1. The Answer In One Page

THE ONE CHANGE.
Add one thing to physics: a PATTERNED STRUCTURE can move energy and momentum at a distance, through a finite, conserved, out-of-equilibrium field M (mana, qi, Investiture; the name is lore).

Everything in the change, in one breath:
- M is a field. It has a density (joules per cubic metre), it spreads, it leaks into heat, it is used up where it is used, and it refills slowly from booked sources. Places differ. M = 0 everywhere is a low-magic world.
- A structure that holds an information pattern (a brain, a carving, a crystal lattice, a tuned solution, a device) can turn M into the energy forms the sim already has (heat, motion, light, chemical bonds) and back. This is "transduction". The pattern does not make energy. It steers a flow from a store to a place.
- The coupling runs both ways. Whatever can send M can sense it. Whatever sends pays recoil and leaves a tell.
- Every effect's energy splits into useful work, light and waste heat. The three sum to 1. Waste heat appears where the conversion happens.
- Three timescales: casting is slow, taking hold is finite and fast (light-like), the effect then unfolds by the laws the sim already has.

WHY EVERYTHING ELSE FOLLOWS.
1. Energy comes from three places only: the caster's body, the local M, or a charged material (a gem, a potion, a battery). The real numbers fix the scale. A fireball needs about 0.9 GJ (that is about 200 kg of TNT). All the fat in a body holds about 0.37 GJ. So body-only magic is weak by physics, and strong magic needs M or stores. No rule says so.
2. Because the pattern is only a valve, "small push, big release" is lawful when something already holds the energy. Lighting fuel with a few kJ releases MJ. A trigger on a charged gem releases what the gem holds and no more. Knowing real physics (where the metastable stores are) makes a better caster, and that is true in the sim for the same reason it is true in life.
3. Because M is a finite field with waves, you get counters for free: interference gives counterspells, reflection gives shields, resonance gives foci and links, a region of low coupling gives anti-magic.
4. Because every medium (voice, thought, gesture, carving, music) is a channel with a rate and noise, you get cast time, cost by complexity, the visible tell, errors that land on neighbouring valid spells, and a reason for rituals, all from information theory.
5. Because the coupling is a dial in each individual and each material, you get sorcerers (high dial), wizards (practice), patrons (a big store at the far end of a lasting bond), beasts that evolve M organs, and artificers and cyborgs (a device that holds the pattern). No flags.
6. Alchemy is the ordinary chemistry of materials that store or couple to M. Purity and crystal quality set the coupling, like real transducer materials.

WHAT THIS DOCUMENT ADDS TO THE RAW RESEARCH.
The critic found one big hole: if the waste heat of a 0.9 GJ fireball goes into the caster's body, it kills the caster seven times over, even at 99% efficiency. The fix, used throughout: the conversion waste is booked at the SITE where M is absorbed, and only the small control overhead goes to the holder. Section 2.5 gives the two-line rule.

WHAT IS NOT COVERED BY THE ONE CHANGE.
Teleport, true time effects, creation of mass from nothing, other planes, faster-than-light information. These need a second change. Section 9 lists them and recommends keeping them out of the base and offering them only as pack options, each with a booked cost.

## 2. The Core Law

2.1 THE FIELD M

M(x,t) is a scalar density in joules per cubic metre of "convertible energy". It is energy-like on purpose, so the ledger stays in joules.

Properties, each a number or a table row:

- DENSITY and DYNAMICS. dM/dt = D * laplacian(M) + S(x) - k*M - U(x,t).
  D = spreading (m^2/s). S = source in W/m^3 (a ley line, sunlight on a crystal, life). k = leak to heat (1/s, booked as heat). U = use by structures.
  This gives "finite, depletes where used, refills slowly, places differ" (physics P1).
- FLOW. Spreading by D gives a slow current from rich to poor cells. A pack may add a drift (a river of M along a ley). Design default D is small: M does not rush back into a hole.
- SPEED c_M. The speed of the fast mode (pulses, see below). One pack parameter. Arrival tick = ceil(path / (c_M * dt)), minimum 1 tick. Critic C1: one rule for aim lag, homing, scrying and counterspell races. DESIGN DEFAULT: light-like, c_M of 1e7 m/s or more, so every effect arrives next tick and recoil is negligible (see 2.6). A "slow M" world (c_M 100 to 1000 m/s) is a pack option that gives visible, dodgeable bolts and real recoil.
- ATTENUATION. Beer-Lambert (k): I(x) = I0*exp(-mu*x), plus geometric spreading (spherical 1/r^2, beam roughly constant out to the Rayleigh range). Each material has mu_M per band. Seawater kills radio: skin depth about 8 m at 1 kHz (k). The same shape of table gives "M does not pass through lead".
- IMPEDANCE. Each material has Z_M. Normal-incidence power reflection R = ((Z2 - Z1)/(Z2 + Z1))^2 (k). Water to air for sound reflects about 99.9% (k). A "mirror" is a big impedance step. A "shield" is a big mu_M. An "absorber" is matched Z with big mu_M. Absorbed energy becomes heat in the shield, booked.
- WAVES. A cast emits a pulse packet: energy E, frequency f, bandwidth B. It moves at c_M, is attenuated along a ray march (energy, speed and attenuation exact; no wave PDE), and arrives as events with a transmission product. Wavelength = c_M/f, so fine patterns have short wavelength and obey the diffraction limit (smallest spot about lambda/(2 NA), Abbe 1873 (k)). Fine aim and long range pull against each other. Dispersion makes wide-band (complex) pulses smear with distance while narrow-band ones carry far (u, standard result): a lawful range-versus-complexity trade.
- NEAR AND FAR FIELD. Inside about lambda/(2 pi) the coupling is reactive: efficient, no radiation, a hard-to-detect tell. Wireless power transfer shows it: Kurs et al., Science 317:83 (2007): 60 W over more than 2 m at about 40% efficiency (v, critic confirmed). Beyond 2 D^2/lambda it is radiative with 1/r^2 loss. So touch spells are efficient and quiet. Ranged spells are lossy and visible. No rule, only the field.
- RESONANCE. A resonator has f0 and Q. Response is Lorentzian with width f0/Q. Stored energy builds to about Q times the drive per cycle (k). Q values: tuning fork about 1e3 (u), quartz 1e4 to 1e6 (k), superconducting cavity 1e10 (u). The laser is the template (Einstein 1917; Schawlow and Townes 1958): a stored inversion (M or charge) + a cavity (engraved pattern) + a seed (trigger) gives a narrow, intense output with a threshold, then saturation at the store.
- REGIONAL COUPLING. Each place or material has a coupling g(x), the fraction of M a pattern there can use. g is measurable from the material (purity, lattice order, dopant; see 5.1) or, for a living thing, from its dial. A low-g region is anti-magic. Never a per-class tag: critic B1.
- INSTABILITIES. With slow drive and threshold release, M builds up and discharges in avalanches with power-law sizes (Bak, Tang and Wiesenfeld, PRL 59:381, 1987 (k)): mana storms and wild surges with no extra code (pack option). Feedback with gain above 1 grows exponentially but saturates at the local store (laser threshold law). A feedback loop of conversions each below 100% efficient dies out: no perpetual motion.
- NOISE. The field itself is noisy. An ambient fluctuation floor raises the error rate of every medium (pack parameter; engineering lane section 9).

THE SECOND-LAW GUARD (physics P15). M must stay out of equilibrium with matter, or no work can be taken from it (Kelvin-Planck). So M is a battery with a gradient, not a free lunch. The refill source S is an external, booked line. Any tick with net energy gain not matched by S is a bug.

2.2 TRANSDUCTION BY PATTERNS: WHY "PATTERNS TRANSDUCE M" HAS A REAL BASIS

The physics of information and energy:
- Landauer 1961: erasing one bit costs at least k_B*T*ln2 = 2.87e-21 J at 300 K (k). Verified experimentally: Berut et al., Nature 483:187 (2012) (k).
- Szilard 1929 and Bennett 1982: one bit of knowledge can earn at most k_B*T*ln2 of work, and the memory reset pays it back, so the second law holds (k).
- Sagawa and Ueda, PRL 100:080403 (2008) and 104:090602 (2010) (k): extractable work is at most -dF + k_B*T*I, with I the mutual information gained by measuring. Experiments: Toyabe et al., Nature Physics 6:988 (2010): the 28% figure is free energy gained over mutual information (v, critic corrected the lane's wording). Koski et al., PNAS 111:13786 (2014): up to 75% of k_B*T*ln2 per cycle (v). Different ratios; do not compare them.
- The scale is tiny. A 0.9 GJ fireball would need 3e29 bits at 300 K. So pure information-to-work is not a source. It is a VALVE. This is exactly decision 8.1: sources are body, M and charged material. The pattern decides which gradient, when and where.
- Real systems run 1e5 to 1e11 times above the floor (neuron spike about 1e-9 J (u)). So the pack sets "overhead per bit" far above the floor, and the floor is only a lower bound.
- "Better knowledge makes a better caster" is real: a caster with more mutual information about the gradient (thermal, chemical, metastable store) can extract more per decision, up to k_B*T*I. And a caster who knows where a metastable store sits gets the store's whole release for a few kJ of trigger.

THE TRANSDUCTION LAW (physics P9, chemistry M4, biology P5, with the critic's fixes):
  out = eta * in, for in above a threshold. The output type is tied to the input type.
  E_in = E_work + E_light + E_heat, the three fractions sum to 1. They are per-transducer data.
  eta is capped by the right bound: Carnot (1 - Tc/Th) for heat-driven; Curzon-Ahlborn 1 - sqrt(Tc/Th) at maximum power (Curzon and Ahlborn, Am J Phys 43:22, 1975 (k)); a pack-set figure for chemical, mechanical and M conversions.
  Chain efficiency is the product of the links. Five links at 70% give 17% (k, arithmetic). This is the lawful reason "magic with losses at every link" works.
  Faster conversion is less efficient (power versus efficiency trade).
  Charging is the same law run backwards: eta below 1 and heat at the crystal (critic A6).

2.3 CONSERVATION AND THE LEDGER

The wave 15 energy ledger with booked inject is the base. Magic adds these lines, and nothing else:

  M_cell[x]          M in each cell (lazy, see 2.7)
  store[id]          stored energy in a charged material, a body store, a potion
  body_store         ATP and phosphocreatine, glycogen, fat (the existing body clocks of wave 15)
  site_heat          conversion waste, at the SITE where M is absorbed (the effect region, or the crystal on release)
  control_heat       control overhead, to the holder of the pattern (brain, engraving, device)
  light              the light share (visible tell, and cold light)
  work               kinetic or chemical output (the existing kinds)
  reaction           momentum and energy given to the reaction partner (ground, caster, field)
  leak_heat          M leaking to heat, k*M per cell, and store leak
  imported_M[origin] the one refill line. origin is one of: planet, sun, life, patron, other world. Each origin has a pack rate map. Critic A3: print the daily import in every run so leaks show.
  wear               damage D on devices and tissue (not energy, but booked as a state)

CLOSURE RULE, checked every tick, as with heat: the sum over matter heat, kinetic, chemical, M, store, light, plus imported_M and minus radiated light leaving the world, is constant. A cast that asks for more energy than its sources can supply is VALID but delivers only what the sources supply (the type check rejects only malformed spells; the sim stays authoritative; engineering lane section 6).

Interference is booked per cell, not at a point (critic A4). Over an overlap, compute intensity I = I1 + I2 + 2*sqrt(I1*I2)*cos(dphi) per cell and book the total. Full cancellation needs equal amplitude and opposite phase. The energy does not vanish: it moves to the cells of constructive interference, and the canceller's source pays its work. Active noise cancellation shows the limits: it works only at low frequency and in a "zone of quiet" about lambda/10 across (Lueg patent 1936; zone size (u)).

Refill is never from nothing (critic A5). A "slot" or "daily rest refill" is only a UI block of joules over two real flows: food intake (glycogen about 7 MJ refilled over about 24 h (u)) and local M draw. Feruchemy-style "store now, spend later" is lawful because it is a conserved store of a body quantity.

2.4 THE THREE TIMESCALES (decision 8.3, now with numbers)

(a) CASTING is slow. Time = bits of the spell / (medium rate x skill factor). A 60-bit spell takes about 1.5 s by voice, 6 s by thought, 12 s by gesture, 0 s from a carving (the cost was paid at carving time). Each medium has a tell (sound, light, a field leak, visible motion). While casting, the caster draws on control power and can be interrupted (a dodge, a blow, a counter-pulse). Section 3.3 has the table.

(b) TAKING HOLD is finite and fast. The pulse leaves at the end of casting and arrives after path/c_M (at least one tick). At the light-like default that is next tick. Aim error depends on target speed times (cast lock time + flight time) plus the aim channel's noise.

(c) THE EFFECT UNFOLDS by the existing laws at the delivered power. A fireball is heat delivered into voxels; combustion, smoke, fluids, injury and the rest then do what they already do. The sim, not the spell, decides what happens next. The spell has no per-spell code.

2.5 RECOIL AND WASTE (the critic's two-line rule, which replaces the raw 8.2)

Decision 8.2 said "recoil goes to whatever holds the pattern". The arithmetic says that must be split in two, or every large cast kills the caster:

LINE 1, CONTROL OVERHEAD, goes to the holder of the pattern.
  = (bits of the pattern x overhead per bit) + (medium body power x time).
  This is small: the brain adds under 5% to its 15-20 W when thinking hard (Raichle and Gusnard, PNAS 2002 (u); Aiello and Wheeler 1995 gave 14.6 W (v)). Voice, gesture and dance use real muscle power (see the table in 3.3). The holder's body or device heats by this amount and by nothing else, by default.

LINE 2, CONVERSION WASTE, (1 - eta) x E_delivered, goes to the SITE: the effect region, or the crystal on release. Real devices heat where energy is converted, not where the switch is (critic A1).
  Pack option: a share of the conversion waste goes to the holder (for worlds where magic burns the caster). Default share = 0.

Why this matters. A 0.9 GJ fireball at 99% efficiency wastes 9 MJ. The body can take about 1.2 MJ of heat before organ damage (5 K x 245 kJ/K, using c about 3.5 kJ/kg/K (u)). So 9 MJ in the body is 7 times the lethal dose. At the site, it is just a hot fireball.

BODY-ONLY MAGIC KEEPS THE HEAT CAP, because there the body IS the store and its waste is in the body. With a 60% efficient conversion, the most a body can deliver before 1.2 MJ of waste is about 1.8 MJ (biology lane P4). That is a thrown-rock scale, not a fireball. This is lawful and matches 8.1. Pack-set demigod bodies may change the caps.

MOMENTUM RECOIL. A pulse of energy E at speed c_M carries momentum p = E/c_M (like light pressure; Noether 1918 requires M to carry momentum if the coupling exists; physics section 6). For a 0.9 GJ pulse:
  light-like c_M = 3e8 m/s: p = 3 kg m/s (nothing).
  c_M = 1e7 m/s: p = 90 kg m/s (a shove).
  c_M = 100 m/s: p = 9e6 kg m/s (fatal).
So recoil is a function of c_M and the default light-like value makes it negligible (critic A2). With a fast c_M, the holder's pain is Line 1 heat, not momentum.

FORCE EFFECTS (telekinesis, levitation, flight, a push) must NAME THEIR REACTION PARTNER in the ledger (critic D4). The delivered momentum comes with an opposite momentum booked to: the ground (if the caster stands on or touches something anchored), else the caster, else the field with a pack-set share. You cannot lift yourself by your bootstraps. Hover costs only the conversion loss plus holding the push.

2.6 THE ONE-LINE SUMMARY OF THE CAST CYCLE
A cast = (1) build a pattern over bits/rate seconds, paying Line 1; (2) draw E from sources (M in the reach volume at a capped rate, a store, the body); (3) emit a pulse that takes its energy from named stores at emission and returns the unabsorbed rest at the end (critic C3: a transfer between named stores, not two stores of one energy); (4) at the site, absorb and convert with the split (work, light, heat), booking Line 2; (5) the existing laws run on.

DRAW RATE LAW (design default; the lane reports gave no number). A transducer draws power P = g * M_local * V_reach / tau_draw, with V_reach the volume its pattern can reach (a brain about 1 m radius, a big engraving a lot more) and tau_draw 1 s. So a fireball in a place of ordinary M must draw from a big volume, leaves a "crater" of depleted M behind it, and takes a visible time. See section 7.

2.7 CHEAP ESTIMATING LAWS THAT KEEP THE FACTS (critic F; the user wants real time)
- M field LAZY. A cell not drawn on relaxes by the closed form M = S/k + (M0 - S/k)*exp(-k*t). Store (M0, t0) and compute on read. Diffuse only near recent draws.
- Pulses are events with an arrival time and a transmission product. No wave solve.
- Metastable release: one time-to-event from the barrier law. Check the critical size once per heat change, not per tick.
- Neighbour errors: a precomputed discrete kernel; block failure is a binomial tail.
- Practice: x(n) = x1 * n^-b, closed form. Wear: Miner's rule per use.
- Body heat: one scalar core-temperature ODE per caster.
- Spirits: one scalar ODE for M_self, wake on intake change. Ecosystem M map: refresh daily.
- Potions: a one-compartment closed form. Choir: error e/sqrt(n), not n separate casters. Healing: a day-scale rate law with mass from food.
NEVER estimate: the energy ledger, the heat cap, M conservation.

## 3. The Pieces (The Only General Building Blocks)

3.1 TRANSDUCERS

Every transducer is one record: {in_type, out_type, eta, light share, threshold, response time, rated power, max temperature, coupling g, bandwidth in bits/s, noise floor, damage D}. Seven kinds, each a setting of the same record.

a) MINDS (flexible patterns). A brain holds a pattern that changes at the speed of thought. Casting = the mind builds a grammar expression in its medium and the body or reach volume converts. The coupling g is the individual's dial (section 4). Bandwidth is bounded by the 10 bits/s human bottleneck (Zheng and Meister 2024, Neuron; arXiv 2408.10234 (k); the critic could not find it by search but knows the paper).

b) INSCRIPTIONS (fixed patterns). Carved or written once, run many times. The pattern's cost is paid at carving time (1 to 5 bits/s while carving). Persistence: centuries. Three roles (follow-up to 8.4, engineering lane M5): standalone enchantment, focus, shaper. See 3.5.

c) CRYSTALS (store and release). A lattice that holds charge and releases it on a trigger. A crystal is a store (3.2) plus a trigger gate. A crystal can also be a transducer row in the materials table (below).

d) LIVING TISSUE. Nerve and muscle already leak microvolts. Evolution can keep and stack them (electric fish: six independent origins, Gallant et al., Science 344:1522 (2014) (v)). A beast organ is a stacked-cell record: V = n_series * v_cell; I = n_parallel * i_cell; E = V*I*t, with a sync factor 0 to 1 (biology P1). Anchor: eel 860 V (De Santana et al., Nat Commun 10:4000 (2019) (v)); about 1 A and about 1.7 J per pulse are (u). The best real organ moves joules, not gigajoules: a gap of 5e8 to a fireball. So real beast magic is small unless it uses M or a store.

e) SOLUTIONS (potions). A dissolved fuel, charged material or pattern-bearing structure. Delivered through ADME curves (5.3). Energy-type potions are bounded by the drinker's heat limit: 1 MJ in a minute (17 kW) cooks the drinker.

f) DEVICES (artificer and cyborg hardware). Engraved circuits, wired stores and sensors, wands, implants (3.5 and 4.5). Rated power, memory bits, gate count, gain G, Q, leakage, trigger threshold and false-alarm rate, wear.

g) SELF-SUSTAINING FIELD PATTERNS (spirits, patrons, gods, fading curses). A dissipative structure in M. It needs a throughput (intake power at least dissipation plus repair), a boundary or a topological defect, repair that outruns errors, and for minds enough bits. Update: dM_self/dt = intake - leak - repair_cost. It dies when M_self falls below a threshold or the boundary fails, as a decay, not a switch. Real existence proofs: flames, hurricanes (about 1e14 W (u)), the Great Red Spot, Turing and Gray-Scott patterns (Pearson, Science 261:189, 1993 (k)), Prigogine's dissipative structures (Nobel 1977 (k)), England's bound on the heat a self-replicator must dissipate (J Chem Phys 139:121923, 2013 (k)). Size scaling by Kleiber's law, intake about mass^0.75 (Kleiber 1932 (k)). Optional topological persistence (energy barrier E_b, decay rate about exp(-E_b/kT_eff)) covers sealed spirits and enchantments that last centuries. A scalar M cannot hold the bits of a mind (critic D2): see decision Q4 in section 11.

THE MATERIALS TABLE: how real crystals map to M-rows (chemistry M4). Each row is a real transducer class, and M coupling is one more row of the same table:

  class            in            out         real coupling                         notes
  piezo            strain        charge      quartz d 2.3 pC/N; PZT d33 300 to 600 (k); k33 about 0.7   harvest 10-30% (u); power uW to mW
  pyro             dT/dt         charge      LiTaO3 about 2e-4 C/m^2/K (u)         responds to CHANGE, not level; Olsen cycle up to 40% of Carnot (u)
  tribo            contact       charge      10 to 250 uC/m^2 (u); kV, tiny current   "rubbing amber"
  photo            light>gap     charge      Shockley-Queisser 33.7% single junction (k); silicon about 27%  sun-stones
  thermo           gradient      voltage     Bi2Te3 S about 200 uV/K (k); real devices 5-8% (u)    capped by Carnot
  magnetostrictive field         strain      Terfenol-D 1500-2000 ppm (u)          singing stones
  M-couple         M             any         g from purity, lattice order, dopant (5.1)   the new row

Each row has a threshold, a response time, a max temperature (the Curie-like point where it depoles or cracks), and waste (1 - eta) goes into the crystal as heat, so a pushed crystal glows, cracks or melts.

3.2 STORES, WITH AN ENERGY-DENSITY TABLE

Every stored thing has MJ/kg, a form, a barrier Ea, a leak time constant tau, a maximum discharge power per kg, and a trigger site. Release = min(E_store x mass, what the trigger permits). Trigger gain is REPORTED, not assumed: gain = E_released / E_trigger. Real gains: hydrocarbon-air ignition needs about 0.2 mJ (hydrogen 0.017 mJ (u)), and 1 g of gasoline releases 46 kJ, a gain of about 2e8 (physics lane arithmetic).

ENERGY-DENSITY TABLE (MJ/kg unless noted; textbook values (u) to two digits; the user's fireball is 0.9 GJ)

  store                               MJ/kg        mass for 0.9 GJ     notes
  spring steel (elastic, 1 GPa)       0.0003       3,000,000 kg
  supercapacitor                      0.01-0.05    20,000-90,000 kg
  flywheel, steel                     0.1-0.2      4,500-9,000 kg
  lead-acid battery                   0.11-0.14    about 7,000 kg
  Li-ion battery                      0.9-1.0      about 900 kg
  black powder                        about 3      300 kg
  TNT                                 4.2-4.6      about 200 kg        (the lane said 20 kg; the critic corrected it)
  thermite                            about 4      225 kg
  sugar                               17           53 kg
  dry wood                            16-18        about 53 kg
  coal                                24-30        about 33 kg
  FAT                                 37           about 24 kg         a body: 10 kg of fat = 0.37 GJ
  gasoline                            46           about 20 kg
  hydrogen                            120          7.5 kg
  U-235 fission                       8.2e7        about 11 MILLIGRAMS (the lane said 11 g; the critic corrected it)
  D-T fusion                          3.4e8        about 3 mg
  matter-antimatter                   9e10         10 ug

  M-CHARGED MATERIALS (the new dial; design defaults, pack-tunable):
  poor charged stone                  1-10         90-900 kg
  lesser gem                          100          9 kg
  fine gem                            1e4          90 g
  great gem (CAP)                     1e5          9 g                 between chemical and nuclear
  Rule: a pocket gem that holds one fireball must beat chemistry by 100 to 1e4 times. Put a cap per material class: over-driving a store cracks or melts it.

  BODY STORES (biology lane, mostly (u)):
  ATP + phosphocreatine (fast burst)  about 30 kJ readily available    (about 60 pistol bullets of energy at 500 J each (k))
  glycogen                            about 7-8 MJ                      refilled over about 24 h
  fat                                 about 0.37 GJ (10 kg)            mobilised at most about 0.5 g/min = 300 W chemical
  heat cap                            about 1.2 MJ of waste before organ damage
  Consequence: a human cannot turn 0.37 GJ of fat into a fireball in a second. Delivery caps at about 300 W sustained and 1 to 2.5 kW for seconds (elite sprint peak (u)). 0.9 GJ at 300 W is about 35 days. So body-only magic is limited by POWER (W), not only energy (J). The sim needs a power cap and three fatigue channels (store, heat, attention), probably in the physics, not only energy totals.
  Healing and a potion: a 50 ml potion at fat density holds about 1.8 MJ. 1 kg of regrown tissue needs about 5-10 MJ (u).

  STORE DYNAMICS:
  Charging: the transducer law run backwards (eta below 1, heat at the crystal).
  Leak: E(t) = E0 * exp(-t/tau). tau is a parameter: days (Stormlight spheres), years (spirit stones).
  Releasing too fast heats the store (waste fraction). Over-driving: wear D rises (3.5).
  Metastable triggers (physics P7, chemistry M1-M3): the release rate is r = A*exp(-Ea_eff/(kT_eff))*f(E_s), with Ea_eff = Ea - catalyst lowering - trigger energy. A chain factor k_chain above 1 grows as exp((k-1)t/tau) and saturates at the store. Critical size follows the Frank-Kamenetskii number: slab 0.88, infinite cylinder 2.0, sphere 3.32 (v; critic corrected: two lanes had these swapped). A tribulation (cultivation) is exactly a discharge of an over-dense store.
  Real "small push, big release" cases: Dobereiner lamp (1823, platinum ignites hydrogen at room temperature (k)); supercooled water to about -38 C (k); bubble chamber (Glaser, 1952 (k)): keV in, a visible bubble track.

3.3 CHANNELS AND MEDIA: A QUANTITATIVE TABLE

THE CHANNEL LAW (engineering M1). Every medium m has: rate R_m (bits/s), dimensions D_m, precision sigma_m, base error e_m, stress sensitivity k_m, loudness L_m, energy cost E_m (W), persistence tau_m, and coupling c_m (pack-set; a pack may set it to zero to remove a medium).
  Shannon (1948 (k)): C = B * log2(1 + S/N). Bits sent reliably per second <= R_m * (1 - H(e)), H = binary entropy.
  Effective error: e = e_m * (1 + k_m * stress) / (1 + practice). Stress, pain, fatigue, injury and a hostile ambient field raise it.
  Cast time = bits / (R_m * skill factor).

SOURCED NUMBERS:
  Speech about 39.15 bits/s, sd 5.1, in 17 languages, 170 speakers; 4.3-9.1 syllables/s (Coupe, Oh, Dediu and Pellegrino, Science Advances 5(9) eaaw2594, 2019 (v)).
  Overall human behaviour about 10 bits/s (Zheng and Meister 2024 (k)); the senses take in more than 1 Gbit/s, so the body is the bottleneck.
  Speech BCI: 62 wpm, 9.1% word error on 50 words and 23.8% on 125,000 (Willett et al., Nature 2023 (v)); 78 wpm, 25% word error (Metzger et al., Nature 2023 (v)).
  Fitts's law (Fitts 1954, MacKenzie 1992 (k)): MT = a + b*log2(D/W + 1); pointing throughput about 4-5 bits/s.
  Sign language: signs about half the rate of words but equal propositions, using hands, face and space in parallel (Bellugi and Fischer 1972 (k)). Dimensionality multiplies capacity.

THE TABLE (starting data; numbers are bits per second of grammar nodes at the caster; everything not named above is (u), a good default to tune):

  medium        rate        dims    precision            error relaxed / stressed   persists   visible / audible                  body power (u)   notes
  thought       10          1       high (digital)       1% / 15%                   0          none                               1-2 W            silent, hidden, no free limbs
  voice         35-40       1+pitch medium               2% / 10%                   0          loud, 60-90 dB, carries            3-10 W           Coupe (v)
  sign          35          3-5     medium               3% / 12%                   0          visible                            20-40 W          hands, face in parallel
  gesture       5           3       1-2 mm / 1 mrad      5% / 20%                   0          visible                            20-50 W          Fitts; hand tremor about 1-2 mm (u)
  typing/formula 6-30       1       high (digital)       1% / 5%                    hours      on a device                        under 5 W        English text about 1-1.5 bits/char (Shannon 1951 (k))
  handwrite     3           2       medium               3% / 10%                   years      writing                            under 5 W        20-30 wpm
  music         10-50       many    3-20 cents, 20 ms    2% / 10%                   0          loud, carries far                  10-40 W          analog, harmony, choirs; trained ears 3-8 cents (k)
  dance         20          many    low                  8% / 25%                   0          very visible                       200-300 W        tiring: booked in the body
  carving       1-5         2-3     0.1 mm               0.5% / 1%                  centuries  none after                        engraving work   cost paid once; read at trigger speed
  choir         sum         n       average improves     e/sqrt(n)                  0          loud                               n x voice        sync sd about 20-40 ms
  implant (thought via device)  10 (but macros) 1+      learned map          falls with practice  0  device                      mW (heat-limited) see 4.5

Cast time examples: 60 bits by voice about 1.5 s; by thought 6 s; by gesture 12 s; from a carving 0 s.
A fireball is about 40 bits in the grammar (design estimate; section 3.5): about 1 s by voice, 4 s by thought, 8 s by gesture. The 8-second gesture fireball is slow, visible, and interruptible, which is the point.

ERROR AND NEIGHBOURS (engineering section 2; section 8 here). Each medium has its own confusion kernel over the value space of a node: voice = sound-alike, typing = key-near, gesture = angle-near, thought = concept-near, carving = tool-slip size. The result of an error is the NEAREST VALID value, so "an error gives a neighbouring valid spell" costs no special code. Redundancy (a repeat, a check word, verse, a redundant carving) lowers error but lengthens the cast; formal rituals are low-rate codes: slow but safe (Hamming 1950; Reed-Solomon; QR levels L, M, Q, H recover about 7, 15, 25, 30% damage (k)). English is about 50-75% redundant (Shannon 1951 (k)), so speech is robust. A language prior rescues a noisy channel: handwriting BCI 94% raw accuracy, 99% with a language model (Willett 2021 (k)). The spell grammar itself is that language model.

PARALLEL LANES. n casters or lanes add rate and cut noise as e/sqrt(n) when they vote on one message (u), but need timing (sd about 20-30 ms) and coordination bits. So choirs and circles are data: lanes, sync error, voting rule.

HOW THE PLAYER AND THE SMALL MODEL ENTER. A translator maps the player's words, a gesture n-best list or a menu pick into grammar nodes. The grammar prior re-ranks the n-best list, as speech recognition combines an acoustic model with a language model (Bayes: pick the W that maximises P(signal|W) P(W)). The small model picks one node from the legal menu at each step (grammar-constrained decoding, as in Outlines, Willard and Louf 2023; Synchromesh, Poesia et al. 2022 (k)). Each pick is logged as a typed delta. Menu-select is depth 1 of the same thing.

3.4 BONDS: LASTING LINKS

A bond is one record: {bandwidth, latency = path/c_M, exchange rate, sanction, lock-in}. The critic (C5) found that the physics lane ("link with bandwidth"), the engineering lane ("remote server with latency and secret") and the biology lane ("market with rate and sanctions") all describe the same thing. It covers:

  patron bond (warlock, cleric, paladin, Dresden Mantle): the far end is a mind with a big store. Authentication is a checksum or shared secret (an undetected-error chance about 2^-c for c bits (k)); an oath is a state condition of the channel.
  familiar and companion bond: a mobile partner whose body adds store or sensors.
  possession: another mind's act channel writes to the body's motor interface, paid from host metabolism (Ophiocordyceps controls ant muscle, not brain (u)). Critic B7: no new mechanism.
  curse: a lingering pattern in a body that drains M or applies a field, held by a link (hair, blood).
  sympathetic link and true name (critic B3 replaced "a number: share of energy reaching the target"): the target is a RESONATOR with f0 and Q. A caster who holds a copy or a measurement of its pattern (hair, blood, a name, a drawing) broadcasts at f0 through M at c_M. Only a matched resonator absorbs strongly (Lorentzian response, width f0/Q). Fidelity = pattern overlap x path transmission. A true name is a perfect measurement. So linking costs real energy and can be shielded.

THE MUTUALISM LAW (biology P11). A bond persists while each side's net payoff is at least 0, partners are identifiable (fidelity), either can switch (partner choice), and cheating is observable and punishable (sanction). Real anchors: mycorrhizal markets, where plants send more carbon to fungi that give more phosphorus (Kiers et al., Science 2011 (u)); cleaner fish cheat when unwatched and clients punish (Bshary and Grutter 2006 (u)); the stability conditions (Axelrod and Hamilton 1981; Sachs et al. 2004 (u)). If one side's payoff is below 0 and the other can force it, that is parasitism, and the virulence trade-off (Anderson and May 1982 (u)) sets how hard it squeezes. A deep bond can lock in permanently (mitochondria, by gene transfer (u)).
Pack parameters: rate, capacity, sanction strength, observation noise, lock-in.
A patron cutting the channel mid-cast is a bond event: the cast delivers what already arrived.

3.5 THE MAGIC GRAMMAR (decision 8.4): BUILDING BLOCKS, COMBINATORS, COST BY TREE SIZE

THE SHAPE (engineering section 6, with the critic's fix B2: leaves are PHYSICAL NUMBERS, not names):

  Spell    := Effect | Spell ; Spell | Spell & Spell | Trigger -> Spell | Loop(n, Spell) | Cond(Sensor, Spell, Spell) | Delay(t, Spell)
  Effect   := Source ; Region ; Form ; Profile
  Source   := body | field(M, reach) | store(id) | patron(bond-id)
  Region   := Shape(point | line | sphere | cone | cube | shell) x Size x Place(relative to caster | target | seen point | linked target)
  Form     := Energy(heat fraction, kinetic fraction, light fraction, chemical-trigger) | Matter-move(mass set, velocity) | M-wave(f, phase, amplitude) | Neural-bias(dial, delta) | Sense(kind, range)
  Profile  := strength over time (impulse | ramp | pulse train | steady) x Duration
  Trigger  := Sensor(kind, threshold, hysteresis) | Timer | Event
  Link     := Target handle with a fidelity (see bonds)

Every leaf is a quantised number of 3 to 6 bits (joules, metres, seconds, fractions). The names ("fire", "ice", "force") belong to a PACK's naming table, which maps regions of the number space to words. Distance in this space is physical distance (heat fraction, light fraction, radius). "Fire" is a name for a region, not a switch. A neighbour error therefore lands on a physical neighbour: a slightly bigger, hotter or longer version. Ars Magica's 5 verbs x 10 forms and Mage's 9 spheres are names only (critic B4): verbs are node types (release, sense, transform, disperse, move), forms are regions of the material and field tables, and Vim is "Form = M itself" (dispel, drain, store).

THE COST LAW (engineering M2; the critic C2 keeps this one and drops physics P10):
  bits(spell) = sum over nodes of -log2 P(choice | context). A uniform prior gives sum of log2(options). A skewed pack prior makes common spells short (Huffman 1952; Zipf (k)).
  CAST TIME = bits / medium rate.
  ERROR = f(bits, e_effective, redundancy): more nodes, more chances for a wrong one.
  ENERGY OF CONTROL = medium body power x time (Line 1 in 2.5).
  The EFFECT'S ENERGY is not the tree. It is set by the Source and Amount leaves and paid from the sources. A tiny tree can ask for a huge release; the sources decide whether it happens.
  Practised subtrees collapse to MACROS whose cost is the cost of their name (chunking, Miller 1956 (k)).
  DEPTH limit: about 4 open sub-spells (Cowan 2001 (k)). Deeper spells need a focus or writing to offload working memory.
  Combinators are compression: Loop and Cond reuse a subtree (few bits for much effect), but the ledger still charges all the power.
  Diffraction is kept only as the AIM limit (critic C2).
  A spell that asks for more energy than the sources hold is VALID but delivers what they supply. A spell with a dangling reference (target not seen) fizzles.

EXAMPLES, in bits (design estimates; the tree is the cost):
  A fireball from a gem: Effect(store(gem), sphere 6 m at a seen point, heat 0.8 + light 0.1 + kinetic 0.1 of 0.9 GJ, impulse). About 35-40 bits. By voice about 1 s.
  A shield: Effect(field(M), shell, M-wave reflector f0 and Q, steady 60 s). About 30 bits.
  A counterspell: Sense the incoming pulse (frequency, phase), then Effect(M-wave at the same f, opposite phase, a zone of about lambda/10). Needs the phase of the incoming wave; works only narrow-band and locally.
  Telekinesis: Effect(field(M), point at a seen object, kinetic, ramp 5 s) plus a named reaction partner.
  Light: a lot cheaper than a fireball: a firefly flash is about 1e-3 to 1e-2 W optical for 100 ms (u); cold magic light is lawful and cheap.

THE THREE ENGRAVING ROLES, AS ONE DEVICE RECORD (follow-up; engineering M5):
  1) STANDALONE ENCHANTMENT = sensor (trigger) + program (carved expression) + store (gem or M tap) + actuator. Like a fuse with a battery. Its trigger has a detection threshold and a false-alarm rate (ROC; Green and Swets 1966 (k)): a trap with a hair trigger sets itself off in the rain.
  2) FOCUS or ENHANCEMENT = a pre-carved partial expression (a macro with holes). The caster supplies the remaining nodes in their medium. A focus lowers cast time and error because the caster pays only the missing bits, but it fixes some choices (it is ROM). Good for armies of standard wands.
  3) SHAPER = a passive lens, mirror, waveguide, resonant cavity or shield plate. It sets geometry, gain and filtering, not information. Antenna gain G = 4*pi*A/lambda^2 (k): a bigger aperture gives a tighter beam and longer reach. Q is the cavity's.
  A signal through n engraved stages loses energy and picks up noise, unless a regenerator stage draws from a store. So an object has a bounded complexity: size, fan-out and heat dissipation set the limit. Engraved logic (AND, OR, NOT, delay, counter, latch) is built from small engravings joined by M or charge "wires" (Shannon 1938 (k)).

DEVICE WEAR (engineering M9). Every focus, engraving, crystal and implant has damage D in 0..1. Each use adds dD = n_cycles_weighted * (S/S_rated)^b / N_ref * Arrhenius(T), with S the stress (power over rated power plus temperature swings). Basquin's law (b about 5 to 15 for metals (k)) and Miner's rule (1945, fail at D = 1 (k)). Every 10 K of heat roughly halves life (k rule of thumb). Flash cells wear out after 1e4 to 1e5 writes (k). Partial D adds error and loses eta before failure. Thermal shock cracks glass-like solids at a few hundred kelvin of difference (u). This gives the user's "recoil goes to whatever holds the pattern" for objects.

## 4. Who Can Do Magic

RULE FOR THIS WHOLE SECTION. Nobody is "a sorcerer" by a flag. A person has dials, a history and a state. The archetype names are the pack's words for regions of the dial space. (Learned-mind principle: traits = dials + history + state.)

THE DIALS, HISTORY AND STATE OF ANY CASTER
  Dials: g (coupling), finesse (precision noise sigma), channel count and medium couplings (how well each medium couples), capacity of the mind's working memory, store (M pocket, size), rated throughput (flow cap through the coupling).
  History: practice hours per node and per medium, plasticity windows that close with age, wear D on the coupling, bonds held, training with devices.
  State: fatigue in three channels (store, heat, attention), stress, injury, current M in the pocket, core temperature.
  Skill = g * f(N) * s, with noise that grows with output strength (Harris and Wolpert, Nature 1998 (u): the standard deviation is about proportional to force, so bigger actions are sloppier) (biology P10, engineering M7). The practice curve is x(n) = x1 * n^-b with b about 0.2-0.5 (Newell and Rosenbloom 1981 (k)), with a floor set by innate finesse and channel noise. Transfer is high within the neighbourhood of known spells (Sadtler et al., Nature 2014 (k)) and low outside it.

4.1 INNATE COUPLING (SORCERERS, BORN CHANNELERS, MISTBORN)
- g is drawn from a distribution at birth and is partly heritable. Heritability: Polderman et al., Nature Genetics 47:702 (2015): average 49% over about 17,800 traits, mainly additive (v). Cognitive traits about 50-70% in adulthood and rising with age (Bouchard 2004 (u)).
- Model for heredity: child g = mid-parent value + noise (regression to the mean), with narrow-sense heritability h2 as a pack parameter, default 0.5 (v). Critic C4: a purely additive model rarely makes extremes, and fiction wants born casters. So add a small chance (design default 1%) of a large jump in g, a few times the parent value. Still a dial.
- Strong, narrow, hard to control: prodigies and savants come from a wiring change, narrow and deep, sometimes with a cost elsewhere (Ruthsatz and Urbach 2012 (u); Treffert (u)). So the "sorcerer" corner of the space is high g, low finesse (big noise), a narrow set of well-coupled media, and a plasticity window set early. "Wizard" is moderate g, high finesse, broad media and large working memory from practice and books.
- Critical periods: perfect pitch mostly if training starts before about age 6-7 (Deutsch; Baharloo 1998 (u)); language is easier before puberty (Johnson and Newport 1989 (u), debated). So some medium couplings are set early, and later practice cannot fully recover them.
- Allomancy-like "burn a metal" magic is a fuel plus a body dial: the effect comes from the metal's row in the transducer table, fuelled by metabolism (critic B5): no 16 code paths.
- Wheel of Time "strength" is the rated throughput through the coupling. Burnout or stilling is permanent damage D on the coupling (4.1 and section 8).
- The Witcher: a low dial and a tiny fixed vocabulary. Same law.

4.2 LEARNED SKILL (WIZARDS, MAGI, HOGWARTS)
- Practice, books and foci raise skill. Expert level takes about 10 years or 10,000 hours in some fields (Ericsson, Krampe and Tesch-Romer 1993 (u)), but practice explains only 12-26% of variance in many fields (Macnamara, Hambrick and Oswald 2014 (u)). So the innate share is a pack knob.
- Knowledge helps in two ways: (a) a larger vocabulary of macros (cheap bits), and (b) knowing real physics: where the metastable stores are, what ignites what, what resonates. This is "better knowledge makes a better caster" (2.2).
- Spellbooks are written grammar expressions (a medium with persistence); components are consumables or memory aids. The data show that verbal and material components look like MEMORY AIDS, and schools are SOCIAL categories, not physical (the "Gygax field" paper and the spell dictionaries; the dictionaries identify a spell by level, range, dice, area, somatic component and damage type).
- Metamagic (twin, quicken, subtle) is a change of node parameters or the medium: subtle = thought medium; quicken = a macro; twin = a Loop or par combinator.

4.3 GRANTED BY A BOND (WARLOCKS, CLERICS, PALADINS, DRESDEN MANTLE, STORMLIGHT SPREN)
- A patron is a mind (a self-sustaining pattern, 3.1g) with a big store and a lasting bond to the caster. Warlock = a bond with a big lent store and a rich grammar (the patron lends patterns). Cleric and paladin = the same bond with an oath as a state condition on the channel (break it, the channel closes). Stormlight: gems with a leak plus a bonded spren (a mind) plus oaths.
- The caster's own dials still matter: the bond's bandwidth is capped by the caster's 10 bits/s unless the patron holds macros. A patron needs the servants alive if it is a parasite that needs a living host; a patron with many servants can be wasteful (Anderson and May (u)).
- "Belief as energy" is lawful only as tribute passing through minds (worship = real payments in goods, labour, food, M), not as a free field (fiction N6). A bodiless god needs a substrate (Q4).
- Hostile patron: sanctions, cut channels, a market that pays for results (mycorrhizal market) are all in the one bond record.

4.4 BEASTS AND THEIR EVOLUTION

THE SENSE-BEFORE-ACT LADDER (biology P6, from the electric fish):
  Step 0: ordinary muscle leaks microvolts as a by-product (the pre-adaptation).
  Step 1: a sense organ for the field exists (electroreception is in the ancestors of most fish (u)).
  Step 2: a muscle lineage keeps firing but stops contracting: a weak organ, 0.1-1 V (signalling and sensing).
  Step 3: cells stack in series with insulating faces: voltage climbs by adding cells.
  Step 4: a pacemaker synchronises the firing (otherwise voltages cancel).
  Step 5: a strong weapon, only after 1-4, with a big body cost (the organ is 20-80% of body mass in strong fish (u)).
  Each step has a payoff of its own (sense, signal, weapon). Evolution did not need to see the final weapon.
- Real facts: electric organs evolved independently at least 6 times (Gallant et al. 2014 (v)). The sodium channel gene scn4aa was copied after whole-genome duplication and its second copy was turned down in muscle and up in the electric organ: the (v) is downgraded to (u) by the critic, since only secondary summaries say it. Bioluminescence has evolved about 40 to 94 times (Haddock, Moline and Case, Annu Rev Mar Sci 2:443 (2010) for the paper (v); the counts (u)). Eyes about 40 times (u).
- Weakly electric fish spend about 11-22% of the daily energy budget on signalling (Salazar and Stoddard 2008 (u)). A constantly running sense field is a real running cost.
- THE SIM RULE. Fitness change = benefit (food, safety, mates) minus upkeep (as a fraction of daily energy). Mutation = drift in n_series, sensitivity, sync, and in the way a beast gathers M. Pack parameters: mutation rate per parameter per generation, upkeep curve, whether convergence is allowed. Weak organs become common (hundreds of species); strong ones rare (a few dozen (u)).
- BEAST MAGIC LADDER FOR M: M-gradient sense (cheap, like the cryptochrome compass: a change in reaction yield, below kT per molecule, an information trick (u)) -> reflex release (a startle zap from stored energy) -> a store (ATP-like) -> a pattern (stacked cells that fire together) -> use of external M. The reflex magic of beasts is the same transducer record with a small g and little finesse.
- Real sense tricks as templates: sharks (ampullae of Lorenzini, about 5 nV/cm threshold (u)), platypus (electric and touch delay gives range (Pettigrew 1999 (u))), pit vipers (about 0.003 K threshold, thin isolated membrane = a SHAPER) (u), bats and dolphins (echo: power falls as 1/r^4 out and back (k)).
- Apex beasts: M concentrates up the food chain at about 10% per level (Lindeman 1942 (k); Pauly and Christensen 1995 (u)), so top predators are rich and rare. A dragon is a big store with a big body, a huge M intake and a hot body.

4.5 TECHNOLOGY: ARTIFICERS, CYBORGS, FABRIALS, MACHINES
- ARTIFICER: the device holds the pattern (3.5). Power comes from stores, the pattern from carving or formula. Devices have rated power, wear, false-alarm rates, memory bits (ROM carved, PROM write-once, flash with wear).
- CYBORG (engineering M8): the implant adds a TRANSLATOR T: neural features -> grammar nodes, with channel count, decode error, latency, power draw, heat limit and wear. Facts: BCI rates above. Learning to use it: within-manifold maps in days, outside-manifold slowly (Sadtler et al. 2014 (k)). Cochlear implants reach good speech understanding in 1-3 weeks for many users (k). Implant heat: tissue may not rise more than about 1 degree C (ISO 14708, 2 degrees at the implant surface (k)). Implants wear out: roughly half the Utah-array channels usable after about 5 years (u). Prosthetic hands: 1-3 degrees of freedom of control, pattern recognition 6+ gestures at 90-95% in the lab (k).
- THE HARD LIMIT. The implant cannot carry more bits than the human brain can supply (about 10 bits/s) unless the device itself holds a macro. Real cyborg power comes from STORED programs (focus and macros) and from the device running the fast loop (lag about 1 ms against 200 ms for a human visual-motor loop; human visual tracking delay 150-250 ms (k)). Take the human out of the fast loop: that is the true engineering advantage. Trade: power for fragility. Jamming raises error. Heat limits power. Fatigue kills channels.
- Tool embodiment: the body schema stretches to include tools after training (Iriki 1996 (k)).
- GOLEMS and animated objects (critic D7): need a sensor, a controller, an actuator and a power budget. A golem is a device with a stored program or a bonded mind. It is not a free will.
- FORMULAE and DEVICES as media: typing 6-30 bits/s; a written formula persists for hours; a device runs a stored expression at its own rate.

4.6 TWO MINDS, A WAND, A CIRCLE
- Wand (Harry Potter): a focus plus a shaper. "The wand chooses" = the pattern in the wand overlaps well with the caster's pattern (coupling high).
- Circle, ritual, formation (FMA circles, Ars Magica rituals, cultivation arrays): a diagram on a place (an inscription on the ground) plus a group (parallel lanes) plus time. Slow, large, persistent.
- Choir or group: lanes with sync error and voting.

## 5. Alchemy And Materials

5.1 M-ACTIVE MATERIALS ARE CHEMISTRY

The coupling g of a material is not a tag. It is set by structure: purity, lattice order and dopant, exactly as real transducer materials work. A few ppm of dopant changes a semiconductor by orders of magnitude (k); a few percent carbon turns iron to steel (k). Purer, better-grown crystals couple more. So crystal growth, doping and annealing are the craft, and these are state variables set by the path of operations (chemistry M7). The sim already has heat, mass and phase; the alchemist's skill is running them well and measuring the result.

The philosopher's stone is a CATALYST for M-to-matter coupling (chemistry section 4): it lowers a barrier, is not consumed, can be poisoned, and creates no energy. Catalysis law: Arrhenius 1889, k = A*exp(-Ea/RT) with R = 8.314 J/mol/K; lowering Ea by 50 kJ/mol at 300 K speeds the rate by exp(20), about 5e8 (k, arithmetic). Rule of thumb: rates double or triple per +10 K (k).

5.2 UNIT OPERATIONS (historical alchemy is real chemistry)
Each acts on the sim's mass, heat and phase ledgers. A recipe is a PATH through these. No recipe stores a hard-coded result.
- Distillation: separates by boiling point. Ethanol 78 C, water 100 C. Cost is latent heat (water 2.26 MJ/kg, ethanol 0.84 MJ/kg (u)).
- Calcination: heat a solid in air. Metals gain mass as oxides (Jean Rey 1630; Lavoisier 1789 explained oxygen). CaCO3 to CaO + CO2 needs about 178 kJ/mol above about 900 C (u).
- Sublimation: solid to gas to solid. NH4Cl about 340 C; HgS gives Hg on heating (u).
- Fermentation: sugar to ethanol + CO2, about 90 kJ/mol of heat (u). Yeast is a living catalyst.
- Precipitation, crystallisation (supersaturation gives sudden autocatalytic growth), digestion (aqua regia dissolves gold because chloride complexes it and nitric acid oxidises it), amalgamation (Hg dissolves gold and silver), cupellation (lead absorbs base metals; a real assay).
- The people: Jabir ibn Hayyan (attribution disputed (u)), al-Razi about 900, Paracelsus ("the dose makes the poison"), Brandt 1669 (phosphorus from urine), Boettger 1708 (porcelain), Boyle 1661, Newton's alchemy manuscripts, Lavoisier 1789 (conservation of mass). Chinese alchemists found gunpowder while hunting immortality elixirs, 9th century (u).
- Apparatus: athanor (steady furnace), water bath (balneum Mariae, fixed T near 100 C), alembic, crucible, cupel, mortar.
Order of steps, dose and purity matter because the rate laws and equilibria are real.

5.3 POTIONS: PHARMACOLOGY AS THE DELIVERY LAW (chemistry M8)
- Absorption: bioavailability F (0 to 1), first-order rate ka. Inhaled: seconds; IV: F = 1 and instant; oral: 30-120 min to peak (u). Skin and lung are different routes.
- Distribution: volume Vd = dose/C0; body water about 42 L at 70 kg (u).
- Elimination: usually first order, C(t) = C0*exp(-kt), half-life ln2/k. Caffeine about 5 h, ibuprofen about 2 h, diazepam 20-100 h (u). Steady state after about 5 half-lives. Ethanol is zero-order, about 7-10 g/h (u).
- Effect: Hill equation, Emax*C^n/(EC50^n + C^n) (Hill 1910 (k)). Therapeutic index TI = TD50/ED50; narrow TI (under 3) for warfarin, digoxin, lithium (u). Haber's rule C x t = constant for some gases (1924 (k)). Tolerance and interactions are real.
- ENERGY POTION: a solution of fuel or charge. Energy to the body per hour = dose x k_abs, bounded by the body's heat limit. A "poison" is a narrow-TI potion. An "antidote" is a competitive inhibitor or a binder (chelators bind metals; Ki sets potency).
- PATTERN POTION: a dissolved or suspended structure holding an M pattern. It decays with a half-life (shelf life), needs a carrier to survive light and heat, can be contaminated, and has the same Hill curve at the target.
- Competitive inhibition: v = Vmax*S/(Km*(1+I/Ki)+S). Species binding: fraction of free sites = 1/(1 + sum(I_j/Ki_j)). Irreversible poisons remove sites permanently.

5.4 HEALING AS BOOKED TISSUE ASSEMBLY (biology P13)
Healing magic is lawful, and its bill is real:
- New tissue mass comes from the body or from food. Energy = stored (about 5 kJ/g of muscle (u)) + assembly (about the same (u)). Regrowing 0.5 kg of muscle is about 5 MJ and about 100 g of protein (u).
- Waste heat is 30-50% (u), so fast healing overheats the patient to the same 1.2 MJ cap.
- Real rates (all (u)): skin edge advance 0.5-1 mm/day; skin closure 7-14 days; bone 6-12 weeks; nerves about 1 mm/day (30 cm is about 300 days; central nerves do not regrow); liver 1-2 weeks in rodents, 3-6 months in humans; salamander limbs in weeks to months. Burns raise metabolic rate up to 1.5-2x.
- Controls: blood supply, cell cycle (12-24 h), infection, not only energy. A speed multiplier k on healing raises error and cancer risk by about k (Peto's paradox discussion (u)).
- So magic supplies assembly work and speed. Mass and heat stay booked. A healer who heals 1 kg of tissue needs food or fat to spend.

5.5 TRANSMUTATION (chemistry section 6)
- Lead to gold is lawful. Cost: binding energy about 1636.4 MeV (Pb-208) and 1559.4 MeV (Au-197) (u), about 77 MeV per atom at the very least. Per gram of gold: 3.8e10 J (38 GJ/g). The critic recomputed it from atomic masses: about 79 MeV per atom and 3.8e10 J/g. That is about 40 fireballs per gram at 100% efficiency; an ounce is about 1.2 TJ. It is rare by physics, not by a flag.
- Real: Seaborg's group at Berkeley turned a few thousand bismuth atoms into gold in 1980 using the Bevalac (Scientific American report found by the chemistry lane (v for existence, cost claims (u))). Soddy, Rutherford and Aston identified transmutation by decay (Nobel 1908, 1921, 1922 (k)).
- Neighbours on the binding curve (iron) cost less per gram, since binding energy per nucleon varies little (about 8 MeV per nucleon). A table of costs per element pair is enough (cheaper estimating law). Leftover radiation goes to the radiation and heat ledgers.
- The cheap neighbour: chemistry-only "transmutation" (alloys that look like gold, gilding, assay tricks) costs kJ to MJ and is real. The sim tells them apart by density (gold 19.3 g/cm^3; Archimedes).
- Mass creation from nothing is not lawful: 1 kg is about 9e16 J (E = mc^2); 1 g is 9e13 J. So "conjuration" in practice is GATHERING from the surroundings (Create Water condenses water from the air). For true creation see the mass bank (section 9).
- FMA: reshaping matter costs bond energy, about 1-5 eV per atom (u), at most MJ/kg: lawful as a "change shape or state" effect paid from the budget.
- Mistborn hemalurgy: moving a copied slice of a dial value, with loss, held by a physical spike. No flag. (Critic B6.)

## 6. Interactions Between The Kinds Of Magic

All of these are laws already in the pieces. The critic asked for an interaction matrix (D11). Here it is, then the details.

THE INTERACTION MATRIX (rows act on columns):

  acting \ on     store              wave (pulse)            material               mind                 life/tissue            fire
  store           shared M draw;     discharge emits a       leak heats it;         energy potion        transduction heat      a spark is a nucleus:
                  brown-out          pulse                   over-drive cracks      overheat             and patient cap        trigger on fuel
  wave            induced currents   interference;           absorb: heats it;      neural bias on       conversion at site;    heats, ignites;
                  disenchant;        reflection; resonance   reflect; shield;       dials (6.4)          M shock               Frank-Kamenetskii
                  charges a store    beats                   anti-magic region                           (harm, heal)           size effect
  material        catalyst poisoning shapes the wave (lens,  galvanic trade,        poison (narrow TI)   chemical               fuel, quench
                  of a transducer    mirror, waveguide)      passivation (wards)
  mind            decides which      sets phase, aim and     (carving, brewing)     persuasion, fear     cast on tissue         (acts)
                  gradient, when     frequency                                      without M (6.7)
  life            M source (eta x NPP); sensing organs         M-rich/poor material   M-sense in beasts    over-harvest kills     --
                  draws on stores    detect pulses                                                        the grove

6.1 INTERFERENCE (wave x wave). Coherent pulses add in amplitude per cell; the book is kept per cell (2.3). Counterspells are lawful but local, narrow-band and need phase knowledge. Standing waves, beats between engravings that share f0 (critic D11), diffraction through an aperture (the "magic wave theory" proposal of ToM 2: a Gaussian-like Airy pattern) and focusing (spot at least lambda/(2 NA), depth about pi*w0^2/lambda) all come out of the ray march plus a few explicit rules for declared coherent sources.

6.2 CONTAMINATION AND POISONING (store, material). Catalyst poisoning (chemistry section 7): S, CO and Pb adsorb on Pt sites and stop it (Pb in petrol killed catalytic converters (k)). The amount needed is tiny, set by site count, and it cuts the RATE not the energy. This is the real "anti-magic metal": a trace that poisons the coupling. A few ppm of dopant changes a coupling a lot in either direction. Other templates (chemistry):
  - Galvanic corrosion: two M-transducer materials in contact in a medium trade charge; the lower-potential one is consumed at a rate about dV/R_medium. Fe against Cu is about 0.78 V (u).
  - Passivation (Al, Cr, Ti oxide): a ward that self-heals then shuts off more attacks.
  - Quenching (Stern-Volmer: I0/I = 1 + Ksv*[Q]): a dissolved species that takes the excited state's energy without light kills spell light.
  - Le Chatelier: an equilibrium pushes back when you drain it.
  - Acid + base release heat (neutralisation, about -57 kJ/mol (u)). All mixings conserve atoms and energy, so they are ledger-clean.

6.3 COMPETITION FOR M. Two casters on one cell get a brown-out: the draw is shared by coupling and distance, and the cell depletes. A crowd casting at once in a poor place gets less each. A pocket of M (a body store) is not shared. Over-harvest of an M-rich ecosystem lowers local biomass and so future M (6.5).

6.4 MAGIC ON MINDS (wave x mind). The sim has no neurons. So mind magic is a general NEURAL BIAS effect form that moves dials the sim already has: fear, trust, attention, mood, sleep. Tiny energy, strength set by g and signal-to-noise, resisted by the target's noise (critic D3). A neural spike is about 1e-12 to 1e-10 J in the fiction lane (u) and about 1e-9 J in the physics lane (u): the two disagree, so energy is "negligible, order nano- to pico-joules". Reading minds is the same coupling with about 10 bits/s capacity (Zheng and Meister 2024). Illusion is the same plus light shaping. Whether memory and intent can be reached is a decision (Q5).

6.5 MAGIC ON LIFE. A body is an M sensor and source. Over-harvest kills groves (7.3). Healing and harm are the transduction law applied to tissue.

6.6 MAGIC ON FIRE AND COMBUSTION. A spark is a nucleus: the magical ignition event aimed at a metastable store (fuel, supercooled water, a charged crystal, a loaded M pocket). "Fireball" releases heat into fuel and the fuel burns by the existing law. The Frank-Kamenetskii number (slab 0.88, cylinder 2.0, sphere 3.32 (v)) decides whether a pile self-ignites, so a clever caster sees which pile is near critical.

6.7 MAGIC ON ELECTRONICS AND ENGRAVINGS. Faraday induction: emf = -dPhi/dt (k). A pulsed M that induces currents in structures can burn engraved traces: that is DISENCHANTING, and it is why a strong caster fouls "electronics" (Dresden). A pulse spreads as I = I0*exp(-mu*x)/r^2 and ruins engravings whose traces are thinner than a few skin depths. Skin depth delta = sqrt(2*rho/(w*mu)) (k): copper 8.5 mm at 60 Hz, 66 micrometres at 1 MHz, 2 micrometres at 1 GHz (k). Shielding effectiveness about exp(-t/delta), about 8.7 dB per skin depth; a mesh with holes under lambda/10 works (u, rule of thumb).

6.8 COUNTERSPELLS, WARDS, ANTI-MAGIC
- COUNTERSPELL = destructive interference: equal amplitude, opposite phase, a zone about lambda/10, needs the incoming wave's phase (so a Sense node first). It does not delete the energy; the canceller's source pays work.
- DISPEL = Vim: Form = M itself, an effect on M: drain, disperse or store. Absorption heats the absorber, booked.
- SHIELD = reflection (impedance step) or absorption (large mu_M), shaped by a lens or mirror; shielding is frequency-dependent: a cage tuned to one band leaks in another. A conductor-like "M-conductor" reflects; a lossy one absorbs and heats.
- ANTI-MAGIC = low local coupling (a region of low g, or a high mu_M absorber, or a poisoned catalyst). Raised "tension" in the Gygax field paper is the same thing in other words.
- SHIELDING A PERSON (Wheel of Time): cut them off from the source: a bubble of low coupling around them.
- DISENCHANTING: induced currents burn traces (6.7).
- CLOAKING (invisibility): narrow-band, lossy transformation optics (Pendry, Schurig and Smith, Science 312:1780, 2006 (k)); works only in a band.
- A patron cutting a channel mid-cast, a heat limit or an interrupted cast are other counters, all from the laws.

## 7. Places And Ecology

7.1 LEY LINES AND PLACES. S(x) is data in the world: a ley line is a source term of watts per cubic metre; a node is a spot of high M0. The M0(x) map sets how magical a place starts. Hungry spirits cluster at strong sources. Because of the second-law guard, the source has a booked origin (planet, sun, life, patron, other world) with a pack rate map. A "sealed" world with zero import is allowed: then M only falls and magic fades.

DESIGN DEFAULTS FOR A MODERATE WORLD (not sourced; chosen for fit to the 0.9 GJ fireball):
  average M: 0.05 MJ/m^3; ley line or node: 1 MJ/m^3; sealed or dead zone: 0.
  A fireball (0.9 GJ) in an average place draws from a sphere of radius about 16 m (4/3*pi*r^3*0.05 MJ/m^3 = 900 MJ gives r = 16 m); in a ley line, from about 6 m. So a fireball in the open drains a big hole and is almost impossible without a store; a store (a 9 g great gem) makes it possible anywhere.
  Refill: a 100 m cube at 1 MJ/m^3 holds 1e12 J (1 TJ, about 1000 fireballs). A refill of 1% per day is about 116 kW. That is tiny against sunlight (about 1 kW/m^2 on 1e4 m^2 is 10 MW), so a slow M refill is plausible.

7.2 DEPLETION AND REFILL. Casting draws M in its reach volume at a capped rate (2.6). A cell then relaxes: M = S/k + (M0 - S/k)*exp(-k*t). Diffusion fills the hole slowly (D small). Diffusion and relaxation are lazy (2.7). So casters leave visible "dead patches", and a wizard's tower in a poor place is a poor tower. M can be farmed (storing, recharging crystals from a ley line or the sun) or gathered.

7.3 LIFE AS A SOURCE (biology P12). If M is made by life, M_source(place) = eta_life x NPP(place), plus a geological background.
  Global net primary production about 105 Pg C/yr (Field et al., Science 1998: land 56.4, ocean 48.5 (u)).
  Per area per year (all (u)): rainforest about 2,200 g dry/m^2; temperate forest about 1,200; grassland about 600; desert about 90; open ocean about 125; estuaries and reefs about 2,000.
  Plant dry matter is 17-20 MJ/kg (u). A rainforest then gives about 40 MJ/m^2/yr, about 1.3 W/m^2 mean, against about 170 W/m^2 of sunlight at the ground: NPP is about 0.5-1% of sunlight; the theoretical maximum photosynthesis is about 4.6% (C3, Zhu, Long and Ort, Curr Opin Biotech 19:153 (2008) (k)).
  With eta_life = 0.01 (design default; the range 1e-3 to 1e-1 is a pack knob): 13 mW/m^2 in a rainforest, 130 W per hectare, about 400 W for a 100 m radius patch. So even ecological M is a trickle.
  Trophic efficiency about 10% per level (range 5-20%); endotherms put only 1-3% of intake into new body mass (Humphreys 1979 (u)).
  Druid harvest: drawing M from a living system removes growth. Over-drawing kills the grove: M and biomass fall together. A natural tragedy of the commons. Decomposers return M-bearing mass; the world needs a recycle path or M leaks. Critic A3: life-derived M is a real conversion and removes biomass.
  Biomagnification: beasts that eat M-rich plants concentrate it at about 10% per step.

7.4 MANA WEATHER. With slow drive and threshold release, M builds up and discharges in avalanches with a power-law size distribution (Bak, Tang and Wiesenfeld 1987 (k)): many small, few large. This gives mana storms and wild surges with no new code (a pack option; physics P14). Real analogues: snow slab release, sandpiles. Stormlight highstorms are a pack-set periodic source; a moon-modified coupling (Avatar) is an environmental modifier of g. A hot spot of M leaks heat and glows, so a high-M place is a detectable place (critic D11: "M leak heats and glows the place").

7.5 THE TELL AND THE SENSE. Reciprocity means a caster leaks (a faint field, light, sound, M). Beasts and people with high g sense it: antenna theory (a good transmitter is a good receiver; Onsager 1931 (k)). So casting has a detectable tell, M-rich things attract M-sensitive creatures, and ambient M can be sensed at tiny energy (a gradient read, not a power transfer).

## 8. Limits, Costs And Failure Modes, All Emergent

8.1 THE LIMITS CLUSTER (fiction lane section 2): energy budget, innate dial, skill, casting time, components, range, knowing the target, counter-magic, overreach hazard, oaths. All appear below without a rule.

- OVERHEATING. Line 1 heat in the holder; Line 2 at the site. Body core 37 C; heat exhaustion at 39-40 C; heat stroke above 40-41 C; protein denaturation above 42-43 C (u, clinical). Body heat capacity about 3.5 kJ/kg/K (u): 245 kJ per K for 70 kg, so a 5 K rise is about 1.2 MJ. Heat shedding: about 100 W at rest; sweat up to about 0.7-1.4 kW in dry air (u); latent heat of water 2.43 MJ/kg. So a caster that wastes 50 kJ per second at 30% efficiency overheats in minutes. Devices heat the same way: a wand cannot deliver a MW continuously. Pulse spells are cheap in heat (duty cycle); sustained ones hit the dissipation wall. A cooling spell costs work: COP <= Tc/(Th - Tc) (k) (cooling a body from 310 K to 270 K at 300 K outside has a maximum COP of about 6.75 (u, calc)) and warms the surroundings, booked.
- FATIGUE: THREE CHANNELS (biology P3, section 3). Store (metabolic: glycogen, PCr half-recovery about 30 s (u)), heat (core temperature), attention (central fatigue). Each recovers at its own speed. Sleep about 8 h (u). Oxygen debt (EPOC): minutes to hours (u).
- POWER LIMIT. Delivered power <= min(mobilisation rate, heat shedding rate, coupling throughput). A human sustains about 300 W chemical (fit athletes 200-400 (u)); peaks about 1-2.5 kW for seconds; fat oxidation about 0.5 g/min (u).
- DEPLETION. Local M falls where it is used. A caster in a dead zone has no field to draw from and falls back on the body and on stores. A store runs out.
- BURNOUT AND STILLING. If the draw through a coupling exceeds its rated throughput, damage D accumulates on the tissue transducer by Miner's rule (3.5). At D = 1 the coupling g falls permanently, or the channel dies (Wheel of Time stilling; the cultivation "qi deviation", the meridian damage). Partial D adds error and cuts eta first.
- CRACKING AND MELTING. Over-driving a crystal or engraving heats it by (1 - eta)*E_in until it depoles, cracks or melts. Piezo ceramics lose polarisation above the Curie temperature (k).
- TRIBULATION. A store driven past its density cap or a metastable store pushed past its threshold discharges (cultivation tribulation, a lightning release): a lawful over-dense release.
- LEAKAGE. Stores leak (tau), spirits fade (decay), enchantments wear (D), potions go off (half-life).
- FAILURE BY ERROR (mishaps as neighbouring valid spells). Each node has a chance e of an error scaled by stress, injury, fatigue, low skill and ambient field noise. An error draws from the medium's confusion kernel and lands on the nearest valid value. A Wild Magic surge or a "wrong word" is the same thing. With check bits, a detected error is a FIZZLE (the energy is not delivered, small waste); an undetected one gives a wrong-but-valid spell (dangerous). Ratio is a data parameter: more check bits equals more fizzles and fewer wrong spells. Undetected error probability about 2^-c for c check bits (k). Harris-Wolpert noise grows with output (u): stronger casts are sloppier.
- THE ALLOWED-BUT-EXPENSIVE: A big spell cast by a weak medium takes long, is interruptible, and is delicate. Real fiction systems agree: wizard rituals are slow (low-rate codes), battlefield spells short (fast and fragile).
- OATHS AND BOND LIMITS. Break the oath: the channel closes. A patron's price: the exchange rate; sanctions.
- NO INFINITE LOOPS. A cycle of conversions each below 100% dies (the product of efficiencies is below 1). The engine saturates at the store. No cascade can exceed it.
- LUCK (critic A7). Luck magic is lawful only as a tiny push on a physical metastable thing (a spark, a balance). The sim must really have sensitive dependence. Never bias the RNG.

8.2 SPECIFIC CASES THE CRITIC CHECKED
- SHAPESHIFT. A change of 70 kg to 70 kg still needs rearrangement energy: about 5-10 MJ per kg regrown (biology) plus a 1.2 MJ heat cap. A full change is hundreds of MJ. So it is slow (days, heat shed over time) or the heat is dumped to a store. Size change needs mass (the mass bank).
- SUMMONING. "Fetch a real being by fast motion" is a kinetic launch, not a summon, and is removed from the build list (D6). Assembling a golem from local matter is lawful.
- ELEMENTAL BENDING. A campfire gives little work (D8). Bending steers matter that is ALREADY MOVING, with an M force; the local element supplies the mass.
- LIFE DRAIN. A heat engine at body temperature is 3% efficient (Carnot, 310 K vs 300 K), so life drain is a chemical transfer of stores, not a heat engine.
- LEVITATION: the reaction partner rule (2.5).
- LOCKED OUT: about 55 of the roughly 60 D&D damage spells follow from one primitive (force, momentum, fire, cold needs a sink, thunder, radiant, lightning as fast energy delivery, necrotic as an energy drain). The rest need mass moving or sensing (healing, divination, acid, poison), a neural model (psychic), or break conservation (conjuration, teleport, time, Wish): the spell dictionary reports.

## 9. The Coverage Catalogue

KEY. Each row lists the archetype, what builds it from the pieces (E = energy source, P = pattern holder, I = interface/medium, D = delivery), and a verdict.
OK = built from the pieces, only pack numbers. OK+ = needs a small general law (not physics). 2nd = needs a second change, pack-only.

  # archetype                    built from (E | P | I | D)                                                            verdict
  A1 innate sorcery              body + M | body dial (high g, low finesse) | will (thought) | burst, zone               OK
  A2 learned wizardry            M + body + stores | trained mind + book | voice, gesture, components | any             OK
  A3 granted / patron            patron's store | patron's pattern through a bond | pact, prayer, oath | what is lent    OK (oath closes the channel)
  A4 bonded / familiar           shared store | partner mind | oath | shared senses, boosts                             OK (M-sharing bond)
  A5 item / enchantment          charged material | engraving | trigger | fixed effect                                  OK
  A6 ritual / circle             M + group + stores + time | diagram on a place | long rite | large, persistent          OK
  A7 alchemy                     chemistry; M in a substance | the substance's structure | brewing | ingest or apply     OK real chemistry; OK+ for pattern-bearing materials (same law as crystals)
  A8 technology / artificer / cyborg   batteries | engraved circuits, implants | switch, code | any                       OK
  A9 blood / life-force          bodies' energy | caster or altar | cutting, ritual | any, short range                   OK (one body = about 0.37 GJ, so big effects need many bodies: physics gives the dark trade-off)
  A10 necromancy                 M or remnants | corpse + controller, or stored mind | ritual | motion of a dead body   OK for a puppet; resurrection 2nd
  A11 shamanic / spirit          place or spirit store | spirits | dance, drum, offering | spirit acts                   OK+ (a spirit is a dissipative pattern, 3.1g; needs Q4)
  A12 divine / clerical          as A3; belief as tribute through minds                                                OK; belief as a free field 2nd
  A13 wild / chaos               high M | none | none | random, mutation                                                 OK (high M plus noise, avalanches)
  A14 cultivation                M + stones + pills | dantian (store) and meridians (throughput) | meditation, manuals | body boost, flight, strikes   OK (proven by cultivation.yaml)
  A15 rune / sigil               as A5 with carving as the medium                                                      OK
  A16 song / music               M + body | performer | sound | area effects, waves in M, resonance                      OK (choir stacks amplitude)
  A17 true name / word           body + skill | knowing mind | voice | exact target                                      OK (name = a perfect pattern measurement; resonator link, 3.4)
  A18 martial / gesture bending  body + element | trained body | forms | push nearby matter                              OK
  A19 elemental / nature         local conditions as source | caster | gesture, focus | push or direct                   OK (local matter supplies mass; M steers; D8)
  A20 mind magic                 tiny energy | caster | thought | neural bias on dials                                    OK with the neural-bias effect form (6.4); reach is Q5
  A21 healing / life             body energy or M | caster | will, touch | faster repair                                OK (booked mass, heat and food: 5.4)
  A22 self-sustaining pattern / spirit / ghost / god / golem will | field M feeds it | a pattern in the field | contact with minds | its own acts   OK+ if minds always have a substrate (Q4); else reflex-only
  A23 shapeshift                 body mass and energy | caster | will | body change                                     OK at the same mass, slow (8.2); size change needs the mass bank (2nd)
  A24 summoning                  M + mass | caster | ritual | assemble                                                  OK to assemble from local matter; fetch by fast launch is just a launch; create 2nd; instant fetch 2nd (relocate)
  A25 wards / anti-magic         stored M | engraving or caster | trigger | reflection wave or low-coupling zone         OK
  A26 divination / scrying       small | caster or stone | thought | information at distance                            OK with a physical path: an M wave out and an echo back (1/r^4 like bats) or a linked resonator at the far end (3.4). Future 2nd. Past: OK if events leave physical traces
  A27 time magic                 --                                                                                    2nd (rewind or per-region clock)
  A28 teleport / gates           --                                                                                    2nd (relocate)
  A29 luck / fate                tiny push on a physical metastable chaotic system                                     OK+ (never bias the RNG)
  A30 curses / hexes             a pattern in a body that drains M or applies a field | store + link (hair, blood) | ritual | OK
  A31 soul magic                 energy part OK; a separable immortal soul                                             OK for energy; separable soul 2nd (planes)
  A32 death bargain / sacrifice  payment of limb or life | as A9 and A3                                                OK
  A33 rest-to-recharge (slots)   refill of the body store (food) or local M draw                                       OK (a UI block of joules, 2.3)
  A34 rare-material dependence   material sets g or store of an engraving                                              OK
  A35 mundane craft (headology, hedge medicine, cold reading)   real physics and psychology                         OK without M
  A36 will / belief reshapes the world   mental state as the efficiency of one's own link: OK; true reality editing    2nd, or route all belief effects through minds (morale, tribute, placebo)

Additional archetypes from the critic (D10) and how they build:
  weather control            steer matter already moving (air masses), with M force; scale needs stores or ecosystem M (weather moves about 1e14 W); OK but huge
  plant growth               mass from soil and water, with energy from M or sunlight; OK, booked
  buffs (strength, speed)    raise ATP and heat limits; bounded by the body caps; OK
  invisibility               narrow-band transformation optics (Pendry 2006): OK, band-limited and lossy
  life drain                 chemical transfer of stores: OK
  curses, disease (lycanthropy, vampirism)  a replicating pattern (physics P12) plus the biology parasite law: OK
  disenchanting              induced currents burn traces (6.7): OK
  sound and illusion         pulses and light shaping: OK
  long-range communication   Shannon capacity on an M channel: OK; instant speed needs c_M very large

WHAT NEEDS A SECOND CHANGE (PACK-ONLY; keep out of the base):
  N1 Teleport, portals, gates, instant summon (nonlocal transport of matter). Minimal law: a booked RELOCATE operation with a pack switch and an ordering rule (applies at the end of the tick; critic D9). Cost: a booked energy cost and a conservation check on the swapped region. The pack must declare the cost.
  N2 Time manipulation and prophecy. Prophecy can be built in the MIND (a fast world-model run ahead, like the world-tick note), not physics. Rewind is a checkpoint feature; recommend leaving it out.
  N3 Mass creation (create from nothing, new creature, size-change). E = mc^2 makes 1 kg about 9e16 J, so true creation is out. The clean option is a MASS BANK that converts M at a stated pack rate (so 1 g is 9e13 J at 100%); decide in section 11.
  N4 Resurrection and restoring a destroyed mind. With a stored copy (a gem as a data store, with bits per kg) it is OK+; without it, 2nd.
  N5 Faster-than-light information: nearly all play is covered by c_M very large; true instant needs 2nd.
  N6 Belief or story changing reality: route through minds. No second change.
  N7 Other planes, afterlife, a separable soul: a second connected world with its own M is a world-level object, not a law. Flag for the user.
  N8 Probability alteration: see A29.
  N9 Nuclear transmutation: not a second change; booked at about MeV per nucleus.
  Paradox (Mage): social belief resisting reality is a second change unless it is only social reaction.
  Discworld's story-driven reality: 2nd.

THE FICTION'S DEEPER LESSONS (which this design follows):
  1. Verb-noun is real: five verbs (release, sense, transform, disperse, move) times nouns (body, mind, heat, water, air, earth, plants, light and sound, M itself). Vim = acts on M.
  2. Parameters recur: magnitude, duration, range, area, casting time. Our ledger gets the price from physics.
  3. A link-quality dial recurs (Kingkiller sympathy, Earthsea names, Ars Magica arcane connection). Here it is resonator fidelity.
  4. Overreach hurts the pattern holder in nearly every system.
  5. Every system has counters: interference, reflection, low-coupling zones.
  6. Persistence needs a store or a feed.
  7. Patron structure covers warlock, cleric, paladin, spren, Mantle with one record.
  8. Sanderson's Laws (2007-2013): limitations beat powers; expand what exists before adding something new; every archetype is a parameter setting, never a new law.
  9. Packs differ in numbers only.

## 10. What Packs Set (The Parameter List)

RULE (critic B and Q11). Numbers only. A pack may OMIT a medium row (set its coupling to 0), or leave a material without an M row, but may not switch a behaviour on or off. cultivation.yaml already proves this works as data.

FIELD AND WORLD
  M0(x), S(x) and origin class per source (planet, sun, life, patron, other world), D, k, drift
  c_M (speed of the fast mode); dispersion exponent; noise floor of the field
  mu_M and Z_M per material and band; rho_M (M-resistivity)
  g per material (from purity, lattice order, dopant); anti-magic regions
  eta_life and NPP map; trophic efficiency; recycle fraction
  SOC threshold and relaxation rule (mana storms)
  Optional: the share of conversion waste going to the holder (default 0); the share of recoil booked to the field (default: ground first, else caster)

TRANSDUCERS
  per kind: eta, light share, threshold, response time, rated power, max temperature, bandwidth, noise floor, wear constants (S_rated, b, N_ref, T_ref and activation energy)
  draw-rate law constants (tau_draw, V_reach per transducer)
  the materials table rows (class, in, out, coupling)

STORES
  density classes in MJ/kg with a cap, barrier Ea, leak tau, max discharge power per kg, trigger site, charging eta
  catalyst effects; critical size shapes

MEDIA
  per medium: rate, dims, sigma, e_relaxed, e_stressed, k_m, loudness, body power, persistence, coupling c_m, confusion kernel
  lanes for choirs and circles: sync error, voting rule

GRAMMAR
  node priors (common spells short); macro list; depth limit; leaf quantisation (bits per leaf); the naming table (words for regions of the number space)

BONDS
  bandwidth, latency (from c_M), exchange rate, sanction strength, observation noise, lock-in, authentication bits

PEOPLE AND BEASTS
  g distribution, finesse distribution, h2 (default 0.5), jump chance and size (default 1%), practice exponent b (0.2-0.5), plasticity window ages, size of the body store, throughput cap, wear on coupling
  mutation rate per parameter per generation; upkeep curve; sense-before-act ladder; convergence allowed
  heat cap, heat shedding, fatigue recovery rates (three channels)

SELF-SUSTAINING PATTERNS
  intake W, leak 1/s, repair rate, boundary strength, bits, substrate bits per kg, topological barrier E_b, Kleiber exponent (default 0.75)

ALCHEMY
  per material: phase constants, solubility curves, latent heats, reaction dH and Ea; M-structure state variables; transmutation cost table per element pair (eta_nuc); mass-bank rate (if enabled)
  per drug: route, F, ka, Vd, ke, EC50, n, TI

SECOND-CHANGE OPTIONS (off in the base, per pack)
  relocate (cost, ordering rule); mass bank (rate); other planes (a world); rewind (checkpoint)

PACK EXAMPLES, by numbers only:
  D&D-like: moderate M0, ley hot spots, light-like c_M, sorcerer corner with 1% jumps, patron bonds, slots as UI.
  Cultivation: high S at veins, strong body store (dantian) and throughput (meridians), pills and stones as stores, tribulation from over-dense release; the file exists.
  Low magic: M0 = 0, only beasts' reflex tricks via stores.
  Hard science: sealed (zero import) world, slow c_M, conversion waste share to the holder.
  DCC and Beware of Chicken style LitRPG: the "system" is a reaction pack of data on top; no engine code.

## 11. Open Decisions For The User

Each has options and a recommendation. These are the critic's Q1-Q12 plus a few new ones.

Q1. WHERE DOES CONVERSION WASTE HEAT APPEAR?
  Options: (a) at the holder (current 8.2: kills the caster in any large cast); (b) at the site (critic's fix); (c) a pack-set split.
  RECOMMEND (c) with default (b): control overhead to the holder, conversion waste at the site, a pack share to the holder defaulting to 0. This is the most important decision.

Q2. ORIGIN OF M REFILL?
  Options: planet and sun only; life only; a mix; a sealed world with zero import.
  RECOMMEND a mix with one ledger line imported_M[origin], a pack rate map per origin, and sealed worlds allowed. Print the daily import.

Q3. DEFAULT c_M AND RECOIL?
  Options: light-like (arrival next tick; recoil negligible); about 1 km/s (visible, dodgeable bolts; 0.9 GJ recoil is 9e5 kg m/s, fatal); about 100 m/s (very slow; the critic's number).
  RECOMMEND light-like by default, so decision 8.3 holds as written, with slow-M as a pack option and the reaction-partner rule for force effects.

Q4. MUST MINDS HAVE A SUBSTRATE? CAN A SPIRIT THINK?
  Options: (a) every mind has a substrate with a bit capacity per volume (body, gem, stored pattern); (b) spirits are reflex-only dissipative patterns, no thought; (c) a spirit can think with no substrate (breaks the information accounting: a scalar M holds no bits).
  RECOMMEND (a) plus (b): minds need substrate bits; free spirits are reflex-only, so a thinking spirit lives in a gem, a body, a place engraved with a pattern, or a network. A god is a big substrate with many bonds. A brain holds on the order of 1e15 synapses (u), which sets a scale for bits per kg.

Q5. HOW FAR DOES MIND MAGIC REACH?
  Options: dials only (fear, trust, attention, mood, sleep); dials plus memory and intent; full thought reading.
  RECOMMEND dials only in the base, with memory and intent as a pack-only extension that reads at about 10 bits/s.

Q6. WHICH SECOND CHANGES?
  Options: relocate (N1), mass bank (N3), rewind (N2), other planes (N7), none.
  RECOMMEND none in the base. Offer relocate and the mass bank as pack switches with booked costs; leave rewind and planes out until a scenario needs them.

Q7. MUST SCRYING AND LINKS HAVE A PHYSICAL PATH?
  Options: a nonlocal read; a physical path (M echo or resonator link).
  RECOMMEND a physical path. It keeps the ledger honest, and shielding then blocks it. Fiction will want exceptions; make those packs.

Q8. SLOT-STYLE RECHARGE?
  Options: not allowed; allowed as a UI over food and M intake.
  RECOMMEND allowed as a UI block of joules over food intake and local M draw. No free refill.

Q9. HEREDITY?
  Options: smooth only; smooth plus a rare big jump.
  RECOMMEND smooth plus a 1% chance of a big jump in g, so born casters exist.

Q10. DOES THE SIM NAME SPELL ELEMENTS?
  Options: named elements in the engine; numbers with a naming table on top.
  RECOMMEND numbers (heat, light, kinetic fractions, size, duration) with a pack naming table. A neighbour error then lands on a physical neighbour.

Q11. STRICTNESS OF "NO FLAGS IN PACKS"?
  Options: a pack may switch a medium off; a pack may only set numbers.
  RECOMMEND numbers only. Setting a medium's coupling to 0 removes it. Add a test that scans a pack for behaviour switches.

Q12. IS BODY-ONLY MAGIC WEAK ENOUGH?
  Options: thrown-rock scale (about 30 kJ burst, about 300 W sustained, 1.2 MJ heat cap); a harder cap; a softer cap.
  RECOMMEND leave it, as a lawful result of the body numbers. A body is a weak store, which is what 8.1 intended. No separate rule.

Q13. HOW MAGICAL IS THE DEFAULT WORLD? (new)
  Options: low (average 0.005 MJ/m^3); moderate (0.05 MJ/m^3 average, 1 MJ/m^3 at ley lines); high (0.5 MJ/m^3).
  RECOMMEND moderate as the default pack and test M = 0 as a second pack.

Q14. GEM DENSITY CAP? (new)
  Options: chemical-like (1 MJ/kg, a fireball is 900 kg: magic items are heavy); a big jump (1e5 MJ/kg, a fireball is 9 g).
  RECOMMEND a cap of 1e4 to 1e5 MJ/kg for the best class (rarity classes below it), so an item can hold one fireball without being absurd, and pushing it past the cap cracks it.

Q15. HOW DOES THE PLAYER WRITE A SPELL? (new)
  Options: free text parsed into the grammar; a menu of macros with number picks; both.
  RECOMMEND both: macro-first menus (the small model picks from legal choices, depth 1) plus a text translator through the same grammar with an n-best list re-ranked by the grammar prior. Every pick is logged as a typed delta.

Q16. WHAT IS THE FIRST SLICE? (new)
  Options: a fireball from a gem; a body-only spark; a counterspell.
  RECOMMEND the fireball from a gem as the first end-to-end proof, because it tests the ledger, the store, the trigger and the site heat with the fewest pieces, and it uses fire and combustion that already exist.

## 12. A Build Plan In Phases

WHAT THE SIM ALREADY GIVES (waves 12 to 17 plan; docs/realism-map.md and docs/composite-events.md):
  Wave 13 (Ground truths): body heat on the person (core and shell), fatigue, one material record with beam check, fire labels, soot, bleeding, doors. This gives the caster's heat cap, the material rows for g, Z_M and mu_M, and the fire that spells ignite.
  Wave 14 (Seeing and hearing): light field, detecting versus identifying, speech loudness and partial understanding, sustained sounds, odour. This gives the VISIBLE TELL, the light share of spells, sound media (voice 60-90 dB, music, choirs) and perception of casting.
  Wave 15 (Time and scale): clocks, scheduler, skip-ahead, the ENERGY LEDGER WITH BOOKED INJECT (this is imported_M), body clocks (energy and water stores, sleep, healing, infection), bond islands, adaptation. This gives lazy M relaxation by the scheduler, multi-tick effects, food-based refill, healing and the bond records.
  Wave 17 (Living and packs): "magic borders, stores, spells, voxel packs", the corpse, contagion, animal bodies and minds. This is the planned home of magic. This design recommends moving the engine-neutral parts earlier.

PHASE 0, NOW (no sim code): freeze the constants in docs/constants.md; add the M-row to the material record plan (g, Z_M, mu_M, store density, leak tau, depole temperature); write the ledger lines of 2.3 into the wave 15 plan as "reserved". Decide Q1 to Q4 and Q11 (they change data structures). Add the pack-scan test spec.

PHASE 1 (after wave 15 ledger): THE M FIELD AND LEDGER. Lazy M cells with S, k, D, imported_M[origin]; closure check every tick; M = 0 world. Needs: wave 15 ledger and scheduler.
  Tests: (1) closure to rounding for 1e6 ticks; (2) M = 0 gives exactly zero magic and zero ledger lines; (3) a sealed world loses M by k only; (4) a draw and refill test: the hole refills as exp(-kt) within tolerance; (5) printing daily import equals the sum of sources.

PHASE 2: STORES AND TRIGGERS. A charged crystal with density, leak, barrier; discharge into the existing heat laws; the FIREBALL FROM A GEM first slice (Q16). Needs: wave 13 fire and materials.
  Tests: (1) 9 g gem at 1e5 MJ/kg releases 0.9 GJ +/- 1e-9 in the ledger, split work + light + site heat = 1; (2) the trigger gain is reported (E_released/E_trigger); (3) release never exceeds the store; (4) over-driving past the cap cracks it; (5) leak follows exp(-t/tau); (6) a critical-size pile of fuel ignites and a smaller one does not (Frank-Kamenetskii: slab 0.88, cylinder 2.0, sphere 3.32).

PHASE 3: TRANSDUCER LAW AND THE BODY CASTER. The two-line heat rule; body store, power cap, three fatigue channels; draw rate from M reach. Needs: wave 13 body heat and fatigue; wave 15 body clocks (glycogen, PCr, fat).
  Tests: (1) the 0.9 GJ fireball from a gem does NOT kill the caster (site heat 9 MJ goes to the fireball region); (2) a body-only cast stops at about 1.2 MJ of waste and the caster faints; (3) fat 0.37 GJ cannot be spent faster than about 300 W sustained; (4) a cooling spell warms the surroundings by (1 - 1/COP) booked; (5) chain efficiency multiplies.

PHASE 4: MEDIA, GRAMMAR AND ERRORS. The channel table; spell trees with bit costs; cast time; Line 1 heat; the confusion kernel and fizzles. Macro menus for the small model; text translator; typed deltas logged. Needs: wave 14 sound for voice and music loudness.
  Tests: (1) 60-bit spell takes 1.5 s by voice, 6 s by thought, 12 s by gesture; (2) error rises with stress; (3) every error lands on a valid spell (fuzz 10^5 casts: zero invalid expressions); (4) more check bits give more fizzles and fewer wrong spells; (5) a focus lowers bits and error; (6) a pack with a medium coupling of 0 removes it and nothing else.

PHASE 5: PULSES AND WAVES. Ray-marched pulses at c_M with mu_M and Z_M; shields, mirrors, shapers; resonators (f0, Q) and links; interference booked per cell; the echo for scrying. Needs: wave 14 light field (a pulse is similar), wave 13 materials.
  Tests: (1) energy: sum of absorbed + reflected + transmitted + leaked = emitted; (2) a counterspell cancels in a zone about lambda/10 and the energy moves to constructive cells (booked); (3) a matched resonator absorbs strongly, a detuned one weakly (Lorentzian width f0/Q); (4) scrying needs an echo; a shield blocks it; (5) arrival tick = ceil(path/(c_M*dt)), minimum 1.

PHASE 6: ENGRAVINGS AND DEVICES. The three roles; wear by Miner's rule; ROC triggers; focus bit savings. Tests: (1) wear to failure at D = 1; (2) a hair-trigger trap misfires in noise at the ROC-predicted rate; (3) overdrive heats and shortens life.

PHASE 7: BONDS AND PATRONS. The one bond record; the mutualism law; possession; oath as state; the sympathetic link by resonator. Needs: wave 15 bond islands.
  Tests: (1) a bond persists only while payoffs are non-negative; (2) cutting a bond mid-cast delivers only what arrived; (3) a sympathetic link fails if the resonator is shielded or the copy mismatched.

PHASE 8: BEASTS AND ECOLOGY. M from life (eta x NPP); over-harvest; the sense-before-act ladder with upkeep; evolution over generations; apex concentration. Needs: wave 17 animal bodies and minds, wave 15 adaptation.
  Tests: (1) a grove overdrawn collapses with biomass; (2) a lineage gains M-sense before reflex release before stacking; (3) over 1e3 generations different lineages converge on similar organs by different routes; (4) upkeep above benefit prevents the organ.

PHASE 9: ALCHEMY. Unit operations; structure state (purity, lattice, dopant) setting g; potions (one-compartment closed form); transmutation cost table; healing as booked assembly. Needs: wave 16 matter and water, wave 15 body clocks.
  Tests: (1) a purer crystal couples more; (2) a poisoned catalyst loses rate, not energy; (3) potion concentration decays as exp(-kt); (4) lead to gold costs about 3.8e10 J per gram in the ledger; (5) a heal of 1 kg of tissue uses 5-10 MJ and mass from food.

PHASE 10: THE SECOND CHANGES, ONLY IF WANTED. Relocate (booked cost, applied at end of tick) and the mass bank. Tests: conservation holds across relocate; the mass bank costs 9e13 J per gram at 100%.

GLOBAL TESTS (every phase):
  - the closure check on every tick;
  - an A/B run with M = 0 equals the old sim (golden with declared diffs);
  - the no-flags audit: scan every pack for behaviour switches (Q11);
  - the fuzzer: 150 worlds with random casters, stores and bonds, no crash and closure holds;
  - a determinism test (same seed, same ledger);
  - the audit of "free joules": any tick with a net gain not matched by imported_M fails;
  - a speed budget: M and magic should add a small fraction to the tick (design target under 5% with no cast in progress; the lazy field and event pulses are what make that possible); real time is a goal;
  - a coverage test: each archetype in section 9 has a scenario built from pieces, with the missing pieces logged.

THE MISSING-PIECES LOG (kept as a living file). Whatever fiction the pieces cannot build is the useful output. Each entry says whether it is a second change, a pack-only option, or a missing general law. The first entries from this research: relocate, mass bank, rewind, planes, a thinking-spirit substrate rule (Q4), mind-memory reach (Q5), paradox as a social law, story-driven reality.

OPEN PROBLEMS FLAGGED BY THE RESEARCH (to track):
  - The pattern functional I(x), a real-physics measure of "how much information a structure carries", has no unique definition. This design uses the material g and the person's dial instead.
  - Analog versus digital messages: music and dance are analog; the table quantises them. If the sim wants smooth analog spells, it needs continuous leaves with noise.
  - Whether practice is stored per spell or per node (M7).
  - Nuclear-level handling may be too heavy; a cost table per element pair is enough.
  - Numbers marked (u) in this document must be checked before they enter data files: ignition energies, Koski and Toyabe efficiencies, brain energy per spike, persistent-current lifetimes, Casimir rounding, firefly quantum yield 0.41, human brain 15 vs 20 W, maximum fat oxidation, NPP table, mycorrhizal carbon share, wound-healing times, d33, ZT, TENG efficiency, Olsen cycle, binding energies, TI.

SOURCES NAMED (page and volume details are from memory unless marked (v); treat as (u) as a group): Landauer 1961; Szilard 1929; Bennett 1982; Berut et al., Nature 483:187 (2012); Toyabe et al., Nature Physics 6:988 (2010) (v); Koski et al., PNAS 111:13786 (2014) (v); Sagawa and Ueda 2008, 2010; Parrondo, Horowitz and Sagawa, Nature Physics 11:131 (2015); Turing 1952; Pearson, Science 261:189 (1993); Prigogine and Nicolis 1977; England 2013; Bak, Tang and Wiesenfeld 1987; Shockley and Queisser 1961; Curzon and Ahlborn 1975; Zhu, Long and Ort 2008; Kurs et al., Science 317:83 (2007) (v); Semenov 1928 and Frank-Kamenetskii 1939 (constants (v) from Gray, Boddington and Robinson 1979); Shannon 1948, 1951; Noether 1918; Onsager 1931; Pendry, Schurig and Smith, Science 312:1780 (2006); De Santana et al., Nat Commun 10:4000 (2019) (v for 860 V); Gallant et al., Science 344:1522 (2014) (v for at least six lineages); Haddock, Moline and Case 2010; Polderman et al., Nature Genetics 47:702 (2015) (v, 49%); Coupe et al., Science Advances 5(9) (2019) (v); Zheng and Meister 2024; Willett et al., Nature 2023 (v); Metzger et al., Nature 2023 (v); Fitts 1954; Hamming 1950; Huffman 1952; Cowan 2001; Miller 1956; Newell and Rosenbloom 1981; Sadtler et al. 2014; Miner 1945; Arrhenius 1889; Hill 1910; Haber 1924; Seaborg group (Bevalac, 1980); Lavoisier 1789; Boyle 1661. Fan papers used only for structure and feel: Arcane Spectroscopy (Gorilla of Destiny; a packet with stored potential energy spent as work plus waste light, eps0 + epsv + epsL = 1), Theory of Magic and the Gygax field (a metastable field; an "aperture"; energy fitted E = 6077*e^(0.48 L) - 7006 in gelatinous cube units; damage linear in level), Science of the Warlock, Science Spellbook (order-of-magnitude errors: Absolute Zero ignores about 1e7 J of heat to sink; sublimating 1 m^3 of rock needs about 1e10 J), the spell dictionaries (about 280 5e SRD spells). The fiction lane did no web checks (list facts such as 16 allomantic metals, Ars Magica 5 techniques and 10 forms, 9 Mage spheres match the critic's knowledge; slippage and hemalurgy loss numbers are (u)).

END OF DOCUMENT

## Addendum (2026-11-20): candidates from two newer references

From *The Metaphysics of Magic* and the *ToM Issues* whitepaper. The full notes are in `docs/magic-research/metaphysics_and_whitepaper.txt`. These were PROPOSALS. OWNER, 2026-11-21: all nine are ADOPTED AS DESIGN DIRECTION, and each will be decided again when its details are settled at build time:
- **The cast as a chain:** holder, shaper, local environment, primary store. Each stage has a gain, a match and a leak; the unmatched part leaks into neighbouring effects or heat. Attunement is a match.
- **Imprinting:** prepared magic is a pattern written into a metastable "groove". Quality depends on the caster's state, there is no slot count, skill compresses patterns, and grooves fade slowly. This is the physical form of "recorded magic = implicit slots".
- **Three dials for every patterned thing:** store size, pattern bits and persistence, bonds. Gods and patrons draw on the sum over their bonded worshippers, booked from them.
- **Cost by gradient distance:** effects are cheaper when the matching source is near.
- **Circles as coupled lanes** that synchronise through bond strength. Misaligned purpose reflects energy back onto the casters, and an overloaded leader burns. Friendship is a lasting bond, raised by costly acts.
- **Items as designed energy minima:** a failed forging falls into another minimum, and a curse is a minimum with a drain path to the holder. Triggers are deliberate instability.
- **A "divided heart"** (conflicting drives in the learned mind) as a source of casting noise.
- **Range as a coupling threshold** (the whitepaper): range is where the spreading pulse drops below what a receiver can couple.
- **Divergence guards** (the whitepaper's infinite-energy flaw does not apply to our fixed-energy packets, but five places could diverge if coded carelessly): clamp focusing at the diffraction limit, use finite source sizes, cap resonance by input, keep loop gain below 1, and book interference per cell. Each has a stress test.

## Addendum (2026-11-21): the owner's decision on the extension (Q6 options, docs/magic-research/extension.txt)

OPTION 1 PLUS DOORWAYS. Built only after the first magic slices (the gem fireball, then the body-only spark).
- **Conservation holds.** Mass and energy are conserved, so nothing comes from nothing. TRUE CREATION (the mass bank, E = mc^2) is OFF by default and pack-only.
- **Conjure** means TRANSFORMING what exists:
  - reshaping material into another form (earth, metal: a pattern rebuild from local matter);
  - or TEMPORARY CONSTRUCTS made of the field itself, which hold only while paid for and then dissipate, booked like any store.
- **Rewind** is restoration, not time travel, and only for THE PARTS STILL TOGETHER. The rebuilder may use only the matter still bound in the thing. Matter that has left (eaten, rotted away, dispersed) is out of reach: of an apple half eaten and half rotted, only what remains reverts. Replacing lost parts from other matter is conjuring, with its own cost.
- **Teleport** comes in SEVERAL KINDS:
  1. wormhole-style DOORWAYS, an intact passage: a non-local edge with a holding cost and finite speed, through which air and momentum cross;
  2. ROUGH or EXACT COPIES by scan, send and rebuild (rough: the original survives and detail is lost; exact: the read destroys the original).
  The variant and its losses depend on HOW WELL THE SPELL IS CAST: pattern quality sets fidelity and errors.
- **Other planes** are teleports ACROSS DIMENSIONS: the same kinds, through a link to another world with its own field and matter.

## Addendum (2026-11-21): bonds and perceiving others (from the learned-mind scoping, SV2)

By default a mind knows how another person is doing only through VISIBLE SIGNS (the target's expressiveness dial, the observer's acuity dial, noise). The owner: two beings with a MAGICAL CONNECTION may read each other more directly. In this design that is a BOND, a resonant link through the field M that carries the other's state as a signal, with its own bandwidth, noise, range and latency. So it is still a channel under the same physics, and a perfect read is only its limit. It can be blocked, damaged, overheard (scrying reads only fields that produce or interact with magic, Q7) and faked by a skilled caster.

## Addendum (2026-11-21): release at range (from "A Classical Solution to the Ultramagic Catastrophe")

A build-time choice under Q22's "range as a threshold". When a packet spreads below what any receiver can couple, its remaining booked energy must go somewhere: either (a) it decays into diffuse light and heat along the path, or (b) it releases at the threshold point as its intended effect, so a spell pushed past its range "auto-casts" there, as the reference proposes. Both conserve energy. Which one happens should follow from whether the pattern holds together until release (pattern persistence, one of the three dials), not from a rule. Notes: docs/magic-research/metaphysics_and_whitepaper.txt (addendum).

## Addendum (2026-11-21): costs, components and interfaces (owner decisions MC1-MC8)

Research: docs/magic-research/costs_{europe,world,theory,fiction,synthesis}.txt (20 real traditions, 16 fiction and game systems, the theorists, and Rozin and Nemeroff's experiments). Most claims are (k); check the primary works before numbers enter data.

**Findings.**
- The EIGHT COST KINDS hold, and no ninth physical cost was found: energy, waste, information, matter, wear, links, time, being noticed.
- SPLIT two of them:
  - TIME into casting duration and WAITING FOR A WINDOW;
  - MATTER into a REAGENT (physics) and an OFFERING to a being (a price set by a mind, social).
- FREEHAND = low energy, short reach and no kept object. It can be high information if the time is spent.
- STACKED MEDIA are mostly REDUNDANCY FOR RELIABILITY (the same message in several channels, plus error-checking helpers), not power. Add redundant stacking and a target end (the link edge) to the casting chain.
- Purity, taboo, licence, debt, rank, stigma and reputation are SOCIAL, held as beliefs.
- Candidate small laws:
  - I-1 a time-varying source;
  - I-2 idle decay;
  - I-3 the touch hazard of charged stores;
  - I-4 non-heat body load;
  - I-5 the link as a lossy channel;
  - I-6 a name is information;
  - I-7 a ward is a standing barrier;
  - I-8 divination as a cheap noisy information family;
  - I-9 a waste sink at a chosen site (the burned effigy);
  - I-10 innate reflex patterns;
  - I-11 use drives the holder's state;
  - I-12 the holder is vulnerable while working.

**Decisions.**
- MC1 NAMES:
  - every being may have many names (a given name, nicknames, titles);
  - a name has power only where a BOND exists, formed by the being's own identification with it (attention to one's own name), by collective use (worship), or by ritual;
  - each name has its own bond strength;
  - a pack may make names real by bonding.
- MC2 CULTURAL RULES:
  - some have a real physical basis found by accident, others are empty, and people learn which through the knowledge layer;
  - supernatural penalties are a pack option only;
  - SOME PLACES ENFORCE RULES for real, through something physically there (a being acting on its wants, a charged pattern that discharges, a local field condition), never a flag.
- MC3 LASTING CHANGE: only through existing laws in the base (training, overload damage, learned craving, patron bonds); pack-level corruption on top; a warning before anything permanent. USE-DEPENDENT PLASTICITY for every capacity:
  - use raises it with diminishing returns;
  - disuse decays it toward a floor;
  - misuse damages or mis-tunes it (like amblyopia);
  - childhood is more plastic.
- MC4 DEBTS:
  - the patron's memory is the record, and the tally shown is the CHARACTER'S OWN BELIEF;
  - SOME PACTS ARE INTRINSIC ("the world has given observance"): a self-enforcing pattern with a condition and stored energy;
  - an AGREEMENTS TAXONOMY is to be scoped: memory only, witnessed, written, institutional, intrinsic or magical.
- MC5 CYCLES: the field's strength follows the NATURAL PROCESSES THAT PRODUCE IT (sun, growth, tides, storms); there is no calendar rule. The owner's principle: "x affects y if x and y depend on the same thing or y depends on x."
- MC6 SACRIFICE:
  - lawful at physical yield, with no bonus;
  - a gifted body's stored M is released at death;
  - OTHERS' LIFE FORCE can be drawn through a link, and consent and coupling matter;
  - AUSTERITY strengthens magic through existing mechanisms: accumulation, discipline quieting the divided heart, belief lowering anxiety and noise, and arousal or pain raising power at a cost in precision (trainable).
- MC7 SHOWING COSTS:
  - prose by default, numbers on request, a warning before permanent loss;
  - numbers ONLY TO THE PRECISION THE CHARACTER COULD KNOW (introspective acuity, training, instruments);
  - a LitRPG "System" is an in-world source that can be wrong.
- MC8 CULTURES: invented by default, mixing and drawing inspiration from real ones; real traditions named only where public, ancient and well documented; never living sacred practice as mechanics.

## Addendum (2026-11-23): magic scoping MS1-MS5 (from the library's survey of the Gorilla of Destiny channel)

Five candidate rules came from the survey: a per-material magic index, line spectra, aspect costs, mode sets, and wavelength read-out. The owner decides them one by one.

- MS1 UNFAMILIAR EFFECTS, OWNER: (b), PRACTICE RAISES EFFICIENCY, extended into a game of skill and a layered system.
  - THE LAW: a clumsy or unfamiliar pattern converts worse, so more of the drawn energy leaks as waste heat at the site. The work share rises with practice on PHYSICALLY NEARBY effects (a hotter fireball is near a fireball; a frost cone is far). This uses the existing practice law and the work, light and heat split, with no new basis. A pack may add an explicit aspect basis with a distance surcharge (the survey's system) as a pack option.
  - TWO PARTS OF SKILL (owner):
    - an INTRINSIC sense of shaping and control: a per-person dial that changes over time through use, disuse and misuse (MC3 plasticity);
    - EXPLICIT SHAPING: a caster (and the player) can deliberately write the pattern in more detail. More bits buy more specific effects and a better match to the target (impedance matching, resonance, focus geometry), so a well-written spell wastes less. The player's own skill at writing spells then matters, as a game of skill.
  - LAYERS (owner): magic is intrinsically very LOW LEVEL, the field and the grammar of physical leaves (section 3.5). On top of it sit INTERFACES that each model a part of it, and MANY KINDS OF FORMULAE (notations) in which people write intricate spells; each compiles to the grammar. The more complex a spell, the more specific it can be.
  - A CULTURAL PATH (owner's default history; the mechanism is general): early on, powerful minds ("gods") gave people THEMATIC SPELLS STRAIGHT INTO MEMORY, as compiled patterns. They are easy to cast and intuitive to use, but they are complex dials built to the gods' own understanding, held as one opaque unit, and they CANNOT BE CHANGED ON A WHIM. Taking one apart (study, experiment, a scholarly tradition) reveals its dials, the road from gifted spells to low-level magic.
    - Mechanism: any mind with a pattern store can give a compiled pattern to another through a bond (as a patron lends patterns, section 4). Casting it costs the holder few bits, because its design cost was paid by its maker, like a carving. Its efficiency is whatever its maker built in, so it may be poor for the case at hand.
    - The gods' history is world data for the default world. Packs may replace it.
- MS2 MATERIAL AND MAGIC SPEED, OWNER: (b) NOW, a MAGIC INDEX PER MATERIAL (a column in the one material record, 15c F4). Pulses bend at boundaries by Snell's law in the existing ray march, reflect partly at index steps (the impedance-step shield, now the same law) and are trapped in slower cores (waveguides). So lenses, mirrors, waveguides and leaky barriers are shaped material, and a slow-core ley line guides magic like a fibre. LATER EXTENSION: the index rises with the local M density (dense magic bends pulses toward itself: ley lines guide by density, strong bolts self-focus, drained craters bend magic away); non-linear, tested before use.
  - THE OWNER'S NUANCE: how much a material matters DEPENDS ON THE INTERFACE and on why magic happens. Material enters in two places:
    - along the PATH, every pulse reads the index of what it crosses, whatever the interface;
    - in the MEDIUM that holds or carries the pattern. Where the pattern lives in a mind (voice, gesture, thought), the material around matters little beyond the channel (air for voice, the body for gesture). Where the pattern is held in matter (an engraving, a written formula, a tuned solution), the substrate's properties decide what can be written and how well: coupling g, index, Q, the finest feature it holds, and wear.
  - So each INTERFACE is a channel record that NAMES THE MATERIAL PROPERTIES IT READS (data, editable, per pack); none is hard-coded.
- MS3 SIGNATURES, OWNER: (b), a SIGNATURE on a cast's tell, GENERALISED BEYOND LIGHT.
  - The tell and the waste are not always light. Each transducer (and interface) splits its energy across ALL the sim's physical channels: useful work, heat, light, sound and pressure, motion, waves in the field M itself, and any field a pack adds. The split is per-transducer data, replacing the fixed three-way work, light and heat split.
  - Whatever channel carries the tell carries the SIGNATURE: lines or tones set by the spell's structure, shifted by the pulse's speed (Doppler). It is read by the matching sense (eyes, ears, an M-sense, instruments) through resolution dials, as a learned, fallible skill (knowledge design). A gifted spell carries its maker's signature.
  - THE OWNER'S STANDARD FOR WHAT A SPELL IS: never "I cast fireball, the mana bar went down, the enemy went boom". A spell is a mechanism acting on the world's existing fields, for example "a formula sends a signal through the field that isolates the flammable gas x metres away in direction y, and makes a spark z seconds later". The grammar (3.5) already composes this (Matter-move of a species set, Delay, a small heat trigger), and the energy then comes from the chemistry. A named "fireball" is at most a compiled pattern (MS1) whose parts are such operations.
  - OPEN, TO SCOPE (owner): WHAT THE FIELD IS, HOW AND WHY ANYONE CAN INTERFACE WITH IT, and WHAT CHANGES IT MAKES TO THE EXISTING PHYSICAL FIELDS. Sections 2-3 give a working answer (a conserved field transduced by patterned structures); the owner wants it scoped from first principles before magic is built.
- MS4 RESONANCE OF OBJECTS, OWNER: (b), GENERALISED: EVERY object's resonance comes from its REAL BODY, a set of modes (bending, twisting, lengthwise), each with its own frequency and Q, computed from its shape and material; never one f0 per object, and never a harmonic n f1 series except where the physics is harmonic (strings, lengthwise modes). Rods and planks come first (wave 15c computes their modes); compact shapes later (no modal model in the library yet). A passive enchantment is energy stored in a mode, leaking at that mode's rate; damage shifts modes and detunes it.
  - THE OWNER'S QUESTION: why would anyone need a staff? NOBODY NEEDS ONE, and no object is a focus by type. An object helps only where physics makes it help:
    - as an ANTENNA (aperture gain G = 4 pi A / lambda^2; a body-length object near half a typical wavelength is a half-wave dipole, so staff length follows the pack's c_M and typical frequencies);
    - as a RESONATOR (stored energy builds to about Q times the drive; a narrower band);
    - as a STORE (charged material);
    - as a PATTERN HOLDER (an engraving casts at trigger speed, its cost paid at carving time);
    - as a MEMORY AID or through BELIEF (lower anxiety and noise, MC6), a small real effect a culture may misattribute.
  - Where none applies (for example kilometre wavelengths), a staff is a walking stick, and a culture that uses one carries a habit, possibly with a wrong theory of why it works.
