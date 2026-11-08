# OpenWorld as a design reference

Date: 2026-10-29. Source: the public wiki at play-openworld.com/wiki (read pages: technology, power, pipes and conveyors, farming, crops, husbandry, mining, crafting, smelting, alloys, survival, cold and heat, cooking, building, materials). OpenWorld is a closed-source voxel sandbox (alpha, by Tom Pascard). We take design ideas only; no code or data. Numbers below are theirs, paraphrased.

How to read this: for each area, what they do, what is worth taking, and how we change it so the law, not a recipe, produces the result. Law ids refer to docs/composite-events.md.

## 1. The principles worth taking

1. **Real units everywhere.** Watts, kWh, kcal, metres per second, °C, litres (their voxel is 1/64 m³ = 15.6 L). We already do this; their wiki shows players can read real units.
2. **Networks are graphs, not voxels.** Cables, pipes, shafts and conveyors are links with a capacity, a length and a loss. That is cheap and exact. For us these are records (CONDUIT, STRAND, MECHANISM) on L16, with every loss booked as heat.
3. **Conservation stated in plain words.** "Power is conserved; losses come from cables, transformers and gears." "Split, not duplicated." "A machine draws only while it works." This is our ledger rule in player language.
4. **Machines from geometry.** Two gears mesh only when their axes are parallel, in one plane, and their centres are the sum of the radii apart; speed changes by the radius ratio. A furnace is a ring of eight rock cells around an empty middle, not an item. We go one step further: no "furnace" object at all. Any enclosure that holds heat is a furnace, and its temperature comes from its heat balance (L1, Bn).
5. **Fuel and reducer are different jobs.** Their furnace has a fuel slot (heat) and a reducer slot (coal or charcoal that takes the oxygen from the ore). That is the real chemistry of a bloomery. For us it falls out of Bn plus L4 rows: carbon burns for heat, and CO reduces the oxide.
6. **A room is a heat envelope.** The game checks the room you stand in. Walls and closed doors hold heat; any hole or open door leaks it. A room warms in minutes and keeps its heat. For us this is the air-region labeller plus conduction and radiation, with no room check at all.
7. **Weather and place set the temperature.** Climate, altitude, hour and weather; caves hold the ground's steady temperature; wind makes cold feel colder below about 10 °C; damp makes heat feel hotter above about 27 °C. This maps to Fx (outside inputs) and the body-heat model.
8. **Production follows forage density.** A beehive makes honeycomb faster the more flowers lie within 8 m. That is the right shape for L9 populations: yield from what the area actually supplies.
9. **The world keeps working when nobody watches.** Furnaces, crops and pumps carry on while far away or after a restart. That is our L11 slow time and offscreen summary.
10. **Two building scales.** Fine raw matter (their 25 cm) and whole 1 m shapes, plus a blueprint desk: draw a piece cell by cell, then stamp copies. For us: a prefab layer for building scenes fast (it pairs with the MarkovJunior idea), stamped into 5 cm cells.

## 2. Area by area: theirs, and how we make it real

### Power and machines (L16, L18, L1)
- **Theirs:** generator up to 5 kW, 8 kWh per voxel of coal, burns only what the cables take; solar 400 W in high sun, 15% under a roof, nothing at night; battery 10 kWh at up to 5 kW; small cable 400 V / 10 kW, large 10 kV / 250 kW; a cable trips for 5 s when overloaded; transformer keeps 97%; motor draws 2.2 kW for 2 kW on its shaft; shafts lose 1% a metre; each gear mesh loses 1%. Supply order: sun, then batteries, then fuel.
- **Real:**
  - every loss becomes heat in the part that loses it (cable I²R, gear friction, motor heat);
  - coal's energy is about 24-33 MJ/kg, and a generator turns only 25-40% of that into work;
  - solar is about 15-22% of about 1 kW/m² of sun;
  - cable loss follows its resistance and current;
  - a trip is a fuse or breaker that heats and opens.

  The graph form stays.

### Pipes, pumps and conveyors (L2a, L16)
- **Theirs:** small pipe 10 cm bore, 4 voxels/s; large 50 cm, 64 voxels/s; gravity flow only if the far end is lower and the run is 20 m or less, otherwise a booster pump about every 80 m; a pump works best at 30 rpm and loses a third of its efficiency for each doubling or halving of speed; conveyors 1 or 4 m/s; one fluid per line.
- **Real:**
  - flow from head and pipe friction (Darcy-Weisbach), so gravity feeds work over any length while the far end is lower;
  - a real pump curve (head against flow);
  - viscosity per fluid instead of their "spread distance" (water 1.75 m, oil 1.25 m, lava 0.75 m);
  - pumping energy from ρ·g·h·V plus losses.

  Keep the CONDUIT record and the one-fluid-per-line rule (mixed lines need two-phase flow).

### Smelting and kiln work (Bn, L4, L7, L14)
- **Theirs:** 16 voxels of ore + 4 coal make one ingot in about 10-20 s; titanium needs 8 coal; steel is iron + 2 coal; bauxite needs electric power; kiln recipes: logs to charcoal, sand to glass, clay to terracotta, clay + sand to brick, clay + limestone to cement, sand + coal to silicon.
- **Real:**
  - these are real processes, so keep the list as test scenes, not recipes;
  - make them happen by temperature, time and atmosphere;
  - charcoal is wood heated with little air (pyrolysis, about 300-500 °C, 20-30% yield, a day or more);
  - terracotta fires at about 900-1000 °C for hours;
  - soda-lime glass melts near 1000-1100 °C and pure silica only above 1700 °C;
  - lime calcines at about 900 °C;
  - cement clinker forms at about 1450 °C;
  - iron in a bloomery needs about 1200 °C and CO, and leaves slag;
  - steel from carburizing takes hours at heat;
  - aluminium from bauxite needs electrolysis (Hall-Héroult), which is why it needs power, and that is right.

  Yields come from mass balance, not fixed counts.

### Alloys and forging (L6, L7)
- **Theirs:** pressed "alloys" add multipliers to an item (life ×3, damage ×1.4, dig ×1.75); one per item, three on a special metal.
- **Real:** skip the multipliers. Real alloys come from melting metals together in proportion (bronze about 88% copper and 12% tin; steel by carbon content), and properties come from composition and heat treatment: hardness from carbon and quench, toughness from temper. Keep only their idea that a modified item keeps its identity (our composite object with records).

### Farming (L8 plants, L3 soil water, Fx, L11)
- **Theirs:**
  - tilling turns a square metre into farmland, one seed sows one voxel, and crops need farmland under them;
  - four growth stages of 2-4 minutes each;
  - growth is fastest at 12-28 °C, slows when cold or dry, and stops below 2 °C;
  - bone meal moves a square metre one stage on;
  - a ripe harvest gives grain, seed and straw, and an unripe one gives seed back;
  - saplings need still water within 2 m;
  - wild crops grow by biome.
- **Real:**
  - **Growth:** growing degree days above a base temperature per crop; wheat takes about 1,500-2,000 degree-days, roughly 100-120 days.
  - **Water:** a soil water balance (FAO-56) instead of "near water".
  - **Light:** it drives growth.
  - **Nutrients:** N, P and K. Bone meal really is a phosphorus fertilizer, so it speeds a crop only when phosphorus is short, never "one stage on".
  - **Yields:** about 0.3-0.8 kg of wheat per m², with straw and seed as real by-products.
  - **Time:** months, run by the slow-time scheduler.

  Their temperature band (fast at 12-28 °C, stop near 2 °C) is close to real cereal behaviour and can seed our parameters.

### Animals (L8, L9, L22)
- **Theirs:**
  - a baby grows up in 20 minutes;
  - in love for 30 s, with a partner within 8 m;
  - parents rest 5 minutes after a birth;
  - milk 16 voxels (about 250 L!) every 60 s;
  - wool 1-3 per shearing, regrowing in 5 minutes;
  - an egg every 5-10 minutes;
  - taming by feeding with a chance per feed.
- **Real:**
  - **Products come from the animal's energy budget:** milk from feed through metabolism, about 10-30 L a day for a cow; wool about 2-5 kg a year from one shearing; about one egg a day.
  - **Breeding has real seasons and pregnancy times:** a cow is about 9 months.
  - **Taming is learned trust:** the learned-mind valence and partner models, not a dice roll.

  Keep their idea that the right food heals and that animals have breeding foods, as drives.

### Mining (L5, L10, L15, geology in Fx and the world builder)
- **Theirs:**
  - tool tiers (wood, stone, iron, titanium), each lasting a number of voxels mined (an iron pick 288 voxels of stone);
  - ores have depth bands, rarity and favourite places (veins, geodes, giant lodes, oil).
- **Real:**
  - wear by Archard's law and hardness against hardness, so copper barely scratches granite;
  - picks that dull (edge radius) and can be resharpened;
  - ore placed by how it forms: veins along faults, placer gold in rivers, bog iron in wetlands, coal in sedimentary layers, oil in traps under caprock;
  - fire-setting and roof support, from the composite-events study.

### Survival and the body (L8, L17, the body-heat model)
- **Theirs:**
  - 20 kcal per food point and 2,000 kcal per full stomach, plus a fat reserve;
  - sprinting costs about 2,000 kcal per km;
  - cooking doubles food points and makes food "keep you full" about 10 times longer;
  - hunger drains up to three times faster below 12 °C;
  - walking 4.25 m/s, sprinting 5.6 m/s;
  - breath held 20 s;
  - falls safe to 3.5 m, then 6 HP per m;
  - clothing adds +12 °C (sweater) or +20 °C (fur coat).
- **Real:**
  - **Energy:** a person burns about 80 W at rest and about 1 kcal per kg per km walking; cold raises the burn by shivering, and that part they have right.
  - **Cooking:** it raises digestibility by tens of percent, not ×2.
  - **Movement:** walking is about 1.4 m/s and a jog about 3 m/s.
  - **Breath:** a breath hold lasts 30-90 s.
  - **Falls:** fall injury by body region and impact speed (the realism map).
  - **Clothing:** clothing as SHEET records with real insulation (clo), which loses most of it when wet.

  Their food kcal values are close to real portions and can seed our food records.

### Liquids (L2a)
- **Theirs:** a table per liquid of spread distance, burn time and pumping energy; buckets hold 64 voxels.
- **Real:** density, viscosity, flash point and heat of combustion per liquid; the "spread" comes out of viscosity and the surface. Our fluid laws already work this way; their table is a checklist of liquids players expect (water, oil, lava, wet concrete, milk, kerosene, petrol, diesel).

### Building (L5, L13, prefabs)
- **Theirs:** two scales, shapes (pillar, slab, stair, wall, fence, pane), a blueprint desk, doors and trapdoors with keys and locks; no support or collapse rules are described.
- **Real:** keep the shapes and blueprints as an authoring tool; support and collapse come from our mechanics law (beam check and supports); locks as MECHANISM records (L16) that can be picked.

## 3. What not to take
- **Compressed time:** a 25-minute day, 6-12-minute crops, 20-minute calves. It breaks every rate law (cooling, drying, rot). Instead: real time, skipped ahead by closed forms when nobody watches (L11).
- **Multipliers and tiers as stats:** alloy multipliers, food-point doubling, fixed tool lives. Properties must come from state.
- **Chance rolls for taming:** use learned trust instead.
- **Fixed recipes:** their recipe list is valuable as a list of things players expect to work, so turn each into a proof scene for the laws.

## 4. Proof scenes taken from their lists
Each must work with no recipe code:
1. Charcoal from logs in a closed, hot, low-air pile.
2. A brick or terracotta firing in an enclosure that reaches about 950 °C.
3. Lime from limestone at about 900 °C, then mortar that sets with CO₂ over weeks.
4. Iron from ore with charcoal as fuel and reducer, slag left behind.
5. A room that warms with a fireplace and loses its heat through an open door.
6. A wheat plot that ripens by degree days and water, faster with phosphorus where phosphorus is short.
7. A beehive whose output follows the flowers within reach.
8. Two gears that mesh only by geometry, with the speed ratio from the radii and the friction loss booked as heat.
9. Gravity-fed water along a long pipe that runs downhill, with no booster needed.
