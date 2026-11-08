# OpenWorld's engine, as a design reference

Date: 2026-10-29. Companion to docs/openworld-reference.md, which covers their GAMEPLAY rules from the public wiki. This file covers their ENGINE: how the world is stored, ticked, saved, budgeted and debugged.

Sources, all openly downloadable (the a1.1.1 client and dedicated-server zips, sha256-checked against their release manifest):
- their data files: knowledge base, map config, texture pack list, server README;
- the server's browser dashboard and seed inspector;
- the log strings and source-file paths inside their programs;
- a Ghidra pass over the server program, read only to confirm structure (which quantities a system combines, and its thresholds). The client's built-in server and the dedicated server are the same program, byte for byte.

Rules we keep:
- ideas only, written in our words;
- no code or data of theirs goes into Mark_1;
- none of their files leaves this machine.

The downloads live outside the repo, in ~/re/openworld/.

## 0. Their numbers, for scale

| | OpenWorld | Mark_1 |
|---|---|---|
| voxel | 25 cm (64 per m³, 15.6 L) | 5 cm |
| world | cube-sphere planets, 6 faces, about 1,447 km per face | authored scenes |
| column | 16 m square = 64 voxels on a side | — |
| chunk edge | at most 64 voxels: one row = one 64-bit word | — |
| world tick | 20 Hz, 50 ms budget | integer tick |
| machine sim | separate fixed step, 20 Hz (`--ovm-hz` 1-60) | — |
| storage | palette + zstd, region files, sidecar ledgers | snapshots, `.npz` frames |
| language | C++ (SDL3, Vulkan, protobuf, zstd, libsodium) | Python, numpy, numba |

## 1. Decisions worth taking

### 1.1 Simulation has levels of detail, not just rendering
Their only dense, always-running simulation is the machine network (the OVM). Its phases run every step: topology, signals, rotation, electric, fluids, items, radio, events.

Everything else is cheaper than per-voxel:

| Their system | How it runs |
|---|---|
| Heat | lumped per room |
| Weather | one coarse planet field, driven by the seed and the clock |
| Growth and concrete setting | ledgers with deadlines (`.owgrow`, `.owcure`) |
| Light | baked in the background, with lazy stale marks |
| Falling ground | whole columns, as events |

Fluids are conserved volumes inside pipes and vats. There is no water cellular automaton anywhere in the server.

**For Mark_1:** our premise is the opposite, real per-voxel laws, and that stays. What transfers is the LAYERING.
- Voxel laws run where someone watches.
- Lumped forms and ledgers run elsewhere.
- Hand-over happens when attention arrives or leaves.

That is exactly L11 (slow time, docs/composite-events.md), which today has only an integer tick. Their ledgers are a working existence proof of L11: one record per slow process, with a due time, and no per-tick scan.

### 1.2 A room as one heat node
Their thermal model finds the enclosed air around a player with a flood-fill probe. That gives a volume, the open area, and a conductance to outside (`ua`). Heaters add watts into that one node. They log probe cost (about cells and µs) and update only occupied rooms.

**Confirmed in the program (Ghidra and disassembly, read for structure only).** The room's temperature is SOLVED AS A STEADY STATE each time it is probed; it is not stepped in time. In textbook terms:

T_room = (Σ Gᵢ·Tᵢ + Q) / Σ Gᵢ

That is, a conductance-weighted mean of what the room touches, plus heat divided by conductance.
- **Openings** count as a fixed conductance per m² to outside air.
- **Walls** couple to a blend of outside and ground temperature. The blend is weighted by an "indoor" factor from 0 to 1, so deep or buried rooms follow the ground.
- **Heat sources** are summed in watts. Each player counts as a flat 100 W.
- **Clamps:** the result is kept between −60 and 90 °C.
- **Humidity** comes from the Magnus formula.
- **Where probes run:** the probe is keyed to a player connection, so rooms are found only around players and never maintained world-wide.

**For Mark_1:** we already label air regions (air.py islands), so we have the room. The steady-state solve above is exactly the closed form a far room tends to, and we can do one better because we know the room's real heat capacity C:
- relax toward that equilibrium with time constant C / ΣG (an exact exponential, not Euler);
- do this as the FAR tier of L1 heat while nobody is inside;
- hand back to the voxel laws when someone enters.

Measure it against the voxel law before trusting it (rule: an approximation must keep the facts a scene is about). The fireplace-and-open-door proof scene is the test.

### 1.3 Creatures come into being when the land is first seen, and sleep in tiers
- **Lazy population:** "the land is peopled as it is first seen". The wildlife files are kept per cell, and a broken file just re-peoples its cell.
- **Tiers:** active, resting and dormant. Creatures are put away and brought back. There is a hard cap, and its refusals are logged.
- **Spawn niches:** a creature appears only where its condition holds, such as lava within 4 m, dark, still deep water, or crops at night. There is a cap per 16 m area.
- **Budgeted pathing:** searches are counted as OK, partial, failed, stuck, backed off or given up.
- **Voxel cache:** AI reads voxels through its own cache, never the chunk store directly.
- **Despawn distance (from the program):** the despawn path compares the distance to the nearest player against 128, 144 and 160 m. That is a hysteresis band, so a creature at the edge does not flicker in and out. Creatures therefore live only within about 8-10 columns of a player. Timers count in 1/20 s ticks.

**For Mark_1:** this is the mind-cost answer the latency budget needs.
- An ACTIVE mind gets model calls.
- A RESTING mind runs on static drives only (zero model calls).
- A DORMANT mind is a closed-form summary (an L11 record).

Promotion follows perception salience, the same lens as the two-channel halt in the world-tick design. Their niche predicates are also the right shape for our encounter seeding: spawn only from world facts.

### 1.4 Every expensive operation carries a budget and a way to give up
Their budgets:
- edits carry a cost against a per-tick budget;
- path search backs off and gives up;
- streaming has bytes per second per player;
- the client caps meshes and uploaded bytes per frame;
- light convergence has a node cap, and when it is hit the client uses PARTIAL light and logs it, instead of stalling.

**For Mark_1:** give each law and each mind a per-tick budget. When the budget runs out, degrade in a way that is visible and logged, never silently. Pair this with 1.7 so the cost is seen.

### 1.5 A world = seed + edits, and regenerable is kept apart from precious
Generation is a pure function of the seed. The server stores only changed columns, and its dashboard counts "generated (known)", "modified saved", "pending write" and "dirty" separately.

Two directories separate the two kinds of data:
- `--cache-dir` holds everything that can be regenerated;
- `--data-dir` holds what can't, and `world.cfg` there pins the seed forever.

**For Mark_1:** item 25 (a world is a value) is the base. Once scenes are generated rather than hand-authored (MarkovJunior idea), store seed + edits so saves and golden files stay small and diffs show only what play changed.

### 1.6 Saving never stops the world
- **Cut:** a save is a consistent snapshot boundary.
- **Off-tick writes:** a thread pool encodes modified chunks and writes them. The tick pays only for the cut, and both costs are logged.
- **Failed writes:** a failed write stays dirty and retries at the next save.
- **Corruption:** a corrupt region entry truncates that region at the last good entry and the rest still loads.
- **Migrations:** each one-time repair pass leaves a stamp file so it never runs twice.
- **Materials are retired, never renumbered:** each has a stable `packKey`, and a retired one names its successor.
- **Machine state is derived:** if `ovm.owvm` is missing, it is rebuilt from the voxels, so the voxel world is the single source of truth.

**For Mark_1:** take retire-never-renumber now. Our golden files and snapshots store material ids, so a renumbered table silently changes old worlds. Derived state (labels, islands, caches) should always be rebuildable from voxels, which it mostly already is.

### 1.7 Observability is the developer's main tool
On their server dashboard:
- tick time as mean, p99 and max against the 50 ms budget;
- milliseconds per phase (net, chunks, players, edits, sim, snapshot, projectiles, entities, chests, send);
- per-system µs counters (creatures, rooms, paths);
- a "chunk journey" funnel: generated, saved, pending, served, displayed;
- machine cost ranked by µs of CPU per real second per 16 m column, each with a "show on map" link.

Polling is gated on whether the page is visible, and `since=stamp` sends only what changed.

Their seed inspector shows the CAUSES behind the terrain, not just the result: seven climate fields, slices, features and spawn spots.

They have no creature-AI or thermal inspector. Their debugging effort went to terrain, caches and time.

**For Mark_1:**
- Per-law ms per tick, as mean, p99 and max against a stated budget, printed by stepcheck.
- A cost ranking per mind and per region.
- A "why" view over the V5 cause log, which is the inspector OpenWorld never built.

### 1.8 Decisions have ids, and the code cites them
Their error messages cite entries in `docs/DECISIONS.md`: V-04 vertex format, W-05 world grid, L-08 light curve, S-02 tick, S-04 persistence, N-06 trust-on-first-use. A `docs/JOURNAL.md` holds milestones such as M6c.

**For Mark_1:** we already have the ids (D1-D12, the rulings). The missing habit is citing the ruling id inside the assert or log line that enforces it, so a failure points straight at the reasoning.

### 1.9 One vocabulary, and content as data
- One interface vocabulary everywhere: signals 0-255 with "on" from 128, power in kW, fluids in voxels, one fluid per line.
- **Materials:** a table with flags: solid, liquid, plant, render, and tile size in metres.
- **Texture pack:** a list of tile, metres covered, animation kind and family. There is no atlas on disk; one is built at load.
- **Difficulty:** one table of multipliers.
- **Encyclopedia entries:** one schema. Each answers how, what for, and WHY it exists. The why line is also their balance note.
- **Progress steps:** predicates observed from play, never scripted. Examples are `tech:steel`, `act:...`, `res:...` and `mat:bark*`.

**For Mark_1:**
- Progress predicates over the typed event log fit the "LitRPG systems as packs" direction (cultivation.yaml). A quest step is "an event matching this pattern was seen".
- The how/what for/why schema is a good shape for our reaction-pack docs.

### 1.10 Hazard-matched gating instead of levels
Each new world is dangerous in its own way: heat, cold, acid or pressure. The material that answers it comes from the world before it.

**For Mark_1 scenarios:** dungeons gated by a physical hazard that a found or crafted thing answers, such as a smoke-filled hall or a flooded stair. Gates become physics, not keys.

## 2. Rendering
The full rendering notes (streaming, meshing, vertex format, light, far terrain, GPU memory, diagnostics) and how their code is written are in docs/openworld-rendering-and-craft.md.

One point belongs here: at our 5 cm, a 64-voxel chunk is only 3.2 m. Any view would need its own coarser display grid (for example 20-40 cm, sampled from the 5 cm sim), not the sim lattice.

## 3. What not to follow
- **No per-voxel fluids or heat.** Their design trades physics for scale; ours is the physics. Take the layering (1.1), not the abstraction everywhere.
- **Per-species brain code** (BrainCommon, BrainExotic2, boss phase machines). It conflicts with our no-special-case-subsystems rule: creature behaviour comes from drives and laws, not a file per species.
- **The gameplay shortcuts** already listed in openworld-reference.md §3: compressed time, stat multipliers, taming dice, fixed recipes.

## 4. Ranked next steps for Mark_1
The to-do form of this list, with every transferable item from all three OpenWorld notes and the wave each one lands with, is in docs/voxel-roadmap.md, "Stolen designs (OpenWorld)".

1. **Per-law timing:** mean, p99 and max ms per tick against a budget, printed by stepcheck. Cheap, and every later speedup needs it.
2. **Retire, never renumber, material ids:** a stable key per material, and a successor when one is retired. Protects goldens and snapshots.
3. **Mind activity tiers:** active, resting, dormant, with promotion by perception salience. Ties to the latency budget and the world-tick halt.
4. **The simulation LOD plan:** voxel laws near, lumped rooms and ledgers far, closed forms offscreen. This is L11 made concrete. Measure the lumped room against L1 on the fireplace-and-open-door scene (openworld-reference.md proof scene 5).
5. **Cite ruling ids in asserts and log lines.**
