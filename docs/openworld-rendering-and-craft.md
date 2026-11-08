# OpenWorld: how it renders a big voxel world, how its code is written, and what we may copy

Date: 2026-10-29. Third OpenWorld note, alongside:
- docs/openworld-reference.md: gameplay rules, from their public wiki;
- docs/openworld-engine-notes.md: simulation, storage, saving and AI tiers.

Sources:
- their a1.1.1 client and server downloads (sha256-checked);
- the log strings, source paths and assertion text inside the programs;
- the debug names left in their compiled shaders;
- their data files, dashboard and inspector;
- one targeted machine-code read of the room-heat model.

Their builds live outside the repo in ~/re/openworld/.

## 0. What "copy" means here

| We may take | We do not take |
|---|---|
| Ideas, techniques, architecture, the shape of a format, rules of thumb, and plain numbers as starting points | Their code, decompiled or not |
| | Their assets: textures, sounds, music, models |
| | Their text: wiki prose, item descriptions |
| | Their data files verbatim |

Every rendering technique below also has a PUBLIC, documented version (named in §1.12). We learn the method from those and keep OpenWorld only as proof that the combination works on consumer hardware.

## 1. Rendering a large voxel world

### 1.1 The pipeline at a glance
The server sends a compressed column blob for each FullChunk it is asked for. On the client, each blob then goes through these steps:
1. a distance-ordered streaming queue decides what to load and unload;
2. a worker job meshes the chunk (binary greedy sweep);
3. a worker job bakes light into a coarse grid;
4. the quads are packed into "lots" (GPU batches) per face of the cube-sphere;
5. the render thread uploads under a per-frame byte budget and draws.

Far away, a separate system draws heightfield tiles (`TerrainLodTile`) with their own shader. A mask hides those tiles wherever real voxels are already shown.

Everything runs under per-frame budgets. When a budget runs out, the client shows a partial result and logs it rather than stalling.

### 1.2 Streaming: what to load, when, in what order
- **Unit:** the FullChunk (16 m column, 64 voxels on a side), addressed as (cube face, i, j).
- **Radius:** the load radius R is in FullChunks, with a target count and a higher threshold. This hysteresis keeps a chunk at the edge from loading and unloading every frame.
- **Order:** nearest first. Eviction is farthest first. Nothing is weighted by view direction.
- **Limits:** a maximum queue length, a cap on requests in flight, a per-frame mesh budget, a per-frame GPU-byte budget, and a cap on simultaneous jobs.
- **Adaptive workers:** the job cap retunes at runtime from how many ready columns are waiting.
- **Read-ahead and cancellation:** requests go out ahead of need. Out-of-range requests are cancelled. Columns beyond a distance are forgotten in bulk.
- **Teleports:** they preload the landing area first. A teleport that arrives before preloading is refused and logged. If the player's position is invalid, streaming pauses for that frame.
- **Edits:**
  - an edit re-meshes the chunk and its neighbours;
  - re-meshes drain at a fixed rate per frame;
  - the client applies its own breaks at once (prediction) and puts the voxels back if the server never confirms;
  - edits that arrive before their chunk are held until the chunk lands.

### 1.3 Meshing
- **Binary greedy meshing:**
  - each row of a chunk is one 64-bit word, so the chunk edge is at most 64;
  - visible faces come from bit operations (a solid cell next to an empty one);
  - same-material faces merge into large quads.
- **Light cells and chunk edge:** the chunk edge must be a whole multiple of the light-cell edge, so a merged quad never straddles two light cells. This is enforced by an assertion that cites decision W-05.
- **Sweep statistics:** levels, swept, skipped, reused, emitted. Unchanged parts are reused, not re-swept.
- **Threads:** meshing and light run on worker threads; draw and upload run on a dedicated render thread with raised priority.

### 1.4 The vertex format, and the shader building quads
- **Packed quad:** a few small fields, not a list of corner positions. Grid coordinates take 7 bits (0-127, enough for a 64-edge chunk plus its far edge), the material id 12 bits (4,096 materials), and the light value and quad direction have fields of their own.
- **Corners in the shader:** the vertex shader has no position input. It rebuilds each corner from the vertex index and the packed fields (vertex pulling).
- **Shared indices:** one static index buffer holds the fixed quad pattern (0,1,2, 2,1,3, …) and every mesh uses it. A validator checks the pattern and rebuilds a broken buffer, with a log line.
- **Crack hiding:** quads are widened by a hair (a push constant) to hide T-junction cracks between merged quads.
- **Plant sway:** the vertex stage moves plants with wind amplitude, a wave origin and time.

### 1.5 How draws are organised
- **Lots:** a lot is a batch of chunks with its own origin and rotation, sent as push constants. The rotation turns each cube-sphere face into one common frame, and a check refuses to draw a lot whose axes are not orthonormal and right-handed.
- **Face-direction ranges:** quads inside a lot are sorted by the direction they face. A direction that cannot face the camera is skipped as a whole range, so backface culling costs nothing per triangle.
- **Opaque before transparent:** this order is counted every frame, and a violations counter makes ordering bugs visible.
- **Cave culling:** a flood-fill over chunk sections hides caves that cannot be seen. An environment variable switches it off for debugging.
- **No newer GPU tricks:** there is no multi-draw-indirect and no bindless. It is plain draw calls with push constants, so it runs on modest GPUs.

### 1.6 Lighting
- **Two baked channels per light cell:** sky light and block light. Light spreads by a breadth-first flood fill.
- **Capped flood fill:** the fill has a node budget per job. If the cap is hit, the partial light is used and logged, and the light is completed later.
- **Incremental edits:** an edit re-lights locally. Light near chunk borders is re-cooked when a neighbour chunk arrives. A mesh that depends on light can wait until the light settles.
- **Day and night:** the shader combines sky light, block light and a sun term, and scales the sky term by a day multiplier. Dusk therefore needs no re-bake.
- **Light curve:** the mapping from light level to brightness is a user setting, as is a minimum brightness (decisions L-08, U-04).
- **Lamps:** merged into lots. A small, capped set of dynamic lights is picked per frame from the candidates.

### 1.7 Far terrain
- **Separate pipeline:** the server sends heightfield tiles at several levels, in concentric rings (clipmap-like), out to a distance in km.
- **Tile content:** each cell is a column with a small material palette, plus a HOLE mask for caves and overhangs. Tiles use the same texture atlas as near terrain.
- **Hiding the overlap:** a clip/hidden mask switches off far terrain wherever the near voxel mesh is drawn.
- **Paged memory:** far terrain lives in fixed-size pages of GPU memory, and a "far terrain memory full" message is logged.
- **Quality:** a far-detail setting gives finer rings, block textures and smooth level changes. Its "auto" value picks a tier from the GPU model, VRAM and RAM. The far distance can be 0 (off).

### 1.8 GPU memory and uploads
- **Their own layer over Vulkan:**
  - big device-memory blocks, sub-allocated, with blocks, reserved and used reported in diagnostics;
  - a clear split between static buffers (filled once through a staging copy) and dynamic ones (mapped).
- **Uploads:** a dedicated transfer queue (T-05), upload batches with a per-frame byte budget, and a warning on slow batches. A frame that waited on an upload is named as such.
- **Frames in flight:** 1 to 3, guarded by fences. A watchdog fires if a GPU frame is not done after N seconds.
- **Textures:** one atlas built at load from per-material 32 px tiles, plus a material table, uploaded once. Normal, roughness and emission come as optional sister tiles. Animated liquids get extra frames.

### 1.9 Sky, weather, particles, decoration, creatures
- **Sky:** one full-screen ray shader draws:
  - procedural planets and moons;
  - a sun halo;
  - air colour;
  - shadows one body casts on another.

  Stars are point sprites from a catalogue, with airmass extinction, and the Milky Way is a texture.
- **Clouds:** a raymarched cloud layer driven by a weather map on the cube-sphere.
- **Rain, snow and sand:** particles on a grid that follows the camera. A roof-height map stops drops under cover.
- **Decoration (furniture, pipes):** built from pieces, rebuilt in place on change, and drawn only near the camera.
- **Creatures:** bone models in an arena with a bone budget per model. They are lit by sampling the baked light at their cell; nothing is lit per pixel.

### 1.10 Settings
- **One "graphics quality" tier** sets every visual extra together: distances, particles, cloth, flicker and decoration.
- **Separate knobs:** far detail (auto from GPU class), weather quality ("minimum = fog and sky only"), render distance in FullChunks, far distance, field of view, interface scale, wind, minimum brightness, light curve, maximum fps, vsync and render scale.
- **Effective distance:** the log prints the render distance requested and the one actually used.

### 1.11 Diagnostics for rendering
- **The "GEL" hitch line:** printed only when a frame is slow. It splits that one frame across:
  - the GPU stages: fence, acquire, encode, submit, present;
  - the streaming state;
  - the allocator;
  - worker-pool starvation;
  - decoration.

  Any spike points at its cause with no profiler needed.
- **Running counters:** average and max frame time, the 1% low fps, the worst hitch, draw calls, triangles, columns loaded, queued and in flight, and LOD tiles and draws.
- **Other tools:**
  - a CSV trace of all of the above;
  - environment variables that switch off one stage each (skip meshing, partial meshing, cave culling off);
  - scripted camera flights at a set speed, for repeatable benchmarks.

### 1.12 The public versions of each technique
| Their piece | Public technique to learn from |
|---|---|
| 64-bit row mesher | Binary greedy meshing (0fps "Meshing in a Minecraft game"; the open-source binary-greedy-meshing write-ups) |
| packed quad + corners from vertex index | Vertex pulling / "programmable vertex fetch" |
| sky + block light by flood fill | Minecraft-style BFS light propagation |
| hidden caves | Cave culling by section-connectivity flood fill (the Tommaso Checchi algorithm) |
| far rings with masks | Geometry clipmaps / CDLOD |
| planet faces | Cube-sphere mapping |
| big-block GPU sub-allocation | The Vulkan Memory Allocator pattern |

## 2. How they write the code

### 2.1 Layers
They have four layers:
1. **foundation:** crypto, i18n;
2. **engine cores:** one library per concern, named `…Core`, such as RhiCore, MeshCore, VoxelCore, AtlasCore, AudioCore, MediaCore, PlatformCore, PersistCore and NetMTU, plus WorldGen and ServerGame;
3. **game-domain libraries:** CoreApi, DataApi, Knowledge, Crafting, Deco, Streaming, Releases;
4. **apps:** client, server, launcher, updater.

Lower layers know nothing of upper ones. Client and server share VoxelCore and the domain libraries. Solo play launches THE SAME server program as a child process, so there is one code path for the simulation.

### 2.2 Specs, decisions and milestones are first-class
- `PROJET.md` holds numbered requirements per area. Code cites them, for example "client settings (PROJET.md U-01, U-02, U-03)".
- **Areas:** G generation, J player session, L light, N network, S server, T GPU transfer, U interface, V vertex format, W world.
- **More docs:** `docs/DECISIONS.md` holds design decisions, and `docs/JOURNAL.md` holds milestones such as M6c, F4-0, E6 and PERF-2.4.
- **IDs travel into the program.** They appear in assertion messages and log lines, such as "persisted (S-04)", "upload queue (T-05)" and "CompactVertex: grid coordinate outside the 7 bits of the format (V-04)". When something fails, the message names the decision behind it.

### 2.3 Invariants are written as rules
Assertions say the rule in words, not just the expression:
- "Air must be the first palette entry, and only it."
- "the cave lattice step must divide the FullChunk edge (global anchoring)"
- "the Chunk edge must stay an integer multiple of kLightGridEdge (W-05)"

### 2.4 Degrade, never crash, and say so
Bad input is "ignored", "treated as empty", "cut at the next write" or replaced "instead". Each case writes a log line saying what happened and what was used in its place:
- an event that is not legal in the current state is refused by name;
- a broken index buffer is rebuilt;
- a missing translation shows English;
- a refused asteroid sculpt falls back to the procedural shape;
- with no sRGB format it warns "colours will look off" and runs anyway.

### 2.5 Logging as a measuring instrument
- **Every line:** level, category, thread tag, and the source file:line.
- **Machine-readable events:** `EVT event=… key=value` lines.
- **One stats line per subsystem**, always as counters and timings: tick mean, p99 and max, creatures by tier, paths, light jobs, the streaming state, allocator blocks.
- **The hitch line** (§1.11) for anything that runs per frame.

### 2.6 Files and formats
- **Headers:** every file and wire blob starts with a 4-byte magic and a version.
- **Stable names:** materials keep a stable key forever; retired ones name a successor.
- **Files by concern:** state that is not voxels goes in sidecar files per concern (chests, machines, growth, curing).
- **Repairs run once:** a one-time repair leaves a stamp file.
- **Rebuildable state:** derived state can be rebuilt from the voxels if its file is lost.

### 2.7 One class, many files by facet
A big class is split across files named for the facet each one handles: StreamingClient, …Lots, …Light, …BakedLight, …Meshing, …Groups, …Edits, …Doors, …Falls, …Render. Likewise MachineService + MachineServiceFluids. Each file stays readable without inventing new abstractions. Older files are named in French, newer ones in English; they did not rename for the sake of renaming.

### 2.8 Tests, tooling, dependencies
- **Drift tests:** a test asserts that the shipped config file equals the built-in defaults, field by field.
- **Headless decoders:** the dashboard's binary decoders are tested without a browser.
- **Debug switches** are environment variables, one per stage.
- **Few dependencies,** statically linked, every one listed with its licence.
- **Signed releases:** a sha256 and a signature per file.

### 2.9 Content is data
Several things are tables, not code:
- the material table with flags;
- the texture pack list (tile, metres covered, animation, family);
- encyclopedia entries (how, what for, why);
- the progress steps (predicates over observed events);
- difficulty (one table of multipliers).

## 3. Design principles, in one list
1. Budget everything that can grow, and when a budget runs out, show a partial result and log it. Never stall.
2. Put hysteresis on every threshold (load/unload, despawn bands) so nothing flickers at an edge.
3. Order work by distance to whoever is looking, and cancel what has fallen out of range.
4. Keep the rendering grid, the light grid and the simulation grid separate, each sized for its own job, with the rule that ties them asserted in code.
5. Pack data small and let the GPU unpack it (7-bit coordinates, 12-bit material).
6. Do not compute what nobody will see: face-direction ranges, cave culling, decoration only when near, far terrain as heightfields.
7. Bake what changes slowly (light) and apply what changes fast (day/night) as a multiplier at draw time.
8. One authoritative simulation program, used for solo and multiplayer alike.
9. Decisions have ids, and the program cites them.
10. Measure per subsystem all the time, and on a spike, say which part cost it.
11. Degrade with a log line rather than crash, and repair derived state from the source of truth.
12. Content and tuning live in tables; code is the laws and the plumbing.

## 4. What Mark_1 can take, and when

### Now: no renderer needed
- **A slow-tick line in the "GEL" style.** When a sim tick runs over budget, print one line that splits it by law (heat, fluids, fire, minds, seam, labels) plus the active-set sizes. It pairs with the per-law timing in openworld-engine-notes.md §4.
- **Budget, partial, log** for our flood fills and model calls: the air labeller, island labels, perception rays, mind calls. Put a cap on each; when the cap is hit, use the partial result and log it.
- **Hysteresis on activation thresholds:** particle sleep, activity windows, and the mind tiers to come.
- **Rule-worded invariants that cite the ruling id**, as in openworld-engine-notes.md §1.8.
- **Magic + version headers** on saved worlds and on the `.npz` frames the renders read, so an old file is recognised and never misread.
- **One environment variable per law to switch it off,** for bisecting a bad scene, if we do not already have one.

### For the Blender renders, now
- **Mesh with a binary greedy sweep before export** (numba, 64-cell rows). This cuts Blender's polygon count by orders of magnitude on flat surfaces.
- **Render on a coarser display grid sampled from the 5 cm sim** when a shot is wide. Keep the 5 cm grid for close-ups.

### Later, only if Mark_1 gets a real-time view on the 6 GB laptop
- A display grid of 20-40 cm with 64-cell chunks.
- Binary greedy meshing, packed quads and vertex pulling.
- Face-direction ranges.
- Sky and block light channels with a day multiplier.
- Distance-ordered streaming with hysteresis, cancellation and per-frame budgets.
- Far heightfield rings with hole masks.
- One atlas plus a material table.

From Python, the simple route is an OpenGL binding (moderngl); the modern one is a WebGPU binding (wgpu-py). Choose when the time comes.

## 5. Should we decompile their actual code?
**Recommendation: no full decompile. Targeted reads only, when one specific question changes a Mark_1 decision.**

Why not the whole thing:
- **Low yield.** The programs are stripped and heavily inlined C++. Ghidra merged whole systems into single functions of 100+ KB. Reading one system takes hours, and what comes out is register soup that mostly re-confirms what the strings, shaders and data already told us.
- **It would not transfer.** Mark_1 is Python and numpy at 5 cm, with real conserved laws. Their C++ would not drop in even if it were clean. The DESIGN transfers, and we have it.
- **Better teachers exist.** Every technique above has a public, documented and often open-source version (§1.12), which is quicker to learn from and free to use.
- **Legal and our own rule.** Decompiling to rebuild their code goes past studying ideas, past what EU law clearly allows (the developer is French), and probably past their terms. Copying from it would break the rule we set ourselves.

When a targeted read IS worth it:
- **The test:** the question cannot be answered from strings, data or shaders, AND the answer would change something we build.
- **The precedent:** the room-heat read was one. It showed that rooms are solved as a steady state, not stepped in time, which changed our far-tier plan.
- **The method,** about 30 minutes each:
  1. find the log string's address;
  2. find the instruction that loads it in the disassembly;
  3. read the arithmetic and the constants around it.

  We record only the STRUCTURE (what is combined, and the thresholds), never the code.
- **Candidates if a need arises:**
  - how exactly they hide the seam between near voxels and far tiles;
  - the streaming priority formula;
  - the creature tier transitions.
