# Open Designs — the reference shelf

Steal **designs**, not engines. Each entry: the open source, the mechanic worth
taking, and how it lands in our substrate (tags + fluids + reaction rules + deltas).
Priority = how directly it upgrades something players already touch.

## 1. Fire & fields — Cataclysm: DDA  ★ HIGH
**Source:** [CDDA fire wiki](https://srgnis.github.io/cdda-wiki/cdda_wiki/Fire.html) ·
[JSON_INFO (fields/fuel)](https://docs.cataclysmdda.org/JSON/JSON_INFO.html)
- Fire has **intensity 1–3** (smolder / fire / raging), not a binary `on_fire`.
  Damage, spread chance, and light scale with intensity.
- **Fuel is a per-material float**; fire *needs fuel on its tile or it dissipates*;
  negative fuel values smother (water-soaked things subtract).
- Smoke is its own field that spreads and chokes.
**Port:** replace `on_fire` tag with `fire:1..3` intensity prop; per-material fuel
values in MATERIALS/FLUIDS; smoke as a gas fluid. Our reaction interpreter already
supports everything needed — this is data + 3 rule edits.

## 2. Temperature & materials — Dwarf Fortress  ★ HIGH
**Source:** [DF Temperature wiki](https://dwarffortresswiki.org/index.php/DF2014:Temperature)
- Every material has **IGNITE_POINT / MELTING_POINT / BOILING_POINT / HEATDAM_POINT**.
- The gem: **conduction = temperature difference ÷ specific heat, per tick** — one
  formula giving "metal heats fast, stone slow, water absorbs heat" for free.
**Port:** add `specific_heat` + the four points to MATERIALS; replace our flat
`conduct_heat`/`relax_heat` rates with the DF formula. Kills a whole class of
"heat feels wrong" bugs at once.

## 3. Fluids — Dwarf Fortress flow  ★ MEDIUM (we're close)
**Source:** [DF Flow wiki](https://dwarffortresswiki.org/index.php/DF2014:Flow) ·
[Pressure](https://dwarffortresswiki.org/index.php/DF2014:Pressure)
- Depth 1–7 per tile; moves **down first, then sideways, then up**  (we have down→side ✓).
- Depth-1 puddles **stop spreading** (our `cling` ✓).
- **Pressure**: fluid under load paths *through* full tiles to distant openings —
  this is what makes breaching a cistern dramatic.
**Port:** pressure-teleport is the missing piece; take it when a scene has a cistern.

## 4. NPC movement — Dijkstra maps (Brogue)  ★ HIGH
**Source:** [The Incredible Power of Dijkstra Maps](https://www.roguebasin.com/index.php/The_Incredible_Power_of_Dijkstra_Maps) ·
[visualized](https://www.roguebasin.com/index.php/Dijkstra_Maps_Visualized)
- One shared map per goal; every NPC just **rolls downhill** — N NPCs path for the
  cost of one flood-fill.
- **Safety maps** (multiply by −1.2, rescan) give *intelligent fleeing* — around
  corners, through doors — instead of our "step away from danger".
- Layered maps (goal + danger + avoid-the-player) sum into behavior that looks smart.
**Port:** replace per-NPC BFS in `mind._step_toward` when crowds or the energy
scheduler arrive; safety maps fix flee-into-corner immediately.

## 5. Sight — symmetric shadowcasting  ★ MEDIUM
**Source:** [Albert Ford, Symmetric Shadowcasting](https://www.albertford.com/shadowcasting/) ·
[RogueBasin FOV](https://www.roguebasin.com/index.php/FOV_using_recursive_shadowcasting)
- **Symmetry — if A sees B, B sees A** — is the fairness property stealth needs;
  our line-walk LOS is *mostly* symmetric but not guaranteed.
**Port:** when stealth becomes a pillar (fog of war, vision cones), this is the
algorithm; until then our LOS suffices.

## 6. Sound & scent — CDDA senses  ★ HIGH (unbuilt, high drama)
**Source:** [CDDA sounds.cpp](https://github.com/CleverRaven/Cataclysm-DDA/blob/master/src/sounds.cpp) ·
[design-balance](https://docs.cataclysmdda.org/design-balance-lore/design-balance.html)
- Perception is **three channels** — sight, sound, scent — and noise is a core
  balance lever ("loud = dangerous, always").
- Sounds are events with radius/volume, muffled by terrain; scent is a decaying map.
**Port:** our witnessing is SIGHT-ONLY. Sound events (smash, screams, blasts) should
create **heard-not-seen memories** — lower confidence ("something shattered in the
storeroom"), no actor attribution — a new epistemic tier between witnessed and told.
Scent = the mastiff's tracking sense. Both fit the existing deed/memory pipeline.

## 7. NPC agendas — The Sims needs AI  ★ MEDIUM
**Source:** [Zubek, Needs-Based AI (paper)](https://robert.zubek.net/publications/Needs-based-AI-draft.pdf) ·
[GMTK: The Genius AI Behind The Sims](https://gmtk.substack.com/p/the-genius-ai-behind-the-sims) ·
[AiGameDev: 21 tricks](http://aigamedev.com/open/review/the-sims-ai/)
- **Objects advertise** the actions they afford and the motives those satisfy;
  agents pick by need × distance. *This is literally our affordance system plus
  motive decay* — the two designs were made for each other.
**Port:** calm-band NPC goal selection = advertisement scoring (the workbench
advertises `work`, the cask advertises `drink`); motives decay per tick. Deepens
the "living stage" without any LM.

## 8. Structure — DF cave-ins  ★ LOW (someday)
**Source:** [DF wiki cave-in diagrams](https://dwarffortresswiki.org/index.php/40d:Water_flow)
- Unsupported terrain collapses; supports propagate.
**Port:** our `support` tag → connectivity check when a scene stars architecture.

## 9. Immersive-sim principles — Spector/Looking Glass  ★ PRINCIPLES
**Source:** [player-first design of Deus Ex](https://www.gamedeveloper.com/business/warren-spector-explains-the-player-first-design-choices-behind-i-deus-ex-i-)
- "Problems, not puzzles" (many valid solutions) · "never judge the player" ·
  and the underrated third leg: **choice, consequence, RECOVERY** — a ruined
  reputation must be repairable or players stop experimenting.
**Status:** first two are our architecture already. Recovery = keep social damage
healable (gifts/apologies raise disposition ✓; never make edges one-way ratchets).

## 10. "The DevTeam Thinks of Everything" — NetHack  ★ DISCIPLINE
**Source:** [TDTTOE](https://nethack.fandom.com/wiki/The_DevTeam_Thinks_Of_Everything)
- Not a system — **decades of accreted special-case content** (gloves let you wield
  a cockatrice corpse; falling downstairs while doing so petrifies you anyway).
**Status:** this is our eval-growth discipline (every play-miss becomes a rule or
golden case) plus reactions.yaml accretion. TDTTOE is a *practice*, already adopted.

## 11. Emergent narrative — Tarn Adams  ★ VALIDATION
**Source:** [GDC/interviews](https://www.gamedeveloper.com/design/q-a-dissecting-the-development-of-i-dwarf-fortress-i-with-creator-tarn-adams)
- "The dwarves exercise autonomy outside official duties — actors in their own
  stories; that's where emergent narrative comes from."
**Status:** that is precisely our agendas + salience + gossip stack. Keep NPC
autonomy sacred: never script what a goal can produce.

## 12. Social practices — Versu (Richard Evans)  ★ MEDIUM-LATER
**Source:** [Versu papers](https://versu.com/wp-content/uploads/2014/05/ptai_evans.pdf)
- Situations expose **context-scoped social affordances** (a funeral offers
  different moves than a card game); NPCs reason over shared practices.
**Port (later):** our social REACTIONS table is one flat repertoire; practices =
scoping repertoires by situation tag (auction / brawl / mourning). Data reshape,
not a system.

## 13. Drama director — Left 4 Dead  ★ HIGH, TINY
**Source:** [The Director](https://left4dead.fandom.com/wiki/The_Director)
- Estimate player **intensity**, alternate build-up and breathing room.
**Port — the compact trick:** we already track per-NPC salience. A global
`drama(world)` = decayed sum of salience + active fires + recent deeds gives the
scene a pulse for one screen of code; ambient triggers and wait-pacing read it.

## The compression map (many designs → few mechanisms)
| Designs | Collapse into |
|---|---|
| CDDA fuel-or-dissipate ✓ + DF conduction + fire ceilings ✓ | **one material-physics pass** (conduction by specific heat) |
| L4D Director + our salience meters | **one `drama()` scalar** read by pacing/ambience |
| Sims advertisements + our affordances + Versu practices | **affordances with motive/context tags** (later) |
| NetHack TDTTOE + our eval discipline | practice, not code |
| Spector's rules + Tarn's autonomy | principles, already load-bearing |

## 14. User-flagged repos — verdicts (2026-07-10)
- **[DwarfCorp](https://github.com/Blecki/dwarfcorp)** ★ MINED (source deep-dive 2026-07-10) — the portable designs, ranked:
  1. **Act trees as coroutines** (`TaskManagement/Act.cs`): behavior nodes are generators yielding Running/Fail/Success; Sequence/Select composites + per-creature Blackboard. Python generators are a 1:1 match (~40 lines) and give NPC actions natural interruption points for salience/witness checks between yields. *The highest-value port when NPC jobs arrive.*
  2. **Task board, pull model** (`TaskManager.GetBestTask`): one flat colony task list; NPCs pull by priority then cost (distance²), with cost × (1 + assignees) as a one-line anti-crowding trick. Maps onto our trait-scored goals (~30 lines). They abandoned their Hungarian-algorithm optimal assigner — greedy won; don't relitigate.
  3. **Failed-task blacklist with TTL** (`CreatureAI.cs:129`): ~1 min per-creature blacklist of failed tasks kills retry-thrash; failure *reason strings* kept — which for us feeds gossip ("Sly grumbles about the locked vault").
  4. **Two-stage need escalation** (`DwarfAI.Update`): dissatisfied → enqueue a satisfy-task that competes normally; critical → force-swap. Prevents oscillation with two thresholds.
  5. **Thought ledger** (`Thought.cs`): mood = 50 + sum of decaying timestamped thoughts; too unhappy too long → strike/quit announcing the "last straw" thought — free narrative. Sibling of our salience meter; adopt when moods matter.
  6. **The principle worth more than any system** (`WorldManager-Employees.cs`): payday is an ordinary *task* plus a *thought* — obligations flow through existing task+mood machinery, never a parallel economy subsystem. Their code independently validates our no-special-case rule.
- **[kevshakes/dwarf-fortress-simulation](https://github.com/kevshakes/dwarf-fortress-simulation)** ✗ MINED, MOSTLY HOLLOW (source deep-dive 2026-07-10) — scaffold, not simulation: physics/heat/fluids are literal `pass` stubs, the needs→behavior loop isn't wired end-to-end (every decision except move/work silently dropped), stairs unimplemented (agents fly between z-levels), and `is_passable` lets A* path *through solid stone*. Its path cache is a bug factory (FIFO, never invalidated on terrain change). **Worth keeping anyway:**
  1. *The cost-model seam*: `tile.get_movement_cost() -> float|inf` (soft costs: water, soil) — adopt the SIGNATURE so our BFS can become weighted Dijkstra later with no API change. A*/caching/spatial grids themselves: not at our 252-cell scale.
  2. *Needs shape*: per-need decay rates + `min(needs)` urgency threshold feeding decisions and mood — ~30 lines, needs as INPUTS to affordance choice, never a behavior override (their dispatch is the cautionary tale).
  3. *The invalidation pattern they lacked*: if we ever cache paths, a `terrain_changed` delta on our bus clears it — the bus makes correct invalidation trivial where their design made it impossible.
- **[Bonxai](https://github.com/facontidavide/Bonxai)** ✗ for now — VDB-like sparse voxel structure for robotics; pays off at thousands+ of sparse cells. Ours is ~400 dense cells in a dict. Revisit only if worlds go big.
- **[GridForge](https://github.com/mrdav30/GridForge)** ✗ — C# deterministic spatial hashing for streamed worlds; same verdict as Bonxai, wrong language besides.
- **[evoxels](https://github.com/daubners/evoxels)** ✗ — differentiable materials-science solver (Cahn-Hilliard, GPU). Wrong domain; our DF-style heat/fluids are the right fidelity for play.

## 15. NeverEndingQuest — harness steals (source deep-dive 2026-07-10)
The nearest cousin (LLM-authoritative D&D DM in Python). Its LM-as-authority core is
what we deliberately didn't build, but its HARNESS survived contact with real LM
failure modes. Ranked steals:
1. **The validation ladder**: silent REPAIR before reject (rewrite fixably-wrong
   actions, strip invalid ones, keep the rest) → deterministic contracts → schema →
   only then an LM referee; error-notes feed retries with model escalation; failed
   candidates NEVER enter durable history. Port: our composer/social sanitizers
   adopt repair-before-reject; the sim replaces their LM referee.
2. **Delta-only LM output + declarative merge policy + critical-field preservation**
   — independent convergence on our typed-effect bus, plus battle-tested edge rules
   (explicit `[]` means clear-not-noop; cascade-field checks).
3. **Idempotent context rebuild + geography-keyed compression**: strip stale
   state-bearing messages by marker and re-inject fresh authoritative state each
   turn; summaries trigger on LEAVING places (location→summary, module→chronicle).
   Matches our voxel/social structure natively — adopt when campaign length arrives.
Also noted: pre-rolled dice injected into prompts (determinism smuggled into an
LM-authority system — we have the stronger form natively); atomic write discipline
(lock→backup→tmp→fsync→rename) for when saves matter.

## 16. Latitude Games trio (source triage 2026-07-10)
- **AIDungeon (2019, archived)** — the incoherence NULL MODEL: state existed only as
  prose; a hardcoded 20-exchange window; regex death-detection; vestigial unused
  `game_state` dict. Its failure checklist, verified against us: authoritative
  state ✓(deltas) · output validation ✓(bus) · durable memory ✓(sim state) · action
  legality ✓(sim adjudicates) · entity identity ✓(stable ids) · determinism
  ✓(seeded replay). Keep: `ConstrainedStoryManager` (fixed verb menus + cached
  results keyed by choice-path) as historical validation of the affordance-menu bet.
- **clawcraft (active 2026)** ★ — Latitude's own conclusion after AIDungeon:
  *"No LLM calls happen at action-time; all gameplay is deterministic"* — an
  independent team converging on exactly our architecture. Steals: pure-function
  quest resolution `(agent, quest, rewards) → new state`; namespaced seeded RNG
  (`seed:purpose` strings); a no-network smoke-sim harness with a deterministic
  mock LM (template for our replay tests). Their lenient JSON parsing is the
  anti-pattern our constrained selection avoids.
- **world-of-wordcraft (2022, abandoned)** — one idea: PRECOMPUTED-embedding
  nearest-neighbor as a zero-generation classifier (offline CLIP vs 5 theme
  vectors). Fits the 0-LM-call reactive budget if we ever want vibe-classification.

## 18. Six LLM-DM repos (triage 2026-07-10) — four keepers, two skips
- **[Project_Infinity](https://github.com/electronistu/Project_Infinity)** ★ — real engine (SQLite authority, MCP dice server). Steals: **atomic compound resolvers** (attack = to-hit→damage→HP→death in ONE call — no mid-combat drift window; our to_effects already close, keep the principle); in-tool resource validation; `/sync` state-audit diffing narration against truth.
- **[DungeonGPT-JS](https://github.com/EdwardAThomson/DungeonGPT-JS)** ★ — production; the `docs/` design memos are the value: **tiered narration routing** (templates for routine, LM only for novelty/milestones — our reactive-budget as their explicit plan) and **marker-guarding** (an LM token can complete narrative milestones ONLY, never mechanical ones).
- **[claude-dnd](https://github.com/SergeyKhval/claude-dnd)** — tiny but clever: **dual-branch pre-rolled replies** (NPC pre-writes success AND failure responses, engine rolls and picks — one LM call, two branches; fits our latency budget beautifully); NPC knowledge isolation per file.
- **[daicer](https://github.com/lguibr/daicer)** — disciplined: dry-run resolution returning typed diffs before commit (≈our bus); **narration as an async consumer of the event log, never a blocking producer** — the right shape for our future narrator.
- **[NaNoGenMo 2020](https://github.com/NaNoGenMo/2020)** — for the delta-log narrator: **Chess Dreams' insight: narrate ANALYSIS of the log (blunders, turning points), not the log itself.** jaredly's agent-experience-becomes-prose is the sibling.
- **GameMasterAI** ✗ prompt-wrapper, skip.
Verdict: nothing beats the harness playbook; the keepers are seams for LATER features (narrator, combat compounds, dual-branch social).

## 17. NetHack Challenge 2021 report — validation + two cautions
**Source:** [nethackchallenge.com/report.html](https://nethackchallenge.com/report.html)
- **Symbolic crushed neural 3×** across 500k games: full-state memory + domain
  knowledge as decision procedures beat learned policy. Our sim-first thesis, at
  competition scale.
- **The best neural agent (RAPH) was a hybrid switching on monster proximity** —
  independently the same shape as our salience bands (deterministic tier, model
  engaged when it matters). Convergence again.
- **Caution 1 — metric misalignment:** their median-score metric rewarded
  camping over progression. Ours: never let the golden-eval number become the
  goal — the Haiku fun-playtest is the metric that can't be gamed.
- **Caution 2 — "expressible sub-objectives"** won it: strategies you can NAME
  (find Sokoban, get to Mines). Keep our goals/jobs nameable; resist opaque
  utility soups until the drive ladder truly overflows.

## Porting discipline
1. One design per loop iteration, TDD, eval+fuzz must stay green.
2. Every ported rule gets a **saturation/idempotence test** (the ignite-spam lesson).
3. Data first: if it can't be expressed in `reactions.yaml` + MATERIALS, question it.
