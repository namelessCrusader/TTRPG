# Data Engine — Engine & Model Design Consolidation

_Companion to the status report ([data-engine-report.md](data-engine-report.md)). That doc ends at
§4c ("menu-select + a write-nothing propose channel; property-derived menu is the next build"). This
one consolidates the design work done since: the reframe, the decisions taken, the decisions still
open, the ordered engine todos with reference implementations, and the model spec._

---

## 1. The reframe

**Menus are not reductionist. Enumeration is.** A menu built from a fixed verb table caps play at
whatever a human wrote down. A menu *derived* from object properties and coupling rules is not a
cap at all — it is a readout of the world's expressivity at that instant. The complaint "the list
grows linearly with the options" is a complaint about enumeration, and it dissolves when the list
is computed rather than authored.

**The project's own evidence says the same thing.** The `brood` scenario experiment held the pen
fixed (menu-select), varied only the world, and moved inventive picks 0% → 30% (~75% of
non-navigation decision beats). The ceiling was the world, not the constraint.

**Consequence.** Creativity is a property of the environment, not of the author's pen. Free
composition's *reachable* space is larger, but its *realized* space was 78% move+speak with 46% of
programs writing sim-owned state. Expressivity you cannot reach is not expressivity.

**Two kinds of creativity, and only one needs a pen:**

- *Compositional* — novel combinations of existing primitives. Fully expressible under menu-select
  once the menu is derived, parameterized, and compound-capable.
- *Ontological* — invoking a primitive the world lacks. Expressible under **no** pen, because a
  free author writing `set_tag charmed` is committing fiat, not being creative. This is what the
  propose channel is for, and its real job is **world growth**, not expressivity relief.

---

## 2. Decisions taken

| # | Decision | Rationale |
|---|---|---|
| D1 | Sim is authoritative; the author never writes effects | Settled by the §4a ablation; 0 authority violations by construction |
| D2 | Menu is **derived from object properties**, not an enumerated verb table | Grows multiplicatively (objects × properties × coupling rules) instead of linearly in authoring effort |
| D3 | Horizon is **bounded, 2–3 beats**, revised on new perception | §4b: myopia spirals, unbounded planning overruns the r=3 sight radius; bounded reached the goal where both failed |
| D4 | **Propose channel writes nothing**: NL proposal → map-to-macro or diegetic refusal; unmapped proposals log as affordance-gap tickets | Creativity gets a channel at zero coherence cost; the ticket log becomes an evidence-sized enrichment backlog |
| D5 | **Causal licensing**: a condition may only gate an effect if it is upstream of that effect in the declared coupling graph | Kills "take 30 hp if the sun is out" structurally. Legal-but-senseless is a *grammar* problem, not a taste problem |
| D6 | **No numeric literals from the author.** Three permitted forms: point at a world quantity (`carried_ml`, `distance`), pick an ordinal (`light`/`moderate`/`heavy`), or let the sim derive from the upstream delta | The constants audit (hp `{-1,-3,-10}`, 8 distinct disposition values) is the author improvising a currency it doesn't own |
| D7 | **Tag registry**: a tag is emittable only if ≥1 resolution rule reads it | Invented statuses (`blinded`, `blinded_turns`) become unreachable; load-bearing-tags becomes an invariant, not a cleanup pass |
| D8 | **One legality predicate.** Retire scratchpad `menu.py` for `engine.affordance_menu` + a perception filter | Two parallel impls have already produced two bugs (reach, walls) |
| D9 | Model is a **bi-encoder over the candidate set**, not a softmax over N action ids | Parameters independent of menu size; generalizes to options never seen in training; permutation-equivariant by construction |
| D10 | The big model's job is **discrimination, not generation** | It failed at generation because it had to handle grammar + legality + authority + taste at once. Give it only taste |
| D11 | **Selector and narrator are separate roles**, possibly separate models | Voice constraints get enforced in one place; the narrator need not be <500M |
| D12 | Narration layer is **deferred** | Worst case: annotate good trajectories. Not on the critical path until the engine works |
| D13 | **Schema shape = HYBRID** (material templates · flat flags · components-for-behavior) — see §2a | The engine is already templates+flags (`MATERIALS`, `tags`); parts add the minimum ECS for stateful objects without a rewrite. Chosen over pure-flat (no home for behavior) and pure-ECS (rewrite + over-engineering + structured encoding the <500M model doesn't want) |

### 2a. D13 spec — the hybrid schema (resolved)

Three layers, one boundary rule. The engine already has the first two; **parts** is the only new
machinery, and it is populated for only the handful of objects with stateful behavior.

- **Material templates** — extend `MATERIALS` (already `material → {tags, specific_heat, solid}`).
  Physical properties *every instance of a material* carries: flammable, specific_heat, density/
  weight, hardness, solid. Read by the coupling graph (reactions already match `has_material_tag`).
  One edit per material propagates (the DF win). *e.g.* `WOOD → {wooden, flammable}`.
- **Flags** — `entity.tags` / `cell.tags`. *Static boolean predicates the derived menu matches on*:
  `container`, `heavy`, `lever`, `sharp`, `wieldable`, `sealed`, `person`, `animal`, `prize`. Gated
  by the **D7 registry**: a flag is emittable only if ≥1 rule (a menu affordance *or* a reaction)
  reads it. This is what makes the menu *derived* (D2) rather than enumerated.
- **Parts** — a **new `Entity.parts: dict[str, dict]`** field (structured, unlike float-only
  `props`; distinct from `mind`, which is NPC cognition). *Stateful behavior / a held resource*:
  `LiquidVolume{mat, ml}`, `FuelBurn{ticks}`, `Trap{trigger, effect}`, `Container{capacity}`. A part
  (a) **advertises affordances** (`LiquidVolume` → `pour`) and (b) **may tick** (`FuelBurn` →
  decrement, ignite the cell when spent). This is where `fire→creature`, a burning torch, and
  "holds liquid *and* is flammable" live cleanly instead of as special-cased tick-rules.

**Boundary rule** (so this never gets re-litigated per object): *material property* → template;
*static boolean the menu matches* → flag; *carries state that changes over ticks or holds a
resource* → **part**.

**Migration cost:** one new `Entity` field + a part registry; most objects carry `parts: {}`.
No rewrite — templates and flags stay exactly as they are.

---

## 3. Decisions still open

1. **✅ RESOLVED → D13 (Hybrid).** Schema shape settled; spec in §2a. _(Was: flat flags vs
   components vs template inheritance.)_
2. **✅ RESOLVED — "composed tree from engine-listed pieces."** The engine lists the legal PIECES
   (head words, seen-target ids, ordinal values); the author composes a bracket tree from listed
   pieces only — flat for simple actions `(pour water on:#7 all)`, nested for logic
   `(if (near #2) (shove #2 E) (move W))`. The tree parses onto the EXISTING dsl.py JSON-AST
   (no new engine language — a friendly surface for the tree we already run), and the engine
   checks each NODE, so refusal names the exact bad piece. Depth is a dial: menu-select is the
   flat case; nesting is the same grammar deeper. Numbers stay unwritable (ordinals only);
   speech stays the one free-text field. Build: (1) ✅ **the piece-table maker is built** —
   every `Option` now carries `verb` + `args` (structured parts, populated at all 16 menu sites
   + the parts-afford tuples); `src/core/pieces.py` renders the verb-rows table and keeps the
   reverse index (composed answer → exact Option, or a nameable bad piece). Locked by
   `tests/test_pieces.py`: table covers the whole menu, index is one-to-one, only usable
   pieces are listed. (2) ✅ **the parser is built** (`pieces.parse/check`): flat lines
   (`smash target:cask_water`), bare values matched to their slot (`pour water at:SE`),
   sequences (`(then A B)`), and conditions with else (`(if (near guard) A B)`) — conditions
   (`near`/`has`/`carrying`) are checked at RUN time against the live world, so a false one
   skips its step quietly (guard semantics). **Every refusal names the exact bad piece**
   ("'lava' — no slot of 'pour' lists this value"). Plans resolve to real menu Options via
   the index, so a parsed plan is legal by construction. End-to-end locked by test: parse →
   check → step, the else-branch really runs, the item is really picked up (220 green).
   _Deferred, stated:_ `contest` (win/lose branches) waits for the dsl bridge.
3. **Selection signal.** Three candidates, probably all three eventually, order undecided:
   terminal predicates the sim already derives (`has_prize`, `escaped`, `allied`, `dead`, plus
   refusal count and oscillation); quality-diversity behavior descriptors (no ranking needed —
   only "reached *a* terminal" + "different descriptor"); a preference model distilled from
   pairwise big-model judgments.
4. **Where search sits.** MCTS over the affordance menu vs. MAP-Elites / novelty search. QD
   sidesteps the "we can't say which trajectory is better" problem entirely; MCTS needs the
   terminal predicates to be enough.
5. **✅ RESOLVED → build in-house.** MiniHack's action space is *enumerated* (fixed NetHack
   commands) — the very enumeration D2 rejects — so it structurally cannot test the core bets
   (property-derived menu, encode-candidate-as-content / generalize-to-unseen-options, coupling
   graph, propose/narration/social). The reachable-state metric (item 8) *reads* the property-derived
   affordances, so it's coupled to the in-house engine too. MiniHack/NLE kept only as an **optional
   big-model-as-policy calibration baseline** (BALROG-style), not a blocking prototype. Building
   in-house is incremental from the engine that already exists (D13).
6. **Social macros: keep or dissolve?** `reassure`/`intimidate` are at roughly the level of
   "befriend," not "offer water" — i.e. they are the enumerated thing D2 rejects. *Working
   proposal:* keep the sim owning the numbers (asymmetry + hysteresis are genuine world-physics
   and are verified), but derive macro **availability** from properties (shares language,
   perceives target, target has a disposition channel, has something to offer).
7. **Menu size vs. small model.** *Working answer, specified 2026-09-06; first cut built in the
   voxel sim (see `docs/voxel-roadmap.md`, "the menu seam").* Three sizes were being conflated:
   what the world can DO (D2 wants this multiplicative), what the engine COMPUTES, and what the
   picker READS. Only the third needs to stay flat.
   - **The binding constraint is the teacher, not the student.** D9's bi-encoder is fine with 300
     candidates — parameters are independent of menu size. Haiku is not. And Haiku does the
     picking during the harvest, so the harvest sets the cap: **~7, not ~12**. This is not a cost
     argument (30 short lines is ~200 tokens); it is a discrimination argument, and we already
     have the evidence in-house — the shuffle-order test reads vault 0.79 / **rooster 0.25**,
     i.e. picks tracking position rather than content at *today's* list lengths.
   - **World size must drop out entirely.** The menu binds only to perceived things (item 1
     already made `affordance_menu` perception-filtered, contact options binding to fresh belief
     nodes only). So menu size is a function of local scene density inside the perceptual
     horizon, never of map size. **Lock it as a test**: pad the world 10×, keep the room, assert
     the menus compare equal.
   - **Breadth comes from objects, not verbs.** There is no global verb list to cross with a
     target list: `parts.AFFORDS` hangs options on the objects themselves ("snuff the brazier"
     exists because `fuel_burn` does). Seven attended objects × ~3 uses ≈ 21 candidates, not
     50 × 40. The cross product is never built.
   - **Attention does the rest, and the cap IS the model.** Capping at ~7 is not a workaround for
     a weak picker — a person in a burning room weighs the fire, the door, the child, maybe a
     bucket, not forty things. Modelling the bottleneck is *more* true than pretending it is
     absent, which puts it on the right side of Ruling 1.
   - **If a level is still too wide, split it again.** Factored selection recurses: 50 verbs → 6
     classes → ~8 each. Depth grows as log of the action space, not linearly. Two rules keep it
     honest: a slot with exactly one legal filler is auto-filled and never asked, and **a verb is
     offered only if at least one legal completion exists** (no picking "throw" then discovering
     nothing is throwable). Real cost of depth: a bad pick at level 1 kills everything under it,
     and there is no way back unless we build one — *that* is the price to watch, not list length.
   - **Genuinely open-ended composition is not a menu at all.** Pour the oil, then light it, is
     the decision-2 piece grammar: a small tree built from listed pieces, checked per node. Size
     lives in composition, which is verified, not enumerated.
   - **Two builds fall out, neither done.** *Exploration slots* — hold 1–2 slots for low-ranked
     options, or the student permanently inherits the ranker's blind spots; an out-of-rank pick
     that turns out blessed is the highest-information trace available, and it grades the ranker
     for free. *Uncertainty gating* — skip the model call when the top option leads by a wide
     margin, call only when the top few are close; fewer calls, and the harvest concentrates on
     the hard cases.
   - **The failure mode to instrument is recall, not size.** A cap that quietly drops the right
     option is invisible unless measured, so menu recall goes into the trace logger from the
     start. Size is fixable by construction; a silent recall hole is not.
8. **Spatial substrate: grid vs. room/zone graph (continuous is rejected).** What the grid
   actually buys: cheap legality (adjacency, LOS, reach), cheap CA physics, an enumerable move
   set, and a cheap reachable-state counter. Continuous 2D/3D destroys all four — candidate sets
   become infinite (hostile to the pointer model and to search), legality becomes collision
   geometry — and buys nothing for a text DM. The live alternative is a **room/zone graph with
   qualitative relations** (in/on/under/behind/held), the substrate of IF, Versu, and Talk of the
   Town: language-native ("behind the bar," not (4,7)), and menus derive from relations directly.
   The `perceive()` encoding is *already* graph-shaped (id-addressed nodes in distance order,
   typed edges) — the grid is the substrate but the belief view is relational, so a swap is
   smaller than it looks. Costs: CA fire/fluid becomes zone-level quantities spreading along
   adjacency edges; r=3 sight becomes hop/scope visibility; and **tactical positioning degrades**
   (the wall-routing, corridor-blocking, and kiting in `brood`/`vault` are grid tactics).
   *Working proposal:* keep the grid now — BH results, `brood` tactics, and the reachable-state
   counter all lean on it — but keep the DSL/menu substrate-agnostic (targets are already ids;
   `{"cell":dir}` is the one grid-ism), and treat zone-graph as the v2 substrate if/when social
   and propose-channel play dominates over spatial tactics.

---

## 4. Engine todos, ordered by dependency

Everything downstream — legal candidate sets, node-level supervision, the reachable-state metric —
is blocked on the engine. Reference implementations in the right column.

| # | Item | Why | Reference |
|---|---|---|---|
| 1 | **One legality predicate** — ✅ **BUILT** (menu side): `perceive()` ported (`src/core/perceive.py`); `engine.affordance_menu(world, actor, slice_=None)` is now **any-actor + perception-filtered** — contact options bind only to *fresh* belief nodes, stale never yields a grab/strike/word, frozen `slice_` = the double-buffer contract, live path observes-then-perceives; **speech is a menu act** (`social=(eid, intent)`, earshot-gated; words = pick-time parameter, narration stays D11's slot). `tests/test_affordance_menu.py` + `tests/test_perceive.py` lock it (194 green). Scratchpad `menu.py` conceptually retired. ✅ **Any-actor `step()` too** — `step(world, opt, actor=None)`: deeds attributed to the actor on the bus, checked verbs roll against the actor, a social pick speaks AS the actor; derived consequences stay `actor=None` (the world's own). Round-trip locked by test (an NPC picks *and* executes; 195 green). **Item 1 complete.** _Remaining nearby:_ menu-entry output format awaits open-decision-2; per-verb range predicates become property-derived under item 2. **User principle recorded:** *action compositions may allow many things, as long as the world engine can verify them* — i.e., composition freedom is bounded by verifiability, the allow-during (offer-legal-pieces) form, not reject-after | The pointer model's candidate set must come from exactly one place | — (D8) |
| 2 | **Property / flag system** — _started:_ ✅ **the D13 parts layer is live** (`Entity.parts` in state.py; `src/core/parts.py` registry + stepper hooked into `reactions.tick` before rules read heat; first part `fuel_burn` — a self-consuming flame that warms its cell then gutters, the behavior tags/props can't express). Contract locked by `tests/test_parts.py`: private counters mutate internally, world-visible consequences ride the bus as `actor=None`; part-less entities cost nothing (198 green). ✅ **Parts now advertise affordances** (the derived-menu hook, D2 in miniature): `parts.AFFORDS` → plain `(label, effects, target)` tuples the menu wraps into Options — "snuff the brazier" exists *because* the `fuel_burn` component does, vanishes when spent, and rides the same fresh-belief + reach gates as everything else. Consuming an affordance writes part state **through the bus** (`set_part` effect op, attributed to the actor) — never direct mutation. Locked by tests (200 green). ✅ **`liquid_volume` — the first RESOURCE part** (no stepper; acts only when acted upon): affords "tip it over" (all contents → its cell, then the fluid physics spreads the pool) and "draw a skinful" (actor gains `inv:⟨mat⟩`, gated on not-already-holding) — **amounts are the part's own constants** (a draught = 200ml), the author picks the verb, the sim owns every number: D6's first concrete case. An empty vessel affords nothing (201 green). _Next:_ the flag registry (D7) → template extension → migrate the seed casks + burst rule off `holds:` props onto `liquid_volume` (touches seed/reactions/perceive/eval — its own step) | Makes the menu derived instead of enumerated. Everything in §4c of the status report depends on it | **Cataclysm: DDA** — flags applied via JSON item definitions trigger hardcoded effect chunks. `data/json/flags.json`, `doc/JSON/JSON_FLAGS.md`. Note flags are namespaced by domain (item / terrain+furniture / monster / ammo) |
| 2b | **Material templates** (property inheritance) — ✅ **first dedupe landed**: `corrodible` moved from ~10 hand-tags into the wood/metal MATERIALS templates; the acid rule reads `has_material_tag` — one edit place, and free emergence (acid now eats dropped metal tools and wooden *door cells* that never had the hand-tag). **`charred` debt paid**: charred things are brittle under smash (a real consumer; the D7 ledger test forced its removal). Deliberately *not* added: density/hardness — deferred until their consumers (derived-`heavy`, smash-scaling) land, per the property-nobody-reads rule (206 green) | Avoids hand-tagging every object; one `WOOD_TEMPLATE` carries flammability/density/yield to everything made of it | **Dwarf Fortress** — `USE_MATERIAL_TEMPLATE`; templates define broad classes ("stone", "milk") imported into specifics ("granite", "llama milk") |
| 2c | **Component composition** — ✅ **the `holds:` migration landed, item 2 closes**: all ~10 seed containers moved off `holds:` props onto `liquid_volume` parts; the burst op (`spill_contents`) reads the part and empties it **via `set_part` on the bus**; the perceive `holds[…]` surface reads the part; eval harness migrated; 0 `holds:` writers remain. Side effect of the migration: **every seed container now advertises tip/draw** (the menu grew because components spread — D2 working at scale). Two instructive test failures en route: my "no seed entity carries parts" premise went obsolete (good), and a test barrel without `hp` **burst on arrival** (the burst rule reads a missing prop as 0 — containers need hit points, noted). 206 green | "The menu grows with the world, not the verb table," in production | **Caves of Qud** — ECS where components are "parts"; a LiquidVolume part + a MeleeWeapon part = a weapon that holds liquid. Bucklew's IRDC 2015 talk. Their liquid system maps onto the fluid channel |
| 3 | **Coupling graph as data** — ✅ **DECLARED + locked** (`src/core/data/couplings.yaml`): channels + edges with `via:` rule-ids (or `code:` seams), a `missing:` section = the affordance-gap backlog made visible (torch-startle from propose-channel #1, smoke→sight, fire→fear). **Stale-claim correction:** "fire→hp has no owner" was carried forward from the v3-era read, but the base reactions already own it (`living_flesh_burns`, `extreme_heat_scorches_flesh`, …) — item 3's real gap was the *declared graph*, not the rule. `tests/test_couplings.py` locks it **both directions** (every `via` rule exists; every reaction rule is declared on an edge — an interaction not in the table does not exist) + the full **finding-4 arc end-to-end**: ignite → fire→hp drains → hp→life derives death → inventory drops, zero authored effects (210 green). _Remaining:_ the D5 causal-licensing reader (the graph's second consumer) arrives with the AST pen | `fire→hp` becomes row 1. Also read by the grammar for causal licensing (D5), so it is load-bearing twice | **BotW chemistry engine** (GDC: "Breaking Conventions with The Legend of Zelda: Breath of the Wild") — elements × material properties, all interactions as edges rather than special cases. Secondary: immersive-sim systemic design writing (Deus Ex, Dishonored) |
| 4 | **Tag registry with required consumers** — ✅ **BUILT as a CI invariant** (`tests/test_tag_registry.py`): a scanner harvests every tag *emitted* (yaml `add_tag`, pack verbs, literal `set_tag`, seed `tags={}`) vs every tag *read* (yaml conditions, verb targeting, code membership, MATERIALS templates); emitted-but-never-read fails CI unless ledgered **with a reason**, and a second test forces the ledger to shrink when debt is paid. Real harvest: 48 emitted / 50 read; **cell landmark tags** (`door`/`vault`/…) pass via a genuine generic consumer (a non-floor tag makes the cell *notable* to `perceive()`); true DEBT = `calm`, `charred`, `crawler` (each with its candidate consumer noted). 203 green. _Still open:_ runtime gating (offering only registry tags) arrives with the AST-derivation pen | D7 as a test | **DDA** `effects.json` (status with declared duration + consumers); **Qud** `IActivePart` (parts that conditionally enact behavior based on subjects and operational status) |
| 5 | **Ordinal magnitudes; channels own transfer functions** | D6. The constants audit becomes channel constants | — |
| 6 | **Non-dominated payoffs** — differentiated stats where they make choices genuinely different bets | The actual mechanism behind `brood`'s 0%→30% | **DCSS design philosophy** — explicitly avoids illusory choices where one alternative is always superior. **Brogue** (Brian Walker) on real tradeoffs |
| 7 | **Goals in the world** (stairs tile, winnable start, loot table) | Finding 7; the rollout showed a goal-blind actor just wanders | **tcod** tutorial (already in use for `brood`) |
| 8 | **Reachable-state counter** — ✅ **BUILT + first measurement** (`src/core/reach.py`, 74 lines): BFS over the actor's affordance menus with a **banded salient hash** (positions/tags/hp-int/part-state + cell fire/fluid/material; heat drift is noise, not novelty), applying options sweep-safe (bus + composer directly — no trace-harvest, no narration, no LM) + one tick each. **First numbers (vault, depth 2, 161–445ms): baseline 111 distinct states · +2 part-bearing objects → 312 (≈3×, combinatorial — the central claim, measured) · parts stripped → 90.** Deterministic; ordering + combinatorial-growth + noise-isn't-novelty locked by `tests/test_reach.py` (213 green). Social options skipped (need words); graph-search dedup keeps the sweep polynomial-ish | The feedback loop for items 2–7: milliseconds, sweeps over property assignments, tells you whether an edit made the world more generative *before* spending a rollout. Generative worlds grow combinatorially with objects present; flat ones grow linearly | No reference — built it |
| 9 | **NPC utility AI** for background entities | Hand-written is fine here; DF proves it. The model's value concentrates on protagonist-facing and narration-facing roles | Dave Mark's **Infinite Axis Utility System** (GDC; *Behavioral Mathematics for Game AI*); **The Sims** need/advertisement model; **GOAP** (Orkin, F.E.A.R.) if planning beats scoring |

### Sculpting notes from the item-1 walkthrough (user direction, recorded)

- **Self-targeting is legal, curation is not prohibition.** `"self"` is a first-class ref;
  rule-afforded self-actions appear in the menu (drop-and-roll when burning already does). The menu
  only curates *degenerate default spam* (attack-/speak-to-yourself every beat). Madness, grooming,
  self-directed acts are future rule-afforded self-verbs.
- **Body-part granularity** (clean/scratch/strike a specific limb) — entities are atomic today;
  when wanted, it is a **D13 parts-layer** extension (a `Body{limbs}` part; DF wounds-per-limb is
  the reference), not a new mechanism.
- **Range predicates beyond perception.** Perception is necessary, not sufficient; today's bands
  (touch cheb≤1, EARSHOT, adjacency) are hardcoded per verb. Under item 2 they become
  **property-derived**: touch / earshot / sight / thrown-range-N / line-of-effect — so "throw" can
  exist at range while "grab" stays touch.

### Carried over from the status report, still open

- ~~`fire→hp` rule in `reactions.yaml`~~ — **resolved (item 3): the rules already existed**
  (`living_flesh_burns` et al.); the finding-4 arc is now test-locked, and the claim that it
  "had no owner" is corrected — what was missing was the declared graph, which now exists
- Load-bearing tags — **mechanism closed by the D7 registry-as-CI** (item 4); `cowed`-persists
  itself lives in the *prototype* social macros, to be paid when they productionize
- `rally` affordance so `afraid` isn't one-way
- Carried-item co-locate in the **engine** pick-up, not just the rollout prototype
- Combat-ordering quirk: `npc_policy` runs before `reactions.tick` resolves death, so a mortally
  struck enemy still lands a dying bite
- Precise voice checker (named slice-entity + outcome-verb), **reject-and-resample, not strip**
- Presentation-bias result at N shuffles × more beats (current is n=1/scenario)
- Arm B (inject menu, keep free pen) — never run, superseded but not answered

---

## 5. Model spec

**Inputs** (all already built): belief-slice encoding; goal + last pick, round-tripped; bounded
2–3 beat intention. Move annotations are **slice-bound** — `test_annotations.py` locks the
god-channel guard.

**Architecture — bi-encoder over the candidate set.** Encode state once → `s`. Encode each menu
entry (verb + typed target + params) → `a_i`. Score `sᵀa_i`, softmax over the k candidates present
this beat. Compute is linear in k (k cheap encodings), parameters are not.

**Factoring.** Autoregressive verb → target → params, with pointer attention over slice entities
and cells. `pour ⟨amount ≤ carried⟩ ⟨fluid⟩ at ⟨perceived cell⟩` is one entry over a large legal
argument space: the sim owns the type and the bound, the author owns the choice.

**The long-term generalization — grammar-constrained AST derivation.** Build the program node by
node; at each step the candidate set is *the legal next nodes*, derived jointly from the DSL type
system and world state. Candidate set per step is small (10–50); the product over steps is
combinatorially large, so this **is** free composition — but every intermediate is legal, so every
complete program is legal. **Menu-select is the depth-1 special case of this.** Same model, same
encoder; you just stop the tree earlier.

What this does to the broodmother free-form program, violation by violation:

| Violation | Under grammar-constrained derivation |
|---|---|
| `adjust_edge disposition by −30` | `adjust_edge` isn't in the op vocabulary; `−30` is unwritable, not clamped |
| `set_tag blinded` + `blinded_turns` + `startled` | Requires registry entry + declared resolver; invented statuses are never offered |
| `contest [self.agility vs broodmother.agility]` | Contest args typed to stats both entities actually have; the phantom-stat coin flip is unreachable |
| `add_heat` + `spark` faking `blinded` | Stays legal — but with no hand-written consequence to fake with, the missing `heat→creature` edge surfaces as an honest dead end (i.e. a ticket) |

**Training signal.** Any legal program parses back into its derivation sequence, so the v2 + v3
corpora become supervision at **every node** — roughly 8–15× the examples from the same data — and
a partial program is a valid training state, giving dense credit assignment where there is
currently terminal-only. Plus: distillation from sim search; pairwise preference from a big model
for taste.

**Cost, stated honestly.** Beam search over derivations at inference, and the sim must answer
"what's legal next" given a partial tree. Same legality predicate as the menu, called at finer
granularity.

---

## 6. Reference index

**Engine internals (read the source)**

- **Brogue CE** — smallest readable codebase that does emergence well; clean and well-documented. Start here.
- **DCSS** — `github.com/crawl/crawl`. Read the design philosophy doc even if you skip the C++.
- **NetHack** — the reference for dense object interaction; replayability from item interactions and environmental tricks players still discover.
- **Cataclysm: DDA** — `github.com/CleverRaven/Cataclysm-DDA`; docs at `docs.cataclysmdda.org`.
- **Caves of Qud** — modding wiki at `wiki.cavesofqud.com` (Modding:Parts, Modding:Objects, Modding:Liquids, Modding:Active Parts).
- **Dwarf Fortress wiki** — Material definition token, Material science, Raw file, DF2014:Modding.
- **Curated index** — `github.com/marukrap/RoguelikeDevResources` (NetHack, DDA, DCSS, Incursion, Infra Arcana, IVAN, KeeperRL, Hauberk). Hauberk is Bob Nystrom's; relevant given the DSL/AST work.

**RL environments (the category previously missed)**

- **NLE** — `github.com/facebookresearch/nle`. Gym interface to NetHack; procedurally generated, entity-rich, cheap to run.
- **MiniHack** — `github.com/facebookresearch/minihack`. Environment-creation framework over NLE; controlled tasks scaled without arduous engineering; 500+ monsters and 450+ items with unique characteristics. **Best candidate for prototyping the reachable-state metric and the scenario-enrichment experiments.**
- **TextWorld** — generative control over difficulty, scope, and language; can relax partial observability and sparse rewards; varied-but-similar game sets for generalization.
- **Dungeons and Data** (arXiv 2211.00539) — large-scale NetHack dataset; precedent for sourcing policy data without a good author.
- **BALROG** (arXiv 2411.13543) — LLM/VLM agent benchmark on games; calibrates what a big model can do as a policy before committing to distillation.
- **DCSS as an AI evaluation domain** (arXiv 1902.01769).

**Sandboxes, interactive fiction, and social simulation** (the "what people say that games
don't model" category)

- **Inform 7** — open source (`github.com/ganelson/inform`). Its Standard Rules are a formalized
  ontology of everyday verbs: containers/supporters, wearing, edibility, light, locks, reach; its
  rulebooks (before/instead/after) are a reaction system. Forty years of IF is the largest
  catalogue of "things players try to say to a world." Direct input for the property vocabulary
  and the propose-channel mapper. **TADS 3** adv3 library for the richer sense model (sight /
  sound / smell, reach, postures) — maps onto `perceive()`.
- **Space Station 13, /tg/station** (`github.com/tgstation/tgstation`, AGPL) — deep simulation:
  chemistry, atmosphere, power, biology. The deepest open "anything you can say, you can do"
  interaction sim. Read `code/modules/atmospherics/Atmospherics.md` for the design principle
  ("theater, not simulator" — see §8).
- **Ensemble** (`github.com/ensemble-engine/ensemble`) — open rules-based social simulation,
  evolution of Comme il Faut, which powered Prom Week; volition rules over a social state pick
  what characters want to do. **The reference for open decision 6** (deriving social-macro
  availability instead of enumerating macros). Versu (Evans & Short, IEEE ToCI 2014) for the
  social-practices framing.
- **Talk of the Town** (James Ryan) / **Neighborly** (Johnson-Bey — modular Python
  reimplementation) — character knowledge, belief propagation, gossip; maps onto the belief-fold.
  **Kismet** (Samuel) for answer-set-programming social sim.
- **The Powder Toy** — open-source falling-sand: a material × material reaction matrix in
  production (the open cousin of Noita). **Luanti** (ex-Minetest) for an open voxel engine's
  node/callback system. **Barotrauma** (source on GitHub, non-commercial license) for
  fluid/electric/fire in continuous 2D — mostly useful as a warning about substrate cost.
- **ClubFloyd / Jericho transcripts** — real human free-text commands to parser games; candidate
  training data for the propose-channel NL→affordance mapper.
- *LitRPG-systems note:* Inform's extension ecosystem is the working model for community-authored
  packs — the `data/packs/` north star has a precedent with decades of use.

**Model / method**

- Pointer Networks (Vinyals et al. 2015) · DRRN (He et al. 2016) · Dulac-Arnold et al. 2015 (large discrete action spaces) · AlphaStar (autoregressive action factoring)
- Grammar VAE (Kusner et al. 2017) · **SD-VAE** (Dai et al. 2018 — semantic constraints in the decoder mask; closest to the legality-vs-sensibility split) · Abstract Syntax Networks (Rabinovich et al. 2017) · TRANX (Yin & Neubig)
- TreeLSTM (Tai et al. 2015) · code2vec / code2seq (Alon et al.)
- Attribute grammars; type-directed synthesis with typed holes (Osera & Zdancewic; SyGuS)
- MAP-Elites / novelty search (Mouret & Clune; Lehman & Stanley)

**No reference exists for:** the narration layer and the NL-intention channel. Games template
their prose; RL environments emit observations. That's both the risk and the reason the project is
interesting.

---

## 7. Metrics

| Metric | Measures | Status |
|---|---|---|
| Distinct reachable terminal states at depth 2–3 | Environment's generative capacity, **no author needed** | ✅ **built + measured** — vault depth-2: 111 base / 312 with +2 parts / 90 stripped (`reach.py`) |
| Coupling-graph edge count / density | Whether channels can propagate at all | ✅ declared (`couplings.yaml`): 15 edges, 3 declared-missing; drift-locked both ways by test |
| Unmapped-proposal rate over time | Whether the property basis is converging; should decline | Blocked on propose channel |
| Normalized menu rank + shuffle-order test | Picks track content, not index | Partial: vault 0.79, rooster 0.25, shuffle n=1/scenario |
| Inventive % | Play quality — **navigation-diluted**, report decision-beat share too | In use |
| Refusals, oscillation, format footguns | Coherence; all 0 under BH menu-select by construction | In use |
| Menu coverage vs. v2 inventive corpus | Expressivity of the macro set | 100% pacifist / 98% all; only the ~2% fire tail uncovered |

---

## 8. Risks

- **Legal ≠ meaningful.** Procedural spaces usually produce combinatorial soup: legal but
  payoff-equivalent, which is noise. Property density gives legal space; **payoff asymmetry** gives
  meaningful space. Both are required — `brood` worked because one-hit-kill and grab-and-run were
  genuinely different bets.
- **The DF lesson.** DF models far more than the game reads — its combat doesn't use all material
  properties, and weapon/armor wear isn't tracked. A property nothing reads is dead weight *and*
  inflates the menu without adding reachable states. The item-4 registry rule is the guard; keep it
  strict.
- **Scoping.** DF is ~20 years of one person. Build the coupling graph **scenario-driven, from
  affordance-gap tickets**, not comprehensively up front. `brood` moved 0%→30% with roughly four
  additions.
- **The SS13 theater principle.** /tg/station's atmospherics doc states it outright: the goal is
  not to simulate, but to put on a show of simulating — sleep wherever possible, simulate as
  little as you can get away with, performance and gameplay over realism. The deepest interaction
  sim in open source runs on approximations. Physics channels here should too: zone-level
  quantities and threshold events, never fidelity for its own sake — fidelity nobody perceives is
  the property-nobody-reads failure at the physics level.
- **Menu size vs. the <500M student** — see open decision 7.
- **n=1 throughout.** BH is replicated (3 trajectories, 2 maps, 3 goal types), but the scenario
  experiment is n=1, and the expressivity result ("macros can express what v2 authored") is not yet
  the stronger claim that a macro-only author produces comparable play.
