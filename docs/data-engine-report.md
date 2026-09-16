# Data Engine — status report

_A checkpoint of the data-engine work: what was built, what the Sonnet playtest of it
found, the root-cause reframe those findings collapse to, and the one open decision._

> **Companion:** [data-engine-design.md](data-engine-design.md) consolidates the design work done
> since §4c — decisions taken (D1–D12), decisions still open, the dependency-ordered engine todos
> with reference implementations, the model spec (bi-encoder → grammar-constrained AST derivation),
> and the reference/metrics/risks index. This report is the *experiment log*; that doc is the *plan*.

**Status:** v3 corpus reviewed → 10 findings collapse to one root cause → **channel/pen ablation
(§4a)** → **horizon isolated (§4b)**: inventiveness/legitimacy are set by the *author model* + *pen
constraint*, not horizon; horizon is a coherence trade → **bounded (2–3 beats)**. **Bounded-horizon
menu-select reached + grabbed the prize (10 beats, 0 refusals)** where both free-pen horizons failed.
**Social + possession macros** (sim-owned, asymmetric + hysteretic) close v2 coverage **56%→100%**
(pacifist) / **73%→98%** (all), leaving only the ~2% open-physics tail. **The fork is dissolved
(§4c): menu-select + a write-nothing *propose channel*** (map-to-macro-or-diegetic-refuse + log the
gap), not Full-vs-Hybrid. **Open:** remaining n=1 verification (multi-seed BH), then the load-bearing
**scenario experiment** (re-author the 0%-inventive scenarios; is the ceiling the constraint or the
world?), then build the propose channel + property-derived menu. 186 tests green.

---

## 0. What the data engine is for

The north star: a **deterministic sim is authoritative** (typed effect deltas on one bus,
seeded dice, replayable log) and a **small local LM** does constrained selection/composition
over it. To get a <500M model there, we need a **covariate-shift training corpus**: many
`(perceived-state → program → outcome)` pairs where a *big* model (Sonnet, acting for every
entity) plays the world and the *sim* stays the referee. This report is about producing that
corpus and discovering it isn't yet trustworthy.

The DSL is the interface: a **total, loop-free, recursive JSON-AST program** a model emits over
the effect VM — `(state, intent, target) → program → deterministic-or-seeded edit`. See
[dsl.py](../src/core/dsl.py) (250 lines, at cap).

---

## 1. What was built

### v2 — the trajectory harness (`docs/trajectories/`)
Scripted trajectories authored by ~10 DM subagents, executed through the DSL, rendered to prose
+ a canonical corpus (52 pairs). Proved the DSL expressive enough for creative routes (the
**pacifist / no-fight route** worked — "creative scenarios are possible"). It also exposed the
core limitation that motivated v3: a blind author produces **silent contradictions** — e.g. a
thief steals the vial and a second author, unaware, takes it again.

### v3 — the reactive, per-entity harness (`docs/trajectories_v3/`)
The step-server ([scratchpad `v3.py`], pickled between beats) where **each acting entity authors
from its own belief-slice**. The machinery, all newly built:

- **`perceive()` = belief-fold** ([scratchpad `perceive.py`]). An entity's world-model IS the
  fold of events it witnessed. `observe()` refreshes last-known snapshots of everything currently
  visible (sight r=3, line-of-sight); unseen beliefs persist and go **stale** (carry an age; may
  be wrong). Output is an **id-addressed proximity encoding** — nodes once in distance order,
  edges as typed id-triples, notable cells with spatial fields — so output pointers become
  bounded indices. Validated: the player and Sly genuinely disagree about the vial (STALE ~5t on
  the pedestal vs. carried in Sly's coat).
- **Double-buffer.** Everyone decides from the frozen slice-start belief; programs apply in
  **stat-precedence** order against the live world; a **guarded claim** that lost or went stale
  **REFUSES** via its own guard (fixes v2's silent double-take).
- **Physics between beats.** `reactions.tick(world)` runs the CA physics + `world.tick += 1`
  after each slice — the world lives on its own.
- **Rich narration.** Each beat shows a top-down **MAP**, every action's **pseudo-grammar**
  (`aims at X — only if <guard>; contest A vs B [win → … · lose → …]`), and its **resolution**
  (which branch fired + what applied) + what each entity **sees**.

### Engine fixes landed this arc (both surfaced by the runs, both tested)
- **Disposition clamp** — [effects.py `set_edge`](../src/core/effects.py) clamps `disposition`
  edges to `[-1, 1]` (was overshooting to 5.3).
- **Move walkability** — [dsl.py `move`](../src/core/dsl.py) refuses stepping into a solid or
  out-of-bounds cell (an authored move had put a guard *inside a wall*; the invariant detector
  caught it, now it can't happen).
- Plus earlier DSL additions: `{"cell":dir}` as a first-class ref, spawn-then-target in one
  program, cell-fluids readable as props, `min`/`max` args.

### The corpus
**133 per-entity `(perceived-slice → program → outcome)` pairs** across 10 scenarios, once each
([`corpus_v3.jsonl`](trajectories_v3/corpus_v3.jsonl)). Combat engages lethally
(`dungeon_press` = player death; `broodmother_ambush` = 6-beat chase to 2/10 hp). The
double-buffer is visible in the wild (rooster run: a high-precedence NPC can't react to a
same-beat peer, cleanly no-ops). See [trajectories_v3/INDEX.md](trajectories_v3/INDEX.md).

---

## 2. The Sonnet playtest — 10 findings

We then read the v3 corpus adversarially. The verdict: **the traces are incoherent in
systematic ways, even with a strong author.** The thesis — *if Sonnet can't do cool things
coherently, a trained small model has no chance* — so we can't proceed to training on this.

1. **Perception is tracked and then ignored.** The `sees` field gates nothing. Entities target
   and narrate things not in their slice (Meiling addresses a player she never perceives; the
   player targets a `barrel` absent from its own `sees`). Partial observability is the design
   premise (`stealth_past`) but it's decoration everywhere else.
2. **Simultaneity is authored as sequential.** Agents narrate the *outcome* of others' same-tick
   actions ("before it can escape," "as it twists away"). Worst: the **player agent authored its
   own death** ("I lie still, gone") — a category error. Death should fall out of the resolver,
   not be a declarable intent (else nothing stops declaring *survival*).
3. **Range and possession aren't enforced.** Player at (1,1) "snatches the prize" at (5,3), four
   cells away and unseen. **Carried items don't move with the carrier**, so an onlooker
   simultaneously "deduces it's already taken" while seeing it sit on the floor.
4. **Stakes are decorative.** Win-branches that do nothing; lose-branches that grant the goal
   anyway; tags set and never read (`wounded`, `off_balance`, `watchful`); **escape as fiat**
   (declaring yourself out with no exit tile); **fire never touches an agent** (no heat→hp
   channel); **infinite qi** (no pool, no depletion).
5. **Escalation is inverted; thresholds are ad hoc.** The dramatic "killing bite" (−3) is weaker
   than an earlier bite (−5) because the effect table is **re-improvised per beat**. One stat
   (`composure`/`qi`) is attack, defense, and morale at once — no tactical texture.
6. **Disposition ratchets instead of responds.** +0.1 every Meiling turn as a per-turn *tax*,
   for chores she can't see. Transgression cheap, redemption generous, no hysteresis.
7. **Premise/trace divergence.** Four of eight don't do what their header says: `dungeon_press`
   ("loot") is a pure combat trace; `stealth_past` has **no stairs in the state** and no failure
   mode; `two_looters_race` is unwinnable by construction; the earth-spirit spawns and never acts.
8. **Two engine issues corrupt the LM's inputs.** **Glyph collision** hides stacked entities in
   the map (the author reads that grid → hallucination next beat). **Refusals waste a beat**
   silently (`KeyError: 'dir'`, a bad effect) — retried identically, no diegetic content.
9. **The affordance set is tiny** (~5 verbs: move, hit, flee, speak, give-qi) relative to what
   the DSL supports. A crowbar sits unused for whole runs; half a farm floods and no one reacts;
   a fire-side water cask is ignored; the broodmother is just a bigger scuttler.
10. **Register leaks.** "Five hits from dead" — a numeric hp read surfacing as interior
    monologue. And `afraid` is one-way: no rally affordance exists.

---

## 3. The reframe — ten collapse to one

Tracing all ten back through the harness: **the authoritative engine we want already exists —
and it even ran during these traces.**

- [data/reactions.yaml:179-181](../src/core/data/reactions.yaml) derives **death**:
  `alive ∧ hp<0.01 → remove alive, add dead, drop_inventory`.
- [packs/dcc.yaml](../src/core/data/packs/dcc.yaml) derives **escape** (`at_level:0 → escaped`),
  **xp-on-kill**, near-death, levels.
- [engine.py `affordance_menu()`](../src/core/engine.py) hands out **pre-costed, pre-legal**
  moves: you can only "smash" a container *in reach*, "give" *within arm's reach*, "descend" *on
  a stairs cell*. Range is **structural**; `smash`=−5 and `give`=+0.4 are **sim-owned constants**.
- `reactions.tick()` runs **every beat** in v3 — so those derivations *fired*.

**The v3 harness bypassed that authority and handed the author a second pen.** It calls
`dsl.run()` on a **free-composed program per entity**, and `perceive()` is *shown* but never
*enforced*. So the author writes its own damage numbers, declares its own death and escape,
ratchets disposition, targets what it can't see, and narrates others' same-tick outcomes — and
the sim rubber-stamps it, **racing its own reactions that were about to derive the same thing
correctly.** (In `dungeon_press` the base death rule *would* have killed the player at hp<0; the
author narrated the death anyway — point 2 in the raw.)

### The mapping

| Finding | Root | Fix |
|---|---|---|
| 2, 4, 5, 6, 10 | Author owns **consequences** it shouldn't | Author stops writing hp/dead/escaped/xp/disposition — **reactions derive & speak them** |
| 1, 3 | Author owns **legality** it shouldn't | **Bind targets only from the actor's `perceive()` slice; enforce reach**; pick-up co-locates item with carrier |
| 8 | Input corruption | Render **stacks**; a refused program becomes an **in-world stumble** |
| 7 | Scenario authoring | Put the goal *in the world* (a stairs tile; a loot map that rewards looting) |
| 9 | Thin affordance surface | **Enrich** the menu; differentiate stats so actions differ |
| **4 (fire→hp)** | ***No owner*** — physics has no channel to an agent; nothing to derive from | **New heat→hp rule in `reactions.yaml`** (the one finding not covered by "reactions already do it") |

So points 1, 3, 5, 6, 8 (+ authority of 2/4) are the **same fix applied in several places**, and
the engine already does most of the hard half. **This is mostly a reconnection, not a rebuild** —
with two genuinely new builds surfaced under stress-test: the **fire→hp rule** and a **voice
constraint** (findings 2 & 10 are narration leaks no pen-choice touches).

---

## 4. The open fork — where the author's pen stops

> **Resolved by the ablation in §4a.** *Gated-free-compose is eliminated* — removing the channel
> already fixes perception, yet the free pen still wrote `disposition=-25` and improvised `hp-3`,
> and menu-select beat it on coverage. The live choice is now only **Full-menu-select vs Hybrid**.
> The three options below are kept for the record of how the decision was framed.

The shared, no-regrets work is identical under every path: slice-bound targeting, a range gate,
reactions-narrate-consequences, render-stacks, diegetic refusals. The one decision that changes
*what gets built* is **how much authority to strip from the per-entity author**:

- **Hybrid** _(recommended)_ — author picks a **sim-owned verb + a slice-bound target + the
  voice** for canonical-cost actions (attack/take/give/descend); may still **free-compose
  open-ended physics** (spark/pour/transfer-qi/spawn) — but **never** writes
  hp/dead/escaped/xp/disposition, and every target is slice-bound + range-checked. Draws the line
  where it actually is: canonical costs = a small closed menu; physics = genuinely generative.
- **Full menu-select** — author selects verb-id + target + voice, writes **zero** effects. Most
  authoritative and the most learnable for a small model — the runtime north star — but play is
  only as rich as the enumerated menu, so it needs heavy affordance enrichment first (this is
  point 9) and is the biggest lift now.
- **Gated free-compose** — keep full DSL composition; the sim binds targets from the slice,
  enforces range, refuses writes to sim-owned props, caps disposition per beat. Most creative,
  most gate-code, weakest coherence guarantee.

**The single question that resolves it:** do you trust a **rich-enough menu** (so there's nothing
to validate — the locked "affordance menus" direction), or **gates on free composition**?

**Current state:** ablation complete (§4a). Gated-free-compose eliminated; the live choice is
**Full-menu-select vs Hybrid**.

---

## 4a. Diagnosis — arm-A baseline (measured, this pass)

Before building anything, we measured the current corpus (arm A = free-compose, no menu).

**The prompt channel is confirmed.** Dumping one author input verbatim
(`two_looters_race`, t0): a single authoring call is handed **both a god's-eye MAP** (every
entity's position on the storey) **and all three entities' private slices side by side**. The
slices themselves are correctly disjoint — the player's has no vial and no Sly; Sly's has the
vial but not the player; Bran's has the player but not the vial — so `perceive()` is doing its
job. But the author sees the **union + a global map**, so respecting partial observability is
structurally impossible. **Findings 1 and 2 are explained and cheap:** author each entity from
**only its own slice**. Built: `author_view()` in the harness emits exactly that (no map, no
peers); the isolated authoring path is arm A′ and arm C's isolation at once.

**Numbers (`scratchpad/diag_coverage.py` over the 133-pair corpus):**

- **Coverage (finding 9):** of 130 authored programs, **move (46) + speak (56) = 102 = 78%**.
  Every world-manipulation verb is a long tail — hit 8, tag 9, fluid 3, qi 3, and
  fire/spawn/disposition **1 each**. 11 distinct objects targeted, and **35 of 70 entity-targets
  are just "player"**; the crowbar, the water cask, the eggs are never touched. Per-scenario verb
  diversity: `market_haggle` = 1 (pure speak), four scenarios = 2, max = 5 (`vault_ablaze`).
- **Perception violations (finding 1): 10 / 133** programs reference something outside the
  actor's own slice — e.g. the player targets `barrel_oil` and `vial` it can't see; Meiling
  references a `player` she never perceives, and vice-versa. Target driven to **0** once refs
  bind only from the slice.
- **Authority violations (findings 2, 4, 6): 60 / 130** programs write sim-owned state —
  **35 `adjust_edge disposition`** (the +0.1 ratchet), **19 `adjust_prop hp`** (improvised
  damage), 3 `set_tag taken`, 3 `set_tag escaped`. Notably **0 `set_tag dead`**: the "player
  authored its own death" was a *narration* leak (hp hit 0 via the reaction; the author narrated
  it anyway), not an effect — which sharpens finding 2 to "the author must not **narrate**
  outcomes it doesn't own," and the reaction speaks them.

**Cheap fixes landed this pass:** glyph collision fixed (stacked entities now render `#` + an
enumerated `stacked:` legend, so a collision can't hide anyone — everything measured downstream
is measured through an honest render); `author_view()` added for channel-removed authoring.

### The ablation (turns the fork into an experiment)

Same seed, same scenario, three arms; compare **coverage** and **violation count** against the
arm-A baseline above:

| Arm | Author sees | Author writes | Tests |
|---|---|---|---|
| **A** (baseline) | global map + all slices | anything (free-compose) | measured above |
| **A′** | its own slice only | anything | isolates the channel effect on findings 1–2 |
| **B** | own slice **+ the injected affordance menu** | anything (same pen) | does *showing* the menu alone lift coverage / cut violations? |
| **C** | own slice + menu | **menu-select only** (zero effects) | the constrained extreme |

Also instrument **menu-coverage** (per beat, which menu entry the program realizes, or ∅ =
off-menu) so B/C report it as a number. Three data points, ~an afternoon.

### Arm A′ result — the channel test (run)

For each of the 10 arm-A perception-violation beats, we took the **exact slice** the arm-A
author saw (stored in the corpus), stripped the god-channel, and handed it — alone — to a fresh
blind author (`scratchpad/measure_ablAp.py`). Result:

| | arm A | arm A′ (own slice only) |
|---|---|---|
| **Perception violations (finding 1)** | **10** | **0** |
| **Authority writes (findings 2/4/6)** | present | **still 8** (incl. `disposition=-25` out-of-range, `hp-3`) |

**Not one of the 10 blind authors hallucinated the absent entity.** No `purse`, no `neighbour`,
no `barrel_oil`, no `vial`. And the behaviour got *more* coherent, not less: Meiling — who
perceives only herself — tends her own herb-beds ("no need to mind a rooster I can't even see")
instead of addressing a player across the map; the vault player, with no `vial` in view, **flees
toward the exit** instead of snatching a prize four cells away.

**This cleanly separates two axes.** Removing the *channel* (what the author is shown) drives
finding 1 to **zero** — proven, no-regrets, so it becomes the real harness path. It does **not**
touch findings 2/4/6: the same blind authors, still holding a free *pen*, wrote `disposition=-25`
and improvised `hp-3`. **Authority is a separate axis that needs the pen constrained** (reject
sim-owned writes at validation; let reactions derive them) — exactly what arms B/C and the
shared-authority layer are for. The fork is now about the *pen*, not the *channel*.

### Arm B/C surface — is the menu rich, or thin? (built)

The fork's central worry (finding 9): does constraining the author to an affordance menu kill
creativity? We built a **perception-gated menu from the slice alone** (`scratchpad/menu.py`) — it
can never offer an action on something unseen — and rendered it across the corpus:

- **Avg 10.9 legal options per beat** (min 6, max 26). `vault_ablaze` t4 offers **26**: smash the
  cask, push the barrel/keg, give/pour/throw any carried item, ignite the oil pool, speak to Pip
  or Bran, move. **Not thin.**
- **The menu is richer than what free-compose actually used.** Afforded across the 133 beats:
  **push 71×, attack 63×, smash 35×, give 31×, fluid 31×** — yet arm A spent **78% of beats on
  just move + speak** (§4a). The richer actions were legal and available every time; the free pen
  simply reached past them.

**Implication for the fork:** the data **contradicts finding 9's worry**. Menu-select wouldn't
*shrink* the action space — the menu is broader than what free-compose produced, and it's legal
and perception-consistent by construction. So menu-select would raise coverage **and** guarantee
coherence at once, which tilts toward **Full-menu-select or Hybrid** over gated-free-compose.

Two honest caveats: **(1)** the generalized menu covers the physical/social core but **not yet
the pack verbs** (qi-pour, `descend`, `open_loot_box`) — those are the LitRPG systems, and
enriching the menu from each pack's `verbs:` is the next build and the real cost of a menu path.
**(2)** when nothing is in reach the menu is honestly thin (`dungeon_press` t0 = 8 opts:
approach / speak / wait), which is correct — there's little to do but close distance.

### Arm C result — menu-select (run)

Ran 8 blind menu-select authors (own slice + menu → return a *pick* + a voice line, **zero
effects**) across a diverse sample (`scratchpad/measure_armC.py`):

| beat | arm A did | arm C picked |
|---|---|---|
| `two_looters_race` t0 (thief) | move | **take the starfire vial** (the winning grab) |
| `broodmother_ambush` t0 | move | **attack the scuttler** (nearer of two) |
| `rooster_rival` t2 | speak | **give water** to the calm/fed rooster |
| `vault_ablaze` t4 | **tag `escaped`** (fiat) | **move W** toward the real exit |
| `market_haggle` t0 | speak | speak to Maren (haggle) |
| `dungeon_press` t0 | move | move W (scuttler at d3, give ground) |
| `teach_meiling` / `irrigate` t0 | move/tag | look / move-to-goal (degenerate menu) |

- **Coverage doubled: arm A used 3 distinct verb-kinds on these beats (speak/tag/move); arm C used
  6** (speak/move/give/attack/take/look). The decisive verbs free-compose ignored — take, attack,
  give — appear, and appear *appropriately*.
- **Every pick was coherent**, and menu-select killed the fiat moves: where arm A *declared*
  `escaped`, arm C *moved toward the exit* and let the sim decide.
- **0 off-menu picks, 0 perception violations, 0 authority violations — by construction.**
- Caveat confirmed: `teach_meiling`/`irrigate` menus were degenerate (no till/qi-pour offered) →
  authors picked look / move-to-goal. **Pack-verb enrichment is required** for those systems.

### Menu-coverage — against *whose* output? (run — corrected under stress-test)

`scratchpad/menu_coverage.py` buckets each program by whether a perception-gated menu (± pack
verbs) could express it. **The denominator is the trap:** arm A is a *lazy* author (78%
move+speak), so "the menu covers 88% of arm A" partly restates "arm A did little." Bucketing the
**v2 corpus — the *inventive* routes (pacifist, arson, heist, mimic)** — tells the real story:

| bucket | arm-A (lazy) | **v2 (inventive)** |
|---|---|---|
| menu-core (move/speak/attack/fluid) | 85% | **52%** |
| + pack-verb (qi-pour, spawn, descend…) | 88% | **73%** |
| social / raw disposition | ~1% | **11.5%** |
| tag (possession `taken`/`has_prize`; farming `tilled`/`sown`; NPC-state `watchful`/`busy`) | 7% | 13.5% |
| genuine open-physics (fire) | ~1% | ~2% |

**Honest read:** menu + pack-verb covers **~73%** of *inventive* play, not ~88%. The residual ~27%
is **not** mostly a free-compose hatch — it's **more macros to build**: social/persuade macros (the
11.5% raw-disposition), a possession macro (`taken`/`has_prize`), farming verbs (`tilled`/`sown`).
The *genuine* open-physics hatch is small (**~2%, fire**). But the menu-enrichment surface is
**roughly half the action distribution**, not the ~12% afterthought the arm-A number implied. That
enrichment is a **dependency of the menu path**, not phase-3 cleanup. (My earlier "~88% / ~2%" was
the shakiest inference in the doc; you were right to flag it.)

**Narrowed to the pacifist/inventive routes** (`dcc_no_fight`, `boc_redemption`, `heist_social` —
the ones that matter most for creativity) it is lower still: menu-core **50%**, +pack **56%**, with
**38% social/disposition** (persuasion, redemption). So **social/persuade macros are the single
largest enrichment item**, not a footnote — the menu path lives or dies on how well social play
reduces to sim-owned macros.

### Ablation verdict (corrected under stress-test)

| axis | finding |
|---|---|
| **Channel** | Remove it → perception **10 → 0**. Proven, no-regrets. Settled. |
| **Pen — legality** | Menu-select gives **0 perception + 0 authority violations by construction** — *after* fixing a `menu.py` reach bug (smash/take/push were un-gated; the "winning grab" of the vial at cheb-2 was an artifact, now removed; self-check = 0 out-of-reach entries). Findings **1, 3, 4-authority, 5, 6, 8** fall out. |
| **Pen — voice** | **Findings 2 & 10 are NOT eliminated by construction.** Menu-select still writes free voice, and **2 of 8** arm-C lines asserted outcomes/simultaneity ("*before the broodmother can close the gap*"; "*my fingers close around that vial before anyone even smells me*"). Needs a **voice constraint** (name only slice entities · assert no outcomes · no digits) + a cheap post-check — under *any* pen. |
| **Coverage** | Beats lazy arm-A (3→5 distinct kinds, after the illegal `take` is removed). Matching the *inventive* author needs the enrichment surface above (~half the distribution). |

**Still points to Full-menu-select or Hybrid**, but the gap between them is **bigger** than first
claimed: the menu path must build pack verbs + social/possession macros to reach ~90%+; a Hybrid
hatch guards only the ~2% genuine-physics tail. **The decisive open test is a full menu-select
rollout** (§6), not more single-beat picks.

### Stress-test corrections (this pass)

- **Reach bug fixed.** `menu.py` offered smash/take/push at any perceived distance (a *second*
  legality impl diverging from `engine.affordance_menu`, which gates at chebyshev ≤1). Fixed to
  match; a self-check now asserts **0 menu entries out of reach** — the test the 186 lacked. **The
  standing path should reuse `engine.affordance_menu` + a perception filter, not a parallel impl.**
- **Voice audit (8 arm-C lines):** 2 outcome/simultaneity leaks, 1 role-induced off-slice
  reference → the voice constraint above.
- **Idle beats:** **11%** of beats have a degenerate menu (only look/move/drop/wait), 6% no other
  entity — concentrated in the farm openers. **Finding 7 (goals + start positions in the world) is
  a prerequisite** for those scenarios, not phase-3 cleanup.
- **Constants authors write** (each must land as a menu-entry param / pack constant / sim prop — a
  dependency of enrichment): hp `{-1,-3,-10}`, disposition `{-0.1 … 0.3}` (8 distinct), qi-transfer
  `{3,4,8}`, water-ml `{150,200,300,400}`, heat `150`, oil-ml `800`.
- **Finding 4's "fire never touches an agent" has no owner** — it's not authority (nothing to
  derive from); it needs a **new heat→hp rule in `reactions.yaml`**.
- **A′ scope:** measured on the 10 violation beats only — proves removal *fixes* them, not that the
  other 123 are fine (idle-beat fraction above is the relevant tell).

### Full menu-select rollout — result (run + **claims verified**, several corrected)

Ran `two_looters_race` end-to-end (`scratchpad/rollout.py`; all 3 entities, all beats; per-entity
menu-select → pick + voice; sim-owned expansion; double-buffer; `reactions.tick`). **`author_view`
carries NO memory** — no goal beyond a static role, no last pick — which reframes the results:

- **Sly's coherence is STRUCTURAL, not planning.** With the vial in his slice both beats and the
  reach fix withholding `take` until adjacent, the *only* path is `move S` → `take`. The menu + one
  visible goal *forced* the sequence; he didn't plan across beats. (The stronger, honest claim.)
- **Bran THRASHED — not "self-corrected" (my error).** Positions `(0,2)→(0,3)→(0,2)`: **1 reversal,
  net-zero**. Both beats his slice showed the player north; t0 he mis-picked `move S` (away), t1
  `move N` (toward). With no memory linking beats, that is oscillation, not recovery — **the greedy
  degeneracy the rollout was meant to test DID appear**, for the actor without a forced goal.
- **Voice: my "3/6 leak" was mostly false positives (my error).** The keyword checker fired on
  *anticipatory motive* ("I mean to **close the gap**" — Bran's own intent) and generic "**before
  anyone** else" — neither a findings-2/10 outcome-assertion. Scored precisely, **0 of the 6
  rollout lines assert a named agent's outcome this beat.** The checker is over-tuned; the genuine
  leaks were in *arm C* ("before the broodmother can close the gap"). Fixes: a **precise** checker
  (named slice-entity + outcome-verb), and **reject-and-resample, not strip** — stripping filters
  voice by a different process than generated it, a train/inference mismatch that hurts a corpus.
- **Carried-item co-locate works** (vial rides Sly, `#` stack, off-floor). This one holds.
- **No discoverable goal** — the player never perceives the vial → investigates the cask. Finding 7.

**Corrected verdict:** menu-select is coherent **only when the goal is forced or in-slice** (Sly);
an ambiguous actor with no goal/last-pick memory **oscillates** (Bran). The real lesson is not "it
plays" but **"it needs a persistent goal + last-pick field in `author_view`"** — sequencing has no
carrier without it. That field is a **prerequisite, not polish**, and its move-annotations must be
**slice-bound** ("toward where you last saw the vial, ~5t stale"), never ground-truth, or the
god-channel regresses and silently undoes A′ — which needs its own test.

**Built + verified this pass.** `author_view` now carries a persistent **goal** (a
legitimately-known drive) + the **last pick** (round-tripped through the harness), and move options
get **slice-bound** annotations. `scratchpad/test_annotations.py` locks the god-channel guard: with
a stale belief it annotates toward the *last-seen* `(5,3)`, not the true `(5,0)` — and a target the
entity has never perceived gets no annotation at all. In the wired rollout, Bran's t0 nav error is
pre-empted (`move N → toward you`) and the goal-blind player stays honestly un-annotated.

---

## 4b. Horizon vs pen — the natural experiment, isolated (run, several claims corrected)

The prior sections leaned on v2 (**56%** inventive) vs arm A (**22%**) as evidence that the free
pen "can be creative." But that contrast varies **three** things at once — **horizon** (v2 authored
whole trajectories; arm A authored per-beat), **scenario set** (the two corpora are *disjoint*: v2
= pacifist/arson/cultivation, arm A = combat/farm/market), and **author prompt**. So it can't
attribute the gap to any one. This pass isolated them.

**Confound quantified (data-only).** Inventive share is **scenario-dominated, not horizon-dominated**:
*within* arm A (horizon fixed at per-beat) it ranges **0% → 58%** purely by scenario
(`broodmother_ambush`/`market_haggle` 0% — inventive-poor *by premise*; `dungeon_press` 58%), and
*within* v2 `dcc_greed` is 0% despite full lookahead. A scenario lever that moves the metric 58
points at fixed horizon means the disjoint-scenario cross-corpus gap can't be read as horizon. **The
natural experiment is confounded** — so it needed a controlled run (`hexp.py`).

**The controlled run (arm A′ vs A″).** Same scenario, same channel (own belief-slice only), same
free pen, **same author model (Opus)**; the *only* variable is horizon — **A′** authors 1 beat at a
time (myopic, fresh info), **A″** commits a **5-beat plan** from t0 (lookahead, stale info).
Protagonist = player; deterministic NPCs + physics evolve the slice identically. Two scenarios:
`vault_ablaze` (spatial/physical) and `rooster_rival` (local/social).

| vault_ablaze (same model) | inventive | authority-tainted | outcome |
|---|---|---|---|
| **A″ 5-beat plan** | **80%** (4/5) | **0%** | fast forward progress (reached (0,2) by t1), correct pour→shove→ignite sequencing; **back-half break**: committed a path past r=3 sight → **OOB overrun** (2/5 refuse) → cascade; prize NOT reached |
| **A′ myopic 1-beat** | **80%** (4/5) | **0%** | **front-half spiral**: lit a fire in its *own* southward path (t0), then spent **t1+t2 dousing it** (3 beats, net 1 of 3 cells); prize NOT reached |
| arm-A myopic (Sonnet, for ref) | 40% | 40% | improvised `hp` damage + fiat `escaped` |

**Three claims corrected (mine, this pass):**

1. **"Horizon doubles inventiveness (40→80%) and eliminates authority-taint" was a MODEL artifact,
   not horizon.** With the model held fixed, myopic-Opus and plan-Opus are **identical**: 80%
   inventive, **0% authority-taint**, both. The 40%/40% was arm-A *Sonnet*. So **inventiveness and
   legitimacy are governed by the author model, not the horizon** — a stronger author reaches for
   legitimate physics (and *away* from authority shortcuts) at *any* horizon. (Direct consequence for
   the <500M north star: a weaker student will be **both** less inventive **and** more
   authority-abusing — the corpus must not rely on author restraint; the **pen must be constrained**
   because the *model*, not the horizon, governs legitimacy.)
2. **Horizon's real effect is sequencing-coherence, and it is a TRADE, not a win.** Myopia →
   locally-plausible / globally-incoherent (fire-then-douse-then-stuck); unbounded planning → correct
   sequencing but **perception-overrun** (commits a path longer than r=3 sight, refuses in the back
   half). **Neither reached the goal in 5 beats**, for opposite reasons.
3. **The resolution is BOUNDED horizon** — a **2–3 beat intention inside the perception radius,
   revised on new perception** (front-half coherence without back-half overrun). This is now
   *empirically motivated*, not a preference: r=3 sight makes a 5-beat spatial commit run blind. And
   it is **spatial-specific** — `rooster_rival`'s **local social arc completed 5/5, 0% refusal**
   (cow→feed→qi→soothe→befriend), because an adjacent target never drifts out of reach; bounded
   horizon matters for **navigation**, not local interaction.

**Two findings that survive unchanged and reinforce menu-select:**

- **Social play is authority-bound regardless of horizon or model.** `rooster_rival` is **~100%
  authority-tainted in every arm** (A″ 5/5, arm-A 4/5) because "befriend" *is* writing `disposition`
  and there is no legitimate manip alternative. The **macro gap (§4a's 38%) is orthogonal to
  horizon** — no lookahead rescues it; only social/possession macros do.
- **Free-pen format fragility is real and model-independent.** Even Opus emitted **3–6 DSL syntax
  footguns per 5-beat run** — bare cell strings for `add_fluid`/`spark`, `speech` keyed `t` not
  `who`, effects written `{opname:{…}}` instead of `{"op":…}`, an invalid `{"ref":…}` target, bare
  contest args. `hexp.py` auto-corrects the recoverable ones (and *counts* them, so the footgun rate
  is a measured signal), but the unrecoverable ones **refused a beat outright**. A <500M student will
  fumble the grammar **more** — an independent, model-scaling argument for **menu-select** (nothing
  to mis-type). Reach/legality/consequence are already guaranteed by construction there; this adds
  *grammar* to the list the menu removes.

**Net for the fork.** The corpus's inventiveness/legitimacy is set by the **author model** + **pen
constraint**, and horizon should be **bounded (2–3 beats)** and only matters for navigation. All
three arrows point the same way for a small student: **constrain the pen (menu-select), bound the
horizon, don't bank on model restraint.** The next rollout should therefore be **bounded-horizon
menu-select** (commit a 2–3 beat intention, re-pick per beat on slice change), measured against v2's
inventive routes as ceiling — the user's item-3, now the load-bearing run.

### Bounded-horizon menu-select rollout — result (run: `vault_ablaze`, 10 beats)

The direct test of the §4b resolution, on the **same single-protagonist vault driver** as A′/A″ so
it's a third point on one axis (`hexp.py bh_*`). The author holds a **2–3 beat intention** in
`author_view` (memory) but commits only **this beat's perception-gated menu pick** to the sim; the
menu is rebuilt every beat, so a stale intention is re-picked against the live menu. Goal: find the
prize (`vial` @ far SE corner, 8 cells away, **unseen at start**) and take it.

| | reached prize? | idle/refused | oscillation | authority | format footguns | mean menu-rank of pick |
|---|---|---|---|---|---|---|
| A′ myopic (free pen) | ✗ (front-half spiral) | 3 wasted | — | 0% (Opus) | 3 | — |
| A″ 5-beat plan (free pen) | ✗ (back-half OOB overrun) | **40% refuse** | — | 0% (Opus) | 6 | — |
| **BH menu-select** | **✓ took it (t9)** | **0/10** | 1 (a deliberate wall re-route, not thrash) | **0% by construction** | **0 by construction** | **15.0 (never rank 0)** |

- **It structurally can't fail the way A′/A″ did.** No overrun (each beat re-validated → **0
  refusals** vs A″'s 40% OOB), no spiral (the intention carries sequencing memory → no fire-then-
  douse). It navigated *around* Pip, the cask, a barrel, a keg, Sly, and an interior support wall,
  and grabbed the vial. Both free-pen arms never reached it.
- **Reads situation, not menu position — the item-2 degeneracy is absent (verified, one claim
  weakened).** Raw picked-rank mean was 15.0, but that needs the denominator: vault menus averaged
  **20.1 options** (not the corpus-wide 10.9), so **normalized rank = 0.79** — all 10 picks in the
  *bottom* half, strong evidence of situation-reading. **Honest caveat:** the rooster run's
  normalized rank is **0.25** (reassure/feed sit near the menu top), so rank *alone* is weak there.
  The **shuffle-order test settles it**: re-authoring the same situations with menus reordered, the
  rooster author picked `offer water` at a *deep* index (11) — a warm gesture, not the near-top
  `speak` — and the vault author solved the same "Pip blocks east" obstacle by negotiating past him
  with "clear the cask" as its stated fallback. Picks track **content**, not index. (Suggestive at
  n=1 per scenario; the vault shuffle pick did land near the top, so a full presentation-bias result
  wants N shuffles × more beats — logged as remaining debt.)
- **The intention revised coherently** every time perception changed (Pip blocks east → push cask
  south; cask blocks the lane → cut east; barrel blocks → cut south; wall → route north around it),
  and the **goal annotation fired exactly on perception**: 8 blind beats with *no* directional hint
  (honest partial observability), then `move E → toward starfire vial` the instant the vial entered
  sight — the slice-bound guard (`test_annotations.py`) holding in a live rollout.
- **n=1 discharged — replicated across 3 trajectories, 2 maps, 3 goal types** (`hexp bh_*`, seeds
  fixed per scenario): vault-explore-grab (player, farm/vault map), farm-befriend (rooster), and the
  **`two_looters_race` thrash re-test** — **Bran the guard**, the actor that oscillated in the
  memoryless rollout (`(0,2)→(0,3)→(0,2)`, 1 reversal in 2 moves), now runs **E, E, E: 0 reversals,
  0 refusals**, and **re-planned coherently** mid-run (when Sly seized the vial it switched from
  tracking the player to chasing the thief). Normalized rank 0.79 for Bran too. **All three: 0
  refusals; picks situation-driven.** The bounded-intention field *fixes the thrash*. Honest counter-
  note: the thief **won the fair race** (grabbed the vial at t1 in the guard run) — BH plays
  coherently but does **not** magically always win, which is correct. *(A reseeded vault gives the
  same layout — the vault scene is layout-deterministic — so map variety comes from the farm/vault
  scenarios, not seeds.)*
- **Fixed a `menu.py` legality bug found mid-run** (same class as the reach bug): it offered a
  `move` into a *perceived* wall. Now gated from the slice — a cardinal move whose destination cell
  is a perceived solid (`support`/`wall`/`stone`) is dropped. Perception-consistent (only walls the
  entity sees), still the argument for retiring the parallel `menu.py` in favour of
  `engine.affordance_menu` + a perception filter on the standing path.

**Verdict:** bounded-horizon menu-select is the shape that works — coherent, legal, non-degenerate,
goal-reaching — and it beats *both* free-pen horizons on the exact scenario that broke them. The
open limit is **social play**, where the menu is still thin (§4b): the next build is the macros.

### Social + possession macros — built, and the coverage gap closed (run)

The §4b/§4a blocker: social play was **~100% authority-tainted at every horizon and model**, because
"befriend" *is* writing `disposition` and there was no legitimate macro (the 38% of pacifist play).
Built `social.py` (+ wired into `menu.py` and `rollout.expand`): the author **picks** a social verb
(`reassure`/`intimidate`/`offer`/`give`); the **sim owns the number**, with the two properties the
free pen never produced (finding 6, "transgression cheap, redemption generous"):

- **Asymmetry** — trust is *fast to lose, slow to gain*: max positive step `+0.20` vs a
  transgression `-0.40…-0.45`. One insult undoes ~3 kindnesses.
- **Hysteresis** — the step depends on the *current* standing: positives are **halved while hostile**
  (the slow climb out of distrust — this reproduces the `qi_to_ground` arc shape), **diminish near
  the trust ceiling**, and a slight against an already-trusting target is **softened** (a deadband,
  not a cliff). The **relational-state tag** (hostile/wary/neutral/warm/allied) is *derived from the
  disposition band* — the author never writes both the number and the label.

Also: a **possession macro** (`take` of a `prize`-tagged item now sets `has_prize`), and `give` was
re-routed from the flat, too-generous `+0.4` through the asymmetric macro.

**Result — `rooster_rival` re-run under bounded-horizon menu-select** (the scenario that was 100%
authority-tainted): the befriend arc **completed end-to-end** — `reassure → offer water (calms the
bird at disp≥0.3) → reassure ×3` — climbing `0.00 → 0.12 → 0.32 → 0.44 → 0.56 → 0.68 → 0.73`
(**allied / befriended**), the near-ceiling beat showing the diminishing return (`+0.05`, not
`+0.12`). **Authority-taint: 0% by construction** (down from ~100%), **0 refusals**, and the arc
*paid off in world dynamics* (the rooster stopped pecking the moment disposition crossed the calm
threshold — a sim reaction, not an authored one).

**Transgression branch verified (was untested).** The befriend climb above is monotone, so the
`-0.40/-0.45` loss branch never fired in it. Exercised separately in the *real* macro path: build to
`+0.48` (4 reassures), then one `intimidate` → `+0.08` (**−0.40, wiping ~3.3 kindnesses**, and it
sets `cowed`); the **deadband** holds — an `insult` at allied `+0.77` only drops `−0.225` (softened)
vs the full `−0.45` at neutral. Asymmetry is real in a live arc, not just the unit function.
*(Noted gap: `cowed` persists as the disposition re-climbs — a status tag that should probably clear
on re-befriending; a load-bearing-tags item.)*

**Precisely what the coverage number claims.** The 100%/98% is an **expressivity** result: the macro
set can *express what v2 authored* (its disposition/possession writes map to menu picks). It is **not
yet** the stronger claim that a *macro-only author produces comparable play* — that needs
menu-select authors to reproduce v2's inventive routes at similar quality. The `rooster_rival` BH
re-run is the **first data point** of that second test (one social arc, completed cleanly); the
scenario experiment (§6) is where it gets measured at scale.

**Coverage re-run vs v2 (`menu_coverage2.py`)** — the number item-4 said to watch:

| denominator | before (menu+pack) | **after social+possession macros** |
|---|---|---|
| pacifist routes (dcc_no_fight, boc_redemption, heist_social) | 56% | **100%** |
| all v2 (inventive) | 73% | **98%** |

Only **one** v2 program stays uncovered — the genuine open-physics **fire** (~2%), the hatch we
already sized. This confirms §4a's read: the residual was **macros to build, not a free-compose
necessity**. The menu path now expresses essentially all of the inventive corpus; the Full-menu-
select vs Hybrid fork narrows to whether that last ~2% physics tail is worth a gated pen at all.

### Scenario experiment — "was the ceiling the constraint, or the world?" (run: `brood`)

The load-bearing test. A **lean re-authoring of `broodmother_ambush`** (the 0%-inventive-by-premise
chase) applying the standard-roguelike design of the tcod tutorial — *holding the menu-select
constraint fixed, changing only the WORLD* (`hexp _build_brood`, `brood` scenario):

- **Differentiated opposition** (power/defense/hp, ending "the broodmother is just a bigger
  scuttler", finding 9): a **fast glass-cannon scuttler** (hp4/pow5/def0) and a **slow tank
  broodmother** (hp22/pow3/def5). Combat is now `dmg = power − defense` (`rollout.expand` updated).
- **A goal *in* the world with tactical routes**: an **egg-clutch (prizes) she DEFENDS** — a real
  heist (fight through / bait her off / snatch-and-run), not a fiat objective.
- **Enemy AI with a reason** (roguelike `HostileEnemy`, differentiated): the scuttler *chases fast*;
  the broodmother *holds her clutch* and only leaves it when the player threatens the brood.
- Usable terrain kept (loot boxes / crowbar) so the menu offers more than move+speak.

**Result — goal reached (egg taken), 0 refusals, 0 oscillation**, and inventive picks **push +
attack + take = 30%** raw over 10 beats (diluted by 6 transit moves; ~**75% of the non-navigation
decision beats**) — versus the flat original's **0%**. The differentiation produced real tactics the
author *read from the stats*: it **one-hit-killed** the glass-cannon scuttler (pow4 vs def0/hp4), and
against the tank it **chose grab-and-run over fighting** ("your strikes barely scratch her" → "just
one egg, then I'm gone"). The defender AI held ground until the brood was threatened, then lunged;
the author even planned to "draw the broodmother off her clutch."

**Verdict: the ceiling was the WORLD, not the constraint.** A flat scenario caps inventiveness *no
matter the pen*; an enriched one lifts menu-select play to inventive tactics with **no free pen**.
Enrichment (differentiated stats, world-goals, property-bearing objects, reactive AI) is the lever —
which is exactly what the property-derived menu + the four §4c mechanisms are for. *Caveats
(honest): n=1 for this scenario; the 30% is navigation-diluted (a bigger map = more transit); and a
minor combat-ordering quirk — a mortally-struck enemy still lands a dying bite because `npc_policy`
runs before `reactions.tick` resolves death (a load-bearing-ordering item).*

## 4c. The fork, dissolved — a propose channel that writes nothing (design, recommendation)

The fork was "Full-menu-select (safe, maybe narrow) vs Hybrid (expressive, a gated pen that can
still write something illegal)." It dissolves. **Recommendation: menu-select + a natural-language
*propose channel that writes nothing*.** Alongside its pick, the author may optionally *propose* an
action the menu didn't offer, **in prose, not DSL**. The sim adjudicates:

- **maps to an existing macro** → run that macro (sim owns the effect, as always);
- **doesn't map** → **diegetic refusal** — the attempt *happens in-world and fails* (it is corpus
  content, a real beat, not a wasted tick) — and the proposal is **logged as an affordance-gap
  ticket**.

The author never writes effects, so **nothing illegal can enter the corpus** — the A′/A″/menu
guarantees all hold. But creativity gets a channel at **zero coherence cost**, and the unmapped-
proposal log becomes the **evidence-sized enrichment backlog** ("17 runs tried to use the crowbar as
a lever" is a build ticket, not a guess). Discovery moves from ephemeral (happened once in a trace,
vanished) to permanent. *This is exactly how the social macros just got built — proposed by hand,
adjudicated, wired — now formalized as a standing mechanism instead of a one-off pass.* Corpus pairs
carry a flag for whether they came via menu or via a mapped proposal, so the student can be trained
with and without them and we can measure whether it can imitate that tail at all.

**Four mechanisms buy expressivity without buying illegality** (the menu grows with the *world*, not
the verb table) — the build order behind the propose channel:

1. **Property-derived menu, not enumerated.** Compute affordances from object *properties*
   (`flammable`, `brittle`, `heavy`, `lever`, `sharp`, `anchored`, `contains:⟨fluid,ml⟩`), not a
   fixed verb list. The crowbar goes unused because nothing tags it `lever`; tag it, and "pry the
   box" is legal without anyone enumerating it. Generalize what `smash` already does for containers.
2. **Parameterized entries.** `pour ⟨amount ≤ carried⟩ ⟨fluid⟩ at ⟨perceived cell⟩` is *one* entry
   over a large legal argument space — the sim owns the type and the bound, the author owns the
   choice. This is where free-compose's real expressivity lived, and it survives with constants
   sim-owned. **Absorbs the ~2% fire tail** (`ignite ⟨perceived cell with fuel⟩`).
3. **Compound picks.** Two legal entries + a declared relation (`feint→strike`, `throw→retreat`,
   `block→shove`). Combinatorics from composition; legality still checked per component. The
   waterskin-into-the-eyes-then-dive move — the single most inventive thing in the v3 corpus — is
   the target shape.
4. **Reactive world depth** (multiplies what a fixed verb set can *mean*): `fire→hp`, load-bearing
   tags (a set `afraid`/`off_balance`/`watchful` must change a later resolution or be removed), a
   `rally` affordance so `afraid` isn't one-way, differentiated stats, qi pool / jug capacity. Best
   evidence it works just landed: the rooster calming at a disposition threshold is a world doing
   work the author no longer has to fake.

**Status: design + recommendation, not yet built.** The propose channel and the property-derived
menu are the next builds; the load-bearing test that gates them is the scenario experiment (§6).

_(All macro work is in the scratchpad prototype — `social.py`, `menu.py`, `rollout.py`. The standing
path makes them a **data reaction-pack** per the LitRPG-as-packs north star; the prototype proves the
shape and the numbers. No repo code changed, so the 186 tests are unaffected.)_

---

## 5. Where things live / test status

- **Engine (repo):** [dsl.py](../src/core/dsl.py) · [effects.py](../src/core/effects.py) ·
  [engine.py](../src/core/engine.py) · [reactions.py](../src/core/reactions.py) ·
  [data/reactions.yaml](../src/core/data/reactions.yaml) ·
  [data/packs/](../src/core/data/packs/) (dcc, cultivation, farming).
- **Corpus (repo):** [docs/trajectories_v3/](trajectories_v3/) (10 scenario `.md` + INDEX +
  `corpus_v3.jsonl`, 133 pairs); [docs/trajectories/](trajectories/) (v2, 52 pairs).
- **Harness + ablation (scratchpad, temporary):** `v3.py` (step-server, now with `author_view`
  + stack-safe map), `perceive.py` (belief-fold + encoding), `menu.py` (perception-gated
  affordance menu, reach-fixed), `rollout.py` (full menu-select rollout: sim-owned expansion +
  double-buffer + voice post-check), `narrate.py`, `timeslice.py`; measurement:
  `diag_coverage.py` (baseline), `measure_ablAp.py` (arm A′), `show_menus.py` (richness),
  `measure_armC.py` (arm C), `menu_coverage.py` (off-menu %), `verify.py` (reach/idle/voice/tags),
  `verify2.py` (v2-per-scenario + rollout oscillation), `test_annotations.py` (slice-bound guard),
  `hexp.py` (horizon experiment A′ vs A″ **+ bounded-horizon menu-select** `bh_*` mode: single-
  protagonist driver, deterministic NPCs, format-slip normalizer with a footgun counter, frozen-menu
  double-buffer), `analyze_ops.py` (per-op inventive/authority buckets — the metric `verb_class`
  hid), `social.py` (sim-owned social macros: asymmetric + hysteretic disposition, derived relation
  tags), `menu_coverage2.py` (v2 coverage after macro enrichment: pacifist 56%→100%).
- **Tests:** **186 collected, green.** DSL additions covered in
  [tests/test_dsl.py](../tests/test_dsl.py); per-module line caps enforced by
  [tests/test_budget.py](../tests/test_budget.py) (dsl.py at its 250 cap).

## 5a. Plan progress (vs the "diagnose before building" plan)

| todo | status | finding / where |
|---|---|---|
| Dump one author prompt verbatim | ✅ | §4a — the god-map + all-slices channel confirmed; `author_view()` built |
| Fix glyph collision now | ✅ | §4a — stacks render `#` + a `stacked:` legend |
| Instrument affordance coverage | ✅ | §4a — move+speak = 78%; `diag_coverage.py` |
| **Arm A′** (own slice only) | ✅ | §4a — perception **10 → 0** |
| **Arm B** (inject menu, keep free pen) | ❌ **not run** | superseded by menu-richness (§ Arm B/C surface) + arm C + the full rollout. The "does merely *showing* the menu shift the free pen?" point is still open; say the word and I'll run it. |
| **Arm C** (menu-select only) | ✅ | §4a — 0/0 violations by construction; coverage **3 → 5** (was 3→6 before the illegal `take` was reach-fixed out) |
| Menu-coverage / off-menu metric | ✅ (corrected) | §4a — ~73% menu+pack vs *inventive* v2 (not ~88%); enrichment ≈ half the distribution |
| **Reach-gate `menu.py` to the engine's predicate** | ✅ | §4a — bug found + fixed; self-check 0 out-of-reach; "winning grab" was an artifact |
| **Full menu-select rollout** | ✅ (claims verified) | §4a — coherent only when goal *forced/in-slice* (Sly); ambiguous actor **oscillates** (Bran, 1 reversal). My "plays, no thrash" was wrong. |
| Carried-item co-locate on take/move | ✅ prototyped | §4a rollout — the vial rides Sly (off-floor); **engine pick-up still needs it** in the standing path |
| **Voice constraint (findings 2 & 10)** | ⚠️ checker over-tuned | §4a — my "3/6 leak" was ~all false positives ("close the gap" = own intent); genuine rollout leaks **0**. Needs a *precise* checker (named slice-entity + outcome-verb) + **reject-and-resample** (not strip) |
| **Goal + last-pick field in `author_view`** | ✅ **built + tested** | §4a — objective (drive) + last-pick carrier round-trips; **slice-bound** annotations; `test_annotations.py` locks the god-channel guard. Bran's nav error pre-empted (`move N → toward you`); goal-blind player stays *un*-annotated |
| Shared authority layer (slice-bound targeting · range gate · reactions own consequences · diegetic refusals) | ⬜ pending | §6 — the standing-path build (no-regrets) |
| Constants audit — the numbers | ✅ listed / ⬜ moved | §4a — hp/disposition/qi/fluid/heat listed; *moving* them = the enrichment build (menu-entry params), **not** phase-3 |
| Differentiate stats (composure ≠ attack + defense + morale) | ⬜ pending | §6 (phase 3) |
| Enrich menu from object properties + **pack verbs + social/possession macros** | ⬜ pending | §6 — the first build; ≈ half the inventive distribution |
| Put goals in the world (stairs tile · winnable start · loot table) | ⬜ **prerequisite** | §6 — closes finding 7; rollout showed a goal-blind actor just wanders |
| **Finding 4 fire→hp rule** in `reactions.yaml` | ⬜ pending | §6 — the one finding with no existing owner |
| Regenerate a batch + re-run the 10-point read | ⬜ pending | §6 |

---

## 6. Next

1. **✅ Ran + verified the rollout** (§4a): coherence is *structural/forced* (Sly), and an
   ambiguous actor with no memory *oscillates* (Bran, 1 reversal). Two over-claims corrected.
2. **✅ Added the goal/last-pick field** — objective (a legitimately-known drive) + the previous
   pick, round-tripped through the harness; **slice-bound** move annotations, with
   `test_annotations.py` proving they follow *stale belief*, not ground truth (the god-channel
   guard). Bran's nav error is pre-empted; a goal-blind actor stays honestly un-annotated.
3. **✅ Bounded-horizon menu-select rollout** (§4b) — reached + grabbed the far-corner prize in
   `vault` (10 beats, 0 refusals) where both free-pen horizons failed; picks track situation not
   position (normalized rank 0.79 vault; shuffle-test corroborates). **✅ Social + possession macros**
   (§4b) — rooster befriend arc at 0% authority-taint; coverage vs v2 **56%→100%** pacifist. **✅ The
   fork is dissolved** (§4c): the answer is **menu-select + a write-nothing propose channel**, not
   Full-vs-Hybrid.
4. **✅ Verification debt discharged** — menu-rank normalized (vault 0.79 / rooster 0.25, one claim
   corrected), shuffle-order test (picks track content), social transgression branch fired (asymmetry
   + deadband live), and the **multi-run BH** (n=3, 2 maps, 3 goal types incl. the `two_looters_race`
   **Bran thrash re-test → 0 reversals**, thrash fixed). *Still owed:* the **precise voice checker**
   (named slice-entity + outcome-verb, reject-and-resample) scored on a labelled sample; and a
   fuller presentation-bias result (N shuffles × more beats — current is n=1/scenario).
5. **✅ Scenario experiment (lean, `broodmother_ambush` → `brood`)** — the enriched world (roguelike
   differentiated foes + a defended egg-clutch goal + power/defense combat + usable terrain) lifted
   BH menu-select from **0% → 30% inventive** (~75% of decision beats), goal reached, 0 refusals.
   **Verdict: the ceiling was the WORLD, not the constraint** (§4b). *Optional extension:* the same
   treatment on `market_haggle` (the social 0% case) to confirm it generalizes past combat.
6. **Build the propose channel + property-derived menu** (§4c): NL proposal → map-to-macro-or-diegetic-
   refuse + log affordance-gap tickets; menu computed from object *properties* not a verb table;
   **parameterized** entries (absorbs the ~2% fire tail); **compound** picks. Make the social macros a
   **data reaction-pack** (`data/packs/social.yaml`, the LitRPG-as-packs north star) rather than
   `expand()` cases.
7. **Wire the standing path**: retire scratchpad `menu.py` for `engine.affordance_menu` + a perception
   filter (two parallel legality impls have now produced two bugs — reach, walls); carried-item
   co-locate in the **engine** pick-up (not just the rollout prototype); reactions own all
   consequences; diegetic refusals.
8. **World depth**: **fire→hp rule in `reactions.yaml`** (still the only finding with no owner);
   **load-bearing tags** (`afraid`/`off_balance`/`watchful`/`cowed` must modify a later resolution or
   be removed — `cowed`-persists is a known open bug); a `rally` affordance; differentiate stats; qi
   pool / jug capacity; §4a constants → menu-entry params / pack constants.
9. **Regenerate a small batch** and re-run the 10-point adversarial read. Then the deferred **speedup**
   arc (halt-gating + swapping the big author for a small fast model, >10 runs/sec).
