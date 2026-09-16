# TTRPG Mechanics Roadmap — the optimal compact set

Product of a 10-cluster research sweep over the open TTRPG design space
(awesome-tabletop-rpgs BFS: Ironsworn/Mythic solo oracles, Blades in the Dark,
PbtA, GUMSHOE, DramaSystem, Sine Nomine sandbox tools, FATE, OSR procedures,
ORE/YZE/Cypher resolution shapes, ultralight engines, Trophy/Wushu/Lady
Blackbird) synthesized against THIS engine's architecture and budget discipline.

Read together with OPEN_DESIGNS.md (videogame-side references). The rule stands:
mechanisms are budgeted, content is free; build nothing that doesn't collapse
several needs into one mechanism.

# OPTIMAL COMPACT SET — DM Simulator Mechanics Synthesis

## 1. DEDUPLICATION MAP (canonical versions)

| Family | Overlapping findings | Canonical |
|---|---|---|
| Graded outcomes | Ironsworn 3-tier/Pay-the-Price, PbtA 7-9, Blades 4-band+consequence menu, FU/24XX ladder, YZE stunts | **Blades 4-band + closed 5-type consequence menu** |
| Clocks/tracks | Blades clocks, Ironsworn progress tracks, PbtA fronts/grim portents, SWN goal clocks, Trophy ruin, GUMSHOE antagonist clocks | **Blades progress clock** (with per-segment authored events from PbtA portents) |
| Closed DM output | PbtA GM moves, Mythic event focus, Cypher intrusions, Fate compels, "announce badness" | **PbtA GM-move menu** (compels & intrusions become move templates) |
| Enum-then-roll adjudication | Ironsworn oracle ladder, Cypher level 1-10, Blades position×effect, Cairn save-vs-worse, 24XX declared stakes | **One classify-enum→seeded-roll primitive** with two modes: fact-likelihood, action-difficulty |
| Pacing dial | Mythic chaos, Fate point economy, GUMSHOE stall meter, d66 stall die, wandering-monster frequency, Cypher intrusion trigger | **Mythic chaos factor + one lull counter** |
| Tag economics | Fate invoke/create-advantage/boosts, Wushu details-as-dice, Risus clichés | **Tag-with-charges + p-modifier invoke** (Wushu = deterministic grounded-referent invoke) |
| Bargains | Blades devil's bargain, Trophy devil's bargain | **Identical mechanic** — one field on the check event |
| Offscreen world | SWN faction turn, assets-as-entities, Patchwork d6 resolver, rumor surfacing | **Patchwork d6 faction tick + assets-as-tags**, surfaced via existing gossip |

## 2. RANKED SET (per-line-of-code value, with port plans)

**1. Graded outcome bands + closed consequence menu** *(replaces binary p=f(stat))*
Split the existing seeded roll into 4 bands (crit / clean / partial / miss) via two thresholds on the same roll. Partial/miss draw ONE consequence from a closed 5-type menu (reduced effect, complication→clock tick, lost opportunity, worse position, leveled harm), authored as typed-delta templates tagged by context; small LM constrained-selects the fitting one. Kills "you fail, nothing happens." Cairn scars slot in here as pure content: one reaction rule on `hp_after==0` sampling a scar table (persistent tag → feeds gossip/dialogue free). Cost: threshold tweak + content tables.

**2. Universal Clock primitive** *(one ~50-line entity collapses five would-be subsystems)*
`Clock{id, size, fill, on_segment_events[], on_full_event}` fed only by existing bus deltas (tick ±n). Serves: alarm meters, quest/villain progress, faction goals, chase/deadline tension, ruin/doom counters, GUMSHOE antagonist escalation. Per-segment authored events = PbtA grim portents, delivered as gossip/witnessing deltas (players *hear of* the raid). Consequence type "complication" from #1 ticks clocks — the two mechanics interlock.

**3. GM-move menu + soft/hard pending deltas** *(replaces freeform consequence generation)*
Data file of ~12 move templates, each with preconditions over existing streams and a parameterized delta emission + narration stub. Triggers are mechanical (miss band from #1, lull counter from #5, chaos doubles, clock segments from #2) — 0 LM calls until a trigger fires, then one constrained selection. Soft move = a pending delta with a fuse + cancel predicate (one small delta-type extension); ignored pending delta commits as the hard move and is itself the "golden opportunity" trigger. Fate compels = move templates keyed to trait extremes/reputation tags. Threat impulses = a data-only weighting vector over the move list on NPC/faction records.

**4. One enum→seeded-roll adjudication primitive** *(the compression winner)*
Single interface for everything the sim doesn't model: LM classifies into a small enum, engine rolls seeded, result logs as a typed delta. Mode A — fact oracle: 5-step likelihood ladder ("is there rope in the barn?"), result = permanent `fact-established` delta (canon forever, no hallucination drift). Mode B — action rubric: position×effect (or level 1-10) from scene tags; position selects consequence severity band in #1, effect scales delta magnitude. Also subsumes Cairn hazard saves and 24XX pre-declared stakes (risk tag emitted with the classification, applied deterministically on fail). One classifier shape, three adjudication gaps closed.

**5. Chaos factor + lull counter** *(pacing thermostat, ~2 ints)*
One global int 1-9. Scene-end adjustment is arithmetic over the existing delta stream (count PC-negative deltas / failed checks vs plan-fulfilled) — no LM call. Feeds: oracle ladder bias (#4), random-event/GM-move trigger frequency (#3), NPC reaction thresholds. Second trigger: `ticks_since_last_salient_reveal` lull counter (absorbs GUMSHOE stall meter, d66 stall die, wandering-monster cadence — a per-region noise meter from existing loud-deltas can widen the check as a data rule).

**6. Faction tick with assets-as-tags** *(the world moves offscreen, 0 LM calls)*
Faction = data record {stats, goal-clock (#2) with concrete named sub-steps, asset tags}. Coarse tick: one seeded d6 vs 5-row table (1 = complication, 3-5 = tick, 6 = tick 2). Assets are NOT new objects — tags `{faction_id, role, hp_proxy}` on existing buildings/NPCs/stockpiles; a bus filter classifies voxel/NPC damage into faction damage, and faction deltas compile to authored world effects. Results surface only through existing gossip/rumor. Named sub-steps make player interference a ≤5-option classification. Collapses villain clocks, offscreen sim, and rumor content into existing machinery.

**7. Deterministic core clues + floating reattachment** *(anti-softlock)*
Clue = data record {anchor, trigger intent-tags, tier core|bonus, points_to}. Matcher on the existing intent-classification stream emits reveal-deltas — core clues NEVER roll (rolls stay for attack/throw/physical); bonus tier gates on stat/reputation. Floating clues reattach to the player's current location when the lull counter (#5) crosses their urgency threshold. Reveal-deltas can trigger faction-clock ticks (#6). One counter + one matcher rule.

**8. Petition/grant social frame + concession debt** *(typed social outcomes)*
One new intent label: `petition{target, ask, register: emotional|practical}`. Grant/refuse/partial decided by the EXISTING seeded reaction roll (traits + reputation + salience) — the addressed-NPC LM call just verbalizes the decided outcome (fits the 0-extra-call reactive budget). Plus one signed int per relationship edge: refusal builds debt toward the petitioner, modifying future grant rolls, auto-forcing a partial concession at threshold; a per-episode settled-flag stops rehashing. Fixes stonewall and pushover NPCs with a counter.

**9. Tags with charges: `create_advantage` + invoke** *(pre-empts any future buff/prep subsystem)*
Add `charges:int` and `lifetime:enum` to the existing tag type; one composer verb `create_advantage` (opposed check → set-tag delta with 1-2 charges). At check time, relevant tags grant a p-modifier — relevance via authored keywords in reactive mode (0 LM calls) or one constrained selection otherwise; hostile invoke uses the target's tags. Wushu variant folded in deterministically: +p per free-text referent that VERIFIES against world state (a lookup, not a judgment). Uniform adjudication of sand-throwing, flanking, taunting, scouting.

**10. Devil's bargain field** *(negotiation instead of a wall)*
Optional `{p_bonus, complication_deltas[]}` on the existing check event, offered when computed p is below threshold; LM constrained-selects the bargain from the #1 consequence content keyed to active clocks/scene tags; complication enqueues unconditionally on accept. Pure reuse of #1 + #2 content — near-zero code.

## 3. REJECTED (do not relitigate)

- **ORE width×height** — dual-axis return type adds complexity; band margin + stunt content tables already yield the info.
- **YZE push-the-roll** — a retry economy; bands + devil's bargain cover succeed-at-cost without new player currency.
- **Fate point economy** — two currencies + refresh bookkeeping; chaos dial does pacing with one int, compels fire as free move templates.
- **Cypher intrusion XP compensation** — same reason; trigger absorbed by chaos+lull, complication absorbed by move menu.
- **Mythic Action+Subject word tables** — two-word interpretation is open-ended generation, exactly what a 6GB model does worst; move menu + focus-targeting replaces it.
- **Ironsworn ranked progress tracks** — a clock with a size parameter; no second counter type.
- **Trophy ruin ratchet** — a one-way clock; content, not mechanism.
- **Lady Blackbird keys/buyoffs & Roll-for-Shoes emergent skills** — advancement economies; trait consistency is already enforced by trait-weighted seeded reactions, and unbounded tag growth is a liability.
- **Refresh/bond scenes** — gating recovery on classifying "genuine roleplay" invites misclassification; upkeep rest rules suffice.
- **Risus clichés as the skill system** — schema churn; engine already has stats + 4 traits.
- **Dramatic poles** — a 5th personality axis beside the existing 4 traits; violates trait budget for marginal gain.
- **Sacrifice-to-downgrade** — an interrupt menu on every hit; consequence menu already provides gear-break/morale texture.
- **Engagement roll** — author later as one scene-template data rule if approach-grind actually shows up; not core.
- **Wushu threat rating** — a new entity type; shelve until mass scenes are a demonstrated need (then it's ~30 lines).
- **Wandering monster check (standalone)** — absorbed by chaos-driven event frequency + noise-meter data rule.

## 4. ENGINE ALREADY HAS (validation, not work)

- **OSE/Cairn 2d6 reaction roll** ≈ seeded trait-weighted reactions — steal only the 5-band disposition table + reputation modifier as data.
- **Morale triggers** ≈ fight/flight — steal the discipline as data fields: check only on first-death / half-strength, two-pass lock suppresses re-rolls.
- **Mythic Characters/Threads lists** = NPC roster + goals/agendas, verbatim.
- **Faction-results-as-rumors** = existing witnessing/gossip propagation; only rumor text templates are new content.
- **Aspects-gate-affordances** = the affordance-menu design already locked; the steal is discipline (route ALL possibility questions through tag lookups) plus the `lifetime` enum, absorbed into #9.
- **Upkeep tick / torch-fatigue pressure** = pure data periodic deltas on the existing bus; no mechanism.
- **WWN tag packs** = a content schema for existing spawners (five slots → NPC presets, items, region markers); adopt as a data format when authoring sites, zero runtime code.

**Interlock note:** #1–#5 form one organism — enum classification (#4) sets band severity (#1), whose complications tick clocks (#2), whose segments trigger moves (#3), whose frequency chaos (#5) governs. Build them in that order.
