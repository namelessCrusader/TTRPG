# Trajectories v3 — reactive, per-entity, living-world corpus

Ten scenarios, once each, through the **v3 setting**: a big-model author composes a typed
program for **every acting entity** from *that entity's own `perceive()` slice* (what it sees +
remembers, possibly stale); decisions are **double-buffered** (everyone decides from the frozen
slice-start belief), applied in **stat-precedence** order against the live world (a losing/stale
claim **refuses** via its guard), and **physics ticks between beats** so the world moves on its
own. Each corpus row is `(entity, perceived-slice, program, outcome)` — the covariate-shift
training set the trained-DM direction is gated on.

**Each beat's narration shows three things:** a top-down **MAP**, the **pseudo-grammar** of every
action (`aims at X — only if <guard>; contest A vs B [win → … · lose → …]`), and its
**resolution** (which branch fired + what applied) + what the entity **sees**.

**Health:** 10 scenarios, 5–6 beats each, **133 per-entity pairs**, 2 malformed programs refused
(filter working), **0 surviving invariant breaks** (the one that occurred — a move into a wall —
is now prevented engine-side).

## The ten
| file | what happened |
|------|----------------|
| `dungeon_press.md` | Hostiles start adjacent, no opening to loot; the player trades blows (scuttler → hp 7) but the broodmother closes and **both land clean hits at once (−3,−4): player hp 6 → −1, dead** (`ach_neardeath`, xp 5). Real crawler lethality. |
| `broodmother_ambush.md` | 6-beat survival chase: pinned in melee (hp 10→5), a fear-shriek leaves the player `afraid`, they hurl a waterskin and dive south — end at **2/10 hp, both hostiles still at d1**. |
| `two_looters_race.md` | **Sly wins the vial** — two cells away, he claims it on beat 2 before the player (five cells out) reaches line of sight; Bran gives chase, the player arrives too late. |
| `rooster_rival.md` | Cow the charge → feed qi (cowed→calm) → pet → **whistle it back mid-charge, the recall clearing its attack-guard in the same beat** before it pecks Meiling's hen; she warms 0.3→0.5 at the firm-but-kind handling. |
| `vault_ablaze.md` | Oil smashed + sparked; fire **spreads on its own from one tile to five** (heat past 200) while Bran and Pip flee (`escaped`); the player grabs the vial and bursts out as the room engulfs. |
| `market_haggle.md` | On the real guildhall cast: the player warms Maren the alchemist (0→0.61) while Voss the financier holds firm; the purse-claim lands on a **compound guard** (`not taken` AND `alchemist disp > 0.3`). |
| `qi_to_ground.md` | Qi poured 8→16, an earth-spirit wakes at threshold; **Meiling (now in range) arcs wary→awed (0.3→0.55)** while a player↔spirit bond forms (0→0.4). |
| `teach_meiling.md` | Two small qi gifts drift Meiling 0.3→0.6 (`touched`→`fond`); the trust-scaled teach-ask (86 vs 65) wins clean; mentor/apprentice sealed. |
| `irrigate_field.md` | Till → sow → irrigate; water spreads across ~20 cells, seedbed to growth 2; Meiling fetches water and warms 0.3→0.7 (now in-range, no overshoot). |
| `stealth_past.md` | Player stays unseen the whole run (held ≥4 cells from the scuttler, which idles) — seen/unseen is the whole game; it moved to stay safe rather than toward the (never-revealed) stairs. |

## Findings from the batch
1. **Partial observability drives outcomes** — Sly wins on position/info; the stealth run stays safe by staying unperceived; hostiles only act on what *they* see; Meiling participates only when in range. The world isn't omniscient, and the data shows it.
2. **The double-buffer is visible in the data** — in `rooster_rival`, Meiling (high precedence) tried to approve "the settled rooster" but acted *before* the rooster set `settled` that beat → clean `guard-false`. You can't react to a same-beat peer; that's next beat. Correct semantics, caught in the wild.
3. **Combat is lethal when it engages** — the close-start fix turned the earlier fizzles (`press`, `ambush`) into a death and a 2-hp escape. HP falls, fear lands, terrain matters.
4. **Two engine fixes this batch** (both surfaced by the runs, now with tests): `adjust_edge` **clamps disposition to [-1,1]** (was overshooting to 5.3); `move` **refuses walking into a solid/out-of-bounds cell** (an authored move had put a guard inside a wall — the invariant detector caught it; now it can't happen).
5. **Sim-as-filter** — 2 malformed programs (a `move` with no `dir`, a list-dialect `set_tag`) were refused and the authors moved on.
6. **Tuning:** `stealth_past` needs a *known* goal (the stairs never entered perception) so the player has somewhere to head; otherwise "stay unseen" alone is aimless.

## Generators (scratchpad, temp)
`v3.py` (step-server: init/step/finish, pickle-between-steps, map+grammar+resolution narration),
`perceive.py` (belief-fold + id-addressed encoding), `timeslice.py` (loop + conflict tests).
