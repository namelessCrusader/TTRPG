# Trajectories — multi-turn playouts (data engine v2)

Ten short scenarios, each authored as a sequence of turns and executed through **one
accumulating world**, so state drifts and later turns depend on what earlier turns
wrote. Every line of prose is faithful: it reads the actual program AST and reports
what the seeded sim genuinely did (or refused). Read each `<name>.md` as a story;
`trajectory_corpus.jsonl` is the same thing as 52 `(intent → program)` training pairs.

Health this run: **52 turns, 0 refused, 0 dead no-ops** — 41 turns moved state, 10
were spoken lines that landed in the log, 1 was a contest that honestly failed.

## The ten routes (diversity is the point)

| file | route | the consequence chain it proves |
|------|-------|-------------------------------|
| `boc_farmer.md` | quiet farming | till → sow (gated on `tilled`) → water+qi-feed (gated on `sown`) → time passes (compound guard) → sprouts |
| `boc_cultivator.md` | gather & summon | meditate to raise qi → pour into soil twice (amounts scale by an if-arg) → soil qi ≥ 20 wakes a vein-sprite (spawn), which drains the soil and speaks |
| `boc_redemption.md` | social amends | a sneer sets `cruel` & drops her regard → a words-only apology is **taxed by `cruel`** and fails → a real qi-gift thaws her (+0.6) → the closing ask is made **easier by her warmed disposition** |
| `dcc_carl_fight.md` | combat & loot | threaten (contest) → it goes `afraid` → strike, the hit **scaled bigger because it's afraid** → hp≤4 gate unlocks the finisher: despawn, +xp, crack the box |
| `dcc_no_fight.md` | pacifism | de-escalate (contest) clears `hostile` → a water-offering (gated on *not hostile*) pushes disposition positive → slipping past is **gated on disposition > 0** — the win condition is a relationship, no HP touched |
| `heist_social.md` | theft by guile | flatter the apprentice (`trusted_by_pip`) → his goodwill **buys down** the guard check (bonus derived from Pip's edge) → the vial-grab needs a **compound guard**: distracted guard AND Pip's trust |
| `arson_route.md` | theft by fire | smash the barrel → **spill oil into its cell** → ignition **gated on "is there oil here"** → fire spreads to the guard's own tile → grab the vial while he flees. Pure physics. |
| `mimic_wall.md` | the surprise | demand an inert wall apologize (a wry near-no-op) → prod harder and the DM **reveals it as a mimic** (spawn) → it lunges and bites → fight back, scaled by `startled` → hp≤4 finisher |
| `earth_spirit.md` | emergence | speak to bare ground (no listener) → pour qi turn after turn until the cell has **drunk ≥ 15** → an earth spirit wakes (spawn) → the *same words* now find someone home, and it answers |
| `dcc_greed.md` | greed punished | crack boxes (`greed` climbs) → the accumulated greed **unlocks** a reckless grab that rouses the broodmother → she strikes (gated on `enraged`) → low-hp gate forces the rueful flight |

## What each proves about the engine

- **Consequence is real, not scripted.** Guards read tags/props/edges that earlier
  turns wrote; play the turns out of order and the chain breaks.
- **Richer conditionals land.** Compound `and`/`not` guards, prop-comparisons,
  edge-reads, and if-expressions inside magnitudes all show up across the ten.
- **NPCs talk back, and their lines bend on how they feel** (Meiling cold → warm;
  the scuttler wary → almost a farewell; the earth spirit answering at all).
- **The interesting thing is possible.** The wall *can* be a mimic; the ground *can*
  wake a spirit — spawned by accumulation, not special-cased.

## Engine capabilities this batch surfaced (all now covered by tests)

The DMs authored the natural thing and the sim's refusals told us exactly where the
grammar was too narrow. Four small, general widenings (no special-casing):

1. **A cell is a first-class `ref`** — `{"cell":"here"}` works inside args/effects,
   not only as a target, so "pour qi into the ground" and "read what it drank" compose.
2. **`spawn` introduces its id into scope** — a later effect in the *same* program can
   target the just-summoned entity (summon-and-feed atomically).
3. **Pooled fluid is a readable cell field** — reading a cell's `"oil"` returns the
   `add_fluid` amount (mirrors the existing `temperature`→heat bridge), so fire chains
   can gate on "is there oil here to light?".
4. **Free-form effect dialects are absorbed offline** (in the data tool, not the core):
   op-as-key `{"clear_tag":{…}}`, `adjust_edge` with `{"edge":[a,b,kind]}` or
   `from/to/prop/amount`, and `ref`/`target` where `t` was meant — canonicalized, then
   the sim filters by execution.
