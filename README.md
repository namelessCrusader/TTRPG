# Mark-1 — coherent emergent sim

A small, deterministic, emergent world sim with an optional local LM for social
interaction. A Dwarf-Fortress-style engine where the state can never go
incoherent, because **every mutation is one typed delta on one append-only log**.

## Design

- **One effect algebra.** ~20 fine deltas (`set_tag`, `add_heat`, `add_fluid`,
  `adjust_prop`, `move`, …), one `apply()`, one log. A "verb" is just a list of deltas.
- **Unified spatial substrate.** One cell type holds heat + fluids + tags + a
  material block; entities are occupants. Positions are tuples (2D now, 3D later
  is a coordinate change).
- **Reactions are data** (`src/core/data/reactions.yaml`) — tag/property rules run
  over every cell/entity each tick. Fire, acid, cellular-automaton fluid flow,
  water↔acid, structural collapse — all emergent, none authored pair-by-pair.
- **Menu = affordances.** Actions are derived from what's in reach × its tags.
- **The LM never mutates state.** It picks from bounded reactions (decision model)
  and optionally generates the line (voice model); the sanitizer + bus keep the
  world coherent no matter what the model says.

## Run

```bash
python -m src.core web                       # browser UI, deterministic brains
python -m src.core web --brain torch --voice reyna   # + local GPU models
python -m src.core play                      # terminal
python -m src.core auto --turns 2000         # random-bot coherence fuzz
```

Optional local models (see `requirements.txt`): decision = `Qwen/Qwen3-0.6B`,
voice = `aloobun/Reyna-RP-Qwen1.5-0.5B-Chat` — both sub-1B, ~ms/turn on 6 GB VRAM.
