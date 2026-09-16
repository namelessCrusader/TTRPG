# References — local copies

Fetched 2026-08-29. Each entry says **what it's for** (which decision/todo in
[docs/data-engine-design.md](../docs/data-engine-design.md)) and **the mistake it prevents**.
Consult the relevant file BEFORE building the matching engine item.

## engine-docs/

| file | serves | the mistake it prevents |
|---|---|---|
| `dda-JSON_FLAGS.md` | item 2 (flag system) | Flags are **namespaced by domain** (item/terrain/monster/ammo) — don't dump every flag in one bag; a flag's meaning is domain-relative |
| `dda-JSON_INFO.md` | item 2 (object schema) | The full object-definition schema DDA converged on after a decade — check it before inventing a field |
| `dda-EFFECTS_JSON.md` | item 4 (tag registry, D7) | A status effect declares **duration, intensity, decay, and what reads it** — a status is a *contract*, not a bare tag |
| `dda-materials.json` | item 2b (material templates) | A production material table: what per-material properties actually earn their keep (density, chip resistance, fire, thaws-into) |
| `df-material_definition_token.json` / `df-material_token.json` (wikitext) | item 2b | DF's `USE_MATERIAL_TEMPLATE`: broad class → specific override. Also the **DF lesson** (risks §8): DF models more than the game reads — don't add a property nothing consumes |
| `qud-modding-parts.json` / `qud-modding-liquids.json` (wikitext) | item 2c (component composition) | Parts advertise behavior + compose (LiquidVolume + MeleeWeapon = a weapon that holds liquid) — our `parts.py` follows this shape; check before adding a part kind |
| `ss13-Atmospherics.md` | risks (theater principle) | **"Theater, not simulator"**: sleep wherever possible, simulate as little as you can get away with. Fidelity nobody perceives is dead weight — applies to every physics channel |
| `dcss-crawl_manual.rst` | item 6 (non-dominated payoffs) | DCSS design philosophy: no illusory choices where one option is always superior — the actual mechanism behind `brood`'s 0%→30% |
| `brogue-README.md` / `powdertoy-README.md` | reference index pointers | Brogue = smallest readable emergence codebase; Powder Toy = a material×material reaction matrix in production (coupling graph, item 3) |

## if-social/

| file | serves | the mistake it prevents |
|---|---|---|
| `inform-std-physical-world-model.w` | item 2 (property vocabulary) | **The goldmine**: 40 years of IF's everyday ontology as source — containers/supporters, open/closed, lockable, edible, wearable, lit/dark, fixed-in-place, reach. Check here before naming a flag; these names survived four decades of players |
| `inform-std-actions.w` | propose channel (D4) + verbs | The canonical verb set players expect a world to answer, each with preconditions ("check") and effects ("carry out") — a menu-macro catalogue with rules attached |
| `inform-std-command-grammar.w` | propose-channel NL mapper | How free text maps to actions in the most battle-tested parser there is |
| `ensemble-README.md` | open decision 6 (social macros) | Volition rules over social state pick what characters *want* — derive social-macro **availability** from state instead of enumerating macros |
| `neighborly-README.md` / `talktown-README.md` | belief-fold extensions | Character knowledge, belief propagation, gossip — where `perceive()` goes when NPCs *share* beliefs (talktown's README is genuinely minimal; the code is the reference) |
| `jericho-README.md` | propose-channel training data | Handles for human free-text commands to parser games (ClubFloyd transcripts) |

## papers/

| file | serves |
|---|---|
| `pointer-networks.pdf` · `drrn.pdf` · `large-discrete-action-spaces.pdf` | D9 — the bi-encoder over candidate sets (score content, not action ids) |
| `grammar-vae.pdf` · `sd-vae.pdf` · `abstract-syntax-networks.pdf` · `tranx.pdf` | Model spec §5 — grammar-constrained AST derivation; **SD-VAE is the closest to our legality-vs-sensibility split** (semantic constraints in the decoder mask) |
| `treelstm.pdf` · `code2seq.pdf` | Encoding programs/trees for the student model |
| `map-elites.pdf` | open decision 4 — QD search over trajectories (sidesteps "which trajectory is better") |
| `dungeons-and-data.pdf` | Precedent: sourcing policy data without a good author (NetHack corpus) |
| `balrog.pdf` | Calibrate big-model-as-policy on games before committing to distillation |
| `dcss-eval-domain.pdf` | DCSS as an AI evaluation domain |
| `versu-evans-short.pdf` | Social practices framing (open decision 6; zone-graph substrate, open decision 8) |

## Not fetched (and why)

- **TADS 3 adv3 sense model** — no single canonical raw file; fetch the adv3 library source when
  building the sense-model extension of `perceive()`.
- **BotW chemistry engine GDC talk** — video only; the §4-item-3 row summarizes the principle
  (elements × material properties, interactions as edges).
- **Kismet / Comme il Faut papers** — fetch when open decision 6 is actually being settled.
- **Full repos (Brogue/DCSS/DDA/Qud/SS13 source)** — too large to vendor; READMEs/docs here give
  the entry points, clone shallow on demand.
