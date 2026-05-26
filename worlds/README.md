# World Packs

A world pack is a folder of human-editable files that describes a complete
world to the simulation engine. The engine itself ships no specific
entities, factions, lore, or affordances — every concrete fact about a
particular world lives in its pack. Authoring a new setting is a matter
of editing YAML and a text-mode map; no Python is required.

## Loading a Pack

```python
from src.sim.world_loader import load_world_pack

world = load_world_pack("worlds/my_pack")
```

The default pack lives at `worlds/default/`. The REPL and the test
suite both call `make_test_world()` which loads it.

## Pack Layout

```
worlds/<name>/
  world.yaml          # required — metadata + grid dimensions + clock
  map.txt             # required — ASCII terrain grid
  entities.yaml       # required — entities and starting state
  objects.yaml        # optional — loose items (default: none)
  factions.yaml       # optional — groups / orgs / locations
  relations.yaml      # optional — initial graph edges
  lore.yaml           # optional — mythos concepts (deities, magic, …)
  affordances.yaml    # optional — intent-keyword anchors for grounding
  npc_policy.yaml     # optional — tunable weights for ReactivePolicy
  verb_templates.yaml # optional — default effects for non-kernel verbs
  ambient_events.yaml # optional — autonomic environmental events
  goals.yaml          # optional — completable objectives + rewards
  pressures.yaml      # optional — latent triggers (edge/tick/memory → bias)
  scenarios/          # optional — tick beats for NarrativeDirector
    *.yaml
  learned/            # written by engine — NEVER auto-loaded
    *.yaml            # candidate rules captured from LM adjudications;
                      # a human curator promotes vetted entries into the
                      # canonical files above (see "Learning Rules from Play")
```

### Learning Rules from Play

When ``world.yaml`` sets:

```yaml
extra:
  learn_from_adjudication: true
```

then every successful LM-adjudicated player action that produced
mechanical effects (transition proposals or facts, not just narration)
is appended to ``<pack>/learned/<UTC-date>.yaml`` in the same shape as
``verb_templates.yaml``.  The engine **never auto-loads** this folder
— treat it as a staging area.  A curator (or a second LM pass with a
strict schema) can paste promoted entries into the canonical
``verb_templates.yaml`` / ``open_verbs.yaml`` / ``property_interactions.yaml``
files at which point the engine handles those intents deterministically
without an LM call.  This is how creative one-offs become reusable
mechanical depth, in the spirit of Caves of Qud's tag rules and Dwarf
Fortress's job recipes.

Per-process deduplication ensures the same (verb, proposal-kinds,
fact-tags) hash is recorded at most once per run, so a session that
repeats an action does not bloat the staging file.

Link a scenario in `world.yaml`:

```yaml
extra:
  scenario: scenarios/my_night1.yaml
```

Every YAML file is a single mapping at the top level. Local identifiers
(strings of your choosing) are used for cross-references between files;
the loader assigns real engine-level UUIDs internally.

## File Reference

### `world.yaml`

```yaml
name: "Display Name"
description: |
  Free text. Designer notes only — engine does not read this.
spatial:
  width: 20
  height: 20
  map_file: "map.txt"     # path relative to the pack directory
meta:                      # passed to WorldState.meta verbatim;
  location_name: "tavern"  # the narrator/projection may use these as
  noise_level: "quiet"     # ambient context.
clock:                     # optional. Controls the world's day cycle.
  day_length_ticks: 40     # ticks per full DAWN→NIGHT cycle (default 100)
rng_seed: 17               # optional. Seeds deterministic ambient-event
                           # probability rolls. Same seed → same timeline.
```

### `map.txt`

ASCII grid. Anything above the first full-width line of grid characters
is treated as a comment header. Each character inside the grid maps to
a tile type:

| Char | Tile           |
| ---- | -------------- |
| `.`  | floor          |
| `#`  | wall           |
| `+`  | door, closed   |
| `/`  | door, open     |
| `>`  | stairs down    |
| `<`  | stairs up      |
| `~`  | water          |
| `=`  | window         |
| any other character | floor (designer can use letters for landmarks) |

The grid must be exactly `width` characters wide and `height` rows tall.

### `entities.yaml`

```yaml
entities:
  - id: player           # local id; relations.yaml references this
    name: "You"
    kind: player          # player | npc | creature | ambient
    position: [2, 2]      # [x, y] on the grid
    armed: false
    alertness: unaware    # unaware | low | medium | high | combat
    emotional_state: neutral
                          # neutral | happy | angry | fearful | suspicious
                          # | friendly | hostile | grieving | proud | humiliated
    health: 100
    max_health: 100
    sight_range: 8
    attributes:           # arbitrary numeric attributes; used by Compiler
      strength: 50
      perception: 60
    tags: []
    inventory: []         # list of object local-ids (from objects.yaml)
    social_openness: guarded
                          # welcoming | open | guarded | closed
                          # default disposition toward stranger contact

    # The next three fields are read by the LM-driven NPC policy
    # (see src/sim/npc_policy.py::LMNpcPolicy). They are passed
    # verbatim into the NPC's character-sheet prompt. The engine never
    # branches on their contents.
    role: "city watch sergeant"
    personality: "stoic, professional, suspicious of strangers"
    drive: "watch this corridor; challenge anyone who approaches"
```

The ``drive`` field is also consumed by the **long-horizon plan layer**
(``src/sim/npc_planner.py``).  When the drive text matches a known
pattern, the engine synthesises a multi-tick ``NpcPlan`` (DF-style job
stack) that the NPC commits to across many decision cycles.  Recognised
patterns out of the box:

| Pattern (regex, case-insensitive) | Plan template |
| --- | --- |
| ``patrol`` / ``walk`` / ``pace`` | Cycle through a waypoint route (uses ``waypoints`` if set, otherwise auto-generates a small loop) |
| ``tend`` / ``watch`` / ``guard`` / ``mind`` / ``oversee`` | Move to a named target (e.g. "tend Mira"), spend several ticks tending, then observe |
| ``deliver`` / ``bring`` / ``carry`` ... ``to`` | Path to the named recipient and speak a delivery line |
| ``work`` / ``labor`` / ``craft`` / ``build`` / ``repair`` etc. | Stay at workplace, spend several ticks labouring |

Plans are *interrupted* (paused) automatically when the NPC enters
combat, drops below 25% health, or gains the ``fleeing`` condition,
and *resume* once the NPC is back above 50% health and out of combat.
Unknown drives fall back to the reactive baseline — the same behaviour
as before plans existed.

The plan layer requires **no LM call**, so it works in
``--cognition reactive_only`` runs and gives the deterministic
simulation a "things are happening on their own" feel even when the
language model is disabled entirely.

Exactly one entity should usually have `kind: player`.

The PLAYER entity should also have the player-facing UI in mind — the
`role`/`personality`/`drive` fields are ignored for the player and only
read for NPCs that get their actions from the LM.

### `objects.yaml`

```yaml
objects:
  - id: throwing_knife    # local id, referenced from entities.yaml inventory
    name: "throwing knife"
    passable: true
    owner: player         # optional — entity local-id
    position: [4, 5]      # optional — for items on the ground
    tags: [weapon, throwable]
```

### `factions.yaml`

Group-level nodes added to the relational graph.

```yaml
factions:
  - id: city_watch
    name: "City Watch"
    kind: faction         # faction | organization | location
    tags: []              # written to node.meta.tags; affordances may key on this
    meta: {}              # arbitrary free-form data
```

### `relations.yaml`

Initial directed edges between any two nodes (entity, faction, or lore
concept). Refer to local ids declared in any of `entities.yaml`,
`factions.yaml`, or `lore.yaml`.

```yaml
edges:
  - source: guard
    target: city_watch
    kind: faction_member
    weight: 1.0
    meta: {}
```

Valid edge kinds: `faction_member`, `enemy_of`, `ally_of`, `fears`,
`respects`, `distrusts`, `owes_debt`, `remembers_event`, `witnessed`,
`kin`, `employs`, `employed_by`, `believes_claim`.

### `lore.yaml`

Mythos concepts: deities, magic systems, philosophical doctrines —
anything that exists conceptually rather than physically. Each entry
becomes a node in the relational graph just like an entity or a
faction.

```yaml
concepts:
  - id: solra
    name: "Solra, Goddess of Dawn"
    kind: faction         # use existing NodeKind values; tags carry the type
    tags: [deity, divine]
    meta:
      domain: "dawn, renewal, oaths"
```

If you want `"pray to Solra"` to ground here (Zone 1 instead of Zone 3),
ensure your `affordances.yaml` has a rule whose `tags` overlap with the
concept's tags.

### `affordances.yaml`

Tells the grounding classifier which intent keywords are anchored in
this world. A node satisfies a rule when:

```
node.kind in rule.kinds
AND ( any name_keyword appears in node.name
      OR any rule.tag appears in node.meta.tags )
```

```yaml
affordances:
  - keyword: pray
    kinds: [faction, organization]
    name_keywords: [god, deity, divine, holy, temple]
    tags: [deity, divine, religious]
```

If a rule's keyword appears in the player's intent but no node
satisfies it, the action goes to Zone 3 (ORPHANED) and the narrator
produces atmospheric prose rather than mutating state.

The engine ships **no** built-in affordance rules. A pack that omits
`affordances.yaml` cannot ground any symbolic intents — all prayer,
magic, ritual, etc. fall through to Zone 3.

### `npc_policy.yaml`

Tunable weights for the default `ReactivePolicy`.

```yaml
threat_lookback_ticks: 5     # how many ticks of event log to scan for
                             # recent attackers when deciding to retaliate
flee_health_fraction: 0.25   # health/max_health below this → prefer FLEE
```

### `verb_templates.yaml`

Default state-change effects for verbs that are **not** in the engine's
kernel. The kernel (move / take / give / attack / throw) has dedicated
physics compilers and is unaffected by templates. Every other verb —
whether conventional (`intimidate`, `console`, `observe`, …) or
freeform (`bake`, `serenade`, `weave_story`, anything) — is resolved
by a generic compiler that consults, in order:

1. A matching template in this file (if present).
2. A built-in default for conventional verbs (if no template).
3. A pure-freeform path that simply records the verb (if neither).

LM-proposed effects on the action are **always** layered on top of
whichever layer fired, and every effect — pack-defined or LM-proposed —
goes through the same reachability validator (unknown entities dropped,
out-of-bounds values clipped, kernel-only kinds rejected).

```yaml
templates:
  - verb: console
    # Optional contest: if present, the engine runs an attribute roll
    # before picking effects_on_success vs effects_on_failure.
    contest:
      actor_attribute: persuasion
      target_attribute: composure
      difficulty: 50          # used only if target_attribute is absent
    on_success:
      - kind: entity_emotional_state_changed
        payload:
          entity_id: "$target"  # $actor / $target substitute at compile time
          to: friendly
    on_failure:
      - kind: entity_emotional_state_changed
        payload:
          entity_id: "$target"
          to: suspicious
```

Allowed TransitionKinds in a template's effect payloads:

| Kind | Notes |
| --- | --- |
| `entity_emotional_state_changed` | `entity_id` must reference a real entity; `to` must be a valid emotional state |
| `entity_alertness_changed` | same shape; `to` must be a valid alertness level |
| `edge_created` / `edge_updated` / `edge_removed` | `source` and `target` must be known nodes; `weight` clipped to [-1, 1] |
| `entity_health_changed` | `delta` clipped to ±10 for generic verbs |
| `dialogue_spoken` | free-form text, used by narrator |

`entity_moved` and `item_transferred` are **kernel-only** and will be
dropped if proposed — use the `move`, `take`, or `give` verbs instead.

### `ambient_events.yaml`

Pack-declared events that the world clock fires between actor turns.
There is no actor — these are environmental things that "just happen"
(a torch flickers, dawn breaks, a bell tolls in the distance). Every
fired event is appended to the canonical event log under a synthetic
`actor = "system"` and shows up in NPC and player projections, so
characters can react to them in subsequent turns.

```yaml
events:
  - id: torch_flicker
    narrative: "A torch sputters, casting jumping shadows."
    trigger:
      every_ticks: 6         # fire only on ticks where tick % 6 == 0
      time_of_day: night     # optional; fire only during this phase
      probability: 0.7       # optional; pass with this probability
    effects: []              # optional; same shape as verb-template effects
```

Trigger gates are AND-combined. The probability roll is deterministic —
seeded from `(world.rng_seed, event.id, tick)` — so replaying the same
log on a fresh world produces the same ambient timeline.

The `effects` list goes through the same proposal validator as
`verb_templates.yaml` (unknown entities dropped, deltas clipped,
kernel-only kinds rejected). Most ambient events leave `effects`
empty and rely on the narrative alone — the world ticks because
something is *perceived*, not because state changes.

## Authoring Workflow

1. Copy `worlds/default/` to `worlds/my_pack/`.
2. Edit `world.yaml` for size and ambient meta.
3. Redraw `map.txt`.
4. Replace entries in `entities.yaml`, `factions.yaml`, `relations.yaml`.
5. If your setting involves prayer, magic, rituals, etc., declare the
   relevant concepts in `lore.yaml` and the matching anchor rules in
   `affordances.yaml`.
6. Load it: `make_test_world("worlds/my_pack")` or directly via
   `load_world_pack("worlds/my_pack")`.

No engine code changes are required.

## What Lives in Code vs. What Lives in Data

| Layer | Where it lives | Examples |
| --- | --- | --- |
| Schemas, simulation rules | `src/sim/` | spatial math, compiler validation, event log, projection firewall |
| World-specific content | `worlds/<pack>/*.yaml` | who exists, where, in which faction |
| NPC self-description | `worlds/<pack>/entities.yaml` (role/personality/drive) | what the LM-driven NPC policy hands to the model |
| World-specific lore | `worlds/<pack>/lore.yaml` | deities, magic systems, doctrines |
| Intent anchoring | `worlds/<pack>/affordances.yaml` | which keywords are meaningful here |
| Verb effects | `worlds/<pack>/verb_templates.yaml` | what happens when someone consoles / haggles / serenades / etc. |
| Ambient world events | `worlds/<pack>/ambient_events.yaml` | torch flickers, dawn breaks, bell tolls |
| World clock | `worlds/<pack>/world.yaml` (clock block) | how many ticks make a day |
| NPC tuning | `worlds/<pack>/npc_policy.yaml` | how cautious or aggressive NPCs are |
| Custom NPC behavior types | `src/sim/npc_policy.py` (Python) | new `NpcPolicy` implementations |
| LM prompts / adapters | `src/sim/lm_adapter.py` (Python) | model integration, NPC system prompt |

If a future change requires editing engine code to express something
that should be world-specific, that is a signal to push more behavior
into the pack format instead.
