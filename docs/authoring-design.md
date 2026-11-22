# Authoring: content packs and engine mods

Scoped with the owner on 2026-11-22 (decisions AU1-AU5). The owner: interfaces to the engine must be GENERIC and MUTABLE, and reasonably easy to edit for humans and LLM agents, code included. There must be a comprehensive way of modding the core engine for additional content.

## AU1. Two tiers

- CONTENT PACKS are DATA: validated, plain-text YAML files for creatures, anomalies, materials, cultures, spells and entities.
  - A data file cannot break a law, and a mistake is caught by the validator.
  - An entity can only be EXPRESSED through existing laws. What cannot be written is a missing law, and it is logged.
- ENGINE MODS are CODE: new general laws in small kernel modules, each with sources, tests and switch-ledger rows, under the contract of AU2.

## AU2. Engine mods: a law-plugin interface with a conformance suite

- A mod DECLARES:
  - what it adds: new quantities, ledger rows, laws that run each tick, and the gates they live behind;
  - what it may TOUCH (for example energy and momentum only, like magic's narrow kernel).
- It must PASS the standard conformance suite before it loads in a world:
  - the energy and mass books still balance;
  - runs stay deterministic;
  - with the mod off, the world is bit for bit unchanged;
  - its cost is measured against a ceiling;
  - its sources are cited.
- Mods load PER WORLD.
- The suite is the existing wave discipline made official: golden diffs, ledger checks, determinism, cost ceilings, the switch ledger.

## AU3. The entity file: one schema of optional components, extensible

| Component | What it describes |
|---|---|
| form | matter and shape, or a body sheet |
| senses | what it perceives, with dials |
| needs and drives | what it wants |
| mind | none, rules, or a learned mind with its dials |
| patterns and stores | magical structures and the energy held |
| triggers | conditions that cause effects, always through existing laws |
| links and bonds | connections to other things |
| seeded knowledge | the rumours and cultural beliefs people start with about it |
| expectations | testable claims ("touching it at night drains heat"), checked by running a small scenario |

- Include only what applies:
  - a ghost is a pattern, a mind and seeded rumours, with no form;
  - a cursed mirror is a form, a pattern, a trigger and rumours;
  - a wolf is a form, senses, drives and a mind.
- Things that fit no category are fine.
- EXTENSIBLE: a new component type is registered by an ENGINE MOD, which supplies its schema (for the validator), its processor (the law code, under the conformance suite) and its tests.
- An UNKNOWN component is a validator ERROR unless a loaded mod provides it. It is never silently ignored.
- It builds on the existing entity layer: the things registry (src/voxel/things.py), persons, body sheets, the species registry, material rows (matrec) and the verb catalogue (src/mind/data/verbs.yaml).

## AU4. Tools (text first; a graphical editor later, as an optional front end)

- `check`: validates a pack file, with errors that name the field and the rule.
- `try`: runs the file's expectations in a small scenario and shows what happened, in prose and numbers.
- `explain`: every component and field, with examples, GENERATED FROM THE CODE so the docs never drift.
- Authoring guides per component, with worked examples from the coverage test cases (docs/coverage-uncanny.md).
- A missing-law report: when an expectation fails because no law can produce it, it says so, for the missing-law log.
- Files always stay plain text (the owner's text-only rule).

## AU5. When: in stages, with the layers

1. WITH WAVE 16a:
   - the schema, `check`, `explain`, and the mod interface with its conformance suite;
   - existing content moved onto it (materials, verbs, species, body sheets) as the test of the format;
   - the components form, senses, needs and drives, mind and triggers.
2. WITH WAVE 16b: seeded knowledge, expectations and `try`.
3. WITH MAGIC: patterns, stores, links and bonds.

The coverage pass's representative test cases are the checklist for each stage.

Licensing: MARK1 invents its own entities, inspired by public sources but never copied (SCP is CC BY-SA, some creepypasta characters are owned; magic decision MC8).
