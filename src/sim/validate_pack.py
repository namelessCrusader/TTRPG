"""
World pack validator — references, scenarios, pressures, smoke simulation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from .game_loop import GameLoop
from .lm_adapter import MockLMAdapter
from .npc_policy import ReactivePolicy
from .pressure_eval import _resolve_entity_ref
from .world_loader import WorldPackError, load_world_pack

# All shipped world packs (relative to repo root).
KNOWN_PACKS = (
    "default",
    "tavern",
    "spicy",
    "castle",
    "magic_duel",
    "voxel_tavern",
)


def _entity_lookup(world) -> dict[str, str]:
    """Map lowercase name / pack id → canonical entity id string."""
    out: dict[str, str] = {}
    for eid, ent in world.all_entities().items():
        out[str(eid).lower()] = str(eid)
        out[ent.name.lower()] = str(eid)
        # pack yaml ids like "mira" often match entity_id suffix
        slug = str(eid).split("_")[-1].lower()
        if slug:
            out[slug] = str(eid)
    return out


def _resolve_ref(ref: Any, lookup: dict[str, str], world) -> str | None:
    if ref is None:
        return None
    resolved = _resolve_entity_ref(world, ref)
    if resolved:
        return resolved
    key = str(ref).lower()
    return lookup.get(key)


def _ambient_ids(world) -> set[str]:
    return {str(a.id) for a in world.config.ambient_events}


def _validate_goals(world, lookup: dict[str, str]) -> list[str]:
    errors: list[str] = []
    entity_ids = {str(eid) for eid in world.all_entities()}
    for goal in world.config.goals:
        for cond in goal.conditions:
            for field_name in ("source", "target", "entity", "actor"):
                ref = getattr(cond, field_name, None)
                if not ref:
                    continue
                resolved = _resolve_ref(ref, lookup, world)
                if resolved and resolved in entity_ids:
                    continue
                if str(ref).startswith("ent_") and str(ref) not in entity_ids:
                    errors.append(
                        f"Goal '{goal.id}' condition references unknown entity '{ref}'"
                    )
    return errors


def _validate_pressures(world, lookup: dict[str, str], ambient_ids: set[str]) -> list[str]:
    errors: list[str] = []
    for rule in world.config.pressures or []:
        rid = str(rule.get("id", ""))
        effects = rule.get("effects") or {}
        for boost in effects.get("candidate_boosts") or []:
            eid = _resolve_ref(boost.get("entity"), lookup, world)
            if boost.get("entity") and not eid:
                errors.append(
                    f"Pressure '{rid}' candidate_boosts unknown entity "
                    f"'{boost.get('entity')}'"
                )
        for amb_id, _mult in (effects.get("ambient_probability_mult") or {}).items():
            if str(amb_id) not in ambient_ids:
                errors.append(
                    f"Pressure '{rid}' ambient_probability_mult unknown id '{amb_id}'"
                )
    return errors


def _validate_reactive_triggers(
    world, lookup: dict[str, str], ambient_ids: set[str]
) -> list[str]:
    errors: list[str] = []
    for rule in world.config.reactive_triggers or []:
        rid = str(rule.get("id", ""))
        if not rid:
            errors.append("Reactive trigger missing 'id'")
            continue
        effects = rule.get("then") or rule.get("effects") or {}
        if not isinstance(effects, dict):
            continue
        for spec in effects.get("inject_goals") or []:
            if spec.get("entity") and not _resolve_ref(spec.get("entity"), lookup, world):
                errors.append(
                    f"Reactive trigger '{rid}' inject_goals unknown entity "
                    f"'{spec.get('entity')}'"
                )
        amb = effects.get("force_ambient")
        amb_ids = [amb] if isinstance(amb, str) else list(amb or [])
        for amb_id in amb_ids:
            if str(amb_id) not in ambient_ids:
                errors.append(
                    f"Reactive trigger '{rid}' force_ambient unknown id '{amb_id}'"
                )
    return errors


def _validate_scenario(world, lookup: dict[str, str], ambient_ids: set[str]) -> list[str]:
    errors: list[str] = []
    scenario = (world.config.extra or {}).get("scenario") or {}
    if not scenario:
        return errors
    if not scenario.get("id"):
        errors.append("Scenario missing 'id'")
    beats = scenario.get("beats") or []
    if not beats:
        errors.append(f"Scenario '{scenario.get('id')}' has no beats")
    for i, beat in enumerate(beats):
        tick = beat.get("tick")
        if tick is None:
            errors.append(f"Scenario beat {i} missing 'tick'")
        effects = beat.get("effects") or {}
        if not isinstance(effects, dict):
            continue
        for spec in effects.get("inject_goals") or []:
            if spec.get("entity") and not _resolve_ref(spec.get("entity"), lookup, world):
                errors.append(
                    f"Scenario beat tick={tick} inject_goals unknown entity "
                    f"'{spec.get('entity')}'"
                )
        amb = effects.get("force_ambient")
        amb_ids = [amb] if isinstance(amb, str) else list(amb or [])
        for amb_id in amb_ids:
            if str(amb_id) not in ambient_ids:
                errors.append(
                    f"Scenario beat tick={tick} force_ambient unknown id '{amb_id}'"
                )
    return errors


def _validate_portals(world) -> list[str]:
    errors: list[str] = []
    if not world.regions:
        return errors
    region_ids = set(world.regions.keys())
    for portal in world.portals or []:
        pid = getattr(portal, "portal_id", None) or getattr(portal, "id", "?")
        for attr in ("from_region", "to_region"):
            rid = getattr(portal, attr, None)
            if rid and rid not in region_ids:
                errors.append(
                    f"Portal '{pid}' {attr} '{rid}' not in regions {sorted(region_ids)}"
                )
    return errors


def _validate_verb_templates(world) -> list[str]:
    errors: list[str] = []
    for verb, template in world.config.verb_templates.items():
        if not str(verb).strip():
            errors.append("Empty verb template key")
        if template is None:
            errors.append(f"Verb template '{verb}' is null")
    return errors


def _validate_entity_names(world) -> list[str]:
    errors: list[str] = []
    seen: dict[str, str] = {}
    for eid, ent in world.all_entities().items():
        name = ent.name.strip()
        if not name:
            errors.append(f"Entity {eid} has empty name")
            continue
        key = name.lower()
        if key in seen and seen[key] != str(eid):
            errors.append(
                f"Duplicate entity name '{name}' ({seen[key]} vs {eid})"
            )
        seen[key] = str(eid)
    return errors


def validate_pack(pack_path: Path, *, smoke_ticks: int = 20) -> list[str]:
    """
    Validate a world pack. Returns list of error strings (empty = OK).
    """
    errors: list[str] = []
    pack_path = pack_path.resolve()

    try:
        world = load_world_pack(pack_path)
    except WorldPackError as exc:
        return [str(exc)]
    except Exception as exc:
        return [f"Failed to load pack: {exc}"]

    lookup = _entity_lookup(world)
    ambient_ids = _ambient_ids(world)

    errors.extend(_validate_entity_names(world))
    errors.extend(_validate_goals(world, lookup))
    errors.extend(_validate_pressures(world, lookup, ambient_ids))
    errors.extend(_validate_reactive_triggers(world, lookup, ambient_ids))
    errors.extend(_validate_scenario(world, lookup, ambient_ids))
    errors.extend(_validate_portals(world))
    errors.extend(_validate_verb_templates(world))

    if smoke_ticks > 0:
        try:
            loop = GameLoop(
                world,
                adapter=MockLMAdapter(),
                npc_policy=ReactivePolicy(),
            )
            for _ in range(smoke_ticks):
                loop.autonomous_tick()
        except Exception as exc:
            errors.append(f"Smoke run failed at tick {world.tick}: {exc}")

    return errors


def validate_all_packs(
    worlds_root: Path,
    *,
    smoke_ticks: int = 10,
) -> dict[str, list[str]]:
    """Validate every known pack under ``worlds_root``."""
    results: dict[str, list[str]] = {}
    for name in KNOWN_PACKS:
        pack = worlds_root / name
        if pack.is_dir():
            results[name] = validate_pack(pack, smoke_ticks=smoke_ticks)
    return results


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print("Usage: python -m src.sim.validate_pack worlds/tavern [--all]", file=sys.stderr)
        return 1

    if args[0] in ("--all", "-a"):
        root = Path(__file__).resolve().parents[2] / "worlds"
        smoke = 10
        if len(args) > 1 and args[1].isdigit():
            smoke = int(args[1])
        results = validate_all_packs(root, smoke_ticks=smoke)
        failed = False
        for name, errs in sorted(results.items()):
            if errs:
                failed = True
                print(f"FAIL {name}:")
                for e in errs:
                    print(f"  ERROR: {e}")
            else:
                print(f"OK   {name}")
        return 1 if failed else 0

    pack = Path(args[0])
    smoke = 20
    if len(args) > 1 and args[1].isdigit():
        smoke = int(args[1])
    errs = validate_pack(pack, smoke_ticks=smoke)
    if errs:
        for e in errs:
            print(f"ERROR: {e}")
        return 1
    print(f"OK: {pack}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
