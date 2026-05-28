"""
World pack loader.

A "world pack" is a directory of human-editable files describing a
complete world: terrain, entities, factions, relations, lore, soft
rules, and policy weights. The engine itself has no knowledge of any
specific world — every concrete fact about who exists, what they
worship, or how they react to threats comes from a pack.

Pack layout:

    <pack_dir>/
      world.yaml          # name, dimensions, ambient meta, npc_policy
      map.txt             # ASCII terrain grid
      entities.yaml       # players, NPCs, attributes, factions
      objects.yaml        # items
      factions.yaml       # group nodes (faction / organization / location)
      relations.yaml      # initial edges
      lore.yaml           # mythos concepts (deities, magic systems, ...)
      affordances.yaml    # intent-keyword → required-node rules
      npc_policy.yaml     # tunable weights for ReactivePolicy

Authoring a new world is purely a matter of editing these files. No
Python changes are required.

The loader assigns engine-internal UUIDs to every pack-local id. Other
files (relations.yaml, entities.yaml inventories, etc.) reference items
by their pack-local id, never by UUID.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

import yaml

from .relational import add_edge, add_node
from .schemas import (
    AffordanceRule,
    ObjectAffordanceRule,
    AlertnessLevel,
    AmbientEvent,
    AmbientTrigger,
    ContestSpec,
    Coord,
    EdgeKind,
    EmotionalState,
    EntityId,
    EntityKind,
    EntityState,
    EquipSlot,
    FacingDirection,
    FactionId,
    NodeKind,
    NpcPolicyConfig,
    OpenVerbRule,
    ObjectId,
    ObjectState,
    Portal,
    Region,
    RegionType,
    RelationalGraph,
    SpatialGrid,
    Tile,
    TerrainType,
    Goal,
    GoalCondition,
    GoalEdgeUpdate,
    GoalReward,
    TimeOfDay,
    TransitionProposal,
    VerbTemplate,
    ConsequencePolicy,
    WorldClock,
    WorldConfig,
    WorldState,
    new_entity_id,
    new_object_id,
)
from .spatial import place_entity

logger = logging.getLogger(__name__)

# Shared baseline pack — verb templates and reactive triggers merge into
# every non-default world unless the world pack overrides the same verb/id.
_DEFAULT_PACK_DIR = Path(__file__).resolve().parent.parent.parent / "worlds" / "default"


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_extends_chain(pack: Path) -> list[Path]:
    """
    Return pack directories from child → parent (Overhaul C inheritance).

    ``world.yaml`` may declare ``extends: default`` or ``extends: ../default``.
    """
    chain: list[Path] = []
    current = pack.resolve()
    seen: set[Path] = set()
    while current not in seen:
        seen.add(current)
        chain.append(current)
        meta = _read_yaml(current / "world.yaml", required=False) or {}
        ext = meta.get("extends")
        if not ext:
            break
        parent = Path(str(ext))
        if not parent.is_absolute():
            candidate = (_REPO_ROOT / "worlds" / ext).resolve()
            if candidate.is_dir():
                parent = candidate
            else:
                parent = (current.parent / ext).resolve()
        if not parent.is_dir() or parent in seen:
            break
        current = parent
    return chain


def _merged_yaml_items(
    chain: list[Path],
    filename: str,
    list_key: str,
    id_key: str = "id",
) -> list[dict]:
    """Merge list items from pack chain; child entries override parent by id."""
    merged: dict[str, dict] = {}
    for pack_dir in reversed(chain):
        doc = _read_yaml(pack_dir / filename, required=False) or {}
        for raw in doc.get(list_key) or []:
            if not isinstance(raw, dict):
                continue
            iid = str(raw.get(id_key, "")).strip()
            if iid:
                merged[iid] = raw
    return list(merged.values())


def _merge_default_verb_templates(
    pack: Path,
    verb_templates: dict[str, "VerbTemplate"],
) -> None:
    if pack.resolve() == _DEFAULT_PACK_DIR.resolve():
        return
    doc = _read_yaml(_DEFAULT_PACK_DIR / "verb_templates.yaml", required=False) or {}
    for raw in doc.get("templates", []) or []:
        verb = raw.get("verb")
        if verb and verb not in verb_templates:
            verb_templates[verb] = _build_verb_template(str(verb), raw)


def _merge_default_reactive_triggers(
    pack: Path,
    triggers: list[dict],
) -> list[dict]:
    if pack.resolve() == _DEFAULT_PACK_DIR.resolve():
        return triggers
    doc = _read_yaml(_DEFAULT_PACK_DIR / "reactive_triggers.yaml", required=False) or {}
    merged: dict[str, dict] = {}
    for raw in doc.get("triggers") or []:
        tid = str(raw.get("id", "")).strip()
        if tid:
            merged[tid] = raw
    for raw in triggers:
        tid = str(raw.get("id", "")).strip()
        if tid:
            merged[tid] = raw
    return list(merged.values())


def _build_npc_plan_template(raw: dict) -> "NpcPlanTemplate":
    from .schemas import NpcPlanTemplate, PlanStep, PlanStepKind

    steps: list[PlanStep] = []
    for s in raw.get("steps") or []:
        if not isinstance(s, dict):
            continue
        kind_raw = str(s.get("kind", "")).lower().strip()
        payload = dict(s.get("payload") or {})
        tile = s.get("tile")
        if tile is not None and "tile" not in payload:
            payload["tile"] = tile
        ticks = s.get("ticks")
        if ticks is not None and "ticks" not in payload:
            payload["ticks"] = int(ticks)
        for key in ("verb", "target", "entity_id", "line"):
            if s.get(key) is not None and key not in payload:
                payload[key] = s[key]
        steps.append(
            PlanStep(
                kind=PlanStepKind(kind_raw),
                payload=payload,
                label=str(s.get("label") or ""),
            )
        )
    return NpcPlanTemplate(
        id=str(raw.get("id") or ""),
        drive_pattern=str(raw.get("drive_pattern") or raw.get("drive_match") or ""),
        entity_ids=[str(x) for x in (raw.get("entity_ids") or [])],
        entity_roles=[str(x) for x in (raw.get("entity_roles") or [])],
        name=str(raw.get("name") or raw.get("id") or ""),
        builder=str(raw.get("builder") or ""),
        steps=steps,
    )


def _load_npc_plans_chain(chain: list[Path]) -> list["NpcPlanTemplate"]:
    merged: dict[str, "NpcPlanTemplate"] = {}
    for pack_dir in reversed(chain):
        doc = _read_yaml(pack_dir / "npc_plans.yaml", required=False) or {}
        for raw in doc.get("plans") or []:
            if not isinstance(raw, dict):
                continue
            try:
                tmpl = _build_npc_plan_template(raw)
            except Exception as exc:
                logger.warning("npc_plans.yaml: skip entry %r (%s)", raw.get("id"), exc)
                continue
            key = tmpl.id or tmpl.drive_pattern
            if key:
                merged[key] = tmpl
    return list(merged.values())


def _merge_default_open_verbs(
    pack: Path,
    open_verbs: list["OpenVerbRule"],
) -> list["OpenVerbRule"]:
    if pack.resolve() == _DEFAULT_PACK_DIR.resolve():
        return open_verbs
    doc = _read_yaml(_DEFAULT_PACK_DIR / "open_verbs.yaml", required=False) or {}
    merged: dict[str, OpenVerbRule] = {r.id: r for r in open_verbs}
    for raw in doc.get("rules") or []:
        rule = _build_open_verb_rule(raw)
        if rule.id not in merged:
            merged[rule.id] = rule
    return list(merged.values())


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────


class WorldPackError(Exception):
    """Raised when a world pack cannot be loaded or fails validation."""


def load_world_pack(pack_dir: str | Path) -> WorldState:
    """
    Read a world pack directory and produce a fully populated WorldState.

    The returned world has:
      - terrain grid loaded from map.txt
      - entities placed at their declared positions (collision-checked)
      - objects placed in inventories or on the ground
      - relational graph populated with entities, factions, lore concepts,
        and the declared initial edges
      - config populated with affordance rules and NPC policy weights

    Raises
    ------
    WorldPackError
        If a required file is missing or a referenced local id does not
        resolve.
    """
    pack = Path(pack_dir)
    if not pack.is_dir():
        raise WorldPackError(f"World pack directory not found: {pack}")

    pack_chain = _resolve_extends_chain(pack)
    world_meta = _read_yaml(pack / "world.yaml", required=True)
    entities_doc = _read_yaml(pack / "entities.yaml", required=True)
    objects_doc = _read_yaml(pack / "objects.yaml", required=False) or {"objects": []}
    factions_doc = _read_yaml(pack / "factions.yaml", required=False) or {"factions": []}
    relations_doc = _read_yaml(pack / "relations.yaml", required=False) or {"edges": []}
    lore_doc = _read_yaml(pack / "lore.yaml", required=False) or {"concepts": []}
    affordances_doc = _read_yaml(pack / "affordances.yaml", required=False) or {"affordances": []}
    policy_doc = _read_yaml(pack / "npc_policy.yaml", required=False) or {}
    verb_templates_doc = (
        _read_yaml(pack / "verb_templates.yaml", required=False)
        or {"templates": []}
    )
    ambient_doc = (
        _read_yaml(pack / "ambient_events.yaml", required=False)
        or {"events": []}
    )
    goals_doc = (
        _read_yaml(pack / "goals.yaml", required=False)
        or {"goals": []}
    )
    pressures_doc = (
        _read_yaml(pack / "pressures.yaml", required=False)
        or {"pressures": []}
    )
    reactive_triggers_doc = (
        _read_yaml(pack / "reactive_triggers.yaml", required=False)
        or {"triggers": []}
    )
    open_verbs_doc = (
        _read_yaml(pack / "open_verbs.yaml", required=False)
        or {"rules": []}
    )

    # ── 1. Spatial grid ─────────────────────────────────────────────────────
    spatial = _load_spatial(pack, world_meta)

    # ── 2. Build entities (pack_id → real EntityId) ─────────────────────────
    entity_id_map: dict[str, EntityId] = {}
    entities: dict[str, EntityState] = {}
    entity_raw_list = list(entities_doc.get("entities", []) or [])
    for raw in entity_raw_list:
        pack_id = _require(raw, "id", "entities.yaml")
        entity_id_map[pack_id] = new_entity_id()
    for raw in entity_raw_list:
        pack_id = _require(raw, "id", "entities.yaml")
        eid = entity_id_map[pack_id]
        entities[pack_id] = _build_entity(eid, raw, entity_id_map)

    # ── 2b. Pre-register entity IDs from extra regions ──────────────────────
    # Relations are resolved in step 6 using entity_id_map; all region
    # entities must be registered BEFORE that step, even though their grids
    # are built later.
    # We store raw dicts keyed by region_id for later full build.
    _region_raw_entities: dict[str, list[dict]] = {}  # region_id -> raw entity list
    _region_raw_objects: dict[str, list[dict]] = {}   # region_id -> raw object list
    for reg_raw in world_meta.get("regions") or []:
        rid = reg_raw.get("id")
        if not rid:
            continue
        if reg_raw.get("entities_file"):
            reg_ent_doc = _read_yaml(pack / reg_raw["entities_file"], required=False) or {"entities": []}
            raw_ents = reg_ent_doc.get("entities") or []
            _region_raw_entities[rid] = raw_ents
            for raw_e in raw_ents:
                pack_id = _require(raw_e, "id", reg_raw["entities_file"])
                if pack_id not in entity_id_map:
                    entity_id_map[pack_id] = new_entity_id()
        if reg_raw.get("objects_file"):
            reg_obj_doc = _read_yaml(pack / reg_raw["objects_file"], required=False) or {"objects": []}
            raw_objs = reg_obj_doc.get("objects") or []
            _region_raw_objects[rid] = raw_objs
            # Object IDs registered after primary objects below

    # ── 3. Build objects ────────────────────────────────────────────────────
    object_id_map: dict[str, ObjectId] = {}
    objects: dict[str, ObjectState] = {}
    for raw in objects_doc.get("objects", []):
        oid = new_object_id()
        pack_id = _require(raw, "id", "objects.yaml")
        object_id_map[pack_id] = oid
        objects[pack_id] = _build_object(oid, raw, entity_id_map)

    # Pre-register object IDs from extra regions (needed for inventory/key refs)
    for rid, raw_objs in _region_raw_objects.items():
        for raw_o in raw_objs:
            pack_id = _require(raw_o, "id", f"region {rid} objects")
            if pack_id not in object_id_map:
                object_id_map[pack_id] = new_object_id()

    # ── 4. Resolve inventories + equipped items ─────────────────────────────
    for pack_id, entity in entities.items():
        raw = next(
            r for r in entities_doc["entities"] if r["id"] == pack_id
        )
        inv_pack_ids = raw.get("inventory", []) or []
        for inv_pack_id in inv_pack_ids:
            if inv_pack_id not in object_id_map:
                raise WorldPackError(
                    f"entity '{pack_id}' inventory references unknown "
                    f"object '{inv_pack_id}'."
                )
            entity.inventory.append(object_id_map[inv_pack_id])

        # Equipped weapon (legacy field — also populates equipped_slots)
        ew_pack_id = raw.get("equipped_weapon")
        if ew_pack_id:
            if ew_pack_id not in object_id_map:
                raise WorldPackError(
                    f"entity '{pack_id}' equipped_weapon references unknown "
                    f"object '{ew_pack_id}'."
                )
            ew_oid = object_id_map[ew_pack_id]
            entity.equipped_weapon = ew_oid
            entity.equipped_slots[EquipSlot.WEAPON_SLOT] = ew_oid

        # Equipped armor (legacy field — also populates equipped_slots)
        ea_pack_id = raw.get("equipped_armor")
        if ea_pack_id:
            if ea_pack_id not in object_id_map:
                raise WorldPackError(
                    f"entity '{pack_id}' equipped_armor references unknown "
                    f"object '{ea_pack_id}'."
                )
            ea_oid = object_id_map[ea_pack_id]
            entity.equipped_armor = ea_oid
            entity.equipped_slots[EquipSlot.ARMOR_SLOT] = ea_oid

    # ── 5. Place entities & ground objects on the grid ──────────────────────
    for entity in entities.values():
        place_entity(spatial, entity)
    for pack_id, obj in objects.items():
        spatial.objects[obj.object_id] = obj

    # ── 6. Build relational graph ───────────────────────────────────────────
    graph = RelationalGraph()

    # entity nodes
    for pack_id, entity in entities.items():
        add_node(graph, entity.entity_id, NodeKind.ENTITY, entity.name)

    # faction / organization / location nodes
    faction_id_map: dict[str, FactionId] = {}
    for raw in factions_doc.get("factions", []):
        pack_id = _require(raw, "id", "factions.yaml")
        name = _require(raw, "name", "factions.yaml")
        kind = _parse_node_kind(raw.get("kind", "faction"))
        # Faction-class nodes get pack-local id prefix to keep them stable
        # across pack reloads; entities use UUIDs because their identity
        # must be unique per run.
        fid = FactionId(f"faction_{pack_id}")
        faction_id_map[pack_id] = fid
        meta = dict(raw.get("meta") or {})
        tags = raw.get("tags") or []
        if tags:
            meta["tags"] = list(tags)
        add_node(graph, fid, kind, name, meta=meta)

    # lore concept nodes
    concept_id_map: dict[str, str] = {}
    for raw in lore_doc.get("concepts", []) or []:
        pack_id = _require(raw, "id", "lore.yaml")
        name = _require(raw, "name", "lore.yaml")
        kind = _parse_node_kind(raw.get("kind", "faction"))
        cid = f"concept_{pack_id}"
        concept_id_map[pack_id] = cid
        meta = dict(raw.get("meta") or {})
        tags = raw.get("tags") or []
        if tags:
            meta["tags"] = list(tags)
        add_node(graph, cid, kind, name, meta=meta)

    # ── 7. Edges ────────────────────────────────────────────────────────────
    def resolve_local(token: str, location: str) -> str:
        if token in entity_id_map:
            return entity_id_map[token]
        if token in faction_id_map:
            return faction_id_map[token]
        if token in concept_id_map:
            return concept_id_map[token]
        raise WorldPackError(
            f"{location}: unknown local id '{token}' (not an entity, "
            f"faction, or concept)."
        )

    for raw in relations_doc.get("edges", []) or []:
        src = resolve_local(_require(raw, "source", "relations.yaml"), "relations.yaml")
        dst = resolve_local(_require(raw, "target", "relations.yaml"), "relations.yaml")
        kind = _parse_edge_kind(_require(raw, "kind", "relations.yaml"))
        weight = float(raw.get("weight", 1.0))
        meta = raw.get("meta") or {}
        add_edge(graph, src, dst, kind, weight=weight, tick=0, meta=meta)

    # ── 8. Config (affordances + NPC policy) ────────────────────────────────
    affordance_rules: list[AffordanceRule] = []
    for raw in affordances_doc.get("affordances", []) or []:
        affordance_rules.append(
            AffordanceRule(
                keyword=_require(raw, "keyword", "affordances.yaml"),
                kinds=[_parse_node_kind(k) for k in (raw.get("kinds") or [])],
                name_keywords=list(raw.get("name_keywords") or []),
                tags=list(raw.get("tags") or []),
            )
        )

    object_affordance_rules: list[ObjectAffordanceRule] = []
    for raw in affordances_doc.get("object_affordances", []) or []:
        object_affordance_rules.append(
            ObjectAffordanceRule(
                verb=str(_require(raw, "verb", "affordances.yaml object_affordances")),
                object_tags=[str(t).lower() for t in (raw.get("object_tags") or [])],
                absent_tags=[str(t).lower() for t in (raw.get("absent_tags") or [])],
            )
        )

    from .schemas import MindScoringConfig, NpcCognitionConfig

    cog_doc = policy_doc.get("cognition") or {}
    raw_mode = str(cog_doc.get("mode", policy_doc.get("cognition_mode", "lm")))
    if raw_mode in ("menu_lm", "tiered", "infer_allowed"):
        raw_mode = "lm"
    cognition = NpcCognitionConfig(
        mode=raw_mode,
        max_infer_per_npc=int(cog_doc.get("max_infer_per_npc", 2)),
        infer_cooldown_ticks=int(cog_doc.get("infer_cooldown_ticks", 8)),
        max_lm_calls_per_tick=(
            None
            if (_raw := cog_doc.get("max_lm_calls_per_tick", 6)) in (None, 0, "0")
            else int(_raw)
        ),
    )
    ms_doc = policy_doc.get("mind_scoring") or {}
    mind_scoring = MindScoringConfig(
        **{k: v for k, v in ms_doc.items() if k in MindScoringConfig.model_fields}
    )
    npc_policy = NpcPolicyConfig(
        threat_lookback_ticks=int(policy_doc.get("threat_lookback_ticks", 5)),
        flee_health_fraction=float(policy_doc.get("flee_health_fraction", 0.25)),
        cognition=cognition,
        mind_scoring=mind_scoring,
    )

    verb_templates: dict[str, VerbTemplate] = {}
    for raw in verb_templates_doc.get("templates", []) or []:
        verb = _require(raw, "verb", "verb_templates.yaml")
        verb_templates[verb] = _build_verb_template(verb, raw)
    _merge_default_verb_templates(pack, verb_templates)

    open_verbs: list[OpenVerbRule] = [
        _build_open_verb_rule(raw)
        for raw in open_verbs_doc.get("rules") or []
    ]
    open_verbs = _merge_default_open_verbs(pack, open_verbs)

    # World clock — pack may declare a `clock` block in world.yaml.
    clock_doc = world_meta.get("clock") or {}
    clock = WorldClock(
        day_length_ticks=int(clock_doc.get("day_length_ticks", 100)),
    )

    # Ambient events declared in ambient_events.yaml.
    ambient_events: list[AmbientEvent] = []
    for raw in ambient_doc.get("events", []) or []:
        ambient_events.append(_build_ambient_event(raw))

    # Goals declared in goals.yaml.
    goals = []
    for raw in goals_doc.get("goals", []) or []:
        goals.append(_build_goal(raw, entity_id_map))

    pressures = list(pressures_doc.get("pressures") or [])
    reactive_triggers = _merge_default_reactive_triggers(
        pack, list(reactive_triggers_doc.get("triggers") or []),
    )

    # ── 8b. Spell vocabulary (magic.yaml + grimoires/) ───────────────────────
    spell_vocab = _load_spell_vocab(pack)

    # ── 8c. Physics / chemistry config (materials.yaml + reactions.yaml) ─────
    inherit_physics = bool(
        world_meta.get("physics", world_meta.get("inherit_physics", True))
    )
    physics_config = _load_physics_chain(pack_chain, inherit_physics=inherit_physics)

    from .fluids import load_fluid_catalog

    fluid_catalog = load_fluid_catalog(pack)
    if fluid_catalog is None and len(pack_chain) > 1:
        for parent in pack_chain[1:]:
            fluid_catalog = load_fluid_catalog(parent)
            if fluid_catalog is not None:
                break

    from .contamination import load_agent_catalog
    from .fields import load_substance_catalog

    agent_catalog = load_agent_catalog(pack)
    if agent_catalog is None and len(pack_chain) > 1:
        for parent in pack_chain[1:]:
            agent_catalog = load_agent_catalog(parent)
            if agent_catalog is not None:
                break
    substance_catalog = load_substance_catalog(pack)
    if substance_catalog is None and len(pack_chain) > 1:
        for parent in pack_chain[1:]:
            substance_catalog = load_substance_catalog(parent)
            if substance_catalog is not None:
                break

    # ── 8d. Item combination rules (combinations.yaml) ───────────────────────
    combination_rules = _load_combinations_chain(pack_chain)

    # ── 8e. Interaction grammar + adjudication index ─────────────────────────
    interaction_rules = _load_interaction_rules_chain(pack_chain)
    adjudication_index = _load_adjudication_index_chain(pack_chain)
    property_interactions = _load_property_interactions_chain(pack_chain)

    # ── 8g. NPC long-horizon plans (npc_plans.yaml) ─────────────────────────
    npc_plan_templates = _load_npc_plans_chain(pack_chain)

    # ── 8f. Synthesis hints (synthesis_hints.yaml) ────────────────────────────
    synthesis_hints = _load_synthesis_hints_chain(pack_chain)

    extra = dict(world_meta.get("extra") or {})
    from .schemas import SemanticDynamicsConfig

    sem_doc = extra.get("semantic") or world_meta.get("semantic") or {}
    semantic_cfg = SemanticDynamicsConfig(
        enabled=bool(sem_doc.get("enabled", True)),
        interval_ticks=int(sem_doc.get("interval_ticks", 3)),
        max_events_per_slice=int(sem_doc.get("max_events_per_slice", 20)),
        max_deltas_per_lens=int(sem_doc.get("max_deltas_per_lens", 8)),
        edge_delta_cap=float(sem_doc.get("edge_delta_cap", 0.25)),
        use_mock=bool(sem_doc.get("use_mock", True)),
        reaction_memory_ticks=int(sem_doc.get("reaction_memory_ticks", 6)),
        belief_propagation_interval=int(
            sem_doc.get("belief_propagation_interval", 1),
        ),
    )
    scenario_ref = extra.get("scenario")
    if isinstance(scenario_ref, str):
        scenario_path = pack / scenario_ref
        if scenario_path.is_file():
            with open(scenario_path, encoding="utf-8") as sf:
                extra["scenario"] = yaml.safe_load(sf) or {}

    cons_doc = world_meta.get("consequences") or {}
    consequence_policy = ConsequencePolicy(
        require_sticky=bool(cons_doc.get("require_sticky", True)),
        record_traces=bool(cons_doc.get("record_traces", True)),
        trace_limit=int(cons_doc.get("trace_limit", 24)),
        world_memory_window=int(cons_doc.get("world_memory_window", 6)),
        narrative_only_verbs=list(cons_doc.get("narrative_only_verbs") or []),
        default_interacted_delta=float(cons_doc.get("default_interacted_delta", 0.12)),
        max_world_facts=int(cons_doc.get("max_world_facts", 400)),
    )

    config = WorldConfig(
        consequence_policy=consequence_policy,
        affordances=affordance_rules,
        object_affordances=object_affordance_rules,
        npc_policy=npc_policy,
        verb_templates=verb_templates,
        open_verbs=open_verbs,
        clock=clock,
        ambient_events=ambient_events,
        pressures=pressures,
        reactive_triggers=reactive_triggers,
        extra=extra,
        goals=goals,
        spell_vocab=spell_vocab,
        physics_config=physics_config,
        fluid_catalog=fluid_catalog,
        agent_catalog=agent_catalog,
        substance_catalog=substance_catalog,
        combination_rules=combination_rules,
        interaction_rules=interaction_rules,
        adjudication_index=adjudication_index,
        property_interactions=property_interactions,
        synthesis_hints=synthesis_hints,
        semantic=semantic_cfg,
        npc_plan_templates=npc_plan_templates,
    )

    # ── 9. Build region graph ────────────────────────────────────────────────
    # Single-region worlds (the classic format) are auto-wrapped in a
    # Region named "region_main".  Multi-region worlds declare additional
    # regions under world.yaml's `regions:` key.
    #
    # world.yaml multi-region format:
    #
    #   regions:
    #     - id: region_cellar
    #       name: "The Cellar"
    #       type: dungeon          # indoor | outdoor | dungeon | cave | void
    #       floor_level: -1
    #       description: "Dark and damp."
    #       map_file: regions/cellar/map.txt
    #       entities_file: regions/cellar/entities.yaml
    #       objects_file: regions/cellar/objects.yaml
    #
    #   portals:
    #     - id: portal_trapdoor
    #       label: "Trapdoor"
    #       from_region: region_main
    #       from_coord: {x: 5, y: 8}
    #       to_region: region_cellar
    #       to_coord: {x: 5, y: 1}
    #       bidirectional: true

    # Primary region wraps the already-loaded spatial grid.
    rtype_str = world_meta.get("region_type", "indoor")
    try:
        primary_rtype = RegionType(rtype_str)
    except ValueError:
        primary_rtype = RegionType.INDOOR

    primary_region = Region(
        region_id="region_main",
        name=world_meta.get("region_name") or world_meta.get("name", "Main Area"),
        region_type=primary_rtype,
        floor_level=int(world_meta.get("floor_level", 0)),
        description=world_meta.get("region_description", ""),
        grid=spatial,
    )
    regions: dict[str, Region] = {"region_main": primary_region}

    # Additional regions declared in world.yaml — use the pre-loaded raw data.
    for reg_raw in world_meta.get("regions") or []:
        rid = _require(reg_raw, "id", "world.yaml regions")
        if rid in regions:
            raise WorldPackError(f"Duplicate region id: {rid!r}")
        reg_meta = {
            "map_file": reg_raw.get("map_file"),
            "width": int(reg_raw.get("width", world_meta.get("spatial", {}).get("width", 20))),
            "height": int(reg_raw.get("height", world_meta.get("spatial", {}).get("height", 20))),
        }
        reg_grid = _load_spatial(pack, reg_meta)

        # Build entities from pre-loaded raw list (IDs already in entity_id_map)
        for raw_e in _region_raw_entities.get(rid, []):
            pack_id = _require(raw_e, "id", f"region {rid} entities")
            eid = entity_id_map[pack_id]
            ent = _build_entity(eid, raw_e, entity_id_map)
            ent.region_id = rid

            # Resolve inventory
            for inv_pid in raw_e.get("inventory") or []:
                if inv_pid in object_id_map:
                    ent.inventory.append(object_id_map[inv_pid])
            # Equipped weapon
            ew_pid = raw_e.get("equipped_weapon")
            if ew_pid and ew_pid in object_id_map:
                eid_w = object_id_map[ew_pid]
                ent.equipped_weapon = eid_w
                ent.equipped_slots[EquipSlot.WEAPON_SLOT] = eid_w
            # Equipped armor
            ea_pid = raw_e.get("equipped_armor")
            if ea_pid and ea_pid in object_id_map:
                eid_a = object_id_map[ea_pid]
                ent.equipped_armor = eid_a
                ent.equipped_slots[EquipSlot.ARMOR_SLOT] = eid_a

            entities[pack_id] = ent
            place_entity(reg_grid, ent)
            add_node(graph, eid, NodeKind.ENTITY, ent.name)

        # Build objects from pre-loaded raw list
        for raw_o in _region_raw_objects.get(rid, []):
            pack_id = _require(raw_o, "id", f"region {rid} objects")
            oid = object_id_map[pack_id]
            obj = _build_object(oid, raw_o, entity_id_map)
            obj.region_id = rid
            objects[pack_id] = obj
            reg_grid.objects[oid] = obj

        try:
            reg_rtype = RegionType(reg_raw.get("type", "indoor"))
        except ValueError:
            reg_rtype = RegionType.INDOOR

        regions[rid] = Region(
            region_id=rid,
            name=reg_raw.get("name", rid),
            region_type=reg_rtype,
            floor_level=int(reg_raw.get("floor_level", 0)),
            description=reg_raw.get("description", ""),
            grid=reg_grid,
        )

    # Portals declared in world.yaml.
    portals: list[Portal] = []
    for p_raw in world_meta.get("portals") or []:
        fc_raw = p_raw.get("from_coord") or {}
        tc_raw = p_raw.get("to_coord") or {}
        portals.append(Portal(
            portal_id=p_raw.get("id", f"portal_{uuid.uuid4().hex[:8]}"),
            label=p_raw.get("label", ""),
            from_region=_require(p_raw, "from_region", "world.yaml portals"),
            from_coord=Coord(x=int(fc_raw.get("x", 0)), y=int(fc_raw.get("y", 0))),
            to_region=_require(p_raw, "to_region", "world.yaml portals"),
            to_coord=Coord(x=int(tc_raw.get("x", 0)), y=int(tc_raw.get("y", 0))),
            bidirectional=bool(p_raw.get("bidirectional", True)),
            requires_key=object_id_map.get(p_raw["requires_key"])
                if p_raw.get("requires_key") else None,
        ))

    # ── 10. Assemble world ──────────────────────────────────────────────────
    # `rng_seed` is pack-controlled; defaults to the world name's hash so
    # two runs of the same pack share an ambient-event timeline.
    rng_seed = int(world_meta.get("rng_seed") or abs(hash(world_meta.get("name", "")) & 0xFFFFFFFF))

    # Seed faction economy data from factions.yaml into world.meta
    # so the economy tick can read/write it without touching the relational graph.
    meta_base = dict(world_meta.get("meta") or {})
    meta_factions: dict[str, dict] = {}
    for raw in factions_doc.get("factions", []):
        pack_id = raw.get("id", "")
        if not pack_id:
            continue
        raw_meta = dict(raw.get("meta") or {})
        meta_factions[pack_id] = {
            "name": raw.get("name", pack_id),
            "resources": dict(raw_meta.get("resources") or {}),
            "upkeep": dict(raw_meta.get("upkeep") or {"gold": 2.0}),
            "member_ids": list(raw_meta.get("member_ids") or []),
            "shortages": [],
        }
    if meta_factions:
        meta_base["factions"] = meta_factions

    meta_base["_pack_path"] = str(pack)
    meta_base["pack_extends"] = [str(p) for p in pack_chain]
    meta_base["entity_pack_ids"] = {
        str(eid): pack_id for pack_id, eid in entity_id_map.items()
    }
    meta_base["object_pack_ids"] = {
        str(oid): pack_id for pack_id, oid in object_id_map.items()
    }
    if spatial.use_voxels or spatial.depth > 1:
        meta_base["spatial_mode"] = "voxel"
    else:
        meta_base["spatial_mode"] = "grid"
    if physics_config is not None:
        meta_base["physics_enabled"] = True
    world = WorldState(
        regions=regions,
        portals=portals,
        active_region_id="region_main",
        relational=graph,
        name=world_meta.get("name", "unnamed_world"),
        meta=meta_base,
        config=config,
        rng_seed=rng_seed,
    )
    logger.info(
        "Loaded world pack '%s' (%d region(s), %d portal(s), %d entities, "
        "%d factions, %d edges, %d affordance rules)",
        world.name,
        len(regions),
        len(portals),
        len(entities),
        len(faction_id_map),
        sum(len(t) for t in graph.edges.values()),
        len(affordance_rules),
    )
    from .economy import init_market

    init_market(world)
    return world


# ─────────────────────────────────────────────────────────────────────────────
# File helpers
# ─────────────────────────────────────────────────────────────────────────────


def _read_yaml(path: Path, *, required: bool) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise WorldPackError(f"Required pack file missing: {path}")
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise WorldPackError(
            f"{path}: expected a mapping at top level, got {type(data).__name__}."
        )
    return data


def _require(d: dict, key: str, where: str) -> Any:
    if key not in d:
        raise WorldPackError(f"{where}: missing required field '{key}'.")
    return d[key]


# ─────────────────────────────────────────────────────────────────────────────
# Spatial loading
# ─────────────────────────────────────────────────────────────────────────────


_TERRAIN_CHAR_MAP: dict[str, TerrainType] = {
    ".": TerrainType.FLOOR,
    "#": TerrainType.WALL,
    "+": TerrainType.DOOR_CLOSED,
    "/": TerrainType.DOOR_OPEN,
    ">": TerrainType.STAIRS_DOWN,
    "<": TerrainType.STAIRS_UP,
    "~": TerrainType.WATER,
    "=": TerrainType.WINDOW,
}


def _load_spatial(pack: Path, world_meta: dict) -> SpatialGrid:
    spatial_meta = world_meta.get("spatial") or world_meta
    use_voxel = bool(
        spatial_meta.get("voxel")
        or spatial_meta.get("use_voxels")
        or spatial_meta.get("voxel_preset")
        or (pack / spatial_meta.get("voxel_file", "voxel_map.yaml")).exists()
    )
    if use_voxel:
        from .voxel.loader import load_voxel_map

        return load_voxel_map(pack, world_meta)

    width = int(spatial_meta.get("width", 20))
    height = int(spatial_meta.get("height", 20))
    grid = SpatialGrid(width=width, height=height)

    map_file = spatial_meta.get("map_file", "map.txt")
    map_path = pack / map_file
    if not map_path.exists():
        raise WorldPackError(f"map_file '{map_path}' not found in pack.")
    rows = _read_map(map_path, width, height)
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            terrain = _TERRAIN_CHAR_MAP.get(ch, TerrainType.FLOOR)
            grid.set_tile(Coord(x=x, y=y), Tile(terrain=terrain))
    return grid


def _read_map(path: Path, width: int, height: int) -> list[str]:
    """Read an ASCII map, skipping leading lines beginning with '#' as
    long as they're outside the grid. The first line whose leading
    character is not '#' starts the grid; from that point '#' is treated
    as a wall tile."""
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    grid_lines: list[str] = []
    in_grid = False
    for line in raw_lines:
        if not in_grid:
            stripped = line.strip()
            # If the line is exactly the width of the grid and contains only
            # known map characters, that's the start of the grid.
            if stripped and len(stripped) == width and all(
                c in _TERRAIN_CHAR_MAP or c.isalpha() for c in stripped
            ):
                in_grid = True
                grid_lines.append(line)
                continue
            # otherwise: comment / blank line, skip
            continue
        grid_lines.append(line)
    if len(grid_lines) < height:
        raise WorldPackError(
            f"map_file '{path}' has {len(grid_lines)} grid lines; expected "
            f"{height}."
        )
    return grid_lines[:height]


# ─────────────────────────────────────────────────────────────────────────────
# Entity / object construction
# ─────────────────────────────────────────────────────────────────────────────


def _parse_social_aims(
    raw_list: list,
    entity_id_map: dict[str, EntityId],
) -> list:
    from .schemas import SocialAim

    out: list[SocialAim] = []
    for item in raw_list or []:
        if not isinstance(item, dict):
            continue
        target = item.get("target") or item.get("toward")
        aim = item.get("aim") or item.get("do") or item.get("intent")
        if not target or not aim:
            continue
        pack_id = str(target).strip()
        eid = entity_id_map.get(pack_id)
        if eid is None:
            continue
        out.append(
            SocialAim(
                target_entity_id=eid,
                target_pack_id=pack_id,
                aim=str(aim).strip(),
            )
        )
    return out


def _build_entity(
    eid: EntityId,
    raw: dict,
    entity_id_map: dict[str, EntityId] | None = None,
) -> EntityState:
    # Accept both `position: [x, y]` (list) and `x: N  y: N` (flat fields)
    if "x" in raw and "y" in raw:
        pos = [raw["x"], raw["y"], raw.get("z", 0)]
    else:
        pos = list(raw.get("position") or [0, 0])
        if len(pos) == 2:
            pos.append(0)
    # Waypoints: [x, y] or [x, y, z]
    raw_waypoints = raw.get("waypoints") or []
    waypoints = []
    for wp in raw_waypoints:
        if len(wp) >= 3:
            waypoints.append(Coord(x=int(wp[0]), y=int(wp[1]), z=int(wp[2])))
        else:
            waypoints.append(Coord(x=int(wp[0]), y=int(wp[1]), z=0))
    meta = dict(raw.get("meta") or {})
    if raw.get("material"):
        meta["material"] = str(raw["material"])
    if raw.get("temperature") is not None:
        meta["temperature"] = float(raw["temperature"])
    return EntityState(
        entity_id=eid,
        name=_require(raw, "name", "entities.yaml"),
        kind=_parse_entity_kind(raw.get("kind", "npc")),
        position=Coord(x=int(pos[0]), y=int(pos[1]), z=int(pos[2] if len(pos) > 2 else 0)),
        armed=bool(raw.get("armed", False)),
        alertness=_parse_alertness(raw.get("alertness", "unaware")),
        emotional_state=_parse_emotional(raw.get("emotional_state", "neutral")),
        health=int(raw.get("health", 100)),
        max_health=int(raw.get("max_health", 100)),
        sight_range=int(raw.get("sight_range", 8)),
        attributes=dict(raw.get("attributes") or {}),
        tags=list(raw.get("tags") or []),
        social_openness=str(raw.get("social_openness", "guarded")).lower(),
        role=str(raw.get("role", "") or ""),
        personality=str(raw.get("personality", "") or ""),
        drive=str(raw.get("drive", "") or ""),
        waypoints=waypoints,
        knowledge=list(raw.get("knowledge") or []),
        goals=list(raw.get("goals") or []),
        social_aims=_parse_social_aims(
            raw.get("social_aims") or [], entity_id_map or {}
        ),
        secrets=list(raw.get("secrets") or []),
        voice_lines=list(raw.get("voice_lines") or []),
        # Carry capacity: explicit override or derive from strength attribute.
        # Strength 50 → 50 kg base; each point above/below adds/removes 0.5 kg.
        carry_capacity=float(raw.get("carry_capacity") or (
            50.0 + (int((raw.get("attributes") or {}).get("strength", 50)) - 50) * 0.5
        )),
        facing=_parse_facing(raw.get("facing", "south")),
        fov_degrees=float(raw.get("fov_degrees", 135.0)),
        # Spell system
        mana=int(raw.get("mana", 0)),
        max_mana=int(raw.get("max_mana", raw.get("mana", 0))),
        known_spell_ops=list(raw.get("known_spell_ops") or []),
        inscribed_spells=dict(raw.get("inscribed_spells") or {}),
        # Generic stat bag — merge defaults with pack overrides.
        # The YAML `stats:` key takes precedence; missing keys get defaults.
        stats=_build_entity_stats(raw),
        # Skill bag — loaded from YAML; missing keys filled by skill_defaults().
        skills=_build_entity_skills(raw),
        # Non-combat identity
        occupation=str(raw.get("occupation", "") or ""),
        faction=str(raw.get("faction", "") or ""),
        meta=meta,
        # equipped_weapon / equipped_armor / equipped_slots wired up later
        # inventory wired up later after objects are built
    )


def _build_entity_stats(raw: dict) -> dict:
    """
    Merge default stats with any explicit ``stats:`` block from the YAML.

    Defaults:
      stamina    100.0
      gold       0.0
      reputation 0.0
      hunger     75.0
      fatigue    85.0
      social_need 60.0
      purpose    70.0

    The YAML may declare any additional stats; they are stored as floats.
    """
    from .needs import needs_defaults
    occ = str(raw.get("occupation", "") or "")
    defaults: dict[str, float] = {
        "stamina": 100.0,
        "gold": 0.0,
        "reputation": 0.0,
        **needs_defaults(occ),
    }
    raw_stats = raw.get("stats") or {}
    merged = dict(defaults)
    for k, v in raw_stats.items():
        try:
            merged[str(k)] = float(v)
        except (TypeError, ValueError):
            pass
    return merged


def _build_entity_skills(raw: dict) -> dict:
    """
    Merge occupation-appropriate skill defaults with any explicit ``skills:``
    block from the YAML.  Missing keys are filled from skill_defaults(occ).
    """
    from .skills import skill_defaults
    occ = str(raw.get("occupation", "") or "")
    defaults = skill_defaults(occ)
    raw_skills = raw.get("skills") or {}
    merged = dict(defaults)
    for k, v in raw_skills.items():
        try:
            merged[str(k)] = float(v)
        except (TypeError, ValueError):
            pass
    return merged


def _build_object(
    oid: ObjectId, raw: dict, entity_id_map: dict[str, EntityId]
) -> ObjectState:
    owner_pack_id = raw.get("owner")
    owner_real: EntityId | None = None
    if owner_pack_id:
        if owner_pack_id not in entity_id_map:
            raise WorldPackError(
                f"objects.yaml: object '{raw.get('id')}' has unknown owner "
                f"'{owner_pack_id}'."
            )
        owner_real = entity_id_map[owner_pack_id]
    if "x" in raw and "y" in raw:
        coord = Coord(x=int(raw["x"]), y=int(raw["y"]))
    else:
        pos = raw.get("position")
        coord = Coord(x=int(pos[0]), y=int(pos[1])) if pos else None

    # Slot validation: if a slot is specified it must be a canonical name.
    from .schemas import EquipSlot
    slot_raw = raw.get("slot")
    if slot_raw is not None and slot_raw not in EquipSlot.ALL:
        raise WorldPackError(
            f"objects.yaml: object '{raw.get('id')}' has unknown slot "
            f"'{slot_raw}'. Valid slots: {sorted(EquipSlot.ALL)}"
        )

    # Grimoire items: store the grimoire_id in meta so game_loop can find it.
    meta: dict[str, Any] = dict(raw.get("meta") or {})
    if raw.get("grimoire_id"):
        meta["grimoire_id"] = str(raw["grimoire_id"])
    if raw.get("teaches_ops"):
        meta["teaches_ops"] = list(raw["teaches_ops"])
    if raw.get("material"):
        meta["material"] = str(raw["material"])
    if raw.get("temperature") is not None:
        meta["temperature"] = float(raw["temperature"])
    if raw.get("pour_source"):
        meta["pour_source"] = True

    fluid_cap = raw.get("fluid_capacity_ml")
    fluid_vol = raw.get("fluid_volume_ml")
    fluid_mat = raw.get("fluid_material") or raw.get("fluid")
    if isinstance(fluid_mat, dict):
        fluid_vol = fluid_mat.get("volume_ml", fluid_vol)
        fluid_cap = fluid_mat.get("capacity_ml", fluid_cap)
        fluid_mat = fluid_mat.get("material", "ale")
    if fluid_cap is not None:
        meta["fluid"] = {
            "capacity_ml": float(fluid_cap),
            "volume_ml": float(fluid_vol or 0),
            "material": str(fluid_mat or "").lower(),
        }

    return ObjectState(
        object_id=oid,
        name=_require(raw, "name", "objects.yaml"),
        position=coord,
        owner=owner_real,
        passable=bool(raw.get("passable", True)),
        tags=list(raw.get("tags") or []),
        attributes=dict(raw.get("attributes") or {}),
        weight=float(raw.get("weight", 1.0)),
        bulk=float(raw.get("bulk", 1.0)),
        slot=slot_raw,
        durability=int(raw.get("durability", 100)),
        max_durability=int(raw.get("max_durability", raw.get("durability", 100))),
        is_container=bool(raw.get("is_container", False)),
        carry_capacity=float(raw.get("carry_capacity", 0.0)),
        bulk_capacity=float(raw.get("bulk_capacity", 0.0)),
        meta=meta,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Enum parsers (case-insensitive, friendly errors)
# ─────────────────────────────────────────────────────────────────────────────


def _parse_entity_kind(s: str) -> EntityKind:
    try:
        return EntityKind(s.lower())
    except ValueError as exc:
        raise WorldPackError(
            f"unknown entity kind '{s}'. Valid: {[k.value for k in EntityKind]}"
        ) from exc


def _parse_alertness(s: str) -> AlertnessLevel:
    try:
        return AlertnessLevel(s.lower())
    except ValueError as exc:
        raise WorldPackError(
            f"unknown alertness '{s}'. Valid: {[k.value for k in AlertnessLevel]}"
        ) from exc


def _parse_emotional(s: str) -> EmotionalState:
    try:
        return EmotionalState(s.lower())
    except ValueError as exc:
        raise WorldPackError(
            f"unknown emotional_state '{s}'. "
            f"Valid: {[k.value for k in EmotionalState]}"
        ) from exc


def _parse_node_kind(s: str) -> NodeKind:
    try:
        return NodeKind(s.lower())
    except ValueError as exc:
        raise WorldPackError(
            f"unknown node kind '{s}'. Valid: {[k.value for k in NodeKind]}"
        ) from exc


def _build_verb_template(verb: str, raw: dict) -> VerbTemplate:
    """
    Parse a pack-declared verb template into a runtime VerbTemplate.

    YAML shape::

        verb: console
        contest:                         # optional
          actor_attribute: persuasion
          target_attribute: composure
          difficulty: 50
        on_success:                      # list of TransitionProposal-shaped dicts
          - kind: entity_emotional_state_changed
            payload: {entity_id: "$target", to: friendly}
        on_failure:
          - kind: entity_emotional_state_changed
            payload: {entity_id: "$target", to: suspicious}
    """
    contest_raw = raw.get("contest")
    contest: ContestSpec | None = None
    if contest_raw:
        contest = ContestSpec(
            actor_attribute=str(contest_raw.get("actor_attribute", "persuasion")),
            target_attribute=(
                str(contest_raw["target_attribute"])
                if "target_attribute" in contest_raw and contest_raw["target_attribute"]
                else None
            ),
            difficulty=int(contest_raw.get("difficulty", 50)),
        )

    def _proposals(key: str) -> list[TransitionProposal]:
        out: list[TransitionProposal] = []
        for r in raw.get(key, []) or []:
            kind = _require(r, "kind", f"verb_templates.yaml ({verb})")
            payload = dict(r.get("payload") or {})
            out.append(TransitionProposal(kind=str(kind), payload=payload))
        return out

    return VerbTemplate(
        verb=verb,
        contest=contest,
        effects_on_success=_proposals("on_success") or _proposals("effects"),
        effects_on_failure=_proposals("on_failure"),
        narrative_success=(
            str(raw["narrative_success"]).strip()
            if raw.get("narrative_success")
            else None
        ),
        narrative_failure=(
            str(raw["narrative_failure"]).strip()
            if raw.get("narrative_failure")
            else None
        ),
        narrative_object_success=(
            str(raw["narrative_object_success"]).strip()
            if raw.get("narrative_object_success")
            else None
        ),
    )


def _build_open_verb_rule(raw: dict[str, Any]) -> OpenVerbRule:
    """Parse one ``open_verbs.yaml`` rule entry."""
    rule_id = _require(raw, "id", "open_verbs.yaml")
    contest_raw = raw.get("contest")
    contest: ContestSpec | None = None
    if contest_raw:
        contest = ContestSpec(
            actor_attribute=str(contest_raw.get("actor_attribute", "persuasion")),
            target_attribute=(
                str(contest_raw["target_attribute"])
                if contest_raw.get("target_attribute")
                else None
            ),
            difficulty=int(contest_raw.get("difficulty", 50)),
        )

    def _proposals(key: str) -> list[TransitionProposal]:
        out: list[TransitionProposal] = []
        for r in raw.get(key, []) or []:
            kind = _require(r, "kind", f"open_verbs.yaml ({rule_id})")
            payload = dict(r.get("payload") or {})
            out.append(TransitionProposal(kind=str(kind), payload=payload))
        return out

    handler = raw.get("handler")
    return OpenVerbRule(
        id=str(rule_id),
        verbs=[str(v).lower() for v in (raw.get("verbs") or [])],
        keywords=[str(k).lower() for k in (raw.get("keywords") or [])],
        requires=str(raw["requires"]).lower() if raw.get("requires") else None,
        handler=str(handler).lower() if handler else None,
        contest=contest,
        on_success=_proposals("on_success"),
        on_failure=_proposals("on_failure"),
        narrative_success=(
            str(raw["narrative_success"]).strip()
            if raw.get("narrative_success")
            else None
        ),
        narrative_failure=(
            str(raw["narrative_failure"]).strip()
            if raw.get("narrative_failure")
            else None
        ),
    )


def _parse_edge_kind(s: str) -> EdgeKind:
    try:
        return EdgeKind(s.lower())
    except ValueError as exc:
        raise WorldPackError(
            f"unknown edge kind '{s}'. Valid: {[k.value for k in EdgeKind]}"
        ) from exc


def _parse_time_of_day(s: str) -> TimeOfDay:
    try:
        return TimeOfDay(s.lower())
    except ValueError as exc:
        raise WorldPackError(
            f"unknown time_of_day '{s}'. Valid: {[k.value for k in TimeOfDay]}"
        ) from exc


def _parse_facing(s: str) -> FacingDirection:
    """Accept cardinal names ('north', 'n', 'ne', 'northeast', …) with fallback."""
    _aliases: dict[str, str] = {
        "n": "north", "s": "south", "e": "east", "w": "west",
        "ne": "northeast", "nw": "northwest",
        "se": "southeast", "sw": "southwest",
    }
    normalized = s.lower().strip()
    resolved = _aliases.get(normalized, normalized)
    try:
        return FacingDirection(resolved)
    except ValueError:
        return FacingDirection.SOUTH


def _load_spell_vocab(pack: Path) -> "Any":
    """
    Load magic.yaml and grimoires/*.yaml from the world pack.

    Returns a SpellVocabulary if magic.yaml exists, or None if this world
    has no magic system defined.
    """
    magic_path = pack / "magic.yaml"
    if not magic_path.exists():
        return None

    # Import here to avoid circular dependency at module load time.
    from .spell_types import GrimoireDefinition, OpDefinition, SpellVocabulary

    magic_doc = _read_yaml(magic_path, required=False) or {}
    ops_raw = magic_doc.get("operations") or {}

    operations: dict[str, OpDefinition] = {}
    for op_name, op_def in ops_raw.items():
        if not isinstance(op_def, dict):
            continue
        operations[op_name.upper()] = OpDefinition(
            name=op_name.upper(),
            primitive=str(op_def.get("primitive", "ADJUST_PROP")),
            args=list(op_def.get("args") or []),
            prop=op_def.get("prop"),
            sign=int(op_def.get("sign", 1)),
            state=op_def.get("state"),
            condition=op_def.get("condition"),
            tag=op_def.get("tag"),
            edge_kind=op_def.get("edge_kind"),
            requires_tag=op_def.get("requires_tag"),
            mana_formula=str(op_def.get("mana_formula", "0")),
            description=str(op_def.get("description", "")),
        )

    # Load grimoires from grimoires/ subdirectory
    grimoires: dict[str, GrimoireDefinition] = {}
    grimoires_dir = pack / "grimoires"
    if grimoires_dir.is_dir():
        for grimoire_path in sorted(grimoires_dir.glob("*.yaml")):
            try:
                g_doc = _read_yaml(grimoire_path, required=False) or {}
                gid = str(g_doc.get("id", grimoire_path.stem))
                grimoires[gid] = GrimoireDefinition(
                    grimoire_id=gid,
                    name=str(g_doc.get("name", gid)),
                    text=str(g_doc.get("text", "")),
                    teaches_ops=[
                        str(op).upper() for op in (g_doc.get("teaches_ops") or [])
                    ],
                )
            except Exception as exc:
                logger.warning("Failed to load grimoire %s: %s", grimoire_path, exc)

    if not operations and not grimoires:
        return None

    return SpellVocabulary(operations=operations, grimoires=grimoires)


def _load_physics(pack: Path) -> "Any":
    """Load physics from one pack (backward compatible)."""
    return _load_physics_chain([pack], inherit_physics=True)


def _load_physics_chain(chain: list[Path], *, inherit_physics: bool = True) -> "Any":
    """
    Merge physics config across ``extends`` chain (Overhaul C).
    """
    if not inherit_physics and not any(
        (p / "materials.yaml").exists() or (p / "reactions.yaml").exists()
        for p in chain
    ):
        return None

    materials_raw: dict = {}
    reactions_merged: dict[str, dict] = {}
    ambient_temp = 20.0
    heat_spread = 0.04
    found = False

    for pack in reversed(chain):
        mat_path = pack / "materials.yaml"
        react_path = pack / "reactions.yaml"
        if mat_path.exists() or react_path.exists():
            found = True
        mat_doc = _read_yaml(mat_path, required=False) or {}
        react_doc = _read_yaml(react_path, required=False) or {}
        if mat_doc.get("materials"):
            materials_raw.update(mat_doc.get("materials") or {})
        if mat_doc.get("ambient_temp") is not None:
            ambient_temp = float(mat_doc.get("ambient_temp", ambient_temp))
        if mat_doc.get("heat_spread") is not None:
            heat_spread = float(mat_doc.get("heat_spread", heat_spread))
        for raw in react_doc.get("reactions") or []:
            if isinstance(raw, dict):
                rid = str(raw.get("id", "")).strip()
                if rid:
                    reactions_merged[rid] = raw

    if not found or (not materials_raw and not reactions_merged):
        return None

    from .physics import build_physics_config

    try:
        return build_physics_config(
            materials_raw,
            list(reactions_merged.values()),
            ambient_temp=ambient_temp,
            heat_spread=heat_spread,
        )
    except Exception as exc:
        logger.warning("Failed to load physics config: %s", exc)
        return None


def _load_combinations_chain(chain: list[Path]) -> list:
    from .schemas import CombinationRule

    rules: list[CombinationRule] = []
    for raw in _merged_yaml_items(chain, "combinations.yaml", "combinations"):
        try:
            rules.append(CombinationRule(
                rule_id=str(raw.get("id", "unknown")),
                name=str(raw.get("name", "")),
                input_a_tags=list(raw.get("input_a_tags") or []),
                input_b_tags=list(raw.get("input_b_tags") or []),
                consume_a=bool(raw.get("consume_a", True)),
                consume_b=bool(raw.get("consume_b", True)),
                add_tag_to_a=raw.get("add_tag_to_a"),
                add_tag_to_b=raw.get("add_tag_to_b"),
                attribute_set_a=dict(raw.get("attribute_set_a") or {}),
                attribute_set_b=dict(raw.get("attribute_set_b") or {}),
                create_item=raw.get("create_item"),
                add_actor_tag=raw.get("add_actor_tag"),
                damage_actor=int(raw.get("damage_actor", 0)),
                narrative=str(raw.get("narrative", "")),
            ))
        except Exception as exc:
            logger.warning("Failed to load combination rule %s: %s", raw.get("id"), exc)
    return rules


def _load_interaction_rules_chain(chain: list[Path]) -> list:
    from .schemas import ContestSpec, InteractionRule, TransitionProposal

    rules: list[InteractionRule] = []
    for raw in _merged_yaml_items(chain, "interaction_rules.yaml", "interactions"):
        try:
            effects = [
                TransitionProposal(kind=str(e.get("kind")), payload=dict(e.get("payload") or {}))
                for e in (raw.get("effects") or [])
            ]
            contest = ContestSpec(**raw["contest"]) if raw.get("contest") else None
            rules.append(InteractionRule(
                id=str(raw.get("id", "")),
                name=str(raw.get("name", "")),
                verb_keywords=list(raw.get("verb_keywords") or []),
                substance_keywords=list(raw.get("substance_keywords") or []),
                actor_tags=list(raw.get("actor_tags") or []),
                target_tags=list(raw.get("target_tags") or []),
                tool_tags=list(raw.get("tool_tags") or []),
                target_kind=str(raw.get("target_kind", "any")),
                verb=str(raw.get("verb", "interact")),
                max_distance=int(raw.get("max_distance", 10)),
                priority=int(raw.get("priority", 0)),
                contest=contest,
                effects=effects,
                narrative=str(raw.get("narrative", "")),
            ))
        except Exception as exc:
            logger.warning("Failed to load interaction rule %s: %s", raw.get("id"), exc)
    return rules


def _load_adjudication_index_chain(chain: list[Path]) -> list:
    from .schemas import AdjudicationIndexEntry, TransitionProposal

    entries: list[AdjudicationIndexEntry] = []
    for raw in _merged_yaml_items(chain, "adjudication_index.yaml", "entries"):
        try:
            effects = [
                TransitionProposal(kind=str(e.get("kind")), payload=dict(e.get("payload") or {}))
                for e in (raw.get("effects") or [])
            ]
            entries.append(AdjudicationIndexEntry(
                id=str(raw.get("id", "")),
                patterns=list(raw.get("patterns") or []),
                action_types=list(raw.get("action_types") or []),
                substance_keywords=list(raw.get("substance_keywords") or []),
                min_confidence=float(raw.get("min_confidence", 0.55)),
                priority=int(raw.get("priority", 0)),
                ruling=str(raw.get("ruling", "")),
                synthesized_verb=raw.get("synthesized_verb"),
                fact_tags=list(raw.get("fact_tags") or []),
                effects=effects,
            ))
        except Exception as exc:
            logger.warning("Failed to load adjudication index %s: %s", raw.get("id"), exc)
    return entries


def _load_property_interactions_chain(chain: list[Path]) -> list:
    from .schemas import ContestSpec, PropertyInteractionRule, TransitionProposal

    rules: list[PropertyInteractionRule] = []
    for raw in _merged_yaml_items(chain, "property_interactions.yaml", "interactions"):
        try:
            effects = [
                TransitionProposal(kind=str(e.get("kind")), payload=dict(e.get("payload") or {}))
                for e in (raw.get("effects") or [])
            ]
            contest = ContestSpec(**raw["contest"]) if raw.get("contest") else None
            rules.append(PropertyInteractionRule(
                id=str(raw.get("id", "")),
                name=str(raw.get("name", "")),
                actor_or_tool_tags=list(raw.get("actor_or_tool_tags") or []),
                target_tags=list(raw.get("target_tags") or []),
                target_immunity_tags=list(raw.get("target_immunity_tags") or []),
                intent_categories=list(raw.get("intent_categories") or ["any"]),
                max_distance=int(raw.get("max_distance", 20)),
                priority=int(raw.get("priority", 0)),
                contest=contest,
                effects=effects,
                narrative=str(raw.get("narrative", "")),
            ))
        except Exception as exc:
            logger.warning("Failed to load property interaction %s: %s", raw.get("id"), exc)
    return rules


def _load_synthesis_hints_chain(chain: list[Path]) -> list:
    if len(chain) == 1:
        return _load_synthesis_hints(chain[0])
    from .schemas import SynthesisHint

    merged: dict[str, dict] = {}
    for pack_dir in reversed(chain):
        doc = _read_yaml(pack_dir / "synthesis_hints.yaml", required=False) or {}
        for raw in doc.get("hints") or []:
            if isinstance(raw, dict):
                hid = str(raw.get("id", "")).strip()
                if hid:
                    merged[hid] = raw
    hints: list[SynthesisHint] = []
    for raw in merged.values():
        try:
            hints.append(SynthesisHint(
                hint_id=str(raw.get("hint_id", raw.get("id", ""))),
                required_input_tags=list(raw.get("required_input_tags") or []),
                output_tags=list(raw.get("output_tags") or []),
                max_weight=float(raw.get("max_weight", 20.0)),
                max_bulk=float(raw.get("max_bulk", 20.0)),
                max_attribute=int(raw.get("max_attribute", 50)),
                stamina_cost=float(raw.get("stamina_cost", 10.0)),
            ))
        except Exception as exc:
            logger.warning("Failed to load synthesis hint: %s", exc)
    return hints


def _load_combinations(pack: Path) -> list:
    """Load combinations from one pack (backward compatible)."""
    return _load_combinations_chain([pack])


def _build_ambient_event(raw: dict) -> AmbientEvent:
    """
    Parse one entry from ambient_events.yaml.

    YAML shape::

        - id: torch_flicker
          narrative: "The torch flickers, casting shifting shadows."
          trigger:
            every_ticks: 12         # optional
            time_of_day: dawn       # optional, must match phase to fire
            probability: 1.0        # optional, 0..1
          effects:                  # optional list of TransitionProposal
            - kind: entity_emotional_state_changed
              payload: {entity_id: "ent_guard", to: neutral}
    """
    ambient_id = _require(raw, "id", "ambient_events.yaml")
    trigger_raw = raw.get("trigger") or {}
    trigger = AmbientTrigger(
        every_ticks=(
            int(trigger_raw["every_ticks"])
            if trigger_raw.get("every_ticks") is not None
            else None
        ),
        time_of_day=(
            _parse_time_of_day(str(trigger_raw["time_of_day"]))
            if trigger_raw.get("time_of_day")
            else None
        ),
        probability=float(
            trigger_raw.get("probability", raw.get("probability", 1.0))
        ),
    )
    effects: list[TransitionProposal] = []
    for r in raw.get("effects", []) or []:
        kind = _require(r, "kind", f"ambient_events.yaml ({ambient_id})")
        payload = dict(r.get("payload") or {})
        effects.append(TransitionProposal(kind=str(kind), payload=payload))
    return AmbientEvent(
        id=ambient_id,
        trigger=trigger,
        narrative=str(raw.get("narrative", "") or ""),
        effects=effects,
    )


def _build_goal(raw: dict, entity_id_map: dict[str, EntityId]) -> Goal:
    """
    Parse one entry from goals.yaml.

    YAML shape::

        - id: linna_uncovers_letter
          title: "The Letter Revealed"
          description: "Linna discovers Tomas's secret."
          conditions:
            - type: edge_exists
              source: linna          # pack-local id, resolved to UUID
              target: tomas
              edge_kind: knows_secret_of
            - type: event_occurred
              verb: eavesdrop
              actor: linna           # optional
          reward:
            narrative: "Linna's eyes widen as she reads the letter."
            lore_unlock: tomas_secret
            edge_updates:
              - source: linna
                target: tomas
                edge_kind: distrusts
                weight: 0.8

    Pack-local entity ids in condition ``source``, ``target``, ``entity``,
    ``actor`` fields are resolved to real EntityIds if present in
    ``entity_id_map``; otherwise the raw string is kept (allowing faction
    and location ids which have their own namespacing).
    """

    def _resolve(pack_id: str | None) -> str | None:
        if pack_id is None:
            return None
        return entity_id_map.get(pack_id, pack_id)

    goal_id = _require(raw, "id", "goals.yaml")
    conditions: list[GoalCondition] = []
    for c in raw.get("conditions", []) or []:
        conditions.append(
            GoalCondition(
                type=_require(c, "type", f"goals.yaml ({goal_id})"),
                source=_resolve(c.get("source")),
                target=_resolve(c.get("target")),
                edge_kind=c.get("edge_kind"),
                entity=_resolve(c.get("entity")),
                state=c.get("state"),
                tick=c.get("tick"),
                verb=c.get("verb"),
                actor=_resolve(c.get("actor")),
                same_region=bool(c.get("same_region", False)),
            )
        )

    reward_raw = raw.get("reward") or {}
    edge_updates: list[GoalEdgeUpdate] = []
    for upd in reward_raw.get("edge_updates", []) or []:
        edge_updates.append(
            GoalEdgeUpdate(
                source=_resolve(upd.get("source")) or "",
                target=_resolve(upd.get("target")) or "",
                edge_kind=upd.get("edge_kind", ""),
                weight=float(upd.get("weight", 0.5)),
            )
        )
    reward = GoalReward(
        narrative=reward_raw.get("narrative"),
        lore_unlock=reward_raw.get("lore_unlock"),
        edge_updates=edge_updates,
    )

    return Goal(
        id=goal_id,
        title=_require(raw, "title", "goals.yaml"),
        description=raw.get("description", ""),
        conditions=conditions,
        reward=reward,
    )


def _load_synthesis_hints(pack: Path) -> list:
    """
    Load synthesis_hints.yaml — open-crafting plausibility rules.

    Each hint declares which input tags are required and what output tags
    are permissible.  Used by _compile_craft to validate LM-proposed or
    rule-derived crafted objects without requiring every recipe to be
    pre-authored.
    """
    path = pack / "synthesis_hints.yaml"
    if not path.exists():
        return []
    from .schemas import SynthesisHint
    doc = _read_yaml(path, required=False) or {}
    hints = []
    for raw in (doc.get("hints") or []):
        try:
            hints.append(SynthesisHint(
                hint_id=str(raw.get("hint_id", "")),
                required_input_tags=list(raw.get("required_input_tags") or []),
                output_tags=list(raw.get("output_tags") or []),
                max_weight=float(raw.get("max_weight", 20.0)),
                max_bulk=float(raw.get("max_bulk", 20.0)),
                max_attribute=int(raw.get("max_attribute", 50)),
                stamina_cost=float(raw.get("stamina_cost", 10.0)),
            ))
        except Exception as exc:
            logger.warning("Failed to load synthesis hint %s: %s", raw.get("hint_id"), exc)
    return hints
