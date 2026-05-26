"""
Physics and Chemistry Engine.

Architecture
────────────
The engine runs one tick over the active region after every player turn.
It is PURELY reactive — it never invents new facts; it reads world state
and emits Transitions that apply_transitions() will commit.

Layer model
───────────
  WorldState (canonical)
      ↓  read-only
  PhysicsEngine.tick()
      ↓  returns list[Transition]
  apply_transitions()   ← same path everything else uses
      ↓
  WorldState (updated)

Integration with the spell system
──────────────────────────────────
Magic operations like HEAT / CHILL / IGNITE write properties (temperature,
on_fire tag, etc.) directly to entity state.  The physics engine then reads
those properties and fires secondary consequences:

  HEAT 400 TARGET          → temperature += 400 (spell)
  physics tick: entity above ignition_point → ADD on_fire tag
  physics tick: on_fire + biological → ENTITY_HEALTH_CHANGED −damage
  physics tick: on_fire + adjacent flammable → spread fire

Material system
───────────────
Materials are named bundles of physical constants loaded from materials.yaml.
Entities and objects reference materials via:
  • entity.meta["material"]   — explicit override (e.g. "iron_golem")
  • entity.tags               — "biological" → flesh constants
                                "metal"      → iron constants
                                etc.

The engine selects the *first* matching material (priority order from
world_loader) and uses its constants for that entity's physics.

Reaction system
───────────────
Reactions are data-driven rules loaded from reactions.yaml.  Each reaction
specifies:
  • conditions  — AND-list of checks against entity/neighbor state
  • effects     — list of state mutations to apply
  • probability — chance per tick when conditions hold (seeded = deterministic)
  • narrative   — optional text surfaced in the narration layer

Because both the material catalog and reaction rules live in YAML, world
authors can define completely novel physics without changing engine code.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional

from .schemas import (
    AlertnessLevel,
    Coord,
    EmotionalState,
    EntityId,
    EntityState,
    ObjectId,
    ObjectState,
    SpatialGrid,
    Transition,
    TransitionKind,
    WorldState,
)


# ---------------------------------------------------------------------------
# Material definition
# ---------------------------------------------------------------------------

@dataclass
class MaterialDef:
    """
    Physical/chemical constants for one material.

    Loaded from worlds/*/materials.yaml.  The `tags` list is inherited by
    any entity/object that claims this material — e.g. material "wood" has
    tag "flammable", so a wooden crate with material="wood" gains
    "flammable" automatically.

    Temperature thresholds (all in °C; None = threshold does not exist):
      ignition_point  — catches fire above this temperature
      melting_point   — changes to liquid state above this temperature
      boiling_point   — changes to gas state above this temperature (liquids only)
      damage_above    — starts taking heat damage above this temperature
      damage_below    — starts taking cold damage below this temperature
    damage_per_degree — health lost per tick per °C above/below the damage threshold
    """
    name: str
    display: str = ""
    tags: list[str] = field(default_factory=list)
    density: float = 1.0          # g/cm³
    conductivity: float = 0.1     # game units of heat transferred per tick per degree
    ignition_point: Optional[float] = None
    melting_point: Optional[float] = None
    boiling_point: Optional[float] = None
    damage_above: Optional[float] = None
    damage_below: Optional[float] = None
    damage_per_degree: float = 0.05  # health/tick/°C above threshold
    ph: float = 7.0               # 7 = neutral, < 7 = acidic
    hardness: int = 3             # 1-10 scale
    freeze_into: Optional[str] = None   # material produced when frozen (water → ice)
    melt_into: Optional[str] = None     # material produced when melted (ice → water)
    burn_into: Optional[str] = None     # material produced when burned (wood → ash)


# ---------------------------------------------------------------------------
# Reaction system
# ---------------------------------------------------------------------------

@dataclass
class ReactionCondition:
    """
    A single testable condition.  All conditions in a reaction are AND-combined.

    Supported types:
      has_tag        tag_name         entity has this tag
      missing_tag    tag_name         entity does NOT have this tag
      has_material_tag  tag_name      entity's material has this tag
      prop_above     {prop, value}    entity.meta[prop] > value
      prop_below     {prop, value}    entity.meta[prop] < value
      adjacent_tag   tag_name         at least one adjacent entity has this tag
    """
    cond_type: str
    tag: Optional[str] = None
    prop: Optional[str] = None
    value: Optional[float] = None


@dataclass
class ReactionEffect:
    """
    A single state mutation emitted when a reaction fires.

    Supported types:
      damage          value                  → ENTITY_HEALTH_CHANGED delta=-value
      heal            value                  → ENTITY_HEALTH_CHANGED delta=+value
      add_tag         tag_name               → entity.tags.append(tag)
      remove_tag      tag_name               → entity.tags.remove(tag)
      adjust_prop     {prop, delta}          → entity.meta[prop] += delta
      cap_prop        {prop, max}            → entity.meta[prop] = min(value, max)
      damage_by_excess   {prop,threshold,rate} → damage scales with prop - threshold
      damage_by_deficit  {prop,threshold,rate} → damage scales with threshold - prop
      spread_tag      {add_tag, requires_tag, chance}  → spread tag to neighbors
      conduct_heat    {rate}                 → share temperature with neighbors
      add_condition   {name, ticks}          → entity.conditions[name] = ticks
      remove_condition   name               → del entity.conditions[name]
      change_material    new_material_name  → update entity.meta["material"]
    """
    effect_type: str
    tag: Optional[str] = None
    prop: Optional[str] = None
    delta: Optional[float] = None
    value: Optional[float] = None
    rate: Optional[float] = None
    threshold: Optional[float] = None
    requires_tag: Optional[str] = None
    add_tag: Optional[str] = None
    chance: float = 1.0
    condition_name: Optional[str] = None
    ticks: int = 1
    material: Optional[str] = None
    max_value: Optional[float] = None


@dataclass
class ReactionDef:
    """
    A data-driven physics/chemistry rule.

    reaction_id  — unique string identifier for this reaction
    name         — human-readable label (shown in narrator on first occurrence)
    conditions   — all must hold for the reaction to fire
    effects      — applied in order if conditions hold AND probability passes
    probability  — per-tick chance (1.0 = always fires when conditions hold)
    narrative    — optional text pattern; {name} substituted with entity name
    silent       — if True, no narrator output even if narrative is non-empty
    """
    reaction_id: str
    name: str = ""
    conditions: list[ReactionCondition] = field(default_factory=list)
    effects: list[ReactionEffect] = field(default_factory=list)
    probability: float = 1.0
    narrative: Optional[str] = None
    silent: bool = False


# ---------------------------------------------------------------------------
# Physics configuration (loaded from YAML, stored on WorldConfig)
# ---------------------------------------------------------------------------

@dataclass
class PhysicsConfig:
    """
    Complete physics catalog for one world.

    materials    — dict[str, MaterialDef] keyed by material name
    reactions    — ordered list of ReactionDef; evaluated top-to-bottom
    ambient_temp — default temperature for entities with no explicit value (°C)
    heat_spread  — fraction of temperature difference spread to each neighbor per tick
    tag_material — priority-ordered list of (tag → material) mappings used to
                   infer a material from an entity's tags when no explicit
                   meta["material"] is set.  First match wins.
    """
    materials: dict[str, MaterialDef] = field(default_factory=dict)
    reactions: list[ReactionDef] = field(default_factory=list)
    ambient_temp: float = 20.0
    heat_spread: float = 0.05     # fraction transferred to each neighbor
    tag_material: list[tuple[str, str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Deterministic RNG for probability rolls
# ---------------------------------------------------------------------------

def _seeded_float(seed: str) -> float:
    digest = hashlib.sha256(seed.encode()).digest()
    return int.from_bytes(digest[:4], "big") / (2 ** 32)


# ---------------------------------------------------------------------------
# Material resolution helpers
# ---------------------------------------------------------------------------

def _entity_material(
    entity: EntityState, config: PhysicsConfig
) -> Optional[MaterialDef]:
    """Return the material definition for this entity, or None."""
    # Explicit override in meta wins
    mat_name = entity.meta.get("material")
    if mat_name and mat_name in config.materials:
        return config.materials[mat_name]
    # Tag-based fallback
    for tag, mat_name in config.tag_material:
        if tag in entity.tags:
            m = config.materials.get(mat_name)
            if m:
                return m
    return None


def _object_material(
    obj: ObjectState, config: PhysicsConfig
) -> Optional[MaterialDef]:
    """Return the material definition for this object, or None."""
    mat_name = obj.meta.get("material")
    if mat_name and mat_name in config.materials:
        return config.materials[mat_name]
    for tag, mat_name in config.tag_material:
        if tag in obj.tags:
            m = config.materials.get(mat_name)
            if m:
                return m
    return None


def _entity_temp(entity: EntityState, ambient: float) -> float:
    """Return entity's current temperature, defaulting to ambient."""
    from .substance import clamp_temperature

    return clamp_temperature(float(entity.meta.get("temperature", ambient)))


# ---------------------------------------------------------------------------
# Condition evaluation
# ---------------------------------------------------------------------------

def _check_conditions(
    entity: EntityState,
    grid: SpatialGrid,
    conditions: list[ReactionCondition],
    config: PhysicsConfig,
) -> bool:
    """Return True if ALL conditions hold for this entity."""
    for cond in conditions:
        if cond.cond_type == "has_tag":
            if cond.tag not in entity.tags:
                return False

        elif cond.cond_type == "missing_tag":
            if cond.tag in entity.tags:
                return False

        elif cond.cond_type == "has_material_tag":
            mat = _entity_material(entity, config)
            if mat is None or cond.tag not in mat.tags:
                return False

        elif cond.cond_type == "prop_above":
            val = float(entity.meta.get(cond.prop, config.ambient_temp if cond.prop == "temperature" else 0))
            if val <= (cond.value or 0):
                return False

        elif cond.cond_type == "prop_below":
            val = float(entity.meta.get(cond.prop, config.ambient_temp if cond.prop == "temperature" else 0))
            if val >= (cond.value or 0):
                return False

        elif cond.cond_type == "adjacent_tag":
            neighbors = _neighbor_entities(entity.position, grid)
            if not any(cond.tag in n.tags for n in neighbors):
                return False

    return True


# ---------------------------------------------------------------------------
# Neighbor lookup helpers
# ---------------------------------------------------------------------------

def _neighbor_entities(
    pos: Coord, grid: SpatialGrid, *, include_self: bool = False
) -> list[EntityState]:
    neighbors = []
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        npos = Coord(x=pos.x + dx, y=pos.y + dy)
        for ent in grid.entities.values():
            if ent.alive and ent.position == npos:
                neighbors.append(ent)
    if include_self:
        for ent in grid.entities.values():
            if ent.alive and ent.position == pos:
                neighbors.append(ent)
    return neighbors


# ---------------------------------------------------------------------------
# Effect application — returns Transitions
# ---------------------------------------------------------------------------

def _apply_effects(
    entity: EntityState,
    grid: SpatialGrid,
    effects: list[ReactionEffect],
    config: PhysicsConfig,
    tick: int,
    reaction_id: str,
) -> list[Transition]:
    transitions: list[Transition] = []
    eid = str(entity.entity_id)
    temp = _entity_temp(entity, config.ambient_temp)

    for eff in effects:
        seed = f"{eid}_{tick}_{reaction_id}_{eff.effect_type}_{eff.tag}"

        if eff.effect_type == "damage":
            amount = int(eff.value or 1)
            if amount > 0:
                transitions.append(Transition(
                    kind=TransitionKind.ENTITY_HEALTH_CHANGED,
                    payload={
                        "entity_id": eid,
                        "delta": -amount,
                        "cause": f"physics_{reaction_id}",
                        "actor": eid,
                    },
                ))

        elif eff.effect_type == "heal":
            amount = int(eff.value or 1)
            if amount > 0:
                transitions.append(Transition(
                    kind=TransitionKind.ENTITY_HEALTH_CHANGED,
                    payload={
                        "entity_id": eid,
                        "delta": amount,
                        "cause": f"physics_{reaction_id}",
                        "actor": eid,
                    },
                ))

        elif eff.effect_type == "add_tag":
            if eff.tag and eff.tag not in entity.tags:
                transitions.append(Transition(
                    kind=TransitionKind.ENTITY_CONDITION_CHANGED,
                    payload={
                        "entity_id": eid,
                        "condition": f"tag_{eff.tag}",
                        "ticks": 9999,
                        "_tag_add": eff.tag,
                    },
                ))

        elif eff.effect_type == "remove_tag":
            if eff.tag and eff.tag in entity.tags:
                transitions.append(Transition(
                    kind=TransitionKind.ENTITY_CONDITION_CHANGED,
                    payload={
                        "entity_id": eid,
                        "condition": f"tag_{eff.tag}",
                        "ticks": 0,
                        "_tag_remove": eff.tag,
                    },
                ))

        elif eff.effect_type == "adjust_prop":
            prop = eff.prop or "temperature"
            delta = float(eff.delta or 0)
            if prop == "temperature":
                from .substance import apply_temperature_delta

                delta = apply_temperature_delta(entity, delta, config)
            else:
                entity.meta[prop] = float(entity.meta.get(prop, 0)) + delta
            transitions.append(Transition(
                kind=TransitionKind.ENTITY_HEALTH_CHANGED,
                payload={
                    "entity_id": eid,
                    "delta": 0,
                    "prop_delta": {prop: delta},
                    "cause": f"physics_{reaction_id}",
                    "actor": eid,
                },
            ))

        elif eff.effect_type == "cap_prop":
            current = float(entity.meta.get(eff.prop, 0))
            if eff.max_value is not None and current > eff.max_value:
                transitions.append(Transition(
                    kind=TransitionKind.ENTITY_HEALTH_CHANGED,
                    payload={
                        "entity_id": eid,
                        "delta": 0,
                        "prop_set": {eff.prop: eff.max_value},
                        "cause": f"physics_{reaction_id}",
                        "actor": eid,
                    },
                ))

        elif eff.effect_type == "damage_by_excess":
            # damage proportional to (temperature - threshold)
            val = float(entity.meta.get(eff.prop or "temperature",
                                        config.ambient_temp))
            threshold = eff.threshold or 0.0
            if val > threshold:
                excess = val - threshold
                damage = max(1, int(excess * (eff.rate or 0.05)))
                transitions.append(Transition(
                    kind=TransitionKind.ENTITY_HEALTH_CHANGED,
                    payload={
                        "entity_id": eid,
                        "delta": -damage,
                        "cause": f"physics_{reaction_id}",
                        "actor": eid,
                    },
                ))

        elif eff.effect_type == "damage_by_deficit":
            # damage proportional to (threshold - temperature)
            val = float(entity.meta.get(eff.prop or "temperature",
                                        config.ambient_temp))
            threshold = eff.threshold or 0.0
            if val < threshold:
                deficit = threshold - val
                damage = max(1, int(deficit * (eff.rate or 0.05)))
                transitions.append(Transition(
                    kind=TransitionKind.ENTITY_HEALTH_CHANGED,
                    payload={
                        "entity_id": eid,
                        "delta": -damage,
                        "cause": f"physics_{reaction_id}",
                        "actor": eid,
                    },
                ))

        elif eff.effect_type == "spread_tag":
            # Spread a tag to adjacent entities that have requires_tag
            neighbors = _neighbor_entities(entity.position, grid)
            for n in neighbors:
                if eff.requires_tag and eff.requires_tag not in n.tags:
                    continue
                if eff.add_tag and eff.add_tag in n.tags:
                    continue  # already has it
                roll = _seeded_float(f"{seed}_{n.entity_id}")
                if roll < eff.chance:
                    transitions.append(Transition(
                        kind=TransitionKind.ENTITY_CONDITION_CHANGED,
                        payload={
                            "entity_id": str(n.entity_id),
                            "condition": f"tag_{eff.add_tag}",
                            "ticks": 9999,
                            "_tag_add": eff.add_tag,
                        },
                    ))

        elif eff.effect_type == "conduct_heat":
            # Transfer fraction of temperature difference to each neighbor
            rate = eff.rate or config.heat_spread
            neighbors = _neighbor_entities(entity.position, grid)
            for n in neighbors:
                n_temp = _entity_temp(n, config.ambient_temp)
                diff = temp - n_temp
                if abs(diff) < 1.0:
                    continue
                transfer = diff * rate
                # Heat leaves this entity, enters neighbor
                transitions.append(Transition(
                    kind=TransitionKind.ENTITY_HEALTH_CHANGED,
                    payload={
                        "entity_id": eid,
                        "delta": 0,
                        "prop_delta": {"temperature": -transfer},
                        "cause": "physics_heat_conduction",
                        "actor": eid,
                    },
                ))
                transitions.append(Transition(
                    kind=TransitionKind.ENTITY_HEALTH_CHANGED,
                    payload={
                        "entity_id": str(n.entity_id),
                        "delta": 0,
                        "prop_delta": {"temperature": transfer},
                        "cause": "physics_heat_conduction",
                        "actor": eid,
                    },
                ))

        elif eff.effect_type == "add_condition":
            transitions.append(Transition(
                kind=TransitionKind.ENTITY_CONDITION_CHANGED,
                payload={
                    "entity_id": eid,
                    "condition": eff.condition_name or "unknown",
                    "ticks": eff.ticks,
                },
            ))

        elif eff.effect_type == "remove_condition":
            transitions.append(Transition(
                kind=TransitionKind.ENTITY_CONDITION_CHANGED,
                payload={
                    "entity_id": eid,
                    "condition": eff.condition_name or "unknown",
                    "ticks": 0,
                },
            ))

        elif eff.effect_type == "change_material":
            if eff.material:
                transitions.append(Transition(
                    kind=TransitionKind.ENTITY_HEALTH_CHANGED,
                    payload={
                        "entity_id": eid,
                        "delta": 0,
                        "prop_set": {"material": eff.material},
                        "cause": f"physics_{reaction_id}",
                        "actor": eid,
                    },
                ))

    return transitions


# ---------------------------------------------------------------------------
# Temperature normalisation helper
# ---------------------------------------------------------------------------

def _temperature_approaches_ambient(
    entity: EntityState,
    config: PhysicsConfig,
    rate: float = 0.02,
) -> list[Transition]:
    """
    Slowly bring temperature back toward ambient when no other forces act.
    This gives the world a natural "cooling off" behavior.
    """
    temp = _entity_temp(entity, config.ambient_temp)
    diff = config.ambient_temp - temp
    if abs(diff) < 0.5:
        return []
    delta = diff * rate
    return [Transition(
        kind=TransitionKind.ENTITY_HEALTH_CHANGED,
        payload={
            "entity_id": str(entity.entity_id),
            "delta": 0,
            "prop_delta": {"temperature": delta},
            "cause": "physics_ambient_cooling",
            "actor": str(entity.entity_id),
        },
    )]


# ---------------------------------------------------------------------------
# Main physics tick
# ---------------------------------------------------------------------------

def physics_tick(
    world: WorldState,
    config: PhysicsConfig,
) -> list[Transition]:
    """
    Run one physics tick over the active region.

    Evaluation order (phased, replay-stable):
      0. Queued impulses (spells, items) from this tick
      1. Material-derived tag injection (ignition / freeze thresholds)
      2. Reaction rules (reactions.yaml), per entity then per object
      3. Ambient temperature normalisation

    All effects are collected as Transitions and returned.  The caller
    (game_loop.py) applies them via apply_transitions().
    """
    from .substance import drain_impulse_queue

    grid = world.spatial
    transitions: list[Transition] = []
    tick = world.tick

    # Phase 0 — impulses queued during entity actions / spells
    transitions.extend(drain_impulse_queue(world, config))

    # Phase 0.5 — Gravity & Falling physics for 3D/voxel worlds
    if grid.depth > 1:
        from .fields import MEDIA_TILE_GROUND, build_field_transition

        def _get_gravity_support(c: Coord) -> Coord:
            for z in range(c.z, 0, -1):
                below = Coord(x=c.x, y=c.y, z=z-1)
                if grid.is_solid(below):
                    return Coord(x=c.x, y=c.y, z=z)
            return Coord(x=c.x, y=c.y, z=0)

        # 1. Entity Gravity (Falling)
        for eid, entity in list(grid.entities.items()):
            if not entity.alive:
                continue
            if entity.position.z > 0:
                support = _get_gravity_support(entity.position)
                if support.z < entity.position.z:
                    height = entity.position.z - support.z
                    transitions.append(Transition(
                        kind=TransitionKind.ENTITY_MOVED,
                        payload={
                            "entity_id": str(eid),
                            "from": {"x": entity.position.x, "y": entity.position.y, "z": entity.position.z},
                            "to": {"x": support.x, "y": support.y, "z": support.z},
                            "cause": "gravity_fall",
                        }
                    ))
                    entity.position = support  # update in place
                    # Apply fall damage
                    if height > 1:
                        damage = (height - 1) * 15
                        transitions.append(Transition(
                            kind=TransitionKind.ENTITY_HEALTH_CHANGED,
                            payload={
                                "entity_id": str(eid),
                                "delta": -damage,
                                "cause": "fall_damage",
                            }
                        ))

        # 2. Object Gravity (Falling, shattering, fluid spill & splash burns)
        for oid, obj in list(grid.objects.items()):
            if obj.position is None or obj.position.z <= 0:
                continue
            support = _get_gravity_support(obj.position)
            if support.z < obj.position.z:
                height = obj.position.z - support.z
                transitions.append(Transition(
                    kind=TransitionKind.ITEM_TRANSFERRED,
                    payload={
                        "object_id": str(oid),
                        "to_position": {
                            "x": support.x,
                            "y": support.y,
                            "z": support.z,
                        },
                        "cause": "gravity_fall",
                    }
                ))
                obj.position = support  # update in-place

                # Check for fragile shattering
                name_lower = obj.name.lower()
                is_fragile = (
                    "fragile" in obj.tags
                    or "glass" in obj.tags
                    or any(k in name_lower for k in ("flask", "bottle", "vial", "potion"))
                )
                if is_fragile and height >= 1:
                    transitions.append(Transition(
                        kind=TransitionKind.ITEM_DESTROYED,
                        payload={
                            "object_id": str(oid),
                            "cause": "shattered_on_impact",
                        }
                    ))
                    # Extract fluid contents for spilling
                    fluid = obj.meta.get("fluid", {}) or {}
                    fluid_mat = fluid.get("material") or ""
                    fluid_vol = float(fluid.get("volume_ml") or 0.0)

                    if fluid_vol > 0 and fluid_mat:
                        transitions.append(build_field_transition(
                            MEDIA_TILE_GROUND,
                            fluid_mat.lower(),
                            fluid_vol,
                            coord=support,
                            operation="set",
                            cause="shattered_on_impact",
                            tick=tick,
                        ))

                        # Splash damage and wetness
                        for ent_id, ent in list(grid.entities.items()):
                            if not ent.alive:
                                continue
                            if ent.position == support:
                                # Apply wetness tag
                                transitions.append(Transition(
                                    kind=TransitionKind.ENTITY_TAG_CHANGED,
                                    payload={
                                        "entity_id": str(ent_id),
                                        "add": "wet",
                                        "cause": "fluid_splash",
                                    }
                                ))
                                ent.tags = list(ent.tags)
                                if "wet" not in ent.tags:
                                    ent.tags.append("wet")
                                ent.meta["wet_material"] = fluid_mat.lower()

                                # Hot / burning oil burns
                                temp = float(obj.meta.get("temperature", config.ambient_temp))
                                is_burning = "on_fire" in obj.tags
                                if temp >= 80.0 or is_burning:
                                    damage = int(10 * height + 10)
                                    transitions.append(Transition(
                                        kind=TransitionKind.ENTITY_HEALTH_CHANGED,
                                        payload={
                                            "entity_id": str(ent_id),
                                            "delta": -damage,
                                            "cause": "splash_burn",
                                        }
                                    ))
                                    if is_burning and fluid_mat.lower() in ("oil", "lamp_oil", "alcohol"):
                                        transitions.append(Transition(
                                            kind=TransitionKind.ENTITY_TAG_CHANGED,
                                            payload={
                                                "entity_id": str(ent_id),
                                                "add": "on_fire",
                                                "cause": "splash_ignition",
                                            }
                                        ))
                                        if "on_fire" not in ent.tags:
                                            ent.tags.append("on_fire")
                    grid.objects.pop(oid, None)  # destroy in place

    for eid, entity in list(grid.entities.items()):
        if not entity.alive:
            continue

        transitions.extend(_check_material_thresholds(entity, config, tick))

        for reaction in config.reactions:
            if not _check_conditions(entity, grid, reaction.conditions, config):
                continue
            prob_seed = f"{eid}_{tick}_{reaction.reaction_id}"
            if _seeded_float(prob_seed) > reaction.probability:
                continue
            transitions.extend(
                _apply_effects(entity, grid, reaction.effects, config, tick,
                                reaction.reaction_id)
            )

        transitions.extend(_temperature_approaches_ambient(entity, config))

    # Ground objects (hearth, oil lamps, spilled ale)
    for oid, obj in list(grid.objects.items()):
        if obj.position is None:
            continue
        transitions.extend(_physics_tick_object(obj, grid, config, tick))

    transitions.extend(_propagate_fire_to_tiles(grid, tick))

    return transitions


def _propagate_fire_to_tiles(grid: SpatialGrid, tick: int) -> list[Transition]:
    """
    Mirror burning entities/objects onto their tile tags so field_tick
    tile_reactions (fire → smoke) can run on the shared substrate.
    """
    from .schemas import TransitionKind

    transitions: list[Transition] = []
    burning_coords: set[tuple[int, int, int]] = set()

    for entity in grid.entities.values():
        if entity.alive and "on_fire" in entity.tags and entity.position is not None:
            p = entity.position
            burning_coords.add((p.x, p.y, p.z))

    for obj in grid.objects.values():
        if "on_fire" in obj.tags and obj.position is not None:
            p = obj.position
            burning_coords.add((p.x, p.y, p.z))

    for x, y, z in burning_coords:
        coord = Coord(x=x, y=y, z=z)
        if not grid.is_in_bounds(coord):
            continue
        tile = grid.tile_at(coord)
        if "on_fire" in {t.lower() for t in tile.tags}:
            continue
        transitions.append(Transition(
            kind=TransitionKind.STRUCTURE_MODIFIED,
            payload={
                "x": x,
                "y": y,
                "z": z,
                "add_tags": ["on_fire"],
                "cause": "physics_fire_tile",
            },
        ))

    return transitions


def _physics_tick_object(
    obj: ObjectState,
    grid: SpatialGrid,
    config: PhysicsConfig,
    tick: int,
) -> list[Transition]:
    """Ignition / cooling for ground objects (hearth, lamps, spills)."""
    transitions: list[Transition] = []
    mat = _object_material(obj, config)
    if mat is None:
        return transitions

    oid = str(obj.object_id)
    temp = float(obj.meta.get("temperature", config.ambient_temp))

    if mat.ignition_point is not None:
        flammable = "flammable" in obj.tags or "flammable" in mat.tags
        if flammable and temp >= mat.ignition_point and "on_fire" not in obj.tags:
            obj.tags.append("on_fire")
            transitions.append(Transition(
                kind=TransitionKind.ENTITY_CONDITION_CHANGED,
                payload={
                    "entity_id": oid,
                    "object_id": oid,
                    "condition": "tag_on_fire",
                    "ticks": 9999,
                    "_tag_add": "on_fire",
                    "_object": True,
                    "cause": "physics_ignition",
                },
            ))
        elif temp < mat.ignition_point * 0.5 and "on_fire" in obj.tags:
            obj.tags.remove("on_fire")
            transitions.append(Transition(
                kind=TransitionKind.ENTITY_CONDITION_CHANGED,
                payload={
                    "entity_id": oid,
                    "object_id": oid,
                    "condition": "tag_on_fire",
                    "ticks": 0,
                    "_tag_remove": "on_fire",
                    "_object": True,
                    "cause": "physics_extinguish",
                },
            ))

    # Radiate heat when burning (feeds entity reactions via adjacent_tag)
    if "on_fire" in obj.tags:
        obj.meta["temperature"] = max(temp, mat.ignition_point or 300) + 15
        obj.meta["fire_suppressant"] = False

    # Slow drift toward ambient
    amb = config.ambient_temp
    if "on_fire" not in obj.tags and abs(temp - amb) > 0.5:
        delta = (amb - temp) * config.heat_spread
        if abs(delta) >= 0.1:
            obj.meta["temperature"] = temp + delta

    return transitions


# ---------------------------------------------------------------------------
# Material threshold tag injection
# ---------------------------------------------------------------------------

def _check_material_thresholds(
    entity: EntityState,
    config: PhysicsConfig,
    tick: int,
) -> list[Transition]:
    """
    Add or remove tags based on the entity's temperature vs material thresholds.

    Examples:
      • entity above ignition_point + "flammable" tag → add "on_fire"
      • entity below freezing + "liquid" tag          → add "frozen"
      • entity above boiling_point + "liquid" tag     → add "boiling"
    """
    mat = _entity_material(entity, config)
    if mat is None:
        return []

    temp = _entity_temp(entity, config.ambient_temp)
    transitions: list[Transition] = []
    eid = str(entity.entity_id)

    # Ignition
    if mat.ignition_point is not None:
        if temp >= mat.ignition_point and "on_fire" not in entity.tags:
            transitions.append(Transition(
                kind=TransitionKind.ENTITY_CONDITION_CHANGED,
                payload={
                    "entity_id": eid,
                    "condition": "tag_on_fire",
                    "ticks": 9999,
                    "_tag_add": "on_fire",
                },
            ))
        elif temp < mat.ignition_point * 0.5 and "on_fire" in entity.tags:
            # Fire goes out when sufficiently cooled
            transitions.append(Transition(
                kind=TransitionKind.ENTITY_CONDITION_CHANGED,
                payload={
                    "entity_id": eid,
                    "condition": "tag_on_fire",
                    "ticks": 0,
                    "_tag_remove": "on_fire",
                },
            ))

    # Freezing (liquids)
    if mat.melting_point is not None and "liquid" in mat.tags:
        if temp <= mat.melting_point and "frozen" not in entity.tags:
            transitions.append(Transition(
                kind=TransitionKind.ENTITY_CONDITION_CHANGED,
                payload={
                    "entity_id": eid,
                    "condition": "tag_frozen",
                    "ticks": 9999,
                    "_tag_add": "frozen",
                },
            ))
        elif temp > mat.melting_point + 5 and "frozen" in entity.tags:
            transitions.append(Transition(
                kind=TransitionKind.ENTITY_CONDITION_CHANGED,
                payload={
                    "entity_id": eid,
                    "condition": "tag_frozen",
                    "ticks": 0,
                    "_tag_remove": "frozen",
                },
            ))

    # Hypothermia / frostbite for biological entities
    if "biological" in entity.tags:
        if temp <= -10 and "hypothermic" not in entity.tags:
            transitions.append(Transition(
                kind=TransitionKind.ENTITY_CONDITION_CHANGED,
                payload={
                    "entity_id": eid,
                    "condition": "hypothermic",
                    "ticks": 5,
                },
            ))
        if temp >= 40 and "overheated" not in entity.tags:
            transitions.append(Transition(
                kind=TransitionKind.ENTITY_CONDITION_CHANGED,
                payload={
                    "entity_id": eid,
                    "condition": "overheated",
                    "ticks": 3,
                },
            ))

    # Material heat/cold damage (scales with distance past threshold)
    if mat.damage_above is not None and temp > mat.damage_above:
        excess = temp - mat.damage_above
        dmg = max(1, int(excess * mat.damage_per_degree))
        transitions.append(Transition(
            kind=TransitionKind.ENTITY_HEALTH_CHANGED,
            payload={
                "entity_id": eid,
                "delta": -dmg,
                "cause": "physics_heat",
                "actor": eid,
            },
        ))

    if mat.damage_below is not None and temp < mat.damage_below:
        deficit = mat.damage_below - temp
        dmg = max(1, int(deficit * mat.damage_per_degree))
        transitions.append(Transition(
            kind=TransitionKind.ENTITY_HEALTH_CHANGED,
            payload={
                "entity_id": eid,
                "delta": -dmg,
                "cause": "physics_cold",
                "actor": eid,
            },
        ))

    return transitions


# ---------------------------------------------------------------------------
# Narrative summary of physics events
# ---------------------------------------------------------------------------

def summarize_physics_transitions(
    transitions: list[Transition],
    grid: SpatialGrid,
) -> list[str]:
    """
    Produce human-readable strings describing significant physics events.

    Filters out minor/silent changes (small temperature deltas, ambient
    cooling) and returns only noteworthy events (fire damage, ignition,
    freezing, spread fire, etc.).
    """
    lines: list[str] = []

    def ename(eid_str: str) -> str:
        from .schemas import EntityId, ObjectId
        e = grid.entities.get(EntityId(eid_str))
        if e:
            return e.name
        o = grid.objects.get(ObjectId(eid_str))
        if o:
            return o.name
        return eid_str

    # Group by entity to avoid repeating
    fire_damage: dict[str, int] = {}
    fire_spread_to: list[str] = []
    newly_ignited: list[str] = []
    newly_frozen: list[str] = []
    fire_extinguished: list[str] = []
    cold_damage: dict[str, int] = {}
    heat_damage: dict[str, int] = {}

    for t in transitions:
        p = t.payload
        cause = p.get("cause", "")

        if t.kind == TransitionKind.ENTITY_HEALTH_CHANGED:
            delta = p.get("delta", 0)
            eid = p.get("entity_id", "")
            if "fire" in cause and delta < 0:
                fire_damage[eid] = fire_damage.get(eid, 0) + abs(delta)
            elif "cold" in cause or "hypotherm" in cause and delta < 0:
                cold_damage[eid] = cold_damage.get(eid, 0) + abs(delta)
            elif "heat" in cause and delta < 0:
                heat_damage[eid] = heat_damage.get(eid, 0) + abs(delta)

        elif t.kind == TransitionKind.ENTITY_CONDITION_CHANGED:
            eid = p.get("entity_id", "")
            tag_add = p.get("_tag_add")
            tag_rem = p.get("_tag_remove")
            if tag_add == "on_fire":
                # Distinguish: ignited by thresholds (physics) vs spread
                newly_ignited.append(eid)
            elif tag_rem == "on_fire":
                fire_extinguished.append(eid)
            elif tag_add == "frozen":
                newly_frozen.append(eid)

    for eid, dmg in fire_damage.items():
        lines.append(f"{ename(eid)} burns ({dmg} damage).")
    for eid, dmg in heat_damage.items():
        lines.append(f"{ename(eid)} is scorched by extreme heat ({dmg} damage).")
    for eid, dmg in cold_damage.items():
        lines.append(f"{ename(eid)} suffers from the cold ({dmg} damage).")
    for eid in newly_ignited:
        lines.append(f"{ename(eid)} catches fire!")
    for eid in fire_extinguished:
        lines.append(f"The fire on {ename(eid)} goes out.")
    for eid in newly_frozen:
        lines.append(f"{ename(eid)} freezes solid.")

    return lines


# ---------------------------------------------------------------------------
# YAML parsing helpers (called by world_loader)
# ---------------------------------------------------------------------------

def build_physics_config(
    materials_raw: dict,
    reactions_raw: list,
    *,
    ambient_temp: float = 20.0,
    heat_spread: float = 0.05,
) -> PhysicsConfig:
    """
    Parse raw YAML dicts into a PhysicsConfig.

    Called by world_loader._load_physics().
    """
    materials: dict[str, MaterialDef] = {}
    for name, raw in (materials_raw or {}).items():
        if not isinstance(raw, dict):
            continue
        materials[name] = MaterialDef(
            name=name,
            display=str(raw.get("display", name)),
            tags=list(raw.get("tags") or []),
            density=float(raw.get("density", 1.0)),
            conductivity=float(raw.get("conductivity", 0.1)),
            ignition_point=_opt_float(raw.get("ignition_point")),
            melting_point=_opt_float(raw.get("melting_point")),
            boiling_point=_opt_float(raw.get("boiling_point")),
            damage_above=_opt_float(raw.get("damage_above")),
            damage_below=_opt_float(raw.get("damage_below")),
            damage_per_degree=float(raw.get("damage_per_degree", 0.05)),
            ph=float(raw.get("ph", 7.0)),
            hardness=int(raw.get("hardness", 3)),
            freeze_into=raw.get("freeze_into"),
            melt_into=raw.get("melt_into"),
            burn_into=raw.get("burn_into"),
        )

    # Build priority-ordered tag → material mapping
    # (more specific tags win over generic ones)
    tag_material: list[tuple[str, str]] = []
    _TAG_PRIORITY = [
        ("biological", "flesh"),
        ("undead", "bone"),
        ("metal", "iron"),
        ("wooden", "wood"),
        ("stone", "stone"),
        ("cloth", "cloth"),
        ("leather", "leather"),
        ("liquid", "water"),
        ("oil", "oil"),
        ("glass", "glass"),
    ]
    for tag, mat_name in _TAG_PRIORITY:
        if mat_name in materials:
            tag_material.append((tag, mat_name))

    reactions: list[ReactionDef] = []
    for raw in (reactions_raw or []):
        if not isinstance(raw, dict):
            continue
        conds = [_parse_condition(c) for c in (raw.get("conditions") or [])]
        effs = [_parse_effect(e) for e in (raw.get("effects") or [])]
        reactions.append(ReactionDef(
            reaction_id=str(raw.get("id", "unknown")),
            name=str(raw.get("name", "")),
            conditions=[c for c in conds if c is not None],
            effects=[e for e in effs if e is not None],
            probability=float(raw.get("probability", 1.0)),
            narrative=raw.get("narrative"),
            silent=bool(raw.get("silent", False)),
        ))

    return PhysicsConfig(
        materials=materials,
        reactions=reactions,
        ambient_temp=ambient_temp,
        heat_spread=heat_spread,
        tag_material=tag_material,
    )


def _opt_float(val: Any) -> Optional[float]:
    if val is None or val == "null":
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _parse_condition(raw: Any) -> Optional[ReactionCondition]:
    """Parse one condition entry from the reactions YAML."""
    if not isinstance(raw, dict):
        return None
    if "has_tag" in raw:
        return ReactionCondition(cond_type="has_tag", tag=str(raw["has_tag"]))
    if "missing_tag" in raw:
        return ReactionCondition(cond_type="missing_tag", tag=str(raw["missing_tag"]))
    if "has_material_tag" in raw:
        return ReactionCondition(cond_type="has_material_tag", tag=str(raw["has_material_tag"]))
    if "prop_above" in raw:
        sub = raw["prop_above"]
        return ReactionCondition(
            cond_type="prop_above",
            prop=str(sub.get("prop", "temperature")),
            value=float(sub.get("value", 0)),
        )
    if "prop_below" in raw:
        sub = raw["prop_below"]
        return ReactionCondition(
            cond_type="prop_below",
            prop=str(sub.get("prop", "temperature")),
            value=float(sub.get("value", 0)),
        )
    if "adjacent_tag" in raw:
        return ReactionCondition(cond_type="adjacent_tag", tag=str(raw["adjacent_tag"]))
    return None


def _parse_effect(raw: Any) -> Optional[ReactionEffect]:
    """Parse one effect entry from the reactions YAML."""
    if not isinstance(raw, dict):
        return None
    etype = next(iter(raw))
    val = raw[etype]

    if etype == "damage":
        return ReactionEffect(effect_type="damage", value=float(val))
    if etype == "heal":
        return ReactionEffect(effect_type="heal", value=float(val))
    if etype == "add_tag":
        return ReactionEffect(effect_type="add_tag", tag=str(val))
    if etype == "remove_tag":
        return ReactionEffect(effect_type="remove_tag", tag=str(val))
    if etype == "adjust_prop":
        return ReactionEffect(
            effect_type="adjust_prop",
            prop=str(val.get("prop", "temperature")),
            delta=float(val.get("delta", 0)),
        )
    if etype == "cap_prop":
        return ReactionEffect(
            effect_type="cap_prop",
            prop=str(val.get("prop", "temperature")),
            max_value=float(val.get("max", 0)),
        )
    if etype == "damage_by_excess":
        return ReactionEffect(
            effect_type="damage_by_excess",
            prop=str(val.get("prop", "temperature")),
            threshold=float(val.get("threshold", 0)),
            rate=float(val.get("rate", 0.05)),
        )
    if etype == "damage_by_deficit":
        return ReactionEffect(
            effect_type="damage_by_deficit",
            prop=str(val.get("prop", "temperature")),
            threshold=float(val.get("threshold", 0)),
            rate=float(val.get("rate", 0.05)),
        )
    if etype == "spread_tag":
        return ReactionEffect(
            effect_type="spread_tag",
            requires_tag=val.get("requires_tag"),
            add_tag=str(val.get("add_tag", "")),
            chance=float(val.get("chance", 0.15)),
        )
    if etype == "conduct_heat":
        return ReactionEffect(
            effect_type="conduct_heat",
            rate=float(val) if isinstance(val, (int, float)) else float(val.get("rate", 0.05)),
        )
    if etype == "add_condition":
        return ReactionEffect(
            effect_type="add_condition",
            condition_name=str(val.get("name", val) if isinstance(val, dict) else val),
            ticks=int(val.get("ticks", 3) if isinstance(val, dict) else 3),
        )
    if etype == "remove_condition":
        return ReactionEffect(
            effect_type="remove_condition",
            condition_name=str(val),
        )
    if etype == "change_material":
        return ReactionEffect(effect_type="change_material", material=str(val))
    return None
