"""
M0: Canonical data contracts for the semantic simulation runtime.

Architecture invariant: nothing upstream of the Compiler ever touches
WorldState directly.  Every layer receives only the types it is entitled
to read.  The LM adapter receives only SemanticProjection.
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, NewType, Optional, Union

from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Primitive identifiers
# ---------------------------------------------------------------------------

EntityId = NewType("EntityId", str)
ObjectId = NewType("ObjectId", str)
FactionId = NewType("FactionId", str)
EventId = NewType("EventId", str)


def new_entity_id() -> EntityId:
    return EntityId(f"ent_{uuid.uuid4().hex[:8]}")


def new_object_id() -> ObjectId:
    return ObjectId(f"obj_{uuid.uuid4().hex[:8]}")


def new_event_id() -> EventId:
    return EventId(f"evt_{uuid.uuid4().hex[:8]}")


# ---------------------------------------------------------------------------
# Spatial primitives
# ---------------------------------------------------------------------------


def coord_key(coord: "Coord") -> str:
    """Canonical storage key for tiles, voxels, and field layers."""
    return f"{coord.x},{coord.y},{coord.z}"


def parse_coord_key(key: str) -> "Coord":
    parts = key.split(",")
    if len(parts) == 2:
        return Coord(x=int(parts[0]), y=int(parts[1]), z=0)
    if len(parts) == 3:
        return Coord(x=int(parts[0]), y=int(parts[1]), z=int(parts[2]))
    raise ValueError(f"Invalid coord key: {key!r}")


class Coord(BaseModel):
    """Integer voxel/grid coordinate (x, y horizontal; z vertical)."""

    x: int
    y: int
    z: int = 0

    def __hash__(self) -> int:
        return hash((self.x, self.y, self.z))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Coord):
            return NotImplemented
        return self.x == other.x and self.y == other.y and self.z == other.z

    def manhattan(self, other: "Coord") -> int:
        return (
            abs(self.x - other.x)
            + abs(self.y - other.y)
            + abs(self.z - other.z)
        )

    def chebyshev(self, other: "Coord") -> int:
        return max(
            abs(self.x - other.x),
            abs(self.y - other.y),
            abs(self.z - other.z),
        )

    def neighbors(self) -> list["Coord"]:
        """Four horizontal neighbors (2D-compatible movement)."""
        return [
            Coord(x=self.x + dx, y=self.y + dy, z=self.z)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
        ]

    def neighbors_6(self) -> list["Coord"]:
        """Six face-adjacent voxels (±x, ±y, ±z)."""
        out: list[Coord] = []
        for dx, dy, dz in (
            (-1, 0, 0), (1, 0, 0),
            (0, -1, 0), (0, 1, 0),
            (0, 0, -1), (0, 0, 1),
        ):
            out.append(Coord(x=self.x + dx, y=self.y + dy, z=self.z + dz))
        return out

    def neighbors_26(self) -> list["Coord"]:
        """All voxels in a 3×3×3 neighbourhood excluding self."""
        out: list[Coord] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == dy == dz == 0:
                        continue
                    out.append(
                        Coord(x=self.x + dx, y=self.y + dy, z=self.z + dz)
                    )
        return out

    def offset(self, dx: int = 0, dy: int = 0, dz: int = 0) -> "Coord":
        return Coord(x=self.x + dx, y=self.y + dy, z=self.z + dz)

    def as_2d(self) -> "Coord":
        """Horizontal slice key (z forced to 0) for legacy tile lookups."""
        return Coord(x=self.x, y=self.y, z=0)

    def __repr__(self) -> str:
        if self.z == 0:
            return f"({self.x},{self.y})"
        return f"({self.x},{self.y},{self.z})"


class VoxelMaterial(str, Enum):
    """Canonical voxel substance — drives colour, solidity, and spread rules."""

    AIR = "air"
    STONE = "stone"
    BRICK = "brick"
    WOOD = "wood"
    PLANKS = "planks"
    GLASS = "glass"
    WATER = "water"
    GRASS = "grass"
    DIRT = "dirt"
    LEAVES = "leaves"
    METAL = "metal"
    EMBER = "ember"
    # Semantic build materials (tavern, dungeon, …)
    BAR = "bar"
    TABLE = "table"
    THATCH = "thatch"
    ROOF = "roof"


class VoxelCell(BaseModel):
    """One voxel in the world volume."""

    material: VoxelMaterial = VoxelMaterial.AIR
    solid: bool = False
    transparent: bool = True
    emissive: float = 0.0
    tags: list[str] = Field(default_factory=list)

    @classmethod
    def air(cls) -> "VoxelCell":
        return cls(
            material=VoxelMaterial.AIR,
            solid=False,
            transparent=True,
        )

    @classmethod
    def from_material(cls, material: VoxelMaterial) -> "VoxelCell":
        presets: dict[VoxelMaterial, dict] = {
            VoxelMaterial.AIR: {"solid": False, "transparent": True},
            VoxelMaterial.STONE: {"solid": True, "transparent": False},
            VoxelMaterial.BRICK: {"solid": True, "transparent": False},
            VoxelMaterial.WOOD: {"solid": True, "transparent": False},
            VoxelMaterial.PLANKS: {"solid": True, "transparent": False},
            VoxelMaterial.GLASS: {"solid": True, "transparent": True},
            VoxelMaterial.WATER: {"solid": False, "transparent": True, "tags": ["fluid"]},
            VoxelMaterial.GRASS: {"solid": True, "transparent": False},
            VoxelMaterial.DIRT: {"solid": True, "transparent": False},
            VoxelMaterial.LEAVES: {"solid": True, "transparent": True},
            VoxelMaterial.METAL: {"solid": True, "transparent": False},
            VoxelMaterial.EMBER: {"solid": False, "transparent": True, "emissive": 0.9},
            VoxelMaterial.BAR: {"solid": True, "transparent": False, "tags": ["furniture"]},
            VoxelMaterial.TABLE: {"solid": True, "transparent": False, "tags": ["furniture"]},
            VoxelMaterial.THATCH: {"solid": True, "transparent": False},
            VoxelMaterial.ROOF: {"solid": True, "transparent": False},
        }
        data = presets.get(material, {"solid": True, "transparent": False})
        return cls(material=material, **data)

    model_config = {"frozen": False}


# ---------------------------------------------------------------------------
# Facing direction (8-way compass)
# ---------------------------------------------------------------------------

# (dx, dy) vector for each compass direction.
# +x = east, +y = south  (screen convention used throughout the engine).
FACING_VECTORS: dict[str, tuple[int, int]] = {
    "north":     ( 0, -1),
    "northeast": ( 1, -1),
    "east":      ( 1,  0),
    "southeast": ( 1,  1),
    "south":     ( 0,  1),
    "southwest": (-1,  1),
    "west":      (-1,  0),
    "northwest": (-1, -1),
}
# Reverse map: vector → direction name
_VEC_TO_FACING: dict[tuple[int, int], str] = {v: k for k, v in FACING_VECTORS.items()}


class FacingDirection(str, Enum):
    """Cardinal + diagonal directions an entity can face.

    The isometric renderer draws an arrow in this direction.
    ``visible_from()`` restricts sight to a cone centred on this direction.
    """

    NORTH     = "north"
    NORTHEAST = "northeast"
    EAST      = "east"
    SOUTHEAST = "southeast"
    SOUTH     = "south"
    SOUTHWEST = "southwest"
    WEST      = "west"
    NORTHWEST = "northwest"

    @property
    def vector(self) -> tuple[int, int]:
        return FACING_VECTORS[self.value]

    @property
    def opposite(self) -> "FacingDirection":
        dx, dy = self.vector
        return FacingDirection(_VEC_TO_FACING[(-dx, -dy)])

    def turn(self, steps: int) -> "FacingDirection":
        """Rotate clockwise by ``steps`` × 45°.  Negative = counterclockwise."""
        dirs = list(FacingDirection)
        idx = dirs.index(self)
        return dirs[(idx + steps) % 8]

    @staticmethod
    def from_delta(dx: int, dy: int) -> "Optional[FacingDirection]":
        """Infer direction from a movement delta.  Returns None for (0,0)."""
        if dx == 0 and dy == 0:
            return None
        nx = 0 if dx == 0 else (1 if dx > 0 else -1)
        ny = 0 if dy == 0 else (1 if dy > 0 else -1)
        name = _VEC_TO_FACING.get((nx, ny))
        return FacingDirection(name) if name else None


class TerrainType(str, Enum):
    FLOOR = "floor"
    WALL = "wall"
    DOOR_OPEN = "door_open"
    DOOR_CLOSED = "door_closed"
    WINDOW = "window"
    WATER = "water"
    STAIRS_UP = "stairs_up"
    STAIRS_DOWN = "stairs_down"


class Tile(BaseModel):
    terrain: TerrainType = TerrainType.FLOOR
    passable: bool = True
    transparent: bool = True
    # Named tags for semantic projection (e.g. "cover", "exit", "hazard")
    tags: list[str] = Field(default_factory=list)
    # Free-form marks written by player/NPC actions (chalk circle, blood,
    # rune, arrow).  Visible to anyone examining the tile.  Persists until
    # explicitly removed via TILE_MARKED with remove=true.
    marks: list[str] = Field(default_factory=list)
    # Generic environment state for this tile (locked, lit, temperature, etc.)
    # Written by ENVIRONMENT_STATE_CHANGED.  Keys are arbitrary strings.
    env: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _infer_defaults(self) -> "Tile":
        if self.terrain == TerrainType.WALL:
            object.__setattr__(self, "passable", False)
            object.__setattr__(self, "transparent", False)
        elif self.terrain == TerrainType.DOOR_CLOSED:
            object.__setattr__(self, "passable", False)
        return self

    model_config = {"frozen": False}


# ---------------------------------------------------------------------------
# Entities and objects
# ---------------------------------------------------------------------------


class EntityKind(str, Enum):
    PLAYER = "player"
    NPC = "npc"
    CREATURE = "creature"
    AMBIENT = "ambient"


class EmotionalState(str, Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    ANGRY = "angry"
    FEARFUL = "fearful"
    SUSPICIOUS = "suspicious"
    FRIENDLY = "friendly"
    HOSTILE = "hostile"
    GRIEVING = "grieving"
    PROUD = "proud"
    HUMILIATED = "humiliated"


class AlertnessLevel(str, Enum):
    UNAWARE = "unaware"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    COMBAT = "combat"


class EntityState(BaseModel):
    entity_id: EntityId
    name: str
    kind: EntityKind
    position: Coord
    health: int = 100
    max_health: int = 100
    sight_range: int = 8
    hearing_range: int = 12
    emotional_state: EmotionalState = EmotionalState.NEUTRAL
    alertness: AlertnessLevel = AlertnessLevel.LOW
    armed: bool = False
    intoxication: int = Field(default=0, ge=0, le=100)
    inventory: list[ObjectId] = Field(default_factory=list)
    # Attributes used in contested resolution (0-100 scale)
    attributes: dict[str, int] = Field(
        default_factory=lambda: {
            "strength": 50,
            "intimidation": 50,
            "persuasion": 50,
            "deception": 50,
            "perception": 50,
            "stealth": 50,
        }
    )
    tags: list[str] = Field(default_factory=list)
    # Default social openness toward strangers — interpreted by the
    # contact compiler when no specific INTIMATE / WARY_OF / ENEMY_OF
    # edge exists between actor and this entity.
    #   "welcoming"  — physical contact from acquaintances is welcomed
    #   "open"       — neutral; contact may surprise but rarely offends
    #   "guarded"    — physical contact from strangers is unwelcome
    #   "closed"     — physical contact from non-intimates is refused
    # Per-entity overrides live as INTIMATE / WARY_OF edges in the
    # relational graph.
    social_openness: str = "guarded"
    # Free-form descriptors used by the LM-driven NPC policy to ground
    # this entity's behavior. None of these strings are interpreted by
    # the engine — they are passed verbatim into the NPC character
    # sheet that primes the LM's decide() call. Empty strings mean
    # "no override; the LM will fall back on whatever it can infer
    # from kind/name/relations".
    #
    #   role        — what this entity *is* in the world ("city guard",
    #                 "tavern keeper", "stray dog"). One short phrase.
    #   personality — disposition shorthand ("stoic, professional,
    #                 suspicious of outsiders"). One sentence.
    #   drive       — what this entity is currently trying to do when
    #                 nothing acute is happening ("guard the eastern
    #                 gate", "tend the stall", "wander aimlessly").
    role: str = ""
    personality: str = ""
    drive: str = ""
    # Whether this entity is still alive.  Set to False by ENTITY_DIED.
    # Dead entities remain in the spatial grid so the event log can
    # reference them, but they are excluded from NPC turns, visibility
    # checks, and projections.
    alive: bool = True
    # The compass direction this entity is currently facing.
    # Determines the vision cone used by visible_from().
    # Defaults to SOUTH so newly spawned entities face "down the screen"
    # in the isometric view.
    facing: FacingDirection = FacingDirection.SOUTH
    # Total field-of-view angle in degrees.  135° = ±67.5° from facing;
    # 90° = ±45°; 360° = omnidirectional (legacy behaviour).
    fov_degrees: float = 135.0
    # Equipped items for combat resolution.  ObjectId must exist in
    # world.spatial.objects.  None means bare-handed / unarmored.
    equipped_weapon: Optional[ObjectId] = None
    equipped_armor: Optional[ObjectId] = None
    # Active temporary conditions: condition_name → remaining_ticks.
    # Engine-recognized names: "stunned", "poisoned", "exhausted",
    # "blinded".  All others are passed verbatim to the LM character
    # sheet.  world_clock.py decrements them by 1 each tick.
    conditions: dict[str, int] = Field(default_factory=dict)
    # Deterministic patrol route.  ReactivePolicy cycles through these
    # when the entity is idle (alertness <= LOW, no threats).
    waypoints: list[Coord] = Field(default_factory=list)
    waypoint_index: int = 0
    # Which region this entity currently occupies.
    # Must match a key in WorldState.regions.
    region_id: str = "region_main"
    # Factual knowledge this entity has about the world.  Surfaced to the LM
    # so NPC replies are grounded in simulation-authored facts rather than
    # hallucinations (e.g. "no milk in this tavern tonight").
    knowledge: list[str] = Field(default_factory=list)
    # Short-term scene goals (optional). Prefer social_aims for interpersonal drama.
    goals: list[str] = Field(default_factory=list)
    # Long-horizon plan / project this NPC is currently committed to.
    # ``None`` means the NPC is operating in pure-reactive mode (DF-style
    # job stack is empty).  When a plan is active, ``ReactivePolicy``
    # asks the planner for the next step before considering the normal
    # candidate menu, so NPCs can pursue multi-tick projects without
    # getting distracted by every small stimulus.  See ``npc_planner.py``.
    active_plan: Optional["NpcPlan"] = None
    # Who to engage this scene and what you want from them (pack-authored).
    social_aims: list[SocialAim] = Field(default_factory=list)
    # Secrets this entity holds.  Only disclosed to the LM when an INTIMATE
    # or ALLIED relational edge exists toward the projection's focal entity.
    secrets: list[str] = Field(default_factory=list)
    # Maximum total weight (kg) this entity can carry before being over-
    # burdened.  Derived from strength at world-load time if not explicit.
    # Base value is conservative so a default entity without a YAML override
    # can still carry a sword + armor without being blocked.
    carry_capacity: float = 50.0
    # Equipped items keyed by body slot name.  Canonical slot names:
    #   main_hand, off_hand, head, face, neck, chest, back,
    #   waist, legs, feet, hands, ring_l, ring_r
    # Replaces the older equipped_weapon / equipped_armor scalars for new
    # code; both are kept in sync so existing compiler code still works.
    equipped_slots: dict[str, ObjectId] = Field(default_factory=dict)
    # Spell system ────────────────────────────────────────────────────────
    # Current and maximum mana pool.  Defaults to 0/0 (no magic) for NPCs
    # and non-magical worlds.  World loader sets these from entity YAML.
    mana: int = 0
    max_mana: int = 0
    # Set of operation names this entity has learned (by reading grimoires).
    # Spell programs referencing operations not in this set cause backlash.
    # Example: {"HEAT", "CHILL", "HARM"}
    known_spell_ops: list[str] = Field(default_factory=list)
    # Named spells inscribed in the entity's grimoire (if they carry one).
    # Keyed by spell name; value is the raw program source string.
    # Example: {"fireball": "COST mana 25, FOR EACH NEAR 3 { HARM 15 }"}
    inscribed_spells: dict[str, str] = Field(default_factory=dict)
    # ── Generic stat bag ─────────────────────────────────────────────────
    # Any numeric property that isn't special-cased elsewhere lives here.
    # Defaults include economy and exertion stats so NPCs and the player
    # have parity on non-combat resources from the start.
    #
    #   stamina     — energy for sustained actions (fatigue, sprinting, magic)
    #   gold        — economic currency; visible to trade/steal compilers
    #   reputation  — social standing in the world (-100 to 100)
    #
    # World packs may add arbitrary keys (hunger, sanity, influence, etc.)
    # without engine changes.  All keys are floats internally.
    stats: dict[str, float] = Field(
        default_factory=lambda: {
            "stamina": 100.0,
            "gold": 0.0,
            "reputation": 0.0,
        }
    )
    # ── Skills ────────────────────────────────────────────────────────────
    # Earned through successful action use, 0-100 float scale.
    # Standard keys: combat, arcane, social, stealth, crafting, perception.
    # Higher values shift ContestSpec outcomes and unlock skill-gated actions.
    skills: dict[str, float] = Field(default_factory=dict)
    # ── Non-combat identity ───────────────────────────────────────────────
    # occupation: what this entity does in the world ("merchant", "scholar",
    #             "blacksmith", "thief").  Drives NPC policy branching.
    # faction:    named group affiliation ("city_watch", "thieves_guild",
    #             "trade_guild").  Affects social interactions and grounding.
    occupation: str = ""
    faction: str = ""
    # World-pack authors can supply example lines for this character in entities.yaml.
    # These seed the fallback dialogue generator so NPCs stay in-voice even when
    # the LM call fails or returns generic output.
    voice_lines: list[str] = Field(default_factory=list)
    # Arbitrary extensible metadata (e.g. dialect, material)
    meta: dict[str, Any] = Field(default_factory=dict)


# Body-slot constants so caller code can use EquipSlot.MAIN_HAND instead of
# bare strings — prevents typos propagating silently.
class EquipSlot:
    MAIN_HAND = "main_hand"
    OFF_HAND  = "off_hand"
    HEAD      = "head"
    FACE      = "face"
    NECK      = "neck"
    CHEST     = "chest"
    BACK      = "back"
    WAIST     = "waist"
    LEGS      = "legs"
    FEET      = "feet"
    HANDS     = "hands"
    RING_L    = "ring_l"
    RING_R    = "ring_r"

    # Convenience set of all canonical slot names for validation.
    ALL: frozenset[str] = frozenset({
        "main_hand", "off_hand", "head", "face", "neck",
        "chest", "back", "waist", "legs", "feet", "hands",
        "ring_l", "ring_r",
    })

    # Legacy mapping: old equipped_weapon / equipped_armor fields map to these
    # canonical slots so we can bridge old data to the new system.
    WEAPON_SLOT = "main_hand"
    ARMOR_SLOT  = "chest"


class ObjectState(BaseModel):
    object_id: ObjectId
    name: str
    position: Optional[Coord] = None
    # None position means currently in inventory or inside a container.
    owner: Optional[EntityId] = None
    passable: bool = True
    transparent: bool = True
    tags: list[str] = Field(default_factory=list)
    # Numeric properties used by the combat compiler (damage, protection, …).
    # Weapon objects declare `damage: N`; armor objects declare `protection: N`.
    attributes: dict[str, int] = Field(default_factory=dict)
    # Physical properties for the inventory system.
    # weight  — mass in kg.  0.0 = weightless (coins, notes).
    # bulk    — volume in abstract units.  Affects container fit.
    # slot    — canonical EquipSlot name if this is wearable/equippable.
    #           None means the item can only be carried loose in inventory.
    # durability / max_durability — item condition (100 = pristine, 0 = broken).
    #           Not yet decremented by the engine; reserved for future use.
    weight: float = Field(default=1.0, ge=0.0)
    bulk: float = Field(default=1.0, ge=0.0)
    slot: Optional[str] = None
    durability: int = Field(default=100, ge=0, le=100)
    max_durability: int = Field(default=100, ge=1)
    # Container support.  A container can hold other items up to its own
    # carry_capacity (by weight) and bulk_capacity (by bulk).
    # contents lists ObjectIds of items currently stored inside this object.
    is_container: bool = False
    carry_capacity: float = Field(default=0.0, ge=0.0)
    bulk_capacity: float = Field(default=0.0, ge=0.0)
    contents: list[ObjectId] = Field(default_factory=list)
    # Which container this object is currently stored inside (None = loose).
    container_id: Optional[ObjectId] = None
    # Which region this object is currently in (ground objects only;
    # carried objects inherit their owner's region).
    region_id: str = "region_main"
    meta: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Spatial grid
# ---------------------------------------------------------------------------


class SpatialGrid(BaseModel):
    width: int
    height: int
    depth: int = 1
    tiles: dict[str, Tile] = Field(default_factory=dict)
    voxels: dict[str, VoxelCell] = Field(default_factory=dict)
    # When True, passability/transparency use the voxel volume first.
    use_voxels: bool = False
    entities: dict[EntityId, EntityState] = Field(default_factory=dict)
    objects: dict[ObjectId, ObjectState] = Field(default_factory=dict)
    region_env: dict[str, Any] = Field(default_factory=dict)

    def _key(self, coord: Coord) -> str:
        return coord_key(coord)

    def _tile_key(self, coord: Coord) -> str:
        """Legacy 2D tile map keys (z ignored)."""
        return f"{coord.x},{coord.y}"

    def tile_at(self, coord: Coord) -> Tile:
        return self.tiles.get(self._tile_key(coord), Tile())

    def set_tile(self, coord: Coord, tile: Tile) -> None:
        self.tiles[self._tile_key(coord)] = tile

    def voxel_at(self, coord: Coord) -> VoxelCell:
        return self.voxels.get(self._key(coord), VoxelCell.air())

    def set_voxel(self, coord: Coord, cell: VoxelCell) -> None:
        if cell.material == VoxelMaterial.AIR and not cell.solid:
            self.voxels.pop(self._key(coord), None)
        else:
            self.voxels[self._key(coord)] = cell

    def has_voxel(self, coord: Coord) -> bool:
        return self._key(coord) in self.voxels

    def is_in_bounds(self, coord: Coord) -> bool:
        return (
            0 <= coord.x < self.width
            and 0 <= coord.y < self.height
            and 0 <= coord.z < self.depth
        )

    def is_solid(self, coord: Coord) -> bool:
        if not self.is_in_bounds(coord):
            return True
        if self.use_voxels or self.has_voxel(coord):
            return self.voxel_at(coord).solid
        return not self.tile_at(coord).passable

    def is_passable(self, coord: Coord) -> bool:
        if not self.is_in_bounds(coord):
            return False
        if self.is_solid(coord):
            return False
        if not self.use_voxels and not self.tile_at(coord).passable:
            return False
        for entity in self.entities.values():
            if entity.position == coord:
                return False
        for obj in self.objects.values():
            if obj.position == coord and not obj.passable:
                return False
        return True

    def is_transparent(self, coord: Coord) -> bool:
        if not self.is_in_bounds(coord):
            return False
        if self.use_voxels or self.has_voxel(coord):
            if not self.voxel_at(coord).transparent:
                return False
        elif not self.tile_at(coord).transparent:
            return False
        for obj in self.objects.values():
            if obj.position == coord and not obj.transparent:
                return False
        return True

    def volume(self) -> int:
        return self.width * self.height * self.depth


# ---------------------------------------------------------------------------
# Multi-region world graph
# ---------------------------------------------------------------------------


class RegionType(str, Enum):
    """Semantic type of a region — used by the renderer and NPC policies."""
    INDOOR   = "indoor"
    OUTDOOR  = "outdoor"
    DUNGEON  = "dungeon"
    CAVE     = "cave"
    VOID     = "void"


class Region(BaseModel):
    """
    A self-contained spatial area: one tile grid + metadata.

    Entities and objects that occupy this region have their
    ``EntityState.region_id`` / ``ObjectState.region_id`` set to this
    region's ``region_id``.  The grid's ``entities`` / ``objects`` dicts
    are the canonical store for those items while they are here.
    """

    region_id: str
    name: str
    region_type: RegionType = RegionType.INDOOR
    # floor_level: 0 = ground, positive = above ground, negative = below.
    floor_level: int = 0
    grid: SpatialGrid
    tags: list[str] = Field(default_factory=list)
    # Brief prose description surfaced to the LM projection.
    description: str = ""


class Portal(BaseModel):
    """
    A directed (optionally bidirectional) connection between two regions.

    When an entity steps onto ``from_coord`` in ``from_region`` the compiler
    emits a ``REGION_TRANSIT`` transition, teleporting them to ``to_coord``
    in ``to_region``.  The renderer highlights portal tiles distinctively.
    """

    portal_id: str = Field(
        default_factory=lambda: f"portal_{uuid.uuid4().hex[:8]}"
    )
    label: str = ""
    from_region: str
    from_coord: Coord
    to_region: str
    to_coord: Coord
    # If True, the inverse portal is implicitly activated when the entity
    # arrives: standing on ``to_coord`` in ``to_region`` will bring them back.
    bidirectional: bool = True
    # Optional item the entity must be carrying to pass through.
    requires_key: Optional[ObjectId] = None


# ---------------------------------------------------------------------------
# Relational graph edges and nodes
# ---------------------------------------------------------------------------


class EdgeKind(str, Enum):
    FACTION_MEMBER = "faction_member"
    ENEMY_OF = "enemy_of"
    ALLY_OF = "ally_of"
    FEARS = "fears"
    RESPECTS = "respects"
    DISTRUSTS = "distrusts"
    OWES_DEBT = "owes_debt"
    REMEMBERS_EVENT = "remembers_event"
    WITNESSED = "witnessed"
    KIN = "kin"
    EMPLOYS = "employs"
    EMPLOYED_BY = "employed_by"
    # Mutual close relationship — bidirectional INTIMATE means contact
    # is generally welcomed (partners, family, close friends). A single
    # direction means feeling is unrequited; the compiler treats one-way
    # INTIMATE the same as no edge for consent purposes.
    INTIMATE = "intimate"
    # One-directional aversion to contact from a specific entity.
    # Independent of HOSTILE / ENEMY_OF (which imply outright enmity).
    WARY_OF = "wary_of"
    # Zone 2: an agent holds a belief about a claimed fact, not ground truth
    BELIEVES_CLAIM = "believes_claim"
    # Actor made a declarative threat / hostile claim against target
    THREATENED = "threatened"
    # Entity knows another entity's secret (via reveal, eavesdrop, or trade)
    KNOWS_SECRET = "knows_secret"
    # Information was deliberately exchanged between two entities
    TRADED_INFO_WITH = "traded_info_with"
    # Entity overheard something not meant for them
    OVERHEARD = "overheard"
    # Freeform interaction between two entities (shake_hands, hug, bow, …).
    # One edge per ordered pair; meta carries a rolling window of recent verbs
    # (capped at INTERACTION_HISTORY_WINDOW) so the projection can surface
    # "you have previously shaken hands with Guard" without unbounded growth.
    INTERACTED = "interacted"


class RelationalEdge(BaseModel):
    source: str  # EntityId or FactionId
    target: str
    kind: EdgeKind
    weight: float = Field(default=1.0, ge=-1.0, le=1.0)
    tick_created: int = 0
    tick_last_updated: int = 0
    # Linked event ids that justify this edge
    evidence: list[EventId] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)


class NodeKind(str, Enum):
    ENTITY = "entity"
    FACTION = "faction"
    ORGANIZATION = "organization"
    LOCATION = "location"


class RelationalNode(BaseModel):
    node_id: str
    kind: NodeKind
    name: str
    meta: dict[str, Any] = Field(default_factory=dict)


class RelationalGraph(BaseModel):
    nodes: dict[str, RelationalNode] = Field(default_factory=dict)
    # edges stored as source -> target -> edge
    edges: dict[str, dict[str, list[RelationalEdge]]] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Events and transitions
# ---------------------------------------------------------------------------


class TransitionKind(str, Enum):
    ENTITY_MOVED = "entity_moved"
    ENTITY_HEALTH_CHANGED = "entity_health_changed"
    ENTITY_EMOTIONAL_STATE_CHANGED = "entity_emotional_state_changed"
    ENTITY_ALERTNESS_CHANGED = "entity_alertness_changed"
    ITEM_TRANSFERRED = "item_transferred"
    ITEM_CREATED = "item_created"
    ITEM_DESTROYED = "item_destroyed"
    # Item wear and repair. Payload: {"object_id": str, "delta": int,
    # "cause": str, "destroy_at_zero": bool (default true)}.
    ITEM_DURABILITY_CHANGED = "item_durability_changed"
    TILE_CHANGED = "tile_changed"
    EDGE_CREATED = "edge_created"
    EDGE_UPDATED = "edge_updated"
    EDGE_REMOVED = "edge_removed"
    DIALOGUE_SPOKEN = "dialogue_spoken"
    RUMOR_PROPAGATED = "rumor_propagated"
    # CONTACT initiates physical, non-combat interaction. The payload
    # carries actor, target, manner (free-form), consent_state
    # (computed at compile time: welcomed | unwelcomed | refused), and
    # surprise (bool — true when target had no time to anticipate). The
    # target's response is produced separately by their own policy turn.
    CONTACT_INITIATED = "contact_initiated"
    # World-clock / environmental event. Emitted by the WorldClock
    # autonomic loop (NOT by any actor). Payload carries:
    #   id          — the ambient event's pack id
    #   narrative   — pre-authored prose to render
    #   time_of_day — the phase the world is in when this fired
    # Any state-changing side effects from the ambient event are emitted
    # as separate transitions alongside this record so the existing
    # apply_transitions machinery is reused.
    AMBIENT_EVENT = "ambient_event"
    # Entity health reached 0. Payload: {"entity_id": ..., "cause": str}.
    # apply_transitions marks entity.alive=False and drops inventory.
    ENTITY_DIED = "entity_died"
    # Temporary condition added/updated/cleared.
    # Payload: {"entity_id": ..., "condition": str, "ticks": int}.
    # ticks=0 means the condition is removed.
    ENTITY_CONDITION_CHANGED = "entity_condition_changed"
    # Noise propagation. Payload: {"source_id": ..., "noise_level": int}.
    # apply_transitions raises alertness of entities within hearing_range.
    NOISE_EVENT = "noise_event"
    # Entity crossed a staircase / area link. Payload:
    # {"entity_id": ..., "from_area": str, "to_area": str,
    #  "arrival": {"x": int, "y": int}}.
    AREA_TRANSITION = "area_transition"
    # Entity equipped an item into a body slot.
    # Payload: {"entity_id": ..., "object_id": ..., "slot": ...,
    #           "replaced_object_id": ... (or null if slot was empty)}.
    ITEM_EQUIPPED = "item_equipped"
    # Entity unequipped an item from a body slot back into loose inventory.
    # Payload: {"entity_id": ..., "object_id": ..., "slot": ...}.
    ITEM_UNEQUIPPED = "item_unequipped"
    # Item moved from loose inventory / ground into a container object.
    # Payload: {"entity_id": ..., "object_id": ..., "container_id": ...}.
    ITEM_STORED = "item_stored"
    # Item retrieved from a container back into loose inventory.
    # Payload: {"entity_id": ..., "object_id": ..., "container_id": ...}.
    ITEM_RETRIEVED = "item_retrieved"
    # Entity crossed a portal into a different region.
    # Payload: {"entity_id": ..., "from_region": str, "to_region": str,
    #           "arrival": {"x": int, "y": int}}.
    # apply_transitions moves the entity between region grids and updates
    # world.active_region_id when the entity is the player.
    REGION_TRANSIT = "region_transit"
    # Entity changed facing direction.
    # Payload: {"entity_id": ..., "from": str, "to": str}.
    # apply_transitions updates entity.facing.
    ENTITY_TURNED = "entity_turned"
    # Spell runtime: arbitrary numeric property on entity.meta changed.
    # Payload: {"entity_id": ..., "prop": str, "delta": float | None,
    #           "new_value": float | None, "cause": str}.
    # apply_transitions updates entity.meta[prop].
    ENTITY_PROPERTY_CHANGED = "entity_property_changed"
    # Spell runtime: tag added to or removed from entity.tags.
    # Payload: {"entity_id": ..., "tag": str, "added": bool}.
    ENTITY_TAG_CHANGED = "entity_tag_changed"
    # Spell runtime: mana pool changed.
    # Payload: {"entity_id": ..., "delta": int, "cause": str}.
    ENTITY_MANA_CHANGED = "entity_mana_changed"
    # Generic numeric stat changed.  Covers any key in EntityState.stats
    # (stamina, gold, reputation, hunger, …) and also health/mana as
    # aliases so new verbs don't need to hard-code ENTITY_HEALTH_CHANGED.
    # Payload: {"entity_id": ..., "stat": str, "delta": float,
    #           "set": float|None, "cause": str, "actor": str|None}.
    # apply_transitions: writes entity.stats[stat]; if stat=="health",
    # also mirrors into entity.health for backward compat + death check.
    ENTITY_STAT_CHANGED = "entity_stat_changed"
    # ── Open-world synthesis ────────────────────────────────────────────
    # A brand-new item created by the player/NPC from raw materials —
    # distinct from ITEM_CREATED (rule-authored) in that the object schema
    # was proposed by the LM and validated by the engine.
    # Payload: {"object_id": str, "name": str, "tags": list, "weight": float,
    #           "bulk": float, "attributes": dict, "meta": dict,
    #           "owner_entity_id": str, "cause": str}.
    ITEM_SYNTHESIZED = "item_synthesized"
    # A new faction node created mid-game from player/NPC action.
    # Payload: {"faction_id": str, "name": str, "description": str,
    #           "founding_entity_id": str, "initial_members": list[str]}.
    FACTION_CREATED = "faction_created"
    # A goal string added to an entity's goals list.
    # Payload: {"entity_id": str, "goal": str, "cause": str, "priority": float}.
    ENTITY_GOAL_ADDED = "entity_goal_added"
    # A goal removed or marked achieved.
    # Payload: {"entity_id": str, "goal": str, "resolved": bool}.
    ENTITY_GOAL_REMOVED = "entity_goal_removed"
    # A belief edge propagated from one entity to another via their
    # social connections (gossip, rumour, shared observation).
    # Payload: {"from_entity": str, "to_entity": str, "belief": str,
    #           "original_source": str, "fidelity": float (0-1)}.
    BELIEF_PROPAGATED = "belief_propagated"
    # A new persistent open-schema relational node (location, organization, concept)
    # created by player world-authoring (Zone 2 EXTENDING actions).
    # Payload: {"node_id": str, "node_kind": str, "label": str,
    #           "meta": dict, "founding_entity_id": str}.
    RELATIONAL_NODE_CREATED = "relational_node_created"
    # ── Environmental interaction ────────────────────────────────────────
    # An arbitrary string mark written onto a tile (scratch, chalk circle,
    # blood trail, rune, arrow).  The mark persists in tile.marks and is
    # visible to any entity examining that tile.
    # Payload: {"x": int, "y": int, "mark": str, "author_entity_id": str,
    #           "remove": bool (default false — set true to erase)}.
    # A declarative statement made by an entity — threat, boast, confession,
    # rumour, vow, insult, etc.  Persists so NPCs and the consequence system
    # can react to it later.  Payload:
    #   {"actor": str, "text": str, "claim_type": str,
    #    "severity": float (0-1), "targets": list[str]}
    CLAIM_MADE = "claim_made"
    TILE_MARKED = "tile_marked"
    # A new physical structure placed at a coordinate — barricade, campfire,
    # pile of rubble, locked box, locked door mechanism.
    # Unlike ITEM_SYNTHESIZED (inventory) this anchors to a grid position.
    # Payload: {"x": int, "y": int, "name": str, "tags": list,
    #           "passable": bool, "transparent": bool,
    #           "attributes": dict, "meta": dict,
    #           "object_id": str, "author_entity_id": str}.
    STRUCTURE_CREATED = "structure_created"
    # An existing structure/tile's properties are modified — door locked/
    # unlocked, torch lit/extinguished, lever pulled, portcullis raised.
    # Payload: {"x": int, "y": int, "object_id": str|null,
    #           "set_passable": bool|null, "set_transparent": bool|null,
    #           "add_tags": list, "remove_tags": list,
    #           "attribute_set": dict, "cause": str}.
    STRUCTURE_MODIFIED = "structure_modified"
    # Generic key/value write on a tile or region representing
    # environmental state: locked, lit, temperature, smell, noise_level, etc.
    # Payload: {"x": int|null, "y": int|null, "region_id": str|null,
    #           "key": str, "value": Any, "cause": str, "author": str}.
    # When x/y are set it writes to that tile; when only region_id is set
    # it writes to the region-level environment.
    ENVIRONMENT_STATE_CHANGED = "environment_state_changed"
    # ── Needs system ────────────────────────────────────────────────────
    # A need stat (hunger, fatigue, social_need, purpose) changed.
    # Payload: {"entity_id": str, "need": str, "delta": float, "cause": str}.
    # apply_transitions writes entity.stats[need], clamped to [0, 100].
    NEED_CHANGED = "need_changed"
    # Fluid in container, spill on tile, or entity wetness.
    # Payload: target_kind object|tile|entity|spill, volumes, material, cause.
    FLUID_CHANGED = "fluid_changed"
    # Biofluids, stains, pathogens, and airborne agents on tiles/entities/air.
    # Payload: target_kind tile|entity|air, agent, operation, concentration,
    #           x/y or entity_id, cause, optional remaining_ticks (airborne).
    CONTAMINATION_CHANGED = "contamination_changed"
    # Generic substance deposit on a field medium (tile ground/air, entity, region).
    # Payload: medium, substance, amount, operation, x/y, remaining_ticks, cause, tick.
    FIELD_CHANGED = "field_changed"
    # Durable audit / memory on entity, object, or tile (interaction_traces).
    # Payload: subject_kind, subject_id, verb, actor_id, tick, summary, outcome.
    WORLD_MARK = "world_mark"
    # Durable world fact registered (adjudication, compile promotion, economy).
    # Payload: {"fact": WorldFact dict}.  Replay-safe alternative to direct
    # ``world_facts`` list mutation.
    WORLD_FACT_REGISTERED = "world_fact_registered"
    # ── Skill system ────────────────────────────────────────────────────
    # Entity gained skill experience from successful use.
    # Payload: {"entity_id": str, "skill": str, "delta": float, "new_value": float, "cause": str}.
    SKILL_INCREASED = "skill_increased"
    # ── Secret economy ──────────────────────────────────────────────────
    # A secret was shared from one entity to another (willing or overheard).
    # Payload: {"from_entity": str, "to_entity": str, "secret": str,
    #           "voluntary": bool, "cause": str}.
    SECRET_REVEALED = "secret_revealed"
    # ── Faction economy ─────────────────────────────────────────────────
    # A faction's resource level changed (gold, food, information, weapons).
    # Payload: {"faction_id": str, "resource": str, "delta": float,
    #           "cause": str, "actor": str|null}.
    FACTION_RESOURCE_CHANGED = "faction_resource_changed"
    # ── Sound propagation ───────────────────────────────────────────────
    # A loud sound rippled to nearby entities; their alertness was raised.
    # Payload: {"source_id": str, "radius": int, "description": str,
    #           "affected_entities": list[str]}.
    SOUND_PROPAGATED = "sound_propagated"
    # Per-entity observational memory record (replay-safe witness log).
    # Payload: entity_id, event_id, tick, summary, kind, priority, actor_id, ...
    OBSERVATION_RECORDED = "observation_recorded"
    # Delayed consequence queued for a future tick (adjudication, propagation).
    # Payload: {"effect": ScheduledEffect dict}.  Replay-safe alternative to
    # direct ``scheduled_effects`` list mutation.
    SCHEDULED_EFFECT_QUEUED = "scheduled_effect_queued"
    # Regional market entry updated (supply/demand/price drift).
    # Payload: {"region_id": str, "tag": str, "entry": dict}.
    MARKET_UPDATED = "market_updated"
    # Pack pressure evaluation snapshot for this tick.
    # Payload: active_pressures, ambient_probability_mult, entity_boosts,
    # meta_patches, clear_entity_boosts (bool).
    PRESSURE_STATE_UPDATED = "pressure_state_updated"


class Transition(BaseModel):
    kind: TransitionKind
    # Payload is transition-specific; kept generic to avoid explosion of
    # subclasses while still being serializable.
    payload: dict[str, Any]


class ActionZone(str, Enum):
    """
    How well a SemanticAction is grounded in canonical world state.

    GROUNDED  — all referenced entities exist; Compiler resolves fully.
    EXTENDING — player asserts invented facts; Compiler resolves + writes
                belief edges to the relational graph.
    ORPHANED  — no canonical grounding; Compiler records a minimal event
                and the NarratorActor provides the response.
    """
    GROUNDED = "grounded"
    EXTENDING = "extending"
    ORPHANED = "orphaned"


class ActionType:
    """
    Convention namespace for *well-known* verb strings. The engine does
    NOT close action verbs to this list — `SemanticAction.verb` is an
    open string and any verb is legal.

    Only the verbs in `KERNEL` have bespoke compiler logic (spatial /
    inventory / lethality physics). All other verbs flow through the
    generic interaction compiler, which derives effects from:
      • optional pack-defined verb templates;
      • LM-proposed primitive effects (validated and clipped);
      • optional contest resolution.

    These constants exist so call-sites can write `ActionType.MOVE`
    instead of the bare string `"move"`, and to give the LM prompt a
    list of conventional verbs as a starting point — not as a closed
    vocabulary.
    """

    # Kernel verbs (bespoke compilers in compiler.py)
    MOVE = "move"
    ATTACK = "attack"
    TAKE = "take"
    GIVE = "give"
    THROW = "throw"
    # Inventory management kernel verbs
    EQUIP = "equip"
    UNEQUIP = "unequip"
    WEAR = "wear"       # alias for equip
    REMOVE = "remove"   # alias for unequip
    DRAW = "draw"       # alias for equip (weapons)
    SHEATHE = "sheathe" # alias for equip (stowing a weapon — reversed semantics, but
                        # treated identically at compiler level for simplicity)
    STORE = "store"     # put item into a container in inventory
    RETRIEVE = "retrieve"  # take item out of a container in inventory
    # Facing / orientation kernel verbs
    TURN = "turn"       # change facing direction without moving
    LOOK = "look"       # alias for turn (semantically: rotate to face something)
    FACE = "face"       # alias for turn
    # Spell casting kernel verb — routes to spell_runtime.py
    CAST = "cast"
    # Item interaction kernel verbs
    DROP = "drop"       # drop item from inventory onto ground
    USE = "use"         # use an item (consumes consumables, ignites torches, etc.)
    MIX = "mix"         # combine two items from inventory (chemistry)
    # Non-combat interaction kernel verbs
    EXAMINE = "examine" # detailed entity/object inspection (perception check)
    TRADE   = "trade"   # economic exchange — items and/or gold
    STEAL   = "steal"   # covert item acquisition (stealth vs perception check)
    REST    = "rest"    # stamina + health recovery (requires non-combat state)
    # Open-world synthesis kernel verb
    CRAFT   = "craft"   # synthesize a new object from materials in inventory
    # Environmental interaction kernel verbs
    MARK      = "mark"      # scratch/write/draw something on a tile
    BARRICADE = "barricade" # place a physical structure blocking passage
    UNLOCK    = "unlock"    # change locked/passable state of a structure/door
    IGNITE    = "ignite"    # light something (campfire, torch, candle) in place

    # Conventional non-kernel verbs (no bespoke compiler — flow through
    # the generic interaction path). Listed for backward compatibility
    # and for the LM prompt's "for example" enumeration.
    SPEAK = "speak"
    ASK = "ask"
    INTIMIDATE = "intimidate"
    PERSUADE = "persuade"
    DECEIVE = "deceive"
    BRIBE = "bribe"
    THREATEN = "threaten"
    OBSERVE = "observe"
    INSPECT = "inspect"
    OPEN = "open"
    CLOSE = "close"
    USE = "use"
    FLEE = "flee"
    HIDE = "hide"
    WAIT = "wait"
    CONTACT = "contact"
    SYMBOLIC = "symbolic"
    # Relational graph — explicit LM verbs (also available via proposed_effects)
    EDGE_ADD = "edge_add"
    EDGE_UPDATE = "edge_update"
    # Synthetic verb used by the WorldClock for ambient events whose
    # actor is the world itself ("system"). Never produced by the LM
    # or any NPC policy; never reaches the kernel/generic compiler.
    AMBIENT = "ambient"

    KERNEL: frozenset[str] = frozenset({
        MOVE, ATTACK, TAKE, GIVE, THROW,
        EQUIP, UNEQUIP, WEAR, REMOVE, DRAW, SHEATHE, STORE, RETRIEVE,
        TURN, LOOK, FACE,
        CAST,
        DROP, USE, MIX,
        EXAMINE, TRADE, STEAL, REST,
        CRAFT,
        MARK, BARRICADE, UNLOCK, IGNITE,
    })

    @classmethod
    def all(cls) -> list[str]:
        """List of conventional verb strings (NOT a closed vocabulary)."""
        return [
            v for k, v in vars(cls).items()
            if isinstance(v, str) and not k.startswith("_")
            and k != "KERNEL" and k.isupper()
        ]


class IntentBlock(BaseModel):
    desired_outcome: list[str] = Field(default_factory=list)
    # Short rationale for the LM to reason about
    rationale: Optional[str] = None
    # Free-form description of HOW an action is performed (the gesture,
    # the wording, the body language). Used primarily by CONTACT to
    # name the manner of physical interaction (embrace, kiss, shake
    # hands, grab, undress, …). The engine never branches on this
    # string — it is passed through to the narrator and stored on the
    # event for later semantic retrieval.
    manner: Optional[str] = None


class StyleBlock(BaseModel):
    emotional_tone: Optional[str] = None
    aggression: int = Field(default=50, ge=0, le=100)
    # 0=covert, 100=overt
    visibility: int = Field(default=50, ge=0, le=100)


class ConstraintBlock(BaseModel):
    avoid_combat: bool = False
    avoid_witnesses: bool = False
    preserve_relationship: Optional[str] = None


class TransitionProposal(BaseModel):
    """
    A primitive state mutation that the LM (or pack template) proposes
    as a consequence of an action.

    The engine validates each proposal against canonical state before
    applying it: out-of-bounds values are clipped, references to
    non-existent entities/objects/edges are dropped, and proposals
    outside the actor's plausible reach are rejected. Only proposals
    that survive validation are written into the canonical event log.

    `kind` is a TransitionKind string. `payload` follows the same shape
    as Transition.payload for that kind, with two affordances for pack
    templates:
      • String values may contain "$actor" / "$target" — the compiler
        substitutes the action's actor/target EntityIds at compile time.
      • Numeric values may be expressed as either absolute or signed
        deltas depending on the transition kind (see compiler).

    `synthesized_object` is used when kind == ITEM_SYNTHESIZED.  The LM
    populates this dict with the proposed ObjectState schema; the engine
    validates physical plausibility (actor holds required materials, the
    resulting object makes sense given inputs) before writing it.
    """

    kind: str
    payload: dict[str, Any] = Field(default_factory=dict)
    # Optional LM-proposed object schema for ITEM_SYNTHESIZED proposals.
    # Keys mirror ObjectState fields: name, tags, weight, bulk, attributes, meta.
    synthesized_object: Optional[dict[str, Any]] = None


class ContestSpec(BaseModel):
    """
    Optional skill contest for resolving a generic interaction.

    The compiler runs a deterministic seeded roll comparing the actor's
    `actor_attribute` against the target's `target_attribute` (if a
    target is provided), modified by `difficulty`. The outcome (success
    / partial / failure) is passed into the effect-selection step.
    """

    actor_attribute: str = "persuasion"
    target_attribute: Optional[str] = None
    difficulty: int = 50


class SemanticAction(BaseModel):
    action_id: str = Field(default_factory=lambda: f"act_{uuid.uuid4().hex[:8]}")
    # The verb is an OPEN string. `ActionType.*` are conventional
    # constants but any verb is legal — the engine routes the kernel
    # verbs to bespoke compilers and everything else to a generic path
    # that consults pack templates + LM-proposed effects.
    verb: str
    actor: EntityId
    target: Optional[Union[EntityId, Coord]] = None
    intent: IntentBlock = Field(default_factory=IntentBlock)
    style: StyleBlock = Field(default_factory=StyleBlock)
    constraints: ConstraintBlock = Field(default_factory=ConstraintBlock)
    # Primitive state mutations the LM thinks should follow from this
    # action. Validated and clipped by the compiler; empty means
    # "engine derives effects from verb template (if any) only".
    proposed_effects: list[TransitionProposal] = Field(default_factory=list)
    # Optional skill contest. If None, generic actions auto-succeed
    # at their default effects unless the verb template specifies a
    # contest.
    contest: Optional[ContestSpec] = None
    # Raw natural language that produced this action (for logging)
    raw_input: Optional[str] = None


class ConsentState(str, Enum):
    """
    Computed at compile time on a CONTACT action. Captures how the
    target's relational/openness state relates to the action; does NOT
    determine the target's response (the target's own policy decides
    that on its next turn).
    """
    WELCOMED = "welcomed"      # mutual INTIMATE edge or welcoming + low aggression
    UNWELCOMED = "unwelcomed"  # no anchor + non-aggressive: a boundary crossed
    REFUSED = "refused"        # WARY_OF / ENEMY_OF / HOSTILE: explicitly rejected
    HOSTILE = "hostile"        # high aggression + no INTIMATE: assault-shaped


class RejectionReason(str, Enum):
    ENTITY_NOT_VISIBLE = "entity_not_visible"
    PATH_INACCESSIBLE = "path_inaccessible"
    TARGET_OUT_OF_RANGE = "target_out_of_range"
    ACTOR_NOT_FOUND = "actor_not_found"
    TARGET_NOT_FOUND = "target_not_found"
    PHYSICALLY_IMPOSSIBLE = "physically_impossible"
    ACTION_TYPE_UNSUPPORTED = "action_type_unsupported"
    MALFORMED_ACTION = "malformed_action"


class GroundingResult(BaseModel):
    """Output of the grounding classifier — consumed by game_loop.step()."""
    zone: ActionZone
    # Canonical entity ids found to be referenced by the action
    grounded_entities: list[EntityId] = Field(default_factory=list)
    # Zone 2: plain-text description of the fact the player is asserting
    asserted_fact: Optional[str] = None
    # Zone 3: why this couldn't be grounded (hint for the Narrator)
    orphan_reason: Optional[str] = None


class ValidationResult(BaseModel):
    valid: bool
    concrete_transitions: list[Transition] = Field(default_factory=list)
    rejection_reason: Optional[RejectionReason] = None
    rejection_detail: Optional[str] = None
    # Plausibility is estimated by the engine, not the LM
    plausibility: float = Field(default=1.0, ge=0.0, le=1.0)


class Event(BaseModel):
    event_id: EventId = Field(default_factory=new_event_id)
    tick: int
    action: SemanticAction
    transitions: list[Transition] = Field(default_factory=list)
    witnesses: list[EntityId] = Field(default_factory=list)
    # Downstream hint for the narrative renderer; not authoritative
    narrative_hint: Optional[str] = None


# ---------------------------------------------------------------------------
# Semantic projection (what the LM sees)
# ---------------------------------------------------------------------------


class EntitySummary(BaseModel):
    """A lossy but semantically dense representation of a visible entity."""

    entity_id: EntityId
    name: str
    kind: EntityKind
    # Topological description, not raw coordinates
    relative_position: str
    distance: int
    emotional_state: EmotionalState
    alertness: AlertnessLevel
    armed: bool
    intoxication: int
    faction_ids: list[FactionId] = Field(default_factory=list)
    visible_tags: list[str] = Field(default_factory=list)
    # Active conditions visible to the focal entity (poisoned, stunned …)
    conditions: list[str] = Field(default_factory=list)
    # Rolling window of the most recent verbs used toward this entity by the
    # focal entity (from the INTERACTED relational edge).  None when no prior
    # interaction has been recorded (serialised as absent to save tokens).
    # Maximum length = INTERACTION_HISTORY_WINDOW (defined in relational.py).
    interaction_history: Optional[list[str]] = None
    # Compass direction this entity is currently facing ("north", "east", …).
    # Allows the LM / player to reason about who can see whom.
    facing: Optional[str] = None
    # Physical state from substance layer (temperature, fire, material).
    physical_state: Optional[str] = None


class EnvironmentSummary(BaseModel):
    location_name: str = "unknown"
    crowded: bool = False
    lighting: str = "normal"
    exits: list[str] = Field(default_factory=list)
    nearby_cover: list[str] = Field(default_factory=list)
    environmental_hazards: list[str] = Field(default_factory=list)
    noise_level: str = "quiet"
    # Room-level physical cues (hearth warmth, smoke, cold draft).
    physical_notes: list[str] = Field(default_factory=list)
    # Notable objects with physical state in view.
    object_physical_notes: list[str] = Field(default_factory=list)
    # Regional market prices (when economy is active).
    market_notes: list[str] = Field(default_factory=list)


class SocialContext(BaseModel):
    recent_argument: bool = False
    faction_tension: str = "none"
    # Structured summaries of active social dynamics
    active_conflicts: list[str] = Field(default_factory=list)
    known_obligations: list[str] = Field(default_factory=list)
    reputation_notes: list[str] = Field(default_factory=list)


class Affordance(BaseModel):
    """An action the focal entity could plausibly attempt from current state."""

    verb: str
    description: str
    targets: list[str] = Field(default_factory=list)


class EventSummary(BaseModel):
    tick: int
    description: str
    participants: list[str] = Field(default_factory=list)


class TimeOfDay(str, Enum):
    """
    Quarters of the world's day cycle.

    The engine treats this as a derived projection over `WorldState.tick`
    using `WorldClock.day_length_ticks`. Pack authors trigger ambient
    events on these phases; NPCs and the player see the current value
    on every projection.
    """

    DAWN = "dawn"
    DAY = "day"
    DUSK = "dusk"
    NIGHT = "night"


class SocialAim(BaseModel):
    """
    Relational intent toward another entity — loaded from pack data.

    ``target_entity_id`` is resolved at world load from the pack-local
    ``target`` id in entities.yaml.
    """

    target_entity_id: EntityId
    target_pack_id: str = ""
    aim: str = ""


# ---------------------------------------------------------------------------
# Long-horizon NPC plans (DF-style job stacks)
# ---------------------------------------------------------------------------


class PlanStepKind(str, Enum):
    """The handful of step primitives the planner can decompose drives into.

    These intentionally cover the *minimum* required to express a multi-tick
    project: get somewhere, spend time doing something, interact with a
    target, talk, observe.  Richer step kinds can be added as new drives
    require them — every kind needs a candidate emitter in
    ``npc_planner.next_step_action``.
    """

    MOVE_TO = "move_to"        # payload: {tile: [x,y]} OR {entity_id: "..."}
    SPEND_TICKS = "spend_ticks"  # payload: {ticks: int, label: str}
    INTERACT = "interact"      # payload: {verb: str, target: str}
    SPEAK_TO = "speak_to"      # payload: {entity_id: str, line: str?}
    OBSERVE = "observe"        # payload: {tile: [x,y]} OR {entity_id: "..."}


class PlanStep(BaseModel):
    """One discrete step inside an ``NpcPlan``."""

    kind: PlanStepKind
    payload: dict[str, Any] = Field(default_factory=dict)
    label: str = ""          # short human-readable description
    done: bool = False
    attempts: int = 0        # incremented each tick the step is the active step


class NpcPlanTemplate(BaseModel):
    """
    Pack-authored plan pattern from ``npc_plans.yaml``.

    Matched against ``entity.drive`` (regex) and optional entity pack id / role
    filters before the built-in ``_PLAN_PATTERNS`` library runs.
    """

    id: str = ""
    drive_pattern: str = ""
    entity_ids: list[str] = Field(default_factory=list)
    entity_roles: list[str] = Field(default_factory=list)
    name: str = ""
    # Optional built-in builder: patrol, tend, work, deliver
    builder: str = ""
    steps: list[PlanStep] = Field(default_factory=list)


class PlanStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"        # interrupted by combat/threat; resumes when safe
    COMPLETED = "completed"
    FAILED = "failed"        # gave up after too many failed attempts
    ABANDONED = "abandoned"  # explicitly dropped (e.g. drive changed)


class NpcPlan(BaseModel):
    """
    A multi-tick project an NPC is working toward.

    Plans sit *above* per-tick ``ReactivePolicy`` decisions: each tick the
    policy first asks the planner "what does my plan want me to do right
    now?" and only falls back to the candidate menu when there is no
    active plan, the active step has no actionable candidate, or an
    interrupt fires (combat, low health, fleeing).

    Plans are deterministic and engine-owned — no LM call required —
    which means NPCs feel *intentional* even in ``--cognition reactive_only``
    runs (the DF / Qud parity test).
    """

    plan_id: str = Field(default_factory=lambda: f"plan_{uuid.uuid4().hex[:8]}")
    name: str = ""               # short label e.g. "rebuild the workshop"
    owner_id: EntityId
    drive_source: str = ""       # the entity.drive text this plan was synthesized from
    steps: list[PlanStep] = Field(default_factory=list)
    current_step: int = 0
    status: PlanStatus = PlanStatus.ACTIVE
    started_tick: int = 0
    last_progress_tick: int = 0
    failure_reason: str = ""
    # Max attempts on a single step before the plan gives up.  Tunable per-plan.
    max_step_attempts: int = 8
    meta: dict[str, Any] = Field(default_factory=dict)


class NpcCharacterSheet(BaseModel):
    """
    A bundle of NPC self-knowledge handed to the LM-driven NPC policy.

    Built by `npc_policy.build_npc_character_sheet(entity, world)` from
    canonical state — it carries no projection. It complements the
    NPC's `SemanticProjection` (their view of the world *around* them)
    with their view of *themselves*: role, personality, drive, current
    mood, relationships, and a short list of recent perceptions
    (events targeting or witnessed by this NPC).

    The fields are strings / shallow structures so they serialize
    cleanly into the LM prompt and stay easy to inspect.
    """

    npc_id: EntityId
    name: str
    role: str = ""
    personality: str = ""
    drive: str = ""
    emotional_state: EmotionalState = EmotionalState.NEUTRAL
    alertness: AlertnessLevel = AlertnessLevel.LOW
    social_openness: str = "guarded"
    health_fraction: float = 1.0
    # The direction this NPC is currently looking; affects their vision cone.
    facing: str = "south"
    # Each entry is a short prose summary of an outbound relational edge.
    # Example: 'respects ent_player (weight=0.6)'
    relations: list[str] = Field(default_factory=list)
    # Recent events where this NPC was actor / target / witness, in
    # chronological order. Each entry is one short prose line.
    # Example: 'tick 2: ent_player kissed you (surprise)'
    recent_perceptions: list[str] = Field(default_factory=list)
    # LM-generated episode summaries for long-running sessions.
    # Each entry covers EPISODE_SIZE ticks, oldest first.
    # Example: 'Linna approached Tomas. He rebuffed her offer of wine.'
    episode_summaries: list[str] = Field(default_factory=list)
    # Active conversation thread this NPC is currently in, if any.
    # Formatted as a short prose summary of recent exchanges.
    active_dialogue: Optional[str] = None
    # Facts this NPC knows — sourced from EntityState.knowledge.
    # Surfaced to the LM so replies stay grounded in simulation truth.
    knowledge: list[str] = Field(default_factory=list)
    # Goals — sourced from EntityState.goals.
    goals: list[str] = Field(default_factory=list)
    # Relational aims — who to engage (formatted with target entity ids).
    social_aims: list[str] = Field(default_factory=list)
    # Secrets — only included when relationship with focal entity warrants
    # disclosure (INTIMATE or ALLIED edge).  Empty otherwise.
    secrets: list[str] = Field(default_factory=list)
    # Non-combat identity — occupation, faction, and economic/exertion state.
    # Included so the LM-driven NPC policy generates contextually appropriate
    # actions (merchants trade, scholars share knowledge, tired NPCs rest).
    occupation: str = ""
    faction: str = ""
    stamina_fraction: float = 1.0   # stamina / 100
    gold_fraction: float = 0.0      # relative economic wealth (0–1 normalised)
    reputation: float = 0.0
    # Author-supplied example lines from entities.yaml — used as fallback
    # dialogue seeds so character voice survives LM failures.
    voice_lines: list[str] = Field(default_factory=list)
    # Canonized world facts relevant to this NPC (from adjudication layer).
    established_facts: list[str] = Field(default_factory=list)
    # Recent spoken lines in the room (any actor) — anti-repetition for infer_npc.
    recent_room_dialogue: list[str] = Field(default_factory=list)
    # Director + belief context (read-only hints for LM).
    director_hint: str = ""
    taboo_phrases: list[str] = Field(default_factory=list)
    held_beliefs: list[str] = Field(default_factory=list)
    # Updated each turn by npc_reflection.run_npc_reflection (meta → sheet).
    inner_reflection: str = ""
    focus_topic: str = ""
    # Long-horizon retrieved event-log lines (beyond Markov window).
    retrieved_memories: list[str] = Field(default_factory=list)
    # Cold-tier chronicle milestones.
    chronicle_memories: list[str] = Field(default_factory=list)


class BeliefRecord(BaseModel):
    """Structured belief held by an entity (compiled from social transitions)."""

    subject: str
    claim: str
    confidence: float = 0.5
    source_event_id: str = ""


# ---------------------------------------------------------------------------
# World adjudication — durable facts and delayed effects
# ---------------------------------------------------------------------------


class WorldFactScope(str, Enum):
    """Where a canonized fact is anchored in the simulation."""

    WORLD = "world"
    REGION = "region"
    ENTITY = "entity"
    TILE = "tile"


class WorldFact(BaseModel):
    """
    A durable claim the DM/engine has canonized about the world.

    Registered by the adjudication layer when player intent cannot map to a
    kernel verb but should still persist (graffiti, invented lore, room state).
    """

    fact_id: str = Field(default_factory=lambda: f"fact_{uuid.uuid4().hex[:8]}")
    claim: str
    scope: WorldFactScope = WorldFactScope.WORLD
    subject_id: Optional[str] = None
    established_tick: int = 0
    established_by: Optional[str] = None
    source_intent: Optional[str] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)


class WorldChronicleEntry(BaseModel):
    """Cold-tier compressed memory milestone for long campaigns."""

    entry_id: str = Field(default_factory=lambda: f"chron_{uuid.uuid4().hex[:8]}")
    tick: int = 0
    summary: str
    importance: float = Field(default=0.5, ge=0.0, le=20.0)
    actor_id: Optional[str] = None
    subject_id: Optional[str] = None
    region_id: Optional[str] = None
    scope: str = "region"
    tags: list[str] = Field(default_factory=list)


class ScheduledEffectKind(str, Enum):
    TRANSITIONS = "transitions"
    NARRATION = "narration"


class ScheduledEffect(BaseModel):
    """
    A delayed consequence queued by adjudication to fire on a future tick.

    Processed deterministically by ``world_clock`` — no LM calls at fire time.
    """

    effect_id: str = Field(default_factory=lambda: f"sched_{uuid.uuid4().hex[:8]}")
    fire_tick: int
    created_tick: int = 0
    source_intent: Optional[str] = None
    source_actor: Optional[str] = None
    kind: ScheduledEffectKind = ScheduledEffectKind.TRANSITIONS
    transitions: list["TransitionProposal"] = Field(default_factory=list)
    narration: Optional[str] = None
    rationale: Optional[str] = None
    meta: dict[str, Any] = Field(default_factory=dict)


class AdjudicationResult(BaseModel):
    """
    Output of the DM adjudication layer for one unresolved player action.

    All state mutations still pass through ``compile_action`` /
    ``_validate_proposals`` before apply — this type carries intent only.
    """

    ruling_text: str = ""
    facts: list[WorldFact] = Field(default_factory=list)
    scheduled_effects: list[ScheduledEffect] = Field(default_factory=list)
    transition_proposals: list["TransitionProposal"] = Field(default_factory=list)
    synthesized_verb: Optional[str] = None


# ---------------------------------------------------------------------------
# Semantic dynamics (delayed interpretation layer)
# ---------------------------------------------------------------------------


class SemanticLensKind(str, Enum):
    AGENT = "agent"
    FACTION = "faction"
    INSTITUTION = "institution"


class SemanticPassTrigger(str, Enum):
    SCHEDULED = "scheduled"
    CRISIS = "crisis"


class SemanticDynamicsConfig(BaseModel):
    """
    Controls the delayed semantic interpretation pass (see ENGINE_DESIGN.md §24).

    Loaded from world pack ``extra.semantic`` in world.yaml when present.
    """

    enabled: bool = True
    interval_ticks: int = Field(default=3, ge=1)
    max_events_per_slice: int = Field(default=20, ge=1)
    max_deltas_per_lens: int = Field(default=8, ge=1)
    edge_delta_cap: float = Field(default=0.25, ge=0.01, le=0.5)
    use_mock: bool = True
    # NPC hear-and-react: only observations within this many ticks matter.
    reaction_memory_ticks: int = Field(default=6, ge=1)
    # Belief/rumour diffusion interval (1 = every tick).
    belief_propagation_interval: int = Field(default=1, ge=1)


class CanonicalEventLine(BaseModel):
    """One grounded fact from the event log for semantic interpretation."""

    event_id: str
    tick: int
    actor_id: str
    actor_name: str
    verb: str
    target_id: Optional[str] = None
    target_name: Optional[str] = None
    transition_kinds: list[str] = Field(default_factory=list)
    summary: str


class SemanticSlice(BaseModel):
    """
    Structured history window for one interpretive lens.

    Built deterministically from ``event_log``; consumed by semantic pressure
  adapters (mock or LM).  Never includes raw ``WorldState``.
    """

    slice_id: str = Field(default_factory=lambda: f"slice_{uuid.uuid4().hex[:8]}")
    tick_from: int = 0
    tick_to: int = 0
    lens: SemanticLensKind = SemanticLensKind.AGENT
    lens_id: str = ""
    lens_name: str = ""
    trigger: SemanticPassTrigger = SemanticPassTrigger.SCHEDULED
    canonical_events: list[CanonicalEventLine] = Field(default_factory=list)
    prior_beliefs: list[str] = Field(default_factory=list)
    prior_narratives: list[str] = Field(default_factory=list)
    cultural_anchors: list[str] = Field(default_factory=list)


class SemanticDelta(BaseModel):
    """
    One bounded semantic consequence proposed from a slice.

    Must cite at least one canonical ``event_id``.  Compiled by
    ``semantic_compiler`` into transitions and/or graph meta.
    """

    kind: str
    rationale_event_id: str
    source: Optional[str] = None
    target: Optional[str] = None
    edge_kind: Optional[str] = None
    delta: Optional[float] = None
    weight: Optional[float] = None
    claim: Optional[str] = None
    fidelity: Optional[float] = None
    goal: Optional[str] = None
    priority: float = 0.5
    theme: Optional[str] = None
    intensity: Optional[float] = None
    scope: Optional[str] = None
    axis: Optional[str] = None
    rule: Optional[str] = None
    strength: Optional[float] = None


class SemanticProjection(BaseModel):
    """
    The only type the LM adapter is allowed to receive.

    Contains no raw WorldState, no grid coordinates, no full entity objects.
    """

    projection_id: str = Field(
        default_factory=lambda: f"proj_{uuid.uuid4().hex[:8]}"
    )
    focal_entity: EntityId
    tick: int
    visible_entities: list[EntitySummary] = Field(default_factory=list)
    environment: EnvironmentSummary = Field(default_factory=EnvironmentSummary)
    social_context: SocialContext = Field(default_factory=SocialContext)
    available_affordances: list[Affordance] = Field(default_factory=list)
    recent_events: list[EventSummary] = Field(default_factory=list)
    # Current phase of the world clock — DAWN / DAY / DUSK / NIGHT.
    # Derived from world.tick % world.config.clock.day_length_ticks.
    # Surfaced here so both player and NPC LM prompts see the same field.
    time_of_day: TimeOfDay = TimeOfDay.DAY
    # If the focal entity's previous action was rejected by the compiler,
    # this carries the reason so the LM can learn and adjust.
    last_action_rejection: Optional[str] = None
    # Which region the focal entity is currently in.
    current_region: str = "region_main"
    # The compass direction the focal entity is currently facing.
    # Exposed so the LM knows which direction they are looking.
    facing: str = "south"
    # Estimated token cost for budget enforcement
    estimated_tokens: int = 0
    # Beliefs the focal entity holds about others (from relational graph).
    beliefs: list[BeliefRecord] = Field(default_factory=list)
    # Canonized world facts visible to the focal entity (from adjudication).
    world_facts: list[str] = Field(default_factory=list)
    # Retrieved long-horizon event-log memories (scored relevance).
    retrieved_memories: list[str] = Field(default_factory=list)
    # Cold-tier chronicle snippets (compressed past episodes).
    chronicle_snippets: list[str] = Field(default_factory=list)
    # Tag-grammar hints from interaction_rules.yaml (deterministic, no LM).
    interaction_hints: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Top-level world state
# ---------------------------------------------------------------------------


class DialogueExchange(BaseModel):
    """
    A single utterance within a ``ConversationThread``.

    ``text`` is drawn from ``SemanticAction.intent.rationale`` when the
    actor is performing a speaking verb (speak, ask, whisper, etc.).
    """

    tick: int
    speaker_id: EntityId
    speaker_name: str
    listener_id: Optional[EntityId] = None
    listener_name: str = ""
    verb: str
    text: Optional[str] = None


class ConversationThread(BaseModel):
    """
    A tracked conversation between two or more entities.

    Lifecycle
    ---------
    - OPENED: when any speaking verb connects two entities for the first time.
    - UPDATED: each subsequent speaking verb between the same participants.
    - CLOSED: when ``world.tick - last_exchange_at >= THREAD_IDLE_TICKS``.

    The thread is serialized into each participant's ``NpcCharacterSheet``
    so both parties remember what was said, making multi-turn dialogue
    coherent beyond the raw 12-event perception window.
    """

    thread_id: str
    # Sorted participant IDs for stable lookup (pair a,b == pair b,a).
    participant_ids: list[str]
    participant_names: list[str] = Field(default_factory=list)
    exchanges: list[DialogueExchange] = Field(default_factory=list)
    opened_at: int
    last_exchange_at: int
    closed: bool = False


class GoalCondition(BaseModel):
    """
    A single testable condition for a ``Goal``.

    Condition types (``type`` field):
    ----------------------------------
    edge_exists     — a relational edge of ``edge_kind`` from ``source`` to
                      ``target`` exists in the graph.
    entity_state    — ``entity``'s ``emotional_state`` matches ``state``.
    tick_reached    — ``world.tick >= tick``.
    event_occurred  — at least one event with ``verb`` (and optionally
                      ``actor``) appears in the event log.

    All conditions in a Goal's ``conditions`` list are AND-combined.
    """

    type: str  # edge_exists | entity_state | tick_reached | event_occurred
    # edge_exists
    source: Optional[str] = None
    target: Optional[str] = None
    edge_kind: Optional[str] = None
    # entity_state
    entity: Optional[str] = None
    state: Optional[str] = None
    # tick_reached
    tick: Optional[int] = None
    # event_occurred
    verb: Optional[str] = None
    actor: Optional[str] = None
    # same_region: if True, both actor and target must be in the same region
    # at the time the event is evaluated.  Prevents "speak to the King" from
    # completing when the King is in a different room.
    same_region: bool = False


class GoalEdgeUpdate(BaseModel):
    """A relational edge to create/update when a goal is completed."""

    source: str
    target: str
    edge_kind: str
    weight: float = 0.5


class GoalReward(BaseModel):
    """
    Side-effects applied when a goal transitions to COMPLETE.

    ``narrative``    — displayed to the player (REPL / TUI).
    ``lore_unlock``  — key written into ``WorldState.meta["unlocked_lore"]``.
    ``edge_updates`` — relational edges to create or overwrite.
    """

    narrative: Optional[str] = None
    lore_unlock: Optional[str] = None
    edge_updates: list[GoalEdgeUpdate] = Field(default_factory=list)


class Goal(BaseModel):
    """
    A declarative quest objective.

    Loaded from ``worlds/<pack>/goals.yaml`` by ``WorldLoader``.
    Goals are evaluated after every tick by
    ``game_loop.evaluate_goals(world)``.

    Completed goals are never re-evaluated. ``completed_at`` holds the
    tick on which the goal was satisfied.
    """

    id: str
    title: str
    description: str
    # All conditions must be true simultaneously for the goal to complete.
    conditions: list[GoalCondition] = Field(default_factory=list)
    reward: GoalReward = Field(default_factory=GoalReward)
    completed: bool = False
    completed_at: Optional[int] = None


class AffordanceRule(BaseModel):
    """
    Declares what counts as a 'meaningful anchor' for an intent keyword.

    Loaded from a world pack's affordances.yaml. The engine itself has no
    opinion about prayer, magic, or any other thematic domain — each world
    declares its own affordance map. An intent matches this rule when any
    keyword appears in the player's intent and at least one node in the
    relational graph satisfies:
        node.kind in kinds
        AND (any of name_keywords appears in node.name
             OR  any of tags appears in node.meta.tags)
    """

    keyword: str
    kinds: list["NodeKind"] = Field(default_factory=list)
    name_keywords: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class NpcCognitionConfig(BaseModel):
    """
    How NPCs choose actions each tick.

    lm            — structured ``infer_npc`` JSON each tick (default, all models).
    reactive_only — deterministic ReactivePolicy (tests / no LM).
    """

    mode: str = "lm"
    max_infer_per_npc: int = 2
    infer_cooldown_ticks: int = 8
    # Cap total LM infer_npc calls per tick (None / 0 = unlimited).
    max_lm_calls_per_tick: Optional[int] = 4


class MindScoringConfig(BaseModel):
    """Weights for npc_mind.context-driven action scoring."""

    min_speak_gap_ticks: int = 2
    gesture_memory_ticks: int = 4
    intent_reply_speak: float = 0.45
    intent_goal_speak: float = 0.38
    intent_drive_speak: float = 0.32
    gesture_when_reply: float = 0.40
    gesture_repeat: float = 0.55
    same_verb: float = 0.30
    speak_not_ready: float = 0.70
    novelty: float = 0.08
    tension_match: float = 0.18
    acknowledge_gesture: float = 0.28


class NpcPolicyConfig(BaseModel):
    """
    Tunable parameters for the default ReactivePolicy.

    Each world pack can supply its own values. The engine ships no
    hardcoded constants — these defaults are only invoked if a pack
    omits the field entirely.
    """

    threat_lookback_ticks: int = 5
    flee_health_fraction: float = 0.25
    cognition: NpcCognitionConfig = Field(default_factory=NpcCognitionConfig)
    mind_scoring: MindScoringConfig = Field(default_factory=MindScoringConfig)


class VerbTemplate(BaseModel):
    """
    Pack-declared default effects for a verb.

    Templates apply when the verb is not in the engine's kernel
    (move/take/give/attack/throw). They are *defaults* — the LM may
    additionally supply `proposed_effects` on the action that get
    layered on top after the template's effects, all subject to the
    compiler's reachability validator.

    Effect payloads may reference "$actor" and "$target" as string
    substitution placeholders.

    Optional ``narrative_success`` / ``narrative_failure`` use placeholders
    ``{actor}``, ``{target}``, ``{object}`` for rule-based narration.
    """

    verb: str
    contest: Optional["ContestSpec"] = None
    effects_on_success: list["TransitionProposal"] = Field(default_factory=list)
    effects_on_failure: list["TransitionProposal"] = Field(default_factory=list)
    narrative_success: Optional[str] = None
    narrative_failure: Optional[str] = None
    narrative_object_success: Optional[str] = None
    narrative_success_object: Optional[str] = None


class ObjectAffordanceRule(BaseModel):
    """Pack verb offered when a visible object matches tag rules (affordances.yaml)."""

    verb: str
    object_tags: list[str] = Field(default_factory=list)
    absent_tags: list[str] = Field(default_factory=list)


class WorldClock(BaseModel):
    """
    World-level time configuration. Lives on `WorldConfig`.

    `day_length_ticks` is the number of ticks for one full DAWN→NIGHT
    cycle. Phases are evenly divided quarters.
    """

    day_length_ticks: int = Field(default=100, ge=4)

    def time_of_day(self, tick: int) -> TimeOfDay:
        phase = (tick % self.day_length_ticks) / self.day_length_ticks
        if phase < 0.25:
            return TimeOfDay.DAWN
        if phase < 0.5:
            return TimeOfDay.DAY
        if phase < 0.75:
            return TimeOfDay.DUSK
        return TimeOfDay.NIGHT


class AmbientTrigger(BaseModel):
    """
    When does an ambient event fire?

    Triggers are evaluated each tick before NPC turns. An event fires
    when ALL declared conditions hold AND the probability roll passes.
    The probability roll is deterministic — seeded from
    (world.rng_seed, event.id, tick) — so replay is consistent.

    Fields are independent guards (AND-combined):
      every_ticks — fires only on ticks where tick % every_ticks == 0
      time_of_day — fires only while the clock is in this phase
      probability — passes the gate with this probability per evaluation
    """

    every_ticks: Optional[int] = Field(default=None, ge=1)
    time_of_day: Optional[TimeOfDay] = None
    probability: float = Field(default=1.0, ge=0.0, le=1.0)


class AmbientEvent(BaseModel):
    """
    A pack-declared environmental event.

    Fires from the WorldClock autonomic loop — there is no actor. Any
    `effects` are validated by the same TransitionProposal pipeline
    used for LM-proposed effects, so the engine never trusts the pack
    blindly. The narrative is shown to player and stored on the event
    log for later projection windows.
    """

    id: str
    trigger: AmbientTrigger = Field(default_factory=AmbientTrigger)
    narrative: str = ""
    effects: list["TransitionProposal"] = Field(default_factory=list)


class CombinationRule(BaseModel):
    """
    A data-driven item combination / chemistry rule.

    Loaded from worlds/*/combinations.yaml and stored on WorldConfig.

    The rule fires when the player uses ``mix`` (or ``combine``, ``use ... on``)
    with two items whose tags satisfy both input conditions.

    Tag matching: at least one tag from ``input_a_tags`` must be present on
    item A, and at least one from ``input_b_tags`` on item B (if provided).
    Order of inputs is symmetric — the engine tries both orderings.

    Effects applied when the rule fires:
      consume_a / consume_b  — whether the input items are destroyed
      add_tag_to_a / _b      — tag added to the surviving item
      attribute_set_a / _b   — attribute(s) merged into the surviving item
      create_item            — template dict for a new ObjectState placed at actor's feet
      add_actor_tag          — tag added to the actor entity (e.g. "poisoned" from sniffing acid)
      damage_actor           — direct health damage to the actor
    """

    rule_id: str
    name: str = ""
    input_a_tags: list[str] = Field(default_factory=list)
    input_b_tags: list[str] = Field(default_factory=list)
    consume_a: bool = True
    consume_b: bool = True
    add_tag_to_a: Optional[str] = None
    add_tag_to_b: Optional[str] = None
    attribute_set_a: dict[str, Any] = Field(default_factory=dict)
    attribute_set_b: dict[str, Any] = Field(default_factory=dict)
    create_item: Optional[dict[str, Any]] = None   # template for ITEM_CREATED
    add_actor_tag: Optional[str] = None
    damage_actor: int = 0
    narrative: str = ""


class InteractionRule(BaseModel):
    """
    Tag-grammar interaction rule (``interaction_rules.yaml``).

    Matched deterministically before LM infer. When a rule fires the engine
    synthesizes a ``SemanticAction`` with pre-filled ``proposed_effects``.
    """

    id: str
    name: str = ""
    verb_keywords: list[str] = Field(default_factory=list)
    substance_keywords: list[str] = Field(default_factory=list)
    actor_tags: list[str] = Field(default_factory=list)
    target_tags: list[str] = Field(default_factory=list)
    tool_tags: list[str] = Field(default_factory=list)
    target_kind: str = "any"  # entity | object | any
    verb: str = "interact"
    max_distance: int = 10
    priority: int = 0
    contest: Optional["ContestSpec"] = None
    effects: list["TransitionProposal"] = Field(default_factory=list)
    narrative: str = ""


class PropertyInteractionRule(BaseModel):
    """
    Property-intersection rule (``property_interactions.yaml``).

    Resolved inside the compiler AFTER verb templates / open verbs but BEFORE
    the freeform fallback.  Unlike ``InteractionRule``, matching is driven
    entirely by what entities *are* (their tags / material properties), not
    by which words the player typed.

    A rule fires when ALL of the following hold:

    * ``actor_or_tool_tags`` ∩ (actor.tags ∪ any_equipped_object.tags) ≠ ∅
      (empty list → always satisfied — no actor-side requirement)
    * ``target_tags`` ∩ target.tags ≠ ∅
      (empty list → always satisfied — no target requirement)
    * ``target_immunity_tags`` ∩ target.tags = ∅
      (target must not have any immunity tag)
    * The action's intent category is listed in ``intent_categories``
      (``["any"]`` → always satisfied)
    * Euclidean distance between actor and target ≤ ``max_distance``

    Payload placeholders (replaced at substitution time):
      $actor       actor entity id
      $target      target entity / object id
      $tool        matched equipped tool object id (empty string if none)
      $actor_x / $actor_y   actor grid coordinates
      $target_x / $target_y target grid coordinates
    """

    id: str
    name: str = ""
    actor_or_tool_tags: list[str] = Field(
        default_factory=list,
        description="At least one tag that the actor OR a held/equipped tool must have.",
    )
    target_tags: list[str] = Field(
        default_factory=list,
        description="At least one tag the target entity/object must have.",
    )
    target_immunity_tags: list[str] = Field(
        default_factory=list,
        description="Rule is suppressed if the target has ANY of these tags.",
    )
    intent_categories: list[str] = Field(
        default_factory=lambda: ["any"],
        description=(
            "Broad action-intent categories. 'any' matches all. "
            "Values: any | physical | chemical | thermal | movement | social | observation | idle"
        ),
    )
    max_distance: int = 20
    priority: int = 0
    contest: Optional["ContestSpec"] = None
    effects: list["TransitionProposal"] = Field(default_factory=list)
    narrative: str = ""


class AdjudicationIndexEntry(BaseModel):
    """
    Pack-driven adjudication pattern (``adjudication_index.yaml``).

    Slot patterns map creative intent to validated transition proposals
    without calling the LM when confidence is high enough.
    """

    id: str
    patterns: list[str] = Field(default_factory=list)
    action_types: list[str] = Field(default_factory=list)
    substance_keywords: list[str] = Field(default_factory=list)
    min_confidence: float = Field(default=0.55, ge=0.0, le=1.0)
    priority: int = 0
    ruling: str = ""
    synthesized_verb: Optional[str] = None
    fact_tags: list[str] = Field(default_factory=list)
    effects: list["TransitionProposal"] = Field(default_factory=list)


class SynthesisHint(BaseModel):
    """
    An open-ended synthesis hint that tells the engine what *class* of
    objects can be crafted from a class of materials — without needing
    every specific recipe pre-authored.

    The engine uses these hints to validate LM-proposed ITEM_SYNTHESIZED
    transitions: if the proposed output's tags intersect with
    ``output_tags``, AND the actor's inventory contains items whose tags
    cover ``required_input_tags``, the synthesis is considered physically
    plausible.

    Examples from synthesis_hints.yaml:
      - required_input_tags: [wood, sharp_tool]    output_tags: [wooden, tool]
      - required_input_tags: [cloth, plant]        output_tags: [bandage, consumable]
      - required_input_tags: [metal, fire, hammer] output_tags: [weapon, tool]
      - required_input_tags: [leather, needle]     output_tags: [armor, clothing]
    """
    hint_id: str
    required_input_tags: list[str] = Field(default_factory=list)
    output_tags: list[str] = Field(default_factory=list)
    # Rough weight/bulk bounds for plausibility check (optional).
    max_weight: float = 20.0
    max_bulk: float = 20.0
    # Maximum stat value for any single attribute on the output item.
    max_attribute: int = 50
    # Stamina cost to perform the synthesis.
    stamina_cost: float = 10.0


class ConsequencePolicy(BaseModel):
    """
    How strictly actions must leave durable world state.

    Loaded from world.yaml ``consequences:`` block.
    """

    require_sticky: bool = True
    record_traces: bool = True
    trace_limit: int = 24
    # Ticks after establishment during which facts seed beliefs + observation.
    world_memory_window: int = 6
    narrative_only_verbs: list[str] = Field(
        default_factory=lambda: ["wait", "observe", "inspect", "examine", "rest"]
    )
    default_interacted_delta: float = 0.12
    max_world_facts: int = 400


class OpenVerbRule(BaseModel):
    """
    Pack-declared keyword/open verb rule (``open_verbs.yaml``).

    Matched when ``action.verb`` or ``raw_input`` hits ``verbs`` / ``keywords``.
    Either ``handler`` (built-in compiler name) or ``on_success`` template
    effects must be present.
    """

    id: str
    verbs: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    requires: Optional[str] = None  # entity | object
    handler: Optional[str] = None
    contest: Optional["ContestSpec"] = None
    on_success: list["TransitionProposal"] = Field(default_factory=list)
    on_failure: list["TransitionProposal"] = Field(default_factory=list)
    narrative_success: Optional[str] = None
    narrative_failure: Optional[str] = None


class WorldConfig(BaseModel):
    """
    Per-world soft-coded configuration.

    The engine reads from here for any policy that should be authorable
    in data: affordance rules for grounding, NPC policy weights, verb
    templates, future rule packs. Code never embeds world-specific
    opinions.
    """

    consequence_policy: ConsequencePolicy = Field(default_factory=ConsequencePolicy)
    affordances: list[AffordanceRule] = Field(default_factory=list)
    object_affordances: list["ObjectAffordanceRule"] = Field(default_factory=list)
    npc_policy: NpcPolicyConfig = Field(default_factory=NpcPolicyConfig)
    # Keyed by verb string; absent verbs flow through built-in defaults
    # or the pure-freeform path.
    verb_templates: dict[str, VerbTemplate] = Field(default_factory=dict)
    # Keyword/open verb rules from open_verbs.yaml (handler or template effects).
    open_verbs: list["OpenVerbRule"] = Field(default_factory=list)
    # World clock (time-of-day phases derived from tick).
    clock: WorldClock = Field(default_factory=WorldClock)
    # Pack-declared ambient events fired by the WorldClock between turns.
    ambient_events: list[AmbientEvent] = Field(default_factory=list)
    # Latent pressure rules (pressures.yaml) — implicit triggers per tick.
    pressures: list[dict] = Field(default_factory=list)
    # Condition-fired reactive triggers (reactive_triggers.yaml) — emergent beats.
    reactive_triggers: list[dict] = Field(default_factory=list)
    # Free-form bag for additional pack-defined config the engine doesn't
    # interpret directly (passed through to the LM / narrator if useful).
    extra: dict[str, Any] = Field(default_factory=dict)
    # Declarative quest/goal objectives for this world pack.
    goals: list[Goal] = Field(default_factory=list)
    # Spell vocabulary for this world — operations defined in magic.yaml
    # and grimoires found in worlds/*/grimoires/*.yaml.
    # None means magic is not available in this world.
    spell_vocab: Optional[Any] = Field(default=None)  # SpellVocabulary (avoids circular import)
    # Physics/chemistry catalog for this world — materials.yaml + reactions.yaml.
    # None means physics simulation is disabled (pure narrative world).
    physics_config: Optional[Any] = Field(default=None)  # PhysicsConfig (avoids circular import)
    # Item combination / chemistry rules loaded from combinations.yaml.
    combination_rules: list["CombinationRule"] = Field(default_factory=list)
    # Tag-grammar interactions (interaction_rules.yaml) — Qud-style [tag]×[tag].
    interaction_rules: list["InteractionRule"] = Field(default_factory=list)
    # Property-intersection rules (property_interactions.yaml) — fires inside the
    # compiler from entity/object tags, with no verb-keyword requirement.
    property_interactions: list["PropertyInteractionRule"] = Field(default_factory=list)
    # Pack-driven Zone 3 adjudication index (adjudication_index.yaml).
    adjudication_index: list["AdjudicationIndexEntry"] = Field(default_factory=list)
    # Synthesis hints loaded from synthesis_hints.yaml.
    # Each hint tells the engine what output a class of materials can produce
    # (e.g. "wood+metal → tool", "cloth+plant → bandage").
    # Used by _compile_craft to validate LM-proposed objects without
    # requiring every possible recipe to be pre-authored.
    synthesis_hints: list["SynthesisHint"] = Field(default_factory=list)
    # Liquid material catalog (fluids.yaml).
    fluid_catalog: Optional[Any] = Field(default=None)
    # Pathogens, biofluid stains, airborne agents (agents.yaml).
    agent_catalog: Optional[Any] = Field(default=None)
    substance_catalog: Optional[Any] = Field(default=None)
    semantic: SemanticDynamicsConfig = Field(default_factory=SemanticDynamicsConfig)
    # Long-horizon NPC job stacks (npc_plans.yaml).
    npc_plan_templates: list[NpcPlanTemplate] = Field(default_factory=list)


class WorldState(BaseModel):
    """
    Canonical authoritative state.  Nothing outside the spatial simulation,
    relational graph, and the Compiler/Validator should mutate this directly.

    Multi-region design
    -------------------
    The world is a graph of ``Region`` objects, each wrapping a ``SpatialGrid``.
    Entities and objects belong to exactly one region at a time, stored in
    that region's ``grid.entities`` / ``grid.objects``.

    ``world.spatial`` is a backward-compatible property that returns the
    *active* region's grid (the region containing the player).  All engine
    code that reads ``world.spatial`` continues to work unchanged; only the
    compiler's ``apply_transitions`` and the portal-detection path in
    ``_compile_move`` are region-aware.
    """

    world_id: str = Field(default_factory=lambda: f"world_{uuid.uuid4().hex[:8]}")
    tick: int = 0
    # All regions in this world, keyed by region_id string.
    # The primary/starting region has id "region_main" by convention.
    regions: dict[str, "Region"] = Field(default_factory=dict)
    # Portal graph: ordered list of one-way (or bidirectional) region links.
    portals: list["Portal"] = Field(default_factory=list)
    # The region_id the player is currently in.  Keeps world.spatial pointing
    # at the right grid without searching all regions.
    active_region_id: str = "region_main"
    relational: RelationalGraph = Field(default_factory=RelationalGraph)
    event_log: list[Event] = Field(default_factory=list)
    # Metadata
    name: str = "unnamed_world"
    meta: dict[str, Any] = Field(default_factory=dict)
    # Soft-coded per-world configuration loaded from a world pack.
    config: WorldConfig = Field(default_factory=WorldConfig)
    # Replay-stable seed for any place the engine needs randomness that
    # is NOT already keyed off (action_id, tick) — for example ambient
    # event probability rolls in the WorldClock autonomic loop. Combined
    # with the deterministic (event_id, tick) keying used elsewhere,
    # this keeps the whole system replayable from log + initial state.
    rng_seed: int = 0
    # LM-generated prose summaries of past episodes, keyed by entity_id.
    # Populated by episodic_memory.maybe_summarize() every EPISODE_SIZE ticks.
    # Persisted in savegames so long-running sessions retain memory.
    episode_memory: dict[str, list[str]] = Field(default_factory=dict)
    # Active multi-turn conversation threads.  A thread opens when any
    # speaking verb (speak, ask, whisper, …) occurs between two entities and
    # closes when neither party speaks for THREAD_IDLE_TICKS ticks.
    # Managed by dialogue.update_threads().
    active_conversations: list["ConversationThread"] = Field(default_factory=list)
    # DM-adjudicated durable facts and queued delayed effects.
    world_facts: list[WorldFact] = Field(default_factory=list)
    scheduled_effects: list[ScheduledEffect] = Field(default_factory=list)
    # Player-facing story chronicle (episode summaries, oldest first).
    player_chronicle: list[str] = Field(default_factory=list)
    # Cold-tier compressed world milestones (rule-based, replay-safe).
    world_chronicle: list[WorldChronicleEntry] = Field(default_factory=list)

    # ------------------------------------------------------------------
    # Backward-compat shim: world.spatial → active region's grid
    # ------------------------------------------------------------------

    @property
    def spatial(self) -> "SpatialGrid":
        """Return the active region's SpatialGrid.

        All engine code that previously accessed ``world.spatial`` continues
        to work; it now transparently operates on the player's current region.
        """
        region = self.regions.get(self.active_region_id)
        if region is None:
            # Fallback: return whichever region exists (should not happen in
            # a correctly loaded world, but prevents hard crashes during tests).
            for r in self.regions.values():
                return r.grid
            raise KeyError(f"No regions in world; active_region_id={self.active_region_id!r}")
        return region.grid

    # ------------------------------------------------------------------
    # Region lookup helpers
    # ------------------------------------------------------------------

    def region_of_entity(self, entity_id: "EntityId") -> "Region":
        """Return the Region that currently contains ``entity_id``.

        Looks up ``entity.region_id`` for an O(1) path, falling back to a
        linear scan if the denormalized field is stale.
        """
        # Fast path: use the entity's stored region_id
        for region in self.regions.values():
            ent = region.grid.entities.get(entity_id)
            if ent is not None:
                return region
        raise KeyError(f"Entity {entity_id!r} not found in any region")

    def all_entities(self) -> dict["EntityId", "EntityState"]:
        """Flat view of every entity across all regions."""
        result: dict = {}
        for region in self.regions.values():
            result.update(region.grid.entities)
        return result

    def all_objects(self) -> dict["ObjectId", "ObjectState"]:
        """Flat view of every object across all regions."""
        result: dict = {}
        for region in self.regions.values():
            result.update(region.grid.objects)
        return result

    def grid_for_entity(self, entity_id: "EntityId") -> "SpatialGrid":
        """Spatial grid containing ``entity_id`` (multi-region safe)."""
        return self.region_of_entity(entity_id).grid

    def portal_at(self, region_id: str, coord: "Coord") -> "Optional[Portal]":
        """Return the portal whose ``from_coord`` matches ``coord`` in ``region_id``,
        or the reverse portal if this is a bidirectional portal destination."""
        for p in self.portals:
            if p.from_region == region_id and p.from_coord == coord:
                return p
            if p.bidirectional and p.to_region == region_id and p.to_coord == coord:
                # Return a synthetic reverse portal
                return Portal(
                    portal_id=p.portal_id + "_rev",
                    label=p.label,
                    from_region=region_id,
                    from_coord=coord,
                    to_region=p.from_region,
                    to_coord=p.from_coord,
                    bidirectional=False,
                    requires_key=p.requires_key,
                )
        return None


# ---------------------------------------------------------------------------
# Errors (raised, never silently swallowed)
# ---------------------------------------------------------------------------


class SimError(Exception):
    """Base class for all simulation errors."""


class MalformedActionError(SimError):
    """LM output could not be parsed into a valid SemanticAction."""


class ProjectionFirewallError(SimError):
    """SemanticAction references an entity not present in the projection."""


class PhysicsViolationError(SimError):
    """A transition would violate physical simulation invariants."""
