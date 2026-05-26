"""
Fluids, containers, spills, and tile wetness.

Canonical state lives in:
  - object.meta["fluid"]  — {material, volume_ml, capacity_ml}
  - tile.env["fluid_spill"] — {material, volume_ml}
  - entity.tags — "wet" when soaked from spill/accident

Pour/drink compilers call into this module; bodily.py handles bladder pressure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

from .schemas import (
    Coord,
    EntityId,
    EntityState,
    ObjectId,
    ObjectState,
    Transition,
    TransitionKind,
)

if TYPE_CHECKING:
    from .schemas import SpatialGrid, WorldState

DEFAULT_POUR_ML = 150.0
DEFAULT_DRINK_ML = 120.0
RECEPTACLE_TAGS = frozenset({"receptacle", "cup", "mug", "glass", "drink"})
POUR_SOURCE_TAGS = frozenset({"pour_source", "tap", "cask", "keg", "barrel"})


@dataclass
class LiquidDef:
    name: str
    display: str = ""
    thirst_per_100ml: float = 15.0
    hunger_per_100ml: float = 0.0
    bladder_per_100ml: float = 20.0
    intox_per_100ml: float = 0.0
    drinkable: bool = True


@dataclass
class FluidCatalog:
    liquids: dict[str, LiquidDef] = field(default_factory=dict)

    def get(self, material: str) -> Optional[LiquidDef]:
        return self.liquids.get((material or "").lower().strip())


def load_fluid_catalog(pack_path: Any) -> FluidCatalog:
    """Load worlds/<pack>/fluids.yaml, falling back to default pack."""
    from pathlib import Path
    import yaml

    catalog = FluidCatalog()
    for base in (Path(pack_path), Path(pack_path).parent / "default", Path("worlds/default")):
        path = base / "fluids.yaml"
        if not path.is_file():
            continue
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for name, raw in (doc.get("liquids") or {}).items():
            if not isinstance(raw, dict):
                continue
            key = str(name).lower()
            catalog.liquids[key] = LiquidDef(
                name=key,
                display=str(raw.get("display", key)),
                thirst_per_100ml=float(raw.get("thirst_per_100ml", 15.0)),
                hunger_per_100ml=float(raw.get("hunger_per_100ml", 0.0)),
                bladder_per_100ml=float(raw.get("bladder_per_100ml", 20.0)),
                intox_per_100ml=float(raw.get("intox_per_100ml", 0.0)),
                drinkable=bool(raw.get("drinkable", True)),
            )
        if catalog.liquids:
            break
    for base in (Path(pack_path), Path(pack_path).parent / "default", Path("worlds/default")):
        path = base / "biofluids.yaml"
        if not path.is_file():
            continue
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for name, raw in (doc.get("liquids") or {}).items():
            if not isinstance(raw, dict):
                continue
            key = str(name).lower()
            catalog.liquids[key] = LiquidDef(
                name=key,
                display=str(raw.get("display", key)),
                thirst_per_100ml=float(raw.get("thirst_per_100ml", 0.0)),
                hunger_per_100ml=float(raw.get("hunger_per_100ml", 0.0)),
                bladder_per_100ml=float(raw.get("bladder_per_100ml", 0.0)),
                intox_per_100ml=float(raw.get("intox_per_100ml", 0.0)),
                drinkable=bool(raw.get("drinkable", False)),
            )
        if any(k in catalog.liquids for k in ("blood", "urine", "feces")):
            break
    if not catalog.liquids:
        catalog.liquids["water"] = LiquidDef(name="water", display="water")
    return catalog


def fluid_state(subject: ObjectState) -> dict[str, Any]:
    """Return mutable fluid dict on object (may be empty)."""
    raw = subject.meta.get("fluid")
    if isinstance(raw, dict):
        return raw
    return {}


def fluid_capacity_ml(obj: ObjectState) -> float:
    f = fluid_state(obj)
    cap = f.get("capacity_ml")
    if cap is not None:
        return max(0.0, float(cap))
    if obj.is_container and obj.carry_capacity > 0:
        return obj.carry_capacity * 200.0
    tags = {t.lower() for t in obj.tags}
    if tags.intersection(RECEPTACLE_TAGS):
        return 300.0
    return 0.0


def fluid_volume_ml(obj: ObjectState) -> float:
    return max(0.0, float(fluid_state(obj).get("volume_ml", 0.0)))


def fluid_material(obj: ObjectState) -> str:
    return str(fluid_state(obj).get("material") or "").lower()


def init_object_fluid(
    obj: ObjectState,
    *,
    capacity_ml: float,
    volume_ml: float = 0.0,
    material: str = "",
) -> None:
    obj.meta["fluid"] = {
        "capacity_ml": max(0.0, float(capacity_ml)),
        "volume_ml": max(0.0, min(float(volume_ml), float(capacity_ml))),
        "material": (material or "").lower(),
    }


def is_receptacle(obj: ObjectState) -> bool:
    tags = {t.lower() for t in obj.tags}
    if tags.intersection(RECEPTACLE_TAGS):
        return True
    if fluid_capacity_ml(obj) > 0 and tags.intersection({"drink", "receptacle", "cup", "mug"}):
        return True
    return False


def is_pour_source(obj: ObjectState) -> bool:
    tags = {t.lower() for t in obj.tags}
    if tags.intersection(POUR_SOURCE_TAGS):
        return True
    if obj.meta.get("pour_source"):
        return True
    # Full cask with drink tag
    if "drink" in tags and fluid_volume_ml(obj) > 50:
        return True
    return False


def describe_fluid(obj: ObjectState, catalog: Optional[FluidCatalog]) -> str:
    vol = fluid_volume_ml(obj)
    cap = fluid_capacity_ml(obj)
    if cap <= 0:
        return ""
    mat = fluid_material(obj)
    if vol <= 0.01:
        return "empty"
    liq = catalog.get(mat) if catalog else None
    label = liq.display if liq else (mat or "liquid")
    ratio = vol / cap
    if ratio >= 0.95:
        level = "full"
    elif ratio >= 0.65:
        level = "mostly full"
    elif ratio >= 0.35:
        level = "half full"
    else:
        level = "partly full"
    return f"{level} of {label}"


def _set_object_fluid(
    obj: ObjectState,
    volume_ml: float,
    material: str,
) -> None:
    cap = fluid_capacity_ml(obj)
    vol = max(0.0, min(float(volume_ml), cap))
    obj.meta["fluid"] = {
        "capacity_ml": cap,
        "volume_ml": vol,
        "material": (material or "").lower() if vol > 0 else "",
    }
    tags = list(obj.tags)
    if vol > 0 and "contains_liquid" not in tags:
        tags.append("contains_liquid")
    if vol <= 0 and "contains_liquid" in tags:
        tags.remove("contains_liquid")
    obj.tags = tags


def apply_fluid_to_object(obj: ObjectState, volume_ml: float, material: str) -> None:
    _set_object_fluid(obj, volume_ml, material)


def apply_wet_tile(grid: "SpatialGrid", coord: Coord, material: str, volume_ml: float) -> None:
    """Mark a tile wet with ``material`` (volume in mL).

    Writes the canonical ``tile.env["fields"]["ground"][material]`` entry
    AND the legacy ``fluid_spill`` mirror (kept for back-compat readers).
    Both are kept in sync so callers that walk only the canonical field
    layer (npc_planner, hazard_avoidance, pressure_eval, etc.) see the
    spill immediately without waiting for the next field tick.
    """
    tile = grid.tile_at(coord)
    mat = (material or "water").lower()
    vol = float(volume_ml)

    # Canonical: write/merge into the ground field layer.
    fields_bucket = tile.env.setdefault("fields", {})
    if not isinstance(fields_bucket, dict):
        fields_bucket = {}
        tile.env["fields"] = fields_bucket
    ground_layer = dict(fields_bucket.get("ground") or {})
    entry = dict(ground_layer.get(mat) or {})
    prev_vol = float((entry.get("meta") or {}).get("volume_ml", 0.0))
    new_vol = prev_vol + vol
    # Amount is a normalized 0..100 reading; mL feeds the meta payload so
    # downstream consumers (cleaning, bodily evaporation) can still see
    # the physical quantity.
    entry["amount"] = min(100.0, float(entry.get("amount", 0.0)) + max(vol * 0.4, 30.0 if vol > 0 else 0.0))
    entry["since_tick"] = entry.get("since_tick")
    meta = dict(entry.get("meta") or {})
    meta["volume_ml"] = new_vol
    meta["cause"] = meta.get("cause", "wet_floor")
    entry["meta"] = meta
    ground_layer[mat] = entry
    fields_bucket["ground"] = ground_layer

    # Legacy mirror — preserved so older readers keep working.  Phase 2
    # of the fields migration removes these branches.
    spill = dict(tile.env.get("fluid_spill") or {})
    spill["material"] = mat
    spill["volume_ml"] = new_vol
    tile.env["fluid_spill"] = spill
    tile.env["wet"] = True
    if "wet" not in tile.tags:
        tile.tags = list(tile.tags) + ["wet"]
    grid.set_tile(coord, tile)


def wet_entity(entity: EntityState, material: str) -> None:
    if "wet" not in entity.tags:
        entity.tags = list(entity.tags) + ["wet"]
    entity.meta["wet_material"] = material
    entity.meta["wet_since_tick"] = entity.meta.get("wet_since_tick")


def find_pour_source(
    actor: EntityState,
    grid: "SpatialGrid",
    *,
    max_dist: int = 2,
) -> Optional[ObjectState]:
    best: Optional[ObjectState] = None
    best_d = max_dist + 1
    for obj in grid.objects.values():
        if obj.position is None or not is_pour_source(obj):
            continue
        d = actor.position.manhattan(obj.position)
        if d <= max_dist and d < best_d:
            best, best_d = obj, d
    return best


def find_receptacle_near(
    grid: "SpatialGrid",
    center: Coord,
    *,
    max_dist: int = 2,
) -> Optional[ObjectState]:
    best: Optional[ObjectState] = None
    best_d = max_dist + 1
    for obj in grid.objects.values():
        if obj.position is None or not is_receptacle(obj):
            continue
        d = center.manhattan(obj.position)
        if d <= max_dist and d < best_d:
            best, best_d = obj, d
    return best


def resolve_receptacle_for_pour(
    grid: "SpatialGrid",
    actor: EntityState,
    target_raw: Optional[str],
) -> tuple[Optional[ObjectState], Optional[str]]:
    """
    Resolve pour target. Returns (receptacle, rejection_detail).
    """
    if target_raw:
        obj = grid.objects.get(ObjectId(str(target_raw)))
        if obj is not None and is_receptacle(obj):
            return obj, None
        ent = grid.entities.get(EntityId(str(target_raw)))
        if ent is None:
            ent = _entity_by_name(grid, str(target_raw))
        if ent is not None:
            cup = find_receptacle_near(grid, ent.position, max_dist=2)
            if cup is not None:
                return cup, None
            return None, f"No cup or glass near {ent.name} to pour into."
        return None, "That is not a valid pour target."

    cup = find_receptacle_near(grid, actor.position, max_dist=2)
    if cup is not None:
        return cup, None
    return None, "Nothing to pour into — need a cup, mug, or glass in reach."


def _entity_by_name(grid: "SpatialGrid", name: str) -> Optional[EntityState]:
    lower = name.lower().strip()
    for ent in grid.entities.values():
        if ent.name.lower() == lower:
            return ent
    return None


def transfer_pour(
    world: "WorldState",
    actor: EntityState,
    source: ObjectState,
    receptacle: ObjectState,
    pour_ml: float = DEFAULT_POUR_ML,
) -> tuple[list[Transition], str]:
    """Build pour transitions (applied by apply_transitions)."""
    catalog = world.config.fluid_catalog
    transitions: list[Transition] = []

    src_mat = fluid_material(source) or "ale"
    liq = catalog.get(src_mat) if catalog else None
    if liq is None:
        liq = LiquidDef(name=src_mat or "ale", display=src_mat or "ale")

    available = fluid_volume_ml(source)
    if available < 1.0 and is_pour_source(source):
        available = pour_ml * 10
    amount = min(pour_ml, available)
    if amount <= 0:
        return [], "empty_source"

    recv_cap = fluid_capacity_ml(receptacle)
    if recv_cap <= 0:
        return [], "not_receptacle"

    recv_vol = fluid_volume_ml(receptacle)
    recv_mat = fluid_material(receptacle)
    space = recv_cap - recv_vol
    accepted = min(amount, max(0.0, space))
    overflow = amount - accepted

    if accepted > 0:
        new_vol = recv_vol + accepted
        new_mat = recv_mat if recv_vol > 0 else src_mat
        transitions.append(Transition(
            kind=TransitionKind.FLUID_CHANGED,
            payload={
                "target_kind": "object",
                "object_id": str(receptacle.object_id),
                "source_id": str(source.object_id),
                "actor_id": str(actor.entity_id),
                "material": new_mat,
                "volume_ml": new_vol,
                "capacity_ml": recv_cap,
                "cause": "pour",
            },
        ))

    if not is_pour_source(source) or available < 1e8:
        new_src = max(0.0, fluid_volume_ml(source) - amount)
        transitions.append(Transition(
            kind=TransitionKind.FLUID_CHANGED,
            payload={
                "target_kind": "object",
                "object_id": str(source.object_id),
                "material": src_mat if new_src > 0 else "",
                "volume_ml": new_src,
                "capacity_ml": fluid_capacity_ml(source),
                "cause": "pour_source",
            },
        ))

    if overflow > 0 and receptacle.position is not None:
        transitions.append(Transition(
            kind=TransitionKind.FLUID_CHANGED,
            payload={
                "target_kind": "tile",
                "x": receptacle.position.x,
                "y": receptacle.position.y,
                "material": src_mat,
                "volume_ml": overflow,
                "cause": "overflow",
                "wet": True,
            },
        ))
        # Also track spill as a field substance so the field layer sees it.
        transitions.append(Transition(
            kind=TransitionKind.FIELD_CHANGED,
            payload={
                "medium": "tile.ground",
                "substance": src_mat,
                "operation": "add",
                "amount": min(60.0, overflow * 0.4),
                "x": receptacle.position.x,
                "y": receptacle.position.y,
                "cause": "overflow",
                "tick": 0,  # filled in by apply_transitions from world.tick
            },
        ))

    if accepted > 0 and overflow > 0:
        return transitions, "overflow"
    if accepted > 0:
        return transitions, "success"
    return transitions, "overflow"


def drink_from_container(
    world: "WorldState",
    actor: EntityState,
    container: ObjectState,
    drink_ml: float = DEFAULT_DRINK_ML,
) -> tuple[list[Transition], str]:
    catalog = world.config.fluid_catalog
    vol = fluid_volume_ml(container)
    if vol <= 0:
        return [], "empty"

    mat = fluid_material(container)
    liq = catalog.get(mat) if catalog else None
    if liq is None:
        liq = LiquidDef(name=mat or "ale", display=mat or "ale")
    if not liq.drinkable:
        return [], "not_drinkable"

    amount = min(drink_ml, vol)
    new_vol = max(0.0, vol - amount)

    thirst_delta = liq.thirst_per_100ml * (amount / 100.0)
    hunger_delta = liq.hunger_per_100ml * (amount / 100.0)
    bladder_delta = liq.bladder_per_100ml * (amount / 100.0)
    intox_delta = int(liq.intox_per_100ml * (amount / 100.0))

    transitions = [
        Transition(
            kind=TransitionKind.FLUID_CHANGED,
            payload={
                "target_kind": "object",
                "object_id": str(container.object_id),
                "actor_id": str(actor.entity_id),
                "material": mat if new_vol > 0.01 else "",
                "volume_ml": new_vol,
                "capacity_ml": fluid_capacity_ml(container),
                "cause": "drink",
            },
        ),
        Transition(
            kind=TransitionKind.NEED_CHANGED,
            payload={
                "entity_id": str(actor.entity_id),
                "need": "thirst",
                "delta": thirst_delta,
                "cause": "drink",
            },
        ),
        Transition(
            kind=TransitionKind.NEED_CHANGED,
            payload={
                "entity_id": str(actor.entity_id),
                "need": "hunger",
                "delta": hunger_delta,
                "cause": "drink",
            },
        ),
        Transition(
            kind=TransitionKind.NEED_CHANGED,
            payload={
                "entity_id": str(actor.entity_id),
                "need": "bladder",
                "delta": bladder_delta,
                "cause": "drink",
            },
        ),
    ]
    if intox_delta > 0:
        transitions.append(Transition(
            kind=TransitionKind.ENTITY_STAT_CHANGED,
            payload={
                "entity_id": str(actor.entity_id),
                "stat": "intoxication",
                "delta": float(intox_delta),
                "cause": "drink",
            },
        ))
    return transitions, "success"


def tile_is_wet(grid: "SpatialGrid", coord: Coord) -> bool:
    """True iff ``coord`` has wet material per canonical field layer or legacy mirror."""
    tile = grid.tile_at(coord)
    # Canonical first — ground field layer with any positive-volume entry.
    fields_bucket = tile.env.get("fields")
    if isinstance(fields_bucket, dict):
        ground_layer = fields_bucket.get("ground")
        if isinstance(ground_layer, dict):
            for entry in ground_layer.values():
                if not isinstance(entry, dict):
                    continue
                meta = entry.get("meta") or {}
                if float(meta.get("volume_ml", 0.0)) > 0.0:
                    return True
                if float(entry.get("amount", 0.0)) > 0.0:
                    return True
    if tile.env.get("wet") or tile.env.get("fluid_spill"):
        return True
    return "wet" in tile.tags
