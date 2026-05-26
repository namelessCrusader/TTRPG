"""
Generic field / substance layer for physical world state.

Substances occupy *media* (tile ground, tile air, region atmosphere, entity
surfaces, object contents). All mutations use FIELD_CHANGED transitions.

Legacy storage (tile.env contamination/airborne, fluid_spill) is kept in sync
so existing code and tests keep working during migration. See FIELD_SYSTEM.md.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

from .field_coords import (
    attach_field_coords,
    coord_from_field_payload,
    coord_from_tile_key,
    field_spread_neighbors,
    field_tile_coord,
    fluid_tile_payload,
    iter_tiles,
)
from .schemas import (
    Coord,
    EntityId,
    EntityState,
    Transition,
    TransitionKind,
    coord_key,
)

if TYPE_CHECKING:
    from .schemas import SpatialGrid, Tile, WorldState

# ── Media identifiers (pack-extendable) ───────────────────────────────────

MEDIA_TILE_GROUND = "tile.ground"
MEDIA_TILE_AIR = "tile.air"
MEDIA_REGION_ATMOSPHERE = "region.atmosphere"
MEDIA_ENTITY_SURFACE = "entity.surface"
MEDIA_ENTITY_INTERNAL = "entity.internal"
MEDIA_OBJECT_CONTENTS = "object.contents"
MEDIA_OBJECT_SURFACE = "object.surface"

AIR_MEDIA = frozenset({MEDIA_TILE_AIR, MEDIA_REGION_ATMOSPHERE})


@dataclass
class SubstanceDef:
    name: str
    display: str = ""
    category: str = "material"
    media: list[str] = field(default_factory=list)
    decay_per_tick: float = 2.0
    spread_neighbor: float = 0.0
    spread_radius: int = 0
    airborne_ttl_ticks: int = 0
    routes: list[str] = field(default_factory=list)
    infection_per_100: float = 0.0
    incubation_ticks: int = 0
    illness_condition: str = ""
    illness_damage_per_tick: int = 0
    illness_duration_ticks: int = 30
    carries: str = ""
    hosts: list[str] = field(default_factory=list)
    wet_floor: bool = False
    stain: bool = False
    fluid_material: str = ""
    pathogen: bool = False


@dataclass
class EmitterDef:
    id: str
    when: dict[str, Any] = field(default_factory=dict)
    targets: dict[str, Any] = field(default_factory=dict)
    deposit: dict[str, Any] = field(default_factory=dict)
    also_wet: dict[str, Any] = field(default_factory=dict)


@dataclass
class SubstanceCatalog:
    substances: dict[str, SubstanceDef] = field(default_factory=dict)
    emitters: list[EmitterDef] = field(default_factory=list)

    def get(self, name: str) -> Optional[SubstanceDef]:
        return self.substances.get((name or "").lower().strip())


def _parse_substance(name: str, raw: dict[str, Any]) -> SubstanceDef:
    dyn = raw.get("dynamics") or {}
    exp = raw.get("exposure") or {}
    ill = exp.get("illness") or {}
    eff = raw.get("effects") or {}
    cat = str(raw.get("category", "material"))
    return SubstanceDef(
        name=name.lower(),
        display=str(raw.get("display", name)),
        category=cat,
        media=[str(m) for m in (raw.get("media") or [])],
        decay_per_tick=float(dyn.get("decay_per_tick", raw.get("decays_per_tick", 2.0))),
        spread_neighbor=float(dyn.get("spread_neighbor", raw.get("spread_neighbor", 0.0))),
        spread_radius=int(dyn.get("spread_radius", raw.get("spread_radius", 0))),
        airborne_ttl_ticks=int(
            dyn.get("airborne_ttl_ticks", raw.get("airborne_duration_ticks", 0))
        ),
        routes=[str(r).lower() for r in (exp.get("routes") or raw.get("routes") or [])],
        infection_per_100=float(
            exp.get("infection_per_100", raw.get("infection_per_100", 0.0))
        ),
        incubation_ticks=int(exp.get("incubation_ticks", raw.get("incubation_ticks", 0))),
        illness_condition=str(ill.get("condition", raw.get("illness_condition", ""))),
        illness_damage_per_tick=int(
            ill.get("damage_per_tick", raw.get("illness_damage_per_tick", 0))
        ),
        illness_duration_ticks=int(
            ill.get("duration_ticks", raw.get("illness_duration_ticks", 30))
        ),
        carries=str(eff.get("carries", raw.get("carries_pathogen", ""))).lower(),
        hosts=[str(h).lower() for h in (raw.get("hosts") or raw.get("host_agents") or [])],
        wet_floor=bool(eff.get("wet_floor", False)),
        stain=bool(eff.get("stain", raw.get("stain", False))),
        fluid_material=str(eff.get("fluid_material", raw.get("fluid_material", ""))).lower(),
        pathogen=cat == "pathogen" or bool(raw.get("pathogen", False)),
    )


def load_substance_catalog(pack_path: Any) -> SubstanceCatalog:
    from pathlib import Path
    import yaml

    catalog = SubstanceCatalog()
    bases = (Path(pack_path), Path(pack_path).parent / "default", Path("worlds/default"))

    for base in bases:
        path = base / "substances.yaml"
        if path.is_file():
            doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            for name, raw in (doc.get("substances") or {}).items():
                if isinstance(raw, dict):
                    key = str(name).lower()
                    catalog.substances[key] = _parse_substance(key, raw)
            break

    # Legacy agents.yaml merge (older packs)
    for base in bases:
        path = base / "agents.yaml"
        if not path.is_file():
            continue
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for name, raw in (doc.get("agents") or {}).items():
            if not isinstance(raw, dict):
                continue
            key = str(name).lower()
            if key in catalog.substances:
                continue
            sdef = _parse_substance(key, raw)
            if raw.get("airborne"):
                sdef.airborne_ttl_ticks = int(raw.get("airborne_duration_ticks", 10))
                if MEDIA_TILE_AIR not in sdef.media:
                    sdef.media.append(MEDIA_TILE_AIR)
            if sdef.category == "biofluid" and MEDIA_TILE_GROUND not in sdef.media:
                sdef.media.append(MEDIA_TILE_GROUND)
            catalog.substances[key] = sdef
        break

    for base in bases:
        path = base / "emitters.yaml"
        if not path.is_file():
            continue
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for eid, raw in (doc.get("emitters") or {}).items():
            if isinstance(raw, dict):
                catalog.emitters.append(EmitterDef(
                    id=str(eid),
                    when=dict(raw.get("when") or {}),
                    targets=dict(raw.get("targets") or {}),
                    deposit=dict(raw.get("deposit") or {}),
                    also_wet=dict(raw.get("also_wet") or {}),
                ))
        break

    return catalog


def _seeded_float(seed: str) -> float:
    digest = hashlib.sha256(seed.encode()).digest()
    return int.from_bytes(digest[:4], "big") / (2**32)


# ── Storage (canonical fields + legacy mirror) ────────────────────────────


def _fields_bucket(env: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    raw = env.get("fields")
    if isinstance(raw, dict):
        return raw
    return {}


def get_layer(
    env: dict[str, Any],
    medium: str,
    *,
    legacy_tile: Optional["Tile"] = None,
) -> dict[str, dict[str, Any]]:
    """Read substance layer; hydrate from legacy keys if needed."""
    bucket = _fields_bucket(env)
    medium_key = medium.split(".")[-1] if "." in medium else medium
    layer = dict(bucket.get(medium_key) or {})

    if legacy_tile is not None and not layer:
        if medium == MEDIA_TILE_GROUND:
            leg = legacy_tile.env.get("contamination")
            if isinstance(leg, dict):
                for aid, data in leg.items():
                    layer[aid] = {
                        "amount": float(data.get("concentration", 0)),
                        "since_tick": data.get("since_tick"),
                        "meta": {"cause": data.get("cause", "")},
                    }
            spill = legacy_tile.env.get("fluid_spill")
            if isinstance(spill, dict) and spill.get("material"):
                mat = str(spill["material"]).lower()
                layer.setdefault(mat, {"amount": 30.0, "meta": {"volume_ml": spill.get("volume_ml")}})
        elif medium == MEDIA_TILE_AIR:
            leg = legacy_tile.env.get("airborne")
            if isinstance(leg, dict):
                for aid, data in leg.items():
                    layer[aid] = {
                        "amount": float(data.get("concentration", 0)),
                        "meta": {
                            "remaining_ticks": data.get("remaining_ticks", 10),
                            "cause": data.get("cause", ""),
                        },
                    }
    return layer


def _write_layer(env: dict[str, Any], medium: str, layer: dict[str, dict[str, Any]]) -> None:
    bucket = _fields_bucket(env)
    medium_key = medium.split(".")[-1] if "." in medium else medium
    if layer:
        bucket[medium_key] = layer
        env["fields"] = bucket
    else:
        bucket.pop(medium_key, None)
        env["fields"] = bucket if bucket else None
        if not bucket:
            env.pop("fields", None)


def _sync_legacy_tile(tile: "Tile", medium: str, layer: dict[str, dict[str, Any]]) -> None:
    """Mirror canonical fields → legacy contamination / airborne / fluid_spill."""
    if medium == MEDIA_TILE_GROUND:
        cont: dict[str, Any] = {}
        primary_fluid = None
        max_vol = 0.0
        for sid, data in layer.items():
            amt = float(data.get("amount", 0))
            if amt <= 0.5:
                continue
            cont[sid] = {
                "concentration": amt,
                "since_tick": data.get("since_tick"),
                "cause": (data.get("meta") or {}).get("cause", ""),
            }
            meta = data.get("meta") or {}
            vol = float(meta.get("volume_ml", amt * 2))
            if vol > max_vol:
                max_vol = vol
                primary_fluid = sid
        if cont:
            tile.env["contamination"] = cont
        else:
            tile.env.pop("contamination", None)
        if primary_fluid and max_vol > 0:
            tile.env["fluid_spill"] = {"material": primary_fluid, "volume_ml": max_vol}
            tile.env["wet"] = True
            if "wet" not in tile.tags:
                tile.tags = list(tile.tags) + ["wet"]
        elif not cont:
            if not tile.env.get("wet"):
                tile.env.pop("fluid_spill", None)
    elif medium == MEDIA_TILE_AIR:
        air: dict[str, Any] = {}
        for sid, data in layer.items():
            amt = float(data.get("amount", 0))
            if amt <= 0.5:
                continue
            meta = data.get("meta") or {}
            air[sid] = {
                "concentration": amt,
                "remaining_ticks": int(meta.get("remaining_ticks", 10)),
                "cause": meta.get("cause", ""),
            }
        if air:
            tile.env["airborne"] = air
        else:
            tile.env.pop("airborne", None)


def apply_field_transition(world: "WorldState", t: Transition) -> None:
    """Apply FIELD_CHANGED (or legacy contamination payload mapped here)."""
    p = t.payload
    medium = str(p.get("medium", ""))
    if not medium and p.get("target_kind") == "air":
        medium = MEDIA_TILE_AIR
    elif not medium and p.get("target_kind") in ("tile", "tile_dry"):
        medium = MEDIA_TILE_GROUND
    elif not medium:
        medium = MEDIA_TILE_GROUND

    substance = str(p.get("substance", p.get("agent", ""))).lower()
    if not substance:
        return

    op = str(p.get("operation", "add"))
    amount = float(p.get("amount", p.get("concentration", 0.0)))
    tick = int(p.get("tick", world.tick))
    cause = str(p.get("cause", ""))
    catalog = world.config.substance_catalog

    grid = world.spatial
    region_id = str(p.get("region_id", world.active_region_id))

    if medium.startswith("tile."):
        coord = coord_from_field_payload(p)
        if coord is None:
            return
        tile = grid.tile_at(field_tile_coord(coord))
        layer = get_layer(tile.env, medium, legacy_tile=tile)

        if op == "clear" or (op == "set" and amount <= 0):
            layer.pop(substance, None)
        elif op in ("add", "deposit", "spread"):
            entry = dict(layer.get(substance) or {})
            entry["amount"] = min(100.0, float(entry.get("amount", 0)) + amount)
            entry["since_tick"] = tick
            meta = dict(entry.get("meta") or {})
            meta["cause"] = cause or meta.get("cause", "")
            if medium in AIR_MEDIA:
                meta["remaining_ticks"] = int(
                    p.get("remaining_ticks", meta.get("remaining_ticks", 10))
                )
            entry["meta"] = meta
            layer[substance] = entry
        elif op == "set":
            layer[substance] = {
                "amount": amount,
                "since_tick": tick,
                "meta": {
                    "cause": cause,
                    **({"remaining_ticks": int(p["remaining_ticks"])} if "remaining_ticks" in p else {}),
                },
            }

        _write_layer(tile.env, medium, layer)
        _sync_legacy_tile(tile, medium, layer)

        sdef = catalog.get(substance) if catalog else None
        if sdef and sdef.stain and "stained" not in tile.tags:
            tile.tags = list(tile.tags) + ["stained"]
        # NOTE: wet_floor mirroring is handled by ``_sync_legacy_tile``
        # (writes ``fluid_spill`` + sets the wet tag for the dominant
        # substance).  Calling ``apply_wet_tile`` here would re-deposit
        # the substance into the canonical field layer in a feedback
        # loop, so it's intentionally NOT invoked.
        grid.set_tile(field_tile_coord(coord), tile)

    elif medium == MEDIA_ENTITY_SURFACE or medium == MEDIA_ENTITY_INTERNAL:
        eid = EntityId(str(p.get("entity_id", "")))
        ent = grid.entities.get(eid)
        if ent is None:
            return
        key = "surface" if medium == MEDIA_ENTITY_SURFACE else "internal"
        fields = ent.meta.setdefault("fields", {})
        layer = dict(fields.get(key) or {})
        if op == "clear" or amount <= 0:
            layer.pop(substance, None)
        else:
            entry = dict(layer.get(substance) or {})
            entry["amount"] = min(100.0, float(entry.get("amount", 0)) + amount)
            layer[substance] = entry
        if layer:
            fields[key] = layer
        else:
            fields.pop(key, None)
        if medium == MEDIA_ENTITY_INTERNAL and substance and op != "clear":
            inf = ent.meta.get("infection")
            if not inf and p.get("infection_payload"):
                ent.meta["infection"] = p["infection_payload"]


def build_field_transition(
    medium: str,
    substance: str,
    amount: float,
    *,
    coord: Optional[Coord] = None,
    entity_id: Optional[str] = None,
    operation: str = "add",
    cause: str = "",
    tick: int = 0,
    remaining_ticks: Optional[int] = None,
) -> Transition:
    payload: dict[str, Any] = {
        "medium": medium,
        "substance": substance,
        "operation": operation,
        "amount": amount,
        "cause": cause,
        "tick": tick,
    }
    if coord is not None:
        attach_field_coords(payload, coord)
    if entity_id is not None:
        payload["entity_id"] = entity_id
    if remaining_ticks is not None:
        payload["remaining_ticks"] = remaining_ticks
    return Transition(kind=TransitionKind.FIELD_CHANGED, payload=payload)


def deposit_at(
    coord: Coord,
    medium: str,
    substance: str,
    amount: float,
    *,
    cause: str,
    tick: int,
    remaining_ticks: Optional[int] = None,
) -> Transition:
    return build_field_transition(
        medium, substance, amount,
        coord=coord, cause=cause, tick=tick,
        remaining_ticks=remaining_ticks,
    )


# ── Public read helpers (gameplay code should prefer these over raw env dicts) ──


def field_amount(
    world: "WorldState",
    coord: Coord,
    medium: str,
    substance: str,
    *,
    region_id: Optional[str] = None,
) -> float:
    """Return the canonical 0..100 amount of ``substance`` at ``coord``.

    Falls back to the legacy mirror (``tile.env.contamination`` /
    ``airborne`` / ``fluid_spill``) when the field layer is empty so
    pre-migration packs keep working.  Use this in gameplay code instead
    of reaching into ``tile.env`` directly.
    """
    region = (
        world.regions.get(region_id) if region_id and world.regions else None
    )
    if region is None:
        region = world.regions.get(world.active_region_id) if world.regions else None
    grid = region.grid if region is not None else world.spatial
    try:
        tile = grid.tile_at(coord)
    except Exception:
        return 0.0
    layer = get_layer(tile.env, medium, legacy_tile=tile)
    entry = layer.get((substance or "").lower())
    if not isinstance(entry, dict):
        return 0.0
    return float(entry.get("amount", 0.0))


def tile_smoke(
    world: "WorldState",
    coord: Coord,
    *,
    region_id: Optional[str] = None,
) -> float:
    """Effective smoke concentration at ``coord``.

    Combines the per-tile air layer (local smoke) with the region
    atmosphere (global smoke); takes the max because the higher reading
    is the one that drives hazard behaviour.
    """
    local = field_amount(world, coord, MEDIA_TILE_AIR, "smoke", region_id=region_id)
    atmosphere = region_atmosphere_amount(world, "smoke", region_id=region_id)
    return max(local, atmosphere)


# ── High-level helpers (verbs call these, not scenario modules) ───────────


def deposit_biofluid_relief(
    world: "WorldState",
    entity: EntityState,
    *,
    privy: bool,
) -> list[Transition]:
    coord = entity.position
    tick = world.tick
    out: list[Transition] = [
        deposit_at(coord, MEDIA_TILE_GROUND, "urine", 55.0 if not privy else 8.0, cause="relieve", tick=tick),
    ]
    if not privy:
        # Only emit a wet floor event for public accidents, not privy use.
        out.append(Transition(
            kind=TransitionKind.FLUID_CHANGED,
            payload=fluid_tile_payload(
                coord,
                target_kind="tile",
                material="urine",
                volume_ml=180.0,
                cause="relieve",
                wet=True,
            ),
        ))
    if not privy or _seeded_float(f"{world.rng_seed}:feces:{entity.entity_id}:{tick}") < 0.4:
        out.append(deposit_at(coord, MEDIA_TILE_GROUND, "feces", 35.0 if not privy else 10.0, cause="relieve", tick=tick))
    catalog = world.config.substance_catalog
    if not privy and catalog:
        for sid in ("urine", "feces"):
            sdef = catalog.get(sid)
            if sdef and sdef.carries:
                out.append(deposit_at(coord, MEDIA_TILE_GROUND, sdef.carries, 20.0, cause=f"{sid}_carrier", tick=tick))
    return out


def deposit_bleed(
    world: "WorldState",
    entity: EntityState,
    *,
    damage: int,
    cause: str = "wound",
) -> list[Transition]:
    coord = entity.position
    tick = world.tick
    conc = min(100.0, 15.0 + damage * 4.0)
    out = [
        deposit_at(coord, MEDIA_TILE_GROUND, "blood", conc, cause=cause, tick=tick),
        Transition(
            kind=TransitionKind.FLUID_CHANGED,
            payload=fluid_tile_payload(
                coord,
                target_kind="tile",
                material="blood",
                volume_ml=max(25.0, float(damage) * 12.0),
                cause=cause,
                wet=True,
            ),
        ),
    ]
    if damage >= 8 and _seeded_float(f"{world.rng_seed}:bleed:{entity.entity_id}:{tick}") < 0.35:
        out.append(deposit_at(coord, MEDIA_TILE_GROUND, "bloodborne_germs", conc * 0.15, cause="blood_contaminated", tick=tick))
    return out


def deposit_cough_aerosol(
    world: "WorldState",
    entity: EntityState,
    *,
    substance: str = "cough_droplets",
) -> list[Transition]:
    catalog = world.config.substance_catalog
    sdef = catalog.get(substance) if catalog else None
    ttl = sdef.airborne_ttl_ticks if sdef else 12
    radius = sdef.spread_radius if sdef else 1
    pos = entity.position
    tick = world.tick
    out = [deposit_at(pos, MEDIA_TILE_AIR, substance, 40.0, cause="cough", tick=tick, remaining_ticks=ttl)]
    grid = world.spatial
    for nc in field_spread_neighbors(grid, pos, airborne=True):
        if nc == pos:
            continue
        if nc.manhattan(pos) <= radius:
            out.append(deposit_at(
                nc, MEDIA_TILE_AIR, substance, 14.0,
                cause="cough_drift", tick=tick,
                remaining_ticks=max(4, ttl - 2),
            ))
    return out


def describe_medium(
    env: dict[str, Any],
    medium: str,
    catalog: Optional[SubstanceCatalog],
    *,
    legacy_tile: Optional["Tile"] = None,
) -> list[str]:
    lines: list[str] = []
    layer = get_layer(env, medium, legacy_tile=legacy_tile)
    for sid, data in layer.items():
        amt = float(data.get("amount", 0))
        sdef = catalog.get(sid) if catalog else None
        name = sdef.display if sdef else sid
        if medium in AIR_MEDIA and amt >= 8:
            lines.append(f"{name} hangs in the air")
        elif amt >= 15:
            lines.append(f"stained with {name}")
    return lines


# ── Autonomic tick ────────────────────────────────────────────────────────


def _weather(world: "WorldState") -> str:
    grid = world.spatial
    return str(world.meta.get("weather", "") or grid.region_env.get("weather", "")).lower()


def _eval_emitters(world: "WorldState", catalog: SubstanceCatalog) -> list[Transition]:
    transitions: list[Transition] = []
    weather = _weather(world)
    tick = world.tick
    grid = world.spatial

    for emitter in catalog.emitters:
        when = emitter.when
        every = int(when.get("every_ticks", 1))
        if every > 1 and tick % every != 0:
            continue
        wlist = when.get("weather")
        if wlist and weather not in [str(w).lower() for w in wlist]:
            continue

        targets = emitter.targets
        medium = str(targets.get("medium", MEDIA_TILE_GROUND))
        tile_tags = {str(t).lower() for t in (targets.get("tile_tags_any") or [])}
        terrain_filter = targets.get("terrain")

        dep = emitter.deposit
        substance = str(dep.get("substance", "")).lower()
        amount = float(dep.get("amount", 10.0))
        if not substance:
            continue

        for c, tile in iter_tiles(grid):
            if tile_tags and not tile_tags.intersection({t.lower() for t in tile.tags}):
                continue
            if terrain_filter and tile.terrain.value != str(terrain_filter):
                continue
            transitions.append(deposit_at(c, medium, substance, amount, cause=emitter.id, tick=tick))
            wet = emitter.also_wet
            if wet:
                transitions.append(Transition(
                    kind=TransitionKind.FLUID_CHANGED,
                    payload=fluid_tile_payload(
                        c,
                        target_kind="tile",
                        material=str(wet.get("fluid_material", "water")),
                        volume_ml=float(wet.get("volume_ml", 30.0)),
                        cause=emitter.id,
                        wet=True,
                    ),
                ))
    return transitions


def _load_tile_reactions(world: "WorldState") -> list[dict]:
    """Read tile_reactions from reactions.yaml (cached in meta)."""
    cached = world.meta.get("_tile_reactions")
    if cached is not None:
        return cached
    from pathlib import Path
    import yaml

    candidates: list[Path] = []
    pack_path = world.meta.get("_pack_path")
    if pack_path:
        p = Path(pack_path)
        candidates.append(p / "reactions.yaml")
        candidates.append(p.parent / "default" / "reactions.yaml")
    candidates.append(Path("worlds/default/reactions.yaml"))

    # Merge tile_reactions from all candidate files (pack overrides default).
    merged: list[dict] = []
    seen_ids: set[str] = set()
    for path in candidates:
        if not path.is_file():
            continue
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for rxn in (doc.get("tile_reactions") or []):
            rid = rxn.get("id", "")
            if rid not in seen_ids:
                merged.append(rxn)
                seen_ids.add(rid)

    world.meta["_tile_reactions"] = merged
    return merged


def _eval_tile_reactions(
    world: "WorldState",
    catalog: SubstanceCatalog,
    tile_reactions: list[dict],
) -> list[Transition]:
    """Evaluate tile_reactions per tile and emit FIELD_CHANGED transitions."""
    if not tile_reactions:
        return []
    transitions: list[Transition] = []
    grid = world.spatial
    tick = world.tick
    weather = _weather(world)
    rng_seed = str(world.rng_seed)

    region = world.regions.get(world.active_region_id)
    region_temp = float((region.grid.region_env if region else {}).get("temperature", 20.0))
    atm_adds: dict[str, float] = {}  # substance -> amount to add to atmosphere

    for coord, tile in iter_tiles(grid):
        tile_tags = {t.lower() for t in tile.tags}
        ground_layer = get_layer(tile.env, MEDIA_TILE_GROUND, legacy_tile=tile)
        air_layer = get_layer(tile.env, MEDIA_TILE_AIR, legacy_tile=tile)

        for rxn in tile_reactions:
            prob = float(rxn.get("probability", 1.0))
            if prob < 1.0:
                seed = f"tile_rxn:{rng_seed}:{rxn.get('id','')}:{coord_key(coord)}:{tick}"
                if _seeded_float(seed) >= prob:
                    continue

            conditions = rxn.get("conditions") or []
            passed = True
            for cond in conditions:
                if "tile_has_tag" in cond:
                    if str(cond["tile_has_tag"]).lower() not in tile_tags:
                        passed = False
                        break
                if "tile_has_substance" in cond:
                    sub_map = cond["tile_has_substance"]
                    for med_key, sub_name in sub_map.items():
                        layer = ground_layer if med_key == "ground" else air_layer
                        if float((layer.get(sub_name) or {}).get("amount", 0)) < 5.0:
                            passed = False
                            break
                    if not passed:
                        break
                if "region_weather" in cond:
                    allowed = [str(w).lower() for w in (cond["region_weather"] or [])]
                    if weather not in allowed:
                        passed = False
                        break
                if "region_cold" in cond:
                    threshold = float(cond["region_cold"])
                    if region_temp > threshold:
                        passed = False
                        break
            if not passed:
                continue

            for effect in (rxn.get("effects") or []):
                if "tile_add_substance" in effect:
                    e = effect["tile_add_substance"]
                    transitions.append(deposit_at(
                        coord,
                        str(e.get("medium", MEDIA_TILE_GROUND)),
                        str(e["substance"]).lower(),
                        float(e.get("amount", 10.0)),
                        cause=str(rxn.get("id", "tile_rxn")),
                        tick=tick,
                    ))
                if "tile_clear_substance" in effect:
                    e = effect["tile_clear_substance"]
                    transitions.append(build_field_transition(
                        str(e.get("medium", MEDIA_TILE_GROUND)),
                        str(e["substance"]).lower(),
                        0.0,
                        coord=coord,
                        operation="clear",
                        cause=str(rxn.get("id", "tile_rxn")),
                        tick=tick,
                    ))
                if "tile_add_tag" in effect:
                    tag = str(effect["tile_add_tag"]).lower()
                    if tag not in tile_tags:
                        tile.tags = list(tile.tags) + [tag]
                        grid.set_tile(coord, tile)
                if "atmosphere_add_substance" in effect:
                    e = effect["atmosphere_add_substance"]
                    sub = str(e["substance"]).lower()
                    atm_adds[sub] = atm_adds.get(sub, 0.0) + float(e.get("amount", 5.0))

    # Push atmosphere accumulations to region_env fields
    if atm_adds and region:
        atm = region.grid.region_env.setdefault("fields", {}).setdefault("atmosphere", {})
        for sub, amt in atm_adds.items():
            existing = atm.get(sub) or {}
            existing["amount"] = min(100.0, float(existing.get("amount", 0)) + amt)
            existing["since_tick"] = tick
            atm[sub] = existing
        region.grid.region_env["fields"]["atmosphere"] = atm

    return transitions


def field_tick(world: "WorldState") -> list[Transition]:
    """
    Generic autonomic pass: emitters, tile reactions, decay, spread, exposure, illness.
    Replaces scenario-specific ticks over time.
    """
    catalog = world.config.substance_catalog
    if catalog is None or not catalog.substances:
        from .contamination import contamination_tick
        return contamination_tick(world)

    transitions: list[Transition] = []
    transitions.extend(_eval_emitters(world, catalog))

    tile_reactions = _load_tile_reactions(world)
    transitions.extend(_eval_tile_reactions(world, catalog, tile_reactions))

    grid = world.spatial
    tick = world.tick
    pending: dict[str, dict[str, dict[str, Any]]] = {}

    for coord, tile in iter_tiles(grid):
        for medium in (MEDIA_TILE_GROUND, MEDIA_TILE_AIR):
            layer = get_layer(tile.env, medium, legacy_tile=tile)
            old_ids = set(layer.keys())

            for sid, data in list(layer.items()):
                sdef = catalog.get(sid)
                decay = sdef.decay_per_tick if sdef else 2.0
                amt = float(data.get("amount", 0)) - decay
                meta = dict(data.get("meta") or {})

                if medium in AIR_MEDIA:
                    rem = int(meta.get("remaining_ticks", sdef.airborne_ttl_ticks if sdef else 10)) - 1
                    meta["remaining_ticks"] = rem
                    if rem <= 0 or amt <= 0.5:
                        layer.pop(sid, None)
                        continue
                    data = {"amount": amt, "meta": meta}
                    layer[sid] = data
                    if sdef and sdef.spread_radius and tick % 2 == 0:
                        leak = amt * 0.2
                        for c2 in field_spread_neighbors(grid, coord, airborne=True):
                            if c2 == coord or c2.manhattan(coord) > sdef.spread_radius:
                                continue
                            nk = coord_key(c2)
                            pl = pending.setdefault(nk, {})
                            al = dict(pl.get(f"air:{sid}") or {})
                            al["amount"] = min(100.0, float(al.get("amount", 0)) + leak)
                            al["remaining_ticks"] = max(int(al.get("remaining_ticks", 0)), rem - 1)
                            pl[f"air:{sid}"] = al
                else:
                    if amt <= 0.5:
                        layer.pop(sid, None)
                    else:
                        layer[sid] = {"amount": amt, "since_tick": data.get("since_tick"), "meta": meta}
                        if sdef and sdef.spread_neighbor > 0 and tick % 3 == 0:
                            leak = amt * sdef.spread_neighbor
                            for nc in field_spread_neighbors(grid, coord, airborne=False):
                                if nc == coord:
                                    continue
                                nk = coord_key(nc)
                                pl = pending.setdefault(nk, {})
                                gl = dict(pl.get(f"ground:{sid}") or {})
                                gl["amount"] = min(100.0, float(gl.get("amount", 0)) + leak)
                                pl[f"ground:{sid}"] = gl

            for removed in old_ids - set(layer.keys()):
                transitions.append(build_field_transition(
                    medium, removed, 0, coord=coord, operation="clear", tick=tick,
                ))
            for sid, data in layer.items():
                transitions.append(build_field_transition(
                    medium, sid, float(data["amount"]),
                    coord=coord, operation="set", tick=tick,
                    remaining_ticks=int((data.get("meta") or {}).get("remaining_ticks", 0)) or None,
                ))

    for nk, blends in pending.items():
        c = coord_from_tile_key(nk)
        for key, data in blends.items():
            kind, sid = key.split(":", 1)
            med = MEDIA_TILE_AIR if kind == "air" else MEDIA_TILE_GROUND
            transitions.append(deposit_at(
                c, med, sid, float(data["amount"]), cause="spread", tick=tick,
                remaining_ticks=int(data.get("remaining_ticks", 0)) or None,
            ))

    # ── Atmosphere decay ───────────────────────────────────────────────────
    region = world.regions.get(world.active_region_id)
    if region:
        atm = region.grid.region_env.get("fields", {}).get("atmosphere", {})
        for sub_id, data in list(atm.items()):
            sdef = catalog.get(sub_id)
            decay = sdef.decay_per_tick if sdef else 3.0
            new_amt = float(data.get("amount", 0)) - decay
            if new_amt <= 0.5:
                atm.pop(sub_id, None)
            else:
                data = dict(data)
                data["amount"] = new_amt
                atm[sub_id] = data
        if atm:
            region.grid.region_env.setdefault("fields", {})["atmosphere"] = atm
        else:
            region.grid.region_env.get("fields", {}).pop("atmosphere", None)

    from .contamination import exposure_and_illness_tick
    transitions.extend(exposure_and_illness_tick(world))

    transitions.extend(maybe_register_hazard_facts(world))
    return transitions


def region_atmosphere_amount(
    world: "WorldState",
    substance: str,
    *,
    region_id: Optional[str] = None,
) -> float:
    """Return concentration of *substance* in a region's atmosphere (0 if absent).

    ``region_id`` defaults to ``world.active_region_id``.
    """
    rid = region_id if region_id is not None else world.active_region_id
    region = world.regions.get(rid)
    if region is None:
        return 0.0
    atm = (region.grid.region_env.get("fields") or {}).get("atmosphere") or {}
    entry = atm.get(str(substance).lower())
    if not isinstance(entry, dict):
        return 0.0
    return float(entry.get("amount", 0))


def maybe_register_hazard_facts(world: "WorldState") -> list[Transition]:
    """
    Register durable world facts when atmosphere hazards cross thresholds.

    Facts feed ``world_fact_tag`` pressures, conditional scenario beats,
    belief propagation, and chronicle — without coupling field physics to
    narrative code directly.
    """
    from .schemas import WorldFact, WorldFactScope
    from .world_adjudicator import world_fact_transition

    smoke_amt = region_atmosphere_amount(world, "smoke")
    registered: set[str] = set(world.meta.get("_hazard_facts_registered") or [])
    transitions: list[Transition] = []
    tick = world.tick

    if smoke_amt >= 12.0 and "smoke_in_room" not in registered:
        transitions.append(
            world_fact_transition(
                WorldFact(
                    claim="Thick smoke fills the common room.",
                    scope=WorldFactScope.WORLD,
                    established_tick=tick,
                    established_by="system",
                    tags=["smoke", "fire", "hazard", "panic"],
                )
            )
        )
        registered.add("smoke_in_room")

    if smoke_amt >= 35.0 and "smoke_panic" not in registered:
        transitions.append(
            world_fact_transition(
                WorldFact(
                    claim="Patrons are choking and scrambling for the exits.",
                    scope=WorldFactScope.WORLD,
                    established_tick=tick,
                    established_by="system",
                    tags=["smoke", "panic", "hazard"],
                )
            )
        )
        registered.add("smoke_panic")

    if registered:
        world.meta["_hazard_facts_registered"] = sorted(registered)
    return transitions


