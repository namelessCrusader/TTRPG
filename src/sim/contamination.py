"""
Contamination, biofluid stains, airborne pathogens, and illness.

Canonical state:
  tile.env["contamination"]  — {agent_id: {concentration, since_tick, cause}}
  tile.env["airborne"]     — {agent_id: {concentration, remaining_ticks, cause}}
  entity.meta["contamination"] — surface contamination on body/clothes
  entity.meta["infection"]     — {agent, dose, incubation_left, stage, ticks_sick}

All mutations go through CONTAMINATION_CHANGED transitions (or companion
FLUID_CHANGED for visible wet spills). See WORLD_CONSEQUENCES.md.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

from .field_coords import (
    coord_from_tile_key,
    field_spread_neighbors,
    fluid_tile_payload,
    iter_tiles,
)
from .schemas import (
    coord_key,
    Coord,
    EntityId,
    EntityState,
    Transition,
    TransitionKind,
)

if TYPE_CHECKING:
    from .schemas import SpatialGrid, Tile, WorldState


@dataclass
class AgentDef:
    name: str
    display: str = ""
    category: str = "biofluid"  # biofluid | pathogen | material | weather
    fluid_material: str = ""
    stain: bool = True
    decays_per_tick: float = 2.0
    spread_neighbor: float = 0.0
    pathogen: bool = False
    routes: list[str] = field(default_factory=list)  # touch, ingest, inhale
    infection_per_100: float = 0.0
    incubation_ticks: int = 0
    illness_condition: str = ""
    illness_damage_per_tick: int = 0
    illness_duration_ticks: int = 30
    airborne: bool = False
    airborne_duration_ticks: int = 10
    spread_radius: int = 1
    carries_pathogen: str = ""
    host_agents: list[str] = field(default_factory=list)


@dataclass
class AgentCatalog:
    agents: dict[str, AgentDef] = field(default_factory=dict)

    def get(self, name: str) -> Optional[AgentDef]:
        return self.agents.get((name or "").lower().strip())


def load_agent_catalog(pack_path: Any) -> AgentCatalog:
    from pathlib import Path
    import yaml

    catalog = AgentCatalog()
    for base in (Path(pack_path), Path(pack_path).parent / "default", Path("worlds/default")):
        path = base / "agents.yaml"
        if not path.is_file():
            continue
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for name, raw in (doc.get("agents") or {}).items():
            if not isinstance(raw, dict):
                continue
            key = str(name).lower()
            catalog.agents[key] = AgentDef(
                name=key,
                display=str(raw.get("display", key)),
                category=str(raw.get("category", "biofluid")),
                fluid_material=str(raw.get("fluid_material", raw.get("carries_pathogen", ""))).lower(),
                stain=bool(raw.get("stain", True)),
                decays_per_tick=float(raw.get("decays_per_tick", 2.0)),
                spread_neighbor=float(raw.get("spread_neighbor", 0.0)),
                pathogen=bool(raw.get("pathogen", False)),
                routes=[str(r).lower() for r in (raw.get("routes") or [])],
                infection_per_100=float(raw.get("infection_per_100", 0.0)),
                incubation_ticks=int(raw.get("incubation_ticks", 0)),
                illness_condition=str(raw.get("illness_condition", "")),
                illness_damage_per_tick=int(raw.get("illness_damage_per_tick", 0)),
                illness_duration_ticks=int(raw.get("illness_duration_ticks", 30)),
                airborne=bool(raw.get("airborne", False)),
                airborne_duration_ticks=int(raw.get("airborne_duration_ticks", 10)),
                spread_radius=int(raw.get("spread_radius", 1)),
                carries_pathogen=str(raw.get("carries_pathogen", "")).lower(),
                host_agents=[str(h).lower() for h in (raw.get("host_agents") or [])],
            )
        if catalog.agents:
            break
    return catalog


def _seeded_float(seed: str) -> float:
    digest = hashlib.sha256(seed.encode()).digest()
    return int.from_bytes(digest[:4], "big") / (2**32)


def _contamination_tile_payload(
    coord: Coord,
    target_kind: str,
    agent: str,
    operation: str,
    concentration: float,
    *,
    tick: int,
    cause: str = "",
    remaining_ticks: Optional[int] = None,
) -> Transition:
    payload: dict[str, Any] = {
        "target_kind": target_kind,
        "x": coord.x,
        "y": coord.y,
        "z": coord.z,
        "agent": agent,
        "operation": operation,
        "concentration": concentration,
        "tick": tick,
    }
    if cause:
        payload["cause"] = cause
    if remaining_ticks is not None:
        payload["remaining_ticks"] = remaining_ticks
    return Transition(kind=TransitionKind.CONTAMINATION_CHANGED, payload=payload)


def _tile_contamination(tile: "Tile") -> dict[str, dict[str, Any]]:
    raw = tile.env.get("contamination")
    if isinstance(raw, dict):
        return raw
    return {}


def _tile_airborne(tile: "Tile") -> dict[str, dict[str, Any]]:
    raw = tile.env.get("airborne")
    if isinstance(raw, dict):
        return raw
    return {}


def _entity_contamination(entity: EntityState) -> dict[str, dict[str, Any]]:
    raw = entity.meta.get("contamination")
    if isinstance(raw, dict):
        return raw
    return {}


def apply_contamination_transition(world: "WorldState", t: Transition) -> None:
    """Apply one CONTAMINATION_CHANGED payload (delegates to fields layer)."""
    from .fields import apply_field_transition

    p = dict(t.payload)
    if "medium" not in p:
        tk = str(p.get("target_kind", "tile"))
        p["medium"] = "tile.air" if tk == "air" else "tile.ground"
        p["substance"] = p.get("agent", "")
        p["amount"] = p.get("concentration", 0.0)
    apply_field_transition(world, Transition(kind=t.kind, payload=p))


def build_deposit_tile(
    coord: Coord,
    agent: str,
    concentration: float,
    *,
    cause: str,
    tick: int,
) -> Transition:
    return _contamination_tile_payload(
        coord, "tile", agent, "deposit", concentration, tick=tick, cause=cause,
    )


def build_deposit_air(
    coord: Coord,
    agent: str,
    concentration: float,
    *,
    remaining_ticks: int,
    cause: str,
    tick: int,
) -> Transition:
    return _contamination_tile_payload(
        coord, "air", agent, "deposit", concentration,
        tick=tick, cause=cause, remaining_ticks=remaining_ticks,
    )


def bleed_at_entity(
    world: "WorldState",
    entity: EntityState,
    *,
    damage: int,
    cause: str = "wound",
) -> list[Transition]:
    from .fields import deposit_bleed
    return deposit_bleed(world, entity, damage=damage, cause=cause)


def biofluid_deposit_from_relief(
    world: "WorldState",
    entity: EntityState,
    *,
    privy: bool,
) -> list[Transition]:
    from .fields import deposit_biofluid_relief
    return deposit_biofluid_relief(world, entity, privy=privy)


def cough_aerosol_transitions(
    world: "WorldState",
    entity: EntityState,
    *,
    pathogen: str = "cough_droplets",
) -> list[Transition]:
    from .fields import deposit_cough_aerosol
    return deposit_cough_aerosol(world, entity, substance=pathogen)


def _infection_roll(
    world: "WorldState",
    entity: EntityState,
    pathogen: str,
    dose: float,
) -> bool:
    catalog = world.config.agent_catalog
    adef = catalog.get(pathogen) if catalog else None
    if adef is None or not adef.pathogen:
        return False
    if "sick" in entity.conditions or entity.meta.get("infection"):
        return False
    base = adef.infection_per_100 * (dose / 100.0)
    resist = float(entity.attributes.get("constitution", 50)) / 100.0
    chance = max(0.02, min(0.85, base * (1.1 - resist * 0.4)))
    seed = f"infect:{world.rng_seed}:{entity.entity_id}:{pathogen}:{world.tick}"
    return _seeded_float(seed) < chance


def _infection_transition(
    entity_id: str,
    infection: Optional[dict[str, Any]],
) -> Transition:
    return Transition(
        kind=TransitionKind.ENTITY_PROPERTY_CHANGED,
        payload={
            "entity_id": entity_id,
            "prop": "infection",
            "new_value": infection,
            "cause": "contamination",
        },
    )


def exposure_and_illness_tick(world: "WorldState") -> list[Transition]:
    """Entity exposure to tile fields and infection progression."""
    transitions: list[Transition] = []
    grid = world.spatial
    catalog = world.config.agent_catalog
    if catalog is None:
        return transitions
    tick = world.tick

    for entity in grid.entities.values():
        if not entity.alive:
            continue
        eid = str(entity.entity_id)
        coord = entity.position
        tile = grid.tile_at(coord)
        from .fields import get_layer, MEDIA_TILE_AIR, MEDIA_TILE_GROUND

        cont_raw = get_layer(tile.env, MEDIA_TILE_GROUND, legacy_tile=tile)
        air_raw = get_layer(tile.env, MEDIA_TILE_AIR, legacy_tile=tile)
        cont = {k: {"concentration": v.get("amount", 0)} for k, v in cont_raw.items()}
        air = {k: {"concentration": v.get("amount", 0)} for k, v in air_raw.items()}

        total_dose: dict[str, float] = {}

        def _carries_pathogen(adef: "AgentDef") -> Optional[str]:
            """If this biofluid/substance carries a named pathogen, return its ID."""
            carries = getattr(adef, "carries", None)
            if carries:
                return str(carries)
            return None

        for agent_id, layer in cont.items():
            adef = catalog.get(agent_id)
            if adef is None:
                continue
            conc = float(layer.get("concentration", 0.0))
            is_pathogen = getattr(adef, "pathogen", False) or getattr(adef, "category", "") == "pathogen"
            if is_pathogen:
                for route in adef.routes:
                    if route == "touch":
                        total_dose[agent_id] = total_dose.get(agent_id, 0.0) + conc * 0.5
                    if route == "ingest" and "wet" in entity.tags:
                        total_dose[agent_id] = total_dose.get(agent_id, 0.0) + conc * 0.3
            else:
                path = _carries_pathogen(adef)
                if path:
                    total_dose[path] = total_dose.get(path, 0.0) + conc * 0.25

        for agent_id, layer in air.items():
            adef = catalog.get(agent_id)
            is_pathogen = adef and (getattr(adef, "pathogen", False) or getattr(adef, "category", "") == "pathogen")
            if adef is None or not is_pathogen:
                continue
            if "inhale" in adef.routes:
                total_dose[agent_id] = total_dose.get(
                    agent_id, 0.0
                ) + float(layer.get("concentration", 0.0)) * 0.6

        inf = entity.meta.get("infection")
        if isinstance(inf, dict) and inf.get("stage") == "incubating":
            left = int(inf.get("incubation_left", 0)) - 1
            if left <= 0:
                agent = str(inf.get("agent", ""))
                adef = catalog.get(agent)
                cond = adef.illness_condition if adef else "ill"
                transitions.append(Transition(
                    kind=TransitionKind.ENTITY_CONDITION_CHANGED,
                    payload={
                        "entity_id": eid,
                        "condition": cond,
                        "ticks": adef.illness_duration_ticks if adef else 30,
                    },
                ))
                transitions.append(_infection_transition(eid, {
                    "agent": agent,
                    "stage": "sick",
                    "ticks_sick": 0,
                }))
            else:
                inf = dict(inf)
                inf["incubation_left"] = left
                transitions.append(_infection_transition(eid, inf))
        elif isinstance(inf, dict) and inf.get("stage") == "sick":
            agent = str(inf.get("agent", ""))
            adef = catalog.get(agent)
            ts = int(inf.get("ticks_sick", 0)) + 1
            dmg = adef.illness_damage_per_tick if adef else 0
            dur = adef.illness_duration_ticks if adef else 30
            if dmg > 0:
                transitions.append(Transition(
                    kind=TransitionKind.ENTITY_HEALTH_CHANGED,
                    payload={
                        "entity_id": eid,
                        "delta": -dmg,
                        "cause": agent or "illness",
                    },
                ))
            if ts >= dur:
                cond = adef.illness_condition if adef else "ill"
                transitions.append(Transition(
                    kind=TransitionKind.ENTITY_CONDITION_CHANGED,
                    payload={
                        "entity_id": eid,
                        "condition": cond,
                        "ticks": 0,
                    },
                ))
                transitions.append(_infection_transition(eid, None))
            else:
                inf = dict(inf)
                inf["ticks_sick"] = ts
                transitions.append(_infection_transition(eid, inf))

        for pathogen, dose in total_dose.items():
            if dose < 5.0:
                continue
            if entity.meta.get("infection"):
                continue
            if _infection_roll(world, entity, pathogen, dose):
                adef = catalog.get(pathogen)
                transitions.append(_infection_transition(eid, {
                    "agent": pathogen,
                    "dose": dose,
                    "stage": "incubating",
                    "incubation_left": adef.incubation_ticks if adef else 15,
                }))
                transitions.append(Transition(
                    kind=TransitionKind.ENTITY_PROPERTY_CHANGED,
                    payload={
                        "entity_id": eid,
                        "prop": "exposure",
                        "new_value": pathogen,
                        "cause": "contamination",
                    },
                ))
    return transitions


def contamination_tick(world: "WorldState") -> list[Transition]:
    """
    Per-tick autonomic pass: decay stains, spread biofluids, drift aerosols,
    expose entities, progress illness. Returns transitions to apply.

    When ``substance_catalog`` is loaded, tile dynamics run in ``field_tick``;
    this function only handles exposure/illness for backward compatibility.
    """
    sc = getattr(world.config, "substance_catalog", None)
    if sc is not None and sc.substances:
        from .fields import field_tick
        return field_tick(world)

    transitions: list[Transition] = []
    grid = world.spatial
    catalog = world.config.agent_catalog
    if catalog is None:
        return transitions

    tick = world.tick
    weather = str(world.meta.get("weather", "") or grid.region_env.get("weather", "")).lower()
    raining = weather in ("rain", "storm", "drizzle")

    # ── Weather: rain wets mud / outdoor tiles ───────────────────────────
    if raining and tick % 2 == 0:
        for c, tile in iter_tiles(grid):
            if "mud" in {t.lower() for t in tile.tags} or tile.terrain.value in ("water",):
                transitions.append(build_deposit_tile(
                    c, "rainwater", 12.0, cause="rain", tick=tick,
                ))
                transitions.append(Transition(
                    kind=TransitionKind.FLUID_CHANGED,
                    payload=fluid_tile_payload(
                        c,
                        target_kind="tile",
                        material="water",
                        volume_ml=40.0,
                        cause="rain",
                        wet=True,
                    ),
                ))

    # Collect tile updates (decay + spread) in memory first
    pending_tile_sets: dict[str, dict[str, dict[str, Any]]] = {}
    pending_air_sets: dict[str, dict[str, dict[str, Any]]] = {}

    for coord, tile in iter_tiles(grid):
        old_cont = dict(_tile_contamination(tile))
        old_air = dict(_tile_airborne(tile))
        cont = dict(old_cont)
        air = dict(old_air)

        for agent_id, layer in list(cont.items()):
            adef = catalog.get(agent_id)
            decay = adef.decays_per_tick if adef else 2.0
            new_c = float(layer.get("concentration", 0.0)) - decay
            if new_c <= 0.5:
                cont.pop(agent_id, None)
            else:
                layer = dict(layer)
                layer["concentration"] = new_c
                cont[agent_id] = layer
                if adef and adef.spread_neighbor > 0 and tick % 3 == 0:
                    leak = new_c * adef.spread_neighbor
                    for nc in field_spread_neighbors(grid, coord, airborne=False):
                        if nc == coord:
                            continue
                        nk = coord_key(nc)
                        ncont = pending_tile_sets.setdefault(nk, {})
                        nl = dict(ncont.get(agent_id) or {})
                        nl["concentration"] = min(
                            100.0, float(nl.get("concentration", 0.0)) + leak
                        )
                        nl["since_tick"] = tick
                        nl["cause"] = "spread"
                        ncont[agent_id] = nl
                        pending_tile_sets[nk] = ncont

        for agent_id, layer in list(air.items()):
            adef = catalog.get(agent_id)
            if adef is None or not adef.airborne:
                continue
            rem = int(layer.get("remaining_ticks", 0)) - 1
            decay = adef.decays_per_tick
            new_c = float(layer.get("concentration", 0.0)) - decay
            if rem <= 0 or new_c <= 0.5:
                air.pop(agent_id, None)
            else:
                layer = dict(layer)
                layer["remaining_ticks"] = rem
                layer["concentration"] = new_c
                air[agent_id] = layer
                if tick % 2 == 0 and adef.spread_radius > 0:
                    for c2 in field_spread_neighbors(grid, coord, airborne=True):
                        if c2 == coord or c2.manhattan(coord) > adef.spread_radius:
                            continue
                        nk = coord_key(c2)
                        na = pending_air_sets.setdefault(nk, {})
                        al = dict(na.get(agent_id) or {})
                        al["concentration"] = min(
                            100.0,
                            float(al.get("concentration", 0.0)) + new_c * 0.2,
                        )
                        al["remaining_ticks"] = max(
                            int(al.get("remaining_ticks", 0)), rem - 1
                        )
                        al["cause"] = "airborne_drift"
                        na[agent_id] = al
                        pending_air_sets[nk] = na

        for agent_id in old_cont:
            if agent_id not in cont:
                transitions.append(_contamination_tile_payload(
                    coord, "tile", agent_id, "clear", 0.0, tick=tick,
                ))
        for agent_id, layer in cont.items():
            transitions.append(_contamination_tile_payload(
                coord, "tile", agent_id, "set",
                float(layer["concentration"]),
                tick=tick, cause=layer.get("cause", "decay"),
            ))
        for agent_id in old_air:
            if agent_id not in air:
                transitions.append(_contamination_tile_payload(
                    coord, "air", agent_id, "clear", 0.0, tick=tick,
                ))
        for agent_id, layer in air.items():
            transitions.append(_contamination_tile_payload(
                coord, "air", agent_id, "set",
                float(layer["concentration"]),
                tick=tick,
                cause=layer.get("cause", "decay"),
                remaining_ticks=int(layer.get("remaining_ticks", 1)),
            ))

    for nk, agents in pending_tile_sets.items():
        c = coord_from_tile_key(nk)
        for aid, layer in agents.items():
            transitions.append(_contamination_tile_payload(
                c, "tile", aid, "spread",
                float(layer["concentration"]),
                tick=tick, cause="spread",
            ))

    for nk, agents in pending_air_sets.items():
        c = coord_from_tile_key(nk)
        for aid, layer in agents.items():
            transitions.append(_contamination_tile_payload(
                c, "air", aid, "spread",
                float(layer["concentration"]),
                tick=tick,
                cause=layer.get("cause", "airborne_drift"),
                remaining_ticks=int(layer.get("remaining_ticks", 5)),
            ))

    transitions.extend(exposure_and_illness_tick(world))
    return transitions


def describe_tile_contamination(tile: "Tile", catalog: Optional[AgentCatalog]) -> list[str]:
    lines: list[str] = []
    if catalog is None:
        catalog = AgentCatalog()
    for agent_id, layer in _tile_contamination(tile).items():
        adef = catalog.get(agent_id)
        name = adef.display if adef else agent_id
        conc = float(layer.get("concentration", 0.0))
        if conc >= 15:
            lines.append(f"stained with {name}")
    for agent_id, layer in _tile_airborne(tile).items():
        adef = catalog.get(agent_id)
        name = adef.display if adef else agent_id
        if float(layer.get("concentration", 0.0)) >= 8:
            lines.append(f"{name} hangs in the air")
    return lines


def describe_entity_contamination(entity: EntityState, catalog: Optional[AgentCatalog]) -> str:
    parts: list[str] = []
    if catalog is None:
        catalog = AgentCatalog()
    inf = entity.meta.get("infection")
    if isinstance(inf, dict):
        agent = str(inf.get("agent", ""))
        adef = catalog.get(agent)
        name = adef.display if adef else agent
        if inf.get("stage") == "incubating":
            parts.append(f"feeling unwell ({name} exposure)")
        elif inf.get("stage") == "sick":
            parts.append(f"sick with {name}")
    for agent_id, layer in _entity_contamination(entity).items():
        adef = catalog.get(agent_id)
        name = adef.display if adef else agent_id
        if float(layer.get("concentration", 0.0)) >= 10:
            parts.append(f"soiled with {name}")
    return "; ".join(parts)
