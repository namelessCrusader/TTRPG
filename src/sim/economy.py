"""
Regional market simulation — supply, demand, and drifting prices.

Market state lives in ``world.meta["market"]``:

    {
      "region_main": {
        "ale": {"base_price": 5, "price": 5.2, "supply": 80, "demand": 60},
        ...
      }
    }

Initialized from object ``attributes.value`` and tags on world load.
Ticked deterministically by ``world_clock``; trades update supply/demand
via ``record_trade``.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .schemas import EntityState, Transition, TransitionKind, WorldState

logger = logging.getLogger(__name__)

_META_KEY = "market"
_DEFAULT_SUPPLY = 100.0
_DEFAULT_DEMAND = 50.0
_PRICE_DRIFT = 0.08
_SUPPLY_REPLENISH = 2.0
_DEMAND_DECAY = 1.5
_MIN_PRICE_RATIO = 0.25
_MAX_PRICE_RATIO = 4.0
_ECONOMY_EVERY_TICKS = 5


def _market(world: WorldState) -> dict[str, dict[str, dict[str, Any]]]:
    raw = world.meta.setdefault(_META_KEY, {})
    if not isinstance(raw, dict):
        raw = {}
        world.meta[_META_KEY] = raw
    return raw


def _region_market(world: WorldState, region_id: Optional[str] = None) -> dict[str, dict]:
    rid = region_id or world.active_region_id
    market = _market(world)
    reg = market.setdefault(rid, {})
    if not isinstance(reg, dict):
        reg = {}
        market[rid] = reg
    return reg


def init_market(world: WorldState) -> None:
    """Seed market goods from placed objects and pack extra config."""
    if world.meta.get(_META_KEY):
        return

    goods: dict[str, dict[str, float]] = {}
    for region in world.regions.values():
        for obj in region.grid.objects.values():
            val = obj.attributes.get("value")
            if val is None:
                continue
            try:
                base = float(val)
            except (TypeError, ValueError):
                continue
            if base <= 0:
                continue
            for tag in obj.tags or [obj.name.lower().replace(" ", "_")]:
                if not tag:
                    continue
                key = str(tag).lower()
                if key not in goods:
                    goods[key] = {
                        "base_price": base,
                        "price": base,
                        "supply": _DEFAULT_SUPPLY,
                        "demand": _DEFAULT_DEMAND,
                    }

    pack_goods = (world.config.extra or {}).get("market_goods") or {}
    for tag, spec in pack_goods.items():
        if not isinstance(spec, dict):
            continue
        base = float(spec.get("base_price", spec.get("price", 10)))
        goods[str(tag).lower()] = {
            "base_price": base,
            "price": float(spec.get("price", base)),
            "supply": float(spec.get("supply", _DEFAULT_SUPPLY)),
            "demand": float(spec.get("demand", _DEFAULT_DEMAND)),
        }

    if goods:
        _market(world)[world.active_region_id] = goods
        logger.debug("Initialized market with %d goods", len(goods))


def get_market_price(
    world: WorldState,
    good_tag: str,
    *,
    region_id: Optional[str] = None,
) -> float:
    """Current price for a good tag; falls back to base or 10 gold."""
    init_market(world)
    tag = good_tag.lower().strip()
    reg = _region_market(world, region_id)
    entry = reg.get(tag)
    if entry:
        return round(float(entry.get("price", entry.get("base_price", 10))), 1)

    for key, data in reg.items():
        if tag in key or key in tag:
            return round(float(data.get("price", data.get("base_price", 10))), 1)

    return 10.0


def record_trade(
    world: WorldState,
    good_tag: str,
    *,
    quantity: float = 1.0,
    region_id: Optional[str] = None,
) -> None:
    """Apply supply/demand shock after a trade."""
    init_market(world)
    tag = good_tag.lower().strip()
    reg = _region_market(world, region_id)
    entry = reg.setdefault(
        tag,
        {
            "base_price": 10.0,
            "price": 10.0,
            "supply": _DEFAULT_SUPPLY,
            "demand": _DEFAULT_DEMAND,
        },
    )
    entry["supply"] = max(0.0, float(entry.get("supply", _DEFAULT_SUPPLY)) - quantity)
    entry["demand"] = min(
        200.0,
        float(entry.get("demand", _DEFAULT_DEMAND)) + quantity * 3.0,
    )
    _reprice_entry(entry)


def _reprice_entry(entry: dict[str, Any]) -> None:
    base = float(entry.get("base_price", 10))
    supply = max(1.0, float(entry.get("supply", _DEFAULT_SUPPLY)))
    demand = max(1.0, float(entry.get("demand", _DEFAULT_DEMAND)))
    pressure = demand / supply
    new_price = base * (0.75 + 0.5 * pressure)
    ratio = new_price / base if base > 0 else 1.0
    ratio = max(_MIN_PRICE_RATIO, min(_MAX_PRICE_RATIO, ratio))
    entry["price"] = round(base * ratio, 2)


def record_trade_from_transitions(
    world: WorldState,
    transitions: list[Transition],
    *,
    region_id: Optional[str] = None,
) -> None:
    """
    Update market supply/demand from completed trades in a tick's transitions.
    """
    rid = region_id or world.active_region_id
    for t in transitions:
        if t.kind != TransitionKind.ENTITY_STAT_CHANGED:
            continue
        p = t.payload
        if p.get("stat") != "gold" or p.get("cause") not in (
            "trade_payment", "completed_trade", "adjudicated_trade", "adjudicated_bribe",
        ):
            continue
        delta = float(p.get("delta", 0))
        if delta <= 0:
            continue
        # Infer good from nearby item transfers in same batch
        for t2 in transitions:
            if t2.kind == TransitionKind.ITEM_TRANSFERRED:
                oid = t2.payload.get("object_id")
                if oid:
                    obj = world.spatial.objects.get(oid)  # type: ignore[arg-type]
                    if obj:
                        for tag in (obj.tags or [obj.name.lower().replace(" ", "_")]):
                            record_trade(world, str(tag), quantity=1.0, region_id=rid)
                            return
        record_trade(world, "goods", quantity=abs(delta) / 10.0, region_id=rid)


def _maybe_market_fact(
    world: WorldState,
    region_id: str,
    tag: str,
    old_price: float,
    new_price: float,
) -> Optional[Transition]:
    """Register a world fact when a price moves sharply."""
    if old_price <= 0:
        return None
    swing = abs(new_price - old_price) / old_price
    if swing < 0.12:
        return None
    from .schemas import WorldFact, WorldFactScope

    direction = "surged" if new_price > old_price else "dropped"
    claim = (
        f"Market prices for {tag} have {direction} "
        f"({old_price:.0f}g → {new_price:.0f}g)."
    )
    fact = WorldFact(
        claim=claim,
        scope=WorldFactScope.REGION,
        subject_id=region_id,
        established_tick=world.tick,
        tags=["trade", "economy"],
        confidence=0.9,
    )
    from .world_adjudicator import world_fact_transition

    return world_fact_transition(fact)


def economy_tick(world: WorldState) -> list[Transition]:
    """
    Drift regional prices and replenish supply.

    Returns replay-safe transitions (market updates + optional world facts).
    Does not mutate ``world.meta["market"]`` directly — apply via compiler.
    """
    if world.tick % _ECONOMY_EVERY_TICKS != 0:
        return []

    init_market(world)
    transitions: list[Transition] = []

    for region_id, reg in _market(world).items():
        if not isinstance(reg, dict):
            continue
        for tag, entry in reg.items():
            if not isinstance(entry, dict):
                continue
            new_entry = dict(entry)
            new_entry["supply"] = min(
                200.0,
                float(new_entry.get("supply", _DEFAULT_SUPPLY)) + _SUPPLY_REPLENISH,
            )
            new_entry["demand"] = max(
                5.0,
                float(new_entry.get("demand", _DEFAULT_DEMAND)) - _DEMAND_DECAY,
            )
            old_price = float(new_entry.get("price", new_entry.get("base_price", 10)))
            _reprice_entry(new_entry)
            new_price = float(new_entry.get("price", old_price))

            transitions.append(
                Transition(
                    kind=TransitionKind.MARKET_UPDATED,
                    payload={
                        "region_id": region_id,
                        "tag": tag,
                        "entry": new_entry,
                    },
                )
            )

            fact_tr = _maybe_market_fact(world, region_id, tag, old_price, new_price)
            if fact_tr is not None:
                transitions.append(fact_tr)
            if abs(new_price - old_price) / max(old_price, 1) > _PRICE_DRIFT:
                logger.debug(
                    "Market %s/%s: %.1f → %.1f (s=%.0f d=%.0f)",
                    region_id,
                    tag,
                    old_price,
                    new_price,
                    new_entry["supply"],
                    new_entry["demand"],
                )

    return transitions


def format_market_for_projection(
    world: WorldState,
    focal: EntityState,
    *,
    max_items: int = 4,
) -> list[str]:
    """Short price lines for LM projection."""
    init_market(world)
    reg = _region_market(world, focal.region_id)
    if not reg:
        return []

    lines: list[str] = []
    for tag, entry in sorted(reg.items(), key=lambda x: -float(x[1].get("demand", 0))):
        if len(lines) >= max_items:
            break
        price = float(entry.get("price", entry.get("base_price", 0)))
        supply = float(entry.get("supply", 0))
        if supply < 15:
            lines.append(f"{tag}: {price:.0f}g (scarce)")
        else:
            lines.append(f"{tag}: {price:.0f}g")
    return lines


def infer_gold_from_market(
    world: WorldState,
    item_names: list[str],
    *,
    region_id: Optional[str] = None,
) -> float:
    """Sum market prices for named items (trade compiler helper)."""
    total = 0.0
    for name in item_names:
        name = name.strip()
        if not name:
            continue
        tag = name.lower().replace(" ", "_")
        total += get_market_price(world, tag, region_id=region_id)
    return total


__all__ = [
    "economy_tick",
    "format_market_for_projection",
    "get_market_price",
    "infer_gold_from_market",
    "init_market",
    "record_trade",
    "record_trade_from_transitions",
]
