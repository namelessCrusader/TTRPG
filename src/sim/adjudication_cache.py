"""
Session-stable adjudication cache.

When the LM adjudicates a creative action, store the validated result keyed by
(verb, target material tags, grounding zone, intent shape).  Replays of the
same action class within a campaign return the same mechanical outcome without
a second LM call — bridging the gap until a human promotes the rule to YAML.

Cache lives in ``world.meta["adjudication_cache"]`` so it survives save/load
within a campaign.  Opt out via ``world.config.extra["adjudication_cache"] = false``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .schemas import (
        AdjudicationResult,
        GroundingResult,
        SemanticAction,
        ValidationResult,
        WorldState,
    )

logger = logging.getLogger(__name__)

_CACHE_META_KEY = "adjudication_cache"
_ENABLE_KEY = "adjudication_cache"
_MAX_ENTRIES = 256


def is_enabled(world: "WorldState") -> bool:
    try:
        val = world.config.extra.get(_ENABLE_KEY, True)
        return bool(val)
    except Exception:
        return True


def _subject_tags(world: "WorldState", action: "SemanticAction") -> list[str]:
    from .schemas import EntityId, ObjectId

    tags: set[str] = set()
    target = action.target
    if isinstance(target, str):
        tid = EntityId(str(target))
        ent = world.spatial.entities.get(tid)
        if ent is not None:
            tags.update(str(t) for t in (ent.tags or []))
            mat = ent.meta.get("material")
            if mat:
                tags.add(f"mat:{mat}")
            return sorted(tags)
        oid = ObjectId(str(target))
        obj = world.spatial.objects.get(oid)
        if obj is not None:
            tags.update(str(t) for t in (obj.tags or []))
            mat = obj.meta.get("material")
            if mat:
                tags.add(f"mat:{mat}")
    return sorted(tags)


def _intent_shape(intent_norm: str) -> str:
    """Collapse object/actor names; keep structural prepositions."""
    prepositions = frozenset({
        "under", "over", "into", "onto", "inside", "behind", "through",
        "from", "with", "without", "between", "against", "near", "beside",
    })
    words = re.findall(r"[a-z']+", intent_norm)
    return " ".join(w if w in prepositions else "*" for w in words)


def cache_key(
    world: "WorldState",
    action: "SemanticAction",
    intent: str,
    grounding: "GroundingResult",
    result: "ValidationResult",
) -> str:
    """Stable hash for same-situation adjudication reuse."""
    verb = str(action.verb).lower().strip()
    intent_norm = re.sub(
        r"\s+", " ", (intent or action.raw_input or "").lower().strip(),
    )[:160]
    intent_shape = _intent_shape(intent_norm)
    rejection = ""
    if result.rejection_reason is not None:
        rejection = str(result.rejection_reason.value)
    payload = {
        "verb": verb,
        "zone": grounding.zone.value,
        "rejection": rejection,
        "intent_shape": intent_shape,
        "target_tags": _subject_tags(world, action),
        "region": world.active_region_id,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8"),
    ).hexdigest()[:20]
    return f"{verb}:{digest}"


def get_cached(
    world: "WorldState",
    key: str,
) -> Optional["AdjudicationResult"]:
    if not is_enabled(world):
        return None
    raw = (world.meta.get(_CACHE_META_KEY) or {}).get(key)
    if not isinstance(raw, dict):
        return None
    try:
        from .schemas import AdjudicationResult

        adj = AdjudicationResult.model_validate(raw)
        logger.debug("adjudication_cache hit key=%s", key)
        return adj
    except Exception as exc:
        logger.debug("adjudication_cache: invalid entry %s (%s)", key, exc)
        return None


def put_cached(
    world: "WorldState",
    key: str,
    adj: "AdjudicationResult",
) -> None:
    if not is_enabled(world):
        return
    if not (
        adj.ruling_text
        or adj.facts
        or adj.transition_proposals
        or adj.scheduled_effects
        or adj.synthesized_verb
    ):
        return
    cache: dict[str, Any] = world.meta.setdefault(_CACHE_META_KEY, {})
    cache[key] = adj.model_dump(mode="json")
    if len(cache) > _MAX_ENTRIES:
        # Drop oldest half (insertion order preserved in Py3.7+).
        drop = list(cache.keys())[: len(cache) - _MAX_ENTRIES // 2]
        for k in drop:
            cache.pop(k, None)
    logger.debug("adjudication_cache store key=%s (size=%d)", key, len(cache))


def clear_cache(world: "WorldState") -> None:
    world.meta.pop(_CACHE_META_KEY, None)


__all__ = [
    "cache_key",
    "clear_cache",
    "get_cached",
    "is_enabled",
    "put_cached",
]
