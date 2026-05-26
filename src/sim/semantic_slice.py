"""
Build SemanticSlice buffers and slices from canonical event_log.

Deterministic — no LM calls.  See ENGINE_DESIGN.md §24.
"""

from __future__ import annotations

from typing import Any, Optional

from .relational import get_entity_faction_ids
from .schemas import (
    CanonicalEventLine,
    EntityId,
    SemanticDynamicsConfig,
    SemanticLensKind,
    SemanticPassTrigger,
    SemanticSlice,
    TransitionKind,
    WorldState,
)

SEMANTIC_META_KEY = "semantic"
_VIOLENCE_VERBS = frozenset({
    "attack", "accuse", "threaten", "intimidate", "fight", "strike", "kill",
})
_CRISIS_TRANSITIONS = frozenset({
    TransitionKind.ENTITY_DIED.value,
    TransitionKind.ENTITY_HEALTH_CHANGED.value,
})


def ensure_semantic_state(world: WorldState) -> dict[str, Any]:
    return world.meta.setdefault(
        SEMANTIC_META_KEY,
        {
            "buffers": {},
            "last_pass_tick": -1,
            "pressure": {},
            "applied_log": [],
        },
    )


def _entity_name(world: WorldState, eid: str) -> str:
    ent = world.spatial.entities.get(EntityId(eid))
    return ent.name if ent else eid


def _verb_str(verb: Any) -> str:
    if isinstance(verb, str):
        return verb.lower().split(".")[-1]
    return str(verb).lower().split(".")[-1]


def canonical_line_from_event(world: WorldState, event) -> CanonicalEventLine:
    actor_id = str(event.action.actor)
    verb = _verb_str(event.action.verb)
    target_id: Optional[str] = None
    target_name: Optional[str] = None
    raw_target = event.action.target
    if isinstance(raw_target, str):
        target_id = raw_target
        target_name = _entity_name(world, target_id)
    kinds = [
        t.kind.value if hasattr(t.kind, "value") else str(t.kind)
        for t in event.transitions
    ]
    parts = [f"{_entity_name(world, actor_id)} {verb}"]
    if target_name:
        parts.append(f"→ {target_name}")
    if kinds:
        parts.append(f"({', '.join(kinds[:3])})")
    return CanonicalEventLine(
        event_id=str(event.event_id),
        tick=event.tick,
        actor_id=actor_id,
        actor_name=_entity_name(world, actor_id),
        verb=verb,
        target_id=target_id,
        target_name=target_name,
        transition_kinds=kinds,
        summary=" ".join(parts),
    )


def _lens_keys_for_event(world: WorldState, event) -> set[str]:
    keys: set[str] = set()
    actor_id = str(event.action.actor)
    if actor_id != "system":
        keys.add(f"agent:{actor_id}")

    raw_target = event.action.target
    if isinstance(raw_target, str) and raw_target in world.spatial.entities:
        keys.add(f"agent:{raw_target}")

    for wid in event.witnesses:
        keys.add(f"agent:{wid}")

    involved = {actor_id}
    if isinstance(raw_target, str):
        involved.add(raw_target)
    for wid in event.witnesses:
        involved.add(str(wid))

    for eid in involved:
        if eid == "system":
            continue
        for fid in get_entity_faction_ids(world.relational, eid):
            keys.add(f"faction:{fid}")

    return keys


def record_tick_events(world: WorldState, tick: int) -> None:
    """Append one tick's events to per-lens semantic buffers."""
    for event in world.event_log:
        if event.tick != tick:
            continue
        _buffer_one_event(world, event)


def ingest_new_events(world: WorldState) -> None:
    """Append any event_log entries not yet present in semantic buffers."""
    state = ensure_semantic_state(world)
    seen: set[str] = set()
    for ids in state.get("buffers", {}).values():
        seen.update(ids)
    for event in world.event_log:
        eid = str(event.event_id)
        if eid in seen:
            continue
        _buffer_one_event(world, event)


def _buffer_one_event(world: WorldState, event) -> None:
    state = ensure_semantic_state(world)
    buffers: dict[str, list[str]] = state.setdefault("buffers", {})
    raw = str(event.action.raw_input or "")
    if raw.startswith("[semantic]") or raw.startswith("[propagation]"):
        return
    eid = str(event.event_id)
    for key in _lens_keys_for_event(world, event):
        buf = buffers.setdefault(key, [])
        if eid not in buf:
            buf.append(eid)


def _parse_lens_key(key: str) -> tuple[SemanticLensKind, str]:
    kind_str, _, lid = key.partition(":")
    try:
        kind = SemanticLensKind(kind_str)
    except ValueError:
        kind = SemanticLensKind.AGENT
    return kind, lid


def _prior_beliefs_for_lens(world: WorldState, kind: SemanticLensKind, lid: str) -> list[str]:
    from .schemas import EdgeKind

    lines: list[str] = []
    if kind == SemanticLensKind.AGENT:
        for _tgt, edges in world.relational.edges.get(lid, {}).items():
            for edge in edges:
                if edge.kind == EdgeKind.BELIEVES_CLAIM:
                    claim = edge.meta.get("claim", "")
                    if claim:
                        lines.append(f"believes: {claim[:80]} (f={edge.weight:.2f})")
    elif kind == SemanticLensKind.FACTION:
        node = world.relational.nodes.get(lid)
        if node:
            for narrative in (node.meta.get("narratives") or [])[-3:]:
                lines.append(str(narrative))
    return lines[:8]


def build_slice(
    world: WorldState,
    lens_key: str,
    *,
    trigger: SemanticPassTrigger,
    cfg: Optional[SemanticDynamicsConfig] = None,
) -> SemanticSlice:
    cfg = cfg or world.config.semantic
    state = ensure_semantic_state(world)
    buffers: dict[str, list[str]] = state.get("buffers", {})
    event_ids = buffers.get(lens_key, [])[-cfg.max_events_per_slice :]

    id_to_event = {str(ev.event_id): ev for ev in world.event_log}
    canonical: list[CanonicalEventLine] = []
    tick_from = world.tick
    tick_to = world.tick
    for eid in event_ids:
        ev = id_to_event.get(eid)
        if ev is None:
            continue
        line = canonical_line_from_event(world, ev)
        canonical.append(line)
        tick_from = min(tick_from, line.tick)
        tick_to = max(tick_to, line.tick)

    kind, lid = _parse_lens_key(lens_key)
    if kind == SemanticLensKind.FACTION:
        node = world.relational.nodes.get(lid)
        lens_name = node.name if node else lid
    else:
        lens_name = _entity_name(world, lid)

    anchors: list[str] = []
    for aff in world.config.affordances[:5]:
        anchors.append(aff.keyword)

    return SemanticSlice(
        tick_from=tick_from,
        tick_to=tick_to,
        lens=kind,
        lens_id=lid,
        lens_name=lens_name,
        trigger=trigger,
        canonical_events=canonical,
        prior_beliefs=_prior_beliefs_for_lens(world, kind, lid),
        prior_narratives=_prior_beliefs_for_lens(world, kind, lid),
        cultural_anchors=anchors,
    )


def tick_has_crisis(world: WorldState, tick: int) -> bool:
    for event in world.event_log:
        if event.tick != tick:
            continue
        verb = _verb_str(event.action.verb)
        if verb in _VIOLENCE_VERBS:
            return True
        for t in event.transitions:
            k = t.kind.value if hasattr(t.kind, "value") else str(t.kind)
            if k not in _CRISIS_TRANSITIONS:
                continue
            if k == TransitionKind.ENTITY_HEALTH_CHANGED.value:
                if float(t.payload.get("delta", 0)) < -15:
                    return True
            else:
                return True
    return False


def should_run_semantic_pass(world: WorldState, tick: int) -> tuple[bool, SemanticPassTrigger]:
    cfg = world.config.semantic
    if not cfg.enabled:
        return False, SemanticPassTrigger.SCHEDULED
    state = ensure_semantic_state(world)
    last = int(state.get("last_pass_tick", -1))
    if tick_has_crisis(world, tick):
        return True, SemanticPassTrigger.CRISIS
    if last < 0:
        return True, SemanticPassTrigger.SCHEDULED
    if (tick - last) >= cfg.interval_ticks:
        return True, SemanticPassTrigger.SCHEDULED
    return False, SemanticPassTrigger.SCHEDULED


def lenses_with_pending_buffers(world: WorldState) -> list[str]:
    state = ensure_semantic_state(world)
    buffers: dict[str, list[str]] = state.get("buffers", {})
    return [k for k, ids in buffers.items() if ids]


def clear_lens_buffer(world: WorldState, lens_key: str) -> None:
    state = ensure_semantic_state(world)
    buffers = state.get("buffers", {})
    if lens_key in buffers:
        buffers[lens_key] = []
