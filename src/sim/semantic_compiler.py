"""
Compile SemanticDelta batches into canonical transitions and meta updates.

Zero LM calls.  All numeric fields clipped; unknown endpoints dropped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from .relational import get_edges
from .schemas import (
    EdgeKind,
    EntityId,
    RelationalEdge,
    SemanticDelta,
    SemanticDynamicsConfig,
    SemanticLensKind,
    SemanticSlice,
    Transition,
    TransitionKind,
    WorldState,
)
from .semantic_slice import SEMANTIC_META_KEY, ensure_semantic_state

logger = logging.getLogger(__name__)

_ALLOWED_DELTA_KINDS = frozenset({
    "edge_shift",
    "belief_mutated",
    "goal_emerged",
    "faction_narrative",
    "social_momentum",
    "taboo",
    "suspicion_focus",
    "emotional_contagion",
})


@dataclass
class SemanticCompileResult:
    transitions: list[Transition] = field(default_factory=list)
    log_lines: list[str] = field(default_factory=list)
    applied_count: int = 0


def _known_node(world: WorldState, node_id: str) -> bool:
    if node_id in world.spatial.entities:
        return True
    return node_id in world.relational.nodes


def _clip_delta(cfg: SemanticDynamicsConfig, value: float) -> float:
    cap = cfg.edge_delta_cap
    return max(-cap, min(cap, value))


def _valid_event_id(world: WorldState, event_id: str) -> bool:
    return any(str(ev.event_id) == event_id for ev in world.event_log)


def compile_delta(
    world: WorldState,
    delta: SemanticDelta,
    slice_: SemanticSlice,
    *,
    cfg: Optional[SemanticDynamicsConfig] = None,
) -> SemanticCompileResult:
    cfg = cfg or world.config.semantic
    result = SemanticCompileResult()

    if delta.kind not in _ALLOWED_DELTA_KINDS:
        logger.debug("semantic: unknown delta kind %r", delta.kind)
        return result

    if not _valid_event_id(world, delta.rationale_event_id):
        logger.debug("semantic: invalid rationale_event_id %r", delta.rationale_event_id)
        return result

    if delta.kind == "edge_shift":
        src, tgt = delta.source, delta.target
        if not src or not tgt or not _known_node(world, src) or not _known_node(world, tgt):
            return result
        ek = delta.edge_kind or "distrusts"
        try:
            edge_kind = EdgeKind(ek)
        except ValueError:
            edge_kind = EdgeKind.DISTRUSTS
        d = _clip_delta(cfg, float(delta.delta or 0.1))
        existing = get_edges(world.relational, src, tgt, edge_kind)
        if existing:
            result.transitions.append(
                Transition(
                    kind=TransitionKind.EDGE_UPDATED,
                    payload={
                        "source": src,
                        "target": tgt,
                        "edge_kind": edge_kind.value,
                        "delta": d,
                        "cause": "semantic_pressure",
                        "rationale_event_id": delta.rationale_event_id,
                    },
                )
            )
        else:
            result.transitions.append(
                Transition(
                    kind=TransitionKind.EDGE_CREATED,
                    payload={
                        "source": src,
                        "target": tgt,
                        "edge_kind": edge_kind.value,
                        "weight": max(0.1, min(1.0, 0.4 + d)),
                        "meta": {
                            "cause": "semantic_pressure",
                            "rationale_event_id": delta.rationale_event_id,
                        },
                    },
                )
            )
        result.log_lines.append(
            f"{slice_.lens_name}: {src}→{tgt} {edge_kind.value} {d:+.2f}"
        )
        result.applied_count = 1
        return result

    if delta.kind == "belief_mutated":
        holder = delta.source
        if not holder or not _known_node(world, holder):
            return result
        claim = (delta.claim or "").strip()[:200]
        if not claim:
            return result
        fidelity = max(0.05, min(1.0, float(delta.fidelity or 0.5)))
        original = delta.target or holder
        result.transitions.append(
            Transition(
                kind=TransitionKind.EDGE_CREATED,
                payload={
                    "source": holder,
                    "target": original,
                    "edge_kind": EdgeKind.BELIEVES_CLAIM.value,
                    "weight": fidelity,
                    "meta": {
                        "claim": claim,
                        "rationale_event_id": delta.rationale_event_id,
                        "lens": slice_.lens.value,
                    },
                },
            )
        )
        result.log_lines.append(
            f"{slice_.lens_name} believes: \"{claim[:50]}…\" ({fidelity:.0%})"
            if len(claim) > 50
            else f"{slice_.lens_name} believes: \"{claim}\" ({fidelity:.0%})"
        )
        result.applied_count = 1
        return result

    if delta.kind == "goal_emerged":
        eid = delta.source
        if not eid or EntityId(eid) not in world.spatial.entities:
            return result
        goal = (delta.goal or "").strip()[:160]
        if not goal:
            return result
        result.transitions.append(
            Transition(
                kind=TransitionKind.ENTITY_GOAL_ADDED,
                payload={
                    "entity_id": eid,
                    "goal": goal,
                    "cause": "semantic_pressure",
                    "priority": float(delta.priority),
                    "rationale_event_id": delta.rationale_event_id,
                },
            )
        )
        result.log_lines.append(f"{slice_.lens_name} goal: {goal[:60]}")
        result.applied_count = 1
        return result

    state = ensure_semantic_state(world)
    pressure: dict[str, Any] = state.setdefault("pressure", {})

    if delta.kind == "faction_narrative":
        fid = delta.source or slice_.lens_id
        node = world.relational.nodes.get(fid)
        if node is None:
            return result
        theme = (delta.theme or "tension").strip()[:80]
        intensity = max(0.0, min(1.0, float(delta.intensity or 0.3)))
        narratives = node.meta.setdefault("narratives", [])
        line = f"[tick {world.tick}] {theme} (intensity={intensity:.2f})"
        narratives.append(line)
        node.meta["narratives"] = narratives[-12:]
        node.meta["dominant_theme"] = theme
        pressure[f"faction:{fid}"] = {"theme": theme, "intensity": intensity}
        result.log_lines.append(f"Faction {node.name}: narrative «{theme}»")
        result.applied_count = 1
        return result

    if delta.kind == "social_momentum":
        scope = delta.scope or "world"
        axis = delta.axis or "violence_vs_reconciliation"
        d = _clip_delta(cfg, float(delta.delta or 0.1))
        bucket = pressure.setdefault(scope, {})
        if isinstance(bucket, dict):
            bucket[axis] = max(-1.0, min(1.0, float(bucket.get(axis, 0.0)) + d))
        result.log_lines.append(f"{scope}: {axis} {d:+.2f}")
        result.applied_count = 1
        return result

    if delta.kind == "taboo":
        fid = delta.source or slice_.lens_id
        node = world.relational.nodes.get(fid)
        if node is None:
            return result
        rule = (delta.rule or "").strip()[:120]
        if not rule:
            return result
        strength = max(0.0, min(1.0, float(delta.strength or 0.5)))
        taboos = node.meta.setdefault("taboos", [])
        taboos.append({"rule": rule, "strength": strength, "tick": world.tick})
        node.meta["taboos"] = taboos[-10:]
        result.log_lines.append(f"Taboo ({node.name}): {rule}")
        result.applied_count = 1
        return result

    if delta.kind in ("suspicion_focus", "emotional_contagion"):
        eid = delta.source
        if not eid or EntityId(eid) not in world.spatial.entities:
            return result
        ent = world.spatial.entities[EntityId(eid)]
        focus = (delta.claim or delta.theme or delta.kind).strip()[:80]
        ent.meta.setdefault("semantic_focus", []).append({
            "kind": delta.kind,
            "focus": focus,
            "tick": world.tick,
            "event_id": delta.rationale_event_id,
        })
        ent.meta["semantic_focus"] = ent.meta["semantic_focus"][-8:]
        result.log_lines.append(f"{ent.name}: {delta.kind} → {focus}")
        result.applied_count = 1
        return result

    return result


def compile_deltas(
    world: WorldState,
    slice_: SemanticSlice,
    deltas: list[SemanticDelta],
) -> SemanticCompileResult:
    cfg = world.config.semantic
    combined = SemanticCompileResult()
    for delta in deltas[: cfg.max_deltas_per_lens]:
        part = compile_delta(world, delta, slice_, cfg=cfg)
        combined.transitions.extend(part.transitions)
        combined.log_lines.extend(part.log_lines)
        combined.applied_count += part.applied_count

    if combined.log_lines:
        log = ensure_semantic_state(world).setdefault("applied_log", [])
        for line in combined.log_lines:
            log.append(f"tick {world.tick}: {line}")
        ensure_semantic_state(world)["applied_log"] = log[-40:]

    return combined


def apply_semantic_transitions(
    world: WorldState,
    transitions: list[Transition],
    *,
    tick: int,
) -> None:
    """Apply semantic transitions to world + relational graph."""
    from .compiler import apply_transitions
    from .relational import apply_event_to_graph
    from .schemas import ActionType, Event, SemanticAction, new_event_id

    if not transitions:
        return

    apply_transitions(world, transitions)
    event = Event(
        event_id=new_event_id(),
        tick=tick,
        action=SemanticAction(
            verb=ActionType.SYMBOLIC,
            actor=EntityId("system"),
            raw_input="[semantic]",
        ),
        transitions=transitions,
        witnesses=[],
        narrative_hint="[semantic dynamics]",
    )
    world.event_log.append(event)
    apply_event_to_graph(world.relational, event, tick)
