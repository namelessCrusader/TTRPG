"""
M2: Persistent relational graph.

Stores the high-level symbolic world: factions, kinship, emotional bonds,
debts, memories, historical events, etc.

Responsibilities:
  - Node and edge CRUD with full type enforcement
  - Event-driven updates (an Event triggers graph mutations)
  - Serialization / deserialization with no loss
  - Memory decay via explicit tick-based aging

This module never touches SpatialGrid or SemanticProjection.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from .schemas import (
    EdgeKind,
    EntityId,
    Event,
    EventId,
    FactionId,
    NodeKind,
    RelationalEdge,
    RelationalGraph,
    RelationalNode,
    TransitionKind,
)

# Maximum number of verbs stored in the INTERACTED edge's rolling window.
INTERACTION_HISTORY_WINDOW: int = 5


# ---------------------------------------------------------------------------
# Graph construction helpers
# ---------------------------------------------------------------------------


def add_node(
    graph: RelationalGraph,
    node_id: str,
    kind: NodeKind,
    name: str,
    meta: Optional[dict[str, Any]] = None,
) -> RelationalNode:
    node = RelationalNode(
        node_id=node_id, kind=kind, name=name, meta=meta or {}
    )
    graph.nodes[node_id] = node
    return node


def get_node(graph: RelationalGraph, node_id: str) -> Optional[RelationalNode]:
    return graph.nodes.get(node_id)


def remove_node(graph: RelationalGraph, node_id: str) -> None:
    """Remove a node and all its edges."""
    graph.nodes.pop(node_id, None)
    graph.edges.pop(node_id, None)
    for targets in graph.edges.values():
        targets.pop(node_id, None)


# ---------------------------------------------------------------------------
# Edge management
# ---------------------------------------------------------------------------


def add_edge(
    graph: RelationalGraph,
    source: str,
    target: str,
    kind: EdgeKind,
    *,
    weight: float = 1.0,
    tick: int = 0,
    evidence: Optional[list[EventId]] = None,
    meta: Optional[dict[str, Any]] = None,
) -> RelationalEdge:
    """
    Add a directed edge.  Multiple edges of *different* kinds between the same
    pair are allowed (a person can be both an ally and a debtor).  Duplicate
    edges of the *same* kind are deduplicated by updating in place.
    """
    edge = RelationalEdge(
        source=source,
        target=target,
        kind=kind,
        weight=weight,
        tick_created=tick,
        tick_last_updated=tick,
        evidence=evidence or [],
        meta=meta or {},
    )
    if source not in graph.edges:
        graph.edges[source] = {}
    if target not in graph.edges[source]:
        graph.edges[source][target] = []

    # Replace if same kind already exists
    existing = graph.edges[source][target]
    for i, e in enumerate(existing):
        if e.kind == kind:
            existing[i] = edge
            return edge
    existing.append(edge)
    return edge


def get_edges(
    graph: RelationalGraph,
    source: str,
    target: Optional[str] = None,
    kind: Optional[EdgeKind] = None,
) -> list[RelationalEdge]:
    """Return edges from source, optionally filtered by target and/or kind."""
    result: list[RelationalEdge] = []
    targets = graph.edges.get(source, {})
    if target is not None:
        edge_list = targets.get(target, [])
        result = [e for e in edge_list if kind is None or e.kind == kind]
    else:
        for edge_list in targets.values():
            result.extend(
                e for e in edge_list if kind is None or e.kind == kind
            )
    return result


def remove_edge(
    graph: RelationalGraph,
    source: str,
    target: str,
    kind: EdgeKind,
) -> bool:
    """Remove a specific directed edge. Returns True if found and removed."""
    edges = graph.edges.get(source, {}).get(target, [])
    for i, e in enumerate(edges):
        if e.kind == kind:
            edges.pop(i)
            return True
    return False


def update_edge_weight(
    graph: RelationalGraph,
    source: str,
    target: str,
    kind: EdgeKind,
    delta: float,
    tick: int,
) -> Optional[RelationalEdge]:
    """Adjust the weight of an existing edge by delta, clamped to [-1, 1]."""
    edges = graph.edges.get(source, {}).get(target, [])
    for edge in edges:
        if edge.kind == kind:
            new_weight = max(-1.0, min(1.0, edge.weight + delta))
            edge.weight = new_weight
            edge.tick_last_updated = tick
            return edge
    return None


# ---------------------------------------------------------------------------
# Faction helpers
# ---------------------------------------------------------------------------


def faction_members(
    graph: RelationalGraph, faction_id: FactionId
) -> list[str]:
    """Return all node ids that are members of the given faction."""
    members: list[str] = []
    for source, targets in graph.edges.items():
        for target, edges in targets.items():
            if target == faction_id:
                for e in edges:
                    if e.kind == EdgeKind.FACTION_MEMBER:
                        members.append(source)
    return members


def shared_factions(
    graph: RelationalGraph, a: str, b: str
) -> list[str]:
    """Return faction ids that both a and b belong to."""
    def factions_of(node: str) -> set[str]:
        result: set[str] = set()
        for target, edges in graph.edges.get(node, {}).items():
            for e in edges:
                if e.kind == EdgeKind.FACTION_MEMBER:
                    result.add(target)
        return result

    return list(factions_of(a) & factions_of(b))


# ---------------------------------------------------------------------------
# Event-driven graph updates
# ---------------------------------------------------------------------------


def apply_event_to_graph(
    graph: RelationalGraph,
    event: Event,
    tick: int,
) -> None:
    """
    Update the relational graph based on an event's transitions.

    This is the *only* sanctioned way to mutate the graph as a result of
    in-simulation events.  Manual graph mutations outside of this path should
    only be used during world construction.
    """
    for transition in event.transitions:
        kind = transition.kind
        payload = transition.payload

        if kind == TransitionKind.EDGE_CREATED:
            edge_kind_raw = payload.get("edge_kind") or payload.get("kind")
            if edge_kind_raw is None:
                continue
            add_edge(
                graph,
                source=payload["source"],
                target=payload["target"],
                kind=EdgeKind(edge_kind_raw),
                weight=payload.get("weight", 1.0),
                tick=tick,
                evidence=[event.event_id],
                meta=payload.get("meta", {}),
            )

        elif kind == TransitionKind.EDGE_UPDATED:
            edge_kind_raw = payload.get("edge_kind") or payload.get("kind")
            if edge_kind_raw is None:
                continue
            update_edge_weight(
                graph,
                source=payload["source"],
                target=payload["target"],
                kind=EdgeKind(edge_kind_raw),
                delta=payload.get("delta", 0.0),
                tick=tick,
            )

        elif kind == TransitionKind.EDGE_REMOVED:
            edge_kind_raw = payload.get("edge_kind") or payload.get("kind")
            if edge_kind_raw is None:
                continue
            remove_edge(
                graph,
                source=payload["source"],
                target=payload["target"],
                kind=EdgeKind(edge_kind_raw),
            )

        elif kind == TransitionKind.RUMOR_PROPAGATED:
            # Record that the witness remembers the event
            for witness in event.witnesses:
                add_edge(
                    graph,
                    source=witness,
                    target=payload.get("subject", event.action.actor),
                    kind=EdgeKind.REMEMBERS_EVENT,
                    weight=payload.get("salience", 0.5),
                    tick=tick,
                    evidence=[event.event_id],
                )

    # ── Interaction memory ───────────────────────────────────────────────────
    # For any freeform (non-kernel) action that targets an entity and produced
    # a DIALOGUE_SPOKEN transition, upsert an INTERACTED edge between the
    # actor and the target.  This gives the relational graph a bounded,
    # O(entity_pairs × K) summary of interaction history that the projection
    # can surface to the LM without reading the full event log.
    _maybe_upsert_interacted_edge(graph, event, tick)


# ---------------------------------------------------------------------------
# Interaction memory helper
# ---------------------------------------------------------------------------


def _maybe_upsert_interacted_edge(
    graph: RelationalGraph,
    event: Event,
    tick: int,
) -> None:
    """
    Upsert an INTERACTED edge between actor and target for freeform events.

    Triggered when the event has a DIALOGUE_SPOKEN transition (the standard
    record for any freeform action compiled by _compile_freeform or
    _compile_from_template) and an entity-id target.

    The edge's `meta` carries:
      - recent_verbs: list[str]  — rolling window of last K verbs, newest last
      - last_tick:    int        — tick of the most recent interaction
      - count:        int        — total number of interactions recorded

    Storage is O(entity_pairs × INTERACTION_HISTORY_WINDOW).  The window is
    trimmed to that bound on every update, so the graph never grows unboundedly.
    """
    # Only trigger for events with an entity-id (string) target.
    if event.action.target is None:
        return
    from .schemas import Coord
    if isinstance(event.action.target, Coord):
        return

    # Only when the event produced at least one DIALOGUE_SPOKEN transition
    # (which is the canonical record for all freeform / generic actions).
    has_freeform_record = any(
        t.kind == TransitionKind.DIALOGUE_SPOKEN
        for t in event.transitions
    )
    if not has_freeform_record:
        return

    actor = str(event.action.actor)
    target = str(event.action.target)
    verb = str(event.action.verb)

    # Skip if either node does not exist in the graph.
    if actor not in graph.nodes or target not in graph.nodes:
        return

    # Find or create the INTERACTED edge.
    existing = _get_edge(graph, actor, target, EdgeKind.INTERACTED)
    if existing is None:
        add_edge(
            graph,
            source=actor,
            target=target,
            kind=EdgeKind.INTERACTED,
            weight=1.0,
            tick=tick,
            evidence=[event.event_id],
            meta={
                "recent_verbs": [verb],
                "last_tick": tick,
                "count": 1,
            },
        )
    else:
        # Update rolling window.
        recent: list[str] = existing.meta.get("recent_verbs", [])
        recent.append(verb)
        # Keep only the last INTERACTION_HISTORY_WINDOW entries.
        existing.meta["recent_verbs"] = recent[-INTERACTION_HISTORY_WINDOW:]
        existing.meta["last_tick"] = tick
        existing.meta["count"] = existing.meta.get("count", 0) + 1
        existing.tick_last_updated = tick
        existing.evidence = (existing.evidence + [event.event_id])[-10:]


def _get_edge(
    graph: RelationalGraph,
    source: str,
    target: str,
    kind: EdgeKind,
) -> Optional[RelationalEdge]:
    """
    Return the edge from source→target of the given kind, or None.

    graph.edges is structured as source → target → list[RelationalEdge].
    """
    target_map = graph.edges.get(source, {})
    for edge in target_map.get(target, []):
        if edge.kind == kind:
            return edge
    return None


# ---------------------------------------------------------------------------
# Memory decay
# ---------------------------------------------------------------------------


def decay_edges(
    graph: RelationalGraph,
    current_tick: int,
    *,
    decay_rate: float = 0.01,
    min_weight_to_remove: float = 0.05,
    kinds_to_decay: Optional[set[EdgeKind]] = None,
) -> int:
    """
    Age all edges of the specified kinds (default: REMEMBERS_EVENT, WITNESSED).
    Edges whose absolute weight drops below min_weight_to_remove are removed.

    Returns the number of edges removed.
    """
    if kinds_to_decay is None:
        kinds_to_decay = {EdgeKind.REMEMBERS_EVENT, EdgeKind.WITNESSED}

    to_remove: list[tuple[str, str, EdgeKind]] = []

    for source, targets in graph.edges.items():
        for target, edge_list in targets.items():
            for edge in edge_list:
                if edge.kind not in kinds_to_decay:
                    continue
                age = current_tick - edge.tick_last_updated
                decayed = abs(edge.weight) - decay_rate * age
                if decayed < min_weight_to_remove:
                    to_remove.append((source, target, edge.kind))
                else:
                    edge.weight = (
                        decayed if edge.weight >= 0 else -decayed
                    )

    for source, target, kind in to_remove:
        remove_edge(graph, source, target, kind)

    return len(to_remove)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def serialize_graph(graph: RelationalGraph) -> str:
    """Serialize the graph to a JSON string with full fidelity."""
    return graph.model_dump_json()


def deserialize_graph(data: str) -> RelationalGraph:
    """Deserialize a graph from a JSON string produced by serialize_graph."""
    return RelationalGraph.model_validate_json(data)


def graph_to_dict(graph: RelationalGraph) -> dict:
    return json.loads(serialize_graph(graph))


# ---------------------------------------------------------------------------
# Query helpers for the projection layer
# ---------------------------------------------------------------------------


def get_entity_faction_ids(
    graph: RelationalGraph, entity_id: EntityId
) -> list[FactionId]:
    factions: list[FactionId] = []
    for target, edges in graph.edges.get(entity_id, {}).items():
        for e in edges:
            if e.kind == EdgeKind.FACTION_MEMBER:
                node = graph.nodes.get(target)
                if node and node.kind == NodeKind.FACTION:
                    factions.append(FactionId(target))
    return factions


def get_relationship_summary(
    graph: RelationalGraph,
    entity_a: str,
    entity_b: str,
) -> list[str]:
    """
    Return human-readable descriptions of known relationships between
    entity_a and entity_b (both directions).
    """
    descriptions: list[str] = []
    label_map: dict[EdgeKind, str] = {
        EdgeKind.ENEMY_OF: "is an enemy of",
        EdgeKind.ALLY_OF: "is an ally of",
        EdgeKind.FEARS: "fears",
        EdgeKind.RESPECTS: "respects",
        EdgeKind.DISTRUSTS: "distrusts",
        EdgeKind.OWES_DEBT: "owes a debt to",
        EdgeKind.KIN: "is kin to",
        EdgeKind.EMPLOYS: "employs",
        EdgeKind.EMPLOYED_BY: "is employed by",
    }
    name_a = graph.nodes.get(entity_a, RelationalNode(node_id=entity_a, kind=NodeKind.ENTITY, name=entity_a)).name
    name_b = graph.nodes.get(entity_b, RelationalNode(node_id=entity_b, kind=NodeKind.ENTITY, name=entity_b)).name

    for edge in get_edges(graph, entity_a, entity_b):
        label = label_map.get(edge.kind, edge.kind.value)
        descriptions.append(f"{name_a} {label} {name_b}")
    for edge in get_edges(graph, entity_b, entity_a):
        label = label_map.get(edge.kind, edge.kind.value)
        descriptions.append(f"{name_b} {label} {name_a}")
    return descriptions


def get_faction_tension(
    graph: RelationalGraph,
    entity_a: str,
    entity_b: str,
) -> str:
    """
    Return a qualitative tension level between two entities based on their
    faction relationships and direct edges.
    """
    factions_a = set(get_entity_faction_ids(graph, EntityId(entity_a)))
    factions_b = set(get_entity_faction_ids(graph, EntityId(entity_b)))

    direct_enemy = any(
        e.kind == EdgeKind.ENEMY_OF
        for e in get_edges(graph, entity_a, entity_b)
        + get_edges(graph, entity_b, entity_a)
    )
    if direct_enemy:
        return "high"

    # Check if their factions are enemies
    for fa in factions_a:
        for fb in factions_b:
            if any(
                e.kind == EdgeKind.ENEMY_OF
                for e in get_edges(graph, fa, fb) + get_edges(graph, fb, fa)
            ):
                return "medium"

    if factions_a & factions_b:
        return "low"

    return "none"
