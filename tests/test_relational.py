"""
Tests for M2: Relational graph.

Acceptance criteria from the spec:
  - Adding event E with participants [A, B] updates edges for A and B only
  - Graph state after N ticks can be serialized and deserialized with no loss
  - Reading faction membership returns identical result on two consecutive calls
"""

import pytest

from src.sim.relational import (
    add_edge,
    add_node,
    decay_edges,
    deserialize_graph,
    faction_members,
    get_edges,
    get_entity_faction_ids,
    get_faction_tension,
    remove_edge,
    serialize_graph,
    update_edge_weight,
)
from src.sim.schemas import (
    EdgeKind,
    EntityId,
    FactionId,
    NodeKind,
    RelationalGraph,
)


def _make_graph_with_two_entities():
    graph = RelationalGraph()
    add_node(graph, "ent_alice", NodeKind.ENTITY, "Alice")
    add_node(graph, "ent_bob", NodeKind.ENTITY, "Bob")
    add_node(graph, "faction_guards", NodeKind.FACTION, "City Guard")
    return graph


# ---------------------------------------------------------------------------
# Node / edge CRUD
# ---------------------------------------------------------------------------


def test_add_and_get_node():
    graph = RelationalGraph()
    node = add_node(graph, "n1", NodeKind.ENTITY, "Alice")
    assert graph.nodes["n1"].name == "Alice"
    assert graph.nodes["n1"].kind == NodeKind.ENTITY


def test_add_and_get_edge():
    graph = _make_graph_with_two_entities()
    edge = add_edge(graph, "ent_alice", "ent_bob", EdgeKind.ENEMY_OF, weight=1.0)
    edges = get_edges(graph, "ent_alice", "ent_bob", EdgeKind.ENEMY_OF)
    assert len(edges) == 1
    assert edges[0].kind == EdgeKind.ENEMY_OF


def test_edge_deduplication_same_kind():
    """Adding the same edge kind twice should update, not duplicate."""
    graph = _make_graph_with_two_entities()
    add_edge(graph, "ent_alice", "ent_bob", EdgeKind.ALLY_OF, weight=0.5)
    add_edge(graph, "ent_alice", "ent_bob", EdgeKind.ALLY_OF, weight=0.9)
    edges = get_edges(graph, "ent_alice", "ent_bob", EdgeKind.ALLY_OF)
    assert len(edges) == 1
    assert edges[0].weight == 0.9


def test_multiple_edge_kinds_between_same_pair():
    graph = _make_graph_with_two_entities()
    add_edge(graph, "ent_alice", "ent_bob", EdgeKind.ALLY_OF)
    add_edge(graph, "ent_alice", "ent_bob", EdgeKind.OWES_DEBT)
    edges = get_edges(graph, "ent_alice", "ent_bob")
    assert len(edges) == 2


def test_remove_edge():
    graph = _make_graph_with_two_entities()
    add_edge(graph, "ent_alice", "ent_bob", EdgeKind.ENEMY_OF)
    removed = remove_edge(graph, "ent_alice", "ent_bob", EdgeKind.ENEMY_OF)
    assert removed is True
    edges = get_edges(graph, "ent_alice", "ent_bob", EdgeKind.ENEMY_OF)
    assert len(edges) == 0


def test_remove_nonexistent_edge():
    graph = _make_graph_with_two_entities()
    removed = remove_edge(graph, "ent_alice", "ent_bob", EdgeKind.FEARS)
    assert removed is False


def test_update_edge_weight():
    graph = _make_graph_with_two_entities()
    add_edge(graph, "ent_alice", "ent_bob", EdgeKind.ALLY_OF, weight=0.5)
    edge = update_edge_weight(graph, "ent_alice", "ent_bob", EdgeKind.ALLY_OF, delta=-0.3, tick=5)
    assert edge is not None
    assert abs(edge.weight - 0.2) < 1e-9
    assert edge.tick_last_updated == 5


def test_update_edge_weight_clamped():
    graph = _make_graph_with_two_entities()
    add_edge(graph, "ent_alice", "ent_bob", EdgeKind.ALLY_OF, weight=0.9)
    edge = update_edge_weight(graph, "ent_alice", "ent_bob", EdgeKind.ALLY_OF, delta=0.5, tick=1)
    assert edge.weight == 1.0


# ---------------------------------------------------------------------------
# Faction helpers
# ---------------------------------------------------------------------------


def test_faction_membership():
    graph = _make_graph_with_two_entities()
    add_edge(graph, "ent_alice", "faction_guards", EdgeKind.FACTION_MEMBER)
    add_edge(graph, "ent_bob", "faction_guards", EdgeKind.FACTION_MEMBER)
    members = faction_members(graph, FactionId("faction_guards"))
    assert "ent_alice" in members
    assert "ent_bob" in members


def test_faction_membership_idempotent():
    """Reading faction membership twice returns the same result."""
    graph = _make_graph_with_two_entities()
    add_edge(graph, "ent_alice", "faction_guards", EdgeKind.FACTION_MEMBER)
    r1 = faction_members(graph, FactionId("faction_guards"))
    r2 = faction_members(graph, FactionId("faction_guards"))
    assert sorted(r1) == sorted(r2)


def test_faction_tension_enemy():
    graph = _make_graph_with_two_entities()
    add_edge(graph, "ent_alice", "ent_bob", EdgeKind.ENEMY_OF)
    tension = get_faction_tension(graph, "ent_alice", "ent_bob")
    assert tension == "high"


def test_faction_tension_none():
    graph = _make_graph_with_two_entities()
    tension = get_faction_tension(graph, "ent_alice", "ent_bob")
    assert tension == "none"


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------


def test_serialization_round_trip():
    graph = _make_graph_with_two_entities()
    add_edge(graph, "ent_alice", "ent_bob", EdgeKind.ALLY_OF, weight=0.75, tick=3)
    add_edge(graph, "ent_bob", "faction_guards", EdgeKind.FACTION_MEMBER)
    serialized = serialize_graph(graph)
    restored = deserialize_graph(serialized)
    assert restored.nodes.keys() == graph.nodes.keys()
    orig_edges = get_edges(graph, "ent_alice", "ent_bob", EdgeKind.ALLY_OF)
    rest_edges = get_edges(restored, "ent_alice", "ent_bob", EdgeKind.ALLY_OF)
    assert len(orig_edges) == len(rest_edges) == 1
    assert abs(orig_edges[0].weight - rest_edges[0].weight) < 1e-9


# ---------------------------------------------------------------------------
# Memory decay
# ---------------------------------------------------------------------------


def test_decay_removes_stale_edges():
    graph = _make_graph_with_two_entities()
    add_edge(graph, "ent_alice", "ent_bob", EdgeKind.REMEMBERS_EVENT, weight=0.1, tick=0)
    removed = decay_edges(graph, current_tick=100, decay_rate=0.01, min_weight_to_remove=0.05)
    assert removed == 1
    edges = get_edges(graph, "ent_alice", "ent_bob", EdgeKind.REMEMBERS_EVENT)
    assert len(edges) == 0


def test_decay_preserves_strong_edges():
    graph = _make_graph_with_two_entities()
    add_edge(graph, "ent_alice", "ent_bob", EdgeKind.REMEMBERS_EVENT, weight=0.9, tick=0)
    removed = decay_edges(graph, current_tick=5, decay_rate=0.01, min_weight_to_remove=0.05)
    assert removed == 0
    edges = get_edges(graph, "ent_alice", "ent_bob", EdgeKind.REMEMBERS_EVENT)
    assert len(edges) == 1
