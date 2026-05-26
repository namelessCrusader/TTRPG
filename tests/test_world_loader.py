"""
Tests for the world-pack loader.

These tests prove the moddability contract:
  - The default pack loads into a valid WorldState.
  - All soft-coded config (affordances, NPC policy weights) is reachable
    from world.config.
  - A user-authored pack (built in a tmp_path) loads and produces
    different behavior, with zero engine code edits.
  - Missing required files raise a clear WorldPackError.
"""

import textwrap
from pathlib import Path

import pytest

from src.sim.game_loop import GameLoop, make_test_world
from src.sim.grounding import classify
from src.sim.lm_adapter import MockLMAdapter
from src.sim.schemas import (
    ActionType,
    ActionZone,
    EntityKind,
    SemanticAction,
)
from src.sim.world_loader import WorldPackError, load_world_pack


# ─────────────────────────────────────────────────────────────────────────────
# Default pack
# ─────────────────────────────────────────────────────────────────────────────


def test_default_pack_loads():
    world = make_test_world()
    assert world.name == "Test World"
    # Two entities at the canonical positions
    kinds = {e.kind for e in world.spatial.entities.values()}
    assert EntityKind.PLAYER in kinds
    assert EntityKind.NPC in kinds


def test_default_pack_carries_affordances():
    world = make_test_world()
    keywords = {rule.keyword for rule in world.config.affordances}
    # The default affordance vocabulary should at least cover prayer and magic
    assert "pray" in keywords
    assert "magic" in keywords


def test_default_pack_carries_npc_policy_weights():
    world = make_test_world()
    cfg = world.config.npc_policy
    assert cfg.threat_lookback_ticks == 5
    assert cfg.flee_health_fraction == 0.25


def test_player_has_inventory_from_pack():
    world = make_test_world()
    player = next(
        e for e in world.spatial.entities.values()
        if e.kind == EntityKind.PLAYER
    )
    # Player starts with a throwing knife + a leather satchel.
    assert len(player.inventory) >= 1
    names = {world.spatial.objects[oid].name for oid in player.inventory}
    assert "throwing knife" in names
    # All items have weight/bulk fields set
    for oid in player.inventory:
        obj = world.spatial.objects[oid]
        assert obj.weight >= 0.0
        assert obj.bulk >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# User-authored pack (in tmp_path) — proves data-only moddability
# ─────────────────────────────────────────────────────────────────────────────


def _write_minimal_pack(root: Path, *, with_deity: bool) -> Path:
    """Author a small pack on disk. Returns the pack directory."""
    pack = root / "user_pack"
    pack.mkdir()

    (pack / "world.yaml").write_text(textwrap.dedent("""\
        name: "User-Authored Sandbox"
        spatial:
          width: 10
          height: 10
          map_file: "map.txt"
        meta:
          location_name: "shrine grove"
    """))

    # 10×10 walled room
    rows = []
    for y in range(10):
        if y == 0 or y == 9:
            rows.append("##########")
        else:
            rows.append("#........#")
    (pack / "map.txt").write_text("\n".join(rows))

    (pack / "entities.yaml").write_text(textwrap.dedent("""\
        entities:
          - id: player
            name: "Pilgrim"
            kind: player
            position: [2, 2]
            attributes: {strength: 50, perception: 60}
          - id: priest
            name: "Old Priest"
            kind: npc
            position: [6, 6]
            alertness: low
            emotional_state: friendly
            attributes: {strength: 30, perception: 50}
    """))

    (pack / "objects.yaml").write_text("objects: []\n")

    if with_deity:
        (pack / "factions.yaml").write_text(textwrap.dedent("""\
            factions:
              - id: temple_order
                name: "Order of the Dawn"
                kind: organization
                tags: [religious]
        """))
        (pack / "lore.yaml").write_text(textwrap.dedent("""\
            concepts:
              - id: solra
                name: "Solra, Goddess of Dawn"
                kind: faction
                tags: [deity, divine]
                meta:
                  domain: "dawn, renewal, oaths"
        """))
    else:
        (pack / "factions.yaml").write_text("factions: []\n")
        (pack / "lore.yaml").write_text("concepts: []\n")

    (pack / "relations.yaml").write_text("edges: []\n")

    (pack / "affordances.yaml").write_text(textwrap.dedent("""\
        affordances:
          - keyword: pray
            kinds: [faction, organization]
            name_keywords: []
            tags: [deity, divine, religious]
          - keyword: gods
            kinds: [faction]
            tags: [deity]
    """))

    (pack / "npc_policy.yaml").write_text(textwrap.dedent("""\
        threat_lookback_ticks: 10
        flee_health_fraction: 0.5
    """))

    return pack


def test_user_pack_with_deity_makes_prayer_grounded(tmp_path):
    """
    The smoking-gun test: a pack that declares a deity (no code changes)
    should make 'pray' route to Zone 1 instead of Zone 3.
    """
    pack = _write_minimal_pack(tmp_path, with_deity=True)
    world = load_world_pack(pack)
    player = next(
        e for e in world.spatial.entities.values()
        if e.kind == EntityKind.PLAYER
    )
    action = SemanticAction(
        verb=ActionType.SYMBOLIC,
        actor=player.entity_id,
        raw_input="I pray to Solra",
    )
    result = classify(action, world, "I pray to Solra")
    assert result.zone == ActionZone.GROUNDED


def test_user_pack_without_deity_keeps_prayer_orphaned(tmp_path):
    """
    Same affordance vocabulary, but no deity declared → prayer is Zone 3.
    Proves the affordance rule alone doesn't ground; the world data must.
    """
    pack = _write_minimal_pack(tmp_path, with_deity=False)
    world = load_world_pack(pack)
    player = next(
        e for e in world.spatial.entities.values()
        if e.kind == EntityKind.PLAYER
    )
    action = SemanticAction(
        verb=ActionType.SYMBOLIC,
        actor=player.entity_id,
        raw_input="I pray to the gods",
    )
    result = classify(action, world, "I pray to the gods")
    assert result.zone == ActionZone.ORPHANED
    assert "pray" in result.orphan_reason.lower() or "gods" in result.orphan_reason.lower()


def test_user_pack_overrides_npc_policy_weights(tmp_path):
    """The NPC policy reads its weights from the pack."""
    pack = _write_minimal_pack(tmp_path, with_deity=False)
    world = load_world_pack(pack)
    loop = GameLoop(world, adapter=MockLMAdapter())
    # ReactivePolicy is instantiated with the pack's values
    assert loop.npc_policy.threat_lookback_ticks == 10
    assert loop.npc_policy.flee_health_fraction == 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Error handling
# ─────────────────────────────────────────────────────────────────────────────


def test_missing_directory_raises(tmp_path):
    with pytest.raises(WorldPackError):
        load_world_pack(tmp_path / "does_not_exist")


def test_missing_world_yaml_raises(tmp_path):
    pack = tmp_path / "broken"
    pack.mkdir()
    with pytest.raises(WorldPackError):
        load_world_pack(pack)


def test_unknown_local_id_in_relations_raises(tmp_path):
    pack = _write_minimal_pack(tmp_path, with_deity=False)
    (pack / "relations.yaml").write_text(textwrap.dedent("""\
        edges:
          - source: player
            target: ghost_who_does_not_exist
            kind: enemy_of
    """))
    with pytest.raises(WorldPackError, match="unknown local id"):
        load_world_pack(pack)


def test_unknown_inventory_object_raises(tmp_path):
    pack = _write_minimal_pack(tmp_path, with_deity=False)
    (pack / "entities.yaml").write_text(textwrap.dedent("""\
        entities:
          - id: player
            name: "Pilgrim"
            kind: player
            position: [2, 2]
            attributes: {strength: 50, perception: 60}
            inventory:
              - phantom_sword
    """))
    with pytest.raises(WorldPackError, match="unknown.*object"):
        load_world_pack(pack)
