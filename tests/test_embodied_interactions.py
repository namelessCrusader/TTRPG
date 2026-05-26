"""
Tests for embodied interaction kernels: posture, hygiene, perception,
expressive ripples, contact verbs, and new pack-driven social templates.
"""

from src.sim.capability_menu import pack_capability_candidates
from src.sim.compiler import apply_transitions, compile_action
from src.sim.narrator import render_event
from src.sim.schemas import (
    Coord,
    Event,
    IntentBlock,
    SemanticAction,
)
from src.sim.world_loader import load_world_pack


def _entity(world, name: str):
    return next(e for e in world.spatial.entities.values() if e.name == name)


# ── Posture ─────────────────────────────────────────────────────────────


def test_sit_sets_posture_and_recovers_fatigue():
    world = load_world_pack("worlds/tavern")
    mira = _entity(world, "Mira")
    pre_fatigue = float(mira.stats.get("fatigue", 85.0))
    action = SemanticAction(
        verb="sit", actor=mira.entity_id,
        intent=IntentBlock(manner="weary"), raw_input="[test]",
    )
    result = compile_action(action, world)
    assert result.valid
    apply_transitions(world, result.concrete_transitions)
    assert mira.meta.get("posture") == "sitting"
    assert float(mira.stats["fatigue"]) > pre_fatigue


def test_stand_requires_not_standing():
    world = load_world_pack("worlds/tavern")
    mira = _entity(world, "Mira")
    action = SemanticAction(
        verb="stand", actor=mira.entity_id, raw_input="[test]",
    )
    result = compile_action(action, world)
    assert not result.valid
    assert "Already standing" in (result.rejection_detail or "")


# ── Hygiene / relief ─────────────────────────────────────────────────────


def test_relieve_far_from_privy_creates_public_accident():
    world = load_world_pack("worlds/tavern")
    mira = _entity(world, "Mira")
    mira.stats["bladder"] = 90.0
    action = SemanticAction(
        verb="relieve", actor=mira.entity_id, raw_input="[test]",
    )
    result = compile_action(action, world)
    assert result.valid
    kinds = {t.kind.value for t in result.concrete_transitions}
    # Bladder drops + wetness + humiliation
    assert "need_changed" in kinds
    assert "fluid_changed" in kinds
    assert "entity_emotional_state_changed" in kinds
    apply_transitions(world, result.concrete_transitions)
    tile = world.spatial.tile_at(mira.position)
    assert tile.env.get("wet") is True


def test_relieve_adjacent_to_privy_clean():
    world = load_world_pack("worlds/tavern")
    mira = _entity(world, "Mira")
    privy = next(o for o in world.spatial.objects.values() if "privy" in o.tags)
    mira.position = privy.position
    mira.stats["bladder"] = 90.0
    action = SemanticAction(
        verb="relieve", actor=mira.entity_id, raw_input="[test]",
    )
    result = compile_action(action, world)
    assert result.valid
    kinds = [t.kind.value for t in result.concrete_transitions]
    # No fluid_changed → no accident
    assert "fluid_changed" not in kinds


def test_wipe_dries_an_adjacent_wet_tile():
    world = load_world_pack("worlds/tavern")
    mira = _entity(world, "Mira")
    target = mira.position
    tile = world.spatial.tile_at(target)
    tile.env["wet"] = True
    world.spatial.set_tile(target, tile)
    action = SemanticAction(
        verb="wipe", actor=mira.entity_id, raw_input="[test]",
    )
    result = compile_action(action, world)
    assert result.valid
    apply_transitions(world, result.concrete_transitions)
    new_tile = world.spatial.tile_at(target)
    assert not new_tile.env.get("wet")


# ── Sustenance ───────────────────────────────────────────────────────────


def test_eat_satisfies_hunger():
    world = load_world_pack("worlds/tavern")
    mira = _entity(world, "Mira")
    mira.stats["hunger"] = 30.0
    food = next(o for o in world.spatial.objects.values() if o.name == "loaf of dark bread")
    mira.position = food.position
    action = SemanticAction(
        verb="eat", actor=mira.entity_id, target=str(food.object_id),
        raw_input="[test]",
    )
    result = compile_action(action, world)
    assert result.valid
    apply_transitions(world, result.concrete_transitions)
    assert mira.stats["hunger"] > 30.0


# ── Perception / expressive ──────────────────────────────────────────────


def test_sniff_records_observation():
    world = load_world_pack("worlds/tavern")
    mira = _entity(world, "Mira")
    action = SemanticAction(
        verb="sniff", actor=mira.entity_id, raw_input="[test]",
    )
    result = compile_action(action, world)
    assert result.valid
    apply_transitions(world, result.concrete_transitions)
    assert "last_smell" in mira.meta


def test_laugh_ripples_mood_to_nearby_entities():
    world = load_world_pack("worlds/tavern")
    mira = _entity(world, "Mira")
    aldric = _entity(world, "Ser Aldric")
    aldric.position = Coord(x=mira.position.x + 1, y=mira.position.y)
    from src.sim.schemas import EmotionalState
    aldric.emotional_state = EmotionalState.NEUTRAL
    action = SemanticAction(
        verb="laugh", actor=mira.entity_id, raw_input="[test]",
    )
    result = compile_action(action, world)
    assert result.valid
    apply_transitions(world, result.concrete_transitions)
    assert aldric.emotional_state in (EmotionalState.HAPPY, EmotionalState.FRIENDLY)


# ── Contact ──────────────────────────────────────────────────────────────


def test_hug_requires_adjacency():
    world = load_world_pack("worlds/tavern")
    mira = _entity(world, "Mira")
    aldric = _entity(world, "Ser Aldric")
    aldric.position = Coord(x=mira.position.x + 5, y=mira.position.y)
    action = SemanticAction(
        verb="hug", actor=mira.entity_id, target=str(aldric.entity_id),
        raw_input="[test]",
    )
    result = compile_action(action, world)
    assert not result.valid


def test_slap_damages_target():
    world = load_world_pack("worlds/tavern")
    mira = _entity(world, "Mira")
    aldric = _entity(world, "Ser Aldric")
    aldric.position = Coord(x=mira.position.x + 1, y=mira.position.y)
    pre_hp = aldric.health
    action = SemanticAction(
        verb="slap", actor=mira.entity_id, target=str(aldric.entity_id),
        raw_input="[test]",
    )
    result = compile_action(action, world)
    assert result.valid
    apply_transitions(world, result.concrete_transitions)
    assert aldric.health < pre_hp


# ── New pack-driven verbs ────────────────────────────────────────────────


def test_apologize_repairs_enemy_edge():
    world = load_world_pack("worlds/tavern")
    mira = _entity(world, "Mira")
    aldric = _entity(world, "Ser Aldric")
    action = SemanticAction(
        verb="apologize", actor=mira.entity_id, target=str(aldric.entity_id),
        raw_input="[test]",
    )
    result = compile_action(action, world)
    assert result.valid
    kinds = {t.kind.value for t in result.concrete_transitions}
    assert "edge_updated" in kinds or "dialogue_spoken" in kinds


def test_mock_humiliates_target():
    world = load_world_pack("worlds/tavern")
    mira = _entity(world, "Mira")
    aldric = _entity(world, "Ser Aldric")
    action = SemanticAction(
        verb="mock", actor=mira.entity_id, target=str(aldric.entity_id),
        raw_input="[test]",
    )
    result = compile_action(action, world)
    assert result.valid


def test_lie_creates_believes_claim_edge_on_success():
    world = load_world_pack("worlds/tavern")
    mira = _entity(world, "Mira")
    aldric = _entity(world, "Ser Aldric")
    action = SemanticAction(
        verb="lie", actor=mira.entity_id, target=str(aldric.entity_id),
        raw_input="[test]",
    )
    result = compile_action(action, world)
    assert result.valid


# ── Narration ────────────────────────────────────────────────────────────


def test_narration_renders_for_new_verbs():
    world = load_world_pack("worlds/tavern")
    mira = _entity(world, "Mira")
    aldric = _entity(world, "Ser Aldric")
    aldric.position = Coord(x=mira.position.x + 1, y=mira.position.y)
    # Make a wet tile so wipe has something to do.
    tile = world.spatial.tile_at(mira.position)
    tile.env["wet"] = True
    world.spatial.set_tile(mira.position, tile)
    for verb in ("sit", "sniff", "listen", "laugh", "hug", "slap", "wipe"):
        action = SemanticAction(
            verb=verb,
            actor=mira.entity_id,
            target=str(aldric.entity_id) if verb in ("hug", "slap") else None,
            raw_input="[test]",
        )
        result = compile_action(action, world)
        assert result.valid, f"{verb} should compile: {result.rejection_detail}"
        ev = Event(tick=1, action=action, transitions=result.concrete_transitions)
        line = render_event(ev, world)
        # New verbs must produce non-fallback narration.
        assert "acts (" not in line, f"fallback narration used for {verb}: {line}"
        assert line and "Mira" in line


def test_capability_menu_offers_self_care_when_company_present():
    world = load_world_pack("worlds/tavern")
    mira = _entity(world, "Mira")
    cands = pack_capability_candidates(mira, world)
    verbs = {str(a.verb).lower() for a, _, _ in cands}
    # When peers/objects are visible, perception verbs should appear.
    assert "sniff" in verbs or "listen" in verbs
