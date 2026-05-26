"""Tests for property-intersection rules (property_interactions.yaml)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.sim.compiler import compile_action
from src.sim.game_loop import make_test_world
from src.sim.property_interaction_resolver import (
    _intent_category,
    try_property_interaction,
)
from src.sim.schemas import (
    EntityKind,
    IntentBlock,
    ObjectState,
    PropertyInteractionRule,
    SemanticAction,
    StyleBlock,
    TransitionKind,
    TransitionProposal,
)
from src.sim.world_loader import load_world_pack


# ── helpers ────────────────────────────────────────────────────────────────

def _player_id(world):
    return next(
        eid for eid, e in world.spatial.entities.items()
        if e.kind == EntityKind.PLAYER
    )


def _npc_id(world):
    return next(
        eid for eid, e in world.spatial.entities.items()
        if e.kind == EntityKind.NPC
    )


def _add_prop_rule(world, rule: PropertyInteractionRule) -> None:
    world.config.property_interactions = [rule] + list(world.config.property_interactions)


def _give_item(world, actor_id, tags: list[str], name: str = "item") -> str:
    """Add an object with given tags to actor's inventory and return its id."""
    import uuid
    oid = f"obj_{uuid.uuid4().hex[:8]}"
    region_id = world.active_region_id or "region_main"
    obj = ObjectState(
        object_id=oid,
        name=name,
        tags=tags,
        position=None,
        region_id=region_id,
    )
    world.spatial.objects[oid] = obj
    actor = world.spatial.entities[actor_id]
    actor.inventory.append(oid)
    # equip it as weapon so the resolver finds it
    actor.equipped_weapon = oid
    return oid


def _action(actor, verb, target=None, intent="test"):
    return SemanticAction(
        verb=verb,
        actor=actor,
        target=target,
        intent=IntentBlock(rationale=intent),
        style=StyleBlock(aggression=30, visibility=50),
        raw_input=intent,
    )


# ── intent-category classification ────────────────────────────────────────

def test_intent_category_social():
    assert _intent_category("speak") == "social"
    assert _intent_category("persuade") == "social"


def test_intent_category_physical():
    assert _intent_category("attack") == "physical"
    assert _intent_category("smash") == "physical"
    assert _intent_category("unknown_verb") == "physical"  # default


def test_intent_category_chemical():
    assert _intent_category("pour") == "chemical"
    assert _intent_category("apply") == "chemical"


def test_intent_category_thermal():
    assert _intent_category("ignite") == "thermal"
    assert _intent_category("melt") == "thermal"


# ── resolver unit tests ────────────────────────────────────────────────────

def test_no_rules_returns_none():
    world = make_test_world()
    pid = _player_id(world)
    nid = _npc_id(world)
    world.config.property_interactions = []
    result = try_property_interaction(_action(pid, "pour", nid), world)
    assert result is None


def test_rule_with_no_matching_tags_returns_none():
    world = make_test_world()
    pid = _player_id(world)
    nid = _npc_id(world)
    rule = PropertyInteractionRule(
        id="test_no_match",
        actor_or_tool_tags=["NONEXISTENT_TAG"],
        target_tags=["ALSO_NONEXISTENT"],
        effects=[TransitionProposal(kind="entity_health_changed", payload={"entity_id": "$target", "delta": -5})],
    )
    _add_prop_rule(world, rule)
    assert try_property_interaction(_action(pid, "pour", nid), world) is None


def test_rule_fires_when_actor_has_matching_tag():
    world = make_test_world()
    pid = _player_id(world)
    nid = _npc_id(world)

    # Give actor the [corrosive] tag directly (simulating acid-soaked hands)
    world.spatial.entities[pid].tags = list(world.spatial.entities[pid].tags or []) + ["corrosive"]
    # Give NPC the [metal] tag (iron armor)
    world.spatial.entities[nid].tags = list(world.spatial.entities[nid].tags or []) + ["metal"]

    rule = PropertyInteractionRule(
        id="test_corrode",
        actor_or_tool_tags=["corrosive"],
        target_tags=["metal"],
        effects=[
            TransitionProposal(kind="entity_health_changed", payload={"entity_id": "$target", "delta": -6, "cause": "acid_corrosion"}),
        ],
        priority=10,
    )
    _add_prop_rule(world, rule)

    result = try_property_interaction(_action(pid, "pour", nid), world)
    assert result is not None
    assert result.valid
    # Should produce at least the health-changed transition.
    kinds = {t.kind for t in result.concrete_transitions}
    assert TransitionKind.ENTITY_HEALTH_CHANGED in kinds


def test_rule_fires_when_equipped_tool_has_matching_tag():
    world = make_test_world()
    pid = _player_id(world)
    nid = _npc_id(world)

    # Give NPC a [flammable] tag
    world.spatial.entities[nid].tags = list(world.spatial.entities[nid].tags or []) + ["flammable"]

    # Give player a torch (fire_source tag)
    _give_item(world, pid, tags=["on_fire", "fire_source"], name="torch")

    rule = PropertyInteractionRule(
        id="test_ignite",
        actor_or_tool_tags=["fire_source", "on_fire"],
        target_tags=["flammable"],
        effects=[
            TransitionProposal(kind="entity_tag_changed", payload={"entity_id": "$target", "add": "on_fire"}),
        ],
        priority=15,
    )
    _add_prop_rule(world, rule)

    result = try_property_interaction(_action(pid, "apply", nid), world)
    assert result is not None
    assert result.valid


def test_immunity_tag_blocks_rule():
    world = make_test_world()
    pid = _player_id(world)
    nid = _npc_id(world)

    world.spatial.entities[pid].tags = list(world.spatial.entities[pid].tags or []) + ["corrosive"]
    # NPC has metal but also acid_resistant immunity
    world.spatial.entities[nid].tags = list(world.spatial.entities[nid].tags or []) + ["metal", "acid_resistant"]

    rule = PropertyInteractionRule(
        id="test_immune",
        actor_or_tool_tags=["corrosive"],
        target_tags=["metal"],
        target_immunity_tags=["acid_resistant"],
        effects=[
            TransitionProposal(kind="entity_health_changed", payload={"entity_id": "$target", "delta": -6}),
        ],
        priority=10,
    )
    _add_prop_rule(world, rule)
    result = try_property_interaction(_action(pid, "pour", nid), world)
    assert result is None


def test_social_verb_excluded_from_any():
    """Rules with intent_categories=[any] should NOT fire on social verbs."""
    world = make_test_world()
    pid = _player_id(world)
    nid = _npc_id(world)

    world.spatial.entities[pid].tags = list(world.spatial.entities[pid].tags or []) + ["corrosive"]
    world.spatial.entities[nid].tags = list(world.spatial.entities[nid].tags or []) + ["metal"]

    rule = PropertyInteractionRule(
        id="test_social_excluded",
        actor_or_tool_tags=["corrosive"],
        target_tags=["metal"],
        intent_categories=["any"],  # "any" excludes social
        effects=[
            TransitionProposal(kind="entity_health_changed", payload={"entity_id": "$target", "delta": -6}),
        ],
        priority=10,
    )
    _add_prop_rule(world, rule)
    # Using a social verb — should not fire
    result = try_property_interaction(_action(pid, "speak", nid), world)
    assert result is None


def test_explicit_social_category_fires_on_social_verb():
    world = make_test_world()
    pid = _player_id(world)
    nid = _npc_id(world)

    world.spatial.entities[pid].tags = list(world.spatial.entities[pid].tags or []) + ["silver"]
    world.spatial.entities[nid].tags = list(world.spatial.entities[nid].tags or []) + ["undead"]

    rule = PropertyInteractionRule(
        id="test_silver_passive",
        actor_or_tool_tags=["silver"],
        target_tags=["undead"],
        intent_categories=["social", "any"],
        effects=[
            TransitionProposal(kind="entity_health_changed", payload={"entity_id": "$target", "delta": -2}),
        ],
        priority=10,
    )
    _add_prop_rule(world, rule)
    result = try_property_interaction(_action(pid, "speak", nid), world)
    assert result is not None
    assert result.valid


def test_highest_priority_rule_wins():
    world = make_test_world()
    pid = _player_id(world)
    nid = _npc_id(world)

    world.spatial.entities[pid].tags = list(world.spatial.entities[pid].tags or []) + ["corrosive"]
    world.spatial.entities[nid].tags = list(world.spatial.entities[nid].tags or []) + ["metal"]

    low = PropertyInteractionRule(
        id="low_priority",
        actor_or_tool_tags=["corrosive"],
        target_tags=["metal"],
        effects=[TransitionProposal(kind="entity_health_changed", payload={"entity_id": "$target", "delta": -1})],
        priority=1,
    )
    high = PropertyInteractionRule(
        id="high_priority",
        actor_or_tool_tags=["corrosive"],
        target_tags=["metal"],
        effects=[TransitionProposal(kind="entity_health_changed", payload={"entity_id": "$target", "delta": -99})],
        priority=50,
    )
    world.config.property_interactions = [low, high]

    result = try_property_interaction(_action(pid, "pour", nid), world)
    assert result is not None
    assert result.valid
    health_t = next(t for t in result.concrete_transitions if t.kind == TransitionKind.ENTITY_HEALTH_CHANGED)
    assert health_t.payload["delta"] == -99


def test_payload_substitution():
    world = make_test_world()
    pid = _player_id(world)
    nid = _npc_id(world)

    world.spatial.entities[pid].tags = list(world.spatial.entities[pid].tags or []) + ["corrosive"]
    world.spatial.entities[nid].tags = list(world.spatial.entities[nid].tags or []) + ["metal"]

    rule = PropertyInteractionRule(
        id="test_sub",
        actor_or_tool_tags=["corrosive"],
        target_tags=["metal"],
        effects=[
            TransitionProposal(
                kind="entity_health_changed",
                payload={"entity_id": "$target", "delta": -5, "cause": "corrosion"},
            ),
        ],
        priority=5,
    )
    _add_prop_rule(world, rule)

    result = try_property_interaction(_action(pid, "pour", nid), world)
    assert result is not None
    t = result.concrete_transitions[0]
    assert t.payload["entity_id"] == nid, "Target id should be substituted"


# ── integration: default world pack loads rules ─────────────────────────

def test_default_world_loads_property_interactions():
    pack = Path(__file__).resolve().parents[1] / "worlds" / "default"
    world = load_world_pack(pack)
    assert world.config.property_interactions, (
        "default pack should load property_interactions.yaml"
    )
    ids = {r.id for r in world.config.property_interactions}
    assert "fire_source_ignites_flammable" in ids
    assert "corrosive_dissolves_metal" in ids
    assert "silver_harms_undead" in ids


def test_tavern_inherits_default_property_interactions():
    pack = Path(__file__).resolve().parents[1] / "worlds" / "tavern"
    world = load_world_pack(pack)
    ids = {r.id for r in world.config.property_interactions}
    assert "fire_source_ignites_flammable" in ids


# ── integration: compile_action uses property rules ─────────────────────

def test_compile_action_resolves_via_property_interaction():
    """
    End-to-end: player with [corrosive] tag attacks target with [metal] tag.
    The property-intersection rule should resolve it without hitting freeform.
    """
    world = make_test_world()
    pid = _player_id(world)
    nid = _npc_id(world)

    world.spatial.entities[pid].tags = list(world.spatial.entities[pid].tags or []) + ["corrosive"]
    world.spatial.entities[nid].tags = list(world.spatial.entities[nid].tags or []) + ["metal"]

    rule = PropertyInteractionRule(
        id="acid_attack_metal",
        actor_or_tool_tags=["corrosive"],
        target_tags=["metal"],
        effects=[
            TransitionProposal(
                kind="entity_health_changed",
                payload={"entity_id": "$target", "delta": -6, "cause": "acid"},
            ),
        ],
        priority=20,
    )
    world.config.property_interactions = [rule]

    action = _action(pid, "corrode", nid, intent="pour my acid flask on the guard")
    result = compile_action(action, world)

    assert result.valid, result.rejection_detail
    kinds = {t.kind for t in result.concrete_transitions}
    assert TransitionKind.ENTITY_HEALTH_CHANGED in kinds


# ── physical-possibility rescue: deterministic last resort before LM ──────

def test_property_rescue_fires_for_orphaned_intent_with_physical_means():
    """
    Player has a tagged tool, a nearby target has matching tags, but the
    parsed action has a verb the compiler doesn't recognise and no target.
    The rescue should detect the physical possibility, synthesize a real
    action, and return a successful compile — without any LM call.
    """
    from src.sim.property_interaction_resolver import (
        try_property_possibility_rescue,
    )

    world = make_test_world()
    pid = _player_id(world)
    nid = _npc_id(world)

    # Put player and NPC adjacent so the rescue's distance gate is satisfied.
    player_pos = world.spatial.entities[pid].position
    world.spatial.entities[nid].position = player_pos.model_copy(
        update={"x": player_pos.x + 1}
    )

    _give_item(world, pid, tags=["on_fire", "fire_source"], name="torch")
    world.spatial.entities[nid].tags = (
        list(world.spatial.entities[nid].tags or []) + ["flammable"]
    )

    rule = PropertyInteractionRule(
        id="rescue_ignite",
        actor_or_tool_tags=["fire_source", "on_fire"],
        target_tags=["flammable"],
        intent_categories=["thermal"],
        effects=[
            TransitionProposal(
                kind="entity_tag_changed",
                payload={"entity_id": "$target", "add": "on_fire"},
            ),
        ],
        priority=20,
    )
    world.config.property_interactions = [rule]

    # Action with an unknown verb and no target — would normally fall
    # through to LM adjudication / Zone-3 narration.
    action = _action(pid, "do_something_creative", target=None,
                     intent="I jam my burning torch against that guy")

    rescued = try_property_possibility_rescue(action, world, action.raw_input)
    assert rescued is not None, (
        "Rescue should fire when actor has fire_source and target has flammable."
    )
    rescued_action, rescued_result = rescued
    assert rescued_result.valid
    assert rescued_action.target == nid, "Rescue should retarget at the flammable NPC"
    # The synthesized verb should be a physical category, never speech/observation.
    assert rescued_action.verb in {"ignite", "apply", "strike", "open", "use"}
    # Original raw_input preserved so narration reads in the player's words.
    assert rescued_action.raw_input == action.raw_input


def test_property_rescue_returns_none_when_no_physical_means():
    """No tagged tool / no tagged nearby target → rescue declines, action
    is genuinely orphaned and should fall through to Z3 narration."""
    from src.sim.property_interaction_resolver import (
        try_property_possibility_rescue,
    )

    world = make_test_world()
    pid = _player_id(world)

    # Strip all tags from the player and remove inventory so there is
    # nothing to lever the world with.
    world.spatial.entities[pid].tags = []
    world.spatial.entities[pid].inventory = []
    world.spatial.entities[pid].equipped_weapon = None
    world.spatial.entities[pid].equipped_armor = None
    world.spatial.entities[pid].equipped_slots = {}

    rule = PropertyInteractionRule(
        id="needs_corrosive_tool",
        actor_or_tool_tags=["corrosive"],
        target_tags=["metal"],
        intent_categories=["chemical"],
        effects=[
            TransitionProposal(
                kind="entity_health_changed",
                payload={"entity_id": "$target", "delta": -3},
            ),
        ],
        priority=10,
    )
    world.config.property_interactions = [rule]

    action = _action(pid, "do_magic", target=None,
                     intent="I will the metal to rust away")
    assert try_property_possibility_rescue(
        action, world, action.raw_input
    ) is None


def test_property_rescue_skips_social_only_rules():
    """A rule that only fires on social/observation/idle categories must
    never be used to synthesize an action — that would invent dialogue."""
    from src.sim.property_interaction_resolver import (
        try_property_possibility_rescue,
    )

    world = make_test_world()
    pid = _player_id(world)
    nid = _npc_id(world)

    world.spatial.entities[pid].tags = (
        list(world.spatial.entities[pid].tags or []) + ["silver"]
    )
    world.spatial.entities[nid].tags = (
        list(world.spatial.entities[nid].tags or []) + ["undead"]
    )

    rule = PropertyInteractionRule(
        id="silver_speech_only",
        actor_or_tool_tags=["silver"],
        target_tags=["undead"],
        intent_categories=["social"],  # social-only — should NOT be used to rescue
        effects=[
            TransitionProposal(
                kind="entity_health_changed",
                payload={"entity_id": "$target", "delta": -1},
            ),
        ],
        priority=50,
    )
    world.config.property_interactions = [rule]

    action = _action(pid, "do_something", target=None,
                     intent="I do something inscrutable")
    assert try_property_possibility_rescue(
        action, world, action.raw_input
    ) is None
